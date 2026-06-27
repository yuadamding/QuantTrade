"""Stage-2 training for the DAILY cross-sectional policy with cross-day memory (the daily_raw path).

Operates on FROZEN end-of-day context (detached) + full-day raw bars + raw news, assembled into day-sequence
episodes (rl_quant.datasets.build_daily_raw_episodes). Per optimizer step: encode the episode ONCE (cross-day
temporal state, with the full-day raw encoder under grad), then roll the long-only portfolio forward day by day --
held position a = g*w + (1-g)*prev (turnover-free holding), T+1 close-to-close credit -- carrying the position
across the WHOLE episode (truncated BPTT). A terminal liquidation cost charges the final exit to CASH so a
buy-and-hold can't dodge its exit. Cost is applied from step 1 (no friction warm-up) -- for a daily strategy the
cost defines whether the edge exists. Evaluation is a CONTINUOUS chronological rollout over the whole split (one
episode), never resetting to CASH mid-stream.
"""
from __future__ import annotations

from typing import Callable

import torch

from rl_quant.training._optim import apply_lr, lr_scale

CASH_INDEX = 0


def _stack(episodes: list[dict], idx, device):
    g = [episodes[i] for i in idx]
    s = lambda key: torch.stack([w[key] for w in g]).to(device)  # noqa: E731
    return {k: s(k) for k in ("market", "per_stock", "bars", "bar_mask", "news_raw", "news_mask",
                              "avail", "ret", "ret_valid")} | {"label": s("ret_valid")[:, :, 1:].any(-1)}


def _daily_rollout(policy, batch, cost: float, bptt_window: int = 1, terminal_liquidate: bool = True):
    """Roll the long-only daily portfolio over the episode. -> nets [B,T], gates [B,T], ents [B,T], cash_w [B,T],
    turn [B,T], missing_w [B,T]. encode_episode() runs ONCE (cross-day memory); the per-day step carries the
    position with truncated BPTT. A terminal liquidation cost is folded into the last day's net."""
    market, per_stock = batch["market"], batch["per_stock"]
    avail, ret, ret_valid = batch["avail"], batch["ret"], batch["ret_valid"]
    state = policy.encode_episode(market, per_stock, batch["bars"], batch["bar_mask"],
                                  batch["news_raw"], batch["news_mask"], avail)        # [B,T,A,token_dim]
    B, T, A, _ = per_stock.shape
    prev_w = torch.zeros(B, A, device=per_stock.device)
    prev_w[:, CASH_INDEX] = 1.0
    nets, gates, ents, cash_w, turn, missing_w = [], [], [], [], [], []
    last_a = prev_w
    for t in range(T):
        w, g = policy.step(state[:, t], prev_w.detach(), avail[:, t])
        a = g.unsqueeze(-1) * w + (1.0 - g.unsqueeze(-1)) * prev_w
        valid = ret_valid[:, t].bool()
        missing = (a * (~valid).to(a.dtype)).sum(-1)
        realized = (a * torch.where(valid, ret[:, t], torch.zeros_like(ret[:, t]))).sum(-1)
        turnover = 0.5 * (a - prev_w).abs().sum(-1)
        nets.append(realized - cost * turnover)
        gates.append(g)
        ents.append(-(w * w.clamp_min(1e-9).log()).sum(-1))
        cash_w.append(a[:, CASH_INDEX])
        turn.append(turnover)
        missing_w.append(missing)
        last_a = a
        prev_w = a.detach() if (bptt_window <= 1 or (t + 1) % bptt_window == 0) else a
    if terminal_liquidate and nets:                                  # charge the exit of the final position to CASH
        cash_vec = torch.zeros_like(last_a)
        cash_vec[:, CASH_INDEX] = 1.0
        term_turn = 0.5 * (last_a - cash_vec).abs().sum(-1)
        nets[-1] = nets[-1] - cost * term_turn
        turn[-1] = turn[-1] + term_turn
    st = lambda xs: torch.stack(xs, 1)  # noqa: E731
    return st(nets), st(gates), st(ents), st(cash_w), st(turn), st(missing_w)


def _daily_loss(nets, gates, ents, missing_w, label, risk_lambda, entropy_coef, max_actions, budget_lambda,
                gate_entropy_coef, missing_label_penalty):
    lm = label.float()
    denom = lm.sum(1).clamp_min(1.0)
    mean_net = (nets * lm).sum(1) / denom
    downside = (torch.clamp(nets, max=0.0) ** 2 * lm).sum(1) / denom
    mean_ent = (ents * lm).sum(1) / denom
    missing_pen = (missing_w * lm).sum(1) / denom
    target_rate = max_actions / gates.shape[1]
    budget_pen = torch.clamp(gates.mean(1) - target_rate, min=0.0)
    g = gates.clamp(1e-6, 1 - 1e-6)
    gate_ent = (-(g * g.log() + (1 - g) * (1 - g).log())).mean(1)
    return (-mean_net.mean() + risk_lambda * downside.mean()
            - entropy_coef * mean_ent.mean() - gate_entropy_coef * gate_ent.mean()
            + missing_label_penalty * missing_pen.mean()
            + budget_lambda * budget_pen.mean())


def train_daily_policy(
    policy, train_eps, *, steps: int, lr: float = 3e-4, weight_decay: float = 3e-2, batch_days: int = 6,
    cost: float = 5e-4, risk_lambda: float = 0.1, entropy_coef: float = 0.0, max_actions: float = 5.0,
    budget_lambda: float = 0.1, gate_entropy_coef: float = 1e-3, missing_label_penalty: float = 1.0,
    bptt_window: int = 1, terminal_liquidate: bool = True, warmup_steps: int = 0, schedule: str = "cosine",
    grad_clip: float = 0.0, amp: bool = False, start_step: int = 0, optimizer=None, best_val: float = -1e9,
    best_state: dict | None = None, eval_every: int = 0, val_eps: list[dict] | None = None, device=None,
    min_val_label_reportable_fraction: float = 0.95,
    on_eval: Callable[[int, float, float, dict | None, object], None] | None = None,
    grad_reduce: Callable[[list], None] | None = None, is_main: bool = True,
):
    """Train the daily cross-sectional policy. Cost is full from step 1 (no friction warm-up). Returns
    (optimizer, best_val, best_state). Validation uses the CONTINUOUS rollout and selects on mean net return only
    when label-reportable coverage is adequate."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    dev_type = (device.type if hasattr(device, "type") else "cuda")
    n = len(train_eps)
    for step in range(start_step, steps):
        policy.train()
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        idx = torch.randint(0, n, (min(batch_days, n),)).tolist()
        batch = _stack(train_eps, idx, device)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
            nets, gates, ents, _, _, missing_w = _daily_rollout(policy, batch, cost, bptt_window=bptt_window,
                                                                terminal_liquidate=terminal_liquidate)
            loss = _daily_loss(nets, gates, ents, missing_w, batch["label"], risk_lambda, entropy_coef,
                               max_actions, budget_lambda, gate_entropy_coef, missing_label_penalty)
        optimizer.zero_grad()
        loss.backward()
        if grad_reduce is not None:                  # data-parallel: average grads across ranks before the step
            grad_reduce(list(policy.parameters()))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        if ((eval_every and (step + 1) % eval_every == 0) or step == steps - 1) and is_main:
            if val_eps:
                vr, vstats = evaluate_daily_detailed(policy, val_eps, device, cost)
                ok = vstats["label_reportable_fraction"] >= min_val_label_reportable_fraction
                vmean = (sum(vr) / len(vr)) if (vr and ok) else -1e9
            else:
                vmean = -1e9
            if vmean > best_val:
                best_val = vmean
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            if on_eval:
                on_eval(step + 1, vmean, best_val, best_state, optimizer)
    return optimizer, best_val, best_state


@torch.no_grad()
def evaluate_daily_detailed(policy, eps: list[dict], device, cost: float, batch_days: int = 4,
                            max_missing_label_weight: float = 1e-6) -> tuple[list[float], dict]:
    """Per-decision net return + coverage over the (continuous) evaluation episode(s). Pass a single full-split
    episode for a continuous chronological rollout (no mid-stream CASH reset)."""
    policy.eval()
    rows: list[float] = []
    total = lab = rep = 0
    gross, costs, turns, cash, gate, miss = [], [], [], [], [], []
    for i in range(0, len(eps), batch_days):
        batch = _stack(eps, list(range(i, min(i + batch_days, len(eps)))), device)
        nets, gates, _, cash_w, turn, missing_w = _daily_rollout(policy, batch, cost)
        label = batch["label"].bool()
        reportable = label & (missing_w <= max_missing_label_weight)
        total += int(label.numel())
        lab += int(label.sum())
        rep += int(reportable.sum())
        if reportable.any():
            tr = turn[reportable].cpu()
            nr = nets[reportable].cpu()
            gross.append(nr + cost * tr)
            costs.append(cost * tr)
            turns.append(tr)
            cash.append(cash_w[reportable].cpu())
            gate.append(gates[reportable].cpu())
        if label.any():
            miss.append(missing_w[label].cpu())
        rows += nets[reportable].cpu().tolist()
    mean = lambda xs: float(torch.cat(xs).mean()) if xs else 0.0  # noqa: E731
    stats = {"total_blocks": total, "label_blocks": lab, "reportable_blocks": rep,
             "reportable_fraction": rep / total if total else 0.0,
             "label_reportable_fraction": rep / lab if lab else 0.0,
             "mean_gross_return": mean(gross), "mean_turnover_cost": mean(costs),
             "mean_net_return": (sum(rows) / len(rows)) if rows else 0.0, "mean_turnover": mean(turns),
             "mean_cash_weight": mean(cash), "mean_gate": mean(gate), "mean_missing_label_weight": mean(miss)}
    return rows, stats


@torch.no_grad()
def daily_policy_telemetry(policy, eps: list[dict], device, cost: float, batch_days: int = 4) -> dict:
    policy.eval()
    gates_all, cash_all, turn_all, miss_all, trades = [], [], [], [], []
    for i in range(0, len(eps), batch_days):
        batch = _stack(eps, list(range(i, min(i + batch_days, len(eps)))), device)
        _, gates, _, cw, tv, mw = _daily_rollout(policy, batch, cost)
        gates_all.append(gates.flatten())
        cash_all.append(cw.flatten())
        turn_all.append(tv.flatten())
        miss_all.append(mw.flatten())
        trades.append(gates.sum(1))
    if not gates_all:
        return {"mean_gate": 0.0, "trades_per_episode": 0.0, "mean_cash_weight": 1.0, "mean_turnover": 0.0,
                "mean_missing_label_weight": 0.0}
    return {"mean_gate": float(torch.cat(gates_all).mean()), "trades_per_episode": float(torch.cat(trades).mean()),
            "mean_cash_weight": float(torch.cat(cash_all).mean()), "mean_turnover": float(torch.cat(turn_all).mean()),
            "mean_missing_label_weight": float(torch.cat(miss_all).mean())}


def daily_cost_paid_baselines(eps: list[dict]) -> tuple[float, float]:
    """(CASH=0, mean per-stock per-day buy-and-hold) over the SAME labelled decisions the policy is scored on."""
    bh = []
    for w in eps:
        ret, val = w["ret"], w["ret_valid"]
        for ai in range(1, ret.shape[-1]):
            col = ret[:, ai][val[:, ai]]
            if col.numel():
                bh.append(float(col.mean()))
    return 0.0, (sum(bh) / len(bh) if bh else 0.0)
