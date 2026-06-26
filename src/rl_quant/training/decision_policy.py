"""Stage-2 training: POLICY LEARNING via a differentiable, EVENT-TIMED portfolio.

Operates on cached FROZEN context embeddings plus raw 1-second OHLCV carried through the Stage-2 batch. The
policy owns a separate trainable raw-second encoder, so profit gradients can learn a raw-bar policy representation
without reaching the frozen context encoder -- the context/policy split remains structural.

The policy chooses WHEN to trade: at each 5-min block it emits an act-gate g in [0,1] (trade vs hold) and a
target allocation w. The held position is a = g*w + (1-g)*prev (holding is free of turnover). Trades are T+1:
the position decided at block b is realized over the label horizon AFTER block b (ret[b] is already the T+1
forward return).

Escaping the CASH basin (why the naive objective collapses): CASH has return identically 0, so doing nothing is
an exact zero-loss sink, and the act-gate can shut (g->0) before the allocation head ever learns an edge -- a
self-reinforcing collapse (da/dw = g vanishes too). Three things prevent it: (1) the gate is initialized OPEN
(gate_init_bias); (2) a FRICTION WARM-UP scales the turnover cost AND the budget penalty from 0 -> full over
`friction_warmup_steps`, so early training trades freely and the allocation head discovers the cross-sectional
signal before friction applies; (3) the budget penalty is a per-block RATE (mean gate over the day vs the target
rate max_actions/nB), commensurate with the per-decision return term -- not an unnormalized sum that dwarfs it.
A gate-entropy bonus adds mild exploration. Objective/day: maximize realized net return - turnover cost, with
downside-variance, allocation- and gate-entropy bonuses, and the soft per-day trade budget.

Resumability mirrors Stage 1 (start_step / optimizer / best_* + an on_eval checkpoint hook).
"""
from __future__ import annotations

from typing import Callable

import torch

from rl_quant.training._optim import apply_lr, lr_scale

CASH_INDEX = 0


def _stack(days_emb: list[dict], idx, device):
    g = [days_emb[i] for i in idx]
    s = lambda key: torch.stack([w[key] for w in g]).to(device)  # noqa: E731
    return {
        "market": s("market"), "per_stock": s("per_stock"),
        "bars": s("bars"), "bar_mask": s("bar_mask"),
        "news_raw": s("news_raw"), "news_mask": s("news_mask"),
        "ret": s("ret"), "ret_valid": s("ret_valid"),
        "avail": s("avail"),                                     # as-of tradeability (NOT label existence -> no leak)
        "label": s("ret_valid")[:, :, 1:].any(-1),               # [B,nB] block has a non-CASH T+1 label
    }


def _rollout(policy, batch, cost: float, bptt_window: int = 1):
    """Roll the policy forward over the sequence (intraday blocks OR cross-day days), carrying the previous
    position (T+1, gated holding). -> nets [B,T], gates [B,T], entropies [B,T], cash_w [B,T], turnover [B,T].

    Credit assignment via TRUNCATED BPTT: the held position carries the autograd graph for `bptt_window` steps
    before detaching, so a position's MULTI-step returns back-propagate to the allocation/gate that set it (needed
    to learn long holds -- e.g. the 180-day range). `bptt_window=1` detaches every step (myopic 1-step credit, the
    original behaviour). The policy's prev-weight INPUT is always detached (it reads its position as a feature)."""
    market, per_stock = batch["market"], batch["per_stock"]
    news_raw, news_mask, ret, ret_valid, avail = (
        batch["news_raw"], batch["news_mask"], batch["ret"], batch["ret_valid"], batch["avail"]
    )
    B, nB, A, _ = per_stock.shape
    prev_w = torch.zeros(B, A, device=per_stock.device)
    prev_w[:, CASH_INDEX] = 1.0
    nets, gates, ents, cash_w, turn, missing_w = [], [], [], [], [], []
    for b in range(nB):
        raw_ctx = policy.encode_raw_policy_step(batch["bars"], batch["bar_mask"], b)
        w, g = policy(market[:, b], per_stock[:, b], raw_ctx, news_raw[:, b], news_mask[:, b],
                      prev_w.detach(), avail[:, b])
        a = g.unsqueeze(-1) * w + (1.0 - g.unsqueeze(-1)) * prev_w   # carry WITH grad (truncated below) -> T+1
        valid = ret_valid[:, b].bool()
        missing = (a * (~valid).to(a.dtype)).sum(-1)
        realized = (a * torch.where(valid, ret[:, b], torch.zeros_like(ret[:, b]))).sum(-1)
        turnover = 0.5 * (a - prev_w).abs().sum(-1)
        nets.append(realized - cost * turnover)
        gates.append(g)
        ents.append(-(w * w.clamp_min(1e-9).log()).sum(-1))      # allocation entropy (masked actions contribute 0)
        cash_w.append(a[:, CASH_INDEX])
        turn.append(turnover)
        missing_w.append(missing)
        prev_w = a.detach() if (bptt_window <= 1 or (b + 1) % bptt_window == 0) else a   # truncation boundary
    st = lambda xs: torch.stack(xs, 1)  # noqa: E731
    return st(nets), st(gates), st(ents), st(cash_w), st(turn), st(missing_w)


def _loss(nets, gates, ents, missing_w, label, risk_lambda, entropy_coef, max_actions, budget_lambda,
          gate_entropy_coef, missing_label_penalty):
    lm = label.float()
    denom = lm.sum(1).clamp_min(1.0)
    mean_net = (nets * lm).sum(1) / denom
    downside = (torch.clamp(nets, max=0.0) ** 2 * lm).sum(1) / denom
    mean_ent = (ents * lm).sum(1) / denom
    missing_pen = (missing_w * lm).sum(1) / denom
    target_rate = max_actions / gates.shape[1]                               # trades/day cap as a per-block RATE
    budget_pen = torch.clamp(gates.mean(1) - target_rate, min=0.0)           # excess gate RATE over the cap, in [0,1]
    g = gates.clamp(1e-6, 1 - 1e-6)
    gate_ent = (-(g * g.log() + (1 - g) * (1 - g).log())).mean(1)            # Bernoulli gate entropy -> exploration
    return (-mean_net.mean() + risk_lambda * downside.mean()
            - entropy_coef * mean_ent.mean() - gate_entropy_coef * gate_ent.mean()
            + missing_label_penalty * missing_pen.mean()
            + budget_lambda * budget_pen.mean())


def train_decision_policy(
    policy, train_days, *, steps: int, lr: float = 3e-4, weight_decay: float = 3e-2,
    batch_days: int = 16, cost: float = 5e-4, risk_lambda: float = 0.1, entropy_coef: float = 0.0,
    max_actions: float = 5.0, budget_lambda: float = 0.1, gate_entropy_coef: float = 1e-3,
    missing_label_penalty: float = 1.0, friction_warmup_steps: int = 0, bptt_window: int = 1,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, best_val: float = -1e9, best_state: dict | None = None,
    eval_every: int = 0, val_days: list[dict] | None = None, device=None,
    on_eval: Callable[[int, float, float, dict | None, object], None] | None = None,
):
    """Train the event-timed differentiable-portfolio policy on detached context plus raw bars. The turnover cost
    and the budget penalty are warmed up from 0 -> full over `friction_warmup_steps` (curriculum: learn the edge
    first, then constrain frequency). Returns (optimizer, best_val, best_state)."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    dev_type = (device.type if hasattr(device, "type") else "cuda")
    n = len(train_days)
    for step in range(start_step, steps):
        policy.train()
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        friction = min(1.0, (step + 1) / friction_warmup_steps) if friction_warmup_steps > 0 else 1.0
        idx = torch.randint(0, n, (min(batch_days, n),)).tolist()
        batch = _stack(train_days, idx, device)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
            nets, gates, ents, _, _, missing_w = _rollout(
                policy, batch, cost * friction, bptt_window=bptt_window
            )
            loss = _loss(nets, gates, ents, missing_w, batch["label"], risk_lambda, entropy_coef,
                         max_actions, budget_lambda * friction, gate_entropy_coef, missing_label_penalty)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        if (eval_every and (step + 1) % eval_every == 0) or step == steps - 1:
            vr = evaluate_policy(policy, val_days, device, cost) if val_days else []
            vmean = (sum(vr) / len(vr)) if vr else -1e9
            if vmean > best_val:
                best_val = vmean
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            if on_eval:
                on_eval(step + 1, vmean, best_val, best_state, optimizer)
    return optimizer, best_val, best_state


@torch.no_grad()
def evaluate_policy(policy, days_emb, device, cost: float, batch_days: int = 32,
                    max_missing_label_weight: float = 1e-6) -> list[float]:
    """Realized per-decision net return on label-valid/reportable blocks, chunked over days. Pooled list."""
    policy.eval()
    rows = []
    for i in range(0, len(days_emb), batch_days):
        batch = _stack(days_emb, list(range(i, min(i + batch_days, len(days_emb)))), device)
        nets, _, _, _, _, missing_w = _rollout(policy, batch, cost)         # [B,nB]
        reportable = batch["label"] & (missing_w <= max_missing_label_weight)
        rows += nets[reportable].cpu().tolist()
    return rows


@torch.no_grad()
def policy_telemetry(policy, days_emb, device, cost: float, batch_days: int = 32) -> dict:
    """Behaviour telemetry so an all-CASH collapse is visible, not mistaken for zero alpha:
    mean act-gate, expected trades/day (sum of gates over the day), mean CASH weight, mean per-block turnover,
    and mean allocation weight on actions whose future label is missing."""
    policy.eval()
    gates_all, cash_all, turn_all, missing_all, trades = [], [], [], [], []
    for i in range(0, len(days_emb), batch_days):
        batch = _stack(days_emb, list(range(i, min(i + batch_days, len(days_emb)))), device)
        _, gates, _, cw, tv, mw = _rollout(policy, batch, cost)
        gates_all.append(gates.flatten())
        cash_all.append(cw.flatten())
        turn_all.append(tv.flatten())
        missing_all.append(mw.flatten())
        trades.append(gates.sum(1))                                  # [B] per-day trade count
    if not gates_all:
        return {"mean_gate": 0.0, "trades_per_day": 0.0, "mean_cash_weight": 1.0, "mean_turnover": 0.0,
                "mean_missing_label_weight": 0.0}
    return {"mean_gate": float(torch.cat(gates_all).mean()), "trades_per_day": float(torch.cat(trades).mean()),
            "mean_cash_weight": float(torch.cat(cash_all).mean()), "mean_turnover": float(torch.cat(turn_all).mean()),
            "mean_missing_label_weight": float(torch.cat(missing_all).mean())}


def cost_paid_baselines(days_emb) -> tuple[float, float]:
    """(CASH = 0.0, mean per-stock per-block buy-and-hold) on the same labeled blocks -- the honest bar."""
    bh = []
    for w in days_emb:
        ret, val = w["ret"], w["ret_valid"]
        for ai in range(1, ret.shape[-1]):
            col = ret[:, ai][val[:, ai]]
            if col.numel():
                bh.append(float(col.mean()))
    return 0.0, (sum(bh) / len(bh) if bh else 0.0)
