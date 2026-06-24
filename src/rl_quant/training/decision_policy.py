"""Stage-2 training: POLICY LEARNING via a differentiable, EVENT-TIMED portfolio over the FROZEN context.

Operates only on cached per-block context embeddings (one entry per trading DAY = nB blocks). The encoder is
absent here, so policy gradients cannot reach it -- the context/policy split is structural.

The policy chooses WHEN to trade: at each 5-min block it emits an act-gate g in [0,1] (trade vs hold) and a
target allocation w. The held position is a = g*w + (1-g)*prev (holding is free of turnover). Trades are T+1:
the position decided at block b is realized over the label horizon AFTER block b (ret[b] is already the T+1
forward return). A SOFT per-day budget penalizes sum_b g beyond `max_actions_per_day`. Objective per day:
maximize realized net return - turnover cost, with downside-variance and entropy terms + the budget penalty.

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
        "news_raw": s("news_raw"), "news_mask": s("news_mask"),
        "ret": torch.nan_to_num(s("ret")), "avail": s("ret_valid"),
        "label": s("ret_valid")[:, :, 1:].any(-1),               # [B,nB] block has a non-CASH T+1 label
    }


def _rollout(policy, batch, cost: float):
    """Roll the policy forward over a day's blocks, carrying the previous position (T+1, gated holding).
    -> nets [B,nB], gates [B,nB], entropies [B,nB]."""
    market, per_stock = batch["market"], batch["per_stock"]
    news_raw, news_mask, ret, avail = batch["news_raw"], batch["news_mask"], batch["ret"], batch["avail"]
    B, nB, A, _ = per_stock.shape
    prev_w = torch.zeros(B, A, device=per_stock.device)
    prev_w[:, CASH_INDEX] = 1.0
    nets, gates, ents = [], [], []
    for b in range(nB):
        w, g = policy(market[:, b], per_stock[:, b], news_raw[:, b], news_mask[:, b], prev_w.detach(), avail[:, b])
        a = g.unsqueeze(-1) * w + (1.0 - g.unsqueeze(-1)) * prev_w.detach()   # gated, T+1 position
        realized = (a * ret[:, b]).sum(-1)
        turnover = 0.5 * (a - prev_w.detach()).abs().sum(-1)
        nets.append(realized - cost * turnover)
        gates.append(g)
        ents.append(-(w * w.clamp_min(1e-9).log()).sum(-1))      # allocation entropy (masked actions contribute 0)
        prev_w = a
    return torch.stack(nets, 1), torch.stack(gates, 1), torch.stack(ents, 1)


def _loss(nets, gates, ents, label, risk_lambda, entropy_coef, max_actions, budget_lambda):
    lm = label.float()
    denom = lm.sum(1).clamp_min(1.0)
    mean_net = (nets * lm).sum(1) / denom
    downside = (torch.clamp(nets, max=0.0) ** 2 * lm).sum(1) / denom
    mean_ent = (ents * lm).sum(1) / denom
    budget_pen = torch.clamp(gates.sum(1) - max_actions, min=0.0)             # excess trades/day over the cap
    return (-mean_net.mean() + risk_lambda * downside.mean()
            - entropy_coef * mean_ent.mean() + budget_lambda * budget_pen.mean())


def train_decision_policy(
    policy, train_days, *, steps: int, lr: float = 3e-4, weight_decay: float = 3e-2,
    batch_days: int = 16, cost: float = 5e-4, risk_lambda: float = 0.1, entropy_coef: float = 0.0,
    max_actions: float = 5.0, budget_lambda: float = 0.1,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, best_val: float = -1e9, best_state: dict | None = None,
    eval_every: int = 0, val_days: list[dict] | None = None, device=None,
    on_eval: Callable[[int, float, float, dict | None, object], None] | None = None,
):
    """Train the event-timed differentiable-portfolio policy on frozen per-block embeddings.
    Returns (optimizer, best_val, best_state)."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    dev_type = (device.type if hasattr(device, "type") else "cuda")
    n = len(train_days)
    for step in range(start_step, steps):
        policy.train()
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        idx = torch.randint(0, n, (min(batch_days, n),)).tolist()
        batch = _stack(train_days, idx, device)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
            nets, gates, ents = _rollout(policy, batch, cost)
            loss = _loss(nets, gates, ents, batch["label"], risk_lambda, entropy_coef, max_actions, budget_lambda)
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
def evaluate_policy(policy, days_emb, device, cost: float, batch_days: int = 32) -> list[float]:
    """Realized per-decision net return on every label-valid block, chunked over days. Pooled list."""
    policy.eval()
    rows = []
    for i in range(0, len(days_emb), batch_days):
        batch = _stack(days_emb, list(range(i, min(i + batch_days, len(days_emb)))), device)
        nets, _, _ = _rollout(policy, batch, cost)               # [B,nB]
        rows += nets[batch["label"]].cpu().tolist()
    return rows


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
