"""Stage-2 training: POLICY LEARNING via a differentiable portfolio over the FROZEN context.

Operates ONLY on cached context embeddings (the output of rl_quant.training.context_pretrain.encode_windows) +
covariates/news/labels. The encoder is absent here, so policy gradients cannot reach it -- the context/policy
split is enforced structurally.

Objective (the chosen differentiable portfolio): for each decision the set-transformer policy emits allocation
WEIGHTS over {CASH, stocks}; the window is rolled forward carrying the previous weights, and we maximize the
realized net return minus turnover cost, with a downside-variance penalty. The previous weights enter the
policy as state (so it is turnover-aware) but are DETACHED when fed back, keeping each step's graph one-deep
(stable, memory-light) -- the turnover penalty still shapes the current decision.

Resumability mirrors Stage 1: pass ``start_step``/``optimizer``/``best_*`` to resume; ``on_eval`` is the
checkpoint hook (called at each validation point).
"""
from __future__ import annotations

from typing import Callable

import torch

from rl_quant.training._optim import apply_lr, lr_scale

CASH_INDEX = 0


def _stack(emb_windows: list[dict], idx, device):
    g = [emb_windows[i] for i in idx]
    s = lambda key: torch.stack([w[key] for w in g]).to(device)  # noqa: E731
    return {
        "market": s("market"), "per_stock": s("per_stock"), "cov": s("cov"),
        "news_raw": s("news_raw"), "news_mask": s("news_mask"),
        "ret": torch.nan_to_num(s("ret")), "avail": s("ret_valid"), "dmask": s("decision_mask"),
    }


def _rollout(policy, batch, cost: float):
    """Roll the policy forward through each window's decisions, carrying previous weights.
    -> nets [B,maxD], entropies [B,maxD] (Shannon entropy of each decision's allocation)."""
    market, per_stock, cov = batch["market"], batch["per_stock"], batch["cov"]
    news_raw, news_mask, ret, avail = batch["news_raw"], batch["news_mask"], batch["ret"], batch["avail"]
    B, maxD, A, _ = per_stock.shape
    prev_w = torch.zeros(B, A, device=per_stock.device)
    prev_w[:, CASH_INDEX] = 1.0
    nets, ents = [], []
    for d in range(maxD):
        w = policy(market[:, d], per_stock[:, d], cov[:, d], news_raw[:, d], news_mask[:, d],
                   prev_w.detach(), avail[:, d])  # [B,A]
        realized = (w * ret[:, d]).sum(-1)
        turnover = 0.5 * (w - prev_w.detach()).abs().sum(-1)
        nets.append(realized - cost * turnover)
        ents.append(-(w * w.clamp_min(1e-9).log()).sum(-1))   # 0*log0=0 for masked actions (exact-zero weight)
        prev_w = w
    return torch.stack(nets, dim=1), torch.stack(ents, dim=1)


def _portfolio_loss(nets, entropies, dmask, risk_lambda: float, entropy_coef: float):
    dm = dmask.float()
    denom = dm.sum(1).clamp_min(1.0)
    mean_net = (nets * dm).sum(1) / denom
    downside = (torch.clamp(nets, max=0.0) ** 2 * dm).sum(1) / denom
    mean_ent = (entropies * dm).sum(1) / denom
    return -mean_net.mean() + risk_lambda * downside.mean() - entropy_coef * mean_ent.mean()


def train_decision_policy(
    policy, train_emb, *, steps: int, lr: float = 3e-4, weight_decay: float = 3e-2,
    batch_windows: int = 8, cost: float = 5e-4, risk_lambda: float = 0.1, entropy_coef: float = 0.0,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, best_val: float = -1e9, best_state: dict | None = None,
    eval_every: int = 0, val_emb: list[dict] | None = None, device=None,
    on_eval: Callable[[int, float, float, dict | None, object], None] | None = None,
):
    """Train the differentiable-portfolio policy on frozen embeddings (warmup+cosine LR, grad clip, optional
    bf16 AMP, entropy bonus). Returns (optimizer, best_val, best_state)."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    dev_type = (device.type if hasattr(device, "type") else "cuda")
    n = len(train_emb)
    for step in range(start_step, steps):
        policy.train()
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        idx = torch.randint(0, n, (min(batch_windows, n),)).tolist()
        batch = _stack(train_emb, idx, device)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
            nets, ents = _rollout(policy, batch, cost)
            loss = _portfolio_loss(nets, ents, batch["dmask"], risk_lambda, entropy_coef)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        if (eval_every and (step + 1) % eval_every == 0) or step == steps - 1:
            vr = evaluate_policy(policy, val_emb, device, cost) if val_emb else -1e9
            vmean = (sum(vr) / len(vr)) if vr else -1e9
            if vmean > best_val:
                best_val = vmean
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            if on_eval:
                on_eval(step + 1, vmean, best_val, best_state, optimizer)
    return optimizer, best_val, best_state


@torch.no_grad()
def evaluate_policy(policy, emb_windows, device, cost: float) -> list[float]:
    """Realized per-decision net return of the policy on every real decision (padding excluded). Pooled list."""
    policy.eval()
    if not emb_windows:
        return []
    batch = _stack(emb_windows, list(range(len(emb_windows))), device)
    nets, _ = _rollout(policy, batch, cost)           # [B,maxD]
    return nets[batch["dmask"]].cpu().tolist()


def cost_paid_baselines(windows) -> tuple[float, float]:
    """(CASH = 0.0, mean per-stock buy-and-hold) on the same decisions -- the honest bar. Accepts raw built
    windows or padded embedding windows (padding excluded via decision_mask when present)."""
    bh = []
    for w in windows:
        dm = w.get("decision_mask")
        ret = w["ret"][dm] if dm is not None else w["ret"]
        val = w["ret_valid"][dm] if dm is not None else w["ret_valid"]
        for ai in range(1, ret.shape[1]):
            col = ret[:, ai][val[:, ai]]
            if col.numel():
                bh.append(float(col.mean()))
    return 0.0, (sum(bh) / len(bh) if bh else 0.0)
