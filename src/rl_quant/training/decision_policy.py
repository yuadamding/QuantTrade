"""Stage-2 training: POLICY LEARNING via a differentiable, EVENT-TIMED portfolio.

Operates on cached FROZEN context embeddings plus raw 1-second OHLCV carried through the Stage-2 batch. The
policy owns a separate trainable raw-second encoder, so profit gradients can learn a raw-bar policy representation
without reaching the frozen context encoder -- the context/policy split remains structural.

The policy chooses WHEN to trade: at each 5-min block it emits an act-gate g in [0,1] (rebalance intensity) and a
target allocation w. Trades are T+1. The policy observes only the previously submitted allocation that has
executed by its decision timestamp; when the new instruction executes one interval later, its gate interpolates
against the then-current drifted book. That future pre-trade book is used only for execution/turnover accounting
and never enters the current policy observation. ``ret[b]`` is the return after block b's delayed execution.

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
import torch.utils.checkpoint

from rl_quant.execution import drift_weights, force_unavailable_to_cash, one_way_turnover
from rl_quant.training._optim import apply_lr, lr_scale, make_adamw

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


def _held_drift(weights: torch.Tensor, realized_ret: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Advance simplex weights through the realized return before the next decision.

    Holding a portfolio means letting its constituents drift; carrying the pre-return target unchanged would be a
    free rebalance at every step. Missing returns grow by zero rather than entering the denominator as NaN.
    """
    return drift_weights(weights, realized_ret, valid)


def _rollout(
    policy,
    batch,
    cost: float,
    bptt_window: int = 1,
    grad_checkpoint: bool = False,
    terminal_liquidate: bool = True,
):
    """Roll the policy forward over the sequence (intraday blocks OR cross-day days), carrying the previous
    position (T+1, gated holding). -> nets [B,T], gates [B,T], entropies [B,T], cash_w [B,T], turnover [B,T].

    Between decisions the held allocation is marked through the realized return and renormalized, so gate=hold is
    a true buy-and-hold ride rather than a free rebalance. By default the post-return final book is liquidated to
    CASH and its exit turnover is charged on the final row.

    Credit assignment via TRUNCATED BPTT: the held position carries the autograd graph for `bptt_window` steps
    before detaching, so a position's MULTI-step returns back-propagate to the allocation/gate that set it (needed
    to learn long holds -- e.g. the 180-day range). `bptt_window=1` detaches every step (myopic 1-step credit, the
    original behaviour). The policy's prev-weight INPUT is always detached (it reads its position as a feature).

    `grad_checkpoint` recomputes each block's raw-second policy encode in backward instead of retaining it: the
    rollout re-encodes raw bars at EVERY block and accumulates all nB block losses before backward, so without
    this the nB raw-encoder graphs (heavy: per-second attention over ~300s x A stocks x batch_days) pile up and
    OOM a full-session run. Checkpointing frees those activations in forward and recomputes one block at a time;
    the inputs (bars/bar_mask) are already resident so no copy is added, and RNG state is preserved so dropout --
    hence the loss and gradients -- is bit-identical to the non-checkpointed path."""
    market, per_stock = batch["market"], batch["per_stock"]
    news_raw, news_mask, ret, ret_valid, avail = (
        batch["news_raw"], batch["news_mask"], batch["ret"], batch["ret_valid"], batch["avail"]
    )
    B, nB, A, _ = per_stock.shape
    # Delayed labels require two books.  ``decision_weights`` has just
    # executed at the current decision and is observable.  The previous
    # action's post-return ``execution_pretrade`` is the book one interval
    # later, when today's instruction executes; it may determine turnover and
    # the gate's execution result, but feeding it to today's policy leaks the
    # next price.  The old single ``prev_w`` mixed those timestamps.
    decision_weights = torch.zeros(B, A, device=per_stock.device)
    decision_weights[:, CASH_INDEX] = 1.0
    execution_pretrade = decision_weights
    ckpt = grad_checkpoint and policy.training
    # Evaluation has no activation-retention constraint, so encode all policy raw steps in one vectorized call.
    # DecisionPolicyHead's batched path is algebraically identical to encode_raw_policy_step for both intraday
    # [B,A,S,F] inputs and daily [B,T,A,S,F] inputs. Lightweight test policies without that API keep the fallback.
    raw_sequence = None
    if not policy.training and hasattr(policy, "encode_raw_policy_context"):
        raw_sequence = policy.encode_raw_policy_context(batch["bars"], batch["bar_mask"], nB)
    nets, gates, ents, cash_w, turn, missing_w, post_weights = [], [], [], [], [], [], []
    for b in range(nB):
        if raw_sequence is not None:
            raw_ctx = raw_sequence[:, b]
        elif ckpt:
            raw_ctx = torch.utils.checkpoint.checkpoint(
                policy.encode_raw_policy_step, batch["bars"], batch["bar_mask"], b, use_reentrant=False)
        else:
            raw_ctx = policy.encode_raw_policy_step(batch["bars"], batch["bar_mask"], b)
        decision_visible = force_unavailable_to_cash(
            decision_weights, avail[:, b], cash_index=CASH_INDEX
        )
        w, g = policy(market[:, b], per_stock[:, b], raw_ctx, news_raw[:, b], news_mask[:, b],
                      decision_visible.detach(), avail[:, b])
        before_trade = execution_pretrade
        feasible_pretrade = force_unavailable_to_cash(before_trade, avail[:, b], cash_index=CASH_INDEX)
        # Apply the submitted rebalance intensity at the delayed execution
        # timestamp.  gate=0 leaves the then-current drifted book untouched.
        a = g.unsqueeze(-1) * w + (1.0 - g.unsqueeze(-1)) * feasible_pretrade
        valid = ret_valid[:, b].bool()
        missing = (a * (~valid).to(a.dtype)).sum(-1)
        realized = (a * torch.where(valid, ret[:, b], torch.zeros_like(ret[:, b]))).sum(-1)
        turnover = one_way_turnover(a, before_trade)
        nets.append(realized - cost * turnover)
        gates.append(g)
        entropy = -(w * w.clamp_min(1e-9).log()).sum(-1)
        entropy_scale = avail[:, b].sum(-1).clamp_min(2).to(w.dtype).log()
        ents.append(entropy / entropy_scale)                    # normalized [0,1], invariant to universe width
        cash_w.append(a[:, CASH_INDEX])
        turn.append(turnover)
        missing_w.append(missing)
        # The next decision sees the post-return book. Without this mark-to-market drift, gate=hold silently
        # rebalances back to ``a`` for free and understates subsequent turnover.
        final_weights = _held_drift(a, ret[:, b], valid)
        post_weights.append(final_weights)
        detach = bptt_window <= 1 or (b + 1) % bptt_window == 0
        decision_weights = a.detach() if detach else a
        execution_pretrade = final_weights.detach() if detach else final_weights
    st = lambda xs: torch.stack(xs, 1)  # noqa: E731
    nets_t, turn_t = st(nets), st(turn)
    if terminal_liquidate and nets:
        # Intraday caches can contain an unlabeled tail after their final executable return. Liquidate the book
        # immediately after each sample's last non-CASH label, then place that charge on the same scored row.
        label = ret_valid[:, :, 1:].bool().any(-1)
        has_label = label.any(-1)
        steps = torch.arange(nB, device=label.device).view(1, nB)
        last_index = torch.where(label, steps, torch.full_like(steps, -1)).amax(-1)
        post = st(post_weights)
        selected = post[torch.arange(B, device=post.device), last_index.clamp_min(0)]
        cash = torch.zeros_like(selected)
        cash[:, CASH_INDEX] = 1.0
        liquidation_turnover = one_way_turnover(selected, cash) * has_label.to(selected.dtype)
        charge_row = (steps == last_index.unsqueeze(1)).to(selected.dtype) * liquidation_turnover.unsqueeze(1)
        nets_t = nets_t - cost * charge_row
        turn_t = turn_t + charge_row
    return nets_t, st(gates), st(ents), st(cash_w), turn_t, st(missing_w)


def _loss(nets, gates, ents, missing_w, label, risk_lambda, entropy_coef, max_actions, budget_lambda,
          gate_entropy_coef, missing_label_penalty):
    lm = label.float()
    denom = lm.sum(1).clamp_min(1.0)
    mean_net = (nets * lm).sum(1) / denom
    downside = (((torch.clamp(nets, max=0.0) ** 2 * lm).sum(1) / denom).clamp_min(0.0) + 1e-12).sqrt()
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
    max_actions: float = 5.0, budget_lambda: float = 1e-3, gate_entropy_coef: float = 1e-5,
    missing_label_penalty: float = 1e-3, friction_warmup_steps: int = 0, bptt_window: int = 1,
    grad_checkpoint: bool = False,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, best_val: float = -1e9, best_state: dict | None = None,
    eval_every: int = 0, val_days: list[dict] | None = None, device=None,
    min_val_label_reportable_fraction: float = 0.95,
    on_eval: Callable[[int, float, float, dict | None, object], None] | None = None,
    grad_reduce: Callable[[list], None] | None = None, is_main: bool = True,
    prepare_checkpoint: Callable[[], None] | None = None,
    sync_after_eval: Callable[[], None] | None = None,
):
    """Train the event-timed differentiable-portfolio policy on detached context plus raw bars. The turnover cost
    and the budget penalty are warmed up from 0 -> full over `friction_warmup_steps` (curriculum: learn the edge
    first, then constrain frequency). Validation ranks checkpoints on the fixed, policy-independent set of
    label-valid blocks and ignores checkpoints whose label-valid reportable coverage is below
    `min_val_label_reportable_fraction`. Missing-label allocations receive zero return credit and still pay
    turnover cost; coverage is an eligibility gate, never a score-row filter. Distributed callers
    may provide ``prepare_checkpoint`` for an all-rank snapshot after
    validation, then ``sync_after_eval`` to rendezvous after rank-0 checkpoint
    I/O. Returns (optimizer, best_val, best_state)."""
    if optimizer is None:
        optimizer = make_adamw(policy.parameters(), lr=lr, weight_decay=weight_decay)
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
                policy, batch, cost * friction, bptt_window=bptt_window, grad_checkpoint=grad_checkpoint
            )
            loss = _loss(nets, gates, ents, missing_w, batch["label"], risk_lambda, entropy_coef,
                         max_actions, budget_lambda * friction, gate_entropy_coef, missing_label_penalty)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_reduce is not None:                  # data-parallel: average grads across ranks before the step
            grad_reduce(list(policy.parameters()))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        should_eval = bool((eval_every and (step + 1) % eval_every == 0) or step == steps - 1)
        if should_eval:
            if is_main:
                if val_days:
                    vr, vstats = evaluate_policy_detailed(policy, val_days, device, cost)
                    ok_coverage = vstats["label_reportable_fraction"] >= min_val_label_reportable_fraction
                    vmean = (sum(vr) / len(vr)) if (vr and ok_coverage) else -1e9
                else:
                    vmean = -1e9
                if vmean > best_val:
                    best_val = vmean
                    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            if prepare_checkpoint is not None:
                prepare_checkpoint()
            if is_main:
                if on_eval:
                    on_eval(step + 1, vmean, best_val, best_state, optimizer)
            if sync_after_eval is not None:
                sync_after_eval()
    return optimizer, best_val, best_state


@torch.no_grad()
def evaluate_policy_detailed(policy, days_emb, device, cost: float, batch_days: int = 32,
                             max_missing_label_weight: float = 0.05) -> tuple[list[float], dict]:
    """Realized per-decision net return plus coverage stats.

    `reportable_fraction` is reportable blocks over all evaluated blocks; `label_reportable_fraction` uses only
    blocks with at least one non-CASH label as the denominator. Every label-valid block remains in the returned
    score and accounting summaries. Invalid held names receive zero return credit while their turnover cost is
    retained; reportable coverage is a separate eligibility diagnostic so a policy cannot improve its score by
    making an unfavorable block non-reportable.
    """
    policy.eval()
    rows = []
    decision_ids: list[str] = []
    total_blocks = 0
    label_blocks = 0
    reportable_blocks = 0
    missing_values, gross_values, cost_values, turn_values, cash_values, gate_values = [], [], [], [], [], []
    for i in range(0, len(days_emb), batch_days):
        batch = _stack(days_emb, list(range(i, min(i + batch_days, len(days_emb)))), device)
        nets, gates, _, cash_w, turn, missing_w = _rollout(policy, batch, cost)         # [B,nB]
        label = batch["label"].bool()
        reportable = label & (missing_w <= max_missing_label_weight)
        total_blocks += int(label.numel())
        label_blocks += int(label.sum().item())
        reportable_blocks += int(reportable.sum().item())
        if label.any():
            missing_values.append(missing_w[label].detach().cpu())
        if label.any():
            turn_r = turn[label].detach().cpu()
            net_r = nets[label].detach().cpu()
            gross_values.append(net_r + cost * turn_r)
            cost_values.append(cost * turn_r)
            turn_values.append(turn_r)
            cash_values.append(cash_w[label].detach().cpu())
            gate_values.append(gates[label].detach().cpu())
        rows += nets[label].cpu().tolist()
        label_cpu = label.detach().cpu()
        group = days_emb[i:min(i + batch_days, len(days_emb))]
        for local_index, day in enumerate(group):
            date = str(day.get("date", f"day_{i + local_index}"))
            decision_ids.extend(
                f"{date}:step_{step}" for step in range(reportable.shape[1])
                if bool(label_cpu[local_index, step])
            )
    def mean_or_zero(xs: list[torch.Tensor]) -> float:
        return float(torch.cat(xs).mean()) if xs else 0.0

    mean_missing = float(torch.cat(missing_values).mean()) if missing_values else 0.0
    stats = {
        "total_blocks": total_blocks,
        "label_blocks": label_blocks,
        "reportable_blocks": reportable_blocks,
        "reportable_fraction": reportable_blocks / total_blocks if total_blocks else 0.0,
        "label_reportable_fraction": reportable_blocks / label_blocks if label_blocks else 0.0,
        "mean_gross_return": mean_or_zero(gross_values),
        "mean_turnover_cost": mean_or_zero(cost_values),
        "mean_net_return": (sum(rows) / len(rows)) if rows else 0.0,
        "mean_turnover": mean_or_zero(turn_values),
        "mean_cash_weight": mean_or_zero(cash_values),
        "mean_gate": mean_or_zero(gate_values),
        "mean_missing_label_weight": mean_missing,
        "decision_ids": decision_ids,
        "return_date_basis": "fixed_labeled_blocks",
        "missing_label_accounting": "zero_return_credit_cost_charged",
        "coverage_role": "eligibility_gate_only",
    }
    return rows, stats


@torch.no_grad()
def evaluate_policy(policy, days_emb, device, cost: float, batch_days: int = 32,
                    max_missing_label_weight: float = 0.05) -> list[float]:
    """Realized per-decision net return on the fixed label-valid block set, chunked over days. Pooled list."""
    rows, _ = evaluate_policy_detailed(policy, days_emb, device, cost, batch_days, max_missing_label_weight)
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
