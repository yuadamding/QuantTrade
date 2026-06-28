"""Stage-1 training: self-supervised CONTEXT LEARNING over full sessions, then freeze + encode.

The unit is a trading DAY (a full RTH session = nB blocks). The two-tier encoder turns each day into a context
at EVERY block; the SSL pretext has TWO heads (both targets derived from the T+1 labels -- no extra inputs):
  * MARKET head: from each block's market context predict that block's next-interval equal-weight market return
    + realized vol.
  * PER-STOCK head: from each stock's per-block context predict that stock's next-block CROSS-SECTIONALLY
    DEMEANED return (r_i - equal-weight). The market head alone would train only the pooled mean, leaving the
    per-stock embeddings with no gradient rewarding which-stock-wins discrimination; this head makes the FROZEN
    context carry the relative-value signal the policy needs (without it the policy has nothing to select on).
The encoder is then FROZEN and used to ENCODE every day into detached per-block context embeddings while carrying
the raw bars forward for Stage 2. The policy trains on those detached contexts plus its own trainable raw-second
encoder (it never holds a Stage-1 encoder reference -> the split is literal).

Days stream from CPU-resident storage to ``device`` per micro-batch (full sessions are too big to hold all on
GPU) and gradients are ACCUMULATED over ``accum_steps`` micro-batches. Resumability is delegated to the caller
(start_step + optimizer + an on_checkpoint that persists model+opt+step+RNG); RNG is the global torch RNG.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from rl_quant.datasets.streaming import LazyDay
from rl_quant.training._optim import apply_lr, lr_scale


def ssl_targets(ret: torch.Tensor, ret_valid: torch.Tensor) -> torch.Tensor:
    """Per-block [equal-weight return, realized vol] over the valid non-CASH actions. ret/ret_valid [nB,A] -> [nB,2]."""
    r, v = ret[:, 1:], ret_valid[:, 1:]
    n = v.float().sum(1).clamp_min(1.0)
    ew = torch.where(v, r, torch.zeros_like(r)).sum(1) / n
    vol = torch.sqrt(torch.where(v, r * r, torch.zeros_like(r)).sum(1) / n)
    return torch.stack([ew, vol], dim=-1)


def ssl_targets_perstock(ret: torch.Tensor, ret_valid: torch.Tensor):
    """Per-stock CROSS-SECTIONALLY-DEMEANED next-block return (r_i - equal-weight) + its validity mask, over the
    full action axis (CASH column is invalid). ret/ret_valid [nB,A] -> (tgt [nB,A], mask [nB,A])."""
    v = ret_valid.clone()
    v[:, 0] = False                                                            # CASH carries no relative signal
    n = v[:, 1:].float().sum(1, keepdim=True).clamp_min(1.0)                    # valid non-CASH per block
    ew = torch.where(v[:, 1:], ret[:, 1:], torch.zeros_like(ret[:, 1:])).sum(1, keepdim=True) / n
    tgt = torch.zeros_like(ret)
    tgt[:, 1:] = ret[:, 1:] - ew                                               # demeaned -> the relative-value target
    return torch.where(v, tgt, torch.zeros_like(tgt)), v


def ssl_targets_daily(day_close: torch.Tensor, horizon: int, exec_delay: int = 1):
    """DAILY per-stock SSL target: each day's next-H-day CROSS-SECTIONALLY-DEMEANED close-to-close return -- the
    relative-value signal a DAILY cross-sectional policy actually needs (vs the intraday next-block target). Built
    over a chronological day_close sequence [N,A]; PIT-clean (uses only the close series within the split, so the
    last exec_delay+horizon days are invalid rather than peeking ahead). -> (tgt [N,A], mask [N,A]) (CASH invalid)."""
    from rl_quant.datasets.daily import horizon_close_returns
    ret, valid = horizon_close_returns(day_close, horizon, exec_delay)
    valid = valid.clone()
    valid[:, 0] = False                                                        # CASH carries no relative signal
    n = valid[:, 1:].float().sum(1, keepdim=True).clamp_min(1.0)
    ew = torch.where(valid[:, 1:], ret[:, 1:], torch.zeros_like(ret[:, 1:])).sum(1, keepdim=True) / n
    tgt = torch.zeros_like(ret)
    tgt[:, 1:] = ret[:, 1:] - ew
    return torch.where(valid, tgt, torch.zeros_like(tgt)), valid


def train_context_encoder(
    encoder, head, train_days, *, device, perstock_head=None, perstock_coef: float = 1.0,
    daily_head=None, daily_targets=None, daily_coef: float = 1.0,
    steps: int, lr: float = 3e-4, weight_decay: float = 1e-2, batch_size: int = 1, accum_steps: int = 1,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, object], None] | None = None,
    grad_reduce: Callable[[list], None] | None = None,
):
    """Fit the encoder + the market SSL head (+ optional per-stock and DAILY SSL heads) over full sessions.
    ``daily_head`` + ``daily_targets`` (a list aligned with train_days of (tgt [A], mask [A]) = each day's next-H-day
    cross-sectional return) add a DAILY relative-value pretext on the END-OF-DAY context -- the target a daily
    cross-sectional policy needs (see ssl_targets_daily). STREAMS ``batch_size`` days/micro-batch and
    GRADIENT-ACCUMULATES ``accum_steps`` micro-batches per step. Returns the optimizer."""
    heads = [head] + ([perstock_head] if perstock_head is not None else []) + \
            ([daily_head] if daily_head is not None else [])
    params = list(encoder.parameters()) + [p for h in heads for p in h.parameters()]
    if optimizer is None:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    targets = [ssl_targets(d["ret"], d["ret_valid"]) for d in train_days]      # per day [nB,2]
    valid = [d["ret_valid"][:, 1:].any(1) for d in train_days]                 # per day [nB] block has a target
    ps = [ssl_targets_perstock(d["ret"], d["ret_valid"]) for d in train_days]  # per day ([nB,A], [nB,A])
    use_daily = daily_head is not None and daily_targets is not None
    n = len(train_days)
    encoder.train()
    for h in heads:
        h.train()

    def micro():
        idx = torch.randint(0, n, (batch_size,)).tolist()
        st = lambda src: torch.stack([src[i] for i in idx]).to(device, non_blocking=True)  # noqa: E731
        bars = torch.stack([train_days[i]["bars"] for i in idx]).to(device, non_blocking=True)
        mask = torch.stack([train_days[i]["bar_mask"] for i in idx]).to(device, non_blocking=True)
        cov = torch.stack([train_days[i]["cov_blocks"] for i in idx]).to(device, non_blocking=True)
        tgt, vm = st(targets), st(valid)                                          # [b,nB,2], [b,nB]
        ps_tgt = torch.stack([ps[i][0] for i in idx]).to(device, non_blocking=True)   # [b,nB,A]
        ps_vm = torch.stack([ps[i][1] for i in idx]).to(device, non_blocking=True)    # [b,nB,A]
        d_tgt = d_vm = None
        if use_daily:
            d_tgt = torch.stack([daily_targets[i][0] for i in idx]).to(device, non_blocking=True)  # [b,A]
            d_vm = torch.stack([daily_targets[i][1] for i in idx]).to(device, non_blocking=True)   # [b,A]
        return bars, mask, cov, tgt, vm, ps_tgt, ps_vm, d_tgt, d_vm

    for step in range(start_step, steps):
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        optimizer.zero_grad()
        for _ in range(accum_steps):
            bars, mask, cov, tgt, vm, ps_tgt, ps_vm, d_tgt, d_vm = micro()
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
                per_stock, market = encoder(bars, mask, cov)          # [b,nB,A,d], [b,nB,d]
                pred = head(market)                                   # [b,nB,2]
                loss = F.smooth_l1_loss(pred[vm], tgt[vm]) if vm.any() else (pred.sum() * 0.0)
                if perstock_head is not None and ps_vm.any():         # cross-sectional relative-value pretext
                    ps_pred = perstock_head(per_stock)                # [b,nB,A]
                    loss = loss + perstock_coef * F.smooth_l1_loss(ps_pred[ps_vm], ps_tgt[ps_vm])
                if use_daily and d_vm.any():                          # DAILY next-H-day relative-value pretext
                    d_pred = daily_head(per_stock[:, -1])             # end-of-day context -> [b,A]
                    loss = loss + daily_coef * F.smooth_l1_loss(d_pred[d_vm], d_tgt[d_vm])
                loss = loss / accum_steps
            loss.backward()
        if grad_reduce is not None:                  # data-parallel: average grads across ranks before the step
            grad_reduce(params)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        if checkpoint_every and on_checkpoint and (step + 1) % checkpoint_every == 0:
            on_checkpoint(step + 1, optimizer)
    return optimizer


def freeze_encoder(encoder) -> None:
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()


@torch.no_grad()
def encode_days(encoder, days, device, batch: int = 2, amp: bool = False) -> list[dict]:
    """Run the FROZEN encoder over each day's full session -> per-block context embeddings, in chunks of ``batch``
    days (peak VRAM = batch * A sequences). The returned per-day dicts carry detached context plus the raw
    seconds the policy-side raw encoder consumes; they carry NO encoder reference."""
    encoder.eval()
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    out = []
    for i in range(0, len(days), batch):
        chunk = days[i:i + batch]
        bars = torch.stack([d["bars"] for d in chunk]).to(device, non_blocking=True)
        mask = torch.stack([d["bar_mask"] for d in chunk]).to(device, non_blocking=True)
        cov = torch.stack([d["cov_blocks"] for d in chunk]).to(device, non_blocking=True)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
            per_stock, market = encoder(bars, mask, cov)              # [b,nB,A,d], [b,nB,d]
        per_stock, market = per_stock.float().cpu(), market.float().cpu()
        for j, d in enumerate(chunk):
            if isinstance(d, LazyDay):
                # STREAMING: attach the embeddings + materialize the SMALL per-day fields in RAM (the window is
                # already resident here for the encode, so this is an LRU hit, not a reload). ONLY bars/bar_mask
                # stay lazy -> downstream (episode build, rollout) never reloads a 1GB window for a label or mask.
                out.append(d.with_overrides(
                    market=market[j].clone(), per_stock=per_stock[j].clone(),
                    avail=d["avail"].clone(), news_raw=d["news_raw"].clone(), news_mask=d["news_mask"].clone(),
                    ret=d["ret"].clone(), ret_valid=d["ret_valid"].clone(),
                    day_open=d["day_open"].clone(), day_close=d["day_close"].clone()))
            else:
                out.append({
                    "market": market[j], "per_stock": per_stock[j],
                    "bars": d["bars"], "bar_mask": d["bar_mask"],
                    "news_raw": d["news_raw"], "news_mask": d["news_mask"], "avail": d["avail"],
                    "ret": d["ret"], "ret_valid": d["ret_valid"], "n_blocks": d["ret"].shape[0],
                })
    return out
