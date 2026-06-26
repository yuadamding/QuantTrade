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


def train_context_encoder(
    encoder, head, train_days, *, device, perstock_head=None, perstock_coef: float = 1.0,
    steps: int, lr: float = 3e-4, weight_decay: float = 1e-2, batch_size: int = 1, accum_steps: int = 1,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, object], None] | None = None,
):
    """Fit the encoder + the market SSL head (+ optional per-stock SSL head) over full sessions. STREAMS
    ``batch_size`` days/micro-batch from CPU to ``device`` and GRADIENT-ACCUMULATES ``accum_steps`` micro-batches
    per optimizer step. Returns the optimizer."""
    heads = [head] + ([perstock_head] if perstock_head is not None else [])
    params = list(encoder.parameters()) + [p for h in heads for p in h.parameters()]
    if optimizer is None:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    targets = [ssl_targets(d["ret"], d["ret_valid"]) for d in train_days]      # per day [nB,2]
    valid = [d["ret_valid"][:, 1:].any(1) for d in train_days]                 # per day [nB] block has a target
    ps = [ssl_targets_perstock(d["ret"], d["ret_valid"]) for d in train_days]  # per day ([nB,A], [nB,A])
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
        return bars, mask, cov, tgt, vm, ps_tgt, ps_vm

    for step in range(start_step, steps):
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        optimizer.zero_grad()
        for _ in range(accum_steps):
            bars, mask, cov, tgt, vm, ps_tgt, ps_vm = micro()
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
                per_stock, market = encoder(bars, mask, cov)          # [b,nB,A,d], [b,nB,d]
                pred = head(market)                                   # [b,nB,2]
                loss = F.smooth_l1_loss(pred[vm], tgt[vm]) if vm.any() else (pred.sum() * 0.0)
                if perstock_head is not None and ps_vm.any():         # cross-sectional relative-value pretext
                    ps_pred = perstock_head(per_stock)                # [b,nB,A]
                    loss = loss + perstock_coef * F.smooth_l1_loss(ps_pred[ps_vm], ps_tgt[ps_vm])
                loss = loss / accum_steps
            loss.backward()
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
            out.append({
                "market": market[j], "per_stock": per_stock[j],
                "bars": d["bars"], "bar_mask": d["bar_mask"],
                "news_raw": d["news_raw"], "news_mask": d["news_mask"], "avail": d["avail"],
                "ret": d["ret"], "ret_valid": d["ret_valid"], "n_blocks": d["ret"].shape[0],
            })
    return out
