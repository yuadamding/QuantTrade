"""Stage-1 training: self-supervised CONTEXT LEARNING over full sessions, then freeze + encode.

The unit is a trading DAY (a full RTH session = nB blocks). The two-tier encoder turns each day into a context
at EVERY block; the SSL pretext predicts, from each block's market context, that block's next-interval
equal-weight market return + realized vol (per-block targets derived from the T+1 labels -- no extra inputs).
The encoder is then FROZEN and used to ENCODE every day into per-block cached context embeddings; Stage 2 trains
the policy on those (it never holds an encoder reference -> the split is literal).

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


def train_context_encoder(
    encoder, head, train_days, *, device,
    steps: int, lr: float = 3e-4, weight_decay: float = 1e-2, batch_size: int = 1, accum_steps: int = 1,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, object], None] | None = None,
):
    """Fit the encoder + per-block SSL head over full sessions. STREAMS ``batch_size`` days/micro-batch from CPU
    to ``device`` and GRADIENT-ACCUMULATES ``accum_steps`` micro-batches per optimizer step. Returns the optimizer."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                      lr=lr, weight_decay=weight_decay)
    params = list(encoder.parameters()) + list(head.parameters())
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    targets = [ssl_targets(d["ret"], d["ret_valid"]) for d in train_days]      # per day [nB,2]
    valid = [d["ret_valid"][:, 1:].any(1) for d in train_days]                 # per day [nB] block has a target
    n = len(train_days)
    encoder.train()
    head.train()

    def micro():
        idx = torch.randint(0, n, (batch_size,)).tolist()
        bars = torch.stack([train_days[i]["bars"] for i in idx]).to(device, non_blocking=True)
        mask = torch.stack([train_days[i]["bar_mask"] for i in idx]).to(device, non_blocking=True)
        cov = torch.stack([train_days[i]["cov_blocks"] for i in idx]).to(device, non_blocking=True)
        tgt = torch.stack([targets[i] for i in idx]).to(device, non_blocking=True)      # [b,nB,2]
        vm = torch.stack([valid[i] for i in idx]).to(device, non_blocking=True)         # [b,nB]
        return bars, mask, cov, tgt, vm

    for step in range(start_step, steps):
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        optimizer.zero_grad()
        for _ in range(accum_steps):
            bars, mask, cov, tgt, vm = micro()
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
                _, market = encoder(bars, mask, cov)                  # market [b,nB,d]
                pred = head(market)                                   # [b,nB,2]
                loss = F.smooth_l1_loss(pred[vm], tgt[vm]) / accum_steps if vm.any() else (pred.sum() * 0.0)
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
    days (peak VRAM = batch * A sequences). The returned per-day dicts carry NO raw seconds and NO encoder
    reference -- they are the Stage-2 inputs (one entry per trading day, each with nB blocks)."""
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
                "news_raw": d["news_raw"], "news_mask": d["news_mask"],
                "ret": d["ret"], "ret_valid": d["ret_valid"], "n_blocks": d["ret"].shape[0],
            })
    return out
