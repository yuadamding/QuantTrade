"""Stage-1 training: self-supervised CONTEXT LEARNING, then freeze + encode.

The context encoder is trained on a pretext task (predict the next-interval equal-weight market return and
realized vol from the market context) -- no policy, no action returns-as-reward. It is then FROZEN and used to
ENCODE every window into cached context embeddings. Stage 2 trains the policy on those cached tensors, so the
policy never holds a reference to the encoder and its gradients cannot reach it -- the split is literal, not
merely a convention.

Resumability is delegated to the caller: pass ``start_step`` + an ``optimizer`` to resume, and an
``on_checkpoint(step, optimizer)`` callback that persists state. RNG is the global torch RNG (the caller
seeds it and saves/restores ``torch.get_rng_state()`` in the checkpoint), so a resumed run is bit-identical.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from rl_quant.training._optim import apply_lr, lr_scale


def ssl_targets(ret: torch.Tensor, ret_valid: torch.Tensor) -> torch.Tensor:
    """Next-interval [equal-weight return, realized vol] over the valid non-CASH actions. -> [D,2]."""
    r, v = ret[:, 1:], ret_valid[:, 1:]
    n = v.float().sum(1).clamp_min(1.0)
    ew = torch.where(v, r, torch.zeros_like(r)).sum(1) / n
    vol = torch.sqrt(torch.where(v, r * r, torch.zeros_like(r)).sum(1) / n)
    return torch.stack([ew, vol], dim=-1)


def train_context_encoder(
    encoder, head, train_windows, *, device,
    steps: int, lr: float = 3e-4, weight_decay: float = 1e-2, batch_size: int = 1, accum_steps: int = 1,
    warmup_steps: int = 0, schedule: str = "cosine", grad_clip: float = 0.0, amp: bool = False,
    start_step: int = 0, optimizer=None, checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, object], None] | None = None,
):
    """Fit the context encoder + SSL head on the pretext task. STREAMS each micro-batch of decision-rows from
    CPU-resident windows to ``device`` (the raw-second tensors are far too big to hold all on GPU), and
    GRADIENT-ACCUMULATES ``accum_steps`` micro-batches per optimizer step (so the SSL loss batch is
    ``batch_size*accum_steps`` targets while peak VRAM stays at ``batch_size`` decision-rows = batch_size*A
    raw-second sequences). warmup+cosine LR, grad clip, optional bf16 AMP. Returns the optimizer (for resume)."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                      lr=lr, weight_decay=weight_decay)
    params = list(encoder.parameters()) + list(head.parameters())
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    # flat (window, decision) row index + per-window SSL targets (small, kept on CPU)
    rows = [(wi, di) for wi, w in enumerate(train_windows) for di in range(w["decisions"])]
    wi_arr = torch.tensor([r[0] for r in rows])
    di_arr = torch.tensor([r[1] for r in rows])
    tgt_w = [ssl_targets(w["ret"], w["ret_valid"]) for w in train_windows]
    n = len(rows)
    encoder.train()
    head.train()

    def micro_batch():
        sel = torch.randint(0, n, (batch_size,))
        ws, ds = wi_arr[sel].tolist(), di_arr[sel].tolist()
        bars = torch.stack([train_windows[w]["bars"][d] for w, d in zip(ws, ds)]).to(device, non_blocking=True)
        mask = torch.stack([train_windows[w]["bar_mask"][d] for w, d in zip(ws, ds)]).to(device, non_blocking=True)
        cov = torch.stack([train_windows[w]["cov"][d] for w, d in zip(ws, ds)]).to(device, non_blocking=True)
        tgt = torch.stack([tgt_w[w][d] for w, d in zip(ws, ds)]).to(device, non_blocking=True)
        return bars, mask, cov, tgt

    for step in range(start_step, steps):
        apply_lr(optimizer, lr, lr_scale(step, steps, warmup_steps, schedule))
        optimizer.zero_grad()
        for _ in range(accum_steps):
            bars, mask, cov, tgt = micro_batch()
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
                _, market = encoder(bars, mask, cov)
                loss = F.smooth_l1_loss(head(market), tgt) / accum_steps
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
def encode_windows(encoder, windows, device, max_decisions: int) -> list[dict]:
    """Run the FROZEN encoder over each window once -> cached context embeddings, decision-padded to
    ``max_decisions`` (so the policy stage can batch windows). The returned dicts carry NO raw seconds and NO
    encoder reference -- they are the Stage-2 inputs."""
    encoder.eval()
    out = []
    for w in windows:
        per_stock, market = encoder(w["bars"].to(device), w["bar_mask"].to(device),
                                    w["cov"].to(device))  # [D,A,d], [D,d]
        d = per_stock.shape[0]
        out.append({
            "market": _pad(market.detach().cpu(), max_decisions),
            "per_stock": _pad(per_stock.detach().cpu(), max_decisions),
            "cov": _pad(w["cov"], max_decisions),
            "news_raw": _pad(w["news_raw"], max_decisions),
            "news_mask": _pad(w["news_mask"], max_decisions, dtype=torch.bool),
            "ret": _pad(w["ret"], max_decisions),
            "ret_valid": _pad(w["ret_valid"], max_decisions, dtype=torch.bool),
            "decision_mask": _decision_mask(d, max_decisions),
            "window": w.get("window"), "decisions": d,
        })
    return out


def _pad(t: torch.Tensor, max_d: int, dtype=None) -> torch.Tensor:
    d = t.shape[0]
    if d >= max_d:
        return t[:max_d]
    pad = torch.zeros((max_d - d, *t.shape[1:]), dtype=dtype or t.dtype)
    return torch.cat([t, pad], dim=0)


def _decision_mask(d: int, max_d: int) -> torch.Tensor:
    m = torch.zeros(max_d, dtype=torch.bool)
    m[:min(d, max_d)] = True
    return m
