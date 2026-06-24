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


def ssl_targets(ret: torch.Tensor, ret_valid: torch.Tensor) -> torch.Tensor:
    """Next-interval [equal-weight return, realized vol] over the valid non-CASH actions. -> [D,2]."""
    r, v = ret[:, 1:], ret_valid[:, 1:]
    n = v.float().sum(1).clamp_min(1.0)
    ew = torch.where(v, r, torch.zeros_like(r)).sum(1) / n
    vol = torch.sqrt(torch.where(v, r * r, torch.zeros_like(r)).sum(1) / n)
    return torch.stack([ew, vol], dim=-1)


def train_context_encoder(
    encoder, head, chunk, chunk_mask, targets, *,
    steps: int, lr: float = 3e-4, weight_decay: float = 1e-2, batch_size: int = 8,
    start_step: int = 0, optimizer=None, checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, object], None] | None = None,
):
    """Fit the context encoder + SSL head on the pretext task. Returns the optimizer (for resume)."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                      lr=lr, weight_decay=weight_decay)
    encoder.train()
    head.train()
    n = chunk.shape[0]
    for step in range(start_step, steps):
        idx = torch.randint(0, n, (min(batch_size, n),))
        _, market = encoder(chunk[idx], chunk_mask[idx])
        loss = F.smooth_l1_loss(head(market), targets[idx])
        optimizer.zero_grad()
        loss.backward()
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
        per_stock, market = encoder(w["chunk"].to(device), w["chunk_mask"].to(device))  # [D,A,d], [D,d]
        d = per_stock.shape[0]
        out.append({
            "market": _pad(market.detach().cpu(), max_decisions),
            "per_stock": _pad(per_stock.detach().cpu(), max_decisions),
            "cov": _pad(w["cov"], max_decisions),
            "news": _pad(w["news"], max_decisions),
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
