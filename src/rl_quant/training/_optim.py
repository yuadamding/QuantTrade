"""Shared training-strategy helpers: a step-driven LR schedule (no stored scheduler state, so resume stays
trivially correct -- the lr is a pure function of the step counter the trainers already checkpoint)."""
from __future__ import annotations

import math


def lr_scale(step: int, total_steps: int, warmup_steps: int, kind: str = "cosine") -> float:
    """Multiplier on the base lr at `step`: linear warmup, then cosine decay to 0 (or flat for 'constant')."""
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))  # keep a non-empty decay phase
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if kind == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr(optimizer, base_lr: float, scale: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = base_lr * scale
