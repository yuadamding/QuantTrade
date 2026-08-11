"""Pure immutable identities shared by M03R-v8 pretraining orchestration."""

from __future__ import annotations

M03R_V8_PRETRAINING_WORKER_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-pretraining-worker-v1"
)
M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION = (0, 2, 3, 4, 5, 6, 7)

__all__ = [
    "M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION",
    "M03R_V8_PRETRAINING_WORKER_SCHEMA",
]
