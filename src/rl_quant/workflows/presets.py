"""Named workflow presets.

These replace the ``DEFAULT_ARGS`` lists that used to live inside thin wrapper scripts (e.g.
``train_hourly_from_second_context_rl.py``). Storing them in one registry keeps paths and defaults
in a single place and lets the ``qt`` CLI expand ``--preset NAME`` into the underlying workflow's
CLI arguments. Each preset reproduces the exact roots its original wrapper used (the second/minute
TRAIN wrappers used the shared workspace data root; the minute BUILD wrapper used repo-local paths),
so behavior is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    workflow: str  # "<group>.<workflow>" the preset targets, e.g. "train.second"
    description: str
    build_args: Callable[[], list[str]]  # () -> underlying-script CLI args (roots resolved here)


# The per-second train/build presets were removed along with the precomputed-feature / per-second stack
# (2026-06-23, "keep the LLM-generated part only"). New presets register here.
PRESETS: dict[str, Preset] = {}


def resolve_preset(name: str) -> list[str]:
    """Expand a preset name into underlying-workflow CLI arguments."""
    try:
        preset = PRESETS[name]
    except KeyError as exc:
        raise SystemExit(f"qt: unknown preset {name!r}; run `qt preset list` to see available presets.") from exc
    return list(preset.build_args())
