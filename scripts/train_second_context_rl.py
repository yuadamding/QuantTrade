#!/usr/bin/env python3
"""Deprecated compatibility wrapper. Prefer: `qt train second-context`.

The current trainer is a contextual action scorer, not a full sequential RL policy.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.cli import main  # noqa: E402

if __name__ == "__main__":
    warnings.warn(
        "train_second_context_rl.py is a compatibility wrapper. Use `qt train second-context` "
        "(train_second_context_action_scorer.py); the current trainer is a contextual action "
        "scorer, not a full sequential RL policy.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(main(["train", "second-context", *sys.argv[1:]]))
