#!/usr/bin/env python3
from __future__ import annotations

import warnings

from train_second_context_action_scorer import main


if __name__ == "__main__":
    warnings.warn(
        "train_second_context_rl.py is a compatibility wrapper. "
        "Use train_second_context_action_scorer.py; the current trainer is a contextual action scorer, "
        "not a full sequential RL policy.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(main())
