#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_hourly_causal_transformer_rl import main as train_main  # noqa: E402


DEFAULT_ARGS = [
    "--dataset",
    str(PROJECT_ROOT / "data" / "rl_minute" / "top_volume_1m_recent" / "minute_transformer_dataset.pt"),
    "--output-dir",
    str(PROJECT_ROOT / "data" / "rl_minute_runs"),
    "--lookback",
    "128",
    "--train-end",
    "2026-06-05T23:59:59+00:00",
    "--val-end",
    "2026-06-10T23:59:59+00:00",
    "--test-start",
    "2026-06-11T00:00:00+00:00",
    "--episode-length",
    "128",
]


if __name__ == "__main__":
    sys.argv[1:1] = DEFAULT_ARGS
    raise SystemExit(train_main())
