#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_hourly_from_minute_context_rl import DATA_ROOT, main as train_main  # noqa: E402


DEFAULT_ARGS = [
    "--dataset",
    str(DATA_ROOT / "rl_hour_from_second" / "top500_1s_recent" / "hour_from_second_dataset.pt"),
    "--output-dir",
    str(DATA_ROOT / "rl_hour_from_second_runs"),
    "--run-name",
    "second_to_hour_causal_transformer",
    "--d-model",
    "192",
    "--n-heads",
    "6",
    "--minute-layers",
    "2",
    "--hour-layers",
    "3",
    "--max-subhour-tokens",
    "512",
    "--episode-length",
    "32",
    "--max-switches-per-day",
    "2",
    "--max-switches-per-episode",
    "3",
    "--max-order-legs-per-episode",
    "6",
]


if __name__ == "__main__":
    sys.argv[1:1] = DEFAULT_ARGS
    raise SystemExit(train_main())
