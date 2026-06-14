#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hourly_transformer_dataset import main as build_main  # noqa: E402


DEFAULT_ARGS = [
    "--bar-interval",
    "1m",
    "--stock-bar-dir",
    str(PROJECT_ROOT / "derived" / "minute_ohlcv" / "top_us_volume_stocks_nasdaq_1000_2026-06-14_1m_2026-05-25_2026-06-15"),
    "--etf-bar-dir",
    str(PROJECT_ROOT / "derived" / "minute_ohlcv" / "top_us_volume_etfs_500_2026-06-14_1m_2026-05-25_2026-06-15"),
    "--stock-universe",
    str(PROJECT_ROOT / "derived" / "universes" / "top_us_volume_stocks_nasdaq_1000_2026-06-14.csv"),
    "--etf-universe",
    str(PROJECT_ROOT / "derived" / "universes" / "top_us_volume_etfs_500_2026-06-14.csv"),
    "--output-dir",
    str(PROJECT_ROOT / "derived" / "rl_minute" / "top_volume_1m_recent"),
    "--dataset-file-name",
    "minute_transformer_dataset.pt",
    "--start",
    "2026-05-25T00:00:00+00:00",
    "--end-exclusive",
    "2026-06-15T00:00:00+00:00",
    "--drop-session-gaps",
    "--require-same-session-lookback",
]


if __name__ == "__main__":
    sys.argv[1:1] = DEFAULT_ARGS
    raise SystemExit(build_main())
