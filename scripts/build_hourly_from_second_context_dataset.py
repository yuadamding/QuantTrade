#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hourly_from_minute_context_dataset import (  # noqa: E402
    DATA_ROOT,
    POLYGON_SECOND_ROOT,
    POLYGON_TOP500_UNIVERSE,
    main as build_main,
)


DEFAULT_ARGS = [
    "--source-bar-interval",
    "1s",
    "--stock-bar-dir",
    str(POLYGON_SECOND_ROOT),
    "--action-bar-dir",
    str(POLYGON_SECOND_ROOT),
    "--stock-universe",
    str(POLYGON_TOP500_UNIVERSE),
    "--action-universe",
    str(POLYGON_TOP500_UNIVERSE),
    "--output-dir",
    str(DATA_ROOT / "rl_hour_from_second" / "top500_1s_recent"),
    "--dataset-file-name",
    "hour_from_second_dataset.pt",
    "--start",
    "2026-06-12T00:00:00+00:00",
    "--end-exclusive",
    "2026-06-13T00:00:00+00:00",
    "--stock-limit",
    "500",
    "--action-count",
    "16",
    "--context-bars-per-hour",
    "3600",
    "--min-active-stock-fraction",
    "0.01",
    "--min-context-valid-fraction",
    "0.005",
    "--max-action-staleness-seconds",
    "300",
    "--dense-hourly-grid",
    "--allow-missing-action-context",
]


if __name__ == "__main__":
    sys.argv[1:1] = DEFAULT_ARGS
    raise SystemExit(build_main())
