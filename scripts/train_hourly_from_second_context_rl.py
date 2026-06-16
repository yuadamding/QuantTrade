#!/usr/bin/env python3
"""Compatibility wrapper. Prefer: `qt train subhour --source 1s` (defaults live in rl_quant.presets)."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["train", "subhour", "--source", "1s", *sys.argv[1:]]))
