#!/usr/bin/env python3
"""Compatibility wrapper. Logic lives in rl_quant.workflows.commands.validate (the package owns implementation)."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.workflows.commands.validate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
