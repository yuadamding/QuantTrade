"""Deterministic subprocess entry point for the fixed V5 qualification suite."""

from __future__ import annotations

from collections.abc import Sequence
import os
import sys


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["QUANTTRADE_ADAPTIVE_RL_VERTICAL_QUALIFICATION"] = "1"
    from rl_quant.workflows.massive_adaptive_rl_deterministic_runtime_v1 import (
        configure_massive_adaptive_rl_deterministic_runtime_v1,
    )

    configure_massive_adaptive_rl_deterministic_runtime_v1()
    import pytest

    return int(pytest.main(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
