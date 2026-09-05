"""Process-start launcher for the adaptive-RL command line."""

from __future__ import annotations

from collections.abc import Sequence
import os
import sys


_BOOTSTRAP_MARKER = "QUANTTRADE_ADAPTIVE_RL_RUNTIME_V1"


def _startup_environment() -> dict[str, str]:
    from rl_quant.workflows.massive_adaptive_rl_deterministic_runtime_v1 import (
        massive_adaptive_rl_deterministic_environment_v1,
    )

    result = massive_adaptive_rl_deterministic_environment_v1(os.environ)
    result[_BOOTSTRAP_MARKER] = "1"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Re-exec once, configure PyTorch, then dispatch the existing CLI."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if os.environ.get(_BOOTSTRAP_MARKER) != "1":
        os.execve(
            sys.executable,
            (
                sys.executable,
                "-m",
                "rl_quant.workflows.massive_adaptive_rl_cli_v1",
                *arguments,
            ),
            _startup_environment(),
        )
        raise AssertionError("adaptive RL deterministic re-exec returned")

    from rl_quant.workflows.massive_adaptive_rl_deterministic_runtime_v1 import (
        configure_massive_adaptive_rl_deterministic_runtime_v1,
    )

    configure_massive_adaptive_rl_deterministic_runtime_v1()
    from rl_quant.workflows.massive_adaptive_rl_v2 import main as dispatch

    return dispatch(list(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
