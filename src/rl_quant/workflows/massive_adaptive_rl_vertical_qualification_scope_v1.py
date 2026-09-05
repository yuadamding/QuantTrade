"""Process-local scope for the nonrecursive V5 synthetic qualification run.

The production implementation registration launches the fixed vertical test
suite.  That suite must exercise the real V5 root, whose validation boundary
normally requires a qualified implementation registration.  This module
provides the narrow bootstrap needed to break that cycle: only a reserved
synthetic experiment running inside the package-owned qualification subprocess
may replay a bootstrap registration, and that registration is nonauthorizing
outside this process-local scope.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
import os


MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_ENVIRONMENT_V1 = (
    "QUANTTRADE_ADAPTIVE_RL_VERTICAL_QUALIFICATION"
)
MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_EXPERIMENT_PREFIX_V1 = (
    "v5-vertical-qualification-"
)

_ACTIVE = ContextVar(
    "massive_adaptive_rl_vertical_qualification_v1",
    default=(
        os.environ.get(MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_ENVIRONMENT_V1) == "1"
    ),
)


def massive_adaptive_rl_vertical_qualification_scope_active_v1() -> bool:
    """Return whether this process is the package-owned qualification child."""

    return bool(
        _ACTIVE.get()
        or os.environ.get(MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_ENVIRONMENT_V1)
        == "1"
    )


def massive_adaptive_rl_vertical_qualification_experiment_v1(
    experiment_id: str,
) -> bool:
    """Return whether an experiment belongs to the nonproduction namespace."""

    return bool(
        isinstance(experiment_id, str)
        and experiment_id.startswith(
            MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_EXPERIMENT_PREFIX_V1
        )
    )


@contextmanager
def massive_adaptive_rl_vertical_qualification_scope_v1() -> Iterator[None]:
    """Activate the bootstrap only for a package-owned synthetic test scope."""

    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def require_massive_adaptive_rl_vertical_qualification_experiment_v1(
    experiment_id: str,
) -> None:
    """Reject bootstrap use outside the reserved synthetic namespace."""

    if (
        not massive_adaptive_rl_vertical_qualification_scope_active_v1()
        or not massive_adaptive_rl_vertical_qualification_experiment_v1(experiment_id)
    ):
        raise RuntimeError(
            "V5 qualification bootstrap is restricted to the package-owned "
            "synthetic qualification process"
        )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_ENVIRONMENT_V1",
    "MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_EXPERIMENT_PREFIX_V1",
    "massive_adaptive_rl_vertical_qualification_experiment_v1",
    "massive_adaptive_rl_vertical_qualification_scope_active_v1",
    "massive_adaptive_rl_vertical_qualification_scope_v1",
    "require_massive_adaptive_rl_vertical_qualification_experiment_v1",
]
