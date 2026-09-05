"""Package-owned deterministic CPU runtime for V5 economic execution.

The process-start environment is established by the CLI/qualification launchers
before Python starts.  This module applies and verifies the remaining PyTorch
runtime controls before any adaptive-RL computation is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
import os

import torch


MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT = (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("PYTHONHASHSEED", "0"),
)


class MassiveAdaptiveRLDeterministicRuntimeV1Error(RuntimeError):
    """The process was not started with the registered V5 runtime."""


def massive_adaptive_rl_deterministic_environment_v1(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with the registered startup values applied."""

    result = dict(os.environ if environment is None else environment)
    result.update(MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT)
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def configure_massive_adaptive_rl_deterministic_runtime_v1() -> None:
    """Configure and attest the deterministic single-thread CPU runtime."""

    expected = dict(MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT)
    actual = {name: os.environ.get(name) for name in expected}
    if actual != expected:
        raise MassiveAdaptiveRLDeterministicRuntimeV1Error(
            "adaptive RL deterministic process-start environment differs"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != 1:
            raise MassiveAdaptiveRLDeterministicRuntimeV1Error(
                "adaptive RL inter-op threads were initialized before startup"
            ) from error
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
    ):
        raise MassiveAdaptiveRLDeterministicRuntimeV1Error(
            "adaptive RL deterministic PyTorch runtime differs"
        )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT",
    "MassiveAdaptiveRLDeterministicRuntimeV1Error",
    "configure_massive_adaptive_rl_deterministic_runtime_v1",
    "massive_adaptive_rl_deterministic_environment_v1",
]
