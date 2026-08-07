"""Two-stage training for the learning framework.

The public convenience exports remain available, but their implementation
modules are imported only when a caller requests an exported name.  This keeps
lightweight orchestration modules below :mod:`rl_quant.training` usable from a
host-side lifecycle process that deliberately does not install PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "encode_days": ("rl_quant.training.context_pretrain", "encode_days"),
    "freeze_encoder": ("rl_quant.training.context_pretrain", "freeze_encoder"),
    "ssl_targets": ("rl_quant.training.context_pretrain", "ssl_targets"),
    "ssl_targets_daily": (
        "rl_quant.training.context_pretrain",
        "ssl_targets_daily",
    ),
    "ssl_targets_perstock": (
        "rl_quant.training.context_pretrain",
        "ssl_targets_perstock",
    ),
    "train_context_encoder": (
        "rl_quant.training.context_pretrain",
        "train_context_encoder",
    ),
    "daily_cost_paid_baselines": (
        "rl_quant.training.daily_policy",
        "daily_cost_paid_baselines",
    ),
    "daily_policy_telemetry": (
        "rl_quant.training.daily_policy",
        "daily_policy_telemetry",
    ),
    "evaluate_daily_detailed": (
        "rl_quant.training.daily_policy",
        "evaluate_daily_detailed",
    ),
    "train_daily_policy": (
        "rl_quant.training.daily_policy",
        "train_daily_policy",
    ),
    "cost_paid_baselines": (
        "rl_quant.training.decision_policy",
        "cost_paid_baselines",
    ),
    "evaluate_policy": (
        "rl_quant.training.decision_policy",
        "evaluate_policy",
    ),
    "evaluate_policy_detailed": (
        "rl_quant.training.decision_policy",
        "evaluate_policy_detailed",
    ),
    "policy_telemetry": (
        "rl_quant.training.decision_policy",
        "policy_telemetry",
    ),
    "train_decision_policy": (
        "rl_quant.training.decision_policy",
        "train_decision_policy",
    ),
    "DEFAULT_DESIGN": ("rl_quant.training.designs", "DEFAULT_DESIGN"),
    "DESIGNS": ("rl_quant.training.designs", "DESIGNS"),
    "SWEEP": ("rl_quant.training.designs", "SWEEP"),
    "TOP2000_H100_CORE_SWEEP": (
        "rl_quant.training.designs",
        "TOP2000_H100_CORE_SWEEP",
    ),
    "TOP2000_H100_WIDE_SWEEP": (
        "rl_quant.training.designs",
        "TOP2000_H100_WIDE_SWEEP",
    ),
    "TOP50_H100_CORE_SWEEP": (
        "rl_quant.training.designs",
        "TOP50_H100_CORE_SWEEP",
    ),
    "TOP50_H100_WIDE_SWEEP": (
        "rl_quant.training.designs",
        "TOP50_H100_WIDE_SWEEP",
    ),
    "Phase1Design": ("rl_quant.training.designs", "Phase1Design"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one legacy convenience export on first use."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to interactive and documentation tooling."""

    return sorted(set(globals()) | set(__all__))
