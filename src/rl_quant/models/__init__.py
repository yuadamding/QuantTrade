"""Models layer: neural networks only -- they consume typed tensors and return scores/Q-values, owning no
portfolio state, reward, or data loading. Part of the protocol-first layered architecture
(see architecture_migration_plan.md). Sole model family: ``second_to_hour`` (the causal per-second->hour
transformer Q-network; "minute" is the legacy name for the sub-hour bar slot, =3600 at 1s source)."""

from rl_quant.models.second_to_hour import (
    DEFAULT_MAX_SECOND_TOKENS,
    SecondToHourCausalTransformerQNetwork,
)

__all__ = [
    "DEFAULT_MAX_SECOND_TOKENS",
    "SecondToHourCausalTransformerQNetwork",
]
