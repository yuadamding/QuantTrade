"""Models layer: neural networks only -- they consume typed tensors and return scores/Q-values, owning no
portfolio state, reward, or data loading. Part of the protocol-first layered architecture
(see architecture_migration_plan.md). Submodule: ``minute_to_hour`` (the minute->hour causal-transformer
Q-network)."""

from rl_quant.models.minute_to_hour import (
    DEFAULT_MAX_SUBHOUR_TOKENS,
    MinuteToHourCausalTransformerQNetwork,
)

__all__ = [
    "DEFAULT_MAX_SUBHOUR_TOKENS",
    "MinuteToHourCausalTransformerQNetwork",
]
