"""Models layer: neural networks only -- they consume typed tensors and return scores/Q-values, owning no
portfolio state, reward, or data loading. Part of the protocol-first layered architecture
(see architecture_migration_plan.md). Submodules: ``minute_to_hour``, ``hourly``, ``second_context``,
``intraday``, ``strategy`` (the policy Q-networks)."""

from rl_quant.models.hourly import CausalTransformerQNetwork
from rl_quant.models.intraday import ConvQNetwork
from rl_quant.models.minute_to_hour import (
    DEFAULT_MAX_SUBHOUR_TOKENS,
    MinuteToHourCausalTransformerQNetwork,
)
from rl_quant.models.second_context import SecondContextTransformerQNetwork
from rl_quant.models.strategy import StrategyQNetwork

__all__ = [
    "DEFAULT_MAX_SUBHOUR_TOKENS",
    "CausalTransformerQNetwork",
    "ConvQNetwork",
    "MinuteToHourCausalTransformerQNetwork",
    "SecondContextTransformerQNetwork",
    "StrategyQNetwork",
]
