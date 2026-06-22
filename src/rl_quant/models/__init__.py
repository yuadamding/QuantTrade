"""Models layer: neural networks only -- they consume typed tensors and return scores/Q-values, owning no
portfolio state, reward, or data loading. Part of the protocol-first layered architecture
(see architecture_migration_plan.md). The second->hour model is split into two decoupled stages: a policy-free
market-context encoder (``SecondToHourContextEncoder``, trained self-supervised via ``SecondContextForwardHead``)
and a decision policy (``DecisionPolicyQNetwork``) over its embedding. ``SecondToHourPolicyQNetwork`` composes the
two into the pre-split end-to-end interface for the transitional DQN stack."""

from rl_quant.models.decision_policy import DecisionPolicyQNetwork
from rl_quant.models.second_to_hour import (
    DEFAULT_MAX_SECOND_TOKENS,
    SecondContextForwardHead,
    SecondToHourContextEncoder,
    SecondToHourPolicyQNetwork,
)

__all__ = [
    "DEFAULT_MAX_SECOND_TOKENS",
    "DecisionPolicyQNetwork",
    "SecondContextForwardHead",
    "SecondToHourContextEncoder",
    "SecondToHourPolicyQNetwork",
]
