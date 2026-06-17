"""Models layer: StrategyQNetwork (strategy allocation DQN wrapper).

Extracted verbatim from rl_quant.strategy_dqn in the protocol-first reorganization (architecture_migration_plan.md, Phase 4). Pure model code: consumes typed tensors and returns scores/Q-values; owns no portfolio state, reward, or data loading. Re-exported from the source module for backward compatibility; behaviour is byte-identical."""

from __future__ import annotations

import torch
from torch import nn

from rl_quant.core import TemporalQNetwork


class StrategyQNetwork(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
        action_count: int,
        hidden_size: int = 128,
        action_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.network = TemporalQNetwork(
            feature_dim=feature_dim,
            lookback=lookback,
            action_count=action_count,
            previous_action_count=action_count,
            hidden_size=hidden_size,
            previous_action_embedding_dim=action_embedding_dim,
        )

    def forward(self, state_windows: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
        return self.network(state_windows, previous_actions)
