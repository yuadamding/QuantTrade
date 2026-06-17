"""Models layer: ConvQNetwork (intraday signed-position DQN wrapper).

Extracted verbatim from rl_quant.intraday_dqn in the protocol-first reorganization (architecture_migration_plan.md, Phase 4). Pure model code: consumes typed tensors and returns scores/Q-values; owns no portfolio state, reward, or data loading. Re-exported from the source module for backward compatibility; behaviour is byte-identical."""

from __future__ import annotations

import torch
from torch import nn

from rl_quant.core import TemporalQNetwork


class ConvQNetwork(nn.Module):
    def __init__(self, feature_dim: int, lookback: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = TemporalQNetwork(
            feature_dim=feature_dim,
            lookback=lookback,
            action_count=3,
            previous_action_count=3,
            hidden_size=hidden_size,
            previous_action_embedding_dim=8,
        )

    def forward(self, state_windows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.network(state_windows, positions + 1)
