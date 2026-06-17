"""Models layer: CausalTransformerQNetwork (hourly causal-transformer Q-network).

Extracted verbatim from rl_quant.hourly_transformer in the protocol-first reorganization (architecture_migration_plan.md, Phase 4). Pure model code: consumes typed tensors and returns scores/Q-values; owns no portfolio state, reward, or data loading. Re-exported from the source module for backward compatibility; behaviour is byte-identical."""

from __future__ import annotations

import torch
from torch import nn

from rl_quant.protocol.constraints import CONSTRAINT_FEATURE_DIM


class CausalTransformerQNetwork(nn.Module):
    """Causal Q-network over bar-based market context windows."""

    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
        action_count: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.05,
        action_embedding_dim: int = 32,
        constraint_feature_dim: int = CONSTRAINT_FEATURE_DIM,
        require_constraint_features: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.lookback = int(lookback)
        self.action_count = int(action_count)
        self.constraint_feature_dim = int(constraint_feature_dim)
        self.require_constraint_features = bool(require_constraint_features)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(lookback, d_model))
        self.previous_action_embedding = nn.Embedding(action_count, action_embedding_dim)
        self.previous_action_proj = nn.Linear(action_embedding_dim, d_model)
        self.constraint_proj = nn.Linear(self.constraint_feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, action_count),
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(
                torch.full((length, length), torch.finfo(torch.float32).min, device=device),
                diagonal=1,
            )
            self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        state_windows: torch.Tensor,
        previous_actions: torch.Tensor,
        constraint_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        length = state_windows.shape[1]
        if length > self.lookback:
            raise ValueError(f"Window length {length} exceeds configured lookback {self.lookback}.")
        if constraint_features is None:
            if self.require_constraint_features:
                raise ValueError("constraint_features are required for constrained policy inference.")
            constraint_features = torch.zeros(
                state_windows.shape[0],
                self.constraint_feature_dim,
                dtype=state_windows.dtype,
                device=state_windows.device,
            )
        if constraint_features.shape[-1] != self.constraint_feature_dim:
            raise ValueError(
                f"constraint_features must have last dimension {self.constraint_feature_dim}; "
                f"got {constraint_features.shape[-1]}."
            )
        x = self.input_proj(state_windows)
        x = x + self.position_embedding[-length:][None, :, :]
        action_context = self.previous_action_proj(self.previous_action_embedding(previous_actions.long()))
        constraint_context = self.constraint_proj(constraint_features.float())
        x = x + action_context[:, None, :] + constraint_context[:, None, :]
        x = self.encoder(x, mask=self._causal_mask(length, x.device))
        return self.head(self.out_norm(x[:, -1, :]))
