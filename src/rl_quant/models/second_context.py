"""Models layer: SecondContextTransformerQNetwork (second-context action-conditioned Q-network).

Extracted verbatim from rl_quant.second_context_transformer in the protocol-first reorganization (architecture_migration_plan.md, Phase 4). Pure model code: consumes typed tensors and returns scores/Q-values; owns no portfolio state, reward, or data loading. Re-exported from the source module for backward compatibility; behaviour is byte-identical."""

from __future__ import annotations

import torch
from torch import nn


class SecondContextTransformerQNetwork(nn.Module):
    """Action-conditioned Q-network for compact second-derived decision datasets."""

    def __init__(
        self,
        *,
        market_feature_dim: int,
        action_feature_dim: int,
        portfolio_state_dim: int,
        constraint_state_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        temporal_layers: int = 2,
        feedforward_dim: int = 384,
        dropout: float = 0.10,
        max_lookback_blocks: int = 64,
        action_count: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if max_lookback_blocks <= 0:
            raise ValueError("max_lookback_blocks must be positive.")
        if action_count is not None and action_count <= 0:
            raise ValueError("action_count must be positive when supplied.")
        self.max_lookback_blocks = int(max_lookback_blocks)
        self.action_count = None if action_count is None else int(action_count)
        self.market_proj = nn.Sequential(nn.Linear(market_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.position = nn.Parameter(torch.zeros(max_lookback_blocks, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.market_encoder = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.portfolio_encoder = nn.Linear(portfolio_state_dim, d_model)
        self.constraint_encoder = nn.Linear(constraint_state_dim, d_model)
        self.state_norm = nn.LayerNorm(d_model)
        self.action_encoder = nn.Sequential(nn.Linear(action_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.action_id_embedding = None if action_count is None else nn.Embedding(int(action_count), d_model)
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim // 2),
            nn.GELU(),
            nn.Linear(feedforward_dim // 2, 1),
        )

    def forward(
        self,
        market_context: torch.Tensor,
        market_context_mask: torch.Tensor,
        action_features: torch.Tensor,
        portfolio_state: torch.Tensor,
        constraint_state: torch.Tensor,
        action_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, blocks, _ = market_context.shape
        if blocks > self.max_lookback_blocks:
            raise ValueError("market_context exceeds max_lookback_blocks.")
        x = self.market_proj(market_context)
        x = x + self.position[:blocks][None, :, :]
        valid = market_context_mask.bool()
        padding_mask = ~valid
        empty_rows = ~valid.any(dim=1)
        if bool(empty_rows.any().item()):
            padding_mask = padding_mask.clone()
            padding_mask[empty_rows, 0] = False
        encoded = self.market_encoder(x, src_key_padding_mask=padding_mask)
        valid_positions = torch.arange(blocks, device=encoded.device).expand(batch, -1)
        last_valid = torch.where(valid, valid_positions, torch.full_like(valid_positions, -1)).max(dim=1).values
        last_valid = last_valid.clamp_min(0)
        market_token = encoded[torch.arange(batch, device=encoded.device), last_valid]
        market_token = market_token.masked_fill(empty_rows.unsqueeze(1), 0.0)
        state_token = self.state_norm(
            market_token
            + self.portfolio_encoder(portfolio_state.float())
            + self.constraint_encoder(constraint_state.float())
        )
        action_token = self.action_encoder(action_features.float())
        if self.action_id_embedding is not None:
            action_count = action_token.shape[1]
            if self.action_count is not None and action_count > self.action_count:
                raise ValueError("action_features include more actions than action_count.")
            if action_ids is None:
                action_ids = torch.arange(action_count, device=action_token.device).expand(batch, -1)
            action_token = action_token + self.action_id_embedding(action_ids.long())
        state_expanded = state_token[:, None, :].expand(-1, action_token.shape[1], -1)
        pair = torch.cat([state_expanded, action_token, state_expanded * action_token], dim=-1)
        return self.scorer(pair).squeeze(-1)
