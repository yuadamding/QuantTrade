"""Context-learning ONLY: the second->hour causal-transformer market-context ENCODER.

``SecondToHourContextEncoder`` consumes the whole-market per-second context (``second_features`` over a 4-hour
lookback + ``hour_features``) and emits an hourly market embedding. It owns NO decision-policy concern -- there is
no action universe, previous-action, constraint state, Q-head, or reward here. Decision-policy lives in
``rl_quant.models.decision_policy.DecisionPolicyQNetwork`` and consumes this encoder's (frozen) embedding.

This split makes the two stages independent (architecture_migration_plan.md): the context encoder is trained
SELF-SUPERVISED via ``SecondContextForwardHead`` (predict the next-period market move/vol) then frozen, and the
policy is trained by DQN on the precomputed embeddings. ``SecondToHourPolicyQNetwork`` is a transitional
composition (encoder o policy) that preserves the pre-split end-to-end forward signature so the existing
training/eval/env stack keeps working while Stage-2 is wired up.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from rl_quant.models.decision_policy import DecisionPolicyQNetwork
from rl_quant.protocol.constraints import CONSTRAINT_FEATURE_DIM

# Default number of sub-hour tokens the encoder attends to before mean-pool compression (architecture knob).
DEFAULT_MAX_SECOND_TOKENS = 512


class SecondToHourContextEncoder(nn.Module):
    """Policy-free market-context encoder: (second_features, second_mask, hour_features) -> hourly embedding."""

    def __init__(
        self,
        *,
        second_feature_dim: int,
        hour_feature_dim: int,
        hours_lookback: int,
        seconds_per_hour: int,
        d_model: int = 256,
        n_heads: int = 8,
        second_layers: int = 2,
        hour_layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.05,
        max_second_tokens: int | None = DEFAULT_MAX_SECOND_TOKENS,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if max_second_tokens is not None and int(max_second_tokens) <= 0:
            raise ValueError("max_second_tokens must be positive when provided.")
        self.hours_lookback = int(hours_lookback)
        self.seconds_per_hour = int(seconds_per_hour)
        self.d_model = int(d_model)
        self.max_second_tokens = None if max_second_tokens is None else int(max_second_tokens)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.second_proj = nn.Sequential(nn.Linear(second_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.second_pos = nn.Parameter(torch.zeros(seconds_per_hour, d_model))
        second_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=feedforward_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.second_encoder = nn.TransformerEncoder(second_layer, num_layers=second_layers)
        self.hour_proj = nn.Sequential(
            nn.Linear(d_model + hour_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        self.hour_pos = nn.Parameter(torch.zeros(hours_lookback, d_model))
        hour_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=feedforward_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.hour_encoder = nn.TransformerEncoder(hour_layer, num_layers=hour_layers)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
            self._mask_cache[key] = mask
        return mask

    def _compress_second_tokens(
        self, tokens: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = tokens.shape[1]
        if self.max_second_tokens is None or length <= self.max_second_tokens:
            return tokens, valid_mask
        chunk_size = int(math.ceil(length / float(self.max_second_tokens)))
        chunk_count = int(math.ceil(length / float(chunk_size)))
        padded_length = chunk_count * chunk_size
        if padded_length != length:
            pad_tokens = torch.zeros(
                (tokens.shape[0], padded_length - length, tokens.shape[2]), dtype=tokens.dtype, device=tokens.device
            )
            pad_mask = torch.zeros(
                (valid_mask.shape[0], padded_length - length), dtype=valid_mask.dtype, device=valid_mask.device
            )
            tokens = torch.cat([tokens, pad_tokens], dim=1)
            valid_mask = torch.cat([valid_mask, pad_mask], dim=1)
        grouped_tokens = tokens.reshape(tokens.shape[0], chunk_count, chunk_size, tokens.shape[2])
        grouped_mask = valid_mask.reshape(valid_mask.shape[0], chunk_count, chunk_size)
        weights = grouped_mask.to(tokens.dtype).unsqueeze(-1)
        counts = weights.sum(dim=2).clamp_min(1.0)
        compressed = (grouped_tokens * weights).sum(dim=2) / counts
        compressed_mask = grouped_mask.any(dim=2)
        return compressed, compressed_mask

    def encode_hours(
        self, second_features: torch.Tensor, second_mask: torch.Tensor, hour_features: torch.Tensor
    ) -> torch.Tensor:
        """Per-hour causal market tokens ``[B, H, d_model]`` (tier-1 over seconds -> per-hour token -> tier-2)."""
        batch, hours, seconds, _ = second_features.shape
        if hours > self.hours_lookback or seconds > self.seconds_per_hour:
            raise ValueError("Input context exceeds configured hours_lookback or seconds_per_hour.")
        x = self.second_proj(second_features)
        x = x + self.second_pos[:seconds][None, None, :, :]
        x = x.reshape(batch * hours, seconds, -1)
        flat_mask = second_mask.reshape(batch * hours, seconds).bool()
        x, flat_mask = self._compress_second_tokens(x, flat_mask)
        seconds = x.shape[1]
        safe_padding_mask = ~flat_mask
        empty_rows = ~flat_mask.any(dim=1)
        if bool(empty_rows.any().item()):
            safe_padding_mask[empty_rows, 0] = False
        second_context = self.second_encoder(
            x, mask=self._causal_mask(seconds, x.device), src_key_padding_mask=safe_padding_mask
        )
        valid_positions = torch.arange(seconds, device=x.device).expand(batch * hours, -1)
        last_valid = torch.where(flat_mask, valid_positions, torch.full_like(valid_positions, -1)).max(dim=1).values
        last_valid = last_valid.clamp_min(0)
        hour_context = second_context[torch.arange(batch * hours, device=x.device), last_valid].reshape(batch, hours, -1)
        hour_context = hour_context.masked_fill(empty_rows.reshape(batch, hours, 1), 0.0)
        hour_tokens = self.hour_proj(torch.cat([hour_context, hour_features], dim=-1))
        hour_tokens = hour_tokens + self.hour_pos[:hours][None, :, :]
        return self.hour_encoder(hour_tokens, mask=self._causal_mask(hours, x.device))

    def forward(
        self, second_features: torch.Tensor, second_mask: torch.Tensor, hour_features: torch.Tensor
    ) -> torch.Tensor:
        """Hourly market-context embedding ``[B, d_model]`` (the final causal hour token)."""
        return self.encode_hours(second_features, second_mask, hour_features)[:, -1, :]


class SecondContextForwardHead(nn.Module):
    """Self-supervised pretraining head: predict the next-period whole-market move + realized vol from the hourly
    context embedding. Trained jointly with the encoder in Stage 1; discarded (encoder frozen) for Stage 2."""

    def __init__(self, *, d_model: int, target_dim: int = 2, hidden_dim: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class SecondToHourPolicyQNetwork(nn.Module):
    """Transitional composition (context encoder o decision policy) preserving the pre-split end-to-end forward
    signature, so the existing DQN training/eval/env stack runs unchanged while Stage-2 is wired. The ENCODER is
    still policy-free; this class only wires it to a DecisionPolicyQNetwork."""

    def __init__(
        self,
        *,
        second_feature_dim: int,
        hour_feature_dim: int,
        action_count: int,
        hours_lookback: int,
        seconds_per_hour: int,
        d_model: int = 256,
        n_heads: int = 8,
        second_layers: int = 2,
        hour_layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.05,
        action_embedding_dim: int = 32,
        constraint_feature_dim: int = CONSTRAINT_FEATURE_DIM,
        max_second_tokens: int | None = DEFAULT_MAX_SECOND_TOKENS,
        action_feature_dim: int = 0,
        transition_feature_dim: int = 0,
        transition_table: torch.Tensor | None = None,
        dynamic_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = SecondToHourContextEncoder(
            second_feature_dim=second_feature_dim, hour_feature_dim=hour_feature_dim,
            hours_lookback=hours_lookback, seconds_per_hour=seconds_per_hour, d_model=d_model, n_heads=n_heads,
            second_layers=second_layers, hour_layers=hour_layers, feedforward_dim=feedforward_dim,
            dropout=dropout, max_second_tokens=max_second_tokens,
        )
        self.policy = DecisionPolicyQNetwork(
            d_model=d_model, action_count=action_count, action_embedding_dim=action_embedding_dim,
            constraint_feature_dim=constraint_feature_dim, feedforward_dim=feedforward_dim, dropout=dropout,
            action_feature_dim=action_feature_dim, transition_feature_dim=transition_feature_dim,
            transition_table=transition_table, dynamic_feature_dim=dynamic_feature_dim,
        )
        # Passthrough attributes the training/eval code reads off the model.
        self.action_count = self.policy.action_count
        self.action_feature_dim = self.policy.action_feature_dim
        self.transition_feature_dim = self.policy.transition_feature_dim
        self.dynamic_feature_dim = self.policy.dynamic_feature_dim
        self.hours_lookback = self.encoder.hours_lookback
        self.seconds_per_hour = self.encoder.seconds_per_hour
        self.max_second_tokens = self.encoder.max_second_tokens

    def forward(
        self,
        second_features: torch.Tensor,
        second_mask: torch.Tensor,
        hour_features: torch.Tensor,
        previous_actions: torch.Tensor,
        constraint_features: torch.Tensor,
        action_features: torch.Tensor | None = None,
        dynamic_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.encoder(second_features, second_mask, hour_features)
        return self.policy(
            context, previous_actions, constraint_features,
            action_features=action_features, dynamic_state=dynamic_state,
        )
