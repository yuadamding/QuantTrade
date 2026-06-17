"""Models layer: the minute->hour causal-transformer Q-network.

Extracted verbatim from ``minute_to_hour_transformer`` in the protocol-first reorganization
(architecture_migration_plan.md, Phase 4). Pure model code: it consumes typed tensors and returns per-action
Q-values; it owns no portfolio state, reward, or data loading. Re-exported from
``rl_quant.minute_to_hour_transformer`` for backward compatibility; behaviour is byte-identical to the
pre-extraction class."""

from __future__ import annotations

import math

import torch
from torch import nn

from rl_quant.protocol.constraints import CONSTRAINT_FEATURE_DIM

# Default number of sub-hour tokens the model attends to before mean-pool compression (a model-architecture knob).
DEFAULT_MAX_SUBHOUR_TOKENS = 512


class MinuteToHourCausalTransformerQNetwork(nn.Module):
    def __init__(
        self,
        *,
        minute_feature_dim: int,
        hour_feature_dim: int,
        action_count: int,
        hours_lookback: int,
        minutes_per_hour: int,
        d_model: int = 256,
        n_heads: int = 8,
        minute_layers: int = 2,
        hour_layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.05,
        action_embedding_dim: int = 32,
        constraint_feature_dim: int = CONSTRAINT_FEATURE_DIM,
        max_subhour_tokens: int | None = DEFAULT_MAX_SUBHOUR_TOKENS,
        action_feature_dim: int = 0,
        transition_feature_dim: int = 0,
        transition_table: torch.Tensor | None = None,
        dynamic_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if max_subhour_tokens is not None and int(max_subhour_tokens) <= 0:
            raise ValueError("max_subhour_tokens must be positive when provided.")
        self.hours_lookback = int(hours_lookback)
        self.minutes_per_hour = int(minutes_per_hour)
        self.max_subhour_tokens = None if max_subhour_tokens is None else int(max_subhour_tokens)
        self.action_count = int(action_count)
        self.action_feature_dim = int(action_feature_dim)
        self.transition_feature_dim = int(transition_feature_dim)
        self.dynamic_feature_dim = int(dynamic_feature_dim)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.minute_proj = nn.Sequential(nn.Linear(minute_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.minute_pos = nn.Parameter(torch.zeros(minutes_per_hour, d_model))
        minute_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.minute_encoder = nn.TransformerEncoder(minute_layer, num_layers=minute_layers)
        self.hour_proj = nn.Sequential(
            nn.Linear(d_model + hour_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.hour_pos = nn.Parameter(torch.zeros(hours_lookback, d_model))
        self.previous_action_embedding = nn.Embedding(action_count, action_embedding_dim)
        self.action_context = nn.Linear(action_embedding_dim, d_model)
        self.constraint_context = nn.Linear(constraint_feature_dim, d_model)
        if self.action_feature_dim > 0:
            self.action_id_embedding = nn.Embedding(action_count, d_model)
            self.action_feature_encoder = nn.Sequential(
                nn.Linear(self.action_feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.action_feature_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, feedforward_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, 1),
            )
        else:
            self.action_id_embedding = None
            self.action_feature_encoder = None
            self.action_feature_head = None
        # Position-aware transition features (opt-in). When enabled, a static [A, A, F] table of
        # (previous_action, candidate_action) features is gathered by previous_action id inside forward
        # and fed per-candidate into the Q head. The encoders are ZERO-INITIALISED so a freshly-built
        # transition-aware model scores identically to the pre-feature model until trained (and a model
        # built with transition_feature_dim=0 has no new params at all -> existing checkpoints load strict).
        if self.transition_feature_dim > 0:
            if transition_table is None:
                raise ValueError("transition_feature_dim > 0 requires a transition_table [A, A, F].")
            self.register_buffer("transition_table", transition_table.float())
            # Build the (zero-init) transition submodule WITHOUT perturbing the construction RNG of the rest
            # of the network (hour_encoder/head are built below). Like the dynamic block, its random init is
            # immediately overwritten by zeros_, so saving/restoring the RNG makes the shared backbone's init
            # identical whether transition_feature_dim is 0 or > 0 -> a freshly built transition-aware model is
            # a CLEAN perturbation of the non-transition one (same backbone init + zero-init head => identical
            # until trained), so a transition A/B isolates the feature, not a different random initialization.
            _rng_state = torch.get_rng_state()
            if self.action_feature_dim > 0:
                self.transition_encoder = nn.Sequential(
                    nn.Linear(self.transition_feature_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )
                nn.init.zeros_(self.transition_encoder[0].weight)
                nn.init.zeros_(self.transition_encoder[0].bias)
                self.transition_bias = None
            else:
                # Fallback head emits Q[B, A] from one Linear over context, with no per-candidate tokens
                # to add to; inject transition awareness as an additive per-candidate [B, A] bias instead.
                self.transition_encoder = None
                self.transition_bias = nn.Linear(self.transition_feature_dim, 1)
                nn.init.zeros_(self.transition_bias.weight)
                nn.init.zeros_(self.transition_bias.bias)
            torch.set_rng_state(_rng_state)
        else:
            self.transition_table = None
            self.transition_encoder = None
            self.transition_bias = None
        # Dynamic position-state features (opt-in, PR-D). A per-env [B, dynamic_feature_dim] vector of the
        # HELD position's realized-P&L excursion is passed into forward() and injected per-candidate
        # (broadcast across candidates, since it is position-level not candidate-level). Encoders are
        # ZERO-INITIALISED so a freshly-built dynamic-aware model scores identically until trained; and
        # dynamic_feature_dim=0 registers no params at all -> existing checkpoints load strict.
        if self.dynamic_feature_dim > 0:
            # Build the (zero-init) dynamic submodule WITHOUT perturbing the construction RNG of the rest of
            # the network (hour_encoder/head are built below). Its random init is immediately overwritten by
            # zeros_, so saving/restoring the RNG makes the shared backbone's init identical whether the flag
            # is off or on -> a freshly built dynamic-aware model is a CLEAN perturbation of the non-dynamic
            # one (same backbone init + zero-init dynamic head => identical until trained), so the D4 A/B
            # isolates the feature rather than a different random initialization.
            _rng_state = torch.get_rng_state()
            if self.action_feature_dim > 0:
                self.dynamic_encoder = nn.Sequential(
                    nn.Linear(self.dynamic_feature_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )
                nn.init.zeros_(self.dynamic_encoder[0].weight)
                nn.init.zeros_(self.dynamic_encoder[0].bias)
                self.dynamic_bias = None
            else:
                self.dynamic_encoder = None
                self.dynamic_bias = nn.Linear(self.dynamic_feature_dim, 1)
                nn.init.zeros_(self.dynamic_bias.weight)
                nn.init.zeros_(self.dynamic_bias.bias)
            torch.set_rng_state(_rng_state)
        else:
            self.dynamic_encoder = None
            self.dynamic_bias = None
        hour_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hour_encoder = nn.TransformerEncoder(hour_layer, num_layers=hour_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, action_count),
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
            self._mask_cache[key] = mask
        return mask

    def _transition_rows(self, previous_actions: torch.Tensor) -> torch.Tensor:
        # Gather the [B, A, F] per-candidate transition features for the held positions. Validate the
        # ids up front so an out-of-range previous_action raises a clear error instead of silently
        # wrapping (negative index) or tripping a cryptic CUDA assert deep in the gather.
        ids = previous_actions.long()
        if bool(((ids < 0) | (ids >= self.action_count)).any().item()):
            raise ValueError(
                f"previous_actions must be valid action ids in [0, {self.action_count}); "
                "got an out-of-range id (CASH=0 is the expected reset state)."
            )
        return self.transition_table[ids]

    def _compress_subhour_tokens(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = tokens.shape[1]
        if self.max_subhour_tokens is None or length <= self.max_subhour_tokens:
            return tokens, valid_mask
        chunk_size = int(math.ceil(length / float(self.max_subhour_tokens)))
        chunk_count = int(math.ceil(length / float(chunk_size)))
        padded_length = chunk_count * chunk_size
        if padded_length != length:
            pad_tokens = torch.zeros(
                (tokens.shape[0], padded_length - length, tokens.shape[2]),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            pad_mask = torch.zeros(
                (valid_mask.shape[0], padded_length - length),
                dtype=valid_mask.dtype,
                device=valid_mask.device,
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

    def forward(
        self,
        minute_features: torch.Tensor,
        minute_mask: torch.Tensor,
        hour_features: torch.Tensor,
        previous_actions: torch.Tensor,
        constraint_features: torch.Tensor,
        action_features: torch.Tensor | None = None,
        dynamic_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, hours, minutes, _ = minute_features.shape
        if hours > self.hours_lookback or minutes > self.minutes_per_hour:
            raise ValueError("Input context exceeds configured hours_lookback or minutes_per_hour.")
        # A model built with dynamic_feature_dim > 0 MUST be given dynamic_state -- silently omitting it would
        # let a "dynamic-aware" run (so labelled in its manifest) score like the non-dynamic model. Fail
        # closed. For a zero ablation, pass an explicit zero tensor and record that in the run manifest. Shape
        # is checked here (cheap, no device sync); finiteness is left to the env/dataset boundary to avoid a
        # per-step GPU sync on the training hot path.
        if self.dynamic_feature_dim > 0:
            if dynamic_state is None:
                raise ValueError(
                    "dynamic_state is required because the model was built with dynamic_feature_dim > 0 "
                    "(pass an explicit zero tensor for a zero-ablation and record it in the manifest)."
                )
            if tuple(dynamic_state.shape) != (batch, self.dynamic_feature_dim):
                raise ValueError(
                    f"dynamic_state shape {tuple(dynamic_state.shape)} does not match "
                    f"(batch={batch}, dynamic_feature_dim={self.dynamic_feature_dim})."
                )
        x = self.minute_proj(minute_features)
        x = x + self.minute_pos[:minutes][None, None, :, :]
        x = x.reshape(batch * hours, minutes, -1)
        flat_mask = minute_mask.reshape(batch * hours, minutes).bool()
        x, flat_mask = self._compress_subhour_tokens(x, flat_mask)
        minutes = x.shape[1]
        safe_padding_mask = ~flat_mask
        empty_rows = ~flat_mask.any(dim=1)
        if bool(empty_rows.any().item()):
            safe_padding_mask[empty_rows, 0] = False
        minute_context = self.minute_encoder(
            x,
            mask=self._causal_mask(minutes, x.device),
            src_key_padding_mask=safe_padding_mask,
        )
        valid_positions = torch.arange(minutes, device=x.device).expand(batch * hours, -1)
        last_valid = torch.where(flat_mask, valid_positions, torch.full_like(valid_positions, -1)).max(dim=1).values
        last_valid = last_valid.clamp_min(0)
        hour_context = minute_context[
            torch.arange(batch * hours, device=x.device),
            last_valid,
        ].reshape(batch, hours, -1)
        hour_context = hour_context.masked_fill(empty_rows.reshape(batch, hours, 1), 0.0)

        hour_tokens = self.hour_proj(torch.cat([hour_context, hour_features], dim=-1))
        hour_tokens = hour_tokens + self.hour_pos[:hours][None, :, :]
        action_ctx = self.action_context(self.previous_action_embedding(previous_actions.long()))
        constraint_ctx = self.constraint_context(constraint_features.float())
        hour_tokens = hour_tokens + action_ctx[:, None, :] + constraint_ctx[:, None, :]
        encoded = self.hour_encoder(hour_tokens, mask=self._causal_mask(hours, x.device))
        context = encoded[:, -1, :]
        if self.action_feature_encoder is None:
            out = self.head(context)
            if self.transition_bias is not None:
                # Per-candidate transition bias gathered by the held position id (zero at init).
                out = out + self.transition_bias(self._transition_rows(previous_actions)).squeeze(-1)
            if self.dynamic_bias is not None and dynamic_state is not None:
                # Per-env dynamic position-state bias, broadcast across candidates ([B,1]->[B,A]; zero at init).
                out = out + self.dynamic_bias(dynamic_state.float())
            return out
        if action_features is None:
            raise ValueError("Model was configured with action_feature_dim > 0 but action_features were not provided.")
        if action_features.shape[1] != self.action_count or action_features.shape[2] != self.action_feature_dim:
            raise ValueError("action_features shape does not match configured action count/feature dimension.")
        action_ids = torch.arange(self.action_count, device=action_features.device)
        action_tokens = self.action_feature_encoder(action_features.float())
        action_tokens = action_tokens + self.action_id_embedding(action_ids)[None, :, :]
        if self.transition_encoder is not None:
            # Add a per-candidate token encoding the cost/risk of moving from the held position
            # (previous_actions) to each candidate. Gathered from the static table; zero at init.
            action_tokens = action_tokens + self.transition_encoder(self._transition_rows(previous_actions))
        if self.dynamic_encoder is not None and dynamic_state is not None:
            # Add the held position's dynamic state (P&L excursion) as a per-env token, broadcast across
            # candidates ([B,d_model]->[B,1,d_model]); zero at init so an untrained dynamic model is identical.
            action_tokens = action_tokens + self.dynamic_encoder(dynamic_state.float())[:, None, :]
        q_tokens = context[:, None, :] + action_tokens
        return self.action_feature_head(q_tokens).squeeze(-1)
