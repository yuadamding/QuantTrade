"""Decision-policy Q-network -- the SECOND of the two decoupled stages (the first is the context encoder in
``rl_quant.models.second_to_hour``).

It consumes a (frozen, precomputed) hourly market-context embedding ``context [B, d_model]`` plus the POLICY STATE
(held action + constraint state, and optional per-action features / transition / dynamic-position features) and
emits per-action Q-values ``[B, A]``. All the decision-policy machinery that used to live inside the second->hour
Q-network now lives here, so the context encoder is policy-free and can be trained self-supervised and frozen,
while this network is trained by DQN on the embeddings (architecture_migration_plan.md, context/policy split)."""

from __future__ import annotations

import torch
from torch import nn

from rl_quant.protocol.constraints import CONSTRAINT_FEATURE_DIM


class DecisionPolicyQNetwork(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        action_count: int,
        action_embedding_dim: int = 32,
        constraint_feature_dim: int = CONSTRAINT_FEATURE_DIM,
        feedforward_dim: int = 768,
        dropout: float = 0.05,
        action_feature_dim: int = 0,
        transition_feature_dim: int = 0,
        transition_table: torch.Tensor | None = None,
        dynamic_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.action_feature_dim = int(action_feature_dim)
        self.transition_feature_dim = int(transition_feature_dim)
        self.dynamic_feature_dim = int(dynamic_feature_dim)
        # Policy state -> context-space shifts (held action + constraint state), fused onto the frozen context.
        self.previous_action_embedding = nn.Embedding(action_count, action_embedding_dim)
        self.action_context = nn.Linear(action_embedding_dim, d_model)
        self.constraint_context = nn.Linear(constraint_feature_dim, d_model)
        if self.action_feature_dim > 0:
            self.action_id_embedding = nn.Embedding(action_count, d_model)
            self.action_feature_encoder = nn.Sequential(
                nn.Linear(self.action_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
            )
            self.action_feature_head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, feedforward_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(feedforward_dim, 1),
            )
        else:
            self.action_id_embedding = None
            self.action_feature_encoder = None
            self.action_feature_head = None
        # Position-aware transition features (opt-in, zero-init so a fresh transition-aware model scores
        # identically until trained; transition_feature_dim=0 registers no params -> checkpoints load strict).
        if self.transition_feature_dim > 0:
            if transition_table is None:
                raise ValueError("transition_feature_dim > 0 requires a transition_table [A, A, F].")
            self.register_buffer("transition_table", transition_table.float())
            _rng_state = torch.get_rng_state()
            if self.action_feature_dim > 0:
                self.transition_encoder = nn.Sequential(
                    nn.Linear(self.transition_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
                )
                nn.init.zeros_(self.transition_encoder[0].weight)
                nn.init.zeros_(self.transition_encoder[0].bias)
                self.transition_bias = None
            else:
                self.transition_encoder = None
                self.transition_bias = nn.Linear(self.transition_feature_dim, 1)
                nn.init.zeros_(self.transition_bias.weight)
                nn.init.zeros_(self.transition_bias.bias)
            torch.set_rng_state(_rng_state)
        else:
            self.transition_table = None
            self.transition_encoder = None
            self.transition_bias = None
        # Dynamic held-position-state features (opt-in, PR-D; same zero-init contract).
        if self.dynamic_feature_dim > 0:
            _rng_state = torch.get_rng_state()
            if self.action_feature_dim > 0:
                self.dynamic_encoder = nn.Sequential(
                    nn.Linear(self.dynamic_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
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
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, feedforward_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(feedforward_dim, action_count),
        )

    def _transition_rows(self, previous_actions: torch.Tensor) -> torch.Tensor:
        ids = previous_actions.long()
        if bool(((ids < 0) | (ids >= self.action_count)).any().item()):
            raise ValueError(
                f"previous_actions must be valid action ids in [0, {self.action_count}); "
                "got an out-of-range id (CASH=0 is the expected reset state)."
            )
        return self.transition_table[ids]

    def forward(
        self,
        context: torch.Tensor,
        previous_actions: torch.Tensor,
        constraint_features: torch.Tensor,
        action_features: torch.Tensor | None = None,
        dynamic_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``context`` is the hourly market embedding [B, d_model] from the (frozen) context encoder."""
        batch = context.shape[0]
        if self.dynamic_feature_dim > 0:
            if dynamic_state is None:
                raise ValueError(
                    "dynamic_state is required because the policy was built with dynamic_feature_dim > 0 "
                    "(pass an explicit zero tensor for a zero-ablation and record it in the manifest)."
                )
            if tuple(dynamic_state.shape) != (batch, self.dynamic_feature_dim):
                raise ValueError(
                    f"dynamic_state shape {tuple(dynamic_state.shape)} does not match "
                    f"(batch={batch}, dynamic_feature_dim={self.dynamic_feature_dim})."
                )
        # Fuse policy state onto the frozen market context (was injected into the hour tokens pre-split).
        action_ctx = self.action_context(self.previous_action_embedding(previous_actions.long()))
        constraint_ctx = self.constraint_context(constraint_features.float())
        state = context + action_ctx + constraint_ctx
        if self.action_feature_encoder is None:
            out = self.head(state)
            if self.transition_bias is not None:
                out = out + self.transition_bias(self._transition_rows(previous_actions)).squeeze(-1)
            if self.dynamic_bias is not None and dynamic_state is not None:
                out = out + self.dynamic_bias(dynamic_state.float())
            return out
        if action_features is None:
            raise ValueError("Policy was configured with action_feature_dim > 0 but action_features were not provided.")
        if action_features.shape[1] != self.action_count or action_features.shape[2] != self.action_feature_dim:
            raise ValueError("action_features shape does not match configured action count/feature dimension.")
        action_ids = torch.arange(self.action_count, device=action_features.device)
        action_tokens = self.action_feature_encoder(action_features.float())
        action_tokens = action_tokens + self.action_id_embedding(action_ids)[None, :, :]
        if self.transition_encoder is not None:
            action_tokens = action_tokens + self.transition_encoder(self._transition_rows(previous_actions))
        if self.dynamic_encoder is not None and dynamic_state is not None:
            action_tokens = action_tokens + self.dynamic_encoder(dynamic_state.float())[:, None, :]
        q_tokens = state[:, None, :] + action_tokens
        return self.action_feature_head(q_tokens).squeeze(-1)
