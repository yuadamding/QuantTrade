"""Stage-2 POLICY LEARNING model: a cross-sectional attention policy over the FROZEN context.

This module consumes context embeddings as plain tensors -- it holds NO reference to the context encoder, so
no gradient can flow back into it. That is the other half of the enforced context/policy split: ALL policy
machinery (per-action scoring, previous-position state, constraint masking, allocation) lives here and ONLY
here; the context encoder (rl_quant.models.context_encoder) stays pure market state.

Design (the chosen "differentiable portfolio"): each action becomes a token
``[ market context (broadcast) | that action's per-stock context | as-of covariates | news | prev weight ]``;
a permutation-equivariant set-transformer attends ACROSS actions (relative cross-sectional valuation -- the
core of equity alpha), unavailable actions are masked, and a softmax over {CASH, stocks} yields allocation
WEIGHTS. CASH (action 0) is the abstention sink. Shared weights across actions => the same head scales from 51
to ~2000 actions (swap full attention for inducing-point/ISAB attention at the large end). The training
objective maximizes realized net return minus turnover cost (it is allocation/turnover aware via the prev
weight), so this is genuine policy learning, not a return regression.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DecisionPolicyConfig:
    context_dim: int                 # d_model of the frozen context encoder
    covariate_dim: int
    news_dim: int
    token_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0


class DecisionPolicyHead(nn.Module):
    """Cross-sectional attention policy: (frozen context + covariates + news + previous position) -> allocation
    weights over {CASH, stocks}. Permutation-equivariant and shared-weight across the action axis."""

    def __init__(self, config: DecisionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.context_dim + config.context_dim + config.covariate_dim + config.news_dim + 1  # +prev w
        self.token_proj = nn.Linear(in_dim, config.token_dim)
        self.cash_bias = nn.Parameter(torch.zeros(config.token_dim))  # marks the CASH token in the set
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim, nhead=config.n_heads, dim_feedforward=config.feedforward_dim,
            dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1))

    def forward(self, market, per_stock, covariates, news, prev_weights, available):
        """All [B,A,*] except market [B,d] and prev_weights/available [B,A]. -> weights [B,A] (rows sum to 1)."""
        B, A, d = per_stock.shape
        mkt = market.unsqueeze(1).expand(B, A, d)
        tok = self.token_proj(torch.cat([mkt, per_stock, covariates, news, prev_weights.unsqueeze(-1)], dim=-1))
        tok = tok + self.cash_bias * (torch.arange(A, device=tok.device) == 0).float().view(1, A, 1)  # CASH token
        kpm = ~available.bool()                                  # constraint mask: unavailable actions are dropped
        kpm = kpm.clone()
        kpm[:, 0] = False                                        # CASH is always available (abstention sink)
        h = self.attn(tok, src_key_padding_mask=kpm)             # cross-sectional attention (no positional axis)
        scores = self.score(h).squeeze(-1)                       # [B,A]
        scores = scores.masked_fill(kpm, float("-inf"))          # never allocate to unavailable actions
        return torch.softmax(scores, dim=1)
