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
    context_dim: int                 # d_model of the frozen context encoder (covariate-fused per-block context)
    news_raw_dim: int = 1            # raw fields per news article (the qwen3 sentiment_score)
    max_news: int = 32               # articles per (stock, decision) the model aggregates at train time
    news_embed_dim: int = 32
    token_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    temperature: float = 1.0         # softmax temperature on the allocation: <1 concentrates, >1 diversifies


class _NewsAggregator(nn.Module):
    """Aggregates the RAW per-article news scores into a per-stock embedding AT TRAIN TIME (no precomputed
    count/mean). A learned per-article projection, masked-summed over articles (the sum preserves both sentiment
    and volume) then normalized -- the model decides how to use the raw news."""

    def __init__(self, raw_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(raw_dim, out_dim), nn.GELU())
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # scores [B,A,M,raw_dim], mask [B,A,M] -> [B,A,out_dim]
        e = self.proj(scores) * mask.unsqueeze(-1).to(scores.dtype)
        return self.norm(e.sum(dim=2))


class DecisionPolicyHead(nn.Module):
    """Cross-sectional attention policy: (frozen context + covariates + raw-news embedding + previous position)
    -> allocation weights over {CASH, stocks}. Permutation-equivariant and shared-weight across the action axis.
    All inputs are raw/learned -- covariates are normalized in-model and news is aggregated from raw scores
    in-model, so the policy uses NO precomputed features."""

    def __init__(self, config: DecisionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.news_agg = _NewsAggregator(config.news_raw_dim, config.news_embed_dim)
        in_dim = config.context_dim * 2 + config.news_embed_dim + 1  # market + per-stock ctx + news + prev weight
        self.token_proj = nn.Linear(in_dim, config.token_dim)
        self.cash_bias = nn.Parameter(torch.zeros(config.token_dim))  # marks the CASH token in the set
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim, nhead=config.n_heads, dim_feedforward=config.feedforward_dim,
            dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1))
        self.gate_head = nn.Sequential(nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1))  # act/hold
        self.temperature = config.temperature

    def forward(self, market, per_stock, news_scores, news_mask, prev_weights, available):
        """market [B,d]; per_stock [B,A,d] (covariate-fused context); news_scores [B,A,M,raw]; news_mask [B,A,M];
        prev_weights/available [B,A]. -> (target_weights [B,A] summing to 1, act_gate [B] in [0,1])."""
        B, A, d = per_stock.shape
        mkt = market.unsqueeze(1).expand(B, A, d)
        news = self.news_agg(news_scores, news_mask)                          # in-model raw-news aggregation
        tok = self.token_proj(torch.cat([mkt, per_stock, news, prev_weights.unsqueeze(-1)], dim=-1))
        tok = tok + self.cash_bias * (torch.arange(A, device=tok.device) == 0).float().view(1, A, 1)  # CASH token
        kpm = ~available.bool()                                  # constraint mask: unavailable actions are dropped
        kpm = kpm.clone()
        kpm[:, 0] = False                                        # CASH is always available (abstention sink)
        h = self.attn(tok, src_key_padding_mask=kpm)             # cross-sectional attention (no positional axis)
        scores = self.score(h).squeeze(-1) / self.temperature    # [B,A]; temperature shapes allocation sharpness
        scores = scores.masked_fill(kpm, float("-inf"))          # never allocate to unavailable actions
        weights = torch.softmax(scores, dim=1)
        gate = torch.sigmoid(self.gate_head(h.mean(dim=1)).squeeze(-1))  # [B] trade (->target) vs hold (->prev)
        return weights, gate
