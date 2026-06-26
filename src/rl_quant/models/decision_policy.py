"""Stage-2 POLICY LEARNING model: cross-sectional attention over FROZEN context + raw seconds.

This module consumes context embeddings as plain tensors -- it holds NO reference to the context encoder, so
no gradient can flow back into it. That is the other half of the enforced context/policy split: ALL policy
machinery (per-action scoring, previous-position state, constraint masking, allocation) lives here and ONLY
here; the context encoder (rl_quant.models.context_encoder) stays pure market state.

The policy has its OWN trainable raw-second encoder. Stage 1 learns a frozen context from raw bars; Stage 2 also
sees raw OHLCV through this policy-side encoder, so profit gradients can shape a raw-second representation without
backpropagating into the context encoder. The policy token is
``[market context | per-stock context | policy raw-second context | news | prev weight]``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class DecisionPolicyConfig:
    context_dim: int                 # d_model of the frozen context encoder (covariate-fused per-block context)
    bar_feature_dim: int = 5         # raw OHLCV fields consumed by the policy-side raw-second encoder
    raw_policy_dim: int | None = None
    raw_block_seconds: int = 300
    raw_policy_layers: int = 1
    raw_policy_heads: int | None = None
    raw_policy_feedforward_dim: int | None = None
    news_raw_dim: int = 1            # raw fields per news article (the qwen3 sentiment_score)
    max_news: int = 32               # articles per (stock, decision) the model aggregates at train time
    news_embed_dim: int = 32
    token_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    temperature: float = 1.0         # softmax temperature on the allocation: <1 concentrates, >1 diversifies
    gate_init_bias: float = 2.0      # initial act-gate logit -> sigmoid(2)=0.88: start TRADING (escape CASH basin)


def _sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class RawSecondPolicyEncoder(nn.Module):
    """Trainable policy-side encoder over raw 1-second OHLCV bars.

    This is not the frozen context encoder. Profit gradients update it. Inputs are raw bars plus the raw-bar
    validity mask only; missing seconds are masked as attention keys rather than converted to engineered inputs.
    """

    def __init__(
        self,
        *,
        bar_feature_dim: int,
        d_model: int,
        block_seconds: int,
        n_heads: int,
        n_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"raw_policy_dim {d_model} must be divisible by raw_policy_heads {n_heads}")
        if block_seconds <= 0:
            raise ValueError("raw policy block_seconds must be positive")
        self.block_seconds = int(block_seconds)
        self.input_norm = nn.LayerNorm(bar_feature_dim)
        self.input_proj = nn.Linear(bar_feature_dim, d_model)
        self.register_buffer("pos", _sinusoidal(self.block_seconds, d_model), persistent=False)
        self.n_layers = int(n_layers)
        if self.n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=feedforward_dim,
                dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
            )
            self.local = nn.TransformerEncoder(layer, num_layers=self.n_layers, enable_nested_tensor=False)
        else:
            self.local = nn.Identity()
        self.out_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, bars: torch.Tensor, bar_mask: torch.Tensor) -> torch.Tensor:
        """bars [B,A,S,F], bar_mask [B,A,S] -> [B,nB,A,d_model] raw policy context."""
        B, A, S, F = bars.shape
        bl = self.block_seconds
        nB = S // bl
        if nB <= 0:
            raise ValueError(f"raw policy encoder needs at least one {bl}s block; got S={S}")
        bars = bars[:, :, : nB * bl]
        bar_mask = bar_mask[:, :, : nB * bl].bool()
        x = self.input_proj(self.input_norm(bars)).reshape(B * A * nB, bl, self.d_model)
        x = x + self.pos[:bl].view(1, bl, self.d_model)
        bm = bar_mask.reshape(B * A * nB, bl)
        key_padding = ~bm
        if bool(key_padding.all(dim=1).any()):
            key_padding = key_padding.clone()
            key_padding[key_padding.all(dim=1), 0] = False
        causal = torch.triu(torch.full((bl, bl), float("-inf"), device=bars.device), diagonal=1)
        h = self.local(x, mask=causal, src_key_padding_mask=key_padding) if self.n_layers > 0 else self.local(x)
        h = self.out_norm(h)
        ar = torch.arange(bl, device=bars.device)
        idx = torch.where(bm, ar.view(1, bl), torch.full((1, bl), -1, device=bars.device)).amax(dim=1).clamp_min(0)
        rows = torch.arange(h.shape[0], device=bars.device)
        summary = h[rows, idx] * bm.any(dim=1, keepdim=True).to(h.dtype)
        return summary.reshape(B, A, nB, self.d_model).permute(0, 2, 1, 3)


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
    """Cross-sectional attention policy over frozen context, policy raw-second context, news, and previous weight.

    Permutation-equivariant and shared-weight across the action axis. The policy-side raw-second encoder and news
    aggregation are train-time model paths, not persisted engineered features.
    """

    def __init__(self, config: DecisionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        raw_dim = config.raw_policy_dim or config.context_dim
        raw_heads = config.raw_policy_heads or config.n_heads
        raw_ff = config.raw_policy_feedforward_dim or raw_dim * 2
        self.raw_encoder = RawSecondPolicyEncoder(
            bar_feature_dim=config.bar_feature_dim,
            d_model=raw_dim,
            block_seconds=config.raw_block_seconds,
            n_heads=raw_heads,
            n_layers=config.raw_policy_layers,
            feedforward_dim=raw_ff,
            dropout=config.dropout,
        )
        self.news_agg = _NewsAggregator(config.news_raw_dim, config.news_embed_dim)
        in_dim = config.context_dim * 2 + raw_dim + config.news_embed_dim + 1
        self.token_proj = nn.Linear(in_dim, config.token_dim)
        self.cash_bias = nn.Parameter(torch.zeros(config.token_dim))  # marks the CASH token in the set
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim, nhead=config.n_heads, dim_feedforward=config.feedforward_dim,
            dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1))
        self.gate_head = nn.Sequential(nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1))  # act/hold
        nn.init.constant_(self.gate_head[-1].bias, config.gate_init_bias)  # start with the gate OPEN (trade early)
        self.temperature = config.temperature
        self.raw_policy_dim = raw_dim

    def encode_raw_policy_context(self, bars: torch.Tensor, bar_mask: torch.Tensor, target_steps: int) -> torch.Tensor:
        """Encode raw bars for the policy sequence.

        Intraday batches pass bars as [B,A,S,F] and receive one raw context per block. Daily episodes pass
        [B,T,A,S,F]; each day is encoded from its full raw session and the last block is used for that day step.
        """
        if bars.ndim == 4:
            ctx = self.raw_encoder(bars, bar_mask)
            if ctx.shape[1] < target_steps:
                raise ValueError(f"raw policy context has {ctx.shape[1]} steps, need {target_steps}")
            return ctx[:, :target_steps]
        if bars.ndim == 5:
            B, T, A, S, F = bars.shape
            ctx = self.raw_encoder(bars.reshape(B * T, A, S, F), bar_mask.reshape(B * T, A, S))
            last = ctx[:, -1].reshape(B, T, A, self.raw_policy_dim)
            if T < target_steps:
                raise ValueError(f"raw policy daily context has {T} steps, need {target_steps}")
            return last[:, :target_steps]
        raise ValueError(f"bars must be [B,A,S,F] or [B,T,A,S,F]; got shape {tuple(bars.shape)}")

    def encode_raw_policy_step(self, bars: torch.Tensor, bar_mask: torch.Tensor, step: int) -> torch.Tensor:
        """Encode only the current policy step to keep Stage-2 peak memory bounded."""
        bl = self.raw_encoder.block_seconds
        if bars.ndim == 4:
            start, stop = step * bl, (step + 1) * bl
            if stop > bars.shape[2]:
                raise ValueError(f"raw policy step {step} exceeds session length {bars.shape[2]}")
            return self.raw_encoder(bars[:, :, start:stop], bar_mask[:, :, start:stop])[:, 0]
        if bars.ndim == 5:
            if step >= bars.shape[1]:
                raise ValueError(f"raw policy day step {step} exceeds episode length {bars.shape[1]}")
            start = max(0, bars.shape[3] - bl)
            return self.raw_encoder(bars[:, step, :, start:], bar_mask[:, step, :, start:])[:, -1]
        raise ValueError(f"bars must be [B,A,S,F] or [B,T,A,S,F]; got shape {tuple(bars.shape)}")

    def forward(self, market, per_stock, raw_policy_ctx, news_scores, news_mask, prev_weights, available):
        """Return target weights over {CASH, stocks} and an act-gate probability.

        market [B,d]; per_stock [B,A,d]; raw_policy_ctx [B,A,raw_d]; news_scores [B,A,M,raw];
        news_mask [B,A,M]; prev_weights/available [B,A].
        """
        B, A, d = per_stock.shape
        mkt = market.unsqueeze(1).expand(B, A, d)
        news = self.news_agg(news_scores, news_mask)                          # in-model raw-news aggregation
        tok = self.token_proj(torch.cat([mkt, per_stock, raw_policy_ctx, news, prev_weights.unsqueeze(-1)], dim=-1))
        tok = tok + self.cash_bias * (torch.arange(A, device=tok.device) == 0).float().view(1, A, 1)  # CASH token
        kpm = ~available.bool()                                  # constraint mask: unavailable actions are dropped
        kpm = kpm.clone()
        kpm[:, 0] = False                                        # CASH is always available (abstention sink)
        h = self.attn(tok, src_key_padding_mask=kpm)             # cross-sectional attention (no positional axis)
        scores = self.score(h).squeeze(-1) / self.temperature    # [B,A]; temperature shapes allocation sharpness
        scores = scores.masked_fill(kpm, float("-inf"))          # never allocate to unavailable actions
        weights = torch.softmax(scores, dim=1)
        avail = (~kpm).float().unsqueeze(-1)                     # gate reads only AVAILABLE actions (incl. CASH)
        summary = (h * avail).sum(dim=1) / avail.sum(dim=1).clamp_min(1.0)  # masked mean (no padded-row dilution)
        gate = torch.sigmoid(self.gate_head(summary).squeeze(-1))  # [B] trade (->target) vs hold (->prev)
        return weights, gate
