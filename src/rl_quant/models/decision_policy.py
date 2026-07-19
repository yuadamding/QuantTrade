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

from rl_quant.protocol.constraints import project_capped_risky_simplex


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
    max_stock_weight: float = 1.0    # hard cap on each non-CASH target; CASH remains the residual sink
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
    validity mask only; missing seconds are omitted from attention rather than converted to engineered inputs.
    Price fields are expressed relative to the first valid close in each decision block, while remaining fields
    receive a signed-log magnitude transform. Both transforms are token/block local: they do not mix OHLCV fields,
    batch peers, later blocks, or future decisions.
    """

    pos: torch.Tensor

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
        self.bar_feature_dim = int(bar_feature_dim)
        self.input_proj = nn.Linear(bar_feature_dim, d_model)
        self.register_buffer("pos", _sinusoidal(self.block_seconds, d_model), persistent=False)
        self.n_layers = int(n_layers)
        if self.n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=feedforward_dim,
                dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
            )
            self.local: nn.Module = nn.TransformerEncoder(
                layer, num_layers=self.n_layers, enable_nested_tensor=False
            )
        else:
            self.local = nn.Identity()
        self.out_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    @staticmethod
    def _normalize_ohlcv(bars: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Causally scale block-local raw fields without cross-field LayerNorm.

        ``bars`` is ``[N, L, F]`` after block partitioning. Standardizing the feature axis of each raw token makes
        a large volume observation determine the apparent scale of all four price fields. Instead, OHLC (when
        present) is represented in return units relative to the block's first valid close and volume/extra raw
        fields use a signed ``log1p``. The anchor is the earliest observation, so the transform is point-in-time
        and invariant to later or peer samples. Invalid/non-finite values become the neutral zero input.
        """
        finite_valid = valid.unsqueeze(-1) & torch.isfinite(bars)
        clean = torch.where(finite_valid, bars, torch.zeros_like(bars))
        if bars.shape[-1] < 4:
            transformed = torch.sign(clean) * torch.log1p(clean.abs())
            return transformed * valid.unsqueeze(-1).to(bars.dtype)

        first_index = valid.to(torch.int64).argmax(dim=1)
        rows = torch.arange(bars.shape[0], device=bars.device)
        anchor = clean[rows, first_index, 3:4]
        denominator = anchor.abs().clamp_min(1e-4)
        prices = (clean[..., :4] - anchor.unsqueeze(1)) / denominator.unsqueeze(1)
        other = clean[..., 4:]
        other = torch.sign(other) * torch.log1p(other.abs())
        transformed = torch.cat((prices, other), dim=-1)
        return transformed * valid.unsqueeze(-1).to(bars.dtype)

    def forward(self, bars: torch.Tensor, bar_mask: torch.Tensor) -> torch.Tensor:
        """bars [B,A,S,F], bar_mask [B,A,S] -> [B,nB,A,d_model] raw policy context."""
        B, A, S, F = bars.shape
        bl = self.block_seconds
        nB = S // bl
        if nB <= 0:
            raise ValueError(f"raw policy encoder needs at least one {bl}s block; got S={S}")
        bars = bars[:, :, : nB * bl].reshape(B * A * nB, bl, F)
        bm = bar_mask[:, :, : nB * bl].bool().reshape(B * A * nB, bl)
        x = self.input_proj(self._normalize_ohlcv(bars, bm))
        # Autocast Linear emits BF16, while the persistent positional table is FP32.  Cast the tiny table slice
        # instead of promoting the full raw-token grid and all following attention residuals to FP32.
        x = x + self.pos[:bl].to(dtype=x.dtype).view(1, bl, self.d_model)

        # We consume only the last valid token from each block. Pack rows by valid length so attention never sees
        # padding and needs only one shared ``[L,L]`` causal mask, instead of an expanded per-row causal+padding
        # mask. Absolute positions remain in ``x`` before compaction, preserving the timestamp of gaps.
        counts = bm.sum(-1)
        summary = torch.zeros(x.shape[0], self.d_model, dtype=x.dtype, device=x.device)
        for length_value in counts.unique(sorted=True).tolist():
            length = int(length_value)
            if length <= 0:
                continue
            selected = counts == length
            selected_x, selected_mask = x[selected], bm[selected]
            if length == bl:
                packed = selected_x
            else:
                positions = torch.arange(bl, device=x.device).expand(selected_x.shape[0], bl)[selected_mask]
                positions = positions.reshape(selected_x.shape[0], length)
                packed = torch.gather(
                    selected_x, 1, positions.unsqueeze(-1).expand(-1, -1, self.d_model)
                )
            if self.n_layers > 0:
                causal = torch.triu(
                    torch.ones((length, length), dtype=torch.bool, device=x.device), diagonal=1
                )
                packed = self.local(packed, mask=causal)
            else:
                packed = self.local(packed)
            # CUDA autocast LayerNorm returns FP32; indexed assignment requires an explicit compute-dtype cast.
            summary[selected] = self.out_norm(packed[:, -1]).to(dtype=summary.dtype)
        return summary.reshape(B, A, nB, self.d_model).permute(0, 2, 1, 3)


class _NewsAggregator(nn.Module):
    """Aggregates the RAW per-article news scores into a per-stock embedding AT TRAIN TIME (no precomputed
    count/mean). A learned per-article projection, masked-summed over articles (the sum preserves both sentiment
    and volume) then normalized -- the model decides how to use the raw news."""

    def __init__(self, raw_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(raw_dim, out_dim), nn.GELU())
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = int(out_dim)

    def forward(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # scores [B,A,M,raw_dim], mask [B,A,M] -> [B,A,out_dim]
        if mask.numel() == 0:
            all_masked = True
        elif mask.untyped_storage().nbytes() == mask.element_size():
            # Expanded scalar is the default no-news representation. Inspect one value instead of reducing its
            # potentially multi-million-element logical view.
            all_masked = not bool(mask[(0,) * mask.ndim])
        else:
            all_masked = not bool(mask.any())
        if all_masked:
            # Exact semantics are LayerNorm(sum(0 * projection)) = LayerNorm(0). Evaluate that once and expand it
            # over [B,A], retaining a learned LayerNorm bias from a news-enabled checkpoint without allocating the
            # [B,A,M,out_dim] article activation. Zero dependencies keep skipped projection gradients (and thus
            # AdamW decay behavior) equivalent to the ordinary all-masked path: zero tensors rather than None.
            zero = scores.new_zeros(1, 1, self.out_dim)
            for parameter in self.proj.parameters():
                zero = zero + (parameter.sum() * 0.0).to(dtype=zero.dtype)
            return self.norm(zero).expand(*scores.shape[:2], self.out_dim)
        e = self.proj(scores)
        # ``scores`` is commonly an FP32 input even under autocast.  Following its dtype here would widen the
        # dense [B,A,M,out] projected article activation immediately after Linear produced BF16.
        e = e * mask.unsqueeze(-1).to(dtype=e.dtype)
        return self.norm(e.sum(dim=2))


class DecisionPolicyHead(nn.Module):
    """Cross-sectional attention policy over frozen context, policy raw-second context, news, and previous weight.

    Permutation-equivariant and shared-weight across the action axis. The policy-side raw-second encoder and news
    aggregation are train-time model paths, not persisted engineered features.
    """

    def __init__(self, config: DecisionPolicyConfig) -> None:
        super().__init__()
        if not 0 < float(config.max_stock_weight) <= 1:
            raise ValueError("max_stock_weight must lie in (0, 1]")
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
        low_precision_context = per_stock.dtype in (torch.float16, torch.bfloat16)
        assembly_dtype = (
            per_stock.dtype
            if low_precision_context and torch.is_autocast_enabled(per_stock.device.type)
            else torch.float32
        )
        mkt = market.to(dtype=assembly_dtype).unsqueeze(1).expand(B, A, d)
        per_stock = per_stock.to(dtype=assembly_dtype)
        raw_policy_ctx = raw_policy_ctx.to(dtype=assembly_dtype)
        news = self.news_agg(news_scores, news_mask).to(dtype=assembly_dtype)  # in-model raw-news aggregation
        prev_feature = prev_weights.unsqueeze(-1).to(dtype=assembly_dtype)
        tok = self.token_proj(torch.cat([mkt, per_stock, raw_policy_ctx, news, prev_feature], dim=-1))
        cash_marker = (torch.arange(A, device=tok.device) == 0).to(dtype=tok.dtype).view(1, A, 1)
        tok = tok + self.cash_bias.to(dtype=tok.dtype) * cash_marker           # CASH token
        kpm = ~available.bool()                                  # constraint mask: unavailable actions are dropped
        kpm = kpm.clone()
        kpm[:, 0] = False                                        # CASH is always available (abstention sink)
        h = self.attn(tok, src_key_padding_mask=kpm)             # cross-sectional attention (no positional axis)
        scores = self.score(h).squeeze(-1) / self.temperature    # [B,A]; temperature shapes allocation sharpness
        scores = scores.masked_fill(kpm, float("-inf"))          # never allocate to unavailable actions
        weights = torch.softmax(scores, dim=1)
        weights = project_capped_risky_simplex(
            weights,
            ~kpm,
            max_risky_weight=self.config.max_stock_weight,
            cash_index=0,
        )
        avail = (~kpm).to(dtype=h.dtype).unsqueeze(-1)            # gate reads only AVAILABLE actions (incl. CASH)
        summary = (
            (h * avail).sum(dim=1, dtype=torch.float32)
            / avail.sum(dim=1, dtype=torch.float32).clamp_min(1.0)
        )                                                         # stable masked mean; only [B,d] is FP32
        gate = torch.sigmoid(self.gate_head(summary).squeeze(-1))  # [B] trade (->target) vs hold (->prev)
        return weights, gate
