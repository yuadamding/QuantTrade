"""Stage-1 CONTEXT LEARNING model: a TWO-TIER causal-attention transformer over the RAW 1-second bars.

The encoder consumes the raw per-second bars DIRECTLY -- one token per second, the raw OHLCV values -- with NO
pooling and NO hand-computed (scale-free) features. The only transform is the model's own input normalization
(a BatchNorm layer over the raw bar fields, learned at train time) + a linear embedding: that is the model
learning from the data, not precomputed feature engineering. A full RTH session is fed SESSION-ALIGNED (index s =
second s after the 09:30 open) and attention is CAUSAL (is_causal), so a block's context depends only on the
seconds up to that block -- no look-ahead, and padding (after a stock's valid tail) is never attended.

The encoder produces a context at EVERY `block_seconds` block of the session (the candidate/decision grid for the
event-timed policy): tier-1 attends locally within each block, tier-2 attends causally across the block summaries.
Full-session SSL is dominated by the tier-1 activations, so `grad_checkpoint` recomputes tier-1 in backward.

This module is PURE market context: no action/policy concept (the enforced context/policy split). Per block, the
per-stock contexts (bars fused with as-of covariates) are pooled cross-sectionally into a market-context vector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


@dataclass
class ContextEncoderConfig:
    bar_feature_dim: int             # number of RAW bar fields per second (e.g. OHLCV = 5)
    covariate_dim: int               # number of as-of stock covariate fields (fundamentals/market-cap/news-vol)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2                # split across the two tiers (tier1 = n_layers//2, tier2 = the rest)
    feedforward_dim: int = 256
    dropout: float = 0.0
    max_seconds: int = 3600          # full session length in seconds (rolls from the 09:30 open)
    block_seconds: int = 300         # tier-1 block length: seconds attended LOCALLY before the global tier-2
    grad_checkpoint: bool = False    # recompute tier-1 blocks in backward (full-session SSL memory relief)


def _sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class _CausalBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention.

    When no key padding is needed, SDPA's native causal path avoids a materialized [S,S] mask. If padding is
    present, causal and key-valid constraints are combined into one boolean mask because SDPA rejects mixing a
    custom mask with `is_causal=True` on some PyTorch versions.
    """

    def __init__(self, d: int, n_heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.attn_dropout = dropout
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:  # x [N, S, d]
        N, S, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(N, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each [N, n_heads, S, head_dim]
        attn_mask = None
        is_causal = True
        if key_padding_mask is not None and bool(key_padding_mask.any()):
            kpm = key_padding_mask.bool()
            if bool(kpm.all(dim=1).any()):
                kpm = kpm.clone()
                kpm[kpm.all(dim=1), 0] = False
            key_allowed = (~kpm).view(N, 1, 1, S)        # bool mask: True keys may be attended
            causal_allowed = torch.ones(S, S, dtype=torch.bool, device=x.device).tril().view(1, 1, S, S)
            attn_mask = causal_allowed & key_allowed
            is_causal = False
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        a = a.transpose(1, 2).reshape(N, S, d)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


def _last_valid(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather each sequence's LAST valid token by its boolean mask -- correct even when valid positions are NOT
    contiguous (bars sit at absolute second-offsets). seq [N,L,d], mask [N,L] bool -> [N,d] (position 0 if none).
    Cheap: gathers a single position per sequence (no full-length gather)."""
    ar = torch.arange(seq.shape[1], device=seq.device)
    idx = torch.where(mask, ar, torch.full_like(ar, -1)).amax(dim=1).clamp_min(0)
    return torch.gather(seq, 1, idx[:, None, None].expand(seq.shape[0], 1, seq.shape[2])).squeeze(1)


class ContextEncoder(nn.Module):
    """TWO-TIER causal transformer over RAW 1-second bars, fused with the stock's as-of covariates.
      Tier 1: LOCAL causal attention within fixed `block_seconds` blocks of raw seconds -> a LEARNED summary per
              block (its most-recent-valid token). The model compresses raw seconds; nothing is hand-pooled.
      Tier 2: GLOBAL causal attention over the block summaries across the whole session -> per-stock session
              context (its most-recent-valid block). Reaches the full session at O(S*block) + O(n_blocks^2) cost.
    Cross-sectional mean over all involved stocks (bars + covariates) -> market context. PURE market state."""

    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.block_seconds = config.block_seconds
        self.grad_checkpoint = config.grad_checkpoint
        t1 = max(1, config.n_layers // 2)
        t2 = max(1, config.n_layers - t1)
        self.bar_norm = nn.BatchNorm1d(config.bar_feature_dim)   # input normalization over RAW bar fields
        self.input_proj = nn.Linear(config.bar_feature_dim, d)   # learned embedding of the raw bar
        self.register_buffer("pos1", _sinusoidal(config.block_seconds, d), persistent=False)  # within-block
        n_blocks_max = config.max_seconds // max(1, config.block_seconds) + 2
        self.register_buffer("pos2", _sinusoidal(n_blocks_max, d), persistent=False)           # over blocks
        self.tier1 = nn.ModuleList([_CausalBlock(d, config.n_heads, config.feedforward_dim, config.dropout)
                                    for _ in range(t1)])
        self.tier2 = nn.ModuleList([_CausalBlock(d, config.n_heads, config.feedforward_dim, config.dropout)
                                    for _ in range(t2)])
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        # covariate path: the encoder also learns from each stock's as-of covariates
        self.cov_norm = nn.BatchNorm1d(config.covariate_dim)
        self.cov_mlp = nn.Sequential(nn.Linear(config.covariate_dim, d), nn.GELU(), nn.Linear(d, d))
        self.fuse = nn.LayerNorm(d)
        self.d_model = d

    def forward(self, bars: torch.Tensor, bar_mask: torch.Tensor, cov_blocks: torch.Tensor):
        """Encode a full session per (batch) day -> a context at EVERY 5-min block (causal). The decision at
        block b uses only blocks 0..b (no look-ahead). bars [B,A,S,F] RAW (session-aligned: index s = second s
        after the 09:30 open), bar_mask [B,A,S], cov_blocks [B,nB,A,C] (as-of covariates at each block).
        -> per_stock [B,nB,A,d], market [B,nB,d]."""
        B, A, S, F = bars.shape
        d = self.d_model
        bl = self.block_seconds
        nB = S // bl
        if nB * bl != S:                                         # pad the session up to a whole number of blocks
            pad = (nB + 1) * bl - S
            bars = torch.nn.functional.pad(bars, (0, 0, 0, pad))
            bar_mask = torch.nn.functional.pad(bar_mask, (0, pad))
            S = bars.shape[2]
            nB = S // bl
        # input normalization on the RAW valid bars (per-feature), then learned embedding
        flat = bars.reshape(-1, F)
        mv = bar_mask.reshape(-1)
        normed = torch.zeros_like(flat)
        if mv.any():
            normed[mv] = self.bar_norm(flat[mv])
        x = self.input_proj(normed).reshape(B * A * nB, bl, d) + self.pos1[:bl].view(1, bl, d)
        # --- Tier 1: local causal attention within each block -> learned per-block summaries.
        # Per-block checkpointing: checkpoint each _CausalBlock independently. During forward, each block's
        # input is saved (n_blocks × [B*A*nB, bl, d] ≈ 9.6 GB for d512/8L with B=1 day). During backward,
        # only ONE block's intermediates are recomputed and held at a time (~12 GB peak), so the backward
        # peak is ~24 GB instead of ~70+ GB from a one-segment recompute of all 8 blocks simultaneously.
        bm1 = bar_mask.reshape(B * A * nB, bl)                   # per-block validity (bars sit at absolute offsets)
        kpm1 = ~bm1
        if self.grad_checkpoint and self.training:
            for blk in self.tier1:
                x = torch.utils.checkpoint.checkpoint(blk, x, kpm1, use_reentrant=False)
        else:
            for blk in self.tier1:
                x = blk(x, kpm1)
        x = self.norm1(x)
        summ = _last_valid(x, bm1).reshape(B * A, nB, d)         # TRUE last-valid second's token (gap-correct, cheap)
        block_has = bm1.any(-1).reshape(B * A, nB)               # [B*A, nB]
        summ = summ * block_has.unsqueeze(-1).float()
        # --- Tier 2: global causal attention over block summaries -> a context at EVERY block ---
        h = summ + self.pos2[:nB].unsqueeze(0)
        for blk in self.tier2:
            h = blk(h, ~block_has)
        h = self.norm2(h)                                        # [B*A, nB, d] per-block tier-2 context
        bar_blocks = h.reshape(B, A, nB, d).permute(0, 2, 1, 3)  # [B, nB, A, d]
        # Once a stock has traded, later no-trade blocks still get the causal stale context emitted at that
        # timestamp. Empty blocks are not attention keys above, so they cannot become synthetic history.
        seen_blocks = block_has.to(torch.int8).cummax(dim=1).values.bool()
        seen = seen_blocks.reshape(B, A, nB).permute(0, 2, 1).unsqueeze(-1).float()
        has = seen                                                     # [B, nB, A, 1]
        cf = cov_blocks[:, :nB].reshape(-1, cov_blocks.shape[-1])   # [B*nB*A, C]
        cm = has.reshape(-1) > 0                                    # normalize only PRESENT-stock rows (mirror bars):
        cov_flat = torch.zeros_like(cf)                            # absent-stock zero rows must not pollute BN stats
        if cm.any():
            cov_flat[cm] = self.cov_norm(cf[cm])
        cov = self.cov_mlp(cov_flat).reshape(B, nB, A, d)          # learned C->d embedding, per block
        per_stock = self.fuse(bar_blocks + cov) * has            # fuse bars + as-of covariates, per block
        market = per_stock.sum(dim=2) / has.sum(dim=2).clamp_min(1.0)   # cross-sectional mean per block
        return per_stock, market


class ContextForwardHead(nn.Module):
    """Self-supervised pretext head: from the market context predict the next-interval [equal-weight market
    return, realized vol]. Trained jointly with the encoder in Stage 1, then discarded (the encoder is frozen)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        return self.net(market)


class PerStockForwardHead(nn.Module):
    """Self-supervised CROSS-SECTIONAL pretext head: from each stock's per-block context predict that stock's
    next-block CROSS-SECTIONALLY-DEMEANED return (r_i - equal-weight market). The market head alone trains only
    the pooled mean, so per-stock embeddings get no gradient rewarding relative-value discrimination -- this head
    puts a direct signal on each stock's embedding so the FROZEN context carries the which-stock-wins information
    the policy needs. Trained jointly in Stage 1, then discarded."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, per_stock: torch.Tensor) -> torch.Tensor:
        return self.net(per_stock).squeeze(-1)   # [B,nB,A,d] -> [B,nB,A]
