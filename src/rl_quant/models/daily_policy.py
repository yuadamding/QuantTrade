"""Stage-2 DAILY cross-sectional policy WITH cross-day memory -- the ``daily_raw`` path.

This is the day-level redesign. It addresses the two structural gaps of the generic daily mode: (1) the only
profit-trained raw-second representation saw just the last block of each day, and (2) the policy had no learned
memory across days (only the carried portfolio weight). Here:

  * FullDayRawEncoder -- a TRAINABLE two-tier causal transformer over the WHOLE RTH session of raw 1s bars (not
    the last block) -> a per-stock end-of-day embedding. Profit gradients shape it. Per-FIELD BatchNorm (not a
    LayerNorm across OHLCV) so price and volume normalize on their own scales.
  * CrossDayTemporalEncoder -- a CAUSAL transformer over the DAY axis (per stock, shared weights), windowed to a
    `lookback` of prior days, so the policy can compute multi-day patterns (reversal/momentum/vol) from the
    sequence of daily embeddings. This is the learned cross-day memory BPTT alone cannot provide.
  * DailyCrossSectionPolicy -- fuses the FROZEN Stage-1 context (detached), the trainable full-day raw embedding,
    and raw news into a per-day per-stock token; runs the temporal encoder; then a per-day cross-sectional
    set-transformer emits long-only target weights + an act-gate. The portfolio carry / turnover / T+1 credit
    happen in the rollout (rl_quant.training.daily_policy), which carries the position across the whole episode.

The frozen context enters as plain detached tensors -- no gradient reaches the Stage-1 encoder (the context/policy
split holds). Only this module's parameters are trained by the PnL objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn

from rl_quant.models.context_encoder import _CausalBlock, _last_valid, _sinusoidal
from rl_quant.models.decision_policy import _NewsAggregator


class FullDayRawEncoder(nn.Module):
    """Trainable two-tier causal encoder over a full RTH session of raw 1s bars -> per-stock end-of-day embedding.

    Tier-1 attends locally within `block_seconds` blocks; tier-2 attends causally over the block summaries; the
    last valid block's context is the day embedding. Unlike the frozen Stage-1 context encoder, profit gradients
    update this. Per-field BatchNorm normalizes each raw bar field on its own scale (price vs volume)."""

    def __init__(self, *, bar_feature_dim: int, d_model: int, n_heads: int, n_layers: int,
                 feedforward_dim: int, dropout: float, block_seconds: int, max_seconds: int,
                 grad_checkpoint: bool = False) -> None:
        super().__init__()
        d = d_model
        if d % n_heads:
            raise ValueError(f"raw d_model {d} must be divisible by n_heads {n_heads}")
        self.block_seconds = int(block_seconds)
        self.grad_checkpoint = grad_checkpoint
        t1 = max(1, n_layers // 2)
        t2 = max(1, n_layers - t1)
        self.input_proj = nn.Linear(bar_feature_dim, d)
        self.register_buffer("pos1", _sinusoidal(self.block_seconds, d), persistent=False)
        self.register_buffer("pos2", _sinusoidal(max_seconds // max(1, self.block_seconds) + 2, d), persistent=False)
        self.tier1 = nn.ModuleList([_CausalBlock(d, n_heads, feedforward_dim, dropout) for _ in range(t1)])
        self.tier2 = nn.ModuleList([_CausalBlock(d, n_heads, feedforward_dim, dropout) for _ in range(t2)])
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.d_model = d

    def forward(self, bars: torch.Tensor, bar_mask: torch.Tensor) -> torch.Tensor:
        """bars [B,A,S,F] raw OHLCV (session-aligned), bar_mask [B,A,S] -> [B,A,d] end-of-day per-stock embedding."""
        B, A, S, Fdim = bars.shape
        d = self.d_model
        bl = self.block_seconds
        nB = S // bl
        if nB <= 0:
            raise ValueError(f"FullDayRawEncoder needs at least one {bl}s block; got S={S}")
        bars = bars[:, :, :nB * bl]
        bar_mask = bar_mask[:, :, :nB * bl].bool()
        # Per-(stock,day) per-FIELD instance normalization over that stock-day's valid seconds. NO coupling across
        # the batch/day axis (so a future day cannot affect a past day's normalization -> strictly causal) and each
        # field is standardized on its own scale (price vs volume), avoiding the LayerNorm-over-OHLCV units mix.
        # Using the whole day's stats is PIT-clean for the END-OF-DAY embedding (all of day d is known at EOD d).
        m = bar_mask.unsqueeze(-1).to(bars.dtype)               # [B,A,Sd,1]
        cnt = m.sum(dim=2).clamp_min(1.0)                       # [B,A,1]
        mean = (bars * m).sum(dim=2) / cnt                      # [B,A,F]
        var = ((bars - mean.unsqueeze(2)) ** 2 * m).sum(dim=2) / cnt
        normed = ((bars - mean.unsqueeze(2)) / (var.unsqueeze(2) + 1e-5).sqrt()) * m
        x = self.input_proj(normed).reshape(B * A * nB, bl, d) + self.pos1[:bl].view(1, bl, d)
        bm1 = bar_mask.reshape(B * A * nB, bl)
        kpm = ~bm1                                               # key-padding for missing seconds (SDPA-safe in block)
        for blk in self.tier1:                                  # tier-1: local causal within each block
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, kpm, use_reentrant=False)
            else:
                x = blk(x, kpm)
        x = self.norm1(x)
        summ = _last_valid(x, bm1).reshape(B * A, nB, d)        # last valid second per block -> block summary
        block_has = bm1.any(-1).reshape(B * A, nB)
        summ = summ * block_has.unsqueeze(-1).float()
        h = summ + self.pos2[:nB].unsqueeze(0)
        for blk in self.tier2:                                  # tier-2: causal over block summaries
            h = blk(h, ~block_has)
        h = self.norm2(h)
        day = _last_valid(h, block_has).reshape(B, A, d)        # last valid BLOCK = end-of-day per-stock embedding
        return day * block_has.any(-1).reshape(B, A, 1).float()  # zero for stocks absent all day


class CrossDayTemporalEncoder(nn.Module):
    """CAUSAL transformer over the DAY axis (per stock, shared weights) -> learned multi-day memory.

    Input [B,T,A,d] sequence of per-day per-stock embeddings -> [B,T,A,d] where position t attends only to days
    0..t (strictly causal -- no future leak). Per stock, so the representation is permutation-equivariant across
    the action axis. The effective memory horizon is bounded by the training episode length / eval window
    (`daily_lookback`), not by a hard attention band -- attending to all in-window prior days is correct and lets
    the model weight recent vs distant days itself."""

    def __init__(self, *, d_model: int, n_heads: int, n_layers: int, feedforward_dim: int,
                 dropout: float, max_days: int) -> None:
        super().__init__()
        self.register_buffer("pos", _sinusoidal(max_days + 2, d_model), persistent=False)
        self.blocks = nn.ModuleList([_CausalBlock(d_model, n_heads, feedforward_dim, dropout)
                                     for _ in range(max(1, n_layers))])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, seq: torch.Tensor, day_valid: torch.Tensor | None = None) -> torch.Tensor:
        """seq [B,T,A,d] -> [B,T,A,d]. day_valid [B,T,A] (a stock has a real embedding that day) -> absent days are
        masked as attention KEYS (a not-yet-listed stock never feeds the memory); the causal order is in _CausalBlock."""
        B, T, A, d = seq.shape
        if T > self.pos.shape[0]:
            raise ValueError(f"episode/eval length {T} exceeds temporal max_days {self.pos.shape[0]}")
        x = seq.permute(0, 2, 1, 3).reshape(B * A, T, d) + self.pos[:T].unsqueeze(0)  # [B*A, T, d]
        kpm = (~day_valid.bool()).permute(0, 2, 1).reshape(B * A, T) if day_valid is not None else None
        for blk in self.blocks:
            x = blk(x, kpm)
        x = self.norm(x)
        return x.reshape(B, A, T, d).permute(0, 2, 1, 3)         # [B,T,A,d]


@dataclass
class DailyCrossSectionConfig:
    context_dim: int                 # frozen Stage-1 per-stock/market context width (d_model)
    bar_feature_dim: int = 5
    raw_policy_dim: int = 128        # trainable full-day raw encoder width
    raw_policy_layers: int = 2
    raw_policy_heads: int = 4
    raw_block_seconds: int = 300
    session_seconds: int = 23400
    news_raw_dim: int = 1
    max_news: int = 32
    news_embed_dim: int = 32
    token_dim: int = 256             # per-day per-stock token + temporal/allocator width
    temporal_layers: int = 2
    temporal_heads: int = 4
    daily_lookback: int = 60
    max_days: int = 256
    alloc_layers: int = 2
    alloc_heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.0
    temperature: float = 1.0
    gate_init_bias: float = 2.0
    grad_checkpoint: bool = False


class DailyCrossSectionPolicy(nn.Module):
    """Long-only daily cross-sectional policy with cross-day memory. See module docstring.

    encode_episode(): frozen context (detached) + trainable full-day raw + raw news -> per-day per-stock token ->
    causal cross-day temporal state [B,T,A,token_dim]. step(): per-day cross-sectional set-transformer over the
    temporal state + carried weight -> long-only target weights + act-gate. The frozen context never receives a
    gradient (it arrives as a plain tensor)."""

    def __init__(self, config: DailyCrossSectionConfig) -> None:
        super().__init__()
        self.config = config
        self.raw_encoder = FullDayRawEncoder(
            bar_feature_dim=config.bar_feature_dim, d_model=config.raw_policy_dim,
            n_heads=config.raw_policy_heads, n_layers=config.raw_policy_layers,
            feedforward_dim=config.raw_policy_dim * 2, dropout=config.dropout,
            block_seconds=config.raw_block_seconds, max_seconds=config.session_seconds,
            grad_checkpoint=config.grad_checkpoint)
        self.news_agg = _NewsAggregator(config.news_raw_dim, config.news_embed_dim)
        # per-day per-stock token: [market | per-stock frozen ctx | full-day raw | news]
        tok_in = config.context_dim * 2 + config.raw_policy_dim + config.news_embed_dim
        self.token_proj = nn.Linear(tok_in, config.token_dim)
        self.temporal = CrossDayTemporalEncoder(
            d_model=config.token_dim, n_heads=config.temporal_heads, n_layers=config.temporal_layers,
            feedforward_dim=config.feedforward_dim, dropout=config.dropout, max_days=config.max_days)
        # allocator: cross-sectional set-transformer over [temporal state | prev weight] per day
        self.alloc_in = nn.Linear(config.token_dim + 1, config.token_dim)
        self.cash_bias = nn.Parameter(torch.zeros(config.token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim, nhead=config.alloc_heads, dim_feedforward=config.feedforward_dim,
            dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.attn = nn.TransformerEncoder(layer, num_layers=config.alloc_layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1))
        self.gate_head = nn.Sequential(nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1))
        nn.init.constant_(self.gate_head[-1].bias, config.gate_init_bias)
        self.temperature = config.temperature
        self.token_dim = config.token_dim

    def encode_episode(self, market, per_stock, bars, bar_mask, news_raw, news_mask, avail):
        """Precompute the cross-day temporal state for a whole episode (once per rollout).

        market [B,T,dc] (detached frozen), per_stock [B,T,A,dc] (detached frozen), bars [B,T,A,S,F],
        bar_mask [B,T,A,S], news_raw [B,T,A,M,raw], news_mask [B,T,A,M], avail [B,T,A].
        -> temporal_state [B,T,A,token_dim]."""
        B, T, A, dc = per_stock.shape
        raw = self.raw_encoder(bars.reshape(B * T, A, bars.shape[3], bars.shape[4]),
                               bar_mask.reshape(B * T, A, bars.shape[3])).reshape(B, T, A, -1)  # [B,T,A,dr]
        news = self.news_agg(news_raw.reshape(B * T, A, news_raw.shape[3], news_raw.shape[4]),
                             news_mask.reshape(B * T, A, news_mask.shape[3])).reshape(B, T, A, -1)  # [B,T,A,ne]
        mkt = market.unsqueeze(2).expand(B, T, A, dc)
        tok = self.token_proj(torch.cat([mkt, per_stock, raw, news], dim=-1))   # [B,T,A,token_dim]
        state = self.temporal(tok, day_valid=avail.bool())                      # causal cross-day memory
        return state

    def step(self, state_t, prev_weights, available):
        """One day's cross-sectional allocation. state_t [B,A,token_dim], prev_weights/available [B,A]
        -> (weights [B,A] long-only over {CASH,stocks}, gate [B])."""
        B, A, _ = state_t.shape
        tok = self.alloc_in(torch.cat([state_t, prev_weights.unsqueeze(-1)], dim=-1))
        tok = tok + self.cash_bias * (torch.arange(A, device=tok.device) == 0).float().view(1, A, 1)
        kpm = ~available.bool()
        kpm = kpm.clone()
        kpm[:, 0] = False                                        # CASH always available
        h = self.attn(tok, src_key_padding_mask=kpm)
        scores = self.score(h).squeeze(-1) / self.temperature
        scores = scores.masked_fill(kpm, float("-inf"))
        weights = torch.softmax(scores, dim=1)                   # long-only, sums to 1
        avail = (~kpm).float().unsqueeze(-1)
        summary = (h * avail).sum(dim=1) / avail.sum(dim=1).clamp_min(1.0)
        gate = torch.sigmoid(self.gate_head(summary).squeeze(-1))
        return weights, gate


class DailyForwardHead(nn.Module):
    """Daily SSL pretext head: from each stock's per-day context predict its next-H-day cross-sectionally
    demeaned close-to-close return (the daily relative-value target). Trained jointly with Stage-1, then discarded."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, per_stock: torch.Tensor) -> torch.Tensor:
        return self.net(per_stock).squeeze(-1)
