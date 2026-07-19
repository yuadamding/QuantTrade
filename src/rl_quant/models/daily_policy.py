"""Stage-2 DAILY cross-sectional policy WITH cross-day memory -- the ``daily_raw`` path.

This is the day-level redesign. It addresses the two structural gaps of the generic daily mode: (1) the only
profit-trained raw-second representation saw just the last block of each day, and (2) the policy had no learned
memory across days (only the carried portfolio weight). Here:

  * FullDayRawEncoder -- a TRAINABLE two-tier causal transformer over the WHOLE RTH session of raw 1s bars (not
    the last block) -> a per-stock end-of-day embedding. Profit gradients shape it. Stock-day-local, per-field
    normalization keeps price and volume on meaningful independent scales without batch/future-day coupling.
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

from rl_quant.models.context_encoder import _CausalBlock, _sinusoidal
from rl_quant.models.decision_policy import _NewsAggregator


class FullDayRawEncoder(nn.Module):
    """Trainable two-tier causal encoder over a full RTH session of raw 1s bars -> per-stock end-of-day embedding.

    Tier-1 attends locally within `block_seconds` blocks; tier-2 attends causally over the block summaries; the
    last valid block's context is the day embedding. Unlike the frozen Stage-1 context encoder, profit gradients
    update this. Its normalization never couples separate stocks or days."""

    pos1: torch.Tensor
    pos2: torch.Tensor

    def __init__(self, *, bar_feature_dim: int, d_model: int, n_heads: int, n_layers: int,
                 feedforward_dim: int, dropout: float, block_seconds: int, max_seconds: int,
                 grad_checkpoint: bool = False, raw_norm: str = "instance", stock_chunk: int = 0) -> None:
        super().__init__()
        d = d_model
        if d % n_heads:
            raise ValueError(f"raw d_model {d} must be divisible by n_heads {n_heads}")
        if raw_norm not in ("instance", "level"):
            raise ValueError(f"raw_norm must be 'instance' or 'level', got {raw_norm!r}")
        self.block_seconds = int(block_seconds)
        self.grad_checkpoint = grad_checkpoint
        self.raw_norm = raw_norm
        self.stock_chunk = int(stock_chunk)   # >0: encode the stock axis in chunks (bit-identical: every norm here
        #                                       is per-(stock,day); huge universes need it for activation memory)
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
        """bars [B,A,S,F] raw OHLCV (session-aligned), bar_mask [B,A,S] -> [B,A,d] end-of-day per-stock embedding.
        `stock_chunk>0` encodes the (weight-shared, per-stock-normalized) stock axis in chunks -- bit-identical,
        bounded activation memory; with grad_checkpoint each chunk is checkpointed (backward recomputes one chunk)."""
        A = bars.shape[1]
        ck = self.stock_chunk if self.stock_chunk and 0 < self.stock_chunk < A else A
        if ck >= A:
            return self._encode_stocks(bars, bar_mask)
        outs = []
        for lo in range(0, A, ck):
            bc, mc = bars[:, lo:lo + ck], bar_mask[:, lo:lo + ck]
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                outs.append(torch.utils.checkpoint.checkpoint(self._encode_stocks, bc, mc, use_reentrant=False))
            else:
                outs.append(self._encode_stocks(bc, mc))
        return torch.cat(outs, dim=1)                            # [B,A,d]

    def _encode_stocks(self, bars: torch.Tensor, bar_mask: torch.Tensor) -> torch.Tensor:
        B, A, S, Fdim = bars.shape
        d = self.d_model
        bl = self.block_seconds
        nB = S // bl
        if nB <= 0:
            raise ValueError(f"FullDayRawEncoder needs at least one {bl}s block; got S={S}")
        bars = bars[:, :, :nB * bl]
        bar_mask = bar_mask[:, :, :nB * bl].bool()
        # Per-(stock,day) normalization over that stock-day's valid seconds. BOTH modes have NO coupling across the
        # batch/day axis (a future day cannot affect a past day's normalization -> strictly causal) and use only
        # day-d's own bars (PIT-clean for the END-OF-DAY embedding). They differ in what they preserve:
        m = bar_mask.unsqueeze(-1).to(bars.dtype)               # [B,A,Sd,1]
        cnt = m.sum(dim=2).clamp_min(1.0)                       # [B,A,1]
        if self.raw_norm == "instance":
            # per-FIELD standardize: affine-invariant, so the day's intraday move MAGNITUDE is whitened away (only
            # the vol-normalized SHAPE survives) -- bad for a cross-sectional RETURN policy, kept for back-compat.
            mean = (bars * m).sum(dim=2) / cnt                  # [B,A,F]
            var = ((bars - mean.unsqueeze(2)) ** 2 * m).sum(dim=2) / cnt
            normed = ((bars - mean.unsqueeze(2)) / (var.unsqueeze(2) + 1e-5).sqrt()) * m
        else:                                                   # "level": magnitude-preserving (the daily_raw default)
            # Prices -> deviation from the day's mean CLOSE expressed in RETURN units (divide by the price level, do
            # NOT divide by std): multiplicatively scale-INVARIANT (a $5 and a $500 name are comparable; splits don't
            # matter) yet magnitude-SENSITIVE (a +5% day reads ~10x a +0.5% day -- the cross-sectional signal the
            # instance norm destroyed). Volume -> centered log1p (relative intraday volume; absolute level isn't a
            # return signal). This is INPUT NORMALIZATION of raw OHLCV, not an engineered feature column.
            price, vol = bars[..., :4], bars[..., 4:]
            anchor = ((bars[..., 3:4] * m).sum(dim=2) / cnt).clamp_min(1e-2)        # [B,A,1] mean close level
            price_n = (price - anchor.unsqueeze(2)) / anchor.unsqueeze(2)           # ~ price/anchor - 1 (return units)
            vlog = torch.log1p(vol.clamp_min(0.0))
            vol_n = vlog - (vlog * m).sum(dim=2, keepdim=True) / cnt.unsqueeze(2)   # centered log-volume
            normed = torch.cat([price_n, vol_n], dim=-1) * m
        x = self.input_proj(normed).reshape(B * A * nB, bl, d) + self.pos1[:bl].view(1, bl, d)
        bm1 = bar_mask.reshape(B * A * nB, bl)

        def packed_last(rows: torch.Tensor, valid: torch.Tensor, blocks: nn.ModuleList,
                        norm: nn.Module) -> torch.Tensor:
            """Encode ragged causal rows and retain only their last valid state.

            Grouping by valid length lets SDPA use its native causal kernel. In contrast, combining a key-padding
            mask with causality materializes an ``[N, heads, S, S]`` mask; at full-session universe sizes that mask
            alone can consume multiple GiB. Positional embeddings are already attached before compaction, so a
            gap's absolute time remains represented even though invalid query/key slots are not carried through
            attention. Empty rows remain exactly zero.
            """
            counts = valid.sum(-1)
            result = torch.zeros(rows.shape[0], d, dtype=rows.dtype, device=rows.device)
            for length_value in counts.unique(sorted=True).tolist():
                length = int(length_value)
                if length <= 0:
                    continue
                selected = counts == length
                selected_rows, selected_valid = rows[selected], valid[selected]
                if length == rows.shape[1]:
                    packed = selected_rows
                else:
                    positions = torch.arange(rows.shape[1], device=rows.device).expand(
                        selected_rows.shape[0], -1
                    )[selected_valid].reshape(selected_rows.shape[0], length)
                    packed = torch.gather(selected_rows, 1, positions.unsqueeze(-1).expand(-1, -1, d))
                for block in blocks:
                    if self.grad_checkpoint and self.training:
                        packed = torch.utils.checkpoint.checkpoint(
                            lambda value, layer=block: layer(value, None), packed, use_reentrant=False
                        )
                    else:
                        packed = block(packed, None)
                result[selected] = norm(packed[:, -1])
            return result

        summ = packed_last(x, bm1, self.tier1, self.norm1).reshape(B * A, nB, d)
        block_has = bm1.any(-1).reshape(B * A, nB)
        summ = summ * block_has.unsqueeze(-1).float()
        h = summ + self.pos2[:nB].unsqueeze(0)
        # Only the final day state is consumed, so tier 2 can use the same ragged-last path and avoid retaining a
        # padded full-session output. Missing blocks still keep their absolute ``pos2`` timestamp.
        day = packed_last(h, block_has, self.tier2, self.norm2).reshape(B, A, d)
        return day * block_has.any(-1).reshape(B, A, 1).float()  # zero for stocks absent all day


class CrossDayTemporalEncoder(nn.Module):
    """CAUSAL transformer over the DAY axis (per stock, shared weights) -> learned multi-day memory.

    Input [B,T,A,d] sequence of per-day per-stock embeddings -> [B,T,A,d] where position t attends only to days
    0..t (strictly causal -- no future leak). Per stock, so the representation is permutation-equivariant across
    the action axis. The effective memory horizon is bounded by the training episode length / eval window
    (`daily_lookback`), not by a hard attention band -- attending to all in-window prior days is correct and lets
    the model weight recent vs distant days itself."""

    pos: torch.Tensor

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
    raw_norm: str = "level"          # full-day raw input norm: "level" (magnitude-preserving) | "instance" (whitened)
    raw_recent_days: int = 0         # TWO-SPEED tokens: >0 -> only the LAST this-many days of an episode/eval
    #                                  window get the (expensive, trainable) full-day raw encode; older days'
    #                                  tokens carry frozen ctx + news + the past-return channel only (has_raw=0).
    #                                  Extends the cross-day memory to e.g. 252d at ~the 42d raw compute.
    #                                  0 = every day raw (the original behavior).
    raw_stock_chunk: int = 0         # >0: the full-day raw encoder processes the stock axis in chunks of this
    #                                  many stocks (bit-identical -- all its norms are per-(stock,day)); REQUIRED
    #                                  for huge universes (TOP2000: ~512/chunk on an 80GB H100). 0 = single pass.


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
            grad_checkpoint=config.grad_checkpoint, raw_norm=config.raw_norm,
            stock_chunk=config.raw_stock_chunk)
        self.news_agg = _NewsAggregator(config.news_raw_dim, config.news_embed_dim)
        # per-day per-stock token: [market | per-stock frozen ctx | full-day raw | news | past_ret | past_valid |
        # has_raw]. past_ret = the stock's OWN 1-day close-to-close return for that day (PIT: known at EOD) -- the
        # raw close series under a scale-invariant normalization, so the cross-day temporal encoder can compute
        # momentum/reversal over its window; has_raw flags whether the raw component is real or a two-speed zero.
        tok_in = config.context_dim * 2 + config.raw_policy_dim + config.news_embed_dim + 3
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

    def _raw_day(self, day_bars_fn, t):
        """Full-day raw embedding for day t across the batch: day_bars_fn(t) -> (bars [B,A,S,F], mask [B,A,S]).
        The raw encoder is per-(stock,day) independent (instance-norm, no batch coupling), so encoding day-by-day
        is bit-identical to encoding the [B*T] reshape at once."""
        bars_t, mask_t = day_bars_fn(t)
        return self.raw_encoder(bars_t, mask_t)                # [B,A,dr]

    def _raw_day_mask(self, T: int) -> list[bool]:
        """Two-speed assignment for a length-T episode: the last `raw_recent_days` days get the trainable raw
        encode (all days if raw_recent_days<=0)."""
        r = self.config.raw_recent_days
        return [True] * T if r <= 0 else [t >= T - r for t in range(T)]

    def _episode_tokens(self, market, per_stock, day_bars_fn, news_raw, news_mask, past_ret, past_ret_valid,
                        raw_day_mask, reload_ckpt):
        """Build the per-day per-stock TOKENS (everything BEFORE the cross-day temporal encoder): frozen context +
        (two-speed) trainable full-day raw + news + the past-return channel -> tok [B,T,A,token_dim].
        day_bars_fn(t) yields day-t bars/mask (a tensor slice in-RAM, or a lazy disk load when streaming);
        raw_day_mask[t] selects which days get the raw encode (False -> zeros + has_raw=0: the day contributes
        frozen ctx/news/past-return only -- and its bars are NEVER loaded, the two-speed compute saving). When
        `reload_ckpt` and training, each raw day's encode is checkpointed so backward RE-LOADS + recomputes it."""
        B, T, A, dc = per_stock.shape
        ckpt = reload_ckpt and self.training
        dr = self.config.raw_policy_dim
        raw_days = []
        for t in range(T):
            if not raw_day_mask[t]:
                raw_days.append(torch.zeros(B, A, dr, device=per_stock.device, dtype=per_stock.dtype))
            elif ckpt:
                raw_days.append(torch.utils.checkpoint.checkpoint(self._raw_day, day_bars_fn, t, use_reentrant=False))
            else:
                raw_days.append(self._raw_day(day_bars_fn, t))
        raw = torch.stack(raw_days, dim=1)                     # [B,T,A,dr]
        news = self.news_agg(news_raw.reshape(B * T, A, news_raw.shape[3], news_raw.shape[4]),
                             news_mask.reshape(B * T, A, news_mask.shape[3])).reshape(B, T, A, -1)  # [B,T,A,ne]
        mkt = market.unsqueeze(2).expand(B, T, A, dc)
        flag = torch.tensor(raw_day_mask, device=per_stock.device, dtype=per_stock.dtype)
        flag = flag.view(1, T, 1, 1).expand(B, T, A, 1)
        # Fixed input scaling to ~unit variance (daily moves are ~2%, the other token channels are ~unit scale;
        # unscaled, the momentum channel would start ~100x under-weighted into token_proj). A constant, applied
        # identically everywhere -- input normalization, not a learned/engineered feature.
        pr = (past_ret * 50.0).unsqueeze(-1).to(per_stock.dtype)
        pv = past_ret_valid.unsqueeze(-1).to(per_stock.dtype)
        return self.token_proj(torch.cat([mkt, per_stock, raw, news, pr, pv, flag], dim=-1))  # [B,T,A,token_dim]

    def temporal_state(self, tok, avail):
        """Run the CAUSAL cross-day memory over a (possibly windowed) token slice. tok [B,W,A,token_dim],
        avail [B,W,A] -> [B,W,A,token_dim]. For a rolling EVAL the caller slices the last `daily_lookback` days so
        the memory horizon (and positional range) matches what TRAINING exercised (episode_len), not the full
        split -- otherwise eval runs the temporal encoder at sequence positions/contexts it never saw in training."""
        return self.temporal(tok, day_valid=avail.bool())

    def encode_episode(self, market, per_stock, bars, bar_mask, news_raw, news_mask, avail,
                       past_ret, past_ret_valid):
        """In-RAM encode: bars/bar_mask are pre-stacked [B,T,A,S,F]/[B,T,A,S]. -> temporal_state [B,T,A,token_dim].
        Two-speed: only the last `raw_recent_days` days are raw-encoded (all, if 0)."""
        T = per_stock.shape[1]
        tok = self._episode_tokens(market, per_stock, lambda t: (bars[:, t], bar_mask[:, t]),
                                   news_raw, news_mask, past_ret, past_ret_valid,
                                   self._raw_day_mask(T), reload_ckpt=False)
        return self.temporal_state(tok, avail)

    def encode_episode_streaming(self, market, per_stock, day_bars_fn, news_raw, news_mask, avail, n_days,
                                 past_ret, past_ret_valid):
        """Streaming encode: day_bars_fn(t) lazily loads day-t bars/mask [B,A,S,F]/[B,A,S] from disk; backward
        reloads + recomputes per day (reload_ckpt) so the whole episode's bars are never resident. Two-speed days
        outside `raw_recent_days` never load their bars at all."""
        tok = self._episode_tokens(market, per_stock, day_bars_fn, news_raw, news_mask, past_ret, past_ret_valid,
                                   self._raw_day_mask(n_days), reload_ckpt=True)
        return self.temporal_state(tok, avail)

    def encode_tokens_dual(self, market, per_stock, day_bars_fn, news_raw, news_mask, past_ret, past_ret_valid):
        """EVAL: BOTH token variants for every day, computed ONCE -- (tok_raw [B,T,A,D], tok_noraw [B,T,A,D]).
        The expensive full-day raw encode runs once per day (for tok_raw); tok_noraw re-projects the same
        ctx/news/past-return with a zero raw component + has_raw=0. The rolling-window rollout then assembles each
        decision's slice PER-DECISION (raw variant for its most recent `raw_recent_days`, no-raw before), matching
        the two-speed geometry training saw. day_bars_fn(t) -> (bars [B,A,S,F], mask [B,A,S]); works for in-RAM
        slices and streaming loaders alike (eval is no_grad, so no reload-checkpoint is needed)."""
        T = per_stock.shape[1]
        tok_raw = self._episode_tokens(market, per_stock, day_bars_fn, news_raw, news_mask,
                                       past_ret, past_ret_valid, [True] * T, reload_ckpt=False)
        tok_noraw = self._episode_tokens(market, per_stock, day_bars_fn, news_raw, news_mask,
                                         past_ret, past_ret_valid, [False] * T, reload_ckpt=False)
        return tok_raw, tok_noraw

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
