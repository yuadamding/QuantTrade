"""Stage-1 CONTEXT LEARNING model: a TWO-TIER causal-attention transformer over the RAW 1-second bars.

The encoder consumes the raw per-second bars DIRECTLY -- one token per second, the raw OHLCV values -- with NO
pooling and NO hand-computed (scale-free) features. The only transform before the learned linear embedding is a
fixed per-field standardization. Its moments are updated ONLY through ``calibrate_normalization`` on training
data; ``forward`` never reads statistics from the submitted session and never mutates them. An uncalibrated field
is passed through unchanged. A full RTH session is fed SESSION-ALIGNED (index s = second s after the 09:30 open)
and attention is CAUSAL: unpadded rows use SDPA's native causal path, while padded rows use one combined causal +
key-valid mask. A block's context depends only on the seconds up to that block -- no look-ahead, and padding
(after a stock's valid tail) is never attended.

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
    stock_chunk: int = 0             # >0: process the stock axis in chunks of this many stocks (REQUIRED for huge
    #                                  universes -- one un-chunked TOP2000 day is a ~0.5TB tier-1 activation).
    #                                  NUMERICALLY IDENTICAL to un-chunked (exact at eval / dropout=0): per-stock
    #                                  weight-shared, and input normalization uses fixed calibrated statistics.
    #                                  With dropout>0 the RNG is consumed per-chunk, so train-mode is statistically
    #                                  -- not bit -- equivalent (the chunk value is part of the config identity).
    #                                  With grad_checkpoint, each chunk is also checkpointed (backward recomputes
    #                                  one chunk at a time; the checkpoint keeps the normalized-bars base alive).
    #                                  0 = single pass (small universes).


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


class _FixedFeatureNormalizer(nn.Module):
    """Fixed per-feature standardization with explicitly calibrated streaming moments.

    Unlike BatchNorm, ``forward`` is a pure read: train/eval mode and the other samples in a forward batch cannot
    change an observation's normalized value. ``update`` is the sole mutation API. It excludes masked and
    non-finite feature values and merges each incoming batch with the persistent population moments using the
    parallel/Chan variance formula. Features with no calibration samples use the identity transform.

    ``sample_count``, ``running_mean`` and ``running_var`` are float64 state-dict buffers. Keeping the accumulated
    moments in float64 makes repeated calibration calls insensitive to ordinary batching differences while the
    normalized output is converted back to the input dtype.
    """

    sample_count: torch.Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.register_buffer("sample_count", torch.zeros(num_features, dtype=torch.float64))
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(num_features, dtype=torch.float64))

    @property
    def calibrated(self) -> bool:
        """Whether every feature has at least one valid calibration sample."""
        return bool((self.sample_count > 0).all())

    @torch.no_grad()
    def reset(self) -> None:
        """Discard all calibrated moments and restore the identity transform."""
        self.sample_count.zero_()
        self.running_mean.zero_()
        self.running_var.fill_(1.0)

    @torch.no_grad()
    def update(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Merge valid samples from ``values[..., F]`` into the fixed moments.

        ``mask`` may select whole samples (shape ``values.shape[:-1]``) or individual features (the same shape as
        ``values``). Non-finite values are always excluded feature-by-feature. This method is intentionally never
        called by ``forward``; callers must calibrate on point-in-time training data explicitly.
        """
        if values.ndim < 1 or values.shape[-1] != self.num_features:
            raise ValueError(
                f"expected values[..., {self.num_features}], got shape {tuple(values.shape)}"
            )
        flat = values.reshape(-1, self.num_features)
        sample_mask: torch.Tensor | None = None
        feature_mask: torch.Tensor | None = None
        if mask is not None:
            if mask.shape == values.shape[:-1]:
                sample_mask = mask.to(device=values.device, dtype=torch.bool).reshape(-1)
            elif mask.shape == values.shape:
                feature_mask = mask.to(device=values.device, dtype=torch.bool).reshape(-1, self.num_features)
            else:
                raise ValueError(
                    "mask must have shape values.shape[:-1] or values.shape; "
                    f"got {tuple(mask.shape)} for values {tuple(values.shape)}"
                )

        stats_device = self.running_mean.device
        batch_count = torch.zeros_like(self.sample_count)
        batch_mean = torch.zeros_like(self.running_mean)
        batch_m2 = torch.zeros_like(self.running_var)
        # Feature counts may differ because the mask can be feature-wise and non-finite values are excluded.
        # OHLCV/covariate dimensionality is small, so processing one column at a time also avoids allocating a
        # second full [N,F] validity tensor for production-scale universes.
        for feature in range(self.num_features):
            column = flat[:, feature]
            valid = torch.isfinite(column)
            if sample_mask is not None:
                valid = valid & sample_mask
            elif feature_mask is not None:
                valid = valid & feature_mask[:, feature]
            selected = column[valid]
            if selected.numel() == 0:
                continue
            batch_count[feature] = selected.numel()
            if not selected.is_floating_point() or selected.dtype in (torch.float16, torch.bfloat16):
                selected = selected.float()
            variance, mean = torch.var_mean(selected, correction=0)
            batch_mean[feature] = mean.to(device=stats_device, dtype=torch.float64)
            batch_m2[feature] = variance.to(device=stats_device, dtype=torch.float64) * batch_count[feature]

        old_count = self.sample_count
        total = old_count + batch_count
        safe_total = total.clamp_min(1.0)
        delta = batch_mean - self.running_mean
        merged_mean = self.running_mean + delta * batch_count / safe_total
        old_m2 = self.running_var * old_count
        correction = delta.square() * old_count * batch_count / safe_total
        merged_m2 = old_m2 + batch_m2 + correction
        merged_var = torch.where(total > 0, merged_m2 / safe_total, torch.ones_like(merged_m2))

        self.sample_count.copy_(total)
        self.running_mean.copy_(torch.where(total > 0, merged_mean, self.running_mean))
        self.running_var.copy_(merged_var.clamp_min(0.0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.num_features:
            raise ValueError(
                f"expected values[..., {self.num_features}], got shape {tuple(values.shape)}"
            )
        fitted = self.sample_count > 0
        mean = torch.where(fitted, self.running_mean, torch.zeros_like(self.running_mean)).to(values)
        variance = torch.where(fitted, self.running_var, torch.ones_like(self.running_var)).to(values)
        scale = (variance + self.eps).sqrt()
        # Preserve exact identity for uncalibrated fields rather than introducing the eps scale factor.
        scale = torch.where(fitted.to(device=values.device), scale, torch.ones_like(scale))
        return (values - mean) / scale


class ContextEncoder(nn.Module):
    """TWO-TIER causal transformer over RAW 1-second bars, fused with the stock's as-of covariates.
      Tier 1: LOCAL causal attention within fixed `block_seconds` blocks of raw seconds -> a LEARNED summary per
              block (its most-recent-valid token). The model compresses raw seconds; nothing is hand-pooled.
      Tier 2: GLOBAL causal attention over the block summaries across the whole session -> per-stock session
              context (its most-recent-valid block). Reaches the full session at O(S*block) + O(n_blocks^2) cost.
    Cross-sectional mean over all involved stocks (bars + covariates) -> market context. PURE market state."""

    pos1: torch.Tensor
    pos2: torch.Tensor

    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.block_seconds = config.block_seconds
        self.grad_checkpoint = config.grad_checkpoint
        self.stock_chunk = config.stock_chunk
        t1 = max(1, config.n_layers // 2)
        t2 = max(1, config.n_layers - t1)
        # Fixed-stat normalizers are calibrated explicitly on training data. Forward is identical in train/eval
        # mode and cannot leak future/session peers through batch statistics. Unfitted fields are identity-mapped.
        self.bar_norm = _FixedFeatureNormalizer(config.bar_feature_dim)
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
        self.cov_norm = _FixedFeatureNormalizer(config.covariate_dim)
        self.cov_mlp = nn.Sequential(nn.Linear(config.covariate_dim, d), nn.GELU(), nn.Linear(d, d))
        self.cov_valid_proj = nn.Linear(config.covariate_dim, d, bias=False)
        self.fuse = nn.LayerNorm(d)
        self.d_model = d

    @property
    def normalization_calibrated(self) -> bool:
        """Whether every bar and covariate field has fixed training-set statistics."""
        return self.bar_norm.calibrated and self.cov_norm.calibrated

    @torch.no_grad()
    def reset_normalization(self) -> None:
        """Reset bar and covariate normalization to the safe identity default."""
        self.bar_norm.reset()
        self.cov_norm.reset()

    @torch.no_grad()
    def calibrate_normalization(
        self,
        bars: torch.Tensor,
        bar_mask: torch.Tensor,
        cov_blocks: torch.Tensor,
        cov_mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate fixed normalization moments from one TRAINING batch.

        Call repeatedly to stream the training split before optimization, then leave the statistics fixed for all
        training/evaluation forwards. ``bars``/``bar_mask`` and ``cov_blocks`` have the same layouts as
        :meth:`forward`. Covariates are calibrated only after a stock has appeared in the bar stream; optional
        ``cov_mask`` (shape ``[B,nB,A]`` or ``[B,nB,A,C]``) can additionally exclude missing/stale covariates.
        Masked and non-finite values do not contribute. Never calibrate with validation/test/future observations.
        """
        if bars.ndim != 4 or bar_mask.shape != bars.shape[:-1]:
            raise ValueError(
                f"expected bars [B,A,S,F] and bar_mask [B,A,S], got {tuple(bars.shape)}, {tuple(bar_mask.shape)}"
            )
        B, A, S, _ = bars.shape
        if cov_blocks.ndim != 4 or cov_blocks.shape[0] != B or cov_blocks.shape[2] != A:
            raise ValueError(
                f"expected cov_blocks [B,nB,A,C] matching B={B}, A={A}, got {tuple(cov_blocks.shape)}"
            )
        bl = self.block_seconds
        nB = (S + bl - 1) // bl
        if cov_blocks.shape[1] < nB:
            raise ValueError(f"cov_blocks has {cov_blocks.shape[1]} blocks, but padded bars require {nB}")

        self.bar_norm.update(bars, bar_mask)
        padded_mask = bar_mask.bool()
        if S < nB * bl:
            padded_mask = F.pad(padded_mask, (0, nB * bl - S))
        # Match forward's covariate semantics: once observed, a stock carries its last causal context through
        # later empty blocks, so those as-of covariates are part of the model input too.
        block_has = padded_mask.reshape(B, A, nB, bl).any(dim=-1)
        seen = block_has.to(torch.int8).cummax(dim=2).values.bool().permute(0, 2, 1)
        cov_values = cov_blocks[:, :nB]
        effective_mask: torch.Tensor = seen
        if cov_mask is not None:
            cov_mask = cov_mask[:, :nB].to(device=seen.device, dtype=torch.bool)
            if cov_mask.shape == cov_values.shape[:-1]:
                effective_mask = seen & cov_mask
            elif cov_mask.shape == cov_values.shape:
                effective_mask = seen.unsqueeze(-1) & cov_mask
            else:
                raise ValueError(
                    "cov_mask must have shape [B,nB,A] or [B,nB,A,C]; "
                    f"got {tuple(cov_mask.shape)} for covariates {tuple(cov_values.shape)}"
                )
        self.cov_norm.update(cov_values, effective_mask)

    def forward(
        self,
        bars: torch.Tensor,
        bar_mask: torch.Tensor,
        cov_blocks: torch.Tensor,
        cov_valid: torch.Tensor | None = None,
    ):
        """Encode a full session per (batch) day -> a context at EVERY 5-min block (causal). The decision at
        block b uses only blocks 0..b (no look-ahead). bars [B,A,S,F] RAW (session-aligned: index s = second s
        after the 09:30 open), bar_mask [B,A,S], cov_blocks [B,nB,A,C] (as-of covariates at each block).
        -> per_stock [B,nB,A,d], market [B,nB,d]. `stock_chunk>0` processes the (weight-shared) stock axis in
        chunks -- bit-identical, bounded activation memory (huge universes)."""
        B, A, S, F = bars.shape
        bl = self.block_seconds
        nB = S // bl
        if nB * bl != S:                                         # pad the session up to a whole number of blocks
            pad = (nB + 1) * bl - S
            bars = torch.nn.functional.pad(bars, (0, 0, 0, pad))
            bar_mask = torch.nn.functional.pad(bar_mask, (0, pad))
            S = bars.shape[2]
            nB = S // bl
        # Fixed-stat normalization on RAW valid bars. Statistics come only from explicit TRAINING calibration;
        # this read-only transform is identical in train/eval and independent of future/session-peer observations.
        flat = bars.reshape(-1, F)
        mv = bar_mask.reshape(-1)
        normed_flat = torch.zeros_like(flat)
        if mv.any():
            normed_flat[mv] = self.bar_norm(flat[mv])
        normed = normed_flat.reshape(B, A, S, F)
        ck = self.stock_chunk if self.stock_chunk and 0 < self.stock_chunk < A else A
        outs, seens = [], []
        for lo in range(0, A, ck):                               # per-stock weight-shared -> chunk the stock axis
            nc, mc = normed[:, lo:lo + ck], bar_mask[:, lo:lo + ck]
            if ck < A and self.grad_checkpoint and self.training and torch.is_grad_enabled():
                # chunk-level checkpoint: backward recomputes ONE chunk's tier-1/tier-2 at a time; only the small
                # normalized-bars slice is saved (the inner per-block checkpoints engage during the recompute).
                bb, sn = torch.utils.checkpoint.checkpoint(self._stock_blocks, nc, mc, use_reentrant=False)
            else:
                bb, sn = self._stock_blocks(nc, mc)
            outs.append(bb)
            seens.append(sn)
        bar_blocks = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]      # [B, nB, A, d]
        seen = torch.cat(seens, dim=2) if len(seens) > 1 else seens[0]         # [B, nB, A, 1]
        return self._fuse_market(bar_blocks, seen, cov_blocks, nB, cov_valid)

    def _stock_blocks(self, normed: torch.Tensor, bar_mask: torch.Tensor):
        """Per-stock bars path for one stock chunk: embed + tier-1 + tier-2. normed [B,Ac,S,F] (ALREADY
        bar-normalized -- fixed normalization stays outside the chunk loop), bar_mask [B,Ac,S] ->
        (bar_blocks [B,nB,Ac,d], seen [B,nB,Ac,1])."""
        B, A, S, F = normed.shape
        d = self.d_model
        bl = self.block_seconds
        nB = S // bl
        x = self.input_proj(normed.reshape(-1, F)).reshape(B * A * nB, bl, d)
        # Positional buffers are stored in FP32 for checkpoint-independent accuracy.  A plain add would promote
        # the BF16 projection produced by autocast back to FP32, after which every residual in both transformer
        # tiers (and the full [B,nB,A,d] output) would stay FP32.  Cast only the tiny positional slice to the
        # activation dtype; explicit FP32/non-AMP execution is unchanged.
        x = x + self.pos1[:bl].to(dtype=x.dtype).view(1, bl, d)
        # --- Tier 1: local causal attention within each block -> learned per-block summaries.
        # Per-block checkpointing: checkpoint each _CausalBlock independently. During forward, each block's
        # input is saved (n_blocks × [B*A*nB, bl, d] ≈ 9.6 GB for d512/8L with B=1 day). During backward,
        # only ONE block's intermediates are recomputed and held at a time (~12 GB peak), so the backward
        # peak is ~24 GB instead of ~70+ GB from a one-segment recompute of all 8 blocks simultaneously.
        bm1 = bar_mask.reshape(B * A * nB, bl)                   # per-block validity (bars sit at absolute offsets)

        def run_tier1_rows(rows: torch.Tensor, kpm: torch.Tensor | None) -> torch.Tensor:
            y = rows
            if self.grad_checkpoint and self.training:
                for blk in self.tier1:
                    if kpm is None:
                        y = torch.utils.checkpoint.checkpoint(lambda z, b=blk: b(z, None), y,
                                                              use_reentrant=False)
                    else:
                        y = torch.utils.checkpoint.checkpoint(blk, y, kpm, use_reentrant=False)
            else:
                for blk in self.tier1:
                    y = blk(y, kpm)
            return y

        # Pack rows by their valid-token count. Absolute positional encodings were added before packing, so gaps
        # remain observable, while every attention call can use native causal SDPA without a materialized
        # [N,1,S,S] padding mask. We only need the final valid token from tier 1; avoiding a scatter back into the
        # full padded grid also saves an S-sized activation. Actual second data is sparse enough that this removes
        # multi-GiB masks in production daily-raw batches.
        counts = bm1.sum(-1)
        summaries = torch.zeros(x.shape[0], d, dtype=x.dtype, device=x.device)
        for length in counts.unique(sorted=True).tolist():
            length = int(length)
            if length <= 0:
                continue
            rows = counts == length
            row_x, row_mask = x[rows], bm1[rows]
            if length == bl:
                packed = row_x
            else:
                positions = torch.arange(bl, device=x.device).expand(row_x.shape[0], bl)[row_mask]
                positions = positions.reshape(row_x.shape[0], length)
                packed = torch.gather(row_x, 1, positions.unsqueeze(-1).expand(-1, -1, d))
            encoded = run_tier1_rows(packed, None)
            # CUDA autocast LayerNorm returns FP32; indexed assignment does not implicitly narrow its source.
            summaries[rows] = self.norm1(encoded[:, -1]).to(dtype=summaries.dtype)
        summ = summaries.reshape(B * A, nB, d)                    # true last-valid, gap-position-aware token
        block_has = bm1.any(-1).reshape(B * A, nB)               # [B*A, nB]
        summ = summ * block_has.unsqueeze(-1).to(dtype=summ.dtype)
        # --- Tier 2: global causal attention over block summaries -> a context at EVERY block ---
        h = summ + self.pos2[:nB].to(dtype=summ.dtype).unsqueeze(0)
        for blk in self.tier2:
            h = blk(h, ~block_has)
        # CUDA autocast executes standalone LayerNorm in FP32.  Its accuracy is useful, but retaining that output
        # would widen the full stock/block context; quantize the normalized result back to the residual dtype.
        h = self.norm2(h).to(dtype=h.dtype)                       # [B*A, nB, d] per-block tier-2 context
        bar_blocks = h.reshape(B, A, nB, d).permute(0, 2, 1, 3)  # [B, nB, A, d]
        # Once a stock has traded, later no-trade blocks still get the causal stale context emitted at that
        # timestamp. Empty blocks are not attention keys above, so they cannot become synthetic history.
        seen_blocks = block_has.to(torch.int8).cummax(dim=1).values.bool()
        seen = seen_blocks.reshape(B, A, nB).permute(0, 2, 1).unsqueeze(-1).to(dtype=bar_blocks.dtype)
        return bar_blocks, seen

    def _fuse_market(
        self,
        bar_blocks: torch.Tensor,
        seen: torch.Tensor,
        cov_blocks: torch.Tensor,
        nB: int,
        cov_valid: torch.Tensor | None = None,
    ):
        """Fuse the (full cross-section) bar contexts with as-of covariates and pool the market context. Runs on
        the FULL stock axis (cheap: [B*nB*A, C] rows), so the cross-sectional mean is chunk-invariant.
        bar_blocks [B,nB,A,d], seen [B,nB,A,1], cov_blocks [B,>=nB,A,C]."""
        B, _, A, d = bar_blocks.shape
        has = seen                                                     # [B, nB, A, 1]
        cf = cov_blocks[:, :nB].reshape(-1, cov_blocks.shape[-1])   # [B*nB*A, C]
        cm = has.reshape(-1) > 0                                # normalize only PRESENT-stock rows (mirror bars)
        cov_flat = torch.zeros_like(cf)                         # keep absent-stock rows at the neutral zero input
        if cov_valid is None:
            cv = cm[:, None].expand_as(cf)
        else:
            cv = cov_valid[:, :nB].bool()
            if cv.shape == cov_blocks[:, :nB].shape[:-1]:
                cv = cv.unsqueeze(-1).expand_as(cov_blocks[:, :nB])
            if cv.shape != cov_blocks[:, :nB].shape:
                raise ValueError(
                    f"cov_valid must match cov_blocks or omit its feature axis; got {tuple(cv.shape)}"
                )
            cv = cv.reshape_as(cf) & cm[:, None]
        if cm.any():
            normalized = self.cov_norm(cf[cm])
            cov_flat[cm] = torch.where(cv[cm], normalized, torch.zeros_like(normalized))
        cov_embed = self.cov_mlp(cov_flat) + self.cov_valid_proj(cv.to(cov_flat.dtype))
        cov = cov_embed.reshape(B, nB, A, d)                         # values + explicit validity, per block
        fused = self.fuse(bar_blocks + cov).to(dtype=bar_blocks.dtype)
        per_stock = fused * has                                  # fuse bars + as-of covariates, per block
        # Accumulate the wide TOP2000 cross-section in FP32 without materializing an FP32 copy of per_stock, then
        # return to the activation dtype.  This preserves BF16 storage/throughput while avoiding a low-precision
        # cancellation-prone reduction over ~2,000 names.
        market = (
            per_stock.sum(dim=2, dtype=torch.float32)
            / has.sum(dim=2, dtype=torch.float32).clamp_min(1.0)
        ).to(dtype=per_stock.dtype)
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
