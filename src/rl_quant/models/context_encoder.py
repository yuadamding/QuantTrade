"""Stage-1 CONTEXT LEARNING model: a causal-attention transformer over session-open-rolling chunk tokens.

This module is deliberately PURE market-context: it contains no action/policy concept whatsoever (no previous
position, no constraints, no per-action scoring). That is the enforced split -- the context encoder learns
"what is the market doing", trained self-supervised and then frozen; the decision policy (a separate module)
learns "what to do about it" on top of the frozen context. See rl_quant.models.decision_policy.

Input is a per-stock sequence of CHUNK TOKENS: each token pools the raw 1-second bars of a fixed sub-window
(organized at train time from raw data -- nothing precomputed is stored). Tokens are LEFT-aligned (the day's
first chunk at position 0) and a causal attention mask makes each position attend only to itself and earlier
chunks, so the representation rolls forward from the session open with no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ContextEncoderConfig:
    chunk_feature_dim: int           # features per chunk token (set by the train-time organizer)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    max_chunks: int = 80             # longest token sequence (~ a full RTH session at the chosen chunk size)


class ContextEncoder(nn.Module):
    """Per-stock causal transformer over chunk tokens -> per-stock context; cross-sectional mean -> market
    context. PURE market state: no policy inputs, by design (the context/policy split)."""

    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.input_proj = nn.Linear(config.chunk_feature_dim, d)
        self.pos = nn.Parameter(torch.zeros(config.max_chunks, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.n_heads, dim_feedforward=config.feedforward_dim,
            dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.d_model = d

    def forward(self, chunk: torch.Tensor, chunk_mask: torch.Tensor):
        """chunk [B,A,C,Fc] (left-aligned), chunk_mask [B,A,C] bool -> per_stock [B,A,d], market [B,d]."""
        B, A, C, _ = chunk.shape
        x = self.input_proj(chunk.reshape(B * A, C, -1)) + self.pos[:C].unsqueeze(0)
        causal = torch.triu(torch.ones(C, C, dtype=torch.bool, device=chunk.device), diagonal=1)  # True=block future
        kpm = ~chunk_mask.reshape(B * A, C)            # True = padding key (masked)
        all_pad = kpm.all(dim=1)
        kpm = kpm.clone()
        kpm[all_pad] = False                           # unmask fully-empty rows -> avoids softmax-over-(-inf) NaN
        out = self.encoder(x, mask=causal, src_key_padding_mask=kpm)
        out = self.norm(out).reshape(B, A, C, self.d_model)
        count = chunk_mask.reshape(B * A, C).sum(dim=1)            # valid tokens per (stock, decision)
        last = (count - 1).clamp_min(0).long().reshape(B, A)      # index of the most recent valid chunk
        per_stock = torch.gather(out, 2, last[..., None, None].expand(B, A, 1, self.d_model)).squeeze(2)
        has = (count.reshape(B, A) > 0).float().unsqueeze(-1)     # 0 for stocks absent at this decision
        per_stock = per_stock * has
        market = per_stock.sum(dim=1) / has.sum(dim=1).clamp_min(1.0)
        return per_stock, market


class ContextForwardHead(nn.Module):
    """Self-supervised pretext head: from the market context predict the next-interval [equal-weight market
    return, realized vol]. Trained jointly with the encoder in Stage 1, then discarded (the encoder is frozen).
    Uses only market state -> keeps the context objective policy-free."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        return self.net(market)
