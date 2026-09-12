"""Raw seconds -> trainable causal attention -> direct equity/CASH policy.

No normalizer, engineered covariate, forecast or embedding-cache interface.
Hidden-state LayerNorm operates only AFTER the learned raw input projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondCatalog, RawSecondObservation, SCHEMA
from rl_quant.rl.ppo import MaskedDirichlet, PPOActorCritic, PPOModelOutput


@dataclass(frozen=True)
class RawSecondModelConfig:
    d_model: int = 128
    attention_heads: int = 4
    local_layers: int = 2
    temporal_layers: int = 2
    local_block_seconds: int = 300
    max_context_seconds: int = 23_400
    asset_chunk_size: int = 4
    local_batch_blocks: int = 32
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name != "activation_checkpointing" and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if (self.d_model % self.attention_heads or self.max_context_seconds > 23_400
                or self.local_block_seconds > self.max_context_seconds
                or type(self.activation_checkpointing) is not bool):
            raise ValueError("Invalid bounded raw-second model configuration")


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor, valid: torch.Tensor, *, causal: bool) -> torch.Tensor:
        # Fully padded blocks need a harmless dummy key to avoid softmax(-inf).
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(dim=1)
        # For left padding, an early padded query otherwise has no causal key.
        # Give padded queries access to all safe keys, then zero their outputs.
        blocked = torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device).triu(1) if causal else None
        if blocked is not None:
            allowed = (~blocked)[None] & safe[:, None, :]
            allowed = torch.where(valid[:, :, None], allowed, safe[:, None, :])
            blocked = (~allowed).repeat_interleave(self.attention.num_heads, dim=0)
        h = self.norm1(x)
        # Distinct value view disables MHA's inference-only native fast path;
        # collection and PPO recomputation then use the same forward kernels.
        update, _ = self.attention(h, h, h.clone(), attn_mask=blocked,
                                   key_padding_mask=None if causal else ~safe, need_weights=False)
        x = x + update
        x = x + self.ff(self.norm2(x))
        return torch.where(valid[..., None], x, torch.zeros_like(x))


class _Readout(nn.Module):
    """Learned attention over already encoded tokens, never over raw OHLCV."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(dim=1)
        value, _ = self.attention(self.query.expand(x.shape[0], -1, -1), self.norm(x), self.norm(x),
                                  key_padding_mask=~safe, need_weights=False)
        return torch.where(valid.any(dim=1, keepdim=True), value[:, 0], torch.zeros_like(value[:, 0]))


class RawSecondActorCritic(PPOActorCritic):
    """Bounded raw-context recomputation on every PPO forward.

    CASH is action index zero, followed by the catalog's fixed issue IDs.
    Account state is cash and share holdings, not market statistics.
    Input/hidden computation initially stays FP32; reduced precision is rejected.
    """

    def __init__(self, catalog: RawSecondCatalog, config: RawSecondModelConfig | None = None):
        super().__init__()
        self._frozen_evaluation = False
        self.catalog = catalog
        self.config = RawSecondModelConfig() if config is None else config
        c = self.config
        n = len(catalog.asset_ids)
        self.input_projection = nn.Linear(5, c.d_model)
        self.input_hidden_norm = nn.LayerNorm(c.d_model)
        self.position = nn.Embedding(c.max_context_seconds, c.d_model)
        self.instrument = nn.Embedding(n, c.d_model)
        self.missing_observation = nn.Parameter(torch.randn(c.d_model) * 0.02)
        self.local = nn.ModuleList([_Attention(c.d_model, c.attention_heads) for _ in range(c.local_layers)])
        self.block_readout = _Readout(c.d_model, c.attention_heads)
        self.temporal = nn.ModuleList([_Attention(c.d_model, c.attention_heads) for _ in range(c.temporal_layers)])
        self.temporal_readout = _Readout(c.d_model, c.attention_heads)
        self.cross_stock = _Attention(c.d_model, c.attention_heads)
        self.market_readout = _Readout(c.d_model, c.attention_heads)
        self.account_projection = nn.Sequential(nn.Linear(n + 1, c.d_model), nn.LayerNorm(c.d_model), nn.GELU())
        self.actor = nn.Sequential(nn.Linear(2 * c.d_model, c.d_model), nn.GELU(), nn.Linear(c.d_model, 1))
        self.cash_actor = nn.Linear(2 * c.d_model, 1)
        self.critic = nn.Sequential(nn.Linear(2 * c.d_model, c.d_model), nn.GELU(), nn.Linear(c.d_model, 1))

    def get_extra_state(self) -> dict:
        return dict(schema=SCHEMA, catalog=self.catalog.identity, model=asdict(self.config),
                    asset_ids=self.catalog.asset_ids, market_input="unscaled-provider-ohlcv-fp32")

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("Checkpoint input schema, configuration or raw catalog differs")

    def policy_contract(self) -> dict:
        """Portable inference compatibility; dates remain training provenance."""
        return {k: v for k, v in self.get_extra_state().items() if k != "catalog"} | {
            "input_contract": asdict(self.catalog.windows[0].contract),
        }

    def freeze_for_evaluation(self) -> None:
        self.eval()
        self.requires_grad_(False)
        self._frozen_evaluation = True

    def train(self, mode: bool = True):
        if mode and self._frozen_evaluation:
            raise ValueError("Frozen evaluation policy cannot enter training mode")
        return super().train(mode)

    def requires_grad_(self, requires_grad: bool = True):
        if requires_grad and self._frozen_evaluation:
            raise ValueError("Frozen evaluation policy cannot enable gradients")
        return super().requires_grad_(requires_grad)

    def _asset_chunk(self, raw: torch.Tensor, observed: torch.Tensor, valid: torch.Tensor,
                     asset_indices: torch.Tensor) -> torch.Tensor:
        c = self.config
        b, a, s, _ = raw.shape
        # Only masked payloads are replaced. Observed values arrive UNCHANGED at
        # this trainable Linear; no log, ratio, centering or feature fusion.
        safe_raw = torch.where(observed[..., None], raw, torch.zeros_like(raw))
        with torch.autocast(device_type=raw.device.type, enabled=False):
            x = self.input_projection(safe_raw)
            if not bool(torch.isfinite(x).all()):
                raise ValueError("FP32 projection overflow; do not restore a raw scaler")
            x = self.input_hidden_norm(x)
        x = torch.where(observed[..., None], x, self.missing_observation)
        x = x + self.position(torch.arange(s, device=raw.device))[None, None]
        x = x + self.instrument(asset_indices)[None, :, None]
        length = c.local_block_seconds
        padding = (-s) % length
        x = F.pad(x, (0, 0, 0, padding))
        valid = F.pad(valid, (0, padding), value=False)
        blocks = (s + padding) // length
        x = x.reshape(b * a * blocks, length, c.d_model)
        valid = valid.reshape(b * a * blocks, length)
        compressed = []
        for start in range(0, x.shape[0], c.local_batch_blocks):
            h, mask = x[start:start + c.local_batch_blocks], valid[start:start + c.local_batch_blocks]
            for layer in self.local:
                h = layer(h, mask, causal=True)
            compressed.append(self.block_readout(h, mask))
        z = torch.cat(compressed).reshape(b * a, blocks, c.d_model)
        block_valid = valid.reshape(b * a, blocks, length).any(dim=-1)
        for layer in self.temporal:
            z = layer(z, block_valid, causal=True)
        return self.temporal_readout(z, block_valid).reshape(b, a, c.d_model)

    def encode_market(self, observation: RawSecondObservation) -> torch.Tensor:
        observation.validate()
        if observation.asset_ids != self.catalog.asset_ids:
            raise ValueError("Raw equity identity/order differs")
        if observation.raw_ohlcv.shape[2] > self.config.max_context_seconds:
            raise ValueError("Raw context exceeds model bound; no resampling fallback")
        if any(p.dtype != torch.float32 or p.requires_grad == self._frozen_evaluation for p in self.parameters()):
            raise ValueError("Raw-second FP32 train/frozen parameter contract differs")
        raw = observation.raw_ohlcv
        chunks = []
        for start in range(0, raw.shape[1], self.config.asset_chunk_size):
            end = min(start + self.config.asset_chunk_size, raw.shape[1])
            args = (raw[:, start:end], observation.observed_mask[:, start:end],
                    ~observation.padding_mask[:, start:end], torch.arange(start, end, device=raw.device))
            if self.config.activation_checkpointing and torch.is_grad_enabled():
                chunks.append(checkpoint(self._asset_chunk, *args, use_reentrant=False))
            else:
                chunks.append(self._asset_chunk(*args))
        market = torch.cat(chunks, dim=1)
        return self.cross_stock(market, torch.ones(market.shape[:2], dtype=torch.bool, device=market.device), causal=False)

    def forward_raw(self, observation: RawSecondObservation, account_state: torch.Tensor,
                    action_mask: torch.Tensor) -> PPOModelOutput:
        with torch.autocast(device_type=observation.raw_ohlcv.device.type, enabled=False):
            market = self.encode_market(observation)
            b, n, _ = market.shape
            if (account_state.shape != (b, n + 1) or account_state.dtype != torch.float32
                    or account_state.device != market.device or not bool(torch.isfinite(account_state).all())
                    or bool((account_state < 0).any())):
                raise ValueError("Account branch requires finite nonnegative cash/share holdings only")
            if action_mask.shape != (b, n + 1) or action_mask.dtype != torch.bool or not bool(action_mask[:, 0].all()):
                raise ValueError("Direct allocation mask must include CASH at index zero")
            account = self.account_projection(account_state)
            pooled = self.market_readout(market, torch.ones((b, n), dtype=torch.bool, device=market.device))
            joint = torch.cat((pooled, account), dim=-1)
            logits = torch.cat((self.cash_actor(joint), self.actor(torch.cat((market, account[:, None].expand_as(market)), dim=-1)).squeeze(-1)), dim=-1)
            return PPOModelOutput(MaskedDirichlet(F.softplus(logits) + 0.1, action_mask), self.critic(joint).squeeze(-1), {})

    def forward(self, observations: Mapping[str, torch.Tensor], *, action_mask=None,
                recurrent_state=None, episode_start=None, valid_mask=None, burn_in=0) -> PPOModelOutput:
        del episode_start
        if set(observations) != {"raw_window_index", "raw_catalog_sha256", "account_state"}:
            raise ValueError("Raw references/account only; engineered inputs or stale embeddings rejected")
        if recurrent_state or burn_in:
            raise ValueError("No recurrent/KV/embedding cache in bounded recomputation mode")
        indices = observations["raw_window_index"]
        account = observations["account_state"]
        identities = observations["raw_catalog_sha256"]
        leading = indices.shape[:-1]
        if indices.dtype != torch.int64 or indices.shape[-1:] != (1,) or len(leading) not in (1, 2):
            raise ValueError("Raw references require int64 [..., 1]")
        if identities.shape != (*leading, 32) or identities.dtype != torch.uint8:
            raise ValueError("Trajectory lacks its immutable raw catalog identity")
        if account.shape != (*leading, len(self.catalog.asset_ids) + 1) or action_mask is None:
            raise ValueError("Account/action shape differs")
        flat_valid = torch.ones(indices.numel(), dtype=torch.bool, device=indices.device) if valid_mask is None else valid_mask.reshape(-1)
        if flat_valid.numel() != indices.numel() or flat_valid.dtype != torch.bool:
            raise ValueError("Invalid PPO padding mask")
        expected = self.catalog.identity_tensor(device=indices.device)
        if not bool((identities.reshape(-1, 32)[flat_valid] == expected).all()):
            raise ValueError("Trajectory raw catalog changed")
        rows, values = [], []
        flat_account = account.reshape(-1, account.shape[-1])
        masks = action_mask.reshape_as(flat_account)
        for row, index in enumerate(indices.reshape(-1).tolist()):
            if not bool(flat_valid[row]):
                rows.append(torch.ones_like(flat_account[row]))
                values.append(torch.zeros((), device=indices.device))
                continue
            raw = self.catalog.load(index, device=indices.device)
            result = self.forward_raw(raw, flat_account[row:row + 1], masks[row:row + 1])
            rows.append(result.distribution.concentration[0])
            values.append(result.value[0])
        return PPOModelOutput(MaskedDirichlet(torch.stack(rows).reshape_as(account), action_mask),
                              torch.stack(values).reshape(leading), {})
