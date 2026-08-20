"""Ordered five-minute, cross-day, and low-rank market alpha encoders.

The modules consume raw interval fields plus explicit validity masks.  They do
not calculate technical indicators or batch-derived normalization statistics.
All temporal attention is causal, while cross-sectional context is mediated by
a fixed number of market latents so complexity is linear in the stock count.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class AlphaModelError(ValueError):
    """The ordered alpha representation received a malformed input."""


def _sinusoidal(length: int, dimension: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisors = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    result = torch.zeros(length, dimension, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * divisors)
    result[:, 1::2] = torch.cos(
        positions * divisors[: result[:, 1::2].shape[1]]
    )
    return result


def _last_valid(sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(sequence.shape[1], device=sequence.device)
    indices = torch.where(
        valid,
        positions.unsqueeze(0),
        torch.full_like(valid, -1, dtype=torch.long),
    ).amax(dim=1)
    if bool((indices < 0).any()):
        raise AlphaModelError("every encoded sequence needs at least one valid token")
    return torch.gather(
        sequence,
        1,
        indices[:, None, None].expand(-1, 1, sequence.shape[-1]),
    ).squeeze(1)


@dataclass(frozen=True, slots=True)
class OrderedFiveMinuteConfig:
    feature_dimension: int = 5
    token_dimension: int = 128
    layers: int = 3
    heads: int = 4
    feedforward_dimension: int = 512
    dropout: float = 0.05
    intervals_per_session: int = 78

    def validate(self) -> None:
        for name in (
            "feature_dimension",
            "token_dimension",
            "layers",
            "heads",
            "feedforward_dimension",
            "intervals_per_session",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AlphaModelError(f"{name} must be a positive integer")
        if self.token_dimension % self.heads != 0:
            raise AlphaModelError("intraday token dimension must divide across heads")
        if not 0.0 <= self.dropout < 1.0:
            raise AlphaModelError("intraday dropout is outside [0, 1)")


class OrderedFiveMinuteEncoder(nn.Module):
    """Encode ordered raw intervals for independent stock-days."""

    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    positions: torch.Tensor

    def __init__(self, config: OrderedFiveMinuteConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(
            config.feature_dimension + 1,
            config.token_dimension,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dimension,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.token_dimension)
        self.register_buffer(
            "positions",
            _sinusoidal(config.intervals_per_session, config.token_dimension),
            persistent=False,
        )
        self.register_buffer(
            "feature_mean",
            torch.zeros(config.feature_dimension, dtype=torch.float64),
        )
        self.register_buffer(
            "feature_scale",
            torch.ones(config.feature_dimension, dtype=torch.float64),
        )

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        """Set moments computed only from the training partition."""

        expected = (self.config.feature_dimension,)
        if (
            tuple(mean.shape) != expected
            or tuple(scale.shape) != expected
            or not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0.0).any())
        ):
            raise AlphaModelError("intraday normalization moments are invalid")
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_scale.copy_(scale.to(self.feature_scale))

    def forward_sequence(
        self,
        raw_intervals: torch.Tensor,
        valid_intervals: torch.Tensor,
    ) -> torch.Tensor:
        """Return one causal token per interval.

        Shapes are ``[stock_day, interval, raw_feature]`` and
        ``[stock_day, interval]``.  Invalid interval payloads are ignored and
        may contain NaNs; valid payloads must be finite.
        """

        if (
            not isinstance(raw_intervals, torch.Tensor)
            or raw_intervals.ndim != 3
            or raw_intervals.shape[-1] != self.config.feature_dimension
            or raw_intervals.shape[1] > self.config.intervals_per_session
            or not raw_intervals.is_floating_point()
            or not isinstance(valid_intervals, torch.Tensor)
            or valid_intervals.dtype != torch.bool
            or tuple(valid_intervals.shape) != tuple(raw_intervals.shape[:2])
            or valid_intervals.device != raw_intervals.device
            or bool((valid_intervals.sum(dim=1) == 0).any())
            or not bool(torch.isfinite(raw_intervals[valid_intervals]).all())
        ):
            raise AlphaModelError("ordered intraday inputs are malformed")
        mean = self.feature_mean.to(raw_intervals)
        scale = self.feature_scale.to(raw_intervals)
        cleaned = torch.where(
            valid_intervals.unsqueeze(-1),
            raw_intervals,
            mean.view(1, 1, -1),
        )
        normalized = (cleaned - mean) / scale
        missing_channel = (~valid_intervals).to(raw_intervals.dtype).unsqueeze(-1)
        tokens = self.input_projection(torch.cat((normalized, missing_channel), dim=-1))
        tokens = tokens + self.positions[: tokens.shape[1]].to(tokens)
        causal_mask = torch.ones(
            tokens.shape[1],
            tokens.shape[1],
            dtype=torch.bool,
            device=tokens.device,
        ).triu(diagonal=1)
        encoded = self.encoder(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=~valid_intervals,
        )
        encoded = self.output_norm(encoded)
        return torch.where(valid_intervals.unsqueeze(-1), encoded, torch.zeros_like(encoded))

    def forward(
        self,
        raw_intervals: torch.Tensor,
        valid_intervals: torch.Tensor,
    ) -> torch.Tensor:
        return _last_valid(
            self.forward_sequence(raw_intervals, valid_intervals),
            valid_intervals,
        )


@dataclass(frozen=True, slots=True)
class CrossDayAlphaConfig:
    token_dimension: int = 128
    layers: int = 4
    heads: int = 4
    feedforward_dimension: int = 512
    dropout: float = 0.05
    context_sessions: int = 252

    def validate(self) -> None:
        for name in (
            "token_dimension",
            "layers",
            "heads",
            "feedforward_dimension",
            "context_sessions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AlphaModelError(f"{name} must be a positive integer")
        if self.token_dimension % self.heads != 0:
            raise AlphaModelError("cross-day token dimension must divide across heads")
        if not 0.0 <= self.dropout < 1.0:
            raise AlphaModelError("cross-day dropout is outside [0, 1)")


class CrossDayAlphaEncoder(nn.Module):
    """Causally encode stock-day tokens across sessions."""

    positions: torch.Tensor

    def __init__(self, config: CrossDayAlphaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dimension,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.token_dimension)
        self.register_buffer(
            "positions",
            _sinusoidal(config.context_sessions, config.token_dimension),
            persistent=False,
        )

    def forward_sequence(
        self,
        day_tokens: torch.Tensor,
        valid_days: torch.Tensor,
    ) -> torch.Tensor:
        if (
            day_tokens.ndim != 3
            or day_tokens.shape[-1] != self.config.token_dimension
            or day_tokens.shape[1] > self.config.context_sessions
            or not day_tokens.is_floating_point()
            or valid_days.dtype != torch.bool
            or tuple(valid_days.shape) != tuple(day_tokens.shape[:2])
            or valid_days.device != day_tokens.device
            or bool((valid_days.sum(dim=1) == 0).any())
            or not bool(torch.isfinite(day_tokens[valid_days]).all())
        ):
            raise AlphaModelError("cross-day inputs are malformed")
        cleaned = torch.where(valid_days.unsqueeze(-1), day_tokens, torch.zeros_like(day_tokens))
        tokens = cleaned + self.positions[: cleaned.shape[1]].to(cleaned)
        causal_mask = torch.ones(
            tokens.shape[1],
            tokens.shape[1],
            dtype=torch.bool,
            device=tokens.device,
        ).triu(diagonal=1)
        encoded = self.encoder(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=~valid_days,
        )
        encoded = self.output_norm(encoded)
        return torch.where(valid_days.unsqueeze(-1), encoded, torch.zeros_like(encoded))

    def forward(self, day_tokens: torch.Tensor, valid_days: torch.Tensor) -> torch.Tensor:
        return _last_valid(self.forward_sequence(day_tokens, valid_days), valid_days)


@dataclass(frozen=True, slots=True)
class MarketLatentConfig:
    token_dimension: int = 128
    latent_count: int = 32
    heads: int = 4
    dropout: float = 0.05

    def validate(self) -> None:
        if (
            self.token_dimension <= 0
            or self.latent_count <= 0
            or self.heads <= 0
            or self.token_dimension % self.heads != 0
            or not 0.0 <= self.dropout < 1.0
        ):
            raise AlphaModelError("market latent configuration is invalid")


class MarketLatentEncoder(nn.Module):
    """Exchange market context through K latents instead of A-squared attention."""

    def __init__(self, config: MarketLatentConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.market_latents = nn.Parameter(
            torch.empty(config.latent_count, config.token_dimension)
        )
        nn.init.normal_(self.market_latents, mean=0.0, std=0.02)
        self.latent_attention = nn.MultiheadAttention(
            config.token_dimension,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.stock_attention = nn.MultiheadAttention(
            config.token_dimension,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.latent_norm = nn.LayerNorm(config.token_dimension)
        self.stock_norm = nn.LayerNorm(config.token_dimension)

    def forward(
        self,
        stock_tokens: torch.Tensor,
        valid_stocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            stock_tokens.ndim != 3
            or stock_tokens.shape[-1] != self.config.token_dimension
            or valid_stocks.dtype != torch.bool
            or tuple(valid_stocks.shape) != tuple(stock_tokens.shape[:2])
            or valid_stocks.device != stock_tokens.device
            or bool((valid_stocks.sum(dim=1) == 0).any())
            or not bool(torch.isfinite(stock_tokens[valid_stocks]).all())
        ):
            raise AlphaModelError("market-latent inputs are malformed")
        cleaned = torch.where(
            valid_stocks.unsqueeze(-1), stock_tokens, torch.zeros_like(stock_tokens)
        )
        latents = self.market_latents.to(cleaned).unsqueeze(0).expand(
            cleaned.shape[0], -1, -1
        )
        latent_update, _ = self.latent_attention(
            latents,
            cleaned,
            cleaned,
            key_padding_mask=~valid_stocks,
            need_weights=False,
        )
        latents = self.latent_norm(latents + latent_update)
        stock_update, _ = self.stock_attention(
            cleaned,
            latents,
            latents,
            need_weights=False,
        )
        stocks = self.stock_norm(cleaned + stock_update)
        stocks = torch.where(valid_stocks.unsqueeze(-1), stocks, torch.zeros_like(stocks))
        return stocks, latents


class AlphaDistribution(NamedTuple):
    mean: torch.Tensor
    downside_quantile: torch.Tensor
    median: torch.Tensor
    upside_quantile: torch.Tensor
    scale: torch.Tensor


class AlphaDistributionHead(nn.Module):
    """Per-stock ordered quantiles and positive aleatoric scale."""

    def __init__(self, token_dimension: int, horizons: int) -> None:
        super().__init__()
        if token_dimension <= 0 or horizons <= 0:
            raise AlphaModelError("alpha distribution head dimensions must be positive")
        self.horizons = horizons
        self.projection = nn.Linear(token_dimension, 5 * horizons)

    def forward(self, stock_tokens: torch.Tensor) -> AlphaDistribution:
        if stock_tokens.ndim < 2 or not bool(torch.isfinite(stock_tokens).all()):
            raise AlphaModelError("alpha distribution head input is malformed")
        raw = self.projection(stock_tokens)
        mean, median, lower_gap, upper_gap, raw_scale = raw.split(
            self.horizons, dim=-1
        )
        return AlphaDistribution(
            mean=mean,
            downside_quantile=median - F.softplus(lower_gap),
            median=median,
            upside_quantile=median + F.softplus(upper_gap),
            scale=F.softplus(raw_scale) + 1e-6,
        )


class HierarchicalAlphaOutput(NamedTuple):
    distribution: AlphaDistribution
    stock_context: torch.Tensor
    market_context: torch.Tensor


class HierarchicalAlphaModel(nn.Module):
    """Compose raw intervals, cross-day memory, and market latents."""

    def __init__(
        self,
        intraday: OrderedFiveMinuteConfig,
        cross_day: CrossDayAlphaConfig,
        market: MarketLatentConfig,
        *,
        horizons: int,
    ) -> None:
        super().__init__()
        if not (
            intraday.token_dimension
            == cross_day.token_dimension
            == market.token_dimension
        ):
            raise AlphaModelError("hierarchical alpha token dimensions disagree")
        self.intraday = OrderedFiveMinuteEncoder(intraday)
        self.cross_day = CrossDayAlphaEncoder(cross_day)
        self.market = MarketLatentEncoder(market)
        self.head = AlphaDistributionHead(market.token_dimension, horizons)

    def forward(
        self,
        raw_intervals: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_stock_days: torch.Tensor,
        valid_stocks: torch.Tensor,
    ) -> HierarchicalAlphaOutput:
        """Encode ``[batch, day, stock, interval, feature]`` raw observations."""

        if (
            raw_intervals.ndim != 5
            or valid_intervals.shape != raw_intervals.shape[:-1]
            or valid_stock_days.shape != raw_intervals.shape[:3]
            or valid_stocks.shape != (raw_intervals.shape[0], raw_intervals.shape[2])
            or valid_intervals.dtype != torch.bool
            or valid_stock_days.dtype != torch.bool
            or valid_stocks.dtype != torch.bool
        ):
            raise AlphaModelError("hierarchical alpha input shapes disagree")
        batch, days, assets, intervals, features = raw_intervals.shape
        flattened = raw_intervals.reshape(batch * days * assets, intervals, features)
        flattened_valid = valid_intervals.reshape(batch * days * assets, intervals)
        expected_stock_days = valid_stock_days.reshape(-1)
        if not torch.equal(flattened_valid.any(dim=1), expected_stock_days):
            raise AlphaModelError("stock-day validity disagrees with interval validity")
        # The standalone encoders reject all-missing rows.  Encode only valid
        # stock-days and scatter zeros for structurally absent observations.
        day_tokens = torch.zeros(
            batch * days * assets,
            self.intraday.config.token_dimension,
            dtype=raw_intervals.dtype,
            device=raw_intervals.device,
        )
        selected_days = torch.nonzero(expected_stock_days, as_tuple=False).flatten()
        if selected_days.numel() == 0:
            raise AlphaModelError("hierarchical batch has no valid stock-day")
        day_tokens[selected_days] = self.intraday(
            flattened.index_select(0, selected_days),
            flattened_valid.index_select(0, selected_days),
        )
        by_stock = day_tokens.reshape(batch, days, assets, -1).permute(0, 2, 1, 3)
        by_stock = by_stock.reshape(batch * assets, days, -1)
        day_mask = valid_stock_days.permute(0, 2, 1).reshape(batch * assets, days)
        expected_stocks = valid_stocks.reshape(-1)
        if not torch.equal(day_mask.any(dim=1), expected_stocks):
            raise AlphaModelError("stock validity disagrees with stock-day validity")
        stock_tokens = torch.zeros(
            batch * assets,
            self.cross_day.config.token_dimension,
            dtype=raw_intervals.dtype,
            device=raw_intervals.device,
        )
        selected_stocks = torch.nonzero(expected_stocks, as_tuple=False).flatten()
        stock_tokens[selected_stocks] = self.cross_day(
            by_stock.index_select(0, selected_stocks),
            day_mask.index_select(0, selected_stocks),
        )
        stock_tokens = stock_tokens.reshape(batch, assets, -1)
        stock_context, market_context = self.market(stock_tokens, valid_stocks)
        return HierarchicalAlphaOutput(
            distribution=self.head(stock_context),
            stock_context=stock_context,
            market_context=market_context,
        )


__all__ = [
    "AlphaDistribution",
    "AlphaDistributionHead",
    "AlphaModelError",
    "CrossDayAlphaConfig",
    "CrossDayAlphaEncoder",
    "HierarchicalAlphaModel",
    "HierarchicalAlphaOutput",
    "MarketLatentConfig",
    "MarketLatentEncoder",
    "OrderedFiveMinuteConfig",
    "OrderedFiveMinuteEncoder",
]
