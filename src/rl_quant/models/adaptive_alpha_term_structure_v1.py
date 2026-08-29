"""Causal seven-bucket alpha term-structure model for adaptive-profit research.

The model is deliberately separate from the frozen Massive P0 tabular model.
It consumes only decision-time histories, carries explicit validity and
staleness channels, exchanges cross-sectional information through a fixed
number of market latents, and routes source-qualified temporal experts into
each non-overlapping return bucket.  It contains no portfolio-age or duration
state and grants no training, evaluation, profitability, lockbox, or RL
authority by itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from rl_quant.models.alpha_hierarchical import (
    AlphaDistribution,
    MarketLatentConfig,
    MarketLatentEncoder,
    OrderedFiveMinuteConfig,
    OrderedFiveMinuteEncoder,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SCHEMA = (
    "rl-quant.massive-adaptive-alpha-term-structure-model-v1"
)
MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_ALPHA_BARS_DIMENSION_V1 = 19
MASSIVE_ADAPTIVE_ALPHA_TAPE_DIMENSION_V1 = 15
MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1 = (
    "bars-fast",
    "bars-slow",
    "tape-fast",
    "tape-slow",
    "intraday-path",
    "source-status",
)


class MassiveAdaptiveAlphaModelV1Error(ValueError):
    """Adaptive model configuration or causal inputs are malformed."""


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveAdaptiveAlphaModelV1Error(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAlphaModelSpecV1:
    """Architecture identity for the canonical AD11 engineering model."""

    setting_id: str = "AD11"
    bars_dimension: int = MASSIVE_ADAPTIVE_ALPHA_BARS_DIMENSION_V1
    tape_dimension: int = MASSIVE_ADAPTIVE_ALPHA_TAPE_DIMENSION_V1
    intraday_feature_dimension: int = 5
    token_dimension: int = 128
    fast_window_sessions: int = 10
    maximum_context_sessions: int = 504
    maximum_intraday_intervals: int = 78
    market_latent_count: int = 32
    attention_heads: int = 4
    dropout_probability: float = 0.05
    bucket_count: int = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
    expert_ids: tuple[str, ...] = MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    implementation_source_sha256: str = MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256
    economic_training_authorized: bool = False
    outer_evaluation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SCHEMA

    def validate(self) -> None:
        if self.schema != MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SCHEMA:
            raise MassiveAdaptiveAlphaModelV1Error("adaptive model schema drifted")
        if self.setting_id != "AD11":
            raise MassiveAdaptiveAlphaModelV1Error(
                "the term-structure model implements only canonical AD11"
            )
        if (
            self.bars_dimension != MASSIVE_ADAPTIVE_ALPHA_BARS_DIMENSION_V1
            or self.tape_dimension != MASSIVE_ADAPTIVE_ALPHA_TAPE_DIMENSION_V1
            or self.bucket_count != len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
            or self.expert_ids != MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256
        ):
            raise MassiveAdaptiveAlphaModelV1Error(
                "adaptive model source or scientific identity drifted"
            )
        for name in (
            "intraday_feature_dimension",
            "token_dimension",
            "fast_window_sessions",
            "maximum_context_sessions",
            "maximum_intraday_intervals",
            "market_latent_count",
            "attention_heads",
        ):
            _positive_int(name, getattr(self, name))
        if (
            self.fast_window_sessions > self.maximum_context_sessions
            or self.token_dimension % self.attention_heads != 0
            or not 0.0 <= self.dropout_probability < 1.0
        ):
            raise MassiveAdaptiveAlphaModelV1Error(
                "adaptive temporal or attention configuration is invalid"
            )
        if any(
            (
                self.economic_training_authorized,
                self.outer_evaluation_authorized,
                self.profitability_reporting_authorized,
                self.lockbox_access_authorized,
                self.reinforcement_learning_authorized,
            )
        ):
            raise MassiveAdaptiveAlphaModelV1Error(
                "an engineering model cannot authorize downstream stages"
            )
        assert_no_adaptive_hold_semantics(self)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1 = MassiveAdaptiveAlphaModelSpecV1()


class MassiveAdaptiveAlphaSequenceOutputV1(NamedTuple):
    """Causal forecast sequence over all context-universe security-days."""

    residual_distribution: AlphaDistribution
    raw_distribution: AlphaDistribution
    factor_return_mean: torch.Tensor
    executable_score: torch.Tensor
    bucket_router_weights: torch.Tensor
    router_weights: torch.Tensor
    stock_context: torch.Tensor
    market_context: torch.Tensor
    valid: torch.Tensor


class MassiveAdaptiveAlphaOutputV1(NamedTuple):
    """Last-session forecasts restricted by the decision-time action mask."""

    residual_distribution: AlphaDistribution
    raw_distribution: AlphaDistribution
    factor_return_mean: torch.Tensor
    executable_score: torch.Tensor
    bucket_router_weights: torch.Tensor
    router_weights: torch.Tensor
    stock_context: torch.Tensor
    market_context: torch.Tensor
    valid: torch.Tensor


class _CausalWindowExpert(nn.Module):
    def __init__(self, dimension: int, window: int) -> None:
        super().__init__()
        self.window = window
        self.convolution = nn.Conv1d(dimension, dimension, window)
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, sessions, assets, dimension = values.shape
        flattened = values.permute(0, 2, 3, 1).reshape(
            batch * assets, dimension, sessions
        )
        encoded = self.convolution(F.pad(flattened, (self.window - 1, 0)))
        encoded = encoded.reshape(batch, assets, dimension, sessions).permute(
            0, 3, 1, 2
        )
        return self.output_norm(F.gelu(encoded))


class _CausalRecurrentExpert(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.recurrent = nn.GRU(dimension, dimension, batch_first=True)
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, sessions, assets, dimension = values.shape
        flattened = values.permute(0, 2, 1, 3).reshape(
            batch * assets, sessions, dimension
        )
        encoded, _ = self.recurrent(flattened)
        encoded = encoded.reshape(batch, assets, sessions, dimension).permute(
            0, 2, 1, 3
        )
        return self.output_norm(encoded)


class _BucketDistributionHead(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dimension, 5)

    def forward(self, values: torch.Tensor) -> AlphaDistribution:
        raw = self.projection(values)
        mean, median, lower_gap, upper_gap, raw_scale = raw.unbind(dim=-1)
        return AlphaDistribution(
            mean=mean,
            downside_quantile=median - F.softplus(lower_gap),
            median=median,
            upside_quantile=median + F.softplus(upper_gap),
            scale=F.softplus(raw_scale) + 1.0e-6,
        )


def _mask_distribution(
    distribution: AlphaDistribution,
    valid: torch.Tensor,
) -> AlphaDistribution:
    expanded = valid.unsqueeze(-1)
    zeros = torch.zeros_like(distribution.mean)
    return AlphaDistribution(
        mean=torch.where(expanded, distribution.mean, zeros),
        downside_quantile=torch.where(
            expanded, distribution.downside_quantile, zeros
        ),
        median=torch.where(expanded, distribution.median, zeros),
        upside_quantile=torch.where(expanded, distribution.upside_quantile, zeros),
        scale=torch.where(expanded, distribution.scale, torch.ones_like(zeros)),
    )


def _rolling_any(values: torch.Tensor, window: int) -> torch.Tensor:
    batch, sessions, assets = values.shape
    flattened = values.permute(0, 2, 1).reshape(batch * assets, 1, sessions)
    pooled = F.max_pool1d(
        F.pad(flattened.to(torch.float32), (window - 1, 0)),
        kernel_size=window,
        stride=1,
    )
    return pooled.to(torch.bool).reshape(batch, assets, sessions).permute(0, 2, 1)


def _validate_inputs(
    *,
    spec: MassiveAdaptiveAlphaModelSpecV1,
    bars_values: torch.Tensor,
    bars_valid: torch.Tensor,
    tape_values: torch.Tensor,
    tape_valid: torch.Tensor,
    source_staleness: torch.Tensor,
    context_membership: torch.Tensor,
    action_mask: torch.Tensor,
    intraday_values: torch.Tensor | None,
    intraday_valid: torch.Tensor | None,
) -> None:
    spec.validate()
    if (
        bars_values.ndim != 4
        or bars_values.shape[-1] != spec.bars_dimension
        or bars_values.shape[1] > spec.maximum_context_sessions
        or tape_values.shape != bars_values.shape[:-1] + (spec.tape_dimension,)
        or bars_valid.shape != bars_values.shape
        or tape_valid.shape != tape_values.shape
        or bars_valid.dtype != torch.bool
        or tape_valid.dtype != torch.bool
        or source_staleness.shape != bars_values.shape[:-1] + (2,)
        or context_membership.shape != bars_values.shape[:-1]
        or context_membership.dtype != torch.bool
        or action_mask.shape != context_membership.shape
        or action_mask.dtype != torch.bool
        or bars_values.dtype != tape_values.dtype
        or bars_values.dtype != source_staleness.dtype
        or not bars_values.is_floating_point()
        or any(
            tensor.device != bars_values.device
            for tensor in (
                bars_valid,
                tape_values,
                tape_valid,
                source_staleness,
                context_membership,
                action_mask,
            )
        )
        or not bool(torch.isfinite(bars_values).all())
        or not bool(torch.isfinite(tape_values).all())
        or not bool(torch.isfinite(source_staleness).all())
        or bool((source_staleness < 0.0).any())
        or bool((~bars_valid & (bars_values != 0.0)).any())
        or bool((~tape_valid & (tape_values != 0.0)).any())
        or bool((bars_valid.any(dim=-1) & ~context_membership).any())
        or bool((tape_valid.any(dim=-1) & ~context_membership).any())
        or bool((~context_membership.unsqueeze(-1) & (source_staleness != 0.0)).any())
        or bool((context_membership.sum(dim=2) == 0).any())
        or bool((action_mask.sum(dim=2) == 0).any())
        or bool((action_mask & ~context_membership).any())
    ):
        raise MassiveAdaptiveAlphaModelV1Error(
            "adaptive bars, tape, staleness, membership, or action inputs are malformed"
        )
    if (intraday_values is None) != (intraday_valid is None):
        raise MassiveAdaptiveAlphaModelV1Error(
            "intraday values and validity must be supplied together"
        )
    if intraday_values is None or intraday_valid is None:
        return
    if (
        intraday_values.ndim != 5
        or intraday_values.shape[:3] != bars_values.shape[:3]
        or intraday_values.shape[3] > spec.maximum_intraday_intervals
        or intraday_values.shape[-1] != spec.intraday_feature_dimension
        or intraday_valid.shape != intraday_values.shape[:-1]
        or intraday_valid.dtype != torch.bool
        or intraday_values.dtype != bars_values.dtype
        or intraday_values.device != bars_values.device
        or intraday_valid.device != bars_values.device
        or not bool(torch.isfinite(intraday_values).all())
        or bool((~intraday_valid.unsqueeze(-1) & (intraday_values != 0.0)).any())
        or bool((intraday_valid.any(dim=-1) & ~context_membership).any())
    ):
        raise MassiveAdaptiveAlphaModelV1Error("adaptive intraday inputs are malformed")


class MassiveAdaptiveAlphaTermStructureModelV1(nn.Module):
    """Temporal mixture of source-gated experts with market-latent routing."""

    def __init__(
        self,
        spec: MassiveAdaptiveAlphaModelSpecV1 = MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
    ) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        dimension = spec.token_dimension
        self.bars_projection = nn.Sequential(
            nn.Linear(2 * spec.bars_dimension + 2, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.tape_projection = nn.Sequential(
            nn.Linear(2 * spec.tape_dimension + 2, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.intraday = OrderedFiveMinuteEncoder(
            OrderedFiveMinuteConfig(
                feature_dimension=spec.intraday_feature_dimension,
                token_dimension=dimension,
                layers=2,
                heads=spec.attention_heads,
                feedforward_dimension=4 * dimension,
                dropout=spec.dropout_probability,
                intervals_per_session=spec.maximum_intraday_intervals,
            )
        )
        self.status_projection = nn.Sequential(
            nn.Linear(6, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.bars_fast = _CausalWindowExpert(
            dimension, spec.fast_window_sessions
        )
        self.bars_slow = _CausalRecurrentExpert(dimension)
        self.tape_fast = _CausalWindowExpert(
            dimension, spec.fast_window_sessions
        )
        self.tape_slow = _CausalRecurrentExpert(dimension)
        self.intraday_slow = _CausalRecurrentExpert(dimension)
        self.market = MarketLatentEncoder(
            MarketLatentConfig(
                token_dimension=dimension,
                latent_count=spec.market_latent_count,
                heads=spec.attention_heads,
                dropout=spec.dropout_probability,
            )
        )
        router_input_dimension = 2 * dimension
        self.router = nn.Linear(
            router_input_dimension,
            spec.bucket_count * len(spec.expert_ids),
        )
        self.term_router = nn.Linear(router_input_dimension, spec.bucket_count)
        self.bucket_embeddings = nn.Parameter(
            torch.empty(spec.bucket_count, dimension)
        )
        nn.init.normal_(self.bucket_embeddings, mean=0.0, std=0.02)
        self.bucket_norm = nn.LayerNorm(dimension)
        self.residual_head = _BucketDistributionHead(dimension)
        self.raw_head = _BucketDistributionHead(dimension)
        self.factor_head = nn.Linear(dimension, spec.bucket_count)

    def _intraday_day_tokens(
        self,
        *,
        bars_values: torch.Tensor,
        intraday_values: torch.Tensor | None,
        intraday_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sessions, assets, _ = bars_values.shape
        output = torch.zeros(
            batch * sessions * assets,
            self.spec.token_dimension,
            dtype=bars_values.dtype,
            device=bars_values.device,
        )
        available = torch.zeros(
            batch * sessions * assets,
            dtype=torch.bool,
            device=bars_values.device,
        )
        if intraday_values is not None and intraday_valid is not None:
            flattened_values = intraday_values.reshape(
                batch * sessions * assets,
                intraday_values.shape[3],
                intraday_values.shape[4],
            )
            flattened_valid = intraday_valid.reshape(
                batch * sessions * assets,
                intraday_valid.shape[3],
            )
            available = flattened_valid.any(dim=1)
            selected = torch.nonzero(available, as_tuple=False).flatten()
            if selected.numel() > 0:
                output[selected] = self.intraday(
                    flattened_values.index_select(0, selected),
                    flattened_valid.index_select(0, selected),
                )
        return (
            output.reshape(batch, sessions, assets, self.spec.token_dimension),
            available.reshape(batch, sessions, assets),
        )

    def forward_sequence(
        self,
        *,
        bars_values: torch.Tensor,
        bars_valid: torch.Tensor,
        tape_values: torch.Tensor,
        tape_valid: torch.Tensor,
        source_staleness: torch.Tensor,
        context_membership: torch.Tensor,
        action_mask: torch.Tensor,
        intraday_values: torch.Tensor | None = None,
        intraday_valid: torch.Tensor | None = None,
    ) -> MassiveAdaptiveAlphaSequenceOutputV1:
        """Return causal forecasts at every supplied decision-time session."""

        _validate_inputs(
            spec=self.spec,
            bars_values=bars_values,
            bars_valid=bars_valid,
            tape_values=tape_values,
            tape_valid=tape_valid,
            source_staleness=source_staleness,
            context_membership=context_membership,
            action_mask=action_mask,
            intraday_values=intraday_values,
            intraday_valid=intraday_valid,
        )
        membership = context_membership.to(bars_values.dtype).unsqueeze(-1)
        bars_day = self.bars_projection(
            torch.cat(
                (
                    bars_values,
                    bars_valid.to(bars_values.dtype),
                    torch.log1p(source_staleness[..., :1]),
                    membership,
                ),
                dim=-1,
            )
        )
        tape_day = self.tape_projection(
            torch.cat(
                (
                    tape_values,
                    tape_valid.to(tape_values.dtype),
                    torch.log1p(source_staleness[..., 1:]),
                    membership,
                ),
                dim=-1,
            )
        )
        intraday_day, intraday_available = self._intraday_day_tokens(
            bars_values=bars_values,
            intraday_values=intraday_values,
            intraday_valid=intraday_valid,
        )
        bars_observed = bars_valid.any(dim=-1)
        tape_observed = tape_valid.any(dim=-1)
        status = self.status_projection(
            torch.cat(
                (
                    torch.log1p(source_staleness),
                    bars_valid.to(bars_values.dtype).mean(dim=-1, keepdim=True),
                    tape_valid.to(tape_values.dtype).mean(dim=-1, keepdim=True),
                    (
                        intraday_valid.to(bars_values.dtype).mean(
                            dim=-1, keepdim=True
                        )
                        if intraday_valid is not None
                        else torch.zeros_like(source_staleness[..., :1])
                    ),
                    membership,
                ),
                dim=-1,
            )
        )
        expert_tokens = torch.stack(
            (
                self.bars_fast(bars_day),
                self.bars_slow(bars_day),
                self.tape_fast(tape_day),
                self.tape_slow(tape_day),
                self.intraday_slow(intraday_day),
                status,
            ),
            dim=-2,
        )
        expert_available = torch.stack(
            (
                _rolling_any(bars_observed, self.spec.fast_window_sessions),
                bars_observed.cumsum(dim=1) > 0,
                _rolling_any(tape_observed, self.spec.fast_window_sessions),
                tape_observed.cumsum(dim=1) > 0,
                intraday_available.cumsum(dim=1) > 0,
                context_membership,
            ),
            dim=-1,
        ) & context_membership.unsqueeze(-1)
        availability = expert_available.to(expert_tokens.dtype).unsqueeze(-1)
        base = (expert_tokens * availability).sum(dim=-2) / availability.sum(
            dim=-2
        ).clamp_min(1.0)
        batch, sessions, assets, dimension = base.shape
        stock_context, market_context = self.market(
            base.reshape(batch * sessions, assets, dimension),
            context_membership.reshape(batch * sessions, assets),
        )
        stock_context = stock_context.reshape(batch, sessions, assets, dimension)
        market_context = market_context.reshape(
            batch,
            sessions,
            self.spec.market_latent_count,
            dimension,
        )
        market_summary = market_context.mean(dim=2)
        router_input = torch.cat(
            (
                stock_context,
                market_summary.unsqueeze(2).expand(-1, -1, assets, -1),
            ),
            dim=-1,
        )
        router_logits = self.router(router_input).reshape(
            batch,
            sessions,
            assets,
            self.spec.bucket_count,
            len(self.spec.expert_ids),
        )
        router_logits = router_logits.masked_fill(
            ~expert_available.unsqueeze(-2),
            torch.finfo(router_logits.dtype).min,
        )
        router_weights = torch.softmax(router_logits, dim=-1)
        routed = (
            router_weights.unsqueeze(-1) * expert_tokens.unsqueeze(-3)
        ).sum(dim=-2)
        bucket_hidden = self.bucket_norm(
            routed
            + stock_context.unsqueeze(-2)
            + self.bucket_embeddings.to(routed).view(
                1, 1, 1, self.spec.bucket_count, dimension
            )
        )
        residual = _mask_distribution(self.residual_head(bucket_hidden), action_mask)
        raw = _mask_distribution(self.raw_head(bucket_hidden), action_mask)
        bucket_router_weights = torch.softmax(self.term_router(router_input), dim=-1)
        bucket_router_weights = torch.where(
            action_mask.unsqueeze(-1),
            bucket_router_weights,
            torch.zeros_like(bucket_router_weights),
        )
        executable_score = (
            bucket_router_weights * residual.mean
        ).sum(dim=-1)
        router_weights = torch.where(
            action_mask.unsqueeze(-1).unsqueeze(-1),
            router_weights,
            torch.zeros_like(router_weights),
        )
        stock_context = torch.where(
            action_mask.unsqueeze(-1),
            stock_context,
            torch.zeros_like(stock_context),
        )
        return MassiveAdaptiveAlphaSequenceOutputV1(
            residual_distribution=residual,
            raw_distribution=raw,
            factor_return_mean=self.factor_head(market_summary),
            executable_score=executable_score,
            bucket_router_weights=bucket_router_weights,
            router_weights=router_weights,
            stock_context=stock_context,
            market_context=market_context,
            valid=action_mask,
        )

    def forward(
        self,
        *,
        bars_values: torch.Tensor,
        bars_valid: torch.Tensor,
        tape_values: torch.Tensor,
        tape_valid: torch.Tensor,
        source_staleness: torch.Tensor,
        context_membership: torch.Tensor,
        action_mask: torch.Tensor,
        intraday_values: torch.Tensor | None = None,
        intraday_valid: torch.Tensor | None = None,
    ) -> MassiveAdaptiveAlphaOutputV1:
        """Return the last legal decision output on the explicit action support."""

        sequence = self.forward_sequence(
            bars_values=bars_values,
            bars_valid=bars_valid,
            tape_values=tape_values,
            tape_valid=tape_valid,
            source_staleness=source_staleness,
            context_membership=context_membership,
            action_mask=action_mask,
            intraday_values=intraday_values,
            intraday_valid=intraday_valid,
        )
        residual = AlphaDistribution(
            *(value[:, -1] for value in sequence.residual_distribution)
        )
        raw = AlphaDistribution(*(value[:, -1] for value in sequence.raw_distribution))
        current_action_mask = action_mask[:, -1]
        return MassiveAdaptiveAlphaOutputV1(
            residual_distribution=_mask_distribution(residual, current_action_mask),
            raw_distribution=_mask_distribution(raw, current_action_mask),
            factor_return_mean=sequence.factor_return_mean[:, -1],
            executable_score=sequence.executable_score[:, -1],
            bucket_router_weights=sequence.bucket_router_weights[:, -1],
            router_weights=torch.where(
                current_action_mask.unsqueeze(-1).unsqueeze(-1),
                sequence.router_weights[:, -1],
                torch.zeros_like(sequence.router_weights[:, -1]),
            ),
            stock_context=torch.where(
                current_action_mask.unsqueeze(-1),
                sequence.stock_context[:, -1],
                torch.zeros_like(sequence.stock_context[:, -1]),
            ),
            market_context=sequence.market_context[:, -1],
            valid=current_action_mask,
        )


__all__ = [
    "MASSIVE_ADAPTIVE_ALPHA_BARS_DIMENSION_V1",
    "MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1",
    "MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1",
    "MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_ALPHA_TAPE_DIMENSION_V1",
    "MassiveAdaptiveAlphaModelSpecV1",
    "MassiveAdaptiveAlphaModelV1Error",
    "MassiveAdaptiveAlphaOutputV1",
    "MassiveAdaptiveAlphaSequenceOutputV1",
    "MassiveAdaptiveAlphaTermStructureModelV1",
]
