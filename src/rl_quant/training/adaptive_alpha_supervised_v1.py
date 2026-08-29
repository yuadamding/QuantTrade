"""Source-shaped supervised objective for the adaptive-alpha term structure.

The exact CPU portfolio compiler remains the authoritative economic selector.
This module supplies a differentiable training surrogate that combines
distributional raw-return and factor-residual learning with a long-only,
benchmark-relative, cost-paid soft portfolio utility.  All targets remain
economic returns; there is no position-age input, duration loss, persistence
reward, or fixed-exit objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import string

import torch
from torch.nn import functional as F

from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MassiveAdaptiveAlphaSequenceOutputV1,
)
from rl_quant.models.alpha_hierarchical import AlphaDistribution
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.alpha_supervised import (
    AlphaObjectiveConfig,
    AlphaObjectiveLoss,
    AlphaSupervisedBatch,
    alpha_supervised_loss,
)


MASSIVE_ADAPTIVE_ALPHA_SUPERVISED_V1_SCHEMA = (
    "rl-quant.massive-adaptive-alpha-supervised-v1"
)
MASSIVE_ADAPTIVE_ALPHA_TRAINING_BATCH_V1_SCHEMA = (
    "rl-quant.massive-adaptive-alpha-training-batch-v1"
)
_BUCKET_COUNT = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
_BUCKET_ENDS = tuple(row.end_offset_sessions for row in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
_BUCKET_WIDTHS = tuple(
    row.end_offset_sessions - row.start_offset_sessions
    for row in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS
)
_DEFAULT_TARGET_SCALES = tuple(0.02 * math.sqrt(width) for width in _BUCKET_WIDTHS)
_HEX = frozenset(string.hexdigits.lower())


class MassiveAdaptiveAlphaSupervisedV1Error(ValueError):
    """Adaptive training evidence or its economic objective is malformed."""


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveAdaptiveAlphaSupervisedV1Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MassiveAdaptiveAlphaSupervisedV1Error(f"{name} must be finite")
    return result


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAlphaObjectiveConfigV1:
    """Frozen-shape engineering loss and soft economic-surrogate weights."""

    residual_objective: AlphaObjectiveConfig = AlphaObjectiveConfig(
        huber_weight=1.0,
        rank_weight=0.20,
        quantile_weight=0.25,
        calibration_weight=0.10,
        residual_ssl_weight=0.0,
    )
    raw_objective: AlphaObjectiveConfig = AlphaObjectiveConfig(
        huber_weight=1.0,
        rank_weight=0.0,
        quantile_weight=0.20,
        calibration_weight=0.10,
        residual_ssl_weight=0.0,
    )
    residual_loss_weight: float = 1.0
    raw_loss_weight: float = 0.50
    factor_loss_weight: float = 0.25
    soft_portfolio_utility_weight: float = 0.20
    softmax_temperature: float = 0.25
    one_way_cost_return: float = 0.002
    target_scale_by_bucket: tuple[float, ...] = _DEFAULT_TARGET_SCALES
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    economic_training_authorized: bool = False
    outer_evaluation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_ALPHA_SUPERVISED_V1_SCHEMA

    def validate(self) -> None:
        self.residual_objective.validate()
        self.raw_objective.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_ALPHA_SUPERVISED_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or len(self.target_scale_by_bucket) != _BUCKET_COUNT
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "adaptive objective identity or bucket inventory drifted"
            )
        for name in (
            "residual_loss_weight",
            "raw_loss_weight",
            "factor_loss_weight",
            "soft_portfolio_utility_weight",
            "one_way_cost_return",
        ):
            if _finite(name, getattr(self, name)) < 0.0:
                raise MassiveAdaptiveAlphaSupervisedV1Error(
                    f"{name} must be nonnegative"
                )
        if _finite("softmax_temperature", self.softmax_temperature) <= 0.0:
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "softmax temperature must be positive"
            )
        if any(_finite("target scale", value) <= 0.0 for value in self.target_scale_by_bucket):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "target scales must be positive"
            )
        if self.residual_loss_weight <= 0.0:
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "factor-residual learning must remain primary"
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
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "an engineering objective cannot authorize downstream stages"
            )
        assert_no_adaptive_hold_semantics(self)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


MASSIVE_ADAPTIVE_ALPHA_OBJECTIVE_CONFIG_V1 = (
    MassiveAdaptiveAlphaObjectiveConfigV1()
)


def _distribution_tensors(
    output: MassiveAdaptiveAlphaSequenceOutputV1,
) -> tuple[torch.Tensor, ...]:
    return (
        output.residual_distribution.mean,
        output.residual_distribution.downside_quantile,
        output.residual_distribution.median,
        output.residual_distribution.upside_quantile,
        output.residual_distribution.scale,
        output.raw_distribution.mean,
        output.raw_distribution.downside_quantile,
        output.raw_distribution.median,
        output.raw_distribution.upside_quantile,
        output.raw_distribution.scale,
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAlphaTrainingBatchV1:
    """One split-safe source-bound model output and target decomposition."""

    output: MassiveAdaptiveAlphaSequenceOutputV1
    raw_return_target: torch.Tensor
    factor_component_target: torch.Tensor
    residual_return_target: torch.Tensor
    target_valid: torch.Tensor
    factor_return_target: torch.Tensor
    factor_valid: torch.Tensor
    action_mask: torch.Tensor
    benchmark_weights: torch.Tensor
    benchmark_net_returns: torch.Tensor
    initial_pretrade_weights: torch.Tensor
    portfolio_utility_valid: torch.Tensor
    origin_indices: torch.Tensor
    split_start_inclusive: int
    split_stop_exclusive: int
    split_role: str
    source_bundle_receipt_sha256: str
    target_bundle_receipt_sha256: str
    factor_operator_receipt_sha256: str
    split_plan_receipt_sha256: str
    outer_test_accessed: bool = False
    lockbox_accessed: bool = False
    profitability_reporting_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_ALPHA_TRAINING_BATCH_V1_SCHEMA

    def validate(self) -> None:
        reference = self.raw_return_target
        if (
            self.schema != MASSIVE_ADAPTIVE_ALPHA_TRAINING_BATCH_V1_SCHEMA
            or not isinstance(reference, torch.Tensor)
            or reference.ndim != 4
            or reference.shape[-1] != _BUCKET_COUNT
            or not reference.is_floating_point()
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "adaptive training target geometry drifted"
            )
        batch, sessions, assets, _ = reference.shape
        aligned_float = (
            self.factor_component_target,
            self.residual_return_target,
            *_distribution_tensors(self.output),
        )
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            or not bool(torch.isfinite(value).all())
            for value in aligned_float
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "adaptive target and distribution tensors are misaligned"
            )
        if (
            self.target_valid.shape != reference.shape
            or self.target_valid.dtype != torch.bool
            or self.target_valid.device != reference.device
            or self.action_mask.shape != (batch, sessions, assets)
            or self.action_mask.dtype != torch.bool
            or self.action_mask.device != reference.device
            or self.output.valid.shape != self.action_mask.shape
            or not torch.equal(self.output.valid, self.action_mask)
            or bool((self.target_valid & ~self.action_mask.unsqueeze(-1)).any())
            or bool((self.target_valid.sum(dim=2) < 2).any())
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "adaptive action or target support is malformed"
            )
        if (
            self.output.executable_score.shape != (batch, sessions, assets)
            or self.output.executable_score.dtype != reference.dtype
            or self.output.executable_score.device != reference.device
            or not bool(torch.isfinite(self.output.executable_score).all())
            or bool(
                (
                    ~self.action_mask
                    & (self.output.executable_score != 0.0)
                ).any()
            )
            or self.output.bucket_router_weights.shape != reference.shape
            or self.output.bucket_router_weights.dtype != reference.dtype
            or self.output.bucket_router_weights.device != reference.device
            or not bool(torch.isfinite(self.output.bucket_router_weights).all())
            or bool((self.output.bucket_router_weights < 0.0).any())
            or bool(
                (
                    ~self.action_mask.unsqueeze(-1)
                    & (self.output.bucket_router_weights != 0.0)
                ).any()
            )
            or not torch.allclose(
                self.output.bucket_router_weights[self.action_mask].sum(dim=-1),
                torch.ones_like(self.output.executable_score[self.action_mask]),
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            or self.output.router_weights.shape[:4] != reference.shape
            or self.output.router_weights.dtype != reference.dtype
            or self.output.router_weights.device != reference.device
            or not bool(torch.isfinite(self.output.router_weights).all())
            or self.output.stock_context.shape[:3] != (batch, sessions, assets)
            or self.output.stock_context.dtype != reference.dtype
            or self.output.stock_context.device != reference.device
            or not bool(torch.isfinite(self.output.stock_context).all())
            or self.output.market_context.shape[:2] != (batch, sessions)
            or self.output.market_context.dtype != reference.dtype
            or self.output.market_context.device != reference.device
            or not bool(torch.isfinite(self.output.market_context).all())
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "adaptive executable-score or context surface is malformed"
            )
        for name, value in (
            ("raw target", self.raw_return_target),
            ("factor component", self.factor_component_target),
            ("residual target", self.residual_return_target),
        ):
            if (
                not bool(torch.isfinite(value[self.target_valid]).all())
                or bool((~self.target_valid & (value != 0.0)).any())
            ):
                raise MassiveAdaptiveAlphaSupervisedV1Error(
                    f"{name} has noncanonical missing payload"
                )
        if bool((self.raw_return_target[self.target_valid] < -1.0).any()):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "raw target contains a return below total loss"
            )
        if not torch.allclose(
            self.raw_return_target[self.target_valid],
            (
                self.factor_component_target
                + self.residual_return_target
            )[self.target_valid],
            atol=1.0e-7,
            rtol=1.0e-6,
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "raw, factor-component, and residual targets do not reconcile"
            )
        if (
            self.factor_return_target.shape != (batch, sessions, _BUCKET_COUNT)
            or self.factor_valid.shape != self.factor_return_target.shape
            or self.factor_return_target.dtype != reference.dtype
            or self.factor_return_target.device != reference.device
            or self.factor_valid.dtype != torch.bool
            or self.factor_valid.device != reference.device
            or not bool(torch.isfinite(self.factor_return_target).all())
            or bool((~self.factor_valid & (self.factor_return_target != 0.0)).any())
            or bool((self.factor_valid.sum(dim=-1) == 0).any())
            or self.output.factor_return_mean.shape != self.factor_return_target.shape
            or self.output.factor_return_mean.dtype != reference.dtype
            or self.output.factor_return_mean.device != reference.device
            or not bool(torch.isfinite(self.output.factor_return_mean).all())
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "factor-return target support is malformed"
            )
        if (
            self.benchmark_weights.shape != (batch, sessions, assets)
            or self.benchmark_net_returns.shape != (batch, sessions)
            or self.initial_pretrade_weights.shape != (batch, assets)
            or any(
                value.dtype != reference.dtype or value.device != reference.device
                for value in (
                    self.benchmark_weights,
                    self.benchmark_net_returns,
                    self.initial_pretrade_weights,
                )
            )
            or not bool(torch.isfinite(self.benchmark_weights).all())
            or not bool(torch.isfinite(self.benchmark_net_returns).all())
            or not bool(torch.isfinite(self.initial_pretrade_weights).all())
            or bool((self.benchmark_weights < 0.0).any())
            or bool((self.initial_pretrade_weights < 0.0).any())
            or bool((self.benchmark_weights * ~self.action_mask).any())
            or bool(
                (
                    self.initial_pretrade_weights
                    * ~self.action_mask[:, 0]
                ).any()
            )
            or bool((self.benchmark_weights.sum(dim=2) > 1.0 + 1.0e-6).any())
            or bool((self.benchmark_weights.sum(dim=2) <= 0.0).any())
            or bool((self.initial_pretrade_weights.sum(dim=1) > 1.0 + 1.0e-6).any())
            or bool((self.benchmark_net_returns <= -1.0).any())
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "benchmark or initial pretrade book is malformed"
            )
        complete_one_session = (
            ~self.action_mask | self.target_valid[..., 0]
        ).all(dim=2)
        if (
            self.portfolio_utility_valid.shape != (batch, sessions)
            or self.portfolio_utility_valid.dtype != torch.bool
            or self.portfolio_utility_valid.device != reference.device
            or bool((self.portfolio_utility_valid & ~complete_one_session).any())
            or not bool(self.portfolio_utility_valid.any())
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "soft portfolio utility inspects incomplete one-session outcomes"
            )
        if (
            self.origin_indices.shape != (batch, sessions)
            or self.origin_indices.dtype != torch.long
            or self.origin_indices.device != reference.device
            or self.split_role not in {"training", "inner_validation"}
            or isinstance(self.split_start_inclusive, bool)
            or not isinstance(self.split_start_inclusive, int)
            or isinstance(self.split_stop_exclusive, bool)
            or not isinstance(self.split_stop_exclusive, int)
            or not 0 <= self.split_start_inclusive < self.split_stop_exclusive
            or bool((self.origin_indices < self.split_start_inclusive).any())
            or bool((self.origin_indices[:, 1:] <= self.origin_indices[:, :-1]).any())
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "adaptive training split chronology is malformed"
            )
        endpoint = self.origin_indices.unsqueeze(-1) + reference.new_tensor(
            _BUCKET_ENDS, dtype=torch.long
        )
        supported = self.target_valid.any(dim=2)
        if bool((supported & (endpoint >= self.split_stop_exclusive)).any()):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "adaptive target crosses its frozen split"
            )
        for name in (
            "source_bundle_receipt_sha256",
            "target_bundle_receipt_sha256",
            "factor_operator_receipt_sha256",
            "split_plan_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.outer_test_accessed
            or self.lockbox_accessed
            or self.profitability_reporting_authorized
        ):
            raise MassiveAdaptiveAlphaSupervisedV1Error(
                "adaptive supervised training accepts development splits only"
            )
        assert_no_adaptive_hold_semantics(self)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAlphaObjectiveLossV1:
    total: torch.Tensor
    residual: AlphaObjectiveLoss
    raw: AlphaObjectiveLoss
    factor: torch.Tensor
    soft_active_log_utility: torch.Tensor
    soft_policy_gross_return: torch.Tensor
    soft_execution_cost: torch.Tensor
    soft_one_way_turnover: torch.Tensor


def _flatten_distribution(distribution: AlphaDistribution) -> AlphaDistribution:
    dates = distribution.mean.shape[0] * distribution.mean.shape[1]
    assets = distribution.mean.shape[2]
    buckets = distribution.mean.shape[3]
    return AlphaDistribution(
        *(value.reshape(dates, assets, buckets) for value in distribution)
    )


def _scaled_distribution(
    distribution: AlphaDistribution,
    scales: torch.Tensor,
) -> AlphaDistribution:
    return AlphaDistribution(*(value / scales for value in distribution))


def _soft_portfolio_utility(
    batch: MassiveAdaptiveAlphaTrainingBatchV1,
    config: MassiveAdaptiveAlphaObjectiveConfigV1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    score = batch.output.executable_score
    masked_logits = (score / config.softmax_temperature).masked_fill(
        ~batch.action_mask,
        torch.finfo(score.dtype).min,
    )
    weights = torch.softmax(masked_logits, dim=2)
    weights = torch.where(batch.action_mask, weights, torch.zeros_like(weights))
    pretrade = torch.cat(
        (batch.initial_pretrade_weights.unsqueeze(1), weights[:, :-1]), dim=1
    )
    risky_change = (weights - pretrade).abs().sum(dim=2)
    cash_before = 1.0 - pretrade.sum(dim=2)
    cash_after = 1.0 - weights.sum(dim=2)
    turnover = 0.5 * (risky_change + (cash_after - cash_before).abs())
    cost = config.one_way_cost_return * turnover
    gross = (
        weights * batch.raw_return_target[..., 0].detach()
    ).sum(dim=2)
    policy_net = gross - cost
    if bool((policy_net[batch.portfolio_utility_valid] <= -1.0).any()):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "soft portfolio produced a return at or below total loss"
        )
    active_log = torch.log1p(policy_net) - torch.log1p(
        batch.benchmark_net_returns.detach()
    )
    selected = batch.portfolio_utility_valid
    if not bool(selected.any()):
        zero = score.sum() * 0.0
        return zero, zero, zero, zero
    return (
        active_log[selected].mean(),
        gross[selected].mean(),
        cost[selected].mean(),
        turnover[selected].mean(),
    )


def massive_adaptive_alpha_supervised_loss_v1(
    batch: MassiveAdaptiveAlphaTrainingBatchV1,
    config: MassiveAdaptiveAlphaObjectiveConfigV1 = (
        MASSIVE_ADAPTIVE_ALPHA_OBJECTIVE_CONFIG_V1
    ),
) -> MassiveAdaptiveAlphaObjectiveLossV1:
    """Compute distributional alpha losses plus cost-paid active utility."""

    batch.validate()
    config.validate()
    dates = batch.raw_return_target.shape[0] * batch.raw_return_target.shape[1]
    assets = batch.raw_return_target.shape[2]
    scales = batch.raw_return_target.new_tensor(config.target_scale_by_bucket).view(
        1, 1, _BUCKET_COUNT
    )
    valid = batch.target_valid.reshape(dates, assets, _BUCKET_COUNT)
    residual_distribution = _scaled_distribution(
        _flatten_distribution(batch.output.residual_distribution), scales
    )
    raw_distribution = _scaled_distribution(
        _flatten_distribution(batch.output.raw_distribution), scales
    )
    residual_target = batch.residual_return_target.detach().reshape(
        dates, assets, _BUCKET_COUNT
    ) / scales
    raw_target = batch.raw_return_target.detach().reshape(
        dates, assets, _BUCKET_COUNT
    ) / scales
    residual = alpha_supervised_loss(
        AlphaSupervisedBatch(
            distribution=residual_distribution,
            target=residual_target,
            valid=valid,
            executable_score=residual_distribution.mean,
        ),
        config.residual_objective,
    )
    raw = alpha_supervised_loss(
        AlphaSupervisedBatch(
            distribution=raw_distribution,
            target=raw_target,
            valid=valid,
            executable_score=raw_distribution.mean,
        ),
        config.raw_objective,
    )
    factor_scale = batch.factor_return_target.new_tensor(
        config.target_scale_by_bucket
    ).view(1, 1, _BUCKET_COUNT)
    factor_rows = F.huber_loss(
        batch.output.factor_return_mean / factor_scale,
        batch.factor_return_target.detach() / factor_scale,
        reduction="none",
        delta=1.0,
    )
    factor = torch.where(
        batch.factor_valid, factor_rows, torch.zeros_like(factor_rows)
    ).sum() / batch.factor_valid.sum().clamp_min(1).to(factor_rows.dtype)
    utility, gross, cost, turnover = _soft_portfolio_utility(batch, config)
    total = (
        config.residual_loss_weight * residual.total
        + config.raw_loss_weight * raw.total
        + config.factor_loss_weight * factor
        - config.soft_portfolio_utility_weight * utility
    )
    if not bool(torch.isfinite(total)):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive supervised objective is nonfinite"
        )
    return MassiveAdaptiveAlphaObjectiveLossV1(
        total=total,
        residual=residual,
        raw=raw,
        factor=factor,
        soft_active_log_utility=utility,
        soft_policy_gross_return=gross,
        soft_execution_cost=cost,
        soft_one_way_turnover=turnover,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_ALPHA_OBJECTIVE_CONFIG_V1",
    "MASSIVE_ADAPTIVE_ALPHA_SUPERVISED_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_ALPHA_TRAINING_BATCH_V1_SCHEMA",
    "MassiveAdaptiveAlphaObjectiveConfigV1",
    "MassiveAdaptiveAlphaObjectiveLossV1",
    "MassiveAdaptiveAlphaSupervisedV1Error",
    "MassiveAdaptiveAlphaTrainingBatchV1",
    "massive_adaptive_alpha_supervised_loss_v1",
]
