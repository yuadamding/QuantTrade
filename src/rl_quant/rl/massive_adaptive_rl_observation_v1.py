"""Compact observation derived from the prepared compiler state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import MassiveAdaptiveRLActionV1

if TYPE_CHECKING:
    from rl_quant.evaluation.massive_adaptive_economic_step_v1 import (
        MassiveAdaptivePreparedStepV1,
    )

MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-observation-v1"
)
MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_MAXIMUM_DIMENSION = 128
MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "prepared-compiler-inputs-and-three-economic-books",
        "forecast_statistics_per_bucket": 8,
        "normalization": "fixed-deterministic-transforms",
        "online_running_normalization": False,
        "future_fill": False,
        "supervised_target": False,
    }
)


class MassiveAdaptiveRLObservationV1Error(ValueError):
    """Observation contents or causal source receipts differ."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLTrailingStateV1:
    strategy_active_log_returns: tuple[float, ...] = ()
    incremental_rl_log_returns: tuple[float, ...] = ()
    previous_realized_turnover: float = 0.0
    previous_fill_fraction: float = 1.0
    previous_unfilled_fraction: float = 0.0
    previous_capacity_utilization: float = 0.0
    current_drawdown: float = 0.0

    def validate(self) -> None:
        if (
            len(self.strategy_active_log_returns) > 63
            or len(self.incremental_rl_log_returns) > 63
            or any(
                not math.isfinite(value)
                for value in (
                    *self.strategy_active_log_returns,
                    *self.incremental_rl_log_returns,
                    self.previous_realized_turnover,
                    self.previous_fill_fraction,
                    self.previous_unfilled_fraction,
                    self.previous_capacity_utilization,
                    self.current_drawdown,
                )
            )
            or not 0.0 <= self.previous_fill_fraction <= 1.0
            or not 0.0 <= self.previous_unfilled_fraction <= 1.0
            or self.previous_realized_turnover < 0.0
            or self.previous_capacity_utilization < 0.0
            or not 0.0 <= self.current_drawdown <= 1.0
        ):
            raise MassiveAdaptiveRLObservationV1Error(
                "adaptive RL trailing economic state differs"
            )
        assert_no_adaptive_hold_semantics(self)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLObservationV1:
    decision_session_date: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    prepared_step_receipt_sha256: str
    strategy_book_receipt_sha256: str
    neutral_book_receipt_sha256: str
    benchmark_book_receipt_sha256: str
    previous_action_receipt_sha256: str
    trailing_state_receipt_sha256: str
    source_inventory_sha256: str
    semantic_receipt_sha256: str
    development_observation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SCHEMA
            or not self.decision_session_date
            or not self.values
            or len(self.values) != len(self.feature_names)
            or len(self.values) > MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_MAXIMUM_DIMENSION
            or len(set(self.feature_names)) != len(self.feature_names)
            or any(not name or name != name.strip() for name in self.feature_names)
            or any(not math.isfinite(value) for value in self.values)
            or not isinstance(self.development_observation_authorized, bool)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SPEC_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLObservationV1Error(
                "adaptive RL observation differs"
            )
        for value in (
            self.prepared_step_receipt_sha256,
            self.strategy_book_receipt_sha256,
            self.neutral_book_receipt_sha256,
            self.benchmark_book_receipt_sha256,
            self.previous_action_receipt_sha256,
            self.trailing_state_receipt_sha256,
            self.source_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            if len(value) != 64:
                raise MassiveAdaptiveRLObservationV1Error(
                    "adaptive RL observation digest differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _fixed_return_transform(value: float) -> float:
    return math.copysign(math.log1p(min(abs(value) * 10_000.0, 10_000.0)), value) / math.log(10_001.0)


def _trailing_sum(values: tuple[float, ...], count: int) -> float:
    return _fixed_return_transform(sum(values[-count:]))


def build_massive_adaptive_rl_observation_v1(
    *,
    prepared: MassiveAdaptivePreparedStepV1,
    previous_action: MassiveAdaptiveRLActionV1,
    trailing_state: MassiveAdaptiveRLTrailingStateV1,
) -> MassiveAdaptiveRLObservationV1:
    """Summarize only causal forecast, book, risk, cost, and execution state."""

    prepared.validate()
    previous_action.validate()
    trailing_state.validate()
    inputs = prepared.strategy_compiler_inputs
    expected = np.asarray(inputs.bucket_expected_residual_returns, dtype=np.float64)
    bucket_covariance = np.asarray(inputs.bucket_covariances, dtype=np.float64)
    valid = np.asarray(inputs.buy_eligible, dtype=bool)
    forced = np.asarray(inputs.forced_exit, dtype=bool)
    if not bool(valid.any()):
        raise MassiveAdaptiveRLObservationV1Error(
            "adaptive RL observation has no valid forecast support"
        )
    feature_names: list[str] = []
    values: list[float] = []
    for bucket_index, bucket in enumerate(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS):
        bucket_values = expected[valid, bucket_index]
        uncertainty = np.sqrt(
            np.maximum(bucket_covariance[valid, bucket_index, bucket_index], 1.0e-12)
        )
        scale = max(float(np.mean(uncertainty)), 1.0e-8)
        normalized = np.clip(bucket_values / scale, -20.0, 20.0)
        q10, q50, q90 = np.quantile(normalized, (0.10, 0.50, 0.90))
        top = normalized[normalized >= q90]
        statistics = (
            float(np.mean(normalized)),
            float(np.std(normalized)),
            float(q10),
            float(q50),
            float(q90),
            float(np.mean(bucket_values > 0.0)),
            float(np.clip(np.mean(uncertainty) * 100.0, 0.0, 20.0)),
            float(np.mean(top) - q50),
        )
        labels = (
            "mean",
            "standard_deviation",
            "q10",
            "median",
            "q90",
            "positive_fraction",
            "uncertainty",
            "top_decile_spread",
        )
        feature_names.extend(f"{bucket.bucket_id}_{label}" for label in labels)
        values.extend(statistics)

    strategy_weights = np.asarray(inputs.pretrade_weights, dtype=np.float64)
    neutral_weights = np.asarray(
        prepared.neutral_compiler_inputs.pretrade_weights, dtype=np.float64
    )
    benchmark_weights = np.asarray(inputs.benchmark_weights, dtype=np.float64)
    active = strategy_weights - benchmark_weights
    covariance = np.asarray(inputs.risk_covariance, dtype=np.float64)
    betas = np.asarray(inputs.active_betas, dtype=np.float64)
    sorted_weights = np.sort(strategy_weights)[::-1]
    squared = float(np.square(strategy_weights).sum())
    effective = 0.0 if squared <= 0.0 else min(1.0 / squared / len(strategy_weights), 1.0)
    cumulative_alpha = np.cumsum(expected, axis=1)
    best_alpha = np.max(cumulative_alpha, axis=1)
    forced_turnover = float(strategy_weights[forced].sum())
    portfolio_values = (
        1.0 - float(strategy_weights.sum()),
        1.0 - float(neutral_weights.sum()),
        float(sorted_weights[0]) if sorted_weights.size else 0.0,
        float(sorted_weights[:10].sum()),
        effective,
        float(np.clip(betas @ active, -1.0, 1.0)),
        float(
            np.clip(
                math.sqrt(252.0 * max(float(active @ covariance @ active), 0.0)),
                0.0,
                1.0,
            )
        ),
        _fixed_return_transform(float(strategy_weights @ best_alpha)),
        min(forced_turnover, 1.0),
        min(trailing_state.previous_realized_turnover, 1.0),
        trailing_state.previous_fill_fraction,
        trailing_state.previous_unfilled_fraction,
        min(trailing_state.previous_capacity_utilization, 1.0),
        trailing_state.current_drawdown,
        _trailing_sum(trailing_state.strategy_active_log_returns, 5),
        _trailing_sum(trailing_state.strategy_active_log_returns, 20),
        _trailing_sum(trailing_state.strategy_active_log_returns, 63),
        _trailing_sum(trailing_state.incremental_rl_log_returns, 5),
        _trailing_sum(trailing_state.incremental_rl_log_returns, 20),
        _trailing_sum(trailing_state.incremental_rl_log_returns, 63),
    )
    feature_names.extend(
        (
            "strategy_cash_weight",
            "neutral_cash_weight",
            "largest_position_weight",
            "top10_concentration",
            "effective_position_fraction",
            "active_beta",
            "tracking_error",
            "portfolio_forecast_alpha",
            "forced_turnover_estimate",
            "previous_realized_turnover",
            "previous_fill_fraction",
            "previous_unfilled_fraction",
            "previous_capacity_utilization",
            "current_drawdown",
            "strategy_active_return_5",
            "strategy_active_return_20",
            "strategy_active_return_63",
            "incremental_return_5",
            "incremental_return_20",
            "incremental_return_63",
        )
    )
    values.extend(portfolio_values)
    risk_diagonal = np.diag(covariance)
    coverage_values = (
        float(np.mean(valid)),
        float(np.mean(valid & ~forced)),
        float(np.mean(forced)),
        float(np.mean(~np.isfinite(risk_diagonal) | (risk_diagonal <= 0.0))),
    )
    feature_names.extend(
        (
            "valid_forecast_fraction",
            "buy_eligible_fraction",
            "forced_exit_fraction",
            "missing_risk_support_fraction",
        )
    )
    values.extend(coverage_values)
    action_values = (
        *previous_action.bucket_controls,
        previous_action.uncertainty_control,
        previous_action.risk_control,
        previous_action.turnover_control,
    )
    feature_names.extend(
        (
            *(f"previous_bucket_control_{index}" for index in range(7)),
            "previous_uncertainty_control",
            "previous_risk_control",
            "previous_turnover_control",
        )
    )
    values.extend(action_values)
    trailing_receipt = semantic_sha256(asdict(trailing_state))
    source_inventory = semantic_sha256(
        (
            prepared.semantic_receipt_sha256,
            prepared.strategy_pretrade_book.semantic_receipt_sha256,
            prepared.neutral_pretrade_book.semantic_receipt_sha256,
            prepared.benchmark_pretrade_book.semantic_receipt_sha256,
            previous_action.semantic_receipt_sha256,
            trailing_receipt,
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SCHEMA,
        "decision_session_date": prepared.decision_session_date,
        "feature_names": tuple(feature_names),
        "values": tuple(float(value) for value in values),
        "prepared_step_receipt_sha256": prepared.semantic_receipt_sha256,
        "strategy_book_receipt_sha256": prepared.strategy_pretrade_book.semantic_receipt_sha256,
        "neutral_book_receipt_sha256": prepared.neutral_pretrade_book.semantic_receipt_sha256,
        "benchmark_book_receipt_sha256": prepared.benchmark_pretrade_book.semantic_receipt_sha256,
        "previous_action_receipt_sha256": previous_action.semantic_receipt_sha256,
        "trailing_state_receipt_sha256": trailing_receipt,
        "source_inventory_sha256": source_inventory,
        "development_observation_authorized": bool(
            prepared.strategy_compiler_input_authority.development_compiler_authorized
            and prepared.neutral_compiler_input_authority.development_compiler_authorized
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLObservationV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_OBSERVATION_V1_MAXIMUM_DIMENSION",
    "MassiveAdaptiveRLObservationV1",
    "MassiveAdaptiveRLObservationV1Error",
    "MassiveAdaptiveRLTrailingStateV1",
    "build_massive_adaptive_rl_observation_v1",
]
