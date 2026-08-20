"""Causal spread, participation, impact, delay, and fee cost primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class AlphaExecutionCostError(ValueError):
    """An execution-cost input is unavailable or outside its economic domain."""


@dataclass(frozen=True, slots=True)
class ExecutionCostObservation:
    decision_at_ms: int
    available_at_ms: int
    spread_basis_points: float
    daily_volatility: float
    order_notional: float
    trailing_adv_notional: float
    delay_cost_basis_points: float
    fee_basis_points: float

    def validate(self) -> None:
        if (
            isinstance(self.decision_at_ms, bool)
            or not isinstance(self.decision_at_ms, int)
            or self.decision_at_ms < 0
            or isinstance(self.available_at_ms, bool)
            or not isinstance(self.available_at_ms, int)
            or self.available_at_ms < 0
            or self.available_at_ms > self.decision_at_ms
        ):
            raise AlphaExecutionCostError("execution inputs were unavailable at decision time")
        for name in (
            "spread_basis_points",
            "daily_volatility",
            "order_notional",
            "trailing_adv_notional",
            "delay_cost_basis_points",
            "fee_basis_points",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise AlphaExecutionCostError(f"{name} must be finite and nonnegative")
        if self.trailing_adv_notional <= 0.0:
            raise AlphaExecutionCostError("trailing ADV must be positive")

    @property
    def participation(self) -> float:
        self.validate()
        return self.order_notional / self.trailing_adv_notional


@dataclass(frozen=True, slots=True)
class SquareRootImpactConfig:
    volatility_coefficient: float
    linear_participation_coefficient: float
    maximum_participation: float

    def validate(self) -> None:
        for name in (
            "volatility_coefficient",
            "linear_participation_coefficient",
            "maximum_participation",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise AlphaExecutionCostError(f"{name} must be finite and nonnegative")
        if not 0.0 < self.maximum_participation <= 1.0:
            raise AlphaExecutionCostError("maximum participation must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ExecutionCostEstimate:
    participation: float
    half_spread_basis_points: float
    nonlinear_impact_basis_points: float
    linear_impact_basis_points: float
    delay_cost_basis_points: float
    fee_basis_points: float
    total_one_way_cost_basis_points: float
    clipped: bool


def estimate_execution_cost(
    observation: ExecutionCostObservation,
    config: SquareRootImpactConfig,
) -> ExecutionCostEstimate:
    observation.validate()
    config.validate()
    participation = observation.participation
    clipped = participation > config.maximum_participation
    effective_participation = min(participation, config.maximum_participation)
    nonlinear = (
        10_000.0
        * config.volatility_coefficient
        * observation.daily_volatility
        * math.sqrt(effective_participation)
    )
    linear = (
        10_000.0
        * config.linear_participation_coefficient
        * effective_participation
    )
    half_spread = 0.5 * observation.spread_basis_points
    total = (
        half_spread
        + nonlinear
        + linear
        + observation.delay_cost_basis_points
        + observation.fee_basis_points
    )
    return ExecutionCostEstimate(
        participation=participation,
        half_spread_basis_points=half_spread,
        nonlinear_impact_basis_points=nonlinear,
        linear_impact_basis_points=linear,
        delay_cost_basis_points=observation.delay_cost_basis_points,
        fee_basis_points=observation.fee_basis_points,
        total_one_way_cost_basis_points=total,
        clipped=clipped,
    )


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    capital: float
    weighted_cost_basis_points: float
    mean_participation: float
    maximum_participation: float
    clipped_order_fraction: float
    lost_notional_fraction: float


def evaluate_capacity(
    *,
    capital: float,
    order_weight_changes: Sequence[float],
    observations: Sequence[ExecutionCostObservation],
    config: SquareRootImpactConfig,
) -> CapacityEstimate:
    """Evaluate a frozen order vector at one declared capital level."""

    if not math.isfinite(capital) or capital <= 0.0:
        raise AlphaExecutionCostError("capacity capital must be finite and positive")
    if not order_weight_changes or len(order_weight_changes) != len(observations):
        raise AlphaExecutionCostError("capacity order inventory is empty or misaligned")
    estimates: list[ExecutionCostEstimate] = []
    requested: list[float] = []
    executed: list[float] = []
    for weight, source in zip(order_weight_changes, observations, strict=True):
        if not math.isfinite(weight):
            raise AlphaExecutionCostError("capacity order weight is nonfinite")
        notional = abs(weight) * capital
        adjusted = ExecutionCostObservation(
            decision_at_ms=source.decision_at_ms,
            available_at_ms=source.available_at_ms,
            spread_basis_points=source.spread_basis_points,
            daily_volatility=source.daily_volatility,
            order_notional=notional,
            trailing_adv_notional=source.trailing_adv_notional,
            delay_cost_basis_points=source.delay_cost_basis_points,
            fee_basis_points=source.fee_basis_points,
        )
        estimate = estimate_execution_cost(adjusted, config)
        estimates.append(estimate)
        requested.append(notional)
        executed.append(
            min(notional, config.maximum_participation * source.trailing_adv_notional)
        )
    total_notional = math.fsum(requested)
    if total_notional <= 0.0:
        raise AlphaExecutionCostError("capacity order vector has zero turnover")
    weighted_cost = math.fsum(
        notional * estimate.total_one_way_cost_basis_points
        for notional, estimate in zip(requested, estimates, strict=True)
    ) / total_notional
    clipped_count = sum(estimate.clipped for estimate in estimates)
    return CapacityEstimate(
        capital=capital,
        weighted_cost_basis_points=weighted_cost,
        mean_participation=math.fsum(
            estimate.participation for estimate in estimates
        )
        / len(estimates),
        maximum_participation=max(estimate.participation for estimate in estimates),
        clipped_order_fraction=clipped_count / len(estimates),
        lost_notional_fraction=1.0 - math.fsum(executed) / total_notional,
    )


__all__ = [
    "AlphaExecutionCostError",
    "CapacityEstimate",
    "ExecutionCostEstimate",
    "ExecutionCostObservation",
    "SquareRootImpactConfig",
    "estimate_execution_cost",
    "evaluate_capacity",
]
