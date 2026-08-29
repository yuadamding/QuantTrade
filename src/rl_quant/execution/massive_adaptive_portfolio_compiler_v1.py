"""Deterministic no-duration compiler for the Massive adaptive-alpha surface.

This module is an engineering compiler, not an authorization boundary.  It
turns one source-bound seven-bucket forecast term structure and one drifted
pretrade book into a feasible long-only risky book plus residual CASH.  The
compiler owns one integrated trade vector, so recommendations from different
forecast buckets are netted before costs, liquidity, and risk constraints are
applied.

The state variable is the current economic book.  There is deliberately no
position-age input, scheduled exit, persistence term, or duration criterion.
An existing position differs from a new purchase only because its entry cost
is already sunk.  The complete forecast curve is reconsidered on every call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import string
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SCHEMA = (
    "rl-quant.massive-adaptive-portfolio-compiler-v1"
)
MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SOLVER = (
    "deterministic-float64-projected-concave-ascent-v1"
)
_BUCKET_IDS = tuple(row.bucket_id for row in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
_SQRT_SESSIONS_PER_YEAR = math.sqrt(252.0)
_HEX = frozenset(string.hexdigits.lower())


class MassiveAdaptivePortfolioCompilerError(ValueError):
    """The compiler input, numerical solve, or economic reconciliation failed."""


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveAdaptivePortfolioCompilerError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MassiveAdaptivePortfolioCompilerError(f"{name} must be finite")
    return result


def _nonnegative(name: str, value: object) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise MassiveAdaptivePortfolioCompilerError(f"{name} must be nonnegative")
    return result


def _positive(name: str, value: object) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise MassiveAdaptivePortfolioCompilerError(f"{name} must be positive")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveAdaptivePortfolioCompilerError(f"{name} must be a positive integer")
    return value


def _canonical_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveAdaptivePortfolioCompilerError(
            f"{name} must be a canonical nonempty string"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise MassiveAdaptivePortfolioCompilerError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _as_float_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MassiveAdaptivePortfolioCompilerError(
            f"{name} must be a rectangular numeric array"
        ) from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise MassiveAdaptivePortfolioCompilerError(
            f"{name} must be finite with shape {shape}; got {result.shape}"
        )
    return result


def _as_bool_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.bool_:
        raise MassiveAdaptivePortfolioCompilerError(
            f"{name} must be Boolean with shape {shape}; got {result.shape}"
        )
    return result


def _tuple_float(value: NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(item) for item in value.tolist())


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePortfolioCompilerConfigV1:
    """One deterministic compiler configuration.

    The defaults mirror the immutable adaptive-alpha source contract.  The
    values remain configurable here because this engineering component does
    not authorize an economic experiment or select a point on a risk frontier.
    """

    maximum_security_weight: float = 0.01
    maximum_issuer_weight: float = 0.015
    tracking_error_limit_annualized: float = 0.06
    absolute_active_beta_limit: float = 0.10
    maximum_daily_one_way_turnover: float = 0.10
    maximum_adv_participation: float = 0.02
    uncertainty_standard_deviations: float = 1.0
    risk_aversion: float = 1.0
    tail_risk_aversion: float = 0.0
    tail_confidence: float = 0.95
    solver_step_size: float = 0.25
    solver_max_iterations: int = 2_000
    projection_max_iterations: int = 500
    numerical_tolerance: float = 1.0e-9
    schema: str = "rl-quant.massive-adaptive-portfolio-compiler-config-v1"

    def validate(self) -> None:
        if self.schema != "rl-quant.massive-adaptive-portfolio-compiler-config-v1":
            raise MassiveAdaptivePortfolioCompilerError("compiler config schema drifted")
        for name in (
            "maximum_security_weight",
            "maximum_issuer_weight",
            "tracking_error_limit_annualized",
            "absolute_active_beta_limit",
            "maximum_daily_one_way_turnover",
            "maximum_adv_participation",
            "solver_step_size",
            "numerical_tolerance",
        ):
            value = _positive(name, getattr(self, name))
            if name in {
                "maximum_security_weight",
                "maximum_issuer_weight",
                "maximum_daily_one_way_turnover",
                "maximum_adv_participation",
            } and value > 1.0:
                raise MassiveAdaptivePortfolioCompilerError(f"{name} exceeds one")
        if self.maximum_issuer_weight < self.maximum_security_weight:
            raise MassiveAdaptivePortfolioCompilerError(
                "maximum issuer weight is below the security cap"
            )
        _nonnegative(
            "uncertainty_standard_deviations",
            self.uncertainty_standard_deviations,
        )
        _nonnegative("risk_aversion", self.risk_aversion)
        _nonnegative("tail_risk_aversion", self.tail_risk_aversion)
        confidence = _finite("tail_confidence", self.tail_confidence)
        if not 0.0 < confidence < 1.0:
            raise MassiveAdaptivePortfolioCompilerError(
                "tail_confidence must lie in (0, 1)"
            )
        _positive_int("solver_max_iterations", self.solver_max_iterations)
        _positive_int("projection_max_iterations", self.projection_max_iterations)
        assert_no_adaptive_hold_semantics(self)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePortfolioCompilerInputsV1:
    """Source-bound forecast, book, cost, liquidity, and risk inputs."""

    decision_id: str
    security_ids: tuple[str, ...]
    issuer_ids: tuple[str, ...]
    bucket_expected_residual_returns: tuple[tuple[float, ...], ...]
    bucket_covariances: tuple[tuple[tuple[float, ...], ...], ...]
    pretrade_weights: tuple[float, ...]
    benchmark_weights: tuple[float, ...]
    risk_covariance: tuple[tuple[float, ...], ...]
    active_betas: tuple[float, ...]
    trailing_adv_notional: tuple[float, ...]
    entry_cost_basis_points: tuple[float, ...]
    current_exit_cost_basis_points: tuple[float, ...]
    expected_future_exit_cost_basis_points: tuple[tuple[float, ...], ...]
    buy_eligible: tuple[bool, ...]
    forced_exit: tuple[bool, ...]
    capital: float
    forecast_receipt_sha256: str
    risk_receipt_sha256: str
    cost_receipt_sha256: str
    portfolio_state_receipt_sha256: str
    eligibility_receipt_sha256: str
    tail_scenario_returns: tuple[tuple[float, ...], ...] = ()
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = "rl-quant.massive-adaptive-portfolio-compiler-input-v1"

    def validate(self) -> None:
        if self.schema != "rl-quant.massive-adaptive-portfolio-compiler-input-v1":
            raise MassiveAdaptivePortfolioCompilerError("compiler input schema drifted")
        _canonical_text("decision_id", self.decision_id)
        count = len(self.security_ids)
        if count == 0 or self.security_ids != tuple(sorted(set(self.security_ids))):
            raise MassiveAdaptivePortfolioCompilerError(
                "security IDs must be unique and sorted"
            )
        if len(self.issuer_ids) != count:
            raise MassiveAdaptivePortfolioCompilerError("issuer inventory is misaligned")
        for issuer_id in self.issuer_ids:
            _canonical_text("issuer ID", issuer_id)
        bucket_count = len(_BUCKET_IDS)
        expected = _as_float_array(
            "bucket_expected_residual_returns",
            self.bucket_expected_residual_returns,
            shape=(count, bucket_count),
        )
        bucket_covariance = _as_float_array(
            "bucket_covariances",
            self.bucket_covariances,
            shape=(count, bucket_count, bucket_count),
        )
        _validate_covariance_inventory("bucket covariance", bucket_covariance)
        pretrade = _as_float_array(
            "pretrade_weights", self.pretrade_weights, shape=(count,)
        )
        benchmark = _as_float_array(
            "benchmark_weights", self.benchmark_weights, shape=(count,)
        )
        if (pretrade < 0.0).any() or pretrade.sum() > 1.0 + 1.0e-10:
            raise MassiveAdaptivePortfolioCompilerError(
                "pretrade risky weights must define a long-only book"
            )
        if (benchmark < 0.0).any() or benchmark.sum() > 1.0 + 1.0e-10:
            raise MassiveAdaptivePortfolioCompilerError(
                "benchmark risky weights must define a long-only book"
            )
        risk = _as_float_array(
            "risk_covariance", self.risk_covariance, shape=(count, count)
        )
        _validate_covariance_inventory("risk covariance", risk[None, :, :])
        for name, value in (
            ("active_betas", self.active_betas),
            ("trailing_adv_notional", self.trailing_adv_notional),
            ("entry_cost_basis_points", self.entry_cost_basis_points),
            ("current_exit_cost_basis_points", self.current_exit_cost_basis_points),
        ):
            vector = _as_float_array(name, value, shape=(count,))
            if name != "active_betas" and (vector < 0.0).any():
                raise MassiveAdaptivePortfolioCompilerError(
                    f"{name} must be nonnegative"
                )
        if (np.asarray(self.trailing_adv_notional, dtype=np.float64) <= 0.0).any():
            raise MassiveAdaptivePortfolioCompilerError(
                "trailing ADV must be strictly positive"
            )
        future_exit = _as_float_array(
            "expected_future_exit_cost_basis_points",
            self.expected_future_exit_cost_basis_points,
            shape=(count, bucket_count),
        )
        if (future_exit < 0.0).any():
            raise MassiveAdaptivePortfolioCompilerError(
                "expected future exit costs must be nonnegative"
            )
        buy_eligible = _as_bool_array(
            "buy_eligible", self.buy_eligible, shape=(count,)
        )
        forced_exit = _as_bool_array("forced_exit", self.forced_exit, shape=(count,))
        if (buy_eligible & forced_exit).any():
            raise MassiveAdaptivePortfolioCompilerError(
                "a forced-exit security cannot be buy eligible"
            )
        _positive("capital", self.capital)
        if self.tail_scenario_returns:
            _as_float_array(
                "tail_scenario_returns",
                self.tail_scenario_returns,
                shape=(len(self.tail_scenario_returns), count),
            )
        for name in (
            "forecast_receipt_sha256",
            "risk_receipt_sha256",
            "cost_receipt_sha256",
            "portfolio_state_receipt_sha256",
            "eligibility_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256:
            raise MassiveAdaptivePortfolioCompilerError(
                "compiler input does not bind the frozen adaptive-alpha protocol"
            )
        # Reference the singleton as well as its receipt so this module cannot
        # silently run after an in-process protocol substitution.
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.validate()
        if MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.receipt_sha256 != self.protocol_receipt_sha256:
            raise MassiveAdaptivePortfolioCompilerError("adaptive protocol root drifted")
        assert_no_adaptive_hold_semantics(self)
        del expected

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return semantic_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePortfolioDecisionV1:
    """One feasible first-step decision and its numerical evidence."""

    decision_id: str
    security_ids: tuple[str, ...]
    repaired_pretrade_weights: tuple[float, ...]
    target_weights: tuple[float, ...]
    cash_weight: float
    discretionary_buy_weights: tuple[float, ...]
    discretionary_sell_weights: tuple[float, ...]
    forced_exit_weights: tuple[float, ...]
    best_buy_bucket_ids: tuple[str, ...]
    best_existing_position_bucket_ids: tuple[str, ...]
    buy_net_values: tuple[float, ...]
    existing_position_net_values: tuple[float, ...]
    discretionary_one_way_turnover: float
    forced_one_way_turnover: float
    annualized_tracking_error: float
    active_beta: float
    maximum_intended_participation: float
    objective_value: float
    expected_value_component: float
    current_exit_cost_component: float
    risk_penalty_component: float
    tail_penalty_component: float
    solver_iterations: int
    projection_iterations: int
    primal_residual: float
    dual_residual_surrogate: float
    kkt_residual_surrogate: float
    converged: bool
    input_receipt_sha256: str
    config_receipt_sha256: str
    semantic_receipt_sha256: str
    economic_optimization_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    solver: str = MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SOLVER
    schema: str = MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SCHEMA:
            raise MassiveAdaptivePortfolioCompilerError("decision schema drifted")
        if self.solver != MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SOLVER:
            raise MassiveAdaptivePortfolioCompilerError("decision solver drifted")
        _canonical_text("decision_id", self.decision_id)
        count = len(self.security_ids)
        if count == 0 or self.security_ids != tuple(sorted(set(self.security_ids))):
            raise MassiveAdaptivePortfolioCompilerError(
                "decision security inventory is invalid"
            )
        for name in (
            "repaired_pretrade_weights",
            "target_weights",
            "discretionary_buy_weights",
            "discretionary_sell_weights",
            "forced_exit_weights",
            "buy_net_values",
            "existing_position_net_values",
        ):
            vector = _as_float_array(name, getattr(self, name), shape=(count,))
            if name not in {"buy_net_values", "existing_position_net_values"} and (
                vector < -1.0e-12
            ).any():
                raise MassiveAdaptivePortfolioCompilerError(
                    f"{name} contains a negative economic weight"
                )
        if len(self.best_buy_bucket_ids) != count or len(
            self.best_existing_position_bucket_ids
        ) != count:
            raise MassiveAdaptivePortfolioCompilerError(
                "decision bucket diagnostics are misaligned"
            )
        if any(value not in _BUCKET_IDS for value in self.best_buy_bucket_ids) or any(
            value not in _BUCKET_IDS
            for value in self.best_existing_position_bucket_ids
        ):
            raise MassiveAdaptivePortfolioCompilerError(
                "decision references an unknown forecast bucket"
            )
        for name in (
            "cash_weight",
            "discretionary_one_way_turnover",
            "forced_one_way_turnover",
            "annualized_tracking_error",
            "maximum_intended_participation",
            "current_exit_cost_component",
            "risk_penalty_component",
            "tail_penalty_component",
            "primal_residual",
            "dual_residual_surrogate",
            "kkt_residual_surrogate",
        ):
            _nonnegative(name, getattr(self, name))
        for name in ("objective_value", "expected_value_component", "active_beta"):
            _finite(name, getattr(self, name))
        _positive_int("solver_iterations", self.solver_iterations)
        _positive_int("projection_iterations", self.projection_iterations)
        if not self.converged:
            raise MassiveAdaptivePortfolioCompilerError(
                "a nonconverged decision cannot validate"
            )
        if any(
            (
                self.economic_optimization_authorized,
                self.profitability_reporting_authorized,
                self.lockbox_access_authorized,
                self.reinforcement_learning_authorized,
            )
        ):
            raise MassiveAdaptivePortfolioCompilerError(
                "the engineering compiler cannot authorize downstream research"
            )
        _digest("input_receipt_sha256", self.input_receipt_sha256)
        _digest("config_receipt_sha256", self.config_receipt_sha256)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptivePortfolioCompilerError("decision receipt differs")
        assert_no_adaptive_hold_semantics(self)


def _validate_covariance_inventory(
    name: str,
    matrices: NDArray[np.float64],
) -> None:
    for index, matrix in enumerate(matrices):
        if not np.allclose(matrix, matrix.T, atol=1.0e-12, rtol=1.0e-12):
            raise MassiveAdaptivePortfolioCompilerError(
                f"{name} {index} is not symmetric"
            )
        eigenvalues = np.linalg.eigvalsh(matrix)
        if float(eigenvalues.min(initial=0.0)) < -1.0e-10:
            raise MassiveAdaptivePortfolioCompilerError(
                f"{name} {index} is not positive semidefinite"
            )


def _project_simplex_equal(
    values: NDArray[np.float64],
    radius: float,
) -> NDArray[np.float64]:
    if radius <= 0.0:
        return np.zeros_like(values)
    if float(values.sum()) <= radius:
        return values.copy()
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indexes = np.arange(1, values.size + 1, dtype=np.float64)
    eligible = ordered - (cumulative - radius) / indexes > 0.0
    rho = int(np.flatnonzero(eligible)[-1])
    threshold = float((cumulative[rho] - radius) / (rho + 1))
    return np.maximum(values - threshold, 0.0)


def _project_positive_change(
    values: NDArray[np.float64],
    anchor: NDArray[np.float64],
    limit: float,
) -> NDArray[np.float64]:
    delta = values - anchor
    positive = np.maximum(delta, 0.0)
    if float(positive.sum()) <= limit:
        return values.copy()
    projected = _project_simplex_equal(positive, limit)
    return anchor + np.where(delta > 0.0, projected, delta)


def _project_negative_change(
    values: NDArray[np.float64],
    anchor: NDArray[np.float64],
    limit: float,
) -> NDArray[np.float64]:
    return -_project_positive_change(-values, -anchor, limit)


@dataclass(slots=True)
class _FeasibleProjector:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    anchor: NDArray[np.float64]
    benchmark: NDArray[np.float64]
    issuer_groups: tuple[NDArray[np.int64], ...]
    issuer_cap: float
    turnover_limit: float
    beta: NDArray[np.float64]
    beta_limit: float
    covariance: NDArray[np.float64]
    tracking_error_limit_annualized: float
    max_iterations: int
    tolerance: float
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]

    @classmethod
    def build(
        cls,
        *,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        anchor: NDArray[np.float64],
        benchmark: NDArray[np.float64],
        issuer_ids: tuple[str, ...],
        issuer_cap: float,
        turnover_limit: float,
        beta: NDArray[np.float64],
        beta_limit: float,
        covariance: NDArray[np.float64],
        tracking_error_limit_annualized: float,
        max_iterations: int,
        tolerance: float,
    ) -> _FeasibleProjector:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        groups = tuple(
            np.asarray(
                [index for index, value in enumerate(issuer_ids) if value == issuer],
                dtype=np.int64,
            )
            for issuer in sorted(set(issuer_ids))
        )
        return cls(
            lower=lower,
            upper=upper,
            anchor=anchor,
            benchmark=benchmark,
            issuer_groups=groups,
            issuer_cap=issuer_cap,
            turnover_limit=turnover_limit,
            beta=beta,
            beta_limit=beta_limit,
            covariance=covariance,
            tracking_error_limit_annualized=tracking_error_limit_annualized,
            max_iterations=max_iterations,
            tolerance=tolerance,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    def _project_issuer_caps(
        self,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        projected = values.copy()
        for indexes in self.issuer_groups:
            total = float(projected[indexes].sum())
            if total > self.issuer_cap:
                projected[indexes] -= (total - self.issuer_cap) / indexes.size
        return projected

    def _project_risky_sum(
        self,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        total = float(values.sum())
        if total <= 1.0:
            return values.copy()
        return values - (total - 1.0) / values.size

    def _project_beta(
        self,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        loading_norm = float(self.beta @ self.beta)
        if loading_norm <= self.tolerance:
            return values.copy()
        exposure = float(self.beta @ (values - self.benchmark))
        bounded = float(np.clip(exposure, -self.beta_limit, self.beta_limit))
        return values + ((bounded - exposure) / loading_norm) * self.beta

    def _project_tracking_error(
        self,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        radius = self.tracking_error_limit_annualized / _SQRT_SESSIONS_PER_YEAR
        active = values - self.benchmark
        coordinates = self.eigenvectors.T @ active
        quadratic = float(np.sum(self.eigenvalues * coordinates * coordinates))
        if quadratic <= radius * radius + self.tolerance:
            return values.copy()
        positive = self.eigenvalues > self.tolerance
        if not bool(positive.any()):
            return values.copy()

        def residual(multiplier: float) -> float:
            scaled = coordinates / (1.0 + multiplier * self.eigenvalues)
            return float(np.sum(self.eigenvalues * scaled * scaled) - radius * radius)

        low = 0.0
        high = 1.0
        while residual(high) > 0.0:
            high *= 2.0
            if not math.isfinite(high):
                raise MassiveAdaptivePortfolioCompilerError(
                    "tracking-error projection failed to bracket its root"
                )
        for _ in range(80):
            middle = 0.5 * (low + high)
            if residual(middle) > 0.0:
                low = middle
            else:
                high = middle
        scaled = coordinates / (1.0 + high * self.eigenvalues)
        return self.benchmark + self.eigenvectors @ scaled

    def _projections(
        self,
    ) -> tuple[Callable[[NDArray[np.float64]], NDArray[np.float64]], ...]:
        return (
            lambda values: np.clip(values, self.lower, self.upper),
            self._project_risky_sum,
            self._project_issuer_caps,
            lambda values: _project_positive_change(
                values, self.anchor, self.turnover_limit
            ),
            lambda values: _project_negative_change(
                values, self.anchor, self.turnover_limit
            ),
            self._project_beta,
            self._project_tracking_error,
        )

    def primal_residual(self, values: NDArray[np.float64]) -> float:
        active = values - self.benchmark
        annual_tracking_error = _SQRT_SESSIONS_PER_YEAR * math.sqrt(
            max(float(active @ self.covariance @ active), 0.0)
        )
        delta = values - self.anchor
        violations = [
            float(np.max(self.lower - values, initial=0.0)),
            float(np.max(values - self.upper, initial=0.0)),
            float(values.sum() - 1.0),
            float(np.maximum(delta, 0.0).sum() - self.turnover_limit),
            float(np.maximum(-delta, 0.0).sum() - self.turnover_limit),
            abs(float(self.beta @ active)) - self.beta_limit,
            annual_tracking_error - self.tracking_error_limit_annualized,
        ]
        violations.extend(
            float(values[indexes].sum() - self.issuer_cap)
            for indexes in self.issuer_groups
        )
        return max(0.0, *violations)

    def project(
        self,
        values: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], int]:
        projections = self._projections()
        corrections = [np.zeros_like(values) for _ in projections]
        current = values.astype(np.float64, copy=True)
        for iteration in range(1, self.max_iterations + 1):
            previous = current.copy()
            for index, projection in enumerate(projections):
                shifted = current + corrections[index]
                projected = projection(shifted)
                corrections[index] = shifted - projected
                current = projected
            movement = float(np.max(np.abs(current - previous), initial=0.0))
            if movement <= self.tolerance and self.primal_residual(current) <= max(
                self.tolerance * 10.0, 1.0e-11
            ):
                return current, iteration
        raise MassiveAdaptivePortfolioCompilerError(
            "portfolio constraints did not admit a converged projection"
        )


@dataclass(frozen=True, slots=True)
class _ObjectiveComponents:
    objective: float
    expected_value: float
    current_exit_cost: float
    risk_penalty: float
    tail_penalty: float


def _tail_cvar_and_gradient(
    weights: NDArray[np.float64],
    scenarios: NDArray[np.float64],
    confidence: float,
) -> tuple[float, NDArray[np.float64]]:
    if scenarios.shape[0] == 0:
        return 0.0, np.zeros_like(weights)
    losses = -(scenarios @ weights)
    tail_count = max(1, math.ceil((1.0 - confidence) * scenarios.shape[0]))
    worst = np.argsort(-losses, kind="stable")[:tail_count]
    cvar = float(np.mean(losses[worst]))
    utility_gradient = np.mean(scenarios[worst], axis=0)
    return cvar, utility_gradient


def _objective_components(
    weights: NDArray[np.float64],
    *,
    anchor: NDArray[np.float64],
    buy_value: NDArray[np.float64],
    existing_value: NDArray[np.float64],
    current_exit_cost: NDArray[np.float64],
    benchmark: NDArray[np.float64],
    covariance: NDArray[np.float64],
    risk_aversion: float,
    scenarios: NDArray[np.float64],
    tail_risk_aversion: float,
    tail_confidence: float,
) -> _ObjectiveComponents:
    retained = np.minimum(weights, anchor)
    purchases = np.maximum(weights - anchor, 0.0)
    sales = np.maximum(anchor - weights, 0.0)
    expected_value = float(existing_value @ retained + buy_value @ purchases)
    exit_cost = float(current_exit_cost @ sales)
    active = weights - benchmark
    risk_penalty = float(risk_aversion * (active @ covariance @ active))
    cvar, _ = _tail_cvar_and_gradient(weights, scenarios, tail_confidence)
    tail_penalty = float(tail_risk_aversion * cvar)
    return _ObjectiveComponents(
        objective=expected_value - exit_cost - risk_penalty - tail_penalty,
        expected_value=expected_value,
        current_exit_cost=exit_cost,
        risk_penalty=risk_penalty,
        tail_penalty=tail_penalty,
    )


def _objective_gradient(
    weights: NDArray[np.float64],
    *,
    anchor: NDArray[np.float64],
    buy_value: NDArray[np.float64],
    existing_value: NDArray[np.float64],
    current_exit_cost: NDArray[np.float64],
    benchmark: NDArray[np.float64],
    covariance: NDArray[np.float64],
    risk_aversion: float,
    scenarios: NDArray[np.float64],
    tail_risk_aversion: float,
    tail_confidence: float,
    tolerance: float,
) -> NDArray[np.float64]:
    active = weights - benchmark
    smooth = -2.0 * risk_aversion * (covariance @ active)
    _, tail_utility_gradient = _tail_cvar_and_gradient(
        weights, scenarios, tail_confidence
    )
    smooth += tail_risk_aversion * tail_utility_gradient

    below = weights < anchor - tolerance
    above = weights > anchor + tolerance
    piecewise = np.empty_like(weights)
    piecewise[below] = existing_value[below] + current_exit_cost[below]
    piecewise[above] = buy_value[above]
    at_anchor = ~(below | above)
    upward = smooth + buy_value
    downward = smooth + existing_value + current_exit_cost
    piecewise[at_anchor] = np.where(
        upward[at_anchor] > 0.0,
        buy_value[at_anchor],
        np.where(
            downward[at_anchor] < 0.0,
            existing_value[at_anchor] + current_exit_cost[at_anchor],
            -smooth[at_anchor],
        ),
    )
    return smooth + piecewise


def _forecast_values(
    expected: NDArray[np.float64],
    bucket_covariance: NDArray[np.float64],
    entry_cost: NDArray[np.float64],
    future_exit_cost: NDArray[np.float64],
    uncertainty_standard_deviations: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    count, bucket_count = expected.shape
    cumulative = np.cumsum(expected, axis=1)
    variance = np.zeros((count, bucket_count), dtype=np.float64)
    for bucket_index in range(bucket_count):
        slab = bucket_covariance[:, : bucket_index + 1, : bucket_index + 1]
        variance[:, bucket_index] = slab.sum(axis=(1, 2))
    if float(variance.min(initial=0.0)) < -1.0e-10:
        raise MassiveAdaptivePortfolioCompilerError(
            "cumulative bucket uncertainty became negative"
        )
    adjusted = cumulative - uncertainty_standard_deviations * np.sqrt(
        np.maximum(variance, 0.0)
    )
    existing_curve = adjusted - future_exit_cost / 10_000.0
    buy_curve = existing_curve - entry_cost[:, None] / 10_000.0
    existing_index = np.argmax(existing_curve, axis=1).astype(np.int64)
    buy_index = np.argmax(buy_curve, axis=1).astype(np.int64)
    row = np.arange(count, dtype=np.int64)
    return (
        buy_curve[row, buy_index],
        existing_curve[row, existing_index],
        buy_index,
        existing_index,
    )


def compile_massive_adaptive_portfolio_v1(
    inputs: MassiveAdaptivePortfolioCompilerInputsV1,
    *,
    config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptivePortfolioDecisionV1:
    """Compile one feasible first action from the current no-duration state.

    Capacity constraints enter the optimization itself.  Capital that cannot
    reach the highest-value security can therefore move to another feasible
    security or remain in CASH; there is no post-hoc clipping step.
    """

    inputs.validate()
    resolved = config or MassiveAdaptivePortfolioCompilerConfigV1()
    resolved.validate()
    count = len(inputs.security_ids)
    bucket_count = len(_BUCKET_IDS)
    expected = _as_float_array(
        "bucket_expected_residual_returns",
        inputs.bucket_expected_residual_returns,
        shape=(count, bucket_count),
    )
    bucket_covariance = _as_float_array(
        "bucket_covariances",
        inputs.bucket_covariances,
        shape=(count, bucket_count, bucket_count),
    )
    pretrade = _as_float_array(
        "pretrade_weights", inputs.pretrade_weights, shape=(count,)
    )
    benchmark = _as_float_array(
        "benchmark_weights", inputs.benchmark_weights, shape=(count,)
    )
    covariance = _as_float_array(
        "risk_covariance", inputs.risk_covariance, shape=(count, count)
    )
    beta = _as_float_array("active_betas", inputs.active_betas, shape=(count,))
    adv = _as_float_array(
        "trailing_adv_notional", inputs.trailing_adv_notional, shape=(count,)
    )
    entry_cost = _as_float_array(
        "entry_cost_basis_points", inputs.entry_cost_basis_points, shape=(count,)
    )
    current_exit_cost = _as_float_array(
        "current_exit_cost_basis_points",
        inputs.current_exit_cost_basis_points,
        shape=(count,),
    ) / 10_000.0
    future_exit_cost = _as_float_array(
        "expected_future_exit_cost_basis_points",
        inputs.expected_future_exit_cost_basis_points,
        shape=(count, bucket_count),
    )
    buy_eligible = _as_bool_array(
        "buy_eligible", inputs.buy_eligible, shape=(count,)
    )
    forced_exit = _as_bool_array("forced_exit", inputs.forced_exit, shape=(count,))
    scenarios = (
        _as_float_array(
            "tail_scenario_returns",
            inputs.tail_scenario_returns,
            shape=(len(inputs.tail_scenario_returns), count),
        )
        if inputs.tail_scenario_returns
        else np.empty((0, count), dtype=np.float64)
    )

    buy_value, existing_value, buy_index, existing_index = _forecast_values(
        expected,
        bucket_covariance,
        entry_cost,
        future_exit_cost,
        resolved.uncertainty_standard_deviations,
    )

    forced_weights = np.where(forced_exit, pretrade, 0.0)
    anchor = np.where(forced_exit, 0.0, pretrade)
    capacity = resolved.maximum_adv_participation * adv / float(inputs.capital)
    lower = np.maximum(0.0, anchor - capacity)
    upper = np.minimum(resolved.maximum_security_weight, anchor + capacity)
    upper = np.where(buy_eligible, upper, np.minimum(upper, anchor))
    lower = np.where(forced_exit, 0.0, lower)
    upper = np.where(forced_exit, 0.0, upper)
    if bool((lower > upper + resolved.numerical_tolerance).any()):
        raise MassiveAdaptivePortfolioCompilerError(
            "security caps and capacity cannot repair the pretrade book"
        )

    projector = _FeasibleProjector.build(
        lower=lower,
        upper=upper,
        anchor=anchor,
        benchmark=benchmark,
        issuer_ids=inputs.issuer_ids,
        issuer_cap=resolved.maximum_issuer_weight,
        turnover_limit=resolved.maximum_daily_one_way_turnover,
        beta=beta,
        beta_limit=resolved.absolute_active_beta_limit,
        covariance=covariance,
        tracking_error_limit_annualized=resolved.tracking_error_limit_annualized,
        max_iterations=resolved.projection_max_iterations,
        tolerance=resolved.numerical_tolerance,
    )
    weights, projection_iterations = projector.project(anchor)
    components = _objective_components(
        weights,
        anchor=anchor,
        buy_value=buy_value,
        existing_value=existing_value,
        current_exit_cost=current_exit_cost,
        benchmark=benchmark,
        covariance=covariance,
        risk_aversion=resolved.risk_aversion,
        scenarios=scenarios,
        tail_risk_aversion=resolved.tail_risk_aversion,
        tail_confidence=resolved.tail_confidence,
    )

    converged = False
    iteration = 0
    for iteration in range(1, resolved.solver_max_iterations + 1):
        gradient = _objective_gradient(
            weights,
            anchor=anchor,
            buy_value=buy_value,
            existing_value=existing_value,
            current_exit_cost=current_exit_cost,
            benchmark=benchmark,
            covariance=covariance,
            risk_aversion=resolved.risk_aversion,
            scenarios=scenarios,
            tail_risk_aversion=resolved.tail_risk_aversion,
            tail_confidence=resolved.tail_confidence,
            tolerance=resolved.numerical_tolerance,
        )
        local_step = resolved.solver_step_size
        candidate = weights
        candidate_components = components
        accepted = False
        for _ in range(24):
            candidate, used = projector.project(weights + local_step * gradient)
            projection_iterations += used
            candidate_components = _objective_components(
                candidate,
                anchor=anchor,
                buy_value=buy_value,
                existing_value=existing_value,
                current_exit_cost=current_exit_cost,
                benchmark=benchmark,
                covariance=covariance,
                risk_aversion=resolved.risk_aversion,
                scenarios=scenarios,
                tail_risk_aversion=resolved.tail_risk_aversion,
                tail_confidence=resolved.tail_confidence,
            )
            if candidate_components.objective >= (
                components.objective - resolved.numerical_tolerance
            ):
                accepted = True
                break
            local_step *= 0.5
        if not accepted:
            raise MassiveAdaptivePortfolioCompilerError(
                "portfolio objective could not find a nondecreasing feasible step"
            )
        movement = float(np.max(np.abs(candidate - weights), initial=0.0))
        weights = candidate
        components = candidate_components
        if movement <= resolved.numerical_tolerance:
            converged = True
            break
    if not converged:
        raise MassiveAdaptivePortfolioCompilerError(
            "portfolio objective did not converge within its frozen iteration limit"
        )

    # Dykstra's method can finish a few ULPs below an exact box boundary while
    # already satisfying the configured primal tolerance.  Canonicalize only
    # those lower-bound residues; never widen a scientific constraint or snap
    # an interior economic weight.
    boundary_tolerance = max(resolved.numerical_tolerance * 10.0, 1.0e-12)
    weights = np.where(
        np.abs(weights - lower) <= boundary_tolerance,
        lower,
        weights,
    )
    excess_risky_mass = max(float(weights.sum()) - 1.0, 0.0)
    if excess_risky_mass > 0.0:
        slack = weights - lower
        slack_index = int(np.argmax(slack))
        if slack[slack_index] + boundary_tolerance < excess_risky_mass:
            raise MassiveAdaptivePortfolioCompilerError(
                "simplex roundoff cannot be reconciled without crossing a lower bound"
            )
        weights[slack_index] -= excess_risky_mass
    components = _objective_components(
        weights,
        anchor=anchor,
        buy_value=buy_value,
        existing_value=existing_value,
        current_exit_cost=current_exit_cost,
        benchmark=benchmark,
        covariance=covariance,
        risk_aversion=resolved.risk_aversion,
        scenarios=scenarios,
        tail_risk_aversion=resolved.tail_risk_aversion,
        tail_confidence=resolved.tail_confidence,
    )

    final_gradient = _objective_gradient(
        weights,
        anchor=anchor,
        buy_value=buy_value,
        existing_value=existing_value,
        current_exit_cost=current_exit_cost,
        benchmark=benchmark,
        covariance=covariance,
        risk_aversion=resolved.risk_aversion,
        scenarios=scenarios,
        tail_risk_aversion=resolved.tail_risk_aversion,
        tail_confidence=resolved.tail_confidence,
        tolerance=resolved.numerical_tolerance,
    )
    fixed_point, used = projector.project(
        weights + resolved.solver_step_size * final_gradient
    )
    projection_iterations += used
    dual_residual = float(
        np.max(np.abs(fixed_point - weights), initial=0.0)
        / resolved.solver_step_size
    )
    primal_residual = projector.primal_residual(weights)
    kkt_residual = max(primal_residual, dual_residual)
    if primal_residual > max(resolved.numerical_tolerance * 20.0, 2.0e-8):
        raise MassiveAdaptivePortfolioCompilerError(
            "compiled portfolio violates a hard feasibility constraint"
        )

    delta = weights - anchor
    buys = np.maximum(delta, 0.0)
    sells = np.maximum(-delta, 0.0)
    active = weights - benchmark
    annual_tracking_error = _SQRT_SESSIONS_PER_YEAR * math.sqrt(
        max(float(active @ covariance @ active), 0.0)
    )
    active_beta = float(beta @ active)
    intended_participation = np.abs(delta) * float(inputs.capital) / adv
    body = {
        "decision_id": inputs.decision_id,
        "security_ids": inputs.security_ids,
        "repaired_pretrade_weights": _tuple_float(anchor),
        "target_weights": _tuple_float(weights),
        "cash_weight": float(max(0.0, 1.0 - weights.sum())),
        "discretionary_buy_weights": _tuple_float(buys),
        "discretionary_sell_weights": _tuple_float(sells),
        "forced_exit_weights": _tuple_float(forced_weights),
        "best_buy_bucket_ids": tuple(_BUCKET_IDS[index] for index in buy_index),
        "best_existing_position_bucket_ids": tuple(
            _BUCKET_IDS[index] for index in existing_index
        ),
        "buy_net_values": _tuple_float(buy_value),
        "existing_position_net_values": _tuple_float(existing_value),
        "discretionary_one_way_turnover": float(
            max(float(buys.sum()), float(sells.sum()))
        ),
        "forced_one_way_turnover": float(forced_weights.sum()),
        "annualized_tracking_error": float(annual_tracking_error),
        "active_beta": active_beta,
        "maximum_intended_participation": float(
            intended_participation.max(initial=0.0)
        ),
        "objective_value": components.objective,
        "expected_value_component": components.expected_value,
        "current_exit_cost_component": components.current_exit_cost,
        "risk_penalty_component": components.risk_penalty,
        "tail_penalty_component": components.tail_penalty,
        "solver_iterations": iteration,
        "projection_iterations": projection_iterations,
        "primal_residual": primal_residual,
        "dual_residual_surrogate": dual_residual,
        "kkt_residual_surrogate": kkt_residual,
        "converged": True,
        "input_receipt_sha256": inputs.receipt_sha256,
        "config_receipt_sha256": resolved.receipt_sha256,
        "economic_optimization_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "solver": MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SOLVER,
        "schema": MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SCHEMA,
    }
    decision = MassiveAdaptivePortfolioDecisionV1(
        **body,
        semantic_receipt_sha256=semantic_sha256(body),
    )
    decision.validate()
    return decision


__all__ = [
    "MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SOLVER",
    "MassiveAdaptivePortfolioCompilerConfigV1",
    "MassiveAdaptivePortfolioCompilerError",
    "MassiveAdaptivePortfolioCompilerInputsV1",
    "MassiveAdaptivePortfolioDecisionV1",
    "compile_massive_adaptive_portfolio_v1",
]
