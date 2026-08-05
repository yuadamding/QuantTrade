"""Versioned M03R v5 inference and finite-control diagnostics.

The frozen v3 and v4 evaluators remain unchanged.  This module owns corrected
active-alpha, active-beta-equivalence, uncertainty, and control-tail semantics
for the M03R v5 generation only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from statistics import NormalDist
from typing import Any

import numpy as np

from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN,
    M03R_PROTOCOL_GENERATION,
    resolve_m03r_v5_setting,
)

M03R_EVALUATION_SCHEMA = "rl-quant.hold30.m03r-evaluation-v5"
M03R_INFERENCE_PLAN_SCHEMA = "rl-quant.hold30.m03r-inference-plan-v2"
M03R_SOURCE_ARRAY_SCHEMA = "rl-quant.hold30.m03r-source-arrays-v1"
M03R_COMMON_EVALUATOR_INPUT_SCHEMA = (
    "rl-quant.hold30.m03r-common-evaluator-inputs-v1"
)
M03R_CANDIDATE_POLICY_RETURNS_SCHEMA = (
    "rl-quant.hold30.m03r-candidate-policy-returns-v1"
)
M03R_OUTER_FOLDS = 6
M03R_SCORE_SESSIONS_PER_FOLD = 63
M03R_HAC_LAGS = (10, 21, 30, 42)
M03R_PRIMARY_HAC_LAG = 30
M03R_PRIMARY_BOOTSTRAP_BLOCK_LENGTH = 21
M03R_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS = (10, 30)
M03R_MARKET_FACTOR_NAME = "PIT_CAP_MARKET_EXCESS"
M03R_SUPPORTED_FACTOR_RETURN_CONVENTIONS = frozenset(
    {
        "daily-simple-long-short-return",
        "daily-simple-excess-return",
    }
)
M03R_PROMOTION_BLOCKERS = (
    "multiplicity_adjusted_factor_alpha_family_not_bound",
    "outer_data_role_and_access_receipts_not_bound",
    "empirical_capacity_and_closed_loop_40bp_not_bound",
)
_DIGEST_CHARS = frozenset("0123456789abcdef")
_BOOTSTRAP_DOMAIN = b"rl-quant.hold30.m03r-moving-block-v1\x00"
_SOURCE_ARRAY_DOMAIN = b"rl-quant.hold30.m03r-source-arrays-v1\x00"
_COMMON_EVALUATOR_INPUT_DOMAIN = (
    b"rl-quant.hold30.m03r-common-evaluator-inputs-v1\x00"
)
_CANDIDATE_POLICY_RETURNS_DOMAIN = (
    b"rl-quant.hold30.m03r-candidate-policy-returns-v1\x00"
)


class M03REvaluationError(ValueError):
    """M03R inference inputs or immutable bindings are invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise M03REvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise M03REvaluationError(f"{name} must be finite with shape {shape}")
    return result


def _inference_plan_semantics(
    *,
    factor_names: tuple[str, ...],
    factor_return_conventions: tuple[str, ...],
    primary_bootstrap_block_length_trading_sessions: int,
    sensitivity_bootstrap_block_lengths_trading_sessions: tuple[int, ...],
    bootstrap_replicates: int,
    bootstrap_seed_sha256: str,
    one_sided_alpha: float,
    primary_hac_lag_trading_sessions: int,
) -> dict[str, Any]:
    if len(factor_names) != len(factor_return_conventions):
        raise M03REvaluationError(
            "factor names and return conventions must have equal length"
        )
    return {
        "schema": M03R_INFERENCE_PLAN_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "outer_fold_count": M03R_OUTER_FOLDS,
        "score_sessions_per_fold": M03R_SCORE_SESSIONS_PER_FOLD,
        "policy_net_return_convention": "daily-simple-net-return",
        "benchmark_net_return_convention": "daily-simple-net-return",
        "risk_free_return_convention": "daily-simple-return",
        "market_total_return_convention": "daily-simple-total-return",
        "market_excess_definition": "market_total_return-minus-risk_free_return",
        "market_factor_name": M03R_MARKET_FACTOR_NAME,
        "factor_return_conventions": [
            {"factor_name": name, "return_convention": convention}
            for name, convention in zip(
                factor_names,
                factor_return_conventions,
                strict=True,
            )
        ],
        "active_beta_return_definition": "policy_net_return-minus-benchmark_net_return",
        "active_performance_definition": "log1p(policy_net_return)-log1p(benchmark_net_return)",
        "regression_fold_treatment": (
            "six-fold-fixed-effects-effect-coded-common-intercept"
        ),
        "regression_common_intercept_definition": (
            "equal-weight-mean-of-six-fold-specific-intercepts"
        ),
        "hac_cross_fold_lag_pairs": "excluded",
        "regression_intercept_annualization": "252-times-daily-arithmetic-intercept",
        "hac_lags_trading_sessions": list(M03R_HAC_LAGS),
        "primary_hac_lag_trading_sessions": primary_hac_lag_trading_sessions,
        "primary_bootstrap_block_length_trading_sessions": (
            primary_bootstrap_block_length_trading_sessions
        ),
        "sensitivity_bootstrap_block_lengths_trading_sessions": list(
            sensitivity_bootstrap_block_lengths_trading_sessions
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed_sha256": bootstrap_seed_sha256,
        "bootstrap_resampling": "within-fold-circular-moving-block",
        "one_sided_alpha": float(one_sided_alpha),
        "quantile_method": "inverted_cdf",
    }


@dataclass(frozen=True, slots=True)
class M03RInferencePlan:
    """Content-addressed uncertainty contract frozen before outer access."""

    factor_names: tuple[str, ...]
    factor_return_conventions: tuple[str, ...]
    primary_bootstrap_block_length_trading_sessions: int
    sensitivity_bootstrap_block_lengths_trading_sessions: tuple[int, ...]
    bootstrap_replicates: int
    bootstrap_seed_sha256: str
    one_sided_alpha: float
    primary_hac_lag_trading_sessions: int
    inference_contract_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.factor_names) is not tuple
            or not self.factor_names
            or any(
                type(name) is not str or not name or name.strip() != name
                for name in self.factor_names
            )
            or len(set(self.factor_names)) != len(self.factor_names)
        ):
            raise M03REvaluationError("factor_names must be non-empty and unique")
        if M03R_MARKET_FACTOR_NAME in self.factor_names:
            raise M03REvaluationError(
                f"factor_names cannot collide with {M03R_MARKET_FACTOR_NAME}"
            )
        if (
            type(self.factor_return_conventions) is not tuple
            or len(self.factor_return_conventions) != len(self.factor_names)
            or any(
                type(convention) is not str
                or convention not in M03R_SUPPORTED_FACTOR_RETURN_CONVENTIONS
                for convention in self.factor_return_conventions
            )
        ):
            raise M03REvaluationError(
                "each factor needs one supported return convention"
            )
        if (
            type(self.primary_bootstrap_block_length_trading_sessions) is not int
            or self.primary_bootstrap_block_length_trading_sessions
            >= M03R_SCORE_SESSIONS_PER_FOLD
        ):
            raise M03REvaluationError(
                "primary bootstrap block length must be shorter than a complete fold"
            )
        if (
            self.primary_bootstrap_block_length_trading_sessions
            != M03R_PRIMARY_BOOTSTRAP_BLOCK_LENGTH
        ):
            raise M03REvaluationError(
                "primary bootstrap block length must be the frozen 21 sessions"
            )
        if (
            type(self.sensitivity_bootstrap_block_lengths_trading_sessions)
            is not tuple
            or self.sensitivity_bootstrap_block_lengths_trading_sessions
            != M03R_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS
        ):
            raise M03REvaluationError(
                "bootstrap sensitivity block lengths must be the frozen (10, 30)"
            )
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates < 1_000
        ):
            raise M03REvaluationError("bootstrap_replicates must be at least 1,000")
        if type(self.one_sided_alpha) is not float or not (
            0.0 < self.one_sided_alpha < 0.5
        ):
            raise M03REvaluationError("one_sided_alpha must lie in (0,0.5)")
        if self.primary_hac_lag_trading_sessions not in M03R_HAC_LAGS:
            raise M03REvaluationError("primary HAC lag must be one of M03R_HAC_LAGS")
        _require_digest("bootstrap_seed_sha256", self.bootstrap_seed_sha256)
        claimed = _require_digest(
            "inference_contract_sha256", self.inference_contract_sha256
        )
        expected = _sha256(self.semantics())
        if claimed != expected:
            raise M03REvaluationError(
                "inference_contract_sha256 does not bind the exact plan semantics"
            )

    def semantics(self) -> dict[str, Any]:
        """Return the exact canonical semantics bound by this plan."""

        return _inference_plan_semantics(
            factor_names=self.factor_names,
            factor_return_conventions=self.factor_return_conventions,
            primary_bootstrap_block_length_trading_sessions=(
                self.primary_bootstrap_block_length_trading_sessions
            ),
            sensitivity_bootstrap_block_lengths_trading_sessions=(
                self.sensitivity_bootstrap_block_lengths_trading_sessions
            ),
            bootstrap_replicates=self.bootstrap_replicates,
            bootstrap_seed_sha256=self.bootstrap_seed_sha256,
            one_sided_alpha=self.one_sided_alpha,
            primary_hac_lag_trading_sessions=self.primary_hac_lag_trading_sessions,
        )

    @property
    def bootstrap_block_lengths_trading_sessions(self) -> tuple[int, ...]:
        """Return primary first, followed by the frozen sensitivity blocks."""

        return (
            self.primary_bootstrap_block_length_trading_sessions,
            *self.sensitivity_bootstrap_block_lengths_trading_sessions,
        )


def build_m03r_inference_plan(
    *,
    factor_names: tuple[str, ...],
    factor_return_conventions: tuple[str, ...],
    bootstrap_replicates: int,
    bootstrap_seed_sha256: str,
    one_sided_alpha: float,
    primary_hac_lag_trading_sessions: int = M03R_PRIMARY_HAC_LAG,
    primary_bootstrap_block_length_trading_sessions: int = (
        M03R_PRIMARY_BOOTSTRAP_BLOCK_LENGTH
    ),
    sensitivity_bootstrap_block_lengths_trading_sessions: tuple[int, ...] = (
        M03R_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS
    ),
) -> M03RInferencePlan:
    """Build a plan whose identifier is derived from all inference semantics."""

    semantics = _inference_plan_semantics(
        factor_names=factor_names,
        factor_return_conventions=factor_return_conventions,
        primary_bootstrap_block_length_trading_sessions=(
            primary_bootstrap_block_length_trading_sessions
        ),
        sensitivity_bootstrap_block_lengths_trading_sessions=(
            sensitivity_bootstrap_block_lengths_trading_sessions
        ),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed_sha256=bootstrap_seed_sha256,
        one_sided_alpha=one_sided_alpha,
        primary_hac_lag_trading_sessions=primary_hac_lag_trading_sessions,
    )
    return M03RInferencePlan(
        factor_names=factor_names,
        factor_return_conventions=factor_return_conventions,
        primary_bootstrap_block_length_trading_sessions=(
            primary_bootstrap_block_length_trading_sessions
        ),
        sensitivity_bootstrap_block_lengths_trading_sessions=(
            sensitivity_bootstrap_block_lengths_trading_sessions
        ),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed_sha256=bootstrap_seed_sha256,
        one_sided_alpha=one_sided_alpha,
        primary_hac_lag_trading_sessions=primary_hac_lag_trading_sessions,
        inference_contract_sha256=_sha256(semantics),
    )


def _update_array_hash(
    digest: Any,
    *,
    name: str,
    value: np.ndarray,
) -> None:
    """Hash normalized float64 bytes together with explicit shape and identity."""

    normalized = np.ascontiguousarray(value, dtype=">f8")
    metadata = {
        "name": name,
        "shape": list(normalized.shape),
        "normalized_dtype": "big-endian-float64",
        "memory_order": "C",
    }
    encoded = _canonical_json(metadata)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(normalized.tobytes(order="C"))


def _source_arrays_sha256(
    *,
    policy: np.ndarray,
    benchmark: np.ndarray,
    risk_free: np.ndarray,
    market: np.ndarray,
    factors: np.ndarray,
    plan: M03RInferencePlan,
) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_ARRAY_DOMAIN)
    header = {
        "schema": M03R_SOURCE_ARRAY_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "inference_contract_sha256": plan.inference_contract_sha256,
    }
    encoded = _canonical_json(header)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    for name, value in (
        ("policy_net_returns", policy),
        ("benchmark_net_returns", benchmark),
        ("risk_free_returns", risk_free),
        ("market_total_returns", market),
        ("factor_returns", factors),
    ):
        _update_array_hash(digest, name=name, value=value)
    return digest.hexdigest()


def m03r_source_arrays_sha256(
    *,
    policy_net_returns: Any,
    benchmark_net_returns: Any,
    risk_free_returns: Any,
    market_total_returns: Any,
    factor_returns: Any,
    plan: M03RInferencePlan,
) -> str:
    """Compute a legacy combined array digest for non-governed diagnostics.

    Governed v5 evaluation uses separate common-input and candidate-policy
    bindings so multiple checkpoint paths can share one evaluator input set.
    """

    shape = (M03R_OUTER_FOLDS, M03R_SCORE_SESSIONS_PER_FOLD)
    return _source_arrays_sha256(
        policy=_finite_array("policy_net_returns", policy_net_returns, shape),
        benchmark=_finite_array("benchmark_net_returns", benchmark_net_returns, shape),
        risk_free=_finite_array("risk_free_returns", risk_free_returns, shape),
        market=_finite_array("market_total_returns", market_total_returns, shape),
        factors=_finite_array(
            "factor_returns",
            factor_returns,
            (*shape, len(plan.factor_names)),
        ),
        plan=plan,
    )


def _canonical_string_array(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    *,
    iso_dates: bool = False,
) -> np.ndarray:
    result = np.asarray(value, dtype=object)
    if result.shape != shape:
        raise M03REvaluationError(f"{name} must have shape {shape}")
    normalized = np.empty(shape, dtype=object)
    for index, item in np.ndenumerate(result):
        if type(item) is not str or not item or item.strip() != item or "\x00" in item:
            raise M03REvaluationError(f"{name} must contain canonical strings")
        if iso_dates:
            try:
                parsed = date.fromisoformat(item)
            except ValueError as error:
                raise M03REvaluationError(
                    f"{name} must contain canonical ISO-8601 dates"
                ) from error
            if parsed.isoformat() != item:
                raise M03REvaluationError(
                    f"{name} must contain canonical ISO-8601 dates"
                )
        normalized[index] = item
    return normalized


def _update_string_array_hash(
    digest: Any,
    *,
    name: str,
    value: np.ndarray,
) -> None:
    metadata = {
        "name": name,
        "shape": list(value.shape),
        "normalized_dtype": "canonical-utf8-string",
        "memory_order": "C",
    }
    encoded = _canonical_json(metadata)
    payload = _canonical_json(value.tolist())
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _common_evaluator_inputs_sha256(
    *,
    score_dates: np.ndarray,
    fold_ids: np.ndarray,
    benchmark: np.ndarray,
    risk_free: np.ndarray,
    market: np.ndarray,
    factors: np.ndarray,
    plan: M03RInferencePlan,
) -> str:
    digest = hashlib.sha256()
    digest.update(_COMMON_EVALUATOR_INPUT_DOMAIN)
    header = {
        "schema": M03R_COMMON_EVALUATOR_INPUT_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "inference_contract_sha256": plan.inference_contract_sha256,
    }
    encoded = _canonical_json(header)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    _update_string_array_hash(digest, name="score_dates", value=score_dates)
    _update_string_array_hash(digest, name="fold_ids", value=fold_ids)
    for name, value in (
        ("benchmark_net_returns", benchmark),
        ("risk_free_returns", risk_free),
        ("market_total_returns", market),
        ("factor_returns", factors),
    ):
        _update_array_hash(digest, name=name, value=value)
    return digest.hexdigest()


def m03r_common_evaluator_inputs_sha256(
    *,
    score_dates: Any,
    fold_ids: Any,
    benchmark_net_returns: Any,
    risk_free_returns: Any,
    market_total_returns: Any,
    factor_returns: Any,
    plan: M03RInferencePlan,
) -> str:
    """Bind dates, folds, non-policy returns, factors, and inference semantics."""

    shape = (M03R_OUTER_FOLDS, M03R_SCORE_SESSIONS_PER_FOLD)
    dates = _canonical_string_array("score_dates", score_dates, shape, iso_dates=True)
    folds = _canonical_string_array("fold_ids", fold_ids, shape)
    if any(len(set(folds[row].tolist())) != 1 for row in range(M03R_OUTER_FOLDS)):
        raise M03REvaluationError("each fold_ids row must identify exactly one fold")
    row_fold_ids = tuple(str(folds[row, 0]) for row in range(M03R_OUTER_FOLDS))
    if len(set(row_fold_ids)) != M03R_OUTER_FOLDS:
        raise M03REvaluationError("fold_ids must identify six unique folds")
    flattened_dates = [date.fromisoformat(str(item)) for item in dates.reshape(-1)]
    if any(
        current <= previous
        for previous, current in pairwise(flattened_dates)
    ):
        raise M03REvaluationError("score_dates must be globally strictly increasing")
    return _common_evaluator_inputs_sha256(
        score_dates=dates,
        fold_ids=folds,
        benchmark=_finite_array("benchmark_net_returns", benchmark_net_returns, shape),
        risk_free=_finite_array("risk_free_returns", risk_free_returns, shape),
        market=_finite_array("market_total_returns", market_total_returns, shape),
        factors=_finite_array(
            "factor_returns",
            factor_returns,
            (*shape, len(plan.factor_names)),
        ),
        plan=plan,
    )


def _candidate_policy_returns_sha256(
    *,
    policy: np.ndarray,
    common_evaluator_inputs_sha256: str,
    plan: M03RInferencePlan,
) -> str:
    digest = hashlib.sha256()
    digest.update(_CANDIDATE_POLICY_RETURNS_DOMAIN)
    header = {
        "schema": M03R_CANDIDATE_POLICY_RETURNS_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "inference_contract_sha256": plan.inference_contract_sha256,
        "common_evaluator_inputs_sha256": common_evaluator_inputs_sha256,
    }
    encoded = _canonical_json(header)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    _update_array_hash(digest, name="policy_net_returns", value=policy)
    return digest.hexdigest()


def m03r_candidate_policy_returns_sha256(
    *,
    policy_net_returns: Any,
    common_evaluator_inputs_sha256: str,
    plan: M03RInferencePlan,
) -> str:
    """Bind one candidate policy path to the exact common evaluator inputs."""

    shape = (M03R_OUTER_FOLDS, M03R_SCORE_SESSIONS_PER_FOLD)
    common_digest = _require_digest(
        "common_evaluator_inputs_sha256", common_evaluator_inputs_sha256
    )
    return _candidate_policy_returns_sha256(
        policy=_finite_array("policy_net_returns", policy_net_returns, shape),
        common_evaluator_inputs_sha256=common_digest,
        plan=plan,
    )


def finite_control_diagnostic(
    target_statistic: float,
    control_statistics: Any,
) -> dict[str, Any]:
    """Correct plus-one tail for the M03R 64-control diagnostic.

    This is explicitly a matched-placebo diagnostic.  It is not described as
    exact randomization inference when controls were selected using realized
    risk or holding statistics.
    """

    controls = _finite_array("control_statistics", control_statistics, (64,))
    if not math.isfinite(float(target_statistic)):
        raise M03REvaluationError("target_statistic must be finite")
    count_ge = int(np.count_nonzero(controls >= float(target_statistic)))
    return {
        "schema": "rl-quant.hold30.m03r-matched-placebo-v1",
        "diagnostic_kind": "matched-placebo-diagnostic",
        "exact_randomization_inference": False,
        "control_count": 64,
        "ties_count_against_candidate": True,
        "control_count_greater_than_or_equal_to_candidate": count_ge,
        "finite_control_p_value": float((1 + count_ge) / 65),
        "passes_nominal_5_percent_point_gate": count_ge <= 2,
    }


def _fold_fixed_effect_design(regressors: np.ndarray) -> np.ndarray:
    """Build a balanced, effect-coded six-fold regression design.

    The first coefficient is the equal-weight mean of the six fold-specific
    intercepts.  Five effect columns identify folds 0--4; fold 5 receives
    ``-1`` in every effect column, imposing a zero-sum fold-effect constraint.
    Regressor slopes therefore use only within-fold covariation while retaining
    one scientifically interpretable common intercept.
    """

    expected_prefix = (M03R_OUTER_FOLDS, M03R_SCORE_SESSIONS_PER_FOLD)
    if regressors.ndim != 3 or regressors.shape[:2] != expected_prefix:
        raise M03REvaluationError(
            "fold-fixed-effect regressors must have shape (6, 63, K)"
        )
    flattened = regressors.reshape(-1, regressors.shape[-1])
    fold_effects = np.zeros(
        (flattened.shape[0], M03R_OUTER_FOLDS - 1),
        dtype=np.float64,
    )
    for fold in range(M03R_OUTER_FOLDS - 1):
        start = fold * M03R_SCORE_SESSIONS_PER_FOLD
        stop = start + M03R_SCORE_SESSIONS_PER_FOLD
        fold_effects[start:stop, fold] = 1.0
    fold_effects[-M03R_SCORE_SESSIONS_PER_FOLD :, :] = -1.0
    return np.column_stack(
        (np.ones(flattened.shape[0]), flattened, fold_effects)
    )


def _fold_fixed_effect_coefficients(
    dependent: np.ndarray,
    regressors: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    expected = (M03R_OUTER_FOLDS, M03R_SCORE_SESSIONS_PER_FOLD)
    if dependent.shape != expected:
        raise M03REvaluationError(
            f"{context} dependent array must have shape {expected}"
        )
    design = _fold_fixed_effect_design(regressors)
    coefficients = _full_rank_coefficients(
        design,
        dependent.reshape(-1),
        context=context,
    )
    return coefficients, design


def _regression(
    dependent: np.ndarray,
    regressors: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, Any]:
    coefficients, design = _fold_fixed_effect_coefficients(
        dependent,
        regressors,
        context="primary fold-fixed-effect regression",
    )
    flattened_dependent = dependent.reshape(-1)
    residual = flattened_dependent - design @ coefficients
    bread = np.linalg.pinv(design.T @ design)
    hac: dict[str, Any] = {}
    for lag in M03R_HAC_LAGS:
        scores = (design * residual[:, None]).reshape(
            M03R_OUTER_FOLDS,
            M03R_SCORE_SESSIONS_PER_FOLD,
            design.shape[1],
        )
        meat = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
        for fold_scores in scores:
            meat += fold_scores.T @ fold_scores
        for offset in range(
            1,
            min(lag, M03R_SCORE_SESSIONS_PER_FOLD - 1) + 1,
        ):
            weight = 1.0 - offset / (lag + 1.0)
            for fold_scores in scores:
                cross = fold_scores[offset:].T @ fold_scores[:-offset]
                meat += weight * (cross + cross.T)
        covariance = bread @ meat @ bread
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        t_values = np.divide(
            coefficients,
            standard_errors,
            out=np.full_like(coefficients, np.nan),
            where=standard_errors > 0.0,
        )
        loading_se = {
            name: float(value)
            for name, value in zip(
                names,
                standard_errors[1 : 1 + len(names)],
                strict=True,
            )
        }
        loading_t = {
            name: float(value) if math.isfinite(float(value)) else None
            for name, value in zip(
                names,
                t_values[1 : 1 + len(names)],
                strict=True,
            )
        }
        hac[str(lag)] = {
            "cross_fold_lag_pairs": 0,
            "alpha_se_daily": float(standard_errors[0]),
            "alpha_se_annualized_arithmetic": float(252.0 * standard_errors[0]),
            "alpha_t": float(t_values[0])
            if math.isfinite(float(t_values[0]))
            else None,
            "alpha_two_sided_normal_p": (
                2.0 * (1.0 - NormalDist().cdf(abs(float(t_values[0]))))
                if math.isfinite(float(t_values[0]))
                else None
            ),
            "loading_se": loading_se,
            "loading_t": loading_t,
        }
    return {
        "alpha_daily": float(coefficients[0]),
        "alpha_annualized_arithmetic": float(252.0 * coefficients[0]),
        "loadings": {
            name: float(value)
            for name, value in zip(
                names,
                coefficients[1 : 1 + len(names)],
                strict=True,
            )
        },
        "residual_volatility_annualized": float(
            math.sqrt(252.0) * np.std(residual, ddof=1)
        ),
        "fold_fixed_effect_contract": (
            "six-fold-effect-coded-zero-sum-common-intercept"
        ),
        "fold_effects": [
            *(
                float(value)
                for value in coefficients[
                    1 + len(names) : 1 + len(names) + M03R_OUTER_FOLDS - 1
                ]
            ),
            float(
                -np.sum(
                    coefficients[
                        1 + len(names) : 1 + len(names) + M03R_OUTER_FOLDS - 1
                    ]
                )
            ),
        ],
        "hac": hac,
    }


def _full_rank_coefficients(
    design: np.ndarray,
    dependent: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    if design.ndim != 2 or dependent.ndim != 1 or design.shape[0] != dependent.size:
        raise M03REvaluationError(f"{context} arrays are misaligned")
    rows, columns = design.shape
    if rows <= columns:
        raise M03REvaluationError(
            f"{context} design is underdetermined: {rows} rows for {columns} columns"
        )
    if int(np.linalg.matrix_rank(design)) != columns:
        raise M03REvaluationError(f"{context} design is rank deficient")
    coefficients = np.linalg.lstsq(design, dependent, rcond=None)[0]
    return np.asarray(coefficients)


def _block_indices(
    seed: bytes,
    replicate: int,
    fold: int,
    block_length: int,
) -> np.ndarray:
    needed = M03R_SCORE_SESSIONS_PER_FOLD
    material = (
        _BOOTSTRAP_DOMAIN
        + seed
        + replicate.to_bytes(8, "big")
        + fold.to_bytes(2, "big")
        + block_length.to_bytes(2, "big")
    )
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    )
    starts = rng.integers(0, needed, size=math.ceil(needed / block_length))
    return np.concatenate(
        [(start + np.arange(block_length, dtype=np.int64)) % needed for start in starts]
    )[:needed]


def evaluate_m03r_inference(
    *,
    setting_id: str,
    score_dates: Any,
    fold_ids: Any,
    policy_net_returns: Any,
    benchmark_net_returns: Any,
    risk_free_returns: Any,
    market_total_returns: Any,
    factor_returns: Any,
    plan: M03RInferencePlan,
    common_evaluator_inputs_sha256: str,
    candidate_policy_returns_sha256: str,
) -> dict[str, Any]:
    """Recompute active beta, active factor alpha, and chronological uncertainty."""

    setting = resolve_m03r_v5_setting(setting_id)
    shape = (M03R_OUTER_FOLDS, M03R_SCORE_SESSIONS_PER_FOLD)
    policy = _finite_array("policy_net_returns", policy_net_returns, shape)
    benchmark = _finite_array("benchmark_net_returns", benchmark_net_returns, shape)
    risk_free = _finite_array("risk_free_returns", risk_free_returns, shape)
    market = _finite_array("market_total_returns", market_total_returns, shape)
    factors = _finite_array(
        "factor_returns",
        factor_returns,
        (*shape, len(plan.factor_names)),
    )
    if np.any(policy <= -1.0) or np.any(benchmark <= -1.0):
        raise M03REvaluationError("policy and benchmark returns must exceed -1")
    claimed_common_sha256 = _require_digest(
        "common_evaluator_inputs_sha256", common_evaluator_inputs_sha256
    )
    computed_common_sha256 = m03r_common_evaluator_inputs_sha256(
        score_dates=score_dates,
        fold_ids=fold_ids,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factors,
        plan=plan,
    )
    if claimed_common_sha256 != computed_common_sha256:
        raise M03REvaluationError(
            "common_evaluator_inputs_sha256 does not match the supplied common arrays"
        )
    claimed_candidate_sha256 = _require_digest(
        "candidate_policy_returns_sha256", candidate_policy_returns_sha256
    )
    computed_candidate_sha256 = m03r_candidate_policy_returns_sha256(
        policy_net_returns=policy,
        common_evaluator_inputs_sha256=computed_common_sha256,
        plan=plan,
    )
    if claimed_candidate_sha256 != computed_candidate_sha256:
        raise M03REvaluationError(
            "candidate_policy_returns_sha256 does not match the supplied policy path"
        )

    active_simple = policy - benchmark
    market_excess = market - risk_free
    policy_excess = policy - risk_free
    active_market = _regression(
        active_simple,
        market_excess[..., None],
        (M03R_MARKET_FACTOR_NAME,),
    )
    active_market["active_market_beta"] = active_market["loadings"][
        M03R_MARKET_FACTOR_NAME
    ]
    for inference in active_market["hac"].values():
        inference["active_beta_standard_error"] = inference["loading_se"][
            M03R_MARKET_FACTOR_NAME
        ]
        inference["active_beta_t"] = inference["loading_t"][
            M03R_MARKET_FACTOR_NAME
        ]
    market_alpha = _regression(
        policy_excess,
        market_excess[..., None],
        (M03R_MARKET_FACTOR_NAME,),
    )
    market_alpha["portfolio_market_beta"] = market_alpha["loadings"][
        M03R_MARKET_FACTOR_NAME
    ]
    benchmark_market = _regression(
        benchmark - risk_free,
        market_excess[..., None],
        (M03R_MARKET_FACTOR_NAME,),
    )
    benchmark_market["benchmark_market_beta"] = benchmark_market["loadings"][
        M03R_MARKET_FACTOR_NAME
    ]
    multifactor_regressors = np.concatenate(
        (market_excess[..., None], factors), axis=-1
    )
    multifactor_names = (M03R_MARKET_FACTOR_NAME, *plan.factor_names)
    portfolio_multifactor = _regression(
        policy_excess,
        multifactor_regressors,
        multifactor_names,
    )
    benchmark_multifactor = _regression(
        benchmark - risk_free,
        multifactor_regressors,
        multifactor_names,
    )
    active_multifactor = _regression(
        active_simple,
        multifactor_regressors,
        multifactor_names,
    )
    primary_active_hac = active_market["hac"][
        str(plan.primary_hac_lag_trading_sessions)
    ]
    active_beta = float(active_market["active_market_beta"])
    active_beta_standard_error = float(
        primary_active_hac["active_beta_standard_error"]
    )
    equivalence_z = float(NormalDist().inv_cdf(1.0 - plan.one_sided_alpha))
    active_beta_absolute_deviation = abs(
        active_beta - M03R_DESIGN.active_risk.active_market_beta_target
    )
    active_beta_equivalence_upper_bound = (
        active_beta_absolute_deviation
        + equivalence_z * active_beta_standard_error
    )
    active_beta_maximum = float(
        M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
    )
    beta_diagnostics = {
        "portfolio_market_beta": float(market_alpha["portfolio_market_beta"]),
        "benchmark_market_beta": float(
            benchmark_market["benchmark_market_beta"]
        ),
        "active_market_beta": active_beta,
        "primary_hac_lag_trading_sessions": (
            plan.primary_hac_lag_trading_sessions
        ),
        "active_beta_standard_error": active_beta_standard_error,
        "active_beta_constraint_maximum_absolute": active_beta_maximum,
        "active_beta_point_constraint_satisfied": bool(
            active_beta_absolute_deviation <= active_beta_maximum
        ),
        "active_beta_equivalence_one_sided_confidence_level": float(
            1.0 - plan.one_sided_alpha
        ),
        "active_beta_equivalence_z": equivalence_z,
        "active_beta_equivalence_upper_bound": float(
            active_beta_equivalence_upper_bound
        ),
        "active_beta_constraint_satisfied": bool(
            active_beta_equivalence_upper_bound <= active_beta_maximum
        ),
    }

    bootstrap: dict[str, Any] = {}
    seed = bytes.fromhex(plan.bootstrap_seed_sha256)
    for block_length in plan.bootstrap_block_lengths_trading_sessions:
        active_means = np.empty(plan.bootstrap_replicates, dtype=np.float64)
        portfolio_market_alphas = np.empty(
            plan.bootstrap_replicates, dtype=np.float64
        )
        portfolio_factor_alphas = np.empty(
            plan.bootstrap_replicates, dtype=np.float64
        )
        benchmark_factor_alphas = np.empty(
            plan.bootstrap_replicates, dtype=np.float64
        )
        active_factor_alphas = np.empty(
            plan.bootstrap_replicates, dtype=np.float64
        )
        sharpe_differences = np.empty(plan.bootstrap_replicates, dtype=np.float64)
        for replicate in range(plan.bootstrap_replicates):
            indexes = tuple(
                _block_indices(seed, replicate, fold, block_length)
                for fold in range(M03R_OUTER_FOLDS)
            )
            sampled_policy = np.stack(
                [policy[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_benchmark = np.stack(
                [benchmark[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_risk_free = np.stack(
                [risk_free[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_market = np.stack(
                [market_excess[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_factors = np.stack(
                [factors[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_active = np.log1p(sampled_policy) - np.log1p(sampled_benchmark)
            sampled_policy_excess = sampled_policy - sampled_risk_free
            sampled_benchmark_excess = sampled_benchmark - sampled_risk_free
            sampled_active_simple = sampled_policy - sampled_benchmark
            active_means[replicate] = float(sampled_active.mean())
            portfolio_market_alphas[replicate] = _fold_fixed_effect_coefficients(
                sampled_policy_excess,
                context=f"bootstrap market regression replicate {replicate}",
                regressors=sampled_market[..., None],
            )[0][0]
            sampled_factor_regressors = np.concatenate(
                (sampled_market[..., None], sampled_factors), axis=-1
            )
            portfolio_factor_alphas[replicate] = _fold_fixed_effect_coefficients(
                sampled_policy_excess,
                context=(
                    f"bootstrap portfolio multifactor regression replicate {replicate}"
                ),
                regressors=sampled_factor_regressors,
            )[0][0]
            benchmark_factor_alphas[replicate] = _fold_fixed_effect_coefficients(
                sampled_benchmark_excess,
                context=(
                    f"bootstrap benchmark multifactor regression replicate {replicate}"
                ),
                regressors=sampled_factor_regressors,
            )[0][0]
            active_factor_alphas[replicate] = _fold_fixed_effect_coefficients(
                sampled_active_simple,
                context=(
                    f"bootstrap active multifactor regression replicate {replicate}"
                ),
                regressors=sampled_factor_regressors,
            )[0][0]
            policy_excess_sample = sampled_policy - sampled_risk_free
            benchmark_excess_sample = sampled_benchmark - sampled_risk_free
            policy_std = float(np.std(policy_excess_sample, ddof=1))
            benchmark_std = float(np.std(benchmark_excess_sample, ddof=1))
            if policy_std <= 0.0 or benchmark_std <= 0.0:
                raise M03REvaluationError(
                    "bootstrap Sharpe is undefined at zero volatility"
                )
            sharpe_differences[replicate] = math.sqrt(252.0) * (
                float(np.mean(policy_excess_sample)) / policy_std
                - float(np.mean(benchmark_excess_sample)) / benchmark_std
            )
        quantile = float(plan.one_sided_alpha)
        bootstrap[str(block_length)] = {
            "one_sided_confidence_level": float(1.0 - quantile),
            "active_mean_log_return_daily_lcb": float(
                np.quantile(active_means, quantile, method="inverted_cdf")
            ),
            "portfolio_market_alpha_daily_lcb": float(
                np.quantile(
                    portfolio_market_alphas, quantile, method="inverted_cdf"
                )
            ),
            "portfolio_multifactor_alpha_daily_lcb": float(
                np.quantile(
                    portfolio_factor_alphas, quantile, method="inverted_cdf"
                )
            ),
            "benchmark_multifactor_alpha_daily_lcb": float(
                np.quantile(
                    benchmark_factor_alphas, quantile, method="inverted_cdf"
                )
            ),
            "active_multifactor_alpha_daily_lcb": float(
                np.quantile(
                    active_factor_alphas, quantile, method="inverted_cdf"
                )
            ),
            "active_multifactor_alpha_annualized_lcb": float(
                252.0
                * np.quantile(
                    active_factor_alphas, quantile, method="inverted_cdf"
                )
            ),
            "policy_minus_benchmark_sharpe_lcb": float(
                np.quantile(sharpe_differences, quantile, method="inverted_cdf")
            ),
        }

    unsigned: dict[str, Any] = {
        "schema": M03R_EVALUATION_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "setting_id": setting.setting_id,
        "promotion_eligible_setting": setting.promotion_eligible,
        "inference_contract_sha256": plan.inference_contract_sha256,
        "inference_plan": plan.semantics(),
        "common_evaluator_inputs_schema": M03R_COMMON_EVALUATOR_INPUT_SCHEMA,
        "common_evaluator_inputs_sha256": computed_common_sha256,
        "candidate_policy_returns_schema": M03R_CANDIDATE_POLICY_RETURNS_SCHEMA,
        "candidate_policy_returns_sha256": computed_candidate_sha256,
        "factor_names": list(plan.factor_names),
        "bootstrap_replicates": plan.bootstrap_replicates,
        "bootstrap_seed_sha256": plan.bootstrap_seed_sha256,
        "promotion_authorized": False,
        "promotion_blockers": list(M03R_PROMOTION_BLOCKERS),
        "market_beta_diagnostics": beta_diagnostics,
        "active_market_regression": active_market,
        "portfolio_market_regression": market_alpha,
        "benchmark_market_regression": benchmark_market,
        "portfolio_multifactor_regression": portfolio_multifactor,
        "benchmark_multifactor_regression": benchmark_multifactor,
        "active_multifactor_regression": active_multifactor,
        "bootstrap": bootstrap,
    }
    return {**unsigned, "receipt_sha256": _sha256(unsigned)}


def _require_exact_keys(
    value: Any,
    expected: frozenset[str],
    *,
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise M03REvaluationError(f"{name} key schema mismatch")
    return value


def _require_plain_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise M03REvaluationError(f"{name} must be an integer")
    return value


def _require_plain_float(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise M03REvaluationError(f"{name} must be a finite float")
    if minimum is not None and value < minimum:
        raise M03REvaluationError(f"{name} is below its minimum")
    if maximum is not None and value > maximum:
        raise M03REvaluationError(f"{name} is above its maximum")
    return value


def _require_nullable_float(name: str, value: Any) -> None:
    if value is not None:
        _require_plain_float(name, value)


def _plan_from_receipt(value: Any, claimed_sha256: str) -> M03RInferencePlan:
    expected_keys = frozenset(
        {
            "schema",
            "protocol_generation",
            "outer_fold_count",
            "score_sessions_per_fold",
            "policy_net_return_convention",
            "benchmark_net_return_convention",
            "risk_free_return_convention",
            "market_total_return_convention",
            "market_excess_definition",
            "market_factor_name",
            "factor_return_conventions",
            "active_beta_return_definition",
            "active_performance_definition",
            "regression_fold_treatment",
            "regression_common_intercept_definition",
            "hac_cross_fold_lag_pairs",
            "regression_intercept_annualization",
            "hac_lags_trading_sessions",
            "primary_hac_lag_trading_sessions",
            "primary_bootstrap_block_length_trading_sessions",
            "sensitivity_bootstrap_block_lengths_trading_sessions",
            "bootstrap_replicates",
            "bootstrap_seed_sha256",
            "bootstrap_resampling",
            "one_sided_alpha",
            "quantile_method",
        }
    )
    payload = _require_exact_keys(value, expected_keys, name="inference_plan")
    fixed_strings = {
        "schema": M03R_INFERENCE_PLAN_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "policy_net_return_convention": "daily-simple-net-return",
        "benchmark_net_return_convention": "daily-simple-net-return",
        "risk_free_return_convention": "daily-simple-return",
        "market_total_return_convention": "daily-simple-total-return",
        "market_excess_definition": "market_total_return-minus-risk_free_return",
        "market_factor_name": M03R_MARKET_FACTOR_NAME,
        "active_beta_return_definition": (
            "policy_net_return-minus-benchmark_net_return"
        ),
        "active_performance_definition": (
            "log1p(policy_net_return)-log1p(benchmark_net_return)"
        ),
        "regression_fold_treatment": (
            "six-fold-fixed-effects-effect-coded-common-intercept"
        ),
        "regression_common_intercept_definition": (
            "equal-weight-mean-of-six-fold-specific-intercepts"
        ),
        "hac_cross_fold_lag_pairs": "excluded",
        "regression_intercept_annualization": (
            "252-times-daily-arithmetic-intercept"
        ),
        "bootstrap_resampling": "within-fold-circular-moving-block",
        "quantile_method": "inverted_cdf",
    }
    for field, expected in fixed_strings.items():
        if type(payload[field]) is not str or payload[field] != expected:
            raise M03REvaluationError(f"inference_plan.{field} drifted")
    if _require_plain_int(
        "inference_plan.outer_fold_count", payload["outer_fold_count"]
    ) != M03R_OUTER_FOLDS:
        raise M03REvaluationError("inference plan outer-fold count drifted")
    if _require_plain_int(
        "inference_plan.score_sessions_per_fold",
        payload["score_sessions_per_fold"],
    ) != M03R_SCORE_SESSIONS_PER_FOLD:
        raise M03REvaluationError("inference plan score-session count drifted")
    factor_rows = payload["factor_return_conventions"]
    if type(factor_rows) is not list or not factor_rows:
        raise M03REvaluationError(
            "inference_plan.factor_return_conventions must be a non-empty list"
        )
    factor_names: list[str] = []
    factor_conventions: list[str] = []
    for index, row in enumerate(factor_rows):
        typed_row = _require_exact_keys(
            row,
            frozenset({"factor_name", "return_convention"}),
            name=f"inference_plan.factor_return_conventions[{index}]",
        )
        if type(typed_row["factor_name"]) is not str:
            raise M03REvaluationError("factor_name must be a string")
        if type(typed_row["return_convention"]) is not str:
            raise M03REvaluationError("factor return convention must be a string")
        factor_names.append(typed_row["factor_name"])
        factor_conventions.append(typed_row["return_convention"])
    sensitivity_block_lengths = payload[
        "sensitivity_bootstrap_block_lengths_trading_sessions"
    ]
    if type(sensitivity_block_lengths) is not list or any(
        type(length) is not int for length in sensitivity_block_lengths
    ):
        raise M03REvaluationError(
            "inference plan sensitivity block lengths must be integers"
        )
    hac_lags = payload["hac_lags_trading_sessions"]
    if (
        type(hac_lags) is not list
        or any(type(lag) is not int for lag in hac_lags)
        or tuple(hac_lags) != M03R_HAC_LAGS
    ):
        raise M03REvaluationError("inference plan HAC lags drifted")
    plan = M03RInferencePlan(
        factor_names=tuple(factor_names),
        factor_return_conventions=tuple(factor_conventions),
        primary_bootstrap_block_length_trading_sessions=_require_plain_int(
            "inference_plan.primary_bootstrap_block_length_trading_sessions",
            payload["primary_bootstrap_block_length_trading_sessions"],
        ),
        sensitivity_bootstrap_block_lengths_trading_sessions=tuple(
            sensitivity_block_lengths
        ),
        bootstrap_replicates=_require_plain_int(
            "inference_plan.bootstrap_replicates",
            payload["bootstrap_replicates"],
            minimum=1_000,
        ),
        bootstrap_seed_sha256=_require_digest(
            "inference_plan.bootstrap_seed_sha256",
            payload["bootstrap_seed_sha256"],
        ),
        one_sided_alpha=_require_plain_float(
            "inference_plan.one_sided_alpha",
            payload["one_sided_alpha"],
            minimum=0.0,
            maximum=0.5,
        ),
        primary_hac_lag_trading_sessions=_require_plain_int(
            "inference_plan.primary_hac_lag_trading_sessions",
            payload["primary_hac_lag_trading_sessions"],
        ),
        inference_contract_sha256=claimed_sha256,
    )
    if payload != plan.semantics():
        raise M03REvaluationError("inference plan semantics drifted")
    return plan


def _validate_regression(
    value: Any,
    *,
    name: str,
    loading_names: tuple[str, ...],
    extra_beta_name: str | None = None,
    active_hac: bool = False,
) -> dict[str, Any]:
    keys = {
        "alpha_daily",
        "alpha_annualized_arithmetic",
        "loadings",
        "residual_volatility_annualized",
        "fold_fixed_effect_contract",
        "fold_effects",
        "hac",
    }
    if extra_beta_name is not None:
        keys.add(extra_beta_name)
    regression = _require_exact_keys(value, frozenset(keys), name=name)
    for field in (
        "alpha_daily",
        "alpha_annualized_arithmetic",
        "residual_volatility_annualized",
    ):
        _require_plain_float(f"{name}.{field}", regression[field])
    if regression["fold_fixed_effect_contract"] != (
        "six-fold-effect-coded-zero-sum-common-intercept"
    ):
        raise M03REvaluationError(f"{name} fold-fixed-effect contract drifted")
    fold_effects = regression["fold_effects"]
    if type(fold_effects) is not list or len(fold_effects) != M03R_OUTER_FOLDS:
        raise M03REvaluationError(f"{name}.fold_effects must contain six values")
    for index, effect in enumerate(fold_effects):
        _require_plain_float(f"{name}.fold_effects[{index}]", effect)
    if not math.isclose(sum(fold_effects), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise M03REvaluationError(f"{name}.fold_effects must sum to zero")
    if regression["alpha_annualized_arithmetic"] != 252.0 * regression["alpha_daily"]:
        raise M03REvaluationError(f"{name} alpha annualization drifted")
    loadings = _require_exact_keys(
        regression["loadings"], frozenset(loading_names), name=f"{name}.loadings"
    )
    for loading_name in loading_names:
        _require_plain_float(
            f"{name}.loadings.{loading_name}", loadings[loading_name]
        )
    if extra_beta_name is not None:
        beta = _require_plain_float(
            f"{name}.{extra_beta_name}", regression[extra_beta_name]
        )
        if beta != loadings[M03R_MARKET_FACTOR_NAME]:
            raise M03REvaluationError(f"{name} beta does not match its loading")
    hac = _require_exact_keys(
        regression["hac"],
        frozenset(str(lag) for lag in M03R_HAC_LAGS),
        name=f"{name}.hac",
    )
    base_hac_keys = {
        "cross_fold_lag_pairs",
        "alpha_se_daily",
        "alpha_se_annualized_arithmetic",
        "alpha_t",
        "alpha_two_sided_normal_p",
        "loading_se",
        "loading_t",
    }
    if active_hac:
        base_hac_keys.update({"active_beta_standard_error", "active_beta_t"})
    for lag, row_value in hac.items():
        row = _require_exact_keys(
            row_value, frozenset(base_hac_keys), name=f"{name}.hac.{lag}"
        )
        if _require_plain_int(
            f"{name}.hac.{lag}.cross_fold_lag_pairs",
            row["cross_fold_lag_pairs"],
        ) != 0:
            raise M03REvaluationError(
                f"{name}.hac.{lag} included cross-fold lag pairs"
            )
        _require_plain_float(
            f"{name}.hac.{lag}.alpha_se_daily",
            row["alpha_se_daily"],
            minimum=0.0,
        )
        _require_plain_float(
            f"{name}.hac.{lag}.alpha_se_annualized_arithmetic",
            row["alpha_se_annualized_arithmetic"],
            minimum=0.0,
        )
        if (
            row["alpha_se_annualized_arithmetic"]
            != 252.0 * row["alpha_se_daily"]
        ):
            raise M03REvaluationError(f"{name}.hac.{lag} SE annualization drifted")
        _require_nullable_float(f"{name}.hac.{lag}.alpha_t", row["alpha_t"])
        _require_nullable_float(
            f"{name}.hac.{lag}.alpha_two_sided_normal_p",
            row["alpha_two_sided_normal_p"],
        )
        loading_se = _require_exact_keys(
            row["loading_se"],
            frozenset(loading_names),
            name=f"{name}.hac.{lag}.loading_se",
        )
        loading_t = _require_exact_keys(
            row["loading_t"],
            frozenset(loading_names),
            name=f"{name}.hac.{lag}.loading_t",
        )
        for loading_name in loading_names:
            _require_plain_float(
                f"{name}.hac.{lag}.loading_se.{loading_name}",
                loading_se[loading_name],
                minimum=0.0,
            )
            _require_nullable_float(
                f"{name}.hac.{lag}.loading_t.{loading_name}",
                loading_t[loading_name],
            )
        if active_hac:
            active_se = _require_plain_float(
                f"{name}.hac.{lag}.active_beta_standard_error",
                row["active_beta_standard_error"],
                minimum=0.0,
            )
            _require_nullable_float(
                f"{name}.hac.{lag}.active_beta_t", row["active_beta_t"]
            )
            if active_se != loading_se[M03R_MARKET_FACTOR_NAME]:
                raise M03REvaluationError("active-beta standard error drifted")
    return regression


def validate_m03r_inference_receipt(receipt: dict[str, Any]) -> None:
    """Reject incomplete, mutated, or cross-generation M03R receipts."""

    top_level_keys = frozenset(
        {
            "schema",
            "protocol_generation",
            "setting_id",
            "promotion_eligible_setting",
            "inference_contract_sha256",
            "inference_plan",
            "common_evaluator_inputs_schema",
            "common_evaluator_inputs_sha256",
            "candidate_policy_returns_schema",
            "candidate_policy_returns_sha256",
            "factor_names",
            "bootstrap_replicates",
            "bootstrap_seed_sha256",
            "promotion_authorized",
            "promotion_blockers",
            "market_beta_diagnostics",
            "active_market_regression",
            "portfolio_market_regression",
            "benchmark_market_regression",
            "portfolio_multifactor_regression",
            "benchmark_multifactor_regression",
            "active_multifactor_regression",
            "bootstrap",
            "receipt_sha256",
        }
    )
    typed_receipt = _require_exact_keys(
        receipt, top_level_keys, name="M03R inference receipt"
    )
    if (
        type(typed_receipt["schema"]) is not str
        or typed_receipt["schema"] != M03R_EVALUATION_SCHEMA
    ):
        raise M03REvaluationError("M03R inference schema mismatch")
    if (
        type(typed_receipt["protocol_generation"]) is not str
        or typed_receipt["protocol_generation"] != M03R_PROTOCOL_GENERATION
    ):
        raise M03REvaluationError("M03R inference protocol generation mismatch")
    if type(typed_receipt["setting_id"]) is not str:
        raise M03REvaluationError("setting_id must be a string")
    setting = resolve_m03r_v5_setting(typed_receipt["setting_id"])
    if type(typed_receipt["promotion_eligible_setting"]) is not bool or (
        typed_receipt["promotion_eligible_setting"] is not setting.promotion_eligible
    ):
        raise M03REvaluationError("promotion eligibility does not match the setting")
    inference_contract_sha256 = _require_digest(
        "inference_contract_sha256", typed_receipt["inference_contract_sha256"]
    )
    plan = _plan_from_receipt(
        typed_receipt["inference_plan"], inference_contract_sha256
    )
    if (
        type(typed_receipt["common_evaluator_inputs_schema"]) is not str
        or typed_receipt["common_evaluator_inputs_schema"]
        != M03R_COMMON_EVALUATOR_INPUT_SCHEMA
    ):
        raise M03REvaluationError("M03R common-evaluator-input schema mismatch")
    _require_digest(
        "common_evaluator_inputs_sha256",
        typed_receipt["common_evaluator_inputs_sha256"],
    )
    if (
        type(typed_receipt["candidate_policy_returns_schema"]) is not str
        or typed_receipt["candidate_policy_returns_schema"]
        != M03R_CANDIDATE_POLICY_RETURNS_SCHEMA
    ):
        raise M03REvaluationError("M03R candidate-policy-return schema mismatch")
    _require_digest(
        "candidate_policy_returns_sha256",
        typed_receipt["candidate_policy_returns_sha256"],
    )
    if type(typed_receipt["factor_names"]) is not list or (
        typed_receipt["factor_names"] != list(plan.factor_names)
    ):
        raise M03REvaluationError("factor_names do not match the inference plan")
    if type(typed_receipt["bootstrap_replicates"]) is not int or (
        typed_receipt["bootstrap_replicates"] != plan.bootstrap_replicates
    ):
        raise M03REvaluationError("bootstrap replicate count drifted")
    if _require_digest(
        "bootstrap_seed_sha256", typed_receipt["bootstrap_seed_sha256"]
    ) != plan.bootstrap_seed_sha256:
        raise M03REvaluationError("bootstrap seed binding drifted")
    if typed_receipt["promotion_authorized"] is not False:
        raise M03REvaluationError(
            "M03R inference receipts cannot authorize promotion before the "
            "multiplicity, outer-access, and execution bindings are complete"
        )
    if type(typed_receipt["promotion_blockers"]) is not list or (
        typed_receipt["promotion_blockers"] != list(M03R_PROMOTION_BLOCKERS)
    ):
        raise M03REvaluationError("M03R promotion-blocker inventory mismatch")

    active = _validate_regression(
        typed_receipt["active_market_regression"],
        name="active_market_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME,),
        extra_beta_name="active_market_beta",
        active_hac=True,
    )
    portfolio = _validate_regression(
        typed_receipt["portfolio_market_regression"],
        name="portfolio_market_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME,),
        extra_beta_name="portfolio_market_beta",
    )
    benchmark = _validate_regression(
        typed_receipt["benchmark_market_regression"],
        name="benchmark_market_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME,),
        extra_beta_name="benchmark_market_beta",
    )
    for active_field, portfolio_field, benchmark_field in (
        ("alpha_daily", "alpha_daily", "alpha_daily"),
        (
            "alpha_annualized_arithmetic",
            "alpha_annualized_arithmetic",
            "alpha_annualized_arithmetic",
        ),
        (
            "active_market_beta",
            "portfolio_market_beta",
            "benchmark_market_beta",
        ),
    ):
        expected_active_value = (
            portfolio[portfolio_field] - benchmark[benchmark_field]
        )
        observed_active_value = active[active_field]
        if not math.isclose(
            observed_active_value,
            expected_active_value,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise M03REvaluationError(
                f"active market {active_field} is inconsistent with portfolio-minus-C1"
            )
    portfolio_multifactor = _validate_regression(
        typed_receipt["portfolio_multifactor_regression"],
        name="portfolio_multifactor_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME, *plan.factor_names),
    )
    benchmark_multifactor = _validate_regression(
        typed_receipt["benchmark_multifactor_regression"],
        name="benchmark_multifactor_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME, *plan.factor_names),
    )
    active_multifactor = _validate_regression(
        typed_receipt["active_multifactor_regression"],
        name="active_multifactor_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME, *plan.factor_names),
    )
    for field in ("alpha_daily", "alpha_annualized_arithmetic"):
        expected_active = (
            portfolio_multifactor[field] - benchmark_multifactor[field]
        )
        if not math.isclose(
            active_multifactor[field],
            expected_active,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise M03REvaluationError(
                f"active multifactor {field} is inconsistent with portfolio-minus-C1"
            )
    for factor_name in (M03R_MARKET_FACTOR_NAME, *plan.factor_names):
        expected_active_loading = (
            portfolio_multifactor["loadings"][factor_name]
            - benchmark_multifactor["loadings"][factor_name]
        )
        if not math.isclose(
            active_multifactor["loadings"][factor_name],
            expected_active_loading,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise M03REvaluationError(
                "active multifactor loading is inconsistent with portfolio-minus-C1"
            )
    beta = _require_exact_keys(
        typed_receipt["market_beta_diagnostics"],
        frozenset(
            {
                "portfolio_market_beta",
                "benchmark_market_beta",
                "active_market_beta",
                "primary_hac_lag_trading_sessions",
                "active_beta_standard_error",
                "active_beta_constraint_maximum_absolute",
                "active_beta_point_constraint_satisfied",
                "active_beta_equivalence_one_sided_confidence_level",
                "active_beta_equivalence_z",
                "active_beta_equivalence_upper_bound",
                "active_beta_constraint_satisfied",
            }
        ),
        name="market_beta_diagnostics",
    )
    for field in (
        "portfolio_market_beta",
        "benchmark_market_beta",
        "active_market_beta",
        "active_beta_standard_error",
        "active_beta_constraint_maximum_absolute",
        "active_beta_equivalence_one_sided_confidence_level",
        "active_beta_equivalence_z",
        "active_beta_equivalence_upper_bound",
    ):
        _require_plain_float(f"market_beta_diagnostics.{field}", beta[field])
    for field in (
        "active_beta_point_constraint_satisfied",
        "active_beta_constraint_satisfied",
    ):
        if type(beta[field]) is not bool:
            raise M03REvaluationError(f"{field} must be a boolean")
    if _require_plain_int(
        "market_beta_diagnostics.primary_hac_lag_trading_sessions",
        beta["primary_hac_lag_trading_sessions"],
    ) != plan.primary_hac_lag_trading_sessions:
        raise M03REvaluationError("primary active-beta HAC lag drifted")
    expected_beta_values = (
        portfolio["portfolio_market_beta"],
        benchmark["benchmark_market_beta"],
        active["active_market_beta"],
        active["hac"][str(plan.primary_hac_lag_trading_sessions)][
            "active_beta_standard_error"
        ],
        M03R_DESIGN.active_risk.absolute_active_market_beta_maximum,
    )
    observed_beta_values = tuple(
        beta[field]
        for field in (
            "portfolio_market_beta",
            "benchmark_market_beta",
            "active_market_beta",
            "active_beta_standard_error",
            "active_beta_constraint_maximum_absolute",
        )
    )
    if observed_beta_values != expected_beta_values:
        raise M03REvaluationError("market-beta diagnostics are internally inconsistent")
    expected_absolute_deviation = abs(
        active["active_market_beta"]
        - M03R_DESIGN.active_risk.active_market_beta_target
    )
    expected_point_constraint = bool(
        expected_absolute_deviation
        <= M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
    )
    if beta["active_beta_point_constraint_satisfied"] is not expected_point_constraint:
        raise M03REvaluationError("active-beta point constraint verdict drifted")
    expected_confidence_level = 1.0 - plan.one_sided_alpha
    expected_z = NormalDist().inv_cdf(expected_confidence_level)
    expected_equivalence_upper_bound = (
        expected_absolute_deviation
        + expected_z * beta["active_beta_standard_error"]
    )
    if (
        beta["active_beta_equivalence_one_sided_confidence_level"]
        != expected_confidence_level
        or beta["active_beta_equivalence_z"] != expected_z
        or beta["active_beta_equivalence_upper_bound"]
        != expected_equivalence_upper_bound
    ):
        raise M03REvaluationError("active-beta equivalence evidence drifted")
    expected_constraint = bool(
        expected_equivalence_upper_bound
        <= M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
    )
    if beta["active_beta_constraint_satisfied"] is not expected_constraint:
        raise M03REvaluationError("active-beta constraint verdict drifted")

    bootstrap = _require_exact_keys(
        typed_receipt["bootstrap"],
        frozenset(
            str(length)
            for length in plan.bootstrap_block_lengths_trading_sessions
        ),
        name="bootstrap",
    )
    bootstrap_keys = frozenset(
        {
            "one_sided_confidence_level",
            "active_mean_log_return_daily_lcb",
            "portfolio_market_alpha_daily_lcb",
            "portfolio_multifactor_alpha_daily_lcb",
            "benchmark_multifactor_alpha_daily_lcb",
            "active_multifactor_alpha_daily_lcb",
            "active_multifactor_alpha_annualized_lcb",
            "policy_minus_benchmark_sharpe_lcb",
        }
    )
    for length, row_value in bootstrap.items():
        row = _require_exact_keys(
            row_value, bootstrap_keys, name=f"bootstrap.{length}"
        )
        for field in bootstrap_keys:
            _require_plain_float(f"bootstrap.{length}.{field}", row[field])
        if row["one_sided_confidence_level"] != 1.0 - plan.one_sided_alpha:
            raise M03REvaluationError("bootstrap confidence level drifted")
        if (
            row["active_multifactor_alpha_annualized_lcb"]
            != 252.0 * row["active_multifactor_alpha_daily_lcb"]
        ):
            raise M03REvaluationError(
                "bootstrap active multifactor alpha annualization drifted"
            )

    claimed = _require_digest("receipt_sha256", typed_receipt["receipt_sha256"])
    unsigned = {
        key: value for key, value in typed_receipt.items() if key != "receipt_sha256"
    }
    if _sha256(unsigned) != claimed:
        raise M03REvaluationError("M03R inference receipt hash mismatch")


__all__ = [
    "M03R_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS",
    "M03R_CANDIDATE_POLICY_RETURNS_SCHEMA",
    "M03R_COMMON_EVALUATOR_INPUT_SCHEMA",
    "M03R_EVALUATION_SCHEMA",
    "M03R_HAC_LAGS",
    "M03R_INFERENCE_PLAN_SCHEMA",
    "M03R_MARKET_FACTOR_NAME",
    "M03R_OUTER_FOLDS",
    "M03R_PRIMARY_BOOTSTRAP_BLOCK_LENGTH",
    "M03R_PRIMARY_HAC_LAG",
    "M03R_PROMOTION_BLOCKERS",
    "M03R_SCORE_SESSIONS_PER_FOLD",
    "M03R_SOURCE_ARRAY_SCHEMA",
    "M03R_SUPPORTED_FACTOR_RETURN_CONVENTIONS",
    "M03REvaluationError",
    "M03RInferencePlan",
    "build_m03r_inference_plan",
    "evaluate_m03r_inference",
    "finite_control_diagnostic",
    "m03r_candidate_policy_returns_sha256",
    "m03r_common_evaluator_inputs_sha256",
    "m03r_source_arrays_sha256",
    "validate_m03r_inference_receipt",
]
