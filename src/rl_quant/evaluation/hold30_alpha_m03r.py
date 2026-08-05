"""Versioned M03R inference and finite-control diagnostics.

The frozen v3 evaluator intentionally remains byte-for-byte unchanged.  This
module owns corrected active-beta, one-sided uncertainty, and control-tail
semantics for ``prelockbox-hold30-active-alpha-m03r-v4`` only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_DESIGN,
    M03R_PROTOCOL_GENERATION,
    resolve_m03r_setting,
)

M03R_EVALUATION_SCHEMA = "rl-quant.hold30.m03r-evaluation-v4"
M03R_INFERENCE_PLAN_SCHEMA = "rl-quant.hold30.m03r-inference-plan-v1"
M03R_SOURCE_ARRAY_SCHEMA = "rl-quant.hold30.m03r-source-arrays-v1"
M03R_OUTER_FOLDS = 6
M03R_SCORE_SESSIONS_PER_FOLD = 63
M03R_HAC_LAGS = (10, 21, 30, 42)
M03R_PRIMARY_HAC_LAG = 30
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
    block_lengths_trading_sessions: tuple[int, ...],
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
        "regression_intercept_annualization": "252-times-daily-arithmetic-intercept",
        "hac_lags_trading_sessions": list(M03R_HAC_LAGS),
        "primary_hac_lag_trading_sessions": primary_hac_lag_trading_sessions,
        "block_lengths_trading_sessions": list(block_lengths_trading_sessions),
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
    block_lengths_trading_sessions: tuple[int, ...]
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
            type(self.block_lengths_trading_sessions) is not tuple
            or not self.block_lengths_trading_sessions
            or len(set(self.block_lengths_trading_sessions))
            != len(self.block_lengths_trading_sessions)
            or any(
                isinstance(length, bool)
                or not isinstance(length, int)
                or not 1 <= length <= M03R_SCORE_SESSIONS_PER_FOLD
                for length in self.block_lengths_trading_sessions
            )
        ):
            raise M03REvaluationError("block lengths must be unique integers in [1,63]")
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
            block_lengths_trading_sessions=self.block_lengths_trading_sessions,
            bootstrap_replicates=self.bootstrap_replicates,
            bootstrap_seed_sha256=self.bootstrap_seed_sha256,
            one_sided_alpha=self.one_sided_alpha,
            primary_hac_lag_trading_sessions=self.primary_hac_lag_trading_sessions,
        )


def build_m03r_inference_plan(
    *,
    factor_names: tuple[str, ...],
    factor_return_conventions: tuple[str, ...],
    block_lengths_trading_sessions: tuple[int, ...],
    bootstrap_replicates: int,
    bootstrap_seed_sha256: str,
    one_sided_alpha: float,
    primary_hac_lag_trading_sessions: int = M03R_PRIMARY_HAC_LAG,
) -> M03RInferencePlan:
    """Build a plan whose identifier is derived from all inference semantics."""

    semantics = _inference_plan_semantics(
        factor_names=factor_names,
        factor_return_conventions=factor_return_conventions,
        block_lengths_trading_sessions=block_lengths_trading_sessions,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed_sha256=bootstrap_seed_sha256,
        one_sided_alpha=one_sided_alpha,
        primary_hac_lag_trading_sessions=primary_hac_lag_trading_sessions,
    )
    return M03RInferencePlan(
        factor_names=factor_names,
        factor_return_conventions=factor_return_conventions,
        block_lengths_trading_sessions=block_lengths_trading_sessions,
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
    """Compute the canonical source-array binding used by M03R evaluation."""

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


def _regression(
    dependent: np.ndarray,
    regressors: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, Any]:
    if regressors.ndim != 2 or regressors.shape[0] != dependent.size:
        raise M03REvaluationError("regression arrays are misaligned")
    design = np.column_stack((np.ones(dependent.size), regressors))
    coefficients = _full_rank_coefficients(
        design,
        dependent,
        context="primary regression",
    )
    residual = dependent - design @ coefficients
    bread = np.linalg.pinv(design.T @ design)
    hac: dict[str, Any] = {}
    for lag in M03R_HAC_LAGS:
        scores = design * residual[:, None]
        meat = scores.T @ scores
        for offset in range(1, min(lag, dependent.size - 1) + 1):
            weight = 1.0 - offset / (lag + 1.0)
            cross = scores[offset:].T @ scores[:-offset]
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
            for name, value in zip(names, standard_errors[1:], strict=True)
        }
        loading_t = {
            name: float(value) if math.isfinite(float(value)) else None
            for name, value in zip(names, t_values[1:], strict=True)
        }
        hac[str(lag)] = {
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
            for name, value in zip(names, coefficients[1:], strict=True)
        },
        "residual_volatility_annualized": float(
            math.sqrt(252.0) * np.std(residual, ddof=1)
        ),
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
    return np.linalg.lstsq(design, dependent, rcond=None)[0]


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
    policy_net_returns: Any,
    benchmark_net_returns: Any,
    risk_free_returns: Any,
    market_total_returns: Any,
    factor_returns: Any,
    plan: M03RInferencePlan,
    source_arrays_sha256: str,
) -> dict[str, Any]:
    """Recompute active beta, formal alpha, and chronological uncertainty."""

    setting = resolve_m03r_setting(setting_id)
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
    claimed_source_arrays_sha256 = _require_digest(
        "source_arrays_sha256", source_arrays_sha256
    )
    computed_source_arrays_sha256 = _source_arrays_sha256(
        policy=policy,
        benchmark=benchmark,
        risk_free=risk_free,
        market=market,
        factors=factors,
        plan=plan,
    )
    if claimed_source_arrays_sha256 != computed_source_arrays_sha256:
        raise M03REvaluationError(
            "source_arrays_sha256 does not match the supplied canonical arrays"
        )

    active_simple = policy - benchmark
    market_excess = market - risk_free
    policy_excess = policy - risk_free
    active_market = _regression(
        active_simple.reshape(-1),
        market_excess.reshape(-1, 1),
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
        policy_excess.reshape(-1),
        market_excess.reshape(-1, 1),
        (M03R_MARKET_FACTOR_NAME,),
    )
    market_alpha["portfolio_market_beta"] = market_alpha["loadings"][
        M03R_MARKET_FACTOR_NAME
    ]
    benchmark_market = _regression(
        (benchmark - risk_free).reshape(-1),
        market_excess.reshape(-1, 1),
        (M03R_MARKET_FACTOR_NAME,),
    )
    benchmark_market["benchmark_market_beta"] = benchmark_market["loadings"][
        M03R_MARKET_FACTOR_NAME
    ]
    multifactor_alpha = _regression(
        policy_excess.reshape(-1),
        np.column_stack(
            (market_excess.reshape(-1), factors.reshape(-1, len(plan.factor_names)))
        ),
        (M03R_MARKET_FACTOR_NAME, *plan.factor_names),
    )
    primary_active_hac = active_market["hac"][
        str(plan.primary_hac_lag_trading_sessions)
    ]
    active_beta = float(active_market["active_market_beta"])
    beta_diagnostics = {
        "portfolio_market_beta": float(market_alpha["portfolio_market_beta"]),
        "benchmark_market_beta": float(
            benchmark_market["benchmark_market_beta"]
        ),
        "active_market_beta": active_beta,
        "primary_hac_lag_trading_sessions": (
            plan.primary_hac_lag_trading_sessions
        ),
        "active_beta_standard_error": float(
            primary_active_hac["active_beta_standard_error"]
        ),
        "active_beta_constraint_maximum_absolute": float(
            M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
        ),
        "active_beta_constraint_satisfied": bool(
            abs(
                active_beta
                - M03R_DESIGN.active_risk.active_market_beta_target
            )
            <= M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
        ),
    }

    bootstrap: dict[str, Any] = {}
    seed = bytes.fromhex(plan.bootstrap_seed_sha256)
    for block_length in plan.block_lengths_trading_sessions:
        active_means = np.empty(plan.bootstrap_replicates, dtype=np.float64)
        market_alphas = np.empty(plan.bootstrap_replicates, dtype=np.float64)
        factor_alphas = np.empty(plan.bootstrap_replicates, dtype=np.float64)
        sharpe_differences = np.empty(plan.bootstrap_replicates, dtype=np.float64)
        for replicate in range(plan.bootstrap_replicates):
            indexes = tuple(
                _block_indices(seed, replicate, fold, block_length)
                for fold in range(M03R_OUTER_FOLDS)
            )
            sampled_policy = np.concatenate(
                [policy[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_benchmark = np.concatenate(
                [benchmark[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_risk_free = np.concatenate(
                [risk_free[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_market = np.concatenate(
                [market_excess[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_factors = np.concatenate(
                [factors[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_active = np.log1p(sampled_policy) - np.log1p(sampled_benchmark)
            sampled_excess = sampled_policy - sampled_risk_free
            active_means[replicate] = float(sampled_active.mean())
            market_alphas[replicate] = _full_rank_coefficients(
                np.column_stack((np.ones(sampled_excess.size), sampled_market)),
                sampled_excess,
                context=f"bootstrap market regression replicate {replicate}",
            )[0]
            factor_alphas[replicate] = _full_rank_coefficients(
                np.column_stack(
                    (np.ones(sampled_excess.size), sampled_market, sampled_factors)
                ),
                sampled_excess,
                context=f"bootstrap multifactor regression replicate {replicate}",
            )[0]
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
            "market_alpha_daily_lcb": float(
                np.quantile(market_alphas, quantile, method="inverted_cdf")
            ),
            "multifactor_alpha_daily_lcb": float(
                np.quantile(factor_alphas, quantile, method="inverted_cdf")
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
        "source_array_schema": M03R_SOURCE_ARRAY_SCHEMA,
        "source_arrays_sha256": computed_source_arrays_sha256,
        "factor_names": list(plan.factor_names),
        "bootstrap_replicates": plan.bootstrap_replicates,
        "bootstrap_seed_sha256": plan.bootstrap_seed_sha256,
        "promotion_authorized": False,
        "promotion_blockers": list(M03R_PROMOTION_BLOCKERS),
        "market_beta_diagnostics": beta_diagnostics,
        "active_market_regression": active_market,
        "market_alpha_regression": market_alpha,
        "benchmark_market_regression": benchmark_market,
        "multifactor_alpha_regression": multifactor_alpha,
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
            "regression_intercept_annualization",
            "hac_lags_trading_sessions",
            "primary_hac_lag_trading_sessions",
            "block_lengths_trading_sessions",
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
    block_lengths = payload["block_lengths_trading_sessions"]
    if type(block_lengths) is not list or any(
        type(length) is not int for length in block_lengths
    ):
        raise M03REvaluationError("inference plan block lengths must be integers")
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
        block_lengths_trading_sessions=tuple(block_lengths),
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
            "source_array_schema",
            "source_arrays_sha256",
            "factor_names",
            "bootstrap_replicates",
            "bootstrap_seed_sha256",
            "promotion_authorized",
            "promotion_blockers",
            "market_beta_diagnostics",
            "active_market_regression",
            "market_alpha_regression",
            "benchmark_market_regression",
            "multifactor_alpha_regression",
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
    setting = resolve_m03r_setting(typed_receipt["setting_id"])
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
        type(typed_receipt["source_array_schema"]) is not str
        or typed_receipt["source_array_schema"] != M03R_SOURCE_ARRAY_SCHEMA
    ):
        raise M03REvaluationError("M03R source-array schema mismatch")
    _require_digest("source_arrays_sha256", typed_receipt["source_arrays_sha256"])
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
        typed_receipt["market_alpha_regression"],
        name="market_alpha_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME,),
        extra_beta_name="portfolio_market_beta",
    )
    benchmark = _validate_regression(
        typed_receipt["benchmark_market_regression"],
        name="benchmark_market_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME,),
        extra_beta_name="benchmark_market_beta",
    )
    _validate_regression(
        typed_receipt["multifactor_alpha_regression"],
        name="multifactor_alpha_regression",
        loading_names=(M03R_MARKET_FACTOR_NAME, *plan.factor_names),
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
    ):
        _require_plain_float(f"market_beta_diagnostics.{field}", beta[field])
    if type(beta["active_beta_constraint_satisfied"]) is not bool:
        raise M03REvaluationError(
            "active_beta_constraint_satisfied must be a boolean"
        )
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
    expected_constraint = bool(
        abs(
            active["active_market_beta"]
            - M03R_DESIGN.active_risk.active_market_beta_target
        )
        <= M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
    )
    if beta["active_beta_constraint_satisfied"] is not expected_constraint:
        raise M03REvaluationError("active-beta constraint verdict drifted")

    bootstrap = _require_exact_keys(
        typed_receipt["bootstrap"],
        frozenset(str(length) for length in plan.block_lengths_trading_sessions),
        name="bootstrap",
    )
    bootstrap_keys = frozenset(
        {
            "one_sided_confidence_level",
            "active_mean_log_return_daily_lcb",
            "market_alpha_daily_lcb",
            "multifactor_alpha_daily_lcb",
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

    claimed = _require_digest("receipt_sha256", typed_receipt["receipt_sha256"])
    unsigned = {
        key: value for key, value in typed_receipt.items() if key != "receipt_sha256"
    }
    if _sha256(unsigned) != claimed:
        raise M03REvaluationError("M03R inference receipt hash mismatch")


__all__ = [
    "M03R_EVALUATION_SCHEMA",
    "M03R_HAC_LAGS",
    "M03R_INFERENCE_PLAN_SCHEMA",
    "M03R_MARKET_FACTOR_NAME",
    "M03R_OUTER_FOLDS",
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
    "m03r_source_arrays_sha256",
    "validate_m03r_inference_receipt",
]
