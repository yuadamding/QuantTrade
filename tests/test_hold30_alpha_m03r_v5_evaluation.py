from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from rl_quant.evaluation.hold30_alpha_m03r_v5 import (
    M03R_EVALUATION_SCHEMA,
    M03REvaluationError,
    M03RInferencePlan,
    build_m03r_inference_plan,
    evaluate_m03r_inference,
    finite_control_diagnostic,
    m03r_candidate_policy_returns_sha256,
    m03r_common_evaluator_inputs_sha256,
    validate_m03r_inference_receipt,
)


def _digest(character: str) -> str:
    return character * 64


def _panel() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(17)
    shape = (6, 63)
    market = rng.normal(0.0004, 0.009, size=shape)
    risk_free = np.full(shape, 0.0001)
    benchmark = (
        risk_free + 0.96 * (market - risk_free) + rng.normal(0.0, 0.0015, size=shape)
    )
    active = 0.00008 + 0.04 * (market - risk_free) + rng.normal(0.0, 0.001, size=shape)
    policy = benchmark + active
    factors = rng.normal(0.0, 0.004, size=(*shape, 2))
    return policy, benchmark, risk_free, market, factors


def _plan() -> M03RInferencePlan:
    return build_m03r_inference_plan(
        factor_names=("SIZE", "VALUE"),
        factor_return_conventions=(
            "daily-simple-long-short-return",
            "daily-simple-long-short-return",
        ),
        bootstrap_replicates=1_000,
        bootstrap_seed_sha256=_digest("1"),
        one_sided_alpha=0.05,
    )


def _metadata() -> tuple[np.ndarray, np.ndarray]:
    start = date(2020, 1, 1)
    score_dates = np.asarray(
        [(start + timedelta(days=index)).isoformat() for index in range(6 * 63)]
    ).reshape(6, 63)
    fold_ids = np.asarray(
        [[f"outer-{fold}"] * 63 for fold in range(6)],
        dtype=object,
    )
    return score_dates, fold_ids


def _within_fold_slope(dependent: np.ndarray, regressor: np.ndarray) -> float:
    centered_dependent = dependent - dependent.mean(axis=1, keepdims=True)
    centered_regressor = regressor - regressor.mean(axis=1, keepdims=True)
    return float(
        np.sum(centered_dependent * centered_regressor)
        / np.sum(centered_regressor * centered_regressor)
    )


def _hashes(
    panel: tuple[np.ndarray, ...],
    plan: M03RInferencePlan,
) -> tuple[str, str]:
    policy, benchmark, risk_free, market, factors = panel
    score_dates, fold_ids = _metadata()
    common = m03r_common_evaluator_inputs_sha256(
        score_dates=score_dates,
        fold_ids=fold_ids,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factors,
        plan=plan,
    )
    candidate = m03r_candidate_policy_returns_sha256(
        policy_net_returns=policy,
        common_evaluator_inputs_sha256=common,
        plan=plan,
    )
    return common, candidate


def _evaluate(
    panel: tuple[np.ndarray, ...],
    *,
    plan: M03RInferencePlan | None = None,
    common_evaluator_inputs_sha256: str | None = None,
    candidate_policy_returns_sha256: str | None = None,
) -> dict[str, object]:
    policy, benchmark, risk_free, market, factors = panel
    resolved_plan = _plan() if plan is None else plan
    score_dates, fold_ids = _metadata()
    common, candidate = _hashes(panel, resolved_plan)
    return evaluate_m03r_inference(
        setting_id="M03R-active-alpha-hold30",
        score_dates=score_dates,
        fold_ids=fold_ids,
        policy_net_returns=policy,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factors,
        plan=resolved_plan,
        common_evaluator_inputs_sha256=(
            common
            if common_evaluator_inputs_sha256 is None
            else common_evaluator_inputs_sha256
        ),
        candidate_policy_returns_sha256=(
            candidate
            if candidate_policy_returns_sha256 is None
            else candidate_policy_returns_sha256
        ),
    )


def _rehash(receipt: dict[str, object]) -> None:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()


def test_m03r_inference_is_versioned_content_bound_and_active_beta_aware() -> None:
    panel = _panel()
    policy, benchmark, risk_free, market, _ = panel
    receipt = _evaluate(panel)
    validate_m03r_inference_receipt(receipt)
    assert receipt["promotion_authorized"] is False
    assert receipt["promotion_blockers"] == [
        "multiplicity_adjusted_factor_alpha_family_not_bound",
        "outer_data_role_and_access_receipts_not_bound",
        "empirical_capacity_and_closed_loop_40bp_not_bound",
    ]
    market_excess = market - risk_free
    active = policy - benchmark
    expected_beta = _within_fold_slope(active, market_excess)
    assert receipt["active_market_regression"]["active_market_beta"] == pytest.approx(
        expected_beta
    )
    expected_portfolio_beta = _within_fold_slope(
        policy - risk_free, market_excess
    )
    expected_benchmark_beta = _within_fold_slope(
        benchmark - risk_free, market_excess
    )
    beta = receipt["market_beta_diagnostics"]
    assert beta["portfolio_market_beta"] == pytest.approx(expected_portfolio_beta)
    assert beta["benchmark_market_beta"] == pytest.approx(expected_benchmark_beta)
    assert beta["active_market_beta"] == pytest.approx(expected_beta)
    assert beta["active_beta_standard_error"] >= 0.0
    assert beta["active_beta_equivalence_upper_bound"] >= abs(expected_beta)
    assert beta["active_beta_constraint_satisfied"] is True
    assert (
        receipt["active_market_regression"]["hac"]["30"]["loading_se"][
            "PIT_CAP_MARKET_EXCESS"
        ]
        >= 0.0
    )
    assert (
        receipt["active_market_regression"]["hac"]["30"][
            "active_beta_standard_error"
        ]
        >= 0.0
    )
    assert (
        receipt["active_market_regression"]["hac"]["30"][
            "cross_fold_lag_pairs"
        ]
        == 0
    )
    assert receipt["inference_plan"]["regression_fold_treatment"] == (
        "six-fold-fixed-effects-effect-coded-common-intercept"
    )
    interval = receipt["bootstrap"]["21"]
    assert interval["one_sided_confidence_level"] == pytest.approx(0.95)
    assert isinstance(interval["active_mean_log_return_daily_lcb"], float)
    assert isinstance(interval["active_multifactor_alpha_daily_lcb"], float)
    assert interval["active_multifactor_alpha_annualized_lcb"] == pytest.approx(
        252.0 * interval["active_multifactor_alpha_daily_lcb"]
    )
    assert isinstance(interval["policy_minus_benchmark_sharpe_lcb"], float)
    assert set(receipt["bootstrap"]) == {"10", "21", "30"}

    portfolio_factor = receipt["portfolio_multifactor_regression"]
    benchmark_factor = receipt["benchmark_multifactor_regression"]
    active_factor = receipt["active_multifactor_regression"]
    assert active_factor["alpha_daily"] == pytest.approx(
        portfolio_factor["alpha_daily"] - benchmark_factor["alpha_daily"]
    )
    for factor_name in ("PIT_CAP_MARKET_EXCESS", "SIZE", "VALUE"):
        assert active_factor["loadings"][factor_name] == pytest.approx(
            portfolio_factor["loadings"][factor_name]
            - benchmark_factor["loadings"][factor_name]
        )

    changed = deepcopy(receipt)
    changed["active_market_regression"]["active_market_beta"] += 1e-12
    with pytest.raises(M03REvaluationError, match="beta does not match its loading"):
        validate_m03r_inference_receipt(changed)


def test_corrected_finite_control_tail_counts_ties_and_stays_diagnostic() -> None:
    passing = finite_control_diagnostic(1.0, np.asarray([0.0] * 62 + [1.0, 2.0]))
    assert passing["diagnostic_kind"] == "matched-placebo-diagnostic"
    assert passing["exact_randomization_inference"] is False
    assert passing["control_count_greater_than_or_equal_to_candidate"] == 2
    assert passing["finite_control_p_value"] == pytest.approx(3.0 / 65.0)
    assert passing["passes_nominal_5_percent_point_gate"] is True

    failing = finite_control_diagnostic(
        1.0,
        np.asarray([0.0] * 61 + [1.0, 1.0, 2.0]),
    )
    assert failing["control_count_greater_than_or_equal_to_candidate"] == 3
    assert failing["finite_control_p_value"] == pytest.approx(4.0 / 65.0)
    assert failing["passes_nominal_5_percent_point_gate"] is False


def test_multifactor_promotion_evidence_uses_active_not_portfolio_alpha() -> None:
    rng = np.random.default_rng(81)
    shape = (6, 63)
    risk_free = np.full(shape, 0.0001)
    market = rng.normal(0.0004, 0.008, size=shape)
    market_excess = market - risk_free
    factors = rng.normal(0.0, 0.003, size=(*shape, 2))
    benchmark = (
        risk_free + 0.0005 + 0.90 * market_excess + 0.20 * factors[..., 0]
    )
    active = -0.0001 + 0.04 * market_excess + 0.10 * factors[..., 1]
    policy = benchmark + active
    receipt = _evaluate((policy, benchmark, risk_free, market, factors))

    portfolio = receipt["portfolio_multifactor_regression"]
    c1 = receipt["benchmark_multifactor_regression"]
    active_regression = receipt["active_multifactor_regression"]
    assert portfolio["alpha_daily"] == pytest.approx(0.0004)
    assert c1["alpha_daily"] == pytest.approx(0.0005)
    assert active_regression["alpha_daily"] == pytest.approx(-0.0001)
    assert receipt["bootstrap"]["21"][
        "active_multifactor_alpha_daily_lcb"
    ] == pytest.approx(-0.0001)


def test_active_beta_gate_uses_one_sided_equivalence_uncertainty() -> None:
    policy, benchmark, risk_free, market, factors = _panel()
    x = market - risk_free
    centered_x = x - x.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(29)
    residual = rng.normal(0.0, 0.002, size=x.shape)
    residual -= residual.mean(axis=1, keepdims=True)
    for fold in range(6):
        residual[fold] -= centered_x[fold] * (
            np.dot(residual[fold], centered_x[fold])
            / np.dot(centered_x[fold], centered_x[fold])
        )
    active = 0.095 * x + residual
    policy = benchmark + active
    receipt = _evaluate((policy, benchmark, risk_free, market, factors))
    diagnostics = receipt["market_beta_diagnostics"]

    assert diagnostics["active_market_beta"] == pytest.approx(0.095)
    assert diagnostics["active_beta_point_constraint_satisfied"] is True
    assert diagnostics["active_beta_equivalence_upper_bound"] > 0.10
    assert diagnostics["active_beta_constraint_satisfied"] is False


def test_fold_fixed_effects_absorb_fold_levels_and_common_alpha_is_mean() -> None:
    _, benchmark, risk_free, market, factors = _panel()
    market_excess = market - risk_free
    fold_intercepts = np.asarray(
        [-0.0005, -0.0003, -0.0001, 0.0001, 0.0003, 0.0005]
    )
    active = fold_intercepts[:, None] + 0.05 * market_excess
    receipt = _evaluate(
        (benchmark + active, benchmark, risk_free, market, factors)
    )
    regression = receipt["active_market_regression"]

    assert regression["active_market_beta"] == pytest.approx(0.05)
    assert regression["alpha_daily"] == pytest.approx(fold_intercepts.mean())
    assert regression["fold_effects"] == pytest.approx(fold_intercepts)
    assert sum(regression["fold_effects"]) == pytest.approx(0.0, abs=1e-15)
    assert all(
        row["cross_fold_lag_pairs"] == 0
        for row in regression["hac"].values()
    )


def test_m03r_inference_rejects_v3_identity() -> None:
    panel = _panel()
    policy, benchmark, risk_free, market, factors = panel
    plan = _plan()
    score_dates, fold_ids = _metadata()
    common, candidate = _hashes(panel, plan)
    with pytest.raises(ValueError, match="cannot identify an M03R artifact"):
        evaluate_m03r_inference(
            setting_id="hold30a-m03-alpha-core",
            score_dates=score_dates,
            fold_ids=fold_ids,
            policy_net_returns=policy,
            benchmark_net_returns=benchmark,
            risk_free_returns=risk_free,
            market_total_returns=market,
            factor_returns=factors,
            plan=plan,
            common_evaluator_inputs_sha256=common,
            candidate_policy_returns_sha256=candidate,
        )


def test_inference_plan_hash_is_derived_from_exact_return_semantics() -> None:
    plan = _plan()
    assert plan.inference_contract_sha256 != _digest("2")
    assert plan.semantics()["factor_return_conventions"] == [
        {
            "factor_name": "SIZE",
            "return_convention": "daily-simple-long-short-return",
        },
        {
            "factor_name": "VALUE",
            "return_convention": "daily-simple-long-short-return",
        },
    ]
    with pytest.raises(M03REvaluationError, match="exact plan semantics"):
        replace(plan, bootstrap_replicates=1_001)

    alternative = build_m03r_inference_plan(
        factor_names=("SIZE", "VALUE"),
        factor_return_conventions=(
            "daily-simple-long-short-return",
            "daily-simple-excess-return",
        ),
        bootstrap_replicates=1_000,
        bootstrap_seed_sha256=_digest("1"),
        one_sided_alpha=0.05,
    )
    assert alternative.inference_contract_sha256 != plan.inference_contract_sha256
    assert plan.primary_bootstrap_block_length_trading_sessions == 21
    assert plan.sensitivity_bootstrap_block_lengths_trading_sessions == (10, 30)
    assert plan.bootstrap_block_lengths_trading_sessions == (21, 10, 30)

    with pytest.raises(M03REvaluationError, match="shorter than a complete fold"):
        build_m03r_inference_plan(
            factor_names=("SIZE", "VALUE"),
            factor_return_conventions=(
                "daily-simple-long-short-return",
                "daily-simple-long-short-return",
            ),
            bootstrap_replicates=1_000,
            bootstrap_seed_sha256=_digest("1"),
            one_sided_alpha=0.05,
            primary_bootstrap_block_length_trading_sessions=63,
        )


def test_factor_name_cannot_collide_with_market_excess_name() -> None:
    with pytest.raises(M03REvaluationError, match="cannot collide"):
        build_m03r_inference_plan(
            factor_names=("PIT_CAP_MARKET_EXCESS",),
            factor_return_conventions=("daily-simple-excess-return",),
            bootstrap_replicates=1_000,
            bootstrap_seed_sha256=_digest("1"),
            one_sided_alpha=0.05,
        )


def test_common_and_candidate_hashes_are_recomputed_independently() -> None:
    panel = _panel()
    plan = _plan()
    common, candidate = _hashes(panel, plan)
    mutated = tuple(value.copy() for value in panel)
    mutated[0][0, 0] += 1e-12
    with pytest.raises(M03REvaluationError, match="supplied policy path"):
        _evaluate(
            mutated,
            plan=plan,
            common_evaluator_inputs_sha256=common,
            candidate_policy_returns_sha256=candidate,
        )

    mutated = tuple(value.copy() for value in panel)
    mutated[1][0, 0] += 1e-12
    with pytest.raises(M03REvaluationError, match="supplied common arrays"):
        _evaluate(
            mutated,
            plan=plan,
            common_evaluator_inputs_sha256=common,
        )


def test_common_hash_rejects_ambiguous_fold_and_date_metadata() -> None:
    _, benchmark, risk_free, market, factors = _panel()
    plan = _plan()
    score_dates, fold_ids = _metadata()
    bad_folds = fold_ids.copy()
    bad_folds[0, 1] = "outer-1"
    with pytest.raises(M03REvaluationError, match="exactly one fold"):
        m03r_common_evaluator_inputs_sha256(
            score_dates=score_dates,
            fold_ids=bad_folds,
            benchmark_net_returns=benchmark,
            risk_free_returns=risk_free,
            market_total_returns=market,
            factor_returns=factors,
            plan=plan,
        )

    bad_dates = score_dates.copy()
    bad_dates[0, 1] = bad_dates[0, 0]
    with pytest.raises(M03REvaluationError, match="strictly increasing"):
        m03r_common_evaluator_inputs_sha256(
            score_dates=bad_dates,
            fold_ids=fold_ids,
            benchmark_net_returns=benchmark,
            risk_free_returns=risk_free,
            market_total_returns=market,
            factor_returns=factors,
            plan=plan,
        )


def test_primary_multifactor_regression_rejects_rank_deficiency() -> None:
    policy, benchmark, risk_free, market, factors = _panel()
    factors[..., 0] = market - risk_free
    panel = (policy, benchmark, risk_free, market, factors)
    with pytest.raises(
        M03REvaluationError,
        match="primary fold-fixed-effect regression.*rank deficient",
    ):
        _evaluate(panel)


def test_primary_multifactor_regression_rejects_underdetermined_design() -> None:
    policy, benchmark, risk_free, market, _ = _panel()
    factor_names = tuple(f"F{index}" for index in range(377))
    plan = build_m03r_inference_plan(
        factor_names=factor_names,
        factor_return_conventions=("daily-simple-long-short-return",) * 377,
        bootstrap_replicates=1_000,
        bootstrap_seed_sha256=_digest("1"),
        one_sided_alpha=0.05,
    )
    factors = np.zeros((6, 63, 377), dtype=np.float64)
    panel = (policy, benchmark, risk_free, market, factors)
    with pytest.raises(M03REvaluationError, match="underdetermined"):
        _evaluate(panel, plan=plan)


def test_bootstrap_multifactor_regression_rejects_rank_deficiency() -> None:
    policy, benchmark, risk_free, market, factors = _panel()
    factors[..., 0] = 0.0
    factors[0, 0, 0] = 1.0
    panel = (policy, benchmark, risk_free, market, factors)
    with pytest.raises(
        M03REvaluationError,
        match="bootstrap .* multifactor regression replicate .*rank deficient",
    ):
        _evaluate(panel)


def test_receipt_validator_rejects_minimal_rehash_and_wrong_field_types() -> None:
    minimal: dict[str, object] = {
        "schema": M03R_EVALUATION_SCHEMA,
        "promotion_authorized": False,
    }
    _rehash(minimal)
    with pytest.raises(M03REvaluationError, match="key schema mismatch"):
        validate_m03r_inference_receipt(minimal)

    receipt = _evaluate(_panel())
    receipt["bootstrap_replicates"] = 1_000.0
    _rehash(receipt)
    with pytest.raises(M03REvaluationError, match="bootstrap replicate count"):
        validate_m03r_inference_receipt(receipt)


def test_receipt_validator_rejects_unknown_nested_keys_even_when_rehashed() -> None:
    receipt = _evaluate(_panel())
    receipt["market_beta_diagnostics"]["unbound_field"] = 0.0
    _rehash(receipt)
    with pytest.raises(M03REvaluationError, match="key schema mismatch"):
        validate_m03r_inference_receipt(receipt)


def test_receipt_validator_rejects_portfolio_active_factor_inconsistency() -> None:
    receipt = _evaluate(_panel())
    receipt["active_multifactor_regression"]["alpha_daily"] += 1e-8
    receipt["active_multifactor_regression"][
        "alpha_annualized_arithmetic"
    ] += 252e-8
    _rehash(receipt)
    with pytest.raises(M03REvaluationError, match="portfolio-minus-C1"):
        validate_m03r_inference_receipt(receipt)


def test_receipt_validator_enforces_nested_integer_types_after_rehash() -> None:
    receipt = _evaluate(_panel())
    receipt["inference_plan"]["outer_fold_count"] = 6.0
    _rehash(receipt)
    with pytest.raises(M03REvaluationError, match="must be an integer"):
        validate_m03r_inference_receipt(receipt)
