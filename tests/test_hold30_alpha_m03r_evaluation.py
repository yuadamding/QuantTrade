from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from rl_quant.evaluation.hold30_alpha_m03r import (
    M03R_EVALUATION_SCHEMA,
    M03REvaluationError,
    M03RInferencePlan,
    build_m03r_inference_plan,
    evaluate_m03r_inference,
    finite_control_diagnostic,
    m03r_source_arrays_sha256,
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
        block_lengths_trading_sessions=(21,),
        bootstrap_replicates=1_000,
        bootstrap_seed_sha256=_digest("1"),
        one_sided_alpha=0.05,
    )


def _source_hash(panel: tuple[np.ndarray, ...], plan: M03RInferencePlan) -> str:
    policy, benchmark, risk_free, market, factors = panel
    return m03r_source_arrays_sha256(
        policy_net_returns=policy,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factors,
        plan=plan,
    )


def _evaluate(
    panel: tuple[np.ndarray, ...],
    *,
    plan: M03RInferencePlan | None = None,
    source_arrays_sha256: str | None = None,
) -> dict[str, object]:
    policy, benchmark, risk_free, market, factors = panel
    resolved_plan = _plan() if plan is None else plan
    return evaluate_m03r_inference(
        setting_id="M03R-active-alpha-hold30",
        policy_net_returns=policy,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factors,
        plan=resolved_plan,
        source_arrays_sha256=(
            _source_hash(panel, resolved_plan)
            if source_arrays_sha256 is None
            else source_arrays_sha256
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
    market_excess = (market - risk_free).reshape(-1)
    active = (policy - benchmark).reshape(-1)
    expected_beta = np.cov(active, market_excess, ddof=1)[0, 1] / np.var(
        market_excess, ddof=1
    )
    assert receipt["active_market_regression"]["active_market_beta"] == pytest.approx(
        expected_beta
    )
    expected_portfolio_beta = np.cov(
        (policy - risk_free).reshape(-1), market_excess, ddof=1
    )[0, 1] / np.var(market_excess, ddof=1)
    expected_benchmark_beta = np.cov(
        (benchmark - risk_free).reshape(-1), market_excess, ddof=1
    )[0, 1] / np.var(market_excess, ddof=1)
    beta = receipt["market_beta_diagnostics"]
    assert beta["portfolio_market_beta"] == pytest.approx(expected_portfolio_beta)
    assert beta["benchmark_market_beta"] == pytest.approx(expected_benchmark_beta)
    assert beta["active_market_beta"] == pytest.approx(expected_beta)
    assert beta["active_beta_standard_error"] >= 0.0
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
    interval = receipt["bootstrap"]["21"]
    assert interval["one_sided_confidence_level"] == pytest.approx(0.95)
    assert isinstance(interval["active_mean_log_return_daily_lcb"], float)
    assert isinstance(interval["multifactor_alpha_daily_lcb"], float)
    assert isinstance(interval["policy_minus_benchmark_sharpe_lcb"], float)

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


def test_m03r_inference_rejects_v3_identity() -> None:
    panel = _panel()
    policy, benchmark, risk_free, market, factors = panel
    plan = _plan()
    with pytest.raises(ValueError, match="cannot identify an M03R artifact"):
        evaluate_m03r_inference(
            setting_id="hold30a-m03-alpha-core",
            policy_net_returns=policy,
            benchmark_net_returns=benchmark,
            risk_free_returns=risk_free,
            market_total_returns=market,
            factor_returns=factors,
            plan=plan,
            source_arrays_sha256=_source_hash(panel, plan),
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
        block_lengths_trading_sessions=(21,),
        bootstrap_replicates=1_000,
        bootstrap_seed_sha256=_digest("1"),
        one_sided_alpha=0.05,
    )
    assert alternative.inference_contract_sha256 != plan.inference_contract_sha256


def test_factor_name_cannot_collide_with_market_excess_name() -> None:
    with pytest.raises(M03REvaluationError, match="cannot collide"):
        build_m03r_inference_plan(
            factor_names=("PIT_CAP_MARKET_EXCESS",),
            factor_return_conventions=("daily-simple-excess-return",),
            block_lengths_trading_sessions=(21,),
            bootstrap_replicates=1_000,
            bootstrap_seed_sha256=_digest("1"),
            one_sided_alpha=0.05,
        )


def test_source_array_hash_is_recomputed_internally() -> None:
    panel = _panel()
    plan = _plan()
    source_hash = _source_hash(panel, plan)
    mutated = tuple(value.copy() for value in panel)
    mutated[0][0, 0] += 1e-12
    with pytest.raises(M03REvaluationError, match="supplied canonical arrays"):
        _evaluate(mutated, plan=plan, source_arrays_sha256=source_hash)


def test_primary_multifactor_regression_rejects_rank_deficiency() -> None:
    policy, benchmark, risk_free, market, factors = _panel()
    factors[..., 0] = market - risk_free
    panel = (policy, benchmark, risk_free, market, factors)
    with pytest.raises(M03REvaluationError, match="primary regression.*rank deficient"):
        _evaluate(panel)


def test_primary_multifactor_regression_rejects_underdetermined_design() -> None:
    policy, benchmark, risk_free, market, _ = _panel()
    factor_names = tuple(f"F{index}" for index in range(377))
    plan = build_m03r_inference_plan(
        factor_names=factor_names,
        factor_return_conventions=("daily-simple-long-short-return",) * 377,
        block_lengths_trading_sessions=(21,),
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
        match="bootstrap multifactor regression replicate .*rank deficient",
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


def test_receipt_validator_enforces_nested_integer_types_after_rehash() -> None:
    receipt = _evaluate(_panel())
    receipt["inference_plan"]["outer_fold_count"] = 6.0
    _rehash(receipt)
    with pytest.raises(M03REvaluationError, match="must be an integer"):
        validate_m03r_inference_receipt(receipt)
