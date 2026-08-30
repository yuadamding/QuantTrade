from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

import rl_quant.execution.massive_adaptive_portfolio_compiler_v1 as compiler_module
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
    MassiveAdaptivePortfolioCompilerError,
    MassiveAdaptivePortfolioCompilerInputsV1,
    compile_massive_adaptive_portfolio_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    assert_adaptive_import_firewall,
    assert_no_adaptive_hold_semantics,
)


_BUCKETS = 7


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _cube_tuple(
    value: np.ndarray,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(_matrix_tuple(matrix) for matrix in value)


def _inputs(
    *,
    expected_first_bucket: tuple[float, ...],
    pretrade: tuple[float, ...],
    benchmark: tuple[float, ...] | None = None,
    issuer_ids: tuple[str, ...] | None = None,
    adv: tuple[float, ...] | None = None,
    entry_cost_bps: tuple[float, ...] | None = None,
    current_exit_cost_bps: tuple[float, ...] | None = None,
    buy_eligible: tuple[bool, ...] | None = None,
    forced_exit: tuple[bool, ...] | None = None,
    active_betas: tuple[float, ...] | None = None,
    risk_covariance: np.ndarray | None = None,
    expected_buckets: np.ndarray | None = None,
) -> MassiveAdaptivePortfolioCompilerInputsV1:
    count = len(expected_first_bucket)
    security_ids = tuple(f"SEC-{index:02d}" for index in range(count))
    expected = np.zeros((count, _BUCKETS), dtype=np.float64)
    expected[:, 0] = expected_first_bucket
    if expected_buckets is not None:
        expected = expected_buckets.astype(np.float64, copy=True)
    bucket_covariance = np.zeros((count, _BUCKETS, _BUCKETS), dtype=np.float64)
    risk = (
        np.zeros((count, count), dtype=np.float64)
        if risk_covariance is None
        else risk_covariance.astype(np.float64, copy=True)
    )
    return MassiveAdaptivePortfolioCompilerInputsV1(
        decision_id="2025-06-30T20:00:00Z",
        security_ids=security_ids,
        issuer_ids=issuer_ids or security_ids,
        bucket_expected_residual_returns=_matrix_tuple(expected),
        bucket_covariances=_cube_tuple(bucket_covariance),
        pretrade_weights=pretrade,
        benchmark_weights=benchmark or pretrade,
        risk_covariance=_matrix_tuple(risk),
        active_betas=active_betas or tuple(0.0 for _ in range(count)),
        trailing_adv_notional=adv or tuple(1_000_000.0 for _ in range(count)),
        entry_cost_basis_points=entry_cost_bps
        or tuple(0.0 for _ in range(count)),
        current_exit_cost_basis_points=current_exit_cost_bps
        or tuple(0.0 for _ in range(count)),
        expected_future_exit_cost_basis_points=tuple(
            tuple(0.0 for _ in range(_BUCKETS)) for _ in range(count)
        ),
        buy_eligible=buy_eligible or tuple(True for _ in range(count)),
        forced_exit=forced_exit or tuple(False for _ in range(count)),
        capital=100.0,
        forecast_receipt_sha256=semantic_sha256("forecast"),
        risk_receipt_sha256=semantic_sha256("risk"),
        cost_receipt_sha256=semantic_sha256("cost"),
        portfolio_state_receipt_sha256=semantic_sha256("book"),
        eligibility_receipt_sha256=semantic_sha256("eligibility"),
    )


def _config(**changes: object) -> MassiveAdaptivePortfolioCompilerConfigV1:
    base = MassiveAdaptivePortfolioCompilerConfigV1(
        maximum_security_weight=1.0,
        maximum_issuer_weight=1.0,
        tracking_error_limit_annualized=1.0,
        absolute_active_beta_limit=1.0,
        maximum_daily_one_way_turnover=1.0,
        maximum_adv_participation=1.0,
        uncertainty_standard_deviations=0.0,
        risk_aversion=0.0,
        tail_risk_aversion=0.0,
        solver_step_size=0.5,
        numerical_tolerance=1.0e-8,
    )
    return replace(base, **changes)


def test_sunk_entry_cost_retains_only_until_replacement_is_better() -> None:
    current_is_better = _inputs(
        expected_first_bucket=(0.02, 0.025),
        pretrade=(1.0, 0.0),
        entry_cost_bps=(0.0, 100.0),
        current_exit_cost_bps=(50.0, 0.0),
    )
    retained = compile_massive_adaptive_portfolio_v1(
        current_is_better, config=_config()
    )

    assert retained.target_weights == pytest.approx((1.0, 0.0), abs=2.0e-7)
    assert retained.existing_position_net_values[0] == pytest.approx(0.02)
    assert retained.buy_net_values[1] == pytest.approx(0.015)

    replacement_is_better = replace(
        current_is_better,
        bucket_expected_residual_returns=(
            current_is_better.bucket_expected_residual_returns[0],
            (0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    replaced = compile_massive_adaptive_portfolio_v1(
        replacement_is_better, config=_config()
    )

    assert replaced.target_weights == pytest.approx((0.0, 1.0), abs=2.0e-7)
    assert replaced.discretionary_one_way_turnover == pytest.approx(1.0)


def test_policy_can_exit_after_one_session() -> None:
    reversal = _inputs(
        expected_first_bucket=(-0.05, 0.04),
        pretrade=(1.0, 0.0),
        benchmark=(0.0, 0.0),
        entry_cost_bps=(0.0, 0.0),
        current_exit_cost_bps=(0.0, 0.0),
    )
    decision = compile_massive_adaptive_portfolio_v1(reversal, config=_config())

    assert decision.target_weights == pytest.approx((0.0, 1.0), abs=2.0e-7)
    assert decision.discretionary_sell_weights[0] == pytest.approx(1.0)


def test_policy_can_hold_indefinitely_when_alpha_remains_positive() -> None:
    inputs = _inputs(
        expected_first_bucket=(0.03, -0.02),
        pretrade=(1.0, 0.0),
        benchmark=(1.0, 0.0),
        entry_cost_bps=(0.0, 50.0),
        current_exit_cost_bps=(20.0, 0.0),
    )
    for session_index in range(64):
        decision = compile_massive_adaptive_portfolio_v1(inputs, config=_config())
        assert decision.target_weights == pytest.approx((1.0, 0.0), abs=2.0e-7)
        inputs = replace(
            inputs,
            pretrade_weights=decision.target_weights,
            portfolio_state_receipt_sha256=semantic_sha256(
                ("persistent-positive-alpha", session_index)
            ),
        )


def test_capacity_is_in_the_optimizer_and_reallocates_to_next_best_name() -> None:
    inputs = _inputs(
        expected_first_bucket=(0.10, 0.08, -0.10),
        pretrade=(0.0, 0.0, 0.0),
        benchmark=(0.0, 0.0, 0.0),
        adv=(200.0, 1_000.0, 1_000.0),
    )
    decision = compile_massive_adaptive_portfolio_v1(
        inputs,
        config=_config(maximum_adv_participation=0.10),
    )

    assert decision.target_weights[0] == pytest.approx(0.20, abs=2.0e-7)
    assert decision.target_weights[1] == pytest.approx(0.80, abs=2.0e-7)
    assert decision.target_weights[2] == pytest.approx(0.0, abs=2.0e-7)
    assert decision.cash_weight == pytest.approx(0.0, abs=2.0e-7)
    assert decision.maximum_intended_participation <= 0.10 + 2.0e-8


def test_forced_exit_is_separate_from_discretionary_turnover() -> None:
    inputs = _inputs(
        expected_first_bucket=(0.0, 0.10),
        pretrade=(0.5, 0.5),
        benchmark=(0.0, 0.5),
        buy_eligible=(False, True),
        forced_exit=(True, False),
    )
    decision = compile_massive_adaptive_portfolio_v1(
        inputs,
        config=_config(maximum_daily_one_way_turnover=0.10),
    )

    assert decision.repaired_pretrade_weights == pytest.approx((0.0, 0.5))
    assert decision.forced_exit_weights == pytest.approx((0.5, 0.0))
    assert decision.forced_one_way_turnover == pytest.approx(0.5)
    assert decision.target_weights == pytest.approx((0.0, 0.6), abs=2.0e-7)
    assert decision.discretionary_one_way_turnover == pytest.approx(0.1, abs=2.0e-7)


def test_integrated_book_respects_issuer_beta_tracking_error_and_turnover() -> None:
    covariance = np.diag([0.01, 0.01, 0.01])
    inputs = _inputs(
        expected_first_bucket=(0.10, 0.09, -0.10),
        pretrade=(0.30, 0.30, 0.40),
        benchmark=(0.30, 0.30, 0.40),
        issuer_ids=("ISSUER-X", "ISSUER-X", "ISSUER-Y"),
        active_betas=(1.0, 1.0, -1.0),
        risk_covariance=covariance,
    )
    config = _config(
        maximum_security_weight=0.60,
        maximum_issuer_weight=0.65,
        maximum_daily_one_way_turnover=0.10,
        absolute_active_beta_limit=0.05,
        tracking_error_limit_annualized=0.20,
    )
    decision = compile_massive_adaptive_portfolio_v1(inputs, config=config)

    target = np.asarray(decision.target_weights)
    assert target[:2].sum() <= 0.65 + 2.0e-8
    assert target.max() <= 0.60 + 2.0e-8
    assert decision.discretionary_one_way_turnover <= 0.10 + 2.0e-8
    assert abs(decision.active_beta) <= 0.05 + 2.0e-8
    assert decision.annualized_tracking_error <= 0.20 + 2.0e-8
    assert decision.primal_residual <= 2.0e-8


def test_bucket_choice_is_diagnostic_and_recomputed_from_the_full_curve() -> None:
    expected = np.asarray([[0.01, -0.02, 0.04, -0.01, 0.0, 0.0, 0.0]])
    inputs = _inputs(
        expected_first_bucket=(0.0,),
        expected_buckets=expected,
        pretrade=(0.0,),
        benchmark=(0.0,),
    )
    decision = compile_massive_adaptive_portfolio_v1(inputs, config=_config())

    assert decision.best_buy_bucket_ids == ("B06_10",)
    assert decision.best_existing_position_bucket_ids == ("B06_10",)
    assert decision.target_weights == pytest.approx((1.0,), abs=2.0e-7)


def test_receipt_is_deterministic_and_engineering_result_is_nonauthorizing() -> None:
    inputs = _inputs(
        expected_first_bucket=(0.03, -0.01),
        pretrade=(0.0, 0.0),
        benchmark=(0.0, 0.0),
    )
    first = compile_massive_adaptive_portfolio_v1(inputs, config=_config())
    second = compile_massive_adaptive_portfolio_v1(inputs, config=_config())

    assert first == second
    assert not first.economic_optimization_authorized
    assert not first.profitability_reporting_authorized
    assert not first.lockbox_access_authorized
    assert not first.reinforcement_learning_authorized
    with pytest.raises(MassiveAdaptivePortfolioCompilerError, match="receipt differs"):
        replace(first, target_weights=(0.0, 0.0)).validate()


def test_no_minimum_holding_period_in_compiler() -> None:
    config = _config()
    assert_no_adaptive_hold_semantics(config)
    assert_adaptive_import_firewall([Path(compiler_module.__file__)])
    forbidden_fragments = ("age", "duration", "persistence", "scheduled_exit")
    assert all(
        not any(fragment in field.name for fragment in forbidden_fragments)
        for field in fields(MassiveAdaptivePortfolioCompilerConfigV1)
    )


def test_default_limits_match_the_frozen_adaptive_source_contract() -> None:
    config = MassiveAdaptivePortfolioCompilerConfigV1()

    assert config.maximum_security_weight == (
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.maximum_security_weight
    )
    assert config.maximum_issuer_weight == (
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.maximum_issuer_weight
    )
    assert config.tracking_error_limit_annualized == (
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.tracking_error_limit_annualized
    )
    assert config.absolute_active_beta_limit == (
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.absolute_active_beta_limit
    )
    assert config.maximum_daily_one_way_turnover == (
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.maximum_daily_one_way_turnover
    )
    assert config.maximum_adv_participation == (
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.maximum_adv_participation
    )


def test_infeasible_pretrade_cap_and_capacity_fail_closed() -> None:
    inputs = _inputs(
        expected_first_bucket=(0.01,),
        pretrade=(0.80,),
        benchmark=(0.80,),
        adv=(1.0,),
    )
    with pytest.raises(
        MassiveAdaptivePortfolioCompilerError,
        match="cannot repair the pretrade book",
    ):
        compile_massive_adaptive_portfolio_v1(
            inputs,
            config=_config(
                maximum_security_weight=0.50,
                maximum_issuer_weight=0.50,
                maximum_adv_participation=0.01,
            ),
        )
