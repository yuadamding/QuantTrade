from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

import rl_quant.training.adaptive_alpha_profit_checkpoint_selection_v1 as selection_module
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    assert_adaptive_import_firewall,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.adaptive_alpha_profit_checkpoint_selection_v1 import (
    MassiveAdaptiveProfitCheckpointSelectionConfigV1,
    MassiveAdaptiveProfitCheckpointSelectionError,
    build_massive_adaptive_profit_validation_trace_v1,
    evaluate_massive_adaptive_profit_checkpoint_v1,
    select_massive_adaptive_profit_checkpoint_v1,
)


_DATES = ("2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03")


def _config(**changes: object) -> MassiveAdaptiveProfitCheckpointSelectionConfigV1:
    base = MassiveAdaptiveProfitCheckpointSelectionConfigV1(
        minimum_validation_sessions=4,
        maximum_date_gross_profit_share=1.0,
        maximum_year_gross_profit_share=1.0,
        maximum_sector_gross_profit_share=1.0,
        maximum_security_gross_profit_share=1.0,
    )
    return replace(base, **changes)


def _trace(
    epoch: int,
    *,
    net20: tuple[float, ...] = (0.001, 0.001, 0.001, 0.001),
    net40: tuple[float, ...] | None = None,
    benchmark20: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    factor: tuple[float, ...] = (0.0005, 0.0005, 0.0005, 0.0005),
    rank_ic: tuple[float, ...] = (0.02,) * 7,
    calibration_error: float = 0.01,
    eligible: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9),
    constraint: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    capacity_loss: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    turnover: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02),
    gross: tuple[float, ...] = (0.001, 0.001, 0.001, 0.001),
    sector_contributions: tuple[tuple[str, float], ...] | None = None,
    security_contributions: tuple[tuple[str, float], ...] | None = None,
    model_parameter_count: int = 1_000,
    decision_salt: str = "common",
    validation_plan_salt: str = "common",
    outer_test_accessed: bool = False,
) -> object:
    gross_total = sum(gross)
    sector = sector_contributions or (
        ("SECTOR-A", 0.25 * gross_total),
        ("SECTOR-B", 0.25 * gross_total),
        ("SECTOR-C", 0.25 * gross_total),
        ("SECTOR-D", 0.25 * gross_total),
    )
    security = security_contributions or (
        ("SEC-00", 0.25 * gross_total),
        ("SEC-01", 0.25 * gross_total),
        ("SEC-02", 0.25 * gross_total),
        ("SEC-03", 0.25 * gross_total),
    )
    return build_massive_adaptive_profit_validation_trace_v1(
        setting_id="AD11",
        fold_index=0,
        epoch_index=epoch,
        model_parameter_count=model_parameter_count,
        checkpoint_state_receipt_sha256=semantic_sha256(("state", epoch)),
        checkpoint_source_receipt_sha256=semantic_sha256(("checkpoint", epoch)),
        prediction_receipt_sha256=semantic_sha256(("prediction", epoch)),
        source_bundle_receipt_sha256=semantic_sha256("source-bundle"),
        validation_plan_receipt_sha256=semantic_sha256(
            ("validation-plan", validation_plan_salt)
        ),
        compiler_config_receipt_sha256=semantic_sha256("compiler-config"),
        factor_model_receipt_sha256=semantic_sha256("factor-model"),
        attribution_receipt_sha256=semantic_sha256(("attribution", epoch)),
        session_dates=_DATES,
        validation_block_ids=(0, 0, 1, 1),
        compiler_decision_receipts=tuple(
            semantic_sha256(("decision", decision_salt, session_date))
            for session_date in _DATES
        ),
        portfolio_net_returns_20bp=net20,
        portfolio_net_returns_40bp=net40 or net20,
        benchmark_net_returns_20bp=benchmark20,
        factor_residual_returns_20bp=factor,
        gross_signal_returns=gross,
        eligible_fractions=eligible,
        constraint_violations=constraint,
        capacity_lost_notional_fractions=capacity_loss,
        one_way_turnovers=turnover,
        rank_ic_by_bucket=rank_ic,
        calibration_error=calibration_error,
        sector_gross_return_contributions=sector,
        security_gross_return_contributions=security,
        outer_test_accessed=outer_test_accessed,
    )


def test_checkpoint_selection_prefers_executable_profit_over_higher_rank_ic() -> None:
    high_ic_low_profit = _trace(
        0,
        net20=(0.0005,) * 4,
        rank_ic=(0.10,) * 7,
    )
    lower_ic_high_profit = _trace(
        1,
        net20=(0.0010,) * 4,
        rank_ic=(0.02,) * 7,
    )

    selected = select_massive_adaptive_profit_checkpoint_v1(
        (high_ic_low_profit, lower_ic_high_profit),
        config=_config(),
    )

    assert selected.selected_epoch_index == 1
    assert selected.selected_checkpoint_state_receipt_sha256 == (
        lower_ic_high_profit.checkpoint_state_receipt_sha256
    )
    assert not selected.economic_training_authorized
    assert not selected.outer_evaluation_authorized
    assert not selected.profitability_reporting_authorized
    assert not selected.lockbox_access_authorized
    assert not selected.reinforcement_learning_authorized


def test_ineligible_high_profit_candidate_cannot_win() -> None:
    valid = _trace(0, net20=(0.001,) * 4, net40=(0.0002,) * 4)
    invalid = _trace(
        1,
        net20=(0.01,) * 4,
        net40=(-0.02,) * 4,
        gross=(0.02,) * 4,
    )

    invalid_metrics = evaluate_massive_adaptive_profit_checkpoint_v1(
        invalid, config=_config()
    )
    selected = select_massive_adaptive_profit_checkpoint_v1(
        (valid, invalid), config=_config()
    )

    assert "net-return-40bp" in invalid_metrics.eligibility_failures
    assert not invalid_metrics.eligible
    assert selected.selected_epoch_index == 0
    assert selected.eligible_epoch_indices == (0,)


def test_factor_alpha_breaks_equal_profit_tie_before_risk_diagnostics() -> None:
    lower_alpha = _trace(0, factor=(0.0001,) * 4)
    higher_alpha = _trace(1, factor=(0.0008,) * 4)

    selected = select_massive_adaptive_profit_checkpoint_v1(
        (lower_alpha, higher_alpha), config=_config()
    )

    assert selected.selected_epoch_index == 1


def test_exact_economic_tie_selects_earliest_epoch() -> None:
    first = _trace(0)
    second = _trace(1)

    selected = select_massive_adaptive_profit_checkpoint_v1(
        (second, first), config=_config()
    )

    assert selected.selected_epoch_index == 0


def test_eligibility_ladder_reports_predictive_and_execution_failures() -> None:
    invalid = _trace(
        0,
        net20=(-0.001,) * 4,
        net40=(-0.002,) * 4,
        factor=(-0.0002,) * 4,
        rank_ic=(-0.01,) + (0.02,) * 6,
        calibration_error=0.20,
        eligible=(0.70,) * 4,
        constraint=(1.0e-4,) * 4,
        capacity_loss=(0.20,) * 4,
    )

    metrics = evaluate_massive_adaptive_profit_checkpoint_v1(
        invalid, config=_config()
    )

    assert metrics.eligibility_failures == (
        "coverage",
        "calibration",
        "rank-ic:B01",
        "net-return-20bp",
        "net-return-40bp",
        "factor-residual-alpha",
        "constraint-violation",
        "capacity-loss",
    )
    with pytest.raises(
        MassiveAdaptiveProfitCheckpointSelectionError,
        match="no checkpoint passed",
    ):
        select_massive_adaptive_profit_checkpoint_v1((invalid,), config=_config())


def test_profit_concentration_is_computed_from_daily_and_group_contributions() -> None:
    concentrated = _trace(
        0,
        net20=(0.004, 0.0, 0.0, 0.0),
        net40=(0.004, 0.0, 0.0, 0.0),
        gross=(0.004, 0.0, 0.0, 0.0),
        sector_contributions=(
            ("SECTOR-A", 0.0036),
            ("SECTOR-B", 0.0004),
        ),
        security_contributions=(
            ("SEC-00", 0.0036),
            ("SEC-01", 0.0004),
        ),
    )
    config = _config(
        maximum_date_gross_profit_share=0.60,
        maximum_year_gross_profit_share=1.0,
        maximum_sector_gross_profit_share=0.60,
        maximum_security_gross_profit_share=0.60,
    )

    metrics = evaluate_massive_adaptive_profit_checkpoint_v1(
        concentrated, config=config
    )

    assert metrics.maximum_date_gross_profit_share == pytest.approx(1.0)
    assert metrics.maximum_sector_gross_profit_share == pytest.approx(0.9)
    assert metrics.maximum_security_gross_profit_share == pytest.approx(0.9)
    assert metrics.eligibility_failures == (
        "profit-concentration:date",
        "profit-concentration:sector",
        "profit-concentration:security",
    )


def test_candidate_support_must_be_identical() -> None:
    first = _trace(0)
    changed_support = _trace(1, validation_plan_salt="changed")

    with pytest.raises(
        MassiveAdaptiveProfitCheckpointSelectionError,
        match="exact validation support",
    ):
        select_massive_adaptive_profit_checkpoint_v1(
            (first, changed_support), config=_config()
        )

    changed_eligibility = _trace(1, eligible=(0.85,) * 4)
    with pytest.raises(
        MassiveAdaptiveProfitCheckpointSelectionError,
        match="exact validation support",
    ):
        select_massive_adaptive_profit_checkpoint_v1(
            (first, changed_eligibility), config=_config()
        )


def test_checkpoint_specific_compiler_decisions_are_allowed() -> None:
    first = _trace(0, decision_salt="epoch-0")
    second = _trace(1, decision_salt="epoch-1", net20=(0.002,) * 4, gross=(0.002,) * 4)

    selected = select_massive_adaptive_profit_checkpoint_v1(
        (first, second), config=_config()
    )

    assert selected.selected_epoch_index == 1
    assert selected.candidate_epoch_indices == (0, 1)
    assert selected.selected_validation_trace_receipt_sha256 == (
        selected.candidate_validation_trace_receipts[1]
    )


def test_trace_receipt_detects_metric_mutation_and_outer_access_fails_closed() -> None:
    trace = _trace(0)
    with pytest.raises(
        MassiveAdaptiveProfitCheckpointSelectionError,
        match="receipt differs",
    ):
        replace(trace, calibration_error=0.02).validate()
    with pytest.raises(
        MassiveAdaptiveProfitCheckpointSelectionError,
        match="inner-validation evidence only",
    ):
        _trace(0, outer_test_accessed=True)


def test_cost_ladder_and_gross_net_reconciliation_fail_closed() -> None:
    with pytest.raises(
        MassiveAdaptiveProfitCheckpointSelectionError,
        match="40-bp returns",
    ):
        _trace(0, net20=(0.001,) * 4, net40=(0.002,) * 4, gross=(0.003,) * 4)
    with pytest.raises(
        MassiveAdaptiveProfitCheckpointSelectionError,
        match="exceeds gross",
    ):
        _trace(0, net20=(0.002,) * 4, net40=(0.001,) * 4, gross=(0.001,) * 4)


def test_selection_surface_has_no_duration_semantics_or_forbidden_imports() -> None:
    config = _config()
    assert_no_adaptive_hold_semantics(config)
    assert_adaptive_import_firewall((Path(selection_module.__file__),))
    forbidden_fragments = ("age", "duration", "persistence", "scheduled_exit")
    assert all(
        not any(fragment in field.name for fragment in forbidden_fragments)
        for field in fields(MassiveAdaptiveProfitCheckpointSelectionConfigV1)
    )
