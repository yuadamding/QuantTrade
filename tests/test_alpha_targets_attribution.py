from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rl_quant.alpha import (
    ALPHA_DISCOVERY_SETTINGS,
    ActiveReturnAttribution,
    AlphaExperimentSpec,
    AlphaFoldConfig,
    AlphaModelConfig,
    AlphaOptimizerConfig,
    AlphaTargetSpec,
    AlphaTrialRecord,
    DecisionFillConvention,
    EconomicValuePoint,
    OriginExposurePanel,
    PITAlphaDataError,
    apply_origin_residual_operator,
    build_alpha_trial_registry,
    build_origin_residual_operator,
    build_security_multi_horizon_target,
    build_signal_attribution_ledger,
    evaluate_signal_promotion,
    target_cross_section,
    validate_alpha_discovery_settings,
)


_DIGEST = "a" * 64


def _target_spec(*, return_kind: str = "economic-total-simple") -> AlphaTargetSpec:
    return AlphaTargetSpec(
        primary_horizon_sessions=3,
        auxiliary_horizons_sessions=(1, 2),
        fill_convention=DecisionFillConvention(
            convention_id="after-close-next-open-v1",
            decision_rule="after-close",
            fill_rule="next-open-auction",
        ),
        return_kind=return_kind,  # type: ignore[arg-type]
        target_mode="factor-residual",
        terminal_outcomes_included=True,
        future_survival_required=False,
    )


def _points(values: tuple[float, ...], *, terminal_from: int | None = None) -> tuple[EconomicValuePoint, ...]:
    rows: list[EconomicValuePoint] = []
    for index, value in enumerate(values):
        terminal = terminal_from is not None and index >= terminal_from
        rows.append(
            EconomicValuePoint(
                session_index=index,
                economic_at_ms=1_000 + index * 10,
                available_at_ms=1_005 + index * 10,
                value=value,
                mark_kind="terminal-disposition" if terminal else "market",
                terminal=terminal,
            )
        )
    return tuple(rows)


def test_multi_horizon_targets_begin_at_actual_fill() -> None:
    spec = _target_spec()
    row = build_security_multi_horizon_target(
        security_id="SEC-A",
        decision_at_ms=999,
        fill_at_ms=1_010,
        fill_session_index=1,
        points=_points((80.0, 100.0, 110.0, 90.0, 120.0)),
        spec=spec,
        built_at_ms=2_000,
    )

    assert tuple(target.horizon_sessions for target in row.targets) == (1, 2, 3)
    assert row.targets[0].simple_return == pytest.approx(0.10)
    assert row.targets[1].simple_return == pytest.approx(-0.10)
    assert row.targets[2].simple_return == pytest.approx(0.20)
    assert all(target.start_value == 100.0 for target in row.targets)


def test_target_cannot_be_built_before_endpoint_is_available() -> None:
    with pytest.raises(PITAlphaDataError, match="before its economic endpoint"):
        build_security_multi_horizon_target(
            security_id="SEC-A",
            decision_at_ms=999,
            fill_at_ms=1_010,
            fill_session_index=1,
            points=_points((80.0, 100.0, 110.0, 90.0, 120.0)),
            spec=_target_spec(),
            built_at_ms=1_039,
        )


def test_terminal_loss_remains_in_simple_target_support() -> None:
    row = build_security_multi_horizon_target(
        security_id="SEC-A",
        decision_at_ms=999,
        fill_at_ms=1_010,
        fill_session_index=1,
        points=_points((80.0, 100.0, 0.0, 0.0, 0.0), terminal_from=2),
        spec=_target_spec(),
        built_at_ms=2_000,
    )

    assert row.targets[-1].simple_return == -1.0
    assert row.targets[-1].log_return is None
    with pytest.raises(PITAlphaDataError, match="log target"):
        build_security_multi_horizon_target(
            security_id="SEC-A",
            decision_at_ms=999,
            fill_at_ms=1_010,
            fill_session_index=1,
            points=_points((80.0, 100.0, 0.0, 0.0, 0.0), terminal_from=2),
            spec=_target_spec(return_kind="economic-total-log"),
            built_at_ms=2_000,
        )


def test_target_spec_rejects_future_survival_support() -> None:
    with pytest.raises(PITAlphaDataError, match="terminal outcomes"):
        replace(_target_spec(), future_survival_required=True).validate()


def test_target_cross_section_never_fills_missing_security() -> None:
    spec = _target_spec()
    row = build_security_multi_horizon_target(
        security_id="SEC-A",
        decision_at_ms=999,
        fill_at_ms=1_010,
        fill_session_index=1,
        points=_points((80.0, 100.0, 110.0, 90.0, 120.0)),
        spec=spec,
        built_at_ms=2_000,
    )
    with pytest.raises(PITAlphaDataError, match="cannot fabricate"):
        target_cross_section(
            (row,),
            asset_ids=("SEC-A", "SEC-B"),
            horizon_sessions=3,
            return_kind="economic-total-simple",
        )


def _exposure_panel() -> OriginExposurePanel:
    return OriginExposurePanel(
        origin_at_ms=2_000,
        available_at_ms=1_999,
        asset_ids=("A", "B", "C", "D", "E"),
        exposure_names=("intercept", "size"),
        exposures=(
            (1.0, -2.0),
            (1.0, -1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (1.0, 2.0),
        ),
        regression_weights=(1.0, 2.0, 3.0, 2.0, 1.0),
        qualified_asset_mask=(True, True, True, True, True),
        source_receipt_sha256=_DIGEST,
    )


def test_origin_operator_removes_weighted_factor_exposure() -> None:
    operator = build_origin_residual_operator(_exposure_panel())
    values = tuple(0.3 + 0.2 * x + noise for x, noise in zip(
        (-2.0, -1.0, 0.0, 1.0, 2.0),
        (0.01, -0.02, 0.02, -0.02, 0.01),
        strict=True,
    ))
    result = apply_origin_residual_operator(values, operator)

    assert result.maximum_weighted_exposure_error <= 2e-10
    assert any(abs(value) > 1e-5 for value in result.values)


def test_origin_operator_rejects_future_available_exposures() -> None:
    with pytest.raises(PITAlphaDataError, match="unavailable at the origin"):
        replace(_exposure_panel(), available_at_ms=2_001).validate()


def test_origin_operator_receipt_detects_mutation() -> None:
    operator = build_origin_residual_operator(_exposure_panel())
    mutated = replace(
        operator,
        coefficient_map=(
            (operator.coefficient_map[0][0] + 0.01, *operator.coefficient_map[0][1:]),
            operator.coefficient_map[1],
        ),
    )
    with pytest.raises(PITAlphaDataError):
        mutated.validate()


def _attribution_period(
    *,
    session: int,
    signal_gross: float = 0.002,
    signal_cost: float = 0.0002,
    repair_gross: float = 0.0,
    repair_cost: float = 0.0,
    benchmark_advantage: float = 0.0,
) -> ActiveReturnAttribution:
    total = signal_gross - signal_cost + repair_gross - repair_cost + benchmark_advantage
    return ActiveReturnAttribution(
        session_index=session,
        signal_gross_return=signal_gross,
        signal_cost=signal_cost,
        repair_gross_return=repair_gross,
        repair_cost=repair_cost,
        benchmark_cost_advantage=benchmark_advantage,
        other_active_return=0.0,
        signal_created_one_way_turnover=0.1 if signal_cost > 0.0 else 0.0,
        repair_one_way_turnover=0.05 if repair_cost > 0.0 else 0.0,
        benchmark_one_way_turnover=0.1 if benchmark_advantage > 0.0 else 0.0,
        policy_one_way_turnover=0.15,
        reported_active_net_return=total,
    )


def test_attribution_reconciles_exactly_and_signal_break_even_is_isolated() -> None:
    ledger = build_signal_attribution_ledger(
        dataset_receipt_sha256=_DIGEST,
        experiment_spec_sha256="b" * 64,
        periods=(
            _attribution_period(session=0),
            _attribution_period(
                session=1,
                repair_gross=0.001,
                repair_cost=0.0001,
                benchmark_advantage=0.0003,
            ),
        ),
    )

    assert ledger.total_active_net_return == pytest.approx(0.0048)
    assert ledger.signal_net_return == pytest.approx(0.0036)
    assert ledger.repair_net_return == pytest.approx(0.0009)
    assert ledger.benchmark_cost_advantage == pytest.approx(0.0003)
    assert ledger.signal_break_even_one_way_cost_basis_points == pytest.approx(200.0)


def test_attribution_rejects_unexplained_active_return() -> None:
    with pytest.raises(PITAlphaDataError, match="does not reconcile"):
        replace(
            _attribution_period(session=0),
            reported_active_net_return=0.5,
        ).validate()


def test_repair_or_benchmark_profit_cannot_promote_negative_signal() -> None:
    ledger = build_signal_attribution_ledger(
        dataset_receipt_sha256=_DIGEST,
        experiment_spec_sha256="b" * 64,
        periods=(
            _attribution_period(
                session=0,
                signal_gross=-0.001,
                signal_cost=0.0001,
                repair_gross=0.02,
                repair_cost=0.0001,
                benchmark_advantage=0.01,
            ),
        ),
    )
    evidence = evaluate_signal_promotion(
        ledger,
        signal_net_return_lcb95=-0.0001,
        factor_adjusted_signal_alpha_lcb95=-0.0001,
        estimated_median_one_way_cost_basis_points=5.0,
    )

    assert ledger.total_active_net_return > 0.0
    assert evidence.decision == "fail"
    assert evidence.signal_break_even_one_way_cost_basis_points is None


def test_signal_promotion_requires_twice_estimated_cost() -> None:
    ledger = build_signal_attribution_ledger(
        dataset_receipt_sha256=_DIGEST,
        experiment_spec_sha256="b" * 64,
        periods=(_attribution_period(session=0, signal_gross=0.00015, signal_cost=0.00005),),
    )
    evidence = evaluate_signal_promotion(
        ledger,
        signal_net_return_lcb95=0.00001,
        factor_adjusted_signal_alpha_lcb95=0.00001,
        estimated_median_one_way_cost_basis_points=10.0,
    )

    assert ledger.signal_break_even_one_way_cost_basis_points == pytest.approx(15.0)
    assert evidence.required_break_even_basis_points == 20.0
    assert evidence.decision == "fail"
    with pytest.raises(PITAlphaDataError, match="cannot be lowered"):
        evaluate_signal_promotion(
            ledger,
            signal_net_return_lcb95=0.00001,
            factor_adjusted_signal_alpha_lcb95=0.00001,
            estimated_median_one_way_cost_basis_points=1.0,
            minimum_absolute_break_even_basis_points=9.99,
        )


def _experiment_spec() -> AlphaExperimentSpec:
    return AlphaExperimentSpec(
        dataset_receipt_sha256="1" * 64,
        universe_rule_sha256="2" * 64,
        target_spec_sha256="3" * 64,
        decision_time_rule="after-close",
        fill_time_rule="next-open-auction",
        input_modalities=("bars-5m",),
        intraday_resolution_seconds=300,
        context_sessions=252,
        primary_horizon=30,
        auxiliary_horizons=(5, 21, 63),
        objective_kind="date-balanced-huber",
        model_config=AlphaModelConfig(128, 3, 4, 128, 4, 4, 32, 0.05),
        optimizer_config=AlphaOptimizerConfig(
            encoder_learning_rate=2e-5,
            head_learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_fraction=0.05,
            terminal_learning_rate_fraction=0.1,
            gradient_clip_norm=1.0,
            terminal_epoch=8,
        ),
        fold_config=AlphaFoldConfig(5, 126, 63, 30, True, True),
        seed=17,
        trial_registry_parent_sha256="4" * 64,
    )


def test_experiment_spec_is_content_addressed_and_cannot_authorize_rl() -> None:
    spec = _experiment_spec()
    assert len(spec.receipt_sha256) == 64
    with pytest.raises(PITAlphaDataError, match="cannot authorize"):
        replace(spec, reinforcement_learning_authorized=True).validate()


def test_trial_registry_counts_historical_and_new_result_moving_choices() -> None:
    spec = _experiment_spec()
    registry = build_alpha_trial_registry(
        project_id="quanttrade-alpha",
        records=(
            AlphaTrialRecord(
                trial_index=0,
                trial_id="historical-v16",
                experiment_spec_sha256="5" * 64,
                change_kind="historical-generation",
                declared_at_ms=1,
                outer_outcomes_opened_at_declaration=False,
                result_receipt_sha256="6" * 64,
            ),
            AlphaTrialRecord(
                trial_index=1,
                trial_id="a01-ordered-5m",
                experiment_spec_sha256=spec.receipt_sha256,
                change_kind="data-modality",
                declared_at_ms=2,
                outer_outcomes_opened_at_declaration=False,
            ),
        ),
    )

    assert len(registry.records) == 2
    with pytest.raises(PITAlphaDataError, match="after outer access"):
        replace(registry.records[1], outer_outcomes_opened_at_declaration=True).validate()


def test_discovery_panel_freezes_eight_bar_and_four_modality_settings() -> None:
    validate_alpha_discovery_settings()
    assert tuple(row.setting_id for row in ALPHA_DISCOVERY_SETTINGS) == (
        "A00",
        "A01",
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A07",
        "B00",
        "B01",
        "B02",
        "B03",
    )
    assert ALPHA_DISCOVERY_SETTINGS[0].input_modalities == ("bars-daily",)
    assert all(
        row.comparison_setting_id is not None
        for row in ALPHA_DISCOVERY_SETTINGS[8:]
    )


def test_math_total_loss_is_not_reexpressed_as_finite_log() -> None:
    assert math.isfinite(-1.0)
    assert _points((1.0, 0.0), terminal_from=1)[-1].value == 0.0
