from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

import rl_quant.rl.massive_adaptive_rl_action_v1 as action_module
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    compile_massive_adaptive_portfolio_v1,
)
from rl_quant.execution.massive_adaptive_rl_compiler_control_v1 import (
    apply_massive_adaptive_rl_action_v1,
    compile_massive_adaptive_rl_control_v1,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    assert_adaptive_import_firewall,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    MassiveAdaptiveRLActionV1,
    MassiveAdaptiveRLActionV1Error,
    build_massive_adaptive_rl_action_v1,
    neutral_massive_adaptive_rl_action_v1,
)
from test_massive_adaptive_portfolio_compiler_v1 import _config, _inputs


def test_zero_action_is_exact_deterministic_compiler_route() -> None:
    inputs = _inputs(
        expected_first_bucket=(0.03, 0.01, -0.02),
        pretrade=(0.2, 0.3, 0.1),
        benchmark=(0.2, 0.3, 0.1),
    )
    config = _config(
        uncertainty_standard_deviations=1.0,
        risk_aversion=1.0,
        maximum_daily_one_way_turnover=0.25,
    )
    action = neutral_massive_adaptive_rl_action_v1()
    adjusted_inputs, adjusted_config, application = (
        apply_massive_adaptive_rl_action_v1(
            inputs=inputs,
            config=config,
            action=action,
        )
    )
    baseline = compile_massive_adaptive_portfolio_v1(inputs, config=config)
    controlled_application, controlled = compile_massive_adaptive_rl_control_v1(
        inputs=inputs,
        config=config,
        action=action,
    )

    assert action.is_neutral
    assert adjusted_inputs is inputs
    assert adjusted_config is config
    assert application.neutral_equivalence
    assert application.base_input_receipt_sha256 == (
        application.adjusted_input_receipt_sha256
    )
    assert application.base_config_receipt_sha256 == (
        application.adjusted_config_receipt_sha256
    )
    assert controlled_application == application
    assert controlled == baseline
    assert controlled.input_receipt_sha256 == baseline.input_receipt_sha256
    assert controlled.config_receipt_sha256 == baseline.config_receipt_sha256
    assert controlled.semantic_receipt_sha256 == baseline.semantic_receipt_sha256
    assert not application.reinforcement_learning_authorized


def test_nonzero_action_is_bounded_and_cannot_loosen_hard_constraints() -> None:
    inputs = _inputs(
        expected_first_bucket=(0.04, 0.02),
        pretrade=(0.5, 0.2),
        benchmark=(0.5, 0.2),
    )
    config = _config(
        maximum_security_weight=0.8,
        maximum_issuer_weight=0.8,
        tracking_error_limit_annualized=0.4,
        absolute_active_beta_limit=0.3,
        maximum_daily_one_way_turnover=0.4,
        maximum_adv_participation=0.2,
        uncertainty_standard_deviations=1.0,
        risk_aversion=1.0,
    )
    action = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        uncertainty_control=0.5,
        risk_control=-0.5,
        trade_cost_control=1.0,
    )
    adjusted_inputs, adjusted_config, application = (
        apply_massive_adaptive_rl_action_v1(
            inputs=inputs,
            config=config,
            action=action,
        )
    )

    assert not action.is_neutral
    assert application.bucket_multipliers == (1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert adjusted_inputs.bucket_expected_residual_returns[0][0] == pytest.approx(
        inputs.bucket_expected_residual_returns[0][0] * 1.25
    )
    assert adjusted_config.maximum_daily_one_way_turnover == pytest.approx(0.4)
    assert adjusted_inputs.entry_cost_basis_points == pytest.approx(
        tuple(2.0 * value for value in inputs.entry_cost_basis_points)
    )
    assert adjusted_inputs.current_exit_cost_basis_points == pytest.approx(
        tuple(2.0 * value for value in inputs.current_exit_cost_basis_points)
    )
    assert application.trade_cost_multiplier == pytest.approx(2.0)
    assert adjusted_config.maximum_security_weight == config.maximum_security_weight
    assert adjusted_config.maximum_issuer_weight == config.maximum_issuer_weight
    assert adjusted_config.tracking_error_limit_annualized == (
        config.tracking_error_limit_annualized
    )
    assert adjusted_config.absolute_active_beta_limit == (
        config.absolute_active_beta_limit
    )
    assert adjusted_config.maximum_adv_participation == (
        config.maximum_adv_participation
    )
    assert application.hard_constraints_unchanged
    assert not application.compiler_control_authorized
    assert not application.profitability_reporting_authorized
    assert not application.outer_evaluation_authorized
    assert not application.lockbox_access_authorized
    assert not application.reinforcement_learning_authorized


def test_action_range_authority_and_no_duration_firewall_fail_closed() -> None:
    with pytest.raises(MassiveAdaptiveRLActionV1Error, match=r"\[-1, 1\]"):
        build_massive_adaptive_rl_action_v1(
            bucket_controls=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            uncertainty_control=0.0,
            risk_control=0.0,
            trade_cost_control=0.0,
        )
    with pytest.raises(MassiveAdaptiveRLActionV1Error, match=r"\[-1, 1\]"):
        build_massive_adaptive_rl_action_v1(
            bucket_controls=(0.0,) * 7,
            uncertainty_control=0.0,
            risk_control=0.0,
            trade_cost_control=-1.5,
        )
    neutral = neutral_massive_adaptive_rl_action_v1()
    corrupted = replace(
        neutral,
        reinforcement_learning_authorized=True,
        semantic_receipt_sha256="0" * 64,
    )
    with pytest.raises(MassiveAdaptiveRLActionV1Error):
        corrupted.validate()

    assert_no_adaptive_hold_semantics(neutral)
    assert_adaptive_import_firewall([Path(action_module.__file__)])
    forbidden_fragments = ("age", "duration", "persistence", "scheduled_exit")
    assert all(
        not any(fragment in field.name for fragment in forbidden_fragments)
        for field in fields(MassiveAdaptiveRLActionV1)
    )


def test_turnover_control_allows_more_and_less_aggressive_trading() -> None:
    inputs = _inputs(
        expected_first_bucket=(0.04, 0.02),
        pretrade=(0.5, 0.2),
        benchmark=(0.5, 0.2),
    )
    config = _config(maximum_daily_one_way_turnover=0.4)
    low_hurdle = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=0.0,
        trade_cost_control=-1.0,
    )
    high_hurdle = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=0.0,
        trade_cost_control=1.0,
    )

    low_inputs, low_config, low_control = apply_massive_adaptive_rl_action_v1(
        inputs=inputs,
        config=config,
        action=low_hurdle,
    )
    high_inputs, high_config, high_control = apply_massive_adaptive_rl_action_v1(
        inputs=inputs,
        config=config,
        action=high_hurdle,
    )

    assert low_control.trade_cost_multiplier == pytest.approx(0.5)
    assert high_control.trade_cost_multiplier == pytest.approx(2.0)
    assert low_inputs.entry_cost_basis_points == pytest.approx(
        tuple(0.5 * value for value in inputs.entry_cost_basis_points)
    )
    assert high_inputs.entry_cost_basis_points == pytest.approx(
        tuple(2.0 * value for value in inputs.entry_cost_basis_points)
    )
    assert low_inputs.expected_future_exit_cost_basis_points[0] == pytest.approx(
        tuple(
            0.5 * value
            for value in inputs.expected_future_exit_cost_basis_points[0]
        )
    )
    assert high_inputs.expected_future_exit_cost_basis_points[0] == pytest.approx(
        tuple(
            2.0 * value
            for value in inputs.expected_future_exit_cost_basis_points[0]
        )
    )
    assert low_config.maximum_daily_one_way_turnover == (
        config.maximum_daily_one_way_turnover
    )
    assert high_config.maximum_daily_one_way_turnover == (
        config.maximum_daily_one_way_turnover
    )
    assert low_control.hard_constraints_unchanged
    assert high_control.hard_constraints_unchanged
