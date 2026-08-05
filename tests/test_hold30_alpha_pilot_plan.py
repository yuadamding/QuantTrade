from __future__ import annotations

import math
from dataclasses import fields, replace
from inspect import signature

import pytest

from rl_quant.execution.hold30 import (
    HOLD30_EXPOSURE_BAND,
    HOLD30_MAX_DISCRETIONARY_TURNOVER,
    HOLD30_MAX_STOCK_WEIGHT,
    build_alpha_hold30_action,
    build_scalar_gate_hold30_action,
    centered_benchmark_tilt,
)
from rl_quant.models.daily_policy import (
    HOLD30_AGE_CAP,
    HOLD30_HAZARD_MAX,
    HOLD30_HAZARD_MIN,
)
from rl_quant.protocol.hold30_alpha_v3 import resolve_hold30_alpha_setting
from rl_quant.training.hold30_alpha import Hold30AlphaObjectiveConfig
from rl_quant.training.hold30_alpha_pilot_plan import (
    HOLD30_ALPHA_PILOT_PROFILE,
    HOLD30_ALPHA_PILOT_PROFILE_RECEIPT_SHA256,
    Hold30AlphaPilotPlanError,
    build_hold30_alpha_pilot_training_plan,
    qualify_hold30_alpha_pilot_training_plan,
)
from rl_quant.training.hold30_alpha_plan import (
    HOLD30_ALPHA_CONFIG_SETTING_IDS,
    unresolved_hold30_alpha_training_plan,
)


def _plan():
    return build_hold30_alpha_pilot_training_plan(
        a06_optimizer_spec_receipt_sha256="a" * 64
    )


def test_pilot_factory_freezes_all_result_moving_numeric_choices() -> None:
    plan = _plan()
    profile = HOLD30_ALPHA_PILOT_PROFILE
    assert plan.scientific_decision_receipt_sha256 == profile.receipt_id
    assert profile.receipt_id == HOLD30_ALPHA_PILOT_PROFILE_RECEIPT_SHA256
    assert plan.checkpoint_contract.projection_distance_max == pytest.approx(0.01)
    assert plan.checkpoint_contract.forced_turnover_fraction_max == pytest.approx(0.10)
    assert profile.active_log_scale_bounds == pytest.approx(
        (math.log(0.5), math.log(1.5))
    )
    assert tuple(
        0.04 * math.exp(value) for value in profile.active_log_scale_bounds
    ) == pytest.approx((0.02, 0.06))
    assert profile.uncertainty_log_scale_bounds == pytest.approx(
        (math.log(0.01), 0.0)
    )
    assert profile.auxiliary_horizon_weights == (0.10, 0.20, 0.50, 0.20)
    assert profile.auxiliary_horizon_scales == pytest.approx(
        tuple(0.02 * math.sqrt(horizon) for horizon in (5, 21, 30, 63))
    )

    configs = {config.setting_id: config for config in plan.objective_configs}
    assert tuple(configs) == HOLD30_ALPHA_CONFIG_SETTING_IDS
    m03 = configs["hold30a-m03-alpha-core"]
    assert m03.lambda_te_floor == pytest.approx(0.25)
    assert m03.lambda_te_ceiling == pytest.approx(0.50)
    assert m03.lambda_beta == pytest.approx(0.01)
    assert m03.lambda_turnover == pytest.approx(0.25)
    assert m03.lambda_early_exit == pytest.approx(0.002)
    assert m03.lambda_auxiliary_alpha == pytest.approx(0.0001)
    assert m03.lambda_uncertainty == pytest.approx(0.00005)
    assert m03.downside_penalty_kappa == pytest.approx(0.25)


def test_pilot_classifies_every_optional_result_moving_objective_field() -> None:
    """A newly added optional coefficient cannot evade the pilot freeze."""

    optional_fields = {
        field.name
        for field in fields(Hold30AlphaObjectiveConfig)
        if field.default is None
    }
    configs = {config.setting_id: config for config in _plan().objective_configs}
    for setting_id, config in configs.items():
        setting = resolve_hold30_alpha_setting(setting_id)
        expected_populated = {
            "lambda_te_ceiling",
            "lambda_turnover",
            "lambda_early_exit",
        }
        if setting.te_floor_annual is not None:
            expected_populated.add("lambda_te_floor")
        if setting.beta_targeting:
            expected_populated.add("lambda_beta")
        if setting.supervised_residual_alpha_heads:
            expected_populated.update(
                {
                    "active_log_scale_bounds",
                    "auxiliary_horizon_weights",
                    "auxiliary_horizon_scales",
                    "lambda_auxiliary_alpha",
                }
            )
        if setting.uncertainty_downside_heads:
            expected_populated.update(
                {
                    "downside_penalty_kappa",
                    "uncertainty_log_scale_bounds",
                    "lambda_uncertainty",
                }
            )
        if setting.sharpe_mode == "separate-total-risk-overlay":
            expected_populated.update(
                {
                    "a06_total_risk_step",
                    "alpha_core_parameter_selector",
                    "overlay_parameter_selector",
                    "stop_gradient_core_to_overlay",
                    "stop_gradient_overlay_to_core",
                    "separate_optimizer_spec_receipt_sha256",
                    "lambda_total_excess_mean",
                    "lambda_total_sharpe_overlay",
                    "total_sharpe_epsilon",
                    "lambda_volatility_ratio",
                    "target_volatility_ratio",
                    "lambda_drawdown",
                    "drawdown_limit",
                }
            )
        if setting.sharpe_mode == "direct-two-pass-gradient":
            expected_populated.update(
                {"lambda_direct_sharpe", "direct_sharpe_epsilon"}
            )

        populated = {
            name for name in optional_fields if getattr(config, name) is not None
        }
        assert populated == expected_populated, setting_id
        config.require_resolved()


def test_pilot_inherits_exact_rfc_action_bounds_from_bound_source() -> None:
    """Action geometry is frozen by the RFC/source hashes, not pilot outcomes."""

    assert (HOLD30_HAZARD_MIN, HOLD30_HAZARD_MAX) == (-12.0, 12.0)
    assert HOLD30_AGE_CAP == 60
    assert HOLD30_MAX_STOCK_WEIGHT == pytest.approx(0.01)
    assert HOLD30_MAX_DISCRETIONARY_TURNOVER == pytest.approx(0.10)
    assert HOLD30_EXPOSURE_BAND == pytest.approx(0.02)
    assert signature(centered_benchmark_tilt).parameters[
        "score_clip"
    ].default == pytest.approx(2.0)
    assert signature(build_scalar_gate_hold30_action).parameters[
        "temperature"
    ].default == pytest.approx(0.5)
    assert signature(build_alpha_hold30_action).parameters[
        "max_turnover"
    ].default == pytest.approx(HOLD30_MAX_DISCRETIONARY_TURNOVER)


def test_registered_ablations_and_sharpe_profiles_remain_distinct() -> None:
    configs = {config.setting_id: config for config in _plan().objective_configs}
    a04 = configs["hold30a-a04-no-uncertainty"]
    assert a04.lambda_uncertainty is None
    assert a04.downside_penalty_kappa is None
    assert a04.uncertainty_log_scale_bounds is None

    a05 = configs["hold30a-a05-no-te-floor"]
    assert a05.lambda_te_floor is None
    assert a05.lambda_te_ceiling == pytest.approx(0.50)

    a06 = configs["hold30a-a06-sharpe-overlay"]
    assert a06.lambda_total_excess_mean == pytest.approx(1.0)
    assert a06.lambda_total_sharpe_overlay == pytest.approx(0.00005)
    assert a06.total_sharpe_epsilon == pytest.approx(1e-6)
    assert a06.lambda_volatility_ratio == pytest.approx(0.01)
    assert a06.target_volatility_ratio == pytest.approx(1.0)
    assert a06.lambda_drawdown == pytest.approx(0.04)
    assert a06.drawdown_limit == pytest.approx(-math.log(0.85))
    assert a06.a06_total_risk_step == pytest.approx(0.05)
    assert a06.alpha_core_parameter_selector == "alpha-core-only"
    assert a06.overlay_parameter_selector == "a06-overlay-only"
    assert a06.stop_gradient_core_to_overlay is True
    assert a06.stop_gradient_overlay_to_core is True
    assert a06.separate_optimizer_spec_receipt_sha256 == "a" * 64

    a07 = configs["hold30a-a07-direct-sharpe"]
    assert a07.lambda_direct_sharpe == pytest.approx(0.00005)
    assert a07.direct_sharpe_epsilon == pytest.approx(1e-6)
    assert a07.lambda_total_sharpe_overlay is None
    assert a06.total_sharpe_epsilon == a07.direct_sharpe_epsilon


def test_pilot_qualification_binds_prelaunch_a06_sharpe_floor() -> None:
    plan = _plan()
    for config in plan.objective_configs:
        config.require_resolved()

    qualification = qualify_hold30_alpha_pilot_training_plan(plan)
    assert qualification.numerical_profile_complete is True
    assert qualification.checkpoint_thresholds_complete is True
    assert qualification.remaining_implementation_blockers == ()
    assert qualification.executable_eight_setting_run is True
    assert qualification.promotion_authorized is False
    assert len(qualification.receipt_id) == 64
    assert qualification.scientific_decision_receipt_sha256 == (
        HOLD30_ALPHA_PILOT_PROFILE.receipt_id
    )
    assumptions = " ".join(HOLD30_ALPHA_PILOT_PROFILE.assumptions)
    assert "same 1e-6 variance floor" in assumptions
    assert "negligible at normal equity volatility" in assumptions
    assert "frozen before launch" in assumptions


def test_canonical_unresolved_plan_is_unchanged_and_pilot_tampering_fails() -> None:
    canonical = unresolved_hold30_alpha_training_plan()
    assert canonical.checkpoint_contract.projection_distance_max is None
    assert canonical.checkpoint_contract.forced_turnover_fraction_max is None
    assert canonical.scientific_decision_receipt_sha256 is None
    assert all(config.lambda_turnover is None for config in canonical.objective_configs)

    plan = _plan()
    changed = list(plan.objective_configs)
    changed[0] = replace(changed[0], lambda_turnover=0.30)
    tampered = replace(plan, objective_configs=tuple(changed))
    with pytest.raises(Hold30AlphaPilotPlanError, match="differs from the exact pilot"):
        qualify_hold30_alpha_pilot_training_plan(tampered)

    changed = list(plan.objective_configs)
    a06_index = HOLD30_ALPHA_CONFIG_SETTING_IDS.index(
        "hold30a-a06-sharpe-overlay"
    )
    changed[a06_index] = replace(changed[a06_index], total_sharpe_epsilon=2e-6)
    with pytest.raises(Hold30AlphaPilotPlanError, match="differs from the exact pilot"):
        qualify_hold30_alpha_pilot_training_plan(
            replace(plan, objective_configs=tuple(changed))
        )


def test_pilot_factory_requires_real_a06_runtime_receipt_identity() -> None:
    with pytest.raises(Hold30AlphaPilotPlanError, match="lowercase SHA-256"):
        build_hold30_alpha_pilot_training_plan(
            a06_optimizer_spec_receipt_sha256="not-a-receipt"
        )
