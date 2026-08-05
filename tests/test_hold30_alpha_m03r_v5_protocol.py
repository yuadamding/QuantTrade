"""Identity and causal-contract tests for the immutable M03R v5 generation."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_PROTOCOL_GENERATION as M03R_V4_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SETTING_IDS,
    M03R_SETTINGS_BY_ID,
    M03R_SUPERSEDED_PROTOCOL_GENERATION,
    M03R_V5_PROTOCOL_GENERATION,
    M03R_V5_SCHEMA_VERSION,
    M03RProtocolError,
    m03r_design_payload,
)
from rl_quant.training.designs import DESIGNS, HOLD30_M03R_V5_BASE_DESIGN


def test_v5_is_disjoint_from_the_frozen_v4_generation() -> None:
    assert M03R_V4_PROTOCOL_GENERATION == "prelockbox-hold30-active-alpha-m03r-v4"
    assert M03R_PROTOCOL_GENERATION == "prelockbox-hold30-active-alpha-m03r-v5"
    assert M03R_V5_PROTOCOL_GENERATION == M03R_PROTOCOL_GENERATION
    assert M03R_SUPERSEDED_PROTOCOL_GENERATION == M03R_V4_PROTOCOL_GENERATION
    assert M03R_V5_SCHEMA_VERSION == 5
    assert M03R_DESIGN_ID == "daily_raw_pit300_hold30_m03r_v5"
    assert HOLD30_M03R_V5_BASE_DESIGN == M03R_DESIGN_ID
    assert DESIGNS[M03R_DESIGN_ID].note.startswith("M03R v5")
    payload = m03r_design_payload()
    assert payload["schema_version"] == 5
    assert payload["supersedes_protocol_generation"] == M03R_V4_PROTOCOL_GENERATION
    assert payload["v4_artifacts_retain_their_original_identity"] is True


def test_v5_names_the_implemented_temporal_information_contract_accurately() -> None:
    temporal = asdict(M03R_DESIGN.temporal)
    assert temporal["fast_raw_context_trading_sessions"] == 42
    assert temporal["learned_temporal_context_trading_sessions"] == 252
    assert "slow_raw_context_trading_sessions" not in temporal

    model = M03R_DESIGN.model
    assert model.trainable_raw_fast_branch_required
    assert model.trainable_raw_fast_branch_trading_sessions == 42
    assert model.learned_temporal_context_required
    assert model.learned_temporal_context_trading_sessions == 252
    assert not model.trainable_raw_slow_branch_required


def test_v5_projection_contract_states_the_actual_two_stage_operation() -> None:
    projection = M03R_DESIGN.factor_sector_projection
    assert projection.projection_objective == (
        "linear-minimum-l2-then-benchmark-radial-tracking-error-scaling"
    )
    assert projection.linear_projection_objective == (
        "minimum-l2-distance-over-affine-box-and-linear-exposure-sets"
    )
    assert projection.tracking_error_operation == (
        "benchmark-radial-scaling-after-linear-projection"
    )
    assert not projection.joint_covariance_ellipsoid_minimum_l2_claimed
    execution = M03R_DESIGN.ensemble_execution
    assert execution.post_ensemble_projection_application_count == 2
    assert execution.hazard_anchor_projection_application_count == 1
    assert execution.replacement_proposal_projection_application_count == 1


def test_a04_is_exactly_the_downside_score_ablation() -> None:
    assert "A04-no-downside-score-adjustment" in M03R_SETTING_IDS
    assert "A04-no-uncertainty-scaling" not in M03R_SETTING_IDS
    canonical = M03R_SETTINGS_BY_ID[M03R_CANONICAL_SETTING_ID]
    a04 = M03R_SETTINGS_BY_ID["A04-no-downside-score-adjustment"]
    causal_fields = (
        "objective_mode",
        "age_aware_holding",
        "residual_alpha_heads",
        "use_downside_adjusted_stock_score",
        "use_confidence_scaled_active_risk_budget",
        "annual_tracking_error_floor",
        "annual_tracking_error_ceiling",
        "confidence_preferred_tracking_error",
        "active_beta_neutrality",
        "factor_sector_projection",
        "exit_hazard_mode",
        "slow_context_trading_sessions",
        "sharpe_mode",
    )
    assert [
        field
        for field in causal_fields
        if getattr(a04, field) != getattr(canonical, field)
    ] == ["use_downside_adjusted_stock_score"]
    assert a04.use_confidence_scaled_active_risk_budget


def test_m01_is_formally_a_gradient_null_governance_control() -> None:
    m00 = M03R_SETTINGS_BY_ID["M00-absolute-return"]
    m01 = M03R_SETTINGS_BY_ID["M01-benchmark-subtraction"]
    assert m01.gradient_null_control_of == m00.setting_id
    assert m00.gradient_null_control_of is None
    assert "identical training gradients" in m01.description


def test_confidence_has_one_channel_and_derisk_is_separate() -> None:
    risk = M03R_DESIGN.active_risk
    assert risk.calibrated_confidence_controls_new_active_risk_only
    assert risk.learned_hazard_exits_are_confidence_independent
    assert risk.confidence_budget_applies_to_replacement_entry_only
    assert not risk.calibrated_confidence_scales_entry_scores
    assert not risk.zero_confidence_forces_benchmark_derisk
    assert risk.benchmark_derisk_request_is_separate
    assert risk.canonical_benchmark_derisk_request == pytest.approx(0.0)
    assert risk.maximum_confidence_incremental_one_way_turnover == pytest.approx(1.0)
    assert risk.maximum_incremental_one_way_turnover(0.25) == pytest.approx(0.25)
    with pytest.raises(M03RProtocolError, match="active-risk values drifted"):
        replace(risk, maximum_confidence_incremental_one_way_turnover=0.5)
