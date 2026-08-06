"""Focused deterministic tests for the immutable M03R v7 panel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_PROTOCOL_GENERATION as M03R_V6_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_SETTING_IDS as M03R_V6_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_A05_RESERVE_SETTING_ID,
    M03R_V7_AUXILIARY_CONTROLS_SHA256,
    M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE,
    M03R_V7_CANONICAL_SETTING_ID,
    M03R_V7_CAUSAL_FIELDS,
    M03R_V7_DESIGN_ID,
    M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE,
    M03R_V7_FIXED_TE_FLOOR_RESERVE,
    M03R_V7_GRADIENT_NULL_QUALIFICATION,
    M03R_V7_M00_QUALIFICATION_SETTING_ID,
    M03R_V7_M01_QUALIFICATION_SETTING_ID,
    M03R_V7_PERSISTENCE_NORMALIZATION,
    M03R_V7_PRIMARY_PANEL_SHA256,
    M03R_V7_PRIMARY_SETTING_IDS,
    M03R_V7_PRIMARY_SETTINGS,
    M03R_V7_PRIMARY_SETTINGS_BY_ID,
    M03R_V7_PROTOCOL_GENERATION,
    M03R_V7_SHARED_CONFIGURATION,
    M03R_V7_SHARED_CONFIGURATION_SHA256,
    M03R_V7_SUPERSEDED_PROTOCOL_GENERATION,
    M03RV7ProtocolError,
    m03r_v7_protocol_payload,
    resolve_m03r_v7_setting,
    validate_m03r_v7_artifact_identity,
)


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_v7_identity_is_new_and_preserves_v6() -> None:
    assert M03R_V7_PROTOCOL_GENERATION == ("prelockbox-hold30-active-alpha-m03r-v7")
    assert M03R_V7_DESIGN_ID == "daily_raw_pit300_hold30_m03r_v7"
    assert M03R_V7_SUPERSEDED_PROTOCOL_GENERATION == M03R_V6_PROTOCOL_GENERATION
    assert set(M03R_V7_PRIMARY_SETTING_IDS).isdisjoint(M03R_V6_SETTING_IDS)

    with pytest.raises(M03RV7ProtocolError, match="v6 remains immutable"):
        validate_m03r_v7_artifact_identity(
            protocol_generation=M03R_V6_PROTOCOL_GENERATION,
            design_id=M03R_V7_DESIGN_ID,
            setting_id=M03R_V7_CANONICAL_SETTING_ID,
        )


def test_primary_inventory_has_exact_user_order_and_one_candidate() -> None:
    assert M03R_V7_PRIMARY_SETTING_IDS == (
        "M03R-soft-persistence-active-alpha-hold30-v7",
        "P00-no-soft-persistence-v7",
        "P10-soft-persistence-10bp-v7",
        "A08-fixed-exit-hazard-v7",
        "A11-no-exact-hold-atom-v7",
        "A09-no-long-context-v7",
        "M02-active-risk-no-alpha-heads-v7",
        "A04-no-downside-score-adjustment-v7",
        "A12-fixed-2pct-active-risk-budget-v7",
        "A10-no-factor-neutral-projection-v7",
        "A06-sharpe-overlay-v7",
        "A07-direct-sharpe-v7",
    )
    assert tuple(row.setting_index for row in M03R_V7_PRIMARY_SETTINGS) == tuple(
        range(12)
    )
    assert [
        row.setting_id for row in M03R_V7_PRIMARY_SETTINGS if row.promotion_eligible
    ] == [M03R_V7_CANONICAL_SETTING_ID]
    assert M03R_V7_M01_QUALIFICATION_SETTING_ID not in (M03R_V7_PRIMARY_SETTINGS_BY_ID)
    assert M03R_V7_A05_RESERVE_SETTING_ID not in M03R_V7_PRIMARY_SETTINGS_BY_ID


def test_each_primary_ablation_changes_exactly_one_causal_field() -> None:
    canonical = M03R_V7_PRIMARY_SETTINGS_BY_ID[M03R_V7_CANONICAL_SETTING_ID]
    for row in M03R_V7_PRIMARY_SETTINGS[1:]:
        changed = tuple(
            field
            for field in M03R_V7_CAUSAL_FIELDS
            if getattr(row, field) != getattr(canonical, field)
        )
        assert changed == (row.declared_causal_field,)
        assert row.ablation_of == M03R_V7_CANONICAL_SETTING_ID
        assert not row.promotion_eligible


def test_shared_configuration_matches_the_frozen_scientific_contract() -> None:
    config = M03R_V7_SHARED_CONFIGURATION
    assert config.universe_id == "point-in-time-active-300"
    assert config.decisions_per_trading_session == 1
    assert config.recent_raw_context_trading_sessions == 42
    assert config.canonical_learned_temporal_context_trading_sessions == 252
    assert config.rollout_trading_sessions == 63
    assert config.economic_credit_post_fill_return_count == 30
    assert config.auxiliary_horizons_trading_sessions == (5, 21, 30, 63)
    assert config.training_one_way_cost_basis_points == 20
    assert config.validation_one_way_costs_basis_points == (10, 20, 40)
    assert config.annual_tracking_error_floor is None
    assert config.annual_tracking_error_ceiling == pytest.approx(0.06)
    assert config.active_market_beta_target == pytest.approx(0.0)
    assert config.active_beta_equivalence_absolute_upper_bound == pytest.approx(0.10)
    assert config.maximum_stock_weight_fraction == pytest.approx(0.01)
    assert config.age_state_bin_count == 61
    assert config.exact_hold_action_available_but_optional
    assert config.validation_fold_count == 6
    assert config.paired_seeds == (17, 29, 43, 71, 101)
    assert config.seed_outputs_ensembled_before_chronological_inference
    assert config.identical_setting_order_within_each_fold_seed
    assert config.identical_data_stream_across_settings
    assert config.identical_initialization_convention_across_settings
    assert config.identical_calibration_procedure_across_settings
    assert config.identical_optimizer_schedule_across_settings


def test_persistence_contract_is_nav_session_proportional_and_nonbinding() -> None:
    persistence = M03R_V7_SHARED_CONFIGURATION.persistence
    assert persistence.normalization == M03R_V7_PERSISTENCE_NORMALIZATION
    assert not persistence.denominator_uses_total_sold_notional
    assert persistence.forced_and_unavailable_exits_exempt
    assert persistence.age_ledger_bin_count == 61
    assert persistence.preference_horizon_sessions == 30
    assert persistence.minimum_holding_period_sessions is None
    assert not persistence.pre_horizon_sell_mask
    assert not persistence.forced_expiry_at_preference_horizon
    assert not persistence.holding_duration_is_promotion_gate
    assert persistence.age_weight(0.0) == pytest.approx(1.0)
    assert persistence.age_weight(15.0) == pytest.approx(0.25)
    assert persistence.age_weight(30.0) == pytest.approx(0.0)
    assert persistence.age_weight(45.0) == pytest.approx(0.0)
    assert persistence.warmup_multiplier(0.05) == pytest.approx(0.5)
    assert persistence.coefficient_as_return(0.0) == pytest.approx(0.0)
    assert persistence.coefficient_as_return(5.0) == pytest.approx(0.0005)
    assert persistence.coefficient_as_return(10.0) == pytest.approx(0.001)

    with pytest.raises(M03RV7ProtocolError, match="0, 5, or 10"):
        persistence.coefficient_as_return(2.0)
    with pytest.raises(M03RV7ProtocolError, match="sold-notional denominator"):
        replace(persistence, denominator_uses_total_sold_notional=True)
    with pytest.raises(M03RV7ProtocolError, match="soft preference"):
        replace(persistence, minimum_holding_period_sessions=30)


def test_persistence_and_exit_rows_have_the_intended_mechanisms() -> None:
    rows = M03R_V7_PRIMARY_SETTINGS_BY_ID
    assert (
        rows[M03R_V7_CANONICAL_SETTING_ID].persistence_coefficient_basis_points == 5.0
    )
    assert (
        rows["P00-no-soft-persistence-v7"].persistence_coefficient_basis_points == 0.0
    )
    assert (
        rows["P10-soft-persistence-10bp-v7"].persistence_coefficient_basis_points
        == 10.0
    )
    assert rows["A08-fixed-exit-hazard-v7"].exit_hazard_mode == (
        "fixed-structural-30-session-prior"
    )
    assert not rows["A08-fixed-exit-hazard-v7"].emits_exact_hold_action
    assert rows[M03R_V7_CANONICAL_SETTING_ID].emits_exact_hold_action
    assert not rows["A11-no-exact-hold-atom-v7"].emits_exact_hold_action


def test_alpha_context_risk_projection_and_sharpe_rows_are_exact() -> None:
    rows = M03R_V7_PRIMARY_SETTINGS_BY_ID
    assert (
        rows["A09-no-long-context-v7"].learned_temporal_context_trading_sessions == 63
    )

    no_heads = rows["M02-active-risk-no-alpha-heads-v7"]
    assert no_heads.residual_alpha_head_mode == "none"
    assert not no_heads.residual_alpha_heads
    assert not no_heads.downside_score_adjustment

    mean_only = rows["A04-no-downside-score-adjustment-v7"]
    assert mean_only.residual_alpha_head_mode == "mean-only"
    assert mean_only.residual_alpha_heads
    assert not mean_only.downside_score_adjustment

    canonical = rows[M03R_V7_CANONICAL_SETTING_ID]
    fixed = rows["A12-fixed-2pct-active-risk-budget-v7"]
    assert canonical.active_risk_budget_mode == (
        M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE
    )
    assert fixed.active_risk_budget_mode == M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE
    assert canonical.preferred_active_risk_budget_annualized(0.25) == pytest.approx(
        0.01
    )
    assert canonical.preferred_active_risk_budget_annualized(1.0) == pytest.approx(0.04)
    assert fixed.preferred_active_risk_budget_annualized(0.0) == pytest.approx(0.02)
    assert fixed.preferred_active_risk_budget_annualized(1.0) == pytest.approx(0.02)

    assert not rows[
        "A10-no-factor-neutral-projection-v7"
    ].factor_sector_neutral_projection
    assert rows["A06-sharpe-overlay-v7"].sharpe_mode == ("separate-total-risk-overlay")
    assert rows["A07-direct-sharpe-v7"].sharpe_mode == (
        "direct-full-batch-two-pass-gradient"
    )


def test_m01_is_short_qualification_and_a05_is_inactive_reserve() -> None:
    qualification = M03R_V7_GRADIENT_NULL_QUALIFICATION
    assert qualification.reference_setting_id == M03R_V7_M00_QUALIFICATION_SETTING_ID
    assert qualification.benchmark_subtraction_setting_id == (
        M03R_V7_M01_QUALIFICATION_SETTING_ID
    )
    assert qualification.reference_objective == "negative-portfolio-net-log-return"
    assert qualification.benchmark_subtraction_objective == (
        "negative-portfolio-net-log-return-plus-detached-C1-net-log-return"
    )
    assert qualification.benchmark_term_is_parameter_independent_and_detached
    assert (
        qualification.minimum_optimizer_updates,
        qualification.maximum_optimizer_updates,
    ) == (
        2,
        4,
    )
    assert qualification.compare_gradients
    assert qualification.compare_parameter_updates
    assert qualification.compare_model_state_hashes
    assert qualification.compare_optimizer_state_hashes
    assert not qualification.complete_checkpoint_or_receipt_hash_equality_required
    assert not qualification.primary_panel_member
    assert not qualification.full_fold_seed_study_authorized
    assert not qualification.promotion_eligible

    reserve = M03R_V7_FIXED_TE_FLOOR_RESERVE
    assert reserve.setting_id == M03R_V7_A05_RESERVE_SETTING_ID
    assert reserve.annual_tracking_error_floor == pytest.approx(0.02)
    assert "collapses-to-near-zero-active-risk" in reserve.activation_condition
    assert not reserve.primary_panel_member
    assert not reserve.automatically_scheduled
    assert not reserve.promotion_eligible

    with pytest.raises(M03RV7ProtocolError, match="unknown primary"):
        resolve_m03r_v7_setting(M03R_V7_M01_QUALIFICATION_SETTING_ID)
    with pytest.raises(M03RV7ProtocolError, match="unknown primary"):
        resolve_m03r_v7_setting(M03R_V7_A05_RESERVE_SETTING_ID)


def test_content_hashes_bind_shared_config_primary_rows_and_auxiliary_controls() -> (
    None
):
    assert M03R_V7_SHARED_CONFIGURATION_SHA256 == _sha256(
        asdict(M03R_V7_SHARED_CONFIGURATION)
    )
    assert M03R_V7_PRIMARY_PANEL_SHA256 == _sha256(
        {
            "protocol_generation": M03R_V7_PROTOCOL_GENERATION,
            "design_id": M03R_V7_DESIGN_ID,
            "shared_configuration_sha256": M03R_V7_SHARED_CONFIGURATION_SHA256,
            "primary_settings": [asdict(row) for row in M03R_V7_PRIMARY_SETTINGS],
        }
    )
    assert len(M03R_V7_AUXILIARY_CONTROLS_SHA256) == 64

    mutated = [asdict(row) for row in M03R_V7_PRIMARY_SETTINGS]
    mutated[1]["persistence_coefficient_basis_points"] = 10.0
    assert (
        _sha256(
            {
                "protocol_generation": M03R_V7_PROTOCOL_GENERATION,
                "design_id": M03R_V7_DESIGN_ID,
                "shared_configuration_sha256": M03R_V7_SHARED_CONFIGURATION_SHA256,
                "primary_settings": mutated,
            }
        )
        != M03R_V7_PRIMARY_PANEL_SHA256
    )


def test_payload_is_deterministic_and_launches_fail_closed() -> None:
    first = m03r_v7_protocol_payload()
    second = m03r_v7_protocol_payload()
    assert first == second
    assert first["primary_panel_sha256"] == M03R_V7_PRIMARY_PANEL_SHA256
    assert first["launch_authorized"] is False
    assert first["h100_training_authorized"] is False
    assert first["launch_blockers"]
    assert first["v6_artifacts_retain_their_original_identity"] is True

    canonical = validate_m03r_v7_artifact_identity(
        protocol_generation=M03R_V7_PROTOCOL_GENERATION,
        design_id=M03R_V7_DESIGN_ID,
        setting_id=M03R_V7_CANONICAL_SETTING_ID,
    )
    assert canonical.setting_id == M03R_V7_CANONICAL_SETTING_ID
