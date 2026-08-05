"""Frozen identity and causal-inventory tests for M03R."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SETTING_IDS,
    M03R_SETTINGS,
    M03R_SETTINGS_BY_ID,
    M03R_SUPERSEDED_PROTOCOL_GENERATION,
    M03RActiveRiskContract,
    M03RProtocolError,
    m03r_design_payload,
    resolve_m03r_setting,
    validate_m03r_artifact_identity,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_V3_CANONICAL_ID,
    HOLD30_ALPHA_V3_IDS,
    HOLD30_ALPHA_V3_PROTOCOL_GENERATION,
)

EXPECTED_M03R_IDS = (
    "M00-absolute-return",
    "M01-benchmark-subtraction",
    "M02-active-risk-no-alpha-heads",
    "M03R-active-alpha-hold30",
    "A04-no-uncertainty-scaling",
    "A05-fixed-te-floor",
    "A06-sharpe-overlay",
    "A07-direct-sharpe",
    "A08-fixed-exit-hazard",
    "A09-no-long-context",
    "A10-no-factor-neutral-projection",
)


def _sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m03r_is_new_and_v3_identity_remains_unchanged() -> None:
    assert M03R_PROTOCOL_GENERATION == "prelockbox-hold30-active-alpha-m03r-v4"
    assert M03R_SUPERSEDED_PROTOCOL_GENERATION == ("prelockbox-hold30-alpha-mech8-v3")
    assert M03R_DESIGN_ID == "daily_raw_pit300_hold30_m03r"
    assert M03R_CANONICAL_SETTING_ID == "M03R-active-alpha-hold30"

    # These snapshots prevent the new generation from relabeling V3 history.
    assert HOLD30_ALPHA_V3_PROTOCOL_GENERATION == "prelockbox-hold30-alpha-mech8-v3"
    assert HOLD30_ALPHA_V3_CANONICAL_ID == "hold30a-m03-alpha-core"
    assert HOLD30_ALPHA_V3_IDS == (
        "hold30a-m00-legacy-absolute",
        "hold30a-m01-persistent-absolute",
        "hold30a-m02-active-te",
        "hold30a-m03-alpha-core",
        "hold30a-a04-no-uncertainty",
        "hold30a-a05-no-te-floor",
        "hold30a-a06-sharpe-overlay",
        "hold30a-a07-direct-sharpe",
    )


def test_temporal_contract_names_disjoint_economic_and_context_horizons() -> None:
    temporal = M03R_DESIGN.temporal
    assert asdict(temporal) == {
        "decisions_per_trading_session": 1,
        "fast_raw_context_trading_sessions": 42,
        "slow_raw_context_trading_sessions": 252,
        "rollout_trading_sessions": 63,
        "economic_origin_post_fill_return_count": 30,
        "maximum_auxiliary_label_horizon_trading_sessions": 63,
        "target_holding_trading_sessions": 30,
        "evaluation_warmup_trading_sessions": 63,
        "evaluation_score_trading_sessions": 63,
    }
    assert temporal.economic_origin_state_row_count == 31
    assert "credit_span" not in asdict(temporal)

    with pytest.raises(M03RProtocolError, match="cannot be substituted"):
        replace(temporal, economic_origin_post_fill_return_count=63)


def test_canonical_has_no_te_floor_and_confidence_scaled_preference() -> None:
    risk = M03R_DESIGN.active_risk
    assert risk.annual_tracking_error_floor == 0.0
    assert risk.annual_tracking_error_ceiling == pytest.approx(0.06)
    assert risk.confidence_preferred_annual_tracking_error_minimum == 0.0
    assert risk.confidence_preferred_annual_tracking_error_maximum == pytest.approx(
        0.04
    )
    assert risk.preferred_annual_tracking_error(0.0) == 0.0
    assert risk.preferred_annual_tracking_error(0.5) == pytest.approx(0.02)
    assert risk.preferred_annual_tracking_error(1.0) == pytest.approx(0.04)
    assert risk.active_market_beta_target == 0.0
    assert risk.absolute_active_market_beta_maximum == pytest.approx(0.10)
    assert risk.total_portfolio_market_beta_is_secondary_diagnostic

    for value in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(M03RProtocolError, match="calibrated_confidence"):
            risk.preferred_annual_tracking_error(value)

    with pytest.raises(M03RProtocolError, match="active-risk values drifted"):
        M03RActiveRiskContract(annual_tracking_error_floor=0.02)


def test_factor_sector_projection_is_explicit_and_actor_blind() -> None:
    projection = M03R_DESIGN.factor_sector_projection
    assert projection.requested_quantity == "active-weight-delta-versus-C1"
    assert projection.projection_objective == (
        "minimum-l2-distance-to-requested-active-weights"
    )
    assert projection.exposure_families == (
        "market",
        "sector",
        "size",
        "momentum",
        "value",
        "volatility",
        "liquidity",
    )
    assert projection.exposure_loadings_are_point_in_time
    assert not projection.exposure_loadings_actor_feature_access
    assert projection.active_weight_sum_target == 0.0
    assert projection.long_only_post_projection
    assert projection.projection_applied_after_seed_ensemble
    assert projection.numerical_exposure_bounds_manifest_sha256_required
    assert projection.infeasible_projection_behavior == "fail-closed-no-artifact"

    payload = m03r_design_payload()
    required = set(payload["required_launch_bindings"])
    assert "point_in_time_factor_manifest_sha256" in required
    assert "point_in_time_sector_manifest_sha256" in required
    assert "factor_sector_exposure_bounds_manifest_sha256" in required


def test_canonical_model_requires_a_true_exact_hold_branch() -> None:
    assert M03R_DESIGN.model.exact_hold_branch_required is True
    assert M03R_DESIGN.model.confidence_calibration_manifest_sha256_required is True


def test_seed_ensemble_and_projection_are_frozen_once_after_aggregation() -> None:
    execution = M03R_DESIGN.ensemble_execution
    assert execution.ensemble_member_count == 5
    assert execution.aggregation_order == "ascending-integer-seed"
    assert execution.post_ensemble_projection_application_count == 1
    assert execution.risk_manifest_schema == (
        "rl-quant.m03r-factor-sector-risk-manifest-v1"
    )
    assert execution.projection_solver == "deterministic-dykstra-euclidean"
    assert execution.projection_tolerance == pytest.approx(1e-10)
    assert execution.projection_maximum_iterations == 4_000


def test_inventory_freezes_candidate_and_one_change_ablation_family() -> None:
    assert M03R_SETTING_IDS == EXPECTED_M03R_IDS
    assert tuple(M03R_SETTINGS_BY_ID) == EXPECTED_M03R_IDS
    assert tuple(setting.setting_index for setting in M03R_SETTINGS) == tuple(range(11))
    assert [row.setting_id for row in M03R_SETTINGS if row.promotion_eligible] == [
        M03R_CANONICAL_SETTING_ID
    ]

    canonical = M03R_SETTINGS_BY_ID[M03R_CANONICAL_SETTING_ID]
    assert canonical.annual_tracking_error_floor == 0.0
    assert canonical.annual_tracking_error_ceiling == pytest.approx(0.06)
    assert canonical.active_beta_neutrality
    assert canonical.factor_sector_projection
    assert canonical.exit_hazard_mode == "learned-age-aware"
    assert canonical.slow_context_trading_sessions == 252
    assert canonical.sharpe_mode == "none"

    causal_fields = (
        "objective_mode",
        "age_aware_holding",
        "residual_alpha_heads",
        "uncertainty_scaled_sizing",
        "annual_tracking_error_floor",
        "annual_tracking_error_ceiling",
        "confidence_preferred_tracking_error",
        "active_beta_neutrality",
        "factor_sector_projection",
        "exit_hazard_mode",
        "slow_context_trading_sessions",
        "sharpe_mode",
    )
    expected = {
        "A04-no-uncertainty-scaling": "uncertainty_scaled_sizing",
        "A05-fixed-te-floor": "annual_tracking_error_floor",
        "A06-sharpe-overlay": "sharpe_mode",
        "A07-direct-sharpe": "sharpe_mode",
        "A08-fixed-exit-hazard": "exit_hazard_mode",
        "A09-no-long-context": "slow_context_trading_sessions",
        "A10-no-factor-neutral-projection": "factor_sector_projection",
    }
    for setting_id, expected_field in expected.items():
        setting = M03R_SETTINGS_BY_ID[setting_id]
        assert setting.ablation_of == M03R_CANONICAL_SETTING_ID
        assert [
            field
            for field in causal_fields
            if getattr(setting, field) != getattr(canonical, field)
        ] == [expected_field]


def test_m03r_identity_resolution_fails_closed_for_v3() -> None:
    assert resolve_m03r_setting(M03R_CANONICAL_SETTING_ID).promotion_eligible
    assert (
        validate_m03r_artifact_identity(
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
        ).setting_id
        == M03R_CANONICAL_SETTING_ID
    )

    with pytest.raises(M03RProtocolError, match="immutable audit generation"):
        validate_m03r_artifact_identity(
            protocol_generation=HOLD30_ALPHA_V3_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
        )
    with pytest.raises(M03RProtocolError, match="V3 setting"):
        resolve_m03r_setting(HOLD30_ALPHA_V3_CANONICAL_ID)
    with pytest.raises(M03RProtocolError, match="design_id"):
        validate_m03r_artifact_identity(
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id="daily_raw_pit300_hold30_alpha_v3",
            setting_id=M03R_CANONICAL_SETTING_ID,
        )


def test_payload_is_deterministic_content_and_dataclasses_are_frozen() -> None:
    first = m03r_design_payload()
    second = m03r_design_payload()
    assert first == second
    assert _sha256(first) == _sha256(second)
    assert first["schema_version"] == 4
    assert first["v3_artifacts_retain_their_original_identity"] is True

    mutated = dict(first)
    mutated["canonical_setting_id"] = "not-M03R"
    assert _sha256(mutated) != _sha256(first)

    with pytest.raises(FrozenInstanceError):
        M03R_DESIGN.temporal.rollout_trading_sessions = 30  # type: ignore[misc]
