from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_DESIGN_ID,
    M03R_V7_PRIMARY_SETTING_IDS,
    M03R_V7_PRIMARY_SETTINGS,
    M03R_V7_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_CACHE_REQUIREMENT,
    M03R_TOP2000_DEV_DESIGN_ID,
    M03R_TOP2000_DEV_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE,
    M03R_TOP2000_DEV_GEOMETRY,
    M03R_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_TOP2000_DEV_RAW_SIGMOID_ACTIVE_RISK_BUDGET_MODE,
    M03R_TOP2000_DEV_SETTING_IDS,
    M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID,
    M03R_TOP2000_DEV_SETTINGS,
    M03RTop2000DevProtocolError,
    m03r_top2000_dev_protocol_payload,
    resolve_m03r_top2000_dev_setting,
    validate_m03r_top2000_dev_artifact_identity,
)


def test_top2000_dev_identity_is_disjoint_nonreportable_and_deterministic() -> None:
    assert "top2000-dev" in M03R_TOP2000_DEV_PROTOCOL_GENERATION
    assert M03R_TOP2000_DEV_DESIGN_ID.startswith("daily_ohlcv_aggregated_")
    assert M03R_TOP2000_DEV_PROTOCOL_GENERATION != M03R_V7_PROTOCOL_GENERATION
    assert M03R_TOP2000_DEV_DESIGN_ID != M03R_V7_DESIGN_ID
    assert len(M03R_TOP2000_DEV_SETTING_IDS) == 12
    assert set(M03R_TOP2000_DEV_SETTING_IDS).isdisjoint(M03R_V7_PRIMARY_SETTING_IDS)
    assert all("top2000-dev" in row for row in M03R_TOP2000_DEV_SETTING_IDS)

    payload = m03r_top2000_dev_protocol_payload()
    assert payload == m03r_top2000_dev_protocol_payload()
    assert len(M03R_TOP2000_DEV_PROTOCOL_SHA256) == 64
    assert payload["development_only"] is True
    assert payload["training_authorized"] is False
    assert payload["promotion_authorized"] is False
    assert payload["scientific_reporting_authorized"] is False
    assert payload["outer_lockbox_evaluation_authorized"] is False
    assert all(not row.promotion_eligible for row in M03R_TOP2000_DEV_SETTINGS)
    assert all(
        not row.scientific_reporting_eligible for row in M03R_TOP2000_DEV_SETTINGS
    )


def test_top2000_dev_preserves_reviewed_v7_order_and_semantics() -> None:
    assert tuple(M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID.values()) == (
        M03R_V7_PRIMARY_SETTING_IDS
    )
    for development, reviewed in zip(
        M03R_TOP2000_DEV_SETTINGS,
        M03R_V7_PRIMARY_SETTINGS,
        strict=True,
    ):
        assert development.setting_index == reviewed.setting_index
        assert development.reviewed_v7_setting_id == reviewed.setting_id
        assert development.persistence_coefficient_basis_points == (
            reviewed.persistence_coefficient_basis_points
        )
        assert development.exit_hazard_mode == reviewed.exit_hazard_mode
        assert development.exact_hold_action_supported is (
            reviewed.exact_hold_action_supported
        )
        assert development.learned_temporal_context_trading_sessions == (
            reviewed.learned_temporal_context_trading_sessions
        )
        assert development.residual_alpha_head_mode == reviewed.residual_alpha_head_mode
        assert development.active_risk_budget_mode == reviewed.active_risk_budget_mode
        assert development.development_active_risk_budget_execution_mode == (
            M03R_TOP2000_DEV_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE
            if development.setting_index == 8
            else M03R_TOP2000_DEV_RAW_SIGMOID_ACTIVE_RISK_BUDGET_MODE
        )
        assert development.factor_sector_neutral_projection is (
            reviewed.factor_sector_neutral_projection
        )
        assert development.sharpe_mode == reviewed.sharpe_mode
        assert len(development.reviewed_semantics_sha256) == 64


def test_top2000_dev_geometry_and_cache_contract_are_explicit() -> None:
    assert asdict(M03R_TOP2000_DEV_GEOMETRY) == {
        "input_observation_contract": "daily-ohlcv-aggregated-from-300s-source",
        "daily_ohlcv_feature_count": 5,
        "recent_daily_ohlcv_context_trading_sessions": 42,
        "canonical_learned_temporal_context_trading_sessions": 252,
        "lightweight_daily_ohlcv_token_for_all_context_sessions": True,
        "rollout_trading_sessions": 63,
        "economic_credit_post_fill_return_count": 30,
        "maximum_auxiliary_label_horizon_return_count": 63,
        "training_replay_state_rows": 378,
        "training_observation_warmup_decisions": 251,
        "training_loss_bearing_origin_count": 63,
        "validation_score_transition_count": 63,
        "age_state_bin_count": 61,
        "minimum_age_sessions": 0,
        "maximum_age_sessions": 60,
    }
    assert M03R_TOP2000_DEV_CACHE_REQUIREMENT.expected_daily_ohlcv_tensor_shape == (
        1001,
        1999,
        5,
    )
    assert M03R_TOP2000_DEV_CACHE_REQUIREMENT.cache_contract_required
    assert not M03R_TOP2000_DEV_CACHE_REQUIREMENT.cache_contract_bound
    assert M03R_TOP2000_DEV_CACHE_REQUIREMENT.cache_contract_sha256 is None
    assert M03R_TOP2000_DEV_CACHE_REQUIREMENT.cache_manifest_sha256 is None
    assert M03R_TOP2000_DEV_CACHE_REQUIREMENT.later_content_addressed_binding_allowed


def test_top2000_dev_identity_rejects_canonical_v7_and_semantic_drift() -> None:
    setting_id = M03R_TOP2000_DEV_SETTING_IDS[0]
    assert (
        validate_m03r_top2000_dev_artifact_identity(
            protocol_generation=M03R_TOP2000_DEV_PROTOCOL_GENERATION,
            design_id=M03R_TOP2000_DEV_DESIGN_ID,
            setting_id=setting_id,
        ).setting_id
        == setting_id
    )
    with pytest.raises(M03RTop2000DevProtocolError, match="canonical PIT-300"):
        validate_m03r_top2000_dev_artifact_identity(
            protocol_generation=M03R_V7_PROTOCOL_GENERATION,
            design_id=M03R_TOP2000_DEV_DESIGN_ID,
            setting_id=setting_id,
        )
    with pytest.raises(M03RTop2000DevProtocolError, match="canonical PIT-300"):
        validate_m03r_top2000_dev_artifact_identity(
            protocol_generation=M03R_TOP2000_DEV_PROTOCOL_GENERATION,
            design_id=M03R_V7_DESIGN_ID,
            setting_id=setting_id,
        )
    with pytest.raises(M03RTop2000DevProtocolError, match="unknown TOP2000"):
        resolve_m03r_top2000_dev_setting(M03R_V7_PRIMARY_SETTING_IDS[0])
    with pytest.raises(M03RTop2000DevProtocolError, match="changed reviewed"):
        replace(M03R_TOP2000_DEV_SETTINGS[0], sharpe_mode="separate-total-risk-overlay")
