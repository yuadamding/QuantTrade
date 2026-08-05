"""Identity and soft-persistence tests for immutable M03R v6."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN_ID as M03R_V5_DESIGN_ID,
    M03R_PROTOCOL_GENERATION as M03R_V5_PROTOCOL_GENERATION,
    M03R_SETTING_IDS as M03R_V5_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SETTING_IDS,
    M03R_SETTINGS,
    M03R_SETTINGS_BY_ID,
    M03R_SOFT_PERSISTENCE,
    M03R_SUPERSEDED_PROTOCOL_GENERATION,
    M03R_V6_SCHEMA_VERSION,
    M03RProtocolError,
    m03r_v6_design_payload,
    validate_m03r_v6_artifact_identity,
)
from rl_quant.training.designs import DESIGNS, HOLD30_M03R_V6_BASE_DESIGN


def test_v6_identity_is_disjoint_from_v5() -> None:
    assert M03R_PROTOCOL_GENERATION == "prelockbox-hold30-active-alpha-m03r-v6"
    assert M03R_DESIGN_ID == "daily_raw_pit300_hold30_m03r_v6"
    assert M03R_SUPERSEDED_PROTOCOL_GENERATION == M03R_V5_PROTOCOL_GENERATION
    assert M03R_V6_SCHEMA_VERSION == 6
    assert M03R_DESIGN_ID != M03R_V5_DESIGN_ID
    assert set(M03R_SETTING_IDS).isdisjoint(M03R_V5_SETTING_IDS)
    assert HOLD30_M03R_V6_BASE_DESIGN == M03R_DESIGN_ID
    assert DESIGNS[M03R_DESIGN_ID].note.startswith("M03R v6 soft-persistence")

    payload = m03r_v6_design_payload()
    assert payload["schema_version"] == 6
    assert payload["supersedes_protocol_generation"] == M03R_V5_PROTOCOL_GENERATION
    assert payload["v5_artifacts_retain_their_original_identity"] is True
    assert payload["launch_authorized"] is False
    assert payload["launch_blockers"]


def test_30_sessions_is_only_a_soft_inductive_bias() -> None:
    contract = M03R_SOFT_PERSISTENCE
    assert M03R_DESIGN.soft_persistence is contract
    assert contract.holding_preference_horizon_sessions == 30
    assert contract.holding_preference_is_inductive_bias_only
    assert contract.minimum_holding_period_sessions is None
    assert not contract.sell_mask_before_preference_horizon
    assert not contract.forced_expiry_at_preference_horizon
    assert not contract.holding_duration_is_promotion_gate
    assert not contract.turnover_target_is_holding_duration_proxy
    assert not contract.turnover_is_hard_holding_constraint
    assert contract.early_exit_always_allowed

    with pytest.raises(M03RProtocolError, match="minimum holding period"):
        replace(contract, minimum_holding_period_sessions=1)
    with pytest.raises(M03RProtocolError, match="mask sales"):
        replace(contract, sell_mask_before_preference_horizon=True)
    with pytest.raises(M03RProtocolError, match="mask sales"):
        replace(contract, forced_expiry_at_preference_horizon=True)
    with pytest.raises(M03RProtocolError, match="mask sales"):
        replace(contract, holding_duration_is_promotion_gate=True)


def test_soft_penalty_and_warmup_are_content_bound() -> None:
    contract = M03R_SOFT_PERSISTENCE
    assert contract.early_exit_penalty_shape == "quadratic-one-sided"
    assert contract.early_exit_penalty_bp_per_unit_at_age_zero == pytest.approx(5.0)
    assert (
        contract.early_exit_penalty_inner_development_grid_bp_per_unit_at_age_zero
        == (2.0, 5.0, 10.0)
    )
    assert contract.early_exit_penalty_warmup_shape == "linear-from-zero"
    assert contract.early_exit_penalty_linear_warmup_fraction == pytest.approx(0.10)
    assert contract.early_exit_sold_notional_epsilon == pytest.approx(1e-12)
    assert contract.holding_to_economic_gradient_norm_ratio_diagnostic_band == (
        0.05,
        0.15,
    )
    assert not contract.gradient_norm_ratio_is_promotion_gate

    assert contract.age_weight(0.0) == pytest.approx(1.0)
    assert contract.age_weight(15.0) == pytest.approx(0.25)
    assert contract.age_weight(30.0) == pytest.approx(0.0)
    assert contract.age_weight(45.0) == pytest.approx(0.0)
    assert contract.warmup_scale(0.0) == pytest.approx(0.0)
    assert contract.warmup_scale(0.05) == pytest.approx(0.5)
    assert contract.warmup_scale(0.10) == pytest.approx(1.0)
    assert contract.warmup_scale(1.0) == pytest.approx(1.0)

    with pytest.raises(M03RProtocolError, match="shape drifted"):
        replace(contract, early_exit_penalty_shape="linear")
    with pytest.raises(M03RProtocolError, match="10% linear warmup"):
        replace(contract, early_exit_penalty_linear_warmup_fraction=0.20)


def test_exact_hold_is_available_but_never_required() -> None:
    persistence = M03R_SOFT_PERSISTENCE
    model = M03R_DESIGN.model
    assert persistence.exact_hold_action_supported
    assert not persistence.exact_hold_action_required
    assert model.exact_hold_action_supported
    assert not model.exact_hold_action_required

    with pytest.raises(M03RProtocolError, match="supports but never requires"):
        replace(persistence, exact_hold_action_required=True)


def test_temporal_credit_is_not_a_holding_rule() -> None:
    temporal = asdict(M03R_DESIGN.temporal)
    assert temporal["economic_origin_post_fill_return_count"] == 30
    assert temporal["rollout_trading_sessions"] == 63
    assert temporal["learned_temporal_context_trading_sessions"] == 252
    assert "target_holding_trading_sessions" not in temporal
    assert "minimum_holding_period_sessions" not in temporal


def test_v6_inventory_has_12_settings_and_preserves_fixed_prior() -> None:
    assert len(M03R_SETTINGS) == 12
    assert tuple(row.setting_index for row in M03R_SETTINGS) == tuple(range(12))
    assert [row.setting_id for row in M03R_SETTINGS if row.promotion_eligible] == [
        M03R_CANONICAL_SETTING_ID
    ]
    assert M03R_SETTINGS_BY_ID["A08-fixed-exit-hazard-v6"].exit_hazard_mode == (
        "fixed-hold30-prior"
    )
    assert not M03R_SETTINGS_BY_ID[
        "A08-fixed-exit-hazard-v6"
    ].emits_exact_hold_action
    assert M03R_SETTINGS_BY_ID[M03R_CANONICAL_SETTING_ID].emits_exact_hold_action
    assert not M03R_SETTINGS_BY_ID[
        "A11-no-exact-hold-atom"
    ].emits_exact_hold_action
    assert "A11-no-exact-hold-atom" in M03R_SETTING_IDS


def test_v6_ablation_inventory_is_single_field_relative_to_canonical() -> None:
    canonical = M03R_SETTINGS_BY_ID[M03R_CANONICAL_SETTING_ID]
    causal_fields = (
        "objective_mode",
        "age_aware_holding",
        "soft_persistence",
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
        "exact_hold_action_supported",
    )
    expected = {
        "A04-no-downside-score-adjustment-v6": "use_downside_adjusted_stock_score",
        "A05-fixed-te-floor-v6": "annual_tracking_error_floor",
        "A06-sharpe-overlay-v6": "sharpe_mode",
        "A07-direct-sharpe-v6": "sharpe_mode",
        "A08-fixed-exit-hazard-v6": "exit_hazard_mode",
        "A09-no-long-context-v6": "slow_context_trading_sessions",
        "A10-no-factor-neutral-projection-v6": "factor_sector_projection",
        "A11-no-exact-hold-atom": "exact_hold_action_supported",
    }
    for setting_id, expected_field in expected.items():
        row = M03R_SETTINGS_BY_ID[setting_id]
        changed = [
            field
            for field in causal_fields
            if getattr(row, field) != getattr(canonical, field)
        ]
        assert changed == [expected_field]


def test_v6_artifact_identity_rejects_v5_generation() -> None:
    row = validate_m03r_v6_artifact_identity(
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=M03R_CANONICAL_SETTING_ID,
    )
    assert row.setting_id == M03R_CANONICAL_SETTING_ID

    with pytest.raises(M03RProtocolError, match="v5 remains immutable"):
        validate_m03r_v6_artifact_identity(
            protocol_generation=M03R_V5_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
        )
