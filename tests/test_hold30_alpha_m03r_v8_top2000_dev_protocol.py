from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_TOP2000_DEV_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ACTIVE_POLICY,
    M03R_V8_ALPHA_PRETRAINING,
    M03R_V8_TOP2000_DEV_CAUSAL_FIELDS,
    M03R_V8_TOP2000_DEV_DESIGN_ID,
    M03R_V8_TOP2000_DEV_PROTOCOL,
    M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_V8_TOP2000_DEV_REFERENCE_SETTING_ID,
    M03R_V8_TOP2000_DEV_SETTING_IDS,
    M03R_V8_TOP2000_DEV_SETTINGS,
    M03RV8Top2000DevProtocolError,
    m03r_v8_top2000_dev_protocol_payload,
    resolve_m03r_v8_top2000_dev_setting,
)


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_v8_identity_is_disjoint_and_visibly_development_only() -> None:
    assert M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION != (
        M03R_TOP2000_DEV_PROTOCOL_GENERATION
    )
    assert M03R_V8_TOP2000_DEV_DESIGN_ID.endswith("pretrained-incremental-costgate-v1")
    assert set(M03R_V8_TOP2000_DEV_SETTING_IDS).isdisjoint(M03R_TOP2000_DEV_SETTING_IDS)
    assert len(M03R_V8_TOP2000_DEV_SETTING_IDS) == 8
    assert all("top2000-dev" in value for value in M03R_V8_TOP2000_DEV_SETTING_IDS)


def test_alpha_pretraining_contract_matches_the_phase0_decision() -> None:
    spec = M03R_V8_ALPHA_PRETRAINING
    assert spec.horizons_trading_sessions == (5, 21, 30, 63)
    assert spec.horizon_loss_weights == pytest.approx((0.10, 0.35, 0.40, 0.15))
    assert spec.ranking_loss_weight == pytest.approx(0.50)
    assert spec.huber_loss_weight == pytest.approx(0.30)
    assert spec.distributional_loss_weight == pytest.approx(0.20)
    assert spec.early_stop_horizons_trading_sessions == (21, 30)
    assert spec.minimum_mean_spearman_rank_ic == pytest.approx(0.02)
    assert spec.minimum_positive_rank_ic_fold_count == 4
    assert spec.training_fold_only
    assert spec.inner_validation_only_for_early_stop_and_calibration
    assert spec.benchmark_relative_or_factor_residual_targets
    assert spec.raw_ohlcv_only_actor_inputs
    assert not spec.factors_are_actor_inputs


def test_active_policy_contract_bounds_new_risk_without_suppressing_exits() -> None:
    spec = M03R_V8_ACTIVE_POLICY
    assert spec.recent_raw_context_trading_sessions == 42
    assert spec.learned_temporal_context_trading_sessions == 252
    assert spec.persistence_coefficient_basis_points == pytest.approx(2.0)
    assert spec.persistence_preference_horizon_sessions == 30
    assert spec.persistence_is_soft_and_one_sided
    assert spec.reference_exact_hold_action_temperature == pytest.approx(1.0)
    assert spec.softened_exact_hold_action_temperature == pytest.approx(1.5)
    assert spec.policy_operates_on_incremental_active_weights
    assert spec.learned_exit_hazard_precedes_cost_gate
    assert spec.confidence_controls_new_and_expanding_risk_only
    assert spec.confidence_does_not_suppress_learned_exits
    assert spec.maximum_incremental_one_way_turnover == pytest.approx(0.02)
    assert spec.retention_hurdle_multiplier < spec.entry_hurdle_multiplier
    assert spec.training_one_way_cost_basis_points == 20
    assert spec.evaluation_one_way_cost_basis_points == (0, 10, 20, 40)
    assert spec.annual_tracking_error_floor is None
    assert spec.annual_tracking_error_ceiling == pytest.approx(0.06)
    assert spec.active_beta_equivalence_absolute_upper_bound == pytest.approx(0.10)
    assert spec.factor_sector_projection_required
    assert spec.nonzero_content_bound_factor_sector_slabs_required
    assert spec.projection_mode == "benchmark-radial-factor-beta-te-v1"
    assert spec.relaxed_factor_sector_bound_multiplier == pytest.approx(1.5)
    assert spec.benchmark_anchoring_required


def test_eight_rows_change_only_their_declared_causal_field() -> None:
    assert tuple(row.setting_index for row in M03R_V8_TOP2000_DEV_SETTINGS) == tuple(
        range(8)
    )
    reference = M03R_V8_TOP2000_DEV_SETTINGS[0]
    assert reference.setting_id == M03R_V8_TOP2000_DEV_REFERENCE_SETTING_ID
    assert reference.ablation_of is None
    assert reference.declared_causal_field is None
    for row in M03R_V8_TOP2000_DEV_SETTINGS[1:]:
        changed = tuple(
            field
            for field in M03R_V8_TOP2000_DEV_CAUSAL_FIELDS
            if getattr(row, field) != getattr(reference, field)
        )
        assert changed == (row.declared_causal_field,)
        assert row.ablation_of == reference.setting_id

    assert M03R_V8_TOP2000_DEV_SETTINGS[1].alpha_pretraining_mode == (
        "joint-random-initialization"
    )
    assert M03R_V8_TOP2000_DEV_SETTINGS[2].ranking_loss_weight == 0.0
    assert M03R_V8_TOP2000_DEV_SETTINGS[
        3
    ].exact_hold_action_temperature == pytest.approx(1.5)
    assert M03R_V8_TOP2000_DEV_SETTINGS[4].cost_gate_mode == "disabled"
    assert M03R_V8_TOP2000_DEV_SETTINGS[5].cost_gate_mode == "strong"
    assert M03R_V8_TOP2000_DEV_SETTINGS[6].exit_hazard_mode.startswith("fixed-")
    assert M03R_V8_TOP2000_DEV_SETTINGS[7].factor_sector_bound_multiplier == 1.5


def test_all_rows_remain_nonreportable_and_nonpromotable() -> None:
    for row in M03R_V8_TOP2000_DEV_SETTINGS:
        assert row.development_only
        assert row.future_selected_universe
        assert not row.reportable
        assert not row.promotable
        assert not row.outer_lockbox_evaluation_authorized


def test_protocol_payload_and_setting_resolution_are_content_addressed() -> None:
    payload = m03r_v8_top2000_dev_protocol_payload()
    unsigned = dict(payload)
    assert unsigned.pop("receipt_sha256") == _sha256(unsigned)
    assert payload == M03R_V8_TOP2000_DEV_PROTOCOL
    assert payload["receipt_sha256"] == M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    assert payload["phase0_distinct_policy_gate_must_pass_before_remote_training"]
    assert payload["automatic_update_extension_forbidden"]
    for row in M03R_V8_TOP2000_DEV_SETTINGS:
        assert resolve_m03r_v8_top2000_dev_setting(row.setting_index) == row
        assert resolve_m03r_v8_top2000_dev_setting(row.setting_id) == row
        assert row.receipt_sha256 == _sha256(asdict(row))


def test_identity_drift_and_unknown_settings_fail_closed() -> None:
    reference = M03R_V8_TOP2000_DEV_SETTINGS[0]
    with pytest.raises(M03RV8Top2000DevProtocolError, match="nonreportable"):
        replace(reference, reportable=True)
    with pytest.raises(M03RV8Top2000DevProtocolError, match="shared-contract"):
        replace(reference, shared_active_policy_sha256="0" * 64)
    with pytest.raises(M03RV8Top2000DevProtocolError, match="unknown"):
        resolve_m03r_v8_top2000_dev_setting(8)
    with pytest.raises(M03RV8Top2000DevProtocolError, match="unknown"):
        resolve_m03r_v8_top2000_dev_setting("missing")
    with pytest.raises(M03RV8Top2000DevProtocolError, match="boolean"):
        resolve_m03r_v8_top2000_dev_setting(True)
