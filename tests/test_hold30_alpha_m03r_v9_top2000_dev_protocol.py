from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256,
    M03R_V9_PREDICTIVE_SPEC,
    M03R_V9_PROTOCOL_SHA256,
    M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES,
    M03R_V9_SETTINGS,
    M03RV9HorizonBinding,
    M03RV9ProtocolError,
    resolve_m03r_v9_setting,
)


def test_v9_is_a_disjoint_three_setting_predictive_only_generation() -> None:
    assert len(M03R_V9_SETTINGS) == 3
    assert len(M03R_V9_PROTOCOL_SHA256) == 64
    assert len(M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256) == 64
    assert M03R_V9_PREDICTIVE_SPEC.maximum_optimizer_updates == 64
    assert M03R_V9_PREDICTIVE_SPEC.qualification_evaluation_updates == (64,)
    assert not M03R_V9_PREDICTIVE_SPEC.early_stopping_enabled
    assert M03R_V9_PREDICTIVE_SPEC.economic_optimizer_updates == 0
    assert M03R_V9_PREDICTIVE_SPEC.required_risk_exposure_families == (
        M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES
    )
    assert M03R_V9_PREDICTIVE_SPEC.maximum_h100_requests == 6
    assert not M03R_V9_PREDICTIVE_SPEC.outer_lockbox_access_authorized
    assert not M03R_V9_PREDICTIVE_SPEC.reportable
    assert not M03R_V9_PREDICTIVE_SPEC.promotable


def test_settings_change_only_target_or_normalized_ranking_contract() -> None:
    reference, no_ranking, benchmark = M03R_V9_SETTINGS
    assert reference.target_mode == no_ranking.target_mode == "factor-residual"
    assert reference.ranking_enabled and not no_ranking.ranking_enabled
    assert benchmark.target_mode == "benchmark-relative"
    assert benchmark.ranking_enabled
    assert resolve_m03r_v9_setting(0) is reference
    assert resolve_m03r_v9_setting(benchmark.setting_id) is benchmark
    with pytest.raises(M03RV9ProtocolError):
        resolve_m03r_v9_setting(3)
    with pytest.raises(M03RV9ProtocolError):
        replace(reference, setting_index=99)


def test_horizon_binding_forbids_qualify_one_horizon_trade_another() -> None:
    binding = M03RV9HorizonBinding(21, 21, 21)
    assert len(binding.receipt_sha256) == 64
    with pytest.raises(M03RV9ProtocolError, match="must be identical"):
        M03RV9HorizonBinding(21, 30, 30)
    with pytest.raises(M03RV9ProtocolError, match="must be identical"):
        M03RV9HorizonBinding(63, 63, 63)


def test_no_ranking_weights_are_unit_normalized() -> None:
    assert M03R_V9_PREDICTIVE_SPEC.ranked_component_weights == (0.50, 0.30, 0.20)
    assert M03R_V9_PREDICTIVE_SPEC.no_ranking_component_weights == (0.0, 0.60, 0.40)
    assert sum(M03R_V9_PREDICTIVE_SPEC.no_ranking_component_weights) == pytest.approx(
        1.0
    )
