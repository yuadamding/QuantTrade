from __future__ import annotations

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_ELIGIBLE_EXECUTION_HORIZONS,
    M03R_V12_HORIZONS,
    M03R_V12_PREDICTIVE_SPEC,
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_SETTINGS,
    M03R_V12_SELECTED_HORIZON_SESSIONS,
    M03RV12ProtocolError,
    resolve_m03r_v12_setting,
)


def test_v12_is_predictive_only_and_separates_rank_from_economic_scale() -> None:
    assert len(M03R_V12_PROTOCOL_SHA256) == 64
    assert M03R_V12_PREDICTIVE_SPEC.economic_optimizer_updates == 0
    assert not M03R_V12_PREDICTIVE_SPEC.outer_2026_access_authorized
    assert not M03R_V12_PREDICTIVE_SPEC.v11_model_or_optimizer_state_reuse_authorized
    assert [row.ranking_objective for row in M03R_V12_SETTINGS] == [
        "standardized-return-listwise",
        "rank-gaussian-correlation",
        "none",
    ]
    assert all(row.separate_rank_score_head for row in M03R_V12_SETTINGS)
    assert all(row.separate_economic_mean_scale_heads for row in M03R_V12_SETTINGS)
    assert M03R_V12_SETTINGS[2].component_weights == (0.0, 0.60, 0.40)
    assert M03R_V12_HORIZONS == (3, 5, 21, 30, 63)
    assert M03R_V12_SELECTED_HORIZON_SESSIONS == 3
    assert M03R_V12_ELIGIBLE_EXECUTION_HORIZONS == (3,)


def test_v12_setting_resolution_is_typed_and_fail_closed() -> None:
    for setting in M03R_V12_SETTINGS:
        assert resolve_m03r_v12_setting(setting.setting_index) == setting
        assert resolve_m03r_v12_setting(setting.setting_id) == setting
    with pytest.raises(M03RV12ProtocolError):
        resolve_m03r_v12_setting(True)
    with pytest.raises(M03RV12ProtocolError):
        resolve_m03r_v12_setting(3)
