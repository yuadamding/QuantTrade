from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import (
    M03R_V13_HORIZONS,
    M03R_V13_PREDICTIVE_SPEC,
    M03R_V13_PROTOCOL_SHA256,
    M03R_V13_SETTINGS,
    M03RV13ProtocolError,
    resolve_m03r_v13_setting,
)


def test_v13_is_h3_only_predictive_research() -> None:
    assert M03R_V13_HORIZONS == (3,)
    assert len(M03R_V13_PROTOCOL_SHA256) == 64
    assert M03R_V13_PREDICTIVE_SPEC.economic_optimizer_updates == 0
    assert M03R_V13_PREDICTIVE_SPEC.outer_2026_access_authorized is False
    assert M03R_V13_PREDICTIVE_SPEC.v12_model_or_optimizer_state_reuse_authorized is False
    assert M03R_V13_PREDICTIVE_SPEC.maximum_h100_requests == 4
    assert M03R_V13_PREDICTIVE_SPEC.reportable is False
    assert M03R_V13_PREDICTIVE_SPEC.promotable is False


def test_v13_rank_ablation_shares_the_executed_mean() -> None:
    ranked, control = M03R_V13_SETTINGS
    assert ranked.ranking_objective == "rank-gaussian-correlation"
    assert control.ranking_objective == "none"
    assert ranked.rank_score_is_economic_mean is True
    assert control.rank_score_is_economic_mean is True
    assert ranked.selected_horizon_sessions == control.selected_horizon_sessions == 3
    assert ranked.component_weights == (0.25, 0.45, 0.30)
    assert control.component_weights == (0.0, 0.60, 0.40)
    assert resolve_m03r_v13_setting(0) is ranked
    assert resolve_m03r_v13_setting(control.setting_id) is control


def test_v13_protocol_rejects_result_moving_drift() -> None:
    with pytest.raises(M03RV13ProtocolError, match="setting drifted"):
        replace(M03R_V13_SETTINGS[0], selected_horizon_sessions=21)
    with pytest.raises(M03RV13ProtocolError, match="specification drifted"):
        replace(M03R_V13_PREDICTIVE_SPEC, economic_optimizer_updates=1)
    with pytest.raises(M03RV13ProtocolError, match="specification drifted"):
        replace(M03R_V13_PREDICTIVE_SPEC, outer_2026_access_authorized=True)
