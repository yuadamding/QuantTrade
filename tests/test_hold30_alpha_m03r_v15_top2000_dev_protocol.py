from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_HORIZONS,
    M03R_V15_PREDICTIVE_SPEC,
    M03R_V15_PROTOCOL_SHA256,
    M03R_V15_SETTINGS,
    M03RV15ProtocolError,
    resolve_m03r_v15_setting,
)


def test_v15_is_h3_only_predictive_research() -> None:
    assert M03R_V15_HORIZONS == (3,)
    assert len(M03R_V15_PROTOCOL_SHA256) == 64
    assert M03R_V15_PREDICTIVE_SPEC.economic_optimizer_updates == 0
    assert M03R_V15_PREDICTIVE_SPEC.outer_2026_access_authorized is False
    assert M03R_V15_PREDICTIVE_SPEC.v14_model_or_optimizer_state_reuse_authorized is False
    assert M03R_V15_PREDICTIVE_SPEC.maximum_h100_requests == 4
    assert M03R_V15_PREDICTIVE_SPEC.inner_validation_origins_per_fold == 32
    assert M03R_V15_PREDICTIVE_SPEC.checkpoint_selection_enabled is True
    assert M03R_V15_PREDICTIVE_SPEC.early_stopping_enabled is False
    assert M03R_V15_PREDICTIVE_SPEC.reportable is False
    assert M03R_V15_PREDICTIVE_SPEC.promotable is False


def test_v15_rank_ablation_shares_the_executed_mean() -> None:
    ranked, control = M03R_V15_SETTINGS
    assert ranked.ranking_objective == "rank-gaussian-correlation"
    assert control.ranking_objective == "none"
    assert ranked.rank_score_is_economic_mean is True
    assert control.rank_score_is_economic_mean is True
    assert ranked.selected_horizon_sessions == control.selected_horizon_sessions == 3
    assert ranked.component_weights == (0.25, 0.45, 0.30)
    assert control.component_weights == (0.0, 0.45, 0.30)
    assert resolve_m03r_v15_setting(0) is ranked
    assert resolve_m03r_v15_setting(control.setting_id) is control


def test_v15_protocol_rejects_result_moving_drift() -> None:
    with pytest.raises(M03RV15ProtocolError, match="setting drifted"):
        replace(M03R_V15_SETTINGS[0], selected_horizon_sessions=21)
    with pytest.raises(M03RV15ProtocolError, match="specification drifted"):
        replace(M03R_V15_PREDICTIVE_SPEC, economic_optimizer_updates=1)
    with pytest.raises(M03RV15ProtocolError, match="specification drifted"):
        replace(M03R_V15_PREDICTIVE_SPEC, outer_2026_access_authorized=True)
