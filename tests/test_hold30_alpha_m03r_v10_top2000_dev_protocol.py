from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import (
    M03R_V10_PREDICTIVE_SPEC,
    M03R_V10_PROTOCOL_SHA256,
    M03R_V10_SETTINGS,
    M03RV10ProtocolError,
    resolve_m03r_v10_setting,
)


def test_v10_is_a_fresh_failed_v9_bound_predictive_only_generation() -> None:
    assert len(M03R_V10_PROTOCOL_SHA256) == 64
    assert M03R_V10_PREDICTIVE_SPEC.minimum_mean_spearman_rank_ic == 0.020
    assert M03R_V10_PREDICTIVE_SPEC.economic_optimizer_updates == 0
    assert not M03R_V10_PREDICTIVE_SPEC.v9_state_reuse_authorized
    assert not M03R_V10_PREDICTIVE_SPEC.outer_2026_access_authorized
    assert all(row.target_mode == "factor-residual" for row in M03R_V10_SETTINGS)


def test_v10_settings_isolate_rank_geometry_and_horizon_support() -> None:
    control, rank_gaussian, focused = M03R_V10_SETTINGS
    assert control.ranking_objective == "standardized-return-listwise"
    assert rank_gaussian.ranking_objective == "rank-gaussian-correlation"
    assert rank_gaussian.horizon_loss_weights == control.horizon_loss_weights
    assert focused.ranking_objective == rank_gaussian.ranking_objective
    assert focused.horizon_loss_weights[:1] == (0.0,)
    assert focused.horizon_loss_weights[-1:] == (0.0,)
    assert resolve_m03r_v10_setting(1) == rank_gaussian
    assert resolve_m03r_v10_setting(rank_gaussian.setting_id) == rank_gaussian


def test_v10_gate_and_setting_drift_fail_closed() -> None:
    with pytest.raises(M03RV10ProtocolError, match="specification"):
        replace(M03R_V10_PREDICTIVE_SPEC, minimum_mean_spearman_rank_ic=0.019)
    with pytest.raises(M03RV10ProtocolError, match="setting"):
        replace(  # type: ignore[arg-type]
            M03R_V10_SETTINGS[1], target_mode="benchmark-relative"
        )
