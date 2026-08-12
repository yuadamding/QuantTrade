from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PREDICTIVE_SPEC,
    M03R_V11_PROTOCOL_SHA256,
    M03R_V11_SETTINGS,
    M03RV11ProtocolError,
)


def test_v11_is_new_predictive_only_identity_with_frozen_gates() -> None:
    assert len(M03R_V11_PROTOCOL_SHA256) == 64
    assert tuple(row.setting_index for row in M03R_V11_SETTINGS) == (0, 1, 2)
    assert M03R_V11_PREDICTIVE_SPEC.optimizer_updates == 64
    assert M03R_V11_PREDICTIVE_SPEC.minimum_mean_spearman_rank_ic == 0.020
    assert M03R_V11_PREDICTIVE_SPEC.bootstrap_replicates == 10_000
    assert M03R_V11_PREDICTIVE_SPEC.economic_optimizer_updates == 0
    assert not M03R_V11_PREDICTIVE_SPEC.outer_2026_access_authorized
    assert not M03R_V11_PREDICTIVE_SPEC.v9_or_v10_state_reuse_authorized


def test_v11_setting_and_resource_drift_fail_closed() -> None:
    with pytest.raises(M03RV11ProtocolError, match="setting"):
        replace(M03R_V11_SETTINGS[1], horizon_loss_weights=(0.25,) * 4)
    with pytest.raises(M03RV11ProtocolError, match="specification"):
        replace(M03R_V11_PREDICTIVE_SPEC, economic_optimizer_updates=64)
