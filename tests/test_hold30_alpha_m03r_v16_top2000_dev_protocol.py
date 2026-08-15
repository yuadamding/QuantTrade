from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_FILL_RULE,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
    M03R_V16_SURVIVAL_WEIGHTS,
    M03RV16ProtocolError,
    resolve_m03r_v16_setting,
)


def test_v16_is_selection_first_predictive_research() -> None:
    assert len(M03R_V16_PROTOCOL_SHA256) == 64
    assert [row.selection_support_sessions for row in M03R_V16_SETTINGS] == [21, 30, 30]
    assert all(row.timing_horizon_sessions == 3 for row in M03R_V16_SETTINGS)
    assert all(row.score_component_weights == (0.85, 0.15) for row in M03R_V16_SETTINGS)
    assert M03R_V16_PREDICTIVE_SPEC.maximum_h100_requests == 6
    assert M03R_V16_PREDICTIVE_SPEC.reinforcement_learning_updates == 0
    assert M03R_V16_PREDICTIVE_SPEC.economic_optimizer_updates == 0
    assert M03R_V16_PREDICTIVE_SPEC.outer_2026_access_authorized is False
    assert M03R_V16_PREDICTIVE_SPEC.v15_model_or_optimizer_state_reuse_authorized is False
    assert M03R_V16_PREDICTIVE_SPEC.score_learning_rates == (2.0e-5, 1.0e-4)
    assert M03R_V16_PREDICTIVE_SPEC.scale_calibration_learning_rate == 1.0e-4
    assert M03R_V16_PREDICTIVE_SPEC.reportable is False
    assert M03R_V16_FILL_RULE.startswith("observe-close-t-fill-next-close")


def test_v16_survival_target_is_normalized_geometric_and_nonmandatory() -> None:
    assert len(M03R_V16_SURVIVAL_WEIGHTS) == 30
    assert math.fsum(M03R_V16_SURVIVAL_WEIGHTS) == pytest.approx(1.0)
    assert all(
        later < earlier
        for earlier, later in zip(
            M03R_V16_SURVIVAL_WEIGHTS,
            M03R_V16_SURVIVAL_WEIGHTS[1:],
            strict=False,
        )
    )
    assert M03R_V16_SURVIVAL_WEIGHTS[-1] > 0.0


def test_v16_settings_are_target_only_comparisons() -> None:
    assert resolve_m03r_v16_setting(0) is M03R_V16_SETTINGS[0]
    assert resolve_m03r_v16_setting(M03R_V16_SETTINGS[2].setting_id) is M03R_V16_SETTINGS[2]
    assert len({row.selection_target for row in M03R_V16_SETTINGS}) == 3
    with pytest.raises(M03RV16ProtocolError, match="setting drifted"):
        replace(M03R_V16_SETTINGS[0], selection_support_sessions=30)
    with pytest.raises(M03RV16ProtocolError, match="specification drifted"):
        replace(M03R_V16_PREDICTIVE_SPEC, reinforcement_learning_updates=1)
