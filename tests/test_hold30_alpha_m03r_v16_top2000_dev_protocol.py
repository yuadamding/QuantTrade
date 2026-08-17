from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from rl_quant.models.daily_policy import hold30_release_hazard
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_FILL_RULE,
    M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SCHEDULING_POLICY,
    M03R_V16_SETTINGS,
    M03R_V16_SURVIVAL_AFTER_DAY_30,
    M03R_V16_SURVIVAL_WEIGHTS,
    M03RV16ProtocolError,
    resolve_m03r_v16_setting,
)
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC


def test_v16_is_a_selection_only_primary_hypothesis_screen() -> None:
    assert len(M03R_V16_PROTOCOL_SHA256) == 64
    assert [row.numerical_target_support_sessions for row in M03R_V16_SETTINGS] == [
        21,
        30,
        30,
    ]
    assert {row.common_label_support_sessions for row in M03R_V16_SETTINGS} == {30}
    assert [row.promotion_eligible for row in M03R_V16_SETTINGS] == [False, False, True]
    assert M03R_V16_PREDICTIVE_SPEC.primary_setting_index == 2
    assert M03R_V16_PREDICTIVE_SPEC.maximum_h100_requests == 6
    assert M03R_V16_PREDICTIVE_SPEC.timing_optimizer_updates == 0
    assert M03R_V16_PREDICTIVE_SPEC.uncertainty_calibration_updates == 0
    assert M03R_V16_PREDICTIVE_SPEC.reinforcement_learning_updates == 0
    assert M03R_V16_PREDICTIVE_SPEC.economic_optimizer_updates == 0
    assert M03R_V16_PREDICTIVE_SPEC.outer_2026_access_authorized is False
    assert M03R_V16_PREDICTIVE_SPEC.reportable is False
    assert M03R_V16_PREDICTIVE_SPEC.hold_target_sessions == 30
    assert M03R_V16_PREDICTIVE_SPEC.hold_age_cap_sessions == 60
    assert M03R_V16_PREDICTIVE_SPEC.hold_prior_family == "legacy-hold30-v1"
    assert M03R_V16_PREDICTIVE_SPEC.hold_target_spec_sha256 == (
        LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    )
    assert M03R_V16_FILL_RULE.startswith("observe-close-t-fill-next-close")
    assert M03R_V16_SCHEDULING_POLICY == "independent-per-completion"


def test_v16_survival_target_uses_the_exact_reference_age_clock() -> None:
    observed = hold30_release_hazard(
        torch.arange(1, 31, dtype=torch.float64),
        torch.zeros(30, dtype=torch.float64),
    )
    assert tuple(float(value) for value in observed) == pytest.approx(
        M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS
    )
    assert len(M03R_V16_SURVIVAL_WEIGHTS) == 30
    assert M03R_V16_SURVIVAL_WEIGHTS[0] == 1.0
    assert math.fsum(M03R_V16_SURVIVAL_WEIGHTS) > 20.0
    assert 0.5 < M03R_V16_SURVIVAL_AFTER_DAY_30 < 0.6
    assert all(
        later < earlier
        for earlier, later in pairwise(M03R_V16_SURVIVAL_WEIGHTS)
    )


def test_v16_settings_differ_only_in_numerical_target() -> None:
    assert resolve_m03r_v16_setting(0) is M03R_V16_SETTINGS[0]
    assert (
        resolve_m03r_v16_setting(M03R_V16_SETTINGS[2].setting_id)
        is M03R_V16_SETTINGS[2]
    )
    assert len({row.selection_target for row in M03R_V16_SETTINGS}) == 3
    assert len({row.common_label_support_sessions for row in M03R_V16_SETTINGS}) == 1
    with pytest.raises(M03RV16ProtocolError, match="setting drifted"):
        replace(M03R_V16_SETTINGS[0], common_label_support_sessions=21)
    with pytest.raises(M03RV16ProtocolError, match="specification drifted"):
        replace(M03R_V16_PREDICTIVE_SPEC, reinforcement_learning_updates=1)
