from __future__ import annotations

import math

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_SETTINGS,
)
from rl_quant.training.top2000_m03r_v16_objective import (
    M03RV16PredictiveBatch,
    m03r_v16_scale_calibration_loss,
    m03r_v16_score_loss,
    m03r_v16_selection_target_scale,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)


def _batch(setting_index: int = 0) -> M03RV16PredictiveBatch:
    selection = torch.randn((3, 12), requires_grad=True)
    timing = torch.randn((3, 12), requires_grad=True)
    selection_scale = torch.zeros((3, 12), requires_grad=True)
    timing_scale = torch.zeros((3, 12), requires_grad=True)
    selection_target = torch.randn((3, 12))
    timing_target = torch.randn((3, 12))
    valid = torch.ones((3, 12), dtype=torch.bool)
    valid[:, 0] = False
    return M03RV16PredictiveBatch(
        executable_selection_mean=selection,
        selection_log_scale=selection_scale,
        selection_target=selection_target,
        selection_valid=valid,
        executable_timing_mean=timing,
        timing_log_scale=timing_scale,
        timing_target=timing_target,
        timing_valid=valid,
        setting=M03R_V16_SETTINGS[setting_index],
    )


def test_v16_policy_emits_distinct_raw_selection_and_timing_distributions() -> None:
    torch.manual_seed(17)
    policy = Top2000M03RV16PredictivePolicy(
        0,
        token_dim=8,
        raw_stock_chunk=4,
        activation_checkpointing=False,
    )
    output = policy.predictive_output(
        torch.randn((2, 6, 8)),
        torch.ones((2, 6), dtype=torch.bool),
    )
    assert not hasattr(output, "rank_score")
    assert not hasattr(output, "execution_score")
    assert output.raw_selection_mean.shape == (2, 6)
    assert output.raw_timing_mean.shape == (2, 6)
    assert torch.equal(output.raw_selection_mean[:, 0], torch.zeros(2))
    assert torch.equal(output.raw_timing_mean[:, 0], torch.zeros(2))
    assert output.selection_support_sessions == 21


def test_v16_target_settings_start_from_identical_parameter_bytes() -> None:
    policies = []
    for setting_index in range(3):
        torch.manual_seed(17)
        policies.append(
            Top2000M03RV16PredictivePolicy(
                setting_index,
                token_dim=8,
                raw_stock_chunk=4,
                activation_checkpointing=False,
            )
        )
    reference = policies[0].state_dict()
    for policy in policies[1:]:
        assert reference.keys() == policy.state_dict().keys()
        assert all(
            torch.equal(reference[name], policy.state_dict()[name])
            for name in reference
        )


def test_v16_score_and_scale_training_are_gradient_separated() -> None:
    batch = _batch()
    score = m03r_v16_score_loss(batch)
    score.total.backward(retain_graph=True)
    assert batch.executable_selection_mean.grad is not None
    assert batch.executable_timing_mean.grad is not None
    assert batch.selection_log_scale.grad is None
    assert batch.timing_log_scale.grad is None

    for tensor in (
        batch.executable_selection_mean,
        batch.executable_timing_mean,
    ):
        tensor.grad = None
    calibration = m03r_v16_scale_calibration_loss(batch)
    calibration.total.backward()
    assert batch.executable_selection_mean.grad is None
    assert batch.executable_timing_mean.grad is None
    assert batch.selection_log_scale.grad is not None
    assert batch.timing_log_scale.grad is not None


def test_v16_selection_target_scales_match_target_units() -> None:
    h21, h30, survival = (
        m03r_v16_selection_target_scale(setting) for setting in M03R_V16_SETTINGS
    )
    assert h21 == pytest.approx(0.02 * math.sqrt(21.0))
    assert h30 == pytest.approx(0.02 * math.sqrt(30.0))
    assert 0.0 < survival < 0.02
