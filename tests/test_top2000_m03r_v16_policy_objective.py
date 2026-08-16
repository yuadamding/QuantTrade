from __future__ import annotations

import math

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_SETTINGS,
    M03R_V16_SURVIVAL_WEIGHTS,
)
from rl_quant.training.top2000_m03r_v16_numerical import (
    M03RV16NumericalTrainingError,
)
from rl_quant.training.top2000_m03r_v16_objective import (
    M03RV16PredictiveBatch,
    m03r_v16_score_loss,
    m03r_v16_selection_target_scale,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)


def _batch(setting_index: int = 0) -> M03RV16PredictiveBatch:
    setting = M03R_V16_SETTINGS[setting_index]
    score = torch.randn((3, 12), requires_grad=True)
    target_z = torch.randn((3, 12))
    valid = torch.ones((3, 12), dtype=torch.bool)
    valid[:, 0] = False
    return M03RV16PredictiveBatch(
        executable_selection_score_z=score,
        selection_target_z=target_z,
        selection_target_economic=target_z * setting.selection_target_scale,
        selection_valid=valid,
        setting=setting,
    )


def test_v16_policy_emits_only_a_dimensionless_raw_selection_score() -> None:
    torch.manual_seed(17)
    policy = Top2000M03RV16PredictivePolicy(
        0, token_dim=8, raw_stock_chunk=4, activation_checkpointing=False
    )
    output = policy.predictive_output(
        torch.randn((2, 6, 8)), torch.ones((2, 6), dtype=torch.bool)
    )
    assert not hasattr(output, "rank_score")
    assert not hasattr(output, "execution_score")
    assert not hasattr(output, "raw_timing_mean")
    assert not hasattr(output, "raw_selection_log_scale")
    assert output.raw_selection_score_z.shape == (2, 6)
    assert torch.equal(output.raw_selection_score_z[:, 0], torch.zeros(2))
    assert output.numerical_target_support_sessions == 21


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


def test_v16_equal_standardized_errors_have_equal_head_gradients() -> None:
    gradients = []
    for setting_index in range(3):
        torch.manual_seed(1600)
        batch = _batch(setting_index)
        loss = m03r_v16_score_loss(batch)
        loss.total.backward()
        gradients.append(batch.executable_selection_score_z.grad.detach().clone())
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradients[0], gradients[2], rtol=0.0, atol=0.0)


def test_v16_nonfinite_training_tensor_uses_typed_numerical_boundary() -> None:
    batch = _batch()
    invalid_target = batch.selection_target_z.clone()
    invalid_target[1, 2] = torch.nan
    invalid = M03RV16PredictiveBatch(
        executable_selection_score_z=batch.executable_selection_score_z,
        selection_target_z=invalid_target,
        selection_target_economic=batch.selection_target_economic,
        selection_valid=batch.selection_valid,
        setting=batch.setting,
    )
    with pytest.raises(M03RV16NumericalTrainingError, match="non-finite"):
        m03r_v16_score_loss(invalid)


def test_v16_selection_target_scales_match_cumulative_units() -> None:
    h21, h30, survival = (
        m03r_v16_selection_target_scale(setting) for setting in M03R_V16_SETTINGS
    )
    assert h21 == pytest.approx(0.02 * math.sqrt(21.0))
    assert h30 == pytest.approx(0.02 * math.sqrt(30.0))
    assert survival == pytest.approx(
        0.02
        * math.sqrt(math.fsum(value * value for value in M03R_V16_SURVIVAL_WEIGHTS))
    )
    assert 0.09 < survival < 0.11
