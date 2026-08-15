from __future__ import annotations

import torch

from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_pretraining_optimizer import (
    build_m03r_v16_optimizer,
    validate_m03r_v16_optimizer,
)


def _policy() -> Top2000M03RV16PredictivePolicy:
    return Top2000M03RV16PredictivePolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def test_v16_score_optimizer_excludes_both_scale_heads() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v16_optimizer(policy, "score")
    validate_m03r_v16_optimizer(policy, optimizer, partition)
    assert partition.encoder_parameter_names
    assert set(partition.mean_parameter_names) == {
        "selection_mean_head.bias",
        "selection_mean_head.weight",
        "timing_mean_head.bias",
        "timing_mean_head.weight",
    }
    assert not partition.scale_parameter_names
    assert not policy.selection_scale_head.weight.requires_grad
    assert not policy.timing_scale_head.weight.requires_grad
    assert all(
        float(group["weight_decay"]) == 0.0
        for group in optimizer.param_groups
        if str(group["group_name"]).endswith("no-decay")
    )


def test_v16_scale_calibration_freezes_encoder_and_means() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v16_optimizer(policy, "scale_calibration")
    validate_m03r_v16_optimizer(policy, optimizer, partition)
    assert not partition.encoder_parameter_names
    assert not partition.mean_parameter_names
    assert set(partition.scale_parameter_names) == {
        "selection_scale_head.bias",
        "selection_scale_head.weight",
        "timing_scale_head.bias",
        "timing_scale_head.weight",
    }
    assert not policy.selection_mean_head.weight.requires_grad
    assert not policy.timing_mean_head.weight.requires_grad
    assert policy.selection_scale_head.weight.requires_grad
    assert policy.timing_scale_head.weight.requires_grad


def test_v16_setting_neutral_initial_parameter_bytes() -> None:
    torch.manual_seed(16017)
    first = Top2000M03RV16PredictivePolicy(
        0, token_dim=16, raw_stock_chunk=8, activation_checkpointing=False
    )
    torch.manual_seed(16017)
    second = Top2000M03RV16PredictivePolicy(
        2, token_dim=16, raw_stock_chunk=8, activation_checkpointing=False
    )
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
