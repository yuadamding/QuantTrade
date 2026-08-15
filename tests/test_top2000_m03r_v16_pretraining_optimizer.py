from __future__ import annotations

import pytest
import torch
from torch import nn

from rl_quant.training.top2000_m03r_v16_policy import Top2000M03RV16PredictivePolicy
from rl_quant.training.top2000_m03r_v16_pretraining_optimizer import (
    M03RV16OptimizerError,
    build_m03r_v16_optimizer,
    validate_m03r_v16_optimizer,
)


def _policy(setting: int = 0) -> Top2000M03RV16PredictivePolicy:
    return Top2000M03RV16PredictivePolicy(
        setting, token_dim=16, raw_stock_chunk=8, activation_checkpointing=False
    )


def test_v16_optimizer_trains_only_encoder_and_selection_head() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v16_optimizer(policy)
    validate_m03r_v16_optimizer(policy, optimizer, partition)
    assert partition.encoder_parameter_names
    assert set(partition.selection_head_parameter_names) == {
        "selection_score_head.bias",
        "selection_score_head.weight",
    }
    with pytest.raises(M03RV16OptimizerError, match="selection score training only"):
        build_m03r_v16_optimizer(policy, "scale_calibration")  # type: ignore[arg-type]


def test_v16_all_layernorm_bias_and_cash_bias_parameters_have_no_decay() -> None:
    policy = _policy()
    optimizer, _partition = build_m03r_v16_optimizer(policy)
    decay_by_id = {
        id(parameter): float(group["weight_decay"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for module in policy.modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters(recurse=False):
                if parameter.requires_grad:
                    assert decay_by_id[id(parameter)] == 0.0
        bias = getattr(module, "bias", None)
        if isinstance(bias, nn.Parameter) and bias.requires_grad:
            assert decay_by_id[id(bias)] == 0.0
    assert decay_by_id[id(policy.source_policy.core.cash_bias)] == 0.0


def test_v16_setting_neutral_initial_parameter_bytes() -> None:
    torch.manual_seed(16017)
    first = _policy(0)
    torch.manual_seed(16017)
    second = _policy(2)
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
