from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9PolicyError,
    Top2000M03RV9PredictivePolicy,
)


def _policy(horizon: int = 30) -> Top2000M03RV9PredictivePolicy:
    return Top2000M03RV9PredictivePolicy(
        0,
        M03RV9HorizonBinding(horizon, horizon, horizon),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def test_one_distribution_supplies_training_and_selected_execution_tensors() -> None:
    policy = _policy(21)
    state = torch.randn((2, 5, 16), requires_grad=True)
    available = torch.tensor(
        [[True, True, True, True, True], [True, True, False, True, True]]
    )
    distribution = policy.alpha_distribution(state, available)

    assert distribution.mean_by_horizon.shape == (2, 5, 4)
    assert distribution.log_scale_by_horizon.shape == (2, 5, 4)
    assert distribution.selected_horizon_sessions == 21
    assert torch.equal(distribution.selected_mean, distribution.mean_by_horizon[..., 1])
    assert torch.equal(
        distribution.selected_scale,
        torch.exp(distribution.log_scale_by_horizon[..., 1]),
    )
    assert not hasattr(distribution, "alpha_downside_30d")
    assert torch.equal(distribution.mean_by_horizon[:, 0], torch.zeros((2, 4)))
    assert torch.equal(distribution.mean_by_horizon[1, 2], torch.zeros(4))
    (distribution.selected_mean.sum() + distribution.selected_scale.sum()).backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert policy.alpha_scale_head.weight.grad is not None


def test_selected_horizon_tamper_is_rejected() -> None:
    distribution = _policy(30).alpha_distribution(
        torch.randn((1, 4, 16)), torch.ones((1, 4), dtype=torch.bool)
    )
    with pytest.raises(M03RV9PolicyError, match="selected mean"):
        replace(
            distribution, selected_mean=distribution.mean_by_horizon[..., 1]
        ).validate()
    with pytest.raises(M03RV9PolicyError, match="selected scale"):
        replace(
            distribution, selected_scale=2.0 * distribution.selected_scale
        ).validate()


def test_checkpoint_head_identity_binds_horizon_and_both_heads() -> None:
    policy = _policy(30)
    first = policy.alpha_head_identity()
    assert first.selected_alpha_horizon == 30
    assert len(first.alpha_mean_head_state_sha256) == 64
    assert len(first.alpha_scale_head_state_sha256) == 64
    with torch.no_grad():
        policy.alpha_scale_head.bias.add_(0.1)
    second = policy.alpha_head_identity()
    assert second.alpha_mean_head_state_sha256 == first.alpha_mean_head_state_sha256
    assert second.alpha_scale_head_state_sha256 != first.alpha_scale_head_state_sha256
