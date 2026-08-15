from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_SETTINGS,
)
from rl_quant.training.top2000_m03r_v14_objective import (
    M03RV14PredictiveBatch,
    m03r_v14_predictive_loss,
)
from rl_quant.training.top2000_m03r_v14_policy import (
    M03RV14PolicyError,
    Top2000M03RV14PredictivePolicy,
)


def _batch(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    *,
    setting_index: int = 0,
) -> M03RV14PredictiveBatch:
    valid = torch.ones_like(mean, dtype=torch.bool)
    valid[:, 0] = False
    return M03RV14PredictiveBatch(
        predicted_mean=mean,
        predicted_log_scale=log_scale,
        target_log_return=target,
        valid=valid,
        setting=M03R_V14_SETTINGS[setting_index],
    )


def test_v14_policy_exposes_one_exact_rank_and_execution_score() -> None:
    torch.manual_seed(17)
    policy = Top2000M03RV14PredictivePolicy(
        0,
        selected_horizon_sessions=3,
        token_dim=8,
        raw_stock_chunk=4,
        activation_checkpointing=False,
    )
    state = torch.randn((2, 6, 8))
    available = torch.ones((2, 6), dtype=torch.bool)
    output = policy.predictive_output(state, available)
    assert output.rank_score.data_ptr() == output.economic_mean.data_ptr()
    assert output.economic_mean.shape == (2, 6)
    assert output.economic_scale.shape == (2, 6)
    assert torch.equal(output.economic_mean[:, 0], torch.zeros(2))
    assert float(policy.economic_mean_head.weight.std()) < 0.01
    assert torch.equal(
        policy.economic_scale_head.weight,
        torch.zeros_like(policy.economic_scale_head.weight),
    )
    assert policy.v14_head_identity().selected_alpha_horizon == 3
    with pytest.raises(M03RV14PolicyError, match="selected horizon"):
        Top2000M03RV14PredictivePolicy(
            0,
            selected_horizon_sessions=21,
            token_dim=8,
            activation_checkpointing=False,
        )


def test_v14_scale_calibration_does_not_update_the_shared_encoder() -> None:
    policy = Top2000M03RV14PredictivePolicy(
        0,
        selected_horizon_sessions=3,
        token_dim=8,
        raw_stock_chunk=4,
        activation_checkpointing=False,
    )
    output = policy.predictive_output(
        torch.randn((2, 6, 8)),
        torch.ones((2, 6), dtype=torch.bool),
    )
    output.economic_log_scale.sum().backward()
    assert policy.economic_scale_head.bias.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in policy.named_parameters()
        if name.startswith("source_policy.")
    )


def test_v14_rank_loss_updates_the_same_mean_consumed_by_execution() -> None:
    mean = torch.randn((3, 12), requires_grad=True)
    log_scale = torch.zeros_like(mean, requires_grad=True)
    target = torch.randn_like(mean)
    loss = m03r_v14_predictive_loss(_batch(mean, log_scale, target))
    rank_gradient = torch.autograd.grad(loss.ranking, mean, retain_graph=True)[0]
    assert torch.isfinite(rank_gradient).all()
    assert rank_gradient.abs().sum().item() > 0.0
    assert torch.autograd.grad(
        loss.ranking, log_scale, allow_unused=True, retain_graph=True
    )[0] is None
    assert torch.autograd.grad(
        loss.distributional, mean, allow_unused=True, retain_graph=True
    )[0] is None
    loss.total.backward()
    assert mean.grad is not None and mean.grad.abs().sum().item() > 0.0
    assert log_scale.grad is not None and log_scale.grad.abs().sum().item() > 0.0


def test_v14_no_rank_control_changes_only_the_objective_component() -> None:
    mean = torch.randn((3, 12), requires_grad=True)
    log_scale = torch.zeros_like(mean, requires_grad=True)
    target = torch.randn_like(mean)
    ranked = m03r_v14_predictive_loss(_batch(mean, log_scale, target))
    control = m03r_v14_predictive_loss(
        replace(_batch(mean, log_scale, target), setting=M03R_V14_SETTINGS[1])
    )
    assert ranked.ranking.item() != 0.0
    assert control.ranking.item() == 0.0
    assert ranked.robust_regression.item() == control.robust_regression.item()
    assert ranked.distributional.item() == control.distributional.item()
    assert control.component_weights == (0.0, 0.60, 0.40)


def test_v14_rank_loss_is_scale_invariant_in_dimensionless_units() -> None:
    base = torch.linspace(-0.2, 0.2, 12).repeat(3, 1)
    target = torch.flip(base, dims=(1,))
    log_scale = torch.zeros_like(base)
    first = m03r_v14_predictive_loss(_batch(base, log_scale, target)).ranking
    second = m03r_v14_predictive_loss(_batch(3.0 * base, log_scale, target)).ranking
    assert float(first) == pytest.approx(float(second), abs=1.0e-7)
