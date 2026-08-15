from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_SETTINGS,
)
from rl_quant.training.top2000_m03r_v15_objective import (
    M03RV15PredictiveBatch,
    m03r_v15_predictive_loss,
)
from rl_quant.training.top2000_m03r_v15_policy import (
    M03RV15PolicyError,
    Top2000M03RV15PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v10_rank_objective import rank_gaussian_scores


def _batch(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    *,
    setting_index: int = 0,
) -> M03RV15PredictiveBatch:
    valid = torch.ones_like(mean, dtype=torch.bool)
    valid[:, 0] = False
    return M03RV15PredictiveBatch(
        predicted_mean=mean,
        predicted_log_scale=log_scale,
        target_log_return=target,
        valid=valid,
        setting=M03R_V15_SETTINGS[setting_index],
    )


def test_v15_policy_exposes_only_raw_prediction_fields() -> None:
    torch.manual_seed(17)
    policy = Top2000M03RV15PredictivePolicy(
        0,
        selected_horizon_sessions=3,
        token_dim=8,
        raw_stock_chunk=4,
        activation_checkpointing=False,
    )
    state = torch.randn((2, 6, 8))
    available = torch.ones((2, 6), dtype=torch.bool)
    output = policy.predictive_output(state, available)
    assert not hasattr(output, "rank_score")
    assert not hasattr(output, "execution_score")
    assert output.raw_mean.shape == (2, 6)
    assert output.economic_scale.shape == (2, 6)
    assert torch.equal(output.raw_mean[:, 0], torch.zeros(2))
    assert float(policy.economic_mean_head.weight.std()) < 0.01
    assert torch.equal(
        policy.economic_scale_head.weight,
        torch.zeros_like(policy.economic_scale_head.weight),
    )
    assert policy.v15_head_identity().selected_alpha_horizon == 3
    with pytest.raises(M03RV15PolicyError, match="selected horizon"):
        Top2000M03RV15PredictivePolicy(
            0,
            selected_horizon_sessions=21,
            token_dim=8,
            activation_checkpointing=False,
        )


def test_v15_scale_calibration_does_not_update_the_shared_encoder() -> None:
    policy = Top2000M03RV15PredictivePolicy(
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
    output.raw_log_scale.sum().backward()
    assert policy.economic_scale_head.bias.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in policy.named_parameters()
        if name.startswith("source_policy.")
    )


def test_v15_rank_loss_updates_the_same_mean_consumed_by_execution() -> None:
    mean = torch.randn((3, 12), requires_grad=True)
    log_scale = torch.zeros_like(mean, requires_grad=True)
    target = torch.randn_like(mean)
    loss = m03r_v15_predictive_loss(_batch(mean, log_scale, target))
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


def test_v15_no_rank_control_changes_only_the_objective_component() -> None:
    mean = torch.randn((3, 12), requires_grad=True)
    log_scale = torch.zeros_like(mean, requires_grad=True)
    target = torch.randn_like(mean)
    ranked = m03r_v15_predictive_loss(_batch(mean, log_scale, target))
    control = m03r_v15_predictive_loss(
        replace(_batch(mean, log_scale, target), setting=M03R_V15_SETTINGS[1])
    )
    assert ranked.ranking.item() != 0.0
    assert control.ranking.item() == 0.0
    assert ranked.robust_regression.item() == control.robust_regression.item()
    assert ranked.distributional.item() == control.distributional.item()
    assert control.component_weights == (0.0, 0.45, 0.30)
    ranked_robust_gradient = torch.autograd.grad(
        0.45 * ranked.robust_regression, mean, retain_graph=True
    )[0]
    control_robust_gradient = torch.autograd.grad(
        0.45 * control.robust_regression, mean, retain_graph=True
    )[0]
    ranked_scale_gradient = torch.autograd.grad(
        0.30 * ranked.distributional, log_scale, retain_graph=True
    )[0]
    control_scale_gradient = torch.autograd.grad(
        0.30 * control.distributional, log_scale, retain_graph=True
    )[0]
    assert torch.equal(ranked_robust_gradient, control_robust_gradient)
    assert torch.equal(ranked_scale_gradient, control_scale_gradient)


def test_v15_rank_loss_is_scale_invariant_in_dimensionless_units() -> None:
    base = torch.linspace(-0.2, 0.2, 12).repeat(3, 1)
    target = torch.flip(base, dims=(1,))
    log_scale = torch.zeros_like(base)
    first = m03r_v15_predictive_loss(_batch(base, log_scale, target)).ranking
    second = m03r_v15_predictive_loss(_batch(3.0 * base, log_scale, target)).ranking
    assert float(first) == pytest.approx(float(second), abs=1.0e-7)


def _perfect_rank_prediction(*, sign: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.linspace(-0.03, 0.03, 12).repeat(2, 1)
    scores = torch.stack(
        tuple(rank_gaussian_scores(row) for row in target[:, 1:])
    )
    scores = scores / torch.sqrt(scores.square().mean(dim=1, keepdim=True))
    prediction = torch.cat(
        (
            torch.zeros((2, 1)),
            sign * 0.02 * math.sqrt(3.0) * scores,
        ),
        dim=1,
    ).requires_grad_(True)
    return prediction, target


def test_v15_perfect_rank_alignment_has_zero_radial_gradient() -> None:
    prediction, target = _perfect_rank_prediction()
    loss = m03r_v15_predictive_loss(
        _batch(prediction, torch.zeros_like(prediction), target)
    ).ranking
    gradient = torch.autograd.grad(loss, prediction)[0]
    centered = prediction[:, 1:] - prediction[:, 1:].mean(dim=1, keepdim=True)
    radial = (gradient[:, 1:] * centered).sum()
    assert float(loss) == pytest.approx(0.0, abs=2.0e-6)
    assert float(torch.linalg.vector_norm(gradient)) == pytest.approx(0.0, abs=2.0e-6)
    assert float(radial) == pytest.approx(0.0, abs=2.0e-7)


def test_v15_negative_rank_alignment_points_toward_sign_correction() -> None:
    prediction, target = _perfect_rank_prediction(sign=-1.0)
    loss = m03r_v15_predictive_loss(
        _batch(prediction, torch.zeros_like(prediction), target)
    ).ranking
    gradient = torch.autograd.grad(loss, prediction)[0]
    target_scores = torch.stack(
        tuple(rank_gaussian_scores(row) for row in target[:, 1:])
    )
    target_direction = torch.cat((torch.zeros((2, 1)), target_scores), dim=1)
    assert float(loss) == pytest.approx(2.0, abs=2.0e-6)
    assert float((gradient * target_direction).sum()) < 0.0


def test_v15_rank_floor_preserves_finite_anti_collapse_gradient() -> None:
    prediction = (1.0e-8 * torch.randn((2, 12))).requires_grad_(True)
    target = torch.linspace(-0.03, 0.03, 12).repeat(2, 1)
    loss = m03r_v15_predictive_loss(
        _batch(prediction, torch.zeros_like(prediction), target)
    ).ranking
    gradient = torch.autograd.grad(loss, prediction)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0
