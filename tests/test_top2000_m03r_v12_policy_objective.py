from __future__ import annotations

from dataclasses import replace

import torch
import pytest

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import M03R_V12_SETTINGS
from rl_quant.training.top2000_m03r_v12_objective import (
    M03R_V12_RANK_SCORE_STANDARD_DEVIATION_FLOOR,
    M03RV12PredictiveBatch,
    m03r_v12_predictive_loss,
)
from rl_quant.training.top2000_m03r_v12_policy import (
    M03RV12PolicyError,
    Top2000M03RV12PredictivePolicy,
)


def _batch(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
) -> M03RV12PredictiveBatch:
    valid = torch.ones_like(mean, dtype=torch.bool)
    valid[:, 0] = False
    count = mean.shape[0] * mean.shape[2]
    return M03RV12PredictiveBatch(
        predicted_mean=mean,
        predicted_log_scale=log_scale,
        predicted_rank_score=torch.zeros_like(mean),
        target_log_return=target,
        valid=valid,
        origin_indices=torch.tensor([1, 2, 3]),
        split="training",
        target_mode="factor-residual",
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256="c" * 64,
        setting=M03R_V12_SETTINGS[1],
        residual_operator_receipt_sha256=tuple("d" * 64 for _ in range(count)),
        available_risky_asset_count=tuple(mean.shape[1] - 1 for _ in range(count)),
        factor_qualified_risky_asset_count=tuple(
            mean.shape[1] - 1 for _ in range(count)
        ),
        effective_design_rank=tuple(5 for _ in range(count)),
        weighted_residual_degrees_of_freedom=tuple(4 for _ in range(count)),
    )


def test_v12_policy_emits_distinct_rank_and_economic_tensors() -> None:
    torch.manual_seed(7)
    policy = Top2000M03RV12PredictivePolicy(
        1,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    state = torch.randn((2, 7, 16))
    available = torch.ones((2, 7), dtype=torch.bool)
    output = policy.predictive_output(state, available)
    output.validate()
    assert output.rank_score_by_horizon.shape == (2, 7, 5)
    assert output.economic_distribution.selected_horizon_sessions == 3
    assert torch.equal(
        output.economic_distribution.selected_mean,
        output.economic_distribution.mean_by_horizon[..., 0],
    )
    assert output.rank_score_by_horizon.data_ptr() != (
        output.economic_distribution.mean_by_horizon.data_ptr()
    )
    identity = policy.v12_head_identity()
    assert identity.rank_score_head_state_sha256 not in {
        identity.economic_mean_head_state_sha256,
        identity.economic_scale_head_state_sha256,
    }


def test_v12_selected_horizon_is_a_required_fail_closed_model_argument() -> None:
    with pytest.raises(M03RV12PolicyError, match="selected horizon"):
        Top2000M03RV12PredictivePolicy(
            0,
            selected_horizon_sessions=21,
            token_dim=16,
            raw_stock_chunk=8,
            activation_checkpointing=False,
        )


def test_rank_gradient_does_not_enter_economic_heads() -> None:
    mean = torch.randn((3, 12, 5), requires_grad=True)
    log_scale = torch.zeros_like(mean, requires_grad=True)
    rank = torch.full_like(mean, 1.0e-8, requires_grad=True)
    target = torch.randn_like(mean)
    batch = replace(_batch(mean, log_scale, target), predicted_rank_score=rank)
    loss = m03r_v12_predictive_loss(batch)
    rank_gradient = torch.autograd.grad(loss.ranking, rank, retain_graph=True)[0]
    assert torch.isfinite(rank_gradient).all()
    assert rank_gradient.norm().item() < 10.0
    assert M03R_V12_RANK_SCORE_STANDARD_DEVIATION_FLOOR == 0.25
    assert torch.autograd.grad(
        loss.ranking, (mean, log_scale), allow_unused=True, retain_graph=True
    ) == (None, None)
    economic_gradients = torch.autograd.grad(
        loss.economic_total, (mean, log_scale), retain_graph=True
    )
    assert all(value is not None for value in economic_gradients)
    assert torch.autograd.grad(loss.economic_total, rank, allow_unused=True)[0] is None


def test_no_rank_control_has_zero_rank_gradient_and_economic_gradients() -> None:
    mean = torch.randn((3, 12, 5), requires_grad=True)
    log_scale = torch.zeros_like(mean, requires_grad=True)
    rank = torch.randn_like(mean, requires_grad=True)
    target = torch.randn_like(mean)
    loss = m03r_v12_predictive_loss(
        replace(
            _batch(mean, log_scale, target),
            predicted_rank_score=rank,
            setting=M03R_V12_SETTINGS[2],
        )
    )
    assert loss.ranking.item() == 0.0
    assert torch.autograd.grad(loss.total, rank, allow_unused=True)[0] is None
    loss.total.backward()
    assert mean.grad is not None and mean.grad.abs().sum().item() > 0.0
    assert log_scale.grad is not None and log_scale.grad.abs().sum().item() > 0.0
