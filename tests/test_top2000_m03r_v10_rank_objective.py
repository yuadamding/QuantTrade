from __future__ import annotations

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import M03R_V10_SETTINGS
from rl_quant.training.top2000_m03r_v10_rank_objective import (
    M03RV10RankObjectiveError,
    m03r_v10_cross_sectional_ranking_loss,
    m03r_v10_predictive_loss,
    rank_gaussian_scores,
)


def _panel() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_row = torch.tensor([-0.04, -0.01, 0.00, 0.02, 0.05], dtype=torch.float64)
    target = target_row.view(1, 5, 1).repeat(2, 1, 4)
    target[1] = target[1].flip(0)
    prediction = (0.7 * target).clone().requires_grad_(True)
    valid = torch.ones_like(target, dtype=torch.bool)
    return prediction, target, valid


def test_rank_gaussian_scores_use_average_ties_and_are_axis_symmetric() -> None:
    values = torch.tensor([3.0, 1.0, 1.0, 2.0], dtype=torch.float64)
    score = rank_gaussian_scores(values)
    assert score[1] == score[2]
    assert score[1] < score[3] < score[0]
    assert float(score.mean()) == pytest.approx(0.0, abs=1.0e-12)


def test_rank_gaussian_loss_rewards_global_order_and_backpropagates() -> None:
    prediction, target, valid = _panel()
    aligned = m03r_v10_cross_sectional_ranking_loss(
        prediction,
        target,
        valid,
        M03R_V10_SETTINGS[1],
    )
    reversed_loss = m03r_v10_cross_sectional_ranking_loss(
        -prediction,
        target,
        valid,
        M03R_V10_SETTINGS[1],
    )
    assert aligned < reversed_loss
    aligned.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.abs().sum()) > 0.0


def test_21_30_setting_ignores_5_and_63_rank_rows() -> None:
    prediction, target, valid = _panel()
    base = m03r_v10_cross_sectional_ranking_loss(
        prediction,
        target,
        valid,
        M03R_V10_SETTINGS[2],
    )
    mutated = prediction.detach().clone()
    mutated[..., 0] *= -100.0
    mutated[..., 3] *= -100.0
    observed = m03r_v10_cross_sectional_ranking_loss(
        mutated,
        target,
        valid,
        M03R_V10_SETTINGS[2],
    )
    torch.testing.assert_close(base, observed)


def test_full_loss_changes_only_rank_geometry_between_control_and_p1() -> None:
    prediction, target, valid = _panel()
    log_scale = torch.full_like(prediction, -3.0, requires_grad=True)
    control = m03r_v10_predictive_loss(
        prediction,
        log_scale,
        target,
        valid,
        M03R_V10_SETTINGS[0],
    )
    rank_gaussian = m03r_v10_predictive_loss(
        prediction,
        log_scale,
        target,
        valid,
        M03R_V10_SETTINGS[1],
    )
    torch.testing.assert_close(
        control.robust_regression,
        rank_gaussian.robust_regression,
    )
    torch.testing.assert_close(control.distributional, rank_gaussian.distributional)
    assert (
        control.component_weights
        == rank_gaussian.component_weights
        == (
            0.50,
            0.30,
            0.20,
        )
    )
    rank_gaussian.total.backward()
    assert prediction.grad is not None
    assert log_scale.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.isfinite(log_scale.grad).all()


def test_21_30_full_loss_ignores_outer_horizon_predictions_and_scales() -> None:
    prediction, target, valid = _panel()
    log_scale = torch.full_like(prediction, -3.0)
    base = m03r_v10_predictive_loss(
        prediction,
        log_scale,
        target,
        valid,
        M03R_V10_SETTINGS[2],
    ).total
    mutated_prediction = prediction.detach().clone()
    mutated_scale = log_scale.clone()
    mutated_prediction[..., 0] += 100.0
    mutated_prediction[..., 3] -= 100.0
    mutated_scale[..., 0] = 2.0
    mutated_scale[..., 3] = -8.0
    observed = m03r_v10_predictive_loss(
        mutated_prediction,
        mutated_scale,
        target,
        valid,
        M03R_V10_SETTINGS[2],
    ).total
    torch.testing.assert_close(base, observed)


def test_rank_objective_rejects_empty_or_misaligned_support() -> None:
    prediction, target, valid = _panel()
    with pytest.raises(M03RV10RankObjectiveError, match="no supported"):
        m03r_v10_cross_sectional_ranking_loss(
            prediction,
            target,
            torch.zeros_like(valid),
            M03R_V10_SETTINGS[1],
        )
    with pytest.raises(M03RV10RankObjectiveError, match="not aligned"):
        m03r_v10_cross_sectional_ranking_loss(
            prediction,
            target[:, :-1],
            valid,
            M03R_V10_SETTINGS[1],
        )
