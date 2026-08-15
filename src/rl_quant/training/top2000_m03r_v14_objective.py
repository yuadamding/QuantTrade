"""Direct h3 rank/economic objective for M03R-v14."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_SELECTED_HORIZON_SESSIONS,
    M03RV14PredictiveSetting,
)
from rl_quant.training.top2000_m03r_v10_rank_objective import rank_gaussian_scores

M03R_V14_RANK_DIMENSIONLESS_RMS_FLOOR = 0.05


class M03RV14ObjectiveError(ValueError):
    """The v14 direct h3 objective inputs drifted."""


@dataclass(frozen=True, slots=True)
class M03RV14PredictiveBatch:
    predicted_mean: torch.Tensor
    predicted_log_scale: torch.Tensor
    target_log_return: torch.Tensor
    valid: torch.Tensor
    setting: M03RV14PredictiveSetting

    def validate(self) -> None:
        self.setting.__post_init__()
        if (
            not isinstance(self.predicted_mean, torch.Tensor)
            or self.predicted_mean.ndim != 2
            or not self.predicted_mean.is_floating_point()
            or not bool(torch.isfinite(self.predicted_mean).all())
            or not isinstance(self.predicted_log_scale, torch.Tensor)
            or tuple(self.predicted_log_scale.shape)
            != tuple(self.predicted_mean.shape)
            or self.predicted_log_scale.dtype != self.predicted_mean.dtype
            or self.predicted_log_scale.device != self.predicted_mean.device
            or not bool(torch.isfinite(self.predicted_log_scale).all())
            or not isinstance(self.target_log_return, torch.Tensor)
            or tuple(self.target_log_return.shape) != tuple(self.predicted_mean.shape)
            or self.target_log_return.dtype != self.predicted_mean.dtype
            or self.target_log_return.device != self.predicted_mean.device
            or not bool(torch.isfinite(self.target_log_return).all())
            or not isinstance(self.valid, torch.Tensor)
            or tuple(self.valid.shape) != tuple(self.predicted_mean.shape)
            or self.valid.dtype != torch.bool
            or self.valid.device != self.predicted_mean.device
            or not bool((self.valid.sum(dim=1) >= 2).any())
        ):
            raise M03RV14ObjectiveError("v14 predictive batch drifted")


@dataclass(frozen=True, slots=True)
class M03RV14PredictiveLoss:
    total: torch.Tensor
    ranking: torch.Tensor
    robust_regression: torch.Tensor
    distributional: torch.Tensor
    component_weights: tuple[float, float, float]


def _date_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    return torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / count


def _rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    setting: M03RV14PredictiveSetting,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for date_index in range(prediction.shape[0]):
        mask = valid[date_index]
        if setting.ranking_objective == "none" or int(mask.sum()) < 2:
            rows.append(prediction.new_zeros(()))
            continue
        horizon_scale = 0.02 * math.sqrt(M03R_V14_SELECTED_HORIZON_SESSIONS)
        selected = prediction[date_index, mask] / horizon_scale
        centered = selected - selected.mean()
        prediction_scale = (
            torch.sqrt(centered.square().mean())
            .detach()
            .clamp_min(M03R_V14_RANK_DIMENSIONLESS_RMS_FLOOR)
        )
        prediction_z = centered / prediction_scale
        target_score = rank_gaussian_scores(target[date_index, mask])
        target_z = target_score / torch.sqrt(
            target_score.square().mean().clamp_min(1.0e-12)
        )
        rows.append(1.0 - (prediction_z * target_z).mean())
    return torch.stack(rows).mean()


def m03r_v14_predictive_loss(
    batch: M03RV14PredictiveBatch,
) -> M03RV14PredictiveLoss:
    batch.validate()
    mean = batch.predicted_mean
    log_scale = batch.predicted_log_scale.clamp(-8.0, 2.0)
    target = batch.target_log_return.detach()
    valid = batch.valid
    ranking = _rank_loss(mean, target, valid, batch.setting)
    horizon_scale = 0.02 * math.sqrt(M03R_V14_SELECTED_HORIZON_SESSIONS)
    robust = _date_mean(
        functional.huber_loss(
            mean / horizon_scale,
            target / horizon_scale,
            reduction="none",
            delta=1.0,
        ),
        valid,
    ).mean()
    distributional = _date_mean(
        0.5
        * (
            torch.exp(-2.0 * log_scale) * (target - mean.detach()).square()
            + 2.0 * log_scale
        ),
        valid,
    ).mean()
    weights = batch.setting.component_weights
    total = weights[0] * ranking + weights[1] * robust + weights[2] * distributional
    if not bool(torch.isfinite(total)):
        raise M03RV14ObjectiveError("v14 predictive loss is non-finite")
    return M03RV14PredictiveLoss(
        total=total,
        ranking=ranking,
        robust_regression=robust,
        distributional=distributional,
        component_weights=weights,
    )


__all__ = [
    "M03R_V14_RANK_DIMENSIONLESS_RMS_FLOOR",
    "M03RV14ObjectiveError",
    "M03RV14PredictiveBatch",
    "M03RV14PredictiveLoss",
    "m03r_v14_predictive_loss",
]
