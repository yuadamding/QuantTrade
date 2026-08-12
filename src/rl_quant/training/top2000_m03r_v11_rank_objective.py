"""V11 rank objective bound to factor-qualified corrected batches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_HORIZONS,
    M03RV11PredictiveSetting,
)
from rl_quant.training.top2000_m03r_v10_rank_objective import rank_gaussian_scores
from rl_quant.training.top2000_m03r_v11_pretraining_runtime import (
    M03RV11AlphaPretrainingBatch,
)


class M03RV11RankObjectiveError(ValueError):
    """The v11 rank geometry or qualified batch drifted."""


@dataclass(frozen=True, slots=True)
class M03RV11PredictiveLoss:
    total: torch.Tensor
    ranking: torch.Tensor
    robust_regression: torch.Tensor
    distributional: torch.Tensor
    component_weights: tuple[float, float, float]


def _rank_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    setting: M03RV11PredictiveSetting,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    supported = torch.zeros(
        (predicted.shape[0], predicted.shape[2]),
        dtype=torch.bool,
        device=predicted.device,
    )
    scales = predicted.new_tensor(
        tuple(0.02 * math.sqrt(horizon) for horizon in M03R_V11_HORIZONS)
    )
    for date in range(predicted.shape[0]):
        date_rows: list[torch.Tensor] = []
        for horizon in range(predicted.shape[2]):
            mask = valid[date, :, horizon]
            if int(mask.sum()) < 2:
                date_rows.append(predicted.new_zeros(()))
                continue
            supported[date, horizon] = True
            prediction = predicted[date, mask, horizon]
            realized = target[date, mask, horizon]
            if setting.ranking_objective == "rank-gaussian-correlation":
                centered = prediction - prediction.mean()
                prediction_z = centered / torch.sqrt(centered.square().mean() + 1.0e-8)
                target_score = rank_gaussian_scores(realized)
                target_z = target_score / torch.sqrt(
                    target_score.square().mean().clamp_min(1.0e-12)
                )
                date_rows.append(1.0 - (prediction_z * target_z).mean())
            else:
                prediction_scaled = prediction / scales[horizon]
                realized_scaled = realized.detach() / scales[horizon]
                date_rows.append(
                    -(
                        torch.softmax(realized_scaled, dim=0)
                        * torch.log_softmax(prediction_scaled, dim=0)
                    ).sum()
                )
        rows.append(torch.stack(date_rows))
    return torch.stack(rows), supported


def _date_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    return torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / count


def _weighted_horizon_mean(
    values: torch.Tensor,
    supported: torch.Tensor,
    setting: M03RV11PredictiveSetting,
) -> torch.Tensor:
    counts = supported.sum(dim=0)
    active = counts > 0
    per_horizon = torch.where(supported, values, torch.zeros_like(values)).sum(dim=0)
    per_horizon = per_horizon / counts.clamp_min(1).to(values.dtype)
    weights = values.new_tensor(setting.horizon_loss_weights)
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    if float(active_weights.sum()) <= 0.0:
        raise M03RV11RankObjectiveError("v11 loss has no supported horizon")
    return (per_horizon * active_weights).sum() / active_weights.sum()


def m03r_v11_predictive_loss(
    batch: M03RV11AlphaPretrainingBatch,
) -> M03RV11PredictiveLoss:
    batch.validate()
    base = batch.corrected_batch
    mean = base.predicted_mean
    log_scale = base.predicted_log_scale
    target = base.target_log_return
    valid = base.valid
    rank_rows, supported = _rank_loss(mean, target, valid, batch.setting)
    ranking = _weighted_horizon_mean(rank_rows, supported, batch.setting)
    horizon_scale = mean.new_tensor(
        tuple(0.02 * math.sqrt(value) for value in M03R_V11_HORIZONS)
    ).view(1, 1, -1)
    robust_rows = _date_mean(
        functional.huber_loss(
            mean / horizon_scale,
            target / horizon_scale,
            reduction="none",
            delta=1.0,
        ),
        valid,
    )
    distribution_rows = _date_mean(
        0.5
        * (torch.exp(-2.0 * log_scale) * (target - mean).square() + 2.0 * log_scale),
        valid,
    )
    robust = _weighted_horizon_mean(robust_rows, supported, batch.setting)
    distributional = _weighted_horizon_mean(distribution_rows, supported, batch.setting)
    weights = batch.setting.component_weights
    total = weights[0] * ranking + weights[1] * robust + weights[2] * distributional
    if not bool(torch.isfinite(total)):
        raise M03RV11RankObjectiveError("v11 predictive loss is non-finite")
    return M03RV11PredictiveLoss(
        total=total,
        ranking=ranking,
        robust_regression=robust,
        distributional=distributional,
        component_weights=weights,
    )


__all__ = [
    "M03RV11PredictiveLoss",
    "M03RV11RankObjectiveError",
    "m03r_v11_predictive_loss",
]
