"""Rank-aligned cross-sectional objectives for the M03R-v10 study."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import (
    M03R_V10_HORIZONS,
    M03RV10PredictiveSetting,
)


class M03RV10RankObjectiveError(ValueError):
    """The v10 prediction, target, or rank geometry is malformed."""


@dataclass(frozen=True, slots=True)
class M03RV10PredictiveLoss:
    """Complete v10 loss with economic-unit mean and uncertainty outputs."""

    total: torch.Tensor
    ranking: torch.Tensor
    robust_regression: torch.Tensor
    distributional: torch.Tensor
    component_weights: tuple[float, float, float]


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(value, stable=True)
    sorted_value = value.index_select(0, order)
    unique, counts = torch.unique_consecutive(sorted_value, return_counts=True)
    del unique
    stops = counts.cumsum(0)
    starts = stops - counts
    average = (starts + stops - 1).to(value.dtype) / 2.0
    sorted_ranks = torch.repeat_interleave(average, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks.scatter_(0, order, sorted_ranks)
    return ranks


def rank_gaussian_scores(value: torch.Tensor) -> torch.Tensor:
    """Return deterministic average-tie normal scores for one cross section."""

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or not value.is_floating_point()
        or value.numel() < 2
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV10RankObjectiveError(
            "rank Gaussian input must be a finite floating cross section"
        )
    ranks = _average_ranks(value.detach())
    probability = (ranks + 0.5) / float(value.numel())
    score = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    return (score - score.mean()).detach()


def _rank_gaussian_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    supported = torch.zeros(
        (prediction.shape[0], prediction.shape[2]),
        dtype=torch.bool,
        device=prediction.device,
    )
    for date_index in range(prediction.shape[0]):
        date_rows: list[torch.Tensor] = []
        for horizon_index in range(prediction.shape[2]):
            mask = valid[date_index, :, horizon_index]
            if int(mask.sum()) < 2:
                date_rows.append(prediction.new_zeros(()))
                continue
            supported[date_index, horizon_index] = True
            predicted = prediction[date_index, mask, horizon_index]
            realized = target[date_index, mask, horizon_index]
            predicted_centered = predicted - predicted.mean()
            predicted_z = predicted_centered / torch.sqrt(
                predicted_centered.square().mean() + 1.0e-8
            )
            target_score = rank_gaussian_scores(realized)
            target_z = target_score / torch.sqrt(
                target_score.square().mean().clamp_min(1.0e-12)
            )
            date_rows.append(1.0 - (predicted_z * target_z).mean())
        rows.append(torch.stack(date_rows))
    return torch.stack(rows), supported


def _standardized_listwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = prediction.new_tensor(
        tuple(0.02 * math.sqrt(horizon) for horizon in M03R_V10_HORIZONS)
    ).view(1, 1, -1)
    predicted = torch.where(
        valid,
        prediction / scales,
        torch.full_like(prediction, -torch.inf),
    )
    realized = torch.where(
        valid,
        target.detach() / scales,
        torch.full_like(target, -torch.inf),
    )
    supported = valid.sum(dim=1) >= 2
    loss = -torch.where(
        valid,
        torch.softmax(realized, dim=1) * torch.log_softmax(predicted, dim=1),
        torch.zeros_like(prediction),
    ).sum(dim=1)
    return loss, supported


def m03r_v10_cross_sectional_ranking_loss(
    predicted_mean: torch.Tensor,
    target_log_return: torch.Tensor,
    valid: torch.Tensor,
    setting: M03RV10PredictiveSetting,
) -> torch.Tensor:
    """Apply one frozen v10 rank geometry with date-balanced horizon weights."""

    if (
        not isinstance(predicted_mean, torch.Tensor)
        or predicted_mean.ndim != 3
        or predicted_mean.shape[-1] != len(M03R_V10_HORIZONS)
        or not predicted_mean.is_floating_point()
        or not bool(torch.isfinite(predicted_mean).all())
        or not isinstance(target_log_return, torch.Tensor)
        or target_log_return.shape != predicted_mean.shape
        or target_log_return.dtype != predicted_mean.dtype
        or target_log_return.device != predicted_mean.device
        or not bool(torch.isfinite(target_log_return).all())
        or not isinstance(valid, torch.Tensor)
        or valid.shape != predicted_mean.shape
        or valid.dtype != torch.bool
        or valid.device != predicted_mean.device
    ):
        raise M03RV10RankObjectiveError("v10 rank tensors are not aligned")
    setting.__post_init__()
    if setting.ranking_objective == "rank-gaussian-correlation":
        per_date_horizon, supported = _rank_gaussian_correlation_loss(
            predicted_mean,
            target_log_return,
            valid,
        )
    else:
        per_date_horizon, supported = _standardized_listwise_loss(
            predicted_mean,
            target_log_return,
            valid,
        )
    counts = supported.sum(dim=0)
    active = counts > 0
    weights = predicted_mean.new_tensor(setting.horizon_loss_weights)
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    if float(active_weights.sum()) <= 0.0:
        raise M03RV10RankObjectiveError("v10 rank objective has no supported horizon")
    per_horizon = torch.where(
        supported,
        per_date_horizon,
        torch.zeros_like(per_date_horizon),
    ).sum(dim=0) / counts.clamp_min(1).to(predicted_mean.dtype)
    result = (per_horizon * active_weights).sum() / active_weights.sum()
    if not bool(torch.isfinite(result)):
        raise M03RV10RankObjectiveError("v10 rank objective is non-finite")
    return result


def _date_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    return torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / count


def _weighted_horizon_mean(
    values: torch.Tensor,
    supported: torch.Tensor,
    setting: M03RV10PredictiveSetting,
) -> torch.Tensor:
    counts = supported.sum(dim=0)
    active = counts > 0
    per_horizon = torch.where(
        supported,
        values,
        torch.zeros_like(values),
    ).sum(dim=0) / counts.clamp_min(1).to(values.dtype)
    weights = values.new_tensor(setting.horizon_loss_weights)
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    if float(active_weights.sum()) <= 0.0:
        raise M03RV10RankObjectiveError("v10 predictive loss has no support")
    return (per_horizon * active_weights).sum() / active_weights.sum()


def m03r_v10_predictive_loss(
    predicted_mean: torch.Tensor,
    predicted_log_scale: torch.Tensor,
    target_log_return: torch.Tensor,
    valid: torch.Tensor,
    setting: M03RV10PredictiveSetting,
) -> M03RV10PredictiveLoss:
    """Combine rank geometry with economic-unit robust/distributional terms."""

    ranking = m03r_v10_cross_sectional_ranking_loss(
        predicted_mean,
        target_log_return,
        valid,
        setting,
    )
    if (
        not isinstance(predicted_log_scale, torch.Tensor)
        or predicted_log_scale.shape != predicted_mean.shape
        or predicted_log_scale.dtype != predicted_mean.dtype
        or predicted_log_scale.device != predicted_mean.device
        or not bool(torch.isfinite(predicted_log_scale).all())
    ):
        raise M03RV10RankObjectiveError("v10 scale tensor is not aligned")
    supported = valid.sum(dim=1) >= 2
    horizon_scale = predicted_mean.new_tensor(
        tuple(0.02 * math.sqrt(horizon) for horizon in M03R_V10_HORIZONS)
    ).view(1, 1, -1)
    standardized_error = (predicted_mean - target_log_return.detach()) / horizon_scale
    robust_points = functional.huber_loss(
        standardized_error,
        torch.zeros_like(standardized_error),
        delta=1.0,
        reduction="none",
    )
    robust = _weighted_horizon_mean(
        _date_mean(robust_points, valid),
        supported,
        setting,
    )
    log_scale = predicted_log_scale.clamp(-8.0, 2.0)
    distributional_points = (
        0.5
        * (predicted_mean - target_log_return.detach()).square()
        * torch.exp(-2.0 * log_scale)
        + log_scale
    )
    distributional = _weighted_horizon_mean(
        _date_mean(distributional_points, valid),
        supported,
        setting,
    )
    component = setting.component_weights
    total = (
        component[0] * ranking + component[1] * robust + component[2] * distributional
    )
    if not bool(torch.isfinite(total)):
        raise M03RV10RankObjectiveError("v10 predictive objective is non-finite")
    return M03RV10PredictiveLoss(
        total=total,
        ranking=ranking,
        robust_regression=robust,
        distributional=distributional,
        component_weights=component,
    )


__all__ = [
    "M03RV10RankObjectiveError",
    "M03RV10PredictiveLoss",
    "m03r_v10_cross_sectional_ranking_loss",
    "m03r_v10_predictive_loss",
    "rank_gaussian_scores",
]
