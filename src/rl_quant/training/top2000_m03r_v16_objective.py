"""Paired selection/timing score and separate uncertainty losses for V16."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_SURVIVAL_WEIGHTS,
    M03R_V16_TIMING_HORIZON_SESSIONS,
    M03RV16PredictiveSetting,
)


class M03RV16ObjectiveError(ValueError):
    """The V16 selection/timing objective inputs drifted."""


def m03r_v16_selection_target_scale(setting: M03RV16PredictiveSetting) -> float:
    setting.__post_init__()
    if setting.selection_target == "h21-cumulative-factor-residual":
        return 0.02 * math.sqrt(21.0)
    if setting.selection_target == "h30-cumulative-factor-residual":
        return 0.02 * math.sqrt(30.0)
    return 0.02 * math.sqrt(math.fsum(w * w for w in M03R_V16_SURVIVAL_WEIGHTS))


@dataclass(frozen=True, slots=True)
class M03RV16PredictiveBatch:
    executable_selection_mean: torch.Tensor
    selection_log_scale: torch.Tensor
    selection_target: torch.Tensor
    selection_valid: torch.Tensor
    executable_timing_mean: torch.Tensor
    timing_log_scale: torch.Tensor
    timing_target: torch.Tensor
    timing_valid: torch.Tensor
    setting: M03RV16PredictiveSetting

    def validate(self) -> None:
        self.setting.__post_init__()
        reference = self.executable_selection_mean
        float_tensors = (
            reference,
            self.selection_log_scale,
            self.selection_target,
            self.executable_timing_mean,
            self.timing_log_scale,
            self.timing_target,
        )
        masks = (self.selection_valid, self.timing_valid)
        if (
            not isinstance(reference, torch.Tensor)
            or reference.ndim != 2
            or not reference.is_floating_point()
            or any(
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(reference.shape)
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not bool(torch.isfinite(value).all())
                for value in float_tensors
            )
            or any(
                not isinstance(mask, torch.Tensor)
                or tuple(mask.shape) != tuple(reference.shape)
                or mask.dtype != torch.bool
                or mask.device != reference.device
                or bool((mask.sum(dim=1) < 2).any())
                for mask in masks
            )
        ):
            raise M03RV16ObjectiveError("V16 predictive batch drifted")


@dataclass(frozen=True, slots=True)
class M03RV16ScoreLoss:
    total: torch.Tensor
    selection_robust: torch.Tensor
    timing_robust: torch.Tensor
    component_weights: tuple[float, float]


@dataclass(frozen=True, slots=True)
class M03RV16ScaleCalibrationLoss:
    total: torch.Tensor
    selection_distributional: torch.Tensor
    timing_distributional: torch.Tensor


def _date_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    counts = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    return torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / counts


def _robust(
    mean: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return _date_mean(
        functional.huber_loss(
            mean / scale,
            target.detach() / scale,
            reduction="none",
            delta=1.0,
        ),
        valid,
    ).mean()


def m03r_v16_score_loss(batch: M03RV16PredictiveBatch) -> M03RV16ScoreLoss:
    """Train only the executable selection and timing means."""

    batch.validate()
    selection = _robust(
        batch.executable_selection_mean,
        batch.selection_target,
        batch.selection_valid,
        m03r_v16_selection_target_scale(batch.setting),
    )
    timing = _robust(
        batch.executable_timing_mean,
        batch.timing_target,
        batch.timing_valid,
        0.02 * math.sqrt(M03R_V16_TIMING_HORIZON_SESSIONS),
    )
    weights = batch.setting.score_component_weights
    total = weights[0] * selection + weights[1] * timing
    if not bool(torch.isfinite(total)):
        raise M03RV16ObjectiveError("V16 score loss is non-finite")
    return M03RV16ScoreLoss(total, selection, timing, weights)


def _distributional(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    bounded = log_scale.clamp(-8.0, 2.0)
    values = 0.5 * (
        torch.exp(-2.0 * bounded) * (target.detach() - mean.detach()).square()
        + 2.0 * bounded
    )
    return _date_mean(values, valid).mean()


def m03r_v16_scale_calibration_loss(
    batch: M03RV16PredictiveBatch,
) -> M03RV16ScaleCalibrationLoss:
    """Fit scale heads only after the selected mean checkpoint is frozen."""

    batch.validate()
    selection = _distributional(
        batch.executable_selection_mean,
        batch.selection_log_scale,
        batch.selection_target,
        batch.selection_valid,
    )
    timing = _distributional(
        batch.executable_timing_mean,
        batch.timing_log_scale,
        batch.timing_target,
        batch.timing_valid,
    )
    total = 0.5 * (selection + timing)
    if not bool(torch.isfinite(total)):
        raise M03RV16ObjectiveError("V16 scale calibration loss is non-finite")
    return M03RV16ScaleCalibrationLoss(total, selection, timing)


__all__ = [
    "M03RV16ObjectiveError",
    "M03RV16PredictiveBatch",
    "M03RV16ScaleCalibrationLoss",
    "M03RV16ScoreLoss",
    "m03r_v16_scale_calibration_loss",
    "m03r_v16_score_loss",
    "m03r_v16_selection_target_scale",
]
