"""Dimensionless executable selection loss for corrected M03R-v16."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03RV16PredictiveSetting,
)
from rl_quant.training.top2000_m03r_v16_numerical import (
    M03RV16NumericalTrainingError,
)


class M03RV16ObjectiveError(ValueError):
    """The V16 selection objective inputs drifted."""


def m03r_v16_selection_target_scale(setting: M03RV16PredictiveSetting) -> float:
    """Return the immutable economic-unit scale bound to the setting."""

    setting.__post_init__()
    return setting.selection_target_scale


@dataclass(frozen=True, slots=True)
class M03RV16PredictiveBatch:
    executable_selection_score_z: torch.Tensor
    selection_target_z: torch.Tensor
    selection_target_economic: torch.Tensor
    selection_valid: torch.Tensor
    setting: M03RV16PredictiveSetting

    def validate(self) -> None:
        self.setting.__post_init__()
        reference = self.executable_selection_score_z
        scale = self.setting.selection_target_scale
        for name, value in (
            ("executable score", reference),
            ("standardized target", self.selection_target_z),
            ("economic target", self.selection_target_economic),
        ):
            if isinstance(value, torch.Tensor) and not bool(
                torch.isfinite(value).all()
            ):
                raise M03RV16NumericalTrainingError(
                    f"V16 {name} is non-finite"
                )
        if not math.isfinite(scale):
            raise M03RV16NumericalTrainingError(
                "V16 selection target scale is non-finite"
            )
        if (
            not isinstance(reference, torch.Tensor)
            or reference.ndim != 2
            or not reference.is_floating_point()
            or any(
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(reference.shape)
                or value.dtype != reference.dtype
                or value.device != reference.device
                for value in (
                    self.selection_target_z,
                    self.selection_target_economic,
                )
            )
            or not isinstance(self.selection_valid, torch.Tensor)
            or tuple(self.selection_valid.shape) != tuple(reference.shape)
            or self.selection_valid.dtype != torch.bool
            or self.selection_valid.device != reference.device
            or bool((self.selection_valid.sum(dim=1) < 2).any())
            or scale <= 0.0
            or not torch.allclose(
                self.selection_target_z,
                self.selection_target_economic / scale,
                rtol=2.0e-6,
                atol=2.0e-7,
            )
        ):
            raise M03RV16ObjectiveError("V16 predictive batch drifted")

    @property
    def executable_selection_mean(self) -> torch.Tensor:
        """Return the exact executable score in cumulative-return units."""

        self.validate()
        return self.executable_selection_score_z * self.setting.selection_target_scale

    @property
    def selection_target(self) -> torch.Tensor:
        """Compatibility alias for the economic-unit target."""

        return self.selection_target_economic


@dataclass(frozen=True, slots=True)
class M03RV16ScoreLoss:
    total: torch.Tensor
    selection_robust: torch.Tensor
    selection_loss_weight: float = 1.0


def _date_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    counts = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    return torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / counts


def m03r_v16_score_loss(batch: M03RV16PredictiveBatch) -> M03RV16ScoreLoss:
    """Train only the action-projected dimensionless selection score."""

    batch.validate()
    selection = _date_mean(
        functional.huber_loss(
            batch.executable_selection_score_z,
            batch.selection_target_z.detach(),
            reduction="none",
            delta=1.0,
        ),
        batch.selection_valid,
    ).mean()
    total = batch.setting.selection_loss_weight * selection
    if not bool(torch.isfinite(total)):
        raise M03RV16NumericalTrainingError("V16 score loss is non-finite")
    return M03RV16ScoreLoss(total, selection)


__all__ = [
    "M03RV16ObjectiveError",
    "M03RV16PredictiveBatch",
    "M03RV16ScoreLoss",
    "m03r_v16_score_loss",
    "m03r_v16_selection_target_scale",
]
