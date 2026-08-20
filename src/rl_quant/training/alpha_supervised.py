"""Date-balanced supervised objective for executable alpha distributions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from rl_quant.models.alpha_hierarchical import AlphaDistribution


class AlphaObjectiveError(ValueError):
    """The alpha prediction batch or objective configuration is invalid."""


@dataclass(frozen=True, slots=True)
class AlphaObjectiveConfig:
    huber_weight: float = 1.0
    rank_weight: float = 0.2
    quantile_weight: float = 0.25
    calibration_weight: float = 0.1
    residual_ssl_weight: float = 0.03
    huber_delta: float = 1.0
    maximum_pairs_per_date_horizon: int = 128

    def validate(self) -> None:
        for name in (
            "huber_weight",
            "rank_weight",
            "quantile_weight",
            "calibration_weight",
            "residual_ssl_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise AlphaObjectiveError(f"{name} must be finite and nonnegative")
        if self.huber_weight <= 0.0:
            raise AlphaObjectiveError("the robust mean loss must remain primary")
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0.0:
            raise AlphaObjectiveError("Huber delta must be finite and positive")
        if (
            isinstance(self.maximum_pairs_per_date_horizon, bool)
            or not isinstance(self.maximum_pairs_per_date_horizon, int)
            or self.maximum_pairs_per_date_horizon <= 0
        ):
            raise AlphaObjectiveError("maximum pair count must be positive")


@dataclass(frozen=True, slots=True)
class AlphaSupervisedBatch:
    distribution: AlphaDistribution
    target: torch.Tensor
    valid: torch.Tensor
    executable_score: torch.Tensor
    ssl_loss: torch.Tensor | None = None

    def validate(self) -> None:
        reference = self.target
        tensors = (
            self.distribution.mean,
            self.distribution.downside_quantile,
            self.distribution.median,
            self.distribution.upside_quantile,
            self.distribution.scale,
            self.executable_score,
        )
        if (
            not isinstance(reference, torch.Tensor)
            or reference.ndim != 3
            or not reference.is_floating_point()
            or not isinstance(self.valid, torch.Tensor)
            or self.valid.dtype != torch.bool
            or tuple(self.valid.shape) != tuple(reference.shape)
            or self.valid.device != reference.device
            or any(
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(reference.shape)
                or value.dtype != reference.dtype
                or value.device != reference.device
                for value in tensors
            )
            or bool((self.valid.sum(dim=1) < 2).any())
            or not bool(torch.isfinite(reference[self.valid]).all())
            or any(not bool(torch.isfinite(value[self.valid]).all()) for value in tensors)
            or bool((self.distribution.scale[self.valid] <= 0.0).any())
            or bool(
                (
                    self.distribution.downside_quantile[self.valid]
                    > self.distribution.median[self.valid]
                ).any()
            )
            or bool(
                (
                    self.distribution.median[self.valid]
                    > self.distribution.upside_quantile[self.valid]
                ).any()
            )
        ):
            raise AlphaObjectiveError("alpha supervised batch is malformed")
        if self.ssl_loss is not None and (
            self.ssl_loss.ndim != 0 or not bool(torch.isfinite(self.ssl_loss))
        ):
            raise AlphaObjectiveError("residual self-supervised loss is malformed")


@dataclass(frozen=True, slots=True)
class AlphaObjectiveLoss:
    total: torch.Tensor
    robust_mean: torch.Tensor
    rank: torch.Tensor
    quantile: torch.Tensor
    calibration: torch.Tensor
    residual_ssl: torch.Tensor


def _date_balanced_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average assets inside date/horizon, then date and horizon equally."""

    counts = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    per_date_horizon = torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / counts
    return per_date_horizon.mean()


def _deterministic_tail_pair_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    maximum_pairs: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    dates, _assets, horizons = score.shape
    for date in range(dates):
        for horizon in range(horizons):
            selected = torch.nonzero(valid[date, :, horizon], as_tuple=False).flatten()
            observed_target = target[date, selected, horizon].detach()
            order = selected.index_select(0, torch.argsort(observed_target, stable=True))
            pair_count = min(maximum_pairs, order.numel() // 2)
            if pair_count == 0:
                continue
            lower = order[:pair_count]
            upper = order[-pair_count:].flip(0)
            target_difference = (
                target[date, upper, horizon] - target[date, lower, horizon]
            ).detach()
            non_tie = target_difference != 0.0
            if not bool(non_tie.any()):
                continue
            score_difference = score[date, upper, horizon] - score[date, lower, horizon]
            direction = target_difference.sign()
            losses.append(F.softplus(-direction[non_tie] * score_difference[non_tie]).mean())
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def alpha_supervised_loss(
    batch: AlphaSupervisedBatch,
    config: AlphaObjectiveConfig,
) -> AlphaObjectiveLoss:
    """Calculate robust, rank, quantile, and calibrated alpha losses."""

    batch.validate()
    config.validate()
    target = batch.target.detach()
    robust = _date_balanced_mean(
        F.huber_loss(
            batch.distribution.mean,
            target,
            reduction="none",
            delta=config.huber_delta,
        ),
        batch.valid,
    )
    rank = _deterministic_tail_pair_loss(
        batch.executable_score,
        target,
        batch.valid,
        maximum_pairs=config.maximum_pairs_per_date_horizon,
    )
    quantile_rows: list[torch.Tensor] = []
    for prediction, probability in (
        (batch.distribution.downside_quantile, 0.10),
        (batch.distribution.median, 0.50),
        (batch.distribution.upside_quantile, 0.90),
    ):
        error = target - prediction
        pinball = torch.maximum(probability * error, (probability - 1.0) * error)
        quantile_rows.append(_date_balanced_mean(pinball, batch.valid))
    quantile = torch.stack(quantile_rows).mean()
    absolute_error = (target - batch.distribution.mean).abs()
    calibration = _date_balanced_mean(
        absolute_error / batch.distribution.scale
        + torch.log(batch.distribution.scale),
        batch.valid,
    )
    residual_ssl = (
        batch.ssl_loss
        if batch.ssl_loss is not None
        else batch.distribution.mean.sum() * 0.0
    )
    total = (
        config.huber_weight * robust
        + config.rank_weight * rank
        + config.quantile_weight * quantile
        + config.calibration_weight * calibration
        + config.residual_ssl_weight * residual_ssl
    )
    if not bool(torch.isfinite(total)):
        raise AlphaObjectiveError("alpha supervised loss is nonfinite")
    return AlphaObjectiveLoss(
        total=total,
        robust_mean=robust,
        rank=rank,
        quantile=quantile,
        calibration=calibration,
        residual_ssl=residual_ssl,
    )


__all__ = [
    "AlphaObjectiveConfig",
    "AlphaObjectiveError",
    "AlphaObjectiveLoss",
    "AlphaSupervisedBatch",
    "alpha_supervised_loss",
]
