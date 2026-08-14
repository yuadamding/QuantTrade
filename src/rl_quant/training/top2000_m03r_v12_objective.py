"""Decoupled rank and economic-scale objective for M03R-v12."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_HORIZONS,
    M03R_V12_PROTOCOL_SHA256,
    M03RV12PredictiveSetting,
)
from rl_quant.training.top2000_m03r_v10_rank_objective import rank_gaussian_scores
from rl_quant.training.top2000_m03r_v11_residual_operator import (
    M03RV11ResidualOperator,
)

M03R_V12_ALPHA_BATCH_SCHEMA = "rl-quant.top2000-dev.m03r-v12-alpha-batch-v2"
M03R_V12_RANK_SCORE_STANDARD_DEVIATION_FLOOR = 0.25


class M03RV12ObjectiveError(ValueError):
    """The v12 rank/economic objective inputs drifted."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class M03RV12PredictiveBatch:
    predicted_mean: torch.Tensor
    predicted_log_scale: torch.Tensor
    predicted_rank_score: torch.Tensor
    target_log_return: torch.Tensor
    valid: torch.Tensor
    origin_indices: torch.Tensor
    split: Literal["training", "qualification"]
    fold_index: int
    split_start_inclusive: int
    split_stop_exclusive: int
    source_array_sha256: str
    asset_axis_sha256: str
    exposure_receipt_sha256: str
    setting: M03RV12PredictiveSetting
    residual_operator_receipt_sha256: tuple[str, ...]
    available_risky_asset_count: tuple[int, ...]
    factor_qualified_risky_asset_count: tuple[int, ...]
    effective_design_rank: tuple[int, ...]
    weighted_residual_degrees_of_freedom: tuple[int, ...]
    residual_operators: tuple[M03RV11ResidualOperator, ...] | None = None
    target_mode: Literal["factor-residual"] = "factor-residual"
    outer_score_accessed: bool = False
    lockbox_accessed: bool = False
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    schema: str = M03R_V12_ALPHA_BATCH_SCHEMA

    def validate(self) -> None:
        self.setting.__post_init__()
        mean = self.predicted_mean
        aligned = (
            self.predicted_log_scale,
            self.predicted_rank_score,
            self.target_log_return,
        )
        if (
            not isinstance(mean, torch.Tensor)
            or mean.ndim != 3
            or mean.shape[-1] != len(M03R_V12_HORIZONS)
            or not mean.is_floating_point()
            or not bool(torch.isfinite(mean).all())
            or any(
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(mean.shape)
                or value.dtype != mean.dtype
                or value.device != mean.device
                or not bool(torch.isfinite(value).all())
                for value in aligned
            )
            or self.predicted_rank_score.data_ptr() == mean.data_ptr()
            or not isinstance(self.valid, torch.Tensor)
            or tuple(self.valid.shape) != tuple(mean.shape)
            or self.valid.dtype != torch.bool
            or self.valid.device != mean.device
            or not isinstance(self.origin_indices, torch.Tensor)
            or tuple(self.origin_indices.shape) != (mean.shape[0],)
            or self.origin_indices.dtype != torch.long
            or self.origin_indices.device != mean.device
            or self.origin_indices.numel() == 0
            or bool((self.origin_indices[1:] <= self.origin_indices[:-1]).any())
        ):
            raise M03RV12ObjectiveError("v12 predictive tensor geometry drifted")
        if (
            self.split not in {"training", "qualification"}
            or self.fold_index not in range(6)
            or not 0 <= self.split_start_inclusive < self.split_stop_exclusive
            or int(self.origin_indices[0]) < self.split_start_inclusive
            or self.target_mode != "factor-residual"
            or self.outer_score_accessed
            or self.lockbox_accessed
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.schema != M03R_V12_ALPHA_BATCH_SCHEMA
        ):
            raise M03RV12ObjectiveError("v12 batch split or research identity drifted")
        horizons = torch.tensor(M03R_V12_HORIZONS, device=mean.device)
        stops = self.origin_indices.unsqueeze(-1) + horizons + 1
        if bool((stops[self.valid.any(dim=1)] > self.split_stop_exclusive).any()):
            raise M03RV12ObjectiveError("v12 targets cross their frozen split")
        if not bool((self.valid.sum(dim=1) >= 2).any()):
            raise M03RV12ObjectiveError("v12 batch has no supported cross section")
        if not all(
            _valid_digest(value)
            for value in (
                self.source_array_sha256,
                self.asset_axis_sha256,
                self.exposure_receipt_sha256,
            )
        ):
            raise M03RV12ObjectiveError("v12 batch identity digest is malformed")
        expected = mean.shape[0] * len(M03R_V12_HORIZONS)
        vectors = (
            self.residual_operator_receipt_sha256,
            self.available_risky_asset_count,
            self.factor_qualified_risky_asset_count,
            self.effective_design_rank,
            self.weighted_residual_degrees_of_freedom,
        )
        if (
            any(len(value) != expected for value in vectors)
            or any(
                not _valid_digest(value)
                for value in self.residual_operator_receipt_sha256
            )
            or any(
                qualified > available or qualified <= 0
                for qualified, available in zip(
                    self.factor_qualified_risky_asset_count,
                    self.available_risky_asset_count,
                    strict=True,
                )
            )
            or any(rank <= 0 for rank in self.effective_design_rank)
            or any(value <= 0 for value in self.weighted_residual_degrees_of_freedom)
            or (
                self.residual_operators is not None
                and (
                    len(self.residual_operators) != expected
                    or tuple(row.receipt_sha256 for row in self.residual_operators)
                    != self.residual_operator_receipt_sha256
                )
            )
        ):
            raise M03RV12ObjectiveError("v12 residual-operator evidence drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": self.schema,
                    "protocol_sha256": self.protocol_sha256,
                    "setting_sha256": self.setting.receipt_sha256,
                    "split": self.split,
                    "fold_index": self.fold_index,
                    "origin_indices": tuple(
                        int(value) for value in self.origin_indices
                    ),
                    "source_array_sha256": self.source_array_sha256,
                    "asset_axis_sha256": self.asset_axis_sha256,
                    "exposure_receipt_sha256": self.exposure_receipt_sha256,
                    "predicted_mean_sha256": _tensor_sha256(self.predicted_mean),
                    "predicted_log_scale_sha256": _tensor_sha256(
                        self.predicted_log_scale
                    ),
                    "predicted_rank_score_sha256": _tensor_sha256(
                        self.predicted_rank_score
                    ),
                    "target_log_return_sha256": _tensor_sha256(self.target_log_return),
                    "valid_sha256": _tensor_sha256(self.valid),
                    "residual_operator_receipt_sha256": (
                        self.residual_operator_receipt_sha256
                    ),
                    "available_risky_asset_count": self.available_risky_asset_count,
                    "factor_qualified_risky_asset_count": (
                        self.factor_qualified_risky_asset_count
                    ),
                    "effective_design_rank": self.effective_design_rank,
                    "weighted_residual_degrees_of_freedom": (
                        self.weighted_residual_degrees_of_freedom
                    ),
                    "outer_score_accessed": self.outer_score_accessed,
                    "lockbox_accessed": self.lockbox_accessed,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV12PredictiveLoss:
    total: torch.Tensor
    ranking: torch.Tensor
    economic_total: torch.Tensor
    robust_regression: torch.Tensor
    distributional: torch.Tensor
    component_weights: tuple[float, float, float]


def _date_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    return torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / count


def _weighted_horizon_mean(
    values: torch.Tensor,
    supported: torch.Tensor,
    setting: M03RV12PredictiveSetting,
) -> torch.Tensor:
    counts = supported.sum(dim=0)
    active = counts > 0
    weights = values.new_tensor(setting.horizon_loss_weights)
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    if float(active_weights.sum()) <= 0.0:
        raise M03RV12ObjectiveError("v12 objective has no supported horizon")
    per_horizon = torch.where(supported, values, torch.zeros_like(values)).sum(
        dim=0
    ) / counts.clamp_min(1).to(values.dtype)
    return (per_horizon * active_weights).sum() / active_weights.sum()


def _rank_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    setting: M03RV12PredictiveSetting,
) -> tuple[torch.Tensor, torch.Tensor]:
    supported = valid.sum(dim=1) >= 2
    if setting.ranking_objective == "none":
        return score.new_zeros(supported.shape), supported
    rows: list[torch.Tensor] = []
    target_scales = score.new_tensor(
        tuple(0.02 * math.sqrt(horizon) for horizon in M03R_V12_HORIZONS)
    )
    for date_index in range(score.shape[0]):
        date_rows: list[torch.Tensor] = []
        for horizon_index in range(score.shape[2]):
            mask = valid[date_index, :, horizon_index]
            if int(mask.sum()) < 2:
                date_rows.append(score.new_zeros(()))
                continue
            prediction = score[date_index, mask, horizon_index]
            realized = target[date_index, mask, horizon_index]
            if setting.ranking_objective == "rank-gaussian-correlation":
                centered = prediction - prediction.mean()
                prediction_scale = torch.sqrt(
                    centered.square()
                    .mean()
                    .clamp_min(M03R_V12_RANK_SCORE_STANDARD_DEVIATION_FLOOR**2)
                )
                prediction_z = centered / prediction_scale
                target_score = rank_gaussian_scores(realized)
                target_z = target_score / torch.sqrt(
                    target_score.square().mean().clamp_min(1.0e-12)
                )
                date_rows.append(1.0 - (prediction_z * target_z).mean())
            else:
                realized_scaled = realized.detach() / target_scales[horizon_index]
                date_rows.append(
                    -(
                        torch.softmax(realized_scaled, dim=0)
                        * torch.log_softmax(prediction, dim=0)
                    ).sum()
                )
        rows.append(torch.stack(date_rows))
    return torch.stack(rows), supported


def m03r_v12_predictive_loss(
    batch: M03RV12PredictiveBatch,
) -> M03RV12PredictiveLoss:
    batch.validate()
    mean = batch.predicted_mean
    log_scale = batch.predicted_log_scale.clamp(-8.0, 2.0)
    target = batch.target_log_return.detach()
    valid = batch.valid
    rank_rows, supported = _rank_loss(
        batch.predicted_rank_score, target, valid, batch.setting
    )
    ranking = _weighted_horizon_mean(rank_rows, supported, batch.setting)
    horizon_scale = mean.new_tensor(
        tuple(0.02 * math.sqrt(value) for value in M03R_V12_HORIZONS)
    ).view(1, 1, -1)
    robust = _weighted_horizon_mean(
        _date_mean(
            functional.huber_loss(
                mean / horizon_scale,
                target / horizon_scale,
                reduction="none",
                delta=1.0,
            ),
            valid,
        ),
        supported,
        batch.setting,
    )
    distributional = _weighted_horizon_mean(
        _date_mean(
            0.5
            * (
                torch.exp(-2.0 * log_scale) * (target - mean).square() + 2.0 * log_scale
            ),
            valid,
        ),
        supported,
        batch.setting,
    )
    weights = batch.setting.component_weights
    economic_total = weights[1] * robust + weights[2] * distributional
    total = weights[0] * ranking + economic_total
    if not bool(torch.isfinite(total)):
        raise M03RV12ObjectiveError("v12 predictive loss is non-finite")
    return M03RV12PredictiveLoss(
        total=total,
        ranking=ranking,
        economic_total=economic_total,
        robust_regression=robust,
        distributional=distributional,
        component_weights=weights,
    )


__all__ = [
    "M03R_V12_ALPHA_BATCH_SCHEMA",
    "M03R_V12_RANK_SCORE_STANDARD_DEVIATION_FLOOR",
    "M03RV12ObjectiveError",
    "M03RV12PredictiveBatch",
    "M03RV12PredictiveLoss",
    "m03r_v12_predictive_loss",
]
