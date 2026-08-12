"""Date-balanced predictive loss and diagnostics for TOP2000 M03R-v9."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch.nn import functional

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_HORIZONS,
    M03R_V9_PREDICTIVE_SPEC,
    M03R_V9_PROTOCOL_SHA256,
)

M03R_V9_ALPHA_BATCH_SCHEMA = "rl-quant.top2000-dev.m03r-v9-alpha-batch-v1"
M03R_V9_ALPHA_FOLD_SCHEMA = "rl-quant.top2000-dev.m03r-v9-alpha-fold-v1"
M03R_V9_HORIZON_SCALES = tuple(
    0.02 * math.sqrt(horizon) for horizon in M03R_V9_HORIZONS
)


class M03RV9AlphaPretrainingError(ValueError):
    """V9 alpha targets, losses, or evidence are invalid."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV9AlphaPretrainingError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV9AlphaPretrainingBatch:
    predicted_mean: torch.Tensor  # [date, asset, horizon]
    predicted_log_scale: torch.Tensor  # [date, asset, horizon]
    target_log_return: torch.Tensor  # [date, asset, horizon]
    valid: torch.Tensor  # [date, asset, horizon]
    origin_indices: torch.Tensor  # [date]
    split: Literal["training", "qualification"]
    target_mode: Literal["factor-residual", "benchmark-relative"]
    fold_index: int
    split_start_inclusive: int
    split_stop_exclusive: int
    source_array_sha256: str
    asset_axis_sha256: str
    exposure_receipt_sha256: str | None
    outer_score_accessed: bool = False
    lockbox_accessed: bool = False
    schema: str = M03R_V9_ALPHA_BATCH_SCHEMA

    def validate(self) -> None:
        mean = self.predicted_mean
        if (
            not isinstance(mean, torch.Tensor)
            or mean.ndim != 3
            or mean.shape[-1] != len(M03R_V9_HORIZONS)
            or not mean.is_floating_point()
            or not bool(torch.isfinite(mean).all())
        ):
            raise M03RV9AlphaPretrainingError(
                "predicted mean must be finite [date,asset,4]"
            )
        for name in ("predicted_log_scale", "target_log_return"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(mean.shape)
                or value.dtype != mean.dtype
                or value.device != mean.device
                or not bool(torch.isfinite(value).all())
            ):
                raise M03RV9AlphaPretrainingError(
                    f"{name} is not aligned with predictions"
                )
        if (
            not isinstance(self.valid, torch.Tensor)
            or tuple(self.valid.shape) != tuple(mean.shape)
            or self.valid.dtype != torch.bool
            or self.valid.device != mean.device
            or not isinstance(self.origin_indices, torch.Tensor)
            or tuple(self.origin_indices.shape) != (mean.shape[0],)
            or self.origin_indices.dtype != torch.long
            or self.origin_indices.device != mean.device
            or bool((self.origin_indices[1:] <= self.origin_indices[:-1]).any())
        ):
            raise M03RV9AlphaPretrainingError("valid/origin axes are malformed")
        if (
            self.split not in {"training", "qualification"}
            or self.target_mode not in {"factor-residual", "benchmark-relative"}
            or isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < 6
            or not 0 <= self.split_start_inclusive < self.split_stop_exclusive
            or int(self.origin_indices[0]) < self.split_start_inclusive
            or self.schema != M03R_V9_ALPHA_BATCH_SCHEMA
            or self.outer_score_accessed
            or self.lockbox_accessed
        ):
            raise M03RV9AlphaPretrainingError(
                "V9 batch split or research identity drifted"
            )
        horizons = torch.tensor(M03R_V9_HORIZONS, device=mean.device)
        stops = self.origin_indices.unsqueeze(-1) + horizons + 1
        if bool((stops[self.valid.any(dim=1)] > self.split_stop_exclusive).any()):
            raise M03RV9AlphaPretrainingError("valid targets cross their frozen split")
        _digest("source_array_sha256", self.source_array_sha256)
        _digest("asset_axis_sha256", self.asset_axis_sha256)
        if self.target_mode == "factor-residual":
            if self.exposure_receipt_sha256 is None:
                raise M03RV9AlphaPretrainingError(
                    "factor-residual targets lack exposure evidence"
                )
            _digest("exposure_receipt_sha256", self.exposure_receipt_sha256)
        elif self.exposure_receipt_sha256 is not None:
            raise M03RV9AlphaPretrainingError(
                "benchmark-relative targets must not claim residualization"
            )
        if not bool((self.valid.sum(dim=1) >= 2).any()):
            raise M03RV9AlphaPretrainingError("batch has no supported cross section")


@dataclass(frozen=True, slots=True)
class M03RV9AlphaPretrainingLoss:
    total: torch.Tensor
    listwise_ranking: torch.Tensor
    robust_regression: torch.Tensor
    distributional: torch.Tensor
    component_weights: tuple[float, float, float]
    valid_date_horizon_count: int


def _weighted_horizon_mean(
    values: torch.Tensor,
    supported: torch.Tensor,
) -> torch.Tensor:
    counts = supported.sum(dim=0)
    active = counts > 0
    per_horizon = torch.where(supported, values, torch.zeros_like(values)).sum(dim=0)
    per_horizon = per_horizon / counts.clamp_min(1).to(values.dtype)
    weights = values.new_tensor(M03R_V9_PREDICTIVE_SPEC.horizon_loss_weights)
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    return (per_horizon * active_weights).sum() / active_weights.sum()


def _date_means(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=1).clamp_min(1).to(values.dtype)
    return torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / count


def m03r_v9_alpha_pretraining_loss(
    batch: M03RV9AlphaPretrainingBatch,
    *,
    ranking_enabled: bool,
) -> M03RV9AlphaPretrainingLoss:
    """Apply horizon scaling and a unit-normalized active component mixture."""

    batch.validate()
    if not isinstance(ranking_enabled, bool):
        raise M03RV9AlphaPretrainingError("ranking_enabled must be boolean")
    valid = batch.valid
    supported = valid.sum(dim=1) >= 2
    if not bool(supported.any()):
        raise M03RV9AlphaPretrainingError("objective has no supported date/horizon")
    mean = batch.predicted_mean
    target = batch.target_log_return.detach()
    scales = mean.new_tensor(M03R_V9_HORIZON_SCALES).view(1, 1, -1)
    standardized_mean = mean / scales
    standardized_target = target / scales

    masked_mean = torch.where(
        valid, standardized_mean, torch.full_like(mean, -torch.inf)
    )
    masked_target = torch.where(
        valid, standardized_target, torch.full_like(target, -torch.inf)
    )
    log_probability = torch.log_softmax(masked_mean, dim=1)
    target_probability = torch.softmax(masked_target, dim=1)
    ranking_by_date_horizon = -torch.where(
        valid,
        target_probability * log_probability,
        torch.zeros_like(log_probability),
    ).sum(dim=1)
    ranking = _weighted_horizon_mean(ranking_by_date_horizon, supported)

    error = standardized_mean - standardized_target
    robust_points = functional.huber_loss(
        error,
        torch.zeros_like(error),
        delta=1.0,
        reduction="none",
    )
    robust = _weighted_horizon_mean(_date_means(robust_points, valid), supported)

    log_scale = batch.predicted_log_scale.clamp(-8.0, 2.0)
    distributional_points = (
        0.5 * (mean - target).square() * torch.exp(-2.0 * log_scale) + log_scale
    )
    distributional = _weighted_horizon_mean(
        _date_means(distributional_points, valid),
        supported,
    )
    weights = (
        M03R_V9_PREDICTIVE_SPEC.ranked_component_weights
        if ranking_enabled
        else M03R_V9_PREDICTIVE_SPEC.no_ranking_component_weights
    )
    total = weights[0] * ranking + weights[1] * robust + weights[2] * distributional
    if not bool(torch.isfinite(total)):
        raise M03RV9AlphaPretrainingError("V9 predictive objective is non-finite")
    return M03RV9AlphaPretrainingLoss(
        total=total,
        listwise_ranking=ranking,
        robust_regression=robust,
        distributional=distributional,
        component_weights=weights,
        valid_date_horizon_count=int(supported.sum()),
    )


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    row = value.detach().to(device="cpu", dtype=torch.float64)
    order = torch.argsort(row, stable=True)
    sorted_row = row.index_select(0, order)
    ranks = torch.empty_like(sorted_row)
    start = 0
    while start < sorted_row.numel():
        stop = start + 1
        while stop < sorted_row.numel() and bool(sorted_row[stop] == sorted_row[start]):
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    result = torch.empty_like(ranks)
    result[order] = ranks
    return result


def _spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
    first = _average_ranks(prediction)
    second = _average_ranks(target)
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) == 0.0:
        return 0.0
    return float((first * second).sum() / denominator)


@dataclass(frozen=True, slots=True)
class M03RV9AlphaFoldEvidence:
    fold_index: int
    target_mode: Literal["factor-residual", "benchmark-relative"]
    mean_spearman_rank_ic: tuple[float, float, float, float]
    mean_top_bottom_decile_spread: tuple[float, float, float, float]
    valid_date_counts: tuple[int, int, int, int]
    source_array_sha256: str
    asset_axis_sha256: str
    exposure_receipt_sha256: str | None
    evaluated_update: int = 64
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    schema: str = M03R_V9_ALPHA_FOLD_SCHEMA

    def __post_init__(self) -> None:
        if (
            not 0 <= self.fold_index < 6
            or self.target_mode not in {"factor-residual", "benchmark-relative"}
            or len(self.mean_spearman_rank_ic) != 4
            or len(self.mean_top_bottom_decile_spread) != 4
            or len(self.valid_date_counts) != 4
            or not all(math.isfinite(value) for value in self.mean_spearman_rank_ic)
            or not all(
                math.isfinite(value) for value in self.mean_top_bottom_decile_spread
            )
            or not all(count > 0 for count in self.valid_date_counts)
            or self.evaluated_update != 64
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or self.schema != M03R_V9_ALPHA_FOLD_SCHEMA
        ):
            raise M03RV9AlphaPretrainingError("V9 fold evidence is invalid")
        _digest("source_array_sha256", self.source_array_sha256)
        _digest("asset_axis_sha256", self.asset_axis_sha256)
        if self.target_mode == "factor-residual":
            if self.exposure_receipt_sha256 is None:
                raise M03RV9AlphaPretrainingError(
                    "factor fold evidence lacks exposure receipt"
                )
            _digest("exposure_receipt_sha256", self.exposure_receipt_sha256)
        elif self.exposure_receipt_sha256 is not None:
            raise M03RV9AlphaPretrainingError(
                "benchmark fold evidence claims factor receipt"
            )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def build_m03r_v9_alpha_fold_evidence(
    batch: M03RV9AlphaPretrainingBatch,
) -> M03RV9AlphaFoldEvidence:
    batch.validate()
    if batch.split != "qualification":
        raise M03RV9AlphaPretrainingError(
            "fold evidence requires the untouched qualification tail"
        )
    predictions = batch.predicted_mean.detach()
    targets = batch.target_log_return.detach()
    rank_ic: list[float] = []
    spreads: list[float] = []
    counts: list[int] = []
    for horizon_index in range(len(M03R_V9_HORIZONS)):
        horizon_ic: list[float] = []
        horizon_spread: list[float] = []
        for date_index in range(predictions.shape[0]):
            mask = batch.valid[date_index, :, horizon_index]
            if int(mask.sum()) < 2:
                continue
            predicted = predictions[date_index, mask, horizon_index]
            realized = targets[date_index, mask, horizon_index]
            horizon_ic.append(_spearman(predicted, realized))
            order = torch.argsort(predicted, stable=True)
            decile = max(1, int(order.numel()) // 10)
            horizon_spread.append(
                float(
                    realized.index_select(0, order[-decile:]).mean()
                    - realized.index_select(0, order[:decile]).mean()
                )
            )
        if not horizon_ic:
            raise M03RV9AlphaPretrainingError(
                "qualification horizon has no supported dates"
            )
        rank_ic.append(sum(horizon_ic) / len(horizon_ic))
        spreads.append(sum(horizon_spread) / len(horizon_spread))
        counts.append(len(horizon_ic))
    return M03RV9AlphaFoldEvidence(
        fold_index=batch.fold_index,
        target_mode=batch.target_mode,
        mean_spearman_rank_ic=tuple(rank_ic),  # type: ignore[arg-type]
        mean_top_bottom_decile_spread=tuple(spreads),  # type: ignore[arg-type]
        valid_date_counts=tuple(counts),  # type: ignore[arg-type]
        source_array_sha256=batch.source_array_sha256,
        asset_axis_sha256=batch.asset_axis_sha256,
        exposure_receipt_sha256=batch.exposure_receipt_sha256,
    )


__all__ = [
    "M03R_V9_ALPHA_BATCH_SCHEMA",
    "M03R_V9_ALPHA_FOLD_SCHEMA",
    "M03R_V9_HORIZON_SCALES",
    "M03RV9AlphaFoldEvidence",
    "M03RV9AlphaPretrainingBatch",
    "M03RV9AlphaPretrainingError",
    "M03RV9AlphaPretrainingLoss",
    "build_m03r_v9_alpha_fold_evidence",
    "m03r_v9_alpha_pretraining_loss",
]
