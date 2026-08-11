"""Training-fold-only alpha pretraining for TOP2000 M03R-v8 development.

The v8 discovery protocol requires predictive qualification before economic
policy optimization.  This module owns the differentiable, date-balanced
pretraining loss and the detached six-fold qualification gate.  It accepts
only already constructed training or inner-validation targets; outer-score
and lockbox labels have no route through this API.

The ranking term is listwise rather than all-pairs.  Its memory complexity is
linear in the stock axis, avoiding a roughly 4-million-element pair matrix per
date for TOP2000 while preserving a cross-sectional ordering objective.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch.nn import functional

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
)

M03R_V8_ALPHA_PRETRAINING_SCHEMA = "rl-quant.top2000-dev.m03r-v8-alpha-pretraining-v1"
M03R_V8_ALPHA_QUALIFICATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-alpha-qualification-v1"
)
M03R_V8_ALPHA_HORIZON_SCALES = tuple(
    0.02 * math.sqrt(horizon)
    for horizon in M03R_V8_ALPHA_PRETRAINING.horizons_trading_sessions
)
M03R_V8_ALPHA_LISTWISE_TEMPERATURE = 0.02
M03R_V8_ALPHA_HUBER_DELTA = 1.0


class M03RV8AlphaPretrainingError(ValueError):
    """Alpha pretraining inputs or qualification evidence are invalid."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor(name: str, value: torch.Tensor, *, ndim: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV8AlphaPretrainingError(
            f"{name} must be a finite floating rank-{ndim} tensor"
        )
    return value


@dataclass(frozen=True, slots=True)
class M03RV8AlphaPretrainingBatch:
    """One causally bounded training or inner-validation prediction batch."""

    predicted_mean: torch.Tensor  # [date, asset, horizon]
    predicted_log_scale: torch.Tensor  # [date, asset, horizon]
    target_residual_log_return: torch.Tensor  # [date, asset, horizon]
    valid: torch.Tensor  # [date, asset, horizon]
    origin_indices: torch.Tensor  # [date]
    split: Literal["training", "inner-validation"]
    fold_index: int
    split_start_inclusive: int
    split_stop_exclusive: int
    source_array_sha256: str
    outer_score_accessed: bool = False
    lockbox_accessed: bool = False
    schema: str = M03R_V8_ALPHA_PRETRAINING_SCHEMA

    def validate(self) -> None:
        prediction = _tensor("predicted_mean", self.predicted_mean, ndim=3)
        shape = tuple(prediction.shape)
        if shape[-1] != len(M03R_V8_ALPHA_PRETRAINING.horizons_trading_sessions):
            raise M03RV8AlphaPretrainingError(
                "alpha predictions must carry exactly the frozen four horizons"
            )
        for name in ("predicted_log_scale", "target_residual_log_return"):
            value = _tensor(name, getattr(self, name), ndim=3)
            if (
                tuple(value.shape) != shape
                or value.dtype != prediction.dtype
                or value.device != prediction.device
            ):
                raise M03RV8AlphaPretrainingError(
                    f"{name} must align exactly with predicted_mean"
                )
        if (
            not isinstance(self.valid, torch.Tensor)
            or tuple(self.valid.shape) != shape
            or self.valid.dtype != torch.bool
            or self.valid.device != prediction.device
        ):
            raise M03RV8AlphaPretrainingError(
                "valid must be boolean and aligned with alpha predictions"
            )
        if (
            not isinstance(self.origin_indices, torch.Tensor)
            or tuple(self.origin_indices.shape) != (shape[0],)
            or self.origin_indices.dtype != torch.long
            or self.origin_indices.device != prediction.device
            or bool((self.origin_indices[1:] <= self.origin_indices[:-1]).any())
        ):
            raise M03RV8AlphaPretrainingError(
                "origin_indices must be a strictly increasing int64 date vector"
            )
        if (
            self.split not in {"training", "inner-validation"}
            or isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < 6
            or isinstance(self.split_start_inclusive, bool)
            or isinstance(self.split_stop_exclusive, bool)
            or not isinstance(self.split_start_inclusive, int)
            or not isinstance(self.split_stop_exclusive, int)
            or not 0 <= self.split_start_inclusive < self.split_stop_exclusive
            or int(self.origin_indices[0]) < self.split_start_inclusive
        ):
            raise M03RV8AlphaPretrainingError("pretraining split identity is invalid")
        horizons = torch.tensor(
            M03R_V8_ALPHA_PRETRAINING.horizons_trading_sessions,
            dtype=torch.long,
            device=prediction.device,
        )
        target_stops = self.origin_indices.unsqueeze(-1) + horizons + 1
        if bool(
            (target_stops[self.valid.any(dim=1)] > self.split_stop_exclusive).any()
        ):
            raise M03RV8AlphaPretrainingError(
                "a valid alpha target crosses its training/inner-validation boundary"
            )
        if (
            not isinstance(self.source_array_sha256, str)
            or len(self.source_array_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_array_sha256
            )
        ):
            raise M03RV8AlphaPretrainingError(
                "source_array_sha256 must be a lowercase SHA-256 digest"
            )
        if (
            self.outer_score_accessed
            or self.lockbox_accessed
            or self.schema != M03R_V8_ALPHA_PRETRAINING_SCHEMA
        ):
            raise M03RV8AlphaPretrainingError(
                "v8 alpha pretraining forbids outer-score/lockbox access or schema drift"
            )
        if not bool((self.valid.sum(dim=1) >= 2).any()):
            raise M03RV8AlphaPretrainingError(
                "pretraining batch has no date/horizon with two valid risky assets"
            )


@dataclass(frozen=True, slots=True)
class M03RV8AlphaPretrainingLoss:
    """Date-balanced objective components for one prediction batch."""

    total: torch.Tensor
    listwise_ranking: torch.Tensor
    robust_regression: torch.Tensor
    distributional: torch.Tensor
    valid_date_horizon_count: int


def _date_horizon_means(
    values: torch.Tensor,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    count = valid_rows.sum(dim=1).clamp_min(1).to(values.dtype)
    return torch.where(valid_rows, values, torch.zeros_like(values)).sum(dim=1) / count


def _frozen_horizon_weighted_mean(
    values_by_date_horizon: torch.Tensor,
    supported_date_horizon: torch.Tensor,
) -> torch.Tensor:
    """Average dates within horizons, then apply the frozen horizon weights."""

    supported_counts = supported_date_horizon.sum(dim=0)
    active_horizons = supported_counts > 0
    per_horizon = torch.where(
        supported_date_horizon,
        values_by_date_horizon,
        torch.zeros_like(values_by_date_horizon),
    ).sum(dim=0) / supported_counts.clamp_min(1).to(values_by_date_horizon.dtype)
    frozen_weights = values_by_date_horizon.new_tensor(
        M03R_V8_ALPHA_PRETRAINING.horizon_loss_weights
    )
    active_weights = torch.where(
        active_horizons,
        frozen_weights,
        torch.zeros_like(frozen_weights),
    )
    return (per_horizon * active_weights).sum() / active_weights.sum()


def m03r_v8_alpha_pretraining_loss(
    batch: M03RV8AlphaPretrainingBatch,
    *,
    ranking_loss_weight: float | None = None,
) -> M03RV8AlphaPretrainingLoss:
    """Return the frozen listwise, Huber, and Gaussian pretraining objective."""

    batch.validate()
    ranking_weight = (
        M03R_V8_ALPHA_PRETRAINING.ranking_loss_weight
        if ranking_loss_weight is None
        else float(ranking_loss_weight)
    )
    if ranking_weight not in {0.0, M03R_V8_ALPHA_PRETRAINING.ranking_loss_weight}:
        raise M03RV8AlphaPretrainingError(
            "ranking loss weight must match the reference or no-ranking ablation"
        )
    prediction = batch.predicted_mean
    target = batch.target_residual_log_return.detach()
    log_scale = batch.predicted_log_scale
    valid = batch.valid
    date_horizon_valid = valid.sum(dim=1) >= 2
    if not bool(date_horizon_valid.any()):
        raise M03RV8AlphaPretrainingError(
            "alpha objective has no supported date/horizon rows"
        )

    masked_prediction = torch.where(
        valid,
        prediction / M03R_V8_ALPHA_LISTWISE_TEMPERATURE,
        torch.full_like(prediction, -torch.inf),
    )
    masked_target = torch.where(
        valid,
        target / M03R_V8_ALPHA_LISTWISE_TEMPERATURE,
        torch.full_like(target, -torch.inf),
    )
    log_probability = torch.log_softmax(masked_prediction, dim=1)
    target_probability = torch.softmax(masked_target, dim=1)
    listwise_by_date_horizon = -torch.where(
        valid,
        target_probability * log_probability,
        torch.zeros_like(log_probability),
    ).sum(dim=1)
    listwise = _frozen_horizon_weighted_mean(
        listwise_by_date_horizon,
        date_horizon_valid,
    )

    scales = prediction.new_tensor(M03R_V8_ALPHA_HORIZON_SCALES).view(1, 1, -1)
    standardized_error = (prediction - target) / scales
    huber_point = functional.huber_loss(
        standardized_error,
        torch.zeros_like(standardized_error),
        delta=M03R_V8_ALPHA_HUBER_DELTA,
        reduction="none",
    )
    huber_by_date_horizon = _date_horizon_means(huber_point, valid)
    huber = _frozen_horizon_weighted_mean(
        huber_by_date_horizon,
        date_horizon_valid,
    )

    bounded_log_scale = log_scale.clamp(-8.0, 2.0)
    distributional_point = (
        0.5 * ((prediction - target).square() * torch.exp(-2.0 * bounded_log_scale))
        + bounded_log_scale
    )
    distributional_by_date_horizon = _date_horizon_means(
        distributional_point,
        valid,
    )
    distributional = _frozen_horizon_weighted_mean(
        distributional_by_date_horizon,
        date_horizon_valid,
    )

    total = (
        ranking_weight * listwise
        + M03R_V8_ALPHA_PRETRAINING.huber_loss_weight * huber
        + M03R_V8_ALPHA_PRETRAINING.distributional_loss_weight * distributional
    )
    if not bool(torch.isfinite(total)):
        raise M03RV8AlphaPretrainingError("alpha pretraining objective is non-finite")
    return M03RV8AlphaPretrainingLoss(
        total=total,
        listwise_ranking=listwise,
        robust_regression=huber,
        distributional=distributional,
        valid_date_horizon_count=int(date_horizon_valid.sum()),
    )


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    """Return deterministic average ranks for a detached one-dimensional row."""

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
class M03RV8AlphaFoldEvidence:
    """Detached predictive diagnostics for one chronological fold."""

    fold_index: int
    mean_spearman_rank_ic: tuple[float, float, float, float]
    mean_top_bottom_decile_spread: tuple[float, float, float, float]
    valid_date_counts: tuple[int, int, int, int]
    source_array_sha256: str
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = M03R_V8_ALPHA_QUALIFICATION_SCHEMA

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < 6
            or len(self.mean_spearman_rank_ic) != 4
            or len(self.mean_top_bottom_decile_spread) != 4
            or len(self.valid_date_counts) != 4
            or not all(math.isfinite(value) for value in self.mean_spearman_rank_ic)
            or not all(
                math.isfinite(value) for value in self.mean_top_bottom_decile_spread
            )
            or not all(count > 0 for count in self.valid_date_counts)
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or self.schema != M03R_V8_ALPHA_QUALIFICATION_SCHEMA
        ):
            raise M03RV8AlphaPretrainingError("fold alpha evidence is invalid")
        if len(self.source_array_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_array_sha256
        ):
            raise M03RV8AlphaPretrainingError("fold source hash is invalid")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def build_m03r_v8_alpha_fold_evidence(
    batch: M03RV8AlphaPretrainingBatch,
) -> M03RV8AlphaFoldEvidence:
    """Compute date-balanced rank IC and top-minus-bottom realized spread."""

    batch.validate()
    if batch.split != "inner-validation":
        raise M03RV8AlphaPretrainingError(
            "alpha qualification requires an inner-validation batch"
        )
    predictions = batch.predicted_mean.detach()
    targets = batch.target_residual_log_return.detach()
    rank_ic: list[float] = []
    spreads: list[float] = []
    counts: list[int] = []
    for horizon_index in range(predictions.shape[-1]):
        per_date_ic: list[float] = []
        per_date_spread: list[float] = []
        for date_index in range(predictions.shape[0]):
            mask = batch.valid[date_index, :, horizon_index]
            if int(mask.sum()) < 2:
                continue
            predicted = predictions[date_index, mask, horizon_index]
            realized = targets[date_index, mask, horizon_index]
            per_date_ic.append(_spearman(predicted, realized))
            order = torch.argsort(predicted, stable=True)
            decile = max(1, int(order.numel()) // 10)
            lower = realized.index_select(0, order[:decile]).mean()
            upper = realized.index_select(0, order[-decile:]).mean()
            per_date_spread.append(float(upper - lower))
        if not per_date_ic:
            raise M03RV8AlphaPretrainingError(
                "inner-validation horizon has no supported dates"
            )
        rank_ic.append(sum(per_date_ic) / len(per_date_ic))
        spreads.append(sum(per_date_spread) / len(per_date_spread))
        counts.append(len(per_date_ic))
    return M03RV8AlphaFoldEvidence(
        fold_index=batch.fold_index,
        mean_spearman_rank_ic=tuple(rank_ic),  # type: ignore[arg-type]
        mean_top_bottom_decile_spread=tuple(spreads),  # type: ignore[arg-type]
        valid_date_counts=tuple(counts),  # type: ignore[arg-type]
        source_array_sha256=batch.source_array_sha256,
    )


@dataclass(frozen=True, slots=True)
class M03RV8AlphaPanelQualification:
    """Six-fold predictive gate required before v8 policy optimization."""

    fold_receipt_sha256: tuple[str, ...]
    mean_rank_ic_21d: float
    mean_rank_ic_30d: float
    positive_rank_ic_fold_count_21d: int
    positive_rank_ic_fold_count_30d: int
    qualifying_horizons: tuple[int, ...]
    passed: bool
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = M03R_V8_ALPHA_QUALIFICATION_SCHEMA

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def qualify_m03r_v8_alpha_panel(
    folds: tuple[M03RV8AlphaFoldEvidence, ...],
) -> M03RV8AlphaPanelQualification:
    """Apply the frozen 21/30-session predictive gate across exactly six folds."""

    if len(folds) != 6 or tuple(sorted(row.fold_index for row in folds)) != tuple(
        range(6)
    ):
        raise M03RV8AlphaPretrainingError(
            "alpha qualification requires one unique evidence row for each of six folds"
        )
    ordered = tuple(sorted(folds, key=lambda row: row.fold_index))
    horizon_indexes = {21: 1, 30: 2}
    means: dict[int, float] = {}
    positive: dict[int, int] = {}
    qualifying: list[int] = []
    for horizon, index in horizon_indexes.items():
        values = [row.mean_spearman_rank_ic[index] for row in ordered]
        means[horizon] = sum(values) / len(values)
        positive[horizon] = sum(value > 0.0 for value in values)
        if (
            means[horizon] >= M03R_V8_ALPHA_PRETRAINING.minimum_mean_spearman_rank_ic
            and positive[horizon]
            >= M03R_V8_ALPHA_PRETRAINING.minimum_positive_rank_ic_fold_count
        ):
            qualifying.append(horizon)
    return M03RV8AlphaPanelQualification(
        fold_receipt_sha256=tuple(row.receipt_sha256 for row in ordered),
        mean_rank_ic_21d=means[21],
        mean_rank_ic_30d=means[30],
        positive_rank_ic_fold_count_21d=positive[21],
        positive_rank_ic_fold_count_30d=positive[30],
        qualifying_horizons=tuple(qualifying),
        passed=bool(qualifying),
    )


__all__ = [
    "M03R_V8_ALPHA_HORIZON_SCALES",
    "M03R_V8_ALPHA_LISTWISE_TEMPERATURE",
    "M03R_V8_ALPHA_PRETRAINING_SCHEMA",
    "M03R_V8_ALPHA_QUALIFICATION_SCHEMA",
    "M03RV8AlphaFoldEvidence",
    "M03RV8AlphaPanelQualification",
    "M03RV8AlphaPretrainingBatch",
    "M03RV8AlphaPretrainingError",
    "M03RV8AlphaPretrainingLoss",
    "build_m03r_v8_alpha_fold_evidence",
    "m03r_v8_alpha_pretraining_loss",
    "qualify_m03r_v8_alpha_panel",
]
