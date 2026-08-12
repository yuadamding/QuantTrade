"""Untouched-tail predictive diagnostics for M03R-v10."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import (
    M03R_V10_HORIZONS,
    M03R_V10_PROTOCOL_SHA256,
    M03RV10PredictiveSetting,
)
from rl_quant.training.top2000_m03r_v10_pretraining_step import (
    M03RV10AlphaPretrainingBatch,
)

M03R_V10_FOLD_DIAGNOSTICS_SCHEMA = "rl-quant.top2000-dev.m03r-v10-fold-diagnostics-v1"


class M03RV10DiagnosticsError(ValueError):
    """The v10 untouched-tail evidence is malformed."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    row = value.detach().to(device="cpu", dtype=torch.float64)
    order = torch.argsort(row, stable=True)
    sorted_row = row.index_select(0, order)
    _unique, counts = torch.unique_consecutive(sorted_row, return_counts=True)
    stops = counts.cumsum(0)
    starts = stops - counts
    average = (starts + stops - 1).to(row.dtype) / 2.0
    sorted_ranks = torch.repeat_interleave(average, counts)
    result = torch.empty_like(sorted_ranks)
    result.scatter_(0, order, sorted_ranks)
    return result


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    ranked_first = _average_ranks(first)
    ranked_second = _average_ranks(second)
    ranked_first -= ranked_first.mean()
    ranked_second -= ranked_second.mean()
    denominator = torch.linalg.vector_norm(ranked_first) * torch.linalg.vector_norm(
        ranked_second
    )
    if float(denominator) == 0.0:
        return 0.0
    return float((ranked_first * ranked_second).sum() / denominator)


@dataclass(frozen=True, slots=True)
class M03RV10FoldDiagnostics:
    setting_index: int
    setting_id: str
    setting_receipt_sha256: str
    fold_index: int
    mean_spearman_rank_ic: tuple[float, float, float, float]
    population_std_spearman_rank_ic: tuple[float, float, float, float]
    median_spearman_rank_ic: tuple[float, float, float, float]
    positive_ic_date_fraction: tuple[float, float, float, float]
    mean_top_bottom_decile_spread: tuple[float, float, float, float]
    mean_prediction_cross_sectional_std: tuple[float, float, float, float]
    mean_target_cross_sectional_std: tuple[float, float, float, float]
    mean_predicted_scale: tuple[float, float, float, float]
    valid_date_counts: tuple[int, int, int, int]
    source_array_sha256: str
    asset_axis_sha256: str
    exposure_receipt_sha256: str
    evaluated_update: int = 64
    qualification_tail_used_for_optimization: bool = False
    protocol_sha256: str = M03R_V10_PROTOCOL_SHA256
    schema: str = M03R_V10_FOLD_DIAGNOSTICS_SCHEMA

    def validate(self) -> None:
        vectors = (
            self.mean_spearman_rank_ic,
            self.population_std_spearman_rank_ic,
            self.median_spearman_rank_ic,
            self.positive_ic_date_fraction,
            self.mean_top_bottom_decile_spread,
            self.mean_prediction_cross_sectional_std,
            self.mean_target_cross_sectional_std,
            self.mean_predicted_scale,
        )
        if (
            not 0 <= self.setting_index < 3
            or not self.setting_id
            or len(self.setting_receipt_sha256) != 64
            or not 0 <= self.fold_index < 6
            or any(len(values) != 4 for values in vectors)
            or any(not math.isfinite(value) for values in vectors for value in values)
            or any(value < 0.0 for value in self.population_std_spearman_rank_ic)
            or any(not 0.0 <= value <= 1.0 for value in self.positive_ic_date_fraction)
            or any(value < 0.0 for value in self.mean_prediction_cross_sectional_std)
            or any(value < 0.0 for value in self.mean_target_cross_sectional_std)
            or any(value <= 0.0 for value in self.mean_predicted_scale)
            or len(self.valid_date_counts) != 4
            or any(count <= 0 for count in self.valid_date_counts)
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.setting_receipt_sha256,
                    self.source_array_sha256,
                    self.asset_axis_sha256,
                    self.exposure_receipt_sha256,
                )
            )
            or self.evaluated_update != 64
            or self.qualification_tail_used_for_optimization
            or self.protocol_sha256 != M03R_V10_PROTOCOL_SHA256
            or self.schema != M03R_V10_FOLD_DIAGNOSTICS_SCHEMA
        ):
            raise M03RV10DiagnosticsError("v10 fold diagnostics drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def build_m03r_v10_fold_diagnostics(
    batch: M03RV10AlphaPretrainingBatch,
) -> M03RV10FoldDiagnostics:
    batch.validate()
    base = batch.imported_v9_batch
    if base.split != "qualification":
        raise M03RV10DiagnosticsError(
            "v10 fold diagnostics require the untouched qualification tail"
        )
    summaries: dict[str, list[float | int]] = {
        "mean_ic": [],
        "std_ic": [],
        "median_ic": [],
        "positive_ic": [],
        "spread": [],
        "prediction_std": [],
        "target_std": [],
        "scale": [],
        "count": [],
    }
    for horizon_index in range(len(M03R_V10_HORIZONS)):
        ic_rows: list[float] = []
        spread_rows: list[float] = []
        prediction_std_rows: list[float] = []
        target_std_rows: list[float] = []
        scale_rows: list[float] = []
        for date_index in range(base.predicted_mean.shape[0]):
            mask = base.valid[date_index, :, horizon_index]
            if int(mask.sum()) < 2:
                continue
            predicted = base.predicted_mean[date_index, mask, horizon_index]
            target = base.target_log_return[date_index, mask, horizon_index]
            scale = torch.exp(base.predicted_log_scale[date_index, mask, horizon_index])
            ic_rows.append(_spearman(predicted, target))
            order = torch.argsort(predicted, stable=True)
            decile = max(1, int(order.numel()) // 10)
            spread_rows.append(
                float(
                    target.index_select(0, order[-decile:]).mean()
                    - target.index_select(0, order[:decile]).mean()
                )
            )
            prediction_std_rows.append(float(predicted.std(unbiased=False)))
            target_std_rows.append(float(target.std(unbiased=False)))
            scale_rows.append(float(scale.mean()))
        if not ic_rows:
            raise M03RV10DiagnosticsError("v10 horizon has no qualification support")
        ic_tensor = torch.tensor(ic_rows, dtype=torch.float64)
        summaries["mean_ic"].append(float(ic_tensor.mean()))
        summaries["std_ic"].append(float(ic_tensor.std(unbiased=False)))
        summaries["median_ic"].append(float(ic_tensor.median()))
        summaries["positive_ic"].append(float((ic_tensor > 0.0).double().mean()))
        summaries["spread"].append(sum(spread_rows) / len(spread_rows))
        summaries["prediction_std"].append(
            sum(prediction_std_rows) / len(prediction_std_rows)
        )
        summaries["target_std"].append(sum(target_std_rows) / len(target_std_rows))
        summaries["scale"].append(sum(scale_rows) / len(scale_rows))
        summaries["count"].append(len(ic_rows))
    setting: M03RV10PredictiveSetting = batch.setting
    evidence = M03RV10FoldDiagnostics(
        setting_index=setting.setting_index,
        setting_id=setting.setting_id,
        setting_receipt_sha256=setting.receipt_sha256,
        fold_index=base.fold_index,
        mean_spearman_rank_ic=tuple(summaries["mean_ic"]),  # type: ignore[arg-type]
        population_std_spearman_rank_ic=tuple(summaries["std_ic"]),  # type: ignore[arg-type]
        median_spearman_rank_ic=tuple(summaries["median_ic"]),  # type: ignore[arg-type]
        positive_ic_date_fraction=tuple(summaries["positive_ic"]),  # type: ignore[arg-type]
        mean_top_bottom_decile_spread=tuple(summaries["spread"]),  # type: ignore[arg-type]
        mean_prediction_cross_sectional_std=tuple(  # type: ignore[arg-type]
            summaries["prediction_std"]
        ),
        mean_target_cross_sectional_std=tuple(  # type: ignore[arg-type]
            summaries["target_std"]
        ),
        mean_predicted_scale=tuple(summaries["scale"]),  # type: ignore[arg-type]
        valid_date_counts=tuple(summaries["count"]),  # type: ignore[arg-type]
        source_array_sha256=base.source_array_sha256,
        asset_axis_sha256=base.asset_axis_sha256,
        exposure_receipt_sha256=base.exposure_receipt_sha256 or "",
    )
    evidence.validate()
    return evidence


__all__ = [
    "M03R_V10_FOLD_DIAGNOSTICS_SCHEMA",
    "M03RV10DiagnosticsError",
    "M03RV10FoldDiagnostics",
    "build_m03r_v10_fold_diagnostics",
]
