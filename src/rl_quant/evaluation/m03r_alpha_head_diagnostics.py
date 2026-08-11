"""Date-balanced diagnostics for frozen M03R-v7 residual-alpha heads."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

M03R_ALPHA_DIAGNOSTIC_HORIZONS = (5, 21, 30, 63)
M03R_ALPHA_HEAD_DIAGNOSTIC_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-alpha-head-diagnostic-v1"
)
M03R_ALPHA_HEAD_DIAGNOSTIC_UNAVAILABLE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-alpha-head-diagnostic-unavailable-v1"
)


class M03RAlphaHeadDiagnosticError(RuntimeError):
    """Alpha prediction/target evidence is malformed or causally incomplete."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise M03RAlphaHeadDiagnosticError(f"{name} must have shape {shape}")
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result


def _rankdata(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    sorted_value = value[order]
    ranks = np.empty(value.size, dtype=np.float64)
    first = 0
    while first < value.size:
        stop = first + 1
        while stop < value.size and sorted_value[stop] == sorted_value[first]:
            stop += 1
        ranks[order[first:stop]] = 0.5 * (first + stop - 1)
        first = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    if left.size < 2:
        return None
    if rank:
        left = _rankdata(left)
        right = _rankdata(right)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _mean_optional(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _ic_ir(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    std = float(np.std(values, ddof=1))
    return None if std <= 0.0 else float(np.mean(values) / std)


@dataclass(frozen=True, slots=True)
class M03RAlphaHeadDiagnosticInput:
    setting_index: int
    setting_id: str
    fold_index: int
    score_dates: tuple[str, ...]
    action_ids: tuple[str, ...]
    predictions: np.ndarray
    targets: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray | None = None
    breakdowns: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        rows = len(self.score_dates)
        assets = len(self.action_ids)
        if (
            not 0 <= self.setting_index < 12
            or not 0 <= self.fold_index < 6
            or rows != 63
            or assets < 2
            or self.action_ids[0] != "CASH"
            or tuple(sorted(self.score_dates)) != self.score_dates
            or len(set(self.score_dates)) != rows
            or len(set(self.action_ids)) != assets
        ):
            raise M03RAlphaHeadDiagnosticError("alpha diagnostic identity or axes drifted")
        shape = (rows, assets, len(M03R_ALPHA_DIAGNOSTIC_HORIZONS))
        predictions = _readonly_array(self.predictions, name="predictions", shape=shape)
        targets = _readonly_array(self.targets, name="targets", shape=shape)
        valid = _readonly_array(self.valid, name="valid", shape=shape)
        if predictions.dtype.kind not in "fc" or targets.dtype.kind not in "fc":
            raise M03RAlphaHeadDiagnosticError("predictions and targets must be floating")
        if valid.dtype != np.bool_:
            raise M03RAlphaHeadDiagnosticError("valid must be boolean")
        if not np.isfinite(predictions[valid]).all() or not np.isfinite(targets[valid]).all():
            raise M03RAlphaHeadDiagnosticError("valid predictions and targets must be finite")
        if np.any(valid[:, 0]):
            raise M03RAlphaHeadDiagnosticError("CASH cannot enter residual-alpha diagnostics")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "valid", valid)
        if self.confidence is not None:
            confidence = _readonly_array(
                self.confidence,
                name="confidence",
                shape=(rows, assets),
            )
            if confidence.dtype.kind not in "fc" or not np.isfinite(confidence).all():
                raise M03RAlphaHeadDiagnosticError("confidence must be finite floating")
            object.__setattr__(self, "confidence", confidence)
        if self.breakdowns is not None:
            validated: dict[str, np.ndarray] = {}
            for name, values in self.breakdowns.items():
                if not isinstance(name, str) or not name:
                    raise M03RAlphaHeadDiagnosticError("breakdown names must be non-empty")
                validated[name] = _readonly_array(
                    values,
                    name=f"breakdowns[{name}]",
                    shape=(rows, assets),
                )
            object.__setattr__(self, "breakdowns", validated)


def _horizon_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    pearson: list[float] = []
    spearman: list[float] = []
    spreads: list[float] = []
    quintile_precision: list[float] = []
    directional: list[float] = []
    calibration_mae: list[float] = []
    prediction_dispersion: list[float] = []
    target_dispersion: list[float] = []
    valid_counts: list[int] = []
    for date_index in range(predictions.shape[0]):
        mask = valid[date_index]
        predicted = predictions[date_index, mask]
        target = targets[date_index, mask]
        valid_counts.append(int(mask.sum()))
        if predicted.size < 2:
            continue
        if (value := _correlation(predicted, target, rank=False)) is not None:
            pearson.append(value)
        if (value := _correlation(predicted, target, rank=True)) is not None:
            spearman.append(value)
        directional.append(float(np.mean(np.sign(predicted) == np.sign(target))))
        calibration_mae.append(float(np.mean(np.abs(predicted - target))))
        prediction_dispersion.append(float(np.std(predicted, ddof=1)))
        target_dispersion.append(float(np.std(target, ddof=1)))
        if predicted.size >= 10:
            tail = max(1, predicted.size // 10)
            ordering = np.argsort(predicted, kind="mergesort")
            spreads.append(
                float(target[ordering[-tail:]].mean() - target[ordering[:tail]].mean())
            )
        if predicted.size >= 5:
            top = max(1, predicted.size // 5)
            predicted_top = set(np.argsort(predicted, kind="mergesort")[-top:].tolist())
            target_top = set(np.argsort(target, kind="mergesort")[-top:].tolist())
            quintile_precision.append(len(predicted_top & target_top) / top)
    return {
        "valid_date_count": int(sum(count >= 2 for count in valid_counts)),
        "mean_valid_asset_count": float(np.mean(valid_counts)),
        "pearson_correlation": _mean_optional(pearson),
        "spearman_rank_ic": _mean_optional(spearman),
        "rank_ic_information_ratio_across_dates": _ic_ir(spearman),
        "positive_rank_ic_date_count": int(sum(value > 0.0 for value in spearman)),
        "top_minus_bottom_decile_residual_return": _mean_optional(spreads),
        "top_quintile_precision": _mean_optional(quintile_precision),
        "directional_accuracy": _mean_optional(directional),
        "calibration_mean_absolute_error": _mean_optional(calibration_mae),
        "prediction_dispersion": _mean_optional(prediction_dispersion),
        "target_dispersion": _mean_optional(target_dispersion),
        "aggregation": "date-balanced-equal-weight-over-valid-dates",
    }


def evaluate_m03r_alpha_head_diagnostics(
    value: M03RAlphaHeadDiagnosticInput,
) -> dict[str, Any]:
    """Evaluate all four auxiliary horizons without portfolio-policy returns."""

    horizons = {
        str(horizon): _horizon_metrics(
            value.predictions[:, :, index],
            value.targets[:, :, index],
            value.valid[:, :, index],
        )
        for index, horizon in enumerate(M03R_ALPHA_DIAGNOSTIC_HORIZONS)
    }
    confidence_rows: dict[str, Any]
    if value.confidence is None:
        confidence_rows = {
            "status": "unavailable",
            "reason": "frozen-confidence-array-not-supplied",
        }
    else:
        confidence_rows = {"status": "available", "deciles": {}}
        common = value.valid[:, :, 2]
        observed = value.confidence[common]
        if observed.size < 10:
            confidence_rows = {
                "status": "unavailable",
                "reason": "fewer-than-ten-valid-30-session-confidence-observations",
            }
        else:
            edges = np.quantile(observed, np.linspace(0.0, 1.0, 11))
            for decile in range(10):
                low, high = edges[decile], edges[decile + 1]
                mask = common & (value.confidence >= low)
                mask &= value.confidence <= high if decile == 9 else value.confidence < high
                errors = np.abs(value.predictions[:, :, 2] - value.targets[:, :, 2])
                confidence_rows["deciles"][str(decile + 1)] = {
                    "count": int(mask.sum()),
                    "mean_absolute_error": (
                        None if not np.any(mask) else float(errors[mask].mean())
                    ),
                    "mean_realized_target": (
                        None
                        if not np.any(mask)
                        else float(value.targets[:, :, 2][mask].mean())
                    ),
                }
    breakdown_rows: dict[str, Any] = {}
    if value.breakdowns is not None:
        for name, categories in value.breakdowns.items():
            category_rows: dict[str, Any] = {}
            for category in np.unique(categories[value.valid[:, :, 2]]):
                mask = value.valid[:, :, 2] & (categories == category)
                category_rows[str(category)] = _horizon_metrics(
                    value.predictions[:, :, 2],
                    value.targets[:, :, 2],
                    mask,
                )
            breakdown_rows[name] = category_rows
    payload: dict[str, Any] = {
        "schema": M03R_ALPHA_HEAD_DIAGNOSTIC_SCHEMA,
        "setting_index": value.setting_index,
        "setting_id": value.setting_id,
        "fold_index": value.fold_index,
        "score_dates": list(value.score_dates),
        "action_ids_sha256": _sha256(list(value.action_ids)),
        "array_sha256": {
            "predictions": _array_sha256(value.predictions),
            "targets": _array_sha256(value.targets),
            "valid": _array_sha256(value.valid),
            "confidence": (
                None if value.confidence is None else _array_sha256(value.confidence)
            ),
            "breakdowns": {
                name: _sha256(np.asarray(categories).tolist())
                for name, categories in (value.breakdowns or {}).items()
            },
        },
        "horizons": horizons,
        "confidence_diagnostics": confidence_rows,
        "conditional_breakdowns": breakdown_rows,
        "portfolio_policy_returns_used": False,
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    return {**payload, "receipt_sha256": _sha256(payload)}


def build_unavailable_m03r_alpha_head_diagnostics(
    *,
    setting_index: int,
    setting_id: str,
    fold_index: int,
    score_dates: tuple[str, ...],
    action_ids: tuple[str, ...],
    residual_alpha_head_mode: str,
) -> dict[str, Any]:
    """Bind the intentional absence of residual-alpha heads without fake arrays."""

    if (
        not 0 <= setting_index < 12
        or not isinstance(setting_id, str)
        or not setting_id
        or not 0 <= fold_index < 6
        or len(score_dates) != 63
        or tuple(sorted(score_dates)) != score_dates
        or len(set(score_dates)) != len(score_dates)
        or len(action_ids) < 2
        or action_ids[0] != "CASH"
        or len(set(action_ids)) != len(action_ids)
        or residual_alpha_head_mode != "none"
    ):
        raise M03RAlphaHeadDiagnosticError(
            "unavailable alpha diagnostic identity or head mode drifted"
        )
    reason = "setting-intentionally-disables-residual-alpha-heads"
    payload: dict[str, Any] = {
        "schema": M03R_ALPHA_HEAD_DIAGNOSTIC_UNAVAILABLE_SCHEMA,
        "status": "unavailable",
        "reason": reason,
        "setting_index": setting_index,
        "setting_id": setting_id,
        "fold_index": fold_index,
        "score_dates": list(score_dates),
        "action_ids_sha256": _sha256(list(action_ids)),
        "residual_alpha_head_mode": residual_alpha_head_mode,
        "alpha_heads_present": False,
        "array_sha256": {
            "predictions": None,
            "targets": None,
            "valid": None,
            "confidence": None,
            "breakdowns": {},
        },
        "horizons": {
            str(horizon): {"status": "unavailable", "reason": reason}
            for horizon in M03R_ALPHA_DIAGNOSTIC_HORIZONS
        },
        "confidence_diagnostics": {"status": "unavailable", "reason": reason},
        "conditional_breakdowns": {},
        "portfolio_policy_returns_used": False,
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    return {**payload, "receipt_sha256": _sha256(payload)}


__all__ = [
    "M03R_ALPHA_DIAGNOSTIC_HORIZONS",
    "M03R_ALPHA_HEAD_DIAGNOSTIC_SCHEMA",
    "M03R_ALPHA_HEAD_DIAGNOSTIC_UNAVAILABLE_SCHEMA",
    "M03RAlphaHeadDiagnosticError",
    "M03RAlphaHeadDiagnosticInput",
    "build_unavailable_m03r_alpha_head_diagnostics",
    "evaluate_m03r_alpha_head_diagnostics",
]
