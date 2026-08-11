"""Attribute active-book attenuation across the M03R-v7 execution path."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

M03R_PROJECTION_ATTRIBUTION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-projection-attribution-v1"
)


class M03RProjectionAttributionError(RuntimeError):
    """Execution-stage arrays are unbound, infeasible, or inconsistent."""


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


def _rankdata(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    result = np.empty(value.size, dtype=np.float64)
    result[order] = np.arange(value.size, dtype=np.float64)
    return result


@dataclass(frozen=True, slots=True)
class M03RProjectionAttributionInput:
    setting_index: int
    setting_id: str
    fold_index: int
    score_dates: tuple[str, ...]
    benchmark_weights: np.ndarray
    requested_weights: np.ndarray
    post_hazard_weights: np.ndarray
    post_projection_weights: np.ndarray
    executed_weights: np.ndarray
    alpha_scores: np.ndarray | None = None
    covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = len(self.score_dates)
        if (
            not 0 <= self.setting_index < 12
            or not 0 <= self.fold_index < 6
            or rows != 63
            or tuple(sorted(self.score_dates)) != self.score_dates
        ):
            raise M03RProjectionAttributionError("projection identity or chronology drifted")
        assets: int | None = None
        for name in (
            "benchmark_weights",
            "requested_weights",
            "post_hazard_weights",
            "post_projection_weights",
            "executed_weights",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 2 or value.shape[0] != rows or not np.isfinite(value).all():
                raise M03RProjectionAttributionError(f"{name} must be finite [date,asset]")
            if assets is None:
                assets = value.shape[1]
            elif value.shape[1] != assets:
                raise M03RProjectionAttributionError("execution stages disagree on asset axis")
            if np.any(value < -1e-7) or not np.allclose(value.sum(1), 1.0, atol=2e-6):
                raise M03RProjectionAttributionError(f"{name} must be long-only simplexes")
            value = np.ascontiguousarray(value)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        assert assets is not None
        if self.alpha_scores is not None:
            alpha = np.asarray(self.alpha_scores, dtype=np.float64)
            if alpha.shape != (rows, assets) or not np.isfinite(alpha).all():
                raise M03RProjectionAttributionError("alpha_scores must be finite [date,asset]")
            alpha = np.ascontiguousarray(alpha)
            alpha.setflags(write=False)
            object.__setattr__(self, "alpha_scores", alpha)
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=np.float64)
            if covariance.shape != (rows, assets, assets) or not np.isfinite(covariance).all():
                raise M03RProjectionAttributionError(
                    "covariance must be finite [date,asset,asset]"
                )
            if not np.allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-10):
                raise M03RProjectionAttributionError("covariance must be symmetric")
            covariance = np.ascontiguousarray(covariance)
            covariance.setflags(write=False)
            object.__setattr__(self, "covariance", covariance)


def evaluate_m03r_projection_attribution(
    value: M03RProjectionAttributionInput,
) -> dict[str, Any]:
    """Measure how much requested active risk and score survive each stage."""

    stages = {
        "requested": value.requested_weights,
        "post_hazard": value.post_hazard_weights,
        "post_projection": value.post_projection_weights,
        "executed": value.executed_weights,
    }
    stage_rows: dict[str, Any] = {}
    for name, weights in stages.items():
        active = weights - value.benchmark_weights
        row: dict[str, Any] = {
            "mean_active_l2_norm": float(np.linalg.norm(active, axis=1).mean()),
            "mean_active_l1_norm": float(np.abs(active).sum(1).mean()),
        }
        if value.covariance is None:
            row["mean_ex_ante_annual_tracking_error"] = None
            row["tracking_error_status"] = "unavailable-covariance-not-supplied"
        else:
            variance = np.einsum("di,dij,dj->d", active, value.covariance, active)
            if np.any(variance < -1e-10):
                raise M03RProjectionAttributionError("covariance produced negative variance")
            row["mean_ex_ante_annual_tracking_error"] = float(
                np.sqrt(np.maximum(variance, 0.0) * 252.0).mean()
            )
            row["tracking_error_status"] = "available"
        if value.alpha_scores is None:
            row["mean_expected_active_alpha"] = None
            row["alpha_attribution_status"] = "unavailable-alpha-scores-not-supplied"
        else:
            row["mean_expected_active_alpha"] = float(
                np.einsum("da,da->d", active, value.alpha_scores).mean()
            )
            row["alpha_attribution_status"] = "available"
        stage_rows[name] = row

    pairs = (
        ("requested_to_post_hazard", value.requested_weights, value.post_hazard_weights),
        (
            "post_hazard_to_post_projection",
            value.post_hazard_weights,
            value.post_projection_weights,
        ),
        (
            "post_projection_to_executed",
            value.post_projection_weights,
            value.executed_weights,
        ),
    )
    transitions: dict[str, Any] = {}
    for name, left, right in pairs:
        distance = np.linalg.norm(right - left, axis=1)
        transitions[name] = {
            "mean_l2_distance": float(distance.mean()),
            "maximum_l2_distance": float(distance.max()),
            "fraction_of_decisions_changed": float(np.mean(distance > 1e-10)),
        }
    requested_active = value.requested_weights - value.benchmark_weights
    executed_active = value.executed_weights - value.benchmark_weights
    denominator = np.linalg.norm(requested_active, axis=1)
    retention = np.divide(
        np.linalg.norm(executed_active, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    rank_correlations: list[float] = []
    if value.alpha_scores is not None:
        for scores, active in zip(value.alpha_scores, executed_active, strict=True):
            if np.std(scores[1:]) == 0.0 or np.std(active[1:]) == 0.0:
                continue
            correlation = float(np.corrcoef(_rankdata(scores[1:]), _rankdata(active[1:]))[0, 1])
            if math.isfinite(correlation):
                rank_correlations.append(correlation)
    payload: dict[str, Any] = {
        "schema": M03R_PROJECTION_ATTRIBUTION_SCHEMA,
        "setting_index": value.setting_index,
        "setting_id": value.setting_id,
        "fold_index": value.fold_index,
        "score_dates": list(value.score_dates),
        "array_sha256": {
            name: _array_sha256(getattr(value, name))
            for name in (
                "benchmark_weights",
                "requested_weights",
                "post_hazard_weights",
                "post_projection_weights",
                "executed_weights",
            )
        },
        "stages": stage_rows,
        "stage_transitions": transitions,
        "mean_projection_retention_ratio": float(retention.mean()),
        "median_projection_retention_ratio": float(np.median(retention)),
        "requested_versus_executed_score_rank_correlation": (
            None if not rank_correlations else float(np.mean(rank_correlations))
        ),
        "pre_exact_hold_requested_book_status": (
            "unavailable-runtime-retains-post-hazard-built-action-only"
        ),
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    return {**payload, "receipt_sha256": _sha256(payload)}


__all__ = [
    "M03R_PROJECTION_ATTRIBUTION_SCHEMA",
    "M03RProjectionAttributionError",
    "M03RProjectionAttributionInput",
    "evaluate_m03r_projection_attribution",
]
