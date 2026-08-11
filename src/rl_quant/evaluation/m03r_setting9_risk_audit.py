"""Focused risk-route audit for TOP2000 M03R-v7 setting index 9."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_IDS,
    resolve_m03r_top2000_dev_setting,
)

M03R_SETTING9_RISK_AUDIT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-setting9-risk-audit-v1"
)
M03R_SETTING9_INDEX = 9
M03R_SETTING9_ID = M03R_TOP2000_DEV_SETTING_IDS[M03R_SETTING9_INDEX]


class M03RSetting9RiskAuditError(RuntimeError):
    """Setting-9 evidence is incomplete or does not belong to its exact route."""


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


@dataclass(frozen=True, slots=True)
class M03RSetting9RiskAuditInput:
    fold_index: int
    initial_policy_weights: np.ndarray
    initial_benchmark_weights: np.ndarray
    requested_annual_tracking_error: np.ndarray | None
    post_projection_annual_tracking_error: np.ndarray | None
    realized_active_returns: np.ndarray
    reported_total_one_way_turnover: np.ndarray
    startup_turnover: float
    startup_turnover_in_reported_mean: bool
    benchmark_anchoring_enabled: bool
    tracking_error_control_enabled: bool
    active_beta_control_enabled: bool
    factor_sector_projection_enabled: bool = False
    setting_index: int = M03R_SETTING9_INDEX
    setting_id: str = M03R_SETTING9_ID

    def __post_init__(self) -> None:
        setting = resolve_m03r_top2000_dev_setting(self.setting_id)
        if (
            self.setting_index != M03R_SETTING9_INDEX
            or self.setting_id != M03R_SETTING9_ID
            or setting.setting_index != M03R_SETTING9_INDEX
            or setting.factor_sector_neutral_projection
            or self.factor_sector_projection_enabled
            or not 0 <= self.fold_index < 6
        ):
            raise M03RSetting9RiskAuditError("setting-9 route identity drifted")
        initial_policy = np.asarray(self.initial_policy_weights, dtype=np.float64)
        initial_benchmark = np.asarray(self.initial_benchmark_weights, dtype=np.float64)
        if (
            initial_policy.ndim != 1
            or initial_policy.shape != initial_benchmark.shape
            or not np.isfinite(initial_policy).all()
            or not np.isfinite(initial_benchmark).all()
            or np.any(initial_policy < -1e-7)
            or np.any(initial_benchmark < -1e-7)
            or not math.isclose(float(initial_policy.sum()), 1.0, abs_tol=2e-6)
            or not math.isclose(float(initial_benchmark.sum()), 1.0, abs_tol=2e-6)
        ):
            raise M03RSetting9RiskAuditError("initial books must be aligned simplexes")
        object.__setattr__(self, "initial_policy_weights", initial_policy)
        object.__setattr__(self, "initial_benchmark_weights", initial_benchmark)
        for name in ("realized_active_returns", "reported_total_one_way_turnover"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
                raise M03RSetting9RiskAuditError(f"{name} must be a finite vector")
            if name.endswith("turnover") and np.any(array < 0.0):
                raise M03RSetting9RiskAuditError("turnover cannot be negative")
            object.__setattr__(self, name, array)
        if self.realized_active_returns.shape != self.reported_total_one_way_turnover.shape:
            raise M03RSetting9RiskAuditError("realized return and turnover dates must align")
        for name in (
            "requested_annual_tracking_error",
            "post_projection_annual_tracking_error",
        ):
            raw = getattr(self, name)
            if raw is None:
                continue
            array = np.asarray(raw, dtype=np.float64)
            if array.shape != self.realized_active_returns.shape or not np.isfinite(array).all():
                raise M03RSetting9RiskAuditError(f"{name} must align with realized returns")
            if np.any(array < 0.0):
                raise M03RSetting9RiskAuditError("tracking error cannot be negative")
            object.__setattr__(self, name, array)
        if not math.isfinite(self.startup_turnover) or self.startup_turnover < 0.0:
            raise M03RSetting9RiskAuditError("startup_turnover must be nonnegative")


def evaluate_m03r_setting9_risk_audit(
    value: M03RSetting9RiskAuditInput,
) -> dict[str, Any]:
    """Classify the high-TE/low-turnover setting without overclaiming causality."""

    realized_te = float(np.std(value.realized_active_returns, ddof=1) * math.sqrt(252.0))
    initial_distance = float(
        0.5 * np.abs(value.initial_policy_weights - value.initial_benchmark_weights).sum()
    )
    post_projection_max = (
        None
        if value.post_projection_annual_tracking_error is None
        else float(value.post_projection_annual_tracking_error.max())
    )
    requested_max = (
        None
        if value.requested_annual_tracking_error is None
        else float(value.requested_annual_tracking_error.max())
    )
    missing_controls = [
        name
        for name, enabled in (
            ("benchmark_anchoring", value.benchmark_anchoring_enabled),
            ("tracking_error_control", value.tracking_error_control_enabled),
            ("active_beta_control", value.active_beta_control_enabled),
        )
        if not enabled
    ]
    diagnosis: list[str] = []
    if missing_controls:
        diagnosis.append("route-disabled-common-risk-controls")
    if post_projection_max is None:
        diagnosis.append("post-projection-tracking-error-evidence-unavailable")
    elif post_projection_max > 0.06 + 1e-8:
        diagnosis.append("post-projection-ex-ante-tracking-error-exceeded-six-percent")
    elif realized_te > 0.06:
        diagnosis.append("realized-tracking-error-exceeded-ex-ante-control")
    if not value.startup_turnover_in_reported_mean:
        diagnosis.append("reported-turnover-excludes-startup")
    if initial_distance > 1e-8:
        diagnosis.append("initial-policy-book-differs-from-c1")
    if not diagnosis:
        diagnosis.append("no-mechanism-explanation-found")
    payload: dict[str, Any] = {
        "schema": M03R_SETTING9_RISK_AUDIT_SCHEMA,
        "setting_index": value.setting_index,
        "setting_id": value.setting_id,
        "fold_index": value.fold_index,
        "initial_policy_to_c1_one_way_distance": initial_distance,
        "startup_turnover": value.startup_turnover,
        "startup_turnover_in_reported_mean": value.startup_turnover_in_reported_mean,
        "mean_reported_total_one_way_turnover": float(
            value.reported_total_one_way_turnover.mean()
        ),
        "maximum_requested_annual_tracking_error": requested_max,
        "maximum_post_projection_annual_tracking_error": post_projection_max,
        "realized_annual_tracking_error": realized_te,
        "common_controls": {
            "benchmark_anchoring_enabled": value.benchmark_anchoring_enabled,
            "tracking_error_control_enabled": value.tracking_error_control_enabled,
            "active_beta_control_enabled": value.active_beta_control_enabled,
            "factor_sector_projection_enabled": value.factor_sector_projection_enabled,
        },
        "diagnosis": diagnosis,
        "clean_factor_neutrality_ablation": (
            not missing_controls
            and post_projection_max is not None
            and post_projection_max <= 0.06 + 1e-8
        ),
        "causal_interpretation_authorized": False,
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    return {**payload, "receipt_sha256": _sha256(payload)}


__all__ = [
    "M03R_SETTING9_ID",
    "M03R_SETTING9_INDEX",
    "M03R_SETTING9_RISK_AUDIT_SCHEMA",
    "M03RSetting9RiskAuditError",
    "M03RSetting9RiskAuditInput",
    "evaluate_m03r_setting9_risk_audit",
]
