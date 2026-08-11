"""Exact trace-integrity evidence for frozen TOP2000 M03R-v7 checkpoints.

The original development receipt intentionally retained only the compact
63-decision return/turnover evidence needed by the mechanism screen.  This
module is a retrospective, development-only audit surface.  It consumes a
fresh deterministic replay of an already frozen checkpoint and records the
intermediate books that were not part of the original immutable receipt.

Nothing in this module selects a checkpoint, trains a model, or authorizes
promotion.  Missing historical evidence is represented explicitly; it is
never reconstructed from rounded report values.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from rl_quant.envs.hold30 import TURNOVER_CAUSES
from rl_quant.evaluation.top2000_m03r_v7_dev import tensor_sha256
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_IDS,
)
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace

M03R_V7_TRACE_AUDIT_SCHEMA = "rl-quant.top2000-dev.m03r-v7-trace-audit-v1"
M03R_V7_PAIRWISE_TRACE_AUDIT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-pairwise-trace-audit-v1"
)
M03R_V7_FROZEN_FINAL_UPDATE = 64


class M03RV7TraceAuditError(RuntimeError):
    """Frozen trace evidence is absent, inconsistent, or malformed."""


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


def _require_sha256(name: str, value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV7TraceAuditError(f"{name} must be a lowercase SHA-256 digest")


def _stack_transition_tensor(
    trace: Hold30CanonicalTrace,
    name: str,
    start: int,
    stop: int,
) -> torch.Tensor:
    return torch.stack(
        [getattr(row, name).detach().to(device="cpu") for row in trace.transitions[start:stop]]
    )


def _array(value: object, *, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != ndim or result.dtype.kind not in "fc" or not np.isfinite(result).all():
        raise M03RV7TraceAuditError(f"{name} must be a finite floating {ndim}-D array")
    result = np.ascontiguousarray(result, dtype=np.float64)
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class M03RV7FrozenCheckpointIdentity:
    """Exact selected model and optimizer identity for one fold replay."""

    setting_index: int
    setting_id: str
    fold_index: int
    seed: int
    checkpoint_file_sha256: str
    model_state_sha256: str
    alpha_core_optimizer_state_sha256: str
    overlay_optimizer_state_sha256: str | None
    factor_calibration_receipt_sha256: str
    confidence_calibration_sha256: str | None = None
    confidence_calibration_status: str = (
        "unavailable-development-route-used-uncalibrated-raw-sigmoid-sizing"
    )
    selected_update: int = M03R_V7_FROZEN_FINAL_UPDATE
    checkpoint_selection_rule: str = (
        "frozen-final-optimizer-update-no-validation-selection-v1"
    )

    def __post_init__(self) -> None:
        if (
            not 0 <= self.setting_index < len(M03R_TOP2000_DEV_SETTING_IDS)
            or self.setting_id != M03R_TOP2000_DEV_SETTING_IDS[self.setting_index]
            or not 0 <= self.fold_index < 6
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.selected_update != M03R_V7_FROZEN_FINAL_UPDATE
            or self.checkpoint_selection_rule
            != "frozen-final-optimizer-update-no-validation-selection-v1"
        ):
            raise M03RV7TraceAuditError("checkpoint identity or selection rule drifted")
        for name in (
            "checkpoint_file_sha256",
            "model_state_sha256",
            "alpha_core_optimizer_state_sha256",
            "factor_calibration_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_sha256(
            "overlay_optimizer_state_sha256",
            self.overlay_optimizer_state_sha256,
            optional=True,
        )
        _require_sha256(
            "confidence_calibration_sha256",
            self.confidence_calibration_sha256,
            optional=True,
        )
        if self.confidence_calibration_sha256 is None and self.confidence_calibration_status != (
            "unavailable-development-route-used-uncalibrated-raw-sigmoid-sizing"
        ):
            raise M03RV7TraceAuditError("missing confidence calibration needs its exact status")


@dataclass(frozen=True, slots=True)
class M03RV7ForensicTrace:
    """Full-precision scored arrays from one frozen chronological replay."""

    checkpoint: M03RV7FrozenCheckpointIdentity
    score_dates: tuple[str, ...]
    decision_weights: np.ndarray
    requested_weights: np.ndarray
    post_hazard_weights: np.ndarray
    post_projection_weights: np.ndarray
    executed_weights: np.ndarray
    benchmark_weights: np.ndarray
    policy_gross_returns: np.ndarray
    policy_net_returns_20bp: np.ndarray
    benchmark_gross_returns: np.ndarray
    benchmark_net_returns_20bp: np.ndarray
    policy_total_one_way_turnover: np.ndarray
    benchmark_total_one_way_turnover: np.ndarray
    requested_action_trace_sha256: str
    development_only: bool = True
    future_selected_universe: bool = True
    reportable: bool = False
    promotable: bool = False
    schema: str = M03R_V7_TRACE_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_V7_TRACE_AUDIT_SCHEMA
            or not self.development_only
            or not self.future_selected_universe
            or self.reportable
            or self.promotable
            or len(self.score_dates) != 63
            or len(set(self.score_dates)) != 63
            or tuple(sorted(self.score_dates)) != self.score_dates
        ):
            raise M03RV7TraceAuditError("trace scope, chronology, or research gate drifted")
        _require_sha256("requested_action_trace_sha256", self.requested_action_trace_sha256)
        rows = len(self.score_dates)
        weights = (
            "decision_weights",
            "requested_weights",
            "post_hazard_weights",
            "post_projection_weights",
            "executed_weights",
            "benchmark_weights",
        )
        expected_assets: int | None = None
        for name in weights:
            value = _array(getattr(self, name), name=name, ndim=2)
            object.__setattr__(self, name, value)
            if value.shape[0] != rows:
                raise M03RV7TraceAuditError(f"{name} does not align with score dates")
            if expected_assets is None:
                expected_assets = value.shape[1]
            elif value.shape[1] != expected_assets:
                raise M03RV7TraceAuditError("weight arrays disagree on the asset axis")
            if np.any(value < -1e-7) or not np.allclose(value.sum(1), 1.0, atol=2e-6):
                raise M03RV7TraceAuditError(f"{name} must contain long-only simplexes")
        for name in (
            "policy_gross_returns",
            "policy_net_returns_20bp",
            "benchmark_gross_returns",
            "benchmark_net_returns_20bp",
            "policy_total_one_way_turnover",
            "benchmark_total_one_way_turnover",
        ):
            value = _array(getattr(self, name), name=name, ndim=1)
            object.__setattr__(self, name, value)
            if value.shape != (rows,):
                raise M03RV7TraceAuditError(f"{name} does not align with score dates")
        if np.any(self.policy_total_one_way_turnover < 0.0) or np.any(
            self.benchmark_total_one_way_turnover < 0.0
        ):
            raise M03RV7TraceAuditError("turnover cannot be negative")
        if not np.allclose(
            self.policy_net_returns_20bp,
            self.policy_gross_returns - 0.002 * self.policy_total_one_way_turnover,
            atol=2e-7,
            rtol=2e-7,
        ):
            raise M03RV7TraceAuditError("policy 20-bp return does not reconcile")
        if not np.allclose(
            self.benchmark_net_returns_20bp,
            self.benchmark_gross_returns - 0.002 * self.benchmark_total_one_way_turnover,
            atol=2e-7,
            rtol=2e-7,
        ):
            raise M03RV7TraceAuditError("benchmark 20-bp return does not reconcile")

    def array_sha256s(self) -> dict[str, str]:
        return {
            name: _array_sha256(getattr(self, name))
            for name in (
                "decision_weights",
                "requested_weights",
                "post_hazard_weights",
                "post_projection_weights",
                "executed_weights",
                "benchmark_weights",
                "policy_gross_returns",
                "policy_net_returns_20bp",
                "benchmark_gross_returns",
                "benchmark_net_returns_20bp",
                "policy_total_one_way_turnover",
                "benchmark_total_one_way_turnover",
            )
        }

    @property
    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "checkpoint": asdict(self.checkpoint),
            "score_dates": list(self.score_dates),
            "score_dates_sha256": _sha256(list(self.score_dates)),
            "arrays": self.array_sha256s(),
            "requested_action_trace_sha256": self.requested_action_trace_sha256,
            "development_only": self.development_only,
            "future_selected_universe": self.future_selected_universe,
            "reportable": self.reportable,
            "promotable": self.promotable,
        }
        return {**payload, "receipt_sha256": _sha256(payload)}


def build_m03r_v7_forensic_trace(
    trace: Hold30CanonicalTrace,
    *,
    checkpoint: M03RV7FrozenCheckpointIdentity,
    score_dates: tuple[str, ...],
    score_transition_start: int,
    score_transition_stop_exclusive: int,
    benchmark_weights: torch.Tensor,
    benchmark_gross_returns: torch.Tensor,
    benchmark_net_returns_20bp: torch.Tensor,
    benchmark_total_one_way_turnover: torch.Tensor,
) -> M03RV7ForensicTrace:
    """Extract exact intermediate books from one frozen checkpoint replay."""

    start = score_transition_start
    stop = score_transition_stop_exclusive
    if stop - start != 63 or len(trace.transitions) < stop or len(score_dates) != 63:
        raise M03RV7TraceAuditError("forensic replay must select exactly 63 transitions")
    risk_repaired = _stack_transition_tensor(trace, "risk_repaired_weights", start, stop)
    requested_delta = _stack_transition_tensor(trace, "requested_delta", start, stop)
    constructed_delta = _stack_transition_tensor(trace, "constructed_delta", start, stop)
    decisions = _stack_transition_tensor(trace, "decision_weights", start, stop)
    executed = _stack_transition_tensor(trace, "post_cost_weights", start, stop)
    requested = risk_repaired + requested_delta
    projected = risk_repaired + constructed_delta
    policy_gross = _stack_transition_tensor(trace, "holding_return", start, stop).mean(1)
    policy_net = _stack_transition_tensor(trace, "net_return", start, stop).mean(1)
    cause_rows = {
        cause: torch.stack(
            [row.turnover_by_cause[cause].mean() for row in trace.transitions[start:stop]]
        )
        .detach()
        .to(device="cpu", dtype=torch.float64)
        for cause in TURNOVER_CAUSES
    }
    policy_turnover = sum(
        cause_rows.values(),
        torch.zeros(63, dtype=torch.float64),
    )
    action_rows: list[dict[str, str | None]] = []
    for row in trace.transitions[start:stop]:
        intent = row.raw_intent
        action_payload = {
            name: None if (value := getattr(intent, name)) is None else tensor_sha256(value)
            for name in (
                "entry_scores",
                "target_logits",
                "gate",
                "raw_hazard_residual",
                "hazard_residual",
                "exact_hold_decision_st",
                "active_risk_scale",
                "signal_confidence",
                "auxiliary_alpha_mean",
            )
        }
        exit_action = intent.exit_action_v6
        action_payload.update(
            {
                "exit_action_v6.logits": (
                    None if exit_action is None else tensor_sha256(exit_action.logits)
                ),
                "exit_action_v6.soft_probabilities": (
                    None
                    if exit_action is None
                    else tensor_sha256(exit_action.soft_probabilities)
                ),
                "exit_action_v6.decision_st": (
                    None if exit_action is None else tensor_sha256(exit_action.decision_st)
                ),
            }
        )
        action_rows.append(action_payload)

    def scored(value: torch.Tensor, *, name: str, two_dimensional: bool) -> np.ndarray:
        tensor = value.detach().to(device="cpu", dtype=torch.float64)
        expected = (63, decisions.shape[-1]) if two_dimensional else (63,)
        if tuple(tensor.shape) == (trace.transitions[0].decision_weights.shape[0], 63):
            tensor = tensor.transpose(0, 1)
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tuple(tensor.shape) != expected:
            raise M03RV7TraceAuditError(f"{name} must have shape {expected}")
        return tensor.numpy()

    return M03RV7ForensicTrace(
        checkpoint=checkpoint,
        score_dates=score_dates,
        decision_weights=decisions.mean(1).to(torch.float64).numpy(),
        requested_weights=requested.mean(1).to(torch.float64).numpy(),
        post_hazard_weights=requested.mean(1).to(torch.float64).numpy(),
        post_projection_weights=projected.mean(1).to(torch.float64).numpy(),
        executed_weights=executed.mean(1).to(torch.float64).numpy(),
        benchmark_weights=scored(
            benchmark_weights, name="benchmark_weights", two_dimensional=True
        ),
        policy_gross_returns=policy_gross.to(torch.float64).numpy(),
        policy_net_returns_20bp=policy_net.to(torch.float64).numpy(),
        benchmark_gross_returns=scored(
            benchmark_gross_returns,
            name="benchmark_gross_returns",
            two_dimensional=False,
        ),
        benchmark_net_returns_20bp=scored(
            benchmark_net_returns_20bp,
            name="benchmark_net_returns_20bp",
            two_dimensional=False,
        ),
        policy_total_one_way_turnover=policy_turnover.to(torch.float64).numpy(),
        benchmark_total_one_way_turnover=scored(
            benchmark_total_one_way_turnover,
            name="benchmark_total_one_way_turnover",
            two_dimensional=False,
        ),
        requested_action_trace_sha256=_sha256(action_rows),
    )


def compare_m03r_v7_forensic_traces(
    left: M03RV7ForensicTrace,
    right: M03RV7ForensicTrace,
) -> dict[str, Any]:
    """Compare two causal settings on the same fold and fail on hidden identity reuse."""

    if (
        left.checkpoint.fold_index != right.checkpoint.fold_index
        or left.score_dates != right.score_dates
        or left.executed_weights.shape != right.executed_weights.shape
    ):
        raise M03RV7TraceAuditError("pairwise traces must share fold/date/asset geometry")
    requested_delta = np.abs(left.requested_weights - right.requested_weights)
    executed_delta = np.abs(left.executed_weights - right.executed_weights)
    identical_dates = np.all(left.executed_weights == right.executed_weights, axis=1)
    result = {
        "schema": M03R_V7_PAIRWISE_TRACE_AUDIT_SCHEMA,
        "fold_index": left.checkpoint.fold_index,
        "left_setting_index": left.checkpoint.setting_index,
        "left_setting_id": left.checkpoint.setting_id,
        "right_setting_index": right.checkpoint.setting_index,
        "right_setting_id": right.checkpoint.setting_id,
        "maximum_absolute_requested_weight_difference": float(requested_delta.max()),
        "maximum_absolute_executed_weight_difference": float(executed_delta.max()),
        "fraction_of_dates_with_identical_executed_weights": float(identical_dates.mean()),
        "gross_return_correlation": _pearson(
            left.policy_gross_returns, right.policy_gross_returns
        ),
        "net_active_return_correlation": _pearson(
            left.policy_net_returns_20bp - left.benchmark_net_returns_20bp,
            right.policy_net_returns_20bp - right.benchmark_net_returns_20bp,
        ),
        "selected_checkpoint_equal": (
            left.checkpoint.checkpoint_file_sha256
            == right.checkpoint.checkpoint_file_sha256
        ),
        "model_state_equal": (
            left.checkpoint.model_state_sha256 == right.checkpoint.model_state_sha256
        ),
        "requested_action_trace_equal": (
            left.requested_action_trace_sha256 == right.requested_action_trace_sha256
        ),
        "executed_weight_trace_equal": (
            left.array_sha256s()["executed_weights"]
            == right.array_sha256s()["executed_weights"]
        ),
        "causal_distinctness_gate_passed": not (
            left.checkpoint.setting_id != right.checkpoint.setting_id
            and left.array_sha256s()["executed_weights"]
            == right.array_sha256s()["executed_weights"]
        ),
        "development_only": True,
        "reportable": False,
        "promotable": False,
    }
    return {**result, "receipt_sha256": _sha256(result)}


__all__ = [
    "M03R_V7_FROZEN_FINAL_UPDATE",
    "M03R_V7_PAIRWISE_TRACE_AUDIT_SCHEMA",
    "M03R_V7_TRACE_AUDIT_SCHEMA",
    "M03RV7ForensicTrace",
    "M03RV7FrozenCheckpointIdentity",
    "M03RV7TraceAuditError",
    "build_m03r_v7_forensic_trace",
    "compare_m03r_v7_forensic_traces",
]
