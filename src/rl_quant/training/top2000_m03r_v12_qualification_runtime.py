"""Round-trip checkpoint lineage for M03R-v12 predictive qualification."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_HORIZONS,
    M03R_V12_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v12_checkpoint import M03RV12LoadedCheckpoint
from rl_quant.training.top2000_m03r_v12_objective import M03RV12PredictiveBatch
from rl_quant.training.top2000_m03r_v12_runtime import M03RV12SimpleSleeveTrace
from rl_quant.training.top2000_m03r_v12_selection import (
    M03RV12BootstrapPlan,
    M03RV12FoldEvidence,
    M03RV12PredictiveQualification,
    build_m03r_v12_fold_evidence,
    qualify_m03r_v12_predictive_candidate,
)

M03R_V12_FOLD_QUALIFICATION_LINEAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-fold-qualification-lineage-v1"
)


class M03RV12QualificationRuntimeError(ValueError):
    """Qualification evidence is not bound to exact reloaded checkpoint bytes."""


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


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV12QualificationRuntimeError(f"{name} must be a lowercase SHA-256")
    return value


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
class M03RV12FoldQualificationLineage:
    loaded_checkpoint: M03RV12LoadedCheckpoint
    fold_evidence: M03RV12FoldEvidence
    evaluation_trace_sha256: str
    qualification_source_array_sha256: str
    qualification_asset_axis_sha256: str
    qualification_residual_operator_root_sha256: str
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    schema: str = M03R_V12_FOLD_QUALIFICATION_LINEAGE_SCHEMA

    def validate(self) -> None:
        evidence = self.fold_evidence
        loaded = self.loaded_checkpoint
        loaded.validate()
        evidence.validate()
        _digest("evaluation_trace_sha256", self.evaluation_trace_sha256)
        if (
            loaded.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or loaded.completed_updates != 64
            or evidence.setting_index != loaded.setting_index
            or evidence.setting_id != loaded.setting_id
            or evidence.fold_index != loaded.fold_index
            or evidence.horizon_sessions != loaded.selected_horizon_sessions
            or evidence.checkpoint_file_sha256 != loaded.checkpoint_file_sha256
            or evidence.episode_schedule_sha256 != loaded.episode_schedule_sha256
            or evidence.residual_operator_root_sha256
            != self.qualification_residual_operator_root_sha256
            or self.qualification_asset_axis_sha256 != loaded.asset_axis_sha256
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.schema != M03R_V12_FOLD_QUALIFICATION_LINEAGE_SCHEMA
        ):
            raise M03RV12QualificationRuntimeError(
                "v12 fold evidence is not bound to its reloaded checkpoint"
            )
        for name, value in (
            ("model_state_sha256", loaded.model_state_sha256),
            ("source_array_sha256", loaded.source_array_sha256),
            ("asset_axis_sha256", loaded.asset_axis_sha256),
            (
                "qualification_source_array_sha256",
                self.qualification_source_array_sha256,
            ),
            (
                "qualification_residual_operator_root_sha256",
                self.qualification_residual_operator_root_sha256,
            ),
        ):
            _digest(name, value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "schema": self.schema,
                "protocol_sha256": self.protocol_sha256,
                "loaded_checkpoint": asdict(self.loaded_checkpoint),
                "fold_evidence_sha256": self.fold_evidence.receipt_sha256,
                "evaluation_trace_sha256": self.evaluation_trace_sha256,
                "qualification_source_array_sha256": (
                    self.qualification_source_array_sha256
                ),
                "qualification_asset_axis_sha256": (
                    self.qualification_asset_axis_sha256
                ),
                "qualification_residual_operator_root_sha256": (
                    self.qualification_residual_operator_root_sha256
                ),
            }
        )


def build_m03r_v12_fold_qualification_lineage(
    loaded: M03RV12LoadedCheckpoint,
    batch: M03RV12PredictiveBatch,
    trace: M03RV12SimpleSleeveTrace,
) -> M03RV12FoldQualificationLineage:
    """Derive one fold's metrics only from the loaded model's exact trace."""

    batch.validate()
    trace.validate()
    base = batch
    if batch.residual_operators is None:
        raise M03RV12QualificationRuntimeError(
            "v12 qualification batch omits its executable residual operators"
        )
    horizon_index = M03R_V12_HORIZONS.index(loaded.selected_horizon_sessions)
    selected_operator_receipts = tuple(
        batch.residual_operator_receipt_sha256[
            date_index * len(M03R_V12_HORIZONS) + horizon_index
        ]
        for date_index in range(base.predicted_mean.shape[0])
    )
    if (
        base.split != "qualification"
        or base.fold_index != loaded.fold_index
        or batch.setting.setting_index != loaded.setting_index
        or base.asset_axis_sha256 != loaded.asset_axis_sha256
        or trace.setting_index != loaded.setting_index
        or trace.fold_index != loaded.fold_index
        or trace.selected_horizon_sessions != loaded.selected_horizon_sessions
        or trace.checkpoint_file_sha256 != loaded.checkpoint_file_sha256
        or trace.checkpoint_model_state_sha256 != loaded.model_state_sha256
        or trace.source_receipt_sha256 != base.source_array_sha256
        or trace.asset_axis_sha256 != loaded.asset_axis_sha256
        or trace.signal_operator_receipt_sha256 != selected_operator_receipts
        or trace.policy_gross_returns.numel() != base.predicted_mean.shape[0]
    ):
        raise M03RV12QualificationRuntimeError(
            "v12 qualification batch, trace, and loaded checkpoint drifted"
        )

    date_ic: list[float] = []
    date_spread: list[float] = []
    prediction_std: list[float] = []
    target_std: list[float] = []
    for date_index in range(base.predicted_mean.shape[0]):
        valid = base.valid[date_index, :, horizon_index]
        if int(valid.sum()) < 2:
            raise M03RV12QualificationRuntimeError(
                "v12 selected qualification horizon has an unsupported date"
            )
        prediction = base.predicted_mean[date_index, valid, horizon_index]
        target = base.target_log_return[date_index, valid, horizon_index]
        date_ic.append(_spearman(prediction, target))
        order = torch.argsort(prediction, stable=True)
        decile = max(1, int(order.numel()) // 10)
        date_spread.append(
            float(
                target.index_select(0, order[-decile:]).mean()
                - target.index_select(0, order[:decile]).mean()
            )
        )
        prediction_std.append(float(prediction.to(torch.float64).std(unbiased=False)))
        target_std.append(float(target.to(torch.float64).std(unbiased=False)))
    if not all(
        math.isfinite(value)
        for value in (*date_ic, *date_spread, *prediction_std, *target_std)
    ):
        raise M03RV12QualificationRuntimeError(
            "v12 fold diagnostic contains a non-finite value"
        )
    evidence = build_m03r_v12_fold_evidence(
        setting_index=loaded.setting_index,
        fold_index=loaded.fold_index,
        horizon_sessions=loaded.selected_horizon_sessions,
        score_session_index=base.origin_indices.detach().to(
            device="cpu", dtype=torch.int64
        ),
        gross_active_return=(
            trace.policy_gross_returns - trace.benchmark_gross_returns
        ),
        policy_one_way_turnover=trace.policy_one_way_turnover,
        benchmark_one_way_turnover=trace.benchmark_one_way_turnover,
        top_bottom_spread=torch.tensor(date_spread, dtype=torch.float64),
        requested_to_executed_retention=(trace.requested_to_executed_retention),
        mean_spearman_rank_ic=sum(date_ic) / len(date_ic),
        median_spearman_rank_ic=float(torch.tensor(date_ic).median()),
        positive_ic_date_fraction=(
            sum(value > 0.0 for value in date_ic) / len(date_ic)
        ),
        mean_prediction_cross_sectional_std=(sum(prediction_std) / len(prediction_std)),
        mean_target_cross_sectional_std=sum(target_std) / len(target_std),
        checkpoint_file_sha256=loaded.checkpoint_file_sha256,
        episode_schedule_sha256=loaded.episode_schedule_sha256,
        residual_operator_root_sha256=_sha256(batch.residual_operator_receipt_sha256),
    )
    result = M03RV12FoldQualificationLineage(
        loaded_checkpoint=loaded,
        fold_evidence=evidence,
        evaluation_trace_sha256=trace.trace_sha256,
        qualification_source_array_sha256=base.source_array_sha256,
        qualification_asset_axis_sha256=base.asset_axis_sha256,
        qualification_residual_operator_root_sha256=_sha256(
            batch.residual_operator_receipt_sha256
        ),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV12RoundTripQualificationResult:
    fold_lineage_sha256: tuple[str, ...]
    qualification: M03RV12PredictiveQualification

    def validate(self) -> None:
        self.qualification.validate()
        if (
            len(self.fold_lineage_sha256) != 6
            or len(set(self.fold_lineage_sha256)) != 6
        ):
            raise M03RV12QualificationRuntimeError(
                "v12 qualification lacks six distinct round-trip lineages"
            )
        for value in self.fold_lineage_sha256:
            _digest("fold_lineage_sha256", value)


def qualify_m03r_v12_round_trip_candidate(
    lineages: tuple[M03RV12FoldQualificationLineage, ...],
    bootstrap: M03RV12BootstrapPlan,
) -> M03RV12RoundTripQualificationResult:
    if len(lineages) != 6:
        raise M03RV12QualificationRuntimeError(
            "v12 qualification requires six round-trip fold lineages"
        )
    ordered = tuple(sorted(lineages, key=lambda row: row.fold_evidence.fold_index))
    for row in ordered:
        row.validate()
    if tuple(row.fold_evidence.fold_index for row in ordered) != tuple(range(6)):
        raise M03RV12QualificationRuntimeError(
            "v12 round-trip fold lineage is incomplete"
        )
    qualification = qualify_m03r_v12_predictive_candidate(
        tuple(row.fold_evidence for row in ordered),
        bootstrap,
    )
    result = M03RV12RoundTripQualificationResult(
        fold_lineage_sha256=tuple(row.receipt_sha256 for row in ordered),
        qualification=qualification,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V12_FOLD_QUALIFICATION_LINEAGE_SCHEMA",
    "M03RV12FoldQualificationLineage",
    "M03RV12QualificationRuntimeError",
    "M03RV12RoundTripQualificationResult",
    "build_m03r_v12_fold_qualification_lineage",
    "qualify_m03r_v12_round_trip_candidate",
]
