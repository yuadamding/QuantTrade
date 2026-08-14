"""Exact-checkpoint fold adapter for the M03R-v11 a15 post-hoc audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import torch

from rl_quant.evaluation.top2000_m03r_v11_a15_inference_audit import (
    M03RV11A15AuditFoldEvidence,
    build_m03r_v11_a15_audit_fold_evidence,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
    M03RV11A15AuditVariant,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_HORIZONS,
    resolve_m03r_v11_setting,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
    Top2000M03RV7DecisionStateProvider,
    Top2000M03RV7DevelopmentFold,
    top2000_m03r_v7_decision_inputs,
)
from rl_quant.training.top2000_m03r_v9_fold import M03R_V9_QUALIFICATION_ORIGINS
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9AlphaDistribution,
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256
from rl_quant.training.top2000_m03r_v9_projection import M03RV9DeviceRiskState
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_runtime import (
    M03RV11A15AuditReplayTrace,
    run_m03r_v11_a15_inference_audit_replay,
)
from rl_quant.training.top2000_m03r_v11_checkpoint import M03RV11LoadedCheckpoint
from rl_quant.training.top2000_m03r_v11_fold_qualification import (
    _move_sequence,
    _slice_sleeve_sequence,
)
from rl_quant.training.top2000_m03r_v11_predictive_worker import (
    M03RV11PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v11_pretraining_runtime import (
    build_m03r_v11_alpha_batch_from_origin_states,
)

M03R_V11_A15_AUDIT_FOLD_RESULT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-fold-result-v2"
)
M03R_V11_A15_AUDIT_RISK_SEMANTIC_LINEAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-risk-semantic-lineage-v1"
)


class M03RV11A15InferenceAuditFoldError(ValueError):
    """The exact checkpoint, fold context, or audit output drifted."""


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


def m03r_v11_a15_audit_risk_semantic_lineage_sha256(
    risk_state: M03RV9DeviceRiskState,
) -> str:
    """Bind exact risk inputs while leaving derived cross-node FP bytes explicit.

    The factor-plus-diagonal tensors are recomputed from exact immutable inputs.
    Their byte hash remains evidence, but it is not a portable identity across
    CPU implementations or Kubernetes nodes.
    """

    risk_state.validate()
    return _sha256(
        {
            "schema": M03R_V11_A15_AUDIT_RISK_SEMANTIC_LINEAGE_SCHEMA,
            "risk_state_schema": risk_state.schema,
            "asset_count": risk_state.asset_count,
            "cash_index": risk_state.cash_index,
            "origin_state_indices": risk_state.origin_state_indices,
            "projector_manifest_sha256": risk_state.manifest_sha256,
            "source_binding_sha256": risk_state.source_binding_sha256,
            "source_exposure_receipt_sha256": (
                risk_state.source_exposure_receipt_sha256
            ),
            "daily_returns_receipt_sha256": risk_state.daily_returns_receipt_sha256,
            "asset_axis_sha256": risk_state.asset_axis_sha256,
            "construction_rule": (
                "exact-input-factor-plus-diagonal-cross-node-recomputation-v1"
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditFoldResult:
    setting_index: int
    fold_index: int
    horizon_sessions: int
    checkpoint_file_sha256: str
    checkpoint_model_state_sha256: str
    qualification_source_array_sha256: str
    qualification_asset_axis_sha256: str
    qualification_residual_operator_root_sha256: str
    parent_fold_risk_state_sha256: str
    audit_fold_risk_state_sha256: str
    audit_risk_semantic_lineage_sha256: str
    parent_fold_risk_state_byte_match: bool
    variant_trace_sha256: tuple[tuple[str, str], ...]
    variant_fold_evidence_sha256: tuple[tuple[str, str], ...]
    receipt_sha256: str
    training_performed: bool = False
    checkpoint_selection_performed: bool = False
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V11_A15_AUDIT_FOLD_RESULT_SCHEMA

    def unsigned_payload(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        digests = (
            self.checkpoint_file_sha256,
            self.checkpoint_model_state_sha256,
            self.qualification_source_array_sha256,
            self.qualification_asset_axis_sha256,
            self.qualification_residual_operator_root_sha256,
            self.parent_fold_risk_state_sha256,
            self.audit_fold_risk_state_sha256,
            self.audit_risk_semantic_lineage_sha256,
            *(value for _, value in self.variant_trace_sha256),
            *(value for _, value in self.variant_fold_evidence_sha256),
        )
        if (
            self.setting_index not in (0, 1)
            or self.fold_index not in range(6)
            or self.horizon_sessions not in (21, 30)
            or len(self.variant_trace_sha256) == 0
            or tuple(name for name, _ in self.variant_trace_sha256)
            != tuple(name for name, _ in self.variant_fold_evidence_sha256)
            or len(set(name for name, _ in self.variant_trace_sha256))
            != len(self.variant_trace_sha256)
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or self.parent_fold_risk_state_byte_match
            != (self.parent_fold_risk_state_sha256 == self.audit_fold_risk_state_sha256)
            or self.training_performed
            or self.checkpoint_selection_performed
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V11_A15_AUDIT_FOLD_RESULT_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditFoldError("a15 audit fold result drifted")


def evaluate_m03r_v11_a15_loaded_audit_fold(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV11PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    risk_state: M03RV9DeviceRiskState,
    policy: Top2000M03RV9PredictivePolicy,
    loaded: M03RV11LoadedCheckpoint,
    variants: tuple[M03RV11A15AuditVariant, ...],
    *,
    expected_parent_fold_risk_state_sha256: str,
    device: torch.device,
) -> tuple[
    M03RV11A15AuditFoldResult,
    tuple[M03RV11A15AuditReplayTrace, ...],
    tuple[M03RV11A15AuditFoldEvidence, ...],
]:
    """Replay predeclared controls from exact reloaded update-64 bytes."""

    worker.validate()
    cache.validate_unmodified()
    risk_source.validate()
    risk_state.validate()
    if len(expected_parent_fold_risk_state_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_parent_fold_risk_state_sha256
    ):
        raise M03RV11A15InferenceAuditFoldError(
            "a15 audit parent fold risk-state hash is malformed"
        )
    if not variants or len({row.variant_id for row in variants}) != len(variants):
        raise M03RV11A15InferenceAuditFoldError(
            "a15 audit variants are empty or duplicated"
        )
    for variant in variants:
        variant.validate()
    geometry = render_m03r_v10_fold_geometry(fold)
    expected_origins = tuple(
        range(
            geometry.qualification_start_inclusive,
            geometry.qualification_origin_stop_exclusive,
        )
    )
    if (
        loaded.setting_index != worker.setting_index
        or loaded.setting_index not in (0, 1)
        or loaded.setting_id != worker.setting_id
        or loaded.fold_index != fold.fold_index
        or loaded.episode_schedule_sha256 != worker.panel_episode_schedule_sha256
        or loaded.asset_axis_sha256 != cache.action_hash
        or worker.cache_sha256 != cache.cache_sha256
        or state_dict_sha256(policy.state_dict()) != loaded.model_state_sha256
        or policy.horizon_binding.checkpoint_selection_horizon
        != loaded.selected_horizon_sessions
        or policy.horizon_binding.qualification_horizon
        != loaded.selected_horizon_sessions
        or policy.horizon_binding.economic_execution_horizon
        != loaded.selected_horizon_sessions
        or risk_state.asset_axis_sha256 != cache.action_hash
        or risk_state.source_exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
        or risk_state.origin_state_indices != expected_origins
    ):
        raise M03RV11A15InferenceAuditFoldError(
            "a15 audit checkpoint, worker, cache, or risk identity drifted"
        )
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=geometry.qualification_episode_state_start,
        state_stop_index_exclusive=(
            geometry.qualification_episode_state_start
            + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
        ),
        max_state_rows=TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = _move_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    global_origins = torch.arange(
        geometry.qualification_start_inclusive,
        geometry.qualification_origin_stop_exclusive,
        dtype=torch.long,
        device=device,
    )
    local_origins = global_origins - geometry.qualification_episode_state_start
    inputs = top2000_m03r_v7_decision_inputs(sequence)
    bound = replace(
        sequence,
        decision_state=torch.zeros(
            (*sequence.decision_state.shape[:3], 1),
            dtype=sequence.decision_state.dtype,
            device=sequence.decision_state.device,
        ),
    )
    provider = Top2000M03RV7DecisionStateProvider(inputs)
    policy.eval()
    with torch.no_grad():
        origin_states = provider.replay_origin_states(
            policy.source_policy,
            bound,
            local_origins,
        )
        batch = build_m03r_v11_alpha_batch_from_origin_states(
            policy,
            resolve_m03r_v11_setting(worker.setting_index),
            origin_states,
            bound,
            local_origins,
            sequence_global_state_start=geometry.qualification_episode_state_start,
            split="qualification",
            split_start_inclusive=geometry.qualification_start_inclusive,
            split_stop_exclusive=geometry.qualification_target_stop_exclusive,
            fold_index=fold.fold_index,
            source_array_sha256=built.identity.receipt_sha256,
            asset_axis_sha256=cache.action_hash,
            origin_risk_exposures=risk_source.exposures,
        )
    if batch.residual_operators is None:
        raise M03RV11A15InferenceAuditFoldError(
            "a15 audit batch omitted executable residual operators"
        )
    horizon_index = M03R_V11_HORIZONS.index(loaded.selected_horizon_sessions)
    distributions = tuple(
        M03RV9AlphaDistribution(
            mean_by_horizon=batch.corrected_batch.predicted_mean[index].unsqueeze(0),
            log_scale_by_horizon=(
                batch.corrected_batch.predicted_log_scale[index].unsqueeze(0)
            ),
            selected_horizon_sessions=loaded.selected_horizon_sessions,
            selected_mean=batch.corrected_batch.predicted_mean[
                index, :, horizon_index
            ].unsqueeze(0),
            selected_scale=torch.exp(
                batch.corrected_batch.predicted_log_scale[index, :, horizon_index]
            ).unsqueeze(0),
        )
        for index in range(M03R_V9_QUALIFICATION_ORIGINS)
    )
    selected_operators = tuple(
        batch.residual_operators[index * len(M03R_V11_HORIZONS) + horizon_index]
        for index in range(M03R_V9_QUALIFICATION_ORIGINS)
    )
    local_start = (
        geometry.qualification_start_inclusive
        - geometry.qualification_episode_state_start
    )
    local_stop = local_start + M03R_V9_QUALIFICATION_ORIGINS + 1
    sleeve_sequence = _slice_sleeve_sequence(
        sequence,
        local_start=local_start,
        local_stop_exclusive=local_stop,
    )
    traces = tuple(
        run_m03r_v11_a15_inference_audit_replay(
            sleeve_sequence,
            distributions,
            selected_operators,
            risk_state,
            variant,
            setting_index=worker.setting_index,
            fold_index=fold.fold_index,
            selected_horizon_sessions=loaded.selected_horizon_sessions,
            state_start_index=geometry.qualification_start_inclusive,
            checkpoint_file_sha256=loaded.checkpoint_file_sha256,
            checkpoint_model_state_sha256=loaded.model_state_sha256,
            checkpoint_asset_axis_sha256=loaded.asset_axis_sha256,
            source_receipt_sha256=built.identity.receipt_sha256,
            benchmark_gross_returns=(
                built.benchmark.gross_returns[local_start : local_stop - 1].to(device)
            ),
            benchmark_one_way_turnover=(
                built.benchmark.total_one_way_turnover[local_start : local_stop - 1].to(
                    device
                )
            ),
        )
        for variant in variants
    )
    evidence = tuple(
        build_m03r_v11_a15_audit_fold_evidence(
            trace,
            score_session_index=batch.corrected_batch.origin_indices.detach().to(
                device="cpu", dtype=torch.int64
            ),
            target_log_return=batch.corrected_batch.target_log_return[
                :, :, horizon_index
            ]
            .detach()
            .to(device="cpu", dtype=torch.float64),
            valid=batch.corrected_batch.valid[:, :, horizon_index]
            .detach()
            .to(device="cpu"),
            target_source_array_sha256=batch.corrected_batch.source_array_sha256,
        )
        for trace in traces
    )
    residual_root = _sha256(batch.residual_operator_receipt_sha256)
    risk_semantic_lineage = m03r_v11_a15_audit_risk_semantic_lineage_sha256(risk_state)
    provisional = M03RV11A15AuditFoldResult(
        setting_index=worker.setting_index,
        fold_index=fold.fold_index,
        horizon_sessions=loaded.selected_horizon_sessions,
        checkpoint_file_sha256=loaded.checkpoint_file_sha256,
        checkpoint_model_state_sha256=loaded.model_state_sha256,
        qualification_source_array_sha256=(batch.corrected_batch.source_array_sha256),
        qualification_asset_axis_sha256=batch.corrected_batch.asset_axis_sha256,
        qualification_residual_operator_root_sha256=residual_root,
        parent_fold_risk_state_sha256=expected_parent_fold_risk_state_sha256,
        audit_fold_risk_state_sha256=risk_state.state_sha256,
        audit_risk_semantic_lineage_sha256=risk_semantic_lineage,
        parent_fold_risk_state_byte_match=(
            expected_parent_fold_risk_state_sha256 == risk_state.state_sha256
        ),
        variant_trace_sha256=tuple(
            (row.variant_id, row.trace_sha256) for row in traces
        ),
        variant_fold_evidence_sha256=tuple(
            (row.variant_id, row.receipt_sha256) for row in evidence
        ),
        receipt_sha256="0" * 64,
    )
    result = M03RV11A15AuditFoldResult(
        **{
            **asdict(provisional),
            "receipt_sha256": _sha256(provisional.unsigned_payload()),
        }
    )
    result.validate()
    return result, traces, evidence


__all__ = [
    "M03R_V11_A15_AUDIT_FOLD_RESULT_SCHEMA",
    "M03R_V11_A15_AUDIT_RISK_SEMANTIC_LINEAGE_SCHEMA",
    "M03RV11A15AuditFoldResult",
    "M03RV11A15InferenceAuditFoldError",
    "evaluate_m03r_v11_a15_loaded_audit_fold",
    "m03r_v11_a15_audit_risk_semantic_lineage_sha256",
]
