"""Checkpoint-owned fold qualification authority for M03R-v16.

The public qualification boundary rebuilds states and scores from the exact
reloaded policy.  A caller cannot pair an arbitrary score batch with a valid
checkpoint digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
    Top2000M03RV7DecisionStateProvider,
    top2000_m03r_v7_decision_inputs,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v9_projection import M03RV9DeviceRiskState
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v16_checkpoint import (
    M03RV16LoadedEpochCheckpoint,
)
from rl_quant.training.top2000_m03r_v16_cohort_runtime import (
    M03RV16CohortTrace,
    run_m03r_v16_horizon_matched_cohort_sleeve,
)
from rl_quant.training.top2000_m03r_v16_fold import M03RV16FoldGeometry
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    M03RV16BuiltPredictiveBatch,
    build_m03r_v16_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    M03RV16ValidatedStructuralSlab,
)
from rl_quant.training.top2000_m03r_v16_training_runtime import (
    move_and_bind_m03r_v16_sequence,
)

M03R_V16_QUALIFIED_SCORE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-qualified-score-authority-v1"
)
_QUALIFIED_SCORE_ISSUER = object()


class M03RV16QualificationRuntimeError(ValueError):
    """The reloaded checkpoint, cache, score, slab, or risk lineage drifted."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV16QualifiedScoreAuthority:
    """Privately issued proof of checkpoint-owned score reconstruction."""

    batch: M03RV16BuiltPredictiveBatch
    checkpoint_model_state_sha256: str
    input_state_sha256: str
    raw_score_sha256: str
    executable_score_sha256: str
    action_operator_receipts: tuple[str, ...]
    receipt_sha256: str
    _issuer: object = field(repr=False)
    projection_recomputed: bool = True
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_QUALIFIED_SCORE_SCHEMA

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "batch_receipt_sha256": self.batch.receipt_sha256,
            "checkpoint_model_state_sha256": self.checkpoint_model_state_sha256,
            "input_state_sha256": self.input_state_sha256,
            "raw_score_sha256": self.raw_score_sha256,
            "executable_score_sha256": self.executable_score_sha256,
            "action_operator_receipts": self.action_operator_receipts,
            "action_valid_sha256": _tensor_sha256(self.batch.action_valid),
            "diagnostic_valid_sha256": _tensor_sha256(
                self.batch.objective.selection_valid
            ),
            "projection_recomputed": self.projection_recomputed,
        }

    def validate(self) -> None:
        self.batch.validate()
        if (
            self._issuer is not _QUALIFIED_SCORE_ISSUER
            or self.batch.split != "qualification"
            or self.batch.policy_state_binding_kind != "model-state-sha256"
            or self.batch.policy_state_binding_sha256
            != self.checkpoint_model_state_sha256
            or self.raw_score_sha256 != _tensor_sha256(self.batch.raw_selection_score_z)
            or self.executable_score_sha256
            != _tensor_sha256(self.batch.objective.executable_selection_score_z)
            or self.action_operator_receipts
            != tuple(row.receipt_sha256 for row in self.batch.action_operators)
            or not self.projection_recomputed
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_QUALIFIED_SCORE_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV16QualificationRuntimeError(
                "V16 qualified score authority drifted"
            )


@dataclass(frozen=True, slots=True)
class M03RV16FoldQualificationResult:
    loaded_checkpoint: M03RV16LoadedEpochCheckpoint
    score_authority: M03RV16QualifiedScoreAuthority
    trace: M03RV16CohortTrace

    def validate(self) -> None:
        self.loaded_checkpoint.validate()
        self.score_authority.validate()
        self.trace.validate()
        batch = self.score_authority.batch
        if (
            batch.fold_index != self.loaded_checkpoint.fold_index
            or batch.objective.setting.setting_index
            != self.loaded_checkpoint.setting_index
            or self.trace.fold_index != self.loaded_checkpoint.fold_index
            or self.trace.setting_index != self.loaded_checkpoint.setting_index
            or self.trace.checkpoint_file_sha256
            != self.loaded_checkpoint.checkpoint_file_sha256
            or self.trace.checkpoint_model_state_sha256
            != self.loaded_checkpoint.model_state_sha256
            or self.trace.qualification_batch_receipt_sha256 != batch.receipt_sha256
        ):
            raise M03RV16QualificationRuntimeError(
                "V16 fold qualification receipt chain drifted"
            )


def _issue_score_authority(
    checkpoint: M03RV16LoadedEpochCheckpoint,
    batch: M03RV16BuiltPredictiveBatch,
    origin_states: torch.Tensor,
    structural_slab: M03RV16ValidatedStructuralSlab,
) -> M03RV16QualifiedScoreAuthority:
    recomputed: list[torch.Tensor] = []
    for row, origin in enumerate(batch.origin_indices.tolist()):
        device_operator = structural_slab.device_origin(
            origin, batch.raw_selection_score_z.device
        ).action_operator
        projected, _error = device_operator.apply(batch.raw_selection_score_z[row])
        recomputed.append(projected)
    recomputed_score = torch.stack(recomputed)
    if not torch.equal(
        recomputed_score,
        batch.objective.executable_selection_score_z,
    ):
        raise M03RV16QualificationRuntimeError(
            "V16 qualification executable score was not reproduced"
        )
    provisional = M03RV16QualifiedScoreAuthority(
        batch=batch,
        checkpoint_model_state_sha256=checkpoint.model_state_sha256,
        input_state_sha256=_tensor_sha256(origin_states),
        raw_score_sha256=_tensor_sha256(batch.raw_selection_score_z),
        executable_score_sha256=_tensor_sha256(
            batch.objective.executable_selection_score_z
        ),
        action_operator_receipts=tuple(
            row.receipt_sha256 for row in batch.action_operators
        ),
        receipt_sha256="0" * 64,
        _issuer=_QUALIFIED_SCORE_ISSUER,
    )
    result = replace(
        provisional,
        receipt_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


def run_m03r_v16_fold_qualification(
    cache: Top2000VerifiedDevelopmentCache,
    geometry: M03RV16FoldGeometry,
    risk_source: M03RV9MaterializedRiskSource,
    risk_state: M03RV9DeviceRiskState,
    structural_slab: M03RV16ValidatedStructuralSlab,
    policy: Top2000M03RV16PredictivePolicy,
    checkpoint: M03RV16LoadedEpochCheckpoint,
    *,
    device: torch.device,
) -> M03RV16FoldQualificationResult:
    """Rebuild one exact qualification path from the reloaded checkpoint."""

    cache.validate_unmodified()
    geometry.validate()
    risk_source.validate()
    risk_state.validate()
    structural_slab.require_fast_identity()
    checkpoint.validate()
    spec = M03R_V16_PREDICTIVE_SPEC
    steps = spec.qualification_origins_per_fold + (
        spec.cohort_no_new_decision_tail_sessions
    )
    execution_origins = tuple(
        range(
            geometry.qualification_origin_start_inclusive,
            geometry.qualification_origin_start_inclusive + steps,
        )
    )
    setting = M03R_V16_SETTINGS[checkpoint.setting_index]
    if (
        checkpoint.fold_index != geometry.fold_index
        or policy.v16_setting != setting
        or model_state_sha256(policy) != checkpoint.model_state_sha256
        or checkpoint.asset_axis_sha256 != cache.action_hash
        or risk_source.cache_sha256 != cache.cache_sha256
        or risk_source.action_hash != cache.action_hash
        or risk_state.asset_axis_sha256 != cache.action_hash
        or tuple(risk_state.origin_state_indices) != execution_origins
        or risk_state.source_exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
        or structural_slab.receipt.cache_sha256 != cache.cache_sha256
        or structural_slab.receipt.asset_axis_sha256 != cache.action_hash
        or structural_slab.receipt.risk_source_receipt_sha256
        != risk_source.receipt_sha256
        or structural_slab.receipt.exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
        or checkpoint.selection_target_operator_root_sha256
        != structural_slab.receipt.common_target_operator_root_sha256
        or checkpoint.action_operator_root_sha256
        != structural_slab.receipt.action_operator_root_sha256
    ):
        raise M03RV16QualificationRuntimeError(
            "V16 qualification cache, checkpoint, slab, or risk identity drifted"
        )

    episode_stop = geometry.qualification_target_stop_exclusive
    episode_start = episode_stop - spec.episode_state_rows
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=episode_start,
        state_stop_index_exclusive=episode_stop,
        max_state_rows=spec.episode_state_rows,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = move_and_bind_m03r_v16_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    global_origins = torch.arange(
        geometry.qualification_origin_start_inclusive,
        geometry.qualification_origin_stop_exclusive,
        dtype=torch.long,
        device=device,
    )
    local_origins = global_origins - episode_start
    inputs = top2000_m03r_v7_decision_inputs(sequence)
    bound_sequence = replace(
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
            bound_sequence,
            local_origins,
        )
        batch = build_m03r_v16_batch_from_origin_states(
            policy,
            setting,
            origin_states,
            bound_sequence,
            local_origins,
            sequence_global_state_start=episode_start,
            split="qualification",
            split_start_inclusive=geometry.qualification_origin_start_inclusive,
            split_stop_exclusive=geometry.qualification_target_stop_exclusive,
            fold_index=geometry.fold_index,
            source_array_sha256=built.identity.receipt_sha256,
            asset_axis_sha256=cache.action_hash,
            origin_risk_exposures=risk_source.exposures,
            structural_slab=structural_slab,
        )
    score_authority = _issue_score_authority(
        checkpoint,
        batch,
        origin_states,
        structural_slab,
    )

    execution_global = torch.tensor(
        execution_origins,
        dtype=torch.long,
        device=device,
    )
    fill_indices = execution_global - episode_start + 1
    fill_available = (
        bound_sequence.fill_membership.index_select(0, fill_indices)[:, 0]
        & bound_sequence.fill_availability.index_select(0, fill_indices)[:, 0]
    )
    fill_available = fill_available.clone()
    fill_available[:, bound_sequence.initial_ledger.cash_index] = False
    trace = run_m03r_v16_horizon_matched_cohort_sleeve(
        setting,
        fold_index=checkpoint.fold_index,
        checkpoint_file_sha256=checkpoint.checkpoint_file_sha256,
        checkpoint_model_state_sha256=checkpoint.model_state_sha256,
        qualification_batch_receipt_sha256=batch.receipt_sha256,
        asset_axis_sha256=batch.asset_axis_sha256,
        decision_origin_indices=batch.origin_indices,
        executable_selection_scores=(
            batch.objective.executable_selection_score_z
            * setting.selection_target_scale
        ),
        action_valid=batch.action_valid,
        diagnostic_valid=batch.objective.selection_valid,
        post_fill_asset_returns=bound_sequence.asset_returns.index_select(
            0, fill_indices
        )[:, 0],
        benchmark_weights=bound_sequence.benchmark_weights.index_select(
            0, fill_indices
        )[:, 0],
        fill_available=fill_available,
        risk_asset_caps=bound_sequence.risk_asset_caps.index_select(0, fill_indices)[
            :, 0
        ],
        risk_gross_max=bound_sequence.risk_gross_max.index_select(0, fill_indices)[
            :, 0
        ],
        risk_state=risk_state,
    )
    result = M03RV16FoldQualificationResult(
        loaded_checkpoint=checkpoint,
        score_authority=score_authority,
        trace=trace,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V16_QUALIFIED_SCORE_SCHEMA",
    "M03RV16FoldQualificationResult",
    "M03RV16QualificationRuntimeError",
    "M03RV16QualifiedScoreAuthority",
    "run_m03r_v16_fold_qualification",
]
