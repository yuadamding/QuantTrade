"""Execute one canonical V2 validation fold through policy selection V3.

This is the package-owned orchestration boundary for inner validation.  The
caller supplies experiment authorities and a fold index, never environments,
actions, targets, metrics, candidates, or a selected checkpoint.  Existing V1
objects remain exact computational witnesses; only their V2 envelopes and the
resulting Selection V3 authority are returned as authorizing evidence.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import stat
from typing import Iterator

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256,
    MassiveAdaptiveRLCostLadderAuthorityV1,
    authorize_massive_adaptive_rl_cost_ladder_authority_v1,
    materialize_massive_adaptive_rl_cost_ladder_authority_v1,
    parse_massive_adaptive_rl_cost_ladder_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_validation_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
    MassiveAdaptiveRLFixedControlValidationAuthorityV1,
    authorize_massive_adaptive_rl_fixed_control_validation_authority_v1,
    materialize_massive_adaptive_rl_fixed_control_validation_authority_v1,
    parse_massive_adaptive_rl_fixed_control_validation_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
    MassiveAdaptiveRLFoldValidationAuthorityV1,
    authorize_massive_adaptive_rl_fold_validation_authority_v1,
    fold_validation_authority_relative_path_v1,
    materialize_massive_adaptive_rl_fold_validation_authority_v1,
    parse_massive_adaptive_rl_fold_validation_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v1 import (
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SHA256,
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    validate_massive_adaptive_rl_validation_outcome_barrier_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_trace_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256,
    MassiveAdaptiveRLPolicyTraceAuthorityV1,
    authorize_massive_adaptive_rl_policy_trace_authority_v1,
    materialize_massive_adaptive_rl_policy_trace_authority_v1,
    parse_massive_adaptive_rl_policy_trace_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_evidence_v2 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256,
    MassiveAdaptiveRLFoldValidationAuthorityV2,
    MassiveAdaptiveRLValidationOutcomeAuthorityV2,
    authorize_massive_adaptive_rl_fold_validation_authority_v2,
    authorize_massive_adaptive_rl_validation_outcome_authority_v2,
    fold_validation_authority_relative_path_v2,
    load_massive_adaptive_rl_fold_validation_authority_v2,
    load_massive_adaptive_rl_validation_outcome_authority_v2,
    materialize_massive_adaptive_rl_fold_validation_authority_v2,
    materialize_massive_adaptive_rl_validation_outcome_authority_v2,
    validation_outcome_authority_relative_path_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    MassiveAdaptiveRLValidationSourcesAuthorityV1,
    validation_cost_ladder_artifact_id_v1,
    validation_cost_ladder_relative_path_v1,
    validation_fixed_control_artifact_id_v1,
    validation_fixed_control_relative_path_v1,
    validation_primary_trace_artifact_id_v1,
    validation_primary_trace_relative_path_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v2 import (
    MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    MassiveAdaptiveRLValidationSourcesAuthorityV2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v3 import (
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256,
    MassiveAdaptiveRLPolicySelectionAuthorityV3,
    policy_selection_authority_relative_path_v3,
    policy_selection_v2_witness_relative_path_v3,
    run_or_resume_massive_adaptive_rl_policy_selection_authority_v3,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_v1 import (
    MassiveAdaptiveRLFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
)
from rl_quant.workflows.massive_adaptive_rl_process_state_v1 import (
    preserve_massive_adaptive_rl_process_rng_state_v1,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility,
)


MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-runtime-sources-v2-fit-and-four-fold-validation-barrier-v2",
        "candidate_discovery": "exact-fold-fit-policy-checkpoint-inventory",
        "primary": "canonical-v1-20bp-witness-immediately-wrapped-v2",
        "ladder": "canonical-v1-10-20-40bp-witness-immediately-wrapped-v2",
        "fixed_control": "canonical-fit-selected-fc06-v1-witness-wrapped-v2",
        "fold_validation": "canonical-v1-witness-then-v2-authority",
        "selection": "canonical-v2-computation-witness-then-selection-v3",
        "execution_device": "cpu",
        "resume": "strict-canonical-prefix-replay-with-one-fold-lease",
        "read_only_replay": "no-source-or-operational-artifact-creation",
        "caller_actions": False,
        "caller_targets": False,
        "caller_metrics": False,
        "caller_candidates": False,
        "caller_selection": False,
        "policy_trace_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256
        ),
        "cost_ladder_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256
        ),
        "fixed_control_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
        ),
        "fold_validation_v1_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
        ),
        "validation_barrier_v2_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SHA256
        ),
        "validation_evidence_v2_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256
        ),
        "selection_v3_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256
        ),
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_PRIMARY = "ppo-primary"
_LADDER = "ppo-cost-ladder"
_FC06 = "fc06-primary"


class MassiveAdaptiveRLFoldValidationExecutorV1Error(ValueError):
    """The canonical fold execution is absent, mixed, late, or inconsistent."""


class MassiveAdaptiveRLFoldValidationExecutionLeaseUnavailable(
    MassiveAdaptiveRLFoldValidationExecutorV1Error
):
    """Another process owns this fold's validation execution."""


@dataclass(frozen=True, slots=True)
class _CanonicalValidationStageV1:
    name: str
    relative_path: str
    commit_offset_ms: int


@dataclass(frozen=True, slots=True)
class _FoldExecutionRootsV1:
    fold_fit: MassiveAdaptiveRLFoldFitAuthorityV1
    checkpoints: tuple[MassiveAdaptiveRLCheckpointAuthorityV1, ...]
    fixed_fit: MassiveAdaptiveRLFixedControlFitAuthorityV1
    fixed_selection: MassiveAdaptiveRLFixedControlSelectionAuthorityV1


def _transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution source transaction is incomplete"
        )
    return all(present)


def _load(
    *, root: str | Path, relative: str, verified_at_ms: int
) -> LoadedMassiveSourceObject:
    return load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=verified_at_ms,
    )


def _canonical_validation_stages_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    checkpoint_authority_receipts: tuple[str, ...],
    fixed_control_selection_authority_receipt_sha256: str,
) -> tuple[_CanonicalValidationStageV1, ...]:
    """Return the one legal create/replay order and its logical timestamps."""

    stages: list[tuple[str, str]] = []
    for checkpoint_receipt in checkpoint_authority_receipts:
        stages.extend(
            (
                (
                    "primary-v1",
                    validation_primary_trace_relative_path_v1(
                        manifest=manifest,
                        fold_index=fold_index,
                        checkpoint_authority_receipt_sha256=checkpoint_receipt,
                    ),
                ),
                (
                    "primary-v2",
                    validation_outcome_authority_relative_path_v2(
                        manifest=manifest,
                        fold_index=fold_index,
                        outcome_kind=_PRIMARY,
                        subject_receipt_sha256=checkpoint_receipt,
                    ),
                ),
                (
                    "ladder-v1",
                    validation_cost_ladder_relative_path_v1(
                        manifest=manifest,
                        fold_index=fold_index,
                        checkpoint_authority_receipt_sha256=checkpoint_receipt,
                    ),
                ),
                (
                    "ladder-v2",
                    validation_outcome_authority_relative_path_v2(
                        manifest=manifest,
                        fold_index=fold_index,
                        outcome_kind=_LADDER,
                        subject_receipt_sha256=checkpoint_receipt,
                    ),
                ),
            )
        )
    stages.extend(
        (
            (
                "fc06-v1",
                validation_fixed_control_relative_path_v1(
                    manifest=manifest,
                    fold_index=fold_index,
                ),
            ),
            (
                "fc06-v2",
                validation_outcome_authority_relative_path_v2(
                    manifest=manifest,
                    fold_index=fold_index,
                    outcome_kind=_FC06,
                    subject_receipt_sha256=(
                        fixed_control_selection_authority_receipt_sha256
                    ),
                ),
            ),
            (
                "fold-validation-v1",
                fold_validation_authority_relative_path_v1(
                    manifest=manifest,
                    fold_index=fold_index,
                ),
            ),
            (
                "fold-validation-v2",
                fold_validation_authority_relative_path_v2(
                    manifest=manifest,
                    fold_index=fold_index,
                ),
            ),
            (
                "selection-v2-computation",
                policy_selection_v2_witness_relative_path_v3(
                    manifest=manifest,
                    fold_index=fold_index,
                ),
            ),
            (
                "selection-v3",
                policy_selection_authority_relative_path_v3(
                    manifest=manifest,
                    fold_index=fold_index,
                ),
            ),
        )
    )
    return tuple(
        _CanonicalValidationStageV1(
            name=name,
            relative_path=relative,
            commit_offset_ms=offset,
        )
        for offset, (name, relative) in enumerate(stages)
    )


def _execution_anchor_ms(
    *,
    root: str | Path,
    stages: tuple[_CanonicalValidationStageV1, ...],
    requested_first_commit_ms: int,
    barrier_committed_at_ms: int,
) -> int:
    """Validate a strict artifact prefix and recover its initial timestamp."""

    existing = tuple(
        _transaction_exists(root=root, relative=stage.relative_path) for stage in stages
    )
    first_missing = next(
        (index for index, value in enumerate(existing) if not value), len(existing)
    )
    if any(existing[first_missing:]):
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution cannot repair a missing upstream stage"
        )
    if not any(existing):
        anchor = requested_first_commit_ms
    else:
        first = _load(
            root=root,
            relative=stages[0].relative_path,
            verified_at_ms=requested_first_commit_ms,
        )
        anchor = first.commit.committed_at_ms
        for stage in stages[:first_missing]:
            loaded = _load(
                root=root,
                relative=stage.relative_path,
                verified_at_ms=max(requested_first_commit_ms, anchor),
            )
            if loaded.commit.committed_at_ms != anchor + stage.commit_offset_ms:
                raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                    "fold-validation execution chronology is not canonical"
                )
    if anchor <= barrier_committed_at_ms:
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution must follow the V2 input barrier"
        )
    return anchor


def _validate_execution_roots(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    fold_index: int,
) -> _FoldExecutionRootsV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(runtime_sources_v2) is not MassiveAdaptiveRLRuntimeSourcesV2
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
        or type(four_fold_validation_inputs_v2)
        is not MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
        or isinstance(fold_index, bool)
        or fold_index not in range(4)
    ):
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution requires exact V2 experiment roots"
        )
    manifest.validate()
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    four_fold_validation_inputs_v2.validate()
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    if (
        not four_fold_fit_authority.development_stage_authorized
        or not four_fold_validation_inputs_v2.development_stage_authorized
        or manifest.experiment_id != runtime_sources_v2.experiment_id
        or manifest.base_manifest.semantic_receipt_sha256
        != runtime_sources_v2.manifest_v3_receipt_sha256
        or four_fold_validation_inputs_v2.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or four_fold_validation_inputs_v2.runtime_sources_v2_receipt_sha256
        != runtime_sources_v2.semantic_receipt_sha256
        or four_fold_validation_inputs_v2.four_fold_fit_authority_receipt_sha256
        != four_fold_fit_authority.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution roots are mixed or unauthorized"
        )
    fold_fit = four_fold_fit_authority.fold_fit(fold_index)
    workflow = fold_fit.training_workflow.runtime_workflow
    checkpoints = tuple(workflow.policy_checkpoint_authorities)
    fixed_fit = workflow.fixed_control_fit_authority
    fixed_selection = workflow.fixed_control_selection_authority
    if (
        any(
            type(checkpoint) is not MassiveAdaptiveRLCheckpointAuthorityV1
            for checkpoint in checkpoints
        )
        or type(fixed_fit) is not MassiveAdaptiveRLFixedControlFitAuthorityV1
        or type(fixed_selection)
        is not MassiveAdaptiveRLFixedControlSelectionAuthorityV1
    ):
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution children have mixed generations"
        )
    expected = four_fold_validation_inputs_v2.fold_indices.index(fold_index)
    receipts = tuple(row.semantic_receipt_sha256 for row in checkpoints)
    if (
        receipts != fold_fit.candidate_checkpoint_authority_receipts
        or receipts
        != four_fold_validation_inputs_v2.expected_candidate_checkpoint_authority_receipt_inventories[
            expected
        ]
        or len(checkpoints) != fold_index + 1
    ):
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation candidate population differs"
        )
    sources_v2 = four_fold_validation_inputs_v2.validation_sources(fold_index)
    registry_v2 = four_fold_validation_inputs_v2.validation_registry(fold_index)
    if (
        not sources_v2.development_stage_authorized
        or not registry_v2.development_stage_authorized
        or sources_v2.runtime_sources_v2_receipt_sha256
        != runtime_sources_v2.semantic_receipt_sha256
        or registry_v2.runtime_sources_v2_receipt_sha256
        != runtime_sources_v2.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation V2 source generation is not authorized"
        )
    return _FoldExecutionRootsV1(
        fold_fit=fold_fit,
        checkpoints=checkpoints,
        fixed_fit=fixed_fit,
        fixed_selection=fixed_selection,
    )


def _assert_commit_time(
    *, loaded: LoadedMassiveSourceObject, expected_at_ms: int
) -> None:
    if loaded.commit.committed_at_ms != expected_at_ms:
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution chronology differs"
        )


def _primary_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    checkpoint: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology: MassiveAdaptiveRLChronologyAuthorityV1,
    registry_v1: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    barrier_v1: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    fold_index: int,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLPolicyTraceAuthorityV1:
    relative = validation_primary_trace_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
        checkpoint_authority_receipt_sha256=checkpoint.semantic_receipt_sha256,
    )
    if _transaction_exists(root=root, relative=relative):
        result = authorize_massive_adaptive_rl_policy_trace_authority_v1(
            root=root,
            authority=parse_massive_adaptive_rl_policy_trace_authority_v1(
                root=root,
                loaded_source=_load(
                    root=root,
                    relative=relative,
                    verified_at_ms=committed_at_ms,
                ),
            ),
            checkpoint_authority=checkpoint,
            chronology_authority=chronology,
            validation_environment_registry=registry_v1,
            four_fold_validation_inputs_authority=barrier_v1,
            device="cpu",
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "canonical PPO primary validation evidence is absent"
            )
        result = materialize_massive_adaptive_rl_policy_trace_authority_v1(
            root=root,
            artifact_id=validation_primary_trace_artifact_id_v1(
                manifest=manifest,
                fold_index=fold_index,
                checkpoint_authority_receipt_sha256=(
                    checkpoint.semantic_receipt_sha256
                ),
            ),
            checkpoint_authority=checkpoint,
            chronology_authority=chronology,
            fold_index=fold_index,
            evaluation_role="inner_validation",
            committed_at_ms=committed_at_ms,
            validation_environment_registry=registry_v1,
            four_fold_validation_inputs_authority=barrier_v1,
            device="cpu",
        )
    _assert_commit_time(loaded=result.loaded_source, expected_at_ms=committed_at_ms)
    return result


def _ladder_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    checkpoint: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology: MassiveAdaptiveRLChronologyAuthorityV1,
    registry_v1: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    barrier_v1: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    fold_index: int,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLCostLadderAuthorityV1:
    relative = validation_cost_ladder_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
        checkpoint_authority_receipt_sha256=checkpoint.semantic_receipt_sha256,
    )
    if _transaction_exists(root=root, relative=relative):
        result = authorize_massive_adaptive_rl_cost_ladder_authority_v1(
            root=root,
            authority=parse_massive_adaptive_rl_cost_ladder_authority_v1(
                root=root,
                loaded_source=_load(
                    root=root,
                    relative=relative,
                    verified_at_ms=committed_at_ms,
                ),
            ),
            checkpoint_authority=checkpoint,
            chronology_authority=chronology,
            validation_environment_registry=registry_v1,
            four_fold_validation_inputs_authority=barrier_v1,
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "canonical PPO validation cost ladder is absent"
            )
        result = materialize_massive_adaptive_rl_cost_ladder_authority_v1(
            root=root,
            artifact_id=validation_cost_ladder_artifact_id_v1(
                manifest=manifest,
                fold_index=fold_index,
                checkpoint_authority_receipt_sha256=(
                    checkpoint.semantic_receipt_sha256
                ),
            ),
            checkpoint_authority=checkpoint,
            chronology_authority=chronology,
            fold_index=fold_index,
            evaluation_role="inner_validation",
            committed_at_ms=committed_at_ms,
            validation_environment_registry=registry_v1,
            four_fold_validation_inputs_authority=barrier_v1,
        )
    _assert_commit_time(loaded=result.loaded_source, expected_at_ms=committed_at_ms)
    return result


def _fixed_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fit: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    selection: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology: MassiveAdaptiveRLChronologyAuthorityV1,
    registry_v1: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    barrier_v1: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    fold_index: int,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLFixedControlValidationAuthorityV1:
    relative = validation_fixed_control_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    if _transaction_exists(root=root, relative=relative):
        result = authorize_massive_adaptive_rl_fixed_control_validation_authority_v1(
            root=root,
            authority=parse_massive_adaptive_rl_fixed_control_validation_authority_v1(
                root=root,
                loaded_source=_load(
                    root=root,
                    relative=relative,
                    verified_at_ms=committed_at_ms,
                ),
            ),
            fit_authority=fit,
            selection_authority=selection,
            chronology_authority=chronology,
            validation_environment_registry=registry_v1,
            four_fold_validation_inputs_authority=barrier_v1,
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "canonical FC06 validation evidence is absent"
            )
        result = materialize_massive_adaptive_rl_fixed_control_validation_authority_v1(
            root=root,
            artifact_id=validation_fixed_control_artifact_id_v1(
                manifest=manifest,
                fold_index=fold_index,
            ),
            fit_authority=fit,
            selection_authority=selection,
            chronology_authority=chronology,
            committed_at_ms=committed_at_ms,
            validation_environment_registry=registry_v1,
            four_fold_validation_inputs_authority=barrier_v1,
        )
    _assert_commit_time(loaded=result.loaded_source, expected_at_ms=committed_at_ms)
    return result


def _outcome_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    base_outcome: (
        MassiveAdaptiveRLPolicyTraceAuthorityV1
        | MassiveAdaptiveRLCostLadderAuthorityV1
        | MassiveAdaptiveRLFixedControlValidationAuthorityV1
    ),
    sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    barrier_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    outcome_kind: str,
    subject_receipt_sha256: str,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV2:
    relative = validation_outcome_authority_relative_path_v2(
        manifest=manifest,
        fold_index=base_outcome.fold_index,
        outcome_kind=outcome_kind,
        subject_receipt_sha256=subject_receipt_sha256,
    )
    if _transaction_exists(root=root, relative=relative):
        result = authorize_massive_adaptive_rl_validation_outcome_authority_v2(
            authority=load_massive_adaptive_rl_validation_outcome_authority_v2(
                root=root,
                manifest=manifest,
                fold_index=base_outcome.fold_index,
                outcome_kind=outcome_kind,
                subject_receipt_sha256=subject_receipt_sha256,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            base_outcome=base_outcome,
            validation_sources_v2=sources_v2,
            validation_registry_v2=registry_v2,
            four_fold_validation_inputs_v2=barrier_v2,
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "canonical validation outcome V2 envelope is absent"
            )
        result = materialize_massive_adaptive_rl_validation_outcome_authority_v2(
            root=root,
            manifest=manifest,
            base_outcome=base_outcome,
            validation_sources_v2=sources_v2,
            validation_registry_v2=registry_v2,
            four_fold_validation_inputs_v2=barrier_v2,
            committed_at_ms=committed_at_ms,
        )
    result_time = result.source_transaction_committed_at_ms
    if result_time != committed_at_ms:
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "validation outcome V2 envelope chronology differs"
        )
    return result


def _fold_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_fit: MassiveAdaptiveRLFoldFitAuthorityV1,
    sources_v1: MassiveAdaptiveRLValidationSourcesAuthorityV1,
    registry_v1: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    barrier_v1: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    primary: tuple[MassiveAdaptiveRLPolicyTraceAuthorityV1, ...],
    ladders: tuple[MassiveAdaptiveRLCostLadderAuthorityV1, ...],
    fixed: MassiveAdaptiveRLFixedControlValidationAuthorityV1,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLFoldValidationAuthorityV1:
    relative = fold_validation_authority_relative_path_v1(
        manifest=manifest,
        fold_index=fold_fit.outer_fold_index,
    )
    if _transaction_exists(root=root, relative=relative):
        result = authorize_massive_adaptive_rl_fold_validation_authority_v1(
            root=root,
            authority=parse_massive_adaptive_rl_fold_validation_authority_v1(
                root=root,
                loaded_source=_load(
                    root=root,
                    relative=relative,
                    verified_at_ms=committed_at_ms,
                ),
            ),
            manifest=manifest,
            fold_fit_authority=fold_fit,
            validation_sources_authority=sources_v1,
            validation_environment_registry=registry_v1,
            four_fold_validation_inputs_authority=barrier_v1,
            primary_trace_authorities=primary,
            cost_ladder_authorities=ladders,
            fixed_control_validation_authority=fixed,
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "canonical fold-validation V1 witness is absent"
            )
        result = materialize_massive_adaptive_rl_fold_validation_authority_v1(
            root=root,
            manifest=manifest,
            fold_fit_authority=fold_fit,
            validation_sources_authority=sources_v1,
            validation_environment_registry=registry_v1,
            four_fold_validation_inputs_authority=barrier_v1,
            primary_trace_authorities=primary,
            cost_ladder_authorities=ladders,
            fixed_control_validation_authority=fixed,
            committed_at_ms=committed_at_ms,
        )
    _assert_commit_time(loaded=result.loaded_source, expected_at_ms=committed_at_ms)
    return result


def _fold_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    base_fold: MassiveAdaptiveRLFoldValidationAuthorityV1,
    sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    barrier_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    primary: tuple[MassiveAdaptiveRLValidationOutcomeAuthorityV2, ...],
    ladders: tuple[MassiveAdaptiveRLValidationOutcomeAuthorityV2, ...],
    fixed: MassiveAdaptiveRLValidationOutcomeAuthorityV2,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLFoldValidationAuthorityV2:
    relative = fold_validation_authority_relative_path_v2(
        manifest=manifest,
        fold_index=base_fold.fold_index,
    )
    if _transaction_exists(root=root, relative=relative):
        result = authorize_massive_adaptive_rl_fold_validation_authority_v2(
            authority=load_massive_adaptive_rl_fold_validation_authority_v2(
                root=root,
                manifest=manifest,
                fold_index=base_fold.fold_index,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            base_fold_validation_v1=base_fold,
            validation_sources_v2=sources_v2,
            validation_registry_v2=registry_v2,
            four_fold_validation_inputs_v2=barrier_v2,
            primary_outcomes_v2=primary,
            ladder_outcomes_v2=ladders,
            fixed_control_outcome_v2=fixed,
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "canonical fold-validation V2 authority is absent"
            )
        result = materialize_massive_adaptive_rl_fold_validation_authority_v2(
            root=root,
            manifest=manifest,
            base_fold_validation_v1=base_fold,
            validation_sources_v2=sources_v2,
            validation_registry_v2=registry_v2,
            four_fold_validation_inputs_v2=barrier_v2,
            primary_outcomes_v2=primary,
            ladder_outcomes_v2=ladders,
            fixed_control_outcome_v2=fixed,
            committed_at_ms=committed_at_ms,
        )
    if result.source_transaction_committed_at_ms != committed_at_ms:
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation V2 chronology differs"
        )
    return result


@contextmanager
def _fold_validation_execution_lease_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
) -> Iterator[None]:
    manifest.validate()
    directory = Path(root) / "massive-adaptive" / "rl-fold-validation-leases-v1"
    descriptor = -1

    def close_after_setup_failure() -> None:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass

    try:
        Path(root).mkdir(parents=True, exist_ok=True)
        if Path(root).is_symlink():
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "fold-validation execution root is a symlink"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "fold-validation execution lease directory is a symlink"
            )
        descriptor = os.open(
            directory / f"v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}.lock",
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "fold-validation execution lease identity differs"
            )
    except OSError as error:
        close_after_setup_failure()
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution lease is unavailable"
        ) from error
    except MassiveAdaptiveRLFoldValidationExecutorV1Error:
        close_after_setup_failure()
        raise
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        close_after_setup_failure()
        raise MassiveAdaptiveRLFoldValidationExecutionLeaseUnavailable(
            "fold-validation execution lease is already held"
        ) from error
    except OSError as error:
        close_after_setup_failure()
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution lease is unavailable"
        ) from error
    try:
        yield
    finally:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    fold_index: int,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    """Run or strictly replay one fold from V2 inputs through Selection V3.

    ``committed_at_ms`` is the first V1 outcome's timestamp for a new fold.
    Later stages use consecutive logical milliseconds.  Once the first outcome
    exists, its timestamp is the resume anchor and a later invocation cannot
    choose a new generation.
    """

    if (
        isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms < 0
        or not isinstance(allow_materialize, bool)
    ):
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation execution arguments differ"
        )
    roots = _validate_execution_roots(
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        fold_index=fold_index,
    )
    sources_v2 = four_fold_validation_inputs_v2.validation_sources(fold_index)
    registry_v2 = four_fold_validation_inputs_v2.validation_registry(fold_index)
    sources_v1 = sources_v2.base_authority_v1
    registry_v1 = registry_v2.base_registry_v1
    barrier_v1 = four_fold_validation_inputs_v2.base_authority_v1
    chronology = sources_v2.runtime_chronology_authority
    barrier_time = four_fold_validation_inputs_v2.source_transaction_committed_at_ms
    if barrier_time is None:
        raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
            "fold-validation V2 input barrier is not persisted"
        )
    stages = _canonical_validation_stages_v1(
        manifest=manifest,
        fold_index=fold_index,
        checkpoint_authority_receipts=tuple(
            row.semantic_receipt_sha256 for row in roots.checkpoints
        ),
        fixed_control_selection_authority_receipt_sha256=(
            roots.fixed_selection.semantic_receipt_sha256
        ),
    )
    lease = (
        _fold_validation_execution_lease_v1(
            root=root, manifest=manifest, fold_index=fold_index
        )
        if allow_materialize
        else nullcontext()
    )
    with (
        lease,
        preserve_massive_adaptive_rl_process_rng_state_v1(include_cuda=False),
    ):
        anchor = _execution_anchor_ms(
            root=root,
            stages=stages,
            requested_first_commit_ms=committed_at_ms,
            barrier_committed_at_ms=barrier_time,
        )
        primary_v1: list[MassiveAdaptiveRLPolicyTraceAuthorityV1] = []
        primary_v2: list[MassiveAdaptiveRLValidationOutcomeAuthorityV2] = []
        ladders_v1: list[MassiveAdaptiveRLCostLadderAuthorityV1] = []
        ladders_v2: list[MassiveAdaptiveRLValidationOutcomeAuthorityV2] = []
        offset = 0
        for checkpoint in roots.checkpoints:
            primary_time = anchor + offset
            validate_massive_adaptive_rl_validation_outcome_barrier_v2(
                authority=four_fold_validation_inputs_v2,
                validation_environment_registry=registry_v2,
                fold_index=fold_index,
                outcome_committed_at_ms=primary_time,
                checkpoint_authority_receipt_sha256=(
                    checkpoint.semantic_receipt_sha256
                ),
            )
            primary = _primary_v1(
                root=root,
                manifest=manifest,
                checkpoint=checkpoint,
                chronology=chronology,
                registry_v1=registry_v1,
                barrier_v1=barrier_v1,
                fold_index=fold_index,
                committed_at_ms=primary_time,
                allow_materialize=allow_materialize,
            )
            primary_envelope = _outcome_v2(
                root=root,
                manifest=manifest,
                base_outcome=primary,
                sources_v2=sources_v2,
                registry_v2=registry_v2,
                barrier_v2=four_fold_validation_inputs_v2,
                outcome_kind=_PRIMARY,
                subject_receipt_sha256=checkpoint.semantic_receipt_sha256,
                committed_at_ms=anchor + offset + 1,
                allow_materialize=allow_materialize,
            )
            ladder_time = anchor + offset + 2
            validate_massive_adaptive_rl_validation_outcome_barrier_v2(
                authority=four_fold_validation_inputs_v2,
                validation_environment_registry=registry_v2,
                fold_index=fold_index,
                outcome_committed_at_ms=ladder_time,
                checkpoint_authority_receipt_sha256=(
                    checkpoint.semantic_receipt_sha256
                ),
            )
            ladder = _ladder_v1(
                root=root,
                manifest=manifest,
                checkpoint=checkpoint,
                chronology=chronology,
                registry_v1=registry_v1,
                barrier_v1=barrier_v1,
                fold_index=fold_index,
                committed_at_ms=ladder_time,
                allow_materialize=allow_materialize,
            )
            ladder_envelope = _outcome_v2(
                root=root,
                manifest=manifest,
                base_outcome=ladder,
                sources_v2=sources_v2,
                registry_v2=registry_v2,
                barrier_v2=four_fold_validation_inputs_v2,
                outcome_kind=_LADDER,
                subject_receipt_sha256=checkpoint.semantic_receipt_sha256,
                committed_at_ms=anchor + offset + 3,
                allow_materialize=allow_materialize,
            )
            primary_v1.append(primary)
            primary_v2.append(primary_envelope)
            ladders_v1.append(ladder)
            ladders_v2.append(ladder_envelope)
            offset += 4

        fixed_time = anchor + offset
        validate_massive_adaptive_rl_validation_outcome_barrier_v2(
            authority=four_fold_validation_inputs_v2,
            validation_environment_registry=registry_v2,
            fold_index=fold_index,
            outcome_committed_at_ms=fixed_time,
        )
        fixed_v1 = _fixed_v1(
            root=root,
            manifest=manifest,
            fit=roots.fixed_fit,
            selection=roots.fixed_selection,
            chronology=chronology,
            registry_v1=registry_v1,
            barrier_v1=barrier_v1,
            fold_index=fold_index,
            committed_at_ms=fixed_time,
            allow_materialize=allow_materialize,
        )
        fixed_v2 = _outcome_v2(
            root=root,
            manifest=manifest,
            base_outcome=fixed_v1,
            sources_v2=sources_v2,
            registry_v2=registry_v2,
            barrier_v2=four_fold_validation_inputs_v2,
            outcome_kind=_FC06,
            subject_receipt_sha256=roots.fixed_selection.semantic_receipt_sha256,
            committed_at_ms=anchor + offset + 1,
            allow_materialize=allow_materialize,
        )
        fold_v1 = _fold_v1(
            root=root,
            manifest=manifest,
            fold_fit=roots.fold_fit,
            sources_v1=sources_v1,
            registry_v1=registry_v1,
            barrier_v1=barrier_v1,
            primary=tuple(primary_v1),
            ladders=tuple(ladders_v1),
            fixed=fixed_v1,
            committed_at_ms=anchor + offset + 2,
            allow_materialize=allow_materialize,
        )
        fold_v2 = _fold_v2(
            root=root,
            manifest=manifest,
            base_fold=fold_v1,
            sources_v2=sources_v2,
            registry_v2=registry_v2,
            barrier_v2=four_fold_validation_inputs_v2,
            primary=tuple(primary_v2),
            ladders=tuple(ladders_v2),
            fixed=fixed_v2,
            committed_at_ms=anchor + offset + 3,
            allow_materialize=allow_materialize,
        )
        selection = run_or_resume_massive_adaptive_rl_policy_selection_authority_v3(
            root=root,
            manifest=manifest,
            validation_authority=fold_v2,
            committed_at_ms=anchor + offset + 4,
            allow_materialize=allow_materialize,
        )
        if (
            type(selection) is not MassiveAdaptiveRLPolicySelectionAuthorityV3
            or selection.source_transaction_committed_at_ms != anchor + offset + 5
            or not selection.development_stage_authorized
        ):
            raise MassiveAdaptiveRLFoldValidationExecutorV1Error(
                "fold-validation execution did not reach authorized Selection V3"
            )
        return selection


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFoldValidationExecutionLeaseUnavailable",
    "MassiveAdaptiveRLFoldValidationExecutorV1Error",
    "run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v1",
]
