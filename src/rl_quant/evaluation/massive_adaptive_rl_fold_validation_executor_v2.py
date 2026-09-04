"""Execute one validation fold under an attested numerical environment.

V2 keeps the V1 economic and selection authorities as exact computational
witnesses, but replaces synthetic timestamp offsets with controller-captured,
monotonic publication times and publishes a final predecessor-linked execution
authority.  A validation execution environment is committed before the first
outcome and replayed exactly on every resume.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_execution_authority_v1 import (
    MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
    fold_validation_execution_authority_relative_path_v1,
    run_or_resume_massive_adaptive_rl_fold_validation_execution_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_executor_v1 import (
    _FC06,
    _LADDER,
    _PRIMARY,
    _canonical_validation_stages_v1,
    _fixed_v1,
    _fold_v1,
    _fold_v2,
    _fold_validation_execution_lease_v1,
    _ladder_v1,
    _outcome_v2,
    _primary_v1,
    _validate_execution_roots,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    validate_massive_adaptive_rl_validation_outcome_barrier_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.training.massive_adaptive_rl_policy_selection_v3 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV3,
    run_or_resume_massive_adaptive_rl_policy_selection_authority_v3,
)
from rl_quant.workflows.massive_adaptive_rl_execution_environment_v1 import (
    massive_adaptive_rl_deterministic_execution_v1,
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
)
from rl_quant.workflows.massive_adaptive_rl_validation_execution_environment_v1 import (
    run_or_resume_massive_adaptive_rl_validation_execution_environment_v1,
    validation_execution_environment_relative_path_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    legacy_manifest_v5_rejecting_writer_guard_v1,
)


MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-v2-validation-barrier-and-completed-four-fold-fit",
        "environment": "canonical-persisted-exactly-replayed-cpu-attestation-v1",
        "candidate_discovery": "exact-fold-fit-policy-checkpoint-inventory",
        "economic_witnesses": "canonical-v1-primary-ladder-and-fc06",
        "v2_lineage": "immediate-outcome-envelopes-fold-v2-and-selection-v3",
        "chronology": "wall-clock-monotonic-no-caller-timestamp",
        "resume": "strict-canonical-prefix-with-no-upstream-repair",
        "completion": "predecessor-linked-fold-execution-authority-v1",
        "execution_device": "cpu",
        "caller_environment": False,
        "caller_device": False,
        "caller_timestamp": False,
        "caller_actions": False,
        "caller_targets": False,
        "caller_metrics": False,
        "caller_candidates": False,
        "caller_selection": False,
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)


class MassiveAdaptiveRLFoldValidationExecutorV2Error(ValueError):
    """The attested validation execution is absent, mixed, or inconsistent."""


class MassiveAdaptiveRLFoldValidationRecoveryGenerationRequired(
    MassiveAdaptiveRLFoldValidationExecutorV2Error
):
    """A partial immutable transaction requires an explicit new generation."""


def _wall_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _next_publication_time_ms(*, previous_at_ms: int) -> int:
    current = _wall_clock_ms()
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
            "validation publication clock differs"
        )
    return max(current, previous_at_ms + 1)


def _transaction_exists_v2(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLFoldValidationRecoveryGenerationRequired(
            "partial validation transaction cannot be repaired in this generation"
        )
    return all(present)


def _load_stage(
    *, root: str | Path, relative: str, verified_at_ms: int
) -> LoadedMassiveSourceObject:
    return load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=verified_at_ms,
    )


def _existing_stage_prefix(
    *,
    root: str | Path,
    stages,
    environment_committed_at_ms: int,
    verified_at_ms: int,
) -> tuple[LoadedMassiveSourceObject, ...]:
    exists = tuple(
        _transaction_exists_v2(root=root, relative=stage.relative_path)
        for stage in stages
    )
    first_missing = next(
        (index for index, present in enumerate(exists) if not present), len(exists)
    )
    if any(exists[first_missing:]):
        raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
            "validation execution cannot repair a missing upstream stage"
        )
    loaded = tuple(
        _load_stage(
            root=root,
            relative=stage.relative_path,
            verified_at_ms=verified_at_ms,
        )
        for stage in stages[:first_missing]
    )
    times = tuple(row.commit.committed_at_ms for row in loaded)
    if times and (
        times[0] <= environment_committed_at_ms
        or any(right <= left for left, right in zip(times, times[1:]))
    ):
        raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
            "validation execution publication chronology differs"
        )
    return loaded


@legacy_manifest_v5_rejecting_writer_guard_v1(
    materialize_parameter="allow_materialize"
)
def run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    fold_index: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    """Run or replay one attested fold through Selection V3.

    Publication time is captured internally.  Existing evidence must form one
    strict canonical prefix, and a partial source transaction is never repaired
    under the same scientific generation.
    """

    if not isinstance(allow_materialize, bool):
        raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
            "validation execution arguments differ"
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
        raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
            "validation V2 input barrier is not persisted"
        )
    checkpoint_receipts = tuple(
        row.semantic_receipt_sha256 for row in roots.checkpoints
    )
    stages = _canonical_validation_stages_v1(
        manifest=manifest,
        fold_index=fold_index,
        checkpoint_authority_receipts=checkpoint_receipts,
        fixed_control_selection_authority_receipt_sha256=(
            roots.fixed_selection.semantic_receipt_sha256
        ),
    )
    environment_relative = validation_execution_environment_relative_path_v1(
        manifest=manifest,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
    )
    environment_exists = _transaction_exists_v2(
        root=root, relative=environment_relative
    )
    stage_exists = tuple(
        _transaction_exists_v2(root=root, relative=stage.relative_path)
        for stage in stages
    )
    completion_relative = fold_validation_execution_authority_relative_path_v1(
        manifest=manifest, fold_index=fold_index
    )
    completion_exists = _transaction_exists_v2(root=root, relative=completion_relative)
    if any(stage_exists) and not environment_exists:
        raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
            "validation outcomes exist without their execution environment"
        )
    if completion_exists and not all(stage_exists):
        raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
            "validation completion exists without every upstream stage"
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
        massive_adaptive_rl_deterministic_execution_v1(device="cpu"),
    ):
        environment_request_time = _next_publication_time_ms(
            previous_at_ms=barrier_time
        )
        environment = (
            run_or_resume_massive_adaptive_rl_validation_execution_environment_v1(
                root=root,
                manifest=manifest,
                runtime_sources_v2=runtime_sources_v2,
                four_fold_fit_authority=four_fold_fit_authority,
                four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
                executor_implementation_source_sha256=(
                    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SOURCE_SHA256
                ),
                committed_at_ms=environment_request_time,
                allow_materialize=allow_materialize,
            )
        )
        environment_time = environment.source_transaction_committed_at_ms
        if environment_time is None or not environment.development_stage_authorized:
            raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
                "validation execution environment did not authorize execution"
            )
        existing = _existing_stage_prefix(
            root=root,
            stages=stages,
            environment_committed_at_ms=environment_time,
            verified_at_ms=_wall_clock_ms(),
        )
        existing_times = tuple(row.commit.committed_at_ms for row in existing)
        last_time = existing_times[-1] if existing_times else environment_time

        def stage_time(index: int) -> int:
            nonlocal last_time
            if index < len(existing_times):
                value = existing_times[index]
            else:
                value = _next_publication_time_ms(previous_at_ms=last_time)
            if value <= last_time and index >= len(existing_times):
                raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
                    "validation publication clock did not advance"
                )
            last_time = value
            return value

        primary_v1 = []
        primary_v2 = []
        ladders_v1 = []
        ladders_v2 = []
        offset = 0
        for checkpoint in roots.checkpoints:
            primary_time = stage_time(offset)
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
                committed_at_ms=stage_time(offset + 1),
                allow_materialize=allow_materialize,
            )
            ladder_time = stage_time(offset + 2)
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
                committed_at_ms=stage_time(offset + 3),
                allow_materialize=allow_materialize,
            )
            primary_v1.append(primary)
            primary_v2.append(primary_envelope)
            ladders_v1.append(ladder)
            ladders_v2.append(ladder_envelope)
            offset += 4

        fixed_time = stage_time(offset)
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
            committed_at_ms=stage_time(offset + 1),
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
            committed_at_ms=stage_time(offset + 2),
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
            committed_at_ms=stage_time(offset + 3),
            allow_materialize=allow_materialize,
        )
        selection_v2_exists = offset + 4 < len(existing_times)
        selection_v3_exists = offset + 5 < len(existing_times)
        if selection_v2_exists and not selection_v3_exists:
            selection_request_time = _next_publication_time_ms(
                previous_at_ms=existing_times[offset + 4]
            )
        elif selection_v3_exists:
            selection_request_time = existing_times[offset + 5]
        else:
            selection_request_time = stage_time(offset + 4)
        selection = run_or_resume_massive_adaptive_rl_policy_selection_authority_v3(
            root=root,
            manifest=manifest,
            validation_authority=fold_v2,
            committed_at_ms=selection_request_time,
            allow_materialize=allow_materialize,
        )
        if (
            type(selection) is not MassiveAdaptiveRLPolicySelectionAuthorityV3
            or not selection.development_stage_authorized
        ):
            raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
                "validation execution did not reach authorized Selection V3"
            )
        refreshed = _existing_stage_prefix(
            root=root,
            stages=stages,
            environment_committed_at_ms=environment_time,
            verified_at_ms=_wall_clock_ms(),
        )
        if len(refreshed) != len(stages):
            raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
                "validation execution did not complete every canonical stage"
            )
        selection_time = selection.source_transaction_committed_at_ms
        if (
            selection_time is None
            or selection_time != refreshed[-1].commit.committed_at_ms
        ):
            raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
                "validation Selection V3 chronology differs"
            )
        completion_time = _next_publication_time_ms(previous_at_ms=selection_time)
        result = (
            run_or_resume_massive_adaptive_rl_fold_validation_execution_authority_v1(
                root=root,
                manifest=manifest,
                fold_index=fold_index,
                checkpoint_authority_receipts=checkpoint_receipts,
                fixed_control_selection_authority_receipt_sha256=(
                    roots.fixed_selection.semantic_receipt_sha256
                ),
                validation_execution_environment=environment,
                policy_selection_v3=selection,
                committed_at_ms=completion_time,
                allow_materialize=allow_materialize,
            )
        )
        if not result.development_stage_authorized:
            raise MassiveAdaptiveRLFoldValidationExecutorV2Error(
                "attested fold validation execution is not authorized"
            )
        return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SPEC_SHA256",
    "MassiveAdaptiveRLFoldValidationExecutorV2Error",
    "MassiveAdaptiveRLFoldValidationRecoveryGenerationRequired",
    "run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2",
]
