"""Package-owned Manifest-V5 continuation through the complete policy schedule.

This workflow consumes the authenticated outer-zero seal and exact persisted
state head.  It releases and freezes policy two, executes and seals outer fold
one with the already-frozen policy one, then releases and freezes policy three.
No caller supplies a fold index, market environment, action, target, transition,
or economic result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
    run_or_resume_massive_adaptive_rl_delayed_validation_release_v1,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v2 import (
    MassiveAdaptiveFrozenRLPolicyV2,
)
from rl_quant.training.massive_adaptive_rl_frozen_fc06_v2 import (
    MassiveAdaptiveRLFrozenFC06V2,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_initial_validation_execution_v1 import (
    MassiveAdaptiveRLInitialValidationExecutionV1,
    MassiveAdaptiveRLReleasedFoldValidationExecutionV1,
    run_or_resume_massive_adaptive_rl_released_fold_validation_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_outer_fold_execution_v1 import (
    MassiveAdaptiveRLOuterFoldExecutionV1,
    run_or_resume_massive_adaptive_rl_outer_fold_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_experiment_state_v1 import (
    MassiveAdaptiveRLPrequentialExperimentStateV1,
    MassiveAdaptiveRLPrequentialStageV1,
    load_massive_adaptive_rl_prequential_experiment_states_v1,
    run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1,
)


class MassiveAdaptiveRLPrequentialContinuationV1Error(ValueError):
    """The delayed validation or outer-one causal sequence differs."""


def _materialization_mode_after_state_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    predecessor_state: MassiveAdaptiveRLPrequentialExperimentStateV1,
    completed_stage: MassiveAdaptiveRLPrequentialStageV1,
    allow_materialize: bool,
) -> bool:
    """Permit writes only when the supplied predecessor is the current head."""

    states = load_massive_adaptive_rl_prequential_experiment_states_v1(
        root=root, manifest=manifest
    )
    matching_positions = tuple(
        index
        for index, state in enumerate(states)
        if state.semantic_receipt_sha256 == predecessor_state.semantic_receipt_sha256
        and state.source_receipt_sha256 == predecessor_state.source_receipt_sha256
        and state.source_transaction_receipt_sha256
        == predecessor_state.source_transaction_receipt_sha256
        and state.source_transaction_committed_at_ms
        == predecessor_state.source_transaction_committed_at_ms
    )
    if len(matching_positions) != 1:
        raise MassiveAdaptiveRLPrequentialContinuationV1Error(
            "continuation requires an exact persisted predecessor state"
        )
    position = matching_positions[0]
    if position == len(states) - 1:
        return allow_materialize
    if states[position + 1].stage is not completed_stage:
        raise MassiveAdaptiveRLPrequentialContinuationV1Error(
            "historical continuation has no exact completed successor state"
        )
    return False


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPrequentialContinuationV1:
    validation_releases: tuple[MassiveAdaptiveRLValidationReleaseAuthorityV1, ...]
    released_fold_executions: tuple[
        MassiveAdaptiveRLReleasedFoldValidationExecutionV1, ...
    ]
    policy_schedules: tuple[MassiveAdaptiveRLWalkForwardPolicyScheduleV1, ...]
    outer_one_execution: MassiveAdaptiveRLOuterFoldExecutionV1
    prequential_state: MassiveAdaptiveRLPrequentialExperimentStateV1

    @property
    def frozen_ppo_policies(self) -> tuple[MassiveAdaptiveFrozenRLPolicyV2, ...]:
        return tuple(
            execution.frozen_ppo_policy for execution in self.released_fold_executions
        )

    @property
    def frozen_fc06_controls(self) -> tuple[MassiveAdaptiveRLFrozenFC06V2, ...]:
        return tuple(
            execution.frozen_fc06_control
            for execution in self.released_fold_executions
        )

    def validate(self) -> None:
        if (
            len(self.validation_releases) != 2
            or len(self.released_fold_executions) != 2
            or len(self.policy_schedules) != 2
        ):
            raise MassiveAdaptiveRLPrequentialContinuationV1Error(
                "prequential continuation inventory differs"
            )
        for authority in (
            *self.validation_releases,
            *self.released_fold_executions,
            *self.policy_schedules,
            self.outer_one_execution,
            self.prequential_state,
        ):
            authority.validate()
        release_two, release_three = self.validation_releases
        execution_two, execution_three = self.released_fold_executions
        schedule_two, schedule_three = self.policy_schedules
        outer_one = self.outer_one_execution
        state = self.prequential_state
        if (
            tuple(row.released_validation_fold_indices for row in self.validation_releases)
            != ((2,), (3,))
            or tuple(row.fold_index for row in self.released_fold_executions)
            != (2, 3)
            or tuple(row.fold_indices for row in self.policy_schedules)
            != ((0, 1, 2), (0, 1, 2, 3))
            or execution_two.fold_validation_authority.release_authority_receipt_sha256
            != release_two.semantic_receipt_sha256
            or execution_three.fold_validation_authority.release_authority_receipt_sha256
            != release_three.semantic_receipt_sha256
            or schedule_two.frozen_ppo_policy_receipts[-1]
            != execution_two.frozen_ppo_policy.semantic_receipt_sha256
            or schedule_three.frozen_ppo_policy_receipts[-1]
            != execution_three.frozen_ppo_policy.semantic_receipt_sha256
            or outer_one.fold_index != 1
            or outer_one.outer_access.policy_schedule_receipt_sha256
            != schedule_two.semantic_receipt_sha256
            or release_three.predecessor_outer_fold_seal_receipt_sha256
            != outer_one.outer_fold_seal.semantic_receipt_sha256
            or state.stage is not MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN
            or state.stage_artifact_semantic_receipt_sha256
            != schedule_three.semantic_receipt_sha256
            or not state.prequential_execution_authorized
        ):
            raise MassiveAdaptiveRLPrequentialContinuationV1Error(
                "prequential continuation lineage differs"
            )


def run_or_resume_massive_adaptive_rl_prequential_continuation_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    initial_execution: MassiveAdaptiveRLInitialValidationExecutionV1,
    initial_policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    outer_zero_execution: MassiveAdaptiveRLOuterFoldExecutionV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLPrequentialContinuationV1:
    """Advance O0 -> V2/P2 -> O1 -> V3/P3 under one experiment lock."""

    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or type(initial_inputs)
        is not MassiveAdaptiveRLInitialValidationInputsAuthorityV1
        or type(initial_execution) is not MassiveAdaptiveRLInitialValidationExecutionV1
        or type(initial_policy_schedule)
        is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
        or type(outer_zero_execution) is not MassiveAdaptiveRLOuterFoldExecutionV1
        or type(allow_materialize) is not bool
    ):
        raise MassiveAdaptiveRLPrequentialContinuationV1Error(
            "prequential continuation requires exact Manifest-V5 authorities"
        )
    for authority in (
        manifest,
        manifest_registration,
        execution_registration,
        initial_inputs,
        initial_execution,
        initial_policy_schedule,
        outer_zero_execution,
    ):
        authority.validate()
    if (
        not manifest_registration.development_protocol_registered
        or not execution_registration.development_execution_registered
        or not initial_execution.initial_policy_freezing_complete
        or not initial_policy_schedule.development_stage_authorized
        or initial_policy_schedule.fold_indices != (0, 1)
        or outer_zero_execution.fold_index != 0
        or outer_zero_execution.outer_access.policy_schedule_receipt_sha256
        != initial_policy_schedule.semantic_receipt_sha256
        or outer_zero_execution.prequential_state.stage
        is not MassiveAdaptiveRLPrequentialStageV1.OUTER_0_SEALED
    ):
        raise MassiveAdaptiveRLPrequentialContinuationV1Error(
            "prequential continuation roots differ"
        )

    initial_ppo = initial_execution.frozen_ppo_policies
    initial_fc06 = initial_execution.frozen_fc06_controls
    outer_zero_seal = outer_zero_execution.outer_fold_seal
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root, experiment_id=manifest.experiment_id
    ):
        release_two = run_or_resume_massive_adaptive_rl_delayed_validation_release_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            initial_inputs=initial_inputs,
            predecessor_outer_fold_seal=outer_zero_seal,
            predecessor_state=outer_zero_execution.prequential_state,
            allow_materialize=allow_materialize,
        )
        release_two_state = (
            run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                execution_registration=execution_registration,
                stage_artifact=release_two,
                allow_materialize=allow_materialize,
            )
        )
        policy_two_allow_materialize = _materialization_mode_after_state_v1(
            root=root,
            manifest=manifest,
            predecessor_state=release_two_state,
            completed_stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_2_FROZEN,
            allow_materialize=allow_materialize,
        )
        execution_two = (
            run_or_resume_massive_adaptive_rl_released_fold_validation_execution_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                execution_registration=execution_registration,
                validation_release=release_two,
                allow_materialize=policy_two_allow_materialize,
            )
        )
        schedule_two = (
            run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                execution_registration=execution_registration,
                frozen_ppo_policies=(*initial_ppo, execution_two.frozen_ppo_policy),
                frozen_fc06_controls=(*initial_fc06, execution_two.frozen_fc06_control),
                predecessor_outer_fold_seals=(outer_zero_seal,),
                allow_materialize=policy_two_allow_materialize,
            )
        )
        policy_two_state = (
            run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                execution_registration=execution_registration,
                stage_artifact=schedule_two,
                allow_materialize=policy_two_allow_materialize,
            )
        )
        outer_one = run_or_resume_massive_adaptive_rl_outer_fold_execution_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            policy_schedule=schedule_two,
            predecessor_state=policy_two_state,
            allow_materialize=allow_materialize,
        )
        release_three = run_or_resume_massive_adaptive_rl_delayed_validation_release_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            initial_inputs=initial_inputs,
            predecessor_outer_fold_seal=outer_one.outer_fold_seal,
            predecessor_state=outer_one.prequential_state,
            allow_materialize=allow_materialize,
        )
        release_three_state = (
            run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                execution_registration=execution_registration,
                stage_artifact=release_three,
                allow_materialize=allow_materialize,
            )
        )
        policy_three_allow_materialize = _materialization_mode_after_state_v1(
            root=root,
            manifest=manifest,
            predecessor_state=release_three_state,
            completed_stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN,
            allow_materialize=allow_materialize,
        )
        execution_three = (
            run_or_resume_massive_adaptive_rl_released_fold_validation_execution_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                execution_registration=execution_registration,
                validation_release=release_three,
                allow_materialize=policy_three_allow_materialize,
            )
        )
        schedule_three = (
            run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                execution_registration=execution_registration,
                frozen_ppo_policies=(
                    *initial_ppo,
                    execution_two.frozen_ppo_policy,
                    execution_three.frozen_ppo_policy,
                ),
                frozen_fc06_controls=(
                    *initial_fc06,
                    execution_two.frozen_fc06_control,
                    execution_three.frozen_fc06_control,
                ),
                predecessor_outer_fold_seals=(
                    outer_zero_seal,
                    outer_one.outer_fold_seal,
                ),
                allow_materialize=policy_three_allow_materialize,
            )
        )
        state = run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            stage_artifact=schedule_three,
            allow_materialize=policy_three_allow_materialize,
        )

    result = MassiveAdaptiveRLPrequentialContinuationV1(
        validation_releases=(release_two, release_three),
        released_fold_executions=(execution_two, execution_three),
        policy_schedules=(schedule_two, schedule_three),
        outer_one_execution=outer_one,
        prequential_state=state,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLPrequentialContinuationV1",
    "MassiveAdaptiveRLPrequentialContinuationV1Error",
    "run_or_resume_massive_adaptive_rl_prequential_continuation_v1",
]
