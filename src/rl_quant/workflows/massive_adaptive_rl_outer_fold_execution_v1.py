"""State-gated package orchestration for one Manifest-V5 outer fold.

The public surface accepts no market environment, forecast, action, target,
transition, or economic result.  It derives the next outer fold from the exact
persisted prequential-state head, commits access before opening inputs, replays
the frozen PPO and fixed control, seals the complete fold, and immediately
advances the append-only state chain.

All four folds derive their predictor roots from the global development-origin
inventories committed in the runtime-source graph.  No caller selects a fold
index: the exact current state head determines the only legal transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v2 import (
    MassiveAdaptiveOuterAccessCommitmentV2,
    run_or_resume_massive_adaptive_outer_access_commitment_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
    MassiveAdaptiveRLOuterFoldSealAuthorityV1,
    run_or_resume_massive_adaptive_rl_outer_fold_seal_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_inputs_v1 import (
    MassiveAdaptiveRLOuterInputAuthorityV1,
    run_or_resume_massive_adaptive_rl_outer_inputs_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_authority_v2 import (
    MassiveAdaptiveRLOuterRolloutAuthorityV2,
    run_or_resume_massive_adaptive_rl_outer_rollout_authority_v2,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_experiment_state_v1 import (
    MassiveAdaptiveRLPrequentialExperimentStateV1,
    MassiveAdaptiveRLPrequentialStageV1,
    load_massive_adaptive_rl_prequential_experiment_states_v1,
    run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
)


class MassiveAdaptiveRLOuterFoldExecutionV1Error(ValueError):
    """The requested outer transition is stale, out of order, or incomplete."""


_OUTER_FOLD_BY_PREDECESSOR_STAGE = {
    MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN: 0,
    MassiveAdaptiveRLPrequentialStageV1.POLICY_2_FROZEN: 1,
    MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN: 2,
    MassiveAdaptiveRLPrequentialStageV1.OUTER_2_SEALED: 3,
}


def _committed_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLOuterFoldExecutionV1Error(f"{name} is absent or invalid")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterFoldExecutionV1:
    fold_index: int
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2
    outer_inputs: MassiveAdaptiveRLOuterInputAuthorityV1
    outer_rollout: MassiveAdaptiveRLOuterRolloutAuthorityV2
    outer_fold_seal: MassiveAdaptiveRLOuterFoldSealAuthorityV1
    prequential_state: MassiveAdaptiveRLPrequentialExperimentStateV1

    def validate(self) -> None:
        for authority in (
            self.outer_access,
            self.outer_inputs,
            self.outer_rollout,
            self.outer_fold_seal,
            self.prequential_state,
        ):
            authority.validate()
        access_time = _committed_time(
            "outer access timestamp",
            self.outer_access.source_transaction_committed_at_ms,
        )
        input_time = _committed_time(
            "outer input timestamp",
            self.outer_inputs.source_transaction_committed_at_ms,
        )
        rollout_time = _committed_time(
            "outer rollout timestamp",
            self.outer_rollout.source_transaction_committed_at_ms,
        )
        seal_time = _committed_time(
            "outer seal timestamp",
            self.outer_fold_seal.source_transaction_committed_at_ms,
        )
        state_time = _committed_time(
            "outer state timestamp",
            self.prequential_state.source_transaction_committed_at_ms,
        )
        expected_stage = MassiveAdaptiveRLPrequentialStageV1(
            f"outer-{self.fold_index}-sealed"
        )
        if (
            self.fold_index not in range(4)
            or not access_time < input_time < rollout_time < seal_time < state_time
            or self.outer_access.fold_index != self.fold_index
            or self.outer_inputs.fold_index != self.fold_index
            or self.outer_rollout.fold_index != self.fold_index
            or self.outer_fold_seal.fold_index != self.fold_index
            or self.prequential_state.stage is not expected_stage
            or self.outer_inputs.outer_access_commitment_receipt_sha256
            != self.outer_access.semantic_receipt_sha256
            or self.outer_rollout.outer_access_commitment_receipt_sha256
            != self.outer_access.semantic_receipt_sha256
            or self.outer_fold_seal.outer_rollout_authority_receipt_sha256
            != self.outer_rollout.semantic_receipt_sha256
            or self.prequential_state.stage_artifact_semantic_receipt_sha256
            != self.outer_fold_seal.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLOuterFoldExecutionV1Error(
                "outer-fold execution lineage or chronology differs"
            )


def _require_exact_current_state_head_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    predecessor_state: MassiveAdaptiveRLPrequentialExperimentStateV1,
    allow_materialize: bool,
) -> tuple[int, bool]:
    predecessor_state.validate()
    fold_index = _OUTER_FOLD_BY_PREDECESSOR_STAGE.get(predecessor_state.stage)
    if fold_index is None:
        raise MassiveAdaptiveRLOuterFoldExecutionV1Error(
            "prequential state does not authorize an outer transition"
        )
    states = load_massive_adaptive_rl_prequential_experiment_states_v1(
        root=root, manifest=manifest
    )
    if not states:
        raise MassiveAdaptiveRLOuterFoldExecutionV1Error(
            "prequential state history is absent"
        )
    matching_positions = tuple(
        index
        for index, state in enumerate(states)
        if (
            predecessor_state.semantic_receipt_sha256 == state.semantic_receipt_sha256
            and predecessor_state.source_receipt_sha256 == state.source_receipt_sha256
            and predecessor_state.source_transaction_receipt_sha256
            == state.source_transaction_receipt_sha256
            and predecessor_state.source_transaction_committed_at_ms
            == state.source_transaction_committed_at_ms
        )
    )
    if (
        not predecessor_state.prequential_execution_authorized
        or predecessor_state.experiment_id != manifest.experiment_id
        or predecessor_state.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or len(matching_positions) != 1
    ):
        raise MassiveAdaptiveRLOuterFoldExecutionV1Error(
            "outer execution requires an exact persisted predecessor state"
        )
    position = matching_positions[0]
    if position == len(states) - 1:
        return fold_index, allow_materialize
    expected_completed_stage = MassiveAdaptiveRLPrequentialStageV1(
        f"outer-{fold_index}-sealed"
    )
    if states[position + 1].stage is not expected_completed_stage:
        raise MassiveAdaptiveRLOuterFoldExecutionV1Error(
            "historical outer execution has no exact committed successor state"
        )
    return fold_index, False


def run_or_resume_massive_adaptive_rl_outer_fold_execution_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    predecessor_state: MassiveAdaptiveRLPrequentialExperimentStateV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLOuterFoldExecutionV1:
    """Advance the exact current state head through one sealed outer fold."""

    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or type(policy_schedule) is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
        or type(predecessor_state) is not MassiveAdaptiveRLPrequentialExperimentStateV1
        or type(allow_materialize) is not bool
    ):
        raise MassiveAdaptiveRLOuterFoldExecutionV1Error(
            "outer execution requires exact Manifest-V5 authorities"
        )
    manifest.validate()
    manifest_registration.validate()
    execution_registration.validate()
    policy_schedule.validate()
    predecessor_state.validate()
    if (
        not manifest_registration.development_protocol_registered
        or not execution_registration.development_execution_registered
        or not policy_schedule.development_stage_authorized
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or policy_schedule.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or predecessor_state.execution_implementation_registration_receipt_sha256
        != execution_registration.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterFoldExecutionV1Error(
            "outer execution Manifest-V5 lineage differs"
        )
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root, experiment_id=manifest.experiment_id
    ):
        fold_index, stage_allow_materialize = _require_exact_current_state_head_v1(
            root=root,
            manifest=manifest,
            predecessor_state=predecessor_state,
            allow_materialize=allow_materialize,
        )
        expected_schedule_folds = tuple(range((2, 3, 4, 4)[fold_index]))
        if (
            policy_schedule.fold_indices != expected_schedule_folds
            or fold_index in (0, 1, 2)
            and (
                predecessor_state.stage_artifact_semantic_receipt_sha256
                != policy_schedule.semantic_receipt_sha256
                or predecessor_state.stage_artifact_source_receipt_sha256
                != policy_schedule.source_receipt_sha256
                or predecessor_state.stage_artifact_commit_receipt_sha256
                != policy_schedule.source_transaction_receipt_sha256
                or predecessor_state.stage_artifact_committed_at_ms
                != policy_schedule.source_transaction_committed_at_ms
            )
        ):
            raise MassiveAdaptiveRLOuterFoldExecutionV1Error(
                "outer fold requires its persisted causal policy-schedule head"
            )
        frozen_policy = policy_schedule.frozen_policy(fold_index)
        frozen_control = policy_schedule.frozen_control(fold_index)
        outer_access = run_or_resume_massive_adaptive_outer_access_commitment_v2(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            policy_schedule=policy_schedule,
            frozen_policy=frozen_policy,
            frozen_control=frozen_control,
            predecessor_state=predecessor_state,
            allow_materialize=stage_allow_materialize,
        )
        input_execution = run_or_resume_massive_adaptive_rl_outer_inputs_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            outer_access=outer_access,
            allow_materialize=stage_allow_materialize,
        )
        outer_rollout = run_or_resume_massive_adaptive_rl_outer_rollout_authority_v2(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            outer_access=input_execution.outer_access,
            allow_materialize=stage_allow_materialize,
        )
        outer_fold_seal = (
            run_or_resume_massive_adaptive_rl_outer_fold_seal_authority_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                policy_schedule=policy_schedule,
                outer_access=input_execution.outer_access,
                outer_rollout=outer_rollout,
                allow_materialize=stage_allow_materialize,
            )
        )
        state = run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            stage_artifact=outer_fold_seal,
            allow_materialize=stage_allow_materialize,
        )
    result = MassiveAdaptiveRLOuterFoldExecutionV1(
        fold_index=fold_index,
        outer_access=input_execution.outer_access,
        outer_inputs=input_execution.outer_inputs,
        outer_rollout=outer_rollout,
        outer_fold_seal=outer_fold_seal,
        prequential_state=state,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLOuterFoldExecutionV1",
    "MassiveAdaptiveRLOuterFoldExecutionV1Error",
    "run_or_resume_massive_adaptive_rl_outer_fold_execution_v1",
]
