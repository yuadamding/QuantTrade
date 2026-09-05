"""Complete Manifest-V5 outer execution through report publication.

This package-owned workflow resumes from the exact ``policy-3-frozen`` state,
executes outer folds two and three in ledger order, publishes the four-fold
profitability report, and immediately appends the report state.  It accepts no
market environment, forecast, action, target, transition, return, or metric.

Report publication completes the economic experiment but does not claim full
end-to-end verification.  A distinct nonmaterializing finalizer remains
required before the final cold-replay state can be issued.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
    MassiveAdaptiveRLOuterFoldSealAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v2 import (
    MassiveAdaptiveRLProfitabilityReportAuthorityV2,
    run_or_resume_massive_adaptive_rl_profitability_report_authority_v2,
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
from rl_quant.workflows.massive_adaptive_rl_outer_fold_execution_v1 import (
    MassiveAdaptiveRLOuterFoldExecutionV1,
    run_or_resume_massive_adaptive_rl_outer_fold_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_continuation_v1 import (
    MassiveAdaptiveRLPrequentialContinuationV1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_experiment_state_v1 import (
    MassiveAdaptiveRLPrequentialExperimentStateV1,
    MassiveAdaptiveRLPrequentialStageV1,
    load_massive_adaptive_rl_prequential_experiment_states_v1,
    run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1,
)


class MassiveAdaptiveRLFinalOuterExecutionV1Error(ValueError):
    """The final outer sequence or report boundary is stale or inconsistent."""


def _same_persisted_state(
    left: MassiveAdaptiveRLPrequentialExperimentStateV1,
    right: MassiveAdaptiveRLPrequentialExperimentStateV1,
) -> bool:
    return bool(
        left.semantic_receipt_sha256 == right.semantic_receipt_sha256
        and left.source_receipt_sha256 == right.source_receipt_sha256
        and left.source_transaction_receipt_sha256
        == right.source_transaction_receipt_sha256
        and left.source_transaction_committed_at_ms
        == right.source_transaction_committed_at_ms
    )


def _report_materialization_mode_after_state_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    predecessor_state: MassiveAdaptiveRLPrequentialExperimentStateV1,
    allow_materialize: bool,
) -> bool:
    """Permit report creation only from the exact persisted outer-three head."""

    predecessor_state.validate()
    states = load_massive_adaptive_rl_prequential_experiment_states_v1(
        root=root,
        manifest=manifest,
    )
    positions = tuple(
        index
        for index, state in enumerate(states)
        if _same_persisted_state(state, predecessor_state)
    )
    if (
        predecessor_state.stage
        is not MassiveAdaptiveRLPrequentialStageV1.OUTER_3_SEALED
        or not predecessor_state.prequential_execution_authorized
        or predecessor_state.experiment_id != manifest.experiment_id
        or predecessor_state.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or len(positions) != 1
    ):
        raise MassiveAdaptiveRLFinalOuterExecutionV1Error(
            "profitability report requires the exact persisted outer-three state"
        )
    position = positions[0]
    if position == len(states) - 1:
        return allow_materialize
    if (
        states[position + 1].stage
        is not MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED
    ):
        raise MassiveAdaptiveRLFinalOuterExecutionV1Error(
            "historical outer-three state has no exact report successor"
        )
    return False


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFinalOuterExecutionV1:
    outer_executions: tuple[MassiveAdaptiveRLOuterFoldExecutionV1, ...]
    outer_fold_seals: tuple[MassiveAdaptiveRLOuterFoldSealAuthorityV1, ...]
    profitability_report: MassiveAdaptiveRLProfitabilityReportAuthorityV2
    prequential_state: MassiveAdaptiveRLPrequentialExperimentStateV1

    def validate(self) -> None:
        if len(self.outer_executions) != 2 or len(self.outer_fold_seals) != 4:
            raise MassiveAdaptiveRLFinalOuterExecutionV1Error(
                "final outer execution inventory differs"
            )
        for authority in (
            *self.outer_executions,
            *self.outer_fold_seals,
            self.profitability_report,
            self.prequential_state,
        ):
            authority.validate()
        outer_two, outer_three = self.outer_executions
        report = self.profitability_report
        state = self.prequential_state
        if (
            tuple(row.fold_index for row in self.outer_executions) != (2, 3)
            or tuple(row.fold_index for row in self.outer_fold_seals) != (0, 1, 2, 3)
            or outer_two.outer_fold_seal.semantic_receipt_sha256
            != self.outer_fold_seals[2].semantic_receipt_sha256
            or outer_three.outer_fold_seal.semantic_receipt_sha256
            != self.outer_fold_seals[3].semantic_receipt_sha256
            or outer_three.outer_access.predecessor_state_receipt_sha256
            != outer_two.prequential_state.semantic_receipt_sha256
            or report.outer_fold_seal_receipts
            != tuple(row.semantic_receipt_sha256 for row in self.outer_fold_seals)
            or not report.development_profitability_reporting_authorized
            or report.end_to_end_profitability_execution_complete
            or state.stage
            is not MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED
            or state.stage_artifact_semantic_receipt_sha256
            != report.semantic_receipt_sha256
            or not state.development_profitability_reporting_authorized
            or state.full_cold_replay_verified
            or state.positive_profitability_authorization_eligible
        ):
            raise MassiveAdaptiveRLFinalOuterExecutionV1Error(
                "final outer execution report lineage differs"
            )


def run_or_resume_massive_adaptive_rl_final_outer_execution_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    outer_zero_execution: MassiveAdaptiveRLOuterFoldExecutionV1,
    continuation: MassiveAdaptiveRLPrequentialContinuationV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFinalOuterExecutionV1:
    """Execute O2, O3, and Report V2 from the exact policy-three state."""

    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or type(outer_zero_execution) is not MassiveAdaptiveRLOuterFoldExecutionV1
        or type(continuation) is not MassiveAdaptiveRLPrequentialContinuationV1
        or type(allow_materialize) is not bool
    ):
        raise MassiveAdaptiveRLFinalOuterExecutionV1Error(
            "final outer execution requires exact Manifest-V5 authorities"
        )
    for authority in (
        manifest,
        manifest_registration,
        execution_registration,
        outer_zero_execution,
        continuation,
    ):
        authority.validate()
    schedule = continuation.policy_schedules[-1]
    outer_one_execution = continuation.outer_one_execution
    predecessor_state = continuation.prequential_state
    if (
        not manifest_registration.development_protocol_registered
        or not execution_registration.development_execution_registered
        or outer_zero_execution.fold_index != 0
        or outer_one_execution.fold_index != 1
        or not schedule.complete_schedule
        or not schedule.development_stage_authorized
        or predecessor_state.stage
        is not MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN
        or predecessor_state.stage_artifact_semantic_receipt_sha256
        != schedule.semantic_receipt_sha256
        or schedule.predecessor_outer_fold_seal_receipts
        != (
            outer_zero_execution.outer_fold_seal.semantic_receipt_sha256,
            outer_one_execution.outer_fold_seal.semantic_receipt_sha256,
        )
    ):
        raise MassiveAdaptiveRLFinalOuterExecutionV1Error(
            "final outer execution causal roots differ"
        )
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root,
        experiment_id=manifest.experiment_id,
    ):
        outer_two = run_or_resume_massive_adaptive_rl_outer_fold_execution_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            policy_schedule=schedule,
            predecessor_state=predecessor_state,
            allow_materialize=allow_materialize,
        )
        outer_three = run_or_resume_massive_adaptive_rl_outer_fold_execution_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            policy_schedule=schedule,
            predecessor_state=outer_two.prequential_state,
            allow_materialize=allow_materialize,
        )
        report_allow_materialize = _report_materialization_mode_after_state_v1(
            root=root,
            manifest=manifest,
            predecessor_state=outer_three.prequential_state,
            allow_materialize=allow_materialize,
        )
        seals = (
            outer_zero_execution.outer_fold_seal,
            outer_one_execution.outer_fold_seal,
            outer_two.outer_fold_seal,
            outer_three.outer_fold_seal,
        )
        report = run_or_resume_massive_adaptive_rl_profitability_report_authority_v2(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            policy_schedule=schedule,
            outer_fold_seals=seals,
            allow_materialize=report_allow_materialize,
        )
        state = run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            stage_artifact=report,
            allow_materialize=report_allow_materialize,
        )
    result = MassiveAdaptiveRLFinalOuterExecutionV1(
        outer_executions=(outer_two, outer_three),
        outer_fold_seals=seals,
        profitability_report=report,
        prequential_state=state,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLFinalOuterExecutionV1",
    "MassiveAdaptiveRLFinalOuterExecutionV1Error",
    "run_or_resume_massive_adaptive_rl_final_outer_execution_v1",
]
