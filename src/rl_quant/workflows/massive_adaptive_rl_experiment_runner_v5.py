"""Manifest-V5 root through the complete development profitability report.

V5 is the unique authoring generation for the prequential experiment.  It
registers that ownership before resuming training, freezes the qualified
evaluation implementation after causal training, and only then permits the
fold-0/1 validation inputs.  It evaluates and freezes both initial policy pairs,
publishes schedule prefixes, executes all four causally ordered outer folds,
releases delayed validations only after their predecessor seals, and publishes
the four-fold profitability report.  Full end-to-end completion remains false
until the separate nonmaterializing cold-replay finalizer verifies that report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
    run_or_resume_massive_adaptive_rl_initial_validation_release_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MassiveAdaptiveRLExperimentLockV1Error,
    MassiveAdaptiveRLExperimentLockV1Unavailable,
    massive_adaptive_rl_experiment_orchestration_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    execution_implementation_registration_transaction_state_v1,
    massive_adaptive_rl_preimplementation_economic_evidence_v1,
    run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    MassiveAdaptiveRLEndToEndRunV2,
    _run_massive_adaptive_rl_experiment_v2_unlocked,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v4 import (
    MassiveAdaptiveRLPrequentialRunV4,
    _replay_prequential_root_with_inputs,
    _validate_training_handoff,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2,
    load_massive_adaptive_rl_experiment_states_v2,
)
from rl_quant.workflows.massive_adaptive_rl_final_outer_execution_v1 import (
    MassiveAdaptiveRLFinalOuterExecutionV1,
    run_or_resume_massive_adaptive_rl_final_outer_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1,
    MassiveAdaptiveRLExperimentManifestV5,
    load_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_initial_inputs_capability_v1,
    issue_massive_adaptive_rl_manifest_v5_training_capability_v1,
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_outer_fold_execution_v1 import (
    MassiveAdaptiveRLOuterFoldExecutionV1,
    run_or_resume_massive_adaptive_rl_outer_fold_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_continuation_v1 import (
    MassiveAdaptiveRLPrequentialContinuationV1,
    run_or_resume_massive_adaptive_rl_prequential_continuation_v1,
)
from rl_quant.workflows.massive_adaptive_rl_initial_validation_execution_v1 import (
    MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1,
    MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_QUALIFIED_V1,
    MassiveAdaptiveRLInitialValidationExecutionV1,
    run_or_resume_massive_adaptive_rl_initial_validation_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_experiment_state_v1 import (
    MassiveAdaptiveRLPrequentialExperimentStateV1,
    MassiveAdaptiveRLPrequentialStageV1,
    run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_DIAGNOSTIC_V1,
    MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1,
    MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V5_SCHEMA = (
    "rl-quant.massive-adaptive-rl-prequential-run-v5"
)


class MassiveAdaptiveRLExperimentRunnerV5Error(ValueError):
    """The Manifest-V5 prequential root cannot advance or replay safely."""


class MassiveAdaptiveRLExperimentRunnerV5LeaseUnavailable(RuntimeError):
    """Another process owns the experiment-global writer lock."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveRLExperimentRunnerV5Error(f"{name} is absent")
    return _digest(name, value)


def _required_timestamp(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLExperimentRunnerV5Error(f"{name} is absent or invalid")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPrequentialRunV5:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    base_manifest_v4_receipt_sha256: str
    manifest_v5_registration_authority_receipt_sha256: str
    manifest_v5_registration_source_receipt_sha256: str
    manifest_v5_registration_commit_receipt_sha256: str
    manifest_v5_registration_committed_at_ms: int
    training_state_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    predecessor_run_v4_receipt_sha256: str | None
    initial_validation_inputs_authority_receipt_sha256: str | None
    initial_validation_inputs_source_receipt_sha256: str | None
    initial_validation_inputs_commit_receipt_sha256: str | None
    initial_validation_inputs_committed_at_ms: int | None
    initial_validation_release_authority_receipt_sha256: str | None
    initial_validation_release_source_receipt_sha256: str | None
    initial_validation_release_commit_receipt_sha256: str | None
    initial_validation_release_committed_at_ms: int | None
    execution_implementation_registration_authority_receipt_sha256: str | None
    execution_implementation_registration_source_receipt_sha256: str | None
    execution_implementation_registration_commit_receipt_sha256: str | None
    execution_implementation_registration_committed_at_ms: int | None
    initial_fold_validation_authority_receipts: tuple[str, ...]
    initial_policy_selection_authority_receipts: tuple[str, ...]
    initial_frozen_ppo_policy_receipts: tuple[str, ...]
    initial_frozen_fc06_control_receipts: tuple[str, ...]
    initial_policy_schedule_prefix_receipts: tuple[str, ...]
    initial_policy_schedule_disposition: str | None
    delayed_validation_release_authority_receipts: tuple[str, ...]
    delayed_fold_validation_authority_receipts: tuple[str, ...]
    delayed_policy_selection_authority_receipts: tuple[str, ...]
    delayed_frozen_ppo_policy_receipts: tuple[str, ...]
    delayed_frozen_fc06_control_receipts: tuple[str, ...]
    delayed_policy_schedule_prefix_receipts: tuple[str, ...]
    outer_access_commitment_receipts: tuple[str, ...]
    outer_input_authority_receipts: tuple[str, ...]
    outer_rollout_authority_receipts: tuple[str, ...]
    outer_fold_seal_authority_receipts: tuple[str, ...]
    sealed_outer_fold_indices: tuple[int, ...]
    profitability_report_authority_receipt_sha256: str | None
    profitability_report_source_receipt_sha256: str | None
    profitability_report_commit_receipt_sha256: str | None
    profitability_report_committed_at_ms: int | None
    profitability_gates_passed: bool | None
    prequential_state_head_receipt_sha256: str | None
    prequential_state_head_source_receipt_sha256: str | None
    prequential_state_head_commit_receipt_sha256: str | None
    prequential_state_head_committed_at_ms: int | None
    prequential_state_head_stage: str | None
    released_validation_fold_indices: tuple[int, ...]
    withheld_validation_fold_indices: tuple[int, ...]
    authoritative_writer_generation: str
    protocol_registered: bool
    training_evidence_adopted: bool
    source_generation_v2_replayed: bool
    initial_validation_inputs_replayed: bool
    initial_validation_release_replayed: bool
    execution_implementation_registered: bool
    diagnostic_continuation_registered: bool
    validation_execution_complete: bool
    initial_policy_freezing_complete: bool
    outer_zero_preparation_authorized: bool
    outer_zero_execution_complete: bool
    outer_one_execution_complete: bool
    outer_two_execution_complete: bool
    outer_three_execution_complete: bool
    next_required_stage: str
    semantic_receipt_sha256: str
    policy_schedule_disposition: str | None = None
    final_policy_freezing_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    end_to_end_profitability_execution_complete: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V5_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    @property
    def positive_profitability_authorization_eligible(self) -> bool:
        return False

    def validate(self) -> None:
        initial_receipts = (
            self.predecessor_run_v4_receipt_sha256,
            self.initial_validation_inputs_authority_receipt_sha256,
            self.initial_validation_inputs_source_receipt_sha256,
            self.initial_validation_inputs_commit_receipt_sha256,
        )
        initial_present = all(value is not None for value in initial_receipts)
        implementation_receipts = (
            self.execution_implementation_registration_authority_receipt_sha256,
            self.execution_implementation_registration_source_receipt_sha256,
            self.execution_implementation_registration_commit_receipt_sha256,
        )
        implementation_present = all(
            value is not None for value in implementation_receipts
        )
        release_receipts = (
            self.initial_validation_release_authority_receipt_sha256,
            self.initial_validation_release_source_receipt_sha256,
            self.initial_validation_release_commit_receipt_sha256,
        )
        release_present = all(value is not None for value in release_receipts)
        implementation_committed_at_ms = (
            self.execution_implementation_registration_committed_at_ms
        )
        initial_committed_at_ms = self.initial_validation_inputs_committed_at_ms
        release_committed_at_ms = self.initial_validation_release_committed_at_ms
        implementation_chronology_invalid = bool(
            implementation_present
            and implementation_committed_at_ms is not None
            and (
                self.manifest_v5_registration_committed_at_ms
                >= implementation_committed_at_ms
                or initial_present
                and initial_committed_at_ms is not None
                and implementation_committed_at_ms >= initial_committed_at_ms
            )
        )
        initial_execution_inventories = (
            self.initial_fold_validation_authority_receipts,
            self.initial_policy_selection_authority_receipts,
            self.initial_frozen_ppo_policy_receipts,
            self.initial_frozen_fc06_control_receipts,
        )
        initial_execution_present = all(
            len(inventory) == 2 for inventory in initial_execution_inventories
        )
        initial_execution_partial = any(initial_execution_inventories) and not (
            initial_execution_present
        )
        delayed_execution_inventories = (
            self.delayed_validation_release_authority_receipts,
            self.delayed_fold_validation_authority_receipts,
            self.delayed_policy_selection_authority_receipts,
            self.delayed_frozen_ppo_policy_receipts,
            self.delayed_frozen_fc06_control_receipts,
            self.delayed_policy_schedule_prefix_receipts,
        )
        delayed_execution_present = all(
            len(inventory) == 2 for inventory in delayed_execution_inventories
        )
        delayed_execution_partial = any(delayed_execution_inventories) and not (
            delayed_execution_present
        )
        outer_inventories = (
            self.outer_access_commitment_receipts,
            self.outer_input_authority_receipts,
            self.outer_rollout_authority_receipts,
            self.outer_fold_seal_authority_receipts,
        )
        outer_lengths = tuple(len(inventory) for inventory in outer_inventories)
        outer_count = outer_lengths[0] if len(set(outer_lengths)) == 1 else -1
        outer_partial = outer_count not in (0, 1, 2, 4)
        outer_zero_present = outer_count >= 1
        outer_one_present = outer_count >= 2
        outer_two_present = outer_count == 4
        outer_three_present = outer_count == 4
        report_receipts = (
            self.profitability_report_authority_receipt_sha256,
            self.profitability_report_source_receipt_sha256,
            self.profitability_report_commit_receipt_sha256,
        )
        report_present = all(value is not None for value in report_receipts)
        state_head_receipts = (
            self.prequential_state_head_receipt_sha256,
            self.prequential_state_head_source_receipt_sha256,
            self.prequential_state_head_commit_receipt_sha256,
        )
        state_head_present = all(value is not None for value in state_head_receipts)
        expected_next_stage = (
            "full-cold-replay-verification"
            if report_present
            else "outer-fold-2-access-and-seal"
            if delayed_execution_present
            else "validation-fold-2-release-selection-and-freeze"
            if outer_zero_present
            else "outer-fold-0-access-and-seal"
            if initial_execution_present
            else "prequential-fold-0-and-fold-1-validation-selection-and-freeze"
            if initial_present
            else "initial-validation-input-commitment"
            if self.execution_implementation_registered
            else "execution-implementation-registration"
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V5_SCHEMA
            or not self.experiment_id
            or any(value is not None for value in initial_receipts) != initial_present
            or any(value is not None for value in implementation_receipts)
            != implementation_present
            or any(value is not None for value in release_receipts) != release_present
            or release_present != initial_present
            or initial_execution_partial
            or initial_execution_present
            and not release_present
            or initial_execution_present
            and any(
                len(set(inventory)) != 2 for inventory in initial_execution_inventories
            )
            or delayed_execution_partial
            or delayed_execution_present
            and not initial_execution_present
            or delayed_execution_present
            and any(
                len(set(inventory)) != 2 for inventory in delayed_execution_inventories
            )
            or outer_partial
            or outer_zero_present
            and not initial_execution_present
            or outer_one_present != delayed_execution_present
            or outer_two_present != report_present
            or outer_three_present != report_present
            or any(value is not None for value in report_receipts) != report_present
            or (self.profitability_report_committed_at_ms is not None) != report_present
            or (self.profitability_gates_passed is not None) != report_present
            or report_present
            and not isinstance(self.profitability_gates_passed, bool)
            or self.sealed_outer_fold_indices != tuple(range(max(0, outer_count)))
            or any(
                len(set(inventory)) != outer_count for inventory in outer_inventories
            )
            or len(self.initial_policy_schedule_prefix_receipts)
            != (2 if initial_execution_present else 0)
            or len(set(self.initial_policy_schedule_prefix_receipts))
            != len(self.initial_policy_schedule_prefix_receipts)
            or any(value is not None for value in state_head_receipts)
            != state_head_present
            or (self.prequential_state_head_stage is not None) != state_head_present
            or (self.prequential_state_head_committed_at_ms is not None)
            != state_head_present
            or state_head_present != initial_execution_present
            or self.prequential_state_head_stage
            != (
                "profitability-report-published"
                if report_present
                else "policy-3-frozen"
                if delayed_execution_present
                else "outer-0-sealed"
                if outer_zero_present
                else "policy-1-frozen"
                if initial_execution_present
                else None
            )
            or self.initial_policy_schedule_disposition
            not in (
                None,
                MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_QUALIFIED_V1,
                MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1,
            )
            or (self.initial_policy_schedule_disposition is not None)
            != initial_execution_present
            or self.policy_schedule_disposition
            not in (
                None,
                MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1,
                MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_DIAGNOSTIC_V1,
            )
            or (self.policy_schedule_disposition is not None)
            != delayed_execution_present
            or self.initial_policy_schedule_disposition
            == MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1
            and self.policy_schedule_disposition
            == MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1
            or self.released_validation_fold_indices
            != (
                (0, 1, 2, 3)
                if delayed_execution_present
                else MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
                if release_present
                else ()
            )
            or self.withheld_validation_fold_indices
            != (
                ()
                if delayed_execution_present
                else MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1
            )
            or self.authoritative_writer_generation
            != "massive-adaptive-rl-experiment-runner-v5"
            or not self.protocol_registered
            or not self.training_evidence_adopted
            or self.source_generation_v2_replayed != initial_present
            or self.initial_validation_inputs_replayed != initial_present
            or self.initial_validation_release_replayed != release_present
            or not self.diagnostic_continuation_registered
            or initial_present
            and initial_committed_at_ms is not None
            and self.manifest_v5_registration_committed_at_ms >= initial_committed_at_ms
            or self.execution_implementation_registered != implementation_present
            or initial_present
            and not self.execution_implementation_registered
            or (initial_committed_at_ms is not None) != initial_present
            or (release_committed_at_ms is not None) != release_present
            or release_present
            and initial_committed_at_ms is not None
            and release_committed_at_ms is not None
            and initial_committed_at_ms >= release_committed_at_ms
            or (self.execution_implementation_registration_committed_at_ms is not None)
            != self.execution_implementation_registered
            or implementation_chronology_invalid
            or self.validation_execution_complete != delayed_execution_present
            or self.initial_policy_freezing_complete != initial_execution_present
            or self.outer_zero_preparation_authorized
            != bool(initial_execution_present and not outer_zero_present)
            or self.outer_zero_execution_complete != outer_zero_present
            or self.outer_one_execution_complete != outer_one_present
            or self.outer_two_execution_complete != outer_two_present
            or self.outer_three_execution_complete != outer_three_present
            or self.next_required_stage != expected_next_stage
            or self.final_policy_freezing_authorized != delayed_execution_present
            or self.outer_access_authorized
            or self.profitability_reporting_authorized != report_present
            or self.end_to_end_profitability_execution_complete
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentRunnerV5Error(
                "adaptive RL prequential run V5 differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256") and value is not None:
                _digest(name, value)
        for inventory in initial_execution_inventories:
            for value in inventory:
                _digest("initial validation execution inventory", value)
        for inventory in delayed_execution_inventories:
            for value in inventory:
                _digest("delayed validation execution inventory", value)
        for inventory in outer_inventories:
            for value in inventory:
                _digest("outer-fold execution inventory", value)
        for value in report_receipts:
            if value is not None:
                _digest("profitability report receipt", value)
        for value in self.initial_policy_schedule_prefix_receipts:
            _digest("initial policy schedule prefix", value)
        _required_timestamp(
            "Manifest V5 registration timestamp",
            self.manifest_v5_registration_committed_at_ms,
        )
        if self.initial_validation_inputs_committed_at_ms is not None:
            _required_timestamp(
                "initial validation-input timestamp",
                self.initial_validation_inputs_committed_at_ms,
            )
        if self.execution_implementation_registration_committed_at_ms is not None:
            _required_timestamp(
                "execution implementation registration timestamp",
                self.execution_implementation_registration_committed_at_ms,
            )
        if self.initial_validation_release_committed_at_ms is not None:
            _required_timestamp(
                "initial validation-release timestamp",
                self.initial_validation_release_committed_at_ms,
            )
        if self.prequential_state_head_committed_at_ms is not None:
            _required_timestamp(
                "prequential state-head timestamp",
                self.prequential_state_head_committed_at_ms,
            )
        if self.profitability_report_committed_at_ms is not None:
            _required_timestamp(
                "profitability report timestamp",
                self.profitability_report_committed_at_ms,
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _training_receipts(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    states: tuple[MassiveAdaptiveRLExperimentStateV2, ...],
) -> tuple[str, str]:
    fit_receipt = _validate_training_handoff(
        manifest=manifest.base_manifest,
        states=states,
    )
    matches = tuple(
        state
        for state in states
        if state.stage
        is MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
    )
    if len(matches) != 1:
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "Manifest V5 completed training state differs"
        )
    return _required_digest(
        "completed training state receipt",
        getattr(matches[0], "semantic_receipt_sha256", None),
    ), fit_receipt


def _build_preimplementation_result(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    states: tuple[MassiveAdaptiveRLExperimentStateV2, ...],
    implementation_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1 | None
    ) = None,
) -> MassiveAdaptiveRLPrequentialRunV5:
    """Return the V5-owned status before any validation input is opened."""

    manifest.validate()
    registration.validate()
    training_state_receipt, fit_receipt = _training_receipts(
        manifest=manifest, states=states
    )
    registration_time = _required_timestamp(
        "Manifest V5 registration timestamp",
        registration.source_transaction_committed_at_ms,
    )
    if (
        not registration.development_protocol_registered
        or registration.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "Manifest V5 registration did not replay before implementation freeze"
        )
    if implementation_registration is not None:
        implementation_registration.validate()
        implementation_time = _required_timestamp(
            "execution implementation registration timestamp",
            implementation_registration.source_transaction_committed_at_ms,
        )
        if (
            not implementation_registration.development_execution_registered
            or implementation_registration.manifest_v5_receipt_sha256
            != manifest.semantic_receipt_sha256
            or implementation_registration.manifest_v5_registration_authority_receipt_sha256
            != registration.semantic_receipt_sha256
            or implementation_registration.training_state_receipt_sha256
            != training_state_receipt
            or implementation_registration.four_fold_fit_authority_receipt_sha256
            != fit_receipt
            or implementation_time <= registration_time
        ):
            raise MassiveAdaptiveRLExperimentRunnerV5Error(
                "Manifest V5 execution implementation did not replay before inputs"
            )
    else:
        implementation_time = None
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "base_manifest_v4_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "manifest_v5_registration_authority_receipt_sha256": (
            registration.semantic_receipt_sha256
        ),
        "manifest_v5_registration_source_receipt_sha256": _required_digest(
            "Manifest V5 registration source receipt",
            registration.source_receipt_sha256,
        ),
        "manifest_v5_registration_commit_receipt_sha256": _required_digest(
            "Manifest V5 registration commit receipt",
            registration.source_transaction_receipt_sha256,
        ),
        "manifest_v5_registration_committed_at_ms": registration_time,
        "training_state_receipt_sha256": training_state_receipt,
        "four_fold_fit_authority_receipt_sha256": fit_receipt,
        "predecessor_run_v4_receipt_sha256": None,
        "initial_validation_inputs_authority_receipt_sha256": None,
        "initial_validation_inputs_source_receipt_sha256": None,
        "initial_validation_inputs_commit_receipt_sha256": None,
        "initial_validation_inputs_committed_at_ms": None,
        "initial_validation_release_authority_receipt_sha256": None,
        "initial_validation_release_source_receipt_sha256": None,
        "initial_validation_release_commit_receipt_sha256": None,
        "initial_validation_release_committed_at_ms": None,
        "execution_implementation_registration_authority_receipt_sha256": (
            None
            if implementation_registration is None
            else implementation_registration.semantic_receipt_sha256
        ),
        "execution_implementation_registration_source_receipt_sha256": (
            None
            if implementation_registration is None
            else _required_digest(
                "execution implementation registration source receipt",
                implementation_registration.source_receipt_sha256,
            )
        ),
        "execution_implementation_registration_commit_receipt_sha256": (
            None
            if implementation_registration is None
            else _required_digest(
                "execution implementation registration commit receipt",
                implementation_registration.source_transaction_receipt_sha256,
            )
        ),
        "execution_implementation_registration_committed_at_ms": implementation_time,
        "initial_fold_validation_authority_receipts": (),
        "initial_policy_selection_authority_receipts": (),
        "initial_frozen_ppo_policy_receipts": (),
        "initial_frozen_fc06_control_receipts": (),
        "initial_policy_schedule_prefix_receipts": (),
        "initial_policy_schedule_disposition": None,
        "delayed_validation_release_authority_receipts": (),
        "delayed_fold_validation_authority_receipts": (),
        "delayed_policy_selection_authority_receipts": (),
        "delayed_frozen_ppo_policy_receipts": (),
        "delayed_frozen_fc06_control_receipts": (),
        "delayed_policy_schedule_prefix_receipts": (),
        "outer_access_commitment_receipts": (),
        "outer_input_authority_receipts": (),
        "outer_rollout_authority_receipts": (),
        "outer_fold_seal_authority_receipts": (),
        "sealed_outer_fold_indices": (),
        "profitability_report_authority_receipt_sha256": None,
        "profitability_report_source_receipt_sha256": None,
        "profitability_report_commit_receipt_sha256": None,
        "profitability_report_committed_at_ms": None,
        "profitability_gates_passed": None,
        "prequential_state_head_receipt_sha256": None,
        "prequential_state_head_source_receipt_sha256": None,
        "prequential_state_head_commit_receipt_sha256": None,
        "prequential_state_head_committed_at_ms": None,
        "prequential_state_head_stage": None,
        "released_validation_fold_indices": (),
        "withheld_validation_fold_indices": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1
        ),
        "authoritative_writer_generation": manifest.authoritative_writer_generation,
        "protocol_registered": True,
        "training_evidence_adopted": True,
        "source_generation_v2_replayed": False,
        "initial_validation_inputs_replayed": False,
        "initial_validation_release_replayed": False,
        "execution_implementation_registered": implementation_registration is not None,
        "diagnostic_continuation_registered": True,
        "validation_execution_complete": False,
        "initial_policy_freezing_complete": False,
        "outer_zero_preparation_authorized": False,
        "outer_zero_execution_complete": False,
        "outer_one_execution_complete": False,
        "outer_two_execution_complete": False,
        "outer_three_execution_complete": False,
        "next_required_stage": (
            "execution-implementation-registration"
            if implementation_registration is None
            else "initial-validation-input-commitment"
        ),
        "policy_schedule_disposition": None,
        "final_policy_freezing_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "end_to_end_profitability_execution_complete": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V5_SCHEMA,
    }
    provisional = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _build_result(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    predecessor: MassiveAdaptiveRLPrequentialRunV4,
    training_state_receipt_sha256: str,
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    validation_release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    implementation_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
) -> MassiveAdaptiveRLPrequentialRunV5:
    manifest.validate()
    registration.validate()
    predecessor.validate()
    initial_inputs.validate()
    validation_release.validate()
    implementation_registration.validate()
    initial = initial_inputs
    registration_committed_at_ms = _required_timestamp(
        "Manifest V5 registration timestamp",
        registration.source_transaction_committed_at_ms,
    )
    initial_committed_at_ms = _required_timestamp(
        "initial validation-input timestamp",
        initial.source_transaction_committed_at_ms,
    )
    release_committed_at_ms = _required_timestamp(
        "initial validation-release timestamp",
        validation_release.source_transaction_committed_at_ms,
    )
    if (
        not registration.development_protocol_registered
        or predecessor.manifest_v4_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or predecessor.initial_validation_inputs_authority_receipt_sha256
        != initial.semantic_receipt_sha256
        or predecessor.initial_validation_inputs_source_receipt_sha256
        != initial.source_receipt_sha256
        or predecessor.initial_validation_inputs_commit_receipt_sha256
        != initial.source_transaction_receipt_sha256
        or initial.manifest_v4_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or not implementation_registration.development_execution_registered
        or implementation_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or implementation_registration.manifest_v5_registration_authority_receipt_sha256
        != registration.semantic_receipt_sha256
        or implementation_registration.training_state_receipt_sha256
        != training_state_receipt_sha256
        or implementation_registration.four_fold_fit_authority_receipt_sha256
        != predecessor.four_fold_fit_authority_receipt_sha256
        or registration_committed_at_ms
        >= _required_timestamp(
            "execution implementation registration timestamp",
            implementation_registration.source_transaction_committed_at_ms,
        )
        or _required_timestamp(
            "execution implementation registration timestamp",
            implementation_registration.source_transaction_committed_at_ms,
        )
        >= initial_committed_at_ms
        or not validation_release.development_stage_authorized
        or validation_release.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or validation_release.manifest_v5_registration_authority_receipt_sha256
        != registration.semantic_receipt_sha256
        or validation_release.execution_implementation_registration_authority_receipt_sha256
        != implementation_registration.semantic_receipt_sha256
        or validation_release.initial_validation_inputs_authority_receipt_sha256
        != initial.semantic_receipt_sha256
        or validation_release.training_state_receipt_sha256
        != training_state_receipt_sha256
        or validation_release.four_fold_fit_authority_receipt_sha256
        != predecessor.four_fold_fit_authority_receipt_sha256
        or release_committed_at_ms <= initial_committed_at_ms
    ):
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "Manifest V5 initial prequential boundary did not replay"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "base_manifest_v4_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "manifest_v5_registration_authority_receipt_sha256": (
            registration.semantic_receipt_sha256
        ),
        "manifest_v5_registration_source_receipt_sha256": _required_digest(
            "Manifest V5 registration source receipt",
            registration.source_receipt_sha256,
        ),
        "manifest_v5_registration_commit_receipt_sha256": _required_digest(
            "Manifest V5 registration commit receipt",
            registration.source_transaction_receipt_sha256,
        ),
        "manifest_v5_registration_committed_at_ms": registration_committed_at_ms,
        "training_state_receipt_sha256": training_state_receipt_sha256,
        "four_fold_fit_authority_receipt_sha256": (
            predecessor.four_fold_fit_authority_receipt_sha256
        ),
        "predecessor_run_v4_receipt_sha256": predecessor.semantic_receipt_sha256,
        "initial_validation_inputs_authority_receipt_sha256": (
            initial.semantic_receipt_sha256
        ),
        "initial_validation_inputs_source_receipt_sha256": _required_digest(
            "initial validation-input source receipt",
            initial.source_receipt_sha256,
        ),
        "initial_validation_inputs_commit_receipt_sha256": _required_digest(
            "initial validation-input commit receipt",
            initial.source_transaction_receipt_sha256,
        ),
        "initial_validation_inputs_committed_at_ms": initial_committed_at_ms,
        "initial_validation_release_authority_receipt_sha256": (
            validation_release.semantic_receipt_sha256
        ),
        "initial_validation_release_source_receipt_sha256": _required_digest(
            "initial validation-release source receipt",
            validation_release.source_receipt_sha256,
        ),
        "initial_validation_release_commit_receipt_sha256": _required_digest(
            "initial validation-release commit receipt",
            validation_release.source_transaction_receipt_sha256,
        ),
        "initial_validation_release_committed_at_ms": release_committed_at_ms,
        "execution_implementation_registration_authority_receipt_sha256": (
            implementation_registration.semantic_receipt_sha256
        ),
        "execution_implementation_registration_source_receipt_sha256": (
            _required_digest(
                "execution implementation registration source receipt",
                implementation_registration.source_receipt_sha256,
            )
        ),
        "execution_implementation_registration_commit_receipt_sha256": (
            _required_digest(
                "execution implementation registration commit receipt",
                implementation_registration.source_transaction_receipt_sha256,
            )
        ),
        "execution_implementation_registration_committed_at_ms": (
            _required_timestamp(
                "execution implementation registration timestamp",
                implementation_registration.source_transaction_committed_at_ms,
            )
        ),
        "initial_fold_validation_authority_receipts": (),
        "initial_policy_selection_authority_receipts": (),
        "initial_frozen_ppo_policy_receipts": (),
        "initial_frozen_fc06_control_receipts": (),
        "initial_policy_schedule_prefix_receipts": (),
        "initial_policy_schedule_disposition": None,
        "delayed_validation_release_authority_receipts": (),
        "delayed_fold_validation_authority_receipts": (),
        "delayed_policy_selection_authority_receipts": (),
        "delayed_frozen_ppo_policy_receipts": (),
        "delayed_frozen_fc06_control_receipts": (),
        "delayed_policy_schedule_prefix_receipts": (),
        "outer_access_commitment_receipts": (),
        "outer_input_authority_receipts": (),
        "outer_rollout_authority_receipts": (),
        "outer_fold_seal_authority_receipts": (),
        "sealed_outer_fold_indices": (),
        "profitability_report_authority_receipt_sha256": None,
        "profitability_report_source_receipt_sha256": None,
        "profitability_report_commit_receipt_sha256": None,
        "profitability_report_committed_at_ms": None,
        "profitability_gates_passed": None,
        "prequential_state_head_receipt_sha256": None,
        "prequential_state_head_source_receipt_sha256": None,
        "prequential_state_head_commit_receipt_sha256": None,
        "prequential_state_head_committed_at_ms": None,
        "prequential_state_head_stage": None,
        "released_validation_fold_indices": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
        ),
        "withheld_validation_fold_indices": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1
        ),
        "authoritative_writer_generation": manifest.authoritative_writer_generation,
        "protocol_registered": True,
        "training_evidence_adopted": predecessor.training_evidence_adopted,
        "source_generation_v2_replayed": predecessor.source_generation_v2_replayed,
        "initial_validation_inputs_replayed": (
            predecessor.initial_validation_inputs_replayed
        ),
        "initial_validation_release_replayed": True,
        "execution_implementation_registered": True,
        "diagnostic_continuation_registered": (
            predecessor.diagnostic_continuation_registered
        ),
        "validation_execution_complete": False,
        "initial_policy_freezing_complete": False,
        "outer_zero_preparation_authorized": False,
        "outer_zero_execution_complete": False,
        "outer_one_execution_complete": False,
        "outer_two_execution_complete": False,
        "outer_three_execution_complete": False,
        "next_required_stage": (
            "prequential-fold-0-and-fold-1-validation-selection-and-freeze"
        ),
        "policy_schedule_disposition": None,
        "final_policy_freezing_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "end_to_end_profitability_execution_complete": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V5_SCHEMA,
    }
    provisional = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _build_initial_execution_result(
    *,
    boundary: MassiveAdaptiveRLPrequentialRunV5,
    execution: MassiveAdaptiveRLInitialValidationExecutionV1,
    policy_schedule_prefixes: tuple[MassiveAdaptiveRLWalkForwardPolicyScheduleV1, ...],
    prequential_state_head: MassiveAdaptiveRLPrequentialExperimentStateV1,
) -> MassiveAdaptiveRLPrequentialRunV5:
    """Promote the status envelope only after both initial policy pairs replay."""

    boundary.validate()
    execution.validate()
    if (
        any(
            type(schedule) is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
            for schedule in policy_schedule_prefixes
        )
        or type(prequential_state_head)
        is not MassiveAdaptiveRLPrequentialExperimentStateV1
    ):
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "initial V5 schedule or state generation differs"
        )
    for schedule in policy_schedule_prefixes:
        schedule.validate()
    prequential_state_head.validate()
    if (
        type(execution) is not MassiveAdaptiveRLInitialValidationExecutionV1
        or not execution.initial_policy_freezing_complete
        or not execution.outer_zero_preparation_authorized
        or execution.experiment_id != boundary.experiment_id
        or execution.manifest_v5_receipt_sha256 != boundary.manifest_v5_receipt_sha256
        or execution.validation_release_authority_receipt_sha256
        != boundary.initial_validation_release_authority_receipt_sha256
        or execution.execution_implementation_registration_receipt_sha256
        != boundary.execution_implementation_registration_authority_receipt_sha256
        or tuple(schedule.fold_indices for schedule in policy_schedule_prefixes)
        != ((0,), (0, 1))
        or any(
            not schedule.development_stage_authorized
            or schedule.experiment_id != boundary.experiment_id
            or schedule.manifest_v5_receipt_sha256
            != boundary.manifest_v5_receipt_sha256
            for schedule in policy_schedule_prefixes
        )
        or prequential_state_head.stage.value != "policy-1-frozen"
        or not prequential_state_head.prequential_execution_authorized
        or prequential_state_head.experiment_id != boundary.experiment_id
        or prequential_state_head.manifest_v5_receipt_sha256
        != boundary.manifest_v5_receipt_sha256
        or prequential_state_head.execution_implementation_registration_receipt_sha256
        != boundary.execution_implementation_registration_authority_receipt_sha256
        or prequential_state_head.stage_artifact_semantic_receipt_sha256
        != policy_schedule_prefixes[-1].semantic_receipt_sha256
        or prequential_state_head.policy_schedule_disposition
        != policy_schedule_prefixes[-1].policy_schedule_disposition
    ):
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "initial V5 policy freezes do not replay from the root boundary"
        )
    body = boundary.semantic_unsigned()
    body.update(
        {
            "initial_fold_validation_authority_receipts": tuple(
                row.semantic_receipt_sha256
                for row in execution.fold_validation_authorities
            ),
            "initial_policy_selection_authority_receipts": tuple(
                row.semantic_receipt_sha256
                for row in execution.policy_selection_authorities
            ),
            "initial_frozen_ppo_policy_receipts": tuple(
                row.semantic_receipt_sha256 for row in execution.frozen_ppo_policies
            ),
            "initial_frozen_fc06_control_receipts": tuple(
                row.semantic_receipt_sha256 for row in execution.frozen_fc06_controls
            ),
            "initial_policy_schedule_prefix_receipts": tuple(
                row.semantic_receipt_sha256 for row in policy_schedule_prefixes
            ),
            "initial_policy_schedule_disposition": (
                execution.policy_schedule_disposition
            ),
            "prequential_state_head_receipt_sha256": (
                prequential_state_head.semantic_receipt_sha256
            ),
            "prequential_state_head_source_receipt_sha256": _required_digest(
                "prequential state-head source receipt",
                prequential_state_head.source_receipt_sha256,
            ),
            "prequential_state_head_commit_receipt_sha256": _required_digest(
                "prequential state-head commit receipt",
                prequential_state_head.source_transaction_receipt_sha256,
            ),
            "prequential_state_head_committed_at_ms": _required_timestamp(
                "prequential state-head timestamp",
                prequential_state_head.source_transaction_committed_at_ms,
            ),
            "prequential_state_head_stage": prequential_state_head.stage.value,
            "initial_policy_freezing_complete": True,
            "outer_zero_preparation_authorized": True,
            "next_required_stage": "outer-fold-0-access-and-seal",
        }
    )
    provisional = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _build_outer_zero_result(
    *,
    boundary: MassiveAdaptiveRLPrequentialRunV5,
    policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    execution: MassiveAdaptiveRLOuterFoldExecutionV1,
) -> MassiveAdaptiveRLPrequentialRunV5:
    """Promote the root envelope only after O0 is economically sealed."""

    boundary.validate()
    policy_schedule.validate()
    execution.validate()
    state = execution.prequential_state
    if (
        not boundary.initial_policy_freezing_complete
        or not boundary.outer_zero_preparation_authorized
        or boundary.prequential_state_head_receipt_sha256
        != execution.outer_access.predecessor_state_receipt_sha256
        or boundary.initial_policy_schedule_prefix_receipts[-1]
        != policy_schedule.semantic_receipt_sha256
        or execution.fold_index != 0
        or execution.outer_access.policy_schedule_receipt_sha256
        != policy_schedule.semantic_receipt_sha256
        or state.stage.value != "outer-0-sealed"
        or not state.prequential_execution_authorized
        or state.immediate_predecessor_state_receipt_sha256
        != boundary.prequential_state_head_receipt_sha256
    ):
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "outer fold zero does not descend from the initial V5 state head"
        )
    body = boundary.semantic_unsigned()
    body.update(
        {
            "outer_access_commitment_receipts": (
                execution.outer_access.semantic_receipt_sha256,
            ),
            "outer_input_authority_receipts": (
                execution.outer_inputs.semantic_receipt_sha256,
            ),
            "outer_rollout_authority_receipts": (
                execution.outer_rollout.semantic_receipt_sha256,
            ),
            "outer_fold_seal_authority_receipts": (
                execution.outer_fold_seal.semantic_receipt_sha256,
            ),
            "sealed_outer_fold_indices": (0,),
            "prequential_state_head_receipt_sha256": state.semantic_receipt_sha256,
            "prequential_state_head_source_receipt_sha256": _required_digest(
                "outer-zero state-head source receipt", state.source_receipt_sha256
            ),
            "prequential_state_head_commit_receipt_sha256": _required_digest(
                "outer-zero state-head commit receipt",
                state.source_transaction_receipt_sha256,
            ),
            "prequential_state_head_committed_at_ms": _required_timestamp(
                "outer-zero state-head timestamp",
                state.source_transaction_committed_at_ms,
            ),
            "prequential_state_head_stage": state.stage.value,
            "outer_zero_preparation_authorized": False,
            "outer_zero_execution_complete": True,
            "next_required_stage": ("validation-fold-2-release-selection-and-freeze"),
        }
    )
    provisional = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _build_delayed_policy_result(
    *,
    boundary: MassiveAdaptiveRLPrequentialRunV5,
    continuation: MassiveAdaptiveRLPrequentialContinuationV1,
) -> MassiveAdaptiveRLPrequentialRunV5:
    """Promote the root after V2/O1/V3 produce the complete policy schedule."""

    boundary.validate()
    continuation.validate()
    release_two, release_three = continuation.validation_releases
    execution_two, execution_three = continuation.released_fold_executions
    schedule_two, schedule_three = continuation.policy_schedules
    outer_one = continuation.outer_one_execution
    state = continuation.prequential_state
    if (
        not boundary.outer_zero_execution_complete
        or boundary.prequential_state_head_receipt_sha256
        != release_two.predecessor_state_receipt_sha256
        or boundary.outer_fold_seal_authority_receipts[-1]
        != release_two.predecessor_outer_fold_seal_receipt_sha256
        or outer_one.prequential_state.semantic_receipt_sha256
        != release_three.predecessor_state_receipt_sha256
        or outer_one.outer_fold_seal.semantic_receipt_sha256
        != release_three.predecessor_outer_fold_seal_receipt_sha256
        or schedule_three.fold_indices != (0, 1, 2, 3)
        or state.stage is not MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN
        or state.immediate_predecessor_state_receipt_sha256 is None
    ):
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "delayed policy sequence does not descend from outer fold zero"
        )
    body = boundary.semantic_unsigned()
    body.update(
        {
            "delayed_validation_release_authority_receipts": (
                release_two.semantic_receipt_sha256,
                release_three.semantic_receipt_sha256,
            ),
            "delayed_fold_validation_authority_receipts": (
                execution_two.fold_validation_authority.semantic_receipt_sha256,
                execution_three.fold_validation_authority.semantic_receipt_sha256,
            ),
            "delayed_policy_selection_authority_receipts": (
                execution_two.policy_selection_authority.semantic_receipt_sha256,
                execution_three.policy_selection_authority.semantic_receipt_sha256,
            ),
            "delayed_frozen_ppo_policy_receipts": (
                execution_two.frozen_ppo_policy.semantic_receipt_sha256,
                execution_three.frozen_ppo_policy.semantic_receipt_sha256,
            ),
            "delayed_frozen_fc06_control_receipts": (
                execution_two.frozen_fc06_control.semantic_receipt_sha256,
                execution_three.frozen_fc06_control.semantic_receipt_sha256,
            ),
            "delayed_policy_schedule_prefix_receipts": (
                schedule_two.semantic_receipt_sha256,
                schedule_three.semantic_receipt_sha256,
            ),
            "outer_access_commitment_receipts": (
                *boundary.outer_access_commitment_receipts,
                outer_one.outer_access.semantic_receipt_sha256,
            ),
            "outer_input_authority_receipts": (
                *boundary.outer_input_authority_receipts,
                outer_one.outer_inputs.semantic_receipt_sha256,
            ),
            "outer_rollout_authority_receipts": (
                *boundary.outer_rollout_authority_receipts,
                outer_one.outer_rollout.semantic_receipt_sha256,
            ),
            "outer_fold_seal_authority_receipts": (
                *boundary.outer_fold_seal_authority_receipts,
                outer_one.outer_fold_seal.semantic_receipt_sha256,
            ),
            "sealed_outer_fold_indices": (0, 1),
            "prequential_state_head_receipt_sha256": state.semantic_receipt_sha256,
            "prequential_state_head_source_receipt_sha256": _required_digest(
                "policy-three state-head source receipt", state.source_receipt_sha256
            ),
            "prequential_state_head_commit_receipt_sha256": _required_digest(
                "policy-three state-head commit receipt",
                state.source_transaction_receipt_sha256,
            ),
            "prequential_state_head_committed_at_ms": _required_timestamp(
                "policy-three state-head timestamp",
                state.source_transaction_committed_at_ms,
            ),
            "prequential_state_head_stage": state.stage.value,
            "released_validation_fold_indices": (0, 1, 2, 3),
            "withheld_validation_fold_indices": (),
            "validation_execution_complete": True,
            "outer_one_execution_complete": True,
            "next_required_stage": "outer-fold-2-access-and-seal",
            "policy_schedule_disposition": schedule_three.policy_schedule_disposition,
            "final_policy_freezing_authorized": True,
        }
    )
    provisional = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _build_profitability_report_result(
    *,
    boundary: MassiveAdaptiveRLPrequentialRunV5,
    execution: MassiveAdaptiveRLFinalOuterExecutionV1,
) -> MassiveAdaptiveRLPrequentialRunV5:
    """Promote the root envelope after O2, O3, and Report V2 replay."""

    boundary.validate()
    execution.validate()
    outer_two, outer_three = execution.outer_executions
    report = execution.profitability_report
    state = execution.prequential_state
    if (
        not boundary.final_policy_freezing_authorized
        or not boundary.outer_one_execution_complete
        or boundary.prequential_state_head_receipt_sha256
        != outer_two.outer_access.predecessor_state_receipt_sha256
        or outer_three.outer_access.predecessor_state_receipt_sha256
        != outer_two.prequential_state.semantic_receipt_sha256
        or tuple(row.semantic_receipt_sha256 for row in execution.outer_fold_seals[:2])
        != boundary.outer_fold_seal_authority_receipts
        or state.stage
        is not MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED
        or state.stage_artifact_semantic_receipt_sha256
        != report.semantic_receipt_sha256
        or not state.development_profitability_reporting_authorized
    ):
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "profitability report does not descend from the policy-three state"
        )
    body = boundary.semantic_unsigned()
    body.update(
        {
            "outer_access_commitment_receipts": (
                *boundary.outer_access_commitment_receipts,
                outer_two.outer_access.semantic_receipt_sha256,
                outer_three.outer_access.semantic_receipt_sha256,
            ),
            "outer_input_authority_receipts": (
                *boundary.outer_input_authority_receipts,
                outer_two.outer_inputs.semantic_receipt_sha256,
                outer_three.outer_inputs.semantic_receipt_sha256,
            ),
            "outer_rollout_authority_receipts": (
                *boundary.outer_rollout_authority_receipts,
                outer_two.outer_rollout.semantic_receipt_sha256,
                outer_three.outer_rollout.semantic_receipt_sha256,
            ),
            "outer_fold_seal_authority_receipts": tuple(
                row.semantic_receipt_sha256 for row in execution.outer_fold_seals
            ),
            "sealed_outer_fold_indices": (0, 1, 2, 3),
            "profitability_report_authority_receipt_sha256": (
                report.semantic_receipt_sha256
            ),
            "profitability_report_source_receipt_sha256": _required_digest(
                "profitability report source receipt",
                report.source_receipt_sha256,
            ),
            "profitability_report_commit_receipt_sha256": _required_digest(
                "profitability report commit receipt",
                report.source_transaction_receipt_sha256,
            ),
            "profitability_report_committed_at_ms": _required_timestamp(
                "profitability report timestamp",
                report.source_transaction_committed_at_ms,
            ),
            "profitability_gates_passed": report.profitability_gates_passed,
            "prequential_state_head_receipt_sha256": state.semantic_receipt_sha256,
            "prequential_state_head_source_receipt_sha256": _required_digest(
                "report state-head source receipt",
                state.source_receipt_sha256,
            ),
            "prequential_state_head_commit_receipt_sha256": _required_digest(
                "report state-head commit receipt",
                state.source_transaction_receipt_sha256,
            ),
            "prequential_state_head_committed_at_ms": _required_timestamp(
                "report state-head timestamp",
                state.source_transaction_committed_at_ms,
            ),
            "prequential_state_head_stage": state.stage.value,
            "outer_two_execution_complete": True,
            "outer_three_execution_complete": True,
            "next_required_stage": "full-cold-replay-verification",
            "profitability_reporting_authorized": True,
        }
    )
    provisional = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLPrequentialRunV5(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _record_initial_prequential_execution_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    implementation_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    validation_release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    execution: MassiveAdaptiveRLInitialValidationExecutionV1,
    allow_materialize: bool,
) -> tuple[
    tuple[MassiveAdaptiveRLWalkForwardPolicyScheduleV1, ...],
    MassiveAdaptiveRLPrequentialExperimentStateV1,
]:
    """Persist the two initial schedule prefixes and their exact state chain."""

    initial_stage_artifacts = (
        validation_release.four_fold_fit_authority,
        implementation_registration,
        validation_release,
    )
    head: MassiveAdaptiveRLPrequentialExperimentStateV1 | None = None
    for artifact in initial_stage_artifacts:
        head = run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
            root=root,
            manifest=manifest,
            manifest_registration=registration,
            execution_registration=implementation_registration,
            stage_artifact=artifact,
            allow_materialize=allow_materialize,
        )
    schedule_zero = run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1(
        root=root,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation_registration,
        frozen_ppo_policies=execution.frozen_ppo_policies[:1],
        frozen_fc06_controls=execution.frozen_fc06_controls[:1],
        allow_materialize=allow_materialize,
    )
    head = run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
        root=root,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation_registration,
        stage_artifact=schedule_zero,
        allow_materialize=allow_materialize,
    )
    schedule_one = run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1(
        root=root,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation_registration,
        frozen_ppo_policies=execution.frozen_ppo_policies,
        frozen_fc06_controls=execution.frozen_fc06_controls,
        allow_materialize=allow_materialize,
    )
    head = run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
        root=root,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation_registration,
        stage_artifact=schedule_one,
        allow_materialize=allow_materialize,
    )
    if head is None:
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "initial V5 prequential state prefix is absent"
        )
    return (schedule_zero, schedule_one), head


def _replay_v5_boundary(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    allow_materialize: bool,
) -> MassiveAdaptiveRLPrequentialRunV5:
    states = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    training_state_receipt, _fit_receipt = _training_receipts(
        manifest=manifest, states=states
    )
    complete, partial = execution_implementation_registration_transaction_state_v1(
        root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    if partial:
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "execution implementation registration transaction is incomplete"
        )
    if not complete:
        if massive_adaptive_rl_preimplementation_economic_evidence_v1(
            root=artifact_root,
            manifest=manifest,
        ):
            raise MassiveAdaptiveRLExperimentRunnerV5Error(
                "validation inputs exist before execution implementation registration"
            )
        return _build_preimplementation_result(
            manifest=manifest,
            registration=registration,
            states=states,
        )
    implementation_registration = (
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=artifact_root,
            manifest=manifest,
            manifest_registration=registration,
            allow_materialize=False,
        )
    )
    initial_inputs_capability = (
        issue_massive_adaptive_rl_manifest_v5_initial_inputs_capability_v1(
            root=artifact_root,
            authority=registration,
            source_root=source_root,
        )
    )
    with massive_adaptive_rl_manifest_v5_writer_scope_v1(
        root=artifact_root,
        capability=initial_inputs_capability,
    ):
        predecessor, initial_inputs = _replay_prequential_root_with_inputs(
            manifest=manifest.base_manifest,
            source_root=source_root,
            artifact_root=artifact_root,
            device=device,
            states=states,
            allow_materialize=allow_materialize,
            v5_writer_capability=initial_inputs_capability,
        )
    validation_release = (
        run_or_resume_massive_adaptive_rl_initial_validation_release_v1(
            root=artifact_root,
            manifest=manifest,
            manifest_registration=registration,
            execution_registration=implementation_registration,
            initial_inputs=initial_inputs,
            allow_materialize=allow_materialize,
        )
    )
    boundary = _build_result(
        manifest=manifest,
        registration=registration,
        predecessor=predecessor,
        training_state_receipt_sha256=training_state_receipt,
        initial_inputs=initial_inputs,
        validation_release=validation_release,
        implementation_registration=implementation_registration,
    )
    execution = run_or_resume_massive_adaptive_rl_initial_validation_execution_v1(
        root=artifact_root,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation_registration,
        validation_release=validation_release,
        allow_materialize=allow_materialize,
    )
    schedules, state_head = _record_initial_prequential_execution_v1(
        root=artifact_root,
        manifest=manifest,
        registration=registration,
        implementation_registration=implementation_registration,
        validation_release=validation_release,
        execution=execution,
        allow_materialize=allow_materialize,
    )
    initial_result = _build_initial_execution_result(
        boundary=boundary,
        execution=execution,
        policy_schedule_prefixes=schedules,
        prequential_state_head=state_head,
    )
    outer_zero = run_or_resume_massive_adaptive_rl_outer_fold_execution_v1(
        root=artifact_root,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation_registration,
        policy_schedule=schedules[-1],
        predecessor_state=state_head,
        allow_materialize=allow_materialize,
    )
    outer_zero_result = _build_outer_zero_result(
        boundary=initial_result,
        policy_schedule=schedules[-1],
        execution=outer_zero,
    )
    continuation = run_or_resume_massive_adaptive_rl_prequential_continuation_v1(
        root=artifact_root,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation_registration,
        initial_inputs=initial_inputs,
        initial_execution=execution,
        initial_policy_schedule=schedules[-1],
        outer_zero_execution=outer_zero,
        allow_materialize=allow_materialize,
    )
    delayed_result = _build_delayed_policy_result(
        boundary=outer_zero_result,
        continuation=continuation,
    )
    final_outer = run_or_resume_massive_adaptive_rl_final_outer_execution_v1(
        root=artifact_root,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation_registration,
        outer_zero_execution=outer_zero,
        continuation=continuation,
        allow_materialize=allow_materialize,
    )
    return _build_profitability_report_result(
        boundary=delayed_result,
        execution=final_outer,
    )


def run_massive_adaptive_rl_experiment_v5(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    resume: bool = True,
) -> MassiveAdaptiveRLPrequentialRunV5 | MassiveAdaptiveRLEndToEndRunV2:
    """Register V5, train, and stop at the next causally authorized boundary."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v5(manifest_path)
    if str(device) != manifest.execution_device_specification:
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "requested device differs from the Manifest-V5 training device"
        )
    try:
        registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
            root=artifact_root,
            manifest=manifest,
            allow_materialize=True,
        )
        with massive_adaptive_rl_experiment_orchestration_lock_v1(
            artifact_root=artifact_root,
            experiment_id=manifest.experiment_id,
        ):
            training_capability = (
                issue_massive_adaptive_rl_manifest_v5_training_capability_v1(
                    root=artifact_root,
                    authority=registration,
                )
            )
            with massive_adaptive_rl_manifest_v5_writer_scope_v1(
                root=artifact_root,
                capability=training_capability,
            ):
                training = _run_massive_adaptive_rl_experiment_v2_unlocked(
                    manifest=manifest.base_manifest.base_manifest,
                    source_root=source_root,
                    artifact_root=artifact_root,
                    device=device,
                    resume=resume,
                )
            if training.four_fold_fit_authority_receipt_sha256 is None:
                return training
            return _replay_v5_boundary(
                manifest=manifest,
                registration=registration,
                source_root=source_root,
                artifact_root=artifact_root,
                device=device,
                allow_materialize=True,
            )
    except MassiveAdaptiveRLExperimentLockV1Unavailable as error:
        raise MassiveAdaptiveRLExperimentRunnerV5LeaseUnavailable(
            "adaptive RL V5 execution is already owned"
        ) from error
    except MassiveAdaptiveRLExperimentLockV1Error as error:
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "adaptive RL V5 experiment-global lock is invalid"
        ) from error


def verify_massive_adaptive_rl_experiment_v5(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
) -> MassiveAdaptiveRLPrequentialRunV5:
    """Cold-replay the registered causal prefix without creating artifacts."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v5(manifest_path)
    if str(device) != manifest.execution_device_specification:
        raise MassiveAdaptiveRLExperimentRunnerV5Error(
            "requested device differs from the Manifest-V5 training device"
        )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=artifact_root,
        manifest=manifest,
        allow_materialize=False,
    )
    return _replay_v5_boundary(
        manifest=manifest,
        registration=registration,
        source_root=source_root,
        artifact_root=artifact_root,
        device=device,
        allow_materialize=False,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V5_SCHEMA",
    "MassiveAdaptiveRLExperimentRunnerV5Error",
    "MassiveAdaptiveRLExperimentRunnerV5LeaseUnavailable",
    "MassiveAdaptiveRLPrequentialRunV5",
    "run_massive_adaptive_rl_experiment_v5",
    "verify_massive_adaptive_rl_experiment_v5",
]
