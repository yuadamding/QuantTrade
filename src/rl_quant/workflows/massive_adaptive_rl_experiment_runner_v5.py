"""Manifest-V5 root through the registered initial prequential boundary.

V5 is the unique authoring generation for the prequential experiment.  It
registers that ownership before resuming training, freezes the qualified
evaluation implementation after causal training, and only then permits the
fold-0/1 validation inputs.  Later validation, policy freezing, outer access,
and profitability reporting remain closed until their manifest-bound authority
generations exist.
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
from rl_quant.workflows.massive_adaptive_rl_initial_validation_execution_v1 import (
    MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1,
    MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_QUALIFIED_V1,
    MassiveAdaptiveRLInitialValidationExecutionV1,
    run_or_resume_massive_adaptive_rl_initial_validation_execution_v1,
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
    initial_policy_schedule_disposition: str | None
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
        expected_next_stage = (
            "outer-fold-0-access-and-seal"
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
            or self.initial_policy_schedule_disposition
            not in (
                None,
                MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_QUALIFIED_V1,
                MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1,
            )
            or (self.initial_policy_schedule_disposition is not None)
            != initial_execution_present
            or self.released_validation_fold_indices
            != (
                MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
                if release_present
                else ()
            )
            or self.withheld_validation_fold_indices
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1
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
            or self.validation_execution_complete
            or self.initial_policy_freezing_complete != initial_execution_present
            or self.outer_zero_preparation_authorized != initial_execution_present
            or self.next_required_stage != expected_next_stage
            or self.policy_schedule_disposition is not None
            or self.final_policy_freezing_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
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
        "initial_policy_schedule_disposition": None,
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
        "initial_policy_schedule_disposition": None,
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
) -> MassiveAdaptiveRLPrequentialRunV5:
    """Promote the status envelope only after both initial policy pairs replay."""

    boundary.validate()
    execution.validate()
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
            "initial_policy_schedule_disposition": (
                execution.policy_schedule_disposition
            ),
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
    return _build_initial_execution_result(boundary=boundary, execution=execution)


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
    """Cold-replay the registered initial boundary without creating artifacts."""

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
