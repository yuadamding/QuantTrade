from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest

from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_experiment_runner_v5 as runner
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v4 import (
    MassiveAdaptiveRLPrequentialRunV4,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v5 import (
    MassiveAdaptiveRLPrequentialRunV5,
    _build_full_cold_replay_result,
    _build_delayed_policy_result,
    _build_initial_execution_result,
    _build_outer_zero_result,
    _build_preimplementation_result,
    _build_profitability_report_result,
    _build_result,
    _record_initial_prequential_execution_v1,
    run_massive_adaptive_rl_experiment_v5,
    verify_massive_adaptive_rl_experiment_v5,
)
from rl_quant.workflows.massive_adaptive_rl_outer_fold_execution_v1 import (
    MassiveAdaptiveRLOuterFoldExecutionV1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_continuation_v1 import (
    MassiveAdaptiveRLPrequentialContinuationV1,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_final_outer_execution_v1 import (
    MassiveAdaptiveRLFinalOuterExecutionV1,
)
from rl_quant.workflows.massive_adaptive_rl_full_cold_replay_v1 import (
    MassiveAdaptiveRLFullColdReplayAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    build_massive_adaptive_rl_experiment_manifest_v5,
    write_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_initial_validation_execution_v1 import (
    MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1,
    MassiveAdaptiveRLInitialValidationExecutionV1,
    MassiveAdaptiveRLReleasedFoldValidationExecutionV1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_experiment_state_v1 import (
    MassiveAdaptiveRLPrequentialExperimentStateV1,
    MassiveAdaptiveRLPrequentialStageV1,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
)


_T = TypeVar("_T")


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type: type[_T], /, **values: object) -> _T:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def test_v5_result_freezes_implementation_before_initial_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="runner-v5-result"
    )
    registration = _typed_shell(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("registration"),
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    training_state_receipt = _digest("training-state")
    fit_receipt = _digest("fit")
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "development_protocol_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "source_receipt_sha256",
        property(lambda _: _digest("registration-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("registration-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "source_transaction_committed_at_ms",
        property(lambda _: 10),
    )
    monkeypatch.setattr(
        runner,
        "_training_receipts",
        lambda **_: (training_state_receipt, fit_receipt),
    )
    result = _build_preimplementation_result(
        manifest=manifest,
        registration=registration,
        states=(),
    )
    assert isinstance(result, MassiveAdaptiveRLPrequentialRunV5)
    assert result.manifest_v5_registration_committed_at_ms == 10
    assert result.initial_validation_inputs_committed_at_ms is None
    assert result.released_validation_fold_indices == ()
    assert result.withheld_validation_fold_indices == (2, 3)
    assert result.protocol_registered
    assert not result.validation_execution_complete
    assert result.next_required_stage == "execution-implementation-registration"
    assert not result.positive_profitability_authorization_eligible
    assert not result.outer_access_authorized
    assert not result.end_to_end_profitability_execution_complete

    implementation_registration = _typed_shell(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("implementation-registration"),
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        manifest_v5_registration_authority_receipt_sha256=(
            registration.semantic_receipt_sha256
        ),
        training_state_receipt_sha256=training_state_receipt,
        four_fold_fit_authority_receipt_sha256=fit_receipt,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "development_execution_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "source_receipt_sha256",
        property(lambda _: _digest("implementation-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("implementation-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "source_transaction_committed_at_ms",
        property(lambda _: 20),
    )
    registered = _build_preimplementation_result(
        manifest=manifest,
        registration=registration,
        states=(),
        implementation_registration=implementation_registration,
    )
    assert registered.execution_implementation_registered
    assert registered.execution_implementation_registration_committed_at_ms == 20
    assert registered.initial_validation_inputs_committed_at_ms is None
    assert registered.next_required_stage == "initial-validation-input-commitment"

    predecessor = _typed_shell(
        MassiveAdaptiveRLPrequentialRunV4,
        semantic_receipt_sha256=_digest("run-v4"),
        manifest_v4_receipt_sha256=manifest.base_manifest.semantic_receipt_sha256,
        four_fold_fit_authority_receipt_sha256=fit_receipt,
        initial_validation_inputs_authority_receipt_sha256=_digest("initial"),
        initial_validation_inputs_source_receipt_sha256=_digest("initial-source"),
        initial_validation_inputs_commit_receipt_sha256=_digest("initial-commit"),
        training_evidence_adopted=True,
        source_generation_v2_replayed=True,
        initial_validation_inputs_replayed=True,
        diagnostic_continuation_registered=True,
    )
    initial = _typed_shell(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        semantic_receipt_sha256=_digest("initial"),
        manifest_v4_receipt_sha256=manifest.base_manifest.semantic_receipt_sha256,
    )
    monkeypatch.setattr(MassiveAdaptiveRLPrequentialRunV4, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        "source_receipt_sha256",
        property(lambda _: _digest("initial-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("initial-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        "source_transaction_committed_at_ms",
        property(lambda _: 30),
    )
    validation_release = _typed_shell(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        semantic_receipt_sha256=_digest("validation-release"),
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        manifest_v5_registration_authority_receipt_sha256=(
            registration.semantic_receipt_sha256
        ),
        execution_implementation_registration_authority_receipt_sha256=(
            implementation_registration.semantic_receipt_sha256
        ),
        initial_validation_inputs_authority_receipt_sha256=(
            initial.semantic_receipt_sha256
        ),
        training_state_receipt_sha256=training_state_receipt,
        four_fold_fit_authority_receipt_sha256=fit_receipt,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        "development_stage_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        "source_receipt_sha256",
        property(lambda _: _digest("validation-release-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("validation-release-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        "source_transaction_committed_at_ms",
        property(lambda _: 40),
    )
    complete = _build_result(
        manifest=manifest,
        registration=registration,
        predecessor=predecessor,
        training_state_receipt_sha256=training_state_receipt,
        initial_inputs=initial,
        validation_release=validation_release,
        implementation_registration=implementation_registration,
    )
    assert complete.released_validation_fold_indices == (0, 1)
    assert complete.initial_validation_inputs_committed_at_ms == 30
    assert complete.initial_validation_release_committed_at_ms == 40
    assert complete.initial_validation_release_replayed
    assert (
        complete.next_required_stage
        == "prequential-fold-0-and-fold-1-validation-selection-and-freeze"
    )

    receipt_rows = tuple(
        SimpleNamespace(semantic_receipt_sha256=_digest(("stage", index)))
        for index in range(8)
    )
    execution = _typed_shell(
        MassiveAdaptiveRLInitialValidationExecutionV1,
        experiment_id=manifest.experiment_id,
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        validation_release_authority_receipt_sha256=(
            validation_release.semantic_receipt_sha256
        ),
        execution_implementation_registration_receipt_sha256=(
            implementation_registration.semantic_receipt_sha256
        ),
        fold_validation_authorities=receipt_rows[:2],
        policy_selection_authorities=receipt_rows[2:4],
        frozen_ppo_policies=receipt_rows[4:6],
        frozen_fc06_controls=receipt_rows[6:8],
        policy_schedule_disposition=(
            MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1
        ),
        initial_policy_freezing_complete=True,
        outer_zero_preparation_authorized=True,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLInitialValidationExecutionV1, "validate", lambda _: None
    )
    schedules = tuple(
        _typed_shell(
            MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
            fold_indices=tuple(range(index + 1)),
            experiment_id=manifest.experiment_id,
            manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
            semantic_receipt_sha256=_digest(("schedule", index)),
            policy_schedule_disposition="policy-prefix-diagnostic-only",
        )
        for index in range(2)
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "development_stage_authorized",
        property(lambda _: True),
    )
    state_head = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN,
        experiment_id=manifest.experiment_id,
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        execution_implementation_registration_receipt_sha256=(
            implementation_registration.semantic_receipt_sha256
        ),
        stage_artifact_semantic_receipt_sha256=schedules[-1].semantic_receipt_sha256,
        policy_schedule_disposition="policy-prefix-diagnostic-only",
        semantic_receipt_sha256=_digest("state-head"),
        prequential_execution_authorized=True,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_receipt_sha256",
        property(lambda _: _digest("state-head-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("state-head-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_transaction_committed_at_ms",
        property(lambda _: 50),
    )
    advanced = _build_initial_execution_result(
        boundary=complete,
        execution=execution,
        policy_schedule_prefixes=schedules,
        prequential_state_head=state_head,
    )
    assert advanced.initial_policy_freezing_complete
    assert advanced.outer_zero_preparation_authorized
    assert advanced.initial_policy_schedule_disposition == (
        MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1
    )
    assert advanced.next_required_stage == "outer-fold-0-access-and-seal"
    assert advanced.initial_policy_schedule_prefix_receipts == tuple(
        row.semantic_receipt_sha256 for row in schedules
    )
    assert advanced.prequential_state_head_stage == "policy-1-frozen"
    assert not advanced.final_policy_freezing_authorized
    assert not advanced.outer_access_authorized
    assert not advanced.positive_profitability_authorization_eligible

    outer_state = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_0_SEALED,
        semantic_receipt_sha256=_digest("outer-zero-state"),
        immediate_predecessor_state_receipt_sha256=(state_head.semantic_receipt_sha256),
        prequential_execution_authorized=True,
    )
    outer_execution = MassiveAdaptiveRLOuterFoldExecutionV1(
        fold_index=0,
        outer_access=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-access"),
            predecessor_state_receipt_sha256=state_head.semantic_receipt_sha256,
            policy_schedule_receipt_sha256=schedules[-1].semantic_receipt_sha256,
        ),  # type: ignore[arg-type]
        outer_inputs=SimpleNamespace(semantic_receipt_sha256=_digest("outer-inputs")),  # type: ignore[arg-type]
        outer_rollout=SimpleNamespace(semantic_receipt_sha256=_digest("outer-rollout")),  # type: ignore[arg-type]
        outer_fold_seal=SimpleNamespace(semantic_receipt_sha256=_digest("outer-seal")),  # type: ignore[arg-type]
        prequential_state=outer_state,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLOuterFoldExecutionV1, "validate", lambda _: None
    )
    sealed = _build_outer_zero_result(
        boundary=advanced,
        policy_schedule=schedules[-1],
        execution=outer_execution,
    )
    assert sealed.outer_zero_execution_complete
    assert not sealed.outer_zero_preparation_authorized
    assert sealed.sealed_outer_fold_indices == (0,)
    assert sealed.prequential_state_head_stage == "outer-0-sealed"
    assert (
        sealed.next_required_stage == "validation-fold-2-release-selection-and-freeze"
    )
    assert not sealed.positive_profitability_authorization_eligible

    release_two = _typed_shell(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        semantic_receipt_sha256=_digest("release-two"),
        predecessor_state_receipt_sha256=outer_state.semantic_receipt_sha256,
        predecessor_outer_fold_seal_receipt_sha256=(
            outer_execution.outer_fold_seal.semantic_receipt_sha256
        ),
    )
    release_three = _typed_shell(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        semantic_receipt_sha256=_digest("release-three"),
        predecessor_state_receipt_sha256=_digest("outer-one-state"),
        predecessor_outer_fold_seal_receipt_sha256=_digest("outer-one-seal"),
    )
    delayed_rows = tuple(
        _typed_shell(
            MassiveAdaptiveRLReleasedFoldValidationExecutionV1,
            fold_index=index,
            fold_validation_authority=SimpleNamespace(
                semantic_receipt_sha256=_digest(("delayed-validation", index))
            ),
            policy_selection_authority=SimpleNamespace(
                semantic_receipt_sha256=_digest(("delayed-selection", index))
            ),
            frozen_ppo_policy=SimpleNamespace(
                semantic_receipt_sha256=_digest(("delayed-ppo", index))
            ),
            frozen_fc06_control=SimpleNamespace(
                semantic_receipt_sha256=_digest(("delayed-control", index))
            ),
        )
        for index in (2, 3)
    )
    delayed_schedules = tuple(
        _typed_shell(
            MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
            fold_indices=tuple(range(index + 1)),
            semantic_receipt_sha256=_digest(("delayed-schedule", index)),
            policy_schedule_disposition="policy-prefix-diagnostic-only",
        )
        for index in (2, 3)
    )
    outer_one_state = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_1_SEALED,
        semantic_receipt_sha256=release_three.predecessor_state_receipt_sha256,
    )
    outer_one_execution = _typed_shell(
        MassiveAdaptiveRLOuterFoldExecutionV1,
        fold_index=1,
        outer_access=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-one-access")
        ),
        outer_inputs=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-one-inputs")
        ),
        outer_rollout=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-one-rollout")
        ),
        outer_fold_seal=SimpleNamespace(
            semantic_receipt_sha256=(
                release_three.predecessor_outer_fold_seal_receipt_sha256
            )
        ),
        prequential_state=outer_one_state,
    )
    policy_three_state = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN,
        semantic_receipt_sha256=_digest("policy-three-state"),
        immediate_predecessor_state_receipt_sha256=_digest("validation-three-state"),
        prequential_execution_authorized=True,
    )
    continuation = _typed_shell(
        MassiveAdaptiveRLPrequentialContinuationV1,
        validation_releases=(release_two, release_three),
        released_fold_executions=delayed_rows,
        policy_schedules=delayed_schedules,
        outer_one_execution=outer_one_execution,
        prequential_state=policy_three_state,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialContinuationV1, "validate", lambda _: None
    )
    completed_schedule = _build_delayed_policy_result(
        boundary=sealed,
        continuation=continuation,
    )
    assert completed_schedule.validation_execution_complete
    assert completed_schedule.final_policy_freezing_authorized
    assert completed_schedule.outer_one_execution_complete
    assert completed_schedule.sealed_outer_fold_indices == (0, 1)
    assert completed_schedule.released_validation_fold_indices == (0, 1, 2, 3)
    assert completed_schedule.withheld_validation_fold_indices == ()
    assert completed_schedule.prequential_state_head_stage == "policy-3-frozen"
    assert completed_schedule.next_required_stage == "outer-fold-2-access-and-seal"
    assert not completed_schedule.positive_profitability_authorization_eligible

    outer_two_state = SimpleNamespace(
        semantic_receipt_sha256=_digest("outer-two-state")
    )
    outer_two_execution = SimpleNamespace(
        outer_access=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-two-access"),
            predecessor_state_receipt_sha256=(
                policy_three_state.semantic_receipt_sha256
            ),
        ),
        outer_inputs=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-two-inputs")
        ),
        outer_rollout=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-two-rollout")
        ),
        outer_fold_seal=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-two-seal")
        ),
        prequential_state=outer_two_state,
    )
    outer_three_execution = SimpleNamespace(
        outer_access=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-three-access"),
            predecessor_state_receipt_sha256=(outer_two_state.semantic_receipt_sha256),
        ),
        outer_inputs=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-three-inputs")
        ),
        outer_rollout=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-three-rollout")
        ),
        outer_fold_seal=SimpleNamespace(
            semantic_receipt_sha256=_digest("outer-three-seal")
        ),
    )
    report = SimpleNamespace(
        semantic_receipt_sha256=_digest("report"),
        source_receipt_sha256=_digest("report-source"),
        source_transaction_receipt_sha256=_digest("report-commit"),
        source_transaction_committed_at_ms=80,
        profitability_gates_passed=False,
    )
    report_state = SimpleNamespace(
        stage=MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED,
        stage_artifact_semantic_receipt_sha256=report.semantic_receipt_sha256,
        semantic_receipt_sha256=_digest("report-state"),
        source_receipt_sha256=_digest("report-state-source"),
        source_transaction_receipt_sha256=_digest("report-state-commit"),
        source_transaction_committed_at_ms=90,
        development_profitability_reporting_authorized=True,
    )
    final_execution = _typed_shell(
        MassiveAdaptiveRLFinalOuterExecutionV1,
        outer_executions=(outer_two_execution, outer_three_execution),
        outer_fold_seals=(
            outer_execution.outer_fold_seal,
            outer_one_execution.outer_fold_seal,
            outer_two_execution.outer_fold_seal,
            outer_three_execution.outer_fold_seal,
        ),
        profitability_report=report,
        prequential_state=report_state,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFinalOuterExecutionV1,
        "validate",
        lambda _: None,
    )
    reported = _build_profitability_report_result(
        boundary=completed_schedule,
        execution=final_execution,
    )
    assert reported.sealed_outer_fold_indices == (0, 1, 2, 3)
    assert reported.outer_two_execution_complete
    assert reported.outer_three_execution_complete
    assert reported.profitability_reporting_authorized
    assert reported.profitability_gates_passed is False
    assert reported.prequential_state_head_stage == "profitability-report-published"
    assert reported.next_required_stage == "full-cold-replay-verification"
    assert not reported.end_to_end_profitability_execution_complete
    assert not reported.positive_profitability_authorization_eligible

    cold_replay = _typed_shell(
        MassiveAdaptiveRLFullColdReplayAuthorityV1,
        experiment_id=reported.experiment_id,
        manifest_v5_receipt_sha256=reported.manifest_v5_receipt_sha256,
        execution_implementation_registration_receipt_sha256=(
            reported.execution_implementation_registration_authority_receipt_sha256
        ),
        replayed_run_receipt_sha256=reported.semantic_receipt_sha256,
        profitability_report_authority_receipt_sha256=(
            reported.profitability_report_authority_receipt_sha256
        ),
        profitability_report_state_receipt_sha256=(
            reported.prequential_state_head_receipt_sha256
        ),
        outer_fold_seal_receipts=reported.outer_fold_seal_authority_receipts,
        policy_schedule_disposition=reported.policy_schedule_disposition,
        profitability_gates_passed=reported.profitability_gates_passed,
        semantic_receipt_sha256=_digest("cold-replay"),
        development_full_cold_replay_verified=True,
        _loaded_source=SimpleNamespace(
            receipt=SimpleNamespace(receipt_sha256=_digest("cold-replay-source")),
            commit=SimpleNamespace(
                receipt_sha256=_digest("cold-replay-commit"),
                committed_at_ms=100,
            ),
        ),
    )
    final_state = SimpleNamespace(
        stage=MassiveAdaptiveRLPrequentialStageV1.FULL_COLD_REPLAY_VERIFIED,
        full_cold_replay_verified=True,
        stage_artifact_semantic_receipt_sha256=(cold_replay.semantic_receipt_sha256),
        immediate_predecessor_state_receipt_sha256=(
            reported.prequential_state_head_receipt_sha256
        ),
        policy_schedule_disposition=reported.policy_schedule_disposition,
        profitability_gates_passed=reported.profitability_gates_passed,
        semantic_receipt_sha256=_digest("final-state"),
        source_receipt_sha256=_digest("final-state-source"),
        source_transaction_receipt_sha256=_digest("final-state-commit"),
        source_transaction_committed_at_ms=110,
        validate=lambda: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFullColdReplayAuthorityV1,
        "validate",
        lambda _: None,
    )
    completed = _build_full_cold_replay_result(
        boundary=reported,
        cold_replay=cold_replay,
        prequential_state_head=final_state,
    )
    assert completed.full_cold_replay_verified
    assert completed.end_to_end_profitability_execution_complete
    assert completed.prequential_state_head_stage == "full-cold-replay-verified"
    assert completed.next_required_stage == (
        "development-profitability-execution-complete"
    )
    assert not completed.positive_profitability_authorization_eligible


def test_v5_root_registers_before_resuming_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="runner-v5-order"
    )
    manifest_path = tmp_path / "manifest-v5.json"
    write_massive_adaptive_rl_experiment_manifest_v5(
        path=manifest_path, manifest=manifest
    )
    registration = object()
    training = SimpleNamespace(four_fold_fit_authority_receipt_sha256=_digest("fit"))
    expected = object()
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "massive_adaptive_rl_experiment_orchestration_lock_v1",
        lambda **_: nullcontext(),
    )

    def register(**kwargs):
        assert kwargs["manifest"] == manifest
        calls.append("registration")
        return registration

    def train(**kwargs):
        assert kwargs["manifest"] == manifest.base_manifest.base_manifest
        calls.append("training")
        return training

    def boundary(**kwargs):
        assert kwargs["registration"] is registration
        calls.append("initial-boundary")
        return expected

    monkeypatch.setattr(
        runner,
        "run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1",
        register,
    )
    monkeypatch.setattr(
        runner,
        "issue_massive_adaptive_rl_manifest_v5_training_capability_v1",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        runner,
        "massive_adaptive_rl_manifest_v5_writer_scope_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        runner, "_run_massive_adaptive_rl_experiment_v2_unlocked", train
    )
    monkeypatch.setattr(runner, "_replay_v5_boundary", boundary)
    assert (
        run_massive_adaptive_rl_experiment_v5(
            manifest_path=manifest_path,
            source_root=tmp_path / "source",
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        is expected
    )
    assert calls == ["registration", "training", "initial-boundary"]


def test_initial_execution_records_schedule_and_state_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="runner-v5-initial-state-prefix"
    )
    registration = object()
    implementation = object()
    fit = object()
    release = SimpleNamespace(four_fold_fit_authority=fit)
    ppo = (object(), object())
    controls = (object(), object())
    execution = SimpleNamespace(
        frozen_ppo_policies=ppo,
        frozen_fc06_controls=controls,
    )
    schedules: list[SimpleNamespace] = []
    state_artifacts: list[object] = []

    def schedule(**kwargs):
        assert kwargs["root"] == tmp_path
        count = len(kwargs["frozen_ppo_policies"])
        result = SimpleNamespace(
            fold_indices=tuple(range(count)),
            semantic_receipt_sha256=_digest(("schedule", count)),
        )
        schedules.append(result)
        return result

    def state(**kwargs):
        state_artifacts.append(kwargs["stage_artifact"])
        return SimpleNamespace(
            stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN,
        )

    monkeypatch.setattr(
        runner,
        "run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1",
        schedule,
    )
    monkeypatch.setattr(
        runner,
        "run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1",
        state,
    )
    result_schedules, head = _record_initial_prequential_execution_v1(
        root=tmp_path,
        manifest=manifest,
        registration=registration,  # type: ignore[arg-type]
        implementation_registration=implementation,  # type: ignore[arg-type]
        validation_release=release,  # type: ignore[arg-type]
        execution=execution,  # type: ignore[arg-type]
        allow_materialize=True,
    )

    assert result_schedules == tuple(schedules)
    assert tuple(schedule.fold_indices for schedule in schedules) == ((0,), (0, 1))
    assert state_artifacts == [fit, implementation, release, *schedules]
    assert head.stage is MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN


def test_v5_training_blocker_does_not_open_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="runner-v5-training-blocked"
    )
    manifest_path = tmp_path / "manifest-v5.json"
    write_massive_adaptive_rl_experiment_manifest_v5(
        path=manifest_path, manifest=manifest
    )
    training = SimpleNamespace(four_fold_fit_authority_receipt_sha256=None)
    monkeypatch.setattr(
        runner,
        "massive_adaptive_rl_experiment_orchestration_lock_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        runner,
        "run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        runner,
        "issue_massive_adaptive_rl_manifest_v5_training_capability_v1",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        runner,
        "massive_adaptive_rl_manifest_v5_writer_scope_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        runner,
        "_run_massive_adaptive_rl_experiment_v2_unlocked",
        lambda **_: training,
    )
    monkeypatch.setattr(
        runner,
        "_replay_v5_boundary",
        lambda **_: pytest.fail("validation inputs must remain unopened"),
    )
    assert (
        run_massive_adaptive_rl_experiment_v5(
            manifest_path=manifest_path,
            source_root=tmp_path / "source",
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        is training
    )


def test_v5_verify_is_strictly_nonmaterializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="runner-v5-verify"
    )
    manifest_path = tmp_path / "manifest-v5.json"
    write_massive_adaptive_rl_experiment_manifest_v5(
        path=manifest_path, manifest=manifest
    )
    registration = object()
    expected = object()
    artifact_root = tmp_path / "artifacts"
    lock_path = artifact_root / runner.massive_adaptive_rl_experiment_lock_relative_path_v1(
        experiment_id=manifest.experiment_id
    )
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")

    def register(**kwargs):
        assert kwargs["allow_materialize"] is False
        return registration

    def boundary(**kwargs):
        assert kwargs["registration"] is registration
        assert kwargs["allow_materialize"] is False
        return expected

    monkeypatch.setattr(
        runner,
        "run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1",
        register,
    )
    monkeypatch.setattr(
        runner,
        "massive_adaptive_rl_protected_evidence_inventory_v1",
        lambda **_: (),
    )
    monkeypatch.setattr(
        runner,
        "massive_adaptive_rl_experiment_orchestration_lock_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(runner, "_replay_v5_boundary", boundary)
    assert (
        verify_massive_adaptive_rl_experiment_v5(
            manifest_path=manifest_path,
            source_root=tmp_path / "source",
            artifact_root=artifact_root,
            device="cpu",
        )
        is expected
    )


def test_v5_verify_requires_persisted_cold_replay_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="runner-v5-complete-verify"
    )
    manifest_path = tmp_path / "manifest-v5.json"
    write_massive_adaptive_rl_experiment_manifest_v5(
        path=manifest_path, manifest=manifest
    )
    artifact_root = tmp_path / "artifacts"
    source_root = tmp_path / "source"
    source_root.mkdir()
    lock_path = artifact_root / runner.massive_adaptive_rl_experiment_lock_relative_path_v1(
        experiment_id=manifest.experiment_id
    )
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    registration = object()
    report_boundary = _typed_shell(
        MassiveAdaptiveRLPrequentialRunV5,
        profitability_reporting_authorized=True,
    )
    completed = object()
    inventory = (object(),)

    monkeypatch.setattr(
        runner,
        "run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1",
        lambda **_: registration,
    )
    monkeypatch.setattr(
        runner,
        "massive_adaptive_rl_experiment_orchestration_lock_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        runner,
        "massive_adaptive_rl_protected_evidence_inventory_v1",
        lambda **_: inventory,
    )
    monkeypatch.setattr(
        runner,
        "_replay_v5_boundary",
        lambda **kwargs: (
            report_boundary
            if kwargs["allow_materialize"] is False
            else pytest.fail("verify cannot materialize the report boundary")
        ),
    )

    def finalize(**kwargs):
        assert kwargs["report_boundary"] is report_boundary
        assert kwargs["replayed_boundary"] is report_boundary
        assert kwargs["evidence_inventory_before"] is inventory
        assert kwargs["evidence_inventory_after"] is inventory
        assert kwargs["allow_materialize"] is False
        return completed

    monkeypatch.setattr(runner, "_finalize_full_cold_replay_v1", finalize)
    assert (
        verify_massive_adaptive_rl_experiment_v5(
            manifest_path=manifest_path,
            source_root=source_root,
            artifact_root=artifact_root,
            device="cpu",
        )
        is completed
    )


def test_v5_root_api_has_no_validation_or_outer_choice_surface() -> None:
    assert tuple(
        inspect.signature(run_massive_adaptive_rl_experiment_v5).parameters
    ) == (
        "manifest_path",
        "source_root",
        "artifact_root",
        "device",
        "resume",
    )
    assert not {
        "fold_index",
        "validation_inputs",
        "sealed_outer_fold_indices",
        "environment",
        "actions",
        "targets",
        "metrics",
        "selected_checkpoint",
        "outer_data",
    }.intersection(inspect.signature(run_massive_adaptive_rl_experiment_v5).parameters)
