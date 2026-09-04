from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    publish_massive_source_object,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA,
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA,
    build_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
)
import rl_quant.workflows.massive_adaptive_rl_prequential_experiment_state_v1 as state_module
from rl_quant.workflows.massive_adaptive_rl_prequential_experiment_state_v1 import (
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_DATASET,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1,
    MassiveAdaptiveRLPrequentialExperimentStateV1,
    MassiveAdaptiveRLPrequentialExperimentStateV1Error,
    MassiveAdaptiveRLPrequentialExperimentStateV1StaleError,
    MassiveAdaptiveRLPrequentialStageV1,
    load_massive_adaptive_rl_prequential_experiment_states_v1,
    massive_adaptive_rl_prequential_state_relative_path_v1,
    run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
)


def _receipt(value: object) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type: type[object], /, **values: object) -> object:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


_SCHEMA_BY_STAGE = {
    MassiveAdaptiveRLPrequentialStageV1.TRAINED: (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.EXECUTION_IMPLEMENTATION_REGISTERED: (
        MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.INITIAL_VALIDATION_INPUTS_COMMITTED: (
        MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.POLICY_0_FROZEN: (
        MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED: (
        MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA
    ),
}


def _state(
    stage: MassiveAdaptiveRLPrequentialStageV1,
    *,
    disposition: str | None = None,
    gates_passed: bool | None = None,
    predecessor: MassiveAdaptiveRLPrequentialExperimentStateV1 | None = None,
    artifact_time: int | None = None,
) -> MassiveAdaptiveRLPrequentialExperimentStateV1:
    index = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(stage)
    if (
        index
        >= MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(
            MassiveAdaptiveRLPrequentialStageV1.POLICY_0_FROZEN
        )
        and disposition is None
    ):
        disposition = "policy-prefix-qualified"
    if (
        index
        >= MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(
            MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED
        )
        and gates_passed is None
    ):
        gates_passed = False
    when = artifact_time if artifact_time is not None else 100 + index * 10
    schema = _SCHEMA_BY_STAGE.get(stage, state_module._STAGE_ARTIFACT_SCHEMAS[stage])
    has_predecessor = index > 0
    provisional = MassiveAdaptiveRLPrequentialExperimentStateV1(
        experiment_id="state-test",
        manifest_v5_receipt_sha256=_receipt("manifest-v5"),
        manifest_v5_registration_receipt_sha256=_receipt("manifest-registration"),
        execution_implementation_registration_receipt_sha256=_receipt(
            "execution-registration"
        ),
        sequence_index=index,
        stage=stage,
        immediate_predecessor_state_receipt_sha256=(
            None
            if not has_predecessor
            else (
                _receipt((stage.value, "predecessor"))
                if predecessor is None
                else predecessor.semantic_receipt_sha256
            )
        ),
        immediate_predecessor_state_committed_at_ms=(
            None if not has_predecessor else when - 2
        ),
        previous_stage_artifact_committed_at_ms=(
            None
            if not has_predecessor
            else (
                when - 1
                if predecessor is None
                else predecessor.stage_artifact_committed_at_ms
            )
        ),
        stage_artifact_schema=schema,
        stage_artifact_semantic_receipt_sha256=_receipt((stage.value, "artifact")),
        stage_artifact_source_receipt_sha256=_receipt((stage.value, "source")),
        stage_artifact_commit_receipt_sha256=_receipt((stage.value, "commit")),
        stage_artifact_committed_at_ms=when,
        policy_schedule_disposition=disposition,
        policy_schedule_qualified=(
            None if disposition is None else disposition == "policy-prefix-qualified"
        ),
        profitability_gates_passed=gates_passed,
        source_data_qualified=True,
        blocker_code=None,
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _publish_state(
    *, root: Path, state: MassiveAdaptiveRLPrequentialExperimentStateV1, when: int
) -> None:
    relative = state_module._state_relative_path(
        experiment_id=state.experiment_id,
        stage=state.stage,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(state.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=when,
        downloaded_at_ms=when,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=state.semantic_receipt_sha256,
        committed_at_ms=when,
        request_id=f"PREQUENTIAL-STATE-TEST-{state.sequence_index}",
    )


def test_stage_inventory_exactly_matches_manifest_v5() -> None:
    assert tuple(
        stage.value for stage in MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1
    ) == (MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1)


def test_generic_state_is_integrity_valid_but_nonauthorizing() -> None:
    state = _state(MassiveAdaptiveRLPrequentialStageV1.TRAINED)

    assert not state.runtime_state_replayed
    assert not state.prequential_execution_authorized
    assert not state.development_profitability_reporting_authorized
    assert not state.positive_profitability_authorization_eligible


def test_transition_builder_rejects_a_skipped_stage() -> None:
    trained = _state(MassiveAdaptiveRLPrequentialStageV1.TRAINED)
    facts = state_module._StageArtifactFactsV1(
        stage=MassiveAdaptiveRLPrequentialStageV1.INITIAL_VALIDATION_INPUTS_COMMITTED,
        schema=MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
        semantic_receipt=_receipt("release"),
        source_receipt=_receipt("release-source"),
        commit_receipt=_receipt("release-commit"),
        committed_at_ms=200,
        source_data_qualified=True,
        policy_schedule_disposition=None,
        policy_schedule_qualified=None,
        profitability_gates_passed=None,
    )

    with pytest.raises(
        MassiveAdaptiveRLPrequentialExperimentStateV1StaleError,
        match="exact next stage",
    ):
        state_module._state_body(
            manifest=SimpleNamespace(
                experiment_id="state-test",
                semantic_receipt_sha256=_receipt("manifest-v5"),
            ),
            manifest_registration=SimpleNamespace(
                semantic_receipt_sha256=_receipt("manifest-registration")
            ),
            execution_registration=SimpleNamespace(
                semantic_receipt_sha256=_receipt("execution-registration")
            ),
            previous=trained,
            facts=facts,
        )


def test_diagnostic_schedule_cannot_become_qualified() -> None:
    predecessor = _state(
        MassiveAdaptiveRLPrequentialStageV1.POLICY_0_FROZEN,
        disposition="policy-prefix-diagnostic-only",
    )
    facts = state_module._StageArtifactFactsV1(
        stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN,
        schema=MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA,
        semantic_receipt=_receipt("schedule-one"),
        source_receipt=_receipt("schedule-one-source"),
        commit_receipt=_receipt("schedule-one-commit"),
        committed_at_ms=200,
        source_data_qualified=True,
        policy_schedule_disposition="policy-prefix-qualified",
        policy_schedule_qualified=True,
        profitability_gates_passed=None,
    )

    with pytest.raises(
        MassiveAdaptiveRLPrequentialExperimentStateV1Error,
        match="cannot become qualified",
    ):
        state_module._state_body(
            manifest=SimpleNamespace(
                experiment_id="state-test",
                semantic_receipt_sha256=_receipt("manifest-v5"),
            ),
            manifest_registration=SimpleNamespace(
                semantic_receipt_sha256=_receipt("manifest-registration")
            ),
            execution_registration=SimpleNamespace(
                semantic_receipt_sha256=_receipt("execution-registration")
            ),
            previous=predecessor,
            facts=facts,
        )


def test_failed_profitability_gates_are_a_complete_report_state() -> None:
    report = _state(
        MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED,
        disposition="policy-prefix-qualified",
        gates_passed=False,
    )

    assert report.profitability_gates_passed is False
    assert not report.positive_profitability_authorization_eligible
    assert not report.full_cold_replay_verified


def test_loader_rejects_a_gap_in_the_persisted_state_chain(tmp_path: Path) -> None:
    trained = _state(MassiveAdaptiveRLPrequentialStageV1.TRAINED)
    implementation = _state(
        MassiveAdaptiveRLPrequentialStageV1.EXECUTION_IMPLEMENTATION_REGISTERED,
        predecessor=trained,
    )
    release = _state(
        MassiveAdaptiveRLPrequentialStageV1.INITIAL_VALIDATION_INPUTS_COMMITTED,
        predecessor=implementation,
    )
    _publish_state(root=tmp_path, state=trained, when=1_000)
    _publish_state(root=tmp_path, state=release, when=1_020)

    with pytest.raises(
        MassiveAdaptiveRLPrequentialExperimentStateV1Error,
        match="gap or branch",
    ):
        load_massive_adaptive_rl_prequential_experiment_states_v1(
            root=tmp_path,
            manifest=SimpleNamespace(
                experiment_id="state-test",
                semantic_receipt_sha256=_receipt("manifest-v5"),
                validate=lambda: None,
            ),
        )


def test_state_path_is_stage_indexed() -> None:
    manifest = SimpleNamespace(
        experiment_id="state-test",
        validate=lambda: None,
    )
    assert massive_adaptive_rl_prequential_state_relative_path_v1(
        manifest=manifest,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_0_SEALED,
    ).endswith("/005-outer-0-sealed.json")


def test_create_only_state_prefix_replays_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="state-prefix-replay"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    fit_receipt = _receipt("fit")
    execution_receipt = _receipt("implementation")
    execution = _typed_shell(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        experiment_id=manifest.experiment_id,
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        manifest_v5_registration_authority_receipt_sha256=(
            registration.semantic_receipt_sha256
        ),
        four_fold_fit_authority_receipt_sha256=fit_receipt,
        semantic_receipt_sha256=execution_receipt,
        source_data_qualified=True,
        schema=MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA,
    )
    fit = _typed_shell(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        semantic_receipt_sha256=fit_receipt,
        source_data_qualified=True,
        schema=MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA,
    )
    release = _typed_shell(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        release_kind="initial-folds-0-1",
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        execution_implementation_registration_authority_receipt_sha256=(
            execution_receipt
        ),
        semantic_receipt_sha256=_receipt("release"),
        source_data_qualified=True,
        schema=MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
    )
    schedule = _typed_shell(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        fold_indices=(0,),
        policy_schedule_disposition="policy-prefix-diagnostic-only",
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        execution_implementation_registration_receipt_sha256=execution_receipt,
        semantic_receipt_sha256=_receipt("schedule-zero"),
        source_data_qualified=True,
        schema=MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA,
    )
    artifacts = (fit, execution, release, schedule)
    source_receipts = {
        id(artifact): _receipt(("source", index))
        for index, artifact in enumerate(artifacts)
    }
    commit_receipts = {
        id(artifact): _receipt(("commit", index))
        for index, artifact in enumerate(artifacts)
    }
    commit_times = {
        id(artifact): 20 + index * 10 for index, artifact in enumerate(artifacts)
    }

    for authority_type in (
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    ):
        monkeypatch.setattr(authority_type, "validate", lambda _: None)
        monkeypatch.setattr(
            authority_type,
            "source_receipt_sha256",
            property(lambda self: source_receipts[id(self)]),
            raising=False,
        )
        monkeypatch.setattr(
            authority_type,
            "source_transaction_receipt_sha256",
            property(lambda self: commit_receipts[id(self)]),
            raising=False,
        )
        monkeypatch.setattr(
            authority_type,
            "source_transaction_committed_at_ms",
            property(lambda self: commit_times[id(self)]),
            raising=False,
        )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "development_execution_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        state_module,
        "issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        state_module,
        "massive_adaptive_rl_manifest_v5_writer_scope_v1",
        lambda **_: nullcontext(),
    )
    import rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 as writer_guard

    monkeypatch.setattr(
        writer_guard,
        "authorize_and_lock_massive_adaptive_rl_source_publication_v5",
        lambda **_: nullcontext(),
    )

    heads = tuple(
        run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            execution_registration=execution,  # type: ignore[arg-type]
            stage_artifact=artifact,
        )
        for artifact in artifacts
    )
    replayed = run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=execution,  # type: ignore[arg-type]
        stage_artifact=schedule,
        allow_materialize=False,
    )

    assert tuple(row.sequence_index for row in heads) == (0, 1, 2, 3)
    assert replayed.semantic_receipt_sha256 == heads[-1].semantic_receipt_sha256
    assert replayed.prequential_execution_authorized
    assert replayed.policy_schedule_disposition == "policy-prefix-diagnostic-only"
