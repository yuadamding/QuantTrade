from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest

from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
    MassiveAdaptiveRLOuterFoldSealAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v2 import (
    MassiveAdaptiveRLProfitabilityReportAuthorityV2,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_final_outer_execution_v1 as final
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_final_outer_execution_v1 import (
    MassiveAdaptiveRLFinalOuterExecutionV1,
    run_or_resume_massive_adaptive_rl_final_outer_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    build_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_outer_fold_execution_v1 import (
    MassiveAdaptiveRLOuterFoldExecutionV1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_continuation_v1 import (
    MassiveAdaptiveRLPrequentialContinuationV1,
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


def _state(
    *,
    manifest_receipt: str,
    experiment_id: str,
    implementation_receipt: str,
    stage: MassiveAdaptiveRLPrequentialStageV1,
) -> MassiveAdaptiveRLPrequentialExperimentStateV1:
    return _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        experiment_id=experiment_id,
        manifest_v5_receipt_sha256=manifest_receipt,
        execution_implementation_registration_receipt_sha256=(implementation_receipt),
        stage=stage,
        semantic_receipt_sha256=_digest(("state", stage.value)),
        prequential_execution_authorized=True,
    )


def test_final_outer_public_surface_has_no_caller_economic_inputs() -> None:
    assert set(
        inspect.signature(
            run_or_resume_massive_adaptive_rl_final_outer_execution_v1
        ).parameters
    ) == {
        "root",
        "manifest",
        "manifest_registration",
        "execution_registration",
        "outer_zero_execution",
        "continuation",
        "allow_materialize",
    }


def test_final_outer_executes_o2_then_o3_then_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="final-outer-execution"
    )
    registration = _typed_shell(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    )
    implementation = _typed_shell(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("implementation"),
    )
    seal_zero = _typed_shell(
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
        fold_index=0,
        semantic_receipt_sha256=_digest("seal-zero"),
    )
    seal_one = _typed_shell(
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
        fold_index=1,
        semantic_receipt_sha256=_digest("seal-one"),
    )
    schedule = _typed_shell(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        semantic_receipt_sha256=_digest("complete-schedule"),
        predecessor_outer_fold_seal_receipts=(
            seal_zero.semantic_receipt_sha256,
            seal_one.semantic_receipt_sha256,
        ),
    )
    policy_three = _state(
        manifest_receipt=manifest.semantic_receipt_sha256,
        experiment_id=manifest.experiment_id,
        implementation_receipt=implementation.semantic_receipt_sha256,
        stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN,
    )
    object.__setattr__(
        policy_three,
        "stage_artifact_semantic_receipt_sha256",
        schedule.semantic_receipt_sha256,
    )
    outer_zero = _typed_shell(
        MassiveAdaptiveRLOuterFoldExecutionV1,
        fold_index=0,
        outer_fold_seal=seal_zero,
    )
    outer_one = _typed_shell(
        MassiveAdaptiveRLOuterFoldExecutionV1,
        fold_index=1,
        outer_fold_seal=seal_one,
    )
    continuation = _typed_shell(
        MassiveAdaptiveRLPrequentialContinuationV1,
        policy_schedules=(schedule,),
        outer_one_execution=outer_one,
        prequential_state=policy_three,
    )
    outer_two_state = _state(
        manifest_receipt=manifest.semantic_receipt_sha256,
        experiment_id=manifest.experiment_id,
        implementation_receipt=implementation.semantic_receipt_sha256,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_2_SEALED,
    )
    outer_three_state = _state(
        manifest_receipt=manifest.semantic_receipt_sha256,
        experiment_id=manifest.experiment_id,
        implementation_receipt=implementation.semantic_receipt_sha256,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_3_SEALED,
    )
    seal_two = _typed_shell(
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
        fold_index=2,
        semantic_receipt_sha256=_digest("seal-two"),
    )
    seal_three = _typed_shell(
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
        fold_index=3,
        semantic_receipt_sha256=_digest("seal-three"),
    )
    execution_two = _typed_shell(
        MassiveAdaptiveRLOuterFoldExecutionV1,
        fold_index=2,
        outer_fold_seal=seal_two,
        prequential_state=outer_two_state,
        outer_access=SimpleNamespace(
            predecessor_state_receipt_sha256=policy_three.semantic_receipt_sha256
        ),
    )
    execution_three = _typed_shell(
        MassiveAdaptiveRLOuterFoldExecutionV1,
        fold_index=3,
        outer_fold_seal=seal_three,
        prequential_state=outer_three_state,
        outer_access=SimpleNamespace(
            predecessor_state_receipt_sha256=(outer_two_state.semantic_receipt_sha256)
        ),
    )
    report = _typed_shell(
        MassiveAdaptiveRLProfitabilityReportAuthorityV2,
        semantic_receipt_sha256=_digest("report"),
        outer_fold_seal_receipts=tuple(
            row.semantic_receipt_sha256
            for row in (seal_zero, seal_one, seal_two, seal_three)
        ),
        development_profitability_reporting_authorized=True,
        end_to_end_profitability_execution_complete=False,
    )
    report_state = _state(
        manifest_receipt=manifest.semantic_receipt_sha256,
        experiment_id=manifest.experiment_id,
        implementation_receipt=implementation.semantic_receipt_sha256,
        stage=MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED,
    )
    object.__setattr__(
        report_state,
        "stage_artifact_semantic_receipt_sha256",
        report.semantic_receipt_sha256,
    )
    object.__setattr__(
        report_state,
        "development_profitability_reporting_authorized",
        True,
    )
    object.__setattr__(report_state, "full_cold_replay_verified", False)
    object.__setattr__(
        report_state,
        "positive_profitability_authorization_eligible",
        False,
    )
    for authority_type in (
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        MassiveAdaptiveRLOuterFoldExecutionV1,
        MassiveAdaptiveRLPrequentialContinuationV1,
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
        MassiveAdaptiveRLProfitabilityReportAuthorityV2,
    ):
        monkeypatch.setattr(authority_type, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "development_protocol_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "development_execution_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "complete_schedule",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "development_stage_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_receipt_sha256",
        property(
            lambda state: _digest(("state-source", state.semantic_receipt_sha256))
        ),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_transaction_receipt_sha256",
        property(
            lambda state: _digest(("state-commit", state.semantic_receipt_sha256))
        ),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_transaction_committed_at_ms",
        property(
            lambda state: list(MassiveAdaptiveRLPrequentialStageV1).index(state.stage)
        ),
    )
    monkeypatch.setattr(
        final,
        "massive_adaptive_rl_experiment_materialization_lock_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        final,
        "load_massive_adaptive_rl_prequential_experiment_states_v1",
        lambda **_: (outer_three_state,),
    )
    calls: list[str] = []

    def execute_outer(**kwargs):
        if kwargs["predecessor_state"] is policy_three:
            calls.append("outer-two")
            return execution_two
        assert kwargs["predecessor_state"] is outer_two_state
        calls.append("outer-three")
        return execution_three

    def publish_report(**kwargs):
        calls.append("report")
        assert tuple(kwargs["outer_fold_seals"]) == (
            seal_zero,
            seal_one,
            seal_two,
            seal_three,
        )
        return report

    def append_state(**kwargs):
        calls.append("state")
        assert kwargs["stage_artifact"] is report
        return report_state

    monkeypatch.setattr(
        final,
        "run_or_resume_massive_adaptive_rl_outer_fold_execution_v1",
        execute_outer,
    )
    monkeypatch.setattr(
        final,
        "run_or_resume_massive_adaptive_rl_profitability_report_authority_v2",
        publish_report,
    )
    monkeypatch.setattr(
        final,
        "run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1",
        append_state,
    )

    result = run_or_resume_massive_adaptive_rl_final_outer_execution_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation,
        outer_zero_execution=outer_zero,
        continuation=continuation,
    )

    assert isinstance(result, MassiveAdaptiveRLFinalOuterExecutionV1)
    assert calls == ["outer-two", "outer-three", "report", "state"]
    assert result.outer_fold_seals == (seal_zero, seal_one, seal_two, seal_three)
    assert result.prequential_state.stage is (
        MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED
    )


def test_report_replay_is_read_only_after_exact_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="historical-report-replay"
    )
    implementation_receipt = _digest("implementation")
    outer_three = _state(
        manifest_receipt=manifest.semantic_receipt_sha256,
        experiment_id=manifest.experiment_id,
        implementation_receipt=implementation_receipt,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_3_SEALED,
    )
    report = _state(
        manifest_receipt=manifest.semantic_receipt_sha256,
        experiment_id=manifest.experiment_id,
        implementation_receipt=implementation_receipt,
        stage=MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_receipt_sha256",
        property(lambda state: _digest(("source", state.semantic_receipt_sha256))),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_transaction_receipt_sha256",
        property(lambda state: _digest(("commit", state.semantic_receipt_sha256))),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_transaction_committed_at_ms",
        property(
            lambda state: list(MassiveAdaptiveRLPrequentialStageV1).index(state.stage)
        ),
    )
    monkeypatch.setattr(
        final,
        "load_massive_adaptive_rl_prequential_experiment_states_v1",
        lambda **_: (outer_three, report),
    )

    assert not final._report_materialization_mode_after_state_v1(
        root=tmp_path,
        manifest=manifest,
        predecessor_state=outer_three,
        allow_materialize=True,
    )
