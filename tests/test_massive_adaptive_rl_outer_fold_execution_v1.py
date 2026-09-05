from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest

from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v2 import (
    MassiveAdaptiveOuterAccessCommitmentV2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
    MassiveAdaptiveRLOuterFoldSealAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_inputs_v1 import (
    MassiveAdaptiveRLOuterInputAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_authority_v2 import (
    MassiveAdaptiveRLOuterRolloutAuthorityV2,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_outer_fold_execution_v1 as outer
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    build_massive_adaptive_rl_experiment_manifest_v5,
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


def _roots(monkeypatch: pytest.MonkeyPatch):
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="outer-fold-orchestrator"
    )
    registration = _typed_shell(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    execution = _typed_shell(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        semantic_receipt_sha256=_digest("implementation"),
    )
    schedule = _typed_shell(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        experiment_id=manifest.experiment_id,
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        fold_indices=(0, 1),
        semantic_receipt_sha256=_digest("schedule"),
    )
    predecessor = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        experiment_id=manifest.experiment_id,
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        execution_implementation_registration_receipt_sha256=(
            execution.semantic_receipt_sha256
        ),
        stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN,
        stage_artifact_semantic_receipt_sha256=schedule.semantic_receipt_sha256,
        stage_artifact_source_receipt_sha256=_digest("schedule-source"),
        stage_artifact_commit_receipt_sha256=_digest("schedule-commit"),
        stage_artifact_committed_at_ms=90,
        semantic_receipt_sha256=_digest("predecessor-state"),
    )
    for authority_type in (
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        MassiveAdaptiveRLPrequentialExperimentStateV1,
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
        "development_stage_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "source_receipt_sha256",
        property(lambda _: _digest("schedule-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("schedule-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "source_transaction_committed_at_ms",
        property(lambda _: 90),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "prequential_execution_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_receipt_sha256",
        property(lambda _: _digest("predecessor-state-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("predecessor-state-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "source_transaction_committed_at_ms",
        property(lambda state: 140 if state.stage.value.startswith("outer-") else 95),
    )
    return manifest, registration, execution, schedule, predecessor


def test_outer_fold_public_surface_has_no_caller_economic_inputs() -> None:
    assert set(
        inspect.signature(
            run_or_resume_massive_adaptive_rl_outer_fold_execution_v1
        ).parameters
    ) == {
        "root",
        "manifest",
        "manifest_registration",
        "execution_registration",
        "policy_schedule",
        "predecessor_state",
        "allow_materialize",
    }


@pytest.mark.parametrize(
    ("fold_index", "predecessor_stage", "sealed_stage", "schedule_folds"),
    (
        (
            0,
            MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN,
            MassiveAdaptiveRLPrequentialStageV1.OUTER_0_SEALED,
            (0, 1),
        ),
        (
            1,
            MassiveAdaptiveRLPrequentialStageV1.POLICY_2_FROZEN,
            MassiveAdaptiveRLPrequentialStageV1.OUTER_1_SEALED,
            (0, 1, 2),
        ),
        (
            2,
            MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN,
            MassiveAdaptiveRLPrequentialStageV1.OUTER_2_SEALED,
            (0, 1, 2, 3),
        ),
        (
            3,
            MassiveAdaptiveRLPrequentialStageV1.OUTER_2_SEALED,
            MassiveAdaptiveRLPrequentialStageV1.OUTER_3_SEALED,
            (0, 1, 2, 3),
        ),
    ),
)
def test_all_outer_folds_execute_and_seal_in_package_owned_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fold_index: int,
    predecessor_stage: MassiveAdaptiveRLPrequentialStageV1,
    sealed_stage: MassiveAdaptiveRLPrequentialStageV1,
    schedule_folds: tuple[int, ...],
) -> None:
    manifest, registration, implementation, schedule, predecessor = _roots(monkeypatch)
    object.__setattr__(predecessor, "stage", predecessor_stage)
    object.__setattr__(schedule, "fold_indices", schedule_folds)
    frozen_policy = object()
    frozen_control = object()
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "frozen_policy",
        lambda _, requested_fold: (
            frozen_policy if requested_fold == fold_index else None
        ),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "frozen_control",
        lambda _, requested_fold: (
            frozen_control if requested_fold == fold_index else None
        ),
    )
    monkeypatch.setattr(
        outer,
        "massive_adaptive_rl_experiment_materialization_lock_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        outer,
        "load_massive_adaptive_rl_prequential_experiment_states_v1",
        lambda **_: (predecessor,),
    )
    calls: list[str] = []
    access = _typed_shell(
        MassiveAdaptiveOuterAccessCommitmentV2,
        fold_index=fold_index,
        semantic_receipt_sha256=_digest("access"),
        predecessor_state_receipt_sha256=predecessor.semantic_receipt_sha256,
        policy_schedule_receipt_sha256=schedule.semantic_receipt_sha256,
    )
    inputs = _typed_shell(
        MassiveAdaptiveRLOuterInputAuthorityV1,
        fold_index=fold_index,
        semantic_receipt_sha256=_digest("inputs"),
        outer_access_commitment_receipt_sha256=access.semantic_receipt_sha256,
    )
    rollout = _typed_shell(
        MassiveAdaptiveRLOuterRolloutAuthorityV2,
        fold_index=fold_index,
        semantic_receipt_sha256=_digest("rollout"),
        outer_access_commitment_receipt_sha256=access.semantic_receipt_sha256,
    )
    seal = _typed_shell(
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
        fold_index=fold_index,
        semantic_receipt_sha256=_digest("seal"),
        outer_rollout_authority_receipt_sha256=rollout.semantic_receipt_sha256,
    )
    state = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=sealed_stage,
        semantic_receipt_sha256=_digest("outer-state"),
        stage_artifact_semantic_receipt_sha256=seal.semantic_receipt_sha256,
        immediate_predecessor_state_receipt_sha256=(
            predecessor.semantic_receipt_sha256
        ),
    )
    for authority_type, when in (
        (MassiveAdaptiveOuterAccessCommitmentV2, 100),
        (MassiveAdaptiveRLOuterInputAuthorityV1, 110),
        (MassiveAdaptiveRLOuterRolloutAuthorityV2, 120),
        (MassiveAdaptiveRLOuterFoldSealAuthorityV1, 130),
    ):
        monkeypatch.setattr(authority_type, "validate", lambda _: None)
        monkeypatch.setattr(
            authority_type,
            "source_transaction_committed_at_ms",
            property(lambda _, value=when: value),
            raising=False,
        )

    def commit_access(**kwargs):
        calls.append("access")
        assert kwargs["frozen_policy"] is frozen_policy
        assert kwargs["frozen_control"] is frozen_control
        assert kwargs["predecessor_state"] is predecessor
        return access

    def build_inputs(**kwargs):
        calls.append("inputs")
        assert kwargs["outer_access"] is access
        return SimpleNamespace(outer_access=access, outer_inputs=inputs)

    def run_rollout(**kwargs):
        calls.append("rollout")
        assert kwargs["outer_access"] is access
        return rollout

    def seal_fold(**kwargs):
        calls.append("seal")
        assert kwargs["outer_rollout"] is rollout
        return seal

    def append_state(**kwargs):
        calls.append("state")
        assert kwargs["stage_artifact"] is seal
        return state

    monkeypatch.setattr(
        outer,
        "run_or_resume_massive_adaptive_outer_access_commitment_v2",
        commit_access,
    )
    monkeypatch.setattr(
        outer, "run_or_resume_massive_adaptive_rl_outer_inputs_v1", build_inputs
    )
    monkeypatch.setattr(
        outer,
        "run_or_resume_massive_adaptive_rl_outer_rollout_authority_v2",
        run_rollout,
    )
    monkeypatch.setattr(
        outer,
        "run_or_resume_massive_adaptive_rl_outer_fold_seal_authority_v1",
        seal_fold,
    )
    monkeypatch.setattr(
        outer,
        "run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1",
        append_state,
    )

    result = run_or_resume_massive_adaptive_rl_outer_fold_execution_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation,
        policy_schedule=schedule,
        predecessor_state=predecessor,
    )

    assert isinstance(result, MassiveAdaptiveRLOuterFoldExecutionV1)
    assert result.fold_index == fold_index
    assert calls == ["access", "inputs", "rollout", "seal", "state"]


def test_policy_three_state_selects_outer_two_before_outer_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, registration, implementation, schedule, predecessor = _roots(monkeypatch)
    object.__setattr__(
        predecessor,
        "stage",
        MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN,
    )
    object.__setattr__(schedule, "fold_indices", (0, 1, 2, 3))
    monkeypatch.setattr(
        outer,
        "massive_adaptive_rl_experiment_materialization_lock_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        outer,
        "load_massive_adaptive_rl_prequential_experiment_states_v1",
        lambda **_: (predecessor,),
    )
    observed_fold: int | None = None

    def access(**kwargs):
        nonlocal observed_fold
        observed_fold = kwargs["frozen_policy"].fold_index
        raise RuntimeError("stop after fold selection")

    frozen_policy = SimpleNamespace(fold_index=2)
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "frozen_policy",
        lambda _, requested_fold: (
            frozen_policy if requested_fold == 2 else SimpleNamespace(fold_index=-1)
        ),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        "frozen_control",
        lambda _, requested_fold: SimpleNamespace(fold_index=requested_fold),
    )

    monkeypatch.setattr(
        outer, "run_or_resume_massive_adaptive_outer_access_commitment_v2", access
    )

    with pytest.raises(RuntimeError, match="stop after fold selection"):
        run_or_resume_massive_adaptive_rl_outer_fold_execution_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            execution_registration=implementation,
            policy_schedule=schedule,
            predecessor_state=predecessor,
        )
    assert observed_fold == 2


def test_historical_outer_predecessor_forces_read_only_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _, _, predecessor = _roots(monkeypatch)
    completed = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        experiment_id=manifest.experiment_id,
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_0_SEALED,
        semantic_receipt_sha256=_digest("completed-outer-state"),
    )
    monkeypatch.setattr(
        outer,
        "load_massive_adaptive_rl_prequential_experiment_states_v1",
        lambda **_: (predecessor, completed),
    )

    assert outer._require_exact_current_state_head_v1(
        root=tmp_path,
        manifest=manifest,
        predecessor_state=predecessor,
        allow_materialize=True,
    ) == (0, False)
