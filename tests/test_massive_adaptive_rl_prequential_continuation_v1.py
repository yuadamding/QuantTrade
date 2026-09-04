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
from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_frozen_rl_policy_v2 import (
    MassiveAdaptiveFrozenRLPolicyV2,
)
from rl_quant.training.massive_adaptive_rl_frozen_fc06_v2 import (
    MassiveAdaptiveRLFrozenFC06V2,
)
from rl_quant.workflows import massive_adaptive_rl_prequential_continuation_v1 as flow
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_initial_validation_execution_v1 import (
    MassiveAdaptiveRLInitialValidationExecutionV1,
    MassiveAdaptiveRLReleasedFoldValidationExecutionV1,
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
    run_or_resume_massive_adaptive_rl_prequential_continuation_v1,
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


def test_continuation_public_surface_has_no_caller_economic_inputs() -> None:
    assert set(
        inspect.signature(
            run_or_resume_massive_adaptive_rl_prequential_continuation_v1
        ).parameters
    ) == {
        "root",
        "manifest",
        "manifest_registration",
        "execution_registration",
        "initial_inputs",
        "initial_execution",
        "initial_policy_schedule",
        "outer_zero_execution",
        "allow_materialize",
    }


def test_continuation_releases_two_then_executes_one_then_releases_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="prequential-continuation"
    )
    registration = _typed_shell(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("registration"),
    )
    implementation = _typed_shell(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("implementation"),
    )
    initial_inputs = _typed_shell(MassiveAdaptiveRLInitialValidationInputsAuthorityV1)
    ppo = tuple(
        _typed_shell(
            MassiveAdaptiveFrozenRLPolicyV2,
            semantic_receipt_sha256=_digest(f"ppo-{index}"),
        )
        for index in range(4)
    )
    controls = tuple(
        _typed_shell(
            MassiveAdaptiveRLFrozenFC06V2,
            semantic_receipt_sha256=_digest(f"control-{index}"),
        )
        for index in range(4)
    )
    initial_execution = _typed_shell(
        MassiveAdaptiveRLInitialValidationExecutionV1,
        frozen_ppo_policies=ppo[:2],
        frozen_fc06_controls=controls[:2],
        initial_policy_freezing_complete=True,
    )
    schedule_one = _typed_shell(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        fold_indices=(0, 1),
        semantic_receipt_sha256=_digest("schedule-1"),
    )
    seal_zero = _typed_shell(
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
        fold_index=0,
        semantic_receipt_sha256=_digest("seal-0"),
    )
    state_zero = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_0_SEALED,
        semantic_receipt_sha256=_digest("state-outer-0"),
    )
    outer_zero = _typed_shell(
        MassiveAdaptiveRLOuterFoldExecutionV1,
        fold_index=0,
        outer_access=_typed_shell(
            MassiveAdaptiveOuterAccessCommitmentV2,
            policy_schedule_receipt_sha256=schedule_one.semantic_receipt_sha256,
        ),
        outer_fold_seal=seal_zero,
        prequential_state=state_zero,
    )

    release_two = _typed_shell(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        released_validation_fold_indices=(2,),
        semantic_receipt_sha256=_digest("release-2"),
    )
    release_three = _typed_shell(
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        released_validation_fold_indices=(3,),
        semantic_receipt_sha256=_digest("release-3"),
        predecessor_outer_fold_seal_receipt_sha256=_digest("seal-1"),
    )
    execution_two = _typed_shell(
        MassiveAdaptiveRLReleasedFoldValidationExecutionV1,
        fold_index=2,
        fold_validation_authority=SimpleNamespace(
            release_authority_receipt_sha256=release_two.semantic_receipt_sha256,
            semantic_receipt_sha256=_digest("validation-2"),
        ),
        policy_selection_authority=SimpleNamespace(
            semantic_receipt_sha256=_digest("selection-2")
        ),
        frozen_ppo_policy=ppo[2],
        frozen_fc06_control=controls[2],
    )
    execution_three = _typed_shell(
        MassiveAdaptiveRLReleasedFoldValidationExecutionV1,
        fold_index=3,
        fold_validation_authority=SimpleNamespace(
            release_authority_receipt_sha256=release_three.semantic_receipt_sha256,
            semantic_receipt_sha256=_digest("validation-3"),
        ),
        policy_selection_authority=SimpleNamespace(
            semantic_receipt_sha256=_digest("selection-3")
        ),
        frozen_ppo_policy=ppo[3],
        frozen_fc06_control=controls[3],
    )
    schedule_two = _typed_shell(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        fold_indices=(0, 1, 2),
        frozen_ppo_policy_receipts=tuple(
            row.semantic_receipt_sha256 for row in ppo[:3]
        ),
        semantic_receipt_sha256=_digest("schedule-2"),
    )
    schedule_three = _typed_shell(
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        fold_indices=(0, 1, 2, 3),
        frozen_ppo_policy_receipts=tuple(
            row.semantic_receipt_sha256 for row in ppo
        ),
        semantic_receipt_sha256=_digest("schedule-3"),
    )
    state_two_release = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.VALIDATION_2_RELEASED,
    )
    state_policy_two = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_2_FROZEN,
    )
    seal_one = _typed_shell(
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
        fold_index=1,
        semantic_receipt_sha256=_digest("seal-1"),
    )
    state_outer_one = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.OUTER_1_SEALED,
    )
    outer_one = _typed_shell(
        MassiveAdaptiveRLOuterFoldExecutionV1,
        fold_index=1,
        outer_access=_typed_shell(
            MassiveAdaptiveOuterAccessCommitmentV2,
            policy_schedule_receipt_sha256=schedule_two.semantic_receipt_sha256,
        ),
        outer_fold_seal=seal_one,
        prequential_state=state_outer_one,
    )
    state_three_release = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.VALIDATION_3_RELEASED,
    )
    state_policy_three = _typed_shell(
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        stage=MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN,
        stage_artifact_semantic_receipt_sha256=schedule_three.semantic_receipt_sha256,
        semantic_receipt_sha256=_digest("state-policy-3"),
    )

    for authority_type in (
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        MassiveAdaptiveRLInitialValidationExecutionV1,
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
        MassiveAdaptiveRLReleasedFoldValidationExecutionV1,
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
        MassiveAdaptiveRLOuterFoldExecutionV1,
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
        MassiveAdaptiveRLPrequentialExperimentStateV1,
        "prequential_execution_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        flow,
        "massive_adaptive_rl_experiment_materialization_lock_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        flow,
        "_materialization_mode_after_state_v1",
        lambda **_: True,
    )

    calls: list[str] = []
    releases = iter((release_two, release_three))
    executions = iter((execution_two, execution_three))
    states = iter(
        (
            state_two_release,
            state_policy_two,
            state_three_release,
            state_policy_three,
        )
    )

    def release(**kwargs):
        result = next(releases)
        calls.append(f"release-{result.released_validation_fold_indices[0]}")
        if result is release_two:
            assert kwargs["predecessor_state"] is state_zero
            assert kwargs["predecessor_outer_fold_seal"] is seal_zero
        else:
            assert kwargs["predecessor_state"] is state_outer_one
            assert kwargs["predecessor_outer_fold_seal"] is seal_one
        return result

    def validate_fold(**kwargs):
        result = next(executions)
        calls.append(f"validate-{result.fold_index}")
        assert kwargs["validation_release"].released_validation_fold_indices == (
            result.fold_index,
        )
        return result

    def schedule(**kwargs):
        policies = kwargs["frozen_ppo_policies"]
        result = schedule_two if len(policies) == 3 else schedule_three
        calls.append(f"schedule-{len(policies) - 1}")
        assert policies == ppo[: len(policies)]
        return result

    def append_state(**kwargs):
        result = next(states)
        calls.append(f"state-{result.stage.value}")
        return result

    def run_outer_one(**kwargs):
        calls.append("outer-1")
        assert kwargs["policy_schedule"] is schedule_two
        assert kwargs["predecessor_state"] is state_policy_two
        return outer_one

    monkeypatch.setattr(
        flow,
        "run_or_resume_massive_adaptive_rl_delayed_validation_release_v1",
        release,
    )
    monkeypatch.setattr(
        flow,
        "run_or_resume_massive_adaptive_rl_released_fold_validation_execution_v1",
        validate_fold,
    )
    monkeypatch.setattr(
        flow,
        "run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1",
        schedule,
    )
    monkeypatch.setattr(
        flow,
        "run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1",
        append_state,
    )
    monkeypatch.setattr(
        flow,
        "run_or_resume_massive_adaptive_rl_outer_fold_execution_v1",
        run_outer_one,
    )

    result = run_or_resume_massive_adaptive_rl_prequential_continuation_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=implementation,
        initial_inputs=initial_inputs,
        initial_execution=initial_execution,
        initial_policy_schedule=schedule_one,
        outer_zero_execution=outer_zero,
    )

    assert isinstance(result, MassiveAdaptiveRLPrequentialContinuationV1)
    assert calls == [
        "release-2",
        "state-validation-2-released",
        "validate-2",
        "schedule-2",
        "state-policy-2-frozen",
        "outer-1",
        "release-3",
        "state-validation-3-released",
        "validate-3",
        "schedule-3",
        "state-policy-3-frozen",
    ]
