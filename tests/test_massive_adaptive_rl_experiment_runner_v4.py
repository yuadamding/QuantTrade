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
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_experiment_runner_v4 as runner
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v4 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SPEC_SHA256,
    MassiveAdaptiveRLExperimentRunnerV4Error,
    _build_result,
    _experiment_v4_orchestration_lease,
    _replay_prequential_root,
    _validate_training_handoff,
    run_massive_adaptive_rl_experiment_v4,
    verify_massive_adaptive_rl_experiment_v4,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    build_massive_adaptive_rl_experiment_manifest_v4,
    write_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
)


_T = TypeVar("_T")


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type: type[_T], /, **values: object) -> _T:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _states(manifest):
    common = {
        "experiment_id": manifest.experiment_id,
        "manifest_receipt_sha256": manifest.base_manifest.semantic_receipt_sha256,
        "source_data_qualified": True,
    }
    trained = _typed_shell(
        MassiveAdaptiveRLExperimentStateV2,
        **common,
        stage=MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED,
        stage_artifact_receipt_sha256=_digest("fit"),
        blocker_code=None,
        semantic_receipt_sha256=_digest("trained-state"),
    )
    blocked = _typed_shell(
        MassiveAdaptiveRLExperimentStateV2,
        **common,
        stage=MassiveAdaptiveRLExperimentStageV2.BLOCKED,
        stage_artifact_receipt_sha256=None,
        blocker_code="inner-validation-backend-required",
        semantic_receipt_sha256=_digest("blocked-state"),
    )
    return (trained, blocked)


def test_prequential_result_stops_before_outcomes_and_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-v4-result"
    )
    states = _states(manifest)
    plan = SimpleNamespace(semantic_receipt_sha256=_digest("plan"))
    runtime = _typed_shell(
        MassiveAdaptiveRLRuntimeSourcesV2,
        semantic_receipt_sha256=_digest("runtime-v2"),
    )
    initial = _typed_shell(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        semantic_receipt_sha256=_digest("initial-inputs"),
        manifest_v4_receipt_sha256=manifest.semantic_receipt_sha256,
        training_manifest_v3_receipt_sha256=(
            manifest.base_manifest.semantic_receipt_sha256
        ),
        runtime_sources_v2_receipt_sha256=runtime.semantic_receipt_sha256,
        four_fold_fit_authority_receipt_sha256=_digest("fit"),
        _runtime_plan=plan,
    )
    monkeypatch.setattr(MassiveAdaptiveRLRuntimeSourcesV2, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        "validate",
        lambda _: None,
    )
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
        "development_stage_authorized",
        property(lambda _: True),
    )
    result = _build_result(
        manifest=manifest,
        states=states,
        runtime_sources_v2=runtime,
        initial_inputs=initial,
    )
    assert result.released_validation_fold_indices == (0, 1)
    assert result.withheld_validation_fold_indices == (2, 3)
    assert result.diagnostic_continuation_registered
    assert not result.validation_execution_complete
    assert result.policy_schedule_disposition is None
    assert result.next_required_stage == (
        "prequential-fold-0-and-fold-1-validation-selection-and-freeze"
    )
    assert not result.positive_profitability_authorization_eligible
    assert not result.final_policy_freezing_authorized
    assert not result.outer_access_authorized
    assert not result.profitability_reporting_authorized
    assert not result.end_to_end_profitability_execution_complete


def test_training_handoff_requires_the_exact_backend_blocker() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-v4-handoff"
    )
    trained, blocked = _states(manifest)
    object.__setattr__(blocked, "blocker_code", "different-blocker")
    with pytest.raises(
        MassiveAdaptiveRLExperimentRunnerV4Error,
        match="no exact training handoff",
    ):
        _validate_training_handoff(manifest=manifest, states=(trained, blocked))


def test_replay_root_commits_only_initial_prequential_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-v4-replay"
    )
    states = _states(manifest)
    runtime = _typed_shell(
        MassiveAdaptiveRLRuntimeSourcesV2,
        _base_runtime_sources_v1=object(),
    )
    fit = _typed_shell(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        semantic_receipt_sha256=_digest("fit"),
    )
    initial = _typed_shell(MassiveAdaptiveRLInitialValidationInputsAuthorityV1)
    expected = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(MassiveAdaptiveRLRuntimeSourcesV2, "validate", lambda _: None)
    monkeypatch.setattr(
        runner,
        "reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v2",
        lambda **_: runtime,
    )
    monkeypatch.setattr(
        runner,
        "load_massive_adaptive_rl_four_fold_fit_authority_v1",
        lambda **_: fit,
    )

    def initial_inputs(**kwargs):
        observed.update(kwargs)
        return initial

    monkeypatch.setattr(
        runner,
        "run_or_resume_massive_adaptive_rl_initial_validation_inputs_v1",
        initial_inputs,
    )
    monkeypatch.setattr(runner, "_build_result", lambda **_: expected)
    assert (
        _replay_prequential_root(
            manifest=manifest,
            source_root=tmp_path / "source",
            artifact_root=tmp_path / "artifacts",
            device="cpu",
            states=states,
            allow_materialize=True,
        )
        is expected
    )
    assert observed["runtime_sources_v2"] is runtime
    assert observed["four_fold_fit_authority"] is fit
    assert observed["allow_materialize"] is True
    assert not hasattr(runner, "run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1")


def test_root_runner_adopts_training_then_invokes_prequential_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-v4-root"
    )
    manifest_path = tmp_path / "manifest-v4.json"
    write_massive_adaptive_rl_experiment_manifest_v4(
        path=manifest_path, manifest=manifest
    )
    training = SimpleNamespace(four_fold_fit_authority_receipt_sha256=_digest("fit"))
    states = _states(manifest)
    expected = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        runner, "_experiment_v4_orchestration_lease", lambda **_: nullcontext()
    )

    def training_run(**kwargs):
        observed["training_manifest"] = kwargs["manifest"]
        return training

    monkeypatch.setattr(
        runner, "_run_massive_adaptive_rl_experiment_v2_unlocked", training_run
    )
    monkeypatch.setattr(
        runner, "load_massive_adaptive_rl_experiment_states_v2", lambda **_: states
    )

    def prequential_root(**kwargs):
        observed["root_manifest"] = kwargs["manifest"]
        observed["allow_materialize"] = kwargs["allow_materialize"]
        return expected

    monkeypatch.setattr(runner, "_replay_prequential_root", prequential_root)
    assert (
        run_massive_adaptive_rl_experiment_v4(
            manifest_path=manifest_path,
            source_root=tmp_path / "source",
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        is expected
    )
    assert observed["training_manifest"] == manifest.base_manifest
    assert observed["root_manifest"] == manifest
    assert observed["allow_materialize"] is True


def test_root_runner_returns_training_blocker_without_opening_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-v4-training-blocked"
    )
    manifest_path = tmp_path / "manifest-v4.json"
    write_massive_adaptive_rl_experiment_manifest_v4(
        path=manifest_path, manifest=manifest
    )
    training = SimpleNamespace(four_fold_fit_authority_receipt_sha256=None)
    monkeypatch.setattr(
        runner, "_experiment_v4_orchestration_lease", lambda **_: nullcontext()
    )
    monkeypatch.setattr(
        runner,
        "_run_massive_adaptive_rl_experiment_v2_unlocked",
        lambda **_: training,
    )
    monkeypatch.setattr(
        runner,
        "_replay_prequential_root",
        lambda **_: pytest.fail("validation inputs must remain unopened"),
    )
    assert (
        run_massive_adaptive_rl_experiment_v4(
            manifest_path=manifest_path,
            source_root=tmp_path / "source",
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        is training
    )


def test_root_verification_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-v4-verify"
    )
    manifest_path = tmp_path / "manifest-v4.json"
    write_massive_adaptive_rl_experiment_manifest_v4(
        path=manifest_path, manifest=manifest
    )
    states = _states(manifest)
    expected = object()
    monkeypatch.setattr(
        runner, "load_massive_adaptive_rl_experiment_states_v2", lambda **_: states
    )

    def prequential_root(**kwargs):
        assert kwargs["allow_materialize"] is False
        return expected

    monkeypatch.setattr(runner, "_replay_prequential_root", prequential_root)
    assert (
        verify_massive_adaptive_rl_experiment_v4(
            manifest_path=manifest_path,
            source_root=tmp_path / "source",
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        is expected
    )


def test_root_runner_api_has_no_validation_or_outer_choice_surface() -> None:
    parameters = inspect.signature(run_massive_adaptive_rl_experiment_v4).parameters
    assert tuple(parameters) == (
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
    }.intersection(parameters)


def test_root_lease_does_not_mask_body_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="root body failed"):
        with _experiment_v4_orchestration_lease(
            artifact_root=tmp_path, experiment_id="lease-body-error"
        ):
            raise OSError("root body failed")


def test_runner_v4_protocol_hashes_are_bound() -> None:
    for value in (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SPEC_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)
