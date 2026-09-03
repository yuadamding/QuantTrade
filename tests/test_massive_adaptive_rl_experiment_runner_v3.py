from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_rl_four_fold_policy_selection_v1 import (
    MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
    MassiveAdaptiveRLFourFoldSelectionDispositionV1,
)
from rl_quant.workflows import massive_adaptive_rl_experiment_runner_v3 as runner
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v3 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SPEC_SHA256,
    MassiveAdaptiveRLExperimentRunnerV3Error,
    _build_result,
    _experiment_v3_orchestration_lease,
    _validate_training_handoff,
    run_massive_adaptive_rl_experiment_v3,
    verify_massive_adaptive_rl_experiment_v3,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2,
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


@pytest.mark.parametrize(
    "disposition",
    tuple(row.value for row in MassiveAdaptiveRLFourFoldSelectionDispositionV1),
)
def test_validation_run_result_stops_before_freeze_and_outer(
    monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id=f"runner-result-{disposition}"
    )
    states = _states(manifest)
    monkeypatch.setattr(MassiveAdaptiveRLRuntimeSourcesV2, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
        "source_receipt_sha256",
        property(lambda _: _digest("aggregate-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("aggregate-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
        "development_stage_authorized",
        property(lambda _: True),
    )
    runtime = _typed_shell(
        MassiveAdaptiveRLRuntimeSourcesV2,
        semantic_receipt_sha256=_digest("runtime-v2"),
    )
    aggregate = _typed_shell(
        MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
        semantic_receipt_sha256=_digest("aggregate"),
        manifest_v4_receipt_sha256=manifest.semantic_receipt_sha256,
        training_manifest_v3_receipt_sha256=(
            manifest.base_manifest.semantic_receipt_sha256
        ),
        runtime_sources_v2_receipt_sha256=runtime.semantic_receipt_sha256,
        four_fold_fit_authority_receipt_sha256=_digest("fit"),
        selection_disposition=disposition,
    )
    result = _build_result(
        manifest=manifest,
        states=states,
        runtime_sources_v2=runtime,
        selection_authority=aggregate,
    )
    qualified = (
        disposition
        == MassiveAdaptiveRLFourFoldSelectionDispositionV1.FOUR_FOLD_SELECTIONS_QUALIFIED.value
    )
    assert result.validation_execution_complete
    assert result.positive_profitability_authorization_eligible is qualified
    assert result.no_qualified_policy is not qualified
    assert result.next_required_stage == (
        "selection-v3-aware-walk-forward-policy-freeze" if qualified else None
    )
    assert not result.final_policy_freezing_authorized
    assert not result.outer_access_authorized
    assert not result.profitability_reporting_authorized
    assert not result.end_to_end_profitability_execution_complete


def test_training_handoff_requires_the_exact_backend_blocker() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-handoff"
    )
    trained, blocked = _states(manifest)
    object.__setattr__(blocked, "blocker_code", "different-blocker")
    with pytest.raises(
        MassiveAdaptiveRLExperimentRunnerV3Error,
        match="no exact training handoff",
    ):
        _validate_training_handoff(manifest=manifest, states=(trained, blocked))


def test_root_runner_adopts_training_then_invokes_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-root"
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
        runner,
        "_experiment_v3_orchestration_lease",
        lambda **_: nullcontext(),
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

    def validation_root(**kwargs):
        observed["validation_manifest"] = kwargs["manifest"]
        observed["allow_materialize"] = kwargs["allow_materialize"]
        return expected

    monkeypatch.setattr(runner, "_replay_validation_root", validation_root)
    result = run_massive_adaptive_rl_experiment_v3(
        manifest_path=manifest_path,
        source_root=tmp_path / "source",
        artifact_root=tmp_path / "artifacts",
        device="cpu",
    )
    assert result is expected
    assert observed["training_manifest"] == manifest.base_manifest
    assert observed["validation_manifest"] == manifest
    assert observed["allow_materialize"] is True


def test_root_runner_returns_training_blocker_without_opening_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="runner-training-blocked"
    )
    manifest_path = tmp_path / "manifest-v4.json"
    write_massive_adaptive_rl_experiment_manifest_v4(
        path=manifest_path, manifest=manifest
    )
    training = SimpleNamespace(four_fold_fit_authority_receipt_sha256=None)
    monkeypatch.setattr(
        runner,
        "_experiment_v3_orchestration_lease",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        runner,
        "_run_massive_adaptive_rl_experiment_v2_unlocked",
        lambda **_: training,
    )
    monkeypatch.setattr(
        runner,
        "_replay_validation_root",
        lambda **_: pytest.fail("validation must remain unopened"),
    )
    assert (
        run_massive_adaptive_rl_experiment_v3(
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
        experiment_id="runner-verify"
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

    def validation_root(**kwargs):
        assert kwargs["allow_materialize"] is False
        return expected

    monkeypatch.setattr(runner, "_replay_validation_root", validation_root)
    assert (
        verify_massive_adaptive_rl_experiment_v3(
            manifest_path=manifest_path,
            source_root=tmp_path / "source",
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        is expected
    )


def test_root_runner_api_has_no_validation_outcome_choice_surface() -> None:
    parameters = inspect.signature(run_massive_adaptive_rl_experiment_v3).parameters
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
        "environment",
        "actions",
        "targets",
        "metrics",
        "selected_checkpoint",
        "outer_data",
    }.intersection(parameters)


def test_root_lease_does_not_mask_body_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="root body failed"):
        with _experiment_v3_orchestration_lease(
            artifact_root=tmp_path, experiment_id="lease-body-error"
        ):
            raise OSError("root body failed")


def test_runner_v3_protocol_hashes_are_bound() -> None:
    for value in (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SPEC_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)
