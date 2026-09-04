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
from rl_quant.workflows import massive_adaptive_rl_experiment_runner_v5 as runner
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v4 import (
    MassiveAdaptiveRLPrequentialRunV4,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v5 import (
    MassiveAdaptiveRLPrequentialRunV5,
    _build_result,
    run_massive_adaptive_rl_experiment_v5,
    verify_massive_adaptive_rl_experiment_v5,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    build_massive_adaptive_rl_experiment_manifest_v5,
    write_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
)


_T = TypeVar("_T")


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type: type[_T], /, **values: object) -> _T:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def test_v5_result_binds_registration_before_initial_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="runner-v5-result"
    )
    registration = _typed_shell(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("registration"),
    )
    predecessor = _typed_shell(
        MassiveAdaptiveRLPrequentialRunV4,
        semantic_receipt_sha256=_digest("run-v4"),
        manifest_v4_receipt_sha256=manifest.base_manifest.semantic_receipt_sha256,
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
        property(lambda _: 20),
    )
    monkeypatch.setattr(
        runner,
        "load_massive_adaptive_rl_initial_validation_inputs_authority_v1",
        lambda **_: initial,
    )
    result = _build_result(
        manifest=manifest,
        registration=registration,
        predecessor=predecessor,
        artifact_root=tmp_path,
    )
    assert isinstance(result, MassiveAdaptiveRLPrequentialRunV5)
    assert result.manifest_v5_registration_committed_at_ms == 10
    assert result.initial_validation_inputs_committed_at_ms == 20
    assert result.released_validation_fold_indices == (0, 1)
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
        initial_validation_inputs_authority_receipt_sha256=(
            initial.semantic_receipt_sha256
        ),
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
        property(lambda _: 30),
    )
    registered = _build_result(
        manifest=manifest,
        registration=registration,
        predecessor=predecessor,
        artifact_root=tmp_path,
        implementation_registration=implementation_registration,
    )
    assert registered.execution_implementation_registered
    assert registered.execution_implementation_registration_committed_at_ms == 30
    assert (
        registered.next_required_stage
        == "prequential-fold-0-and-fold-1-validation-selection-and-freeze"
    )


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
    monkeypatch.setattr(runner, "_replay_v5_boundary", boundary)
    assert (
        verify_massive_adaptive_rl_experiment_v5(
            manifest_path=manifest_path,
            source_root=tmp_path / "source",
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        is expected
    )


def test_v5_root_api_has_no_validation_or_outer_choice_surface() -> None:
    assert tuple(inspect.signature(run_massive_adaptive_rl_experiment_v5).parameters) == (
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
