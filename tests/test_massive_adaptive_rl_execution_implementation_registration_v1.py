from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest

from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import (
    massive_adaptive_rl_execution_implementation_registration_v1 as implementation,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationV1Error,
    load_massive_adaptive_rl_execution_implementation_registration_v1,
    run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    build_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
)


_T = TypeVar("_T")


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type: type[_T], /, **values: object) -> _T:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _initial_inputs_shell(
    *,
    manifest_v4_receipt_sha256: str,
    committed_at_ms: int,
    monkeypatch: pytest.MonkeyPatch,
) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    initial = _typed_shell(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        semantic_receipt_sha256=_digest("initial-inputs"),
        manifest_v4_receipt_sha256=manifest_v4_receipt_sha256,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        "development_stage_authorized",
        property(lambda _: True),
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
        "source_transaction_committed_at_ms",
        property(lambda _: committed_at_ms),
    )
    return initial


def _qualified_capture_body(
    *,
    manifest,
    registration,
    initial_inputs,
) -> dict[str, object]:
    body = implementation._capture_body(
        manifest=manifest,
        manifest_registration=registration,
        initial_inputs=initial_inputs,
    )
    body.update(
        {
            "source_worktree_clean": True,
            "source_worktree_status": (),
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "float32_matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "torch_cpu_threads": 1,
            "torch_interop_threads": 1,
            "process_thread_environment": (
                ("OMP_NUM_THREADS", "1"),
                ("MKL_NUM_THREADS", "1"),
                ("OPENBLAS_NUM_THREADS", "1"),
                ("NUMEXPR_NUM_THREADS", "1"),
                ("PYTHONHASHSEED", "0"),
            ),
            "source_data_qualified": True,
        }
    )
    return body


def test_execution_registration_is_separate_create_only_and_replay_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="execution-registration"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    registration_time = registration.source_transaction_committed_at_ms
    assert registration_time is not None
    initial = _initial_inputs_shell(
        manifest_v4_receipt_sha256=manifest.base_manifest.semantic_receipt_sha256,
        committed_at_ms=registration_time + 1,
        monkeypatch=monkeypatch,
    )
    body = _qualified_capture_body(
        manifest=manifest,
        registration=registration,
        initial_inputs=initial,
    )
    monkeypatch.setattr(implementation, "_capture_body", lambda **_: dict(body))

    authority = (
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            initial_inputs=initial,
        )
    )
    assert authority.development_execution_registered
    assert authority.manifest_v5_registration_committed_at_ms == registration_time
    assert (
        authority.initial_validation_inputs_committed_at_ms
        == registration_time + 1
    )
    assert authority.source_transaction_committed_at_ms is not None
    assert (
        authority.source_transaction_committed_at_ms
        > authority.initial_validation_inputs_committed_at_ms
    )
    assert not authority.outer_access_authorized
    assert not authority.profitability_reporting_authorized

    generic = load_massive_adaptive_rl_execution_implementation_registration_v1(
        root=tmp_path,
        experiment_id=manifest.experiment_id,
        verified_at_ms=authority.source_transaction_committed_at_ms,
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_implementation_replayed
    assert not generic.development_execution_registered

    resumed = (
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            initial_inputs=initial,
            allow_materialize=False,
        )
    )
    assert resumed.semantic_receipt_sha256 == authority.semantic_receipt_sha256
    assert resumed.development_execution_registered


def test_execution_registration_rejects_untracked_source_and_late_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="execution-registration-fail-closed"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    registration_time = registration.source_transaction_committed_at_ms
    assert registration_time is not None
    initial = _initial_inputs_shell(
        manifest_v4_receipt_sha256=manifest.base_manifest.semantic_receipt_sha256,
        committed_at_ms=registration_time + 1,
        monkeypatch=monkeypatch,
    )
    body = _qualified_capture_body(
        manifest=manifest,
        registration=registration,
        initial_inputs=initial,
    )
    body.update(
        {
            "source_worktree_clean": False,
            "source_worktree_status": ("?? src/rl_quant/unregistered.py",),
            "source_data_qualified": False,
        }
    )
    monkeypatch.setattr(implementation, "_capture_body", lambda **_: dict(body))
    with pytest.raises(
        MassiveAdaptiveRLExecutionImplementationRegistrationV1Error,
        match="not scientifically qualified",
    ):
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            initial_inputs=initial,
        )

    late = (
        tmp_path
        / "adaptive-rl"
        / manifest.experiment_id
        / "validation-outcome-v3"
    )
    late.mkdir(parents=True)
    body.update(
        {
            "source_worktree_clean": True,
            "source_worktree_status": (),
            "source_data_qualified": True,
        }
    )
    with pytest.raises(
        MassiveAdaptiveRLExecutionImplementationRegistrationV1Error,
        match="must precede every validation outcome",
    ):
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            initial_inputs=initial,
        )
