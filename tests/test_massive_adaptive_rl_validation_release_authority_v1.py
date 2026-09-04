from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import pytest

from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1Error,
    build_massive_adaptive_rl_initial_validation_release_authority_v1,
    load_massive_adaptive_rl_validation_release_authority_v1,
    run_or_resume_massive_adaptive_rl_initial_validation_release_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import (
    massive_adaptive_rl_execution_implementation_registration_v1 as implementation_module,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
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


def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="initial-validation-release"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    registration_time = registration.source_transaction_committed_at_ms
    assert registration_time is not None
    fit_receipt = _digest("four-fold-fit")
    execution_time = registration_time + 1
    execution = _typed_shell(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("execution-registration"),
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        manifest_v5_registration_authority_receipt_sha256=(
            registration.semantic_receipt_sha256
        ),
        training_state_receipt_sha256=_digest("training-state"),
        four_fold_fit_authority_receipt_sha256=fit_receipt,
        source_data_qualified=True,
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
        property(lambda _: _digest("execution-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("execution-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "source_transaction_committed_at_ms",
        property(lambda _: execution_time),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "scientific_execution_fingerprint_sha256",
        property(lambda _: _digest("scientific-execution-fingerprint")),
    )
    monkeypatch.setattr(
        implementation_module,
        "run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1",
        lambda **_: execution,
    )

    initial_time = registration_time + 6
    initial = _typed_shell(
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
        semantic_receipt_sha256=_digest("initial-inputs"),
        manifest_v4_receipt_sha256=manifest.base_manifest.semantic_receipt_sha256,
        four_fold_fit_authority_receipt_sha256=fit_receipt,
        prequential_validation_plan_receipt_sha256=_digest("prequential-plan"),
        runtime_sources_v2_receipt_sha256=_digest("runtime-sources-v2"),
        source_bundle_v2_receipt_sha256=_digest("source-bundle-v2"),
        runtime_source_graph_v2_receipt_sha256=_digest("source-graph-v2"),
        runtime_source_graph_v2_witness_receipt_sha256=_digest(
            "source-graph-v2-witness"
        ),
        replay_dependency_index_v2_receipt_sha256=_digest("dependency-index-v2"),
        training_source_projection_sha256=_digest("training-projection"),
        validation_source_projection_sha256=_digest("validation-projection"),
        validation_sources_v2_receipts=(
            _digest("validation-source-0"),
            _digest("validation-source-1"),
        ),
        validation_sources_v2_source_receipts=(
            _digest("validation-source-source-0"),
            _digest("validation-source-source-1"),
        ),
        validation_sources_v2_commit_receipts=(
            _digest("validation-source-commit-0"),
            _digest("validation-source-commit-1"),
        ),
        validation_sources_v2_committed_at_ms=(
            registration_time + 2,
            registration_time + 3,
        ),
        validation_environment_registry_v2_receipts=(
            _digest("validation-registry-0"),
            _digest("validation-registry-1"),
        ),
        validation_registry_v2_source_receipts=(
            _digest("validation-registry-source-0"),
            _digest("validation-registry-source-1"),
        ),
        validation_registry_v2_commit_receipts=(
            _digest("validation-registry-commit-0"),
            _digest("validation-registry-commit-1"),
        ),
        validation_registry_v2_committed_at_ms=(
            registration_time + 4,
            registration_time + 5,
        ),
        validation_context_receipts=(
            _digest("validation-context-0"),
            _digest("validation-context-1"),
        ),
        validation_decision_session_date_inventories=(
            ("2020-01-02",),
            ("2020-07-02",),
        ),
        expected_candidate_checkpoint_authority_receipt_inventories=(
            (_digest("checkpoint-0-0"),),
            (_digest("checkpoint-1-0"), _digest("checkpoint-1-1")),
        ),
        source_data_qualified=True,
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
        property(lambda _: initial_time),
    )
    return manifest, registration, execution, initial


def test_initial_validation_release_is_create_only_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, registration, execution, initial = _roots(tmp_path, monkeypatch)

    authority = run_or_resume_massive_adaptive_rl_initial_validation_release_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=execution,
        initial_inputs=initial,
    )
    assert authority.development_stage_authorized
    assert authority.released_validation_fold_indices == (0, 1)
    assert authority.withheld_validation_fold_indices == (2, 3)
    assert authority.predecessor_outer_fold_seal_receipt_sha256 is None
    assert (
        authority.execution_implementation_registration_authority_receipt_sha256
        == execution.semantic_receipt_sha256
    )
    assert authority.source_transaction_committed_at_ms is not None
    assert (
        authority.source_transaction_committed_at_ms
        > authority.initial_validation_inputs_committed_at_ms
    )

    generic = load_massive_adaptive_rl_validation_release_authority_v1(
        root=tmp_path,
        manifest=manifest,
        verified_at_ms=authority.source_transaction_committed_at_ms,
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_lineage_replayed
    assert not generic.development_stage_authorized

    resumed = run_or_resume_massive_adaptive_rl_initial_validation_release_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=execution,
        initial_inputs=initial,
        allow_materialize=False,
    )
    assert resumed.semantic_receipt_sha256 == authority.semantic_receipt_sha256
    assert resumed.development_stage_authorized


def test_initial_validation_release_rejects_changed_lineage_and_later_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, registration, execution, initial = _roots(tmp_path, monkeypatch)
    authority = build_massive_adaptive_rl_initial_validation_release_authority_v1(
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=execution,
        initial_inputs=initial,
    )

    changed = replace(
        authority,
        released_validation_fold_indices=(0, 1, 2),
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    with pytest.raises(
        MassiveAdaptiveRLValidationReleaseAuthorityV1Error,
        match="release differs",
    ):
        changed.validate()

    run_or_resume_massive_adaptive_rl_initial_validation_release_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=execution,
        initial_inputs=initial,
    )

    changed_execution = _typed_shell(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("different-execution-registration"),
        manifest_v5_receipt_sha256=manifest.semantic_receipt_sha256,
        manifest_v5_registration_authority_receipt_sha256=(
            registration.semantic_receipt_sha256
        ),
        training_state_receipt_sha256=_digest("training-state"),
        four_fold_fit_authority_receipt_sha256=_digest("four-fold-fit"),
        source_data_qualified=True,
    )
    with pytest.raises(
        MassiveAdaptiveRLValidationReleaseAuthorityV1Error,
        match="did not replay",
    ):
        run_or_resume_massive_adaptive_rl_initial_validation_release_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            execution_registration=changed_execution,
            initial_inputs=initial,
            allow_materialize=False,
        )
