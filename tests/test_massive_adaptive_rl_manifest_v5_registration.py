from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.source_receipts import publish_massive_source_object
from rl_quant.protocol.canonical_artifact import semantic_sha256

from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v1 import (
    materialize_massive_adaptive_outer_access_commitment_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_executor_v1 import (
    run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v1 as run_fold_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_executor_v2 import (
    run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_executor_v1 import (
    run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v1 import (
    run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    prepare_or_resume_massive_adaptive_rl_validation_sources_v1,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v1 import (
    materialize_massive_adaptive_frozen_rl_policy_v1,
)
from rl_quant.training.massive_adaptive_rl_four_fold_policy_selection_v1 import (
    materialize_massive_adaptive_rl_four_fold_policy_selection_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v1 import (
    MassiveAdaptiveRLExperimentRunnerV1Error,
    run_massive_adaptive_rl_experiment_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    build_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    MassiveAdaptiveRLExperimentRunnerV2Error,
    run_massive_adaptive_rl_experiment_v2,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v3 import (
    MassiveAdaptiveRLExperimentRunnerV3Error,
    run_massive_adaptive_rl_experiment_v3,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v4 import (
    MassiveAdaptiveRLExperimentRunnerV4Error,
    run_massive_adaptive_rl_experiment_v4,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_orchestration_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    write_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    write_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
    MassiveAdaptiveRLManifestV5RegistrationError,
    load_massive_adaptive_rl_manifest_v5_registration_authority_v1,
    manifest_v5_registration_relative_path_v1,
    issue_massive_adaptive_rl_manifest_v5_initial_inputs_capability_v1,
    reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration,
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    write_massive_adaptive_rl_experiment_manifest_v2,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    authorize_legacy_or_manifest_v5_compatibility_writer_v1,
    authorize_massive_adaptive_rl_source_publication_v5,
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


def test_registration_is_create_only_exact_and_generic_load_is_nonauthorizing(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-v5"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    assert registration.source_transaction_verified
    assert registration.runtime_manifest_replayed
    assert registration.development_protocol_registered
    assert not registration.validation_outcome_access_authorized
    assert not registration.outer_access_authorized

    generic = load_massive_adaptive_rl_manifest_v5_registration_authority_v1(
        root=tmp_path,
        experiment_id=manifest.experiment_id,
        verified_at_ms=registration.source_transaction_committed_at_ms or 0,
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_manifest_replayed
    assert not generic.development_protocol_registered

    resumed = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
        allow_materialize=False,
    )
    assert resumed.semantic_receipt_sha256 == registration.semantic_receipt_sha256
    assert (
        resumed.source_transaction_receipt_sha256
        == registration.source_transaction_receipt_sha256
    )


def test_registration_fixed_path_rejects_a_different_manifest(tmp_path: Path) -> None:
    first = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-fixed", seeds=(17,)
    )
    second = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-fixed", seeds=(23,)
    )
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=first,
    )
    with pytest.raises(
        MassiveAdaptiveRLManifestV5RegistrationError, match="did not replay"
    ):
        run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
            root=tmp_path,
            manifest=second,
        )


def test_missing_read_only_registration_creates_nothing(tmp_path: Path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-read-only"
    )
    root = tmp_path / "absent"
    with pytest.raises(
        MassiveAdaptiveRLManifestV5RegistrationError, match="is absent"
    ):
        run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
            root=root,
            manifest=manifest,
            allow_materialize=False,
        )
    assert not root.exists()


def test_registration_must_precede_even_initial_validation_inputs(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-chronology"
    )
    relative = (
        tmp_path
        / "massive-adaptive"
        / "rl-prequential-initial-validation-inputs-authority-v1"
        / f"v4-{manifest.base_manifest.semantic_receipt_sha256}.json"
    )
    relative.parent.mkdir(parents=True)
    relative.write_text("partial", encoding="utf-8")
    with pytest.raises(
        MassiveAdaptiveRLManifestV5RegistrationError,
        match="must precede every validation input",
    ):
        run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
            root=tmp_path,
            manifest=manifest,
        )


def test_direct_registration_participates_in_the_experiment_global_lock(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-global-lock"
    )
    with massive_adaptive_rl_experiment_orchestration_lock_v1(
        artifact_root=tmp_path,
        experiment_id=manifest.experiment_id,
    ):
        with pytest.raises(
            MassiveAdaptiveRLManifestV5RegistrationError,
            match="already owned",
        ):
            run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
                root=tmp_path,
                manifest=manifest,
            )


def test_complete_or_partial_registration_disables_legacy_writers(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-legacy-guard"
    )
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    with pytest.raises(MassiveAdaptiveRLLegacyWriterRejectedByManifestV5):
        reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration(
            root=tmp_path,
            experiment_id=manifest.experiment_id,
        )

    partial_root = tmp_path / "partial"
    relative = partial_root / manifest_v5_registration_relative_path_v1(
        experiment_id="partial-registration"
    )
    relative.parent.mkdir(parents=True)
    relative.write_bytes(b"partial")
    with pytest.raises(MassiveAdaptiveRLLegacyWriterRejectedByManifestV5):
        reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration(
            root=partial_root,
            experiment_id="partial-registration",
        )


def test_registered_v5_disables_all_legacy_root_writers(tmp_path: Path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-root-guards"
    )
    artifact_root = tmp_path / "artifacts"
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=artifact_root,
        manifest=manifest,
    )
    manifest_v2_path = tmp_path / "manifest-v2.json"
    manifest_v3_path = tmp_path / "manifest-v3.json"
    manifest_v4_path = tmp_path / "manifest-v4.json"
    write_massive_adaptive_rl_experiment_manifest_v2(
        path=manifest_v2_path,
        manifest=manifest.base_manifest.base_manifest.base_manifest,
    )
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_v3_path,
        manifest=manifest.base_manifest.base_manifest,
    )
    write_massive_adaptive_rl_experiment_manifest_v4(
        path=manifest_v4_path,
        manifest=manifest.base_manifest,
    )
    with pytest.raises(MassiveAdaptiveRLExperimentRunnerV1Error, match="V1 writer"):
        run_massive_adaptive_rl_experiment_v1(
            manifest_path=manifest_v2_path,
            source_root=tmp_path / "source",
            artifact_root=artifact_root,
            device="cpu",
        )
    with pytest.raises(MassiveAdaptiveRLExperimentRunnerV2Error, match="V2 writer"):
        run_massive_adaptive_rl_experiment_v2(
            manifest_path=manifest_v3_path,
            source_root=tmp_path / "source",
            artifact_root=artifact_root,
            device="cpu",
        )
    with pytest.raises(MassiveAdaptiveRLExperimentRunnerV3Error, match="V3 writer"):
        run_massive_adaptive_rl_experiment_v3(
            manifest_path=manifest_v4_path,
            source_root=tmp_path / "source",
            artifact_root=artifact_root,
            device="cpu",
        )
    with pytest.raises(MassiveAdaptiveRLExperimentRunnerV4Error, match="V4 writer"):
        run_massive_adaptive_rl_experiment_v4(
            manifest_path=manifest_v4_path,
            source_root=tmp_path / "source",
            artifact_root=artifact_root,
            device="cpu",
        )


def test_registered_v5_disables_direct_legacy_child_materializers(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-child-guards"
    )
    artifact_root = tmp_path / "artifacts"
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=artifact_root,
        manifest=manifest,
    )
    manifest_v4 = manifest.base_manifest
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=artifact_root,
        manifest=manifest,
        allow_materialize=False,
    )
    capability = issue_massive_adaptive_rl_manifest_v5_initial_inputs_capability_v1(
        root=artifact_root,
        authority=registration
    )

    calls = (
        lambda: run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v1(
            root=artifact_root,
            manifest=manifest_v4,
            four_fold_fit_authority=None,  # type: ignore[arg-type]
            runtime_sources=None,  # type: ignore[arg-type]
            committed_at_ms=1,
        ),
        lambda: run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v2(
            root=artifact_root,
            manifest=manifest_v4,
            four_fold_fit_authority=None,  # type: ignore[arg-type]
            runtime_sources_v2=None,  # type: ignore[arg-type]
            committed_at_ms=1,
        ),
        lambda: run_fold_v1(
            root=artifact_root,
            manifest=manifest_v4,
            runtime_sources_v2=None,  # type: ignore[arg-type]
            four_fold_fit_authority=None,  # type: ignore[arg-type]
            four_fold_validation_inputs_v2=None,  # type: ignore[arg-type]
            fold_index=0,
            committed_at_ms=1,
        ),
        lambda: run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2(
            root=artifact_root,
            manifest=manifest_v4,
            runtime_sources_v2=None,  # type: ignore[arg-type]
            four_fold_fit_authority=None,  # type: ignore[arg-type]
            four_fold_validation_inputs_v2=None,  # type: ignore[arg-type]
            fold_index=0,
        ),
        lambda: run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1(
            root=artifact_root,
            manifest=manifest_v4,
            runtime_sources_v2=None,  # type: ignore[arg-type]
            four_fold_fit_authority=None,  # type: ignore[arg-type]
        ),
        lambda: materialize_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
            root=artifact_root,
            manifest=manifest_v4,
            authority=None,  # type: ignore[arg-type]
            committed_at_ms=1,
        ),
        lambda: materialize_massive_adaptive_frozen_rl_policy_v1(
            root=artifact_root,
            artifact_id="legacy",
            checkpoint=None,  # type: ignore[arg-type]
            selection_authority=None,  # type: ignore[arg-type]
            committed_at_ms=1,
        ),
        lambda: materialize_massive_adaptive_outer_access_commitment_v1(
            root=artifact_root,
            artifact_id="legacy",
            outer_inference_plan=None,  # type: ignore[arg-type]
            calibration=None,  # type: ignore[arg-type]
            policy_selection_authority=None,  # type: ignore[arg-type]
            frozen_policy=None,  # type: ignore[arg-type]
            fixed_control_registry=None,  # type: ignore[arg-type]
            fixed_control_fit_authority=None,  # type: ignore[arg-type]
            fixed_control_selection_authority=None,  # type: ignore[arg-type]
            chronology_authority=None,  # type: ignore[arg-type]
            compiler_config=None,  # type: ignore[arg-type]
            committed_at_ms=1,
        ),
    )
    for call in calls:
        with pytest.raises(MassiveAdaptiveRLLegacyWriterRejectedByManifestV5):
            call()

    with pytest.raises(MassiveAdaptiveRLLegacyWriterRejectedByManifestV5):
        prepare_or_resume_massive_adaptive_rl_validation_sources_v1(
            root=artifact_root,
            manifest=manifest_v4,
            four_fold_fit_authority=None,  # type: ignore[arg-type]
            runtime_sources=None,  # type: ignore[arg-type]
            fold_index=0,
            committed_at_ms=1,
        )
    with pytest.raises(
        MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
        match="exceeds its fold release",
    ):
        prepare_or_resume_massive_adaptive_rl_validation_sources_v1(
            root=artifact_root,
            manifest=manifest_v4,
            four_fold_fit_authority=None,  # type: ignore[arg-type]
            runtime_sources=None,  # type: ignore[arg-type]
            fold_index=2,
            committed_at_ms=1,
            v5_writer_capability=capability,
        )


def test_writer_capability_replays_registration_and_is_role_scoped(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-capability-replay"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    capability = issue_massive_adaptive_rl_manifest_v5_initial_inputs_capability_v1(
        root=tmp_path,
        authority=registration,
    )
    forged = replace(capability, registration_commit_receipt_sha256="f" * 64)
    with pytest.raises(
        MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
        match="persisted registration",
    ):
        authorize_legacy_or_manifest_v5_compatibility_writer_v1(
            root=tmp_path,
            experiment_id=manifest.experiment_id,
            manifest_v4_receipt_sha256=(
                manifest.base_manifest.semantic_receipt_sha256
            ),
            writer_role="initial-validation-inputs",
            fold_index=0,
            capability=forged,
        )

    with massive_adaptive_rl_manifest_v5_writer_scope_v1(
        root=tmp_path,
        capability=capability,
    ):
        authorize_massive_adaptive_rl_source_publication_v5(
            root=tmp_path,
            relative_payload_path=(
                "massive-adaptive/rl-validation-inputs-v1/allowed.json"
            ),
        )
        with pytest.raises(
            MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
            match="does not authorize",
        ):
            authorize_massive_adaptive_rl_source_publication_v5(
                root=tmp_path,
                relative_payload_path=(
                    "massive-adaptive/rl-policy-selection-authority-v3/forbidden.json"
                ),
            )

    relative = "massive-adaptive/rl-policy-selection-authority-v3/blocked.json"
    with pytest.raises(MassiveAdaptiveRLLegacyWriterRejectedByManifestV5):
        publish_massive_source_object(
            stream=BytesIO(b"{}"),
            root=tmp_path,
            relative_payload_path=relative,
            dataset_id="guard-test",
            source_object_key=relative,
            requested_at_ms=1,
            downloaded_at_ms=1,
            schema_sha256=semantic_sha256("schema"),
            entitlement_receipt_sha256=semantic_sha256("entitlement"),
            committed_at_ms=1,
            request_id="GUARD-TEST",
        )
    assert not (tmp_path / relative).exists()


def test_writer_capability_binds_separate_source_publication_root(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="registration-source-root"
    )
    artifact_root = tmp_path / "artifacts"
    source_root = tmp_path / "source"
    other_root = tmp_path / "other"
    source_root.mkdir()
    other_root.mkdir()
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=artifact_root,
        manifest=manifest,
    )
    capability = issue_massive_adaptive_rl_manifest_v5_initial_inputs_capability_v1(
        root=artifact_root,
        authority=registration,
        source_root=source_root,
    )

    with pytest.raises(MassiveAdaptiveRLLegacyWriterRejectedByManifestV5):
        authorize_massive_adaptive_rl_source_publication_v5(
            root=artifact_root,
            relative_payload_path="adaptive-rl/source-bundle-v2/unguarded.json",
        )

    with massive_adaptive_rl_manifest_v5_writer_scope_v1(
        root=artifact_root,
        capability=capability,
    ):
        authorize_massive_adaptive_rl_source_publication_v5(
            root=source_root,
            relative_payload_path="adaptive-rl/source-bundle-v2/allowed.json",
        )
        with pytest.raises(
            MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
            match="publication root",
        ):
            authorize_massive_adaptive_rl_source_publication_v5(
                root=other_root,
                relative_payload_path="adaptive-rl/source-bundle-v2/forbidden.json",
            )
