from __future__ import annotations

from pathlib import Path

import pytest

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
    reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration,
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
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
    manifest_v3_path = tmp_path / "manifest-v3.json"
    manifest_v4_path = tmp_path / "manifest-v4.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_v3_path,
        manifest=manifest.base_manifest.base_manifest,
    )
    write_massive_adaptive_rl_experiment_manifest_v4(
        path=manifest_v4_path,
        manifest=manifest.base_manifest,
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
