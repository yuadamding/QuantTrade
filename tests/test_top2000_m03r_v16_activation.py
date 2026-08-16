from __future__ import annotations

from pathlib import Path

import pytest

from rl_quant.training.top2000_m03r_v16_activation import (
    M03RV16ActivationError,
    M03RV16TrainingActivation,
    issue_m03r_v16_qualification_activation,
    issue_m03r_v16_training_activation,
    load_m03r_v16_qualification_activation,
    load_m03r_v16_training_activation,
    write_m03r_v16_qualification_activation,
    write_m03r_v16_training_activation,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackageArtifacts,
    build_m03r_v16_package_plan,
)


def _surfaces() -> tuple[object, M03RV16ExecutionAuthorization]:
    digest = "a" * 64
    artifacts = M03RV16PackageArtifacts(
        source_archive_sha256=digest,
        source_manifest_sha256=digest,
        dependency_lock_sha256=digest,
        cache_artifact_sha256=digest,
        cache_manifest_sha256=digest,
        asset_axis_sha256=digest,
        risk_artifact_sha256=digest,
        risk_source_manifest_file_sha256=digest,
        risk_source_receipt_sha256=digest,
        exposure_receipt_sha256=digest,
        projector_manifest_file_sha256=digest,
        projector_manifest_sha256=digest,
        projector_binding_sha256=digest,
        worker_source_sha256=digest,
        operator_source_sha256=digest,
        initial_parameter_state_file_sha256=digest,
        initial_parameter_state_sha256=digest,
        initial_parameter_architecture_sha256=digest,
        structural_slab_file_sha256=digest,
        structural_slab_receipt_sha256=digest,
        structural_action_operator_root_sha256=digest,
        structural_target_operator_root_sha256=digest,
        structural_target_root_sha256=(digest, digest, digest),
        image_reference=f"registry.invalid/q@sha256:{digest}",
        image_digest_sha256=digest,
    )
    schedule = M03RV16PanelSchedule(
        protocol_common_data_sha256=digest,
        cache_sha256=digest,
        asset_axis_sha256=digest,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v16_fold_geometries(1001)
        ),
    )
    package = build_m03r_v16_package_plan(artifacts, schedule)
    authorization = M03RV16ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256="b" * 64,
        source_archive_sha256=digest,
        source_manifest_sha256=digest,
        worker_source_sha256=digest,
        structural_slab_file_sha256=digest,
        structural_slab_receipt_sha256=digest,
        image_reference=artifacts.image_reference,
    )
    authorization.validate(package)
    return package, authorization


def test_v16_phase_activations_round_trip_and_cannot_be_forged(
    tmp_path: Path,
) -> None:
    package, authorization = _surfaces()
    training = issue_m03r_v16_training_activation(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        static_gate_receipt_sha256="c" * 64,
        capacity_gate_receipt_sha256="d" * 64,
        source_tree_root_sha256="e" * 64,
    )
    training_path = tmp_path / "training.json"
    training_file = write_m03r_v16_training_activation(training_path, training)
    assert (
        load_m03r_v16_training_activation(
            training_path,
            expected_file_sha256=training_file,
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
        )
        == training
    )
    qualification = issue_m03r_v16_qualification_activation(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        training_panel_receipt_sha256="f" * 64,
        training_terminal_file_sha256=("1" * 64, "2" * 64, "3" * 64),
        primary_training_adequacy_receipt_sha256=tuple(
            f"{index + 10:064x}" for index in range(5)
        ),
        source_tree_root_sha256="e" * 64,
    )
    qualification_path = tmp_path / "qualification.json"
    qualification_file = write_m03r_v16_qualification_activation(
        qualification_path, qualification
    )
    assert (
        load_m03r_v16_qualification_activation(
            qualification_path,
            expected_file_sha256=qualification_file,
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
        )
        == qualification
    )
    with pytest.raises(TypeError):
        M03RV16TrainingActivation(  # type: ignore[call-arg]
            package_plan_sha256=package.package_plan_sha256,  # type: ignore[union-attr]
            execution_authorization_receipt_sha256=authorization.receipt_sha256,
            static_gate_receipt_sha256="c" * 64,
            capacity_gate_receipt_sha256="d" * 64,
            source_tree_root_sha256="e" * 64,
            image_digest_sha256="a" * 64,
        )

    qualification_path.chmod(0o600)
    qualification_path.write_bytes(qualification_path.read_bytes() + b"\n")
    with pytest.raises(M03RV16ActivationError, match="hash drifted"):
        load_m03r_v16_qualification_activation(
            qualification_path,
            expected_file_sha256=qualification_file,
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
        )
