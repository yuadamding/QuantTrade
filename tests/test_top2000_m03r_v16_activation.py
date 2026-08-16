from __future__ import annotations

import json
from pathlib import Path

import pytest

import rl_quant.training.top2000_m03r_v16_kubernetes as kubernetes_runtime
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    M03RV16ActivationError,
    M03RV16TrainingActivation,
    _issue_m03r_v16_qualification_activation_from_panel,
    _issue_m03r_v16_training_activation_from_gates,
    load_m03r_v16_qualification_activation,
    load_m03r_v16_training_activation,
    write_m03r_v16_qualification_activation,
    write_m03r_v16_training_activation,
)
from rl_quant.training.top2000_m03r_v16_capacity import (
    M03R_V16_CAPACITY_TERMINAL_SCHEMA,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_kubernetes import (
    M03RV16CapacityGateQualification,
    M03RV16StaticGateQualification,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackageArtifacts,
    build_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_static_contract import (
    M03R_V16_STATIC_RESULT_SCHEMA,
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
    source_root = "e" * 64
    static_unsigned = {
        "schema": M03R_V16_STATIC_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "source_tree_root_sha256": source_root,
        "training_performed": False,
        "gpu_mask": "none",
        "gpu_requests": 0,
        "gpu_limits": 0,
        "unmasked_visibility_claimed": False,
        "initial_state_strict_loaded_all_settings": True,
    }
    static_result = {
        **static_unsigned,
        "receipt_sha256": semantic_sha256(static_unsigned),
    }
    static_path = tmp_path / "static-result.json"
    static_path.write_bytes(canonical_json_file_bytes(static_result))
    capacity_result = {
        "schema": M03R_V16_CAPACITY_TERMINAL_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
        "authorization_receipt_sha256": authorization.receipt_sha256,
        "source_tree_root_sha256": source_root,
        "scientific_training_performed": False,
        "disposable_train_validate_train_executed": True,
        "disposable_optimizer_update_executed": True,
        "scientific_checkpoint_published": False,
        "capacity": {"synthetic": True},
        "capacity_receipt_sha256": semantic_sha256({"synthetic": True}),
    }
    capacity_path = tmp_path / "capacity.json"
    capacity_path.write_bytes(canonical_json_file_bytes(capacity_result))
    static = M03RV16StaticGateQualification(
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[union-attr]
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        rendered_manifest_sha256="6" * 64,
        result_file_sha256=file_sha256(static_path),
        result_receipt_sha256=static_result["receipt_sha256"],
        image_digest_sha256=package.artifacts.image_digest_sha256,  # type: ignore[union-attr]
        source_tree_root_sha256=source_root,
        _issuer=kubernetes_runtime._STATIC_GATE_ISSUER,
    )
    capacity = M03RV16CapacityGateQualification(
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[union-attr]
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        static_gate_receipt_sha256=static.receipt_sha256,
        rendered_manifest_sha256="7" * 64,
        terminal_file_sha256=file_sha256(capacity_path),
        terminal_receipt_sha256=semantic_sha256(capacity_result),
        image_digest_sha256=package.artifacts.image_digest_sha256,  # type: ignore[union-attr]
        source_tree_root_sha256=source_root,
        _issuer=kubernetes_runtime._CAPACITY_GATE_ISSUER,
    )
    training = _issue_m03r_v16_training_activation_from_gates(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        static=static,
        capacity=capacity,
    )
    training_path = tmp_path / "training.json"
    training_file = write_m03r_v16_training_activation(training_path, training)
    assert (
        load_m03r_v16_training_activation(
            training_path,
            expected_file_sha256=training_file,
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
            static_result_path=static_path,
            expected_static_result_file_sha256=file_sha256(static_path),
            capacity_terminal_path=capacity_path,
            expected_capacity_terminal_file_sha256=file_sha256(capacity_path),
        )
        == training
    )
    forged_payload = json.loads(training_path.read_bytes())
    forged_payload["activation"]["static_result_file_sha256"] = "f" * 64
    forged_payload["receipt_sha256"] = semantic_sha256(
        forged_payload["activation"]
    )
    forged_path = tmp_path / "forged-training.json"
    forged_path.write_bytes(canonical_json_file_bytes(forged_payload))
    with pytest.raises(M03RV16ActivationError):
        load_m03r_v16_training_activation(
            forged_path,
            expected_file_sha256=file_sha256(forged_path),
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
            static_result_path=static_path,
            expected_static_result_file_sha256="f" * 64,
            capacity_terminal_path=capacity_path,
            expected_capacity_terminal_file_sha256=file_sha256(capacity_path),
        )
    adequacy = tuple(
        tuple(f"{10 + setting * 5 + fold:064x}" for fold in range(5))
        for setting in range(3)
    )
    checkpoints = tuple(
        tuple(f"{30 + setting * 5 + fold:064x}" for fold in range(5))
        for setting in range(3)
    )
    closure = "9" * 64
    terminal_files = ("1" * 64, "2" * 64, "3" * 64)
    panel_unsigned = {
        "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "source_tree_root_sha256": source_root,
        "outer_qualification_authorized": True,
        "setting_fold_adequacy_receipt_sha256": adequacy,
        "terminal_checkpoint_file_sha256": checkpoints,
        "prequalification_closure_receipt_sha256": closure,
    }
    panel = {**panel_unsigned, "receipt_sha256": semantic_sha256(panel_unsigned)}
    panel_path = tmp_path / "panel.json"
    panel_path.write_bytes(canonical_json_file_bytes(panel))
    terminal_paths = tuple(tmp_path / f"terminal-{index}.json" for index in range(3))
    for terminal_path in terminal_paths:
        terminal_path.write_bytes(
            canonical_json_file_bytes(
                {
                    "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "source_tree_root_sha256": source_root,
                }
            )
        )
    terminal_files = tuple(file_sha256(path) for path in terminal_paths)
    qualification = _issue_m03r_v16_qualification_activation_from_panel(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        training_panel_receipt_sha256=panel["receipt_sha256"],
        training_panel_file_sha256=file_sha256(panel_path),
        training_terminal_file_sha256=terminal_files,  # type: ignore[arg-type]
        setting_fold_training_adequacy_receipt_sha256=adequacy,  # type: ignore[arg-type]
        terminal_checkpoint_file_sha256=checkpoints,  # type: ignore[arg-type]
        prequalification_closure_receipt_sha256=closure,
        source_tree_root_sha256=source_root,
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
            training_panel_path=panel_path,
            training_terminal_paths=terminal_paths,  # type: ignore[arg-type]
        )
        == qualification
    )
    with pytest.raises(TypeError):
        M03RV16TrainingActivation(  # type: ignore[call-arg]
            package_plan_sha256=package.package_plan_sha256,  # type: ignore[union-attr]
            execution_authorization_receipt_sha256=authorization.receipt_sha256,
            static_gate_receipt_sha256="c" * 64,
            static_rendered_manifest_sha256="5" * 64,
            static_result_file_sha256="1" * 64,
            static_result_receipt_sha256="2" * 64,
            capacity_gate_receipt_sha256="d" * 64,
            capacity_rendered_manifest_sha256="6" * 64,
            capacity_terminal_file_sha256="3" * 64,
            capacity_terminal_receipt_sha256="4" * 64,
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
            training_panel_path=panel_path,
            training_terminal_paths=terminal_paths,  # type: ignore[arg-type]
        )
