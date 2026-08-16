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
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    M03R_V16_PREQUALIFICATION_CLOSURE_SCHEMA,
    M03R_V16_TRAINING_PANEL_SCHEMA,
    M03RV16ActivationError,
    M03RV16TrainingActivation,
    _issue_m03r_v16_qualification_activation_from_panel_authority,
    _issue_m03r_v16_training_activation_from_gates,
    load_m03r_v16_qualification_activation,
    load_m03r_v16_training_activation,
    load_m03r_v16_training_panel_authority,
    write_m03r_v16_qualification_activation,
    write_m03r_v16_training_activation,
)
from rl_quant.training.top2000_m03r_v16_capacity import (
    M03R_V16_CAPACITY_TERMINAL_SCHEMA,
    M03RV16CapacityRankEvidence,
    build_m03r_v16_capacity_terminal,
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
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03R_V16_TRAINING_TERMINAL_SCHEMA,
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


def _capacity_rank(rank: int) -> M03RV16CapacityRankEvidence:
    return M03RV16CapacityRankEvidence(
        setting_index=0,
        distributed_rank=rank,
        distributed_world_size=2,
        cuda_device_name="NVIDIA H100 80GB HBM3",
        cuda_total_memory_bytes=80 * 1024**3,
        peak_allocated_bytes=40 * 1024**3 + rank,
        peak_reserved_bytes=50 * 1024**3 + rank,
        pre_validation_update_receipt_sha256=("1" if rank == 0 else "2") * 64,
        validation_batch_receipt_sha256=("3" if rank == 0 else "4") * 64,
        update_plan_sha256="5" * 64,
        batch_receipt_sha256=("6" if rank == 0 else "7") * 64,
        score_step_receipt_sha256=("8" if rank == 0 else "9") * 64,
        structural_slab_receipt_sha256="a" * 64,
        qualification_projection_receipt_sha256=(
            ("b" if rank == 0 else "c") * 64
        ),
        qualification_requested_active_one_way_mass=0.01,
        qualification_projected_active_one_way_mass=0.0025,
        qualification_requested_to_executed_retention=0.25,
        post_update_model_state_sha256="d" * 64,
        post_update_optimizer_state_sha256="e" * 64,
        episode_state_rows=345,
        global_origin_count=43,
        local_origin_count=22 if rank == 0 else 21,
    )


def test_v16_phase_activations_round_trip_and_cannot_be_forged(
    tmp_path: Path,
) -> None:
    package, authorization = _surfaces()
    source_root = "e" * 64
    static_unsigned = {
        "schema": M03R_V16_STATIC_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
        "package_plan_file_sha256": authorization.package_plan_file_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "execution_authorization_file_sha256": "c" * 64,
        "source_archive_sha256": package.artifacts.source_archive_sha256,  # type: ignore[union-attr]
        "source_manifest_sha256": package.artifacts.source_manifest_sha256,  # type: ignore[union-attr]
        "worker_source_sha256": package.artifacts.worker_source_sha256,  # type: ignore[union-attr]
        "source_tree_root_sha256": source_root,
        "structural_slab_file_sha256": package.artifacts.structural_slab_file_sha256,  # type: ignore[union-attr]
        "structural_slab_receipt_sha256": package.artifacts.structural_slab_receipt_sha256,  # type: ignore[union-attr]
        "panel_schedule_sha256": package.schedule.receipt_sha256,  # type: ignore[union-attr]
        "hold_target_sessions": 30,
        "hold_target_spec_sha256": package.hold_target_spec_sha256,  # type: ignore[union-attr]
        "image_digest_sha256": package.artifacts.image_digest_sha256,  # type: ignore[union-attr]
        "training_performed": False,
        "gpu_mask": "none",
        "gpu_requests": 0,
        "gpu_limits": 0,
        "unmasked_visibility_claimed": False,
        "initial_state_strict_loaded_all_settings": True,
        "output_empty": True,
        "container_started": True,
        "economic_training_authorized": False,
        "reinforcement_learning_authorized": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    static_result = {
        **static_unsigned,
        "receipt_sha256": semantic_sha256(static_unsigned),
    }
    static_path = tmp_path / "static-result.json"
    static_path.write_bytes(canonical_json_file_bytes(static_result))
    typed_capacity = build_m03r_v16_capacity_terminal(
        (_capacity_rank(0), _capacity_rank(1))
    )
    capacity_result = {
        "schema": M03R_V16_CAPACITY_TERMINAL_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
        "authorization_receipt_sha256": authorization.receipt_sha256,
        "source_tree_root_sha256": source_root,
        "scientific_training_performed": False,
        "disposable_train_validate_train_executed": True,
        "disposable_optimizer_update_executed": True,
        "scientific_checkpoint_published": False,
        "capacity": __import__("dataclasses").asdict(typed_capacity),
        "capacity_receipt_sha256": typed_capacity.receipt_sha256,
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
            expected_authorization_file_sha256="c" * 64,
            static_result_path=static_path,
            expected_static_result_file_sha256=file_sha256(static_path),
            capacity_terminal_path=capacity_path,
            expected_capacity_terminal_file_sha256=file_sha256(capacity_path),
        )
        == training
    )
    synthetic_capacity = {
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
    synthetic_capacity_path = tmp_path / "synthetic-capacity.json"
    synthetic_capacity_path.write_bytes(
        canonical_json_file_bytes(synthetic_capacity)
    )
    synthetic_activation = json.loads(training_path.read_bytes())
    synthetic_activation["activation"]["capacity_terminal_file_sha256"] = (
        file_sha256(synthetic_capacity_path)
    )
    synthetic_activation["activation"]["capacity_terminal_receipt_sha256"] = (
        semantic_sha256(synthetic_capacity)
    )
    synthetic_activation["receipt_sha256"] = semantic_sha256(
        synthetic_activation["activation"]
    )
    synthetic_activation_path = tmp_path / "synthetic-training.json"
    synthetic_activation_path.write_bytes(
        canonical_json_file_bytes(synthetic_activation)
    )
    with pytest.raises(M03RV16ActivationError, match="capacity terminal"):
        load_m03r_v16_training_activation(
            synthetic_activation_path,
            expected_file_sha256=file_sha256(synthetic_activation_path),
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
            expected_authorization_file_sha256="c" * 64,
            static_result_path=static_path,
            expected_static_result_file_sha256=file_sha256(static_path),
            capacity_terminal_path=synthetic_capacity_path,
            expected_capacity_terminal_file_sha256=file_sha256(
                synthetic_capacity_path
            ),
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
            expected_authorization_file_sha256="c" * 64,
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
    terminal_files = ("1" * 64, "2" * 64, "3" * 64)
    terminal_paths = tuple(tmp_path / f"terminal-{index}.json" for index in range(3))
    terminal_receipts: list[str] = []
    for setting, terminal_path in enumerate(terminal_paths):
        terminal_unsigned = {
            "schema": M03R_V16_TRAINING_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "setting_index": setting,
            "source_tree_root_sha256": source_root,
            "fold_training_adequacy_status": ("adequate",) * 5,
            "qualification_tail_accessed": False,
            "outer_qualification_authorized": False,
            "three_seed_confirmation_may_be_minted": False,
        }
        terminal = {
            **terminal_unsigned,
            "receipt_sha256": semantic_sha256(terminal_unsigned),
        }
        terminal_receipts.append(terminal["receipt_sha256"])
        terminal_path.write_bytes(canonical_json_file_bytes(terminal))
    terminal_files = tuple(file_sha256(path) for path in terminal_paths)
    closure_unsigned = {
        "schema": M03R_V16_PREQUALIFICATION_CLOSURE_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
        "training_terminal_file_sha256": terminal_files,
        "terminal_checkpoint_file_sha256": checkpoints,
        "all_setting_folds_adequate": True,
        "outer_qualification_outcomes_accessed": False,
    }
    closure_payload = {
        **closure_unsigned,
        "receipt_sha256": semantic_sha256(closure_unsigned),
    }
    closure_path = tmp_path / "prequalification-closure.json"
    closure_path.write_bytes(canonical_json_file_bytes(closure_payload))
    closure = closure_payload["receipt_sha256"]
    panel_unsigned = {
        "schema": M03R_V16_TRAINING_PANEL_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "training_terminal_file_sha256": terminal_files,
        "training_terminal_receipt_sha256": tuple(terminal_receipts),
        "source_tree_root_sha256": source_root,
        "outer_qualification_authorized": True,
        "setting_fold_adequacy_receipt_sha256": adequacy,
        "setting_fold_adequacy_status": (("adequate",) * 5,) * 3,
        "terminal_checkpoint_file_sha256": checkpoints,
        "prequalification_closure_receipt_sha256": closure,
        "prequalification_closure_file_sha256": file_sha256(closure_path),
        "all_setting_folds_adequate": True,
        "outer_qualification_outcomes_accessed": False,
        "next_research_action": "qualification-only-execution",
        "economic_generation_may_be_minted": False,
        "reinforcement_learning_authorized": False,
        "outer_2026_accessed": False,
    }
    panel = {**panel_unsigned, "receipt_sha256": semantic_sha256(panel_unsigned)}
    panel_path = tmp_path / "panel.json"
    panel_path.write_bytes(canonical_json_file_bytes(panel))
    minimal_panel = {
        "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "outer_qualification_authorized": True,
    }
    minimal_panel_path = tmp_path / "minimal-panel.json"
    minimal_panel_path.write_bytes(canonical_json_file_bytes(minimal_panel))
    with pytest.raises(M03RV16ActivationError, match="malformed"):
        load_m03r_v16_training_panel_authority(
            training_panel_path=minimal_panel_path,
            expected_training_panel_file_sha256=file_sha256(minimal_panel_path),
            prequalification_closure_path=closure_path,
            expected_prequalification_closure_file_sha256=file_sha256(
                closure_path
            ),
            training_terminal_paths=terminal_paths,  # type: ignore[arg-type]
            expected_training_terminal_file_sha256=terminal_files,  # type: ignore[arg-type]
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
        )
    panel_authority = load_m03r_v16_training_panel_authority(
        training_panel_path=panel_path,
        expected_training_panel_file_sha256=file_sha256(panel_path),
        prequalification_closure_path=closure_path,
        expected_prequalification_closure_file_sha256=file_sha256(closure_path),
        training_terminal_paths=terminal_paths,  # type: ignore[arg-type]
        expected_training_terminal_file_sha256=terminal_files,  # type: ignore[arg-type]
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
    )
    qualification = _issue_m03r_v16_qualification_activation_from_panel_authority(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        panel=panel_authority,
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
            prequalification_closure_path=closure_path,
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
            prequalification_closure_path=closure_path,
            training_terminal_paths=terminal_paths,  # type: ignore[arg-type]
        )
