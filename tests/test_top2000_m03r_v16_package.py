from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03R_V16_RUNTIME_ENTRYPOINT,
    M03RV16ExecutionAuthorization,
    M03RV16PackageArtifacts,
    build_m03r_v16_package_plan,
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
    write_m03r_v16_execution_authorization,
    write_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_worker import (
    M03RV16PredictiveWorkerError,
)


def _artifacts() -> M03RV16PackageArtifacts:
    image_digest = "f" * 64
    return M03RV16PackageArtifacts(
        source_archive_sha256="0" * 64,
        source_manifest_sha256="1" * 64,
        dependency_lock_sha256="2" * 64,
        cache_artifact_sha256="3" * 64,
        cache_manifest_sha256="4" * 64,
        asset_axis_sha256="5" * 64,
        risk_artifact_sha256="6" * 64,
        risk_source_manifest_file_sha256="7" * 64,
        risk_source_receipt_sha256="8" * 64,
        exposure_receipt_sha256="9" * 64,
        projector_manifest_file_sha256="a" * 64,
        projector_manifest_sha256="b" * 64,
        projector_binding_sha256="c" * 64,
        worker_source_sha256="d" * 64,
        operator_source_sha256="e" * 64,
        initial_parameter_state_file_sha256="f" * 64,
        initial_parameter_state_sha256="0" * 64,
        initial_parameter_architecture_sha256="1" * 64,
        structural_slab_file_sha256="2" * 64,
        structural_slab_receipt_sha256="3" * 64,
        structural_action_operator_root_sha256="4" * 64,
        structural_target_operator_root_sha256="5" * 64,
        structural_target_root_sha256=("6" * 64, "7" * 64, "8" * 64),
        image_reference=f"registry/research@sha256:{image_digest}",
        image_digest_sha256=image_digest,
    )


def _schedule() -> M03RV16PanelSchedule:
    return M03RV16PanelSchedule(
        protocol_common_data_sha256="9" * 64,
        cache_sha256="3" * 64,
        asset_axis_sha256="5" * 64,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v16_fold_geometries(1001)
        ),
    )


def test_v16_package_plan_binds_three_workers_and_structural_slab(
    tmp_path: Path,
) -> None:
    plan = build_m03r_v16_package_plan(_artifacts(), _schedule())
    assert plan.panel.maximum_h100_requests == 6
    assert plan.panel.primary_setting_index == 2
    assert plan.panel.workers[0].structural_slab_receipt_sha256 == "3" * 64
    assert plan.panel.workers[2].structural_target_root_sha256 == (
        "6" * 64,
        "7" * 64,
        "8" * 64,
    )
    path = tmp_path / "package-plan.json"
    file_sha = write_m03r_v16_package_plan(path, plan)
    assert load_m03r_v16_package_plan(path, expected_file_sha256=file_sha) == plan


def test_v16_package_plan_rejects_cross_setting_slab_drift() -> None:
    plan = build_m03r_v16_package_plan(_artifacts(), _schedule())
    workers = list(plan.panel.workers)
    workers[1] = replace(workers[1], structural_slab_receipt_sha256="f" * 64)
    with pytest.raises(M03RV16PredictiveWorkerError, match="panel"):
        replace(plan, panel=replace(plan.panel, workers=tuple(workers))).validate()


def test_v16_execution_authorization_is_predictive_only(tmp_path: Path) -> None:
    plan = build_m03r_v16_package_plan(_artifacts(), _schedule())
    plan_path = tmp_path / "package-plan.json"
    plan_file_sha = write_m03r_v16_package_plan(plan_path, plan)
    authorization = M03RV16ExecutionAuthorization(
        package_plan_sha256=plan.package_plan_sha256,
        package_plan_file_sha256=plan_file_sha,
        source_archive_sha256=plan.artifacts.source_archive_sha256,
        source_manifest_sha256=plan.artifacts.source_manifest_sha256,
        worker_source_sha256=plan.artifacts.worker_source_sha256,
        structural_slab_file_sha256=plan.artifacts.structural_slab_file_sha256,
        structural_slab_receipt_sha256=(
            plan.artifacts.structural_slab_receipt_sha256
        ),
        image_reference=plan.artifacts.image_reference,
    )
    path = tmp_path / "execution-authorization.json"
    file_sha = write_m03r_v16_execution_authorization(path, authorization, plan)
    loaded = load_m03r_v16_execution_authorization(
        path,
        expected_file_sha256=file_sha,
        package=plan,
    )
    assert loaded.runtime_entrypoint == M03R_V16_RUNTIME_ENTRYPOINT
    assert loaded.maximum_h100_requests == 6
    assert loaded.predictive_training_authorized is True
    assert loaded.economic_training_authorized is False
    assert loaded.reinforcement_learning_authorized is False
    assert loaded.outer_2026_access_authorized is False


def test_v16_execution_authorization_rejects_broader_access() -> None:
    plan = build_m03r_v16_package_plan(_artifacts(), _schedule())
    authorization = M03RV16ExecutionAuthorization(
        package_plan_sha256=plan.package_plan_sha256,
        package_plan_file_sha256="0" * 64,
        source_archive_sha256=plan.artifacts.source_archive_sha256,
        source_manifest_sha256=plan.artifacts.source_manifest_sha256,
        worker_source_sha256=plan.artifacts.worker_source_sha256,
        structural_slab_file_sha256=plan.artifacts.structural_slab_file_sha256,
        structural_slab_receipt_sha256=(
            plan.artifacts.structural_slab_receipt_sha256
        ),
        image_reference=plan.artifacts.image_reference,
    )
    with pytest.raises(ValueError, match="authorization"):
        replace(authorization, economic_training_authorized=True).validate(plan)
    with pytest.raises(ValueError, match="authorization"):
        replace(authorization, reinforcement_learning_authorized=True).validate(plan)
    with pytest.raises(ValueError, match="authorization"):
        replace(authorization, outer_2026_access_authorized=True).validate(plan)
