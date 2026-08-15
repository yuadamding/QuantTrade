from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rl_quant.training.top2000_m03r_v15_fold import (
    M03RV15PanelEpisodeSchedule,
    render_m03r_v15_fold_geometries,
)
from rl_quant.training.top2000_m03r_v15_package import (
    M03R_V15_RUNTIME_ENTRYPOINT,
    M03RV15ExecutionAuthorization,
    M03RV15PackageArtifacts,
    build_m03r_v15_package_plan,
    load_m03r_v15_execution_authorization,
    load_m03r_v15_package_plan,
    write_m03r_v15_execution_authorization,
    write_m03r_v15_package_plan,
)
from rl_quant.training.top2000_m03r_v15_predictive_worker import (
    M03RV15PredictiveWorkerError,
)


def _artifacts() -> M03RV15PackageArtifacts:
    image_digest = "f" * 64
    return M03RV15PackageArtifacts(
        source_archive_sha256="0" * 64,
        source_manifest_sha256="1" * 64,
        dependency_lock_sha256="2" * 64,
        cache_artifact_sha256="3" * 64,
        cache_manifest_sha256="4" * 64,
        asset_axis_sha256="2" * 64,
        risk_artifact_sha256="5" * 64,
        risk_source_manifest_file_sha256="6" * 64,
        risk_source_receipt_sha256="e" * 64,
        exposure_receipt_sha256="f" * 64,
        projector_manifest_file_sha256="7" * 64,
        projector_manifest_sha256="8" * 64,
        projector_binding_sha256="9" * 64,
        worker_source_sha256="a" * 64,
        operator_source_sha256="1" * 64,
        initial_parameter_state_file_sha256="b" * 64,
        initial_parameter_state_sha256="c" * 64,
        initial_parameter_architecture_sha256="d" * 64,
        structural_preflight_file_sha256="e" * 64,
        structural_preflight_receipt_sha256="0" * 64,
        image_reference=f"registry/research@sha256:{image_digest}",
        image_digest_sha256=image_digest,
    )


def _schedule() -> M03RV15PanelEpisodeSchedule:
    return M03RV15PanelEpisodeSchedule(
        protocol_common_data_sha256="1" * 64,
        cache_sha256="3" * 64,
        asset_axis_sha256="2" * 64,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v15_fold_geometries(1001)
        ),
    )


def test_v15_package_plan_binds_two_workers_and_round_trips(tmp_path: Path) -> None:
    plan = build_m03r_v15_package_plan(_artifacts(), _schedule())
    assert plan.panel.maximum_h100_requests == 4
    assert plan.panel.workers[0].initial_parameter_architecture_sha256 == "d" * 64
    assert plan.package_authorized is False
    path = tmp_path / "package-plan.json"
    file_sha = write_m03r_v15_package_plan(path, plan)
    assert load_m03r_v15_package_plan(path, expected_file_sha256=file_sha) == plan


def test_v15_package_plan_rejects_worker_artifact_drift() -> None:
    plan = build_m03r_v15_package_plan(_artifacts(), _schedule())
    workers = list(plan.panel.workers)
    workers[1] = replace(workers[1], projector_manifest_sha256="f" * 64)
    with pytest.raises(M03RV15PredictiveWorkerError, match="panel"):
        replace(plan, panel=replace(plan.panel, workers=tuple(workers))).validate()


def test_v15_execution_authorization_is_exact_and_predictive_only(
    tmp_path: Path,
) -> None:
    plan = build_m03r_v15_package_plan(_artifacts(), _schedule())
    plan_path = tmp_path / "package-plan.json"
    plan_file_sha = write_m03r_v15_package_plan(plan_path, plan)
    authorization = M03RV15ExecutionAuthorization(
        package_plan_sha256=plan.package_plan_sha256,
        package_plan_file_sha256=plan_file_sha,
        source_archive_sha256=plan.artifacts.source_archive_sha256,
        source_manifest_sha256=plan.artifacts.source_manifest_sha256,
        worker_source_sha256=plan.artifacts.worker_source_sha256,
        image_reference=plan.artifacts.image_reference,
    )
    path = tmp_path / "execution-authorization.json"
    file_sha = write_m03r_v15_execution_authorization(path, authorization, plan)
    loaded = load_m03r_v15_execution_authorization(
        path,
        expected_file_sha256=file_sha,
        package=plan,
    )
    assert loaded == authorization
    assert loaded.runtime_entrypoint == M03R_V15_RUNTIME_ENTRYPOINT
    assert loaded.maximum_h100_requests == 4
    assert loaded.predictive_training_authorized is True
    assert loaded.economic_training_authorized is False
    assert loaded.outer_2026_access_authorized is False


def test_v15_execution_authorization_rejects_economic_or_2026_access(
    tmp_path: Path,
) -> None:
    plan = build_m03r_v15_package_plan(_artifacts(), _schedule())
    plan_path = tmp_path / "package-plan.json"
    plan_file_sha = write_m03r_v15_package_plan(plan_path, plan)
    authorization = M03RV15ExecutionAuthorization(
        package_plan_sha256=plan.package_plan_sha256,
        package_plan_file_sha256=plan_file_sha,
        source_archive_sha256=plan.artifacts.source_archive_sha256,
        source_manifest_sha256=plan.artifacts.source_manifest_sha256,
        worker_source_sha256=plan.artifacts.worker_source_sha256,
        image_reference=plan.artifacts.image_reference,
    )
    with pytest.raises(ValueError, match="authorization"):
        replace(authorization, economic_training_authorized=True).validate(plan)
    with pytest.raises(ValueError, match="authorization"):
        replace(authorization, outer_2026_access_authorized=True).validate(plan)
