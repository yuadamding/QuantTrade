from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v12_package import (
    M03R_V12_RUNTIME_ENTRYPOINT,
    M03RV12ExecutionAuthorization,
    M03RV12PackageArtifacts,
    M03RV12PackageError,
    build_m03r_v12_package_plan,
    load_m03r_v12_execution_authorization,
    load_m03r_v12_package_plan,
    write_m03r_v12_execution_authorization,
    write_m03r_v12_package_plan,
)
from rl_quant.training.top2000_m03r_v12_schedule import (
    M03RV12PanelEpisodeSchedule,
)


def _artifacts() -> M03RV12PackageArtifacts:
    return M03RV12PackageArtifacts(
        source_archive_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        dependency_lock_sha256="3" * 64,
        cache_artifact_sha256="4" * 64,
        cache_manifest_sha256="5" * 64,
        risk_artifact_sha256="6" * 64,
        risk_source_manifest_file_sha256="7" * 64,
        projector_manifest_file_sha256="8" * 64,
        projector_manifest_sha256="9" * 64,
        projector_binding_sha256="a" * 64,
        worker_source_sha256="b" * 64,
        initial_parameter_state_file_sha256="e" * 64,
        initial_parameter_state_sha256="c" * 64,
        structural_preflight_file_sha256="e" * 64,
        structural_preflight_receipt_sha256="f" * 64,
        image_reference=f"example/research@sha256:{'d' * 64}",
        image_digest_sha256="d" * 64,
    )


def _schedule() -> M03RV12PanelEpisodeSchedule:
    folds = render_top2000_m03r_v7_development_folds(1001)
    return M03RV12PanelEpisodeSchedule(
        protocol_common_data_sha256="e" * 64,
        cache_sha256="4" * 64,
        fold_geometry_sha256=tuple(
            render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
        ),
    )


def test_v12_package_round_trip_binds_common_schedule_and_stays_unlaunched(
    tmp_path: Path,
) -> None:
    plan = build_m03r_v12_package_plan(_artifacts(), _schedule())
    path = tmp_path / "v12-package.json"
    file_sha = write_m03r_v12_package_plan(path, plan)
    loaded = load_m03r_v12_package_plan(path, expected_file_sha256=file_sha)
    assert loaded.package_plan_sha256 == plan.package_plan_sha256
    assert len({row.panel_episode_schedule_sha256 for row in loaded.panel.workers}) == 1
    assert (
        len({row.initial_parameter_state_sha256 for row in loaded.panel.workers}) == 1
    )
    assert not loaded.package_authorized
    assert not loaded.kubernetes_launch_authorized
    assert not loaded.outer_2026_access_authorized


def test_v12_package_rejects_cache_or_worker_artifact_drift(tmp_path: Path) -> None:
    with pytest.raises(M03RV12PackageError, match="cache"):
        build_m03r_v12_package_plan(
            replace(_artifacts(), cache_artifact_sha256="f" * 64),
            _schedule(),
        )
    plan = build_m03r_v12_package_plan(_artifacts(), _schedule())
    workers = list(plan.panel.workers)
    workers[1] = replace(workers[1], projector_manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="panel plan"):
        replace(plan.panel, workers=tuple(workers)).validate()
    path = tmp_path / "v12-package.json"
    file_sha = write_m03r_v12_package_plan(path, plan)
    with pytest.raises(M03RV12PackageError, match="file hash"):
        load_m03r_v12_package_plan(path, expected_file_sha256="f" * 64)
    assert len(file_sha) == 64


def test_v12_separate_execution_authorization_binds_exact_unlaunched_package(
    tmp_path: Path,
) -> None:
    package = build_m03r_v12_package_plan(_artifacts(), _schedule())
    package_path = tmp_path / "package.json"
    package_file_sha = write_m03r_v12_package_plan(package_path, package)
    authorization = M03RV12ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=package_file_sha,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        source_manifest_sha256=package.artifacts.source_manifest_sha256,
        worker_source_sha256=package.artifacts.worker_source_sha256,
        image_reference=package.artifacts.image_reference,
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_file_sha = write_m03r_v12_execution_authorization(
        authorization_path,
        authorization,
        package,
    )
    loaded = load_m03r_v12_execution_authorization(
        authorization_path,
        expected_file_sha256=authorization_file_sha,
        package=package,
    )
    assert loaded.runtime_entrypoint == M03R_V12_RUNTIME_ENTRYPOINT
    assert loaded.maximum_h100_requests == 6
    assert loaded.predictive_training_authorized
    assert not loaded.economic_training_authorized
    assert not loaded.outer_2026_access_authorized
    assert not package.kubernetes_launch_authorized

    with pytest.raises(M03RV12PackageError, match="authorization"):
        replace(authorization, maximum_h100_requests=8).validate(package)
