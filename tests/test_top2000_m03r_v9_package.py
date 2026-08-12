from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rl_quant.training.top2000_m03r_v9_package import (
    M03RV9PackageArtifacts,
    M03RV9PackageError,
    build_m03r_v9_package_plan,
    load_m03r_v9_package_plan,
    package_plan_file_payload,
)


def _artifacts() -> M03RV9PackageArtifacts:
    image = "f" * 64
    return M03RV9PackageArtifacts(
        source_archive_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        dependency_lock_sha256="c" * 64,
        cache_artifact_sha256="d" * 64,
        cache_manifest_sha256="e" * 64,
        risk_artifact_sha256="1" * 64,
        risk_source_manifest_file_sha256="2" * 64,
        projector_manifest_file_sha256="3" * 64,
        projector_manifest_sha256="4" * 64,
        projector_binding_sha256="5" * 64,
        worker_source_sha256="6" * 64,
        image_reference=f"registry/research@sha256:{image}",
        image_digest_sha256=image,
    )


def test_package_has_only_three_predictive_rows_and_disjoint_outputs() -> None:
    package = build_m03r_v9_package_plan(artifacts=_artifacts())
    assert tuple(worker.setting_index for worker in package.panel.workers) == (0, 1, 2)
    assert len({worker.output_root for worker in package.panel.workers}) == 3
    assert package.panel.maximum_h100_requests == 6
    assert not package.economic_panel_authorized
    assert all(
        worker.economic_optimizer_updates == 0 for worker in package.panel.workers
    )


def test_package_round_trip_and_wrong_artifact_binding_rejects(tmp_path) -> None:
    package = build_m03r_v9_package_plan(artifacts=_artifacts())
    path = tmp_path / "package-plan.json"
    path.write_text(
        json.dumps(
            package_plan_file_payload(package),
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    loaded = load_m03r_v9_package_plan(
        path,
        expected_package_plan_sha256=package.package_plan_sha256,
    )
    assert loaded == package

    payload = package_plan_file_payload(package)
    payload["package"]["panel"]["workers"][0]["projector_binding_sha256"] = "9" * 64
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    with pytest.raises(M03RV9PackageError, match="bind package artifacts"):
        load_m03r_v9_package_plan(
            path,
            expected_package_plan_sha256=package.package_plan_sha256,
        )


def test_image_and_economic_authorization_drift_fail_closed() -> None:
    with pytest.raises(M03RV9PackageError, match="digest pinned"):
        replace(_artifacts(), image_reference="registry/research:latest").validate()
    package = build_m03r_v9_package_plan(artifacts=_artifacts())
    with pytest.raises(M03RV9PackageError, match="identity drifted"):
        replace(package, economic_panel_authorized=True).validate()
