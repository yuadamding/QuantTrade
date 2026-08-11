from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rl_quant.training.top2000_m03r_v8_pretraining_package import (
    M03RV8PretrainingArtifactBindings,
    M03RV8PretrainingPackageError,
    build_m03r_v8_pretraining_package_plan,
    load_m03r_v8_pretraining_package_plan,
    package_plan_file_payload,
)


def _artifacts() -> M03RV8PretrainingArtifactBindings:
    image = "f" * 64
    return M03RV8PretrainingArtifactBindings(
        source_archive_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        dependency_lock_sha256="c" * 64,
        cache_artifact_sha256="d" * 64,
        cache_manifest_sha256="e" * 64,
        worker_source_sha256="1" * 64,
        image_reference=f"registry/research@sha256:{image}",
        image_digest_sha256=image,
    )


def test_package_contains_exact_seven_pretraining_rows_and_disjoint_outputs() -> None:
    package = build_m03r_v8_pretraining_package_plan(artifacts=_artifacts())
    assert [plan.setting_index for plan in package.plans] == [0, 2, 3, 4, 5, 6, 7]
    assert len({plan.output_root for plan in package.plans}) == 7
    assert all(plan.alpha_pretraining_required for plan in package.plans)
    assert package.package_plan_sha256


def test_package_file_round_trip_and_tamper_rejection(tmp_path) -> None:
    package = build_m03r_v8_pretraining_package_plan(artifacts=_artifacts())
    path = tmp_path / "package-plan.json"
    path.write_text(
        json.dumps(
            package_plan_file_payload(package),
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    loaded = load_m03r_v8_pretraining_package_plan(
        path,
        expected_package_plan_sha256=package.package_plan_sha256,
    )
    assert loaded == package

    payload = package_plan_file_payload(package)
    payload["package"]["plans"][0]["output_root"] = "/mnt/output/collision"
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    with pytest.raises(M03RV8PretrainingPackageError, match="output identity"):
        load_m03r_v8_pretraining_package_plan(
            path,
            expected_package_plan_sha256=package.package_plan_sha256,
        )


def test_image_and_package_hash_drift_fail_closed() -> None:
    artifacts = _artifacts()
    with pytest.raises(M03RV8PretrainingPackageError, match="digest pinned"):
        replace(artifacts, image_reference="registry/research:latest").validate()
    package = build_m03r_v8_pretraining_package_plan(artifacts=artifacts)
    with pytest.raises(M03RV8PretrainingPackageError, match="identity drifted"):
        replace(package, package_plan_sha256="0" * 64).validate()
