from __future__ import annotations

import json
import shutil

import pytest

from rl_quant.training.top2000_m03r_v9_package import load_m03r_v9_package_plan
from rl_quant.workflows.top2000_m03r_v9_package_builder import (
    M03RV9PackageBuildError,
    build_m03r_v9_local_package,
    build_m03r_v9_transfer_archive,
    validate_m03r_v9_local_package,
    validate_m03r_v9_transfer_archive,
)


def _inputs(tmp_path):
    source = tmp_path / "source-root"
    worker = source / "src/rl_quant/workflows/top2000_m03r_v9_predictive.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("VALUE = 1\n")
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (source / "uv.lock").write_text("fixture-lock\n")
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"cache-fixture")
    cache_manifest = tmp_path / "cache-manifest.json"
    cache_manifest.write_text('{"schema":"cache-fixture"}\n')
    risk = tmp_path / "risk"
    risk.mkdir()
    (risk / "risk-exposures.pt").write_bytes(b"risk-fixture")
    (risk / "risk-source-manifest.json").write_text('{"schema":"risk-fixture"}\n')
    (risk / "projector-manifest.json").write_text(
        json.dumps(
            {
                "projector": {"manifest_sha256": "a" * 64},
                "binding": {"binding_sha256": "b" * 64},
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return source, cache, cache_manifest, risk


def test_package_builder_is_no_clobber_and_round_trip_validated(tmp_path) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    receipt = build_m03r_v9_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    validated = validate_m03r_v9_local_package(output)
    assert validated["receipt_sha256"] == receipt["receipt_sha256"]
    package = load_m03r_v9_package_plan(
        output / "package/plans/package-plan.json",
        expected_package_plan_sha256=receipt["package_plan_sha256"],
    )
    assert package.panel.maximum_h100_requests == 6
    assert not package.economic_panel_authorized
    relocated = tmp_path / "relocated-stage"
    shutil.copytree(output, relocated)
    assert (
        validate_m03r_v9_local_package(relocated)["receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    with pytest.raises(FileExistsError):
        build_m03r_v9_local_package(
            source_root=source,
            cache_path=cache,
            cache_manifest_path=cache_manifest,
            risk_root=risk,
            output_root=output,
        )


def test_package_validator_rejects_member_tamper(tmp_path) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    build_m03r_v9_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    cache_member = output / "package/cache/top2000-daily-bars.pt"
    cache_member.chmod(0o640)
    cache_member.write_bytes(b"tampered")
    with pytest.raises(M03RV9PackageBuildError, match="inventory"):
        validate_m03r_v9_local_package(output)


def test_transfer_archive_is_deterministic_safe_and_receipt_bound(tmp_path) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    build_m03r_v9_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    first = build_m03r_v9_transfer_archive(output, tmp_path / "first.tar")
    second = build_m03r_v9_transfer_archive(output, tmp_path / "second.tar")
    assert first["archive_sha256"] == second["archive_sha256"]
    validated = validate_m03r_v9_transfer_archive(
        tmp_path / "first.tar",
        expected_archive_sha256=first["archive_sha256"],
        expected_package_build_receipt_file_sha256=(
            first["package_build_receipt_file_sha256"]
        ),
    )
    assert validated["receipt_sha256"] == first["package_build_receipt_sha256"]
    with pytest.raises(M03RV9PackageBuildError, match="SHA-256"):
        validate_m03r_v9_transfer_archive(
            tmp_path / "first.tar",
            expected_archive_sha256="0" * 64,
            expected_package_build_receipt_file_sha256=(
                first["package_build_receipt_file_sha256"]
            ),
        )
