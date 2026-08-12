from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rl_quant.training.top2000_m03r_v11_package import (
    load_m03r_v11_execution_authorization,
    load_m03r_v11_package_plan,
)
from rl_quant.workflows.top2000_m03r_v11_package_builder import (
    M03RV11PackageBuildError,
    build_m03r_v11_local_package,
    build_m03r_v11_transfer_archive,
    validate_m03r_v11_local_package,
    validate_m03r_v11_transfer_archive,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source"
    worker = source / "src/rl_quant/workflows/top2000_m03r_v11_predictive.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("VALUE = 11\n")
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (source / "uv.lock").write_text("fixture-lock\n")
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"cache")
    cache_manifest = tmp_path / "cache-manifest.json"
    cache_manifest.write_text('{"schema":"cache"}\n')
    risk = tmp_path / "risk"
    risk.mkdir()
    (risk / "risk-exposures.pt").write_bytes(b"risk")
    (risk / "risk-source-manifest.json").write_text('{"schema":"risk"}\n')
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


def test_v11_builder_mints_separate_predictive_authorization_and_safe_archive(
    tmp_path: Path,
) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    receipt = build_m03r_v11_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    assert (
        validate_m03r_v11_local_package(output)["receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    package = load_m03r_v11_package_plan(
        output / "package/plans/package-plan.json",
        expected_file_sha256=receipt["package_plan_file_sha256"],
    )
    authorization = load_m03r_v11_execution_authorization(
        output / "package/plans/execution-authorization.json",
        expected_file_sha256=receipt["execution_authorization_file_sha256"],
        package=package,
    )
    assert not package.kubernetes_launch_authorized
    assert authorization.predictive_training_authorized
    assert authorization.maximum_h100_requests == 6
    assert not authorization.economic_training_authorized
    assert not authorization.outer_2026_access_authorized
    initial_state = output / "package/model/common-initial-parameter-state.pt"
    assert initial_state.is_file()
    assert (
        hashlib.sha256(initial_state.read_bytes()).hexdigest()
        == package.artifacts.initial_parameter_state_file_sha256
    )

    first = build_m03r_v11_transfer_archive(output, tmp_path / "first.tar")
    second = build_m03r_v11_transfer_archive(output, tmp_path / "second.tar")
    assert first["archive_sha256"] == second["archive_sha256"]
    validate_m03r_v11_transfer_archive(
        tmp_path / "first.tar",
        expected_archive_sha256=first["archive_sha256"],
        expected_package_receipt_file_sha256=(
            hashlib.sha256(
                (output / "package-build-receipt.json").read_bytes()
            ).hexdigest()
        ),
    )


def test_v11_builder_rejects_package_member_tamper(tmp_path: Path) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    build_m03r_v11_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    member = output / "package/cache/top2000-daily-bars.pt"
    member.chmod(0o640)
    member.write_bytes(b"tampered")
    with pytest.raises(M03RV11PackageBuildError, match="inventory"):
        validate_m03r_v11_local_package(output)
