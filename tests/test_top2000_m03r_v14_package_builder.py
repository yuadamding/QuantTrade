from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rl_quant.training.top2000_m03r_v14_fold import (
    render_m03r_v14_fold_geometries,
)
from rl_quant.training.top2000_m03r_v14_package import (
    load_m03r_v14_execution_authorization,
    load_m03r_v14_package_plan,
)
from rl_quant.training.top2000_m03r_v14_preflight import (
    M03RV14StructuralPreflightReceipt,
    _scheduled_origins,
    _sha256 as _preflight_sha256,
    write_m03r_v14_structural_preflight,
)
from rl_quant.workflows.top2000_m03r_v14_package_builder import (
    M03RV14PackageBuildError,
    _file_sha256,
    build_m03r_v14_local_package,
    build_m03r_v14_transfer_archive,
    validate_m03r_v14_local_package,
    validate_m03r_v14_transfer_archive,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source"
    worker = source / "src/rl_quant/workflows/top2000_m03r_v14_predictive.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("VALUE = 13\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n",
        encoding="utf-8",
    )
    (source / "uv.lock").write_text("fixture-lock\n", encoding="utf-8")
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"cache")
    cache_manifest = tmp_path / "cache-manifest.json"
    cache_manifest.write_text('{"schema":"cache"}\n', encoding="utf-8")
    risk = tmp_path / "risk"
    risk.mkdir()
    (risk / "risk-exposures.pt").write_bytes(b"risk")
    (risk / "risk-source-manifest.json").write_text(
        '{"schema":"risk"}\n',
        encoding="utf-8",
    )
    (risk / "projector-manifest.json").write_text(
        json.dumps(
            {
                "projector": {"manifest_sha256": "a" * 64},
                "binding": {"binding_sha256": "b" * 64},
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source, cache, cache_manifest, risk


def _preflight(tmp_path: Path, cache: Path) -> Path:
    origins = _scheduled_origins()
    receipt = M03RV14StructuralPreflightReceipt(
        cache_sha256=_file_sha256(cache),
        asset_axis_sha256="1" * 64,
        risk_source_receipt_sha256="2" * 64,
        exposure_receipt_sha256="3" * 64,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v14_fold_geometries(1001)
        ),
        scheduled_origin_sha256=_preflight_sha256(origins),
        scheduled_origin_count=len(origins),
        first_scheduled_origin=origins[0],
        last_scheduled_origin=origins[-1],
        minimum_target_qualified_assets=10,
        minimum_action_qualified_assets=10,
        minimum_target_residual_degrees_of_freedom=1,
        minimum_action_residual_degrees_of_freedom=1,
        target_action_mask_difference_origin_count=len(origins),
        target_operator_root_sha256="4" * 64,
        action_operator_root_sha256="5" * 64,
    )
    path = tmp_path / "real-data-structural-preflight.json"
    write_m03r_v14_structural_preflight(path, receipt)
    return path


def test_v14_builder_mints_predictive_only_package_and_deterministic_archive(
    tmp_path: Path,
) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    receipt = build_m03r_v14_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        structural_preflight_path=_preflight(tmp_path, cache),
        output_root=output,
    )
    assert validate_m03r_v14_local_package(output)["receipt_sha256"] == receipt[
        "receipt_sha256"
    ]
    package = load_m03r_v14_package_plan(
        output / "package/plans/package-plan.json",
        expected_file_sha256=receipt["package_plan_file_sha256"],
    )
    authorization = load_m03r_v14_execution_authorization(
        output / "package/plans/execution-authorization.json",
        expected_file_sha256=receipt["execution_authorization_file_sha256"],
        package=package,
    )
    assert authorization.maximum_h100_requests == 4
    assert authorization.predictive_training_authorized is True
    assert authorization.economic_training_authorized is False
    assert authorization.outer_2026_access_authorized is False
    assert package.artifacts.initial_parameter_architecture_sha256 == receipt[
        "initial_parameter_architecture_sha256"
    ]
    assert (output / "package/cache/cache-manifest.json").is_file()

    first = build_m03r_v14_transfer_archive(output, tmp_path / "first.tar")
    second = build_m03r_v14_transfer_archive(output, tmp_path / "second.tar")
    assert first["archive_sha256"] == second["archive_sha256"]
    validate_m03r_v14_transfer_archive(
        tmp_path / "first.tar",
        expected_archive_sha256=first["archive_sha256"],
        expected_package_receipt_file_sha256=hashlib.sha256(
            (output / "package-build-receipt.json").read_bytes()
        ).hexdigest(),
    )


def test_v14_builder_rejects_member_tamper(tmp_path: Path) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    build_m03r_v14_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        structural_preflight_path=_preflight(tmp_path, cache),
        output_root=output,
    )
    member = output / "package/cache/top2000-daily-bars.pt"
    member.chmod(0o640)
    member.write_bytes(b"tampered")
    with pytest.raises(M03RV14PackageBuildError, match="inventory"):
        validate_m03r_v14_local_package(output)
