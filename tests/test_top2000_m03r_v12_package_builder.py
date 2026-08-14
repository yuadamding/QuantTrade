from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from rl_quant.training.top2000_m03r_v12_package import (
    load_m03r_v12_execution_authorization,
    load_m03r_v12_package_plan,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import M03R_V12_HORIZONS
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v12_schedule import M03RV12PanelEpisodeSchedule
from rl_quant.training.top2000_m03r_v12_structural_preflight import (
    M03RV12StructuralPreflightReceipt,
    _sha256 as _preflight_sha256,
)
from rl_quant.workflows.top2000_m03r_v12_package_builder import (
    M03RV12PackageBuildError,
    build_m03r_v12_local_package,
    build_m03r_v12_transfer_archive,
    validate_m03r_v12_local_package,
    validate_m03r_v12_transfer_archive,
    _file_sha256,
    _sha256,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source"
    worker = source / "src/rl_quant/workflows/top2000_m03r_v12_predictive.py"
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


def _preflight(tmp_path: Path, cache: Path, risk: Path) -> Path:
    folds = render_top2000_m03r_v7_development_folds(1001)
    schedule = M03RV12PanelEpisodeSchedule(
        protocol_common_data_sha256=_sha256(
            {
                "cache": _file_sha256(cache),
                "risk": _file_sha256(risk / "risk-exposures.pt"),
                "risk_manifest": _file_sha256(risk / "risk-source-manifest.json"),
                "projector": _file_sha256(risk / "projector-manifest.json"),
            }
        ),
        cache_sha256=_file_sha256(cache),
        fold_geometry_sha256=tuple(
            render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
        ),
    )
    provisional = M03RV12StructuralPreflightReceipt(
        panel_episode_schedule_sha256=schedule.receipt_sha256,
        cache_file_sha256=_file_sha256(cache),
        risk_artifact_file_sha256=_file_sha256(risk / "risk-exposures.pt"),
        risk_manifest_file_sha256=_file_sha256(risk / "risk-source-manifest.json"),
        asset_axis_sha256="a" * 64,
        exposure_receipt_sha256="b" * 64,
        scheduled_fold_index=0,
        scheduled_origin_count=1,
        minimum_origin_state_index=20,
        maximum_origin_state_index=20,
        maximum_target_state_index=83,
        first_origin_exchange_date="2022-02-01",
        last_target_exchange_date="2022-05-03",
        horizons=M03R_V12_HORIZONS,
        operator_count=len(M03R_V12_HORIZONS),
        minimum_factor_qualified_fraction=1.0,
        minimum_effective_design_rank=1,
        maximum_effective_design_rank=1,
        minimum_weighted_residual_degrees_of_freedom=1,
        operator_inventory_sha256="c" * 64,
        receipt_sha256="0" * 64,
    )
    receipt = replace(
        provisional,
        receipt_sha256=_preflight_sha256(provisional.unsigned_payload()),
    )
    path = tmp_path / "real-data-structural-preflight.json"
    path.write_text(
        json.dumps(asdict(receipt), separators=(",", ":"), sort_keys=True) + "\n"
    )
    return path


def test_v12_builder_mints_separate_predictive_authorization_and_safe_archive(
    tmp_path: Path,
) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    preflight = _preflight(tmp_path, cache, risk)
    receipt = build_m03r_v12_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        structural_preflight_path=preflight,
        output_root=output,
    )
    assert (
        validate_m03r_v12_local_package(output)["receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    package = load_m03r_v12_package_plan(
        output / "package/plans/package-plan.json",
        expected_file_sha256=receipt["package_plan_file_sha256"],
    )
    authorization = load_m03r_v12_execution_authorization(
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

    first = build_m03r_v12_transfer_archive(output, tmp_path / "first.tar")
    second = build_m03r_v12_transfer_archive(output, tmp_path / "second.tar")
    assert first["archive_sha256"] == second["archive_sha256"]
    validate_m03r_v12_transfer_archive(
        tmp_path / "first.tar",
        expected_archive_sha256=first["archive_sha256"],
        expected_package_receipt_file_sha256=(
            hashlib.sha256(
                (output / "package-build-receipt.json").read_bytes()
            ).hexdigest()
        ),
    )


def test_v12_builder_rejects_package_member_tamper(tmp_path: Path) -> None:
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    preflight = _preflight(tmp_path, cache, risk)
    build_m03r_v12_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        structural_preflight_path=preflight,
        output_root=output,
    )
    member = output / "package/cache/top2000-daily-bars.pt"
    member.chmod(0o640)
    member.write_bytes(b"tampered")
    with pytest.raises(M03RV12PackageBuildError, match="inventory"):
        validate_m03r_v12_local_package(output)
