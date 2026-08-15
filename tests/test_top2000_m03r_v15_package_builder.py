from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rl_quant.training.top2000_m03r_v15_fold import (
    render_m03r_v15_fold_geometries,
)
from rl_quant.training.top2000_m03r_v15_package import (
    load_m03r_v15_execution_authorization,
    load_m03r_v15_package_plan,
)
from rl_quant.training.top2000_m03r_v15_preflight import (
    M03RV15StructuralPreflightReceipt,
    _scheduled_origins,
    _sha256 as _preflight_sha256,
    write_m03r_v15_structural_preflight,
)
from rl_quant.workflows.top2000_m03r_v15_package_builder import (
    M03RV15PackageBuildError,
    _file_sha256,
    build_m03r_v15_local_package,
    build_m03r_v15_transfer_archive,
    validate_m03r_v15_local_package,
    validate_m03r_v15_transfer_archive,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source"
    worker = source / "src/rl_quant/workflows/top2000_m03r_v15_predictive.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("VALUE = 13\n", encoding="utf-8")
    operator = (
        source
        / "src/rl_quant/training/top2000_m03r_v15_residual_operator.py"
    )
    operator.parent.mkdir(parents=True)
    operator.write_text("OPERATOR = 15\n", encoding="utf-8")
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


def _package_preflight_receipt(**identities: str) -> M03RV15StructuralPreflightReceipt:
    origins = _scheduled_origins()
    return M03RV15StructuralPreflightReceipt(
        cache_sha256=identities["cache_sha256"],
        cache_manifest_sha256=identities["cache_manifest_sha256"],
        asset_axis_sha256="1" * 64,
        source_manifest_sha256=identities["source_manifest_sha256"],
        operator_source_sha256=identities["operator_source_sha256"],
        risk_artifact_file_sha256=identities["risk_artifact_file_sha256"],
        risk_source_manifest_file_sha256=identities[
            "risk_source_manifest_file_sha256"
        ],
        risk_source_receipt_sha256="2" * 64,
        exposure_receipt_sha256="3" * 64,
        projector_manifest_file_sha256=identities[
            "projector_manifest_file_sha256"
        ],
        projector_manifest_sha256="a" * 64,
        projector_binding_sha256="b" * 64,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v15_fold_geometries(1001)
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


def _install_package_owned_fakes(monkeypatch: Any) -> None:
    import rl_quant.workflows.top2000_m03r_v15_package_builder as builder

    def package_preflight(package_root: Path, **identities: str) -> tuple[Any, str]:
        receipt = _package_preflight_receipt(**identities)
        file_sha = write_m03r_v15_structural_preflight(
            package_root / "plans/real-data-structural-preflight.json", receipt
        )
        return receipt, file_sha

    def risk_loader(*_args: Any, **kwargs: Any) -> tuple[Any, Any]:
        manifest_sha = kwargs["expected_manifest_file_sha256"]
        risk = SimpleNamespace(
            receipt_sha256="2" * 64,
            exposures=SimpleNamespace(receipt_sha256="3" * 64),
        )
        written = SimpleNamespace(
            artifact_file_sha256=_file_sha256(
                Path(_args[0]).parent / "risk-exposures.pt"
            ),
            manifest_file_sha256=manifest_sha,
        )
        return risk, written

    def projector_loader(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        return (
            SimpleNamespace(manifest_sha256="a" * 64),
            SimpleNamespace(binding_sha256="b" * 64),
        )

    monkeypatch.setattr(builder, "_run_package_owned_structural_preflight", package_preflight)
    monkeypatch.setattr(builder, "load_top2000_m03r_v9_risk_source", risk_loader)
    monkeypatch.setattr(builder, "load_m03r_v9_projector_manifest", projector_loader)


def test_v15_builder_mints_predictive_only_package_and_deterministic_archive(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _install_package_owned_fakes(monkeypatch)
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    receipt = build_m03r_v15_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    assert validate_m03r_v15_local_package(output)["receipt_sha256"] == receipt[
        "receipt_sha256"
    ]
    package = load_m03r_v15_package_plan(
        output / "package/plans/package-plan.json",
        expected_file_sha256=receipt["package_plan_file_sha256"],
    )
    authorization = load_m03r_v15_execution_authorization(
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

    first = build_m03r_v15_transfer_archive(output, tmp_path / "first.tar")
    second = build_m03r_v15_transfer_archive(output, tmp_path / "second.tar")
    assert first["archive_sha256"] == second["archive_sha256"]
    validate_m03r_v15_transfer_archive(
        tmp_path / "first.tar",
        expected_archive_sha256=first["archive_sha256"],
        expected_package_receipt_file_sha256=hashlib.sha256(
            (output / "package-build-receipt.json").read_bytes()
        ).hexdigest(),
    )


def test_v15_builder_rejects_member_tamper(monkeypatch: Any, tmp_path: Path) -> None:
    _install_package_owned_fakes(monkeypatch)
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    build_m03r_v15_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    member = output / "package/cache/top2000-daily-bars.pt"
    member.chmod(0o640)
    member.write_bytes(b"tampered")
    with pytest.raises(M03RV15PackageBuildError, match="inventory"):
        validate_m03r_v15_local_package(output)
