from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rl_quant.training.top2000_m03r_v16_package import (
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)
from rl_quant.workflows.top2000_m03r_v16_package_builder import (
    M03RV16PackageBuildError,
    build_m03r_v16_local_package,
    build_m03r_v16_transfer_archive,
    validate_m03r_v16_local_package,
    validate_m03r_v16_transfer_archive,
    _freeze_source_tree,
    _isolated_subprocess_environment,
)


class _Receipt(SimpleNamespace):
    def validate_for_package(self, **expected: str) -> None:
        names = (
            "cache_sha256",
            "cache_manifest_sha256",
            "asset_axis_sha256",
            "source_manifest_sha256",
            "operator_source_sha256",
            "risk_artifact_file_sha256",
            "risk_source_manifest_file_sha256",
            "risk_source_receipt_sha256",
            "exposure_receipt_sha256",
            "projector_manifest_file_sha256",
            "projector_manifest_sha256",
            "projector_binding_sha256",
        )
        assert set(expected) == set(names)
        assert all(getattr(self, name) == expected[name] for name in names)


def test_v16_isolated_read_only_source_import_creates_no_bytecode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    module = source / "mini/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 17\n", encoding="utf-8")
    _freeze_source_tree(source)
    subprocess.run(
        (
            sys.executable,
            "-I",
            "-B",
            "-c",
            (
                "import sys;"
                "sys.dont_write_bytecode=True;"
                "sys.path.insert(0,sys.argv[1]);"
                "import mini;"
                "assert mini.VALUE == 17"
            ),
            str(source),
        ),
        check=True,
    )
    assert not tuple(source.rglob("__pycache__"))
    assert not tuple(source.rglob("*.pyc"))


def test_v16_isolated_builder_preserves_only_absolute_library_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_path = os.pathsep.join(("/risapps/python/lib", "/opt/runtime/lib"))
    monkeypatch.setenv("LD_LIBRARY_PATH", library_path)
    environment = _isolated_subprocess_environment(deterministic_seed=17)
    assert environment["LD_LIBRARY_PATH"] == library_path
    assert environment["PYTHONHASHSEED"] == "17"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    monkeypatch.setenv("LD_LIBRARY_PATH", ".:/opt/runtime/lib")
    with pytest.raises(M03RV16PackageBuildError, match="absolute entries"):
        _isolated_subprocess_environment()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source"
    files = {
        "src/rl_quant/workflows/top2000_m03r_v16_predictive.py": "VALUE = 16\n",
        "src/rl_quant/workflows/top2000_m03r_v16_structural_build.py": (
            "VALUE = 17\n"
        ),
        "src/rl_quant/workflows/top2000_m03r_v16_initial_state_build.py": (
            "VALUE = 18\n"
        ),
        "src/rl_quant/training/top2000_m03r_v16_policy.py": "POLICY = 16\n",
        "src/rl_quant/training/top2000_m03r_v15_residual_operator.py": (
            "OPERATOR = 15\n"
        ),
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
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
        '{"schema":"risk"}\n', encoding="utf-8"
    )
    (risk / "projector-manifest.json").write_text(
        '{"schema":"projector"}\n', encoding="utf-8"
    )
    return source, cache, cache_manifest, risk


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    import rl_quant.workflows.top2000_m03r_v16_package_builder as builder

    holder: dict[str, Any] = {}

    def initial_state(package_root: Path) -> dict[str, Any]:
        path = package_root / "model/common-initial-parameter-state.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"initial-state")
        unsigned = {
            "schema": (
                "rl-quant.top2000-dev.m03r-v16-package-owned-initial-state-v1"
            ),
            "protocol_sha256": builder.M03R_V16_PROTOCOL_SHA256,
            "initial_parameter_state_file_sha256": builder._file_sha256(path),
            "initial_parameter_state_sha256": "1" * 64,
            "initial_parameter_architecture_sha256": "2" * 64,
            "policy_source_sha256": builder._file_sha256(
                package_root / "source" / builder.M03R_V16_POLICY_SOURCE
            ),
            "builder_source_sha256": builder._file_sha256(
                package_root
                / "source"
                / builder.M03R_V16_INITIAL_STATE_BUILDER
            ),
            "setting_index": 0,
            "seed": builder.M03R_V16_PREDICTIVE_SPEC.seed,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        row = {**unsigned, "receipt_sha256": builder._sha256(unsigned)}
        receipt = package_root / "plans/initial-state-build.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_bytes(builder._canonical(row))
        return row

    def structural(package_root: Path, **identities: str) -> dict[str, Any]:
        slab_path = package_root / "structural/structural-slab.pt"
        slab_path.parent.mkdir(parents=True, exist_ok=True)
        slab_path.write_bytes(b"structural-slab")
        receipt = _Receipt(
            cache_sha256=identities["cache_sha256"],
            cache_manifest_sha256=identities["cache_manifest_sha256"],
            asset_axis_sha256="3" * 64,
            source_manifest_sha256=identities["source_manifest_sha256"],
            operator_source_sha256=identities["operator_source_sha256"],
            risk_artifact_file_sha256=identities["risk_artifact_file_sha256"],
            risk_source_manifest_file_sha256=identities[
                "risk_source_manifest_file_sha256"
            ],
            risk_source_receipt_sha256="4" * 64,
            exposure_receipt_sha256="5" * 64,
            projector_manifest_file_sha256=identities[
                "projector_manifest_file_sha256"
            ],
            projector_manifest_sha256="6" * 64,
            projector_binding_sha256="7" * 64,
            fold_geometry_sha256=tuple(
                row.receipt_sha256
                for row in builder.render_m03r_v16_fold_geometries(1001)
            ),
            action_operator_root_sha256="8" * 64,
            common_target_operator_root_sha256="9" * 64,
            target_root_sha256=("a" * 64, "b" * 64, "c" * 64),
            receipt_sha256="d" * 64,
        )
        holder["receipt"] = receipt
        unsigned = {
            "schema": "fixture",
            "slab_file_sha256": builder._file_sha256(slab_path),
            "slab_receipt_sha256": receipt.receipt_sha256,
        }
        row = {**unsigned, "receipt_sha256": builder._sha256(unsigned)}
        plan_receipt = package_root / "plans/structural-slab-build.json"
        plan_receipt.parent.mkdir(parents=True, exist_ok=True)
        plan_receipt.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return row

    def load_slab(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(slab=SimpleNamespace(receipt=holder["receipt"]))

    monkeypatch.setattr(builder, "_run_package_owned_initial_state", initial_state)
    monkeypatch.setattr(builder, "_run_package_owned_structural_slab", structural)
    monkeypatch.setattr(builder, "load_m03r_v16_structural_slab", load_slab)
    monkeypatch.setattr(
        builder,
        "load_verified_top2000_hold30_development_cache",
        lambda *_args, **_kwargs: object(),
    )


def test_v16_builder_seals_package_owned_structural_slab(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fakes(monkeypatch)
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    receipt = build_m03r_v16_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    assert validate_m03r_v16_local_package(output)["receipt_sha256"] == receipt[
        "receipt_sha256"
    ]
    package = load_m03r_v16_package_plan(
        output / "package/plans/package-plan.json",
        expected_file_sha256=receipt["package_plan_file_sha256"],
    )
    authorization = load_m03r_v16_execution_authorization(
        output / "package/plans/execution-authorization.json",
        expected_file_sha256=receipt["execution_authorization_file_sha256"],
        package=package,
    )
    assert package.artifacts.structural_slab_receipt_sha256 == "d" * 64
    assert authorization.maximum_h100_requests == 6
    assert authorization.economic_training_authorized is False
    assert authorization.reinforcement_learning_authorized is False


def test_v16_builder_rejects_member_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fakes(monkeypatch)
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    build_m03r_v16_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    member = output / "package/cache/top2000-daily-bars.pt"
    member.chmod(0o640)
    member.write_bytes(b"tampered")
    with pytest.raises(M03RV16PackageBuildError, match="inventory"):
        validate_m03r_v16_local_package(output)


def test_v16_transfer_archive_is_safe_and_inventory_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fakes(monkeypatch)
    source, cache, cache_manifest, risk = _inputs(tmp_path)
    output = tmp_path / "stage"
    build_m03r_v16_local_package(
        source_root=source,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        risk_root=risk,
        output_root=output,
    )
    archive = tmp_path / "m03r-v16-package.tar"
    receipt = build_m03r_v16_transfer_archive(output, archive)
    assert receipt["safe_member_inventory_verified"] is True
    assert validate_m03r_v16_transfer_archive(
        archive,
        expected_archive_sha256=receipt["archive_sha256"],
    ) == receipt

    archive.chmod(0o640)
    with archive.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(M03RV16PackageBuildError, match="hash drifted"):
        validate_m03r_v16_transfer_archive(
            archive,
            expected_archive_sha256=receipt["archive_sha256"],
        )
