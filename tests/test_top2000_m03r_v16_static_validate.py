from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackageArtifacts,
    build_m03r_v16_package_plan,
    write_m03r_v16_execution_authorization,
    write_m03r_v16_package_plan,
)
from rl_quant.workflows.top2000_m03r_v16_static_validate import (
    M03RV16StaticValidationError,
    validate_static_package,
)


class _Receipt(SimpleNamespace):
    def validate_for_package(self, **expected: str) -> None:
        assert all(getattr(self, name) == value for name, value in expected.items())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surfaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, str, str, Path]:
    import rl_quant.training.top2000_m03r_v16_package as package_module
    import rl_quant.training.top2000_m03r_v16_structural as structural_module
    import rl_quant.workflows.top2000_m03r_v16_static_validate as static_module

    root = tmp_path / "package"
    files = {
        "source.tar": b"source-archive",
        "source-manifest.json": b"source-manifest",
        "cache/top2000-daily-bars.pt": b"cache",
        "cache/cache-manifest.json": b"cache-manifest",
        "risk/risk-exposures.pt": b"risk",
        "risk/risk-source-manifest.json": b"risk-manifest",
        "risk/projector-manifest.json": b"projector-manifest",
        "model/common-initial-parameter-state.pt": b"initial-state",
        "structural/structural-slab.pt": b"structural-slab",
        "source/src/rl_quant/workflows/top2000_m03r_v16_static_validate.py": (
            b"static-source"
        ),
        "source/src/rl_quant/training/top2000_m03r_v16_package.py": (
            b"package-source"
        ),
        "source/src/rl_quant/training/top2000_m03r_v16_structural.py": (
            b"structural-source"
        ),
        "source/src/rl_quant/workflows/top2000_m03r_v16_predictive.py": (
            b"worker-source"
        ),
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    receipt = _Receipt(
        cache_sha256=_sha(root / "cache/top2000-daily-bars.pt"),
        cache_manifest_sha256=_sha(root / "cache/cache-manifest.json"),
        asset_axis_sha256="1" * 64,
        source_manifest_sha256=_sha(root / "source-manifest.json"),
        operator_source_sha256="2" * 64,
        risk_artifact_file_sha256=_sha(root / "risk/risk-exposures.pt"),
        risk_source_manifest_file_sha256=_sha(
            root / "risk/risk-source-manifest.json"
        ),
        risk_source_receipt_sha256="3" * 64,
        exposure_receipt_sha256="4" * 64,
        projector_manifest_file_sha256=_sha(
            root / "risk/projector-manifest.json"
        ),
        projector_manifest_sha256="5" * 64,
        projector_binding_sha256="6" * 64,
        receipt_sha256="7" * 64,
    )
    artifacts = M03RV16PackageArtifacts(
        source_archive_sha256=_sha(root / "source.tar"),
        source_manifest_sha256=receipt.source_manifest_sha256,
        dependency_lock_sha256="8" * 64,
        cache_artifact_sha256=receipt.cache_sha256,
        cache_manifest_sha256=receipt.cache_manifest_sha256,
        asset_axis_sha256=receipt.asset_axis_sha256,
        risk_artifact_sha256=receipt.risk_artifact_file_sha256,
        risk_source_manifest_file_sha256=(
            receipt.risk_source_manifest_file_sha256
        ),
        risk_source_receipt_sha256=receipt.risk_source_receipt_sha256,
        exposure_receipt_sha256=receipt.exposure_receipt_sha256,
        projector_manifest_file_sha256=receipt.projector_manifest_file_sha256,
        projector_manifest_sha256=receipt.projector_manifest_sha256,
        projector_binding_sha256=receipt.projector_binding_sha256,
        worker_source_sha256=_sha(
            root / "source/src/rl_quant/workflows/top2000_m03r_v16_predictive.py"
        ),
        operator_source_sha256=receipt.operator_source_sha256,
        initial_parameter_state_file_sha256=_sha(
            root / "model/common-initial-parameter-state.pt"
        ),
        initial_parameter_state_sha256="9" * 64,
        initial_parameter_architecture_sha256="a" * 64,
        structural_slab_file_sha256=_sha(
            root / "structural/structural-slab.pt"
        ),
        structural_slab_receipt_sha256=receipt.receipt_sha256,
        structural_action_operator_root_sha256="b" * 64,
        structural_target_operator_root_sha256="c" * 64,
        structural_target_root_sha256=("d" * 64, "e" * 64, "f" * 64),
        image_reference=f"registry/research@sha256:{'0' * 64}",
        image_digest_sha256="0" * 64,
    )
    schedule = M03RV16PanelSchedule(
        protocol_common_data_sha256="f" * 64,
        cache_sha256=artifacts.cache_artifact_sha256,
        asset_axis_sha256=artifacts.asset_axis_sha256,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v16_fold_geometries(1001)
        ),
    )
    plan = build_m03r_v16_package_plan(artifacts, schedule)
    plan_path = root / "plans/package-plan.json"
    plan_sha = write_m03r_v16_package_plan(plan_path, plan)
    authorization = M03RV16ExecutionAuthorization(
        package_plan_sha256=plan.package_plan_sha256,
        package_plan_file_sha256=plan_sha,
        source_archive_sha256=artifacts.source_archive_sha256,
        source_manifest_sha256=artifacts.source_manifest_sha256,
        worker_source_sha256=artifacts.worker_source_sha256,
        structural_slab_file_sha256=artifacts.structural_slab_file_sha256,
        structural_slab_receipt_sha256=artifacts.structural_slab_receipt_sha256,
        image_reference=artifacts.image_reference,
    )
    auth_path = root / "plans/execution-authorization.json"
    auth_sha = write_m03r_v16_execution_authorization(
        auth_path, authorization, plan
    )
    monkeypatch.setattr(
        static_module,
        "load_m03r_v16_structural_slab",
        lambda *_args, **_kwargs: SimpleNamespace(
            slab=SimpleNamespace(receipt=receipt),
            receipt_sha256=receipt.receipt_sha256,
        ),
    )
    monkeypatch.setattr(
        static_module,
        "__file__",
        str(root / "source/src/rl_quant/workflows/top2000_m03r_v16_static_validate.py"),
    )
    monkeypatch.setattr(
        package_module,
        "__file__",
        str(root / "source/src/rl_quant/training/top2000_m03r_v16_package.py"),
    )
    monkeypatch.setattr(
        structural_module,
        "__file__",
        str(root / "source/src/rl_quant/training/top2000_m03r_v16_structural.py"),
    )
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")
    output = tmp_path / "output"
    output.mkdir()
    return plan_path, plan_sha, auth_sha, output


def test_v16_static_validator_binds_slab_and_zero_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, plan_sha, auth_sha, output = _surfaces(monkeypatch, tmp_path)
    result = validate_static_package(
        package_plan_path=plan,
        package_plan_file_sha256=plan_sha,
        execution_authorization_path=plan.parent / "execution-authorization.json",
        execution_authorization_file_sha256=auth_sha,
        output_root=output,
        expected_package_root=plan.parent.parent,
    )
    assert result["gpu_mask"] == "none"
    assert result["gpu_requests"] == 0
    assert result["structural_slab_receipt_sha256"] == "7" * 64
    assert result["reinforcement_learning_authorized"] is False


def test_v16_static_validator_rejects_unmasked_gpu_visibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, plan_sha, auth_sha, output = _surfaces(monkeypatch, tmp_path)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    with pytest.raises(M03RV16StaticValidationError, match="GPU-masked"):
        validate_static_package(
            package_plan_path=plan,
            package_plan_file_sha256=plan_sha,
            execution_authorization_path=plan.parent
            / "execution-authorization.json",
            execution_authorization_file_sha256=auth_sha,
            output_root=output,
            expected_package_root=plan.parent.parent,
        )
