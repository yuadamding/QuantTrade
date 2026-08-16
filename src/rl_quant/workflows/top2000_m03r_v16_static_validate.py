"""Same-image, zero-GPU static validation for the M03R-v16 package."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC
from rl_quant.training.top2000_m03r_v16_package import (
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_initial_state import (
    load_m03r_v16_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_static_contract import (
    M03R_V16_STATIC_RESULT_SCHEMA,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    load_m03r_v16_structural_slab,
)
from rl_quant.training.top2000_m03r_v16_source import (
    verify_m03r_v16_source_tree,
)


class M03RV16StaticValidationError(RuntimeError):
    """The immutable V16 package or zero-GPU process surface drifted."""


def _require_hash(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise M03RV16StaticValidationError(f"{label} hash or file type drifted")


def _write_result(path: Path, value: dict[str, Any]) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_file_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def validate_static_package(
    *,
    package_plan_path: str | Path,
    package_plan_file_sha256: str,
    execution_authorization_path: str | Path,
    execution_authorization_file_sha256: str,
    output_root: str | Path,
    expected_package_root: str | Path = "/mnt/package",
) -> dict[str, Any]:
    package = load_m03r_v16_package_plan(
        package_plan_path,
        expected_file_sha256=package_plan_file_sha256,
    )
    authorization = load_m03r_v16_execution_authorization(
        execution_authorization_path,
        expected_file_sha256=execution_authorization_file_sha256,
        package=package,
    )
    package_root = Path(package_plan_path).resolve().parent.parent
    if package_root != Path(expected_package_root).resolve():
        raise M03RV16StaticValidationError("V16 package root is not the bound mount")
    output = Path(output_root)
    if output.is_symlink() or not output.is_dir() or tuple(output.iterdir()):
        raise M03RV16StaticValidationError("V16 static output must start empty")
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none":
        raise M03RV16StaticValidationError("V16 static process is not GPU-masked")
    expected = {
        package_root / "source.tar": package.artifacts.source_archive_sha256,
        package_root / "source-manifest.json": package.artifacts.source_manifest_sha256,
        package_root / "cache/top2000-daily-bars.pt": (
            package.artifacts.cache_artifact_sha256
        ),
        package_root / "cache/cache-manifest.json": (
            package.artifacts.cache_manifest_sha256
        ),
        package_root / "risk/risk-exposures.pt": (
            package.artifacts.risk_artifact_sha256
        ),
        package_root / "risk/risk-source-manifest.json": (
            package.artifacts.risk_source_manifest_file_sha256
        ),
        package_root / "risk/projector-manifest.json": (
            package.artifacts.projector_manifest_file_sha256
        ),
        package_root / "model/common-initial-parameter-state.pt": (
            package.artifacts.initial_parameter_state_file_sha256
        ),
        package_root / "structural/structural-slab.pt": (
            package.artifacts.structural_slab_file_sha256
        ),
    }
    for path, digest in expected.items():
        _require_hash(path, digest, path.name)
    for setting_index in range(3):
        policy = Top2000M03RV16PredictivePolicy(setting_index)
        load_m03r_v16_initial_parameter_state(
            package_root / "model/common-initial-parameter-state.pt",
            policy,
            expected_file_sha256=(
                package.artifacts.initial_parameter_state_file_sha256
            ),
            expected_state_sha256=package.artifacts.initial_parameter_state_sha256,
            expected_architecture_sha256=(
                package.artifacts.initial_parameter_architecture_sha256
            ),
        )
    structural = load_m03r_v16_structural_slab(
        package_root / "structural/structural-slab.pt",
        expected_file_sha256=package.artifacts.structural_slab_file_sha256,
        expected_receipt_sha256=package.artifacts.structural_slab_receipt_sha256,
    )
    structural.slab.receipt.validate_for_package(
        cache_sha256=package.artifacts.cache_artifact_sha256,
        cache_manifest_sha256=package.artifacts.cache_manifest_sha256,
        asset_axis_sha256=package.artifacts.asset_axis_sha256,
        source_manifest_sha256=package.artifacts.source_manifest_sha256,
        operator_source_sha256=package.artifacts.operator_source_sha256,
        risk_artifact_file_sha256=package.artifacts.risk_artifact_sha256,
        risk_source_manifest_file_sha256=(
            package.artifacts.risk_source_manifest_file_sha256
        ),
        risk_source_receipt_sha256=package.artifacts.risk_source_receipt_sha256,
        exposure_receipt_sha256=package.artifacts.exposure_receipt_sha256,
        projector_manifest_file_sha256=(
            package.artifacts.projector_manifest_file_sha256
        ),
        projector_manifest_sha256=package.artifacts.projector_manifest_sha256,
        projector_binding_sha256=package.artifacts.projector_binding_sha256,
    )
    module_root = package_root / "source" / "src"
    for module in (
        __file__,
        __import__(
            "rl_quant.training.top2000_m03r_v16_package",
            fromlist=["__file__"],
        ).__file__,
        __import__(
            "rl_quant.training.top2000_m03r_v16_structural",
            fromlist=["__file__"],
        ).__file__,
    ):
        if module is None or not Path(module).resolve().is_relative_to(module_root):
            raise M03RV16StaticValidationError(
                "V16 static module resolved outside immutable source"
            )
    verified_source = verify_m03r_v16_source_tree(
        package_root / "source",
        package_root / "source-manifest.json",
        expected_source_manifest_file_sha256=(
            package.artifacts.source_manifest_sha256
        ),
        expected_runtime_worker_sha256=package.artifacts.worker_source_sha256,
    )
    unsigned = {
        "schema": M03R_V16_STATIC_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_plan_file_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "execution_authorization_file_sha256": (
            execution_authorization_file_sha256
        ),
        "source_archive_sha256": package.artifacts.source_archive_sha256,
        "source_manifest_sha256": package.artifacts.source_manifest_sha256,
        "worker_source_sha256": package.artifacts.worker_source_sha256,
        "source_tree_root_sha256": verified_source.source_tree_root_sha256,
        "structural_slab_file_sha256": (
            package.artifacts.structural_slab_file_sha256
        ),
        "structural_slab_receipt_sha256": structural.receipt_sha256,
        "panel_schedule_sha256": package.schedule.receipt_sha256,
        "hold_target_sessions": LEGACY_HOLD30_TARGET_SPEC.target_sessions,
        "hold_target_spec_sha256": LEGACY_HOLD30_TARGET_SPEC.receipt_sha256,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "gpu_mask": "none",
        "gpu_requests": 0,
        "gpu_limits": 0,
        "unmasked_visibility_claimed": False,
        "output_empty": True,
        "container_started": True,
        "initial_state_strict_loaded_all_settings": True,
        "training_performed": False,
        "economic_training_authorized": False,
        "reinforcement_learning_authorized": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    result = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    result_file_sha256 = _write_result(output / "static-result.json", result)
    return {**result, "result_file_sha256": result_file_sha256}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--execution-authorization-file-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_static_package(
        package_plan_path=args.package_plan,
        package_plan_file_sha256=args.package_plan_file_sha256,
        execution_authorization_path=args.execution_authorization,
        execution_authorization_file_sha256=(
            args.execution_authorization_file_sha256
        ),
        output_root=args.output_root,
    )
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V16_STATIC_RESULT_SCHEMA",
    "M03RV16StaticValidationError",
    "main",
    "validate_static_package",
]
