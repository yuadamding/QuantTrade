"""Same-image, zero-GPU static validation for the M03R-v12 package.

This entrypoint reads only immutable development-package metadata and hashes.
It never opens market tensors, executes a model, or accesses 2026 outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rl_quant.training.top2000_m03r_v12_package import (
    load_m03r_v12_execution_authorization,
    load_m03r_v12_package_plan,
)
from rl_quant.training.top2000_m03r_v12_static_contract import (
    M03R_V12_STATIC_RESULT_SCHEMA,
)
from rl_quant.training.top2000_m03r_v12_structural_preflight import (
    load_m03r_v12_structural_preflight,
)

class M03RV12StaticValidationError(RuntimeError):
    """The immutable package or zero-GPU process surface drifted."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or _file_sha256(path) != expected:
        raise M03RV12StaticValidationError(f"{label} hash or file type drifted")


def validate_static_package(
    *,
    package_plan_path: str | Path,
    package_plan_file_sha256: str,
    execution_authorization_path: str | Path,
    execution_authorization_file_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    package = load_m03r_v12_package_plan(
        package_plan_path,
        expected_file_sha256=package_plan_file_sha256,
    )
    authorization = load_m03r_v12_execution_authorization(
        execution_authorization_path,
        expected_file_sha256=execution_authorization_file_sha256,
        package=package,
    )
    package_root = Path(package_plan_path).resolve().parent.parent
    expected_root = Path("/mnt/package")
    if package_root != expected_root:
        raise M03RV12StaticValidationError("package root is not the bound mount")
    output = Path(output_root)
    if output.is_symlink() or not output.is_dir() or tuple(output.iterdir()):
        raise M03RV12StaticValidationError("static output root must start empty")
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none":
        raise M03RV12StaticValidationError("static process is not GPU-masked")
    _require_hash(
        package_root / "source.tar",
        package.artifacts.source_archive_sha256,
        "source archive",
    )
    _require_hash(
        package_root / "source-manifest.json",
        package.artifacts.source_manifest_sha256,
        "source manifest",
    )
    _require_hash(
        package_root / "cache" / "top2000-daily-bars.pt",
        package.artifacts.cache_artifact_sha256,
        "development cache",
    )
    _require_hash(
        package_root / "cache-manifest.json",
        package.artifacts.cache_manifest_sha256,
        "cache manifest",
    )
    _require_hash(
        package_root / "risk" / "risk-exposures.pt",
        package.artifacts.risk_artifact_sha256,
        "risk artifact",
    )
    _require_hash(
        package_root / "risk" / "risk-source-manifest.json",
        package.artifacts.risk_source_manifest_file_sha256,
        "risk source manifest",
    )
    _require_hash(
        package_root / "risk" / "projector-manifest.json",
        package.artifacts.projector_manifest_file_sha256,
        "projector manifest",
    )
    _require_hash(
        package_root / "plans" / "real-data-structural-preflight.json",
        package.artifacts.structural_preflight_file_sha256,
        "real-data structural preflight",
    )
    structural = load_m03r_v12_structural_preflight(
        package_root / "plans" / "real-data-structural-preflight.json",
        expected_file_sha256=package.artifacts.structural_preflight_file_sha256,
        package=package,
    )
    module_root = package_root / "source" / "src"
    for module in (
        __file__,
        __import__(
            "rl_quant.training.top2000_m03r_v12_package",
            fromlist=["__file__"],
        ).__file__,
    ):
        if module is None or not Path(module).resolve().is_relative_to(module_root):
            raise M03RV12StaticValidationError(
                "static module resolved outside the immutable source root"
            )
    return {
        "schema": M03R_V12_STATIC_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_plan_file_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "execution_authorization_file_sha256": (execution_authorization_file_sha256),
        "source_archive_sha256": package.artifacts.source_archive_sha256,
        "structural_preflight_receipt_sha256": structural.receipt_sha256,
        "selected_horizon_sessions": 3,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "gpu_mask": "none",
        "gpu_requests": 0,
        "gpu_limits": 0,
        "unmasked_visibility_claimed": False,
        "output_empty": True,
        "container_started": True,
        "training_performed": False,
        "economic_training_authorized": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }


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
        execution_authorization_file_sha256=(args.execution_authorization_file_sha256),
        output_root=args.output_root,
    )
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V12_STATIC_RESULT_SCHEMA",
    "M03RV12StaticValidationError",
    "main",
    "validate_static_package",
]
