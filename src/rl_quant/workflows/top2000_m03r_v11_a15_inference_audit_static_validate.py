"""Zero-GPU same-image static validator for the a15 inference audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_AUDIT_STATIC_TERMINAL_SCHEMA,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_package import (
    M03R_V11_A15_AUDIT_RUNTIME_ENTRYPOINT,
    load_m03r_v11_a15_inference_audit_bundle,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan import (
    build_m03r_v11_a15_inference_audit_plan,
)
from rl_quant.training.top2000_m03r_v11_package import (
    load_m03r_v11_execution_authorization,
    load_m03r_v11_package_plan,
)

_MAX_SOURCE_MANIFEST_BYTES = 2 * 1024 * 1024


class M03RV11A15InferenceAuditStaticError(RuntimeError):
    """The static source, package, or parent evidence drifted."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        status = path.lstat()
    except OSError as exc:
        raise M03RV11A15InferenceAuditStaticError(
            "static package member is unavailable"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise M03RV11A15InferenceAuditStaticError(
            "static package member must be a nonempty regular file"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_inventory(package_root: Path, expected_manifest_sha: str) -> int:
    manifest_path = package_root / "source-manifest.json"
    if _file_sha256(manifest_path) != expected_manifest_sha:
        raise M03RV11A15InferenceAuditStaticError(
            "static source manifest file hash drifted"
        )
    if manifest_path.stat().st_size > _MAX_SOURCE_MANIFEST_BYTES:
        raise M03RV11A15InferenceAuditStaticError("static source manifest is too large")
    payload = json.loads(manifest_path.read_bytes())
    rows = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(rows, list)
        or payload.get("file_count") != len(rows)
        or payload.get("runtime_worker")
        != "src/rl_quant/workflows/top2000_m03r_v11_a15_inference_audit.py"
        or payload.get("training_authorized") is not False
        or payload.get("checkpoint_selection_authorized") is not False
        or payload.get("economic_training_authorized") is not False
        or payload.get("outer_2026_access_authorized") is not False
    ):
        raise M03RV11A15InferenceAuditStaticError(
            "static source manifest semantics drifted"
        )
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise M03RV11A15InferenceAuditStaticError(
                "static source inventory row is invalid"
            )
        relative = row.get("path")
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        path = package_root / "source" / str(relative)
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or "." in pure.parts
            or ".." in pure.parts
            or str(pure) != relative
            or relative in names
            or row.get("size") != path.stat().st_size
            or row.get("sha256") != _file_sha256(path)
        ):
            raise M03RV11A15InferenceAuditStaticError("static source inventory drifted")
        names.add(relative)
    return len(rows)


def _write(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise M03RV11A15InferenceAuditStaticError(
            "static terminal target already exists"
        )
    encoded = _canonical(payload)
    path.parent.mkdir(parents=True, mode=0o750, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def validate_m03r_v11_a15_inference_audit_static(
    *,
    audit_package_root: str | Path,
    audit_plan_file_sha256: str,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    parent_package_root: str | Path,
    parent_output_root: str | Path,
    parent_lifecycle_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    package_root = Path(audit_package_root)
    audit, package, authorization = load_m03r_v11_a15_inference_audit_bundle(
        audit_plan_path=package_root / "plans" / "audit-plan.json",
        audit_plan_file_sha256=audit_plan_file_sha256,
        package_plan_path=package_root / "plans" / "package-plan.json",
        package_plan_file_sha256=package_plan_file_sha256,
        authorization_path=package_root / "plans" / "execution-authorization.json",
        authorization_file_sha256=authorization_file_sha256,
    )
    parent_root = Path(parent_package_root)
    parent_package = load_m03r_v11_package_plan(
        parent_root / "plans" / "package-plan.json",
        expected_file_sha256=audit.parent_package_plan_file_sha256,
    )
    parent_authorization = load_m03r_v11_execution_authorization(
        parent_root / "plans" / "execution-authorization.json",
        expected_file_sha256=audit.parent_execution_authorization_file_sha256,
        package=parent_package,
    )
    if (
        parent_package.package_plan_sha256 != audit.parent_package_plan_sha256
        or parent_authorization.receipt_sha256
        != audit.parent_execution_authorization_receipt_sha256
        or package.artifacts.worker_source_sha256
        != _file_sha256(
            package_root
            / "source"
            / "src/rl_quant/workflows/top2000_m03r_v11_a15_inference_audit.py"
        )
        or package.artifacts.source_archive_sha256
        != _file_sha256(package_root / "source.tar")
        or package.artifacts.dependency_lock_sha256
        != _file_sha256(package_root / "source" / "uv.lock")
        or authorization.runtime_entrypoint != M03R_V11_A15_AUDIT_RUNTIME_ENTRYPOINT
    ):
        raise M03RV11A15InferenceAuditStaticError(
            "static package or parent identity drifted"
        )
    source_count = _validate_source_inventory(
        package_root,
        package.artifacts.source_manifest_sha256,
    )
    rebuilt = build_m03r_v11_a15_inference_audit_plan(
        parent_package_plan_path=parent_root / "plans" / "package-plan.json",
        parent_package_plan_file_sha256=audit.parent_package_plan_file_sha256,
        parent_execution_authorization_path=(
            parent_root / "plans" / "execution-authorization.json"
        ),
        parent_execution_authorization_file_sha256=(
            audit.parent_execution_authorization_file_sha256
        ),
        parent_output_root=parent_output_root,
        parent_worker_terminal_file_sha256=tuple(
            row.terminal_file_sha256 for row in audit.workers
        ),  # type: ignore[arg-type]
        parent_fold_terminal_file_sha256=tuple(
            row.fold_terminal_file_sha256 for row in audit.workers
        ),  # type: ignore[arg-type]
        parent_launch_root=parent_lifecycle_root,
        parent_terminal_evidence_file_sha256=(
            audit.parent_terminal_evidence_file_sha256
        ),
        parent_cleanup_receipt_file_sha256=(audit.parent_cleanup_receipt_file_sha256),
    )
    if rebuilt != audit:
        raise M03RV11A15InferenceAuditStaticError(
            "static parent lineage rebuild drifted"
        )
    if (
        os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none"
        or torch.cuda.is_available()
        or torch.cuda.device_count() != 0
    ):
        raise M03RV11A15InferenceAuditStaticError(
            "static gate must have no visible GPU"
        )
    # Import every result-moving runtime boundary in the exact worker image.
    from rl_quant.evaluation import top2000_m03r_v11_a15_inference_audit as evaluation
    from rl_quant.training import (
        top2000_m03r_v11_a15_inference_audit_fold as fold,
        top2000_m03r_v11_a15_inference_audit_runtime as runtime,
    )

    expected_source_root = (package_root / "source").resolve()
    imported_paths = (
        Path(__file__).resolve(),
        Path(evaluation.__file__).resolve(),
        Path(fold.__file__).resolve(),
        Path(runtime.__file__).resolve(),
    )
    if any(not path.is_relative_to(expected_source_root) for path in imported_paths):
        raise M03RV11A15InferenceAuditStaticError(
            "static imports did not resolve from the immutable audit package"
        )
    source_hashes = {
        "static": _file_sha256(Path(__file__)),
        "evaluation": _file_sha256(Path(evaluation.__file__)),
        "fold": _file_sha256(Path(fold.__file__)),
        "runtime": _file_sha256(Path(runtime.__file__)),
    }
    unsigned = {
        "schema": M03R_V11_A15_AUDIT_STATIC_TERMINAL_SCHEMA,
        "protocol_sha256": package.protocol_sha256,
        "package_plan_sha256": package.package_plan_sha256,
        "authorization_receipt_sha256": authorization.receipt_sha256,
        "audit_plan_file_sha256": audit_plan_file_sha256,
        "audit_plan_receipt_sha256": audit.receipt_sha256,
        "parent_package_plan_sha256": audit.parent_package_plan_sha256,
        "parent_cleanup_receipt_sha256": audit.parent_cleanup_receipt_sha256,
        "source_archive_sha256": package.artifacts.source_archive_sha256,
        "source_manifest_sha256": package.artifacts.source_manifest_sha256,
        "source_file_count": source_count,
        "runtime_source_sha256": source_hashes,
        "parent_workers_validated": 2,
        "parent_fold_terminals_validated": 12,
        "parent_checkpoints_bound": 24,
        "gpu_mask": "none",
        "gpu_requests": 0,
        "gpu_limits": 0,
        "unmasked_visibility_claimed": False,
        "h100_capacity_evidence": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_training_authorized": False,
        "economic_generation_may_be_minted": False,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    terminal = {**unsigned, "receipt_sha256": _sha256(unsigned)}
    _write(Path(output_root) / "static-terminal.json", terminal)
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-package-root", required=True)
    parser.add_argument("--audit-plan-file-sha256", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--authorization-file-sha256", required=True)
    parser.add_argument("--parent-package-root", required=True)
    parser.add_argument("--parent-output-root", required=True)
    parser.add_argument("--parent-lifecycle-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    validate_m03r_v11_a15_inference_audit_static(
        audit_package_root=arguments.audit_package_root,
        audit_plan_file_sha256=arguments.audit_plan_file_sha256,
        package_plan_file_sha256=arguments.package_plan_file_sha256,
        authorization_file_sha256=arguments.authorization_file_sha256,
        parent_package_root=arguments.parent_package_root,
        parent_output_root=arguments.parent_output_root,
        parent_lifecycle_root=arguments.parent_lifecycle_root,
        output_root=arguments.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "M03R_V11_A15_AUDIT_STATIC_TERMINAL_SCHEMA",
    "M03RV11A15InferenceAuditStaticError",
    "main",
    "validate_m03r_v11_a15_inference_audit_static",
]
