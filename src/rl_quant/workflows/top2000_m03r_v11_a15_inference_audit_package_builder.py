"""Build one immutable source-only package for the a15 inference audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_package import (
    M03R_V11_A15_AUDIT_RUNTIME_ENTRYPOINT,
    M03RV11A15InferenceAuditPackageArtifacts,
    build_m03r_v11_a15_inference_audit_authorization,
    build_m03r_v11_a15_inference_audit_package_plan,
    load_m03r_v11_a15_inference_audit_bundle,
    write_m03r_v11_a15_inference_audit_authorization,
    write_m03r_v11_a15_inference_audit_package_plan,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan import (
    load_m03r_v11_a15_inference_audit_plan,
)
from rl_quant.workflows.top2000_m03r_v11_package_builder import (
    _canonical,
    _copy,
    _exclusive,
    _file_sha256,
    _inventory,
    _regular,
    _safe_tar,
    _source_members,
    _write_source_tar,
)

M03R_V11_A15_AUDIT_LOCAL_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-local-package-v1"
)
M03R_V11_A15_AUDIT_SOURCE_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-source-manifest-v1"
)
M03R_V11_A15_AUDIT_EXECUTION_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-execution-manifest-v1"
)
M03R_V11_A15_AUDIT_TRANSFER_ROOT = "qt-m03r-v11-a15-inference-audit-package-v1"
M03R_V11_A15_AUDIT_RUNTIME_WORKER = (
    "src/rl_quant/workflows/top2000_m03r_v11_a15_inference_audit.py"
)


class M03RV11A15InferenceAuditPackageBuildError(ValueError):
    """The local audit package, source inventory, or archive drifted."""


def _receipt_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def build_m03r_v11_a15_inference_audit_local_package(
    *,
    source_root: str | Path,
    audit_plan_path: str | Path,
    audit_plan_file_sha256: str,
    parent_terminal_evidence_path: str | Path,
    parent_cleanup_receipt_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    audit = load_m03r_v11_a15_inference_audit_plan(
        audit_plan_path,
        expected_file_sha256=audit_plan_file_sha256,
    )
    terminal = _regular(Path(parent_terminal_evidence_path), "terminal evidence")
    cleanup = _regular(Path(parent_cleanup_receipt_path), "cleanup receipt")
    if (
        _file_sha256(terminal) != audit.parent_terminal_evidence_file_sha256
        or _file_sha256(cleanup) != audit.parent_cleanup_receipt_file_sha256
    ):
        raise M03RV11A15InferenceAuditPackageBuildError(
            "parent lifecycle files do not match the audit plan"
        )
    output = Path(output_root)
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    package_root = output / "package"
    package_root.mkdir(mode=0o750)

    source_rows: list[dict[str, Any]] = []
    for member in _source_members(source):
        relative = member.relative_to(source).as_posix()
        sha = _copy(member, package_root / "source" / relative)
        source_rows.append(
            {"path": relative, "sha256": sha, "size": member.stat().st_size}
        )
    source_tuple = tuple(source_rows)
    if M03R_V11_A15_AUDIT_RUNTIME_WORKER not in {row["path"] for row in source_tuple}:
        raise M03RV11A15InferenceAuditPackageBuildError(
            "audit runtime worker is absent from the source inventory"
        )
    source_manifest_sha = _exclusive(
        package_root / "source-manifest.json",
        _canonical(
            {
                "schema": M03R_V11_A15_AUDIT_SOURCE_MANIFEST_SCHEMA,
                "protocol_sha256": (M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256),
                "file_count": len(source_tuple),
                "files": source_tuple,
                "runtime_worker": M03R_V11_A15_AUDIT_RUNTIME_WORKER,
                "training_authorized": False,
                "checkpoint_selection_authorized": False,
                "economic_training_authorized": False,
                "outer_2026_access_authorized": False,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
        ),
    )
    source_archive_sha = _write_source_tar(package_root, source_tuple)
    copied_audit_plan_sha = _copy(
        Path(audit_plan_path), package_root / "plans" / "audit-plan.json"
    )
    copied_terminal_sha = _copy(
        terminal,
        package_root
        / "parent-lifecycle"
        / "predictive-evidence"
        / "terminal-evidence.json",
    )
    copied_cleanup_sha = _copy(
        cleanup,
        package_root
        / "parent-lifecycle"
        / "predictive-evidence"
        / "cleanup-receipt.json",
    )
    if (
        copied_audit_plan_sha != audit_plan_file_sha256
        or copied_terminal_sha != audit.parent_terminal_evidence_file_sha256
        or copied_cleanup_sha != audit.parent_cleanup_receipt_file_sha256
    ):
        raise M03RV11A15InferenceAuditPackageBuildError("copied audit lineage changed")
    image_digest = audit.parent_image_reference.rsplit("@sha256:", 1)[-1]
    artifacts = M03RV11A15InferenceAuditPackageArtifacts(
        source_archive_sha256=source_archive_sha,
        source_manifest_sha256=source_manifest_sha,
        dependency_lock_sha256=_file_sha256(source / "uv.lock"),
        worker_source_sha256=_file_sha256(source / M03R_V11_A15_AUDIT_RUNTIME_WORKER),
        audit_plan_file_sha256=audit_plan_file_sha256,
        audit_plan_receipt_sha256=audit.receipt_sha256,
        parent_package_plan_file_sha256=(audit.parent_package_plan_file_sha256),
        parent_package_plan_sha256=audit.parent_package_plan_sha256,
        parent_execution_authorization_file_sha256=(
            audit.parent_execution_authorization_file_sha256
        ),
        parent_execution_authorization_receipt_sha256=(
            audit.parent_execution_authorization_receipt_sha256
        ),
        parent_source_archive_sha256=audit.parent_source_archive_sha256,
        parent_terminal_evidence_file_sha256=(
            audit.parent_terminal_evidence_file_sha256
        ),
        parent_cleanup_receipt_file_sha256=(audit.parent_cleanup_receipt_file_sha256),
        parent_cleanup_receipt_sha256=audit.parent_cleanup_receipt_sha256,
        image_reference=audit.parent_image_reference,
        image_digest_sha256=image_digest,
    )
    package = build_m03r_v11_a15_inference_audit_package_plan(artifacts, audit)
    package_plan_path = package_root / "plans" / "package-plan.json"
    package_plan_file_sha = write_m03r_v11_a15_inference_audit_package_plan(
        package_plan_path,
        package,
        audit,
    )
    authorization = build_m03r_v11_a15_inference_audit_authorization(
        package,
        audit,
        package_plan_file_sha256=package_plan_file_sha,
    )
    authorization_path = package_root / "plans" / "execution-authorization.json"
    authorization_file_sha = write_m03r_v11_a15_inference_audit_authorization(
        authorization_path,
        authorization,
        package,
        audit,
    )
    execution_manifest_sha = _exclusive(
        package_root / "execution-manifest.json",
        _canonical(
            {
                "schema": M03R_V11_A15_AUDIT_EXECUTION_MANIFEST_SCHEMA,
                "protocol_sha256": (M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256),
                "package_plan_sha256": package.package_plan_sha256,
                "package_plan_file_sha256": package_plan_file_sha,
                "authorization_receipt_sha256": authorization.receipt_sha256,
                "authorization_file_sha256": authorization_file_sha,
                "audit_plan_receipt_sha256": audit.receipt_sha256,
                "audit_plan_file_sha256": audit_plan_file_sha256,
                "parent_cleanup_receipt_sha256": (audit.parent_cleanup_receipt_sha256),
                "runtime_entrypoint": M03R_V11_A15_AUDIT_RUNTIME_ENTRYPOINT,
                "indexed_completions": 2,
                "h100s_per_completion": 1,
                "maximum_h100_requests": 2,
                "training_authorized": False,
                "checkpoint_selection_authorized": False,
                "inference_audit_authorized": True,
                "economic_training_authorized": False,
                "economic_generation_may_be_minted": False,
                "outer_2026_access_authorized": False,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
        ),
    )
    inventory = _inventory(package_root)
    unsigned = {
        "schema": M03R_V11_A15_AUDIT_LOCAL_PACKAGE_SCHEMA,
        "package_relative_root": "package",
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_plan_file_sha,
        "authorization_receipt_sha256": authorization.receipt_sha256,
        "authorization_file_sha256": authorization_file_sha,
        "audit_plan_receipt_sha256": audit.receipt_sha256,
        "audit_plan_file_sha256": audit_plan_file_sha256,
        "source_archive_sha256": source_archive_sha,
        "source_manifest_sha256": source_manifest_sha,
        "worker_source_sha256": artifacts.worker_source_sha256,
        "execution_manifest_sha256": execution_manifest_sha,
        "parent_terminal_evidence_file_sha256": copied_terminal_sha,
        "parent_cleanup_receipt_file_sha256": copied_cleanup_sha,
        "parent_cleanup_receipt_sha256": audit.parent_cleanup_receipt_sha256,
        "image_reference": audit.parent_image_reference,
        "image_digest_sha256": image_digest,
        "file_inventory": inventory,
        "source_file_count": len(source_tuple),
        "training_authorized": False,
        "checkpoint_selection_authorized": False,
        "inference_audit_authorized": True,
        "economic_training_authorized": False,
        "economic_generation_may_be_minted": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    receipt = {**unsigned, "receipt_sha256": _receipt_sha256(unsigned)}
    _exclusive(output / "package-build-receipt.json", _canonical(receipt))
    validate_m03r_v11_a15_inference_audit_local_package(output)
    return receipt


def validate_m03r_v11_a15_inference_audit_local_package(
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root)
    receipt_path = _regular(
        output / "package-build-receipt.json", "audit package receipt"
    )
    receipt = json.loads(receipt_path.read_bytes())
    if not isinstance(receipt, dict):
        raise M03RV11A15InferenceAuditPackageBuildError(
            "audit package receipt is not an object"
        )
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != M03R_V11_A15_AUDIT_LOCAL_PACKAGE_SCHEMA
        or receipt.get("receipt_sha256") != _receipt_sha256(unsigned)
        or receipt.get("training_authorized") is not False
        or receipt.get("checkpoint_selection_authorized") is not False
        or receipt.get("inference_audit_authorized") is not True
        or receipt.get("economic_training_authorized") is not False
        or receipt.get("outer_2026_access_authorized") is not False
    ):
        raise M03RV11A15InferenceAuditPackageBuildError(
            "audit package receipt semantics drifted"
        )
    package_root = output / "package"
    if tuple(receipt.get("file_inventory", ())) != _inventory(package_root):
        raise M03RV11A15InferenceAuditPackageBuildError(
            "audit package inventory drifted"
        )
    audit, package, authorization = load_m03r_v11_a15_inference_audit_bundle(
        audit_plan_path=package_root / "plans" / "audit-plan.json",
        audit_plan_file_sha256=receipt["audit_plan_file_sha256"],
        package_plan_path=package_root / "plans" / "package-plan.json",
        package_plan_file_sha256=receipt["package_plan_file_sha256"],
        authorization_path=package_root / "plans" / "execution-authorization.json",
        authorization_file_sha256=receipt["authorization_file_sha256"],
    )
    if (
        package.package_plan_sha256 != receipt["package_plan_sha256"]
        or authorization.receipt_sha256 != receipt["authorization_receipt_sha256"]
        or audit.receipt_sha256 != receipt["audit_plan_receipt_sha256"]
        or _file_sha256(package_root / "source.tar") != receipt["source_archive_sha256"]
    ):
        raise M03RV11A15InferenceAuditPackageBuildError(
            "audit package typed binding drifted"
        )
    _safe_tar(package_root / "source.tar")
    return receipt


def build_m03r_v11_a15_inference_audit_transfer_archive(
    output_root: str | Path,
    archive_path: str | Path,
) -> dict[str, str]:
    receipt = validate_m03r_v11_a15_inference_audit_local_package(output_root)
    root = Path(output_root)
    target = Path(archive_path)
    if target.exists() or target.is_symlink():
        raise M03RV11A15InferenceAuditPackageBuildError(
            "audit transfer archive already exists"
        )
    files = (root / "package-build-receipt.json", *(root / "package").rglob("*"))
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as archive:
        for source in sorted(
            (path for path in files if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        ):
            relative = source.relative_to(root).as_posix()
            info = tarfile.TarInfo(f"{M03R_V11_A15_AUDIT_TRANSFER_ROOT}/{relative}")
            info.size = source.stat().st_size
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    archive_sha = _file_sha256(target)
    validate_m03r_v11_a15_inference_audit_transfer_archive(
        target,
        expected_archive_sha256=archive_sha,
        expected_package_receipt_file_sha256=_file_sha256(
            root / "package-build-receipt.json"
        ),
    )
    return {
        "archive_sha256": archive_sha,
        "package_receipt_sha256": receipt["receipt_sha256"],
    }


def validate_m03r_v11_a15_inference_audit_transfer_archive(
    archive_path: str | Path,
    *,
    expected_archive_sha256: str,
    expected_package_receipt_file_sha256: str,
) -> None:
    path = _regular(Path(archive_path), "audit transfer archive")
    if _file_sha256(path) != expected_archive_sha256:
        raise M03RV11A15InferenceAuditPackageBuildError(
            "audit transfer archive hash drifted"
        )
    _safe_tar(path, required_root=M03R_V11_A15_AUDIT_TRANSFER_ROOT)
    receipt_name = f"{M03R_V11_A15_AUDIT_TRANSFER_ROOT}/package-build-receipt.json"
    names: set[str] = set()
    receipt_bytes: bytes | None = None
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or member.name in names:
                raise M03RV11A15InferenceAuditPackageBuildError(
                    "audit transfer archive member drifted"
                )
            names.add(member.name)
            if member.name == receipt_name:
                stream = archive.extractfile(member)
                if stream is None:
                    raise M03RV11A15InferenceAuditPackageBuildError(
                        "audit transfer receipt is unreadable"
                    )
                receipt_bytes = stream.read()
    if (
        receipt_bytes is None
        or hashlib.sha256(receipt_bytes).hexdigest()
        != expected_package_receipt_file_sha256
    ):
        raise M03RV11A15InferenceAuditPackageBuildError(
            "audit transfer receipt hash drifted"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-root", required=True)
    build.add_argument("--audit-plan", required=True)
    build.add_argument("--audit-plan-file-sha256", required=True)
    build.add_argument("--parent-terminal-evidence", required=True)
    build.add_argument("--parent-cleanup-receipt", required=True)
    build.add_argument("--output-root", required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--output-root", required=True)
    archive.add_argument("--archive", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "build":
        result = build_m03r_v11_a15_inference_audit_local_package(
            source_root=arguments.source_root,
            audit_plan_path=arguments.audit_plan,
            audit_plan_file_sha256=arguments.audit_plan_file_sha256,
            parent_terminal_evidence_path=arguments.parent_terminal_evidence,
            parent_cleanup_receipt_path=arguments.parent_cleanup_receipt,
            output_root=arguments.output_root,
        )
    else:
        result = build_m03r_v11_a15_inference_audit_transfer_archive(
            arguments.output_root,
            arguments.archive,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "M03R_V11_A15_AUDIT_EXECUTION_MANIFEST_SCHEMA",
    "M03R_V11_A15_AUDIT_LOCAL_PACKAGE_SCHEMA",
    "M03R_V11_A15_AUDIT_SOURCE_MANIFEST_SCHEMA",
    "M03R_V11_A15_AUDIT_TRANSFER_ROOT",
    "M03RV11A15InferenceAuditPackageBuildError",
    "build_m03r_v11_a15_inference_audit_local_package",
    "build_m03r_v11_a15_inference_audit_transfer_archive",
    "main",
    "validate_m03r_v11_a15_inference_audit_local_package",
    "validate_m03r_v11_a15_inference_audit_transfer_archive",
]
