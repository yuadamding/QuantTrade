"""Build the source-only package for the M03R-v12 post-hoc audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from rl_quant.protocol.hold30_alpha_m03r_v12_posthoc_inference_audit import (
    M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v12_package import load_m03r_v12_package_plan
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_package import (
    M03RV12PosthocAuditCheckpointBinding,
    M03RV12PosthocAuditParentBinding,
    M03RV12PosthocAuditSourceArtifacts,
    build_m03r_v12_posthoc_audit_package_plan,
    load_m03r_v12_posthoc_audit_package_plan,
    write_m03r_v12_posthoc_audit_package_plan,
)

M03R_V12_POSTHOC_AUDIT_SOURCE_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-source-manifest-v1"
)
M03R_V12_POSTHOC_AUDIT_BUILD_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-build-receipt-v1"
)
M03R_V12_POSTHOC_AUDIT_TRANSFER_ROOT = "qt-m03r-v12-posthoc-audit-package-v1"
M03R_V12_POSTHOC_AUDIT_RUNTIME_WORKER = (
    "src/rl_quant/workflows/top2000_m03r_v12_posthoc_inference_audit.py"
)
_MAX_JSON_BYTES = 8 * 1024 * 1024


class M03RV12PosthocAuditPackageBuildError(ValueError):
    """The source-only audit package could not be built exactly."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, name: str) -> Path:
    try:
        status = path.lstat()
    except OSError as exc:
        raise M03RV12PosthocAuditPackageBuildError(f"{name} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise M03RV12PosthocAuditPackageBuildError(
            f"{name} must be a nonempty regular file"
        )
    return path


def _copy(source: Path, target: Path) -> str:
    _regular(source, str(source))
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            while chunk := reader.read(1024 * 1024):
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != _file_sha256(source):
        raise M03RV12PosthocAuditPackageBuildError(
            "audit source changed during copy"
        )
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    raw = _canonical(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _source_members(root: Path) -> tuple[Path, ...]:
    required = (
        root / "pyproject.toml",
        root / "uv.lock",
        root / M03R_V12_POSTHOC_AUDIT_RUNTIME_WORKER,
    )
    for path in required:
        _regular(path, str(path.relative_to(root)))
    candidates = (
        root / "pyproject.toml",
        root / "uv.lock",
        *(root / "src/rl_quant").rglob("*.py"),
    )
    result: list[Path] = []
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        _regular(path, str(path.relative_to(root)))
        if not path.resolve().is_relative_to(root):
            raise M03RV12PosthocAuditPackageBuildError(
                "audit source member leaves repository root"
            )
        result.append(path)
    if len(result) != len(set(result)):
        raise M03RV12PosthocAuditPackageBuildError(
            "audit source inventory contains duplicates"
        )
    return tuple(result)


def _source_tar(
    package_root: Path,
    source_rows: tuple[dict[str, Any], ...],
) -> str:
    target = package_root / "source.tar"
    directories = {"source"}
    for row in source_rows:
        parent = PurePosixPath("source", row["path"]).parent
        while str(parent) not in {"", "."}:
            directories.add(str(parent))
            parent = parent.parent
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as archive:
        for directory in sorted(
            directories, key=lambda value: (value.count("/"), value)
        ):
            info = tarfile.TarInfo(directory + "/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            archive.addfile(info)
        for row in source_rows:
            source = package_root / "source" / row["path"]
            info = tarfile.TarInfo("source/" + row["path"])
            info.size = row["size"]
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    os.chmod(target, 0o440)
    return _file_sha256(target)


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    _regular(path, str(path))
    raw = path.read_bytes()
    if len(raw) > _MAX_JSON_BYTES:
        raise M03RV12PosthocAuditPackageBuildError(
            "parent receipt exceeds its bound"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M03RV12PosthocAuditPackageBuildError(
            "parent receipt is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise M03RV12PosthocAuditPackageBuildError(
            "parent receipt is not an object"
        )
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != _sha256(unsigned):
        raise M03RV12PosthocAuditPackageBuildError(
            "parent semantic receipt drifted"
        )
    return value, hashlib.sha256(raw).hexdigest()


def _parent_binding(
    package_plan_path: Path,
    package_plan_file_sha256: str,
    parent_output_root: Path,
) -> M03RV12PosthocAuditParentBinding:
    package = load_m03r_v12_package_plan(
        package_plan_path,
        expected_file_sha256=package_plan_file_sha256,
    )
    checkpoints: list[M03RV12PosthocAuditCheckpointBinding] = []
    terminal_paths: list[str] = []
    terminal_files: list[str] = []
    terminal_receipts: list[str] = []
    for setting in range(3):
        relative_root = f"completion-{setting:02d}-setting-{setting:02d}"
        terminal_relative = f"{relative_root}/predictive-terminal.json"
        terminal, terminal_file_sha = _read_json(
            parent_output_root / terminal_relative
        )
        fold_file_inventory = terminal.get("fold_terminal_file_sha256")
        if (
            terminal.get("setting_index") != setting
            or terminal.get("predictive_gate_passed") is not False
            or terminal.get("selected_horizon") is not None
            or terminal.get("economic_generation_may_be_minted") is not False
            or terminal.get("outer_2026_accessed") is not False
            or not isinstance(fold_file_inventory, list)
            or len(fold_file_inventory) != 6
        ):
            raise M03RV12PosthocAuditPackageBuildError(
                "parent predictive terminal is not immutable failed evidence"
            )
        terminal_paths.append(terminal_relative)
        terminal_files.append(terminal_file_sha)
        terminal_receipts.append(terminal["receipt_sha256"])
        for fold in range(6):
            receipt_relative = f"{relative_root}/receipts/fold-{fold:02d}-terminal.json"
            receipt, receipt_file_sha = _read_json(
                parent_output_root / receipt_relative
            )
            horizon = receipt.get("horizon_candidates", {}).get("3")
            checkpoint_relative = (
                f"{relative_root}/checkpoints/"
                f"fold-{fold:02d}-horizon-03-update-0064.pt"
            )
            checkpoint_path = parent_output_root / checkpoint_relative
            if (
                receipt_file_sha != fold_file_inventory[fold]
                or receipt.get("setting_index") != setting
                or receipt.get("fold_index") != fold
                or receipt.get("completed_updates") != 64
                or receipt.get("outer_2026_accessed") is not False
                or not isinstance(horizon, dict)
                or _file_sha256(_regular(checkpoint_path, checkpoint_relative))
                != horizon.get("checkpoint_file_sha256")
            ):
                raise M03RV12PosthocAuditPackageBuildError(
                    "parent fold/checkpoint lineage drifted"
                )
            checkpoints.append(
                M03RV12PosthocAuditCheckpointBinding(
                    setting_index=setting,
                    setting_id=receipt["setting_id"],
                    fold_index=fold,
                    checkpoint_relative_path=checkpoint_relative,
                    checkpoint_file_sha256=horizon["checkpoint_file_sha256"],
                    model_state_sha256=horizon["model_state_sha256"],
                    training_residual_operator_root_sha256=receipt[
                        "training_residual_operator_root_sha256"
                    ],
                    training_source_array_sha256=receipt[
                        "training_source_array_sha256"
                    ],
                    parent_fold_terminal_relative_path=receipt_relative,
                    parent_fold_terminal_file_sha256=receipt_file_sha,
                    parent_fold_terminal_receipt_sha256=receipt["receipt_sha256"],
                )
            )
    result = M03RV12PosthocAuditParentBinding(
        run_id="qt-m03r-v12-h3-predictive-s17-20260813-a05",
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=package_plan_file_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        parent_protocol_sha256=package.protocol_sha256,
        checkpoint_bindings=tuple(checkpoints),
        predictive_terminal_relative_paths=tuple(terminal_paths),
        predictive_terminal_file_sha256=tuple(terminal_files),
        predictive_terminal_receipt_sha256=tuple(terminal_receipts),
    )
    result.validate()
    return result


def build_m03r_v12_posthoc_audit_local_package(
    *,
    source_root: str | Path,
    parent_package_plan_path: str | Path,
    parent_package_plan_file_sha256: str,
    parent_output_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    output = Path(output_root)
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    package_root = output / "package"
    package_root.mkdir(mode=0o750)
    rows: list[dict[str, Any]] = []
    for member in _source_members(source):
        relative = member.relative_to(source).as_posix()
        digest = _copy(member, package_root / "source" / relative)
        rows.append({"path": relative, "sha256": digest, "size": member.stat().st_size})
    source_rows = tuple(rows)
    source_inventory_sha256 = hashlib.sha256(_canonical(source_rows)).hexdigest()
    source_manifest_file_sha256 = _write(
        package_root / "source-manifest.json",
        {
            "schema": M03R_V12_POSTHOC_AUDIT_SOURCE_MANIFEST_SCHEMA,
            "protocol_sha256": M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
            "source_inventory_sha256": source_inventory_sha256,
            "file_count": len(source_rows),
            "files": source_rows,
            "runtime_worker": M03R_V12_POSTHOC_AUDIT_RUNTIME_WORKER,
            "training_authorized": False,
            "outer_2026_access_authorized": False,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        },
    )
    source_archive_sha256 = _source_tar(package_root, source_rows)
    parent = _parent_binding(
        Path(parent_package_plan_path),
        parent_package_plan_file_sha256,
        Path(parent_output_root),
    )
    parent_package = load_m03r_v12_package_plan(
        parent_package_plan_path,
        expected_file_sha256=parent_package_plan_file_sha256,
    )
    artifacts = M03RV12PosthocAuditSourceArtifacts(
        source_archive_sha256=source_archive_sha256,
        source_manifest_file_sha256=source_manifest_file_sha256,
        source_inventory_sha256=source_inventory_sha256,
        dependency_lock_sha256=_file_sha256(source / "uv.lock"),
        worker_source_sha256=_file_sha256(
            source / M03R_V12_POSTHOC_AUDIT_RUNTIME_WORKER
        ),
        image_reference=parent_package.artifacts.image_reference,
        image_digest_sha256=parent_package.artifacts.image_digest_sha256,
    )
    plan = build_m03r_v12_posthoc_audit_package_plan(artifacts, parent)
    plan_file_sha256 = write_m03r_v12_posthoc_audit_package_plan(
        package_root / "plans" / "audit-plan.json",
        plan,
    )
    load_m03r_v12_posthoc_audit_package_plan(
        package_root / "plans" / "audit-plan.json",
        expected_file_sha256=plan_file_sha256,
    )
    unsigned = {
        "schema": M03R_V12_POSTHOC_AUDIT_BUILD_RECEIPT_SCHEMA,
        "protocol_sha256": M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
        "package_plan_sha256": plan.package_plan_sha256,
        "package_plan_file_sha256": plan_file_sha256,
        "source_archive_sha256": source_archive_sha256,
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "parent_package_plan_sha256": parent.package_plan_sha256,
        "parent_checkpoint_count": len(parent.checkpoint_bindings),
        "indexed_completions": 3,
        "h100s_per_completion": 1,
        "maximum_h100_requests": 3,
        "training_authorized": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256(unsigned)}
    _write(output / "package-build-receipt.json", receipt)
    return receipt


def build_m03r_v12_posthoc_audit_transfer_archive(
    package_root: str | Path,
    archive_path: str | Path,
) -> str:
    root = Path(package_root)
    target = Path(archive_path)
    if target.exists():
        raise M03RV12PosthocAuditPackageBuildError(
            "audit transfer archive already exists"
        )
    files = tuple(path for path in sorted(root.rglob("*")) if path.is_file())
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = tarfile.TarInfo(f"{M03R_V12_POSTHOC_AUDIT_TRANSFER_ROOT}/{relative}")
            info.size = path.stat().st_size
            info.mode = 0o440
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    return _file_sha256(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-root", required=True)
    build.add_argument("--parent-package-plan", required=True)
    build.add_argument("--parent-package-plan-file-sha256", required=True)
    build.add_argument("--parent-output-root", required=True)
    build.add_argument("--output-root", required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--package-root", required=True)
    archive.add_argument("--archive-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        value: Any = build_m03r_v12_posthoc_audit_local_package(
            source_root=args.source_root,
            parent_package_plan_path=args.parent_package_plan,
            parent_package_plan_file_sha256=(
                args.parent_package_plan_file_sha256
            ),
            parent_output_root=args.parent_output_root,
            output_root=args.output_root,
        )
    else:
        value = {
            "archive_sha256": build_m03r_v12_posthoc_audit_transfer_archive(
                args.package_root, args.archive_path
            )
        }
    print(json.dumps(value, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V12_POSTHOC_AUDIT_BUILD_RECEIPT_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_SOURCE_MANIFEST_SCHEMA",
    "M03RV12PosthocAuditPackageBuildError",
    "build_m03r_v12_posthoc_audit_local_package",
    "build_m03r_v12_posthoc_audit_transfer_archive",
    "main",
]
