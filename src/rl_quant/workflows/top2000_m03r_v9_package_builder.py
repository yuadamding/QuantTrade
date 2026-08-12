"""Build and validate one immutable local M03R-v9 predictive package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v9_package import (
    M03RV9PackageArtifacts,
    build_m03r_v9_package_plan,
    load_m03r_v9_package_plan,
    package_plan_file_payload,
)

M03R_V9_LOCAL_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-local-predictive-package-v1"
)
M03R_V9_SOURCE_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-runtime-source-manifest-v1"
)
M03R_V9_EXECUTION_MANIFEST_SCHEMA = "rl-quant.top2000-dev.m03r-v9-execution-manifest-v1"
M03R_V9_TRANSFER_ARCHIVE_SCHEMA = "rl-quant.top2000-dev.m03r-v9-transfer-archive-v1"
M03R_V9_TRANSFER_ROOT = "qt-m03r-v9-predictive-package-v1"
PINNED_QUANTTRADE_IMAGE = (
    "hpcharbor.mdanderson.edu/yding41/ml2:quanttrade-ppo-cu124-py311-85cf781d3e08"
    "@sha256:7cff8faedcfb44ad25e1001d7e1634569f7cd3f5365bbd8ff8caa9b10d8bcdf9"
)
M03R_V9_RUNTIME_WORKER = "src/rl_quant/workflows/top2000_m03r_v9_predictive.py"


class M03RV9PackageBuildError(ValueError):
    """A package input, member, hash, or no-clobber boundary drifted."""


def _canonical_bytes(value: Any) -> bytes:
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular(path: Path, *, name: str) -> Path:
    try:
        status = path.lstat()
    except OSError as exc:
        raise M03RV9PackageBuildError(f"{name} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise M03RV9PackageBuildError(f"{name} must be a nonempty regular file")
    return path


def _write_exclusive(path: Path, data: bytes, *, mode: int = 0o440) -> str:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(data)
        sink.flush()
        os.fsync(sink.fileno())
    return hashlib.sha256(data).hexdigest()


def _copy_exclusive(source: Path, target: Path) -> str:
    _require_regular(source, name=str(source))
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as sink:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                digest.update(block)
                sink.write(block)
            sink.flush()
            os.fsync(sink.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != _file_sha256(source):
        raise M03RV9PackageBuildError("source changed while copying")
    return digest.hexdigest()


def _source_inventory(source_root: Path) -> tuple[Path, ...]:
    root = source_root.resolve()
    required = (
        root / "pyproject.toml",
        root / "uv.lock",
        root / M03R_V9_RUNTIME_WORKER,
    )
    for path in required:
        _require_regular(path, name=str(path.relative_to(root)))
    files = (
        root / "pyproject.toml",
        root / "uv.lock",
        *(root / "src/rl_quant").rglob("*.py"),
    )
    resolved: list[Path] = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        _require_regular(path, name=str(path.relative_to(root)))
        if path.resolve().is_relative_to(root):
            resolved.append(path)
        else:
            raise M03RV9PackageBuildError("source member leaves the repository root")
    if len(set(resolved)) != len(resolved):
        raise M03RV9PackageBuildError("source inventory contains duplicate members")
    return tuple(resolved)


def _copy_source_tree(
    source_root: Path,
    destination: Path,
    members: tuple[Path, ...],
) -> tuple[dict[str, Any], ...]:
    inventory: list[dict[str, Any]] = []
    for source in members:
        relative = source.relative_to(source_root).as_posix()
        target = destination / relative
        sha = _copy_exclusive(source, target)
        inventory.append(
            {"path": relative, "sha256": sha, "size": target.stat().st_size}
        )
    return tuple(inventory)


def _write_deterministic_source_tar(
    package_root: Path,
    source_inventory: tuple[dict[str, Any], ...],
) -> str:
    target = package_root / "source.tar"
    if target.exists() or target.is_symlink():
        raise M03RV9PackageBuildError("source archive target already exists")
    seen_directories: set[str] = {"source"}
    for row in source_inventory:
        parent = PurePosixPath("source", row["path"]).parent
        while str(parent) not in {".", ""}:
            seen_directories.add(str(parent))
            parent = parent.parent
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as archive:
        for directory in sorted(
            seen_directories, key=lambda item: (item.count("/"), item)
        ):
            info = tarfile.TarInfo(directory + "/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            archive.addfile(info)
        for row in source_inventory:
            member = "source/" + row["path"]
            source = package_root / member
            info = tarfile.TarInfo(member)
            info.size = int(row["size"])
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    os.chmod(target, 0o440)
    return _file_sha256(target)


def _safe_tar_inventory(path: Path) -> tuple[tuple[str, int, bool], ...]:
    rows: list[tuple[str, int, bool]] = []
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
                or not (member.isfile() or member.isdir())
            ):
                raise M03RV9PackageBuildError("source archive has an unsafe member")
            rows.append((member.name, member.size, member.isfile()))
    return tuple(rows)


def _inventory(root: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _file_sha256(path),
                    "size": path.stat().st_size,
                }
            )
    return tuple(rows)


def build_m03r_v9_local_package(
    *,
    source_root: str | Path,
    cache_path: str | Path,
    cache_manifest_path: str | Path,
    risk_root: str | Path,
    output_root: str | Path,
    image_reference: str = PINNED_QUANTTRADE_IMAGE,
) -> dict[str, Any]:
    """Create one fresh package root; no existing target is reused."""

    source = Path(source_root).resolve()
    cache = _require_regular(Path(cache_path), name="development cache")
    cache_manifest = _require_regular(Path(cache_manifest_path), name="cache manifest")
    risk = Path(risk_root)
    risk_artifact = _require_regular(risk / "risk-exposures.pt", name="risk artifact")
    risk_manifest = _require_regular(
        risk / "risk-source-manifest.json", name="risk source manifest"
    )
    projector_manifest = _require_regular(
        risk / "projector-manifest.json", name="projector manifest"
    )
    output = Path(output_root)
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    package_root = output / "package"
    package_root.mkdir(mode=0o750)

    members = _source_inventory(source)
    source_rows = _copy_source_tree(source, package_root / "source", members)
    source_manifest_payload = {
        "schema": M03R_V9_SOURCE_MANIFEST_SCHEMA,
        "protocol_sha256": M03R_V9_PROTOCOL_SHA256,
        "file_count": len(source_rows),
        "files": source_rows,
        "runtime_worker": M03R_V9_RUNTIME_WORKER,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    source_manifest_sha = _write_exclusive(
        package_root / "source-manifest.json",
        _canonical_bytes(source_manifest_payload),
    )
    source_archive_sha = _write_deterministic_source_tar(package_root, source_rows)

    cache_sha = _copy_exclusive(
        cache,
        package_root / "cache" / "top2000-daily-bars.pt",
    )
    cache_manifest_sha = _copy_exclusive(
        cache_manifest,
        package_root / "cache-manifest.json",
    )
    risk_artifact_sha = _copy_exclusive(
        risk_artifact,
        package_root / "risk" / "risk-exposures.pt",
    )
    risk_manifest_sha = _copy_exclusive(
        risk_manifest,
        package_root / "risk" / "risk-source-manifest.json",
    )
    projector_file_sha = _copy_exclusive(
        projector_manifest,
        package_root / "risk" / "projector-manifest.json",
    )
    projector_payload = json.loads(projector_manifest.read_bytes())
    if not isinstance(projector_payload, dict):
        raise M03RV9PackageBuildError("projector manifest is not an object")
    projector = projector_payload.get("projector")
    binding = projector_payload.get("binding")
    if not isinstance(projector, dict) or not isinstance(binding, dict):
        raise M03RV9PackageBuildError("projector manifest lacks typed payloads")

    image_digest = image_reference.rsplit("@sha256:", 1)[-1]
    artifacts = M03RV9PackageArtifacts(
        source_archive_sha256=source_archive_sha,
        source_manifest_sha256=source_manifest_sha,
        dependency_lock_sha256=_file_sha256(source / "uv.lock"),
        cache_artifact_sha256=cache_sha,
        cache_manifest_sha256=cache_manifest_sha,
        risk_artifact_sha256=risk_artifact_sha,
        risk_source_manifest_file_sha256=risk_manifest_sha,
        projector_manifest_file_sha256=projector_file_sha,
        projector_manifest_sha256=projector["manifest_sha256"],
        projector_binding_sha256=binding["binding_sha256"],
        worker_source_sha256=_file_sha256(source / M03R_V9_RUNTIME_WORKER),
        image_reference=image_reference,
        image_digest_sha256=image_digest,
    )
    package = build_m03r_v9_package_plan(artifacts=artifacts)
    package_plan_file_sha = _write_exclusive(
        package_root / "plans" / "package-plan.json",
        _canonical_bytes(package_plan_file_payload(package)),
    )
    inventory_before_execution = _inventory(package_root)
    execution = {
        "schema": M03R_V9_EXECUTION_MANIFEST_SCHEMA,
        "protocol_sha256": M03R_V9_PROTOCOL_SHA256,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_plan_file_sha,
        "artifacts": asdict(artifacts),
        "inventory_before_execution_manifest": inventory_before_execution,
        "predictive_settings": 3,
        "h100s_per_worker": 2,
        "maximum_h100_requests": 6,
        "economic_optimizer_updates": 0,
        "economic_panel_authorized": False,
        "outer_evaluation_authorized": False,
        "research_only": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    execution_manifest_sha = _write_exclusive(
        package_root / "execution-manifest.json",
        _canonical_bytes(execution),
    )
    final_inventory = _inventory(package_root)
    receipt_unsigned = {
        "schema": M03R_V9_LOCAL_PACKAGE_SCHEMA,
        "package_relative_root": "package",
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_plan_file_sha,
        "source_archive_sha256": source_archive_sha,
        "source_manifest_sha256": source_manifest_sha,
        "dependency_lock_sha256": artifacts.dependency_lock_sha256,
        "cache_artifact_sha256": cache_sha,
        "cache_manifest_sha256": cache_manifest_sha,
        "risk_artifact_sha256": risk_artifact_sha,
        "risk_source_manifest_file_sha256": risk_manifest_sha,
        "projector_manifest_file_sha256": projector_file_sha,
        "projector_manifest_sha256": artifacts.projector_manifest_sha256,
        "projector_binding_sha256": artifacts.projector_binding_sha256,
        "worker_source_sha256": artifacts.worker_source_sha256,
        "execution_manifest_sha256": execution_manifest_sha,
        "image_reference": image_reference,
        "image_digest_sha256": image_digest,
        "file_inventory": final_inventory,
        "source_file_count": len(source_rows),
        "economic_panel_authorized": False,
        "outer_evaluation_authorized": False,
        "research_only": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    receipt = {
        **receipt_unsigned,
        "receipt_sha256": hashlib.sha256(
            _canonical_bytes(receipt_unsigned)
        ).hexdigest(),
    }
    _write_exclusive(
        output / "package-build-receipt.json",
        _canonical_bytes(receipt),
    )
    validate_m03r_v9_local_package(output)
    return receipt


def validate_m03r_v9_local_package(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root)
    receipt_path = _require_regular(
        output / "package-build-receipt.json", name="package build receipt"
    )
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise M03RV9PackageBuildError("package receipt is invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise M03RV9PackageBuildError("package receipt is not an object")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != M03R_V9_LOCAL_PACKAGE_SCHEMA
        or receipt.get("receipt_sha256")
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
        or receipt.get("economic_panel_authorized") is not False
        or receipt.get("outer_evaluation_authorized") is not False
    ):
        raise M03RV9PackageBuildError("package receipt semantics drifted")
    package_root = output / "package"
    if receipt.get("package_relative_root") != "package":
        raise M03RV9PackageBuildError("package root identity drifted")
    inventory = _inventory(package_root)
    if tuple(receipt.get("file_inventory", ())) != inventory:
        raise M03RV9PackageBuildError("package file inventory drifted")
    expected = {row["path"]: row["sha256"] for row in inventory}
    required = {
        "source.tar": "source_archive_sha256",
        "source-manifest.json": "source_manifest_sha256",
        "cache/top2000-daily-bars.pt": "cache_artifact_sha256",
        "cache-manifest.json": "cache_manifest_sha256",
        "risk/risk-exposures.pt": "risk_artifact_sha256",
        "risk/risk-source-manifest.json": "risk_source_manifest_file_sha256",
        "risk/projector-manifest.json": "projector_manifest_file_sha256",
        "plans/package-plan.json": "package_plan_file_sha256",
        "execution-manifest.json": "execution_manifest_sha256",
    }
    if any(
        expected.get(path) != receipt.get(receipt_key)
        for path, receipt_key in required.items()
    ):
        raise M03RV9PackageBuildError("package artifact hash binding drifted")
    plan = load_m03r_v9_package_plan(
        package_root / "plans/package-plan.json",
        expected_package_plan_sha256=receipt["package_plan_sha256"],
    )
    if plan.artifacts.worker_source_sha256 != receipt["worker_source_sha256"]:
        raise M03RV9PackageBuildError("worker source identity drifted")
    archive_rows = _safe_tar_inventory(package_root / "source.tar")
    source_manifest = json.loads((package_root / "source-manifest.json").read_bytes())
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("schema") != M03R_V9_SOURCE_MANIFEST_SCHEMA
        or source_manifest.get("file_count") != receipt["source_file_count"]
        or len([row for row in archive_rows if row[2]]) != receipt["source_file_count"]
    ):
        raise M03RV9PackageBuildError("source archive/manifest inventory drifted")
    return receipt


def build_m03r_v9_transfer_archive(
    output_root: str | Path,
    archive_path: str | Path,
) -> dict[str, Any]:
    """Write one deterministic, relocation-safe transfer archive."""

    output = Path(output_root)
    receipt = validate_m03r_v9_local_package(output)
    target = Path(archive_path)
    if target.exists() or target.is_symlink():
        raise M03RV9PackageBuildError("transfer archive target already exists")
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    files = (output / "package-build-receipt.json",) + tuple(
        output / "package" / row["path"] for row in receipt["file_inventory"]
    )
    directories: set[str] = {M03R_V9_TRANSFER_ROOT}
    for source in files:
        relative = source.relative_to(output).as_posix()
        parent = PurePosixPath(M03R_V9_TRANSFER_ROOT, relative).parent
        while str(parent) not in {".", ""}:
            directories.add(str(parent))
            parent = parent.parent
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as archive:
        for directory in sorted(directories, key=lambda item: (item.count("/"), item)):
            info = tarfile.TarInfo(directory + "/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o750
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            archive.addfile(info)
        for source in files:
            _require_regular(source, name=str(source))
            relative = source.relative_to(output).as_posix()
            member = f"{M03R_V9_TRANSFER_ROOT}/{relative}"
            info = tarfile.TarInfo(member)
            info.size = source.stat().st_size
            info.mode = 0o440
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    os.chmod(target, 0o440)
    archive_sha = _file_sha256(target)
    result = {
        "schema": M03R_V9_TRANSFER_ARCHIVE_SCHEMA,
        "archive_path": str(target),
        "archive_sha256": archive_sha,
        "top_level_directory": M03R_V9_TRANSFER_ROOT,
        "package_build_receipt_file_sha256": _file_sha256(
            output / "package-build-receipt.json"
        ),
        "package_build_receipt_sha256": receipt["receipt_sha256"],
        "regular_member_count": len(files),
        "research_only": True,
        "development_only": True,
        "economic_panel_authorized": False,
    }
    validate_m03r_v9_transfer_archive(
        target,
        expected_archive_sha256=archive_sha,
        expected_package_build_receipt_file_sha256=(
            result["package_build_receipt_file_sha256"]
        ),
    )
    return result


def validate_m03r_v9_transfer_archive(
    archive_path: str | Path,
    *,
    expected_archive_sha256: str,
    expected_package_build_receipt_file_sha256: str,
) -> dict[str, Any]:
    """Validate safe members and every receipt-bound file without extraction."""

    _require_regular(Path(archive_path), name="transfer archive")
    if _file_sha256(Path(archive_path)) != expected_archive_sha256:
        raise M03RV9PackageBuildError("transfer archive SHA-256 drifted")
    receipt_member_name = f"{M03R_V9_TRANSFER_ROOT}/package-build-receipt.json"
    observed_files: dict[str, tuple[int, str]] = {}
    receipt_bytes: bytes | None = None
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
                or not (member.isfile() or member.isdir())
                or not pure.parts
                or pure.parts[0] != M03R_V9_TRANSFER_ROOT
            ):
                raise M03RV9PackageBuildError("transfer archive has an unsafe member")
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise M03RV9PackageBuildError("archive member cannot be read")
                digest = hashlib.sha256()
                payload = bytearray() if member.name == receipt_member_name else None
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                    if payload is not None:
                        payload.extend(block)
                observed_files[member.name] = (member.size, digest.hexdigest())
                if payload is not None:
                    receipt_bytes = bytes(payload)
    if (
        receipt_bytes is None
        or hashlib.sha256(receipt_bytes).hexdigest()
        != expected_package_build_receipt_file_sha256
    ):
        raise M03RV9PackageBuildError("archive package receipt drifted")
    try:
        receipt = json.loads(receipt_bytes)
    except json.JSONDecodeError as exc:
        raise M03RV9PackageBuildError("archive package receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise M03RV9PackageBuildError("archive package receipt is not an object")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != M03R_V9_LOCAL_PACKAGE_SCHEMA
        or receipt.get("receipt_sha256")
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
        or receipt.get("package_relative_root") != "package"
        or receipt.get("economic_panel_authorized") is not False
    ):
        raise M03RV9PackageBuildError("archive receipt semantics drifted")
    expected_files = {
        receipt_member_name: (
            len(receipt_bytes),
            expected_package_build_receipt_file_sha256,
        )
    }
    for row in receipt.get("file_inventory", ()):
        if not isinstance(row, dict):
            raise M03RV9PackageBuildError("archive inventory row is malformed")
        expected_files[f"{M03R_V9_TRANSFER_ROOT}/package/{row['path']}"] = (
            row["size"],
            row["sha256"],
        )
    if observed_files != expected_files:
        raise M03RV9PackageBuildError("transfer archive inventory drifted")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-root", required=True)
    build.add_argument("--cache", required=True)
    build.add_argument("--cache-manifest", required=True)
    build.add_argument("--risk-root", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--image-reference", default=PINNED_QUANTTRADE_IMAGE)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-root", required=True)
    archive = subparsers.add_parser("archive")
    archive.add_argument("--output-root", required=True)
    archive.add_argument("--archive-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        build_m03r_v9_local_package(
            source_root=args.source_root,
            cache_path=args.cache,
            cache_manifest_path=args.cache_manifest,
            risk_root=args.risk_root,
            output_root=args.output_root,
            image_reference=args.image_reference,
        )
    elif args.command == "validate":
        validate_m03r_v9_local_package(args.output_root)
    else:
        build_m03r_v9_transfer_archive(args.output_root, args.archive_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V9_EXECUTION_MANIFEST_SCHEMA",
    "M03R_V9_LOCAL_PACKAGE_SCHEMA",
    "M03R_V9_SOURCE_MANIFEST_SCHEMA",
    "M03R_V9_TRANSFER_ARCHIVE_SCHEMA",
    "M03R_V9_TRANSFER_ROOT",
    "PINNED_QUANTTRADE_IMAGE",
    "M03RV9PackageBuildError",
    "build_m03r_v9_local_package",
    "build_m03r_v9_transfer_archive",
    "main",
    "validate_m03r_v9_local_package",
    "validate_m03r_v9_transfer_archive",
]
