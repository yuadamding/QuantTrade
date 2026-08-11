#!/usr/bin/env python3
"""Independently validate an immutable M03R-v8 pretraining package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.training.top2000_m03r_v8_pretraining_package import (
    load_m03r_v8_pretraining_package_plan,
)


class PackageValidationError(RuntimeError):
    """The package bytes or their immutable lineage are inconsistent."""


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


def _read_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        first = os.fstat(descriptor)
        if not stat.S_ISREG(first.st_mode):
            raise PackageValidationError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        second = os.fstat(descriptor)
        if (
            first.st_dev,
            first.st_ino,
            first.st_size,
            first.st_mtime_ns,
        ) != (
            second.st_dev,
            second.st_ino,
            second.st_size,
            second.st_mtime_ns,
        ):
            raise PackageValidationError(f"file changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular(path)).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    raw = _read_regular(path)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PackageValidationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise PackageValidationError(f"JSON is not canonical: {path}")
    return value


def _validate_tar(package: Path, source_rows: list[dict[str, Any]]) -> None:
    expected = {
        "source/" + str(row["path"]): (int(row["size"]), str(row["sha256"]))
        for row in source_rows
    }
    observed: dict[str, tuple[int, str]] = {}
    with tarfile.open(package / "source.tar", "r:") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or member.issym()
                or member.islnk()
                or (not member.isfile() and not member.isdir())
            ):
                raise PackageValidationError(f"unsafe archive member: {member.name}")
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None or member.name in observed:
                    raise PackageValidationError("archive member cannot be read uniquely")
                raw = stream.read()
                observed[member.name] = (len(raw), hashlib.sha256(raw).hexdigest())
    if observed != expected:
        raise PackageValidationError("source archive inventory differs from manifest")


def validate(package: Path, receipt_path: Path) -> dict[str, Any]:
    if package.is_symlink() or not package.is_dir():
        raise PackageValidationError("package root must be a real directory")
    receipt = _json(receipt_path)
    if Path(str(receipt["package_root"])).resolve() != package.resolve():
        raise PackageValidationError("build receipt points at another package root")
    execution = _json(package / "execution-manifest.json")
    source = _json(package / "source-manifest.json")
    if execution.get("artifacts") != {
        key: receipt[key]
        for key in (
            "source_archive_sha256",
            "source_manifest_sha256",
            "dependency_lock_sha256",
            "cache_artifact_sha256",
            "cache_manifest_sha256",
            "worker_source_sha256",
            "image_reference",
            "image_digest_sha256",
        )
    }:
        raise PackageValidationError("execution artifact bindings differ from receipt")
    required_hashes = {
        "source.tar": receipt["source_archive_sha256"],
        "source-manifest.json": receipt["source_manifest_sha256"],
        "cache/top2000-daily-bars.pt": receipt["cache_artifact_sha256"],
        "cache-manifest.json": receipt["cache_manifest_sha256"],
        "source/uv.lock": receipt["dependency_lock_sha256"],
        "source/src/rl_quant/workflows/top2000_m03r_v8_pretraining.py": receipt[
            "worker_source_sha256"
        ],
        "package-plan.json": receipt["package_plan_file_sha256"],
        "execution-manifest.json": receipt["execution_manifest_sha256"],
    }
    for relative, expected in required_hashes.items():
        if _sha256(package / relative) != expected:
            raise PackageValidationError(f"hash mismatch: {relative}")
    rows = source.get("files")
    if not isinstance(rows, list) or len(rows) != receipt["source_file_count"]:
        raise PackageValidationError("source manifest count is inconsistent")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise PackageValidationError("source manifest row is malformed")
        path = package / "source" / str(row["path"])
        if path.stat().st_size != row["size"] or _sha256(path) != row["sha256"]:
            raise PackageValidationError(f"source manifest mismatch: {row['path']}")
    _validate_tar(package, rows)
    inventory = execution.get("inventory_before_execution_manifest")
    if not isinstance(inventory, list):
        raise PackageValidationError("execution inventory is absent")
    expected_inventory = {
        str(row["path"]): (int(row["size"]), str(row["sha256"])) for row in inventory
    }
    actual_inventory = {
        path.relative_to(package).as_posix(): (path.stat().st_size, _sha256(path))
        for path in package.rglob("*")
        if path.is_file() and path.name != "execution-manifest.json"
    }
    if actual_inventory != expected_inventory:
        raise PackageValidationError("package inventory differs from execution manifest")
    if any(path.is_symlink() for path in package.rglob("*")):
        raise PackageValidationError("package contains a symlink")
    if any(path.stat().st_mode & 0o222 for path in (package, *package.rglob("*"))):
        raise PackageValidationError("package contains writable content")
    plan = load_m03r_v8_pretraining_package_plan(
        package / "package-plan.json",
        expected_package_plan_sha256=str(receipt["package_plan_sha256"]),
    )
    for training_plan in plan.plans:
        row = _json(package / "plans" / f"setting-{training_plan.setting_index:02d}.json")
        if row != {
            field: getattr(training_plan, field)
            for field in training_plan.__dataclass_fields__
        }:
            raise PackageValidationError("standalone plan differs from package plan")
    return {
        "validated": True,
        "package_plan_sha256": plan.package_plan_sha256,
        "package_plan_file_sha256": receipt["package_plan_file_sha256"],
        "execution_manifest_sha256": receipt["execution_manifest_sha256"],
        "source_archive_sha256": receipt["source_archive_sha256"],
        "source_file_count": receipt["source_file_count"],
        "plan_count": len(plan.plans),
        "remote_accessed": False,
        "remote_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.package, args.receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
