#!/usr/bin/env python3
"""Build a fresh immutable local package for M03R-v8 pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v8_pretraining_package import (
    M03RV8PretrainingArtifactBindings,
    build_m03r_v8_pretraining_package_plan,
    package_plan_file_payload,
)

SCHEMA = "rl-quant.top2000-dev.m03r-v8-pretraining-local-package-v1"
IMAGE_REFERENCE = (
    "hpcharbor.mdanderson.edu/yding41/ml2:quanttrade-ppo-cu124-py311-85cf781d3e08"
    "@sha256:7cff8faedcfb44ad25e1001d7e1634569f7cd3f5365bbd8ff8caa9b10d8bcdf9"
)
IMAGE_DIGEST = "7cff8faedcfb44ad25e1001d7e1634569f7cd3f5365bbd8ff8caa9b10d8bcdf9"


class PackageBuildError(RuntimeError):
    """The local package input or immutable publication is unsafe."""


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise PackageBuildError(f"input is not a regular file: {path}")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> str:
    encoded = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _copy_regular(source: Path, destination: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise PackageBuildError(f"source is not a regular non-symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PackageBuildError(f"destination already exists: {destination}")
    shutil.copyfile(source, destination, follow_symlinks=False)
    os.chmod(destination, 0o600)
    return _file_sha256(destination)


def _source_inventory(repo: Path) -> tuple[Path, ...]:
    roots = (repo / "pyproject.toml", repo / "uv.lock")
    python = tuple(sorted((repo / "src" / "rl_quant").rglob("*.py")))
    files = (*roots, *python)
    if not python or any(path.is_symlink() or not path.is_file() for path in files):
        raise PackageBuildError("runtime source inventory is incomplete or unsafe")
    return files


def _deterministic_tar(source_root: Path, target: Path, top_level: str) -> str:
    with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise PackageBuildError(f"archive member is unsafe: {path}")
            relative = path.relative_to(source_root)
            info = archive.gettarinfo(str(path), arcname=str(Path(top_level) / relative))
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            info.mode = 0o555 if path.is_dir() else 0o444
            if path.is_dir():
                archive.addfile(info)
            else:
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    return _file_sha256(target)


def build_package(
    *,
    repo: Path,
    output: Path,
    cache: Path,
    parent_cache_manifest: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise PackageBuildError("fresh package output already exists")
    output.mkdir(parents=True, mode=0o700)
    package = output / "package"
    source_root = package / "source"
    source_rows: list[dict[str, Any]] = []
    for source in _source_inventory(repo):
        relative = source.relative_to(repo)
        destination = source_root / relative
        digest = _copy_regular(source, destination)
        source_rows.append(
            {"path": relative.as_posix(), "size": source.stat().st_size, "sha256": digest}
        )
    source_manifest_payload = {
        "schema": "rl-quant.top2000-dev.m03r-v8-runtime-source-manifest-v1",
        "protocol_sha256": M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
        "file_count": len(source_rows),
        "files": source_rows,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    source_manifest_sha256 = _write(
        package / "source-manifest.json", source_manifest_payload
    )
    source_archive_sha256 = _deterministic_tar(
        source_root, package / "source.tar", "source"
    )
    cache_sha256 = _copy_regular(
        cache, package / "cache" / "top2000-daily-bars.pt"
    )
    if cache_sha256 != "0ba73414c3adea7712f7a68b1e76d934a17694a27671f35b8aa191bcc6aa1ee0":
        raise PackageBuildError("cache bytes differ from the reviewed pre-2026 cache")
    parent_cache_manifest_sha256 = _file_sha256(parent_cache_manifest)
    cache_manifest_sha256 = _write(
        package / "cache-manifest.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v8-cache-lineage-v1",
            "cache_artifact_sha256": cache_sha256,
            "parent_cache_manifest_path": str(parent_cache_manifest),
            "parent_cache_manifest_sha256": parent_cache_manifest_sha256,
            "contains_2026_lockbox": False,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        },
    )
    dependency_lock_sha256 = _file_sha256(repo / "uv.lock")
    worker_source_sha256 = _file_sha256(
        repo / "src/rl_quant/workflows/top2000_m03r_v8_pretraining.py"
    )
    artifacts = M03RV8PretrainingArtifactBindings(
        source_archive_sha256=source_archive_sha256,
        source_manifest_sha256=source_manifest_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        cache_artifact_sha256=cache_sha256,
        cache_manifest_sha256=cache_manifest_sha256,
        worker_source_sha256=worker_source_sha256,
        image_reference=IMAGE_REFERENCE,
        image_digest_sha256=IMAGE_DIGEST,
    )
    package_plan = build_m03r_v8_pretraining_package_plan(artifacts=artifacts)
    for plan in package_plan.plans:
        _write(package / "plans" / f"setting-{plan.setting_index:02d}.json", asdict(plan))
    package_plan_file_sha256 = _write(
        package / "package-plan.json", package_plan_file_payload(package_plan)
    )
    inventory = []
    for path in sorted(package.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": path.relative_to(package).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    execution_manifest_sha256 = _write(
        package / "execution-manifest.json",
        {
            "schema": SCHEMA,
            "protocol_sha256": M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
            "package_plan_sha256": package_plan.package_plan_sha256,
            "package_plan_file_sha256": package_plan_file_sha256,
            "artifacts": asdict(artifacts),
            "inventory_before_execution_manifest": inventory,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        },
    )
    for path in sorted(package.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(package, 0o555)
    receipt = {
        "schema": SCHEMA,
        "package_root": str(package),
        "package_plan_sha256": package_plan.package_plan_sha256,
        "package_plan_file_sha256": package_plan_file_sha256,
        "source_archive_sha256": source_archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "cache_artifact_sha256": cache_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "worker_source_sha256": worker_source_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
        "image_reference": IMAGE_REFERENCE,
        "image_digest_sha256": IMAGE_DIGEST,
        "source_file_count": len(source_rows),
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    _write(output / "package-build-receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent-cache-manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = build_package(
        repo=args.repo.resolve(),
        output=args.output.resolve(),
        cache=args.cache.resolve(),
        parent_cache_manifest=args.parent_cache_manifest.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
