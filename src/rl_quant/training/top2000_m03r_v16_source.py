"""Exact package-source inventory verification for M03R-v16."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PROTOCOL_SHA256,
)

M03R_V16_SOURCE_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-runtime-source-manifest-v2"
)
_MAX_SOURCE_MANIFEST_BYTES = 32 * 1024**2


class M03RV16SourceError(ValueError):
    """The package-owned executing source tree drifted."""


@dataclass(frozen=True, slots=True)
class M03RV16VerifiedSourceTree:
    source_root: Path
    source_manifest_file_sha256: str
    source_tree_root_sha256: str
    runtime_worker_sha256: str
    file_count: int


def _read_manifest(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16SourceError("V16 source manifest is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_SOURCE_MANIFEST_BYTES
        ):
            raise M03RV16SourceError("V16 source manifest type or size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV16SourceError("V16 source manifest changed while read")
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or file_sha256(path) != expected_file_sha256:
        raise M03RV16SourceError("V16 source manifest hash drifted")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16SourceError("V16 source manifest is malformed") from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise M03RV16SourceError("V16 source manifest is not canonical")
    return payload


def verify_m03r_v16_source_tree(
    source_root: str | Path,
    source_manifest_path: str | Path,
    *,
    expected_source_manifest_file_sha256: str,
    expected_runtime_worker_sha256: str,
) -> M03RV16VerifiedSourceTree:
    """Verify the exact regular-file inventory and executing worker bytes."""

    root = Path(source_root).resolve()
    payload = _read_manifest(
        Path(source_manifest_path), expected_source_manifest_file_sha256
    )
    files = payload.get("files")
    if (
        payload.get("schema") != M03R_V16_SOURCE_MANIFEST_SCHEMA
        or payload.get("protocol_sha256") != M03R_V16_PROTOCOL_SHA256
        or not isinstance(files, list)
        or payload.get("file_count") != len(files)
        or payload.get("runtime_worker")
        != "src/rl_quant/workflows/top2000_m03r_v16_predictive.py"
        or payload.get("structural_builder")
        != "src/rl_quant/workflows/top2000_m03r_v16_structural_build.py"
    ):
        raise M03RV16SourceError("V16 source manifest identity drifted")
    expected: dict[str, tuple[int, str]] = {}
    for row in files:
        if not isinstance(row, dict):
            raise M03RV16SourceError("V16 source inventory row is malformed")
        relative = row.get("path")
        size = row.get("size")
        digest = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise M03RV16SourceError("V16 source inventory row drifted")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts or relative in expected:
            raise M03RV16SourceError("V16 source inventory path is unsafe")
        expected[relative] = (size, digest)
    observed: set[str] = set()
    for path in root.rglob("*"):
        status = path.lstat()
        if path.is_symlink() or not (stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)):
            raise M03RV16SourceError("V16 source tree contains a special member")
        if stat.S_ISREG(status.st_mode):
            relative = path.relative_to(root).as_posix()
            if relative not in expected:
                raise M03RV16SourceError("V16 source tree contains an extra file")
            expected_size, expected_sha = expected[relative]
            if status.st_size != expected_size or file_sha256(path) != expected_sha:
                raise M03RV16SourceError("V16 source tree member drifted")
            observed.add(relative)
    if observed != set(expected):
        raise M03RV16SourceError("V16 source tree inventory is incomplete")
    worker_relative = str(payload["runtime_worker"])
    structural_builder_relative = str(payload["structural_builder"])
    if worker_relative not in expected or structural_builder_relative not in expected:
        raise M03RV16SourceError("V16 executable source inventory is incomplete")
    worker_sha = expected[worker_relative][1]
    if worker_sha != expected_runtime_worker_sha256:
        raise M03RV16SourceError("V16 runtime worker identity drifted")
    rows = tuple(
        {"path": path, "size": expected[path][0], "sha256": expected[path][1]}
        for path in sorted(expected)
    )
    return M03RV16VerifiedSourceTree(
        source_root=root,
        source_manifest_file_sha256=expected_source_manifest_file_sha256,
        source_tree_root_sha256=semantic_sha256(rows),
        runtime_worker_sha256=worker_sha,
        file_count=len(rows),
    )


__all__ = [
    "M03R_V16_SOURCE_MANIFEST_SCHEMA",
    "M03RV16SourceError",
    "M03RV16VerifiedSourceTree",
    "verify_m03r_v16_source_tree",
]
