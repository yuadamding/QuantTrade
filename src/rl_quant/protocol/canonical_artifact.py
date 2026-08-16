"""One canonical JSON and digest contract for governed artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_payload(value: Any) -> bytes:
    """Return compact, sorted, finite UTF-8 JSON without a trailing newline."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_json_file_bytes(value: Any) -> bytes:
    """Return the canonical file representation with exactly one newline."""

    return canonical_json_payload(value) + b"\n"


def semantic_sha256(value: Any) -> str:
    """Hash the semantic payload rather than its on-disk line terminator."""

    return hashlib.sha256(canonical_json_payload(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash the exact regular-file bytes at *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "canonical_json_file_bytes",
    "canonical_json_payload",
    "file_sha256",
    "semantic_sha256",
]
