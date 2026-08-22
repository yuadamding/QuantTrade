"""Create-only byte authority for Massive source objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import BinaryIO

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)


MASSIVE_SOURCE_OBJECT_SCHEMA = "rl-quant.massive-source-object-v1"
MASSIVE_SOURCE_COMMIT_SCHEMA = "rl-quant.massive-source-commit-v1"


class MassiveSourceObjectError(ValueError):
    """A Massive source object or publication transaction is invalid."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveSourceObjectError(f"{name} must be a canonical nonempty string")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveSourceObjectError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveSourceObjectError(f"{name} must be a nonnegative integer")
    return value


def _safe_object_key(value: object) -> str:
    key = PurePosixPath(_text("source object key", value))
    if key.is_absolute() or any(part in {"", ".", ".."} for part in key.parts):
        raise MassiveSourceObjectError("source object key must be a safe relative path")
    return key.as_posix()


def _canonical_write_once(path: Path, payload: object) -> str:
    data = canonical_json_file_bytes(payload)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise MassiveSourceObjectError(f"existing immutable artifact differs: {path}")
        return hashlib.sha256(data).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class MassiveSourceObjectReceipt:
    dataset_id: str
    source_object_key: str
    requested_at_ms: int
    downloaded_at_ms: int
    content_length: int
    etag: str | None
    request_id: str | None
    physical_sha256: str
    schema_sha256: str
    entitlement_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_SOURCE_OBJECT_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_SOURCE_OBJECT_SCHEMA:
            raise MassiveSourceObjectError("source-object schema drifted")
        _text("dataset ID", self.dataset_id)
        _safe_object_key(self.source_object_key)
        requested = _nonnegative_int("requested timestamp", self.requested_at_ms)
        downloaded = _nonnegative_int("downloaded timestamp", self.downloaded_at_ms)
        if downloaded < requested:
            raise MassiveSourceObjectError("download predates its request")
        _nonnegative_int("content length", self.content_length)
        for name in ("etag", "request_id"):
            value = getattr(self, name)
            if value is not None:
                _text(name, value)
        for name in (
            "physical_sha256",
            "schema_sha256",
            "entitlement_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSourceObjectError("source-object receipt differs from payload")


@dataclass(frozen=True, slots=True)
class MassiveSourceCommit:
    payload_relative_path: str
    payload_file_sha256: str
    receipt_relative_path: str
    receipt_file_sha256: str
    source_receipt_sha256: str
    committed_at_ms: int
    receipt_sha256: str
    schema: str = MASSIVE_SOURCE_COMMIT_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_SOURCE_COMMIT_SCHEMA:
            raise MassiveSourceObjectError("source commit schema drifted")
        _safe_object_key(self.payload_relative_path)
        _safe_object_key(self.receipt_relative_path)
        for name in (
            "payload_file_sha256",
            "receipt_file_sha256",
            "source_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        _nonnegative_int("commit timestamp", self.committed_at_ms)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSourceObjectError("source commit receipt differs from payload")


def publish_massive_source_object(
    *,
    stream: BinaryIO,
    root: str | Path,
    relative_payload_path: str,
    dataset_id: str,
    source_object_key: str,
    requested_at_ms: int,
    downloaded_at_ms: int,
    schema_sha256: str,
    entitlement_receipt_sha256: str,
    committed_at_ms: int,
    etag: str | None = None,
    request_id: str | None = None,
    expected_physical_sha256: str | None = None,
    block_bytes: int = 8 * 1024 * 1024,
) -> tuple[MassiveSourceObjectReceipt, MassiveSourceCommit]:
    """Publish payload, receipt, and commit marker in that exact order."""

    destination_root = Path(root)
    payload_relative = _safe_object_key(relative_payload_path)
    source_key = _safe_object_key(source_object_key)
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise MassiveSourceObjectError("source root must be a no-follow directory")
    payload_path = destination_root / payload_relative
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    if payload_path.parent.is_symlink():
        raise MassiveSourceObjectError("payload parent cannot be a symlink")
    receipt_path = payload_path.with_name(payload_path.name + ".receipt.json")
    commit_path = payload_path.with_name(payload_path.name + ".commit.json")
    if any(path.exists() for path in (payload_path, receipt_path, commit_path)):
        raise MassiveSourceObjectError("source publication target already exists")
    if isinstance(block_bytes, bool) or not isinstance(block_bytes, int) or block_bytes <= 0:
        raise MassiveSourceObjectError("stream block size must be positive")

    digest = hashlib.sha256()
    content_length = 0
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{payload_path.name}.", suffix=".partial", dir=payload_path.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as output:
            while True:
                block = stream.read(block_bytes)
                if not block:
                    break
                if not isinstance(block, bytes):
                    raise MassiveSourceObjectError("source stream must yield bytes")
                output.write(block)
                digest.update(block)
                content_length += len(block)
            output.flush()
            os.fsync(output.fileno())
        observed_sha256 = digest.hexdigest()
        if expected_physical_sha256 is not None:
            _digest("expected physical SHA", expected_physical_sha256)
            if observed_sha256 != expected_physical_sha256:
                raise MassiveSourceObjectError("source object physical hash mismatch")
        requested_timestamp = _nonnegative_int(
            "requested timestamp", requested_at_ms
        )
        downloaded_timestamp = _nonnegative_int(
            "downloaded timestamp", downloaded_at_ms
        )
        committed_timestamp = _nonnegative_int("commit timestamp", committed_at_ms)
        if committed_timestamp < downloaded_timestamp:
            raise MassiveSourceObjectError("commit predates completed download")
        body = {
            "schema": MASSIVE_SOURCE_OBJECT_SCHEMA,
            "dataset_id": _text("dataset ID", dataset_id),
            "source_object_key": source_key,
            "requested_at_ms": requested_timestamp,
            "downloaded_at_ms": downloaded_timestamp,
            "content_length": content_length,
            "etag": etag,
            "request_id": request_id,
            "physical_sha256": observed_sha256,
            "schema_sha256": _digest("schema SHA", schema_sha256),
            "entitlement_receipt_sha256": _digest(
                "entitlement receipt", entitlement_receipt_sha256
            ),
        }
        receipt = MassiveSourceObjectReceipt(
            dataset_id=dataset_id,
            source_object_key=source_key,
            requested_at_ms=requested_at_ms,
            downloaded_at_ms=downloaded_at_ms,
            content_length=content_length,
            etag=etag,
            request_id=request_id,
            physical_sha256=observed_sha256,
            schema_sha256=schema_sha256,
            entitlement_receipt_sha256=entitlement_receipt_sha256,
            receipt_sha256=semantic_sha256(body),
        )
        receipt.validate()
        receipt_file_sha256 = hashlib.sha256(
            canonical_json_file_bytes(asdict(receipt))
        ).hexdigest()
        relative_receipt = receipt_path.relative_to(destination_root).as_posix()
        commit_body = {
            "schema": MASSIVE_SOURCE_COMMIT_SCHEMA,
            "payload_relative_path": payload_relative,
            "payload_file_sha256": observed_sha256,
            "receipt_relative_path": relative_receipt,
            "receipt_file_sha256": receipt_file_sha256,
            "source_receipt_sha256": receipt.receipt_sha256,
            "committed_at_ms": committed_timestamp,
        }
        commit = MassiveSourceCommit(
            payload_relative_path=payload_relative,
            payload_file_sha256=observed_sha256,
            receipt_relative_path=relative_receipt,
            receipt_file_sha256=receipt_file_sha256,
            source_receipt_sha256=receipt.receipt_sha256,
            committed_at_ms=committed_at_ms,
            receipt_sha256=semantic_sha256(commit_body),
        )
        commit.validate()

        os.link(temporary, payload_path)
        temporary.unlink()
        temporary = None
        os.chmod(payload_path, 0o444)
        _fsync_directory(payload_path.parent)
        observed_receipt_file_sha256 = _canonical_write_once(
            receipt_path, asdict(receipt)
        )
        if observed_receipt_file_sha256 != receipt_file_sha256:
            raise MassiveSourceObjectError("receipt file identity drifted")
        _fsync_directory(receipt_path.parent)
        _canonical_write_once(commit_path, asdict(commit))
        _fsync_directory(commit_path.parent)
        return receipt, commit
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        for created_path in (commit_path, receipt_path, payload_path):
            created_path.unlink(missing_ok=True)
        if payload_path.parent.is_dir():
            _fsync_directory(payload_path.parent)
        raise


def load_massive_source_object(
    *, root: str | Path, relative_payload_path: str
) -> tuple[MassiveSourceObjectReceipt, MassiveSourceCommit]:
    """Reopen one complete source transaction and verify every exact byte."""

    destination_root = Path(root)
    payload_path = destination_root / _safe_object_key(relative_payload_path)
    receipt_path = payload_path.with_name(payload_path.name + ".receipt.json")
    commit_path = payload_path.with_name(payload_path.name + ".commit.json")
    for path in (payload_path, receipt_path, commit_path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise MassiveSourceObjectError(f"not a no-follow regular file: {path}")
    receipt = MassiveSourceObjectReceipt(**json.loads(receipt_path.read_bytes()))
    commit = MassiveSourceCommit(**json.loads(commit_path.read_bytes()))
    receipt.validate()
    commit.validate()
    if file_sha256(payload_path) != receipt.physical_sha256:
        raise MassiveSourceObjectError("published payload bytes changed")
    if payload_path.stat().st_size != receipt.content_length:
        raise MassiveSourceObjectError("published payload size changed")
    if file_sha256(receipt_path) != commit.receipt_file_sha256:
        raise MassiveSourceObjectError("published receipt bytes changed")
    if commit.payload_file_sha256 != receipt.physical_sha256:
        raise MassiveSourceObjectError("commit payload identity drifted")
    if commit.source_receipt_sha256 != receipt.receipt_sha256:
        raise MassiveSourceObjectError("commit semantic identity drifted")
    if commit.payload_relative_path != payload_path.relative_to(destination_root).as_posix():
        raise MassiveSourceObjectError("commit payload path drifted")
    return receipt, commit


__all__ = [
    "MASSIVE_SOURCE_COMMIT_SCHEMA",
    "MASSIVE_SOURCE_OBJECT_SCHEMA",
    "MassiveSourceCommit",
    "MassiveSourceObjectError",
    "MassiveSourceObjectReceipt",
    "load_massive_source_object",
    "publish_massive_source_object",
]
