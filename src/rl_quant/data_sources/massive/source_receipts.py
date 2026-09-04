"""Create-only byte authority for Massive source objects."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import BinaryIO, Iterator
import uuid

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)


MASSIVE_SOURCE_OBJECT_SCHEMA = "rl-quant.massive-source-object-v1"
MASSIVE_SOURCE_COMMIT_SCHEMA = "rl-quant.massive-source-commit-v1"
MASSIVE_LOADED_SOURCE_OBJECT_SCHEMA = "rl-quant.massive-loaded-source-object-v1"


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


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


def _open_parent_directory(
    root: Path, relative_path: str, *, create: bool
) -> tuple[int, str]:
    try:
        current = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise MassiveSourceObjectError(
            "source root must be a no-follow directory"
        ) from exc
    parts = PurePosixPath(relative_path).parts
    try:
        for component in parts[:-1]:
            if create:
                try:
                    os.mkdir(component, 0o755, dir_fd=current)
                except FileExistsError:
                    pass
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            except OSError as exc:
                raise MassiveSourceObjectError(
                    "source path traverses a non-directory or symlink"
                ) from exc
            os.close(current)
            current = child
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def _read_regular_at(directory_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise MassiveSourceObjectError(f"cannot open no-follow artifact: {name}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise MassiveSourceObjectError(f"not a regular artifact: {name}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        return data, info
    finally:
        os.close(descriptor)


def _exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _canonical_write_once_at(directory_fd: int, name: str, payload: object) -> tuple[str, tuple[int, int]]:
    data = canonical_json_file_bytes(payload)
    if _exists_at(directory_fd, name):
        raise MassiveSourceObjectError(f"immutable artifact already exists: {name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o444, dir_fd=directory_fd)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        info = os.fstat(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    return hashlib.sha256(data).hexdigest(), (info.st_dev, info.st_ino)


def _unlink_owned_at(
    directory_fd: int, name: str, identity: tuple[int, int] | None
) -> None:
    if identity is None:
        return
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (info.st_dev, info.st_ino) == identity:
        os.unlink(name, dir_fd=directory_fd)


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


@dataclass(frozen=True, slots=True)
class LoadedMassiveSourceObject:
    """One committed source transaction reopened from its exact final inode."""

    receipt: MassiveSourceObjectReceipt
    commit: MassiveSourceCommit
    payload_relative_path: str
    payload_device: int
    payload_inode: int
    payload_ctime_ns: int
    verified_at_ms: int
    receipt_sha256: str
    schema: str = MASSIVE_LOADED_SOURCE_OBJECT_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt": asdict(self.receipt),
            "commit": asdict(self.commit),
            "payload_relative_path": self.payload_relative_path,
            "payload_device": self.payload_device,
            "payload_inode": self.payload_inode,
            "payload_ctime_ns": self.payload_ctime_ns,
            "verified_at_ms": self.verified_at_ms,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_LOADED_SOURCE_OBJECT_SCHEMA:
            raise MassiveSourceObjectError("loaded source schema drifted")
        self.receipt.validate()
        self.commit.validate()
        if self.payload_relative_path != self.commit.payload_relative_path:
            raise MassiveSourceObjectError("loaded payload path differs from commit")
        for name in (
            "payload_device",
            "payload_inode",
            "payload_ctime_ns",
            "verified_at_ms",
        ):
            _nonnegative_int(name, getattr(self, name))
        _digest("loaded source receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSourceObjectError("loaded source receipt differs")


def _publish_massive_source_object_unlocked(
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
    if isinstance(block_bytes, bool) or not isinstance(block_bytes, int) or block_bytes <= 0:
        raise MassiveSourceObjectError("stream block size must be positive")
    parent_fd, payload_name = _open_parent_directory(
        destination_root, payload_relative, create=True
    )
    receipt_name = payload_name + ".receipt.json"
    commit_name = payload_name + ".commit.json"
    if any(
        _exists_at(parent_fd, name)
        for name in (payload_name, receipt_name, commit_name)
    ):
        os.close(parent_fd)
        raise MassiveSourceObjectError("source publication target already exists")

    temporary_name: str | None = None
    temporary_identity: tuple[int, int] | None = None
    created: dict[str, tuple[int, int] | None] = {
        payload_name: None,
        receipt_name: None,
        commit_name: None,
    }
    try:
        temporary_name = f".{payload_name}.{uuid.uuid4().hex}.partial"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        digest = hashlib.sha256()
        content_length = 0
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
            info = os.fstat(output.fileno())
            temporary_identity = (info.st_dev, info.st_ino)
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
        relative_receipt = str(
            PurePosixPath(payload_relative).with_name(receipt_name)
        )
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

        os.link(
            temporary_name,
            payload_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        created[payload_name] = temporary_identity
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = None
        payload_descriptor = os.open(payload_name, _READ_FLAGS, dir_fd=parent_fd)
        try:
            final_info = os.fstat(payload_descriptor)
            if (final_info.st_dev, final_info.st_ino) != temporary_identity:
                raise MassiveSourceObjectError(
                    "final payload inode differs from the published temporary"
                )
            final_digest = hashlib.sha256()
            while True:
                block = os.read(payload_descriptor, block_bytes)
                if not block:
                    break
                final_digest.update(block)
            if final_digest.hexdigest() != observed_sha256:
                raise MassiveSourceObjectError("final payload bytes changed before commit")
            os.fchmod(payload_descriptor, 0o444)
        finally:
            os.close(payload_descriptor)
        os.fsync(parent_fd)
        observed_receipt_file_sha256, receipt_identity = _canonical_write_once_at(
            parent_fd, receipt_name, asdict(receipt)
        )
        created[receipt_name] = receipt_identity
        if observed_receipt_file_sha256 != receipt_file_sha256:
            raise MassiveSourceObjectError("receipt file identity drifted")
        os.fsync(parent_fd)
        _, commit_identity = _canonical_write_once_at(
            parent_fd, commit_name, asdict(commit)
        )
        created[commit_name] = commit_identity
        os.fsync(parent_fd)
        return receipt, commit
    except BaseException:
        if temporary_name is not None:
            _unlink_owned_at(parent_fd, temporary_name, temporary_identity)
        for name in (commit_name, receipt_name, payload_name):
            _unlink_owned_at(parent_fd, name, created[name])
        os.fsync(parent_fd)
        raise
    finally:
        os.close(parent_fd)


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
    """Authorize and publish one source transaction under the same writer lock."""

    payload_relative = _safe_object_key(relative_payload_path)
    # The late import preserves the generic source layer while making the V5
    # ownership decision and the create-only write one atomic lock scope.
    from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
        authorize_and_lock_massive_adaptive_rl_source_publication_v5,
    )

    with authorize_and_lock_massive_adaptive_rl_source_publication_v5(
        root=root,
        relative_payload_path=payload_relative,
    ):
        return _publish_massive_source_object_unlocked(
            stream=stream,
            root=root,
            relative_payload_path=payload_relative,
            dataset_id=dataset_id,
            source_object_key=source_object_key,
            requested_at_ms=requested_at_ms,
            downloaded_at_ms=downloaded_at_ms,
            schema_sha256=schema_sha256,
            entitlement_receipt_sha256=entitlement_receipt_sha256,
            committed_at_ms=committed_at_ms,
            etag=etag,
            request_id=request_id,
            expected_physical_sha256=expected_physical_sha256,
            block_bytes=block_bytes,
        )


def load_massive_source_object(
    *, root: str | Path, relative_payload_path: str
) -> tuple[MassiveSourceObjectReceipt, MassiveSourceCommit]:
    """Reopen one complete source transaction and verify every exact byte."""

    destination_root = Path(root)
    payload_relative = _safe_object_key(relative_payload_path)
    parent_fd, payload_name = _open_parent_directory(
        destination_root, payload_relative, create=False
    )
    receipt_name = payload_name + ".receipt.json"
    commit_name = payload_name + ".commit.json"
    try:
        payload_bytes, payload_info = _read_regular_at(parent_fd, payload_name)
        receipt_bytes, _ = _read_regular_at(parent_fd, receipt_name)
        commit_bytes, _ = _read_regular_at(parent_fd, commit_name)
        receipt = MassiveSourceObjectReceipt(**json.loads(receipt_bytes))
        commit = MassiveSourceCommit(**json.loads(commit_bytes))
        receipt.validate()
        commit.validate()
        if hashlib.sha256(payload_bytes).hexdigest() != receipt.physical_sha256:
            raise MassiveSourceObjectError("published payload bytes changed")
        if payload_info.st_size != receipt.content_length:
            raise MassiveSourceObjectError("published payload size changed")
        if hashlib.sha256(receipt_bytes).hexdigest() != commit.receipt_file_sha256:
            raise MassiveSourceObjectError("published receipt bytes changed")
        if commit.payload_file_sha256 != receipt.physical_sha256:
            raise MassiveSourceObjectError("commit payload identity drifted")
        if commit.source_receipt_sha256 != receipt.receipt_sha256:
            raise MassiveSourceObjectError("commit semantic identity drifted")
        if commit.payload_relative_path != payload_relative:
            raise MassiveSourceObjectError("commit payload path drifted")
        if commit.receipt_relative_path != str(
            PurePosixPath(payload_relative).with_name(receipt_name)
        ):
            raise MassiveSourceObjectError("commit receipt path drifted")
        return receipt, commit
    finally:
        os.close(parent_fd)


def load_massive_source_bundle(
    *,
    root: str | Path,
    relative_payload_path: str,
    verified_at_ms: int,
) -> LoadedMassiveSourceObject:
    """Return committed bytes plus the final inode identity used for replay."""

    receipt, commit = load_massive_source_object(
        root=root, relative_payload_path=relative_payload_path
    )
    payload_relative = _safe_object_key(relative_payload_path)
    parent_fd, payload_name = _open_parent_directory(
        Path(root), payload_relative, create=False
    )
    try:
        payload_bytes, info = _read_regular_at(parent_fd, payload_name)
        if hashlib.sha256(payload_bytes).hexdigest() != receipt.physical_sha256:
            raise MassiveSourceObjectError("loaded source bytes changed after verification")
    finally:
        os.close(parent_fd)
    body = {
        "schema": MASSIVE_LOADED_SOURCE_OBJECT_SCHEMA,
        "receipt": asdict(receipt),
        "commit": asdict(commit),
        "payload_relative_path": payload_relative,
        "payload_device": info.st_dev,
        "payload_inode": info.st_ino,
        "payload_ctime_ns": info.st_ctime_ns,
        "verified_at_ms": _nonnegative_int("verified timestamp", verified_at_ms),
    }
    value = LoadedMassiveSourceObject(
        receipt=receipt,
        commit=commit,
        payload_relative_path=payload_relative,
        payload_device=info.st_dev,
        payload_inode=info.st_ino,
        payload_ctime_ns=info.st_ctime_ns,
        verified_at_ms=verified_at_ms,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


def read_loaded_massive_source_bytes(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> bytes:
    """Read the exact inode previously sealed by ``load_massive_source_bundle``."""

    loaded_source.validate()
    parent_fd, payload_name = _open_parent_directory(
        Path(root), loaded_source.payload_relative_path, create=False
    )
    try:
        payload, info = _read_regular_at(parent_fd, payload_name)
    finally:
        os.close(parent_fd)
    if (info.st_dev, info.st_ino, info.st_ctime_ns) != (
        loaded_source.payload_device,
        loaded_source.payload_inode,
        loaded_source.payload_ctime_ns,
    ):
        raise MassiveSourceObjectError("loaded source inode was replaced")
    if hashlib.sha256(payload).hexdigest() != loaded_source.receipt.physical_sha256:
        raise MassiveSourceObjectError("loaded source bytes were replaced")
    return payload


@contextmanager
def open_loaded_massive_source_stream(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> Iterator[BinaryIO]:
    """Open the exact committed source inode without materializing it in memory."""

    loaded_source.validate()
    parent_fd, payload_name = _open_parent_directory(
        Path(root), loaded_source.payload_relative_path, create=False
    )
    descriptor = -1
    try:
        descriptor = os.open(payload_name, _READ_FLAGS, dir_fd=parent_fd)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise MassiveSourceObjectError("loaded source is not a regular file")
        if (info.st_dev, info.st_ino, info.st_ctime_ns) != (
            loaded_source.payload_device,
            loaded_source.payload_inode,
            loaded_source.payload_ctime_ns,
        ):
            raise MassiveSourceObjectError("loaded source inode was replaced")
        if info.st_size != loaded_source.receipt.content_length:
            raise MassiveSourceObjectError("loaded source size changed")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            try:
                yield stream
            finally:
                final_info = os.fstat(descriptor)
                if (
                    final_info.st_dev,
                    final_info.st_ino,
                    final_info.st_ctime_ns,
                ) != (
                    loaded_source.payload_device,
                    loaded_source.payload_inode,
                    loaded_source.payload_ctime_ns,
                ):
                    raise MassiveSourceObjectError(
                        "loaded source changed while streaming"
                    )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


__all__ = [
    "LoadedMassiveSourceObject",
    "MASSIVE_SOURCE_COMMIT_SCHEMA",
    "MASSIVE_LOADED_SOURCE_OBJECT_SCHEMA",
    "MASSIVE_SOURCE_OBJECT_SCHEMA",
    "MassiveSourceCommit",
    "MassiveSourceObjectError",
    "MassiveSourceObjectReceipt",
    "load_massive_source_object",
    "load_massive_source_bundle",
    "open_loaded_massive_source_stream",
    "publish_massive_source_object",
    "read_loaded_massive_source_bytes",
]
