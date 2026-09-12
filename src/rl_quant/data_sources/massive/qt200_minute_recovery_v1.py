"""Read-only recovery of minute capture publication, never source promotion.

An SSH acknowledgement can be lost after publication. Authenticate the existing
transaction before deciding whether any new transfer is required. Never repair
or overwrite a partially reserved destination in this reader.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


def _read(path: Path, cap: int) -> bytes:
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("Noncanonical publication input")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or before.st_size > cap):
            raise ValueError("Unsafe publication member")
        body = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_ctime_ns", "st_mtime_ns")
    if len(body) != before.st_size or any(getattr(before, f) != getattr(after, f) for f in fields):
        raise ValueError("Publication changed during read")
    return body


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json(body: bytes) -> dict:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("Duplicate publication key")
            out[k] = v
        return out

    value = json.loads(body, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("Publication is not an object")
    return value


def verify_published_bundle(*, root: Path, ticker: str, inventory_sha256: str,
                            publication_sha256: str | None = None) -> dict:
    """Require actual complete source bytes, even without a returned SSH ACK."""
    inventory_bytes = _read(root / "transfer-inventory.json", 1048576)
    publication_bytes = _read(root / "PUBLICATION.json", 16384)
    if (_sha(inventory_bytes) != inventory_sha256
            or publication_sha256 is not None and _sha(publication_bytes) != publication_sha256):
        raise ValueError("Publication transaction identity differs")
    inv, pub = _json(inventory_bytes), _json(publication_bytes)
    if (inv.get("ticker") != ticker or pub.get("ticker") != ticker
            or pub.get("inventory_sha256") != inventory_sha256
            or inv.get("training_ready") is not False or pub.get("training_ready") is not False
            or pub.get("remote_root") != str(root)):
        raise ValueError("Publication scope differs")
    rows = inv.get("files")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 400:
        raise ValueError("Publication member count differs")
    names, total = set(), 0
    for row in rows:
        name = row["path"]
        p = PurePosixPath(name)
        if (p.is_absolute() or ".." in p.parts or str(p) != name or "\\" in name
                or name in names or name not in ("acquisition-binding.json",) and not name.startswith("capture/")):
            raise ValueError("Unsafe or duplicate publication path")
        names.add(name)
        body = _read(root / name, 35651584)
        if len(body) != row["bytes"] or _sha(body) != row["sha256"]:
            raise ValueError("Published source bytes differ")
        total += len(body)
        if total > 134217728:
            raise ValueError("Publication exceeds rolling bound")
    if not {"acquisition-binding.json", "capture/plan.json", "capture/COMPLETE.json"} <= names:
        raise ValueError("Publication lacks a complete source binding")
    binding_bytes = _read(root / "acquisition-binding.json", 1048576)
    binding = _json(binding_bytes)
    if (_sha(binding_bytes) != inv.get("acquisition_binding_sha256")
            or _sha(binding_bytes) != pub.get("acquisition_binding_sha256")
            or binding.get("ticker") != ticker
            or _sha(_read(root / "capture/plan.json", 1048576)) != binding.get("plan_sha256")
            or _sha(_read(root / "capture/COMPLETE.json", 1048576)) != binding.get("completion_sha256")):
        raise ValueError("Capture binding differs")
    actual = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise ValueError("Symlinked publication directory")
        actual.update((Path(directory) / name).relative_to(root).as_posix() for name in files)
    required = names | {"PUBLICATION.json", "transfer-inventory.json"}
    if not required <= actual or actual - required - {"TRANSFER-DUPLICATE-REMOVED.json", "transfer.tar.gz"}:
        raise ValueError("Unexpected publication members")
    if _read(root / "PUBLICATION.json", 16384) != publication_bytes:
        raise ValueError("Publication changed during audit")
    return dict(ticker=ticker, publication_sha256=_sha(publication_bytes),
                inventory_sha256=inventory_sha256, publication=pub,
                verified_files=len(rows), verified_bytes=total, training_ready=False)


def recovery_partition(*, ordered: tuple[str, ...], published: tuple[str, ...],
                       captured: tuple[str, ...]) -> dict:
    """Disjoint operations; completed captures never become network requests."""
    if (not ordered or len(set(ordered)) != len(ordered)
            or len(set(published)) != len(published) or len(set(captured)) != len(captured)
            or not set(published) <= set(captured) <= set(ordered)):
        raise ValueError("Invalid recovery population")
    if tuple(x for x in ordered if x in published) != published or tuple(x for x in ordered if x in captured) != captured:
        raise ValueError("Recovery order differs from frozen study")
    return dict(reuse_publications=published,
                publish_existing_captures=tuple(x for x in ordered if x in captured and x not in published),
                acquire_missing=tuple(x for x in ordered if x not in captured), training_ready=False)
