"""Explicit, read-only relocation of byte-identical original source evidence.

An externally hash-bound package/path map supplies locations, never identity or
availability authority. Original absolute references and capture hashes remain
unchanged. There is no prefix rewrite, original-path fallback, or write API.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport

SCHEMA = "rl-quant.raw-second-evidence-relocation-v1"
MAX_FILES = 64
MAX_FILE_BYTES = 192 * 1024**2
MAX_PACKAGE_BYTES = 512 * 1024**2
MAP_NAME = "evidence/path-map.json"
_FALSE = ("training_ready", "training_authorized", "remote_native_replay_qualified",
          "native_catalog_activated", "original_references_rewritten")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _digest(value: str) -> None:
    _require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None,
             "Evidence digest is not canonical SHA256")


def _name(value: str, *, absolute: bool = False) -> str:
    _require(isinstance(value, str) and bool(value) and "\\" not in value and "\x00" not in value,
             "Unsafe evidence path")
    parts = value.split("/")
    _require(value.startswith("/") == absolute and (not absolute or len(parts) > 2)
             and all(p not in ("", ".", "..") for p in (parts[1:] if absolute else parts)),
             "Noncanonical evidence path")
    return value


def _directory(path: Path) -> None:
    info = path.lstat()
    _require(path.is_absolute() and path.resolve() == path and stat.S_ISDIR(info.st_mode)
             and info.st_uid == os.getuid(), "Unsafe evidence directory")


def _read(path: Path, cap: int) -> bytes:
    # Include the current directory entry in the existing open-file race check.
    before = path.lstat()
    raw = transport.read_regular(path, cap)
    after = path.lstat()
    _require(all(getattr(before, key) == getattr(after, key)
                 for key in ("st_dev", "st_ino", "st_ctime_ns", "st_mtime_ns", "st_size")),
             "Evidence directory entry changed during read")
    return raw


@dataclass(frozen=True)
class EvidenceRelocation:
    """One explicit replay-only package, externally pinned before source access.

    The old byte-publication map's nonauthorizing flags remain unchanged. This
    typed request opts into location resolution, not scientific promotion.
    """

    package_root: str
    inventory_sha256: str
    path_map_sha256: str
    plan_sha256: str
    completion_sha256: str
    capture_relative_path: str = "capture"

    def _metadata(self) -> tuple[Path, dict, dict]:
        _name(self.package_root, absolute=True)
        _name(self.capture_relative_path)
        for value in (self.inventory_sha256, self.path_map_sha256, self.plan_sha256,
                      self.completion_sha256):
            _digest(value)
        root = Path(self.package_root)
        _directory(root)
        raw = _read(root / "inventory.json", 1024**2)
        _require(transport.digest(raw) == self.inventory_sha256, "Relocation inventory changed")
        inventory = transport.parse_json(raw)
        _require(inventory.get("schema") in ("qt200-direct-canary-byte-inventory-v1",
                 "rl-quant.raw-second-evidence-package-v1")
                 and all(inventory.get(k) is False for k in _FALSE), "Evidence package flags differ")
        rows = inventory.get("files")
        _require(isinstance(rows, list) and 1 <= len(rows) < MAX_FILES, "Unbounded evidence inventory")
        expected = {}
        for row in rows:
            _require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "Invalid evidence file proof")
            name = _name(row["path"])
            _digest(row["sha256"])
            _require(name not in expected and name != "inventory.json"
                     and type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_FILE_BYTES,
                     "Duplicate/oversized evidence file")
            expected[name] = row
        _require(list(expected) == sorted(expected)
                 and sum(r["bytes"] for r in rows) + len(raw) <= MAX_PACKAGE_BYTES,
                 "Unsorted or oversized evidence package")
        _require(MAP_NAME in expected and expected[MAP_NAME]["sha256"] == self.path_map_sha256,
                 "Path map is not bound by the package inventory")
        map_raw = _read(root / MAP_NAME, 1024**2)
        _require(len(map_raw) == expected[MAP_NAME]["bytes"]
                 and transport.digest(map_raw) == self.path_map_sha256, "Evidence path map changed")
        mapping = transport.parse_json(map_raw)
        _require(mapping.get("native_path_resolution_authorized") is False
                 and all(mapping.get(k) is False for k in _FALSE), "Path map is not byte-only evidence")
        entries = mapping.get("files")
        _require(isinstance(entries, list) and 1 <= len(entries) < MAX_FILES, "Unbounded evidence mapping")
        mapped, destinations = {}, set()
        for row in entries:
            _require(type(row) is dict and set(row) == {"original_path", "published_path", "bytes", "sha256"},
                     "Invalid evidence mapping")
            original = _name(row["original_path"], absolute=True)
            name = _name(row["published_path"])
            _require(original not in mapped and name not in destinations and name in expected
                     and type(row["bytes"]) is int and row["bytes"] == expected[name]["bytes"]
                     and row["sha256"] == expected[name]["sha256"], "Ambiguous or unbound evidence mapping")
            mapped[original] = row
            destinations.add(name)
        by_name = {row["published_path"]: row for row in mapped.values()}
        plan_name, complete_name = (self.capture_relative_path + "/" + name
                                    for name in ("plan.json", "COMPLETE.json"))
        _require(plan_name in by_name and complete_name in by_name
                 and by_name[plan_name]["sha256"] == self.plan_sha256
                 and by_name[complete_name]["sha256"] == self.completion_sha256
                 and str(Path(by_name[plan_name]["original_path"]).parent)
                 == str(Path(by_name[complete_name]["original_path"]).parent),
                 "Relocation map does not bind this original capture")
        return root, expected, mapped

    def read(self, original_path: Path, expected_sha256: str, cap: int) -> bytes:
        """Read only the explicit mapped copy, even if an original path exists."""
        _require(type(cap) is int and 0 < cap <= MAX_FILE_BYTES, "Invalid evidence read bound")
        _digest(expected_sha256)
        root, _, mapped = self._metadata()
        row = mapped.get(_name(str(original_path), absolute=True))
        _require(row is not None and row["sha256"] == expected_sha256 and row["bytes"] <= cap,
                 "Original evidence has no exact hash-bound relocation")
        raw = _read(root / row["published_path"], cap)
        _require(len(raw) == row["bytes"] and transport.digest(raw) == expected_sha256,
                 "Relocated evidence bytes changed")
        return raw

    def names(self, original_directory: Path) -> set[str]:
        """Exact original supplemental population, not the destination siblings."""
        _, _, mapped = self._metadata()
        prefix = _name(str(original_directory), absolute=True) + "/"
        names = {name[len(prefix):] for name in mapped if name.startswith(prefix)}
        _require(names and all("/" not in name for name in names), "Unmapped or nested original evidence directory")
        return names

    def verify(self, *, capture_root: Path, plan_sha256: str, completion_sha256: str) -> dict:
        root, expected, _ = self._metadata()
        _require(capture_root == root / self.capture_relative_path
                 and plan_sha256 == self.plan_sha256 and completion_sha256 == self.completion_sha256,
                 "Relocation is for a different immutable capture")
        children = {"": set()}
        for name in (*expected, "inventory.json"):
            parts = name.split("/")
            for index, part in enumerate(parts):
                children.setdefault("/".join(parts[:index]), set()).add(part)
                if index < len(parts) - 1:
                    children.setdefault("/".join(parts[:index + 1]), set())
        for name, members in children.items():
            directory = root / name
            _directory(directory)
            _require({p.name for p in directory.iterdir()} == members, "Evidence package population differs")
        for name, row in expected.items():
            raw = _read(root / name, MAX_FILE_BYTES)
            _require(len(raw) == row["bytes"] and transport.digest(raw) == row["sha256"],
                     "Evidence package payload changed")
        # Recheck pinned metadata after the bounded complete read.
        _require(self._metadata()[1] == expected, "Relocation changed during verification")
        return dict(schema=SCHEMA, package_root=self.package_root, inventory_sha256=self.inventory_sha256,
                    path_map_sha256=self.path_map_sha256, plan_sha256=plan_sha256,
                    completion_sha256=completion_sha256, exact_bytes_verified=True,
                    original_references_rewritten=False, provider_requests_issued=0,
                    continuous_identity_qualified=False, training_ready=False)
