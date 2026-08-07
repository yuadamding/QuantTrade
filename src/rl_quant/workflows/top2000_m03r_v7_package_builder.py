"""Build one immutable local package for the TOP2000 M03R-v7 research Job.

The builder is deliberately local-only.  It owns no Kubernetes client and no
remote transport.  It requires a clean Git worktree, snapshots only stage-zero
regular tracked blobs, validates the exact development cache, and publishes a
new output directory without overwriting any prior evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import torch

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_DESIGN_ID,
    M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA,
    M03R_SEED17_TOP2000_PROTOCOL_GENERATION,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_DATA_ROLE,
    M03R_TOP2000_DEV_DESIGN_ID,
    M03R_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_TOP2000_DEV_SETTING_IDS,
)
from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03R_TOP2000_PACKAGE_SOURCE_PYTHONPATH,
    M03RV7Top2000ArtifactBindings,
    M03RV7Top2000IndexPlan,
    M03RV7Top2000PackagePlan,
    M03RV7Top2000RuntimeProfile,
    build_m03r_v7_top2000_package_plan,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_package import (
    M03RV7Seed17IndexPlan,
    M03RV7Seed17PackagePlan,
    build_m03r_v7_seed17_top2000_package_plan,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    TOP2000_HOLD30_BENCHMARK_ID,
    TOP2000_HOLD30_BENCHMARK_RISK_REPAIR_RULE,
    TOP2000_HOLD30_DEVELOPMENT_ADAPTER_SCHEMA,
    TOP2000_HOLD30_MAX_STOCK_WEIGHT,
    build_top2000_hold30_development_sequence_from_loaded_cache,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_ALPHA_HORIZONS,
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_FOLD_COUNT,
    TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS,
    TOP2000_M03R_V7_DEV_SEEDS,
    TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS,
    TOP2000_M03R_V7_DEV_WARMUP_DECISIONS,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v7_factor_calibration import (
    TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS,
)

PACKAGE_BUILDER_SCHEMA: Final = "rl-quant.top2000-dev.m03r-v7-package-builder-v1"
SOURCE_MANIFEST_SCHEMA: Final = "rl-quant.top2000-dev.source-manifest-v1"
CACHE_MANIFEST_SCHEMA: Final = "rl-quant.top2000-dev.cache-manifest-v1"
DATA_MANIFEST_SCHEMA: Final = "rl-quant.top2000-dev.data-manifest-v1"
DEPENDENCY_MANIFEST_SCHEMA: Final = "rl-quant.top2000-dev.dependency-manifest-v1"
RUNTIME_MANIFEST_SCHEMA: Final = "rl-quant.top2000-dev.runtime-manifest-v1"
EXECUTION_MANIFEST_SCHEMA: Final = "rl-quant.top2000-dev.execution-manifest-v1"
BENCHMARK_PREFLIGHT_SCHEMA: Final = (
    "rl-quant.top2000-dev.seed17-benchmark-feasibility-preflight-v1"
)

PINNED_QUANTTRADE_IMAGE: Final = (
    "hpcharbor.mdanderson.edu/yding41/ml2:"
    "quanttrade-ppo-cu124-py311-85cf781d3e08@sha256:"
    "7cff8faedcfb44ad25e1001d7e1634569f7cd3f5365bbd8ff8caa9b10d8bcdf9"
)
PINNED_QUANTTRADE_IMAGE_DIGEST: Final = (
    "7cff8faedcfb44ad25e1001d7e1634569f7cd3f5365bbd8ff8caa9b10d8bcdf9"
)
PACKAGE_PLAN_CONTAINER_PATH: Final = "/mnt/package/package-plan.json"
PACKAGE_MOUNT_PATH: Final = "/mnt/package"
OUTPUT_MOUNT_PATH: Final = "/mnt/output"
PINNED_IMAGE_PYTHON: Final = "/opt/conda/envs/quanttrade/bin/python"

DEFAULT_CRITICAL_SOURCE_PATHS: Final = (
    "src/rl_quant/evaluation/top2000_m03r_v7_dev.py",
    "src/rl_quant/models/hold30_alpha_m03r_v7_top2000_dev.py",
    "src/rl_quant/protocol/hold30_alpha_m03r_v7_top2000_dev.py",
    "src/rl_quant/training/hold30_alpha_m03r_v7_kubernetes.py",
    "src/rl_quant/training/hold30_alpha_m03r_v7_package.py",
    "src/rl_quant/training/hold30_top2000_development.py",
    "src/rl_quant/training/top2000_m03r_v7_dev.py",
    "src/rl_quant/training/top2000_m03r_v7_factor_calibration.py",
    "src/rl_quant/workflows/top2000_m03r_v7_dev.py",
    "src/rl_quant/workflows/top2000_m03r_v7_package_builder.py",
)
SEED17_CRITICAL_SOURCE_PATHS: Final = DEFAULT_CRITICAL_SOURCE_PATHS + (
    "src/rl_quant/protocol/hold30_alpha_m03r_v7_seed17_top2000_dev.py",
    "src/rl_quant/training/hold30_alpha_m03r_v7_seed17_kubernetes.py",
    "src/rl_quant/training/hold30_alpha_m03r_v7_seed17_package.py",
    "src/rl_quant/training/top2000_m03r_v7_seadragon_lifecycle.py",
    "src/rl_quant/workflows/top2000_m03r_v7_seed17_dev.py",
    "src/rl_quant/workflows/top2000_m03r_v7_seed17_operator.py",
)
ROOT_FILE_ALLOWLIST: Final = frozenset(
    {
        "cache.pt",
        "cache-manifest.json",
        "data-manifest.json",
        "dependency-manifest.json",
        "execution-manifest.json",
        "package-plan.json",
        "runtime-manifest.json",
        "source-manifest.json",
        "source.tar",
    }
)
SEED17_ROOT_FILE_ALLOWLIST: Final = ROOT_FILE_ALLOWLIST | {
    "benchmark-preflight.json"
}


class Top2000M03RV7PackageBuildError(RuntimeError):
    """The repository, cache, output, or generated package failed closed."""


def _require_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV7PackageBuildError(
            f"{name} must be one lowercase SHA-256 digest"
        )


@dataclass(frozen=True, slots=True)
class Top2000M03RV7CacheContract:
    """Exact cache identity required by the package builder."""

    cache_sha256: str
    cache_identity: str
    search_identity: str
    base_dataset_identity: str
    lockbox_partition_names_hash: str
    action_hash: str
    date_hash: str
    state_rows: int
    action_count: int
    first_exchange_date: str
    last_exchange_date: str
    daily_dtype: str = "torch.float32"
    availability_dtype: str = "torch.bool"
    feature_count: int = 5
    bar_seconds: int = 300
    development_only: bool = True
    bars_only: bool = True
    data_role: str = M03R_TOP2000_DEV_DATA_ROLE

    def __post_init__(self) -> None:
        for name in (
            "cache_sha256",
            "cache_identity",
            "search_identity",
            "base_dataset_identity",
            "lockbox_partition_names_hash",
            "action_hash",
            "date_hash",
        ):
            _require_sha256(name, getattr(self, name))
        try:
            first = dt.date.fromisoformat(self.first_exchange_date)
            last = dt.date.fromisoformat(self.last_exchange_date)
        except ValueError as exc:
            raise Top2000M03RV7PackageBuildError(
                "cache contract exchange dates are invalid"
            ) from exc
        if (
            self.state_rows < 2
            or self.action_count < 2
            or self.feature_count != 5
            or self.bar_seconds != 300
            or first >= last
            or last >= dt.date(2026, 1, 1)
            or self.daily_dtype not in {"torch.float32", "torch.float64"}
            or self.availability_dtype != "torch.bool"
            or not self.development_only
            or not self.bars_only
            or self.data_role != "development-only-nonreportable"
        ):
            raise Top2000M03RV7PackageBuildError(
                "cache contract must remain pre-2026, bars-only, and development-only"
            )


CURRENT_TOP2000_CACHE_CONTRACT: Final = Top2000M03RV7CacheContract(
    cache_sha256="0ba73414c3adea7712f7a68b1e76d934a17694a27671f35b8aa191bcc6aa1ee0",
    cache_identity="f08931bae1d07a54af2133e80a6aba631ce29803ce476050bdf3da498090c3eb",
    search_identity="b45c59f0cd163dcb067bb5eda25eb40e9229f1c401c5b23b0b0ca528c2815ba7",
    base_dataset_identity=(
        "81e1c5d85bd7753751c03508d947745723e3176a758e56676cee0037a75841e3"
    ),
    lockbox_partition_names_hash=(
        "5b49865a58c877acdffe4fbad537d705bd58426c65c3aac3d37be4328eacfb95"
    ),
    action_hash="94d4367c9e2959b3822463a636793e032a051db5051ac4f29f0adeb223321116",
    date_hash="b72ca069bce8ef4fb3c575427285a24992758cb01035ee86e818f4c71dd2dfd4",
    state_rows=1001,
    action_count=1999,
    first_exchange_date="2022-01-03",
    last_exchange_date="2025-12-29",
)


@dataclass(frozen=True, slots=True)
class _TrackedBlob:
    path: str
    git_mode: str
    git_blob_oid: str
    content: bytes

    @property
    def file_mode(self) -> int:
        return 0o755 if self.git_mode == "100755" else 0o644

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True, slots=True)
class Top2000M03RV7PackageBuildResult:
    output_root: str
    git_commit: str
    git_tree: str
    package_plan_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    cache_sha256: str
    cache_manifest_sha256: str
    data_manifest_sha256: str
    dependency_manifest_sha256: str
    runtime_manifest_sha256: str
    execution_manifest_sha256: str
    image_reference: str = PINNED_QUANTTRADE_IMAGE
    data_role: str = M03R_TOP2000_DEV_DATA_ROLE
    schema: str = PACKAGE_BUILDER_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV7PackageBuildError(
            "package manifest is not canonical-JSON safe"
        ) from exc
    return encoded + b"\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _axis_sha256(values: Sequence[str]) -> str:
    encoded = (
        json.dumps(
            list(values),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    current = Path(parts[0])
    stop = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:stop]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise Top2000M03RV7PackageBuildError(
                f"required path component is absent: {current}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise Top2000M03RV7PackageBuildError(
                f"symlink path components are forbidden: {current}"
            )


def _require_regular_file(path: Path, *, label: str) -> Path:
    _assert_no_symlink_components(path, include_leaf=True)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise Top2000M03RV7PackageBuildError(f"{label} is absent: {path}") from exc
    if not stat.S_ISREG(mode):
        raise Top2000M03RV7PackageBuildError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return path.absolute()


def _require_repository(path: Path) -> Path:
    _assert_no_symlink_components(path, include_leaf=True)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise Top2000M03RV7PackageBuildError(f"repository is absent: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise Top2000M03RV7PackageBuildError(
            "repository must be a regular non-symlink directory"
        )
    root = Path(_git_text(path, "rev-parse", "--show-toplevel"))
    if root != path.absolute():
        raise Top2000M03RV7PackageBuildError(
            "--repo must name the exact Git worktree root"
        )
    return root


def _git_bytes(repository: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        input=input_bytes,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Top2000M03RV7PackageBuildError(
            f"git {' '.join(arguments)} failed: {detail}"
        )
    return completed.stdout


def _git_text(repository: Path, *arguments: str) -> str:
    return _git_bytes(repository, *arguments).decode("utf-8").strip()


def _require_clean_git(repository: Path) -> None:
    status = _git_bytes(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise Top2000M03RV7PackageBuildError(
            "Git worktree must be completely clean, including untracked files"
        )


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {"", "."}
        or ".." in path.parts
        or any(part in {"", ".git"} for part in path.parts)
    ):
        raise Top2000M03RV7PackageBuildError(
            f"unsafe tracked source path: {value!r}"
        )
    return path


def _tracked_blobs(repository: Path) -> tuple[_TrackedBlob, ...]:
    raw = _git_bytes(repository, "ls-files", "--stage", "-z")
    blobs: list[_TrackedBlob] = []
    paths: set[str] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, oid, stage = metadata.decode("ascii").split(" ")
            path = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise Top2000M03RV7PackageBuildError(
                "tracked Git inventory cannot be decoded safely"
            ) from exc
        _safe_relative_path(path)
        if stage != "0" or mode not in {"100644", "100755"}:
            raise Top2000M03RV7PackageBuildError(
                "only stage-zero regular tracked files may enter the package"
            )
        if path in paths:
            raise Top2000M03RV7PackageBuildError(
                f"duplicate tracked source path: {path}"
            )
        worktree_path = repository.joinpath(*PurePosixPath(path).parts)
        _require_regular_file(worktree_path, label="tracked source")
        content = _git_bytes(repository, "cat-file", "blob", oid)
        descriptor = os.open(worktree_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            worktree_content = bytearray()
            while block := os.read(descriptor, 1024 * 1024):
                worktree_content.extend(block)
        finally:
            os.close(descriptor)
        if bytes(worktree_content) != content:
            raise Top2000M03RV7PackageBuildError(
                f"tracked source changed during snapshot: {path}"
            )
        blobs.append(_TrackedBlob(path, mode, oid, content))
        paths.add(path)
    if not blobs:
        raise Top2000M03RV7PackageBuildError("repository has no tracked source files")
    return tuple(sorted(blobs, key=lambda value: value.path))


def _mkdir_inside(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current /= part
        try:
            os.mkdir(current, 0o750)
        except FileExistsError:
            mode = os.lstat(current).st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                raise Top2000M03RV7PackageBuildError(
                    f"package path collision is not a directory: {current}"
                )
    return current


def _write_exclusive(path: Path, content: bytes, *, mode: int = 0o640) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_tracked_source(output: Path, blobs: Sequence[_TrackedBlob]) -> None:
    source = output / "source"
    os.mkdir(source, 0o750)
    for blob in blobs:
        relative = _safe_relative_path(blob.path)
        parent = _mkdir_inside(source, relative.parent)
        _write_exclusive(parent / relative.name, blob.content, mode=blob.file_mode)


def _write_source_archive(path: Path, blobs: Sequence[_TrackedBlob]) -> str:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as destination:
            with tarfile.open(
                fileobj=destination,
                mode="w",
                format=tarfile.GNU_FORMAT,
            ) as archive:
                for blob in blobs:
                    info = tarfile.TarInfo(f"source/{blob.path}")
                    info.size = len(blob.content)
                    info.mode = blob.file_mode
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.type = tarfile.REGTYPE
                    archive.addfile(info, BytesIO(blob.content))
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        os.close(descriptor)
    return _file_sha256(path)


def _copy_cache(source: Path, destination: Path, *, expected_sha256: str) -> None:
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    destination_descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o640,
    )
    digest = hashlib.sha256()
    try:
        while block := os.read(source_descriptor, 1024 * 1024):
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        os.fsync(destination_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    if digest.hexdigest() != expected_sha256:
        raise Top2000M03RV7PackageBuildError(
            "cache changed while it was copied into the package"
        )


def _validate_cache(
    path: Path,
    contract: Top2000M03RV7CacheContract,
) -> dict[str, Any]:
    _require_regular_file(path, label="TOP2000 development cache")
    if _file_sha256(path) != contract.cache_sha256:
        raise Top2000M03RV7PackageBuildError("TOP2000 cache SHA-256 mismatch")
    try:
        verified = load_verified_top2000_hold30_development_cache(
            path,
            expected_cache_sha256=contract.cache_sha256,
            acknowledgement=DEVELOPMENT_ACK,
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Top2000M03RV7PackageBuildError(
            "TOP2000 cache failed its typed development-only loader"
        ) from exc
    if not isinstance(payload, dict):
        raise Top2000M03RV7PackageBuildError("TOP2000 cache payload is not a mapping")
    required_identity = {
        "cache_identity": contract.cache_identity,
        "search_identity": contract.search_identity,
        "base_dataset_identity": contract.base_dataset_identity,
        "lockbox_partition_names_hash": contract.lockbox_partition_names_hash,
        "action_hash": contract.action_hash,
        "date_hash": contract.date_hash,
        "development_only": True,
        "bars_only": True,
        "bar_seconds": contract.bar_seconds,
    }
    if any(payload.get(name) != value for name, value in required_identity.items()):
        raise Top2000M03RV7PackageBuildError(
            "TOP2000 cache identity or development-only role drifted"
        )
    dates = tuple(verified.exchange_dates)
    actions = tuple(verified.action_ids)
    if (
        tuple(verified.daily_ohlcv.shape)
        != (contract.state_rows, contract.action_count, contract.feature_count)
        or tuple(verified.availability.shape)
        != (contract.state_rows, contract.action_count)
        or str(verified.daily_ohlcv.dtype) != contract.daily_dtype
        or str(verified.availability.dtype) != contract.availability_dtype
        or len(dates) != contract.state_rows
        or dates[0] != contract.first_exchange_date
        or dates[-1] != contract.last_exchange_date
        or len(actions) != contract.action_count
        or actions[0] != "CASH"
        or _axis_sha256(dates) != contract.date_hash
        or _axis_sha256(actions) != contract.action_hash
    ):
        raise Top2000M03RV7PackageBuildError(
            "TOP2000 cache tensor, date, or action geometry drifted"
        )
    parsed_dates = tuple(dt.date.fromisoformat(value) for value in dates)
    if any(left >= right for left, right in pairwise(parsed_dates)):
        raise Top2000M03RV7PackageBuildError(
            "TOP2000 exchange dates are not strictly increasing"
        )
    return {
        "cache_sha256": contract.cache_sha256,
        "cache_identity": contract.cache_identity,
        "search_identity": contract.search_identity,
        "base_dataset_identity": contract.base_dataset_identity,
        "lockbox_partition_names_hash": contract.lockbox_partition_names_hash,
        "action_hash": contract.action_hash,
        "date_hash": contract.date_hash,
        "daily_ohlcv_shape": list(verified.daily_ohlcv.shape),
        "daily_ohlcv_dtype": str(verified.daily_ohlcv.dtype),
        "availability_shape": list(verified.availability.shape),
        "availability_dtype": str(verified.availability.dtype),
        "exchange_date_count": len(dates),
        "first_exchange_date": dates[0],
        "last_exchange_date": dates[-1],
        "action_count": len(actions),
        "cash_action_id": actions[0],
        "bar_seconds": verified.bar_seconds,
        "observation_representation": "daily-ohlcv-aggregated-from-5-minute-bars",
        "universe": "future-selected-top2000-development-only",
        "data_role": contract.data_role,
        "development_only": True,
        "outer_evaluation_authorized": False,
        "promotion_eligible": False,
    }


def _benchmark_feasibility_preflight(
    path: Path,
    contract: Top2000M03RV7CacheContract,
) -> dict[str, Any]:
    """Replay every governed fold anchor on CPU before GPU admission.

    Both the 378-state fold chronology and its 64-state factor-calibration
    prefix are constructed through the authoritative adapter.  That adapter's
    v2 fill-time preflight therefore rejects an infeasible C1 anchor before a
    package can be rendered or a Pod can request H100 capacity.
    """

    try:
        cache = load_verified_top2000_hold30_development_cache(
            path,
            expected_cache_sha256=contract.cache_sha256,
            acknowledgement=DEVELOPMENT_ACK,
        )
        folds = render_top2000_m03r_v7_development_folds(
            len(cache.exchange_dates)
        )
        rows: list[dict[str, Any]] = []
        for fold in folds:
            state_start = (
                fold.validation_decision_start
                - TOP2000_M03R_V7_DEV_WARMUP_DECISIONS
            )
            validation = (
                build_top2000_hold30_development_sequence_from_loaded_cache(
                    cache,
                    state_start_index=state_start,
                    state_stop_index_exclusive=(
                        state_start + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
                    ),
                    max_state_rows=TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
                    max_stock_weight=TOP2000_HOLD30_MAX_STOCK_WEIGHT,
                    output_device="cpu",
                )
            )
            calibration_rows = (
                TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS + 1
            )
            calibration = (
                build_top2000_hold30_development_sequence_from_loaded_cache(
                    cache,
                    state_start_index=state_start,
                    state_stop_index_exclusive=(
                        state_start + calibration_rows
                    ),
                    max_state_rows=calibration_rows,
                    max_stock_weight=TOP2000_HOLD30_MAX_STOCK_WEIGHT,
                    output_device="cpu",
                )
            )
            rows.append(
                {
                    "fold_index": fold.fold_index,
                    "fold_receipt_sha256": fold.receipt_sha256,
                    "state_start_index": state_start,
                    "validation_state_stop_index_exclusive": (
                        state_start + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
                    ),
                    "calibration_state_stop_index_exclusive": (
                        state_start + calibration_rows
                    ),
                    "validation_slice_receipt_sha256": (
                        validation.identity.receipt_sha256
                    ),
                    "validation_benchmark_trace_sha256": (
                        validation.identity.benchmark_trace_sha256
                    ),
                    "validation_benchmark_weights_sha256": (
                        validation.identity.benchmark_weights_sha256
                    ),
                    "calibration_slice_receipt_sha256": (
                        calibration.identity.receipt_sha256
                    ),
                    "calibration_benchmark_trace_sha256": (
                        calibration.identity.benchmark_trace_sha256
                    ),
                    "calibration_benchmark_weights_sha256": (
                        calibration.identity.benchmark_weights_sha256
                    ),
                }
            )
    except Exception as exc:
        raise Top2000M03RV7PackageBuildError(
            "TOP2000 v2 benchmark feasibility preflight failed: " + str(exc)
        ) from exc

    identity = {
        "adapter_schema": TOP2000_HOLD30_DEVELOPMENT_ADAPTER_SCHEMA,
        "benchmark_id": TOP2000_HOLD30_BENCHMARK_ID,
        "benchmark_risk_repair_rule": (
            TOP2000_HOLD30_BENCHMARK_RISK_REPAIR_RULE
        ),
        "max_stock_weight": TOP2000_HOLD30_MAX_STOCK_WEIGHT,
        "folds": rows,
    }
    return {
        "schema": BENCHMARK_PREFLIGHT_SCHEMA,
        **identity,
        "benchmark_trace_cap_identity_sha256": hashlib.sha256(
            _canonical_json_bytes(identity)
        ).hexdigest(),
        "fold_count": len(rows),
        "cpu_only": True,
        "all_validation_and_calibration_slices_feasible": True,
        "development_only": True,
        "promotion_eligible": False,
    }


def _read_canonical_json(path: Path) -> dict[str, Any]:
    _require_regular_file(path, label="package JSON manifest")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Top2000M03RV7PackageBuildError(
            f"package JSON is invalid: {path.name}"
        ) from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise Top2000M03RV7PackageBuildError(
            f"package JSON is not canonical: {path.name}"
        )
    return cast(dict[str, Any], value)


def _reconstruct_plan(
    payload: Mapping[str, Any],
) -> M03RV7Top2000PackagePlan | M03RV7Seed17PackagePlan:
    try:
        artifacts = M03RV7Top2000ArtifactBindings(**payload["artifacts"])
        profile = M03RV7Top2000RuntimeProfile(**payload["runtime_profile"])
        if (
            payload.get("protocol_generation")
            == M03R_SEED17_TOP2000_PROTOCOL_GENERATION
        ):
            seed17_indices = tuple(
                M03RV7Seed17IndexPlan(
                    **{
                        **row,
                        "fold_indices": tuple(row["fold_indices"]),
                        "paired_seeds": tuple(row["paired_seeds"]),
                    }
                )
                for row in payload["indices"]
            )
            return M03RV7Seed17PackagePlan(
                artifacts=artifacts,
                indices=seed17_indices,
                runtime_profile=profile,
                plan_artifact_path=payload["plan_artifact_path"],
                benchmark_preflight_sha256=(
                    payload["benchmark_preflight_sha256"]
                ),
                package_plan_sha256=payload["package_plan_sha256"],
                source_pythonpath=payload["source_pythonpath"],
                protocol_sha256=payload["protocol_sha256"],
                protocol_generation=payload["protocol_generation"],
                design_id=payload["design_id"],
                data_role=payload["data_role"],
                one_member_fold_execution=payload[
                    "one_member_fold_execution"
                ],
                five_seed_ensemble_eligible=payload[
                    "five_seed_ensemble_eligible"
                ],
                promotion_eligible=payload["promotion_eligible"],
                outer_evaluation_authorized=payload[
                    "outer_evaluation_authorized"
                ],
            )
        indices = tuple(
            M03RV7Top2000IndexPlan(
                **{
                    **row,
                    "fold_indices": tuple(row["fold_indices"]),
                    "paired_seeds": tuple(row["paired_seeds"]),
                }
            )
            for row in payload["indices"]
        )
        return M03RV7Top2000PackagePlan(
            artifacts=artifacts,
            indices=indices,
            runtime_profile=profile,
            plan_artifact_path=payload["plan_artifact_path"],
            package_plan_sha256=payload["package_plan_sha256"],
            source_pythonpath=payload["source_pythonpath"],
            protocol_sha256=payload["protocol_sha256"],
            protocol_generation=payload["protocol_generation"],
            design_id=payload["design_id"],
            data_role=payload["data_role"],
            promotion_eligible=payload["promotion_eligible"],
            outer_evaluation_authorized=payload["outer_evaluation_authorized"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Top2000M03RV7PackageBuildError(
            "package-plan.json failed typed reconstruction"
        ) from exc


def validate_top2000_m03r_v7_package(
    output_root: str | Path,
    *,
    cache_contract: Top2000M03RV7CacheContract = CURRENT_TOP2000_CACHE_CONTRACT,
    seed17_diagnostic: bool = False,
) -> M03RV7Top2000PackagePlan | M03RV7Seed17PackagePlan:
    """Independently replay all package hashes and typed plan invariants."""

    root = Path(output_root)
    _assert_no_symlink_components(root, include_leaf=True)
    if not root.is_dir() or root.is_symlink():
        raise Top2000M03RV7PackageBuildError(
            "package root must be a regular non-symlink directory"
        )
    root_entries = {entry.name for entry in root.iterdir()}
    expected_root = (
        SEED17_ROOT_FILE_ALLOWLIST
        if seed17_diagnostic
        else ROOT_FILE_ALLOWLIST
    )
    if root_entries != expected_root | {"source"}:
        raise Top2000M03RV7PackageBuildError("package root inventory drifted")
    for directory, child_directories, files in os.walk(root / "source"):
        for name in child_directories + files:
            if Path(directory, name).is_symlink():
                raise Top2000M03RV7PackageBuildError(
                    "symlinks are forbidden in packaged source"
                )

    source_manifest = _read_canonical_json(root / "source-manifest.json")
    cache_manifest = _read_canonical_json(root / "cache-manifest.json")
    data_manifest = _read_canonical_json(root / "data-manifest.json")
    dependency_manifest = _read_canonical_json(root / "dependency-manifest.json")
    runtime_manifest = _read_canonical_json(root / "runtime-manifest.json")
    execution_manifest = _read_canonical_json(root / "execution-manifest.json")
    benchmark_preflight = (
        _read_canonical_json(root / "benchmark-preflight.json")
        if seed17_diagnostic
        else None
    )
    plan_payload = _read_canonical_json(root / "package-plan.json")
    expected_plan_file_schema = (
        M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA
        if seed17_diagnostic
        else "rl-quant.m03r-v7-top2000-package-plan-file-v1"
    )
    if plan_payload.get("schema") != expected_plan_file_schema:
        raise Top2000M03RV7PackageBuildError(
            "package plan file schema drifted from the worker loader"
        )
    plan = _reconstruct_plan(plan_payload)
    if seed17_diagnostic != isinstance(plan, M03RV7Seed17PackagePlan):
        raise Top2000M03RV7PackageBuildError(
            "package generation does not match the requested validator"
        )

    expected_hashes = {
        "source_archive_sha256": _file_sha256(root / "source.tar"),
        "source_manifest_sha256": _file_sha256(root / "source-manifest.json"),
        "cache_artifact_sha256": _file_sha256(root / "cache.pt"),
        "cache_manifest_sha256": _file_sha256(root / "cache-manifest.json"),
        "data_manifest_sha256": _file_sha256(root / "data-manifest.json"),
        "dependency_lock_sha256": _file_sha256(root / "dependency-manifest.json"),
        "execution_model_sha256": _file_sha256(root / "execution-manifest.json"),
    }
    if any(
        getattr(plan.artifacts, name) != value for name, value in expected_hashes.items()
    ):
        raise Top2000M03RV7PackageBuildError(
            "package-plan artifact hash does not match package bytes"
        )
    if (
        plan.artifacts.image_reference != PINNED_QUANTTRADE_IMAGE
        or plan.artifacts.image_digest_sha256 != PINNED_QUANTTRADE_IMAGE_DIGEST
        or source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or cache_manifest.get("schema") != CACHE_MANIFEST_SCHEMA
        or data_manifest.get("schema") != DATA_MANIFEST_SCHEMA
        or dependency_manifest.get("schema") != DEPENDENCY_MANIFEST_SCHEMA
        or runtime_manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA
        or runtime_manifest.get("source_pythonpath") != plan.source_pythonpath
        or execution_manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or execution_manifest.get("runtime_manifest_sha256")
        != _file_sha256(root / "runtime-manifest.json")
    ):
        raise Top2000M03RV7PackageBuildError(
            "package schema, image, or runtime binding drifted"
        )
    _validate_cache(root / "cache.pt", cache_contract)
    if seed17_diagnostic:
        assert benchmark_preflight is not None
        replayed_preflight = _benchmark_feasibility_preflight(
            root / "cache.pt", cache_contract
        )
        benchmark_preflight_sha256 = _file_sha256(
            root / "benchmark-preflight.json"
        )
        if (
            benchmark_preflight != replayed_preflight
            or benchmark_preflight.get("schema")
            != BENCHMARK_PREFLIGHT_SCHEMA
            or not isinstance(plan, M03RV7Seed17PackagePlan)
            or plan.benchmark_preflight_sha256
            != benchmark_preflight_sha256
            or execution_manifest.get("benchmark_preflight_sha256")
            != benchmark_preflight_sha256
            or execution_manifest.get(
                "benchmark_trace_cap_identity_sha256"
            )
            != benchmark_preflight.get(
                "benchmark_trace_cap_identity_sha256"
            )
            or data_manifest.get("benchmark_trace_cap_identity_sha256")
            != benchmark_preflight.get(
                "benchmark_trace_cap_identity_sha256"
            )
            or data_manifest.get("benchmark_max_stock_weight")
            != TOP2000_HOLD30_MAX_STOCK_WEIGHT
            or runtime_manifest.get("worker_argv_prefix", [None])[-1]
            != "rl_quant.workflows.top2000_m03r_v7_seed17_dev"
            or execution_manifest.get("protocol_generation")
            != M03R_SEED17_TOP2000_PROTOCOL_GENERATION
            or execution_manifest.get("protocol_sha256")
            != M03R_SEED17_TOP2000_PROTOCOL_SHA256
        ):
            raise Top2000M03RV7PackageBuildError(
                "seed-17 benchmark preflight bytes or identity drifted"
            )

    file_rows = source_manifest.get("files")
    if not isinstance(file_rows, list):
        raise Top2000M03RV7PackageBuildError("source manifest omitted file inventory")
    expected_source: dict[str, Mapping[str, Any]] = {}
    for row in file_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise Top2000M03RV7PackageBuildError("source manifest row is invalid")
        path = cast(str, row["path"])
        if path in expected_source:
            raise Top2000M03RV7PackageBuildError("source manifest contains duplicates")
        expected_source[path] = row
        source_path = root / "source" / Path(*PurePosixPath(path).parts)
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or _file_sha256(source_path) != row.get("sha256")
            or source_path.stat().st_size != row.get("size")
        ):
            raise Top2000M03RV7PackageBuildError(
                f"packaged source does not match its manifest: {path}"
            )
    actual_source = {
        path.relative_to(root / "source").as_posix()
        for path in (root / "source").rglob("*")
        if path.is_file()
    }
    if actual_source != set(expected_source):
        raise Top2000M03RV7PackageBuildError("packaged source inventory drifted")

    archive_paths: set[str] = set()
    with tarfile.open(root / "source.tar", mode="r:") as archive:
        for member in archive:
            if not member.isfile() or member.name in archive_paths:
                raise Top2000M03RV7PackageBuildError(
                    "source archive contains a duplicate or non-regular member"
                )
            archive_paths.add(member.name)
            prefix = "source/"
            if not member.name.startswith(prefix):
                raise Top2000M03RV7PackageBuildError(
                    "source archive member escaped its source prefix"
                )
            path = member.name[len(prefix) :]
            row = expected_source.get(path)
            extracted = archive.extractfile(member)
            if row is None or extracted is None:
                raise Top2000M03RV7PackageBuildError(
                    "source archive inventory does not match source manifest"
                )
            content = extracted.read()
            if (
                hashlib.sha256(content).hexdigest() != row.get("sha256")
                or len(content) != row.get("size")
            ):
                raise Top2000M03RV7PackageBuildError(
                    "source archive content does not match source manifest"
                )
    if archive_paths != {f"source/{path}" for path in expected_source}:
        raise Top2000M03RV7PackageBuildError("source archive inventory is incomplete")
    return plan


def build_top2000_m03r_v7_package(
    *,
    repository_root: str | Path,
    cache_path: str | Path,
    output_root: str | Path,
    cache_contract: Top2000M03RV7CacheContract = CURRENT_TOP2000_CACHE_CONTRACT,
    critical_source_paths: Sequence[str] | None = None,
    runtime_profile: M03RV7Top2000RuntimeProfile | None = None,
    seed17_diagnostic: bool = False,
) -> Top2000M03RV7PackageBuildResult:
    """Create and verify one new package directory without overwriting."""

    repository = _require_repository(Path(repository_root))
    cache = _require_regular_file(Path(cache_path), label="TOP2000 development cache")
    output = Path(output_root).absolute()
    _assert_no_symlink_components(output, include_leaf=False)
    if os.path.lexists(output):
        raise Top2000M03RV7PackageBuildError(
            "output path already exists; package builds never overwrite"
        )
    try:
        output.relative_to(repository)
    except ValueError:
        pass
    else:
        raise Top2000M03RV7PackageBuildError(
            "package output must be outside the source repository"
        )
    _require_clean_git(repository)
    cache_summary = _validate_cache(cache, cache_contract)
    benchmark_preflight = (
        _benchmark_feasibility_preflight(cache, cache_contract)
        if seed17_diagnostic
        else None
    )
    selected_critical_sources = tuple(
        critical_source_paths
        if critical_source_paths is not None
        else (
            SEED17_CRITICAL_SOURCE_PATHS
            if seed17_diagnostic
            else DEFAULT_CRITICAL_SOURCE_PATHS
        )
    )
    protocol_generation = (
        M03R_SEED17_TOP2000_PROTOCOL_GENERATION
        if seed17_diagnostic
        else M03R_TOP2000_DEV_PROTOCOL_GENERATION
    )
    protocol_sha256 = (
        M03R_SEED17_TOP2000_PROTOCOL_SHA256
        if seed17_diagnostic
        else M03R_TOP2000_DEV_PROTOCOL_SHA256
    )
    design_id = (
        M03R_SEED17_TOP2000_DESIGN_ID
        if seed17_diagnostic
        else M03R_TOP2000_DEV_DESIGN_ID
    )
    setting_ids = (
        M03R_SEED17_TOP2000_SETTING_IDS
        if seed17_diagnostic
        else M03R_TOP2000_DEV_SETTING_IDS
    )
    paired_seeds = (
        (17,) if seed17_diagnostic else TOP2000_M03R_V7_DEV_SEEDS
    )
    commit = _git_text(repository, "rev-parse", "HEAD")
    tree = _git_text(repository, "rev-parse", "HEAD^{tree}")
    blobs = _tracked_blobs(repository)
    blob_by_path = {blob.path: blob for blob in blobs}
    missing = sorted(set(selected_critical_sources) - set(blob_by_path))
    if missing:
        raise Top2000M03RV7PackageBuildError(
            "critical execution source is not tracked: " + ", ".join(missing)
        )
    _require_clean_git(repository)

    os.mkdir(output, 0o700)
    _copy_tracked_source(output, blobs)
    source_archive_sha256 = _write_source_archive(output / "source.tar", blobs)
    source_manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "git_commit": commit,
        "git_tree": tree,
        "git_clean": True,
        "tracked_regular_file_count": len(blobs),
        "archive_path": "source.tar",
        "archive_root": "source",
        "source_archive_sha256": source_archive_sha256,
        "files": [
            {
                "path": blob.path,
                "git_mode": blob.git_mode,
                "git_blob_oid": blob.git_blob_oid,
                "size": len(blob.content),
                "sha256": blob.sha256,
            }
            for blob in blobs
        ],
    }
    _write_exclusive(
        output / "source-manifest.json",
        _canonical_json_bytes(source_manifest),
    )

    _copy_cache(cache, output / "cache.pt", expected_sha256=cache_contract.cache_sha256)
    copied_cache_summary = _validate_cache(output / "cache.pt", cache_contract)
    if copied_cache_summary != cache_summary:
        raise Top2000M03RV7PackageBuildError(
            "copied cache validation does not reproduce the source validation"
        )
    cache_manifest = {
        "schema": CACHE_MANIFEST_SCHEMA,
        "artifact_path": "cache.pt",
        "artifact_size_bytes": (output / "cache.pt").stat().st_size,
        **cache_summary,
    }
    _write_exclusive(
        output / "cache-manifest.json",
        _canonical_json_bytes(cache_manifest),
    )
    data_manifest = {
        "schema": DATA_MANIFEST_SCHEMA,
        "cache_artifact_sha256": cache_contract.cache_sha256,
        "cache_identity": cache_contract.cache_identity,
        "search_identity": cache_contract.search_identity,
        "universe": "future-selected-top2000-development-only",
        "observation_representation": "daily-ohlcv-aggregated-from-5-minute-bars",
        "decision_frequency": "one-portfolio-decision-per-trading-session",
        "first_exchange_date": cache_contract.first_exchange_date,
        "last_exchange_date": cache_contract.last_exchange_date,
        "state_rows": cache_contract.state_rows,
        "action_count": cache_contract.action_count,
        "data_role": cache_contract.data_role,
        "development_only": True,
        "not_reportable": True,
        "promotion_eligible": False,
        "outer_evaluation_authorized": False,
        "contains_2026_lockbox": False,
    }
    if benchmark_preflight is not None:
        data_manifest.update(
            {
                "benchmark_adapter_schema": benchmark_preflight[
                    "adapter_schema"
                ],
                "benchmark_id": benchmark_preflight["benchmark_id"],
                "benchmark_risk_repair_rule": benchmark_preflight[
                    "benchmark_risk_repair_rule"
                ],
                "benchmark_max_stock_weight": benchmark_preflight[
                    "max_stock_weight"
                ],
                "benchmark_trace_cap_identity_sha256": (
                    benchmark_preflight[
                        "benchmark_trace_cap_identity_sha256"
                    ]
                ),
            }
        )
    _write_exclusive(
        output / "data-manifest.json",
        _canonical_json_bytes(data_manifest),
    )
    benchmark_preflight_sha256: str | None = None
    if benchmark_preflight is not None:
        _write_exclusive(
            output / "benchmark-preflight.json",
            _canonical_json_bytes(benchmark_preflight),
        )
        benchmark_preflight_sha256 = _file_sha256(
            output / "benchmark-preflight.json"
        )

    dependency_paths = tuple(
        blob.path
        for blob in blobs
        if blob.path == "pyproject.toml"
        or blob.path.endswith(".lock")
        or PurePosixPath(blob.path).name.startswith("requirements")
    )
    if "pyproject.toml" not in dependency_paths:
        raise Top2000M03RV7PackageBuildError(
            "tracked pyproject.toml is required for dependency binding"
        )
    dependency_manifest = {
        "schema": DEPENDENCY_MANIFEST_SCHEMA,
        "resolution_contract": (
            "digest-pinned-image-plus-tracked-dependency-declarations"
        ),
        "image_reference": PINNED_QUANTTRADE_IMAGE,
        "files": [
            {
                "path": path,
                "sha256": blob_by_path[path].sha256,
                "git_blob_oid": blob_by_path[path].git_blob_oid,
            }
            for path in dependency_paths
        ],
    }
    _write_exclusive(
        output / "dependency-manifest.json",
        _canonical_json_bytes(dependency_manifest),
    )

    profile = runtime_profile or M03RV7Top2000RuntimeProfile()
    worker_argv_prefix = (
        PINNED_IMAGE_PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--max-restarts=1",
        "--nproc-per-node=2",
        "-m",
        (
            "rl_quant.workflows.top2000_m03r_v7_seed17_dev"
            if seed17_diagnostic
            else "rl_quant.workflows.top2000_m03r_v7_dev"
        ),
    )
    runtime_manifest = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "image_reference": PINNED_QUANTTRADE_IMAGE,
        "image_digest_sha256": PINNED_QUANTTRADE_IMAGE_DIGEST,
        "worker_argv_prefix": list(worker_argv_prefix),
        "runtime_profile": asdict(profile),
        "package_plan_container_path": PACKAGE_PLAN_CONTAINER_PATH,
        "package_mount_path": PACKAGE_MOUNT_PATH,
        "source_pythonpath": M03R_TOP2000_PACKAGE_SOURCE_PYTHONPATH,
        "output_mount_path": OUTPUT_MOUNT_PATH,
        "local_world_size": 2,
        "torchrun_max_restarts": 1,
        "gpu_count_per_worker": 2,
        "gpu_product": "NVIDIA-H100-80GB-HBM3",
        "distributed_backend": "nccl",
        "complete_cross_section_per_rank": True,
        "stock_axis_partitioning": False,
        "service_account_name": "default",
        "pvc_claim_name": "yding4-gpu-home",
        "pvc_training_subpath": "quant/training",
        "automount_service_account_token": False,
        "development_only": True,
        "promotion_eligible": False,
    }
    _write_exclusive(
        output / "runtime-manifest.json",
        _canonical_json_bytes(runtime_manifest),
    )

    manifest_hashes = {
        "source_manifest_sha256": _file_sha256(output / "source-manifest.json"),
        "cache_manifest_sha256": _file_sha256(output / "cache-manifest.json"),
        "data_manifest_sha256": _file_sha256(output / "data-manifest.json"),
        "dependency_manifest_sha256": _file_sha256(
            output / "dependency-manifest.json"
        ),
        "runtime_manifest_sha256": _file_sha256(output / "runtime-manifest.json"),
    }
    execution_manifest = {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "protocol_generation": protocol_generation,
        "protocol_sha256": protocol_sha256,
        "design_id": design_id,
        "setting_ids": list(setting_ids),
        "runtime_profile": asdict(profile),
        "episode_state_rows": TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
        "observation_warmup_decisions": TOP2000_M03R_V7_DEV_WARMUP_DECISIONS,
        "scored_decisions": TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS,
        "label_support_decisions": TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS,
        "auxiliary_alpha_horizons": list(TOP2000_M03R_V7_DEV_ALPHA_HORIZONS),
        "fold_count": TOP2000_M03R_V7_DEV_FOLD_COUNT,
        "paired_seeds": list(paired_seeds),
        "critical_source_files": [
            {
                "path": path,
                "sha256": blob_by_path[path].sha256,
                "git_blob_oid": blob_by_path[path].git_blob_oid,
            }
            for path in selected_critical_sources
        ],
        "source_archive_sha256": source_archive_sha256,
        "cache_artifact_sha256": cache_contract.cache_sha256,
        "image_reference": PINNED_QUANTTRADE_IMAGE,
        "data_role": M03R_TOP2000_DEV_DATA_ROLE,
        "development_only": True,
        "outer_evaluation_authorized": False,
        "promotion_eligible": False,
        **manifest_hashes,
    }
    if benchmark_preflight is not None:
        assert benchmark_preflight_sha256 is not None
        execution_manifest.update(
            {
                "benchmark_preflight_sha256": benchmark_preflight_sha256,
                "benchmark_trace_cap_identity_sha256": (
                    benchmark_preflight[
                        "benchmark_trace_cap_identity_sha256"
                    ]
                ),
                "benchmark_adapter_schema": (
                    benchmark_preflight["adapter_schema"]
                ),
                "benchmark_id": benchmark_preflight["benchmark_id"],
                "benchmark_max_stock_weight": benchmark_preflight[
                    "max_stock_weight"
                ],
            }
        )
    _write_exclusive(
        output / "execution-manifest.json",
        _canonical_json_bytes(execution_manifest),
    )

    artifacts = M03RV7Top2000ArtifactBindings(
        source_archive_sha256=source_archive_sha256,
        source_manifest_sha256=manifest_hashes["source_manifest_sha256"],
        dependency_lock_sha256=manifest_hashes["dependency_manifest_sha256"],
        cache_artifact_sha256=cache_contract.cache_sha256,
        cache_manifest_sha256=manifest_hashes["cache_manifest_sha256"],
        data_manifest_sha256=manifest_hashes["data_manifest_sha256"],
        execution_model_sha256=_file_sha256(output / "execution-manifest.json"),
        image_reference=PINNED_QUANTTRADE_IMAGE,
        image_digest_sha256=PINNED_QUANTTRADE_IMAGE_DIGEST,
    )
    if seed17_diagnostic:
        if benchmark_preflight_sha256 is None:
            raise Top2000M03RV7PackageBuildError(
                "seed-17 package omitted its benchmark preflight"
            )
        plan: M03RV7Top2000PackagePlan | M03RV7Seed17PackagePlan = (
            build_m03r_v7_seed17_top2000_package_plan(
                artifacts=artifacts,
                plan_artifact_path=PACKAGE_PLAN_CONTAINER_PATH,
                benchmark_preflight_sha256=benchmark_preflight_sha256,
                runtime_profile=profile,
            )
        )
        plan_file_schema = M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA
    else:
        plan = build_m03r_v7_top2000_package_plan(
            artifacts=artifacts,
            plan_artifact_path=PACKAGE_PLAN_CONTAINER_PATH,
            runtime_profile=profile,
        )
        plan_file_schema = "rl-quant.m03r-v7-top2000-package-plan-file-v1"
    plan_payload = {
        **asdict(plan),
        "schema": plan_file_schema,
    }
    _write_exclusive(
        output / "package-plan.json",
        _canonical_json_bytes(plan_payload),
    )
    _require_clean_git(repository)
    verified_plan = validate_top2000_m03r_v7_package(
        output,
        cache_contract=cache_contract,
        seed17_diagnostic=seed17_diagnostic,
    )
    if verified_plan != plan:
        raise Top2000M03RV7PackageBuildError(
            "self-validation reconstructed a different package plan"
        )
    return Top2000M03RV7PackageBuildResult(
        output_root=str(output),
        git_commit=commit,
        git_tree=tree,
        package_plan_sha256=plan.package_plan_sha256,
        source_archive_sha256=source_archive_sha256,
        source_manifest_sha256=artifacts.source_manifest_sha256,
        cache_sha256=artifacts.cache_artifact_sha256,
        cache_manifest_sha256=artifacts.cache_manifest_sha256,
        data_manifest_sha256=artifacts.data_manifest_sha256,
        dependency_manifest_sha256=artifacts.dependency_lock_sha256,
        runtime_manifest_sha256=manifest_hashes["runtime_manifest_sha256"],
        execution_manifest_sha256=artifacts.execution_model_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the clean-source, exact-cache TOP2000 M03R-v7 development package"
        )
    )
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--seed17-diagnostic",
        action="store_true",
        help=(
            "build the disjoint six-fold seed-17 diagnostic package and run "
            "its CPU benchmark-feasibility preflight"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = build_top2000_m03r_v7_package(
            repository_root=arguments.repo,
            cache_path=arguments.cache,
            output_root=arguments.output,
            seed17_diagnostic=arguments.seed17_diagnostic,
        )
    except Top2000M03RV7PackageBuildError as exc:
        parser.error(str(exc))
    print(_canonical_json_bytes(result.canonical_payload()).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_PREFLIGHT_SCHEMA",
    "CURRENT_TOP2000_CACHE_CONTRACT",
    "PACKAGE_PLAN_CONTAINER_PATH",
    "PINNED_IMAGE_PYTHON",
    "PINNED_QUANTTRADE_IMAGE",
    "PINNED_QUANTTRADE_IMAGE_DIGEST",
    "SEED17_CRITICAL_SOURCE_PATHS",
    "Top2000M03RV7CacheContract",
    "Top2000M03RV7PackageBuildError",
    "Top2000M03RV7PackageBuildResult",
    "build_top2000_m03r_v7_package",
    "main",
    "validate_top2000_m03r_v7_package",
]
