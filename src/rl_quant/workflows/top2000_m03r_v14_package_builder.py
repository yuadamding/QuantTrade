"""Build and validate one immutable local M03R-v14 predictive package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_PROTOCOL_SHA256,
    M03R_V14_SELECTED_HORIZON_SESSIONS,
)
from rl_quant.training.top2000_m03r_v14_fold import (
    M03RV14PanelEpisodeSchedule,
    render_m03r_v14_fold_geometries,
)
from rl_quant.training.top2000_m03r_v14_initial_state import (
    write_m03r_v14_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v14_package import (
    M03R_V14_RUNTIME_ENTRYPOINT,
    M03RV14ExecutionAuthorization,
    M03RV14PackageArtifacts,
    build_m03r_v14_package_plan,
    load_m03r_v14_execution_authorization,
    load_m03r_v14_package_plan,
    write_m03r_v14_execution_authorization,
    write_m03r_v14_package_plan,
)
from rl_quant.training.top2000_m03r_v14_policy import (
    Top2000M03RV14PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v14_preflight import (
    load_m03r_v14_structural_preflight,
)

M03R_V14_LOCAL_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-local-predictive-package-v1"
)
M03R_V14_SOURCE_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-runtime-source-manifest-v1"
)
M03R_V14_EXECUTION_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-execution-manifest-v1"
)
M03R_V14_TRANSFER_ROOT = "qt-m03r-v14-predictive-package-v1"
M03R_V14_RUNTIME_WORKER = "src/rl_quant/workflows/top2000_m03r_v14_predictive.py"
PINNED_QUANTTRADE_IMAGE = (
    "hpcharbor.mdanderson.edu/yding41/ml2:quanttrade-ppo-cu124-py311-85cf781d3e08"
    "@sha256:7cff8faedcfb44ad25e1001d7e1634569f7cd3f5365bbd8ff8caa9b10d8bcdf9"
)


class M03RV14PackageBuildError(ValueError):
    """A v14 package member, identity, or no-clobber boundary drifted."""


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


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, name: str) -> Path:
    try:
        status = path.lstat()
    except OSError as exc:
        raise M03RV14PackageBuildError(f"{name} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise M03RV14PackageBuildError(f"{name} must be a nonempty regular file")
    return path


def _exclusive(path: Path, data: bytes, mode: int = 0o440) -> str:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def _copy(source: Path, target: Path) -> str:
    _regular(source, str(source))
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            while block := reader.read(1024 * 1024):
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != _file_sha256(source):
        target.unlink(missing_ok=True)
        raise M03RV14PackageBuildError("source changed during copy")
    return digest.hexdigest()


def _source_members(root: Path) -> tuple[Path, ...]:
    required = (root / "pyproject.toml", root / "uv.lock", root / M03R_V14_RUNTIME_WORKER)
    for path in required:
        _regular(path, str(path.relative_to(root)))
    candidates = (
        root / "pyproject.toml",
        root / "uv.lock",
        *(root / "src/rl_quant").rglob("*.py"),
    )
    result: list[Path] = []
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        _regular(path, str(path.relative_to(root)))
        if not path.resolve().is_relative_to(root):
            raise M03RV14PackageBuildError("source member leaves repository root")
        result.append(path)
    if len(result) != len(set(result)):
        raise M03RV14PackageBuildError("source inventory contains duplicates")
    return tuple(result)


def _inventory(root: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise M03RV14PackageBuildError("package inventory contains a symlink")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _file_sha256(path),
                    "size": path.stat().st_size,
                }
            )
    return tuple(rows)


def _write_source_tar(
    package_root: Path,
    source_rows: tuple[dict[str, Any], ...],
) -> str:
    target = package_root / "source.tar"
    directories = {"source"}
    for row in source_rows:
        parent = PurePosixPath("source", row["path"]).parent
        while str(parent) not in {"", "."}:
            directories.add(str(parent))
            parent = parent.parent
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as archive:
        for directory in sorted(
            directories,
            key=lambda value: (value.count("/"), value),
        ):
            info = tarfile.TarInfo(directory + "/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            archive.addfile(info)
        for row in source_rows:
            source = package_root / "source" / row["path"]
            info = tarfile.TarInfo("source/" + row["path"])
            info.size = row["size"]
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    os.chmod(target, 0o440)
    return _file_sha256(target)


def _safe_tar(path: Path, *, required_root: str | None = None) -> tuple[str, ...]:
    names: list[str] = []
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
                or (
                    required_root is not None
                    and (not pure.parts or pure.parts[0] != required_root)
                )
            ):
                raise M03RV14PackageBuildError("archive contains an unsafe member")
            names.append(member.name)
    if len(names) != len(set(names)):
        raise M03RV14PackageBuildError("archive contains duplicate members")
    return tuple(names)


def _initial_policy() -> Top2000M03RV14PredictivePolicy:
    random.seed(17)
    torch.manual_seed(17)
    return Top2000M03RV14PredictivePolicy(
        0,
        selected_horizon_sessions=M03R_V14_SELECTED_HORIZON_SESSIONS,
    )


def _preflight_identity(path: Path) -> tuple[str, str]:
    try:
        payload = json.loads(path.read_bytes())
        receipt_sha256 = payload["receipt_sha256"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise M03RV14PackageBuildError("structural preflight is malformed") from exc
    if not isinstance(receipt_sha256, str):
        raise M03RV14PackageBuildError("structural preflight receipt is malformed")
    file_sha256 = _file_sha256(path)
    load_m03r_v14_structural_preflight(
        path,
        expected_file_sha256=file_sha256,
        expected_receipt_sha256=receipt_sha256,
    )
    return file_sha256, receipt_sha256


def build_m03r_v14_local_package(
    *,
    source_root: str | Path,
    cache_path: str | Path,
    cache_manifest_path: str | Path,
    risk_root: str | Path,
    structural_preflight_path: str | Path,
    output_root: str | Path,
    image_reference: str = PINNED_QUANTTRADE_IMAGE,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    cache = _regular(Path(cache_path), "development cache")
    cache_manifest = _regular(Path(cache_manifest_path), "cache manifest")
    risk = Path(risk_root)
    risk_artifact = _regular(risk / "risk-exposures.pt", "risk artifact")
    risk_manifest = _regular(risk / "risk-source-manifest.json", "risk manifest")
    projector_manifest = _regular(
        risk / "projector-manifest.json",
        "projector manifest",
    )
    structural_preflight = _regular(
        Path(structural_preflight_path),
        "real-data structural preflight",
    )
    structural_file_sha, structural_receipt_sha = _preflight_identity(
        structural_preflight
    )
    structural_receipt = load_m03r_v14_structural_preflight(
        structural_preflight,
        expected_file_sha256=structural_file_sha,
        expected_receipt_sha256=structural_receipt_sha,
    )
    output = Path(output_root)
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    package_root = output / "package"
    package_root.mkdir(mode=0o750)

    source_rows: list[dict[str, Any]] = []
    for member in _source_members(source):
        relative = member.relative_to(source).as_posix()
        sha256 = _copy(member, package_root / "source" / relative)
        source_rows.append(
            {"path": relative, "sha256": sha256, "size": member.stat().st_size}
        )
    source_tuple = tuple(source_rows)
    source_manifest_sha = _exclusive(
        package_root / "source-manifest.json",
        _canonical(
            {
                "schema": M03R_V14_SOURCE_MANIFEST_SCHEMA,
                "protocol_sha256": M03R_V14_PROTOCOL_SHA256,
                "file_count": len(source_tuple),
                "files": source_tuple,
                "runtime_worker": M03R_V14_RUNTIME_WORKER,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
        ),
    )
    source_archive_sha = _write_source_tar(package_root, source_tuple)
    cache_sha = _copy(cache, package_root / "cache" / "top2000-daily-bars.pt")
    cache_manifest_sha = _copy(
        cache_manifest,
        package_root / "cache" / "cache-manifest.json",
    )
    risk_artifact_sha = _copy(
        risk_artifact,
        package_root / "risk" / "risk-exposures.pt",
    )
    risk_manifest_sha = _copy(
        risk_manifest,
        package_root / "risk" / "risk-source-manifest.json",
    )
    projector_file_sha = _copy(
        projector_manifest,
        package_root / "risk" / "projector-manifest.json",
    )
    copied_preflight_sha = _copy(
        structural_preflight,
        package_root / "plans" / "real-data-structural-preflight.json",
    )
    if copied_preflight_sha != structural_file_sha:
        raise M03RV14PackageBuildError("structural preflight changed during copy")
    initial_state_sha, initial_state_file_sha, architecture_sha = (
        write_m03r_v14_initial_parameter_state(
            package_root / "model" / "common-initial-parameter-state.pt",
            _initial_policy(),
        )
    )
    try:
        projector_payload = json.loads(projector_manifest.read_bytes())
        projector = projector_payload["projector"]
        binding = projector_payload["binding"]
        projector_sha256 = projector["manifest_sha256"]
        binding_sha256 = binding["binding_sha256"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise M03RV14PackageBuildError(
            "projector manifest lacks typed identities"
        ) from exc
    image_digest = image_reference.rsplit("@sha256:", 1)[-1]
    artifacts = M03RV14PackageArtifacts(
        source_archive_sha256=source_archive_sha,
        source_manifest_sha256=source_manifest_sha,
        dependency_lock_sha256=_file_sha256(source / "uv.lock"),
        cache_artifact_sha256=cache_sha,
        cache_manifest_sha256=cache_manifest_sha,
        risk_artifact_sha256=risk_artifact_sha,
        risk_source_manifest_file_sha256=risk_manifest_sha,
        projector_manifest_file_sha256=projector_file_sha,
        projector_manifest_sha256=projector_sha256,
        projector_binding_sha256=binding_sha256,
        worker_source_sha256=_file_sha256(source / M03R_V14_RUNTIME_WORKER),
        initial_parameter_state_file_sha256=initial_state_file_sha,
        initial_parameter_state_sha256=initial_state_sha,
        initial_parameter_architecture_sha256=architecture_sha,
        structural_preflight_file_sha256=structural_file_sha,
        structural_preflight_receipt_sha256=structural_receipt_sha,
        image_reference=image_reference,
        image_digest_sha256=image_digest,
    )
    geometries = render_m03r_v14_fold_geometries(1001)
    schedule = M03RV14PanelEpisodeSchedule(
        protocol_common_data_sha256=_sha256(
            {
                "cache": cache_sha,
                "risk": risk_artifact_sha,
                "risk_manifest": risk_manifest_sha,
                "projector": projector_file_sha,
                "preflight": structural_receipt_sha,
            }
        ),
        cache_sha256=cache_sha,
        asset_axis_sha256=structural_receipt.asset_axis_sha256,
        fold_geometry_sha256=tuple(row.receipt_sha256 for row in geometries),
    )
    if (
        structural_receipt.cache_sha256 != cache_sha
        or structural_receipt.fold_geometry_sha256
        != schedule.fold_geometry_sha256
    ):
        raise M03RV14PackageBuildError(
            "structural preflight does not bind packaged data or folds"
        )
    package = build_m03r_v14_package_plan(artifacts, schedule)
    load_m03r_v14_structural_preflight(
        package_root / "plans" / "real-data-structural-preflight.json",
        expected_file_sha256=structural_file_sha,
        expected_receipt_sha256=structural_receipt_sha,
    )
    plan_path = package_root / "plans" / "package-plan.json"
    plan_file_sha = write_m03r_v14_package_plan(plan_path, package)
    authorization = M03RV14ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=plan_file_sha,
        source_archive_sha256=source_archive_sha,
        source_manifest_sha256=source_manifest_sha,
        worker_source_sha256=artifacts.worker_source_sha256,
        image_reference=image_reference,
    )
    authorization_path = package_root / "plans" / "execution-authorization.json"
    authorization_file_sha = write_m03r_v14_execution_authorization(
        authorization_path,
        authorization,
        package,
    )
    execution_sha = _exclusive(
        package_root / "execution-manifest.json",
        _canonical(
            {
                "schema": M03R_V14_EXECUTION_MANIFEST_SCHEMA,
                "protocol_sha256": M03R_V14_PROTOCOL_SHA256,
                "package_plan_sha256": package.package_plan_sha256,
                "package_plan_file_sha256": plan_file_sha,
                "execution_authorization_receipt_sha256": (
                    authorization.receipt_sha256
                ),
                "execution_authorization_file_sha256": authorization_file_sha,
                "runtime_entrypoint": M03R_V14_RUNTIME_ENTRYPOINT,
                "predictive_settings": 2,
                "selected_horizon_sessions": M03R_V14_SELECTED_HORIZON_SESSIONS,
                "h100s_per_worker": 2,
                "maximum_h100_requests": 4,
                "economic_optimizer_updates": 0,
                "outer_2026_access_authorized": False,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
        ),
    )
    final_inventory = _inventory(package_root)
    unsigned = {
        "schema": M03R_V14_LOCAL_PACKAGE_SCHEMA,
        "package_relative_root": "package",
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": plan_file_sha,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "execution_authorization_file_sha256": authorization_file_sha,
        "source_archive_sha256": source_archive_sha,
        "source_manifest_sha256": source_manifest_sha,
        "cache_artifact_sha256": cache_sha,
        "cache_manifest_sha256": cache_manifest_sha,
        "risk_artifact_sha256": risk_artifact_sha,
        "risk_source_manifest_file_sha256": risk_manifest_sha,
        "projector_manifest_file_sha256": projector_file_sha,
        "worker_source_sha256": artifacts.worker_source_sha256,
        "initial_parameter_state_file_sha256": initial_state_file_sha,
        "initial_parameter_state_sha256": initial_state_sha,
        "initial_parameter_architecture_sha256": architecture_sha,
        "structural_preflight_file_sha256": structural_file_sha,
        "structural_preflight_receipt_sha256": structural_receipt_sha,
        "execution_manifest_sha256": execution_sha,
        "image_reference": image_reference,
        "image_digest_sha256": image_digest,
        "file_inventory": final_inventory,
        "source_file_count": len(source_tuple),
        "selected_horizon_sessions": M03R_V14_SELECTED_HORIZON_SESSIONS,
        "predictive_training_authorized": True,
        "economic_training_authorized": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256(unsigned)}
    _exclusive(output / "package-build-receipt.json", _canonical(receipt))
    validate_m03r_v14_local_package(output)
    return receipt


def validate_m03r_v14_local_package(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root)
    receipt_path = _regular(
        output / "package-build-receipt.json",
        "package receipt",
    )
    receipt = json.loads(receipt_path.read_bytes())
    if not isinstance(receipt, dict):
        raise M03RV14PackageBuildError("package receipt is not an object")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != M03R_V14_LOCAL_PACKAGE_SCHEMA
        or receipt.get("receipt_sha256") != _sha256(unsigned)
        or receipt.get("predictive_training_authorized") is not True
        or receipt.get("economic_training_authorized") is not False
        or receipt.get("outer_2026_access_authorized") is not False
        or receipt.get("selected_horizon_sessions")
        != M03R_V14_SELECTED_HORIZON_SESSIONS
    ):
        raise M03RV14PackageBuildError("package receipt semantics drifted")
    package_root = output / "package"
    if tuple(receipt.get("file_inventory", ())) != _inventory(package_root):
        raise M03RV14PackageBuildError("package inventory drifted")
    plan = load_m03r_v14_package_plan(
        package_root / "plans" / "package-plan.json",
        expected_file_sha256=receipt["package_plan_file_sha256"],
    )
    authorization = load_m03r_v14_execution_authorization(
        package_root / "plans" / "execution-authorization.json",
        expected_file_sha256=receipt["execution_authorization_file_sha256"],
        package=plan,
    )
    load_m03r_v14_structural_preflight(
        package_root / "plans" / "real-data-structural-preflight.json",
        expected_file_sha256=receipt["structural_preflight_file_sha256"],
        expected_receipt_sha256=receipt["structural_preflight_receipt_sha256"],
    )
    if (
        authorization.receipt_sha256
        != receipt["execution_authorization_receipt_sha256"]
        or plan.artifacts.worker_source_sha256 != receipt["worker_source_sha256"]
        or plan.artifacts.initial_parameter_architecture_sha256
        != receipt["initial_parameter_architecture_sha256"]
    ):
        raise M03RV14PackageBuildError("package authorization binding drifted")
    _safe_tar(package_root / "source.tar")
    return receipt


def build_m03r_v14_transfer_archive(
    output_root: str | Path,
    archive_path: str | Path,
) -> dict[str, Any]:
    receipt = validate_m03r_v14_local_package(output_root)
    root = Path(output_root)
    target = Path(archive_path)
    if target.exists() or target.is_symlink():
        raise M03RV14PackageBuildError("transfer archive already exists")
    files = (root / "package-build-receipt.json", *(root / "package").rglob("*"))
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as archive:
        for source in sorted(
            (path for path in files if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        ):
            relative = source.relative_to(root).as_posix()
            info = tarfile.TarInfo(f"{M03R_V14_TRANSFER_ROOT}/{relative}")
            info.size = source.stat().st_size
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    archive_sha = _file_sha256(target)
    validate_m03r_v14_transfer_archive(
        target,
        expected_archive_sha256=archive_sha,
        expected_package_receipt_file_sha256=_file_sha256(
            root / "package-build-receipt.json"
        ),
    )
    return {
        "archive_sha256": archive_sha,
        "package_receipt_sha256": receipt["receipt_sha256"],
    }


def validate_m03r_v14_transfer_archive(
    archive_path: str | Path,
    *,
    expected_archive_sha256: str,
    expected_package_receipt_file_sha256: str,
) -> None:
    path = _regular(Path(archive_path), "transfer archive")
    if _file_sha256(path) != expected_archive_sha256:
        raise M03RV14PackageBuildError("transfer archive hash drifted")
    _safe_tar(path, required_root=M03R_V14_TRANSFER_ROOT)
    receipt_name = f"{M03R_V14_TRANSFER_ROOT}/package-build-receipt.json"
    observed: dict[str, tuple[int, str]] = {}
    content: bytes | None = None
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise M03RV14PackageBuildError("transfer member is unreadable")
            digest = hashlib.sha256()
            payload = bytearray() if member.name == receipt_name else None
            while block := stream.read(1024 * 1024):
                digest.update(block)
                if payload is not None:
                    payload.extend(block)
            observed[member.name] = (member.size, digest.hexdigest())
            if payload is not None:
                content = bytes(payload)
    if content is None:
        raise M03RV14PackageBuildError("transfer receipt is absent")
    if hashlib.sha256(content).hexdigest() != expected_package_receipt_file_sha256:
        raise M03RV14PackageBuildError("transfer package receipt hash drifted")
    receipt = json.loads(content)
    if not isinstance(receipt, dict):
        raise M03RV14PackageBuildError("transfer package receipt is malformed")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("schema") != M03R_V14_LOCAL_PACKAGE_SCHEMA or receipt.get(
        "receipt_sha256"
    ) != _sha256(unsigned):
        raise M03RV14PackageBuildError("transfer package receipt semantics drifted")
    expected = {
        receipt_name: (len(content), expected_package_receipt_file_sha256),
        **{
            f"{M03R_V14_TRANSFER_ROOT}/package/{row['path']}": (
                row["size"],
                row["sha256"],
            )
            for row in receipt.get("file_inventory", ())
        },
    }
    if observed != expected:
        raise M03RV14PackageBuildError("transfer archive inventory drifted")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-root", required=True)
    build.add_argument("--cache", required=True)
    build.add_argument("--cache-manifest", required=True)
    build.add_argument("--risk-root", required=True)
    build.add_argument("--structural-preflight", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--image-reference", default=PINNED_QUANTTRADE_IMAGE)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-root", required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--output-root", required=True)
    archive.add_argument("--archive-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        build_m03r_v14_local_package(
            source_root=args.source_root,
            cache_path=args.cache,
            cache_manifest_path=args.cache_manifest,
            risk_root=args.risk_root,
            structural_preflight_path=args.structural_preflight,
            output_root=args.output_root,
            image_reference=args.image_reference,
        )
    elif args.command == "validate":
        validate_m03r_v14_local_package(args.output_root)
    else:
        build_m03r_v14_transfer_archive(args.output_root, args.archive_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V14_EXECUTION_MANIFEST_SCHEMA",
    "M03R_V14_LOCAL_PACKAGE_SCHEMA",
    "M03R_V14_SOURCE_MANIFEST_SCHEMA",
    "M03R_V14_TRANSFER_ROOT",
    "M03RV14PackageBuildError",
    "PINNED_QUANTTRADE_IMAGE",
    "build_m03r_v14_local_package",
    "build_m03r_v14_transfer_archive",
    "main",
    "validate_m03r_v14_local_package",
    "validate_m03r_v14_transfer_archive",
]
