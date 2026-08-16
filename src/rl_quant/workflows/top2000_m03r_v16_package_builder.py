"""Build and validate one immutable local M03R-v16 predictive package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
)
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_initial_state import (
    write_m03r_v16_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03R_V16_RUNTIME_ENTRYPOINT,
    M03RV16ExecutionAuthorization,
    M03RV16PackageArtifacts,
    build_m03r_v16_package_plan,
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
    write_m03r_v16_execution_authorization,
    write_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    load_m03r_v16_structural_slab,
)

M03R_V16_LOCAL_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-local-predictive-package-v1"
)
M03R_V16_SOURCE_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-runtime-source-manifest-v1"
)
M03R_V16_EXECUTION_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-execution-manifest-v1"
)
M03R_V16_TRANSFER_ARCHIVE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-transfer-archive-v1"
)
M03R_V16_RUNTIME_WORKER = "src/rl_quant/workflows/top2000_m03r_v16_predictive.py"
M03R_V16_STRUCTURAL_BUILDER = (
    "src/rl_quant/workflows/top2000_m03r_v16_structural_build.py"
)
M03R_V16_OPERATOR_SOURCE = (
    "src/rl_quant/training/top2000_m03r_v15_residual_operator.py"
)
PINNED_QUANTTRADE_IMAGE = (
    "hpcharbor.mdanderson.edu/yding41/ml2:quanttrade-ppo-cu124-py311-85cf781d3e08"
    "@sha256:7cff8faedcfb44ad25e1001d7e1634569f7cd3f5365bbd8ff8caa9b10d8bcdf9"
)


class M03RV16PackageBuildError(ValueError):
    """A V16 package member, identity, or no-clobber boundary drifted."""


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
        raise M03RV16PackageBuildError(f"{name} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise M03RV16PackageBuildError(f"{name} must be a nonempty regular file")
    return path


def _exclusive(path: Path, data: bytes, mode: int = 0o440) -> str:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except OSError as exc:
        raise M03RV16PackageBuildError("V16 immutable target already exists") from exc
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
        raise M03RV16PackageBuildError("V16 source changed during copy")
    return digest.hexdigest()


def _source_members(root: Path) -> tuple[Path, ...]:
    required = (
        root / "pyproject.toml",
        root / "uv.lock",
        root / M03R_V16_RUNTIME_WORKER,
        root / M03R_V16_STRUCTURAL_BUILDER,
        root / M03R_V16_OPERATOR_SOURCE,
    )
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
            raise M03RV16PackageBuildError("V16 source member leaves repository")
        result.append(path)
    if len(result) != len(set(result)):
        raise M03RV16PackageBuildError("V16 source inventory has duplicates")
    return tuple(result)


def _inventory(root: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise M03RV16PackageBuildError("V16 package contains a symlink")
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
        for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
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


def _safe_source_tar(path: Path) -> None:
    names: set[str] = set()
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or not pure.parts
                or pure.parts[0] != "source"
                or ".." in pure.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
                or not (member.isfile() or member.isdir())
                or member.name in names
            ):
                raise M03RV16PackageBuildError("V16 source archive is unsafe")
            names.add(member.name)


def _initial_policy() -> Top2000M03RV16PredictivePolicy:
    random.seed(M03R_V16_PREDICTIVE_SPEC.seed)
    torch.manual_seed(M03R_V16_PREDICTIVE_SPEC.seed)
    return Top2000M03RV16PredictivePolicy(0)


def _read_structural_build_receipt(path: Path) -> dict[str, Any]:
    _regular(path, "structural build receipt")
    try:
        receipt = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise M03RV16PackageBuildError(
            "V16 structural build receipt is malformed"
        ) from exc
    if not isinstance(receipt, dict):
        raise M03RV16PackageBuildError("V16 structural receipt is not an object")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(unsigned):
        raise M03RV16PackageBuildError("V16 structural receipt hash drifted")
    return receipt


def _run_package_owned_structural_slab(
    package_root: Path,
    *,
    cache_sha256: str,
    cache_manifest_sha256: str,
    source_manifest_sha256: str,
    operator_source_sha256: str,
    risk_artifact_file_sha256: str,
    risk_source_manifest_file_sha256: str,
    projector_manifest_file_sha256: str,
) -> dict[str, Any]:
    slab = package_root / "structural" / "structural-slab.pt"
    receipt = package_root / "plans" / "structural-slab-build.json"
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = (
        sys.executable,
        "-I",
        "-c",
        (
            "import sys;"
            "sys.path.insert(0,sys.argv.pop(1));"
            "from rl_quant.workflows.top2000_m03r_v16_structural_build "
            "import main;"
            "raise SystemExit(main())"
        ),
        str(package_root / "source" / "src"),
        "--cache",
        str(package_root / "cache" / "top2000-daily-bars.pt"),
        "--cache-sha256",
        cache_sha256,
        "--cache-manifest-sha256",
        cache_manifest_sha256,
        "--risk-manifest",
        str(package_root / "risk" / "risk-source-manifest.json"),
        "--risk-manifest-file-sha256",
        risk_source_manifest_file_sha256,
        "--projector-manifest",
        str(package_root / "risk" / "projector-manifest.json"),
        "--projector-manifest-file-sha256",
        projector_manifest_file_sha256,
        "--source-manifest-sha256",
        source_manifest_sha256,
        "--operator-source-sha256",
        operator_source_sha256,
        "--risk-artifact-file-sha256",
        risk_artifact_file_sha256,
        "--output-slab",
        str(slab),
        "--output-receipt",
        str(receipt),
    )
    try:
        subprocess.run(
            command,
            cwd=package_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise M03RV16PackageBuildError(
            "package-owned V16 structural builder failed"
        ) from exc
    return _read_structural_build_receipt(receipt)


def build_m03r_v16_local_package(
    *,
    source_root: str | Path,
    cache_path: str | Path,
    cache_manifest_path: str | Path,
    risk_root: str | Path,
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
        risk / "projector-manifest.json", "projector manifest"
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
                "schema": M03R_V16_SOURCE_MANIFEST_SCHEMA,
                "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
                "file_count": len(source_tuple),
                "files": source_tuple,
                "runtime_worker": M03R_V16_RUNTIME_WORKER,
                "structural_builder": M03R_V16_STRUCTURAL_BUILDER,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
        ),
    )
    source_archive_sha = _write_source_tar(package_root, source_tuple)
    cache_sha = _copy(cache, package_root / "cache" / "top2000-daily-bars.pt")
    cache_manifest_sha = _copy(
        cache_manifest, package_root / "cache" / "cache-manifest.json"
    )
    risk_artifact_sha = _copy(
        risk_artifact, package_root / "risk" / "risk-exposures.pt"
    )
    risk_manifest_sha = _copy(
        risk_manifest, package_root / "risk" / "risk-source-manifest.json"
    )
    projector_file_sha = _copy(
        projector_manifest, package_root / "risk" / "projector-manifest.json"
    )
    initial_state_sha, initial_file_sha, architecture_sha = (
        write_m03r_v16_initial_parameter_state(
            package_root / "model" / "common-initial-parameter-state.pt",
            _initial_policy(),
        )
    )
    operator_source_sha = next(
        row["sha256"]
        for row in source_tuple
        if row["path"] == M03R_V16_OPERATOR_SOURCE
    )
    structural_build = _run_package_owned_structural_slab(
        package_root,
        cache_sha256=cache_sha,
        cache_manifest_sha256=cache_manifest_sha,
        source_manifest_sha256=source_manifest_sha,
        operator_source_sha256=operator_source_sha,
        risk_artifact_file_sha256=risk_artifact_sha,
        risk_source_manifest_file_sha256=risk_manifest_sha,
        projector_manifest_file_sha256=projector_file_sha,
    )
    structural = load_m03r_v16_structural_slab(
        package_root / "structural" / "structural-slab.pt",
        expected_file_sha256=structural_build["slab_file_sha256"],
        expected_receipt_sha256=structural_build["slab_receipt_sha256"],
    )
    receipt = structural.slab.receipt
    image_digest = image_reference.rsplit("@sha256:", 1)[-1]
    artifacts = M03RV16PackageArtifacts(
        source_archive_sha256=source_archive_sha,
        source_manifest_sha256=source_manifest_sha,
        dependency_lock_sha256=_file_sha256(source / "uv.lock"),
        cache_artifact_sha256=cache_sha,
        cache_manifest_sha256=cache_manifest_sha,
        asset_axis_sha256=receipt.asset_axis_sha256,
        risk_artifact_sha256=risk_artifact_sha,
        risk_source_manifest_file_sha256=risk_manifest_sha,
        risk_source_receipt_sha256=receipt.risk_source_receipt_sha256,
        exposure_receipt_sha256=receipt.exposure_receipt_sha256,
        projector_manifest_file_sha256=projector_file_sha,
        projector_manifest_sha256=receipt.projector_manifest_sha256,
        projector_binding_sha256=receipt.projector_binding_sha256,
        worker_source_sha256=next(
            row["sha256"]
            for row in source_tuple
            if row["path"] == M03R_V16_RUNTIME_WORKER
        ),
        operator_source_sha256=operator_source_sha,
        initial_parameter_state_file_sha256=initial_file_sha,
        initial_parameter_state_sha256=initial_state_sha,
        initial_parameter_architecture_sha256=architecture_sha,
        structural_slab_file_sha256=structural_build["slab_file_sha256"],
        structural_slab_receipt_sha256=receipt.receipt_sha256,
        structural_action_operator_root_sha256=receipt.action_operator_root_sha256,
        structural_target_operator_root_sha256=(
            receipt.common_target_operator_root_sha256
        ),
        structural_target_root_sha256=receipt.target_root_sha256,
        image_reference=image_reference,
        image_digest_sha256=image_digest,
    )
    schedule = M03RV16PanelSchedule(
        protocol_common_data_sha256=_sha256(
            {
                "cache": cache_sha,
                "risk": risk_artifact_sha,
                "risk_manifest": risk_manifest_sha,
                "projector": projector_file_sha,
                "structural_slab": receipt.receipt_sha256,
            }
        ),
        cache_sha256=cache_sha,
        asset_axis_sha256=receipt.asset_axis_sha256,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v16_fold_geometries(1001)
        ),
    )
    if receipt.fold_geometry_sha256 != schedule.fold_geometry_sha256:
        raise M03RV16PackageBuildError("V16 slab fold geometry drifted")
    package = build_m03r_v16_package_plan(artifacts, schedule)
    plan_path = package_root / "plans" / "package-plan.json"
    plan_file_sha = write_m03r_v16_package_plan(plan_path, package)
    authorization = M03RV16ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=plan_file_sha,
        source_archive_sha256=source_archive_sha,
        source_manifest_sha256=source_manifest_sha,
        worker_source_sha256=artifacts.worker_source_sha256,
        structural_slab_file_sha256=artifacts.structural_slab_file_sha256,
        structural_slab_receipt_sha256=artifacts.structural_slab_receipt_sha256,
        image_reference=image_reference,
    )
    authorization_file_sha = write_m03r_v16_execution_authorization(
        package_root / "plans" / "execution-authorization.json",
        authorization,
        package,
    )
    execution_sha = _exclusive(
        package_root / "execution-manifest.json",
        _canonical(
            {
                "schema": M03R_V16_EXECUTION_MANIFEST_SCHEMA,
                "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
                "package_plan_sha256": package.package_plan_sha256,
                "package_plan_file_sha256": plan_file_sha,
                "execution_authorization_receipt_sha256": (
                    authorization.receipt_sha256
                ),
                "execution_authorization_file_sha256": authorization_file_sha,
                "runtime_entrypoint": M03R_V16_RUNTIME_ENTRYPOINT,
                "predictive_settings": 3,
                "primary_setting_index": 2,
                "h100s_per_worker": 2,
                "maximum_h100_requests": 6,
                "hold_target_sessions": (
                    LEGACY_HOLD30_TARGET_SPEC.target_sessions
                ),
                "hold_target_spec_sha256": (
                    LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
                ),
                "economic_optimizer_updates": 0,
                "reinforcement_learning_updates": 0,
                "outer_2026_access_authorized": False,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
        ),
    )
    final_inventory = _inventory(package_root)
    unsigned = {
        "schema": M03R_V16_LOCAL_PACKAGE_SCHEMA,
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
        "initial_parameter_state_file_sha256": initial_file_sha,
        "initial_parameter_state_sha256": initial_state_sha,
        "initial_parameter_architecture_sha256": architecture_sha,
        "structural_slab_file_sha256": artifacts.structural_slab_file_sha256,
        "structural_slab_receipt_sha256": artifacts.structural_slab_receipt_sha256,
        "structural_build_receipt_sha256": structural_build["receipt_sha256"],
        "execution_manifest_sha256": execution_sha,
        "image_reference": image_reference,
        "image_digest_sha256": image_digest,
        "file_inventory": final_inventory,
        "source_file_count": len(source_tuple),
        "hold_target_sessions": LEGACY_HOLD30_TARGET_SPEC.target_sessions,
        "hold_target_spec_sha256": LEGACY_HOLD30_TARGET_SPEC.receipt_sha256,
        "predictive_training_authorized": True,
        "economic_training_authorized": False,
        "reinforcement_learning_authorized": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    build_receipt = {**unsigned, "receipt_sha256": _sha256(unsigned)}
    _exclusive(output / "package-build-receipt.json", _canonical(build_receipt))
    validate_m03r_v16_local_package(output)
    return build_receipt


def validate_m03r_v16_local_package(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root)
    receipt_path = _regular(
        output / "package-build-receipt.json", "package receipt"
    )
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise M03RV16PackageBuildError("V16 package receipt is malformed") from exc
    if not isinstance(receipt, dict):
        raise M03RV16PackageBuildError("V16 package receipt is not an object")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != M03R_V16_LOCAL_PACKAGE_SCHEMA
        or receipt.get("receipt_sha256") != _sha256(unsigned)
        or receipt.get("predictive_training_authorized") is not True
        or receipt.get("economic_training_authorized") is not False
        or receipt.get("reinforcement_learning_authorized") is not False
        or receipt.get("outer_2026_access_authorized") is not False
        or receipt.get("hold_target_sessions")
        != LEGACY_HOLD30_TARGET_SPEC.target_sessions
        or receipt.get("hold_target_spec_sha256")
        != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    ):
        raise M03RV16PackageBuildError("V16 package receipt semantics drifted")
    package_root = output / "package"
    if tuple(receipt.get("file_inventory", ())) != _inventory(package_root):
        raise M03RV16PackageBuildError("V16 package inventory drifted")
    plan = load_m03r_v16_package_plan(
        package_root / "plans" / "package-plan.json",
        expected_file_sha256=receipt["package_plan_file_sha256"],
    )
    authorization = load_m03r_v16_execution_authorization(
        package_root / "plans" / "execution-authorization.json",
        expected_file_sha256=receipt["execution_authorization_file_sha256"],
        package=plan,
    )
    structural = load_m03r_v16_structural_slab(
        package_root / "structural" / "structural-slab.pt",
        expected_file_sha256=receipt["structural_slab_file_sha256"],
        expected_receipt_sha256=receipt["structural_slab_receipt_sha256"],
    )
    structural.slab.receipt.validate_for_package(
        cache_sha256=plan.artifacts.cache_artifact_sha256,
        cache_manifest_sha256=plan.artifacts.cache_manifest_sha256,
        asset_axis_sha256=plan.artifacts.asset_axis_sha256,
        source_manifest_sha256=plan.artifacts.source_manifest_sha256,
        operator_source_sha256=plan.artifacts.operator_source_sha256,
        risk_artifact_file_sha256=plan.artifacts.risk_artifact_sha256,
        risk_source_manifest_file_sha256=(
            plan.artifacts.risk_source_manifest_file_sha256
        ),
        risk_source_receipt_sha256=plan.artifacts.risk_source_receipt_sha256,
        exposure_receipt_sha256=plan.artifacts.exposure_receipt_sha256,
        projector_manifest_file_sha256=(
            plan.artifacts.projector_manifest_file_sha256
        ),
        projector_manifest_sha256=plan.artifacts.projector_manifest_sha256,
        projector_binding_sha256=plan.artifacts.projector_binding_sha256,
    )
    if (
        authorization.receipt_sha256
        != receipt["execution_authorization_receipt_sha256"]
        or plan.artifacts.worker_source_sha256 != receipt["worker_source_sha256"]
        or plan.artifacts.initial_parameter_architecture_sha256
        != receipt["initial_parameter_architecture_sha256"]
    ):
        raise M03RV16PackageBuildError("V16 package authority binding drifted")
    _safe_source_tar(package_root / "source.tar")
    load_verified_top2000_hold30_development_cache(
        package_root / "cache" / "top2000-daily-bars.pt",
        expected_cache_sha256=plan.artifacts.cache_artifact_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    return receipt


def _safe_transfer_member(member: tarfile.TarInfo, seen: set[str]) -> None:
    pure = PurePosixPath(member.name)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.parts[0] != "m03r-v16-package"
        or ".." in pure.parts
        or member.issym()
        or member.islnk()
        or member.isdev()
        or not (member.isfile() or member.isdir())
        or member.name in seen
    ):
        raise M03RV16PackageBuildError("V16 transfer archive is unsafe")
    seen.add(member.name)


def validate_m03r_v16_transfer_archive(
    archive_path: str | Path,
    *,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    archive = _regular(Path(archive_path), "transfer archive")
    if _file_sha256(archive) != expected_archive_sha256:
        raise M03RV16PackageBuildError("V16 transfer archive hash drifted")
    seen: set[str] = set()
    files: dict[str, tuple[str, int]] = {}
    receipt: dict[str, Any] | None = None
    with tarfile.open(archive, "r") as stream:
        for member in stream.getmembers():
            _safe_transfer_member(member, seen)
            if not member.isfile():
                continue
            source = stream.extractfile(member)
            if source is None:
                raise M03RV16PackageBuildError("V16 transfer member is unreadable")
            digest = hashlib.sha256()
            size = 0
            while block := source.read(1024 * 1024):
                digest.update(block)
                size += len(block)
            relative = PurePosixPath(member.name).relative_to("m03r-v16-package")
            files[relative.as_posix()] = (digest.hexdigest(), size)
            if relative.as_posix() == "package-build-receipt.json":
                receipt_source = stream.extractfile(member)
                if receipt_source is None:
                    raise M03RV16PackageBuildError(
                        "V16 package receipt is unreadable"
                    )
                try:
                    parsed = json.loads(receipt_source.read())
                except json.JSONDecodeError as exc:
                    raise M03RV16PackageBuildError(
                        "V16 transfer package receipt is malformed"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise M03RV16PackageBuildError(
                        "V16 transfer package receipt is not an object"
                    )
                receipt = parsed
    if receipt is None:
        raise M03RV16PackageBuildError("V16 transfer package receipt is absent")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected_files = {
        f"package/{row['path']}": (row["sha256"], row["size"])
        for row in receipt.get("file_inventory", ())
    }
    receipt_bytes = _canonical(receipt)
    expected_files["package-build-receipt.json"] = (
        hashlib.sha256(receipt_bytes).hexdigest(),
        len(receipt_bytes),
    )
    if (
        receipt.get("receipt_sha256") != _sha256(unsigned)
        or files != expected_files
    ):
        raise M03RV16PackageBuildError("V16 transfer inventory drifted")
    return {
        "schema": M03R_V16_TRANSFER_ARCHIVE_SCHEMA,
        "archive_sha256": expected_archive_sha256,
        "package_receipt_sha256": receipt["receipt_sha256"],
        "regular_file_count": len(files),
        "safe_member_inventory_verified": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }


def build_m03r_v16_transfer_archive(
    output_root: str | Path,
    archive_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_root)
    validate_m03r_v16_local_package(output)
    target = Path(archive_path)
    if target.exists() or target.is_symlink():
        raise M03RV16PackageBuildError("V16 transfer target already exists")
    files = tuple(
        sorted(
            (
                output / "package-build-receipt.json",
                *(
                    path
                    for path in (output / "package").rglob("*")
                    if path.is_file()
                ),
            ),
            key=lambda path: path.relative_to(output).as_posix(),
        )
    )
    directories = {"m03r-v16-package"}
    for path in files:
        _regular(path, str(path.relative_to(output)))
        parent = PurePosixPath("m03r-v16-package", path.relative_to(output)).parent
        while str(parent) not in {"", "."}:
            directories.add(str(parent))
            parent = parent.parent
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as archive:
        for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
            info = tarfile.TarInfo(directory + "/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            archive.addfile(info)
        for source in files:
            name = PurePosixPath("m03r-v16-package", source.relative_to(output))
            info = tarfile.TarInfo(name.as_posix())
            info.size = source.stat().st_size
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with source.open("rb") as reader:
                archive.addfile(info, reader)
    os.chmod(target, 0o440)
    archive_sha256 = _file_sha256(target)
    return validate_m03r_v16_transfer_archive(
        target,
        expected_archive_sha256=archive_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-root", required=True)
    build.add_argument("--cache", required=True)
    build.add_argument("--cache-manifest", required=True)
    build.add_argument("--risk-root", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--image-reference", default=PINNED_QUANTTRADE_IMAGE)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-root", required=True)
    transfer = commands.add_parser("transfer")
    transfer.add_argument("--output-root", required=True)
    transfer.add_argument("--archive", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        build_m03r_v16_local_package(
            source_root=args.source_root,
            cache_path=args.cache,
            cache_manifest_path=args.cache_manifest,
            risk_root=args.risk_root,
            output_root=args.output_root,
            image_reference=args.image_reference,
        )
    elif args.command == "validate":
        validate_m03r_v16_local_package(args.output_root)
    else:
        build_m03r_v16_transfer_archive(args.output_root, args.archive)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V16_EXECUTION_MANIFEST_SCHEMA",
    "M03R_V16_LOCAL_PACKAGE_SCHEMA",
    "M03R_V16_SOURCE_MANIFEST_SCHEMA",
    "M03R_V16_TRANSFER_ARCHIVE_SCHEMA",
    "M03RV16PackageBuildError",
    "PINNED_QUANTTRADE_IMAGE",
    "build_m03r_v16_local_package",
    "build_m03r_v16_transfer_archive",
    "main",
    "validate_m03r_v16_local_package",
    "validate_m03r_v16_transfer_archive",
]
