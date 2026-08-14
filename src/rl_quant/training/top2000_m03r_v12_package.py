"""Content-bound local package plan for the M03R-v12 predictive panel."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_DESIGN_ID,
    M03R_V12_PROTOCOL_GENERATION,
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v12_predictive_worker import (
    M03RV12PredictivePanelPlan,
    M03RV12PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v12_schedule import M03RV12PanelEpisodeSchedule

M03R_V12_PACKAGE_SCHEMA = "rl-quant.top2000-dev.m03r-v12-predictive-package-v1"
M03R_V12_PACKAGE_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-predictive-package-file-v1"
)
M03R_V12_EXECUTION_AUTHORIZATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-predictive-execution-authorization-v1"
)
M03R_V12_PACKAGE_SOURCE_PYTHONPATH = "/mnt/package/source/src"
M03R_V12_RUNTIME_ENTRYPOINT = "rl_quant.workflows.top2000_m03r_v12_predictive"
_MAX_PACKAGE_PLAN_BYTES = 4 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^[^@\s]+@sha256:([0-9a-f]{64})$")


class M03RV12PackageError(ValueError):
    """The v12 source, data, risk, image, or panel package drifted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(name: str, value: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise M03RV12PackageError(f"{name} must be one lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV12PackageArtifacts:
    source_archive_sha256: str
    source_manifest_sha256: str
    dependency_lock_sha256: str
    cache_artifact_sha256: str
    cache_manifest_sha256: str
    risk_artifact_sha256: str
    risk_source_manifest_file_sha256: str
    projector_manifest_file_sha256: str
    projector_manifest_sha256: str
    projector_binding_sha256: str
    worker_source_sha256: str
    initial_parameter_state_file_sha256: str
    initial_parameter_state_sha256: str
    structural_preflight_file_sha256: str
    structural_preflight_receipt_sha256: str
    image_reference: str
    image_digest_sha256: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "image_reference":
                continue
            _digest(name, value)
        match = _IMAGE.fullmatch(self.image_reference)
        if match is None or match.group(1) != self.image_digest_sha256:
            raise M03RV12PackageError("v12 image reference must be digest pinned")


@dataclass(frozen=True, slots=True)
class M03RV12PackagePlan:
    artifacts: M03RV12PackageArtifacts
    schedule: M03RV12PanelEpisodeSchedule
    panel: M03RV12PredictivePanelPlan
    plan_directory: str
    package_plan_sha256: str
    source_pythonpath: str = M03R_V12_PACKAGE_SOURCE_PYTHONPATH
    protocol_generation: str = M03R_V12_PROTOCOL_GENERATION
    design_id: str = M03R_V12_DESIGN_ID
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    structural_gates_complete: bool = True
    package_authorized: bool = False
    kubernetes_launch_authorized: bool = False
    outer_2026_access_authorized: bool = False
    economic_panel_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V12_PACKAGE_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifacts": asdict(self.artifacts),
            "schedule": asdict(self.schedule),
            "panel": asdict(self.panel),
            "plan_directory": self.plan_directory,
            "source_pythonpath": self.source_pythonpath,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "protocol_sha256": self.protocol_sha256,
            "structural_gates_complete": self.structural_gates_complete,
            "package_authorized": self.package_authorized,
            "kubernetes_launch_authorized": self.kubernetes_launch_authorized,
            "outer_2026_access_authorized": self.outer_2026_access_authorized,
            "economic_panel_authorized": self.economic_panel_authorized,
            "development_only": self.development_only,
            "reportable": self.reportable,
            "promotion_eligible": self.promotion_eligible,
        }

    def validate(self) -> None:
        self.artifacts.validate()
        self.schedule.validate()
        self.panel.validate()
        expected_paths = {
            "cache_path": "/mnt/package/cache/top2000-daily-bars.pt",
            "initial_parameter_state_path": (
                "/mnt/package/model/common-initial-parameter-state.pt"
            ),
            "risk_source_manifest_path": (
                "/mnt/package/risk/risk-source-manifest.json"
            ),
            "projector_manifest_path": "/mnt/package/risk/projector-manifest.json",
            "structural_preflight_path": (
                "/mnt/package/plans/real-data-structural-preflight.json"
            ),
        }
        for completion, worker in enumerate(self.panel.workers):
            if (
                worker.output_root
                != f"/mnt/output/completion-{completion:02d}-setting-{completion:02d}"
                or any(
                    getattr(worker, name) != value
                    for name, value in expected_paths.items()
                )
                or worker.panel_episode_schedule_sha256 != self.schedule.receipt_sha256
                or worker.selected_horizon_sessions != 3
                or worker.initial_parameter_state_sha256
                != self.artifacts.initial_parameter_state_sha256
                or worker.initial_parameter_state_file_sha256
                != self.artifacts.initial_parameter_state_file_sha256
                or worker.cache_sha256 != self.artifacts.cache_artifact_sha256
                or worker.risk_source_manifest_file_sha256
                != self.artifacts.risk_source_manifest_file_sha256
                or worker.projector_manifest_file_sha256
                != self.artifacts.projector_manifest_file_sha256
                or worker.projector_manifest_sha256
                != self.artifacts.projector_manifest_sha256
                or worker.projector_binding_sha256
                != self.artifacts.projector_binding_sha256
                or worker.source_manifest_sha256
                != self.artifacts.source_manifest_sha256
                or worker.source_archive_sha256 != self.artifacts.source_archive_sha256
                or worker.structural_preflight_file_sha256
                != self.artifacts.structural_preflight_file_sha256
                or worker.structural_preflight_receipt_sha256
                != self.artifacts.structural_preflight_receipt_sha256
            ):
                raise M03RV12PackageError(
                    "v12 worker does not bind exact package artifacts"
                )
        if (
            self.schedule.cache_sha256 != self.artifacts.cache_artifact_sha256
            or PurePosixPath(self.plan_directory) != PurePosixPath("/mnt/package/plans")
            or self.source_pythonpath != M03R_V12_PACKAGE_SOURCE_PYTHONPATH
            or self.protocol_generation != M03R_V12_PROTOCOL_GENERATION
            or self.design_id != M03R_V12_DESIGN_ID
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or not self.structural_gates_complete
            or self.package_authorized
            or self.kubernetes_launch_authorized
            or self.outer_2026_access_authorized
            or self.economic_panel_authorized
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V12_PACKAGE_SCHEMA
            or self.package_plan_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV12PackageError("v12 package plan identity drifted")


@dataclass(frozen=True, slots=True)
class M03RV12ExecutionAuthorization:
    package_plan_sha256: str
    package_plan_file_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    worker_source_sha256: str
    image_reference: str
    runtime_entrypoint: str = M03R_V12_RUNTIME_ENTRYPOINT
    indexed_completions: int = 3
    parallelism: int = 3
    h100s_per_completion: int = 2
    maximum_h100_requests: int = 6
    predictive_training_authorized: bool = True
    economic_training_authorized: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    schema: str = M03R_V12_EXECUTION_AUTHORIZATION_SCHEMA

    def validate(self, package: M03RV12PackagePlan) -> None:
        package.validate()
        for name in (
            "package_plan_sha256",
            "package_plan_file_sha256",
            "source_archive_sha256",
            "source_manifest_sha256",
            "worker_source_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.package_plan_sha256 != package.package_plan_sha256
            or self.source_archive_sha256 != package.artifacts.source_archive_sha256
            or self.source_manifest_sha256 != package.artifacts.source_manifest_sha256
            or self.worker_source_sha256 != package.artifacts.worker_source_sha256
            or self.image_reference != package.artifacts.image_reference
            or self.runtime_entrypoint != M03R_V12_RUNTIME_ENTRYPOINT
            or self.indexed_completions != package.panel.indexed_completions
            or self.parallelism != package.panel.parallelism
            or self.h100s_per_completion != package.panel.h100s_per_completion
            or self.maximum_h100_requests != package.panel.maximum_h100_requests
            or self.parallelism * self.h100s_per_completion
            != self.maximum_h100_requests
            or not self.predictive_training_authorized
            or self.economic_training_authorized
            or self.outer_2026_access_authorized
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.schema != M03R_V12_EXECUTION_AUTHORIZATION_SCHEMA
        ):
            raise M03RV12PackageError("v12 execution authorization drifted")

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


def write_m03r_v12_execution_authorization(
    path: str | Path,
    authorization: M03RV12ExecutionAuthorization,
    package: M03RV12PackagePlan,
) -> str:
    authorization.validate(package)
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV12PackageError("v12 execution authorization already exists")
    payload = {
        "authorization": asdict(authorization),
        "receipt_sha256": authorization.receipt_sha256,
    }
    encoded = _canonical(payload) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def load_m03r_v12_execution_authorization(
    path: str | Path,
    *,
    expected_file_sha256: str,
    package: M03RV12PackagePlan,
) -> M03RV12ExecutionAuthorization:
    expected_file_sha256 = _digest(
        "expected_authorization_file_sha256", expected_file_sha256
    )
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV12PackageError(
            "v12 execution authorization must be a regular non-symlink"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_PACKAGE_PLAN_BYTES
        ):
            raise M03RV12PackageError("v12 execution authorization is invalid")
        content = bytearray()
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            content.extend(block)
        after = os.fstat(descriptor)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV12PackageError("v12 execution authorization hash drifted")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV12PackageError("v12 execution authorization changed while read")
    finally:
        os.close(descriptor)
    payload = json.loads(bytes(content))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("authorization"), dict
    ):
        raise M03RV12PackageError("v12 execution authorization schema drifted")
    authorization = M03RV12ExecutionAuthorization(**payload["authorization"])
    authorization.validate(package)
    if payload.get("receipt_sha256") != authorization.receipt_sha256:
        raise M03RV12PackageError("v12 execution authorization receipt drifted")
    return authorization


def build_m03r_v12_package_plan(
    artifacts: M03RV12PackageArtifacts,
    schedule: M03RV12PanelEpisodeSchedule,
) -> M03RV12PackagePlan:
    artifacts.validate()
    schedule.validate()
    if schedule.cache_sha256 != artifacts.cache_artifact_sha256:
        raise M03RV12PackageError("v12 schedule and packaged cache disagree")
    workers = tuple(
        M03RV12PredictiveWorkerPlan(
            setting_index=index,
            setting_id=M03R_V12_SETTING_IDS[index],
            output_root=f"/mnt/output/completion-{index:02d}-setting-{index:02d}",
            cache_path="/mnt/package/cache/top2000-daily-bars.pt",
            initial_parameter_state_path=(
                "/mnt/package/model/common-initial-parameter-state.pt"
            ),
            panel_episode_schedule_sha256=schedule.receipt_sha256,
            initial_parameter_state_file_sha256=(
                artifacts.initial_parameter_state_file_sha256
            ),
            initial_parameter_state_sha256=(artifacts.initial_parameter_state_sha256),
            cache_sha256=artifacts.cache_artifact_sha256,
            risk_source_manifest_path=("/mnt/package/risk/risk-source-manifest.json"),
            risk_source_manifest_file_sha256=(
                artifacts.risk_source_manifest_file_sha256
            ),
            projector_manifest_path="/mnt/package/risk/projector-manifest.json",
            projector_manifest_file_sha256=(artifacts.projector_manifest_file_sha256),
            projector_manifest_sha256=artifacts.projector_manifest_sha256,
            projector_binding_sha256=artifacts.projector_binding_sha256,
            source_manifest_sha256=artifacts.source_manifest_sha256,
            source_archive_sha256=artifacts.source_archive_sha256,
            selected_horizon_sessions=3,
            structural_preflight_path=(
                "/mnt/package/plans/real-data-structural-preflight.json"
            ),
            structural_preflight_file_sha256=(
                artifacts.structural_preflight_file_sha256
            ),
            structural_preflight_receipt_sha256=(
                artifacts.structural_preflight_receipt_sha256
            ),
        )
        for index in range(3)
    )
    provisional = M03RV12PackagePlan(
        artifacts=artifacts,
        schedule=schedule,
        panel=M03RV12PredictivePanelPlan(workers),
        plan_directory="/mnt/package/plans",
        package_plan_sha256="0" * 64,
    )
    result = replace(
        provisional,
        package_plan_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


def write_m03r_v12_package_plan(path: str | Path, plan: M03RV12PackagePlan) -> str:
    plan.validate()
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV12PackageError("v12 package plan target already exists")
    payload = {
        "schema": M03R_V12_PACKAGE_FILE_SCHEMA,
        "plan": asdict(plan),
        "package_plan_sha256": plan.package_plan_sha256,
    }
    encoded = _canonical(payload) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def load_m03r_v12_package_plan(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> M03RV12PackagePlan:
    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV12PackageError(
            "v12 package plan must be a readable regular non-symlink"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_PACKAGE_PLAN_BYTES
        ):
            raise M03RV12PackageError("v12 package plan size or type is invalid")
        content = bytearray()
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            content.extend(block)
        after = os.fstat(descriptor)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV12PackageError("v12 package plan file hash drifted")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV12PackageError("v12 package plan changed while read")
    finally:
        os.close(descriptor)
    payload = json.loads(bytes(content))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != M03R_V12_PACKAGE_FILE_SCHEMA
        or not isinstance(payload.get("plan"), dict)
    ):
        raise M03RV12PackageError("v12 package plan file schema drifted")
    row = payload["plan"]
    artifacts = M03RV12PackageArtifacts(**row["artifacts"])
    schedule = M03RV12PanelEpisodeSchedule(
        **{
            **row["schedule"],
            "fold_geometry_sha256": tuple(row["schedule"]["fold_geometry_sha256"]),
        }
    )
    workers = tuple(
        M03RV12PredictiveWorkerPlan(
            **{
                **value,
                "qualification_updates": tuple(value["qualification_updates"]),
            }
        )
        for value in row["panel"]["workers"]
    )
    panel = M03RV12PredictivePanelPlan(
        **{
            **row["panel"],
            "workers": workers,
        }
    )
    plan = M03RV12PackagePlan(
        **{
            **row,
            "artifacts": artifacts,
            "schedule": schedule,
            "panel": panel,
        }
    )
    plan.validate()
    if payload.get("package_plan_sha256") != plan.package_plan_sha256:
        raise M03RV12PackageError("v12 package file and plan disagree")
    return plan


__all__ = [
    "M03R_V12_EXECUTION_AUTHORIZATION_SCHEMA",
    "M03R_V12_PACKAGE_FILE_SCHEMA",
    "M03R_V12_PACKAGE_SCHEMA",
    "M03R_V12_PACKAGE_SOURCE_PYTHONPATH",
    "M03R_V12_RUNTIME_ENTRYPOINT",
    "M03RV12ExecutionAuthorization",
    "M03RV12PackageArtifacts",
    "M03RV12PackageError",
    "M03RV12PackagePlan",
    "build_m03r_v12_package_plan",
    "load_m03r_v12_execution_authorization",
    "load_m03r_v12_package_plan",
    "write_m03r_v12_execution_authorization",
    "write_m03r_v12_package_plan",
]
