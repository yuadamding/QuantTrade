"""Content-bound local package plan for the M03R-v14 predictive panel."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_DESIGN_ID,
    M03R_V14_PROTOCOL_GENERATION,
    M03R_V14_PROTOCOL_SHA256,
    M03R_V14_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v14_fold import M03RV14PanelEpisodeSchedule
from rl_quant.training.top2000_m03r_v14_predictive_worker import (
    M03RV14PredictivePanelPlan,
    M03RV14PredictiveWorkerPlan,
)

M03R_V14_PACKAGE_SCHEMA = "rl-quant.top2000-dev.m03r-v14-predictive-package-v1"
M03R_V14_PACKAGE_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-predictive-package-file-v1"
)
M03R_V14_EXECUTION_AUTHORIZATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-predictive-execution-authorization-v1"
)
M03R_V14_PACKAGE_SOURCE_PYTHONPATH = "/mnt/package/source/src"
M03R_V14_RUNTIME_ENTRYPOINT = "rl_quant.workflows.top2000_m03r_v14_predictive"
_MAX_PACKAGE_PLAN_BYTES = 4 * 1024**2
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^[^@\s]+@sha256:([0-9a-f]{64})$")


class M03RV14PackageError(ValueError):
    """The v14 source, data, risk, image, or panel package drifted."""


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
        raise M03RV14PackageError(f"{name} must be one lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV14PackageArtifacts:
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
    initial_parameter_architecture_sha256: str
    structural_preflight_file_sha256: str
    structural_preflight_receipt_sha256: str
    image_reference: str
    image_digest_sha256: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name != "image_reference":
                _digest(name, value)
        match = _IMAGE.fullmatch(self.image_reference)
        if match is None or match.group(1) != self.image_digest_sha256:
            raise M03RV14PackageError("v14 image reference must be digest pinned")


@dataclass(frozen=True, slots=True)
class M03RV14PackagePlan:
    artifacts: M03RV14PackageArtifacts
    schedule: M03RV14PanelEpisodeSchedule
    panel: M03RV14PredictivePanelPlan
    plan_directory: str
    package_plan_sha256: str
    source_pythonpath: str = M03R_V14_PACKAGE_SOURCE_PYTHONPATH
    protocol_generation: str = M03R_V14_PROTOCOL_GENERATION
    design_id: str = M03R_V14_DESIGN_ID
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    structural_gates_complete: bool = True
    package_authorized: bool = False
    kubernetes_launch_authorized: bool = False
    outer_2026_access_authorized: bool = False
    economic_panel_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V14_PACKAGE_SCHEMA

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
            "projector_manifest_path": (
                "/mnt/package/risk/projector-manifest.json"
            ),
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
                or worker.panel_episode_schedule_sha256
                != self.schedule.receipt_sha256
                or worker.initial_parameter_state_file_sha256
                != self.artifacts.initial_parameter_state_file_sha256
                or worker.initial_parameter_state_sha256
                != self.artifacts.initial_parameter_state_sha256
                or worker.initial_parameter_architecture_sha256
                != self.artifacts.initial_parameter_architecture_sha256
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
                or worker.source_archive_sha256
                != self.artifacts.source_archive_sha256
                or worker.structural_preflight_file_sha256
                != self.artifacts.structural_preflight_file_sha256
                or worker.structural_preflight_receipt_sha256
                != self.artifacts.structural_preflight_receipt_sha256
            ):
                raise M03RV14PackageError(
                    "v14 worker does not bind exact package artifacts"
                )
        if (
            self.schedule.cache_sha256 != self.artifacts.cache_artifact_sha256
            or PurePosixPath(self.plan_directory) != PurePosixPath("/mnt/package/plans")
            or self.source_pythonpath != M03R_V14_PACKAGE_SOURCE_PYTHONPATH
            or self.protocol_generation != M03R_V14_PROTOCOL_GENERATION
            or self.design_id != M03R_V14_DESIGN_ID
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or not self.structural_gates_complete
            or self.package_authorized
            or self.kubernetes_launch_authorized
            or self.outer_2026_access_authorized
            or self.economic_panel_authorized
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V14_PACKAGE_SCHEMA
            or self.package_plan_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV14PackageError("v14 package plan identity drifted")


@dataclass(frozen=True, slots=True)
class M03RV14ExecutionAuthorization:
    package_plan_sha256: str
    package_plan_file_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    worker_source_sha256: str
    image_reference: str
    runtime_entrypoint: str = M03R_V14_RUNTIME_ENTRYPOINT
    indexed_completions: int = 2
    parallelism: int = 2
    h100s_per_completion: int = 2
    maximum_h100_requests: int = 4
    predictive_training_authorized: bool = True
    economic_training_authorized: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_EXECUTION_AUTHORIZATION_SCHEMA

    def validate(self, package: M03RV14PackagePlan) -> None:
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
            or self.runtime_entrypoint != M03R_V14_RUNTIME_ENTRYPOINT
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
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_EXECUTION_AUTHORIZATION_SCHEMA
        ):
            raise M03RV14PackageError("v14 execution authorization drifted")

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


def build_m03r_v14_package_plan(
    artifacts: M03RV14PackageArtifacts,
    schedule: M03RV14PanelEpisodeSchedule,
) -> M03RV14PackagePlan:
    artifacts.validate()
    schedule.validate()
    workers = tuple(
        M03RV14PredictiveWorkerPlan(
            setting_index=index,
            setting_id=setting_id,
            output_root=f"/mnt/output/completion-{index:02d}-setting-{index:02d}",
            cache_path="/mnt/package/cache/top2000-daily-bars.pt",
            initial_parameter_state_path=(
                "/mnt/package/model/common-initial-parameter-state.pt"
            ),
            panel_episode_schedule_sha256=schedule.receipt_sha256,
            initial_parameter_state_file_sha256=(
                artifacts.initial_parameter_state_file_sha256
            ),
            initial_parameter_state_sha256=artifacts.initial_parameter_state_sha256,
            initial_parameter_architecture_sha256=(
                artifacts.initial_parameter_architecture_sha256
            ),
            cache_sha256=artifacts.cache_artifact_sha256,
            risk_source_manifest_path="/mnt/package/risk/risk-source-manifest.json",
            risk_source_manifest_file_sha256=(
                artifacts.risk_source_manifest_file_sha256
            ),
            projector_manifest_path="/mnt/package/risk/projector-manifest.json",
            projector_manifest_file_sha256=(
                artifacts.projector_manifest_file_sha256
            ),
            projector_manifest_sha256=artifacts.projector_manifest_sha256,
            projector_binding_sha256=artifacts.projector_binding_sha256,
            source_manifest_sha256=artifacts.source_manifest_sha256,
            source_archive_sha256=artifacts.source_archive_sha256,
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
        for index, setting_id in enumerate(M03R_V14_SETTING_IDS)
    )
    panel = M03RV14PredictivePanelPlan(workers)
    provisional = M03RV14PackagePlan(
        artifacts=artifacts,
        schedule=schedule,
        panel=panel,
        plan_directory="/mnt/package/plans",
        package_plan_sha256="0" * 64,
    )
    result = replace(
        provisional,
        package_plan_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


def write_m03r_v14_package_plan(path: str | Path, plan: M03RV14PackagePlan) -> str:
    plan.validate()
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV14PackageError("v14 package-plan target already exists")
    encoded = _canonical(
        {
            "schema": M03R_V14_PACKAGE_FILE_SCHEMA,
            "plan": asdict(plan),
            "package_plan_sha256": plan.package_plan_sha256,
        }
    ) + b"\n"
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def load_m03r_v14_package_plan(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> M03RV14PackagePlan:
    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV14PackageError("v14 package plan is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_PACKAGE_PLAN_BYTES
        ):
            raise M03RV14PackageError("v14 package plan type or size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV14PackageError("v14 package plan changed while read")
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise M03RV14PackageError("v14 package plan file hash drifted")
    try:
        payload = json.loads(raw)
        row = dict(payload["plan"])
        row["artifacts"] = M03RV14PackageArtifacts(**row["artifacts"])
        schedule_row = dict(row["schedule"])
        schedule_row["fold_geometry_sha256"] = tuple(
            schedule_row["fold_geometry_sha256"]
        )
        row["schedule"] = M03RV14PanelEpisodeSchedule(**schedule_row)
        panel_row = dict(row["panel"])
        workers = []
        for worker_row in panel_row["workers"]:
            worker_payload = dict(worker_row)
            worker_payload["fold_optimizer_updates"] = tuple(
                worker_payload["fold_optimizer_updates"]
            )
            workers.append(M03RV14PredictiveWorkerPlan(**worker_payload))
        panel_row["workers"] = tuple(workers)
        row["panel"] = M03RV14PredictivePanelPlan(**panel_row)
        plan = M03RV14PackagePlan(**row)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise M03RV14PackageError("v14 package plan is malformed") from exc
    if (
        payload.get("schema") != M03R_V14_PACKAGE_FILE_SCHEMA
        or payload.get("package_plan_sha256") != plan.package_plan_sha256
    ):
        raise M03RV14PackageError("v14 package plan receipt drifted")
    plan.validate()
    return plan


def write_m03r_v14_execution_authorization(
    path: str | Path,
    authorization: M03RV14ExecutionAuthorization,
    package: M03RV14PackagePlan,
) -> str:
    authorization.validate(package)
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV14PackageError("v14 execution authorization already exists")
    encoded = _canonical(
        {
            "authorization": asdict(authorization),
            "receipt_sha256": authorization.receipt_sha256,
        }
    ) + b"\n"
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def load_m03r_v14_execution_authorization(
    path: str | Path,
    *,
    expected_file_sha256: str,
    package: M03RV14PackagePlan,
) -> M03RV14ExecutionAuthorization:
    expected_file_sha256 = _digest(
        "expected_authorization_file_sha256", expected_file_sha256
    )
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV14PackageError(
            "v14 execution authorization is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_PACKAGE_PLAN_BYTES
        ):
            raise M03RV14PackageError("v14 execution authorization is invalid")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV14PackageError(
                "v14 execution authorization changed while read"
            )
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise M03RV14PackageError("v14 execution authorization hash drifted")
    try:
        payload = json.loads(raw)
        authorization = M03RV14ExecutionAuthorization(
            **dict(payload["authorization"])
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise M03RV14PackageError(
            "v14 execution authorization is malformed"
        ) from exc
    authorization.validate(package)
    if payload.get("receipt_sha256") != authorization.receipt_sha256:
        raise M03RV14PackageError("v14 execution authorization receipt drifted")
    return authorization


__all__ = [
    "M03R_V14_PACKAGE_FILE_SCHEMA",
    "M03R_V14_PACKAGE_SCHEMA",
    "M03R_V14_EXECUTION_AUTHORIZATION_SCHEMA",
    "M03R_V14_PACKAGE_SOURCE_PYTHONPATH",
    "M03R_V14_RUNTIME_ENTRYPOINT",
    "M03RV14PackageArtifacts",
    "M03RV14PackageError",
    "M03RV14PackagePlan",
    "M03RV14ExecutionAuthorization",
    "build_m03r_v14_package_plan",
    "load_m03r_v14_package_plan",
    "load_m03r_v14_execution_authorization",
    "write_m03r_v14_execution_authorization",
    "write_m03r_v14_package_plan",
]
