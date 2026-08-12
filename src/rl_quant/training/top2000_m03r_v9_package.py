"""Content-bound package plan for the three-setting M03R-v9 panel."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_DESIGN_ID,
    M03R_V9_PROTOCOL_GENERATION,
    M03R_V9_PROTOCOL_SHA256,
    M03R_V9_SETTING_IDS,
    M03RV9HorizonBinding,
)
from rl_quant.training.top2000_m03r_v9_predictive_worker import (
    M03RV9PredictivePanelPlan,
    M03RV9PredictiveWorkerPlan,
)

M03R_V9_PACKAGE_SCHEMA = "rl-quant.top2000-dev.m03r-v9-predictive-package-v1"
M03R_V9_PACKAGE_PLAN_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-predictive-package-plan-file-v1"
)
M03R_V9_PACKAGE_SOURCE_PYTHONPATH = "/mnt/package/source/src"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_RE = re.compile(r"^[^@\s]+@sha256:([0-9a-f]{64})$")


class M03RV9PackageError(ValueError):
    """The v9 source, risk, package, or worker identity drifted."""


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


def _digest(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise M03RV9PackageError(f"{name} must be one lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class M03RV9PackageArtifacts:
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
    image_reference: str
    image_digest_sha256: str

    def validate(self) -> None:
        for name in (
            "source_archive_sha256",
            "source_manifest_sha256",
            "dependency_lock_sha256",
            "cache_artifact_sha256",
            "cache_manifest_sha256",
            "risk_artifact_sha256",
            "risk_source_manifest_file_sha256",
            "projector_manifest_file_sha256",
            "projector_manifest_sha256",
            "projector_binding_sha256",
            "worker_source_sha256",
            "image_digest_sha256",
        ):
            _digest(name, getattr(self, name))
        match = _IMAGE_RE.fullmatch(self.image_reference)
        if match is None or match.group(1) != self.image_digest_sha256:
            raise M03RV9PackageError("image reference must be digest pinned")


@dataclass(frozen=True, slots=True)
class M03RV9PackagePlan:
    artifacts: M03RV9PackageArtifacts
    panel: M03RV9PredictivePanelPlan
    plan_directory: str
    package_plan_sha256: str
    source_pythonpath: str = M03R_V9_PACKAGE_SOURCE_PYTHONPATH
    protocol_generation: str = M03R_V9_PROTOCOL_GENERATION
    design_id: str = M03R_V9_DESIGN_ID
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    outer_evaluation_authorized: bool = False
    economic_panel_authorized: bool = False
    schema: str = M03R_V9_PACKAGE_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifacts": asdict(self.artifacts),
            "panel": asdict(self.panel),
            "plan_directory": self.plan_directory,
            "source_pythonpath": self.source_pythonpath,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "protocol_sha256": self.protocol_sha256,
            "development_only": self.development_only,
            "reportable": self.reportable,
            "promotion_eligible": self.promotion_eligible,
            "outer_evaluation_authorized": self.outer_evaluation_authorized,
            "economic_panel_authorized": self.economic_panel_authorized,
        }

    def validate(self) -> None:
        self.artifacts.validate()
        self.panel.validate()
        for completion, worker in enumerate(self.panel.workers):
            expected_output = (
                f"/mnt/output/completion-{completion:02d}-setting-{completion:02d}"
            )
            if (
                worker.output_root != expected_output
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
            ):
                raise M03RV9PackageError("worker plan does not bind package artifacts")
        directory = PurePosixPath(self.plan_directory)
        if (
            not directory.is_absolute()
            or self.plan_directory != "/mnt/package/plans"
            or self.source_pythonpath != M03R_V9_PACKAGE_SOURCE_PYTHONPATH
            or self.protocol_generation != M03R_V9_PROTOCOL_GENERATION
            or self.design_id != M03R_V9_DESIGN_ID
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.outer_evaluation_authorized
            or self.economic_panel_authorized
            or self.schema != M03R_V9_PACKAGE_SCHEMA
            or self.package_plan_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV9PackageError("v9 package plan identity drifted")


def build_m03r_v9_package_plan(
    *,
    artifacts: M03RV9PackageArtifacts,
) -> M03RV9PackagePlan:
    artifacts.validate()
    workers = tuple(
        M03RV9PredictiveWorkerPlan(
            setting_index=index,
            setting_id=M03R_V9_SETTING_IDS[index],
            cache_path="/mnt/package/cache/top2000-daily-bars.pt",
            cache_sha256=artifacts.cache_artifact_sha256,
            risk_source_manifest_path="/mnt/package/risk/risk-source-manifest.json",
            risk_source_manifest_file_sha256=(
                artifacts.risk_source_manifest_file_sha256
            ),
            projector_manifest_path="/mnt/package/risk/projector-manifest.json",
            projector_manifest_file_sha256=(artifacts.projector_manifest_file_sha256),
            projector_manifest_sha256=artifacts.projector_manifest_sha256,
            projector_binding_sha256=artifacts.projector_binding_sha256,
            source_manifest_sha256=artifacts.source_manifest_sha256,
            source_archive_sha256=artifacts.source_archive_sha256,
            output_root=f"/mnt/output/completion-{index:02d}-setting-{index:02d}",
        )
        for index in range(3)
    )
    panel = M03RV9PredictivePanelPlan(workers=workers)
    provisional = M03RV9PackagePlan(
        artifacts=artifacts,
        panel=panel,
        plan_directory="/mnt/package/plans",
        package_plan_sha256="0" * 64,
    )
    result = M03RV9PackagePlan(
        artifacts=artifacts,
        panel=panel,
        plan_directory="/mnt/package/plans",
        package_plan_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


def package_plan_file_payload(package: M03RV9PackagePlan) -> dict[str, Any]:
    package.validate()
    return {
        "schema": M03R_V9_PACKAGE_PLAN_FILE_SCHEMA,
        "package": {
            **package.unsigned_payload(),
            "package_plan_sha256": package.package_plan_sha256,
        },
    }


def load_m03r_v9_package_plan(
    path: str | Path,
    *,
    expected_package_plan_sha256: str,
) -> M03RV9PackagePlan:
    _digest("expected_package_plan_sha256", expected_package_plan_sha256)
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise M03RV9PackageError("package plan cannot be read") from exc
    if (
        not isinstance(payload, dict)
        or raw != _canonical(payload) + b"\n"
        or payload.get("schema") != M03R_V9_PACKAGE_PLAN_FILE_SCHEMA
        or set(payload) != {"schema", "package"}
        or not isinstance(payload.get("package"), dict)
    ):
        raise M03RV9PackageError("package plan file is not canonical")
    package_payload = payload["package"]
    package_payload["artifacts"] = M03RV9PackageArtifacts(
        **package_payload["artifacts"]
    )
    panel_payload = package_payload["panel"]
    workers: list[M03RV9PredictiveWorkerPlan] = []
    for worker_payload in panel_payload["workers"]:
        worker_payload["horizon_bindings"] = tuple(
            M03RV9HorizonBinding(**row) for row in worker_payload["horizon_bindings"]
        )
        worker_payload["qualification_evaluation_updates"] = tuple(
            worker_payload["qualification_evaluation_updates"]
        )
        workers.append(M03RV9PredictiveWorkerPlan(**worker_payload))
    panel_payload["workers"] = tuple(workers)
    package_payload["panel"] = M03RV9PredictivePanelPlan(**panel_payload)
    package = M03RV9PackagePlan(**package_payload)
    package.validate()
    if package.package_plan_sha256 != expected_package_plan_sha256:
        raise M03RV9PackageError("package plan hash mismatch")
    return package


__all__ = [
    "M03R_V9_PACKAGE_PLAN_FILE_SCHEMA",
    "M03R_V9_PACKAGE_SCHEMA",
    "M03R_V9_PACKAGE_SOURCE_PYTHONPATH",
    "M03RV9PackageArtifacts",
    "M03RV9PackageError",
    "M03RV9PackagePlan",
    "build_m03r_v9_package_plan",
    "load_m03r_v9_package_plan",
    "package_plan_file_payload",
]
