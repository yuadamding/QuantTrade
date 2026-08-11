"""Content-bound package plan for M03R-v8 predictive pretraining."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_TOP2000_DEV_DESIGN_ID,
    M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_V8_TOP2000_DEV_SETTINGS,
)
from rl_quant.training.top2000_m03r_v8_plan import (
    M03RV8DevelopmentTrainingPlan,
)
from rl_quant.training.top2000_m03r_v8_pretraining_contract import (
    M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION,
    M03R_V8_PRETRAINING_WORKER_SCHEMA,
)

M03R_V8_PRETRAINING_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-pretraining-package-v1"
)
M03R_V8_PRETRAINING_PLAN_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-pretraining-package-plan-file-v1"
)
M03R_V8_PRETRAINING_SOURCE_PYTHONPATH = "/mnt/package/source/src"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_RE = re.compile(r"^[^@\s]+@sha256:([0-9a-f]{64})$")


class M03RV8PretrainingPackageError(ValueError):
    """The v8 source/package/plan identity is inconsistent."""


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
        raise M03RV8PretrainingPackageError(
            f"{name} must be one lowercase SHA-256"
        )


@dataclass(frozen=True, slots=True)
class M03RV8PretrainingArtifactBindings:
    """Independent source, dependency, cache, worker, and image identities."""

    source_archive_sha256: str
    source_manifest_sha256: str
    dependency_lock_sha256: str
    cache_artifact_sha256: str
    cache_manifest_sha256: str
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
            "worker_source_sha256",
            "image_digest_sha256",
        ):
            _digest(name, getattr(self, name))
        match = _IMAGE_RE.fullmatch(self.image_reference)
        if match is None or match.group(1) != self.image_digest_sha256:
            raise M03RV8PretrainingPackageError(
                "image reference must be digest pinned"
            )


@dataclass(frozen=True, slots=True)
class M03RV8PretrainingPackagePlan:
    """Seven pretrained rows and one shared exact execution surface."""

    artifacts: M03RV8PretrainingArtifactBindings
    plans: tuple[M03RV8DevelopmentTrainingPlan, ...]
    plan_directory: str
    package_plan_sha256: str
    source_pythonpath: str = M03R_V8_PRETRAINING_SOURCE_PYTHONPATH
    worker_schema: str = M03R_V8_PRETRAINING_WORKER_SCHEMA
    protocol_generation: str = M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION
    design_id: str = M03R_V8_TOP2000_DEV_DESIGN_ID
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    outer_evaluation_authorized: bool = False
    schema: str = M03R_V8_PRETRAINING_PACKAGE_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifacts": asdict(self.artifacts),
            "plans": [asdict(plan) for plan in self.plans],
            "plan_directory": self.plan_directory,
            "source_pythonpath": self.source_pythonpath,
            "worker_schema": self.worker_schema,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "protocol_sha256": self.protocol_sha256,
            "development_only": self.development_only,
            "reportable": self.reportable,
            "promotion_eligible": self.promotion_eligible,
            "outer_evaluation_authorized": self.outer_evaluation_authorized,
        }

    def validate(self) -> None:
        self.artifacts.validate()
        expected_indices = M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION
        if (
            tuple(plan.setting_index for plan in self.plans) != expected_indices
            or len({plan.output_root for plan in self.plans}) != len(self.plans)
        ):
            raise M03RV8PretrainingPackageError(
                "package must contain the exact seven disjoint pretrained rows"
            )
        for completion_index, plan in enumerate(self.plans):
            plan.validate()
            expected_output = (
                f"/mnt/output/completion-{completion_index:02d}-"
                f"setting-{plan.setting_index:02d}"
            )
            if (
                plan.output_root != expected_output
                or plan.cache_sha256 != self.artifacts.cache_artifact_sha256
                or plan.source_manifest_sha256
                != self.artifacts.source_manifest_sha256
            ):
                raise M03RV8PretrainingPackageError(
                    "worker plan does not bind package cache/source/output identity"
                )
        directory = PurePosixPath(self.plan_directory)
        if (
            not directory.is_absolute()
            or self.plan_directory != "/mnt/package/plans"
            or self.source_pythonpath != M03R_V8_PRETRAINING_SOURCE_PYTHONPATH
            or self.worker_schema != M03R_V8_PRETRAINING_WORKER_SCHEMA
            or self.protocol_generation != M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION
            or self.design_id != M03R_V8_TOP2000_DEV_DESIGN_ID
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.outer_evaluation_authorized
            or self.schema != M03R_V8_PRETRAINING_PACKAGE_SCHEMA
            or self.package_plan_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV8PretrainingPackageError("v8 package plan identity drifted")


def build_m03r_v8_pretraining_package_plan(
    *,
    artifacts: M03RV8PretrainingArtifactBindings,
) -> M03RV8PretrainingPackagePlan:
    """Build the deterministic seven-plan package without touching disk."""

    plans = tuple(
        M03RV8DevelopmentTrainingPlan(
            setting_index=setting_index,
            setting_id=M03R_V8_TOP2000_DEV_SETTINGS[setting_index].setting_id,
            cache_path="/mnt/package/cache/top2000-daily-bars.pt",
            cache_sha256=artifacts.cache_artifact_sha256,
            output_root=(
                f"/mnt/output/completion-{completion_index:02d}-"
                f"setting-{setting_index:02d}"
            ),
            source_manifest_sha256=artifacts.source_manifest_sha256,
        )
        for completion_index, setting_index in enumerate(
            M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION
        )
    )
    unsigned = M03RV8PretrainingPackagePlan(
        artifacts=artifacts,
        plans=plans,
        plan_directory="/mnt/package/plans",
        package_plan_sha256="0" * 64,
    )
    package = M03RV8PretrainingPackagePlan(
        artifacts=artifacts,
        plans=plans,
        plan_directory="/mnt/package/plans",
        package_plan_sha256=_sha256(unsigned.unsigned_payload()),
    )
    package.validate()
    return package


def package_plan_file_payload(
    package: M03RV8PretrainingPackagePlan,
) -> dict[str, Any]:
    package.validate()
    return {
        "schema": M03R_V8_PRETRAINING_PLAN_FILE_SCHEMA,
        "package": {
            **package.unsigned_payload(),
            "package_plan_sha256": package.package_plan_sha256,
        },
    }


def load_m03r_v8_pretraining_package_plan(
    path: str | Path,
    *,
    expected_package_plan_sha256: str,
) -> M03RV8PretrainingPackagePlan:
    """Load and revalidate one canonical package-plan file."""

    _digest("expected_package_plan_sha256", expected_package_plan_sha256)
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise M03RV8PretrainingPackageError("package plan cannot be read") from exc
    if (
        not isinstance(payload, dict)
        or raw != _canonical(payload) + b"\n"
        or payload.get("schema") != M03R_V8_PRETRAINING_PLAN_FILE_SCHEMA
        or set(payload) != {"schema", "package"}
        or not isinstance(payload.get("package"), dict)
    ):
        raise M03RV8PretrainingPackageError("package plan file is not canonical")
    payload = payload["package"]
    payload["artifacts"] = M03RV8PretrainingArtifactBindings(**payload["artifacts"])
    payload["plans"] = tuple(
        M03RV8DevelopmentTrainingPlan(**row) for row in payload["plans"]
    )
    package = M03RV8PretrainingPackagePlan(**payload)
    package.validate()
    if package.package_plan_sha256 != expected_package_plan_sha256:
        raise M03RV8PretrainingPackageError("package plan hash mismatch")
    return package


__all__ = [
    "M03R_V8_PRETRAINING_PACKAGE_SCHEMA",
    "M03R_V8_PRETRAINING_PLAN_FILE_SCHEMA",
    "M03R_V8_PRETRAINING_SOURCE_PYTHONPATH",
    "M03RV8PretrainingArtifactBindings",
    "M03RV8PretrainingPackageError",
    "M03RV8PretrainingPackagePlan",
    "build_m03r_v8_pretraining_package_plan",
    "load_m03r_v8_pretraining_package_plan",
    "package_plan_file_payload",
]
