"""Fail-closed Kubernetes schemas for the TOP2000 M03R-v7 seed-17 run.

This module is deliberately disjoint from
``hold30_alpha_m03r_v7_kubernetes``.  The older module continues to require
five paired seeds and thirty fold/seed cells per setting.  This module accepts
only the immutable development-only seed-17 package, renders the dedicated
seed-17 worker, and requires exactly six one-member fold executions per
setting.

Everything here is a pure renderer or receipt validator.  It performs no
Kubernetes operation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, cast

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_FOLDS,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SEEDS,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_GPUS_PER_COMPLETION,
    M03R_TOP2000_H100_POOL_NODE_SELECTOR,
    M03R_TOP2000_INDEXED_COMPLETIONS,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03R_TOP2000_MULTI_GPU_TOLERATION,
    M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS,
    M03R_TOP2000_PRIORITY_CLASS_NAME,
    M03R_TOP2000_TERMINATION_MESSAGE_PATH,
    M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
    M03RV7AdmittedJobBinding,
    M03RV7KubernetesTemplateConfig,
    M03RV7LiveAdmissionEvidence,
    M03RV7RenderedSuspendedJob,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_package import (
    M03RV7Seed17IndexPlan,
    M03RV7Seed17PackagePlan,
)

M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX = (
    "/opt/conda/envs/quanttrade/bin/python",
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--max-restarts=1",
    "--nproc-per-node=2",
    "-m",
    "rl_quant.workflows.top2000_m03r_v7_seed17_dev",
)
M03R_SEED17_TOP2000_EXECUTION_QUALIFICATION_SCHEMA = (
    "rl-quant.m03r-v7-top2000-seed17-execution-qualification-v1"
)
M03R_SEED17_TOP2000_INDEX_RECEIPT_SCHEMA = (
    "rl-quant.m03r-v7-top2000-seed17-index-receipt-v1"
)
M03R_SEED17_TOP2000_BATCH_RECEIPT_SCHEMA = (
    "rl-quant.m03r-v7-top2000-seed17-indexed-batch-receipt-v1"
)
M03R_SEED17_TOP2000_QUALIFICATION_ARTIFACT_SCHEMA = (
    "rl-quant.m03r-v7-top2000-seed17-validation-artifact-ref-v1"
)
M03R_SEED17_TOP2000_CAPACITY_RECEIPT_SCHEMA = (
    "rl-quant.m03r-v7-top2000-seed17-two-h100-capacity-v1"
)
M03R_SEED17_TOP2000_GPU_NAME = "NVIDIA H100 80GB HBM3"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class M03RV7Seed17KubernetesError(ValueError):
    """A seed-17 render or completion-evidence invariant failed."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise M03RV7Seed17KubernetesError(
            "seed-17 Kubernetes payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise M03RV7Seed17KubernetesError(
            f"{name} must be one lowercase hexadecimal SHA-256"
        )


def _canonical_pod_spec_sha256(value: dict[str, Any]) -> str:
    """Match the admitted-binding ASCII JSON plus one-newline contract."""

    try:
        encoded = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise M03RV7Seed17KubernetesError(
            "seed-17 Pod spec is not canonical finite ASCII JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _execution_surface_payload(
    *,
    plan: M03RV7Seed17PackagePlan,
    worker_entrypoint_sha256: str,
    runtime_manifest_sha256: str,
    validation_sentinel_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "rl-quant.m03r-v7-top2000-seed17-execution-surface-v1",
        "protocol_sha256": plan.protocol_sha256,
        "package_plan_sha256": plan.package_plan_sha256,
        "benchmark_preflight_sha256": plan.benchmark_preflight_sha256,
        "source_archive_sha256": plan.artifacts.source_archive_sha256,
        "source_manifest_sha256": plan.artifacts.source_manifest_sha256,
        "dependency_lock_sha256": plan.artifacts.dependency_lock_sha256,
        "cache_artifact_sha256": plan.artifacts.cache_artifact_sha256,
        "cache_manifest_sha256": plan.artifacts.cache_manifest_sha256,
        "data_manifest_sha256": plan.artifacts.data_manifest_sha256,
        "execution_model_sha256": plan.artifacts.execution_model_sha256,
        "image_digest_sha256": plan.artifacts.image_digest_sha256,
        "runtime_profile": asdict(plan.runtime_profile),
        "worker_argv_prefix": list(M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX),
        "worker_entrypoint_sha256": worker_entrypoint_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "validation_sentinel_receipt_sha256": (
            validation_sentinel_receipt_sha256
        ),
        "gpu_count_per_completion": 2,
        "torchrun_nproc_per_node": 2,
        "complete_cross_section_per_rank": True,
        "stock_axis_partitioning": False,
        "fold_indices": list(M03R_SEED17_TOP2000_FOLDS),
        "paired_seeds": list(M03R_SEED17_TOP2000_SEEDS),
        "one_member_fold_execution": True,
        "five_seed_ensemble_eligible": False,
    }


@dataclass(frozen=True, slots=True)
class M03RV7Seed17QualificationArtifactRef:
    """One real two-rank validation plus one-member fold execution artifact."""

    completion_index: int
    setting_index: int
    setting_id: str
    runtime_setting_id: str
    package_plan_sha256: str
    source_archive_sha256: str
    runtime_manifest_sha256: str
    qualification_receipt_sha256: str
    validation_receipt_sha256: str
    fold_execution_receipt_sha256: str
    rank_gpu_names: tuple[str, ...]
    world_size: int
    receipt_sha256: str
    schema: str = M03R_SEED17_TOP2000_QUALIFICATION_ARTIFACT_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "completion_index": self.completion_index,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "runtime_setting_id": self.runtime_setting_id,
            "package_plan_sha256": self.package_plan_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "qualification_receipt_sha256": self.qualification_receipt_sha256,
            "validation_receipt_sha256": self.validation_receipt_sha256,
            "fold_execution_receipt_sha256": (
                self.fold_execution_receipt_sha256
            ),
            "rank_gpu_names": list(self.rank_gpu_names),
            "world_size": self.world_size,
        }

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_SEED17_TOP2000_QUALIFICATION_ARTIFACT_SCHEMA
            or not 0 <= self.completion_index < 12
            or self.rank_gpu_names != (M03R_SEED17_TOP2000_GPU_NAME,) * 2
            or self.world_size != 2
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 qualification artifact must prove two H100 80GB ranks"
            )
        for name in (
            "package_plan_sha256",
            "source_archive_sha256",
            "runtime_manifest_sha256",
            "qualification_receipt_sha256",
            "validation_receipt_sha256",
            "fold_execution_receipt_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Seed17KubernetesError(
                "seed-17 qualification artifact hash mismatch"
            )

    def verify_for(
        self,
        plan: M03RV7Seed17PackagePlan,
        *,
        expected_completion_index: int,
        runtime_manifest_sha256: str,
    ) -> None:
        row = plan.indices[expected_completion_index]
        if (
            self.completion_index != expected_completion_index
            or self.setting_index != row.setting_index
            or self.setting_id != row.setting_id
            or self.runtime_setting_id != row.runtime_setting_id
            or self.package_plan_sha256 != plan.package_plan_sha256
            or self.source_archive_sha256
            != plan.artifacts.source_archive_sha256
            or self.runtime_manifest_sha256 != runtime_manifest_sha256
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 qualification artifact identity drifted from the package"
            )


def build_m03r_v7_seed17_qualification_artifact_ref(
    *,
    plan: M03RV7Seed17PackagePlan,
    completion_index: int,
    runtime_manifest_sha256: str,
    qualification_receipt_sha256: str,
    validation_receipt_sha256: str,
    fold_execution_receipt_sha256: str,
) -> M03RV7Seed17QualificationArtifactRef:
    """Build a typed reference after independently verifying one real artifact."""

    if not 0 <= completion_index < 12:
        raise M03RV7Seed17KubernetesError(
            "qualification completion index must be in [0, 11]"
        )
    row = plan.indices[completion_index]
    fields: dict[str, Any] = {
        "completion_index": completion_index,
        "setting_index": row.setting_index,
        "setting_id": row.setting_id,
        "runtime_setting_id": row.runtime_setting_id,
        "package_plan_sha256": plan.package_plan_sha256,
        "source_archive_sha256": plan.artifacts.source_archive_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "qualification_receipt_sha256": qualification_receipt_sha256,
        "validation_receipt_sha256": validation_receipt_sha256,
        "fold_execution_receipt_sha256": fold_execution_receipt_sha256,
        "rank_gpu_names": (M03R_SEED17_TOP2000_GPU_NAME,) * 2,
        "world_size": 2,
        "schema": M03R_SEED17_TOP2000_QUALIFICATION_ARTIFACT_SCHEMA,
    }
    unsigned = M03RV7Seed17QualificationArtifactRef.__new__(
        M03RV7Seed17QualificationArtifactRef
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    artifact = M03RV7Seed17QualificationArtifactRef(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )
    artifact.verify_for(
        plan,
        expected_completion_index=completion_index,
        runtime_manifest_sha256=runtime_manifest_sha256,
    )
    return artifact


@dataclass(frozen=True, slots=True)
class M03RV7Seed17CapacityReceipt:
    """Typed sentinel plus all-setting two-H100 qualification coverage."""

    package_plan_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    runtime_manifest_sha256: str
    worker_entrypoint_sha256: str
    sentinel_completion_index: int
    sentinel: M03RV7Seed17QualificationArtifactRef
    all_setting_qualifications: tuple[
        M03RV7Seed17QualificationArtifactRef, ...
    ]
    execution_surface_sha256: str
    two_h100_capacity_qualified: bool
    all_twelve_settings_qualified: bool
    receipt_sha256: str
    protocol_sha256: str = M03R_SEED17_TOP2000_PROTOCOL_SHA256
    schema: str = M03R_SEED17_TOP2000_CAPACITY_RECEIPT_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "package_plan_sha256": self.package_plan_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "worker_entrypoint_sha256": self.worker_entrypoint_sha256,
            "sentinel_completion_index": self.sentinel_completion_index,
            "sentinel": asdict(self.sentinel),
            "all_setting_qualifications": [
                asdict(row) for row in self.all_setting_qualifications
            ],
            "execution_surface_sha256": self.execution_surface_sha256,
            "two_h100_capacity_qualified": self.two_h100_capacity_qualified,
            "all_twelve_settings_qualified": self.all_twelve_settings_qualified,
        }

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_SEED17_TOP2000_CAPACITY_RECEIPT_SCHEMA
            or self.protocol_sha256 != M03R_SEED17_TOP2000_PROTOCOL_SHA256
            or self.sentinel_completion_index != 3
            or self.sentinel.completion_index != 3
            or len(self.all_setting_qualifications) != 12
            or tuple(
                row.completion_index for row in self.all_setting_qualifications
            )
            != tuple(range(12))
            or not self.two_h100_capacity_qualified
            or not self.all_twelve_settings_qualified
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 capacity requires index-3 sentinel and all 12 settings"
            )
        for name in (
            "package_plan_sha256",
            "source_archive_sha256",
            "source_manifest_sha256",
            "runtime_manifest_sha256",
            "worker_entrypoint_sha256",
            "execution_surface_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Seed17KubernetesError(
                "seed-17 capacity receipt hash mismatch"
            )

    def verify_for(self, plan: M03RV7Seed17PackagePlan) -> None:
        self.sentinel.verify_for(
            plan,
            expected_completion_index=3,
            runtime_manifest_sha256=self.runtime_manifest_sha256,
        )
        for index, row in enumerate(self.all_setting_qualifications):
            row.verify_for(
                plan,
                expected_completion_index=index,
                runtime_manifest_sha256=self.runtime_manifest_sha256,
            )
        expected_surface = _sha256(
            _execution_surface_payload(
                plan=plan,
                worker_entrypoint_sha256=self.worker_entrypoint_sha256,
                runtime_manifest_sha256=self.runtime_manifest_sha256,
                validation_sentinel_receipt_sha256=self.sentinel.receipt_sha256,
            )
        )
        if (
            self.package_plan_sha256 != plan.package_plan_sha256
            or self.source_archive_sha256
            != plan.artifacts.source_archive_sha256
            or self.source_manifest_sha256
            != plan.artifacts.source_manifest_sha256
            or self.execution_surface_sha256 != expected_surface
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 capacity receipt does not bind the exact package surface"
            )


def build_m03r_v7_seed17_capacity_receipt(
    *,
    plan: M03RV7Seed17PackagePlan,
    worker_entrypoint_sha256: str,
    runtime_manifest_sha256: str,
    sentinel: M03RV7Seed17QualificationArtifactRef,
    all_setting_qualifications: tuple[
        M03RV7Seed17QualificationArtifactRef, ...
    ],
) -> M03RV7Seed17CapacityReceipt:
    """Bind the real sentinel and ordered all-setting qualification artifacts."""

    surface = _sha256(
        _execution_surface_payload(
            plan=plan,
            worker_entrypoint_sha256=worker_entrypoint_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
            validation_sentinel_receipt_sha256=sentinel.receipt_sha256,
        )
    )
    fields: dict[str, Any] = {
        "package_plan_sha256": plan.package_plan_sha256,
        "source_archive_sha256": plan.artifacts.source_archive_sha256,
        "source_manifest_sha256": plan.artifacts.source_manifest_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "worker_entrypoint_sha256": worker_entrypoint_sha256,
        "sentinel_completion_index": 3,
        "sentinel": sentinel,
        "all_setting_qualifications": all_setting_qualifications,
        "execution_surface_sha256": surface,
        "two_h100_capacity_qualified": True,
        "all_twelve_settings_qualified": True,
        "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
        "schema": M03R_SEED17_TOP2000_CAPACITY_RECEIPT_SCHEMA,
    }
    unsigned = M03RV7Seed17CapacityReceipt.__new__(
        M03RV7Seed17CapacityReceipt
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    receipt = M03RV7Seed17CapacityReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )
    receipt.verify_for(plan)
    return receipt


@dataclass(frozen=True, slots=True)
class M03RV7Seed17ExecutionQualification:
    """Bind sentinel, two-rank capacity, and final seed-17 worker identity."""

    package_plan_sha256: str
    worker_entrypoint_sha256: str
    runtime_manifest_sha256: str
    validation_sentinel_receipt_sha256: str
    execution_surface_sha256: str
    capacity_execution_surface_sha256: str
    capacity_receipt_sha256: str
    capacity_receipt: M03RV7Seed17CapacityReceipt
    two_h100_capacity_qualified: bool
    validation_boundary_qualified: bool
    receipt_sha256: str
    protocol_sha256: str = M03R_SEED17_TOP2000_PROTOCOL_SHA256
    worker_argv_prefix: tuple[str, ...] = (
        M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX
    )
    schema: str = M03R_SEED17_TOP2000_EXECUTION_QUALIFICATION_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "package_plan_sha256": self.package_plan_sha256,
            "worker_argv_prefix": list(self.worker_argv_prefix),
            "worker_entrypoint_sha256": self.worker_entrypoint_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "validation_sentinel_receipt_sha256": (
                self.validation_sentinel_receipt_sha256
            ),
            "execution_surface_sha256": self.execution_surface_sha256,
            "capacity_execution_surface_sha256": (
                self.capacity_execution_surface_sha256
            ),
            "capacity_receipt_sha256": self.capacity_receipt_sha256,
            "capacity_receipt": asdict(self.capacity_receipt),
            "two_h100_capacity_qualified": self.two_h100_capacity_qualified,
            "validation_boundary_qualified": (
                self.validation_boundary_qualified
            ),
        }

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_SEED17_TOP2000_EXECUTION_QUALIFICATION_SCHEMA
            or self.protocol_sha256 != M03R_SEED17_TOP2000_PROTOCOL_SHA256
            or self.worker_argv_prefix
            != M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX
            or not self.two_h100_capacity_qualified
            or not self.validation_boundary_qualified
            or self.execution_surface_sha256
            != self.capacity_execution_surface_sha256
            or self.capacity_receipt_sha256
            != self.capacity_receipt.receipt_sha256
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 execution qualification or two-H100 surface drifted"
            )
        for name in (
            "package_plan_sha256",
            "worker_entrypoint_sha256",
            "runtime_manifest_sha256",
            "validation_sentinel_receipt_sha256",
            "execution_surface_sha256",
            "capacity_execution_surface_sha256",
            "capacity_receipt_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Seed17KubernetesError(
                "seed-17 execution qualification hash mismatch"
            )

    def verify_for(self, plan: M03RV7Seed17PackagePlan) -> None:
        self.capacity_receipt.verify_for(plan)
        expected_surface = _sha256(
            _execution_surface_payload(
                plan=plan,
                worker_entrypoint_sha256=self.worker_entrypoint_sha256,
                runtime_manifest_sha256=self.runtime_manifest_sha256,
                validation_sentinel_receipt_sha256=(
                    self.validation_sentinel_receipt_sha256
                ),
            )
        )
        if (
            self.package_plan_sha256 != plan.package_plan_sha256
            or self.protocol_sha256 != plan.protocol_sha256
            or self.execution_surface_sha256 != expected_surface
            or self.execution_surface_sha256
            != self.capacity_receipt.execution_surface_sha256
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 qualification does not bind the supplied package"
            )


def build_m03r_v7_seed17_execution_qualification(
    *,
    plan: M03RV7Seed17PackagePlan,
    capacity_receipt: M03RV7Seed17CapacityReceipt,
) -> M03RV7Seed17ExecutionQualification:
    """Bind externally verified sentinel and capacity receipts to the plan."""

    capacity_receipt.verify_for(plan)
    surface = capacity_receipt.execution_surface_sha256
    fields: dict[str, Any] = {
        "package_plan_sha256": plan.package_plan_sha256,
        "worker_entrypoint_sha256": capacity_receipt.worker_entrypoint_sha256,
        "runtime_manifest_sha256": capacity_receipt.runtime_manifest_sha256,
        "validation_sentinel_receipt_sha256": (
            capacity_receipt.sentinel.receipt_sha256
        ),
        "execution_surface_sha256": surface,
        "capacity_execution_surface_sha256": surface,
        "capacity_receipt_sha256": capacity_receipt.receipt_sha256,
        "capacity_receipt": capacity_receipt,
        "two_h100_capacity_qualified": True,
        "validation_boundary_qualified": True,
        "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
        "worker_argv_prefix": M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX,
        "schema": M03R_SEED17_TOP2000_EXECUTION_QUALIFICATION_SCHEMA,
    }
    unsigned = M03RV7Seed17ExecutionQualification.__new__(
        M03RV7Seed17ExecutionQualification
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    receipt = M03RV7Seed17ExecutionQualification(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )
    receipt.verify_for(plan)
    return receipt


@dataclass(frozen=True, slots=True)
class M03RV7Seed17QualifiedPackage:
    """A seed-17 package paired with its exact sentinel/capacity evidence."""

    plan: M03RV7Seed17PackagePlan
    qualification: M03RV7Seed17ExecutionQualification

    def __post_init__(self) -> None:
        self.qualification.verify_for(self.plan)

    def require_launch_ready(self) -> None:
        self.qualification.verify_for(self.plan)


@dataclass(frozen=True, slots=True)
class M03RV7Seed17RenderedQualificationJob:
    """Suspended seed-17 validation sentinel or all-setting qualification."""

    manifest_json: str
    manifest_sha256: str
    pod_template_sha256: str
    package_plan_sha256: str
    live_evidence_receipt_sha256: str
    completions: int
    parallelism: int
    completion_index: int | None
    activation_authorized: bool = False

    def __post_init__(self) -> None:
        try:
            manifest = json.loads(self.manifest_json)
        except json.JSONDecodeError as exc:
            raise M03RV7Seed17KubernetesError(
                "seed-17 qualification manifest JSON is invalid"
            ) from exc
        if not isinstance(manifest, dict) or self.manifest_json != _canonical_json(
            manifest
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 qualification manifest is not canonical"
            )
        spec = cast(dict[str, Any], manifest.get("spec"))
        template = cast(dict[str, Any], spec.get("template"))
        pod_spec = cast(dict[str, Any], template.get("spec"))
        containers = cast(list[dict[str, Any]], pod_spec.get("containers"))
        container = containers[0] if len(containers) == 1 else {}
        args = container.get("args")
        resources = cast(dict[str, Any], container.get("resources"))
        requests = cast(dict[str, Any], resources.get("requests"))
        limits = cast(dict[str, Any], resources.get("limits"))
        expected_completion = 1 if self.completion_index is not None else 12
        expected_parallelism = 1 if self.completion_index is not None else self.parallelism
        if (
            manifest.get("apiVersion") != "batch/v1"
            or manifest.get("kind") != "Job"
            or spec.get("suspend") is not True
            or spec.get("completionMode") != "Indexed"
            or spec.get("completions") != expected_completion
            or spec.get("completions") != self.completions
            or spec.get("parallelism") != expected_parallelism
            or spec.get("backoffLimit") != 0
            or spec.get("activeDeadlineSeconds")
            != M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS
            or self.activation_authorized
            or not isinstance(args, list)
            or args[: len(M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX) - 1]
            != list(M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX[1:])
            or "--validation-sentinel" not in args
            or requests.get("nvidia.com/gpu") != "2"
            or limits.get("nvidia.com/gpu") != "2"
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 qualification must stay suspended, bounded, and two-H100"
            )
        if self.completion_index is not None and (
            not 0 <= self.completion_index < 12
            or args[-3:]
            != [
                "--completion-index",
                str(self.completion_index),
                "--validation-sentinel",
            ]
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 sentinel completion index drifted"
            )
        for name in (
            "manifest_sha256",
            "pod_template_sha256",
            "package_plan_sha256",
            "live_evidence_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            self.manifest_sha256 != _sha256(manifest)
            or self.pod_template_sha256
            != _canonical_pod_spec_sha256(pod_spec)
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 qualification manifest hash mismatch"
            )

    @property
    def manifest(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.manifest_json))


def _render_m03r_v7_seed17_qualification_job(
    *,
    plan: M03RV7Seed17PackagePlan,
    live_evidence: M03RV7LiveAdmissionEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
    completion_index: int | None,
) -> M03RV7Seed17RenderedQualificationJob:
    live_evidence.require_fresh(
        now_utc=now_utc,
        require_runtime_selector_proof=False,
    )
    if not plan.plan_artifact_path.startswith(
        template.package_mount_path.rstrip("/") + "/"
    ):
        raise M03RV7Seed17KubernetesError(
            "seed-17 package plan must be below the read-only package mount"
        )
    if completion_index is not None and not 0 <= completion_index < 12:
        raise M03RV7Seed17KubernetesError(
            "seed-17 sentinel completion index must be in [0, 11]"
        )
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v7-seed17-qualification",
        "app.kubernetes.io/managed-by": "receipt-gated-research",
        "k8s-user": "yding4",
        "rl-quant/run-id": template.run_id,
        "rl-quant/owner": "yding4",
        "runai/queue": template.runai_queue,
    }
    annotations = {
        "rl-quant/run-id": template.run_id,
        "rl-quant/package-plan-sha256": plan.package_plan_sha256,
        "rl-quant/protocol-sha256": plan.protocol_sha256,
        "rl-quant/source-archive-sha256": plan.artifacts.source_archive_sha256,
        "rl-quant/cache-artifact-sha256": plan.artifacts.cache_artifact_sha256,
        "rl-quant/image-digest-sha256": plan.artifacts.image_digest_sha256,
        "rl-quant/live-evidence-receipt-sha256": live_evidence.receipt_sha256,
        "rl-quant/data-role": "development-only-nonreportable",
        "rl-quant/validation-boundary": "seed-validation-plus-fold-execution",
        "rl-quant/qualification-setting-coverage": (
            str(completion_index) if completion_index is not None else "all-12"
        ),
    }
    argv = list(M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX) + [
        "--package-plan",
        plan.plan_artifact_path,
        "--package-plan-sha256",
        plan.package_plan_sha256,
        "--output-root",
        template.output_mount_path,
    ]
    if completion_index is not None:
        argv.extend(["--completion-index", str(completion_index)])
    argv.append("--validation-sentinel")
    index_environment: dict[str, Any]
    if completion_index is None:
        index_environment = {
            "name": "JOB_COMPLETION_INDEX",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": (
                        "metadata.annotations['batch.kubernetes.io/"
                        "job-completion-index']"
                    ),
                }
            },
        }
    else:
        index_environment = {
            "name": "JOB_COMPLETION_INDEX",
            "value": str(completion_index),
        }
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "serviceAccount": template.service_account_name,
        "serviceAccountName": template.service_account_name,
        "schedulerName": template.scheduler_name,
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "dnsPolicy": "ClusterFirst",
        "nodeSelector": dict(M03R_TOP2000_H100_POOL_NODE_SELECTOR),
        "priorityClassName": M03R_TOP2000_PRIORITY_CLASS_NAME,
        "terminationGracePeriodSeconds": 60,
        "tolerations": [dict(M03R_TOP2000_MULTI_GPU_TOLERATION)],
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": live_evidence.gpu_product_label_key,
                                    "operator": "In",
                                    "values": list(
                                        live_evidence.gpu_product_label_values
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": template.run_as_user,
            "runAsGroup": template.run_as_group,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "trainer",
                "image": plan.artifacts.image_reference,
                "imagePullPolicy": "IfNotPresent",
                "terminationMessagePath": M03R_TOP2000_TERMINATION_MESSAGE_PATH,
                "terminationMessagePolicy": M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
                "command": [argv[0]],
                "args": argv[1:],
                "env": [
                    index_environment,
                    {"name": "NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
                    {"name": "TORCH_NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
                    {"name": "PYTHONNOUSERSITE", "value": "1"},
                    {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                    {"name": "PYTHONHASHSEED", "value": "0"},
                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                    {"name": "CUBLAS_WORKSPACE_CONFIG", "value": ":4096:8"},
                    {"name": "OMP_NUM_THREADS", "value": "8"},
                    {"name": "MKL_NUM_THREADS", "value": "8"},
                    {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
                    {"name": "TORCHINDUCTOR_CACHE_DIR", "value": "/tmp/torchinductor"},
                    {"name": "TRITON_CACHE_DIR", "value": "/tmp/triton"},
                    {"name": "PYTHONPATH", "value": plan.source_pythonpath},
                ],
                "resources": {
                    "requests": {
                        "cpu": template.cpu_request,
                        "memory": template.memory_request,
                        "ephemeral-storage": template.ephemeral_storage_request,
                        "nvidia.com/gpu": "2",
                    },
                    "limits": {
                        "cpu": template.cpu_limit,
                        "memory": template.memory_limit,
                        "ephemeral-storage": template.ephemeral_storage_limit,
                        "nvidia.com/gpu": "2",
                    },
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [
                    {
                        "name": "research-data",
                        "mountPath": template.package_mount_path,
                        "subPath": (
                            template.pvc_training_subpath.rstrip("/")
                            + "/packages/"
                            + template.run_id
                        ),
                        "readOnly": True,
                    },
                    {
                        "name": "research-data",
                        "mountPath": template.output_mount_path,
                        "subPath": (
                            template.pvc_training_subpath.rstrip("/")
                            + "/runs/"
                            + template.run_id
                        ),
                    },
                    {"name": "tmp", "mountPath": "/tmp"},
                    {"name": "dshm", "mountPath": "/dev/shm"},
                ],
            }
        ],
        "volumes": [
            {
                "name": "research-data",
                "persistentVolumeClaim": {"claimName": template.pvc_claim_name},
            },
            {"name": "tmp", "emptyDir": {}},
            {
                "name": "dshm",
                "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
            },
        ],
    }
    completions = 1 if completion_index is not None else 12
    parallelism = 1 if completion_index is not None else live_evidence.allowed_parallelism
    manifest: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": template.job_name,
            "namespace": M03R_TOP2000_KUBERNETES_NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "suspend": True,
            "completionMode": "Indexed",
            "completions": completions,
            "parallelism": parallelism,
            "backoffLimit": 0,
            "activeDeadlineSeconds": M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS,
            "ttlSecondsAfterFinished": template.ttl_seconds_after_finished,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": pod_spec,
            },
        },
    }
    return M03RV7Seed17RenderedQualificationJob(
        manifest_json=_canonical_json(manifest),
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_canonical_pod_spec_sha256(pod_spec),
        package_plan_sha256=plan.package_plan_sha256,
        live_evidence_receipt_sha256=live_evidence.receipt_sha256,
        completions=completions,
        parallelism=parallelism,
        completion_index=completion_index,
    )


def render_m03r_v7_seed17_top2000_suspended_validation_sentinel_job(
    *,
    plan: M03RV7Seed17PackagePlan,
    completion_index: int,
    live_evidence: M03RV7LiveAdmissionEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV7Seed17RenderedQualificationJob:
    """Render one two-H100 sentinel that crosses validation/fold execution."""

    if completion_index != 3:
        raise M03RV7Seed17KubernetesError(
            "the frozen seed-17 validation sentinel is completion index 3"
        )
    return _render_m03r_v7_seed17_qualification_job(
        plan=plan,
        live_evidence=live_evidence,
        template=template,
        now_utc=now_utc,
        completion_index=completion_index,
    )


def render_m03r_v7_seed17_top2000_suspended_qualification_batch_job(
    *,
    plan: M03RV7Seed17PackagePlan,
    live_evidence: M03RV7LiveAdmissionEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV7Seed17RenderedQualificationJob:
    """Render all twelve validation-boundary qualifications in one Job."""

    return _render_m03r_v7_seed17_qualification_job(
        plan=plan,
        live_evidence=live_evidence,
        template=template,
        now_utc=now_utc,
        completion_index=None,
    )


def render_m03r_v7_seed17_top2000_suspended_indexed_job(
    *,
    package: M03RV7Seed17QualifiedPackage,
    live_evidence: M03RV7LiveAdmissionEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV7RenderedSuspendedJob:
    """Render the fresh 12-setting seed-17 Job, still suspended."""

    package.require_launch_ready()
    live_evidence.require_fresh(now_utc=now_utc)
    plan = package.plan
    qualification = package.qualification
    if not plan.plan_artifact_path.startswith(
        template.package_mount_path.rstrip("/") + "/"
    ):
        raise M03RV7Seed17KubernetesError(
            "seed-17 package plan must be below the read-only package mount"
        )
    if PurePosixPath(plan.source_pythonpath) != (
        PurePosixPath(template.package_mount_path) / "source" / "src"
    ):
        raise M03RV7Seed17KubernetesError(
            "seed-17 source Python path must bind the mounted package"
        )

    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v7-seed17-dev",
        "app.kubernetes.io/managed-by": "receipt-gated-research",
        "k8s-user": "yding4",
        "rl-quant/run-id": template.run_id,
        "rl-quant/owner": "yding4",
        "runai/queue": template.runai_queue,
    }
    annotations = {
        "rl-quant/run-id": template.run_id,
        "rl-quant/package-plan-sha256": plan.package_plan_sha256,
        "rl-quant/protocol-sha256": plan.protocol_sha256,
        "rl-quant/source-archive-sha256": (
            plan.artifacts.source_archive_sha256
        ),
        "rl-quant/cache-artifact-sha256": (
            plan.artifacts.cache_artifact_sha256
        ),
        "rl-quant/image-digest-sha256": (
            plan.artifacts.image_digest_sha256
        ),
        "rl-quant/execution-surface-sha256": (
            qualification.execution_surface_sha256
        ),
        "rl-quant/capacity-receipt-sha256": (
            qualification.capacity_receipt_sha256
        ),
        "rl-quant/validation-sentinel-receipt-sha256": (
            qualification.validation_sentinel_receipt_sha256
        ),
        "rl-quant/live-evidence-receipt-sha256": (
            live_evidence.receipt_sha256
        ),
        "rl-quant/data-role": "development-only-nonreportable",
        "rl-quant/paired-seeds": "17",
        "rl-quant/fold-count": "6",
        "rl-quant/one-member-fold-execution": "true",
        "rl-quant/five-seed-ensemble-eligible": "false",
    }
    argv = list(M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX) + [
        "--package-plan",
        plan.plan_artifact_path,
        "--package-plan-sha256",
        plan.package_plan_sha256,
        "--output-root",
        template.output_mount_path,
    ]
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "serviceAccount": template.service_account_name,
        "serviceAccountName": template.service_account_name,
        "schedulerName": template.scheduler_name,
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "dnsPolicy": "ClusterFirst",
        "nodeSelector": dict(M03R_TOP2000_H100_POOL_NODE_SELECTOR),
        "priorityClassName": M03R_TOP2000_PRIORITY_CLASS_NAME,
        "terminationGracePeriodSeconds": 60,
        "tolerations": [dict(M03R_TOP2000_MULTI_GPU_TOLERATION)],
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": live_evidence.gpu_product_label_key,
                                    "operator": "In",
                                    "values": list(
                                        live_evidence.gpu_product_label_values
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": template.run_as_user,
            "runAsGroup": template.run_as_group,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "trainer",
                "image": plan.artifacts.image_reference,
                "imagePullPolicy": "IfNotPresent",
                "terminationMessagePath": (
                    M03R_TOP2000_TERMINATION_MESSAGE_PATH
                ),
                "terminationMessagePolicy": (
                    M03R_TOP2000_TERMINATION_MESSAGE_POLICY
                ),
                "command": [argv[0]],
                "args": argv[1:],
                "env": [
                    {
                        "name": "JOB_COMPLETION_INDEX",
                        "valueFrom": {
                            "fieldRef": {
                                "apiVersion": "v1",
                                "fieldPath": (
                                    "metadata.annotations['batch.kubernetes.io/"
                                    "job-completion-index']"
                                ),
                            }
                        },
                    },
                    {"name": "NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
                    {"name": "TORCH_NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
                    {"name": "PYTHONNOUSERSITE", "value": "1"},
                    {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                    {"name": "PYTHONHASHSEED", "value": "0"},
                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                    {"name": "CUBLAS_WORKSPACE_CONFIG", "value": ":4096:8"},
                    {"name": "OMP_NUM_THREADS", "value": "8"},
                    {"name": "MKL_NUM_THREADS", "value": "8"},
                    {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
                    {
                        "name": "TORCHINDUCTOR_CACHE_DIR",
                        "value": "/tmp/torchinductor",
                    },
                    {"name": "TRITON_CACHE_DIR", "value": "/tmp/triton"},
                    {"name": "PYTHONPATH", "value": plan.source_pythonpath},
                ],
                "resources": {
                    "requests": {
                        "cpu": template.cpu_request,
                        "memory": template.memory_request,
                        "ephemeral-storage": (
                            template.ephemeral_storage_request
                        ),
                        "nvidia.com/gpu": str(
                            M03R_TOP2000_GPUS_PER_COMPLETION
                        ),
                    },
                    "limits": {
                        "cpu": template.cpu_limit,
                        "memory": template.memory_limit,
                        "ephemeral-storage": (
                            template.ephemeral_storage_limit
                        ),
                        "nvidia.com/gpu": str(
                            M03R_TOP2000_GPUS_PER_COMPLETION
                        ),
                    },
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [
                    {
                        "name": "research-data",
                        "mountPath": template.package_mount_path,
                        "subPath": (
                            template.pvc_training_subpath.rstrip("/")
                            + "/packages/"
                            + template.run_id
                        ),
                        "readOnly": True,
                    },
                    {
                        "name": "research-data",
                        "mountPath": template.output_mount_path,
                        "subPath": (
                            template.pvc_training_subpath.rstrip("/")
                            + "/runs/"
                            + template.run_id
                        ),
                    },
                    {"name": "tmp", "mountPath": "/tmp"},
                    {"name": "dshm", "mountPath": "/dev/shm"},
                ],
            }
        ],
        "volumes": [
            {
                "name": "research-data",
                "persistentVolumeClaim": {
                    "claimName": template.pvc_claim_name
                },
            },
            {"name": "tmp", "emptyDir": {}},
            {
                "name": "dshm",
                "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
            },
        ],
    }
    manifest: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": template.job_name,
            "namespace": M03R_TOP2000_KUBERNETES_NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "suspend": True,
            "completionMode": "Indexed",
            "completions": M03R_TOP2000_INDEXED_COMPLETIONS,
            "parallelism": live_evidence.allowed_parallelism,
            "backoffLimit": 0,
            "activeDeadlineSeconds": template.active_deadline_seconds,
            "ttlSecondsAfterFinished": template.ttl_seconds_after_finished,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": pod_spec,
            },
        },
    }
    return M03RV7RenderedSuspendedJob(
        manifest_json=_canonical_json(manifest),
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_canonical_pod_spec_sha256(pod_spec),
        package_plan_sha256=plan.package_plan_sha256,
        execution_surface_sha256=qualification.execution_surface_sha256,
        live_evidence_receipt_sha256=live_evidence.receipt_sha256,
        capacity_receipt_sha256=qualification.capacity_receipt_sha256,
        parallelism=live_evidence.allowed_parallelism,
    )


@dataclass(frozen=True, slots=True)
class M03RV7Seed17FoldReceiptRef:
    """One coordinate binds training, validation, and one-member execution."""

    fold_index: int
    seed: int
    completion_receipt_sha256: str
    validation_receipt_sha256: str
    fold_execution_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.fold_index not in M03R_SEED17_TOP2000_FOLDS or self.seed != 17:
            raise M03RV7Seed17KubernetesError(
                "seed-17 receipt coordinate must be fold 0..5 and seed 17"
            )
        for name in (
            "completion_receipt_sha256",
            "validation_receipt_sha256",
            "fold_execution_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class M03RV7Seed17IndexCompletionReceipt:
    """One setting completion with exactly six one-member fold receipts."""

    completion_index: int
    setting_index: int
    setting_id: str
    runtime_setting_id: str
    job_uid: str
    admitted_binding_sha256: str
    package_plan_sha256: str
    source_archive_sha256: str
    cache_artifact_sha256: str
    image_digest_sha256: str
    execution_surface_sha256: str
    capacity_receipt_sha256: str
    fold_receipts: tuple[M03RV7Seed17FoldReceiptRef, ...]
    output_manifest_sha256: str
    process_exit_code: int
    completion_succeeded: bool
    one_member_fold_execution: bool
    five_seed_ensemble_eligible: bool
    development_only: bool
    promotion_eligible: bool
    receipt_sha256: str
    schema: str = M03R_SEED17_TOP2000_INDEX_RECEIPT_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "completion_index": self.completion_index,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "runtime_setting_id": self.runtime_setting_id,
            "job_uid": self.job_uid,
            "admitted_binding_sha256": self.admitted_binding_sha256,
            "package_plan_sha256": self.package_plan_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "cache_artifact_sha256": self.cache_artifact_sha256,
            "image_digest_sha256": self.image_digest_sha256,
            "execution_surface_sha256": self.execution_surface_sha256,
            "capacity_receipt_sha256": self.capacity_receipt_sha256,
            "fold_receipts": [asdict(row) for row in self.fold_receipts],
            "output_manifest_sha256": self.output_manifest_sha256,
            "process_exit_code": self.process_exit_code,
            "completion_succeeded": self.completion_succeeded,
            "one_member_fold_execution": self.one_member_fold_execution,
            "five_seed_ensemble_eligible": self.five_seed_ensemble_eligible,
            "development_only": self.development_only,
            "promotion_eligible": self.promotion_eligible,
        }

    def __post_init__(self) -> None:
        expected_coordinates = tuple((fold, 17) for fold in range(6))
        observed_coordinates = tuple(
            (row.fold_index, row.seed) for row in self.fold_receipts
        )
        if (
            self.schema != M03R_SEED17_TOP2000_INDEX_RECEIPT_SCHEMA
            or not 0 <= self.completion_index < 12
            or not self.job_uid
            or self.process_exit_code != 0
            or not self.completion_succeeded
            or not self.one_member_fold_execution
            or self.five_seed_ensemble_eligible
            or not self.development_only
            or self.promotion_eligible
            or len(self.fold_receipts) != 6
            or observed_coordinates != expected_coordinates
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 index receipt requires six ordered successful fold executions"
            )
        for name in (
            "admitted_binding_sha256",
            "package_plan_sha256",
            "source_archive_sha256",
            "cache_artifact_sha256",
            "image_digest_sha256",
            "execution_surface_sha256",
            "capacity_receipt_sha256",
            "output_manifest_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Seed17KubernetesError(
                "seed-17 index completion receipt hash mismatch"
            )

    def verify_for(
        self,
        *,
        index_plan: M03RV7Seed17IndexPlan,
        package_plan: M03RV7Seed17PackagePlan,
        binding: M03RV7AdmittedJobBinding,
        qualification: M03RV7Seed17ExecutionQualification,
    ) -> None:
        qualification.verify_for(package_plan)
        if (
            self.completion_index != index_plan.completion_index
            or self.setting_index != index_plan.setting_index
            or self.setting_id != index_plan.setting_id
            or self.runtime_setting_id != index_plan.runtime_setting_id
            or self.job_uid != binding.job_uid
            or self.admitted_binding_sha256 != binding.receipt_sha256
            or self.package_plan_sha256 != package_plan.package_plan_sha256
            or self.source_archive_sha256
            != package_plan.artifacts.source_archive_sha256
            or self.cache_artifact_sha256
            != package_plan.artifacts.cache_artifact_sha256
            or self.image_digest_sha256
            != package_plan.artifacts.image_digest_sha256
            or self.execution_surface_sha256
            != qualification.execution_surface_sha256
            or self.capacity_receipt_sha256
            != qualification.capacity_receipt_sha256
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 index receipt does not bind the exact admitted completion"
            )


def build_m03r_v7_seed17_index_completion_receipt(
    *,
    package: M03RV7Seed17QualifiedPackage,
    binding: M03RV7AdmittedJobBinding,
    completion_index: int,
    fold_receipts: tuple[M03RV7Seed17FoldReceiptRef, ...],
    output_manifest_sha256: str,
) -> M03RV7Seed17IndexCompletionReceipt:
    """Bind one successful seed-17 setting and its exact six folds."""

    package.require_launch_ready()
    if not 0 <= completion_index < 12:
        raise M03RV7Seed17KubernetesError(
            "seed-17 completion_index must be in [0, 11]"
        )
    row = package.plan.indices[completion_index]
    fields: dict[str, Any] = {
        "completion_index": completion_index,
        "setting_index": row.setting_index,
        "setting_id": row.setting_id,
        "runtime_setting_id": row.runtime_setting_id,
        "job_uid": binding.job_uid,
        "admitted_binding_sha256": binding.receipt_sha256,
        "package_plan_sha256": package.plan.package_plan_sha256,
        "source_archive_sha256": package.plan.artifacts.source_archive_sha256,
        "cache_artifact_sha256": package.plan.artifacts.cache_artifact_sha256,
        "image_digest_sha256": package.plan.artifacts.image_digest_sha256,
        "execution_surface_sha256": (
            package.qualification.execution_surface_sha256
        ),
        "capacity_receipt_sha256": (
            package.qualification.capacity_receipt_sha256
        ),
        "fold_receipts": fold_receipts,
        "output_manifest_sha256": output_manifest_sha256,
        "process_exit_code": 0,
        "completion_succeeded": True,
        "one_member_fold_execution": True,
        "five_seed_ensemble_eligible": False,
        "development_only": True,
        "promotion_eligible": False,
        "schema": M03R_SEED17_TOP2000_INDEX_RECEIPT_SCHEMA,
    }
    unsigned = M03RV7Seed17IndexCompletionReceipt.__new__(
        M03RV7Seed17IndexCompletionReceipt
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    receipt = M03RV7Seed17IndexCompletionReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )
    receipt.verify_for(
        index_plan=row,
        package_plan=package.plan,
        binding=binding,
        qualification=package.qualification,
    )
    return receipt


@dataclass(frozen=True, slots=True)
class M03RV7Seed17IndexedBatchReceipt:
    """Final exact coverage of all twelve seed-17 setting completions."""

    index_receipts: tuple[M03RV7Seed17IndexCompletionReceipt, ...]
    job_uid: str
    admitted_binding_sha256: str
    package_plan_sha256: str
    all_twelve_complete: bool
    total_fold_executions: int
    one_member_fold_execution: bool
    five_seed_ensemble_eligible: bool
    development_only: bool
    promotion_eligible: bool
    receipt_sha256: str
    schema: str = M03R_SEED17_TOP2000_BATCH_RECEIPT_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "index_receipts": [asdict(row) for row in self.index_receipts],
            "job_uid": self.job_uid,
            "admitted_binding_sha256": self.admitted_binding_sha256,
            "package_plan_sha256": self.package_plan_sha256,
            "all_twelve_complete": self.all_twelve_complete,
            "total_fold_executions": self.total_fold_executions,
            "one_member_fold_execution": self.one_member_fold_execution,
            "five_seed_ensemble_eligible": self.five_seed_ensemble_eligible,
            "development_only": self.development_only,
            "promotion_eligible": self.promotion_eligible,
        }

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_SEED17_TOP2000_BATCH_RECEIPT_SCHEMA
            or len(self.index_receipts) != 12
            or tuple(row.completion_index for row in self.index_receipts)
            != tuple(range(12))
            or any(row.job_uid != self.job_uid for row in self.index_receipts)
            or not self.all_twelve_complete
            or self.total_fold_executions != 72
            or not self.one_member_fold_execution
            or self.five_seed_ensemble_eligible
            or not self.development_only
            or self.promotion_eligible
        ):
            raise M03RV7Seed17KubernetesError(
                "seed-17 batch receipt requires 12 settings and 72 fold executions"
            )
        for name in (
            "admitted_binding_sha256",
            "package_plan_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Seed17KubernetesError(
                "seed-17 indexed batch receipt hash mismatch"
            )


def build_m03r_v7_seed17_indexed_batch_receipt(
    *,
    package: M03RV7Seed17QualifiedPackage,
    binding: M03RV7AdmittedJobBinding,
    index_receipts: tuple[M03RV7Seed17IndexCompletionReceipt, ...],
) -> M03RV7Seed17IndexedBatchReceipt:
    """Validate all twelve index receipts and bind the 72-cell result."""

    package.require_launch_ready()
    ordered = tuple(
        sorted(index_receipts, key=lambda row: row.completion_index)
    )
    if len(ordered) != 12:
        raise M03RV7Seed17KubernetesError(
            "all twelve seed-17 index receipts are required"
        )
    for index_plan, receipt in zip(
        package.plan.indices, ordered, strict=True
    ):
        receipt.verify_for(
            index_plan=index_plan,
            package_plan=package.plan,
            binding=binding,
            qualification=package.qualification,
        )
    fields: dict[str, Any] = {
        "index_receipts": ordered,
        "job_uid": binding.job_uid,
        "admitted_binding_sha256": binding.receipt_sha256,
        "package_plan_sha256": package.plan.package_plan_sha256,
        "all_twelve_complete": True,
        "total_fold_executions": 72,
        "one_member_fold_execution": True,
        "five_seed_ensemble_eligible": False,
        "development_only": True,
        "promotion_eligible": False,
        "schema": M03R_SEED17_TOP2000_BATCH_RECEIPT_SCHEMA,
    }
    unsigned = M03RV7Seed17IndexedBatchReceipt.__new__(
        M03RV7Seed17IndexedBatchReceipt
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7Seed17IndexedBatchReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


__all__ = [
    "M03R_SEED17_TOP2000_BATCH_RECEIPT_SCHEMA",
    "M03R_SEED17_TOP2000_CAPACITY_RECEIPT_SCHEMA",
    "M03R_SEED17_TOP2000_EXECUTION_QUALIFICATION_SCHEMA",
    "M03R_SEED17_TOP2000_GPU_NAME",
    "M03R_SEED17_TOP2000_INDEX_RECEIPT_SCHEMA",
    "M03R_SEED17_TOP2000_QUALIFICATION_ARTIFACT_SCHEMA",
    "M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX",
    "M03RV7Seed17CapacityReceipt",
    "M03RV7Seed17ExecutionQualification",
    "M03RV7Seed17FoldReceiptRef",
    "M03RV7Seed17IndexCompletionReceipt",
    "M03RV7Seed17IndexedBatchReceipt",
    "M03RV7Seed17KubernetesError",
    "M03RV7Seed17QualificationArtifactRef",
    "M03RV7Seed17QualifiedPackage",
    "M03RV7Seed17RenderedQualificationJob",
    "build_m03r_v7_seed17_capacity_receipt",
    "build_m03r_v7_seed17_execution_qualification",
    "build_m03r_v7_seed17_index_completion_receipt",
    "build_m03r_v7_seed17_indexed_batch_receipt",
    "build_m03r_v7_seed17_qualification_artifact_ref",
    "render_m03r_v7_seed17_top2000_suspended_indexed_job",
    "render_m03r_v7_seed17_top2000_suspended_qualification_batch_job",
    "render_m03r_v7_seed17_top2000_suspended_validation_sentinel_job",
]
