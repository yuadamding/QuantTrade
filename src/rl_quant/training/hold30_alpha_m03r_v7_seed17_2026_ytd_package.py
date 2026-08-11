"""Immutable package identities for the seed-17 2026-YTD retrospective.

The historical seed-17 training package is intentionally a two-H100 training
surface.  This module owns a separate evaluation-only identity for one zero-GPU
preaccess stage, one scientifically reusable setting-0 H100 sentinel, the
remaining eleven one-H100 setting workers, and one zero-GPU panel aggregation.
It performs no staging, outcome access, Kubernetes calls, or file mutation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
)

TOP2000_M03R_V7_2026_SOURCE_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-source-package-v1"
)
TOP2000_M03R_V7_2026_EXECUTION_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-execution-package-v1"
)
TOP2000_M03R_V7_2026_SENTINEL_QUALIFICATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-sentinel-qualification-v1"
)
TOP2000_M03R_V7_2026_PANEL_ARTIFACT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-panel-artifacts-v1"
)

TOP2000_M03R_V7_2026_IMAGE_PYTHON = "/opt/conda/envs/quanttrade/bin/python"
TOP2000_M03R_V7_2026_SOURCE_PACKAGE_MOUNT = "/mnt/package"
TOP2000_M03R_V7_2026_Q2_PACKAGE_MOUNT = "/mnt/q2-package"
TOP2000_M03R_V7_2026_Q2_OUTPUT_MOUNT = "/mnt/output"
TOP2000_M03R_V7_2026_RAW_DATA_MOUNT = "/mnt/top2000-raw"
TOP2000_M03R_V7_2026_PREACCESS_MOUNT = "/mnt/preaccess"
TOP2000_M03R_V7_2026_EVALUATION_OUTPUT_MOUNT = "/mnt/evaluation-output"
TOP2000_M03R_V7_2026_AGGREGATION_OUTPUT_MOUNT = "/mnt/aggregate-output"

TOP2000_M03R_V7_2026_PREACCESS_WORKER_MODULE = (
    "rl_quant.workflows.top2000_m03r_v7_seed17_2026_ytd"
)
TOP2000_M03R_V7_2026_SETTING_WORKER_MODULE = (
    "rl_quant.workflows.top2000_m03r_v7_seed17_2026_execution"
)
TOP2000_M03R_V7_2026_PANEL_WORKER_MODULE = (
    "rl_quant.workflows.top2000_m03r_v7_seed17_2026_panel"
)
TOP2000_M03R_V7_2026_FOLD_EXECUTION_ORDER = (5, 0, 1, 2, 3, 4)

TOP2000_M03R_V7_2026_H100_GPU_NAME = "NVIDIA H100 80GB HBM3"
TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL = "NVIDIA-H100-80GB-HBM3"
TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES = 79 * 1024**3
TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES = 81 * 1024**3
TOP2000_M03R_V7_2026_H100_COMPUTE_CAPABILITY = (9, 0)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_RE = re.compile(r"^[^@\s]+@sha256:([0-9a-f]{64})$")


class Top2000M03RV72026PackageError(ValueError):
    """An evaluation package identity or phase boundary drifted."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Top2000M03RV72026PackageError(
            "evaluation package is not canonical finite ASCII JSON"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise Top2000M03RV72026PackageError(
            f"{name} must be one lowercase SHA-256"
        )


def _absolute_path(name: str, value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or value in {"", "/"} or ".." in path.parts:
        raise Top2000M03RV72026PackageError(
            f"{name} must be one scoped absolute container path"
        )
    return path


def _strict_child(name: str, value: str, parent: str) -> None:
    path = _absolute_path(name, value)
    root = _absolute_path(f"{name} parent", parent)
    if path == root or not path.is_relative_to(root):
        raise Top2000M03RV72026PackageError(
            f"{name} must be below {parent}"
        )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SourceArtifactBindings:
    """Inputs frozen before a zero-GPU Job may open 2026 outcomes."""

    source_archive_sha256: str
    source_manifest_sha256: str
    dependency_lock_sha256: str
    evaluation_source_inventory_sha256: str
    preaccess_entrypoint_sha256: str
    setting_worker_entrypoint_sha256: str
    panel_worker_entrypoint_sha256: str
    q2_package_plan_file_sha256: str
    q2_package_plan_receipt_sha256: str
    q2_completion_coverage_receipt_file_sha256: str
    q2_training_output_inventory_sha256: str
    pre2026_cache_file_sha256: str
    raw_data_manifest_file_sha256: str
    raw_universe_file_sha256: str
    image_reference: str
    image_digest_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {"image_reference"}:
                continue
            _require_sha256(name, value)
        match = _IMAGE_RE.fullmatch(self.image_reference)
        if match is None or match.group(1) != self.image_digest_sha256:
            raise Top2000M03RV72026PackageError(
                "image reference must be digest-pinned and match its digest"
            )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026ContainerPaths:
    """Stable absolute paths shared by preaccess, GPU, and aggregation Pods."""

    source_package_plan_path: str = "/mnt/package/evaluation-package-plan.json"
    evaluation_source_root: str = "/mnt/package/source"
    evaluation_source_pythonpath: str = "/mnt/package/source/src"
    q2_package_plan_path: str = "/mnt/q2-package/package-plan.json"
    pre2026_cache_path: str = "/mnt/q2-package/pre2026-cache.pt"
    q2_training_output_root: str = TOP2000_M03R_V7_2026_Q2_OUTPUT_MOUNT
    raw_data_root: str = TOP2000_M03R_V7_2026_RAW_DATA_MOUNT
    frozen_plan_path: str = "/mnt/preaccess/frozen-plan.json"
    retrospective_cache_path: str = "/mnt/preaccess/retrospective-cache.pt"
    cache_stage_receipt_path: str = "/mnt/preaccess/cache-stage-receipt.json"
    factor_retrieval_receipt_path: str = (
        "/mnt/preaccess/factor-retrieval-receipt.json"
    )
    factor_data_path: str = "/mnt/preaccess/factor-data.json"
    factor_stage_receipt_path: str = "/mnt/preaccess/factor-stage-receipt.json"
    preaccess_completion_receipt_path: str = (
        "/mnt/preaccess/preaccess-completion.json"
    )
    evaluation_output_root: str = TOP2000_M03R_V7_2026_EVALUATION_OUTPUT_MOUNT
    aggregation_output_root: str = TOP2000_M03R_V7_2026_AGGREGATION_OUTPUT_MOUNT

    def __post_init__(self) -> None:
        if (
            self.evaluation_source_root
            != TOP2000_M03R_V7_2026_SOURCE_PACKAGE_MOUNT + "/source"
            or self.evaluation_source_pythonpath
            != self.evaluation_source_root + "/src"
            or self.q2_training_output_root
            != TOP2000_M03R_V7_2026_Q2_OUTPUT_MOUNT
            or self.raw_data_root != TOP2000_M03R_V7_2026_RAW_DATA_MOUNT
            or self.evaluation_output_root
            != TOP2000_M03R_V7_2026_EVALUATION_OUTPUT_MOUNT
            or self.aggregation_output_root
            != TOP2000_M03R_V7_2026_AGGREGATION_OUTPUT_MOUNT
        ):
            raise Top2000M03RV72026PackageError(
                "stable source, q2, raw-data, or output mount path drifted"
            )
        for name in (
            "source_package_plan_path",
            "evaluation_source_root",
            "evaluation_source_pythonpath",
        ):
            _strict_child(
                name,
                getattr(self, name),
                TOP2000_M03R_V7_2026_SOURCE_PACKAGE_MOUNT,
            )
        for name in ("q2_package_plan_path", "pre2026_cache_path"):
            _strict_child(
                name,
                getattr(self, name),
                TOP2000_M03R_V7_2026_Q2_PACKAGE_MOUNT,
            )
        for name in (
            "frozen_plan_path",
            "retrospective_cache_path",
            "cache_stage_receipt_path",
            "factor_retrieval_receipt_path",
            "factor_data_path",
            "factor_stage_receipt_path",
            "preaccess_completion_receipt_path",
        ):
            _strict_child(
                name,
                getattr(self, name),
                TOP2000_M03R_V7_2026_PREACCESS_MOUNT,
            )
        roots = (
            TOP2000_M03R_V7_2026_SOURCE_PACKAGE_MOUNT,
            TOP2000_M03R_V7_2026_Q2_PACKAGE_MOUNT,
            TOP2000_M03R_V7_2026_Q2_OUTPUT_MOUNT,
            TOP2000_M03R_V7_2026_RAW_DATA_MOUNT,
            TOP2000_M03R_V7_2026_PREACCESS_MOUNT,
            self.evaluation_output_root,
            self.aggregation_output_root,
        )
        if len(set(roots)) != len(roots):
            raise Top2000M03RV72026PackageError(
                "input and phase-output mount paths must remain disjoint"
            )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SettingMapEntry:
    phase: Literal["sentinel", "remaining"]
    local_completion_index: int
    setting_index: int
    setting_id: str

    def __post_init__(self) -> None:
        if (
            self.phase not in {"sentinel", "remaining"}
            or isinstance(self.local_completion_index, bool)
            or not isinstance(self.local_completion_index, int)
            or isinstance(self.setting_index, bool)
            or not isinstance(self.setting_index, int)
            or self.setting_index not in range(12)
            or self.setting_id != M03R_SEED17_TOP2000_SETTING_IDS[self.setting_index]
        ):
            raise Top2000M03RV72026PackageError(
                "setting map row has an invalid local/global identity"
            )
        expected = 0 if self.phase == "sentinel" else self.setting_index - 1
        if self.local_completion_index != expected:
            raise Top2000M03RV72026PackageError(
                "sentinel must map 0->0 and remaining rows must map local i to i+1"
            )


def _sentinel_map() -> tuple[Top2000M03RV72026SettingMapEntry, ...]:
    return (
        Top2000M03RV72026SettingMapEntry(
            phase="sentinel",
            local_completion_index=0,
            setting_index=0,
            setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        ),
    )


def _remaining_map() -> tuple[Top2000M03RV72026SettingMapEntry, ...]:
    return tuple(
        Top2000M03RV72026SettingMapEntry(
            phase="remaining",
            local_completion_index=setting_index - 1,
            setting_index=setting_index,
            setting_id=M03R_SEED17_TOP2000_SETTING_IDS[setting_index],
        )
        for setting_index in range(1, 12)
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SourcePackagePlan:
    """Pre-outcome package consumed by the sole zero-GPU preaccess Job."""

    artifacts: Top2000M03RV72026SourceArtifactBindings
    paths: Top2000M03RV72026ContainerPaths
    sentinel_map: tuple[Top2000M03RV72026SettingMapEntry, ...]
    remaining_map: tuple[Top2000M03RV72026SettingMapEntry, ...]
    package_plan_sha256: str
    evaluation_protocol_sha256: str = (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
    )
    preaccess_worker_module: str = TOP2000_M03R_V7_2026_PREACCESS_WORKER_MODULE
    setting_worker_module: str = TOP2000_M03R_V7_2026_SETTING_WORKER_MODULE
    panel_worker_module: str = TOP2000_M03R_V7_2026_PANEL_WORKER_MODULE
    fold_execution_order: tuple[int, ...] = TOP2000_M03R_V7_2026_FOLD_EXECUTION_ORDER
    setting_zero_repeated_in_remaining_job: bool = False
    disposable_all_setting_qualification_required: bool = False
    training_artifacts_mutable: bool = False
    development_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_SOURCE_PACKAGE_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("package_plan_sha256")
        return payload

    def __post_init__(self) -> None:
        sentinel_settings = tuple(row.setting_index for row in self.sentinel_map)
        remaining_settings = tuple(row.setting_index for row in self.remaining_map)
        if (
            self.schema != TOP2000_M03R_V7_2026_SOURCE_PACKAGE_SCHEMA
            or self.evaluation_protocol_sha256
            != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
            or self.preaccess_worker_module
            != TOP2000_M03R_V7_2026_PREACCESS_WORKER_MODULE
            or self.setting_worker_module
            != TOP2000_M03R_V7_2026_SETTING_WORKER_MODULE
            or self.panel_worker_module != TOP2000_M03R_V7_2026_PANEL_WORKER_MODULE
            or self.fold_execution_order != TOP2000_M03R_V7_2026_FOLD_EXECUTION_ORDER
            or self.sentinel_map != _sentinel_map()
            or self.remaining_map != _remaining_map()
            or sentinel_settings != (0,)
            or remaining_settings != tuple(range(1, 12))
            or set(sentinel_settings).intersection(remaining_settings)
            or set(sentinel_settings + remaining_settings) != set(range(12))
            or self.setting_zero_repeated_in_remaining_job
            or self.disposable_all_setting_qualification_required
            or self.training_artifacts_mutable
            or not self.development_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026PackageError(
                "source package map, execution surface, or evidence role drifted"
            )
        _require_sha256("package_plan_sha256", self.package_plan_sha256)
        if self.package_plan_sha256 != _sha256(self.canonical_payload()):
            raise Top2000M03RV72026PackageError("source package hash mismatch")

    @property
    def setting_map_sha256(self) -> str:
        return _sha256(
            {
                "sentinel": [asdict(row) for row in self.sentinel_map],
                "remaining": [asdict(row) for row in self.remaining_map],
            }
        )

    @property
    def expected_output_inventory_sha256(self) -> str:
        rows: list[dict[str, Any]] = []
        for setting_index in range(12):
            setting_root = f"{self.paths.evaluation_output_root}/setting-{setting_index:02d}"
            for fold_index in self.fold_execution_order:
                rows.append(
                    {
                        "setting_index": setting_index,
                        "fold_index": fold_index,
                        "execution_path": (
                            f"{setting_root}/fold-{fold_index:02d}.execution.json"
                        ),
                        "binding_path": (
                            f"{setting_root}/fold-{fold_index:02d}.artifact-binding.json"
                        ),
                    }
                )
            rows.append(
                {
                    "setting_index": setting_index,
                    "completion_path": f"{setting_root}/setting-completion.json",
                }
            )
        return _sha256(rows)


def build_top2000_m03r_v7_seed17_2026_source_package_plan(
    *,
    artifacts: Top2000M03RV72026SourceArtifactBindings,
    paths: Top2000M03RV72026ContainerPaths | None = None,
) -> Top2000M03RV72026SourcePackagePlan:
    fields: dict[str, Any] = {
        "artifacts": artifacts,
        "paths": paths or Top2000M03RV72026ContainerPaths(),
        "sentinel_map": _sentinel_map(),
        "remaining_map": _remaining_map(),
        "evaluation_protocol_sha256": (
            M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
        ),
        "preaccess_worker_module": TOP2000_M03R_V7_2026_PREACCESS_WORKER_MODULE,
        "setting_worker_module": TOP2000_M03R_V7_2026_SETTING_WORKER_MODULE,
        "panel_worker_module": TOP2000_M03R_V7_2026_PANEL_WORKER_MODULE,
        "fold_execution_order": TOP2000_M03R_V7_2026_FOLD_EXECUTION_ORDER,
        "setting_zero_repeated_in_remaining_job": False,
        "disposable_all_setting_qualification_required": False,
        "training_artifacts_mutable": False,
        "development_only": True,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
        "schema": TOP2000_M03R_V7_2026_SOURCE_PACKAGE_SCHEMA,
    }
    unsigned = Top2000M03RV72026SourcePackagePlan.__new__(
        Top2000M03RV72026SourcePackagePlan
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    object.__setattr__(unsigned, "package_plan_sha256", "0" * 64)
    return Top2000M03RV72026SourcePackagePlan(
        **fields,
        package_plan_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026PreaccessArtifactBindings:
    frozen_plan_file_sha256: str
    frozen_plan_receipt_sha256: str
    retrospective_cache_file_sha256: str
    cache_stage_receipt_file_sha256: str
    chronology_receipt_sha256: str
    factor_retrieval_receipt_file_sha256: str
    factor_retrieval_receipt_sha256: str
    factor_data_file_sha256: str
    factor_data_receipt_sha256: str
    factor_stage_receipt_file_sha256: str
    preaccess_completion_receipt_file_sha256: str
    preaccess_completion_receipt_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_sha256(name, value)


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026ExecutionPackagePlan:
    source_package: Top2000M03RV72026SourcePackagePlan
    preaccess: Top2000M03RV72026PreaccessArtifactBindings
    execution_package_sha256: str
    expected_gpu_name: str = TOP2000_M03R_V7_2026_H100_GPU_NAME
    expected_gpu_product_label: str = TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL
    expected_gpu_memory_min_bytes: int = (
        TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES
    )
    expected_gpu_memory_max_bytes: int = (
        TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES
    )
    expected_compute_capability: tuple[int, int] = (
        TOP2000_M03R_V7_2026_H100_COMPUTE_CAPABILITY
    )
    gpu_count_per_setting_worker: int = 1
    process_count_per_setting_worker: int = 1
    sentinel_outputs_reused: bool = True
    remaining_job_setting_zero_count: int = 0
    training_artifacts_mutable: bool = False
    development_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_EXECUTION_PACKAGE_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("execution_package_sha256")
        return payload

    def __post_init__(self) -> None:
        self.source_package.__post_init__()
        self.preaccess.__post_init__()
        if (
            self.schema != TOP2000_M03R_V7_2026_EXECUTION_PACKAGE_SCHEMA
            or self.expected_gpu_name != TOP2000_M03R_V7_2026_H100_GPU_NAME
            or self.expected_gpu_product_label
            != TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL
            or self.expected_gpu_memory_min_bytes
            != TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES
            or self.expected_gpu_memory_max_bytes
            != TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES
            or self.expected_compute_capability
            != TOP2000_M03R_V7_2026_H100_COMPUTE_CAPABILITY
            or self.gpu_count_per_setting_worker != 1
            or self.process_count_per_setting_worker != 1
            or not self.sentinel_outputs_reused
            or self.remaining_job_setting_zero_count != 0
            or self.training_artifacts_mutable
            or not self.development_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026PackageError(
                "execution package changed the one-H100 research-only surface"
            )
        _require_sha256("execution_package_sha256", self.execution_package_sha256)
        if self.execution_package_sha256 != _sha256(self.canonical_payload()):
            raise Top2000M03RV72026PackageError(
                "execution package hash mismatch"
            )


def build_top2000_m03r_v7_seed17_2026_execution_package_plan(
    *,
    source_package: Top2000M03RV72026SourcePackagePlan,
    preaccess: Top2000M03RV72026PreaccessArtifactBindings,
) -> Top2000M03RV72026ExecutionPackagePlan:
    fields: dict[str, Any] = {
        "source_package": source_package,
        "preaccess": preaccess,
        "expected_gpu_name": TOP2000_M03R_V7_2026_H100_GPU_NAME,
        "expected_gpu_product_label": TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL,
        "expected_gpu_memory_min_bytes": (
            TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES
        ),
        "expected_gpu_memory_max_bytes": (
            TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES
        ),
        "expected_compute_capability": (
            TOP2000_M03R_V7_2026_H100_COMPUTE_CAPABILITY
        ),
        "gpu_count_per_setting_worker": 1,
        "process_count_per_setting_worker": 1,
        "sentinel_outputs_reused": True,
        "remaining_job_setting_zero_count": 0,
        "training_artifacts_mutable": False,
        "development_only": True,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
        "schema": TOP2000_M03R_V7_2026_EXECUTION_PACKAGE_SCHEMA,
    }
    unsigned = Top2000M03RV72026ExecutionPackagePlan.__new__(
        Top2000M03RV72026ExecutionPackagePlan
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    object.__setattr__(unsigned, "execution_package_sha256", "0" * 64)
    return Top2000M03RV72026ExecutionPackagePlan(
        **fields,
        execution_package_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SentinelQualification:
    execution_package_sha256: str
    setting_completion_file_sha256: str
    setting_completion_receipt_sha256: str
    six_fold_artifact_inventory_sha256: str
    h100_runtime_proof_inventory_sha256: str
    maximum_peak_reserved_bytes: int
    receipt_sha256: str
    setting_index: int = 0
    completed_fold_count: int = 6
    output_reused_as_setting_zero: bool = True
    disposable_capacity_only_run: bool = False
    development_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_SENTINEL_QUALIFICATION_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def __post_init__(self) -> None:
        for name in (
            "execution_package_sha256",
            "setting_completion_file_sha256",
            "setting_completion_receipt_sha256",
            "six_fold_artifact_inventory_sha256",
            "h100_runtime_proof_inventory_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            self.schema != TOP2000_M03R_V7_2026_SENTINEL_QUALIFICATION_SCHEMA
            or self.setting_index != 0
            or self.completed_fold_count != 6
            or self.maximum_peak_reserved_bytes <= 0
            or not self.output_reused_as_setting_zero
            or self.disposable_capacity_only_run
            or not self.development_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
            or self.receipt_sha256 != _sha256(self.canonical_payload())
        ):
            raise Top2000M03RV72026PackageError(
                "sentinel qualification is incomplete, disposable, or overclaims"
            )


def build_top2000_m03r_v7_seed17_2026_sentinel_qualification(
    *,
    execution_package: Top2000M03RV72026ExecutionPackagePlan,
    setting_completion_file_sha256: str,
    setting_completion_receipt_sha256: str,
    six_fold_artifact_inventory_sha256: str,
    h100_runtime_proof_inventory_sha256: str,
    maximum_peak_reserved_bytes: int,
) -> Top2000M03RV72026SentinelQualification:
    fields: dict[str, Any] = {
        "execution_package_sha256": execution_package.execution_package_sha256,
        "setting_completion_file_sha256": setting_completion_file_sha256,
        "setting_completion_receipt_sha256": setting_completion_receipt_sha256,
        "six_fold_artifact_inventory_sha256": six_fold_artifact_inventory_sha256,
        "h100_runtime_proof_inventory_sha256": h100_runtime_proof_inventory_sha256,
        "maximum_peak_reserved_bytes": maximum_peak_reserved_bytes,
        "setting_index": 0,
        "completed_fold_count": 6,
        "output_reused_as_setting_zero": True,
        "disposable_capacity_only_run": False,
        "development_only": True,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
        "schema": TOP2000_M03R_V7_2026_SENTINEL_QUALIFICATION_SCHEMA,
    }
    unsigned = Top2000M03RV72026SentinelQualification.__new__(
        Top2000M03RV72026SentinelQualification
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    object.__setattr__(unsigned, "receipt_sha256", "0" * 64)
    return Top2000M03RV72026SentinelQualification(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026PanelArtifactBindings:
    execution_package_sha256: str
    sentinel_qualification_receipt_sha256: str
    sentinel_setting_completion_file_sha256: str
    remaining_batch_completion_file_sha256: str
    twelve_setting_completion_inventory_sha256: str
    seventy_two_fold_artifact_inventory_sha256: str
    panel_artifact_receipt_sha256: str
    setting_count: int = 12
    fold_artifact_count: int = 72
    setting_zero_source: str = "reused-sentinel"
    training_artifacts_mutated: bool = False
    development_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_PANEL_ARTIFACT_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("panel_artifact_receipt_sha256")
        return payload

    def __post_init__(self) -> None:
        for name in (
            "execution_package_sha256",
            "sentinel_qualification_receipt_sha256",
            "sentinel_setting_completion_file_sha256",
            "remaining_batch_completion_file_sha256",
            "twelve_setting_completion_inventory_sha256",
            "seventy_two_fold_artifact_inventory_sha256",
            "panel_artifact_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            self.schema != TOP2000_M03R_V7_2026_PANEL_ARTIFACT_SCHEMA
            or self.setting_count != 12
            or self.fold_artifact_count != 72
            or self.setting_zero_source != "reused-sentinel"
            or self.training_artifacts_mutated
            or not self.development_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
            or self.panel_artifact_receipt_sha256
            != _sha256(self.canonical_payload())
        ):
            raise Top2000M03RV72026PackageError(
                "panel artifact binding is incomplete or repeats setting zero"
            )


def build_top2000_m03r_v7_seed17_2026_panel_artifact_bindings(
    *,
    execution_package: Top2000M03RV72026ExecutionPackagePlan,
    sentinel: Top2000M03RV72026SentinelQualification,
    remaining_batch_completion_file_sha256: str,
    twelve_setting_completion_inventory_sha256: str,
    seventy_two_fold_artifact_inventory_sha256: str,
) -> Top2000M03RV72026PanelArtifactBindings:
    if sentinel.execution_package_sha256 != execution_package.execution_package_sha256:
        raise Top2000M03RV72026PackageError(
            "sentinel and panel execution package hashes differ"
        )
    fields: dict[str, Any] = {
        "execution_package_sha256": execution_package.execution_package_sha256,
        "sentinel_qualification_receipt_sha256": sentinel.receipt_sha256,
        "sentinel_setting_completion_file_sha256": (
            sentinel.setting_completion_file_sha256
        ),
        "remaining_batch_completion_file_sha256": (
            remaining_batch_completion_file_sha256
        ),
        "twelve_setting_completion_inventory_sha256": (
            twelve_setting_completion_inventory_sha256
        ),
        "seventy_two_fold_artifact_inventory_sha256": (
            seventy_two_fold_artifact_inventory_sha256
        ),
        "setting_count": 12,
        "fold_artifact_count": 72,
        "setting_zero_source": "reused-sentinel",
        "training_artifacts_mutated": False,
        "development_only": True,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
        "schema": TOP2000_M03R_V7_2026_PANEL_ARTIFACT_SCHEMA,
    }
    unsigned = Top2000M03RV72026PanelArtifactBindings.__new__(
        Top2000M03RV72026PanelArtifactBindings
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    object.__setattr__(unsigned, "panel_artifact_receipt_sha256", "0" * 64)
    return Top2000M03RV72026PanelArtifactBindings(
        **fields,
        panel_artifact_receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


__all__ = [
    "TOP2000_M03R_V7_2026_AGGREGATION_OUTPUT_MOUNT",
    "TOP2000_M03R_V7_2026_EVALUATION_OUTPUT_MOUNT",
    "TOP2000_M03R_V7_2026_FOLD_EXECUTION_ORDER",
    "TOP2000_M03R_V7_2026_H100_COMPUTE_CAPABILITY",
    "TOP2000_M03R_V7_2026_H100_GPU_NAME",
    "TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES",
    "TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES",
    "TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL",
    "TOP2000_M03R_V7_2026_IMAGE_PYTHON",
    "TOP2000_M03R_V7_2026_PANEL_WORKER_MODULE",
    "TOP2000_M03R_V7_2026_PREACCESS_MOUNT",
    "TOP2000_M03R_V7_2026_PREACCESS_WORKER_MODULE",
    "TOP2000_M03R_V7_2026_Q2_OUTPUT_MOUNT",
    "TOP2000_M03R_V7_2026_RAW_DATA_MOUNT",
    "TOP2000_M03R_V7_2026_SETTING_WORKER_MODULE",
    "TOP2000_M03R_V7_2026_SOURCE_PACKAGE_MOUNT",
    "Top2000M03RV72026ContainerPaths",
    "Top2000M03RV72026ExecutionPackagePlan",
    "Top2000M03RV72026PackageError",
    "Top2000M03RV72026PanelArtifactBindings",
    "Top2000M03RV72026PreaccessArtifactBindings",
    "Top2000M03RV72026SentinelQualification",
    "Top2000M03RV72026SettingMapEntry",
    "Top2000M03RV72026SourceArtifactBindings",
    "Top2000M03RV72026SourcePackagePlan",
    "build_top2000_m03r_v7_seed17_2026_execution_package_plan",
    "build_top2000_m03r_v7_seed17_2026_panel_artifact_bindings",
    "build_top2000_m03r_v7_seed17_2026_sentinel_qualification",
    "build_top2000_m03r_v7_seed17_2026_source_package_plan",
]
