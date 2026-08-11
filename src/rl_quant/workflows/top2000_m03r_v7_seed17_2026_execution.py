"""One-setting, one-H100 execution worker for the frozen 2026 retrospective.

``JOB_COMPLETION_INDEX`` is a local worker index resolved through an explicit,
content-bound setting tuple.  The governed topology uses ``(0,)`` for the
reusable sentinel and ``(1, ..., 11)`` for the remainder; a full tuple remains
available for local qualification.  Each worker loads the immutable chronology
once, executes fold 5 first, then folds 0..4, and publishes six no-clobber
checkpoint artifacts plus one setting completion receipt.  This is
evaluation-only research code: it cannot train, aggregate the panel, report a
scientific result, or promote a setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from rl_quant.evaluation.top2000_m03r_v7_2026_execution import (
    TOP2000_M03R_V7_2026_MAXIMUM_GPU_MEMORY_BYTES,
    TOP2000_M03R_V7_2026_MINIMUM_GPU_MEMORY_BYTES,
    TOP2000_M03R_V7_2026_REQUIRED_COMPUTE_CAPABILITY,
    TOP2000_M03R_V7_2026_REQUIRED_GPU_NAME,
    Top2000M03RV72026ExecutionArtifactBinding,
    Top2000M03RV72026ExecutionError,
    Top2000M03RV72026ExecutionSession,
    Top2000M03RV72026LoadedExecutionArtifact,
    load_top2000_m03r_v7_seed17_2026_execution_artifact,
    prepare_top2000_m03r_v7_seed17_2026_execution_session,
    run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session,
)
from rl_quant.workflows.top2000_m03r_v7_seed17_2026_ytd import (
    Top2000M03RV7Seed172026YTDCheckpointBinding,
    Top2000M03RV7Seed172026YTDFrozenPlan,
    Top2000M03RV7Seed172026YTDWorkflowError,
    load_top2000_m03r_v7_seed17_2026_ytd_plan,
)

TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER = (5, 0, 1, 2, 3, 4)
TOP2000_M03R_V7_2026_SENTINEL_SETTING_INDEX_MAP = (0,)
TOP2000_M03R_V7_2026_REMAINDER_SETTING_INDEX_MAP = tuple(range(1, 12))
TOP2000_M03R_V7_2026_FULL_SETTING_INDEX_MAP = tuple(range(12))
TOP2000_M03R_V7_2026_ALLOWED_SETTING_INDEX_MAPS = (
    TOP2000_M03R_V7_2026_SENTINEL_SETTING_INDEX_MAP,
    TOP2000_M03R_V7_2026_REMAINDER_SETTING_INDEX_MAP,
    TOP2000_M03R_V7_2026_FULL_SETTING_INDEX_MAP,
)
TOP2000_M03R_V7_2026_SETTING_INDEX_MAP_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-setting-index-map-v1"
)
TOP2000_M03R_V7_2026_FOLD_ARTIFACT_BINDING_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-fold-artifact-binding-v1"
)
TOP2000_M03R_V7_2026_SETTING_COMPLETION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-setting-completion-v1"
)
_RECEIPT_MAX_BYTES = 1024 * 1024


class Top2000M03RV72026SettingWorkerError(RuntimeError):
    """A setting index, immutable child, or completion graph failed closed."""


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SettingIndexResolution:
    local_completion_index: int
    setting_index: int
    setting_index_map: tuple[int, ...]
    setting_index_map_sha256: str

    def __post_init__(self) -> None:
        expected_map = _validated_setting_index_map(self.setting_index_map)
        _require_digest("setting_index_map_sha256", self.setting_index_map_sha256)
        if (
            self.setting_index_map != expected_map
            or self.setting_index_map_sha256 != _setting_index_map_sha256(expected_map)
            or self.local_completion_index not in range(len(expected_map))
            or self.setting_index != expected_map[self.local_completion_index]
        ):
            raise Top2000M03RV72026SettingWorkerError(
                "local completion index does not match its frozen setting map"
            )


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026SettingWorkerError(
            "setting-worker payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026SettingWorkerError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _validated_setting_index_map(value: Sequence[int]) -> tuple[int, ...]:
    result = tuple(value)
    if (
        result not in TOP2000_M03R_V7_2026_ALLOWED_SETTING_INDEX_MAPS
        or any(isinstance(item, bool) or not isinstance(item, int) for item in result)
        or len(set(result)) != len(result)
        or any(item not in range(12) for item in result)
    ):
        raise Top2000M03RV72026SettingWorkerError(
            "setting-index map must be the frozen sentinel, remainder, or full tuple"
        )
    return result


def _setting_index_map_sha256(value: Sequence[int]) -> str:
    setting_map = _validated_setting_index_map(value)
    return _sha256(
        {
            "schema": TOP2000_M03R_V7_2026_SETTING_INDEX_MAP_SCHEMA,
            "setting_indices": list(setting_map),
        }
    )


def _parse_setting_index_map(value: str) -> tuple[int, ...]:
    if not value or value.strip() != value:
        raise argparse.ArgumentTypeError("setting-index map must be comma-separated")
    try:
        result = tuple(int(item) for item in value.split(","))
        return _validated_setting_index_map(result)
    except (ValueError, Top2000M03RV72026SettingWorkerError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _publish_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    raw = _canonical_json(dict(payload))
    if len(raw) > _RECEIPT_MAX_BYTES:
        raise Top2000M03RV72026SettingWorkerError(
            "setting-worker receipt exceeds its one-MiB bound"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable receipt {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary receipt path already exists: {temporary}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(raw)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(raw).hexdigest()


def _read_canonical_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size > _RECEIPT_MAX_BYTES
    ):
        raise Top2000M03RV72026SettingWorkerError(
            f"{label} must be a bounded regular non-symlink file"
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV72026SettingWorkerError(
            f"{label} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        raise Top2000M03RV72026SettingWorkerError(
            f"{label} is not exact canonical JSON"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def _typed_payload(payload: Mapping[str, Any], cls: type[Any], *, label: str) -> dict[str, Any]:
    if set(payload) != {field.name for field in fields(cls)}:
        raise Top2000M03RV72026SettingWorkerError(f"{label} fields drifted")
    return dict(payload)


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve(strict=False)


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026FoldArtifactBindingReceipt:
    """Exact plan/checkpoint/artifact binding for one completed fold."""

    local_completion_index: int
    setting_index_map: tuple[int, ...]
    setting_index_map_sha256: str
    training_completion_index: int
    setting_index: int
    setting_id: str
    runtime_setting_id: str
    training_fold_index: int
    checkpoint_role: str
    frozen_plan_file_sha256: str
    frozen_plan_receipt_sha256: str
    execution_source_inventory_sha256: str
    pre2026_cache_file_sha256: str
    retrospective_cache_file_sha256: str
    retrospective_chronology_receipt_sha256: str
    frozen_checkpoint_binding_sha256: str
    checkpoint_model_file_sha256: str
    checkpoint_model_state_sha256: str
    policy_model_state_sha256_before: str
    policy_model_state_sha256_after: str
    artifact_path: str
    artifact_file_sha256: str
    execution_receipt_sha256: str
    elapsed_wall_seconds: float
    visible_cuda_device_count: int
    gpu_name: str
    gpu_total_memory_bytes: int
    compute_capability: tuple[int, int]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    allocator_oom_count_delta: int
    allocator_retry_count_delta: int
    canonical_pass_count: int = 1
    optimizer_step_count: int = 0
    policy_state_changed: bool = False
    training_artifacts_mutated: bool = False
    development_only: bool = True
    dataset_reportable: bool = False
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_FOLD_ARTIFACT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "frozen_plan_file_sha256",
            "frozen_plan_receipt_sha256",
            "execution_source_inventory_sha256",
            "pre2026_cache_file_sha256",
            "retrospective_cache_file_sha256",
            "retrospective_chronology_receipt_sha256",
            "frozen_checkpoint_binding_sha256",
            "checkpoint_model_file_sha256",
            "checkpoint_model_state_sha256",
            "policy_model_state_sha256_before",
            "policy_model_state_sha256_after",
            "artifact_file_sha256",
            "execution_receipt_sha256",
            "setting_index_map_sha256",
        ):
            _require_digest(name, getattr(self, name))
        setting_map = _validated_setting_index_map(self.setting_index_map)
        if (
            self.schema != TOP2000_M03R_V7_2026_FOLD_ARTIFACT_BINDING_SCHEMA
            or self.setting_index_map != setting_map
            or self.setting_index_map_sha256 != _setting_index_map_sha256(setting_map)
            or self.local_completion_index not in range(len(setting_map))
            or self.setting_index != setting_map[self.local_completion_index]
            or self.training_completion_index not in range(12)
            or not self.setting_id
            or not self.runtime_setting_id
            or self.training_fold_index not in range(6)
            or self.checkpoint_role
            != ("headline" if self.training_fold_index == 5 else "cutoff-sensitivity")
            or not self.artifact_path
            or not np.isfinite(self.elapsed_wall_seconds)
            or self.elapsed_wall_seconds <= 0.0
            or self.visible_cuda_device_count != 1
            or self.gpu_name != TOP2000_M03R_V7_2026_REQUIRED_GPU_NAME
            or not TOP2000_M03R_V7_2026_MINIMUM_GPU_MEMORY_BYTES
            <= self.gpu_total_memory_bytes
            <= TOP2000_M03R_V7_2026_MAXIMUM_GPU_MEMORY_BYTES
            or self.compute_capability
            != TOP2000_M03R_V7_2026_REQUIRED_COMPUTE_CAPABILITY
            or self.peak_allocated_bytes < 0
            or self.peak_reserved_bytes < self.peak_allocated_bytes
            or self.allocator_oom_count_delta != 0
            or self.allocator_retry_count_delta != 0
            or self.policy_model_state_sha256_before
            != self.checkpoint_model_state_sha256
            or self.policy_model_state_sha256_after
            != self.checkpoint_model_state_sha256
            or self.canonical_pass_count != 1
            or self.optimizer_step_count != 0
            or self.policy_state_changed
            or self.training_artifacts_mutated
            or not self.development_only
            or self.dataset_reportable
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026SettingWorkerError(
                "fold artifact binding drifted or overclaims its evidence"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SettingFoldCompletion:
    training_fold_index: int
    artifact_binding_path: str
    artifact_binding_file_sha256: str
    artifact_binding_receipt_sha256: str
    artifact_path: str
    artifact_file_sha256: str
    execution_receipt_sha256: str
    elapsed_wall_seconds: float
    visible_cuda_device_count: int
    gpu_name: str
    gpu_total_memory_bytes: int
    compute_capability: tuple[int, int]
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "artifact_binding_file_sha256",
            "artifact_binding_receipt_sha256",
            "artifact_file_sha256",
            "execution_receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.training_fold_index not in range(6)
            or not self.artifact_binding_path
            or not self.artifact_path
            or not np.isfinite(self.elapsed_wall_seconds)
            or self.elapsed_wall_seconds <= 0.0
            or self.visible_cuda_device_count != 1
            or self.gpu_name != TOP2000_M03R_V7_2026_REQUIRED_GPU_NAME
            or not TOP2000_M03R_V7_2026_MINIMUM_GPU_MEMORY_BYTES
            <= self.gpu_total_memory_bytes
            <= TOP2000_M03R_V7_2026_MAXIMUM_GPU_MEMORY_BYTES
            or self.compute_capability
            != TOP2000_M03R_V7_2026_REQUIRED_COMPUTE_CAPABILITY
            or self.peak_allocated_bytes < 0
            or self.peak_reserved_bytes < self.peak_allocated_bytes
        ):
            raise Top2000M03RV72026SettingWorkerError(
                "setting fold-completion inventory drifted"
            )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SettingCompletionReceipt:
    local_completion_index: int
    setting_index_map: tuple[int, ...]
    setting_index_map_sha256: str
    setting_index: int
    setting_id: str
    runtime_setting_id: str
    frozen_plan_path: str
    frozen_plan_file_sha256: str
    frozen_plan_receipt_sha256: str
    execution_source_inventory_sha256: str
    pre2026_cache_file_sha256: str
    retrospective_cache_file_sha256: str
    retrospective_chronology_receipt_sha256: str
    fold_execution_order: tuple[int, ...]
    fold_artifacts: tuple[Top2000M03RV72026SettingFoldCompletion, ...]
    completed_fold_count: int
    total_elapsed_wall_seconds: float
    visible_cuda_device_count: int
    gpu_name: str
    gpu_total_memory_bytes: int
    compute_capability: tuple[int, int]
    maximum_peak_allocated_bytes: int
    maximum_peak_reserved_bytes: int
    one_resident_data_session: bool = True
    folds_pooled: bool = False
    panel_aggregation_performed: bool = False
    policy_training_authorized: bool = False
    training_artifacts_mutated: bool = False
    development_only: bool = True
    dataset_reportable: bool = False
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_SETTING_COMPLETION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "frozen_plan_file_sha256",
            "frozen_plan_receipt_sha256",
            "execution_source_inventory_sha256",
            "pre2026_cache_file_sha256",
            "retrospective_cache_file_sha256",
            "retrospective_chronology_receipt_sha256",
            "setting_index_map_sha256",
        ):
            _require_digest(name, getattr(self, name))
        setting_map = _validated_setting_index_map(self.setting_index_map)
        if (
            self.schema != TOP2000_M03R_V7_2026_SETTING_COMPLETION_SCHEMA
            or self.setting_index_map != setting_map
            or self.setting_index_map_sha256 != _setting_index_map_sha256(setting_map)
            or self.local_completion_index not in range(len(setting_map))
            or self.setting_index != setting_map[self.local_completion_index]
            or not self.setting_id
            or not self.runtime_setting_id
            or not self.frozen_plan_path
            or self.fold_execution_order
            != TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER
            or tuple(row.training_fold_index for row in self.fold_artifacts)
            != self.fold_execution_order
            or self.completed_fold_count != 6
            or len(self.fold_artifacts) != 6
            or not np.isfinite(self.total_elapsed_wall_seconds)
            or self.total_elapsed_wall_seconds <= 0.0
            or self.visible_cuda_device_count != 1
            or self.gpu_name != TOP2000_M03R_V7_2026_REQUIRED_GPU_NAME
            or not TOP2000_M03R_V7_2026_MINIMUM_GPU_MEMORY_BYTES
            <= self.gpu_total_memory_bytes
            <= TOP2000_M03R_V7_2026_MAXIMUM_GPU_MEMORY_BYTES
            or self.compute_capability
            != TOP2000_M03R_V7_2026_REQUIRED_COMPUTE_CAPABILITY
            or any(
                row.visible_cuda_device_count != self.visible_cuda_device_count
                or row.gpu_name != self.gpu_name
                or row.gpu_total_memory_bytes != self.gpu_total_memory_bytes
                or row.compute_capability != self.compute_capability
                for row in self.fold_artifacts
            )
            or self.maximum_peak_allocated_bytes
            != max(row.peak_allocated_bytes for row in self.fold_artifacts)
            or self.maximum_peak_reserved_bytes
            != max(row.peak_reserved_bytes for row in self.fold_artifacts)
            or not self.one_resident_data_session
            or self.folds_pooled
            or self.panel_aggregation_performed
            or self.policy_training_authorized
            or self.training_artifacts_mutated
            or not self.development_only
            or self.dataset_reportable
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026SettingWorkerError(
                "setting completion is partial, pooled, or overclaims its evidence"
            )
        expected_elapsed = sum(row.elapsed_wall_seconds for row in self.fold_artifacts)
        if not np.isclose(
            self.total_elapsed_wall_seconds,
            expected_elapsed,
            rtol=0.0,
            atol=1e-12,
        ):
            raise Top2000M03RV72026SettingWorkerError(
                "setting elapsed-time inventory does not reconcile"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026LoadedFoldArtifactBinding:
    receipt: Top2000M03RV72026FoldArtifactBindingReceipt
    artifact: Top2000M03RV72026LoadedExecutionArtifact
    binding_file_sha256: str


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SettingCompletionBinding:
    completion_path: str
    completion_file_sha256: str
    completion_receipt_sha256: str
    receipt: Top2000M03RV72026SettingCompletionReceipt
    fold_artifacts: tuple[Top2000M03RV72026LoadedFoldArtifactBinding, ...] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_digest("completion_file_sha256", self.completion_file_sha256)
        _require_digest("completion_receipt_sha256", self.completion_receipt_sha256)
        if (
            not self.completion_path
            or self.completion_receipt_sha256 != self.receipt.receipt_sha256
            or tuple(row.receipt.training_fold_index for row in self.fold_artifacts)
            != TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER
        ):
            raise Top2000M03RV72026SettingWorkerError(
                "setting completion binding does not match its receipt"
            )


def resolve_top2000_m03r_v7_2026_setting_index(
    completion_index: int | None = None,
    *,
    setting_index_map: Sequence[int] = TOP2000_M03R_V7_2026_FULL_SETTING_INDEX_MAP,
    environment: Mapping[str, str] | None = None,
) -> Top2000M03RV72026SettingIndexResolution:
    """Resolve one local Kubernetes index through an explicit frozen tuple."""

    env = os.environ if environment is None else environment
    setting_map = _validated_setting_index_map(setting_index_map)
    raw = env.get("JOB_COMPLETION_INDEX")
    env_index: int | None = None
    if raw is not None:
        try:
            env_index = int(raw)
        except ValueError as exc:
            raise Top2000M03RV72026SettingWorkerError(
                "JOB_COMPLETION_INDEX must be an integer"
            ) from exc
    if completion_index is not None and (
        isinstance(completion_index, bool) or not isinstance(completion_index, int)
    ):
        raise Top2000M03RV72026SettingWorkerError(
            "completion index must be an integer"
        )
    if completion_index is not None and env_index is not None and (
        completion_index != env_index
    ):
        raise Top2000M03RV72026SettingWorkerError(
            "explicit completion index disagrees with JOB_COMPLETION_INDEX"
        )
    result = env_index if completion_index is None else completion_index
    if result is None or result not in range(len(setting_map)):
        raise Top2000M03RV72026SettingWorkerError(
            "JOB_COMPLETION_INDEX is outside the explicit setting-index map"
        )
    return Top2000M03RV72026SettingIndexResolution(
        local_completion_index=result,
        setting_index=setting_map[result],
        setting_index_map=setting_map,
        setting_index_map_sha256=_setting_index_map_sha256(setting_map),
    )


def _setting_checkpoints(
    plan: Top2000M03RV7Seed172026YTDFrozenPlan,
    setting_index: int,
) -> dict[int, Top2000M03RV7Seed172026YTDCheckpointBinding]:
    result = {
        row.training_fold_index: row
        for row in plan.checkpoints
        if row.setting_index == setting_index
    }
    if set(result) != set(range(6)) or len(result) != 6:
        raise Top2000M03RV72026SettingWorkerError(
            "frozen plan does not provide exactly six checkpoints for the setting"
        )
    return result


def _artifact_paths(output_root: Path, setting_index: int, fold_index: int) -> tuple[Path, Path]:
    setting_root = output_root / f"setting-{setting_index:02d}"
    return (
        setting_root / f"fold-{fold_index:02d}.execution.json",
        setting_root / f"fold-{fold_index:02d}.artifact-binding.json",
    )


def _fold_receipt_from_loaded(
    *,
    index_resolution: Top2000M03RV72026SettingIndexResolution,
    checkpoint: Top2000M03RV7Seed172026YTDCheckpointBinding,
    artifact_path: Path,
    artifact_file_sha256: str,
    artifact: Top2000M03RV72026LoadedExecutionArtifact,
    frozen_plan_file_sha256: str,
    frozen_plan_receipt_sha256: str,
    execution_source_inventory_sha256: str,
    pre2026_cache_file_sha256: str,
    retrospective_cache_file_sha256: str,
) -> Top2000M03RV72026FoldArtifactBindingReceipt:
    execution = artifact.execution_receipt
    cuda = artifact.cuda_proof
    return Top2000M03RV72026FoldArtifactBindingReceipt(
        local_completion_index=index_resolution.local_completion_index,
        setting_index_map=index_resolution.setting_index_map,
        setting_index_map_sha256=index_resolution.setting_index_map_sha256,
        training_completion_index=checkpoint.completion_index,
        setting_index=checkpoint.setting_index,
        setting_id=checkpoint.setting_id,
        runtime_setting_id=checkpoint.runtime_setting_id,
        training_fold_index=checkpoint.training_fold_index,
        checkpoint_role=checkpoint.checkpoint_role,
        frozen_plan_file_sha256=frozen_plan_file_sha256,
        frozen_plan_receipt_sha256=frozen_plan_receipt_sha256,
        execution_source_inventory_sha256=execution_source_inventory_sha256,
        pre2026_cache_file_sha256=pre2026_cache_file_sha256,
        retrospective_cache_file_sha256=retrospective_cache_file_sha256,
        retrospective_chronology_receipt_sha256=(
            artifact.chronology_identity.receipt_sha256
        ),
        frozen_checkpoint_binding_sha256=checkpoint.receipt_sha256,
        checkpoint_model_file_sha256=checkpoint.model_file_sha256,
        checkpoint_model_state_sha256=checkpoint.model_state_sha256,
        policy_model_state_sha256_before=(
            execution.policy_model_state_sha256_before
        ),
        policy_model_state_sha256_after=execution.policy_model_state_sha256_after,
        artifact_path=str(artifact_path),
        artifact_file_sha256=artifact_file_sha256,
        execution_receipt_sha256=execution.receipt_sha256,
        elapsed_wall_seconds=execution.elapsed_wall_seconds,
        visible_cuda_device_count=cuda.visible_cuda_device_count,
        gpu_name=cuda.gpu_name,
        gpu_total_memory_bytes=cuda.gpu_total_memory_bytes,
        compute_capability=cuda.compute_capability,
        peak_allocated_bytes=cuda.peak_allocated_bytes,
        peak_reserved_bytes=cuda.peak_reserved_bytes,
        allocator_oom_count_delta=cuda.allocator_oom_count_delta,
        allocator_retry_count_delta=cuda.allocator_retry_count_delta,
    )


def load_top2000_m03r_v7_seed17_2026_fold_artifact_binding(
    path: str | Path,
    *,
    expected_index_resolution: Top2000M03RV72026SettingIndexResolution,
    checkpoint: Top2000M03RV7Seed172026YTDCheckpointBinding,
    expected_artifact_path: str | Path,
    expected_frozen_plan_file_sha256: str,
    expected_frozen_plan_receipt_sha256: str,
    expected_execution_source_inventory_sha256: str,
    expected_pre2026_cache_file_sha256: str,
    expected_retrospective_cache_file_sha256: str,
) -> Top2000M03RV72026LoadedFoldArtifactBinding:
    """Revalidate one binding, artifact, and complete child receipt graph."""

    payload, binding_file_sha256 = _read_canonical_json(
        Path(path), label="fold artifact binding"
    )
    try:
        typed = _typed_payload(
            payload,
            Top2000M03RV72026FoldArtifactBindingReceipt,
            label="fold artifact binding",
        )
        typed["setting_index_map"] = tuple(typed["setting_index_map"])
        typed["compute_capability"] = tuple(typed["compute_capability"])
        receipt = Top2000M03RV72026FoldArtifactBindingReceipt(
            **typed
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, Top2000M03RV72026SettingWorkerError):
            raise
        raise Top2000M03RV72026SettingWorkerError(
            "fold artifact binding cannot be reconstructed"
        ) from exc
    expected_artifact = _resolved(expected_artifact_path)
    if (
        receipt.local_completion_index
        != expected_index_resolution.local_completion_index
        or receipt.setting_index_map != expected_index_resolution.setting_index_map
        or receipt.setting_index_map_sha256
        != expected_index_resolution.setting_index_map_sha256
        or receipt.setting_index != expected_index_resolution.setting_index
        or receipt.training_completion_index != checkpoint.completion_index
        or receipt.setting_index != checkpoint.setting_index
        or receipt.setting_id != checkpoint.setting_id
        or receipt.runtime_setting_id != checkpoint.runtime_setting_id
        or receipt.training_fold_index != checkpoint.training_fold_index
        or receipt.checkpoint_role != checkpoint.checkpoint_role
        or receipt.frozen_plan_file_sha256
        != expected_frozen_plan_file_sha256
        or receipt.frozen_plan_receipt_sha256
        != expected_frozen_plan_receipt_sha256
        or receipt.execution_source_inventory_sha256
        != expected_execution_source_inventory_sha256
        or receipt.pre2026_cache_file_sha256
        != expected_pre2026_cache_file_sha256
        or receipt.retrospective_cache_file_sha256
        != expected_retrospective_cache_file_sha256
        or receipt.frozen_checkpoint_binding_sha256
        != checkpoint.receipt_sha256
        or receipt.checkpoint_model_file_sha256 != checkpoint.model_file_sha256
        or receipt.checkpoint_model_state_sha256 != checkpoint.model_state_sha256
        or _resolved(receipt.artifact_path) != expected_artifact
    ):
        raise Top2000M03RV72026SettingWorkerError(
            "fold artifact binding does not match the frozen worker inputs"
        )
    artifact = load_top2000_m03r_v7_seed17_2026_execution_artifact(
        expected_artifact,
        expected_file_sha256=receipt.artifact_file_sha256,
        expected_evaluation_plan_receipt_sha256=(
            expected_frozen_plan_receipt_sha256
        ),
        expected_execution_source_inventory_sha256=(
            expected_execution_source_inventory_sha256
        ),
    )
    execution = artifact.execution_receipt
    checkpoint_load = artifact.checkpoint_load_receipt
    cuda = artifact.cuda_proof
    if (
        receipt.execution_receipt_sha256 != execution.receipt_sha256
        or receipt.retrospective_chronology_receipt_sha256
        != artifact.chronology_identity.receipt_sha256
        or artifact.economic_execution_receipt.pre2026_cache_sha256
        != expected_pre2026_cache_file_sha256
        or checkpoint_load.frozen_checkpoint_binding_sha256
        != checkpoint.receipt_sha256
        or checkpoint_load.model_file_sha256 != checkpoint.model_file_sha256
        or checkpoint_load.model_state_sha256 != checkpoint.model_state_sha256
        or execution.policy_model_state_sha256_before
        != checkpoint.model_state_sha256
        or execution.policy_model_state_sha256_after
        != checkpoint.model_state_sha256
        or execution.elapsed_wall_seconds != receipt.elapsed_wall_seconds
        or cuda.visible_cuda_device_count != receipt.visible_cuda_device_count
        or cuda.gpu_name != receipt.gpu_name
        or cuda.gpu_total_memory_bytes != receipt.gpu_total_memory_bytes
        or cuda.compute_capability != receipt.compute_capability
        or cuda.peak_allocated_bytes != receipt.peak_allocated_bytes
        or cuda.peak_reserved_bytes != receipt.peak_reserved_bytes
        or cuda.allocator_oom_count_delta != receipt.allocator_oom_count_delta
        or cuda.allocator_retry_count_delta != receipt.allocator_retry_count_delta
    ):
        raise Top2000M03RV72026SettingWorkerError(
            "fold artifact receipt graph does not close against its checkpoint"
        )
    return Top2000M03RV72026LoadedFoldArtifactBinding(
        receipt=receipt,
        artifact=artifact,
        binding_file_sha256=binding_file_sha256,
    )


def _fold_completion(
    path: Path,
    loaded: Top2000M03RV72026LoadedFoldArtifactBinding,
) -> Top2000M03RV72026SettingFoldCompletion:
    receipt = loaded.receipt
    return Top2000M03RV72026SettingFoldCompletion(
        training_fold_index=receipt.training_fold_index,
        artifact_binding_path=str(path),
        artifact_binding_file_sha256=loaded.binding_file_sha256,
        artifact_binding_receipt_sha256=receipt.receipt_sha256,
        artifact_path=receipt.artifact_path,
        artifact_file_sha256=receipt.artifact_file_sha256,
        execution_receipt_sha256=receipt.execution_receipt_sha256,
        elapsed_wall_seconds=receipt.elapsed_wall_seconds,
        visible_cuda_device_count=receipt.visible_cuda_device_count,
        gpu_name=receipt.gpu_name,
        gpu_total_memory_bytes=receipt.gpu_total_memory_bytes,
        compute_capability=receipt.compute_capability,
        peak_allocated_bytes=receipt.peak_allocated_bytes,
        peak_reserved_bytes=receipt.peak_reserved_bytes,
    )


def _load_setting_completion(
    path: Path,
    *,
    expected_file_sha256: str | None,
    plan: Top2000M03RV7Seed172026YTDFrozenPlan,
    plan_path: Path,
    plan_file_sha256: str,
    plan_receipt_sha256: str,
    source_inventory_sha256: str,
    retrospective_cache_file_sha256: str,
    output_root: Path,
    index_resolution: Top2000M03RV72026SettingIndexResolution,
) -> Top2000M03RV72026SettingCompletionBinding:
    payload, file_sha256 = _read_canonical_json(
        path, label="setting completion receipt"
    )
    if expected_file_sha256 is not None and (
        file_sha256 != _require_digest(
            "expected setting completion file SHA-256", expected_file_sha256
        )
    ):
        raise Top2000M03RV72026SettingWorkerError(
            "setting completion file SHA-256 drifted"
        )
    typed = _typed_payload(
        payload,
        Top2000M03RV72026SettingCompletionReceipt,
        label="setting completion receipt",
    )
    fold_payloads = typed.get("fold_artifacts")
    if not isinstance(fold_payloads, list):
        raise Top2000M03RV72026SettingWorkerError(
            "setting completion fold inventory is malformed"
        )
    try:
        typed["fold_execution_order"] = tuple(typed["fold_execution_order"])
        typed["setting_index_map"] = tuple(typed["setting_index_map"])
        typed["compute_capability"] = tuple(typed["compute_capability"])
        typed["fold_artifacts"] = tuple(
            Top2000M03RV72026SettingFoldCompletion(
                **{**row, "compute_capability": tuple(row["compute_capability"])}
            )
            for row in fold_payloads
        )
        receipt = Top2000M03RV72026SettingCompletionReceipt(**typed)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, Top2000M03RV72026SettingWorkerError):
            raise
        raise Top2000M03RV72026SettingWorkerError(
            "setting completion receipt cannot be reconstructed"
        ) from exc
    setting_index = index_resolution.setting_index
    checkpoints = _setting_checkpoints(plan, setting_index)
    first = checkpoints[0]
    if (
        receipt.local_completion_index != index_resolution.local_completion_index
        or receipt.setting_index_map != index_resolution.setting_index_map
        or receipt.setting_index_map_sha256
        != index_resolution.setting_index_map_sha256
        or receipt.setting_index != setting_index
        or receipt.setting_id != first.setting_id
        or receipt.runtime_setting_id != first.runtime_setting_id
        or _resolved(receipt.frozen_plan_path) != plan_path
        or receipt.frozen_plan_file_sha256 != plan_file_sha256
        or receipt.frozen_plan_receipt_sha256 != plan_receipt_sha256
        or receipt.execution_source_inventory_sha256 != source_inventory_sha256
        or receipt.pre2026_cache_file_sha256 != plan.pre2026_cache.cache_file_sha256
        or receipt.retrospective_cache_file_sha256
        != retrospective_cache_file_sha256
    ):
        raise Top2000M03RV72026SettingWorkerError(
            "setting completion does not match the exact frozen inputs"
        )
    validated: list[Top2000M03RV72026SettingFoldCompletion] = []
    loaded_children: list[Top2000M03RV72026LoadedFoldArtifactBinding] = []
    chronology: str | None = None
    for row in receipt.fold_artifacts:
        artifact_path, binding_path = _artifact_paths(
            output_root, setting_index, row.training_fold_index
        )
        if (
            _resolved(row.artifact_binding_path) != binding_path
            or _resolved(row.artifact_path) != artifact_path
        ):
            raise Top2000M03RV72026SettingWorkerError(
                "setting completion child path drifted"
            )
        loaded = load_top2000_m03r_v7_seed17_2026_fold_artifact_binding(
            binding_path,
            expected_index_resolution=index_resolution,
            checkpoint=checkpoints[row.training_fold_index],
            expected_artifact_path=artifact_path,
            expected_frozen_plan_file_sha256=plan_file_sha256,
            expected_frozen_plan_receipt_sha256=plan_receipt_sha256,
            expected_execution_source_inventory_sha256=source_inventory_sha256,
            expected_pre2026_cache_file_sha256=(
                plan.pre2026_cache.cache_file_sha256
            ),
            expected_retrospective_cache_file_sha256=(
                retrospective_cache_file_sha256
            ),
        )
        observed = _fold_completion(binding_path, loaded)
        if observed != row:
            raise Top2000M03RV72026SettingWorkerError(
                "setting completion child inventory does not replay exactly"
            )
        child_chronology = loaded.receipt.retrospective_chronology_receipt_sha256
        chronology = child_chronology if chronology is None else chronology
        if child_chronology != chronology:
            raise Top2000M03RV72026SettingWorkerError(
                "setting folds do not share one retrospective chronology"
            )
        validated.append(observed)
        loaded_children.append(loaded)
    if (
        tuple(validated) != receipt.fold_artifacts
        or chronology != receipt.retrospective_chronology_receipt_sha256
    ):
        raise Top2000M03RV72026SettingWorkerError(
            "setting completion receipt graph is incomplete"
        )
    return Top2000M03RV72026SettingCompletionBinding(
        completion_path=str(path),
        completion_file_sha256=file_sha256,
        completion_receipt_sha256=receipt.receipt_sha256,
        receipt=receipt,
        fold_artifacts=tuple(loaded_children),
    )


def load_top2000_m03r_v7_seed17_2026_setting_completion(
    path: str | Path,
    *,
    expected_completion_file_sha256: str | None,
    plan_path: str | Path,
    expected_plan_file_sha256: str,
    expected_plan_receipt_sha256: str,
    expected_execution_source_inventory_sha256: str,
    expected_retrospective_cache_file_sha256: str,
    output_root: str | Path,
    completion_index: int | None = None,
    setting_index_map: Sequence[int] = TOP2000_M03R_V7_2026_FULL_SETTING_INDEX_MAP,
    environment: Mapping[str, str] | None = None,
) -> Top2000M03RV72026SettingCompletionBinding:
    """Public fail-closed loader for one complete six-fold setting graph."""

    index_resolution = resolve_top2000_m03r_v7_2026_setting_index(
        completion_index,
        setting_index_map=setting_index_map,
        environment=environment,
    )
    plan_file = _resolved(plan_path)
    plan = load_top2000_m03r_v7_seed17_2026_ytd_plan(
        plan_file,
        expected_file_sha256=expected_plan_file_sha256,
        expected_receipt_sha256=expected_plan_receipt_sha256,
    )
    source_sha256 = _require_digest(
        "expected execution source inventory SHA-256",
        expected_execution_source_inventory_sha256,
    )
    if source_sha256 != plan.evaluation_source_sha256:
        raise Top2000M03RV72026SettingWorkerError(
            "execution source inventory does not match the frozen plan"
        )
    root = _resolved(output_root)
    return _load_setting_completion(
        _resolved(path),
        expected_file_sha256=expected_completion_file_sha256,
        plan=plan,
        plan_path=plan_file,
        plan_file_sha256=expected_plan_file_sha256,
        plan_receipt_sha256=expected_plan_receipt_sha256,
        source_inventory_sha256=source_sha256,
        retrospective_cache_file_sha256=_require_digest(
            "expected retrospective cache file SHA-256",
            expected_retrospective_cache_file_sha256,
        ),
        output_root=root,
        index_resolution=index_resolution,
    )


def _prepare_session(
    plan: Top2000M03RV7Seed172026YTDFrozenPlan,
    *,
    retrospective_cache_path: Path,
    retrospective_cache_file_sha256: str,
    plan_receipt_sha256: str,
    source_inventory_sha256: str,
) -> Top2000M03RV72026ExecutionSession:
    return prepare_top2000_m03r_v7_seed17_2026_execution_session(
        pre2026_cache_path=plan.pre2026_cache.cache_path,
        expected_pre2026_cache_sha256=plan.pre2026_cache.cache_file_sha256,
        retrospective_cache_path=retrospective_cache_path,
        expected_retrospective_cache_sha256=retrospective_cache_file_sha256,
        evaluation_plan_receipt_sha256=plan_receipt_sha256,
        execution_source_inventory_sha256=source_inventory_sha256,
        device="cuda:0",
    )


def run_top2000_m03r_v7_seed17_2026_setting_worker(
    *,
    plan_path: str | Path,
    expected_plan_file_sha256: str,
    expected_plan_receipt_sha256: str,
    expected_execution_source_inventory_sha256: str,
    retrospective_cache_path: str | Path,
    expected_retrospective_cache_file_sha256: str,
    output_root: str | Path,
    completion_index: int | None = None,
    setting_index_map: Sequence[int] = TOP2000_M03R_V7_2026_FULL_SETTING_INDEX_MAP,
    environment: Mapping[str, str] | None = None,
) -> Top2000M03RV72026SettingCompletionBinding:
    """Execute or exactly resume one setting using one resident CUDA session."""

    index_resolution = resolve_top2000_m03r_v7_2026_setting_index(
        completion_index,
        setting_index_map=setting_index_map,
        environment=environment,
    )
    setting_index = index_resolution.setting_index
    plan_file = _resolved(plan_path)
    plan = load_top2000_m03r_v7_seed17_2026_ytd_plan(
        plan_file,
        expected_file_sha256=expected_plan_file_sha256,
        expected_receipt_sha256=expected_plan_receipt_sha256,
    )
    source_inventory_sha256 = _require_digest(
        "expected execution source inventory SHA-256",
        expected_execution_source_inventory_sha256,
    )
    retrospective_cache_file_sha256 = _require_digest(
        "expected retrospective cache file SHA-256",
        expected_retrospective_cache_file_sha256,
    )
    if source_inventory_sha256 != plan.evaluation_source_sha256:
        raise Top2000M03RV72026SettingWorkerError(
            "execution source inventory does not match the frozen plan"
        )
    retrospective_cache = _resolved(retrospective_cache_path)
    root = _resolved(output_root)
    training_root = _resolved(plan.source_training_output_root)
    source_root = _resolved(plan.evaluation_source.source_root)
    if root == training_root or root.is_relative_to(training_root):
        raise Top2000M03RV72026SettingWorkerError(
            "evaluation output root may not mutate the completed training root"
        )
    if root == source_root or root.is_relative_to(source_root):
        raise Top2000M03RV72026SettingWorkerError(
            "evaluation output root may not mutate the frozen source root"
        )
    checkpoints = _setting_checkpoints(plan, setting_index)
    completion_path = root / f"setting-{setting_index:02d}" / "setting-completion.json"
    if completion_path.exists() or completion_path.is_symlink():
        return _load_setting_completion(
            completion_path,
            expected_file_sha256=None,
            plan=plan,
            plan_path=plan_file,
            plan_file_sha256=expected_plan_file_sha256,
            plan_receipt_sha256=expected_plan_receipt_sha256,
            source_inventory_sha256=source_inventory_sha256,
            retrospective_cache_file_sha256=retrospective_cache_file_sha256,
            output_root=root,
            index_resolution=index_resolution,
        )

    completed: dict[int, Top2000M03RV72026SettingFoldCompletion] = {}
    loaded_by_fold: dict[int, Top2000M03RV72026LoadedFoldArtifactBinding] = {}
    missing: list[int] = []
    for fold_index in TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER:
        artifact_path, binding_path = _artifact_paths(root, setting_index, fold_index)
        artifact_exists = artifact_path.exists() or artifact_path.is_symlink()
        binding_exists = binding_path.exists() or binding_path.is_symlink()
        if artifact_exists != binding_exists:
            raise Top2000M03RV72026SettingWorkerError(
                f"fold {fold_index} has a partial artifact/binding pair"
            )
        if not artifact_exists:
            missing.append(fold_index)
            continue
        loaded = load_top2000_m03r_v7_seed17_2026_fold_artifact_binding(
            binding_path,
            expected_index_resolution=index_resolution,
            checkpoint=checkpoints[fold_index],
            expected_artifact_path=artifact_path,
            expected_frozen_plan_file_sha256=expected_plan_file_sha256,
            expected_frozen_plan_receipt_sha256=expected_plan_receipt_sha256,
            expected_execution_source_inventory_sha256=source_inventory_sha256,
            expected_pre2026_cache_file_sha256=plan.pre2026_cache.cache_file_sha256,
            expected_retrospective_cache_file_sha256=(
                retrospective_cache_file_sha256
            ),
        )
        loaded_by_fold[fold_index] = loaded
        completed[fold_index] = _fold_completion(binding_path, loaded)

    session: Top2000M03RV72026ExecutionSession | None = None
    if missing:
        if not retrospective_cache.is_file() or retrospective_cache.is_symlink():
            raise Top2000M03RV72026SettingWorkerError(
                "retrospective cache must be a regular non-symlink file"
            )
        session = _prepare_session(
            plan,
            retrospective_cache_path=retrospective_cache,
            retrospective_cache_file_sha256=retrospective_cache_file_sha256,
            plan_receipt_sha256=expected_plan_receipt_sha256,
            source_inventory_sha256=source_inventory_sha256,
        )
    for fold_index in TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER:
        if fold_index not in missing:
            continue
        if session is None:  # pragma: no cover - guarded by the missing branch
            raise AssertionError("missing fold execution has no resident session")
        artifact_path, binding_path = _artifact_paths(root, setting_index, fold_index)
        artifact_binding: Top2000M03RV72026ExecutionArtifactBinding = (
            run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session(
                session,
                checkpoints[fold_index],
                training_output_root=training_root,
                output_path=artifact_path,
            )
        )
        artifact = load_top2000_m03r_v7_seed17_2026_execution_artifact(
            artifact_path,
            expected_file_sha256=artifact_binding.artifact_file_sha256,
            expected_evaluation_plan_receipt_sha256=(
                expected_plan_receipt_sha256
            ),
            expected_execution_source_inventory_sha256=source_inventory_sha256,
        )
        fold_receipt = _fold_receipt_from_loaded(
            index_resolution=index_resolution,
            checkpoint=checkpoints[fold_index],
            artifact_path=artifact_path,
            artifact_file_sha256=artifact_binding.artifact_file_sha256,
            artifact=artifact,
            frozen_plan_file_sha256=expected_plan_file_sha256,
            frozen_plan_receipt_sha256=expected_plan_receipt_sha256,
            execution_source_inventory_sha256=source_inventory_sha256,
            pre2026_cache_file_sha256=plan.pre2026_cache.cache_file_sha256,
            retrospective_cache_file_sha256=retrospective_cache_file_sha256,
        )
        binding_file_sha256 = _publish_immutable_json(
            binding_path, asdict(fold_receipt)
        )
        loaded = Top2000M03RV72026LoadedFoldArtifactBinding(
            receipt=fold_receipt,
            artifact=artifact,
            binding_file_sha256=binding_file_sha256,
        )
        loaded_by_fold[fold_index] = loaded
        completed[fold_index] = _fold_completion(binding_path, loaded)

    ordered = tuple(completed[index] for index in TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER)
    chronologies = {
        loaded_by_fold[row.training_fold_index]
        .receipt.retrospective_chronology_receipt_sha256
        for row in ordered
    }
    if len(chronologies) != 1:
        raise Top2000M03RV72026SettingWorkerError(
            "setting artifacts do not share one retrospective chronology"
        )
    first_checkpoint = checkpoints[0]
    completion = Top2000M03RV72026SettingCompletionReceipt(
        local_completion_index=index_resolution.local_completion_index,
        setting_index_map=index_resolution.setting_index_map,
        setting_index_map_sha256=index_resolution.setting_index_map_sha256,
        setting_index=setting_index,
        setting_id=first_checkpoint.setting_id,
        runtime_setting_id=first_checkpoint.runtime_setting_id,
        frozen_plan_path=str(plan_file),
        frozen_plan_file_sha256=expected_plan_file_sha256,
        frozen_plan_receipt_sha256=expected_plan_receipt_sha256,
        execution_source_inventory_sha256=source_inventory_sha256,
        pre2026_cache_file_sha256=plan.pre2026_cache.cache_file_sha256,
        retrospective_cache_file_sha256=retrospective_cache_file_sha256,
        retrospective_chronology_receipt_sha256=next(iter(chronologies)),
        fold_execution_order=TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER,
        fold_artifacts=ordered,
        completed_fold_count=6,
        total_elapsed_wall_seconds=sum(row.elapsed_wall_seconds for row in ordered),
        visible_cuda_device_count=ordered[0].visible_cuda_device_count,
        gpu_name=ordered[0].gpu_name,
        gpu_total_memory_bytes=ordered[0].gpu_total_memory_bytes,
        compute_capability=ordered[0].compute_capability,
        maximum_peak_allocated_bytes=max(
            row.peak_allocated_bytes for row in ordered
        ),
        maximum_peak_reserved_bytes=max(
            row.peak_reserved_bytes for row in ordered
        ),
    )
    completion_file_sha256 = _publish_immutable_json(
        completion_path, asdict(completion)
    )
    return Top2000M03RV72026SettingCompletionBinding(
        completion_path=str(completion_path),
        completion_file_sha256=completion_file_sha256,
        completion_receipt_sha256=completion.receipt_sha256,
        receipt=completion,
        fold_artifacts=tuple(
            loaded_by_fold[index]
            for index in TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--plan-receipt-sha256", required=True)
    parser.add_argument("--execution-source-inventory-sha256", required=True)
    parser.add_argument("--retrospective-cache", required=True)
    parser.add_argument("--retrospective-cache-file-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--completion-index", type=int)
    parser.add_argument(
        "--setting-index-map",
        type=_parse_setting_index_map,
        default=TOP2000_M03R_V7_2026_FULL_SETTING_INDEX_MAP,
        help=(
            "explicit local-to-scientific tuple: 0; 1,2,...,11; or 0,1,...,11"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run_top2000_m03r_v7_seed17_2026_setting_worker(
            plan_path=args.plan,
            expected_plan_file_sha256=args.plan_file_sha256,
            expected_plan_receipt_sha256=args.plan_receipt_sha256,
            expected_execution_source_inventory_sha256=(
                args.execution_source_inventory_sha256
            ),
            retrospective_cache_path=args.retrospective_cache,
            expected_retrospective_cache_file_sha256=(
                args.retrospective_cache_file_sha256
            ),
            output_root=args.output_root,
            completion_index=args.completion_index,
            setting_index_map=args.setting_index_map,
        )
    except (
        OSError,
        ValueError,
        Top2000M03RV72026ExecutionError,
        Top2000M03RV72026SettingWorkerError,
        Top2000M03RV7Seed172026YTDWorkflowError,
    ) as exc:
        print(f"TOP2000 M03R-v7 2026 setting worker failed: {exc}", file=sys.stderr)
        return 2
    print(
        _canonical_json(
            {
                "completion_path": result.completion_path,
                "completion_file_sha256": result.completion_file_sha256,
                "completion_receipt_sha256": result.completion_receipt_sha256,
                "setting_index": result.receipt.setting_index,
                "local_completion_index": (
                    result.receipt.local_completion_index
                ),
                "setting_index_map": list(result.receipt.setting_index_map),
                "setting_index_map_sha256": (
                    result.receipt.setting_index_map_sha256
                ),
                "completed_fold_count": result.receipt.completed_fold_count,
                "fold_execution_order": list(result.receipt.fold_execution_order),
                "total_elapsed_wall_seconds": (
                    result.receipt.total_elapsed_wall_seconds
                ),
                "maximum_peak_allocated_bytes": (
                    result.receipt.maximum_peak_allocated_bytes
                ),
                "maximum_peak_reserved_bytes": (
                    result.receipt.maximum_peak_reserved_bytes
                ),
                "visible_cuda_device_count": (
                    result.receipt.visible_cuda_device_count
                ),
                "gpu_name": result.receipt.gpu_name,
                "gpu_total_memory_bytes": (
                    result.receipt.gpu_total_memory_bytes
                ),
                "compute_capability": list(result.receipt.compute_capability),
                "development_only": True,
                "dataset_reportable": False,
                "scientific_reporting_eligible": False,
                "promotion_eligible": False,
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TOP2000_M03R_V7_2026_ALLOWED_SETTING_INDEX_MAPS",
    "TOP2000_M03R_V7_2026_FOLD_ARTIFACT_BINDING_SCHEMA",
    "TOP2000_M03R_V7_2026_FULL_SETTING_INDEX_MAP",
    "TOP2000_M03R_V7_2026_REMAINDER_SETTING_INDEX_MAP",
    "TOP2000_M03R_V7_2026_SENTINEL_SETTING_INDEX_MAP",
    "TOP2000_M03R_V7_2026_SETTING_COMPLETION_SCHEMA",
    "TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER",
    "TOP2000_M03R_V7_2026_SETTING_INDEX_MAP_SCHEMA",
    "Top2000M03RV72026FoldArtifactBindingReceipt",
    "Top2000M03RV72026LoadedFoldArtifactBinding",
    "Top2000M03RV72026SettingCompletionBinding",
    "Top2000M03RV72026SettingCompletionReceipt",
    "Top2000M03RV72026SettingFoldCompletion",
    "Top2000M03RV72026SettingIndexResolution",
    "Top2000M03RV72026SettingWorkerError",
    "load_top2000_m03r_v7_seed17_2026_fold_artifact_binding",
    "load_top2000_m03r_v7_seed17_2026_setting_completion",
    "main",
    "resolve_top2000_m03r_v7_2026_setting_index",
    "run_top2000_m03r_v7_seed17_2026_setting_worker",
]
