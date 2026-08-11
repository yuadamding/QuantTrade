"""Single-GPU, inference-only execution of one frozen 2026 checkpoint.

This boundary loads one immutable seed-17 checkpoint, fits only the frozen
pre-score execution controls, computes encoder states from the full causal
history, and runs one leakage-safe economic suffix exactly once.  It emits a
canonical-JSON artifact containing only evaluator-ready arrays and replayable
receipts.  The artifact is development-only, nonreportable, and nonpromotable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from rl_quant.envs.hold30 import TURNOVER_CAUSES
from rl_quant.evaluation.top2000_m03r_v7_2026 import (
    Top2000M03RV72026Telemetry,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_checkpoint import (
    Top2000M03RV72026CheckpointLoadReceipt,
    Top2000M03RV72026FrozenCheckpointBinding,
    load_top2000_m03r_v7_seed17_2026_checkpoint,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_cohort_survival import (
    Top2000M03RV72026CohortTrajectories,
    Top2000M03RV72026CohortTrajectoryReceipt,
    build_top2000_m03r_v7_2026_cohort_trajectories,
    validate_top2000_m03r_v7_2026_cohort_trajectories,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_execution_view import (
    Top2000M03RV72026EconomicExecutionReceipt,
    Top2000M03RV72026EconomicExecutionView,
    build_top2000_m03r_v7_2026_economic_execution_view,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_factor_calibration import (
    Top2000M03RV72026PreScoreFactorCalibration,
    fit_top2000_m03r_v7_2026_pre_score_factor_calibration,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    Top2000M03RV72026RetrospectiveData,
    Top2000M03RV72026RetrospectiveIdentity,
    Top2000M03RV72026RetrospectiveSourceEvidence,
    load_top2000_m03r_v7_2026_retrospective_cache,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_trace_telemetry import (
    Top2000M03RV72026TraceEvaluationInputs,
    Top2000M03RV72026TraceTelemetryReceipt,
    adapt_top2000_m03r_v7_2026_trace,
    validate_top2000_m03r_v7_2026_trace_evaluation_inputs,
)
from rl_quant.evaluation.top2000_m03r_v7_dev import (
    build_top2000_m03r_v7_validation_runtime,
    model_state_sha256,
)
from rl_quant.training.hold30 import Hold30ReplayGeometry
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    Top2000VerifiedDevelopmentCache,
    load_verified_top2000_hold30_development_cache,
)

TOP2000_M03R_V7_2026_EXECUTION_ARTIFACT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-single-checkpoint-artifact-v1"
)
TOP2000_M03R_V7_2026_EXECUTION_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-single-checkpoint-execution-v1"
)
TOP2000_M03R_V7_2026_CUDA_PROOF_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-single-cuda-proof-v1"
)
TOP2000_M03R_V7_2026_REQUIRED_GPU_NAME = "NVIDIA H100 80GB HBM3"
TOP2000_M03R_V7_2026_REQUIRED_COMPUTE_CAPABILITY = (9, 0)
TOP2000_M03R_V7_2026_MINIMUM_GPU_MEMORY_BYTES = 79 * 1024**3
TOP2000_M03R_V7_2026_MAXIMUM_GPU_MEMORY_BYTES = 81 * 1024**3
TOP2000_M03R_V7_2026_EXECUTION_GEOMETRY = Hold30ReplayGeometry(
    warmup_decisions=63,
    credit_returns=30,
    support_decisions=30,
    label_support_decisions=63,
    max_origin_batch=1,
)
_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024


class Top2000M03RV72026ExecutionError(RuntimeError):
    """A single-checkpoint inference or artifact boundary failed closed."""


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
        raise Top2000M03RV72026ExecutionError(
            "execution artifact is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026ExecutionError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _CudaStartup:
    device: torch.device
    cuda_visible_device: str | None
    gpu_name: str
    gpu_total_memory_bytes: int
    compute_capability: tuple[int, int]
    startup_allocated_bytes: int
    startup_reserved_bytes: int
    startup_free_bytes: int
    startup_allocator_oom_count: int
    startup_allocator_retry_count: int


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026CudaExecutionProof:
    """Exact one-process/one-visible-CUDA startup and inference peak proof."""

    device: str
    cuda_visible_device: str | None
    visible_cuda_device_count: int
    gpu_name: str
    gpu_total_memory_bytes: int
    compute_capability: tuple[int, int]
    startup_allocated_bytes: int
    startup_reserved_bytes: int
    startup_free_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    final_allocated_bytes: int
    final_reserved_bytes: int
    final_free_bytes: int
    allocator_oom_count_delta: int
    allocator_retry_count_delta: int
    completed_transition_rows: int
    canonical_pass_count: int = 1
    optimizer_step_count: int = 0
    gradient_enabled_during_pass: bool = False
    distributed_initialized: bool = False
    process_world_size: int = 1
    process_rank: int = 0
    local_world_size: int = 1
    local_rank: int = 0
    python_version: str = platform.python_version()
    torch_version: str = torch.__version__
    torch_cuda_version: str | None = torch.version.cuda
    development_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_CUDA_PROOF_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != TOP2000_M03R_V7_2026_CUDA_PROOF_SCHEMA
            or self.device != "cuda:0"
            or self.visible_cuda_device_count != 1
            or self.gpu_name != TOP2000_M03R_V7_2026_REQUIRED_GPU_NAME
            or not TOP2000_M03R_V7_2026_MINIMUM_GPU_MEMORY_BYTES
            <= self.gpu_total_memory_bytes
            <= TOP2000_M03R_V7_2026_MAXIMUM_GPU_MEMORY_BYTES
            or self.compute_capability
            != TOP2000_M03R_V7_2026_REQUIRED_COMPUTE_CAPABILITY
            or any(
                value < 0
                for value in (
                    self.startup_allocated_bytes,
                    self.startup_reserved_bytes,
                    self.startup_free_bytes,
                    self.peak_allocated_bytes,
                    self.peak_reserved_bytes,
                    self.final_allocated_bytes,
                    self.final_reserved_bytes,
                    self.final_free_bytes,
                )
            )
            or self.peak_allocated_bytes < self.startup_allocated_bytes
            or self.peak_reserved_bytes < self.startup_reserved_bytes
            or self.peak_reserved_bytes > self.gpu_total_memory_bytes
            or self.allocator_oom_count_delta != 0
            or self.allocator_retry_count_delta != 0
            or self.completed_transition_rows <= 0
            or self.canonical_pass_count != 1
            or self.optimizer_step_count != 0
            or self.gradient_enabled_during_pass
            or self.distributed_initialized
            or self.process_world_size != 1
            or self.process_rank != 0
            or self.local_world_size != 1
            or self.local_rank != 0
            or not self.python_version
            or not self.torch_version
            or not self.torch_cuda_version
            or not self.development_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026ExecutionError(
                "CUDA proof does not establish one clean inference process/GPU"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


def _env_int(name: str, expected: int) -> None:
    raw = os.environ.get(name)
    if raw is None:
        return
    try:
        value = int(raw)
    except ValueError as exc:
        raise Top2000M03RV72026ExecutionError(
            f"{name} must be an integer when set"
        ) from exc
    if value != expected:
        raise Top2000M03RV72026ExecutionError(
            f"{name} must equal {expected} for single-process inference"
        )


def _start_single_cuda(device: str | torch.device) -> _CudaStartup:
    requested = torch.device(device)
    if requested.type != "cuda" or requested.index not in (None, 0):
        raise Top2000M03RV72026ExecutionError(
            "2026 checkpoint execution requires the sole visible device cuda:0"
        )
    if torch.distributed.is_initialized():
        raise Top2000M03RV72026ExecutionError(
            "single-checkpoint inference may not initialize torch.distributed"
        )
    for name, expected in (
        ("WORLD_SIZE", 1),
        ("RANK", 0),
        ("LOCAL_WORLD_SIZE", 1),
        ("LOCAL_RANK", 0),
    ):
        _env_int(name, expected)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and (
        not visible.strip() or len(visible.split(",")) != 1
    ):
        raise Top2000M03RV72026ExecutionError(
            "CUDA_VISIBLE_DEVICES must expose exactly one device"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Top2000M03RV72026ExecutionError(
            "single-checkpoint inference requires exactly one visible CUDA device"
        )
    concrete = torch.device("cuda:0")
    torch.cuda.set_device(concrete)
    torch.cuda.synchronize(concrete)
    properties = torch.cuda.get_device_properties(concrete)
    free, total = torch.cuda.mem_get_info(concrete)
    stats = torch.cuda.memory_stats(concrete)
    gpu_name = torch.cuda.get_device_name(concrete)
    compute_capability = (int(properties.major), int(properties.minor))
    if (
        gpu_name != TOP2000_M03R_V7_2026_REQUIRED_GPU_NAME
        or not TOP2000_M03R_V7_2026_MINIMUM_GPU_MEMORY_BYTES
        <= int(properties.total_memory)
        <= TOP2000_M03R_V7_2026_MAXIMUM_GPU_MEMORY_BYTES
        or compute_capability
        != TOP2000_M03R_V7_2026_REQUIRED_COMPUTE_CAPABILITY
    ):
        raise Top2000M03RV72026ExecutionError(
            "2026 checkpoint execution requires one exact H100 80GB device"
        )
    startup = _CudaStartup(
        device=concrete,
        cuda_visible_device=None if visible is None else visible.strip(),
        gpu_name=gpu_name,
        gpu_total_memory_bytes=int(properties.total_memory),
        compute_capability=compute_capability,
        startup_allocated_bytes=int(torch.cuda.memory_allocated(concrete)),
        startup_reserved_bytes=int(torch.cuda.memory_reserved(concrete)),
        startup_free_bytes=int(free),
        startup_allocator_oom_count=int(stats.get("num_ooms", 0)),
        startup_allocator_retry_count=int(stats.get("num_alloc_retries", 0)),
    )
    if total != startup.gpu_total_memory_bytes:
        raise Top2000M03RV72026ExecutionError(
            "CUDA runtime and device properties disagree on total memory"
        )
    torch.cuda.reset_peak_memory_stats(concrete)
    return startup


def _finish_single_cuda(
    startup: _CudaStartup,
    *,
    completed_transition_rows: int,
) -> Top2000M03RV72026CudaExecutionProof:
    torch.cuda.synchronize(startup.device)
    free, total = torch.cuda.mem_get_info(startup.device)
    stats = torch.cuda.memory_stats(startup.device)
    if total != startup.gpu_total_memory_bytes:
        raise Top2000M03RV72026ExecutionError(
            "visible CUDA device changed during inference"
        )
    return Top2000M03RV72026CudaExecutionProof(
        device=str(startup.device),
        cuda_visible_device=startup.cuda_visible_device,
        visible_cuda_device_count=torch.cuda.device_count(),
        gpu_name=torch.cuda.get_device_name(startup.device),
        gpu_total_memory_bytes=startup.gpu_total_memory_bytes,
        compute_capability=startup.compute_capability,
        startup_allocated_bytes=startup.startup_allocated_bytes,
        startup_reserved_bytes=startup.startup_reserved_bytes,
        startup_free_bytes=startup.startup_free_bytes,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(startup.device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(startup.device)),
        final_allocated_bytes=int(torch.cuda.memory_allocated(startup.device)),
        final_reserved_bytes=int(torch.cuda.memory_reserved(startup.device)),
        final_free_bytes=int(free),
        allocator_oom_count_delta=(
            int(stats.get("num_ooms", 0)) - startup.startup_allocator_oom_count
        ),
        allocator_retry_count_delta=(
            int(stats.get("num_alloc_retries", 0))
            - startup.startup_allocator_retry_count
        ),
        completed_transition_rows=completed_transition_rows,
    )


def _begin_checkpoint_cuda(startup: _CudaStartup) -> _CudaStartup:
    """Reset only per-checkpoint peaks after one session-level CUDA guard."""

    torch.cuda.synchronize(startup.device)
    if (
        torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(startup.device) != startup.gpu_name
    ):
        raise Top2000M03RV72026ExecutionError(
            "qualified CUDA identity changed inside the resident session"
        )
    properties = torch.cuda.get_device_properties(startup.device)
    free, total = torch.cuda.mem_get_info(startup.device)
    stats = torch.cuda.memory_stats(startup.device)
    if (
        int(properties.total_memory) != startup.gpu_total_memory_bytes
        or int(total) != startup.gpu_total_memory_bytes
        or (int(properties.major), int(properties.minor))
        != startup.compute_capability
    ):
        raise Top2000M03RV72026ExecutionError(
            "qualified CUDA capacity changed inside the resident session"
        )
    checkpoint_start = _CudaStartup(
        device=startup.device,
        cuda_visible_device=startup.cuda_visible_device,
        gpu_name=startup.gpu_name,
        gpu_total_memory_bytes=startup.gpu_total_memory_bytes,
        compute_capability=startup.compute_capability,
        startup_allocated_bytes=int(torch.cuda.memory_allocated(startup.device)),
        startup_reserved_bytes=int(torch.cuda.memory_reserved(startup.device)),
        startup_free_bytes=int(free),
        startup_allocator_oom_count=int(stats.get("num_ooms", 0)),
        startup_allocator_retry_count=int(stats.get("num_alloc_retries", 0)),
    )
    torch.cuda.reset_peak_memory_stats(startup.device)
    return checkpoint_start


def _geometry_sha256(n_positions: int) -> str:
    roles = TOP2000_M03R_V7_2026_EXECUTION_GEOMETRY.roles(n_positions)
    return _sha256(
        {
            "geometry": asdict(TOP2000_M03R_V7_2026_EXECUTION_GEOMETRY),
            "n_positions": n_positions,
            "warmup": roles.warmup.tolist(),
            "anchors": roles.anchors.tolist(),
            "support": roles.support.tolist(),
            "terminal_observation": roles.terminal_observation,
        }
    )


def _flatten_hashes(prefix: str, value: Any) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []

    def visit(path: str, item: Any) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                visit(f"{path}/{key}", item[key])
        elif isinstance(item, (list, tuple)):
            if (
                len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
                and len(item[1]) == 64
                and all(character in "0123456789abcdef" for character in item[1])
            ):
                _require_digest(f"{path}/{item[0]}", item[1])
                result.append((f"{path}/{item[0]}", item[1]))
            else:
                for index, nested in enumerate(item):
                    visit(f"{path}/{index}", nested)
        elif path.rsplit("/", 1)[-1].endswith("sha256") and isinstance(item, str):
            _require_digest(path, item)
            result.append((path, item))

    visit(prefix, value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026SingleCheckpointExecutionReceipt:
    """All trust-boundary identities for one immutable execution artifact."""

    setting_index: int
    setting_id: str
    runtime_setting_id: str
    training_fold_index: int
    checkpoint_role: str
    evaluation_plan_receipt_sha256: str
    execution_source_inventory_sha256: str
    checkpoint_load_receipt_sha256: str
    factor_calibration_receipt_sha256: str
    economic_execution_receipt_sha256: str
    trace_telemetry_receipt_sha256: str
    cohort_trajectory_receipt_sha256: str
    cuda_execution_receipt_sha256: str
    source_evidence_receipt_sha256: str
    chronology_receipt_sha256: str
    retrospective_cache_file_sha256: str
    runtime_geometry_sha256: str
    policy_factor_loadings_sha256: str
    policy_factor_constraint_pinv_sha256: str
    policy_model_state_sha256_before: str
    policy_model_state_sha256_after: str
    evaluator_array_sha256s: tuple[tuple[str, str], ...]
    cohort_array_sha256s: tuple[tuple[str, str], ...]
    bound_hash_inventory: tuple[tuple[str, str], ...]
    completed_transition_rows: int
    scored_transition_rows: int
    economic_execution_start: int
    elapsed_wall_seconds: float
    canonical_pass_count: int = 1
    policy_optimizer_step_count: int = 0
    policy_state_changed: bool = False
    gradient_enabled_during_pass: bool = False
    one_process: bool = True
    one_gpu: bool = True
    one_continuous_economic_trace: bool = True
    state_reset_count: int = 0
    policy_training_authorized: bool = False
    future_selected_universe: bool = True
    development_only: bool = True
    dataset_reportable: bool = False
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "evaluation_plan_receipt_sha256",
            "execution_source_inventory_sha256",
            "checkpoint_load_receipt_sha256",
            "factor_calibration_receipt_sha256",
            "economic_execution_receipt_sha256",
            "trace_telemetry_receipt_sha256",
            "cohort_trajectory_receipt_sha256",
            "cuda_execution_receipt_sha256",
            "source_evidence_receipt_sha256",
            "chronology_receipt_sha256",
            "retrospective_cache_file_sha256",
            "runtime_geometry_sha256",
            "policy_factor_loadings_sha256",
            "policy_factor_constraint_pinv_sha256",
            "policy_model_state_sha256_before",
            "policy_model_state_sha256_after",
        ):
            _require_digest(name, getattr(self, name))
        for inventory_name, inventory in (
            ("evaluator_array_sha256s", self.evaluator_array_sha256s),
            ("cohort_array_sha256s", self.cohort_array_sha256s),
            ("bound_hash_inventory", self.bound_hash_inventory),
        ):
            names = tuple(name for name, _digest_value in inventory)
            if not inventory or names != tuple(sorted(names)) or len(set(names)) != len(names):
                raise Top2000M03RV72026ExecutionError(
                    f"{inventory_name} must be nonempty, unique, and sorted"
                )
            for name, digest in inventory:
                if not name:
                    raise Top2000M03RV72026ExecutionError(
                        f"{inventory_name} contains an empty name"
                    )
                _require_digest(f"{inventory_name}[{name}]", digest)
        if (
            self.schema != TOP2000_M03R_V7_2026_EXECUTION_RECEIPT_SCHEMA
            or self.setting_index not in range(12)
            or not self.setting_id
            or not self.runtime_setting_id
            or self.training_fold_index not in range(6)
            or self.checkpoint_role
            != ("headline" if self.training_fold_index == 5 else "cutoff-sensitivity")
            or self.completed_transition_rows <= 0
            or self.scored_transition_rows <= 0
            or not 0 <= self.economic_execution_start < self.completed_transition_rows
            or not np.isfinite(self.elapsed_wall_seconds)
            or self.elapsed_wall_seconds <= 0.0
            or self.policy_model_state_sha256_before
            != self.policy_model_state_sha256_after
            or self.canonical_pass_count != 1
            or self.policy_optimizer_step_count != 0
            or self.policy_state_changed
            or self.gradient_enabled_during_pass
            or not self.one_process
            or not self.one_gpu
            or not self.one_continuous_economic_trace
            or self.state_reset_count != 0
            or self.policy_training_authorized
            or not self.future_selected_universe
            or not self.development_only
            or self.dataset_reportable
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026ExecutionError(
                "single-checkpoint execution receipt overclaims or drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


def _evaluation_arrays(
    result: Top2000M03RV72026TraceEvaluationInputs,
) -> dict[str, np.ndarray]:
    telemetry = result.telemetry
    arrays = {
        "benchmark_gross_returns": np.asarray(result.benchmark_gross_returns),
        "benchmark_net_returns_20bp": np.asarray(
            result.benchmark_net_returns_20bp
        ),
        "construction_to_fill_safety_projection_distance": np.asarray(
            result.construction_to_fill_safety_projection_distance
        ),
        "portfolio_gross_returns": np.asarray(result.portfolio_gross_returns),
        "portfolio_net_returns_20bp": np.asarray(
            result.portfolio_net_returns_20bp
        ),
        "telemetry/age_notional_at_risk": np.asarray(
            telemetry.age_notional_at_risk
        ),
        "telemetry/continuous_hazard": np.asarray(telemetry.continuous_hazard),
        "telemetry/continuous_hazard_observed": np.asarray(
            telemetry.continuous_hazard_observed
        ),
        "telemetry/discretionary_exit_notional_by_age": np.asarray(
            telemetry.discretionary_exit_notional_by_age
        ),
        "telemetry/requested_to_executed_projection_distance": np.asarray(
            telemetry.requested_to_executed_projection_distance
        ),
    }
    arrays.update(
        {
            f"portfolio_turnover_by_cause/{name}": np.asarray(value)
            for name, value in result.portfolio_turnover_by_cause.items()
        }
    )
    arrays.update(
        {
            f"benchmark_turnover_by_cause/{name}": np.asarray(value)
            for name, value in result.benchmark_turnover_by_cause.items()
        }
    )
    arrays.update(
        {
            f"telemetry/forced_exit_notional_by_cause_and_age/{name}": np.asarray(
                value
            )
            for name, value in telemetry.forced_exit_notional_by_cause_and_age.items()
        }
    )
    arrays.update(
        {
            f"telemetry/action_counts_by_type/{name}": np.asarray(value)
            for name, value in telemetry.action_counts_by_type.items()
        }
    )
    return arrays


def _cohort_arrays(
    trajectories: Top2000M03RV72026CohortTrajectories,
) -> dict[str, np.ndarray]:
    arrays = {
        "cohort/entry_units": np.asarray(trajectories.entry_units),
        "cohort/discretionary_event_units_by_age": np.asarray(
            trajectories.discretionary_event_units_by_age
        ),
        "cohort/terminal_censor_units_by_age": np.asarray(
            trajectories.terminal_censor_units_by_age
        ),
    }
    arrays.update(
        {
            f"cohort/forced_censor_units_by_cause_and_age/{cause}": np.asarray(
                value
            )
            for cause, value in trajectories.forced_censor_units_by_cause_and_age.items()
        }
    )
    return arrays


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype not in (
        np.dtype("float32"),
        np.dtype("float64"),
        np.dtype("bool"),
    ):
        raise Top2000M03RV72026ExecutionError(
            f"artifact array dtype {array.dtype} is not frozen"
        )
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        "sha256": _array_sha256(array),
    }


def _decode_array(name: str, payload: object) -> np.ndarray:
    if not isinstance(payload, dict) or set(payload) != {
        "dtype",
        "shape",
        "data_base64",
        "sha256",
    }:
        raise Top2000M03RV72026ExecutionError(
            f"artifact array {name!r} has an invalid envelope"
        )
    try:
        dtype = np.dtype(payload["dtype"])
        shape = tuple(int(value) for value in payload["shape"])
        raw = base64.b64decode(payload["data_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026ExecutionError(
            f"artifact array {name!r} cannot be decoded"
        ) from exc
    if (
        dtype not in (np.dtype("float32"), np.dtype("float64"), np.dtype("bool"))
        or any(value < 0 for value in shape)
        or len(raw) != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    ):
        raise Top2000M03RV72026ExecutionError(
            f"artifact array {name!r} geometry or dtype drifted"
        )
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if _array_sha256(array) != _require_digest(
        f"artifact array {name} sha256", payload["sha256"]
    ):
        raise Top2000M03RV72026ExecutionError(
            f"artifact array {name!r} content hash drifted"
        )
    array.setflags(write=False)
    return array


def _publish_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = _canonical_json(payload)
    if len(encoded) > _ARTIFACT_MAX_BYTES:
        raise Top2000M03RV72026ExecutionError(
            "single-checkpoint artifact exceeds the 64-MiB bound"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact {path}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026ExecutionArtifactBinding:
    artifact_path: str
    artifact_file_sha256: str
    execution_receipt_sha256: str
    setting_index: int
    training_fold_index: int
    development_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        _require_digest("artifact_file_sha256", self.artifact_file_sha256)
        _require_digest("execution_receipt_sha256", self.execution_receipt_sha256)
        if (
            not self.artifact_path
            or self.setting_index not in range(12)
            or self.training_fold_index not in range(6)
            or not self.development_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026ExecutionError(
                "artifact binding identity or research-only semantics drifted"
            )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026ExecutionSession:
    """One resident cache/calibration shared by sequential fold executions."""

    retrospective: Top2000M03RV72026RetrospectiveData
    pre2026_cache: Top2000VerifiedDevelopmentCache
    factor_calibration: Top2000M03RV72026PreScoreFactorCalibration
    device: torch.device
    retrospective_cache_file_sha256: str
    pre2026_cache_file_sha256: str
    chronology_receipt_sha256: str
    evaluation_plan_receipt_sha256: str
    execution_source_inventory_sha256: str
    _qualified_cuda: _CudaStartup
    _tensor_versions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in (
            "retrospective_cache_file_sha256",
            "pre2026_cache_file_sha256",
            "chronology_receipt_sha256",
            "evaluation_plan_receipt_sha256",
            "execution_source_inventory_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.device != torch.device("cuda:0")
            or self.retrospective.cache_file_sha256
            != self.retrospective_cache_file_sha256
            or self.pre2026_cache.cache_sha256 != self.pre2026_cache_file_sha256
            or self.retrospective.identity.receipt_sha256
            != self.chronology_receipt_sha256
            or self.factor_calibration.retrospective_data_receipt_sha256
            != self.chronology_receipt_sha256
            or self.factor_calibration.retrospective_cache_file_sha256
            != self.retrospective_cache_file_sha256
            or self._qualified_cuda.device != self.device
        ):
            raise Top2000M03RV72026ExecutionError(
                "execution session identities or sole CUDA device drifted"
            )
        self.validate_unmodified()

    def _versioned_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        sequence = self.retrospective.sequence
        return (
            ("decision_state", sequence.decision_state),
            ("asset_returns", sequence.asset_returns),
            ("decision_available", sequence.decision_available),
            ("benchmark_weights", sequence.benchmark_weights),
            ("benchmark_net_returns", sequence.benchmark_net_returns),
            ("factor_loadings", self.factor_calibration.loadings),
        )

    def validate_unmodified(self) -> None:
        self.pre2026_cache.validate_unmodified()
        observed = tuple(
            (name, value._version) for name, value in self._versioned_tensors()
        )
        if observed != self._tensor_versions:
            raise Top2000M03RV72026ExecutionError(
                "resident execution-session tensors changed after qualification"
            )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026LoadedExecutionArtifact:
    evaluation_inputs: Top2000M03RV72026TraceEvaluationInputs
    cohort_trajectories: Top2000M03RV72026CohortTrajectories
    execution_receipt: Top2000M03RV72026SingleCheckpointExecutionReceipt
    checkpoint_load_receipt: Top2000M03RV72026CheckpointLoadReceipt
    factor_calibration: Top2000M03RV72026PreScoreFactorCalibration
    economic_execution_receipt: Top2000M03RV72026EconomicExecutionReceipt
    cuda_proof: Top2000M03RV72026CudaExecutionProof
    source_evidence: Top2000M03RV72026RetrospectiveSourceEvidence
    chronology_identity: Top2000M03RV72026RetrospectiveIdentity
    artifact_file_sha256: str


def prepare_top2000_m03r_v7_seed17_2026_execution_session(
    *,
    pre2026_cache_path: str | Path,
    expected_pre2026_cache_sha256: str,
    retrospective_cache_path: str | Path,
    expected_retrospective_cache_sha256: str,
    evaluation_plan_receipt_sha256: str,
    execution_source_inventory_sha256: str,
    device: str | torch.device = "cuda:0",
) -> Top2000M03RV72026ExecutionSession:
    """Load immutable data once for sequential folds in one setting worker."""

    _require_digest("expected_pre2026_cache_sha256", expected_pre2026_cache_sha256)
    _require_digest(
        "expected_retrospective_cache_sha256",
        expected_retrospective_cache_sha256,
    )
    _require_digest(
        "evaluation_plan_receipt_sha256", evaluation_plan_receipt_sha256
    )
    _require_digest(
        "execution_source_inventory_sha256", execution_source_inventory_sha256
    )
    startup = _start_single_cuda(device)
    retrospective = load_top2000_m03r_v7_2026_retrospective_cache(
        retrospective_cache_path,
        expected_cache_sha256=expected_retrospective_cache_sha256,
        output_device=startup.device,
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )
    if retrospective.cache_file_sha256 is None:
        raise Top2000M03RV72026ExecutionError(
            "loaded retrospective omitted its immutable cache hash"
        )
    pre2026 = load_verified_top2000_hold30_development_cache(
        pre2026_cache_path,
        expected_cache_sha256=expected_pre2026_cache_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    if (
        pre2026.cache_sha256 != retrospective.identity.pre2026_cache_sha256
        or pre2026.cache_identity != retrospective.identity.pre2026_cache_identity
        or pre2026.action_hash != retrospective.identity.action_hash
    ):
        raise Top2000M03RV72026ExecutionError(
            "pre-2026 cache does not match the retrospective encoder context"
        )
    calibration = fit_top2000_m03r_v7_2026_pre_score_factor_calibration(
        retrospective
    )
    tensors = (
        ("decision_state", retrospective.sequence.decision_state),
        ("asset_returns", retrospective.sequence.asset_returns),
        ("decision_available", retrospective.sequence.decision_available),
        ("benchmark_weights", retrospective.sequence.benchmark_weights),
        ("benchmark_net_returns", retrospective.sequence.benchmark_net_returns),
        ("factor_loadings", calibration.loadings),
    )
    return Top2000M03RV72026ExecutionSession(
        retrospective=retrospective,
        pre2026_cache=pre2026,
        factor_calibration=calibration,
        device=startup.device,
        retrospective_cache_file_sha256=expected_retrospective_cache_sha256,
        pre2026_cache_file_sha256=pre2026.cache_sha256,
        chronology_receipt_sha256=retrospective.identity.receipt_sha256,
        evaluation_plan_receipt_sha256=evaluation_plan_receipt_sha256,
        execution_source_inventory_sha256=execution_source_inventory_sha256,
        _tensor_versions=tuple((name, value._version) for name, value in tensors),
        _qualified_cuda=startup,
    )


def _artifact_payload(
    inputs: Top2000M03RV72026TraceEvaluationInputs,
    *,
    cohort_trajectories: Top2000M03RV72026CohortTrajectories,
    policy_factor_constraint_pinv: np.ndarray,
    execution_receipt: Top2000M03RV72026SingleCheckpointExecutionReceipt,
    checkpoint_receipt: Top2000M03RV72026CheckpointLoadReceipt,
    calibration: Top2000M03RV72026PreScoreFactorCalibration,
    execution_view: Top2000M03RV72026EconomicExecutionView,
    cuda_proof: Top2000M03RV72026CudaExecutionProof,
    source: Top2000M03RV72026RetrospectiveSourceEvidence,
    identity: Top2000M03RV72026RetrospectiveIdentity,
) -> dict[str, Any]:
    arrays = _evaluation_arrays(inputs)
    arrays["factor_calibration/loadings"] = (
        calibration.loadings.detach().to(device="cpu").numpy().copy()
    )
    arrays["factor_calibration/constraint_pinv"] = np.asarray(
        policy_factor_constraint_pinv
    )
    arrays.update(_cohort_arrays(cohort_trajectories))
    return {
        "schema": TOP2000_M03R_V7_2026_EXECUTION_ARTIFACT_SCHEMA,
        "execution_receipt": asdict(execution_receipt),
        "checkpoint_load_receipt": asdict(checkpoint_receipt),
        "factor_calibration_receipt": calibration.canonical_payload(),
        "economic_execution_receipt": execution_view.receipt.canonical_payload(),
        "trace_telemetry_receipt": inputs.receipt.canonical_payload(),
        "cohort_trajectory_receipt": asdict(cohort_trajectories.receipt),
        "cuda_execution_proof": asdict(cuda_proof),
        "source_evidence": asdict(source),
        "chronology_identity": identity.canonical_payload(),
        "score_dates": list(inputs.score_dates),
        "arrays": {
            name: _encode_array(value) for name, value in sorted(arrays.items())
        },
    }


def run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session(
    session: Top2000M03RV72026ExecutionSession,
    binding: Top2000M03RV72026FrozenCheckpointBinding,
    *,
    training_output_root: str | Path,
    output_path: str | Path,
) -> Top2000M03RV72026ExecutionArtifactBinding:
    """Run one checkpoint from a resident per-setting data session."""

    if not isinstance(session, Top2000M03RV72026ExecutionSession):
        raise Top2000M03RV72026ExecutionError(
            "single-checkpoint execution requires a typed resident session"
        )
    started_at = time.perf_counter()
    session.validate_unmodified()
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable execution artifact {destination}"
        )
    startup = _begin_checkpoint_cuda(session._qualified_cuda)
    retrospective = session.retrospective
    pre2026 = session.pre2026_cache
    calibration = session.factor_calibration
    loaded = load_top2000_m03r_v7_seed17_2026_checkpoint(
        binding,
        training_output_root=training_output_root,
        device=startup.device,
    )
    if (
        loaded.training_plan.cache_sha256 != session.pre2026_cache_file_sha256
        or loaded.training_plan.cache_sha256
        != retrospective.identity.pre2026_cache_sha256
        or pre2026.cache_identity != retrospective.identity.pre2026_cache_identity
    ):
        raise Top2000M03RV72026ExecutionError(
            "checkpoint training cache and retrospective encoder context differ"
        )

    loaded.policy.bind_episode_factor_loadings(calibration.loadings)
    policy_factor_loadings_sha256 = _tensor_sha256(
        loaded.policy.episode_factor_loadings
    )
    policy_factor_constraint_pinv_sha256 = _tensor_sha256(
        loaded.policy.episode_factor_constraint_pinv
    )
    view = build_top2000_m03r_v7_2026_economic_execution_view(
        retrospective,
        pre2026,
        loaded.training_fold,
        loaded.policy,
    )
    runtime = build_top2000_m03r_v7_validation_runtime(
        loaded.policy,
        state_provider=view.state_provider,
    )
    roles = TOP2000_M03R_V7_2026_EXECUTION_GEOMETRY.roles(
        view.sequence.n_positions
    )
    runtime_geometry_sha256 = _geometry_sha256(view.sequence.n_positions)
    loaded.policy.requires_grad_(False)
    loaded.policy.eval()
    if any(parameter.requires_grad for parameter in loaded.policy.parameters()):
        raise Top2000M03RV72026ExecutionError(
            "inference policy retained a trainable parameter"
        )
    state_before = model_state_sha256(loaded.policy)
    if state_before != loaded.receipt.model_state_sha256:
        raise Top2000M03RV72026ExecutionError(
            "policy state changed before chronological inference"
        )
    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise AssertionError("torch.inference_mode failed to disable gradients")
        trace, _rows = runtime.canonical_pass(
            loaded.policy,
            view.sequence,
            roles,
        )
    if not isinstance(trace, Hold30CanonicalTrace):
        raise Top2000M03RV72026ExecutionError(
            "chronological runtime returned an unexpected trace type"
        )
    if (
        len(trace.transitions) != view.receipt.executed_transition_rows
        or len(trace.boundary_states) != view.receipt.executed_state_rows
        or trace.boundary_states[0].position_index != 0
        or trace.terminal_state.position_index != len(trace.transitions)
    ):
        raise Top2000M03RV72026ExecutionError(
            "runtime did not execute the post-cutoff economic chronology once"
        )
    state_after = model_state_sha256(loaded.policy)
    if state_after != state_before:
        raise Top2000M03RV72026ExecutionError(
            "policy state changed during inference-only execution"
        )
    inputs = adapt_top2000_m03r_v7_2026_trace(
        trace,
        retrospective,
        setting_id=loaded.receipt.setting_id,
        checkpoint_sha256=loaded.receipt.model_file_sha256,
        checkpoint_fold_index=loaded.receipt.training_fold_index,
        economic_execution_view=view,
    )
    validate_top2000_m03r_v7_2026_trace_evaluation_inputs(inputs)
    cohort_trajectories = build_top2000_m03r_v7_2026_cohort_trajectories(
        trace,
        retrospective,
        view,
        setting_id=loaded.receipt.setting_id,
        checkpoint_sha256=loaded.receipt.model_file_sha256,
        checkpoint_fold_index=loaded.receipt.training_fold_index,
    )
    validate_top2000_m03r_v7_2026_cohort_trajectories(cohort_trajectories)
    cuda_proof = _finish_single_cuda(
        startup,
        completed_transition_rows=len(trace.transitions),
    )

    evaluator_arrays = _evaluation_arrays(inputs)
    evaluator_array_hashes = tuple(
        sorted((name, _array_sha256(value)) for name, value in evaluator_arrays.items())
    )
    cohort_array_hashes = tuple(
        sorted(
            (name, _array_sha256(value))
            for name, value in _cohort_arrays(cohort_trajectories).items()
        )
    )
    payloads = {
        "checkpoint_load": asdict(loaded.receipt),
        "factor_calibration": calibration.canonical_payload(),
        "economic_execution": view.receipt.canonical_payload(),
        "trace_telemetry": inputs.receipt.canonical_payload(),
        "cohort_trajectory": asdict(cohort_trajectories.receipt),
        "cuda_execution": asdict(cuda_proof),
        "source_evidence": asdict(retrospective.source_evidence),
        "chronology_identity": retrospective.identity.canonical_payload(),
    }
    bound_hashes = tuple(
        sorted(
            item
            for name, payload in payloads.items()
            for item in _flatten_hashes(name, payload)
        )
    )
    execution_receipt = Top2000M03RV72026SingleCheckpointExecutionReceipt(
        setting_index=loaded.receipt.setting_index,
        setting_id=loaded.receipt.setting_id,
        runtime_setting_id=loaded.receipt.runtime_setting_id,
        training_fold_index=loaded.receipt.training_fold_index,
        checkpoint_role=loaded.receipt.checkpoint_role,
        evaluation_plan_receipt_sha256=(
            session.evaluation_plan_receipt_sha256
        ),
        execution_source_inventory_sha256=(
            session.execution_source_inventory_sha256
        ),
        checkpoint_load_receipt_sha256=loaded.receipt.receipt_sha256,
        factor_calibration_receipt_sha256=calibration.receipt_sha256,
        economic_execution_receipt_sha256=view.receipt.receipt_sha256,
        trace_telemetry_receipt_sha256=inputs.receipt.receipt_sha256,
        cohort_trajectory_receipt_sha256=(
            cohort_trajectories.receipt.receipt_sha256
        ),
        cuda_execution_receipt_sha256=cuda_proof.receipt_sha256,
        source_evidence_receipt_sha256=(
            retrospective.source_evidence.receipt_sha256
        ),
        chronology_receipt_sha256=retrospective.identity.receipt_sha256,
        retrospective_cache_file_sha256=(
            session.retrospective_cache_file_sha256
        ),
        runtime_geometry_sha256=runtime_geometry_sha256,
        policy_factor_loadings_sha256=policy_factor_loadings_sha256,
        policy_factor_constraint_pinv_sha256=(
            policy_factor_constraint_pinv_sha256
        ),
        policy_model_state_sha256_before=state_before,
        policy_model_state_sha256_after=state_after,
        evaluator_array_sha256s=evaluator_array_hashes,
        cohort_array_sha256s=cohort_array_hashes,
        bound_hash_inventory=bound_hashes,
        completed_transition_rows=len(trace.transitions),
        scored_transition_rows=len(inputs.score_dates),
        economic_execution_start=view.receipt.economic_execution_start,
        elapsed_wall_seconds=time.perf_counter() - started_at,
    )
    payload = _artifact_payload(
        inputs,
        cohort_trajectories=cohort_trajectories,
        policy_factor_constraint_pinv=(
            loaded.policy.episode_factor_constraint_pinv.detach()
            .to(device="cpu")
            .numpy()
            .copy()
        ),
        execution_receipt=execution_receipt,
        checkpoint_receipt=loaded.receipt,
        calibration=calibration,
        execution_view=view,
        cuda_proof=cuda_proof,
        source=retrospective.source_evidence,
        identity=retrospective.identity,
    )
    artifact_sha256 = _publish_immutable_json(destination, payload)
    return Top2000M03RV72026ExecutionArtifactBinding(
        artifact_path=str(destination),
        artifact_file_sha256=artifact_sha256,
        execution_receipt_sha256=execution_receipt.receipt_sha256,
        setting_index=execution_receipt.setting_index,
        training_fold_index=execution_receipt.training_fold_index,
    )


def run_top2000_m03r_v7_seed17_2026_single_checkpoint(
    binding: Top2000M03RV72026FrozenCheckpointBinding,
    *,
    training_output_root: str | Path,
    pre2026_cache_path: str | Path,
    expected_pre2026_cache_sha256: str,
    retrospective_cache_path: str | Path,
    expected_retrospective_cache_sha256: str,
    evaluation_plan_receipt_sha256: str,
    execution_source_inventory_sha256: str,
    output_path: str | Path,
    device: str | torch.device = "cuda:0",
) -> Top2000M03RV72026ExecutionArtifactBinding:
    """Convenience surface for an isolated one-checkpoint process.

    A per-setting worker should prepare one resident session and use the
    session surface for headline fold 5 followed by folds 0..4.
    """

    session = prepare_top2000_m03r_v7_seed17_2026_execution_session(
        pre2026_cache_path=pre2026_cache_path,
        expected_pre2026_cache_sha256=expected_pre2026_cache_sha256,
        retrospective_cache_path=retrospective_cache_path,
        expected_retrospective_cache_sha256=(
            expected_retrospective_cache_sha256
        ),
        evaluation_plan_receipt_sha256=evaluation_plan_receipt_sha256,
        execution_source_inventory_sha256=execution_source_inventory_sha256,
        device=device,
    )
    return run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session(
        session,
        binding,
        training_output_root=training_output_root,
        output_path=output_path,
    )


def _typed_payload(payload: object, cls: type[Any], *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise Top2000M03RV72026ExecutionError(f"{label} must be a JSON object")
    expected = {field.name for field in fields(cls)}
    if set(payload) != expected:
        raise Top2000M03RV72026ExecutionError(f"{label} fields drifted")
    return dict(payload)


def load_top2000_m03r_v7_seed17_2026_execution_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_evaluation_plan_receipt_sha256: str,
    expected_execution_source_inventory_sha256: str,
) -> Top2000M03RV72026LoadedExecutionArtifact:
    """Load, reconstruct, and independently revalidate one small artifact."""

    _require_digest("expected_file_sha256", expected_file_sha256)
    _require_digest(
        "expected_evaluation_plan_receipt_sha256",
        expected_evaluation_plan_receipt_sha256,
    )
    _require_digest(
        "expected_execution_source_inventory_sha256",
        expected_execution_source_inventory_sha256,
    )
    artifact_path = Path(path)
    if (
        not artifact_path.is_file()
        or artifact_path.is_symlink()
        or artifact_path.stat().st_size > _ARTIFACT_MAX_BYTES
    ):
        raise Top2000M03RV72026ExecutionError(
            "execution artifact must be a bounded regular non-symlink file"
        )
    actual_sha256 = _file_sha256(artifact_path)
    if actual_sha256 != expected_file_sha256:
        raise Top2000M03RV72026ExecutionError(
            "execution artifact file SHA-256 mismatch"
        )
    try:
        raw = artifact_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV72026ExecutionError(
            "execution artifact is not valid JSON"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != actual_sha256:
        raise Top2000M03RV72026ExecutionError(
            "execution artifact changed while it was being loaded"
        )
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "execution_receipt",
        "checkpoint_load_receipt",
        "factor_calibration_receipt",
        "economic_execution_receipt",
        "trace_telemetry_receipt",
        "cohort_trajectory_receipt",
        "cuda_execution_proof",
        "source_evidence",
        "chronology_identity",
        "score_dates",
        "arrays",
    }:
        raise Top2000M03RV72026ExecutionError(
            "execution artifact inventory drifted"
        )
    if (
        payload["schema"] != TOP2000_M03R_V7_2026_EXECUTION_ARTIFACT_SCHEMA
        or _canonical_json(payload) != raw
    ):
        raise Top2000M03RV72026ExecutionError(
            "execution artifact schema or canonical serialization drifted"
        )
    try:
        execution_payload = _typed_payload(
            payload["execution_receipt"],
            Top2000M03RV72026SingleCheckpointExecutionReceipt,
            label="execution receipt",
        )
        for name in (
            "evaluator_array_sha256s",
            "cohort_array_sha256s",
            "bound_hash_inventory",
        ):
            execution_payload[name] = tuple(
                tuple(value) for value in execution_payload[name]
            )
        execution_receipt = Top2000M03RV72026SingleCheckpointExecutionReceipt(
            **execution_payload
        )
        checkpoint_receipt = Top2000M03RV72026CheckpointLoadReceipt(
            **_typed_payload(
                payload["checkpoint_load_receipt"],
                Top2000M03RV72026CheckpointLoadReceipt,
                label="checkpoint load receipt",
            )
        )
        view_receipt = Top2000M03RV72026EconomicExecutionReceipt(
            **_typed_payload(
                payload["economic_execution_receipt"],
                Top2000M03RV72026EconomicExecutionReceipt,
                label="economic execution receipt",
            )
        )
        cuda_payload = _typed_payload(
            payload["cuda_execution_proof"],
            Top2000M03RV72026CudaExecutionProof,
            label="CUDA execution proof",
        )
        cuda_payload["compute_capability"] = tuple(cuda_payload["compute_capability"])
        cuda_proof = Top2000M03RV72026CudaExecutionProof(**cuda_payload)
        source = Top2000M03RV72026RetrospectiveSourceEvidence(
            **_typed_payload(
                payload["source_evidence"],
                Top2000M03RV72026RetrospectiveSourceEvidence,
                label="source evidence",
            )
        )
        identity = Top2000M03RV72026RetrospectiveIdentity(
            **_typed_payload(
                payload["chronology_identity"],
                Top2000M03RV72026RetrospectiveIdentity,
                label="chronology identity",
            )
        )
        trace_payload = _typed_payload(
            payload["trace_telemetry_receipt"],
            Top2000M03RV72026TraceTelemetryReceipt,
            label="trace telemetry receipt",
        )
        trace_payload["scored_array_sha256s"] = tuple(
            tuple(value) for value in trace_payload["scored_array_sha256s"]
        )
        trace_payload["benchmark_cause_mapping"] = tuple(
            tuple(value) for value in trace_payload["benchmark_cause_mapping"]
        )
        trace_receipt = Top2000M03RV72026TraceTelemetryReceipt(**trace_payload)
        cohort_payload = _typed_payload(
            payload["cohort_trajectory_receipt"],
            Top2000M03RV72026CohortTrajectoryReceipt,
            label="cohort trajectory receipt",
        )
        cohort_payload["forced_censor_units_by_cause_and_age_sha256"] = tuple(
            tuple(value)
            for value in cohort_payload[
                "forced_censor_units_by_cause_and_age_sha256"
            ]
        )
        cohort_receipt = Top2000M03RV72026CohortTrajectoryReceipt(
            **cohort_payload
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, Top2000M03RV72026ExecutionError):
            raise
        raise Top2000M03RV72026ExecutionError(
            "execution artifact receipt reconstruction failed"
        ) from exc

    array_payload = payload["arrays"]
    if not isinstance(array_payload, dict):
        raise Top2000M03RV72026ExecutionError("artifact arrays must be an object")
    arrays = {
        name: _decode_array(name, value)
        for name, value in sorted(array_payload.items())
    }
    calibration_loading = arrays.pop("factor_calibration/loadings", None)
    calibration_constraint_pinv = arrays.pop(
        "factor_calibration/constraint_pinv", None
    )
    if calibration_loading is None or calibration_constraint_pinv is None:
        raise Top2000M03RV72026ExecutionError(
            "artifact omitted factor-calibration execution tensors"
        )
    raw_calibration_payload = payload["factor_calibration_receipt"]
    expected_calibration_fields = {
        field.name
        for field in fields(Top2000M03RV72026PreScoreFactorCalibration)
        if field.name != "loadings"
    }
    if (
        not isinstance(raw_calibration_payload, dict)
        or set(raw_calibration_payload) != expected_calibration_fields
    ):
        raise Top2000M03RV72026ExecutionError(
            "factor calibration receipt fields drifted"
        )
    calibration_payload = dict(raw_calibration_payload)
    calibration_payload["loadings"] = torch.from_numpy(
        calibration_loading.copy()
    )
    for name in ("action_ids", "factor_names", "calibration_state_dates"):
        calibration_payload[name] = tuple(calibration_payload[name])
    try:
        calibration = Top2000M03RV72026PreScoreFactorCalibration(
            **calibration_payload
        )
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026ExecutionError(
            "factor calibration cannot be independently reconstructed"
        ) from exc

    score_dates = payload["score_dates"]
    if not isinstance(score_dates, list) or any(
        not isinstance(value, str) for value in score_dates
    ):
        raise Top2000M03RV72026ExecutionError(
            "artifact score dates must be a string array"
        )
    cohort_prefix = "cohort/forced_censor_units_by_cause_and_age/"
    cohort_forced_arrays = {
        cause: arrays.pop(f"{cohort_prefix}{cause}").copy()
        for cause, _digest_value in (
            cohort_receipt.forced_censor_units_by_cause_and_age_sha256
        )
    }
    try:
        cohort_trajectories = Top2000M03RV72026CohortTrajectories(
            origin_dates=tuple(score_dates),
            entry_units=arrays.pop("cohort/entry_units").copy(),
            discretionary_event_units_by_age=arrays.pop(
                "cohort/discretionary_event_units_by_age"
            ).copy(),
            forced_censor_units_by_cause_and_age=cohort_forced_arrays,
            terminal_censor_units_by_age=arrays.pop(
                "cohort/terminal_censor_units_by_age"
            ).copy(),
            receipt=cohort_receipt,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Top2000M03RV72026ExecutionError(
            "cohort trajectory arrays cannot be reconstructed"
        ) from exc
    validate_top2000_m03r_v7_2026_cohort_trajectories(cohort_trajectories)

    cause_names = {cause.value for cause in TURNOVER_CAUSES}
    portfolio_turnover = {
        cause: arrays.pop(f"portfolio_turnover_by_cause/{cause}")
        for cause in cause_names
    }
    benchmark_turnover = {
        cause: arrays.pop(f"benchmark_turnover_by_cause/{cause}")
        for cause in cause_names
    }
    forced_prefix = "telemetry/forced_exit_notional_by_cause_and_age/"
    action_prefix = "telemetry/action_counts_by_type/"
    forced = {
        name.removeprefix(forced_prefix): arrays.pop(name)
        for name in tuple(arrays)
        if name.startswith(forced_prefix)
    }
    actions = {
        name.removeprefix(action_prefix): arrays.pop(name)
        for name in tuple(arrays)
        if name.startswith(action_prefix)
    }
    telemetry = Top2000M03RV72026Telemetry(
        requested_to_executed_projection_distance=arrays.pop(
            "telemetry/requested_to_executed_projection_distance"
        ),
        age_notional_at_risk=arrays.pop("telemetry/age_notional_at_risk"),
        discretionary_exit_notional_by_age=arrays.pop(
            "telemetry/discretionary_exit_notional_by_age"
        ),
        forced_exit_notional_by_cause_and_age=MappingProxyType(forced),
        action_counts_by_type=MappingProxyType(actions),
        continuous_hazard=arrays.pop("telemetry/continuous_hazard"),
        continuous_hazard_observed=arrays.pop(
            "telemetry/continuous_hazard_observed"
        ),
    )
    try:
        inputs = Top2000M03RV72026TraceEvaluationInputs(
            score_dates=tuple(score_dates),
            portfolio_gross_returns=arrays.pop("portfolio_gross_returns"),
            benchmark_gross_returns=arrays.pop("benchmark_gross_returns"),
            portfolio_net_returns_20bp=arrays.pop(
                "portfolio_net_returns_20bp"
            ),
            benchmark_net_returns_20bp=arrays.pop(
                "benchmark_net_returns_20bp"
            ),
            portfolio_turnover_by_cause=MappingProxyType(portfolio_turnover),
            benchmark_turnover_by_cause=MappingProxyType(benchmark_turnover),
            telemetry=telemetry,
            construction_to_fill_safety_projection_distance=arrays.pop(
                "construction_to_fill_safety_projection_distance"
            ),
            receipt=trace_receipt,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Top2000M03RV72026ExecutionError(
            "artifact evaluator arrays cannot be reconstructed"
        ) from exc
    if arrays:
        raise Top2000M03RV72026ExecutionError(
            f"artifact contains unknown arrays: {sorted(arrays)}"
        )
    validate_top2000_m03r_v7_2026_trace_evaluation_inputs(inputs)

    payloads = {
        "checkpoint_load": asdict(checkpoint_receipt),
        "factor_calibration": calibration.canonical_payload(),
        "economic_execution": view_receipt.canonical_payload(),
        "trace_telemetry": trace_receipt.canonical_payload(),
        "cohort_trajectory": asdict(cohort_receipt),
        "cuda_execution": asdict(cuda_proof),
        "source_evidence": asdict(source),
        "chronology_identity": identity.canonical_payload(),
    }
    expected_bound_hashes = tuple(
        sorted(
            item
            for name, embedded in payloads.items()
            for item in _flatten_hashes(name, embedded)
        )
    )
    expected_arrays = tuple(
        sorted(
            (name, _array_sha256(value))
            for name, value in _evaluation_arrays(inputs).items()
        )
    )
    expected_cohort_arrays = tuple(
        sorted(
            (name, _array_sha256(value))
            for name, value in _cohort_arrays(cohort_trajectories).items()
        )
    )
    if (
        execution_receipt.evaluation_plan_receipt_sha256
        != expected_evaluation_plan_receipt_sha256
        or execution_receipt.execution_source_inventory_sha256
        != expected_execution_source_inventory_sha256
        or
        execution_receipt.checkpoint_load_receipt_sha256
        != checkpoint_receipt.receipt_sha256
        or execution_receipt.factor_calibration_receipt_sha256
        != calibration.receipt_sha256
        or execution_receipt.economic_execution_receipt_sha256
        != view_receipt.receipt_sha256
        or execution_receipt.trace_telemetry_receipt_sha256
        != trace_receipt.receipt_sha256
        or execution_receipt.cohort_trajectory_receipt_sha256
        != cohort_receipt.receipt_sha256
        or execution_receipt.cuda_execution_receipt_sha256
        != cuda_proof.receipt_sha256
        or execution_receipt.source_evidence_receipt_sha256
        != source.receipt_sha256
        or execution_receipt.chronology_receipt_sha256 != identity.receipt_sha256
        or execution_receipt.evaluator_array_sha256s != expected_arrays
        or execution_receipt.cohort_array_sha256s != expected_cohort_arrays
        or execution_receipt.bound_hash_inventory != expected_bound_hashes
        or execution_receipt.policy_model_state_sha256_before
        != checkpoint_receipt.model_state_sha256
        or execution_receipt.policy_model_state_sha256_after
        != checkpoint_receipt.model_state_sha256
        or execution_receipt.policy_factor_loadings_sha256
        != _tensor_sha256(calibration.loadings)
        or execution_receipt.policy_factor_constraint_pinv_sha256
        != _tensor_sha256(torch.from_numpy(calibration_constraint_pinv.copy()))
        or execution_receipt.setting_index != checkpoint_receipt.setting_index
        or execution_receipt.setting_id != checkpoint_receipt.setting_id
        or execution_receipt.runtime_setting_id
        != checkpoint_receipt.runtime_setting_id
        or execution_receipt.training_fold_index
        != checkpoint_receipt.training_fold_index
        or execution_receipt.checkpoint_role != checkpoint_receipt.checkpoint_role
        or execution_receipt.economic_execution_start
        != view_receipt.economic_execution_start
        or execution_receipt.completed_transition_rows
        != trace_receipt.completed_transition_rows
        or execution_receipt.completed_transition_rows
        != cuda_proof.completed_transition_rows
        or execution_receipt.scored_transition_rows
        != trace_receipt.scored_transition_rows
        or trace_receipt.setting_id != checkpoint_receipt.setting_id
        or trace_receipt.runtime_setting_id != checkpoint_receipt.runtime_setting_id
        or trace_receipt.checkpoint_sha256
        != checkpoint_receipt.model_file_sha256
        or trace_receipt.checkpoint_fold_index
        != checkpoint_receipt.training_fold_index
        or identity.source_evidence_sha256 != source.receipt_sha256
        or calibration.retrospective_data_receipt_sha256
        != identity.receipt_sha256
        or view_receipt.chronology_receipt_sha256 != identity.receipt_sha256
        or trace_receipt.chronology_receipt_sha256 != identity.receipt_sha256
        or trace_receipt.economic_execution_receipt_sha256
        != view_receipt.receipt_sha256
        or cohort_receipt.checkpoint_sha256 != checkpoint_receipt.model_file_sha256
        or cohort_receipt.checkpoint_fold_index
        != checkpoint_receipt.training_fold_index
        or cohort_receipt.chronology_receipt_sha256 != identity.receipt_sha256
        or cohort_receipt.economic_execution_receipt_sha256
        != view_receipt.receipt_sha256
    ):
        raise Top2000M03RV72026ExecutionError(
            "execution artifact receipt graph does not close"
        )
    return Top2000M03RV72026LoadedExecutionArtifact(
        evaluation_inputs=inputs,
        cohort_trajectories=cohort_trajectories,
        execution_receipt=execution_receipt,
        checkpoint_load_receipt=checkpoint_receipt,
        factor_calibration=calibration,
        economic_execution_receipt=view_receipt,
        cuda_proof=cuda_proof,
        source_evidence=source,
        chronology_identity=identity,
        artifact_file_sha256=actual_sha256,
    )


__all__ = [
    "TOP2000_M03R_V7_2026_CUDA_PROOF_SCHEMA",
    "TOP2000_M03R_V7_2026_EXECUTION_ARTIFACT_SCHEMA",
    "TOP2000_M03R_V7_2026_EXECUTION_GEOMETRY",
    "TOP2000_M03R_V7_2026_EXECUTION_RECEIPT_SCHEMA",
    "TOP2000_M03R_V7_2026_MAXIMUM_GPU_MEMORY_BYTES",
    "TOP2000_M03R_V7_2026_MINIMUM_GPU_MEMORY_BYTES",
    "TOP2000_M03R_V7_2026_REQUIRED_COMPUTE_CAPABILITY",
    "TOP2000_M03R_V7_2026_REQUIRED_GPU_NAME",
    "Top2000M03RV72026CudaExecutionProof",
    "Top2000M03RV72026ExecutionArtifactBinding",
    "Top2000M03RV72026ExecutionError",
    "Top2000M03RV72026ExecutionSession",
    "Top2000M03RV72026LoadedExecutionArtifact",
    "Top2000M03RV72026SingleCheckpointExecutionReceipt",
    "load_top2000_m03r_v7_seed17_2026_execution_artifact",
    "prepare_top2000_m03r_v7_seed17_2026_execution_session",
    "run_top2000_m03r_v7_seed17_2026_single_checkpoint",
    "run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session",
]
