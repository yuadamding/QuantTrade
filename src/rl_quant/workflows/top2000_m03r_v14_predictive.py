"""Governed two-H100 worker for the M03R-v14 predictive-only panel.

Each indexed completion owns one setting and two NCCL ranks.  The worker trains
six chronological folds from the common packaged parameter bytes, publishes an
immutable checkpoint, destroys the training objects, reloads the exact file,
and only then evaluates the qualification tail.  Predictive-gate failure is a
valid scientific terminal; economic training and 2026 access are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_PROTOCOL_SHA256,
    M03R_V14_SELECTED_HORIZON_SESSIONS,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    optimizer_state_sha256,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    load_m03r_v9_projector_manifest,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    load_top2000_m03r_v9_risk_source,
)
from rl_quant.training.top2000_m03r_v14_checkpoint import (
    load_m03r_v14_alpha_checkpoint_for_evaluation,
    write_immutable_m03r_v14_alpha_checkpoint,
)
from rl_quant.training.top2000_m03r_v14_fold import (
    render_m03r_v14_fold_geometries,
)
from rl_quant.training.top2000_m03r_v14_initial_state import (
    load_m03r_v14_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v14_package import (
    M03RV14PackagePlan,
    load_m03r_v14_execution_authorization,
    load_m03r_v14_package_plan,
)
from rl_quant.training.top2000_m03r_v14_policy import (
    Top2000M03RV14PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v14_predictive_worker import (
    M03RV14PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v14_preflight import (
    load_m03r_v14_structural_preflight,
)
from rl_quant.training.top2000_m03r_v14_pretraining_optimizer import (
    build_m03r_v14_optimizer,
)
from rl_quant.training.top2000_m03r_v14_qualification_runtime import (
    M03RV14FoldQualificationResult,
    build_m03r_v14_qualification_risk_state,
    run_m03r_v14_fold_qualification,
)
from rl_quant.training.top2000_m03r_v14_selection import (
    build_m03r_v14_bootstrap_plan,
    qualify_m03r_v14_predictive_candidate,
)
from rl_quant.training.top2000_m03r_v14_training_runtime import (
    run_m03r_v14_pretraining_fold_update,
)

M03R_V14_STARTUP_SCHEMA = "rl-quant.top2000-dev.m03r-v14-two-h100-startup-v1"
M03R_V14_CAPACITY_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-two-h100-capacity-terminal-v1"
)
M03R_V14_FOLD_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-predictive-fold-terminal-v1"
)
M03R_V14_WORKER_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-predictive-worker-terminal-v1"
)
M03R_V14_WORKER_ERROR_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-predictive-worker-error-v1"
)
M03R_V14_PROGRESS_SCHEMA = "rl-quant.top2000-dev.m03r-v14-progress-v1"


class M03RV14PredictiveWorkflowError(RuntimeError):
    """The exact v14 package, distributed runtime, or output drifted."""


def _canonical_bytes(value: Any) -> bytes:
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
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def _write_immutable_torch(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise M03RV14PredictiveWorkflowError("v14 artifact target already exists")
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _distributed_context() -> tuple[int, int, int, torch.device, bool]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    except (KeyError, ValueError) as exc:
        raise M03RV14PredictiveWorkflowError(
            "v14 worker requires exact torchrun rank variables"
        ) from exc
    if (
        world_size != 2
        or rank not in range(2)
        or local_rank not in range(2)
        or restart_count != 0
    ):
        raise M03RV14PredictiveWorkflowError(
            "v14 requires one fresh non-resumed two-rank process group"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise M03RV14PredictiveWorkflowError(
            "v14 requires exactly two visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if (
        torch.cuda.get_device_name(device) != "NVIDIA H100 80GB HBM3"
        or not 79 * 1024**3 <= properties.total_memory <= 81 * 1024**3
        or (properties.major, properties.minor) != (9, 0)
    ):
        raise M03RV14PredictiveWorkflowError(
            "each v14 rank requires one NVIDIA H100 80GB HBM3"
        )
    owns = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        owns = True
    if dist.get_world_size() != 2 or dist.get_rank() != rank:
        raise M03RV14PredictiveWorkflowError("v14 NCCL identity drifted")
    return rank, local_rank, world_size, device, owns


def _gather(value: Any, world_size: int) -> list[Any]:
    rows: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(rows, value)
    return rows


def _broadcast(value: Any, rank: int) -> Any:
    rows = [value if rank == 0 else None]
    dist.broadcast_object_list(rows, src=0)
    return rows[0]


def resolve_m03r_v14_completion_index(value: int | None) -> int:
    if value is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        try:
            value = None if raw is None else int(raw)
        except ValueError as exc:
            raise M03RV14PredictiveWorkflowError(
                "JOB_COMPLETION_INDEX must be an integer"
            ) from exc
    if not isinstance(value, int) or isinstance(value, bool) or value not in range(2):
        raise M03RV14PredictiveWorkflowError(
            "v14 completion index must be 0 or 1"
        )
    return value


def _resolve_worker(
    package: M03RV14PackagePlan,
    completion_index: int,
) -> M03RV14PredictiveWorkerPlan:
    worker = package.panel.workers[completion_index]
    worker.validate()
    if worker.setting_index != completion_index:
        raise M03RV14PredictiveWorkflowError("v14 completion mapping drifted")
    return worker


def _validate_runtime_package_members(
    package_plan_path: str | Path,
    package: M03RV14PackagePlan,
) -> None:
    package_root = Path(package_plan_path).parent.parent
    expected = {
        package_root / "source.tar": package.artifacts.source_archive_sha256,
        package_root / "source-manifest.json": (
            package.artifacts.source_manifest_sha256
        ),
        package_root / "source" / "uv.lock": package.artifacts.dependency_lock_sha256,
        package_root
        / "source"
        / "src/rl_quant/workflows/top2000_m03r_v14_predictive.py": (
            package.artifacts.worker_source_sha256
        ),
        package_root / "cache" / "top2000-daily-bars.pt": (
            package.artifacts.cache_artifact_sha256
        ),
        package_root / "cache" / "cache-manifest.json": (
            package.artifacts.cache_manifest_sha256
        ),
        package_root / "risk" / "risk-exposures.pt": (
            package.artifacts.risk_artifact_sha256
        ),
        package_root / "risk" / "risk-source-manifest.json": (
            package.artifacts.risk_source_manifest_file_sha256
        ),
        package_root / "risk" / "projector-manifest.json": (
            package.artifacts.projector_manifest_file_sha256
        ),
        package_root / "model" / "common-initial-parameter-state.pt": (
            package.artifacts.initial_parameter_state_file_sha256
        ),
        package_root / "plans" / "real-data-structural-preflight.json": (
            package.artifacts.structural_preflight_file_sha256
        ),
    }
    for path, digest in expected.items():
        try:
            status = path.lstat()
        except OSError as exc:
            raise M03RV14PredictiveWorkflowError(
                "v14 runtime package member is unavailable"
            ) from exc
        if path.is_symlink() or not path.is_file() or status.st_size <= 0:
            raise M03RV14PredictiveWorkflowError(
                "v14 runtime package member is not a regular file"
            )
        if _file_sha256(path) != digest:
            raise M03RV14PredictiveWorkflowError(
                "v14 runtime package member hash drifted"
            )


def _new_policy(
    setting_index: int,
    device: torch.device,
) -> Top2000M03RV14PredictivePolicy:
    return Top2000M03RV14PredictivePolicy(
        setting_index,
        selected_horizon_sessions=M03R_V14_SELECTED_HORIZON_SESSIONS,
    ).to(device)


def _qualification_artifact(
    result: M03RV14FoldQualificationResult,
) -> dict[str, Any]:
    return {
        "schema": "rl-quant.top2000-dev.m03r-v14-fold-qualification-artifact-v1",
        "protocol_sha256": M03R_V14_PROTOCOL_SHA256,
        "loaded_checkpoint": asdict(result.loaded_checkpoint),
        "qualification_batch_receipt_sha256": result.batch.receipt_sha256,
        "trace": asdict(result.trace),
        "trace_sha256": result.trace.trace_sha256,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
    }


def _rank_update_row(result: Any) -> dict[str, Any]:
    return {
        "update_plan_sha256": result.update_plan.receipt_sha256,
        "paired_input_sha256": result.paired_input.receipt_sha256,
        "source_array_sha256": result.paired_input.source_array_sha256,
        "step_receipt_sha256": result.step.receipt_sha256,
        "model_state_after_sha256": result.step.model_state_after_sha256,
        "optimizer_state_after_sha256": result.step.optimizer_state_after_sha256,
        "target_residual_operator_root_sha256": (
            result.step.target_residual_operator_root_sha256
        ),
        "action_residual_operator_root_sha256": (
            result.step.action_residual_operator_root_sha256
        ),
        "local_origin_count": result.step.local_origin_count,
        "global_origin_count": result.step.global_origin_count,
        "distributed_gradient_synchronized": (
            result.step.distributed_gradient_synchronized
        ),
        "total_loss": result.step.total_loss,
        "ranking_loss": result.step.ranking_loss,
        "robust_regression_loss": result.step.robust_regression_loss,
        "distributional_loss": result.step.distributional_loss,
        "valid_asset_observation_count": (
            result.step.valid_asset_observation_count
        ),
        "raw_score_rms": result.step.raw_score_rms,
        "executable_score_rms": result.step.executable_score_rms,
        "target_rms": result.step.target_rms,
        "raw_to_executable_score_retention": (
            result.step.raw_to_executable_score_retention
        ),
        "encoder_gradient_norm_before_clip": (
            result.step.encoder_gradient_norm_before_clip
        ),
        "head_gradient_norm_before_clip": (
            result.step.head_gradient_norm_before_clip
        ),
        "encoder_gradient_clipped": result.step.encoder_gradient_clipped,
        "head_gradient_clipped": result.step.head_gradient_clipped,
    }


def _validate_gathered_update(rows: list[Any], world_size: int) -> None:
    if len(rows) != world_size or any(not isinstance(row, dict) for row in rows):
        raise M03RV14PredictiveWorkflowError("v14 rank update evidence is malformed")
    shared = (
        "update_plan_sha256",
        "paired_input_sha256",
        "source_array_sha256",
        "model_state_after_sha256",
        "optimizer_state_after_sha256",
        "global_origin_count",
    )
    if (
        any(len({row[name] for row in rows}) != 1 for name in shared)
        or len({row["step_receipt_sha256"] for row in rows}) != world_size
        or sum(int(row["local_origin_count"]) for row in rows)
        != int(rows[0]["global_origin_count"])
        or any(row["distributed_gradient_synchronized"] is not True for row in rows)
    ):
        raise M03RV14PredictiveWorkflowError(
            "v14 distributed state, input, or origin coverage diverged"
        )


def _emit_progress(
    *,
    worker: M03RV14PredictiveWorkerPlan,
    fold_index: int,
    completed_updates: int,
    planned_updates: int,
    row: dict[str, Any],
    device: torch.device,
) -> None:
    print(
        json.dumps(
            {
                "schema": M03R_V14_PROGRESS_SCHEMA,
                "observed_at_utc": datetime.now(UTC).isoformat(),
                "setting_index": worker.setting_index,
                "setting_id": worker.setting_id,
                "fold_index": fold_index,
                "completed_updates": completed_updates,
                "planned_updates": planned_updates,
                "model_state_sha256": row["model_state_after_sha256"],
                "optimizer_state_sha256": row["optimizer_state_after_sha256"],
                "total_loss": row["total_loss"],
                "ranking_loss": row["ranking_loss"],
                "robust_regression_loss": row["robust_regression_loss"],
                "distributional_loss": row["distributional_loss"],
                "valid_asset_observation_count": row[
                    "valid_asset_observation_count"
                ],
                "raw_score_rms": row["raw_score_rms"],
                "executable_score_rms": row["executable_score_rms"],
                "target_rms": row["target_rms"],
                "raw_to_executable_score_retention": row[
                    "raw_to_executable_score_retention"
                ],
                "encoder_gradient_norm_before_clip": row[
                    "encoder_gradient_norm_before_clip"
                ],
                "head_gradient_norm_before_clip": row[
                    "head_gradient_norm_before_clip"
                ],
                "encoder_gradient_clipped": row["encoder_gradient_clipped"],
                "head_gradient_clipped": row["head_gradient_clipped"],
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "economic_optimizer_updates": 0,
                "outer_2026_accessed": False,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def run_m03r_v14_predictive_worker(
    package_plan_path: str | Path,
    authorization_path: str | Path,
    *,
    expected_package_plan_file_sha256: str,
    expected_authorization_file_sha256: str,
    completion_index: int | None = None,
    startup_only: bool = False,
    startup_output_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Run one exact setting; a failed predictive gate is a valid terminal."""

    if startup_only and startup_output_root is None:
        raise M03RV14PredictiveWorkflowError(
            "startup-only capacity evidence requires a disjoint output root"
        )
    package = load_m03r_v14_package_plan(
        package_plan_path,
        expected_file_sha256=expected_package_plan_file_sha256,
    )
    authorization = load_m03r_v14_execution_authorization(
        authorization_path,
        expected_file_sha256=expected_authorization_file_sha256,
        package=package,
    )
    _validate_runtime_package_members(package_plan_path, package)
    load_m03r_v14_structural_preflight(
        Path(package_plan_path).parent / "real-data-structural-preflight.json",
        expected_file_sha256=package.artifacts.structural_preflight_file_sha256,
        expected_receipt_sha256=(
            package.artifacts.structural_preflight_receipt_sha256
        ),
    )
    if authorization.package_plan_file_sha256 != expected_package_plan_file_sha256:
        raise M03RV14PredictiveWorkflowError(
            "v14 authorization and package-plan file disagree"
        )
    index = resolve_m03r_v14_completion_index(completion_index)
    worker = _resolve_worker(package, index)
    rank, local_rank, world_size, device, owns = _distributed_context()
    output = (
        Path(startup_output_root)
        if startup_only and startup_output_root is not None
        else Path(worker.output_root)
    )
    try:
        if rank == 0:
            output.mkdir(mode=0o750, parents=True, exist_ok=False)
        dist.barrier()
        properties = torch.cuda.get_device_properties(device)
        runtime_rows = _gather(
            {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "visible_device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(device),
                "device_total_memory": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "torch_cuda_version": torch.version.cuda,
                "nccl_version": list(
                    torch.cuda.nccl.version()  # type: ignore[no-untyped-call]
                ),
            },
            world_size,
        )
        startup_sha: str | None = None
        if rank == 0:
            startup_sha = _write_immutable_json(
                output / "two-h100-startup.json",
                {
                    "schema": M03R_V14_STARTUP_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "package_plan_file_sha256": expected_package_plan_file_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "mode": "capacity" if startup_only else "predictive",
                    "rank_runtime": runtime_rows,
                    "exact_h100_80gb_per_rank": True,
                    "nccl_process_group_initialized": True,
                    "restart_count": 0,
                    "economic_optimizer_updates": 0,
                    "outer_2026_accessed": False,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                },
            )
        startup_sha = _broadcast(startup_sha, rank)
        if not isinstance(startup_sha, str):
            raise M03RV14PredictiveWorkflowError("v14 startup receipt missing")
        if startup_only:
            capacity_terminal: dict[str, Any] | None = None
            _seed_everything(worker.seed)
            capacity_policy = _new_policy(worker.setting_index, device)
            load_m03r_v14_initial_parameter_state(
                worker.initial_parameter_state_path,
                capacity_policy,
                expected_file_sha256=worker.initial_parameter_state_file_sha256,
                expected_state_sha256=worker.initial_parameter_state_sha256,
                expected_architecture_sha256=(
                    worker.initial_parameter_architecture_sha256
                ),
            )
            if rank == 0:
                unsigned = {
                    "schema": M03R_V14_CAPACITY_TERMINAL_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "startup_file_sha256": startup_sha,
                    "world_size": 2,
                    "gpus_per_worker": 2,
                    "exact_h100_80gb_per_rank": True,
                    "nccl_process_group_initialized": True,
                    "initial_parameter_state_file_sha256": (
                        worker.initial_parameter_state_file_sha256
                    ),
                    "initial_parameter_state_sha256": (
                        worker.initial_parameter_state_sha256
                    ),
                    "initial_parameter_architecture_sha256": (
                        worker.initial_parameter_architecture_sha256
                    ),
                    "packaged_initial_state_loaded": True,
                    "training_performed": False,
                    "h100_capacity_evidence": True,
                    "economic_optimizer_updates": 0,
                    "outer_2026_accessed": False,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                }
                capacity_terminal = {
                    **unsigned,
                    "receipt_sha256": _sha256(unsigned),
                }
                _write_immutable_json(
                    output / "two-h100-capacity-terminal.json", capacity_terminal
                )
            dist.barrier()
            return capacity_terminal

        cache = load_verified_top2000_hold30_development_cache(
            worker.cache_path,
            expected_cache_sha256=worker.cache_sha256,
            acknowledgement=DEVELOPMENT_ACK,
        )
        if cache.daily_ohlcv.shape[0] != TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS:
            raise M03RV14PredictiveWorkflowError("v14 cache geometry drifted")
        risk_source, written_risk = load_top2000_m03r_v9_risk_source(
            Path(worker.risk_source_manifest_path),
            expected_manifest_file_sha256=worker.risk_source_manifest_file_sha256,
        )
        projector, risk_binding = load_m03r_v9_projector_manifest(
            Path(worker.projector_manifest_path),
            expected_file_sha256=worker.projector_manifest_file_sha256,
        )
        if (
            projector.manifest_sha256 != worker.projector_manifest_sha256
            or risk_binding.binding_sha256 != worker.projector_binding_sha256
        ):
            raise M03RV14PredictiveWorkflowError("v14 risk identity drifted")

        geometries = render_m03r_v14_fold_geometries(
            TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        )
        fold_traces = []
        fold_terminal_file_sha256: list[str] = []
        for geometry in geometries:
            _seed_everything(worker.seed)
            training_policy = _new_policy(worker.setting_index, device)
            load_m03r_v14_initial_parameter_state(
                worker.initial_parameter_state_path,
                training_policy,
                expected_file_sha256=worker.initial_parameter_state_file_sha256,
                expected_state_sha256=worker.initial_parameter_state_sha256,
                expected_architecture_sha256=(
                    worker.initial_parameter_architecture_sha256
                ),
            )
            if model_state_sha256(training_policy) != worker.initial_parameter_state_sha256:
                raise M03RV14PredictiveWorkflowError(
                    "v14 fresh common initial parameter state drifted"
                )
            optimizer, partition = build_m03r_v14_optimizer(training_policy)
            update_evidence: list[list[Any]] = []
            for completed in range(geometry.optimizer_updates):
                result = run_m03r_v14_pretraining_fold_update(
                    cache,
                    worker,
                    package.schedule,
                    geometry,
                    risk_source,
                    written_risk,
                    training_policy,
                    optimizer,
                    partition,
                    completed_updates=completed,
                    distributed_rank=rank,
                    distributed_world_size=world_size,
                    device=device,
                )
                rows = _gather(_rank_update_row(result), world_size)
                _validate_gathered_update(rows, world_size)
                update_evidence.append(rows)
                if rank == 0:
                    _emit_progress(
                        worker=worker,
                        fold_index=geometry.fold_index,
                        completed_updates=completed + 1,
                        planned_updates=geometry.optimizer_updates,
                        row=rows[0],
                        device=device,
                    )
            if len(update_evidence) != geometry.optimizer_updates:
                raise M03RV14PredictiveWorkflowError(
                    "v14 fold update coverage drifted"
                )
            final_model_hashes = _gather(
                model_state_sha256(training_policy), world_size
            )
            final_optimizer_hashes = _gather(
                optimizer_state_sha256(optimizer), world_size
            )
            if (
                len(set(final_model_hashes)) != 1
                or len(set(final_optimizer_hashes)) != 1
            ):
                raise M03RV14PredictiveWorkflowError(
                    "v14 final rank model or optimizer states diverged"
                )
            source_root = _sha256(
                [rows[0]["source_array_sha256"] for rows in update_evidence]
            )
            target_root = _sha256(
                [
                    [row["target_residual_operator_root_sha256"] for row in rows]
                    for rows in update_evidence
                ]
            )
            action_root = _sha256(
                [
                    [row["action_residual_operator_root_sha256"] for row in rows]
                    for rows in update_evidence
                ]
            )
            update_evidence_root = _sha256(update_evidence)
            checkpoint_path = (
                output
                / "checkpoints"
                / (
                    f"fold-{geometry.fold_index:02d}-horizon-"
                    f"{M03R_V14_SELECTED_HORIZON_SESSIONS:02d}-update-"
                    f"{geometry.optimizer_updates:04d}.pt"
                )
            )
            checkpoint_sha: str | None = None
            if rank == 0:
                checkpoint_sha = write_immutable_m03r_v14_alpha_checkpoint(
                    checkpoint_path,
                    training_policy,
                    fold_index=geometry.fold_index,
                    completed_updates=geometry.optimizer_updates,
                    episode_schedule_sha256=package.schedule.receipt_sha256,
                    target_residual_operator_root_sha256=target_root,
                    action_residual_operator_root_sha256=action_root,
                    source_array_sha256=source_root,
                    asset_axis_sha256=cache.action_hash,
                )
            checkpoint_sha = _broadcast(checkpoint_sha, rank)
            if not isinstance(checkpoint_sha, str):
                raise M03RV14PredictiveWorkflowError(
                    "v14 checkpoint publication failed"
                )
            dist.barrier()
            del optimizer, partition, training_policy
            torch.cuda.empty_cache()

            risk_state = build_m03r_v14_qualification_risk_state(
                cache,
                geometry,
                risk_source,
                risk_binding,
                projector,
                device=device,
            )
            risk_hashes = _gather(risk_state.state_sha256, world_size)
            if len(set(risk_hashes)) != 1:
                raise M03RV14PredictiveWorkflowError(
                    "v14 qualified risk state diverged between ranks"
                )
            loaded_policy = _new_policy(worker.setting_index, device)
            loaded = load_m03r_v14_alpha_checkpoint_for_evaluation(
                checkpoint_path,
                expected_file_sha256=checkpoint_sha,
                expected_setting_index=worker.setting_index,
                expected_fold_index=geometry.fold_index,
                expected_completed_updates=geometry.optimizer_updates,
                expected_episode_schedule_sha256=package.schedule.receipt_sha256,
                expected_target_residual_operator_root_sha256=target_root,
                expected_action_residual_operator_root_sha256=action_root,
                expected_source_array_sha256=source_root,
                expected_asset_axis_sha256=cache.action_hash,
                policy=loaded_policy,
            )
            fold_qualification = run_m03r_v14_fold_qualification(
                cache,
                geometry,
                risk_source,
                risk_state,
                loaded_policy,
                loaded,
                device=device,
            )
            trace_hashes = _gather(
                fold_qualification.trace.trace_sha256,
                world_size,
            )
            if len(set(trace_hashes)) != 1:
                raise M03RV14PredictiveWorkflowError(
                    "v14 qualification trace diverged between ranks"
                )
            artifact_path = (
                output
                / "fold-artifacts"
                / f"fold-{geometry.fold_index:02d}-horizon-03.pt"
            )
            artifact_sha: str | None = None
            if rank == 0:
                artifact_sha = _write_immutable_torch(
                    artifact_path,
                    _qualification_artifact(fold_qualification),
                )
            artifact_sha = _broadcast(artifact_sha, rank)
            if not isinstance(artifact_sha, str):
                raise M03RV14PredictiveWorkflowError(
                    "v14 qualification artifact publication failed"
                )
            if rank == 0:
                fold_traces.append(fold_qualification.trace)
                unsigned_fold = {
                    "schema": M03R_V14_FOLD_TERMINAL_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "fold_index": geometry.fold_index,
                    "completed_updates": geometry.optimizer_updates,
                    "training_epoch_count": (
                        geometry.optimizer_updates // geometry.training_block_count
                    ),
                    "model_state_sha256": final_model_hashes[0],
                    "optimizer_state_sha256": final_optimizer_hashes[0],
                    "training_update_evidence_root_sha256": update_evidence_root,
                    "training_source_array_root_sha256": source_root,
                    "training_target_operator_root_sha256": target_root,
                    "training_action_operator_root_sha256": action_root,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_file_sha256": checkpoint_sha,
                    "qualification_artifact_path": str(artifact_path),
                    "qualification_artifact_file_sha256": artifact_sha,
                    "qualification_trace_sha256": (
                        fold_qualification.trace.trace_sha256
                    ),
                    "qualification_batch_receipt_sha256": (
                        fold_qualification.batch.receipt_sha256
                    ),
                    "fold_risk_state_sha256": risk_state.state_sha256,
                    "qualification_evaluated_only_after_checkpoint_reload": True,
                    "economic_optimizer_updates": 0,
                    "outer_2026_accessed": False,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                }
                fold_terminal = {
                    **unsigned_fold,
                    "receipt_sha256": _sha256(unsigned_fold),
                }
                fold_terminal_file_sha256.append(
                    _write_immutable_json(
                        output
                        / "receipts"
                        / f"fold-{geometry.fold_index:02d}-terminal.json",
                        fold_terminal,
                    )
                )
            dist.barrier()
            del loaded_policy, risk_state
            torch.cuda.empty_cache()

        terminal: dict[str, Any] | None = None
        if rank == 0:
            traces = tuple(fold_traces)
            bootstrap = build_m03r_v14_bootstrap_plan(
                tuple(row.origin_indices for row in traces)
            )
            predictive_qualification = qualify_m03r_v14_predictive_candidate(
                traces,
                bootstrap,
            )
            unsigned_terminal = {
                "schema": M03R_V14_WORKER_TERMINAL_SCHEMA,
                "package_plan_sha256": package.package_plan_sha256,
                "package_plan_file_sha256": expected_package_plan_file_sha256,
                "authorization_receipt_sha256": authorization.receipt_sha256,
                "worker_plan_sha256": worker.receipt_sha256,
                "startup_file_sha256": startup_sha,
                "setting_index": worker.setting_index,
                "setting_id": worker.setting_id,
                "fold_terminal_file_sha256": fold_terminal_file_sha256,
                "bootstrap_plan": asdict(bootstrap),
                "bootstrap_plan_sha256": bootstrap.receipt_sha256,
                "predictive_qualification": asdict(predictive_qualification),
                "predictive_qualification_sha256": (
                    predictive_qualification.receipt_sha256
                ),
                "selected_horizon": (
                    M03R_V14_SELECTED_HORIZON_SESSIONS
                    if predictive_qualification.passed
                    else None
                ),
                "predictive_gate_passed": predictive_qualification.passed,
                "economic_generation_may_be_minted": (
                    predictive_qualification.economic_generation_may_be_minted
                ),
                "economic_panel_authorized": False,
                "economic_optimizer_updates": 0,
                "outer_2026_accessed": False,
                "world_size": 2,
                "gpus_per_worker": 2,
                "h100_capacity_evidence": True,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
            terminal = {
                **unsigned_terminal,
                "receipt_sha256": _sha256(unsigned_terminal),
            }
            _write_immutable_json(output / "predictive-terminal.json", terminal)
        dist.barrier()
        return terminal
    except BaseException as exc:
        if rank == 0 and output.is_dir() and not (output / "worker-error.json").exists():
            _write_immutable_json(
                output / "worker-error.json",
                {
                    "schema": M03R_V14_WORKER_ERROR_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "mode": "capacity" if startup_only else "predictive",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "checkpoint_published_after_failure": False,
                    "economic_optimizer_updates": 0,
                    "outer_2026_accessed": False,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                },
            )
        raise
    finally:
        if owns and dist.is_initialized():
            dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--execution-authorization-file-sha256", required=True)
    parser.add_argument("--completion-index", type=int)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--startup-output-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_m03r_v14_predictive_worker(
            args.package_plan,
            args.execution_authorization,
            expected_package_plan_file_sha256=args.package_plan_file_sha256,
            expected_authorization_file_sha256=(
                args.execution_authorization_file_sha256
            ),
            completion_index=args.completion_index,
            startup_only=args.startup_only,
            startup_output_root=args.startup_output_root,
        )
    except BaseException as exc:
        rank = os.environ.get("RANK")
        output_raw = args.startup_output_root if args.startup_only else None
        if output_raw is not None and rank == "0":
            output = Path(output_raw)
            if output.is_dir() and not (output / "worker-error.json").exists():
                _write_immutable_json(
                    output / "worker-error.json",
                    {
                        "schema": M03R_V14_WORKER_ERROR_SCHEMA,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "economic_optimizer_updates": 0,
                        "outer_2026_accessed": False,
                    },
                )
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V14_CAPACITY_TERMINAL_SCHEMA",
    "M03R_V14_FOLD_TERMINAL_SCHEMA",
    "M03R_V14_PROGRESS_SCHEMA",
    "M03R_V14_STARTUP_SCHEMA",
    "M03R_V14_WORKER_ERROR_SCHEMA",
    "M03R_V14_WORKER_TERMINAL_SCHEMA",
    "M03RV14PredictiveWorkflowError",
    "main",
    "resolve_m03r_v14_completion_index",
    "run_m03r_v14_predictive_worker",
]
