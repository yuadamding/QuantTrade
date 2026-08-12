"""Governed two-H100 worker for the corrected M03R-v11 predictive panel.

The entrypoint performs exactly 64 predictive updates for one of three paired
settings, writes both horizon checkpoints before opening the qualification
tail, reloads the exact checkpoint bytes, and performs no economic training or
2026 evaluation.  A failed predictive gate is a valid scientific terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03RV9HorizonBinding,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v9_fold import (
    build_m03r_v9_qualification_risk_state,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    build_m03r_v9_alpha_pretraining_optimizer,
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
from rl_quant.training.top2000_m03r_v11_checkpoint import (
    load_m03r_v11_alpha_checkpoint_for_evaluation,
    write_immutable_m03r_v11_alpha_checkpoint,
)
from rl_quant.training.top2000_m03r_v11_fold_qualification import (
    evaluate_m03r_v11_loaded_qualification_fold,
)
from rl_quant.training.top2000_m03r_v11_initial_state import (
    load_m03r_v11_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v11_package import (
    M03RV11PackagePlan,
    load_m03r_v11_execution_authorization,
    load_m03r_v11_package_plan,
)
from rl_quant.training.top2000_m03r_v11_predictive_worker import (
    M03RV11PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v11_selection import (
    M03RV11PredictiveQualification,
    build_m03r_v11_bootstrap_plan,
    qualify_m03r_v11_predictive_candidate,
)
from rl_quant.training.top2000_m03r_v11_training_runtime import (
    run_m03r_v11_pretraining_fold_update,
)

M03R_V11_STARTUP_SCHEMA = "rl-quant.top2000-dev.m03r-v11-two-h100-startup-v1"
M03R_V11_CAPACITY_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-two-h100-capacity-terminal-v1"
)
M03R_V11_FOLD_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-predictive-fold-terminal-v1"
)
M03R_V11_WORKER_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-predictive-worker-terminal-v1"
)
M03R_V11_WORKER_ERROR_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-predictive-worker-error-v1"
)


class M03RV11PredictiveWorkflowError(RuntimeError):
    """The exact package, distributed runtime, or worker output drifted."""


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
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _write_immutable_torch(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise M03RV11PredictiveWorkflowError("v11 artifact target already exists")
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


def _distributed_context() -> tuple[int, int, int, torch.device, bool]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    except (KeyError, ValueError) as exc:
        raise M03RV11PredictiveWorkflowError(
            "v11 worker requires exact torchrun rank variables"
        ) from exc
    if (
        world_size != 2
        or rank not in range(2)
        or local_rank not in range(2)
        or restart_count != 0
    ):
        raise M03RV11PredictiveWorkflowError(
            "v11 requires one fresh non-resumed two-rank process group"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise M03RV11PredictiveWorkflowError(
            "v11 requires exactly two visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if (
        torch.cuda.get_device_name(device) != "NVIDIA H100 80GB HBM3"
        or not 79 * 1024**3 <= properties.total_memory <= 81 * 1024**3
        or (properties.major, properties.minor) != (9, 0)
    ):
        raise M03RV11PredictiveWorkflowError(
            "each v11 rank requires one NVIDIA H100 80GB HBM3"
        )
    owns = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        owns = True
    if dist.get_world_size() != 2 or dist.get_rank() != rank:
        raise M03RV11PredictiveWorkflowError("v11 NCCL identity drifted")
    return rank, local_rank, world_size, device, owns


def _gather(value: Any, world_size: int) -> list[Any]:
    rows: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(rows, value)
    return rows


def _broadcast(value: Any, rank: int) -> Any:
    rows = [value if rank == 0 else None]
    dist.broadcast_object_list(rows, src=0)
    return rows[0]


def resolve_m03r_v11_completion_index(value: int | None) -> int:
    if value is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        try:
            value = None if raw is None else int(raw)
        except ValueError as exc:
            raise M03RV11PredictiveWorkflowError(
                "JOB_COMPLETION_INDEX must be an integer"
            ) from exc
    if not isinstance(value, int) or isinstance(value, bool) or value not in range(3):
        raise M03RV11PredictiveWorkflowError("v11 completion index must be 0, 1, or 2")
    return value


def _resolve_worker(
    package: M03RV11PackagePlan,
    completion_index: int,
) -> M03RV11PredictiveWorkerPlan:
    worker = package.panel.workers[completion_index]
    worker.validate()
    if worker.setting_index != completion_index:
        raise M03RV11PredictiveWorkflowError("v11 completion mapping drifted")
    return worker


def _validate_runtime_package_members(
    package_plan_path: str | Path,
    package: M03RV11PackagePlan,
) -> None:
    plan_path = Path(package_plan_path)
    package_root = plan_path.parent.parent
    expected = {
        package_root / "source.tar": package.artifacts.source_archive_sha256,
        package_root / "source-manifest.json": (
            package.artifacts.source_manifest_sha256
        ),
        package_root / "source" / "uv.lock": (package.artifacts.dependency_lock_sha256),
        package_root
        / "source"
        / "src/rl_quant/workflows/top2000_m03r_v11_predictive.py": (
            package.artifacts.worker_source_sha256
        ),
        package_root / "cache-manifest.json": package.artifacts.cache_manifest_sha256,
        package_root / "risk" / "risk-exposures.pt": (
            package.artifacts.risk_artifact_sha256
        ),
        package_root / "model" / "common-initial-parameter-state.pt": (
            package.artifacts.initial_parameter_state_file_sha256
        ),
    }
    for path, digest in expected.items():
        try:
            status = path.lstat()
        except OSError as exc:
            raise M03RV11PredictiveWorkflowError(
                "v11 runtime package member is unavailable"
            ) from exc
        if path.is_symlink() or not path.is_file() or status.st_size <= 0:
            raise M03RV11PredictiveWorkflowError(
                "v11 runtime package member is not a regular file"
            )
        if _file_sha256(path) != digest:
            raise M03RV11PredictiveWorkflowError(
                "v11 runtime package member hash drifted"
            )


def _new_policy(horizon: int, device: torch.device) -> Top2000M03RV9PredictivePolicy:
    binding = M03RV9HorizonBinding(horizon, horizon, horizon)
    return Top2000M03RV9PredictivePolicy(0, binding).to(device)


def _qualification_artifact(lineage: Any) -> dict[str, Any]:
    return {
        "schema": "rl-quant.top2000-dev.m03r-v11-fold-qualification-artifact-v1",
        "protocol_sha256": M03R_V11_PROTOCOL_SHA256,
        "loaded_checkpoint": asdict(lineage.loaded_checkpoint),
        "fold_evidence": asdict(lineage.fold_evidence),
        "evaluation_trace_sha256": lineage.evaluation_trace_sha256,
        "qualification_source_array_sha256": (
            lineage.qualification_source_array_sha256
        ),
        "qualification_asset_axis_sha256": (lineage.qualification_asset_axis_sha256),
        "qualification_residual_operator_root_sha256": (
            lineage.qualification_residual_operator_root_sha256
        ),
        "lineage_receipt_sha256": lineage.receipt_sha256,
    }


def _select_horizon(
    qualifications: tuple[M03RV11PredictiveQualification, ...],
) -> M03RV11PredictiveQualification | None:
    passed = tuple(row for row in qualifications if row.passed)
    if not passed:
        return None
    return max(
        passed,
        key=lambda row: (row.net_active_return_10bp_lcb, row.horizon_sessions == 30),
    )


def run_m03r_v11_predictive_worker(
    package_plan_path: str | Path,
    authorization_path: str | Path,
    *,
    expected_package_plan_file_sha256: str,
    expected_authorization_file_sha256: str,
    completion_index: int | None = None,
    startup_only: bool = False,
    startup_output_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Run one exact setting; predictive-gate failure still exits successfully."""

    if startup_only and startup_output_root is None:
        raise M03RV11PredictiveWorkflowError(
            "startup-only capacity evidence requires a disjoint output root"
        )
    package = load_m03r_v11_package_plan(
        package_plan_path,
        expected_file_sha256=expected_package_plan_file_sha256,
    )
    authorization = load_m03r_v11_execution_authorization(
        authorization_path,
        expected_file_sha256=expected_authorization_file_sha256,
        package=package,
    )
    _validate_runtime_package_members(package_plan_path, package)
    if authorization.package_plan_file_sha256 != expected_package_plan_file_sha256:
        raise M03RV11PredictiveWorkflowError(
            "v11 authorization and package-plan file disagree"
        )
    index = resolve_m03r_v11_completion_index(completion_index)
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
                    "schema": M03R_V11_STARTUP_SCHEMA,
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
            raise M03RV11PredictiveWorkflowError("v11 startup receipt missing")
        if startup_only:
            capacity_terminal: dict[str, Any] | None = None
            _seed_everything(worker.seed)
            capacity_policy = _new_policy(30, device)
            load_m03r_v11_initial_parameter_state(
                worker.initial_parameter_state_path,
                capacity_policy,
                expected_file_sha256=worker.initial_parameter_state_file_sha256,
                expected_state_sha256=worker.initial_parameter_state_sha256,
            )
            if rank == 0:
                unsigned = {
                    "schema": M03R_V11_CAPACITY_TERMINAL_SCHEMA,
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
            raise M03RV11PredictiveWorkflowError("v11 cache geometry drifted")
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
            raise M03RV11PredictiveWorkflowError("v11 risk identity drifted")

        folds = render_top2000_m03r_v7_development_folds(
            TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        )
        evidence_by_horizon: dict[int, list[Any]] = {21: [], 30: []}
        fold_file_sha256: list[str] = []
        for fold in folds:
            _seed_everything(worker.seed)
            training_policy = _new_policy(30, device)
            load_m03r_v11_initial_parameter_state(
                worker.initial_parameter_state_path,
                training_policy,
                expected_file_sha256=worker.initial_parameter_state_file_sha256,
                expected_state_sha256=worker.initial_parameter_state_sha256,
            )
            if (
                model_state_sha256(training_policy)
                != worker.initial_parameter_state_sha256
            ):
                raise M03RV11PredictiveWorkflowError(
                    "v11 fresh common initial parameter state drifted"
                )
            optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(
                training_policy
            )
            last_result = None
            for completed in range(worker.predictive_optimizer_updates):
                last_result = run_m03r_v11_pretraining_fold_update(
                    cache,
                    worker,
                    package.schedule,
                    fold,
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
            if last_result is None:
                raise M03RV11PredictiveWorkflowError("v11 produced no optimizer step")
            model_hashes = _gather(model_state_sha256(training_policy), world_size)
            optimizer_hashes = _gather(optimizer_state_sha256(optimizer), world_size)
            paired_hashes = _gather(last_result.paired_input.receipt_sha256, world_size)
            source_hashes = _gather(
                last_result.step_receipt.source_array_sha256, world_size
            )
            residual_roots = _gather(
                last_result.step_receipt.residual_operator_root_sha256,
                world_size,
            )
            step_hashes = _gather(last_result.step_receipt.receipt_sha256, world_size)
            if (
                len(set(model_hashes)) != 1
                or len(set(optimizer_hashes)) != 1
                or len(set(paired_hashes)) != 1
                or len(set(source_hashes)) != 1
                or len(set(step_hashes)) != world_size
            ):
                raise M03RV11PredictiveWorkflowError(
                    "v11 rank state or paired-input lineage diverged"
                )
            training_residual_root = _sha256(
                {"rank_ordered_residual_operator_roots": residual_roots}
            )
            checkpoint_rows: dict[int, tuple[Path, str]] = {}
            for horizon in (21, 30):
                checkpoint_path = (
                    output
                    / "checkpoints"
                    / f"fold-{fold.fold_index:02d}-horizon-{horizon:02d}-update-0064.pt"
                )
                checkpoint_sha: str | None = None
                if rank == 0:
                    checkpoint_sha = write_immutable_m03r_v11_alpha_checkpoint(
                        checkpoint_path,
                        training_policy,
                        setting_index=worker.setting_index,
                        fold_index=fold.fold_index,
                        selected_horizon_sessions=horizon,
                        episode_schedule_sha256=package.schedule.receipt_sha256,
                        residual_operator_root_sha256=training_residual_root,
                        source_array_sha256=source_hashes[0],
                        asset_axis_sha256=cache.action_hash,
                    )
                checkpoint_sha = _broadcast(checkpoint_sha, rank)
                if not isinstance(checkpoint_sha, str):
                    raise M03RV11PredictiveWorkflowError(
                        "v11 checkpoint publication failed"
                    )
                checkpoint_rows[horizon] = (checkpoint_path, checkpoint_sha)
            dist.barrier()

            risk_state = build_m03r_v9_qualification_risk_state(
                cache,
                fold,
                risk_source,
                risk_binding,
                projector,
                device=device,
            )
            risk_hashes = _gather(risk_state.state_sha256, world_size)
            if len(set(risk_hashes)) != 1:
                raise M03RV11PredictiveWorkflowError(
                    "v11 qualified risk state diverged between ranks"
                )
            horizon_rows: dict[str, Any] = {}
            for horizon in (21, 30):
                checkpoint_path, checkpoint_sha = checkpoint_rows[horizon]
                loaded_policy = _new_policy(horizon, device)
                loaded = load_m03r_v11_alpha_checkpoint_for_evaluation(
                    checkpoint_path,
                    expected_file_sha256=checkpoint_sha,
                    expected_setting_index=worker.setting_index,
                    expected_fold_index=fold.fold_index,
                    expected_selected_horizon_sessions=horizon,
                    expected_episode_schedule_sha256=package.schedule.receipt_sha256,
                    expected_residual_operator_root_sha256=training_residual_root,
                    expected_source_array_sha256=source_hashes[0],
                    expected_asset_axis_sha256=cache.action_hash,
                    policy=loaded_policy,
                )
                lineage = evaluate_m03r_v11_loaded_qualification_fold(
                    cache,
                    worker,
                    fold,
                    risk_source,
                    risk_state,
                    loaded_policy,
                    loaded,
                    device=device,
                )
                lineage_hashes = _gather(lineage.receipt_sha256, world_size)
                if len(set(lineage_hashes)) != 1:
                    raise M03RV11PredictiveWorkflowError(
                        "v11 qualification lineage diverged between ranks"
                    )
                artifact_path = (
                    output
                    / "fold-artifacts"
                    / f"fold-{fold.fold_index:02d}-horizon-{horizon:02d}.pt"
                )
                artifact_sha: str | None = None
                if rank == 0:
                    artifact_sha = _write_immutable_torch(
                        artifact_path, _qualification_artifact(lineage)
                    )
                artifact_sha = _broadcast(artifact_sha, rank)
                if not isinstance(artifact_sha, str):
                    raise M03RV11PredictiveWorkflowError(
                        "v11 qualification artifact publication failed"
                    )
                horizon_rows[str(horizon)] = {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_file_sha256": checkpoint_sha,
                    "model_state_sha256": loaded.model_state_sha256,
                    "qualification_artifact_path": str(artifact_path),
                    "qualification_artifact_file_sha256": artifact_sha,
                    "qualification_lineage_sha256": lineage.receipt_sha256,
                    "fold_evidence_sha256": lineage.fold_evidence.receipt_sha256,
                    "qualification_source_array_sha256": (
                        lineage.qualification_source_array_sha256
                    ),
                    "qualification_residual_operator_root_sha256": (
                        lineage.qualification_residual_operator_root_sha256
                    ),
                    "fold_risk_state_sha256": risk_state.state_sha256,
                }
                evidence_by_horizon[horizon].append(lineage.fold_evidence)
                del loaded_policy
            if rank == 0:
                unsigned_fold = {
                    "schema": M03R_V11_FOLD_TERMINAL_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "fold_index": fold.fold_index,
                    "completed_updates": 64,
                    "model_state_sha256": model_hashes[0],
                    "optimizer_state_sha256": optimizer_hashes[0],
                    "paired_input_receipt_sha256": paired_hashes[0],
                    "rank_step_receipt_sha256": step_hashes,
                    "training_source_array_sha256": source_hashes[0],
                    "training_residual_operator_root_sha256": training_residual_root,
                    "horizon_candidates": horizon_rows,
                    "qualification_evaluated_only_after_checkpoint_publication": True,
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
                fold_file_sha256.append(
                    _write_immutable_json(
                        output
                        / "receipts"
                        / f"fold-{fold.fold_index:02d}-terminal.json",
                        fold_terminal,
                    )
                )
            dist.barrier()
            del risk_state, optimizer, training_policy

        terminal: dict[str, Any] | None = None
        if rank == 0:
            score_rows = tuple(
                row.score_session_index for row in evidence_by_horizon[21]
            )
            bootstrap = build_m03r_v11_bootstrap_plan(
                score_rows,
                bootstrap_seed=worker.seed,
            )
            qualifications = tuple(
                qualify_m03r_v11_predictive_candidate(
                    tuple(evidence_by_horizon[horizon]), bootstrap
                )
                for horizon in (21, 30)
            )
            selected = _select_horizon(qualifications)
            unsigned_terminal = {
                "schema": M03R_V11_WORKER_TERMINAL_SCHEMA,
                "package_plan_sha256": package.package_plan_sha256,
                "package_plan_file_sha256": expected_package_plan_file_sha256,
                "authorization_receipt_sha256": authorization.receipt_sha256,
                "worker_plan_sha256": worker.receipt_sha256,
                "startup_file_sha256": startup_sha,
                "setting_index": worker.setting_index,
                "setting_id": worker.setting_id,
                "fold_terminal_file_sha256": fold_file_sha256,
                "bootstrap_plan": asdict(bootstrap),
                "bootstrap_plan_sha256": bootstrap.receipt_sha256,
                "horizon_qualification": [
                    {**asdict(row), "receipt_sha256": row.receipt_sha256}
                    for row in qualifications
                ],
                "selected_horizon": (
                    None if selected is None else selected.horizon_sessions
                ),
                "selected_qualification_sha256": (
                    None if selected is None else selected.receipt_sha256
                ),
                "predictive_gate_passed": selected is not None,
                "economic_generation_may_be_minted": selected is not None,
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
        run_m03r_v11_predictive_worker(
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
                        "schema": M03R_V11_WORKER_ERROR_SCHEMA,
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
    "M03R_V11_CAPACITY_TERMINAL_SCHEMA",
    "M03R_V11_FOLD_TERMINAL_SCHEMA",
    "M03R_V11_STARTUP_SCHEMA",
    "M03R_V11_WORKER_ERROR_SCHEMA",
    "M03R_V11_WORKER_TERMINAL_SCHEMA",
    "M03RV11PredictiveWorkflowError",
    "main",
    "resolve_m03r_v11_completion_index",
    "run_m03r_v11_predictive_worker",
]
