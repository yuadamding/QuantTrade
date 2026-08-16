"""Two-rank capacity and predictive worker for the sealed M03R-v16 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes as _canonical,
    file_sha256 as _file_sha256,
    semantic_sha256 as _sha256,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    build_top2000_hold30_development_sequence_from_loaded_cache,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
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
from rl_quant.training.top2000_m03r_v16_capacity import (
    build_m03r_v16_capacity_terminal,
    run_m03r_v16_disposable_capacity_rank,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    M03RV16QualificationActivation,
    M03RV16TrainingActivation,
    load_m03r_v16_qualification_activation,
    load_m03r_v16_training_activation,
)
from rl_quant.training.top2000_m03r_v16_checkpoint import (
    load_m03r_v16_epoch_checkpoint_for_evaluation,
    write_immutable_m03r_v16_epoch_checkpoint,
)
from rl_quant.training.top2000_m03r_v16_evaluation_runtime import (
    build_m03r_v16_inner_validation_batch,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16FoldGeometry,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_fit import (
    build_m03r_v16_epoch_fit_payload,
    classify_m03r_v16_training_adequacy,
)
from rl_quant.training.top2000_m03r_v16_initial_state import (
    load_m03r_v16_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16PackagePlan,
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_pretraining_optimizer import (
    build_m03r_v16_optimizer,
)
from rl_quant.training.top2000_m03r_v16_qualification_runtime import (
    M03RV16FoldQualificationResult,
    build_m03r_v16_qualification_risk_state,
    issue_m03r_v16_terminal_checkpoint_authority,
    run_m03r_v16_fold_qualification,
)
from rl_quant.training.top2000_m03r_v16_selection import (
    build_m03r_v16_bootstrap_plan,
    qualify_m03r_v16_predictive_candidate,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    load_m03r_v16_structural_slab,
)
from rl_quant.training.top2000_m03r_v16_source import (
    verify_m03r_v16_source_tree,
)
from rl_quant.training.top2000_m03r_v16_training_runtime import (
    move_and_bind_m03r_v16_sequence,
    run_m03r_v16_pretraining_fold_update,
)
from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16InnerValidationReceipt,
    evaluate_m03r_v16_inner_validation_batch,
    select_m03r_v16_score_checkpoint,
)

M03R_V16_STARTUP_SCHEMA = "rl-quant.top2000-dev.m03r-v16-startup-v1"
M03R_V16_FOLD_TERMINAL_SCHEMA = "rl-quant.top2000-dev.m03r-v16-fold-terminal-v2"
M03R_V16_TRAINING_FOLD_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-fold-terminal-v1"
)
M03R_V16_TRAINING_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-terminal-v1"
)
M03R_V16_WORKER_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-qualification-worker-terminal-v2"
)
M03R_V16_WORKER_ERROR_SCHEMA = "rl-quant.top2000-dev.m03r-v16-worker-error-v1"
M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-fold-qualification-artifact-v1"
)


class M03RV16PredictiveWorkflowError(RuntimeError):
    """The sealed V16 worker, rank topology, or artifact lineage drifted."""


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    data = _canonical(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def _write_immutable_torch(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return _file_sha256(path)


def _read_immutable_json(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16PredictiveWorkflowError("V16 immutable input is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 64 * 1024**2:
            raise M03RV16PredictiveWorkflowError("V16 immutable input size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise M03RV16PredictiveWorkflowError("V16 immutable input changed while read")
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise M03RV16PredictiveWorkflowError("V16 immutable input hash drifted")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16PredictiveWorkflowError("V16 immutable input is malformed") from exc
    if not isinstance(payload, dict) or raw != _canonical(payload):
        raise M03RV16PredictiveWorkflowError("V16 immutable input is not canonical")
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _distributed_context() -> tuple[int, int, int, torch.device, bool]:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if any(name not in os.environ for name in required):
        raise M03RV16PredictiveWorkflowError(
            "V16 worker requires torchrun rank environment"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if (
        world_size != 2
        or rank not in {0, 1}
        or local_rank not in {0, 1}
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 2
    ):
        raise M03RV16PredictiveWorkflowError("V16 requires exactly two CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if (
        torch.cuda.get_device_name(device) != "NVIDIA H100 80GB HBM3"
        or not 79 * 1024**3 <= properties.total_memory <= 81 * 1024**3
        or (properties.major, properties.minor) != (9, 0)
    ):
        raise M03RV16PredictiveWorkflowError(
            "each V16 rank requires one NVIDIA H100 80GB HBM3"
        )
    owns = not dist.is_initialized()
    if owns:
        dist.init_process_group(backend="nccl", init_method="env://")
    if dist.get_world_size() != 2 or dist.get_rank() != rank:
        raise M03RV16PredictiveWorkflowError("V16 NCCL rank identity drifted")
    return rank, local_rank, world_size, device, owns


def _gather(value: Any, world_size: int) -> list[Any]:
    rows: list[Any] = [None] * world_size
    dist.all_gather_object(rows, value)
    return rows


def _broadcast(value: Any, rank: int) -> Any:
    rows = [value if rank == 0 else None]
    dist.broadcast_object_list(rows, src=0)
    return rows[0]


def resolve_m03r_v16_completion_index(value: int | None) -> int:
    result = value
    if result is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        if raw is None:
            raise M03RV16PredictiveWorkflowError(
                "V16 completion index is unavailable"
            )
        try:
            result = int(raw)
        except ValueError as exc:
            raise M03RV16PredictiveWorkflowError(
                "V16 completion index drifted"
            ) from exc
    if isinstance(result, bool) or result not in range(3):
        raise M03RV16PredictiveWorkflowError("V16 completion index drifted")
    return result


def _validate_runtime_package_members(
    package_plan_path: str | Path,
    package: M03RV16PackagePlan,
) -> tuple[Path, str]:
    package_root = Path(package_plan_path).resolve().parent.parent
    expected = {
        package_root / "source.tar": package.artifacts.source_archive_sha256,
        package_root / "source-manifest.json": package.artifacts.source_manifest_sha256,
        package_root / "cache/top2000-daily-bars.pt": (
            package.artifacts.cache_artifact_sha256
        ),
        package_root / "risk/risk-exposures.pt": package.artifacts.risk_artifact_sha256,
        package_root / "risk/risk-source-manifest.json": (
            package.artifacts.risk_source_manifest_file_sha256
        ),
        package_root / "risk/projector-manifest.json": (
            package.artifacts.projector_manifest_file_sha256
        ),
        package_root / "model/common-initial-parameter-state.pt": (
            package.artifacts.initial_parameter_state_file_sha256
        ),
        package_root / "structural/structural-slab.pt": (
            package.artifacts.structural_slab_file_sha256
        ),
    }
    for path, expected_sha in expected.items():
        try:
            status = path.lstat()
        except OSError as exc:
            raise M03RV16PredictiveWorkflowError(
                "V16 runtime package member is unavailable"
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(status.st_mode)
            or _file_sha256(path) != expected_sha
        ):
            raise M03RV16PredictiveWorkflowError(
                "V16 runtime package member drifted"
            )
    module_root = package_root / "source" / "src"
    if not Path(__file__).resolve().is_relative_to(module_root):
        raise M03RV16PredictiveWorkflowError(
            "V16 worker resolved outside immutable package source"
        )
    verified_source = verify_m03r_v16_source_tree(
        package_root / "source",
        package_root / "source-manifest.json",
        expected_source_manifest_file_sha256=(
            package.artifacts.source_manifest_sha256
        ),
        expected_runtime_worker_sha256=package.artifacts.worker_source_sha256,
    )
    if _file_sha256(Path(__file__)) != verified_source.runtime_worker_sha256:
        raise M03RV16PredictiveWorkflowError(
            "V16 executing worker bytes drifted from the package manifest"
        )
    return package_root, verified_source.source_tree_root_sha256


def _new_policy(setting_index: int, device: torch.device) -> Top2000M03RV16PredictivePolicy:
    policy = Top2000M03RV16PredictivePolicy(setting_index)
    return policy.to(device)


def _rank_update_row(result: Any) -> dict[str, Any]:
    return {
        "setting_index": result.step.setting_index,
        "fold_index": result.step.fold_index,
        "update_plan_sha256": result.update_plan.receipt_sha256,
        "batch_receipt_sha256": result.batch.receipt_sha256,
        "step_receipt_sha256": result.step.receipt_sha256,
        "source_array_sha256": result.batch.source_array_sha256,
        "selection_target_operator_root_sha256": (
            result.step.selection_target_operator_root_sha256
        ),
        "action_operator_root_sha256": result.step.action_operator_root_sha256,
        "completed_updates_after": result.step.completed_updates_after,
        "distributed_rank": result.step.distributed_rank,
        "local_origin_count": result.step.local_origin_count,
        "global_origin_count": result.step.global_origin_count,
        "encoder_version_root_before": result.step.encoder_version_root_before,
        "encoder_version_root_after": result.step.encoder_version_root_after,
        "selection_head_version_root_before": (
            result.step.selection_head_version_root_before
        ),
        "selection_head_version_root_after": (
            result.step.selection_head_version_root_after
        ),
        "total_loss": result.step.total_loss,
        "selection_robust_loss": result.step.selection_robust_loss,
        "encoder_gradient_norm_before_clip": (
            result.step.encoder_gradient_norm_before_clip
        ),
        "selection_head_gradient_norm_before_clip": (
            result.step.selection_head_gradient_norm_before_clip
        ),
        "encoder_gradient_clipped": result.step.encoder_gradient_clipped,
        "selection_head_gradient_clipped": (
            result.step.selection_head_gradient_clipped
        ),
        "learning_rate_multiplier": result.step.learning_rate_multiplier,
        "encoder_learning_rate": result.step.encoder_learning_rate,
        "selection_head_learning_rate": result.step.selection_head_learning_rate,
    }


def _validate_gathered_update(rows: list[Any], world_size: int) -> None:
    if (
        len(rows) != world_size
        or tuple(row["distributed_rank"] for row in rows) != tuple(range(world_size))
        or len({row["update_plan_sha256"] for row in rows}) != 1
        or len({row["source_array_sha256"] for row in rows}) != 1
        or len({row["completed_updates_after"] for row in rows}) != 1
        or len({row["selection_target_operator_root_sha256"] for row in rows}) != 1
        or len({row["action_operator_root_sha256"] for row in rows}) != 1
        or len({row["global_origin_count"] for row in rows}) != 1
        or sum(row["local_origin_count"] for row in rows)
        != rows[0]["global_origin_count"]
        or len({row["encoder_version_root_before"] for row in rows}) != 1
        or len({row["encoder_version_root_after"] for row in rows}) != 1
        or len({row["selection_head_version_root_before"] for row in rows}) != 1
        or len({row["selection_head_version_root_after"] for row in rows}) != 1
    ):
        raise M03RV16PredictiveWorkflowError("V16 rank update evidence diverged")


def _qualification_artifact(
    result: M03RV16FoldQualificationResult,
    selection_receipt_sha256: str,
) -> dict[str, Any]:
    result.validate()
    batch = result.score_authority.batch
    trace = result.trace
    return {
        "schema": M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "terminal_checkpoint_authority_sha256": (
            result.terminal_checkpoint_authority.receipt_sha256
        ),
        "qualified_score_authority_sha256": result.score_authority.receipt_sha256,
        "checkpoint_selection_receipt_sha256": selection_receipt_sha256,
        "trace_unsigned_payload": trace.unsigned_payload(),
        "trace_arrays": tuple(row.detach().cpu() for row in trace.arrays),
        "decision_origin_indices": batch.origin_indices.detach().cpu(),
        "executable_selection_mean": (
            batch.objective.executable_selection_mean.detach().cpu()
        ),
        "selection_target_economic": (
            batch.objective.selection_target_economic.detach().cpu()
        ),
        "selection_valid": batch.objective.selection_valid.detach().cpu(),
        "action_valid": batch.action_valid.detach().cpu(),
        "outer_2026_accessed": False,
        "economic_optimizer_updates": 0,
        "reinforcement_learning_updates": 0,
    }


def _load_package_surfaces(
    package_root: Path,
    package: M03RV16PackagePlan,
) -> tuple[Any, Any, Any, Any, Any]:
    cache = load_verified_top2000_hold30_development_cache(
        package_root / "cache/top2000-daily-bars.pt",
        expected_cache_sha256=package.artifacts.cache_artifact_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    risk_source, written = load_top2000_m03r_v9_risk_source(
        package_root / "risk/risk-source-manifest.json",
        expected_manifest_file_sha256=(
            package.artifacts.risk_source_manifest_file_sha256
        ),
    )
    projector, binding = load_m03r_v9_projector_manifest(
        package_root / "risk/projector-manifest.json",
        expected_file_sha256=package.artifacts.projector_manifest_file_sha256,
    )
    structural = load_m03r_v16_structural_slab(
        package_root / "structural/structural-slab.pt",
        expected_file_sha256=package.artifacts.structural_slab_file_sha256,
        expected_receipt_sha256=package.artifacts.structural_slab_receipt_sha256,
    )
    if (
        written.artifact_file_sha256 != package.artifacts.risk_artifact_sha256
        or projector.manifest_sha256 != package.artifacts.projector_manifest_sha256
        or binding.binding_sha256 != package.artifacts.projector_binding_sha256
        or structural.receipt_sha256
        != package.artifacts.structural_slab_receipt_sha256
    ):
        raise M03RV16PredictiveWorkflowError("V16 package surfaces drifted")
    return cache, risk_source, projector, binding, structural


def _capacity_probe_inputs(
    cache: Any,
    geometry: M03RV16FoldGeometry,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    episode_stop = geometry.qualification_target_stop_exclusive
    episode_start = episode_stop - M03R_V16_PREDICTIVE_SPEC.episode_state_rows
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=episode_start,
        state_stop_index_exclusive=episode_stop,
        max_state_rows=episode_stop - episode_start,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = move_and_bind_m03r_v16_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    fill = (
        geometry.qualification_origin_start_inclusive
        - episode_start
        + 1
    )
    return (
        sequence.benchmark_weights[fill],
        sequence.fill_membership[fill] & sequence.fill_availability[fill],
        sequence.risk_asset_caps[fill],
        sequence.risk_gross_max[fill],
    )


def _training_fold_inputs(
    training_root: Path,
    fold_index: int,
    fold_terminal_file_sha256: str,
) -> tuple[dict[str, Any], tuple[M03RV16InnerValidationReceipt, ...]]:
    fold = _read_immutable_json(
        training_root / "receipts" / f"fold-{fold_index:02d}-training-terminal.json",
        fold_terminal_file_sha256,
    )
    unsigned = {key: value for key, value in fold.items() if key != "receipt_sha256"}
    if (
        fold.get("schema") != M03R_V16_TRAINING_FOLD_TERMINAL_SCHEMA
        or fold.get("receipt_sha256") != _sha256(unsigned)
        or fold.get("fold_index") != fold_index
        or fold.get("qualification_tail_accessed") is not False
    ):
        raise M03RV16PredictiveWorkflowError("V16 training fold terminal drifted")
    fit_files = tuple(fold.get("epoch_fit_file_sha256", ()))
    fit_receipts = tuple(fold.get("epoch_fit_receipt_sha256", ()))
    if (
        len(fit_files) != M03R_V16_PREDICTIVE_SPEC.score_training_epochs
        or len(fit_receipts) != M03R_V16_PREDICTIVE_SPEC.score_training_epochs
    ):
        raise M03RV16PredictiveWorkflowError("V16 training epoch inventory drifted")
    validations: list[M03RV16InnerValidationReceipt] = []
    for epoch, expected_sha in enumerate(fit_files):
        payload = _read_immutable_json(
            training_root
            / "receipts"
            / f"fold-{fold_index:02d}-epoch-{epoch + 1:02d}-fit.json",
            str(expected_sha),
        )
        payload_unsigned = {
            key: value for key, value in payload.items() if key != "receipt_sha256"
        }
        if (
            payload.get("receipt_sha256") != _sha256(payload_unsigned)
            or payload.get("receipt_sha256") != fit_receipts[epoch]
            or payload.get("epoch_index") != epoch
            or payload.get("qualification_tail_accessed") is not False
        ):
            raise M03RV16PredictiveWorkflowError("V16 epoch fit evidence drifted")
        try:
            validation = M03RV16InnerValidationReceipt(
                **dict(payload["inner_validation"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise M03RV16PredictiveWorkflowError(
                "V16 training validation evidence is malformed"
            ) from exc
        validation.validate()
        validations.append(validation)
    return fold, tuple(validations)


def _run_m03r_v16_qualification_phase(
    *,
    package: M03RV16PackagePlan,
    authorization: Any,
    worker: Any,
    cache: Any,
    risk_source: Any,
    projector: Any,
    risk_binding: Any,
    structural: Any,
    geometries: tuple[M03RV16FoldGeometry, ...],
    qualification_activation: M03RV16QualificationActivation,
    training_root: Path,
    output: Path,
    startup_sha: str,
    rank: int,
    world_size: int,
    device: torch.device,
    rendered_manifest_sha256: str,
    pod_template_sha256: str,
) -> dict[str, Any] | None:
    """Open outer folds only under the adequate-training authority."""

    terminal_path = training_root / "training-terminal.json"
    expected_terminal_sha = qualification_activation.training_terminal_file_sha256[
        worker.setting_index
    ]
    training_terminal = _read_immutable_json(terminal_path, expected_terminal_sha)
    training_unsigned = {
        key: value for key, value in training_terminal.items() if key != "receipt_sha256"
    }
    fold_hashes = tuple(training_terminal.get("fold_terminal_file_sha256", ()))
    if (
        training_terminal.get("schema") != M03R_V16_TRAINING_TERMINAL_SCHEMA
        or training_terminal.get("receipt_sha256") != _sha256(training_unsigned)
        or training_terminal.get("package_plan_sha256") != package.package_plan_sha256
        or training_terminal.get("authorization_receipt_sha256")
        != authorization.receipt_sha256
        or training_terminal.get("worker_plan_sha256") != worker.receipt_sha256
        or training_terminal.get("setting_index") != worker.setting_index
        or training_terminal.get("qualification_tail_accessed") is not False
        or len(fold_hashes) != len(geometries)
    ):
        raise M03RV16PredictiveWorkflowError("V16 training terminal drifted")
    if (training_root / "fold-artifacts").exists():
        raise M03RV16PredictiveWorkflowError(
            "V16 training phase improperly contains outer artifacts"
        )

    fold_results: list[M03RV16FoldQualificationResult] = []
    fold_terminal_files: list[str] = []
    for geometry, fold_file_sha in zip(geometries, fold_hashes, strict=True):
        fold, validations = _training_fold_inputs(
            training_root, geometry.fold_index, str(fold_file_sha)
        )
        selection = select_m03r_v16_score_checkpoint(validations)
        if selection.receipt_sha256 != fold.get(
            "checkpoint_selection_receipt_sha256"
        ):
            raise M03RV16PredictiveWorkflowError(
                "V16 terminal checkpoint selection evidence drifted"
            )
        checkpoint_sha = str(fold["checkpoint_file_sha256"])
        checkpoint_source = str(fold["checkpoint_source_array_sha256"])
        loaded_policy = _new_policy(worker.setting_index, device)
        loaded = load_m03r_v16_epoch_checkpoint_for_evaluation(
            training_root
            / "checkpoints"
            / f"fold-{geometry.fold_index:02d}-epoch-{M03R_V16_PREDICTIVE_SPEC.score_training_epochs:02d}.pt",
            expected_file_sha256=checkpoint_sha,
            expected_setting_index=worker.setting_index,
            expected_fold_index=geometry.fold_index,
            expected_epoch_index=M03R_V16_PREDICTIVE_SPEC.score_training_epochs - 1,
            expected_completed_score_updates=geometry.maximum_optimizer_updates,
            expected_panel_schedule_sha256=package.schedule.receipt_sha256,
            expected_selection_target_operator_root_sha256=(
                structural.receipt.common_target_operator_root_sha256
            ),
            expected_action_operator_root_sha256=(
                structural.receipt.action_operator_root_sha256
            ),
            expected_source_array_sha256=checkpoint_source,
            expected_asset_axis_sha256=cache.action_hash,
            policy=loaded_policy,
        )
        terminal_authority = issue_m03r_v16_terminal_checkpoint_authority(
            loaded,
            selection,
            package.schedule,
            geometry,
            structural_slab_receipt_sha256=structural.receipt_sha256,
        )
        risk_state = build_m03r_v16_qualification_risk_state(
            cache,
            geometry,
            risk_source,
            risk_binding,
            projector,
            device=device,
        )
        result = run_m03r_v16_fold_qualification(
            cache,
            geometry,
            risk_source,
            risk_state,
            structural,
            loaded_policy,
            terminal_authority,
            device=device,
        )
        if len(set(_gather(result.trace.trace_sha256, world_size))) != 1:
            raise M03RV16PredictiveWorkflowError(
                "V16 qualification trace diverged between ranks"
            )
        artifact_sha: str | None = None
        if rank == 0:
            artifact_sha = _write_immutable_torch(
                output
                / "fold-artifacts"
                / f"fold-{geometry.fold_index:02d}-qualification.pt",
                _qualification_artifact(result, selection.receipt_sha256),
            )
        artifact_sha = _broadcast(artifact_sha, rank)
        if not isinstance(artifact_sha, str):
            raise M03RV16PredictiveWorkflowError(
                "V16 qualification artifact publication failed"
            )
        if rank == 0:
            fold_results.append(result)
            unsigned_fold = {
                "schema": M03R_V16_FOLD_TERMINAL_SCHEMA,
                "package_plan_sha256": package.package_plan_sha256,
                "authorization_receipt_sha256": authorization.receipt_sha256,
                "qualification_activation_receipt_sha256": (
                    qualification_activation.receipt_sha256
                ),
                "training_terminal_file_sha256": expected_terminal_sha,
                "worker_plan_sha256": worker.receipt_sha256,
                "setting_index": worker.setting_index,
                "setting_id": worker.setting_id,
                "fold_index": geometry.fold_index,
                "panel_schedule_sha256": package.schedule.receipt_sha256,
                "terminal_checkpoint_authority_sha256": (
                    terminal_authority.receipt_sha256
                ),
                "qualified_score_authority_sha256": result.score_authority.receipt_sha256,
                "qualification_artifact_file_sha256": artifact_sha,
                "qualification_trace_sha256": result.trace.trace_sha256,
                "qualification_after_strict_terminal_reload": True,
                "economic_optimizer_updates": 0,
                "reinforcement_learning_updates": 0,
                "outer_2026_accessed": False,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
            fold_terminal_files.append(
                _write_immutable_json(
                    output
                    / "receipts"
                    / f"fold-{geometry.fold_index:02d}-terminal.json",
                    {**unsigned_fold, "receipt_sha256": _sha256(unsigned_fold)},
                )
            )
        dist.barrier()
        del loaded_policy, risk_state
        torch.cuda.empty_cache()

    terminal: dict[str, Any] | None = None
    if rank == 0:
        results = tuple(fold_results)
        bootstrap = build_m03r_v16_bootstrap_plan(
            tuple(row.trace.decision_origin_indices for row in results),
            tuple(row.trace.execution_origin_indices for row in results),
        )
        qualification = qualify_m03r_v16_predictive_candidate(results, bootstrap)
        unsigned = {
            "schema": M03R_V16_WORKER_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "qualification_activation_receipt_sha256": (
                qualification_activation.receipt_sha256
            ),
            "training_terminal_file_sha256": expected_terminal_sha,
            "worker_plan_sha256": worker.receipt_sha256,
            "startup_file_sha256": startup_sha,
            "setting_index": worker.setting_index,
            "setting_id": worker.setting_id,
            "fold_terminal_file_sha256": tuple(fold_terminal_files),
            "bootstrap_plan": asdict(bootstrap),
            "bootstrap_plan_sha256": bootstrap.receipt_sha256,
            "predictive_qualification": asdict(qualification),
            "predictive_qualification_sha256": qualification.receipt_sha256,
            "raw_predictive_gates_passed": qualification.primary_hypothesis_passed,
            "three_seed_confirmation_may_be_minted": False,
            "rendered_manifest_sha256": rendered_manifest_sha256,
            "pod_template_sha256": pod_template_sha256,
            "economic_generation_may_be_minted": False,
            "reinforcement_learning_authorized": False,
            "outer_2026_accessed": False,
            "world_size": 2,
            "gpus_per_worker": 2,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        terminal = {**unsigned, "receipt_sha256": _sha256(unsigned)}
        _write_immutable_json(output / "predictive-terminal.json", terminal)
    dist.barrier()
    return terminal


def run_m03r_v16_predictive_worker(
    package_plan_path: str | Path,
    authorization_path: str | Path,
    *,
    expected_package_plan_file_sha256: str,
    expected_authorization_file_sha256: str,
    completion_index: int | None = None,
    capacity_only: bool = False,
    capacity_output_root: str | Path | None = None,
    training_activation_path: str | Path | None = None,
    expected_training_activation_file_sha256: str | None = None,
    qualification_activation_path: str | Path | None = None,
    expected_qualification_activation_file_sha256: str | None = None,
    qualification_only: bool = False,
    training_root: str | Path | None = None,
    rendered_manifest_sha256: str | None = None,
    pod_template_sha256: str | None = None,
) -> dict[str, Any] | None:
    if capacity_only and capacity_output_root is None:
        raise M03RV16PredictiveWorkflowError(
            "V16 capacity evidence requires a disjoint output root"
        )
    if capacity_only and qualification_only:
        raise M03RV16PredictiveWorkflowError("V16 worker phase is ambiguous")
    package = load_m03r_v16_package_plan(
        package_plan_path,
        expected_file_sha256=expected_package_plan_file_sha256,
    )
    authorization = load_m03r_v16_execution_authorization(
        authorization_path,
        expected_file_sha256=expected_authorization_file_sha256,
        package=package,
    )
    package_root, source_tree_root_sha256 = _validate_runtime_package_members(
        package_plan_path, package
    )
    if authorization.package_plan_file_sha256 != expected_package_plan_file_sha256:
        raise M03RV16PredictiveWorkflowError(
            "V16 authorization and package plan disagree"
        )
    training_activation: M03RV16TrainingActivation | None = None
    qualification_activation: M03RV16QualificationActivation | None = None
    if not capacity_only and not qualification_only:
        if training_activation_path is None or expected_training_activation_file_sha256 is None:
            raise M03RV16PredictiveWorkflowError(
                "V16 training requires an immutable activation authority"
            )
        training_activation = load_m03r_v16_training_activation(
            training_activation_path,
            expected_file_sha256=expected_training_activation_file_sha256,
            package=package,
            authorization=authorization,
        )
    elif qualification_only:
        if (
            qualification_activation_path is None
            or expected_qualification_activation_file_sha256 is None
            or training_root is None
        ):
            raise M03RV16PredictiveWorkflowError(
                "V16 qualification requires activation and frozen training evidence"
            )
        qualification_activation = load_m03r_v16_qualification_activation(
            qualification_activation_path,
            expected_file_sha256=expected_qualification_activation_file_sha256,
            package=package,
            authorization=authorization,
        )
    active_source_root = (
        None
        if capacity_only
        else (
            qualification_activation.source_tree_root_sha256
            if qualification_activation is not None
            else training_activation.source_tree_root_sha256
            if training_activation is not None
            else None
        )
    )
    if active_source_root is not None and active_source_root != source_tree_root_sha256:
        raise M03RV16PredictiveWorkflowError(
            "V16 phase activation source tree drifted"
        )
    if not capacity_only and (
        rendered_manifest_sha256 is not None or pod_template_sha256 is not None
    ):
        if rendered_manifest_sha256 is None or pod_template_sha256 is None:
            raise M03RV16PredictiveWorkflowError(
                "V16 rendered manifest identities must be supplied together"
            )
        for name, value in (
            ("rendered_manifest_sha256", rendered_manifest_sha256),
            ("pod_template_sha256", pod_template_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise M03RV16PredictiveWorkflowError(f"{name} drifted")
    index = resolve_m03r_v16_completion_index(completion_index)
    worker = package.panel.workers[index]
    if qualification_only and training_root is not None:
        training_root = (
            Path(training_root)
            / f"completion-{index:02d}-setting-{index:02d}"
        )
    rank, local_rank, world_size, device, owns = _distributed_context()
    output = (
        Path(capacity_output_root)
        if capacity_only and capacity_output_root is not None
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
            },
            world_size,
        )
        startup_sha: str | None = None
        if rank == 0:
            startup_sha = _write_immutable_json(
                output / "two-h100-startup.json",
                {
                    "schema": M03R_V16_STARTUP_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "source_tree_root_sha256": source_tree_root_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "mode": (
                        "capacity"
                        if capacity_only
                        else "qualification"
                        if qualification_only
                        else "training"
                    ),
                    "training_activation_receipt_sha256": (
                        None
                        if training_activation is None
                        else training_activation.receipt_sha256
                    ),
                    "qualification_activation_receipt_sha256": (
                        None
                        if qualification_activation is None
                        else qualification_activation.receipt_sha256
                    ),
                    "rendered_manifest_sha256": rendered_manifest_sha256,
                    "pod_template_sha256": pod_template_sha256,
                    "rank_runtime": runtime_rows,
                    "exact_h100_80gb_per_rank": True,
                    "nccl_process_group_initialized": True,
                    "economic_optimizer_updates": 0,
                    "reinforcement_learning_updates": 0,
                    "outer_2026_accessed": False,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                },
            )
        startup_sha = _broadcast(startup_sha, rank)
        if not isinstance(startup_sha, str):
            raise M03RV16PredictiveWorkflowError("V16 startup receipt is absent")
        cache, risk_source, projector, risk_binding, structural = (
            _load_package_surfaces(package_root, package)
        )
        geometries = render_m03r_v16_fold_geometries(1001)
        if capacity_only:
            _seed_everything(worker.seed)
            policy = _new_policy(worker.setting_index, device)
            load_m03r_v16_initial_parameter_state(
                worker.initial_parameter_state_path,
                policy,
                expected_file_sha256=worker.initial_parameter_state_file_sha256,
                expected_state_sha256=worker.initial_parameter_state_sha256,
                expected_architecture_sha256=(
                    worker.initial_parameter_architecture_sha256
                ),
            )
            optimizer, partition = build_m03r_v16_optimizer(policy)
            geometry = geometries[0]
            risk_state = build_m03r_v16_qualification_risk_state(
                cache,
                geometry,
                risk_source,
                risk_binding,
                projector,
                device=device,
            )
            benchmark, trade_mask, caps, gross = _capacity_probe_inputs(
                cache, geometry, device
            )
            rank_evidence = run_m03r_v16_disposable_capacity_rank(
                cache,
                package.schedule,
                geometry,
                risk_source,
                structural,
                policy,
                optimizer,
                partition,
                distributed_rank=rank,
                device=device,
                qualification_risk_state=risk_state,
                qualification_benchmark_weights=benchmark,
                qualification_trade_mask=trade_mask,
                qualification_risk_asset_caps=caps,
                qualification_risk_gross_max=gross,
            )
            gathered = _gather(rank_evidence, world_size)
            capacity_terminal_payload: dict[str, Any] | None = None
            if rank == 0:
                capacity = build_m03r_v16_capacity_terminal(tuple(gathered))
                capacity_terminal_payload = {
                    "schema": capacity.schema,
                    "package_plan_sha256": package.package_plan_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "startup_file_sha256": startup_sha,
                    "source_tree_root_sha256": source_tree_root_sha256,
                    "capacity": asdict(capacity),
                    "capacity_receipt_sha256": capacity.receipt_sha256,
                    "scientific_training_performed": False,
                    "disposable_optimizer_update_executed": True,
                    "disposable_train_validate_train_executed": True,
                    "scientific_checkpoint_published": False,
                    "economic_optimizer_updates": 0,
                    "reinforcement_learning_updates": 0,
                    "outer_2026_accessed": False,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                }
                _write_immutable_json(
                    output / "two-h100-capacity-terminal.json",
                    capacity_terminal_payload,
                )
            dist.barrier()
            return capacity_terminal_payload

        if qualification_only:
            if qualification_activation is None or training_root is None:
                raise M03RV16PredictiveWorkflowError(
                    "V16 qualification authority is absent"
                )
            return _run_m03r_v16_qualification_phase(
                package=package,
                authorization=authorization,
                worker=worker,
                cache=cache,
                risk_source=risk_source,
                projector=projector,
                risk_binding=risk_binding,
                structural=structural,
                geometries=geometries,
                qualification_activation=qualification_activation,
                training_root=Path(training_root),
                output=output,
                startup_sha=startup_sha,
                rank=rank,
                world_size=world_size,
                device=device,
                rendered_manifest_sha256=str(rendered_manifest_sha256),
                pod_template_sha256=str(pod_template_sha256),
            )

        fold_terminal_files: list[str] = []
        fold_training_adequacy_files: list[str] = []
        fold_training_adequacy_receipts: list[str] = []
        fold_training_adequacy_status: list[str] = []
        for geometry in geometries:
            _seed_everything(worker.seed)
            policy = _new_policy(worker.setting_index, device)
            load_m03r_v16_initial_parameter_state(
                worker.initial_parameter_state_path,
                policy,
                expected_file_sha256=worker.initial_parameter_state_file_sha256,
                expected_state_sha256=worker.initial_parameter_state_sha256,
                expected_architecture_sha256=(
                    worker.initial_parameter_architecture_sha256
                ),
            )
            if model_state_sha256(policy) != worker.initial_parameter_state_sha256:
                raise M03RV16PredictiveWorkflowError(
                    "V16 common initial state drifted"
                )
            optimizer, partition = build_m03r_v16_optimizer(policy)
            validation_receipts = []
            update_rows: list[list[Any]] = []
            epoch_checkpoint_paths: list[Path] = []
            epoch_checkpoint_hashes: list[str] = []
            epoch_fit_payloads: list[dict[str, Any]] = []
            epoch_fit_file_hashes: list[str] = []
            source_rows: list[str] = []
            for completed in range(geometry.maximum_optimizer_updates):
                update = run_m03r_v16_pretraining_fold_update(
                    cache,
                    package.schedule,
                    geometry,
                    risk_source,
                    structural,
                    policy,
                    optimizer,
                    partition,
                    completed_updates=completed,
                    distributed_rank=rank,
                    distributed_world_size=world_size,
                    device=device,
                )
                gathered = _gather(_rank_update_row(update), world_size)
                _validate_gathered_update(gathered, world_size)
                update_rows.append(gathered)
                source_rows.append(gathered[0]["source_array_sha256"])
                if (completed + 1) % geometry.training_block_count != 0:
                    continue
                epoch = completed // geometry.training_block_count
                source_root = _sha256(tuple(source_rows))
                checkpoint_path = (
                    output
                    / "checkpoints"
                    / f"fold-{geometry.fold_index:02d}-epoch-{epoch + 1:02d}.pt"
                )
                checkpoint_sha: str | None = None
                if rank == 0:
                    checkpoint_sha = write_immutable_m03r_v16_epoch_checkpoint(
                        checkpoint_path,
                        policy,
                        fold_index=geometry.fold_index,
                        epoch_index=epoch,
                        completed_score_updates=completed + 1,
                        panel_schedule_sha256=package.schedule.receipt_sha256,
                        selection_target_operator_root_sha256=(
                            structural.receipt.common_target_operator_root_sha256
                        ),
                        action_operator_root_sha256=(
                            structural.receipt.action_operator_root_sha256
                        ),
                        source_array_sha256=source_root,
                        asset_axis_sha256=cache.action_hash,
                    )
                checkpoint_sha = _broadcast(checkpoint_sha, rank)
                if not isinstance(checkpoint_sha, str):
                    raise M03RV16PredictiveWorkflowError(
                        "V16 epoch checkpoint publication failed"
                    )
                dist.barrier()
                validation_batch = build_m03r_v16_inner_validation_batch(
                    cache,
                    geometry,
                    risk_source,
                    structural,
                    policy,
                    device=device,
                )
                validation = evaluate_m03r_v16_inner_validation_batch(
                    validation_batch,
                    geometry,
                    epoch_index=epoch,
                    completed_score_updates=completed + 1,
                    model_state_sha256=model_state_sha256(policy),
                    epoch_checkpoint_file_sha256=checkpoint_sha,
                )
                validation_rows = _gather(asdict(validation), world_size)
                if len({_sha256(row) for row in validation_rows}) != 1:
                    raise M03RV16PredictiveWorkflowError(
                        "V16 inner validation diverged between ranks"
                    )
                validation_receipts.append(validation)
                epoch_checkpoint_paths.append(checkpoint_path)
                epoch_checkpoint_hashes.append(checkpoint_sha)
                if rank == 0:
                    epoch_fit = build_m03r_v16_epoch_fit_payload(
                        validation,
                        tuple(update_rows[-geometry.training_block_count :]),
                        package_plan_sha256=package.package_plan_sha256,
                        worker_plan_sha256=worker.receipt_sha256,
                    )
                    epoch_fit_payloads.append(epoch_fit)
                    epoch_fit_file_hashes.append(
                        _write_immutable_json(
                            output
                            / "receipts"
                            / (
                                f"fold-{geometry.fold_index:02d}-"
                                f"epoch-{epoch + 1:02d}-fit.json"
                            ),
                            epoch_fit,
                        )
                    )
            if len(validation_receipts) != M03R_V16_PREDICTIVE_SPEC.score_training_epochs:
                raise M03RV16PredictiveWorkflowError(
                    "V16 fixed epoch coverage drifted"
                )
            selection = select_m03r_v16_score_checkpoint(
                tuple(validation_receipts)
            )
            if (
                selection.selected_checkpoint_file_sha256
                != epoch_checkpoint_hashes[-1]
                or selection.selected_model_state_sha256
                != model_state_sha256(policy)
            ):
                raise M03RV16PredictiveWorkflowError(
                    "V16 terminal checkpoint selection drifted"
                )
            training_adequacy = None
            training_adequacy_file_sha: str | None = None
            if rank == 0:
                training_adequacy = classify_m03r_v16_training_adequacy(
                    tuple(validation_receipts), tuple(epoch_fit_payloads)
                )
                training_adequacy_payload = {
                    **asdict(training_adequacy),
                    "receipt_sha256": training_adequacy.receipt_sha256,
                    "epoch_fit_file_sha256": tuple(epoch_fit_file_hashes),
                }
                training_adequacy_file_sha = _write_immutable_json(
                    output
                    / "receipts"
                    / f"fold-{geometry.fold_index:02d}-training-adequacy.json",
                    training_adequacy_payload,
                )
            final_optimizer_hashes = _gather(
                optimizer_state_sha256(optimizer), world_size
            )
            if len(set(final_optimizer_hashes)) != 1:
                raise M03RV16PredictiveWorkflowError(
                    "V16 terminal optimizer states diverged"
                )
            final_source_root = _sha256(tuple(source_rows))
            del optimizer, partition, policy
            torch.cuda.empty_cache()
            if rank == 0:
                if (
                    training_adequacy is None
                    or training_adequacy_file_sha is None
                ):
                    raise M03RV16PredictiveWorkflowError(
                        "V16 training adequacy was not published"
                    )
                unsigned_fold = {
                    "schema": M03R_V16_TRAINING_FOLD_TERMINAL_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "training_activation_receipt_sha256": (
                        training_activation.receipt_sha256
                        if training_activation is not None
                        else None
                    ),
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "fold_index": geometry.fold_index,
                    "completed_updates": geometry.maximum_optimizer_updates,
                    "training_epoch_count": (
                        M03R_V16_PREDICTIVE_SPEC.score_training_epochs
                    ),
                    "optimizer_state_sha256": final_optimizer_hashes[0],
                    "checkpoint_selection_receipt_sha256": (
                        selection.receipt_sha256
                    ),
                    "inner_validation_receipt_sha256": tuple(
                        row.receipt_sha256 for row in validation_receipts
                    ),
                    "epoch_fit_file_sha256": tuple(epoch_fit_file_hashes),
                    "epoch_fit_receipt_sha256": (
                        training_adequacy.epoch_fit_receipt_sha256
                    ),
                    "training_adequacy_status": training_adequacy.status,
                    "training_adequacy_receipt_sha256": (
                        training_adequacy.receipt_sha256
                    ),
                    "training_adequacy_file_sha256": (
                        training_adequacy_file_sha
                    ),
                    "panel_schedule_sha256": package.schedule.receipt_sha256,
                    "structural_slab_receipt_sha256": (
                        structural.receipt.receipt_sha256
                    ),
                    "checkpoint_file_sha256": epoch_checkpoint_hashes[-1],
                    "checkpoint_source_array_sha256": final_source_root,
                    "qualification_tail_accessed": False,
                    "economic_optimizer_updates": 0,
                    "reinforcement_learning_updates": 0,
                    "outer_2026_accessed": False,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                }
                fold_terminal = {
                    **unsigned_fold,
                    "receipt_sha256": _sha256(unsigned_fold),
                }
                fold_terminal_files.append(
                    _write_immutable_json(
                        output
                        / "receipts"
                        / f"fold-{geometry.fold_index:02d}-training-terminal.json",
                        fold_terminal,
                    )
                )
                fold_training_adequacy_files.append(training_adequacy_file_sha)
                fold_training_adequacy_receipts.append(
                    training_adequacy.receipt_sha256
                )
                fold_training_adequacy_status.append(training_adequacy.status)
            dist.barrier()

        terminal: dict[str, Any] | None = None
        if rank == 0:
            unsigned_terminal = {
                "schema": M03R_V16_TRAINING_TERMINAL_SCHEMA,
                "package_plan_sha256": package.package_plan_sha256,
                "package_plan_file_sha256": expected_package_plan_file_sha256,
                "authorization_receipt_sha256": authorization.receipt_sha256,
                "training_activation_receipt_sha256": (
                    training_activation.receipt_sha256
                    if training_activation is not None
                    else None
                ),
                "worker_plan_sha256": worker.receipt_sha256,
                "startup_file_sha256": startup_sha,
                "source_tree_root_sha256": source_tree_root_sha256,
                "setting_index": worker.setting_index,
                "setting_id": worker.setting_id,
                "fold_terminal_file_sha256": tuple(fold_terminal_files),
                "fold_training_adequacy_file_sha256": tuple(
                    fold_training_adequacy_files
                ),
                "fold_training_adequacy_receipt_sha256": tuple(
                    fold_training_adequacy_receipts
                ),
                "fold_training_adequacy_status": tuple(
                    fold_training_adequacy_status
                ),
                "rendered_manifest_sha256": rendered_manifest_sha256,
                "pod_template_sha256": pod_template_sha256,
                "qualification_tail_accessed": False,
                "outer_qualification_authorized": False,
                "three_seed_confirmation_may_be_minted": False,
                "economic_generation_may_be_minted": False,
                "reinforcement_learning_authorized": False,
                "outer_2026_accessed": False,
                "world_size": 2,
                "gpus_per_worker": 2,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
            terminal = {
                **unsigned_terminal,
                "receipt_sha256": _sha256(unsigned_terminal),
            }
            _write_immutable_json(output / "training-terminal.json", terminal)
        dist.barrier()
        return terminal
    except BaseException as exc:
        if rank == 0 and output.is_dir() and not (output / "worker-error.json").exists():
            _write_immutable_json(
                output / "worker-error.json",
                {
                    "schema": M03R_V16_WORKER_ERROR_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "authorization_receipt_sha256": authorization.receipt_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "mode": (
                        "capacity"
                        if capacity_only
                        else "qualification"
                        if qualification_only
                        else "training"
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "checkpoint_published_after_failure": False,
                    "economic_optimizer_updates": 0,
                    "reinforcement_learning_updates": 0,
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
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--capacity-output-root")
    parser.add_argument("--training-activation")
    parser.add_argument("--training-activation-file-sha256")
    parser.add_argument("--qualification-activation")
    parser.add_argument("--qualification-activation-file-sha256")
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--training-root")
    parser.add_argument("--rendered-manifest-sha256")
    parser.add_argument("--pod-template-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_m03r_v16_predictive_worker(
        args.package_plan,
        args.execution_authorization,
        expected_package_plan_file_sha256=args.package_plan_file_sha256,
        expected_authorization_file_sha256=(
            args.execution_authorization_file_sha256
        ),
        completion_index=args.completion_index,
        capacity_only=args.capacity_only,
        capacity_output_root=args.capacity_output_root,
        training_activation_path=args.training_activation,
        expected_training_activation_file_sha256=(
            args.training_activation_file_sha256
        ),
        qualification_activation_path=args.qualification_activation,
        expected_qualification_activation_file_sha256=(
            args.qualification_activation_file_sha256
        ),
        qualification_only=args.qualification_only,
        training_root=args.training_root,
        rendered_manifest_sha256=args.rendered_manifest_sha256,
        pod_template_sha256=args.pod_template_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V16_FOLD_TERMINAL_SCHEMA",
    "M03R_V16_TRAINING_FOLD_TERMINAL_SCHEMA",
    "M03R_V16_TRAINING_TERMINAL_SCHEMA",
    "M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA",
    "M03R_V16_STARTUP_SCHEMA",
    "M03R_V16_WORKER_ERROR_SCHEMA",
    "M03R_V16_WORKER_TERMINAL_SCHEMA",
    "M03RV16PredictiveWorkflowError",
    "main",
    "resolve_m03r_v16_completion_index",
    "run_m03r_v16_predictive_worker",
]
