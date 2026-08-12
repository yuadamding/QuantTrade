"""Exact two-H100 worker for the three-setting M03R-v9 predictive panel.

This entrypoint performs no economic optimization.  It trains each fold for
exactly 64 predictive updates, publishes horizon-specific update-64
checkpoints before opening the untouched qualification tail, evaluates both
21- and 30-session candidates, and stops successfully even when neither
candidate passes the frozen gate.
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
    M03R_V9_PREDICTIVE_SPEC,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v9_checkpoint import (
    write_immutable_m03r_v9_alpha_checkpoint,
)
from rl_quant.training.top2000_m03r_v9_fold import (
    M03RV9QualificationFoldResult,
    build_m03r_v9_qualification_risk_state,
    evaluate_m03r_v9_qualification_fold,
    run_m03r_v9_pretraining_fold_update,
)
from rl_quant.training.top2000_m03r_v9_package import (
    M03RV9PackagePlan,
    load_m03r_v9_package_plan,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_predictive_worker import (
    M03RV9PredictiveWorkerPlan,
    resolve_m03r_v9_predictive_setting_index,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    build_m03r_v9_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    M03RV9AlphaStepReceipt,
    model_state_sha256,
    optimizer_state_sha256,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    load_m03r_v9_projector_manifest,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    load_top2000_m03r_v9_risk_source,
)
from rl_quant.training.top2000_m03r_v9_selection import (
    M03RV9PredictiveQualification,
    qualify_m03r_v9_predictive_candidate,
    select_m03r_v9_horizon,
)

M03R_V9_STARTUP_SCHEMA = "rl-quant.top2000-dev.m03r-v9-two-h100-startup-v1"
M03R_V9_CAPACITY_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-two-h100-capacity-terminal-v1"
)
M03R_V9_FOLD_RESULT_SCHEMA = "rl-quant.top2000-dev.m03r-v9-fold-result-v1"
M03R_V9_TERMINAL_SCHEMA = "rl-quant.top2000-dev.m03r-v9-worker-terminal-v1"


class M03RV9PredictiveWorkflowError(RuntimeError):
    """The package, rank runtime, or immutable worker output drifted."""


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(encoded)


def _write_immutable_torch(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise M03RV9PredictiveWorkflowError("torch artifact target already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise M03RV9PredictiveWorkflowError(
            "worker requires exact torchrun rank environment"
        ) from exc
    if (
        world_size != 2
        or rank not in range(2)
        or local_rank not in range(2)
        or restart_count != 0
    ):
        raise M03RV9PredictiveWorkflowError(
            "v9 requires one fresh, non-resumed two-rank attempt"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise M03RV9PredictiveWorkflowError(
            "v9 predictive training requires exactly two visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if (
        torch.cuda.get_device_name(device) != "NVIDIA H100 80GB HBM3"
        or not 79 * 1024**3 <= properties.total_memory <= 81 * 1024**3
        or (properties.major, properties.minor) != (9, 0)
    ):
        raise M03RV9PredictiveWorkflowError(
            "each rank requires one NVIDIA H100 80GB HBM3"
        )
    owns = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        owns = True
    if dist.get_world_size() != 2 or dist.get_rank() != rank:
        raise M03RV9PredictiveWorkflowError("NCCL process-group identity drifted")
    return rank, local_rank, world_size, device, owns


def _gather(value: Any, world_size: int) -> list[Any]:
    rows: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(rows, value)
    return rows


def _broadcast(value: Any, rank: int) -> Any:
    rows = [value if rank == 0 else None]
    dist.broadcast_object_list(rows, src=0)
    return rows[0]


def _resolve_worker(
    package: M03RV9PackagePlan,
    completion_index: int,
) -> M03RV9PredictiveWorkerPlan:
    index = resolve_m03r_v9_predictive_setting_index(completion_index)
    worker = package.panel.workers[index]
    worker.validate()
    if worker.setting_index != index:
        raise M03RV9PredictiveWorkflowError("completion-to-setting mapping drifted")
    return worker


def resolve_m03r_v9_completion_index(value: int | None) -> int:
    if value is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        try:
            value = None if raw is None else int(raw)
        except ValueError as exc:
            raise M03RV9PredictiveWorkflowError(
                "JOB_COMPLETION_INDEX must be an integer"
            ) from exc
    if value is None:
        raise M03RV9PredictiveWorkflowError("completion index is required")
    return resolve_m03r_v9_predictive_setting_index(value)


def _clone_evaluation_policy(
    policy: Top2000M03RV9PredictivePolicy,
    worker: M03RV9PredictiveWorkerPlan,
    horizon_index: int,
    device: torch.device,
) -> Top2000M03RV9PredictivePolicy:
    clone = Top2000M03RV9PredictivePolicy(
        worker.setting_index,
        worker.horizon_bindings[horizon_index],
        token_dim=worker.token_dim,
        raw_stock_chunk=worker.raw_stock_chunk,
        activation_checkpointing=worker.activation_checkpointing,
    ).to(device)
    clone.load_state_dict(policy.state_dict(), strict=True)
    if model_state_sha256(clone) != model_state_sha256(policy):
        raise M03RV9PredictiveWorkflowError("horizon clone changed model state")
    return clone


def _fold_artifact_payload(
    result: M03RV9QualificationFoldResult,
) -> dict[str, Any]:
    return {
        "schema": M03R_V9_FOLD_RESULT_SCHEMA,
        "alpha_evidence": asdict(result.alpha_evidence),
        "sleeve_evidence": asdict(result.sleeve_evidence),
        "sleeve_trace": asdict(result.sleeve_trace),
        "geometry_sha256": result.geometry_sha256,
        "model_state_source_receipt_sha256": (result.model_state_source_receipt_sha256),
    }


def run_m03r_v9_predictive_worker(
    package_plan_path: str | Path,
    *,
    expected_package_plan_sha256: str,
    completion_index: int | None = None,
    startup_only: bool = False,
) -> dict[str, Any] | None:
    """Run one exact setting; a failed scientific gate is a valid terminal."""

    package = load_m03r_v9_package_plan(
        package_plan_path,
        expected_package_plan_sha256=expected_package_plan_sha256,
    )
    index = resolve_m03r_v9_completion_index(completion_index)
    worker = _resolve_worker(package, index)
    rank, local_rank, world_size, device, owns = _distributed_context()
    output = Path(worker.output_root)
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
        startup_file_sha: str | None = None
        if rank == 0:
            startup_file_sha = _write_immutable_json(
                output / "two-h100-startup.json",
                {
                    "schema": M03R_V9_STARTUP_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "mode": "two-h100-capacity" if startup_only else "predictive",
                    "rank_runtime": runtime_rows,
                    "exact_h100_80gb_per_rank": True,
                    "nccl_process_group_initialized": True,
                    "restart_count": 0,
                    "research_only": True,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                },
            )
        startup_file_sha = _broadcast(startup_file_sha, rank)
        if not isinstance(startup_file_sha, str):
            raise M03RV9PredictiveWorkflowError("startup receipt was not published")

        if startup_only:
            capacity_terminal: dict[str, Any] | None = None
            if rank == 0:
                unsigned_capacity = {
                    "schema": M03R_V9_CAPACITY_TERMINAL_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "startup_file_sha256": startup_file_sha,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "world_size": 2,
                    "gpus_per_worker": 2,
                    "exact_h100_80gb_per_rank": True,
                    "nccl_process_group_initialized": True,
                    "training_performed": False,
                    "economic_optimizer_updates": 0,
                    "h100_capacity_evidence": True,
                    "research_only": True,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                }
                capacity_terminal = {
                    **unsigned_capacity,
                    "receipt_sha256": _sha256_bytes(
                        _canonical_bytes(unsigned_capacity)
                    ),
                }
                _write_immutable_json(
                    output / "two-h100-capacity-terminal.json",
                    capacity_terminal,
                )
            dist.barrier()
            return capacity_terminal

        cache = load_verified_top2000_hold30_development_cache(
            worker.cache_path,
            expected_cache_sha256=worker.cache_sha256,
            acknowledgement=DEVELOPMENT_ACK,
        )
        if cache.daily_ohlcv.shape[0] != TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS:
            raise M03RV9PredictiveWorkflowError("cache state geometry drifted")
        risk_source, written_risk = load_top2000_m03r_v9_risk_source(
            Path(worker.risk_source_manifest_path),
            expected_manifest_file_sha256=(worker.risk_source_manifest_file_sha256),
        )
        projector, risk_binding = load_m03r_v9_projector_manifest(
            Path(worker.projector_manifest_path),
            expected_file_sha256=worker.projector_manifest_file_sha256,
        )
        if (
            written_risk.manifest_file_sha256 != worker.risk_source_manifest_file_sha256
            or projector.manifest_sha256 != worker.projector_manifest_sha256
            or risk_binding.binding_sha256 != worker.projector_binding_sha256
        ):
            raise M03RV9PredictiveWorkflowError("risk package identity drifted")

        folds = render_top2000_m03r_v7_development_folds(
            TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        )
        results: dict[int, list[M03RV9QualificationFoldResult]] = {21: [], 30: []}
        fold_receipt_file_sha: list[str] = []
        for fold in folds:
            _seed_everything(worker.seed)
            training_policy = Top2000M03RV9PredictivePolicy(
                worker.setting_index,
                worker.horizon_bindings[1],
                token_dim=worker.token_dim,
                raw_stock_chunk=worker.raw_stock_chunk,
                activation_checkpointing=worker.activation_checkpointing,
            ).to(device)
            optimizer, partition = build_m03r_v9_alpha_pretraining_optimizer(
                training_policy
            )
            step_receipts: list[M03RV9AlphaStepReceipt] = []
            for completed in range(M03R_V9_PREDICTIVE_SPEC.maximum_optimizer_updates):
                step_receipts.append(
                    run_m03r_v9_pretraining_fold_update(
                        cache,
                        worker,
                        fold,
                        risk_source,
                        training_policy,
                        optimizer,
                        partition,
                        completed_updates=completed,
                        distributed_rank=rank,
                        distributed_world_size=world_size,
                        device=device,
                    )
                )
            last_step = step_receipts[-1]
            model_hashes = _gather(model_state_sha256(training_policy), world_size)
            optimizer_hashes = _gather(optimizer_state_sha256(optimizer), world_size)
            step_hashes = _gather(last_step.receipt_sha256, world_size)
            if (
                len(set(model_hashes)) != 1
                or len(set(optimizer_hashes)) != 1
                or len(set(step_hashes)) != 1
            ):
                raise M03RV9PredictiveWorkflowError(
                    "rank states diverged before update-64 checkpoint"
                )

            checkpoint_evidence: dict[int, tuple[Path, str]] = {}
            for horizon_index, horizon in enumerate((21, 30)):
                candidate = _clone_evaluation_policy(
                    training_policy,
                    worker,
                    horizon_index,
                    device,
                )
                candidate_optimizer, candidate_partition = (
                    build_m03r_v9_alpha_pretraining_optimizer(candidate)
                )
                candidate_optimizer.load_state_dict(optimizer.state_dict())
                checkpoint_file_sha: str | None = None
                checkpoint_path = (
                    output
                    / "checkpoints"
                    / f"fold-{fold.fold_index:02d}-horizon-{horizon:02d}-update-0064.pt"
                )
                if rank == 0:
                    checkpoint_file_sha = write_immutable_m03r_v9_alpha_checkpoint(
                        checkpoint_path,
                        candidate,
                        candidate_optimizer,
                        candidate_partition,
                        last_step,
                        setting_index=worker.setting_index,
                        fold_index=fold.fold_index,
                        rank=0,
                        world_size=world_size,
                        plan_sha256=package.package_plan_sha256,
                        source_array_sha256=last_step.source_array_sha256,
                        asset_axis_sha256=cache.action_hash,
                        risk_binding_sha256=risk_binding.binding_sha256,
                    )
                checkpoint_file_sha = _broadcast(checkpoint_file_sha, rank)
                if not isinstance(checkpoint_file_sha, str):
                    raise M03RV9PredictiveWorkflowError(
                        "update-64 checkpoint was not published"
                    )
                checkpoint_evidence[horizon] = (
                    checkpoint_path,
                    checkpoint_file_sha,
                )
                del candidate_optimizer, candidate

            # Both horizon identities are immutable before the qualification
            # chronology is opened.  The common causal covariance surface is
            # then qualified and moved to the device exactly once per fold.
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
                raise M03RV9PredictiveWorkflowError(
                    "qualified fold risk state diverged between ranks"
                )

            per_horizon: dict[str, Any] = {}
            for horizon_index, horizon in enumerate((21, 30)):
                candidate = _clone_evaluation_policy(
                    training_policy,
                    worker,
                    horizon_index,
                    device,
                )
                checkpoint_path, checkpoint_file_sha = checkpoint_evidence[horizon]
                evaluated = evaluate_m03r_v9_qualification_fold(
                    cache,
                    worker,
                    fold,
                    risk_source,
                    risk_binding,
                    projector,
                    risk_state,
                    candidate,
                    device=device,
                )
                result_hashes = _gather(
                    (
                        evaluated.alpha_evidence.receipt_sha256,
                        evaluated.sleeve_evidence.receipt_sha256,
                        evaluated.sleeve_trace.trace_sha256,
                    ),
                    world_size,
                )
                if len(set(result_hashes)) != 1:
                    raise M03RV9PredictiveWorkflowError(
                        "qualification result diverged between ranks"
                    )
                result_artifact_sha: str | None = None
                result_artifact = (
                    output
                    / "fold-artifacts"
                    / f"fold-{fold.fold_index:02d}-horizon-{horizon:02d}.pt"
                )
                if rank == 0:
                    result_artifact_sha = _write_immutable_torch(
                        result_artifact,
                        _fold_artifact_payload(evaluated),
                    )
                result_artifact_sha = _broadcast(result_artifact_sha, rank)
                if not isinstance(result_artifact_sha, str):
                    raise M03RV9PredictiveWorkflowError(
                        "qualification artifact was not published"
                    )
                per_horizon[str(horizon)] = {
                    "horizon_binding_sha256": (
                        candidate.horizon_binding.receipt_sha256
                    ),
                    "alpha_head_identity": asdict(candidate.alpha_head_identity()),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_file_sha256": checkpoint_file_sha,
                    "qualification_artifact_path": str(result_artifact),
                    "qualification_artifact_file_sha256": result_artifact_sha,
                    "alpha_evidence_sha256": (evaluated.alpha_evidence.receipt_sha256),
                    "sleeve_evidence_sha256": (
                        evaluated.sleeve_evidence.receipt_sha256
                    ),
                    "sleeve_trace_sha256": evaluated.sleeve_trace.trace_sha256,
                    "fold_risk_state_sha256": risk_state.state_sha256,
                }
                results[horizon].append(evaluated)
                del candidate
            if rank == 0:
                unsigned_fold = {
                    "schema": M03R_V9_FOLD_RESULT_SCHEMA,
                    "package_plan_sha256": package.package_plan_sha256,
                    "worker_plan_sha256": worker.receipt_sha256,
                    "setting_index": worker.setting_index,
                    "setting_id": worker.setting_id,
                    "fold_index": fold.fold_index,
                    "completed_updates": 64,
                    "early_stopping_enabled": False,
                    "qualification_evaluated_only_after_update64": True,
                    "rank_state_equal": True,
                    "model_state_sha256": model_hashes[0],
                    "optimizer_state_sha256": optimizer_hashes[0],
                    "step_receipt_sha256": [
                        row.receipt_sha256 for row in step_receipts
                    ],
                    "horizon_candidates": per_horizon,
                    "economic_optimizer_updates": 0,
                    "research_only": True,
                    "development_only": True,
                    "reportable": False,
                    "promotion_eligible": False,
                }
                fold_receipt = {
                    **unsigned_fold,
                    "receipt_sha256": _sha256_bytes(_canonical_bytes(unsigned_fold)),
                }
                fold_receipt_file_sha.append(
                    _write_immutable_json(
                        output / "receipts" / f"fold-{fold.fold_index:02d}.json",
                        fold_receipt,
                    )
                )
            dist.barrier()
            del risk_state
            del optimizer, training_policy

        terminal: dict[str, Any] | None = None
        if rank == 0:
            qualifications: list[M03RV9PredictiveQualification] = []
            for horizon_index, horizon in enumerate((21, 30)):
                qualifications.append(
                    qualify_m03r_v9_predictive_candidate(
                        setting_id=worker.setting_id,
                        horizon_binding=worker.horizon_bindings[horizon_index],
                        alpha_folds=tuple(
                            row.alpha_evidence for row in results[horizon]
                        ),
                        sleeve_folds=tuple(
                            row.sleeve_evidence for row in results[horizon]
                        ),
                    )
                )
            passed = tuple(row for row in qualifications if row.passed)
            selected = select_m03r_v9_horizon(passed) if passed else None
            unsigned_terminal = {
                "schema": M03R_V9_TERMINAL_SCHEMA,
                "package_plan_sha256": package.package_plan_sha256,
                "worker_plan_sha256": worker.receipt_sha256,
                "startup_file_sha256": startup_file_sha,
                "setting_index": worker.setting_index,
                "setting_id": worker.setting_id,
                "fold_receipt_file_sha256": fold_receipt_file_sha,
                "horizon_qualification": [
                    {**asdict(row), "receipt_sha256": row.receipt_sha256}
                    for row in qualifications
                ],
                "selected_horizon": (
                    None if selected is None else selected.selected_horizon_sessions
                ),
                "selected_qualification_sha256": (
                    None if selected is None else selected.receipt_sha256
                ),
                "predictive_gate_passed": selected is not None,
                "economic_generation_may_be_minted": selected is not None,
                "economic_panel_authorized": False,
                "economic_optimizer_updates": 0,
                "h100_capacity_evidence": True,
                "world_size": world_size,
                "gpus_per_worker": 2,
                "research_only": True,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
            terminal = {
                **unsigned_terminal,
                "receipt_sha256": _sha256_bytes(_canonical_bytes(unsigned_terminal)),
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
    parser.add_argument("--package-plan-sha256", required=True)
    parser.add_argument("--completion-index", type=int)
    parser.add_argument("--startup-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_m03r_v9_predictive_worker(
        args.package_plan,
        expected_package_plan_sha256=args.package_plan_sha256,
        completion_index=args.completion_index,
        startup_only=args.startup_only,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V9_CAPACITY_TERMINAL_SCHEMA",
    "M03R_V9_FOLD_RESULT_SCHEMA",
    "M03R_V9_STARTUP_SCHEMA",
    "M03R_V9_TERMINAL_SCHEMA",
    "M03RV9PredictiveWorkflowError",
    "main",
    "resolve_m03r_v9_completion_index",
    "run_m03r_v9_predictive_worker",
]
