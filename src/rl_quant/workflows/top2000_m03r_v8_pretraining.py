"""Two-H100 worker for the M03R-v8 predictive pretraining stage.

This worker does not perform economic fine tuning.  It trains and qualifies
the seven rows whose frozen protocol requires predictive pretraining.  The
no-pretraining row remains untouched for the later, separately gated economic
stage.
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

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03RV8AlphaFoldEvidence,
    qualify_m03r_v8_alpha_panel,
)
from rl_quant.training.top2000_m03r_v8_plan import (
    M03RV8DevelopmentTrainingPlan,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_checkpoint import (
    load_m03r_v8_alpha_checkpoint,
    write_immutable_m03r_v8_alpha_checkpoint,
)
from rl_quant.training.top2000_m03r_v8_pretraining_contract import (
    M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION,
    M03R_V8_PRETRAINING_WORKER_SCHEMA,
)
from rl_quant.training.top2000_m03r_v8_pretraining_fold import (
    evaluate_m03r_v8_pretraining_fold,
    run_m03r_v8_pretraining_fold_update,
)
from rl_quant.training.top2000_m03r_v8_pretraining_optimizer import (
    build_m03r_v8_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v8_pretraining_step import (
    M03RV8AlphaEarlyStoppingState,
    M03RV8AlphaStepReceipt,
    advance_m03r_v8_alpha_early_stopping,
    model_state_sha256,
    optimizer_state_sha256,
)

M03R_V8_PRETRAINING_CHECKPOINT_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-pretraining-checkpoint-manifest-v1"
)
M03R_V8_PRETRAINING_FOLD_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-pretraining-fold-v1"
)
M03R_V8_PRETRAINING_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-pretraining-terminal-v1"
)


class M03RV8PretrainingWorkerError(RuntimeError):
    """The worker environment, checkpoint, or terminal receipt is invalid."""


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


def _rng_state_sha256(device: torch.device) -> str:
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("ascii"))
    cpu = torch.get_rng_state().contiguous()
    digest.update(cpu.numpy().tobytes())
    cuda = torch.cuda.get_rng_state(device).cpu().contiguous()
    digest.update(cuda.numpy().tobytes())
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == encoded:
            return _sha256_bytes(encoded)
        raise M03RV8PretrainingWorkerError(
            f"immutable receipt collision at {path}"
        ) from exc
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(encoded)


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise M03RV8PretrainingWorkerError(f"required receipt is absent: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise M03RV8PretrainingWorkerError(f"invalid JSON receipt {path}") from exc
    if not isinstance(payload, dict):
        raise M03RV8PretrainingWorkerError(f"receipt is not an object: {path}")
    return payload


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
    except (KeyError, ValueError) as exc:
        raise M03RV8PretrainingWorkerError(
            "worker requires torchrun rank environment"
        ) from exc
    if world_size != 2 or rank not in range(2) or local_rank not in range(2):
        raise M03RV8PretrainingWorkerError(
            "v8 pretraining requires exactly two local ranks"
        )
    try:
        restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    except ValueError as exc:
        raise M03RV8PretrainingWorkerError(
            "TORCHELASTIC_RESTART_COUNT must be an integer"
        ) from exc
    if restart_count not in {0, 1}:
        raise M03RV8PretrainingWorkerError(
            "v8 pretraining permits exactly one torchrun restart"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise M03RV8PretrainingWorkerError(
            "v8 pretraining requires exactly two visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if (
        torch.cuda.get_device_name(device) != "NVIDIA H100 80GB HBM3"
        or not 79 * 1024**3 <= properties.total_memory <= 81 * 1024**3
        or (properties.major, properties.minor) != (9, 0)
    ):
        raise M03RV8PretrainingWorkerError(
            "each rank requires one NVIDIA H100 80GB HBM3"
        )
    owns = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        owns = True
    if dist.get_world_size() != 2 or dist.get_rank() != rank:
        raise M03RV8PretrainingWorkerError("process-group identity drifted")
    return rank, local_rank, world_size, device, owns


def _gather(value: Any, world_size: int) -> list[Any]:
    values: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(values, value)
    return values


def _broadcast(value: Any, rank: int) -> Any:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _checkpoint_manifest_path(root: Path, fold_index: int, update: int) -> Path:
    return root / "checkpoints" / f"fold-{fold_index:02d}-update-{update:04d}.json"


def _checkpoint_path(root: Path, fold_index: int, update: int) -> Path:
    return root / "checkpoints" / f"fold-{fold_index:02d}-update-{update:04d}.pt"


def _find_resume_manifest(
    root: Path,
    fold_index: int,
    maximum_updates: int,
    *,
    rank: int,
) -> dict[str, Any] | None:
    payload: dict[str, Any] | None = None
    if rank == 0:
        interval = M03R_V8_ALPHA_PRETRAINING.inner_validation_interval_updates
        for update in range(maximum_updates - maximum_updates % interval, 0, -interval):
            path = _checkpoint_manifest_path(root, fold_index, update)
            if path.is_file() and not path.is_symlink():
                candidate = _load_json(path)
                if (
                    candidate.get("schema")
                    != M03R_V8_PRETRAINING_CHECKPOINT_MANIFEST_SCHEMA
                    or candidate.get("fold_index") != fold_index
                    or candidate.get("completed_updates") != update
                    or candidate.get("manifest_sha256")
                    != _sha256_bytes(
                        _canonical_bytes(
                            {
                                key: value
                                for key, value in candidate.items()
                                if key != "manifest_sha256"
                            }
                        )
                    )
                ):
                    raise M03RV8PretrainingWorkerError(
                        "checkpoint manifest identity drifted"
                    )
                payload = candidate
                break
    resolved = _broadcast(payload, rank)
    if resolved is not None and not isinstance(resolved, dict):
        raise M03RV8PretrainingWorkerError("checkpoint broadcast is malformed")
    return resolved


def _publish_checkpoint(
    root: Path,
    plan: M03RV8DevelopmentTrainingPlan,
    plan_file_sha256: str,
    fold_index: int,
    policy: Top2000M03RV8DevelopmentPolicy,
    optimizer: torch.optim.Optimizer,
    partition: Any,
    early: M03RV8AlphaEarlyStoppingState,
    step: M03RV8AlphaStepReceipt,
    *,
    rank: int,
    world_size: int,
) -> None:
    model_hashes = _gather(model_state_sha256(policy), world_size)
    optimizer_hashes = _gather(optimizer_state_sha256(optimizer), world_size)
    step_hashes = _gather(step.receipt_sha256, world_size)
    rng_hashes = _gather(_rng_state_sha256(next(policy.parameters()).device), world_size)
    if (
        len(set(model_hashes)) != 1
        or len(set(optimizer_hashes)) != 1
        or len(set(step_hashes)) != 1
        or len(set(rng_hashes)) != 1
    ):
        raise M03RV8PretrainingWorkerError(
            "rank states diverged before checkpoint publication"
        )
    completed = step.completed_updates_after
    checkpoint = _checkpoint_path(root, fold_index, completed)
    checkpoint_hash: str | None = None
    if rank == 0:
        checkpoint_hash = write_immutable_m03r_v8_alpha_checkpoint(
            checkpoint,
            policy,
            optimizer,
            partition,
            early,
            step,
            setting_index=plan.setting_index,
            fold_index=fold_index,
            rank=0,
            world_size=world_size,
            completed_updates=completed,
            plan_sha256=plan.receipt_sha256,
            source_array_sha256=step.source_array_sha256,
        )
        unsigned = {
            "schema": M03R_V8_PRETRAINING_CHECKPOINT_MANIFEST_SCHEMA,
            "worker_schema": M03R_V8_PRETRAINING_WORKER_SCHEMA,
            "plan_file_sha256": plan_file_sha256,
            "plan_sha256": plan.receipt_sha256,
            "setting_index": plan.setting_index,
            "setting_id": plan.setting_id,
            "fold_index": fold_index,
            "completed_updates": completed,
            "checkpoint_path": str(checkpoint),
            "checkpoint_file_sha256": checkpoint_hash,
            "source_array_sha256": step.source_array_sha256,
            "model_state_sha256": model_hashes[0],
            "optimizer_state_sha256": optimizer_hashes[0],
            "step_receipt_sha256": step_hashes[0],
            "early_stopping_sha256": early.receipt_sha256,
            "rank_state_equal": True,
            "rank_rng_state_sha256": rng_hashes[0],
            "rank_rng_state_equal": True,
            "development_only": True,
            "promotion_eligible": False,
        }
        payload = {**unsigned, "manifest_sha256": _sha256_bytes(_canonical_bytes(unsigned))}
        _write_immutable_json(
            _checkpoint_manifest_path(root, fold_index, completed), payload
        )
    dist.barrier()


def _resume_checkpoint(
    manifest: dict[str, Any],
    plan: M03RV8DevelopmentTrainingPlan,
    fold_index: int,
    policy: Top2000M03RV8DevelopmentPolicy,
    optimizer: torch.optim.Optimizer,
    partition: Any,
) -> tuple[int, M03RV8AlphaEarlyStoppingState, M03RV8AlphaStepReceipt]:
    if (
        manifest.get("plan_sha256") != plan.receipt_sha256
        or manifest.get("setting_id") != plan.setting_id
        or manifest.get("fold_index") != fold_index
        or manifest.get("rank_state_equal") is not True
        or manifest.get("rank_rng_state_equal") is not True
    ):
        raise M03RV8PretrainingWorkerError("resume manifest does not bind this cell")
    result = load_m03r_v8_alpha_checkpoint(
        manifest["checkpoint_path"],
        expected_file_sha256=manifest["checkpoint_file_sha256"],
        expected_plan_sha256=plan.receipt_sha256,
        expected_setting_index=plan.setting_index,
        expected_fold_index=fold_index,
        expected_rank=0,
        expected_world_size=plan.expected_world_size,
        expected_source_array_sha256=manifest["source_array_sha256"],
        policy=policy,
        optimizer=optimizer,
        partition=partition,
    )
    if (
        manifest.get("model_state_sha256") != model_state_sha256(policy)
        or manifest.get("optimizer_state_sha256") != optimizer_state_sha256(optimizer)
        or manifest.get("step_receipt_sha256") != result[2].receipt_sha256
        or manifest.get("early_stopping_sha256") != result[1].receipt_sha256
        or manifest.get("rank_rng_state_sha256")
        != _rng_state_sha256(next(policy.parameters()).device)
    ):
        raise M03RV8PretrainingWorkerError("resumed state does not match manifest")
    return result


def _write_selected_model(
    path: Path,
    plan: M03RV8DevelopmentTrainingPlan,
    fold_index: int,
    policy: Top2000M03RV8DevelopmentPolicy,
    evidence: M03RV8AlphaFoldEvidence,
    early: M03RV8AlphaEarlyStoppingState,
) -> str:
    if early.best_update is None or early.best_model_state_sha256 != model_state_sha256(policy):
        raise M03RV8PretrainingWorkerError("selected model does not match early-stop state")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise M03RV8PretrainingWorkerError("selected model target already exists")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(
        {
            "schema": M03R_V8_PRETRAINING_FOLD_SCHEMA,
            "plan_sha256": plan.receipt_sha256,
            "setting_index": plan.setting_index,
            "setting_id": plan.setting_id,
            "fold_index": fold_index,
            "selected_update": early.best_update,
            "model_state_sha256": early.best_model_state_sha256,
            "fold_evidence_sha256": evidence.receipt_sha256,
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            },
            "development_only": True,
            "promotion_eligible": False,
        },
        temporary,
    )
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        raise M03RV8PretrainingWorkerError("selected model publication raced") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _existing_fold_evidence(
    output: Path,
    plan: M03RV8DevelopmentTrainingPlan,
    plan_file_sha256: str,
    fold_index: int,
    *,
    rank: int,
) -> M03RV8AlphaFoldEvidence | None:
    resolved: dict[str, Any] | None = None
    if rank == 0:
        receipt_path = output / "receipts" / f"fold-{fold_index:02d}.json"
        if receipt_path.is_file() and not receipt_path.is_symlink():
            payload = _load_json(receipt_path)
            unsigned = {
                key: value for key, value in payload.items() if key != "receipt_sha256"
            }
            if (
                payload.get("schema") != M03R_V8_PRETRAINING_FOLD_SCHEMA
                or payload.get("plan_file_sha256") != plan_file_sha256
                or payload.get("plan_sha256") != plan.receipt_sha256
                or payload.get("setting_index") != plan.setting_index
                or payload.get("setting_id") != plan.setting_id
                or payload.get("fold_index") != fold_index
                or payload.get("receipt_sha256")
                != _sha256_bytes(_canonical_bytes(unsigned))
                or payload.get("rank_state_equal") is not True
            ):
                raise M03RV8PretrainingWorkerError(
                    "existing fold receipt identity drifted"
                )
            try:
                evidence = M03RV8AlphaFoldEvidence(**payload["fold_evidence"])
            except (KeyError, TypeError) as exc:
                raise M03RV8PretrainingWorkerError(
                    "existing fold evidence is malformed"
                ) from exc
            model_path = Path(payload["model_path"])
            if (
                payload.get("fold_evidence_sha256") != evidence.receipt_sha256
                or model_path.is_symlink()
                or not model_path.is_file()
                or _file_sha256(model_path) != payload.get("model_file_sha256")
            ):
                raise M03RV8PretrainingWorkerError(
                    "existing fold artifacts do not reconcile"
                )
            resolved = asdict(evidence)
    broadcast = _broadcast(resolved, rank)
    if broadcast is None:
        return None
    if not isinstance(broadcast, dict):
        raise M03RV8PretrainingWorkerError("fold evidence broadcast is malformed")
    evidence = M03RV8AlphaFoldEvidence(**broadcast)
    evidence.__post_init__()
    return evidence


def _load_best_checkpoint_into_policy(
    root: Path,
    fold_index: int,
    early: M03RV8AlphaEarlyStoppingState,
    policy: Top2000M03RV8DevelopmentPolicy,
) -> None:
    if early.best_update is None or early.best_model_state_sha256 is None:
        raise M03RV8PretrainingWorkerError("fold has no selected predictive checkpoint")
    manifest = _load_json(
        _checkpoint_manifest_path(root, fold_index, early.best_update)
    )
    checkpoint = Path(manifest["checkpoint_path"])
    if _file_sha256(checkpoint) != manifest["checkpoint_file_sha256"]:
        raise M03RV8PretrainingWorkerError("selected checkpoint file hash drifted")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("model_state_sha256") != early.best_model_state_sha256
    ):
        raise M03RV8PretrainingWorkerError("selected checkpoint semantic hash drifted")
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    if model_state_sha256(policy) != early.best_model_state_sha256:
        raise M03RV8PretrainingWorkerError("selected model restore failed")


def run_pretraining_worker(
    plan_path: str | Path,
    *,
    qualification_updates: int | None = None,
) -> dict[str, Any] | None:
    """Run one setting through six predictive folds or one bounded gate fold."""

    plan_file = Path(plan_path)
    plan_file_sha256 = _file_sha256(plan_file)
    plan = M03RV8DevelopmentTrainingPlan(**_load_json(plan_file))
    plan.validate()
    if not plan.alpha_pretraining_required:
        raise M03RV8PretrainingWorkerError(
            "the no-pretraining control has no predictive worker stage"
        )
    if qualification_updates is not None and qualification_updates != 4:
        raise M03RV8PretrainingWorkerError(
            "bounded qualification is exactly four updates"
        )
    maximum_updates = (
        plan.alpha_pretraining_updates
        if qualification_updates is None
        else qualification_updates
    )
    rank, _local_rank, world_size, device, owns = _distributed_context()
    output = Path(plan.output_root) / (
        "pretraining" if qualification_updates is None else "qualification-pretraining"
    )
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        properties = torch.cuda.get_device_properties(device)
        startup_rows = _gather(
            {
                "rank": rank,
                "local_rank": _local_rank,
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
        startup_receipt_sha256: str | None = None
        if rank == 0:
            startup_receipt_sha256 = _write_immutable_json(
                output / "two-h100-startup.json",
                {
                    "schema": "rl-quant.top2000-dev.m03r-v8-two-h100-startup-v1",
                    "worker_schema": M03R_V8_PRETRAINING_WORKER_SCHEMA,
                    "plan_file_sha256": plan_file_sha256,
                    "plan_sha256": plan.receipt_sha256,
                    "rank_runtime": startup_rows,
                    "exact_h100_80gb_per_rank": True,
                    "nccl_process_group_initialized": True,
                    "development_only": True,
                    "promotion_eligible": False,
                },
            )
        startup_receipt_sha256 = _broadcast(startup_receipt_sha256, rank)
        if not isinstance(startup_receipt_sha256, str):
            raise M03RV8PretrainingWorkerError("startup receipt was not published")
        cache = load_verified_top2000_hold30_development_cache(
            plan.cache_path,
            expected_cache_sha256=plan.cache_sha256,
            acknowledgement=DEVELOPMENT_ACK,
        )
        if cache.daily_ohlcv.shape[0] != TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS:
            raise M03RV8PretrainingWorkerError("cache state geometry drifted")
        folds = render_top2000_m03r_v7_development_folds(
            TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        )
        fold_evidence: list[M03RV8AlphaFoldEvidence] = []
        fold_count = 1 if qualification_updates is not None else len(folds)
        for fold in folds[:fold_count]:
            existing = _existing_fold_evidence(
                output,
                plan,
                plan_file_sha256,
                fold.fold_index,
                rank=rank,
            )
            if existing is not None:
                fold_evidence.append(existing)
                dist.barrier()
                continue
            _seed_everything(plan.seed)
            policy = Top2000M03RV8DevelopmentPolicy(
                plan.setting_index,
                token_dim=plan.token_dim,
                raw_stock_chunk=plan.raw_stock_chunk,
                activation_checkpointing=plan.activation_checkpointing,
            ).to(device)
            optimizer, partition = build_m03r_v8_alpha_pretraining_optimizer(policy)
            early = M03RV8AlphaEarlyStoppingState()
            completed = 0
            last_step: M03RV8AlphaStepReceipt | None = None
            manifest = _find_resume_manifest(
                output,
                fold.fold_index,
                maximum_updates,
                rank=rank,
            )
            if manifest is not None:
                completed, early, last_step = _resume_checkpoint(
                    manifest,
                    plan,
                    fold.fold_index,
                    policy,
                    optimizer,
                    partition,
                )
            while completed < maximum_updates and not early.stopped:
                last_step = run_m03r_v8_pretraining_fold_update(
                    cache,
                    plan,
                    fold,
                    policy,
                    optimizer,
                    partition,
                    completed_updates=completed,
                    distributed_rank=rank,
                    distributed_world_size=world_size,
                    device=device,
                )
                completed = last_step.completed_updates_after
                if completed % M03R_V8_ALPHA_PRETRAINING.inner_validation_interval_updates == 0:
                    evidence = evaluate_m03r_v8_pretraining_fold(
                        cache,
                        plan,
                        fold,
                        policy,
                        device=device,
                    )
                    early = advance_m03r_v8_alpha_early_stopping(
                        early,
                        completed_updates=completed,
                        evidence=evidence,
                        model_state_sha256_value=model_state_sha256(policy),
                    )
                    _publish_checkpoint(
                        output,
                        plan,
                        plan_file_sha256,
                        fold.fold_index,
                        policy,
                        optimizer,
                        partition,
                        early,
                        last_step,
                        rank=rank,
                        world_size=world_size,
                    )
            if last_step is None or early.best_update is None:
                raise M03RV8PretrainingWorkerError("fold produced no selected checkpoint")
            _load_best_checkpoint_into_policy(output, fold.fold_index, early, policy)
            evidence = evaluate_m03r_v8_pretraining_fold(
                cache,
                plan,
                fold,
                policy,
                device=device,
            )
            hashes = _gather(model_state_sha256(policy), world_size)
            evidence_hashes = _gather(evidence.receipt_sha256, world_size)
            if len(set(hashes)) != 1 or len(set(evidence_hashes)) != 1:
                raise M03RV8PretrainingWorkerError("selected fold result diverged by rank")
            if rank == 0:
                model_path = output / "models" / f"fold-{fold.fold_index:02d}.pt"
                model_file_sha256 = _write_selected_model(
                    model_path,
                    plan,
                    fold.fold_index,
                    policy,
                    evidence,
                    early,
                )
                unsigned = {
                    "schema": M03R_V8_PRETRAINING_FOLD_SCHEMA,
                    "worker_schema": M03R_V8_PRETRAINING_WORKER_SCHEMA,
                    "plan_file_sha256": plan_file_sha256,
                    "plan_sha256": plan.receipt_sha256,
                    "setting_index": plan.setting_index,
                    "setting_id": plan.setting_id,
                    "fold_index": fold.fold_index,
                    "selected_update": early.best_update,
                    "early_stopping_sha256": early.receipt_sha256,
                    "fold_evidence": asdict(evidence),
                    "fold_evidence_sha256": evidence.receipt_sha256,
                    "model_path": str(model_path),
                    "model_file_sha256": model_file_sha256,
                    "model_state_sha256": hashes[0],
                    "rank_state_equal": True,
                    "development_only": True,
                    "promotion_eligible": False,
                }
                _write_immutable_json(
                    output / "receipts" / f"fold-{fold.fold_index:02d}.json",
                    {
                        **unsigned,
                        "receipt_sha256": _sha256_bytes(_canonical_bytes(unsigned)),
                    },
                )
            fold_evidence.append(evidence)
            dist.barrier()
        terminal: dict[str, Any] | None = None
        peak_rows = _gather(
            {
                "rank": rank,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
            world_size,
        )
        if rank == 0:
            qualification = (
                None
                if qualification_updates is not None
                else qualify_m03r_v8_alpha_panel(tuple(fold_evidence))
            )
            unsigned = {
                "schema": M03R_V8_PRETRAINING_TERMINAL_SCHEMA,
                "worker_schema": M03R_V8_PRETRAINING_WORKER_SCHEMA,
                "mode": (
                    "four-update-two-h100-qualification"
                    if qualification_updates is not None
                    else "six-fold-alpha-pretraining"
                ),
                "plan_file_sha256": plan_file_sha256,
                "plan_sha256": plan.receipt_sha256,
                "startup_receipt_sha256": startup_receipt_sha256,
                "setting_index": plan.setting_index,
                "setting_id": plan.setting_id,
                "fold_receipt_sha256": [
                    _file_sha256(output / "receipts" / f"fold-{index:02d}.json")
                    for index in range(fold_count)
                ],
                "alpha_panel_qualification": (
                    None if qualification is None else asdict(qualification)
                ),
                "alpha_panel_qualification_sha256": (
                    None if qualification is None else qualification.receipt_sha256
                ),
                "passed": True if qualification is None else qualification.passed,
                "h100_capacity_evidence": qualification_updates is not None,
                "world_size": world_size,
                "gpus_per_worker": 2,
                "rank_peak_cuda_memory": peak_rows,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
            terminal = {
                **unsigned,
                "receipt_sha256": _sha256_bytes(_canonical_bytes(unsigned)),
            }
            _write_immutable_json(output / "pretraining-terminal.json", terminal)
        dist.barrier()
        return terminal
    finally:
        if owns and dist.is_initialized():
            dist.destroy_process_group()


def resolve_m03r_v8_pretraining_plan_path(
    *,
    plan: str | None,
    plan_directory: str | None,
    completion_index: int | None,
) -> Path:
    """Resolve one explicit plan or one exact seven-row Indexed-Job mapping."""

    if plan is not None:
        if plan_directory is not None or completion_index is not None:
            raise M03RV8PretrainingWorkerError(
                "explicit plan cannot be combined with an Indexed-Job mapping"
            )
        return Path(plan)
    if plan_directory is None:
        raise M03RV8PretrainingWorkerError("plan or plan-directory is required")
    if completion_index is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        try:
            completion_index = int(raw) if raw is not None else None
        except ValueError as exc:
            raise M03RV8PretrainingWorkerError(
                "JOB_COMPLETION_INDEX must be an integer"
            ) from exc
    if completion_index is None or not 0 <= completion_index < len(
        M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION
    ):
        raise M03RV8PretrainingWorkerError(
            "pretraining completion index must be in [0, 6]"
        )
    setting_index = M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION[
        completion_index
    ]
    return Path(plan_directory) / f"setting-{setting_index:02d}.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--plan-directory")
    parser.add_argument("--completion-index", type=int)
    parser.add_argument("--qualification-updates", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan_path = resolve_m03r_v8_pretraining_plan_path(
        plan=args.plan,
        plan_directory=args.plan_directory,
        completion_index=args.completion_index,
    )
    run_pretraining_worker(
        plan_path,
        qualification_updates=args.qualification_updates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "M03R_V8_PRETRAINING_CHECKPOINT_MANIFEST_SCHEMA",
    "M03R_V8_PRETRAINING_FOLD_SCHEMA",
    "M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION",
    "M03R_V8_PRETRAINING_TERMINAL_SCHEMA",
    "M03R_V8_PRETRAINING_WORKER_SCHEMA",
    "M03RV8PretrainingWorkerError",
    "main",
    "resolve_m03r_v8_pretraining_plan_path",
    "run_pretraining_worker",
]
