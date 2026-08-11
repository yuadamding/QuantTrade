"""Immutable checkpoint and validation-only resume for M03R-v8 pretraining."""

from __future__ import annotations

import hashlib
import json
import os
import random
import stat
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_optimizer import (
    M03RV8AlphaOptimizerPartition,
    validate_m03r_v8_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v8_pretraining_step import (
    M03RV8AlphaEarlyStoppingState,
    M03RV8AlphaStepReceipt,
    model_state_sha256,
    optimizer_state_dict_sha256,
    optimizer_state_sha256,
    state_dict_sha256,
)

M03R_V8_ALPHA_CHECKPOINT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-alpha-checkpoint-v1"
)
_MAX_CHECKPOINT_BYTES = 2 * 1024**3


class M03RV8AlphaCheckpointError(ValueError):
    """Checkpoint bytes, identity, or resume state are invalid."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV8AlphaCheckpointError(f"{name} is not a lowercase SHA-256")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_payload(
    policy: Top2000M03RV8DevelopmentPolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV8AlphaOptimizerPartition,
    early_stopping: M03RV8AlphaEarlyStoppingState,
    last_step: M03RV8AlphaStepReceipt,
    *,
    setting_index: int,
    fold_index: int,
    rank: int,
    world_size: int,
    completed_updates: int,
    plan_sha256: str,
    source_array_sha256: str,
) -> dict[str, Any]:
    validate_m03r_v8_alpha_pretraining_optimizer(policy, optimizer, partition)
    early_stopping.validate()
    last_step.validate()
    if (
        setting_index != policy.setting.setting_index
        or fold_index != last_step.fold_index
        or last_step.setting_id != policy.setting.setting_id
        or completed_updates != last_step.completed_updates_after
        or not 0 < completed_updates
        <= M03R_V8_ALPHA_PRETRAINING.maximum_optimizer_updates
        or world_size not in {1, 2}
        or rank not in range(world_size)
        or last_step.distributed_world_size != world_size
        or last_step.source_array_sha256 != source_array_sha256
    ):
        raise M03RV8AlphaCheckpointError("checkpoint cursor or identity drifted")
    _digest("plan_sha256", plan_sha256)
    _digest("source_array_sha256", source_array_sha256)
    parameter_device = next(policy.parameters()).device
    return {
        "schema": M03R_V8_ALPHA_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
        "plan_sha256": plan_sha256,
        "setting_index": setting_index,
        "setting_id": policy.setting.setting_id,
        "fold_index": fold_index,
        "rank": rank,
        "world_size": world_size,
        "completed_updates": completed_updates,
        "source_array_sha256": source_array_sha256,
        "optimizer_partition": asdict(partition),
        "optimizer_partition_sha256": partition.receipt_sha256,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        },
        "model_state_sha256": model_state_sha256(policy),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_state_sha256": optimizer_state_sha256(optimizer),
        "early_stopping": asdict(early_stopping),
        "early_stopping_sha256": early_stopping.receipt_sha256,
        "last_step": asdict(last_step),
        "last_step_sha256": last_step.receipt_sha256,
        "python_rng_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state(parameter_device).cpu().clone()
            if parameter_device.type == "cuda"
            else None
        ),
        "development_only": True,
        "promotion_eligible": False,
    }


def write_immutable_m03r_v8_alpha_checkpoint(
    path: str | Path,
    policy: Top2000M03RV8DevelopmentPolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV8AlphaOptimizerPartition,
    early_stopping: M03RV8AlphaEarlyStoppingState,
    last_step: M03RV8AlphaStepReceipt,
    *,
    setting_index: int,
    fold_index: int,
    rank: int,
    world_size: int,
    completed_updates: int,
    plan_sha256: str,
    source_array_sha256: str,
) -> str:
    """Publish one checkpoint atomically without replacing an existing path."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV8AlphaCheckpointError("checkpoint target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        policy,
        optimizer,
        partition,
        early_stopping,
        last_step,
        setting_index=setting_index,
        fold_index=fold_index,
        rank=rank,
        world_size=world_size,
        completed_updates=completed_updates,
        plan_sha256=plan_sha256,
        source_array_sha256=source_array_sha256,
    )
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise M03RV8AlphaCheckpointError(
                "checkpoint target appeared during publication"
            ) from exc
        directory_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(target)


def load_m03r_v8_alpha_checkpoint(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_plan_sha256: str,
    expected_setting_index: int,
    expected_fold_index: int,
    expected_rank: int,
    expected_world_size: int,
    expected_source_array_sha256: str,
    policy: Top2000M03RV8DevelopmentPolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV8AlphaOptimizerPartition,
) -> tuple[int, M03RV8AlphaEarlyStoppingState, M03RV8AlphaStepReceipt]:
    """Validate exact bytes and restore one already-constructed policy/optimizer."""

    source = Path(path)
    expected_file_sha256 = _digest(
        "expected_file_sha256", expected_file_sha256
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise M03RV8AlphaCheckpointError(
            "checkpoint must be a readable regular non-symlink"
        ) from exc
    try:
        stat_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(stat_before.st_mode)
            or stat_before.st_size <= 0
            or stat_before.st_size > _MAX_CHECKPOINT_BYTES
        ):
            raise M03RV8AlphaCheckpointError("checkpoint size or type is invalid")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV8AlphaCheckpointError("checkpoint file hash drifted")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        stat_after = os.fstat(descriptor)
        if (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
        ) != (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
        ):
            raise M03RV8AlphaCheckpointError("checkpoint changed while it was read")
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise M03RV8AlphaCheckpointError("checkpoint payload is not a mapping")
    required = {
        "schema": M03R_V8_ALPHA_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
        "plan_sha256": _digest("expected_plan_sha256", expected_plan_sha256),
        "setting_index": expected_setting_index,
        "setting_id": policy.setting.setting_id,
        "fold_index": expected_fold_index,
        "rank": expected_rank,
        "world_size": expected_world_size,
        "source_array_sha256": _digest(
            "expected_source_array_sha256", expected_source_array_sha256
        ),
        "optimizer_partition_sha256": partition.receipt_sha256,
        "development_only": True,
        "promotion_eligible": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise M03RV8AlphaCheckpointError("checkpoint immutable identity drifted")
    try:
        observed_partition = M03RV8AlphaOptimizerPartition(
            **payload["optimizer_partition"]
        )
        early_stopping = M03RV8AlphaEarlyStoppingState(**payload["early_stopping"])
        last_step = M03RV8AlphaStepReceipt(**payload["last_step"])
    except (KeyError, TypeError) as exc:
        raise M03RV8AlphaCheckpointError(
            "checkpoint nested receipt is malformed"
        ) from exc
    observed_partition.validate()
    early_stopping.validate()
    last_step.validate()
    completed_updates = payload.get("completed_updates")
    if (
        observed_partition != partition
        or payload.get("early_stopping_sha256") != early_stopping.receipt_sha256
        or payload.get("last_step_sha256") != last_step.receipt_sha256
        or not isinstance(completed_updates, int)
        or isinstance(completed_updates, bool)
        or completed_updates != last_step.completed_updates_after
        or not 0 < completed_updates
        <= M03R_V8_ALPHA_PRETRAINING.maximum_optimizer_updates
        or not isinstance(payload.get("model_state_dict"), Mapping)
        or not isinstance(payload.get("optimizer_state_dict"), Mapping)
        or not isinstance(payload.get("torch_cpu_rng_state"), torch.Tensor)
    ):
        raise M03RV8AlphaCheckpointError("checkpoint semantic cursor drifted")
    model_state = payload["model_state_dict"]
    optimizer_state = payload["optimizer_state_dict"]
    if (
        any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in model_state.items()
        )
        or payload.get("model_state_sha256") != state_dict_sha256(model_state)
        or payload.get("optimizer_state_sha256")
        != optimizer_state_dict_sha256(optimizer_state)
    ):
        raise M03RV8AlphaCheckpointError(
            "checkpoint tensor state hashes do not reconcile before restore"
        )
    policy.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    validate_m03r_v8_alpha_pretraining_optimizer(policy, optimizer, partition)
    if (
        payload.get("model_state_sha256") != model_state_sha256(policy)
        or payload.get("optimizer_state_sha256") != optimizer_state_sha256(optimizer)
    ):
        raise M03RV8AlphaCheckpointError("checkpoint state hashes do not reconcile")
    python_rng_state = payload.get("python_rng_state")
    cuda_rng_state = payload.get("torch_cuda_rng_state")
    if not isinstance(python_rng_state, tuple):
        raise M03RV8AlphaCheckpointError("checkpoint Python RNG state is malformed")
    random.setstate(python_rng_state)
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    parameter_device = next(policy.parameters()).device
    if parameter_device.type == "cuda":
        if not isinstance(cuda_rng_state, torch.Tensor):
            raise M03RV8AlphaCheckpointError("checkpoint CUDA RNG state is missing")
        torch.cuda.set_rng_state(cuda_rng_state, parameter_device)
    elif cuda_rng_state is not None:
        raise M03RV8AlphaCheckpointError("CPU resume received CUDA RNG state")
    return completed_updates, early_stopping, last_step


__all__ = [
    "M03R_V8_ALPHA_CHECKPOINT_SCHEMA",
    "M03RV8AlphaCheckpointError",
    "load_m03r_v8_alpha_checkpoint",
    "write_immutable_m03r_v8_alpha_checkpoint",
]
