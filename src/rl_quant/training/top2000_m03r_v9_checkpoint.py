"""Immutable update-64 predictive checkpoint for M03R-v9.

V9 does not resume V8 state and does not select an earlier checkpoint.  Each
rank publishes one final update-64 checkpoint whose model identity explicitly
binds the selected alpha horizon and both mean/scale head states.  The loader
is evaluation-only; it never restores optimizer or RNG state for continuation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PREDICTIVE_SPEC,
    M03R_V9_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9AlphaHeadIdentity,
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    M03RV9AlphaOptimizerPartition,
    validate_m03r_v9_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    M03RV9AlphaStepReceipt,
    model_state_sha256,
    optimizer_state_sha256,
    state_dict_sha256,
)

M03R_V9_ALPHA_CHECKPOINT_SCHEMA = "rl-quant.top2000-dev.m03r-v9-alpha-checkpoint-v1"
_MAX_CHECKPOINT_BYTES = 2 * 1024**3


class M03RV9AlphaCheckpointError(ValueError):
    """The update-64 checkpoint or external identity drifted."""


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
        raise M03RV9AlphaCheckpointError(f"{name} is not a lowercase SHA-256")
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


@dataclass(frozen=True, slots=True)
class M03RV9LoadedCheckpoint:
    completed_updates: int
    setting_index: int
    fold_index: int
    rank: int
    world_size: int
    model_state_sha256: str
    alpha_head_identity: M03RV9AlphaHeadIdentity
    checkpoint_file_sha256: str
    plan_sha256: str
    source_array_sha256: str
    asset_axis_sha256: str
    risk_binding_sha256: str


def _checkpoint_payload(
    policy: Top2000M03RV9PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV9AlphaOptimizerPartition,
    last_step: M03RV9AlphaStepReceipt,
    *,
    setting_index: int,
    fold_index: int,
    rank: int,
    world_size: int,
    plan_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
    risk_binding_sha256: str,
) -> dict[str, Any]:
    validate_m03r_v9_alpha_pretraining_optimizer(policy, optimizer, partition)
    last_step.validate()
    alpha_identity = policy.alpha_head_identity()
    if (
        setting_index != policy.setting.setting_index
        or fold_index != last_step.fold_index
        or last_step.setting_id != policy.setting.setting_id
        or last_step.completed_updates_after
        != M03R_V9_PREDICTIVE_SPEC.maximum_optimizer_updates
        or world_size not in {1, 2}
        or rank not in range(world_size)
        or last_step.distributed_world_size != world_size
        or last_step.source_array_sha256 != source_array_sha256
        or alpha_identity.horizon_binding_sha256
        != policy.horizon_binding.receipt_sha256
    ):
        raise M03RV9AlphaCheckpointError("checkpoint cursor or identity drifted")
    for name, value in (
        ("plan_sha256", plan_sha256),
        ("source_array_sha256", source_array_sha256),
        ("asset_axis_sha256", asset_axis_sha256),
        ("risk_binding_sha256", risk_binding_sha256),
    ):
        _digest(name, value)
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    return {
        "schema": M03R_V9_ALPHA_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V9_PROTOCOL_SHA256,
        "plan_sha256": plan_sha256,
        "setting_index": setting_index,
        "setting_id": policy.setting.setting_id,
        "fold_index": fold_index,
        "rank": rank,
        "world_size": world_size,
        "completed_updates": 64,
        "source_array_sha256": source_array_sha256,
        "asset_axis_sha256": asset_axis_sha256,
        "risk_binding_sha256": risk_binding_sha256,
        "horizon_binding_sha256": policy.horizon_binding.receipt_sha256,
        "selected_alpha_horizon": alpha_identity.selected_alpha_horizon,
        "alpha_mean_head_state_sha256": (alpha_identity.alpha_mean_head_state_sha256),
        "alpha_scale_head_state_sha256": (alpha_identity.alpha_scale_head_state_sha256),
        "alpha_distribution_contract_sha256": (
            alpha_identity.alpha_distribution_contract_sha256
        ),
        "alpha_head_identity": asdict(alpha_identity),
        "alpha_head_identity_sha256": _canonical_sha256(asdict(alpha_identity)),
        "optimizer_partition": asdict(partition),
        "optimizer_partition_sha256": partition.receipt_sha256,
        "model_state_dict": model_state,
        "model_state_sha256": state_dict_sha256(model_state),
        "optimizer_state_sha256": optimizer_state_sha256(optimizer),
        "last_step": asdict(last_step),
        "last_step_sha256": last_step.receipt_sha256,
        "early_stopping_enabled": False,
        "qualification_evaluated_before_checkpoint": False,
        "evaluation_only_load": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }


def write_immutable_m03r_v9_alpha_checkpoint(
    path: str | Path,
    policy: Top2000M03RV9PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV9AlphaOptimizerPartition,
    last_step: M03RV9AlphaStepReceipt,
    *,
    setting_index: int,
    fold_index: int,
    rank: int,
    world_size: int,
    plan_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
    risk_binding_sha256: str,
) -> str:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV9AlphaCheckpointError("checkpoint target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        policy,
        optimizer,
        partition,
        last_step,
        setting_index=setting_index,
        fold_index=fold_index,
        rank=rank,
        world_size=world_size,
        plan_sha256=plan_sha256,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
        risk_binding_sha256=risk_binding_sha256,
    )
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise M03RV9AlphaCheckpointError(
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


def load_m03r_v9_alpha_checkpoint_for_evaluation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_plan_sha256: str,
    expected_setting_index: int,
    expected_fold_index: int,
    expected_rank: int,
    expected_world_size: int,
    expected_source_array_sha256: str,
    expected_asset_axis_sha256: str,
    expected_risk_binding_sha256: str,
    policy: Top2000M03RV9PredictivePolicy,
) -> M03RV9LoadedCheckpoint:
    source = Path(path)
    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise M03RV9AlphaCheckpointError(
            "checkpoint must be a readable regular non-symlink"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_CHECKPOINT_BYTES
        ):
            raise M03RV9AlphaCheckpointError("checkpoint size or type is invalid")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV9AlphaCheckpointError("checkpoint file hash drifted")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV9AlphaCheckpointError("checkpoint changed while read")
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise M03RV9AlphaCheckpointError("checkpoint payload is not a mapping")
    required = {
        "schema": M03R_V9_ALPHA_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V9_PROTOCOL_SHA256,
        "plan_sha256": _digest("expected_plan_sha256", expected_plan_sha256),
        "setting_index": expected_setting_index,
        "setting_id": policy.setting.setting_id,
        "fold_index": expected_fold_index,
        "rank": expected_rank,
        "world_size": expected_world_size,
        "completed_updates": 64,
        "source_array_sha256": _digest(
            "expected_source_array_sha256", expected_source_array_sha256
        ),
        "asset_axis_sha256": _digest(
            "expected_asset_axis_sha256", expected_asset_axis_sha256
        ),
        "risk_binding_sha256": _digest(
            "expected_risk_binding_sha256", expected_risk_binding_sha256
        ),
        "horizon_binding_sha256": policy.horizon_binding.receipt_sha256,
        "early_stopping_enabled": False,
        "qualification_evaluated_before_checkpoint": False,
        "evaluation_only_load": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise M03RV9AlphaCheckpointError("checkpoint immutable identity drifted")
    nested = payload.get("alpha_head_identity")
    model_state = payload.get("model_state_dict")
    if not isinstance(nested, Mapping) or not isinstance(model_state, Mapping):
        raise M03RV9AlphaCheckpointError("checkpoint nested payload is malformed")
    alpha_identity = M03RV9AlphaHeadIdentity(**nested)
    alpha_identity.validate()
    if (
        payload.get("alpha_head_identity_sha256")
        != _canonical_sha256(asdict(alpha_identity))
        or payload.get("selected_alpha_horizon")
        != alpha_identity.selected_alpha_horizon
        or payload.get("alpha_mean_head_state_sha256")
        != alpha_identity.alpha_mean_head_state_sha256
        or payload.get("alpha_scale_head_state_sha256")
        != alpha_identity.alpha_scale_head_state_sha256
        or payload.get("alpha_distribution_contract_sha256")
        != alpha_identity.alpha_distribution_contract_sha256
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in model_state.items()
        )
        or payload.get("model_state_sha256") != state_dict_sha256(model_state)
    ):
        raise M03RV9AlphaCheckpointError("checkpoint alpha/model identity drifted")
    policy.load_state_dict(model_state, strict=True)
    if (
        model_state_sha256(policy) != payload.get("model_state_sha256")
        or policy.alpha_head_identity() != alpha_identity
    ):
        raise M03RV9AlphaCheckpointError("loaded v9 policy identity drifted")
    return M03RV9LoadedCheckpoint(
        completed_updates=64,
        setting_index=expected_setting_index,
        fold_index=expected_fold_index,
        rank=expected_rank,
        world_size=expected_world_size,
        model_state_sha256=payload["model_state_sha256"],
        alpha_head_identity=alpha_identity,
        checkpoint_file_sha256=expected_file_sha256,
        plan_sha256=expected_plan_sha256,
        source_array_sha256=expected_source_array_sha256,
        asset_axis_sha256=expected_asset_axis_sha256,
        risk_binding_sha256=expected_risk_binding_sha256,
    )


__all__ = [
    "M03R_V9_ALPHA_CHECKPOINT_SCHEMA",
    "M03RV9AlphaCheckpointError",
    "M03RV9LoadedCheckpoint",
    "load_m03r_v9_alpha_checkpoint_for_evaluation",
    "write_immutable_m03r_v9_alpha_checkpoint",
]
