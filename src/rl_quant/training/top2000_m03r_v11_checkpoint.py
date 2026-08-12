"""Immutable write-reload-evaluate checkpoint boundary for M03R-v11."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
    M03R_V11_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256

M03R_V11_ALPHA_CHECKPOINT_SCHEMA = "rl-quant.top2000-dev.m03r-v11-alpha-checkpoint-v1"
_MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024 * 1024
EvaluationResult = TypeVar("EvaluationResult")


class M03RV11CheckpointError(ValueError):
    """The v11 checkpoint artifact or round-trip evaluation drifted."""


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11CheckpointError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV11LoadedCheckpoint:
    setting_index: int
    setting_id: str
    fold_index: int
    completed_updates: int
    selected_horizon_sessions: int
    model_state_sha256: str
    checkpoint_file_sha256: str
    episode_schedule_sha256: str
    residual_operator_root_sha256: str
    source_array_sha256: str
    asset_axis_sha256: str
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256


def write_immutable_m03r_v11_alpha_checkpoint(
    path: str | Path,
    policy: Top2000M03RV9PredictivePolicy,
    *,
    setting_index: int,
    fold_index: int,
    selected_horizon_sessions: int,
    episode_schedule_sha256: str,
    residual_operator_root_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
) -> str:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV11CheckpointError("v11 checkpoint target already exists")
    if (
        setting_index not in range(3)
        or fold_index not in range(6)
        or selected_horizon_sessions not in {21, 30}
    ):
        raise M03RV11CheckpointError("v11 checkpoint scientific cursor drifted")
    for name, value in (
        ("episode_schedule_sha256", episode_schedule_sha256),
        ("residual_operator_root_sha256", residual_operator_root_sha256),
        ("source_array_sha256", source_array_sha256),
        ("asset_axis_sha256", asset_axis_sha256),
    ):
        _digest(name, value)
    state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    payload = {
        "schema": M03R_V11_ALPHA_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V11_PROTOCOL_SHA256,
        "setting_index": setting_index,
        "setting_id": M03R_V11_SETTING_IDS[setting_index],
        "fold_index": fold_index,
        "completed_updates": 64,
        "selected_horizon_sessions": selected_horizon_sessions,
        "episode_schedule_sha256": episode_schedule_sha256,
        "residual_operator_root_sha256": residual_operator_root_sha256,
        "source_array_sha256": source_array_sha256,
        "asset_axis_sha256": asset_axis_sha256,
        "model_state_dict": state,
        "model_state_sha256": state_dict_sha256(state),
        "evaluation_only_load": True,
        "predecessor_state_reused": False,
        "outer_2026_accessed": False,
        "economic_optimizer_updates": 0,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target, follow_symlinks=False)
        directory_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(target)


def load_m03r_v11_alpha_checkpoint_for_evaluation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_setting_index: int,
    expected_fold_index: int,
    expected_selected_horizon_sessions: int,
    expected_episode_schedule_sha256: str,
    expected_residual_operator_root_sha256: str,
    expected_source_array_sha256: str,
    expected_asset_axis_sha256: str,
    policy: Top2000M03RV9PredictivePolicy,
) -> M03RV11LoadedCheckpoint:
    source = Path(path)
    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV11CheckpointError(
            "v11 checkpoint must be a readable regular non-symlink"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_CHECKPOINT_BYTES
        ):
            raise M03RV11CheckpointError("v11 checkpoint size or type is invalid")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV11CheckpointError("v11 checkpoint file hash drifted")
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
            raise M03RV11CheckpointError("v11 checkpoint changed while read")
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise M03RV11CheckpointError("v11 checkpoint payload is not a mapping")
    required = {
        "schema": M03R_V11_ALPHA_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V11_PROTOCOL_SHA256,
        "setting_index": expected_setting_index,
        "setting_id": M03R_V11_SETTING_IDS[expected_setting_index],
        "fold_index": expected_fold_index,
        "completed_updates": 64,
        "selected_horizon_sessions": expected_selected_horizon_sessions,
        "episode_schedule_sha256": _digest(
            "expected_episode_schedule_sha256", expected_episode_schedule_sha256
        ),
        "residual_operator_root_sha256": _digest(
            "expected_residual_operator_root_sha256",
            expected_residual_operator_root_sha256,
        ),
        "source_array_sha256": _digest(
            "expected_source_array_sha256", expected_source_array_sha256
        ),
        "asset_axis_sha256": _digest(
            "expected_asset_axis_sha256", expected_asset_axis_sha256
        ),
        "evaluation_only_load": True,
        "predecessor_state_reused": False,
        "outer_2026_accessed": False,
        "economic_optimizer_updates": 0,
    }
    if any(payload.get(name) != value for name, value in required.items()):
        raise M03RV11CheckpointError("v11 checkpoint immutable identity drifted")
    state = payload.get("model_state_dict")
    if (
        not isinstance(state, Mapping)
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
        or payload.get("model_state_sha256") != state_dict_sha256(state)
    ):
        raise M03RV11CheckpointError("v11 checkpoint model state drifted")
    policy.load_state_dict(state, strict=True)
    if state_dict_sha256(policy.state_dict()) != payload["model_state_sha256"]:
        raise M03RV11CheckpointError("loaded v11 model state drifted")
    return M03RV11LoadedCheckpoint(
        setting_index=expected_setting_index,
        setting_id=M03R_V11_SETTING_IDS[expected_setting_index],
        fold_index=expected_fold_index,
        completed_updates=64,
        selected_horizon_sessions=expected_selected_horizon_sessions,
        model_state_sha256=payload["model_state_sha256"],
        checkpoint_file_sha256=expected_file_sha256,
        episode_schedule_sha256=expected_episode_schedule_sha256,
        residual_operator_root_sha256=expected_residual_operator_root_sha256,
        source_array_sha256=expected_source_array_sha256,
        asset_axis_sha256=expected_asset_axis_sha256,
    )


def write_reload_evaluate_m03r_v11_checkpoint(
    path: str | Path,
    policy: Top2000M03RV9PredictivePolicy,
    fresh_policy_factory: Callable[[], Top2000M03RV9PredictivePolicy],
    evaluator: Callable[
        [Top2000M03RV9PredictivePolicy, M03RV11LoadedCheckpoint], EvaluationResult
    ],
    *,
    setting_index: int,
    fold_index: int,
    selected_horizon_sessions: int,
    episode_schedule_sha256: str,
    residual_operator_root_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
) -> tuple[M03RV11LoadedCheckpoint, EvaluationResult]:
    """Write, discard the candidate reference, reload, and evaluate exact bytes."""

    file_sha256 = write_immutable_m03r_v11_alpha_checkpoint(
        path,
        policy,
        setting_index=setting_index,
        fold_index=fold_index,
        selected_horizon_sessions=selected_horizon_sessions,
        episode_schedule_sha256=episode_schedule_sha256,
        residual_operator_root_sha256=residual_operator_root_sha256,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
    )
    del policy
    loaded_policy = fresh_policy_factory()
    loaded = load_m03r_v11_alpha_checkpoint_for_evaluation(
        path,
        expected_file_sha256=file_sha256,
        expected_setting_index=setting_index,
        expected_fold_index=fold_index,
        expected_selected_horizon_sessions=selected_horizon_sessions,
        expected_episode_schedule_sha256=episode_schedule_sha256,
        expected_residual_operator_root_sha256=residual_operator_root_sha256,
        expected_source_array_sha256=source_array_sha256,
        expected_asset_axis_sha256=asset_axis_sha256,
        policy=loaded_policy,
    )
    return loaded, evaluator(loaded_policy, loaded)


__all__ = [
    "M03R_V11_ALPHA_CHECKPOINT_SCHEMA",
    "M03RV11CheckpointError",
    "M03RV11LoadedCheckpoint",
    "load_m03r_v11_alpha_checkpoint_for_evaluation",
    "write_immutable_m03r_v11_alpha_checkpoint",
    "write_reload_evaluate_m03r_v11_checkpoint",
]
