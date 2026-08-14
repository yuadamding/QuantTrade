"""Immutable direct-h3 checkpoint round trip for M03R-v13."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import torch

from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import (
    M03R_V13_PROTOCOL_SHA256,
    M03R_V13_SELECTED_HORIZON_SESSIONS,
    M03R_V13_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256
from rl_quant.training.top2000_m03r_v13_fold import render_m03r_v13_fold_geometries
from rl_quant.training.top2000_m03r_v13_policy import (
    M03RV13HeadIdentity,
    Top2000M03RV13PredictivePolicy,
)

M03R_V13_ALPHA_CHECKPOINT_SCHEMA = "rl-quant.top2000-dev.m03r-v13-alpha-checkpoint-v1"
_MAX_CHECKPOINT_BYTES = 8 * 1024**3
EvaluationResult = TypeVar("EvaluationResult")


class M03RV13CheckpointError(ValueError):
    """The v13 checkpoint artifact or exact reload drifted."""


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV13CheckpointError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _expected_updates(fold_index: int) -> int:
    if isinstance(fold_index, bool) or fold_index not in range(6):
        raise M03RV13CheckpointError("v13 checkpoint fold index drifted")
    try:
        return render_m03r_v13_fold_geometries(1001)[fold_index].optimizer_updates
    except (IndexError, TypeError) as exc:
        raise M03RV13CheckpointError("v13 checkpoint fold index drifted") from exc


@dataclass(frozen=True, slots=True)
class M03RV13LoadedCheckpoint:
    setting_index: int
    setting_id: str
    fold_index: int
    completed_updates: int
    selected_horizon_sessions: int
    model_state_sha256: str
    checkpoint_file_sha256: str
    episode_schedule_sha256: str
    target_residual_operator_root_sha256: str
    action_residual_operator_root_sha256: str
    source_array_sha256: str
    asset_axis_sha256: str
    head_identity: M03RV13HeadIdentity
    protocol_sha256: str = M03R_V13_PROTOCOL_SHA256

    def validate(self) -> None:
        self.head_identity.validate()
        if (
            self.setting_index not in range(len(M03R_V13_SETTING_IDS))
            or self.setting_id != M03R_V13_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.completed_updates != _expected_updates(self.fold_index)
            or self.selected_horizon_sessions
            != M03R_V13_SELECTED_HORIZON_SESSIONS
            or self.head_identity.setting_id != self.setting_id
            or self.head_identity.selected_alpha_horizon
            != self.selected_horizon_sessions
            or self.protocol_sha256 != M03R_V13_PROTOCOL_SHA256
        ):
            raise M03RV13CheckpointError("v13 loaded checkpoint drifted")
        for name, value in (
            ("model_state_sha256", self.model_state_sha256),
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("episode_schedule_sha256", self.episode_schedule_sha256),
            (
                "target_residual_operator_root_sha256",
                self.target_residual_operator_root_sha256,
            ),
            (
                "action_residual_operator_root_sha256",
                self.action_residual_operator_root_sha256,
            ),
            ("source_array_sha256", self.source_array_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
        ):
            _digest(name, value)


def write_immutable_m03r_v13_alpha_checkpoint(
    path: str | Path,
    policy: Top2000M03RV13PredictivePolicy,
    *,
    fold_index: int,
    completed_updates: int,
    episode_schedule_sha256: str,
    target_residual_operator_root_sha256: str,
    action_residual_operator_root_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
) -> str:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV13CheckpointError("v13 checkpoint target already exists")
    if (
        fold_index not in range(6)
        or completed_updates != _expected_updates(fold_index)
        or policy.selected_horizon_sessions
        != M03R_V13_SELECTED_HORIZON_SESSIONS
    ):
        raise M03RV13CheckpointError("v13 checkpoint cursor drifted")
    for name, value in (
        ("episode_schedule_sha256", episode_schedule_sha256),
        (
            "target_residual_operator_root_sha256",
            target_residual_operator_root_sha256,
        ),
        (
            "action_residual_operator_root_sha256",
            action_residual_operator_root_sha256,
        ),
        ("source_array_sha256", source_array_sha256),
        ("asset_axis_sha256", asset_axis_sha256),
    ):
        _digest(name, value)
    state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    identity = policy.v13_head_identity()
    payload = {
        "schema": M03R_V13_ALPHA_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V13_PROTOCOL_SHA256,
        "setting_index": policy.v13_setting.setting_index,
        "setting_id": policy.v13_setting.setting_id,
        "fold_index": fold_index,
        "completed_updates": completed_updates,
        "selected_horizon_sessions": M03R_V13_SELECTED_HORIZON_SESSIONS,
        "episode_schedule_sha256": episode_schedule_sha256,
        "target_residual_operator_root_sha256": (
            target_residual_operator_root_sha256
        ),
        "action_residual_operator_root_sha256": (
            action_residual_operator_root_sha256
        ),
        "source_array_sha256": source_array_sha256,
        "asset_axis_sha256": asset_axis_sha256,
        "head_identity": asdict(identity),
        "model_state_dict": state,
        "model_state_sha256": state_dict_sha256(state),
        "evaluation_only_load": True,
        "v12_state_reused": False,
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


def load_m03r_v13_alpha_checkpoint_for_evaluation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_setting_index: int,
    expected_fold_index: int,
    expected_completed_updates: int,
    expected_episode_schedule_sha256: str,
    expected_target_residual_operator_root_sha256: str,
    expected_action_residual_operator_root_sha256: str,
    expected_source_array_sha256: str,
    expected_asset_axis_sha256: str,
    policy: Top2000M03RV13PredictivePolicy,
) -> M03RV13LoadedCheckpoint:
    source = Path(path)
    if (
        policy.selected_horizon_sessions
        != M03R_V13_SELECTED_HORIZON_SESSIONS
        or expected_completed_updates != _expected_updates(expected_fold_index)
    ):
        raise M03RV13CheckpointError("v13 checkpoint horizon or update drifted")
    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV13CheckpointError(
            "v13 checkpoint must be a readable regular non-symlink"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_CHECKPOINT_BYTES
        ):
            raise M03RV13CheckpointError("v13 checkpoint size or type is invalid")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV13CheckpointError("v13 checkpoint file hash drifted")
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
            raise M03RV13CheckpointError("v13 checkpoint changed while read")
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise M03RV13CheckpointError("v13 checkpoint payload is not a mapping")
    if expected_setting_index not in range(len(M03R_V13_SETTING_IDS)):
        raise M03RV13CheckpointError("v13 expected setting index drifted")
    expected = {
        "schema": M03R_V13_ALPHA_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V13_PROTOCOL_SHA256,
        "setting_index": expected_setting_index,
        "setting_id": M03R_V13_SETTING_IDS[expected_setting_index],
        "fold_index": expected_fold_index,
        "completed_updates": expected_completed_updates,
        "selected_horizon_sessions": M03R_V13_SELECTED_HORIZON_SESSIONS,
        "episode_schedule_sha256": _digest(
            "expected_episode_schedule_sha256", expected_episode_schedule_sha256
        ),
        "target_residual_operator_root_sha256": _digest(
            "expected_target_residual_operator_root_sha256",
            expected_target_residual_operator_root_sha256,
        ),
        "action_residual_operator_root_sha256": _digest(
            "expected_action_residual_operator_root_sha256",
            expected_action_residual_operator_root_sha256,
        ),
        "source_array_sha256": _digest(
            "expected_source_array_sha256", expected_source_array_sha256
        ),
        "asset_axis_sha256": _digest(
            "expected_asset_axis_sha256", expected_asset_axis_sha256
        ),
        "evaluation_only_load": True,
        "v12_state_reused": False,
        "outer_2026_accessed": False,
        "economic_optimizer_updates": 0,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise M03RV13CheckpointError("v13 checkpoint immutable identity drifted")
    state = payload.get("model_state_dict")
    head_row = payload.get("head_identity")
    if (
        not isinstance(state, Mapping)
        or not isinstance(head_row, Mapping)
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
        or payload.get("model_state_sha256") != state_dict_sha256(state)
    ):
        raise M03RV13CheckpointError("v13 checkpoint model state drifted")
    try:
        policy.load_state_dict(dict(state), strict=True)
        stored_identity = M03RV13HeadIdentity(**dict(head_row))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M03RV13CheckpointError("v13 checkpoint strict load failed") from exc
    observed_identity = policy.v13_head_identity()
    if (
        state_dict_sha256(policy.state_dict()) != payload["model_state_sha256"]
        or observed_identity != stored_identity
    ):
        raise M03RV13CheckpointError("loaded v13 semantic state drifted")
    result = M03RV13LoadedCheckpoint(
        setting_index=expected_setting_index,
        setting_id=M03R_V13_SETTING_IDS[expected_setting_index],
        fold_index=expected_fold_index,
        completed_updates=expected_completed_updates,
        selected_horizon_sessions=M03R_V13_SELECTED_HORIZON_SESSIONS,
        model_state_sha256=payload["model_state_sha256"],
        checkpoint_file_sha256=expected_file_sha256,
        episode_schedule_sha256=expected_episode_schedule_sha256,
        target_residual_operator_root_sha256=(
            expected_target_residual_operator_root_sha256
        ),
        action_residual_operator_root_sha256=(
            expected_action_residual_operator_root_sha256
        ),
        source_array_sha256=expected_source_array_sha256,
        asset_axis_sha256=expected_asset_axis_sha256,
        head_identity=observed_identity,
    )
    result.validate()
    return result


def write_reload_evaluate_m03r_v13_checkpoint(
    path: str | Path,
    policy: Top2000M03RV13PredictivePolicy,
    fresh_policy_factory: Callable[[], Top2000M03RV13PredictivePolicy],
    evaluator: Callable[
        [Top2000M03RV13PredictivePolicy, M03RV13LoadedCheckpoint], EvaluationResult
    ],
    *,
    fold_index: int,
    completed_updates: int,
    episode_schedule_sha256: str,
    target_residual_operator_root_sha256: str,
    action_residual_operator_root_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
) -> tuple[M03RV13LoadedCheckpoint, EvaluationResult]:
    file_sha256 = write_immutable_m03r_v13_alpha_checkpoint(
        path,
        policy,
        fold_index=fold_index,
        completed_updates=completed_updates,
        episode_schedule_sha256=episode_schedule_sha256,
        target_residual_operator_root_sha256=(
            target_residual_operator_root_sha256
        ),
        action_residual_operator_root_sha256=(
            action_residual_operator_root_sha256
        ),
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
    )
    del policy
    loaded_policy = fresh_policy_factory()
    loaded = load_m03r_v13_alpha_checkpoint_for_evaluation(
        path,
        expected_file_sha256=file_sha256,
        expected_setting_index=loaded_policy.v13_setting.setting_index,
        expected_fold_index=fold_index,
        expected_completed_updates=completed_updates,
        expected_episode_schedule_sha256=episode_schedule_sha256,
        expected_target_residual_operator_root_sha256=(
            target_residual_operator_root_sha256
        ),
        expected_action_residual_operator_root_sha256=(
            action_residual_operator_root_sha256
        ),
        expected_source_array_sha256=source_array_sha256,
        expected_asset_axis_sha256=asset_axis_sha256,
        policy=loaded_policy,
    )
    return loaded, evaluator(loaded_policy, loaded)


__all__ = [
    "M03R_V13_ALPHA_CHECKPOINT_SCHEMA",
    "M03RV13CheckpointError",
    "M03RV13LoadedCheckpoint",
    "load_m03r_v13_alpha_checkpoint_for_evaluation",
    "write_immutable_m03r_v13_alpha_checkpoint",
    "write_reload_evaluate_m03r_v13_checkpoint",
]
