"""No-clobber epoch checkpoint round trip for M03R-v16."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar, cast

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256
from rl_quant.training.top2000_m03r_v16_fold import render_m03r_v16_fold_geometries
from rl_quant.training.top2000_m03r_v16_policy import (
    M03RV16HeadIdentity,
    Top2000M03RV16PredictivePolicy,
    m03r_v16_score_component_state_sha256,
)

M03R_V16_EPOCH_CHECKPOINT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-score-epoch-checkpoint-v4"
)
_MAX_CHECKPOINT_BYTES = 8 * 1024**3
_LOADED_EPOCH_CHECKPOINT_ISSUER = object()
EvaluationResult = TypeVar("EvaluationResult")


class M03RV16CheckpointError(ValueError):
    """The V16 epoch checkpoint or strict reload drifted."""


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV16CheckpointError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _expected_updates(fold_index: int, epoch_index: int) -> int:
    if (
        isinstance(fold_index, bool)
        or fold_index not in range(M03R_V16_PREDICTIVE_SPEC.chronological_fold_count)
        or isinstance(epoch_index, bool)
        or epoch_index not in range(M03R_V16_PREDICTIVE_SPEC.score_training_epochs)
    ):
        raise M03RV16CheckpointError("V16 checkpoint cursor drifted")
    return render_m03r_v16_fold_geometries(1001)[fold_index].training_block_count * (
        epoch_index + 1
    )


@dataclass(frozen=True, slots=True)
class M03RV16LoadedEpochCheckpoint:
    setting_index: int
    setting_id: str
    fold_index: int
    epoch_index: int
    completed_score_updates: int
    model_state_sha256: str
    score_component_state_sha256: str
    checkpoint_file_sha256: str
    panel_schedule_sha256: str
    selection_target_operator_root_sha256: str
    action_operator_root_sha256: str
    source_array_sha256: str
    asset_axis_sha256: str
    head_identity: M03RV16HeadIdentity
    _issuer: object = field(repr=False)
    hold_target_sessions: int = LEGACY_HOLD30_TARGET_SPEC.target_sessions
    hold_target_spec_sha256: str = LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256

    def validate(self) -> None:
        self.head_identity.validate()
        if (
            self._issuer is not _LOADED_EPOCH_CHECKPOINT_ISSUER
            or self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.setting_id != M03R_V16_SETTINGS[self.setting_index].setting_id
            or self.fold_index
            not in range(M03R_V16_PREDICTIVE_SPEC.chronological_fold_count)
            or self.completed_score_updates
            != _expected_updates(self.fold_index, self.epoch_index)
            or self.head_identity.setting_id != self.setting_id
            or self.hold_target_sessions != LEGACY_HOLD30_TARGET_SPEC.target_sessions
            or self.hold_target_spec_sha256 != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
        ):
            raise M03RV16CheckpointError("V16 loaded checkpoint drifted")
        for name, value in (
            ("model_state_sha256", self.model_state_sha256),
            ("score_component_state_sha256", self.score_component_state_sha256),
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("panel_schedule_sha256", self.panel_schedule_sha256),
            (
                "selection_target_operator_root_sha256",
                self.selection_target_operator_root_sha256,
            ),
            ("action_operator_root_sha256", self.action_operator_root_sha256),
            ("source_array_sha256", self.source_array_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
        ):
            _digest(name, value)


def write_immutable_m03r_v16_epoch_checkpoint(
    path: str | Path,
    policy: Top2000M03RV16PredictivePolicy,
    *,
    fold_index: int,
    epoch_index: int,
    completed_score_updates: int,
    panel_schedule_sha256: str,
    selection_target_operator_root_sha256: str,
    action_operator_root_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
) -> str:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV16CheckpointError("V16 checkpoint target already exists")
    if completed_score_updates != _expected_updates(fold_index, epoch_index):
        raise M03RV16CheckpointError("V16 checkpoint epoch/update cursor drifted")
    identities = {
        "panel_schedule_sha256": panel_schedule_sha256,
        "selection_target_operator_root_sha256": (
            selection_target_operator_root_sha256
        ),
        "action_operator_root_sha256": action_operator_root_sha256,
        "source_array_sha256": source_array_sha256,
        "asset_axis_sha256": asset_axis_sha256,
    }
    for name, value in identities.items():
        _digest(name, value)
    state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    payload = {
        "schema": M03R_V16_EPOCH_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "hold_target_sessions": LEGACY_HOLD30_TARGET_SPEC.target_sessions,
        "hold_target_spec_sha256": LEGACY_HOLD30_TARGET_SPEC.receipt_sha256,
        "setting_index": policy.v16_setting.setting_index,
        "setting_id": policy.v16_setting.setting_id,
        "fold_index": fold_index,
        "epoch_index": epoch_index,
        "completed_score_updates": completed_score_updates,
        **identities,
        "head_identity": asdict(policy.v16_head_identity()),
        "model_state_dict": state,
        "model_state_sha256": state_dict_sha256(state),
        "score_component_state_sha256": (m03r_v16_score_component_state_sha256(policy)),
        "score_stage_only": True,
        "timing_optimizer_updates": 0,
        "uncertainty_calibration_updates": 0,
        "v15_state_reused": False,
        "qualification_tail_accessed": False,
        "outer_2026_accessed": False,
        "economic_optimizer_updates": 0,
        "reinforcement_learning_updates": 0,
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


def load_m03r_v16_epoch_checkpoint_for_evaluation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_setting_index: int,
    expected_fold_index: int,
    expected_epoch_index: int,
    expected_completed_score_updates: int,
    expected_panel_schedule_sha256: str,
    expected_selection_target_operator_root_sha256: str,
    expected_action_operator_root_sha256: str,
    expected_source_array_sha256: str,
    expected_asset_axis_sha256: str,
    policy: Top2000M03RV16PredictivePolicy,
) -> M03RV16LoadedEpochCheckpoint:
    if (
        policy.v16_setting.setting_index != expected_setting_index
        or expected_completed_score_updates
        != _expected_updates(expected_fold_index, expected_epoch_index)
    ):
        raise M03RV16CheckpointError("V16 expected checkpoint identity drifted")
    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16CheckpointError(
            "V16 checkpoint must be a readable regular non-symlink"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_CHECKPOINT_BYTES
        ):
            raise M03RV16CheckpointError("V16 checkpoint type or size is invalid")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV16CheckpointError("V16 checkpoint file hash drifted")
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
            raise M03RV16CheckpointError("V16 checkpoint changed while read")
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise M03RV16CheckpointError("V16 checkpoint payload is not a mapping")
    if expected_setting_index not in range(len(M03R_V16_SETTINGS)):
        raise M03RV16CheckpointError("V16 expected setting index drifted")
    expected = {
        "schema": M03R_V16_EPOCH_CHECKPOINT_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "hold_target_sessions": LEGACY_HOLD30_TARGET_SPEC.target_sessions,
        "hold_target_spec_sha256": LEGACY_HOLD30_TARGET_SPEC.receipt_sha256,
        "setting_index": expected_setting_index,
        "setting_id": M03R_V16_SETTINGS[expected_setting_index].setting_id,
        "fold_index": expected_fold_index,
        "epoch_index": expected_epoch_index,
        "completed_score_updates": expected_completed_score_updates,
        "panel_schedule_sha256": _digest(
            "expected_panel_schedule_sha256", expected_panel_schedule_sha256
        ),
        "selection_target_operator_root_sha256": _digest(
            "expected_selection_target_operator_root_sha256",
            expected_selection_target_operator_root_sha256,
        ),
        "action_operator_root_sha256": _digest(
            "expected_action_operator_root_sha256", expected_action_operator_root_sha256
        ),
        "source_array_sha256": _digest(
            "expected_source_array_sha256", expected_source_array_sha256
        ),
        "asset_axis_sha256": _digest(
            "expected_asset_axis_sha256", expected_asset_axis_sha256
        ),
        "score_stage_only": True,
        "timing_optimizer_updates": 0,
        "uncertainty_calibration_updates": 0,
        "v15_state_reused": False,
        "qualification_tail_accessed": False,
        "outer_2026_accessed": False,
        "economic_optimizer_updates": 0,
        "reinforcement_learning_updates": 0,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise M03RV16CheckpointError("V16 checkpoint immutable identity drifted")
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
        raise M03RV16CheckpointError("V16 checkpoint model state drifted")
    try:
        policy.load_state_dict(dict(state), strict=True)
        stored_identity = M03RV16HeadIdentity(**dict(head_row))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M03RV16CheckpointError("V16 checkpoint strict load failed") from exc
    observed_identity = policy.v16_head_identity()
    if (
        state_dict_sha256(policy.state_dict()) != payload["model_state_sha256"]
        or observed_identity != stored_identity
        or m03r_v16_score_component_state_sha256(policy)
        != payload.get("score_component_state_sha256")
    ):
        raise M03RV16CheckpointError("loaded V16 semantic state drifted")
    result = M03RV16LoadedEpochCheckpoint(
        setting_index=expected_setting_index,
        setting_id=M03R_V16_SETTINGS[expected_setting_index].setting_id,
        fold_index=expected_fold_index,
        epoch_index=expected_epoch_index,
        completed_score_updates=expected_completed_score_updates,
        model_state_sha256=str(payload["model_state_sha256"]),
        score_component_state_sha256=str(payload["score_component_state_sha256"]),
        checkpoint_file_sha256=expected_file_sha256,
        panel_schedule_sha256=cast(str, expected["panel_schedule_sha256"]),
        selection_target_operator_root_sha256=cast(
            str, expected["selection_target_operator_root_sha256"]
        ),
        action_operator_root_sha256=cast(str, expected["action_operator_root_sha256"]),
        source_array_sha256=cast(str, expected["source_array_sha256"]),
        asset_axis_sha256=cast(str, expected["asset_axis_sha256"]),
        head_identity=stored_identity,
        _issuer=_LOADED_EPOCH_CHECKPOINT_ISSUER,
    )
    result.validate()
    return result


def write_reload_evaluate_m03r_v16_epoch_checkpoint(
    path: str | Path,
    policy: Top2000M03RV16PredictivePolicy,
    policy_factory: Callable[[], Top2000M03RV16PredictivePolicy],
    evaluator: Callable[
        [Top2000M03RV16PredictivePolicy, M03RV16LoadedEpochCheckpoint], EvaluationResult
    ],
    **identity: object,
) -> tuple[M03RV16LoadedEpochCheckpoint, EvaluationResult]:
    """Destroy the training reference and evaluate only exact reloaded bytes."""

    file_sha256 = write_immutable_m03r_v16_epoch_checkpoint(
        path,
        policy,
        **identity,  # type: ignore[arg-type]
    )
    expected_setting_index = policy.v16_setting.setting_index
    del policy
    reloaded_policy = policy_factory()
    loaded = load_m03r_v16_epoch_checkpoint_for_evaluation(
        path,
        expected_file_sha256=file_sha256,
        expected_setting_index=expected_setting_index,
        expected_fold_index=cast(int, identity["fold_index"]),
        expected_epoch_index=cast(int, identity["epoch_index"]),
        expected_completed_score_updates=cast(int, identity["completed_score_updates"]),
        expected_panel_schedule_sha256=str(identity["panel_schedule_sha256"]),
        expected_selection_target_operator_root_sha256=str(
            identity["selection_target_operator_root_sha256"]
        ),
        expected_action_operator_root_sha256=str(
            identity["action_operator_root_sha256"]
        ),
        expected_source_array_sha256=str(identity["source_array_sha256"]),
        expected_asset_axis_sha256=str(identity["asset_axis_sha256"]),
        policy=reloaded_policy,
    )
    return loaded, evaluator(reloaded_policy, loaded)


__all__ = [
    "M03R_V16_EPOCH_CHECKPOINT_SCHEMA",
    "M03RV16CheckpointError",
    "M03RV16LoadedEpochCheckpoint",
    "load_m03r_v16_epoch_checkpoint_for_evaluation",
    "write_immutable_m03r_v16_epoch_checkpoint",
    "write_reload_evaluate_m03r_v16_epoch_checkpoint",
]
