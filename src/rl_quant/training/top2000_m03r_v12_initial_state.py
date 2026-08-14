"""Immutable common initial parameter bytes for M03R-v12."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path

import torch

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    state_dict_sha256,
)
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)

M03R_V12_INITIAL_STATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-common-initial-parameter-state-v2"
)
_MAX_INITIAL_STATE_BYTES = 4 * 1024**3


class M03RV12InitialStateError(ValueError):
    """The v12 common initial-state artifact drifted."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_m03r_v12_initial_parameter_state(
    path: str | Path,
    policy: Top2000M03RV12PredictivePolicy,
) -> tuple[str, str]:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV12InitialStateError("v12 initial-state target already exists")
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in policy.state_dict().items()
    }
    semantic = state_dict_sha256(state)
    payload = {
        "schema": M03R_V12_INITIAL_STATE_SCHEMA,
        "protocol_sha256": M03R_V12_PROTOCOL_SHA256,
        "setting_neutral": True,
        "selected_horizon_sessions": policy.selected_horizon_sessions,
        "model_state_sha256": semantic,
        "state_dict": state,
    }
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return semantic, _file_sha256(target)


def load_m03r_v12_initial_parameter_state(
    path: str | Path,
    policy: Top2000M03RV12PredictivePolicy,
    *,
    expected_file_sha256: str,
    expected_state_sha256: str,
) -> None:
    if len(expected_file_sha256) != 64 or len(expected_state_sha256) != 64:
        raise M03RV12InitialStateError("v12 initial-state digest is malformed")
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV12InitialStateError(
            "v12 initial-state artifact is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_INITIAL_STATE_BYTES
        ):
            raise M03RV12InitialStateError(
                "v12 initial-state artifact type or size is invalid"
            )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV12InitialStateError("v12 initial-state file hash drifted")
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
            raise M03RV12InitialStateError(
                "v12 initial-state artifact changed while read"
            )
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise M03RV12InitialStateError("v12 initial-state payload is malformed")
    state = payload.get("state_dict")
    if (
        payload.get("schema") != M03R_V12_INITIAL_STATE_SCHEMA
        or payload.get("protocol_sha256") != M03R_V12_PROTOCOL_SHA256
        or payload.get("setting_neutral") is not True
        or payload.get("selected_horizon_sessions") != policy.selected_horizon_sessions
        or payload.get("model_state_sha256") != expected_state_sha256
        or not isinstance(state, Mapping)
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
        or state_dict_sha256(state) != expected_state_sha256
    ):
        raise M03RV12InitialStateError("v12 initial-state semantics drifted")
    try:
        policy.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M03RV12InitialStateError(
            "v12 initial state does not strictly match the policy"
        ) from exc
    if model_state_sha256(policy) != expected_state_sha256:
        raise M03RV12InitialStateError("loaded v12 initial parameter state drifted")


__all__ = [
    "M03R_V12_INITIAL_STATE_SCHEMA",
    "M03RV12InitialStateError",
    "load_m03r_v12_initial_parameter_state",
    "write_m03r_v12_initial_parameter_state",
]
