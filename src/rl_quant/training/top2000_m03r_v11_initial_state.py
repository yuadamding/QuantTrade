"""Package-owned common initial parameter state for M03R-v11."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    state_dict_sha256,
)

M03R_V11_INITIAL_STATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-common-initial-parameter-state-v1"
)
_MAX_INITIAL_STATE_BYTES = 4 * 1024**3


class M03RV11InitialStateError(ValueError):
    """The packaged common initialization is unavailable or has drifted."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_m03r_v11_initial_parameter_state(
    path: str | Path,
    policy: nn.Module,
) -> tuple[str, str]:
    """Publish one immutable CPU state and return semantic/file SHA-256 values."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV11InitialStateError("v11 initial-state target already exists")
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in policy.state_dict().items()
    }
    semantic_sha256 = state_dict_sha256(state)
    payload = {
        "schema": M03R_V11_INITIAL_STATE_SCHEMA,
        "model_state_sha256": semantic_sha256,
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
    return semantic_sha256, _file_sha256(target)


def load_m03r_v11_initial_parameter_state(
    path: str | Path,
    policy: nn.Module,
    *,
    expected_file_sha256: str,
    expected_state_sha256: str,
) -> None:
    """Load the exact package-owned state strictly into a fresh policy."""

    if len(expected_file_sha256) != 64 or len(expected_state_sha256) != 64:
        raise M03RV11InitialStateError("v11 initial-state digest is malformed")
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV11InitialStateError(
            "v11 initial-state artifact is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_INITIAL_STATE_BYTES
        ):
            raise M03RV11InitialStateError(
                "v11 initial-state artifact type or size is invalid"
            )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV11InitialStateError("v11 initial-state file hash drifted")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV11InitialStateError(
                "v11 initial-state artifact changed while read"
            )
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise M03RV11InitialStateError("v11 initial-state payload is malformed")
    state = payload.get("state_dict")
    if (
        payload.get("schema") != M03R_V11_INITIAL_STATE_SCHEMA
        or payload.get("model_state_sha256") != expected_state_sha256
        or not isinstance(state, Mapping)
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
        or state_dict_sha256(state) != expected_state_sha256
    ):
        raise M03RV11InitialStateError("v11 initial-state semantics drifted")
    try:
        policy.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M03RV11InitialStateError(
            "v11 initial-state does not strictly match the policy"
        ) from exc
    if model_state_sha256(policy) != expected_state_sha256:
        raise M03RV11InitialStateError("v11 loaded initial parameter state drifted")


__all__ = [
    "M03R_V11_INITIAL_STATE_SCHEMA",
    "M03RV11InitialStateError",
    "load_m03r_v11_initial_parameter_state",
    "write_m03r_v11_initial_parameter_state",
]
