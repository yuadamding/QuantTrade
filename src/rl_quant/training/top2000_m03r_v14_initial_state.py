"""Immutable common initial parameter bytes shared by both M03R-v14 settings."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path

import torch

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    state_dict_sha256,
)
from rl_quant.training.top2000_m03r_v14_policy import (
    M03R_V14_ALPHA_OUTPUT_CONTRACT_SHA256,
    Top2000M03RV14PredictivePolicy,
)

M03R_V14_INITIAL_STATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-common-initial-parameter-state-v1"
)
_MAX_INITIAL_STATE_BYTES = 4 * 1024**3


class M03RV14InitialStateError(ValueError):
    """The common v14 initialization is unavailable or has drifted."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _architecture_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return hashlib.sha256(
        json.dumps(
            tuple(
                (name, str(value.dtype), tuple(value.shape))
                for name, value in sorted(state.items())
            ),
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV14InitialStateError(f"{name} must be a lowercase SHA-256")
    return value


def write_m03r_v14_initial_parameter_state(
    path: str | Path,
    policy: Top2000M03RV14PredictivePolicy,
) -> tuple[str, str, str]:
    """Publish one canonical CPU state; return semantic, file, architecture SHAs."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV14InitialStateError("v14 initial-state target already exists")
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in policy.state_dict().items()
    }
    semantic_sha256 = state_dict_sha256(state)
    architecture_sha256 = _architecture_sha256(state)
    payload = {
        "schema": M03R_V14_INITIAL_STATE_SCHEMA,
        "protocol_sha256": M03R_V14_PROTOCOL_SHA256,
        "output_contract_sha256": M03R_V14_ALPHA_OUTPUT_CONTRACT_SHA256,
        "model_state_sha256": semantic_sha256,
        "architecture_sha256": architecture_sha256,
        "state_dict": state,
        "common_across_settings": True,
        "v13_state_reused": False,
        "outer_2026_accessed": False,
    }
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return semantic_sha256, _file_sha256(target), architecture_sha256


def load_m03r_v14_initial_parameter_state(
    path: str | Path,
    policy: Top2000M03RV14PredictivePolicy,
    *,
    expected_file_sha256: str,
    expected_state_sha256: str,
    expected_architecture_sha256: str,
) -> None:
    """Strict-load exact common bytes into either structurally identical setting."""

    expected_file_sha256 = _digest("expected_file_sha256", expected_file_sha256)
    expected_state_sha256 = _digest("expected_state_sha256", expected_state_sha256)
    expected_architecture_sha256 = _digest(
        "expected_architecture_sha256", expected_architecture_sha256
    )
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV14InitialStateError(
            "v14 initial-state artifact is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_INITIAL_STATE_BYTES
        ):
            raise M03RV14InitialStateError(
                "v14 initial-state artifact type or size is invalid"
            )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV14InitialStateError("v14 initial-state file hash drifted")
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
            raise M03RV14InitialStateError(
                "v14 initial-state artifact changed while read"
            )
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise M03RV14InitialStateError("v14 initial-state payload is malformed")
    state = payload.get("state_dict")
    if (
        payload.get("schema") != M03R_V14_INITIAL_STATE_SCHEMA
        or payload.get("protocol_sha256") != M03R_V14_PROTOCOL_SHA256
        or payload.get("output_contract_sha256")
        != M03R_V14_ALPHA_OUTPUT_CONTRACT_SHA256
        or payload.get("model_state_sha256") != expected_state_sha256
        or payload.get("architecture_sha256") != expected_architecture_sha256
        or payload.get("common_across_settings") is not True
        or payload.get("v13_state_reused") is not False
        or payload.get("outer_2026_accessed") is not False
        or not isinstance(state, Mapping)
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
        or state_dict_sha256(state) != expected_state_sha256
        or _architecture_sha256(state) != expected_architecture_sha256
    ):
        raise M03RV14InitialStateError("v14 initial-state semantics drifted")
    observed_architecture = _architecture_sha256(policy.state_dict())
    if observed_architecture != expected_architecture_sha256:
        raise M03RV14InitialStateError("v14 policy architecture drifted")
    try:
        policy.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M03RV14InitialStateError(
            "v14 initial-state does not strictly match the policy"
        ) from exc
    if model_state_sha256(policy) != expected_state_sha256:
        raise M03RV14InitialStateError("v14 loaded initial parameter state drifted")


__all__ = [
    "M03R_V14_INITIAL_STATE_SCHEMA",
    "M03RV14InitialStateError",
    "load_m03r_v14_initial_parameter_state",
    "write_m03r_v14_initial_parameter_state",
]
