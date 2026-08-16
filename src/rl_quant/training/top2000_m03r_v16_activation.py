"""Immutable phase activations for the two-stage M03R-v16 workflow."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, semantic_sha256
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import M03R_V16_PROTOCOL_SHA256
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
)

M03R_V16_TRAINING_ACTIVATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-activation-v1"
)
M03R_V16_QUALIFICATION_ACTIVATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-qualification-activation-v1"
)
_TRAINING_ISSUER = object()
_QUALIFICATION_ISSUER = object()
_MAX_BYTES = 4 * 1024**2


class M03RV16ActivationError(ValueError):
    """A V16 phase authority is absent, malformed, or mismatched."""


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise M03RV16ActivationError(f"{name} must be a lowercase SHA-256")


def _read_exact(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16ActivationError("V16 activation file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_BYTES:
            raise M03RV16ActivationError("V16 activation file type or size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise M03RV16ActivationError("V16 activation changed while read")
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise M03RV16ActivationError("V16 activation file hash drifted")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16ActivationError("V16 activation file is malformed") from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise M03RV16ActivationError("V16 activation file is not canonical")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> str:
    data = canonical_json_file_bytes(payload)
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV16TrainingActivation:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    static_gate_receipt_sha256: str
    capacity_gate_receipt_sha256: str
    source_tree_root_sha256: str
    image_digest_sha256: str
    _issuer: object = field(repr=False)
    predictive_training_authorized: bool = True
    outer_qualification_authorized: bool = False
    economic_training_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    outer_2026_access_authorized: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_TRAINING_ACTIVATION_SCHEMA

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
    ) -> None:
        package.validate()
        authorization.validate(package)
        for name in (
            "static_gate_receipt_sha256",
            "capacity_gate_receipt_sha256",
            "source_tree_root_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self._issuer is not _TRAINING_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256 != authorization.receipt_sha256
            or self.image_digest_sha256 != package.artifacts.image_digest_sha256
            or not self.predictive_training_authorized
            or self.outer_qualification_authorized
            or self.economic_training_authorized
            or self.reinforcement_learning_authorized
            or self.outer_2026_access_authorized
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_TRAINING_ACTIVATION_SCHEMA
        ):
            raise M03RV16ActivationError("V16 training activation drifted")

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)


@dataclass(frozen=True, slots=True)
class M03RV16QualificationActivation:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    training_panel_receipt_sha256: str
    training_terminal_file_sha256: tuple[str, str, str]
    primary_training_adequacy_receipt_sha256: tuple[str, ...]
    source_tree_root_sha256: str
    image_digest_sha256: str
    _issuer: object = field(repr=False)
    primary_training_adequacy: str = "adequate"
    predictive_training_authorized: bool = False
    outer_qualification_authorized: bool = True
    economic_training_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    outer_2026_access_authorized: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_QUALIFICATION_ACTIVATION_SCHEMA

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
    ) -> None:
        package.validate()
        authorization.validate(package)
        _digest("training_panel_receipt_sha256", self.training_panel_receipt_sha256)
        _digest("source_tree_root_sha256", self.source_tree_root_sha256)
        for value in self.training_terminal_file_sha256:
            _digest("training_terminal_file_sha256", value)
        for value in self.primary_training_adequacy_receipt_sha256:
            _digest("primary_training_adequacy_receipt_sha256", value)
        if (
            self._issuer is not _QUALIFICATION_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256 != authorization.receipt_sha256
            or self.image_digest_sha256 != package.artifacts.image_digest_sha256
            or len(self.training_terminal_file_sha256) != 3
            or len(self.primary_training_adequacy_receipt_sha256) != 5
            or self.primary_training_adequacy != "adequate"
            or self.predictive_training_authorized
            or not self.outer_qualification_authorized
            or self.economic_training_authorized
            or self.reinforcement_learning_authorized
            or self.outer_2026_access_authorized
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_QUALIFICATION_ACTIVATION_SCHEMA
        ):
            raise M03RV16ActivationError("V16 qualification activation drifted")

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)


def write_m03r_v16_training_activation(path: str | Path, value: M03RV16TrainingActivation) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return _write(Path(path), {"activation": payload, "receipt_sha256": value.receipt_sha256})


def load_m03r_v16_training_activation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
) -> M03RV16TrainingActivation:
    payload = _read_exact(Path(path), expected_file_sha256)
    try:
        value = M03RV16TrainingActivation(**payload["activation"], _issuer=_TRAINING_ISSUER)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError("V16 training activation is malformed") from exc
    if payload.get("receipt_sha256") != value.receipt_sha256:
        raise M03RV16ActivationError("V16 training activation receipt drifted")
    value.validate_for(package, authorization)
    return value


def write_m03r_v16_qualification_activation(
    path: str | Path, value: M03RV16QualificationActivation
) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return _write(Path(path), {"activation": payload, "receipt_sha256": value.receipt_sha256})


def load_m03r_v16_qualification_activation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
) -> M03RV16QualificationActivation:
    payload = _read_exact(Path(path), expected_file_sha256)
    try:
        row = dict(payload["activation"])
        row["training_terminal_file_sha256"] = tuple(row["training_terminal_file_sha256"])
        row["primary_training_adequacy_receipt_sha256"] = tuple(
            row["primary_training_adequacy_receipt_sha256"]
        )
        value = M03RV16QualificationActivation(**row, _issuer=_QUALIFICATION_ISSUER)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError("V16 qualification activation is malformed") from exc
    if payload.get("receipt_sha256") != value.receipt_sha256:
        raise M03RV16ActivationError("V16 qualification activation receipt drifted")
    value.validate_for(package, authorization)
    return value


def issue_m03r_v16_training_activation(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    static_gate_receipt_sha256: str,
    capacity_gate_receipt_sha256: str,
    source_tree_root_sha256: str,
) -> M03RV16TrainingActivation:
    value = M03RV16TrainingActivation(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        static_gate_receipt_sha256=static_gate_receipt_sha256,
        capacity_gate_receipt_sha256=capacity_gate_receipt_sha256,
        source_tree_root_sha256=source_tree_root_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        _issuer=_TRAINING_ISSUER,
    )
    value.validate_for(package, authorization)
    return value


def issue_m03r_v16_qualification_activation(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    training_panel_receipt_sha256: str,
    training_terminal_file_sha256: tuple[str, str, str],
    primary_training_adequacy_receipt_sha256: tuple[str, ...],
    source_tree_root_sha256: str,
) -> M03RV16QualificationActivation:
    value = M03RV16QualificationActivation(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        training_panel_receipt_sha256=training_panel_receipt_sha256,
        training_terminal_file_sha256=training_terminal_file_sha256,
        primary_training_adequacy_receipt_sha256=(
            primary_training_adequacy_receipt_sha256
        ),
        source_tree_root_sha256=source_tree_root_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        _issuer=_QUALIFICATION_ISSUER,
    )
    value.validate_for(package, authorization)
    return value


__all__ = [
    "M03R_V16_QUALIFICATION_ACTIVATION_SCHEMA",
    "M03R_V16_TRAINING_ACTIVATION_SCHEMA",
    "M03RV16ActivationError",
    "M03RV16QualificationActivation",
    "M03RV16TrainingActivation",
    "issue_m03r_v16_qualification_activation",
    "issue_m03r_v16_training_activation",
    "load_m03r_v16_qualification_activation",
    "load_m03r_v16_training_activation",
    "write_m03r_v16_qualification_activation",
    "write_m03r_v16_training_activation",
]
