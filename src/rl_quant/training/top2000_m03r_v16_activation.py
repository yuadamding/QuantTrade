"""Immutable phase activations for the two-stage M03R-v16 workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import M03R_V16_PROTOCOL_SHA256
from rl_quant.training.top2000_m03r_v16_capacity import (
    M03R_V16_CAPACITY_TERMINAL_SCHEMA,
    M03RV16CapacityRankEvidence,
    M03RV16CapacityTerminal,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
)
from rl_quant.training.top2000_m03r_v16_static_contract import (
    M03R_V16_STATIC_RESULT_SCHEMA,
)

M03R_V16_TRAINING_ACTIVATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-activation-v3"
)
M03R_V16_QUALIFICATION_ACTIVATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-qualification-preflight-activation-v4"
)
M03R_V16_PHASE_LAUNCH_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-phase-launch-authority-v4"
)
M03R_V16_ADMITTED_JOB_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-prelaunch-job-authority-v2"
)
M03R_V16_POD_RUNTIME_ATTESTATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-pod-runtime-attestation-v4"
)
M03R_V16_QUALIFICATION_OUTER_ACCESS_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-qualification-outer-access-authority-v1"
)
M03R_V16_DRY_RUN_RESULT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-server-dry-run-result-v1"
)
M03R_V16_ADMITTED_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-admitted-manifest-result-v1"
)
M03R_V16_TRAINING_PANEL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-adequacy-panel-v4"
)
M03R_V16_PREQUALIFICATION_CLOSURE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-prequalification-closure-v4"
)
M03R_V16_TRAINING_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-terminal-v2"
)
_TRAINING_ISSUER = object()
_QUALIFICATION_ISSUER = object()
_LAUNCH_ISSUER = object()
_ADMISSION_ISSUER = object()
_POD_ATTESTATION_ISSUER = object()
_TRAINING_PANEL_ISSUER = object()
_QUALIFICATION_OUTER_ACCESS_ISSUER = object()
_MAX_BYTES = 4 * 1024**2


class M03RV16ActivationError(ValueError):
    """A V16 phase authority is absent, malformed, or mismatched."""


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise M03RV16ActivationError(f"{name} must be a lowercase SHA-256")


def _safe_path_component(name: str, value: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
    if not value or value in {".", ".."} or any(char not in allowed for char in value):
        raise M03RV16ActivationError(f"{name} is not a safe path component")


def _read_canonical(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
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
    observed_file_sha256 = hashlib.sha256(raw).hexdigest()
    if len(raw) != before.st_size:
        raise M03RV16ActivationError("V16 activation file size drifted")
    if (
        expected_file_sha256 is not None
        and observed_file_sha256 != expected_file_sha256
    ):
        raise M03RV16ActivationError("V16 activation file hash drifted")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16ActivationError("V16 activation file is malformed") from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise M03RV16ActivationError("V16 activation file is not canonical")
    return payload, observed_file_sha256


def _read_exact(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    _digest("expected_file_sha256", expected_file_sha256)
    payload, _ = _read_canonical(
        path, expected_file_sha256=expected_file_sha256
    )
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
    static_rendered_manifest_sha256: str
    static_result_file_sha256: str
    static_result_receipt_sha256: str
    capacity_gate_receipt_sha256: str
    capacity_rendered_manifest_sha256: str
    capacity_terminal_file_sha256: str
    capacity_terminal_receipt_sha256: str
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
            "static_rendered_manifest_sha256",
            "static_result_file_sha256",
            "static_result_receipt_sha256",
            "capacity_gate_receipt_sha256",
            "capacity_rendered_manifest_sha256",
            "capacity_terminal_file_sha256",
            "capacity_terminal_receipt_sha256",
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
    training_panel_file_sha256: str
    training_terminal_file_sha256: tuple[str, str, str]
    setting_fold_training_adequacy_receipt_sha256: tuple[
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
    ]
    terminal_checkpoint_file_sha256: tuple[
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
    ]
    prequalification_closure_receipt_sha256: str
    prequalification_closure_file_sha256: str
    source_tree_root_sha256: str
    image_digest_sha256: str
    _issuer: object = field(repr=False)
    primary_training_adequacy: str = "adequate"
    qualification_input_preflight_authorized: bool = True
    predictive_training_authorized: bool = False
    outer_qualification_authorized: bool = False
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
        _digest("training_panel_file_sha256", self.training_panel_file_sha256)
        _digest(
            "prequalification_closure_receipt_sha256",
            self.prequalification_closure_receipt_sha256,
        )
        _digest(
            "prequalification_closure_file_sha256",
            self.prequalification_closure_file_sha256,
        )
        _digest("source_tree_root_sha256", self.source_tree_root_sha256)
        for value in self.training_terminal_file_sha256:
            _digest("training_terminal_file_sha256", value)
        for matrix in (
            self.setting_fold_training_adequacy_receipt_sha256,
            self.terminal_checkpoint_file_sha256,
        ):
            for row in matrix:
                for value in row:
                    _digest("qualification activation matrix digest", value)
        if (
            self._issuer is not _QUALIFICATION_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256 != authorization.receipt_sha256
            or self.image_digest_sha256 != package.artifacts.image_digest_sha256
            or len(self.training_terminal_file_sha256) != 3
            or len(self.setting_fold_training_adequacy_receipt_sha256) != 3
            or any(
                len(row) != 5
                for row in self.setting_fold_training_adequacy_receipt_sha256
            )
            or len(self.terminal_checkpoint_file_sha256) != 3
            or any(len(row) != 5 for row in self.terminal_checkpoint_file_sha256)
            or self.primary_training_adequacy != "adequate"
            or not self.qualification_input_preflight_authorized
            or self.predictive_training_authorized
            or self.outer_qualification_authorized
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


@dataclass(frozen=True, slots=True)
class M03RV16QualificationOuterAccessAuthority:
    """All-setting CPU input closure required before outer access."""

    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    qualification_activation_receipt_sha256: str
    training_panel_receipt_sha256: str
    setting_input_closure_file_sha256: tuple[str, str, str]
    setting_input_closure_receipt_sha256: tuple[str, str, str]
    setting_preflight_terminal_file_sha256: tuple[str, str, str]
    setting_preflight_terminal_receipt_sha256: tuple[str, str, str]
    qualification_risk_input_root_sha256: str
    source_tree_root_sha256: str
    _issuer: object = field(repr=False)
    setting_indices: tuple[int, int, int] = (0, 1, 2)
    outer_access_authorized: bool = True
    outer_qualification_access_started: bool = False
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_QUALIFICATION_OUTER_ACCESS_SCHEMA

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
        activation: M03RV16QualificationActivation,
    ) -> None:
        package.validate()
        authorization.validate(package)
        activation.validate_for(package, authorization)
        for name in (
            "qualification_activation_receipt_sha256",
            "training_panel_receipt_sha256",
            "qualification_risk_input_root_sha256",
            "source_tree_root_sha256",
        ):
            _digest(name, getattr(self, name))
        for value in (
            *self.setting_input_closure_file_sha256,
            *self.setting_input_closure_receipt_sha256,
            *self.setting_preflight_terminal_file_sha256,
            *self.setting_preflight_terminal_receipt_sha256,
        ):
            _digest("qualification closure digest", value)
        if (
            self._issuer is not _QUALIFICATION_OUTER_ACCESS_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256
            != authorization.receipt_sha256
            or self.qualification_activation_receipt_sha256
            != activation.receipt_sha256
            or self.training_panel_receipt_sha256
            != activation.training_panel_receipt_sha256
            or len(self.setting_input_closure_file_sha256) != 3
            or len(self.setting_input_closure_receipt_sha256) != 3
            or len(self.setting_preflight_terminal_file_sha256) != 3
            or len(self.setting_preflight_terminal_receipt_sha256) != 3
            or self.setting_indices != (0, 1, 2)
            or not self.outer_access_authorized
            or self.outer_qualification_access_started
            or self.outer_2026_accessed
            or self.source_tree_root_sha256
            != activation.source_tree_root_sha256
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_QUALIFICATION_OUTER_ACCESS_SCHEMA
        ):
            raise M03RV16ActivationError(
                "V16 qualification outer-access authority drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)


@dataclass(frozen=True, slots=True)
class M03RV16TrainingPanelAuthority:
    """Validated all-setting fit and checkpoint closure authority."""

    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    training_panel_receipt_sha256: str
    training_panel_file_sha256: str
    training_terminal_file_sha256: tuple[str, str, str]
    training_terminal_receipt_sha256: tuple[str, str, str]
    setting_fold_training_adequacy_receipt_sha256: tuple[
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
    ]
    setting_fold_training_adequacy_status: tuple[
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
    ]
    terminal_checkpoint_file_sha256: tuple[
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
    ]
    prequalification_closure_receipt_sha256: str
    prequalification_closure_file_sha256: str
    source_tree_root_sha256: str
    _issuer: object = field(repr=False)
    all_setting_folds_adequate: bool = True
    outer_qualification_outcomes_accessed: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = "rl-quant.top2000-dev.m03r-v16-training-panel-authority-v1"

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
    ) -> None:
        package.validate()
        authorization.validate(package)
        for name in (
            "training_panel_receipt_sha256",
            "training_panel_file_sha256",
            "prequalification_closure_receipt_sha256",
            "prequalification_closure_file_sha256",
            "source_tree_root_sha256",
        ):
            _digest(name, getattr(self, name))
        for value in self.training_terminal_file_sha256:
            _digest("training_terminal_file_sha256", value)
        for value in self.training_terminal_receipt_sha256:
            _digest("training_terminal_receipt_sha256", value)
        for matrix in (
            self.setting_fold_training_adequacy_receipt_sha256,
            self.terminal_checkpoint_file_sha256,
        ):
            for row in matrix:
                for value in row:
                    _digest("training panel matrix digest", value)
        if (
            self._issuer is not _TRAINING_PANEL_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256
            != authorization.receipt_sha256
            or len(self.training_terminal_file_sha256) != 3
            or len(self.training_terminal_receipt_sha256) != 3
            or len(self.setting_fold_training_adequacy_receipt_sha256) != 3
            or any(
                len(row) != 5
                for row in self.setting_fold_training_adequacy_receipt_sha256
            )
            or len(self.setting_fold_training_adequacy_status) != 3
            or any(
                len(row) != 5
                for row in self.setting_fold_training_adequacy_status
            )
            or any(
                status != "adequate"
                for row in self.setting_fold_training_adequacy_status
                for status in row
            )
            or len(self.terminal_checkpoint_file_sha256) != 3
            or any(len(row) != 5 for row in self.terminal_checkpoint_file_sha256)
            or not self.all_setting_folds_adequate
            or self.outer_qualification_outcomes_accessed
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
        ):
            raise M03RV16ActivationError("V16 training-panel authority drifted")

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)


@dataclass(frozen=True, slots=True)
class M03RV16PhaseLaunchAuthority:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    phase: Literal[
        "capacity", "training", "qualification-preflight", "qualification"
    ]
    prerequisite_authority_receipt_sha256: str
    admission_receipt_sha256: str
    admission_file_sha256: str
    job_contract_sha256: str
    pod_contract_sha256: str
    run_id: str
    job_uid: str
    completions: int
    one_shot_nonce_sha256: str
    pod_runtime_attestation_path_template: str
    source_tree_root_sha256: str
    image_digest_sha256: str
    storage_semantics_file_sha256: str
    storage_semantics_receipt_sha256: str
    storage_authority_root_sha256: str
    storage_observer_root_sha256: str
    _issuer: object = field(repr=False)
    economic_training_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    outer_2026_access_authorized: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_PHASE_LAUNCH_SCHEMA

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
        *,
        expected_phase: str,
        expected_prerequisite_receipt_sha256: str,
        expected_job_contract_sha256: str,
        expected_pod_contract_sha256: str,
        admission: M03RV16AdmittedJobAuthority,
        expected_admission_file_sha256: str,
    ) -> None:
        package.validate()
        authorization.validate(package)
        for name in (
            "prerequisite_authority_receipt_sha256",
            "admission_receipt_sha256",
            "admission_file_sha256",
            "job_contract_sha256",
            "pod_contract_sha256",
            "one_shot_nonce_sha256",
            "source_tree_root_sha256",
            "storage_semantics_file_sha256",
            "storage_semantics_receipt_sha256",
            "storage_authority_root_sha256",
            "storage_observer_root_sha256",
        ):
            _digest(name, getattr(self, name))
        expected_completions = {
            "capacity": 1,
            "training": 3,
            "qualification-preflight": 3,
            "qualification": 3,
        }
        _safe_path_component("job_uid", self.job_uid)
        expected_attestation_template = (
            f"pod-runtime/{self.phase}/{self.job_uid}/completion-{{completion_index:02d}}.json"
        )
        if (
            self._issuer is not _LAUNCH_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256
            != authorization.receipt_sha256
            or self.phase != expected_phase
            or self.prerequisite_authority_receipt_sha256
            != expected_prerequisite_receipt_sha256
            or self.admission_receipt_sha256 != admission.receipt_sha256
            or self.admission_file_sha256 != expected_admission_file_sha256
            or self.job_contract_sha256 != expected_job_contract_sha256
            or self.pod_contract_sha256 != expected_pod_contract_sha256
            or self.completions != expected_completions.get(self.phase)
            or not self.run_id
            or self.run_id != admission.run_id
            or self.job_uid != admission.job_uid
            or self.pod_runtime_attestation_path_template
            != expected_attestation_template
            or self.image_digest_sha256 != package.artifacts.image_digest_sha256
            or self.economic_training_authorized
            or self.reinforcement_learning_authorized
            or self.outer_2026_access_authorized
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_PHASE_LAUNCH_SCHEMA
        ):
            raise M03RV16ActivationError("V16 phase launch authority drifted")

    def pod_runtime_attestation_relative_path(self, completion_index: int) -> str:
        if not 0 <= completion_index < self.completions:
            raise M03RV16ActivationError(
                "V16 Pod attestation completion index is outside the launch"
            )
        return self.pod_runtime_attestation_path_template.format(
            completion_index=completion_index
        )

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)


@dataclass(frozen=True, slots=True)
class M03RV16AdmittedJobAuthority:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    phase: Literal[
        "capacity", "training", "qualification-preflight", "qualification"
    ]
    run_id: str
    job_name: str
    job_contract_sha256: str
    pod_contract_sha256: str
    server_side_dry_run_file_sha256: str
    server_side_dry_run_receipt_sha256: str
    admitted_manifest_file_sha256: str
    admitted_manifest_sha256: str
    job_uid: str
    completions: int
    image_reference: str
    image_digest_sha256: str
    _issuer: object = field(repr=False)
    suspended_at_admission: bool = True
    economic_training_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    outer_2026_access_authorized: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_ADMITTED_JOB_SCHEMA

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
        *,
        expected_phase: str,
        expected_job_contract_sha256: str,
        expected_pod_contract_sha256: str,
    ) -> None:
        package.validate()
        authorization.validate(package)
        expected = {
            "capacity": 1,
            "training": 3,
            "qualification-preflight": 3,
            "qualification": 3,
        }
        for name in (
            "job_contract_sha256",
            "pod_contract_sha256",
            "server_side_dry_run_file_sha256",
            "server_side_dry_run_receipt_sha256",
            "admitted_manifest_file_sha256",
            "admitted_manifest_sha256",
            "image_digest_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self._issuer is not _ADMISSION_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256
            != authorization.receipt_sha256
            or self.phase != expected_phase
            or self.job_contract_sha256 != expected_job_contract_sha256
            or self.pod_contract_sha256 != expected_pod_contract_sha256
            or self.completions != expected.get(self.phase)
            or not self.run_id
            or not self.job_name
            or not self.job_uid
            or self.image_reference != package.artifacts.image_reference
            or self.image_digest_sha256
            != package.artifacts.image_digest_sha256
            or not self.suspended_at_admission
            or self.economic_training_authorized
            or self.reinforcement_learning_authorized
            or self.outer_2026_access_authorized
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_ADMITTED_JOB_SCHEMA
        ):
            raise M03RV16ActivationError("V16 admitted Job authority drifted")

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)


def _issue_m03r_v16_admitted_job_authority(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    phase: Literal[
        "capacity", "training", "qualification-preflight", "qualification"
    ],
    run_id: str,
    job_name: str,
    job_contract_sha256: str,
    pod_contract_sha256: str,
    server_side_dry_run_file_sha256: str,
    server_side_dry_run_receipt_sha256: str,
    admitted_manifest_file_sha256: str,
    admitted_manifest_sha256: str,
    job_uid: str,
) -> M03RV16AdmittedJobAuthority:
    """Issue lifecycle evidence after admission; never from digest strings alone."""

    value = M03RV16AdmittedJobAuthority(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        phase=phase,
        run_id=run_id,
        job_name=job_name,
        job_contract_sha256=job_contract_sha256,
        pod_contract_sha256=pod_contract_sha256,
        server_side_dry_run_file_sha256=server_side_dry_run_file_sha256,
        server_side_dry_run_receipt_sha256=server_side_dry_run_receipt_sha256,
        admitted_manifest_file_sha256=admitted_manifest_file_sha256,
        admitted_manifest_sha256=admitted_manifest_sha256,
        job_uid=job_uid,
        completions={
            "capacity": 1,
            "training": 3,
            "qualification-preflight": 3,
            "qualification": 3,
        }[phase],
        image_reference=package.artifacts.image_reference,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        _issuer=_ADMISSION_ISSUER,
    )
    value.validate_for(
        package,
        authorization,
        expected_phase=phase,
        expected_job_contract_sha256=job_contract_sha256,
        expected_pod_contract_sha256=pod_contract_sha256,
    )
    return value


def write_m03r_v16_admitted_job_authority(
    path: str | Path, value: M03RV16AdmittedJobAuthority
) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return _write(
        Path(path), {"authority": payload, "receipt_sha256": value.receipt_sha256}
    )


def admitted_job_authority_file_sha256(
    value: M03RV16AdmittedJobAuthority,
) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return hashlib.sha256(
        canonical_json_file_bytes(
            {"authority": payload, "receipt_sha256": value.receipt_sha256}
        )
    ).hexdigest()


def load_m03r_v16_admitted_job_authority(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    expected_phase: str,
    expected_job_contract_sha256: str,
    expected_pod_contract_sha256: str,
    server_side_dry_run_path: str | Path,
    admitted_manifest_path: str | Path,
) -> M03RV16AdmittedJobAuthority:
    payload = _read_exact(Path(path), expected_file_sha256)
    try:
        row = dict(payload["authority"])
        value = M03RV16AdmittedJobAuthority(**row, _issuer=_ADMISSION_ISSUER)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError(
            "V16 admitted Job authority file is malformed"
        ) from exc
    if (
        payload.get("receipt_sha256") != value.receipt_sha256
        or value.receipt_sha256 != expected_receipt_sha256
    ):
        raise M03RV16ActivationError("V16 admitted Job receipt drifted")
    value.validate_for(
        package,
        authorization,
        expected_phase=expected_phase,
        expected_job_contract_sha256=expected_job_contract_sha256,
        expected_pod_contract_sha256=expected_pod_contract_sha256,
    )
    dry_run = _read_exact(
        Path(server_side_dry_run_path), value.server_side_dry_run_file_sha256
    )
    admitted = _read_exact(
        Path(admitted_manifest_path), value.admitted_manifest_file_sha256
    )
    dry_unsigned = {key: row for key, row in dry_run.items() if key != "receipt_sha256"}
    admitted_unsigned = {
        key: row for key, row in admitted.items() if key != "receipt_sha256"
    }
    if (
        dry_run.get("schema") != M03R_V16_DRY_RUN_RESULT_SCHEMA
        or dry_run.get("receipt_sha256")
        != value.server_side_dry_run_receipt_sha256
        or value.server_side_dry_run_receipt_sha256
        != semantic_sha256(dry_unsigned)
        or dry_run.get("package_plan_sha256") != package.package_plan_sha256
        or dry_run.get("phase") != expected_phase
        or dry_run.get("job_contract_sha256") != expected_job_contract_sha256
        or dry_run.get("pod_contract_sha256") != expected_pod_contract_sha256
        or dry_run.get("passed") is not True
        or admitted.get("schema") != M03R_V16_ADMITTED_MANIFEST_SCHEMA
        or admitted.get("receipt_sha256") != value.admitted_manifest_sha256
        or value.admitted_manifest_sha256 != semantic_sha256(admitted_unsigned)
        or admitted.get("package_plan_sha256") != package.package_plan_sha256
        or admitted.get("phase") != expected_phase
        or admitted.get("job_contract_sha256") != expected_job_contract_sha256
        or admitted.get("pod_contract_sha256") != expected_pod_contract_sha256
        or admitted.get("job_uid") != value.job_uid
        or admitted.get("job_name") != value.job_name
        or admitted.get("image_reference") != value.image_reference
        or admitted.get("image_digest_sha256") != value.image_digest_sha256
        or tuple(admitted.get("pod_uids", ())) != ()
        or tuple(admitted.get("container_image_ids", ())) != ()
        or tuple(admitted.get("node_names", ())) != ()
        or admitted.get("suspended_at_admission") is not True
    ):
        raise M03RV16ActivationError(
            "V16 admitted Job authority lacks exact lifecycle evidence"
        )
    return value


def _issue_m03r_v16_phase_launch_authority(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    phase: Literal[
        "capacity", "training", "qualification-preflight", "qualification"
    ],
    prerequisite_authority_receipt_sha256: str,
    job_contract_sha256: str,
    pod_contract_sha256: str,
    run_id: str,
    source_tree_root_sha256: str,
    admission: M03RV16AdmittedJobAuthority,
    admission_file_sha256: str,
    storage_semantics_file_sha256: str,
    storage_semantics_receipt_sha256: str,
    storage_authority_root_sha256: str,
    storage_observer_root_sha256: str,
) -> M03RV16PhaseLaunchAuthority:
    _digest("admission_file_sha256", admission_file_sha256)
    for name, digest_value in (
        ("storage_semantics_file_sha256", storage_semantics_file_sha256),
        ("storage_semantics_receipt_sha256", storage_semantics_receipt_sha256),
        ("storage_authority_root_sha256", storage_authority_root_sha256),
        ("storage_observer_root_sha256", storage_observer_root_sha256),
    ):
        _digest(name, digest_value)
    admission.validate_for(
        package,
        authorization,
        expected_phase=phase,
        expected_job_contract_sha256=job_contract_sha256,
        expected_pod_contract_sha256=pod_contract_sha256,
    )
    nonce = semantic_sha256(
        {
            "package": package.package_plan_sha256,
            "phase": phase,
            "prerequisite": prerequisite_authority_receipt_sha256,
            "job_contract": job_contract_sha256,
            "run_id": run_id,
            "admission_receipt": admission.receipt_sha256,
            "job_uid": admission.job_uid,
        }
    )
    value = M03RV16PhaseLaunchAuthority(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        phase=phase,
        prerequisite_authority_receipt_sha256=(
            prerequisite_authority_receipt_sha256
        ),
        admission_receipt_sha256=admission.receipt_sha256,
        admission_file_sha256=admission_file_sha256,
        job_contract_sha256=job_contract_sha256,
        pod_contract_sha256=pod_contract_sha256,
        run_id=run_id,
        job_uid=admission.job_uid,
        completions={
            "capacity": 1,
            "training": 3,
            "qualification-preflight": 3,
            "qualification": 3,
        }[phase],
        one_shot_nonce_sha256=nonce,
        pod_runtime_attestation_path_template=(
            f"pod-runtime/{phase}/{admission.job_uid}/"
            "completion-{completion_index:02d}.json"
        ),
        source_tree_root_sha256=source_tree_root_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        storage_semantics_file_sha256=storage_semantics_file_sha256,
        storage_semantics_receipt_sha256=storage_semantics_receipt_sha256,
        storage_authority_root_sha256=storage_authority_root_sha256,
        storage_observer_root_sha256=storage_observer_root_sha256,
        _issuer=_LAUNCH_ISSUER,
    )
    value.validate_for(
        package,
        authorization,
        expected_phase=phase,
        expected_prerequisite_receipt_sha256=prerequisite_authority_receipt_sha256,
        expected_job_contract_sha256=job_contract_sha256,
        expected_pod_contract_sha256=pod_contract_sha256,
        admission=admission,
        expected_admission_file_sha256=admission_file_sha256,
    )
    return value


def write_m03r_v16_phase_launch_authority(
    path: str | Path, value: M03RV16PhaseLaunchAuthority
) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return _write(
        Path(path), {"authority": payload, "receipt_sha256": value.receipt_sha256}
    )


def phase_launch_authority_file_sha256(
    value: M03RV16PhaseLaunchAuthority,
) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return hashlib.sha256(
        canonical_json_file_bytes(
            {"authority": payload, "receipt_sha256": value.receipt_sha256}
        )
    ).hexdigest()


def load_m03r_v16_phase_launch_authority(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    expected_phase: str,
    expected_prerequisite_receipt_sha256: str,
    expected_job_contract_sha256: str,
    expected_pod_contract_sha256: str,
    admission: M03RV16AdmittedJobAuthority,
    expected_admission_file_sha256: str,
) -> M03RV16PhaseLaunchAuthority:
    payload = _read_exact(Path(path), expected_file_sha256)
    try:
        row = dict(payload["authority"])
        value = M03RV16PhaseLaunchAuthority(**row, _issuer=_LAUNCH_ISSUER)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError("V16 phase launch file is malformed") from exc
    if (
        payload.get("receipt_sha256") != value.receipt_sha256
        or value.receipt_sha256 != expected_receipt_sha256
    ):
        raise M03RV16ActivationError("V16 phase launch receipt drifted")
    value.validate_for(
        package,
        authorization,
        expected_phase=expected_phase,
        expected_prerequisite_receipt_sha256=(
            expected_prerequisite_receipt_sha256
        ),
        expected_job_contract_sha256=expected_job_contract_sha256,
        expected_pod_contract_sha256=expected_pod_contract_sha256,
        admission=admission,
        expected_admission_file_sha256=expected_admission_file_sha256,
    )
    return value


def _image_identity(value: str) -> tuple[str | None, str]:
    normalized = value.removeprefix("docker-pullable://")
    if "@sha256:" in normalized:
        repository, digest = normalized.rsplit("@sha256:", 1)
    elif normalized.startswith(("containerd://sha256:", "docker://sha256:")):
        repository = None
        digest = normalized.rsplit("sha256:", 1)[1]
    else:
        raise M03RV16ActivationError("V16 runtime image is not digest pinned")
    _digest("runtime image digest", digest)
    if repository == "":
        raise M03RV16ActivationError("V16 runtime image repository is absent")
    return repository, digest


def _status_image_matches(
    value: str,
    *,
    expected_repository: str | None,
    expected_digest: str,
) -> bool:
    """Accept a pinned reference or a CRI-local image/config identifier."""

    if value.startswith("sha256:"):
        _digest("runtime status image ID", value.removeprefix("sha256:"))
        return True
    repository, digest = _image_identity(value)
    return repository == expected_repository and digest == expected_digest


def _image_id_repository_matches(
    repository: str | None,
    *,
    expected_repository: str | None,
) -> bool:
    if repository is None or repository == expected_repository:
        return True
    if expected_repository is None:
        return False
    last_slash = expected_repository.rfind("/")
    last_colon = expected_repository.rfind(":")
    tagless_repository = (
        expected_repository[:last_colon]
        if last_colon > last_slash
        else expected_repository
    )
    return repository == tagless_repository


@dataclass(frozen=True, slots=True)
class M03RV16PodRuntimeAttestation:
    """Per-completion identity observed only after a suspended Job resumes."""

    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    phase: Literal[
        "capacity", "training", "qualification-preflight", "qualification"
    ]
    run_id: str
    admission_receipt_sha256: str
    launch_authority_receipt_sha256: str
    job_contract_sha256: str
    pod_contract_sha256: str
    job_uid: str
    completion_index: int
    pod_uid: str
    pod_name: str
    node_name: str
    observed_owner_job_uid: str
    observed_owner_job_name: str
    observed_completion_index: int
    observed_pod_resource_version: str
    relative_path: str
    attested_container_name: str
    attested_container_kind: Literal["init", "app"]
    observed_spec_image: str
    observed_status_image: str
    observed_status_image_id: str
    normalized_image_digest: str
    output_root_sha256: str
    storage_semantics_file_sha256: str
    storage_semantics_receipt_sha256: str
    _issuer: object = field(repr=False)
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_POD_RUNTIME_ATTESTATION_SCHEMA

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
        *,
        admission: M03RV16AdmittedJobAuthority,
        launch: M03RV16PhaseLaunchAuthority,
        expected_completion_index: int,
        expected_output_root_sha256: str,
    ) -> None:
        package.validate()
        authorization.validate(package)
        for name in (
            "admission_receipt_sha256",
            "launch_authority_receipt_sha256",
            "job_contract_sha256",
            "pod_contract_sha256",
            "output_root_sha256",
            "normalized_image_digest",
            "storage_semantics_file_sha256",
            "storage_semantics_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        expected_repository, expected_digest = _image_identity(
            package.artifacts.image_reference
        )
        spec_repository, spec_digest = _image_identity(self.observed_spec_image)
        status_image_matches = _status_image_matches(
            self.observed_status_image,
            expected_repository=expected_repository,
            expected_digest=expected_digest,
        )
        image_id_repository, image_id_digest = _image_identity(
            self.observed_status_image_id
        )
        expected_relative_path = launch.pod_runtime_attestation_relative_path(
            expected_completion_index
        )
        if (
            self._issuer is not _POD_ATTESTATION_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256
            != authorization.receipt_sha256
            or self.phase != admission.phase
            or self.phase != launch.phase
            or self.run_id != admission.run_id
            or self.run_id != launch.run_id
            or self.admission_receipt_sha256 != admission.receipt_sha256
            or self.launch_authority_receipt_sha256 != launch.receipt_sha256
            or self.job_contract_sha256 != admission.job_contract_sha256
            or self.pod_contract_sha256 != admission.pod_contract_sha256
            or self.job_uid != admission.job_uid
            or self.completion_index != expected_completion_index
            or not 0 <= self.completion_index < admission.completions
            or not self.pod_uid
            or not self.pod_name
            or not self.node_name
            or self.observed_owner_job_uid != admission.job_uid
            or self.observed_owner_job_name != admission.job_name
            or self.observed_completion_index != expected_completion_index
            or not self.observed_pod_resource_version
            or self.relative_path != expected_relative_path
            or self.attested_container_name != "runtime-attestation-gate"
            or self.attested_container_kind != "init"
            or spec_repository != expected_repository
            or spec_digest != expected_digest
            or not status_image_matches
            or image_id_digest != expected_digest
            or not _image_id_repository_matches(
                image_id_repository,
                expected_repository=expected_repository,
            )
            or self.normalized_image_digest != expected_digest
            or self.output_root_sha256 != expected_output_root_sha256
            or self.storage_semantics_file_sha256
            != launch.storage_semantics_file_sha256
            or self.storage_semantics_receipt_sha256
            != launch.storage_semantics_receipt_sha256
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_POD_RUNTIME_ATTESTATION_SCHEMA
        ):
            raise M03RV16ActivationError("V16 Pod runtime attestation drifted")

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)


def _issue_m03r_v16_pod_runtime_attestation(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    admission: M03RV16AdmittedJobAuthority,
    launch: M03RV16PhaseLaunchAuthority,
    completion_index: int,
    pod_uid: str,
    pod_name: str,
    node_name: str,
    observed_owner_job_uid: str,
    observed_owner_job_name: str,
    observed_completion_index: int,
    observed_pod_resource_version: str,
    relative_path: str,
    attested_container_name: str,
    attested_container_kind: Literal["init", "app"],
    observed_spec_image: str,
    observed_status_image: str,
    observed_status_image_id: str,
    output_root_sha256: str,
) -> M03RV16PodRuntimeAttestation:
    value = M03RV16PodRuntimeAttestation(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        phase=admission.phase,
        run_id=admission.run_id,
        admission_receipt_sha256=admission.receipt_sha256,
        launch_authority_receipt_sha256=launch.receipt_sha256,
        job_contract_sha256=admission.job_contract_sha256,
        pod_contract_sha256=admission.pod_contract_sha256,
        job_uid=admission.job_uid,
        completion_index=completion_index,
        pod_uid=pod_uid,
        pod_name=pod_name,
        node_name=node_name,
        observed_owner_job_uid=observed_owner_job_uid,
        observed_owner_job_name=observed_owner_job_name,
        observed_completion_index=observed_completion_index,
        observed_pod_resource_version=observed_pod_resource_version,
        relative_path=relative_path,
        attested_container_name=attested_container_name,
        attested_container_kind=attested_container_kind,
        observed_spec_image=observed_spec_image,
        observed_status_image=observed_status_image,
        observed_status_image_id=observed_status_image_id,
        normalized_image_digest=package.artifacts.image_digest_sha256,
        output_root_sha256=output_root_sha256,
        storage_semantics_file_sha256=launch.storage_semantics_file_sha256,
        storage_semantics_receipt_sha256=(
            launch.storage_semantics_receipt_sha256
        ),
        _issuer=_POD_ATTESTATION_ISSUER,
    )
    value.validate_for(
        package,
        authorization,
        admission=admission,
        launch=launch,
        expected_completion_index=completion_index,
        expected_output_root_sha256=output_root_sha256,
    )
    return value


def write_m03r_v16_pod_runtime_attestation(
    path: str | Path, value: M03RV16PodRuntimeAttestation
) -> str:
    """Atomically publish a complete attestation at its final immutable path."""

    final_path = Path(path)
    expected_suffix = Path(value.relative_path)
    if tuple(final_path.parts[-len(expected_suffix.parts) :]) != expected_suffix.parts:
        raise M03RV16ActivationError(
            "V16 Pod attestation path differs from its launch namespace"
        )
    payload = asdict(value)
    payload.pop("_issuer")
    complete = {"attestation": payload, "receipt_sha256": value.receipt_sha256}
    data = canonical_json_file_bytes(complete)
    expected_file_sha256 = hashlib.sha256(data).hexdigest()
    final_path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    if final_path.exists() or final_path.is_symlink():
        if final_path.is_symlink():
            raise M03RV16ActivationError(
                "V16 Pod attestation final path is a symlink"
            )
        observed = _read_exact(final_path, expected_file_sha256)
        if observed != complete:
            raise M03RV16ActivationError(
                "V16 existing Pod attestation differs from the retry"
            )
        return expected_file_sha256
    temporary_path = final_path.with_name(
        f".{final_path.name}.{value.receipt_sha256}."
        f"{secrets.token_hex(8)}.tmp"
    )
    descriptor = os.open(
        temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
    )
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, final_path)
        published = True
        directory_descriptor = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if published:
            final_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)
    return expected_file_sha256


def pod_runtime_attestation_file_sha256(
    value: M03RV16PodRuntimeAttestation,
) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return hashlib.sha256(
        canonical_json_file_bytes(
            {"attestation": payload, "receipt_sha256": value.receipt_sha256}
        )
    ).hexdigest()


def pod_runtime_attestation_file_identity(
    path: str | Path,
) -> tuple[str, str]:
    """Discover the identities of one canonical, append-only attestation."""

    payload, file_sha256 = _read_canonical(Path(path))
    attestation = payload.get("attestation")
    receipt_sha256 = payload.get("receipt_sha256")
    if not isinstance(attestation, dict) or not isinstance(receipt_sha256, str):
        raise M03RV16ActivationError(
            "V16 Pod runtime attestation identity is malformed"
        )
    _digest("Pod runtime attestation receipt", receipt_sha256)
    if semantic_sha256(attestation) != receipt_sha256:
        raise M03RV16ActivationError(
            "V16 Pod runtime attestation self-receipt drifted"
        )
    return file_sha256, receipt_sha256


def load_m03r_v16_pod_runtime_attestation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    admission: M03RV16AdmittedJobAuthority,
    launch: M03RV16PhaseLaunchAuthority,
    expected_completion_index: int,
    expected_output_root_sha256: str,
    current_pod_uid: str,
    current_pod_name: str,
    current_node_name: str,
    expected_relative_path: str | None = None,
) -> M03RV16PodRuntimeAttestation:
    payload = _read_exact(Path(path), expected_file_sha256)
    try:
        value = M03RV16PodRuntimeAttestation(
            **dict(payload["attestation"]), _issuer=_POD_ATTESTATION_ISSUER
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError(
            "V16 Pod runtime attestation is malformed"
        ) from exc
    if (
        payload.get("receipt_sha256") != value.receipt_sha256
        or value.receipt_sha256 != expected_receipt_sha256
    ):
        raise M03RV16ActivationError("V16 Pod runtime attestation receipt drifted")
    value.validate_for(
        package,
        authorization,
        admission=admission,
        launch=launch,
        expected_completion_index=expected_completion_index,
        expected_output_root_sha256=expected_output_root_sha256,
    )
    if (
        value.pod_uid != current_pod_uid
        or value.pod_name != current_pod_name
        or value.node_name != current_node_name
        or (
            expected_relative_path is not None
            and value.relative_path != expected_relative_path
        )
    ):
        raise M03RV16ActivationError(
            "V16 worker Pod identity differs from runtime attestation"
        )
    return value


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
    expected_authorization_file_sha256: str,
    static_result_path: str | Path,
    expected_static_result_file_sha256: str,
    capacity_terminal_path: str | Path,
    expected_capacity_terminal_file_sha256: str,
) -> M03RV16TrainingActivation:
    _digest("expected_authorization_file_sha256", expected_authorization_file_sha256)
    payload = _read_exact(Path(path), expected_file_sha256)
    try:
        value = M03RV16TrainingActivation(**payload["activation"], _issuer=_TRAINING_ISSUER)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError("V16 training activation is malformed") from exc
    if payload.get("receipt_sha256") != value.receipt_sha256:
        raise M03RV16ActivationError("V16 training activation receipt drifted")
    value.validate_for(package, authorization)
    if (
        expected_static_result_file_sha256 != value.static_result_file_sha256
        or expected_capacity_terminal_file_sha256
        != value.capacity_terminal_file_sha256
    ):
        raise M03RV16ActivationError(
            "V16 predecessor file identity differs from the activation"
        )
    static_result = _read_exact(
        Path(static_result_path), value.static_result_file_sha256
    )
    static_unsigned = {
        key: row for key, row in static_result.items() if key != "receipt_sha256"
    }
    capacity_terminal = _read_exact(
        Path(capacity_terminal_path), value.capacity_terminal_file_sha256
    )
    try:
        capacity_row = dict(capacity_terminal["capacity"])
        rank_rows = tuple(
            M03RV16CapacityRankEvidence(**dict(row))
            for row in capacity_row.pop("rank_evidence")
        )
        typed_capacity = M03RV16CapacityTerminal(
            rank_evidence=rank_rows, **capacity_row
        )
        typed_capacity.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError(
            "V16 training activation capacity terminal is incomplete"
        ) from exc
    expected_static_gate = {
        "package_plan_sha256": package.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "rendered_manifest_sha256": value.static_rendered_manifest_sha256,
        "result_file_sha256": value.static_result_file_sha256,
        "result_receipt_sha256": value.static_result_receipt_sha256,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "source_tree_root_sha256": value.source_tree_root_sha256,
        "zero_gpu_requested": True,
        "zero_gpu_admitted": True,
        "zero_gpu_observed": True,
        "training_performed": False,
        "passed": True,
        "development_only": True,
        "schema": "rl-quant.top2000-dev.m03r-v16-static-gate-qualification-v2",
    }
    expected_capacity_gate = {
        "package_plan_sha256": package.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "static_gate_receipt_sha256": value.static_gate_receipt_sha256,
        "rendered_manifest_sha256": value.capacity_rendered_manifest_sha256,
        "terminal_file_sha256": value.capacity_terminal_file_sha256,
        "terminal_receipt_sha256": value.capacity_terminal_receipt_sha256,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "source_tree_root_sha256": value.source_tree_root_sha256,
        "world_size": 2,
        "h100s_per_worker": 2,
        "exact_h100_80gb_per_rank": True,
        "disposable_exact_shape_update_performed": True,
        "disposable_train_validate_train_executed": True,
        "nontrivial_qualification_projection_performed": True,
        "scientific_checkpoint_published": False,
        "scientific_training_performed": False,
        "passed": True,
        "development_only": True,
        "schema": "rl-quant.top2000-dev.m03r-v16-capacity-gate-qualification-v2",
    }
    if (
        static_result.get("receipt_sha256") != value.static_result_receipt_sha256
        or value.static_result_receipt_sha256 != semantic_sha256(static_unsigned)
        or static_result.get("schema") != M03R_V16_STATIC_RESULT_SCHEMA
        or static_result.get("package_plan_sha256") != package.package_plan_sha256
        or static_result.get("package_plan_file_sha256")
        != authorization.package_plan_file_sha256
        or static_result.get("execution_authorization_receipt_sha256")
        != authorization.receipt_sha256
        or static_result.get("execution_authorization_file_sha256")
        != expected_authorization_file_sha256
        or static_result.get("source_archive_sha256")
        != package.artifacts.source_archive_sha256
        or static_result.get("source_manifest_sha256")
        != package.artifacts.source_manifest_sha256
        or static_result.get("worker_source_sha256")
        != package.artifacts.worker_source_sha256
        or static_result.get("source_tree_root_sha256")
        != value.source_tree_root_sha256
        or static_result.get("structural_slab_file_sha256")
        != package.artifacts.structural_slab_file_sha256
        or static_result.get("structural_slab_receipt_sha256")
        != package.artifacts.structural_slab_receipt_sha256
        or static_result.get("panel_schedule_sha256")
        != package.schedule.receipt_sha256
        or static_result.get("hold_target_sessions") != 30
        or static_result.get("hold_target_spec_sha256")
        != package.hold_target_spec_sha256
        or static_result.get("image_digest_sha256")
        != package.artifacts.image_digest_sha256
        or static_result.get("training_performed") is not False
        or static_result.get("gpu_mask") != "none"
        or static_result.get("gpu_requests") != 0
        or static_result.get("gpu_limits") != 0
        or static_result.get("unmasked_visibility_claimed") is not False
        or static_result.get("initial_state_strict_loaded_all_settings") is not True
        or static_result.get("output_empty") is not True
        or static_result.get("container_started") is not True
        or static_result.get("economic_training_authorized") is not False
        or static_result.get("reinforcement_learning_authorized") is not False
        or static_result.get("outer_2026_access_authorized") is not False
        or static_result.get("development_only") is not True
        or static_result.get("reportable") is not False
        or static_result.get("promotion_eligible") is not False
        or capacity_terminal.get("schema") != M03R_V16_CAPACITY_TERMINAL_SCHEMA
        or capacity_terminal.get("package_plan_sha256")
        != package.package_plan_sha256
        or capacity_terminal.get("authorization_receipt_sha256")
        != authorization.receipt_sha256
        or capacity_terminal.get("source_tree_root_sha256")
        != value.source_tree_root_sha256
        or semantic_sha256(capacity_terminal)
        != value.capacity_terminal_receipt_sha256
        or capacity_terminal.get("scientific_training_performed") is not False
        or capacity_terminal.get("disposable_train_validate_train_executed")
        is not True
        or capacity_terminal.get("disposable_optimizer_update_executed") is not True
        or capacity_terminal.get("scientific_checkpoint_published") is not False
        or not isinstance(capacity_terminal.get("capacity"), dict)
        or capacity_terminal.get("capacity_receipt_sha256")
        != typed_capacity.receipt_sha256
        or typed_capacity.setting_index != 0
        or semantic_sha256(expected_static_gate)
        != value.static_gate_receipt_sha256
        or semantic_sha256(expected_capacity_gate)
        != value.capacity_gate_receipt_sha256
    ):
        raise M03RV16ActivationError(
            "V16 training activation lacks exact predecessor evidence"
        )
    return value


def write_m03r_v16_qualification_activation(
    path: str | Path, value: M03RV16QualificationActivation
) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return _write(Path(path), {"activation": payload, "receipt_sha256": value.receipt_sha256})


def _issue_m03r_v16_qualification_outer_access_authority(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    activation: M03RV16QualificationActivation,
    setting_input_closure_file_sha256: tuple[str, str, str],
    setting_input_closure_receipt_sha256: tuple[str, str, str],
    setting_preflight_terminal_file_sha256: tuple[str, str, str],
    setting_preflight_terminal_receipt_sha256: tuple[str, str, str],
    qualification_risk_input_root_sha256: str,
) -> M03RV16QualificationOuterAccessAuthority:
    value = M03RV16QualificationOuterAccessAuthority(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        qualification_activation_receipt_sha256=activation.receipt_sha256,
        training_panel_receipt_sha256=(
            activation.training_panel_receipt_sha256
        ),
        setting_input_closure_file_sha256=(
            setting_input_closure_file_sha256
        ),
        setting_input_closure_receipt_sha256=(
            setting_input_closure_receipt_sha256
        ),
        setting_preflight_terminal_file_sha256=(
            setting_preflight_terminal_file_sha256
        ),
        setting_preflight_terminal_receipt_sha256=(
            setting_preflight_terminal_receipt_sha256
        ),
        qualification_risk_input_root_sha256=(
            qualification_risk_input_root_sha256
        ),
        source_tree_root_sha256=activation.source_tree_root_sha256,
        _issuer=_QUALIFICATION_OUTER_ACCESS_ISSUER,
    )
    value.validate_for(package, authorization, activation)
    return value


def write_m03r_v16_qualification_outer_access_authority(
    path: str | Path,
    value: M03RV16QualificationOuterAccessAuthority,
) -> str:
    payload = asdict(value)
    payload.pop("_issuer")
    return _write(
        Path(path),
        {"authority": payload, "receipt_sha256": value.receipt_sha256},
    )


def load_m03r_v16_qualification_outer_access_authority(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    activation: M03RV16QualificationActivation,
    setting_input_closure_paths: tuple[Path, Path, Path],
    setting_preflight_terminal_paths: tuple[Path, Path, Path],
) -> M03RV16QualificationOuterAccessAuthority:
    payload = _read_exact(Path(path), expected_file_sha256)
    try:
        row = dict(payload["authority"])
        row["setting_input_closure_file_sha256"] = tuple(
            row["setting_input_closure_file_sha256"]
        )
        row["setting_input_closure_receipt_sha256"] = tuple(
            row["setting_input_closure_receipt_sha256"]
        )
        row["setting_preflight_terminal_file_sha256"] = tuple(
            row["setting_preflight_terminal_file_sha256"]
        )
        row["setting_preflight_terminal_receipt_sha256"] = tuple(
            row["setting_preflight_terminal_receipt_sha256"]
        )
        row["setting_indices"] = tuple(row["setting_indices"])
        value = M03RV16QualificationOuterAccessAuthority(
            **row, _issuer=_QUALIFICATION_OUTER_ACCESS_ISSUER
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError(
            "V16 qualification outer-access authority is malformed"
        ) from exc
    if (
        payload.get("receipt_sha256") != value.receipt_sha256
        or value.receipt_sha256 != expected_receipt_sha256
    ):
        raise M03RV16ActivationError(
            "V16 qualification outer-access receipt drifted"
        )
    value.validate_for(package, authorization, activation)
    closures = tuple(
        _read_exact(path, expected)
        for path, expected in zip(
            setting_input_closure_paths,
            value.setting_input_closure_file_sha256,
            strict=True,
        )
    )
    terminals = tuple(
        _read_exact(path, expected)
        for path, expected in zip(
            setting_preflight_terminal_paths,
            value.setting_preflight_terminal_file_sha256,
            strict=True,
        )
    )
    risk_rows: list[str] = []
    for index, closure in enumerate(closures):
        unsigned = {
            key: row for key, row in closure.items() if key != "receipt_sha256"
        }
        fold_rows = closure.get("folds")
        if (
            closure.get("schema")
            != "rl-quant.top2000-dev.m03r-v16-qualification-inputs-complete-v3"
            or closure.get("receipt_sha256") != semantic_sha256(unsigned)
            or closure.get("receipt_sha256")
            != value.setting_input_closure_receipt_sha256[index]
            or closure.get("package_plan_sha256")
            != package.package_plan_sha256
            or closure.get("authorization_receipt_sha256")
            != authorization.receipt_sha256
            or closure.get("qualification_activation_receipt_sha256")
            != activation.receipt_sha256
            or closure.get("setting_index") != index
            or closure.get("protocol_sha256") != M03R_V16_PROTOCOL_SHA256
            or not isinstance(fold_rows, (list, tuple))
            or len(fold_rows) != 5
            or closure.get("outer_qualification_access_started") is not False
            or closure.get("outer_2026_accessed") is not False
        ):
            raise M03RV16ActivationError(
                "V16 qualification input closure drifted"
            )
        setting_risk_rows: list[str] = []
        for fold_index, fold_row in enumerate(fold_rows):
            if not isinstance(fold_row, dict):
                raise M03RV16ActivationError(
                    "V16 qualification fold closure is malformed"
                )
            checkpoint_sha = str(fold_row.get("checkpoint_file_sha256", ""))
            selection_sha = str(
                fold_row.get("checkpoint_selection_receipt_sha256", "")
            )
            risk_sha = str(fold_row.get("risk_state_sha256", ""))
            for digest_name, digest_value in (
                ("closure checkpoint", checkpoint_sha),
                ("closure selection", selection_sha),
                ("closure risk state", risk_sha),
            ):
                _digest(digest_name, digest_value)
            if (
                fold_row.get("fold_index") != fold_index
                or checkpoint_sha
                != activation.terminal_checkpoint_file_sha256[index][fold_index]
                or fold_row.get("risk_inputs_validated") is not True
            ):
                raise M03RV16ActivationError(
                    "V16 qualification fold closure drifted"
                )
            setting_risk_rows.append(risk_sha)
        setting_risk_root = semantic_sha256(tuple(setting_risk_rows))
        if (
            closure.get("qualification_risk_input_root_sha256")
            != setting_risk_root
        ):
            raise M03RV16ActivationError(
                "V16 qualification setting risk root drifted"
            )
        risk_rows.append(setting_risk_root)
        terminal = terminals[index]
        terminal_unsigned = {
            key: row for key, row in terminal.items() if key != "receipt_sha256"
        }
        if (
            terminal.get("schema")
            != "rl-quant.top2000-dev.m03r-v16-qualification-preflight-terminal-v2"
            or terminal.get("receipt_sha256")
            != semantic_sha256(terminal_unsigned)
            or terminal.get("receipt_sha256")
            != value.setting_preflight_terminal_receipt_sha256[index]
            or terminal.get("package_plan_sha256")
            != package.package_plan_sha256
            or terminal.get("authorization_receipt_sha256")
            != authorization.receipt_sha256
            or terminal.get("qualification_activation_receipt_sha256")
            != activation.receipt_sha256
            or terminal.get("setting_index") != index
            or terminal.get("qualification_input_closure_file_sha256")
            != value.setting_input_closure_file_sha256[index]
            or terminal.get("qualification_input_closure_receipt_sha256")
            != value.setting_input_closure_receipt_sha256[index]
            or terminal.get("qualification_risk_input_root_sha256")
            != setting_risk_root
            or terminal.get("source_tree_root_sha256")
            != activation.source_tree_root_sha256
            or terminal.get("gpu_requested") is not False
            or terminal.get("gpu_visible") is not False
            or terminal.get("outer_qualification_access_started") is not False
            or terminal.get("outer_2026_accessed") is not False
            or terminal.get("resource_profile_id")
            != "qualification-preflight-cpu12-memory64gi-limit128gi-v1"
            or not isinstance(terminal.get("measured_peak_rss_bytes"), int)
            or isinstance(terminal.get("measured_peak_rss_bytes"), bool)
            or int(terminal.get("measured_peak_rss_bytes", 0)) <= 0
            or not isinstance(
                terminal.get("measured_process_cpu_seconds"), (int, float)
            )
            or isinstance(terminal.get("measured_process_cpu_seconds"), bool)
            or not math.isfinite(
                float(terminal.get("measured_process_cpu_seconds", -1.0))
            )
            or float(terminal.get("measured_process_cpu_seconds", -1.0)) < 0.0
            or not isinstance(
                terminal.get("measured_wall_seconds"), (int, float)
            )
            or isinstance(terminal.get("measured_wall_seconds"), bool)
            or not math.isfinite(
                float(terminal.get("measured_wall_seconds", -1.0))
            )
            or float(terminal.get("measured_wall_seconds", -1.0)) <= 0.0
        ):
            raise M03RV16ActivationError(
                "V16 qualification preflight terminal drifted"
            )
        for digest_name in (
            "launch_authority_receipt_sha256",
            "pod_runtime_attestation_receipt_sha256",
            "storage_semantics_file_sha256",
            "storage_semantics_receipt_sha256",
        ):
            _digest(digest_name, str(terminal.get(digest_name, "")))
    if semantic_sha256(tuple(risk_rows)) != value.qualification_risk_input_root_sha256:
        raise M03RV16ActivationError(
            "V16 qualification risk-input closure root drifted"
        )
    return value


def load_m03r_v16_training_panel_authority(
    *,
    training_panel_path: str | Path,
    expected_training_panel_file_sha256: str,
    prequalification_closure_path: str | Path,
    expected_prequalification_closure_file_sha256: str,
    training_terminal_paths: tuple[Path, Path, Path],
    expected_training_terminal_file_sha256: tuple[str, str, str],
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
) -> M03RV16TrainingPanelAuthority:
    """Reopen the complete panel/closure/terminal boundary and issue authority."""

    panel = _read_exact(
        Path(training_panel_path), expected_training_panel_file_sha256
    )
    panel_unsigned = {
        key: row for key, row in panel.items() if key != "receipt_sha256"
    }
    closure = _read_exact(
        Path(prequalification_closure_path),
        expected_prequalification_closure_file_sha256,
    )
    closure_unsigned = {
        key: row for key, row in closure.items() if key != "receipt_sha256"
    }
    terminals = tuple(
        _read_exact(path, expected)
        for path, expected in zip(
            training_terminal_paths,
            expected_training_terminal_file_sha256,
            strict=True,
        )
    )
    try:
        terminal_receipts = tuple(str(row["receipt_sha256"]) for row in terminals)
        adequacy_receipts = tuple(
            tuple(str(value) for value in row)
            for row in panel["setting_fold_adequacy_receipt_sha256"]
        )
        adequacy_status = tuple(
            tuple(str(value) for value in row)
            for row in panel["setting_fold_adequacy_status"]
        )
        checkpoint_matrix = tuple(
            tuple(str(value) for value in row)
            for row in panel["terminal_checkpoint_file_sha256"]
        )
        authority = M03RV16TrainingPanelAuthority(
            package_plan_sha256=str(panel["package_plan_sha256"]),
            execution_authorization_receipt_sha256=str(
                panel["execution_authorization_receipt_sha256"]
            ),
            training_panel_receipt_sha256=str(panel["receipt_sha256"]),
            training_panel_file_sha256=expected_training_panel_file_sha256,
            training_terminal_file_sha256=expected_training_terminal_file_sha256,
            training_terminal_receipt_sha256=terminal_receipts,  # type: ignore[arg-type]
            setting_fold_training_adequacy_receipt_sha256=adequacy_receipts,  # type: ignore[arg-type]
            setting_fold_training_adequacy_status=adequacy_status,  # type: ignore[arg-type]
            terminal_checkpoint_file_sha256=checkpoint_matrix,  # type: ignore[arg-type]
            prequalification_closure_receipt_sha256=str(
                panel["prequalification_closure_receipt_sha256"]
            ),
            prequalification_closure_file_sha256=(
                expected_prequalification_closure_file_sha256
            ),
            source_tree_root_sha256=str(panel["source_tree_root_sha256"]),
            all_setting_folds_adequate=bool(
                panel["all_setting_folds_adequate"]
            ),
            outer_qualification_outcomes_accessed=bool(
                panel["outer_qualification_outcomes_accessed"]
            ),
            _issuer=_TRAINING_PANEL_ISSUER,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError(
            "V16 training panel authority is malformed"
        ) from exc
    authority.validate_for(package, authorization)
    if (
        panel.get("schema") != M03R_V16_TRAINING_PANEL_SCHEMA
        or panel.get("protocol_sha256") != M03R_V16_PROTOCOL_SHA256
        or panel.get("receipt_sha256") != semantic_sha256(panel_unsigned)
        or tuple(panel.get("training_terminal_file_sha256", ()))
        != expected_training_terminal_file_sha256
        or tuple(panel.get("training_terminal_receipt_sha256", ()))
        != terminal_receipts
        or panel.get("prequalification_closure_file_sha256")
        != expected_prequalification_closure_file_sha256
        or panel.get("qualification_input_preflight_authorized") is not True
        or panel.get("outer_qualification_authorized") is not False
        or panel.get("next_research_action") != "qualification-input-preflight"
        or panel.get("economic_generation_may_be_minted") is not False
        or panel.get("reinforcement_learning_authorized") is not False
        or panel.get("outer_2026_accessed") is not False
        or closure.get("schema") != M03R_V16_PREQUALIFICATION_CLOSURE_SCHEMA
        or closure.get("protocol_sha256") != M03R_V16_PROTOCOL_SHA256
        or closure.get("receipt_sha256") != semantic_sha256(closure_unsigned)
        or closure.get("receipt_sha256")
        != authority.prequalification_closure_receipt_sha256
        or tuple(closure.get("training_terminal_file_sha256", ()))
        != expected_training_terminal_file_sha256
        or tuple(
            tuple(row)
            for row in closure.get("terminal_checkpoint_file_sha256", ())
        )
        != authority.terminal_checkpoint_file_sha256
        or closure.get("all_setting_folds_adequate") is not True
        or closure.get("outer_qualification_outcomes_accessed") is not False
    ):
        raise M03RV16ActivationError(
            "V16 training panel lacks complete prequalification evidence"
        )
    for setting, terminal in enumerate(terminals):
        unsigned = {
            key: row for key, row in terminal.items() if key != "receipt_sha256"
        }
        if (
            terminal.get("schema") != M03R_V16_TRAINING_TERMINAL_SCHEMA
            or terminal.get("receipt_sha256") != semantic_sha256(unsigned)
            or terminal.get("package_plan_sha256") != package.package_plan_sha256
            or terminal.get("authorization_receipt_sha256")
            != authorization.receipt_sha256
            or terminal.get("setting_index") != setting
            or terminal.get("source_tree_root_sha256")
            != authority.source_tree_root_sha256
            or tuple(terminal.get("fold_training_adequacy_status", ()))
            != ("adequate",) * 5
            or terminal.get("qualification_tail_accessed") is not False
            or terminal.get("outer_qualification_authorized") is not False
            or terminal.get("three_seed_confirmation_may_be_minted") is not False
        ):
            raise M03RV16ActivationError(
                "V16 training-panel terminal evidence drifted"
            )
    return authority


def load_m03r_v16_qualification_activation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    training_panel_path: str | Path,
    prequalification_closure_path: str | Path,
    training_terminal_paths: tuple[Path, Path, Path],
) -> M03RV16QualificationActivation:
    payload = _read_exact(Path(path), expected_file_sha256)
    try:
        row = dict(payload["activation"])
        row["training_terminal_file_sha256"] = tuple(row["training_terminal_file_sha256"])
        row["setting_fold_training_adequacy_receipt_sha256"] = tuple(
            tuple(value)
            for value in row["setting_fold_training_adequacy_receipt_sha256"]
        )
        row["terminal_checkpoint_file_sha256"] = tuple(
            tuple(value) for value in row["terminal_checkpoint_file_sha256"]
        )
        value = M03RV16QualificationActivation(**row, _issuer=_QUALIFICATION_ISSUER)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16ActivationError("V16 qualification activation is malformed") from exc
    if payload.get("receipt_sha256") != value.receipt_sha256:
        raise M03RV16ActivationError("V16 qualification activation receipt drifted")
    value.validate_for(package, authorization)
    panel = load_m03r_v16_training_panel_authority(
        training_panel_path=training_panel_path,
        expected_training_panel_file_sha256=value.training_panel_file_sha256,
        prequalification_closure_path=prequalification_closure_path,
        expected_prequalification_closure_file_sha256=(
            value.prequalification_closure_file_sha256
        ),
        training_terminal_paths=training_terminal_paths,
        expected_training_terminal_file_sha256=(
            value.training_terminal_file_sha256
        ),
        package=package,
        authorization=authorization,
    )
    if (
        panel.training_panel_receipt_sha256
        != value.training_panel_receipt_sha256
        or panel.setting_fold_training_adequacy_receipt_sha256
        != value.setting_fold_training_adequacy_receipt_sha256
        or panel.terminal_checkpoint_file_sha256
        != value.terminal_checkpoint_file_sha256
        or panel.prequalification_closure_receipt_sha256
        != value.prequalification_closure_receipt_sha256
        or panel.source_tree_root_sha256 != value.source_tree_root_sha256
    ):
        raise M03RV16ActivationError(
            "V16 qualification activation lacks exact training-panel evidence"
        )
    return value


def _issue_m03r_v16_training_activation_from_gates(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    static: Any,
    capacity: Any,
) -> M03RV16TrainingActivation:
    static.validate_for(package, authorization)
    capacity.validate_for(package, authorization, static)
    value = M03RV16TrainingActivation(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        static_gate_receipt_sha256=static.receipt_sha256,
        static_rendered_manifest_sha256=static.rendered_manifest_sha256,
        static_result_file_sha256=static.result_file_sha256,
        static_result_receipt_sha256=static.result_receipt_sha256,
        capacity_gate_receipt_sha256=capacity.receipt_sha256,
        capacity_rendered_manifest_sha256=capacity.rendered_manifest_sha256,
        capacity_terminal_file_sha256=capacity.terminal_file_sha256,
        capacity_terminal_receipt_sha256=capacity.terminal_receipt_sha256,
        source_tree_root_sha256=capacity.source_tree_root_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        _issuer=_TRAINING_ISSUER,
    )
    value.validate_for(package, authorization)
    return value


def _issue_m03r_v16_qualification_activation_from_panel_authority(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    panel: M03RV16TrainingPanelAuthority,
) -> M03RV16QualificationActivation:
    panel.validate_for(package, authorization)
    value = M03RV16QualificationActivation(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        training_panel_receipt_sha256=panel.training_panel_receipt_sha256,
        training_panel_file_sha256=panel.training_panel_file_sha256,
        training_terminal_file_sha256=panel.training_terminal_file_sha256,
        setting_fold_training_adequacy_receipt_sha256=(
            panel.setting_fold_training_adequacy_receipt_sha256
        ),
        terminal_checkpoint_file_sha256=panel.terminal_checkpoint_file_sha256,
        prequalification_closure_receipt_sha256=(
            panel.prequalification_closure_receipt_sha256
        ),
        prequalification_closure_file_sha256=(
            panel.prequalification_closure_file_sha256
        ),
        source_tree_root_sha256=panel.source_tree_root_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        _issuer=_QUALIFICATION_ISSUER,
    )
    value.validate_for(package, authorization)
    return value


__all__ = [
    "M03R_V16_ADMITTED_JOB_SCHEMA",
    "M03R_V16_PHASE_LAUNCH_SCHEMA",
    "M03R_V16_POD_RUNTIME_ATTESTATION_SCHEMA",
    "M03R_V16_QUALIFICATION_ACTIVATION_SCHEMA",
    "M03R_V16_QUALIFICATION_OUTER_ACCESS_SCHEMA",
    "M03R_V16_TRAINING_ACTIVATION_SCHEMA",
    "M03RV16ActivationError",
    "M03RV16AdmittedJobAuthority",
    "M03RV16PhaseLaunchAuthority",
    "M03RV16PodRuntimeAttestation",
    "M03RV16QualificationActivation",
    "M03RV16QualificationOuterAccessAuthority",
    "M03RV16TrainingActivation",
    "M03RV16TrainingPanelAuthority",
    "admitted_job_authority_file_sha256",
    "load_m03r_v16_admitted_job_authority",
    "load_m03r_v16_phase_launch_authority",
    "load_m03r_v16_pod_runtime_attestation",
    "load_m03r_v16_qualification_activation",
    "load_m03r_v16_qualification_outer_access_authority",
    "load_m03r_v16_training_activation",
    "load_m03r_v16_training_panel_authority",
    "phase_launch_authority_file_sha256",
    "pod_runtime_attestation_file_identity",
    "pod_runtime_attestation_file_sha256",
    "write_m03r_v16_admitted_job_authority",
    "write_m03r_v16_phase_launch_authority",
    "write_m03r_v16_pod_runtime_attestation",
    "write_m03r_v16_qualification_activation",
    "write_m03r_v16_qualification_outer_access_authority",
    "write_m03r_v16_training_activation",
]
