"""Immutable phase activations for the two-stage M03R-v16 workflow."""

from __future__ import annotations

import hashlib
import json
import os
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
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
)
from rl_quant.training.top2000_m03r_v16_static_contract import (
    M03R_V16_STATIC_RESULT_SCHEMA,
)

M03R_V16_TRAINING_ACTIVATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-activation-v2"
)
M03R_V16_QUALIFICATION_ACTIVATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-qualification-activation-v2"
)
M03R_V16_PHASE_LAUNCH_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-phase-launch-authority-v1"
)
M03R_V16_ADMITTED_JOB_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-admitted-job-authority-v1"
)
M03R_V16_DRY_RUN_RESULT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-server-dry-run-result-v1"
)
M03R_V16_ADMITTED_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-admitted-manifest-result-v1"
)
_TRAINING_ISSUER = object()
_QUALIFICATION_ISSUER = object()
_LAUNCH_ISSUER = object()
_ADMISSION_ISSUER = object()
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
        _digest("training_panel_file_sha256", self.training_panel_file_sha256)
        _digest(
            "prequalification_closure_receipt_sha256",
            self.prequalification_closure_receipt_sha256,
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


@dataclass(frozen=True, slots=True)
class M03RV16PhaseLaunchAuthority:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    phase: Literal["capacity", "training", "qualification"]
    prerequisite_authority_receipt_sha256: str
    admission_receipt_sha256: str
    admission_file_sha256: str
    job_contract_sha256: str
    pod_contract_sha256: str
    run_id: str
    job_uid: str
    pod_uids: tuple[str, ...]
    completions: int
    one_shot_nonce_sha256: str
    source_tree_root_sha256: str
    image_digest_sha256: str
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
        ):
            _digest(name, getattr(self, name))
        expected_completions = {"capacity": 1, "training": 3, "qualification": 3}
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
            or self.pod_uids != admission.pod_uids
            or self.image_digest_sha256 != package.artifacts.image_digest_sha256
            or self.economic_training_authorized
            or self.reinforcement_learning_authorized
            or self.outer_2026_access_authorized
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_PHASE_LAUNCH_SCHEMA
        ):
            raise M03RV16ActivationError("V16 phase launch authority drifted")

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)


@dataclass(frozen=True, slots=True)
class M03RV16AdmittedJobAuthority:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    phase: Literal["capacity", "training", "qualification"]
    run_id: str
    job_contract_sha256: str
    pod_contract_sha256: str
    server_side_dry_run_file_sha256: str
    server_side_dry_run_receipt_sha256: str
    admitted_manifest_file_sha256: str
    admitted_manifest_sha256: str
    job_uid: str
    pod_uids: tuple[str, ...]
    container_image_ids: tuple[str, ...]
    node_names: tuple[str, ...]
    completions: int
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
        expected = {"capacity": 1, "training": 3, "qualification": 3}
        for name in (
            "job_contract_sha256",
            "pod_contract_sha256",
            "server_side_dry_run_file_sha256",
            "server_side_dry_run_receipt_sha256",
            "admitted_manifest_file_sha256",
            "admitted_manifest_sha256",
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
            or len(self.pod_uids) != self.completions
            or len(self.container_image_ids) != self.completions
            or len(self.node_names) != self.completions
            or len(set(self.pod_uids)) != self.completions
            or not self.run_id
            or not self.job_uid
            or any(not value for value in (*self.pod_uids, *self.node_names))
            or any(
                "@sha256:" not in value for value in self.container_image_ids
            )
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
    phase: Literal["capacity", "training", "qualification"],
    run_id: str,
    job_contract_sha256: str,
    pod_contract_sha256: str,
    server_side_dry_run_file_sha256: str,
    server_side_dry_run_receipt_sha256: str,
    admitted_manifest_file_sha256: str,
    admitted_manifest_sha256: str,
    job_uid: str,
    pod_uids: tuple[str, ...],
    container_image_ids: tuple[str, ...],
    node_names: tuple[str, ...],
) -> M03RV16AdmittedJobAuthority:
    """Issue lifecycle evidence after admission; never from digest strings alone."""

    value = M03RV16AdmittedJobAuthority(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        phase=phase,
        run_id=run_id,
        job_contract_sha256=job_contract_sha256,
        pod_contract_sha256=pod_contract_sha256,
        server_side_dry_run_file_sha256=server_side_dry_run_file_sha256,
        server_side_dry_run_receipt_sha256=server_side_dry_run_receipt_sha256,
        admitted_manifest_file_sha256=admitted_manifest_file_sha256,
        admitted_manifest_sha256=admitted_manifest_sha256,
        job_uid=job_uid,
        pod_uids=pod_uids,
        container_image_ids=container_image_ids,
        node_names=node_names,
        completions={"capacity": 1, "training": 3, "qualification": 3}[phase],
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
        row["pod_uids"] = tuple(row["pod_uids"])
        row["container_image_ids"] = tuple(row["container_image_ids"])
        row["node_names"] = tuple(row["node_names"])
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
        or tuple(admitted.get("pod_uids", ())) != value.pod_uids
        or tuple(admitted.get("container_image_ids", ()))
        != value.container_image_ids
        or tuple(admitted.get("node_names", ())) != value.node_names
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
    phase: Literal["capacity", "training", "qualification"],
    prerequisite_authority_receipt_sha256: str,
    job_contract_sha256: str,
    pod_contract_sha256: str,
    run_id: str,
    source_tree_root_sha256: str,
    admission: M03RV16AdmittedJobAuthority,
    admission_file_sha256: str,
) -> M03RV16PhaseLaunchAuthority:
    _digest("admission_file_sha256", admission_file_sha256)
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
        pod_uids=admission.pod_uids,
        completions={"capacity": 1, "training": 3, "qualification": 3}[phase],
        one_shot_nonce_sha256=nonce,
        source_tree_root_sha256=source_tree_root_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
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
        row["pod_uids"] = tuple(row["pod_uids"])
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
    static_result_path: str | Path,
    expected_static_result_file_sha256: str,
    capacity_terminal_path: str | Path,
    expected_capacity_terminal_file_sha256: str,
) -> M03RV16TrainingActivation:
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
        or static_result.get("execution_authorization_receipt_sha256")
        != authorization.receipt_sha256
        or static_result.get("source_tree_root_sha256")
        != value.source_tree_root_sha256
        or static_result.get("training_performed") is not False
        or static_result.get("gpu_mask") != "none"
        or static_result.get("gpu_requests") != 0
        or static_result.get("gpu_limits") != 0
        or static_result.get("unmasked_visibility_claimed") is not False
        or static_result.get("initial_state_strict_loaded_all_settings") is not True
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
        != semantic_sha256(capacity_terminal["capacity"])
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


def load_m03r_v16_qualification_activation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    training_panel_path: str | Path,
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
    panel = _read_exact(Path(training_panel_path), value.training_panel_file_sha256)
    panel_unsigned = {key: row for key, row in panel.items() if key != "receipt_sha256"}
    terminals = tuple(
        _read_exact(path, expected)
        for path, expected in zip(
            training_terminal_paths,
            value.training_terminal_file_sha256,
            strict=True,
        )
    )
    if (
        panel.get("receipt_sha256") != value.training_panel_receipt_sha256
        or value.training_panel_receipt_sha256 != semantic_sha256(panel_unsigned)
        or panel.get("package_plan_sha256") != package.package_plan_sha256
        or panel.get("execution_authorization_receipt_sha256")
        != authorization.receipt_sha256
        or panel.get("source_tree_root_sha256") != value.source_tree_root_sha256
        or panel.get("outer_qualification_authorized") is not True
        or tuple(
            tuple(row) for row in panel.get(
                "setting_fold_adequacy_receipt_sha256", ()
            )
        ) != value.setting_fold_training_adequacy_receipt_sha256
        or tuple(
            tuple(row) for row in panel.get("terminal_checkpoint_file_sha256", ())
        ) != value.terminal_checkpoint_file_sha256
        or panel.get("prequalification_closure_receipt_sha256")
        != value.prequalification_closure_receipt_sha256
        or any(
            terminal.get("package_plan_sha256") != package.package_plan_sha256
            or terminal.get("authorization_receipt_sha256")
            != authorization.receipt_sha256
            or terminal.get("source_tree_root_sha256")
            != value.source_tree_root_sha256
            for terminal in terminals
        )
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


def _issue_m03r_v16_qualification_activation_from_panel(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    training_panel_receipt_sha256: str,
    training_panel_file_sha256: str,
    training_terminal_file_sha256: tuple[str, str, str],
    setting_fold_training_adequacy_receipt_sha256: tuple[
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
    ],
    terminal_checkpoint_file_sha256: tuple[
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
    ],
    prequalification_closure_receipt_sha256: str,
    source_tree_root_sha256: str,
) -> M03RV16QualificationActivation:
    value = M03RV16QualificationActivation(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        training_panel_receipt_sha256=training_panel_receipt_sha256,
        training_panel_file_sha256=training_panel_file_sha256,
        training_terminal_file_sha256=training_terminal_file_sha256,
        setting_fold_training_adequacy_receipt_sha256=(
            setting_fold_training_adequacy_receipt_sha256
        ),
        terminal_checkpoint_file_sha256=terminal_checkpoint_file_sha256,
        prequalification_closure_receipt_sha256=(
            prequalification_closure_receipt_sha256
        ),
        source_tree_root_sha256=source_tree_root_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        _issuer=_QUALIFICATION_ISSUER,
    )
    value.validate_for(package, authorization)
    return value


__all__ = [
    "M03R_V16_ADMITTED_JOB_SCHEMA",
    "M03R_V16_PHASE_LAUNCH_SCHEMA",
    "M03R_V16_QUALIFICATION_ACTIVATION_SCHEMA",
    "M03R_V16_TRAINING_ACTIVATION_SCHEMA",
    "M03RV16ActivationError",
    "M03RV16AdmittedJobAuthority",
    "M03RV16PhaseLaunchAuthority",
    "M03RV16QualificationActivation",
    "M03RV16TrainingActivation",
    "admitted_job_authority_file_sha256",
    "load_m03r_v16_admitted_job_authority",
    "load_m03r_v16_phase_launch_authority",
    "load_m03r_v16_qualification_activation",
    "load_m03r_v16_training_activation",
    "phase_launch_authority_file_sha256",
    "write_m03r_v16_admitted_job_authority",
    "write_m03r_v16_phase_launch_authority",
    "write_m03r_v16_qualification_activation",
    "write_m03r_v16_training_activation",
]
