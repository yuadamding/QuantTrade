"""Immutable package and authorization for the M03R-v11 a15 audit."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan import (
    M03R_V11_A15_PARENT_RUN_ID,
    M03RV11A15InferenceAuditPlan,
    load_m03r_v11_a15_inference_audit_plan,
)

M03R_V11_A15_AUDIT_RUN_ID = "qt-m03r-v11-a15-inference-audit-s17-20260813-a05"
M03R_V11_A15_AUDIT_JOB_NAME = "qt-m03r-v11-a15-audit-a05"
M03R_V11_A15_AUDIT_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-package-v1"
)
M03R_V11_A15_AUDIT_PACKAGE_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-package-file-v1"
)
M03R_V11_A15_AUDIT_AUTHORIZATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-authorization-v1"
)
M03R_V11_A15_AUDIT_SOURCE_PYTHONPATH = "/mnt/audit-package/source/src"
M03R_V11_A15_AUDIT_RUNTIME_ENTRYPOINT = (
    "rl_quant.workflows.top2000_m03r_v11_a15_inference_audit"
)
M03R_V11_A15_AUDIT_PVC_TRAINING_SUBPATH = "home/bcb/yding4/quant/training"
M03R_V11_A15_AUDIT_PARENT_PACKAGE_MOUNT = "/mnt/package"
M03R_V11_A15_AUDIT_PARENT_OUTPUT_MOUNT = "/mnt/parent-output"
M03R_V11_A15_AUDIT_PACKAGE_MOUNT = "/mnt/audit-package"
M03R_V11_A15_AUDIT_OUTPUT_MOUNT = "/mnt/output"

_MAX_JSON_BYTES = 4 * 1024 * 1024


class M03RV11A15InferenceAuditPackageError(ValueError):
    """The audit source, plan, parent lineage, or authorization drifted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11A15InferenceAuditPackageError(
            f"{name} must be one lowercase SHA-256"
        )
    return value


def _relative(name: str, value: str) -> str:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or "." in pure.parts
        or ".." in pure.parts
        or str(pure) != value
    ):
        raise M03RV11A15InferenceAuditPackageError(
            f"{name} must be one normalized relative path"
        )
    return value


def _read_json(path: str | Path, expected_file_sha256: str) -> dict[str, Any]:
    source = Path(path)
    _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV11A15InferenceAuditPackageError(
            "audit package input is not a readable regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_JSON_BYTES
        ):
            raise M03RV11A15InferenceAuditPackageError(
                "audit package input size or type is invalid"
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV11A15InferenceAuditPackageError(
                "audit package input changed while reading"
            )
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV11A15InferenceAuditPackageError(
                "audit package input file hash drifted"
            )
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M03RV11A15InferenceAuditPackageError(
            "audit package input is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise M03RV11A15InferenceAuditPackageError(
            "audit package input must be an object"
        )
    return dict(value)


@dataclass(frozen=True, slots=True)
class M03RV11A15InferenceAuditPackageArtifacts:
    source_archive_sha256: str
    source_manifest_sha256: str
    dependency_lock_sha256: str
    worker_source_sha256: str
    audit_plan_file_sha256: str
    audit_plan_receipt_sha256: str
    parent_package_plan_file_sha256: str
    parent_package_plan_sha256: str
    parent_execution_authorization_file_sha256: str
    parent_execution_authorization_receipt_sha256: str
    parent_source_archive_sha256: str
    parent_terminal_evidence_file_sha256: str
    parent_cleanup_receipt_file_sha256: str
    parent_cleanup_receipt_sha256: str
    image_reference: str
    image_digest_sha256: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "image_reference":
                continue
            _digest(name, value)
        expected_image = "@sha256:" + self.image_digest_sha256
        if not self.image_reference.endswith(expected_image):
            raise M03RV11A15InferenceAuditPackageError(
                "audit image must be digest pinned"
            )


@dataclass(frozen=True, slots=True)
class M03RV11A15InferenceAuditPackagePlan:
    artifacts: M03RV11A15InferenceAuditPackageArtifacts
    run_id: str
    job_name: str
    package_plan_sha256: str
    parent_run_id: str = M03R_V11_A15_PARENT_RUN_ID
    audit_plan_path: str = "/mnt/audit-package/plans/audit-plan.json"
    parent_package_plan_path: str = "/mnt/package/plans/package-plan.json"
    parent_authorization_path: str = "/mnt/package/plans/execution-authorization.json"
    parent_output_root: str = M03R_V11_A15_AUDIT_PARENT_OUTPUT_MOUNT
    parent_lifecycle_root: str = "/mnt/audit-package/parent-lifecycle"
    output_root: str = M03R_V11_A15_AUDIT_OUTPUT_MOUNT
    source_pythonpath: str = M03R_V11_A15_AUDIT_SOURCE_PYTHONPATH
    indexed_completions: int = 2
    parallelism: int = 2
    h100s_per_completion: int = 1
    maximum_h100_requests: int = 2
    training_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    inference_audit_authorized: bool = True
    economic_training_authorized: bool = False
    economic_generation_may_be_minted: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V11_A15_AUDIT_PACKAGE_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "package_plan_sha256"
        }

    def validate(self, audit: M03RV11A15InferenceAuditPlan) -> None:
        self.artifacts.validate()
        audit.validate()
        if (
            self.run_id != M03R_V11_A15_AUDIT_RUN_ID
            or self.job_name != M03R_V11_A15_AUDIT_JOB_NAME
            or self.parent_run_id != audit.parent_run_id
            or self.artifacts.audit_plan_receipt_sha256 != audit.receipt_sha256
            or self.artifacts.parent_package_plan_file_sha256
            != audit.parent_package_plan_file_sha256
            or self.artifacts.parent_package_plan_sha256
            != audit.parent_package_plan_sha256
            or self.artifacts.parent_execution_authorization_file_sha256
            != audit.parent_execution_authorization_file_sha256
            or self.artifacts.parent_execution_authorization_receipt_sha256
            != audit.parent_execution_authorization_receipt_sha256
            or self.artifacts.parent_source_archive_sha256
            != audit.parent_source_archive_sha256
            or self.artifacts.parent_terminal_evidence_file_sha256
            != audit.parent_terminal_evidence_file_sha256
            or self.artifacts.parent_cleanup_receipt_file_sha256
            != audit.parent_cleanup_receipt_file_sha256
            or self.artifacts.parent_cleanup_receipt_sha256
            != audit.parent_cleanup_receipt_sha256
            or self.artifacts.image_reference != audit.parent_image_reference
            or self.indexed_completions != 2
            or self.parallelism != 2
            or self.h100s_per_completion != 1
            or self.maximum_h100_requests != 2
            or self.parallelism * self.h100s_per_completion
            != self.maximum_h100_requests
            or self.training_authorized
            or self.checkpoint_selection_authorized
            or not self.inference_audit_authorized
            or self.economic_training_authorized
            or self.economic_generation_may_be_minted
            or self.outer_2026_access_authorized
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V11_A15_AUDIT_PACKAGE_SCHEMA
            or self.package_plan_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditPackageError("a15 audit package plan drifted")


@dataclass(frozen=True, slots=True)
class M03RV11A15InferenceAuditAuthorization:
    package_plan_sha256: str
    package_plan_file_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    worker_source_sha256: str
    audit_plan_file_sha256: str
    audit_plan_receipt_sha256: str
    parent_cleanup_receipt_sha256: str
    image_reference: str
    receipt_sha256: str
    runtime_entrypoint: str = M03R_V11_A15_AUDIT_RUNTIME_ENTRYPOINT
    indexed_completions: int = 2
    parallelism: int = 2
    h100s_per_completion: int = 1
    maximum_h100_requests: int = 2
    training_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    inference_audit_authorized: bool = True
    economic_training_authorized: bool = False
    economic_generation_may_be_minted: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V11_A15_AUDIT_AUTHORIZATION_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(
        self,
        package: M03RV11A15InferenceAuditPackagePlan,
        audit: M03RV11A15InferenceAuditPlan,
    ) -> None:
        package.validate(audit)
        for name in (
            "package_plan_sha256",
            "package_plan_file_sha256",
            "source_archive_sha256",
            "source_manifest_sha256",
            "worker_source_sha256",
            "audit_plan_file_sha256",
            "audit_plan_receipt_sha256",
            "parent_cleanup_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.package_plan_sha256 != package.package_plan_sha256
            or self.source_archive_sha256 != package.artifacts.source_archive_sha256
            or self.source_manifest_sha256 != package.artifacts.source_manifest_sha256
            or self.worker_source_sha256 != package.artifacts.worker_source_sha256
            or self.audit_plan_file_sha256 != package.artifacts.audit_plan_file_sha256
            or self.audit_plan_receipt_sha256 != audit.receipt_sha256
            or self.parent_cleanup_receipt_sha256 != audit.parent_cleanup_receipt_sha256
            or self.image_reference != package.artifacts.image_reference
            or self.runtime_entrypoint != M03R_V11_A15_AUDIT_RUNTIME_ENTRYPOINT
            or self.indexed_completions != package.indexed_completions
            or self.parallelism != package.parallelism
            or self.h100s_per_completion != package.h100s_per_completion
            or self.maximum_h100_requests != package.maximum_h100_requests
            or self.training_authorized
            or self.checkpoint_selection_authorized
            or not self.inference_audit_authorized
            or self.economic_training_authorized
            or self.economic_generation_may_be_minted
            or self.outer_2026_access_authorized
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V11_A15_AUDIT_AUTHORIZATION_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditPackageError(
                "a15 audit authorization drifted"
            )


def build_m03r_v11_a15_inference_audit_package_plan(
    artifacts: M03RV11A15InferenceAuditPackageArtifacts,
    audit: M03RV11A15InferenceAuditPlan,
) -> M03RV11A15InferenceAuditPackagePlan:
    provisional = M03RV11A15InferenceAuditPackagePlan(
        artifacts=artifacts,
        run_id=M03R_V11_A15_AUDIT_RUN_ID,
        job_name=M03R_V11_A15_AUDIT_JOB_NAME,
        package_plan_sha256="0" * 64,
    )
    plan = replace(
        provisional,
        package_plan_sha256=_sha256(provisional.unsigned_payload()),
    )
    plan.validate(audit)
    return plan


def build_m03r_v11_a15_inference_audit_authorization(
    package: M03RV11A15InferenceAuditPackagePlan,
    audit: M03RV11A15InferenceAuditPlan,
    *,
    package_plan_file_sha256: str,
) -> M03RV11A15InferenceAuditAuthorization:
    provisional = M03RV11A15InferenceAuditAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=_digest(
            "package_plan_file_sha256", package_plan_file_sha256
        ),
        source_archive_sha256=package.artifacts.source_archive_sha256,
        source_manifest_sha256=package.artifacts.source_manifest_sha256,
        worker_source_sha256=package.artifacts.worker_source_sha256,
        audit_plan_file_sha256=package.artifacts.audit_plan_file_sha256,
        audit_plan_receipt_sha256=audit.receipt_sha256,
        parent_cleanup_receipt_sha256=audit.parent_cleanup_receipt_sha256,
        image_reference=package.artifacts.image_reference,
        receipt_sha256="0" * 64,
    )
    authorization = replace(
        provisional,
        receipt_sha256=_sha256(provisional.unsigned_payload()),
    )
    authorization.validate(package, audit)
    return authorization


def _write(path: Path, payload: dict[str, Any], *, schema: str) -> str:
    if path.exists() or path.is_symlink():
        raise M03RV11A15InferenceAuditPackageError(
            "audit package target already exists"
        )
    encoded = _canonical({"schema": schema, "payload": payload}) + b"\n"
    path.parent.mkdir(parents=True, mode=0o750, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def write_m03r_v11_a15_inference_audit_package_plan(
    path: str | Path,
    package: M03RV11A15InferenceAuditPackagePlan,
    audit: M03RV11A15InferenceAuditPlan,
) -> str:
    package.validate(audit)
    return _write(
        Path(path),
        asdict(package),
        schema=M03R_V11_A15_AUDIT_PACKAGE_FILE_SCHEMA,
    )


def write_m03r_v11_a15_inference_audit_authorization(
    path: str | Path,
    authorization: M03RV11A15InferenceAuditAuthorization,
    package: M03RV11A15InferenceAuditPackagePlan,
    audit: M03RV11A15InferenceAuditPlan,
) -> str:
    authorization.validate(package, audit)
    return _write(
        Path(path),
        asdict(authorization),
        schema=M03R_V11_A15_AUDIT_AUTHORIZATION_SCHEMA,
    )


def load_m03r_v11_a15_inference_audit_package_plan(
    path: str | Path,
    *,
    expected_file_sha256: str,
    audit: M03RV11A15InferenceAuditPlan,
) -> M03RV11A15InferenceAuditPackagePlan:
    value = _read_json(path, expected_file_sha256)
    payload = value.get("payload")
    if (
        value.get("schema") != M03R_V11_A15_AUDIT_PACKAGE_FILE_SCHEMA
        or not isinstance(payload, dict)
        or not isinstance(payload.get("artifacts"), dict)
    ):
        raise M03RV11A15InferenceAuditPackageError("audit package plan file drifted")
    try:
        package = M03RV11A15InferenceAuditPackagePlan(
            **{
                **payload,
                "artifacts": M03RV11A15InferenceAuditPackageArtifacts(
                    **payload["artifacts"]
                ),
            }
        )
    except (TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditPackageError(
            "audit package plan cannot be decoded"
        ) from exc
    package.validate(audit)
    return package


def load_m03r_v11_a15_inference_audit_authorization(
    path: str | Path,
    *,
    expected_file_sha256: str,
    package: M03RV11A15InferenceAuditPackagePlan,
    audit: M03RV11A15InferenceAuditPlan,
) -> M03RV11A15InferenceAuditAuthorization:
    value = _read_json(path, expected_file_sha256)
    payload = value.get("payload")
    if value.get("schema") != M03R_V11_A15_AUDIT_AUTHORIZATION_SCHEMA or not isinstance(
        payload, dict
    ):
        raise M03RV11A15InferenceAuditPackageError("audit authorization file drifted")
    try:
        authorization = M03RV11A15InferenceAuditAuthorization(**payload)
    except (TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditPackageError(
            "audit authorization cannot be decoded"
        ) from exc
    authorization.validate(package, audit)
    return authorization


def load_m03r_v11_a15_inference_audit_bundle(
    *,
    audit_plan_path: str | Path,
    audit_plan_file_sha256: str,
    package_plan_path: str | Path,
    package_plan_file_sha256: str,
    authorization_path: str | Path,
    authorization_file_sha256: str,
) -> tuple[
    M03RV11A15InferenceAuditPlan,
    M03RV11A15InferenceAuditPackagePlan,
    M03RV11A15InferenceAuditAuthorization,
]:
    audit = load_m03r_v11_a15_inference_audit_plan(
        audit_plan_path,
        expected_file_sha256=audit_plan_file_sha256,
    )
    package = load_m03r_v11_a15_inference_audit_package_plan(
        package_plan_path,
        expected_file_sha256=package_plan_file_sha256,
        audit=audit,
    )
    authorization = load_m03r_v11_a15_inference_audit_authorization(
        authorization_path,
        expected_file_sha256=authorization_file_sha256,
        package=package,
        audit=audit,
    )
    if authorization.package_plan_file_sha256 != package_plan_file_sha256:
        raise M03RV11A15InferenceAuditPackageError(
            "audit authorization and package-plan file disagree"
        )
    return audit, package, authorization


__all__ = [
    "M03R_V11_A15_AUDIT_AUTHORIZATION_SCHEMA",
    "M03R_V11_A15_AUDIT_JOB_NAME",
    "M03R_V11_A15_AUDIT_PACKAGE_FILE_SCHEMA",
    "M03R_V11_A15_AUDIT_PACKAGE_MOUNT",
    "M03R_V11_A15_AUDIT_PACKAGE_SCHEMA",
    "M03R_V11_A15_AUDIT_PARENT_OUTPUT_MOUNT",
    "M03R_V11_A15_AUDIT_PVC_TRAINING_SUBPATH",
    "M03R_V11_A15_AUDIT_RUN_ID",
    "M03R_V11_A15_AUDIT_RUNTIME_ENTRYPOINT",
    "M03R_V11_A15_AUDIT_SOURCE_PYTHONPATH",
    "M03RV11A15InferenceAuditAuthorization",
    "M03RV11A15InferenceAuditPackageArtifacts",
    "M03RV11A15InferenceAuditPackageError",
    "M03RV11A15InferenceAuditPackagePlan",
    "build_m03r_v11_a15_inference_audit_authorization",
    "build_m03r_v11_a15_inference_audit_package_plan",
    "load_m03r_v11_a15_inference_audit_authorization",
    "load_m03r_v11_a15_inference_audit_bundle",
    "load_m03r_v11_a15_inference_audit_package_plan",
    "write_m03r_v11_a15_inference_audit_authorization",
    "write_m03r_v11_a15_inference_audit_package_plan",
]
