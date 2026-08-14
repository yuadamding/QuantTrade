"""Content-bound source-only package for the M03R-v12 post-hoc audit."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v12_posthoc_inference_audit import (
    M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_SETTING_IDS,
)

M03R_V12_POSTHOC_AUDIT_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-package-v1"
)
M03R_V12_POSTHOC_AUDIT_PACKAGE_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-package-file-v1"
)
M03R_V12_POSTHOC_AUDIT_AUTHORIZATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-authorization-v1"
)
M03R_V12_POSTHOC_AUDIT_RUNTIME_ENTRYPOINT = (
    "rl_quant.workflows.top2000_m03r_v12_posthoc_inference_audit"
)
M03R_V12_POSTHOC_AUDIT_SOURCE_PYTHONPATH = "/mnt/audit-package/source/src"
M03R_V12_POSTHOC_AUDIT_PARENT_PACKAGE_PLAN = (
    "/mnt/parent-package/plans/package-plan.json"
)
M03R_V12_POSTHOC_AUDIT_PARENT_OUTPUT_ROOT = "/mnt/parent-output"
M03R_V12_POSTHOC_AUDIT_OUTPUT_ROOT = "/mnt/output"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 8 * 1024 * 1024


class M03RV12PosthocAuditPackageError(ValueError):
    """The audit source, parent evidence, or authorization drifted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(name: str, value: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise M03RV12PosthocAuditPackageError(
            f"{name} must be one lowercase SHA-256"
        )
    return value


def _relative(name: str, value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise M03RV12PosthocAuditPackageError(
            f"{name} must be one safe relative path"
        )
    return value


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditCheckpointBinding:
    setting_index: int
    setting_id: str
    fold_index: int
    checkpoint_relative_path: str
    checkpoint_file_sha256: str
    model_state_sha256: str
    training_residual_operator_root_sha256: str
    training_source_array_sha256: str
    parent_fold_terminal_relative_path: str
    parent_fold_terminal_file_sha256: str
    parent_fold_terminal_receipt_sha256: str

    def validate(self) -> None:
        if (
            self.setting_index not in range(3)
            or self.setting_id != M03R_V12_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.checkpoint_relative_path
            != (
                f"completion-{self.setting_index:02d}-setting-{self.setting_index:02d}"
                f"/checkpoints/fold-{self.fold_index:02d}-horizon-03-update-0064.pt"
            )
            or self.parent_fold_terminal_relative_path
            != (
                f"completion-{self.setting_index:02d}-setting-{self.setting_index:02d}"
                f"/receipts/fold-{self.fold_index:02d}-terminal.json"
            )
        ):
            raise M03RV12PosthocAuditPackageError(
                "v12 post-hoc checkpoint binding drifted"
            )
        for name in (
            "checkpoint_file_sha256",
            "model_state_sha256",
            "training_residual_operator_root_sha256",
            "training_source_array_sha256",
            "parent_fold_terminal_file_sha256",
            "parent_fold_terminal_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        _relative("checkpoint_relative_path", self.checkpoint_relative_path)
        _relative(
            "parent_fold_terminal_relative_path",
            self.parent_fold_terminal_relative_path,
        )


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditParentBinding:
    run_id: str
    package_plan_sha256: str
    package_plan_file_sha256: str
    source_archive_sha256: str
    parent_protocol_sha256: str
    checkpoint_bindings: tuple[M03RV12PosthocAuditCheckpointBinding, ...]
    predictive_terminal_relative_paths: tuple[str, ...]
    predictive_terminal_file_sha256: tuple[str, ...]
    predictive_terminal_receipt_sha256: tuple[str, ...]

    def validate(self) -> None:
        if (
            self.run_id != "qt-m03r-v12-h3-predictive-s17-20260813-a05"
            or self.parent_protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or len(self.checkpoint_bindings) != 18
            or tuple(
                (row.setting_index, row.fold_index) for row in self.checkpoint_bindings
            )
            != tuple((setting, fold) for setting in range(3) for fold in range(6))
            or self.predictive_terminal_relative_paths
            != tuple(
                f"completion-{setting:02d}-setting-{setting:02d}/predictive-terminal.json"
                for setting in range(3)
            )
            or len(self.predictive_terminal_file_sha256) != 3
            or len(self.predictive_terminal_receipt_sha256) != 3
        ):
            raise M03RV12PosthocAuditPackageError(
                "v12 post-hoc parent binding drifted"
            )
        for row in self.checkpoint_bindings:
            row.validate()
        for name, value in (
            ("package_plan_sha256", self.package_plan_sha256),
            ("package_plan_file_sha256", self.package_plan_file_sha256),
            ("source_archive_sha256", self.source_archive_sha256),
            ("parent_protocol_sha256", self.parent_protocol_sha256),
            *(
                ("predictive_terminal_file_sha256", value)
                for value in self.predictive_terminal_file_sha256
            ),
            *(
                ("predictive_terminal_receipt_sha256", value)
                for value in self.predictive_terminal_receipt_sha256
            ),
        ):
            _digest(name, value)
        for value in self.predictive_terminal_relative_paths:
            _relative("predictive_terminal_relative_path", value)


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditSourceArtifacts:
    source_archive_sha256: str
    source_manifest_file_sha256: str
    source_inventory_sha256: str
    dependency_lock_sha256: str
    worker_source_sha256: str
    image_reference: str
    image_digest_sha256: str

    def validate(self) -> None:
        for name in (
            "source_archive_sha256",
            "source_manifest_file_sha256",
            "source_inventory_sha256",
            "dependency_lock_sha256",
            "worker_source_sha256",
            "image_digest_sha256",
        ):
            _digest(name, getattr(self, name))
        if not self.image_reference.endswith(
            f"@sha256:{self.image_digest_sha256}"
        ):
            raise M03RV12PosthocAuditPackageError(
                "v12 post-hoc image reference is not digest pinned"
            )


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditPackagePlan:
    artifacts: M03RV12PosthocAuditSourceArtifacts
    parent: M03RV12PosthocAuditParentBinding
    package_plan_sha256: str
    runtime_entrypoint: str = M03R_V12_POSTHOC_AUDIT_RUNTIME_ENTRYPOINT
    source_pythonpath: str = M03R_V12_POSTHOC_AUDIT_SOURCE_PYTHONPATH
    parent_package_plan_path: str = M03R_V12_POSTHOC_AUDIT_PARENT_PACKAGE_PLAN
    parent_output_root: str = M03R_V12_POSTHOC_AUDIT_PARENT_OUTPUT_ROOT
    output_root: str = M03R_V12_POSTHOC_AUDIT_OUTPUT_ROOT
    indexed_completions: int = 3
    parallelism: int = 3
    h100s_per_completion: int = 1
    maximum_h100_requests: int = 3
    inference_audit_authorized: bool = True
    training_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    economic_generation_may_be_minted: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V12_POSTHOC_AUDIT_PACKAGE_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "artifacts": asdict(self.artifacts),
            "parent": asdict(self.parent),
            "runtime_entrypoint": self.runtime_entrypoint,
            "source_pythonpath": self.source_pythonpath,
            "parent_package_plan_path": self.parent_package_plan_path,
            "parent_output_root": self.parent_output_root,
            "output_root": self.output_root,
            "indexed_completions": self.indexed_completions,
            "parallelism": self.parallelism,
            "h100s_per_completion": self.h100s_per_completion,
            "maximum_h100_requests": self.maximum_h100_requests,
            "inference_audit_authorized": self.inference_audit_authorized,
            "training_authorized": self.training_authorized,
            "checkpoint_selection_authorized": self.checkpoint_selection_authorized,
            "economic_generation_may_be_minted": (
                self.economic_generation_may_be_minted
            ),
            "outer_2026_access_authorized": self.outer_2026_access_authorized,
            "development_only": self.development_only,
            "reportable": self.reportable,
            "promotion_eligible": self.promotion_eligible,
        }

    def validate(self) -> None:
        self.artifacts.validate()
        self.parent.validate()
        if (
            self.runtime_entrypoint != M03R_V12_POSTHOC_AUDIT_RUNTIME_ENTRYPOINT
            or self.source_pythonpath != M03R_V12_POSTHOC_AUDIT_SOURCE_PYTHONPATH
            or self.parent_package_plan_path
            != M03R_V12_POSTHOC_AUDIT_PARENT_PACKAGE_PLAN
            or self.parent_output_root != M03R_V12_POSTHOC_AUDIT_PARENT_OUTPUT_ROOT
            or self.output_root != M03R_V12_POSTHOC_AUDIT_OUTPUT_ROOT
            or self.indexed_completions != 3
            or self.parallelism != 3
            or self.h100s_per_completion != 1
            or self.maximum_h100_requests != 3
            or self.parallelism * self.h100s_per_completion
            != self.maximum_h100_requests
            or not self.inference_audit_authorized
            or self.training_authorized
            or self.checkpoint_selection_authorized
            or self.economic_generation_may_be_minted
            or self.outer_2026_access_authorized
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V12_POSTHOC_AUDIT_PACKAGE_SCHEMA
            or self.package_plan_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV12PosthocAuditPackageError(
                "v12 post-hoc package plan drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return self.package_plan_sha256


def build_m03r_v12_posthoc_audit_package_plan(
    artifacts: M03RV12PosthocAuditSourceArtifacts,
    parent: M03RV12PosthocAuditParentBinding,
) -> M03RV12PosthocAuditPackagePlan:
    provisional = M03RV12PosthocAuditPackagePlan(
        artifacts=artifacts,
        parent=parent,
        package_plan_sha256="0" * 64,
    )
    result = replace(
        provisional,
        package_plan_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


def _read_package_file(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise M03RV12PosthocAuditPackageError(
            "v12 post-hoc package file is unavailable"
        ) from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or not 0 < status.st_size <= _MAX_JSON_BYTES:
            raise M03RV12PosthocAuditPackageError(
                "v12 post-hoc package file is not bounded and regular"
            )
        raw = b""
        while chunk := os.read(descriptor, min(1024 * 1024, status.st_size + 1)):
            raw += chunk
            if len(raw) > _MAX_JSON_BYTES:
                raise M03RV12PosthocAuditPackageError(
                    "v12 post-hoc package file exceeds its bound"
                )
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise M03RV12PosthocAuditPackageError(
            "v12 post-hoc package file hash drifted"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M03RV12PosthocAuditPackageError(
            "v12 post-hoc package file is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise M03RV12PosthocAuditPackageError(
            "v12 post-hoc package file is not an object"
        )
    return value


def write_m03r_v12_posthoc_audit_package_plan(
    path: str | Path,
    plan: M03RV12PosthocAuditPackagePlan,
) -> str:
    plan.validate()
    target = Path(path)
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    payload = {
        "schema": M03R_V12_POSTHOC_AUDIT_PACKAGE_FILE_SCHEMA,
        "plan": asdict(plan),
        "receipt_sha256": plan.receipt_sha256,
    }
    raw = _canonical(payload) + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def load_m03r_v12_posthoc_audit_package_plan(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> M03RV12PosthocAuditPackagePlan:
    value = _read_package_file(Path(path), expected_file_sha256)
    plan = value.get("plan")
    if value.get("schema") != M03R_V12_POSTHOC_AUDIT_PACKAGE_FILE_SCHEMA or not isinstance(
        plan, dict
    ):
        raise M03RV12PosthocAuditPackageError(
            "v12 post-hoc package wrapper drifted"
        )
    artifacts = plan.get("artifacts")
    parent = plan.get("parent")
    if not isinstance(artifacts, dict) or not isinstance(parent, dict):
        raise M03RV12PosthocAuditPackageError(
            "v12 post-hoc package lacks typed bindings"
        )
    checkpoint_rows = parent.get("checkpoint_bindings")
    if not isinstance(checkpoint_rows, list):
        raise M03RV12PosthocAuditPackageError(
            "v12 post-hoc package checkpoint inventory drifted"
        )
    result = M03RV12PosthocAuditPackagePlan(
        **{
            **plan,
            "artifacts": M03RV12PosthocAuditSourceArtifacts(**artifacts),
            "parent": M03RV12PosthocAuditParentBinding(
                **{
                    **parent,
                    "checkpoint_bindings": tuple(
                        M03RV12PosthocAuditCheckpointBinding(**row)
                        for row in checkpoint_rows
                    ),
                    "predictive_terminal_relative_paths": tuple(
                        parent["predictive_terminal_relative_paths"]
                    ),
                    "predictive_terminal_file_sha256": tuple(
                        parent["predictive_terminal_file_sha256"]
                    ),
                    "predictive_terminal_receipt_sha256": tuple(
                        parent["predictive_terminal_receipt_sha256"]
                    ),
                }
            ),
        }
    )
    result.validate()
    if value.get("receipt_sha256") != result.receipt_sha256:
        raise M03RV12PosthocAuditPackageError(
            "v12 post-hoc package wrapper receipt drifted"
        )
    return result


__all__ = [
    "M03R_V12_POSTHOC_AUDIT_AUTHORIZATION_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_PACKAGE_FILE_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_PACKAGE_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_RUNTIME_ENTRYPOINT",
    "M03RV12PosthocAuditCheckpointBinding",
    "M03RV12PosthocAuditPackageError",
    "M03RV12PosthocAuditPackagePlan",
    "M03RV12PosthocAuditParentBinding",
    "M03RV12PosthocAuditSourceArtifacts",
    "build_m03r_v12_posthoc_audit_package_plan",
    "load_m03r_v12_posthoc_audit_package_plan",
    "write_m03r_v12_posthoc_audit_package_plan",
]
