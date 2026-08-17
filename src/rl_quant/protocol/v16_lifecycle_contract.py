"""Torch-free file and identity checks for the M03R-v16 init gate.

This module is intentionally standard-library only.  The init container imports
it before the scientific worker is allowed to start, so importing this module
must never pull model, optimizer, market-data, or PyTorch code into the process.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_MAX_BYTES = 4 * 1024**2
_HEX = frozenset("0123456789abcdef")
_PACKAGE_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-predictive-package-file-v2"
)
_AUTHORIZATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-package-stage-authorization-v2"
)
_ADMISSION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-prelaunch-job-authority-v2"
)
_LAUNCH_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-phase-launch-authority-v4"
)
_ATTESTATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-pod-runtime-attestation-v4"
)
_STORAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-storage-semantics-evidence-v3"
)
_DRY_RUN_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-server-dry-run-result-v1"
)
_ADMITTED_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-admitted-manifest-result-v1"
)


class V16LifecycleContractError(ValueError):
    """A lightweight lifecycle file was absent, malformed, or mismatched."""


def canonical_json_payload(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_json_file_bytes(value: Any) -> bytes:
    return canonical_json_payload(value) + b"\n"


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_payload(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise V16LifecycleContractError(f"{name} must be a lowercase SHA-256")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise V16LifecycleContractError(f"{name} is absent")
    return value


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V16LifecycleContractError(f"{name} is not an object")
    return dict(value)


def _read_exact(path: str | Path, expected_file_sha256: str) -> dict[str, Any]:
    _digest("expected_file_sha256", expected_file_sha256)
    candidate = Path(path)
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise V16LifecycleContractError(
            "V16 lifecycle authority file is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_BYTES:
            raise V16LifecycleContractError(
                "V16 lifecycle authority file type or size drifted"
            )
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise V16LifecycleContractError(
                "V16 lifecycle authority changed while read"
            )
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise V16LifecycleContractError("V16 lifecycle authority file hash drifted")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V16LifecycleContractError(
            "V16 lifecycle authority file is malformed"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise V16LifecycleContractError(
            "V16 lifecycle authority file is not canonical"
        )
    return payload


def _receipt_payload(
    path: str | Path,
    expected_file_sha256: str,
    *,
    value_key: str,
    expected_receipt_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    payload = _read_exact(path, expected_file_sha256)
    value = _mapping(value_key, payload.get(value_key))
    receipt = _digest("receipt_sha256", payload.get("receipt_sha256"))
    if receipt != semantic_sha256(value) or (
        expected_receipt_sha256 is not None
        and receipt != _digest(
            "expected_receipt_sha256", expected_receipt_sha256
        )
    ):
        raise V16LifecycleContractError("V16 lifecycle receipt drifted")
    return value, receipt


def _image_identity(value: object) -> tuple[str | None, str]:
    text = _text("runtime image", value).removeprefix("docker-pullable://")
    if "@sha256:" in text:
        repository, digest = text.rsplit("@sha256:", 1)
        if not repository:
            raise V16LifecycleContractError("V16 runtime image repository is absent")
    elif text.startswith(("containerd://sha256:", "docker://sha256:")):
        repository = None
        digest = text.rsplit("sha256:", 1)[1]
    else:
        raise V16LifecycleContractError("V16 runtime image is not digest pinned")
    return repository, _digest("runtime image digest", digest)


@dataclass(frozen=True, slots=True)
class V16LifecyclePackageView:
    package_plan_sha256: str
    protocol_sha256: str
    source_pythonpath: str
    image_reference: str
    image_digest_sha256: str
    worker_output_roots: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class V16LifecycleAuthorizationView:
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class V16LifecycleAdmissionView:
    receipt_sha256: str
    job_uid: str
    run_id: str
    job_name: str


@dataclass(frozen=True, slots=True)
class V16LifecycleStorageView:
    receipt_sha256: str
    file_sha256: str
    authority_root_sha256: str
    observer_root_sha256: str


@dataclass(frozen=True, slots=True)
class V16LifecycleLaunchView:
    receipt_sha256: str
    phase: Literal[
        "capacity", "training", "qualification-preflight", "qualification"
    ]
    job_uid: str
    completions: int
    relative_path_template: str
    storage_semantics_receipt_sha256: str
    storage_semantics_file_sha256: str

    def relative_path(self, completion_index: int) -> str:
        if not 0 <= completion_index < self.completions:
            raise V16LifecycleContractError(
                "V16 completion index is outside the launch authority"
            )
        return self.relative_path_template.format(
            completion_index=completion_index
        )


@dataclass(frozen=True, slots=True)
class V16LifecycleAttestationView:
    receipt_sha256: str
    relative_path: str
    pod_uid: str
    pod_name: str
    node_name: str
    observed_owner_job_uid: str
    observed_owner_job_name: str
    observed_completion_index: int
    observed_pod_resource_version: str


def load_v16_lifecycle_package(
    path: str | Path, expected_file_sha256: str
) -> V16LifecyclePackageView:
    payload = _read_exact(path, expected_file_sha256)
    plan = _mapping("plan", payload.get("plan"))
    package_sha = _digest("package_plan_sha256", plan.get("package_plan_sha256"))
    unsigned = {key: value for key, value in plan.items() if key != "package_plan_sha256"}
    artifacts = _mapping("plan.artifacts", plan.get("artifacts"))
    panel = _mapping("plan.panel", plan.get("panel"))
    workers = panel.get("workers")
    if not isinstance(workers, list) or len(workers) != 3:
        raise V16LifecycleContractError("V16 package worker inventory drifted")
    worker_output_roots = tuple(
        _text("worker.output_root", _mapping("worker", row).get("output_root"))
        for row in workers
    )
    image_reference = _text("image_reference", artifacts.get("image_reference"))
    repository, image_digest = _image_identity(image_reference)
    if repository is None:
        raise V16LifecycleContractError("V16 package image repository is absent")
    if (
        payload.get("schema") != _PACKAGE_FILE_SCHEMA
        or payload.get("package_plan_sha256") != package_sha
        or semantic_sha256(unsigned) != package_sha
        or plan.get("structural_gates_complete") is not True
        or plan.get("package_authorized") is not False
        or plan.get("kubernetes_launch_authorized") is not False
        or plan.get("outer_2026_access_authorized") is not False
        or plan.get("economic_panel_authorized") is not False
        or plan.get("development_only") is not True
        or plan.get("reportable") is not False
        or plan.get("promotion_eligible") is not False
        or artifacts.get("image_digest_sha256") != image_digest
    ):
        raise V16LifecycleContractError("V16 lightweight package identity drifted")
    return V16LifecyclePackageView(
        package_plan_sha256=package_sha,
        protocol_sha256=_digest("protocol_sha256", plan.get("protocol_sha256")),
        source_pythonpath=_text("source_pythonpath", plan.get("source_pythonpath")),
        image_reference=image_reference,
        image_digest_sha256=image_digest,
        worker_output_roots=worker_output_roots,  # type: ignore[arg-type]
    )


def load_v16_lifecycle_authorization(
    path: str | Path,
    expected_file_sha256: str,
    *,
    package: V16LifecyclePackageView,
) -> V16LifecycleAuthorizationView:
    value, receipt = _receipt_payload(
        path, expected_file_sha256, value_key="authorization"
    )
    if (
        value.get("schema") != _AUTHORIZATION_SCHEMA
        or value.get("package_plan_sha256") != package.package_plan_sha256
        or value.get("image_reference") != package.image_reference
        or value.get("protocol_sha256") != package.protocol_sha256
        or value.get("static_validation_authorized") is not True
        or value.get("capacity_qualification_authorized") is not False
        or value.get("predictive_training_authorized") is not False
        or value.get("outer_qualification_authorized") is not False
        or value.get("economic_training_authorized") is not False
        or value.get("reinforcement_learning_authorized") is not False
        or value.get("outer_2026_access_authorized") is not False
    ):
        raise V16LifecycleContractError(
            "V16 lightweight execution authorization drifted"
        )
    return V16LifecycleAuthorizationView(receipt_sha256=receipt)


def load_v16_lifecycle_admission(
    path: str | Path,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    *,
    package: V16LifecyclePackageView,
    authorization: V16LifecycleAuthorizationView,
    expected_phase: str,
    expected_job_contract_sha256: str,
    expected_pod_contract_sha256: str,
    server_side_dry_run_path: str | Path,
    admitted_manifest_path: str | Path,
) -> V16LifecycleAdmissionView:
    value, receipt = _receipt_payload(
        path,
        expected_file_sha256,
        value_key="authority",
        expected_receipt_sha256=expected_receipt_sha256,
    )
    expected_completions = {
        "capacity": 1,
        "training": 3,
        "qualification-preflight": 3,
        "qualification": 3,
    }
    if (
        value.get("schema") != _ADMISSION_SCHEMA
        or value.get("package_plan_sha256") != package.package_plan_sha256
        or value.get("execution_authorization_receipt_sha256")
        != authorization.receipt_sha256
        or value.get("phase") != expected_phase
        or value.get("job_contract_sha256") != expected_job_contract_sha256
        or value.get("pod_contract_sha256") != expected_pod_contract_sha256
        or value.get("completions") != expected_completions.get(expected_phase)
        or value.get("image_reference") != package.image_reference
        or value.get("image_digest_sha256") != package.image_digest_sha256
        or value.get("suspended_at_admission") is not True
        or value.get("economic_training_authorized") is not False
        or value.get("reinforcement_learning_authorized") is not False
        or value.get("outer_2026_access_authorized") is not False
        or value.get("protocol_sha256") != package.protocol_sha256
    ):
        raise V16LifecycleContractError("V16 lightweight admission drifted")
    dry_run = _read_exact(
        server_side_dry_run_path,
        _digest(
            "server_side_dry_run_file_sha256",
            value.get("server_side_dry_run_file_sha256"),
        ),
    )
    admitted = _read_exact(
        admitted_manifest_path,
        _digest(
            "admitted_manifest_file_sha256",
            value.get("admitted_manifest_file_sha256"),
        ),
    )
    dry_unsigned = {key: row for key, row in dry_run.items() if key != "receipt_sha256"}
    admitted_unsigned = {
        key: row for key, row in admitted.items() if key != "receipt_sha256"
    }
    if (
        dry_run.get("schema") != _DRY_RUN_SCHEMA
        or dry_run.get("receipt_sha256") != semantic_sha256(dry_unsigned)
        or dry_run.get("receipt_sha256")
        != value.get("server_side_dry_run_receipt_sha256")
        or dry_run.get("package_plan_sha256") != package.package_plan_sha256
        or dry_run.get("phase") != expected_phase
        or dry_run.get("job_contract_sha256") != expected_job_contract_sha256
        or dry_run.get("pod_contract_sha256") != expected_pod_contract_sha256
        or dry_run.get("passed") is not True
        or admitted.get("schema") != _ADMITTED_MANIFEST_SCHEMA
        or admitted.get("receipt_sha256") != semantic_sha256(admitted_unsigned)
        or admitted.get("receipt_sha256") != value.get("admitted_manifest_sha256")
        or admitted.get("package_plan_sha256") != package.package_plan_sha256
        or admitted.get("phase") != expected_phase
        or admitted.get("job_contract_sha256") != expected_job_contract_sha256
        or admitted.get("pod_contract_sha256") != expected_pod_contract_sha256
        or admitted.get("job_uid") != value.get("job_uid")
        or admitted.get("job_name") != value.get("job_name")
        or admitted.get("image_reference") != package.image_reference
        or admitted.get("image_digest_sha256") != package.image_digest_sha256
        or admitted.get("suspended_at_admission") is not True
    ):
        raise V16LifecycleContractError(
            "V16 lightweight admission evidence drifted"
        )
    return V16LifecycleAdmissionView(
        receipt_sha256=receipt,
        job_uid=_text("job_uid", value.get("job_uid")),
        run_id=_text("run_id", value.get("run_id")),
        job_name=_text("job_name", value.get("job_name")),
    )


def load_v16_lifecycle_storage(
    path: str | Path,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    *,
    authority_root: str | Path,
    observer_root: str | Path,
) -> V16LifecycleStorageView:
    value, receipt = _receipt_payload(
        path,
        expected_file_sha256,
        value_key="evidence",
        expected_receipt_sha256=expected_receipt_sha256,
    )
    authority_root_sha256 = semantic_sha256(
        {"resolved_root": str(Path(authority_root).resolve())}
    )
    observer_root_sha256 = semantic_sha256(
        {"resolved_root": str(Path(observer_root).resolve())}
    )
    if (
        value.get("schema") != _STORAGE_SCHEMA
        or value.get("authority_root_sha256") != authority_root_sha256
        or value.get("observer_root_sha256") != observer_root_sha256
        or authority_root_sha256 == observer_root_sha256
        or value.get("distinct_observer_mount") is not True
        or value.get("hard_link_supported") is not True
        or value.get("directory_fsync_supported") is not True
        or value.get("observer_read_matched") is not True
        or value.get("observer_same_file") is not True
        or value.get("duplicate_publication_rejected") is not True
    ):
        raise V16LifecycleContractError(
            "V16 lightweight storage authority drifted"
        )
    _digest("storage payload", value.get("payload_sha256"))
    return V16LifecycleStorageView(
        receipt_sha256=receipt,
        file_sha256=_digest("storage file", expected_file_sha256),
        authority_root_sha256=authority_root_sha256,
        observer_root_sha256=observer_root_sha256,
    )


def load_v16_lifecycle_launch(
    path: str | Path,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    *,
    package: V16LifecyclePackageView,
    authorization: V16LifecycleAuthorizationView,
    admission: V16LifecycleAdmissionView,
    expected_phase: str,
    expected_prerequisite_receipt_sha256: str,
    expected_job_contract_sha256: str,
    expected_pod_contract_sha256: str,
    expected_admission_file_sha256: str,
    storage: V16LifecycleStorageView,
) -> V16LifecycleLaunchView:
    value, receipt = _receipt_payload(
        path,
        expected_file_sha256,
        value_key="authority",
        expected_receipt_sha256=expected_receipt_sha256,
    )
    expected_completions = {
        "capacity": 1,
        "training": 3,
        "qualification-preflight": 3,
        "qualification": 3,
    }
    expected_template = (
        f"pod-runtime/{expected_phase}/{admission.job_uid}/"
        "completion-{completion_index:02d}.json"
    )
    if (
        value.get("schema") != _LAUNCH_SCHEMA
        or value.get("package_plan_sha256") != package.package_plan_sha256
        or value.get("execution_authorization_receipt_sha256")
        != authorization.receipt_sha256
        or value.get("phase") != expected_phase
        or value.get("prerequisite_authority_receipt_sha256")
        != expected_prerequisite_receipt_sha256
        or value.get("admission_receipt_sha256") != admission.receipt_sha256
        or value.get("admission_file_sha256") != expected_admission_file_sha256
        or value.get("job_contract_sha256") != expected_job_contract_sha256
        or value.get("pod_contract_sha256") != expected_pod_contract_sha256
        or value.get("job_uid") != admission.job_uid
        or value.get("run_id") != admission.run_id
        or value.get("completions") != expected_completions.get(expected_phase)
        or value.get("pod_runtime_attestation_path_template") != expected_template
        or value.get("source_tree_root_sha256") is None
        or value.get("storage_semantics_file_sha256") != storage.file_sha256
        or value.get("storage_semantics_receipt_sha256")
        != storage.receipt_sha256
        or value.get("storage_authority_root_sha256")
        != storage.authority_root_sha256
        or value.get("storage_observer_root_sha256")
        != storage.observer_root_sha256
        or value.get("image_digest_sha256") != package.image_digest_sha256
        or value.get("economic_training_authorized") is not False
        or value.get("reinforcement_learning_authorized") is not False
        or value.get("outer_2026_access_authorized") is not False
        or value.get("protocol_sha256") != package.protocol_sha256
    ):
        raise V16LifecycleContractError("V16 lightweight launch authority drifted")
    _digest("source_tree_root_sha256", value.get("source_tree_root_sha256"))
    return V16LifecycleLaunchView(
        receipt_sha256=receipt,
        phase=expected_phase,  # type: ignore[arg-type]
        job_uid=admission.job_uid,
        completions=expected_completions[expected_phase],
        relative_path_template=expected_template,
        storage_semantics_receipt_sha256=storage.receipt_sha256,
        storage_semantics_file_sha256=storage.file_sha256,
    )


def load_v16_lifecycle_pod_attestation(
    path: str | Path,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
    *,
    package: V16LifecyclePackageView,
    authorization: V16LifecycleAuthorizationView,
    admission: V16LifecycleAdmissionView,
    launch: V16LifecycleLaunchView,
    expected_completion_index: int,
    expected_output_root_sha256: str,
    current_pod_uid: str,
    current_pod_name: str,
    current_node_name: str,
    expected_relative_path: str,
) -> V16LifecycleAttestationView:
    value, receipt = _receipt_payload(
        path,
        expected_file_sha256,
        value_key="attestation",
        expected_receipt_sha256=expected_receipt_sha256,
    )
    spec_repository, spec_digest = _image_identity(value.get("observed_spec_image"))
    status_repository, status_digest = _image_identity(
        value.get("observed_status_image")
    )
    image_id_repository, image_id_digest = _image_identity(
        value.get("observed_status_image_id")
    )
    package_repository, package_digest = _image_identity(package.image_reference)
    if (
        value.get("schema") != _ATTESTATION_SCHEMA
        or value.get("package_plan_sha256") != package.package_plan_sha256
        or value.get("execution_authorization_receipt_sha256")
        != authorization.receipt_sha256
        or value.get("phase") != launch.phase
        or value.get("launch_authority_receipt_sha256") != launch.receipt_sha256
        or value.get("admission_receipt_sha256") != admission.receipt_sha256
        or value.get("job_uid") != admission.job_uid
        or value.get("completion_index") != expected_completion_index
        or value.get("relative_path") != expected_relative_path
        or expected_relative_path != launch.relative_path(expected_completion_index)
        or value.get("pod_uid") != current_pod_uid
        or value.get("pod_name") != current_pod_name
        or value.get("node_name") != current_node_name
        or value.get("observed_owner_job_uid") != admission.job_uid
        or value.get("observed_completion_index") != expected_completion_index
        or value.get("observed_owner_job_name") != admission.job_name
        or not value.get("observed_pod_resource_version")
        or value.get("attested_container_name") != "runtime-attestation-gate"
        or value.get("attested_container_kind") != "init"
        or spec_repository != package_repository
        or status_repository != package_repository
        or image_id_repository not in (None, package_repository)
        or {spec_digest, status_digest, image_id_digest}
        != {package_digest}
        or value.get("normalized_image_digest") != package_digest
        or value.get("storage_semantics_file_sha256")
        != launch.storage_semantics_file_sha256
        or value.get("storage_semantics_receipt_sha256")
        != launch.storage_semantics_receipt_sha256
        or value.get("output_root_sha256")
        != _digest("expected_output_root_sha256", expected_output_root_sha256)
        or value.get("protocol_sha256") != package.protocol_sha256
    ):
        raise V16LifecycleContractError("V16 lightweight Pod attestation drifted")
    return V16LifecycleAttestationView(
        receipt_sha256=receipt,
        relative_path=expected_relative_path,
        pod_uid=current_pod_uid,
        pod_name=current_pod_name,
        node_name=current_node_name,
        observed_owner_job_uid=admission.job_uid,
        observed_owner_job_name=_text(
            "observed_owner_job_name", value.get("observed_owner_job_name")
        ),
        observed_completion_index=expected_completion_index,
        observed_pod_resource_version=_text(
            "observed_pod_resource_version",
            value.get("observed_pod_resource_version"),
        ),
    )


__all__ = [
    "V16LifecycleAdmissionView",
    "V16LifecycleAttestationView",
    "V16LifecycleAuthorizationView",
    "V16LifecycleContractError",
    "V16LifecycleLaunchView",
    "V16LifecyclePackageView",
    "V16LifecycleStorageView",
    "canonical_json_file_bytes",
    "file_sha256",
    "load_v16_lifecycle_admission",
    "load_v16_lifecycle_authorization",
    "load_v16_lifecycle_launch",
    "load_v16_lifecycle_package",
    "load_v16_lifecycle_pod_attestation",
    "load_v16_lifecycle_storage",
    "semantic_sha256",
]
