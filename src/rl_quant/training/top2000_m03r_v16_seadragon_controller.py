"""Exact-identity Kubernetes lifecycle controller for M03R-v16.

This module is intentionally inert at import time.  Its concrete transport pins
the approved Seadragon context, namespace, kubeconfig, and kubectl binary on
every request.  Callers must still pass the typed scientific authorities and
must explicitly authorize UID-bound cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_SCHEDULING_POLICY,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    M03R_V16_ADMITTED_MANIFEST_SCHEMA,
    M03R_V16_DRY_RUN_RESULT_SCHEMA,
    M03RV16AdmittedJobAuthority,
    _issue_m03r_v16_admitted_job_authority,
    admitted_job_authority_file_sha256,
    load_m03r_v16_admitted_job_authority,
    load_m03r_v16_phase_launch_authority,
    load_m03r_v16_pod_runtime_attestation,
    phase_launch_authority_file_sha256,
    write_m03r_v16_admitted_job_authority,
)
from rl_quant.training.top2000_m03r_v16_kubernetes import (
    M03R_V16_NAMESPACE,
    M03RV16RenderedSuspendedJob,
    bind_m03r_v16_admitted_launch_authority,
    m03r_v16_pod_runtime_attestation_annotations,
    write_m03r_v16_rendered_launch_authority,
)
from rl_quant.training.top2000_m03r_v16_lifecycle import (
    M03RV16PodAnnotationReadback,
    M03RV16PodObservation,
    M03RV16PublishedPodAttestation,
    M03RV16StorageSemanticsEvidence,
    load_m03r_v16_storage_semantics_evidence,
    publish_m03r_v16_pod_runtime_attestation_after_annotation_patch,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)

M03R_V16_KUBECTL = "/risapps/noarch/kubectl/1.28.4/bin/kubectl"
M03R_V16_KUBECONFIG = "/rsrch8/home/bcb/yding4/.kube/config"
M03R_V16_CONTEXT = "yding4_yn-gpu-workload@kubernetes-admin@kubernetes"
M03R_V16_POD_AUTHORITY_ROOT = "/mnt/authority"
M03R_V16_POD_AUTHORITY_OBSERVER_ROOT = "/mnt/authority-observer"
M03R_V16_CONTROLLER_ADMISSION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-admission-v3"
)
M03R_V16_CONTROLLER_ATTESTATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-attestation-v3"
)
M03R_V16_CONTROLLER_COMPLETION_ATTESTATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-completion-attestation-v1"
)
M03R_V16_CONTROLLER_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-terminal-v2"
)
M03R_V16_CONTROLLER_CLEANUP_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-cleanup-v3"
)
M03R_V16_ZERO_GPU_TRANSITION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-zero-gpu-transition-v1"
)
M03R_V16_CONTROLLER_JOB_PLAN_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-job-plan-v1"
)
_MAX_KUBECTL_RESPONSE_BYTES = 64 * 1024**2
_POD_ATTESTATION_PATH_ANNOTATION = "rl-quant/pod-runtime-attestation-path"
_POD_ATTESTATION_FILE_SHA_ANNOTATION = (
    "rl-quant/pod-runtime-attestation-file-sha256"
)
_POD_ATTESTATION_RECEIPT_ANNOTATION = (
    "rl-quant/pod-runtime-attestation-receipt-sha256"
)


class M03RV16SeadragonControllerError(RuntimeError):
    """An exact Kubernetes identity or lifecycle transition drifted."""


class M03RV16KubernetesConflictError(M03RV16SeadragonControllerError):
    """A Kubernetes optimistic-concurrency precondition was stale."""


class M03RV16KubernetesTransport(Protocol):
    """Minimal exact-object transport used by the lifecycle controller."""

    def invoke(
        self,
        arguments: tuple[str, ...],
        *,
        payload: Mapping[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None: ...


class M03RV16ExactJobAuthority(Protocol):
    @property
    def phase(self) -> str: ...

    @property
    def job_name(self) -> str: ...

    @property
    def job_uid(self) -> str: ...

    @property
    def completions(self) -> int: ...


@dataclass(frozen=True, slots=True)
class M03RV16KubectlTransport:
    """Pinned kubectl transport for the approved Seadragon namespace."""

    kubectl: str = M03R_V16_KUBECTL
    kubeconfig: str = M03R_V16_KUBECONFIG
    context: str = M03R_V16_CONTEXT
    namespace: str = M03R_V16_NAMESPACE
    timeout_seconds: float = 120.0

    def invoke(
        self,
        arguments: tuple[str, ...],
        *,
        payload: Mapping[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        if (
            self.kubectl != M03R_V16_KUBECTL
            or self.kubeconfig != M03R_V16_KUBECONFIG
            or self.context != M03R_V16_CONTEXT
            or self.namespace != M03R_V16_NAMESPACE
            or not arguments
        ):
            raise M03RV16SeadragonControllerError(
                "V16 kubectl transport is outside the approved boundary"
            )
        command = (
            self.kubectl,
            "--kubeconfig",
            self.kubeconfig,
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            *arguments,
        )
        encoded = None if payload is None else canonical_json_file_bytes(payload)
        try:
            completed = subprocess.run(
                command,
                input=encoded,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise M03RV16SeadragonControllerError(
                "V16 exact kubectl request timed out"
            ) from exc
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            if allow_not_found and "not found" in stderr.lower():
                return None
            if "conflict" in stderr.lower() or "409" in stderr:
                raise M03RV16KubernetesConflictError(
                    "V16 Kubernetes resourceVersion conflict"
                )
            raise M03RV16SeadragonControllerError(
                "V16 exact kubectl request failed: " + stderr.strip()[:2048]
            )
        if not 0 < len(completed.stdout) <= _MAX_KUBECTL_RESPONSE_BYTES:
            raise M03RV16SeadragonControllerError(
                "V16 kubectl response is absent or oversized"
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise M03RV16SeadragonControllerError(
                "V16 kubectl response is malformed"
            ) from exc
        if not isinstance(value, dict):
            raise M03RV16SeadragonControllerError(
                "V16 kubectl response is not an object"
            )
        return value


@dataclass(frozen=True, slots=True)
class M03RV16ControllerAdmission:
    rendered: M03RV16RenderedSuspendedJob
    admission: M03RV16AdmittedJobAuthority
    dry_run_path: Path
    admitted_manifest_path: Path
    admission_path: Path
    launch_path: Path
    controller_receipt_path: Path
    prelaunch_job_resource_version: str


@dataclass(frozen=True, slots=True)
class M03RV16ControllerAttestations:
    rows: tuple[M03RV16PublishedPodAttestation, ...]
    controller_receipt_path: Path


@dataclass(frozen=True, slots=True)
class M03RV16ZeroGpuJobAuthority:
    phase: str
    job_name: str
    job_uid: str
    completions: int
    controller_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _M03RV16CompletionAttestationJournal:
    row: M03RV16PublishedPodAttestation
    pod_uid: str
    file_sha256: str
    receipt_sha256: str
    recovery_mode: str


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_json_file_bytes(value)
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16SeadragonControllerError(
            "V16 controller artifact is unavailable"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise M03RV16SeadragonControllerError(
                "V16 controller artifact is not regular"
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise M03RV16SeadragonControllerError(
            "V16 controller artifact changed while hashed"
        )
    return digest.hexdigest()


def _write_or_validate_json(path: Path, value: Mapping[str, Any]) -> str:
    """Publish once or validate the exact prior state after controller restart."""

    raw = canonical_json_file_bytes(value)
    expected = hashlib.sha256(raw).hexdigest()
    try:
        return _write_create_only_json(path, value)
    except FileExistsError:
        observed = _read_exact_json(path, expected)
        if canonical_json_file_bytes(observed) != raw:
            raise M03RV16SeadragonControllerError(
                "V16 durable controller state conflicts with the requested transition"
            )
        return expected


def _read_exact_json(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16SeadragonControllerError(
            "V16 controller file is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= _MAX_KUBECTL_RESPONSE_BYTES
        ):
            raise M03RV16SeadragonControllerError(
                "V16 controller file size drifted"
            )
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise M03RV16SeadragonControllerError(
            "V16 controller file changed or its hash drifted"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16SeadragonControllerError(
            "V16 controller file is malformed"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_file_bytes(value):
        raise M03RV16SeadragonControllerError(
            "V16 controller file is not canonical"
        )
    return value


def _read_self_receipted_json(path: Path) -> tuple[dict[str, Any], str]:
    file_sha256 = _file_sha256(path)
    value = _read_exact_json(path, file_sha256)
    unsigned = {key: row for key, row in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != semantic_sha256(unsigned):
        raise M03RV16SeadragonControllerError(
            "V16 durable controller receipt drifted"
        )
    return value, file_sha256


def write_m03r_v16_controller_job_plan(
    path: str | Path,
    rendered: M03RV16RenderedSuspendedJob,
) -> str:
    """Seal one unbound rendered Job for the external controller."""

    rendered.validate()
    if rendered.launch_authority is not None or rendered.admitted_job_authority is not None:
        raise M03RV16SeadragonControllerError(
            "V16 controller plan must precede live admission"
        )
    row = asdict(rendered)
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_JOB_PLAN_SCHEMA,
        "rendered": row,
    }
    value = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    return _write_create_only_json(Path(path), value)


def load_m03r_v16_controller_job_plan(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> M03RV16RenderedSuspendedJob:
    """Load and validate one exact unbound controller Job plan."""

    value = _read_exact_json(Path(path), expected_file_sha256)
    unsigned = {key: row for key, row in value.items() if key != "receipt_sha256"}
    try:
        row = dict(value["rendered"])
        rendered = M03RV16RenderedSuspendedJob(**row)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16SeadragonControllerError(
            "V16 controller Job plan is malformed"
        ) from exc
    if (
        value.get("schema") != M03R_V16_CONTROLLER_JOB_PLAN_SCHEMA
        or value.get("receipt_sha256") != semantic_sha256(unsigned)
        or rendered.launch_authority is not None
        or rendered.admitted_job_authority is not None
    ):
        raise M03RV16SeadragonControllerError(
            "V16 controller Job plan receipt drifted"
        )
    rendered.validate()
    return rendered


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    row = value.get("metadata")
    if not isinstance(row, dict):
        raise M03RV16SeadragonControllerError("V16 object metadata is absent")
    return row


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise M03RV16SeadragonControllerError(f"V16 {name} is absent")
    return value


def _job_name(rendered: M03RV16RenderedSuspendedJob) -> str:
    return _text("Job name", _metadata(rendered.manifest).get("name"))


def _validate_zero_gpu_job_profile(value: Mapping[str, Any]) -> None:
    spec = value.get("spec")
    template = spec.get("template") if isinstance(spec, dict) else None
    pod = template.get("spec") if isinstance(template, dict) else None
    containers = pod.get("containers") if isinstance(pod, dict) else None
    if not isinstance(containers, list) or len(containers) != 1:
        raise M03RV16SeadragonControllerError(
            "V16 zero-GPU Job container inventory drifted"
        )
    container = containers[0]
    if not isinstance(container, dict):
        raise M03RV16SeadragonControllerError(
            "V16 zero-GPU Job container is malformed"
        )
    resources = container.get("resources")
    requests = resources.get("requests") if isinstance(resources, dict) else None
    limits = resources.get("limits") if isinstance(resources, dict) else None
    environment = container.get("env")
    if (
        not isinstance(requests, dict)
        or not isinstance(limits, dict)
        or requests.get("nvidia.com/gpu") != "0"
        or limits.get("nvidia.com/gpu") != "0"
        or not isinstance(environment, list)
        or not any(
            isinstance(row, dict)
            and row.get("name") == "NVIDIA_VISIBLE_DEVICES"
            and row.get("value") == "none"
            for row in environment
        )
    ):
        raise M03RV16SeadragonControllerError(
            "V16 zero-GPU Job admitted a GPU or lost its visibility mask"
        )


def _validate_exact_job(
    value: Mapping[str, Any],
    rendered: M03RV16RenderedSuspendedJob,
    *,
    expected_uid: str | None,
    expected_suspended: bool,
    require_admitted_identity: bool = True,
) -> tuple[str, str]:
    metadata = _metadata(value)
    spec = value.get("spec")
    if not isinstance(spec, dict):
        raise M03RV16SeadragonControllerError("V16 admitted Job spec is absent")
    annotations = metadata.get("annotations")
    labels = metadata.get("labels")
    if not isinstance(annotations, dict) or not isinstance(labels, dict):
        raise M03RV16SeadragonControllerError(
            "V16 admitted Job labels or annotations are absent"
        )
    uid = (
        _text("Job UID", metadata.get("uid"))
        if require_admitted_identity
        else str(metadata.get("uid") or "dry-run-not-admitted")
    )
    resource_version = (
        _text("Job resourceVersion", metadata.get("resourceVersion"))
        if require_admitted_identity
        else str(metadata.get("resourceVersion") or "dry-run-not-admitted")
    )
    if (
        metadata.get("name") != _job_name(rendered)
        or metadata.get("namespace") != M03R_V16_NAMESPACE
        or (expected_uid is not None and uid != expected_uid)
        or labels.get("rl-quant/run-id")
        != _metadata(rendered.manifest).get("labels", {}).get("rl-quant/run-id")
        or annotations.get("rl-quant/job-contract-sha256")
        != rendered.job_contract_sha256
        or annotations.get("rl-quant/pod-contract-sha256")
        != rendered.pod_contract_sha256
        or spec.get("suspend") is not expected_suspended
        or spec.get("completions") != rendered.completions
        or spec.get("parallelism") != rendered.parallelism
    ):
        raise M03RV16SeadragonControllerError(
            "V16 exact admitted Job contract drifted"
        )
    if rendered.mode in {"static", "storage"}:
        _validate_zero_gpu_job_profile(value)
    return uid, resource_version


def _list_exact_job_pods(
    transport: M03RV16KubernetesTransport,
    *,
    job_name: str,
) -> tuple[dict[str, Any], ...]:
    value = transport.invoke(
        (
            "get",
            "pods",
            "--selector",
            f"job-name={job_name}",
            "--output",
            "json",
        )
    )
    if value is None or not isinstance(value.get("items"), list):
        raise M03RV16SeadragonControllerError("V16 Pod inventory is malformed")
    return tuple(dict(row) for row in value["items"] if isinstance(row, dict))


def _create_or_reconcile_suspended_job(
    transport: M03RV16KubernetesTransport,
    rendered: M03RV16RenderedSuspendedJob,
) -> dict[str, Any]:
    """Create once, reconciling the exact name after ambiguous output."""

    try:
        admitted = transport.invoke(
            ("create", "-f", "-", "--output", "json"),
            payload=rendered.manifest,
        )
    except M03RV16SeadragonControllerError:
        admitted = transport.invoke(
            ("get", "job", _job_name(rendered), "--output", "json"),
            allow_not_found=True,
        )
        if admitted is None:
            raise
    if admitted is None:
        raise M03RV16SeadragonControllerError("V16 admitted Job vanished")
    _validate_exact_job(
        admitted,
        rendered,
        expected_uid=None,
        expected_suspended=True,
    )
    return admitted


def admit_m03r_v16_suspended_job(
    *,
    transport: M03RV16KubernetesTransport,
    rendered: M03RV16RenderedSuspendedJob,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    storage_evidence: M03RV16StorageSemanticsEvidence,
    storage_evidence_file_sha256: str,
    authority_root: str | Path,
    source_tree_root_sha256: str,
    run_id: str,
    storage_authority_identity_root: str | Path = M03R_V16_POD_AUTHORITY_ROOT,
    storage_observer_identity_root: str | Path = (
        M03R_V16_POD_AUTHORITY_OBSERVER_ROOT
    ),
) -> M03RV16ControllerAdmission:
    """Converge one exact Job through dry-run, admission, and launch binding."""

    rendered.validate()
    if rendered.mode in {"static", "storage"}:
        raise M03RV16SeadragonControllerError(
            "V16 scientific admission requires a phase authority"
        )
    root = Path(authority_root)
    job_name = _job_name(rendered)
    if (
        run_id
        != _metadata(rendered.manifest).get("labels", {}).get("rl-quant/run-id")
    ):
        raise M03RV16SeadragonControllerError(
            "V16 controller run identity differs from the rendered Job"
        )
    dry_path = root / f"{rendered.mode}-dry-run.json"
    admitted_path = root / f"{rendered.mode}-admitted-manifest.json"
    admission_path = root / f"{rendered.mode}-admission.json"
    launch_path = root / f"{rendered.mode}-launch.json"
    controller_receipt_path = root / f"{rendered.mode}-controller-admission.json"

    if dry_path.exists():
        dry_value, dry_file_sha = _read_self_receipted_json(dry_path)
        if (
            dry_value.get("schema") != M03R_V16_DRY_RUN_RESULT_SCHEMA
            or dry_value.get("package_plan_sha256") != package.package_plan_sha256
            or dry_value.get("phase") != rendered.mode
            or dry_value.get("job_contract_sha256")
            != rendered.job_contract_sha256
            or dry_value.get("pod_contract_sha256")
            != rendered.pod_contract_sha256
            or dry_value.get("passed") is not True
        ):
            raise M03RV16SeadragonControllerError(
                "V16 prior server dry-run state drifted"
            )
    else:
        dry_observed = transport.invoke(
            ("create", "--dry-run=server", "-f", "-", "--output", "json"),
            payload=rendered.manifest,
        )
        if dry_observed is None:
            raise M03RV16SeadragonControllerError("V16 server dry-run vanished")
        _validate_exact_job(
            dry_observed,
            rendered,
            expected_uid=None,
            expected_suspended=True,
            require_admitted_identity=False,
        )
        dry_unsigned: dict[str, Any] = {
            "schema": M03R_V16_DRY_RUN_RESULT_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "phase": rendered.mode,
            "job_contract_sha256": rendered.job_contract_sha256,
            "pod_contract_sha256": rendered.pod_contract_sha256,
            "server_response_sha256": semantic_sha256(dry_observed),
            "passed": True,
        }
        dry_value = {
            **dry_unsigned,
            "receipt_sha256": semantic_sha256(dry_unsigned),
        }
        dry_file_sha = _write_or_validate_json(dry_path, dry_value)

    if admitted_path.exists():
        admitted_value, admitted_file_sha = _read_self_receipted_json(
            admitted_path
        )
        job_uid = _text("admitted Job UID", admitted_value.get("job_uid"))
        current = transport.invoke(
            ("get", "job", job_name, "--output", "json"),
            allow_not_found=True,
        )
        if current is None:
            raise M03RV16SeadragonControllerError(
                "V16 previously admitted Job is absent"
            )
        current_spec = current.get("spec")
        is_suspended = (
            isinstance(current_spec, dict)
            and current_spec.get("suspend") is True
        )
        _, job_resource_version = _validate_exact_job(
            current,
            rendered,
            expected_uid=job_uid,
            expected_suspended=is_suspended,
        )
    else:
        admitted = _create_or_reconcile_suspended_job(transport, rendered)
        job_uid, job_resource_version = _validate_exact_job(
            admitted,
            rendered,
            expected_uid=None,
            expected_suspended=True,
        )
        if _list_exact_job_pods(transport, job_name=job_name):
            raise M03RV16SeadragonControllerError(
                "V16 suspended Job created Pods before launch authority"
            )
        admitted_unsigned: dict[str, Any] = {
            "schema": M03R_V16_ADMITTED_MANIFEST_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "phase": rendered.mode,
            "job_contract_sha256": rendered.job_contract_sha256,
            "pod_contract_sha256": rendered.pod_contract_sha256,
            "job_uid": job_uid,
            "job_name": job_name,
            "job_resource_version": job_resource_version,
            "image_reference": package.artifacts.image_reference,
            "image_digest_sha256": package.artifacts.image_digest_sha256,
            "observed_manifest_sha256": semantic_sha256(admitted),
            "pod_uids": [],
            "container_image_ids": [],
            "node_names": [],
            "suspended_at_admission": True,
        }
        admitted_value = {
            **admitted_unsigned,
            "receipt_sha256": semantic_sha256(admitted_unsigned),
        }
        admitted_file_sha = _write_or_validate_json(
            admitted_path, admitted_value
        )

    if (
        admitted_value.get("schema") != M03R_V16_ADMITTED_MANIFEST_SCHEMA
        or admitted_value.get("package_plan_sha256") != package.package_plan_sha256
        or admitted_value.get("phase") != rendered.mode
        or admitted_value.get("job_contract_sha256")
        != rendered.job_contract_sha256
        or admitted_value.get("pod_contract_sha256")
        != rendered.pod_contract_sha256
        or admitted_value.get("job_uid") != job_uid
        or admitted_value.get("job_name") != job_name
        or admitted_value.get("suspended_at_admission") is not True
    ):
        raise M03RV16SeadragonControllerError(
            "V16 prior admitted-manifest state drifted"
        )
    admission = _issue_m03r_v16_admitted_job_authority(
        package=package,
        authorization=authorization,
        phase=rendered.mode,  # type: ignore[arg-type]
        run_id=run_id,
        job_name=job_name,
        job_contract_sha256=rendered.job_contract_sha256,
        pod_contract_sha256=rendered.pod_contract_sha256,
        server_side_dry_run_file_sha256=dry_file_sha,
        server_side_dry_run_receipt_sha256=dry_value["receipt_sha256"],
        admitted_manifest_file_sha256=admitted_file_sha,
        admitted_manifest_sha256=admitted_value["receipt_sha256"],
        job_uid=job_uid,
    )
    expected_admission_file_sha = admitted_job_authority_file_sha256(admission)
    if admission_path.exists():
        admission_file_sha = _file_sha256(admission_path)
        if admission_file_sha != expected_admission_file_sha:
            raise M03RV16SeadragonControllerError(
                "V16 prior admission authority differs"
            )
        admission = load_m03r_v16_admitted_job_authority(
            admission_path,
            expected_file_sha256=admission_file_sha,
            expected_receipt_sha256=admission.receipt_sha256,
            package=package,
            authorization=authorization,
            expected_phase=rendered.mode,
            expected_job_contract_sha256=rendered.job_contract_sha256,
            expected_pod_contract_sha256=rendered.pod_contract_sha256,
            server_side_dry_run_path=dry_path,
            admitted_manifest_path=admitted_path,
        )
    else:
        admission_file_sha = write_m03r_v16_admitted_job_authority(
            admission_path, admission
        )
    bound = bind_m03r_v16_admitted_launch_authority(
        rendered=rendered,
        package=package,
        authorization=authorization,
        admission=admission,
        admission_file_sha256=admission_file_sha,
        source_tree_root_sha256=source_tree_root_sha256,
        storage_evidence=storage_evidence,
        storage_evidence_file_sha256=storage_evidence_file_sha256,
        authority_root=storage_authority_identity_root,
        observer_root=storage_observer_identity_root,
    )
    launch = bound.launch_authority
    if launch is None or bound.launch_authority_file_sha256 is None:
        raise M03RV16SeadragonControllerError(
            "V16 bound launch authority is absent"
        )
    expected_launch_file_sha = phase_launch_authority_file_sha256(launch)
    if launch_path.exists():
        launch_file_sha = _file_sha256(launch_path)
        if launch_file_sha != expected_launch_file_sha:
            raise M03RV16SeadragonControllerError(
                "V16 prior launch authority differs"
            )
        load_m03r_v16_phase_launch_authority(
            launch_path,
            expected_file_sha256=launch_file_sha,
            expected_receipt_sha256=launch.receipt_sha256,
            package=package,
            authorization=authorization,
            expected_phase=rendered.mode,
            expected_prerequisite_receipt_sha256=(
                launch.prerequisite_authority_receipt_sha256
            ),
            expected_job_contract_sha256=rendered.job_contract_sha256,
            expected_pod_contract_sha256=rendered.pod_contract_sha256,
            admission=admission,
            expected_admission_file_sha256=admission_file_sha,
        )
    else:
        launch_file_sha = write_m03r_v16_rendered_launch_authority(
            launch_path, bound
        )
    bound_annotations = _metadata(bound.manifest).get("annotations", {})
    bound_pod_annotations = (
        bound.manifest.get("spec", {})
        .get("template", {})
        .get("metadata", {})
        .get("annotations", {})
    )
    patched: dict[str, Any] | None = None
    for _attempt in range(5):
        current = transport.invoke(("get", "job", job_name, "--output", "json"))
        if current is None:
            raise M03RV16SeadragonControllerError("V16 bound Job vanished")
        current_spec = current.get("spec")
        suspended = isinstance(current_spec, dict) and current_spec.get("suspend") is True
        _, current_resource_version = _validate_exact_job(
            current,
            rendered,
            expected_uid=job_uid,
            expected_suspended=suspended,
        )
        current_annotations = _metadata(current).get("annotations", {})
        current_pod_annotations = (
            current.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get("annotations", {})
        )
        if all(
            current_annotations.get(key) == value
            for key, value in bound_annotations.items()
        ) and all(
            current_pod_annotations.get(key) == value
            for key, value in bound_pod_annotations.items()
        ):
            patched = current
            break
        if not suspended:
            raise M03RV16SeadragonControllerError(
                "V16 resumed Job lacks frozen launch annotations"
            )
        try:
            patched = transport.invoke(
                (
                    "patch", "job", job_name, "--type", "merge",
                    "--patch-file", "/dev/stdin", "--output", "json",
                ),
                payload={
                    "metadata": {
                        "resourceVersion": current_resource_version,
                        "annotations": bound_annotations,
                    },
                    "spec": {
                        "suspend": True,
                        "template": {
                            "metadata": {"annotations": bound_pod_annotations}
                        },
                    },
                },
            )
            break
        except M03RV16KubernetesConflictError:
            continue
        except M03RV16SeadragonControllerError:
            # Reconcile a response that may have been lost after the API
            # accepted the exact patch.  Any uncommitted mutation still fails.
            observed = transport.invoke(
                ("get", "job", job_name, "--output", "json"),
                allow_not_found=True,
            )
            if observed is None or _metadata(observed).get("uid") != job_uid:
                raise
            observed_annotations = _metadata(observed).get("annotations", {})
            observed_pod_annotations = (
                observed.get("spec", {})
                .get("template", {})
                .get("metadata", {})
                .get("annotations", {})
            )
            if not (
                all(
                    observed_annotations.get(key) == value
                    for key, value in bound_annotations.items()
                )
                and all(
                    observed_pod_annotations.get(key) == value
                    for key, value in bound_pod_annotations.items()
                )
            ):
                raise
            patched = observed
            break
    if patched is None:
        raise M03RV16SeadragonControllerError(
            "V16 launch annotation patch did not converge"
        )
    patched_spec = patched.get("spec")
    patched_suspended = (
        isinstance(patched_spec, dict) and patched_spec.get("suspend") is True
    )
    _, patched_resource_version = _validate_exact_job(
        patched,
        rendered,
        expected_uid=job_uid,
        expected_suspended=patched_suspended,
    )
    patched_metadata_annotations = _metadata(patched).get("annotations", {})
    patched_pod_annotations = (
        patched.get("spec", {})
        .get("template", {})
        .get("metadata", {})
        .get("annotations", {})
    )
    if any(
        patched_metadata_annotations.get(key) != value
        for key, value in bound_annotations.items()
    ) or any(
        patched_pod_annotations.get(key) != value
        for key, value in bound_pod_annotations.items()
    ):
        raise M03RV16SeadragonControllerError(
            "V16 launch annotations were not admitted while suspended"
        )
    if patched_suspended and _list_exact_job_pods(transport, job_name=job_name):
        raise M03RV16SeadragonControllerError(
            "V16 launch-bound Job created Pods while suspended"
        )
    controller_unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_ADMISSION_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "phase": rendered.mode,
        "run_id": run_id,
        "job_name": job_name,
        "job_uid": job_uid,
        "job_contract_sha256": rendered.job_contract_sha256,
        "pod_contract_sha256": rendered.pod_contract_sha256,
        "source_tree_root_sha256": source_tree_root_sha256,
        "prelaunch_job_resource_version": patched_resource_version,
        "server_side_dry_run_file_sha256": dry_file_sha,
        "admitted_manifest_file_sha256": admitted_file_sha,
        "admission_authority_file_sha256": admission_file_sha,
        "admission_authority_receipt_sha256": admission.receipt_sha256,
        "launch_authority_file_sha256": launch_file_sha,
        "launch_authority_receipt_sha256": (
            None
            if bound.launch_authority is None
            else bound.launch_authority.receipt_sha256
        ),
        "storage_semantics_receipt_sha256": storage_evidence.receipt_sha256,
        "scheduling_policy": M03R_V16_SCHEDULING_POLICY,
        "gang_scheduling_required": False,
        "minimum_schedulable_h100s": (
            2 if rendered.mode in {"capacity", "training", "qualification"} else 0
        ),
        "suspended": True,
        "pod_inventory_empty": True,
    }
    controller_value = {
        **controller_unsigned,
        "receipt_sha256": semantic_sha256(controller_unsigned),
    }
    if controller_receipt_path.exists():
        prior, _ = _read_self_receipted_json(controller_receipt_path)
        stable_keys = (
            "schema", "package_plan_sha256",
            "execution_authorization_receipt_sha256", "phase", "run_id",
            "job_name", "job_uid", "job_contract_sha256",
            "pod_contract_sha256", "source_tree_root_sha256",
            "server_side_dry_run_file_sha256", "admitted_manifest_file_sha256",
            "admission_authority_file_sha256", "admission_authority_receipt_sha256",
            "launch_authority_file_sha256", "launch_authority_receipt_sha256",
            "storage_semantics_receipt_sha256",
            "scheduling_policy", "gang_scheduling_required",
            "minimum_schedulable_h100s",
        )
        if any(prior.get(key) != controller_value.get(key) for key in stable_keys):
            raise M03RV16SeadragonControllerError(
                "V16 recovered controller admission differs"
            )
    else:
        _write_or_validate_json(controller_receipt_path, controller_value)
    return M03RV16ControllerAdmission(
        rendered=bound,
        admission=admission,
        dry_run_path=dry_path,
        admitted_manifest_path=admitted_path,
        admission_path=admission_path,
        launch_path=launch_path,
        controller_receipt_path=controller_receipt_path,
        prelaunch_job_resource_version=patched_resource_version,
    )


def load_m03r_v16_controller_admission(
    path: str | Path,
    *,
    expected_file_sha256: str,
    rendered: M03RV16RenderedSuspendedJob,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    storage_evidence: M03RV16StorageSemanticsEvidence,
    storage_evidence_file_sha256: str,
    storage_authority_identity_root: str | Path = M03R_V16_POD_AUTHORITY_ROOT,
    storage_observer_identity_root: str | Path = (
        M03R_V16_POD_AUTHORITY_OBSERVER_ROOT
    ),
) -> M03RV16ControllerAdmission:
    """Reconstruct one controller transaction without mutating Kubernetes."""

    controller_path = Path(path)
    value = _read_exact_json(controller_path, expected_file_sha256)
    unsigned = {key: row for key, row in value.items() if key != "receipt_sha256"}
    root = controller_path.parent
    phase = rendered.mode
    dry_path = root / f"{phase}-dry-run.json"
    admitted_path = root / f"{phase}-admitted-manifest.json"
    admission_path = root / f"{phase}-admission.json"
    launch_path = root / f"{phase}-launch.json"
    if (
        value.get("schema") != M03R_V16_CONTROLLER_ADMISSION_SCHEMA
        or value.get("receipt_sha256") != semantic_sha256(unsigned)
        or value.get("package_plan_sha256") != package.package_plan_sha256
        or value.get("execution_authorization_receipt_sha256")
        != authorization.receipt_sha256
        or value.get("phase") != phase
        or value.get("job_name") != _job_name(rendered)
        or value.get("job_contract_sha256") != rendered.job_contract_sha256
        or value.get("pod_contract_sha256") != rendered.pod_contract_sha256
        or value.get("storage_semantics_receipt_sha256")
        != storage_evidence.receipt_sha256
        or value.get("scheduling_policy") != M03R_V16_SCHEDULING_POLICY
        or value.get("gang_scheduling_required") is not False
        or value.get("minimum_schedulable_h100s")
        != (2 if phase in {"capacity", "training", "qualification"} else 0)
    ):
        raise M03RV16SeadragonControllerError(
            "V16 controller admission receipt drifted"
        )
    admission = load_m03r_v16_admitted_job_authority(
        admission_path,
        expected_file_sha256=_text(
            "admission authority file SHA",
            value.get("admission_authority_file_sha256"),
        ),
        expected_receipt_sha256=_text(
            "admission authority receipt",
            value.get("admission_authority_receipt_sha256"),
        ),
        package=package,
        authorization=authorization,
        expected_phase=phase,
        expected_job_contract_sha256=rendered.job_contract_sha256,
        expected_pod_contract_sha256=rendered.pod_contract_sha256,
        server_side_dry_run_path=dry_path,
        admitted_manifest_path=admitted_path,
    )
    bound = bind_m03r_v16_admitted_launch_authority(
        rendered=rendered,
        package=package,
        authorization=authorization,
        admission=admission,
        admission_file_sha256=_text(
            "admission authority file SHA",
            value.get("admission_authority_file_sha256"),
        ),
        source_tree_root_sha256=_text(
            "source-tree root", value.get("source_tree_root_sha256")
        ),
        storage_evidence=storage_evidence,
        storage_evidence_file_sha256=storage_evidence_file_sha256,
        authority_root=storage_authority_identity_root,
        observer_root=storage_observer_identity_root,
    )
    launch = bound.launch_authority
    if launch is None:
        raise M03RV16SeadragonControllerError(
            "V16 recovered launch authority is absent"
        )
    load_m03r_v16_phase_launch_authority(
        launch_path,
        expected_file_sha256=_text(
            "launch authority file SHA",
            value.get("launch_authority_file_sha256"),
        ),
        expected_receipt_sha256=_text(
            "launch authority receipt",
            value.get("launch_authority_receipt_sha256"),
        ),
        package=package,
        authorization=authorization,
        expected_phase=phase,
        expected_prerequisite_receipt_sha256=(
            launch.prerequisite_authority_receipt_sha256
        ),
        expected_job_contract_sha256=rendered.job_contract_sha256,
        expected_pod_contract_sha256=rendered.pod_contract_sha256,
        admission=admission,
        expected_admission_file_sha256=_text(
            "admission authority file SHA",
            value.get("admission_authority_file_sha256"),
        ),
    )
    return M03RV16ControllerAdmission(
        rendered=bound,
        admission=admission,
        dry_run_path=dry_path,
        admitted_manifest_path=admitted_path,
        admission_path=admission_path,
        launch_path=launch_path,
        controller_receipt_path=controller_path,
        prelaunch_job_resource_version=_text(
            "prelaunch Job resourceVersion",
            value.get("prelaunch_job_resource_version"),
        ),
    )


def launch_m03r_v16_zero_gpu_gate(
    *,
    transport: M03RV16KubernetesTransport,
    rendered: M03RV16RenderedSuspendedJob,
    authority_root: str | Path,
) -> M03RV16ZeroGpuJobAuthority:
    """Converge one static or storage Job through durable launch stages."""

    rendered.validate()
    if rendered.mode not in {"static", "storage"}:
        raise M03RV16SeadragonControllerError(
            "V16 zero-GPU launch received a scientific phase"
        )
    job_name = _job_name(rendered)
    root = Path(authority_root)
    receipt_path = root / f"{rendered.mode}-controller-admission.json"
    dry_stage_path = root / f"{rendered.mode}-zero-gpu-dry-run.json"
    admitted_stage_path = root / f"{rendered.mode}-zero-gpu-admitted.json"
    resumed_stage_path = root / f"{rendered.mode}-zero-gpu-resumed.json"
    if receipt_path.exists():
        prior, _ = _read_self_receipted_json(receipt_path)
        job_uid = _text("zero-GPU Job UID", prior.get("job_uid"))
        current = transport.invoke(
            ("get", "job", job_name, "--output", "json"),
            allow_not_found=True,
        )
        if (
            prior.get("schema") != M03R_V16_CONTROLLER_ADMISSION_SCHEMA
            or prior.get("phase") != rendered.mode
            or prior.get("job_name") != job_name
            or prior.get("completions") != rendered.completions
            or prior.get("zero_gpu") is not True
            or current is None
        ):
            raise M03RV16SeadragonControllerError(
                "V16 prior zero-GPU controller state drifted"
            )
        _validate_exact_job(
            current,
            rendered,
            expected_uid=job_uid,
            expected_suspended=False,
        )
        return M03RV16ZeroGpuJobAuthority(
            phase=rendered.mode,
            job_name=job_name,
            job_uid=job_uid,
            completions=rendered.completions,
            controller_receipt_sha256=_text(
                "zero-GPU controller receipt", prior.get("receipt_sha256")
            ),
        )
    if dry_stage_path.exists():
        dry_stage, _ = _read_self_receipted_json(dry_stage_path)
        if (
            dry_stage.get("schema") != M03R_V16_ZERO_GPU_TRANSITION_SCHEMA
            or dry_stage.get("stage") != "server-dry-run-complete"
            or dry_stage.get("phase") != rendered.mode
            or dry_stage.get("job_name") != job_name
            or dry_stage.get("job_contract_sha256")
            != rendered.job_contract_sha256
        ):
            raise M03RV16SeadragonControllerError(
                "V16 prior zero-GPU dry-run stage drifted"
            )
        dry_sha = _text(
            "zero-GPU dry-run response SHA",
            dry_stage.get("server_dry_run_response_sha256"),
        )
    else:
        dry = transport.invoke(
            ("create", "--dry-run=server", "-f", "-", "--output", "json"),
            payload=rendered.manifest,
        )
        if dry is None:
            raise M03RV16SeadragonControllerError("V16 zero-GPU dry-run vanished")
        _validate_exact_job(
            dry,
            rendered,
            expected_uid=None,
            expected_suspended=True,
            require_admitted_identity=False,
        )
        dry_sha = semantic_sha256(dry)
        dry_unsigned = {
            "schema": M03R_V16_ZERO_GPU_TRANSITION_SCHEMA,
            "stage": "server-dry-run-complete",
            "phase": rendered.mode,
            "job_name": job_name,
            "job_contract_sha256": rendered.job_contract_sha256,
            "server_dry_run_response_sha256": dry_sha,
        }
        _write_or_validate_json(
            dry_stage_path,
            {**dry_unsigned, "receipt_sha256": semantic_sha256(dry_unsigned)},
        )

    current = transport.invoke(
        ("get", "job", job_name, "--output", "json"),
        allow_not_found=True,
    )
    if admitted_stage_path.exists():
        admitted_stage, _ = _read_self_receipted_json(admitted_stage_path)
        job_uid = _text("zero-GPU Job UID", admitted_stage.get("job_uid"))
        if (
            admitted_stage.get("schema")
            != M03R_V16_ZERO_GPU_TRANSITION_SCHEMA
            or admitted_stage.get("stage") != "suspended-job-admitted"
            or admitted_stage.get("phase") != rendered.mode
            or admitted_stage.get("job_name") != job_name
            or admitted_stage.get("server_dry_run_response_sha256") != dry_sha
            or current is None
        ):
            raise M03RV16SeadragonControllerError(
                "V16 prior zero-GPU admission stage drifted"
            )
        current_spec = current.get("spec")
        currently_suspended = (
            isinstance(current_spec, dict)
            and current_spec.get("suspend") is True
        )
        _, resource_version = _validate_exact_job(
            current,
            rendered,
            expected_uid=job_uid,
            expected_suspended=currently_suspended,
        )
        admitted_sha = _text(
            "zero-GPU admitted manifest SHA",
            admitted_stage.get("admitted_manifest_sha256"),
        )
    else:
        admitted = (
            _create_or_reconcile_suspended_job(transport, rendered)
            if current is None
            else current
        )
        admitted_spec = admitted.get("spec")
        currently_suspended = (
            isinstance(admitted_spec, dict)
            and admitted_spec.get("suspend") is True
        )
        job_uid, resource_version = _validate_exact_job(
            admitted,
            rendered,
            expected_uid=None,
            expected_suspended=currently_suspended,
        )
        admitted_sha = semantic_sha256(admitted)
        admitted_unsigned = {
            "schema": M03R_V16_ZERO_GPU_TRANSITION_SCHEMA,
            "stage": "suspended-job-admitted",
            "phase": rendered.mode,
            "job_uid": job_uid,
            "job_name": job_name,
            "job_contract_sha256": rendered.job_contract_sha256,
            "server_dry_run_response_sha256": dry_sha,
            "admitted_manifest_sha256": admitted_sha,
            "admitted_job_resource_version": resource_version,
            "suspended_when_journaled": currently_suspended,
        }
        _write_or_validate_json(
            admitted_stage_path,
            {
                **admitted_unsigned,
                "receipt_sha256": semantic_sha256(admitted_unsigned),
            },
        )

    if currently_suspended:
        if _list_exact_job_pods(transport, job_name=job_name):
            raise M03RV16SeadragonControllerError(
                "V16 suspended zero-GPU Job created a Pod"
            )
        for _attempt in range(5):
            try:
                resumed = transport.invoke(
                    (
                        "patch", "job", job_name, "--type", "merge",
                        "--patch-file", "/dev/stdin", "--output", "json",
                    ),
                    payload={
                        "metadata": {"resourceVersion": resource_version},
                        "spec": {"suspend": False},
                    },
                )
            except M03RV16KubernetesConflictError:
                resumed = transport.invoke(
                    ("get", "job", job_name, "--output", "json")
                )
                if resumed is None:
                    continue
                resumed_spec = resumed.get("spec")
                if isinstance(resumed_spec, dict) and resumed_spec.get("suspend") is False:
                    break
                _, resource_version = _validate_exact_job(
                    resumed,
                    rendered,
                    expected_uid=job_uid,
                    expected_suspended=True,
                )
                continue
            except M03RV16SeadragonControllerError:
                resumed = transport.invoke(
                    ("get", "job", job_name, "--output", "json"),
                    allow_not_found=True,
                )
                if resumed is None:
                    raise
                resumed_spec = resumed.get("spec")
                if not (
                    isinstance(resumed_spec, dict)
                    and resumed_spec.get("suspend") is False
                ):
                    raise
                break
            if resumed is None:
                raise M03RV16SeadragonControllerError(
                    "V16 zero-GPU Job resume vanished"
                )
            break
        else:
            raise M03RV16SeadragonControllerError(
                "V16 zero-GPU resume conflicts did not converge"
            )
    else:
        resumed = current
    if resumed is None:
        raise M03RV16SeadragonControllerError("V16 zero-GPU resumed Job vanished")
    _, resumed_resource_version = _validate_exact_job(
        resumed,
        rendered,
        expected_uid=job_uid,
        expected_suspended=False,
    )
    resumed_unsigned = {
        "schema": M03R_V16_ZERO_GPU_TRANSITION_SCHEMA,
        "stage": "job-resume-observed",
        "phase": rendered.mode,
        "job_uid": job_uid,
        "job_name": job_name,
        "job_contract_sha256": rendered.job_contract_sha256,
        "resumed_job_resource_version": resumed_resource_version,
    }
    _write_or_validate_json(
        resumed_stage_path,
        {**resumed_unsigned, "receipt_sha256": semantic_sha256(resumed_unsigned)},
    )
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_ADMISSION_SCHEMA,
        "phase": rendered.mode,
        "job_uid": job_uid,
        "job_name": job_name,
        "completions": rendered.completions,
        "server_dry_run_response_sha256": dry_sha,
        "admitted_manifest_sha256": admitted_sha,
        "resumed_job_resource_version": resumed_resource_version,
        "zero_gpu_dry_run_stage_file_sha256": _file_sha256(dry_stage_path),
        "zero_gpu_admitted_stage_file_sha256": _file_sha256(
            admitted_stage_path
        ),
        "zero_gpu_resumed_stage_file_sha256": _file_sha256(resumed_stage_path),
        "zero_gpu": True,
    }
    receipt = semantic_sha256(unsigned)
    _write_or_validate_json(
        receipt_path, {**unsigned, "receipt_sha256": receipt}
    )
    return M03RV16ZeroGpuJobAuthority(
        phase=rendered.mode,
        job_name=job_name,
        job_uid=job_uid,
        completions=rendered.completions,
        controller_receipt_sha256=receipt,
    )


def _pod_completion_index(pod: Mapping[str, Any]) -> int:
    annotations = _metadata(pod).get("annotations")
    if not isinstance(annotations, dict):
        raise M03RV16SeadragonControllerError("V16 Pod annotations are absent")
    raw = annotations.get("batch.kubernetes.io/job-completion-index")
    if not isinstance(raw, (str, int)):
        raise M03RV16SeadragonControllerError(
            "V16 Pod completion index is malformed"
        )
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise M03RV16SeadragonControllerError(
            "V16 Pod completion index is malformed"
        ) from exc


def _observe_pod(
    pod: Mapping[str, Any],
    *,
    admission: M03RV16AdmittedJobAuthority,
) -> M03RV16PodObservation:
    metadata = _metadata(pod)
    spec = pod.get("spec")
    status = pod.get("status")
    if not isinstance(spec, dict) or not isinstance(status, dict):
        raise M03RV16SeadragonControllerError("V16 Pod spec or status is absent")
    owners = metadata.get("ownerReferences")
    if not isinstance(owners, list):
        raise M03RV16SeadragonControllerError("V16 Pod owner inventory is absent")
    owner = next(
        (
            row
            for row in owners
            if isinstance(row, dict)
            and row.get("kind") == "Job"
            and row.get("controller") is True
        ),
        None,
    )
    if (
        owner is None
        or owner.get("uid") != admission.job_uid
        or owner.get("name") != admission.job_name
    ):
        raise M03RV16SeadragonControllerError("V16 Pod owner Job drifted")
    init_specs = spec.get("initContainers")
    init_statuses = status.get("initContainerStatuses")
    if not isinstance(init_specs, list) or not isinstance(init_statuses, list):
        raise M03RV16SeadragonControllerError(
            "V16 runtime attestation init container is not observable"
        )
    init_spec = next(
        (
            row
            for row in init_specs
            if isinstance(row, dict)
            and row.get("name") == "runtime-attestation-gate"
        ),
        None,
    )
    init_status = next(
        (
            row
            for row in init_statuses
            if isinstance(row, dict)
            and row.get("name") == "runtime-attestation-gate"
        ),
        None,
    )
    if init_spec is None or init_status is None:
        raise M03RV16SeadragonControllerError(
            "V16 runtime attestation init status is absent"
        )
    return M03RV16PodObservation(
        completion_index=_pod_completion_index(pod),
        pod_uid=_text("Pod UID", metadata.get("uid")),
        pod_name=_text("Pod name", metadata.get("name")),
        node_name=_text("Pod node", spec.get("nodeName")),
        observed_owner_job_uid=_text("owner Job UID", owner.get("uid")),
        observed_owner_job_name=_text("owner Job name", owner.get("name")),
        observed_completion_index=_pod_completion_index(pod),
        observed_pod_resource_version=_text(
            "Pod resourceVersion", metadata.get("resourceVersion")
        ),
        attested_container_name="runtime-attestation-gate",
        attested_container_kind="init",
        observed_spec_image=_text("init spec image", init_spec.get("image")),
        observed_status_image=_text("init status image", init_status.get("image")),
        observed_status_image_id=_text(
            "init status image ID", init_status.get("imageID")
        ),
    )


def _output_root_sha256(
    package: M03RV16PackagePlan,
    *,
    phase: str,
    completion_index: int,
) -> str:
    root = (
        "/mnt/output/capacity-sentinel"
        if phase == "capacity"
        else package.panel.workers[completion_index].output_root
    )
    return semantic_sha256({"output_root": str(Path(root).resolve())})


def _runtime_image_digest(value: object) -> str:
    text = _text("container runtime image", value).removeprefix(
        "docker-pullable://"
    )
    if "@sha256:" in text:
        digest = text.rsplit("@sha256:", 1)[1]
    elif text.startswith(("containerd://sha256:", "docker://sha256:")):
        digest = text.rsplit("sha256:", 1)[1]
    else:
        raise M03RV16SeadragonControllerError(
            "V16 container runtime image is not digest pinned"
        )
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise M03RV16SeadragonControllerError(
            "V16 container runtime image digest is malformed"
        )
    return digest


def _completion_attestation_journal_path(
    authority_root: str | Path,
    *,
    phase: str,
    job_uid: str,
    completion_index: int,
) -> Path:
    return (
        Path(authority_root)
        / "controller-attestations"
        / phase
        / job_uid
        / f"completion-{completion_index:02d}.json"
    )


def _write_completion_attestation_journal(
    *,
    authority_root: str | Path,
    phase: str,
    job_uid: str,
    job_name: str,
    launch_receipt_sha256: str,
    storage_receipt_sha256: str,
    completion_index: int,
    observation: M03RV16PodObservation,
    row: M03RV16PublishedPodAttestation,
    recovery_mode: str,
) -> _M03RV16CompletionAttestationJournal:
    if recovery_mode not in {
        "published-by-controller",
        "recovered-after-final-publication",
    }:
        raise M03RV16SeadragonControllerError(
            "V16 completion attestation recovery mode drifted"
        )
    path = _completion_attestation_journal_path(
        authority_root,
        phase=phase,
        job_uid=job_uid,
        completion_index=completion_index,
    )
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_COMPLETION_ATTESTATION_SCHEMA,
        "phase": phase,
        "job_uid": job_uid,
        "job_name": job_name,
        "launch_authority_receipt_sha256": launch_receipt_sha256,
        "storage_semantics_receipt_sha256": storage_receipt_sha256,
        "completion_index": completion_index,
        "pod_uid": observation.pod_uid,
        "pod_name": observation.pod_name,
        "node_name": observation.node_name,
        "attestation_relative_path": row.relative_path,
        "attestation_file_sha256": row.file_sha256,
        "attestation_receipt_sha256": row.receipt_sha256,
        "controller_transaction_receipt_sha256": (
            row.controller_transaction_receipt_sha256
        ),
        "recovery_mode": recovery_mode,
    }
    receipt = semantic_sha256(unsigned)
    file_sha = _write_or_validate_json(
        path, {**unsigned, "receipt_sha256": receipt}
    )
    return _M03RV16CompletionAttestationJournal(
        row=row,
        pod_uid=observation.pod_uid,
        file_sha256=file_sha,
        receipt_sha256=receipt,
        recovery_mode=recovery_mode,
    )


def _load_completion_attestation_journal(
    *,
    authority_root: str | Path,
    phase: str,
    job_uid: str,
    job_name: str,
    launch_receipt_sha256: str,
    storage_receipt_sha256: str,
    completion_index: int,
    pod: Mapping[str, Any],
    observation: M03RV16PodObservation,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    admission: M03RV16AdmittedJobAuthority,
    launch: Any,
) -> _M03RV16CompletionAttestationJournal:
    path = _completion_attestation_journal_path(
        authority_root,
        phase=phase,
        job_uid=job_uid,
        completion_index=completion_index,
    )
    value, file_sha = _read_self_receipted_json(path)
    relative_path = launch.pod_runtime_attestation_relative_path(completion_index)
    recovery_mode = _text(
        "completion attestation recovery mode", value.get("recovery_mode")
    )
    if (
        value.get("schema")
        != M03R_V16_CONTROLLER_COMPLETION_ATTESTATION_SCHEMA
        or value.get("phase") != phase
        or value.get("job_uid") != job_uid
        or value.get("job_name") != job_name
        or value.get("launch_authority_receipt_sha256")
        != launch_receipt_sha256
        or value.get("storage_semantics_receipt_sha256")
        != storage_receipt_sha256
        or value.get("completion_index") != completion_index
        or value.get("pod_uid") != observation.pod_uid
        or value.get("pod_name") != observation.pod_name
        or value.get("node_name") != observation.node_name
        or value.get("attestation_relative_path") != relative_path
        or recovery_mode
        not in {"published-by-controller", "recovered-after-final-publication"}
    ):
        raise M03RV16SeadragonControllerError(
            "V16 completion attestation journal drifted"
        )
    attestation_file_sha = _text(
        "Pod attestation file SHA", value.get("attestation_file_sha256")
    )
    attestation_receipt = _text(
        "Pod attestation receipt", value.get("attestation_receipt_sha256")
    )
    attestation = load_m03r_v16_pod_runtime_attestation(
        Path(authority_root) / relative_path,
        expected_file_sha256=attestation_file_sha,
        expected_receipt_sha256=attestation_receipt,
        package=package,
        authorization=authorization,
        admission=admission,
        launch=launch,
        expected_completion_index=completion_index,
        expected_output_root_sha256=_output_root_sha256(
            package,
            phase=phase,
            completion_index=completion_index,
        ),
        current_pod_uid=observation.pod_uid,
        current_pod_name=observation.pod_name,
        current_node_name=observation.node_name,
        expected_relative_path=relative_path,
    )
    annotations = m03r_v16_pod_runtime_attestation_annotations(attestation)
    observed_annotations = _metadata(pod).get("annotations", {})
    if not isinstance(observed_annotations, dict) or any(
        observed_annotations.get(key) != expected
        for key, expected in annotations.items()
    ):
        raise M03RV16SeadragonControllerError(
            "V16 recovered Pod attestation annotations drifted"
        )
    transaction_receipt = _text(
        "Pod attestation transaction receipt",
        value.get("controller_transaction_receipt_sha256"),
    )
    row = M03RV16PublishedPodAttestation(
        final_path=Path(authority_root) / relative_path,
        file_sha256=attestation_file_sha,
        receipt_sha256=attestation.receipt_sha256,
        relative_path=relative_path,
        patched_annotations=annotations,
        controller_transaction_receipt_sha256=transaction_receipt,
    )
    return _M03RV16CompletionAttestationJournal(
        row=row,
        pod_uid=observation.pod_uid,
        file_sha256=file_sha,
        receipt_sha256=_text(
            "completion attestation journal receipt", value.get("receipt_sha256")
        ),
        recovery_mode=recovery_mode,
    )


def _recover_published_attestation_without_journal(
    *,
    authority_root: str | Path,
    phase: str,
    job_uid: str,
    job_name: str,
    completion_index: int,
    pod: Mapping[str, Any],
    observation: M03RV16PodObservation,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    admission: M03RV16AdmittedJobAuthority,
    launch: Any,
    storage_receipt_sha256: str,
) -> _M03RV16CompletionAttestationJournal:
    annotations = _metadata(pod).get("annotations", {})
    if not isinstance(annotations, dict):
        raise M03RV16SeadragonControllerError(
            "V16 recovered Pod annotations are absent"
        )
    relative_path = launch.pod_runtime_attestation_relative_path(completion_index)
    file_sha = _text(
        "Pod attestation file SHA",
        annotations.get(_POD_ATTESTATION_FILE_SHA_ANNOTATION),
    )
    receipt = _text(
        "Pod attestation receipt",
        annotations.get(_POD_ATTESTATION_RECEIPT_ANNOTATION),
    )
    if annotations.get(_POD_ATTESTATION_PATH_ANNOTATION) != relative_path:
        raise M03RV16SeadragonControllerError(
            "V16 recovered Pod attestation path drifted"
        )
    attestation = load_m03r_v16_pod_runtime_attestation(
        Path(authority_root) / relative_path,
        expected_file_sha256=file_sha,
        expected_receipt_sha256=receipt,
        package=package,
        authorization=authorization,
        admission=admission,
        launch=launch,
        expected_completion_index=completion_index,
        expected_output_root_sha256=_output_root_sha256(
            package,
            phase=phase,
            completion_index=completion_index,
        ),
        current_pod_uid=observation.pod_uid,
        current_pod_name=observation.pod_name,
        current_node_name=observation.node_name,
        expected_relative_path=relative_path,
    )
    expected_annotations = m03r_v16_pod_runtime_attestation_annotations(attestation)
    if any(
        annotations.get(key) != expected
        for key, expected in expected_annotations.items()
    ):
        raise M03RV16SeadragonControllerError(
            "V16 recovered immutable Pod attestation annotations drifted"
        )
    recovery_transaction = semantic_sha256(
        {
            "schema": M03R_V16_CONTROLLER_COMPLETION_ATTESTATION_SCHEMA,
            "recovered_after_final_publication": True,
            "phase": phase,
            "job_uid": job_uid,
            "completion_index": completion_index,
            "pod_uid": observation.pod_uid,
            "attestation_file_sha256": file_sha,
            "attestation_receipt_sha256": receipt,
            "recovery_observed_pod_resource_version": (
                observation.observed_pod_resource_version
            ),
        }
    )
    row = M03RV16PublishedPodAttestation(
        final_path=Path(authority_root) / relative_path,
        file_sha256=file_sha,
        receipt_sha256=receipt,
        relative_path=relative_path,
        patched_annotations=expected_annotations,
        controller_transaction_receipt_sha256=recovery_transaction,
    )
    return _write_completion_attestation_journal(
        authority_root=authority_root,
        phase=phase,
        job_uid=job_uid,
        job_name=job_name,
        launch_receipt_sha256=launch.receipt_sha256,
        storage_receipt_sha256=storage_receipt_sha256,
        completion_index=completion_index,
        observation=observation,
        row=row,
        recovery_mode="recovered-after-final-publication",
    )


def resume_and_attest_m03r_v16_job(
    *,
    transport: M03RV16KubernetesTransport,
    controller_admission: M03RV16ControllerAdmission,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    storage_evidence: M03RV16StorageSemanticsEvidence,
    authority_root: str | Path,
    storage_authority_identity_root: str | Path = M03R_V16_POD_AUTHORITY_ROOT,
    storage_observer_identity_root: str | Path = (
        M03R_V16_POD_AUTHORITY_OBSERVER_ROOT
    ),
    timeout_seconds: float = 1800.0,
    poll_seconds: float = 2.0,
) -> M03RV16ControllerAttestations:
    """Resume one exact Job and release each observable completion immediately."""

    bound = controller_admission.rendered
    admission = controller_admission.admission
    launch = bound.launch_authority
    if launch is None:
        raise M03RV16SeadragonControllerError("V16 launch authority is absent")
    job_name = admission.job_name
    for _attempt in range(5):
        current_job = transport.invoke(
            ("get", "job", job_name, "--output", "json")
        )
        if current_job is None:
            raise M03RV16SeadragonControllerError("V16 Job resume target vanished")
        current_spec = current_job.get("spec")
        is_suspended = (
            isinstance(current_spec, dict)
            and current_spec.get("suspend") is True
        )
        _, current_resource_version = _validate_exact_job(
            current_job,
            bound,
            expected_uid=admission.job_uid,
            expected_suspended=is_suspended,
        )
        if not is_suspended:
            break
        try:
            resumed = transport.invoke(
                (
                    "patch", "job", job_name, "--type", "merge",
                    "--patch-file", "/dev/stdin", "--output", "json",
                ),
                payload={
                    "metadata": {"resourceVersion": current_resource_version},
                    "spec": {"suspend": False},
                },
            )
        except M03RV16KubernetesConflictError:
            continue
        except M03RV16SeadragonControllerError:
            # A lost response may follow a successful API mutation.  Reconcile
            # the exact UID before deciding whether the resume failed.
            observed = transport.invoke(
                ("get", "job", job_name, "--output", "json"),
                allow_not_found=True,
            )
            if observed is None:
                raise
            observed_spec = observed.get("spec")
            if not (
                isinstance(observed_spec, dict)
                and observed_spec.get("suspend") is False
            ):
                raise
            resumed = observed
        if resumed is None:
            raise M03RV16SeadragonControllerError("V16 Job resume vanished")
        _validate_exact_job(
            resumed,
            bound,
            expected_uid=admission.job_uid,
            expected_suspended=False,
        )
        break
    else:
        raise M03RV16SeadragonControllerError(
            "V16 Job resume resourceVersion conflicts did not converge"
        )

    receipt_path = (
        Path(authority_root) / f"{bound.mode}-controller-attestations.json"
    )
    if receipt_path.exists():
        prior, _ = _read_self_receipted_json(receipt_path)
        if (
            prior.get("schema") != M03R_V16_CONTROLLER_ATTESTATION_SCHEMA
            or prior.get("phase") != bound.mode
            or prior.get("job_uid") != admission.job_uid
            or prior.get("job_name") != job_name
            or prior.get("launch_authority_receipt_sha256")
            != launch.receipt_sha256
            or prior.get("storage_semantics_receipt_sha256")
            != storage_evidence.receipt_sha256
            or tuple(prior.get("completion_indices", ()))
            != tuple(range(bound.completions))
        ):
            raise M03RV16SeadragonControllerError(
                "V16 prior controller attestation aggregate drifted"
            )
        pods_by_index: dict[int, tuple[dict[str, Any], M03RV16PodObservation]] = {}
        for pod in _list_exact_job_pods(transport, job_name=job_name):
            observation = _observe_pod(pod, admission=admission)
            pods_by_index[observation.completion_index] = (pod, observation)
        if tuple(sorted(pods_by_index)) != tuple(range(bound.completions)):
            raise M03RV16SeadragonControllerError(
                "V16 recovered controller lacks its exact Pod inventory"
            )
        file_rows = tuple(prior.get("attestation_file_sha256", ()))
        receipt_rows = tuple(prior.get("attestation_receipt_sha256", ()))
        transaction_rows = tuple(
            prior.get("controller_transaction_receipt_sha256", ())
        )
        pod_uid_rows = tuple(prior.get("pod_uids", ()))
        journal_file_rows = tuple(
            prior.get("completion_journal_file_sha256", ())
        )
        journal_receipt_rows = tuple(
            prior.get("completion_journal_receipt_sha256", ())
        )
        journal_recovery_modes = tuple(
            prior.get("completion_journal_recovery_mode", ())
        )
        if not all(
            len(rows) == bound.completions
            for rows in (
                file_rows,
                receipt_rows,
                transaction_rows,
                pod_uid_rows,
                journal_file_rows,
                journal_receipt_rows,
                journal_recovery_modes,
            )
        ):
            raise M03RV16SeadragonControllerError(
                "V16 recovered attestation inventory drifted"
            )
        recovered_rows: list[M03RV16PublishedPodAttestation] = []
        for index in range(bound.completions):
            pod, observation = pods_by_index[index]
            journal = _load_completion_attestation_journal(
                authority_root=authority_root,
                phase=bound.mode,
                job_uid=admission.job_uid,
                job_name=job_name,
                launch_receipt_sha256=launch.receipt_sha256,
                storage_receipt_sha256=storage_evidence.receipt_sha256,
                completion_index=index,
                pod=pod,
                observation=observation,
                package=package,
                authorization=authorization,
                admission=admission,
                launch=launch,
            )
            if (
                observation.pod_uid != pod_uid_rows[index]
                or journal.file_sha256 != journal_file_rows[index]
                or journal.receipt_sha256 != journal_receipt_rows[index]
                or journal.row.file_sha256 != file_rows[index]
                or journal.row.receipt_sha256 != receipt_rows[index]
                or journal.row.controller_transaction_receipt_sha256
                != transaction_rows[index]
                or journal.recovery_mode != journal_recovery_modes[index]
            ):
                raise M03RV16SeadragonControllerError(
                    "V16 recovered Pod attestation aggregate drifted"
                )
            recovered_rows.append(journal.row)
        return M03RV16ControllerAttestations(
            rows=tuple(recovered_rows),
            controller_receipt_path=receipt_path,
        )

    deadline = time.monotonic() + timeout_seconds
    published: dict[int, M03RV16PublishedPodAttestation] = {}
    published_pod_uids: dict[int, str] = {}
    published_journals: dict[int, _M03RV16CompletionAttestationJournal] = {}
    conflicts: dict[int, int] = {}
    while len(published) != bound.completions:
        observed_this_poll: dict[int, M03RV16PodObservation] = {}
        for pod in _list_exact_job_pods(transport, job_name=job_name):
            try:
                pod_observation = _observe_pod(pod, admission=admission)
            except M03RV16SeadragonControllerError:
                continue
            index = pod_observation.completion_index
            if index in observed_this_poll:
                raise M03RV16SeadragonControllerError(
                    "V16 completion has multiple observable Pods"
                )
            observed_this_poll[index] = pod_observation
            if index in published:
                if published_pod_uids[index] != pod_observation.pod_uid:
                    raise M03RV16SeadragonControllerError(
                        "V16 attested completion was replaced by another Pod"
                    )
                continue

            journal_path = _completion_attestation_journal_path(
                authority_root,
                phase=bound.mode,
                job_uid=admission.job_uid,
                completion_index=index,
            )
            relative_path = launch.pod_runtime_attestation_relative_path(index)
            final_attestation_path = Path(authority_root) / relative_path
            if journal_path.exists():
                journal = _load_completion_attestation_journal(
                    authority_root=authority_root,
                    phase=bound.mode,
                    job_uid=admission.job_uid,
                    job_name=job_name,
                    launch_receipt_sha256=launch.receipt_sha256,
                    storage_receipt_sha256=storage_evidence.receipt_sha256,
                    completion_index=index,
                    pod=pod,
                    observation=pod_observation,
                    package=package,
                    authorization=authorization,
                    admission=admission,
                    launch=launch,
                )
                published[index] = journal.row
                published_pod_uids[index] = journal.pod_uid
                published_journals[index] = journal
                continue
            if final_attestation_path.exists():
                journal = _recover_published_attestation_without_journal(
                    authority_root=authority_root,
                    phase=bound.mode,
                    job_uid=admission.job_uid,
                    job_name=job_name,
                    completion_index=index,
                    pod=pod,
                    observation=pod_observation,
                    package=package,
                    authorization=authorization,
                    admission=admission,
                    launch=launch,
                    storage_receipt_sha256=storage_evidence.receipt_sha256,
                )
                published[index] = journal.row
                published_pod_uids[index] = journal.pod_uid
                published_journals[index] = journal
                continue

            def patch_annotations(
                precondition: object,
                values: Mapping[str, str],
                *,
                pod_name: str = pod_observation.pod_name,
                expected_pod_uid: str = pod_observation.pod_uid,
            ) -> None:
                resource_version = getattr(
                    precondition, "pod_resource_version", None
                )
                pod_uid = getattr(precondition, "pod_uid", None)
                if pod_uid != expected_pod_uid:
                    raise M03RV16SeadragonControllerError(
                        "V16 Pod patch UID precondition drifted"
                    )
                patched = transport.invoke(
                    (
                        "patch", "pod", pod_name, "--type", "merge",
                        "--patch-file", "/dev/stdin", "--output", "json",
                    ),
                    payload={
                        "metadata": {
                            "resourceVersion": resource_version,
                            "annotations": dict(values),
                        }
                    },
                )
                if patched is None or _metadata(patched).get("uid") != pod_uid:
                    raise M03RV16SeadragonControllerError(
                        "V16 Pod changed across annotation patch"
                    )

            def read_annotations(
                precondition: object,
                *,
                pod_name: str = pod_observation.pod_name,
            ) -> M03RV16PodAnnotationReadback:
                current = transport.invoke(
                    ("get", "pod", pod_name, "--output", "json")
                )
                if current is None:
                    raise M03RV16SeadragonControllerError("V16 Pod vanished")
                metadata = _metadata(current)
                expected_uid = getattr(precondition, "pod_uid", None)
                if metadata.get("uid") != expected_uid:
                    raise M03RV16SeadragonControllerError(
                        "V16 Pod UID changed before annotation readback"
                    )
                annotations = metadata.get("annotations")
                if not isinstance(annotations, dict):
                    raise M03RV16SeadragonControllerError(
                        "V16 Pod annotations vanished"
                    )
                return M03RV16PodAnnotationReadback(
                    pod_uid=_text("Pod UID", metadata.get("uid")),
                    pod_resource_version=_text(
                        "Pod resourceVersion", metadata.get("resourceVersion")
                    ),
                    annotations={
                        str(key): str(value) for key, value in annotations.items()
                    },
                )

            try:
                row = publish_m03r_v16_pod_runtime_attestation_after_annotation_patch(
                    package=package,
                    authorization=authorization,
                    admission=admission,
                    launch=launch,
                    observation=pod_observation,
                    output_root_sha256=_output_root_sha256(
                        package,
                        phase=bound.mode,
                        completion_index=index,
                    ),
                    authority_root=authority_root,
                    observer_root=storage_observer_identity_root,
                    storage_evidence=storage_evidence,
                    storage_authority_identity_root=(
                        storage_authority_identity_root
                    ),
                    storage_observer_identity_root=(
                        storage_observer_identity_root
                    ),
                    patch_annotations=patch_annotations,
                    read_annotations=read_annotations,
                )
            except M03RV16KubernetesConflictError:
                conflicts[index] = conflicts.get(index, 0) + 1
                if conflicts[index] >= 5:
                    raise M03RV16SeadragonControllerError(
                        "V16 Pod annotation conflicts did not converge"
                    )
                continue
            journal = _write_completion_attestation_journal(
                authority_root=authority_root,
                phase=bound.mode,
                job_uid=admission.job_uid,
                job_name=job_name,
                launch_receipt_sha256=launch.receipt_sha256,
                storage_receipt_sha256=storage_evidence.receipt_sha256,
                completion_index=index,
                observation=pod_observation,
                row=row,
                recovery_mode="published-by-controller",
            )
            published[index] = journal.row
            published_pod_uids[index] = journal.pod_uid
            published_journals[index] = journal

        if len(published) == bound.completions:
            break
        if time.monotonic() >= deadline:
            raise M03RV16SeadragonControllerError(
                "V16 exact Pod inventory did not become attestable"
            )
        time.sleep(poll_seconds)
    published_rows = tuple(published[index] for index in range(bound.completions))
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_ATTESTATION_SCHEMA,
        "phase": bound.mode,
        "job_uid": admission.job_uid,
        "job_name": job_name,
        "launch_authority_receipt_sha256": launch.receipt_sha256,
        "storage_semantics_receipt_sha256": storage_evidence.receipt_sha256,
        "scheduling_policy": M03R_V16_SCHEDULING_POLICY,
        "attestation_release_mode": "as-each-completion-becomes-observable",
        "gang_scheduling_required": False,
        "completion_indices": list(range(bound.completions)),
        "pod_uids": [published_pod_uids[index] for index in range(bound.completions)],
        "attestation_file_sha256": [row.file_sha256 for row in published_rows],
        "attestation_receipt_sha256": [
            row.receipt_sha256 for row in published_rows
        ],
        "controller_transaction_receipt_sha256": [
            row.controller_transaction_receipt_sha256 for row in published_rows
        ],
        "completion_journal_file_sha256": [
            published_journals[index].file_sha256
            for index in range(bound.completions)
        ],
        "completion_journal_receipt_sha256": [
            published_journals[index].receipt_sha256
            for index in range(bound.completions)
        ],
        "completion_journal_recovery_mode": [
            published_journals[index].recovery_mode
            for index in range(bound.completions)
        ],
    }
    value = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    _write_or_validate_json(receipt_path, value)
    return M03RV16ControllerAttestations(
        rows=published_rows,
        controller_receipt_path=receipt_path,
    )


def snapshot_m03r_v16_exact_job(
    *,
    transport: M03RV16KubernetesTransport,
    admission: M03RV16ExactJobAuthority,
    package: M03RV16PackagePlan,
) -> dict[str, Any]:
    """Return one compact snapshot for exactly one admitted Job UID."""

    job = transport.invoke(
        ("get", "job", admission.job_name, "--output", "json"),
        allow_not_found=True,
    )
    if job is None:
        return {
            "job_name": admission.job_name,
            "job_uid": admission.job_uid,
            "state": "absent",
            "pods": [],
        }
    metadata = _metadata(job)
    if metadata.get("uid") != admission.job_uid:
        raise M03RV16SeadragonControllerError(
            "V16 Job name now refers to a different UID"
        )
    raw_status = job.get("status")
    status: dict[str, Any] = (
        dict(raw_status) if isinstance(raw_status, dict) else {}
    )
    pods: list[dict[str, Any]] = []
    for pod in _list_exact_job_pods(transport, job_name=admission.job_name):
        pod_metadata = _metadata(pod)
        owners = pod_metadata.get("ownerReferences", [])
        if not any(
            isinstance(owner, dict)
            and owner.get("uid") == admission.job_uid
            and owner.get("controller") is True
            for owner in owners
        ):
            raise M03RV16SeadragonControllerError(
                "V16 exact Job selector returned a foreign Pod"
            )
        raw_pod_status = pod.get("status")
        raw_pod_spec = pod.get("spec")
        pod_status: dict[str, Any] = (
            dict(raw_pod_status) if isinstance(raw_pod_status, dict) else {}
        )
        pod_spec: dict[str, Any] = (
            dict(raw_pod_spec) if isinstance(raw_pod_spec, dict) else {}
        )
        containers = pod_spec.get("containers", [])
        container_statuses = pod_status.get("containerStatuses", [])
        main_spec = containers[0] if isinstance(containers, list) and containers else {}
        main_status = (
            container_statuses[0]
            if isinstance(container_statuses, list) and container_statuses
            else {}
        )
        main_image_id = main_status.get("imageID")
        main_image_digest = (
            _runtime_image_digest(main_image_id)
            if isinstance(main_image_id, str) and main_image_id
            else None
        )
        if (
            main_spec.get("image") != package.artifacts.image_reference
            or (
                main_image_digest is not None
                and main_image_digest
                != package.artifacts.image_digest_sha256
            )
        ):
            raise M03RV16SeadragonControllerError(
                "V16 main scientific container image drifted"
            )
        pods.append(
            {
                "completion_index": _pod_completion_index(pod),
                "pod_uid": pod_metadata.get("uid"),
                "pod_name": pod_metadata.get("name"),
                "phase": pod_status.get("phase"),
                "node_name": pod_spec.get("nodeName"),
                "main_container_image_id": main_image_id,
                "main_container_image_digest": main_image_digest,
            }
        )
    return {
        "job_name": admission.job_name,
        "job_uid": admission.job_uid,
        "state": "present",
        "resource_version": metadata.get("resourceVersion"),
        "active": int(status.get("active", 0)),
        "ready": int(status.get("ready", 0)),
        "succeeded": int(status.get("succeeded", 0)),
        "failed": int(status.get("failed", 0)),
        "pods": sorted(pods, key=lambda row: int(row["completion_index"])),
    }


def wait_for_m03r_v16_exact_job_terminal(
    *,
    transport: M03RV16KubernetesTransport,
    admission: M03RV16ExactJobAuthority,
    package: M03RV16PackagePlan,
    authority_root: str | Path,
    timeout_seconds: float,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    """Wait for one exact Job and publish its compact terminal snapshot."""

    path = Path(authority_root) / f"{admission.phase}-controller-terminal.json"
    if path.exists():
        prior, _ = _read_self_receipted_json(path)
        prior_snapshot = prior.get("snapshot")
        if (
            prior.get("schema") != M03R_V16_CONTROLLER_TERMINAL_SCHEMA
            or prior.get("phase") != admission.phase
            or prior.get("job_uid") != admission.job_uid
            or prior.get("job_name") != admission.job_name
            or prior.get("package_plan_sha256") != package.package_plan_sha256
            or prior.get("outcome") not in {"complete", "failed"}
            or prior.get("scheduling_policy") != M03R_V16_SCHEDULING_POLICY
            or prior.get("gang_scheduling_required") is not False
            or not isinstance(prior_snapshot, dict)
        ):
            raise M03RV16SeadragonControllerError(
                "V16 prior controller terminal drifted"
            )
        if prior.get("outcome") == "complete":
            prior_pods = prior_snapshot.get("pods")
            if (
                prior_snapshot.get("succeeded") != admission.completions
                or prior_snapshot.get("failed") != 0
                or not isinstance(prior_pods, list)
                or len(prior_pods) != admission.completions
                or any(
                    not isinstance(row, dict)
                    or row.get("main_container_image_digest")
                    != package.artifacts.image_digest_sha256
                    for row in prior_pods
                )
            ):
                raise M03RV16SeadragonControllerError(
                    "V16 prior completed controller terminal drifted"
                )
        return prior
    deadline = time.monotonic() + timeout_seconds
    while True:
        snapshot = snapshot_m03r_v16_exact_job(
            transport=transport,
            admission=admission,
            package=package,
        )
        if snapshot["state"] == "absent":
            raise M03RV16SeadragonControllerError(
                "V16 admitted Job disappeared before terminal"
            )
        if snapshot["failed"]:
            outcome = "failed"
            break
        if snapshot["succeeded"] == admission.completions:
            if len(snapshot["pods"]) != admission.completions or any(
                row.get("main_container_image_digest")
                != package.artifacts.image_digest_sha256
                for row in snapshot["pods"]
            ):
                raise M03RV16SeadragonControllerError(
                    "V16 completed Pod lacks matching main-container image evidence"
                )
            outcome = "complete"
            break
        if time.monotonic() >= deadline:
            raise M03RV16SeadragonControllerError(
                "V16 exact Job did not reach terminal state"
            )
        time.sleep(poll_seconds)
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_TERMINAL_SCHEMA,
        "phase": admission.phase,
        "package_plan_sha256": package.package_plan_sha256,
        "admitted_job_authority_receipt_sha256": getattr(
            admission, "receipt_sha256", None
        ),
        "job_uid": admission.job_uid,
        "job_name": admission.job_name,
        "outcome": outcome,
        "scheduling_policy": M03R_V16_SCHEDULING_POLICY,
        "gang_scheduling_required": False,
        "snapshot": snapshot,
    }
    value = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    _write_or_validate_json(path, value)
    return value


def _uid_owned_pods(
    transport: M03RV16KubernetesTransport,
    *,
    job_name: str,
    job_uid: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for pod in _list_exact_job_pods(transport, job_name=job_name):
        owners = _metadata(pod).get("ownerReferences", ())
        if any(
            isinstance(owner, dict)
            and owner.get("uid") == job_uid
            and owner.get("controller") is True
            for owner in owners
        ):
            rows.append(pod)
    return tuple(rows)


def cleanup_m03r_v16_exact_job(
    *,
    transport: M03RV16KubernetesTransport,
    controller_admission: M03RV16ControllerAdmission,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    authority_root: str | Path,
    cleanup_authorized: bool,
    timeout_seconds: float = 900.0,
    poll_seconds: float = 2.0,
) -> dict[str, Any]:
    """Delete one authority-bound Job and await foreground garbage collection."""

    if not cleanup_authorized:
        raise M03RV16SeadragonControllerError(
            "V16 exact cleanup lacks explicit authorization"
        )
    package.validate()
    authorization.validate(package)
    admission = controller_admission.admission
    bound = controller_admission.rendered
    admission.validate_for(
        package,
        authorization,
        expected_phase=bound.mode,
        expected_job_contract_sha256=bound.job_contract_sha256,
        expected_pod_contract_sha256=bound.pod_contract_sha256,
    )
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise M03RV16SeadragonControllerError(
            "V16 cleanup wait intervals must be positive"
        )
    cleanup_complete_path = (
        Path(authority_root)
        / f"{admission.phase}-controller-cleanup-complete.json"
    )
    if cleanup_complete_path.exists():
        prior, _ = _read_self_receipted_json(cleanup_complete_path)
        if (
            prior.get("schema") != M03R_V16_CONTROLLER_CLEANUP_SCHEMA
            or prior.get("phase") != admission.phase
            or prior.get("job_uid") != admission.job_uid
            or prior.get("job_name") != admission.job_name
            or prior.get("cleanup_complete") is not True
            or prior.get("job_absent") is not True
            or prior.get("pod_inventory_empty") is not True
        ):
            raise M03RV16SeadragonControllerError(
                "V16 prior completed cleanup terminal drifted"
            )
        return prior
    current = transport.invoke(
        ("get", "job", admission.job_name, "--output", "json"),
        allow_not_found=True,
    )
    resource_version: str | None = None
    delete_submitted = False
    target_already_absent = current is None
    if current is not None:
        metadata = _metadata(current)
        if metadata.get("uid") == admission.job_uid:
            spec = current.get("spec")
            suspended = isinstance(spec, dict) and spec.get("suspend") is True
            _, resource_version = _validate_exact_job(
                current,
                bound,
                expected_uid=admission.job_uid,
                expected_suspended=suspended,
            )
            expected_annotations = _metadata(bound.manifest).get("annotations", {})
            observed_annotations = metadata.get("annotations", {})
            if not isinstance(observed_annotations, dict) or any(
                observed_annotations.get(key) != value
                for key, value in expected_annotations.items()
            ):
                raise M03RV16SeadragonControllerError(
                    "V16 cleanup target lacks the admitted launch authority"
                )
            if metadata.get("deletionTimestamp") is None:
                response = transport.invoke(
                    (
                        "delete", "--raw",
                        (
                            f"/apis/batch/v1/namespaces/{M03R_V16_NAMESPACE}/jobs/"
                            f"{admission.job_name}"
                        ),
                        "-f", "-",
                    ),
                    payload={
                        "apiVersion": "v1",
                        "kind": "DeleteOptions",
                        "propagationPolicy": "Foreground",
                        "preconditions": {
                            "uid": admission.job_uid,
                            "resourceVersion": resource_version,
                        },
                    },
                )
                if response is None:
                    raise M03RV16SeadragonControllerError(
                        "V16 cleanup response vanished"
                    )
                delete_submitted = True
        else:
            # The exact admitted UID is already absent.  A replacement with
            # the same name is outside scope and is never mutated.
            target_already_absent = True

    deadline = time.monotonic() + timeout_seconds
    complete = False
    while True:
        observed = transport.invoke(
            ("get", "job", admission.job_name, "--output", "json"),
            allow_not_found=True,
        )
        original_present = (
            observed is not None
            and _metadata(observed).get("uid") == admission.job_uid
        )
        owned_pods = _uid_owned_pods(
            transport,
            job_name=admission.job_name,
            job_uid=admission.job_uid,
        )
        if not original_present and not owned_pods:
            complete = True
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_seconds)
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_CLEANUP_SCHEMA,
        "phase": admission.phase,
        "job_uid": admission.job_uid,
        "job_name": admission.job_name,
        "resource_version": resource_version,
        "controller_admission_file_sha256": _file_sha256(
            controller_admission.controller_receipt_path
        ),
        "controller_admission_receipt_sha256": _text(
            "controller admission receipt",
            _read_self_receipted_json(
                controller_admission.controller_receipt_path
            )[0].get("receipt_sha256"),
        ),
        "admitted_job_authority_receipt_sha256": admission.receipt_sha256,
        "launch_authority_receipt_sha256": (
            None
            if bound.launch_authority is None
            else bound.launch_authority.receipt_sha256
        ),
        "delete_submitted": delete_submitted,
        "target_already_absent": target_already_absent,
        "job_absent": complete,
        "pod_inventory_empty": complete,
        "cleanup_complete": complete,
    }
    attempt_receipt = semantic_sha256(unsigned)
    value = {**unsigned, "receipt_sha256": attempt_receipt}
    attempt_path = (
        Path(authority_root)
        / "cleanup-attempts"
        / admission.phase
        / admission.job_uid
        / f"attempt-{time.time_ns()}-{secrets.token_hex(8)}.json"
    )
    attempt_file_sha = _write_create_only_json(attempt_path, value)
    if not complete:
        return {
            **value,
            "cleanup_attempt_relative_path": str(
                attempt_path.relative_to(Path(authority_root))
            ),
            "cleanup_attempt_file_sha256": attempt_file_sha,
        }
    complete_unsigned = {
        **unsigned,
        "cleanup_attempt_relative_path": str(
            attempt_path.relative_to(Path(authority_root))
        ),
        "cleanup_attempt_file_sha256": attempt_file_sha,
        "cleanup_attempt_receipt_sha256": attempt_receipt,
    }
    complete_value = {
        **complete_unsigned,
        "receipt_sha256": semantic_sha256(complete_unsigned),
    }
    _write_or_validate_json(cleanup_complete_path, complete_value)
    return complete_value


def _add_package_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument(
        "--execution-authorization-file-sha256",
        required=True,
    )


def _load_cli_package(
    args: argparse.Namespace,
) -> tuple[M03RV16PackagePlan, M03RV16ExecutionAuthorization]:
    package = load_m03r_v16_package_plan(
        args.package_plan,
        expected_file_sha256=args.package_plan_file_sha256,
    )
    authorization = load_m03r_v16_execution_authorization(
        args.execution_authorization,
        expected_file_sha256=args.execution_authorization_file_sha256,
        package=package,
    )
    return package, authorization


def _load_cli_controller_admission(
    args: argparse.Namespace,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
) -> M03RV16ControllerAdmission:
    rendered = load_m03r_v16_controller_job_plan(
        args.controller_job_plan,
        expected_file_sha256=args.controller_job_plan_file_sha256,
    )
    storage = load_m03r_v16_storage_semantics_evidence(
        args.storage_semantics,
        expected_file_sha256=args.storage_semantics_file_sha256,
        authority_root=args.storage_authority_identity_root,
        observer_root=args.storage_observer_identity_root,
    )
    return load_m03r_v16_controller_admission(
        args.controller_admission,
        expected_file_sha256=args.controller_admission_file_sha256,
        rendered=rendered,
        package=package,
        authorization=authorization,
        storage_evidence=storage,
        storage_evidence_file_sha256=args.storage_semantics_file_sha256,
        storage_authority_identity_root=args.storage_authority_identity_root,
        storage_observer_identity_root=args.storage_observer_identity_root,
    )


def _execute_controller_job(args: argparse.Namespace) -> dict[str, Any]:
    package, authorization = _load_cli_package(args)
    rendered = load_m03r_v16_controller_job_plan(
        args.controller_job_plan,
        expected_file_sha256=args.controller_job_plan_file_sha256,
    )
    if (
        rendered.package_plan_sha256 != package.package_plan_sha256
        or rendered.execution_authorization_receipt_sha256
        != authorization.receipt_sha256
        or rendered.package_plan_file_sha256
        != args.package_plan_file_sha256
        or rendered.execution_authorization_file_sha256
        != args.execution_authorization_file_sha256
    ):
        raise M03RV16SeadragonControllerError(
            "V16 controller Job plan differs from the exact package authority"
        )
    transport = M03RV16KubectlTransport()
    authority_root = Path(args.authority_root)
    if rendered.mode in {"static", "storage"}:
        job = launch_m03r_v16_zero_gpu_gate(
            transport=transport,
            rendered=rendered,
            authority_root=authority_root,
        )
        return wait_for_m03r_v16_exact_job_terminal(
            transport=transport,
            admission=job,
            package=package,
            authority_root=authority_root,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    if (
        args.storage_semantics is None
        or args.storage_semantics_file_sha256 is None
        or args.source_tree_root_sha256 is None
    ):
        raise M03RV16SeadragonControllerError(
            "V16 scientific execution lacks storage or source authority"
        )
    storage = load_m03r_v16_storage_semantics_evidence(
        args.storage_semantics,
        expected_file_sha256=args.storage_semantics_file_sha256,
        authority_root=args.storage_authority_identity_root,
        observer_root=args.storage_observer_identity_root,
    )
    labels = _metadata(rendered.manifest).get("labels")
    if not isinstance(labels, dict):
        raise M03RV16SeadragonControllerError(
            "V16 rendered Job lacks a run identity"
        )
    run_id = _text("controller run ID", labels.get("rl-quant/run-id"))
    admitted = admit_m03r_v16_suspended_job(
        transport=transport,
        rendered=rendered,
        package=package,
        authorization=authorization,
        storage_evidence=storage,
        storage_evidence_file_sha256=args.storage_semantics_file_sha256,
        authority_root=authority_root,
        source_tree_root_sha256=args.source_tree_root_sha256,
        run_id=run_id,
        storage_authority_identity_root=args.storage_authority_identity_root,
        storage_observer_identity_root=args.storage_observer_identity_root,
    )
    resume_and_attest_m03r_v16_job(
        transport=transport,
        controller_admission=admitted,
        package=package,
        authorization=authorization,
        storage_evidence=storage,
        authority_root=authority_root,
        storage_authority_identity_root=args.storage_authority_identity_root,
        storage_observer_identity_root=args.storage_observer_identity_root,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    return wait_for_m03r_v16_exact_job_terminal(
        transport=transport,
        admission=admitted.admission,
        package=package,
        authority_root=authority_root,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    execute = subparsers.add_parser(
        "execute",
        help="admit, attest, and supervise one sealed exact Job plan",
    )
    _add_package_arguments(execute)
    execute.add_argument("--controller-job-plan", required=True)
    execute.add_argument("--controller-job-plan-file-sha256", required=True)
    execute.add_argument("--authority-root", required=True)
    execute.add_argument("--source-tree-root-sha256")
    execute.add_argument("--storage-semantics")
    execute.add_argument("--storage-semantics-file-sha256")
    execute.add_argument(
        "--storage-authority-identity-root",
        default=M03R_V16_POD_AUTHORITY_ROOT,
    )
    execute.add_argument(
        "--storage-observer-identity-root",
        default=M03R_V16_POD_AUTHORITY_OBSERVER_ROOT,
    )
    execute.add_argument("--timeout-seconds", type=float, default=86_400.0)
    execute.add_argument("--poll-seconds", type=float, default=5.0)

    for name in ("status", "cleanup"):
        command = subparsers.add_parser(
            name,
            help=f"{name} one authority-bound exact Job UID",
        )
        _add_package_arguments(command)
        command.add_argument("--controller-job-plan", required=True)
        command.add_argument("--controller-job-plan-file-sha256", required=True)
        command.add_argument("--controller-admission", required=True)
        command.add_argument("--controller-admission-file-sha256", required=True)
        command.add_argument("--storage-semantics", required=True)
        command.add_argument("--storage-semantics-file-sha256", required=True)
        command.add_argument(
            "--storage-authority-identity-root",
            default=M03R_V16_POD_AUTHORITY_ROOT,
        )
        command.add_argument(
            "--storage-observer-identity-root",
            default=M03R_V16_POD_AUTHORITY_OBSERVER_ROOT,
        )
        if name == "cleanup":
            command.add_argument("--authority-root", required=True)
            command.add_argument("--timeout-seconds", type=float, default=900.0)
            command.add_argument("--poll-seconds", type=float, default=2.0)
            command.add_argument(
                "--cleanup-authorized",
                action="store_true",
                help="explicitly authorize exact UID-bound foreground deletion",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit, exact-identity lifecycle action."""

    args = _parser().parse_args(argv)
    if args.command == "execute":
        if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
            raise M03RV16SeadragonControllerError(
                "V16 controller wait intervals must be positive"
            )
        value = _execute_controller_job(args)
    else:
        package, authorization = _load_cli_package(args)
        controller_admission = _load_cli_controller_admission(
            args, package, authorization
        )
        transport = M03RV16KubectlTransport()
        if args.command == "status":
            value = snapshot_m03r_v16_exact_job(
                transport=transport,
                admission=controller_admission.admission,
                package=package,
            )
        else:
            value = cleanup_m03r_v16_exact_job(
                transport=transport,
                controller_admission=controller_admission,
                package=package,
                authorization=authorization,
                authority_root=args.authority_root,
                cleanup_authorized=args.cleanup_authorized,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "M03R_V16_CONTEXT",
    "M03R_V16_CONTROLLER_ADMISSION_SCHEMA",
    "M03R_V16_CONTROLLER_ATTESTATION_SCHEMA",
    "M03R_V16_CONTROLLER_CLEANUP_SCHEMA",
    "M03R_V16_CONTROLLER_COMPLETION_ATTESTATION_SCHEMA",
    "M03R_V16_CONTROLLER_JOB_PLAN_SCHEMA",
    "M03R_V16_CONTROLLER_TERMINAL_SCHEMA",
    "M03R_V16_KUBECONFIG",
    "M03R_V16_KUBECTL",
    "M03R_V16_POD_AUTHORITY_OBSERVER_ROOT",
    "M03R_V16_POD_AUTHORITY_ROOT",
    "M03R_V16_SCHEDULING_POLICY",
    "M03R_V16_ZERO_GPU_TRANSITION_SCHEMA",
    "M03RV16ControllerAdmission",
    "M03RV16ControllerAttestations",
    "M03RV16ExactJobAuthority",
    "M03RV16KubectlTransport",
    "M03RV16KubernetesTransport",
    "M03RV16SeadragonControllerError",
    "M03RV16ZeroGpuJobAuthority",
    "admit_m03r_v16_suspended_job",
    "cleanup_m03r_v16_exact_job",
    "launch_m03r_v16_zero_gpu_gate",
    "load_m03r_v16_controller_job_plan",
    "main",
    "resume_and_attest_m03r_v16_job",
    "snapshot_m03r_v16_exact_job",
    "wait_for_m03r_v16_exact_job_terminal",
    "write_m03r_v16_controller_job_plan",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())
