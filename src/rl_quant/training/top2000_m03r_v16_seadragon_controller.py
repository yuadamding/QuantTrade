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
from rl_quant.training.top2000_m03r_v16_activation import (
    M03R_V16_ADMITTED_MANIFEST_SCHEMA,
    M03R_V16_DRY_RUN_RESULT_SCHEMA,
    M03RV16AdmittedJobAuthority,
    _issue_m03r_v16_admitted_job_authority,
    write_m03r_v16_admitted_job_authority,
)
from rl_quant.training.top2000_m03r_v16_kubernetes import (
    M03R_V16_NAMESPACE,
    M03RV16RenderedSuspendedJob,
    bind_m03r_v16_admitted_launch_authority,
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
    "rl-quant.top2000-dev.m03r-v16-controller-admission-v1"
)
M03R_V16_CONTROLLER_ATTESTATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-attestation-v1"
)
M03R_V16_CONTROLLER_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-terminal-v1"
)
M03R_V16_CONTROLLER_CLEANUP_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-cleanup-v1"
)
M03R_V16_CONTROLLER_JOB_PLAN_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-controller-job-plan-v1"
)
_MAX_KUBECTL_RESPONSE_BYTES = 64 * 1024**2


class M03RV16SeadragonControllerError(RuntimeError):
    """An exact Kubernetes identity or lifecycle transition drifted."""


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
        completed = subprocess.run(
            command,
            input=encoded,
            capture_output=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            if allow_not_found and "not found" in stderr.lower():
                return None
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
class _M03RV16CliExactJobAuthority:
    phase: str
    job_name: str
    job_uid: str
    completions: int

    def validate(self) -> None:
        expected = {
            "static": 1,
            "storage": 1,
            "capacity": 1,
            "training": 3,
            "qualification-preflight": 3,
            "qualification": 3,
        }
        if (
            self.phase not in expected
            or self.completions != expected[self.phase]
            or not self.job_name
            or not self.job_uid
        ):
            raise M03RV16SeadragonControllerError(
                "V16 exact CLI Job identity is malformed"
            )


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
            ("create", "--file", "-", "--output", "json"),
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
    """Dry-run, create, bind, and seal one still-suspended exact Job."""

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
    dry_observed = transport.invoke(
        ("create", "--dry-run=server", "--file", "-", "--output", "json"),
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
    dry_path = root / f"{rendered.mode}-dry-run.json"
    dry_file_sha = _write_create_only_json(dry_path, dry_value)

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
    admitted_path = root / f"{rendered.mode}-admitted-manifest.json"
    admitted_file_sha = _write_create_only_json(admitted_path, admitted_value)
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
    admission_path = root / f"{rendered.mode}-admission.json"
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
    launch_path = root / f"{rendered.mode}-launch.json"
    write_m03r_v16_rendered_launch_authority(launch_path, bound)
    bound_annotations = _metadata(bound.manifest).get("annotations", {})
    bound_pod_annotations = (
        bound.manifest.get("spec", {})
        .get("template", {})
        .get("metadata", {})
        .get("annotations", {})
    )
    patched = transport.invoke(
        (
            "patch",
            "job",
            job_name,
            "--type",
            "merge",
            "--patch-file",
            "-",
            "--output",
            "json",
        ),
        payload={
            "metadata": {
                "resourceVersion": job_resource_version,
                "annotations": bound_annotations,
            },
            "spec": {
                "suspend": True,
                "template": {"metadata": {"annotations": bound_pod_annotations}},
            },
        },
    )
    if patched is None:
        raise M03RV16SeadragonControllerError("V16 bound Job patch vanished")
    _, patched_resource_version = _validate_exact_job(
        patched,
        rendered,
        expected_uid=job_uid,
        expected_suspended=True,
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
    if _list_exact_job_pods(transport, job_name=job_name):
        raise M03RV16SeadragonControllerError(
            "V16 launch-bound Job created Pods while suspended"
        )
    controller_unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_ADMISSION_SCHEMA,
        "phase": rendered.mode,
        "run_id": run_id,
        "job_name": job_name,
        "job_uid": job_uid,
        "prelaunch_job_resource_version": patched_resource_version,
        "server_side_dry_run_file_sha256": dry_file_sha,
        "admitted_manifest_file_sha256": admitted_file_sha,
        "admission_authority_file_sha256": admission_file_sha,
        "admission_authority_receipt_sha256": admission.receipt_sha256,
        "launch_authority_file_sha256": bound.launch_authority_file_sha256,
        "launch_authority_receipt_sha256": (
            None
            if bound.launch_authority is None
            else bound.launch_authority.receipt_sha256
        ),
        "storage_semantics_receipt_sha256": storage_evidence.receipt_sha256,
        "suspended": True,
        "pod_inventory_empty": True,
    }
    controller_value = {
        **controller_unsigned,
        "receipt_sha256": semantic_sha256(controller_unsigned),
    }
    controller_receipt_path = root / f"{rendered.mode}-controller-admission.json"
    _write_create_only_json(controller_receipt_path, controller_value)
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


def launch_m03r_v16_zero_gpu_gate(
    *,
    transport: M03RV16KubernetesTransport,
    rendered: M03RV16RenderedSuspendedJob,
    authority_root: str | Path,
) -> M03RV16ZeroGpuJobAuthority:
    """Dry-run, create, audit, and resume one exact static or storage Job."""

    rendered.validate()
    if rendered.mode not in {"static", "storage"}:
        raise M03RV16SeadragonControllerError(
            "V16 zero-GPU launch received a scientific phase"
        )
    job_name = _job_name(rendered)
    dry = transport.invoke(
        ("create", "--dry-run=server", "--file", "-", "--output", "json"),
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
    admitted = _create_or_reconcile_suspended_job(transport, rendered)
    job_uid, resource_version = _validate_exact_job(
        admitted,
        rendered,
        expected_uid=None,
        expected_suspended=True,
    )
    if _list_exact_job_pods(transport, job_name=job_name):
        raise M03RV16SeadragonControllerError(
            "V16 suspended zero-GPU Job created a Pod"
        )
    resumed = transport.invoke(
        (
            "patch",
            "job",
            job_name,
            "--type",
            "merge",
            "--patch-file",
            "-",
            "--output",
            "json",
        ),
        payload={
            "metadata": {"resourceVersion": resource_version},
            "spec": {"suspend": False},
        },
    )
    if resumed is None:
        raise M03RV16SeadragonControllerError(
            "V16 zero-GPU Job resume vanished"
        )
    _, resumed_resource_version = _validate_exact_job(
        resumed,
        rendered,
        expected_uid=job_uid,
        expected_suspended=False,
    )
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_ADMISSION_SCHEMA,
        "phase": rendered.mode,
        "job_uid": job_uid,
        "job_name": job_name,
        "completions": rendered.completions,
        "server_dry_run_response_sha256": semantic_sha256(dry),
        "admitted_manifest_sha256": semantic_sha256(admitted),
        "resumed_job_resource_version": resumed_resource_version,
        "zero_gpu": True,
    }
    receipt = semantic_sha256(unsigned)
    _write_create_only_json(
        Path(authority_root) / f"{rendered.mode}-controller-admission.json",
        {**unsigned, "receipt_sha256": receipt},
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
    """Resume one exact Job and publish every completion's Pod attestation."""

    bound = controller_admission.rendered
    admission = controller_admission.admission
    launch = bound.launch_authority
    if launch is None:
        raise M03RV16SeadragonControllerError("V16 launch authority is absent")
    job_name = admission.job_name
    resumed = transport.invoke(
        (
            "patch",
            "job",
            job_name,
            "--type",
            "merge",
            "--patch-file",
            "-",
            "--output",
            "json",
        ),
        payload={
            "metadata": {
                "resourceVersion": (
                    controller_admission.prelaunch_job_resource_version
                )
            },
            "spec": {"suspend": False},
        },
    )
    if resumed is None:
        raise M03RV16SeadragonControllerError("V16 Job resume vanished")
    _validate_exact_job(
        resumed,
        bound,
        expected_uid=admission.job_uid,
        expected_suspended=False,
    )
    deadline = time.monotonic() + timeout_seconds
    observations: dict[int, M03RV16PodObservation] = {}
    while len(observations) != bound.completions:
        observations.clear()
        for pod in _list_exact_job_pods(transport, job_name=job_name):
            try:
                observed = _observe_pod(pod, admission=admission)
            except M03RV16SeadragonControllerError:
                continue
            if observed.completion_index in observations:
                raise M03RV16SeadragonControllerError(
                    "V16 completion has multiple observable Pods"
                )
            observations[observed.completion_index] = observed
        if len(observations) == bound.completions:
            break
        if time.monotonic() >= deadline:
            raise M03RV16SeadragonControllerError(
                "V16 exact Pod inventory did not become attestable"
            )
        time.sleep(poll_seconds)

    published_rows: list[M03RV16PublishedPodAttestation] = []
    for index in range(bound.completions):
        observation = observations[index]

        def patch_annotations(
            precondition: object,
            values: Mapping[str, str],
            *,
            pod_name: str = observation.pod_name,
            expected_pod_uid: str = observation.pod_uid,
        ) -> None:
            resource_version = getattr(precondition, "pod_resource_version", None)
            pod_uid = getattr(precondition, "pod_uid", None)
            if pod_uid != expected_pod_uid:
                raise M03RV16SeadragonControllerError(
                    "V16 Pod patch UID precondition drifted"
                )
            patched = transport.invoke(
                (
                    "patch",
                    "pod",
                    pod_name,
                    "--type",
                    "merge",
                    "--patch-file",
                    "-",
                    "--output",
                    "json",
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
            pod_name: str = observation.pod_name,
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
                annotations={str(key): str(value) for key, value in annotations.items()},
            )

        published_rows.append(
            publish_m03r_v16_pod_runtime_attestation_after_annotation_patch(
                package=package,
                authorization=authorization,
                admission=admission,
                launch=launch,
                observation=observation,
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
        )
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_ATTESTATION_SCHEMA,
        "phase": bound.mode,
        "job_uid": admission.job_uid,
        "job_name": job_name,
        "launch_authority_receipt_sha256": launch.receipt_sha256,
        "storage_semantics_receipt_sha256": storage_evidence.receipt_sha256,
        "completion_indices": list(range(bound.completions)),
        "pod_uids": [observations[index].pod_uid for index in range(bound.completions)],
        "attestation_file_sha256": [row.file_sha256 for row in published_rows],
        "attestation_receipt_sha256": [
            row.receipt_sha256 for row in published_rows
        ],
        "controller_transaction_receipt_sha256": [
            row.controller_transaction_receipt_sha256 for row in published_rows
        ],
    }
    value = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    receipt_path = (
        Path(authority_root) / f"{bound.mode}-controller-attestations.json"
    )
    _write_create_only_json(receipt_path, value)
    return M03RV16ControllerAttestations(
        rows=tuple(published_rows),
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
        "job_uid": admission.job_uid,
        "job_name": admission.job_name,
        "outcome": outcome,
        "snapshot": snapshot,
    }
    value = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    path = Path(authority_root) / f"{admission.phase}-controller-terminal.json"
    _write_create_only_json(path, value)
    return value


def cleanup_m03r_v16_exact_job(
    *,
    transport: M03RV16KubernetesTransport,
    admission: M03RV16ExactJobAuthority,
    authority_root: str | Path,
    cleanup_authorized: bool,
) -> dict[str, Any]:
    """Delete one exact Job using UID/resourceVersion preconditions."""

    if not cleanup_authorized:
        raise M03RV16SeadragonControllerError(
            "V16 exact cleanup lacks explicit authorization"
        )
    current = transport.invoke(
        ("get", "job", admission.job_name, "--output", "json"),
        allow_not_found=True,
    )
    if current is None:
        raise M03RV16SeadragonControllerError(
            "V16 cleanup target is already absent"
        )
    metadata = _metadata(current)
    if metadata.get("uid") != admission.job_uid:
        raise M03RV16SeadragonControllerError(
            "V16 cleanup target UID drifted"
        )
    resource_version = _text(
        "cleanup resourceVersion", metadata.get("resourceVersion")
    )
    response = transport.invoke(
        (
            "delete",
            "--raw",
            (
                f"/apis/batch/v1/namespaces/{M03R_V16_NAMESPACE}/jobs/"
                f"{admission.job_name}"
            ),
            "--file",
            "-",
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
        raise M03RV16SeadragonControllerError("V16 cleanup response vanished")
    observed = transport.invoke(
        ("get", "job", admission.job_name, "--output", "json"),
        allow_not_found=True,
    )
    if observed is not None:
        raise M03RV16SeadragonControllerError(
            "V16 exact Job remains after cleanup"
        )
    if _list_exact_job_pods(transport, job_name=admission.job_name):
        raise M03RV16SeadragonControllerError(
            "V16 UID-owned Pod inventory remains after cleanup"
        )
    unsigned: dict[str, Any] = {
        "schema": M03R_V16_CONTROLLER_CLEANUP_SCHEMA,
        "phase": admission.phase,
        "job_uid": admission.job_uid,
        "job_name": admission.job_name,
        "resource_version": resource_version,
        "job_absent": True,
        "pod_inventory_empty": True,
    }
    value = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    path = Path(authority_root) / f"{admission.phase}-controller-cleanup.json"
    _write_create_only_json(path, value)
    return value


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


def _cli_exact_job(args: argparse.Namespace) -> _M03RV16CliExactJobAuthority:
    value = _M03RV16CliExactJobAuthority(
        phase=args.phase,
        job_name=args.job_name,
        job_uid=args.job_uid,
        completions=args.completions,
    )
    value.validate()
    return value


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
            help=f"{name} one exact Job name and UID",
        )
        _add_package_arguments(command)
        command.add_argument(
            "--phase",
            choices=(
                "static",
                "storage",
                "capacity",
                "training",
                "qualification-preflight",
                "qualification",
            ),
            required=True,
        )
        command.add_argument("--job-name", required=True)
        command.add_argument("--job-uid", required=True)
        command.add_argument("--completions", type=int, required=True)
        if name == "cleanup":
            command.add_argument("--authority-root", required=True)
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
        package, _ = _load_cli_package(args)
        exact = _cli_exact_job(args)
        transport = M03RV16KubectlTransport()
        if args.command == "status":
            value = snapshot_m03r_v16_exact_job(
                transport=transport,
                admission=exact,
                package=package,
            )
        else:
            value = cleanup_m03r_v16_exact_job(
                transport=transport,
                admission=exact,
                authority_root=args.authority_root,
                cleanup_authorized=args.cleanup_authorized,
            )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "M03R_V16_CONTEXT",
    "M03R_V16_CONTROLLER_ADMISSION_SCHEMA",
    "M03R_V16_CONTROLLER_ATTESTATION_SCHEMA",
    "M03R_V16_CONTROLLER_CLEANUP_SCHEMA",
    "M03R_V16_CONTROLLER_JOB_PLAN_SCHEMA",
    "M03R_V16_CONTROLLER_TERMINAL_SCHEMA",
    "M03R_V16_KUBECONFIG",
    "M03R_V16_KUBECTL",
    "M03R_V16_POD_AUTHORITY_OBSERVER_ROOT",
    "M03R_V16_POD_AUTHORITY_ROOT",
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
