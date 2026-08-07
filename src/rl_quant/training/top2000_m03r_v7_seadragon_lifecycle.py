"""Attach-only Seadragon lifecycle for receipt-gated TOP2000 research Jobs.

This module deliberately has no Kubernetes create, apply, or replace surface.
It can only attach to an already-created suspended Job, activate that exact
UID through a preconditioned JSON Patch, observe that Job and its UID-owned
Pods, capture terminal evidence, and delete it with UID/resourceVersion
preconditions.  It also exposes a narrower command for preserving and
cleaning a previously failed bound Job.

All work covered here is development-only, non-PHI research.  The module does
not inspect Secrets, Nodes, unrelated workloads, or kubeconfig contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_ADMISSION_ORDER,
    M03R_SEED17_TOP2000_COMPLETION_SCHEMA,
    M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA,
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_KUBERNETES_CONTEXT,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03RV7AdmittedJobBinding,
    M03RV7ExactJobActivationRequest,
    M03RV7ExactJobCleanupRequest,
    build_m03r_v7_exact_cleanup_receipt,
    build_m03r_v7_exact_job_activation_request,
    build_m03r_v7_exact_job_cleanup_request,
)

SEADRAGON_KUBECTL: Final = "/risapps/noarch/kubectl/1.28.4/bin/kubectl"
SEADRAGON_KUBECONFIG: Final = "/rsrch8/home/bcb/yding4/.kube/config"
SEADRAGON_QUANTTRADE_ROOT: Final = (
    "/rsrch8/home/bcb/yding4/quant/training"
)
ATTACH_CONFIG_SCHEMA: Final = (
    "rl-quant.top2000-m03r-v7-seadragon-attach-config-v1"
)
FAILED_CLEANUP_CONFIG_SCHEMA: Final = (
    "rl-quant.top2000-m03r-v7-seadragon-failed-cleanup-config-v1"
)
COMPLETION_COVERAGE_SCHEMA: Final = (
    "rl-quant.top2000-m03r-v7-one-seed-coverage-v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DNS_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_RUN_ID_ANNOTATION = "rl-quant/run-id"
_PACKAGE_PLAN_ANNOTATION = "rl-quant/package-plan-sha256"
_SOURCE_ARCHIVE_ANNOTATION = "rl-quant/source-archive-sha256"
_CAPACITY_RECEIPT_ANNOTATION = "rl-quant/capacity-receipt-sha256"
_CELL_RECEIPT_SCHEMA = "rl-quant.top2000-dev.m03r-v7-cell-receipt-v2"
_RECEIPT_MODE = 0o600
_DIRECTORY_MODE = 0o700


class SeadragonLifecycleError(RuntimeError):
    """The attach-only lifecycle or its exact evidence failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SeadragonLifecycleError(
            "lifecycle evidence is not canonical-JSON safe"
        ) from exc


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise SeadragonLifecycleError(f"{name} must be a lowercase SHA-256")


def _require_dns(name: str, value: str) -> None:
    if len(value) > 63 or _DNS_RE.fullmatch(value) is None:
        raise SeadragonLifecycleError(f"{name} must be a Kubernetes DNS label")


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SeadragonLifecycleError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _regular_no_symlink(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise SeadragonLifecycleError(f"{label} must be absolute")
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise SeadragonLifecycleError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise SeadragonLifecycleError(
            f"{label} must be a regular non-symlink: {path}"
        )
    if stat_result.st_size < 0:  # pragma: no cover - defensive filesystem guard
        raise SeadragonLifecycleError(f"{label} has an invalid size")
    return path


def _directory_no_symlink(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise SeadragonLifecycleError(f"{label} must be absolute")
    try:
        path.lstat()
    except OSError as exc:
        raise SeadragonLifecycleError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink() or not path.is_dir():
        raise SeadragonLifecycleError(
            f"{label} must be a directory non-symlink: {path}"
        )
    return path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_project_path(path: Path, *, label: str) -> Path:
    project = Path(SEADRAGON_QUANTTRADE_ROOT)
    if not path.is_absolute() or not _is_within(path, project):
        raise SeadragonLifecycleError(
            f"{label} must stay below {SEADRAGON_QUANTTRADE_ROOT}"
        )
    return path


def _exclusive_write(path: Path, content: bytes) -> str:
    """Write one no-clobber, no-follow, fsynced receipt and return its hash."""

    parent = _directory_no_symlink(path.parent, label="receipt parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, _RECEIPT_MODE)
    except OSError as exc:
        raise SeadragonLifecycleError(
            f"refusing to overwrite or follow receipt path: {path}"
        ) from exc
    try:
        view = memoryview(content)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:  # pragma: no cover - defensive I/O guard
                raise SeadragonLifecycleError("receipt write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return hashlib.sha256(content).hexdigest()


def _exclusive_json(path: Path, value: Any) -> str:
    return _exclusive_write(path, _canonical_bytes(value))


def _create_evidence_root(path: Path) -> Path:
    _require_project_path(path, label="evidence root")
    parent = _directory_no_symlink(path.parent, label="evidence parent")
    try:
        os.mkdir(path, _DIRECTORY_MODE)
    except OSError as exc:
        raise SeadragonLifecycleError(
            f"evidence root must be a new absent directory: {path}"
        ) from exc
    directory_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return _directory_no_symlink(path, label="evidence root")


@dataclass(frozen=True, slots=True)
class ExpectedCompletion:
    completion_index: int
    setting_index: int
    setting_id: str

    def __post_init__(self) -> None:
        if not 0 <= self.completion_index < 12:
            raise SeadragonLifecycleError("completion index must lie in [0, 11]")
        if not 0 <= self.setting_index < 12 or not self.setting_id:
            raise SeadragonLifecycleError("expected setting identity is invalid")


def canonical_one_seed_completions(
    setting_ids: Sequence[str],
) -> tuple[ExpectedCompletion, ...]:
    if tuple(setting_ids) != M03R_SEED17_TOP2000_SETTING_IDS:
        raise SeadragonLifecycleError(
            "setting IDs must match the immutable seed-17 twelve-setting panel"
        )
    return tuple(
        ExpectedCompletion(
            completion_index=completion_index,
            setting_index=setting_index,
            setting_id=setting_ids[setting_index],
        )
        for completion_index, setting_index in enumerate(
            M03R_SEED17_TOP2000_ADMISSION_ORDER
        )
    )


@dataclass(frozen=True, slots=True)
class AttachSupervisorConfig:
    job_name: str
    run_id: str
    job_uid: str
    binding_path: str
    binding_file_sha256: str
    activation_request_path: str
    activation_request_file_sha256: str
    output_root: str
    evidence_root: str
    package_plan_sha256: str
    source_archive_sha256: str
    capacity_receipt_sha256: str
    expected_completions: tuple[ExpectedCompletion, ...]
    host_python_path: str
    pythonpath: str
    kubectl_path: str = SEADRAGON_KUBECTL
    kubeconfig_path: str = SEADRAGON_KUBECONFIG
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    expected_seed: int = 17
    expected_fold_count: int = 6
    completions: int = 12
    parallelism: int = 8
    gpus_per_worker: int = 2
    authorized_h100_cap: int = 16
    poll_interval_seconds: int = 30
    request_timeout_seconds: int = 30
    handshake_timeout_seconds: int = 180
    hard_wall_seconds: int = 218400
    log_limit_bytes: int = 1048576
    schema: str = ATTACH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _require_dns("job_name", self.job_name)
        _require_dns("run_id", self.run_id)
        if not self.job_uid:
            raise SeadragonLifecycleError("job_uid is required")
        for name in (
            "binding_file_sha256",
            "activation_request_file_sha256",
            "package_plan_sha256",
            "source_archive_sha256",
            "capacity_receipt_sha256",
        ):
            _require_sha256(name, cast(str, getattr(self, name)))
        if (
            self.schema != ATTACH_CONFIG_SCHEMA
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.kubectl_path != SEADRAGON_KUBECTL
            or self.kubeconfig_path != SEADRAGON_KUBECONFIG
            or self.expected_seed != 17
            or self.expected_fold_count != 6
            or self.completions != 12
            or self.parallelism != 8
            or self.gpus_per_worker != 2
            or self.authorized_h100_cap != 16
            or self.parallelism * self.gpus_per_worker
            > self.authorized_h100_cap
        ):
            raise SeadragonLifecycleError(
                "attach config drifted from the approved one-seed 12x2-H100 contract"
            )
        if tuple(row.completion_index for row in self.expected_completions) != tuple(
            range(12)
        ) or tuple(row.setting_index for row in self.expected_completions) != tuple(
            M03R_SEED17_TOP2000_ADMISSION_ORDER
        ):
            raise SeadragonLifecycleError(
                "completion map must match the frozen twelve-setting admission order"
            )
        if (
            self.poll_interval_seconds < 5
            or self.request_timeout_seconds < 5
            or self.handshake_timeout_seconds < 30
            or self.hard_wall_seconds <= self.handshake_timeout_seconds
            or self.log_limit_bytes < 4096
        ):
            raise SeadragonLifecycleError("attach timing or log bounds are invalid")
        for name in (
            "binding_path",
            "activation_request_path",
            "output_root",
            "evidence_root",
            "pythonpath",
        ):
            _require_project_path(Path(cast(str, getattr(self, name))), label=name)
        if not Path(self.host_python_path).is_absolute():
            raise SeadragonLifecycleError("host_python_path must be absolute")

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailedJobCleanupConfig:
    job_name: str
    run_id: str
    job_uid: str
    binding_path: str
    binding_file_sha256: str
    evidence_root: str
    kubectl_path: str = SEADRAGON_KUBECTL
    kubeconfig_path: str = SEADRAGON_KUBECONFIG
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    request_timeout_seconds: int = 30
    log_limit_bytes: int = 1048576
    schema: str = FAILED_CLEANUP_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _require_dns("job_name", self.job_name)
        _require_dns("run_id", self.run_id)
        _require_sha256("binding_file_sha256", self.binding_file_sha256)
        if (
            not self.job_uid
            or self.schema != FAILED_CLEANUP_CONFIG_SCHEMA
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.kubectl_path != SEADRAGON_KUBECTL
            or self.kubeconfig_path != SEADRAGON_KUBECONFIG
            or self.request_timeout_seconds < 5
            or self.log_limit_bytes < 4096
        ):
            raise SeadragonLifecycleError("failed-cleanup config is invalid")
        _require_project_path(Path(self.binding_path), label="binding_path")
        _require_project_path(Path(self.evidence_root), label="evidence_root")


def _read_json_file(path: Path, *, expected_sha256: str | None = None) -> Any:
    _regular_no_symlink(path, label="JSON input")
    if expected_sha256 is not None and _file_sha256(path) != expected_sha256:
        raise SeadragonLifecycleError(f"JSON input hash drifted: {path}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise SeadragonLifecycleError(f"JSON input cannot be decoded: {path}") from exc


def _binding_from_file(path: Path, expected_sha256: str) -> M03RV7AdmittedJobBinding:
    payload = _mapping(
        _read_json_file(path, expected_sha256=expected_sha256),
        "admitted binding",
    )
    values = dict(payload)
    values["attached_owned_pod_uids"] = tuple(
        cast(Sequence[str], values.get("attached_owned_pod_uids", ()))
    )
    try:
        return M03RV7AdmittedJobBinding(**values)
    except (TypeError, ValueError) as exc:
        raise SeadragonLifecycleError("admitted binding is invalid") from exc


def _activation_from_file(
    path: Path,
    expected_sha256: str,
) -> M03RV7ExactJobActivationRequest:
    payload = _mapping(
        _read_json_file(path, expected_sha256=expected_sha256),
        "activation request",
    )
    try:
        return M03RV7ExactJobActivationRequest(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise SeadragonLifecycleError("activation request is invalid") from exc


def _same_activation_contract(
    configured: M03RV7ExactJobActivationRequest,
    runtime: M03RV7ExactJobActivationRequest,
) -> bool:
    """Compare immutable attachment identity while allowing a fresh Job RV."""

    fields = (
        "schema",
        "content_type",
        "job_name",
        "namespace",
        "job_uid",
        "run_id",
        "parallelism",
        "binding_receipt_sha256",
        "admitted_selector_sha256",
        "admitted_template_metadata_sha256",
        "admitted_pod_template_sha256",
    )
    return all(
        getattr(configured, name) == getattr(runtime, name) for name in fields
    )


class KubectlTransport(Protocol):
    def get_job(self, *, allow_absent: bool = False) -> Mapping[str, Any] | None: ...

    def get_owned_pods(self) -> tuple[Mapping[str, Any], ...]: ...

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes: ...

    def activate(self, request: M03RV7ExactJobActivationRequest) -> Mapping[str, Any]: ...

    def delete(self, request: M03RV7ExactJobCleanupRequest, options_path: Path) -> None: ...


class AttachOnlyKubectl:
    """Narrow kubectl transport with no create/apply/replace implementation."""

    def __init__(
        self,
        *,
        kubectl_path: str,
        kubeconfig_path: str,
        context: str,
        namespace: str,
        job_name: str,
        job_uid: str,
        request_timeout_seconds: int,
    ) -> None:
        self.kubectl_path = kubectl_path
        self.kubeconfig_path = kubeconfig_path
        self.context = context
        self.namespace = namespace
        self.job_name = job_name
        self.job_uid = job_uid
        self.request_timeout_seconds = request_timeout_seconds

    def _run(self, arguments: Sequence[str]) -> bytes:
        if not arguments or arguments[0] not in {"get", "logs", "patch", "delete"}:
            raise SeadragonLifecycleError(
                "attach-only kubectl rejected a non-observational/non-bound verb"
            )
        if arguments[0] in {"create", "apply", "replace"}:  # pragma: no cover
            raise SeadragonLifecycleError("Job creation is forbidden")
        environment = {
            "KUBECONFIG": self.kubeconfig_path,
            "PATH": str(Path(self.kubectl_path).parent),
            "LANG": "C",
            "LC_ALL": "C",
        }
        command = [
            self.kubectl_path,
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            f"--request-timeout={self.request_timeout_seconds}s",
            *arguments,
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=self.request_timeout_seconds + 5,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SeadragonLifecycleError("bounded kubectl invocation failed") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
            raise SeadragonLifecycleError(
                f"kubectl {arguments[0]} failed: {stderr}"
            )
        return completed.stdout

    def get_job(self, *, allow_absent: bool = False) -> Mapping[str, Any] | None:
        arguments = ["get", "job", self.job_name]
        if allow_absent:
            arguments.append("--ignore-not-found")
        arguments.extend(("-o", "json"))
        payload = self._run(arguments)
        if not payload.strip():
            return None
        return _mapping(json.loads(payload), "kubectl Job")

    def get_owned_pods(self) -> tuple[Mapping[str, Any], ...]:
        selector = f"job-name={self.job_name},controller-uid={self.job_uid}"
        payload = _mapping(
            json.loads(self._run(("get", "pods", "-l", selector, "-o", "json"))),
            "kubectl PodList",
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise SeadragonLifecycleError("PodList items are invalid")
        pods = tuple(_mapping(item, "owned Pod") for item in items)
        _validate_owned_pods(pods, expected_uid=self.job_uid)
        return pods

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
        _require_dns("pod_name", pod_name)
        return self._run(
            (
                "logs",
                pod_name,
                "--container",
                "trainer",
                f"--limit-bytes={limit_bytes}",
            )
        )

    def activate(
        self,
        request: M03RV7ExactJobActivationRequest,
    ) -> Mapping[str, Any]:
        if (
            request.job_name != self.job_name
            or request.job_uid != self.job_uid
            or request.namespace != self.namespace
        ):
            raise SeadragonLifecycleError("activation target is not this exact Job")
        payload = self._run(
            (
                "patch",
                "job",
                self.job_name,
                "--type=json",
                f"--patch={request.json_patch_json}",
                "-o",
                "json",
            )
        )
        return _mapping(json.loads(payload), "activated Job")

    def delete(
        self,
        request: M03RV7ExactJobCleanupRequest,
        options_path: Path,
    ) -> None:
        if (
            request.job_name != self.job_name
            or request.job_uid != self.job_uid
            or request.namespace != self.namespace
        ):
            raise SeadragonLifecycleError("cleanup target is not this exact Job")
        raw_uri = (
            f"/apis/batch/v1/namespaces/{self.namespace}/jobs/{self.job_name}"
        )
        self._run(("delete", "--raw", raw_uri, "-f", str(options_path)))


def _job_identity(
    job: Mapping[str, Any],
    *,
    job_name: str,
    run_id: str,
    job_uid: str,
) -> None:
    metadata = _mapping(job.get("metadata"), "Job metadata")
    annotations = _mapping(metadata.get("annotations"), "Job annotations")
    if (
        metadata.get("name") != job_name
        or metadata.get("namespace") != M03R_TOP2000_KUBERNETES_NAMESPACE
        or metadata.get("uid") != job_uid
        or annotations.get(_RUN_ID_ANNOTATION) != run_id
        or not isinstance(metadata.get("resourceVersion"), str)
    ):
        raise SeadragonLifecycleError("exact Job name/run ID/UID drifted")


def _job_artifact_identity(
    job: Mapping[str, Any],
    config: AttachSupervisorConfig,
) -> None:
    """Bind the live Job and Pod template to this package/capacity contract."""

    expected = {
        _RUN_ID_ANNOTATION: config.run_id,
        _PACKAGE_PLAN_ANNOTATION: config.package_plan_sha256,
        _SOURCE_ARCHIVE_ANNOTATION: config.source_archive_sha256,
        _CAPACITY_RECEIPT_ANNOTATION: config.capacity_receipt_sha256,
    }
    metadata = _mapping(job.get("metadata"), "Job metadata")
    job_annotations = _mapping(metadata.get("annotations"), "Job annotations")
    spec = _mapping(job.get("spec"), "Job spec")
    template = _mapping(spec.get("template"), "Job Pod template")
    template_metadata = _mapping(
        template.get("metadata"), "Job Pod template metadata"
    )
    template_annotations = _mapping(
        template_metadata.get("annotations"), "Job Pod template annotations"
    )
    if any(
        job_annotations.get(name) != value
        or template_annotations.get(name) != value
        for name, value in expected.items()
    ):
        raise SeadragonLifecycleError(
            "live Job package/source/capacity annotation binding drifted"
        )


def _validate_owned_pods(
    pods: Sequence[Mapping[str, Any]],
    *,
    expected_uid: str,
) -> None:
    for pod in pods:
        metadata = _mapping(pod.get("metadata"), "Pod metadata")
        owners = metadata.get("ownerReferences")
        if not isinstance(owners, list) or not any(
            isinstance(owner, Mapping)
            and owner.get("uid") == expected_uid
            and owner.get("controller") is True
            for owner in owners
        ):
            raise SeadragonLifecycleError("Pod is not owned by the exact Job UID")


def _true_condition(job: Mapping[str, Any]) -> str | None:
    status = _mapping(job.get("status", {}), "Job status")
    conditions = status.get("conditions", [])
    if not isinstance(conditions, list):
        raise SeadragonLifecycleError("Job conditions are invalid")
    for preferred in ("Failed", "Complete"):
        if any(
            isinstance(row, Mapping)
            and row.get("type") == preferred
            and row.get("status") == "True"
            for row in conditions
        ):
            return preferred
    return None


def _pod_identity(pod: Mapping[str, Any]) -> tuple[str, str | None]:
    metadata = _mapping(pod.get("metadata"), "Pod metadata")
    name = metadata.get("name")
    if not isinstance(name, str):
        raise SeadragonLifecycleError("owned Pod name is absent")
    annotations = _mapping(metadata.get("annotations", {}), "Pod annotations")
    index = annotations.get("batch.kubernetes.io/job-completion-index")
    return name, index if isinstance(index, str) else None


def _capture_terminal(
    *,
    root: Path,
    job: Mapping[str, Any],
    pods: Sequence[Mapping[str, Any]],
    transport: KubectlTransport,
    reason: str,
    log_limit_bytes: int,
) -> dict[str, Any]:
    job_sha = _exclusive_json(root / "terminal-job.json", job)
    pod_list = {"apiVersion": "v1", "kind": "PodList", "items": list(pods)}
    pods_sha = _exclusive_json(root / "terminal-pods.json", pod_list)
    logs: dict[str, str] = {}
    for pod in pods:
        name, index = _pod_identity(pod)
        suffix = f"index-{index}-{name}" if index is not None else name
        log_path = root / f"terminal-log-{suffix}.txt"
        try:
            content = transport.get_pod_log(name, limit_bytes=log_limit_bytes)
        except SeadragonLifecycleError as exc:
            content = f"log unavailable: {exc}\n".encode()
        logs[log_path.name] = _exclusive_write(log_path, content)
    evidence = {
        "schema": "rl-quant.top2000-m03r-v7-terminal-evidence-v1",
        "captured_at_utc": _utc_now(),
        "reason": reason,
        "job_sha256": job_sha,
        "pods_sha256": pods_sha,
        "log_sha256": logs,
    }
    _exclusive_json(root / "terminal-evidence.json", evidence)
    return evidence


def _completion_paths(
    config: AttachSupervisorConfig,
    row: ExpectedCompletion,
) -> tuple[Path, Path]:
    base = (
        Path(config.output_root)
        / f"completion-{row.completion_index:02d}-setting-{row.setting_index:02d}"
    )
    return (
        base / "execution-plan-binding.json",
        base / "training" / "completion-receipt.json",
    )


def _read_exact_hash_inventory(
    *,
    root: Path,
    value: Any,
    expected_paths: set[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    inventory = _mapping(value, label)
    if set(inventory) != expected_paths:
        raise SeadragonLifecycleError(f"{label} path inventory drifted")
    _directory_no_symlink(root, label=f"{label} root")
    payloads: dict[str, Mapping[str, Any]] = {}
    for relative in sorted(expected_paths):
        parts = Path(relative)
        if parts.is_absolute() or not parts.parts or any(
            part in {"", ".", ".."} for part in parts.parts
        ):
            raise SeadragonLifecycleError(f"{label} contains an unsafe path")
        current = root
        for part in parts.parts[:-1]:
            current = current / part
            _directory_no_symlink(current, label=f"{label} parent")
        path = _regular_no_symlink(root / parts, label=label)
        expected_sha = inventory.get(relative)
        if not isinstance(expected_sha, str):
            raise SeadragonLifecycleError(f"{label} hash is absent")
        _require_sha256(label, expected_sha)
        if _file_sha256(path) != expected_sha:
            raise SeadragonLifecycleError(f"{label} file hash drifted")
        payloads[relative] = _mapping(_read_json_file(path), label)
    return payloads


def _validate_h100_rank_proof(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise SeadragonLifecycleError("worker runtime proof requires two ranks")
    rows = sorted(
        (_mapping(row, "rank runtime proof") for row in value),
        key=lambda row: int(cast(int, row.get("rank", -1))),
    )
    for rank, row in enumerate(rows):
        total_memory = row.get("gpu_total_memory_bytes")
        if (
            row.get("rank") != rank
            or row.get("device") != f"cuda:{rank}"
            or row.get("gpu_name") != "NVIDIA H100 80GB HBM3"
            or isinstance(total_memory, bool)
            or not isinstance(total_memory, int)
            or not 79 * 1024**3 <= total_memory <= 81 * 1024**3
            or row.get("compute_capability") != [9, 0]
            or row.get("allocator_oom_count") != 0
            or row.get("torchrun_restart_count") not in {0, 1}
        ):
            raise SeadragonLifecycleError(
                "worker rank did not prove one qualified H100 80GB device"
            )
    return [dict(row) for row in rows]


def _terminal_pod_inventory(
    pods: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, str]]:
    if len(pods) != 12:
        raise SeadragonLifecycleError(
            "complete Job must retain exactly twelve UID-owned worker Pods"
        )
    inventory: dict[int, dict[str, str]] = {}
    for pod in pods:
        name, raw_index = _pod_identity(pod)
        metadata = _mapping(pod.get("metadata"), "terminal Pod metadata")
        status = _mapping(pod.get("status"), "terminal Pod status")
        uid = metadata.get("uid")
        try:
            index = int(raw_index) if raw_index is not None else -1
        except ValueError as exc:
            raise SeadragonLifecycleError(
                "terminal Pod completion index is invalid"
            ) from exc
        if (
            index in inventory
            or not 0 <= index < 12
            or not isinstance(uid, str)
            or not uid
            or status.get("phase") != "Succeeded"
        ):
            raise SeadragonLifecycleError(
                "terminal Pod inventory is not one successful Pod per completion"
            )
        inventory[index] = {"pod_name": name, "pod_uid": uid}
    if set(inventory) != set(range(12)):
        raise SeadragonLifecycleError("terminal Pod completion coverage drifted")
    return inventory


def validate_one_seed_completion_coverage(
    config: AttachSupervisorConfig,
    *,
    owned_pods: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact twelve-setting, six-fold, seed-17 filesystem coverage."""

    receipt_hashes: dict[str, str] = {}
    worker_runtime_proof: dict[str, dict[str, Any]] = {}
    pod_inventory = _terminal_pod_inventory(owned_pods)
    for row in config.expected_completions:
        binding_path, receipt_path = _completion_paths(config, row)
        binding = _mapping(_read_json_file(binding_path), "execution-plan binding")
        completion = _mapping(_read_json_file(receipt_path), "completion receipt")
        bound_completion = _mapping(binding.get("completion"), "bound completion")
        if (
            binding.get("package_plan_sha256") != config.package_plan_sha256
            or bound_completion.get("completion_index") != row.completion_index
            or bound_completion.get("setting_index") != row.setting_index
            or bound_completion.get("setting_id") != row.setting_id
            or bound_completion.get("paired_seeds") != [config.expected_seed]
            or bound_completion.get("fold_indices")
            != list(range(config.expected_fold_count))
            or bound_completion.get("one_member_fold_execution") is not True
            or bound_completion.get("development_only") is not True
            or bound_completion.get("promotion_eligible") is not False
            or completion.get("setting_index") != row.setting_index
            or completion.get("setting_id") != row.setting_id
            or completion.get("schema")
            != M03R_SEED17_TOP2000_COMPLETION_SCHEMA
            or completion.get("protocol_sha256")
            != M03R_SEED17_TOP2000_PROTOCOL_SHA256
            or completion.get("world_size") != 2
            or completion.get("fold_count") != config.expected_fold_count
            or completion.get("paired_seeds") != [config.expected_seed]
            or completion.get("completed_cells") != config.expected_fold_count
            or completion.get("seed_validation_receipt_count")
            != config.expected_fold_count
            or completion.get("fold_ensemble_receipt_count") != 0
            or completion.get("fold_execution_receipt_count")
            != config.expected_fold_count
            or completion.get("inference_path_count") != config.expected_fold_count
            or completion.get("one_member_fold_execution_required") is not True
            or completion.get("five_seed_ensemble_eligible") is not False
            or completion.get("output_space_ensemble_required") is not False
            or completion.get("complete") is not True
            or completion.get("development_only") is not True
            or completion.get("promotion_eligible") is not False
            or completion.get("future_selected_universe") is not True
            or completion.get("outer_evaluation_authorized") is not False
        ):
            raise SeadragonLifecycleError(
                f"completion {row.completion_index} failed exact one-seed coverage"
            )
        run_root = receipt_path.parent
        cell_paths = {
            f"fold-{fold_index:02d}-seed-17.json"
            for fold_index in range(config.expected_fold_count)
        }
        seed_paths = {
            f"receipts/seed-validation/fold-{fold_index:02d}-seed-17.json"
            for fold_index in range(config.expected_fold_count)
        }
        fold_paths = {
            f"receipts/fold-execution/fold-{fold_index:02d}.json"
            for fold_index in range(config.expected_fold_count)
        }
        cells = _read_exact_hash_inventory(
            root=run_root / "receipts",
            value=completion.get("cell_receipt_sha256"),
            expected_paths=cell_paths,
            label="cell receipt inventory",
        )
        seeds = _read_exact_hash_inventory(
            root=run_root,
            value=completion.get("seed_validation_receipt_sha256"),
            expected_paths=seed_paths,
            label="seed validation receipt inventory",
        )
        folds = _read_exact_hash_inventory(
            root=run_root,
            value=completion.get("fold_execution_receipt_sha256"),
            expected_paths=fold_paths,
            label="fold execution receipt inventory",
        )
        seed_hash_inventory = _mapping(
            completion.get("seed_validation_receipt_sha256"),
            "seed validation hash inventory",
        )
        for fold_index in range(config.expected_fold_count):
            cell = cells[f"fold-{fold_index:02d}-seed-17.json"]
            seed_key = (
                f"receipts/seed-validation/fold-{fold_index:02d}-seed-17.json"
            )
            fold_key = f"receipts/fold-execution/fold-{fold_index:02d}.json"
            seed = seeds[seed_key]
            fold = folds[fold_key]
            seed_sha = seed_hash_inventory[seed_key]
            if (
                cell.get("schema") != _CELL_RECEIPT_SCHEMA
                or cell.get("protocol_sha256")
                != M03R_SEED17_TOP2000_PROTOCOL_SHA256
                or cell.get("setting_index") != row.setting_index
                or cell.get("setting_id") != row.setting_id
                or cell.get("fold_index") != fold_index
                or cell.get("seed") != config.expected_seed
                or cell.get("seed_validation_required") is not True
                or cell.get("seed_validation_receipt_sha256") != seed_sha
                or cell.get("development_only") is not True
                or cell.get("promotion_eligible") is not False
                or seed.get("schema")
                != M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA
                or seed.get("protocol_sha256")
                != M03R_SEED17_TOP2000_PROTOCOL_SHA256
                or seed.get("setting_index") != row.setting_index
                or seed.get("setting_id") != row.setting_id
                or seed.get("fold_index") != fold_index
                or seed.get("seed") != config.expected_seed
                or not isinstance(seed.get("metrics"), Mapping)
                or cast(Mapping[str, Any], seed["metrics"]).get("decision_count")
                != 63
                or seed.get("development_only") is not True
                or seed.get("outer_evaluation_authorized") is not False
                or seed.get("promotion_eligible") is not False
                or fold.get("schema")
                != M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA
                or fold.get("protocol_sha256")
                != M03R_SEED17_TOP2000_PROTOCOL_SHA256
                or fold.get("setting_index") != row.setting_index
                or fold.get("setting_id") != row.setting_id
                or fold.get("fold_index") != fold_index
                or fold.get("ordered_seeds") != [config.expected_seed]
                or fold.get("seed_validation_receipt_sha256s") != [seed_sha]
                or fold.get("member_count") != 1
                or fold.get("chronological_return_path_count") != 1
                or fold.get("one_member_fold_execution") is not True
                or fold.get("output_space_ensemble") is not False
                or fold.get("five_seed_ensemble_eligible") is not False
                or fold.get("development_only") is not True
                or fold.get("outer_evaluation_authorized") is not False
                or fold.get("promotion_eligible") is not False
            ):
                raise SeadragonLifecycleError(
                    f"completion {row.completion_index} evidence chain drifted"
                )
            _validate_h100_rank_proof(cell.get("rank_peak_cuda_memory"))
        ranks = _validate_h100_rank_proof(
            completion.get("rank_peak_cuda_memory")
        )
        worker_runtime_proof[str(row.completion_index)] = {
            **pod_inventory[row.completion_index],
            "setting_index": row.setting_index,
            "setting_id": row.setting_id,
            "rank_runtime": ranks,
        }
        receipt_hashes[
            str(receipt_path.relative_to(Path(config.output_root)))
        ] = _file_sha256(receipt_path)
    payload = {
        "schema": COMPLETION_COVERAGE_SCHEMA,
        "package_plan_sha256": config.package_plan_sha256,
        "source_archive_sha256": config.source_archive_sha256,
        "expected_seed": config.expected_seed,
        "expected_fold_count": config.expected_fold_count,
        "completion_count": len(receipt_hashes),
        "receipt_sha256": receipt_hashes,
        "worker_runtime_proof": worker_runtime_proof,
        "development_only": True,
        "promotion_eligible": False,
    }
    payload["coverage_sha256"] = _content_sha256(payload)
    return payload


def _absence_snapshot(
    transport: KubectlTransport,
) -> tuple[bool, tuple[str, ...], dict[str, Any]]:
    job = transport.get_job(allow_absent=True)
    pods = transport.get_owned_pods()
    pod_uids: list[str] = []
    for pod in pods:
        metadata = _mapping(pod.get("metadata"), "Pod metadata")
        uid = metadata.get("uid")
        if not isinstance(uid, str):
            raise SeadragonLifecycleError("owned Pod UID is absent")
        pod_uids.append(uid)
    evidence = {
        "observed_at_utc": _utc_now(),
        "job_absent": job is None,
        "owned_pod_uids": sorted(pod_uids),
    }
    return job is None, tuple(sorted(pod_uids)), evidence


def _cleanup_exact_job(
    *,
    root: Path,
    binding: M03RV7AdmittedJobBinding,
    transport: KubectlTransport,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    request: M03RV7ExactJobCleanupRequest | None = None
    attempts: list[dict[str, Any]] = []
    first_absent = False
    first_pods: tuple[str, ...] = ()
    first_evidence: dict[str, Any] = {}
    for attempt in range(1, 4):
        fresh_job = transport.get_job(allow_absent=True)
        if fresh_job is None:
            if request is None:
                raise SeadragonLifecycleError(
                    "exact Job disappeared before a cleanup request was issued"
                )
            first_absent, first_pods, first_evidence = _absence_snapshot(transport)
            if first_absent and not first_pods:
                break
            raise SeadragonLifecycleError(
                "Job is absent while exact UID-owned Pods remain"
            )
        _job_identity(
            fresh_job,
            job_name=binding.job_name,
            run_id=binding.run_id,
            job_uid=binding.job_uid,
        )
        request = build_m03r_v7_exact_job_cleanup_request(binding, fresh_job)
        label = "" if attempt == 1 else f"-attempt-{attempt:02d}"
        request_path = root / f"cleanup-request{label}.json"
        options_path = root / f"delete-options{label}.json"
        request_file_sha = _exclusive_json(request_path, asdict(request))
        options_file_sha = _exclusive_json(options_path, request.delete_options)
        attempt_evidence: dict[str, Any] = {
            "attempt": attempt,
            "resource_version": request.resource_version,
            "request_file_sha256": request_file_sha,
            "delete_options_file_sha256": options_file_sha,
            "delete_error": None,
        }
        try:
            transport.delete(request, options_path)
        except SeadragonLifecycleError as exc:
            attempt_evidence["delete_error"] = str(exc)
            _exclusive_json(
                root / f"cleanup-delete-error-attempt-{attempt:02d}.json",
                attempt_evidence,
            )
            first_absent, first_pods, first_evidence = _absence_snapshot(transport)
            attempts.append(attempt_evidence)
            if first_absent and not first_pods:
                break
            if attempt == 3:
                raise SeadragonLifecycleError(
                    "exact cleanup exhausted fresh resourceVersion attempts"
                ) from exc
            continue
        attempts.append(attempt_evidence)
        deadline = time.monotonic() + 120
        while True:
            first_absent, first_pods, first_evidence = _absence_snapshot(transport)
            if first_absent and not first_pods:
                break
            if time.monotonic() >= deadline:
                raise SeadragonLifecycleError(
                    "exact cleanup did not reach absence"
                )
            sleep(2)
        break
    if request is None or not first_absent or first_pods:
        raise SeadragonLifecycleError("exact cleanup did not issue a valid deletion")
    sleep(1)
    second_absent, second_pods, second_evidence = _absence_snapshot(transport)
    verification = {
        "attempts": attempts,
        "first": first_evidence,
        "second": second_evidence,
    }
    verification_sha = _exclusive_json(
        root / "cleanup-verification.json",
        verification,
    )
    receipt = build_m03r_v7_exact_cleanup_receipt(
        request=request,
        first_job_absent=first_absent,
        second_job_absent=second_absent,
        first_owned_pod_uids=first_pods,
        second_owned_pod_uids=second_pods,
        verification_evidence_sha256=verification_sha,
    )
    _exclusive_json(root / "cleanup-receipt.json", asdict(receipt))


def capture_and_cleanup_failed_job(
    config: FailedJobCleanupConfig,
    *,
    transport: KubectlTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Preserve one exact failed bound Job and exact-clean it by UID/RV."""

    binding = _binding_from_file(
        Path(config.binding_path), config.binding_file_sha256
    )
    if (
        binding.job_name != config.job_name
        or binding.run_id != config.run_id
        or binding.job_uid != config.job_uid
    ):
        raise SeadragonLifecycleError("cleanup config does not match binding")
    root = _create_evidence_root(Path(config.evidence_root))
    live = transport or AttachOnlyKubectl(
        kubectl_path=config.kubectl_path,
        kubeconfig_path=config.kubeconfig_path,
        context=config.context,
        namespace=config.namespace,
        job_name=config.job_name,
        job_uid=config.job_uid,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    job = live.get_job()
    if job is None:  # pragma: no cover - get without ignore-not-found
        raise SeadragonLifecycleError("bound failed Job is absent before capture")
    _job_identity(
        job,
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=config.job_uid,
    )
    if _true_condition(job) != "Failed":
        raise SeadragonLifecycleError("bound Job is not terminal Failed")
    pods = live.get_owned_pods()
    _capture_terminal(
        root=root,
        job=job,
        pods=pods,
        transport=live,
        reason="preexisting-failed-job",
        log_limit_bytes=config.log_limit_bytes,
    )
    _cleanup_exact_job(
        root=root,
        binding=binding,
        transport=live,
        sleep=sleep,
    )


def _process_identity(pid: int) -> dict[str, Any]:
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        stat_text = stat_path.read_text(encoding="utf-8")
        command_line = cmdline_path.read_bytes()
    except OSError as exc:
        raise SeadragonLifecycleError("supervisor process identity disappeared") from exc
    end = stat_text.rfind(")")
    if end < 0:
        raise SeadragonLifecycleError("supervisor /proc stat is malformed")
    fields = stat_text[end + 2 :].split()
    if len(fields) <= 19:
        raise SeadragonLifecycleError("supervisor /proc stat is incomplete")
    return {
        "pid": pid,
        "pgrp": int(fields[2]),
        "session": int(fields[3]),
        "tty_nr": int(fields[4]),
        "start_ticks": int(fields[19]),
        "cmdline_sha256": hashlib.sha256(command_line).hexdigest(),
    }


def _validate_spawned_identity(payload: Mapping[str, Any], *, pid: int) -> None:
    observed = _process_identity(pid)
    for name in ("pid", "pgrp", "session", "tty_nr", "start_ticks", "cmdline_sha256"):
        if payload.get(name) != observed[name]:
            raise SeadragonLifecycleError("detached supervisor identity drifted")
    if observed["pgrp"] != pid or observed["session"] != pid or observed["tty_nr"] != 0:
        raise SeadragonLifecycleError("supervisor is not a detached session leader")


def _validated_launch_success(
    *,
    path: Path,
    root: Path,
    config: AttachSupervisorConfig,
    process_receipt: Mapping[str, Any],
    pid: int,
) -> Mapping[str, Any] | None:
    """Return a complete launch receipt, or ``None`` while it is being written.

    Receipt files are published with ``O_EXCL`` and then fsynced in place.  The
    directory entry therefore becomes visible before the final byte is
    necessarily durable.  A parent must not treat mere path existence as a
    successful activation handshake.
    """

    _regular_no_symlink(path, label="launch-success receipt")
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError:
        return None
    except OSError as exc:
        raise SeadragonLifecycleError(
            "launch-success receipt cannot be read"
        ) from exc
    launch = _mapping(payload, "launch-success receipt")
    activation_path = _regular_no_symlink(
        root / "activation.json", label="activation receipt"
    )
    runtime_activation_path = _regular_no_symlink(
        root / "activation-request-runtime.json",
        label="runtime activation request",
    )
    arm_path = _regular_no_symlink(root / "arm.json", label="arm receipt")
    activation = _mapping(
        _read_json_file(activation_path), "activation receipt"
    )
    runtime_activation = _activation_from_file(
        runtime_activation_path,
        _file_sha256(runtime_activation_path),
    )
    for name in (
        "activation_file_sha256",
        "capacity_receipt_sha256",
    ):
        value = launch.get(name)
        if not isinstance(value, str):
            raise SeadragonLifecycleError(
                f"launch-success receipt omitted {name}"
            )
        _require_sha256(name, value)
    for name in (
        "arm_file_sha256",
        "activation_request_sha256",
        "activated_job_sha256",
    ):
        value = activation.get(name)
        if not isinstance(value, str):
            raise SeadragonLifecycleError(
                f"activation receipt omitted {name}"
            )
        _require_sha256(name, value)
    if (
        launch.get("schema")
        != "rl-quant.top2000-m03r-v7-supervisor-launch-success-v1"
        or launch.get("job_name") != config.job_name
        or launch.get("job_uid") != config.job_uid
        or launch.get("run_id") != config.run_id
        or launch.get("parallelism") != config.parallelism
        or launch.get("gpus_per_worker") != config.gpus_per_worker
        or launch.get("request_ceiling")
        != config.parallelism * config.gpus_per_worker
        or launch.get("capacity_receipt_sha256")
        != config.capacity_receipt_sha256
        or launch.get("quota_pending_backfill_accepted") is not True
        or launch.get("activation_file_sha256")
        != _file_sha256(activation_path)
        or activation.get("schema")
        != "rl-quant.top2000-m03r-v7-supervisor-activation-v1"
        or activation.get("arm_file_sha256") != _file_sha256(arm_path)
        or activation.get("activation_request_sha256")
        != runtime_activation.request_sha256
        or runtime_activation.job_name != config.job_name
        or runtime_activation.job_uid != config.job_uid
        or runtime_activation.run_id != config.run_id
        or runtime_activation.namespace != config.namespace
        or runtime_activation.parallelism != config.parallelism
    ):
        raise SeadragonLifecycleError(
            "launch-success receipt does not bind the accepted exact activation"
        )
    _validate_spawned_identity(process_receipt, pid=pid)
    return launch


def _load_config(path: Path, expected_sha256: str) -> AttachSupervisorConfig:
    _require_project_path(path, label="attach config")
    payload = _mapping(
        _read_json_file(path, expected_sha256=expected_sha256),
        "attach config",
    )
    values = dict(payload)
    values["expected_completions"] = tuple(
        ExpectedCompletion(**dict(_mapping(row, "expected completion")))
        for row in cast(Sequence[Any], values.get("expected_completions", ()))
    )
    return AttachSupervisorConfig(**values)


def _load_failed_config(
    path: Path,
    expected_sha256: str,
) -> FailedJobCleanupConfig:
    _require_project_path(path, label="failed cleanup config")
    payload = _mapping(
        _read_json_file(path, expected_sha256=expected_sha256),
        "failed cleanup config",
    )
    return FailedJobCleanupConfig(**dict(payload))


def spawn_attach_supervisor(
    *,
    config_path: Path,
    config_sha256: str,
    wait: Callable[[float], None] = time.sleep,
) -> int:
    """Spawn the attach-only child and retain its process identity immediately."""

    config = _load_config(config_path, config_sha256)
    root = _create_evidence_root(Path(config.evidence_root))
    python = _regular_no_symlink(Path(config.host_python_path), label="host Python")
    kubectl = _regular_no_symlink(Path(config.kubectl_path), label="kubectl")
    kubeconfig = _regular_no_symlink(Path(config.kubeconfig_path), label="kubeconfig")
    pythonpath = _directory_no_symlink(Path(config.pythonpath), label="PYTHONPATH")
    source = _regular_no_symlink(Path(__file__).resolve(), label="supervisor source")
    command = (
        str(python),
        "-m",
        "rl_quant.training.top2000_m03r_v7_seadragon_lifecycle",
        "run",
        "--config",
        str(config_path),
        "--config-sha256",
        config_sha256,
    )
    command_sha = hashlib.sha256(b"\0".join(item.encode() for item in command)).hexdigest()
    intent = {
        "schema": "rl-quant.top2000-m03r-v7-supervisor-spawn-intent-v1",
        "created_at_utc": _utc_now(),
        "config_sha256": config_sha256,
        "command_sha256": command_sha,
        "python_sha256": _file_sha256(python),
        "kubectl_sha256": _file_sha256(kubectl),
        "supervisor_source_sha256": _file_sha256(source),
        "kubeconfig_metadata_validated": kubeconfig.is_file(),
    }
    intent_sha = _exclusive_json(root / "spawn-intent.json", intent)
    log_descriptor = os.open(
        root / "supervisor.log",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        _RECEIPT_MODE,
    )
    environment = {
        "KUBECONFIG": str(kubeconfig),
        "PATH": str(kubectl.parent),
        "PYTHONPATH": str(pythonpath),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
    finally:
        os.close(log_descriptor)
    identity = _process_identity(process.pid)
    process_receipt = {
        "schema": "rl-quant.top2000-m03r-v7-supervisor-process-v1",
        "spawn_intent_file_sha256": intent_sha,
        "config_sha256": config_sha256,
        "command_sha256": command_sha,
        **identity,
    }
    _validate_spawned_identity(process_receipt, pid=process.pid)
    _exclusive_json(root / "spawn-process.json", process_receipt)
    deadline = time.monotonic() + config.handshake_timeout_seconds
    launch_path = root / "launch-success.json"
    while True:
        if process.poll() is not None:
            raise SeadragonLifecycleError(
                "attach-only supervisor exited before launch success"
            )
        if launch_path.exists():
            launch = _validated_launch_success(
                path=launch_path,
                root=root,
                config=config,
                process_receipt=process_receipt,
                pid=process.pid,
            )
            if launch is not None:
                return process.pid
        if time.monotonic() >= deadline:
            raise SeadragonLifecycleError("attach-only launch handshake timed out")
        wait(0.25)


def _run_attach_supervisor_inner(
    config: AttachSupervisorConfig,
    *,
    config_sha256: str,
    hard_deadline: float,
    transport: KubectlTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Attach, activate, monitor, verify twelve completions, and exact-clean."""

    root = _directory_no_symlink(Path(config.evidence_root), label="evidence root")
    binding = _binding_from_file(
        Path(config.binding_path), config.binding_file_sha256
    )
    configured_activation = _activation_from_file(
        Path(config.activation_request_path), config.activation_request_file_sha256
    )
    if (
        binding.job_name != config.job_name
        or binding.run_id != config.run_id
        or binding.job_uid != config.job_uid
        or configured_activation.job_name != config.job_name
        or configured_activation.run_id != config.run_id
        or configured_activation.job_uid != config.job_uid
        or configured_activation.binding_receipt_sha256 != binding.receipt_sha256
    ):
        raise SeadragonLifecycleError("attach config, binding, or activation drifted")
    process_path = root / "spawn-process.json"
    deadline = monotonic() + config.handshake_timeout_seconds
    while not process_path.exists():
        if monotonic() >= deadline:
            raise SeadragonLifecycleError("spawn process receipt was not published")
        sleep(0.1)
    process_receipt = _mapping(_read_json_file(process_path), "spawn process receipt")
    _validate_spawned_identity(process_receipt, pid=os.getpid())
    live = transport or AttachOnlyKubectl(
        kubectl_path=config.kubectl_path,
        kubeconfig_path=config.kubeconfig_path,
        context=config.context,
        namespace=config.namespace,
        job_name=config.job_name,
        job_uid=config.job_uid,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    fresh = live.get_job()
    if fresh is None:  # pragma: no cover
        raise SeadragonLifecycleError("bound suspended Job is absent")
    _job_identity(
        fresh,
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=config.job_uid,
    )
    _job_artifact_identity(fresh, config)
    if live.get_owned_pods():
        raise SeadragonLifecycleError("attach target already owns Pods")
    activation = build_m03r_v7_exact_job_activation_request(binding, fresh)
    if not _same_activation_contract(configured_activation, activation):
        raise SeadragonLifecycleError(
            "fresh activation request drifted from configured immutable identity"
        )
    readiness = {
        "schema": "rl-quant.top2000-m03r-v7-supervisor-readiness-v1",
        "observed_at_utc": _utc_now(),
        "config_sha256": config_sha256,
        "binding_receipt_sha256": binding.receipt_sha256,
        "configured_activation_file_sha256": config.activation_request_file_sha256,
        "configured_activation_request_sha256": (
            configured_activation.request_sha256
        ),
        "runtime_activation_request_sha256": activation.request_sha256,
        "configured_resource_version": configured_activation.resource_version,
        "runtime_resource_version": activation.resource_version,
        "zero_owned_pods": True,
    }
    readiness_sha = _exclusive_json(root / "readiness.json", readiness)
    arm = {
        "schema": "rl-quant.top2000-m03r-v7-supervisor-arm-v1",
        "readiness_file_sha256": readiness_sha,
        "spawn_process_file_sha256": _file_sha256(process_path),
        "binding_receipt_sha256": binding.receipt_sha256,
        "capacity_receipt_sha256": config.capacity_receipt_sha256,
    }
    arm_sha = _exclusive_json(root / "arm.json", arm)
    _exclusive_json(root / "activation-request-runtime.json", asdict(activation))
    activated = live.activate(activation)
    _job_identity(
        activated,
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=config.job_uid,
    )
    _job_artifact_identity(activated, config)
    activated_spec = _mapping(activated.get("spec"), "activated Job spec")
    if activated_spec.get("suspend") is not False:
        raise SeadragonLifecycleError("activation did not unsuspend the exact Job")
    activation_receipt = {
        "schema": "rl-quant.top2000-m03r-v7-supervisor-activation-v1",
        "activated_at_utc": _utc_now(),
        "arm_file_sha256": arm_sha,
        "activation_request_sha256": activation.request_sha256,
        "activated_job_sha256": _content_sha256(activated),
    }
    activation_sha = _exclusive_json(root / "activation.json", activation_receipt)
    _validate_spawned_identity(process_receipt, pid=os.getpid())
    launch = {
        "schema": "rl-quant.top2000-m03r-v7-supervisor-launch-success-v1",
        "launched_at_utc": _utc_now(),
        "activation_file_sha256": activation_sha,
        "job_name": config.job_name,
        "job_uid": config.job_uid,
        "run_id": config.run_id,
        "parallelism": config.parallelism,
        "gpus_per_worker": config.gpus_per_worker,
        "request_ceiling": config.parallelism * config.gpus_per_worker,
        "capacity_receipt_sha256": config.capacity_receipt_sha256,
        "quota_pending_backfill_accepted": True,
    }
    _exclusive_json(root / "launch-success.json", launch)

    terminal_job: Mapping[str, Any] | None = None
    terminal_pods: tuple[Mapping[str, Any], ...] = ()
    terminal_reason = ""
    while monotonic() < hard_deadline:
        observed = live.get_job()
        if observed is None:
            raise SeadragonLifecycleError("exact Job disappeared before terminal capture")
        _job_identity(
            observed,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(observed, config)
        pods = live.get_owned_pods()
        condition = _true_condition(observed)
        if condition is not None:
            terminal_job = observed
            terminal_pods = pods
            terminal_reason = condition.lower()
            break
        sleep(config.poll_interval_seconds)
    if terminal_job is None:
        observed = live.get_job()
        if observed is None:
            raise SeadragonLifecycleError("exact Job disappeared at hard wall")
        terminal_job = observed
        terminal_pods = live.get_owned_pods()
        terminal_reason = "supervisor-hard-wall"
    _capture_terminal(
        root=root,
        job=terminal_job,
        pods=terminal_pods,
        transport=live,
        reason=terminal_reason,
        log_limit_bytes=config.log_limit_bytes,
    )
    if terminal_reason == "complete":
        coverage = validate_one_seed_completion_coverage(
            config,
            owned_pods=terminal_pods,
        )
        _exclusive_json(root / "completion-coverage.json", coverage)
    _cleanup_exact_job(
        root=root,
        binding=binding,
        transport=live,
        sleep=sleep,
    )
    if terminal_reason != "complete":
        raise SeadragonLifecycleError(
            f"research Job terminated without complete coverage: {terminal_reason}"
        )


def _recover_supervisor_failure(
    *,
    config: AttachSupervisorConfig,
    binding: M03RV7AdmittedJobBinding,
    transport: KubectlTransport,
    error: Exception,
    sleep: Callable[[float], None],
) -> None:
    """Best-effort evidence capture and exact cleanup after a child failure."""

    root = _directory_no_symlink(Path(config.evidence_root), label="evidence root")
    error_path = root / "supervisor-error.json"
    if not error_path.exists():
        _exclusive_json(
            error_path,
            {
                "schema": "rl-quant.top2000-m03r-v7-supervisor-error-v1",
                "observed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
    if (root / "cleanup-receipt.json").exists():
        return
    try:
        job = transport.get_job(allow_absent=True)
        if job is None:
            _exclusive_json(
                root / "failure-absence.json",
                {
                    "schema": "rl-quant.top2000-m03r-v7-failure-absence-v1",
                    "observed_at_utc": _utc_now(),
                    "job_absent": True,
                    "owned_pod_uids": [],
                },
            )
            return
        _job_identity(
            job,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(job, config)
        pods = transport.get_owned_pods()
        if not (root / "terminal-evidence.json").exists():
            _capture_terminal(
                root=root,
                job=job,
                pods=pods,
                transport=transport,
                reason="supervisor-error",
                log_limit_bytes=config.log_limit_bytes,
            )
        _cleanup_exact_job(
            root=root,
            binding=binding,
            transport=transport,
            sleep=sleep,
        )
    except Exception as recovery_error:  # noqa: BLE001 - preserve primary error
        recovery_path = root / "supervisor-recovery-error.json"
        if not recovery_path.exists():
            _exclusive_json(
                recovery_path,
                {
                    "schema": (
                        "rl-quant.top2000-m03r-v7-supervisor-recovery-error-v1"
                    ),
                    "observed_at_utc": _utc_now(),
                    "error_type": type(recovery_error).__name__,
                    "error": str(recovery_error),
                },
            )


def run_attach_supervisor(
    config: AttachSupervisorConfig,
    *,
    config_sha256: str,
    transport: KubectlTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    supervisor_started_monotonic: float | None = None,
) -> None:
    """Run one failure-preserving, attach-only exact-Job lifecycle."""

    started = (
        monotonic()
        if supervisor_started_monotonic is None
        else supervisor_started_monotonic
    )
    hard_deadline = started + config.hard_wall_seconds
    binding = _binding_from_file(
        Path(config.binding_path), config.binding_file_sha256
    )
    live = transport or AttachOnlyKubectl(
        kubectl_path=config.kubectl_path,
        kubeconfig_path=config.kubeconfig_path,
        context=config.context,
        namespace=config.namespace,
        job_name=config.job_name,
        job_uid=config.job_uid,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    try:
        _run_attach_supervisor_inner(
            config,
            config_sha256=config_sha256,
            hard_deadline=hard_deadline,
            transport=live,
            sleep=sleep,
            monotonic=monotonic,
        )
    except Exception as error:
        _recover_supervisor_failure(
            config=config,
            binding=binding,
            transport=live,
            error=error,
            sleep=sleep,
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("spawn", "run", "cleanup-failed"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    supervisor_started_monotonic = time.monotonic()
    args = _parser().parse_args(argv)
    _require_sha256("config_sha256", args.config_sha256)
    if args.command == "spawn":
        print(
            json.dumps(
                {"pid": spawn_attach_supervisor(
                    config_path=args.config,
                    config_sha256=args.config_sha256,
                )},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        attach_config = _load_config(args.config, args.config_sha256)
        run_attach_supervisor(
            attach_config,
            config_sha256=args.config_sha256,
            supervisor_started_monotonic=supervisor_started_monotonic,
        )
        return 0
    failed_config = _load_failed_config(args.config, args.config_sha256)
    capture_and_cleanup_failed_job(failed_config)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = [
    "ATTACH_CONFIG_SCHEMA",
    "FAILED_CLEANUP_CONFIG_SCHEMA",
    "AttachOnlyKubectl",
    "AttachSupervisorConfig",
    "ExpectedCompletion",
    "FailedJobCleanupConfig",
    "SeadragonLifecycleError",
    "canonical_one_seed_completions",
    "capture_and_cleanup_failed_job",
    "main",
    "run_attach_supervisor",
    "spawn_attach_supervisor",
    "validate_one_seed_completion_coverage",
]
