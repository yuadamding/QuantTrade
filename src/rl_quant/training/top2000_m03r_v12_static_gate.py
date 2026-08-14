"""Receipt-gated same-image zero-GPU validation for M03R-v12.

The static gate has a deliberately smaller trust surface than the predictive
worker.  It accepts only the observed neutral or RunAI-mutated zero-GPU
profiles, binds a suspended Job before activation, validates the actual Pod and
application log, and never exposes a Kubernetes create operation.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as common
from rl_quant.training import top2000_m03r_v12_seadragon_lifecycle as lifecycle
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_ADMITTED_BINDING_SCHEMA,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03RV7AdmittedJobBinding,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.top2000_m03r_v12_kubernetes import M03RV12RenderedJob
from rl_quant.training.top2000_m03r_v12_static_contract import (
    M03R_V12_STATIC_RESULT_SCHEMA,
)

M03R_V12_STATIC_GATE_SCHEMA = "rl-quant.top2000-dev.m03r-v12-static-gate-v1"
M03R_V12_STATIC_ATTACH_CONFIG_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v12-static-attach-config-v1"
)
_RUNTIME_ANNOTATION_KEYS = {
    "batch.kubernetes.io/job-completion-index",
    "cni.projectcalico.org/containerID",
    "cni.projectcalico.org/podIP",
    "cni.projectcalico.org/podIPs",
    "pod-group-name",
    "received-resource-type",
    "runai-job-id",
}
_DYNAMIC_LABEL_KEYS = (
    "batch.kubernetes.io/controller-uid",
    "batch.kubernetes.io/job-name",
    "controller-uid",
    "job-name",
)


class M03RV12StaticGateError(RuntimeError):
    """The static admission, Pod, or application evidence drifted."""


class M03RV12StaticAttachOnlyKubectl(common.AttachOnlyKubectl):
    """Attach-only transport whose exact log surface is the static validator."""

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
        common._require_dns("pod_name", pod_name)
        return self._run(
            (
                "logs",
                pod_name,
                "--container",
                "validator",
                f"--limit-bytes={limit_bytes}",
            )
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M03RV12StaticGateError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _compact_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _newline_sha(value: Any) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def _identity(job: Mapping[str, Any]) -> tuple[str, str, str, str]:
    metadata = _mapping(job.get("metadata"), "Job metadata")
    values = tuple(
        metadata.get(name) for name in ("name", "namespace", "uid", "resourceVersion")
    )
    if not all(isinstance(value, str) and value for value in values):
        raise M03RV12StaticGateError("Job identity is incomplete")
    return cast(tuple[str, str, str, str], values)


def _resources(container: Mapping[str, Any]) -> dict[str, str]:
    resources = _mapping(container.get("resources"), "static resources")
    if set(resources) != {"requests", "limits"}:
        raise M03RV12StaticGateError("static resource sides drifted")
    names = {"cpu", "memory", "ephemeral-storage", "nvidia.com/gpu"}
    result: dict[str, str] = {}
    for side in ("requests", "limits"):
        values = _mapping(resources.get(side), f"static {side}")
        if set(values) != names:
            raise M03RV12StaticGateError("static resource-key inventory drifted")
        for name in names:
            result[f"{side}.{name}"] = str(values[name])
    if (
        result["requests.nvidia.com/gpu"] != "0"
        or result["limits.nvidia.com/gpu"] != "0"
    ):
        raise M03RV12StaticGateError("static GPU request and limit must both be zero")
    return result


def _validate_static_pod_spec(
    pod_spec: Mapping[str, Any],
    *,
    desired_pod_spec: Mapping[str, Any],
    actual: bool = False,
) -> dict[str, Any]:
    desired_containers = desired_pod_spec.get("containers")
    containers = pod_spec.get("containers")
    if (
        not isinstance(desired_containers, list)
        or len(desired_containers) != 1
        or not isinstance(containers, list)
        or len(containers) != 1
    ):
        raise M03RV12StaticGateError("static surface must contain one container")
    desired_container = _mapping(desired_containers[0], "desired validator")
    container = _mapping(containers[0], "admitted validator")
    allowed_container_keys = {
        "name",
        "image",
        "imagePullPolicy",
        "command",
        "args",
        "env",
        "resources",
        "securityContext",
        "volumeMounts",
        "terminationMessagePath",
        "terminationMessagePolicy",
    }
    if set(container) != allowed_container_keys:
        raise M03RV12StaticGateError("static container gained an executable surface")
    for key in allowed_container_keys - {"resources", "env"}:
        if container.get(key) != desired_container.get(key):
            raise M03RV12StaticGateError(
                f"static protected container field drifted: {key}"
            )
    desired_environment = desired_container.get("env")
    environment = container.get("env")
    expected_environment = desired_environment
    if actual and isinstance(desired_environment, list):
        expected_environment = [
            *desired_environment,
            {
                "name": "JOB_COMPLETION_INDEX",
                "valueFrom": {
                    "fieldRef": {
                        "apiVersion": "v1",
                        "fieldPath": (
                            "metadata.labels['batch.kubernetes.io/"
                            "job-completion-index']"
                        ),
                    }
                },
            },
        ]
    if (
        not isinstance(environment, list)
        or environment != expected_environment
        or {
            row.get("name"): row.get("value")
            for row in environment
            if isinstance(row, Mapping)
        }.get("NVIDIA_VISIBLE_DEVICES")
        != "none"
    ):
        raise M03RV12StaticGateError("static NVIDIA visibility mask drifted")
    profile = _resources(container)
    base_keys = set(desired_pod_spec)
    placement_keys = {"schedulerName", "nodeSelector", "priorityClassName"}
    actual_keys = {
        "hostname",
        "tolerations",
        "nodeName",
        "preemptionPolicy",
        "priority",
    }
    allowed_keys = base_keys | placement_keys | (actual_keys if actual else set())
    if set(pod_spec) - allowed_keys:
        raise M03RV12StaticGateError("static Pod gained an unallowlisted field")
    for key in base_keys - {"containers"}:
        if pod_spec.get(key) != desired_pod_spec.get(key):
            raise M03RV12StaticGateError(f"static protected Pod field drifted: {key}")
    placement = (
        pod_spec.get("schedulerName"),
        pod_spec.get("nodeSelector"),
        pod_spec.get("priorityClassName"),
    )
    cpu_memory = (
        profile["requests.cpu"],
        profile["requests.memory"],
        profile["limits.cpu"],
        profile["limits.memory"],
    )
    neutral = placement == (None, None, None) and cpu_memory == (
        "1",
        "4Gi",
        "1",
        "4Gi",
    )
    dry = placement == (
        "kai-scheduler",
        {"gpu-type": "A100"},
        "high-nonpreempting",
    ) and cpu_memory == (
        "0",
        "0",
        "0",
        "0",
    )
    created = placement == (
        "kai-scheduler",
        {"gpu-type": "A100"},
        "high-nonpreempting",
    ) and cpu_memory == ("0", "4Gi", "0", "4Gi")
    if not (neutral or dry or created):
        raise M03RV12StaticGateError("static placement/CPU/memory profile drifted")
    if (
        profile["requests.ephemeral-storage"] != "1Gi"
        or profile["limits.ephemeral-storage"] != "4Gi"
    ):
        raise M03RV12StaticGateError("static ephemeral-storage profile drifted")
    tolerations = pod_spec.get("tolerations", [])
    standard = [
        {
            "effect": "NoExecute",
            "key": "node.kubernetes.io/not-ready",
            "operator": "Exists",
            "tolerationSeconds": 300,
        },
        {
            "effect": "NoExecute",
            "key": "node.kubernetes.io/unreachable",
            "operator": "Exists",
            "tolerationSeconds": 300,
        },
    ]
    if actual:
        if tolerations not in ([], standard):
            raise M03RV12StaticGateError("actual static tolerations drifted")
        if (
            not isinstance(pod_spec.get("nodeName"), str)
            or not pod_spec.get("nodeName")
            or pod_spec.get("preemptionPolicy") != "Never"
            or (
                "priority" in pod_spec
                and (
                    isinstance(pod_spec.get("priority"), bool)
                    or not isinstance(pod_spec.get("priority"), int)
                )
            )
        ):
            raise M03RV12StaticGateError("actual static scheduling defaults drifted")
    elif tolerations:
        raise M03RV12StaticGateError("pre-Pod static tolerations must be empty")
    return {
        "resources": profile,
        "gpu_requests": 0,
        "gpu_limits": 0,
        "visibility_mask": "none",
        "unmasked_visibility_claimed": False,
        "profile": "neutral" if neutral else "dry" if dry else "created",
    }


def bind_m03r_v12_static_admitted_suspended_job(
    *,
    rendered: M03RV12RenderedJob,
    first_read: Mapping[str, Any],
    second_read: Mapping[str, Any],
    attached_owned_pod_uids: tuple[str, ...],
) -> M03RV7AdmittedJobBinding:
    if rendered.mode != "static" or rendered.gpus_per_completion != 0:
        raise M03RV12StaticGateError("static binder received a non-static Job")
    first_identity = _identity(first_read)
    second_identity = _identity(second_read)
    if first_identity[:3] != second_identity[:3] or attached_owned_pod_uids:
        raise M03RV12StaticGateError("static Job identity changed or already owns Pods")
    job_name, namespace, job_uid, second_rv = second_identity
    desired_metadata = _mapping(rendered.manifest.get("metadata"), "desired metadata")
    admitted_metadata = _mapping(second_read.get("metadata"), "admitted metadata")
    if (
        job_name != desired_metadata.get("name")
        or namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
        or admitted_metadata.get("labels") != desired_metadata.get("labels")
        or admitted_metadata.get("annotations") != desired_metadata.get("annotations")
    ):
        raise M03RV12StaticGateError("static Job metadata drifted")
    first_spec = _mapping(first_read.get("spec"), "first static spec")
    second_spec = _mapping(second_read.get("spec"), "second static spec")
    if _compact_sha(first_spec) != _compact_sha(second_spec):
        raise M03RV12StaticGateError("static Job spec changed between reads")
    desired_spec = _mapping(rendered.manifest.get("spec"), "desired static spec")
    if set(second_spec) != set(desired_spec) | {"selector"}:
        raise M03RV12StaticGateError("static Job spec gained an unknown field")
    for key in set(desired_spec) - {"template"}:
        if second_spec.get(key) != desired_spec.get(key):
            raise M03RV12StaticGateError(f"static Job geometry drifted: {key}")
    selector = _mapping(second_spec.get("selector"), "static selector")
    if dict(selector) != {
        "matchLabels": {"batch.kubernetes.io/controller-uid": job_uid}
    }:
        raise M03RV12StaticGateError("static controller selector drifted")
    desired_template = _mapping(desired_spec.get("template"), "desired template")
    admitted_template = _mapping(second_spec.get("template"), "admitted template")
    desired_template_metadata = _mapping(
        desired_template.get("metadata"), "desired template metadata"
    )
    admitted_template_metadata = _mapping(
        admitted_template.get("metadata"), "admitted template metadata"
    )
    expected_labels = {
        **dict(_mapping(desired_template_metadata.get("labels"), "desired labels")),
        **dict(
            zip(
                _DYNAMIC_LABEL_KEYS,
                (job_uid, job_name, job_uid, job_name),
                strict=True,
            )
        ),
        "runai/queue": "yding4-yn-gpu-workload-queue",
    }
    if (
        set(admitted_template_metadata)
        != {"labels", "annotations", "creationTimestamp"}
        or admitted_template_metadata.get("labels") != expected_labels
        or admitted_template_metadata.get("annotations")
        != desired_template_metadata.get("annotations")
        or admitted_template_metadata.get("creationTimestamp") is not None
    ):
        raise M03RV12StaticGateError("static template metadata drifted")
    admitted_pod_spec = _mapping(admitted_template.get("spec"), "admitted Pod spec")
    desired_pod_spec = _mapping(desired_template.get("spec"), "desired Pod spec")
    _validate_static_pod_spec(
        admitted_pod_spec,
        desired_pod_spec=desired_pod_spec,
    )
    run_id = _mapping(desired_metadata.get("annotations"), "desired annotations").get(
        "rl-quant/run-id"
    )
    if not isinstance(run_id, str) or not run_id:
        raise M03RV12StaticGateError("static run ID is absent")
    fields = {
        "job_name": job_name,
        "namespace": namespace,
        "job_uid": job_uid,
        "run_id": run_id,
        "first_resource_version": first_identity[3],
        "second_resource_version": second_rv,
        "parallelism": 1,
        "admitted_spec_sha256": _compact_sha(second_spec),
        "admitted_pod_template_sha256": _newline_sha(admitted_pod_spec),
        "admitted_selector_sha256": _compact_sha(selector),
        "admitted_template_metadata_sha256": _compact_sha(admitted_template_metadata),
        "desired_manifest_sha256": rendered.manifest_sha256,
        "attached_owned_pod_uids": (),
        "suspended": True,
        "schema": M03R_TOP2000_ADMITTED_BINDING_SCHEMA,
    }
    unsigned = M03RV7AdmittedJobBinding.__new__(M03RV7AdmittedJobBinding)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7AdmittedJobBinding(
        job_name=job_name,
        namespace=namespace,
        job_uid=job_uid,
        run_id=run_id,
        first_resource_version=first_identity[3],
        second_resource_version=second_rv,
        parallelism=1,
        admitted_spec_sha256=cast(str, fields["admitted_spec_sha256"]),
        admitted_pod_template_sha256=cast(str, fields["admitted_pod_template_sha256"]),
        admitted_selector_sha256=cast(str, fields["admitted_selector_sha256"]),
        admitted_template_metadata_sha256=cast(
            str, fields["admitted_template_metadata_sha256"]
        ),
        desired_manifest_sha256=rendered.manifest_sha256,
        attached_owned_pod_uids=(),
        suspended=True,
        schema=M03R_TOP2000_ADMITTED_BINDING_SCHEMA,
        receipt_sha256=_compact_sha(unsigned.canonical_payload()),
    )


def _valid_calico_ips(pod_ip: str, pod_ips: str) -> bool:
    if not pod_ip and not pod_ips:
        return True
    if not pod_ip or not pod_ips:
        return False
    try:
        primary = ipaddress.ip_interface(pod_ip)
        members = tuple(
            ipaddress.ip_interface(value.strip())
            for value in pod_ips.split(",")
            if value.strip()
        )
    except ValueError:
        return False
    return bool(members) and primary in members


def validate_m03r_v12_static_actual_pod(
    *,
    pod: Mapping[str, Any],
    terminal_job: Mapping[str, Any],
    rendered: M03RV12RenderedJob,
    job_uid: str,
) -> dict[str, Any]:
    metadata = _mapping(pod.get("metadata"), "actual Pod metadata")
    pod_name = metadata.get("name")
    pod_uid = metadata.get("uid")
    job_name = _mapping(terminal_job.get("metadata"), "terminal Job metadata").get(
        "name"
    )
    owners = metadata.get("ownerReferences")
    if (
        not isinstance(pod_name, str)
        or not pod_name.startswith(f"{job_name}-")
        or not isinstance(pod_uid, str)
        or not isinstance(owners, list)
        or not any(
            isinstance(row, Mapping)
            and row.get("uid") == job_uid
            and row.get("name") == job_name
            and row.get("kind") == "Job"
            and row.get("controller") is True
            for row in owners
        )
    ):
        raise M03RV12StaticGateError("actual static Pod ownership drifted")
    desired_metadata = _mapping(rendered.manifest.get("metadata"), "desired metadata")
    expected_labels = {
        **dict(_mapping(desired_metadata.get("labels"), "desired labels")),
        **dict(
            zip(
                _DYNAMIC_LABEL_KEYS,
                (job_uid, job_name, job_uid, job_name),
                strict=True,
            )
        ),
        "runai/queue": "yding4-yn-gpu-workload-queue",
        "batch.kubernetes.io/job-completion-index": "0",
    }
    if metadata.get("labels") != expected_labels:
        raise M03RV12StaticGateError("actual static Pod labels drifted")
    base_annotations = dict(
        _mapping(desired_metadata.get("annotations"), "desired annotations")
    )
    annotations = _mapping(metadata.get("annotations"), "actual Pod annotations")
    container_id = annotations.get("cni.projectcalico.org/containerID")
    pod_ip = annotations.get("cni.projectcalico.org/podIP")
    pod_ips = annotations.get("cni.projectcalico.org/podIPs")
    if (
        set(annotations) != set(base_annotations) | _RUNTIME_ANNOTATION_KEYS
        or any(annotations.get(key) != value for key, value in base_annotations.items())
        or not isinstance(container_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or not isinstance(pod_ip, str)
        or not isinstance(pod_ips, str)
        or not _valid_calico_ips(pod_ip, pod_ips)
        or annotations.get("batch.kubernetes.io/job-completion-index") != "0"
        or annotations.get("pod-group-name") != f"pg-{pod_name}-{job_uid}"
        or annotations.get("received-resource-type") != "Regular"
        or annotations.get("runai-job-id") != job_uid
    ):
        raise M03RV12StaticGateError("actual static Pod annotations drifted")
    created_spec = _mapping(
        _mapping(
            _mapping(terminal_job.get("spec"), "terminal Job spec").get("template"),
            "terminal template",
        ).get("spec"),
        "created static Pod spec",
    )
    actual_spec = _mapping(pod.get("spec"), "actual static Pod spec")
    if actual_spec.get("hostname") != f"{job_name}-0":
        raise M03RV12StaticGateError("actual static indexed hostname drifted")
    profile = _validate_static_pod_spec(
        actual_spec,
        desired_pod_spec=_mapping(
            _mapping(
                _mapping(rendered.manifest.get("spec"), "desired Job spec").get(
                    "template"
                ),
                "desired template",
            ).get("spec"),
            "desired Pod spec",
        ),
        actual=True,
    )
    created_container = _mapping(
        cast(list[Any], created_spec["containers"])[0], "created validator"
    )
    actual_container = _mapping(
        cast(list[Any], actual_spec["containers"])[0], "actual validator"
    )
    protected = {
        "name",
        "image",
        "imagePullPolicy",
        "command",
        "args",
        "resources",
        "securityContext",
        "volumeMounts",
        "terminationMessagePath",
        "terminationMessagePolicy",
    }
    if any(
        actual_container.get(key) != created_container.get(key) for key in protected
    ):
        raise M03RV12StaticGateError("actual static container differs from the Job")
    if actual_spec.get("volumes") != created_spec.get("volumes"):
        raise M03RV12StaticGateError("actual static volumes differ from the Job")
    status = _mapping(pod.get("status"), "actual Pod status")
    statuses = status.get("containerStatuses")
    if not isinstance(statuses, list) or len(statuses) != 1:
        raise M03RV12StaticGateError("actual static container status is absent")
    container_status = _mapping(statuses[0], "actual validator status")
    image_id = container_status.get("imageID")
    expected_digest = (
        _mapping(rendered.manifest.get("metadata"), "desired metadata")
        .get("annotations", {})
        .get("rl-quant/image-digest-sha256")
    )
    terminated = _mapping(
        _mapping(container_status.get("state"), "validator state").get("terminated"),
        "validator termination",
    )
    if (
        status.get("phase") != "Succeeded"
        or not isinstance(image_id, str)
        or not image_id.endswith(f"@sha256:{expected_digest}")
        or terminated.get("exitCode") != 0
    ):
        raise M03RV12StaticGateError("actual static image or exit status drifted")
    return {
        "pod_name": pod_name,
        "pod_uid": pod_uid,
        "job_name": job_name,
        "job_uid": job_uid,
        "image_id": image_id,
        "profile": profile,
        "protected_container_match": True,
        "pvc_binding_match": True,
    }


def validate_m03r_v12_static_log(
    log: bytes, *, rendered: M03RV12RenderedJob
) -> dict[str, Any]:
    if len(log) > 32768:
        raise M03RV12StaticGateError("static log exceeds the bounded contract")
    try:
        lines = [line for line in log.decode("utf-8").splitlines() if line]
        value = json.loads(lines[-1])
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
        raise M03RV12StaticGateError("static log is not one canonical result") from exc
    annotations = _mapping(rendered.manifest.get("metadata"), "desired metadata").get(
        "annotations", {}
    )
    if (
        not isinstance(value, dict)
        or value.get("schema") != M03R_V12_STATIC_RESULT_SCHEMA
        or value.get("package_plan_sha256") != rendered.package_plan_sha256
        or value.get("execution_authorization_receipt_sha256")
        != rendered.execution_authorization_receipt_sha256
        or value.get("source_archive_sha256")
        != annotations.get("rl-quant/source-archive-sha256")
        or value.get("image_digest_sha256")
        != annotations.get("rl-quant/image-digest-sha256")
        or value.get("gpu_mask") != "none"
        or value.get("gpu_requests") != 0
        or value.get("gpu_limits") != 0
        or value.get("unmasked_visibility_claimed") is not False
        or value.get("output_empty") is not True
        or value.get("container_started") is not True
        or value.get("training_performed") is not False
        or value.get("economic_training_authorized") is not False
        or value.get("outer_2026_access_authorized") is not False
        or value.get("development_only") is not True
        or value.get("reportable") is not False
        or value.get("promotion_eligible") is not False
    ):
        raise M03RV12StaticGateError("static application result drifted")
    return value


@dataclass(frozen=True, slots=True)
class M03RV12StaticGateReceipt:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    source_archive_sha256: str
    image_digest_sha256: str
    rendered_manifest_sha256: str
    server_dry_run_file_sha256: str
    created_binding_file_sha256: str
    terminal_evidence_file_sha256: str
    actual_pod_proof_file_sha256: str
    static_log_file_sha256: str
    cleanup_receipt_file_sha256: str
    gpu_requests: int = 0
    gpu_limits: int = 0
    visibility_mask: str = "none"
    unmasked_visibility_claimed: bool = False
    training_performed: bool = False
    h100_capacity_evidence: bool = False
    economic_training_authorized: bool = False
    outer_2026_access_authorized: bool = False
    passed: bool = True
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V12_STATIC_GATE_SCHEMA


@dataclass(frozen=True, slots=True)
class M03RV12StaticAttachConfig:
    job_name: str
    run_id: str
    job_uid: str
    rendered_path: str
    rendered_file_sha256: str
    binding_path: str
    binding_file_sha256: str
    activation_request_path: str
    activation_request_file_sha256: str
    package_plan_path: str
    package_plan_file_sha256: str
    package_plan_sha256: str
    execution_authorization_path: str
    execution_authorization_file_sha256: str
    execution_authorization_receipt_sha256: str
    source_archive_sha256: str
    static_source_sha256: str
    create_evidence_root: str
    server_dry_run_file_sha256: str
    evidence_root: str
    request_timeout_seconds: int = 30
    poll_interval_seconds: int = 5
    hard_wall_seconds: int = 900
    log_limit_bytes: int = 32768
    schema: str = M03R_V12_STATIC_ATTACH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "rendered_file_sha256",
            "binding_file_sha256",
            "activation_request_file_sha256",
            "package_plan_file_sha256",
            "package_plan_sha256",
            "execution_authorization_file_sha256",
            "execution_authorization_receipt_sha256",
            "source_archive_sha256",
            "static_source_sha256",
            "server_dry_run_file_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", cast(str, getattr(self, name))) is None:
                raise M03RV12StaticGateError(f"{name} must be a SHA-256")
        root = Path(lifecycle.SEADRAGON_QUANTTRADE_ROOT)
        for name in (
            "rendered_path",
            "binding_path",
            "activation_request_path",
            "package_plan_path",
            "execution_authorization_path",
            "create_evidence_root",
            "evidence_root",
        ):
            value = Path(cast(str, getattr(self, name)))
            if not value.is_absolute() or not value.is_relative_to(root):
                raise M03RV12StaticGateError(f"{name} leaves the project root")
        if (
            self.schema != M03R_V12_STATIC_ATTACH_CONFIG_SCHEMA
            or not self.job_name
            or not self.run_id
            or not self.job_uid
            or self.request_timeout_seconds < 5
            or self.poll_interval_seconds < 1
            or not 60 <= self.hard_wall_seconds <= 1800
            or self.log_limit_bytes < 4096
        ):
            raise M03RV12StaticGateError("static attach config drifted")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_static_config(
    path: Path, expected_file_sha256: str
) -> M03RV12StaticAttachConfig:
    if _file_sha(path) != expected_file_sha256:
        raise M03RV12StaticGateError("static config file hash drifted")
    try:
        return M03RV12StaticAttachConfig(**json.loads(path.read_bytes()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise M03RV12StaticGateError("static attach config is invalid") from exc


def _load_static_rendered(config: M03RV12StaticAttachConfig) -> M03RV12RenderedJob:
    path = common._regular_no_symlink(
        Path(config.rendered_path), label="static rendered"
    )
    if _file_sha(path) != config.rendered_file_sha256:
        raise M03RV12StaticGateError("static rendered file hash drifted")
    try:
        rendered = M03RV12RenderedJob(**json.loads(path.read_bytes()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise M03RV12StaticGateError("static rendered receipt is invalid") from exc
    annotations = _mapping(
        _mapping(rendered.manifest.get("metadata"), "rendered metadata").get(
            "annotations"
        ),
        "rendered annotations",
    )
    if (
        rendered.mode != "static"
        or rendered.gpus_per_completion != 0
        or rendered.completions != 1
        or rendered.parallelism != 1
        or rendered.package_plan_sha256 != config.package_plan_sha256
        or rendered.execution_authorization_receipt_sha256
        != config.execution_authorization_receipt_sha256
        or annotations.get("rl-quant/source-archive-sha256")
        != config.source_archive_sha256
    ):
        raise M03RV12StaticGateError("static rendered identity drifted")
    return rendered


def _validate_static_job_identity(
    job: Mapping[str, Any], config: M03RV12StaticAttachConfig
) -> None:
    metadata = _mapping(job.get("metadata"), "static Job metadata")
    annotations = _mapping(metadata.get("annotations"), "static Job annotations")
    if (
        metadata.get("name") != config.job_name
        or metadata.get("namespace") != M03R_TOP2000_KUBERNETES_NAMESPACE
        or metadata.get("uid") != config.job_uid
        or annotations.get("rl-quant/run-id") != config.run_id
        or annotations.get("rl-quant/package-plan-sha256") != config.package_plan_sha256
        or annotations.get("rl-quant/execution-authorization-sha256")
        != config.execution_authorization_receipt_sha256
        or annotations.get("rl-quant/source-archive-sha256")
        != config.source_archive_sha256
        or annotations.get("rl-quant/static-zero-gpu") != "true"
        or annotations.get("rl-quant/economic-panel-authorized") != "false"
    ):
        raise M03RV12StaticGateError("static Job identity drifted")


def run_m03r_v12_static_gate(
    config_path: str | Path,
    expected_config_sha256: str,
    *,
    transport: common.KubectlTransport | None = None,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> M03RV12StaticGateReceipt:
    """Attach, activate, prove, and exact-clean one static Job synchronously."""

    config = _load_static_config(Path(config_path), expected_config_sha256)
    source = common._regular_no_symlink(Path(__file__), label="static gate source")
    if _file_sha(source) != config.static_source_sha256:
        raise M03RV12StaticGateError("static gate source hash drifted")
    rendered = _load_static_rendered(config)
    package, authorization = lifecycle._load_package_authorization(config)  # type: ignore[arg-type]
    if (
        package.package_plan_sha256 != config.package_plan_sha256
        or authorization.receipt_sha256 != config.execution_authorization_receipt_sha256
    ):
        raise M03RV12StaticGateError("static package authorization drifted")
    binding = common._binding_from_file(
        Path(config.binding_path), config.binding_file_sha256
    )
    configured_activation = common._activation_from_file(
        Path(config.activation_request_path), config.activation_request_file_sha256
    )
    if (
        binding.job_name != config.job_name
        or binding.run_id != config.run_id
        or binding.job_uid != config.job_uid
        or binding.parallelism != 1
        or configured_activation.binding_receipt_sha256 != binding.receipt_sha256
    ):
        raise M03RV12StaticGateError("static binding/activation identity drifted")
    root = common._directory_no_symlink(
        Path(config.evidence_root), label="static evidence root"
    )
    if any(root.iterdir()):
        raise M03RV12StaticGateError("static evidence root must start empty")
    dry_path = common._regular_no_symlink(
        Path(config.create_evidence_root) / "server-dry-run.json",
        label="static server dry run",
    )
    if _file_sha(dry_path) != config.server_dry_run_file_sha256:
        raise M03RV12StaticGateError("static server dry-run hash drifted")
    live = transport or M03RV12StaticAttachOnlyKubectl(
        kubectl_path=lifecycle.SEADRAGON_KUBECTL,
        kubeconfig_path=lifecycle.SEADRAGON_KUBECONFIG,
        context="yding4_yn-gpu-workload@kubernetes-admin@kubernetes",
        namespace=M03R_TOP2000_KUBERNETES_NAMESPACE,
        job_name=config.job_name,
        job_uid=config.job_uid,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    activated = False
    cleaned = False
    hard_deadline = monotonic() + config.hard_wall_seconds
    try:
        reads: list[Mapping[str, Any]] = []
        for ordinal in range(2):
            job = live.get_job()
            pods = live.get_owned_pods()
            if job is None or pods:
                raise M03RV12StaticGateError(
                    "static activation requires a suspended zero-Pod Job"
                )
            _validate_static_job_identity(job, config)
            if (
                _compact_sha(_mapping(job.get("spec"), "static Job spec"))
                != binding.admitted_spec_sha256
            ):
                raise M03RV12StaticGateError(
                    "static Job spec drifted before activation"
                )
            reads.append(job)
            if ordinal == 0:
                sleep(0.1)
        activation = build_m03r_v7_exact_job_activation_request(binding, reads[-1])
        common._exclusive_json(
            root / "preactivation-zero-pods.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v12-static-preactivation-v1",
                "first_job_sha256": common._content_sha256(reads[0]),
                "second_job_sha256": common._content_sha256(reads[1]),
                "owned_pod_uids": [],
                "activation_request_sha256": activation.request_sha256,
            },
        )
        response = live.activate(activation)
        _validate_static_job_identity(response, config)
        if (
            _mapping(response.get("spec"), "activated static spec").get("suspend")
            is not False
        ):
            raise M03RV12StaticGateError("static activation did not unsuspend")
        activated = True
        common._exclusive_json(
            root / "activation.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v12-static-activation-v1",
                "activation_request_sha256": activation.request_sha256,
                "activated_job_sha256": common._content_sha256(response),
                "activation_retried": False,
            },
        )
        terminal_job: Mapping[str, Any] | None = None
        terminal_pods: tuple[Mapping[str, Any], ...] = ()
        while monotonic() < hard_deadline:
            job = live.get_job()
            if job is None:
                raise M03RV12StaticGateError("static Job disappeared before terminal")
            _validate_static_job_identity(job, config)
            pods = live.get_owned_pods()
            condition = common._true_condition(job)
            if condition is not None:
                terminal_job = job
                terminal_pods = pods
                if condition.lower() != "complete":
                    raise M03RV12StaticGateError(
                        f"static Job terminated without success: {condition.lower()}"
                    )
                break
            sleep(config.poll_interval_seconds)
        if terminal_job is None or len(terminal_pods) != 1:
            raise M03RV12StaticGateError("static Job did not complete before hard wall")
        common._capture_terminal(
            root=root,
            job=terminal_job,
            pods=terminal_pods,
            transport=live,
            reason="complete",
            log_limit_bytes=config.log_limit_bytes,
        )
        pod = terminal_pods[0]
        proof = validate_m03r_v12_static_actual_pod(
            pod=pod,
            terminal_job=terminal_job,
            rendered=rendered,
            job_uid=config.job_uid,
        )
        proof_sha = common._exclusive_json(root / "actual-pod-proof.json", proof)
        pod_name = cast(str, _mapping(pod.get("metadata"), "Pod metadata")["name"])
        _name, completion_index = common._pod_identity(pod)
        suffix = (
            f"index-{completion_index}-{pod_name}"
            if completion_index is not None
            else pod_name
        )
        captured_log = root / f"terminal-log-{suffix}.txt"
        raw_log = captured_log.read_bytes()
        validate_m03r_v12_static_log(raw_log, rendered=rendered)
        log_sha = _file_sha(captured_log)

        def validate_cleanup_job(job: Mapping[str, Any]) -> None:
            _validate_static_job_identity(job, config)

        lifecycle._cleanup_postactivation_exact(
            root=root,
            binding=binding,
            transport=live,
            request_timeout_seconds=config.request_timeout_seconds,
            validate_job=validate_cleanup_job,
            sleep=sleep,
        )
        cleaned = True
        receipt = M03RV12StaticGateReceipt(
            package_plan_sha256=config.package_plan_sha256,
            execution_authorization_receipt_sha256=(
                config.execution_authorization_receipt_sha256
            ),
            source_archive_sha256=config.source_archive_sha256,
            image_digest_sha256=cast(
                str,
                _mapping(rendered.manifest["metadata"], "rendered metadata")[
                    "annotations"
                ]["rl-quant/image-digest-sha256"],
            ),
            rendered_manifest_sha256=rendered.manifest_sha256,
            server_dry_run_file_sha256=config.server_dry_run_file_sha256,
            created_binding_file_sha256=config.binding_file_sha256,
            terminal_evidence_file_sha256=_file_sha(root / "terminal-evidence.json"),
            actual_pod_proof_file_sha256=proof_sha,
            static_log_file_sha256=log_sha,
            cleanup_receipt_file_sha256=_file_sha(root / "cleanup-receipt.json"),
        )
        common._exclusive_json(root / "static-gate-receipt.json", asdict(receipt))
        return receipt
    except Exception:
        if not cleaned:
            try:
                job = live.get_job(allow_absent=True)
                pods = live.get_owned_pods()
                if job is not None:
                    _validate_static_job_identity(job, config)
                    if activated or pods:
                        if not (root / "terminal-evidence.json").exists():
                            common._capture_terminal(
                                root=root,
                                job=job,
                                pods=pods,
                                transport=live,
                                reason="static-gate-error",
                                log_limit_bytes=config.log_limit_bytes,
                            )

                        def recovery_validate_cleanup_job(
                            fresh: Mapping[str, Any],
                        ) -> None:
                            _validate_static_job_identity(fresh, config)

                        lifecycle._cleanup_postactivation_exact(
                            root=root,
                            binding=binding,
                            transport=live,
                            request_timeout_seconds=config.request_timeout_seconds,
                            validate_job=recovery_validate_cleanup_job,
                            sleep=sleep,
                        )
                    else:
                        lifecycle._cleanup_preactivation_exact(
                            root=root,
                            config=cast(Any, config),
                            binding=binding,
                            transport=live,
                            sleep=sleep,
                        )
            except Exception as cleanup_error:
                if not (root / "static-cleanup-attach-required.json").exists():
                    common._exclusive_json(
                        root / "static-cleanup-attach-required.json",
                        {
                            "schema": "rl-quant.top2000-dev.m03r-v12-static-cleanup-attach-required-v1",
                            "error_type": type(cleanup_error).__name__,
                            "error": str(cleanup_error),
                            "attach_required": True,
                        },
                    )
                raise M03RV12StaticGateError(
                    "static cleanup is ambiguous; exact state retained"
                ) from cleanup_error
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run_m03r_v12_static_gate(args.config, args.config_sha256)
    print(
        json.dumps(
            {"status": "passed", "schema": receipt.schema},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V12_STATIC_ATTACH_CONFIG_SCHEMA",
    "M03R_V12_STATIC_GATE_SCHEMA",
    "M03RV12StaticAttachConfig",
    "M03RV12StaticGateError",
    "M03RV12StaticGateReceipt",
    "bind_m03r_v12_static_admitted_suspended_job",
    "main",
    "run_m03r_v12_static_gate",
    "validate_m03r_v12_static_actual_pod",
    "validate_m03r_v12_static_log",
]
