"""Suspended Seadragon Jobs for the frozen M03R-v12 post-hoc audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal, cast

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_POOL_NODE_SELECTOR,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03R_TOP2000_MULTI_GPU_TOLERATION,
    M03R_TOP2000_PRIORITY_CLASS_NAME,
    M03R_TOP2000_TERMINATION_MESSAGE_PATH,
    M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
    M03RV7AdmittedJobBinding,
    bind_m03r_v7_top2000_admitted_suspended_job,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_kubernetes import (
    M03RV11A15AuditLiveEvidence,
    M03RV11A15AuditTemplateConfig,
)
from rl_quant.training.top2000_m03r_v11_static_gate import (
    bind_m03r_v11_static_admitted_suspended_job,
)
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_package import (
    M03R_V12_POSTHOC_AUDIT_PARENT_OUTPUT_ROOT,
    M03R_V12_POSTHOC_AUDIT_SOURCE_PYTHONPATH,
    M03RV12PosthocAuditPackagePlan,
)

M03R_V12_POSTHOC_AUDIT_RENDERED_JOB_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-rendered-job-v1"
)
M03R_V12_POSTHOC_AUDIT_CAPACITY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-one-h100-capacity-v1"
)
M03R_V12_POSTHOC_AUDIT_IMAGE_PYTHON = "/opt/conda/envs/quanttrade/bin/python"


class M03RV12PosthocAuditKubernetesError(ValueError):
    """The audit manifest or one-H100 startup evidence drifted."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV12PosthocAuditKubernetesError(
            f"{name} must be one lowercase SHA-256"
        )


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditOneH100Capacity:
    audit_package_plan_sha256: str
    audit_package_plan_file_sha256: str
    static_receipt_sha256: str
    startup_file_sha256: str
    startup_receipt_sha256: str
    source_archive_sha256: str
    job_uid: str
    pod_uid: str
    image_id: str
    cleanup_receipt_file_sha256: str
    receipt_sha256: str
    visible_device_count: int = 1
    exact_h100_80gb: bool = True
    training_performed: bool = False
    checkpoint_selection_performed: bool = False
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    passed: bool = True
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V12_POSTHOC_AUDIT_CAPACITY_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate_for(self, package: M03RV12PosthocAuditPackagePlan) -> None:
        package.validate()
        for name in (
            "audit_package_plan_sha256",
            "audit_package_plan_file_sha256",
            "static_receipt_sha256",
            "startup_file_sha256",
            "startup_receipt_sha256",
            "source_archive_sha256",
            "cleanup_receipt_file_sha256",
        ):
            _digest(name, cast(str, getattr(self, name)))
        if (
            self.schema != M03R_V12_POSTHOC_AUDIT_CAPACITY_SCHEMA
            or self.audit_package_plan_sha256 != package.package_plan_sha256
            or self.source_archive_sha256 != package.artifacts.source_archive_sha256
            or not self.job_uid
            or not self.pod_uid
            or "@sha256:" not in self.image_id
            or self.visible_device_count != 1
            or not self.exact_h100_80gb
            or self.training_performed
            or self.checkpoint_selection_performed
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or not self.passed
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV12PosthocAuditKubernetesError(
                "v12 post-hoc one-H100 capacity evidence drifted"
            )


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditRenderedJob:
    manifest: dict[str, Any]
    manifest_sha256: str
    pod_template_sha256: str
    audit_package_plan_sha256: str
    audit_package_plan_file_sha256: str
    live_evidence_receipt_sha256: str
    mode: Literal["static", "capacity", "audit"]
    completions: int
    parallelism: int
    gpus_per_completion: int
    capacity_receipt_sha256: str
    training_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    economic_training_authorized: bool = False
    outer_2026_access_authorized: bool = False
    schema: str = M03R_V12_POSTHOC_AUDIT_RENDERED_JOB_SCHEMA

    def validate(self) -> None:
        if (
            self.schema != M03R_V12_POSTHOC_AUDIT_RENDERED_JOB_SCHEMA
            or self.manifest_sha256 != _sha256(self.manifest)
            or self.pod_template_sha256
            != _sha256(self.manifest["spec"]["template"]["spec"])
            or (self.mode, self.completions, self.parallelism, self.gpus_per_completion)
            not in {("static", 1, 1, 0), ("capacity", 1, 1, 1), ("audit", 3, 3, 1)}
            or self.training_authorized
            or self.checkpoint_selection_authorized
            or self.economic_training_authorized
            or self.outer_2026_access_authorized
        ):
            raise M03RV12PosthocAuditKubernetesError(
                "v12 post-hoc rendered Job drifted"
            )


def _args(
    package: M03RV12PosthocAuditPackagePlan,
    *,
    audit_plan_file_sha256: str,
    mode: Literal["static", "capacity", "audit"],
) -> list[str]:
    common = [
        "--audit-plan",
        "/mnt/audit-package/plans/audit-plan.json",
        "--audit-plan-file-sha256",
        audit_plan_file_sha256,
        "--parent-package-plan",
        package.parent_package_plan_path,
        "--parent-output-root",
        M03R_V12_POSTHOC_AUDIT_PARENT_OUTPUT_ROOT,
    ]
    if mode == "static":
        return [
            "-m",
            "rl_quant.workflows.top2000_m03r_v12_posthoc_inference_audit_static_validate",
            *common,
            "--output-path",
            "/mnt/output/static-result.json",
        ]
    if mode == "capacity":
        return [
            "-m",
            "rl_quant.workflows.top2000_m03r_v12_posthoc_inference_audit_startup_validate",
            *common,
            "--output-path",
            "/mnt/output/startup-result.json",
        ]
    return [
        "-m",
        package.runtime_entrypoint,
        *common,
        "--indexed-output-root",
        "/mnt/output",
    ]


def render_m03r_v12_posthoc_audit_suspended_job(
    *,
    package: M03RV12PosthocAuditPackagePlan,
    audit_plan_file_sha256: str,
    live: M03RV11A15AuditLiveEvidence,
    template: M03RV11A15AuditTemplateConfig,
    now_utc: datetime,
    mode: Literal["static", "capacity", "audit"],
    capacity: M03RV12PosthocAuditOneH100Capacity | None = None,
) -> M03RV12PosthocAuditRenderedJob:
    package.validate()
    _digest("audit_plan_file_sha256", audit_plan_file_sha256)
    live.require_fresh(now_utc=now_utc)
    if mode == "audit":
        if capacity is None:
            raise M03RV12PosthocAuditKubernetesError(
                "full post-hoc audit requires one-H100 capacity evidence"
            )
        capacity.validate_for(package)
        if capacity.audit_package_plan_file_sha256 != audit_plan_file_sha256:
            raise M03RV12PosthocAuditKubernetesError(
                "post-hoc capacity evidence binds another plan file"
            )
    elif capacity is not None:
        raise M03RV12PosthocAuditKubernetesError(
            "static/capacity render cannot consume later capacity evidence"
        )
    needed = 3 if mode == "audit" else (1 if mode == "capacity" else 0)
    if live.available_under_user_cap < needed:
        raise M03RV12PosthocAuditKubernetesError(
            "live user H100 cap does not permit this audit request"
        )
    completions, parallelism, gpus = {
        "static": (1, 1, 0),
        "capacity": (1, 1, 1),
        "audit": (3, 3, 1),
    }[mode]
    capacity_sha = "not-yet-created" if capacity is None else capacity.receipt_sha256
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v12-posthoc-audit",
        "app.kubernetes.io/managed-by": "receipt-gated-research",
        "k8s-user": "yding4",
        "rl-quant/run-id": template.run_id,
        "rl-quant/owner": "yding4",
    }
    if mode != "static":
        labels["runai/queue"] = template.runai_queue
    annotations = {
        "rl-quant/run-id": template.run_id,
        "rl-quant/package-plan-sha256": package.package_plan_sha256,
        "rl-quant/protocol-sha256": package.protocol_sha256,
        "rl-quant/source-archive-sha256": package.artifacts.source_archive_sha256,
        "rl-quant/parent-package-plan-sha256": package.parent.package_plan_sha256,
        "rl-quant/capacity-receipt-sha256": capacity_sha,
        "rl-quant/stage": f"v12-posthoc-audit-{mode}",
        "rl-quant/training-authorized": "false",
        "rl-quant/checkpoint-selection-authorized": "false",
        "rl-quant/economic-training-authorized": "false",
        "rl-quant/outer-2026-access-authorized": "false",
        "rl-quant/data-role": "posthoc-development-only-nonreportable",
    }
    environment: list[dict[str, Any]] = [
        {"name": "PYTHONPATH", "value": M03R_V12_POSTHOC_AUDIT_SOURCE_PYTHONPATH},
        {"name": "PYTHONNOUSERSITE", "value": "1"},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONHASHSEED", "value": "0"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]
    if mode == "static":
        environment.insert(0, {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"})
    elif mode == "capacity":
        environment.insert(0, {"name": "JOB_COMPLETION_INDEX", "value": "0"})
    else:
        environment.insert(
            0,
            {
                "name": "JOB_COMPLETION_INDEX",
                "valueFrom": {
                    "fieldRef": {
                        "apiVersion": "v1",
                        "fieldPath": (
                            "metadata.annotations['batch.kubernetes.io/"
                            "job-completion-index']"
                        ),
                    }
                },
            },
        )
    phase = {"static": "static", "capacity": "capacity", "audit": "audit"}[mode]
    mounts = [
        {
            "name": "research-data",
            "mountPath": "/mnt/audit-package",
            "subPath": f"{template.pvc_training_subpath}/packages/{template.run_id}",
            "readOnly": True,
        },
        {
            "name": "research-data",
            "mountPath": "/mnt/parent-package",
            "subPath": (
                f"{template.pvc_training_subpath}/packages/{package.parent.run_id}"
            ),
            "readOnly": True,
        },
        {
            "name": "research-data",
            "mountPath": "/mnt/parent-output",
            "subPath": (
                f"{template.pvc_training_subpath}/runs/{package.parent.run_id}/"
                "phases/v12-predictive"
            ),
            "readOnly": True,
        },
        {
            "name": "research-data",
            "mountPath": "/mnt/output",
            "subPath": (
                f"{template.pvc_training_subpath}/runs/{template.run_id}/phases/{phase}"
            ),
        },
        {"name": "tmp", "mountPath": "/tmp"},
        {"name": "dshm", "mountPath": "/dev/shm"},
    ]
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "serviceAccount": template.service_account_name,
        "serviceAccountName": template.service_account_name,
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "dnsPolicy": "ClusterFirst",
        "terminationGracePeriodSeconds": 60,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": template.run_as_user,
            "runAsGroup": template.run_as_group,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "validator" if mode == "static" else "auditor",
                "image": package.artifacts.image_reference,
                "imagePullPolicy": "IfNotPresent",
                "terminationMessagePath": M03R_TOP2000_TERMINATION_MESSAGE_PATH,
                "terminationMessagePolicy": M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
                "command": [M03R_V12_POSTHOC_AUDIT_IMAGE_PYTHON],
                "args": _args(
                    package,
                    audit_plan_file_sha256=audit_plan_file_sha256,
                    mode=mode,
                ),
                "env": environment,
                "resources": {
                    "requests": {
                        "cpu": "1" if mode == "static" else template.cpu_request,
                        "memory": "4Gi" if mode == "static" else template.memory_request,
                        "ephemeral-storage": (
                            "1Gi"
                            if mode == "static"
                            else template.ephemeral_storage_request
                        ),
                        "nvidia.com/gpu": str(gpus),
                    },
                    "limits": {
                        "cpu": "1" if mode == "static" else template.cpu_limit,
                        "memory": "4Gi" if mode == "static" else template.memory_limit,
                        "ephemeral-storage": (
                            "4Gi" if mode == "static" else template.ephemeral_storage_limit
                        ),
                        "nvidia.com/gpu": str(gpus),
                    },
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": mounts,
            }
        ],
        "volumes": [
            {
                "name": "research-data",
                "persistentVolumeClaim": {"claimName": template.pvc_claim_name},
            },
            {"name": "tmp", "emptyDir": {}},
            {
                "name": "dshm",
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": "1Gi" if mode == "static" else "16Gi",
                },
            },
        ],
    }
    if mode != "static":
        pod_spec.update(
            {
                "schedulerName": template.scheduler_name,
                "nodeSelector": dict(M03R_TOP2000_H100_POOL_NODE_SELECTOR),
                "priorityClassName": M03R_TOP2000_PRIORITY_CLASS_NAME,
                "tolerations": [dict(M03R_TOP2000_MULTI_GPU_TOLERATION)],
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": live.gpu_product_label_key,
                                            "operator": "In",
                                            "values": list(live.gpu_product_label_values),
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
            }
        )
    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": template.job_name,
            "namespace": M03R_TOP2000_KUBERNETES_NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "suspend": True,
            "completionMode": "Indexed",
            "completions": completions,
            "parallelism": parallelism,
            "backoffLimit": 0,
            "activeDeadlineSeconds": (
                1_800 if mode in {"static", "capacity"} else template.active_deadline_seconds
            ),
            "ttlSecondsAfterFinished": template.ttl_seconds_after_finished,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": pod_spec,
            },
        },
    }
    result = M03RV12PosthocAuditRenderedJob(
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_sha256(pod_spec),
        audit_package_plan_sha256=package.package_plan_sha256,
        audit_package_plan_file_sha256=audit_plan_file_sha256,
        live_evidence_receipt_sha256=live.receipt_sha256,
        mode=mode,
        completions=completions,
        parallelism=parallelism,
        gpus_per_completion=gpus,
        capacity_receipt_sha256=capacity_sha,
    )
    result.validate()
    return result


def bind_m03r_v12_posthoc_audit_admitted_suspended_job(
    *,
    rendered: M03RV12PosthocAuditRenderedJob,
    first_read: dict[str, Any],
    second_read: dict[str, Any],
    attached_owned_pod_uids: tuple[str, ...],
) -> M03RV7AdmittedJobBinding:
    rendered.validate()
    if rendered.mode == "static":
        return bind_m03r_v11_static_admitted_suspended_job(
            rendered=cast(Any, rendered),
            first_read=first_read,
            second_read=second_read,
            attached_owned_pod_uids=attached_owned_pod_uids,
        )
    return bind_m03r_v7_top2000_admitted_suspended_job(
        rendered=cast(Any, rendered),
        first_read=first_read,
        second_read=second_read,
        attached_owned_pod_uids=attached_owned_pod_uids,
    )


__all__ = [
    "M03R_V12_POSTHOC_AUDIT_CAPACITY_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_RENDERED_JOB_SCHEMA",
    "M03RV12PosthocAuditKubernetesError",
    "M03RV12PosthocAuditOneH100Capacity",
    "M03RV12PosthocAuditRenderedJob",
    "bind_m03r_v12_posthoc_audit_admitted_suspended_job",
    "render_m03r_v12_posthoc_audit_suspended_job",
]
