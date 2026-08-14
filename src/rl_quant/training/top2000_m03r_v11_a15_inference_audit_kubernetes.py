"""Suspended Seadragon Jobs for the frozen v11 a15 inference audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_POOL_NODE_SELECTOR,
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03R_TOP2000_KUBERNETES_CONTEXT,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03R_TOP2000_MULTI_GPU_TOLERATION,
    M03R_TOP2000_PRIORITY_CLASS_NAME,
    M03R_TOP2000_TERMINATION_MESSAGE_PATH,
    M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
    M03R_TOP2000_USER_H100_CAP,
    M03RV7AdmittedJobBinding,
    M03RV7KubernetesRBACEvidence,
    bind_m03r_v7_top2000_admitted_suspended_job,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_package import (
    M03R_V11_A15_AUDIT_PACKAGE_MOUNT,
    M03R_V11_A15_AUDIT_PARENT_OUTPUT_MOUNT,
    M03R_V11_A15_AUDIT_RUN_ID,
    M03RV11A15InferenceAuditAuthorization,
    M03RV11A15InferenceAuditPackagePlan,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan import (
    M03R_V11_A15_PARENT_RUN_ID,
    M03RV11A15InferenceAuditPlan,
)
from rl_quant.training.top2000_m03r_v11_static_gate import (
    bind_m03r_v11_static_admitted_suspended_job,
)

M03R_V11_A15_AUDIT_LIVE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-live-kubernetes-v1"
)
M03R_V11_A15_AUDIT_RENDERED_JOB_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-rendered-job-v1"
)
M03R_V11_A15_AUDIT_CAPACITY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-one-h100-capacity-v2"
)
M03R_V11_A15_AUDIT_IMAGE_PYTHON = "/opt/conda/envs/quanttrade/bin/python"


class M03RV11A15InferenceAuditKubernetesError(ValueError):
    """The audit launch evidence, manifest, or capacity proof drifted."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11A15InferenceAuditKubernetesError(
            f"{name} must be one lowercase SHA-256"
        )


def _dns_label(name: str, value: str) -> None:
    if (
        not value
        or len(value) > 63
        or not value[0].isalnum()
        or not value[-1].isalnum()
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in value
        )
    ):
        raise M03RV11A15InferenceAuditKubernetesError(
            f"{name} must be one Kubernetes DNS label"
        )


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditLiveEvidence:
    observed_at_utc: str
    rbac: M03RV7KubernetesRBACEvidence
    protected_or_other_committed_h100_count: int
    live_schedulable_free_h100_count: int | None
    live_h100_cap_verified: bool
    gpu_selector_observed_live: bool
    receipt_sha256: str
    gpu_product_label_key: str = M03R_TOP2000_H100_PRODUCT_LABEL_KEY
    gpu_product_label_values: tuple[str, ...] = M03R_TOP2000_H100_PRODUCT_LABEL_VALUES
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    user_h100_cap: int = M03R_TOP2000_USER_H100_CAP
    research_only: bool = True
    development_only: bool = True
    schema: str = M03R_V11_A15_AUDIT_LIVE_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        try:
            observed = datetime.fromisoformat(self.observed_at_utc)
        except ValueError as exc:
            raise M03RV11A15InferenceAuditKubernetesError(
                "audit live timestamp is not ISO-8601"
            ) from exc
        if (
            observed.tzinfo is None
            or observed.utcoffset() != UTC.utcoffset(observed)
            or self.schema != M03R_V11_A15_AUDIT_LIVE_SCHEMA
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.user_h100_cap != 16
            or isinstance(self.protected_or_other_committed_h100_count, bool)
            or not isinstance(self.protected_or_other_committed_h100_count, int)
            or not 0 <= self.protected_or_other_committed_h100_count <= 16
            or (
                self.live_schedulable_free_h100_count is not None
                and (
                    isinstance(self.live_schedulable_free_h100_count, bool)
                    or not isinstance(self.live_schedulable_free_h100_count, int)
                    or self.live_schedulable_free_h100_count < 0
                )
            )
            or not self.live_h100_cap_verified
            or not self.gpu_selector_observed_live
            or self.gpu_product_label_key != M03R_TOP2000_H100_PRODUCT_LABEL_KEY
            or self.gpu_product_label_values != M03R_TOP2000_H100_PRODUCT_LABEL_VALUES
            or not self.rbac.complete
            or not self.research_only
            or not self.development_only
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditKubernetesError(
                "audit live Kubernetes evidence drifted"
            )

    def require_fresh(self, *, now_utc: datetime, max_age_seconds: int = 300) -> None:
        self.validate()
        if now_utc.tzinfo is None:
            raise M03RV11A15InferenceAuditKubernetesError(
                "now_utc must be timezone-aware"
            )
        age = (
            now_utc.astimezone(UTC)
            - datetime.fromisoformat(self.observed_at_utc).astimezone(UTC)
        ).total_seconds()
        if age < 0 or age > max_age_seconds:
            raise M03RV11A15InferenceAuditKubernetesError(
                "audit live Kubernetes evidence is stale"
            )

    @property
    def available_under_user_cap(self) -> int:
        return max(0, self.user_h100_cap - self.protected_or_other_committed_h100_count)


def build_m03r_v11_a15_audit_live_evidence(
    *,
    observed_at_utc: str,
    rbac: M03RV7KubernetesRBACEvidence,
    protected_or_other_committed_h100_count: int,
    live_schedulable_free_h100_count: int | None,
    live_h100_cap_verified: bool,
    gpu_selector_observed_live: bool,
) -> M03RV11A15AuditLiveEvidence:
    provisional = M03RV11A15AuditLiveEvidence(
        observed_at_utc=observed_at_utc,
        rbac=rbac,
        protected_or_other_committed_h100_count=protected_or_other_committed_h100_count,
        live_schedulable_free_h100_count=live_schedulable_free_h100_count,
        live_h100_cap_verified=live_h100_cap_verified,
        gpu_selector_observed_live=gpu_selector_observed_live,
        receipt_sha256="0" * 64,
    )
    value = M03RV11A15AuditLiveEvidence(
        observed_at_utc=provisional.observed_at_utc,
        rbac=provisional.rbac,
        protected_or_other_committed_h100_count=(
            provisional.protected_or_other_committed_h100_count
        ),
        live_schedulable_free_h100_count=provisional.live_schedulable_free_h100_count,
        live_h100_cap_verified=provisional.live_h100_cap_verified,
        gpu_selector_observed_live=provisional.gpu_selector_observed_live,
        receipt_sha256=_sha256(provisional.unsigned_payload()),
    )
    value.validate()
    return value


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditTemplateConfig:
    job_name: str
    run_id: str
    service_account_name: str
    pvc_claim_name: str
    audit_package_mount_path: str = M03R_V11_A15_AUDIT_PACKAGE_MOUNT
    parent_package_mount_path: str = "/mnt/package"
    parent_output_mount_path: str = M03R_V11_A15_AUDIT_PARENT_OUTPUT_MOUNT
    output_mount_path: str = "/mnt/output"
    pvc_training_subpath: str = "quant/training"
    scheduler_name: str = "kai-scheduler"
    runai_queue: str = "yding4-yn-gpu-workload-queue"
    run_as_user: int = 307469
    run_as_group: int = 600815
    cpu_request: str = "24"
    cpu_limit: str = "24"
    memory_request: str = "200Gi"
    memory_limit: str = "200Gi"
    ephemeral_storage_request: str = "10Gi"
    ephemeral_storage_limit: str = "50Gi"
    active_deadline_seconds: int = 86_400
    ttl_seconds_after_finished: int = 86_400

    def __post_init__(self) -> None:
        for name in (
            "job_name",
            "run_id",
            "service_account_name",
            "pvc_claim_name",
            "scheduler_name",
            "runai_queue",
        ):
            _dns_label(name, cast(str, getattr(self, name)))
        mounts = (
            self.audit_package_mount_path,
            self.parent_package_mount_path,
            self.parent_output_mount_path,
            self.output_mount_path,
        )
        if (
            len(set(mounts)) != len(mounts)
            or any(
                not value.startswith("/") or ".." in value.split("/")
                for value in mounts
            )
            or PurePosixPath(self.pvc_training_subpath).is_absolute()
            or ".." in PurePosixPath(self.pvc_training_subpath).parts
            or self.service_account_name != "default"
            or self.run_as_user <= 0
            or self.run_as_group <= 0
            or not 60 <= self.active_deadline_seconds <= 216_000
            or self.ttl_seconds_after_finished <= 0
        ):
            raise M03RV11A15InferenceAuditKubernetesError(
                "audit Kubernetes template drifted"
            )


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditOneH100Capacity:
    static_gate_receipt_sha256: str
    capacity_terminal_file_sha256: str
    capacity_terminal_receipt_sha256: str
    startup_file_sha256: str
    cursor_artifact_file_sha256: str
    job_uid: str
    pod_uid: str
    image_id: str
    job_name: str
    run_id: str
    package_plan_sha256: str
    authorization_receipt_sha256: str
    audit_plan_receipt_sha256: str
    parent_cleanup_receipt_sha256: str
    source_archive_sha256: str
    cleanup_receipt_file_sha256: str
    receipt_sha256: str
    visible_device_count: int = 1
    exact_h100_80gb: bool = True
    full_execution_path_proven: bool = True
    training_performed: bool = False
    checkpoint_selection_performed: bool = False
    economic_training_authorized: bool = False
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    passed: bool = True
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V11_A15_AUDIT_CAPACITY_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        for name in (
            "static_gate_receipt_sha256",
            "capacity_terminal_file_sha256",
            "capacity_terminal_receipt_sha256",
            "startup_file_sha256",
            "cursor_artifact_file_sha256",
            "package_plan_sha256",
            "authorization_receipt_sha256",
            "audit_plan_receipt_sha256",
            "parent_cleanup_receipt_sha256",
            "source_archive_sha256",
            "cleanup_receipt_file_sha256",
        ):
            _digest(name, cast(str, getattr(self, name)))
        if (
            self.schema != M03R_V11_A15_AUDIT_CAPACITY_SCHEMA
            or not self.job_uid
            or not self.pod_uid
            or not self.job_name
            or not self.run_id
            or "@sha256:" not in self.image_id
            or self.visible_device_count != 1
            or not self.exact_h100_80gb
            or not self.full_execution_path_proven
            or self.training_performed
            or self.checkpoint_selection_performed
            or self.economic_training_authorized
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or not self.passed
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditKubernetesError(
                "one-H100 audit capacity receipt drifted"
            )


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditRenderedJob:
    manifest: dict[str, Any]
    manifest_sha256: str
    pod_template_sha256: str
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    audit_plan_receipt_sha256: str
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
    economic_panel_authorized: bool = False
    schema: str = M03R_V11_A15_AUDIT_RENDERED_JOB_SCHEMA

    def validate(self) -> None:
        if (
            self.schema != M03R_V11_A15_AUDIT_RENDERED_JOB_SCHEMA
            or self.mode not in {"static", "capacity", "audit"}
            or self.manifest_sha256 != _sha256(self.manifest)
            or self.pod_template_sha256
            != _sha256(self.manifest["spec"]["template"]["spec"])
            or (self.mode, self.completions, self.parallelism, self.gpus_per_completion)
            not in {("static", 1, 1, 0), ("capacity", 1, 1, 1), ("audit", 2, 2, 1)}
            or self.training_authorized
            or self.checkpoint_selection_authorized
            or self.economic_training_authorized
            or self.outer_2026_access_authorized
            or self.economic_panel_authorized
        ):
            raise M03RV11A15InferenceAuditKubernetesError("rendered audit Job drifted")


def _worker_args(
    package: M03RV11A15InferenceAuditPackagePlan,
    authorization: M03RV11A15InferenceAuditAuthorization,
    *,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    startup_only: bool,
) -> list[str]:
    args = [
        "-m",
        authorization.runtime_entrypoint,
        "--audit-plan",
        package.audit_plan_path,
        "--audit-plan-file-sha256",
        package.artifacts.audit_plan_file_sha256,
        "--package-plan",
        f"{M03R_V11_A15_AUDIT_PACKAGE_MOUNT}/plans/package-plan.json",
        "--package-plan-file-sha256",
        package_plan_file_sha256,
        "--authorization",
        f"{M03R_V11_A15_AUDIT_PACKAGE_MOUNT}/plans/execution-authorization.json",
        "--authorization-file-sha256",
        authorization_file_sha256,
        "--parent-package-plan",
        package.parent_package_plan_path,
        "--parent-authorization",
        package.parent_authorization_path,
        "--parent-output-root",
        package.parent_output_root,
        "--parent-lifecycle-root",
        package.parent_lifecycle_root,
        "--output-root",
        package.output_root,
    ]
    if startup_only:
        args.append("--startup-only")
    return args


def _static_args(
    package: M03RV11A15InferenceAuditPackagePlan,
    *,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
) -> list[str]:
    return [
        "-m",
        "rl_quant.workflows.top2000_m03r_v11_a15_inference_audit_static_validate",
        "--audit-package-root",
        M03R_V11_A15_AUDIT_PACKAGE_MOUNT,
        "--audit-plan-file-sha256",
        package.artifacts.audit_plan_file_sha256,
        "--package-plan-file-sha256",
        package_plan_file_sha256,
        "--authorization-file-sha256",
        authorization_file_sha256,
        "--parent-package-root",
        "/mnt/package",
        "--parent-output-root",
        package.parent_output_root,
        "--parent-lifecycle-root",
        package.parent_lifecycle_root,
        "--output-root",
        package.output_root,
    ]


def render_m03r_v11_a15_inference_audit_suspended_job(
    *,
    audit: M03RV11A15InferenceAuditPlan,
    package: M03RV11A15InferenceAuditPackagePlan,
    authorization: M03RV11A15InferenceAuditAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    live: M03RV11A15AuditLiveEvidence,
    template: M03RV11A15AuditTemplateConfig,
    now_utc: datetime,
    mode: Literal["static", "capacity", "audit"],
    capacity: M03RV11A15AuditOneH100Capacity | None = None,
) -> M03RV11A15AuditRenderedJob:
    package.validate(audit)
    authorization.validate(package, audit)
    _digest("package_plan_file_sha256", package_plan_file_sha256)
    _digest("authorization_file_sha256", authorization_file_sha256)
    if (
        authorization.package_plan_file_sha256 != package_plan_file_sha256
        or template.run_id != M03R_V11_A15_AUDIT_RUN_ID
        or template.run_id != package.run_id
        or template.job_name != package.job_name
    ):
        raise M03RV11A15InferenceAuditKubernetesError(
            "audit package, authorization, and template disagree"
        )
    live.require_fresh(now_utc=now_utc)
    if mode == "audit":
        if capacity is None:
            raise M03RV11A15InferenceAuditKubernetesError(
                "full audit requires one-H100 capacity evidence"
            )
        capacity.validate()
        if (
            capacity.package_plan_sha256 != package.package_plan_sha256
            or capacity.authorization_receipt_sha256 != authorization.receipt_sha256
            or capacity.audit_plan_receipt_sha256 != audit.receipt_sha256
            or capacity.parent_cleanup_receipt_sha256
            != audit.parent_cleanup_receipt_sha256
            or capacity.source_archive_sha256 != package.artifacts.source_archive_sha256
        ):
            raise M03RV11A15InferenceAuditKubernetesError(
                "one-H100 capacity receipt does not bind this audit package"
            )
    elif capacity is not None:
        raise M03RV11A15InferenceAuditKubernetesError(
            "static/capacity render cannot consume later capacity evidence"
        )
    if mode != "static" and live.available_under_user_cap < (
        2 if mode == "audit" else 1
    ):
        raise M03RV11A15InferenceAuditKubernetesError(
            "live user H100 cap does not permit the audit request ceiling"
        )

    completions, parallelism, gpus = {
        "static": (1, 1, 0),
        "capacity": (1, 1, 1),
        "audit": (2, 2, 1),
    }[mode]
    capacity_sha = "not-yet-created" if capacity is None else capacity.receipt_sha256
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v11-a15-audit",
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
        "rl-quant/execution-authorization-sha256": authorization.receipt_sha256,
        "rl-quant/audit-plan-sha256": audit.receipt_sha256,
        "rl-quant/protocol-sha256": package.protocol_sha256,
        "rl-quant/source-archive-sha256": package.artifacts.source_archive_sha256,
        "rl-quant/parent-package-plan-sha256": audit.parent_package_plan_sha256,
        "rl-quant/parent-cleanup-receipt-sha256": audit.parent_cleanup_receipt_sha256,
        "rl-quant/image-digest-sha256": package.artifacts.image_digest_sha256,
        "rl-quant/capacity-receipt-sha256": capacity_sha,
        "rl-quant/stage": f"a15-inference-audit-{mode}",
        "rl-quant/training-authorized": "false",
        "rl-quant/checkpoint-selection-authorized": "false",
        "rl-quant/economic-training-authorized": "false",
        "rl-quant/economic-panel-authorized": "false",
        "rl-quant/outer-2026-access-authorized": "false",
        "rl-quant/data-role": "posthoc-development-only-nonreportable",
    }
    args = (
        _static_args(
            package,
            package_plan_file_sha256=package_plan_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
        if mode == "static"
        else _worker_args(
            package,
            authorization,
            package_plan_file_sha256=package_plan_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
            startup_only=mode == "capacity",
        )
    )
    environment: list[dict[str, Any]] = [
        {"name": "PYTHONPATH", "value": package.source_pythonpath},
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
    volume_mounts = [
        {
            "name": "research-data",
            "mountPath": template.audit_package_mount_path,
            "subPath": (
                f"{template.pvc_training_subpath}/packages/{M03R_V11_A15_AUDIT_RUN_ID}"
            ),
            "readOnly": True,
        },
        {
            "name": "research-data",
            "mountPath": template.parent_package_mount_path,
            "subPath": (
                f"{template.pvc_training_subpath}/packages/{M03R_V11_A15_PARENT_RUN_ID}"
            ),
            "readOnly": True,
        },
        {
            "name": "research-data",
            "mountPath": template.parent_output_mount_path,
            "subPath": (
                f"{template.pvc_training_subpath}/runs/{M03R_V11_A15_PARENT_RUN_ID}/"
                "phases/v11-predictive"
            ),
            "readOnly": True,
        },
        {
            "name": "research-data",
            "mountPath": template.output_mount_path,
            "subPath": (
                f"{template.pvc_training_subpath}/runs/{M03R_V11_A15_AUDIT_RUN_ID}/"
                f"phases/{mode}"
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
                "command": [M03R_V11_A15_AUDIT_IMAGE_PYTHON],
                "args": args,
                "env": environment,
                "resources": {
                    "requests": {
                        "cpu": "1" if mode == "static" else template.cpu_request,
                        "memory": "4Gi"
                        if mode == "static"
                        else template.memory_request,
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
                            "4Gi"
                            if mode == "static"
                            else template.ephemeral_storage_limit
                        ),
                        "nvidia.com/gpu": str(gpus),
                    },
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": volume_mounts,
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
                                            "values": list(
                                                live.gpu_product_label_values
                                            ),
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
            "labels": dict(labels),
            "annotations": dict(annotations),
        },
        "spec": {
            "suspend": True,
            "completionMode": "Indexed",
            "completions": completions,
            "parallelism": parallelism,
            "backoffLimit": 0,
            "activeDeadlineSeconds": (
                1_800
                if mode in {"static", "capacity"}
                else template.active_deadline_seconds
            ),
            "ttlSecondsAfterFinished": template.ttl_seconds_after_finished,
            "template": {
                "metadata": {
                    "labels": dict(labels),
                    "annotations": dict(annotations),
                },
                "spec": pod_spec,
            },
        },
    }
    rendered = M03RV11A15AuditRenderedJob(
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_sha256(pod_spec),
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        audit_plan_receipt_sha256=audit.receipt_sha256,
        live_evidence_receipt_sha256=live.receipt_sha256,
        mode=mode,
        completions=completions,
        parallelism=parallelism,
        gpus_per_completion=gpus,
        capacity_receipt_sha256=capacity_sha,
    )
    rendered.validate()
    return rendered


def bind_m03r_v11_a15_audit_admitted_suspended_job(
    *,
    rendered: M03RV11A15AuditRenderedJob,
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
    "M03R_V11_A15_AUDIT_CAPACITY_SCHEMA",
    "M03R_V11_A15_AUDIT_LIVE_SCHEMA",
    "M03R_V11_A15_AUDIT_RENDERED_JOB_SCHEMA",
    "M03RV11A15AuditLiveEvidence",
    "M03RV11A15AuditOneH100Capacity",
    "M03RV11A15AuditRenderedJob",
    "M03RV11A15AuditTemplateConfig",
    "M03RV11A15InferenceAuditKubernetesError",
    "bind_m03r_v11_a15_audit_admitted_suspended_job",
    "build_m03r_v11_a15_audit_live_evidence",
    "render_m03r_v11_a15_inference_audit_suspended_job",
]
