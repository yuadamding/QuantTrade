"""Suspended Seadragon Jobs for M03R-v9 predictive qualification only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_POOL_NODE_SELECTOR,
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03R_TOP2000_KUBERNETES_CONTEXT,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS,
    M03R_TOP2000_MULTI_GPU_TOLERATION,
    M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS,
    M03R_TOP2000_PRIORITY_CLASS_NAME,
    M03R_TOP2000_TERMINATION_MESSAGE_PATH,
    M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
    M03R_TOP2000_USER_H100_CAP,
    M03RV7AdmittedJobBinding,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
    M03RV7Top2000KubernetesError,
    bind_m03r_v7_top2000_admitted_suspended_job,
)
from rl_quant.training.top2000_m03r_v9_package import M03RV9PackagePlan

M03R_V9_KUBERNETES_LIVE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-live-kubernetes-evidence-v1"
)
M03R_V9_CAPACITY_QUALIFICATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-two-h100-capacity-qualification-v1"
)
M03R_V9_RENDERED_JOB_SCHEMA = "rl-quant.top2000-dev.m03r-v9-rendered-job-v1"
M03R_V9_WORKER_ARGV_PREFIX = (
    "/opt/conda/envs/quanttrade/bin/python",
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--max-restarts=0",
    "--nproc-per-node=2",
    "-m",
    "rl_quant.workflows.top2000_m03r_v9_predictive",
)


class M03RV9KubernetesError(ValueError):
    """The v9 manifest, live evidence, or capacity receipt drifted."""


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
        raise M03RV9KubernetesError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class M03RV9LiveEvidence:
    observed_at_utc: str
    rbac: M03RV7KubernetesRBACEvidence
    protected_or_other_committed_h100_count: int
    live_schedulable_free_h100_count: int | None
    gpu_product_label_key: str
    gpu_product_label_values: tuple[str, ...]
    live_h100_cap_verified: bool
    gpu_selector_observed_live: bool
    receipt_sha256: str
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    user_h100_cap: int = M03R_TOP2000_USER_H100_CAP
    exact_manifest_server_dry_run_deferred: bool = True
    research_only: bool = True
    development_only: bool = True
    schema: str = M03R_V9_KUBERNETES_LIVE_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "observed_at_utc": self.observed_at_utc,
            "context": self.context,
            "namespace": self.namespace,
            "user_h100_cap": self.user_h100_cap,
            "protected_or_other_committed_h100_count": (
                self.protected_or_other_committed_h100_count
            ),
            "live_schedulable_free_h100_count": self.live_schedulable_free_h100_count,
            "gpu_product_label_key": self.gpu_product_label_key,
            "gpu_product_label_values": list(self.gpu_product_label_values),
            "live_h100_cap_verified": self.live_h100_cap_verified,
            "gpu_selector_observed_live": self.gpu_selector_observed_live,
            "exact_manifest_server_dry_run_deferred": (
                self.exact_manifest_server_dry_run_deferred
            ),
            "research_only": self.research_only,
            "development_only": self.development_only,
            "rbac": asdict(self.rbac),
        }

    def __post_init__(self) -> None:
        try:
            observed = datetime.fromisoformat(self.observed_at_utc)
        except ValueError as exc:
            raise M03RV9KubernetesError("live timestamp must be ISO-8601") from exc
        if observed.tzinfo is None or observed.utcoffset() != UTC.utcoffset(observed):
            raise M03RV9KubernetesError("live timestamp must carry UTC timezone")
        if (
            self.schema != M03R_V9_KUBERNETES_LIVE_SCHEMA
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
            or self.gpu_product_label_key != M03R_TOP2000_H100_PRODUCT_LABEL_KEY
            or self.gpu_product_label_values != M03R_TOP2000_H100_PRODUCT_LABEL_VALUES
            or not self.live_h100_cap_verified
            or not self.gpu_selector_observed_live
            or not self.exact_manifest_server_dry_run_deferred
            or not self.research_only
            or not self.development_only
            or not self.rbac.complete
            or self.receipt_sha256 != _sha256(self.canonical_payload())
        ):
            raise M03RV9KubernetesError("v9 live Kubernetes evidence is invalid")

    @property
    def allowed_parallelism(self) -> int:
        remaining = max(
            0, self.user_h100_cap - self.protected_or_other_committed_h100_count
        )
        return min(3, remaining // 2)

    def require_fresh(self, *, now_utc: datetime, max_age_seconds: int = 300) -> None:
        if now_utc.tzinfo is None:
            raise M03RV9KubernetesError("now_utc must be timezone-aware")
        observed = datetime.fromisoformat(self.observed_at_utc)
        age = (now_utc.astimezone(UTC) - observed.astimezone(UTC)).total_seconds()
        if age < 0 or age > max_age_seconds:
            raise M03RV9KubernetesError("live Kubernetes evidence is stale")


def build_m03r_v9_live_evidence(
    *,
    observed_at_utc: str,
    rbac: M03RV7KubernetesRBACEvidence,
    protected_or_other_committed_h100_count: int,
    live_schedulable_free_h100_count: int | None,
    gpu_product_label_key: str,
    gpu_product_label_values: tuple[str, ...],
    live_h100_cap_verified: bool,
    gpu_selector_observed_live: bool,
) -> M03RV9LiveEvidence:
    payload = {
        "schema": M03R_V9_KUBERNETES_LIVE_SCHEMA,
        "observed_at_utc": observed_at_utc,
        "context": M03R_TOP2000_KUBERNETES_CONTEXT,
        "namespace": M03R_TOP2000_KUBERNETES_NAMESPACE,
        "user_h100_cap": M03R_TOP2000_USER_H100_CAP,
        "protected_or_other_committed_h100_count": (
            protected_or_other_committed_h100_count
        ),
        "live_schedulable_free_h100_count": live_schedulable_free_h100_count,
        "gpu_product_label_key": gpu_product_label_key,
        "gpu_product_label_values": list(gpu_product_label_values),
        "live_h100_cap_verified": live_h100_cap_verified,
        "gpu_selector_observed_live": gpu_selector_observed_live,
        "exact_manifest_server_dry_run_deferred": True,
        "research_only": True,
        "development_only": True,
        "rbac": asdict(rbac),
    }
    return M03RV9LiveEvidence(
        observed_at_utc=observed_at_utc,
        rbac=rbac,
        protected_or_other_committed_h100_count=(
            protected_or_other_committed_h100_count
        ),
        live_schedulable_free_h100_count=live_schedulable_free_h100_count,
        gpu_product_label_key=gpu_product_label_key,
        gpu_product_label_values=gpu_product_label_values,
        live_h100_cap_verified=live_h100_cap_verified,
        gpu_selector_observed_live=gpu_selector_observed_live,
        receipt_sha256=_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class M03RV9TwoH100CapacityQualification:
    terminal_file_sha256: str
    terminal_receipt_sha256: str
    startup_file_sha256: str
    terminal_evidence_file_sha256: str
    cleanup_receipt_file_sha256: str
    package_plan_sha256: str
    worker_plan_sha256: str
    world_size: int = 2
    gpus_per_worker: int = 2
    exact_h100_80gb_per_rank: bool = True
    nccl_process_group_initialized: bool = True
    training_performed: bool = False
    passed: bool = True
    research_only: bool = True
    development_only: bool = True
    promotion_eligible: bool = False
    schema: str = M03R_V9_CAPACITY_QUALIFICATION_SCHEMA

    def validate_for(self, package: M03RV9PackagePlan) -> None:
        package.validate()
        for name in (
            "terminal_file_sha256",
            "terminal_receipt_sha256",
            "startup_file_sha256",
            "terminal_evidence_file_sha256",
            "cleanup_receipt_file_sha256",
            "package_plan_sha256",
            "worker_plan_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.package_plan_sha256 != package.package_plan_sha256
            or self.worker_plan_sha256 != package.panel.workers[0].receipt_sha256
            or self.world_size != 2
            or self.gpus_per_worker != 2
            or not self.exact_h100_80gb_per_rank
            or not self.nccl_process_group_initialized
            or self.training_performed
            or not self.passed
            or not self.research_only
            or not self.development_only
            or self.promotion_eligible
            or self.schema != M03R_V9_CAPACITY_QUALIFICATION_SCHEMA
        ):
            raise M03RV9KubernetesError("capacity receipt does not bind this package")


@dataclass(frozen=True, slots=True)
class M03RV9RenderedJob:
    manifest: dict[str, Any]
    manifest_sha256: str
    pod_template_sha256: str
    package_plan_sha256: str
    live_evidence_sha256: str
    mode: Literal["capacity", "predictive"]
    completions: int
    parallelism: int
    gpus_per_completion: int = 2
    economic_panel_authorized: bool = False
    schema: str = M03R_V9_RENDERED_JOB_SCHEMA

    @property
    def request_ceiling_h100(self) -> int:
        return self.parallelism * self.gpus_per_completion


def bind_m03r_v9_admitted_suspended_job(
    *,
    rendered: M03RV9RenderedJob,
    first_read: dict[str, Any],
    second_read: dict[str, Any],
    attached_owned_pod_uids: tuple[str, ...],
) -> M03RV7AdmittedJobBinding:
    """Bind a v9 rendered Job through the reviewed exact v7 admission contract.

    The admitted Job schema is generation-independent: the desired manifest,
    two stable UID/spec reads, exact allowed admission mutations, suspension,
    and zero-Pod boundary are identical.  This explicit adapter avoids an
    untyped call at the launch boundary while preserving the reviewed receipt
    format used by activation and cleanup.
    """

    if (
        rendered.schema != M03R_V9_RENDERED_JOB_SCHEMA
        or rendered.mode not in {"capacity", "predictive"}
        or rendered.manifest_sha256 != _sha256(rendered.manifest)
        or rendered.package_plan_sha256
        != rendered.manifest.get("metadata", {})
        .get("annotations", {})
        .get("rl-quant/package-plan-sha256")
        or rendered.economic_panel_authorized
    ):
        raise M03RV9KubernetesError("rendered v9 Job identity drifted before binding")
    # The v7 binder is deliberately reused because its strict admitted-spec
    # allowlist and receipt schema are the shared Kubernetes trust boundary.
    try:
        return bind_m03r_v7_top2000_admitted_suspended_job(
            rendered=rendered,  # type: ignore[arg-type]
            first_read=first_read,
            second_read=second_read,
            attached_owned_pod_uids=attached_owned_pod_uids,
        )
    except M03RV7Top2000KubernetesError as exc:
        raise M03RV9KubernetesError(
            "v9 admitted suspended Job failed the shared strict contract"
        ) from exc


def _render(
    *,
    package: M03RV9PackagePlan,
    live: M03RV9LiveEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
    mode: Literal["capacity", "predictive"],
    capacity: M03RV9TwoH100CapacityQualification | None,
) -> M03RV9RenderedJob:
    package.validate()
    live.require_fresh(now_utc=now_utc)
    if mode == "predictive":
        if capacity is None:
            raise M03RV9KubernetesError(
                "predictive Job requires exact two-H100 capacity evidence"
            )
        capacity.validate_for(package)
    elif capacity is not None:
        raise M03RV9KubernetesError("capacity Job cannot consume future evidence")
    if (
        template.package_mount_path != "/mnt/package"
        or template.output_mount_path != "/mnt/output"
        or PurePosixPath(package.plan_directory)
        != PurePosixPath(template.package_mount_path) / "plans"
        or PurePosixPath(package.source_pythonpath)
        != PurePosixPath(template.package_mount_path) / "source" / "src"
    ):
        raise M03RV9KubernetesError("package and PVC mount identities drifted")
    completions = 1 if mode == "capacity" else 3
    if live.allowed_parallelism <= 0:
        raise M03RV9KubernetesError("live H100 cap does not allow one worker")
    parallelism = 1 if mode == "capacity" else min(3, live.allowed_parallelism)
    if parallelism * 2 > 6 or parallelism * 2 > 16:
        raise M03RV9KubernetesError("predictive H100 request ceiling drifted")
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v9-predictive",
        "app.kubernetes.io/managed-by": "receipt-gated-research",
        "k8s-user": "yding4",
        "rl-quant/run-id": template.run_id,
        "rl-quant/owner": "yding4",
        "runai/queue": template.runai_queue,
    }
    annotations = {
        "rl-quant/run-id": template.run_id,
        "rl-quant/package-plan-sha256": package.package_plan_sha256,
        "rl-quant/protocol-sha256": package.protocol_sha256,
        "rl-quant/source-archive-sha256": package.artifacts.source_archive_sha256,
        "rl-quant/cache-artifact-sha256": package.artifacts.cache_artifact_sha256,
        "rl-quant/risk-artifact-sha256": package.artifacts.risk_artifact_sha256,
        "rl-quant/projector-binding-sha256": package.artifacts.projector_binding_sha256,
        "rl-quant/image-digest-sha256": package.artifacts.image_digest_sha256,
        "rl-quant/stage": mode,
        "rl-quant/data-role": "development-only-nonreportable",
        "rl-quant/economic-panel-authorized": "false",
        "rl-quant/capacity-receipt-sha256": (
            "not-yet-created" if capacity is None else capacity.terminal_receipt_sha256
        ),
    }
    argv = [
        *M03R_V9_WORKER_ARGV_PREFIX,
        "--package-plan",
        package.plan_directory + "/package-plan.json",
        "--package-plan-sha256",
        package.package_plan_sha256,
    ]
    if mode == "capacity":
        argv.extend(("--completion-index", "0", "--startup-only"))
    environment: list[dict[str, Any]] = [
        (
            {"name": "JOB_COMPLETION_INDEX", "value": "0"}
            if mode == "capacity"
            else {
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
            }
        ),
        {"name": "NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
        {"name": "TORCH_NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
        {"name": "PYTHONNOUSERSITE", "value": "1"},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONHASHSEED", "value": "0"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "CUBLAS_WORKSPACE_CONFIG", "value": ":4096:8"},
        {"name": "OMP_NUM_THREADS", "value": "8"},
        {"name": "MKL_NUM_THREADS", "value": "8"},
        {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
        {"name": "TORCHINDUCTOR_CACHE_DIR", "value": "/tmp/torchinductor"},
        {"name": "TRITON_CACHE_DIR", "value": "/tmp/triton"},
        {"name": "PYTHONPATH", "value": package.source_pythonpath},
    ]
    phase = "v9-capacity" if mode == "capacity" else "v9-predictive"
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "serviceAccount": template.service_account_name,
        "serviceAccountName": template.service_account_name,
        "schedulerName": template.scheduler_name,
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "dnsPolicy": "ClusterFirst",
        "nodeSelector": dict(M03R_TOP2000_H100_POOL_NODE_SELECTOR),
        "priorityClassName": M03R_TOP2000_PRIORITY_CLASS_NAME,
        "terminationGracePeriodSeconds": 60,
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
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": template.run_as_user,
            "runAsGroup": template.run_as_group,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "trainer",
                "image": package.artifacts.image_reference,
                "imagePullPolicy": "IfNotPresent",
                "terminationMessagePath": M03R_TOP2000_TERMINATION_MESSAGE_PATH,
                "terminationMessagePolicy": M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
                "command": [argv[0]],
                "args": argv[1:],
                "env": environment,
                "resources": {
                    "requests": {
                        "cpu": template.cpu_request,
                        "memory": template.memory_request,
                        "ephemeral-storage": template.ephemeral_storage_request,
                        "nvidia.com/gpu": "2",
                    },
                    "limits": {
                        "cpu": template.cpu_limit,
                        "memory": template.memory_limit,
                        "ephemeral-storage": template.ephemeral_storage_limit,
                        "nvidia.com/gpu": "2",
                    },
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [
                    {
                        "name": "research-data",
                        "mountPath": template.package_mount_path,
                        "subPath": (
                            template.pvc_training_subpath.rstrip("/")
                            + "/packages/"
                            + template.run_id
                        ),
                        "readOnly": True,
                    },
                    {
                        "name": "research-data",
                        "mountPath": template.output_mount_path,
                        "subPath": (
                            template.pvc_training_subpath.rstrip("/")
                            + "/runs/"
                            + template.run_id
                            + "/phases/"
                            + phase
                        ),
                    },
                    {"name": "tmp", "mountPath": "/tmp"},
                    {"name": "dshm", "mountPath": "/dev/shm"},
                ],
            }
        ],
        "volumes": [
            {
                "name": "research-data",
                "persistentVolumeClaim": {"claimName": template.pvc_claim_name},
            },
            {"name": "tmp", "emptyDir": {}},
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
        ],
    }
    deadline = (
        M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS
        if mode == "capacity"
        else min(
            template.active_deadline_seconds, M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS
        )
    )
    manifest: dict[str, Any] = {
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
            "activeDeadlineSeconds": deadline,
            "ttlSecondsAfterFinished": template.ttl_seconds_after_finished,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": pod_spec,
            },
        },
    }
    return M03RV9RenderedJob(
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_sha256(pod_spec),
        package_plan_sha256=package.package_plan_sha256,
        live_evidence_sha256=live.receipt_sha256,
        mode=mode,
        completions=completions,
        parallelism=parallelism,
    )


def render_m03r_v9_suspended_capacity_job(
    *,
    package: M03RV9PackagePlan,
    live: M03RV9LiveEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV9RenderedJob:
    return _render(
        package=package,
        live=live,
        template=template,
        now_utc=now_utc,
        mode="capacity",
        capacity=None,
    )


def render_m03r_v9_suspended_predictive_job(
    *,
    package: M03RV9PackagePlan,
    capacity: M03RV9TwoH100CapacityQualification,
    live: M03RV9LiveEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV9RenderedJob:
    return _render(
        package=package,
        live=live,
        template=template,
        now_utc=now_utc,
        mode="predictive",
        capacity=capacity,
    )


__all__ = [
    "M03R_V9_CAPACITY_QUALIFICATION_SCHEMA",
    "M03R_V9_KUBERNETES_LIVE_SCHEMA",
    "M03R_V9_RENDERED_JOB_SCHEMA",
    "M03R_V9_WORKER_ARGV_PREFIX",
    "M03RV9KubernetesError",
    "M03RV9LiveEvidence",
    "M03RV9RenderedJob",
    "M03RV9TwoH100CapacityQualification",
    "bind_m03r_v9_admitted_suspended_job",
    "build_m03r_v9_live_evidence",
    "render_m03r_v9_suspended_capacity_job",
    "render_m03r_v9_suspended_predictive_job",
]
