"""Suspended Seadragon Jobs for M03R-v8 predictive pretraining."""

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
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v8_pretraining_contract import (
    M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION,
)
from rl_quant.training.top2000_m03r_v8_pretraining_package import (
    M03RV8PretrainingPackagePlan,
)

M03R_V8_PRETRAINING_WORKER_ARGV_PREFIX = (
    "/opt/conda/envs/quanttrade/bin/python",
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--max-restarts=1",
    "--nproc-per-node=2",
    "-m",
    "rl_quant.workflows.top2000_m03r_v8_pretraining",
)
M03R_V8_PRETRAINING_QUALIFICATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-pretraining-qualification-ref-v1"
)
M03R_V8_PRETRAINING_LIVE_EVIDENCE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-pretraining-live-evidence-v1"
)


class M03RV8PretrainingKubernetesError(ValueError):
    """The v8 Job, qualification, or H100 request surface is invalid."""


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
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise M03RV8PretrainingKubernetesError(
            f"{name} must be a lowercase SHA-256"
        )


@dataclass(frozen=True, slots=True)
class M03RV8PretrainingLiveEvidence:
    """Fresh pre-render RBAC, cap, and H100-selector observation.

    Exact server dry-run evidence is intentionally not part of this object:
    the exact manifest does not exist until after this observation is consumed.
    """

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
    development_only: bool = True
    schema: str = M03R_V8_PRETRAINING_LIVE_EVIDENCE_SCHEMA

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
            "development_only": self.development_only,
            "rbac": asdict(self.rbac),
        }

    def __post_init__(self) -> None:
        try:
            observed = datetime.fromisoformat(self.observed_at_utc)
        except ValueError as exc:
            raise M03RV8PretrainingKubernetesError(
                "live observation timestamp must be ISO-8601"
            ) from exc
        if observed.tzinfo is None or observed.utcoffset() != UTC.utcoffset(observed):
            raise M03RV8PretrainingKubernetesError(
                "live observation timestamp must carry UTC timezone"
            )
        if (
            self.schema != M03R_V8_PRETRAINING_LIVE_EVIDENCE_SCHEMA
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.user_h100_cap != 16
            or isinstance(self.protected_or_other_committed_h100_count, bool)
            or not isinstance(self.protected_or_other_committed_h100_count, int)
            or not 0
            <= self.protected_or_other_committed_h100_count
            <= self.user_h100_cap
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
            or not self.development_only
        ):
            raise M03RV8PretrainingKubernetesError(
                "pre-render live RBAC/cap/H100 evidence is invalid"
            )
        _digest("live evidence receipt_sha256", self.receipt_sha256)
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV8PretrainingKubernetesError("live evidence hash mismatch")

    @property
    def allowed_parallelism(self) -> int:
        if not self.rbac.complete:
            return 0
        cap_remaining = max(
            0,
            self.user_h100_cap - self.protected_or_other_committed_h100_count,
        )
        return min(
            len(M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION),
            cap_remaining // 2,
        )

    def require_fresh(self, *, now_utc: datetime, max_age_seconds: int = 300) -> None:
        if now_utc.tzinfo is None:
            raise M03RV8PretrainingKubernetesError(
                "now_utc must be timezone-aware"
            )
        observed = datetime.fromisoformat(self.observed_at_utc)
        age = (now_utc.astimezone(UTC) - observed.astimezone(UTC)).total_seconds()
        if age < 0 or age > max_age_seconds:
            raise M03RV8PretrainingKubernetesError(
                "pre-render live evidence is stale or from the future"
            )


def build_m03r_v8_pretraining_live_evidence(
    *,
    observed_at_utc: str,
    rbac: M03RV7KubernetesRBACEvidence,
    protected_or_other_committed_h100_count: int,
    live_schedulable_free_h100_count: int | None,
    gpu_product_label_key: str,
    gpu_product_label_values: tuple[str, ...],
    live_h100_cap_verified: bool,
    gpu_selector_observed_live: bool,
) -> M03RV8PretrainingLiveEvidence:
    """Build the pre-render evidence without claiming a future dry run."""

    payload = {
        "schema": M03R_V8_PRETRAINING_LIVE_EVIDENCE_SCHEMA,
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
        "development_only": True,
        "rbac": asdict(rbac),
    }
    return M03RV8PretrainingLiveEvidence(
        observed_at_utc=observed_at_utc,
        rbac=rbac,
        protected_or_other_committed_h100_count=protected_or_other_committed_h100_count,
        live_schedulable_free_h100_count=live_schedulable_free_h100_count,
        gpu_product_label_key=gpu_product_label_key,
        gpu_product_label_values=gpu_product_label_values,
        live_h100_cap_verified=live_h100_cap_verified,
        gpu_selector_observed_live=gpu_selector_observed_live,
        receipt_sha256=_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class M03RV8PretrainingQualification:
    """Exact four-update setting-0 terminal evidence for full pretraining."""

    terminal_file_sha256: str
    terminal_receipt_sha256: str
    startup_receipt_sha256: str
    setting_plan_sha256: str
    world_size: int = 2
    gpus_per_worker: int = 2
    qualification_updates: int = 4
    exact_h100_80gb_per_rank: bool = True
    passed: bool = True
    development_only: bool = True
    promotion_eligible: bool = False
    schema: str = M03R_V8_PRETRAINING_QUALIFICATION_SCHEMA

    def validate_for(self, package: M03RV8PretrainingPackagePlan) -> None:
        package.validate()
        for name in (
            "terminal_file_sha256",
            "terminal_receipt_sha256",
            "startup_receipt_sha256",
            "setting_plan_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.setting_plan_sha256 != package.plans[0].receipt_sha256
            or self.world_size != 2
            or self.gpus_per_worker != 2
            or self.qualification_updates != 4
            or not self.exact_h100_80gb_per_rank
            or not self.passed
            or not self.development_only
            or self.promotion_eligible
            or self.schema != M03R_V8_PRETRAINING_QUALIFICATION_SCHEMA
        ):
            raise M03RV8PretrainingKubernetesError(
                "pretraining qualification does not bind this package"
            )


@dataclass(frozen=True, slots=True)
class M03RV8RenderedPretrainingJob:
    """One exact suspended qualification or seven-setting Job."""

    manifest: dict[str, Any]
    manifest_sha256: str
    pod_template_sha256: str
    package_plan_sha256: str
    live_evidence_sha256: str
    mode: Literal["qualification", "full"]
    completions: int
    parallelism: int
    gpus_per_completion: int = 2

    @property
    def request_ceiling_h100(self) -> int:
        return self.parallelism * self.gpus_per_completion


def _render(
    *,
    package: M03RV8PretrainingPackagePlan,
    live_evidence: M03RV8PretrainingLiveEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
    mode: Literal["qualification", "full"],
    qualification: M03RV8PretrainingQualification | None,
) -> M03RV8RenderedPretrainingJob:
    package.validate()
    live_evidence.require_fresh(now_utc=now_utc)
    if mode == "full":
        if qualification is None:
            raise M03RV8PretrainingKubernetesError(
                "full pretraining requires the four-update qualification"
            )
        qualification.validate_for(package)
    elif qualification is not None:
        raise M03RV8PretrainingKubernetesError(
            "qualification Job cannot consume its own future receipt"
        )
    if (
        template.package_mount_path != "/mnt/package"
        or template.output_mount_path != "/mnt/output"
        or PurePosixPath(package.plan_directory)
        != PurePosixPath(template.package_mount_path) / "plans"
        or PurePosixPath(package.source_pythonpath)
        != PurePosixPath(template.package_mount_path) / "source" / "src"
    ):
        raise M03RV8PretrainingKubernetesError(
            "package, plan, and source mounts are not exact"
        )
    completions = 1 if mode == "qualification" else len(
        M03R_V8_PRETRAINING_SETTING_INDEX_BY_COMPLETION
    )
    allowed_parallelism = live_evidence.allowed_parallelism
    if allowed_parallelism <= 0:
        raise M03RV8PretrainingKubernetesError(
            "live RBAC/cap evidence does not allow one two-H100 worker"
        )
    parallelism = (
        1
        if mode == "qualification"
        else min(completions, allowed_parallelism)
    )
    if parallelism <= 0 or parallelism * 2 > 16:
        raise M03RV8PretrainingKubernetesError("H100 request ceiling is invalid")
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v8-pretraining",
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
        "rl-quant/worker-source-sha256": package.artifacts.worker_source_sha256,
        "rl-quant/image-digest-sha256": package.artifacts.image_digest_sha256,
        "rl-quant/data-role": "development-only-nonreportable",
        "rl-quant/pretraining-mode": mode,
        "rl-quant/qualification-receipt-sha256": (
            "not-yet-created"
            if qualification is None
            else qualification.terminal_receipt_sha256
        ),
    }
    argv = [
        *M03R_V8_PRETRAINING_WORKER_ARGV_PREFIX,
        "--plan-directory",
        package.plan_directory,
    ]
    if mode == "qualification":
        argv.extend(["--completion-index", "0", "--qualification-updates", "4"])
    environment: list[dict[str, Any]] = [
        (
            {"name": "JOB_COMPLETION_INDEX", "value": "0"}
            if mode == "qualification"
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
    output_phase = (
        "pretraining-qualification" if mode == "qualification" else "pretraining-full"
    )
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
                                    "key": live_evidence.gpu_product_label_key,
                                    "operator": "In",
                                    "values": list(
                                        live_evidence.gpu_product_label_values
                                    ),
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
                            + output_phase
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
            {
                "name": "dshm",
                "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
            },
        ],
    }
    deadline = (
        M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS
        if mode == "qualification"
        else min(template.active_deadline_seconds, M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS)
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
    return M03RV8RenderedPretrainingJob(
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_sha256(pod_spec),
        package_plan_sha256=package.package_plan_sha256,
        live_evidence_sha256=live_evidence.receipt_sha256,
        mode=mode,
        completions=completions,
        parallelism=parallelism,
    )


def render_m03r_v8_suspended_pretraining_qualification_job(
    *,
    package: M03RV8PretrainingPackagePlan,
    live_evidence: M03RV8PretrainingLiveEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV8RenderedPretrainingJob:
    return _render(
        package=package,
        live_evidence=live_evidence,
        template=template,
        now_utc=now_utc,
        mode="qualification",
        qualification=None,
    )


def render_m03r_v8_suspended_pretraining_batch_job(
    *,
    package: M03RV8PretrainingPackagePlan,
    qualification: M03RV8PretrainingQualification,
    live_evidence: M03RV8PretrainingLiveEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV8RenderedPretrainingJob:
    return _render(
        package=package,
        live_evidence=live_evidence,
        template=template,
        now_utc=now_utc,
        mode="full",
        qualification=qualification,
    )


__all__ = [
    "M03R_V8_PRETRAINING_LIVE_EVIDENCE_SCHEMA",
    "M03R_V8_PRETRAINING_QUALIFICATION_SCHEMA",
    "M03R_V8_PRETRAINING_WORKER_ARGV_PREFIX",
    "M03RV8PretrainingKubernetesError",
    "M03RV8PretrainingLiveEvidence",
    "M03RV8PretrainingQualification",
    "M03RV8RenderedPretrainingJob",
    "build_m03r_v8_pretraining_live_evidence",
    "render_m03r_v8_suspended_pretraining_batch_job",
    "render_m03r_v8_suspended_pretraining_qualification_job",
]
