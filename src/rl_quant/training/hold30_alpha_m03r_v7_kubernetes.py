"""Fail-closed Kubernetes lifecycle schemas for TOP2000 M03R-v7 development.

Everything here is a pure renderer or receipt validator.  There is no
``kubectl`` invocation, Kubernetes client, launch/unsuspend function, or delete
operation.  A caller must collect live RBAC/cap evidence externally, qualify the
exact worker and two-H100 execution surface, create only the suspended Job,
bind the admitted object, and retain exact-UID cleanup evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03RV7Top2000IndexPlan,
    M03RV7Top2000PackageError,
    M03RV7Top2000PackagePlan,
    M03RV7Top2000QualifiedPackage,
)

M03R_TOP2000_KUBERNETES_CONTEXT = (
    "yding4_yn-gpu-workload@kubernetes-admin@kubernetes"
)
M03R_TOP2000_KUBERNETES_NAMESPACE = "yn-gpu-workload"
M03R_TOP2000_INDEXED_COMPLETIONS = 12
M03R_TOP2000_GPUS_PER_COMPLETION = 2
M03R_TOP2000_PARALLELISM_HARD_CAP = 8
M03R_TOP2000_USER_H100_CAP = 16
M03R_TOP2000_H100_PRODUCT_LABEL_KEY = "nvidia.com/gpu.product"
M03R_TOP2000_H100_PRODUCT_LABEL_VALUES = (
    "NVIDIA-H100-80GB-HBM3",
)
M03R_TOP2000_H100_POOL_NODE_SELECTOR = {"gpu-type": "H100"}
M03R_TOP2000_MULTI_GPU_TOLERATION = {
    "effect": "NoSchedule",
    "key": "multi-gpu",
    "operator": "Equal",
    "value": "true",
}
M03R_TOP2000_PRIORITY_CLASS_NAME = "high-nonpreempting"
M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS = 216000
M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS = 86400
M03R_TOP2000_TERMINATION_MESSAGE_PATH = "/dev/termination-log"
M03R_TOP2000_TERMINATION_MESSAGE_POLICY = "File"
M03R_TOP2000_DYNAMIC_TEMPLATE_LABEL_KEYS = (
    "batch.kubernetes.io/controller-uid",
    "batch.kubernetes.io/job-name",
    "controller-uid",
    "job-name",
)
M03R_TOP2000_LIVE_EVIDENCE_SCHEMA = (
    "rl-quant.m03r-v7-top2000-live-kubernetes-evidence-v1"
)
M03R_TOP2000_SUSPENDED_JOB_SCHEMA = (
    "rl-quant.m03r-v7-top2000-suspended-indexed-job-v1"
)
M03R_TOP2000_ADMITTED_BINDING_SCHEMA = (
    "rl-quant.m03r-v7-top2000-admitted-job-binding-v1"
)
M03R_TOP2000_ACTIVATION_REQUEST_SCHEMA = (
    "rl-quant.m03r-v7-top2000-exact-job-activation-request-v1"
)
M03R_TOP2000_JSON_PATCH_CONTENT_TYPE = "application/json-patch+json"
M03R_TOP2000_INDEX_RECEIPT_SCHEMA = (
    "rl-quant.m03r-v7-top2000-index-receipt-v1"
)
M03R_TOP2000_PILOT_JOB_SCHEMA = (
    "rl-quant.m03r-v7-top2000-suspended-qualification-pilot-v1"
)
M03R_TOP2000_IMAGE_PYTHON = "/opt/conda/envs/quanttrade/bin/python"
M03R_TOP2000_WORKER_ARGV_PREFIX = (
    M03R_TOP2000_IMAGE_PYTHON,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--max-restarts=1",
    "--nproc-per-node=2",
    "-m",
    "rl_quant.workflows.top2000_m03r_v7_dev",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_FORBIDDEN_NODE_SELECTOR_KEYS = {
    "kubernetes.io/hostname",
    "beta.kubernetes.io/instance-type",
    "metadata.name",
    "spec.nodeName",
}
_RUN_ID_ANNOTATION = "rl-quant/run-id"


class M03RV7Top2000KubernetesError(ValueError):
    """A live admission, manifest, receipt, or cleanup invariant failed."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise M03RV7Top2000KubernetesError(
            "Kubernetes receipt is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise M03RV7Top2000KubernetesError(
            f"{name} must be one lowercase hexadecimal SHA-256"
        )


def _require_dns_label(name: str, value: str) -> None:
    if len(value) > 63 or _DNS_LABEL_RE.fullmatch(value) is None:
        raise M03RV7Top2000KubernetesError(
            f"{name} must be one Kubernetes DNS label"
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M03RV7Top2000KubernetesError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise M03RV7Top2000KubernetesError(
            "live observation timestamp must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise M03RV7Top2000KubernetesError(
            "live observation timestamp must carry UTC timezone"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class M03RV7KubernetesRBACEvidence:
    """Narrow verbs needed for suspended attach, inspection, and exact cleanup."""

    jobs_get: bool
    jobs_list: bool
    jobs_create: bool
    jobs_patch: bool
    jobs_delete: bool
    pods_get: bool
    pods_list: bool
    pods_watch: bool
    pod_logs_get: bool

    @property
    def complete(self) -> bool:
        return all(asdict(self).values())


@dataclass(frozen=True, slots=True)
class M03RV7LiveAdmissionEvidence:
    """One fresh, content-bound RBAC/cap/selector observation."""

    observed_at_utc: str
    rbac: M03RV7KubernetesRBACEvidence
    protected_or_other_committed_h100_count: int
    live_schedulable_free_h100_count: int | None
    gpu_product_label_key: str
    gpu_product_label_values: tuple[str, ...]
    live_h100_cap_verified: bool
    gpu_selector_observed_live: bool
    indexed_job_server_dry_run_passed: bool
    receipt_sha256: str
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    user_h100_cap: int = M03R_TOP2000_USER_H100_CAP
    evidence_source: Literal["live-kubectl-rbac-cap-and-selector"] = (
        "live-kubectl-rbac-cap-and-selector"
    )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_TOP2000_LIVE_EVIDENCE_SCHEMA,
            "observed_at_utc": self.observed_at_utc,
            "context": self.context,
            "namespace": self.namespace,
            "user_h100_cap": self.user_h100_cap,
            "protected_or_other_committed_h100_count": (
                self.protected_or_other_committed_h100_count
            ),
            "live_schedulable_free_h100_count": (
                self.live_schedulable_free_h100_count
            ),
            "gpu_product_label_key": self.gpu_product_label_key,
            "gpu_product_label_values": list(self.gpu_product_label_values),
            "live_h100_cap_verified": self.live_h100_cap_verified,
            "gpu_selector_observed_live": self.gpu_selector_observed_live,
            "indexed_job_server_dry_run_passed": (
                self.indexed_job_server_dry_run_passed
            ),
            "evidence_source": self.evidence_source,
            "rbac": asdict(self.rbac),
        }

    def __post_init__(self) -> None:
        _parse_utc(self.observed_at_utc)
        if (
            self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.user_h100_cap != 16
            or self.evidence_source != "live-kubectl-rbac-cap-and-selector"
        ):
            raise M03RV7Top2000KubernetesError(
                "live evidence must target the approved context, namespace, and cap"
            )
        if (
            isinstance(self.protected_or_other_committed_h100_count, bool)
            or not isinstance(self.protected_or_other_committed_h100_count, int)
            or self.protected_or_other_committed_h100_count < 0
            or self.protected_or_other_committed_h100_count > self.user_h100_cap
            or (
                self.live_schedulable_free_h100_count is not None
                and (
                    isinstance(self.live_schedulable_free_h100_count, bool)
                    or not isinstance(self.live_schedulable_free_h100_count, int)
                    or self.live_schedulable_free_h100_count < 0
                )
            )
        ):
            raise M03RV7Top2000KubernetesError("invalid live H100 accounting")
        if (
            self.gpu_product_label_key != M03R_TOP2000_H100_PRODUCT_LABEL_KEY
            or self.gpu_product_label_key in _FORBIDDEN_NODE_SELECTOR_KEYS
            or "hostname" in self.gpu_product_label_key.lower()
            or self.gpu_product_label_values
            != M03R_TOP2000_H100_PRODUCT_LABEL_VALUES
        ):
            raise M03RV7Top2000KubernetesError(
                "GPU scheduling must use an observed product label, never a node name"
            )
        if not (
            self.live_h100_cap_verified
            and self.indexed_job_server_dry_run_passed
        ):
            raise M03RV7Top2000KubernetesError(
                "live cap and exact Indexed Job server dry run are required"
            )
        _require_sha256("live evidence receipt_sha256", self.receipt_sha256)
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000KubernetesError("live evidence hash mismatch")

    @property
    def allowed_parallelism(self) -> int:
        if not self.rbac.complete:
            return 0
        cap_remaining = max(
            0,
            self.user_h100_cap - self.protected_or_other_committed_h100_count,
        )
        return min(
            M03R_TOP2000_PARALLELISM_HARD_CAP,
            M03R_TOP2000_INDEXED_COMPLETIONS,
            cap_remaining // M03R_TOP2000_GPUS_PER_COMPLETION,
        )

    def require_fresh(
        self,
        *,
        now_utc: datetime,
        max_age_seconds: int = 300,
        require_runtime_selector_proof: bool = True,
    ) -> None:
        if now_utc.tzinfo is None:
            raise M03RV7Top2000KubernetesError("now_utc must be timezone-aware")
        age = (
            now_utc.astimezone(UTC) - _parse_utc(self.observed_at_utc)
        ).total_seconds()
        if age < 0 or age > max_age_seconds:
            raise M03RV7Top2000KubernetesError(
                "live RBAC/cap evidence is stale or from the future"
            )
        if not self.rbac.complete or self.allowed_parallelism <= 0:
            raise M03RV7Top2000KubernetesError(
                "live RBAC/cap evidence admits no two-H100 completion"
            )
        if require_runtime_selector_proof and not self.gpu_selector_observed_live:
            raise M03RV7Top2000KubernetesError(
                "final training requires selector proof from actual pilot runtime"
            )


def build_m03r_v7_live_admission_evidence(
    *,
    observed_at_utc: str,
    rbac: M03RV7KubernetesRBACEvidence,
    protected_or_other_committed_h100_count: int,
    live_schedulable_free_h100_count: int | None,
    gpu_product_label_key: str,
    gpu_product_label_values: tuple[str, ...],
    live_h100_cap_verified: bool,
    gpu_selector_observed_live: bool,
    indexed_job_server_dry_run_passed: bool,
) -> M03RV7LiveAdmissionEvidence:
    """Bind externally collected live evidence without querying a cluster."""

    fields: dict[str, Any] = {
        "observed_at_utc": observed_at_utc,
        "rbac": rbac,
        "protected_or_other_committed_h100_count": (
            protected_or_other_committed_h100_count
        ),
        "live_schedulable_free_h100_count": live_schedulable_free_h100_count,
        "gpu_product_label_key": gpu_product_label_key,
        "gpu_product_label_values": gpu_product_label_values,
        "live_h100_cap_verified": live_h100_cap_verified,
        "gpu_selector_observed_live": gpu_selector_observed_live,
        "indexed_job_server_dry_run_passed": (
            indexed_job_server_dry_run_passed
        ),
        "context": M03R_TOP2000_KUBERNETES_CONTEXT,
        "namespace": M03R_TOP2000_KUBERNETES_NAMESPACE,
        "user_h100_cap": M03R_TOP2000_USER_H100_CAP,
        "evidence_source": "live-kubectl-rbac-cap-and-selector",
    }
    unsigned = M03RV7LiveAdmissionEvidence.__new__(M03RV7LiveAdmissionEvidence)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7LiveAdmissionEvidence(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class M03RV7KubernetesTemplateConfig:
    """Names and PVC paths supplied by the approved namespace owner."""

    job_name: str
    run_id: str
    service_account_name: str
    pvc_claim_name: str
    package_mount_path: str
    output_mount_path: str
    pvc_training_subpath: str = "quant/training"
    scheduler_name: str = "kai-scheduler"
    runai_queue: str = "yding4-yn-gpu-workload-queue"
    run_as_user: int = 307469
    run_as_group: int = 600815
    cpu_request: str = "48"
    cpu_limit: str = "48"
    memory_request: str = "400Gi"
    memory_limit: str = "400Gi"
    ephemeral_storage_request: str = "20Gi"
    ephemeral_storage_limit: str = "100Gi"
    active_deadline_seconds: int = M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS
    ttl_seconds_after_finished: int = 86400

    def __post_init__(self) -> None:
        for name in (
            "job_name",
            "run_id",
            "service_account_name",
            "pvc_claim_name",
            "scheduler_name",
            "runai_queue",
        ):
            _require_dns_label(name, getattr(self, name))
        for name in ("package_mount_path", "output_mount_path"):
            value = getattr(self, name)
            if not value.startswith("/") or value in {"/", ""} or ".." in value.split("/"):
                raise M03RV7Top2000KubernetesError(
                    f"{name} must be a scoped absolute container path"
                )
        if self.package_mount_path == self.output_mount_path:
            raise M03RV7Top2000KubernetesError(
                "package input and output paths must be distinct"
            )
        subpath = PurePosixPath(self.pvc_training_subpath)
        if (
            subpath.is_absolute()
            or ".." in subpath.parts
            or self.pvc_training_subpath in {"", "."}
        ):
            raise M03RV7Top2000KubernetesError(
                "PVC training subpath must be one scoped relative path"
            )
        if self.service_account_name != "default":
            raise M03RV7Top2000KubernetesError(
                "the qualified Seadragon execution profile uses service account default"
            )
        if self.run_as_user <= 0 or self.run_as_group <= 0:
            raise M03RV7Top2000KubernetesError(
                "research Pod UID/GID must be positive"
            )
        if (
            self.active_deadline_seconds <= 0
            or self.active_deadline_seconds
            > M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS
            or self.ttl_seconds_after_finished <= 0
        ):
            raise M03RV7Top2000KubernetesError(
                "Job deadline must use the proven bounded profile and TTL must be positive"
            )


def _assert_no_node_identity(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == "nodeName" or str(key) in _FORBIDDEN_NODE_SELECTOR_KEYS:
                raise M03RV7Top2000KubernetesError(
                    "rendered manifest contains a forbidden node identity"
                )
            _assert_no_node_identity(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_node_identity(child)


def _require_proven_h100_pod_profile(
    pod_spec: Mapping[str, Any],
    *,
    service_account_name: str,
) -> None:
    """Validate the explicit, previously admitted Seadragon two-H100 profile."""

    if (
        pod_spec.get("nodeSelector") != M03R_TOP2000_H100_POOL_NODE_SELECTOR
        or pod_spec.get("priorityClassName")
        != M03R_TOP2000_PRIORITY_CLASS_NAME
        or pod_spec.get("tolerations") != [M03R_TOP2000_MULTI_GPU_TOLERATION]
        or pod_spec.get("terminationGracePeriodSeconds") != 60
        or pod_spec.get("dnsPolicy") != "ClusterFirst"
        or pod_spec.get("serviceAccount") != service_account_name
        or pod_spec.get("serviceAccountName") != service_account_name
    ):
        raise M03RV7Top2000KubernetesError(
            "Pod spec does not match the proven Seadragon H100 pool profile"
        )
    containers = pod_spec.get("containers")
    if not isinstance(containers, list) or len(containers) != 1:
        raise M03RV7Top2000KubernetesError(
            "two-H100 execution requires one exact worker container"
        )
    container = _mapping(containers[0], "H100 worker container")
    if (
        container.get("terminationMessagePath")
        != M03R_TOP2000_TERMINATION_MESSAGE_PATH
        or container.get("terminationMessagePolicy")
        != M03R_TOP2000_TERMINATION_MESSAGE_POLICY
    ):
        raise M03RV7Top2000KubernetesError(
            "worker termination-message defaults must be explicit and bound"
        )


def _require_supported_indexed_job_spec(spec: Mapping[str, Any]) -> None:
    """Reject API fields stripped by this Seadragon Kubernetes version."""

    unsupported = {"backoffLimitPerIndex", "maxFailedIndexes"}.intersection(spec)
    if unsupported:
        raise M03RV7Top2000KubernetesError(
            "per-index backoff fields are unsupported by the admitted API profile"
        )


@dataclass(frozen=True, slots=True)
class M03RV7RenderedSuspendedJob:
    """Immutable JSON form of a suspended, never-activated Indexed Job."""

    manifest_json: str
    manifest_sha256: str
    pod_template_sha256: str
    package_plan_sha256: str
    execution_surface_sha256: str
    live_evidence_receipt_sha256: str
    capacity_receipt_sha256: str
    parallelism: int
    activation_authorized: bool = False

    def __post_init__(self) -> None:
        try:
            manifest = json.loads(self.manifest_json)
        except json.JSONDecodeError as exc:
            raise M03RV7Top2000KubernetesError("manifest JSON is invalid") from exc
        if not isinstance(manifest, dict):
            raise M03RV7Top2000KubernetesError("manifest must be a JSON object")
        if self.manifest_json != _canonical_json(manifest):
            raise M03RV7Top2000KubernetesError("manifest JSON is not canonical")
        for name in (
            "manifest_sha256",
            "pod_template_sha256",
            "package_plan_sha256",
            "execution_surface_sha256",
            "live_evidence_receipt_sha256",
            "capacity_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        spec = _mapping(manifest.get("spec"), "manifest.spec")
        _require_supported_indexed_job_spec(spec)
        deadline = spec.get("activeDeadlineSeconds")
        if (
            manifest.get("apiVersion") != "batch/v1"
            or manifest.get("kind") != "Job"
            or spec.get("completionMode") != "Indexed"
            or spec.get("completions") != 12
            or spec.get("parallelism") != self.parallelism
            or spec.get("backoffLimit") != 0
            or isinstance(deadline, bool)
            or not isinstance(deadline, int)
            or not 1 <= deadline <= M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS
            or spec.get("suspend") is not True
            or self.activation_authorized
            or not 1 <= self.parallelism <= 8
        ):
            raise M03RV7Top2000KubernetesError(
                "rendered Job must remain suspended, Indexed, fail-fast, and capped"
            )
        template = _mapping(spec.get("template"), "manifest.spec.template")
        pod_spec = _mapping(template.get("spec"), "manifest.spec.template.spec")
        _require_proven_h100_pod_profile(
            pod_spec,
            service_account_name="default",
        )
        if self.manifest_sha256 != _sha256(manifest):
            raise M03RV7Top2000KubernetesError("rendered manifest hash mismatch")
        if self.pod_template_sha256 != _canonical_pod_spec_sha256(pod_spec):
            raise M03RV7Top2000KubernetesError("rendered Pod template hash mismatch")
        _assert_no_node_identity(manifest)

    @property
    def manifest(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.manifest_json))


@dataclass(frozen=True, slots=True)
class M03RV7RenderedQualificationPilotJob:
    """One suspended two-H100 pilot rendered before qualification exists."""

    manifest_json: str
    manifest_sha256: str
    pod_spec_sha256: str
    package_plan_sha256: str
    live_evidence_receipt_sha256: str
    completion_index: int
    qualification_steps: Literal[4] = 4
    activation_authorized: bool = False
    schema: str = M03R_TOP2000_PILOT_JOB_SCHEMA

    def __post_init__(self) -> None:
        try:
            manifest = json.loads(self.manifest_json)
        except json.JSONDecodeError as exc:
            raise M03RV7Top2000KubernetesError(
                "pilot manifest JSON is invalid"
            ) from exc
        if not isinstance(manifest, dict) or self.manifest_json != _canonical_json(
            manifest
        ):
            raise M03RV7Top2000KubernetesError(
                "pilot manifest must be canonical JSON"
            )
        if self.schema != M03R_TOP2000_PILOT_JOB_SCHEMA:
            raise M03RV7Top2000KubernetesError("pilot schema drifted")
        for name in (
            "manifest_sha256",
            "pod_spec_sha256",
            "package_plan_sha256",
            "live_evidence_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        spec = _mapping(manifest.get("spec"), "pilot.spec")
        _require_supported_indexed_job_spec(spec)
        deadline = spec.get("activeDeadlineSeconds")
        template = _mapping(spec.get("template"), "pilot.spec.template")
        pod_spec = _mapping(template.get("spec"), "pilot.spec.template.spec")
        containers = pod_spec.get("containers")
        if not isinstance(containers, list) or len(containers) != 1:
            raise M03RV7Top2000KubernetesError(
                "pilot must contain one exact worker container"
            )
        container = _mapping(containers[0], "pilot worker")
        resources = _mapping(container.get("resources"), "pilot resources")
        requests = _mapping(resources.get("requests"), "pilot requests")
        limits = _mapping(resources.get("limits"), "pilot limits")
        args = container.get("args")
        required_suffix = [
            "--package-plan",
            "/mnt/package/package-plan.json",
            "--package-plan-sha256",
            self.package_plan_sha256,
            "--output-root",
            "/mnt/output",
            "--completion-index",
            str(self.completion_index),
            "--qualification-only",
            "--qualification-steps",
            "4",
            "--qualification-restart-after-step1",
        ]
        if (
            manifest.get("apiVersion") != "batch/v1"
            or manifest.get("kind") != "Job"
            or spec.get("completionMode") != "Indexed"
            or spec.get("completions") != 1
            or spec.get("parallelism") != 1
            or spec.get("backoffLimit") != 0
            or isinstance(deadline, bool)
            or not isinstance(deadline, int)
            or not 1 <= deadline <= M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS
            or spec.get("suspend") is not True
            or self.activation_authorized
            or not 0 <= self.completion_index < 12
            or self.qualification_steps != 4
            or container.get("command") != [M03R_TOP2000_WORKER_ARGV_PREFIX[0]]
            or args
            != list(M03R_TOP2000_WORKER_ARGV_PREFIX[1:]) + required_suffix
            or requests.get("nvidia.com/gpu") != "2"
            or limits.get("nvidia.com/gpu") != "2"
        ):
            raise M03RV7Top2000KubernetesError(
                "pilot must stay suspended, single-completion, two-H100, and four-update"
            )
        _require_proven_h100_pod_profile(
            pod_spec,
            service_account_name="default",
        )
        if self.manifest_sha256 != _sha256(manifest):
            raise M03RV7Top2000KubernetesError("pilot manifest hash mismatch")
        if self.pod_spec_sha256 != _canonical_pod_spec_sha256(pod_spec):
            raise M03RV7Top2000KubernetesError("pilot Pod-spec hash mismatch")
        _assert_no_node_identity(manifest)

    @property
    def manifest(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.manifest_json))


@dataclass(frozen=True, slots=True)
class M03RV7RenderedQualificationBatchJob:
    """Suspended twelve-index qualification batch under the sixteen-H100 cap."""

    manifest_json: str
    manifest_sha256: str
    pod_spec_sha256: str
    package_plan_sha256: str
    live_evidence_receipt_sha256: str
    parallelism: int
    qualification_steps: Literal[4] = 4
    activation_authorized: bool = False
    schema: str = M03R_TOP2000_PILOT_JOB_SCHEMA

    def __post_init__(self) -> None:
        try:
            manifest = json.loads(self.manifest_json)
        except json.JSONDecodeError as exc:
            raise M03RV7Top2000KubernetesError(
                "qualification batch manifest JSON is invalid"
            ) from exc
        if not isinstance(manifest, dict) or self.manifest_json != _canonical_json(
            manifest
        ):
            raise M03RV7Top2000KubernetesError(
                "qualification batch manifest must be canonical JSON"
            )
        spec = _mapping(manifest.get("spec"), "qualification batch spec")
        _require_supported_indexed_job_spec(spec)
        deadline = spec.get("activeDeadlineSeconds")
        template = _mapping(spec.get("template"), "qualification batch template")
        pod_spec = _mapping(template.get("spec"), "qualification batch Pod spec")
        containers = pod_spec.get("containers")
        if not isinstance(containers, list) or len(containers) != 1:
            raise M03RV7Top2000KubernetesError(
                "qualification batch must have one worker container"
            )
        container = _mapping(containers[0], "qualification batch worker")
        args = container.get("args")
        environment = container.get("env")
        expected_index_environment = {
            "name": "JOB_COMPLETION_INDEX",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": (
                        "metadata.annotations['batch.kubernetes.io/"
                        "job-completion-index']"
                    )
                }
            },
        }
        resources = _mapping(container.get("resources"), "qualification resources")
        requests = _mapping(resources.get("requests"), "qualification requests")
        limits = _mapping(resources.get("limits"), "qualification limits")
        if (
            manifest.get("apiVersion") != "batch/v1"
            or manifest.get("kind") != "Job"
            or spec.get("completionMode") != "Indexed"
            or spec.get("completions") != 12
            or spec.get("parallelism") != self.parallelism
            or not 1 <= self.parallelism <= 8
            or self.parallelism * 2 > 16
            or spec.get("suspend") is not True
            or spec.get("backoffLimit") != 0
            or isinstance(deadline, bool)
            or not isinstance(deadline, int)
            or not 1 <= deadline <= M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS
            or self.activation_authorized
            or self.qualification_steps != 4
            or not isinstance(args, list)
            or "--completion-index" in args
            or args[-4:]
            != [
                "--qualification-only",
                "--qualification-steps",
                "4",
                "--qualification-restart-after-step1",
            ]
            or not isinstance(environment, list)
            or environment.count(expected_index_environment) != 1
            or requests.get("nvidia.com/gpu") != "2"
            or limits.get("nvidia.com/gpu") != "2"
        ):
            raise M03RV7Top2000KubernetesError(
                "qualification batch geometry, cap, index map, or restart gate drifted"
            )
        _require_proven_h100_pod_profile(
            pod_spec,
            service_account_name="default",
        )
        for name in (
            "manifest_sha256",
            "pod_spec_sha256",
            "package_plan_sha256",
            "live_evidence_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.manifest_sha256 != _sha256(manifest):
            raise M03RV7Top2000KubernetesError(
                "qualification batch manifest hash mismatch"
            )
        if self.pod_spec_sha256 != _canonical_pod_spec_sha256(pod_spec):
            raise M03RV7Top2000KubernetesError(
                "qualification batch Pod-spec hash mismatch"
            )
        _assert_no_node_identity(manifest)

    @property
    def manifest(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.manifest_json))


def _canonical_pod_spec_sha256(value: Mapping[str, Any]) -> str:
    """Hash admitted Pod spec with the required ASCII JSON plus one newline."""

    try:
        encoded = (
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise M03RV7Top2000KubernetesError(
            "Pod spec is not canonical finite ASCII JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def render_m03r_v7_top2000_suspended_qualification_pilot_job(
    *,
    plan: M03RV7Top2000PackagePlan,
    completion_index: int,
    live_evidence: M03RV7LiveAdmissionEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV7RenderedQualificationPilotJob:
    """Render a pure suspended pilot from an unqualified immutable plan."""

    live_evidence.require_fresh(
        now_utc=now_utc,
        require_runtime_selector_proof=False,
    )
    if live_evidence.allowed_parallelism < 1:
        raise M03RV7Top2000KubernetesError(
            "live evidence admits no two-H100 qualification pilot"
        )
    if not 0 <= completion_index < len(plan.indices):
        raise M03RV7Top2000KubernetesError(
            "pilot completion index is outside the package plan"
        )
    if not plan.plan_artifact_path.startswith(
        template.package_mount_path.rstrip("/") + "/"
    ):
        raise M03RV7Top2000KubernetesError(
            "pilot package plan must be below the read-only package mount"
        )
    row = plan.indices[completion_index]
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v7-capacity-pilot",
        "app.kubernetes.io/managed-by": "receipt-gated-research",
        "k8s-user": "yding4",
        "rl-quant/run-id": template.run_id,
        "rl-quant/owner": "yding4",
        "runai/queue": template.runai_queue,
    }
    annotations = {
        "rl-quant/run-id": template.run_id,
        "rl-quant/package-plan-sha256": plan.package_plan_sha256,
        "rl-quant/source-archive-sha256": plan.artifacts.source_archive_sha256,
        "rl-quant/cache-artifact-sha256": plan.artifacts.cache_artifact_sha256,
        "rl-quant/image-digest-sha256": plan.artifacts.image_digest_sha256,
        "rl-quant/live-evidence-receipt-sha256": live_evidence.receipt_sha256,
        "rl-quant/qualification-completion-index": str(completion_index),
        "rl-quant/qualification-setting-index": str(row.setting_index),
        "rl-quant/qualification-setting-id": row.development_setting_id,
        "rl-quant/qualification-steps": "4",
        "rl-quant/intentional-restart-after-step": "1",
        "rl-quant/data-role": "development-only-nonreportable",
    }
    argv = list(M03R_TOP2000_WORKER_ARGV_PREFIX) + [
        "--package-plan",
        plan.plan_artifact_path,
        "--package-plan-sha256",
        plan.package_plan_sha256,
        "--output-root",
        template.output_mount_path,
        "--completion-index",
        str(completion_index),
        "--qualification-only",
        "--qualification-steps",
        "4",
        "--qualification-restart-after-step1",
    ]
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
            "completions": 1,
            "parallelism": 1,
            "backoffLimit": 0,
            "activeDeadlineSeconds": min(
                template.active_deadline_seconds,
                M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS,
            ),
            "ttlSecondsAfterFinished": template.ttl_seconds_after_finished,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": {
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
                                                "key": M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
                                                "operator": "In",
                                                "values": list(
                                                    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES
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
                            "image": plan.artifacts.image_reference,
                            "imagePullPolicy": "IfNotPresent",
                            "terminationMessagePath": (
                                M03R_TOP2000_TERMINATION_MESSAGE_PATH
                            ),
                            "terminationMessagePolicy": (
                                M03R_TOP2000_TERMINATION_MESSAGE_POLICY
                            ),
                            "command": [argv[0]],
                            "args": argv[1:],
                            "env": [
                                {"name": "NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
                                {
                                    "name": "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                                    "value": "1",
                                },
                                {"name": "PYTHONNOUSERSITE", "value": "1"},
                                {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                                {"name": "PYTHONHASHSEED", "value": "0"},
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "CUBLAS_WORKSPACE_CONFIG", "value": ":4096:8"},
                                {"name": "OMP_NUM_THREADS", "value": "8"},
                                {"name": "MKL_NUM_THREADS", "value": "8"},
                                {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
                                {
                                    "name": "TORCHINDUCTOR_CACHE_DIR",
                                    "value": "/tmp/torchinductor",
                                },
                                {"name": "TRITON_CACHE_DIR", "value": "/tmp/triton"},
                                {"name": "PYTHONPATH", "value": plan.source_pythonpath},
                            ],
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
                            "persistentVolumeClaim": {
                                "claimName": template.pvc_claim_name
                            },
                        },
                        {"name": "tmp", "emptyDir": {}},
                        {
                            "name": "dshm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
                        },
                    ],
                },
            },
        },
    }
    _assert_no_node_identity(manifest)
    pod_spec = cast(dict[str, Any], manifest["spec"])["template"]["spec"]
    return M03RV7RenderedQualificationPilotJob(
        manifest_json=_canonical_json(manifest),
        manifest_sha256=_sha256(manifest),
        pod_spec_sha256=_canonical_pod_spec_sha256(pod_spec),
        package_plan_sha256=plan.package_plan_sha256,
        live_evidence_receipt_sha256=live_evidence.receipt_sha256,
        completion_index=completion_index,
    )


def render_m03r_v7_top2000_suspended_qualification_batch_job(
    *,
    plan: M03RV7Top2000PackagePlan,
    live_evidence: M03RV7LiveAdmissionEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV7RenderedQualificationBatchJob:
    """Render all twelve qualifications in one capped Indexed Job."""

    single = render_m03r_v7_top2000_suspended_qualification_pilot_job(
        plan=plan,
        completion_index=0,
        live_evidence=live_evidence,
        template=template,
        now_utc=now_utc,
    )
    manifest = single.manifest
    spec = cast(dict[str, Any], manifest["spec"])
    spec["completions"] = 12
    spec["parallelism"] = live_evidence.allowed_parallelism
    metadata = cast(dict[str, Any], manifest["metadata"])
    annotations = cast(dict[str, Any], metadata["annotations"])
    for name in (
        "rl-quant/qualification-completion-index",
        "rl-quant/qualification-setting-index",
        "rl-quant/qualification-setting-id",
    ):
        annotations.pop(name, None)
    annotations["rl-quant/qualification-setting-coverage"] = "all-12"
    template_payload = cast(dict[str, Any], spec["template"])
    template_annotations = cast(
        dict[str, Any],
        cast(dict[str, Any], template_payload["metadata"])["annotations"],
    )
    for name in (
        "rl-quant/qualification-completion-index",
        "rl-quant/qualification-setting-index",
        "rl-quant/qualification-setting-id",
    ):
        template_annotations.pop(name, None)
    template_annotations["rl-quant/qualification-setting-coverage"] = "all-12"
    pod_spec = cast(dict[str, Any], template_payload["spec"])
    container = cast(list[dict[str, Any]], pod_spec["containers"])[0]
    args = cast(list[str], container["args"])
    marker = args.index("--completion-index")
    del args[marker : marker + 2]
    environment = cast(list[dict[str, Any]], container["env"])
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
                    )
                }
            },
        },
    )
    return M03RV7RenderedQualificationBatchJob(
        manifest_json=_canonical_json(manifest),
        manifest_sha256=_sha256(manifest),
        pod_spec_sha256=_canonical_pod_spec_sha256(pod_spec),
        package_plan_sha256=plan.package_plan_sha256,
        live_evidence_receipt_sha256=live_evidence.receipt_sha256,
        parallelism=live_evidence.allowed_parallelism,
    )


def render_m03r_v7_top2000_suspended_indexed_job(
    *,
    package: M03RV7Top2000QualifiedPackage,
    live_evidence: M03RV7LiveAdmissionEvidence,
    template: M03RV7KubernetesTemplateConfig,
    now_utc: datetime,
) -> M03RV7RenderedSuspendedJob:
    """Render only a suspended manifest after every local/live gate passes."""

    try:
        package.require_launch_ready()
    except M03RV7Top2000PackageError as exc:
        raise M03RV7Top2000KubernetesError(str(exc)) from exc
    live_evidence.require_fresh(now_utc=now_utc)
    worker = package.worker_receipt
    capacity = package.capacity_receipt
    if worker is None or capacity is None:  # pragma: no cover - require above narrows
        raise M03RV7Top2000KubernetesError("qualified worker/capacity receipts missing")
    if not package.plan.plan_artifact_path.startswith(
        template.package_mount_path.rstrip("/") + "/"
    ):
        raise M03RV7Top2000KubernetesError(
            "package plan artifact must be below the read-only package mount"
        )
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v7-dev",
        "app.kubernetes.io/managed-by": "receipt-gated-research",
        "k8s-user": "yding4",
        "rl-quant/run-id": template.run_id,
        "rl-quant/owner": "yding4",
        "runai/queue": template.runai_queue,
    }
    annotations = {
        "rl-quant/run-id": template.run_id,
        "rl-quant/package-plan-sha256": package.plan.package_plan_sha256,
        "rl-quant/source-archive-sha256": (
            package.plan.artifacts.source_archive_sha256
        ),
        "rl-quant/cache-artifact-sha256": (
            package.plan.artifacts.cache_artifact_sha256
        ),
        "rl-quant/image-digest-sha256": (
            package.plan.artifacts.image_digest_sha256
        ),
        "rl-quant/execution-surface-sha256": worker.execution_surface_sha256,
        "rl-quant/capacity-receipt-sha256": capacity.receipt_sha256,
        "rl-quant/live-evidence-receipt-sha256": live_evidence.receipt_sha256,
        "rl-quant/data-role": "development-only-nonreportable",
    }
    argv = list(worker.worker_argv_prefix) + [
        "--package-plan",
        package.plan.plan_artifact_path,
        "--package-plan-sha256",
        package.plan.package_plan_sha256,
        "--output-root",
        template.output_mount_path,
    ]
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
            "completions": M03R_TOP2000_INDEXED_COMPLETIONS,
            "parallelism": live_evidence.allowed_parallelism,
            "backoffLimit": 0,
            "activeDeadlineSeconds": template.active_deadline_seconds,
            "ttlSecondsAfterFinished": template.ttl_seconds_after_finished,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": {
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
                                                "key": (
                                                    live_evidence.gpu_product_label_key
                                                ),
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
                            "image": package.plan.artifacts.image_reference,
                            "imagePullPolicy": "IfNotPresent",
                            "terminationMessagePath": (
                                M03R_TOP2000_TERMINATION_MESSAGE_PATH
                            ),
                            "terminationMessagePolicy": (
                                M03R_TOP2000_TERMINATION_MESSAGE_POLICY
                            ),
                            "command": [argv[0]],
                            "args": argv[1:],
                            "env": [
                                {
                                    "name": "JOB_COMPLETION_INDEX",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "apiVersion": "v1",
                                            "fieldPath": (
                                                "metadata.annotations['batch.kubernetes.io/"
                                                "job-completion-index']"
                                            )
                                        }
                                    },
                                },
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
                                {
                                    "name": "PYTHONPATH",
                                    "value": package.plan.source_pythonpath,
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": template.cpu_request,
                                    "memory": template.memory_request,
                                    "ephemeral-storage": (
                                        template.ephemeral_storage_request
                                    ),
                                    "nvidia.com/gpu": "2",
                                },
                                "limits": {
                                    "cpu": template.cpu_limit,
                                    "memory": template.memory_limit,
                                    "ephemeral-storage": (
                                        template.ephemeral_storage_limit
                                    ),
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
                                    ),
                                },
                                {
                                    "name": "tmp",
                                    "mountPath": "/tmp",
                                },
                                {
                                    "name": "dshm",
                                    "mountPath": "/dev/shm",
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "research-data",
                            "persistentVolumeClaim": {
                                "claimName": template.pvc_claim_name
                            },
                        },
                        {"name": "tmp", "emptyDir": {}},
                        {
                            "name": "dshm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
                        },
                    ],
                },
            },
        },
    }
    _assert_no_node_identity(manifest)
    template_payload = cast(dict[str, Any], manifest["spec"])["template"]["spec"]
    return M03RV7RenderedSuspendedJob(
        manifest_json=_canonical_json(manifest),
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_canonical_pod_spec_sha256(template_payload),
        package_plan_sha256=package.plan.package_plan_sha256,
        execution_surface_sha256=worker.execution_surface_sha256,
        live_evidence_receipt_sha256=live_evidence.receipt_sha256,
        capacity_receipt_sha256=capacity.receipt_sha256,
        parallelism=live_evidence.allowed_parallelism,
    )


@dataclass(frozen=True, slots=True)
class M03RV7AdmittedJobBinding:
    """Two-read binding to the exact clean, suspended admitted Job UID."""

    job_name: str
    namespace: str
    job_uid: str
    run_id: str
    first_resource_version: str
    second_resource_version: str
    parallelism: int
    admitted_spec_sha256: str
    admitted_pod_template_sha256: str
    admitted_selector_sha256: str
    admitted_template_metadata_sha256: str
    desired_manifest_sha256: str
    attached_owned_pod_uids: tuple[str, ...]
    suspended: bool
    receipt_sha256: str
    schema: str = M03R_TOP2000_ADMITTED_BINDING_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "job_name": self.job_name,
            "namespace": self.namespace,
            "job_uid": self.job_uid,
            "run_id": self.run_id,
            "first_resource_version": self.first_resource_version,
            "second_resource_version": self.second_resource_version,
            "parallelism": self.parallelism,
            "admitted_spec_sha256": self.admitted_spec_sha256,
            "admitted_pod_template_sha256": self.admitted_pod_template_sha256,
            "admitted_selector_sha256": self.admitted_selector_sha256,
            "admitted_template_metadata_sha256": (
                self.admitted_template_metadata_sha256
            ),
            "desired_manifest_sha256": self.desired_manifest_sha256,
            "attached_owned_pod_uids": list(self.attached_owned_pod_uids),
            "suspended": self.suspended,
        }

    def __post_init__(self) -> None:
        if self.schema != M03R_TOP2000_ADMITTED_BINDING_SCHEMA:
            raise M03RV7Top2000KubernetesError("admitted binding schema drifted")
        _require_dns_label("job_name", self.job_name)
        if self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE:
            raise M03RV7Top2000KubernetesError("admitted namespace drifted")
        _require_dns_label("run_id", self.run_id)
        if (
            not self.job_uid
            or not self.first_resource_version
            or not self.second_resource_version
            or isinstance(self.parallelism, bool)
            or not isinstance(self.parallelism, int)
            or not 1 <= self.parallelism <= M03R_TOP2000_PARALLELISM_HARD_CAP
            or self.attached_owned_pod_uids
            or not self.suspended
        ):
            raise M03RV7Top2000KubernetesError(
                "clean attachment requires exact UID/RVs, suspension, and zero Pods"
            )
        for name in (
            "admitted_spec_sha256",
            "admitted_pod_template_sha256",
            "admitted_selector_sha256",
            "admitted_template_metadata_sha256",
            "desired_manifest_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000KubernetesError("admitted binding hash mismatch")


def _admitted_identity(job: Mapping[str, Any]) -> tuple[str, str, str, str]:
    metadata = _mapping(job.get("metadata"), "admitted.metadata")
    name = metadata.get("name")
    namespace = metadata.get("namespace")
    uid = metadata.get("uid")
    resource_version = metadata.get("resourceVersion")
    if not all(isinstance(value, str) and value for value in (name, namespace, uid, resource_version)):
        raise M03RV7Top2000KubernetesError(
            "admitted Job metadata must include name/namespace/UID/resourceVersion"
        )
    return cast(str, name), cast(str, namespace), cast(str, uid), cast(str, resource_version)


def _require_exact_admitted_metadata(
    *,
    desired_job_metadata: Mapping[str, Any],
    admitted_job_metadata: Mapping[str, Any],
    desired_template_metadata: Mapping[str, Any],
    admitted_template_metadata: Mapping[str, Any],
    job_name: str,
    job_uid: str,
) -> None:
    """Allow only the Job-controller labels and the known null timestamp."""

    desired_job_labels = dict(
        _mapping(desired_job_metadata.get("labels"), "desired Job labels")
    )
    desired_job_annotations = dict(
        _mapping(desired_job_metadata.get("annotations"), "desired Job annotations")
    )
    admitted_job_labels = dict(
        _mapping(admitted_job_metadata.get("labels"), "admitted Job labels")
    )
    admitted_job_annotations = dict(
        _mapping(admitted_job_metadata.get("annotations"), "admitted Job annotations")
    )
    if (
        admitted_job_labels != desired_job_labels
        or admitted_job_annotations != desired_job_annotations
    ):
        raise M03RV7Top2000KubernetesError(
            "admission changed or injected an unbound Job label or annotation"
        )

    desired_template_labels = dict(
        _mapping(
            desired_template_metadata.get("labels"),
            "desired Pod-template labels",
        )
    )
    dynamic_template_label_values = (job_uid, job_name, job_uid, job_name)
    expected_template_labels = {
        **desired_template_labels,
        **dict(
            zip(
                M03R_TOP2000_DYNAMIC_TEMPLATE_LABEL_KEYS,
                dynamic_template_label_values,
                strict=True,
            )
        ),
    }
    admitted_template_labels = dict(
        _mapping(
            admitted_template_metadata.get("labels"),
            "admitted Pod-template labels",
        )
    )
    desired_template_annotations = dict(
        _mapping(
            desired_template_metadata.get("annotations"),
            "desired Pod-template annotations",
        )
    )
    admitted_template_annotations = dict(
        _mapping(
            admitted_template_metadata.get("annotations"),
            "admitted Pod-template annotations",
        )
    )
    allowed_metadata_keys = {"labels", "annotations", "creationTimestamp"}
    if (
        admitted_template_labels != expected_template_labels
        or admitted_template_annotations != desired_template_annotations
        or set(admitted_template_metadata) != allowed_metadata_keys
        or admitted_template_metadata.get("creationTimestamp") is not None
    ):
        raise M03RV7Top2000KubernetesError(
            "admitted Pod-template metadata contains an unknown mutation"
        )


def bind_m03r_v7_top2000_admitted_suspended_job(
    *,
    rendered: (
        M03RV7RenderedSuspendedJob
        | M03RV7RenderedQualificationPilotJob
        | M03RV7RenderedQualificationBatchJob
    ),
    first_read: Mapping[str, Any],
    second_read: Mapping[str, Any],
    attached_owned_pod_uids: tuple[str, ...],
) -> M03RV7AdmittedJobBinding:
    """Bind two read-back observations; never create or unsuspend the Job."""

    first_identity = _admitted_identity(first_read)
    second_identity = _admitted_identity(second_read)
    if first_identity[:3] != second_identity[:3]:
        raise M03RV7Top2000KubernetesError(
            "Job identity changed between admitted-spec reads"
        )
    desired_metadata = _mapping(rendered.manifest.get("metadata"), "desired.metadata")
    job_name, namespace, job_uid, second_resource_version = second_identity
    if (
        job_name != desired_metadata.get("name")
        or namespace != desired_metadata.get("namespace")
        or attached_owned_pod_uids
    ):
        raise M03RV7Top2000KubernetesError(
            "admitted attach target drifted or already owns Pods"
        )
    first_spec = _mapping(first_read.get("spec"), "first.spec")
    second_spec = _mapping(second_read.get("spec"), "second.spec")
    _require_supported_indexed_job_spec(first_spec)
    _require_supported_indexed_job_spec(second_spec)
    if _sha256(first_spec) != _sha256(second_spec):
        raise M03RV7Top2000KubernetesError(
            "admitted Job spec changed between clean attachment reads"
        )
    second_template = _mapping(second_spec.get("template"), "second.spec.template")
    second_selector = _mapping(second_spec.get("selector"), "second.spec.selector")
    desired_spec = _mapping(rendered.manifest.get("spec"), "desired.spec")
    desired_template = _mapping(desired_spec.get("template"), "desired.spec.template")
    admitted_metadata = _mapping(second_read.get("metadata"), "admitted.metadata")
    admitted_template_metadata = _mapping(
        second_template.get("metadata"), "admitted template metadata"
    )
    desired_template_metadata = _mapping(
        desired_template.get("metadata"), "desired template metadata"
    )
    desired_pod_spec = _mapping(desired_template.get("spec"), "desired pod spec")
    admitted_pod_spec = _mapping(second_template.get("spec"), "admitted pod spec")
    _require_supported_indexed_job_spec(desired_spec)
    required_equal = tuple(
        (name, desired_spec.get(name))
        for name in (
            "completionMode",
            "completions",
            "parallelism",
            "backoffLimit",
            "activeDeadlineSeconds",
            "ttlSecondsAfterFinished",
            "suspend",
        )
    )
    if any(second_spec.get(name) != expected for name, expected in required_equal):
        raise M03RV7Top2000KubernetesError("admitted Indexed Job geometry drifted")
    allowed_spec_keys = set(desired_spec) | {"selector"}
    if set(second_spec) != allowed_spec_keys:
        raise M03RV7Top2000KubernetesError(
            "admitted Job spec contains an unknown injected field"
        )
    expected_selector = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": job_uid}
    }
    if dict(second_selector) != expected_selector:
        raise M03RV7Top2000KubernetesError(
            "admitted Job selector does not bind the exact controller UID"
        )
    if dict(admitted_pod_spec) != dict(desired_pod_spec):
        raise M03RV7Top2000KubernetesError(
            "admission changed an execution-bearing Pod field"
        )
    _require_proven_h100_pod_profile(
        admitted_pod_spec,
        service_account_name="default",
    )
    _require_exact_admitted_metadata(
        desired_job_metadata=desired_metadata,
        admitted_job_metadata=admitted_metadata,
        desired_template_metadata=desired_template_metadata,
        admitted_template_metadata=admitted_template_metadata,
        job_name=job_name,
        job_uid=job_uid,
    )
    desired_annotations = _mapping(
        desired_metadata.get("annotations"), "desired Job annotations"
    )
    run_id = desired_annotations.get(_RUN_ID_ANNOTATION)
    parallelism = second_spec.get("parallelism")
    if not isinstance(run_id, str) or not run_id:
        raise M03RV7Top2000KubernetesError(
            "desired Job must bind the rl-quant/run-id annotation"
        )
    if isinstance(parallelism, bool) or not isinstance(parallelism, int):
        raise M03RV7Top2000KubernetesError(
            "admitted Job parallelism must be an integer"
        )
    _assert_no_node_identity(second_spec)
    fields: dict[str, Any] = {
        "job_name": job_name,
        "namespace": namespace,
        "job_uid": job_uid,
        "run_id": run_id,
        "first_resource_version": first_identity[3],
        "second_resource_version": second_resource_version,
        "parallelism": parallelism,
        "admitted_spec_sha256": _sha256(second_spec),
        "admitted_pod_template_sha256": _canonical_pod_spec_sha256(
            admitted_pod_spec
        ),
        "admitted_selector_sha256": _sha256(second_selector),
        "admitted_template_metadata_sha256": _sha256(
            admitted_template_metadata
        ),
        "desired_manifest_sha256": rendered.manifest_sha256,
        "attached_owned_pod_uids": attached_owned_pod_uids,
        "suspended": True,
        "schema": M03R_TOP2000_ADMITTED_BINDING_SCHEMA,
    }
    unsigned = M03RV7AdmittedJobBinding.__new__(M03RV7AdmittedJobBinding)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7AdmittedJobBinding(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


def _json_pointer_token(value: str) -> str:
    """Escape one RFC 6901 JSON Pointer reference token."""

    return value.replace("~", "~0").replace("/", "~1")


def _validate_exact_bound_job_identity(
    *,
    binding: M03RV7AdmittedJobBinding,
    job_read: Mapping[str, Any],
    phase: str,
) -> tuple[Mapping[str, Any], str]:
    if job_read.get("apiVersion") != "batch/v1" or job_read.get("kind") != "Job":
        raise M03RV7Top2000KubernetesError(
            f"{phase} read must be one batch/v1 Job"
        )
    name, namespace, uid, resource_version = _admitted_identity(job_read)
    if (
        name != binding.job_name
        or namespace != binding.namespace
        or uid != binding.job_uid
    ):
        raise M03RV7Top2000KubernetesError(
            f"{phase} Job name/namespace/UID does not match admitted binding"
        )
    metadata = _mapping(job_read.get("metadata"), f"{phase}.metadata")
    annotations = _mapping(
        metadata.get("annotations"), f"{phase}.metadata.annotations"
    )
    if annotations.get(_RUN_ID_ANNOTATION) != binding.run_id:
        raise M03RV7Top2000KubernetesError(
            f"{phase} Job run-ID does not match admitted binding"
        )
    return metadata, resource_version


@dataclass(frozen=True, slots=True)
class M03RV7ExactJobActivationRequest:
    """Content-bound JSON Patch for one exact admitted suspended Job."""

    job_name: str
    namespace: str
    job_uid: str
    run_id: str
    resource_version: str
    parallelism: int
    binding_receipt_sha256: str
    admitted_selector_sha256: str
    admitted_template_metadata_sha256: str
    admitted_pod_template_sha256: str
    json_patch_json: str
    json_patch_sha256: str
    request_sha256: str
    content_type: str = M03R_TOP2000_JSON_PATCH_CONTENT_TYPE
    schema: str = M03R_TOP2000_ACTIVATION_REQUEST_SCHEMA

    @property
    def json_patch(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], json.loads(self.json_patch_json))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "job_name": self.job_name,
            "namespace": self.namespace,
            "job_uid": self.job_uid,
            "run_id": self.run_id,
            "resource_version": self.resource_version,
            "parallelism": self.parallelism,
            "binding_receipt_sha256": self.binding_receipt_sha256,
            "admitted_selector_sha256": self.admitted_selector_sha256,
            "admitted_template_metadata_sha256": (
                self.admitted_template_metadata_sha256
            ),
            "admitted_pod_template_sha256": self.admitted_pod_template_sha256,
            "content_type": self.content_type,
            "json_patch": self.json_patch,
            "json_patch_sha256": self.json_patch_sha256,
        }

    def __post_init__(self) -> None:
        _require_dns_label("activation job_name", self.job_name)
        _require_dns_label("activation run_id", self.run_id)
        if (
            self.schema != M03R_TOP2000_ACTIVATION_REQUEST_SCHEMA
            or self.content_type != M03R_TOP2000_JSON_PATCH_CONTENT_TYPE
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or not self.job_uid
            or not self.resource_version
            or isinstance(self.parallelism, bool)
            or not isinstance(self.parallelism, int)
            or not 1 <= self.parallelism <= M03R_TOP2000_PARALLELISM_HARD_CAP
        ):
            raise M03RV7Top2000KubernetesError(
                "activation request identity, media type, or parallelism is invalid"
            )
        for name in (
            "binding_receipt_sha256",
            "admitted_selector_sha256",
            "admitted_template_metadata_sha256",
            "admitted_pod_template_sha256",
            "json_patch_sha256",
            "request_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        try:
            patch = json.loads(self.json_patch_json)
        except json.JSONDecodeError as exc:
            raise M03RV7Top2000KubernetesError(
                "activation JSON Patch is invalid JSON"
            ) from exc
        if (
            not isinstance(patch, list)
            or len(patch) != 9
            or any(not isinstance(operation, dict) for operation in patch)
            or self.json_patch_json != _canonical_json(patch)
        ):
            raise M03RV7Top2000KubernetesError(
                "activation JSON Patch must contain nine canonical operations"
            )
        run_id_path = (
            "/metadata/annotations/" + _json_pointer_token(_RUN_ID_ANNOTATION)
        )
        expected_prefix = (
            ("/metadata/uid", self.job_uid),
            ("/metadata/resourceVersion", self.resource_version),
            (run_id_path, self.run_id),
            ("/spec/suspend", True),
            ("/spec/parallelism", self.parallelism),
        )
        if any(
            patch[index]
            != {"op": "test", "path": path, "value": expected_value}
            for index, (path, expected_value) in enumerate(expected_prefix)
        ):
            raise M03RV7Top2000KubernetesError(
                "activation identity or scalar test operation drifted"
            )
        structured_tests = (
            (5, "/spec/selector", self.admitted_selector_sha256, _sha256),
            (
                6,
                "/spec/template/metadata",
                self.admitted_template_metadata_sha256,
                _sha256,
            ),
            (
                7,
                "/spec/template/spec",
                self.admitted_pod_template_sha256,
                _canonical_pod_spec_sha256,
            ),
        )
        for index, path, expected_sha256, hasher in structured_tests:
            operation = patch[index]
            if (
                set(operation) != {"op", "path", "value"}
                or operation.get("op") != "test"
                or operation.get("path") != path
            ):
                raise M03RV7Top2000KubernetesError(
                    "activation structured test operation drifted"
                )
            value = _mapping(operation.get("value"), f"activation patch {path}")
            if hasher(value) != expected_sha256:
                raise M03RV7Top2000KubernetesError(
                    f"activation patch value for {path} is not admitted-bound"
                )
        if patch[8] != {"op": "replace", "path": "/spec/suspend", "value": False}:
            raise M03RV7Top2000KubernetesError(
                "activation patch sole mutation must unsuspend the exact Job"
            )
        if self.json_patch_sha256 != _sha256(patch):
            raise M03RV7Top2000KubernetesError("activation JSON Patch hash mismatch")
        if self.request_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000KubernetesError("activation request hash mismatch")


def build_m03r_v7_exact_job_activation_request(
    binding: M03RV7AdmittedJobBinding,
    fresh_job_read: Mapping[str, Any],
) -> M03RV7ExactJobActivationRequest:
    """Validate a fresh suspended read and bind its one safe activation patch."""

    _, resource_version = _validate_exact_bound_job_identity(
        binding=binding,
        job_read=fresh_job_read,
        phase="activation",
    )
    if (
        binding.first_resource_version != binding.second_resource_version
        and resource_version == binding.first_resource_version
    ):
        raise M03RV7Top2000KubernetesError(
            "activation Job read uses a known-stale resourceVersion"
        )
    spec = _mapping(fresh_job_read.get("spec"), "activation.spec")
    if _sha256(spec) != binding.admitted_spec_sha256:
        raise M03RV7Top2000KubernetesError(
            "activation Job spec does not match admitted binding"
        )
    template = _mapping(spec.get("template"), "activation.spec.template")
    selector = _mapping(spec.get("selector"), "activation.spec.selector")
    template_metadata = _mapping(
        template.get("metadata"), "activation.spec.template.metadata"
    )
    pod_spec = _mapping(template.get("spec"), "activation.spec.template.spec")
    if (
        spec.get("suspend") is not True
        or spec.get("parallelism") != binding.parallelism
        or _sha256(selector) != binding.admitted_selector_sha256
        or _sha256(template_metadata)
        != binding.admitted_template_metadata_sha256
        or _canonical_pod_spec_sha256(pod_spec)
        != binding.admitted_pod_template_sha256
    ):
        raise M03RV7Top2000KubernetesError(
            "activation read is not the exact admitted suspended Job"
        )
    _assert_no_node_identity(spec)
    patch: list[dict[str, Any]] = [
        {"op": "test", "path": "/metadata/uid", "value": binding.job_uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": resource_version,
        },
        {
            "op": "test",
            "path": (
                "/metadata/annotations/"
                + _json_pointer_token(_RUN_ID_ANNOTATION)
            ),
            "value": binding.run_id,
        },
        {"op": "test", "path": "/spec/suspend", "value": True},
        {
            "op": "test",
            "path": "/spec/parallelism",
            "value": binding.parallelism,
        },
        {"op": "test", "path": "/spec/selector", "value": dict(selector)},
        {
            "op": "test",
            "path": "/spec/template/metadata",
            "value": dict(template_metadata),
        },
        {
            "op": "test",
            "path": "/spec/template/spec",
            "value": dict(pod_spec),
        },
        {"op": "replace", "path": "/spec/suspend", "value": False},
    ]
    fields: dict[str, Any] = {
        "job_name": binding.job_name,
        "namespace": binding.namespace,
        "job_uid": binding.job_uid,
        "run_id": binding.run_id,
        "resource_version": resource_version,
        "parallelism": binding.parallelism,
        "binding_receipt_sha256": binding.receipt_sha256,
        "admitted_selector_sha256": binding.admitted_selector_sha256,
        "admitted_template_metadata_sha256": (
            binding.admitted_template_metadata_sha256
        ),
        "admitted_pod_template_sha256": binding.admitted_pod_template_sha256,
        "json_patch_json": _canonical_json(patch),
        "json_patch_sha256": _sha256(patch),
        "content_type": M03R_TOP2000_JSON_PATCH_CONTENT_TYPE,
        "schema": M03R_TOP2000_ACTIVATION_REQUEST_SCHEMA,
    }
    unsigned = M03RV7ExactJobActivationRequest.__new__(
        M03RV7ExactJobActivationRequest
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7ExactJobActivationRequest(
        **fields,
        request_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class M03RV7ExactJobCleanupRequest:
    """DeleteOptions only; a caller must apply it to this exact Job name."""

    job_name: str
    namespace: str
    job_uid: str
    run_id: str
    resource_version: str
    binding_receipt_sha256: str
    delete_options_sha256: str

    @property
    def delete_options(self) -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Foreground",
            "preconditions": {
                "uid": self.job_uid,
                "resourceVersion": self.resource_version,
            },
        }

    def __post_init__(self) -> None:
        _require_dns_label("job_name", self.job_name)
        _require_dns_label("run_id", self.run_id)
        if (
            self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or not self.job_uid
            or not self.resource_version
        ):
            raise M03RV7Top2000KubernetesError(
                "cleanup must bind exact namespace, UID, and resourceVersion"
            )
        _require_sha256("binding_receipt_sha256", self.binding_receipt_sha256)
        _require_sha256("delete_options_sha256", self.delete_options_sha256)
        if self.delete_options_sha256 != _sha256(self.delete_options):
            raise M03RV7Top2000KubernetesError("DeleteOptions hash mismatch")


def build_m03r_v7_exact_job_cleanup_request(
    binding: M03RV7AdmittedJobBinding,
    fresh_job_read: Mapping[str, Any],
) -> M03RV7ExactJobCleanupRequest:
    """Bind fresh post-run identity into foreground DeleteOptions."""

    _, resource_version = _validate_exact_bound_job_identity(
        binding=binding,
        job_read=fresh_job_read,
        phase="cleanup",
    )
    if resource_version in {
        binding.first_resource_version,
        binding.second_resource_version,
    }:
        raise M03RV7Top2000KubernetesError(
            "cleanup Job read reuses a stale pre-run resourceVersion"
        )

    fields = {
        "job_name": binding.job_name,
        "namespace": binding.namespace,
        "job_uid": binding.job_uid,
        "run_id": binding.run_id,
        "resource_version": resource_version,
        "binding_receipt_sha256": binding.receipt_sha256,
    }
    unsigned = M03RV7ExactJobCleanupRequest.__new__(M03RV7ExactJobCleanupRequest)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7ExactJobCleanupRequest(
        **fields,
        delete_options_sha256=_sha256(unsigned.delete_options),
    )


@dataclass(frozen=True, slots=True)
class M03RV7ExactCleanupReceipt:
    """Proof that both the exact Job and all UID-owned Pods are absent."""

    request: M03RV7ExactJobCleanupRequest
    first_job_absent: bool
    second_job_absent: bool
    first_owned_pod_uids: tuple[str, ...]
    second_owned_pod_uids: tuple[str, ...]
    verification_evidence_sha256: str
    receipt_sha256: str

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "request": asdict(self.request),
            "first_job_absent": self.first_job_absent,
            "second_job_absent": self.second_job_absent,
            "first_owned_pod_uids": list(self.first_owned_pod_uids),
            "second_owned_pod_uids": list(self.second_owned_pod_uids),
            "verification_evidence_sha256": self.verification_evidence_sha256,
        }

    def __post_init__(self) -> None:
        _require_sha256(
            "verification_evidence_sha256", self.verification_evidence_sha256
        )
        _require_sha256("cleanup receipt_sha256", self.receipt_sha256)
        if (
            not self.first_job_absent
            or not self.second_job_absent
            or self.first_owned_pod_uids
            or self.second_owned_pod_uids
        ):
            raise M03RV7Top2000KubernetesError(
                "cleanup is incomplete while the exact Job or UID-owned Pods remain"
            )
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000KubernetesError("cleanup receipt hash mismatch")


def build_m03r_v7_exact_cleanup_receipt(
    *,
    request: M03RV7ExactJobCleanupRequest,
    first_job_absent: bool,
    second_job_absent: bool,
    first_owned_pod_uids: tuple[str, ...],
    second_owned_pod_uids: tuple[str, ...],
    verification_evidence_sha256: str,
) -> M03RV7ExactCleanupReceipt:
    """Bind two post-delete absence reads without performing deletion."""

    fields: dict[str, Any] = {
        "request": request,
        "first_job_absent": first_job_absent,
        "second_job_absent": second_job_absent,
        "first_owned_pod_uids": first_owned_pod_uids,
        "second_owned_pod_uids": second_owned_pod_uids,
        "verification_evidence_sha256": verification_evidence_sha256,
    }
    unsigned = M03RV7ExactCleanupReceipt.__new__(M03RV7ExactCleanupReceipt)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7ExactCleanupReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class M03RV7FoldSeedReceiptRef:
    fold_index: int
    seed: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not 0 <= self.fold_index < 6:
            raise M03RV7Top2000KubernetesError("fold receipt index out of range")
        _require_sha256("fold/seed receipt_sha256", self.receipt_sha256)


@dataclass(frozen=True, slots=True)
class M03RV7IndexCompletionReceipt:
    """Receipt for one Indexed completion and all its thirty paired cells."""

    completion_index: int
    setting_index: int
    development_setting_id: str
    reviewed_v7_setting_id: str
    job_uid: str
    admitted_binding_sha256: str
    package_plan_sha256: str
    source_archive_sha256: str
    cache_artifact_sha256: str
    image_digest_sha256: str
    execution_surface_sha256: str
    capacity_receipt_sha256: str
    fold_seed_receipts: tuple[M03RV7FoldSeedReceiptRef, ...]
    output_manifest_sha256: str
    process_exit_code: int
    completion_succeeded: bool
    development_only: bool
    promotion_eligible: bool
    receipt_sha256: str
    schema: str = M03R_TOP2000_INDEX_RECEIPT_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "completion_index": self.completion_index,
            "setting_index": self.setting_index,
            "development_setting_id": self.development_setting_id,
            "reviewed_v7_setting_id": self.reviewed_v7_setting_id,
            "job_uid": self.job_uid,
            "admitted_binding_sha256": self.admitted_binding_sha256,
            "package_plan_sha256": self.package_plan_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "cache_artifact_sha256": self.cache_artifact_sha256,
            "image_digest_sha256": self.image_digest_sha256,
            "execution_surface_sha256": self.execution_surface_sha256,
            "capacity_receipt_sha256": self.capacity_receipt_sha256,
            "fold_seed_receipts": [
                asdict(value) for value in self.fold_seed_receipts
            ],
            "output_manifest_sha256": self.output_manifest_sha256,
            "process_exit_code": self.process_exit_code,
            "completion_succeeded": self.completion_succeeded,
            "development_only": self.development_only,
            "promotion_eligible": self.promotion_eligible,
        }

    def __post_init__(self) -> None:
        if self.schema != M03R_TOP2000_INDEX_RECEIPT_SCHEMA:
            raise M03RV7Top2000KubernetesError("index receipt schema drifted")
        if (
            not 0 <= self.completion_index < 12
            or not self.job_uid
            or self.process_exit_code != 0
            or not self.completion_succeeded
            or not self.development_only
            or self.promotion_eligible
        ):
            raise M03RV7Top2000KubernetesError(
                "index receipt must be a successful nonpromotable development result"
            )
        for name in (
            "admitted_binding_sha256",
            "package_plan_sha256",
            "source_archive_sha256",
            "cache_artifact_sha256",
            "image_digest_sha256",
            "execution_surface_sha256",
            "capacity_receipt_sha256",
            "output_manifest_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if len(self.fold_seed_receipts) != 30:
            raise M03RV7Top2000KubernetesError(
                "each index receipt must contain all thirty fold/seed receipts"
            )
        coordinates = tuple(
            (receipt.fold_index, receipt.seed) for receipt in self.fold_seed_receipts
        )
        if len(set(coordinates)) != 30:
            raise M03RV7Top2000KubernetesError(
                "fold/seed receipt coordinates must be unique"
            )
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000KubernetesError("index completion receipt hash mismatch")

    def verify_for(
        self,
        *,
        index_plan: M03RV7Top2000IndexPlan,
        package_plan: M03RV7Top2000PackagePlan,
        binding: M03RV7AdmittedJobBinding,
        execution_surface_sha256: str,
        capacity_receipt_sha256: str,
    ) -> None:
        expected_coordinates = tuple(
            (fold, seed)
            for fold in index_plan.fold_indices
            for seed in index_plan.paired_seeds
        )
        observed_coordinates = tuple(
            (receipt.fold_index, receipt.seed) for receipt in self.fold_seed_receipts
        )
        if (
            self.completion_index != index_plan.completion_index
            or self.setting_index != index_plan.setting_index
            or self.development_setting_id != index_plan.development_setting_id
            or self.reviewed_v7_setting_id != index_plan.reviewed_v7_setting_id
            or observed_coordinates != expected_coordinates
            or self.job_uid != binding.job_uid
            or self.admitted_binding_sha256 != binding.receipt_sha256
            or self.package_plan_sha256 != package_plan.package_plan_sha256
            or self.source_archive_sha256
            != package_plan.artifacts.source_archive_sha256
            or self.cache_artifact_sha256
            != package_plan.artifacts.cache_artifact_sha256
            or self.image_digest_sha256
            != package_plan.artifacts.image_digest_sha256
            or self.execution_surface_sha256 != execution_surface_sha256
            or self.capacity_receipt_sha256 != capacity_receipt_sha256
        ):
            raise M03RV7Top2000KubernetesError(
                "index receipt does not bind the exact admitted package completion"
            )


def build_m03r_v7_index_completion_receipt(
    *,
    package: M03RV7Top2000QualifiedPackage,
    binding: M03RV7AdmittedJobBinding,
    completion_index: int,
    fold_seed_receipts: tuple[M03RV7FoldSeedReceiptRef, ...],
    output_manifest_sha256: str,
) -> M03RV7IndexCompletionReceipt:
    """Bind one successful completion to its exact setting and thirty cells."""

    package.require_launch_ready()
    try:
        index_plan = package.plan.indices[completion_index]
    except IndexError as exc:
        raise M03RV7Top2000KubernetesError(
            "completion_index must be in [0, 11]"
        ) from exc
    worker = package.worker_receipt
    capacity = package.capacity_receipt
    if worker is None or capacity is None:  # pragma: no cover
        raise M03RV7Top2000KubernetesError("qualified receipts disappeared")
    fields: dict[str, Any] = {
        "completion_index": completion_index,
        "setting_index": index_plan.setting_index,
        "development_setting_id": index_plan.development_setting_id,
        "reviewed_v7_setting_id": index_plan.reviewed_v7_setting_id,
        "job_uid": binding.job_uid,
        "admitted_binding_sha256": binding.receipt_sha256,
        "package_plan_sha256": package.plan.package_plan_sha256,
        "source_archive_sha256": package.plan.artifacts.source_archive_sha256,
        "cache_artifact_sha256": package.plan.artifacts.cache_artifact_sha256,
        "image_digest_sha256": package.plan.artifacts.image_digest_sha256,
        "execution_surface_sha256": worker.execution_surface_sha256,
        "capacity_receipt_sha256": capacity.receipt_sha256,
        "fold_seed_receipts": fold_seed_receipts,
        "output_manifest_sha256": output_manifest_sha256,
        "process_exit_code": 0,
        "completion_succeeded": True,
        "development_only": True,
        "promotion_eligible": False,
        "schema": M03R_TOP2000_INDEX_RECEIPT_SCHEMA,
    }
    unsigned = M03RV7IndexCompletionReceipt.__new__(
        M03RV7IndexCompletionReceipt
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    receipt = M03RV7IndexCompletionReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )
    receipt.verify_for(
        index_plan=index_plan,
        package_plan=package.plan,
        binding=binding,
        execution_surface_sha256=worker.execution_surface_sha256,
        capacity_receipt_sha256=capacity.receipt_sha256,
    )
    return receipt


@dataclass(frozen=True, slots=True)
class M03RV7IndexedBatchReceipt:
    """Exact all-twelve receipt coverage for one admitted Job UID."""

    index_receipts: tuple[M03RV7IndexCompletionReceipt, ...]
    job_uid: str
    admitted_binding_sha256: str
    package_plan_sha256: str
    all_twelve_complete: bool
    development_only: bool
    promotion_eligible: bool
    receipt_sha256: str

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "index_receipts": [asdict(value) for value in self.index_receipts],
            "job_uid": self.job_uid,
            "admitted_binding_sha256": self.admitted_binding_sha256,
            "package_plan_sha256": self.package_plan_sha256,
            "all_twelve_complete": self.all_twelve_complete,
            "development_only": self.development_only,
            "promotion_eligible": self.promotion_eligible,
        }

    def __post_init__(self) -> None:
        if (
            len(self.index_receipts) != 12
            or tuple(value.completion_index for value in self.index_receipts)
            != tuple(range(12))
            or any(value.job_uid != self.job_uid for value in self.index_receipts)
            or not self.all_twelve_complete
            or not self.development_only
            or self.promotion_eligible
        ):
            raise M03RV7Top2000KubernetesError(
                "batch receipt requires exact ordered coverage of twelve completions"
            )
        for name in (
            "admitted_binding_sha256",
            "package_plan_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7Top2000KubernetesError("indexed batch receipt hash mismatch")


def build_m03r_v7_indexed_batch_receipt(
    *,
    package: M03RV7Top2000QualifiedPackage,
    binding: M03RV7AdmittedJobBinding,
    index_receipts: tuple[M03RV7IndexCompletionReceipt, ...],
) -> M03RV7IndexedBatchReceipt:
    """Validate and bind all twelve per-index receipts."""

    package.require_launch_ready()
    worker = package.worker_receipt
    capacity = package.capacity_receipt
    if worker is None or capacity is None:  # pragma: no cover
        raise M03RV7Top2000KubernetesError("qualified receipts disappeared")
    ordered = tuple(sorted(index_receipts, key=lambda value: value.completion_index))
    if len(ordered) != 12:
        raise M03RV7Top2000KubernetesError("all twelve index receipts are required")
    for index_plan, receipt in zip(package.plan.indices, ordered, strict=True):
        receipt.verify_for(
            index_plan=index_plan,
            package_plan=package.plan,
            binding=binding,
            execution_surface_sha256=worker.execution_surface_sha256,
            capacity_receipt_sha256=capacity.receipt_sha256,
        )
    fields: dict[str, Any] = {
        "index_receipts": ordered,
        "job_uid": binding.job_uid,
        "admitted_binding_sha256": binding.receipt_sha256,
        "package_plan_sha256": package.plan.package_plan_sha256,
        "all_twelve_complete": True,
        "development_only": True,
        "promotion_eligible": False,
    }
    unsigned = M03RV7IndexedBatchReceipt.__new__(M03RV7IndexedBatchReceipt)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7IndexedBatchReceipt(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


__all__ = [
    "M03R_TOP2000_ACTIVATION_REQUEST_SCHEMA",
    "M03R_TOP2000_GPUS_PER_COMPLETION",
    "M03R_TOP2000_H100_POOL_NODE_SELECTOR",
    "M03R_TOP2000_H100_PRODUCT_LABEL_KEY",
    "M03R_TOP2000_H100_PRODUCT_LABEL_VALUES",
    "M03R_TOP2000_INDEXED_COMPLETIONS",
    "M03R_TOP2000_JSON_PATCH_CONTENT_TYPE",
    "M03R_TOP2000_KUBERNETES_CONTEXT",
    "M03R_TOP2000_KUBERNETES_NAMESPACE",
    "M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS",
    "M03R_TOP2000_MULTI_GPU_TOLERATION",
    "M03R_TOP2000_PARALLELISM_HARD_CAP",
    "M03R_TOP2000_PRIORITY_CLASS_NAME",
    "M03RV7AdmittedJobBinding",
    "M03RV7ExactCleanupReceipt",
    "M03RV7ExactJobActivationRequest",
    "M03RV7ExactJobCleanupRequest",
    "M03RV7FoldSeedReceiptRef",
    "M03RV7IndexCompletionReceipt",
    "M03RV7IndexedBatchReceipt",
    "M03RV7KubernetesRBACEvidence",
    "M03RV7KubernetesTemplateConfig",
    "M03RV7LiveAdmissionEvidence",
    "M03RV7RenderedQualificationBatchJob",
    "M03RV7RenderedQualificationPilotJob",
    "M03RV7RenderedSuspendedJob",
    "M03RV7Top2000KubernetesError",
    "bind_m03r_v7_top2000_admitted_suspended_job",
    "build_m03r_v7_exact_cleanup_receipt",
    "build_m03r_v7_exact_job_activation_request",
    "build_m03r_v7_exact_job_cleanup_request",
    "build_m03r_v7_index_completion_receipt",
    "build_m03r_v7_indexed_batch_receipt",
    "build_m03r_v7_live_admission_evidence",
    "render_m03r_v7_top2000_suspended_indexed_job",
    "render_m03r_v7_top2000_suspended_qualification_batch_job",
    "render_m03r_v7_top2000_suspended_qualification_pilot_job",
]
