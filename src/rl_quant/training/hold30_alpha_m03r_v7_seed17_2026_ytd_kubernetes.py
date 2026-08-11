"""Pure Kubernetes rendering for the seed-17 2026-YTD retrospective.

This module renders suspended Jobs and validates their complete execution
surface.  It never invokes Kubernetes.  Creation, admitted-object binding,
activation, observation, and cleanup remain separate receipt-gated concerns.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping, cast

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_POOL_NODE_SELECTOR,
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_KUBERNETES_CONTEXT,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03R_TOP2000_MULTI_GPU_TOLERATION,
    M03R_TOP2000_PRIORITY_CLASS_NAME,
    M03R_TOP2000_TERMINATION_MESSAGE_PATH,
    M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_2026_ytd_package import (
    TOP2000_M03R_V7_2026_AGGREGATION_OUTPUT_MOUNT,
    TOP2000_M03R_V7_2026_EVALUATION_OUTPUT_MOUNT,
    TOP2000_M03R_V7_2026_H100_COMPUTE_CAPABILITY,
    TOP2000_M03R_V7_2026_H100_GPU_NAME,
    TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES,
    TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES,
    TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL,
    TOP2000_M03R_V7_2026_IMAGE_PYTHON,
    TOP2000_M03R_V7_2026_PREACCESS_MOUNT,
    TOP2000_M03R_V7_2026_Q2_OUTPUT_MOUNT,
    TOP2000_M03R_V7_2026_RAW_DATA_MOUNT,
    TOP2000_M03R_V7_2026_SOURCE_PACKAGE_MOUNT,
    Top2000M03RV72026ExecutionPackagePlan,
    Top2000M03RV72026PanelArtifactBindings,
    Top2000M03RV72026SentinelQualification,
    Top2000M03RV72026SourcePackagePlan,
)

TOP2000_M03R_V7_2026_USER_H100_CAP = 16
TOP2000_M03R_V7_2026_REMAINING_COMPLETIONS = 11
TOP2000_M03R_V7_2026_RENDERED_JOB_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-rendered-job-v1"
)
TOP2000_M03R_V7_2026_LIVE_CAPACITY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-live-capacity-v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DNS_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")


class Top2000M03RV72026KubernetesError(ValueError):
    """A template, capacity receipt, or rendered Job drifted."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026KubernetesError(
            "Kubernetes payload is not canonical finite JSON"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise Top2000M03RV72026KubernetesError(
            f"{name} must be one lowercase SHA-256"
        )


def _require_dns(name: str, value: str) -> None:
    if len(value) > 63 or _DNS_RE.fullmatch(value) is None:
        raise Top2000M03RV72026KubernetesError(
            f"{name} must be one Kubernetes DNS label"
        )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Top2000M03RV72026KubernetesError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _canonical_pod_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256((_canonical_json(dict(value)) + "\n").encode("ascii")).hexdigest()


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise Top2000M03RV72026KubernetesError(
            "capacity observed_at_utc must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise Top2000M03RV72026KubernetesError(
            "capacity observed_at_utc must be timezone-aware"
        )
    return parsed.astimezone(UTC)


def _contains_host_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) == "hostPath" or _contains_host_path(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_host_path(child) for child in value)
    return False


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026PersistentVolumeBinding:
    claim_name: str
    subpath: str

    def __post_init__(self) -> None:
        _require_dns("PVC claim_name", self.claim_name)
        path = PurePosixPath(self.subpath)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise Top2000M03RV72026KubernetesError(
                "PVC subpath must be one scoped relative path"
            )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026JobResources:
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    ephemeral_storage_request: str
    ephemeral_storage_limit: str
    active_deadline_seconds: int
    ttl_seconds_after_finished: int = 86400
    dshm_size_limit: str = "32Gi"

    def __post_init__(self) -> None:
        strings = (
            self.cpu_request,
            self.cpu_limit,
            self.memory_request,
            self.memory_limit,
            self.ephemeral_storage_request,
            self.ephemeral_storage_limit,
            self.dshm_size_limit,
        )
        if (
            any(not value or any(character.isspace() for character in value) for value in strings)
            or isinstance(self.active_deadline_seconds, bool)
            or not 1 <= self.active_deadline_seconds <= 216000
            or isinstance(self.ttl_seconds_after_finished, bool)
            or self.ttl_seconds_after_finished <= 0
        ):
            raise Top2000M03RV72026KubernetesError(
                "resource quantities, deadline, or TTL are invalid"
            )


def _overlaps(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026KubernetesTemplate:
    job_name: str
    run_id: str
    source_package: Top2000M03RV72026PersistentVolumeBinding
    q2_package: Top2000M03RV72026PersistentVolumeBinding
    q2_output: Top2000M03RV72026PersistentVolumeBinding
    raw_data: Top2000M03RV72026PersistentVolumeBinding
    preaccess_output: Top2000M03RV72026PersistentVolumeBinding
    evaluation_output: Top2000M03RV72026PersistentVolumeBinding
    aggregation_output: Top2000M03RV72026PersistentVolumeBinding
    resources: Top2000M03RV72026JobResources
    service_account_name: str = "default"
    scheduler_name: str = "kai-scheduler"
    runai_queue: str = "yding4-yn-gpu-workload-queue"
    run_as_user: int = 307469
    run_as_group: int = 600815

    def __post_init__(self) -> None:
        for name in (
            "job_name",
            "run_id",
            "service_account_name",
            "scheduler_name",
            "runai_queue",
        ):
            _require_dns(name, getattr(self, name))
        if self.service_account_name != "default":
            raise Top2000M03RV72026KubernetesError(
                "the approved research profile uses service account default"
            )
        if self.run_as_user <= 0 or self.run_as_group <= 0:
            raise Top2000M03RV72026KubernetesError(
                "Pod UID and GID must be positive"
            )
        mounts = (
            self.source_package,
            self.q2_package,
            self.q2_output,
            self.raw_data,
            self.preaccess_output,
            self.evaluation_output,
            self.aggregation_output,
        )
        for index, left in enumerate(mounts):
            for right in mounts[index + 1 :]:
                if left.claim_name == right.claim_name and _overlaps(
                    PurePosixPath(left.subpath), PurePosixPath(right.subpath)
                ):
                    raise Top2000M03RV72026KubernetesError(
                        "PVC inputs and phase outputs must use disjoint subpaths"
                    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026LiveCapacityEvidence:
    observed_at_utc: str
    protected_or_other_committed_h100_count: int
    receipt_sha256: str
    user_h100_cap: int = TOP2000_M03R_V7_2026_USER_H100_CAP
    remaining_setting_count: int = TOP2000_M03R_V7_2026_REMAINING_COMPLETIONS
    gpu_count_per_worker: int = 1
    gpu_product_label_key: str = M03R_TOP2000_H100_PRODUCT_LABEL_KEY
    gpu_product_label_value: str = TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    schema: str = TOP2000_M03R_V7_2026_LIVE_CAPACITY_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def __post_init__(self) -> None:
        _parse_utc(self.observed_at_utc)
        if (
            self.schema != TOP2000_M03R_V7_2026_LIVE_CAPACITY_SCHEMA
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.user_h100_cap != 16
            or self.remaining_setting_count != 11
            or self.gpu_count_per_worker != 1
            or self.gpu_product_label_key != M03R_TOP2000_H100_PRODUCT_LABEL_KEY
            or self.gpu_product_label_value
            != TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL
            or isinstance(self.protected_or_other_committed_h100_count, bool)
            or not isinstance(self.protected_or_other_committed_h100_count, int)
            or not 0 <= self.protected_or_other_committed_h100_count <= 16
        ):
            raise Top2000M03RV72026KubernetesError(
                "live capacity identity or H100 accounting drifted"
            )
        _require_sha256("capacity receipt_sha256", self.receipt_sha256)
        if self.receipt_sha256 != _sha256(self.canonical_payload()):
            raise Top2000M03RV72026KubernetesError(
                "live capacity receipt hash mismatch"
            )

    @property
    def cap_remaining(self) -> int:
        return self.user_h100_cap - self.protected_or_other_committed_h100_count

    @property
    def remaining_parallelism(self) -> int:
        return min(self.remaining_setting_count, self.cap_remaining)

    def require_fresh(self, *, now_utc: datetime, max_age_seconds: int = 300) -> None:
        if now_utc.tzinfo is None:
            raise Top2000M03RV72026KubernetesError(
                "now_utc must be timezone-aware"
            )
        age = (now_utc.astimezone(UTC) - _parse_utc(self.observed_at_utc)).total_seconds()
        if age < 0 or age > max_age_seconds:
            raise Top2000M03RV72026KubernetesError(
                "live capacity evidence is stale or from the future"
            )


def build_top2000_m03r_v7_seed17_2026_live_capacity_evidence(
    *,
    observed_at_utc: str,
    protected_or_other_committed_h100_count: int,
) -> Top2000M03RV72026LiveCapacityEvidence:
    fields: dict[str, Any] = {
        "observed_at_utc": observed_at_utc,
        "protected_or_other_committed_h100_count": (
            protected_or_other_committed_h100_count
        ),
        "user_h100_cap": 16,
        "remaining_setting_count": 11,
        "gpu_count_per_worker": 1,
        "gpu_product_label_key": M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
        "gpu_product_label_value": TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL,
        "context": M03R_TOP2000_KUBERNETES_CONTEXT,
        "namespace": M03R_TOP2000_KUBERNETES_NAMESPACE,
        "schema": TOP2000_M03R_V7_2026_LIVE_CAPACITY_SCHEMA,
    }
    unsigned = Top2000M03RV72026LiveCapacityEvidence.__new__(
        Top2000M03RV72026LiveCapacityEvidence
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    object.__setattr__(unsigned, "receipt_sha256", "0" * 64)
    return Top2000M03RV72026LiveCapacityEvidence(
        **fields,
        receipt_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026RenderedJob:
    phase: Literal["preaccess", "sentinel", "remaining", "aggregation"]
    manifest_json: str
    manifest_sha256: str
    pod_template_sha256: str
    source_package_sha256: str
    source_archive_sha256: str
    image_digest_sha256: str
    q2_training_output_inventory_sha256: str
    execution_package_sha256: str | None
    preaccess_completion_receipt_sha256: str | None
    live_capacity_receipt_sha256: str | None
    sentinel_qualification_receipt_sha256: str | None
    panel_artifact_receipt_sha256: str | None
    setting_map_sha256: str
    expected_output_inventory_sha256: str
    local_to_global_setting_indices: tuple[int, ...]
    completions: int
    parallelism: int
    gpu_count_per_pod: int
    activation_authorized: bool = False
    schema: str = TOP2000_M03R_V7_2026_RENDERED_JOB_SCHEMA

    @property
    def manifest(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.manifest_json))

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "pod_template_sha256",
            "source_package_sha256",
            "source_archive_sha256",
            "image_digest_sha256",
            "q2_training_output_inventory_sha256",
            "setting_map_sha256",
            "expected_output_inventory_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "execution_package_sha256",
            "preaccess_completion_receipt_sha256",
            "live_capacity_receipt_sha256",
            "sentinel_qualification_receipt_sha256",
            "panel_artifact_receipt_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(name, value)
        try:
            manifest = json.loads(self.manifest_json)
        except json.JSONDecodeError as exc:
            raise Top2000M03RV72026KubernetesError(
                "rendered manifest is invalid JSON"
            ) from exc
        if not isinstance(manifest, dict) or self.manifest_json != _canonical_json(manifest):
            raise Top2000M03RV72026KubernetesError(
                "rendered manifest must be canonical JSON"
            )
        spec = _mapping(manifest.get("spec"), "Job spec")
        metadata = _mapping(manifest.get("metadata"), "Job metadata")
        annotations = _mapping(metadata.get("annotations"), "Job annotations")
        template = _mapping(spec.get("template"), "Pod template")
        pod = _mapping(template.get("spec"), "Pod spec")
        containers = pod.get("containers")
        if not isinstance(containers, list) or len(containers) != 1:
            raise Top2000M03RV72026KubernetesError(
                "each phase requires one exact container"
            )
        container = _mapping(containers[0], "phase container")
        resources = _mapping(container.get("resources"), "container resources")
        requests = _mapping(resources.get("requests"), "resource requests")
        limits = _mapping(resources.get("limits"), "resource limits")
        env_rows = container.get("env")
        if not isinstance(env_rows, list):
            raise Top2000M03RV72026KubernetesError("container env must be a list")
        env = {
            row.get("name"): row
            for row in env_rows
            if isinstance(row, Mapping) and isinstance(row.get("name"), str)
        }
        expected_geometry = {
            "preaccess": (1, 1, 0, ()),
            "sentinel": (1, 1, 1, (0,)),
            "remaining": (11, self.parallelism, 1, tuple(range(1, 12))),
            "aggregation": (1, 1, 0, ()),
        }[self.phase]
        expected_optional_hashes = {
            "preaccess": (False, False, False, False),
            "sentinel": (True, True, False, False),
            "remaining": (True, True, True, False),
            "aggregation": (True, False, False, True),
        }[self.phase]
        observed_optional_hashes = tuple(
            value is not None
            for value in (
                self.preaccess_completion_receipt_sha256,
                self.live_capacity_receipt_sha256,
                self.sentinel_qualification_receipt_sha256,
                self.panel_artifact_receipt_sha256,
            )
        )
        if (
            self.schema != TOP2000_M03R_V7_2026_RENDERED_JOB_SCHEMA
            or manifest.get("apiVersion") != "batch/v1"
            or manifest.get("kind") != "Job"
            or spec.get("suspend") is not True
            or spec.get("completionMode") != "Indexed"
            or spec.get("completions") != self.completions
            or spec.get("parallelism") != self.parallelism
            or spec.get("backoffLimit") != 0
            or (self.completions, self.parallelism, self.gpu_count_per_pod, self.local_to_global_setting_indices)
            != expected_geometry
            or not 1 <= self.parallelism <= 11
            or self.parallelism * self.gpu_count_per_pod > 16
            or observed_optional_hashes != expected_optional_hashes
            or (self.phase == "preaccess")
            != (self.execution_package_sha256 is None)
            or self.activation_authorized
            or self.manifest_sha256 != _sha256(manifest)
            or self.pod_template_sha256 != _canonical_pod_sha256(pod)
            or _contains_host_path(manifest)
        ):
            raise Top2000M03RV72026KubernetesError(
                "rendered Job geometry, hash, suspension, or volume surface drifted"
            )
        expected_annotations: dict[str, str | None] = {
            "rl-quant/evaluation-phase": self.phase,
            "rl-quant/source-package-sha256": self.source_package_sha256,
            "rl-quant/source-archive-sha256": self.source_archive_sha256,
            "rl-quant/image-digest-sha256": self.image_digest_sha256,
            "rl-quant/q2-output-inventory-sha256": (
                self.q2_training_output_inventory_sha256
            ),
            "rl-quant/setting-map-sha256": self.setting_map_sha256,
            "rl-quant/expected-output-inventory-sha256": (
                self.expected_output_inventory_sha256
            ),
            "rl-quant/execution-package-sha256": self.execution_package_sha256,
            "rl-quant/preaccess-completion-receipt-sha256": (
                self.preaccess_completion_receipt_sha256
            ),
            "rl-quant/live-capacity-receipt-sha256": (
                self.live_capacity_receipt_sha256
            ),
            "rl-quant/sentinel-qualification-receipt-sha256": (
                self.sentinel_qualification_receipt_sha256
            ),
            "rl-quant/panel-artifact-receipt-sha256": (
                self.panel_artifact_receipt_sha256
            ),
        }
        if any(
            (annotations.get(name) != value if value is not None else name in annotations)
            for name, value in expected_annotations.items()
        ):
            raise Top2000M03RV72026KubernetesError(
                "rendered Job annotations do not bind its typed source/output hashes"
            )
        expected_map = {
            "preaccess": None,
            "sentinel": "0",
            "remaining": ",".join(str(value) for value in range(1, 12)),
            "aggregation": None,
        }[self.phase]
        if expected_map is None:
            if "rl-quant/local-to-global-setting-map" in annotations:
                raise Top2000M03RV72026KubernetesError(
                    "CPU phase unexpectedly contains a setting map"
                )
        elif annotations.get("rl-quant/local-to-global-setting-map") != expected_map:
            raise Top2000M03RV72026KubernetesError(
                "local-to-global setting map annotation drifted"
            )
        if (
            pod.get("restartPolicy") != "Never"
            or pod.get("automountServiceAccountToken") is not False
            or pod.get("enableServiceLinks") is not False
            or pod.get("serviceAccountName") != "default"
            or container.get("command") != [TOP2000_M03R_V7_2026_IMAGE_PYTHON]
            or not isinstance(container.get("args"), list)
            or container["args"][:1] != ["-m"]
            or "torch.distributed.run" in container["args"]
            or "torchrun" in container["args"]
            or container.get("terminationMessagePath")
            != M03R_TOP2000_TERMINATION_MESSAGE_PATH
            or container.get("terminationMessagePolicy")
            != M03R_TOP2000_TERMINATION_MESSAGE_POLICY
        ):
            raise Top2000M03RV72026KubernetesError(
                "Pod is not the direct-Python, tokenless, fail-fast profile"
            )
        image = container.get("image")
        if (
            not isinstance(image, str)
            or not image.endswith("@sha256:" + self.image_digest_sha256)
        ):
            raise Top2000M03RV72026KubernetesError(
                "container image does not match the bound digest"
            )
        security = _mapping(container.get("securityContext"), "container security")
        pod_security = _mapping(pod.get("securityContext"), "Pod security")
        capabilities = _mapping(security.get("capabilities"), "capabilities")
        if (
            security.get("allowPrivilegeEscalation") is not False
            or security.get("readOnlyRootFilesystem") is not True
            or capabilities.get("drop") != ["ALL"]
            or pod_security.get("runAsNonRoot") is not True
            or _mapping(pod_security.get("seccompProfile"), "seccomp").get("type")
            != "RuntimeDefault"
        ):
            raise Top2000M03RV72026KubernetesError(
                "Pod or container security context drifted"
            )
        volume_mount_rows = container.get("volumeMounts")
        volume_rows = pod.get("volumes")
        if not isinstance(volume_mount_rows, list) or not isinstance(volume_rows, list):
            raise Top2000M03RV72026KubernetesError(
                "Pod volumes and mounts must be explicit lists"
            )
        mounted_paths = {
            row.get("mountPath"): bool(row.get("readOnly", False))
            for row in volume_mount_rows
            if isinstance(row, Mapping)
        }
        expected_mounts = {
            "preaccess": {
                "/mnt/package": True,
                "/mnt/q2-package": True,
                "/mnt/output": True,
                "/mnt/top2000-raw": True,
                "/mnt/preaccess": False,
                "/tmp": False,
                "/dev/shm": False,
            },
            "sentinel": {
                "/mnt/package": True,
                "/mnt/q2-package": True,
                "/mnt/output": True,
                "/mnt/preaccess": True,
                "/mnt/evaluation-output": False,
                "/tmp": False,
                "/dev/shm": False,
            },
            "remaining": {
                "/mnt/package": True,
                "/mnt/q2-package": True,
                "/mnt/output": True,
                "/mnt/preaccess": True,
                "/mnt/evaluation-output": False,
                "/tmp": False,
                "/dev/shm": False,
            },
            "aggregation": {
                "/mnt/package": True,
                "/mnt/preaccess": True,
                "/mnt/evaluation-output": True,
                "/mnt/aggregate-output": False,
                "/tmp": False,
                "/dev/shm": False,
            },
        }[self.phase]
        volume_names = {
            row.get("name") for row in volume_rows if isinstance(row, Mapping)
        }
        mount_names = {
            row.get("name") for row in volume_mount_rows if isinstance(row, Mapping)
        }
        if (
            mounted_paths != expected_mounts
            or len(volume_names) != len(volume_rows)
            or len(mount_names) != len(volume_mount_rows)
            or volume_names != mount_names
        ):
            raise Top2000M03RV72026KubernetesError(
                "phase mount inventory or read-only ownership drifted"
            )
        gpu_request = requests.get("nvidia.com/gpu")
        gpu_limit = limits.get("nvidia.com/gpu")
        if self.gpu_count_per_pod == 0:
            if (
                gpu_request != "0"
                or gpu_limit != "0"
                or env.get("NVIDIA_VISIBLE_DEVICES", {}).get("value") != "none"
                or env.get("CUDA_VISIBLE_DEVICES", {}).get("value") != ""
                or "nodeSelector" in pod
                or "affinity" in pod
            ):
                raise Top2000M03RV72026KubernetesError(
                    "CPU phase must bind zero GPUs, both masks, and no H100 selector"
                )
        elif (
            gpu_request != "1"
            or gpu_limit != "1"
            or "NVIDIA_VISIBLE_DEVICES" in env
            or "CUDA_VISIBLE_DEVICES" in env
            or pod.get("nodeSelector") != M03R_TOP2000_H100_POOL_NODE_SELECTOR
            or pod.get("priorityClassName") != M03R_TOP2000_PRIORITY_CLASS_NAME
            or pod.get("tolerations") != [M03R_TOP2000_MULTI_GPU_TOLERATION]
        ):
            raise Top2000M03RV72026KubernetesError(
                "GPU phase must request one unmasked H100 on the approved pool"
            )
        if self.gpu_count_per_pod == 1:
            expected_affinity = {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
                                        "operator": "In",
                                        "values": [
                                            TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
            if (
                pod.get("affinity") != expected_affinity
                or annotations.get("rl-quant/h100-gpu-name")
                != TOP2000_M03R_V7_2026_H100_GPU_NAME
                or annotations.get("rl-quant/h100-memory-min-bytes")
                != str(TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES)
                or annotations.get("rl-quant/h100-memory-max-bytes")
                != str(TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES)
                or annotations.get("rl-quant/h100-compute-capability") != "9.0"
            ):
                raise Top2000M03RV72026KubernetesError(
                    "H100 affinity or runtime model requirements drifted"
                )
            args = cast(list[str], container["args"])
            try:
                map_value = args[args.index("--setting-index-map") + 1]
            except (ValueError, IndexError) as exc:
                raise Top2000M03RV72026KubernetesError(
                    "GPU worker omitted its explicit setting-index map"
                ) from exc
            if map_value != expected_map:
                raise Top2000M03RV72026KubernetesError(
                    "worker argv and typed local-to-global map differ"
                )


def validate_top2000_m03r_v7_seed17_2026_rendered_job(
    value: Top2000M03RV72026RenderedJob,
) -> None:
    if not isinstance(value, Top2000M03RV72026RenderedJob):
        raise Top2000M03RV72026KubernetesError(
            "rendered Job must use the typed evaluation surface"
        )
    value.__post_init__()


def _volume(name: str, binding: Top2000M03RV72026PersistentVolumeBinding) -> dict[str, Any]:
    return {
        "name": name,
        "persistentVolumeClaim": {"claimName": binding.claim_name},
    }


def _mount(
    name: str,
    binding: Top2000M03RV72026PersistentVolumeBinding,
    mount_path: str,
    *,
    read_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mount: dict[str, Any] = {
        "name": name,
        "mountPath": mount_path,
        "subPath": binding.subpath,
    }
    if read_only:
        mount["readOnly"] = True
    return _volume(name, binding), mount


def _mount_surface(
    phase: str,
    template: Top2000M03RV72026KubernetesTemplate,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definitions: dict[str, tuple[Top2000M03RV72026PersistentVolumeBinding, str, bool]] = {
        "source-package": (
            template.source_package,
            TOP2000_M03R_V7_2026_SOURCE_PACKAGE_MOUNT,
            True,
        ),
    }
    if phase == "preaccess":
        definitions.update(
            {
                "q2-package": (template.q2_package, "/mnt/q2-package", True),
                "q2-output": (
                    template.q2_output,
                    TOP2000_M03R_V7_2026_Q2_OUTPUT_MOUNT,
                    True,
                ),
                "raw-data": (
                    template.raw_data,
                    TOP2000_M03R_V7_2026_RAW_DATA_MOUNT,
                    True,
                ),
                "preaccess": (
                    template.preaccess_output,
                    TOP2000_M03R_V7_2026_PREACCESS_MOUNT,
                    False,
                ),
            }
        )
    elif phase in {"sentinel", "remaining"}:
        definitions.update(
            {
                "q2-package": (template.q2_package, "/mnt/q2-package", True),
                "q2-output": (
                    template.q2_output,
                    TOP2000_M03R_V7_2026_Q2_OUTPUT_MOUNT,
                    True,
                ),
                "preaccess": (
                    template.preaccess_output,
                    TOP2000_M03R_V7_2026_PREACCESS_MOUNT,
                    True,
                ),
                "evaluation-output": (
                    template.evaluation_output,
                    TOP2000_M03R_V7_2026_EVALUATION_OUTPUT_MOUNT,
                    False,
                ),
            }
        )
    elif phase == "aggregation":
        definitions.update(
            {
                "preaccess": (
                    template.preaccess_output,
                    TOP2000_M03R_V7_2026_PREACCESS_MOUNT,
                    True,
                ),
                "evaluation-output": (
                    template.evaluation_output,
                    TOP2000_M03R_V7_2026_EVALUATION_OUTPUT_MOUNT,
                    True,
                ),
                "aggregation-output": (
                    template.aggregation_output,
                    TOP2000_M03R_V7_2026_AGGREGATION_OUTPUT_MOUNT,
                    False,
                ),
            }
        )
    volumes: list[dict[str, Any]] = []
    mounts: list[dict[str, Any]] = []
    for name, (binding, path, read_only) in definitions.items():
        volume, mount = _mount(name, binding, path, read_only=read_only)
        volumes.append(volume)
        mounts.append(mount)
    volumes.extend(
        [
            {"name": "tmp", "emptyDir": {}},
            {
                "name": "dshm",
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": template.resources.dshm_size_limit,
                },
            },
        ]
    )
    mounts.extend(
        [
            {"name": "tmp", "mountPath": "/tmp"},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ]
    )
    return volumes, mounts


def _base_annotations(
    source: Top2000M03RV72026SourcePackagePlan,
    *,
    phase: str,
    run_id: str,
) -> dict[str, str]:
    artifacts = source.artifacts
    return {
        "rl-quant/run-id": run_id,
        "rl-quant/evaluation-phase": phase,
        "rl-quant/source-package-sha256": source.package_plan_sha256,
        "rl-quant/protocol-sha256": source.evaluation_protocol_sha256,
        "rl-quant/source-archive-sha256": artifacts.source_archive_sha256,
        "rl-quant/source-manifest-sha256": artifacts.source_manifest_sha256,
        "rl-quant/dependency-lock-sha256": artifacts.dependency_lock_sha256,
        "rl-quant/evaluation-source-inventory-sha256": (
            artifacts.evaluation_source_inventory_sha256
        ),
        "rl-quant/q2-package-plan-file-sha256": (
            artifacts.q2_package_plan_file_sha256
        ),
        "rl-quant/q2-package-plan-receipt-sha256": (
            artifacts.q2_package_plan_receipt_sha256
        ),
        "rl-quant/q2-completion-coverage-file-sha256": (
            artifacts.q2_completion_coverage_receipt_file_sha256
        ),
        "rl-quant/q2-output-inventory-sha256": (
            artifacts.q2_training_output_inventory_sha256
        ),
        "rl-quant/pre2026-cache-file-sha256": artifacts.pre2026_cache_file_sha256,
        "rl-quant/raw-data-manifest-file-sha256": (
            artifacts.raw_data_manifest_file_sha256
        ),
        "rl-quant/raw-universe-file-sha256": artifacts.raw_universe_file_sha256,
        "rl-quant/image-digest-sha256": artifacts.image_digest_sha256,
        "rl-quant/setting-map-sha256": source.setting_map_sha256,
        "rl-quant/expected-output-inventory-sha256": (
            source.expected_output_inventory_sha256
        ),
        "rl-quant/data-role": "development-only-nonreportable",
    }


def _execution_annotations(
    package: Top2000M03RV72026ExecutionPackagePlan,
) -> dict[str, str]:
    preaccess = package.preaccess
    return {
        "rl-quant/execution-package-sha256": package.execution_package_sha256,
        "rl-quant/frozen-plan-file-sha256": preaccess.frozen_plan_file_sha256,
        "rl-quant/frozen-plan-receipt-sha256": preaccess.frozen_plan_receipt_sha256,
        "rl-quant/retrospective-cache-file-sha256": (
            preaccess.retrospective_cache_file_sha256
        ),
        "rl-quant/chronology-receipt-sha256": preaccess.chronology_receipt_sha256,
        "rl-quant/preaccess-completion-file-sha256": (
            preaccess.preaccess_completion_receipt_file_sha256
        ),
        "rl-quant/preaccess-completion-receipt-sha256": (
            preaccess.preaccess_completion_receipt_sha256
        ),
        "rl-quant/factor-data-file-sha256": preaccess.factor_data_file_sha256,
        "rl-quant/factor-data-receipt-sha256": preaccess.factor_data_receipt_sha256,
    }


def _worker_args(
    phase: str,
    source: Top2000M03RV72026SourcePackagePlan,
    execution: Top2000M03RV72026ExecutionPackagePlan | None,
    panel: Top2000M03RV72026PanelArtifactBindings | None,
) -> list[str]:
    paths = source.paths
    if phase == "preaccess":
        return [
            "-m",
            source.preaccess_worker_module,
            "run-preaccess",
            "--source-package-plan",
            paths.source_package_plan_path,
            "--source-package-plan-sha256",
            source.package_plan_sha256,
            "--output-root",
            TOP2000_M03R_V7_2026_PREACCESS_MOUNT,
        ]
    if execution is None:
        raise Top2000M03RV72026KubernetesError(
            f"{phase} requires a post-preaccess execution package"
        )
    if phase in {"sentinel", "remaining"}:
        setting_map = (
            "0" if phase == "sentinel" else ",".join(str(value) for value in range(1, 12))
        )
        return [
            "-m",
            source.setting_worker_module,
            "--plan",
            paths.frozen_plan_path,
            "--plan-file-sha256",
            execution.preaccess.frozen_plan_file_sha256,
            "--plan-receipt-sha256",
            execution.preaccess.frozen_plan_receipt_sha256,
            "--execution-source-inventory-sha256",
            source.artifacts.evaluation_source_inventory_sha256,
            "--retrospective-cache",
            paths.retrospective_cache_path,
            "--retrospective-cache-file-sha256",
            execution.preaccess.retrospective_cache_file_sha256,
            "--output-root",
            paths.evaluation_output_root,
            "--setting-index-map",
            setting_map,
        ]
    if panel is None:
        raise Top2000M03RV72026KubernetesError(
            "aggregation requires the complete 12-setting artifact binding"
        )
    return [
        "-m",
        source.panel_worker_module,
        "--plan",
        paths.frozen_plan_path,
        "--plan-file-sha256",
        execution.preaccess.frozen_plan_file_sha256,
        "--plan-receipt-sha256",
        execution.preaccess.frozen_plan_receipt_sha256,
        "--execution-source-inventory-sha256",
        source.artifacts.evaluation_source_inventory_sha256,
        "--factor-data",
        paths.factor_data_path,
        "--factor-data-file-sha256",
        execution.preaccess.factor_data_file_sha256,
        "--execution-output-root",
        paths.evaluation_output_root,
        "--panel-artifact-receipt-sha256",
        panel.panel_artifact_receipt_sha256,
        "--output-root",
        paths.aggregation_output_root,
    ]


def _render_job(
    *,
    phase: Literal["preaccess", "sentinel", "remaining", "aggregation"],
    source: Top2000M03RV72026SourcePackagePlan,
    execution: Top2000M03RV72026ExecutionPackagePlan | None,
    template: Top2000M03RV72026KubernetesTemplate,
    completions: int,
    parallelism: int,
    gpu_count: int,
    local_to_global: tuple[int, ...],
    extra_annotations: Mapping[str, str] | None = None,
    panel: Top2000M03RV72026PanelArtifactBindings | None = None,
) -> Top2000M03RV72026RenderedJob:
    source.__post_init__()
    if execution is not None:
        execution.__post_init__()
        if execution.source_package.package_plan_sha256 != source.package_plan_sha256:
            raise Top2000M03RV72026KubernetesError(
                "execution and source package identities differ"
            )
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v7-2026-evaluation",
        "app.kubernetes.io/managed-by": "receipt-gated-research",
        "k8s-user": "yding4",
        "rl-quant/run-id": template.run_id,
        "rl-quant/phase": phase,
        "runai/queue": template.runai_queue,
    }
    annotations = _base_annotations(
        source,
        phase=phase,
        run_id=template.run_id,
    )
    if execution is not None:
        annotations.update(_execution_annotations(execution))
    if extra_annotations is not None:
        annotations.update(dict(extra_annotations))
    volumes, mounts = _mount_surface(phase, template)
    env: list[dict[str, Any]] = [
        {"name": "PYTHONPATH", "value": source.paths.evaluation_source_pythonpath},
        {"name": "PYTHONNOUSERSITE", "value": "1"},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONHASHSEED", "value": "0"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
    ]
    if gpu_count == 0:
        env.extend(
            [
                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
                {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
            ]
        )
    else:
        env.extend(
            [
                {"name": "WORLD_SIZE", "value": "1"},
                {"name": "RANK", "value": "0"},
                {"name": "LOCAL_WORLD_SIZE", "value": "1"},
                {"name": "LOCAL_RANK", "value": "0"},
                {"name": "CUBLAS_WORKSPACE_CONFIG", "value": ":4096:8"},
            ]
        )
        if phase == "sentinel":
            env.append({"name": "JOB_COMPLETION_INDEX", "value": "0"})
        else:
            env.append(
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
                }
            )
    resource_values = {
        "cpu": template.resources.cpu_request,
        "memory": template.resources.memory_request,
        "ephemeral-storage": template.resources.ephemeral_storage_request,
        "nvidia.com/gpu": str(gpu_count),
    }
    resource_limits = {
        "cpu": template.resources.cpu_limit,
        "memory": template.resources.memory_limit,
        "ephemeral-storage": template.resources.ephemeral_storage_limit,
        "nvidia.com/gpu": str(gpu_count),
    }
    pod: dict[str, Any] = {
        "restartPolicy": "Never",
        "serviceAccount": template.service_account_name,
        "serviceAccountName": template.service_account_name,
        "schedulerName": template.scheduler_name,
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
                "name": "worker",
                "image": source.artifacts.image_reference,
                "imagePullPolicy": "IfNotPresent",
                "command": [TOP2000_M03R_V7_2026_IMAGE_PYTHON],
                "args": _worker_args(phase, source, execution, panel),
                "env": env,
                "resources": {
                    "requests": resource_values,
                    "limits": resource_limits,
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "terminationMessagePath": M03R_TOP2000_TERMINATION_MESSAGE_PATH,
                "terminationMessagePolicy": M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
                "volumeMounts": mounts,
            }
        ],
        "volumes": volumes,
    }
    if gpu_count == 1:
        pod.update(
            {
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
                                            "key": M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
                                            "operator": "In",
                                            "values": [
                                                TOP2000_M03R_V7_2026_H100_PRODUCT_LABEL
                                            ],
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
            }
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
            "activeDeadlineSeconds": template.resources.active_deadline_seconds,
            "ttlSecondsAfterFinished": template.resources.ttl_seconds_after_finished,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": pod,
            },
        },
    }
    rendered = Top2000M03RV72026RenderedJob(
        phase=phase,
        manifest_json=_canonical_json(manifest),
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_canonical_pod_sha256(pod),
        source_package_sha256=source.package_plan_sha256,
        source_archive_sha256=source.artifacts.source_archive_sha256,
        image_digest_sha256=source.artifacts.image_digest_sha256,
        q2_training_output_inventory_sha256=(
            source.artifacts.q2_training_output_inventory_sha256
        ),
        execution_package_sha256=(
            None if execution is None else execution.execution_package_sha256
        ),
        preaccess_completion_receipt_sha256=(
            None
            if execution is None
            else execution.preaccess.preaccess_completion_receipt_sha256
        ),
        live_capacity_receipt_sha256=(
            None
            if extra_annotations is None
            else extra_annotations.get("rl-quant/live-capacity-receipt-sha256")
        ),
        sentinel_qualification_receipt_sha256=(
            None
            if extra_annotations is None
            else extra_annotations.get(
                "rl-quant/sentinel-qualification-receipt-sha256"
            )
        ),
        panel_artifact_receipt_sha256=(
            None if panel is None else panel.panel_artifact_receipt_sha256
        ),
        setting_map_sha256=source.setting_map_sha256,
        expected_output_inventory_sha256=source.expected_output_inventory_sha256,
        local_to_global_setting_indices=local_to_global,
        completions=completions,
        parallelism=parallelism,
        gpu_count_per_pod=gpu_count,
    )
    validate_top2000_m03r_v7_seed17_2026_rendered_job(rendered)
    return rendered


def render_top2000_m03r_v7_seed17_2026_preaccess_job(
    *,
    source_package: Top2000M03RV72026SourcePackagePlan,
    template: Top2000M03RV72026KubernetesTemplate,
) -> Top2000M03RV72026RenderedJob:
    """Render the sole zero-GPU cache/factor stage, still suspended."""

    return _render_job(
        phase="preaccess",
        source=source_package,
        execution=None,
        template=template,
        completions=1,
        parallelism=1,
        gpu_count=0,
        local_to_global=(),
    )


def render_top2000_m03r_v7_seed17_2026_sentinel_job(
    *,
    execution_package: Top2000M03RV72026ExecutionPackagePlan,
    capacity: Top2000M03RV72026LiveCapacityEvidence,
    template: Top2000M03RV72026KubernetesTemplate,
    now_utc: datetime,
) -> Top2000M03RV72026RenderedJob:
    """Render setting 0 as the reusable one-H100 capacity sentinel."""

    capacity.require_fresh(now_utc=now_utc)
    if capacity.cap_remaining < 1:
        raise Top2000M03RV72026KubernetesError(
            "the authorized cap has no room for the setting-0 sentinel"
        )
    package = execution_package
    extra = {
        "rl-quant/live-capacity-receipt-sha256": capacity.receipt_sha256,
        "rl-quant/local-to-global-setting-map": "0",
        "rl-quant/h100-gpu-name": TOP2000_M03R_V7_2026_H100_GPU_NAME,
        "rl-quant/h100-memory-min-bytes": str(
            TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES
        ),
        "rl-quant/h100-memory-max-bytes": str(
            TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES
        ),
        "rl-quant/h100-compute-capability": ".".join(
            str(value) for value in TOP2000_M03R_V7_2026_H100_COMPUTE_CAPABILITY
        ),
        "rl-quant/sentinel-output-reused": "true",
    }
    return _render_job(
        phase="sentinel",
        source=package.source_package,
        execution=package,
        template=template,
        completions=1,
        parallelism=1,
        gpu_count=1,
        local_to_global=(0,),
        extra_annotations=extra,
    )


def render_top2000_m03r_v7_seed17_2026_remaining_indexed_job(
    *,
    execution_package: Top2000M03RV72026ExecutionPackagePlan,
    sentinel: Top2000M03RV72026SentinelQualification,
    capacity: Top2000M03RV72026LiveCapacityEvidence,
    template: Top2000M03RV72026KubernetesTemplate,
    now_utc: datetime,
) -> Top2000M03RV72026RenderedJob:
    """Render settings 1..11 only, with dynamic one-H100 parallelism."""

    capacity.require_fresh(now_utc=now_utc)
    if sentinel.execution_package_sha256 != execution_package.execution_package_sha256:
        raise Top2000M03RV72026KubernetesError(
            "remaining Job cannot consume a sentinel from another package"
        )
    parallelism = capacity.remaining_parallelism
    if parallelism < 1:
        raise Top2000M03RV72026KubernetesError(
            "the authorized cap admits no remaining setting worker"
        )
    local_to_global = tuple(range(1, 12))
    extra = {
        "rl-quant/live-capacity-receipt-sha256": capacity.receipt_sha256,
        "rl-quant/sentinel-qualification-receipt-sha256": sentinel.receipt_sha256,
        "rl-quant/sentinel-setting-completion-file-sha256": (
            sentinel.setting_completion_file_sha256
        ),
        "rl-quant/local-to-global-setting-map": ",".join(
            str(value) for value in local_to_global
        ),
        "rl-quant/setting-zero-in-remaining-job": "false",
        "rl-quant/h100-gpu-name": TOP2000_M03R_V7_2026_H100_GPU_NAME,
        "rl-quant/h100-memory-min-bytes": str(
            TOP2000_M03R_V7_2026_H100_MIN_MEMORY_BYTES
        ),
        "rl-quant/h100-memory-max-bytes": str(
            TOP2000_M03R_V7_2026_H100_MAX_MEMORY_BYTES
        ),
        "rl-quant/h100-compute-capability": ".".join(
            str(value) for value in TOP2000_M03R_V7_2026_H100_COMPUTE_CAPABILITY
        ),
    }
    return _render_job(
        phase="remaining",
        source=execution_package.source_package,
        execution=execution_package,
        template=template,
        completions=11,
        parallelism=parallelism,
        gpu_count=1,
        local_to_global=local_to_global,
        extra_annotations=extra,
    )


def render_top2000_m03r_v7_seed17_2026_aggregation_job(
    *,
    execution_package: Top2000M03RV72026ExecutionPackagePlan,
    panel_artifacts: Top2000M03RV72026PanelArtifactBindings,
    template: Top2000M03RV72026KubernetesTemplate,
) -> Top2000M03RV72026RenderedJob:
    """Render one zero-GPU six-fold aggregation Job, still suspended."""

    if (
        panel_artifacts.execution_package_sha256
        != execution_package.execution_package_sha256
    ):
        raise Top2000M03RV72026KubernetesError(
            "aggregation inputs come from another execution package"
        )
    extra = {
        "rl-quant/panel-artifact-receipt-sha256": (
            panel_artifacts.panel_artifact_receipt_sha256
        ),
        "rl-quant/twelve-setting-inventory-sha256": (
            panel_artifacts.twelve_setting_completion_inventory_sha256
        ),
        "rl-quant/seventy-two-fold-inventory-sha256": (
            panel_artifacts.seventy_two_fold_artifact_inventory_sha256
        ),
        "rl-quant/setting-zero-source": "reused-sentinel",
        "rl-quant/checkpoint-folds-pooled": "false",
    }
    return _render_job(
        phase="aggregation",
        source=execution_package.source_package,
        execution=execution_package,
        template=template,
        completions=1,
        parallelism=1,
        gpu_count=0,
        local_to_global=(),
        extra_annotations=extra,
        panel=panel_artifacts,
    )


__all__ = [
    "TOP2000_M03R_V7_2026_REMAINING_COMPLETIONS",
    "TOP2000_M03R_V7_2026_USER_H100_CAP",
    "Top2000M03RV72026JobResources",
    "Top2000M03RV72026KubernetesError",
    "Top2000M03RV72026KubernetesTemplate",
    "Top2000M03RV72026LiveCapacityEvidence",
    "Top2000M03RV72026PersistentVolumeBinding",
    "Top2000M03RV72026RenderedJob",
    "build_top2000_m03r_v7_seed17_2026_live_capacity_evidence",
    "render_top2000_m03r_v7_seed17_2026_aggregation_job",
    "render_top2000_m03r_v7_seed17_2026_preaccess_job",
    "render_top2000_m03r_v7_seed17_2026_remaining_indexed_job",
    "render_top2000_m03r_v7_seed17_2026_sentinel_job",
    "validate_top2000_m03r_v7_seed17_2026_rendered_job",
]
