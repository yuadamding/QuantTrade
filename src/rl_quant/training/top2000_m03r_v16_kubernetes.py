"""Suspended, receipt-gated Seadragon Job manifests for M03R-v16."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    semantic_sha256 as _sha256,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_POOL_NODE_SELECTOR,
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS,
    M03R_TOP2000_MULTI_GPU_TOLERATION,
    M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS,
    M03R_TOP2000_PRIORITY_CLASS_NAME,
    M03R_TOP2000_TERMINATION_MESSAGE_PATH,
    M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    M03RV16AdmittedJobAuthority,
    M03RV16PhaseLaunchAuthority,
    M03RV16PodRuntimeAttestation,
    M03RV16QualificationActivation,
    M03RV16QualificationOuterAccessAuthority,
    M03RV16TrainingActivation,
    _issue_m03r_v16_phase_launch_authority,
    _issue_m03r_v16_training_activation_from_gates,
    admitted_job_authority_file_sha256,
    phase_launch_authority_file_sha256,
    pod_runtime_attestation_file_sha256,
    write_m03r_v16_phase_launch_authority,
)
from rl_quant.training.top2000_m03r_v16_capacity import (
    M03R_V16_CAPACITY_TERMINAL_SCHEMA,
    M03RV16CapacityRankEvidence,
    M03RV16CapacityTerminal,
)
from rl_quant.training.top2000_m03r_v16_lifecycle import (
    M03RV16StorageSemanticsEvidence,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
)
from rl_quant.training.top2000_m03r_v16_static_contract import (
    M03R_V16_STATIC_RESULT_SCHEMA,
)

M03R_V16_NAMESPACE = "yn-gpu-workload"
M03R_V16_STATIC_GATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-static-gate-qualification-v2"
)
M03R_V16_CAPACITY_GATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-capacity-gate-qualification-v2"
)
M03R_V16_RENDERED_JOB_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-rendered-suspended-job-v8"
)
_JOB_CONTRACT_ANNOTATION = "rl-quant/job-contract-sha256"
_POD_CONTRACT_ANNOTATION = "rl-quant/pod-contract-sha256"
_LAUNCH_RECEIPT_ANNOTATION = "rl-quant/launch-authority-receipt-sha256"
_LAUNCH_FILE_ANNOTATION = "rl-quant/launch-authority-file-sha256"
_ADMISSION_RECEIPT_ANNOTATION = "rl-quant/admission-authority-receipt-sha256"
_ADMISSION_FILE_ANNOTATION = "rl-quant/admission-authority-file-sha256"
_DRY_RUN_FILE_ANNOTATION = "rl-quant/server-dry-run-file-sha256"
_ADMITTED_MANIFEST_FILE_ANNOTATION = "rl-quant/admitted-manifest-file-sha256"
_POD_ATTESTATION_FILE_ANNOTATION = (
    "rl-quant/pod-runtime-attestation-file-sha256"
)
_POD_ATTESTATION_RECEIPT_ANNOTATION = (
    "rl-quant/pod-runtime-attestation-receipt-sha256"
)
_POD_ATTESTATION_PATH_ANNOTATION = "rl-quant/pod-runtime-attestation-path"
_STORAGE_FILE_ANNOTATION = "rl-quant/storage-semantics-file-sha256"
_STORAGE_RECEIPT_ANNOTATION = "rl-quant/storage-semantics-receipt-sha256"
_PYTHON = "/opt/conda/envs/quanttrade/bin/python"
_WORKER_MODULE = "rl_quant.workflows.top2000_m03r_v16_predictive"
_STATIC_MODULE = "rl_quant.workflows.top2000_m03r_v16_static_validate"
_STORAGE_MODULE = "rl_quant.workflows.top2000_m03r_v16_storage_gate"
_INIT_GATE_BOOTSTRAP = (
    "import pathlib,sys;"
    "sys.dont_write_bytecode=True;"
    "root=pathlib.Path('/mnt/package/source/src').resolve();"
    "sys.path.insert(0,str(root));"
    "from rl_quant.workflows import "
    "top2000_m03r_v16_attestation_gate as gate;"
    "resolved=pathlib.Path(gate.__file__).resolve();"
    "resolved.relative_to(root);"
    "raise SystemExit(gate.main())"
)
_STATIC_GATE_ISSUER = object()
_CAPACITY_GATE_ISSUER = object()
_MAX_GATE_RESULT_BYTES = 64 * 1024**2


def m03r_v16_pod_runtime_attestation_annotations(
    value: M03RV16PodRuntimeAttestation,
) -> dict[str, str]:
    """Return the exact annotations patched before atomic publication."""

    return {
        _POD_ATTESTATION_PATH_ANNOTATION: value.relative_path,
        _POD_ATTESTATION_FILE_ANNOTATION: (
            pod_runtime_attestation_file_sha256(value)
        ),
        _POD_ATTESTATION_RECEIPT_ANNOTATION: value.receipt_sha256,
    }


def _contract_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(manifest)
    for annotation_rows in (
        payload.get("metadata", {}).get("annotations", {}),
        payload.get("spec", {})
        .get("template", {})
        .get("metadata", {})
        .get("annotations", {}),
    ):
        annotation_rows.pop(_JOB_CONTRACT_ANNOTATION, None)
        annotation_rows.pop(_POD_CONTRACT_ANNOTATION, None)
        annotation_rows.pop(_LAUNCH_RECEIPT_ANNOTATION, None)
        annotation_rows.pop(_LAUNCH_FILE_ANNOTATION, None)
        annotation_rows.pop(_ADMISSION_RECEIPT_ANNOTATION, None)
        annotation_rows.pop(_ADMISSION_FILE_ANNOTATION, None)
        annotation_rows.pop(_DRY_RUN_FILE_ANNOTATION, None)
        annotation_rows.pop(_ADMITTED_MANIFEST_FILE_ANNOTATION, None)
        annotation_rows.pop(_STORAGE_FILE_ANNOTATION, None)
        annotation_rows.pop(_STORAGE_RECEIPT_ANNOTATION, None)
        annotation_rows.pop(_POD_ATTESTATION_FILE_ANNOTATION, None)
        annotation_rows.pop(_POD_ATTESTATION_RECEIPT_ANNOTATION, None)
        annotation_rows.pop(_POD_ATTESTATION_PATH_ANNOTATION, None)
    return payload


def _job_contract_sha256(manifest: dict[str, Any]) -> str:
    return _sha256(_contract_payload(manifest))


def _pod_contract_sha256(manifest: dict[str, Any]) -> str:
    payload = _contract_payload(manifest)
    return _sha256(payload["spec"]["template"])


class M03RV16KubernetesError(ValueError):
    """A V16 static, capacity, or suspended-Job identity drifted."""


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV16KubernetesError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class M03RV16StaticGateQualification:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    rendered_manifest_sha256: str
    result_file_sha256: str
    result_receipt_sha256: str
    image_digest_sha256: str
    source_tree_root_sha256: str
    _issuer: object = field(repr=False)
    zero_gpu_requested: bool = True
    zero_gpu_admitted: bool = True
    zero_gpu_observed: bool = True
    training_performed: bool = False
    passed: bool = True
    development_only: bool = True
    schema: str = M03R_V16_STATIC_GATE_SCHEMA

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
    ) -> None:
        package.validate()
        authorization.validate(package)
        for name in (
            "rendered_manifest_sha256",
            "result_file_sha256",
            "result_receipt_sha256",
            "source_tree_root_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self._issuer is not _STATIC_GATE_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256
            != authorization.receipt_sha256
            or self.image_digest_sha256 != package.artifacts.image_digest_sha256
            or not self.zero_gpu_requested
            or not self.zero_gpu_admitted
            or not self.zero_gpu_observed
            or self.training_performed
            or not self.passed
            or not self.development_only
            or self.schema != M03R_V16_STATIC_GATE_SCHEMA
        ):
            raise M03RV16KubernetesError("V16 static gate qualification drifted")

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return _sha256(payload)


@dataclass(frozen=True, slots=True)
class M03RV16CapacityGateQualification:
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    static_gate_receipt_sha256: str
    rendered_manifest_sha256: str
    terminal_file_sha256: str
    terminal_receipt_sha256: str
    image_digest_sha256: str
    source_tree_root_sha256: str
    _issuer: object = field(repr=False)
    world_size: int = 2
    h100s_per_worker: int = 2
    exact_h100_80gb_per_rank: bool = True
    disposable_exact_shape_update_performed: bool = True
    disposable_train_validate_train_executed: bool = True
    nontrivial_qualification_projection_performed: bool = True
    scientific_checkpoint_published: bool = False
    scientific_training_performed: bool = False
    passed: bool = True
    development_only: bool = True
    schema: str = M03R_V16_CAPACITY_GATE_SCHEMA

    def validate_for(
        self,
        package: M03RV16PackagePlan,
        authorization: M03RV16ExecutionAuthorization,
        static: M03RV16StaticGateQualification,
    ) -> None:
        static.validate_for(package, authorization)
        for name in (
            "static_gate_receipt_sha256",
            "rendered_manifest_sha256",
            "terminal_file_sha256",
            "terminal_receipt_sha256",
            "source_tree_root_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self._issuer is not _CAPACITY_GATE_ISSUER
            or self.package_plan_sha256 != package.package_plan_sha256
            or self.execution_authorization_receipt_sha256
            != authorization.receipt_sha256
            or self.static_gate_receipt_sha256 != static.receipt_sha256
            or self.image_digest_sha256 != package.artifacts.image_digest_sha256
            or self.source_tree_root_sha256 != static.source_tree_root_sha256
            or self.world_size != 2
            or self.h100s_per_worker != 2
            or not self.exact_h100_80gb_per_rank
            or not self.disposable_exact_shape_update_performed
            or not self.disposable_train_validate_train_executed
            or not self.nontrivial_qualification_projection_performed
            or self.scientific_checkpoint_published
            or self.scientific_training_performed
            or not self.passed
            or not self.development_only
            or self.schema != M03R_V16_CAPACITY_GATE_SCHEMA
        ):
            raise M03RV16KubernetesError("V16 capacity gate qualification drifted")

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return _sha256(payload)


@dataclass(frozen=True, slots=True)
class M03RV16RenderedSuspendedJob:
    manifest: dict[str, Any]
    manifest_sha256: str
    pod_template_sha256: str
    job_contract_sha256: str
    pod_contract_sha256: str
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    package_plan_file_sha256: str
    execution_authorization_file_sha256: str
    mode: Literal[
        "static",
        "storage",
        "capacity",
        "training",
        "qualification-preflight",
        "qualification",
    ]
    completions: int
    parallelism: int
    gpus_per_completion: int
    launch_authority: M03RV16PhaseLaunchAuthority | None = None
    launch_authority_file_sha256: str | None = None
    admitted_job_authority: M03RV16AdmittedJobAuthority | None = None
    admitted_job_authority_file_sha256: str | None = None
    static_gate_receipt_sha256: str | None = None
    capacity_gate_receipt_sha256: str | None = None
    training_activation_receipt_sha256: str | None = None
    qualification_activation_receipt_sha256: str | None = None
    qualification_outer_access_receipt_sha256: str | None = None
    activation_authorized: bool = False
    economic_training_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    outer_2026_access_authorized: bool = False
    schema: str = M03R_V16_RENDERED_JOB_SCHEMA

    @property
    def maximum_gpu_requests(self) -> int:
        return self.parallelism * self.gpus_per_completion

    def validate(self) -> None:
        for name in (
            "manifest_sha256",
            "pod_template_sha256",
            "job_contract_sha256",
            "pod_contract_sha256",
            "package_plan_sha256",
            "execution_authorization_receipt_sha256",
            "package_plan_file_sha256",
            "execution_authorization_file_sha256",
        ):
            _digest(name, getattr(self, name))
        spec = self.manifest.get("spec")
        if not isinstance(spec, dict):
            raise M03RV16KubernetesError("V16 Job spec is absent")
        template = spec.get("template")
        pod = template.get("spec") if isinstance(template, dict) else None
        if not isinstance(pod, dict):
            raise M03RV16KubernetesError("V16 Pod spec is absent")
        containers = pod.get("containers")
        if not isinstance(containers, list) or len(containers) != 1:
            raise M03RV16KubernetesError("V16 Job requires one container")
        resources = containers[0].get("resources", {})
        requests = resources.get("requests", {})
        limits = resources.get("limits", {})
        annotations = self.manifest.get("metadata", {}).get("annotations", {})
        expected = {
            "static": (1, 1, 0),
            "storage": (1, 1, 0),
            "capacity": (1, 1, 2),
            "training": (3, 3, 2),
            "qualification-preflight": (3, 3, 0),
            "qualification": (3, 3, 2),
        }[self.mode]
        gpu_resources_valid = (
            requests.get("nvidia.com/gpu") == str(expected[2])
            and limits.get("nvidia.com/gpu") == str(expected[2])
        )
        if (
            self.manifest_sha256 != _sha256(self.manifest)
            or self.pod_template_sha256 != _sha256(pod)
            or self.manifest.get("apiVersion") != "batch/v1"
            or self.manifest.get("kind") != "Job"
            or self.manifest.get("metadata", {}).get("namespace")
            != M03R_V16_NAMESPACE
            or annotations.get("rl-quant/package-plan-sha256")
            != self.package_plan_sha256
            or annotations.get("rl-quant/execution-authorization-sha256")
            != self.execution_authorization_receipt_sha256
            or annotations.get("rl-quant/package-plan-file-sha256")
            != self.package_plan_file_sha256
            or annotations.get("rl-quant/execution-authorization-file-sha256")
            != self.execution_authorization_file_sha256
            or annotations.get(_JOB_CONTRACT_ANNOTATION)
            != self.job_contract_sha256
            or annotations.get(_POD_CONTRACT_ANNOTATION)
            != self.pod_contract_sha256
            or self.job_contract_sha256 != _job_contract_sha256(self.manifest)
            or self.pod_contract_sha256 != _pod_contract_sha256(self.manifest)
            or spec.get("suspend") is not True
            or spec.get("completionMode") != "Indexed"
            or spec.get("completions") != expected[0]
            or spec.get("parallelism") != expected[1]
            or spec.get("backoffLimit") != 0
            or (self.completions, self.parallelism, self.gpus_per_completion)
            != expected
            or not gpu_resources_valid
            or self.maximum_gpu_requests > 6
            or pod.get("restartPolicy") != "Never"
            or pod.get("automountServiceAccountToken") is not False
            or self.activation_authorized
            or self.economic_training_authorized
            or self.reinforcement_learning_authorized
            or self.outer_2026_access_authorized
            or self.schema != M03R_V16_RENDERED_JOB_SCHEMA
        ):
            raise M03RV16KubernetesError("rendered V16 Job drifted")
        if self.mode == "static":
            environment = {
                row["name"]: row.get("value") for row in containers[0].get("env", [])
            }
            if (
                environment.get("NVIDIA_VISIBLE_DEVICES") != "none"
                or self.static_gate_receipt_sha256 is not None
                or self.capacity_gate_receipt_sha256 is not None
                or self.training_activation_receipt_sha256 is not None
                or self.qualification_activation_receipt_sha256 is not None
                or self.launch_authority is not None
                or self.launch_authority_file_sha256 is not None
                or self.admitted_job_authority is not None
                or self.admitted_job_authority_file_sha256 is not None
            ):
                raise M03RV16KubernetesError("V16 static Job is not GPU neutral")
        elif self.mode == "storage":
            environment = {
                row["name"]: row.get("value")
                for row in containers[0].get("env", [])
            }
            if (
                environment.get("NVIDIA_VISIBLE_DEVICES") != "none"
                or self.static_gate_receipt_sha256 is None
                or self.capacity_gate_receipt_sha256 is not None
                or self.training_activation_receipt_sha256 is not None
                or self.qualification_activation_receipt_sha256 is not None
                or pod.get("initContainers") not in (None, [])
                or self.launch_authority is not None
                or self.admitted_job_authority is not None
            ):
                raise M03RV16KubernetesError(
                    "V16 storage Job is not a zero-GPU predecessor gate"
                )
        elif self.mode == "qualification-preflight":
            if (
                requests
                != {
                    "cpu": "12",
                    "memory": "64Gi",
                    "ephemeral-storage": "8Gi",
                    "nvidia.com/gpu": "0",
                }
                or limits
                != {
                    "cpu": "16",
                    "memory": "128Gi",
                    "ephemeral-storage": "16Gi",
                    "nvidia.com/gpu": "0",
                }
                or pod.get("nodeSelector") is not None
                or pod.get("priorityClassName") is not None
                or pod.get("tolerations") not in (None, [])
                or self.static_gate_receipt_sha256 is None
                or self.capacity_gate_receipt_sha256 is None
                or self.qualification_activation_receipt_sha256 is None
                or self.qualification_outer_access_receipt_sha256 is not None
            ):
                raise M03RV16KubernetesError(
                    "V16 qualification preflight is not CPU neutral"
                )
        elif self.mode != "qualification-preflight" and (
            pod.get("nodeSelector") != M03R_TOP2000_H100_POOL_NODE_SELECTOR
            or pod.get("priorityClassName") != M03R_TOP2000_PRIORITY_CLASS_NAME
            or pod.get("tolerations") != [M03R_TOP2000_MULTI_GPU_TOLERATION]
            or self.static_gate_receipt_sha256 is None
            or (
                self.mode
                in {"training", "qualification-preflight", "qualification"}
                and self.capacity_gate_receipt_sha256 is None
            )
            or (
                self.mode == "training"
                and self.training_activation_receipt_sha256 is None
            )
            or (
                self.mode in {"qualification-preflight", "qualification"}
                and self.qualification_activation_receipt_sha256 is None
            )
            or (
                self.mode == "qualification"
                and self.qualification_outer_access_receipt_sha256 is None
            )
        ):
            raise M03RV16KubernetesError("V16 H100 Job profile drifted")
        elif (
            len(pod.get("initContainers", ())) != 1
            or pod["initContainers"][0].get("name")
            != "runtime-attestation-gate"
            or pod["initContainers"][0].get("image")
            != containers[0].get("image")
            or "@sha256:" not in str(containers[0].get("image"))
            or pod["initContainers"][0].get("command") != [_PYTHON]
            or pod["initContainers"][0].get("args", ())[:4]
            != ["-I", "-B", "-c", _INIT_GATE_BOOTSTRAP]
            or _INIT_GATE_BOOTSTRAP
            not in pod["initContainers"][0].get("args", ())
            or pod["initContainers"][0].get("resources")
            != {
                "requests": {"cpu": "50m", "memory": "512Mi"},
                "limits": {"cpu": "250m", "memory": "1Gi"},
            }
            or not {
                "M03R_V16_CURRENT_POD_UID",
                "M03R_V16_CURRENT_POD_NAME",
                "M03R_V16_CURRENT_NODE_NAME",
                "M03R_V16_POD_ATTESTATION_FILE_SHA256",
                "M03R_V16_POD_ATTESTATION_RECEIPT_SHA256",
                "M03R_V16_POD_ATTESTATION_PATH",
                "M03R_V16_STORAGE_FILE_SHA256",
                "M03R_V16_STORAGE_RECEIPT_SHA256",
            }.issubset(
                {
                    row.get("name")
                    for row in containers[0].get("env", ())
                    if isinstance(row, dict)
                }
            )
            or "--pod-runtime-attestation" not in containers[0].get("args", ())
            or "--pod-runtime-attestation-marker"
            not in containers[0].get("args", ())
        ):
            raise M03RV16KubernetesError(
                "V16 H100 Pod runtime attestation gate drifted"
            )
        elif any(
            value is not None
            for value in (
                self.launch_authority,
                self.launch_authority_file_sha256,
                self.admitted_job_authority,
                self.admitted_job_authority_file_sha256,
            )
        ):
            if (
                self.launch_authority is None
                or self.launch_authority_file_sha256 is None
                or self.admitted_job_authority is None
                or self.admitted_job_authority_file_sha256 is None
                or annotations.get(_LAUNCH_RECEIPT_ANNOTATION)
                != self.launch_authority.receipt_sha256
                or annotations.get(_LAUNCH_FILE_ANNOTATION)
                != self.launch_authority_file_sha256
                or annotations.get(_ADMISSION_RECEIPT_ANNOTATION)
                != self.admitted_job_authority.receipt_sha256
                or annotations.get(_ADMISSION_FILE_ANNOTATION)
                != self.admitted_job_authority_file_sha256
                or annotations.get(_DRY_RUN_FILE_ANNOTATION)
                != self.admitted_job_authority.server_side_dry_run_file_sha256
                or annotations.get(_ADMITTED_MANIFEST_FILE_ANNOTATION)
                != self.admitted_job_authority.admitted_manifest_file_sha256
                or annotations.get(_STORAGE_FILE_ANNOTATION)
                != self.launch_authority.storage_semantics_file_sha256
                or annotations.get(_STORAGE_RECEIPT_ANNOTATION)
                != self.launch_authority.storage_semantics_receipt_sha256
            ):
                raise M03RV16KubernetesError(
                    "V16 admitted launch binding is incomplete"
                )
            _digest(
                "launch_authority_file_sha256",
                str(self.launch_authority_file_sha256),
            )
            _digest(
                "admitted_job_authority_file_sha256",
                str(self.admitted_job_authority_file_sha256),
            )
            prerequisite = {
                "capacity": self.static_gate_receipt_sha256,
                "training": self.training_activation_receipt_sha256,
                "qualification-preflight": (
                    self.qualification_activation_receipt_sha256
                ),
                "qualification": self.qualification_outer_access_receipt_sha256,
            }[self.mode]
            if (
                self.launch_authority.phase != self.mode
                or self.launch_authority.prerequisite_authority_receipt_sha256
                != prerequisite
                or self.launch_authority.job_contract_sha256
                != self.job_contract_sha256
                or self.launch_authority.pod_contract_sha256
                != self.pod_contract_sha256
                or self.launch_authority.admission_receipt_sha256
                != self.admitted_job_authority.receipt_sha256
                or self.launch_authority.admission_file_sha256
                != self.admitted_job_authority_file_sha256
                or phase_launch_authority_file_sha256(self.launch_authority)
                != self.launch_authority_file_sha256
                or admitted_job_authority_file_sha256(
                    self.admitted_job_authority
                )
                != self.admitted_job_authority_file_sha256
            ):
                raise M03RV16KubernetesError("V16 phase launch authority drifted")


def _read_exact_json(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16KubernetesError("V16 gate result is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_GATE_RESULT_BYTES
        ):
            raise M03RV16KubernetesError("V16 gate result type or size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV16KubernetesError("V16 gate result changed while read")
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise M03RV16KubernetesError("V16 gate result hash drifted")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16KubernetesError("V16 gate result is malformed") from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise M03RV16KubernetesError("V16 gate result is not canonical")
    return payload


def load_and_issue_m03r_v16_static_gate(
    result_path: str | Path,
    *,
    expected_result_file_sha256: str,
    rendered: M03RV16RenderedSuspendedJob,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
) -> M03RV16StaticGateQualification:
    """Issue a static authority only from the exact immutable result file."""

    rendered.validate()
    package.validate()
    authorization.validate(package)
    payload = _read_exact_json(Path(result_path), expected_result_file_sha256)
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    receipt = payload.get("receipt_sha256")
    if (
        rendered.mode != "static"
        or payload.get("schema") != M03R_V16_STATIC_RESULT_SCHEMA
        or receipt != _sha256(unsigned)
        or payload.get("package_plan_sha256") != package.package_plan_sha256
        or payload.get("execution_authorization_receipt_sha256")
        != authorization.receipt_sha256
        or payload.get("image_digest_sha256")
        != package.artifacts.image_digest_sha256
        or payload.get("gpu_mask") != "none"
        or payload.get("gpu_requests") != 0
        or payload.get("gpu_limits") != 0
        or payload.get("unmasked_visibility_claimed") is not False
        or payload.get("training_performed") is not False
        or payload.get("initial_state_strict_loaded_all_settings") is not True
        or payload.get("development_only") is not True
    ):
        raise M03RV16KubernetesError("V16 static result authority drifted")
    source_root = str(payload.get("source_tree_root_sha256"))
    _digest("source_tree_root_sha256", source_root)
    result = M03RV16StaticGateQualification(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        rendered_manifest_sha256=rendered.manifest_sha256,
        result_file_sha256=expected_result_file_sha256,
        result_receipt_sha256=str(receipt),
        image_digest_sha256=package.artifacts.image_digest_sha256,
        source_tree_root_sha256=source_root,
        _issuer=_STATIC_GATE_ISSUER,
    )
    result.validate_for(package, authorization)
    return result


def _capacity_from_payload(payload: dict[str, Any]) -> M03RV16CapacityTerminal:
    try:
        row = dict(payload["capacity"])
        rank_rows = tuple(
            M03RV16CapacityRankEvidence(**dict(value))
            for value in row.pop("rank_evidence")
        )
        result = M03RV16CapacityTerminal(rank_evidence=rank_rows, **row)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16KubernetesError("V16 capacity result is malformed") from exc
    result.validate()
    return result


def load_and_issue_m03r_v16_capacity_gate(
    terminal_path: str | Path,
    *,
    expected_terminal_file_sha256: str,
    rendered: M03RV16RenderedSuspendedJob,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    static: M03RV16StaticGateQualification,
) -> M03RV16CapacityGateQualification:
    """Issue a capacity authority only from the exact two-rank terminal."""

    rendered.validate()
    static.validate_for(package, authorization)
    payload = _read_exact_json(Path(terminal_path), expected_terminal_file_sha256)
    capacity = _capacity_from_payload(payload)
    if (
        rendered.mode != "capacity"
        or rendered.launch_authority is None
        or rendered.admitted_job_authority is None
        or rendered.static_gate_receipt_sha256 != static.receipt_sha256
        or payload.get("schema") != M03R_V16_CAPACITY_TERMINAL_SCHEMA
        or payload.get("capacity_receipt_sha256") != capacity.receipt_sha256
        or payload.get("package_plan_sha256") != package.package_plan_sha256
        or payload.get("authorization_receipt_sha256")
        != authorization.receipt_sha256
        or payload.get("rendered_manifest_sha256")
        != rendered.job_contract_sha256
        or payload.get("pod_template_sha256") != rendered.pod_contract_sha256
        or payload.get("launch_authority_receipt_sha256")
        != rendered.launch_authority.receipt_sha256
        or payload.get("admitted_job_authority_receipt_sha256")
        != rendered.admitted_job_authority.receipt_sha256
        or payload.get("job_uid") != rendered.admitted_job_authority.job_uid
        or not isinstance(
            payload.get("pod_runtime_attestation_receipt_sha256"), str
        )
        or not isinstance(payload.get("pod_uid"), str)
        or not payload.get("pod_uid")
        or payload.get("scientific_training_performed") is not False
        or payload.get("disposable_optimizer_update_executed") is not True
        or payload.get("disposable_train_validate_train_executed") is not True
        or payload.get("scientific_checkpoint_published") is not False
        or payload.get("development_only") is not True
    ):
        raise M03RV16KubernetesError("V16 capacity result authority drifted")
    source_root = str(payload.get("source_tree_root_sha256"))
    _digest("source_tree_root_sha256", source_root)
    if source_root != static.source_tree_root_sha256:
        raise M03RV16KubernetesError("V16 capacity source tree differs from static")
    result = M03RV16CapacityGateQualification(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        static_gate_receipt_sha256=static.receipt_sha256,
        rendered_manifest_sha256=rendered.manifest_sha256,
        terminal_file_sha256=expected_terminal_file_sha256,
        terminal_receipt_sha256=_sha256(payload),
        image_digest_sha256=package.artifacts.image_digest_sha256,
        source_tree_root_sha256=source_root,
        _issuer=_CAPACITY_GATE_ISSUER,
    )
    result.validate_for(package, authorization, static)
    return result


def issue_m03r_v16_training_activation_from_gates(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    static: M03RV16StaticGateQualification,
    capacity: M03RV16CapacityGateQualification,
) -> M03RV16TrainingActivation:
    """Issue training authority only after matching static and capacity gates."""

    capacity.validate_for(package, authorization, static)
    return _issue_m03r_v16_training_activation_from_gates(
        package=package,
        authorization=authorization,
        static=static,
        capacity=capacity,
    )


def bind_m03r_v16_admitted_launch_authority(
    *,
    rendered: M03RV16RenderedSuspendedJob,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    admission: M03RV16AdmittedJobAuthority,
    admission_file_sha256: str,
    source_tree_root_sha256: str,
    storage_evidence: M03RV16StorageSemanticsEvidence,
    storage_evidence_file_sha256: str,
    authority_root: str | Path,
    observer_root: str | Path,
) -> M03RV16RenderedSuspendedJob:
    """Bind an admitted suspended Job to a one-shot scientific launch."""

    rendered.validate()
    if rendered.mode in {"static", "storage"} or rendered.launch_authority is not None:
        raise M03RV16KubernetesError(
            "only an unbound H100 Job can receive admission evidence"
        )
    admission.validate_for(
        package,
        authorization,
        expected_phase=rendered.mode,
        expected_job_contract_sha256=rendered.job_contract_sha256,
        expected_pod_contract_sha256=rendered.pod_contract_sha256,
    )
    if (
        admission.run_id
        != rendered.manifest.get("metadata", {})
        .get("labels", {})
        .get("rl-quant/run-id")
    ):
        raise M03RV16KubernetesError("V16 admitted Job run identity drifted")
    _digest("admission_file_sha256", admission_file_sha256)
    _digest("source_tree_root_sha256", source_tree_root_sha256)
    _digest("storage_evidence_file_sha256", storage_evidence_file_sha256)
    storage_evidence.validate_for(authority_root, observer_root)
    if admitted_job_authority_file_sha256(admission) != admission_file_sha256:
        raise M03RV16KubernetesError("V16 admitted Job file bytes drifted")
    prerequisite = {
        "capacity": rendered.static_gate_receipt_sha256,
        "training": rendered.training_activation_receipt_sha256,
        "qualification-preflight": (
            rendered.qualification_activation_receipt_sha256
        ),
        "qualification": rendered.qualification_outer_access_receipt_sha256,
    }[rendered.mode]
    if prerequisite is None:
        raise M03RV16KubernetesError("V16 launch prerequisite is absent")
    launch = _issue_m03r_v16_phase_launch_authority(
        package=package,
        authorization=authorization,
        phase=cast(
            Literal[
                "capacity", "training", "qualification-preflight", "qualification"
            ],
            rendered.mode,
        ),
        prerequisite_authority_receipt_sha256=prerequisite,
        job_contract_sha256=rendered.job_contract_sha256,
        pod_contract_sha256=rendered.pod_contract_sha256,
        run_id=admission.run_id,
        source_tree_root_sha256=source_tree_root_sha256,
        admission=admission,
        admission_file_sha256=admission_file_sha256,
        storage_semantics_file_sha256=storage_evidence_file_sha256,
        storage_semantics_receipt_sha256=storage_evidence.receipt_sha256,
        storage_authority_root_sha256=(
            storage_evidence.authority_root_sha256
        ),
        storage_observer_root_sha256=storage_evidence.observer_root_sha256,
    )
    launch_file_sha256 = phase_launch_authority_file_sha256(launch)
    manifest = copy.deepcopy(rendered.manifest)
    for annotation_rows in (
        manifest["metadata"]["annotations"],
        manifest["spec"]["template"]["metadata"]["annotations"],
    ):
        annotation_rows[_ADMISSION_RECEIPT_ANNOTATION] = admission.receipt_sha256
        annotation_rows[_ADMISSION_FILE_ANNOTATION] = admission_file_sha256
        annotation_rows[_DRY_RUN_FILE_ANNOTATION] = (
            admission.server_side_dry_run_file_sha256
        )
        annotation_rows[_ADMITTED_MANIFEST_FILE_ANNOTATION] = (
            admission.admitted_manifest_file_sha256
        )
        annotation_rows[_LAUNCH_RECEIPT_ANNOTATION] = launch.receipt_sha256
        annotation_rows[_LAUNCH_FILE_ANNOTATION] = launch_file_sha256
        annotation_rows[_STORAGE_FILE_ANNOTATION] = (
            launch.storage_semantics_file_sha256
        )
        annotation_rows[_STORAGE_RECEIPT_ANNOTATION] = (
            launch.storage_semantics_receipt_sha256
        )
    value = replace(
        rendered,
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_sha256(manifest["spec"]["template"]["spec"]),
        launch_authority=launch,
        launch_authority_file_sha256=launch_file_sha256,
        admitted_job_authority=admission,
        admitted_job_authority_file_sha256=admission_file_sha256,
    )
    value.validate()
    return value


def write_m03r_v16_rendered_launch_authority(
    path: str | Path,
    rendered: M03RV16RenderedSuspendedJob,
) -> str:
    """Materialize the exact launch authority already bound to a Job."""

    rendered.validate()
    if rendered.launch_authority is None or rendered.launch_authority_file_sha256 is None:
        raise M03RV16KubernetesError("V16 static Job has no H100 launch authority")
    observed = write_m03r_v16_phase_launch_authority(
        path, rendered.launch_authority
    )
    if observed != rendered.launch_authority_file_sha256:
        raise M03RV16KubernetesError("V16 launch authority bytes drifted")
    return observed


def _base_args(
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
) -> list[str]:
    return [
        "--package-plan",
        f"{package.plan_directory}/package-plan.json",
        "--package-plan-file-sha256",
        package_plan_file_sha256,
        "--execution-authorization",
        f"{package.plan_directory}/execution-authorization.json",
        "--execution-authorization-file-sha256",
        authorization_file_sha256,
    ]


def _render(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    template: M03RV7KubernetesTemplateConfig,
    mode: Literal[
        "static",
        "storage",
        "capacity",
        "training",
        "qualification-preflight",
        "qualification",
    ],
    static: M03RV16StaticGateQualification | None,
    capacity: M03RV16CapacityGateQualification | None,
    training_activation: M03RV16TrainingActivation | None,
    training_activation_file_sha256: str | None,
    qualification_activation: M03RV16QualificationActivation | None,
    qualification_activation_file_sha256: str | None,
    qualification_outer_access: (
        M03RV16QualificationOuterAccessAuthority | None
    ),
    qualification_outer_access_file_sha256: str | None,
) -> M03RV16RenderedSuspendedJob:
    package.validate()
    authorization.validate(package)
    _digest("package_plan_file_sha256", package_plan_file_sha256)
    _digest("authorization_file_sha256", authorization_file_sha256)
    if authorization.package_plan_file_sha256 != package_plan_file_sha256:
        raise M03RV16KubernetesError(
            "V16 authorization and package-plan file disagree"
        )
    if mode == "static":
        if any(
            value is not None
            for value in (
                static,
                capacity,
                training_activation,
                qualification_activation,
                qualification_outer_access,
            )
        ):
            raise M03RV16KubernetesError("V16 static Job precedes all gates")
    else:
        if static is None:
            raise M03RV16KubernetesError("V16 Job requires the static gate")
        static.validate_for(package, authorization)
        if mode == "storage":
            if any(
                value is not None
                for value in (
                    capacity,
                    training_activation,
                    qualification_activation,
                    qualification_outer_access,
                )
            ):
                raise M03RV16KubernetesError(
                    "V16 storage qualification precedes capacity"
                )
        elif mode == "capacity":
            if any(
                value is not None
                for value in (
                    capacity,
                    training_activation,
                    qualification_activation,
                    qualification_outer_access,
                )
            ):
                raise M03RV16KubernetesError("capacity cannot prequalify itself")
        elif capacity is None:
            raise M03RV16KubernetesError(
                "V16 scientific Job requires exact capacity evidence"
            )
        else:
            capacity.validate_for(package, authorization, static)
            if mode == "training":
                if (
                    training_activation is None
                    or qualification_activation is not None
                    or qualification_outer_access is not None
                ):
                    raise M03RV16KubernetesError(
                        "V16 training Job requires only training activation"
                    )
                training_activation.validate_for(package, authorization)
                if (
                    training_activation.static_gate_receipt_sha256
                    != static.receipt_sha256
                    or training_activation.static_result_file_sha256
                    != static.result_file_sha256
                    or training_activation.capacity_gate_receipt_sha256
                    != capacity.receipt_sha256
                    or training_activation.capacity_terminal_file_sha256
                    != capacity.terminal_file_sha256
                    or training_activation.source_tree_root_sha256
                    != capacity.source_tree_root_sha256
                ):
                    raise M03RV16KubernetesError(
                        "V16 training activation predecessor evidence drifted"
                    )
                _digest(
                    "training_activation_file_sha256",
                    str(training_activation_file_sha256),
                )
            elif (
                qualification_activation is None
                or training_activation is not None
            ):
                raise M03RV16KubernetesError(
                    "V16 qualification Job requires only qualification activation"
                )
            else:
                qualification_activation.validate_for(package, authorization)
                if (
                    qualification_activation.source_tree_root_sha256
                    != capacity.source_tree_root_sha256
                ):
                    raise M03RV16KubernetesError(
                        "V16 qualification activation source tree drifted"
                    )
                _digest(
                    "qualification_activation_file_sha256",
                    str(qualification_activation_file_sha256),
                )
                if mode == "qualification-preflight":
                    if qualification_outer_access is not None:
                        raise M03RV16KubernetesError(
                            "V16 preflight cannot receive outer access"
                        )
                else:
                    if qualification_outer_access is None:
                        raise M03RV16KubernetesError(
                            "V16 qualification requires CPU outer access"
                        )
                    qualification_outer_access.validate_for(
                        package, authorization, qualification_activation
                    )
                    _digest(
                        "qualification_outer_access_file_sha256",
                        str(qualification_outer_access_file_sha256),
                    )
    completions, parallelism, gpus = {
        "static": (1, 1, 0),
        "storage": (1, 1, 0),
        "capacity": (1, 1, 2),
        "training": (3, 3, 2),
        "qualification-preflight": (3, 3, 0),
        "qualification": (3, 3, 2),
    }[mode]
    if mode == "static":
        module = _STATIC_MODULE
        args = [
            "-m",
            module,
            *_base_args(
                package,
                authorization,
                package_plan_file_sha256,
                authorization_file_sha256,
            ),
            "--output-root",
            "/mnt/output",
        ]
    elif mode == "storage":
        module = _STORAGE_MODULE
        args = [
            "-I",
            "-B",
            "-c",
            (
                "import pathlib,sys;"
                "sys.dont_write_bytecode=True;"
                "root=pathlib.Path('/mnt/package/source/src').resolve();"
                "sys.path.insert(0,str(root));"
                "from rl_quant.workflows import "
                "top2000_m03r_v16_storage_gate as gate;"
                "resolved=pathlib.Path(gate.__file__).resolve();"
                "resolved.relative_to(root);"
                "raise SystemExit(gate.main())"
            ),
            "--authority-root",
            "/mnt/authority",
            "--observer-root",
            "/mnt/authority-observer",
            "--output",
            "/mnt/authority/storage-semantics.json",
            "--terminal",
            "/mnt/authority/storage-gate-terminal.json",
        ]
    elif mode == "qualification-preflight":
        module = _WORKER_MODULE
        args = [
            "-m",
            module,
            *_base_args(
                package,
                authorization,
                package_plan_file_sha256,
                authorization_file_sha256,
            ),
            "--qualification-preflight-only",
            "--qualification-activation",
            "/mnt/authority/qualification-activation.json",
            "--qualification-activation-file-sha256",
            str(qualification_activation_file_sha256),
            "--training-root",
            "/mnt/training",
            "--training-panel",
            "/mnt/authority/training-panel-decision.json",
        ]
    else:
        module = _WORKER_MODULE
        args = [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--max-restarts=0",
            "--nproc-per-node=2",
            "-m",
            module,
            *_base_args(
                package,
                authorization,
                package_plan_file_sha256,
                authorization_file_sha256,
            ),
        ]
        if mode == "capacity":
            static_value = cast(M03RV16StaticGateQualification, static)
            args.extend(
                (
                    "--completion-index",
                    "0",
                    "--capacity-only",
                    "--capacity-output-root",
                    "/mnt/output/capacity-sentinel",
                    "--static-result",
                    "/mnt/authority/static-result.json",
                    "--static-result-file-sha256",
                    str(static_value.result_file_sha256),
                    "--predecessor-authority-receipt-sha256",
                    str(static_value.receipt_sha256),
                )
            )
        elif mode == "training":
            training_value = cast(M03RV16TrainingActivation, training_activation)
            args.extend(
                (
                    "--training-activation",
                    "/mnt/authority/training-activation.json",
                    "--training-activation-file-sha256",
                    str(training_activation_file_sha256),
                    "--static-result",
                    "/mnt/authority/static-result.json",
                    "--static-result-file-sha256",
                    str(training_value.static_result_file_sha256),
                    "--capacity-terminal",
                    "/mnt/authority/two-h100-capacity-terminal.json",
                    "--capacity-terminal-file-sha256",
                    str(training_value.capacity_terminal_file_sha256),
                )
            )
        elif mode == "qualification":
            args.extend(
                (
                    "--qualification-only",
                    "--qualification-activation",
                    "/mnt/authority/qualification-activation.json",
                    "--qualification-activation-file-sha256",
                    str(qualification_activation_file_sha256),
                    "--training-root",
                    "/mnt/training",
                    "--training-panel",
                    "/mnt/authority/training-panel-decision.json",
                    "--qualification-outer-access-authority",
                    "/mnt/authority/qualification-outer-access.json",
                    "--qualification-outer-access-authority-file-sha256",
                    str(qualification_outer_access_file_sha256),
                    "--qualification-outer-access-authority-receipt-sha256",
                    str(
                        cast(
                            M03RV16QualificationOuterAccessAuthority,
                            qualification_outer_access,
                        ).receipt_sha256
                    ),
                    "--qualification-preflight-root",
                    "/mnt/preflight",
                )
            )
    if mode not in {"static", "storage"}:
        args.extend(
            (
                "--rendered-manifest-sha256",
                "$(M03R_V16_JOB_CONTRACT_SHA256)",
                "--pod-template-sha256",
                "$(M03R_V16_POD_CONTRACT_SHA256)",
                "--launch-authority",
                "/mnt/authority/$(M03R_V16_PHASE)-launch.json",
                "--launch-authority-file-sha256",
                "$(M03R_V16_LAUNCH_FILE_SHA256)",
                "--launch-authority-receipt-sha256",
                "$(M03R_V16_LAUNCH_RECEIPT_SHA256)",
                "--admitted-job-authority",
                "/mnt/authority/$(M03R_V16_PHASE)-admission.json",
                "--admitted-job-authority-file-sha256",
                "$(M03R_V16_ADMISSION_FILE_SHA256)",
                "--admitted-job-authority-receipt-sha256",
                "$(M03R_V16_ADMISSION_RECEIPT_SHA256)",
                "--server-side-dry-run-result",
                "/mnt/authority/$(M03R_V16_PHASE)-dry-run.json",
                "--server-side-dry-run-result-file-sha256",
                "$(M03R_V16_DRY_RUN_FILE_SHA256)",
                "--admitted-manifest-result",
                "/mnt/authority/$(M03R_V16_PHASE)-admitted-manifest.json",
                "--admitted-manifest-result-file-sha256",
                "$(M03R_V16_ADMITTED_MANIFEST_FILE_SHA256)",
                "--pod-runtime-attestation",
                "/mnt/authority/$(M03R_V16_POD_ATTESTATION_PATH)",
                "--pod-runtime-attestation-file-sha256",
                "$(M03R_V16_POD_ATTESTATION_FILE_SHA256)",
                "--pod-runtime-attestation-receipt-sha256",
                "$(M03R_V16_POD_ATTESTATION_RECEIPT_SHA256)",
                "--storage-semantics",
                "/mnt/authority/storage-semantics.json",
                "--storage-semantics-file-sha256",
                "$(M03R_V16_STORAGE_FILE_SHA256)",
                "--storage-semantics-receipt-sha256",
                "$(M03R_V16_STORAGE_RECEIPT_SHA256)",
                "--authority-observer-root",
                "/mnt/authority-observer",
            )
        )
    environment: list[dict[str, Any]] = [
        {
            "name": "JOB_COMPLETION_INDEX",
            "value": "0",
        }
        if mode in {"static", "capacity"}
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
        },
        {"name": "PYTHONPATH", "value": package.source_pythonpath},
        {"name": "PYTHONNOUSERSITE", "value": "1"},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONHASHSEED", "value": "0"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]
    if mode in {"static", "storage"}:
        environment.insert(0, {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"})
    if mode not in {"static", "storage"}:
        environment.extend(
            (
                {"name": "NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
                {"name": "TORCH_NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
                {"name": "CUBLAS_WORKSPACE_CONFIG", "value": ":4096:8"},
                {"name": "OMP_NUM_THREADS", "value": "8"},
                {"name": "MKL_NUM_THREADS", "value": "8"},
                {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
                {"name": "TORCHINDUCTOR_CACHE_DIR", "value": "/tmp/torchinductor"},
                {"name": "TRITON_CACHE_DIR", "value": "/tmp/triton"},
                {
                    "name": "M03R_V16_JOB_CONTRACT_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _JOB_CONTRACT_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_POD_CONTRACT_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _POD_CONTRACT_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {"name": "M03R_V16_PHASE", "value": mode},
                {
                    "name": "M03R_V16_LAUNCH_FILE_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _LAUNCH_FILE_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_LAUNCH_RECEIPT_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _LAUNCH_RECEIPT_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_ADMISSION_FILE_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _ADMISSION_FILE_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_ADMISSION_RECEIPT_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _ADMISSION_RECEIPT_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_DRY_RUN_FILE_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _DRY_RUN_FILE_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_ADMITTED_MANIFEST_FILE_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _ADMITTED_MANIFEST_FILE_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_CURRENT_POD_UID",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": "metadata.uid",
                        }
                    },
                },
                {
                    "name": "M03R_V16_CURRENT_POD_NAME",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": "metadata.name",
                        }
                    },
                },
                {
                    "name": "M03R_V16_CURRENT_NODE_NAME",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": "spec.nodeName",
                        }
                    },
                },
                {
                    "name": "M03R_V16_POD_ATTESTATION_PATH",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _POD_ATTESTATION_PATH_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_POD_ATTESTATION_FILE_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _POD_ATTESTATION_FILE_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_POD_ATTESTATION_RECEIPT_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _POD_ATTESTATION_RECEIPT_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_STORAGE_FILE_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _STORAGE_FILE_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
                {
                    "name": "M03R_V16_STORAGE_RECEIPT_SHA256",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": (
                                "metadata.annotations['"
                                + _STORAGE_RECEIPT_ANNOTATION
                                + "']"
                            ),
                        }
                    },
                },
            )
        )
    phase = {
        "static": "v16-static",
        "storage": "v16-storage",
        "capacity": "v16-capacity",
        "training": "v16-training",
        "qualification-preflight": "v16-qualification-preflight",
        "qualification": "v16-qualification",
    }[mode]
    labels = {
        "app.kubernetes.io/name": "quanttrade-m03r-v16",
        "rl-quant/run-id": template.run_id,
        "rl-quant/stage": phase,
    }
    annotations = {
        "rl-quant/package-plan-sha256": package.package_plan_sha256,
        "rl-quant/package-plan-file-sha256": package_plan_file_sha256,
        "rl-quant/execution-authorization-sha256": authorization.receipt_sha256,
        "rl-quant/execution-authorization-file-sha256": (
            authorization_file_sha256
        ),
        "rl-quant/source-manifest-sha256": package.artifacts.source_manifest_sha256,
        "rl-quant/structural-slab-sha256": (
            package.artifacts.structural_slab_receipt_sha256
        ),
        "rl-quant/static-gate-sha256": (
            "not-yet-created" if static is None else static.receipt_sha256
        ),
        "rl-quant/capacity-gate-sha256": (
            "not-yet-created" if capacity is None else capacity.receipt_sha256
        ),
        "rl-quant/training-activation-sha256": (
            "not-issued"
            if training_activation is None
            else training_activation.receipt_sha256
        ),
        "rl-quant/qualification-activation-sha256": (
            "not-issued"
            if qualification_activation is None
            else qualification_activation.receipt_sha256
        ),
        "rl-quant/economic-training-authorized": "false",
        "rl-quant/reinforcement-learning-authorized": "false",
        "rl-quant/outer-2026-authorized": "false",
    }
    resources = {
        "requests": {
            "cpu": (
                "1"
                if mode in {"static", "storage"}
                else "12"
                if mode == "qualification-preflight"
                else template.cpu_request
            ),
            "memory": (
                "4Gi"
                if mode in {"static", "storage"}
                else "64Gi"
                if mode == "qualification-preflight"
                else template.memory_request
            ),
            "ephemeral-storage": (
                "1Gi"
                if mode in {"static", "storage"}
                else "8Gi"
                if mode == "qualification-preflight"
                else template.ephemeral_storage_request
            ),
        },
        "limits": {
            "cpu": (
                "1"
                if mode in {"static", "storage"}
                else "16"
                if mode == "qualification-preflight"
                else template.cpu_limit
            ),
            "memory": (
                "4Gi"
                if mode in {"static", "storage"}
                else "128Gi"
                if mode == "qualification-preflight"
                else template.memory_limit
            ),
            "ephemeral-storage": (
                "4Gi"
                if mode in {"static", "storage"}
                else "16Gi"
                if mode == "qualification-preflight"
                else template.ephemeral_storage_limit
            ),
        },
    }
    resources["requests"]["nvidia.com/gpu"] = str(gpus)
    resources["limits"]["nvidia.com/gpu"] = str(gpus)
    pod: dict[str, Any] = {
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
                "name": (
                    "validator"
                    if mode == "static"
                    else "storage-gate"
                    if mode == "storage"
                    else "trainer"
                ),
                "image": package.artifacts.image_reference,
                "imagePullPolicy": "IfNotPresent",
                "command": [_PYTHON],
                "args": args,
                "env": environment,
                "resources": resources,
                "terminationMessagePath": M03R_TOP2000_TERMINATION_MESSAGE_PATH,
                "terminationMessagePolicy": M03R_TOP2000_TERMINATION_MESSAGE_POLICY,
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
                            f"{template.pvc_training_subpath.rstrip('/')}/packages/"
                            f"{template.run_id}"
                        ),
                        "readOnly": True,
                    },
                    {
                        "name": "research-data",
                        "mountPath": template.output_mount_path,
                        "subPath": (
                            f"{template.pvc_training_subpath.rstrip('/')}/runs/"
                            f"{template.run_id}/phases/{phase}"
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
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": (
                        "1Gi"
                        if mode in {"static", "storage", "qualification-preflight"}
                        else "32Gi"
                    ),
                },
            },
            {"name": "attestation-status", "emptyDir": {}},
            {
                "name": "podinfo",
                "downwardAPI": {
                    "items": [
                        {
                            "path": "pod-runtime-attestation-path",
                            "fieldRef": {
                                "fieldPath": (
                                    "metadata.annotations['"
                                    + _POD_ATTESTATION_PATH_ANNOTATION
                                    + "']"
                                )
                            },
                        },
                        {
                            "path": "pod-runtime-attestation-file-sha256",
                            "fieldRef": {
                                "fieldPath": (
                                    "metadata.annotations['"
                                    + _POD_ATTESTATION_FILE_ANNOTATION
                                    + "']"
                                )
                            },
                        },
                        {
                            "path": "pod-runtime-attestation-receipt-sha256",
                            "fieldRef": {
                                "fieldPath": (
                                    "metadata.annotations['"
                                    + _POD_ATTESTATION_RECEIPT_ANNOTATION
                                    + "']"
                                )
                            },
                        },
                    ]
                },
            },
        ],
    }
    if mode not in {"static", "storage"}:
        pod["containers"][0]["args"].extend(
            (
                "--pod-runtime-attestation-marker",
                "/var/run/m03r-v16-attestation/validated.json",
            )
        )
        pod["containers"][0]["volumeMounts"].append(
            {
                "name": "attestation-status",
                "mountPath": "/var/run/m03r-v16-attestation",
                "readOnly": True,
            }
        )
    if mode not in {"static", "storage"}:
        init_environment_names = {
            "JOB_COMPLETION_INDEX",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONHASHSEED",
            "PYTHONUNBUFFERED",
            "M03R_V16_JOB_CONTRACT_SHA256",
            "M03R_V16_POD_CONTRACT_SHA256",
            "M03R_V16_PHASE",
            "M03R_V16_LAUNCH_FILE_SHA256",
            "M03R_V16_LAUNCH_RECEIPT_SHA256",
            "M03R_V16_ADMISSION_FILE_SHA256",
            "M03R_V16_ADMISSION_RECEIPT_SHA256",
            "M03R_V16_CURRENT_POD_UID",
            "M03R_V16_CURRENT_POD_NAME",
            "M03R_V16_CURRENT_NODE_NAME",
            "M03R_V16_STORAGE_FILE_SHA256",
            "M03R_V16_STORAGE_RECEIPT_SHA256",
        }
        init_environment = [
            row for row in environment if row["name"] in init_environment_names
        ]
        predecessor_receipt = (
            cast(M03RV16StaticGateQualification, static).receipt_sha256
            if mode == "capacity"
            else cast(M03RV16TrainingActivation, training_activation).receipt_sha256
            if mode == "training"
            else cast(
                M03RV16QualificationActivation, qualification_activation
            ).receipt_sha256
        )
        pod["initContainers"] = [
            {
                "name": "runtime-attestation-gate",
                "image": package.artifacts.image_reference,
                "imagePullPolicy": "IfNotPresent",
                "command": [_PYTHON],
                "args": [
                    "-I",
                    "-B",
                    "-c",
                    _INIT_GATE_BOOTSTRAP,
                    *_base_args(
                        package,
                        authorization,
                        package_plan_file_sha256,
                        authorization_file_sha256,
                    ),
                    "--phase",
                    mode,
                    "--predecessor-authority-receipt-sha256",
                    predecessor_receipt,
                    "--job-contract-sha256",
                    "$(M03R_V16_JOB_CONTRACT_SHA256)",
                    "--pod-contract-sha256",
                    "$(M03R_V16_POD_CONTRACT_SHA256)",
                    "--launch-authority",
                    "/mnt/authority/$(M03R_V16_PHASE)-launch.json",
                    "--launch-authority-file-sha256",
                    "$(M03R_V16_LAUNCH_FILE_SHA256)",
                    "--launch-authority-receipt-sha256",
                    "$(M03R_V16_LAUNCH_RECEIPT_SHA256)",
                    "--admitted-job-authority",
                    "/mnt/authority/$(M03R_V16_PHASE)-admission.json",
                    "--admitted-job-authority-file-sha256",
                    "$(M03R_V16_ADMISSION_FILE_SHA256)",
                    "--admitted-job-authority-receipt-sha256",
                    "$(M03R_V16_ADMISSION_RECEIPT_SHA256)",
                    "--server-side-dry-run-result",
                    "/mnt/authority/$(M03R_V16_PHASE)-dry-run.json",
                    "--admitted-manifest-result",
                    "/mnt/authority/$(M03R_V16_PHASE)-admitted-manifest.json",
                    "--completion-index",
                    "$(JOB_COMPLETION_INDEX)",
                    "--downward-root",
                    "/etc/podinfo",
                    "--authority-root",
                    "/mnt/authority",
                    "--authority-observer-root",
                    "/mnt/authority-observer",
                    "--storage-semantics",
                    "/mnt/authority/storage-semantics.json",
                    "--storage-semantics-file-sha256",
                    "$(M03R_V16_STORAGE_FILE_SHA256)",
                    "--storage-semantics-receipt-sha256",
                    "$(M03R_V16_STORAGE_RECEIPT_SHA256)",
                    "--marker",
                    "/var/run/m03r-v16-attestation/validated.json",
                    "--package-source-root",
                    package.source_pythonpath,
                ],
                "env": init_environment,
                "resources": {
                    "requests": {"cpu": "50m", "memory": "512Mi"},
                    "limits": {"cpu": "250m", "memory": "1Gi"},
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
                            f"{template.pvc_training_subpath.rstrip('/')}/packages/"
                            f"{template.run_id}"
                        ),
                        "readOnly": True,
                    },
                    {
                        "name": "research-data",
                        "mountPath": "/mnt/authority",
                        "subPath": (
                            f"{template.pvc_training_subpath.rstrip('/')}/runs/"
                            f"{template.run_id}/authorities"
                        ),
                        "readOnly": True,
                    },
                    {
                        "name": "research-data",
                        "mountPath": "/mnt/authority-observer",
                        "subPath": (
                            f"{template.pvc_training_subpath.rstrip('/')}/runs/"
                            f"{template.run_id}/authorities"
                        ),
                        "readOnly": True,
                    },
                    {
                        "name": "podinfo",
                        "mountPath": "/etc/podinfo",
                        "readOnly": True,
                    },
                    {
                        "name": "attestation-status",
                        "mountPath": "/var/run/m03r-v16-attestation",
                    },
                ],
            }
        ]
    if mode in {"qualification-preflight", "qualification"}:
        pod["containers"][0]["volumeMounts"].append(
            {
                "name": "research-data",
                "mountPath": "/mnt/training",
                "subPath": (
                    f"{template.pvc_training_subpath.rstrip('/')}/runs/"
                    f"{template.run_id}/phases/v16-training"
                ),
                "readOnly": True,
            }
        )
    if mode == "qualification":
        pod["containers"][0]["volumeMounts"].append(
            {
                "name": "research-data",
                "mountPath": "/mnt/preflight",
                "subPath": (
                    f"{template.pvc_training_subpath.rstrip('/')}/runs/"
                    f"{template.run_id}/phases/v16-qualification-preflight"
                ),
                "readOnly": True,
            }
        )
    if mode in {
        "storage",
        "capacity",
        "training",
        "qualification-preflight",
        "qualification",
    }:
        pod["containers"][0]["volumeMounts"].append(
            {
                "name": "research-data",
                "mountPath": "/mnt/authority",
                "subPath": (
                    f"{template.pvc_training_subpath.rstrip('/')}/runs/"
                    f"{template.run_id}/authorities"
                ),
                "readOnly": mode != "storage",
            }
        )
        pod["containers"][0]["volumeMounts"].append(
            {
                "name": "research-data",
                "mountPath": "/mnt/authority-observer",
                "subPath": (
                    f"{template.pvc_training_subpath.rstrip('/')}/runs/"
                    f"{template.run_id}/authorities"
                ),
                "readOnly": True,
            }
        )
    if mode in {"capacity", "training", "qualification"}:
        pod.update(
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
            }
        )
    deadline = (
        M03R_TOP2000_PILOT_MAX_ACTIVE_DEADLINE_SECONDS
        if mode in {"static", "storage", "capacity", "qualification-preflight"}
        else min(
            template.active_deadline_seconds,
            M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS,
        )
    )
    manifest: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": template.job_name,
            "namespace": M03R_V16_NAMESPACE,
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
                "spec": pod,
            },
        },
    }
    job_contract_sha256 = _job_contract_sha256(manifest)
    pod_contract_sha256 = _pod_contract_sha256(manifest)
    for contract_annotations in (
        manifest["metadata"]["annotations"],
        manifest["spec"]["template"]["metadata"]["annotations"],
    ):
        contract_annotations[_JOB_CONTRACT_ANNOTATION] = job_contract_sha256
        contract_annotations[_POD_CONTRACT_ANNOTATION] = pod_contract_sha256
    rendered = M03RV16RenderedSuspendedJob(
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_sha256(pod),
        job_contract_sha256=job_contract_sha256,
        pod_contract_sha256=pod_contract_sha256,
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        package_plan_file_sha256=package_plan_file_sha256,
        execution_authorization_file_sha256=authorization_file_sha256,
        mode=mode,
        completions=completions,
        parallelism=parallelism,
        gpus_per_completion=gpus,
        static_gate_receipt_sha256=(
            None if static is None else static.receipt_sha256
        ),
        capacity_gate_receipt_sha256=(
            None if capacity is None else capacity.receipt_sha256
        ),
        training_activation_receipt_sha256=(
            None if training_activation is None else training_activation.receipt_sha256
        ),
        qualification_activation_receipt_sha256=(
            None
            if qualification_activation is None
            else qualification_activation.receipt_sha256
        ),
        qualification_outer_access_receipt_sha256=(
            None
            if qualification_outer_access is None
            else qualification_outer_access.receipt_sha256
        ),
    )
    rendered.validate()
    return rendered


def render_m03r_v16_suspended_static_job(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    template: M03RV7KubernetesTemplateConfig,
) -> M03RV16RenderedSuspendedJob:
    return _render(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=package_plan_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        template=template,
        mode="static",
        static=None,
        capacity=None,
        training_activation=None,
        training_activation_file_sha256=None,
        qualification_activation=None,
        qualification_activation_file_sha256=None,
        qualification_outer_access=None,
        qualification_outer_access_file_sha256=None,
    )


def render_m03r_v16_suspended_storage_job(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    template: M03RV7KubernetesTemplateConfig,
    static: M03RV16StaticGateQualification,
) -> M03RV16RenderedSuspendedJob:
    """Render the zero-GPU, cross-mount append-only storage gate."""

    return _render(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=package_plan_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        template=template,
        mode="storage",
        static=static,
        capacity=None,
        training_activation=None,
        training_activation_file_sha256=None,
        qualification_activation=None,
        qualification_activation_file_sha256=None,
        qualification_outer_access=None,
        qualification_outer_access_file_sha256=None,
    )


def render_m03r_v16_suspended_capacity_job(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    template: M03RV7KubernetesTemplateConfig,
    static: M03RV16StaticGateQualification,
) -> M03RV16RenderedSuspendedJob:
    return _render(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=package_plan_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        template=template,
        mode="capacity",
        static=static,
        capacity=None,
        training_activation=None,
        training_activation_file_sha256=None,
        qualification_activation=None,
        qualification_activation_file_sha256=None,
        qualification_outer_access=None,
        qualification_outer_access_file_sha256=None,
    )


def render_m03r_v16_suspended_training_job(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    template: M03RV7KubernetesTemplateConfig,
    static: M03RV16StaticGateQualification,
    capacity: M03RV16CapacityGateQualification,
    training_activation: M03RV16TrainingActivation,
    training_activation_file_sha256: str,
) -> M03RV16RenderedSuspendedJob:
    return _render(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=package_plan_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        template=template,
        mode="training",
        static=static,
        capacity=capacity,
        training_activation=training_activation,
        training_activation_file_sha256=training_activation_file_sha256,
        qualification_activation=None,
        qualification_activation_file_sha256=None,
        qualification_outer_access=None,
        qualification_outer_access_file_sha256=None,
    )


def render_m03r_v16_suspended_qualification_preflight_job(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    template: M03RV7KubernetesTemplateConfig,
    static: M03RV16StaticGateQualification,
    capacity: M03RV16CapacityGateQualification,
    qualification_activation: M03RV16QualificationActivation,
    qualification_activation_file_sha256: str,
) -> M03RV16RenderedSuspendedJob:
    return _render(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=package_plan_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        template=template,
        mode="qualification-preflight",
        static=static,
        capacity=capacity,
        training_activation=None,
        training_activation_file_sha256=None,
        qualification_activation=qualification_activation,
        qualification_activation_file_sha256=(
            qualification_activation_file_sha256
        ),
        qualification_outer_access=None,
        qualification_outer_access_file_sha256=None,
    )


def render_m03r_v16_suspended_qualification_job(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    template: M03RV7KubernetesTemplateConfig,
    static: M03RV16StaticGateQualification,
    capacity: M03RV16CapacityGateQualification,
    qualification_activation: M03RV16QualificationActivation,
    qualification_activation_file_sha256: str,
    qualification_outer_access: M03RV16QualificationOuterAccessAuthority,
    qualification_outer_access_file_sha256: str,
) -> M03RV16RenderedSuspendedJob:
    return _render(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=package_plan_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        template=template,
        mode="qualification",
        static=static,
        capacity=capacity,
        training_activation=None,
        training_activation_file_sha256=None,
        qualification_activation=qualification_activation,
        qualification_activation_file_sha256=qualification_activation_file_sha256,
        qualification_outer_access=qualification_outer_access,
        qualification_outer_access_file_sha256=(
            qualification_outer_access_file_sha256
        ),
    )


__all__ = [
    "M03R_V16_CAPACITY_GATE_SCHEMA",
    "M03R_V16_NAMESPACE",
    "M03R_V16_RENDERED_JOB_SCHEMA",
    "M03R_V16_STATIC_GATE_SCHEMA",
    "M03RV16CapacityGateQualification",
    "M03RV16KubernetesError",
    "M03RV16RenderedSuspendedJob",
    "M03RV16StaticGateQualification",
    "bind_m03r_v16_admitted_launch_authority",
    "issue_m03r_v16_training_activation_from_gates",
    "load_and_issue_m03r_v16_capacity_gate",
    "load_and_issue_m03r_v16_static_gate",
    "m03r_v16_pod_runtime_attestation_annotations",
    "render_m03r_v16_suspended_capacity_job",
    "render_m03r_v16_suspended_qualification_job",
    "render_m03r_v16_suspended_qualification_preflight_job",
    "render_m03r_v16_suspended_static_job",
    "render_m03r_v16_suspended_storage_job",
    "render_m03r_v16_suspended_training_job",
    "write_m03r_v16_rendered_launch_authority",
]
