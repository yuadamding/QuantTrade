"""Suspended, receipt-gated Seadragon Job manifests for M03R-v16."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
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
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
)
from rl_quant.training.top2000_m03r_v16_capacity import (
    M03R_V16_CAPACITY_TERMINAL_SCHEMA,
    M03RV16CapacityRankEvidence,
    M03RV16CapacityTerminal,
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
    "rl-quant.top2000-dev.m03r-v16-rendered-suspended-job-v2"
)
_PYTHON = "/opt/conda/envs/quanttrade/bin/python"
_WORKER_MODULE = "rl_quant.workflows.top2000_m03r_v16_predictive"
_STATIC_MODULE = "rl_quant.workflows.top2000_m03r_v16_static_validate"
_STATIC_GATE_ISSUER = object()
_CAPACITY_GATE_ISSUER = object()
_MAX_GATE_RESULT_BYTES = 64 * 1024**2


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
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    package_plan_file_sha256: str
    execution_authorization_file_sha256: str
    mode: Literal["static", "capacity", "predictive"]
    completions: int
    parallelism: int
    gpus_per_completion: int
    static_gate_receipt_sha256: str | None = None
    capacity_gate_receipt_sha256: str | None = None
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
            "capacity": (1, 1, 2),
            "predictive": (3, 3, 2),
        }[self.mode]
        gpu_resources_valid = (
            "nvidia.com/gpu" not in requests and "nvidia.com/gpu" not in limits
            if self.mode == "static"
            else requests.get("nvidia.com/gpu") == str(expected[2])
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
            ):
                raise M03RV16KubernetesError("V16 static Job is not GPU neutral")
        elif (
            pod.get("nodeSelector") != M03R_TOP2000_H100_POOL_NODE_SELECTOR
            or pod.get("priorityClassName") != M03R_TOP2000_PRIORITY_CLASS_NAME
            or pod.get("tolerations") != [M03R_TOP2000_MULTI_GPU_TOLERATION]
            or self.static_gate_receipt_sha256 is None
            or (
                self.mode == "predictive"
                and self.capacity_gate_receipt_sha256 is None
            )
        ):
            raise M03RV16KubernetesError("V16 H100 Job profile drifted")


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
    if len(raw) != before.st_size or file_sha256(path) != expected_file_sha256:
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
        or rendered.static_gate_receipt_sha256 != static.receipt_sha256
        or payload.get("schema") != M03R_V16_CAPACITY_TERMINAL_SCHEMA
        or payload.get("capacity_receipt_sha256") != capacity.receipt_sha256
        or payload.get("package_plan_sha256") != package.package_plan_sha256
        or payload.get("authorization_receipt_sha256")
        != authorization.receipt_sha256
        or payload.get("scientific_training_performed") is not False
        or payload.get("disposable_optimizer_update_executed") is not True
        or payload.get("disposable_train_validate_train_executed") is not True
        or payload.get("scientific_checkpoint_published") is not False
        or payload.get("development_only") is not True
    ):
        raise M03RV16KubernetesError("V16 capacity result authority drifted")
    source_root = str(payload.get("source_tree_root_sha256"))
    _digest("source_tree_root_sha256", source_root)
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
    mode: Literal["static", "capacity", "predictive"],
    static: M03RV16StaticGateQualification | None,
    capacity: M03RV16CapacityGateQualification | None,
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
        if static is not None or capacity is not None:
            raise M03RV16KubernetesError("V16 static Job precedes all gates")
    else:
        if static is None:
            raise M03RV16KubernetesError("V16 H100 Job requires the static gate")
        static.validate_for(package, authorization)
        if mode == "capacity":
            if capacity is not None:
                raise M03RV16KubernetesError("capacity cannot prequalify itself")
        elif capacity is None:
            raise M03RV16KubernetesError(
                "V16 predictive Job requires exact capacity evidence"
            )
        else:
            capacity.validate_for(package, authorization, static)
    completions, parallelism, gpus = {
        "static": (1, 1, 0),
        "capacity": (1, 1, 2),
        "predictive": (3, 3, 2),
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
            args.extend(
                (
                    "--completion-index",
                    "0",
                    "--capacity-only",
                    "--capacity-output-root",
                    "/mnt/output/capacity-sentinel",
                )
            )
    environment: list[dict[str, Any]] = [
        {
            "name": "JOB_COMPLETION_INDEX",
            "value": "0",
        }
        if mode != "predictive"
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
    if mode == "static":
        environment.insert(0, {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"})
    else:
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
            )
        )
    phase = {
        "static": "v16-static",
        "capacity": "v16-capacity",
        "predictive": "v16-predictive",
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
        "rl-quant/economic-training-authorized": "false",
        "rl-quant/reinforcement-learning-authorized": "false",
        "rl-quant/outer-2026-authorized": "false",
    }
    resources = {
        "requests": {
            "cpu": "1" if mode == "static" else template.cpu_request,
            "memory": "4Gi" if mode == "static" else template.memory_request,
            "ephemeral-storage": (
                "1Gi" if mode == "static" else template.ephemeral_storage_request
            ),
        },
        "limits": {
            "cpu": "1" if mode == "static" else template.cpu_limit,
            "memory": "4Gi" if mode == "static" else template.memory_limit,
            "ephemeral-storage": (
                "4Gi" if mode == "static" else template.ephemeral_storage_limit
            ),
        },
    }
    if mode != "static":
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
                "name": "validator" if mode == "static" else "trainer",
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
                    "sizeLimit": "1Gi" if mode == "static" else "32Gi",
                },
            },
        ],
    }
    if mode != "static":
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
        if mode in {"static", "capacity"}
        else min(
            template.active_deadline_seconds,
            M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS,
        )
    )
    manifest = {
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
    rendered = M03RV16RenderedSuspendedJob(
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        pod_template_sha256=_sha256(pod),
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
    )


def render_m03r_v16_suspended_predictive_job(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    package_plan_file_sha256: str,
    authorization_file_sha256: str,
    template: M03RV7KubernetesTemplateConfig,
    static: M03RV16StaticGateQualification,
    capacity: M03RV16CapacityGateQualification,
) -> M03RV16RenderedSuspendedJob:
    return _render(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=package_plan_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        template=template,
        mode="predictive",
        static=static,
        capacity=capacity,
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
    "load_and_issue_m03r_v16_capacity_gate",
    "load_and_issue_m03r_v16_static_gate",
    "render_m03r_v16_suspended_capacity_job",
    "render_m03r_v16_suspended_predictive_job",
    "render_m03r_v16_suspended_static_job",
]
