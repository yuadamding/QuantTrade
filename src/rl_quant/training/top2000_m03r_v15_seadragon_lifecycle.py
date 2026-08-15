"""Attach-only Seadragon lifecycle for the M03R-v15 predictive panel.

The module intentionally exposes no Kubernetes create, apply, or replace
operation.  It attaches to one already-bound suspended Job, performs one
preconditioned activation, captures exact terminal evidence, verifies both
predictive-worker outputs, and exact-cleans the bound UID.  Ambiguous
activation is retained for reviewed attachment and is never reported as
success.

All work is development-only, non-PHI research.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_PREDICTIVE_SPEC,
    M03R_V15_PROTOCOL_SHA256,
    M03R_V15_SELECTED_HORIZON_SESSIONS,
    M03R_V15_SETTING_IDS,
)
from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as common
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
from rl_quant.training.top2000_m03r_v15_kubernetes import (
    M03RV15TwoH100CapacityQualification,
)
from rl_quant.training.top2000_m03r_v15_package import (
    load_m03r_v15_execution_authorization,
    load_m03r_v15_package_plan,
)
from rl_quant.training.top2000_m03r_v15_validation_contract import (
    M03RV15CheckpointSelectionReceipt,
)
SEADRAGON_KUBECTL: Final = "/risapps/noarch/kubectl/1.28.4/bin/kubectl"
SEADRAGON_KUBECONFIG: Final = "/rsrch8/home/bcb/yding4/.kube/config"
SEADRAGON_QUANTTRADE_ROOT: Final = "/rsrch8/home/bcb/yding4/quant/training"
ATTACH_CONFIG_SCHEMA: Final = "rl-quant.top2000-dev.m03r-v15-seadragon-attach-config-v1"
COMPLETION_COVERAGE_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v15-predictive-coverage-v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CAPACITY_CONFIG_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v15-capacity-attach-config-v1"
)
M03R_V15_STARTUP_SCHEMA: Final = "rl-quant.top2000-dev.m03r-v15-two-h100-startup-v1"
M03R_V15_FOLD_TERMINAL_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v15-predictive-fold-terminal-v1"
)
M03R_V15_WORKER_TERMINAL_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v15-predictive-worker-terminal-v1"
)
M03R_V15_CAPACITY_TERMINAL_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v15-two-h100-capacity-terminal-v1"
)
M03R_V15_BOOTSTRAP_PLAN_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v15-bootstrap-plan-v1"
)
M03R_V15_QUALIFICATION_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v15-qualification-v1"
)
M03R_V15_EXECUTION_HORIZONS: Final = (M03R_V15_SELECTED_HORIZON_SESSIONS,)
M03R_V15_EXECUTION_HORIZON_KEYS: Final = frozenset(
    str(value) for value in M03R_V15_EXECUTION_HORIZONS
)


class M03RV15SeadragonLifecycleError(RuntimeError):
    """The exact v15 attachment or its evidence failed closed."""


class M03RV15ActivationAttachRequired(M03RV15SeadragonLifecycleError):
    """Activation outcome is ambiguous and exact state must be retained."""


@dataclass(frozen=True, slots=True)
class M03RV15CapacityAttachConfig:
    job_name: str
    run_id: str
    job_uid: str
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
    static_gate_path: str
    static_gate_file_sha256: str
    static_gate_receipt_sha256: str
    lifecycle_source_sha256: str
    output_root: str
    evidence_root: str
    host_python_path: str
    pythonpath: str
    kubectl_path: str = SEADRAGON_KUBECTL
    kubeconfig_path: str = SEADRAGON_KUBECONFIG
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    poll_interval_seconds: int = 10
    request_timeout_seconds: int = 30
    handshake_timeout_seconds: int = 180
    hard_wall_seconds: int = 1800
    log_limit_bytes: int = 1048576
    capacity_receipt_sha256: str = "not-yet-created"
    schema: str = CAPACITY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "binding_file_sha256",
            "activation_request_file_sha256",
            "package_plan_file_sha256",
            "package_plan_sha256",
            "execution_authorization_file_sha256",
            "execution_authorization_receipt_sha256",
            "source_archive_sha256",
            "static_gate_file_sha256",
            "static_gate_receipt_sha256",
            "lifecycle_source_sha256",
        ):
            _require_sha256(name, cast(str, getattr(self, name)))
        if (
            self.schema != CAPACITY_CONFIG_SCHEMA
            or not self.job_name
            or not self.run_id
            or not self.job_uid
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.kubectl_path != SEADRAGON_KUBECTL
            or self.kubeconfig_path != SEADRAGON_KUBECONFIG
            or self.poll_interval_seconds < 5
            or self.request_timeout_seconds < 5
            or self.handshake_timeout_seconds < 30
            or not 60 <= self.hard_wall_seconds <= 3600
            or self.hard_wall_seconds <= self.handshake_timeout_seconds
            or self.log_limit_bytes < 4096
            or self.capacity_receipt_sha256 != "not-yet-created"
        ):
            raise M03RV15SeadragonLifecycleError(
                "v15 capacity config drifted from the one-worker 2-H100 contract"
            )
        for name in (
            "binding_path",
            "activation_request_path",
            "package_plan_path",
            "execution_authorization_path",
            "static_gate_path",
            "output_root",
            "evidence_root",
            "pythonpath",
        ):
            _project_path(cast(str, getattr(self, name)), name)
        if not Path(self.host_python_path).is_absolute():
            raise M03RV15SeadragonLifecycleError("host_python_path must be absolute")


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
        raise M03RV15SeadragonLifecycleError(
            "v15 lifecycle evidence is not canonical-JSON safe"
        ) from exc


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _compact_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M03RV15SeadragonLifecycleError(
            "semantic receipt is not canonical-JSON safe"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise M03RV15SeadragonLifecycleError(f"{name} must be a lowercase SHA-256")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M03RV15SeadragonLifecycleError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _receipt_payload(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    receipt = value.get("receipt_sha256")
    if not isinstance(receipt, str):
        raise M03RV15SeadragonLifecycleError(f"{label} receipt hash is absent")
    _require_sha256(f"{label} receipt", receipt)
    unsigned = dict(value)
    del unsigned["receipt_sha256"]
    if receipt != _content_sha256(unsigned):
        raise M03RV15SeadragonLifecycleError(f"{label} receipt hash drifted")
    return unsigned


def _semantic_receipt_payload(
    value: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    receipt = value.get("receipt_sha256")
    if not isinstance(receipt, str):
        raise M03RV15SeadragonLifecycleError(f"{label} receipt hash is absent")
    _require_sha256(f"{label} receipt", receipt)
    unsigned = dict(value)
    del unsigned["receipt_sha256"]
    if receipt != _compact_sha256(unsigned):
        raise M03RV15SeadragonLifecycleError(f"{label} receipt hash drifted")
    return unsigned


def _project_path(path: str, label: str) -> Path:
    value = Path(path)
    root = Path(SEADRAGON_QUANTTRADE_ROOT)
    if not value.is_absolute():
        raise M03RV15SeadragonLifecycleError(f"{label} must be absolute")
    try:
        value.relative_to(root)
    except ValueError as exc:
        raise M03RV15SeadragonLifecycleError(
            f"{label} must stay under the approved QuantTrade root"
        ) from exc
    return value


@dataclass(frozen=True, slots=True)
class M03RV15ExpectedCompletion:
    completion_index: int
    setting_index: int
    setting_id: str
    worker_plan_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("worker_plan_sha256", self.worker_plan_sha256)
        if (
            self.completion_index not in range(2)
            or self.setting_index != self.completion_index
            or self.setting_id != M03R_V15_SETTING_IDS[self.setting_index]
        ):
            raise M03RV15SeadragonLifecycleError(
                "v15 completion identity must be the frozen direct 0..1 map"
            )


@dataclass(frozen=True, slots=True)
class M03RV15AttachSupervisorConfig:
    job_name: str
    run_id: str
    job_uid: str
    binding_path: str
    binding_file_sha256: str
    activation_request_path: str
    activation_request_file_sha256: str
    output_root: str
    evidence_root: str
    package_plan_path: str
    package_plan_file_sha256: str
    package_plan_sha256: str
    execution_authorization_path: str
    execution_authorization_file_sha256: str
    execution_authorization_receipt_sha256: str
    source_archive_sha256: str
    capacity_receipt_sha256: str
    lifecycle_source_sha256: str
    expected_completions: tuple[M03RV15ExpectedCompletion, ...]
    host_python_path: str
    pythonpath: str
    kubectl_path: str = SEADRAGON_KUBECTL
    kubeconfig_path: str = SEADRAGON_KUBECONFIG
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    completions: int = 2
    parallelism: int = 2
    gpus_per_worker: int = 2
    authorized_h100_cap: int = 16
    expected_seed: int = 17
    expected_fold_count: int = 6
    poll_interval_seconds: int = 30
    request_timeout_seconds: int = 30
    handshake_timeout_seconds: int = 180
    hard_wall_seconds: int = 218400
    log_limit_bytes: int = 1048576
    schema: str = ATTACH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "binding_file_sha256",
            "activation_request_file_sha256",
            "package_plan_sha256",
            "package_plan_file_sha256",
            "execution_authorization_file_sha256",
            "execution_authorization_receipt_sha256",
            "source_archive_sha256",
            "capacity_receipt_sha256",
            "lifecycle_source_sha256",
        ):
            _require_sha256(name, cast(str, getattr(self, name)))
        if (
            self.schema != ATTACH_CONFIG_SCHEMA
            or not self.job_name
            or not self.run_id
            or not self.job_uid
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.kubectl_path != SEADRAGON_KUBECTL
            or self.kubeconfig_path != SEADRAGON_KUBECONFIG
            or self.completions != 2
            or self.parallelism not in range(1, 3)
            or self.gpus_per_worker != 2
            or self.authorized_h100_cap != 16
            or self.parallelism * self.gpus_per_worker > 4
            or self.expected_seed != 17
            or self.expected_fold_count != 6
            or tuple(row.completion_index for row in self.expected_completions)
            != (0, 1)
            or tuple(row.setting_index for row in self.expected_completions)
            != (0, 1)
            or len({row.worker_plan_sha256 for row in self.expected_completions}) != 2
            or self.poll_interval_seconds < 5
            or self.request_timeout_seconds < 5
            or self.handshake_timeout_seconds < 30
            or self.hard_wall_seconds <= self.handshake_timeout_seconds
            or self.log_limit_bytes < 4096
        ):
            raise M03RV15SeadragonLifecycleError(
                "v15 attach config drifted from the two-setting 2-H100 contract"
            )
        for name in (
            "binding_path",
            "activation_request_path",
            "package_plan_path",
            "execution_authorization_path",
            "output_root",
            "evidence_root",
            "pythonpath",
        ):
            _project_path(cast(str, getattr(self, name)), name)
        if not Path(self.host_python_path).is_absolute():
            raise M03RV15SeadragonLifecycleError("host_python_path must be absolute")

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)


def _load_config(path: Path, expected_sha256: str) -> M03RV15AttachSupervisorConfig:
    _require_sha256("config file", expected_sha256)
    payload = _mapping(
        common._read_json_file(path, expected_sha256=expected_sha256),
        "v15 attach config",
    )
    values = dict(payload)
    values["expected_completions"] = tuple(
        M03RV15ExpectedCompletion(**dict(_mapping(row, "expected completion")))
        for row in cast(Sequence[Any], values.get("expected_completions", ()))
    )
    try:
        return M03RV15AttachSupervisorConfig(**values)
    except (TypeError, ValueError) as exc:
        raise M03RV15SeadragonLifecycleError("v15 attach config is invalid") from exc


def _load_capacity_config(
    path: Path, expected_sha256: str
) -> M03RV15CapacityAttachConfig:
    _require_sha256("capacity config file", expected_sha256)
    payload = _mapping(
        common._read_json_file(path, expected_sha256=expected_sha256),
        "v15 capacity attach config",
    )
    try:
        return M03RV15CapacityAttachConfig(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise M03RV15SeadragonLifecycleError(
            "v15 capacity attach config is invalid"
        ) from exc


def _load_package_authorization(
    config: M03RV15AttachSupervisorConfig | M03RV15CapacityAttachConfig,
) -> tuple[Any, Any]:
    package = load_m03r_v15_package_plan(
        config.package_plan_path,
        expected_file_sha256=config.package_plan_file_sha256,
    )
    if package.package_plan_sha256 != config.package_plan_sha256:
        raise M03RV15SeadragonLifecycleError("package-plan identity drifted")
    authorization = load_m03r_v15_execution_authorization(
        config.execution_authorization_path,
        expected_file_sha256=config.execution_authorization_file_sha256,
        package=package,
    )
    if authorization.receipt_sha256 != config.execution_authorization_receipt_sha256:
        raise M03RV15SeadragonLifecycleError("execution-authorization identity drifted")
    return package, authorization


def _job_artifact_identity(
    job: Mapping[str, Any],
    config: M03RV15AttachSupervisorConfig | M03RV15CapacityAttachConfig,
) -> None:
    common._job_artifact_identity(job, cast(common.AttachSupervisorConfig, config))
    metadata = _mapping(job.get("metadata"), "Job metadata")
    annotations = _mapping(metadata.get("annotations"), "Job annotations")
    spec = _mapping(job.get("spec"), "Job spec")
    template = _mapping(spec.get("template"), "Job Pod template")
    template_metadata = _mapping(template.get("metadata"), "Pod template metadata")
    template_annotations = _mapping(
        template_metadata.get("annotations"), "Pod template annotations"
    )
    key = "rl-quant/execution-authorization-sha256"
    expected = config.execution_authorization_receipt_sha256
    if annotations.get(key) != expected or template_annotations.get(key) != expected:
        raise M03RV15SeadragonLifecycleError(
            "live Job execution-authorization annotation drifted"
        )


def _host_output_path(config: M03RV15AttachSupervisorConfig, value: Any) -> Path:
    if not isinstance(value, str):
        raise M03RV15SeadragonLifecycleError("worker artifact path is absent")
    remote = PurePosixPath(value)
    prefix = PurePosixPath("/mnt/output")
    try:
        relative = remote.relative_to(prefix)
    except ValueError as exc:
        raise M03RV15SeadragonLifecycleError(
            "worker artifact path is outside /mnt/output"
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise M03RV15SeadragonLifecycleError("worker artifact path is unsafe")
    return Path(config.output_root).joinpath(*relative.parts)


def _validate_rank_runtime(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise M03RV15SeadragonLifecycleError("startup must contain two rank rows")
    rows = sorted(
        (_mapping(row, "rank runtime") for row in value),
        key=lambda row: int(cast(int, row.get("rank", -1))),
    )
    for rank, row in enumerate(rows):
        memory = row.get("device_total_memory")
        if (
            row.get("rank") != rank
            or row.get("local_rank") != rank
            or row.get("world_size") != 2
            or row.get("visible_device_count") != 2
            or row.get("device_name") != "NVIDIA H100 80GB HBM3"
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or not 79 * 1024**3 <= memory <= 81 * 1024**3
            or row.get("compute_capability") != [9, 0]
        ):
            raise M03RV15SeadragonLifecycleError(
                "startup rank did not prove the exact two-H100 contract"
            )


def _validate_capacity_output(
    config: M03RV15CapacityAttachConfig,
) -> tuple[str, str, Mapping[str, Any]]:
    package, authorization = _load_package_authorization(config)
    worker = package.panel.workers[0]
    root = Path(config.output_root) / "capacity-sentinel"
    startup_path = root / "two-h100-startup.json"
    terminal_path = root / "two-h100-capacity-terminal.json"
    startup_sha = _file_sha256(
        common._regular_no_symlink(startup_path, label="capacity startup")
    )
    terminal_sha = _file_sha256(
        common._regular_no_symlink(terminal_path, label="capacity terminal")
    )
    startup = _read_bound_json(startup_path, startup_sha, "capacity startup")
    terminal = _read_bound_json(terminal_path, terminal_sha, "capacity terminal")
    _validate_rank_runtime(startup.get("rank_runtime"))
    # Workflow terminal receipts use the package JSON file convention: the
    # receipt hash covers canonical JSON including its trailing newline.
    _receipt_payload(terminal, "capacity terminal")
    if (
        startup.get("schema") != M03R_V15_STARTUP_SCHEMA
        or startup.get("package_plan_sha256") != package.package_plan_sha256
        or startup.get("package_plan_file_sha256") != config.package_plan_file_sha256
        or startup.get("authorization_receipt_sha256") != authorization.receipt_sha256
        or startup.get("worker_plan_sha256") != worker.receipt_sha256
        or startup.get("setting_index") != 0
        or startup.get("setting_id") != M03R_V15_SETTING_IDS[0]
        or startup.get("mode") != "capacity"
        or startup.get("exact_h100_80gb_per_rank") is not True
        or startup.get("nccl_process_group_initialized") is not True
        or startup.get("restart_count") != 0
        or startup.get("development_only") is not True
        or startup.get("reportable") is not False
        or startup.get("promotion_eligible") is not False
        or terminal.get("schema") != M03R_V15_CAPACITY_TERMINAL_SCHEMA
        or terminal.get("package_plan_sha256") != package.package_plan_sha256
        or terminal.get("authorization_receipt_sha256") != authorization.receipt_sha256
        or terminal.get("worker_plan_sha256") != worker.receipt_sha256
        or terminal.get("startup_file_sha256") != startup_sha
        or terminal.get("setting_index") != 0
        or terminal.get("setting_id") != M03R_V15_SETTING_IDS[0]
        or terminal.get("world_size") != 2
        or terminal.get("gpus_per_worker") != 2
        or terminal.get("exact_h100_80gb_per_rank") is not True
        or terminal.get("nccl_process_group_initialized") is not True
        or terminal.get("initial_parameter_state_file_sha256")
        != package.artifacts.initial_parameter_state_file_sha256
        or terminal.get("initial_parameter_state_sha256")
        != package.artifacts.initial_parameter_state_sha256
        or terminal.get("initial_parameter_architecture_sha256")
        != package.artifacts.initial_parameter_architecture_sha256
        or terminal.get("packaged_initial_state_loaded") is not True
        or terminal.get("disposable_exact_shape_update_performed") is not True
        or terminal.get("disposable_rank_states_equal_after_update") is not True
        or terminal.get("disposable_qualification_projection_performed") is not True
        or terminal.get("scientific_checkpoint_published") is not False
        or not isinstance(terminal.get("peak_cuda_memory_by_rank"), list)
        or len(terminal.get("peak_cuda_memory_by_rank")) != 2
        or terminal.get("training_performed") is not False
        or terminal.get("economic_optimizer_updates") != 0
        or terminal.get("h100_capacity_evidence") is not True
        or terminal.get("outer_2026_accessed") is not False
        or terminal.get("development_only") is not True
        or terminal.get("reportable") is not False
        or terminal.get("promotion_eligible") is not False
    ):
        raise M03RV15SeadragonLifecycleError(
            "capacity output does not prove the exact two-H100 startup-only boundary"
        )
    return startup_sha, terminal_sha, terminal


def _read_bound_file(path: Path, expected_sha256: str, label: str) -> bytes:
    _require_sha256(label, expected_sha256)
    common._regular_no_symlink(path, label=label)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise M03RV15SeadragonLifecycleError(f"{label} file hash drifted")
    return content


def _read_bound_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    content = _read_bound_file(path, expected_sha256, label)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise M03RV15SeadragonLifecycleError(f"{label} is invalid JSON") from exc
    return _mapping(value, label)


def _validate_static_gate_lineage(
    value: Mapping[str, Any],
    *,
    expected_receipt_sha256: str,
    package_plan_sha256: str,
    authorization_receipt_sha256: str,
    source_archive_sha256: str,
    image_digest_sha256: str,
) -> None:
    """Bind the completed zero-GPU gate before any capacity activation."""

    expected_keys = {
        "actual_pod_proof_file_sha256",
        "cleanup_receipt_file_sha256",
        "created_binding_file_sha256",
        "development_only",
        "economic_training_authorized",
        "execution_authorization_receipt_sha256",
        "gpu_limits",
        "gpu_requests",
        "h100_capacity_evidence",
        "image_digest_sha256",
        "outer_2026_access_authorized",
        "package_plan_sha256",
        "passed",
        "promotion_eligible",
        "rendered_manifest_sha256",
        "reportable",
        "schema",
        "server_dry_run_file_sha256",
        "source_archive_sha256",
        "static_log_file_sha256",
        "terminal_evidence_file_sha256",
        "training_performed",
        "unmasked_visibility_claimed",
        "visibility_mask",
    }
    digest_fields = expected_keys - {
        "development_only",
        "economic_training_authorized",
        "gpu_limits",
        "gpu_requests",
        "h100_capacity_evidence",
        "outer_2026_access_authorized",
        "passed",
        "promotion_eligible",
        "reportable",
        "schema",
        "training_performed",
        "unmasked_visibility_claimed",
        "visibility_mask",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != "rl-quant.top2000-dev.m03r-v15-static-gate-v1"
        or value.get("package_plan_sha256") != package_plan_sha256
        or value.get("execution_authorization_receipt_sha256")
        != authorization_receipt_sha256
        or value.get("source_archive_sha256") != source_archive_sha256
        or value.get("image_digest_sha256") != image_digest_sha256
        or value.get("gpu_requests") != 0
        or value.get("gpu_limits") != 0
        or value.get("visibility_mask") != "none"
        or value.get("unmasked_visibility_claimed") is not False
        or value.get("training_performed") is not False
        or value.get("h100_capacity_evidence") is not False
        or value.get("economic_training_authorized") is not False
        or value.get("outer_2026_access_authorized") is not False
        or value.get("passed") is not True
        or value.get("development_only") is not True
        or value.get("reportable") is not False
        or value.get("promotion_eligible") is not False
        or _compact_sha256(value) != expected_receipt_sha256
    ):
        raise M03RV15SeadragonLifecycleError("static gate lineage drifted")
    for name in digest_fields:
        digest = value.get(name)
        if not isinstance(digest, str):
            raise M03RV15SeadragonLifecycleError(f"static gate {name} is not a digest")
        _require_sha256(f"static gate {name}", digest)


def _validate_qualification(
    value: Mapping[str, Any],
    *,
    setting_index: int,
    expected_fold_traces: list[str],
    expected_bootstrap_plan_sha256: str,
    expected_receipt_sha256: str,
) -> None:
    expected_keys = {
        "setting_index",
        "setting_id",
        "fold_trace_sha256",
        "bootstrap_plan_sha256",
        "mean_rank_ic",
        "positive_mean_ic_fold_count",
        "positive_median_ic_fold_count",
        "positive_date_fraction_fold_count",
        "positive_spread_fold_count",
        "annualized_gross_active_return",
        "annualized_net_active_return_10bp",
        "gross_active_lcb_by_block",
        "net_10bp_active_lcb_by_block",
        "spread_lcb_by_block",
        "break_even_category",
        "break_even_one_way_cost_basis_points",
        "median_signal_projection_retention",
        "minimum_fold_median_signal_projection_retention",
        "median_risk_projection_retention",
        "minimum_fold_median_risk_projection_retention",
        "passed",
        "economic_generation_may_be_minted",
        "economic_panel_authorized",
        "outer_2026_accessed",
        "protocol_sha256",
        "schema",
    }
    traces = value.get("fold_trace_sha256")
    trace_rows = list(traces) if isinstance(traces, (list, tuple)) else None
    gross_lcb = value.get("gross_active_lcb_by_block")
    net_lcb = value.get("net_10bp_active_lcb_by_block")
    spread_lcb = value.get("spread_lcb_by_block")
    counts = tuple(
        value.get(name)
        for name in (
            "positive_mean_ic_fold_count",
            "positive_median_ic_fold_count",
            "positive_date_fraction_fold_count",
            "positive_spread_fold_count",
        )
    )
    finite_names = (
        "mean_rank_ic",
        "annualized_gross_active_return",
        "annualized_net_active_return_10bp",
        "median_signal_projection_retention",
        "minimum_fold_median_signal_projection_retention",
        "median_risk_projection_retention",
        "minimum_fold_median_risk_projection_retention",
    )
    finite_values = tuple(value.get(name) for name in finite_names)
    break_even = value.get("break_even_one_way_cost_basis_points")
    category = value.get("break_even_category")
    if (
        set(value) != expected_keys
        or value.get("schema") != M03R_V15_QUALIFICATION_SCHEMA
        or value.get("protocol_sha256") != M03R_V15_PROTOCOL_SHA256
        or value.get("setting_index") != setting_index
        or value.get("setting_id") != M03R_V15_SETTING_IDS[setting_index]
        or trace_rows != expected_fold_traces
        or trace_rows is None
        or len(trace_rows) != 6
        or len(set(trace_rows)) != 6
        or value.get("bootstrap_plan_sha256") != expected_bootstrap_plan_sha256
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item not in range(7)
            for item in counts
        )
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in finite_values
        )
        or any(
            not isinstance(rows, (list, tuple))
            or len(rows) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in rows
            )
            for rows in (gross_lcb, net_lcb, spread_lcb)
        )
        or category
        not in {
            "finite-positive",
            "favorable-cost-dominance",
            "no-positive-break-even",
        }
        or (
            category == "finite-positive"
            and (
                isinstance(break_even, bool)
                or not isinstance(break_even, (int, float))
                or not math.isfinite(float(break_even))
                or float(break_even) <= 0.0
            )
        )
        or (category != "finite-positive" and break_even is not None)
        or value.get("economic_panel_authorized") is not False
        or value.get("outer_2026_accessed") is not False
    ):
        raise M03RV15SeadragonLifecycleError(
            "horizon qualification semantics or fold lineage drifted"
        )
    spec = M03R_V15_PREDICTIVE_SPEC
    count_rows = cast(tuple[int, int, int, int], counts)
    passed = (
        float(value["mean_rank_ic"]) >= spec.minimum_mean_spearman_rank_ic
        and count_rows[0] >= spec.minimum_positive_mean_ic_fold_count
        and count_rows[1] >= spec.minimum_positive_median_ic_fold_count
        and count_rows[2] >= spec.minimum_positive_date_fraction_fold_count
        and count_rows[3] >= spec.minimum_positive_spread_fold_count
        and float(cast(list[float], gross_lcb)[1])
        > spec.minimum_gross_active_return_lcb
        and float(cast(list[float], net_lcb)[1])
        > spec.minimum_net_10bp_active_return_lcb
        and float(cast(list[float], spread_lcb)[1]) > spec.minimum_spread_lcb
        and (
            category == "favorable-cost-dominance"
            or (
                break_even is not None
                and float(break_even)
                >= spec.minimum_break_even_one_way_cost_basis_points
            )
        )
        and float(value["median_risk_projection_retention"])
        >= spec.minimum_median_risk_projection_retention
        and float(value["minimum_fold_median_risk_projection_retention"])
        >= spec.minimum_fold_median_risk_projection_retention
    )
    if (
        value.get("passed") is not passed
        or value.get("economic_generation_may_be_minted") is not passed
        or _compact_sha256(value) != expected_receipt_sha256
    ):
        raise M03RV15SeadragonLifecycleError(
            "horizon qualification pass semantics or receipt drifted"
        )
    assert trace_rows is not None
    for digest in (
        *trace_rows,
        expected_bootstrap_plan_sha256,
        expected_receipt_sha256,
    ):
        if not isinstance(digest, str):
            raise M03RV15SeadragonLifecycleError(
                "horizon qualification digest is absent"
            )
        _require_sha256("horizon qualification digest", digest)


def _validate_host_bootstrap_plan(value: Mapping[str, Any]) -> str:
    """Validate the worker-built bootstrap receipt without importing PyTorch.

    Exact draw bytes are produced and validated inside the immutable worker package.
    The host supervisor independently binds the complete typed plan and its digest;
    it deliberately does not regenerate tensor RNG state on the CPU-only host.
    """

    expected_keys = {
        "chronology_sha256",
        "fold_lengths",
        "bootstrap_seed",
        "draw_sha256_by_block",
        "block_sessions",
        "replicates",
        "protocol_sha256",
        "schema",
    }
    fold_lengths = value.get("fold_lengths")
    draw_hashes = value.get("draw_sha256_by_block")
    block_sessions = value.get("block_sessions")
    if (
        set(value) != expected_keys
        or not isinstance(fold_lengths, list)
        or len(fold_lengths) != 6
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in fold_lengths
        )
        or not isinstance(draw_hashes, list)
        or len(draw_hashes) != 3
        or not all(isinstance(item, str) for item in draw_hashes)
        or block_sessions != [10, 21, 30]
        or fold_lengths != [63] * 6
        or value.get("bootstrap_seed") != M03R_V15_PREDICTIVE_SPEC.bootstrap_seed
        or value.get("replicates") != M03R_V15_PREDICTIVE_SPEC.bootstrap_replicates
        or value.get("protocol_sha256") != M03R_V15_PROTOCOL_SHA256
        or value.get("schema") != M03R_V15_BOOTSTRAP_PLAN_SCHEMA
    ):
        raise M03RV15SeadragonLifecycleError("bootstrap plan is invalid")
    chronology = value.get("chronology_sha256")
    if not isinstance(chronology, str):
        raise M03RV15SeadragonLifecycleError("bootstrap chronology hash is absent")
    _require_sha256("bootstrap chronology", chronology)
    for digest in draw_hashes:
        _require_sha256("bootstrap draw", cast(str, digest))
    return _compact_sha256(value)


def _validate_one_worker(
    config: M03RV15AttachSupervisorConfig,
    row: M03RV15ExpectedCompletion,
) -> tuple[str, dict[str, Any]]:
    root = (
        Path(config.output_root)
        / f"completion-{row.completion_index:02d}-setting-{row.setting_index:02d}"
    )
    startup_path = root / "two-h100-startup.json"
    terminal_path = root / "predictive-terminal.json"
    startup_sha = _file_sha256(
        common._regular_no_symlink(startup_path, label="startup")
    )
    terminal_sha = _file_sha256(
        common._regular_no_symlink(terminal_path, label="predictive terminal")
    )
    startup = _read_bound_json(startup_path, startup_sha, "startup")
    terminal = _read_bound_json(terminal_path, terminal_sha, "predictive terminal")
    _receipt_payload(terminal, "predictive terminal")
    _validate_rank_runtime(startup.get("rank_runtime"))
    if (
        startup.get("schema") != M03R_V15_STARTUP_SCHEMA
        or startup.get("package_plan_sha256") != config.package_plan_sha256
        or startup.get("package_plan_file_sha256") != config.package_plan_file_sha256
        or startup.get("authorization_receipt_sha256")
        != config.execution_authorization_receipt_sha256
        or startup.get("worker_plan_sha256") != row.worker_plan_sha256
        or startup.get("setting_index") != row.setting_index
        or startup.get("setting_id") != row.setting_id
        or startup.get("mode") != "predictive"
        or startup.get("exact_h100_80gb_per_rank") is not True
        or startup.get("nccl_process_group_initialized") is not True
        or startup.get("restart_count") != 0
        or startup.get("development_only") is not True
        or startup.get("reportable") is not False
        or startup.get("promotion_eligible") is not False
        or terminal.get("schema") != M03R_V15_WORKER_TERMINAL_SCHEMA
        or terminal.get("package_plan_sha256") != config.package_plan_sha256
        or terminal.get("package_plan_file_sha256") != config.package_plan_file_sha256
        or terminal.get("authorization_receipt_sha256")
        != config.execution_authorization_receipt_sha256
        or terminal.get("worker_plan_sha256") != row.worker_plan_sha256
        or terminal.get("startup_file_sha256") != startup_sha
        or terminal.get("setting_index") != row.setting_index
        or terminal.get("setting_id") != row.setting_id
        or terminal.get("world_size") != 2
        or terminal.get("gpus_per_worker") != 2
        or terminal.get("economic_panel_authorized") is not False
        or terminal.get("economic_optimizer_updates") != 0
        or terminal.get("h100_capacity_evidence") is not True
        or terminal.get("outer_2026_accessed") is not False
        or terminal.get("development_only") is not True
        or terminal.get("reportable") is not False
        or terminal.get("promotion_eligible") is not False
    ):
        raise M03RV15SeadragonLifecycleError("worker terminal identity drifted")
    fold_hashes = terminal.get("fold_terminal_file_sha256")
    if not isinstance(fold_hashes, list) or len(fold_hashes) != 6:
        raise M03RV15SeadragonLifecycleError("worker must bind six fold receipts")
    package, _authorization = _load_package_authorization(config)
    worker = package.panel.workers[row.setting_index]
    fold_receipts: list[str] = []
    fold_traces: list[str] = []
    for fold_index, expected_sha in enumerate(fold_hashes):
        if not isinstance(expected_sha, str):
            raise M03RV15SeadragonLifecycleError("fold receipt hash is absent")
        fold_path = root / "receipts" / f"fold-{fold_index:02d}-terminal.json"
        fold = _read_bound_json(fold_path, expected_sha, "fold receipt")
        _receipt_payload(fold, "fold receipt")
        checkpoint = _host_output_path(config, fold.get("checkpoint_path"))
        artifact = _host_output_path(
            config, fold.get("qualification_artifact_path")
        )
        checkpoint_sha = fold.get("checkpoint_file_sha256")
        artifact_sha = fold.get("qualification_artifact_file_sha256")
        trace_sha = fold.get("qualification_trace_sha256")
        if (
            not isinstance(checkpoint_sha, str)
            or not isinstance(artifact_sha, str)
            or not isinstance(trace_sha, str)
        ):
            raise M03RV15SeadragonLifecycleError("fold artifact identity is absent")
        _read_bound_file(checkpoint, checkpoint_sha, "checkpoint")
        _read_bound_file(artifact, artifact_sha, "qualification artifact")
        try:
            checkpoint_selection = M03RV15CheckpointSelectionReceipt(
                **dict(_mapping(fold.get("checkpoint_selection"), "selection"))
            )
            checkpoint_selection.validate()
        except (TypeError, ValueError) as exc:
            raise M03RV15SeadragonLifecycleError(
                "fold checkpoint selection is invalid"
            ) from exc
        for name in (
            "model_state_sha256",
            "optimizer_state_sha256",
            "training_update_evidence_root_sha256",
            "training_source_array_root_sha256",
            "training_target_operator_root_sha256",
            "training_action_operator_root_sha256",
            "qualification_trace_sha256",
            "qualification_batch_receipt_sha256",
            "fold_risk_state_sha256",
        ):
            value = fold.get(name)
            if not isinstance(value, str):
                raise M03RV15SeadragonLifecycleError(f"fold {name} is absent")
            _require_sha256(name, value)
        if (
            fold.get("schema") != M03R_V15_FOLD_TERMINAL_SCHEMA
            or fold.get("package_plan_sha256") != config.package_plan_sha256
            or fold.get("authorization_receipt_sha256")
            != config.execution_authorization_receipt_sha256
            or fold.get("worker_plan_sha256") != row.worker_plan_sha256
            or fold.get("setting_index") != row.setting_index
            or fold.get("setting_id") != row.setting_id
            or fold.get("fold_index") != fold_index
            or fold.get("completed_updates") != worker.fold_optimizer_updates[fold_index]
            or fold.get("training_epoch_count")
            != M03R_V15_PREDICTIVE_SPEC.training_epochs
            or fold.get("selected_epoch_index")
            != checkpoint_selection.selected_epoch_index
            or fold.get("checkpoint_selection_sha256")
            != checkpoint_selection.receipt_sha256
            or tuple(fold.get("inner_validation_receipt_sha256", ()))
            != tuple(checkpoint_selection.candidate_validation_receipt_sha256)
            or fold.get("qualification_tail_accessed_for_selection") is not False
            or fold.get("qualification_evaluated_only_after_checkpoint_reload")
            is not True
            or fold.get("economic_optimizer_updates") != 0
            or fold.get("outer_2026_accessed") is not False
            or fold.get("development_only") is not True
            or fold.get("reportable") is not False
            or fold.get("promotion_eligible") is not False
        ):
            raise M03RV15SeadragonLifecycleError("fold receipt semantics drifted")
        fold_receipts.append(expected_sha)
        fold_traces.append(trace_sha)
    bootstrap_value = _mapping(terminal.get("bootstrap_plan"), "bootstrap plan")
    bootstrap_sha256 = _validate_host_bootstrap_plan(bootstrap_value)
    if terminal.get("bootstrap_plan_sha256") != bootstrap_sha256:
        raise M03RV15SeadragonLifecycleError("bootstrap plan identity drifted")
    qualification = _mapping(
        terminal.get("predictive_qualification"), "predictive qualification"
    )
    qualification_sha = terminal.get("predictive_qualification_sha256")
    if not isinstance(qualification_sha, str):
        raise M03RV15SeadragonLifecycleError(
            "terminal predictive qualification hash is absent"
        )
    _validate_qualification(
        qualification,
        setting_index=row.setting_index,
        expected_fold_traces=fold_traces,
        expected_bootstrap_plan_sha256=bootstrap_sha256,
        expected_receipt_sha256=qualification_sha,
    )
    selected = terminal.get("selected_horizon")
    gate = terminal.get("predictive_gate_passed")
    if gate is True:
        if selected != M03R_V15_SELECTED_HORIZON_SESSIONS:
            raise M03RV15SeadragonLifecycleError("passed terminal omitted selection")
        if (
            qualification.get("passed") is not True
            or terminal.get("economic_generation_may_be_minted") is not True
        ):
            raise M03RV15SeadragonLifecycleError("selected qualification drifted")
    elif gate is False:
        if (
            selected is not None
            or terminal.get("economic_generation_may_be_minted") is not False
            or qualification.get("passed") is not False
        ):
            raise M03RV15SeadragonLifecycleError("failed gate selection drifted")
    else:
        raise M03RV15SeadragonLifecycleError("predictive gate flag is invalid")
    return terminal_sha, {
        "setting_index": row.setting_index,
        "setting_id": row.setting_id,
        "startup_file_sha256": startup_sha,
        "terminal_file_sha256": terminal_sha,
        "fold_receipt_file_sha256": fold_receipts,
        "predictive_gate_passed": gate,
        "selected_horizon": selected,
    }


def validate_m03r_v15_predictive_coverage(
    config: M03RV15AttachSupervisorConfig,
    *,
    owned_pods: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    package, authorization = _load_package_authorization(config)
    expected = tuple(
        (index, worker.setting_index, worker.setting_id, worker.receipt_sha256)
        for index, worker in enumerate(package.panel.workers)
    )
    observed = tuple(
        (
            row.completion_index,
            row.setting_index,
            row.setting_id,
            row.worker_plan_sha256,
        )
        for row in config.expected_completions
    )
    if observed != expected:
        raise M03RV15SeadragonLifecycleError(
            "attach completion inventory drifted from the package plan"
        )
    if len(owned_pods) != 2:
        raise M03RV15SeadragonLifecycleError(
            "complete v15 Job must retain exactly two owned Pods"
        )
    common._validate_owned_pods(owned_pods, expected_uid=config.job_uid)
    pod_inventory: dict[int, dict[str, str]] = {}
    for pod in owned_pods:
        name, raw_index = common._pod_identity(pod)
        metadata = _mapping(pod.get("metadata"), "terminal Pod metadata")
        status = _mapping(pod.get("status"), "terminal Pod status")
        uid = metadata.get("uid")
        try:
            index = int(raw_index) if raw_index is not None else -1
        except ValueError as exc:
            raise M03RV15SeadragonLifecycleError(
                "terminal Pod completion index is invalid"
            ) from exc
        if (
            index not in {0, 1}
            or index in pod_inventory
            or not isinstance(uid, str)
            or not uid
            or status.get("phase") != "Succeeded"
        ):
            raise M03RV15SeadragonLifecycleError(
                "terminal Pods are not one success per v15 completion"
            )
        pod_inventory[index] = {"pod_name": name, "pod_uid": uid}
    if set(pod_inventory) != {0, 1}:
        raise M03RV15SeadragonLifecycleError("terminal Pod coverage drifted")
    workers: dict[str, Any] = {}
    receipt_hashes: dict[str, str] = {}
    for row in config.expected_completions:
        terminal_sha, evidence = _validate_one_worker(config, row)
        key = str(row.completion_index)
        workers[key] = {**pod_inventory[row.completion_index], **evidence}
        relative = (
            f"completion-{row.completion_index:02d}-setting-"
            f"{row.setting_index:02d}/predictive-terminal.json"
        )
        receipt_hashes[relative] = terminal_sha
    payload: dict[str, Any] = {
        "schema": COMPLETION_COVERAGE_SCHEMA,
        "protocol_sha256": M03R_V15_PROTOCOL_SHA256,
        "package_plan_sha256": config.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "source_archive_sha256": config.source_archive_sha256,
        "capacity_receipt_sha256": config.capacity_receipt_sha256,
        "completion_count": len(config.expected_completions),
        "terminal_file_sha256": receipt_hashes,
        "worker_runtime_proof": workers,
        "economic_panel_authorized": False,
        "outer_evaluation_authorized": False,
        "research_only": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    payload["coverage_sha256"] = _content_sha256(payload)
    return payload


def _publish_activation_attach_required(
    root: Path,
    config: M03RV15AttachSupervisorConfig | M03RV15CapacityAttachConfig,
    error: Exception,
    transport: common.KubectlTransport,
) -> None:
    try:
        job = transport.get_job(allow_absent=True)
        pods = transport.get_owned_pods()
        job_sha = (
            None
            if job is None
            else common._exclusive_json(root / "activation-current-job.json", job)
        )
        pods_sha = common._exclusive_json(
            root / "activation-current-pods.json",
            {"apiVersion": "v1", "kind": "PodList", "items": list(pods)},
        )
        state = "absent" if job is None else "present"
    except Exception as reconcile_error:  # noqa: BLE001 - preserve ambiguity
        job_sha = None
        pods_sha = None
        state = "unknown"
        error = M03RV15SeadragonLifecycleError(
            f"activation error={error}; reconciliation error={reconcile_error}"
        )
    common._exclusive_json(
        root / "activation-attach-required.json",
        {
            "schema": ("rl-quant.top2000-dev.m03r-v15-activation-attach-required-v1"),
            "job_name": config.job_name,
            "job_uid": config.job_uid,
            "run_id": config.run_id,
            "state": state,
            "current_job_file_sha256": job_sha,
            "current_pods_file_sha256": pods_sha,
            "error_type": type(error).__name__,
            "error": str(error),
            "activation_retried": False,
            "cleanup_performed": False,
            "attach_required": True,
        },
    )


def _cleanup_preactivation_exact(
    *,
    root: Path,
    config: M03RV15AttachSupervisorConfig | M03RV15CapacityAttachConfig,
    binding: M03RV7AdmittedJobBinding,
    transport: common.KubectlTransport,
    sleep: Any,
) -> None:
    """Delete one proven suspended zero-Pod UID using its current RV once."""

    reads: list[Mapping[str, Any]] = []
    read_evidence: list[dict[str, Any]] = []
    for observation in range(2):
        job = transport.get_job(allow_absent=True)
        pods = transport.get_owned_pods()
        if job is None or pods:
            raise M03RV15SeadragonLifecycleError(
                "preactivation cleanup requires a present suspended zero-Pod Job"
            )
        common._job_identity(
            job,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(job, config)
        spec = _mapping(job.get("spec"), "preactivation cleanup Job spec")
        metadata = _mapping(job.get("metadata"), "preactivation cleanup metadata")
        resource_version = metadata.get("resourceVersion")
        spec_sha = _compact_sha256(spec)
        if (
            spec.get("suspend") is not True
            or not isinstance(resource_version, str)
            or not resource_version
            or spec_sha != binding.admitted_spec_sha256
        ):
            raise M03RV15SeadragonLifecycleError(
                "preactivation cleanup spec drifted from the admitted binding"
            )
        reads.append(job)
        read_evidence.append(
            {
                "observation": observation + 1,
                "job_sha256": common._content_sha256(job),
                "resource_version": resource_version,
                "suspended": True,
                "owned_pod_uids": [],
            }
        )
        if observation == 0:
            sleep(0.1)
    first_metadata = _mapping(reads[0].get("metadata"), "first cleanup metadata")
    second_metadata = _mapping(reads[1].get("metadata"), "second cleanup metadata")
    if first_metadata.get("uid") != second_metadata.get("uid"):
        raise M03RV15SeadragonLifecycleError(
            "preactivation cleanup UID changed between reads"
        )
    resource_version = cast(str, second_metadata["resourceVersion"])
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
    request = M03RV7ExactJobCleanupRequest(
        **fields,
        delete_options_sha256=_compact_sha256(unsigned.delete_options),
    )
    common._exclusive_json(
        root / "preactivation-cleanup-reads.json",
        {
            "schema": ("rl-quant.top2000-dev.m03r-v15-preactivation-cleanup-reads-v1"),
            "binding_receipt_sha256": binding.receipt_sha256,
            "fresh_reads": read_evidence,
            "bound_resource_version_equality_permitted": (
                resource_version
                in {binding.first_resource_version, binding.second_resource_version}
            ),
        },
    )
    common._exclusive_json(root / "cleanup-request.json", asdict(request))
    options_path = root / "cleanup-delete-options.json"
    common._exclusive_json(options_path, request.delete_options)
    delete_error: Exception | None = None
    try:
        transport.delete(request, options_path)
    except Exception as exc:  # noqa: BLE001 - reconcile one ambiguous delete
        delete_error = exc
    absence: (
        tuple[
            bool,
            tuple[str, ...],
            dict[str, Any],
            bool,
            tuple[str, ...],
            dict[str, Any],
        ]
        | None
    ) = None
    last_first: dict[str, Any] = {}
    last_second: dict[str, Any] = {}
    for attempt in range(config.request_timeout_seconds):
        first_absent, first_pods, first = common._absence_snapshot(transport)
        sleep(0.1)
        second_absent, second_pods, second = common._absence_snapshot(transport)
        last_first, last_second = first, second
        if first_absent and second_absent and not first_pods and not second_pods:
            absence = (
                first_absent,
                first_pods,
                first,
                second_absent,
                second_pods,
                second,
            )
            break
        if attempt + 1 < config.request_timeout_seconds:
            sleep(1.0)
    if absence is None:
        common._exclusive_json(
            root / "preactivation-cleanup-attach-required.json",
            {
                "schema": (
                    "rl-quant.top2000-dev.m03r-v15-preactivation-cleanup-"
                    "attach-required-v1"
                ),
                "delete_attempted": True,
                "delete_retried": False,
                "error_type": (
                    None if delete_error is None else type(delete_error).__name__
                ),
                "error": None if delete_error is None else str(delete_error),
                "first_reconciliation": last_first,
                "second_reconciliation": last_second,
                "cleanup_performed": False,
                "attach_required": True,
            },
        )
        raise M03RV15ActivationAttachRequired(
            "preactivation delete outcome is ambiguous"
        ) from delete_error
    (
        first_absent,
        first_pods,
        first,
        second_absent,
        second_pods,
        second,
    ) = absence
    verification_sha = common._exclusive_json(
        root / "cleanup-verification.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-cleanup-verification-v1",
            "first": first,
            "second": second,
            "delete_attempted": True,
            "delete_retried": False,
        },
    )
    receipt = build_m03r_v7_exact_cleanup_receipt(
        request=request,
        first_job_absent=first_absent,
        second_job_absent=second_absent,
        first_owned_pod_uids=first_pods,
        second_owned_pod_uids=second_pods,
        verification_evidence_sha256=verification_sha,
    )
    common._exclusive_json(root / "cleanup-receipt.json", asdict(receipt))


def _cleanup_postactivation_exact(
    *,
    root: Path,
    binding: M03RV7AdmittedJobBinding,
    transport: common.KubectlTransport,
    request_timeout_seconds: int,
    validate_job: Any,
    sleep: Any,
) -> None:
    """Issue one post-run UID/RV delete and reconcile without retrying it."""

    fresh = transport.get_job(allow_absent=True)
    if fresh is None:
        raise M03RV15SeadragonLifecycleError(
            "exact Job disappeared before the sole post-run cleanup request"
        )
    validate_job(fresh)
    request = build_m03r_v7_exact_job_cleanup_request(binding, fresh)
    common._exclusive_json(root / "cleanup-request.json", asdict(request))
    options_path = root / "delete-options.json"
    common._exclusive_json(options_path, request.delete_options)
    delete_error: Exception | None = None
    try:
        transport.delete(request, options_path)
    except Exception as exc:  # noqa: BLE001 - reconcile the sole ambiguous request
        delete_error = exc
    absence: (
        tuple[
            bool,
            tuple[str, ...],
            dict[str, Any],
            bool,
            tuple[str, ...],
            dict[str, Any],
        ]
        | None
    ) = None
    last_first: dict[str, Any] = {}
    last_second: dict[str, Any] = {}
    for attempt in range(request_timeout_seconds):
        first_absent, first_pods, first = common._absence_snapshot(transport)
        sleep(0.1)
        second_absent, second_pods, second = common._absence_snapshot(transport)
        last_first, last_second = first, second
        if first_absent and second_absent and not first_pods and not second_pods:
            absence = (
                first_absent,
                first_pods,
                first,
                second_absent,
                second_pods,
                second,
            )
            break
        if attempt + 1 < request_timeout_seconds:
            sleep(1.0)
    if absence is None:
        common._exclusive_json(
            root / "cleanup-attach-required.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v15-cleanup-attach-required-v1",
                "delete_attempted": True,
                "delete_retried": False,
                "error_type": (
                    None if delete_error is None else type(delete_error).__name__
                ),
                "error": None if delete_error is None else str(delete_error),
                "first_reconciliation": last_first,
                "second_reconciliation": last_second,
                "cleanup_performed": False,
                "attach_required": True,
            },
        )
        raise M03RV15ActivationAttachRequired(
            "post-run delete outcome is ambiguous"
        ) from delete_error
    (
        first_absent,
        first_pods,
        first,
        second_absent,
        second_pods,
        second,
    ) = absence
    verification_sha = common._exclusive_json(
        root / "cleanup-verification.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-cleanup-verification-v1",
            "first": first,
            "second": second,
            "delete_attempted": True,
            "delete_retried": False,
        },
    )
    receipt = build_m03r_v7_exact_cleanup_receipt(
        request=request,
        first_job_absent=first_absent,
        second_job_absent=second_absent,
        first_owned_pod_uids=first_pods,
        second_owned_pod_uids=second_pods,
        verification_evidence_sha256=verification_sha,
    )
    common._exclusive_json(root / "cleanup-receipt.json", asdict(receipt))


def _prepare_activation(
    *,
    root: Path,
    config: M03RV15AttachSupervisorConfig,
    config_sha256: str,
    binding: M03RV7AdmittedJobBinding,
    configured_activation: M03RV7ExactJobActivationRequest,
    process_path: Path,
    live: common.KubectlTransport,
    sleep: Any,
) -> tuple[M03RV7ExactJobActivationRequest, str]:
    """Build the final activation request or exact-clean preactivation state."""

    try:
        fresh = live.get_job()
        if fresh is None:
            raise M03RV15SeadragonLifecycleError("bound suspended Job is absent")
        common._job_identity(
            fresh,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(fresh, config)
        if live.get_owned_pods():
            raise M03RV15SeadragonLifecycleError("attach target already owns Pods")
        activation = build_m03r_v7_exact_job_activation_request(binding, fresh)
        if not common._same_activation_contract(configured_activation, activation):
            raise M03RV15SeadragonLifecycleError(
                "fresh activation request drifted from configured identity"
            )
        readiness_sha = common._exclusive_json(
            root / "readiness.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v15-supervisor-readiness-v1",
                "config_sha256": config_sha256,
                "binding_receipt_sha256": binding.receipt_sha256,
                "configured_activation_file_sha256": (
                    config.activation_request_file_sha256
                ),
                "runtime_activation_request_sha256": activation.request_sha256,
                "zero_owned_pods": True,
            },
        )
        arm_sha = common._exclusive_json(
            root / "arm.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v15-supervisor-arm-v1",
                "readiness_file_sha256": readiness_sha,
                "spawn_process_file_sha256": _file_sha256(process_path),
                "capacity_receipt_sha256": config.capacity_receipt_sha256,
            },
        )
        common._exclusive_json(
            root / "activation-request-runtime.json", asdict(activation)
        )
        return activation, arm_sha
    except Exception:
        _cleanup_preactivation_exact(
            root=root,
            config=config,
            binding=binding,
            transport=live,
            sleep=sleep,
        )
        raise


def _prepare_capacity_activation(
    *,
    root: Path,
    config: M03RV15CapacityAttachConfig,
    config_sha256: str,
    binding: M03RV7AdmittedJobBinding,
    configured_activation: M03RV7ExactJobActivationRequest,
    live: common.KubectlTransport,
    sleep: Any,
) -> M03RV7ExactJobActivationRequest:
    try:
        fresh = live.get_job()
        if fresh is None:
            raise M03RV15SeadragonLifecycleError("bound capacity Job is absent")
        common._job_identity(
            fresh,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(fresh, config)
        if live.get_owned_pods():
            raise M03RV15SeadragonLifecycleError(
                "capacity attach target already owns Pods"
            )
        activation = build_m03r_v7_exact_job_activation_request(binding, fresh)
        if not common._same_activation_contract(configured_activation, activation):
            raise M03RV15SeadragonLifecycleError(
                "fresh capacity activation request drifted"
            )
        common._exclusive_json(
            root / "readiness.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v15-capacity-readiness-v1",
                "config_sha256": config_sha256,
                "binding_receipt_sha256": binding.receipt_sha256,
                "configured_activation_file_sha256": (
                    config.activation_request_file_sha256
                ),
                "runtime_activation_request_sha256": activation.request_sha256,
                "zero_owned_pods": True,
            },
        )
        common._exclusive_json(
            root / "activation-request-runtime.json", asdict(activation)
        )
        return activation
    except Exception:
        _cleanup_preactivation_exact(
            root=root,
            config=config,
            binding=binding,
            transport=live,
            sleep=sleep,
        )
        raise


def _run_supervisor_inner(
    config: M03RV15AttachSupervisorConfig,
    *,
    config_sha256: str,
    transport: common.KubectlTransport | None = None,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> None:
    supervisor_hard_deadline = monotonic() + config.hard_wall_seconds
    root = common._directory_no_symlink(
        Path(config.evidence_root), label="evidence root"
    )
    source = common._regular_no_symlink(Path(__file__), label="lifecycle source")
    if _file_sha256(source) != config.lifecycle_source_sha256:
        raise M03RV15SeadragonLifecycleError("lifecycle source hash drifted")
    _load_package_authorization(config)
    binding: M03RV7AdmittedJobBinding = common._binding_from_file(
        Path(config.binding_path), config.binding_file_sha256
    )
    configured_activation = common._activation_from_file(
        Path(config.activation_request_path), config.activation_request_file_sha256
    )
    if (
        binding.job_name != config.job_name
        or binding.run_id != config.run_id
        or binding.job_uid != config.job_uid
        or binding.parallelism != config.parallelism
        or configured_activation.job_uid != config.job_uid
        or configured_activation.binding_receipt_sha256 != binding.receipt_sha256
    ):
        raise M03RV15SeadragonLifecycleError(
            "v15 config, binding, or activation identity drifted"
        )
    process_path = root / "spawn-process.json"
    deadline = monotonic() + config.handshake_timeout_seconds
    while not process_path.exists():
        if monotonic() >= deadline:
            raise M03RV15SeadragonLifecycleError(
                "spawn-process receipt was not published"
            )
        sleep(0.1)
    process_receipt = _mapping(
        common._read_json_file(process_path), "spawn-process receipt"
    )
    common._validate_spawned_identity(process_receipt, pid=os.getpid())
    live = transport or common.AttachOnlyKubectl(
        kubectl_path=config.kubectl_path,
        kubeconfig_path=config.kubeconfig_path,
        context=config.context,
        namespace=config.namespace,
        job_name=config.job_name,
        job_uid=config.job_uid,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    activation, arm_sha = _prepare_activation(
        root=root,
        config=config,
        config_sha256=config_sha256,
        binding=binding,
        configured_activation=configured_activation,
        process_path=process_path,
        live=live,
        sleep=sleep,
    )
    try:
        activated = live.activate(activation)
    except Exception as exc:
        _publish_activation_attach_required(root, config, exc, live)
        raise M03RV15ActivationAttachRequired(
            "activation response was ambiguous; exact Job retained"
        ) from exc
    common._job_identity(
        activated,
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=config.job_uid,
    )
    _job_artifact_identity(activated, config)
    activated_spec = _mapping(activated.get("spec"), "activated Job spec")
    if activated_spec.get("suspend") is not False:
        raise M03RV15SeadragonLifecycleError(
            "direct activation response did not unsuspend the exact Job"
        )
    activation_sha = common._exclusive_json(
        root / "activation.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-supervisor-activation-v1",
            "arm_file_sha256": arm_sha,
            "activation_request_sha256": activation.request_sha256,
            "activated_job_sha256": common._content_sha256(activated),
            "activation_reconciled_after_transport_error": False,
            "activation_retried": False,
        },
    )
    common._validate_spawned_identity(process_receipt, pid=os.getpid())
    common._exclusive_json(
        root / "launch-success.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-supervisor-launch-success-v1",
            "activation_file_sha256": activation_sha,
            "job_name": config.job_name,
            "job_uid": config.job_uid,
            "run_id": config.run_id,
            "parallelism": config.parallelism,
            "gpus_per_worker": 2,
            "request_ceiling": config.parallelism * 2,
            "capacity_receipt_sha256": config.capacity_receipt_sha256,
            "economic_panel_authorized": False,
        },
    )
    terminal_job: Mapping[str, Any] | None = None
    terminal_pods: tuple[Mapping[str, Any], ...] = ()
    terminal_reason = ""
    while monotonic() < supervisor_hard_deadline:
        observed = live.get_job()
        if observed is None:
            raise M03RV15SeadragonLifecycleError(
                "exact Job disappeared before terminal capture"
            )
        common._job_identity(
            observed,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(observed, config)
        pods = live.get_owned_pods()
        condition = common._true_condition(observed)
        if condition is not None:
            terminal_job = observed
            terminal_pods = pods
            terminal_reason = condition.lower()
            break
        sleep(config.poll_interval_seconds)
    if terminal_job is None:
        terminal_job = live.get_job()
        if terminal_job is None:
            raise M03RV15SeadragonLifecycleError(
                "exact Job disappeared at supervisor hard wall"
            )
        terminal_pods = live.get_owned_pods()
        terminal_reason = "supervisor-hard-wall"
    common._capture_terminal(
        root=root,
        job=terminal_job,
        pods=terminal_pods,
        transport=live,
        reason=terminal_reason,
        log_limit_bytes=config.log_limit_bytes,
    )
    if terminal_reason == "complete":
        coverage = validate_m03r_v15_predictive_coverage(
            config, owned_pods=terminal_pods
        )
        common._exclusive_json(root / "completion-coverage.json", coverage)

    def validate_cleanup_job(job: Mapping[str, Any]) -> None:
        common._job_identity(
            job,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(job, config)

    _cleanup_postactivation_exact(
        root=root,
        binding=binding,
        transport=live,
        request_timeout_seconds=config.request_timeout_seconds,
        validate_job=validate_cleanup_job,
        sleep=sleep,
    )
    if terminal_reason != "complete":
        raise M03RV15SeadragonLifecycleError(
            f"v15 predictive Job terminated without completion: {terminal_reason}"
        )


def _run_capacity_supervisor_inner(
    config: M03RV15CapacityAttachConfig,
    *,
    config_sha256: str,
    transport: common.KubectlTransport,
    sleep: Any,
    monotonic: Any,
) -> None:
    hard_deadline = monotonic() + config.hard_wall_seconds
    root = common._directory_no_symlink(
        Path(config.evidence_root), label="capacity evidence root"
    )
    source = common._regular_no_symlink(Path(__file__), label="lifecycle source")
    if _file_sha256(source) != config.lifecycle_source_sha256:
        raise M03RV15SeadragonLifecycleError("capacity lifecycle source hash drifted")
    package, authorization = _load_package_authorization(config)
    static_gate = _read_bound_json(
        Path(config.static_gate_path),
        config.static_gate_file_sha256,
        "static gate receipt",
    )
    _validate_static_gate_lineage(
        static_gate,
        expected_receipt_sha256=config.static_gate_receipt_sha256,
        package_plan_sha256=package.package_plan_sha256,
        authorization_receipt_sha256=authorization.receipt_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
    )
    process_path = root / "spawn-process.json"
    handshake_deadline = monotonic() + config.handshake_timeout_seconds
    while not process_path.exists():
        if monotonic() >= handshake_deadline:
            raise M03RV15SeadragonLifecycleError(
                "capacity spawn-process receipt was not published"
            )
        sleep(0.1)
    process_receipt = _mapping(
        common._read_json_file(process_path), "capacity spawn-process receipt"
    )
    common._validate_spawned_identity(process_receipt, pid=os.getpid())
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
        or configured_activation.job_uid != config.job_uid
        or configured_activation.parallelism != 1
        or configured_activation.binding_receipt_sha256 != binding.receipt_sha256
    ):
        raise M03RV15SeadragonLifecycleError(
            "capacity config, binding, or activation identity drifted"
        )
    activation = _prepare_capacity_activation(
        root=root,
        config=config,
        config_sha256=config_sha256,
        binding=binding,
        configured_activation=configured_activation,
        live=transport,
        sleep=sleep,
    )
    arm_sha = common._exclusive_json(
        root / "arm.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-capacity-arm-v1",
            "config_sha256": config_sha256,
            "spawn_process_file_sha256": _file_sha256(process_path),
            "activation_request_sha256": activation.request_sha256,
        },
    )
    try:
        activated = transport.activate(activation)
    except Exception as exc:
        _publish_activation_attach_required(root, config, exc, transport)
        raise M03RV15ActivationAttachRequired(
            "capacity activation response was ambiguous; exact Job retained"
        ) from exc
    common._job_identity(
        activated,
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=config.job_uid,
    )
    _job_artifact_identity(activated, config)
    activated_spec = _mapping(activated.get("spec"), "activated capacity Job spec")
    if activated_spec.get("suspend") is not False:
        raise M03RV15SeadragonLifecycleError(
            "direct capacity activation response did not unsuspend the exact Job"
        )
    common._exclusive_json(
        root / "activation.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-capacity-activation-v1",
            "arm_file_sha256": arm_sha,
            "activation_request_sha256": activation.request_sha256,
            "activated_job_sha256": common._content_sha256(activated),
            "activation_reconciled_after_transport_error": False,
            "activation_retried": False,
        },
    )
    common._validate_spawned_identity(process_receipt, pid=os.getpid())
    common._exclusive_json(
        root / "launch-success.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-capacity-launch-success-v1",
            "job_name": config.job_name,
            "job_uid": config.job_uid,
            "run_id": config.run_id,
            "parallelism": 1,
            "gpus_per_worker": 2,
            "request_ceiling": 2,
            "training_performed": False,
            "economic_panel_authorized": False,
        },
    )
    terminal_job: Mapping[str, Any] | None = None
    terminal_pods: tuple[Mapping[str, Any], ...] = ()
    terminal_reason = ""
    while monotonic() < hard_deadline:
        observed = transport.get_job()
        if observed is None:
            raise M03RV15SeadragonLifecycleError(
                "exact capacity Job disappeared before terminal capture"
            )
        common._job_identity(
            observed,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(observed, config)
        pods = transport.get_owned_pods()
        condition = common._true_condition(observed)
        if condition is not None:
            terminal_job = observed
            terminal_pods = pods
            terminal_reason = condition.lower()
            break
        sleep(config.poll_interval_seconds)
    if terminal_job is None:
        terminal_job = transport.get_job()
        if terminal_job is None:
            raise M03RV15SeadragonLifecycleError(
                "exact capacity Job disappeared at hard wall"
            )
        terminal_pods = transport.get_owned_pods()
        terminal_reason = "capacity-hard-wall"
    common._capture_terminal(
        root=root,
        job=terminal_job,
        pods=terminal_pods,
        transport=transport,
        reason=terminal_reason,
        log_limit_bytes=config.log_limit_bytes,
    )
    if terminal_reason != "complete" or len(terminal_pods) != 1:
        raise M03RV15SeadragonLifecycleError(
            f"capacity Job did not complete as one exact worker: {terminal_reason}"
        )
    pod = terminal_pods[0]
    _name, index = common._pod_identity(pod)
    status = _mapping(pod.get("status"), "capacity Pod status")
    if index != "0" or status.get("phase") != "Succeeded":
        raise M03RV15SeadragonLifecycleError(
            "capacity Pod is not the successful completion index zero"
        )
    startup_sha, terminal_sha, terminal = _validate_capacity_output(config)

    def validate_cleanup_job(job: Mapping[str, Any]) -> None:
        common._job_identity(
            job,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(job, config)

    _cleanup_postactivation_exact(
        root=root,
        binding=binding,
        transport=transport,
        request_timeout_seconds=config.request_timeout_seconds,
        validate_job=validate_cleanup_job,
        sleep=sleep,
    )
    terminal_receipt = terminal.get("receipt_sha256")
    if not isinstance(terminal_receipt, str):
        raise M03RV15SeadragonLifecycleError("capacity terminal receipt is absent")
    package, authorization = _load_package_authorization(config)
    qualification = M03RV15TwoH100CapacityQualification(
        static_gate_file_sha256=config.static_gate_file_sha256,
        static_gate_receipt_sha256=config.static_gate_receipt_sha256,
        terminal_file_sha256=terminal_sha,
        terminal_receipt_sha256=terminal_receipt,
        startup_file_sha256=startup_sha,
        terminal_evidence_file_sha256=_file_sha256(root / "terminal-evidence.json"),
        cleanup_receipt_file_sha256=_file_sha256(root / "cleanup-receipt.json"),
        package_plan_sha256=config.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        worker_plan_sha256=package.panel.workers[0].receipt_sha256,
        initial_parameter_state_file_sha256=(
            package.artifacts.initial_parameter_state_file_sha256
        ),
        initial_parameter_state_sha256=(
            package.artifacts.initial_parameter_state_sha256
        ),
        initial_parameter_architecture_sha256=(
            package.artifacts.initial_parameter_architecture_sha256
        ),
        disposable_update_step_receipt_sha256=terminal[
            "disposable_update_step_receipt_sha256"
        ],
        qualification_risk_state_sha256=terminal[
            "qualification_risk_state_sha256"
        ],
    )
    qualification.validate_for(package, authorization)
    common._exclusive_json(root / "capacity-qualification.json", asdict(qualification))


def _recover_predictive_supervisor_failure(
    *,
    config: M03RV15AttachSupervisorConfig,
    binding: M03RV7AdmittedJobBinding,
    transport: common.KubectlTransport,
    error: Exception,
    sleep: Any,
) -> None:
    """Capture and issue at most one cleanup delete after a v15 child error."""

    root = common._directory_no_symlink(
        Path(config.evidence_root), label="evidence root"
    )
    if (root / "cleanup-receipt.json").exists():
        return
    try:
        job = transport.get_job(allow_absent=True)
        pods = transport.get_owned_pods()
        if job is None:
            raise M03RV15SeadragonLifecycleError(
                "predictive Job disappeared without exact cleanup evidence"
            )
        common._job_identity(
            job,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )
        _job_artifact_identity(job, config)
        spec = _mapping(job.get("spec"), "predictive recovery Job spec")
        if spec.get("suspend") is True and not pods:
            _cleanup_preactivation_exact(
                root=root,
                config=config,
                binding=binding,
                transport=transport,
                sleep=sleep,
            )
            return
        if spec.get("suspend") is not False:
            raise M03RV15SeadragonLifecycleError(
                "predictive recovery Job suspension state is unknown"
            )
        if not (root / "terminal-evidence.json").exists():
            common._capture_terminal(
                root=root,
                job=job,
                pods=pods,
                transport=transport,
                reason="supervisor-error",
                log_limit_bytes=config.log_limit_bytes,
            )

        def validate_cleanup_job(fresh: Mapping[str, Any]) -> None:
            common._job_identity(
                fresh,
                job_name=config.job_name,
                run_id=config.run_id,
                job_uid=config.job_uid,
            )
            _job_artifact_identity(fresh, config)

        _cleanup_postactivation_exact(
            root=root,
            binding=binding,
            transport=transport,
            request_timeout_seconds=config.request_timeout_seconds,
            validate_job=validate_cleanup_job,
            sleep=sleep,
        )
    except M03RV15ActivationAttachRequired:
        raise
    except Exception as recovery_error:
        attach_path = root / "supervisor-recovery-attach-required.json"
        if not attach_path.exists():
            common._exclusive_json(
                attach_path,
                {
                    "schema": (
                        "rl-quant.top2000-dev.m03r-v15-supervisor-recovery-"
                        "attach-required-v1"
                    ),
                    "primary_error_type": type(error).__name__,
                    "primary_error": str(error),
                    "recovery_error_type": type(recovery_error).__name__,
                    "recovery_error": str(recovery_error),
                    "cleanup_performed": False,
                    "attach_required": True,
                },
            )
        raise M03RV15ActivationAttachRequired(
            "predictive recovery is ambiguous; exact state retained"
        ) from recovery_error


def run_capacity_supervisor(
    config_path: str | Path,
    expected_config_sha256: str,
    *,
    transport: common.KubectlTransport | None = None,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> None:
    config = _load_capacity_config(Path(config_path), expected_config_sha256)
    live = transport or common.AttachOnlyKubectl(
        kubectl_path=config.kubectl_path,
        kubeconfig_path=config.kubeconfig_path,
        context=config.context,
        namespace=config.namespace,
        job_name=config.job_name,
        job_uid=config.job_uid,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    try:
        _run_capacity_supervisor_inner(
            config,
            config_sha256=expected_config_sha256,
            transport=live,
            sleep=sleep,
            monotonic=monotonic,
        )
    except M03RV15ActivationAttachRequired:
        raise
    except Exception as exc:
        root = Path(config.evidence_root)
        if not (root / "capacity-supervisor-error.json").exists():
            common._exclusive_json(
                root / "capacity-supervisor-error.json",
                {
                    "schema": "rl-quant.top2000-dev.m03r-v15-capacity-error-v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "attach_required": False,
                },
            )
        try:
            observed = live.get_job(allow_absent=True)
            pods = live.get_owned_pods()
            if observed is None:
                if pods:
                    raise M03RV15SeadragonLifecycleError(
                        "capacity Job is absent while UID-owned Pods remain"
                    )
                if not (root / "cleanup-receipt.json").exists():
                    raise M03RV15SeadragonLifecycleError(
                        "capacity Job disappeared without exact cleanup evidence"
                    )
            else:
                binding = common._binding_from_file(
                    Path(config.binding_path), config.binding_file_sha256
                )
                common._job_identity(
                    observed,
                    job_name=config.job_name,
                    run_id=config.run_id,
                    job_uid=config.job_uid,
                )
                _job_artifact_identity(observed, config)
                spec = _mapping(observed.get("spec"), "capacity recovery Job spec")
                if spec.get("suspend") is True and not pods:
                    _cleanup_preactivation_exact(
                        root=root,
                        config=config,
                        binding=binding,
                        transport=live,
                        sleep=sleep,
                    )
                elif spec.get("suspend") is False:
                    if not (root / "terminal-evidence.json").exists():
                        common._capture_terminal(
                            root=root,
                            job=observed,
                            pods=pods,
                            transport=live,
                            reason="capacity-supervisor-error",
                            log_limit_bytes=config.log_limit_bytes,
                        )

                    def validate_recovery_job(job: Mapping[str, Any]) -> None:
                        common._job_identity(
                            job,
                            job_name=config.job_name,
                            run_id=config.run_id,
                            job_uid=config.job_uid,
                        )
                        _job_artifact_identity(job, config)

                    _cleanup_postactivation_exact(
                        root=root,
                        binding=binding,
                        transport=live,
                        request_timeout_seconds=config.request_timeout_seconds,
                        validate_job=validate_recovery_job,
                        sleep=sleep,
                    )
                else:
                    raise M03RV15SeadragonLifecycleError(
                        "capacity recovery could not classify the exact Job"
                    )
        except M03RV15ActivationAttachRequired:
            raise
        except Exception as recovery_error:
            common._exclusive_json(
                root / "capacity-recovery-attach-required.json",
                {
                    "schema": (
                        "rl-quant.top2000-dev.m03r-v15-capacity-recovery-"
                        "attach-required-v1"
                    ),
                    "error": str(recovery_error),
                    "cleanup_performed": False,
                    "attach_required": True,
                },
            )
            raise M03RV15ActivationAttachRequired(
                "capacity recovery is ambiguous; exact state retained"
            ) from recovery_error
        raise


def run_attach_supervisor(
    config_path: str | Path,
    expected_config_sha256: str,
    *,
    transport: common.KubectlTransport | None = None,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> None:
    path = Path(config_path)
    config = _load_config(path, expected_config_sha256)
    live = transport or common.AttachOnlyKubectl(
        kubectl_path=config.kubectl_path,
        kubeconfig_path=config.kubeconfig_path,
        context=config.context,
        namespace=config.namespace,
        job_name=config.job_name,
        job_uid=config.job_uid,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    try:
        _run_supervisor_inner(
            config,
            config_sha256=expected_config_sha256,
            transport=live,
            sleep=sleep,
            monotonic=monotonic,
        )
    except M03RV15ActivationAttachRequired:
        raise
    except Exception as exc:
        root = Path(config.evidence_root)
        error_path = root / "supervisor-error.json"
        if not error_path.exists():
            common._exclusive_json(
                error_path,
                {
                    "schema": "rl-quant.top2000-dev.m03r-v15-supervisor-error-v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "attach_required": False,
                },
            )
        binding = common._binding_from_file(
            Path(config.binding_path), config.binding_file_sha256
        )
        _recover_predictive_supervisor_failure(
            config=config,
            binding=binding,
            transport=live,
            error=exc,
            sleep=sleep,
        )
        raise


def _validated_host_tool(path: Path, label: str) -> Path:
    common._regular_no_symlink(path, label=label)
    if not os.access(path, os.X_OK):
        raise M03RV15SeadragonLifecycleError(f"{label} is not executable")
    return path


def _validated_launch_success(
    path: Path,
    config: M03RV15AttachSupervisorConfig,
    process_receipt: Mapping[str, Any],
    pid: int,
) -> Mapping[str, Any] | None:
    try:
        launch = _mapping(common._read_json_file(path), "launch success")
        activation_path = Path(config.evidence_root) / "activation.json"
        activation = _mapping(
            common._read_json_file(activation_path), "activation receipt"
        )
        common._validate_spawned_identity(process_receipt, pid=pid)
        if (
            launch.get("schema")
            != "rl-quant.top2000-dev.m03r-v15-supervisor-launch-success-v1"
            or launch.get("job_name") != config.job_name
            or launch.get("job_uid") != config.job_uid
            or launch.get("run_id") != config.run_id
            or launch.get("parallelism") != config.parallelism
            or launch.get("gpus_per_worker") != 2
            or launch.get("request_ceiling") != config.parallelism * 2
            or launch.get("capacity_receipt_sha256") != config.capacity_receipt_sha256
            or launch.get("economic_panel_authorized") is not False
            or launch.get("activation_file_sha256") != _file_sha256(activation_path)
            or activation.get("schema")
            != "rl-quant.top2000-dev.m03r-v15-supervisor-activation-v1"
            or activation.get("activation_reconciled_after_transport_error")
            is not False
            or activation.get("activation_retried") is not False
        ):
            return None
        return launch
    except (OSError, ValueError, M03RV15SeadragonLifecycleError):
        return None


def _validated_capacity_launch_success(
    path: Path,
    config: M03RV15CapacityAttachConfig,
    process_receipt: Mapping[str, Any],
    pid: int,
) -> Mapping[str, Any] | None:
    try:
        launch = _mapping(common._read_json_file(path), "capacity launch success")
        activation_path = Path(config.evidence_root) / "activation.json"
        activation = _mapping(
            common._read_json_file(activation_path), "capacity activation receipt"
        )
        common._validate_spawned_identity(process_receipt, pid=pid)
        if (
            launch.get("schema")
            != "rl-quant.top2000-dev.m03r-v15-capacity-launch-success-v1"
            or launch.get("job_name") != config.job_name
            or launch.get("job_uid") != config.job_uid
            or launch.get("run_id") != config.run_id
            or launch.get("parallelism") != 1
            or launch.get("gpus_per_worker") != 2
            or launch.get("request_ceiling") != 2
            or launch.get("training_performed") is not False
            or launch.get("economic_panel_authorized") is not False
            or activation.get("schema")
            != "rl-quant.top2000-dev.m03r-v15-capacity-activation-v1"
            or activation.get("activation_reconciled_after_transport_error")
            is not False
            or activation.get("activation_retried") is not False
        ):
            return None
        return launch
    except (OSError, ValueError, M03RV15SeadragonLifecycleError):
        return None


def spawn_capacity_supervisor(
    config_path: str | Path,
    expected_config_sha256: str,
) -> int:
    config_file = common._regular_no_symlink(
        Path(config_path), label="capacity attach config"
    )
    config = _load_capacity_config(config_file, expected_config_sha256)
    root = common._directory_no_symlink(
        Path(config.evidence_root), label="capacity evidence root"
    )
    python = _validated_host_tool(Path(config.host_python_path), "host Python")
    kubectl = _validated_host_tool(Path(config.kubectl_path), "kubectl")
    common._regular_no_symlink(Path(config.kubeconfig_path), label="kubeconfig")
    source = common._regular_no_symlink(Path(__file__), label="lifecycle source")
    if _file_sha256(source) != config.lifecycle_source_sha256:
        raise M03RV15SeadragonLifecycleError("capacity lifecycle source hash drifted")
    command = (
        str(python),
        str(source),
        "run-capacity",
        "--config",
        str(config_file),
        "--config-sha256",
        expected_config_sha256,
    )
    command_sha = hashlib.sha256(
        b"\0".join(item.encode("utf-8") for item in command)
    ).hexdigest()
    intent_sha = common._exclusive_json(
        root / "spawn-intent.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-capacity-spawn-intent-v1",
            "config_sha256": expected_config_sha256,
            "command_sha256": command_sha,
            "python_sha256": _file_sha256(python),
            "kubectl_sha256": _file_sha256(kubectl),
            "lifecycle_source_sha256": config.lifecycle_source_sha256,
            "kubeconfig_metadata_validated": True,
        },
    )
    descriptor = os.open(
        root / "supervisor.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    environment = {
        "KUBECONFIG": str(config.kubeconfig_path),
        "PATH": str(kubectl.parent),
        "PYTHONPATH": str(config.pythonpath),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
    finally:
        os.close(descriptor)
    identity = common._process_identity(process.pid)
    process_receipt = {
        "schema": "rl-quant.top2000-dev.m03r-v15-capacity-process-v1",
        "spawn_intent_file_sha256": intent_sha,
        "config_sha256": expected_config_sha256,
        "command_sha256": command_sha,
        **identity,
    }
    common._validate_spawned_identity(process_receipt, pid=process.pid)
    common._exclusive_json(root / "spawn-process.json", process_receipt)
    deadline = time.monotonic() + config.handshake_timeout_seconds
    launch_path = root / "launch-success.json"
    attach_paths = (
        root / "activation-attach-required.json",
        root / "preactivation-cleanup-attach-required.json",
        root / "cleanup-attach-required.json",
        root / "capacity-recovery-attach-required.json",
    )
    while True:
        if any(path.exists() for path in attach_paths):
            raise M03RV15ActivationAttachRequired(
                "detached capacity supervisor retained ambiguous exact state"
            )
        if process.poll() is not None:
            raise M03RV15SeadragonLifecycleError(
                "capacity supervisor exited before launch success"
            )
        if (
            launch_path.exists()
            and _validated_capacity_launch_success(
                launch_path, config, process_receipt, process.pid
            )
            is not None
        ):
            return process.pid
        if time.monotonic() >= deadline:
            common._exclusive_json(
                root / "spawn-attach-required.json",
                {
                    "schema": (
                        "rl-quant.top2000-dev.m03r-v15-capacity-spawn-attach-required-v1"
                    ),
                    "pid": process.pid,
                    "process_receipt_file_sha256": _file_sha256(
                        root / "spawn-process.json"
                    ),
                    "cleanup_performed": False,
                    "attach_required": True,
                },
            )
            raise M03RV15ActivationAttachRequired(
                "capacity launch handshake timed out; child retained"
            )
        time.sleep(0.25)


def spawn_attach_supervisor(
    config_path: str | Path,
    expected_config_sha256: str,
) -> int:
    config_file = common._regular_no_symlink(Path(config_path), label="attach config")
    config = _load_config(config_file, expected_config_sha256)
    root = common._directory_no_symlink(
        Path(config.evidence_root), label="evidence root"
    )
    python = _validated_host_tool(Path(config.host_python_path), "host Python")
    kubectl = _validated_host_tool(Path(config.kubectl_path), "kubectl")
    common._regular_no_symlink(Path(config.kubeconfig_path), label="kubeconfig")
    source = common._regular_no_symlink(Path(__file__), label="lifecycle source")
    if _file_sha256(source) != config.lifecycle_source_sha256:
        raise M03RV15SeadragonLifecycleError("lifecycle source hash drifted")
    command = (
        str(python),
        str(source),
        "run",
        "--config",
        str(config_file),
        "--config-sha256",
        expected_config_sha256,
    )
    command_sha = hashlib.sha256(
        b"\0".join(item.encode("utf-8") for item in command)
    ).hexdigest()
    intent_sha = common._exclusive_json(
        root / "spawn-intent.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v15-supervisor-spawn-intent-v1",
            "config_sha256": expected_config_sha256,
            "command_sha256": command_sha,
            "python_sha256": _file_sha256(python),
            "kubectl_sha256": _file_sha256(kubectl),
            "lifecycle_source_sha256": config.lifecycle_source_sha256,
            "kubeconfig_metadata_validated": True,
        },
    )
    descriptor = os.open(
        root / "supervisor.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    environment = {
        "KUBECONFIG": str(config.kubeconfig_path),
        "PATH": str(kubectl.parent),
        "PYTHONPATH": str(config.pythonpath),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
    finally:
        os.close(descriptor)
    identity = common._process_identity(process.pid)
    process_receipt = {
        "schema": "rl-quant.top2000-dev.m03r-v15-supervisor-process-v1",
        "spawn_intent_file_sha256": intent_sha,
        "config_sha256": expected_config_sha256,
        "command_sha256": command_sha,
        **identity,
    }
    common._validate_spawned_identity(process_receipt, pid=process.pid)
    common._exclusive_json(root / "spawn-process.json", process_receipt)
    deadline = time.monotonic() + config.handshake_timeout_seconds
    launch_path = root / "launch-success.json"
    attach_path = root / "activation-attach-required.json"
    while True:
        if attach_path.exists():
            raise M03RV15ActivationAttachRequired(
                "detached supervisor retained an ambiguous activation"
            )
        if process.poll() is not None:
            raise M03RV15SeadragonLifecycleError(
                "attach-only supervisor exited before launch success"
            )
        if (
            launch_path.exists()
            and _validated_launch_success(
                launch_path, config, process_receipt, process.pid
            )
            is not None
        ):
            return process.pid
        if time.monotonic() >= deadline:
            common._exclusive_json(
                root / "spawn-attach-required.json",
                {
                    "schema": (
                        "rl-quant.top2000-dev.m03r-v15-spawn-attach-required-v1"
                    ),
                    "pid": process.pid,
                    "process_receipt_file_sha256": _file_sha256(
                        root / "spawn-process.json"
                    ),
                    "cleanup_performed": False,
                    "attach_required": True,
                },
            )
            raise M03RV15ActivationAttachRequired(
                "attach-only launch handshake timed out; child retained"
            )
        time.sleep(0.25)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("spawn", "run", "spawn-capacity", "run-capacity"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "spawn":
            pid = spawn_attach_supervisor(args.config, args.config_sha256)
            print(
                json.dumps(
                    {"status": "launched", "pid": pid},
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        elif args.command == "run":
            run_attach_supervisor(args.config, args.config_sha256)
        elif args.command == "spawn-capacity":
            pid = spawn_capacity_supervisor(args.config, args.config_sha256)
            print(
                json.dumps(
                    {"status": "launched", "pid": pid, "mode": "capacity"},
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        else:
            run_capacity_supervisor(args.config, args.config_sha256)
    except M03RV15ActivationAttachRequired as exc:
        print(
            json.dumps(
                {"status": "attach_required", "error": str(exc)},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTACH_CONFIG_SCHEMA",
    "COMPLETION_COVERAGE_SCHEMA",
    "M03RV15ActivationAttachRequired",
    "M03RV15AttachSupervisorConfig",
    "M03RV15CapacityAttachConfig",
    "M03RV15ExpectedCompletion",
    "M03RV15SeadragonLifecycleError",
    "run_attach_supervisor",
    "run_capacity_supervisor",
    "spawn_attach_supervisor",
    "spawn_capacity_supervisor",
    "validate_m03r_v15_predictive_coverage",
]
