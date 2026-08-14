"""Attach-only lifecycle for the frozen M03R-v12 post-hoc audit.

This operator cannot create, apply, or replace a Kubernetes object.  It may
only activate one already-bound suspended Job, observe that exact UID and its
owned Pods, validate the three immutable audit terminals, and exact-clean the
Job with UID/resourceVersion preconditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from rl_quant.evaluation.top2000_m03r_v12_posthoc_inference_audit import (
    M03RV12PosthocAuditPanelReport,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_posthoc_inference_audit import (
    M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
    M03R_V12_POSTHOC_AUDIT_VARIANTS,
)
from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as common
from rl_quant.training import (
    top2000_m03r_v11_a15_inference_audit_lifecycle as audit_common,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_KUBERNETES_CONTEXT,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03RV7ExactJobActivationRequest,
    M03RV7ExactJobCleanupRequest,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_kubernetes import (
    M03R_V12_POSTHOC_AUDIT_RENDERED_JOB_SCHEMA,
    M03RV12PosthocAuditRenderedJob,
)
SEADRAGON_KUBECTL: Final = "/risapps/noarch/kubectl/1.28.4/bin/kubectl"
SEADRAGON_KUBECONFIG: Final = "/rsrch8/home/bcb/yding4/.kube/config"
SEADRAGON_QUANTTRADE_ROOT: Final = "/rsrch8/home/bcb/yding4/quant/training"
M03R_V12_POSTHOC_AUDIT_WORKER_TERMINAL_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-worker-terminal-v1"
)
M03R_V12_POSTHOC_AUDIT_ATTACH_CONFIG_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-attach-config-v1"
)
M03R_V12_POSTHOC_AUDIT_LAUNCH_SUCCESS_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-launch-success-v1"
)
M03R_V12_POSTHOC_AUDIT_FINAL_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-final-v1"
)
_SHA256 = frozenset("0123456789abcdef")


class M03RV12PosthocAuditLifecycleError(RuntimeError):
    """The exact attach-only audit lifecycle drifted or failed."""


class M03RV12PosthocAuditAttachRequired(M03RV12PosthocAuditLifecycleError):
    """Activation may have occurred and must not be retried blindly."""


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in _SHA256 for character in value):
        raise M03RV12PosthocAuditLifecycleError(
            f"{name} must be one lowercase SHA-256"
        )


def _project_path(value: str, name: str) -> Path:
    path = Path(value)
    root = Path(SEADRAGON_QUANTTRADE_ROOT)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise M03RV12PosthocAuditLifecycleError(
            f"{name} must stay under the approved QuantTrade root"
        )
    return path


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV12PosthocAuditAttachConfig:
    job_name: str
    run_id: str
    job_uid: str
    rendered_path: str
    rendered_file_sha256: str
    binding_path: str
    binding_file_sha256: str
    activation_request_path: str
    activation_request_file_sha256: str
    output_root: str
    evidence_root: str
    phase_receipt_output_path: str
    audit_package_plan_sha256: str
    audit_package_plan_file_sha256: str
    source_archive_sha256: str
    capacity_receipt_sha256: str
    image_digest_sha256: str
    lifecycle_source_sha256: str
    host_python_path: str
    pythonpath: str
    completions: int = 3
    parallelism: int = 3
    gpus_per_completion: int = 1
    request_timeout_seconds: int = 30
    poll_interval_seconds: int = 10
    hard_wall_seconds: int = 86_400
    log_limit_bytes: int = 65_536
    handshake_timeout_seconds: int = 30
    kubectl_path: str = SEADRAGON_KUBECTL
    kubeconfig_path: str = SEADRAGON_KUBECONFIG
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    schema: str = M03R_V12_POSTHOC_AUDIT_ATTACH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "rendered_file_sha256",
            "binding_file_sha256",
            "activation_request_file_sha256",
            "audit_package_plan_sha256",
            "audit_package_plan_file_sha256",
            "source_archive_sha256",
            "capacity_receipt_sha256",
            "image_digest_sha256",
            "lifecycle_source_sha256",
        ):
            _require_sha256(name, cast(str, getattr(self, name)))
        if (
            self.schema != M03R_V12_POSTHOC_AUDIT_ATTACH_CONFIG_SCHEMA
            or not self.job_name
            or not self.run_id
            or not self.job_uid
            or (self.completions, self.parallelism, self.gpus_per_completion)
            != (3, 3, 1)
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.kubectl_path != SEADRAGON_KUBECTL
            or self.kubeconfig_path != SEADRAGON_KUBECONFIG
            or self.request_timeout_seconds < 5
            or self.poll_interval_seconds < 1
            or not 60 <= self.hard_wall_seconds <= 216_000
            or not 4_096 <= self.log_limit_bytes <= 1_048_576
            or not 5 <= self.handshake_timeout_seconds <= 120
        ):
            raise M03RV12PosthocAuditLifecycleError(
                "v12 post-hoc attach config drifted"
            )
        for name in (
            "rendered_path",
            "binding_path",
            "activation_request_path",
            "output_root",
            "evidence_root",
            "phase_receipt_output_path",
            "pythonpath",
        ):
            _project_path(cast(str, getattr(self, name)), name)


class AuditTransport(Protocol):
    def get_job(self, *, allow_absent: bool = False) -> Mapping[str, Any] | None: ...

    def get_owned_pods(self) -> tuple[Mapping[str, Any], ...]: ...

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes: ...

    def activate(
        self, request: M03RV7ExactJobActivationRequest
    ) -> Mapping[str, Any]: ...

    def delete(
        self, request: M03RV7ExactJobCleanupRequest, options_path: Path
    ) -> None: ...


def _load_config(
    path: Path, expected_sha256: str
) -> M03RV12PosthocAuditAttachConfig:
    value, _ = audit_common._read_stable_json(path, expected_sha256=expected_sha256)
    try:
        return M03RV12PosthocAuditAttachConfig(**value)
    except (TypeError, ValueError) as exc:
        raise M03RV12PosthocAuditLifecycleError(
            "v12 post-hoc attach config is invalid"
        ) from exc


def _load_rendered(
    config: M03RV12PosthocAuditAttachConfig,
) -> M03RV12PosthocAuditRenderedJob:
    value, _ = audit_common._read_stable_json(
        Path(config.rendered_path), expected_sha256=config.rendered_file_sha256
    )
    try:
        rendered = M03RV12PosthocAuditRenderedJob(**value)
        rendered.validate()
    except (TypeError, ValueError) as exc:
        raise M03RV12PosthocAuditLifecycleError(
            "v12 post-hoc rendered Job is invalid"
        ) from exc
    annotations = cast(
        Mapping[str, Any], rendered.manifest["metadata"].get("annotations", {})
    )
    if (
        rendered.schema != M03R_V12_POSTHOC_AUDIT_RENDERED_JOB_SCHEMA
        or rendered.mode != "audit"
        or rendered.completions != 3
        or rendered.parallelism != 3
        or rendered.gpus_per_completion != 1
        or rendered.audit_package_plan_sha256
        != config.audit_package_plan_sha256
        or rendered.audit_package_plan_file_sha256
        != config.audit_package_plan_file_sha256
        or rendered.capacity_receipt_sha256 != config.capacity_receipt_sha256
        or annotations.get("rl-quant/source-archive-sha256")
        != config.source_archive_sha256
        or annotations.get("rl-quant/training-authorized") != "false"
        or annotations.get("rl-quant/checkpoint-selection-authorized") != "false"
        or annotations.get("rl-quant/economic-training-authorized") != "false"
        or annotations.get("rl-quant/outer-2026-access-authorized") != "false"
    ):
        raise M03RV12PosthocAuditLifecycleError(
            "v12 post-hoc rendered Job and attach config drifted"
        )
    return rendered


def _job_identity(
    job: Mapping[str, Any], config: M03RV12PosthocAuditAttachConfig
) -> None:
    common._job_identity(
        job,
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=config.job_uid,
    )
    metadata = cast(Mapping[str, Any], job.get("metadata", {}))
    annotations = cast(Mapping[str, Any], metadata.get("annotations", {}))
    expected = {
        "rl-quant/package-plan-sha256": config.audit_package_plan_sha256,
        "rl-quant/source-archive-sha256": config.source_archive_sha256,
        "rl-quant/capacity-receipt-sha256": config.capacity_receipt_sha256,
        "rl-quant/training-authorized": "false",
        "rl-quant/checkpoint-selection-authorized": "false",
        "rl-quant/economic-training-authorized": "false",
        "rl-quant/outer-2026-access-authorized": "false",
    }
    if any(annotations.get(name) != value for name, value in expected.items()):
        raise M03RV12PosthocAuditLifecycleError(
            "live v12 post-hoc Job artifact identity drifted"
        )


def _validate_worker_outputs(
    config: M03RV12PosthocAuditAttachConfig,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    expected_variants = {variant.variant_id for variant in M03R_V12_POSTHOC_AUDIT_VARIANTS}
    for setting_index in range(3):
        root = (
            Path(config.output_root)
            / f"completion-{setting_index:02d}-setting-{setting_index:02d}"
        )
        terminal_path = root / "audit-terminal.json"
        terminal, terminal_file_sha = audit_common._read_stable_json(terminal_path)
        receipt = audit_common._semantic_receipt(
            terminal, label="v12 post-hoc audit worker terminal"
        )
        folds = terminal.get("fold_artifact_file_sha256")
        reports = terminal.get("panel_reports")
        if (
            terminal.get("schema") != M03R_V12_POSTHOC_AUDIT_WORKER_TERMINAL_SCHEMA
            or terminal.get("protocol_sha256")
            != M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
            or terminal.get("audit_package_plan_sha256")
            != config.audit_package_plan_sha256
            or terminal.get("audit_package_plan_file_sha256")
            != config.audit_package_plan_file_sha256
            or terminal.get("setting_index") != setting_index
            or terminal.get("visible_device_count") != 1
            or terminal.get("device_name") != "NVIDIA H100 80GB HBM3"
            or terminal.get("exact_one_h100_80gb") is not True
            or not isinstance(folds, list)
            or len(folds) != 6
            or not isinstance(reports, list)
            or len(reports) != len(expected_variants)
            or terminal.get("training_performed") is not False
            or terminal.get("checkpoint_selection_performed") is not False
            or terminal.get("economic_optimizer_updates") != 0
            or terminal.get("economic_generation_may_be_minted") is not False
            or terminal.get("outer_2026_accessed") is not False
            or terminal.get("posthoc_exploratory") is not True
            or terminal.get("reportable") is not False
            or terminal.get("promotion_eligible") is not False
        ):
            raise M03RV12PosthocAuditLifecycleError(
                "v12 post-hoc worker terminal semantics drifted"
            )
        for fold_index, expected_file_sha in enumerate(folds):
            if not isinstance(expected_file_sha, str):
                raise M03RV12PosthocAuditLifecycleError(
                    "v12 post-hoc fold inventory drifted"
                )
            _require_sha256("fold artifact file", expected_file_sha)
            artifact = root / "fold-artifacts" / f"fold-{fold_index:02d}.pt"
            audit_common._stable_file_sha256(
                artifact, expected_sha256=expected_file_sha
            )
            file_hashes[str(artifact)] = expected_file_sha
        observed_variants: set[str] = set()
        for raw_report in reports:
            if not isinstance(raw_report, Mapping):
                raise M03RV12PosthocAuditLifecycleError(
                    "v12 post-hoc panel report is not an object"
                )
            try:
                report = M03RV12PosthocAuditPanelReport(**dict(raw_report))
                report.validate()
            except (TypeError, ValueError) as exc:
                raise M03RV12PosthocAuditLifecycleError(
                    "v12 post-hoc panel report drifted"
                ) from exc
            if report.setting_index != setting_index or report.variant_id in observed_variants:
                raise M03RV12PosthocAuditLifecycleError(
                    "v12 post-hoc panel identity drifted"
                )
            observed_variants.add(report.variant_id)
        if observed_variants != expected_variants:
            raise M03RV12PosthocAuditLifecycleError(
                "v12 post-hoc panel coverage drifted"
            )
        rows.append(
            {
                "setting_index": setting_index,
                "terminal_file_sha256": terminal_file_sha,
                "terminal_receipt_sha256": receipt,
            }
        )
        file_hashes[str(terminal_path)] = terminal_file_sha
    return rows, file_hashes


def _write_phase_receipt(
    config: M03RV12PosthocAuditAttachConfig, payload: dict[str, Any]
) -> str:
    unsigned = {
        **payload,
        "job_name": config.job_name,
        "job_uid": config.job_uid,
        "run_id": config.run_id,
        "audit_package_plan_sha256": config.audit_package_plan_sha256,
        "audit_package_plan_file_sha256": config.audit_package_plan_file_sha256,
        "source_archive_sha256": config.source_archive_sha256,
        "capacity_receipt_sha256": config.capacity_receipt_sha256,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_optimizer_updates": 0,
        "economic_generation_may_be_minted": False,
        "outer_2026_accessed": False,
        "development_only": True,
        "posthoc_exploratory": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    value = {**unsigned, "receipt_sha256": _content_sha256(unsigned)}
    return common._exclusive_json(Path(config.phase_receipt_output_path), value)


def run_m03r_v12_posthoc_audit_attach_lifecycle(
    config_path: str | Path,
    expected_config_sha256: str,
    *,
    transport: AuditTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    config = _load_config(Path(config_path), expected_config_sha256)
    evidence_path = Path(config.evidence_root)
    spawned = evidence_path.exists()
    root = (
        common._directory_no_symlink(evidence_path, label="audit evidence root")
        if spawned
        else common._create_evidence_root(evidence_path)
    )
    if spawned:
        deadline = monotonic() + config.handshake_timeout_seconds
        process_path = root / "spawn-process.json"
        while not process_path.exists():
            if monotonic() >= deadline:
                raise M03RV12PosthocAuditLifecycleError(
                    "v12 post-hoc spawn process receipt was not published"
                )
            sleep(0.1)
        process_value, _ = audit_common._read_stable_json(process_path)
        common._validate_spawned_identity(process_value, pid=os.getpid())
    source = common._regular_no_symlink(Path(__file__), label="audit lifecycle source")
    if common._file_sha256(source) != config.lifecycle_source_sha256:
        raise M03RV12PosthocAuditLifecycleError(
            "v12 post-hoc lifecycle source SHA-256 drifted"
        )
    rendered = _load_rendered(config)
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
        or binding.parallelism != 3
        or binding.desired_manifest_sha256 != rendered.manifest_sha256
    ):
        raise M03RV12PosthocAuditLifecycleError(
            "v12 post-hoc binding and attach config drifted"
        )
    live = transport or audit_common.AuditAttachOnlyKubectl(
        container_name="auditor",
        kubectl_path=config.kubectl_path,
        kubeconfig_path=config.kubeconfig_path,
        context=config.context,
        namespace=config.namespace,
        job_name=config.job_name,
        job_uid=config.job_uid,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    fresh = live.get_job()
    if fresh is None:
        raise M03RV12PosthocAuditLifecycleError(
            "bound v12 post-hoc Job is absent before activation"
        )
    _job_identity(fresh, config)
    if live.get_owned_pods():
        raise M03RV12PosthocAuditLifecycleError(
            "bound suspended v12 post-hoc Job unexpectedly owns Pods"
        )
    runtime_activation = build_m03r_v7_exact_job_activation_request(binding, fresh)
    if not common._same_activation_contract(
        configured_activation, runtime_activation
    ):
        raise M03RV12PosthocAuditLifecycleError(
            "fresh v12 post-hoc activation identity drifted"
        )
    common._exclusive_json(
        root / "activation-request-runtime.json", asdict(runtime_activation)
    )
    try:
        activated = live.activate(runtime_activation)
    except Exception as exc:  # noqa: BLE001 - ambiguous activation must attach
        common._exclusive_json(
            root / "activation-attach-required.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v12-posthoc-activation-attach-v1",
                "job_name": config.job_name,
                "job_uid": config.job_uid,
                "run_id": config.run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "activation_retried": False,
                "cleanup_performed": False,
                "attach_required": True,
            },
        )
        raise M03RV12PosthocAuditAttachRequired(
            "v12 post-hoc activation is ambiguous; never retry activation"
        ) from exc
    _job_identity(activated, config)
    if cast(Mapping[str, Any], activated.get("spec", {})).get("suspend") is not False:
        raise M03RV12PosthocAuditAttachRequired(
            "v12 post-hoc activation response did not prove unsuspended state"
        )
    activation_sha = common._exclusive_json(root / "activation.json", activated)
    common._exclusive_json(
        root / "launch-success.json",
        {
            "schema": M03R_V12_POSTHOC_AUDIT_LAUNCH_SUCCESS_SCHEMA,
            "job_name": config.job_name,
            "job_uid": config.job_uid,
            "run_id": config.run_id,
            "parallelism": 3,
            "gpus_per_completion": 1,
            "request_ceiling_h100": 3,
            "activation_file_sha256": activation_sha,
            "activation_request_sha256": runtime_activation.request_sha256,
            "activated": True,
            "detached_supervisor_required": True,
            "training_authorized": False,
            "outer_2026_access_authorized": False,
        },
    )
    deadline = monotonic() + config.hard_wall_seconds
    terminal_job: Mapping[str, Any] | None = None
    terminal_state: str | None = None
    terminal_pods: tuple[Mapping[str, Any], ...] = ()
    while monotonic() < deadline:
        job = live.get_job()
        if job is None:
            raise M03RV12PosthocAuditLifecycleError(
                "activated v12 post-hoc Job disappeared before terminal evidence"
            )
        _job_identity(job, config)
        condition = common._true_condition(job)
        if condition is not None:
            terminal_job = job
            terminal_state = condition
            terminal_pods = live.get_owned_pods()
            break
        sleep(config.poll_interval_seconds)
    if terminal_job is None or terminal_state is None:
        common._exclusive_json(
            root / "timeout-attach-required.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v12-posthoc-timeout-attach-v1",
                "job_name": config.job_name,
                "job_uid": config.job_uid,
                "hard_wall_seconds": config.hard_wall_seconds,
                "cleanup_performed": False,
                "attach_required": True,
            },
        )
        raise M03RV12PosthocAuditAttachRequired(
            "v12 post-hoc Job exceeded its hard wall; retain for attachment"
        )
    terminal_phase = "Succeeded" if terminal_state == "Complete" else "Failed"
    common._capture_terminal(
        root=root,
        job=terminal_job,
        pods=terminal_pods,
        transport=cast(Any, live),
        reason=f"v12-posthoc-{terminal_state.lower()}",
        log_limit_bytes=config.log_limit_bytes,
    )
    if terminal_state != "Complete":
        common._cleanup_exact_job(
            root=root, binding=binding, transport=cast(Any, live), sleep=sleep
        )
        raise M03RV12PosthocAuditLifecycleError(
            "v12 post-hoc Job failed; evidence preserved and exact cleanup completed"
        )
    pod_rows = audit_common._pod_rows(
        cast(Any, config), terminal_pods, terminal_phase=terminal_phase
    )
    outputs, output_hashes = _validate_worker_outputs(config)
    phase_payload = {
        "schema": M03R_V12_POSTHOC_AUDIT_FINAL_SCHEMA,
        "worker_outputs": outputs,
        "output_file_sha256": output_hashes,
        "pod_runtime_proof": pod_rows,
        "completion_count": 3,
        "passed": True,
    }
    common._cleanup_exact_job(
        root=root, binding=binding, transport=cast(Any, live), sleep=sleep
    )
    cleanup_path = root / "cleanup-receipt.json"
    cleanup, cleanup_sha = audit_common._read_stable_json(cleanup_path)
    if (
        cleanup.get("first_job_absent") is not True
        or cleanup.get("second_job_absent") is not True
        or cleanup.get("first_owned_pod_uids") != []
        or cleanup.get("second_owned_pod_uids") != []
    ):
        raise M03RV12PosthocAuditLifecycleError(
            "v12 post-hoc exact cleanup receipt drifted"
        )
    phase_payload["cleanup_receipt_file_sha256"] = cleanup_sha
    _write_phase_receipt(config, phase_payload)


def spawn_m03r_v12_posthoc_audit_attach_lifecycle(
    *,
    config_path: Path,
    config_sha256: str,
    wait: Callable[[float], None] = time.sleep,
) -> int:
    """Spawn the attach-only owner and return after exact activation proof."""

    config = _load_config(config_path, config_sha256)
    root = common._create_evidence_root(Path(config.evidence_root))
    python = common._regular_no_symlink(
        Path(config.host_python_path), label="v12 post-hoc host Python"
    )
    pythonpath = common._directory_no_symlink(
        Path(config.pythonpath), label="v12 post-hoc PYTHONPATH"
    )
    kubectl = common._regular_no_symlink(Path(config.kubectl_path), label="kubectl")
    kubeconfig = common._regular_no_symlink(
        Path(config.kubeconfig_path), label="kubeconfig"
    )
    source = common._regular_no_symlink(Path(__file__), label="audit lifecycle source")
    if common._file_sha256(source) != config.lifecycle_source_sha256:
        raise M03RV12PosthocAuditLifecycleError(
            "v12 post-hoc lifecycle source drifted before spawn"
        )
    command = (
        str(python),
        "-m",
        "rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_lifecycle",
        "run",
        "--config",
        str(config_path),
        "--config-sha256",
        config_sha256,
    )
    command_sha = hashlib.sha256(
        b"\0".join(item.encode("utf-8") for item in command)
    ).hexdigest()
    intent_sha = common._exclusive_json(
        root / "spawn-intent.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v12-posthoc-spawn-intent-v1",
            "config_sha256": config_sha256,
            "command_sha256": command_sha,
            "python_sha256": common._file_sha256(python),
            "kubectl_sha256": common._file_sha256(kubectl),
            "supervisor_source_sha256": common._file_sha256(source),
            "kubeconfig_metadata_validated": kubeconfig.is_file(),
            "create_authorized": False,
            "apply_authorized": False,
            "replace_authorized": False,
        },
    )
    log_descriptor = os.open(
        root / "supervisor.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440
    )
    environment = {
        "KUBECONFIG": str(kubeconfig),
        "PATH": str(kubectl.parent),
        "PYTHONPATH": str(pythonpath),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
    finally:
        os.close(log_descriptor)
    identity = common._process_identity(process.pid)
    process_receipt = {
        "schema": "rl-quant.top2000-dev.m03r-v12-posthoc-process-v1",
        "spawn_intent_file_sha256": intent_sha,
        "config_sha256": config_sha256,
        "command_sha256": command_sha,
        **identity,
    }
    common._validate_spawned_identity(process_receipt, pid=process.pid)
    common._exclusive_json(root / "spawn-process.json", process_receipt)
    deadline = time.monotonic() + config.handshake_timeout_seconds
    launch_path = root / "launch-success.json"
    while True:
        if process.poll() is not None:
            raise M03RV12PosthocAuditLifecycleError(
                "v12 post-hoc supervisor exited before activation proof"
            )
        if launch_path.exists():
            launch, _ = audit_common._read_stable_json(launch_path)
            if (
                launch.get("schema")
                != M03R_V12_POSTHOC_AUDIT_LAUNCH_SUCCESS_SCHEMA
                or launch.get("job_name") != config.job_name
                or launch.get("job_uid") != config.job_uid
                or launch.get("run_id") != config.run_id
                or launch.get("parallelism") != 3
                or launch.get("gpus_per_completion") != 1
                or launch.get("request_ceiling_h100") != 3
                or launch.get("activated") is not True
                or launch.get("training_authorized") is not False
                or launch.get("outer_2026_access_authorized") is not False
            ):
                raise M03RV12PosthocAuditLifecycleError(
                    "v12 post-hoc launch-success handshake drifted"
                )
            common._validate_spawned_identity(process_receipt, pid=process.pid)
            return process.pid
        if time.monotonic() >= deadline:
            raise M03RV12PosthocAuditLifecycleError(
                "v12 post-hoc activation handshake timed out"
            )
        wait(0.25)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "spawn"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "spawn":
        spawn_m03r_v12_posthoc_audit_attach_lifecycle(
            config_path=Path(arguments.config),
            config_sha256=arguments.config_sha256,
        )
        return 0
    try:
        run_m03r_v12_posthoc_audit_attach_lifecycle(
            arguments.config, arguments.config_sha256
        )
    except M03RV12PosthocAuditAttachRequired as exc:
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
    "M03R_V12_POSTHOC_AUDIT_ATTACH_CONFIG_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_FINAL_SCHEMA",
    "M03R_V12_POSTHOC_AUDIT_LAUNCH_SUCCESS_SCHEMA",
    "M03RV12PosthocAuditAttachConfig",
    "M03RV12PosthocAuditAttachRequired",
    "M03RV12PosthocAuditLifecycleError",
    "main",
    "run_m03r_v12_posthoc_audit_attach_lifecycle",
    "spawn_m03r_v12_posthoc_audit_attach_lifecycle",
]
