"""Attach-only lifecycle for the frozen M03R-v11 a15 inference audit.

This module has no Job create, apply, or replace surface.  It activates one
already-bound suspended Job, validates exact static/capacity/audit artifacts,
captures terminal Job/Pod/log evidence, and exact-cleans the bound UID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as common
from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS,
    M03R_V11_A15_AUDIT_CAPACITY_TERMINAL_SCHEMA,
    M03R_V11_A15_AUDIT_COST_BASIS_POINTS,
    M03R_V11_A15_AUDIT_HORIZONS,
    M03R_V11_A15_AUDIT_PANEL_SCHEMA,
    M03R_V11_A15_AUDIT_QUANTILE_COUNTS,
    M03R_V11_A15_AUDIT_STATIC_TERMINAL_SCHEMA,
    M03R_V11_A15_AUDIT_STARTUP_SCHEMA,
    M03R_V11_A15_AUDIT_VARIANTS,
    M03R_V11_A15_AUDIT_WORKER_TERMINAL_SCHEMA,
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_KUBERNETES_CONTEXT,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03RV7ExactJobActivationRequest,
    M03RV7ExactJobCleanupRequest,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_kubernetes import (
    M03R_V11_A15_AUDIT_CAPACITY_SCHEMA,
    M03R_V11_A15_AUDIT_RENDERED_JOB_SCHEMA,
    M03RV11A15AuditRenderedJob,
)

SEADRAGON_KUBECTL: Final = "/risapps/noarch/kubectl/1.28.4/bin/kubectl"
SEADRAGON_KUBECONFIG: Final = "/rsrch8/home/bcb/yding4/.kube/config"
SEADRAGON_QUANTTRADE_ROOT: Final = "/rsrch8/home/bcb/yding4/quant/training"
M03R_V11_A15_AUDIT_ATTACH_CONFIG_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-attach-config-v1"
)
M03R_V11_A15_AUDIT_STATIC_GATE_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-static-gate-v1"
)
M03R_V11_A15_AUDIT_FINAL_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-final-v1"
)
M03R_V11_A15_AUDIT_LAUNCH_SUCCESS_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-launch-success-v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024


class M03RV11A15InferenceAuditLifecycleError(RuntimeError):
    """The bound audit lifecycle or result evidence failed closed."""


class M03RV11A15AuditActivationAttachRequired(M03RV11A15InferenceAuditLifecycleError):
    """Activation may have occurred and must be reconciled without retry."""


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise M03RV11A15InferenceAuditLifecycleError(
            f"{name} must be one lowercase SHA-256"
        )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _project_path(value: str, name: str) -> Path:
    path = Path(value)
    root = Path(SEADRAGON_QUANTTRADE_ROOT)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise M03RV11A15InferenceAuditLifecycleError(
            f"{name} must stay under the approved QuantTrade root"
        )
    return path


def _read_stable_json(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV11A15InferenceAuditLifecycleError(
            f"cannot open bound JSON: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= _MAX_JSON_BYTES
        ):
            raise M03RV11A15InferenceAuditLifecycleError(
                f"bound JSON type or size is invalid: {path}"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV11A15InferenceAuditLifecycleError(
                f"bound JSON changed while reading: {path}"
            )
    finally:
        os.close(descriptor)
    file_sha256 = digest.hexdigest()
    if expected_sha256 is not None and file_sha256 != expected_sha256:
        raise M03RV11A15InferenceAuditLifecycleError(
            f"bound JSON file SHA-256 drifted: {path}"
        )
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M03RV11A15InferenceAuditLifecycleError(
            f"bound JSON is invalid: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise M03RV11A15InferenceAuditLifecycleError(
            f"bound JSON is not an object: {path}"
        )
    return dict(value), file_sha256


def _stable_file_sha256(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int = _MAX_ARTIFACT_BYTES,
) -> str:
    _require_sha256("bound artifact file", expected_sha256)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV11A15InferenceAuditLifecycleError(
            f"cannot open bound artifact: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
            raise M03RV11A15InferenceAuditLifecycleError(
                f"bound artifact type or size is invalid: {path}"
            )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV11A15InferenceAuditLifecycleError(
                f"bound artifact changed while reading: {path}"
            )
    finally:
        os.close(descriptor)
    file_sha256 = digest.hexdigest()
    if file_sha256 != expected_sha256:
        raise M03RV11A15InferenceAuditLifecycleError(
            f"bound artifact file SHA-256 drifted: {path}"
        )
    return file_sha256


def _semantic_receipt(value: Mapping[str, Any], *, label: str) -> str:
    receipt = value.get("receipt_sha256")
    if not isinstance(receipt, str):
        raise M03RV11A15InferenceAuditLifecycleError(
            f"{label} omitted its semantic receipt"
        )
    _require_sha256(label, receipt)
    unsigned = {key: row for key, row in value.items() if key != "receipt_sha256"}
    if receipt != _content_sha256(unsigned):
        raise M03RV11A15InferenceAuditLifecycleError(
            f"{label} semantic receipt drifted"
        )
    return receipt


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _number_between(value: Any, lower: float, upper: float) -> bool:
    return _finite_number(value) and lower <= float(cast(int | float, value)) <= upper


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(cast(int | float, value)) > 0.0


def _metric_rows(value: Any, expected_keys: tuple[int, ...]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected_keys)
        and all(
            isinstance(row, list)
            and len(row) == 2
            and row[0] == key
            and _finite_number(row[1])
            for row, key in zip(value, expected_keys, strict=True)
        )
    )


def _nested_metric_rows(
    value: Any,
    outer_keys: tuple[int, ...],
    inner_keys: tuple[int, ...],
) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(outer_keys)
        and all(
            isinstance(row, list)
            and len(row) == 2
            and row[0] == key
            and _metric_rows(row[1], inner_keys)
            for row, key in zip(value, outer_keys, strict=True)
        )
    )


def _validate_panel_report(
    value: Mapping[str, Any],
    *,
    setting_index: int,
    horizon_sessions: int,
    variant_id: str,
    expected_receipt_sha256: str,
) -> None:
    receipt = _semantic_receipt(value, label="audit panel report")
    fold_receipts = value.get("fold_receipt_sha256")
    metrics = (
        value.get("annualized_gross_active_return"),
        value.get("mean_action_cap_hit_fraction"),
        value.get("mean_score_to_action_spearman"),
        value.get("mean_brier_probability_beats_10bp"),
        value.get("mean_ece_probability_beats_10bp"),
        value.get("annualized_carry_active_return"),
        value.get("annualized_anchor_repair_active_return"),
        value.get("annualized_alpha_signal_active_return"),
    )
    category = value.get("break_even_category")
    break_even = value.get("aggregate_break_even_one_way_cost_basis_points")
    if (
        receipt != expected_receipt_sha256
        or value.get("schema") != M03R_V11_A15_AUDIT_PANEL_SCHEMA
        or value.get("protocol_sha256") != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
        or value.get("setting_index") != setting_index
        or not isinstance(value.get("setting_id"), str)
        or not value.get("setting_id")
        or value.get("horizon_sessions") != horizon_sessions
        or value.get("variant_id") != variant_id
        or not isinstance(fold_receipts, list)
        or len(fold_receipts) != 6
        or len(set(fold_receipts)) != 6
        or any(
            not isinstance(row, str) or _SHA256_RE.fullmatch(row) is None
            for row in fold_receipts
        )
        or any(not _finite_number(metric) for metric in metrics)
        or not _number_between(metrics[1], 0.0, 1.0)
        or not _number_between(metrics[2], -1.0, 1.0)
        or not _number_between(metrics[3], 0.0, 1.0)
        or not _number_between(metrics[4], 0.0, 1.0)
        or not _metric_rows(
            value.get("annualized_net_active_return_by_cost"),
            M03R_V11_A15_AUDIT_COST_BASIS_POINTS,
        )
        or not _nested_metric_rows(
            value.get("annualized_lcb_by_block_and_cost"),
            M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS,
            M03R_V11_A15_AUDIT_COST_BASIS_POINTS,
        )
        or not _nested_metric_rows(
            value.get("top_bottom_lcb_by_block_and_quantiles"),
            M03R_V11_A15_AUDIT_BOOTSTRAP_BLOCKS,
            M03R_V11_A15_AUDIT_QUANTILE_COUNTS,
        )
        or category
        not in {
            "finite-positive",
            "favorable-cost-dominance",
            "no-positive-break-even",
        }
        or (category == "finite-positive" and not _positive_number(break_even))
        or (category != "finite-positive" and break_even is not None)
        or value.get("training_performed") is not False
        or value.get("checkpoint_selection_performed") is not False
        or value.get("economic_generation_may_be_minted") is not False
        or value.get("outer_2026_accessed") is not False
    ):
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit panel report semantics drifted"
        )


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditAttachConfig:
    mode: Literal["static", "capacity", "audit"]
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
    package_plan_sha256: str
    authorization_receipt_sha256: str
    audit_plan_receipt_sha256: str
    parent_cleanup_receipt_sha256: str
    source_archive_sha256: str
    image_digest_sha256: str
    lifecycle_source_sha256: str
    completions: int
    parallelism: int
    gpus_per_completion: int
    static_gate_receipt_sha256: str
    capacity_receipt_sha256: str
    phase_receipt_output_path: str
    host_python_path: str
    pythonpath: str
    request_timeout_seconds: int = 30
    poll_interval_seconds: int = 10
    hard_wall_seconds: int = 86_400
    log_limit_bytes: int = 65_536
    handshake_timeout_seconds: int = 30
    kubectl_path: str = SEADRAGON_KUBECTL
    kubeconfig_path: str = SEADRAGON_KUBECONFIG
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    schema: str = M03R_V11_A15_AUDIT_ATTACH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "rendered_file_sha256",
            "binding_file_sha256",
            "activation_request_file_sha256",
            "package_plan_sha256",
            "authorization_receipt_sha256",
            "audit_plan_receipt_sha256",
            "parent_cleanup_receipt_sha256",
            "source_archive_sha256",
            "image_digest_sha256",
            "lifecycle_source_sha256",
        ):
            _require_sha256(name, cast(str, getattr(self, name)))
        if self.mode == "static":
            if self.static_gate_receipt_sha256 != "not-yet-created":
                raise M03RV11A15InferenceAuditLifecycleError(
                    "static phase cannot consume its future receipt"
                )
            expected_capacity = "not-yet-created"
        elif self.mode == "capacity":
            _require_sha256(
                "static_gate_receipt_sha256", self.static_gate_receipt_sha256
            )
            expected_capacity = "not-yet-created"
        else:
            _require_sha256(
                "static_gate_receipt_sha256", self.static_gate_receipt_sha256
            )
            _require_sha256("capacity_receipt_sha256", self.capacity_receipt_sha256)
            expected_capacity = self.capacity_receipt_sha256
        expected_geometry = {
            "static": (1, 1, 0),
            "capacity": (1, 1, 1),
            "audit": (2, 2, 1),
        }
        if (
            self.schema != M03R_V11_A15_AUDIT_ATTACH_CONFIG_SCHEMA
            or self.mode not in expected_geometry
            or (self.completions, self.parallelism, self.gpus_per_completion)
            != expected_geometry[self.mode]
            or self.capacity_receipt_sha256 != expected_capacity
            or not self.job_name
            or not self.run_id
            or not self.job_uid
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.kubectl_path != SEADRAGON_KUBECTL
            or self.kubeconfig_path != SEADRAGON_KUBECONFIG
            or self.request_timeout_seconds < 5
            or self.poll_interval_seconds < 1
            or not 60 <= self.hard_wall_seconds <= 216_000
            or not 4096 <= self.log_limit_bytes <= 1_048_576
            or not 5 <= self.handshake_timeout_seconds <= 120
        ):
            raise M03RV11A15InferenceAuditLifecycleError("audit attach config drifted")
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


class AuditAttachOnlyKubectl(common.AttachOnlyKubectl):
    """The reviewed attach-only transport with the audit container name."""

    def __init__(self, *, container_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if container_name not in {"validator", "auditor"}:
            raise M03RV11A15InferenceAuditLifecycleError(
                "audit log container name drifted"
            )
        self.container_name = container_name

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
        if not pod_name or limit_bytes <= 0:
            raise M03RV11A15InferenceAuditLifecycleError("audit log request is invalid")
        return self._run(
            (
                "logs",
                pod_name,
                "--container",
                self.container_name,
                f"--limit-bytes={limit_bytes}",
            )
        )


def _load_config(path: Path, expected_sha256: str) -> M03RV11A15AuditAttachConfig:
    value, _ = _read_stable_json(path, expected_sha256=expected_sha256)
    try:
        return M03RV11A15AuditAttachConfig(**value)
    except (TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit attach config is invalid"
        ) from exc


def _load_rendered(config: M03RV11A15AuditAttachConfig) -> M03RV11A15AuditRenderedJob:
    value, _ = _read_stable_json(
        Path(config.rendered_path), expected_sha256=config.rendered_file_sha256
    )
    try:
        rendered = M03RV11A15AuditRenderedJob(**value)
        rendered.validate()
    except (TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditLifecycleError(
            "rendered audit Job is invalid"
        ) from exc
    annotations = rendered.manifest.get("metadata", {}).get("annotations", {})
    if (
        rendered.schema != M03R_V11_A15_AUDIT_RENDERED_JOB_SCHEMA
        or rendered.mode != config.mode
        or rendered.completions != config.completions
        or rendered.parallelism != config.parallelism
        or rendered.gpus_per_completion != config.gpus_per_completion
        or rendered.package_plan_sha256 != config.package_plan_sha256
        or rendered.execution_authorization_receipt_sha256
        != config.authorization_receipt_sha256
        or rendered.audit_plan_receipt_sha256 != config.audit_plan_receipt_sha256
        or rendered.capacity_receipt_sha256 != config.capacity_receipt_sha256
        or annotations.get("rl-quant/source-archive-sha256")
        != config.source_archive_sha256
        or annotations.get("rl-quant/image-digest-sha256") != config.image_digest_sha256
        or annotations.get("rl-quant/training-authorized") != "false"
        or annotations.get("rl-quant/outer-2026-access-authorized") != "false"
    ):
        raise M03RV11A15InferenceAuditLifecycleError(
            "rendered audit Job and attach config drifted"
        )
    return rendered


def _job_identity(job: Mapping[str, Any], config: M03RV11A15AuditAttachConfig) -> None:
    common._job_identity(
        job,
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=config.job_uid,
    )
    metadata = cast(Mapping[str, Any], job.get("metadata", {}))
    annotations = cast(Mapping[str, Any], metadata.get("annotations", {}))
    expected = {
        "rl-quant/package-plan-sha256": config.package_plan_sha256,
        "rl-quant/execution-authorization-sha256": config.authorization_receipt_sha256,
        "rl-quant/audit-plan-sha256": config.audit_plan_receipt_sha256,
        "rl-quant/source-archive-sha256": config.source_archive_sha256,
        "rl-quant/capacity-receipt-sha256": config.capacity_receipt_sha256,
    }
    if any(annotations.get(name) != value for name, value in expected.items()):
        raise M03RV11A15InferenceAuditLifecycleError(
            "live audit Job artifact identity drifted"
        )


def _same_activation(
    configured: M03RV7ExactJobActivationRequest,
    runtime: M03RV7ExactJobActivationRequest,
) -> bool:
    return common._same_activation_contract(configured, runtime)


def _terminal_condition(job: Mapping[str, Any]) -> str | None:
    return common._true_condition(job)


def _pod_rows(
    config: M03RV11A15AuditAttachConfig,
    pods: Sequence[Mapping[str, Any]],
    *,
    terminal_phase: str,
) -> list[dict[str, Any]]:
    if len(pods) != config.completions:
        raise M03RV11A15InferenceAuditLifecycleError("terminal audit Pod count drifted")
    rows: dict[int, dict[str, Any]] = {}
    for pod in pods:
        metadata = cast(Mapping[str, Any], pod.get("metadata", {}))
        status = cast(Mapping[str, Any], pod.get("status", {}))
        annotations = cast(Mapping[str, Any], metadata.get("annotations", {}))
        owners = metadata.get("ownerReferences")
        raw_index = annotations.get("batch.kubernetes.io/job-completion-index")
        container_statuses = status.get("containerStatuses")
        try:
            index = int(cast(str, raw_index))
        except (TypeError, ValueError) as exc:
            raise M03RV11A15InferenceAuditLifecycleError(
                "terminal audit Pod index is invalid"
            ) from exc
        if (
            index in rows
            or not 0 <= index < config.completions
            or status.get("phase") != terminal_phase
            or not isinstance(owners, list)
            or not any(
                isinstance(owner, Mapping)
                and owner.get("uid") == config.job_uid
                and owner.get("controller") is True
                for owner in owners
            )
            or not isinstance(container_statuses, list)
            or len(container_statuses) != 1
        ):
            raise M03RV11A15InferenceAuditLifecycleError(
                "terminal audit Pod ownership or status drifted"
            )
        container = cast(Mapping[str, Any], container_statuses[0])
        terminated = cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], container.get("state", {})).get("terminated", {}),
        )
        image_id = container.get("imageID")
        uid = metadata.get("uid")
        name = metadata.get("name")
        if (
            not isinstance(uid, str)
            or not isinstance(name, str)
            or not isinstance(image_id, str)
            or not image_id.endswith("@sha256:" + config.image_digest_sha256)
            or terminated.get("exitCode")
            != (0 if terminal_phase == "Succeeded" else terminated.get("exitCode"))
        ):
            raise M03RV11A15InferenceAuditLifecycleError(
                "terminal audit Pod image or exit status drifted"
            )
        rows[index] = {
            "completion_index": index,
            "pod_name": name,
            "pod_uid": uid,
            "image_id": image_id,
            "exit_code": terminated.get("exitCode"),
        }
    if set(rows) != set(range(config.completions)):
        raise M03RV11A15InferenceAuditLifecycleError(
            "terminal audit Pod completion coverage drifted"
        )
    return [rows[index] for index in range(config.completions)]


def _validate_startup(
    value: Mapping[str, Any],
    *,
    config: M03RV11A15AuditAttachConfig,
    setting_index: int,
    mode: str,
) -> str:
    receipt = _semantic_receipt(value, label="audit startup")
    hardware = value.get("hardware")
    if not isinstance(hardware, Mapping):
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit startup hardware proof is absent"
        )
    memory = hardware.get("device_total_memory")
    if (
        value.get("schema") != M03R_V11_A15_AUDIT_STARTUP_SCHEMA
        or value.get("protocol_sha256") != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
        or value.get("audit_plan_receipt_sha256") != config.audit_plan_receipt_sha256
        or value.get("audit_package_plan_sha256") != config.package_plan_sha256
        or value.get("audit_authorization_receipt_sha256")
        != config.authorization_receipt_sha256
        or value.get("parent_cleanup_receipt_sha256")
        != config.parent_cleanup_receipt_sha256
        or value.get("setting_index") != setting_index
        or value.get("mode") != mode
        or hardware.get("visible_device_count") != 1
        or "H100" not in str(hardware.get("device_name", "")).upper()
        or isinstance(memory, bool)
        or not isinstance(memory, int)
        or not 75 * 1024**3 <= memory <= 85 * 1024**3
        or hardware.get("exact_h100_80gb") is not True
        or value.get("training_performed") is not False
        or value.get("checkpoint_selection_performed") is not False
        or value.get("economic_optimizer_updates") != 0
        or value.get("outer_2026_accessed") is not False
        or value.get("promotion_eligible") is not False
    ):
        raise M03RV11A15InferenceAuditLifecycleError("audit startup semantics drifted")
    return receipt


def _validate_static_output(
    config: M03RV11A15AuditAttachConfig,
) -> tuple[dict[str, Any], str, str]:
    value, file_sha = _read_stable_json(
        Path(config.output_root) / "static-terminal.json"
    )
    receipt = _semantic_receipt(value, label="audit static terminal")
    if (
        value.get("schema") != M03R_V11_A15_AUDIT_STATIC_TERMINAL_SCHEMA
        or value.get("protocol_sha256") != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
        or value.get("package_plan_sha256") != config.package_plan_sha256
        or value.get("authorization_receipt_sha256")
        != config.authorization_receipt_sha256
        or value.get("audit_plan_receipt_sha256") != config.audit_plan_receipt_sha256
        or value.get("parent_cleanup_receipt_sha256")
        != config.parent_cleanup_receipt_sha256
        or value.get("source_archive_sha256") != config.source_archive_sha256
        or value.get("gpu_mask") != "none"
        or value.get("gpu_requests") != 0
        or value.get("gpu_limits") != 0
        or value.get("unmasked_visibility_claimed") is not False
        or value.get("h100_capacity_evidence") is not False
        or value.get("training_performed") is not False
        or value.get("outer_2026_accessed") is not False
        or value.get("promotion_eligible") is not False
    ):
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit static terminal semantics drifted"
        )
    return value, file_sha, receipt


def _validate_capacity_output(
    config: M03RV11A15AuditAttachConfig,
) -> tuple[dict[str, Any], str, str, str, str]:
    root = Path(config.output_root) / "capacity-sentinel"
    startup, startup_sha = _read_stable_json(root / "startup.json")
    _validate_startup(startup, config=config, setting_index=0, mode="capacity")
    terminal, terminal_sha = _read_stable_json(root / "capacity-terminal.json")
    receipt = _semantic_receipt(terminal, label="audit capacity terminal")
    cursor_sha = terminal.get("cursor_artifact_file_sha256")
    if (
        terminal.get("schema") != M03R_V11_A15_AUDIT_CAPACITY_TERMINAL_SCHEMA
        or terminal.get("protocol_sha256")
        != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
        or terminal.get("audit_plan_receipt_sha256") != config.audit_plan_receipt_sha256
        or terminal.get("startup_file_sha256") != startup_sha
        or terminal.get("setting_index") != 0
        or terminal.get("fold_index") != 0
        or terminal.get("horizon_sessions") != 30
        or terminal.get("exact_h100_80gb") is not True
        or terminal.get("full_execution_path_proven") is not True
        or terminal.get("training_performed") is not False
        or terminal.get("checkpoint_selection_performed") is not False
        or terminal.get("economic_optimizer_updates") != 0
        or terminal.get("outer_2026_accessed") is not False
        or terminal.get("h100_capacity_evidence") is not True
        or terminal.get("promotion_eligible") is not False
        or not isinstance(cursor_sha, str)
        or _SHA256_RE.fullmatch(cursor_sha) is None
    ):
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit capacity terminal semantics drifted"
        )
    _stable_file_sha256(
        root / "fold-artifacts" / "fold-00-horizon-30.pt",
        expected_sha256=cursor_sha,
    )
    return terminal, terminal_sha, receipt, startup_sha, cursor_sha


def _validate_audit_output(
    config: M03RV11A15AuditAttachConfig,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    outputs: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    for setting_index in range(2):
        root = (
            Path(config.output_root)
            / f"completion-{setting_index:02d}-setting-{setting_index:02d}"
        )
        startup, startup_sha = _read_stable_json(root / "startup.json")
        _validate_startup(
            startup,
            config=config,
            setting_index=setting_index,
            mode="audit",
        )
        terminal, terminal_sha = _read_stable_json(root / "audit-terminal.json")
        receipt = _semantic_receipt(terminal, label="audit worker terminal")
        artifacts = terminal.get("cursor_artifacts")
        reports = terminal.get("panel_reports")
        if (
            terminal.get("schema") != M03R_V11_A15_AUDIT_WORKER_TERMINAL_SCHEMA
            or terminal.get("protocol_sha256")
            != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
            or terminal.get("audit_plan_receipt_sha256")
            != config.audit_plan_receipt_sha256
            or terminal.get("parent_cleanup_receipt_sha256")
            != config.parent_cleanup_receipt_sha256
            or terminal.get("setting_index") != setting_index
            or terminal.get("startup_file_sha256") != startup_sha
            or not isinstance(artifacts, list)
            or len(artifacts) != 12
            or not isinstance(reports, list)
            or len(reports) != 14
            or terminal.get("training_performed") is not False
            or terminal.get("checkpoint_selection_performed") is not False
            or terminal.get("economic_optimizer_updates") != 0
            or terminal.get("economic_generation_may_be_minted") is not False
            or terminal.get("outer_2026_accessed") is not False
            or terminal.get("posthoc_exploratory") is not True
            or terminal.get("promotion_eligible") is not False
        ):
            raise M03RV11A15InferenceAuditLifecycleError(
                "audit worker terminal semantics drifted"
            )
        cursor_map: dict[tuple[int, int], Mapping[str, Any]] = {}
        for row in artifacts:
            if not isinstance(row, Mapping) or set(row) != {
                "fold_index",
                "horizon_sessions",
                "file_sha256",
            }:
                raise M03RV11A15InferenceAuditLifecycleError(
                    "audit cursor inventory row drifted"
                )
            fold_index = row.get("fold_index")
            horizon_sessions = row.get("horizon_sessions")
            if (
                not isinstance(fold_index, int)
                or not isinstance(horizon_sessions, int)
                or not isinstance(row.get("file_sha256"), str)
                or _SHA256_RE.fullmatch(cast(str, row.get("file_sha256"))) is None
            ):
                raise M03RV11A15InferenceAuditLifecycleError(
                    "audit cursor inventory identity drifted"
                )
            key = (fold_index, horizon_sessions)
            if key in cursor_map:
                raise M03RV11A15InferenceAuditLifecycleError(
                    "audit cursor inventory identity drifted"
                )
            cursor_map[key] = row
        expected_cursors = {
            (fold, horizon)
            for fold in range(6)
            for horizon in M03R_V11_A15_AUDIT_HORIZONS
        }
        if set(cursor_map) != expected_cursors:
            raise M03RV11A15InferenceAuditLifecycleError(
                "audit cursor coverage drifted"
            )
        for (fold, horizon), row in cursor_map.items():
            artifact_path = (
                root / "fold-artifacts" / f"fold-{fold:02d}-horizon-{horizon:02d}.pt"
            )
            artifact_sha = cast(str, row["file_sha256"])
            _stable_file_sha256(artifact_path, expected_sha256=artifact_sha)
            file_hashes[str(artifact_path)] = artifact_sha

        report_map: dict[tuple[int, str], Mapping[str, Any]] = {}
        for row in reports:
            if not isinstance(row, Mapping) or set(row) != {
                "horizon_sessions",
                "variant_id",
                "receipt_sha256",
                "file_sha256",
            }:
                raise M03RV11A15InferenceAuditLifecycleError(
                    "audit report inventory row drifted"
                )
            horizon_sessions = row.get("horizon_sessions")
            variant_id = row.get("variant_id")
            if (
                not isinstance(horizon_sessions, int)
                or not isinstance(variant_id, str)
                or not isinstance(row.get("receipt_sha256"), str)
                or _SHA256_RE.fullmatch(cast(str, row.get("receipt_sha256"))) is None
                or not isinstance(row.get("file_sha256"), str)
                or _SHA256_RE.fullmatch(cast(str, row.get("file_sha256"))) is None
            ):
                raise M03RV11A15InferenceAuditLifecycleError(
                    "audit report inventory identity drifted"
                )
            report_key = (horizon_sessions, variant_id)
            if report_key in report_map:
                raise M03RV11A15InferenceAuditLifecycleError(
                    "audit report inventory identity drifted"
                )
            report_map[report_key] = row
        expected_reports = {
            (horizon, variant.variant_id)
            for horizon in M03R_V11_A15_AUDIT_HORIZONS
            for variant in M03R_V11_A15_AUDIT_VARIANTS
        }
        if set(report_map) != expected_reports:
            raise M03RV11A15InferenceAuditLifecycleError(
                "audit report coverage drifted"
            )
        for (horizon, variant_id), row in report_map.items():
            report_path = (
                root / "panel-reports" / f"horizon-{horizon:02d}-{variant_id}.json"
            )
            report, report_sha = _read_stable_json(
                report_path,
                expected_sha256=cast(str, row["file_sha256"]),
            )
            _validate_panel_report(
                report,
                setting_index=setting_index,
                horizon_sessions=horizon,
                variant_id=variant_id,
                expected_receipt_sha256=cast(str, row["receipt_sha256"]),
            )
            file_hashes[str(report_path)] = report_sha
        outputs.append(
            {
                "setting_index": setting_index,
                "startup_file_sha256": startup_sha,
                "terminal_file_sha256": terminal_sha,
                "terminal_receipt_sha256": receipt,
            }
        )
        file_hashes[str(root / "startup.json")] = startup_sha
        file_hashes[str(root / "audit-terminal.json")] = terminal_sha
    return outputs, file_hashes


def _capture_terminal(
    *,
    root: Path,
    job: Mapping[str, Any],
    pods: Sequence[Mapping[str, Any]],
    transport: AuditTransport,
    reason: str,
    log_limit_bytes: int,
) -> dict[str, Any]:
    return common._capture_terminal(
        root=root,
        job=job,
        pods=pods,
        transport=cast(Any, transport),
        reason=reason,
        log_limit_bytes=log_limit_bytes,
    )


def _write_phase_receipt(
    config: M03RV11A15AuditAttachConfig,
    payload: dict[str, Any],
) -> str:
    unsigned = {
        **payload,
        "job_name": config.job_name,
        "job_uid": config.job_uid,
        "run_id": config.run_id,
        "package_plan_sha256": config.package_plan_sha256,
        "authorization_receipt_sha256": config.authorization_receipt_sha256,
        "audit_plan_receipt_sha256": config.audit_plan_receipt_sha256,
        "parent_cleanup_receipt_sha256": config.parent_cleanup_receipt_sha256,
        "source_archive_sha256": config.source_archive_sha256,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_training_authorized": False,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    value = {**unsigned, "receipt_sha256": _content_sha256(unsigned)}
    return common._exclusive_json(Path(config.phase_receipt_output_path), value)


def run_m03r_v11_a15_audit_attach_lifecycle(
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
                raise M03RV11A15InferenceAuditLifecycleError(
                    "audit spawn process receipt was not published"
                )
            sleep(0.1)
        process_value, _ = _read_stable_json(process_path)
        common._validate_spawned_identity(process_value, pid=os.getpid())
    source = common._regular_no_symlink(Path(__file__), label="audit lifecycle source")
    if common._file_sha256(source) != config.lifecycle_source_sha256:
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit lifecycle source SHA-256 drifted"
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
        or binding.parallelism != config.parallelism
        or binding.desired_manifest_sha256 != rendered.manifest_sha256
    ):
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit binding and attach config drifted"
        )
    live = transport or AuditAttachOnlyKubectl(
        container_name="validator" if config.mode == "static" else "auditor",
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
        raise M03RV11A15InferenceAuditLifecycleError(
            "bound audit Job is absent before activation"
        )
    _job_identity(fresh, config)
    if live.get_owned_pods():
        raise M03RV11A15InferenceAuditLifecycleError(
            "bound suspended audit Job unexpectedly owns Pods"
        )
    runtime_activation = build_m03r_v7_exact_job_activation_request(binding, fresh)
    if not _same_activation(configured_activation, runtime_activation):
        raise M03RV11A15InferenceAuditLifecycleError(
            "fresh audit activation identity drifted"
        )
    common._exclusive_json(
        root / "activation-request-runtime.json", asdict(runtime_activation)
    )
    try:
        activated = live.activate(runtime_activation)
    except Exception as exc:  # noqa: BLE001 - activation ambiguity must attach
        common._exclusive_json(
            root / "activation-attach-required.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v11-a15-audit-activation-attach-v1",
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
        raise M03RV11A15AuditActivationAttachRequired(
            "audit activation result is ambiguous; never retry activation"
        ) from exc
    _job_identity(activated, config)
    if cast(Mapping[str, Any], activated.get("spec", {})).get("suspend") is not False:
        raise M03RV11A15AuditActivationAttachRequired(
            "audit activation response did not prove unsuspended state"
        )
    activation_sha = common._exclusive_json(root / "activation.json", activated)
    common._exclusive_json(
        root / "launch-success.json",
        {
            "schema": M03R_V11_A15_AUDIT_LAUNCH_SUCCESS_SCHEMA,
            "job_name": config.job_name,
            "job_uid": config.job_uid,
            "run_id": config.run_id,
            "mode": config.mode,
            "parallelism": config.parallelism,
            "gpus_per_completion": config.gpus_per_completion,
            "request_ceiling_h100": config.parallelism * config.gpus_per_completion,
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
            raise M03RV11A15InferenceAuditLifecycleError(
                "activated audit Job disappeared before terminal evidence"
            )
        _job_identity(job, config)
        condition = _terminal_condition(job)
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
                "schema": "rl-quant.top2000-dev.m03r-v11-a15-audit-timeout-attach-v1",
                "job_name": config.job_name,
                "job_uid": config.job_uid,
                "hard_wall_seconds": config.hard_wall_seconds,
                "cleanup_performed": False,
                "attach_required": True,
            },
        )
        raise M03RV11A15AuditActivationAttachRequired(
            "audit Job exceeded its hard wall; retain for reviewed attachment"
        )
    terminal_phase = "Succeeded" if terminal_state == "Complete" else "Failed"
    _capture_terminal(
        root=root,
        job=terminal_job,
        pods=terminal_pods,
        transport=live,
        reason=f"audit-{config.mode}-{terminal_state.lower()}",
        log_limit_bytes=config.log_limit_bytes,
    )
    if terminal_state != "Complete":
        common._cleanup_exact_job(
            root=root,
            binding=binding,
            transport=cast(Any, live),
            sleep=sleep,
        )
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit Job failed; terminal evidence preserved and exact cleanup completed"
        )
    pod_rows = _pod_rows(
        config,
        terminal_pods,
        terminal_phase=terminal_phase,
    )
    if config.mode == "static":
        _, terminal_sha, terminal_receipt = _validate_static_output(config)
        phase_payload = {
            "schema": M03R_V11_A15_AUDIT_STATIC_GATE_SCHEMA,
            "static_terminal_file_sha256": terminal_sha,
            "static_terminal_receipt_sha256": terminal_receipt,
            "pod_runtime_proof": pod_rows,
            "gpu_requests": 0,
            "gpu_limits": 0,
            "unmasked_visibility_claimed": False,
            "h100_capacity_evidence": False,
            "passed": True,
        }
    elif config.mode == "capacity":
        (
            _,
            terminal_sha,
            terminal_receipt,
            startup_sha,
            cursor_sha,
        ) = _validate_capacity_output(config)
        phase_payload = {
            "schema": M03R_V11_A15_AUDIT_CAPACITY_SCHEMA,
            "static_gate_receipt_sha256": config.static_gate_receipt_sha256,
            "capacity_terminal_file_sha256": terminal_sha,
            "capacity_terminal_receipt_sha256": terminal_receipt,
            "startup_file_sha256": startup_sha,
            "cursor_artifact_file_sha256": cursor_sha,
            "job_uid": config.job_uid,
            "pod_uid": pod_rows[0]["pod_uid"],
            "image_id": pod_rows[0]["image_id"],
            "visible_device_count": 1,
            "exact_h100_80gb": True,
            "full_execution_path_proven": True,
            "economic_optimizer_updates": 0,
            "passed": True,
        }
    else:
        outputs, output_hashes = _validate_audit_output(config)
        phase_payload = {
            "schema": M03R_V11_A15_AUDIT_FINAL_SCHEMA,
            "static_gate_receipt_sha256": config.static_gate_receipt_sha256,
            "capacity_receipt_sha256": config.capacity_receipt_sha256,
            "worker_outputs": outputs,
            "output_file_sha256": output_hashes,
            "pod_runtime_proof": pod_rows,
            "completion_count": 2,
            "economic_optimizer_updates": 0,
            "posthoc_exploratory": True,
            "economic_generation_may_be_minted": False,
            "passed": True,
        }
    common._cleanup_exact_job(
        root=root,
        binding=binding,
        transport=cast(Any, live),
        sleep=sleep,
    )
    cleanup_path = root / "cleanup-receipt.json"
    cleanup_value, cleanup_sha = _read_stable_json(cleanup_path)
    if (
        cleanup_value.get("first_job_absent") is not True
        or cleanup_value.get("second_job_absent") is not True
        or cleanup_value.get("first_owned_pod_uids") != []
        or cleanup_value.get("second_owned_pod_uids") != []
    ):
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit exact cleanup receipt drifted"
        )
    phase_payload["cleanup_receipt_file_sha256"] = cleanup_sha
    _write_phase_receipt(config, phase_payload)


def spawn_m03r_v11_a15_audit_attach_lifecycle(
    *,
    config_path: Path,
    config_sha256: str,
    wait: Callable[[float], None] = time.sleep,
) -> int:
    """Start the attach-only lifecycle detached and return after activation proof."""

    config = _load_config(config_path, config_sha256)
    root = common._create_evidence_root(Path(config.evidence_root))
    python = common._regular_no_symlink(
        Path(config.host_python_path), label="audit host Python"
    )
    pythonpath = common._directory_no_symlink(
        Path(config.pythonpath), label="audit PYTHONPATH"
    )
    kubectl = common._regular_no_symlink(Path(config.kubectl_path), label="kubectl")
    kubeconfig = common._regular_no_symlink(
        Path(config.kubeconfig_path), label="kubeconfig"
    )
    source = common._regular_no_symlink(Path(__file__), label="audit lifecycle source")
    if common._file_sha256(source) != config.lifecycle_source_sha256:
        raise M03RV11A15InferenceAuditLifecycleError(
            "audit lifecycle source SHA-256 drifted before spawn"
        )
    command = (
        str(python),
        "-m",
        "rl_quant.training.top2000_m03r_v11_a15_inference_audit_lifecycle",
        "run",
        "--config",
        str(config_path),
        "--config-sha256",
        config_sha256,
    )
    command_sha = hashlib.sha256(
        b"\0".join(item.encode("utf-8") for item in command)
    ).hexdigest()
    intent = {
        "schema": "rl-quant.top2000-dev.m03r-v11-a15-audit-spawn-intent-v1",
        "config_sha256": config_sha256,
        "command_sha256": command_sha,
        "python_sha256": common._file_sha256(python),
        "kubectl_sha256": common._file_sha256(kubectl),
        "supervisor_source_sha256": common._file_sha256(source),
        "kubeconfig_metadata_validated": kubeconfig.is_file(),
        "create_authorized": False,
        "apply_authorized": False,
        "replace_authorized": False,
    }
    intent_sha = common._exclusive_json(root / "spawn-intent.json", intent)
    log_descriptor = os.open(
        root / "supervisor.log",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o440,
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
        "schema": "rl-quant.top2000-dev.m03r-v11-a15-audit-process-v1",
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
            raise M03RV11A15InferenceAuditLifecycleError(
                "audit attach-only supervisor exited before activation proof"
            )
        if launch_path.exists():
            launch, _ = _read_stable_json(launch_path)
            if (
                launch.get("schema") != M03R_V11_A15_AUDIT_LAUNCH_SUCCESS_SCHEMA
                or launch.get("job_name") != config.job_name
                or launch.get("job_uid") != config.job_uid
                or launch.get("run_id") != config.run_id
                or launch.get("mode") != config.mode
                or launch.get("parallelism") != config.parallelism
                or launch.get("gpus_per_completion") != config.gpus_per_completion
                or launch.get("request_ceiling_h100")
                != config.parallelism * config.gpus_per_completion
                or launch.get("activated") is not True
                or launch.get("training_authorized") is not False
                or launch.get("outer_2026_access_authorized") is not False
            ):
                raise M03RV11A15InferenceAuditLifecycleError(
                    "audit launch-success handshake drifted"
                )
            common._validate_spawned_identity(process_receipt, pid=process.pid)
            return process.pid
        if time.monotonic() >= deadline:
            raise M03RV11A15InferenceAuditLifecycleError(
                "audit attach-only activation handshake timed out"
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
        spawn_m03r_v11_a15_audit_attach_lifecycle(
            config_path=Path(arguments.config),
            config_sha256=arguments.config_sha256,
        )
        return 0
    try:
        run_m03r_v11_a15_audit_attach_lifecycle(
            arguments.config,
            arguments.config_sha256,
        )
    except M03RV11A15AuditActivationAttachRequired as exc:
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
    "M03R_V11_A15_AUDIT_ATTACH_CONFIG_SCHEMA",
    "M03R_V11_A15_AUDIT_FINAL_SCHEMA",
    "M03R_V11_A15_AUDIT_LAUNCH_SUCCESS_SCHEMA",
    "M03R_V11_A15_AUDIT_STATIC_GATE_SCHEMA",
    "M03RV11A15AuditActivationAttachRequired",
    "M03RV11A15AuditAttachConfig",
    "M03RV11A15InferenceAuditLifecycleError",
    "main",
    "run_m03r_v11_a15_audit_attach_lifecycle",
    "spawn_m03r_v11_a15_audit_attach_lifecycle",
]
