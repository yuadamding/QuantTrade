"""Immutable parent-lineage plan for the M03R-v11 a15 inference audit."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_AUDIT_HORIZONS,
    M03R_V11_A15_AUDIT_SETTING_INDEXES,
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
    M03R_V11_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v11_package import (
    M03RV11PackagePlan,
    load_m03r_v11_execution_authorization,
    load_m03r_v11_package_plan,
)

M03R_V11_A15_PARENT_RUN_ID = "qt-m03r-v11-predictive-s17-20260812-a15"
M03R_V11_A15_PARENT_JOB_NAME = "qt-m03r-v11-predictive-a15"
M03R_V11_A15_AUDIT_PLAN_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-plan-v1"
)
M03R_V11_A15_PARENT_FOLD_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-predictive-fold-terminal-v1"
)
M03R_V11_A15_PARENT_WORKER_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-predictive-worker-terminal-v1"
)
M03R_V11_A15_PARENT_TERMINAL_EVIDENCE_SCHEMA = (
    "rl-quant.top2000-m03r-v7-terminal-evidence-v1"
)

_MAX_JSON_BYTES = 2 * 1024 * 1024


class M03RV11A15InferenceAuditPlanError(ValueError):
    """The parent evidence or frozen audit plan drifted."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()


def _parent_receipt_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11A15InferenceAuditPlanError(f"{name} must be a lowercase SHA-256")
    return value


def _relative(name: str, value: str) -> str:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or "." in pure.parts
        or str(pure) != value
    ):
        raise M03RV11A15InferenceAuditPlanError(
            f"{name} must be a normalized relative path"
        )
    return value


def _read_regular_json(path: str | Path, expected_file_sha256: str) -> dict[str, Any]:
    source = Path(path)
    _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV11A15InferenceAuditPlanError(
            f"audit input is not a readable regular file: {source.name}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_JSON_BYTES
        ):
            raise M03RV11A15InferenceAuditPlanError(
                f"audit input size or type is invalid: {source.name}"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while block := os.read(descriptor, 1024 * 1024):
            total += len(block)
            if total > _MAX_JSON_BYTES:
                raise M03RV11A15InferenceAuditPlanError(
                    f"audit input is too large: {source.name}"
                )
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV11A15InferenceAuditPlanError(
                f"audit input changed while reading: {source.name}"
            )
        if digest.hexdigest() != expected_file_sha256:
            raise M03RV11A15InferenceAuditPlanError(
                f"audit input file hash drifted: {source.name}"
            )
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M03RV11A15InferenceAuditPlanError(
            f"audit input is not valid JSON: {source.name}"
        ) from exc
    if not isinstance(value, dict):
        raise M03RV11A15InferenceAuditPlanError(
            f"audit input must be a JSON object: {source.name}"
        )
    return dict(value)


def _validate_receipt(payload: dict[str, Any], *, name: str) -> str:
    receipt = payload.get("receipt_sha256")
    if not isinstance(receipt, str):
        raise M03RV11A15InferenceAuditPlanError(f"{name} omitted receipt SHA-256")
    _digest(f"{name}.receipt_sha256", receipt)
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if _parent_receipt_sha256(unsigned) != receipt:
        raise M03RV11A15InferenceAuditPlanError(f"{name} receipt hash drifted")
    return receipt


@dataclass(frozen=True, slots=True)
class M03RV11A15ParentCheckpointBinding:
    setting_index: int
    fold_index: int
    horizon_sessions: int
    checkpoint_relative_path: str
    checkpoint_file_sha256: str
    model_state_sha256: str
    fold_terminal_relative_path: str
    fold_terminal_file_sha256: str
    fold_terminal_receipt_sha256: str
    worker_plan_sha256: str
    training_source_array_sha256: str
    training_residual_operator_root_sha256: str
    qualification_source_array_sha256: str
    qualification_residual_operator_root_sha256: str
    fold_risk_state_sha256: str

    def validate(self) -> None:
        expected_checkpoint = (
            f"completion-{self.setting_index:02d}-setting-{self.setting_index:02d}/"
            f"checkpoints/fold-{self.fold_index:02d}-horizon-"
            f"{self.horizon_sessions:02d}-update-0064.pt"
        )
        expected_terminal = (
            f"completion-{self.setting_index:02d}-setting-{self.setting_index:02d}/"
            f"receipts/fold-{self.fold_index:02d}-terminal.json"
        )
        if (
            self.setting_index not in M03R_V11_A15_AUDIT_SETTING_INDEXES
            or self.fold_index not in range(6)
            or self.horizon_sessions not in M03R_V11_A15_AUDIT_HORIZONS
            or _relative("checkpoint_relative_path", self.checkpoint_relative_path)
            != expected_checkpoint
            or _relative(
                "fold_terminal_relative_path", self.fold_terminal_relative_path
            )
            != expected_terminal
        ):
            raise M03RV11A15InferenceAuditPlanError(
                "a15 parent checkpoint cursor or path drifted"
            )
        for name, value in asdict(self).items():
            if name.endswith("sha256"):
                _digest(name, value)


@dataclass(frozen=True, slots=True)
class M03RV11A15ParentWorkerBinding:
    setting_index: int
    terminal_relative_path: str
    terminal_file_sha256: str
    terminal_receipt_sha256: str
    worker_plan_sha256: str
    fold_terminal_file_sha256: tuple[str, ...]

    def validate(self) -> None:
        expected = (
            f"completion-{self.setting_index:02d}-setting-{self.setting_index:02d}/"
            "predictive-terminal.json"
        )
        if (
            self.setting_index not in M03R_V11_A15_AUDIT_SETTING_INDEXES
            or _relative("terminal_relative_path", self.terminal_relative_path)
            != expected
            or len(self.fold_terminal_file_sha256) != 6
        ):
            raise M03RV11A15InferenceAuditPlanError("a15 parent worker binding drifted")
        for name, value in (
            ("terminal_file_sha256", self.terminal_file_sha256),
            ("terminal_receipt_sha256", self.terminal_receipt_sha256),
            ("worker_plan_sha256", self.worker_plan_sha256),
            *(
                ("fold_terminal_file_sha256", row)
                for row in self.fold_terminal_file_sha256
            ),
        ):
            _digest(name, value)


@dataclass(frozen=True, slots=True)
class M03RV11A15InferenceAuditPlan:
    parent_run_id: str
    parent_job_name: str
    parent_protocol_sha256: str
    parent_package_plan_file_sha256: str
    parent_package_plan_sha256: str
    parent_execution_authorization_file_sha256: str
    parent_execution_authorization_receipt_sha256: str
    parent_source_archive_sha256: str
    parent_image_reference: str
    parent_terminal_evidence_relative_path: str
    parent_terminal_evidence_file_sha256: str
    parent_cleanup_receipt_relative_path: str
    parent_cleanup_receipt_file_sha256: str
    parent_cleanup_receipt_sha256: str
    workers: tuple[M03RV11A15ParentWorkerBinding, ...]
    checkpoints: tuple[M03RV11A15ParentCheckpointBinding, ...]
    receipt_sha256: str
    indexed_completions: int = 2
    parallelism: int = 2
    gpus_per_completion: int = 1
    maximum_gpu_requests: int = 2
    economic_optimizer_updates: int = 0
    training_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    economic_generation_may_be_minted: bool = False
    outer_2026_access_authorized: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    protocol_sha256: str = M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V11_A15_AUDIT_PLAN_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        for worker in self.workers:
            worker.validate()
        for checkpoint in self.checkpoints:
            checkpoint.validate()
        expected_cursors = tuple(
            (setting, fold, horizon)
            for setting in M03R_V11_A15_AUDIT_SETTING_INDEXES
            for fold in range(6)
            for horizon in M03R_V11_A15_AUDIT_HORIZONS
        )
        observed_cursors = tuple(
            (row.setting_index, row.fold_index, row.horizon_sessions)
            for row in self.checkpoints
        )
        if (
            self.parent_run_id != M03R_V11_A15_PARENT_RUN_ID
            or self.parent_job_name != M03R_V11_A15_PARENT_JOB_NAME
            or self.parent_protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or tuple(row.setting_index for row in self.workers)
            != M03R_V11_A15_AUDIT_SETTING_INDEXES
            or observed_cursors != expected_cursors
            or self.indexed_completions != 2
            or self.parallelism != 2
            or self.gpus_per_completion != 1
            or self.maximum_gpu_requests != 2
            or self.economic_optimizer_updates != 0
            or self.training_authorized
            or self.checkpoint_selection_authorized
            or self.economic_generation_may_be_minted
            or self.outer_2026_access_authorized
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.protocol_sha256 != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V11_A15_AUDIT_PLAN_SCHEMA
            or self.receipt_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditPlanError("a15 inference-audit plan drifted")
        for name, value in (
            ("parent_package_plan_file_sha256", self.parent_package_plan_file_sha256),
            ("parent_package_plan_sha256", self.parent_package_plan_sha256),
            (
                "parent_execution_authorization_file_sha256",
                self.parent_execution_authorization_file_sha256,
            ),
            (
                "parent_execution_authorization_receipt_sha256",
                self.parent_execution_authorization_receipt_sha256,
            ),
            ("parent_source_archive_sha256", self.parent_source_archive_sha256),
            (
                "parent_terminal_evidence_file_sha256",
                self.parent_terminal_evidence_file_sha256,
            ),
            (
                "parent_cleanup_receipt_file_sha256",
                self.parent_cleanup_receipt_file_sha256,
            ),
            ("parent_cleanup_receipt_sha256", self.parent_cleanup_receipt_sha256),
        ):
            _digest(name, value)
        if (
            _relative(
                "parent_terminal_evidence_relative_path",
                self.parent_terminal_evidence_relative_path,
            )
            != "predictive-evidence/terminal-evidence.json"
            or _relative(
                "parent_cleanup_receipt_relative_path",
                self.parent_cleanup_receipt_relative_path,
            )
            != "predictive-evidence/cleanup-receipt.json"
            or "@sha256:" not in self.parent_image_reference
        ):
            raise M03RV11A15InferenceAuditPlanError(
                "a15 parent lifecycle or image identity drifted"
            )
        worker_by_setting = {row.setting_index: row for row in self.workers}
        for setting in M03R_V11_A15_AUDIT_SETTING_INDEXES:
            checkpoint_files = tuple(
                row.fold_terminal_file_sha256
                for row in self.checkpoints
                if row.setting_index == setting and row.horizon_sessions == 21
            )
            if checkpoint_files != worker_by_setting[setting].fold_terminal_file_sha256:
                raise M03RV11A15InferenceAuditPlanError(
                    "a15 worker/fold terminal lineage drifted"
                )

    def checkpoint(
        self, setting_index: int, fold_index: int, horizon_sessions: int
    ) -> M03RV11A15ParentCheckpointBinding:
        for row in self.checkpoints:
            if (
                row.setting_index,
                row.fold_index,
                row.horizon_sessions,
            ) == (setting_index, fold_index, horizon_sessions):
                return row
        raise M03RV11A15InferenceAuditPlanError("a15 checkpoint cursor is not planned")


def _load_parent_fold(
    path: Path,
    expected_file_sha256: str,
    *,
    setting_index: int,
    fold_index: int,
    package: M03RV11PackagePlan,
) -> tuple[M03RV11A15ParentCheckpointBinding, ...]:
    payload = _read_regular_json(path, expected_file_sha256)
    receipt = _validate_receipt(payload, name="parent_fold_terminal")
    worker = package.panel.workers[setting_index]
    if (
        payload.get("schema") != M03R_V11_A15_PARENT_FOLD_TERMINAL_SCHEMA
        or payload.get("setting_index") != setting_index
        or payload.get("setting_id") != M03R_V11_SETTING_IDS[setting_index]
        or payload.get("fold_index") != fold_index
        or payload.get("completed_updates") != 64
        or payload.get("economic_optimizer_updates") != 0
        or payload.get("outer_2026_accessed") is not False
        or payload.get("development_only") is not True
        or payload.get("reportable") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("package_plan_sha256") != package.package_plan_sha256
        or payload.get("worker_plan_sha256") != worker.receipt_sha256
        or payload.get("qualification_evaluated_only_after_checkpoint_publication")
        is not True
    ):
        raise M03RV11A15InferenceAuditPlanError(
            "a15 parent fold terminal semantics drifted"
        )
    training_source = payload.get("training_source_array_sha256")
    training_residual = payload.get("training_residual_operator_root_sha256")
    candidates = payload.get("horizon_candidates")
    if (
        not isinstance(training_source, str)
        or not isinstance(training_residual, str)
        or not isinstance(candidates, dict)
        or set(candidates) != {"21", "30"}
    ):
        raise M03RV11A15InferenceAuditPlanError(
            "a15 parent fold checkpoint inventory drifted"
        )
    rows = []
    relative_terminal = (
        f"completion-{setting_index:02d}-setting-{setting_index:02d}/receipts/"
        f"fold-{fold_index:02d}-terminal.json"
    )
    for horizon in M03R_V11_A15_AUDIT_HORIZONS:
        value = candidates[str(horizon)]
        if not isinstance(value, dict):
            raise M03RV11A15InferenceAuditPlanError(
                "a15 parent horizon candidate is not an object"
            )
        parent_path = value.get("checkpoint_path")
        expected_parent_path = (
            f"/mnt/output/completion-{setting_index:02d}-setting-"
            f"{setting_index:02d}/checkpoints/fold-{fold_index:02d}-horizon-"
            f"{horizon:02d}-update-0064.pt"
        )
        if parent_path != expected_parent_path:
            raise M03RV11A15InferenceAuditPlanError(
                "a15 parent checkpoint path drifted"
            )
        row = M03RV11A15ParentCheckpointBinding(
            setting_index=setting_index,
            fold_index=fold_index,
            horizon_sessions=horizon,
            checkpoint_relative_path=parent_path.removeprefix("/mnt/output/"),
            checkpoint_file_sha256=str(value.get("checkpoint_file_sha256")),
            model_state_sha256=str(value.get("model_state_sha256")),
            fold_terminal_relative_path=relative_terminal,
            fold_terminal_file_sha256=expected_file_sha256,
            fold_terminal_receipt_sha256=receipt,
            worker_plan_sha256=worker.receipt_sha256,
            training_source_array_sha256=training_source,
            training_residual_operator_root_sha256=training_residual,
            qualification_source_array_sha256=str(
                value.get("qualification_source_array_sha256")
            ),
            qualification_residual_operator_root_sha256=str(
                value.get("qualification_residual_operator_root_sha256")
            ),
            fold_risk_state_sha256=str(value.get("fold_risk_state_sha256")),
        )
        row.validate()
        rows.append(row)
    if rows[0].model_state_sha256 != rows[1].model_state_sha256:
        raise M03RV11A15InferenceAuditPlanError(
            "a15 horizon checkpoint model states disagree"
        )
    return tuple(rows)


def build_m03r_v11_a15_inference_audit_plan(
    *,
    parent_package_plan_path: str | Path,
    parent_package_plan_file_sha256: str,
    parent_execution_authorization_path: str | Path,
    parent_execution_authorization_file_sha256: str,
    parent_output_root: str | Path,
    parent_worker_terminal_file_sha256: tuple[str, str],
    parent_fold_terminal_file_sha256: tuple[tuple[str, ...], tuple[str, ...]],
    parent_launch_root: str | Path,
    parent_terminal_evidence_file_sha256: str,
    parent_cleanup_receipt_file_sha256: str,
) -> M03RV11A15InferenceAuditPlan:
    """Freeze exact completed a15 checkpoint and cleanup lineage."""

    package = load_m03r_v11_package_plan(
        parent_package_plan_path,
        expected_file_sha256=parent_package_plan_file_sha256,
    )
    authorization = load_m03r_v11_execution_authorization(
        parent_execution_authorization_path,
        expected_file_sha256=parent_execution_authorization_file_sha256,
        package=package,
    )
    if (
        package.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
        or len(parent_worker_terminal_file_sha256) != 2
        or len(parent_fold_terminal_file_sha256) != 2
        or any(len(row) != 6 for row in parent_fold_terminal_file_sha256)
    ):
        raise M03RV11A15InferenceAuditPlanError(
            "a15 parent plan or terminal inventory drifted"
        )
    output_root = Path(parent_output_root)
    workers: list[M03RV11A15ParentWorkerBinding] = []
    checkpoints: list[M03RV11A15ParentCheckpointBinding] = []
    for setting in M03R_V11_A15_AUDIT_SETTING_INDEXES:
        terminal_relative = (
            f"completion-{setting:02d}-setting-{setting:02d}/predictive-terminal.json"
        )
        terminal = _read_regular_json(
            output_root / terminal_relative,
            parent_worker_terminal_file_sha256[setting],
        )
        terminal_receipt = _validate_receipt(terminal, name="parent_worker_terminal")
        expected_folds = parent_fold_terminal_file_sha256[setting]
        worker = package.panel.workers[setting]
        if (
            terminal.get("schema") != M03R_V11_A15_PARENT_WORKER_TERMINAL_SCHEMA
            or terminal.get("setting_index") != setting
            or terminal.get("setting_id") != M03R_V11_SETTING_IDS[setting]
            or terminal.get("worker_plan_sha256") != worker.receipt_sha256
            or terminal.get("package_plan_sha256") != package.package_plan_sha256
            or tuple(terminal.get("fold_terminal_file_sha256", ())) != expected_folds
            or terminal.get("predictive_gate_passed") is not False
            or terminal.get("selected_horizon") is not None
            or terminal.get("economic_generation_may_be_minted") is not False
            or terminal.get("economic_panel_authorized") is not False
            or terminal.get("economic_optimizer_updates") != 0
            or terminal.get("outer_2026_accessed") is not False
            or terminal.get("development_only") is not True
            or terminal.get("reportable") is not False
            or terminal.get("promotion_eligible") is not False
        ):
            raise M03RV11A15InferenceAuditPlanError(
                "a15 parent worker terminal semantics drifted"
            )
        worker_binding = M03RV11A15ParentWorkerBinding(
            setting_index=setting,
            terminal_relative_path=terminal_relative,
            terminal_file_sha256=parent_worker_terminal_file_sha256[setting],
            terminal_receipt_sha256=terminal_receipt,
            worker_plan_sha256=worker.receipt_sha256,
            fold_terminal_file_sha256=expected_folds,
        )
        worker_binding.validate()
        workers.append(worker_binding)
        for fold in range(6):
            relative = (
                f"completion-{setting:02d}-setting-{setting:02d}/receipts/"
                f"fold-{fold:02d}-terminal.json"
            )
            checkpoints.extend(
                _load_parent_fold(
                    output_root / relative,
                    expected_folds[fold],
                    setting_index=setting,
                    fold_index=fold,
                    package=package,
                )
            )

    launch_root = Path(parent_launch_root)
    terminal_relative = "predictive-evidence/terminal-evidence.json"
    terminal_evidence = _read_regular_json(
        launch_root / terminal_relative,
        parent_terminal_evidence_file_sha256,
    )
    cleanup_relative = "predictive-evidence/cleanup-receipt.json"
    cleanup = _read_regular_json(
        launch_root / cleanup_relative,
        parent_cleanup_receipt_file_sha256,
    )
    cleanup_receipt = cleanup.get("receipt_sha256")
    request = cleanup.get("request")
    if (
        terminal_evidence.get("schema") != M03R_V11_A15_PARENT_TERMINAL_EVIDENCE_SCHEMA
        or terminal_evidence.get("reason") != "complete"
        or not isinstance(cleanup_receipt, str)
        or not isinstance(request, dict)
        or request.get("job_name") != M03R_V11_A15_PARENT_JOB_NAME
        or request.get("run_id") != M03R_V11_A15_PARENT_RUN_ID
        or cleanup.get("first_job_absent") is not True
        or cleanup.get("second_job_absent") is not True
        or cleanup.get("first_owned_pod_uids") != []
        or cleanup.get("second_owned_pod_uids") != []
    ):
        raise M03RV11A15InferenceAuditPlanError(
            "a15 parent terminal or cleanup evidence drifted"
        )
    provisional = M03RV11A15InferenceAuditPlan(
        parent_run_id=M03R_V11_A15_PARENT_RUN_ID,
        parent_job_name=M03R_V11_A15_PARENT_JOB_NAME,
        parent_protocol_sha256=package.protocol_sha256,
        parent_package_plan_file_sha256=parent_package_plan_file_sha256,
        parent_package_plan_sha256=package.package_plan_sha256,
        parent_execution_authorization_file_sha256=(
            parent_execution_authorization_file_sha256
        ),
        parent_execution_authorization_receipt_sha256=authorization.receipt_sha256,
        parent_source_archive_sha256=package.artifacts.source_archive_sha256,
        parent_image_reference=package.artifacts.image_reference,
        parent_terminal_evidence_relative_path=terminal_relative,
        parent_terminal_evidence_file_sha256=parent_terminal_evidence_file_sha256,
        parent_cleanup_receipt_relative_path=cleanup_relative,
        parent_cleanup_receipt_file_sha256=parent_cleanup_receipt_file_sha256,
        parent_cleanup_receipt_sha256=cleanup_receipt,
        workers=tuple(workers),
        checkpoints=tuple(checkpoints),
        receipt_sha256="0" * 64,
    )
    plan = replace(
        provisional,
        receipt_sha256=_sha256(provisional.unsigned_payload()),
    )
    plan.validate()
    return plan


def write_m03r_v11_a15_inference_audit_plan(
    path: str | Path, plan: M03RV11A15InferenceAuditPlan
) -> str:
    plan.validate()
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise M03RV11A15InferenceAuditPlanError("audit plan target already exists")
    encoded = _canonical({"schema": plan.schema, "plan": asdict(plan)})
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def load_m03r_v11_a15_inference_audit_plan(
    path: str | Path, *, expected_file_sha256: str
) -> M03RV11A15InferenceAuditPlan:
    payload = _read_regular_json(path, expected_file_sha256)
    if payload.get("schema") != M03R_V11_A15_AUDIT_PLAN_SCHEMA:
        raise M03RV11A15InferenceAuditPlanError("audit plan file schema drifted")
    value = payload.get("plan")
    if not isinstance(value, dict):
        raise M03RV11A15InferenceAuditPlanError("audit plan payload is missing")
    try:
        plan = M03RV11A15InferenceAuditPlan(
            **{
                **value,
                "workers": tuple(
                    M03RV11A15ParentWorkerBinding(
                        **{
                            **row,
                            "fold_terminal_file_sha256": tuple(
                                row.get("fold_terminal_file_sha256", ())
                            ),
                        }
                    )
                    for row in value.get("workers", ())
                ),
                "checkpoints": tuple(
                    M03RV11A15ParentCheckpointBinding(**row)
                    for row in value.get("checkpoints", ())
                ),
            }
        )
    except (TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditPlanError(
            "audit plan payload cannot be decoded"
        ) from exc
    plan.validate()
    return plan


__all__ = [
    "M03R_V11_A15_AUDIT_PLAN_SCHEMA",
    "M03R_V11_A15_PARENT_JOB_NAME",
    "M03R_V11_A15_PARENT_RUN_ID",
    "M03RV11A15InferenceAuditPlan",
    "M03RV11A15InferenceAuditPlanError",
    "M03RV11A15ParentCheckpointBinding",
    "M03RV11A15ParentWorkerBinding",
    "build_m03r_v11_a15_inference_audit_plan",
    "load_m03r_v11_a15_inference_audit_plan",
    "write_m03r_v11_a15_inference_audit_plan",
]
