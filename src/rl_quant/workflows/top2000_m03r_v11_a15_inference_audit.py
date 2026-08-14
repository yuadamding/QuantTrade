"""Governed one-H100 worker for the frozen M03R-v11 a15 inference audit.

The worker replays only predeclared transformations of exact update-64
checkpoints.  It performs no optimizer step, checkpoint selection, economic
training, or 2026 access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rl_quant.evaluation.top2000_m03r_v11_a15_inference_audit import (
    M03RV11A15AuditFoldEvidence,
    build_m03r_v11_a15_audit_panel_report,
)
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03RV9HorizonBinding,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_AUDIT_CAPACITY_TERMINAL_SCHEMA,
    M03R_V11_A15_AUDIT_CURSOR_SCHEMA,
    M03R_V11_A15_AUDIT_HORIZONS,
    M03R_V11_A15_AUDIT_SETTING_INDEXES,
    M03R_V11_A15_AUDIT_STARTUP_SCHEMA,
    M03R_V11_A15_AUDIT_VARIANTS,
    M03R_V11_A15_AUDIT_WORKER_ERROR_SCHEMA,
    M03R_V11_A15_AUDIT_WORKER_TERMINAL_SCHEMA,
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v9_fold import (
    build_m03r_v9_qualification_risk_state,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    load_m03r_v9_projector_manifest,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    load_top2000_m03r_v9_risk_source,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_fold import (
    evaluate_m03r_v11_a15_loaded_audit_fold,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan import (
    M03RV11A15InferenceAuditPlan,
    build_m03r_v11_a15_inference_audit_plan,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_package import (
    load_m03r_v11_a15_inference_audit_bundle,
)
from rl_quant.training.top2000_m03r_v11_checkpoint import (
    load_m03r_v11_alpha_checkpoint_for_evaluation,
)
from rl_quant.training.top2000_m03r_v11_package import (
    M03RV11PackagePlan,
    load_m03r_v11_execution_authorization,
    load_m03r_v11_package_plan,
)


class M03RV11A15InferenceAuditWorkflowError(RuntimeError):
    """The audit package, parent bytes, GPU, or output drifted."""


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise M03RV11A15InferenceAuditWorkflowError("audit JSON target already exists")
    encoded = _canonical(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _write_immutable_torch(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit tensor target already exists"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _new_policy(horizon: int, device: torch.device) -> Top2000M03RV9PredictivePolicy:
    binding = M03RV9HorizonBinding(horizon, horizon, horizon)
    return Top2000M03RV9PredictivePolicy(0, binding).to(device)


def _completion_index(explicit: int | None) -> int:
    raw = (
        str(explicit)
        if explicit is not None
        else os.environ.get("JOB_COMPLETION_INDEX", "")
    )
    try:
        index = int(raw)
    except ValueError as exc:
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit completion index is missing or invalid"
        ) from exc
    if index not in M03R_V11_A15_AUDIT_SETTING_INDEXES:
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit completion index is outside the frozen panel"
        )
    return index


def _h100_device() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit requires exactly one visible CUDA device"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    name = torch.cuda.get_device_name(device)
    if "H100" not in name.upper() or properties.total_memory < 75 * 1024**3:
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit requires one NVIDIA H100 80GB"
        )
    return device, {
        "visible_device_count": 1,
        "device_name": name,
        "device_total_memory": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "torch_cuda_version": torch.version.cuda,
        "exact_h100_80gb": True,
    }


def _rebuild_parent_lineage(
    audit: M03RV11A15InferenceAuditPlan,
    *,
    parent_package_plan_path: Path,
    parent_authorization_path: Path,
    parent_output_root: Path,
    parent_lifecycle_root: Path,
) -> M03RV11A15InferenceAuditPlan:
    workers = tuple(row.terminal_file_sha256 for row in audit.workers)
    folds = tuple(row.fold_terminal_file_sha256 for row in audit.workers)
    rebuilt = build_m03r_v11_a15_inference_audit_plan(
        parent_package_plan_path=parent_package_plan_path,
        parent_package_plan_file_sha256=audit.parent_package_plan_file_sha256,
        parent_execution_authorization_path=parent_authorization_path,
        parent_execution_authorization_file_sha256=(
            audit.parent_execution_authorization_file_sha256
        ),
        parent_output_root=parent_output_root,
        parent_worker_terminal_file_sha256=(workers[0], workers[1]),
        parent_fold_terminal_file_sha256=(folds[0], folds[1]),
        parent_launch_root=parent_lifecycle_root,
        parent_terminal_evidence_file_sha256=(
            audit.parent_terminal_evidence_file_sha256
        ),
        parent_cleanup_receipt_file_sha256=(audit.parent_cleanup_receipt_file_sha256),
    )
    if rebuilt != audit:
        raise M03RV11A15InferenceAuditWorkflowError(
            "live parent evidence does not match the frozen audit plan"
        )
    return rebuilt


def _load_common_inputs(
    package: M03RV11PackagePlan,
    *,
    setting_index: int,
    device: torch.device,
) -> tuple[Any, Any, Any, Any, tuple[Any, ...]]:
    worker = package.panel.workers[setting_index]
    cache = load_verified_top2000_hold30_development_cache(
        worker.cache_path,
        expected_cache_sha256=worker.cache_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    if cache.daily_ohlcv.shape[0] != TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS:
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit parent cache geometry drifted"
        )
    risk_source, _ = load_top2000_m03r_v9_risk_source(
        Path(worker.risk_source_manifest_path),
        expected_manifest_file_sha256=worker.risk_source_manifest_file_sha256,
    )
    projector, risk_binding = load_m03r_v9_projector_manifest(
        Path(worker.projector_manifest_path),
        expected_file_sha256=worker.projector_manifest_file_sha256,
    )
    if (
        projector.manifest_sha256 != worker.projector_manifest_sha256
        or risk_binding.binding_sha256 != worker.projector_binding_sha256
    ):
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit parent risk identity drifted"
        )
    folds = render_top2000_m03r_v7_development_folds(
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
    )
    return cache, risk_source, projector, risk_binding, folds


def _evaluate_cursor(
    *,
    audit: M03RV11A15InferenceAuditPlan,
    package: M03RV11PackagePlan,
    parent_output_root: Path,
    output_root: Path,
    setting_index: int,
    fold: Any,
    horizon: int,
    cache: Any,
    risk_source: Any,
    projector: Any,
    risk_binding: Any,
    risk_state: Any,
    device: torch.device,
) -> tuple[str, tuple[M03RV11A15AuditFoldEvidence, ...]]:
    binding = audit.checkpoint(setting_index, fold.fold_index, horizon)
    worker = package.panel.workers[setting_index]
    policy = _new_policy(horizon, device)
    checkpoint_path = parent_output_root / binding.checkpoint_relative_path
    loaded = load_m03r_v11_alpha_checkpoint_for_evaluation(
        checkpoint_path,
        expected_file_sha256=binding.checkpoint_file_sha256,
        expected_setting_index=setting_index,
        expected_fold_index=fold.fold_index,
        expected_selected_horizon_sessions=horizon,
        expected_episode_schedule_sha256=package.schedule.receipt_sha256,
        expected_residual_operator_root_sha256=(
            binding.training_residual_operator_root_sha256
        ),
        expected_source_array_sha256=binding.training_source_array_sha256,
        expected_asset_axis_sha256=cache.action_hash,
        policy=policy,
    )
    if loaded.model_state_sha256 != binding.model_state_sha256:
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit checkpoint semantic model hash drifted"
        )
    result, traces, evidence = evaluate_m03r_v11_a15_loaded_audit_fold(
        cache,
        worker,
        fold,
        risk_source,
        risk_state,
        policy,
        loaded,
        M03R_V11_A15_AUDIT_VARIANTS,
        expected_parent_fold_risk_state_sha256=binding.fold_risk_state_sha256,
        device=device,
    )
    if result.qualification_source_array_sha256 != (
        binding.qualification_source_array_sha256
    ):
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit recomputed qualification source-array lineage drifted"
        )
    if result.qualification_residual_operator_root_sha256 != (
        binding.qualification_residual_operator_root_sha256
    ):
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit recomputed qualification residual-operator lineage drifted"
        )
    if (
        result.parent_fold_risk_state_sha256 != binding.fold_risk_state_sha256
        or result.audit_fold_risk_state_sha256 != risk_state.state_sha256
    ):
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit fold risk-state evidence binding drifted"
        )
    artifact = {
        "schema": M03R_V11_A15_AUDIT_CURSOR_SCHEMA,
        "protocol_sha256": M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
        "audit_plan_receipt_sha256": audit.receipt_sha256,
        "parent_checkpoint_binding": asdict(binding),
        "fold_result": asdict(result),
        "traces": tuple(asdict(row) for row in traces),
        "fold_evidence": tuple(asdict(row) for row in evidence),
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    artifact_path = (
        output_root
        / "fold-artifacts"
        / f"fold-{fold.fold_index:02d}-horizon-{horizon:02d}.pt"
    )
    return _write_immutable_torch(artifact_path, artifact), evidence


def run_m03r_v11_a15_inference_audit_worker(
    audit_plan_path: str | Path,
    *,
    expected_audit_plan_file_sha256: str,
    package_plan_path: str | Path,
    expected_package_plan_file_sha256: str,
    authorization_path: str | Path,
    expected_authorization_file_sha256: str,
    parent_package_plan_path: str | Path,
    parent_authorization_path: str | Path,
    parent_output_root: str | Path,
    parent_lifecycle_root: str | Path,
    output_root: str | Path,
    completion_index: int | None = None,
    startup_only: bool = False,
) -> dict[str, Any]:
    """Run one frozen setting or one full-path capacity sentinel."""

    audit, audit_package, audit_authorization = (
        load_m03r_v11_a15_inference_audit_bundle(
            audit_plan_path=audit_plan_path,
            audit_plan_file_sha256=expected_audit_plan_file_sha256,
            package_plan_path=package_plan_path,
            package_plan_file_sha256=expected_package_plan_file_sha256,
            authorization_path=authorization_path,
            authorization_file_sha256=expected_authorization_file_sha256,
        )
    )
    expected_source_root = Path(package_plan_path).resolve().parent.parent / "source"
    if (
        not Path(__file__).resolve().is_relative_to(expected_source_root)
        or audit_package.artifacts.worker_source_sha256 != _file_sha256(Path(__file__))
        or audit_authorization.audit_plan_file_sha256 != expected_audit_plan_file_sha256
    ):
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit runtime source or plan binding drifted"
        )
    package = load_m03r_v11_package_plan(
        parent_package_plan_path,
        expected_file_sha256=audit.parent_package_plan_file_sha256,
    )
    authorization = load_m03r_v11_execution_authorization(
        parent_authorization_path,
        expected_file_sha256=(audit.parent_execution_authorization_file_sha256),
        package=package,
    )
    if (
        authorization.receipt_sha256
        != audit.parent_execution_authorization_receipt_sha256
        or package.package_plan_sha256 != audit.parent_package_plan_sha256
        or package.artifacts.source_archive_sha256 != audit.parent_source_archive_sha256
        or package.artifacts.image_reference != audit.parent_image_reference
    ):
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit parent package or authorization drifted"
        )
    _rebuild_parent_lineage(
        audit,
        parent_package_plan_path=Path(parent_package_plan_path),
        parent_authorization_path=Path(parent_authorization_path),
        parent_output_root=Path(parent_output_root),
        parent_lifecycle_root=Path(parent_lifecycle_root),
    )
    setting = 0 if startup_only else _completion_index(completion_index)
    output = Path(output_root)
    if startup_only:
        output = output / "capacity-sentinel"
    else:
        output = output / f"completion-{setting:02d}-setting-{setting:02d}"
    if output.exists() or output.is_symlink():
        raise M03RV11A15InferenceAuditWorkflowError(
            "audit output cursor already exists"
        )
    output.mkdir(parents=True, mode=0o750)
    random.seed(17)
    torch.manual_seed(17)
    device, hardware = _h100_device()
    startup_unsigned = {
        "schema": M03R_V11_A15_AUDIT_STARTUP_SCHEMA,
        "protocol_sha256": M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
        "audit_plan_file_sha256": expected_audit_plan_file_sha256,
        "audit_plan_receipt_sha256": audit.receipt_sha256,
        "audit_package_plan_file_sha256": expected_package_plan_file_sha256,
        "audit_package_plan_sha256": audit_package.package_plan_sha256,
        "audit_authorization_file_sha256": expected_authorization_file_sha256,
        "audit_authorization_receipt_sha256": audit_authorization.receipt_sha256,
        "parent_package_plan_file_sha256": (audit.parent_package_plan_file_sha256),
        "parent_package_plan_sha256": audit.parent_package_plan_sha256,
        "parent_execution_authorization_receipt_sha256": (
            audit.parent_execution_authorization_receipt_sha256
        ),
        "parent_cleanup_receipt_sha256": audit.parent_cleanup_receipt_sha256,
        "setting_index": setting,
        "mode": "capacity" if startup_only else "audit",
        "hardware": hardware,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    startup = {
        **startup_unsigned,
        "receipt_sha256": _sha256(startup_unsigned),
    }
    startup_file_sha = _write_immutable_json(output / "startup.json", startup)
    cache, risk_source, projector, risk_binding, folds = _load_common_inputs(
        package,
        setting_index=setting,
        device=device,
    )
    artifact_rows: list[dict[str, Any]] = []
    evidence_by_cursor: dict[tuple[int, str], list[M03RV11A15AuditFoldEvidence]] = {}
    selected_folds = folds[:1] if startup_only else folds
    selected_horizons = (30,) if startup_only else M03R_V11_A15_AUDIT_HORIZONS
    for fold in selected_folds:
        risk_state = build_m03r_v9_qualification_risk_state(
            cache,
            fold,
            risk_source,
            risk_binding,
            projector,
            device=device,
        )
        for horizon in selected_horizons:
            artifact_sha, evidence = _evaluate_cursor(
                audit=audit,
                package=package,
                parent_output_root=Path(parent_output_root),
                output_root=output,
                setting_index=setting,
                fold=fold,
                horizon=horizon,
                cache=cache,
                risk_source=risk_source,
                projector=projector,
                risk_binding=risk_binding,
                risk_state=risk_state,
                device=device,
            )
            artifact_rows.append(
                {
                    "fold_index": fold.fold_index,
                    "horizon_sessions": horizon,
                    "file_sha256": artifact_sha,
                }
            )
            for row in evidence:
                evidence_by_cursor.setdefault((horizon, row.variant_id), []).append(row)
        del risk_state

    if startup_only:
        unsigned = {
            "schema": M03R_V11_A15_AUDIT_CAPACITY_TERMINAL_SCHEMA,
            "protocol_sha256": M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
            "audit_plan_file_sha256": expected_audit_plan_file_sha256,
            "audit_plan_receipt_sha256": audit.receipt_sha256,
            "startup_file_sha256": startup_file_sha,
            "setting_index": 0,
            "fold_index": 0,
            "horizon_sessions": 30,
            "variant_count": len(M03R_V11_A15_AUDIT_VARIANTS),
            "cursor_artifact_file_sha256": artifact_rows[0]["file_sha256"],
            "exact_h100_80gb": True,
            "full_execution_path_proven": True,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "economic_optimizer_updates": 0,
            "outer_2026_accessed": False,
            "h100_capacity_evidence": True,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        terminal = {**unsigned, "receipt_sha256": _sha256(unsigned)}
        _write_immutable_json(output / "capacity-terminal.json", terminal)
        return terminal

    report_rows = []
    for horizon in M03R_V11_A15_AUDIT_HORIZONS:
        for variant in M03R_V11_A15_AUDIT_VARIANTS:
            report = build_m03r_v11_a15_audit_panel_report(
                tuple(evidence_by_cursor[(horizon, variant.variant_id)])
            )
            report_path = (
                output
                / "panel-reports"
                / f"horizon-{horizon:02d}-{variant.variant_id}.json"
            )
            file_sha = _write_immutable_json(report_path, asdict(report))
            report_rows.append(
                {
                    "horizon_sessions": horizon,
                    "variant_id": variant.variant_id,
                    "receipt_sha256": report.receipt_sha256,
                    "file_sha256": file_sha,
                }
            )
    unsigned = {
        "schema": M03R_V11_A15_AUDIT_WORKER_TERMINAL_SCHEMA,
        "protocol_sha256": M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
        "audit_plan_file_sha256": expected_audit_plan_file_sha256,
        "audit_plan_receipt_sha256": audit.receipt_sha256,
        "parent_package_plan_file_sha256": audit.parent_package_plan_file_sha256,
        "parent_package_plan_sha256": audit.parent_package_plan_sha256,
        "parent_execution_authorization_receipt_sha256": (
            audit.parent_execution_authorization_receipt_sha256
        ),
        "parent_cleanup_receipt_sha256": audit.parent_cleanup_receipt_sha256,
        "setting_index": setting,
        "startup_file_sha256": startup_file_sha,
        "cursor_artifacts": artifact_rows,
        "panel_reports": report_rows,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_optimizer_updates": 0,
        "economic_generation_may_be_minted": False,
        "outer_2026_accessed": False,
        "posthoc_exploratory": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    terminal = {**unsigned, "receipt_sha256": _sha256(unsigned)}
    _write_immutable_json(output / "audit-terminal.json", terminal)
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-plan", required=True)
    parser.add_argument("--audit-plan-file-sha256", required=True)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authorization-file-sha256", required=True)
    parser.add_argument("--parent-package-plan", required=True)
    parser.add_argument("--parent-authorization", required=True)
    parser.add_argument("--parent-output-root", required=True)
    parser.add_argument("--parent-lifecycle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--completion-index", type=int)
    parser.add_argument("--startup-only", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        run_m03r_v11_a15_inference_audit_worker(
            arguments.audit_plan,
            expected_audit_plan_file_sha256=(arguments.audit_plan_file_sha256),
            package_plan_path=arguments.package_plan,
            expected_package_plan_file_sha256=(arguments.package_plan_file_sha256),
            authorization_path=arguments.authorization,
            expected_authorization_file_sha256=(arguments.authorization_file_sha256),
            parent_package_plan_path=arguments.parent_package_plan,
            parent_authorization_path=arguments.parent_authorization,
            parent_output_root=arguments.parent_output_root,
            parent_lifecycle_root=arguments.parent_lifecycle_root,
            output_root=arguments.output_root,
            completion_index=arguments.completion_index,
            startup_only=arguments.startup_only,
        )
    except Exception as exc:
        output = Path(arguments.output_root)
        if arguments.startup_only:
            output = output / "capacity-sentinel"
        else:
            raw_index = (
                str(arguments.completion_index)
                if arguments.completion_index is not None
                else os.environ.get("JOB_COMPLETION_INDEX", "")
            )
            if raw_index in {"0", "1"}:
                index = int(raw_index)
                output = output / f"completion-{index:02d}-setting-{index:02d}"
        output.mkdir(parents=True, exist_ok=True)
        error = {
            "schema": M03R_V11_A15_AUDIT_WORKER_ERROR_SCHEMA,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2048],
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "economic_optimizer_updates": 0,
            "outer_2026_accessed": False,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        error["receipt_sha256"] = _sha256(error)
        try:
            _write_immutable_json(output / "audit-worker-error.json", error)
        except Exception:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "M03R_V11_A15_AUDIT_CAPACITY_TERMINAL_SCHEMA",
    "M03R_V11_A15_AUDIT_CURSOR_SCHEMA",
    "M03R_V11_A15_AUDIT_STARTUP_SCHEMA",
    "M03R_V11_A15_AUDIT_WORKER_ERROR_SCHEMA",
    "M03R_V11_A15_AUDIT_WORKER_TERMINAL_SCHEMA",
    "M03RV11A15InferenceAuditWorkflowError",
    "main",
    "run_m03r_v11_a15_inference_audit_worker",
]
