"""One-GPU worker for the frozen M03R-v12 post-hoc inference audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from rl_quant.evaluation.top2000_m03r_v12_posthoc_inference_audit import (
    M03RV12PosthocAuditFoldEvidence,
    build_m03r_v12_posthoc_audit_fold_evidence,
    build_m03r_v12_posthoc_audit_panel_report,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_posthoc_inference_audit import (
    M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
    M03R_V12_POSTHOC_AUDIT_VARIANTS,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    load_top2000_m03r_v9_risk_source,
)
from rl_quant.training.top2000_m03r_v12_checkpoint import (
    load_m03r_v12_alpha_checkpoint_for_evaluation,
)
from rl_quant.training.top2000_m03r_v12_package import load_m03r_v12_package_plan
from rl_quant.training.top2000_m03r_v12_policy import Top2000M03RV12PredictivePolicy
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_fold import (
    build_m03r_v12_posthoc_audit_fold_inputs,
)
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_package import (
    M03RV12PosthocAuditPackagePlan,
    load_m03r_v12_posthoc_audit_package_plan,
)

M03R_V12_POSTHOC_AUDIT_WORKER_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-worker-terminal-v1"
)
_MAX_JSON_BYTES = 4 * 1024 * 1024


class M03RV12PosthocAuditWorkflowError(RuntimeError):
    """The exact parent lineage or post-hoc audit worker drifted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_audit_source_package(
    audit_plan_path: Path,
    audit_plan: M03RV12PosthocAuditPackagePlan,
) -> None:
    package_root = audit_plan_path.parent.parent
    expected = (
        (
            package_root / "source.tar",
            audit_plan.artifacts.source_archive_sha256,
        ),
        (
            package_root / "source-manifest.json",
            audit_plan.artifacts.source_manifest_file_sha256,
        ),
        (
            package_root / "source" / "uv.lock",
            audit_plan.artifacts.dependency_lock_sha256,
        ),
        (
            package_root
            / "source"
            / "src/rl_quant/workflows/top2000_m03r_v12_posthoc_inference_audit.py",
            audit_plan.artifacts.worker_source_sha256,
        ),
    )
    for path, digest in expected:
        if _file_sha256(path) != digest:
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc source package drifted"
            )
    manifest, _manifest_file_sha256 = _read_json(
        package_root / "source-manifest.json"
    )
    files = manifest.get("files")
    if (
        manifest.get("protocol_sha256")
        != M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256
        or manifest.get("source_inventory_sha256")
        != audit_plan.artifacts.source_inventory_sha256
        or manifest.get("training_authorized") is not False
        or manifest.get("outer_2026_access_authorized") is not False
        or not isinstance(files, list)
        or manifest.get("file_count") != len(files)
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc source manifest drifted"
        )
    if hashlib.sha256(_canonical(tuple(files)) + b"\n").hexdigest() != (
        audit_plan.artifacts.source_inventory_sha256
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc source inventory drifted"
        )
    for row in files:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
        ):
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc source inventory row drifted"
            )
        path = package_root / "source" / row["path"]
        if not path.resolve().is_relative_to((package_root / "source").resolve()):
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc source inventory leaves its package"
            )
        if _file_sha256(path) != row["sha256"]:
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc source member drifted"
            )


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc parent receipt is unavailable"
        ) from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or not 0 < status.st_size <= _MAX_JSON_BYTES:
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc parent receipt is not a bounded regular file"
            )
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, min(1024 * 1024, status.st_size + 1)):
            observed += len(chunk)
            if observed > _MAX_JSON_BYTES:
                raise M03RV12PosthocAuditWorkflowError(
                    "v12 post-hoc parent receipt exceeds its bound"
                )
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            after.st_ino != status.st_ino
            or after.st_size != status.st_size
            or after.st_mtime_ns != status.st_mtime_ns
        ):
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc parent receipt changed during its read"
            )
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc parent receipt is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc parent receipt is not an object"
        )
    return value, hashlib.sha256(raw).hexdigest()


def _validate_semantic_receipt(value: dict[str, Any]) -> None:
    expected = value.get("receipt_sha256")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    parent_digest = hashlib.sha256(_canonical(unsigned) + b"\n").hexdigest()
    if not isinstance(expected, str) or expected != parent_digest:
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc parent semantic receipt drifted"
        )


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    raw = _canonical(payload) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _write_immutable_torch(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return _file_sha256(path)


def _new_policy(setting_index: int, device: torch.device) -> Top2000M03RV12PredictivePolicy:
    return Top2000M03RV12PredictivePolicy(
        setting_index,
        selected_horizon_sessions=3,
    ).to(device)


def run_m03r_v12_posthoc_audit_setting(
    audit_plan_path: str | Path,
    parent_package_plan_path: str | Path,
    parent_output_root: str | Path,
    output_root: str | Path,
    *,
    expected_audit_plan_file_sha256: str,
    setting_index: int,
    device: torch.device,
) -> dict[str, Any]:
    """Audit one completed v12 setting without training or 2026 access."""

    if setting_index not in range(3):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc setting index must be 0, 1, or 2"
        )
    if (
        device != torch.device("cuda:0")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(device) != "NVIDIA H100 80GB HBM3"
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc evidence requires exactly one visible H100 80GB"
        )
    audit_path = Path(audit_plan_path)
    audit_plan = load_m03r_v12_posthoc_audit_package_plan(
        audit_path,
        expected_file_sha256=expected_audit_plan_file_sha256,
    )
    _validate_audit_source_package(audit_path, audit_plan)
    package_path = Path(parent_package_plan_path)
    package = load_m03r_v12_package_plan(
        package_path,
        expected_file_sha256=audit_plan.parent.package_plan_file_sha256,
    )
    if (
        package.package_plan_sha256 != audit_plan.parent.package_plan_sha256
        or package.artifacts.source_archive_sha256
        != audit_plan.parent.source_archive_sha256
        or package.protocol_sha256 != audit_plan.parent.parent_protocol_sha256
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc parent package does not match the audit plan"
        )
    worker = package.panel.workers[setting_index]
    worker.validate()
    package_root = package_path.parent.parent
    cache = load_verified_top2000_hold30_development_cache(
        package_root / "cache" / "top2000-daily-bars.pt",
        expected_cache_sha256=worker.cache_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    risk_source, _written = load_top2000_m03r_v9_risk_source(
        package_root / "risk" / "risk-source-manifest.json",
        expected_manifest_file_sha256=worker.risk_source_manifest_file_sha256,
    )
    folds = render_top2000_m03r_v7_development_folds(
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
    )
    parent = (
        Path(parent_output_root)
        / f"completion-{setting_index:02d}-setting-{setting_index:02d}"
    )
    terminal, terminal_file_sha256 = _read_json(parent / "predictive-terminal.json")
    _validate_semantic_receipt(terminal)
    if (
        terminal_file_sha256
        != audit_plan.parent.predictive_terminal_file_sha256[setting_index]
        or terminal.get("receipt_sha256")
        != audit_plan.parent.predictive_terminal_receipt_sha256[setting_index]
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc parent terminal is not audit-plan bound"
        )
    expected_fold_files = terminal.get("fold_terminal_file_sha256")
    if (
        terminal.get("schema")
        != "rl-quant.top2000-dev.m03r-v12-predictive-worker-terminal-v1"
        or terminal.get("setting_index") != setting_index
        or terminal.get("predictive_gate_passed") is not False
        or terminal.get("selected_horizon") is not None
        or terminal.get("economic_generation_may_be_minted") is not False
        or terminal.get("outer_2026_accessed") is not False
        or terminal.get("economic_optimizer_updates") != 0
        or terminal.get("package_plan_sha256") != package.package_plan_sha256
        or not isinstance(expected_fold_files, list)
        or len(expected_fold_files) != 6
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc parent terminal is not the exact failed predictive study"
        )

    output = Path(output_root)
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    by_variant: dict[str, list[M03RV12PosthocAuditFoldEvidence]] = {
        row.variant_id: [] for row in M03R_V12_POSTHOC_AUDIT_VARIANTS
    }
    fold_artifact_file_sha256: list[str] = []
    parent_fold_terminal_file_sha256: list[str] = []
    for fold in folds:
        receipt_path = parent / "receipts" / f"fold-{fold.fold_index:02d}-terminal.json"
        receipt, receipt_file_sha256 = _read_json(receipt_path)
        _validate_semantic_receipt(receipt)
        horizon = receipt.get("horizon_candidates", {}).get("3")
        audit_binding = audit_plan.parent.checkpoint_bindings[
            setting_index * 6 + fold.fold_index
        ]
        if (
            receipt_file_sha256 != expected_fold_files[fold.fold_index]
            or receipt.get("setting_index") != setting_index
            or receipt.get("fold_index") != fold.fold_index
            or receipt.get("completed_updates") != 64
            or receipt.get("economic_optimizer_updates") != 0
            or receipt.get("outer_2026_accessed") is not False
            or not isinstance(horizon, dict)
            or receipt_file_sha256
            != audit_binding.parent_fold_terminal_file_sha256
            or receipt.get("receipt_sha256")
            != audit_binding.parent_fold_terminal_receipt_sha256
        ):
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc fold terminal lineage drifted"
            )
        checkpoint_path = (
            parent
            / "checkpoints"
            / f"fold-{fold.fold_index:02d}-horizon-03-update-0064.pt"
        )
        checkpoint_file_sha256 = horizon.get("checkpoint_file_sha256")
        training_residual_root = receipt.get(
            "training_residual_operator_root_sha256"
        )
        training_source_array = receipt.get("training_source_array_sha256")
        if (
            not isinstance(checkpoint_file_sha256, str)
            or not isinstance(training_residual_root, str)
            or not isinstance(training_source_array, str)
            or checkpoint_file_sha256 != audit_binding.checkpoint_file_sha256
            or horizon.get("model_state_sha256")
            != audit_binding.model_state_sha256
            or training_residual_root
            != audit_binding.training_residual_operator_root_sha256
            or training_source_array != audit_binding.training_source_array_sha256
        ):
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc fold terminal omits checkpoint lineage"
            )
        policy = _new_policy(setting_index, device)
        loaded = load_m03r_v12_alpha_checkpoint_for_evaluation(
            checkpoint_path,
            expected_file_sha256=checkpoint_file_sha256,
            expected_setting_index=setting_index,
            expected_fold_index=fold.fold_index,
            expected_selected_horizon_sessions=3,
            expected_episode_schedule_sha256=package.schedule.receipt_sha256,
            expected_residual_operator_root_sha256=training_residual_root,
            expected_source_array_sha256=training_source_array,
            expected_asset_axis_sha256=cache.action_hash,
            policy=policy,
        )
        fold_inputs = build_m03r_v12_posthoc_audit_fold_inputs(
            cache,
            worker,
            fold,
            risk_source,
            policy,
            loaded,
            device=device,
        )
        evidence = tuple(
            build_m03r_v12_posthoc_audit_fold_evidence(fold_inputs, variant)
            for variant in M03R_V12_POSTHOC_AUDIT_VARIANTS
        )
        for row in evidence:
            by_variant[row.variant_id].append(row)
        artifact_sha = _write_immutable_torch(
            output / "fold-artifacts" / f"fold-{fold.fold_index:02d}.pt",
            {
                "schema": "rl-quant.top2000-dev.m03r-v12-posthoc-audit-fold-bundle-v1",
                "protocol_sha256": M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
                "setting_index": setting_index,
                "fold_index": fold.fold_index,
                "parent_fold_terminal_file_sha256": receipt_file_sha256,
                "input_receipt_sha256": fold_inputs.receipt_sha256,
                "evidence": tuple(asdict(row) for row in evidence),
                "economic_optimizer_updates": 0,
                "outer_2026_accessed": False,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            },
        )
        fold_artifact_file_sha256.append(artifact_sha)
        parent_fold_terminal_file_sha256.append(receipt_file_sha256)
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reports = tuple(
        build_m03r_v12_posthoc_audit_panel_report(tuple(by_variant[variant.variant_id]))
        for variant in M03R_V12_POSTHOC_AUDIT_VARIANTS
    )
    unsigned = {
        "schema": M03R_V12_POSTHOC_AUDIT_WORKER_TERMINAL_SCHEMA,
        "protocol_sha256": M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
        "audit_package_plan_sha256": audit_plan.package_plan_sha256,
        "audit_package_plan_file_sha256": expected_audit_plan_file_sha256,
        "source_protocol_sha256": package.protocol_sha256,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": audit_plan.parent.package_plan_file_sha256,
        "setting_index": setting_index,
        "setting_id": worker.setting_id,
        "parent_predictive_terminal_file_sha256": terminal_file_sha256,
        "parent_predictive_terminal_receipt_sha256": terminal["receipt_sha256"],
        "parent_fold_terminal_file_sha256": tuple(
            parent_fold_terminal_file_sha256
        ),
        "fold_artifact_file_sha256": tuple(fold_artifact_file_sha256),
        "visible_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device),
        "exact_one_h100_80gb": True,
        "panel_reports": tuple(asdict(row) for row in reports),
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
    terminal_result = {**unsigned, "receipt_sha256": _sha256(unsigned)}
    _write_immutable_json(output / "audit-terminal.json", terminal_result)
    return terminal_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-plan", required=True)
    parser.add_argument("--audit-plan-file-sha256", required=True)
    parser.add_argument("--parent-package-plan", required=True)
    parser.add_argument("--parent-output-root", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-root")
    output.add_argument("--indexed-output-root")
    parser.add_argument("--setting-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    setting_index = args.setting_index
    if setting_index is None:
        raw_index = os.environ.get("JOB_COMPLETION_INDEX")
        if raw_index not in {"0", "1", "2"}:
            raise M03RV12PosthocAuditWorkflowError(
                "JOB_COMPLETION_INDEX must be exactly 0, 1, or 2"
            )
        setting_index = int(raw_index)
    if args.indexed_output_root is not None:
        output_root = (
            Path(args.indexed_output_root)
            / f"completion-{setting_index:02d}-setting-{setting_index:02d}"
        )
    else:
        output_root = Path(args.output_root)
    device = torch.device("cuda:0")
    result = run_m03r_v12_posthoc_audit_setting(
        args.audit_plan,
        args.parent_package_plan,
        args.parent_output_root,
        output_root,
        expected_audit_plan_file_sha256=args.audit_plan_file_sha256,
        setting_index=setting_index,
        device=device,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V12_POSTHOC_AUDIT_WORKER_TERMINAL_SCHEMA",
    "M03RV12PosthocAuditWorkflowError",
    "main",
    "run_m03r_v12_posthoc_audit_setting",
]
