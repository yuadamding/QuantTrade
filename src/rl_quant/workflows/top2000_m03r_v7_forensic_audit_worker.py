"""One-H100 Indexed-Job worker for the M03R-v7 Phase-0 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from rl_quant.workflows.top2000_m03r_v7_forensic_audit import (
    M03RV7ForensicWorkflowError,
    _canonical_json,
    _file_sha256,
    _sha256,
    _write_immutable_json,
    run_m03r_v7_seed17_forensic_setting,
)

WORKER_SCHEMA = "rl-quant.top2000-dev.m03r-v7-seed17-forensic-worker-v1"


def _read_manifest(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise M03RV7ForensicWorkflowError("audit package manifest is not regular")
    if _file_sha256(path) != expected_file_sha256:
        raise M03RV7ForensicWorkflowError("audit package manifest hash drifted")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise M03RV7ForensicWorkflowError("cannot decode audit package manifest") from exc
    if not isinstance(value, dict):
        raise M03RV7ForensicWorkflowError("audit package manifest must be an object")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise M03RV7ForensicWorkflowError("audit package receipt hash drifted")
    return value


def _resolve_setting(
    manifest: Mapping[str, Any], allowed_setting_indices: tuple[int, ...]
) -> dict[str, Any]:
    raw_local = os.environ.get("JOB_COMPLETION_INDEX")
    if raw_local is None or not raw_local.isdecimal():
        raise M03RV7ForensicWorkflowError("JOB_COMPLETION_INDEX is missing or invalid")
    local_index = int(raw_local)
    if not 0 <= local_index < len(allowed_setting_indices):
        raise M03RV7ForensicWorkflowError("local completion index is outside its map")
    setting_index = allowed_setting_indices[local_index]
    rows = manifest.get("setting_inputs")
    if not isinstance(rows, list):
        raise M03RV7ForensicWorkflowError("manifest setting_inputs is invalid")
    matches = [row for row in rows if isinstance(row, dict) and row.get("setting_index") == setting_index]
    if len(matches) != 1:
        raise M03RV7ForensicWorkflowError("scientific setting did not resolve uniquely")
    row = dict(matches[0])
    row["local_completion_index"] = local_index
    return row


def _validate_one_h100() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M03RV7ForensicWorkflowError("audit worker requires exactly one visible CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name.upper():
        raise M03RV7ForensicWorkflowError(
            f"audit worker requires one H100, observed {device_name!r}"
        )
    return {
        "cuda_available": True,
        "visible_cuda_device_count": 1,
        "cuda_device_name": device_name,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "h100_startup_guard_passed": True,
    }


def run_worker(
    *,
    parent_run_root: Path,
    cache_path: Path,
    audit_package_manifest: Path,
    audit_package_manifest_file_sha256: str,
    audit_output_root: Path,
    allowed_setting_indices: tuple[int, ...],
) -> dict[str, Any]:
    if not allowed_setting_indices or len(set(allowed_setting_indices)) != len(
        allowed_setting_indices
    ):
        raise M03RV7ForensicWorkflowError("allowed setting map must be nonempty and unique")
    if any(isinstance(value, bool) or not 0 <= value < 12 for value in allowed_setting_indices):
        raise M03RV7ForensicWorkflowError("allowed setting map contains an invalid setting")
    manifest = _read_manifest(
        audit_package_manifest, audit_package_manifest_file_sha256
    )
    if (
        manifest.get("schema")
        != "rl-quant.top2000-dev.m03r-v7-phase0-audit-package-v1"
        or manifest.get("setting_count") != 12
        or manifest.get("fold_count_per_setting") != 6
        or manifest.get("retraining_authorized") is not False
        or manifest.get("checkpoint_selection_authorized") is not False
        or manifest.get("development_only") is not True
        or manifest.get("reportable") is not False
        or manifest.get("promotable") is not False
    ):
        raise M03RV7ForensicWorkflowError("audit package semantic contract drifted")
    selected = _resolve_setting(manifest, allowed_setting_indices)
    setting_index = selected.get("setting_index")
    completion_index = selected.get("completion_index")
    plan_sha256 = selected.get("training_plan_file_sha256")
    if (
        not isinstance(setting_index, int)
        or not isinstance(completion_index, int)
        or not isinstance(plan_sha256, str)
    ):
        raise M03RV7ForensicWorkflowError("resolved setting input is malformed")
    runtime_proof = _validate_one_h100()
    setting_root = (
        parent_run_root
        / f"completion-{completion_index:02d}-setting-{setting_index:02d}"
    )
    output_root = audit_output_root / f"setting-{setting_index:02d}"
    result = run_m03r_v7_seed17_forensic_setting(
        setting_root=setting_root,
        cache_path=cache_path,
        expected_cache_sha256=str(manifest["original_cache_sha256"]),
        expected_training_plan_file_sha256=plan_sha256,
        evaluation_source_inventory_sha256=str(manifest["source_inventory_sha256"]),
        source_training_archive_sha256=str(manifest["original_source_archive_sha256"]),
        output_root=output_root,
        device="cuda:0",
    )
    payload: dict[str, Any] = {
        "schema": WORKER_SCHEMA,
        "audit_package_manifest_file_sha256": audit_package_manifest_file_sha256,
        "audit_package_manifest_receipt_sha256": manifest["receipt_sha256"],
        "allowed_setting_indices": list(allowed_setting_indices),
        "local_completion_index": selected["local_completion_index"],
        "scientific_setting_index": setting_index,
        "parent_completion_index": completion_index,
        "training_plan_file_sha256": plan_sha256,
        "worker_runtime_proof": runtime_proof,
        "setting_receipt_path": result["receipt_path"],
        "setting_receipt_file_sha256": result["receipt_file_sha256"],
        "setting_receipt_sha256": result["receipt_sha256"],
        "fold_count": result["fold_count"],
        "cache_load_count": result["cache_load_count"],
        "retraining_performed": False,
        "checkpoint_selection_performed": False,
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    receipt_path = output_root / "forensic-worker-completion.json"
    file_sha256 = _write_immutable_json(receipt_path, payload)
    return {
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": file_sha256,
        "receipt_sha256": _sha256(payload),
        "setting_index": setting_index,
        "fold_count": 6,
    }


def _parse_setting_indices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("setting map must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("setting map may not be empty")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--audit-package-manifest", type=Path, required=True)
    parser.add_argument("--audit-package-manifest-file-sha256", required=True)
    parser.add_argument("--audit-output-root", type=Path, required=True)
    parser.add_argument(
        "--allowed-setting-indices", type=_parse_setting_indices, required=True
    )
    args = parser.parse_args(argv)
    result = run_worker(
        parent_run_root=args.parent_run_root,
        cache_path=args.cache_path,
        audit_package_manifest=args.audit_package_manifest,
        audit_package_manifest_file_sha256=args.audit_package_manifest_file_sha256,
        audit_output_root=args.audit_output_root,
        allowed_setting_indices=args.allowed_setting_indices,
    )
    print(_canonical_json(result).decode())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["WORKER_SCHEMA", "main", "run_worker"]
