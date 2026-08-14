"""Zero-GPU validation for the frozen M03R-v12 post-hoc audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from rl_quant.protocol.hold30_alpha_m03r_v12_posthoc_inference_audit import (
    M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v12_package import load_m03r_v12_package_plan
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_package import (
    load_m03r_v12_posthoc_audit_package_plan,
)
from rl_quant.workflows.top2000_m03r_v12_posthoc_inference_audit import (
    M03RV12PosthocAuditWorkflowError,
    _file_sha256,
    _read_json,
    _sha256,
    _validate_audit_source_package,
    _validate_semantic_receipt,
    _write_immutable_json,
)

M03R_V12_POSTHOC_AUDIT_STATIC_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-static-v1"
)


def validate_m03r_v12_posthoc_audit_static(
    audit_plan_path: str | Path,
    parent_package_plan_path: str | Path,
    parent_output_root: str | Path,
    *,
    expected_audit_plan_file_sha256: str,
) -> dict[str, Any]:
    """Validate exact mounted source and parent evidence without a GPU."""

    audit_path = Path(audit_plan_path)
    audit_plan = load_m03r_v12_posthoc_audit_package_plan(
        audit_path,
        expected_file_sha256=expected_audit_plan_file_sha256,
    )
    _validate_audit_source_package(audit_path, audit_plan)
    parent_package = load_m03r_v12_package_plan(
        parent_package_plan_path,
        expected_file_sha256=audit_plan.parent.package_plan_file_sha256,
    )
    if (
        parent_package.package_plan_sha256
        != audit_plan.parent.package_plan_sha256
        or parent_package.artifacts.source_archive_sha256
        != audit_plan.parent.source_archive_sha256
        or parent_package.protocol_sha256
        != audit_plan.parent.parent_protocol_sha256
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc static parent package drifted"
        )
    parent_root = Path(parent_output_root)
    observed_terminal_files: list[str] = []
    observed_fold_files: list[str] = []
    observed_checkpoint_files: list[str] = []
    for setting, relative in enumerate(
        audit_plan.parent.predictive_terminal_relative_paths
    ):
        terminal, file_sha256 = _read_json(parent_root / relative)
        _validate_semantic_receipt(terminal)
        if (
            file_sha256
            != audit_plan.parent.predictive_terminal_file_sha256[setting]
            or terminal.get("receipt_sha256")
            != audit_plan.parent.predictive_terminal_receipt_sha256[setting]
            or terminal.get("predictive_gate_passed") is not False
            or terminal.get("economic_generation_may_be_minted") is not False
            or terminal.get("outer_2026_accessed") is not False
        ):
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc static parent terminal drifted"
            )
        observed_terminal_files.append(file_sha256)
    for binding in audit_plan.parent.checkpoint_bindings:
        fold_receipt, fold_file_sha256 = _read_json(
            parent_root / binding.parent_fold_terminal_relative_path
        )
        _validate_semantic_receipt(fold_receipt)
        checkpoint_sha256 = _file_sha256(
            parent_root / binding.checkpoint_relative_path
        )
        horizon = fold_receipt.get("horizon_candidates", {}).get("3")
        if (
            fold_file_sha256 != binding.parent_fold_terminal_file_sha256
            or fold_receipt.get("receipt_sha256")
            != binding.parent_fold_terminal_receipt_sha256
            or not isinstance(horizon, dict)
            or checkpoint_sha256 != binding.checkpoint_file_sha256
            or horizon.get("checkpoint_file_sha256") != checkpoint_sha256
            or horizon.get("model_state_sha256") != binding.model_state_sha256
            or fold_receipt.get("training_residual_operator_root_sha256")
            != binding.training_residual_operator_root_sha256
            or fold_receipt.get("training_source_array_sha256")
            != binding.training_source_array_sha256
            or fold_receipt.get("completed_updates") != 64
            or fold_receipt.get("economic_optimizer_updates") != 0
            or fold_receipt.get("outer_2026_accessed") is not False
        ):
            raise M03RV12PosthocAuditWorkflowError(
                "v12 post-hoc static fold/checkpoint lineage drifted"
            )
        observed_fold_files.append(fold_file_sha256)
        observed_checkpoint_files.append(checkpoint_sha256)
    unsigned = {
        "schema": M03R_V12_POSTHOC_AUDIT_STATIC_SCHEMA,
        "protocol_sha256": M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
        "audit_package_plan_sha256": audit_plan.package_plan_sha256,
        "audit_package_plan_file_sha256": expected_audit_plan_file_sha256,
        "parent_package_plan_sha256": parent_package.package_plan_sha256,
        "parent_package_plan_file_sha256": (
            audit_plan.parent.package_plan_file_sha256
        ),
        "parent_predictive_terminal_file_sha256": tuple(
            observed_terminal_files
        ),
        "parent_fold_terminal_root_sha256": hashlib.sha256(
            json.dumps(observed_fold_files, separators=(",", ":")).encode()
        ).hexdigest(),
        "parent_checkpoint_root_sha256": hashlib.sha256(
            json.dumps(observed_checkpoint_files, separators=(",", ":")).encode()
        ).hexdigest(),
        "parent_checkpoint_count": len(observed_checkpoint_files),
        "source_archive_sha256": audit_plan.artifacts.source_archive_sha256,
        "source_inventory_sha256": audit_plan.artifacts.source_inventory_sha256,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
        "development_only": True,
        "posthoc_exploratory": True,
        "reportable": False,
        "promotion_eligible": False,
        "passed": True,
    }
    return {**unsigned, "receipt_sha256": _sha256(unsigned)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-plan", required=True)
    parser.add_argument("--audit-plan-file-sha256", required=True)
    parser.add_argument("--parent-package-plan", required=True)
    parser.add_argument("--parent-output-root", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_m03r_v12_posthoc_audit_static(
        args.audit_plan,
        args.parent_package_plan,
        args.parent_output_root,
        expected_audit_plan_file_sha256=args.audit_plan_file_sha256,
    )
    _write_immutable_json(Path(args.output_path), result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V12_POSTHOC_AUDIT_STATIC_SCHEMA",
    "main",
    "validate_m03r_v12_posthoc_audit_static",
]
