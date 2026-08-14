"""One-H100 startup gate for the frozen M03R-v12 post-hoc audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from rl_quant.evaluation.top2000_m03r_v12_posthoc_inference_audit import (
    build_m03r_v12_posthoc_causal_action_mask,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_posthoc_inference_audit import (
    M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    load_top2000_m03r_v9_risk_source,
)
from rl_quant.training.top2000_m03r_v12_checkpoint import (
    load_m03r_v12_alpha_checkpoint_for_evaluation,
)
from rl_quant.training.top2000_m03r_v12_package import load_m03r_v12_package_plan
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_package import (
    load_m03r_v12_posthoc_audit_package_plan,
)
from rl_quant.workflows.top2000_m03r_v12_posthoc_inference_audit import (
    M03RV12PosthocAuditWorkflowError,
    _new_policy,
    _sha256,
    _write_immutable_json,
)
from rl_quant.workflows.top2000_m03r_v12_posthoc_inference_audit_static_validate import (
    validate_m03r_v12_posthoc_audit_static,
)

M03R_V12_POSTHOC_AUDIT_STARTUP_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-posthoc-audit-startup-v1"
)


def validate_m03r_v12_posthoc_audit_startup(
    audit_plan_path: str | Path,
    parent_package_plan_path: str | Path,
    parent_output_root: str | Path,
    *,
    expected_audit_plan_file_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    """Prove the exact package, parent evidence, and one-H100 load boundary."""

    if (
        device != torch.device("cuda:0")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(device) != "NVIDIA H100 80GB HBM3"
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc startup requires exactly one visible H100 80GB"
        )
    mixed_device_mask = build_m03r_v12_posthoc_causal_action_mask(
        torch.ones((1, 3), dtype=torch.bool, device=device),
        torch.ones((1, 3), dtype=torch.float64, device="cpu"),
    )
    if mixed_device_mask.device != device or not torch.equal(
        mixed_device_mask,
        torch.tensor([[False, True, True]], dtype=torch.bool, device=device),
    ):
        raise M03RV12PosthocAuditWorkflowError(
            "v12 post-hoc mixed-device causal-mask regression failed"
        )
    static_result = validate_m03r_v12_posthoc_audit_static(
        audit_plan_path,
        parent_package_plan_path,
        parent_output_root,
        expected_audit_plan_file_sha256=expected_audit_plan_file_sha256,
    )
    audit_plan = load_m03r_v12_posthoc_audit_package_plan(
        audit_plan_path,
        expected_file_sha256=expected_audit_plan_file_sha256,
    )
    parent_package = load_m03r_v12_package_plan(
        parent_package_plan_path,
        expected_file_sha256=audit_plan.parent.package_plan_file_sha256,
    )
    worker = parent_package.panel.workers[0]
    package_root = Path(parent_package_plan_path).parent.parent
    cache = load_verified_top2000_hold30_development_cache(
        package_root / "cache" / "top2000-daily-bars.pt",
        expected_cache_sha256=worker.cache_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    risk_source, _written = load_top2000_m03r_v9_risk_source(
        package_root / "risk" / "risk-source-manifest.json",
        expected_manifest_file_sha256=worker.risk_source_manifest_file_sha256,
    )
    binding = audit_plan.parent.checkpoint_bindings[0]
    policy = _new_policy(0, device)
    loaded = load_m03r_v12_alpha_checkpoint_for_evaluation(
        Path(parent_output_root) / binding.checkpoint_relative_path,
        expected_file_sha256=binding.checkpoint_file_sha256,
        expected_setting_index=0,
        expected_fold_index=0,
        expected_selected_horizon_sessions=3,
        expected_episode_schedule_sha256=parent_package.schedule.receipt_sha256,
        expected_residual_operator_root_sha256=(
            binding.training_residual_operator_root_sha256
        ),
        expected_source_array_sha256=binding.training_source_array_sha256,
        expected_asset_axis_sha256=cache.action_hash,
        policy=policy,
    )
    unsigned = {
        "schema": M03R_V12_POSTHOC_AUDIT_STARTUP_SCHEMA,
        "protocol_sha256": M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
        "audit_package_plan_sha256": audit_plan.package_plan_sha256,
        "audit_package_plan_file_sha256": expected_audit_plan_file_sha256,
        "static_receipt_sha256": static_result["receipt_sha256"],
        "parent_package_plan_sha256": parent_package.package_plan_sha256,
        "checkpoint_file_sha256": binding.checkpoint_file_sha256,
        "model_state_sha256": loaded.model_state_sha256,
        "cache_sha256": worker.cache_sha256,
        "risk_source_manifest_sha256": risk_source.receipt_sha256,
        "asset_axis_sha256": cache.action_hash,
        "visible_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device),
        "exact_one_h100_80gb": True,
        "mixed_device_causal_mask_verified": True,
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
    result = validate_m03r_v12_posthoc_audit_startup(
        args.audit_plan,
        args.parent_package_plan,
        args.parent_output_root,
        expected_audit_plan_file_sha256=args.audit_plan_file_sha256,
        device=torch.device("cuda:0"),
    )
    _write_immutable_json(Path(args.output_path), result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V12_POSTHOC_AUDIT_STARTUP_SCHEMA",
    "main",
    "validate_m03r_v12_posthoc_audit_startup",
]
