"""Checkpoint-bound qualification authority for M03R-v16.

This is the only public bridge from a canonical qualification batch to cohort
economics.  It prevents callers from pairing arbitrary score arrays with a
valid-looking checkpoint digest.
"""

from __future__ import annotations

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import M03R_V16_SETTINGS
from rl_quant.training.top2000_m03r_v16_checkpoint import (
    M03RV16LoadedEpochCheckpoint,
)
from rl_quant.training.top2000_m03r_v16_cohort_runtime import (
    M03RV16CohortTrace,
    run_m03r_v16_horizon_matched_cohort_sleeve,
)
from rl_quant.training.top2000_m03r_v16_fold import render_m03r_v16_fold_geometries
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    M03RV16BuiltPredictiveBatch,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    M03RV16ValidatedStructuralSlab,
)
from rl_quant.training.top2000_m03r_v9_projection import M03RV9DeviceRiskState


class M03RV16QualificationRuntimeError(ValueError):
    """The reloaded checkpoint, score batch, or slab lineage drifted."""


def run_m03r_v16_reloaded_checkpoint_cohort_qualification(
    checkpoint: M03RV16LoadedEpochCheckpoint,
    batch: M03RV16BuiltPredictiveBatch,
    structural_slab: M03RV16ValidatedStructuralSlab,
    *,
    post_fill_asset_returns: torch.Tensor,
    benchmark_weights: torch.Tensor,
    fill_available: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    risk_state: M03RV9DeviceRiskState,
) -> M03RV16CohortTrace:
    """Run economics only for scores bound to the exact reloaded checkpoint."""

    checkpoint.validate()
    batch.validate()
    structural_slab.require_fast_identity()
    geometry = render_m03r_v16_fold_geometries(1001)[checkpoint.fold_index]
    expected_origins = torch.arange(
        geometry.qualification_origin_start_inclusive,
        geometry.qualification_origin_stop_exclusive,
        dtype=torch.long,
        device=batch.origin_indices.device,
    )
    setting = M03R_V16_SETTINGS[checkpoint.setting_index]
    if (
        batch.split != "qualification"
        or batch.fold_index != checkpoint.fold_index
        or batch.objective.setting != setting
        or batch.policy_state_binding_kind != "model-state-sha256"
        or batch.policy_state_binding_sha256 != checkpoint.model_state_sha256
        or batch.asset_axis_sha256 != checkpoint.asset_axis_sha256
        or batch.source_array_sha256 != checkpoint.source_array_sha256
        or batch.structural_slab_receipt_sha256
        != structural_slab.receipt.receipt_sha256
        or checkpoint.selection_target_operator_root_sha256
        != structural_slab.receipt.common_target_operator_root_sha256
        or checkpoint.action_operator_root_sha256
        != structural_slab.receipt.action_operator_root_sha256
        or not torch.equal(batch.origin_indices, expected_origins)
    ):
        raise M03RV16QualificationRuntimeError(
            "V16 checkpoint-to-score qualification lineage drifted"
        )
    return run_m03r_v16_horizon_matched_cohort_sleeve(
        setting,
        fold_index=checkpoint.fold_index,
        checkpoint_file_sha256=checkpoint.checkpoint_file_sha256,
        checkpoint_model_state_sha256=checkpoint.model_state_sha256,
        qualification_batch_receipt_sha256=batch.receipt_sha256,
        asset_axis_sha256=batch.asset_axis_sha256,
        decision_origin_indices=batch.origin_indices,
        executable_selection_scores=(
            batch.objective.executable_selection_score_z
            * setting.selection_target_scale
        ),
        action_valid=batch.action_valid,
        diagnostic_valid=batch.objective.selection_valid,
        post_fill_asset_returns=post_fill_asset_returns,
        benchmark_weights=benchmark_weights,
        fill_available=fill_available,
        risk_asset_caps=risk_asset_caps,
        risk_gross_max=risk_gross_max,
        risk_state=risk_state,
    )


__all__ = [
    "M03RV16QualificationRuntimeError",
    "run_m03r_v16_reloaded_checkpoint_cohort_qualification",
]
