"""Training-only inner-validation batch construction for M03R-v16."""

from __future__ import annotations

from dataclasses import replace

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
    Top2000M03RV7DecisionStateProvider,
    top2000_m03r_v7_decision_inputs,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v16_fold import M03RV16FoldGeometry
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    M03RV16BuiltPredictiveBatch,
    build_m03r_v16_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    M03RV16ValidatedStructuralSlab,
)
from rl_quant.training.top2000_m03r_v16_training_runtime import (
    move_and_bind_m03r_v16_sequence,
)


class M03RV16EvaluationRuntimeError(ValueError):
    """The V16 inner-validation cache, fold, or slab identity drifted."""


def build_m03r_v16_inner_validation_batch(
    cache: Top2000VerifiedDevelopmentCache,
    geometry: M03RV16FoldGeometry,
    risk_source: M03RV9MaterializedRiskSource,
    structural_slab: M03RV16ValidatedStructuralSlab,
    policy: Top2000M03RV16PredictivePolicy,
    *,
    device: torch.device,
) -> M03RV16BuiltPredictiveBatch:
    """Rebuild the fixed training-only validation slice without outer access."""

    cache.validate_unmodified()
    geometry.validate()
    risk_source.validate()
    structural_slab.require_fast_identity()
    if (
        risk_source.cache_sha256 != cache.cache_sha256
        or risk_source.action_hash != cache.action_hash
        or structural_slab.receipt.cache_sha256 != cache.cache_sha256
        or structural_slab.receipt.asset_axis_sha256 != cache.action_hash
        or structural_slab.receipt.risk_source_receipt_sha256
        != risk_source.receipt_sha256
    ):
        raise M03RV16EvaluationRuntimeError(
            "V16 validation cache, risk, or slab identity drifted"
        )
    stop = geometry.training_target_stop_exclusive
    start = stop - M03R_V16_PREDICTIVE_SPEC.episode_state_rows
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=start,
        state_stop_index_exclusive=stop,
        max_state_rows=stop - start,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = move_and_bind_m03r_v16_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    global_origins = torch.arange(
        geometry.inner_validation_origin_start_inclusive,
        geometry.inner_validation_origin_stop_exclusive,
        dtype=torch.long,
        device=device,
    )
    local_origins = global_origins - start
    inputs = top2000_m03r_v7_decision_inputs(sequence)
    bound_sequence = replace(
        sequence,
        decision_state=torch.zeros(
            (*sequence.decision_state.shape[:3], 1),
            dtype=sequence.decision_state.dtype,
            device=sequence.decision_state.device,
        ),
    )
    provider = Top2000M03RV7DecisionStateProvider(inputs)
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            origin_states = provider.replay_origin_states(
                policy.source_policy,
                bound_sequence,
                local_origins,
            )
            batch = build_m03r_v16_batch_from_origin_states(
                policy,
                policy.v16_setting,
                origin_states,
                bound_sequence,
                local_origins,
                sequence_global_state_start=start,
                split="inner_validation",
                split_start_inclusive=(
                    geometry.inner_validation_origin_start_inclusive
                ),
                split_stop_exclusive=geometry.training_target_stop_exclusive,
                fold_index=geometry.fold_index,
                source_array_sha256=built.identity.receipt_sha256,
                asset_axis_sha256=cache.action_hash,
                origin_risk_exposures=risk_source.exposures,
                structural_slab=structural_slab,
            )
    finally:
        policy.train(was_training)
    if batch.policy_state_binding_sha256 != model_state_sha256(policy):
        raise M03RV16EvaluationRuntimeError(
            "V16 validation batch did not bind current model state"
        )
    return batch


__all__ = [
    "M03RV16EvaluationRuntimeError",
    "build_m03r_v16_inner_validation_batch",
]
