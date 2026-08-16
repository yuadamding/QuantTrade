"""Exact package-slab fold update runtime for M03R-v16."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
    Top2000M03RV7DecisionStateProvider,
    top2000_m03r_v7_decision_inputs,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16FoldGeometry,
    M03RV16PanelSchedule,
    M03RV16TrainingUpdatePlan,
    render_m03r_v16_training_update_plan,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_pretraining_optimizer import (
    M03RV16OptimizerPartition,
)
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    M03RV16BuiltPredictiveBatch,
    build_m03r_v16_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v16_pretraining_step import (
    M03RV16ScoreStepReceipt,
    train_m03r_v16_score_batch_update,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    M03RV16ValidatedStructuralSlab,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)


class M03RV16TrainingRuntimeError(ValueError):
    """The V16 cache, slab, rank, schedule, or fold update drifted."""


@dataclass(frozen=True, slots=True)
class M03RV16FoldUpdateResult:
    update_plan: M03RV16TrainingUpdatePlan
    batch: M03RV16BuiltPredictiveBatch
    step: M03RV16ScoreStepReceipt

    def validate(self, slab: M03RV16ValidatedStructuralSlab) -> None:
        self.update_plan.validate()
        self.batch.validate()
        self.step.validate()
        slab.require_fast_identity()
        if (
            self.batch.structural_slab_receipt_sha256 != slab.receipt_sha256
            or self.step.batch_receipt_sha256 != self.batch.receipt_sha256
            or self.step.update_plan_sha256 != self.update_plan.receipt_sha256
        ):
            raise M03RV16TrainingRuntimeError("V16 fold update receipt chain drifted")


def move_and_bind_m03r_v16_sequence(
    sequence: Hold30Sequence,
    *,
    device: torch.device,
    asset_axis_sha256: str,
) -> Hold30Sequence:
    ledger = CohortLedger(
        economic_value=sequence.initial_ledger.economic_value.to(device),
        retention_units=sequence.initial_ledger.retention_units.to(device),
        cash_index=sequence.initial_ledger.cash_index,
    )
    return replace(
        sequence,
        decision_state=sequence.decision_state.to(device),
        asset_returns=sequence.asset_returns.to(device),
        decision_available=sequence.decision_available.to(device),
        fill_membership=sequence.fill_membership.to(device),
        fill_availability=sequence.fill_availability.to(device),
        benchmark_weights=sequence.benchmark_weights.to(device),
        risk_asset_caps=sequence.risk_asset_caps.to(device),
        risk_gross_max=sequence.risk_gross_max.to(device),
        benchmark_net_returns=sequence.benchmark_net_returns.to(device),
        initial_ledger=ledger,
        initial_equity=(
            None
            if sequence.initial_equity is None
            else sequence.initial_equity.to(device)
        ),
        track_entry_units=(
            None
            if sequence.track_entry_units is None
            else sequence.track_entry_units.to(device)
        ),
        axis_id=asset_axis_sha256,
    )


def run_m03r_v16_pretraining_fold_update(
    cache: Top2000VerifiedDevelopmentCache,
    schedule: M03RV16PanelSchedule,
    geometry: M03RV16FoldGeometry,
    risk_source: M03RV9MaterializedRiskSource,
    structural_slab: M03RV16ValidatedStructuralSlab,
    policy: Top2000M03RV16PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV16OptimizerPartition,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
    device: torch.device,
) -> M03RV16FoldUpdateResult:
    """Build and mutate one exact full-context V16 rank shard."""

    schedule.validate()
    geometry.validate()
    cache.validate_unmodified()
    risk_source.validate()
    structural_slab.require_fast_identity()
    if (
        schedule.cache_sha256 != cache.cache_sha256
        or schedule.asset_axis_sha256 != cache.action_hash
        or schedule.fold_geometry_sha256[geometry.fold_index] != geometry.receipt_sha256
        or risk_source.cache_sha256 != cache.cache_sha256
        or risk_source.action_hash != cache.action_hash
        or structural_slab.receipt.cache_sha256 != cache.cache_sha256
        or structural_slab.receipt.asset_axis_sha256 != cache.action_hash
        or structural_slab.receipt.risk_source_receipt_sha256
        != risk_source.receipt_sha256
        or structural_slab.receipt.exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
        or distributed_world_size not in {1, 2}
        or distributed_rank not in range(distributed_world_size)
    ):
        raise M03RV16TrainingRuntimeError(
            "V16 schedule, cache, risk, slab, or rank identity drifted"
        )
    update_plan = render_m03r_v16_training_update_plan(
        schedule,
        geometry,
        setting_index=policy.v16_setting.setting_index,
        completed_update=completed_updates,
    )
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=update_plan.episode_start,
        state_stop_index_exclusive=update_plan.episode_stop_exclusive,
        max_state_rows=update_plan.episode_stop_exclusive - update_plan.episode_start,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = move_and_bind_m03r_v16_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    selected_global_origins = (
        update_plan.global_origins
        if distributed_world_size == 1
        else update_plan.rank_origins[distributed_rank]
    )
    global_origins = torch.tensor(
        selected_global_origins,
        dtype=torch.long,
        device=device,
    )
    local_origins = global_origins - update_plan.episode_start
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
    policy.train()
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
        sequence_global_state_start=update_plan.episode_start,
        split="training",
        split_start_inclusive=geometry.training_origin_start_inclusive,
        split_stop_exclusive=geometry.inner_validation_origin_start_inclusive,
        fold_index=geometry.fold_index,
        source_array_sha256=built.identity.receipt_sha256,
        asset_axis_sha256=cache.action_hash,
        origin_risk_exposures=risk_source.exposures,
        structural_slab=structural_slab,
    )
    step = train_m03r_v16_score_batch_update(
        policy,
        batch,
        optimizer,
        partition,
        update_plan,
        completed_updates=completed_updates,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )
    result = M03RV16FoldUpdateResult(update_plan, batch, step)
    result.validate(structural_slab)
    return result


__all__ = [
    "M03RV16FoldUpdateResult",
    "M03RV16TrainingRuntimeError",
    "move_and_bind_m03r_v16_sequence",
    "run_m03r_v16_pretraining_fold_update",
]
