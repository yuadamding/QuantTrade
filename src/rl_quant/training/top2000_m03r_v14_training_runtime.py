"""Executable paired fold-update runtime for M03R-v14 predictive training."""

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
from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
    M03RV9WrittenRiskSource,
)
from rl_quant.training.top2000_m03r_v14_fold import (
    M03RV14FoldGeometry,
    M03RV14PairedInputBinding,
    M03RV14PanelEpisodeSchedule,
    M03RV14TrainingUpdatePlan,
    build_m03r_v14_paired_input_binding,
    render_m03r_v14_training_update_plan,
)
from rl_quant.training.top2000_m03r_v14_policy import (
    Top2000M03RV14PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v14_predictive_worker import (
    M03RV14PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v14_pretraining_optimizer import (
    M03RV14OptimizerPartition,
)
from rl_quant.training.top2000_m03r_v14_pretraining_runtime import (
    build_m03r_v14_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v14_pretraining_step import (
    M03RV14AlphaStepReceipt,
    train_m03r_v14_predictive_batch_update,
)


class M03RV14TrainingRuntimeError(ValueError):
    """The v14 executable fold update drifted before mutation."""


@dataclass(frozen=True, slots=True)
class M03RV14FoldUpdateResult:
    update_plan: M03RV14TrainingUpdatePlan
    paired_input: M03RV14PairedInputBinding
    step: M03RV14AlphaStepReceipt

    def validate(self) -> None:
        self.update_plan.validate()
        self.paired_input.validate()
        self.step.validate()
        if (
            self.step.training_update_plan_sha256 != self.update_plan.receipt_sha256
            or self.step.paired_input_binding_sha256
            != self.paired_input.receipt_sha256
        ):
            raise M03RV14TrainingRuntimeError("v14 fold-update receipt chain drifted")


def move_and_bind_m03r_v14_sequence(
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


def run_m03r_v14_pretraining_fold_update(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV14PredictiveWorkerPlan,
    schedule: M03RV14PanelEpisodeSchedule,
    geometry: M03RV14FoldGeometry,
    risk_source: M03RV9MaterializedRiskSource,
    written_risk_source: M03RV9WrittenRiskSource,
    policy: Top2000M03RV14PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV14OptimizerPartition,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
    device: torch.device,
) -> M03RV14FoldUpdateResult:
    """Build and execute one setting-neutral, full-context v14 update."""

    worker.validate()
    schedule.validate()
    geometry.validate()
    cache.validate_unmodified()
    risk_source.validate()
    if (
        policy.v14_setting.setting_index != worker.setting_index
        or schedule.receipt_sha256 != worker.panel_episode_schedule_sha256
        or schedule.cache_sha256 != cache.cache_sha256
        or schedule.asset_axis_sha256 != cache.action_hash
        or geometry.fold_index not in range(worker.fold_count)
        or geometry.optimizer_updates
        != worker.fold_optimizer_updates[geometry.fold_index]
        or distributed_world_size != worker.expected_world_size
        or distributed_rank not in range(distributed_world_size)
        or risk_source.cache_sha256 != cache.cache_sha256
        or risk_source.action_hash != cache.action_hash
        or worker.cache_sha256 != cache.cache_sha256
        or worker.risk_source_manifest_file_sha256
        != written_risk_source.manifest_file_sha256
        or (
            completed_updates == 0
            and model_state_sha256(policy) != worker.initial_parameter_state_sha256
        )
    ):
        raise M03RV14TrainingRuntimeError(
            "v14 worker, schedule, cache, risk, rank, fold, or initial state drifted"
        )
    update_plan = render_m03r_v14_training_update_plan(
        schedule,
        geometry,
        setting_index=worker.setting_index,
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
    sequence = move_and_bind_m03r_v14_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    paired_input = build_m03r_v14_paired_input_binding(
        update_plan,
        cache_sha256=cache.cache_sha256,
        source_array_sha256=built.identity.receipt_sha256,
        asset_axis_sha256=cache.action_hash,
    )
    local_global_origins = torch.tensor(
        update_plan.rank_origins[distributed_rank],
        dtype=torch.long,
        device=device,
    )
    local_origins = local_global_origins - update_plan.episode_start
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
    origin_states = provider.replay_origin_states(
        policy.source_policy,
        bound_sequence,
        local_origins,
    )
    batch = build_m03r_v14_batch_from_origin_states(
        policy,
        policy.v14_setting,
        origin_states,
        bound_sequence,
        local_origins,
        sequence_global_state_start=update_plan.episode_start,
        split="training",
        split_start_inclusive=geometry.training_state_start,
        split_stop_exclusive=geometry.training_target_stop_exclusive,
        fold_index=geometry.fold_index,
        source_array_sha256=built.identity.receipt_sha256,
        asset_axis_sha256=cache.action_hash,
        origin_risk_exposures=risk_source.exposures,
    )
    step = train_m03r_v14_predictive_batch_update(
        policy,
        batch,
        optimizer,
        partition,
        update_plan,
        paired_input,
        completed_updates=completed_updates,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )
    result = M03RV14FoldUpdateResult(update_plan, paired_input, step)
    result.validate()
    return result


__all__ = [
    "M03RV14FoldUpdateResult",
    "M03RV14TrainingRuntimeError",
    "move_and_bind_m03r_v14_sequence",
    "run_m03r_v14_pretraining_fold_update",
]
