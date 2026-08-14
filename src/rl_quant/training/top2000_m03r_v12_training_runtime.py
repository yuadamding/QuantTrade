"""Executable paired-update runtime for M03R-v12 predictive training.

This module deliberately sits above the fold-plan and optimizer-step modules.
Keeping the orchestration here prevents either immutable receipt type from
importing its mutation boundary and makes the dependency direction acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    resolve_m03r_v12_setting,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
    Top2000M03RV7DecisionStateProvider,
    Top2000M03RV7DevelopmentFold,
    top2000_m03r_v7_decision_inputs,
)
from rl_quant.training.top2000_m03r_v12_policy import Top2000M03RV12PredictivePolicy
from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
    M03RV9WrittenRiskSource,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v12_fold import (
    M03RV12TrainingShardPlan,
    render_m03r_v12_training_shard_plan,
)
from rl_quant.training.top2000_m03r_v12_predictive_worker import (
    M03RV12PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v12_pretraining_runtime import (
    build_m03r_v12_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v12_pretraining_optimizer import (
    M03RV12OptimizerPartition,
)
from rl_quant.training.top2000_m03r_v12_pretraining_step import (
    M03RV12AlphaStepReceipt,
    train_m03r_v12_predictive_batch_update,
)
from rl_quant.training.top2000_m03r_v12_schedule import (
    M03RV12PairedInputReceipt,
    M03RV12PanelEpisodeSchedule,
    build_m03r_v12_paired_input_receipt,
)


class M03RV12TrainingRuntimeError(ValueError):
    """The executable v12 fold-update lineage drifted."""


@dataclass(frozen=True, slots=True)
class M03RV12FoldUpdateResult:
    training_shard: M03RV12TrainingShardPlan
    paired_input: M03RV12PairedInputReceipt
    step_receipt: M03RV12AlphaStepReceipt

    def validate(self) -> None:
        self.training_shard.validate()
        self.paired_input.validate()
        self.step_receipt.validate()
        if (
            self.paired_input.schedule_sha256
            != self.training_shard.panel_episode_schedule_sha256
            or self.step_receipt.training_shard_receipt_sha256
            != self.training_shard.receipt_sha256
            or self.step_receipt.paired_input_receipt_sha256
            != self.paired_input.receipt_sha256
        ):
            raise M03RV12TrainingRuntimeError("v12 fold update lineage drifted")


def _move_and_bind_sequence(
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


def run_m03r_v12_pretraining_fold_update(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV12PredictiveWorkerPlan,
    schedule: M03RV12PanelEpisodeSchedule,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    written_risk_source: M03RV9WrittenRiskSource,
    policy: Top2000M03RV12PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV12OptimizerPartition,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
    device: torch.device,
) -> M03RV12FoldUpdateResult:
    """Build and execute one setting-neutral, tensor-bound v12 update."""

    worker.validate()
    schedule.validate()
    cache.validate_unmodified()
    risk_source.validate()
    if (
        distributed_world_size != worker.expected_world_size
        or distributed_rank not in range(distributed_world_size)
        or fold.fold_index not in range(worker.fold_count)
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
        raise M03RV12TrainingRuntimeError(
            "v12 worker, cache, risk, rank, fold, or initial state drifted"
        )

    shard = render_m03r_v12_training_shard_plan(
        worker,
        schedule,
        fold,
        completed_update=completed_updates,
    )
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=shard.episode_start,
        state_stop_index_exclusive=(
            shard.episode_start + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
        ),
        max_state_rows=TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = _move_and_bind_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    inputs = top2000_m03r_v7_decision_inputs(sequence)
    paired = build_m03r_v12_paired_input_receipt(
        shard,
        (inputs.daily_ohlcv, inputs.availability, inputs.past_returns),
        source_array_sha256=built.identity.receipt_sha256,
        asset_axis_sha256=cache.action_hash,
    )

    local_global_origins = torch.tensor(
        shard.rank_origins[distributed_rank],
        dtype=torch.long,
        device=device,
    )
    local_origins = local_global_origins - shard.episode_start
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
    geometry = render_m03r_v10_fold_geometry(fold)
    batch = build_m03r_v12_batch_from_origin_states(
        policy,
        resolve_m03r_v12_setting(worker.setting_index),
        origin_states,
        bound_sequence,
        local_origins,
        sequence_global_state_start=shard.episode_start,
        split="training",
        split_start_inclusive=geometry.training_state_start,
        split_stop_exclusive=geometry.optimizer_target_stop_exclusive,
        fold_index=fold.fold_index,
        source_array_sha256=built.identity.receipt_sha256,
        asset_axis_sha256=cache.action_hash,
        origin_risk_exposures=risk_source.exposures,
    )
    step = train_m03r_v12_predictive_batch_update(
        policy,
        batch,
        optimizer,
        partition,
        shard,
        paired,
        completed_updates=completed_updates,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )
    result = M03RV12FoldUpdateResult(shard, paired, step)
    result.validate()
    return result


__all__ = [
    "M03RV12FoldUpdateResult",
    "M03RV12TrainingRuntimeError",
    "run_m03r_v12_pretraining_fold_update",
]
