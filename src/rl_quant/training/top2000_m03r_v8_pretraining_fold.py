"""Cache-to-step integration for one M03R-v8 predictive fold."""

from __future__ import annotations

from dataclasses import replace

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.hold30_top2000_development import (
    Top2000Hold30DevelopmentSequence,
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
from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03RV8AlphaFoldEvidence,
    build_m03r_v8_alpha_fold_evidence,
)
from rl_quant.training.top2000_m03r_v8_plan import (
    M03R_V8_MAX_TARGET_HORIZON,
    M03RV8DevelopmentTrainingPlan,
    deterministic_m03r_v8_pretraining_episode_start,
    render_m03r_v8_fold_pretraining_geometry,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_optimizer import (
    M03RV8AlphaOptimizerPartition,
)
from rl_quant.training.top2000_m03r_v8_pretraining_runtime import (
    build_m03r_v8_alpha_pretraining_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v8_pretraining_step import (
    M03RV8AlphaStepReceipt,
    train_m03r_v8_alpha_pretraining_update,
)


class M03RV8PretrainingFoldError(ValueError):
    """The cache slice, rank shard, or fold binding is invalid."""


def _move_sequence(sequence: Hold30Sequence, device: torch.device) -> Hold30Sequence:
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
            None if sequence.initial_equity is None else sequence.initial_equity.to(device)
        ),
        track_entry_units=(
            None
            if sequence.track_entry_units is None
            else sequence.track_entry_units.to(device)
        ),
    )


def _episode(
    cache: Top2000VerifiedDevelopmentCache,
    *,
    start: int,
    device: torch.device,
) -> tuple[Top2000Hold30DevelopmentSequence, Hold30Sequence]:
    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=start,
        state_stop_index_exclusive=start + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
        max_state_rows=TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    return built, _move_sequence(built.sequence, device)


def _states_for_origins(
    policy: Top2000M03RV8DevelopmentPolicy,
    sequence: Hold30Sequence,
    origins: torch.Tensor,
) -> tuple[Hold30Sequence, torch.Tensor]:
    inputs = top2000_m03r_v7_decision_inputs(sequence)
    bound = replace(
        sequence,
        decision_state=torch.zeros(
            (*sequence.decision_state.shape[:3], 1),
            dtype=sequence.decision_state.dtype,
            device=sequence.decision_state.device,
        ),
    )
    provider = Top2000M03RV7DecisionStateProvider(inputs)
    states = provider.replay_origin_states(policy, bound, origins)
    return bound, states


def run_m03r_v8_pretraining_fold_update(
    cache: Top2000VerifiedDevelopmentCache,
    plan: M03RV8DevelopmentTrainingPlan,
    fold: Top2000M03RV7DevelopmentFold,
    policy: Top2000M03RV8DevelopmentPolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV8AlphaOptimizerPartition,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
    device: torch.device,
) -> M03RV8AlphaStepReceipt:
    """Build one deterministic rank shard and perform one optimizer update."""

    plan.validate()
    cache.validate_unmodified()
    if (
        plan.setting_index != policy.setting.setting_index
        or not plan.alpha_pretraining_required
        or fold.fold_index not in range(plan.fold_count)
        or distributed_world_size != plan.expected_world_size
    ):
        raise M03RV8PretrainingFoldError("plan, policy, fold, or rank geometry drifted")
    geometry = render_m03r_v8_fold_pretraining_geometry(fold)
    episode_start = deterministic_m03r_v8_pretraining_episode_start(
        plan,
        fold,
        completed_updates=completed_updates,
    )
    built, sequence = _episode(cache, start=episode_start, device=device)
    last_global_origin = min(
        geometry.optimizer_target_stop_exclusive - M03R_V8_MAX_TARGET_HORIZON - 1,
        episode_start + sequence.asset_returns.shape[0] - M03R_V8_MAX_TARGET_HORIZON,
    )
    if last_global_origin < episode_start:
        raise M03RV8PretrainingFoldError("episode has no split-bounded training origin")
    global_origins = torch.arange(
        episode_start,
        last_global_origin + 1,
        dtype=torch.long,
        device=device,
    )
    if global_origins.numel() % distributed_world_size:
        global_origins = global_origins[:-1]
    if global_origins.numel() < distributed_world_size:
        raise M03RV8PretrainingFoldError(
            "training episode cannot provide an equal nonempty rank shard"
        )
    local_global = global_origins[distributed_rank::distributed_world_size]
    local_origins = local_global - episode_start
    if local_origins.numel() == 0:
        raise M03RV8PretrainingFoldError("rank received no predictive origins")
    bound, states = _states_for_origins(policy, sequence, local_origins)
    batch = build_m03r_v8_alpha_pretraining_batch_from_origin_states(
        policy,
        states,
        bound,
        local_origins,
        sequence_global_state_start=episode_start,
        split="training",
        split_start_inclusive=geometry.training_state_start,
        split_stop_exclusive=geometry.optimizer_target_stop_exclusive,
        fold_index=fold.fold_index,
        source_array_sha256=built.identity.receipt_sha256,
    )
    return train_m03r_v8_alpha_pretraining_update(
        policy,
        batch,
        optimizer,
        partition,
        completed_updates=completed_updates,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )


def evaluate_m03r_v8_pretraining_fold(
    cache: Top2000VerifiedDevelopmentCache,
    plan: M03RV8DevelopmentTrainingPlan,
    fold: Top2000M03RV7DevelopmentFold,
    policy: Top2000M03RV8DevelopmentPolicy,
    *,
    device: torch.device,
) -> M03RV8AlphaFoldEvidence:
    """Evaluate all 64 held-out training-tail origins without parameter mutation."""

    plan.validate()
    cache.validate_unmodified()
    geometry = render_m03r_v8_fold_pretraining_geometry(fold)
    built, sequence = _episode(
        cache,
        start=geometry.validation_episode_state_start,
        device=device,
    )
    global_origins = torch.arange(
        geometry.inner_validation_start_inclusive,
        geometry.inner_validation_origin_stop_exclusive,
        dtype=torch.long,
        device=device,
    )
    local_origins = global_origins - geometry.validation_episode_state_start
    policy.eval()
    with torch.no_grad():
        bound, states = _states_for_origins(policy, sequence, local_origins)
        batch = build_m03r_v8_alpha_pretraining_batch_from_origin_states(
            policy,
            states,
            bound,
            local_origins,
            sequence_global_state_start=geometry.validation_episode_state_start,
            split="inner-validation",
            split_start_inclusive=geometry.inner_validation_start_inclusive,
            split_stop_exclusive=geometry.inner_validation_target_stop_exclusive,
            fold_index=fold.fold_index,
            source_array_sha256=built.identity.receipt_sha256,
        )
        evidence = build_m03r_v8_alpha_fold_evidence(batch)
    policy.train()
    return evidence


__all__ = [
    "M03RV8PretrainingFoldError",
    "evaluate_m03r_v8_pretraining_fold",
    "run_m03r_v8_pretraining_fold_update",
]
