"""Causal target and prediction binding for M03R-v8 alpha pretraining.

This boundary consumes differentiable decision states from the raw encoder and
builds benchmark-relative future log-return targets without reading beyond the
declared training or inner-validation split.  It does not own optimizer or
early-stopping hyperparameters; those remain launch blockers until frozen by a
separate development contract.
"""

from __future__ import annotations

from typing import Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03RV8AlphaPretrainingBatch,
    M03RV8AlphaPretrainingError,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)


def build_m03r_v8_alpha_pretraining_batch_from_origin_states(
    policy: Top2000M03RV8DevelopmentPolicy,
    origin_states: torch.Tensor,
    sequence: Hold30Sequence,
    local_origin_indices: torch.Tensor,
    *,
    sequence_global_state_start: int,
    split: Literal["training", "inner-validation"],
    split_start_inclusive: int,
    split_stop_exclusive: int,
    fold_index: int,
    source_array_sha256: str,
) -> M03RV8AlphaPretrainingBatch:
    """Build a batch from states aligned one-for-one with local origins.

    ``origin_states`` must remain attached to the encoder graph. TOP2000
    fold/seed training owns one economic path, so path batches greater than one
    are rejected rather than flattened into the date axis.
    """

    if not isinstance(policy, Top2000M03RV8DevelopmentPolicy):
        raise M03RV8AlphaPretrainingError(
            "pretraining requires the generation-qualified v8 policy"
        )
    if (
        not isinstance(origin_states, torch.Tensor)
        or origin_states.ndim != 4
        or origin_states.shape[1] != 1
        or origin_states.shape[2] != sequence.num_assets
        or origin_states.shape[3] != policy.token_dim
        or not origin_states.is_floating_point()
        or not bool(torch.isfinite(origin_states).all())
    ):
        raise M03RV8AlphaPretrainingError(
            "decision states must be finite [position,1,asset,token] for one economic path"
        )
    if (
        not isinstance(local_origin_indices, torch.Tensor)
        or local_origin_indices.ndim != 1
        or local_origin_indices.dtype != torch.long
        or local_origin_indices.device != origin_states.device
        or local_origin_indices.numel() == 0
        or origin_states.shape[0] != local_origin_indices.numel()
        or bool((local_origin_indices[1:] <= local_origin_indices[:-1]).any())
        or int(local_origin_indices[0]) < 0
        or int(local_origin_indices[-1]) >= sequence.asset_returns.shape[0]
    ):
        raise M03RV8AlphaPretrainingError(
            "local origins must be strictly increasing in-range int64"
        )
    if (
        isinstance(sequence_global_state_start, bool)
        or not isinstance(sequence_global_state_start, int)
        or sequence_global_state_start < 0
    ):
        raise M03RV8AlphaPretrainingError(
            "sequence_global_state_start must be a nonnegative integer"
        )
    global_origins = local_origin_indices + sequence_global_state_start
    if (
        int(global_origins[0]) < split_start_inclusive
        or int(global_origins[-1]) >= split_stop_exclusive
    ):
        raise M03RV8AlphaPretrainingError(
            "pretraining origins leave the declared split"
        )

    predicted_mean: list[torch.Tensor] = []
    predicted_log_scale: list[torch.Tensor] = []
    for row_index, local_origin in enumerate(local_origin_indices.tolist()):
        distribution = policy.alpha_pretraining_distribution(
            origin_states[row_index],
            sequence.decision_available[local_origin],
        )
        predicted_mean.append(distribution.predicted_mean.squeeze(0))
        predicted_log_scale.append(distribution.predicted_log_scale.squeeze(0))
    means = torch.stack(predicted_mean)
    log_scales = torch.stack(predicted_log_scale)
    targets = torch.zeros_like(means)
    valid = torch.zeros_like(means, dtype=torch.bool)

    for row_index, local_origin in enumerate(local_origin_indices.tolist()):
        global_origin = local_origin + sequence_global_state_start
        for horizon_index, horizon in enumerate(
            M03R_V8_ALPHA_PRETRAINING.horizons_trading_sessions
        ):
            global_target_stop = global_origin + horizon + 1
            local_first = local_origin + 1
            local_stop = local_first + horizon
            if (
                global_target_stop > split_stop_exclusive
                or local_stop > sequence.asset_returns.shape[0]
            ):
                continue
            stock_log_return = torch.log1p(
                sequence.asset_returns[local_first:local_stop, 0].clamp_min(-0.999999)
            ).sum(dim=0)
            benchmark_log_return = torch.log1p(
                sequence.benchmark_net_returns[local_first:local_stop, 0].clamp_min(
                    -0.999999
                )
            ).sum()
            row_valid = sequence.decision_available[
                local_first : local_stop + 1, 0
            ].all(dim=0)
            row_valid = row_valid.clone()
            row_valid[sequence.initial_ledger.cash_index] = False
            targets[row_index, :, horizon_index] = (
                stock_log_return - benchmark_log_return
            ).detach()
            valid[row_index, :, horizon_index] = row_valid

    batch = M03RV8AlphaPretrainingBatch(
        predicted_mean=means,
        predicted_log_scale=log_scales,
        target_residual_log_return=targets,
        valid=valid,
        origin_indices=global_origins,
        split=split,
        fold_index=fold_index,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        source_array_sha256=source_array_sha256,
    )
    batch.validate()
    return batch


def build_m03r_v8_alpha_pretraining_batch_from_states(
    policy: Top2000M03RV8DevelopmentPolicy,
    decision_states: torch.Tensor,
    sequence: Hold30Sequence,
    local_origin_indices: torch.Tensor,
    *,
    sequence_global_state_start: int,
    split: Literal["training", "inner-validation"],
    split_start_inclusive: int,
    split_stop_exclusive: int,
    fold_index: int,
    source_array_sha256: str,
) -> M03RV8AlphaPretrainingBatch:
    """Build a batch from one complete position-major state tensor."""

    if (
        not isinstance(decision_states, torch.Tensor)
        or decision_states.ndim != 4
        or decision_states.shape[0] != sequence.n_positions
    ):
        raise M03RV8AlphaPretrainingError(
            "decision states must cover every sequence position"
        )
    indexes = local_origin_indices.to(device=decision_states.device)
    return build_m03r_v8_alpha_pretraining_batch_from_origin_states(
        policy,
        decision_states.index_select(0, indexes),
        sequence,
        indexes,
        sequence_global_state_start=sequence_global_state_start,
        split=split,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        fold_index=fold_index,
        source_array_sha256=source_array_sha256,
    )


__all__ = [
    "build_m03r_v8_alpha_pretraining_batch_from_origin_states",
    "build_m03r_v8_alpha_pretraining_batch_from_states",
]
