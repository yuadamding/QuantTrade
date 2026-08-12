"""Cache-to-update and update-64 qualification for one M03R-v9 fold.

The optimizer never observes the final 63 training-tail origins.  At update
64, one horizon-bound evaluation policy consumes that untouched chronology,
then the same mean/scale tensors drive the deterministic simple sleeve.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_HORIZONS,
    M03R_V9_PREDICTIVE_SPEC,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.hold30_top2000_development import (
    Top2000Hold30DevelopmentSequence,
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    Top2000M03RV7DecisionStateProvider,
    Top2000M03RV7DevelopmentFold,
    render_top2000_m03r_v7_development_folds,
    top2000_m03r_v7_decision_inputs,
)
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaFoldEvidence,
    build_m03r_v9_alpha_fold_evidence,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9AlphaDistribution,
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_predictive_worker import (
    M03RV9PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    M03RV9AlphaOptimizerPartition,
)
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    build_m03r_v9_alpha_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    M03RV9AlphaStepReceipt,
    train_m03r_v9_alpha_pretraining_update,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    M03RV9DeviceRiskState,
    M03RV9ProjectorManifest,
    M03RV9ProjectorRiskBinding,
    build_m03r_v9_device_risk_state,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v9_runtime import (
    M03RV9SimpleSleeveTrace,
    run_m03r_v9_simple_sleeve,
)
from rl_quant.training.top2000_m03r_v9_selection import (
    M03RV9SimpleSleeveFoldEvidence,
    build_m03r_v9_simple_sleeve_fold_evidence,
)

M03R_V9_FOLD_GEOMETRY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-predictive-fold-geometry-v1"
)
M03R_V9_QUALIFICATION_ORIGINS = 63
M03R_V9_MAX_TARGET_HORIZON = max(M03R_V9_HORIZONS)
M03R_V9_TARGET_SUPPORT_STATES = M03R_V9_MAX_TARGET_HORIZON + 1


class M03RV9FoldError(ValueError):
    """The v9 cache slice, split, risk surface, or rank shard drifted."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV9FoldGeometry:
    fold_index: int
    training_state_start: int
    optimizer_target_stop_exclusive: int
    qualification_start_inclusive: int
    qualification_origin_stop_exclusive: int
    qualification_target_stop_exclusive: int
    qualification_episode_state_start: int
    qualification_episode_state_stop_exclusive: int
    schema: str = M03R_V9_FOLD_GEOMETRY_SCHEMA

    def validate(self) -> None:
        folds = render_top2000_m03r_v7_development_folds(
            TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        )
        if not 0 <= self.fold_index < len(folds):
            raise M03RV9FoldError("fold index leaves the six-fold geometry")
        fold = folds[self.fold_index]
        if (
            self.schema != M03R_V9_FOLD_GEOMETRY_SCHEMA
            or self.training_state_start != fold.training_state_start
            or self.optimizer_target_stop_exclusive
            != fold.training_state_stop_exclusive
            - M03R_V9_QUALIFICATION_ORIGINS
            - M03R_V9_TARGET_SUPPORT_STATES
            or self.qualification_start_inclusive
            != self.optimizer_target_stop_exclusive
            or self.qualification_origin_stop_exclusive
            != fold.training_state_stop_exclusive - M03R_V9_TARGET_SUPPORT_STATES
            or self.qualification_target_stop_exclusive
            != fold.training_state_stop_exclusive
            or self.qualification_origin_stop_exclusive
            - self.qualification_start_inclusive
            != M03R_V9_QUALIFICATION_ORIGINS
            or self.qualification_episode_state_start
            != fold.training_state_stop_exclusive
            - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
            or self.qualification_episode_state_stop_exclusive
            != fold.training_state_stop_exclusive
        ):
            raise M03RV9FoldError("v9 predictive fold geometry drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(asdict(self))


def render_m03r_v9_fold_geometry(
    fold: Top2000M03RV7DevelopmentFold,
) -> M03RV9FoldGeometry:
    if not isinstance(fold, Top2000M03RV7DevelopmentFold):
        raise M03RV9FoldError("v9 geometry requires the reviewed chronological fold")
    result = M03RV9FoldGeometry(
        fold_index=fold.fold_index,
        training_state_start=fold.training_state_start,
        optimizer_target_stop_exclusive=(
            fold.training_state_stop_exclusive
            - M03R_V9_QUALIFICATION_ORIGINS
            - M03R_V9_TARGET_SUPPORT_STATES
        ),
        qualification_start_inclusive=(
            fold.training_state_stop_exclusive
            - M03R_V9_QUALIFICATION_ORIGINS
            - M03R_V9_TARGET_SUPPORT_STATES
        ),
        qualification_origin_stop_exclusive=(
            fold.training_state_stop_exclusive - M03R_V9_TARGET_SUPPORT_STATES
        ),
        qualification_target_stop_exclusive=fold.training_state_stop_exclusive,
        qualification_episode_state_start=(
            fold.training_state_stop_exclusive - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
        ),
        qualification_episode_state_stop_exclusive=(fold.training_state_stop_exclusive),
    )
    result.validate()
    return result


def deterministic_m03r_v9_episode_start(
    worker: M03RV9PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    *,
    completed_updates: int,
) -> int:
    worker.validate()
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or not 0
        <= completed_updates
        < M03R_V9_PREDICTIVE_SPEC.maximum_optimizer_updates
    ):
        raise M03RV9FoldError("v9 update cursor is outside 0..63")
    geometry = render_m03r_v9_fold_geometry(fold)
    maximum_start = (
        geometry.qualification_target_stop_exclusive
        - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
    )
    digest = hashlib.sha256(
        (
            f"{worker.receipt_sha256}:{geometry.receipt_sha256}:"
            f"{completed_updates}:paired-two-rank-v1"
        ).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (maximum_start + 1)


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
        # The reviewed cache adapter axis_id binds the entire slice identity.
        # V9 separately binds that receipt and uses the exact action-axis hash
        # at the projector boundary, where same-length permutations must fail.
        axis_id=asset_axis_sha256,
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
    return built, _move_and_bind_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )


def _states_for_origins(
    policy: Top2000M03RV9PredictivePolicy,
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
    # Only the reviewed raw encoder is used to construct origin states; the
    # v9 mean/scale heads consume those states afterward.
    return bound, provider.replay_origin_states(policy.source_policy, bound, origins)


def run_m03r_v9_pretraining_fold_update(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV9PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    policy: Top2000M03RV9PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV9AlphaOptimizerPartition,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
    device: torch.device,
) -> M03RV9AlphaStepReceipt:
    worker.validate()
    cache.validate_unmodified()
    risk_source.validate()
    if (
        worker.setting_index != policy.setting.setting_index
        or fold.fold_index not in range(worker.fold_count)
        or distributed_world_size != worker.expected_world_size
        or risk_source.cache_sha256 != cache.cache_sha256
        or risk_source.action_hash != cache.action_hash
    ):
        raise M03RV9FoldError("worker, cache, risk, policy, or rank geometry drifted")
    geometry = render_m03r_v9_fold_geometry(fold)
    start = deterministic_m03r_v9_episode_start(
        worker,
        fold,
        completed_updates=completed_updates,
    )
    built, sequence = _episode(cache, start=start, device=device)
    last_global_origin = min(
        geometry.optimizer_target_stop_exclusive - M03R_V9_MAX_TARGET_HORIZON - 1,
        start + sequence.asset_returns.shape[0] - M03R_V9_MAX_TARGET_HORIZON,
    )
    global_origins = torch.arange(
        start,
        last_global_origin + 1,
        dtype=torch.long,
        device=device,
    )
    if global_origins.numel() % distributed_world_size:
        global_origins = global_origins[:-1]
    if global_origins.numel() < distributed_world_size:
        raise M03RV9FoldError("training episode has no equal nonempty rank shard")
    local_global = global_origins[distributed_rank::distributed_world_size]
    local_origins = local_global - start
    bound, states = _states_for_origins(policy, sequence, local_origins)
    batch = build_m03r_v9_alpha_batch_from_origin_states(
        policy,
        policy.setting,
        states,
        bound,
        local_origins,
        sequence_global_state_start=start,
        split="training",
        split_start_inclusive=geometry.training_state_start,
        split_stop_exclusive=geometry.optimizer_target_stop_exclusive,
        fold_index=fold.fold_index,
        source_array_sha256=built.identity.receipt_sha256,
        asset_axis_sha256=cache.action_hash,
        origin_risk_exposures=(
            risk_source.exposures
            if policy.setting.target_mode == "factor-residual"
            else None
        ),
    )
    return train_m03r_v9_alpha_pretraining_update(
        policy,
        batch,
        optimizer,
        partition,
        completed_updates=completed_updates,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )


def _daily_log_returns(
    cache: Top2000VerifiedDevelopmentCache,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    daily = cache.daily_ohlcv.to(dtype=torch.float64)
    close = daily[..., 3]
    valid = (
        cache.availability[:-1]
        & cache.availability[1:]
        & (close[:-1] > 0.0)
        & (close[1:] > 0.0)
    )
    returns = torch.zeros_like(close)
    returns[1:] = torch.where(
        valid,
        torch.log(close[1:] / close[:-1]),
        torch.zeros_like(close[1:]),
    )
    available = torch.zeros_like(cache.availability)
    available[1:] = valid
    returns[:, 0] = 0.0
    available[:, 0] = False
    receipt = _canonical_sha256(
        {
            "schema": "rl-quant.top2000-dev.m03r-v9-past-log-returns-v1",
            "cache_sha256": cache.cache_sha256,
            "asset_axis_sha256": cache.action_hash,
            "return_sha256": _tensor_sha256(returns),
            "availability_sha256": _tensor_sha256(available),
            "current-origin-return-excluded": True,
        }
    )
    return returns, available, receipt


def _slice_sleeve_sequence(
    sequence: Hold30Sequence,
    *,
    local_start: int,
    local_stop_exclusive: int,
) -> Hold30Sequence:
    if local_stop_exclusive - local_start != M03R_V9_QUALIFICATION_ORIGINS + 1:
        raise M03RV9FoldError("sleeve chronology must contain 64 states")
    initial_weights = sequence.benchmark_weights[local_start]
    return Hold30Sequence(
        decision_state=sequence.decision_state[local_start:local_stop_exclusive],
        asset_returns=sequence.asset_returns[local_start : local_stop_exclusive - 1],
        decision_available=sequence.decision_available[
            local_start:local_stop_exclusive
        ],
        fill_membership=sequence.fill_membership[local_start:local_stop_exclusive],
        fill_availability=sequence.fill_availability[local_start:local_stop_exclusive],
        benchmark_weights=sequence.benchmark_weights[local_start:local_stop_exclusive],
        risk_asset_caps=sequence.risk_asset_caps[local_start:local_stop_exclusive],
        risk_gross_max=sequence.risk_gross_max[local_start:local_stop_exclusive],
        benchmark_net_returns=sequence.benchmark_net_returns[
            local_start : local_stop_exclusive - 1
        ],
        initial_ledger=CohortLedger.from_staggered_endowment(
            initial_weights,
            cash_index=sequence.initial_ledger.cash_index,
            youngest_age=0,
            oldest_age=29,
            track_initial_units=False,
        ),
        cost_rate=sequence.cost_rate,
        initial_equity=initial_weights.new_ones((1,)),
        track_entry_units=torch.ones(
            M03R_V9_QUALIFICATION_ORIGINS,
            dtype=torch.bool,
            device=initial_weights.device,
        ),
        axis_id=sequence.axis_id,
    )


@dataclass(frozen=True, slots=True)
class M03RV9QualificationFoldResult:
    alpha_evidence: M03RV9AlphaFoldEvidence
    sleeve_evidence: M03RV9SimpleSleeveFoldEvidence
    sleeve_trace: M03RV9SimpleSleeveTrace
    geometry_sha256: str
    model_state_source_receipt_sha256: str

    def validate(self) -> None:
        self.alpha_evidence.__post_init__()
        self.sleeve_evidence.__post_init__()
        self.sleeve_trace.validate()
        if (
            self.alpha_evidence.fold_index != self.sleeve_evidence.fold_index
            or self.sleeve_evidence.fold_index != self.sleeve_trace.fold_index
            or self.sleeve_evidence.source_receipt_sha256
            != self.sleeve_trace.trace_sha256
            or len(self.geometry_sha256) != 64
            or len(self.model_state_source_receipt_sha256) != 64
        ):
            raise M03RV9FoldError("qualification fold result drifted")


def build_m03r_v9_qualification_risk_state(
    cache: Top2000VerifiedDevelopmentCache,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    risk_binding: M03RV9ProjectorRiskBinding,
    projector: M03RV9ProjectorManifest,
    *,
    device: torch.device,
) -> M03RV9DeviceRiskState:
    """Qualify and transfer the common 21/30-session fold risk surface once."""

    cache.validate_unmodified()
    risk_source.validate()
    geometry = render_m03r_v9_fold_geometry(fold)
    daily_returns, return_available, daily_receipt = _daily_log_returns(cache)
    origin_indices = tuple(
        range(
            geometry.qualification_start_inclusive,
            geometry.qualification_origin_stop_exclusive,
        )
    )
    return build_m03r_v9_device_risk_state(
        risk_source,
        risk_binding,
        projector,
        daily_log_returns=daily_returns,
        return_available=return_available,
        daily_returns_receipt_sha256=daily_receipt,
        sequence_asset_axis_sha256=cache.action_hash,
        checkpoint_asset_axis_sha256=cache.action_hash,
        origin_state_indices=origin_indices,
        device=device,
    )


def evaluate_m03r_v9_qualification_fold(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV9PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    risk_binding: M03RV9ProjectorRiskBinding,
    projector: M03RV9ProjectorManifest,
    risk_state: M03RV9DeviceRiskState,
    policy: Top2000M03RV9PredictivePolicy,
    *,
    device: torch.device,
) -> M03RV9QualificationFoldResult:
    """Evaluate the untouched tail once and trade that exact distribution."""

    worker.validate()
    cache.validate_unmodified()
    risk_source.validate()
    geometry = render_m03r_v9_fold_geometry(fold)
    expected_origins = tuple(
        range(
            geometry.qualification_start_inclusive,
            geometry.qualification_origin_stop_exclusive,
        )
    )
    risk_state.validate()
    risk_state.require_fast_identity(
        sequence_asset_axis_sha256=cache.action_hash,
        checkpoint_asset_axis_sha256=cache.action_hash,
        expected_manifest_sha256=projector.manifest_sha256,
    )
    if (
        risk_state.origin_state_indices != expected_origins
        or risk_state.source_binding_sha256 != risk_binding.binding_sha256
        or risk_state.source_exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
    ):
        raise M03RV9FoldError("qualification risk-state lineage drifted")
    built, sequence = _episode(
        cache,
        start=geometry.qualification_episode_state_start,
        device=device,
    )
    global_origins = torch.arange(
        geometry.qualification_start_inclusive,
        geometry.qualification_origin_stop_exclusive,
        dtype=torch.long,
        device=device,
    )
    local_origins = global_origins - geometry.qualification_episode_state_start
    policy.eval()
    with torch.no_grad():
        bound, states = _states_for_origins(policy, sequence, local_origins)
        batch = build_m03r_v9_alpha_batch_from_origin_states(
            policy,
            policy.setting,
            states,
            bound,
            local_origins,
            sequence_global_state_start=geometry.qualification_episode_state_start,
            split="qualification",
            split_start_inclusive=geometry.qualification_start_inclusive,
            split_stop_exclusive=geometry.qualification_target_stop_exclusive,
            fold_index=fold.fold_index,
            source_array_sha256=built.identity.receipt_sha256,
            asset_axis_sha256=cache.action_hash,
            origin_risk_exposures=(
                risk_source.exposures
                if policy.setting.target_mode == "factor-residual"
                else None
            ),
        )
        alpha_evidence = build_m03r_v9_alpha_fold_evidence(batch)
        distributions = tuple(
            M03RV9AlphaDistribution(
                mean_by_horizon=batch.predicted_mean[index].unsqueeze(0),
                log_scale_by_horizon=batch.predicted_log_scale[index].unsqueeze(0),
                selected_horizon_sessions=(
                    policy.horizon_binding.economic_execution_horizon
                ),
                selected_mean=batch.predicted_mean[
                    index,
                    :,
                    M03R_V9_HORIZONS.index(
                        policy.horizon_binding.economic_execution_horizon
                    ),
                ].unsqueeze(0),
                selected_scale=torch.exp(
                    batch.predicted_log_scale[
                        index,
                        :,
                        M03R_V9_HORIZONS.index(
                            policy.horizon_binding.economic_execution_horizon
                        ),
                    ]
                ).unsqueeze(0),
            )
            for index in range(M03R_V9_QUALIFICATION_ORIGINS)
        )
        for distribution in distributions:
            distribution.validate()

    local_start = geometry.qualification_start_inclusive - (
        geometry.qualification_episode_state_start
    )
    local_stop = local_start + M03R_V9_QUALIFICATION_ORIGINS + 1
    sleeve_sequence = _slice_sleeve_sequence(
        sequence,
        local_start=local_start,
        local_stop_exclusive=local_stop,
    )
    benchmark_gross = built.benchmark.gross_returns[local_start : local_stop - 1].to(
        device
    )
    benchmark_turnover = built.benchmark.total_one_way_turnover[
        local_start : local_stop - 1
    ].to(device)
    trace = run_m03r_v9_simple_sleeve(
        sleeve_sequence,
        distributions,
        risk_state,
        policy.horizon_binding,
        policy.alpha_head_identity(),
        setting_id=policy.setting.setting_id,
        fold_index=fold.fold_index,
        state_start_index=geometry.qualification_start_inclusive,
        checkpoint_asset_axis_sha256=cache.action_hash,
        source_receipt_sha256=built.identity.receipt_sha256,
        benchmark_gross_returns=benchmark_gross,
        benchmark_one_way_turnover=benchmark_turnover,
    )
    sleeve_evidence = build_m03r_v9_simple_sleeve_fold_evidence(
        setting_id=policy.setting.setting_id,
        fold_index=fold.fold_index,
        horizon_binding=policy.horizon_binding,
        policy_gross_returns=trace.policy_gross_returns,
        benchmark_gross_returns=trace.benchmark_gross_returns,
        policy_one_way_turnover=trace.policy_one_way_turnover,
        benchmark_one_way_turnover=trace.benchmark_one_way_turnover,
        requested_weight_trace=trace.requested_weight_trace,
        projected_weight_trace=trace.projected_weight_trace,
        signal_null_retention=trace.signal_null_retention,
        requested_to_executed_retention=(trace.requested_to_executed_retention),
        source_receipt_sha256=trace.trace_sha256,
    )
    result = M03RV9QualificationFoldResult(
        alpha_evidence=alpha_evidence,
        sleeve_evidence=sleeve_evidence,
        sleeve_trace=trace,
        geometry_sha256=geometry.receipt_sha256,
        model_state_source_receipt_sha256=built.identity.receipt_sha256,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V9_FOLD_GEOMETRY_SCHEMA",
    "M03R_V9_MAX_TARGET_HORIZON",
    "M03R_V9_QUALIFICATION_ORIGINS",
    "M03R_V9_TARGET_SUPPORT_STATES",
    "M03RV9FoldError",
    "M03RV9FoldGeometry",
    "M03RV9QualificationFoldResult",
    "build_m03r_v9_qualification_risk_state",
    "deterministic_m03r_v9_episode_start",
    "evaluate_m03r_v9_qualification_fold",
    "render_m03r_v9_fold_geometry",
    "run_m03r_v9_pretraining_fold_update",
]
