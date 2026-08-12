"""Deterministic fold updates and untouched-tail diagnostics for M03R-v10."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import (
    M03R_V10_PROTOCOL_SHA256,
    resolve_m03r_v10_setting,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.hold30_top2000_development import (
    Top2000Hold30DevelopmentSequence,
    Top2000VerifiedDevelopmentCache,
    build_top2000_hold30_development_sequence_from_loaded_cache,
)
from rl_quant.training.top2000_m03r_v10_diagnostics import (
    M03RV10FoldDiagnostics,
    build_m03r_v10_fold_diagnostics,
)
from rl_quant.training.top2000_m03r_v10_predictive_worker import (
    M03RV10PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v10_pretraining_step import (
    M03RV10AlphaPretrainingBatch,
    M03RV10AlphaStepReceipt,
    train_m03r_v10_alpha_pretraining_update,
)
from rl_quant.training.top2000_m03r_v10_selection import (
    M03RV10ImportedSleeveTrace,
    M03RV10SleeveFoldEvidence,
    build_m03r_v10_sleeve_fold_evidence,
    run_m03r_v10_simple_sleeve,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
    Top2000M03RV7DecisionStateProvider,
    Top2000M03RV7DevelopmentFold,
    top2000_m03r_v7_decision_inputs,
)
from rl_quant.training.top2000_m03r_v9_fold import (
    M03R_V9_MAX_TARGET_HORIZON,
    M03R_V9_QUALIFICATION_ORIGINS,
    M03RV9FoldGeometry,
    render_m03r_v9_fold_geometry,
)
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9AlphaDistribution,
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_optimizer import (
    M03RV9AlphaOptimizerPartition,
)
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    build_m03r_v9_alpha_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v9_projection import M03RV9DeviceRiskState
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)

M03R_V10_FOLD_GEOMETRY_SCHEMA = "rl-quant.top2000-dev.m03r-v10-fold-geometry-v1"


class M03RV10FoldError(ValueError):
    """The v10 fold, cache, risk, or sampling geometry drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV10FoldGeometry:
    fold_index: int
    training_state_start: int
    optimizer_target_stop_exclusive: int
    qualification_start_inclusive: int
    qualification_origin_stop_exclusive: int
    qualification_target_stop_exclusive: int
    qualification_episode_state_start: int
    qualification_episode_state_stop_exclusive: int
    imported_v9_geometry_sha256: str
    protocol_sha256: str = M03R_V10_PROTOCOL_SHA256
    schema: str = M03R_V10_FOLD_GEOMETRY_SCHEMA

    def validate(self) -> None:
        if (
            not 0 <= self.fold_index < 6
            or self.optimizer_target_stop_exclusive
            != self.qualification_start_inclusive
            or self.qualification_origin_stop_exclusive
            - self.qualification_start_inclusive
            != M03R_V9_QUALIFICATION_ORIGINS
            or self.qualification_target_stop_exclusive
            - self.qualification_origin_stop_exclusive
            != M03R_V9_MAX_TARGET_HORIZON + 1
            or len(self.imported_v9_geometry_sha256) != 64
            or self.protocol_sha256 != M03R_V10_PROTOCOL_SHA256
            or self.schema != M03R_V10_FOLD_GEOMETRY_SCHEMA
        ):
            raise M03RV10FoldError("v10 fold geometry drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def render_m03r_v10_fold_geometry(
    fold: Top2000M03RV7DevelopmentFold,
) -> M03RV10FoldGeometry:
    imported: M03RV9FoldGeometry = render_m03r_v9_fold_geometry(fold)
    result = M03RV10FoldGeometry(
        fold_index=imported.fold_index,
        training_state_start=imported.training_state_start,
        optimizer_target_stop_exclusive=imported.optimizer_target_stop_exclusive,
        qualification_start_inclusive=imported.qualification_start_inclusive,
        qualification_origin_stop_exclusive=(
            imported.qualification_origin_stop_exclusive
        ),
        qualification_target_stop_exclusive=imported.qualification_target_stop_exclusive,
        qualification_episode_state_start=imported.qualification_episode_state_start,
        qualification_episode_state_stop_exclusive=(
            imported.qualification_episode_state_stop_exclusive
        ),
        imported_v9_geometry_sha256=imported.receipt_sha256,
    )
    result.validate()
    return result


def deterministic_m03r_v10_episode_start(
    worker: M03RV10PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    *,
    completed_updates: int,
) -> int:
    worker.validate()
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or not 0 <= completed_updates < worker.predictive_optimizer_updates
    ):
        raise M03RV10FoldError("v10 update cursor is outside 0..63")
    geometry = render_m03r_v10_fold_geometry(fold)
    maximum_start = (
        geometry.qualification_target_stop_exclusive
        - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
    )
    digest = hashlib.sha256(
        (
            f"{worker.receipt_sha256}:{geometry.receipt_sha256}:"
            f"{completed_updates}:paired-two-rank-v10"
        ).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (maximum_start + 1)


def _move_sequence(
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
    return built, _move_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )


def _states(
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
    return bound, provider.replay_origin_states(policy.source_policy, bound, origins)


def _batch(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV10PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    policy: Top2000M03RV9PredictivePolicy,
    *,
    global_origins: torch.Tensor,
    episode_start: int,
    split: str,
    device: torch.device,
) -> M03RV10AlphaPretrainingBatch:
    geometry = render_m03r_v10_fold_geometry(fold)
    built, sequence = _episode(cache, start=episode_start, device=device)
    local_origins = global_origins - episode_start
    bound, encoded = _states(policy, sequence, local_origins)
    base = build_m03r_v9_alpha_batch_from_origin_states(
        policy,
        policy.setting,
        encoded,
        bound,
        local_origins,
        sequence_global_state_start=episode_start,
        split=split,  # type: ignore[arg-type]
        split_start_inclusive=(
            geometry.training_state_start
            if split == "training"
            else geometry.qualification_start_inclusive
        ),
        split_stop_exclusive=(
            geometry.optimizer_target_stop_exclusive
            if split == "training"
            else geometry.qualification_target_stop_exclusive
        ),
        fold_index=fold.fold_index,
        source_array_sha256=built.identity.receipt_sha256,
        asset_axis_sha256=cache.action_hash,
        origin_risk_exposures=risk_source.exposures,
    )
    return M03RV10AlphaPretrainingBatch(
        base,
        resolve_m03r_v10_setting(worker.setting_index),
    )


def run_m03r_v10_pretraining_fold_update(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV10PredictiveWorkerPlan,
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
) -> M03RV10AlphaStepReceipt:
    worker.validate()
    cache.validate_unmodified()
    risk_source.validate()
    geometry = render_m03r_v10_fold_geometry(fold)
    start = deterministic_m03r_v10_episode_start(
        worker,
        fold,
        completed_updates=completed_updates,
    )
    last_origin = min(
        geometry.optimizer_target_stop_exclusive - M03R_V9_MAX_TARGET_HORIZON - 1,
        start + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS - M03R_V9_MAX_TARGET_HORIZON - 1,
    )
    origins = torch.arange(start, last_origin + 1, dtype=torch.long, device=device)
    if origins.numel() % distributed_world_size:
        origins = origins[:-1]
    local_origins = origins[distributed_rank::distributed_world_size]
    if local_origins.numel() == 0:
        raise M03RV10FoldError("v10 training rank shard is empty")
    batch = _batch(
        cache,
        worker,
        fold,
        risk_source,
        policy,
        global_origins=local_origins,
        episode_start=start,
        split="training",
        device=device,
    )
    return train_m03r_v10_alpha_pretraining_update(
        policy,
        batch,
        optimizer,
        partition,
        completed_updates=completed_updates,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )


def evaluate_m03r_v10_untouched_tail_diagnostics(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV10PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    policy: Top2000M03RV9PredictivePolicy,
    *,
    device: torch.device,
) -> M03RV10FoldDiagnostics:
    """Open the 63-origin tail only after its update-64 checkpoint exists."""

    geometry = render_m03r_v10_fold_geometry(fold)
    origins = torch.arange(
        geometry.qualification_start_inclusive,
        geometry.qualification_origin_stop_exclusive,
        dtype=torch.long,
        device=device,
    )
    policy.eval()
    with torch.no_grad():
        batch = _batch(
            cache,
            worker,
            fold,
            risk_source,
            policy,
            global_origins=origins,
            episode_start=geometry.qualification_episode_state_start,
            split="qualification",
            device=device,
        )
        return build_m03r_v10_fold_diagnostics(batch)


def _slice_sleeve_sequence(
    sequence: Hold30Sequence,
    *,
    local_start: int,
    local_stop_exclusive: int,
) -> Hold30Sequence:
    if local_stop_exclusive - local_start != M03R_V9_QUALIFICATION_ORIGINS + 1:
        raise M03RV10FoldError("v10 sleeve chronology must contain 64 states")
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
class M03RV10QualificationFoldResult:
    diagnostics: M03RV10FoldDiagnostics
    sleeve_trace: M03RV10ImportedSleeveTrace
    sleeve_evidence: M03RV10SleeveFoldEvidence
    geometry_sha256: str
    risk_state_sha256: str

    def validate(self) -> None:
        self.diagnostics.validate()
        self.sleeve_trace.validate()
        self.sleeve_evidence.validate()
        if (
            self.diagnostics.fold_index != self.sleeve_trace.imported_trace.fold_index
            or self.sleeve_trace.imported_trace.fold_index
            != self.sleeve_evidence.imported_evidence.fold_index
            or self.diagnostics.setting_id != self.sleeve_trace.setting_id
            or self.sleeve_trace.setting_id != self.sleeve_evidence.setting_id
            or self.sleeve_evidence.v10_trace_receipt_sha256
            != self.sleeve_trace.receipt_sha256
            or len(self.geometry_sha256) != 64
            or len(self.risk_state_sha256) != 64
        ):
            raise M03RV10FoldError("v10 qualification fold result drifted")


def evaluate_m03r_v10_qualification_fold(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV10PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    risk_state: M03RV9DeviceRiskState,
    policy: Top2000M03RV9PredictivePolicy,
    *,
    device: torch.device,
) -> M03RV10QualificationFoldResult:
    """Evaluate one frozen horizon using paired v10 diagnostics and sleeve."""

    worker.validate()
    cache.validate_unmodified()
    risk_source.validate()
    risk_state.validate()
    geometry = render_m03r_v10_fold_geometry(fold)
    expected_origins = tuple(
        range(
            geometry.qualification_start_inclusive,
            geometry.qualification_origin_stop_exclusive,
        )
    )
    if (
        risk_state.origin_state_indices != expected_origins
        or risk_state.asset_axis_sha256 != cache.action_hash
        or risk_state.source_exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
    ):
        raise M03RV10FoldError("v10 qualification risk lineage drifted")
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
        bound, encoded = _states(policy, sequence, local_origins)
        base = build_m03r_v9_alpha_batch_from_origin_states(
            policy,
            policy.setting,
            encoded,
            bound,
            local_origins,
            sequence_global_state_start=geometry.qualification_episode_state_start,
            split="qualification",
            split_start_inclusive=geometry.qualification_start_inclusive,
            split_stop_exclusive=geometry.qualification_target_stop_exclusive,
            fold_index=fold.fold_index,
            source_array_sha256=built.identity.receipt_sha256,
            asset_axis_sha256=cache.action_hash,
            origin_risk_exposures=risk_source.exposures,
        )
        v10_batch = M03RV10AlphaPretrainingBatch(
            base,
            resolve_m03r_v10_setting(worker.setting_index),
        )
        diagnostics = build_m03r_v10_fold_diagnostics(v10_batch)
        horizon = policy.horizon_binding.economic_execution_horizon
        horizon_index = (5, 21, 30, 63).index(horizon)
        distributions = tuple(
            M03RV9AlphaDistribution(
                mean_by_horizon=base.predicted_mean[index].unsqueeze(0),
                log_scale_by_horizon=base.predicted_log_scale[index].unsqueeze(0),
                selected_horizon_sessions=horizon,
                selected_mean=base.predicted_mean[index, :, horizon_index].unsqueeze(0),
                selected_scale=torch.exp(
                    base.predicted_log_scale[index, :, horizon_index]
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
    trace = run_m03r_v10_simple_sleeve(
        sleeve_sequence,
        distributions,
        risk_state,
        policy.horizon_binding,
        policy.alpha_head_identity(),
        setting_index=worker.setting_index,
        fold_index=fold.fold_index,
        state_start_index=geometry.qualification_start_inclusive,
        checkpoint_asset_axis_sha256=cache.action_hash,
        source_receipt_sha256=built.identity.receipt_sha256,
        benchmark_gross_returns=built.benchmark.gross_returns[
            local_start : local_stop - 1
        ].to(device),
        benchmark_one_way_turnover=built.benchmark.total_one_way_turnover[
            local_start : local_stop - 1
        ].to(device),
    )
    evidence = build_m03r_v10_sleeve_fold_evidence(trace, policy.horizon_binding)
    result = M03RV10QualificationFoldResult(
        diagnostics=diagnostics,
        sleeve_trace=trace,
        sleeve_evidence=evidence,
        geometry_sha256=geometry.receipt_sha256,
        risk_state_sha256=risk_state.state_sha256,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V10_FOLD_GEOMETRY_SCHEMA",
    "M03RV10FoldError",
    "M03RV10FoldGeometry",
    "M03RV10QualificationFoldResult",
    "deterministic_m03r_v10_episode_start",
    "evaluate_m03r_v10_qualification_fold",
    "evaluate_m03r_v10_untouched_tail_diagnostics",
    "render_m03r_v10_fold_geometry",
    "run_m03r_v10_pretraining_fold_update",
]
