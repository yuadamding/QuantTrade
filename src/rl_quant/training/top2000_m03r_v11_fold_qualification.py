"""Untouched-tail fold qualification for an exact reloaded M03R-v11 checkpoint."""

from __future__ import annotations

from dataclasses import replace

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_HORIZONS,
    resolve_m03r_v11_setting,
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
from rl_quant.training.top2000_m03r_v9_fold import M03R_V9_QUALIFICATION_ORIGINS
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9AlphaDistribution,
    Top2000M03RV9PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256
from rl_quant.training.top2000_m03r_v9_projection import M03RV9DeviceRiskState
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v11_checkpoint import M03RV11LoadedCheckpoint
from rl_quant.training.top2000_m03r_v11_predictive_worker import (
    M03RV11PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v11_pretraining_runtime import (
    build_m03r_v11_alpha_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v11_qualification_runtime import (
    M03RV11FoldQualificationLineage,
    build_m03r_v11_fold_qualification_lineage,
)
from rl_quant.training.top2000_m03r_v11_runtime import run_m03r_v11_simple_sleeve


class M03RV11FoldQualificationError(ValueError):
    """The v11 untouched-tail fold evaluation lineage drifted."""


def _move_sequence(
    sequence: Hold30Sequence,
    *,
    device: torch.device,
    asset_axis_sha256: str,
) -> Hold30Sequence:
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
        initial_ledger=CohortLedger(
            economic_value=sequence.initial_ledger.economic_value.to(device),
            retention_units=sequence.initial_ledger.retention_units.to(device),
            cash_index=sequence.initial_ledger.cash_index,
        ),
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


def _slice_sleeve_sequence(
    sequence: Hold30Sequence,
    *,
    local_start: int,
    local_stop_exclusive: int,
) -> Hold30Sequence:
    if local_stop_exclusive - local_start != M03R_V9_QUALIFICATION_ORIGINS + 1:
        raise M03RV11FoldQualificationError(
            "v11 sleeve chronology must contain exactly 64 states"
        )
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


def evaluate_m03r_v11_loaded_qualification_fold(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV11PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    risk_state: M03RV9DeviceRiskState,
    policy: Top2000M03RV9PredictivePolicy,
    loaded: M03RV11LoadedCheckpoint,
    *,
    device: torch.device,
) -> M03RV11FoldQualificationLineage:
    """Evaluate exactly one update-64 checkpoint on the untouched fold tail."""

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
        loaded.setting_index != worker.setting_index
        or loaded.setting_id != worker.setting_id
        or loaded.fold_index != fold.fold_index
        or loaded.episode_schedule_sha256 != worker.panel_episode_schedule_sha256
        or loaded.asset_axis_sha256 != cache.action_hash
        or worker.cache_sha256 != cache.cache_sha256
        or state_dict_sha256(policy.state_dict()) != loaded.model_state_sha256
        or policy.horizon_binding.checkpoint_selection_horizon
        != loaded.selected_horizon_sessions
        or policy.horizon_binding.qualification_horizon
        != loaded.selected_horizon_sessions
        or policy.horizon_binding.economic_execution_horizon
        != loaded.selected_horizon_sessions
        or risk_state.asset_axis_sha256 != cache.action_hash
        or risk_state.source_exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
        or risk_state.origin_state_indices != expected_origins
    ):
        raise M03RV11FoldQualificationError(
            "v11 worker, checkpoint, policy, cache, risk, or horizon drifted"
        )
    risk_state.require_fast_identity(
        sequence_asset_axis_sha256=cache.action_hash,
        checkpoint_asset_axis_sha256=loaded.asset_axis_sha256,
        expected_manifest_sha256=risk_state.manifest_sha256,
    )

    built = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=geometry.qualification_episode_state_start,
        state_stop_index_exclusive=(
            geometry.qualification_episode_state_start
            + TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
        ),
        max_state_rows=TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        output_device="cpu",
    )
    sequence = _move_sequence(
        built.sequence,
        device=device,
        asset_axis_sha256=cache.action_hash,
    )
    global_origins = torch.arange(
        geometry.qualification_start_inclusive,
        geometry.qualification_origin_stop_exclusive,
        dtype=torch.long,
        device=device,
    )
    local_origins = global_origins - geometry.qualification_episode_state_start
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
    policy.eval()
    with torch.no_grad():
        origin_states = provider.replay_origin_states(
            policy.source_policy,
            bound,
            local_origins,
        )
        batch = build_m03r_v11_alpha_batch_from_origin_states(
            policy,
            resolve_m03r_v11_setting(worker.setting_index),
            origin_states,
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
    if batch.residual_operators is None:
        raise M03RV11FoldQualificationError(
            "v11 qualification did not retain executable residual operators"
        )
    horizon_index = M03R_V11_HORIZONS.index(loaded.selected_horizon_sessions)
    distributions = tuple(
        M03RV9AlphaDistribution(
            mean_by_horizon=batch.corrected_batch.predicted_mean[index].unsqueeze(0),
            log_scale_by_horizon=(
                batch.corrected_batch.predicted_log_scale[index].unsqueeze(0)
            ),
            selected_horizon_sessions=loaded.selected_horizon_sessions,
            selected_mean=batch.corrected_batch.predicted_mean[
                index, :, horizon_index
            ].unsqueeze(0),
            selected_scale=torch.exp(
                batch.corrected_batch.predicted_log_scale[index, :, horizon_index]
            ).unsqueeze(0),
        )
        for index in range(M03R_V9_QUALIFICATION_ORIGINS)
    )
    selected_operators = tuple(
        batch.residual_operators[index * len(M03R_V11_HORIZONS) + horizon_index]
        for index in range(M03R_V9_QUALIFICATION_ORIGINS)
    )
    local_start = geometry.qualification_start_inclusive - (
        geometry.qualification_episode_state_start
    )
    local_stop = local_start + M03R_V9_QUALIFICATION_ORIGINS + 1
    sleeve_sequence = _slice_sleeve_sequence(
        sequence,
        local_start=local_start,
        local_stop_exclusive=local_stop,
    )
    trace = run_m03r_v11_simple_sleeve(
        sleeve_sequence,
        distributions,
        selected_operators,
        risk_state,
        setting_index=worker.setting_index,
        fold_index=fold.fold_index,
        selected_horizon_sessions=loaded.selected_horizon_sessions,
        state_start_index=geometry.qualification_start_inclusive,
        checkpoint_file_sha256=loaded.checkpoint_file_sha256,
        checkpoint_model_state_sha256=loaded.model_state_sha256,
        checkpoint_asset_axis_sha256=loaded.asset_axis_sha256,
        source_receipt_sha256=built.identity.receipt_sha256,
        benchmark_gross_returns=(
            built.benchmark.gross_returns[local_start : local_stop - 1].to(device)
        ),
        benchmark_one_way_turnover=(
            built.benchmark.total_one_way_turnover[local_start : local_stop - 1].to(
                device
            )
        ),
    )
    return build_m03r_v11_fold_qualification_lineage(loaded, batch, trace)


__all__ = [
    "M03RV11FoldQualificationError",
    "evaluate_m03r_v11_loaded_qualification_fold",
]
