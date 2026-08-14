"""Build causal post-hoc audit inputs from an exact reloaded M03R-v12 fold."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.evaluation.top2000_m03r_v12_posthoc_inference_audit import (
    M03RV12PosthocAuditInputs,
    build_m03r_v12_posthoc_audit_inputs,
    build_m03r_v12_posthoc_causal_action_mask,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_HORIZONS,
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
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v11_residual_operator import (
    apply_m03r_v11_residual_operator,
    build_m03r_v11_residual_operator,
)
from rl_quant.training.top2000_m03r_v12_checkpoint import M03RV12LoadedCheckpoint
from rl_quant.training.top2000_m03r_v12_policy import Top2000M03RV12PredictivePolicy
from rl_quant.training.top2000_m03r_v12_predictive_worker import (
    M03RV12PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v12_pretraining_runtime import (
    build_m03r_v12_batch_from_origin_states,
)


class M03RV12PosthocAuditFoldError(ValueError):
    """The exact V12 fold cannot be converted into causal audit inputs."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


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


def build_m03r_v12_posthoc_audit_fold_inputs(
    cache: Top2000VerifiedDevelopmentCache,
    worker: M03RV12PredictiveWorkerPlan,
    fold: Top2000M03RV7DevelopmentFold,
    risk_source: M03RV9MaterializedRiskSource,
    policy: Top2000M03RV12PredictivePolicy,
    loaded: M03RV12LoadedCheckpoint,
    *,
    device: torch.device,
) -> M03RV12PosthocAuditInputs:
    """Recompute exact v12 outputs, then separate label and action masks."""

    worker.validate()
    cache.validate_unmodified()
    risk_source.validate()
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
        or loaded.completed_updates != 64
        or loaded.selected_horizon_sessions != 3
        or loaded.episode_schedule_sha256 != worker.panel_episode_schedule_sha256
        or loaded.asset_axis_sha256 != cache.action_hash
        or worker.cache_sha256 != cache.cache_sha256
        or state_dict_sha256(policy.state_dict()) != loaded.model_state_sha256
        or policy.selected_horizon_sessions != loaded.selected_horizon_sessions
        or worker.selected_horizon_sessions != loaded.selected_horizon_sessions
        or risk_source.exposures.asset_axis_sha256 != cache.action_hash
        or tuple(expected_origins)[-1] + 1 >= geometry.qualification_target_stop_exclusive
    ):
        raise M03RV12PosthocAuditFoldError(
            "v12 audit worker, checkpoint, cache, risk, or horizon drifted"
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
    global_origins = torch.tensor(expected_origins, dtype=torch.long, device=device)
    local_origins = global_origins - geometry.qualification_episode_state_start
    if bool((local_origins < 251).any()):
        raise M03RV12PosthocAuditFoldError(
            "v12 audit qualification origin lacks the frozen full context"
        )
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
        batch = build_m03r_v12_batch_from_origin_states(
            policy,
            resolve_m03r_v12_setting(worker.setting_index),
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

    horizon_index = M03R_V12_HORIZONS.index(3)
    raw_mean = batch.predicted_mean[:, :, horizon_index]
    raw_rank = batch.predicted_rank_score[:, :, horizon_index]
    selected_scale = torch.exp(batch.predicted_log_scale[:, :, horizon_index])
    causal_masks: list[torch.Tensor] = []
    feasible_mean: list[torch.Tensor] = []
    feasible_rank: list[torch.Tensor] = []
    causal_operator_receipts: list[str] = []
    for date, (global_origin, local_origin) in enumerate(
        zip(global_origins.tolist(), local_origins.tolist(), strict=True)
    ):
        exposure_row = global_origin - risk_source.exposures.state_start_index
        if not 0 <= exposure_row < risk_source.exposures.regression_weights.shape[0]:
            raise M03RV12PosthocAuditFoldError(
                "v12 audit origin lacks point-in-time risk evidence"
            )
        origin_mask = build_m03r_v12_posthoc_causal_action_mask(
            sequence.decision_available[local_origin],
            risk_source.exposures.regression_weights[exposure_row].unsqueeze(0),
        ).squeeze(0)
        operator = build_m03r_v11_residual_operator(
            origin_state_index=global_origin,
            cash_index=sequence.initial_ledger.cash_index,
            available_mask=origin_mask,
            exposure_loadings=risk_source.exposures.exposure_loadings[exposure_row],
            regression_weights=risk_source.exposures.regression_weights[exposure_row],
            projector_exposure_names=risk_source.exposures.projector_exposure_names,
            projector_exposure_families=(
                risk_source.exposures.projector_exposure_families
            ),
            asset_axis_sha256=cache.action_hash,
            source_exposure_receipt_sha256=risk_source.exposures.receipt_sha256,
        )
        causal_masks.append(operator.qualified_asset_mask)
        feasible_mean.append(
            apply_m03r_v11_residual_operator(raw_mean[date], operator).residual
        )
        feasible_rank.append(
            apply_m03r_v11_residual_operator(raw_rank[date], operator).residual
        )
        causal_operator_receipts.append(operator.receipt_sha256)

    causal_action_mask = torch.stack(causal_masks).to(device=device)
    fill_indices = local_origins + 1
    fill_execution_mask = (
        sequence.fill_membership.index_select(0, fill_indices)[:, 0]
        & sequence.fill_availability.index_select(0, fill_indices)[:, 0]
    )
    fill_execution_mask = fill_execution_mask.clone()
    fill_execution_mask[:, sequence.initial_ledger.cash_index] = False
    post_fill_returns = sequence.asset_returns.index_select(0, fill_indices)[:, 0]
    benchmark_weights = sequence.benchmark_weights.index_select(0, fill_indices)[:, 0]
    asset_caps = sequence.risk_asset_caps.index_select(0, fill_indices)[:, 0]
    label_valid = batch.valid[:, :, horizon_index] & causal_action_mask
    action_mask_source_sha256 = _sha256(
        {
            "decision_available_sha256": _tensor_sha256(
                sequence.decision_available.index_select(0, local_origins)
            ),
            "origin_regression_weights_sha256": _tensor_sha256(
                risk_source.exposures.regression_weights.index_select(
                    0,
                    (
                        global_origins - risk_source.exposures.state_start_index
                    ).to(device="cpu"),
                )
            ),
            "causal_operator_receipt_sha256": tuple(causal_operator_receipts),
            "rule": "origin-only-no-future-label-availability-v1",
        }
    )
    post_fill_source_sha256 = _sha256(
        {
            "post_fill_returns_sha256": _tensor_sha256(post_fill_returns),
            "fill_execution_mask_sha256": _tensor_sha256(fill_execution_mask),
            "benchmark_weights_sha256": _tensor_sha256(benchmark_weights),
            "asset_caps_sha256": _tensor_sha256(asset_caps),
            "rule": "decision-t-action-earns-return-t-plus-1-v1",
        }
    )
    return build_m03r_v12_posthoc_audit_inputs(
        setting_index=worker.setting_index,
        fold_index=fold.fold_index,
        checkpoint_file_sha256=loaded.checkpoint_file_sha256,
        checkpoint_model_state_sha256=loaded.model_state_sha256,
        source_array_sha256=batch.source_array_sha256,
        asset_axis_sha256=cache.action_hash,
        action_mask_source_sha256=action_mask_source_sha256,
        post_fill_return_source_sha256=post_fill_source_sha256,
        origin_indices=global_origins,
        raw_economic_mean=raw_mean,
        raw_rank_score=raw_rank,
        economic_mean=torch.stack(feasible_mean),
        rank_score=torch.stack(feasible_rank),
        selected_scale=selected_scale,
        target_log_return=batch.target_log_return[:, :, horizon_index],
        label_valid=label_valid,
        causal_action_mask=causal_action_mask,
        fill_execution_mask=fill_execution_mask,
        post_fill_asset_returns=post_fill_returns,
        benchmark_target_weights=benchmark_weights,
        asset_weight_caps=asset_caps,
    )


__all__ = [
    "M03RV12PosthocAuditFoldError",
    "build_m03r_v12_posthoc_audit_fold_inputs",
]
