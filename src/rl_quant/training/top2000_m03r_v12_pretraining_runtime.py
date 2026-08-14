"""Five-horizon factor-qualified batch construction for M03R-v12."""

from __future__ import annotations

from typing import Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_HORIZONS,
    M03RV12PredictiveSetting,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    M03RV9OriginRiskExposures,
)
from rl_quant.training.top2000_m03r_v11_residual_operator import (
    M03RV11ResidualOperator,
    apply_m03r_v11_residual_operator,
    build_m03r_v11_residual_operator,
)
from rl_quant.training.top2000_m03r_v12_objective import M03RV12PredictiveBatch
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)


class M03RV12PretrainingRuntimeError(ValueError):
    """V12 failed its selected-horizon predictive batch boundary."""


def build_m03r_v12_batch_from_origin_states(
    policy: Top2000M03RV12PredictivePolicy,
    setting: M03RV12PredictiveSetting,
    origin_states: torch.Tensor,
    sequence: Hold30Sequence,
    local_origin_indices: torch.Tensor,
    *,
    sequence_global_state_start: int,
    split: Literal["training", "qualification"],
    split_start_inclusive: int,
    split_stop_exclusive: int,
    fold_index: int,
    source_array_sha256: str,
    asset_axis_sha256: str,
    origin_risk_exposures: M03RV9OriginRiskExposures,
) -> M03RV12PredictiveBatch:
    """Build all five heads and targets, including the selected 3-session head."""

    setting.__post_init__()
    if (
        not isinstance(policy, Top2000M03RV12PredictivePolicy)
        or policy.v12_setting != setting
        or not isinstance(origin_states, torch.Tensor)
        or origin_states.ndim != 4
        or origin_states.shape[1] != 1
        or origin_states.shape[2] != sequence.num_assets
        or origin_states.shape[3] != policy.token_dim
        or not origin_states.is_floating_point()
        or not bool(torch.isfinite(origin_states).all())
    ):
        raise M03RV12PretrainingRuntimeError("v12 policy/state geometry drifted")
    if (
        not isinstance(local_origin_indices, torch.Tensor)
        or local_origin_indices.ndim != 1
        or local_origin_indices.dtype != torch.long
        or local_origin_indices.device != origin_states.device
        or local_origin_indices.numel() == 0
        or local_origin_indices.numel() != origin_states.shape[0]
        or bool((local_origin_indices[1:] <= local_origin_indices[:-1]).any())
    ):
        raise M03RV12PretrainingRuntimeError(
            "v12 local origins are not strictly increasing"
        )
    global_origins = local_origin_indices + sequence_global_state_start
    if (
        int(global_origins[0]) < split_start_inclusive
        or int(global_origins[-1]) >= split_stop_exclusive
    ):
        raise M03RV12PretrainingRuntimeError("v12 origins leave the declared split")
    origin_risk_exposures.validate()
    if (
        origin_risk_exposures.asset_axis_sha256 != asset_axis_sha256
        or origin_risk_exposures.exposure_loadings.shape[1] != sequence.num_assets
        or origin_risk_exposures.cash_index != sequence.initial_ledger.cash_index
    ):
        raise M03RV12PretrainingRuntimeError(
            "v12 exposure and sequence asset axes differ"
        )

    means: list[torch.Tensor] = []
    log_scales: list[torch.Tensor] = []
    rank_scores: list[torch.Tensor] = []
    for row_index, local_origin in enumerate(local_origin_indices.tolist()):
        output = policy.predictive_output(
            origin_states[row_index], sequence.decision_available[local_origin]
        )
        means.append(output.economic_distribution.mean_by_horizon.squeeze(0))
        log_scales.append(output.economic_distribution.log_scale_by_horizon.squeeze(0))
        rank_scores.append(output.rank_score_by_horizon.squeeze(0))
    prediction = torch.stack(means)
    predicted_log_scale = torch.stack(log_scales)
    predicted_rank_score = torch.stack(rank_scores)
    targets = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)

    operator_receipts: list[str] = []
    available_counts: list[int] = []
    qualified_counts: list[int] = []
    design_ranks: list[int] = []
    residual_dof: list[int] = []
    residual_operators: list[M03RV11ResidualOperator] = []
    cash_index = sequence.initial_ledger.cash_index
    for row_index, local_origin in enumerate(local_origin_indices.tolist()):
        global_origin = local_origin + sequence_global_state_start
        exposure_row = global_origin - origin_risk_exposures.state_start_index
        if not 0 <= exposure_row < origin_risk_exposures.exposure_loadings.shape[0]:
            raise M03RV12PretrainingRuntimeError(
                "v12 origin has no point-in-time exposure row"
            )
        for horizon_index, horizon in enumerate(M03R_V12_HORIZONS):
            local_first = local_origin + 1
            local_stop = local_first + horizon
            if (
                global_origin + horizon + 1 > split_stop_exclusive
                or local_stop > sequence.asset_returns.shape[0]
            ):
                raise M03RV12PretrainingRuntimeError(
                    "v12 paired batch contains an unsupported horizon"
                )
            stock = torch.log1p(
                sequence.asset_returns[local_first:local_stop, 0].clamp_min(-0.999999)
            ).sum(dim=0)
            benchmark = torch.log1p(
                sequence.benchmark_net_returns[local_first:local_stop, 0].clamp_min(
                    -0.999999
                )
            ).sum()
            available = sequence.decision_available[
                local_first : local_stop + 1, 0
            ].all(dim=0)
            available = available.clone()
            available[cash_index] = False
            operator = build_m03r_v11_residual_operator(
                origin_state_index=global_origin,
                cash_index=cash_index,
                available_mask=available,
                exposure_loadings=origin_risk_exposures.exposure_loadings[exposure_row],
                regression_weights=origin_risk_exposures.regression_weights[
                    exposure_row
                ],
                projector_exposure_names=(
                    origin_risk_exposures.projector_exposure_names
                ),
                projector_exposure_families=(
                    origin_risk_exposures.projector_exposure_families
                ),
                asset_axis_sha256=asset_axis_sha256,
                source_exposure_receipt_sha256=(origin_risk_exposures.receipt_sha256),
            )
            residual = apply_m03r_v11_residual_operator(
                (stock - benchmark).detach(), operator
            )
            targets[row_index, :, horizon_index] = residual.residual
            valid[row_index, :, horizon_index] = residual.qualified_asset_mask
            operator_receipts.append(operator.receipt_sha256)
            residual_operators.append(operator)
            available_counts.append(operator.available_risky_asset_count)
            qualified_counts.append(operator.factor_qualified_risky_asset_count)
            design_ranks.append(operator.effective_design_rank)
            residual_dof.append(operator.weighted_residual_degrees_of_freedom)

    result = M03RV12PredictiveBatch(
        predicted_mean=prediction,
        predicted_log_scale=predicted_log_scale,
        predicted_rank_score=predicted_rank_score,
        target_log_return=targets,
        valid=valid,
        origin_indices=global_origins,
        split=split,
        fold_index=fold_index,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
        exposure_receipt_sha256=origin_risk_exposures.receipt_sha256,
        setting=setting,
        residual_operator_receipt_sha256=tuple(operator_receipts),
        available_risky_asset_count=tuple(available_counts),
        factor_qualified_risky_asset_count=tuple(qualified_counts),
        effective_design_rank=tuple(design_ranks),
        weighted_residual_degrees_of_freedom=tuple(residual_dof),
        residual_operators=tuple(residual_operators),
    )
    result.validate()
    return result


__all__ = [
    "M03RV12PretrainingRuntimeError",
    "build_m03r_v12_batch_from_origin_states",
]
