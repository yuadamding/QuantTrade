"""Factor-qualified target construction for M03R-v11."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_HORIZONS,
    M03R_V11_PROTOCOL_SHA256,
    M03RV11PredictiveSetting,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v9_policy import Top2000M03RV9PredictivePolicy
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    M03RV9OriginRiskExposures,
    build_m03r_v9_alpha_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v11_residual_operator import (
    M03RV11ResidualOperator,
    apply_m03r_v11_residual_operator,
    build_m03r_v11_residual_operator,
)

M03R_V11_ALPHA_BATCH_SCHEMA = "rl-quant.top2000-dev.m03r-v11-alpha-batch-v1"
M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID = "V9-P0-factor-residual-ranked"


class M03RV11PretrainingRuntimeError(ValueError):
    """The v11 qualified target batch drifted."""


def _sha256(value: Any) -> str:
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
class M03RV11AlphaPretrainingBatch:
    corrected_batch: M03RV9AlphaPretrainingBatch
    setting: M03RV11PredictiveSetting
    residual_operator_receipt_sha256: tuple[str, ...]
    available_risky_asset_count: tuple[int, ...]
    factor_qualified_risky_asset_count: tuple[int, ...]
    effective_design_rank: tuple[int, ...]
    weighted_residual_degrees_of_freedom: tuple[int, ...]
    residual_operators: tuple[M03RV11ResidualOperator, ...] | None = None
    imported_architecture_setting_id: str = M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_ALPHA_BATCH_SCHEMA

    def validate(self) -> None:
        self.corrected_batch.validate()
        self.setting.__post_init__()
        expected = self.corrected_batch.predicted_mean.shape[0] * len(M03R_V11_HORIZONS)
        vectors = (
            self.residual_operator_receipt_sha256,
            self.available_risky_asset_count,
            self.factor_qualified_risky_asset_count,
            self.effective_design_rank,
            self.weighted_residual_degrees_of_freedom,
        )
        if (
            self.corrected_batch.target_mode != "factor-residual"
            or self.corrected_batch.exposure_receipt_sha256 is None
            or any(len(value) != expected for value in vectors)
            or any(
                qualified > available or qualified <= 0
                for qualified, available in zip(
                    self.factor_qualified_risky_asset_count,
                    self.available_risky_asset_count,
                    strict=True,
                )
            )
            or any(rank <= 0 for rank in self.effective_design_rank)
            or any(value <= 0 for value in self.weighted_residual_degrees_of_freedom)
            or (
                self.residual_operators is not None
                and (
                    len(self.residual_operators) != expected
                    or tuple(row.receipt_sha256 for row in self.residual_operators)
                    != self.residual_operator_receipt_sha256
                )
            )
            or self.imported_architecture_setting_id
            != M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_ALPHA_BATCH_SCHEMA
        ):
            raise M03RV11PretrainingRuntimeError("v11 alpha batch drifted")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.residual_operator_receipt_sha256
        ):
            raise M03RV11PretrainingRuntimeError(
                "v11 residual operator receipt is malformed"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        base = self.corrected_batch
        return _sha256(
            {
                "schema": self.schema,
                "protocol_sha256": self.protocol_sha256,
                "setting_sha256": self.setting.receipt_sha256,
                "split": base.split,
                "fold_index": base.fold_index,
                "origin_indices": tuple(int(value) for value in base.origin_indices),
                "source_array_sha256": base.source_array_sha256,
                "asset_axis_sha256": base.asset_axis_sha256,
                "exposure_receipt_sha256": base.exposure_receipt_sha256,
                "residual_operator_receipt_sha256": (
                    self.residual_operator_receipt_sha256
                ),
                "available_risky_asset_count": self.available_risky_asset_count,
                "factor_qualified_risky_asset_count": (
                    self.factor_qualified_risky_asset_count
                ),
                "effective_design_rank": self.effective_design_rank,
                "weighted_residual_degrees_of_freedom": (
                    self.weighted_residual_degrees_of_freedom
                ),
                "imported_architecture_setting_id": (
                    self.imported_architecture_setting_id
                ),
                "outer_score_accessed": base.outer_score_accessed,
                "lockbox_accessed": base.lockbox_accessed,
            }
        )


def build_m03r_v11_alpha_batch_from_origin_states(
    policy: Top2000M03RV9PredictivePolicy,
    setting: M03RV11PredictiveSetting,
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
) -> M03RV11AlphaPretrainingBatch:
    """Build targets whose validity is the exact shared operator mask."""

    setting.__post_init__()
    if policy.setting.setting_id != M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID:
        raise M03RV11PretrainingRuntimeError(
            "v11 requires the declared imported predictive architecture"
        )
    base = build_m03r_v9_alpha_batch_from_origin_states(
        policy,
        policy.setting,
        origin_states,
        sequence,
        local_origin_indices,
        sequence_global_state_start=sequence_global_state_start,
        split=split,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        fold_index=fold_index,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
        origin_risk_exposures=origin_risk_exposures,
    )
    targets = torch.zeros_like(base.target_log_return)
    valid = torch.zeros_like(base.valid)
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
            raise M03RV11PretrainingRuntimeError(
                "v11 origin has no point-in-time exposure row"
            )
        for horizon_index, horizon in enumerate(M03R_V11_HORIZONS):
            local_first = local_origin + 1
            local_stop = local_first + horizon
            if (
                global_origin + horizon + 1 > split_stop_exclusive
                or local_stop > sequence.asset_returns.shape[0]
            ):
                raise M03RV11PretrainingRuntimeError(
                    "v11 paired batch contains an unsupported horizon"
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
    corrected = replace(base, target_log_return=targets, valid=valid)
    result = M03RV11AlphaPretrainingBatch(
        corrected_batch=corrected,
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
    "M03R_V11_ALPHA_BATCH_SCHEMA",
    "M03R_V11_IMPORTED_ARCHITECTURE_SETTING_ID",
    "M03RV11AlphaPretrainingBatch",
    "M03RV11PretrainingRuntimeError",
    "build_m03r_v11_alpha_batch_from_origin_states",
]
