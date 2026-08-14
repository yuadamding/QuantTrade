"""Causal h3 target and direct-score batch construction for M03R-v13."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import (
    M03R_V13_PROTOCOL_SHA256,
    M03R_V13_SELECTED_HORIZON_SESSIONS,
    M03RV13PredictiveSetting,
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
from rl_quant.training.top2000_m03r_v13_fold import M03R_V13_MINIMUM_LOCAL_ORIGIN
from rl_quant.training.top2000_m03r_v13_objective import M03RV13PredictiveBatch
from rl_quant.training.top2000_m03r_v13_policy import (
    Top2000M03RV13PredictivePolicy,
)

M03R_V13_BUILT_BATCH_SCHEMA = "rl-quant.top2000-dev.m03r-v13-built-batch-v1"


class M03RV13PretrainingRuntimeError(ValueError):
    """V13 failed its full-context direct-h3 batch boundary."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class M03RV13BuiltPredictiveBatch:
    objective: M03RV13PredictiveBatch
    origin_indices: torch.Tensor
    split: Literal["training", "qualification"]
    fold_index: int
    split_start_inclusive: int
    split_stop_exclusive: int
    source_array_sha256: str
    asset_axis_sha256: str
    exposure_receipt_sha256: str
    target_residual_operators: tuple[M03RV11ResidualOperator, ...]
    action_residual_operators: tuple[M03RV11ResidualOperator, ...]
    protocol_sha256: str = M03R_V13_PROTOCOL_SHA256
    schema: str = M03R_V13_BUILT_BATCH_SCHEMA

    def validate(self) -> None:
        self.objective.validate()
        rows = self.objective.predicted_mean.shape[0]
        if (
            not isinstance(self.origin_indices, torch.Tensor)
            or tuple(self.origin_indices.shape) != (rows,)
            or self.origin_indices.dtype != torch.long
            or self.origin_indices.device != self.objective.predicted_mean.device
            or self.origin_indices.numel() == 0
            or bool((self.origin_indices[1:] <= self.origin_indices[:-1]).any())
            or int(self.origin_indices[0]) < self.split_start_inclusive
            or bool(
                (
                    self.origin_indices
                    + M03R_V13_SELECTED_HORIZON_SESSIONS
                    + 2
                    > self.split_stop_exclusive
                ).any()
            )
            or self.split not in {"training", "qualification"}
            or self.fold_index not in range(6)
            or len(self.target_residual_operators) != rows
            or len(self.action_residual_operators) != rows
            or any(
                operator.origin_state_index != int(origin)
                for operator, origin in zip(
                    self.target_residual_operators,
                    self.origin_indices,
                    strict=True,
                )
            )
            or any(
                operator.origin_state_index != int(origin)
                for operator, origin in zip(
                    self.action_residual_operators,
                    self.origin_indices,
                    strict=True,
                )
            )
            or not all(
                _valid_digest(value)
                for value in (
                    self.source_array_sha256,
                    self.asset_axis_sha256,
                    self.exposure_receipt_sha256,
                )
            )
            or self.protocol_sha256 != M03R_V13_PROTOCOL_SHA256
            or self.schema != M03R_V13_BUILT_BATCH_SCHEMA
        ):
            raise M03RV13PretrainingRuntimeError("v13 built batch drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": self.schema,
                    "protocol_sha256": self.protocol_sha256,
                    "setting_sha256": self.objective.setting.receipt_sha256,
                    "split": self.split,
                    "fold_index": self.fold_index,
                    "split_start_inclusive": self.split_start_inclusive,
                    "split_stop_exclusive": self.split_stop_exclusive,
                    "origin_indices": tuple(int(value) for value in self.origin_indices),
                    "source_array_sha256": self.source_array_sha256,
                    "asset_axis_sha256": self.asset_axis_sha256,
                    "exposure_receipt_sha256": self.exposure_receipt_sha256,
                    "predicted_mean_sha256": _tensor_sha256(
                        self.objective.predicted_mean
                    ),
                    "predicted_log_scale_sha256": _tensor_sha256(
                        self.objective.predicted_log_scale
                    ),
                    "target_log_return_sha256": _tensor_sha256(
                        self.objective.target_log_return
                    ),
                    "valid_sha256": _tensor_sha256(self.objective.valid),
                    "target_residual_operator_receipts": tuple(
                        operator.receipt_sha256
                        for operator in self.target_residual_operators
                    ),
                    "action_residual_operator_receipts": tuple(
                        operator.receipt_sha256
                        for operator in self.action_residual_operators
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


def build_m03r_v13_batch_from_origin_states(
    policy: Top2000M03RV13PredictivePolicy,
    setting: M03RV13PredictiveSetting,
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
) -> M03RV13BuiltPredictiveBatch:
    """Build only h3 labels and distinct target/action residual operators."""

    setting.__post_init__()
    if (
        not isinstance(policy, Top2000M03RV13PredictivePolicy)
        or policy.v13_setting != setting
        or not isinstance(origin_states, torch.Tensor)
        or origin_states.ndim != 4
        or origin_states.shape[1] != 1
        or origin_states.shape[2] != sequence.num_assets
        or origin_states.shape[3] != policy.token_dim
        or not origin_states.is_floating_point()
        or not bool(torch.isfinite(origin_states).all())
        or not isinstance(local_origin_indices, torch.Tensor)
        or local_origin_indices.ndim != 1
        or local_origin_indices.dtype != torch.long
        or local_origin_indices.device != origin_states.device
        or local_origin_indices.numel() == 0
        or local_origin_indices.numel() != origin_states.shape[0]
        or bool((local_origin_indices[1:] <= local_origin_indices[:-1]).any())
        or int(local_origin_indices[0]) < M03R_V13_MINIMUM_LOCAL_ORIGIN
    ):
        raise M03RV13PretrainingRuntimeError(
            "v13 policy, state, origin, or full-context geometry drifted"
        )
    global_origins = local_origin_indices + sequence_global_state_start
    if (
        int(global_origins[0]) < split_start_inclusive
        or bool(
            (
                global_origins
                + M03R_V13_SELECTED_HORIZON_SESSIONS
                + 2
                > split_stop_exclusive
            ).any()
        )
    ):
        raise M03RV13PretrainingRuntimeError("v13 origins leave the declared split")
    origin_risk_exposures.validate()
    if (
        origin_risk_exposures.asset_axis_sha256 != asset_axis_sha256
        or origin_risk_exposures.exposure_loadings.shape[1] != sequence.num_assets
        or origin_risk_exposures.cash_index != sequence.initial_ledger.cash_index
    ):
        raise M03RV13PretrainingRuntimeError(
            "v13 exposure and sequence asset axes differ"
        )

    means: list[torch.Tensor] = []
    log_scales: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    target_operators: list[M03RV11ResidualOperator] = []
    action_operators: list[M03RV11ResidualOperator] = []
    cash_index = sequence.initial_ledger.cash_index
    horizon = M03R_V13_SELECTED_HORIZON_SESSIONS
    for row_index, local_origin in enumerate(local_origin_indices.tolist()):
        global_origin = local_origin + sequence_global_state_start
        exposure_row = global_origin - origin_risk_exposures.state_start_index
        if not 0 <= exposure_row < origin_risk_exposures.exposure_loadings.shape[0]:
            raise M03RV13PretrainingRuntimeError(
                "v13 origin has no point-in-time exposure row"
            )
        output = policy.predictive_output(
            origin_states[row_index], sequence.decision_available[local_origin]
        )
        means.append(output.economic_mean.squeeze(0))
        log_scales.append(output.economic_log_scale.squeeze(0))

        local_first = local_origin + 1
        local_stop = local_first + horizon
        if local_stop > sequence.asset_returns.shape[0]:
            raise M03RV13PretrainingRuntimeError("v13 h3 target path is unavailable")
        stock = torch.log1p(
            sequence.asset_returns[local_first:local_stop, 0].clamp_min(-0.999999)
        ).sum(dim=0)
        benchmark = torch.log1p(
            sequence.benchmark_net_returns[local_first:local_stop, 0].clamp_min(
                -0.999999
            )
        ).sum()
        target_available = sequence.decision_available[
            local_first : local_stop + 1, 0
        ].all(dim=0)
        target_available = target_available.clone()
        target_available[cash_index] = False
        action_available = sequence.decision_available[local_origin, 0].clone()
        action_available[cash_index] = False
        operator_kwargs = {
            "origin_state_index": global_origin,
            "cash_index": cash_index,
            "exposure_loadings": origin_risk_exposures.exposure_loadings[exposure_row],
            "regression_weights": origin_risk_exposures.regression_weights[exposure_row],
            "projector_exposure_names": (
                origin_risk_exposures.projector_exposure_names
            ),
            "projector_exposure_families": (
                origin_risk_exposures.projector_exposure_families
            ),
            "asset_axis_sha256": asset_axis_sha256,
            "source_exposure_receipt_sha256": origin_risk_exposures.receipt_sha256,
        }
        target_operator = build_m03r_v11_residual_operator(
            available_mask=target_available,
            **operator_kwargs,
        )
        action_operator = build_m03r_v11_residual_operator(
            available_mask=action_available,
            **operator_kwargs,
        )
        residual = apply_m03r_v11_residual_operator(
            (stock - benchmark).detach(), target_operator
        )
        targets.append(residual.residual)
        valid.append(residual.qualified_asset_mask)
        target_operators.append(target_operator)
        action_operators.append(action_operator)

    objective = M03RV13PredictiveBatch(
        predicted_mean=torch.stack(means),
        predicted_log_scale=torch.stack(log_scales),
        target_log_return=torch.stack(targets),
        valid=torch.stack(valid),
        setting=setting,
    )
    result = M03RV13BuiltPredictiveBatch(
        objective=objective,
        origin_indices=global_origins,
        split=split,
        fold_index=fold_index,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
        exposure_receipt_sha256=origin_risk_exposures.receipt_sha256,
        target_residual_operators=tuple(target_operators),
        action_residual_operators=tuple(action_operators),
    )
    result.validate()
    return result


__all__ = [
    "M03R_V13_BUILT_BATCH_SCHEMA",
    "M03RV13BuiltPredictiveBatch",
    "M03RV13PretrainingRuntimeError",
    "build_m03r_v13_batch_from_origin_states",
]
