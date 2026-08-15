"""Causal long-horizon selection and h3 timing batches for M03R-v16."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SURVIVAL_WEIGHTS,
    M03R_V16_TIMING_HORIZON_SESSIONS,
    M03RV16PredictiveSetting,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    M03RV9OriginRiskExposures,
)
from rl_quant.training.top2000_m03r_v15_residual_operator import (
    M03RV15ResidualOperator,
    apply_m03r_v15_residual_operator,
    build_m03r_v15_residual_operator,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v16_fold import M03R_V16_MINIMUM_LOCAL_ORIGIN
from rl_quant.training.top2000_m03r_v16_objective import M03RV16PredictiveBatch
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
    m03r_v16_score_component_state_sha256,
)

M03R_V16_BUILT_BATCH_SCHEMA = "rl-quant.top2000-dev.m03r-v16-built-batch-v1"


class M03RV16PretrainingRuntimeError(ValueError):
    """The V16 causal multi-horizon batch boundary drifted."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _digest(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def m03r_v16_selection_weights(setting: M03RV16PredictiveSetting) -> torch.Tensor:
    setting.__post_init__()
    if setting.selection_target == "survival-weighted-1-30-mean-factor-residual":
        return torch.tensor(M03R_V16_SURVIVAL_WEIGHTS, dtype=torch.float64)
    return torch.ones(setting.selection_support_sessions, dtype=torch.float64)


@dataclass(frozen=True, slots=True)
class M03RV16BuiltPredictiveBatch:
    objective: M03RV16PredictiveBatch
    raw_selection_mean: torch.Tensor
    raw_timing_mean: torch.Tensor
    origin_indices: torch.Tensor
    split: Literal["training", "inner_validation", "qualification"]
    fold_index: int
    split_start_inclusive: int
    split_stop_exclusive: int
    source_array_sha256: str
    asset_axis_sha256: str
    exposure_receipt_sha256: str
    policy_state_binding_kind: Literal[
        "parameter-version-root", "model-state-sha256"
    ]
    policy_state_binding_sha256: str
    policy_score_component_state_sha256: str
    selection_target_operators: tuple[M03RV15ResidualOperator, ...]
    timing_target_operators: tuple[M03RV15ResidualOperator, ...]
    action_operators: tuple[M03RV15ResidualOperator, ...]
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_BUILT_BATCH_SCHEMA

    def validate(self) -> None:
        self.objective.validate()
        rows = self.objective.executable_selection_mean.shape[0]
        groups = (
            self.selection_target_operators,
            self.timing_target_operators,
            self.action_operators,
        )
        if any(len(group) != rows for group in groups):
            raise M03RV16PretrainingRuntimeError("V16 operator count drifted")
        for operator in (*groups[0], *groups[1], *groups[2]):
            operator.require_fast_identity()
        selection_masks = torch.stack(
            tuple(operator.qualified_asset_mask for operator in groups[0])
        )
        timing_masks = torch.stack(
            tuple(operator.qualified_asset_mask for operator in groups[1])
        )
        support_valid = all(
            not bool((target.qualified_asset_mask & ~action.qualified_asset_mask).any())
            for targets in (groups[0], groups[1])
            for target, action in zip(targets, groups[2], strict=True)
        )
        selection_score_valid = all(
            torch.equal(
                apply_m03r_v15_residual_operator(
                    self.raw_selection_mean[index], operator
                ).residual,
                self.objective.executable_selection_mean[index],
            )
            for index, operator in enumerate(groups[2])
        )
        timing_score_valid = all(
            torch.equal(
                apply_m03r_v15_residual_operator(
                    self.raw_timing_mean[index], operator
                ).residual,
                self.objective.executable_timing_mean[index],
            )
            for index, operator in enumerate(groups[2])
        )
        if (
            tuple(self.raw_selection_mean.shape)
            != tuple(self.objective.executable_selection_mean.shape)
            or tuple(self.raw_timing_mean.shape)
            != tuple(self.objective.executable_timing_mean.shape)
            or not bool(torch.isfinite(self.raw_selection_mean).all())
            or not bool(torch.isfinite(self.raw_timing_mean).all())
            or _tensor_sha256(selection_masks)
            != _tensor_sha256(self.objective.selection_valid)
            or _tensor_sha256(timing_masks) != _tensor_sha256(self.objective.timing_valid)
            or not support_valid
            or not selection_score_valid
            or not timing_score_valid
            or tuple(self.origin_indices.shape) != (rows,)
            or self.origin_indices.dtype != torch.long
            or self.origin_indices.device
            != self.objective.executable_selection_mean.device
            or bool((self.origin_indices[1:] <= self.origin_indices[:-1]).any())
            or int(self.origin_indices[0]) < self.split_start_inclusive
            or bool(
                (
                    self.origin_indices
                    + M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS
                    + 2
                    > self.split_stop_exclusive
                ).any()
            )
            or self.split not in {"training", "inner_validation", "qualification"}
            or self.fold_index not in range(6)
            or not all(
                _digest(value)
                for value in (
                    self.source_array_sha256,
                    self.asset_axis_sha256,
                    self.exposure_receipt_sha256,
                    self.policy_state_binding_sha256,
                    self.policy_score_component_state_sha256,
                )
            )
            or self.policy_state_binding_kind
            not in {"parameter-version-root", "model-state-sha256"}
            or self.policy_state_binding_kind
            != (
                "parameter-version-root"
                if self.split == "training"
                else "model-state-sha256"
            )
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_BUILT_BATCH_SCHEMA
        ):
            raise M03RV16PretrainingRuntimeError("V16 built batch drifted")

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
                    "origin_indices": tuple(int(value) for value in self.origin_indices),
                    "source_array_sha256": self.source_array_sha256,
                    "asset_axis_sha256": self.asset_axis_sha256,
                    "exposure_receipt_sha256": self.exposure_receipt_sha256,
                    "policy_state_binding_kind": self.policy_state_binding_kind,
                    "policy_state_binding_sha256": self.policy_state_binding_sha256,
                    "policy_score_component_state_sha256": (
                        self.policy_score_component_state_sha256
                    ),
                    "raw_selection_mean_sha256": _tensor_sha256(
                        self.raw_selection_mean
                    ),
                    "raw_timing_mean_sha256": _tensor_sha256(self.raw_timing_mean),
                    "selection_target_sha256": _tensor_sha256(
                        self.objective.selection_target
                    ),
                    "timing_target_sha256": _tensor_sha256(
                        self.objective.timing_target
                    ),
                    "selection_operator_receipts": tuple(
                        operator.receipt_sha256
                        for operator in self.selection_target_operators
                    ),
                    "timing_operator_receipts": tuple(
                        operator.receipt_sha256
                        for operator in self.timing_target_operators
                    ),
                    "action_operator_receipts": tuple(
                        operator.receipt_sha256 for operator in self.action_operators
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


def _future_target(
    sequence: Hold30Sequence,
    *,
    local_origin: int,
    horizon_weights: torch.Tensor,
    cash_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = local_origin + 1
    stop = first + horizon_weights.numel()
    if stop > sequence.asset_returns.shape[0]:
        raise M03RV16PretrainingRuntimeError("V16 target path is unavailable")
    weights = horizon_weights.to(
        device=sequence.asset_returns.device,
        dtype=sequence.asset_returns.dtype,
    )
    stock_rows = torch.log1p(
        sequence.asset_returns[first:stop, 0].clamp_min(-0.999999)
    )
    benchmark_rows = torch.log1p(
        sequence.benchmark_net_returns[first:stop, 0].clamp_min(-0.999999)
    )
    target = (stock_rows * weights.unsqueeze(-1)).sum(dim=0)
    benchmark = (benchmark_rows * weights).sum()
    available = sequence.decision_available[first : stop + 1, 0].all(dim=0)
    available = available.clone()
    available[cash_index] = False
    return target - benchmark, available


def build_m03r_v16_batch_from_origin_states(
    policy: Top2000M03RV16PredictivePolicy,
    setting: M03RV16PredictiveSetting,
    origin_states: torch.Tensor,
    sequence: Hold30Sequence,
    local_origin_indices: torch.Tensor,
    *,
    sequence_global_state_start: int,
    split: Literal["training", "inner_validation", "qualification"],
    split_start_inclusive: int,
    split_stop_exclusive: int,
    fold_index: int,
    source_array_sha256: str,
    asset_axis_sha256: str,
    origin_risk_exposures: M03RV9OriginRiskExposures,
) -> M03RV16BuiltPredictiveBatch:
    """Build paired selection/timing targets under one causal action operator."""

    setting.__post_init__()
    if (
        not isinstance(policy, Top2000M03RV16PredictivePolicy)
        or policy.v16_setting != setting
        or origin_states.ndim != 4
        or origin_states.shape[1] != 1
        or origin_states.shape[2] != sequence.num_assets
        or origin_states.shape[3] != policy.token_dim
        or local_origin_indices.ndim != 1
        or local_origin_indices.dtype != torch.long
        or local_origin_indices.device != origin_states.device
        or local_origin_indices.numel() != origin_states.shape[0]
        or int(local_origin_indices[0]) < M03R_V16_MINIMUM_LOCAL_ORIGIN
        or bool((local_origin_indices[1:] <= local_origin_indices[:-1]).any())
    ):
        raise M03RV16PretrainingRuntimeError("V16 policy or origin geometry drifted")
    global_origins = local_origin_indices + sequence_global_state_start
    if (
        int(global_origins[0]) < split_start_inclusive
        or bool(
            (
                global_origins
                + M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS
                + 2
                > split_stop_exclusive
            ).any()
        )
    ):
        raise M03RV16PretrainingRuntimeError("V16 origins leave the declared split")
    origin_risk_exposures.validate()
    if (
        origin_risk_exposures.asset_axis_sha256 != asset_axis_sha256
        or origin_risk_exposures.exposure_loadings.shape[1] != sequence.num_assets
        or origin_risk_exposures.cash_index != sequence.initial_ledger.cash_index
    ):
        raise M03RV16PretrainingRuntimeError("V16 exposure and sequence axes differ")

    selection_weights = m03r_v16_selection_weights(setting)
    timing_weights = torch.ones(M03R_V16_TIMING_HORIZON_SESSIONS, dtype=torch.float64)
    cash_index = sequence.initial_ledger.cash_index
    raw_selection: list[torch.Tensor] = []
    raw_timing: list[torch.Tensor] = []
    executable_selection: list[torch.Tensor] = []
    executable_timing: list[torch.Tensor] = []
    selection_scales: list[torch.Tensor] = []
    timing_scales: list[torch.Tensor] = []
    selection_targets: list[torch.Tensor] = []
    timing_targets: list[torch.Tensor] = []
    selection_valid: list[torch.Tensor] = []
    timing_valid: list[torch.Tensor] = []
    selection_operators: list[M03RV15ResidualOperator] = []
    timing_operators: list[M03RV15ResidualOperator] = []
    action_operators: list[M03RV15ResidualOperator] = []

    if split == "training":
        policy_state_binding_kind: Literal[
            "parameter-version-root", "model-state-sha256"
        ] = "parameter-version-root"
        policy_state_binding_sha256 = hashlib.sha256(
            json.dumps(
                tuple(
                    (name, parameter._version)
                    for name, parameter in policy.named_parameters()
                ),
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
    else:
        policy_state_binding_kind = "model-state-sha256"
        policy_state_binding_sha256 = model_state_sha256(policy)

    for row_index, local_origin in enumerate(local_origin_indices.tolist()):
        global_origin = local_origin + sequence_global_state_start
        exposure_row = global_origin - origin_risk_exposures.state_start_index
        if not 0 <= exposure_row < origin_risk_exposures.exposure_loadings.shape[0]:
            raise M03RV16PretrainingRuntimeError("V16 origin lacks risk exposures")
        output = policy.predictive_output(
            origin_states[row_index], sequence.decision_available[local_origin]
        )
        selection_mean = output.raw_selection_mean.squeeze(0)
        timing_mean = output.raw_timing_mean.squeeze(0)
        action_available = sequence.decision_available[local_origin, 0].clone()
        action_available[cash_index] = False
        selection_raw_target, selection_future = _future_target(
            sequence,
            local_origin=local_origin,
            horizon_weights=selection_weights,
            cash_index=cash_index,
        )
        timing_raw_target, timing_future = _future_target(
            sequence,
            local_origin=local_origin,
            horizon_weights=timing_weights,
            cash_index=cash_index,
        )
        kwargs = {
            "origin_state_index": global_origin,
            "cash_index": cash_index,
            "exposure_loadings": origin_risk_exposures.exposure_loadings[exposure_row],
            "regression_weights": origin_risk_exposures.regression_weights[exposure_row],
            "projector_exposure_names": origin_risk_exposures.projector_exposure_names,
            "projector_exposure_families": (
                origin_risk_exposures.projector_exposure_families
            ),
            "asset_axis_sha256": asset_axis_sha256,
            "source_exposure_receipt_sha256": origin_risk_exposures.receipt_sha256,
        }
        action_operator = build_m03r_v15_residual_operator(
            available_mask=action_available,
            **kwargs,
        )
        selection_operator = build_m03r_v15_residual_operator(
            available_mask=action_available & selection_future,
            **kwargs,
        )
        timing_operator = build_m03r_v15_residual_operator(
            available_mask=action_available & timing_future,
            **kwargs,
        )
        raw_selection.append(selection_mean)
        raw_timing.append(timing_mean)
        executable_selection.append(
            apply_m03r_v15_residual_operator(selection_mean, action_operator).residual
        )
        executable_timing.append(
            apply_m03r_v15_residual_operator(timing_mean, action_operator).residual
        )
        selection_scales.append(output.raw_selection_log_scale.squeeze(0))
        timing_scales.append(output.raw_timing_log_scale.squeeze(0))
        selection_result = apply_m03r_v15_residual_operator(
            selection_raw_target.detach(), selection_operator
        )
        timing_result = apply_m03r_v15_residual_operator(
            timing_raw_target.detach(), timing_operator
        )
        selection_targets.append(selection_result.residual)
        timing_targets.append(timing_result.residual)
        selection_valid.append(selection_result.qualified_asset_mask)
        timing_valid.append(timing_result.qualified_asset_mask)
        selection_operators.append(selection_operator)
        timing_operators.append(timing_operator)
        action_operators.append(action_operator)

    objective = M03RV16PredictiveBatch(
        executable_selection_mean=torch.stack(executable_selection),
        selection_log_scale=torch.stack(selection_scales),
        selection_target=torch.stack(selection_targets),
        selection_valid=torch.stack(selection_valid),
        executable_timing_mean=torch.stack(executable_timing),
        timing_log_scale=torch.stack(timing_scales),
        timing_target=torch.stack(timing_targets),
        timing_valid=torch.stack(timing_valid),
        setting=setting,
    )
    result = M03RV16BuiltPredictiveBatch(
        objective=objective,
        raw_selection_mean=torch.stack(raw_selection),
        raw_timing_mean=torch.stack(raw_timing),
        origin_indices=global_origins,
        split=split,
        fold_index=fold_index,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
        exposure_receipt_sha256=origin_risk_exposures.receipt_sha256,
        policy_state_binding_kind=policy_state_binding_kind,
        policy_state_binding_sha256=policy_state_binding_sha256,
        policy_score_component_state_sha256=(
            m03r_v16_score_component_state_sha256(policy)
        ),
        selection_target_operators=tuple(selection_operators),
        timing_target_operators=tuple(timing_operators),
        action_operators=tuple(action_operators),
    )
    result.validate()
    return result


__all__ = [
    "M03R_V16_BUILT_BATCH_SCHEMA",
    "M03RV16BuiltPredictiveBatch",
    "M03RV16PretrainingRuntimeError",
    "build_m03r_v16_batch_from_origin_states",
    "m03r_v16_selection_weights",
]
