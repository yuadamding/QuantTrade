"""Paired common-support selection batches for corrected M03R-v16."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_COMMON_LABEL_SUPPORT_SESSIONS,
    M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03RV16PredictiveSetting,
    m03r_v16_selection_target_weights_from_id,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    M03RV9OriginRiskExposures,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v15_residual_operator import (
    M03RV15ResidualOperator,
    M03RV15ResidualResult,
    apply_m03r_v15_residual_operator,
    build_m03r_v15_residual_operator,
)
from rl_quant.training.top2000_m03r_v16_fold import M03R_V16_MINIMUM_LOCAL_ORIGIN
from rl_quant.training.top2000_m03r_v16_objective import M03RV16PredictiveBatch
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)

if TYPE_CHECKING:
    from rl_quant.training.top2000_m03r_v16_structural import (
        M03RV16ValidatedStructuralSlab,
    )

M03R_V16_BUILT_BATCH_SCHEMA = "rl-quant.top2000-dev.m03r-v16-built-batch-v4"
M03R_V16_RETURNED_DTYPE_ORTHOGONALITY_TOLERANCE = 1.0e-5
_BUILT_BATCH_ISSUER = object()


class M03RV16PretrainingRuntimeError(ValueError):
    """The V16 causal selection batch boundary drifted."""


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
    """Return economic-unit target weights; they are never normalized."""

    setting.__post_init__()
    values = m03r_v16_selection_target_weights_from_id(setting.selection_target)
    return torch.tensor(values, dtype=torch.float64)


def _returned_dtype_exposure_error(
    result: M03RV15ResidualResult,
    operator: M03RV15ResidualOperator,
) -> float:
    selected = torch.nonzero(
        operator.qualified_asset_mask.to(device=result.residual.device), as_tuple=False
    ).flatten()
    returned = result.residual.index_select(0, selected).to(torch.float64)
    design = operator.base.qualified_design.to(device=returned.device)
    weights = operator.base.qualified_weights.to(device=returned.device)
    return float((design.T @ (weights * returned)).abs().max())


@dataclass(frozen=True, slots=True)
class M03RV16BuiltPredictiveBatch:
    objective: M03RV16PredictiveBatch
    raw_selection_score_z: torch.Tensor
    action_valid: torch.Tensor
    origin_indices: torch.Tensor
    split: Literal["training", "inner_validation", "qualification"]
    fold_index: int
    split_start_inclusive: int
    split_stop_exclusive: int
    source_array_sha256: str
    asset_axis_sha256: str
    exposure_receipt_sha256: str
    policy_state_binding_kind: Literal["parameter-version-root", "model-state-sha256"]
    policy_state_binding_sha256: str
    selection_target_operators: tuple[M03RV15ResidualOperator, ...]
    action_operators: tuple[M03RV15ResidualOperator, ...]
    action_returned_dtype_exposure_errors: tuple[float, ...]
    target_returned_dtype_exposure_errors: tuple[float, ...]
    _issuer: object = field(repr=False)
    structural_slab_receipt_sha256: str | None = None
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_BUILT_BATCH_SCHEMA

    def validate(self) -> None:
        self.objective.validate()
        rows = self.objective.executable_selection_score_z.shape[0]
        if (
            len(self.selection_target_operators) != rows
            or len(self.action_operators) != rows
            or len(self.action_returned_dtype_exposure_errors) != rows
            or len(self.target_returned_dtype_exposure_errors) != rows
        ):
            raise M03RV16PretrainingRuntimeError("V16 operator count drifted")
        for operator in (*self.selection_target_operators, *self.action_operators):
            operator.require_fast_identity()
        target_masks = torch.stack(
            tuple(
                operator.qualified_asset_mask
                for operator in self.selection_target_operators
            )
        )
        action_masks = torch.stack(
            tuple(operator.qualified_asset_mask for operator in self.action_operators)
        )
        support_valid = all(
            not bool((target.qualified_asset_mask & ~action.qualified_asset_mask).any())
            for target, action in zip(
                self.selection_target_operators, self.action_operators, strict=True
            )
        )
        errors = (
            *self.action_returned_dtype_exposure_errors,
            *self.target_returned_dtype_exposure_errors,
        )
        spec = M03R_V16_PREDICTIVE_SPEC

        if (
            tuple(self.raw_selection_score_z.shape)
            != tuple(self.objective.executable_selection_score_z.shape)
            or self.action_valid.shape != self.objective.selection_valid.shape
            or self.action_valid.dtype != torch.bool
            or not bool(torch.isfinite(self.raw_selection_score_z).all())
            or _tensor_sha256(target_masks)
            != _tensor_sha256(self.objective.selection_valid)
            or _tensor_sha256(action_masks) != _tensor_sha256(self.action_valid)
            or bool((self.objective.selection_valid & ~self.action_valid).any())
            or not support_valid
            or any(
                not math.isfinite(value)
                or value > M03R_V16_RETURNED_DTYPE_ORTHOGONALITY_TOLERANCE
                for value in errors
            )
            or tuple(self.origin_indices.shape) != (rows,)
            or self.origin_indices.dtype != torch.long
            or self.origin_indices.device
            != self.objective.executable_selection_score_z.device
            or bool((self.origin_indices[1:] <= self.origin_indices[:-1]).any())
            or int(self.origin_indices[0]) < self.split_start_inclusive
            or bool(
                (
                    self.origin_indices + M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS + 2
                    > self.split_stop_exclusive
                ).any()
            )
            or self.split not in {"training", "inner_validation", "qualification"}
            or self.fold_index not in range(spec.chronological_fold_count)
            or not all(
                _digest(value)
                for value in (
                    self.source_array_sha256,
                    self.asset_axis_sha256,
                    self.exposure_receipt_sha256,
                    self.policy_state_binding_sha256,
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
            or (
                self.structural_slab_receipt_sha256 is not None
                and not _digest(self.structural_slab_receipt_sha256)
            )
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_BUILT_BATCH_SCHEMA
            or self._issuer is not _BUILT_BATCH_ISSUER
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
                    "origin_indices": tuple(
                        int(value) for value in self.origin_indices
                    ),
                    "source_array_sha256": self.source_array_sha256,
                    "asset_axis_sha256": self.asset_axis_sha256,
                    "exposure_receipt_sha256": self.exposure_receipt_sha256,
                    "policy_state_binding_kind": self.policy_state_binding_kind,
                    "policy_state_binding_sha256": self.policy_state_binding_sha256,
                    "raw_selection_score_z_sha256": _tensor_sha256(
                        self.raw_selection_score_z
                    ),
                    "executable_selection_score_z_sha256": _tensor_sha256(
                        self.objective.executable_selection_score_z
                    ),
                    "action_valid_sha256": _tensor_sha256(self.action_valid),
                    "selection_target_economic_sha256": _tensor_sha256(
                        self.objective.selection_target_economic
                    ),
                    "selection_target_z_sha256": _tensor_sha256(
                        self.objective.selection_target_z
                    ),
                    "selection_operator_receipts": tuple(
                        operator.receipt_sha256
                        for operator in self.selection_target_operators
                    ),
                    "action_operator_receipts": tuple(
                        operator.receipt_sha256 for operator in self.action_operators
                    ),
                    "action_returned_dtype_exposure_errors": (
                        self.action_returned_dtype_exposure_errors
                    ),
                    "target_returned_dtype_exposure_errors": (
                        self.target_returned_dtype_exposure_errors
                    ),
                    "structural_slab_receipt_sha256": (
                        self.structural_slab_receipt_sha256
                    ),
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


def _future_selection_target(
    sequence: Hold30Sequence,
    *,
    local_origin: int,
    target_weights: torch.Tensor,
    cash_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = local_origin + 1
    target_stop = first + target_weights.numel()
    support_stop = first + M03R_V16_COMMON_LABEL_SUPPORT_SESSIONS
    if (
        target_stop > sequence.asset_returns.shape[0]
        or support_stop >= sequence.decision_available.shape[0]
    ):
        raise M03RV16PretrainingRuntimeError("V16 target path is unavailable")
    weights = target_weights.to(
        device=sequence.asset_returns.device,
        dtype=sequence.asset_returns.dtype,
    )
    stock_rows = torch.log1p(
        sequence.asset_returns[first:target_stop, 0].clamp_min(-0.999999)
    )
    benchmark_rows = torch.log1p(
        sequence.benchmark_net_returns[first:target_stop, 0].clamp_min(-0.999999)
    )
    target = (stock_rows * weights.unsqueeze(-1)).sum(dim=0)
    benchmark = (benchmark_rows * weights).sum()
    available = sequence.decision_available[first : support_stop + 1, 0].all(dim=0)
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
    structural_slab: M03RV16ValidatedStructuralSlab | None = None,
) -> M03RV16BuiltPredictiveBatch:
    """Build one target-only batch on the common 30-session support."""

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
    if int(global_origins[0]) < split_start_inclusive or bool(
        (
            global_origins + M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS + 2
            > split_stop_exclusive
        ).any()
    ):
        raise M03RV16PretrainingRuntimeError("V16 origins leave the declared split")
    origin_risk_exposures.validate()
    if (
        origin_risk_exposures.asset_axis_sha256 != asset_axis_sha256
        or origin_risk_exposures.exposure_loadings.shape[1] != sequence.num_assets
        or origin_risk_exposures.cash_index != sequence.initial_ledger.cash_index
    ):
        raise M03RV16PretrainingRuntimeError("V16 exposure and sequence axes differ")
    if structural_slab is not None:
        structural_slab.require_fast_identity()
        if (
            structural_slab.receipt.asset_axis_sha256 != asset_axis_sha256
            or structural_slab.receipt.exposure_receipt_sha256
            != origin_risk_exposures.receipt_sha256
        ):
            raise M03RV16PretrainingRuntimeError(
                "V16 structural slab does not bind the batch risk axis"
            )

    target_weights = m03r_v16_selection_weights(setting)
    cash_index = sequence.initial_ledger.cash_index
    raw_scores: list[torch.Tensor] = []
    executable_scores: list[torch.Tensor] = []
    economic_targets: list[torch.Tensor] = []
    target_z: list[torch.Tensor] = []
    target_valid: list[torch.Tensor] = []
    action_valid: list[torch.Tensor] = []
    target_operators: list[M03RV15ResidualOperator] = []
    action_operators: list[M03RV15ResidualOperator] = []
    action_error_tensors: list[torch.Tensor] = []
    target_errors: list[float] = []

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
        raw_score = output.raw_selection_score_z.squeeze(0)
        action_available = sequence.decision_available[local_origin, 0].clone()
        action_available[cash_index] = False
        if structural_slab is None:
            raw_target, common_future = _future_selection_target(
                sequence,
                local_origin=local_origin,
                target_weights=target_weights,
                cash_index=cash_index,
            )
            kwargs = {
                "origin_state_index": global_origin,
                "cash_index": cash_index,
                "exposure_loadings": origin_risk_exposures.exposure_loadings[
                    exposure_row
                ],
                "regression_weights": origin_risk_exposures.regression_weights[
                    exposure_row
                ],
                "projector_exposure_names": (
                    origin_risk_exposures.projector_exposure_names
                ),
                "projector_exposure_families": (
                    origin_risk_exposures.projector_exposure_families
                ),
                "asset_axis_sha256": asset_axis_sha256,
                "source_exposure_receipt_sha256": (
                    origin_risk_exposures.receipt_sha256
                ),
            }
            action_operator = build_m03r_v15_residual_operator(
                available_mask=action_available,
                **kwargs,
            )
            target_operator = build_m03r_v15_residual_operator(
                available_mask=action_available & common_future,
                **kwargs,
            )
            target_result = apply_m03r_v15_residual_operator(
                raw_target.detach(), target_operator
            )
            economic_target = target_result.residual
            standardized_target = (
                target_result.residual / setting.selection_target_scale
            )
            target_error = _returned_dtype_exposure_error(
                target_result, target_operator
            )
        else:
            prepared = structural_slab.origin(global_origin)
            device_prepared = structural_slab.device_origin(
                global_origin, origin_states.device
            )
            action_operator = prepared.action_operator
            target_operator = prepared.common_target_operator
            expected_action_mask = action_available.to(device="cpu") & (
                origin_risk_exposures.regression_weights[exposure_row]
                .to(device="cpu")
                .gt(0.0)
            )
            expected_action_mask[cash_index] = False
            if not torch.equal(
                action_operator.qualified_asset_mask, expected_action_mask
            ):
                raise M03RV16PretrainingRuntimeError(
                    "V16 package-owned action support drifted"
                )
            economic_target = device_prepared.economic_targets[setting.setting_index]
            standardized_target = device_prepared.standardized_targets[
                setting.setting_index
            ]
            target_error = prepared.target_returned_dtype_exposure_errors[
                setting.setting_index
            ]
        if structural_slab is None:
            score_result = apply_m03r_v15_residual_operator(raw_score, action_operator)
            executable_score = score_result.residual
            action_error = raw_score.new_tensor(
                _returned_dtype_exposure_error(score_result, action_operator)
            )
            action_mask = action_operator.qualified_asset_mask.to(
                device=origin_states.device
            )
        else:
            device_operator = device_prepared.action_operator
            executable_score, action_error = device_operator.apply(raw_score)
            action_mask = device_operator.qualified_asset_mask
        raw_scores.append(raw_score)
        executable_scores.append(executable_score)
        economic_targets.append(economic_target)
        target_z.append(standardized_target)
        target_valid.append(
            target_operator.qualified_asset_mask.to(device=origin_states.device)
            if structural_slab is None
            else device_prepared.common_target_mask
        )
        action_valid.append(action_mask)
        target_operators.append(target_operator)
        action_operators.append(action_operator)
        action_error_tensors.append(action_error)
        target_errors.append(target_error)

    action_errors = tuple(
        float(value)
        for value in torch.stack(action_error_tensors).detach().to(device="cpu")
    )

    objective = M03RV16PredictiveBatch(
        executable_selection_score_z=torch.stack(executable_scores),
        selection_target_z=torch.stack(target_z),
        selection_target_economic=torch.stack(economic_targets),
        selection_valid=torch.stack(target_valid),
        setting=setting,
    )
    result = M03RV16BuiltPredictiveBatch(
        objective=objective,
        raw_selection_score_z=torch.stack(raw_scores),
        action_valid=torch.stack(action_valid),
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
        selection_target_operators=tuple(target_operators),
        action_operators=tuple(action_operators),
        action_returned_dtype_exposure_errors=action_errors,
        target_returned_dtype_exposure_errors=tuple(target_errors),
        structural_slab_receipt_sha256=(
            None if structural_slab is None else structural_slab.receipt_sha256
        ),
        _issuer=_BUILT_BATCH_ISSUER,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V16_BUILT_BATCH_SCHEMA",
    "M03R_V16_RETURNED_DTYPE_ORTHOGONALITY_TOLERANCE",
    "M03RV16BuiltPredictiveBatch",
    "M03RV16PretrainingRuntimeError",
    "build_m03r_v16_batch_from_origin_states",
    "m03r_v16_selection_weights",
]
