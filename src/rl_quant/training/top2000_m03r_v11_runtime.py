"""Deterministic corrected simple-sleeve runtime for M03R-v11."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import torch

from rl_quant.execution.cost_aware_active_policy_v3 import (
    build_cost_aware_active_proposal_v3,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
    M03R_V11_SETTING_IDS,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_policy import M03RV9AlphaDistribution
from rl_quant.training.top2000_m03r_v9_projection import (
    M03RV9DeviceRiskState,
    project_m03r_v9_active_book,
)
from rl_quant.training.top2000_m03r_v11_residual_operator import (
    M03RV11ResidualOperator,
    apply_m03r_v11_residual_operator,
)

M03R_V11_SIMPLE_SLEEVE_TRACE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-simple-sleeve-trace-v1"
)


class M03RV11RuntimeError(ValueError):
    """The corrected deterministic v11 sleeve is malformed."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


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


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11RuntimeError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV11SimpleSleeveTrace:
    setting_index: int
    setting_id: str
    fold_index: int
    selected_horizon_sessions: int
    checkpoint_file_sha256: str
    checkpoint_model_state_sha256: str
    source_receipt_sha256: str
    asset_axis_sha256: str
    risk_state_sha256: str
    risk_manifest_sha256: str
    signal_operator_receipt_sha256: tuple[str, ...]
    state_start_index: int
    policy_gross_returns: torch.Tensor
    benchmark_gross_returns: torch.Tensor
    policy_one_way_turnover: torch.Tensor
    benchmark_one_way_turnover: torch.Tensor
    requested_weight_trace: torch.Tensor
    projected_weight_trace: torch.Tensor
    requested_to_executed_retention: torch.Tensor
    array_sha256: tuple[str, ...]
    trace_sha256: str
    learned_hazard_enabled: bool = False
    imported_v9_signal_projector_used: bool = False
    v11_shared_residual_operator_used: bool = True
    v11_cost_aware_allocator_used: bool = True
    economic_optimizer_updates: int = 0
    protocol_sha256: str = M03R_V11_PROTOCOL_SHA256
    schema: str = M03R_V11_SIMPLE_SLEEVE_TRACE_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "selected_horizon_sessions": self.selected_horizon_sessions,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "checkpoint_model_state_sha256": self.checkpoint_model_state_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "asset_axis_sha256": self.asset_axis_sha256,
            "risk_state_sha256": self.risk_state_sha256,
            "risk_manifest_sha256": self.risk_manifest_sha256,
            "signal_operator_receipt_sha256": (self.signal_operator_receipt_sha256),
            "state_start_index": self.state_start_index,
            "array_sha256": self.array_sha256,
            "learned_hazard_enabled": self.learned_hazard_enabled,
            "imported_v9_signal_projector_used": (
                self.imported_v9_signal_projector_used
            ),
            "v11_shared_residual_operator_used": (
                self.v11_shared_residual_operator_used
            ),
            "v11_cost_aware_allocator_used": self.v11_cost_aware_allocator_used,
            "economic_optimizer_updates": self.economic_optimizer_updates,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }

    def validate(self) -> None:
        arrays = (
            self.policy_gross_returns,
            self.benchmark_gross_returns,
            self.policy_one_way_turnover,
            self.benchmark_one_way_turnover,
            self.requested_weight_trace,
            self.projected_weight_trace,
            self.requested_to_executed_retention,
        )
        transitions = self.policy_gross_returns.numel()
        if (
            self.setting_index not in range(3)
            or self.setting_id != M03R_V11_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.selected_horizon_sessions not in {21, 30}
            or transitions < 2
            or len(self.signal_operator_receipt_sha256) != transitions
            or len(set(self.signal_operator_receipt_sha256)) != transitions
            or self.policy_gross_returns.ndim != 1
            or any(
                value.ndim != 1
                or value.shape != self.policy_gross_returns.shape
                or not value.is_floating_point()
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
                for value in arrays[1:4]
            )
            or self.requested_weight_trace.ndim != 2
            or self.requested_weight_trace.shape[0] != transitions
            or self.projected_weight_trace.shape != self.requested_weight_trace.shape
            or tuple(self.requested_to_executed_retention.shape) != (transitions,)
            or any(
                not value.is_floating_point()
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
                for value in arrays[4:]
            )
            or bool((self.policy_one_way_turnover < 0.0).any())
            or bool((self.benchmark_one_way_turnover < 0.0).any())
            or bool((self.requested_to_executed_retention < 0.0).any())
            or bool((self.requested_to_executed_retention > 1.0 + 1.0e-8).any())
            or tuple(_tensor_sha256(value) for value in arrays) != self.array_sha256
            or self.learned_hazard_enabled
            or self.imported_v9_signal_projector_used
            or not self.v11_shared_residual_operator_used
            or not self.v11_cost_aware_allocator_used
            or self.economic_optimizer_updates != 0
            or self.protocol_sha256 != M03R_V11_PROTOCOL_SHA256
            or self.schema != M03R_V11_SIMPLE_SLEEVE_TRACE_SCHEMA
            or self.trace_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11RuntimeError("v11 simple-sleeve trace drifted")
        for name, value in (
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("checkpoint_model_state_sha256", self.checkpoint_model_state_sha256),
            ("source_receipt_sha256", self.source_receipt_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
            ("risk_state_sha256", self.risk_state_sha256),
            ("risk_manifest_sha256", self.risk_manifest_sha256),
            ("trace_sha256", self.trace_sha256),
            *(
                ("signal_operator_receipt_sha256", value)
                for value in self.signal_operator_receipt_sha256
            ),
        ):
            _digest(name, value)


def run_m03r_v11_simple_sleeve(
    sequence: Hold30Sequence,
    alpha_distributions: tuple[M03RV9AlphaDistribution, ...],
    signal_operators: tuple[M03RV11ResidualOperator, ...],
    risk_state: M03RV9DeviceRiskState,
    *,
    setting_index: int,
    fold_index: int,
    selected_horizon_sessions: int,
    state_start_index: int,
    checkpoint_file_sha256: str,
    checkpoint_model_state_sha256: str,
    checkpoint_asset_axis_sha256: str,
    source_receipt_sha256: str,
    benchmark_gross_returns: torch.Tensor,
    benchmark_one_way_turnover: torch.Tensor,
) -> M03RV11SimpleSleeveTrace:
    """Run one hazard-free chronology with exact v11 residual operators."""

    transitions = sequence.asset_returns.shape[0]
    if (
        setting_index not in range(3)
        or fold_index not in range(6)
        or selected_horizon_sessions not in {21, 30}
        or sequence.asset_returns.shape[1] != 1
        or len(alpha_distributions) != transitions
        or len(signal_operators) != transitions
        or sequence.axis_id != risk_state.asset_axis_sha256
        or checkpoint_asset_axis_sha256 != risk_state.asset_axis_sha256
        or tuple(risk_state.origin_state_indices)
        != tuple(state_start_index + index for index in range(transitions))
        or tuple(benchmark_gross_returns.shape) != (transitions,)
        or tuple(benchmark_one_way_turnover.shape) != (transitions,)
        or benchmark_gross_returns.dtype != sequence.asset_returns.dtype
        or benchmark_one_way_turnover.dtype != sequence.asset_returns.dtype
        or benchmark_gross_returns.device != sequence.asset_returns.device
        or benchmark_one_way_turnover.device != sequence.asset_returns.device
        or not bool(torch.isfinite(benchmark_gross_returns).all())
        or not bool(torch.isfinite(benchmark_one_way_turnover).all())
        or bool((benchmark_one_way_turnover < 0.0).any())
    ):
        raise M03RV11RuntimeError("v11 sleeve source axes or chronology drifted")
    risk_state.validate()
    risk_state.require_fast_identity(
        sequence_asset_axis_sha256=sequence.axis_id,
        checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
        expected_manifest_sha256=risk_state.manifest_sha256,
    )
    for name, value in (
        ("checkpoint_file_sha256", checkpoint_file_sha256),
        ("checkpoint_model_state_sha256", checkpoint_model_state_sha256),
        ("source_receipt_sha256", source_receipt_sha256),
    ):
        _digest(name, value)

    weights = sequence.benchmark_weights[0].clone()
    policy_gross: list[torch.Tensor] = []
    policy_turnover: list[torch.Tensor] = []
    requested_weights: list[torch.Tensor] = []
    projected_weights: list[torch.Tensor] = []
    projection_retention: list[torch.Tensor] = []
    for transition_index, (distribution, operator) in enumerate(
        zip(alpha_distributions, signal_operators, strict=True)
    ):
        distribution.validate()
        operator.validate()
        origin_state_index = state_start_index + transition_index
        if (
            distribution.selected_horizon_sessions != selected_horizon_sessions
            or distribution.selected_mean.shape != weights.shape
            or distribution.selected_mean.device != weights.device
            or operator.origin_state_index != origin_state_index
            or operator.asset_axis_sha256 != sequence.axis_id
            or operator.source_exposure_receipt_sha256
            != risk_state.source_exposure_receipt_sha256
            or operator.qualified_asset_mask.numel() != weights.shape[1]
        ):
            raise M03RV11RuntimeError(
                "v11 distribution or shared residual operator drifted"
            )

        returns = sequence.asset_returns[transition_index]
        gross = (weights * returns).sum(-1)
        if bool((gross <= -1.0).any()):
            raise M03RV11RuntimeError("v11 simple sleeve lost all value")
        pretrade = weights * (1.0 + returns) / (1.0 + gross).unsqueeze(-1)
        fill_index = transition_index + 1
        fill_mask = (
            sequence.fill_membership[fill_index]
            & sequence.fill_availability[fill_index]
        )
        repaired = project_m03r_v9_active_book(
            pretrade,
            sequence.benchmark_weights[fill_index],
            fill_mask,
            sequence.risk_asset_caps[fill_index],
            sequence.risk_gross_max[fill_index],
            risk_state,
            origin_state_index=origin_state_index,
            sequence_asset_axis_sha256=sequence.axis_id,
            checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
            expected_manifest_sha256=risk_state.manifest_sha256,
        )
        action_anchor = repaired.projected_weights
        feasible_signal = apply_m03r_v11_residual_operator(
            distribution.selected_mean.squeeze(0).to(dtype=weights.dtype),
            operator,
        )
        qualified_fill_mask = fill_mask & operator.qualified_asset_mask.to(
            device=fill_mask.device
        ).unsqueeze(0)
        risk_row = risk_state.origin_state_indices.index(origin_state_index)
        proposal = build_cost_aware_active_proposal_v3(
            action_anchor,
            sequence.benchmark_weights[fill_index],
            feasible_signal.residual.unsqueeze(0),
            distribution.selected_scale.to(dtype=weights.dtype),
            torch.full_like(weights, 10.0e-4),
            torch.zeros_like(weights),
            torch.zeros(1, device=weights.device, dtype=weights.dtype),
            action_anchor > 1.0e-12,
            qualified_fill_mask,
            sequence.risk_asset_caps[fill_index],
            torch.ones(1, device=weights.device, dtype=weights.dtype),
            risk_state.covariance_factor[risk_row].unsqueeze(0).to(weights.dtype),
            risk_state.specific_variance[risk_row].unsqueeze(0).to(weights.dtype),
            selected_horizon_sessions=selected_horizon_sessions,
            research_contract_sha256=M03R_V11_PROTOCOL_SHA256,
            cash_index=risk_state.cash_index,
        )
        projection = project_m03r_v9_active_book(
            proposal.requested_weights,
            sequence.benchmark_weights[fill_index],
            fill_mask,
            sequence.risk_asset_caps[fill_index],
            sequence.risk_gross_max[fill_index],
            risk_state,
            origin_state_index=origin_state_index,
            sequence_asset_axis_sha256=sequence.axis_id,
            checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
            expected_manifest_sha256=risk_state.manifest_sha256,
        )
        next_weights = projection.projected_weights
        policy_gross.append(gross.squeeze(0))
        policy_turnover.append(
            (0.5 * (next_weights - pretrade).abs().sum(-1)).squeeze(0)
        )
        requested_weights.append(proposal.requested_weights.squeeze(0))
        projected_weights.append(next_weights.squeeze(0))
        projection_retention.append(
            projection.requested_to_executed_retention.squeeze(0)
        )
        weights = next_weights

    arrays = tuple(
        value.detach().to(device="cpu", dtype=torch.float64).clone()
        for value in (
            torch.stack(policy_gross),
            benchmark_gross_returns,
            torch.stack(policy_turnover),
            benchmark_one_way_turnover,
            torch.stack(requested_weights),
            torch.stack(projected_weights),
            torch.stack(projection_retention),
        )
    )
    provisional = M03RV11SimpleSleeveTrace(
        setting_index=setting_index,
        setting_id=M03R_V11_SETTING_IDS[setting_index],
        fold_index=fold_index,
        selected_horizon_sessions=selected_horizon_sessions,
        checkpoint_file_sha256=checkpoint_file_sha256,
        checkpoint_model_state_sha256=checkpoint_model_state_sha256,
        source_receipt_sha256=source_receipt_sha256,
        asset_axis_sha256=checkpoint_asset_axis_sha256,
        risk_state_sha256=risk_state.state_sha256,
        risk_manifest_sha256=risk_state.manifest_sha256,
        signal_operator_receipt_sha256=tuple(
            row.receipt_sha256 for row in signal_operators
        ),
        state_start_index=state_start_index,
        policy_gross_returns=arrays[0],
        benchmark_gross_returns=arrays[1],
        policy_one_way_turnover=arrays[2],
        benchmark_one_way_turnover=arrays[3],
        requested_weight_trace=arrays[4],
        projected_weight_trace=arrays[5],
        requested_to_executed_retention=arrays[6],
        array_sha256=tuple(_tensor_sha256(value) for value in arrays),
        trace_sha256="0" * 64,
    )
    result = replace(
        provisional,
        trace_sha256=_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


__all__ = [
    "M03R_V11_SIMPLE_SLEEVE_TRACE_SCHEMA",
    "M03RV11RuntimeError",
    "M03RV11SimpleSleeveTrace",
    "run_m03r_v11_simple_sleeve",
]
