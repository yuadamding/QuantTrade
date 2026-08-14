"""Target-blind replay controls for the M03R-v11 a15 inference audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import torch

from rl_quant.execution.cost_aware_active_policy_v3 import (
    build_cost_aware_active_proposal_v3,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
    M03R_V11_A15_INFERENCE_AUDIT_SPEC,
    M03RV11A15AuditVariant,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
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

M03R_V11_A15_AUDIT_REPLAY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-a15-inference-audit-replay-v1"
)


class M03RV11A15InferenceAuditRuntimeError(ValueError):
    """The target-blind audit replay or its exact lineage drifted."""


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


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV11A15InferenceAuditRuntimeError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _audit_array(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64).clone()


@dataclass(frozen=True, slots=True)
class M03RV11A15AuditReplayTrace:
    setting_index: int
    setting_id: str
    fold_index: int
    selected_horizon_sessions: int
    variant_id: str
    signal_transform: str
    maximum_incremental_one_way_turnover: float
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
    pretrade_weight_trace: torch.Tensor
    anchor_weight_trace: torch.Tensor
    requested_weight_trace: torch.Tensor
    projected_weight_trace: torch.Tensor
    feasible_signal_trace: torch.Tensor
    selected_scale_trace: torch.Tensor
    entry_probability_trace: torch.Tensor
    exit_probability_trace: torch.Tensor
    buy_gate_trace: torch.Tensor
    sell_gate_trace: torch.Tensor
    requested_incremental_turnover: torch.Tensor
    allowed_incremental_turnover: torch.Tensor
    requested_to_executed_retention: torch.Tensor
    carry_active_return: torch.Tensor
    anchor_repair_active_return: torch.Tensor
    alpha_signal_active_return: torch.Tensor
    array_sha256: tuple[str, ...]
    trace_sha256: str
    targets_or_outcomes_used_to_construct_actions: bool = False
    economic_optimizer_updates: int = 0
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
    schema: str = M03R_V11_A15_AUDIT_REPLAY_SCHEMA

    @property
    def arrays(self) -> tuple[torch.Tensor, ...]:
        return (
            self.policy_gross_returns,
            self.benchmark_gross_returns,
            self.policy_one_way_turnover,
            self.benchmark_one_way_turnover,
            self.pretrade_weight_trace,
            self.anchor_weight_trace,
            self.requested_weight_trace,
            self.projected_weight_trace,
            self.feasible_signal_trace,
            self.selected_scale_trace,
            self.entry_probability_trace,
            self.exit_probability_trace,
            self.buy_gate_trace,
            self.sell_gate_trace,
            self.requested_incremental_turnover,
            self.allowed_incremental_turnover,
            self.requested_to_executed_retention,
            self.carry_active_return,
            self.anchor_repair_active_return,
            self.alpha_signal_active_return,
        )

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "setting_index": self.setting_index,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "selected_horizon_sessions": self.selected_horizon_sessions,
            "variant_id": self.variant_id,
            "signal_transform": self.signal_transform,
            "maximum_incremental_one_way_turnover": (
                self.maximum_incremental_one_way_turnover
            ),
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "checkpoint_model_state_sha256": self.checkpoint_model_state_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "asset_axis_sha256": self.asset_axis_sha256,
            "risk_state_sha256": self.risk_state_sha256,
            "risk_manifest_sha256": self.risk_manifest_sha256,
            "signal_operator_receipt_sha256": self.signal_operator_receipt_sha256,
            "state_start_index": self.state_start_index,
            "array_sha256": self.array_sha256,
            "targets_or_outcomes_used_to_construct_actions": (
                self.targets_or_outcomes_used_to_construct_actions
            ),
            "economic_optimizer_updates": self.economic_optimizer_updates,
            "outer_2026_accessed": self.outer_2026_accessed,
            "development_only": True,
            "posthoc_exploratory": True,
            "reportable": False,
            "promotion_eligible": False,
            "economic_generation_may_be_minted": False,
        }

    def validate(self) -> None:
        variant = M03RV11A15AuditVariant(
            self.variant_id,
            self.signal_transform,  # type: ignore[arg-type]
            self.maximum_incremental_one_way_turnover,
        )
        variant.validate()
        transitions = self.policy_gross_returns.numel()
        asset_shape = self.pretrade_weight_trace.shape
        one_dimensional = (
            self.policy_gross_returns,
            self.benchmark_gross_returns,
            self.policy_one_way_turnover,
            self.benchmark_one_way_turnover,
            self.requested_incremental_turnover,
            self.allowed_incremental_turnover,
            self.requested_to_executed_retention,
            self.carry_active_return,
            self.anchor_repair_active_return,
            self.alpha_signal_active_return,
        )
        two_dimensional = self.arrays[4:14]
        active = self.policy_gross_returns - self.benchmark_gross_returns
        attributed = (
            self.carry_active_return
            + self.anchor_repair_active_return
            + self.alpha_signal_active_return
        )
        if (
            self.setting_index not in (0, 1)
            or self.setting_id != M03R_V11_SETTING_IDS[self.setting_index]
            or self.fold_index not in range(6)
            or self.selected_horizon_sessions not in (21, 30)
            or transitions < 2
            or len(self.signal_operator_receipt_sha256) != transitions
            or len(set(self.signal_operator_receipt_sha256)) != transitions
            or any(
                value.ndim != 1
                or value.shape != self.policy_gross_returns.shape
                or not value.is_floating_point()
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
                for value in one_dimensional
            )
            or self.pretrade_weight_trace.ndim != 2
            or self.pretrade_weight_trace.shape[0] != transitions
            or any(
                value.shape != asset_shape
                or not value.is_floating_point()
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
                for value in two_dimensional
            )
            or bool((self.policy_one_way_turnover < 0.0).any())
            or bool((self.benchmark_one_way_turnover < 0.0).any())
            or bool((self.selected_scale_trace <= 0.0).any())
            or bool((self.entry_probability_trace < 0.0).any())
            or bool((self.entry_probability_trace > 1.0).any())
            or bool((self.exit_probability_trace < 0.0).any())
            or bool((self.exit_probability_trace > 1.0).any())
            or bool((self.buy_gate_trace < 0.0).any())
            or bool((self.buy_gate_trace > 1.0).any())
            or bool((self.sell_gate_trace < 0.0).any())
            or bool((self.sell_gate_trace > 1.0).any())
            or bool((self.requested_incremental_turnover < 0.0).any())
            or bool((self.allowed_incremental_turnover < 0.0).any())
            or bool(
                (
                    self.requested_incremental_turnover
                    - self.allowed_incremental_turnover
                    > 2.0e-7
                ).any()
            )
            or bool((self.requested_to_executed_retention < 0.0).any())
            or bool((self.requested_to_executed_retention > 1.0 + 1.0e-8).any())
            or not bool(torch.allclose(active, attributed, atol=2.0e-7, rtol=2.0e-7))
            or self.array_sha256
            != tuple(_tensor_sha256(value) for value in self.arrays)
            or self.targets_or_outcomes_used_to_construct_actions
            or self.economic_optimizer_updates != 0
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256
            or self.schema != M03R_V11_A15_AUDIT_REPLAY_SCHEMA
            or self.trace_sha256 != _sha256(self.unsigned_payload())
        ):
            raise M03RV11A15InferenceAuditRuntimeError(
                "a15 inference-audit replay drifted"
            )
        for name, value in (
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("checkpoint_model_state_sha256", self.checkpoint_model_state_sha256),
            ("source_receipt_sha256", self.source_receipt_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
            ("risk_state_sha256", self.risk_state_sha256),
            ("risk_manifest_sha256", self.risk_manifest_sha256),
            *(
                ("signal_operator_receipt_sha256", value)
                for value in self.signal_operator_receipt_sha256
            ),
        ):
            _digest(name, value)


def _transform_signal(
    signal: torch.Tensor,
    scale: torch.Tensor,
    operator: M03RV11ResidualOperator,
    variant: M03RV11A15AuditVariant,
    *,
    setting_index: int,
    fold_index: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    variant.validate()
    if variant.signal_transform == "original":
        return signal, scale
    if variant.signal_transform == "zero":
        return torch.zeros_like(signal), scale
    if variant.signal_transform == "sign-flipped":
        return -signal, scale
    qualified = torch.nonzero(
        operator.qualified_asset_mask.to(device=signal.device), as_tuple=False
    ).flatten()
    seed = int(
        _sha256(
            {
                "protocol_sha256": M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
                "shuffle_seed": M03R_V11_A15_INFERENCE_AUDIT_SPEC.shuffle_seed,
                "setting_index": setting_index,
                "fold_index": fold_index,
                "horizon": horizon,
                "origin_state_index": operator.origin_state_index,
                "variant_id": variant.variant_id,
            }
        )[:16],
        16,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(qualified.numel(), generator=generator).to(
        device=signal.device
    )
    shuffled = torch.zeros_like(signal)
    shuffled_scale = scale.clone()
    selected_signal = signal.index_select(0, qualified)
    selected_scale = scale.index_select(0, qualified)
    shuffled[qualified] = selected_signal.index_select(0, permutation)
    shuffled_scale[qualified] = selected_scale.index_select(0, permutation)
    feasible = apply_m03r_v11_residual_operator(shuffled, operator).residual
    return feasible, shuffled_scale


def run_m03r_v11_a15_inference_audit_replay(
    sequence: Hold30Sequence,
    alpha_distributions: tuple[M03RV9AlphaDistribution, ...],
    signal_operators: tuple[M03RV11ResidualOperator, ...],
    risk_state: M03RV9DeviceRiskState,
    variant: M03RV11A15AuditVariant,
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
) -> M03RV11A15AuditReplayTrace:
    """Replay one exact checkpoint without using outcomes to form actions."""

    variant.validate()
    transitions = sequence.asset_returns.shape[0]
    if (
        setting_index not in (0, 1)
        or fold_index not in range(6)
        or selected_horizon_sessions not in (21, 30)
        or sequence.asset_returns.shape[1] != 1
        or len(alpha_distributions) != transitions
        or len(signal_operators) != transitions
        or sequence.axis_id != risk_state.asset_axis_sha256
        or checkpoint_asset_axis_sha256 != risk_state.asset_axis_sha256
        or tuple(risk_state.origin_state_indices)
        != tuple(state_start_index + index for index in range(transitions))
        or tuple(benchmark_gross_returns.shape) != (transitions,)
        or tuple(benchmark_one_way_turnover.shape) != (transitions,)
    ):
        raise M03RV11A15InferenceAuditRuntimeError(
            "a15 replay axes or chronology drifted"
        )
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
    pretrade_rows: list[torch.Tensor] = []
    anchor_rows: list[torch.Tensor] = []
    requested_rows: list[torch.Tensor] = []
    projected_rows: list[torch.Tensor] = []
    signal_rows: list[torch.Tensor] = []
    scale_rows: list[torch.Tensor] = []
    entry_rows: list[torch.Tensor] = []
    exit_rows: list[torch.Tensor] = []
    buy_gate_rows: list[torch.Tensor] = []
    sell_gate_rows: list[torch.Tensor] = []
    requested_turnover_rows: list[torch.Tensor] = []
    allowed_turnover_rows: list[torch.Tensor] = []
    retention_rows: list[torch.Tensor] = []

    for transition_index, (distribution, operator) in enumerate(
        zip(alpha_distributions, signal_operators, strict=True)
    ):
        distribution.validate()
        operator.validate()
        origin_state_index = state_start_index + transition_index
        if (
            distribution.selected_horizon_sessions != selected_horizon_sessions
            or operator.origin_state_index != origin_state_index
            or operator.asset_axis_sha256 != sequence.axis_id
            or operator.source_exposure_receipt_sha256
            != risk_state.source_exposure_receipt_sha256
        ):
            raise M03RV11A15InferenceAuditRuntimeError(
                "a15 distribution or residual operator drifted"
            )

        returns = sequence.asset_returns[transition_index]
        gross = (weights * returns).sum(-1)
        if bool((gross <= -1.0).any()):
            raise M03RV11A15InferenceAuditRuntimeError("a15 replay lost all value")
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
        anchor = repaired.projected_weights
        feasible = apply_m03r_v11_residual_operator(
            distribution.selected_mean.squeeze(0).to(dtype=weights.dtype),
            operator,
        ).residual
        scale = distribution.selected_scale.squeeze(0).to(dtype=weights.dtype)
        feasible, scale = _transform_signal(
            feasible,
            scale,
            operator,
            variant,
            setting_index=setting_index,
            fold_index=fold_index,
            horizon=selected_horizon_sessions,
        )
        qualified_fill_mask = fill_mask & operator.qualified_asset_mask.to(
            device=fill_mask.device
        ).unsqueeze(0)
        risk_row = risk_state.origin_state_indices.index(origin_state_index)
        proposal = build_cost_aware_active_proposal_v3(
            anchor,
            sequence.benchmark_weights[fill_index],
            feasible.unsqueeze(0),
            scale.unsqueeze(0),
            torch.full_like(weights, 10.0e-4),
            torch.zeros_like(weights),
            torch.zeros(1, device=weights.device, dtype=weights.dtype),
            anchor > 1.0e-12,
            qualified_fill_mask,
            sequence.risk_asset_caps[fill_index],
            torch.ones(1, device=weights.device, dtype=weights.dtype),
            risk_state.covariance_factor[risk_row].unsqueeze(0).to(weights.dtype),
            risk_state.specific_variance[risk_row].unsqueeze(0).to(weights.dtype),
            selected_horizon_sessions=selected_horizon_sessions,
            research_contract_sha256=(M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256),
            maximum_incremental_one_way_turnover=(
                variant.maximum_incremental_one_way_turnover
            ),
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
        pretrade_rows.append(pretrade.squeeze(0))
        anchor_rows.append(anchor.squeeze(0))
        requested_rows.append(proposal.requested_weights.squeeze(0))
        projected_rows.append(next_weights.squeeze(0))
        signal_rows.append(feasible)
        scale_rows.append(scale)
        entry_rows.append(proposal.entry_probability.squeeze(0))
        exit_rows.append(proposal.exit_probability.squeeze(0))
        buy_gate_rows.append(proposal.buy_gate.squeeze(0))
        sell_gate_rows.append(proposal.sell_gate.squeeze(0))
        requested_turnover_rows.append(
            proposal.requested_incremental_one_way_turnover.squeeze(0)
        )
        allowed_turnover_rows.append(
            proposal.allowed_incremental_one_way_turnover.squeeze(0)
        )
        retention_rows.append(projection.requested_to_executed_retention.squeeze(0))
        weights = next_weights

    policy = torch.stack(policy_gross)
    benchmark = benchmark_gross_returns.to(device=policy.device, dtype=policy.dtype)
    pretrade_trace = torch.stack(pretrade_rows)
    anchor_trace = torch.stack(anchor_rows)
    projected_trace = torch.stack(projected_rows)
    carry = torch.zeros_like(policy)
    anchor_attribution = torch.zeros_like(policy)
    signal_attribution = torch.zeros_like(policy)
    carry[0] = policy[0] - benchmark[0]
    if transitions > 1:
        next_returns = sequence.asset_returns[1:, 0]
        carry[1:] = (pretrade_trace[:-1] * next_returns).sum(-1) - benchmark[1:]
        anchor_attribution[1:] = (
            (anchor_trace[:-1] - pretrade_trace[:-1]) * next_returns
        ).sum(-1)
        signal_attribution[1:] = (
            (projected_trace[:-1] - anchor_trace[:-1]) * next_returns
        ).sum(-1)

    arrays = tuple(
        _audit_array(value)
        for value in (
            policy,
            benchmark,
            torch.stack(policy_turnover),
            benchmark_one_way_turnover,
            pretrade_trace,
            anchor_trace,
            torch.stack(requested_rows),
            projected_trace,
            torch.stack(signal_rows),
            torch.stack(scale_rows),
            torch.stack(entry_rows),
            torch.stack(exit_rows),
            torch.stack(buy_gate_rows),
            torch.stack(sell_gate_rows),
            torch.stack(requested_turnover_rows),
            torch.stack(allowed_turnover_rows),
            torch.stack(retention_rows),
            carry,
            anchor_attribution,
            signal_attribution,
        )
    )
    provisional = M03RV11A15AuditReplayTrace(
        setting_index=setting_index,
        setting_id=M03R_V11_SETTING_IDS[setting_index],
        fold_index=fold_index,
        selected_horizon_sessions=selected_horizon_sessions,
        variant_id=variant.variant_id,
        signal_transform=variant.signal_transform,
        maximum_incremental_one_way_turnover=(
            variant.maximum_incremental_one_way_turnover
        ),
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
        pretrade_weight_trace=arrays[4],
        anchor_weight_trace=arrays[5],
        requested_weight_trace=arrays[6],
        projected_weight_trace=arrays[7],
        feasible_signal_trace=arrays[8],
        selected_scale_trace=arrays[9],
        entry_probability_trace=arrays[10],
        exit_probability_trace=arrays[11],
        buy_gate_trace=arrays[12],
        sell_gate_trace=arrays[13],
        requested_incremental_turnover=arrays[14],
        allowed_incremental_turnover=arrays[15],
        requested_to_executed_retention=arrays[16],
        carry_active_return=arrays[17],
        anchor_repair_active_return=arrays[18],
        alpha_signal_active_return=arrays[19],
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
    "M03R_V11_A15_AUDIT_REPLAY_SCHEMA",
    "M03RV11A15AuditReplayTrace",
    "M03RV11A15InferenceAuditRuntimeError",
    "run_m03r_v11_a15_inference_audit_replay",
]
