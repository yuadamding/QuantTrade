"""Deterministic nonlearned simple-sleeve runtime for M03R-v9.

The predictive gate must show that one bound 21- or 30-session alpha
distribution survives the exact risk interface before economic/RL training is
allowed.  This runner therefore disables the learned hazard, starts at C1,
carries one chronological book, uses the probability/cost gate with fixed
confidence, projects the signal into the same exposure-null space as the
target, and applies the qualified factor-plus-diagonal safety projector.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from rl_quant.execution.cost_aware_active_policy_v2 import (
    build_cost_aware_active_proposal_v2,
)
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PROTOCOL_SHA256,
    M03RV9HorizonBinding,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9AlphaDistribution,
    M03RV9AlphaHeadIdentity,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    M03RV9DeviceRiskState,
    project_m03r_v9_active_book,
    project_m03r_v9_signal_to_exposure_null,
)

M03R_V9_SIMPLE_SLEEVE_TRACE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-simple-sleeve-trace-v1"
)


class M03RV9RuntimeError(ValueError):
    """The deterministic v9 predictive sleeve is malformed."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV9RuntimeError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV9SimpleSleeveTrace:
    setting_id: str
    fold_index: int
    horizon_binding_sha256: str
    alpha_head_identity_sha256: str
    risk_state_sha256: str
    risk_manifest_sha256: str
    source_receipt_sha256: str
    sequence_asset_axis_sha256: str
    checkpoint_asset_axis_sha256: str
    state_start_index: int
    policy_gross_returns: torch.Tensor
    benchmark_gross_returns: torch.Tensor
    policy_one_way_turnover: torch.Tensor
    benchmark_one_way_turnover: torch.Tensor
    decision_weight_trace: torch.Tensor
    requested_weight_trace: torch.Tensor
    projected_weight_trace: torch.Tensor
    signal_null_retention: torch.Tensor
    requested_to_executed_retention: torch.Tensor
    array_sha256: tuple[str, ...]
    trace_sha256: str
    learned_hazard_enabled: bool = False
    selected_distribution_only: bool = True
    signal_confidence_rule: str = "fixed-one-predictive-isolation-v1"
    simple_sleeve_action_cost_basis_points: float = 10.0
    delayed_fill: bool = True
    economic_optimizer_updates: int = 0
    protocol_sha256: str = M03R_V9_PROTOCOL_SHA256
    schema: str = M03R_V9_SIMPLE_SLEEVE_TRACE_SCHEMA

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "horizon_binding_sha256": self.horizon_binding_sha256,
            "alpha_head_identity_sha256": self.alpha_head_identity_sha256,
            "risk_state_sha256": self.risk_state_sha256,
            "risk_manifest_sha256": self.risk_manifest_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "sequence_asset_axis_sha256": self.sequence_asset_axis_sha256,
            "checkpoint_asset_axis_sha256": self.checkpoint_asset_axis_sha256,
            "state_start_index": self.state_start_index,
            "array_sha256": self.array_sha256,
            "learned_hazard_enabled": self.learned_hazard_enabled,
            "selected_distribution_only": self.selected_distribution_only,
            "signal_confidence_rule": self.signal_confidence_rule,
            "simple_sleeve_action_cost_basis_points": (
                self.simple_sleeve_action_cost_basis_points
            ),
            "delayed_fill": self.delayed_fill,
            "economic_optimizer_updates": self.economic_optimizer_updates,
            "research_only": True,
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
            self.decision_weight_trace,
            self.requested_weight_trace,
            self.projected_weight_trace,
            self.signal_null_retention,
            self.requested_to_executed_retention,
        )
        for name in (
            "horizon_binding_sha256",
            "alpha_head_identity_sha256",
            "risk_state_sha256",
            "risk_manifest_sha256",
            "source_receipt_sha256",
            "sequence_asset_axis_sha256",
            "checkpoint_asset_axis_sha256",
            "trace_sha256",
        ):
            _digest(name, getattr(self, name))
        transitions = self.policy_gross_returns.numel()
        if (
            self.schema != M03R_V9_SIMPLE_SLEEVE_TRACE_SCHEMA
            or self.protocol_sha256 != M03R_V9_PROTOCOL_SHA256
            or not 0 <= self.fold_index < 6
            or transitions < 2
            or self.policy_gross_returns.ndim != 1
            or tuple(self.benchmark_gross_returns.shape) != (transitions,)
            or tuple(self.policy_one_way_turnover.shape) != (transitions,)
            or tuple(self.benchmark_one_way_turnover.shape) != (transitions,)
            or self.decision_weight_trace.ndim != 2
            or self.decision_weight_trace.shape[0] != transitions + 1
            or self.requested_weight_trace.shape != self.decision_weight_trace[1:].shape
            or self.projected_weight_trace.shape != self.requested_weight_trace.shape
            or tuple(self.signal_null_retention.shape) != (transitions,)
            or tuple(self.requested_to_executed_retention.shape) != (transitions,)
            or any(
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
                for value in arrays
            )
            or bool((self.policy_one_way_turnover < 0.0).any())
            or bool((self.benchmark_one_way_turnover < 0.0).any())
            or bool((self.signal_null_retention < 0.0).any())
            or bool((self.signal_null_retention > 1.0 + 1.0e-8).any())
            or bool((self.requested_to_executed_retention < 0.0).any())
            or bool((self.requested_to_executed_retention > 1.0 + 1.0e-8).any())
            or not torch.allclose(
                self.decision_weight_trace.sum(-1),
                torch.ones(transitions + 1, dtype=self.decision_weight_trace.dtype),
                atol=2.0e-6,
                rtol=2.0e-6,
            )
            or self.learned_hazard_enabled
            or not self.selected_distribution_only
            or self.signal_confidence_rule != "fixed-one-predictive-isolation-v1"
            or self.simple_sleeve_action_cost_basis_points != 10.0
            or not self.delayed_fill
            or self.economic_optimizer_updates != 0
            or tuple(_tensor_sha256(value) for value in arrays) != self.array_sha256
            or _canonical_sha256(self.unsigned_payload()) != self.trace_sha256
        ):
            raise M03RV9RuntimeError("simple-sleeve trace drifted")


def run_m03r_v9_simple_sleeve(
    sequence: Hold30Sequence,
    alpha_distributions: tuple[M03RV9AlphaDistribution, ...],
    risk_state: M03RV9DeviceRiskState,
    horizon_binding: M03RV9HorizonBinding,
    alpha_head_identity: M03RV9AlphaHeadIdentity,
    *,
    setting_id: str,
    fold_index: int,
    state_start_index: int,
    checkpoint_asset_axis_sha256: str,
    source_receipt_sha256: str,
    benchmark_gross_returns: torch.Tensor,
    benchmark_one_way_turnover: torch.Tensor,
) -> M03RV9SimpleSleeveTrace:
    """Run one chronological, hazard-free predictor tradeability sleeve."""

    horizon_binding.__post_init__()
    alpha_head_identity.validate()
    transitions = sequence.asset_returns.shape[0]
    if (
        sequence.asset_returns.shape[1] != 1
        or len(alpha_distributions) != transitions
        or sequence.axis_id != risk_state.asset_axis_sha256
        or checkpoint_asset_axis_sha256 != risk_state.asset_axis_sha256
        or alpha_head_identity.selected_alpha_horizon
        != horizon_binding.economic_execution_horizon
        or alpha_head_identity.horizon_binding_sha256 != horizon_binding.receipt_sha256
        or tuple(risk_state.origin_state_indices)
        != tuple(state_start_index + index for index in range(transitions))
        or not isinstance(benchmark_gross_returns, torch.Tensor)
        or tuple(benchmark_gross_returns.shape) != (transitions,)
        or benchmark_gross_returns.dtype != sequence.asset_returns.dtype
        or benchmark_gross_returns.device != sequence.asset_returns.device
        or not bool(torch.isfinite(benchmark_gross_returns).all())
        or not isinstance(benchmark_one_way_turnover, torch.Tensor)
        or tuple(benchmark_one_way_turnover.shape) != (transitions,)
        or benchmark_one_way_turnover.dtype != sequence.asset_returns.dtype
        or benchmark_one_way_turnover.device != sequence.asset_returns.device
        or not bool(torch.isfinite(benchmark_one_way_turnover).all())
        or bool((benchmark_one_way_turnover < 0.0).any())
    ):
        raise M03RV9RuntimeError("simple-sleeve source axes or chronology drifted")
    if risk_state.exposure_loadings.device != sequence.asset_returns.device:
        raise M03RV9RuntimeError("risk state and sequence must share one device")

    weights = sequence.benchmark_weights[0].clone()
    decision_weights = [weights.squeeze(0)]
    requested_weights: list[torch.Tensor] = []
    projected_weights: list[torch.Tensor] = []
    policy_gross: list[torch.Tensor] = []
    policy_turnover: list[torch.Tensor] = []
    signal_retention: list[torch.Tensor] = []
    projection_retention: list[torch.Tensor] = []
    for transition_index, distribution in enumerate(alpha_distributions):
        distribution.validate()
        if (
            distribution.selected_horizon_sessions
            != horizon_binding.economic_execution_horizon
            or distribution.selected_mean.shape != weights.shape
            or distribution.selected_mean.device != weights.device
        ):
            raise M03RV9RuntimeError("alpha distribution and sleeve book drifted")
        origin_state_index = state_start_index + transition_index
        returns = sequence.asset_returns[transition_index]
        gross = (weights * returns).sum(-1)
        if bool((gross <= -1.0).any()):
            raise M03RV9RuntimeError("simple-sleeve portfolio lost all value")
        pretrade = weights * (1.0 + returns) / (1.0 + gross).unsqueeze(-1)
        fill = transition_index + 1
        fill_mask = sequence.fill_membership[fill] & sequence.fill_availability[fill]
        decision_mask = sequence.decision_available[transition_index]
        # Availability/cap/risk drift is repaired before the optional alpha
        # reallocation.  This is a forced fill-time repair, not a learned exit.
        repaired = project_m03r_v9_active_book(
            pretrade,
            sequence.benchmark_weights[fill],
            fill_mask,
            sequence.risk_asset_caps[fill],
            sequence.risk_gross_max[fill],
            risk_state,
            origin_state_index=origin_state_index,
            sequence_asset_axis_sha256=sequence.axis_id,
            checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
            expected_manifest_sha256=risk_state.manifest_sha256,
        )
        action_anchor = repaired.projected_weights
        feasible_signal = project_m03r_v9_signal_to_exposure_null(
            distribution.selected_mean.to(dtype=weights.dtype),
            decision_mask,
            risk_state,
            origin_state_index=origin_state_index,
            sequence_asset_axis_sha256=sequence.axis_id,
            checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
            expected_manifest_sha256=risk_state.manifest_sha256,
        )
        row = risk_state.origin_state_indices.index(origin_state_index)
        proposal = build_cost_aware_active_proposal_v2(
            action_anchor,
            sequence.benchmark_weights[fill],
            feasible_signal.projected_signal,
            distribution.selected_scale.to(dtype=weights.dtype),
            torch.full_like(weights, 10.0e-4),
            torch.zeros_like(weights),
            torch.zeros(1, device=weights.device, dtype=weights.dtype),
            action_anchor > 1.0e-12,
            fill_mask,
            sequence.risk_asset_caps[fill],
            torch.ones(1, device=weights.device, dtype=weights.dtype),
            risk_state.covariance_factor[row].unsqueeze(0).to(weights.dtype),
            risk_state.specific_variance[row].unsqueeze(0).to(weights.dtype),
            cash_index=risk_state.cash_index,
        )
        projection = project_m03r_v9_active_book(
            proposal.requested_weights,
            sequence.benchmark_weights[fill],
            fill_mask,
            sequence.risk_asset_caps[fill],
            sequence.risk_gross_max[fill],
            risk_state,
            origin_state_index=origin_state_index,
            sequence_asset_axis_sha256=sequence.axis_id,
            checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
            expected_manifest_sha256=risk_state.manifest_sha256,
        )
        next_weights = projection.projected_weights
        turnover = 0.5 * (next_weights - pretrade).abs().sum(-1)
        policy_gross.append(gross.squeeze(0))
        policy_turnover.append(turnover.squeeze(0))
        requested_weights.append(proposal.requested_weights.squeeze(0))
        projected_weights.append(next_weights.squeeze(0))
        signal_retention.append(feasible_signal.signal_retention.squeeze(0))
        projection_retention.append(
            projection.requested_to_executed_retention.squeeze(0)
        )
        weights = next_weights
        decision_weights.append(weights.squeeze(0))

    arrays = tuple(
        value.detach().to(device="cpu", dtype=torch.float64).clone()
        for value in (
            torch.stack(policy_gross),
            benchmark_gross_returns,
            torch.stack(policy_turnover),
            benchmark_one_way_turnover,
            torch.stack(decision_weights),
            torch.stack(requested_weights),
            torch.stack(projected_weights),
            torch.stack(signal_retention),
            torch.stack(projection_retention),
        )
    )
    provisional = M03RV9SimpleSleeveTrace(
        setting_id=setting_id,
        fold_index=fold_index,
        horizon_binding_sha256=horizon_binding.receipt_sha256,
        alpha_head_identity_sha256=_canonical_sha256(asdict(alpha_head_identity)),
        risk_state_sha256=risk_state.state_sha256,
        risk_manifest_sha256=risk_state.manifest_sha256,
        source_receipt_sha256=_digest("source_receipt_sha256", source_receipt_sha256),
        sequence_asset_axis_sha256=sequence.axis_id,
        checkpoint_asset_axis_sha256=checkpoint_asset_axis_sha256,
        state_start_index=state_start_index,
        policy_gross_returns=arrays[0],
        benchmark_gross_returns=arrays[1],
        policy_one_way_turnover=arrays[2],
        benchmark_one_way_turnover=arrays[3],
        decision_weight_trace=arrays[4],
        requested_weight_trace=arrays[5],
        projected_weight_trace=arrays[6],
        signal_null_retention=arrays[7],
        requested_to_executed_retention=arrays[8],
        array_sha256=tuple(_tensor_sha256(value) for value in arrays),
        trace_sha256="0" * 64,
    )
    result = replace(
        provisional,
        trace_sha256=_canonical_sha256(provisional.unsigned_payload()),
    )
    result.validate()
    return result


__all__ = [
    "M03R_V9_SIMPLE_SLEEVE_TRACE_SCHEMA",
    "M03RV9RuntimeError",
    "M03RV9SimpleSleeveTrace",
    "run_m03r_v9_simple_sleeve",
]
