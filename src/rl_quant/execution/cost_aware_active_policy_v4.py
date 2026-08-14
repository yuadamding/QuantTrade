"""Nonsaturating turnover utilization for M03R-v12 predictive sleeves."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rl_quant.execution.cost_aware_active_policy_v3 import (
    M03RV11CostAwareActiveProposalV3,
    build_cost_aware_active_proposal_v3,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_ELIGIBLE_EXECUTION_HORIZONS,
    M03R_V12_PREDICTIVE_SPEC,
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_TURNOVER_UTILIZATION_RULE,
)

M03R_V12_COST_AWARE_ACTIVE_POLICY_V4_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v12-cost-aware-active-proposal-v4"
)
_V4_BASE_IMPLEMENTATION_HORIZON = 21


class M03RV12CostAwareActivePolicyError(ValueError):
    """The v12 nonsaturating action contract drifted."""


@dataclass(frozen=True, slots=True)
class M03RV12CostAwareActiveProposalV4:
    proposal: M03RV11CostAwareActiveProposalV3
    raw_signal_confidence: torch.Tensor
    turnover_utilization: torch.Tensor
    effective_signal_confidence: torch.Tensor
    selected_horizon_sessions: int
    turnover_utilization_temperature: float
    turnover_utilization_rule: str = M03R_V12_TURNOVER_UTILIZATION_RULE
    schema: str = M03R_V12_COST_AWARE_ACTIVE_POLICY_V4_SCHEMA

    def validate(self) -> None:
        self.proposal.validate()
        reference = self.proposal.allowed_incremental_one_way_turnover
        for name, value in (
            ("raw_signal_confidence", self.raw_signal_confidence),
            ("turnover_utilization", self.turnover_utilization),
            ("effective_signal_confidence", self.effective_signal_confidence),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not bool(torch.isfinite(value).all())
            ):
                raise M03RV12CostAwareActivePolicyError(
                    f"v12 {name} is not a finite batch vector"
                )
        if (
            self.schema != M03R_V12_COST_AWARE_ACTIVE_POLICY_V4_SCHEMA
            or self.turnover_utilization_rule != M03R_V12_TURNOVER_UTILIZATION_RULE
            or self.turnover_utilization_temperature
            != M03R_V12_PREDICTIVE_SPEC.turnover_utilization_temperature
            or self.proposal.research_contract_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.selected_horizon_sessions
            not in M03R_V12_ELIGIBLE_EXECUTION_HORIZONS
            or self.proposal.selected_horizon_sessions
            != _V4_BASE_IMPLEMENTATION_HORIZON
            or bool((self.turnover_utilization < 0.0).any())
            or bool((self.turnover_utilization >= 1.0).any())
            or not bool(
                torch.allclose(
                    self.effective_signal_confidence,
                    self.raw_signal_confidence * self.turnover_utilization,
                    atol=1.0e-12,
                    rtol=1.0e-12,
                )
            )
        ):
            raise M03RV12CostAwareActivePolicyError(
                "v12 turnover utilization failed reconciliation"
            )


def build_cost_aware_active_proposal_v4(
    post_exit_derisk_anchor_weights: torch.Tensor,
    benchmark_weights: torch.Tensor,
    selected_alpha_mean: torch.Tensor,
    selected_alpha_scale: torch.Tensor,
    one_way_cost: torch.Tensor,
    learned_release: torch.Tensor,
    explicit_derisk_amount: torch.Tensor,
    held_mask: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    signal_confidence: torch.Tensor,
    covariance_factor: torch.Tensor,
    specific_variance: torch.Tensor,
    *,
    selected_horizon_sessions: int,
    maximum_incremental_one_way_turnover: float = 0.02,
    cash_index: int = 0,
) -> M03RV12CostAwareActiveProposalV4:
    """Scale the v3 cap by RMS cost-clearing signal strength before allocation."""

    if (
        not isinstance(selected_alpha_mean, torch.Tensor)
        or selected_alpha_mean.ndim != 2
        or selected_alpha_scale.shape != selected_alpha_mean.shape
        or one_way_cost.shape != selected_alpha_mean.shape
        or trade_mask.shape != selected_alpha_mean.shape
        or trade_mask.dtype != torch.bool
        or not 0 <= cash_index < selected_alpha_mean.shape[1]
        or tuple(signal_confidence.shape) != (selected_alpha_mean.shape[0],)
        or selected_horizon_sessions not in M03R_V12_ELIGIBLE_EXECUTION_HORIZONS
    ):
        raise M03RV12CostAwareActivePolicyError(
            "v12 turnover-utilization inputs are not aligned"
        )
    risky = trade_mask.clone()
    risky[:, cash_index] = False
    tiny = torch.finfo(selected_alpha_scale.dtype).tiny
    economic_edge = torch.relu(selected_alpha_mean.abs() - one_way_cost)
    standardized_edge = economic_edge / selected_alpha_scale.clamp_min(tiny)
    count = risky.sum(-1).clamp_min(1).to(selected_alpha_mean.dtype)
    rms = torch.sqrt(
        torch.where(
            risky, standardized_edge.square(), torch.zeros_like(standardized_edge)
        ).sum(-1)
        / count
    )
    temperature = M03R_V12_PREDICTIVE_SPEC.turnover_utilization_temperature
    utilization = torch.tanh(rms / temperature)
    effective_confidence = signal_confidence * utilization
    risk_ratio = selected_horizon_sessions / _V4_BASE_IMPLEMENTATION_HORIZON
    proposal = build_cost_aware_active_proposal_v3(
        post_exit_derisk_anchor_weights,
        benchmark_weights,
        selected_alpha_mean,
        selected_alpha_scale,
        one_way_cost,
        learned_release,
        explicit_derisk_amount,
        held_mask,
        trade_mask,
        risk_asset_caps,
        effective_confidence,
        covariance_factor * math.sqrt(risk_ratio),
        specific_variance * risk_ratio,
        selected_horizon_sessions=_V4_BASE_IMPLEMENTATION_HORIZON,
        research_contract_sha256=M03R_V12_PROTOCOL_SHA256,
        maximum_incremental_one_way_turnover=(maximum_incremental_one_way_turnover),
        cash_index=cash_index,
    )
    result = M03RV12CostAwareActiveProposalV4(
        proposal=proposal,
        raw_signal_confidence=signal_confidence,
        turnover_utilization=utilization,
        effective_signal_confidence=effective_confidence,
        selected_horizon_sessions=selected_horizon_sessions,
        turnover_utilization_temperature=temperature,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V12_COST_AWARE_ACTIVE_POLICY_V4_SCHEMA",
    "M03RV12CostAwareActivePolicyError",
    "M03RV12CostAwareActiveProposalV4",
    "build_cost_aware_active_proposal_v4",
]
