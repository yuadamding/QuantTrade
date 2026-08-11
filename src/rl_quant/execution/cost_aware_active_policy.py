"""Differentiable cost-aware incremental active-policy construction.

This module is generation-local groundwork for the TOP2000 M03R-v8
development panel.  It deliberately does not modify the immutable v7 action
builder.  Learned exits and any explicit de-risking are applied upstream to a
``hazard_anchor_weights`` book.  This surface controls only the incremental
move away from that anchor, so zero confidence cannot undo an already-approved
exit or force an existing position back to the benchmark.

The proposal is not a substitute for the governed factor, beta, and tracking-
error projector.  It prevents the earlier near-maximal replacement request by
bounding the proposal's one-way turnover before that projector runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_quant.execution.hold30 import capped_waterfill

M03R_V8_COST_AWARE_ACTIVE_POLICY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-cost-aware-active-proposal-v1"
)


class M03RV8CostAwareActivePolicyError(ValueError):
    """The incremental active proposal cannot be constructed safely."""


def _matrix(
    name: str,
    value: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV8CostAwareActivePolicyError(
            f"{name} must be a finite floating [batch,asset] tensor"
        )
    if reference is not None and (
        value.shape != reference.shape
        or value.device != reference.device
        or value.dtype != reference.dtype
    ):
        raise M03RV8CostAwareActivePolicyError(
            f"{name} must match the reference tensor exactly"
        )
    return value


def _vector(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (reference.shape[0],)
        or value.device != reference.device
        or value.dtype != reference.dtype
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV8CostAwareActivePolicyError(
            f"{name} must be a finite floating [batch] tensor matching the book"
        )
    return value


def _boolean_matrix(
    name: str, value: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.shape != reference.shape
        or value.device != reference.device
    ):
        raise M03RV8CostAwareActivePolicyError(
            f"{name} must be boolean [batch,asset] matching the book"
        )
    return value


def _finite_nonnegative_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M03RV8CostAwareActivePolicyError(
            f"{name} must be a finite nonnegative scalar"
        )
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result < 0.0:
        raise M03RV8CostAwareActivePolicyError(
            f"{name} must be a finite nonnegative scalar"
        )
    return result


@dataclass(frozen=True, slots=True)
class M03RV8CostAwareActiveProposal:
    """One bounded request from a hazard/de-risk anchor toward active alpha."""

    hazard_anchor_weights: torch.Tensor
    requested_weights: torch.Tensor
    requested_delta: torch.Tensor
    centered_expected_active_alpha: torch.Tensor
    net_edge: torch.Tensor
    no_trade_gate: torch.Tensor
    buy_weights: torch.Tensor
    sell_weights: torch.Tensor
    requested_incremental_one_way_turnover: torch.Tensor
    allowed_incremental_one_way_turnover: torch.Tensor
    signal_confidence: torch.Tensor
    schema: str = M03R_V8_COST_AWARE_ACTIVE_POLICY_SCHEMA

    def validate(self) -> None:
        reference = _matrix("hazard_anchor_weights", self.hazard_anchor_weights)
        for name in (
            "requested_weights",
            "requested_delta",
            "centered_expected_active_alpha",
            "net_edge",
            "no_trade_gate",
            "buy_weights",
            "sell_weights",
        ):
            _matrix(name, getattr(self, name), reference)
        for name in (
            "requested_incremental_one_way_turnover",
            "allowed_incremental_one_way_turnover",
            "signal_confidence",
        ):
            _vector(name, getattr(self, name), reference)
        if self.schema != M03R_V8_COST_AWARE_ACTIVE_POLICY_SCHEMA:
            raise M03RV8CostAwareActivePolicyError("proposal schema drifted")
        if (
            bool((self.hazard_anchor_weights < -1.0e-8).any())
            or bool((self.requested_weights < -1.0e-8).any())
            or not bool(
                torch.allclose(
                    self.hazard_anchor_weights.sum(-1),
                    torch.ones_like(self.signal_confidence),
                    atol=2.0e-6,
                    rtol=2.0e-6,
                )
            )
            or not bool(
                torch.allclose(
                    self.requested_weights.sum(-1),
                    torch.ones_like(self.signal_confidence),
                    atol=2.0e-6,
                    rtol=2.0e-6,
                )
            )
            or not bool(
                torch.allclose(
                    self.requested_weights,
                    self.hazard_anchor_weights + self.requested_delta,
                    atol=2.0e-7,
                    rtol=2.0e-7,
                )
            )
            or not bool(
                torch.allclose(
                    self.buy_weights.sum(-1),
                    self.sell_weights.sum(-1),
                    atol=2.0e-7,
                    rtol=2.0e-7,
                )
            )
            or bool((self.no_trade_gate < 0.0).any())
            or bool((self.no_trade_gate > 1.0).any())
            or bool((self.requested_incremental_one_way_turnover < 0.0).any())
            or bool(
                (
                    self.requested_incremental_one_way_turnover
                    - self.allowed_incremental_one_way_turnover
                    > 2.0e-7
                ).any()
            )
            or bool((self.signal_confidence < 0.0).any())
            or bool((self.signal_confidence > 1.0).any())
        ):
            raise M03RV8CostAwareActivePolicyError(
                "incremental active proposal failed reconciliation"
            )


def build_cost_aware_active_proposal(
    hazard_anchor_weights: torch.Tensor,
    benchmark_weights: torch.Tensor,
    expected_active_alpha: torch.Tensor,
    uncertainty: torch.Tensor,
    one_way_cost: torch.Tensor,
    signal_confidence: torch.Tensor,
    held_mask: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    *,
    maximum_incremental_one_way_turnover: float = 0.02,
    uncertainty_multiplier: float = 1.0,
    entry_hurdle_multiplier: float = 1.0,
    retention_hurdle_multiplier: float = 0.5,
    gate_temperature: float = 0.002,
    cash_index: int = 0,
) -> M03RV8CostAwareActiveProposal:
    """Build one cost-aware active proposal without suppressing prior exits.

    ``hazard_anchor_weights`` already contains learned exits, forced repairs,
    and explicit benchmark de-risking.  The returned delta is a zero-sum risky
    reallocation.  Its one-way size is bounded by signal confidence and by
    available buy/sell capacity; final factor/beta/TE projection remains a
    separate required stage.
    """

    anchor = _matrix("hazard_anchor_weights", hazard_anchor_weights)
    benchmark = _matrix("benchmark_weights", benchmark_weights, anchor)
    alpha = _matrix("expected_active_alpha", expected_active_alpha, anchor)
    sigma = _matrix("uncertainty", uncertainty, anchor)
    cost = _matrix("one_way_cost", one_way_cost, anchor)
    confidence = _vector("signal_confidence", signal_confidence, anchor)
    held = _boolean_matrix("held_mask", held_mask, anchor)
    tradable = _boolean_matrix("trade_mask", trade_mask, anchor).clone()
    caps = _matrix("risk_asset_caps", risk_asset_caps, anchor)
    batch, assets = anchor.shape
    if not 0 <= cash_index < assets:
        raise M03RV8CostAwareActivePolicyError("cash_index is outside the asset axis")
    if (
        bool((anchor < -1.0e-8).any())
        or bool((benchmark < -1.0e-8).any())
        or bool((sigma < 0.0).any())
        or bool((cost < 0.0).any())
        or bool((caps < 0.0).any())
        or bool((confidence < 0.0).any())
        or bool((confidence > 1.0).any())
        or not bool(
            torch.allclose(
                anchor.sum(-1),
                torch.ones(batch, device=anchor.device, dtype=anchor.dtype),
                atol=2.0e-6,
                rtol=2.0e-6,
            )
        )
        or not bool(
            torch.allclose(
                benchmark.sum(-1),
                torch.ones(batch, device=anchor.device, dtype=anchor.dtype),
                atol=2.0e-6,
                rtol=2.0e-6,
            )
        )
    ):
        raise M03RV8CostAwareActivePolicyError(
            "proposal inputs violate simplex, confidence, cost, or cap bounds"
        )
    maximum_turnover = _finite_nonnegative_scalar(
        "maximum_incremental_one_way_turnover",
        maximum_incremental_one_way_turnover,
    )
    uncertainty_weight = _finite_nonnegative_scalar(
        "uncertainty_multiplier", uncertainty_multiplier
    )
    entry_multiplier = _finite_nonnegative_scalar(
        "entry_hurdle_multiplier", entry_hurdle_multiplier
    )
    retention_multiplier = _finite_nonnegative_scalar(
        "retention_hurdle_multiplier", retention_hurdle_multiplier
    )
    temperature = _finite_nonnegative_scalar("gate_temperature", gate_temperature)
    if temperature <= 0.0:
        raise M03RV8CostAwareActivePolicyError("gate_temperature must be positive")
    if retention_multiplier > entry_multiplier:
        raise M03RV8CostAwareActivePolicyError(
            "retention hurdle cannot exceed the new-entry hurdle"
        )

    tradable[:, cash_index] = False
    risky_benchmark = torch.where(tradable, benchmark.clamp_min(0.0), 0.0)
    benchmark_mass = risky_benchmark.sum(-1, keepdim=True)
    if bool((benchmark_mass <= 0.0).any()):
        raise M03RV8CostAwareActivePolicyError(
            "each batch row needs positive tradable risky benchmark mass"
        )
    benchmark_mean_alpha = (risky_benchmark * alpha).sum(-1, keepdim=True) / (
        benchmark_mass.clamp_min(1.0e-18)
    )
    centered_alpha = torch.where(tradable, alpha - benchmark_mean_alpha, 0.0)
    hurdle_multiplier = torch.where(
        held,
        anchor.new_tensor(retention_multiplier),
        anchor.new_tensor(entry_multiplier),
    )
    hurdle = hurdle_multiplier * (cost + uncertainty_weight * sigma)
    net_edge = torch.where(tradable, centered_alpha.abs() - hurdle, 0.0)
    gate = torch.where(
        tradable,
        torch.sigmoid(net_edge / temperature),
        torch.zeros_like(net_edge),
    )

    positive_strength = torch.relu(centered_alpha) * gate * risky_benchmark
    negative_strength = (
        torch.relu(-centered_alpha) * gate * torch.where(tradable, anchor, 0.0)
    )
    buy_capacity = torch.where(
        tradable,
        (caps - anchor).clamp_min(0.0),
        torch.zeros_like(anchor),
    )
    sell_capacity = torch.where(tradable, anchor.clamp_min(0.0), 0.0)
    count = tradable.sum(-1).clamp_min(1).to(dtype=anchor.dtype)
    signal_mass = torch.where(tradable, centered_alpha.abs() * gate, 0.0).sum(-1) / (
        2.0 * count
    )
    allowed_turnover = confidence * maximum_turnover
    common_mass = torch.minimum(signal_mass, allowed_turnover)
    common_mass = torch.minimum(common_mass, buy_capacity.sum(-1))
    common_mass = torch.minimum(common_mass, sell_capacity.sum(-1))
    has_two_sided_signal = (positive_strength.sum(-1) > 0.0) & (
        negative_strength.sum(-1) > 0.0
    )
    common_mass = torch.where(
        has_two_sided_signal,
        common_mass,
        torch.zeros_like(common_mass),
    )

    buys, buy_mass = capped_waterfill(common_mass, positive_strength, buy_capacity)
    sells, sell_mass = capped_waterfill(
        common_mass,
        negative_strength,
        sell_capacity,
    )
    effective_mass = torch.minimum(buy_mass, sell_mass)
    buy_scale = torch.where(
        buy_mass > 0.0,
        effective_mass / buy_mass.clamp_min(1.0e-18),
        torch.zeros_like(buy_mass),
    )
    sell_scale = torch.where(
        sell_mass > 0.0,
        effective_mass / sell_mass.clamp_min(1.0e-18),
        torch.zeros_like(sell_mass),
    )
    buys = buys * buy_scale.unsqueeze(-1)
    sells = sells * sell_scale.unsqueeze(-1)
    requested_delta = buys - sells
    requested = anchor + requested_delta
    turnover = 0.5 * requested_delta.abs().sum(-1)
    result = M03RV8CostAwareActiveProposal(
        hazard_anchor_weights=anchor,
        requested_weights=requested,
        requested_delta=requested_delta,
        centered_expected_active_alpha=centered_alpha,
        net_edge=net_edge,
        no_trade_gate=gate,
        buy_weights=buys,
        sell_weights=sells,
        requested_incremental_one_way_turnover=turnover,
        allowed_incremental_one_way_turnover=allowed_turnover,
        signal_confidence=confidence,
    )
    result.validate()
    if bool((requested - caps > 2.0e-7).logical_and(tradable).any()):
        raise M03RV8CostAwareActivePolicyError(
            "requested active proposal exceeded a risky asset cap"
        )
    return result


__all__ = [
    "M03R_V8_COST_AWARE_ACTIVE_POLICY_SCHEMA",
    "M03RV8CostAwareActivePolicyError",
    "M03RV8CostAwareActiveProposal",
    "build_cost_aware_active_proposal",
]
