"""Cost-aware incremental active allocation for TOP2000 M03R-v9.

V9 first applies forced exits, learned exits, and explicit portfolio de-risking
upstream.  This module receives that post-exit anchor and uses only the portion
of learned-exit proceeds not reserved for de-risking to replace exited risk.
It then performs a fixed eight-iteration proximal reallocation whose cost term
uses the evaluator's one-way turnover definition ``0.5 * abs(delta).sum()``.

The action is intentionally generation-local.  It neither imports nor accepts
the legacy ``alpha_downside_30d`` surface: callers must pass the selected mean
and selected scale from the bound M03R-v9 alpha distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rl_quant.execution.hold30 import capped_waterfill

M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-cost-aware-active-proposal-v2"
)
M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_ITERATIONS = 8


class M03RV9CostAwareActivePolicyError(ValueError):
    """The v9 incremental allocation failed its frozen contract."""


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
        or (
            reference is not None
            and (
                value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
            )
        )
    ):
        raise M03RV9CostAwareActivePolicyError(
            f"{name} must be a finite floating [batch,asset] tensor"
        )
    return value


def _vector(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (reference.shape[0],)
        or value.dtype != reference.dtype
        or value.device != reference.device
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV9CostAwareActivePolicyError(
            f"{name} must be a finite floating [batch] tensor"
        )
    return value


def _boolean_matrix(
    name: str,
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != reference.shape
        or value.dtype != torch.bool
        or value.device != reference.device
    ):
        raise M03RV9CostAwareActivePolicyError(f"{name} must be boolean [batch,asset]")
    return value


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def _factor_diagonal_risk_gradient(
    delta: torch.Tensor,
    covariance_factor: torch.Tensor,
    specific_variance: torch.Tensor,
) -> torch.Tensor:
    factor_exposure = torch.einsum("baf,ba->bf", covariance_factor, delta)
    factor_component = torch.einsum("baf,bf->ba", covariance_factor, factor_exposure)
    return 2.0 * (factor_component + specific_variance * delta)


@dataclass(frozen=True, slots=True)
class M03RV9CostAwareActiveProposalV2:
    post_exit_derisk_anchor_weights: torch.Tensor
    replacement_anchor_weights: torch.Tensor
    requested_weights: torch.Tensor
    requested_delta: torch.Tensor
    replacement_buy_weights: torch.Tensor
    reallocation_buy_weights: torch.Tensor
    reallocation_sell_weights: torch.Tensor
    learned_exit_proceeds: torch.Tensor
    explicit_derisk_amount: torch.Tensor
    replacement_budget: torch.Tensor
    replacement_used: torch.Tensor
    replacement_one_way_turnover: torch.Tensor
    entry_probability: torch.Tensor
    exit_probability: torch.Tensor
    buy_gate: torch.Tensor
    sell_gate: torch.Tensor
    same_step_buy_mask: torch.Tensor
    requested_incremental_one_way_turnover: torch.Tensor
    total_post_exit_one_way_turnover: torch.Tensor
    allowed_incremental_one_way_turnover: torch.Tensor
    proximal_iterations: int
    schema: str = M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_SCHEMA

    def validate(self) -> None:
        anchor = _matrix(
            "post_exit_derisk_anchor_weights", self.post_exit_derisk_anchor_weights
        )
        for name in (
            "replacement_anchor_weights",
            "requested_weights",
            "requested_delta",
            "replacement_buy_weights",
            "reallocation_buy_weights",
            "reallocation_sell_weights",
            "entry_probability",
            "exit_probability",
            "buy_gate",
            "sell_gate",
        ):
            _matrix(name, getattr(self, name), anchor)
        _boolean_matrix("same_step_buy_mask", self.same_step_buy_mask, anchor)
        for name in (
            "learned_exit_proceeds",
            "explicit_derisk_amount",
            "replacement_budget",
            "replacement_used",
            "replacement_one_way_turnover",
            "requested_incremental_one_way_turnover",
            "total_post_exit_one_way_turnover",
            "allowed_incremental_one_way_turnover",
        ):
            _vector(name, getattr(self, name), anchor)
        if (
            self.schema != M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_SCHEMA
            or self.proximal_iterations
            != M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_ITERATIONS
            or bool((anchor < -1.0e-8).any())
            or bool((self.replacement_anchor_weights < -1.0e-8).any())
            or bool((self.requested_weights < -1.0e-8).any())
            or not bool(
                torch.allclose(
                    anchor.sum(-1),
                    torch.ones_like(self.replacement_budget),
                    atol=2.0e-6,
                    rtol=2.0e-6,
                )
            )
            or not bool(
                torch.allclose(
                    self.replacement_anchor_weights.sum(-1),
                    torch.ones_like(self.replacement_budget),
                    atol=2.0e-6,
                    rtol=2.0e-6,
                )
            )
            or not bool(
                torch.allclose(
                    self.requested_weights.sum(-1),
                    torch.ones_like(self.replacement_budget),
                    atol=2.0e-6,
                    rtol=2.0e-6,
                )
            )
            or not bool(
                torch.allclose(
                    self.requested_weights,
                    self.replacement_anchor_weights + self.requested_delta,
                    atol=2.0e-7,
                    rtol=2.0e-7,
                )
            )
            or not bool(
                torch.allclose(
                    self.reallocation_buy_weights.sum(-1),
                    self.reallocation_sell_weights.sum(-1),
                    atol=2.0e-7,
                    rtol=2.0e-7,
                )
            )
            or bool((self.replacement_budget < -1.0e-12).any())
            or bool((self.replacement_used < -1.0e-12).any())
            or bool((self.replacement_used - self.replacement_budget > 2.0e-7).any())
            or not bool(
                torch.allclose(
                    self.replacement_one_way_turnover,
                    self.replacement_used,
                    atol=2.0e-7,
                    rtol=2.0e-7,
                )
            )
            or not bool(
                torch.allclose(
                    self.total_post_exit_one_way_turnover,
                    self.replacement_one_way_turnover
                    + self.requested_incremental_one_way_turnover,
                    atol=2.0e-7,
                    rtol=2.0e-7,
                )
            )
            or bool(
                (
                    self.requested_incremental_one_way_turnover
                    - self.allowed_incremental_one_way_turnover
                    > 2.0e-7
                ).any()
            )
            or bool((self.entry_probability < 0.0).any())
            or bool((self.entry_probability > 1.0).any())
            or bool((self.exit_probability < 0.0).any())
            or bool((self.exit_probability > 1.0).any())
            or bool((self.buy_gate < 0.0).any())
            or bool((self.buy_gate > 1.0).any())
            or bool((self.sell_gate < 0.0).any())
            or bool((self.sell_gate > 1.0).any())
            or bool(
                torch.where(
                    self.same_step_buy_mask,
                    torch.zeros_like(self.replacement_buy_weights),
                    self.replacement_buy_weights + self.reallocation_buy_weights,
                )
                .abs()
                .gt(2.0e-7)
                .any()
            )
        ):
            raise M03RV9CostAwareActivePolicyError(
                "v9 cost-aware proposal failed reconciliation"
            )


def build_cost_aware_active_proposal_v2(
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
    maximum_incremental_one_way_turnover: float = 0.02,
    uncertainty_multiplier: float = 1.0,
    risk_aversion: float = 1.0,
    proximal_step_size: float = 0.25,
    entry_probability_threshold: float = 0.70,
    expansion_probability_threshold: float = 0.65,
    retention_probability_threshold: float = 0.50,
    exit_probability_threshold: float = 0.60,
    same_step_release_tolerance: float = 1.0e-12,
    cash_index: int = 0,
) -> M03RV9CostAwareActiveProposalV2:
    """Redeploy eligible exit proceeds, then optimize a zero-sum reallocation."""

    anchor = _matrix("post_exit_derisk_anchor_weights", post_exit_derisk_anchor_weights)
    benchmark = _matrix("benchmark_weights", benchmark_weights, anchor)
    mean = _matrix("selected_alpha_mean", selected_alpha_mean, anchor)
    scale = _matrix("selected_alpha_scale", selected_alpha_scale, anchor)
    cost = _matrix("one_way_cost", one_way_cost, anchor)
    release = _matrix("learned_release", learned_release, anchor)
    caps = _matrix("risk_asset_caps", risk_asset_caps, anchor)
    confidence = _vector("signal_confidence", signal_confidence, anchor)
    derisk = _vector("explicit_derisk_amount", explicit_derisk_amount, anchor)
    held = _boolean_matrix("held_mask", held_mask, anchor)
    tradable = _boolean_matrix("trade_mask", trade_mask, anchor).clone()
    batch, assets = anchor.shape
    if not 0 <= cash_index < assets:
        raise M03RV9CostAwareActivePolicyError("cash_index is outside the asset axis")
    if (
        not isinstance(covariance_factor, torch.Tensor)
        or covariance_factor.ndim != 3
        or covariance_factor.shape[:2] != anchor.shape
        or covariance_factor.dtype != anchor.dtype
        or covariance_factor.device != anchor.device
        or not bool(torch.isfinite(covariance_factor).all())
        or not isinstance(specific_variance, torch.Tensor)
        or specific_variance.shape != anchor.shape
        or specific_variance.dtype != anchor.dtype
        or specific_variance.device != anchor.device
        or not bool(torch.isfinite(specific_variance).all())
    ):
        raise M03RV9CostAwareActivePolicyError(
            "factor-plus-diagonal risk tensors do not match the asset axis"
        )
    scalar_values = (
        maximum_incremental_one_way_turnover,
        uncertainty_multiplier,
        risk_aversion,
        proximal_step_size,
        entry_probability_threshold,
        expansion_probability_threshold,
        retention_probability_threshold,
        exit_probability_threshold,
        same_step_release_tolerance,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in scalar_values
    ):
        raise M03RV9CostAwareActivePolicyError(
            "v9 cost-aware scalar configuration is invalid"
        )
    if (
        proximal_step_size <= 0.0
        or not 0.0 <= retention_probability_threshold <= 1.0
        or not 0.0 <= expansion_probability_threshold <= 1.0
        or not 0.0 <= entry_probability_threshold <= 1.0
        or not 0.0 <= exit_probability_threshold <= 1.0
        or retention_probability_threshold > expansion_probability_threshold
        or expansion_probability_threshold > entry_probability_threshold
        or bool((anchor < -1.0e-8).any())
        or bool((benchmark < -1.0e-8).any())
        or bool((scale <= 0.0).any())
        or bool((cost < 0.0).any())
        or bool((release < 0.0).any())
        or bool((caps < 0.0).any())
        or bool((specific_variance < 0.0).any())
        or bool((confidence < 0.0).any())
        or bool((confidence > 1.0).any())
        or bool((derisk < 0.0).any())
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
        raise M03RV9CostAwareActivePolicyError(
            "v9 cost-aware inputs violate the frozen probability/risk contract"
        )

    tradable[:, cash_index] = False
    risky_release = release.clone()
    risky_release[:, cash_index] = 0.0
    learned_exit_proceeds = risky_release.sum(-1)
    replacement_budget = (learned_exit_proceeds - derisk).clamp_min(0.0)
    if bool((replacement_budget - anchor[:, cash_index] > 2.0e-7).any()):
        raise M03RV9CostAwareActivePolicyError(
            "post-exit anchor does not contain the replacement cash budget"
        )
    same_step_buy_mask = tradable & (risky_release <= same_step_release_tolerance)
    z_entry = (mean - cost) / scale.clamp_min(torch.finfo(scale.dtype).tiny)
    entry_probability = _normal_cdf(z_entry)
    exit_probability = _normal_cdf(
        (-cost - mean) / scale.clamp_min(torch.finfo(scale.dtype).tiny)
    )
    buy_threshold = torch.where(
        held,
        mean.new_tensor(expansion_probability_threshold),
        mean.new_tensor(entry_probability_threshold),
    )
    buy_gate_mask = same_step_buy_mask & (entry_probability >= buy_threshold)
    sell_gate_mask = (
        tradable
        & held
        & (entry_probability < retention_probability_threshold)
        & (exit_probability >= exit_probability_threshold)
    )
    buy_gate = torch.where(
        buy_gate_mask,
        entry_probability,
        torch.zeros_like(entry_probability),
    )
    sell_gate = torch.where(
        sell_gate_mask,
        exit_probability,
        torch.zeros_like(exit_probability),
    )

    replacement_capacity = torch.where(
        buy_gate_mask,
        (caps - anchor).clamp_min(0.0),
        torch.zeros_like(anchor),
    )
    replacement_strength = torch.where(
        buy_gate_mask,
        torch.relu(mean - cost) * buy_gate,
        torch.zeros_like(mean),
    )
    replacement_buys, replacement_used = capped_waterfill(
        torch.minimum(replacement_budget, replacement_capacity.sum(-1)),
        replacement_strength,
        replacement_capacity,
    )
    replacement_anchor = anchor + replacement_buys
    replacement_anchor[:, cash_index] = (
        replacement_anchor[:, cash_index] - replacement_used
    )

    delta = torch.zeros_like(anchor)
    allowed_turnover = confidence * float(maximum_incremental_one_way_turnover)
    tiny = torch.finfo(anchor.dtype).tiny
    for _ in range(M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_ITERATIONS):
        risk_gradient = _factor_diagonal_risk_gradient(
            delta,
            covariance_factor,
            specific_variance,
        )
        candidate = delta + float(proximal_step_size) * (
            mean - float(risk_aversion) * risk_gradient
        )
        threshold = float(proximal_step_size) * (
            0.5 * cost + float(uncertainty_multiplier) * scale
        )
        candidate = torch.sign(candidate) * torch.relu(candidate.abs() - threshold)
        positive_strength = torch.where(
            buy_gate_mask,
            torch.relu(candidate),
            torch.zeros_like(candidate),
        )
        negative_strength = torch.where(
            sell_gate_mask,
            torch.relu(-candidate),
            torch.zeros_like(candidate),
        )
        buy_capacity = torch.where(
            buy_gate_mask,
            (caps - replacement_anchor).clamp_min(0.0),
            torch.zeros_like(anchor),
        )
        sell_capacity = torch.where(
            sell_gate_mask,
            replacement_anchor.clamp_min(0.0),
            torch.zeros_like(anchor),
        )
        common_mass = torch.minimum(allowed_turnover, buy_capacity.sum(-1))
        common_mass = torch.minimum(common_mass, sell_capacity.sum(-1))
        common_mass = torch.where(
            (positive_strength.sum(-1) > 0.0) & (negative_strength.sum(-1) > 0.0),
            common_mass,
            torch.zeros_like(common_mass),
        )
        buys, buy_mass = capped_waterfill(common_mass, positive_strength, buy_capacity)
        sells, sell_mass = capped_waterfill(
            common_mass, negative_strength, sell_capacity
        )
        effective = torch.minimum(buy_mass, sell_mass)
        buys = buys * torch.where(
            buy_mass > 0.0,
            effective / buy_mass.clamp_min(tiny),
            torch.zeros_like(buy_mass),
        ).unsqueeze(-1)
        sells = sells * torch.where(
            sell_mass > 0.0,
            effective / sell_mass.clamp_min(tiny),
            torch.zeros_like(sell_mass),
        ).unsqueeze(-1)
        delta = buys - sells

    requested = replacement_anchor + delta
    turnover = 0.5 * delta.abs().sum(-1)
    result = M03RV9CostAwareActiveProposalV2(
        post_exit_derisk_anchor_weights=anchor,
        replacement_anchor_weights=replacement_anchor,
        requested_weights=requested,
        requested_delta=delta,
        replacement_buy_weights=replacement_buys,
        reallocation_buy_weights=buys,
        reallocation_sell_weights=sells,
        learned_exit_proceeds=learned_exit_proceeds,
        explicit_derisk_amount=derisk,
        replacement_budget=replacement_budget,
        replacement_used=replacement_used,
        replacement_one_way_turnover=replacement_used,
        entry_probability=entry_probability,
        exit_probability=exit_probability,
        buy_gate=buy_gate,
        sell_gate=sell_gate,
        same_step_buy_mask=same_step_buy_mask,
        requested_incremental_one_way_turnover=turnover,
        total_post_exit_one_way_turnover=replacement_used + turnover,
        allowed_incremental_one_way_turnover=allowed_turnover,
        proximal_iterations=M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_ITERATIONS,
    )
    result.validate()
    if bool((requested - caps > 2.0e-7).logical_and(tradable).any()):
        raise M03RV9CostAwareActivePolicyError(
            "v9 requested weights exceeded a tradable risky asset cap"
        )
    return result


__all__ = [
    "M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_ITERATIONS",
    "M03R_V9_COST_AWARE_ACTIVE_POLICY_V2_SCHEMA",
    "M03RV9CostAwareActivePolicyError",
    "M03RV9CostAwareActiveProposalV2",
    "build_cost_aware_active_proposal_v2",
]
