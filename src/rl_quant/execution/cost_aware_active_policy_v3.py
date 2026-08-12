"""Magnitude-preserving, soft-gated active allocation for M03R-v11."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rl_quant.execution.hold30 import capped_waterfill

M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v11-cost-aware-active-proposal-v3"
)
M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_ITERATIONS = 8
M03R_V11_COST_AWARE_ELIGIBLE_HORIZONS = (21, 30)


class M03RV11CostAwareActivePolicyError(ValueError):
    """The v11 incremental active proposal failed its contract."""


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
        raise M03RV11CostAwareActivePolicyError(
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
        raise M03RV11CostAwareActivePolicyError(
            f"{name} must be a finite floating [batch] tensor"
        )
    return value


def _boolean(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != reference.shape
        or value.dtype != torch.bool
        or value.device != reference.device
    ):
        raise M03RV11CostAwareActivePolicyError(f"{name} must be boolean [batch,asset]")
    return value


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def _factor_diagonal_risk_gradient(
    delta: torch.Tensor,
    covariance_factor: torch.Tensor,
    specific_variance: torch.Tensor,
    *,
    selected_horizon_sessions: int,
) -> torch.Tensor:
    factor_exposure = torch.einsum("baf,ba->bf", covariance_factor, delta)
    factor_component = torch.einsum("baf,bf->ba", covariance_factor, factor_exposure)
    daily_gradient = 2.0 * (factor_component + specific_variance * delta)
    return float(selected_horizon_sessions) * daily_gradient


@dataclass(frozen=True, slots=True)
class M03RV11CostAwareActiveProposalV3:
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
    desired_replacement_mass: torch.Tensor
    replacement_used: torch.Tensor
    desired_buy_mass: torch.Tensor
    desired_sell_mass: torch.Tensor
    entry_probability: torch.Tensor
    exit_probability: torch.Tensor
    buy_gate: torch.Tensor
    sell_gate: torch.Tensor
    same_step_buy_mask: torch.Tensor
    requested_incremental_one_way_turnover: torch.Tensor
    allowed_incremental_one_way_turnover: torch.Tensor
    selected_horizon_sessions: int
    research_contract_sha256: str
    proximal_iterations: int = M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_ITERATIONS
    uncertainty_mechanism: str = "soft-calibrated-probability-gate-only"
    covariance_units: str = "daily-factor-plus-diagonal-scaled-by-selected-horizon"
    schema: str = M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_SCHEMA

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
        _boolean("same_step_buy_mask", self.same_step_buy_mask, anchor)
        for name in (
            "learned_exit_proceeds",
            "explicit_derisk_amount",
            "replacement_budget",
            "desired_replacement_mass",
            "replacement_used",
            "desired_buy_mass",
            "desired_sell_mass",
            "requested_incremental_one_way_turnover",
            "allowed_incremental_one_way_turnover",
        ):
            _vector(name, getattr(self, name), anchor)
        if (
            self.schema != M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_SCHEMA
            or self.proximal_iterations
            != M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_ITERATIONS
            or self.selected_horizon_sessions
            not in M03R_V11_COST_AWARE_ELIGIBLE_HORIZONS
            or self.uncertainty_mechanism != "soft-calibrated-probability-gate-only"
            or self.covariance_units
            != "daily-factor-plus-diagonal-scaled-by-selected-horizon"
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
            or bool((self.replacement_used - self.replacement_budget > 2.0e-7).any())
            or bool(
                (self.replacement_used - self.desired_replacement_mass > 2.0e-7).any()
            )
            or bool(
                (
                    self.requested_incremental_one_way_turnover
                    - self.allowed_incremental_one_way_turnover
                    > 2.0e-7
                ).any()
            )
            or bool(
                (
                    self.requested_incremental_one_way_turnover
                    - torch.minimum(self.desired_buy_mass, self.desired_sell_mass)
                    > 2.0e-7
                ).any()
            )
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
            raise M03RV11CostAwareActivePolicyError(
                "v11 cost-aware proposal failed reconciliation"
            )
        if len(self.research_contract_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.research_contract_sha256
        ):
            raise M03RV11CostAwareActivePolicyError(
                "v11 cost-aware research contract is not a SHA-256"
            )


def build_cost_aware_active_proposal_v3(
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
    research_contract_sha256: str,
    maximum_incremental_one_way_turnover: float = 0.02,
    risk_aversion: float = 1.0,
    proximal_step_size: float = 0.25,
    probability_temperature: float = 0.05,
    entry_probability_threshold: float = 0.70,
    expansion_probability_threshold: float = 0.65,
    retention_probability_threshold: float = 0.50,
    exit_probability_threshold: float = 0.60,
    same_step_release_tolerance: float = 1.0e-12,
    cash_index: int = 0,
) -> M03RV11CostAwareActiveProposalV3:
    """Use one soft probability gate and preserve proximal candidate magnitude."""

    anchor = _matrix("post_exit_derisk_anchor_weights", post_exit_derisk_anchor_weights)
    benchmark = _matrix("benchmark_weights", benchmark_weights, anchor)
    mean = _matrix("selected_alpha_mean", selected_alpha_mean, anchor)
    scale = _matrix("selected_alpha_scale", selected_alpha_scale, anchor)
    cost = _matrix("one_way_cost", one_way_cost, anchor)
    release = _matrix("learned_release", learned_release, anchor)
    caps = _matrix("risk_asset_caps", risk_asset_caps, anchor)
    confidence = _vector("signal_confidence", signal_confidence, anchor)
    derisk = _vector("explicit_derisk_amount", explicit_derisk_amount, anchor)
    held = _boolean("held_mask", held_mask, anchor)
    tradable = _boolean("trade_mask", trade_mask, anchor).clone()
    batch, assets = anchor.shape
    if (
        not 0 <= cash_index < assets
        or selected_horizon_sessions not in M03R_V11_COST_AWARE_ELIGIBLE_HORIZONS
        or not isinstance(covariance_factor, torch.Tensor)
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
        raise M03RV11CostAwareActivePolicyError(
            "v11 asset, horizon, or factor-diagonal risk geometry drifted"
        )
    scalars = (
        maximum_incremental_one_way_turnover,
        risk_aversion,
        proximal_step_size,
        probability_temperature,
        entry_probability_threshold,
        expansion_probability_threshold,
        retention_probability_threshold,
        exit_probability_threshold,
        same_step_release_tolerance,
    )
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in scalars
        )
        or proximal_step_size <= 0.0
        or probability_temperature <= 0.0
        or retention_probability_threshold > expansion_probability_threshold
        or expansion_probability_threshold > entry_probability_threshold
        or bool((anchor < -1.0e-8).any())
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
                torch.ones(batch, dtype=anchor.dtype, device=anchor.device),
            )
        )
        or not bool(
            torch.allclose(
                benchmark.sum(-1),
                torch.ones(batch, dtype=anchor.dtype, device=anchor.device),
            )
        )
    ):
        raise M03RV11CostAwareActivePolicyError("v11 cost-aware inputs drifted")

    tradable[:, cash_index] = False
    risky_release = release.clone()
    risky_release[:, cash_index] = 0.0
    learned_exit_proceeds = risky_release.sum(-1)
    replacement_budget = (learned_exit_proceeds - derisk).clamp_min(0.0)
    if bool((replacement_budget - anchor[:, cash_index] > 2.0e-7).any()):
        raise M03RV11CostAwareActivePolicyError(
            "v11 post-exit anchor lacks its replacement cash"
        )
    same_step_buy_mask = tradable & (risky_release <= same_step_release_tolerance)
    tiny = torch.finfo(scale.dtype).tiny
    entry_probability = _normal_cdf((mean - cost) / scale.clamp_min(tiny))
    exit_probability = _normal_cdf((-cost - mean) / scale.clamp_min(tiny))
    buy_threshold = torch.where(
        held,
        mean.new_tensor(expansion_probability_threshold),
        mean.new_tensor(entry_probability_threshold),
    )
    buy_gate = torch.sigmoid(
        (entry_probability - buy_threshold) / float(probability_temperature)
    ) * same_step_buy_mask.to(mean.dtype)
    sell_threshold_gate = torch.sigmoid(
        (mean.new_tensor(retention_probability_threshold) - entry_probability)
        / float(probability_temperature)
    )
    exit_gate = torch.sigmoid(
        (exit_probability - float(exit_probability_threshold))
        / float(probability_temperature)
    )
    sell_gate = sell_threshold_gate * exit_gate * (tradable & held).to(mean.dtype)

    replacement_capacity = (caps - anchor).clamp_min(0.0) * buy_gate
    replacement_strength = torch.relu(mean - cost) * buy_gate
    desired_replacement_mass = replacement_strength.sum(-1)
    replacement_mass = torch.minimum(replacement_budget, desired_replacement_mass)
    replacement_mass = torch.minimum(replacement_mass, replacement_capacity.sum(-1))
    replacement_buys, replacement_used = capped_waterfill(
        replacement_mass, replacement_strength, replacement_capacity
    )
    replacement_anchor = anchor + replacement_buys
    replacement_anchor[:, cash_index] -= replacement_used

    delta = torch.zeros_like(anchor)
    allowed_turnover = confidence * float(maximum_incremental_one_way_turnover)
    desired_buy_mass = torch.zeros_like(allowed_turnover)
    desired_sell_mass = torch.zeros_like(allowed_turnover)
    buys = torch.zeros_like(anchor)
    sells = torch.zeros_like(anchor)
    for _ in range(M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_ITERATIONS):
        risk_gradient = _factor_diagonal_risk_gradient(
            delta,
            covariance_factor,
            specific_variance,
            selected_horizon_sessions=selected_horizon_sessions,
        )
        candidate = delta + float(proximal_step_size) * (
            mean - float(risk_aversion) * risk_gradient
        )
        threshold = float(proximal_step_size) * 0.5 * cost
        candidate = torch.sign(candidate) * torch.relu(candidate.abs() - threshold)
        positive_candidate = torch.relu(candidate) * buy_gate
        negative_candidate = torch.relu(-candidate) * sell_gate
        desired_buy_mass = positive_candidate.sum(-1)
        desired_sell_mass = negative_candidate.sum(-1)
        buy_capacity = (caps - replacement_anchor).clamp_min(0.0) * buy_gate
        sell_capacity = replacement_anchor.clamp_min(0.0) * sell_gate
        common_mass = torch.minimum(desired_buy_mass, desired_sell_mass)
        common_mass = torch.minimum(common_mass, allowed_turnover)
        common_mass = torch.minimum(common_mass, buy_capacity.sum(-1))
        common_mass = torch.minimum(common_mass, sell_capacity.sum(-1))
        buys, buy_mass = capped_waterfill(common_mass, positive_candidate, buy_capacity)
        sells, sell_mass = capped_waterfill(
            common_mass, negative_candidate, sell_capacity
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

    result = M03RV11CostAwareActiveProposalV3(
        post_exit_derisk_anchor_weights=anchor,
        replacement_anchor_weights=replacement_anchor,
        requested_weights=replacement_anchor + delta,
        requested_delta=delta,
        replacement_buy_weights=replacement_buys,
        reallocation_buy_weights=buys,
        reallocation_sell_weights=sells,
        learned_exit_proceeds=learned_exit_proceeds,
        explicit_derisk_amount=derisk,
        replacement_budget=replacement_budget,
        desired_replacement_mass=desired_replacement_mass,
        replacement_used=replacement_used,
        desired_buy_mass=desired_buy_mass,
        desired_sell_mass=desired_sell_mass,
        entry_probability=entry_probability,
        exit_probability=exit_probability,
        buy_gate=buy_gate,
        sell_gate=sell_gate,
        same_step_buy_mask=same_step_buy_mask,
        requested_incremental_one_way_turnover=0.5 * delta.abs().sum(-1),
        allowed_incremental_one_way_turnover=allowed_turnover,
        selected_horizon_sessions=selected_horizon_sessions,
        research_contract_sha256=research_contract_sha256,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_ITERATIONS",
    "M03R_V11_COST_AWARE_ACTIVE_POLICY_V3_SCHEMA",
    "M03R_V11_COST_AWARE_ELIGIBLE_HORIZONS",
    "M03RV11CostAwareActivePolicyError",
    "M03RV11CostAwareActiveProposalV3",
    "build_cost_aware_active_proposal_v3",
]
