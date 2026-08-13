"""Differentiable target-weight accounting shared by environments and trainers.

This module deliberately stops short of a venue-fill simulator.  The immediate
target-weight model below records an accounting transition and explicit cost
assumptions; it does not claim latency, queue position, fill price, or empirical
market impact that the input data cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Protocol, runtime_checkable

import torch

from rl_quant.execution.types import (
    WeightExecutionCostConfig,
    weight_transition_cost_bps,
)
from rl_quant.execution.validation import (
    coerce_finite_nonnegative,
    require_nonnegative_int,
)


@dataclass(frozen=True)
class TargetWeightExecutionResult:
    """Authoritative costs for one immediate target-weight accounting transition."""

    execution_cost: torch.Tensor
    modeled_impact_cost: torch.Tensor
    traded_notional: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.execution_cost.shape
        device = self.execution_cost.device
        if self.execution_cost.ndim == 0 or not self.execution_cost.is_floating_point():
            raise ValueError("execution_cost needs a floating leading batch dimension.")
        fields = {
            "execution_cost": self.execution_cost,
            "modeled_impact_cost": self.modeled_impact_cost,
            "traded_notional": self.traded_notional,
        }
        for name, value in fields.items():
            if (
                value.shape != shape
                or value.device != device
                or value.dtype != self.execution_cost.dtype
            ):
                raise ValueError(
                    f"{name} must have shape {shape}, dtype {self.execution_cost.dtype}, on {device}."
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite.")
            if bool((value < 0).any().item()):
                raise ValueError(f"{name} must be nonnegative.")
        for name, value in self.diagnostics.items():
            if not name:
                raise ValueError("Execution diagnostic names must be non-empty.")
            if value.ndim == 0 or value.shape[0] != shape[0] or value.device != device:
                raise ValueError(
                    f"diagnostics[{name!r}] needs leading batch size {shape[0]} on {device}."
                )
            if value.is_floating_point() and not bool(
                torch.isfinite(value).all().item()
            ):
                raise ValueError(f"diagnostics[{name!r}] must be finite.")


@runtime_checkable
class TargetWeightExecutionModel(Protocol):
    """Bounded execution-cost surface for an already feasible target.

    The environment, not this model, owns availability, leverage, turnover, and
    hard-risk enforcement.  Consequently the model receives only the current and
    final feasible target weights and cannot substitute a different allocation.
    """

    def execute(
        self,
        current_weights: torch.Tensor,
        target_weights: torch.Tensor,
        *,
        cash_index: int,
    ) -> TargetWeightExecutionResult: ...


@dataclass(frozen=True)
class FixedTurnoverTargetWeightExecution:
    """Charge a fixed number of basis points on canonical one-way turnover.

    Turnover is half the L1 distance between the *full* current and target
    portfolios, including CASH.  This is the same cost basis used by the legacy
    differentiable-portfolio trainers and deliberately differs from summing both
    risky legs of an asset-to-asset switch.

    The result is expressed in return units for the environment's additive
    reward ledger.  This deterministic accounting model does not claim to model
    market fills or impact.
    """

    cost_bps: float = 0.0
    models_market_fills: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cost_bps", coerce_finite_nonnegative("cost_bps", self.cost_bps)
        )

    def execute(
        self,
        current_weights: torch.Tensor,
        target_weights: torch.Tensor,
        *,
        cash_index: int,
    ) -> TargetWeightExecutionResult:
        if current_weights.shape != target_weights.shape or current_weights.ndim < 2:
            raise ValueError(
                "current_weights and target_weights need identical [..., asset] shapes."
            )
        if (
            not current_weights.is_floating_point()
            or target_weights.dtype != current_weights.dtype
            or target_weights.device != current_weights.device
        ):
            raise ValueError(
                "Current and target weights must share one floating dtype and device."
            )
        checked_cash_index = require_nonnegative_int("cash_index", cash_index)
        if checked_cash_index >= current_weights.shape[-1]:
            raise ValueError("cash_index is outside the target-weight action axis.")
        if not bool(torch.isfinite(current_weights).all().item()) or not bool(
            torch.isfinite(target_weights).all().item()
        ):
            raise ValueError("Current and target weights must be finite.")

        turnover = one_way_turnover(target_weights, current_weights)
        execution_cost = turnover * torch.as_tensor(
            self.cost_bps / 10_000.0,
            dtype=current_weights.dtype,
            device=current_weights.device,
        )
        return TargetWeightExecutionResult(
            execution_cost=execution_cost,
            modeled_impact_cost=torch.zeros_like(turnover),
            traded_notional=turnover,
            diagnostics={"one_way_turnover": turnover},
        )


@dataclass(frozen=True)
class ImmediateTargetWeightExecution:
    """Immediate target-weight accounting with explicit deterministic cost proxies.

    ``spread_bps`` and ``fee_bps`` are charged on absolute non-cash traded
    weight.  The optional linear term is a configured sensitivity analysis, not
    an empirical impact or fill estimate.
    """

    spread_bps: float = 0.0
    fee_bps: float = 0.0
    modeled_linear_impact_bps_per_weight: float = 0.0
    models_market_fills: ClassVar[bool] = False
    _weight_cost: WeightExecutionCostConfig = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "spread_bps", coerce_finite_nonnegative("spread_bps", self.spread_bps)
        )
        object.__setattr__(
            self, "fee_bps", coerce_finite_nonnegative("fee_bps", self.fee_bps)
        )
        object.__setattr__(
            self,
            "modeled_linear_impact_bps_per_weight",
            coerce_finite_nonnegative(
                "modeled_linear_impact_bps_per_weight",
                self.modeled_linear_impact_bps_per_weight,
            ),
        )
        object.__setattr__(
            self,
            "_weight_cost",
            WeightExecutionCostConfig(
                fee_bps=self.fee_bps,
                impact_kind="linear_bps"
                if self.modeled_linear_impact_bps_per_weight > 0.0
                else "none",
                linear_impact_bps_per_weight=self.modeled_linear_impact_bps_per_weight,
            ),
        )

    def execute(
        self,
        current_weights: torch.Tensor,
        target_weights: torch.Tensor,
        *,
        cash_index: int,
    ) -> TargetWeightExecutionResult:
        if current_weights.shape != target_weights.shape or current_weights.ndim < 2:
            raise ValueError(
                "current_weights and target_weights need identical [..., asset] shapes."
            )
        if (
            not current_weights.is_floating_point()
            or target_weights.dtype != current_weights.dtype
            or target_weights.device != current_weights.device
        ):
            raise ValueError(
                "Current and target weights must share one floating dtype and device."
            )
        if not 0 <= cash_index < current_weights.shape[-1]:
            raise ValueError("cash_index is outside the target-weight action axis.")
        if not bool(torch.isfinite(current_weights).all().item()) or not bool(
            torch.isfinite(target_weights).all().item()
        ):
            raise ValueError("Current and target weights must be finite.")

        delta = target_weights - current_weights
        risky_delta = delta.clone()
        risky_delta[..., cash_index] = 0.0
        sell_weight = (-risky_delta).clamp_min(0.0)
        buy_weight = risky_delta.clamp_min(0.0)
        traded_notional = risky_delta.abs().sum(dim=-1)
        fee_and_impact_bps = weight_transition_cost_bps(
            sell_weight,
            buy_weight,
            weight_cost=self._weight_cost,
        ).sum(dim=-1)
        fee_bps = traded_notional * self.fee_bps
        execution_cost = (traded_notional * self.spread_bps + fee_bps) / 10_000.0
        modeled_impact_cost = (fee_and_impact_bps - fee_bps).clamp_min(0.0) / 10_000.0
        return TargetWeightExecutionResult(
            execution_cost=execution_cost,
            modeled_impact_cost=modeled_impact_cost,
            traded_notional=traded_notional,
        )


def one_way_turnover(
    new_weights: torch.Tensor, old_weights: torch.Tensor
) -> torch.Tensor:
    """Half-L1 portfolio turnover, including CASH, on the final axis."""
    if new_weights.shape != old_weights.shape:
        raise ValueError("new_weights and old_weights must have identical shapes")
    return 0.5 * (new_weights - old_weights).abs().sum(dim=-1)


def force_unavailable_to_cash(
    weights: torch.Tensor,
    available: torch.Tensor,
    *,
    cash_index: int = 0,
) -> torch.Tensor:
    """Project unavailable holdings into CASH without detaching gradients."""
    if weights.shape != available.shape:
        raise ValueError("weights and available must have identical shapes")
    if not 0 <= cash_index < weights.shape[-1]:
        raise ValueError("cash_index is outside the action axis")
    kept = torch.where(available.bool(), weights, torch.zeros_like(weights))
    cash = kept[..., cash_index] + (1.0 - kept.sum(dim=-1))
    out = kept.clone()
    out[..., cash_index] = cash
    return out


def drift_weights(
    weights: torch.Tensor,
    asset_returns: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mark target weights through one realized return and renormalize."""
    if weights.shape != asset_returns.shape:
        raise ValueError("weights and asset_returns must have identical shapes")
    marked = asset_returns
    if valid is not None:
        if valid.shape != weights.shape:
            raise ValueError("valid must match weights")
        marked = torch.where(
            valid.bool(), asset_returns, torch.zeros_like(asset_returns)
        )
    growth = 1.0 + marked
    # Compute portfolio growth as one plus the weighted return.  Expanding the
    # leading one per asset and then reducing is mathematically equivalent only
    # under exact arithmetic; in FP32 it can differ by enough ULPs to disagree
    # with cohort accounting's canonical growth scalar.
    portfolio_growth = 1.0 + (weights * marked).sum(dim=-1, keepdim=True)
    # Return inputs are validated/clipped at the data/environment boundary. Avoid a host synchronization in this
    # hot differentiable kernel; the denominator guard keeps tiny positive growth numerically stable.
    return weights * growth / portfolio_growth.clamp_min(1e-12)


def fixed_turnover_cost(
    new_weights: torch.Tensor,
    old_weights: torch.Tensor,
    cost_rate: float | torch.Tensor,
) -> torch.Tensor:
    """Cost in return units under the framework's canonical half-L1 convention."""
    return one_way_turnover(new_weights, old_weights) * torch.as_tensor(
        cost_rate, dtype=new_weights.dtype, device=new_weights.device
    )
