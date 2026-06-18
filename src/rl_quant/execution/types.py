"""Execution config + enums: fill levels, terminal/switch policies, the impact model, and the
weight-bps cost config -- the value types every execution path is parameterised by. Depend only on the
numeric-validation contracts in ``rl_quant.execution.validation``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import torch

from rl_quant.execution.validation import (
    _coerce_finite_nonnegative,
    _require_nonnegative_int,
    _require_positive_int,
)


class FillLevel(str, Enum):
    DELAYED_CLOSE = "delayed_close"  # mid move + symmetric half-spread cost proxy (today's intraday)
    MID_PLUS_SPREAD = "mid_plus_spread"  # fill = mid +/- half_spread * spread_multiplier
    QUOTE_SIDE = "quote_side"  # buy at ask, sell at bid (needs best_bid/best_ask)
    QUOTE_SIDE_PLUS_IMPACT = "quote_side_plus_impact"  # quote_side + size-dependent linear impact


class TerminalPolicy(str, Enum):
    LIQUIDATE_AT_NEXT = "liquidate_at_next"  # charge |new| * cost(half_spread_next) at a true terminal
    CARRY = "carry"  # no liquidation (episode-length truncation / bootstrap-through)


class SwitchFillPolicy(str, Enum):
    # How a multi-leg switch behaves when only SOME of its legs can fill (leg-level path only).
    INDEPENDENT_LEGS = "independent_legs"  # each leg fills on its own book; partial fills allowed (default,
    #                                        not weight-conserving -- can over-allocate or strand in cash)
    ATOMIC_SWITCH = "atomic_switch"  # all-or-nothing: if ANY required leg cannot fill, execute NONE of them
    #                                  (keep prior holdings). Recommended for reportable allocator evaluation.


@dataclass(frozen=True)
class ImpactModel:
    kind: str = "none"  # "none" | "linear"
    coef_per_unit: float = 0.0  # extra $/share per turnover unit (linear market impact)

    def __post_init__(self) -> None:
        # Only "none"/"linear" are implemented; reject anything else (incl. sqrt/almgren_chriss until
        # built) so a typo like "liner" can't silently DISABLE impact (the simulator treats any
        # non-"linear" kind as zero impact).
        if self.kind not in ("none", "linear"):
            raise ValueError(
                f"impact_model.kind must be 'none' or 'linear' (sqrt/almgren_chriss not yet implemented); "
                f"got {self.kind!r}."
            )
        object.__setattr__(
            self, "coef_per_unit", _coerce_finite_nonnegative("impact_model.coef_per_unit", self.coef_per_unit)
        )


def _coerce_impact_model(value: object) -> ImpactModel:
    # @dataclass does NOT enforce field types at runtime, so coerce a mapping and accept an ImpactModel,
    # but reject anything else (e.g. a bare "linear" string) at construction -- otherwise the bad value
    # sits silently until ``_impact_per_share`` reads ``.kind`` and crashes with an opaque AttributeError.
    if isinstance(value, ImpactModel):
        return value
    if isinstance(value, Mapping):
        try:
            return ImpactModel(**value)
        except TypeError as exc:
            raise ValueError(f"invalid impact_model mapping: {value!r}") from exc
    raise ValueError(f"impact_model must be an ImpactModel or mapping; got {type(value).__name__}.")


@dataclass(frozen=True)
class WeightExecutionCostConfig:
    """bps-denominated execution cost for the RETURN/weight-based leg-level path.

    Deliberately separate from ExecutionConfig's per-share DOLLAR fields (commission_per_share /
    extra_cost_per_share / ImpactModel.coef_per_unit), which belong to the scalar signed-position model:
    folding $/share into a weight-return model is a units error. Here every cost is in basis points and is
    charged on a leg's traded weight. The default is zero (leg cost stays spread-only, unchanged)."""

    fee_bps: float = 0.0  # flat per-leg fee (commission + misc) in bps of traded weight
    impact_kind: str = "none"  # "none" | "linear_bps"
    linear_impact_bps_per_weight: float = 0.0  # impact_bps = coef * traded_weight (linear market impact)

    def __post_init__(self) -> None:
        if self.impact_kind not in ("none", "linear_bps"):
            raise ValueError(
                f"weight_cost.impact_kind must be 'none' or 'linear_bps' (sqrt/almgren_chriss not yet "
                f"implemented); got {self.impact_kind!r}."
            )
        object.__setattr__(self, "fee_bps", _coerce_finite_nonnegative("weight_cost.fee_bps", self.fee_bps))
        object.__setattr__(
            self,
            "linear_impact_bps_per_weight",
            _coerce_finite_nonnegative("weight_cost.linear_impact_bps_per_weight", self.linear_impact_bps_per_weight),
        )

    def impact_bps(self, traded_weight: float) -> float:
        # Linear market impact: impact in bps grows with trade size, so total impact COST ~ size^2.
        if self.impact_kind == "linear_bps":
            return self.linear_impact_bps_per_weight * abs(float(traded_weight))
        return 0.0


def weight_transition_cost_bps(
    sell_weight: torch.Tensor,
    buy_weight: torch.Tensor,
    *,
    weight_cost: WeightExecutionCostConfig,
) -> torch.Tensor:
    """Vectorized weight-bps execution cost (in BPS) of a single-slot transition's two legs -- a SELL of the
    prior position's weight + a BUY of the new position's weight. Matches ``simulate_action_transition``'s
    per-leg ``WeightExecutionCostConfig`` charge: each leg costs ``traded * (fee_bps + impact_bps(traded))``
    with linear ``impact_bps(traded) = linear_impact_bps_per_weight * |traded|`` (so impact COST ~ size^2).
    Inputs are the EFFECTIVE traded weights per leg (cash already zeroed; 0 on a hold). The result is in bps,
    i.e. ``1e4 * `` the engine's return-unit ``realized_execution_cost``. Use this in a VECTORIZED env step --
    the dataclass ``simulate_action_transition`` is too heavy per step; an equivalence test pins them together."""
    coef = float(weight_cost.linear_impact_bps_per_weight) if weight_cost.impact_kind == "linear_bps" else 0.0
    fee = float(weight_cost.fee_bps)

    def _leg(traded: torch.Tensor) -> torch.Tensor:
        traded = traded.abs()
        return traded * (fee + coef * traded)

    return _leg(sell_weight) + _leg(buy_weight)


def _coerce_weight_cost(value: object) -> WeightExecutionCostConfig:
    if isinstance(value, WeightExecutionCostConfig):
        return value
    if isinstance(value, Mapping):
        try:
            return WeightExecutionCostConfig(**value)
        except TypeError as exc:
            raise ValueError(f"invalid weight_cost mapping: {value!r}") from exc
    raise ValueError(f"weight_cost must be a WeightExecutionCostConfig or mapping; got {type(value).__name__}.")


@dataclass(frozen=True)
class ExecutionConfig:
    fill_level: FillLevel = FillLevel.DELAYED_CLOSE
    latency_steps: int = 0  # fill bar = min(now + latency, next); see fill_index
    step_horizon: int = 1
    trade_lot_size: int = 1
    commission_per_share: float = 0.0
    extra_cost_per_share: float = 0.0
    spread_multiplier: float = 1.0  # scales the half-spread proxy / mid_plus_spread crossing depth
    impact_model: ImpactModel = field(default_factory=ImpactModel)
    terminal_policy: TerminalPolicy = TerminalPolicy.LIQUIDATE_AT_NEXT
    # Default preserves the current leg-level behavior (independent per-leg fills); a future trainer
    # wiring should set ATOMIC_SWITCH for reportable allocator evaluation. Only affects the leg-level path.
    switch_fill_policy: SwitchFillPolicy = SwitchFillPolicy.INDEPENDENT_LEGS
    # bps-denominated fee/impact for the leg-level (weight-return) path ONLY; default zero -> leg cost stays
    # spread-only. Distinct from the per-share dollar fields above, which drive the scalar dollar model.
    weight_cost: WeightExecutionCostConfig = field(default_factory=WeightExecutionCostConfig)

    def __post_init__(self) -> None:
        # Fail closed on invalid execution parameters: a research run must never silently claim a
        # negative/NaN latency or cost, a non-positive horizon/lot, or an unknown fill level / policy.
        # Coerce the enums so a valid string is accepted but an unknown one raises clearly (instead of
        # falling through to quote-side logic and crashing later on a missing .value).
        object.__setattr__(self, "fill_level", FillLevel(self.fill_level))
        object.__setattr__(self, "terminal_policy", TerminalPolicy(self.terminal_policy))
        object.__setattr__(self, "switch_fill_policy", SwitchFillPolicy(self.switch_fill_policy))
        # Coerce-and-store the bar/lot counts so a fractional value can't slip through int() truncation
        # (int(1.9) == 1) and silently rescale every dollar P&L through trade_scale.
        object.__setattr__(self, "latency_steps", _require_nonnegative_int("latency_steps", self.latency_steps))
        object.__setattr__(self, "step_horizon", _require_positive_int("step_horizon", self.step_horizon))
        object.__setattr__(self, "trade_lot_size", _require_positive_int("trade_lot_size", self.trade_lot_size))
        object.__setattr__(
            self, "commission_per_share", _coerce_finite_nonnegative("commission_per_share", self.commission_per_share)
        )
        object.__setattr__(
            self, "extra_cost_per_share", _coerce_finite_nonnegative("extra_cost_per_share", self.extra_cost_per_share)
        )
        object.__setattr__(
            self, "spread_multiplier", _coerce_finite_nonnegative("spread_multiplier", self.spread_multiplier)
        )
        object.__setattr__(self, "impact_model", _coerce_impact_model(self.impact_model))
        object.__setattr__(self, "weight_cost", _coerce_weight_cost(self.weight_cost))
        # quote_side_plus_impact must carry a REAL (positive linear) impact: otherwise it is numerically
        # identical to plain quote_side yet would still advertise that it models impact. Fail closed and
        # tell the caller to use quote_side for a zero-impact crossable fill.
        if self.fill_level == FillLevel.QUOTE_SIDE_PLUS_IMPACT and not (
            self.impact_model.kind == "linear" and self.impact_model.coef_per_unit > 0.0
        ):
            raise ValueError(
                "quote_side_plus_impact requires impact_model.kind='linear' with coef_per_unit > 0; "
                "use quote_side for a zero-impact crossable fill."
            )

    @property
    def trade_scale(self) -> float:
        return float(self.trade_lot_size) * 100.0

    @property
    def uses_crossable_quote_fills(self) -> bool:
        # quote_side / quote_side_plus_impact buy at ask and sell at bid -> a real, crossable fill model.
        return self.fill_level in (FillLevel.QUOTE_SIDE, FillLevel.QUOTE_SIDE_PLUS_IMPACT)

    @property
    def applies_implemented_impact(self) -> bool:
        # SCALAR (signed-position, per-share dollar) impact axis: True only when the scalar impact_model
        # applies a positive linear impact. __post_init__ guarantees this for quote_side_plus_impact. This
        # is the impact axis for the transition_pnl / simulate_transition path -- NOT the leg-level path.
        return (
            self.fill_level == FillLevel.QUOTE_SIDE_PLUS_IMPACT
            and self.impact_model.kind == "linear"
            and self.impact_model.coef_per_unit > 0.0
        )

    @property
    def applies_weight_impact(self) -> bool:
        # LEG-LEVEL (return/weight) impact axis: True only when weight_cost charges a positive linear impact.
        # The leg path (_make_leg) prices impact from weight_cost, NOT from impact_model -- so a
        # quote_side_plus_impact config (which has a positive SCALAR impact_model) still applies ZERO impact
        # on the leg path unless weight_cost is set. The report layer must check THIS for leg-level runs so a
        # transition is never labelled impact-priced when the leg engine charged no impact.
        return self.weight_cost.impact_kind != "none" and self.weight_cost.linear_impact_bps_per_weight > 0.0

    @property
    def proxy_fill_model(self) -> bool:
        # delayed_close / mid_plus_spread are mid-based proxies, NOT crossable fills.
        return self.fill_level in (FillLevel.DELAYED_CLOSE, FillLevel.MID_PLUS_SPREAD)

    @property
    def real_executable_fill_model(self) -> bool:
        # The FILL MODEL is real iff fills are crossable quote-side fills. NOTE: a fully reportable "real
        # executable trade" is STRICTER -- it also needs latency P&L, applied impact, AND real fill-price
        # logs -- and must be judged at the report layer from this flag AND the decision logs, never from
        # config alone. This property is only the fill-model half of that judgement.
        return self.uses_crossable_quote_fills
