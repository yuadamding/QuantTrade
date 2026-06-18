"""Execution layer: transition-P&L cost/fill model + numeric validation contracts.

Organized as a foundation package:
  * ``validation`` -- numeric/integer coercion contracts (reject bool-as-float / NaN / fractional int);
  * ``types``      -- execution config + enums (fill levels, policies, impact model, weight-bps cost);
  * ``fills``      -- shared fill-pricing primitives (``MarketSnapshot`` + ``_fill_price``);
  * ``scalar``     -- signed-position dollar transition-P&L path (single instrument);
  * ``leg``        -- return-based multi-symbol leg path (ETF allocation switches).

Layering: validation < types < fills < {scalar, leg}. This package re-exports the full public surface,
so ``from rl_quant.execution import simulate_transition`` (etc.) is unchanged -- callers never need to
know which submodule a symbol lives in.
"""

from __future__ import annotations

from rl_quant.execution.fills import (
    MarketSnapshot,
)
from rl_quant.execution.leg import (
    ActionTransitionOutcome,
    ExecutionLeg,
    FillStatus,
    Holdings,
    LegSide,
    SymbolQuote,
    simulate_action_transition,
)
from rl_quant.execution.scalar import (
    PositionState,
    TransitionOutcome,
    fill_index,
    fill_indices,
    simulate_transition,
    transition_pnl,
)
from rl_quant.execution.types import (
    ExecutionConfig,
    FillLevel,
    ImpactModel,
    SwitchFillPolicy,
    TerminalPolicy,
    WeightExecutionCostConfig,
    weight_transition_cost_bps,
)
from rl_quant.execution.validation import (
    coerce_finite_nonnegative,
    coerce_finite_positive,
    require_bool,
    require_nonnegative_int,
    require_positive_int,
)

__all__ = [
    "ActionTransitionOutcome",
    "ExecutionConfig",
    "ExecutionLeg",
    "FillLevel",
    "FillStatus",
    "Holdings",
    "ImpactModel",
    "LegSide",
    "MarketSnapshot",
    "PositionState",
    "SwitchFillPolicy",
    "SymbolQuote",
    "TerminalPolicy",
    "TransitionOutcome",
    "WeightExecutionCostConfig",
    "coerce_finite_nonnegative",
    "coerce_finite_positive",
    "fill_index",
    "fill_indices",
    "require_bool",
    "require_nonnegative_int",
    "require_positive_int",
    "simulate_action_transition",
    "simulate_transition",
    "transition_pnl",
    "weight_transition_cost_bps",
]
