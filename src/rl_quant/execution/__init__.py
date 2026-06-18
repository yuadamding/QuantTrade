"""Execution layer: transition-P&L cost/fill model + numeric validation contracts.

Split for maintainability into ``validation`` (numeric/integer coercion helpers) and ``engine``
(the cost/fill model, scalar path, and weight-aware leg path). This package re-exports the full
public surface, so ``from rl_quant.execution import simulate_transition`` (etc.) is unchanged --
callers never need to know which submodule a symbol lives in.
"""

from __future__ import annotations

from rl_quant.execution.engine import (
    FillLevel,
    TerminalPolicy,
    SwitchFillPolicy,
    ImpactModel,
    WeightExecutionCostConfig,
    weight_transition_cost_bps,
    ExecutionConfig,
    PositionState,
    MarketSnapshot,
    TransitionOutcome,
    fill_index,
    fill_indices,
    transition_pnl,
    simulate_transition,
    LegSide,
    FillStatus,
    SymbolQuote,
    Holdings,
    ExecutionLeg,
    ActionTransitionOutcome,
    simulate_action_transition,
)
from rl_quant.execution.validation import (
    require_positive_int,
    require_nonnegative_int,
    require_bool,
    coerce_finite_nonnegative,
    coerce_finite_positive,
)

__all__ = [
    ActionTransitionOutcome,
    ExecutionConfig,
    ExecutionLeg,
    FillLevel,
    FillStatus,
    Holdings,
    ImpactModel,
    LegSide,
    MarketSnapshot,
    PositionState,
    SwitchFillPolicy,
    SymbolQuote,
    TerminalPolicy,
    TransitionOutcome,
    WeightExecutionCostConfig,
    coerce_finite_nonnegative,
    coerce_finite_positive,
    fill_index,
    fill_indices,
    require_bool,
    require_nonnegative_int,
    require_positive_int,
    simulate_action_transition,
    simulate_transition,
    transition_pnl,
    weight_transition_cost_bps,
]
