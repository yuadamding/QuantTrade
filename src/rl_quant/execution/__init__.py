"""Execution layer with lazy compatibility exports.

Lazy loading keeps independent execution generations from acquiring one
another's runtime dependencies merely because Python initializes this package.
The historical public import surface remains unchanged.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final


def _exports(module: str, *names: str) -> dict[str, tuple[str, str]]:
    return {name: (module, name) for name in names}


_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    **_exports(
        "rl_quant.execution.age_aware_no_trade",
        "AgeAwareNoTradeConfig",
        "AgeAwareNoTradeError",
        "ForecastDistribution",
        "ReplacementDecision",
        "evaluate_replacement",
    ),
    **_exports("rl_quant.execution.fills", "MarketSnapshot"),
    **_exports(
        "rl_quant.execution.leg",
        "ActionTransitionOutcome",
        "ExecutionLeg",
        "FillStatus",
        "Holdings",
        "LegSide",
        "SymbolQuote",
        "simulate_action_transition",
    ),
    **_exports(
        "rl_quant.execution.massive_adaptive_portfolio_compiler_v1",
        "MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SCHEMA",
        "MASSIVE_ADAPTIVE_PORTFOLIO_COMPILER_V1_SOLVER",
        "MassiveAdaptivePortfolioCompilerConfigV1",
        "MassiveAdaptivePortfolioCompilerError",
        "MassiveAdaptivePortfolioCompilerInputsV1",
        "MassiveAdaptivePortfolioDecisionV1",
        "compile_massive_adaptive_portfolio_v1",
    ),
    **_exports(
        "rl_quant.execution.impact_model",
        "AlphaExecutionCostError",
        "CapacityEstimate",
        "ExecutionCostEstimate",
        "ExecutionCostObservation",
        "SquareRootImpactConfig",
        "estimate_execution_cost",
        "evaluate_capacity",
    ),
    **_exports(
        "rl_quant.execution.portfolio",
        "FixedTurnoverTargetWeightExecution",
        "ImmediateTargetWeightExecution",
        "TargetWeightExecutionModel",
        "TargetWeightExecutionResult",
        "drift_weights",
        "fixed_turnover_cost",
        "force_unavailable_to_cash",
        "one_way_turnover",
    ),
    **_exports(
        "rl_quant.execution.scalar",
        "PositionState",
        "TransitionOutcome",
        "fill_index",
        "fill_indices",
        "simulate_transition",
        "transition_pnl",
    ),
    **_exports(
        "rl_quant.execution.types",
        "ExecutionConfig",
        "FillLevel",
        "ImpactModel",
        "SwitchFillPolicy",
        "TerminalPolicy",
        "WeightExecutionCostConfig",
        "weight_transition_cost_bps",
    ),
    **_exports(
        "rl_quant.execution.validation",
        "coerce_finite_nonnegative",
        "coerce_finite_positive",
        "require_bool",
        "require_nonnegative_int",
        "require_positive_int",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one compatibility export only when it is requested."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to introspection."""

    return sorted(set(globals()) | set(__all__))
