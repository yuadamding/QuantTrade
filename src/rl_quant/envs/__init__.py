"""Environment contracts with lazy compatibility exports.

Initializing an adaptive environment submodule does not load any historical
environment generation.  Direct imports from :mod:`rl_quant.envs` retain the
same legacy public API and resolve only the requested symbol.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final


def _exports(module: str, *names: str) -> dict[str, tuple[str, str]]:
    return {name: (module, name) for name in names}


_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    **_exports(
        "rl_quant.envs.hold30",
        "AGE_BIN_COUNT",
        "MAX_EXACT_AGE",
        "TARGET_HOLDING_DAYS",
        "CohortLedger",
        "CohortTradeAccounting",
        "TurnoverCause",
        "net_trade_legs",
    ),
    **_exports(
        "rl_quant.envs.market",
        "HistoricalMarketData",
        "PortfolioObservationAdapter",
        "TensorPortfolioObservationAdapter",
    ),
    **_exports(
        "rl_quant.envs.portfolio",
        "PortfolioConstraints",
        "PortfolioCostModel",
        "PortfolioEnvState",
        "VectorPortfolioEnv",
    ),
    **_exports(
        "rl_quant.envs.robust",
        "LiquidityCostStress",
        "LowerEnvelopeScenario",
        "LowerEnvelopeTransformSuite",
        "SequentialTransitionTransform",
        "TrendReturnFeatureReversal",
        "market_lower_envelope_suite",
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
