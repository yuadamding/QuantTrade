"""Shared fill-pricing primitives: ``MarketSnapshot`` (point-in-time bar prices) and ``_fill_price``
(the honest fill price for a given fill level / side). BOTH the scalar dollar path and the leg-level
return path price fills through these -- the one piece of state they genuinely share -- so it lives here,
below both, rather than being duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_quant.execution.types import (
    ExecutionConfig,
    FillLevel,
)
from rl_quant.execution.validation import (
    _coerce_finite_nonnegative,
    _coerce_positive_price,
)


@dataclass(frozen=True)
class MarketSnapshot:
    mid: float
    half_spread: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None

    def __post_init__(self) -> None:
        # A non-positive mid would produce meaningless P&L and costs (and crossing distances).
        object.__setattr__(self, "mid", _coerce_positive_price("mid", self.mid))
        # A negative half_spread would turn the cost into a NEGATIVE cost (paying the agent to trade).
        object.__setattr__(self, "half_spread", _coerce_finite_nonnegative("half_spread", self.half_spread))
        # Quotes, when present, must be positive prices (a negative/zero bid or ask is not a real quote).
        for name in ("best_bid", "best_ask"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _coerce_positive_price(name, value))
        if self.best_bid is not None and self.best_ask is not None:
            if self.best_bid > self.best_ask:
                raise ValueError(f"best_bid ({self.best_bid}) must be <= best_ask ({self.best_ask}).")
            # The mid must sit inside the quoted market: a mid outside [bid, ask] makes the quote-side
            # crossing distance |fill - mid| nonsensical (negative or absurdly large).
            if not (self.best_bid <= self.mid <= self.best_ask):
                raise ValueError(
                    f"mid ({self.mid}) must lie within [best_bid, best_ask] "
                    f"([{self.best_bid}, {self.best_ask}]) when both quotes are provided."
                )


def _fill_price(snapshot: MarketSnapshot, *, buying: bool, config: ExecutionConfig) -> float | None:
    """Marketable fill price for a side at a bar. ``None`` for the mid proxy (delayed_close)."""
    if config.fill_level == FillLevel.DELAYED_CLOSE:
        return None
    if config.fill_level == FillLevel.MID_PLUS_SPREAD:
        offset = config.spread_multiplier * snapshot.half_spread
        return snapshot.mid + offset if buying else snapshot.mid - offset
    side_price = snapshot.best_ask if buying else snapshot.best_bid  # buy at ask, sell at bid
    if side_price is None:
        raise ValueError(f"{config.fill_level.value} requires best_bid/best_ask but got None.")
    return float(side_price)
