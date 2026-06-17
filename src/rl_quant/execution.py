"""Shared transition-P&L execution simulator.

Encodes the real-life cost of moving from a HELD position to a candidate at decision time:

    net = old * (mid_fill - mid_now)      # old position keeps earning until its (latency-delayed) fill
        + new * (mid_next - mid_fill)     # new position earns from its fill to the horizon
        - entry_cost                      # turnover * per-share cost at the FILL bar
        - exit_cost                       # terminal liquidation: |new| * per-share cost at the NEXT bar
    (all dollar terms scaled by trade_scale = trade_lot_size * 100)

This is the per-step decomposition currently inlined three times in ``intraday_dqn`` (env step, eval
loop, pretraining-target builder). The module is a pure, dependency-light extraction so those sites --
and, later, the weight-aware second-context / minute-to-hour paths -- can share ONE reward engine.

Honesty contract: ``delayed_close`` is a MID-price proxy with a symmetric half-spread cost, NOT a
crossable fill, so ``real_executable_fill_model`` is False and fill prices are ``None``. Only the
``quote_side*`` levels (buy at ask, sell at bid) are real executable fills.

This module changes no trainer's reward on its own; it is wired in (result-preserving for the
``delayed_close`` default) in a separate step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FillLevel(str, Enum):
    DELAYED_CLOSE = "delayed_close"  # mid move + symmetric half-spread cost proxy (today's intraday)
    MID_PLUS_SPREAD = "mid_plus_spread"  # fill = mid +/- half_spread * spread_multiplier
    QUOTE_SIDE = "quote_side"  # buy at ask, sell at bid (needs best_bid/best_ask)
    QUOTE_SIDE_PLUS_IMPACT = "quote_side_plus_impact"  # quote_side + size-dependent linear impact


class TerminalPolicy(str, Enum):
    LIQUIDATE_AT_NEXT = "liquidate_at_next"  # charge |new| * cost(half_spread_next) at a true terminal
    CARRY = "carry"  # no liquidation (episode-length truncation / bootstrap-through)


@dataclass(frozen=True)
class ImpactModel:
    kind: str = "none"  # "none" | "linear"
    coef_per_unit: float = 0.0  # extra $/share per turnover unit (linear market impact)


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

    def __post_init__(self) -> None:
        # Fail closed on invalid execution parameters: a research run must never silently claim a
        # negative latency, a non-positive horizon/lot, or a negative cost/impact.
        if self.latency_steps < 0:
            raise ValueError(f"latency_steps must be >= 0 (0 = no latency); got {self.latency_steps}.")
        if self.step_horizon <= 0:
            raise ValueError(f"step_horizon must be positive; got {self.step_horizon}.")
        if self.trade_lot_size <= 0:
            raise ValueError(f"trade_lot_size must be positive; got {self.trade_lot_size}.")
        for name in ("commission_per_share", "extra_cost_per_share", "spread_multiplier"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative; got {getattr(self, name)}.")
        if self.impact_model.coef_per_unit < 0:
            raise ValueError(f"impact coef_per_unit must be non-negative; got {self.impact_model.coef_per_unit}.")

    @property
    def trade_scale(self) -> float:
        return float(self.trade_lot_size) * 100.0

    @property
    def real_executable_fill_model(self) -> bool:
        # delayed_close / mid_plus_spread are proxies, NOT crossable fills -> honestly not real.
        return self.fill_level in (FillLevel.QUOTE_SIDE, FillLevel.QUOTE_SIDE_PLUS_IMPACT)


@dataclass(frozen=True)
class PositionState:
    position: float  # signed units; intraday {-1, 0, 1}; weight-aware paths may use a weight
    bars_held: int = 0
    entry_price: float | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    mid: float
    half_spread: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None


@dataclass(frozen=True)
class TransitionOutcome:
    old_latency_return: float  # old * (mid_fill - mid_now) * scale   (held leg, now -> fill)
    new_interval_return: float  # new * (mid_next - mid_fill) * scale  (new leg, fill -> next)
    gross_return: float
    entry_fill_price: float | None  # price the NEW leg is acquired at (None for proxy fills)
    exit_fill_price: float | None  # close-out price at terminal liquidation (None otherwise)
    entry_cost: float  # turnover spread+fees cost at the fill bar (>= 0)
    exit_cost: float  # terminal liquidation spread+fees cost at the next bar (>= 0)
    impact_cost: float  # market-impact dollars (0 unless a *_plus_impact level)
    total_cost: float  # entry_cost + exit_cost + impact_cost
    net_return: float  # gross_return - total_cost  == the trainer reward
    order_legs: float  # |new - old| turnover units; a full reversal (-1 -> +1) is 2
    real_executable_fill_model: bool
    next_state: PositionState


def fill_index(now_index: int, *, step_horizon: int, latency_steps: int) -> int:
    """Bar at which a decision fills: ``min(now + latency, next)``, capped at the next decision bar.

    ``latency_steps <= 0`` collapses the fill to the current bar (decision and fill coincide). Mirrors
    intraday ``_fill_indices`` exactly, so a fill can never be pushed past the holding horizon."""
    if int(step_horizon) <= 0:
        raise ValueError(f"step_horizon must be positive; got {step_horizon}.")
    next_index = now_index + int(step_horizon)
    if int(latency_steps) <= 0:
        return now_index
    return min(now_index + int(latency_steps), next_index)


def transition_pnl(
    old_position,
    new_position,
    mid_now,
    mid_fill,
    mid_next,
    half_spread_fill,
    half_spread_next,
    terminal,
    *,
    trade_scale: float,
    commission_per_share: float,
    extra_cost_per_share: float,
):
    """Vectorized ``delayed_close`` transition reward in scaled dollars -- the single source of truth for
    the intraday env-step / eval / pretraining-target reward (previously inlined three times).

    Operates on torch tensors OR python scalars via ``+ - * abs`` only (no torch import, no ``.float()``
    casts -- long*float and bool*float promote identically), so the same function serves the vectorized
    env/grid and the scalar eval loop. The held (old) position earns ``mid_now -> mid_fill`` (the latency
    leg) and the new position earns ``mid_fill -> mid_next``; turnover pays the fill-bar cost and a true
    terminal pays an extra ``|new|`` liquidation cost at the next bar. Equals
    ``simulate_transition(...).net_return`` for the delayed_close fill level (cross-checked in tests)."""
    cost_fill = half_spread_fill + extra_cost_per_share + commission_per_share
    cost_next = half_spread_next + extra_cost_per_share + commission_per_share
    turnover = abs(new_position - old_position)
    return (
        old_position * (mid_fill - mid_now)
        + new_position * (mid_next - mid_fill)
        - turnover * cost_fill
        - terminal * abs(new_position) * cost_next
    ) * trade_scale


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


def _base_cost_per_share(snapshot: MarketSnapshot, *, buying: bool, config: ExecutionConfig) -> float:
    """Spread + fees cost per share (NO impact). Proxy levels use the half-spread; quote-side levels
    use the actual crossing distance |fill_price - mid|."""
    if config.fill_level in (FillLevel.DELAYED_CLOSE, FillLevel.MID_PLUS_SPREAD):
        spread_cost = config.spread_multiplier * snapshot.half_spread
    else:
        fill_px = _fill_price(snapshot, buying=buying, config=config)
        spread_cost = abs(float(fill_px) - snapshot.mid)
    return spread_cost + config.extra_cost_per_share + config.commission_per_share


def _impact_per_share(turnover_units: float, config: ExecutionConfig) -> float:
    if config.fill_level == FillLevel.QUOTE_SIDE_PLUS_IMPACT and config.impact_model.kind == "linear":
        return config.impact_model.coef_per_unit * float(turnover_units)
    return 0.0


def simulate_transition(
    state: PositionState,
    action_position: float,
    now: MarketSnapshot,
    fill: MarketSnapshot,
    nxt: MarketSnapshot,
    *,
    is_terminal: bool,
    config: ExecutionConfig,
) -> TransitionOutcome:
    """Pure per-step transition P&L. ``now``/``fill``/``nxt`` are point-in-time dataset snapshots at the
    current, fill, and next bars; ``state``/``action_position``/``is_terminal`` come from the env. With
    ``fill_level=delayed_close`` this reproduces the intraday reward exactly."""
    scale = config.trade_scale
    old = float(state.position)
    new = float(action_position)
    turnover_units = abs(new - old)

    old_latency_return = old * (fill.mid - now.mid) * scale  # old leg keeps earning until its fill
    new_interval_return = new * (nxt.mid - fill.mid) * scale  # new leg earns from fill to next
    gross_return = old_latency_return + new_interval_return

    entry_cost = 0.0
    impact_cost = 0.0
    entry_fill_price: float | None = None
    if turnover_units > 0.0:
        buying = new > old
        entry_cost = turnover_units * _base_cost_per_share(fill, buying=buying, config=config) * scale
        impact_cost += _impact_per_share(turnover_units, config) * turnover_units * scale
        entry_fill_price = _fill_price(fill, buying=buying, config=config)

    exit_cost = 0.0
    exit_fill_price: float | None = None
    liquidating = is_terminal and config.terminal_policy == TerminalPolicy.LIQUIDATE_AT_NEXT and new != 0.0
    if liquidating:
        units = abs(new)
        buying = new < 0.0  # closing a short is a buy; closing a long is a sell
        exit_cost = units * _base_cost_per_share(nxt, buying=buying, config=config) * scale
        impact_cost += _impact_per_share(units, config) * units * scale
        exit_fill_price = _fill_price(nxt, buying=buying, config=config)

    total_cost = entry_cost + exit_cost + impact_cost
    net_return = gross_return - total_cost

    if liquidating:
        # A terminal liquidation flattens the book: the (now-closed) leg carries no held bars or entry
        # price forward -- otherwise a hold-into-terminal would leave a flat position with stale state.
        next_state = PositionState(position=0.0, bars_held=0, entry_price=None)
    else:
        held = new == old
        next_state = PositionState(
            position=new,
            bars_held=state.bars_held + 1 if held else 0,
            entry_price=state.entry_price if held else entry_fill_price,
        )
    return TransitionOutcome(
        old_latency_return=old_latency_return,
        new_interval_return=new_interval_return,
        gross_return=gross_return,
        entry_fill_price=entry_fill_price,
        exit_fill_price=exit_fill_price,
        entry_cost=entry_cost,
        exit_cost=exit_cost,
        impact_cost=impact_cost,
        total_cost=total_cost,
        net_return=net_return,
        order_legs=turnover_units,
        real_executable_fill_model=config.real_executable_fill_model,
        next_state=next_state,
    )
