"""Shared transition-P&L execution simulator.

Encodes the real-life cost of moving from a HELD position to a candidate at decision time:

    net = old * (mid_fill - mid_now)      # old position keeps earning until its (latency-delayed) fill
        + new * (mid_next - mid_fill)     # new position earns from its fill to the horizon
        - entry_cost                      # turnover * per-share cost at the FILL bar
        - exit_cost                       # terminal liquidation: |new| * per-share cost at the NEXT bar
    (all dollar terms scaled by trade_scale = trade_lot_size * 100)

This is the per-step decomposition currently inlined three times in ``intraday_dqn`` (env step, eval
loop, pretraining-target builder). The reward engine and the scalar helpers stay torch-free arithmetic
(``transition_pnl`` runs on tensors OR python scalars) so those sites -- and, later, the weight-aware
second-context / minute-to-hour paths -- can share ONE reward engine. The single tensor-only helper is
``fill_indices`` (the vectorized counterpart of scalar ``fill_index``), which is why ``torch`` is imported.

Honesty contract: ``delayed_close`` is a MID-price proxy with a symmetric half-spread cost, NOT a
crossable fill, so ``real_executable_fill_model`` is False and fill prices are ``None``. Only the
``quote_side*`` levels (buy at ask, sell at bid) are real executable fills.

This module changes no trainer's reward on its own; it is wired in (result-preserving for the
``delayed_close`` default) in a separate step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import torch


def _require_finite_nonnegative(name: str, value: float) -> None:
    # NOTE: a bare ``value < 0`` does NOT reject NaN (every NaN comparison is False), so an explicit
    # finiteness check is required to keep NaN/Inf out of cost/reward accounting.
    coerced = float(value)
    if not math.isfinite(coerced) or coerced < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")


def _require_nonnegative_int(name: str, value: object) -> int:
    # Bars/lots must be integer-like: reject bool, and reject a float that is non-finite or has a
    # fractional part instead of silently truncating it (int(1.9) == 1 would rescale every dollar P&L).
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool; got {value!r}.")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be integer-like; got {value!r}.")
    try:
        coerced = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be integer-like; got {value!r}.") from exc
    if coerced < 0:
        raise ValueError(f"{name} must be non-negative; got {value!r}.")
    return coerced


def _require_positive_int(name: str, value: object) -> int:
    coerced = _require_nonnegative_int(name, value)
    if coerced <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}.")
    return coerced


def _require_positive_price(name: str, value: float) -> float:
    # Equity/ETF prices must be strictly positive: a non-positive mid/quote/entry would produce
    # meaningless P&L and costs. (A bare ``value <= 0`` would pass NaN, so check finiteness too.)
    coerced = float(value)
    if not math.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    return coerced


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

    def __post_init__(self) -> None:
        # Only "none"/"linear" are implemented; reject anything else (incl. sqrt/almgren_chriss until
        # built) so a typo like "liner" can't silently DISABLE impact (the simulator treats any
        # non-"linear" kind as zero impact).
        if self.kind not in ("none", "linear"):
            raise ValueError(
                f"impact_model.kind must be 'none' or 'linear' (sqrt/almgren_chriss not yet implemented); "
                f"got {self.kind!r}."
            )
        _require_finite_nonnegative("impact_model.coef_per_unit", self.coef_per_unit)


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
        # negative/NaN latency or cost, a non-positive horizon/lot, or an unknown fill level / policy.
        # Coerce the enums so a valid string is accepted but an unknown one raises clearly (instead of
        # falling through to quote-side logic and crashing later on a missing .value).
        object.__setattr__(self, "fill_level", FillLevel(self.fill_level))
        object.__setattr__(self, "terminal_policy", TerminalPolicy(self.terminal_policy))
        # Coerce-and-store the bar/lot counts so a fractional value can't slip through int() truncation
        # (int(1.9) == 1) and silently rescale every dollar P&L through trade_scale.
        object.__setattr__(self, "latency_steps", _require_nonnegative_int("latency_steps", self.latency_steps))
        object.__setattr__(self, "step_horizon", _require_positive_int("step_horizon", self.step_horizon))
        object.__setattr__(self, "trade_lot_size", _require_positive_int("trade_lot_size", self.trade_lot_size))
        _require_finite_nonnegative("commission_per_share", self.commission_per_share)
        _require_finite_nonnegative("extra_cost_per_share", self.extra_cost_per_share)
        _require_finite_nonnegative("spread_multiplier", self.spread_multiplier)
        object.__setattr__(self, "impact_model", _coerce_impact_model(self.impact_model))
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
        # True only when a positive linear impact is actually applied. __post_init__ guarantees this for
        # quote_side_plus_impact, but it is expressed independently so the report layer can check the
        # impact axis on its own (a separate dimension from "are fills crossable").
        return (
            self.fill_level == FillLevel.QUOTE_SIDE_PLUS_IMPACT
            and self.impact_model.kind == "linear"
            and self.impact_model.coef_per_unit > 0.0
        )

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


@dataclass(frozen=True)
class PositionState:
    position: float  # signed units; intraday {-1, 0, 1}; weight-aware paths may use a weight
    bars_held: int = 0
    entry_price: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.position)):
            raise ValueError(f"position must be finite; got {self.position!r}.")
        object.__setattr__(self, "bars_held", _require_nonnegative_int("bars_held", self.bars_held))
        if self.entry_price is not None:
            _require_positive_price("entry_price", self.entry_price)
        # A flat book holds no open position, so it carries no entry price -- the same invariant the
        # terminal-liquidation fix enforces, applied generally so no path can leave stale entry state.
        if self.position == 0.0 and self.entry_price is not None:
            raise ValueError("a flat position (0) must not carry an entry_price.")


@dataclass(frozen=True)
class MarketSnapshot:
    mid: float
    half_spread: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None

    def __post_init__(self) -> None:
        # A non-positive mid would produce meaningless P&L and costs (and crossing distances).
        _require_positive_price("mid", self.mid)
        # A negative half_spread would turn the cost into a NEGATIVE cost (paying the agent to trade).
        _require_finite_nonnegative("half_spread", self.half_spread)
        for name in ("best_bid", "best_ask"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided; got {value!r}.")
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

    ``latency_steps <= 0`` collapses the fill to the current bar (decision and fill coincide). Mirrors the
    vectorized ``fill_indices`` exactly, so a fill can never be pushed past the holding horizon."""
    if int(step_horizon) <= 0:
        raise ValueError(f"step_horizon must be positive; got {step_horizon}.")
    next_index = now_index + int(step_horizon)
    if int(latency_steps) <= 0:
        return now_index
    return min(now_index + int(latency_steps), next_index)


def fill_indices(now_indices: torch.Tensor, *, step_horizon: int, latency_steps: int) -> torch.Tensor:
    """Vectorized counterpart of scalar :func:`fill_index`: the per-element fill bar for a batch of
    decision bars, ``min(now + latency, now + step_horizon)`` with ``latency <= 0`` collapsing to ``now``.

    The single source of truth for vectorized fill timing (intraday env step + pretraining targets). It is
    byte-for-byte equal to ``fill_index`` applied element-wise (``torch.minimum`` is the per-element min),
    preserves the input tensor's device and dtype, and -- unlike the old inline helper -- carries the same
    ``step_horizon > 0`` guard as the scalar version."""
    if int(step_horizon) <= 0:
        raise ValueError(f"step_horizon must be positive; got {step_horizon}.")
    if int(latency_steps) <= 0:
        return now_indices
    next_indices = now_indices + int(step_horizon)
    return torch.minimum(now_indices + int(latency_steps), next_indices)


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
        # A transition that ends FLAT carries no entry price -- the close-out price is recorded on the
        # outcome (entry_fill_price), not on the next state. (For quote-side fills entry_fill_price is a
        # real bid/ask, so without this a flat next_state would carry a stale exit price.)
        if new == 0.0:
            next_entry_price: float | None = None
        else:
            next_entry_price = state.entry_price if held else entry_fill_price
        next_state = PositionState(
            position=new,
            bars_held=state.bars_held + 1 if held else 0,
            entry_price=next_entry_price,
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


# ---------------------------------------------------------------------------
# Leg-level (multi-asset, RETURN-based) execution
#
# transition_pnl / simulate_transition above are the intraday SIGNED-POSITION DOLLAR model (one
# instrument, P&L = units * mid-diff * scale). The layer below is a DISTINCT, RETURN-based per-symbol
# model that matches the ETF allocation paths (second_context / minute_to_hour): P&L = sum over symbols
# of weight * interval_return, with cost in bps of the symbol's mid charged on traded weight. A
# QQQ -> SQQQ switch decomposes into a SELL-QQQ leg (filled at QQQ's bid) and a BUY-SQQQ leg (filled at
# SQQQ's ask). inverse/leverage is already baked into each symbol's own return label, so there is NO
# leverage multiplier and NO "inverse means sell" here -- fill side is purely delta>0 buy / delta<0 sell.
# The two models share only the fill-level vocabulary (FillLevel/_fill_price/real_executable); they are
# NOT merged. No trainer imports these symbols yet: this changes no existing reward.
# ---------------------------------------------------------------------------


class LegSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class FillStatus(str, Enum):
    FILLED = "filled"
    MISSING_QUOTE = "missing_quote"  # a quote-side fill was requested but bid/ask was absent
    BLOCKED = "blocked"  # symbol unavailable (halt / no borrow / constraint refusal)


@dataclass(frozen=True)
class SymbolQuote:
    """Point-in-time per-symbol dataset values at the FILL bar, plus the realized return segments the
    weights earn: ``interval_return`` is the fill->next fractional return the NEW weight earns (the
    action_return label); ``latency_return`` is the now->fill return the OLD weight earns (0 with no
    latency)."""

    symbol: str
    mid: float
    interval_return: float = 0.0
    latency_return: float = 0.0
    half_spread: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.mid)) or self.mid <= 0.0:
            raise ValueError(f"{self.symbol}: mid must be finite and positive; got {self.mid!r}.")
        _require_finite_nonnegative(f"{self.symbol}.half_spread", self.half_spread)
        for name in ("interval_return", "latency_return"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{self.symbol}.{name} must be finite; got {getattr(self, name)!r}.")
        for name in ("best_bid", "best_ask"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{self.symbol}.{name} must be finite when provided; got {value!r}.")
        if self.best_bid is not None and self.best_ask is not None:
            if self.best_bid > self.best_ask:
                raise ValueError(f"{self.symbol}: best_bid ({self.best_bid}) must be <= best_ask ({self.best_ask}).")
            # Enforce the SAME mid-inside-quote invariant as MarketSnapshot so _market() can never forward
            # an out-of-quote mid: otherwise _make_leg would catch the MarketSnapshot ValueError and
            # silently downgrade a (malformed-but-present) quote to a MISSING_QUOTE fill.
            if not (self.best_bid <= self.mid <= self.best_ask):
                raise ValueError(
                    f"{self.symbol}: mid ({self.mid}) must lie within [best_bid, best_ask] "
                    f"([{self.best_bid}, {self.best_ask}])."
                )

    def _market(self) -> MarketSnapshot:
        return MarketSnapshot(mid=self.mid, half_spread=self.half_spread, best_bid=self.best_bid, best_ask=self.best_ask)


@dataclass(frozen=True)
class Holdings:
    """Signed weight per symbol (CASH is the implicit unallocated remainder). The single-slot ETF case
    (at most one non-cash symbol) is what the current trainers use, but the vector form is representable
    for a future multi-instrument path without a schema change."""

    weights: tuple[tuple[str, float], ...] = ()

    @classmethod
    def single_slot(cls, symbol: str | None, weight: float) -> "Holdings":
        if symbol is None or abs(float(weight)) <= 1e-12:
            return cls(())
        return cls(((str(symbol), float(weight)),))

    def weight_of(self, symbol: str) -> float:
        return next((w for s, w in self.weights if s == symbol), 0.0)

    def symbols(self) -> tuple[str, ...]:
        return tuple(s for s, _ in self.weights)


@dataclass(frozen=True)
class ExecutionLeg:
    symbol: str
    side: LegSide
    traded_weight: float  # |delta weight| for this symbol; >= 0
    mark_before: float  # this symbol's signed weight BEFORE the transition
    mid_at_fill: float
    fill_price: float | None  # None at proxy fill levels (delayed_close / mid_plus_spread)
    spread_bps: float
    fill_status: FillStatus


@dataclass(frozen=True)
class ActionTransitionOutcome:
    legs: tuple[ExecutionLeg, ...]
    old_position_latency_pnl: float
    new_position_interval_pnl: float
    gross_mark_pnl: float
    realized_execution_cost: float
    net_pnl: float
    next_state: Holdings
    real_executable_fill_model: bool
    warnings: tuple[str, ...] = ()


def _make_leg(symbol: str, prev_w: float, tgt_w: float, quote: SymbolQuote, config: ExecutionConfig) -> tuple[ExecutionLeg, float]:
    delta = tgt_w - prev_w
    side = LegSide.BUY if delta > 0 else LegSide.SELL
    traded = abs(delta)
    status = FillStatus.FILLED
    fill_price: float | None = None
    # Proxy fill levels price the leg at the symmetric half-spread; quote-side prices at the crossing
    # distance |fill - mid| (buy at ask, sell at bid).
    spread_bps = (config.spread_multiplier * quote.half_spread) / quote.mid * 1e4
    if config.real_executable_fill_model:
        try:
            fill_price = _fill_price(quote._market(), buying=side == LegSide.BUY, config=config)
            spread_bps = abs(float(fill_price) - quote.mid) / quote.mid * 1e4
        except ValueError:
            status = FillStatus.MISSING_QUOTE
            spread_bps = 0.0
    cost = traded * spread_bps / 1e4
    leg = ExecutionLeg(
        symbol=symbol, side=side, traded_weight=traded, mark_before=prev_w,
        mid_at_fill=quote.mid, fill_price=fill_price, spread_bps=spread_bps, fill_status=status,
    )
    return leg, cost


def simulate_action_transition(
    prev_holdings: Holdings,
    target_holdings: Holdings,
    market_by_symbol: dict[str, SymbolQuote],
    config: ExecutionConfig,
    *,
    is_terminal: bool = False,
) -> ActionTransitionOutcome:
    """Return-based, per-symbol leg-level transition P&L (see the module note above). Each prior holding
    earns its latency leg (now->fill) and each target holding earns its interval leg (fill->next); every
    symbol whose weight changes emits a BUY/SELL leg filled on ITS OWN book, costed by the per-leg spread
    in bps of that symbol's mid. A missing quote at a quote-side level flags the leg MISSING_QUOTE and
    marks the whole outcome non-(real-executable). No trainer calls this yet -- it changes no reward."""
    symbols = sorted(set(prev_holdings.symbols()) | set(target_holdings.symbols()))
    old_latency = 0.0
    new_interval = 0.0
    legs: list[ExecutionLeg] = []
    realized_cost = 0.0
    warnings: list[str] = []
    real = config.real_executable_fill_model

    for symbol in symbols:
        quote = market_by_symbol.get(symbol)
        prev_w = prev_holdings.weight_of(symbol)
        tgt_w = target_holdings.weight_of(symbol)
        if quote is None:
            if abs(tgt_w - prev_w) > 1e-12:
                warnings.append(f"missing_quote:{symbol}")
                real = False
            continue
        old_latency += prev_w * quote.latency_return
        new_interval += tgt_w * quote.interval_return
        if abs(tgt_w - prev_w) > 1e-12:
            leg, cost = _make_leg(symbol, prev_w, tgt_w, quote, config)
            legs.append(leg)
            realized_cost += cost
            if leg.fill_status == FillStatus.MISSING_QUOTE:
                real = False
                warnings.append(f"missing_quote:{symbol}")

    next_state = target_holdings
    if is_terminal and config.terminal_policy == TerminalPolicy.LIQUIDATE_AT_NEXT:
        # Liquidate every target holding to flat (one extra exit leg per symbol).
        for symbol in target_holdings.symbols():
            tgt_w = target_holdings.weight_of(symbol)
            quote = market_by_symbol.get(symbol)
            if abs(tgt_w) > 1e-12 and quote is not None:
                leg, cost = _make_leg(symbol, tgt_w, 0.0, quote, config)
                legs.append(leg)
                realized_cost += cost
        next_state = Holdings(())

    gross = old_latency + new_interval
    return ActionTransitionOutcome(
        legs=tuple(legs),
        old_position_latency_pnl=old_latency,
        new_position_interval_pnl=new_interval,
        gross_mark_pnl=gross,
        realized_execution_cost=realized_cost,
        net_pnl=gross - realized_cost,
        next_state=next_state,
        real_executable_fill_model=real,
        warnings=tuple(warnings),
    )
