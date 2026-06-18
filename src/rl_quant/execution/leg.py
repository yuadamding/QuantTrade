"""Leg-level RETURN-based multi-symbol execution path (the ETF allocation paths: second_context /
minute_to_hour). P&L = sum over symbols of weight * interval_return, with cost in bps of each symbol's
mid charged on traded weight. Distinct from the scalar dollar model; the two share only the fill-level
vocabulary (``FillLevel`` / ``_fill_price``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from rl_quant.execution.fills import (
    MarketSnapshot,
    _fill_price,
)
from rl_quant.execution.types import (
    ExecutionConfig,
    SwitchFillPolicy,
    TerminalPolicy,
)
from rl_quant.execution.validation import (
    _coerce_finite,
    _coerce_finite_nonnegative,
    _coerce_positive_price,
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
        object.__setattr__(self, "mid", _coerce_positive_price(f"{self.symbol}.mid", self.mid))
        object.__setattr__(self, "half_spread", _coerce_finite_nonnegative(f"{self.symbol}.half_spread", self.half_spread))
        for name in ("interval_return", "latency_return"):
            object.__setattr__(self, name, _coerce_finite(f"{self.symbol}.{name}", getattr(self, name)))
        for name in ("best_bid", "best_ask"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _coerce_positive_price(f"{self.symbol}.{name}", value))
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

    def __post_init__(self) -> None:
        # Normalize and fail closed: reject duplicate symbols (ambiguous weight_of), non-finite weights,
        # and an explicit "CASH" holding (cash is the implicit unallocated remainder). Drop ~0 weights so
        # equality and symbol enumeration are canonical.
        seen: set[str] = set()
        clean: list[tuple[str, float]] = []
        for symbol, weight in self.weights:
            symbol = str(symbol)
            if symbol.upper() == "CASH":
                raise ValueError("CASH must be implicit (the unallocated remainder), not a holding.")
            if symbol in seen:
                raise ValueError(f"duplicate holding symbol: {symbol}.")
            seen.add(symbol)
            w = _coerce_finite(f"holding[{symbol}]", weight)
            if abs(w) > 1e-12:
                clean.append((symbol, w))
        object.__setattr__(self, "weights", tuple(clean))

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
    fee_bps: float = 0.0  # flat per-leg fee from weight_cost (0 unless filled)
    impact_bps: float = 0.0  # size-dependent impact from weight_cost (0 unless filled)
    total_cost_bps: float = 0.0  # spread_bps + fee_bps + impact_bps (the charged bps for this leg)

    def __post_init__(self) -> None:
        for name in ("traded_weight", "spread_bps", "fee_bps", "impact_bps"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"ExecutionLeg.{name} must be finite and non-negative; got {value!r}.")
        if not math.isclose(
            self.total_cost_bps, self.spread_bps + self.fee_bps + self.impact_bps, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("ExecutionLeg.total_cost_bps must equal spread_bps + fee_bps + impact_bps.")
        # An unfilled leg never carries a fill price. NOTE: a FILLED leg may still have fill_price=None at a
        # PROXY level (delayed_close / mid_plus_spread), so the implication only runs one way.
        if self.fill_status != FillStatus.FILLED and self.fill_price is not None:
            raise ValueError("an unfilled leg must not carry a fill_price.")


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
    # Decomposed reportability status so a downstream evaluator never has to read intent out of the P&L
    # numbers (which are still produced for diagnostics even when the transition is non-reportable):
    #   valuation_complete  -- every held/executed non-zero position had a quote to value it
    #   execution_complete  -- every requested trade leg (and terminal liquidation leg) actually filled
    #   impact_applied      -- the leg engine charged a positive (weight_cost) impact on this transition
    # real_executable_fill_model is the AND of crossable-quote fills + valuation_complete + execution_complete.
    # impact_applied is a SEPARATE axis (impact is not required for a crossable fill to be "real").
    valuation_complete: bool = True
    execution_complete: bool = True
    impact_applied: bool = False
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isclose(
            self.net_pnl, self.gross_mark_pnl - self.realized_execution_cost, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("ActionTransitionOutcome.net_pnl must equal gross_mark_pnl - realized_execution_cost.")
        if self.real_executable_fill_model and not (self.valuation_complete and self.execution_complete):
            raise ValueError("real_executable_fill_model implies valuation_complete and execution_complete.")


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
    # bps fee/impact apply ONLY to a leg that actually fills; a blocked (MISSING_QUOTE) leg costs nothing.
    if status == FillStatus.FILLED:
        fee_bps = config.weight_cost.fee_bps
        impact_bps = config.weight_cost.impact_bps(traded)
    else:
        fee_bps = 0.0
        impact_bps = 0.0
    total_cost_bps = spread_bps + fee_bps + impact_bps
    cost = traded * total_cost_bps / 1e4
    leg = ExecutionLeg(
        symbol=symbol, side=side, traded_weight=traded, mark_before=prev_w,
        mid_at_fill=quote.mid, fill_price=fill_price, spread_bps=spread_bps, fill_status=status,
        fee_bps=fee_bps, impact_bps=impact_bps, total_cost_bps=total_cost_bps,
    )
    return leg, cost


def _leg_fillable(prev_w: float, tgt_w: float, quote: SymbolQuote | None, config: ExecutionConfig) -> bool:
    """Whether a single change leg would FILL under the config's fill model, WITHOUT committing it. Mirrors
    _make_leg's fill logic: quote-side needs the crossed side present (ask for a buy, bid for a sell); proxy
    levels fill on the mid alone. Used by the ATOMIC_SWITCH pre-pass."""
    if quote is None:
        return False
    if not config.uses_crossable_quote_fills:
        return True
    return (quote.best_ask if tgt_w > prev_w else quote.best_bid) is not None


def simulate_action_transition(
    prev_holdings: Holdings,
    target_holdings: Holdings,
    market_by_symbol: dict[str, SymbolQuote],
    config: ExecutionConfig,
    *,
    is_terminal: bool = False,
) -> ActionTransitionOutcome:
    """Return-based, per-symbol leg-level transition P&L (see the module note above). Each prior holding
    earns its latency leg (now->fill); every symbol whose weight changes emits a BUY/SELL leg filled on ITS
    OWN book, costed by the per-leg spread in bps of that symbol's mid; and the EXECUTED (post-fill) holding
    earns its interval leg (fill->next).

    Fail-closed semantics (a trade that cannot fill does NOT move the book): a symbol with no quote, or a
    quote-side leg with no bid/ask (MISSING_QUOTE), keeps its PRIOR weight -- the transition does not silently
    teleport to the target -- and the outcome is flagged non-(real-executable) with a warning. A held position
    (no trade) whose symbol has no quote cannot be VALUED either, so it likewise flags the outcome rather than
    silently earning a 0 return. Terminal liquidation likewise only flattens symbols whose exit leg fills; a
    missing terminal quote keeps the holding and flags the outcome. Legs are ordered exits-before-entries.

    config.switch_fill_policy controls partial-switch behavior. INDEPENDENT_LEGS (default) fills each leg on
    its OWN book independently, so a partially-fillable switch is NOT weight-conserving: if only one leg of an
    A->B switch fills, the book is left over-allocated (bought B, could not sell A) or stranded in cash (sold
    A, could not buy B) -- the honest consequence of independent fills, always flagged
    (real_executable_fill_model=False + a missing_quote warning). ATOMIC_SWITCH is all-or-nothing: if ANY
    required leg cannot fill, NONE execute and the prior holdings are kept (an 'atomic_switch_blocked' warning
    is added) -- the recommended policy for reportable allocator evaluation. (Terminal liquidation is governed
    separately and stays per-symbol fail-closed.) No trainer calls this yet -- it changes no reward."""
    eps = 1e-12
    symbols = sorted(set(prev_holdings.symbols()) | set(target_holdings.symbols()))
    legs: list[ExecutionLeg] = []
    realized_cost = 0.0
    warnings: list[str] = []
    real = config.real_executable_fill_model
    valuation_complete = True  # could every held/executed non-zero position be valued?
    execution_complete = True  # did every requested trade / terminal liquidation leg fill?

    def warn(message: str) -> None:
        # Dedup so the same symbol flagged on multiple legs (latency / interval / trade) warns once.
        if message not in warnings:
            warnings.append(message)

    # The old (prior) position earns its now->fill latency leg on every held symbol that has a quote. A
    # held, non-zero position whose symbol has NO quote cannot be valued -- it is an UNVALUED position, not
    # a zero-return event -- so fail closed (flag + warn) instead of silently crediting it a 0 return.
    old_latency = 0.0
    for symbol in symbols:
        prev_w = prev_holdings.weight_of(symbol)
        quote = market_by_symbol.get(symbol)
        if quote is not None:
            old_latency += prev_w * quote.latency_return
        elif abs(prev_w) > eps:
            warn(f"missing_quote:{symbol}")
            real = False
            valuation_complete = False

    # executed[symbol] = weight ACTUALLY held after fills (== target only where the leg fills).
    executed: dict[str, float] = {}
    changes: list[tuple[str, float, float]] = []
    for symbol in symbols:
        prev_w = prev_holdings.weight_of(symbol)
        tgt_w = target_holdings.weight_of(symbol)
        if abs(tgt_w - prev_w) > eps:
            changes.append((symbol, prev_w, tgt_w))
        else:
            executed[symbol] = tgt_w  # no trade required (== prev)
    # Exits (weight shrinking toward 0 / flipping) before entries; alphabetical within each for determinism.
    sells = sorted(c for c in changes if c[2] < c[1])
    buys = sorted(c for c in changes if c[2] > c[1])
    ordered = sells + buys
    if config.switch_fill_policy == SwitchFillPolicy.ATOMIC_SWITCH and ordered:
        # All-or-nothing: if ANY required leg cannot fill, execute NONE of them and keep prior holdings
        # (a partial switch would over-allocate or strand cash). The transition is non-reportable.
        unfillable = [c for c in ordered if not _leg_fillable(c[1], c[2], market_by_symbol.get(c[0]), config)]
        if unfillable:
            for symbol, _prev_w, _tgt_w in unfillable:
                warn(f"missing_quote:{symbol}")
            warn("atomic_switch_blocked")
            real = False
            execution_complete = False
            for symbol, prev_w, _tgt_w in ordered:
                executed[symbol] = prev_w  # blocked transition: every leg keeps its prior weight
            ordered = []
    for symbol, prev_w, tgt_w in ordered:
        quote = market_by_symbol.get(symbol)
        if quote is None:
            warn(f"missing_quote:{symbol}")
            real = False
            execution_complete = False
            executed[symbol] = prev_w  # blocked: no quote -> trade did not fill, keep prior weight
            continue
        leg, cost = _make_leg(symbol, prev_w, tgt_w, quote, config)
        legs.append(leg)
        if leg.fill_status != FillStatus.FILLED:
            warn(f"missing_quote:{symbol}")
            real = False
            execution_complete = False
            executed[symbol] = prev_w  # blocked: unfilled leg keeps prior weight (no teleport, no cost)
        else:
            realized_cost += cost
            executed[symbol] = tgt_w

    # The executed position earns its fill->next interval leg. An executed, non-zero position whose symbol
    # has NO quote cannot be valued -> fail closed (same rule as the latency leg), never a silent 0 return.
    new_interval = 0.0
    for symbol, weight in executed.items():
        if abs(weight) <= eps:
            continue
        quote = market_by_symbol.get(symbol)
        if quote is not None:
            new_interval += weight * quote.interval_return
        else:
            warn(f"missing_quote:{symbol}")
            real = False
            valuation_complete = False

    next_state = Holdings(tuple((s, w) for s, w in executed.items() if abs(w) > eps))
    if is_terminal and config.terminal_policy == TerminalPolicy.LIQUIDATE_AT_NEXT:
        remaining: list[tuple[str, float]] = []
        for symbol, weight in executed.items():
            if abs(weight) <= eps:
                continue
            quote = market_by_symbol.get(symbol)
            if quote is None:
                warn(f"terminal_missing_quote:{symbol}")
                real = False
                execution_complete = False
                remaining.append((symbol, weight))  # cannot liquidate without a quote -> still held
                continue
            leg, cost = _make_leg(symbol, weight, 0.0, quote, config)
            legs.append(leg)
            if leg.fill_status != FillStatus.FILLED:
                warn(f"terminal_missing_quote:{symbol}")
                real = False
                execution_complete = False
                remaining.append((symbol, weight))
            else:
                realized_cost += cost
        next_state = Holdings(tuple(remaining))

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
        valuation_complete=valuation_complete,
        execution_complete=execution_complete,
        # Transition-ACTUAL, not config-level: impact is "applied" only if at least one leg actually FILLED
        # and carried a positive weight-impact charge. A no-trade / all-blocked / atomic-blocked transition
        # charges no impact even when the config enables it, so it must not be labelled impact-priced (impact_bps
        # is 0 unless weight_cost charged positive impact on a fill, which also implies config.applies_weight_impact).
        impact_applied=any(leg.fill_status == FillStatus.FILLED and leg.impact_bps > 0.0 for leg in legs),
        warnings=tuple(warnings),
    )
