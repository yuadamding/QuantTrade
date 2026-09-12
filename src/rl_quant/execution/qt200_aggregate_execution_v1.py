"""Explicit aggregate-development execution proxy, not observed market fills.

Orders are fixed before a ten-minute window. Each minute uses its OPEN price
and at most 2% of its reported volume, in whole newly traded shares. Missing
windows cannot execute. Higher costs may change affordable quantities; no
monotonic-wealth assertion is structural validity. These pure kernels confer
no source, historical-identity, native-V5 or training authorization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, localcontext
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

D = Decimal
ET = ZoneInfo("America/New_York")
PARTICIPATION = D("0.02")
COST_RUNGS = (D("10"), D("20"), D("40"))


def number(value, *, positive=False) -> Decimal:
    if isinstance(value, (bool, float)) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("Accounting numbers require exact decimal strings/integers")
    try:
        result = D(value)
    except InvalidOperation as exc:
        raise ValueError("Malformed decimal") from exc
    if not result.is_finite() or result < 0 or (positive and not result > 0):
        raise ValueError("Nonfinite/negative accounting input")
    return result


def day(value: str) -> str:
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Noncanonical session date")
    return value


def _identity(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("Missing instrument/event identity")


@dataclass(frozen=True)
class Receivable:
    event_id: str
    instrument: str
    payable_date: str
    amount: Decimal

    def __post_init__(self):
        _identity(self.event_id)
        _identity(self.instrument)
        day(self.payable_date)
        if not isinstance(self.amount, Decimal):
            raise ValueError("Receivable amount must be Decimal")
        number(self.amount, positive=True)


@dataclass(frozen=True)
class Book:
    cash: Decimal
    holdings: tuple[tuple[str, Decimal], ...] = ()
    receivables: tuple[Receivable, ...] = ()
    applied_events: tuple[str, ...] = ()
    action_date: str | None = None

    def __post_init__(self):
        if not isinstance(self.cash, Decimal):
            raise ValueError("Book cash must be Decimal")
        number(self.cash)
        if any(type(value) is not tuple for value in (self.holdings, self.receivables, self.applied_events)):
            raise ValueError("Book inventories must be immutable tuples")
        names = [ticker for ticker, _ in self.holdings]
        if names != sorted(set(names)):
            raise ValueError("Holdings must be unique and sorted")
        for ticker, shares in self.holdings:
            _identity(ticker)
            if not isinstance(shares, Decimal):
                raise ValueError("Book shares must be Decimal")
            number(shares, positive=True)
        ids = [r.event_id for r in self.receivables]
        if ids != sorted(set(ids)) or list(self.applied_events) != sorted(set(self.applied_events)):
            raise ValueError("Duplicate/unordered event history")
        if not set(ids).issubset(self.applied_events):
            raise ValueError("Receivable lacks its applied entitlement event")
        for key in self.applied_events:
            _identity(key)
        if self.action_date is not None:
            day(self.action_date)


@dataclass(frozen=True)
class Split:
    event_id: str
    instrument: str
    effective_date: str
    shares_from: Decimal
    shares_to: Decimal


@dataclass(frozen=True)
class Dividend:
    event_id: str
    instrument: str
    ex_date: str
    payable_date: str
    cash_per_share: Decimal
    currency: str = "USD"


def apply_actions(book: Book, session: str, *, splits: Sequence[Split] = (),
                  dividends: Sequence[Dividend] = ()) -> Book:
    """Before trading: split shares; book ex-date entitlements, pay when due.

    Same-issue split/dividend on one date needs a documented per-share basis
    and is deliberately unsupported here. Event identity/issue applicability
    must be established by the source adapter, not inferred in this kernel.
    """
    day(session)
    if book.action_date is not None and session <= book.action_date:
        raise ValueError("Actions must advance the economic calendar exactly once")
    ids = [e.event_id for e in (*splits, *dividends)]
    if len(ids) != len(set(ids)) or set(ids).intersection(book.applied_events):
        raise ValueError("Duplicate/replayed economic event")
    if (len({e.instrument for e in splits}) != len(splits)
            or {e.instrument for e in splits}.intersection(e.instrument for e in dividends)):
        raise ValueError("Ambiguous same-day share basis")
    for event in (*splits, *dividends):
        _identity(event.event_id)
        _identity(event.instrument)
    for event in splits:
        if day(event.effective_date) != session:
            raise ValueError("Split applied on wrong date")
        number(event.shares_from, positive=True)
        number(event.shares_to, positive=True)
    for event in dividends:
        if (day(event.ex_date) != session or day(event.payable_date) < session or event.currency != "USD"):
            raise ValueError("Unsupported dividend timing/currency")
        number(event.cash_per_share, positive=True)
    with localcontext() as context:
        context.prec = 34
        shares = dict(book.holdings)
        for event in splits:
            if event.instrument in shares:
                shares[event.instrument] *= number(event.shares_to) / number(event.shares_from)
        receivables = list(book.receivables)
        for event in dividends:
            quantity = shares.get(event.instrument, D(0))
            if quantity:
                receivables.append(Receivable(event.event_id, event.instrument, event.payable_date,
                                              quantity * number(event.cash_per_share)))
        paid = sum((r.amount for r in receivables if r.payable_date <= session), D(0))
        unpaid = tuple(sorted((r for r in receivables if r.payable_date > session), key=lambda r: r.event_id))
        return Book(book.cash + paid, tuple(sorted(shares.items())), unpaid,
                    tuple(sorted((*book.applied_events, *ids))), session)


def equity(book: Book, prices: Mapping[str, Decimal]) -> Decimal:
    with localcontext() as context:
        context.prec = 34
        marked = sum((quantity * number(prices[ticker], positive=True) for ticker, quantity in book.holdings), D(0))
        return book.cash + marked + sum((r.amount for r in book.receivables), D(0))


@dataclass(frozen=True)
class Fill:
    instrument: str
    minute_start_ns: int
    signed_shares: Decimal
    price: Decimal
    fee: Decimal


@dataclass(frozen=True)
class Execution:
    book: Book
    fills: tuple[Fill, ...]
    unfilled_shares: tuple[tuple[str, Decimal], ...]
    unsupported_windows: tuple[str, ...]
    gross_traded_notional: Decimal
    transaction_costs: Decimal
    execution_model: str = "minute-open-whole-share-participation-proxy-v1"


def validate_window(rows: Sequence[dict], session: str) -> tuple[tuple[int, Decimal, Decimal], ...]:
    day(session)
    result = []
    for row in rows:
        stamp = row["bar_start_ms"]
        if type(stamp) is not int or stamp % 60_000:
            raise ValueError("Minute timestamp must be exact")
        t = datetime.fromtimestamp(stamp // 1000, timezone.utc).astimezone(ET)
        if t.date().isoformat() != session or t.hour != 9 or not 35 <= t.minute < 45:
            raise ValueError("Minute outside registered execution window")
        o, h, low, c = [number(row[k], positive=True) for k in "ohlc"]
        if not low <= min(o, c) <= max(o, c) <= h:
            raise ValueError("Malformed OHLC")
        v = number(row["v"])
        result.append((stamp * 1_000_000, o, v))
    stamps = [r[0] for r in result]
    if stamps != sorted(set(stamps)):
        raise ValueError("Duplicate/out-of-order execution minutes")
    return tuple(result)


def execute_window(book: Book, orders: Mapping[str, int], windows: Mapping[str, Sequence[dict]], *,
                   session: str, committed_at_ns: int, cost_bps: Decimal = D("20")) -> Execution:
    """One fixed pre-window order population; deterministic proportional buys.

    Sells settle first within each minute. Cash-constrained buys are scaled
    together, then floored; unused residual cash is retained. Incomplete
    windows stay explicitly unsupported, not fabricated zero liquidity.
    """
    day(session)
    rate = number(cost_bps) / D(10_000)
    if cost_bps not in COST_RUNGS or type(committed_at_ns) is not int:
        raise ValueError("Unregistered cost rung or commitment clock")
    start = int(datetime.fromisoformat(session + "T09:35:00").replace(tzinfo=ET).timestamp()) * 10**9
    if not 0 <= committed_at_ns < start:
        raise ValueError("Orders must be committed before window data")
    if book.action_date != session:
        raise ValueError("Apply session corporate actions before execution")
    quantities = dict(book.holdings)
    for ticker, quantity in orders.items():
        _identity(ticker)
        if type(quantity) is not int or D(quantity) < -quantities.get(ticker, D(0)):
            raise ValueError("Noninteger order or short sale")
    observed = {}
    for ticker in orders:
        rows = windows.get(ticker, ())
        if any(row.get("ticker", ticker) != ticker for row in rows):
            raise ValueError("Wrong instrument window")
        observed[ticker] = validate_window(rows, session)
    unsupported = tuple(sorted(t for t, rows in observed.items() if len(rows) != 10))
    remaining = {ticker: D(quantity) for ticker, quantity in orders.items()}
    fills, cash = [], book.cash
    with localcontext() as context:
        context.prec = 34
        for index in range(10):
            available = {t: (rows[index][0], rows[index][1], (rows[index][2] * PARTICIPATION).to_integral_value(rounding=ROUND_FLOOR))
                         for t, rows in observed.items() if t not in unsupported}
            for ticker in sorted(available):
                stamp, price, capacity = available[ticker]
                sold = min(max(-remaining[ticker], D(0)), capacity)
                if sold:
                    fee = sold * price * rate
                    cash += sold * price - fee
                    quantities[ticker] -= sold
                    remaining[ticker] += sold
                    fills.append(Fill(ticker, stamp, -sold, price, fee))
            buys = {t: min(max(remaining[t], D(0)), row[2]) for t, row in available.items()}
            requested_cash = sum((q * available[t][1] * (1 + rate) for t, q in buys.items()), D(0))
            scale = min(D(1), cash / requested_cash) if requested_cash else D(0)
            for ticker in sorted(buys):
                purchased = (buys[ticker] * scale).to_integral_value(rounding=ROUND_FLOOR)
                if purchased:
                    stamp, price, _ = available[ticker]
                    fee = purchased * price * rate
                    cash -= purchased * price + fee
                    quantities[ticker] = quantities.get(ticker, D(0)) + purchased
                    remaining[ticker] -= purchased
                    fills.append(Fill(ticker, stamp, purchased, price, fee))
        result = replace(book, cash=cash, holdings=tuple(sorted((t, q) for t, q in quantities.items() if q)))
        return Execution(result, tuple(fills), tuple(sorted(remaining.items())), unsupported,
                         sum((abs(f.signed_shares) * f.price for f in fills), D(0)),
                         sum((f.fee for f in fills), D(0)))


def liquidate(book: Book, prices: Mapping[str, Decimal], *, cost_bps: Decimal = D("20")) -> tuple[Book, Decimal]:
    """Terminal mark-based liquidation adjustment, explicitly NOT a market fill.

    Dividend receivables retain their value and are not spendable cash. This
    registered stress adjustment does not assert exit capacity at mark prices.
    """
    if cost_bps not in COST_RUNGS:
        raise ValueError("Unregistered liquidation cost rung")
    with localcontext() as context:
        context.prec = 34
        gross = sum((q * number(prices[t], positive=True) for t, q in book.holdings), D(0))
        fee = gross * number(cost_bps) / D(10_000)
        return replace(book, cash=book.cash + gross - fee, holdings=()), fee
