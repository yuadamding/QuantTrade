from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

NANOS_PER_MICROSECOND = 1_000
NANOS_PER_MILLISECOND = 1_000_000
NANOS_PER_SECOND = 1_000_000_000
REGULAR_SESSION_START_NS = (9 * 60 * 60 + 30 * 60) * NANOS_PER_SECOND
REGULAR_SESSION_END_NS = 16 * 60 * 60 * NANOS_PER_SECOND


def parse_time_to_ns(value: str) -> int:
    hour_text, second_text, rest = value.split(":", 2)
    if "." in rest:
        second_text, fractional_text = rest.split(".", 1)
    else:
        second_text, fractional_text = rest, ""

    fractional_ns = int((fractional_text + "000000000")[:9])
    total_seconds = int(hour_text) * 3600 + int(second_text) * 60 + int(second_text)
    return total_seconds * NANOS_PER_SECOND + fractional_ns


def format_second_of_day(second_of_day: int) -> str:
    hour = second_of_day // 3600
    minute = (second_of_day % 3600) // 60
    second = second_of_day % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def format_time_of_day_ns(timestamp_ns: int) -> str:
    total_seconds = timestamp_ns // NANOS_PER_SECOND
    fractional_ns = timestamp_ns % NANOS_PER_SECOND
    base = format_second_of_day(total_seconds)
    if fractional_ns == 0:
        return base
    fractional_text = f"{fractional_ns:09d}".rstrip("0")
    return f"{base}.{fractional_text}"


def format_bucket_label(bucket_ns: int) -> str:
    if bucket_ns <= 0:
        raise ValueError("bucket_ns must be positive")
    if bucket_ns % NANOS_PER_SECOND == 0:
        return f"{bucket_ns // NANOS_PER_SECOND}s"
    if bucket_ns % NANOS_PER_MILLISECOND == 0:
        return f"{bucket_ns // NANOS_PER_MILLISECOND}ms"
    if bucket_ns % NANOS_PER_MICROSECOND == 0:
        return f"{bucket_ns // NANOS_PER_MICROSECOND}us"
    return f"{bucket_ns}ns"


def parse_float(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_int(value: str) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return 0


def in_session(timestamp_ns: int, session: str) -> bool:
    if session == "all":
        return True
    if session != "regular":
        raise ValueError(f"Unsupported session filter: {session}")
    return REGULAR_SESSION_START_NS <= timestamp_ns < REGULAR_SESSION_END_NS


def list_date_files(
    input_dir: Path,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[Path]:
    files = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if start_date:
        files = [path for path in files if path.stem >= start_date]
    if end_date:
        files = [path for path in files if path.stem <= end_date]
    return files


@dataclass
class VenueQuote:
    bid: float = 0.0
    bid_size_lots: int = 0
    ask: float = 0.0
    ask_size_lots: int = 0


@dataclass
class NbboSnapshot:
    timestamp_ns: int
    best_bid: float
    best_ask: float
    bid_depth_lots: int
    ask_depth_lots: int
    spread: float
    mid: float
    microprice: float
    imbalance: float
    locked: bool
    crossed: bool


class NbboBuilder:
    """Maintains the latest top-of-book quote for each exchange."""

    def __init__(self) -> None:
        self._quotes: Dict[str, VenueQuote] = {}

    def update(
        self,
        *,
        exchange: str,
        bid: float,
        bid_size_lots: int,
        ask: float,
        ask_size_lots: int,
        timestamp_ns: int,
    ) -> Optional[NbboSnapshot]:
        if not exchange:
            return None

        current = self._quotes.get(exchange, VenueQuote())

        # These quote files contain many one-sided updates. Preserve the
        # untouched side unless the row clears both sides explicitly.
        # KNOWN LIMITATION: there is no per-side staleness expiry, so a venue that stops
        # quoting one side keeps contributing an arbitrarily old price to the NBBO, which can
        # manufacture locked/crossed books. A crossed book (best_bid >= best_ask) is FLAGGED
        # (crossed/locked below) but NOT repaired here; cost consumers must clamp the spread to
        # >= 0 (see intraday_data.load_split) so a crossed book cannot become a negative cost.
        if bid <= 0.0 and ask <= 0.0 and bid_size_lots <= 0 and ask_size_lots <= 0:
            current = VenueQuote()
        else:
            if bid > 0.0 or bid_size_lots > 0:
                current.bid = bid
                current.bid_size_lots = bid_size_lots
            if ask > 0.0 or ask_size_lots > 0:
                current.ask = ask
                current.ask_size_lots = ask_size_lots

        self._quotes[exchange] = current

        best_bid = 0.0
        best_ask = 0.0
        bid_depth_lots = 0
        ask_depth_lots = 0

        for quote in self._quotes.values():
            if quote.bid > best_bid and quote.bid_size_lots > 0:
                best_bid = quote.bid
            if quote.ask_size_lots > 0 and quote.ask > 0.0:
                if best_ask == 0.0 or quote.ask < best_ask:
                    best_ask = quote.ask

        if best_bid <= 0.0 or best_ask <= 0.0:
            return None

        for quote in self._quotes.values():
            if quote.bid_size_lots > 0 and quote.bid == best_bid:
                bid_depth_lots += quote.bid_size_lots
            if quote.ask_size_lots > 0 and quote.ask == best_ask:
                ask_depth_lots += quote.ask_size_lots

        total_depth = bid_depth_lots + ask_depth_lots
        if total_depth <= 0:
            return None

        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0
        microprice = (best_ask * bid_depth_lots + best_bid * ask_depth_lots) / total_depth
        imbalance = bid_depth_lots / total_depth

        return NbboSnapshot(
            timestamp_ns=timestamp_ns,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth_lots=bid_depth_lots,
            ask_depth_lots=ask_depth_lots,
            spread=spread,
            mid=mid,
            microprice=microprice,
            imbalance=imbalance,
            locked=spread == 0.0,
            crossed=spread < 0.0,
        )
