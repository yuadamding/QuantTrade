"""Hash-bound event coverage and session calendar for bounded second-bar research.

This validates a reviewed normalized input document and every cited source.
It does not certify historical issue identity or reconstruct data vintages.
An absent document is not equivalent to a checked event-free interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondCatalog, _json, _read
from rl_quant.execution.qt200_aggregate_execution_v1 import Book, Dividend, Split, apply_actions

ECONOMIC_SCHEMA = "rl-quant.raw-second-economic-inputs-v1"


@dataclass(frozen=True)
class SecondSession:
    session_date: str
    open_ms: int
    close_ms: int

    def __post_init__(self):
        if (type(self.open_ms) is not int or type(self.close_ms) is not int
                or self.open_ms % 1000 or self.close_ms % 1000 or not 0 <= self.open_ms < self.close_ms):
            raise ValueError("Invalid session clock")
        for stamp in (self.open_ms, self.close_ms):
            day = datetime.fromtimestamp(stamp / 1000, timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
            if day != self.session_date:
                raise ValueError("Session date does not match its Eastern clock")


@dataclass(frozen=True)
class SecondEconomicInputs:
    path: str
    sha256: str

    def load(self, catalog: RawSecondCatalog) -> tuple[tuple[Split, ...], tuple[Dividend, ...], tuple[SecondSession, ...]]:
        body = _json(_read(Path(self.path), self.sha256))
        if (set(body) != {"schema", "asset_ids", "coverage_start_ms", "coverage_end_ms", "basis",
                         "sessions", "splits", "dividends", "unsupported_events", "sources"}
                or body["schema"] != ECONOMIC_SCHEMA or tuple(body["asset_ids"]) != catalog.asset_ids
                or body["basis"] not in ("synthetic-complete-census", "reviewed-normalized-provider-census")
                or body["unsupported_events"] or not body["sources"]):
            raise ValueError("Incomplete/unsupported corporate-action and calendar coverage")
        for source in body["sources"]:
            if set(source) != {"path", "sha256"}:
                raise ValueError("Economic evidence requires exact source references")
            _read(Path(source["path"]), source["sha256"])
        start, end = body["coverage_start_ms"], body["coverage_end_ms"]
        if (type(start) is not int or type(end) is not int or not 0 <= start < end
                or start > catalog.windows[0].decision_ms or end < catalog.windows[-1].decision_ms):
            raise ValueError("Economic coverage does not span the complete scored interval")
        sessions = tuple(SecondSession(**row) for row in body["sessions"])
        if not sessions or [s.open_ms for s in sessions] != sorted(set(s.open_ms for s in sessions)):
            raise ValueError("Missing or unordered session coverage")
        if any(left.close_ms >= right.open_ms for left, right in zip(sessions, sessions[1:])):
            raise ValueError("Overlapping session coverage")
        splits = tuple(Split(**{**row, "shares_from": Decimal(row["shares_from"]),
                               "shares_to": Decimal(row["shares_to"])}) for row in body["splits"])
        dividends = tuple(Dividend(**{**row, "cash_per_share": Decimal(row["cash_per_share"])}) for row in body["dividends"])
        ids = [e.event_id for e in (*splits, *dividends)]
        if len(set(ids)) != len(ids) or any(e.instrument not in catalog.asset_ids for e in (*splits, *dividends)):
            raise ValueError("Duplicate or wrong-issue corporate event")
        # Validate all terms, dates, currencies and ambiguous same-day bases,
        # including events outside a particular evaluation subset.
        for day in sorted({s.effective_date for s in splits} | {d.ex_date for d in dividends}):
            apply_actions(Book(Decimal(0)), day, splits=[s for s in splits if s.effective_date == day],
                          dividends=[d for d in dividends if d.ex_date == day])
        return splits, dividends, sessions
