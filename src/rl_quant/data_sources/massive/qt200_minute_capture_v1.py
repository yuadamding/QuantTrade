"""Rolling full-history minute observations, distinct from trade-derived fills.

Each immutable capture owns one exact ticker interval; the study-wide operator
must additionally enforce its total bytes/deadline and remote storage ceiling.
Raw pre/post-market observations are preserved even though the candidate
execution-input view retains only [09:35,09:45) America/New_York.
"""

from __future__ import annotations

import gzip
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_pilot_v1 import EPOCH, _query_rows, normalize_bar
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import END, START, SYMBOLS
from rl_quant.data_sources.massive.qt200_reference_supplement_v1 import ALIASES

SCHEMA = "rl-quant.qt200-minute-research-capture-v1"
MAX_BYTES, MAX_PAGES = 512_000_000, 128
INTERVALS = frozenset({(ticker, START, END) for ticker in SYMBOLS}
                      | {(ticker, start, end) for _, ticker, start, end in ALIASES})


class MinuteQuery(transport.PilotQuery):
    def validate(self) -> None:
        if self.product != "minute" or (self.ticker, self.start, self.end) not in INTERVALS:
            raise transport.ResearchCaptureError("Minute query outside fixed research population")


def minute_queries() -> tuple[MinuteQuery, ...]:
    queries = [MinuteQuery("minute", ticker, START, END) for ticker in SYMBOLS]
    queries.extend(MinuteQuery("minute", ticker, start, end) for _, ticker, start, end in ALIASES)
    if len(queries) != 203 or len({q.name for q in queries}) != len(queries):
        raise transport.ResearchCaptureError("Minute query population changed")
    return tuple(queries)


def _query(ticker: str) -> MinuteQuery:
    matches = [q for q in minute_queries() if q.ticker == ticker]
    if len(matches) != 1:
        raise transport.ResearchCaptureError("Ticker outside minute study")
    return matches[0]


def plan_fields(ticker: str) -> dict:
    query = _query(ticker)
    return dict(schema=SCHEMA + "-plan", research_track="QT200-AGG-DEV-01",
        queries=[asdict(query)], maximum_raw_response_bytes=MAX_BYTES,
        maximum_page_bytes=transport.MAX_PAGE_BYTES, maximum_pages_per_query=MAX_PAGES,
        maximum_elapsed_seconds=transport.MAX_CAPTURE_SECONDS, retry_count=0, concurrent_requests=1,
        minimum_request_gap_seconds=0.3, raw_outside_execution_window_preserved=True,
        point_in_time_qualified=False, native_v5_qualified=False, training_ready=False)


def _check(root: Path, plan_sha256: str, ticker: str, current: bool) -> MinuteQuery:
    raw = transport.read_regular(root / "plan.json", 1048576)
    plan = transport.parse_json(raw)
    if (transport.digest(raw) != plan_sha256 or any(
            type(plan.get(k)) is not type(v) or plan.get(k) != v for k, v in plan_fields(ticker).items())):
        raise transport.ResearchCaptureError("Minute capture plan differs")
    if current and plan.get("minute_implementation_sha256") != transport.digest(
            transport.read_regular(Path(__file__).resolve(), 1048576)):
        raise transport.ResearchCaptureError("Minute implementation differs")
    return _query(ticker)


def capture_minute(*, root: Path, ticker: str, api_key: str, plan_sha256: str,
                   request_pacer: transport.RequestPacer | None = None) -> dict:
    query = _check(root, plan_sha256, ticker, True)
    return transport._capture_queries(root=root, api_key=api_key, plan_sha256=plan_sha256,
        queries=(query,), schema=SCHEMA, maximum_bytes=MAX_BYTES, maximum_pages=MAX_PAGES,
        request_pacer=request_pacer)


def replay_minute(*, root: Path, ticker: str, plan_sha256: str, completion_sha256: str) -> dict:
    query = _check(root, plan_sha256, ticker, False)
    return transport._replay_queries(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256,
        queries=(query,), schema=SCHEMA, maximum_bytes=MAX_BYTES, maximum_pages=MAX_PAGES)


def in_execution_window(timestamp_ms: int) -> bool:
    stamp = (EPOCH + timedelta(milliseconds=timestamp_ms)).astimezone(transport.ET)
    return stamp.hour == 9 and 35 <= stamp.minute < 45


def normalize_execution_inputs(*, root: Path, ticker: str, plan_sha256: str,
                               completion_sha256: str, output: Path) -> dict:
    """Retain actual window bars and explicit slot gaps, not prices or fills."""
    args = dict(root=root, ticker=ticker, plan_sha256=plan_sha256, completion_sha256=completion_sha256)
    before = replay_minute(**args)
    query = _query(ticker)
    complete = transport.parse_json(transport.read_regular(root / "COMPLETE.json", 1048576))
    dates, slots, failures, total, selected, prior = set(), {}, [], 0, 0, -1
    counts = Counter()
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as stream:
        for row, source in _query_rows(root, query, complete["queries"][0]):
            total += 1
            try:
                bar = normalize_bar(row, query, received_at_ns=source["received_at_ns"])
                stamp = bar["bar_start_ms"]
                if stamp <= prior:
                    raise transport.ResearchCaptureError("Minute observations not unique and increasing")
                prior = stamp
                day = bar["session_date_et"]
                dates.add(day)
                if in_execution_window(stamp):
                    minute = datetime.fromtimestamp(stamp / 1000, transport.ET).minute
                    slots.setdefault(day, []).append(minute)
                    counts["vwap_present"] += bar["vwap_observed"]
                    stream.write(transport.canonical(dict(**bar, source=source,
                        source_capture_sha256=completion_sha256, execution_price_established=False,
                        executable_liquidity_qualified=False, training_eligible=False)))
                    selected += 1
            except (ValueError, OverflowError) as exc:
                failures.append(dict(source=source, error_type=type(exc).__name__))
    if total != complete["queries"][0]["result_count"] or replay_minute(**args) != before:
        raise transport.ResearchCaptureError("Minute source population or bytes changed")
    missing = {day: [m for m in range(35, 45) if m not in slots.get(day, [])] for day in sorted(dates)
               if slots.get(day, []) != list(range(35, 45))}
    output.mkdir(mode=0o700)
    files = {"execution-window-bars.jsonl.gz": transport.write_once(output / "execution-window-bars.jsonl.gz", buffer.getvalue()),
             "coverage.json.gz": transport.write_once(output / "coverage.json.gz", gzip.compress(transport.canonical(
                 dict(observed_dates=sorted(dates), missing_window_slots=missing, invalid_rows=failures,
                      comparison_is_exchange_calendar=False, missing_liquidity_imputed=False)), mtime=0))}
    result = dict(schema=SCHEMA + "-execution-inputs", ticker=ticker, plan_sha256=plan_sha256,
        capture_sha256=completion_sha256, files=files, raw_rows=total, selected_window_rows=selected,
        observed_dates=len(dates), dates_with_window_gaps=len(missing), invalid_rows=len(failures),
        vwap_observed_rows=counts["vwap_present"], source_unchanged=True,
        observed_schema_accepted=selected > 0 and not failures, modeled_fills_created=False,
        historical_issue_identity_qualified=False, point_in_time_qualified=False,
        native_v5_qualified=False, training_ready=False)
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result
