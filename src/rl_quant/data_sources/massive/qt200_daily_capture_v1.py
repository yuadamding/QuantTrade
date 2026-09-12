"""Full-panel finalized daily observations, never native V5 or PIT promotion.

The fixed user panel is a list of requested ticker strings, not a historical
issue mapping. In particular, retained SNOW observations are not automatically
Snowflake history. No fills, prices, dates, or missing observations are invented.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import asdict
from pathlib import Path

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_pilot_v1 import _query_rows, normalize_bar

SCHEMA = "rl-quant.qt200-daily-research-capture-v1"
START, END = "2017-01-03", "2026-08-31"
MAX_BYTES = 512_000_000
MAX_PAGES = 8
ORDERED_SYMBOLS_SHA256 = "92002983b9b6686bf8b6f5d6f88d212cf97f61f1f98bb55d49f262fa33b491fe"
SYMBOLS = tuple("""
AAPL MSFT NVDA AVGO AMD INTC QCOM TXN ADI MU AMAT LRCX KLAC MCHP ON MPWR
SNPS CDNS ORCL CRM ADBE NOW INTU ADSK ANET CSCO IBM DELL HPE HPQ PLTR CRWD
PANW FTNT DDOG SNOW GOOGL META NFLX DIS CMCSA CHTR TMUS VZ T TTWO ROKU PINS
AMZN TSLA HD LOW MCD SBUX NKE TJX BKNG ABNB MAR HLT RCL CCL F GM ORLY AZO
CMG YUM DHI LEN WMT COST PG KO PEP PM MO MDLZ CL KMB GIS KHC KR SYY
LLY JNJ ABBV MRK PFE BMY AMGN GILD REGN VRTX TMO DHR ABT MDT SYK BSX ISRG EW
BDX ZTS UNH ELV CI CVS HUM HCA IQV DXCM JPM BAC WFC C GS MS SCHW BLK BX KKR
V MA AXP COF PYPL CME ICE SPGI MCO PGR TRV ALL MET PRU USB PNC BRK.B COIN
GE CAT DE HON RTX LMT NOC GD BA UNP CSX NSC UPS FDX UAL DAL LUV ETN EMR ROK
ITW PH CMI PCAR WM RSG FAST CTAS XOM CVX COP EOG OXY SLB MPC VLO WMB KMI
LIN APD SHW ECL FCX NEM NUE DOW NEE DUK SO AEP EXC SRE XEL VST AMT PLD EQIX DLR SPG O
""".split())


class DailyQuery(transport.PilotQuery):
    def validate(self) -> None:
        if (self.product != "day" or self.ticker not in SYMBOLS
                or self.start != START or self.end != END):
            raise transport.ResearchCaptureError("Query outside fixed QT200 daily history")


def daily_queries() -> tuple[DailyQuery, ...]:
    if (len(SYMBOLS) != 200 or len(set(SYMBOLS)) != 200
            or transport.digest(transport.canonical(SYMBOLS)) != ORDERED_SYMBOLS_SHA256):
        raise transport.ResearchCaptureError("Frozen ordered user panel differs")
    return tuple(DailyQuery("day", ticker, START, END) for ticker in SYMBOLS)


def plan_fields() -> dict:
    """Static request contract to persist before the credential is opened."""
    return {
        "schema": SCHEMA + "-plan", "research_track": "QT200-AGG-DEV-01",
        "queries": [asdict(query) for query in daily_queries()],
        "ordered_universe": list(SYMBOLS), "ordered_symbols_sha256": ORDERED_SYMBOLS_SHA256,
        "maximum_raw_response_bytes": MAX_BYTES,
        "maximum_page_bytes": transport.MAX_PAGE_BYTES,
        "maximum_pages_per_query": MAX_PAGES,
        "maximum_elapsed_seconds": transport.MAX_CAPTURE_SECONDS,
        "retry_count": 0, "concurrent_requests": 1, "minimum_request_gap_seconds": 0.3,
        "training_ready": False, "point_in_time_qualified": False,
        "native_v5_qualified": False, "historical_issue_identity_qualified": False,
    }


def _check_plan(root: Path, plan_sha256: str, *, current_implementation: bool) -> None:
    body = transport.read_regular(root / "plan.json", 1048576)
    plan = transport.parse_json(body)
    if (transport.digest(body) != plan_sha256
            or any(plan.get(key) != value or type(plan.get(key)) is not type(value)
                   for key, value in plan_fields().items())):
        raise transport.ResearchCaptureError("Frozen daily capture plan differs")
    if current_implementation and plan.get("daily_implementation_sha256") != transport.digest(
        transport.read_regular(Path(__file__).resolve(), 1048576)
    ):
        raise transport.ResearchCaptureError("Daily capture implementation changed")


def capture_daily(*, root: Path, api_key: str, plan_sha256: str) -> dict:
    _check_plan(root, plan_sha256, current_implementation=True)
    return transport._capture_queries(
        root=root, api_key=api_key, plan_sha256=plan_sha256,
        queries=daily_queries(), schema=SCHEMA, maximum_bytes=MAX_BYTES, maximum_pages=MAX_PAGES,
    )


def replay_daily(*, root: Path, plan_sha256: str, completion_sha256: str) -> dict:
    _check_plan(root, plan_sha256, current_implementation=False)
    return transport._replay_queries(
        root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256,
        queries=daily_queries(), schema=SCHEMA, maximum_bytes=MAX_BYTES, maximum_pages=MAX_PAGES,
    )


def normalize_daily(*, root: Path, plan_sha256: str, completion_sha256: str,
                    output: Path) -> dict:
    """Compact all observed rows; report defects and never silently fill gaps.

    A union-of-observed-dates comparison is diagnostic, not an exchange calendar
    or proof of historical eligibility. The immutable raw query is authoritative
    for rejected rows. Analysis is bounded by the verified acquisition limits.
    """
    before = replay_daily(root=root, plan_sha256=plan_sha256,
                          completion_sha256=completion_sha256)
    output.mkdir(mode=0o700)
    complete = transport.parse_json(transport.read_regular(root / "COMPLETE.json", 1048576))
    summaries, dates_by_ticker, invalid = [], {}, []
    total = 0
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as stream:
        for query, entry in zip(daily_queries(), complete["queries"], strict=True):
            dates, prior, accepted = set(), -1, 0
            for row, provenance in _query_rows(root, query, entry):
                try:
                    bar = normalize_bar(row, query, received_at_ns=provenance["received_at_ns"])
                    stamp = bar["bar_start_ms"]
                    if stamp <= prior or bar["session_date_et"] in dates:
                        raise transport.ResearchCaptureError("Duplicate or unordered interval")
                    prior = stamp
                    dates.add(bar["session_date_et"])
                except (ValueError, OverflowError) as exc:
                    invalid.append({"query": query.name, **provenance,
                                    "error_type": type(exc).__name__})
                    continue
                stream.write(transport.canonical({**bar, "source": provenance}))
                total += 1
                accepted += 1
            dates_by_ticker[query.ticker] = dates
            summaries.append({"ticker": query.ticker, "provider_rows": entry["result_count"],
                              "accepted_rows": accepted, "first_observed_date": min(dates) if dates else None,
                              "last_observed_date": max(dates) if dates else None})
    observed_dates = sorted(set().union(*dates_by_ticker.values()))
    for item in summaries:
        item["absent_dates_relative_to_observed_panel"] = [
            day for day in observed_dates if day not in dates_by_ticker[item["ticker"]]
        ]
    after = replay_daily(root=root, plan_sha256=plan_sha256,
                         completion_sha256=completion_sha256)
    if before != after:
        raise transport.ResearchCaptureError("Source replay changed during normalization")
    bars = transport.write_once(output / "observed-daily-bars.jsonl.gz", buffer.getvalue())
    coverage = transport.write_once(output / "observed-coverage.json.gz", gzip.compress(
        transport.canonical({"queries": summaries, "observed_panel_dates": observed_dates,
                             "invalid_rows": invalid, "exchange_calendar_qualified": False}), mtime=0,
    ))
    result = {"schema": SCHEMA + "-normalized", "plan_sha256": plan_sha256,
              "capture_sha256": completion_sha256, "query_count": len(summaries),
              "observed_rows": total, "invalid_rows": len(invalid),
              "observed_panel_date_count": len(observed_dates), "bars": bars, "coverage": coverage,
              "observed_bar_schema_accepted": bool(total) and not invalid,
              "capture_read_only_replayed": True, "source_unchanged": True,
              "full_history_dataset_complete": False, "identity_qualified": False,
              "corporate_actions_qualified": False, "features_targets_qualified": False,
              "point_in_time_qualified": False, "native_v5_qualified": False,
              "training_ready": False}
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result
