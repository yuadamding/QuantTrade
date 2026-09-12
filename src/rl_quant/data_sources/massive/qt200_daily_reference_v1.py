"""Join preserved daily observations to reference evidence without ticker stitching.

The output is a source-keyed preparation sidecar, not an eligibility authority.
Sparse dated observations are used only on their exact query dates. Provider
ticker-event brackets are candidate intervals, never continuous issue proof.
Original identity and economic captures must additionally be reconciled before
these support observations can be used in an aggregate research dataset.
"""

from __future__ import annotations

import gzip
import io
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive import qt200_daily_capture_v1 as daily
from rl_quant.data_sources.massive import qt200_identity_evidence_v1 as identity
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_pilot_v1 import _query_rows, normalize_bar

SCHEMA = "rl-quant.qt200-daily-reference-reconciliation-v1"
_BAD_EVENTS = frozenset({
    "current_ticker_differs_from_last_provider_event",
    "duplicate_or_conflicting_provider_event_dates",
    "provider_events_not_issue_identifier_linked",
})
_CONFLICT = "conflicts_with_current_observed_issue_identifiers"
_MATCH = "supports_current_observed_issue_identifiers"
_ET = ZoneInfo("America/New_York")


def _calendar(body: bytes) -> dict:
    obj = identity._json(body)
    if (obj.get("schema") != "quanttrade-xnys-calendar-source-v1"
            or obj.get("calendar") != "XNYS"
            or not isinstance(obj.get("exchange_calendars_version"), str)
            # The retained calendar producer hashes newline-terminated ASCII
            # JSON, unlike the identity diagnostic's no-newline semantic hash.
            or transport.digest(transport.canonical({k: v for k, v in obj.items()
                                                     if k != "receipt_sha256"})) != obj.get("receipt_sha256")):
        raise transport.ResearchCaptureError("Calendar identity/semantic receipt differs")
    start, end = identity._day(obj["coverage_start_date"]), identity._day(obj["coverage_end_date"])
    sessions = obj.get("sessions")
    if start > end or not isinstance(sessions, list) or not 1 <= len(sessions) <= 3000:
        raise transport.ResearchCaptureError("Calendar coverage invalid")
    prior = ""
    for row in sessions:
        day = identity._day(row["session_date"])
        times = [row.get(key) for key in ("regular_open_ns", "regular_close_ns")]
        if (not prior < day or not start <= day <= end or row.get("exchange") != "XNYS"
                or any(type(t) is not int or t < 0 for t in times)):
            raise transport.ResearchCaptureError("Calendar sessions invalid or duplicated")
        opened, closed = [datetime.fromtimestamp(t // 1_000_000_000, timezone.utc).astimezone(_ET)
                          for t in times]
        intervals = row.get("scheduled_five_minute_intervals")
        if (opened.date().isoformat() != day or closed.date().isoformat() != day
                or opened.weekday() >= 5 or (opened.hour, opened.minute, opened.second) != (9, 30, 0)
                or (closed.hour, closed.minute, closed.second) not in ((13, 0, 0), (16, 0, 0))
                or any(t % 1_000_000_000 for t in times)
                or type(intervals) is not int or intervals not in (42, 78)
                or times[1] - times[0] != intervals * 300_000_000_000):
            raise transport.ResearchCaptureError("Calendar clock/interval contract differs")
        prior = day
    return obj


def _brackets(resolution: dict) -> list[dict]:
    entity = resolution["current_reference_evidence"]
    if _BAD_EVENTS.intersection(entity["gaps"]) or entity["active_exact_reference_count"] != 1:
        return []
    result = []
    for event in entity["ticker_event_observations"]:
        if event["same_current_composite_figi"] is not True:
            continue
        start = max(daily.START, identity._day(event["event_date"]))
        end = daily.END
        if event["next_provider_event_date"] is not None:
            end = min(end, (date.fromisoformat(identity._day(event["next_provider_event_date"]))
                            - timedelta(days=1)).isoformat())
        if start <= end:
            result.append(dict(ticker=event["provider_ticker"], start=start, end=end,
                               event_source=event["source"],
                               provider_event_index=event["provider_event_index"]))
    return result


def _identity_class(resolution: dict, ticker: str, day: str, brackets: list[dict]) -> str:
    assertions = [x for x in resolution["dated_assertions"]
                  if x["query_ticker"] == ticker and x["query_date"] == day]
    # A conflict wins even if a second field/assertion matches. Never propagate
    # a sparse snapshot into neighbouring days or infer absence from HTTP 404.
    if any(x["classification"] == _CONFLICT for x in assertions):
        return "dated_issue_conflict"
    if assertions and all(x["classification"] == _MATCH for x in assertions):
        return "exact_date_issue_identifier_match_only"
    if assertions:
        return "exact_date_issue_unresolved"
    if any(x["ticker"] == ticker and x["start"] <= day <= x["end"] for x in brackets):
        return "provider_event_bracket_candidate_only"
    return "outside_or_missing_issue_event_bracket"


def _alias_queries(resolutions: list[dict]) -> list[dict]:
    queries = []
    for resolution in resolutions:
        target = resolution["qt200_ticker"]
        for bracket in _brackets(resolution):
            if bracket["ticker"] == target:
                continue
            ticker = bracket["ticker"]
            if not isinstance(ticker, str) or not ticker or not all(c.isascii() and
                    (c.isupper() or c.isdigit() or c == ".") for c in ticker):
                raise transport.ResearchCaptureError("Unsafe provider alias")
            path = "/v2/aggs/ticker/" + quote(ticker, safe="") + "/range/1/day/"
            url = "https://api.massive.com" + path + bracket["start"] + "/" + bracket["end"]
            url += "?" + urlencode({"adjusted": "false", "sort": "asc", "limit": "50000"})
            queries.append(dict(qt200_ticker=target, **bracket, request_url=url,
                                automatic_stitching_authorized=False))
    return queries


def reconcile_daily_references(*, daily_root: Path, daily_plan_sha256: str,
                               daily_completion_sha256: str, support_root: Path,
                               support_completion_sha256: str, dated_root: Path,
                               dated_completion_sha256: str, universe_sha256: str,
                               output: Path) -> dict:
    """Reopen source chains and publish compact keyed reference/missingness data.

    No derived caller-supplied rows, identity flags or dates are accepted. Both
    raw daily and reference chains are replayed again before final publication.
    The sidecar preserves every bar, including bars for wrong historical issues.
    Such preservation does not make those rows usable training observations.
    """
    replay_args = dict(root=daily_root, plan_sha256=daily_plan_sha256,
                       completion_sha256=daily_completion_sha256)
    daily_before = daily.replay_daily(**replay_args)
    reference_args = dict(support_root=support_root,
        expected_support_completion_sha256=support_completion_sha256, capture_root=dated_root,
        expected_capture_completion_sha256=dated_completion_sha256,
        expected_universe_sha256=universe_sha256)
    reference = identity.build_qt200_issue_resolution_evidence_v1(**reference_args)
    if tuple(reference["ordered_tickers"]) != daily.SYMBOLS:
        raise transport.ResearchCaptureError("Reference/daily ordered panel mismatch")
    relative = "authority-snapshots/xnys-calendar-source-v1.json"
    binding = [row for row in reference["support_input_bindings"] if row["relative_path"] == relative]
    body = transport.read_regular(support_root / relative, 1048576)
    if len(binding) != 1 or binding[0]["sha256"] != transport.digest(body) or binding[0]["bytes"] != len(body):
        raise transport.ResearchCaptureError("Calendar is not the inventoried support source")
    calendar = _calendar(body)
    session_dates = {x["session_date"] for x in calendar["sessions"]}
    capture = transport.parse_json(transport.read_regular(daily_root / "COMPLETE.json", 1048576))
    by_ticker = {row["qt200_ticker"]: row for row in reference["security_resolutions"]}
    summaries, observed_dates, total = [], set(), 0
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as stream:
        for query, entry in zip(daily.daily_queries(), capture["queries"], strict=True):
            resolution = by_ticker[query.ticker]
            brackets, counts, dates = _brackets(resolution), Counter(), []
            for raw, source in _query_rows(daily_root, query, entry):
                bar = normalize_bar(raw, query, received_at_ns=source["received_at_ns"])
                day = bar["session_date_et"]
                if dates and day <= dates[-1]:
                    raise transport.ResearchCaptureError("Duplicate/unordered daily row")
                dates.append(day)
                observed_dates.add(day)
                identity_class = _identity_class(resolution, query.ticker, day, brackets)
                calendar_class = ("bound_calendar_session" if day in session_dates else
                    "outside_bound_calendar_coverage" if not calendar["coverage_start_date"] <= day
                        <= calendar["coverage_end_date"] else "unexpected_non_session")
                counts[identity_class] += 1
                counts[calendar_class] += 1
                stream.write(transport.canonical(dict(ticker=query.ticker, session_date_et=day,
                    source=source, identity_evidence=identity_class, calendar_evidence=calendar_class,
                    historical_identity_qualified=False, execution_eligible=False)))
                total += 1
            summaries.append(dict(ticker=query.ticker, observed_rows=len(dates),
                first_observed_date=dates[0] if dates else None, last_observed_date=dates[-1] if dates else None,
                evidence_counts=dict(sorted(counts.items())),
                missing_bound_calendar_dates=sorted(session_dates - set(dates)),
                review_flags=resolution["review_flags"], candidate_event_brackets=brackets))
    actions = reference["economic_observations"]
    manifest = identity._json(identity._read(support_root, "manifest.json")[0])
    economic_end = identity._day(manifest["requested_economic_coverage_end"])
    action_summary = dict(observed_counts=dict(Counter(x["surface"] for x in actions)),
        records_with_problems=sum(bool(x["problems"]) for x in actions),
        problem_counts=dict(Counter(p for x in actions for p in x["problems"])),
        coverage_start=manifest["requested_economic_coverage_start"], coverage_end=economic_end,
        observed_dates_beyond_economic_coverage=sorted(day for day in observed_dates if day > economic_end),
        accounting_authorized=False)
    supplements = dict(alias_daily_queries=_alias_queries(reference["security_resolutions"]),
        alias_query_results_require_issue_reconciliation=True,
        identity_review_tickers=[x["qt200_ticker"] for x in reference["security_resolutions"] if x["review_flags"]],
        calendar_extension_dates=sorted(day for day in observed_dates if day > calendar["coverage_end_date"]),
        economic_extension_dates=action_summary["observed_dates_beyond_economic_coverage"],
        outer_access_authorized=False, training_ready=False)
    if daily.replay_daily(**replay_args) != daily_before or identity.build_qt200_issue_resolution_evidence_v1(
            **reference_args) != reference:
        raise transport.ResearchCaptureError("Daily/reference sources changed during reconciliation")
    output.mkdir(mode=0o700)
    files = {}
    for name, payload in (
        ("bar-reference-sidecar.jsonl.gz", buffer.getvalue()),
        ("identity-and-actions.json.gz", gzip.compress(transport.canonical(reference), mtime=0)),
        ("coverage-and-supplements.json.gz", gzip.compress(transport.canonical(dict(
            symbols=summaries, corporate_actions=action_summary, supplements=supplements)), mtime=0)),
    ):
        files[name] = transport.write_once(output / name, payload)
    result = dict(schema=SCHEMA, daily_capture_sha256=daily_completion_sha256,
        support_completion_sha256=support_completion_sha256, dated_completion_sha256=dated_completion_sha256,
        reference_receipt_sha256=reference["receipt_sha256"], universe_sha256=universe_sha256,
        daily_rows=total, symbol_count=len(summaries), observed_date_count=len(observed_dates),
        bound_calendar_session_count=len(session_dates),
        observed_non_session_dates=sorted(day for day in observed_dates if day not in session_dates
            and calendar["coverage_start_date"] <= day <= calendar["coverage_end_date"]),
        missing_bound_calendar_panel_dates=sorted(session_dates - observed_dates),
        corporate_actions=action_summary, alias_daily_query_count=len(supplements["alias_daily_queries"]),
        source_chains_replayed_before_and_after=True, files=files,
        original_reference_bodies_reauthenticated_by_this_join=False,
        identity_qualified=False, corporate_actions_qualified=False,
        point_in_time_qualified=False, native_v5_qualified=False, training_ready=False)
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result
