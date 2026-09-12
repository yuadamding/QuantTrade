"""Reference-join regressions; run on the approved remote LSF GPU allocation."""

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive import qt200_daily_reference_v1 as refs
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport


def _resolution(ticker="META", *, gaps=()):
    events = [dict(provider_ticker="FB", event_date="2012-05-18",
                   next_provider_event_date="2022-06-09", provider_event_index=0),
              dict(provider_ticker=ticker, event_date="2022-06-09",
                   next_provider_event_date=None, provider_event_index=1)]
    for event in events:
        event.update(same_current_composite_figi=True, source={"source_sha256": "a" * 64})
    return dict(qt200_ticker=ticker, dated_assertions=[], current_reference_evidence=dict(
        active_exact_reference_count=1, gaps=list(gaps), ticker_event_observations=events))


def _assertion(classification, day="2022-01-03", ticker="META"):
    return dict(query_ticker=ticker, query_date=day, classification=classification)


def _calendar_value():
    rows = []
    for day, hour, intervals in (("2024-07-02", 16, 78), ("2024-07-03", 13, 42)):
        times = [int(datetime.fromisoformat(day + suffix).replace(
            tzinfo=ZoneInfo("America/New_York")).timestamp()) * 1_000_000_000
                 for suffix in ("T09:30:00", f"T{hour:02}:00:00")]
        rows.append(dict(exchange="XNYS", session_date=day, regular_open_ns=times[0],
                         regular_close_ns=times[1], scheduled_five_minute_intervals=intervals))
    return dict(schema="quanttrade-xnys-calendar-source-v1", calendar="XNYS",
                coverage_start_date="2024-07-02", coverage_end_date="2024-07-03",
                exchange_calendars_version="fixture-only", sessions=rows)


def _seal(obj):
    obj = copy.deepcopy(obj)
    obj["receipt_sha256"] = transport.digest(transport.canonical(obj))
    return transport.canonical(obj)


def test_current_symbol_is_not_inferred_for_earlier_alias_period():
    row = _resolution()
    brackets = refs._brackets(row)
    assert refs._identity_class(row, "META", "2021-06-30", brackets) == "outside_or_missing_issue_event_bracket"
    assert refs._identity_class(row, "META", "2022-06-09", brackets) == "provider_event_bracket_candidate_only"
    assert refs._identity_class(row, "FB", "2022-06-08", brackets) == "provider_event_bracket_candidate_only"
    assert refs._identity_class(row, "FB", "2022-06-09", brackets) == "outside_or_missing_issue_event_bracket"


def test_dated_etf_conflict_wins_over_matching_field_or_event():
    row = _resolution()
    row["dated_assertions"] = [_assertion(refs._CONFLICT), _assertion(refs._MATCH)]
    assert refs._identity_class(row, "META", "2022-01-03", refs._brackets(row)) == "dated_issue_conflict"
    # Evidence is exact-date only: a conflict does not invent an issue interval.
    assert refs._identity_class(row, "META", "2022-01-04", refs._brackets(row)) == "outside_or_missing_issue_event_bracket"


def test_blank_issue_identifier_or_404_is_not_a_join_or_delisting():
    row = _resolution("SNOW")
    row["dated_assertions"] = [_assertion("unknown_issue_identity", "2017-01-03", "SNOW")]
    assert refs._identity_class(row, "SNOW", "2017-01-03", refs._brackets(row)) == "exact_date_issue_unresolved"


def test_one_dated_match_is_not_continuous_identity():
    row = _resolution()
    row["dated_assertions"] = [_assertion(refs._MATCH, "2026-08-26")]
    assert refs._identity_class(row, "META", "2026-08-26", refs._brackets(row)) == "exact_date_issue_identifier_match_only"
    assert refs._identity_class(row, "META", "2026-08-27", refs._brackets(row)) == "provider_event_bracket_candidate_only"


@pytest.mark.parametrize("gap", sorted(refs._BAD_EVENTS))
def test_conflicting_stale_alias_event_sets_cannot_supply_intervals(gap):
    row = _resolution(gaps=[gap])
    assert refs._brackets(row) == []
    assert refs._alias_queries([row]) == []


@pytest.mark.parametrize("count", [0, 2])
def test_unique_current_reference_is_required_for_candidate_brackets(count):
    row = _resolution()
    row["current_reference_evidence"]["active_exact_reference_count"] = count
    assert refs._brackets(row) == []


def test_unlinked_event_cannot_create_a_candidate_alias():
    row = _resolution()
    for event in row["current_reference_evidence"]["ticker_event_observations"]:
        event["same_current_composite_figi"] = False
    assert refs._alias_queries([row]) == []


def test_alias_supplement_has_exact_source_bracket_and_no_stitching_permission():
    queries = refs._alias_queries([_resolution()])
    assert len(queries) == 1
    row = queries[0]
    assert (row["qt200_ticker"], row["ticker"], row["start"], row["end"]) == (
        "META", "FB", "2017-01-03", "2022-06-08")
    assert row["request_url"].endswith("/2017-01-03/2022-06-08?adjusted=false&sort=asc&limit=50000")
    assert row["automatic_stitching_authorized"] is False
    assert row["event_source"]["source_sha256"] == "a" * 64


@pytest.mark.parametrize("ticker", ["../FB", "FB?apiKey=x", "fb", "", "ＦＢ"])
def test_provider_alias_cannot_inject_endpoint_parameters(ticker):
    row = _resolution()
    row["current_reference_evidence"]["ticker_event_observations"][0]["provider_ticker"] = ticker
    with pytest.raises(transport.ResearchCaptureError):
        refs._alias_queries([row])


def test_bound_calendar_preserves_regular_and_early_close_clocks():
    obj = _calendar_value()
    result = refs._calendar(_seal(obj))
    assert result["sessions"][1]["scheduled_five_minute_intervals"] == 42
    assert result["sessions"][0]["regular_open_ns"] == 1719927000000000000


@pytest.mark.parametrize("mutation", ["digest", "duplicate", "clock", "count", "range", "bool"])
def test_invalid_calendar_rejected(mutation):
    obj = _calendar_value()
    if mutation == "duplicate":
        obj["sessions"].append(obj["sessions"][-1])
    elif mutation == "clock":
        obj["sessions"][0]["regular_open_ns"] += 1
    elif mutation == "count":
        obj["sessions"][0]["scheduled_five_minute_intervals"] = 42
    elif mutation == "range":
        obj["coverage_end_date"] = "2024-07-02"
    elif mutation == "bool":
        obj["sessions"][0]["regular_open_ns"] = True
    body = _seal(obj)
    if mutation == "digest":
        body = body.replace(b'"XNYS"', b'"XNAS"')
    with pytest.raises(transport.ResearchCaptureError):
        refs._calendar(body)
