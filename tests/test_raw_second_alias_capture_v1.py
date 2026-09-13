"""LSF-only alias source regressions; synthetic responses are not real evidence."""

import base64
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from rl_quant.data_sources.massive import raw_second_capture_v1 as capture
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS
from rl_quant.data_sources.massive.qt200_reference_supplement_v1 import SupplementQuery
from rl_quant.data_sources.massive.raw_second_alias_v1 import (
    AliasEventRef, AliasIdentityRef, SecondAliasRoute, validate_fixed_slot_identity,
)
from rl_quant.datasets.massive_raw_seconds_v1 import SecondCaptureRef, SecondQuery
from test_raw_second_capture_v1 import _network

pytestmark = pytest.mark.lsf_gpu
ISSUES = {
    "META": ("FB", "BBG000MM2P62", "BBG001SQCQC5", "0001326801", "2022-06-09"),
    "ELV": ("ANTM", "BBG000BCG930", "BBG001S6KBQ8", "0001156039", "2022-06-28"),
}


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def _write(path, body):
    raw = json.dumps(body, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def _original(body):
    raw = json.dumps(body, indent=2).encode()
    return dict(raw_response_body_base64=base64.b64encode(raw).decode(),
                raw_response_body_sha256=sha256(raw).hexdigest(),
                raw_response_content_length=len(raw), provider_request_id=body["request_id"])


def _identity(tmp_path, ticker, day, issue, **override):
    row = dict(ticker=ticker, composite_figi=issue[0], share_class_figi=issue[1], cik=issue[2],
               market="stocks", locale="us", type="CS", active=True, list_date="2000-01-03")
    row.update(override)
    body = dict(status="OK", request_id="synthetic-dated-identity", results=row)
    wrapper = dict(ticker=ticker, date=day, http_status=200, results=row,
                   request_url=f"https://api.massive.com/v3/reference/tickers/{ticker}?date={day}",
                   historical_known_at=None, capture_time_is_historical_availability=False,
                   **_original(body))
    path = tmp_path / f"identity-{ticker}-{day}.json"
    return AliasIdentityRef("dated-ticker-wrapper", str(path), _write(path, wrapper))


def _route(tmp_path, slot="META", day="2017-01-03", **identity_override):
    provider, figi, share, cik, change = ISSUES[slot]
    identity = _identity(tmp_path, provider, day, (figi, share, cik), **identity_override)
    events = [dict(type="ticker_change", date=change, ticker_change=dict(ticker=slot)),
              dict(type="ticker_change", date="2012-05-18" if slot == "META" else "2014-12-03",
                   ticker_change=dict(ticker=provider))]
    body = dict(status="OK", request_id="synthetic-event", results=dict(
        composite_figi=figi, cik=cik, events=events))
    row = dict(identifier=figi, response_status=200, event_count=2, **_original(body))
    path = tmp_path / f"event-{slot}.json"
    event = AliasEventRef(str(path), _write(path, dict(
        schema="quanttrade-massive-ticker-events-capture-v1", rows=[row])), 0)
    route = SecondAliasRoute(slot, provider, day, identity, event)
    start = int(datetime.fromisoformat(day + "T14:30:00+00:00").timestamp() * 1000)
    return route, SecondQuery(provider, start, start + 3599_000)


@pytest.mark.parametrize("slot", ["META", "ELV"])
def test_alias_capture_materialization_and_replay_preserve_literal_provider(tmp_path, monkeypatch, slot):
    route, query = _route(tmp_path, slot)
    root = tmp_path / "capture"
    plan = capture.publish_second_capture_plan(root=root, queries=(query,), alias_routes=(route,))
    assert SecondAliasRoute.from_dict(asdict(route)) == route
    body = json.loads((root / "plan.json").read_bytes())
    assert body["schema"] == capture.ALIAS_SCHEMA + "-plan"
    assert body["queries"] == [asdict(query)]
    assert body["alias_routes"][0]["fixed_slot_ticker"] == slot
    calls = _network(monkeypatch, query)
    capture.capture_seconds(root=root, plan_sha256=plan, api_key="synthetic-transport-only")
    complete = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    output = tmp_path / "native"
    result = capture.materialize_second_sources(root=root, plan_sha256=plan,
                                                completion_sha256=complete, output=output)
    actual, pages = SecondCaptureRef(**result["captures"][0]).load()
    assert actual == query and actual.ticker == ISSUES[slot][0]
    for index, page in enumerate(pages):
        original = gzip.decompress((root / capture.SecondHTTPQuery(**asdict(query)).name /
                                    f"page-{index:04d}.json.gz").read_bytes())
        assert page.body == original
        assert json.loads(original)["ticker"] == query.ticker
        assert f"/ticker/{query.ticker}/" in page.request_url
    assert not result["historical_issue_identity_qualified"]
    assert not result["continuous_identity_qualified"] and not result["training_ready"]
    assert result["provider_tickers_rewritten"] is False
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    capture.verify_second_sources(root=root, plan_sha256=plan, completion_sha256=complete,
        output=output, output_sha256=sha256((output / "COMPLETE.json").read_bytes()).hexdigest())
    assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    assert len(calls) == 2


@pytest.mark.parametrize("case", ["missing", "wrong-slot", "wrong-issue", "wrong-date", "interpolated",
                                   "midnight", "outside-bracket", "duplicate", "unused", "mixed-slot"])
def test_bad_alias_scope_rejected_before_plan_or_acquisition(tmp_path, case):
    route, query = _route(tmp_path, composite_figi="WRONG" if case == "wrong-issue" else ISSUES["META"][1])
    routes, queries = (route,), (query,)
    if case == "missing":
        routes = ()
    elif case == "wrong-slot":
        routes = (replace(route, fixed_slot_ticker="SNOW"),)
    elif case in ("wrong-date", "interpolated"):
        query = replace(query, start_ms=query.start_ms + 86400_000, end_ms=query.end_ms + 86400_000)
        queries = (query,)
        if case == "interpolated":
            routes = (replace(route, session_date="2017-01-04"),)
    elif case == "midnight":
        start = int(datetime(2017, 1, 4, 4, 45, tzinfo=timezone.utc).timestamp() * 1000)
        queries = (replace(query, start_ms=start, end_ms=start + 3599_000),)
    elif case == "outside-bracket":
        other = tmp_path / "after"
        other.mkdir()
        route, query = _route(other, day="2022-06-09")
        routes, queries = (route,), (query,)
    elif case == "duplicate":
        routes = (route, route)
    elif case == "unused":
        queries = (replace(query, ticker="AAPL"),)
    elif case == "mixed-slot":
        queries = (query, replace(query, ticker="META"))
    root = tmp_path / "never-published"
    with pytest.raises(ValueError):
        capture.publish_second_capture_plan(root=root, queries=queries, alias_routes=routes)
    assert not root.exists()


def test_changed_evidence_blocks_before_started_and_network(tmp_path, monkeypatch):
    route, query = _route(tmp_path)
    root = tmp_path / "capture"
    plan = capture.publish_second_capture_plan(root=root, queries=(query,), alias_routes=(route,))
    calls = _network(monkeypatch, query)
    Path(route.identity.path).write_bytes(b"{}")
    with pytest.raises(ValueError, match="evidence changed"):
        capture.capture_seconds(root=root, plan_sha256=plan, api_key="synthetic-transport-only")
    assert not calls and not (root / "STARTED.json").exists()


def test_outer_reference_cannot_disagree_with_original_body(tmp_path):
    route, query = _route(tmp_path)
    path = Path(route.identity.path)
    wrapper = json.loads(path.read_bytes())
    wrapper["results"]["ticker"] = "META"
    route = replace(route, identity=replace(route.identity, sha256=_write(path, wrapper)))
    with pytest.raises(ValueError, match="wrapper/body disagreement"):
        route.validate(query)


def test_existing_literal_plan_retains_original_schema(tmp_path):
    root = tmp_path / "ordinary"
    capture.publish_second_capture_plan(root=root, queries=(SecondQuery("AAPL", 1000, 2000),))
    plan = json.loads((root / "plan.json").read_bytes())
    assert plan["schema"] == capture.SCHEMA + "-plan" and "alias_routes" not in plan
    assert "alias_implementation_sha256" not in plan


def test_one_dated_observation_routes_disjoint_hours_but_not_other_dates(tmp_path):
    route, first = _route(tmp_path)
    second = replace(first, start_ms=first.start_ms + 3600_000, end_ms=first.end_ms + 3600_000)
    root = tmp_path / "two-hours"
    capture.publish_second_capture_plan(root=root, queries=(first, second), alias_routes=(route,))
    plan = json.loads((root / "plan.json").read_bytes())
    assert len(plan["queries"]) == 2 and len(plan["alias_routes"]) == 1


@pytest.mark.parametrize("case", ["other-issue", "different-transition", "ambiguous-events"])
def test_new_hash_does_not_turn_unsupported_event_semantics_into_alias_evidence(tmp_path, case):
    route, query = _route(tmp_path)
    path = Path(route.event.path)
    source = json.loads(path.read_bytes())
    row = source["rows"][0]
    body = json.loads(base64.b64decode(row["raw_response_body_base64"]))
    if case == "other-issue":
        body["results"]["composite_figi"] = "UNRELATED"
    elif case == "different-transition":
        body["results"]["events"][0]["date"] = "2022-06-10"
    else:
        body["results"]["events"][1]["date"] = body["results"]["events"][0]["date"]
    row.update(_original(body))
    route = replace(route, event=replace(route.event, sha256=_write(path, source)))
    with pytest.raises(ValueError):
        route.validate(query)


def _supplement(tmp_path, route, *, duplicate=False):
    query = SupplementQuery("reference-alias", route.provider_ticker,
                            route.session_date, route.session_date, route.fixed_slot_ticker)
    original = json.loads(Path(route.identity.path).read_bytes())["results"]
    body = dict(status="OK", request_id="synthetic-supplement", results=[original] * (2 if duplicate else 1))
    raw, root = json.dumps(body, indent=2).encode(), tmp_path / query.name
    root.mkdir()
    packed = gzip.compress(raw, mtime=0)
    (root / "page-0000.json.gz").write_bytes(packed)
    receipt = dict(request_url=query.url, http_status=200, page_index=0, query=asdict(query),
                   plan_sha256="a" * 64, predecessor_page_sha256=None, requested_at_ns=1, received_at_ns=2,
                   capture_time_is_historical_availability=False, raw_body_sha256=sha256(raw).hexdigest(),
                   raw_body_bytes=len(raw), body=dict(path="page-0000.json.gz", bytes=len(packed),
                                                     sha256=sha256(packed).hexdigest()))
    receipt_hash = _write(root / "page-0000.receipt.json", receipt)
    complete = dict(schema="rl-quant.qt200-reference-supplement-v1-query-complete", query=asdict(query),
                    plan_sha256="a" * 64, pagination_complete=True, point_in_time_qualified=False,
                    training_ready=False, page_count=1, result_count=len(body["results"]),
                    pages=[dict(path="page-0000.receipt.json", sha256=receipt_hash,
                                bytes=(root / "page-0000.receipt.json").stat().st_size)])
    return replace(route, identity=AliasIdentityRef("reference-supplement", str(root),
                                                   _write(root / "COMPLETE.json", complete)))


@pytest.mark.parametrize("duplicate", [False, True])
def test_original_supplement_can_be_reused_but_duplicate_issue_is_ambiguous(tmp_path, duplicate):
    route, query = _route(tmp_path)
    route = _supplement(tmp_path, route, duplicate=duplicate)
    if duplicate:
        with pytest.raises(ValueError, match="ambiguous"):
            route.validate(query)
    else:
        result = route.validate(query)
        assert result["composite_figi"] == ISSUES["META"][1]
        assert result["ticker_event_bracket_end_exclusive"] == "2022-06-09"
        assert not result["continuous_identity_qualified"]


@pytest.mark.parametrize("case", ["correct", "META-ETF", "SNOW-Intrawest", "XOM-predecessor", "prelisting"])
def test_fixed_slot_identity_requires_original_exact_issue_not_ticker_string(tmp_path, case):
    slot = case.split("-")[0] if "-" in case else "AAPL"
    expected = {"AAPL": ("BBG000B9XRY4", "BBG001S5N8V8", "0000320193"),
                "META": ISSUES["META"][1:4], "SNOW": ("BBG007DHGNJ4", "BBG007DHGNK2", "0001640147"),
                "XOM": ("BBG023CY9MM1", "BBG023CY9NL0", "0002115436")}[slot]
    rows = [dict(qt200_ticker=ticker, current_reference_evidence=dict(
        current_security_type="CS", active_exact_reference_count=1, current_issue_identifier_observed=True,
        current_composite_figi=expected[0], current_share_class_figi=expected[1], current_issuer_cik=expected[2]))
        for ticker in SYMBOLS]
    evidence = dict(schema="rl-quant.qt200-issue-resolution-evidence-v1", ordered_tickers=list(SYMBOLS),
                    historical_identity_qualified=False, training_ready_for_adaptive_v5=False,
                    security_resolutions=rows)
    packed = gzip.compress(json.dumps(evidence).encode(), mtime=0)
    path = tmp_path / "intended-issues.json.gz"
    path.write_bytes(packed)
    actual, extra = expected, {}
    if case == "META-ETF":
        actual, extra = ("BBG011J1MN62", "BBG011J1MP12", None), dict(type="ETF")
    elif case == "SNOW-Intrawest":
        actual = (None, None, None)
    elif case == "XOM-predecessor":
        actual = ("BBG000GZQ728", "BBG001S69V32", "0000034088")
    elif case == "prelisting":
        extra = dict(list_date="2022-01-04")
    identity = _identity(tmp_path, slot, "2022-01-03", actual, **extra)
    args = dict(fixed_slot_ticker=slot, provider_ticker=slot, session_date="2022-01-03",
                identity=identity, intended_issue_source=path, intended_issue_sha256=sha256(packed).hexdigest())
    if case == "correct":
        result = validate_fixed_slot_identity(**args)
        assert result["composite_figi"] == expected[0] and not result["training_ready"]
    else:
        with pytest.raises(ValueError, match="reviewed common-stock issue"):
            validate_fixed_slot_identity(**args)


@pytest.mark.parametrize("stamp", [True, 1000.0, 1001])
def test_alias_query_still_requires_exact_integral_second_boundaries(stamp):
    with pytest.raises(ValueError):
        SecondQuery("FB", stamp, 2000)
