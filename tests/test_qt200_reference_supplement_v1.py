"""Remote-only transport regressions; synthetic replies are not market evidence."""

import gzip
import json
from pathlib import Path

import pytest

from rl_quant.data_sources.massive import qt200_reference_supplement_v1 as supplement
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport


def _evidence(path):
    body = {"ordered_tickers": list(supplement.SYMBOLS), "historical_identity_qualified": False,
            "training_ready_for_adaptive_v5": False, "security_resolutions": [
                {"qt200_ticker": ticker, "current_reference_evidence": {
                    "current_issuer_cik": f"{i + 1:010d}"}}
                for i, ticker in enumerate(supplement.SYMBOLS)]}
    packed = gzip.compress(transport.canonical(body), mtime=0)
    transport.write_once(path, packed)
    return body, transport.digest(packed)


def _plan(tmp_path):
    evidence = tmp_path / "evidence.json.gz"
    body, evidence_sha = _evidence(evidence)
    root = tmp_path / "capture"
    root.mkdir()
    fields = supplement.plan_fields(evidence=evidence, evidence_sha256=evidence_sha)
    fields.update(supplement_implementation_sha256=transport.digest(Path(supplement.__file__).read_bytes()),
                  capture_implementation_sha256=transport.digest(Path(transport.__file__).read_bytes()))
    raw = transport.canonical(fields)
    transport.write_once(root / "plan.json", raw)
    return root, evidence, evidence_sha, transport.digest(raw), supplement.queries_from_evidence(body)


def _network(patch, queries, failure=False):
    calls = []
    by_url = {q.url: q for q in queries}

    class Response:
        def __init__(self, url):
            self.url, self.code = url, 403 if failure else 200
            self.headers = {"Content-Type": "application/json"}

        def geturl(self):
            return self.url

        def read(self, bound):
            query = by_url[self.url]
            body = {"status": "ERROR" if failure else "OK", "request_id": "synthetic-response",
                    "results": []}
            if query.product == "day":
                body.update(ticker=query.ticker, adjusted=False, resultsCount=0)
            return transport.canonical(body)[:bound]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Opener:
        def open(self, req, timeout):
            assert req.get_header("Authorization") == "Bearer synthetic-test-only"
            assert timeout == 45 and "synthetic-test-only" not in req.full_url
            calls.append(req.full_url)
            return Response(req.full_url)

    patch.setattr(transport.request, "build_opener", lambda *_: Opener())
    patch.setattr(transport.time, "sleep", lambda *_: None)
    return calls


def test_source_bound_population_is_exact(tmp_path):
    _, _, _, _, queries = _plan(tmp_path)
    assert len(queries) == len({q.name for q in queries}) == 340
    assert sum(q.product == "reference-issuer" for q in queries) == 129
    assert sum(q.product == "reference-ticker" for q in queries) == 200
    assert {q.ticker for q in queries if q.product == "day"} == {"FB", "ANTM", "HONI"}
    assert all(q.validate_url(q.url) == q.url for q in queries)
    assert all("apiKey" not in q.url for q in queries)


@pytest.mark.parametrize("args", [
    ("day", "GOOG", "2017-01-03", "2022-06-08", "GOOGL"),
    ("reference-ticker", "BRK-B", "2026-08-31", "2026-08-31", "BRK-B"),
    ("reference-ticker", "AAPL", "2026-09-01", "2026-09-01", "AAPL"),
    ("reference-issuer", "AAPL", "2017-01-03", "2017-01-03", "AAPL", "0000000001"),
    ("reference-issuer", "META", "2017-01-03", "2017-01-03", "META", "../secret"),
    ("reference-alias", "HONI", "2017-01-03", "2017-01-03", "HON"),
    ("splits", "market", "2017-01-03", "2026-08-31", "market"),
])
def test_unregistered_query_rejected(args):
    with pytest.raises(transport.ResearchCaptureError):
        supplement.SupplementQuery(*args).validate()


@pytest.mark.parametrize("suffix", [
    "&apiKey=synthetic", "&apikey=synthetic", "&date=2026-09-01", "&ticker=WMT",
    "&active=false", "&limit=50000", "&cursor=x&cursor=y", "#fragment",
])
def test_cursor_cannot_change_reference_scope(suffix):
    query = supplement.SupplementQuery("reference-ticker", "AAPL", "2026-08-31", "2026-08-31", "AAPL")
    with pytest.raises(ValueError):
        query.validate_url(query.url + suffix)
    assert query.validate_url("https://api.massive.com/v3/reference/tickers?cursor=opaque")
    for changed in (query.url.replace("https:", "http:"), query.url.replace("api.massive.com", "example.com"),
                    query.url.replace("/v3/reference/tickers", "/v3/trades/AAPL")):
        with pytest.raises(transport.ResearchCaptureError):
            query.validate_url(changed)


@pytest.mark.parametrize("field,value", [("issuer_query_is_issue_identity", True),
    ("automatic_alias_stitching", True), ("training_ready", 0), ("native_v5_qualified", True),
    ("retry_count", 1), ("maximum_pages_per_query", 99)])
def test_plan_changed_before_network_rejected(tmp_path, monkeypatch, field, value):
    root, evidence, evidence_sha, _, queries = _plan(tmp_path)
    plan = json.loads((root / "plan.json").read_bytes())
    plan[field] = value
    raw = transport.canonical(plan)
    (root / "plan.json").chmod(0o600)
    (root / "plan.json").write_bytes(raw)
    calls = _network(monkeypatch, queries)
    with pytest.raises(transport.ResearchCaptureError):
        supplement.capture_supplement(root=root, api_key="synthetic-test-only",
            plan_sha256=transport.digest(raw), evidence=evidence, evidence_sha256=evidence_sha)
    assert calls == [] and not (root / "STARTED.json").exists()


def test_empty_results_preserved_and_replay_is_read_only(tmp_path, monkeypatch):
    root, evidence, evidence_sha, plan_sha, queries = _plan(tmp_path)
    calls = _network(monkeypatch, queries)
    result = supplement.capture_supplement(root=root, api_key="synthetic-test-only",
        plan_sha256=plan_sha, evidence=evidence, evidence_sha256=evidence_sha)
    assert len(calls) == 340 and result["query_count"] == 340
    complete_sha = transport.digest((root / "COMPLETE.json").read_bytes())
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    replay = supplement.replay_supplement(root=root, plan_sha256=plan_sha, completion_sha256=complete_sha,
                                        evidence=evidence, evidence_sha256=evidence_sha)
    assert replay["training_ready"] is replay["native_v5_qualified"] is False
    assert replay["read_only_replay_verified"] is True and replay["page_count"] == 340
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert len(calls) == 340
    (evidence).chmod(0o600)
    evidence.write_bytes(b"changed source")
    with pytest.raises(transport.ResearchCaptureError):
        supplement.replay_supplement(root=root, plan_sha256=plan_sha, completion_sha256=complete_sha,
                                    evidence=evidence, evidence_sha256=evidence_sha)


def test_failed_response_preserved_no_retry(tmp_path, monkeypatch):
    root, evidence, evidence_sha, plan_sha, queries = _plan(tmp_path)
    calls = _network(monkeypatch, queries, failure=True)
    with pytest.raises(transport.ResearchCaptureError):
        supplement.capture_supplement(root=root, api_key="synthetic-test-only",
            plan_sha256=plan_sha, evidence=evidence, evidence_sha256=evidence_sha)
    assert len(calls) == 1 and (root / "BLOCKED.json").exists()
    assert not (root / "COMPLETE.json").exists()
    assert (root / queries[0].name / "page-0000.json.gz").exists()


def test_issuer_response_preserves_multiple_classes_not_a_join():
    query = supplement.SupplementQuery("reference-issuer", "GOOGL", "2017-01-03", "2017-01-03",
                                      "GOOGL", "0001652044")
    rows = [{"market": "stocks", "active": True, "ticker": t, "cik": query.cik,
             "composite_figi": f"different-{t}"} for t in ("GOOG", "GOOGL")]
    raw = transport.canonical({"status": "OK", "request_id": "synthetic", "results": rows})
    result = query.inspect_page(raw)
    assert result["result_count"] == 2 and result["issue_links_established"] == 0
    assert transport.parse_json(raw)["results"] == rows and result["training_ready"] is False
    rows[0]["cik"] = "0000000000"
    with pytest.raises(transport.ResearchCaptureError):
        query.inspect_page(transport.canonical({"status": "OK", "request_id": "x", "results": rows}))


@pytest.mark.parametrize("product,field", [("splits", "execution_date"), ("dividends", "ex_dividend_date")])
def test_economic_tail_bounds_and_preservation(product, field):
    query = supplement.SupplementQuery(product, "market", "2026-08-27", "2026-08-31", "market")
    body = {"status": "OK", "request_id": "synthetic", "results": [{field: "2026-08-28", "ticker": "AAPL"}]}
    assert query.inspect_page(transport.canonical(body))["result_count"] == 1
    body["results"][0][field] = "2026-08-26"
    with pytest.raises(transport.ResearchCaptureError):
        query.inspect_page(transport.canonical(body))
    with pytest.raises(transport.ResearchCaptureError):
        query.validate_url(query.url + "&cursor=x&" + field + ".lte=2027-01-01")
