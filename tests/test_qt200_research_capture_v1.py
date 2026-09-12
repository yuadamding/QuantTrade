"""Source-capture unit tests, not economic or provider qualification.

Execute on the approved remote worker only. Network stubs are confined to
testing transport/persistence; they never represent accepted real sources.
"""

import gzip
import json
from dataclasses import asdict

import pytest

from rl_quant.data_sources.massive import qt200_research_capture_v1 as capture


@pytest.mark.parametrize("query", capture.pilot_queries())
def test_frozen_pilot_urls(query):
    assert capture.validate_url(query.url, query) == query.url
    assert "apiKey" not in query.url
    assert query.start == "2017-01-03"


@pytest.mark.parametrize("url", [
    "http://api.massive.com/v3/trades/AAPL?cursor=x",
    "https://example.com/v3/trades/AAPL?cursor=x",
    "https://api.massive.com:443/v3/trades/AAPL?cursor=x",
    "https://user@api.massive.com/v3/trades/AAPL?cursor=x",
    "https://api.massive.com/v3/trades/WMT?cursor=x",
    "https://api.massive.com/v3/trades/AAPL?apiKey=test-only-not-a-real-key",
    "https://api.massive.com/v3/trades/AAPL?APIKEY=test-only-not-a-real-key",
    "https://api.massive.com/v3/trades/AAPL?cursor=x&cursor=y",
    "https://api.massive.com/v3/trades/AAPL?cursor=x&timestamp=2026-01-01",
    "https://api.massive.com/v3/trades/AAPL?cursor=x&order=desc",
    "https://api.massive.com/v3/trades/AAPL?cursor=x&limit=100000",
    "https://api.massive.com/v3/trades/AAPL?cursor=x#fragment",
    "https://api.massive.com/v3/trades/AAPL?cursor=",
])
def test_unsafe_cursors_rejected(url):
    with pytest.raises((ValueError, capture.ResearchCaptureError)):
        capture.validate_url(url, capture.PilotQuery("trades", "AAPL", "2017-01-03", "2017-01-03"))


def test_aggregate_cursor_can_advance_start_without_changing_scope():
    query = capture.PilotQuery("day", "AAPL", "2017-01-03", "2017-12-29")
    url = "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/1483506000000/2017-12-29?cursor=x"
    assert capture.validate_url(url, query) == url
    for changed in (url.replace("1483506000000", "1483156800000"),
                    url.replace("2017-12-29", "2026-08-26"), url + "&adjusted=true"):
        with pytest.raises(capture.ResearchCaptureError):
            capture.validate_url(changed, query)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'[]', b'not-json'])
def test_invalid_json_rejected(raw):
    with pytest.raises(capture.ResearchCaptureError):
        capture.parse_json(raw)


def test_blank_ids_duplicates_corrections_and_price_only_messages_preserved():
    query = capture.PilotQuery("trades", "AAPL", "2017-01-03", "2017-01-03")
    rows = [{"id": "", "size": 57200, "correction": 12},
            {"id": "", "size": 57200, "correction": 12},
            {"id": "p", "size": 0, "conditions": [38, 41], "correction": 0}]
    raw = capture.canonical({"status": "OK", "request_id": "synthetic-page", "results": rows})
    summary = capture.inspect_page(raw, query)
    assert capture.parse_json(raw)["results"] == rows
    assert summary["result_count"] == 3
    assert summary["blank_or_absent_ids"] == 2
    assert summary["zero_size_records"] == 1
    assert summary["raw_correction_counts"] == {"12": 2, "0": 1}
    assert summary["records_deduplicated"] == summary["correction_links_established"] == 0
    assert summary["training_ready"] is False


def test_missing_vwap_is_not_fabricated():
    query = capture.pilot_queries()[0]
    body = {"status": "OK", "request_id": "synthetic", "ticker": query.ticker,
            "adjusted": False, "resultsCount": 1,
            "results": [{"o": 1, "h": 2, "l": 1, "c": 2, "v": 10, "t": 1483419600000}]}
    assert capture.inspect_page(capture.canonical(body), query)["vwap_present_records"] == 0
    body["adjusted"] = True
    with pytest.raises(capture.ResearchCaptureError):
        capture.inspect_page(capture.canonical(body), query)


def test_exclusive_publication_and_symlink_rejection(tmp_path):
    target = tmp_path / "receipt.json"
    capture.write_once(target, b"original")
    with pytest.raises(FileExistsError):
        capture.write_once(target, b"replacement")
    assert target.read_bytes() == b"original"
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(capture.ResearchCaptureError):
        capture.read_regular(link, 100)


class _Response:
    def __init__(self, url, raw, code=200):
        self.url, self.raw, self.code = url, raw, code
        self.headers = {"Content-Type": "application/json"}

    def geturl(self):
        return self.url

    def read(self, bound):
        return self.raw[:bound]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _plan(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    plan = {"schema": capture.SCHEMA + "-plan",
            "queries": [asdict(q) for q in capture.pilot_queries()],
            "maximum_raw_response_bytes": capture.MAX_CAPTURE_BYTES,
            "maximum_page_bytes": capture.MAX_PAGE_BYTES,
            "maximum_pages_per_query": capture.MAX_PAGES_PER_QUERY,
            "maximum_elapsed_seconds": capture.MAX_CAPTURE_SECONDS,
            "capture_implementation_sha256": capture.digest(capture.Path(capture.__file__).read_bytes()),
            "retry_count": 0, "training_ready": False}
    raw = capture.canonical(plan)
    capture.write_once(root / "plan.json", raw)
    return root, capture.digest(raw)


def _network(monkeypatch, *, failure=False, loop=False):
    calls = []

    class Opener:
        def open(self, req, timeout):
            assert req.get_method() == "GET" and timeout == 45
            assert req.get_header("Authorization") == "Bearer synthetic-transport-only"
            assert "synthetic-transport-only" not in req.full_url
            calls.append(req.full_url)
            if failure:
                return _Response(req.full_url, b'{"status":"ERROR","request_id":"synthetic"}', 403)
            query = next(q for q in capture.pilot_queries() if (
                f"/{q.ticker}" in req.full_url and (q.product == "trades" and "/v3/" in req.full_url
                    or f"/range/1/{q.product}/" in req.full_url)))
            body = {"status": "OK", "request_id": "synthetic-" + str(len(calls)), "results": []}
            if query.product != "trades":
                body.update(ticker=query.ticker, adjusted=False, resultsCount=0)
            if loop or "cursor=" not in req.full_url:
                body["next_url"] = query.url + "&cursor=next"
            return _Response(req.full_url, capture.canonical(body))

    monkeypatch.setattr(capture.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(capture.time, "sleep", lambda *_: None)
    return calls


def test_pagination_raw_retention_and_nonmaterializing_capture_replay(tmp_path, monkeypatch):
    root, sha = _plan(tmp_path)
    calls = _network(monkeypatch)
    result = capture.capture_pilot(root=root, api_key="synthetic-transport-only", plan_sha256=sha)
    assert len(calls) == 18 and result["query_count"] == 9
    assert result["training_ready"] is result["native_v5_qualified"] is False
    raw = gzip.decompress((root / "day-ABBV/page-0000.json.gz").read_bytes())
    assert json.loads(raw)["request_id"] == "synthetic-1"
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    completion_sha = capture.digest((root / "COMPLETE.json").read_bytes())
    replay = capture.replay_pilot(root=root, plan_sha256=sha, completion_sha256=completion_sha)
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after and len(calls) == 18
    assert replay["page_count"] == 18 and replay["read_only_replay_verified"] is True
    assert replay["point_in_time_qualified"] is False
    with pytest.raises(capture.ResearchCaptureError):
        capture.capture_pilot(root=root, api_key="synthetic-transport-only", plan_sha256=sha)
    assert len(calls) == 18


def test_tampered_committed_body_is_rejected(tmp_path, monkeypatch):
    root, sha = _plan(tmp_path)
    _network(monkeypatch)
    capture.capture_pilot(root=root, api_key="synthetic-transport-only", plan_sha256=sha)
    completion_sha = capture.digest((root / "COMPLETE.json").read_bytes())
    target = root / "day-ABBV/page-0000.json.gz"
    target.chmod(0o600)
    target.write_bytes(gzip.compress(b'{}', mtime=0))
    with pytest.raises(capture.ResearchCaptureError):
        capture.replay_pilot(root=root, plan_sha256=sha, completion_sha256=completion_sha)


@pytest.mark.parametrize("failure,loop", [(True, False), (False, True)])
def test_failed_or_looped_pagination_never_completes(tmp_path, monkeypatch, failure, loop):
    root, sha = _plan(tmp_path)
    calls = _network(monkeypatch, failure=failure, loop=loop)
    with pytest.raises(capture.ResearchCaptureError):
        capture.capture_pilot(root=root, api_key="synthetic-transport-only", plan_sha256=sha)
    assert (root / "BLOCKED.json").exists()
    assert not (root / "COMPLETE.json").exists()
    assert (root / "day-ABBV/page-0000.json.gz").exists()
    assert len(calls) == (1 if failure else 2)


def test_redirect_is_not_followed():
    with pytest.raises(capture.ResearchCaptureError):
        capture._NoRedirect().redirect_request(None, None, 302, None, None, "https://example.com")
