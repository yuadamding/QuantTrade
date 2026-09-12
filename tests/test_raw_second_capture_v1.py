"""LSF transport acceptance; synthetic HTTP replies are not real-data evidence."""

from dataclasses import asdict
import gzip
from hashlib import sha256
import json

import pytest
import torch

from rl_quant.data_sources.massive import raw_second_capture_v1 as capture
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.datasets.massive_raw_seconds_v1 import SecondCaptureRef, SecondQuery, _safe_url

pytestmark = pytest.mark.lsf_gpu
START = 1_483_453_800_000


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def _plan(tmp_path):
    root = tmp_path / "capture"
    queries = (SecondQuery("AAPL", START, START + 3599_000),)
    digest = capture.publish_second_capture_plan(root=root, queries=queries)
    return root, digest, queries[0]


def _network(monkeypatch, query, *, code=200, malformed=False, unknown=False):
    calls = []
    cursor = query.url.replace(f"/{query.start_ms}/", f"/{query.start_ms + 1000}/").split("?")[0] + "?cursor=second-page"

    class Response:
        def __init__(self, url, raw):
            self.url, self.raw, self.code = url, raw, code
            self.headers = {"Content-Type": "application/json"}

        def geturl(self):
            return self.url

        def read(self, size):
            return self.raw[:size]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Opener:
        def open(self, request, timeout):
            assert timeout == 45 and request.get_method() == "GET"
            assert request.get_header("Authorization") == "Bearer synthetic-transport-only"
            assert "synthetic-transport-only" not in request.full_url
            calls.append(request.full_url)
            if code != 200:
                return Response(request.full_url, b'{"status":"ERROR","request_id":"synthetic-denial"}')
            first = request.full_url == query.url
            row = dict(t=query.start_ms if first else query.start_ms + 1000,
                       o=100, h=101, l=99, c=100.25, v=1200, vw=100.125, n=17)
            if malformed:
                row["v"] = -1
            body = dict(status="OK", request_id="synthetic-page", ticker=query.ticker,
                        adjusted=False, resultsCount=1, results=[row])
            if first:
                body["next_url"] = cursor
            elif unknown:
                body["queryCount"] = 50000
            return Response(request.full_url, json.dumps(body, indent=2).encode())

    monkeypatch.setattr(transport.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(transport.time, "sleep", lambda *_: None)
    return calls


def test_advancing_cursor_preserves_second_scope():
    query = SecondQuery("BRK.B", START, START + 3599_000)
    changed = query.url.replace(str(START), str(START + 1000)).split("?")[0] + "?cursor=next"
    _safe_url(changed, query)
    assert capture.SecondHTTPQuery(**asdict(query)).validate_url(changed) == changed


@pytest.mark.parametrize("case", ["host", "credentials", "empty-key", "ticker", "end", "before", "after", "minute", "duplicate", "empty-cursor", "adjusted", "fragment"])
def test_unsafe_or_widened_second_cursor_rejected(case):
    query = SecondQuery("AAPL", START, START + 3599_000)
    url = query.url + "&cursor=next"
    changes = {
        "host": url.replace("api.massive.com", "example.com"),
        "credentials": url + "&apiKey=synthetic-transport-only", "empty-key": url + "&apiKey=",
        "ticker": url.replace("AAPL", "WMT"), "end": url.replace(str(query.end_ms), str(query.end_ms + 1000)),
        "before": url.replace(str(START), str(START - 1000)),
        "after": url.replace(str(START), str(query.end_ms + 1000)),
        "minute": url.replace("/second/", "/minute/"), "duplicate": url + "&cursor=again",
        "empty-cursor": url.replace("cursor=next", "cursor="), "adjusted": url.replace("adjusted=false", "adjusted=true"),
        "fragment": url + "#fragment",
    }
    with pytest.raises(ValueError):
        _safe_url(changes[case], query)


def test_bounded_plan_rejects_excess_scope_before_publication(tmp_path):
    query = SecondQuery("AAPL", START, START + 3599_000)
    populations = ((), (query, query), (SecondQuery("AAPL", START, START + 3600_000),),
                   (SecondQuery("NOTQT200", START, START + 1000),),
                   (query, SecondQuery("AAPL", START + 1000, START + 2000)),
                   tuple(SecondQuery("AAPL", START + i * 4_000_000, START + i * 4_000_000 + 1000) for i in range(25)))
    for index, population in enumerate(populations):
        root = tmp_path / str(index)
        with pytest.raises(ValueError):
            capture.publish_second_capture_plan(root=root, queries=population)
        assert not root.exists()


def test_http_capture_to_raw_sources_is_lossless_and_replay_is_read_only(tmp_path_factory, monkeypatch):
    tmp_path = tmp_path_factory.mktemp("second-source-handoff")
    root, plan, query = _plan(tmp_path)
    calls = _network(monkeypatch, query)
    result = capture.capture_seconds(root=root, plan_sha256=plan, api_key="synthetic-transport-only")
    assert len(calls) == 2 and result["training_ready"] is False
    complete = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    out = tmp_path / "raw"
    sources = capture.materialize_second_sources(root=root, plan_sha256=plan, completion_sha256=complete, output=out)
    ref = SecondCaptureRef(**sources["captures"][0])
    actual_query, pages = ref.load()
    assert actual_query == query and len(pages) == 2
    for i, page in enumerate(pages):
        assert page.body == gzip.decompress((root / capture.SecondHTTPQuery(**asdict(query)).name / f"page-{i:04d}.json.gz").read_bytes())
        body = json.loads(page.body)
        assert body["results"][0]["vw"] == 100.125 and body["results"][0]["n"] == 17
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    replay = capture.verify_second_sources(root=root, plan_sha256=plan, completion_sha256=complete,
        output=out, output_sha256=sha256((out / "COMPLETE.json").read_bytes()).hexdigest())
    assert replay["nonmaterializing"] and replay["raw_response_identity_verified"] and not replay["training_ready"]
    assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    assert len(calls) == 2
    with pytest.raises(transport.ResearchCaptureError):
        capture.capture_seconds(root=root, plan_sha256=plan, api_key="synthetic-transport-only")


@pytest.mark.parametrize("mode", ["denied", "limit"])
def test_failed_acquisition_retains_evidence_without_completion(tmp_path, monkeypatch, mode):
    root, plan, query = _plan(tmp_path)
    calls = _network(monkeypatch, query, code=403 if mode == "denied" else 200, unknown=mode == "limit")
    with pytest.raises(transport.ResearchCaptureError):
        capture.capture_seconds(root=root, plan_sha256=plan, api_key="synthetic-transport-only")
    assert not (root / "COMPLETE.json").exists() and (root / "BLOCKED.json").is_file()
    assert len(calls) == (1 if mode == "denied" else 2)
    assert tuple(root.glob("second-*/page-0000.json.gz"))


def test_preserved_bad_ohlcv_cannot_become_a_raw_training_capture(tmp_path, monkeypatch):
    root, plan, query = _plan(tmp_path)
    _network(monkeypatch, query, malformed=True)
    capture.capture_seconds(root=root, plan_sha256=plan, api_key="synthetic-transport-only")
    complete = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Invalid raw field"):
        capture.materialize_second_sources(root=root, plan_sha256=plan, completion_sha256=complete, output=tmp_path / "raw")
    assert not (tmp_path / "raw/COMPLETE.json").exists()


def test_corrupted_capture_blocks_handoff_before_output(tmp_path, monkeypatch):
    root, plan, query = _plan(tmp_path)
    _network(monkeypatch, query)
    capture.capture_seconds(root=root, plan_sha256=plan, api_key="synthetic-transport-only")
    complete = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    body = root / capture.SecondHTTPQuery(**asdict(query)).name / "page-0000.json.gz"
    body.chmod(0o600)
    body.write_bytes(gzip.compress(b"{}", mtime=0))
    with pytest.raises(ValueError):
        capture.materialize_second_sources(root=root, plan_sha256=plan, completion_sha256=complete, output=tmp_path / "raw")
    assert not (tmp_path / "raw").exists()
