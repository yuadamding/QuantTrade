"""LSF-only direct-packed HTTP regression; no live provider acquisition."""

import base64
from dataclasses import asdict, replace
import gzip
from hashlib import sha256
import json
import os
import subprocess
import sys

import pytest
import torch

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive import raw_second_capture_v1 as legacy
from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
from rl_quant.datasets import massive_raw_second_packed_v1 as packed
from rl_quant.datasets.massive_raw_seconds_v1 import (
    CapturedSecondPage, RawSecondContract, RawSecondWindowRef, SecondCaptureRef, SecondQuery,
    load_raw_second_window,
)
from test_raw_second_alias_capture_v1 import _route


pytestmark = pytest.mark.lsf_gpu
START = 1_483_453_800_000
KEY = "synthetic-direct-only"


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def _query(ticker="AAPL"):
    return SecondQuery(ticker, START, START + 3599_000)


def _url(query, page):
    return (query.url if page == 0 else query.url.replace(
        f"/{query.start_ms}/", f"/{query.start_ms + page * 1000}/").split("?")[0] + f"?cursor=page-{page}")


def _body(query, number=0, *, next_page=None, empty=False, **changes):
    rows = [] if empty else [dict(t=query.start_ms + number * 1000,
        o=100, h=101, l=99, c=100.25, v=1200, vw=100.125, n=17)]
    body = dict(status="OK", ticker=query.ticker, adjusted=False,
                request_id=f"synthetic-{number}", resultsCount=len(rows), results=rows)
    if next_page is not None:
        body["next_url"] = _url(query, next_page)
    body.update(changes)
    return json.dumps(body, indent=2).encode()


def _network(monkeypatch, responses):
    """The real header-only transport receives a bounded, deterministic sequence."""
    calls = []

    class Response:
        def __init__(self, url, code, body):
            self.url, self.code, self.body = url, code, body
            self.headers = {"Content-Type": "application/json"}

        def geturl(self):
            return self.url

        def read(self, size):
            return self.body[:size]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Opener:
        def open(self, request, timeout):
            assert request.get_method() == "GET" and timeout == 45
            assert request.get_header("Authorization") == "Bearer " + KEY
            assert KEY not in request.full_url
            expected, status, body = responses[len(calls)]
            assert request.full_url == expected
            calls.append(request.full_url)
            return Response(request.full_url, status, body)

    monkeypatch.setattr(transport.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(transport.time, "sleep", lambda *_: None)
    # Equal original receipt times make legacy/direct native identities comparable.
    monkeypatch.setattr(transport.time, "time_ns", lambda: 1_780_000_000_000_000_000)
    return calls


def _inventory(root):
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def _rewrite(path, body):
    path.chmod(0o600)
    path.write_bytes(body)


def _frames(root):
    """Inspect retained originals even when no packed index could be committed."""
    body = (root / "packed" / packed.PACK_NAME).read_bytes()
    assert body.startswith(packed.MAGIC)
    offset, frames = len(packed.MAGIC), []
    while offset < len(body):
        size = int.from_bytes(body[offset:offset + 4], "big")
        header = json.loads(body[offset + 4:offset + 4 + size])
        start = offset + 4 + size
        end = start + header["page"]["gzip_bytes"]
        frames.append((header, gzip.decompress(body[start:end])))
        offset = end
    assert offset == len(body)
    return frames


def _complete(root, query):
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,))
    terminal = direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    complete = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    replay = direct.replay_direct_second_capture(root=root, plan_sha256=plan, completion_sha256=complete)
    return plan, terminal, complete, replay


def _no_retry(root, plan, calls):
    before, requests = _inventory(root), len(calls)
    with pytest.raises((ValueError, OSError)):
        direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    assert _inventory(root) == before and len(calls) == requests


@pytest.mark.parametrize("empty", [False, True])
def test_direct_original_pages_masks_and_native_identity_match_legacy(tmp_path, monkeypatch, empty):
    query = _query("BRK.B")
    pages = [_body(query, 0, next_page=1, empty=empty), _body(query, 1, empty=empty)]
    replies = [(_url(query, i), 200, body) for i, body in enumerate(pages)]
    calls = _network(monkeypatch, replies * 3)
    legacy_roots = (tmp_path / "legacy-before", tmp_path / "legacy-after")
    left = legacy_roots[0]
    old_plan = legacy.publish_second_capture_plan(root=left, queries=(query,))
    legacy.capture_seconds(root=left, plan_sha256=old_plan, api_key=KEY)
    old_complete = sha256((left / "COMPLETE.json").read_bytes()).hexdigest()
    old_inventory = _inventory(left)
    root = tmp_path / "direct"
    plan, terminal, completion, replay = _complete(root, query)
    assert set(_inventory(root)) == {"plan.json", "STARTED.json", "COMPLETE.json",
                                    "packed/" + packed.PACK_NAME, "packed/" + packed.INDEX_NAME}
    assert replay["query_count"] == 1 and replay["page_count"] == 2
    assert replay["raw_response_bytes"] == sum(map(len, pages))
    assert not terminal["training_ready"] and not replay["training_ready"]
    original = legacy.materialize_second_sources(root=left, plan_sha256=old_plan,
        completion_sha256=old_complete, output=tmp_path / "native")
    a = SecondCaptureRef(**original["captures"][0])
    b = packed.PackedSecondCaptureRef.from_dict(replay["captures"][0])
    assert a.load() == b.load() and a.manifest_sha256 == b.manifest_sha256
    assert [p.body for p in b.load()[1]] == pages
    assert [raw for _, raw in _frames(root)] == pages
    common = dict(asset_ids=("SOURCE-PARITY-ONLY",), start_ms=START, seconds=8,
        decision_ms=START + 901_000, contract=RawSecondContract("developer-delayed-assumption", 900_000))
    first = load_raw_second_window(RawSecondWindowRef(captures=(a,), **common), device="cuda:0")
    second = load_raw_second_window(RawSecondWindowRef(captures=(b,), **common), device="cuda:0")
    for name in ("raw_ohlcv", "observed_mask", "known_mask", "padding_mask", "second_timestamp_ms",
                 "available_at_ms", "decision_timestamp_ms"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert bool(second.observed_mask.any()) is not empty
    assert not bool(second.observed_mask[:, :, 1:].any())  # later row is still unavailable
    before = _inventory(root)
    assert direct.replay_direct_second_capture(root=root, plan_sha256=plan,
        completion_sha256=completion) == replay
    assert _inventory(root) == before and _inventory(left) == old_inventory
    # New mode cannot migrate or change the default legacy output branch.
    after_plan = legacy.publish_second_capture_plan(root=legacy_roots[1], queries=(query,))
    legacy.capture_seconds(root=legacy_roots[1], plan_sha256=after_plan, api_key=KEY)
    assert after_plan == old_plan and _inventory(legacy_roots[1]) == old_inventory
    assert json.loads((left / "plan.json").read_bytes())["schema"] == legacy.SCHEMA + "-plan"
    assert len(calls) == 6
    _no_retry(root, plan, calls)


@pytest.mark.parametrize("slot", ["META", "ELV"])
def test_direct_alias_routes_preserve_provider_ticker_and_block_unobserved_day(tmp_path, monkeypatch, slot):
    route, query = _route(tmp_path, slot)
    original = _body(query)
    calls = _network(monkeypatch, [(query.url, 200, original)])
    root = tmp_path / "direct"
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,), alias_routes=(route,))
    direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    complete = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    result = direct.replay_direct_second_capture(root=root, plan_sha256=plan, completion_sha256=complete)
    actual, pages = packed.PackedSecondCaptureRef.from_dict(result["captures"][0]).load()
    assert actual == query and pages[0].body == original
    assert json.loads(original)["ticker"] == route.provider_ticker and route.fixed_slot_ticker != actual.ticker
    with pytest.raises(ValueError):
        direct.publish_direct_second_capture_plan(root=tmp_path / "unobserved", queries=(replace(query,
            start_ms=query.start_ms + 86400_000, end_ms=query.end_ms + 86400_000),), alias_routes=(route,))
    assert not (tmp_path / "unobserved").exists() and len(calls) == 1


def test_changed_direct_plan_is_rejected_before_any_request_or_start(tmp_path, monkeypatch):
    query = _query()
    calls = _network(monkeypatch, [(query.url, 200, _body(query))])
    root = tmp_path / "direct"
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,))
    _rewrite(root / "plan.json", (root / "plan.json").read_bytes() + b" ")
    with pytest.raises(ValueError):
        direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    assert not calls and set(_inventory(root)) == {"plan.json"}


def test_transport_rejects_caller_owned_duck_typed_storage_sink(tmp_path, monkeypatch):
    query = _query()
    calls = _network(monkeypatch, [(query.url, 200, _body(query))])
    root = tmp_path / "legacy"
    plan = legacy.publish_second_capture_plan(root=root, queries=(query,))
    queries = legacy._plan(root, plan, current=True)
    with pytest.raises(ValueError):
        transport._capture_queries(root=root, api_key=KEY, plan_sha256=plan, queries=queries,
            schema=legacy.SCHEMA, maximum_bytes=legacy.MAX_BYTES, maximum_pages=legacy.MAX_PAGES,
            _packed_sink=object())
    assert not calls and set(_inventory(root)) == {"plan.json"}


def test_direct_receipts_replay_in_a_fresh_read_only_process(tmp_path, monkeypatch):
    query = _query()
    calls = _network(monkeypatch, [(query.url, 200, _body(query))])
    root = tmp_path / "direct"
    plan, _, complete, replay = _complete(root, query)
    before = _inventory(root)
    script = """import json,sys
from pathlib import Path
from rl_quant.data_sources.massive.raw_second_direct_capture_v1 import replay_direct_second_capture
result = replay_direct_second_capture(root=Path(sys.argv[1]), plan_sha256=sys.argv[2], completion_sha256=sys.argv[3])
print(json.dumps(result, sort_keys=True))
"""
    child = subprocess.run([sys.executable, "-B", "-c", script, str(root), plan, complete],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == replay
    assert _inventory(root) == before and len(calls) == 1


@pytest.mark.parametrize("case", ["http", "json", "native", "out-of-range", "loop", "cap", "foreign-url"])
def test_failed_responses_preserve_original_frames_without_commit_or_retry(tmp_path, monkeypatch, case):
    query = _query()
    body, status = _body(query), 200
    if case == "http":
        body, status = b'{"status":"ERROR","request_id":"denied"}', 403
    elif case == "json":
        body = b'{"truncated":'
    elif case in ("native", "out-of-range"):
        value = json.loads(body)
        value["results"][0]["v" if case == "native" else "t"] = -1 if case == "native" else query.end_ms + 1000
        body = json.dumps(value).encode()
    elif case == "loop":
        body = _body(query, next_page=0)
    elif case == "cap":
        body = _body(query, queryCount=50000)
    else:
        body = _body(query, next_url="https://example.com/unapproved")
    calls = _network(monkeypatch, [(query.url, status, body)])
    root = tmp_path / "direct"
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,))
    with pytest.raises((ValueError, OSError)):
        direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    assert len(calls) == 1 and (root / "BLOCKED.json").is_file()
    assert not (root / "COMPLETE.json").exists() and not (root / "packed" / packed.INDEX_NAME).exists()
    frames = _frames(root)
    assert [raw for _, raw in frames] == [body]
    receipt = json.loads(base64.b64decode(frames[0][0]["source_receipt_base64"]))
    assert KEY.encode() not in transport.canonical(receipt)
    _no_retry(root, plan, calls)


def test_cumulative_full_page_reservation_stops_before_unretainable_get(tmp_path, monkeypatch):
    query = _query()
    pages = []
    for number in range(4):
        value = json.loads(_body(query, number, next_page=number + 1))
        value["opaque_original_padding"] = ""
        minimal = transport.canonical(value)
        value["opaque_original_padding"] = "x" * (transport.MAX_PAGE_BYTES - len(minimal))
        page = transport.canonical(value)
        assert len(page) == transport.MAX_PAGE_BYTES
        pages.append(page)
    calls = _network(monkeypatch, [(_url(query, i), 200, body) for i, body in enumerate(pages)])
    root = tmp_path / "direct"
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,))
    with pytest.raises(ValueError):
        direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    assert len(calls) == 3  # 128,000,000 - 3 * 32 MiB cannot reserve another permitted page
    assert [raw for _, raw in _frames(root)] == pages[:3]
    assert (root / "BLOCKED.json").is_file() and not (root / "COMPLETE.json").exists()
    _no_retry(root, plan, calls)


def test_injected_pre_get_deadline_preserves_prior_frames_without_another_request(tmp_path, monkeypatch):
    query = _query()
    first = _body(query, next_page=1)
    calls = _network(monkeypatch, [(query.url, 200, first)])
    root = tmp_path / "direct"
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,))
    original = direct._DirectSecondSink.before_request

    def deadline(self, url):
        original(self, url)
        if self.http_attempts == 1:
            raise TimeoutError("synthetic pre-GET deadline")

    monkeypatch.setattr(direct._DirectSecondSink, "before_request", deadline)
    with pytest.raises(ValueError):
        direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    assert len(calls) == 1 and [body for _, body in _frames(root)] == [first]
    blocked = json.loads((root / "BLOCKED.json").read_bytes())
    assert blocked["error_type"] == "TimeoutError" and blocked["http_attempts"] == 1
    assert not (root / "COMPLETE.json").exists()
    _no_retry(root, plan, calls)


def test_oversized_response_cannot_publish_a_truncated_original(tmp_path, monkeypatch):
    query = _query()
    calls = _network(monkeypatch, [(query.url, 200, b"x" * (transport.MAX_PAGE_BYTES + 1))])
    root = tmp_path / "direct"
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,))
    with pytest.raises(ValueError):
        direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    assert len(calls) == 1 and _frames(root) == []
    blocked = json.loads((root / "BLOCKED.json").read_bytes())
    assert blocked["http_attempts"] == 1 and not blocked["capture_complete"]
    assert not (root / "COMPLETE.json").exists() and not (root / "packed" / packed.INDEX_NAME).exists()
    _no_retry(root, plan, calls)


@pytest.mark.parametrize("damage", ["plan", "start", "index", "receipt", "payload", "terminal", "extra"])
def test_direct_source_and_receipt_tampering_blocks_replay(tmp_path, monkeypatch, damage):
    query = _query()
    calls = _network(monkeypatch, [(query.url, 200, _body(query))])
    root = tmp_path / "direct"
    plan, _, complete, replay = _complete(root, query)
    if damage in ("plan", "start", "terminal", "index"):
        target = {"plan": root / "plan.json", "start": root / "STARTED.json",
            "terminal": root / "COMPLETE.json", "index": root / "packed" / packed.INDEX_NAME}[damage]
        _rewrite(target, target.read_bytes() + b" ")
    elif damage == "extra":
        (root / "unapproved.json").write_bytes(b"{}")
    else:
        index = json.loads((root / "packed" / packed.INDEX_NAME).read_bytes())
        page = index["captures"][0]["pages"][0]
        target = root / "packed" / packed.PACK_NAME
        raw = bytearray(target.read_bytes())
        if damage == "receipt":
            header = json.loads(raw[page["frame_offset"] + 4:page["offset"]])
            encoded = header["source_receipt_base64"].encode()
            offset = raw.index(encoded, page["frame_offset"] + 4, page["offset"])
        else:
            offset = page["offset"] + page["gzip_bytes"] // 2
        raw[offset] ^= 1
        _rewrite(target, bytes(raw))
    before = _inventory(root)
    with pytest.raises((ValueError, OSError)):
        direct.replay_direct_second_capture(root=root, plan_sha256=plan, completion_sha256=complete)
    assert before == _inventory(root) and len(calls) == 1
    assert replay["query_count"] == 1


@pytest.mark.parametrize("damage", ["query_count", "page_count", "http_attempts", "raw_response_bytes",
                                   "census", "population", "qualification"])
def test_rehashed_terminal_cannot_replace_reconstructed_evidence(tmp_path, monkeypatch, damage):
    query = _query()
    calls = _network(monkeypatch, [(query.url, 200, _body(query))])
    root = tmp_path / "direct"
    plan, terminal, _, _ = _complete(root, query)
    if damage == "census":
        terminal["page_census"][0]["result_count"] += 1
    elif damage == "population":
        terminal["queries"] = []
    elif damage == "qualification":
        terminal["training_ready"] = True
    else:
        terminal[damage] += 1
    body = transport.canonical(terminal)
    _rewrite(root / "COMPLETE.json", body)
    before = _inventory(root)
    with pytest.raises(ValueError):
        direct.replay_direct_second_capture(root=root, plan_sha256=plan,
                                            completion_sha256=sha256(body).hexdigest())
    assert _inventory(root) == before and len(calls) == 1


@pytest.mark.parametrize("damage", ["status", "url", "predecessor", "query", "body", "chronology", "index-type"])
def test_valid_rehashed_pack_cannot_authorize_invalid_transport_receipts(tmp_path, monkeypatch, damage):
    query = _query()
    pages = [_body(query, 0, next_page=1), _body(query, 1)]
    calls = _network(monkeypatch, [(_url(query, i), 200, body) for i, body in enumerate(pages)])
    root = tmp_path / "direct"
    plan, terminal, _, _ = _complete(root, query)
    index = json.loads((root / "packed" / packed.INDEX_NAME).read_bytes())
    proofs = []
    # Rebuild *all* physical and terminal hashes around changed receipt bytes.
    # Native packed parsing must pass, but provider acquisition semantics must not.
    with packed.SecondPackWriter(tmp_path / "resealed", provenance=index["provenance"]) as writer:
        writer.begin_query(query)
        for number, (header, original) in enumerate(_frames(root)):
            receipt = json.loads(base64.b64decode(header["source_receipt_base64"]))
            receipt["predecessor_page_sha256"] = proofs[-1]["sha256"] if proofs else None
            if number == (1 if damage == "predecessor" else 0):
                if damage == "status":
                    receipt["http_status"] = 403
                elif damage == "url":
                    receipt["request_url"] = "https://example.com/changed"
                elif damage == "predecessor":
                    receipt["predecessor_page_sha256"] = "f" * 64
                elif damage == "query":
                    receipt["query"]["ticker"] = "MSFT"
                elif damage == "body":
                    receipt["raw_body_sha256"] = "0" * 64
                elif damage == "chronology":
                    receipt["requested_at_ns"] -= 1
                else:
                    receipt["request_index"] = False  # equal to zero is not an integer index
            raw_receipt = transport.canonical(receipt)
            writer.append_page(CapturedSecondPage(header["page"]["request_url"],
                header["page"]["received_at_ms"], original), source_receipt=raw_receipt)
            proofs.append(dict(bytes=len(raw_receipt), sha256=sha256(raw_receipt).hexdigest()))
        manifest = writer.finish_query()
        proof = writer.finalize()
    for name in (packed.PACK_NAME, packed.INDEX_NAME):
        _rewrite(root / "packed" / name, (tmp_path / "resealed" / name).read_bytes())
    verified = packed.verify_packed_second_store(root=root / "packed", index_sha256=proof["index_sha256"])
    assert verified["capture_count"] == 1 and verified["page_count"] == 2
    terminal.update(packed_index_sha256=proof["index_sha256"], packed_sha256=proof["pack_sha256"],
                    packed_bytes=proof["pack_bytes"], index_bytes=proof["index_bytes"])
    query_complete = dict(schema=direct.SCHEMA + "-query-complete", query=asdict(query), plan_sha256=plan,
        pages=proofs, page_count=2, pagination_complete=True, result_count=2,
        empty_provider_response_proves_no_trading=False, point_in_time_qualified=False, training_ready=False)
    complete_bytes = transport.canonical(query_complete)
    terminal["queries"][0]["completion"] = dict(bytes=len(complete_bytes),
        sha256=sha256(complete_bytes).hexdigest(), manifest_sha256=manifest)
    body = transport.canonical(terminal)
    _rewrite(root / "COMPLETE.json", body)
    before = _inventory(root)
    with pytest.raises(ValueError, match="Direct receipt|Direct capture receipt|Direct original"):
        direct.replay_direct_second_capture(root=root, plan_sha256=plan,
                                            completion_sha256=sha256(body).hexdigest())
    assert _inventory(root) == before and len(calls) == 2


@pytest.mark.parametrize("stage", ["append", "finish", "index", "terminal"])
def test_injected_publication_failures_retain_frames_and_never_retry(tmp_path, monkeypatch, stage):
    query = _query()
    body = _body(query)
    calls = _network(monkeypatch, [(query.url, 200, body)])
    root = tmp_path / "direct"
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,))
    if stage in ("append", "finish"):
        name = "append_page" if stage == "append" else "finish_query"
        original = getattr(packed.SecondPackWriter, name)

        def failure(self, *args, **kwargs):
            original(self, *args, **kwargs)
            raise OSError("synthetic publication interruption")

        monkeypatch.setattr(packed.SecondPackWriter, name, failure)
    elif stage == "index":
        original = packed._write

        def failed_index(path, raw):
            if path.name == packed.INDEX_NAME:
                raise OSError("synthetic index interruption")
            return original(path, raw)

        monkeypatch.setattr(packed, "_write", failed_index)
    else:
        original = transport.write_once

        def failed_terminal(path, raw):
            if path == root / "COMPLETE.json":
                raise OSError("synthetic terminal interruption")
            return original(path, raw)

        monkeypatch.setattr(transport, "write_once", failed_terminal)
    with pytest.raises((ValueError, OSError)):
        direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    assert _frames(root)[0][1] == body and len(calls) == 1
    assert not (root / "COMPLETE.json").exists()
    _no_retry(root, plan, calls)
