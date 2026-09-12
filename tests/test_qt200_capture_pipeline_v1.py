"""Remote LSF regressions for bounded acquisition, not economic qualification."""

import gzip
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from rl_quant.data_sources.massive import qt200_minute_capture_v1 as minute
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_capture_pipeline_v1 import (
    PipelineStopped,
    RequestPacer,
    run_bounded,
)


def test_request_grants_share_one_global_rate_limit():
    pacer = RequestPacer()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(pacer.acquire, deadline=time.monotonic() + 10) for _ in range(5)]
        grants = sorted(f.result() for f in futures)
    assert all(b - a >= 0.3 - 1e-9 for a, b in zip(grants, grants[1:]))
    assert pacer.granted_requests == 5


def test_cancelled_pacer_and_expired_deadline_never_grant():
    pacer = RequestPacer()
    with pytest.raises(PipelineStopped):
        pacer.acquire(deadline=time.monotonic() - 1)
    pacer.cancel()
    with pytest.raises(PipelineStopped):
        pacer.acquire(deadline=time.monotonic() + 10)
    assert pacer.granted_requests == 0


def test_four_disjoint_tasks_overlap_but_results_keep_declared_order():
    barrier = threading.Barrier(4, timeout=10)
    seen = []
    lock = threading.Lock()

    def work(key):
        with lock:
            seen.append(key)
        if int(key) < 4:
            barrier.wait()
        return key + "-complete"

    keys = tuple(str(i) for i in range(12))
    result = run_bounded(keys=keys, operation=work, pacer=RequestPacer())
    assert result == tuple(k + "-complete" for k in keys)
    assert sorted(seen) == sorted(keys)


@pytest.mark.parametrize("workers", [0, 5, True, 2.0])
def test_invalid_worker_count_rejected_before_work(workers):
    calls = []
    with pytest.raises(ValueError):
        run_bounded(keys=("AAPL",), operation=calls.append, pacer=RequestPacer(), workers=workers)
    assert calls == []


def test_duplicate_keys_rejected_before_work():
    calls = []
    with pytest.raises(ValueError):
        run_bounded(keys=("AAPL", "AAPL"), operation=calls.append, pacer=RequestPacer())
    assert calls == []


def test_failure_stops_new_work_and_closes_shared_requests():
    pacer = RequestPacer()
    calls = []

    def work(key):
        calls.append(key)
        raise ValueError("synthetic source failure")

    with pytest.raises((ValueError, PipelineStopped)):
        run_bounded(keys=("AAPL", "MSFT", "NVDA"), operation=work, pacer=pacer, workers=1)
    assert calls == ["AAPL"]
    with pytest.raises(PipelineStopped):
        pacer.acquire(deadline=time.monotonic() + 10)


def test_parallel_captures_use_actual_transport_replay_and_identical_raw_pages(tmp_path, monkeypatch):
    originals = {}
    for ticker in ("AAPL", "MSFT"):
        query = minute._query(ticker)
        originals[ticker] = transport.canonical(dict(status="OK", request_id="synthetic-request-id", ticker=ticker,
            adjusted=False, results=[dict(t=int(datetime.fromisoformat(query.start).replace(
                tzinfo=transport.ET).timestamp()) * 1000,
                o=10, h=12, l=9, c=11, v=100)], resultsCount=1))

    class Response:
        code = 200
        headers = {"Content-Type": "application/json"}

        def __init__(self, url):
            self.url = url

        def geturl(self):
            return self.url

        def read(self, cap):
            ticker = self.url.split("/ticker/")[1].split("/")[0]
            return originals[ticker][:cap]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Opener:
        def open(self, req, timeout):
            assert req.get_header("Authorization") == "Bearer synthetic-only"
            return Response(req.full_url)

    monkeypatch.setattr(transport.request, "build_opener", lambda *_: Opener())
    pacer = RequestPacer()

    def work(ticker):
        root = tmp_path / ticker
        root.mkdir()
        plan = transport.canonical(dict(**minute.plan_fields(ticker),
            minute_implementation_sha256=transport.digest(Path(minute.__file__).read_bytes()),
            capture_implementation_sha256=transport.digest(Path(transport.__file__).read_bytes())))
        transport.write_once(root / "plan.json", plan)
        result = minute.capture_minute(root=root, ticker=ticker, api_key="synthetic-only",
            plan_sha256=transport.digest(plan), request_pacer=pacer)
        files = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        minute.replay_minute(root=root, ticker=ticker, plan_sha256=transport.digest(plan),
            completion_sha256=transport.digest((root / "COMPLETE.json").read_bytes()))
        assert files == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        body = root / minute._query(ticker).name / "page-0000.json.gz"
        assert body.read_bytes() == gzip.compress(originals[ticker], compresslevel=9, mtime=0)
        return result["capture_complete"]

    assert run_bounded(keys=("AAPL", "MSFT"), operation=work, pacer=pacer, workers=2) == (True, True)
    assert pacer.granted_requests == 2
