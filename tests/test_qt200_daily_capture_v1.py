"""Remote-only source adapter tests. Synthetic HTTP is not market evidence."""

import gzip
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from rl_quant.data_sources.massive import qt200_daily_capture_v1 as daily
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport


def _plan(root):
    root.mkdir()
    body = transport.canonical({**daily.plan_fields(),
        "daily_implementation_sha256": transport.digest(Path(daily.__file__).read_bytes()),
        "capture_implementation_sha256": transport.digest(Path(transport.__file__).read_bytes())})
    transport.write_once(root / "plan.json", body)
    return transport.digest(body)


def _network(patch, *, failure=False, malformed=False):
    calls = []

    class Response:
        def __init__(self, url):
            self.url = url
            self.code = 403 if failure else 200
            self.headers = {"Content-Type": "application/json"}

        def geturl(self):
            return self.url

        def read(self, bound):
            ticker = urlsplit(self.url).path.split("/")[4]
            second = "cursor=" in self.url
            row = {"o": 10, "h": 12, "l": 9, "c": 11, "v": 100,
                   "t": 1483506000000 if second else 1483419600000}
            if malformed and ticker == "AAPL" and second:
                row["v"] = -1
            rows = [] if ticker == "SNOW" else [row]
            body = {"ticker": ticker, "adjusted": False, "status": "ERROR" if failure else "OK",
                    "request_id": "synthetic-" + str(len(calls)), "results": rows,
                    "resultsCount": len(rows)}
            if not second:
                body["next_url"] = self.url + "&cursor=next"
            return transport.canonical(body)[:bound]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Opener:
        def open(self, req, timeout):
            assert req.get_header("Authorization") == "Bearer synthetic-test-only"
            assert "synthetic-test-only" not in req.full_url
            assert timeout == 45
            calls.append(req.full_url)
            return Response(req.full_url)

    patch.setattr(transport.request, "build_opener", lambda *_: Opener())
    patch.setattr(transport.time, "sleep", lambda *_: None)
    return calls


@pytest.fixture(scope="module")
def observed_capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("full-daily") / "capture"
    plan_sha = _plan(root)
    with pytest.MonkeyPatch.context() as patch:
        calls = _network(patch)
        result = daily.capture_daily(root=root, api_key="synthetic-test-only", plan_sha256=plan_sha)
    assert len(calls) == 400 and result["query_count"] == 200
    return root, plan_sha, transport.digest((root / "COMPLETE.json").read_bytes())


def test_exact_ordered_panel_and_full_requested_interval():
    assert len(daily.SYMBOLS) == len(set(daily.SYMBOLS)) == 200
    assert transport.digest(transport.canonical(daily.SYMBOLS)) == daily.ORDERED_SYMBOLS_SHA256
    assert "BRK.B" in daily.SYMBOLS and "CASH" not in daily.SYMBOLS
    queries = daily.daily_queries()
    assert tuple(q.ticker for q in queries) == daily.SYMBOLS
    assert all(q.start == "2017-01-03" and q.end == "2026-08-31" for q in queries)
    assert all(transport.validate_url(q.url, q) == q.url for q in queries)


@pytest.mark.parametrize("args", [
    ("trades", "AAPL", daily.START, daily.END),
    ("minute", "AAPL", daily.START, daily.END),
    ("day", "GOOG", daily.START, daily.END),
    ("day", "BRK-B", daily.START, daily.END),
    ("day", "../AAPL", daily.START, daily.END),
    ("day", "AAPL", "2020-01-01", daily.END),
    ("day", "AAPL", daily.START, "2026-09-01"),
])
def test_scope_is_not_caller_configurable(args):
    with pytest.raises(transport.ResearchCaptureError):
        daily.DailyQuery(*args).validate()


@pytest.mark.parametrize("field,value", [
    ("training_ready", True), ("point_in_time_qualified", True),
    ("native_v5_qualified", True), ("historical_issue_identity_qualified", True),
    ("retry_count", 1), ("maximum_pages_per_query", 999),
    ("concurrent_requests", 2), ("training_ready", 0),
])
def test_changed_plan_rejected_before_http(tmp_path, monkeypatch, field, value):
    root = tmp_path / "capture"
    _plan(root)
    plan = json.loads((root / "plan.json").read_bytes())
    plan[field] = value
    body = transport.canonical(plan)
    (root / "plan.json").chmod(0o600)
    (root / "plan.json").write_bytes(body)
    calls = _network(monkeypatch)
    with pytest.raises(transport.ResearchCaptureError):
        daily.capture_daily(root=root, api_key="synthetic-test-only", plan_sha256=transport.digest(body))
    assert calls == [] and not (root / "STARTED.json").exists()


def test_full_capture_replay_and_missingness_are_nonpromoting(observed_capture, tmp_path):
    root, plan_sha, complete_sha = observed_capture
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    replay = daily.replay_daily(root=root, plan_sha256=plan_sha, completion_sha256=complete_sha)
    assert replay["query_count"] == 200 and replay["page_count"] == 400
    output = tmp_path / "normalized"
    result = daily.normalize_daily(root=root, plan_sha256=plan_sha,
                                   completion_sha256=complete_sha, output=output)
    assert result["observed_rows"] == 398 and result["observed_bar_schema_accepted"] is True
    assert result["training_ready"] is result["full_history_dataset_complete"] is False
    assert result["identity_qualified"] is result["corporate_actions_qualified"] is False
    coverage = json.loads(gzip.decompress((output / "observed-coverage.json.gz").read_bytes()))
    snow = next(row for row in coverage["queries"] if row["ticker"] == "SNOW")
    assert snow["provider_rows"] == 0 and snow["first_observed_date"] is None
    assert snow["absent_dates_relative_to_observed_panel"] == ["2017-01-03", "2017-01-04"]
    assert coverage["exchange_calendar_qualified"] is False
    bars = [json.loads(line) for line in gzip.decompress((output / "observed-daily-bars.jsonl.gz").read_bytes()).splitlines()]
    assert len(bars) == 398 and all(row["vw"] is None and row["n"] is None for row in bars)
    assert all(row["historical_available_at_ns"] is None for row in bars)
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
    with pytest.raises(FileExistsError):
        daily.normalize_daily(root=root, plan_sha256=plan_sha,
                              completion_sha256=complete_sha, output=output)


@pytest.mark.parametrize("relative", ["day-AAPL/page-0000.json.gz", "day-WMT/COMPLETE.json"])
def test_persisted_tampering_prevents_normalization(observed_capture, tmp_path, relative):
    original, plan_sha, complete_sha = observed_capture
    root = tmp_path / "capture"
    shutil.copytree(original, root)
    path = root / relative
    path.chmod(0o600)
    path.write_bytes(b"altered")
    with pytest.raises(transport.ResearchCaptureError):
        daily.normalize_daily(root=root, plan_sha256=plan_sha,
                              completion_sha256=complete_sha, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_provider_failure_preserved_without_retry(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    plan_sha = _plan(root)
    calls = _network(monkeypatch, failure=True)
    with pytest.raises(transport.ResearchCaptureError):
        daily.capture_daily(root=root, api_key="synthetic-test-only", plan_sha256=plan_sha)
    assert len(calls) == 1 and (root / "BLOCKED.json").exists()
    assert (root / "day-AAPL/page-0000.json.gz").exists() and not (root / "COMPLETE.json").exists()


def test_invalid_observed_row_blocks_schema_acceptance(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    plan_sha = _plan(root)
    _network(monkeypatch, malformed=True)
    daily.capture_daily(root=root, api_key="synthetic-test-only", plan_sha256=plan_sha)
    result = daily.normalize_daily(root=root, plan_sha256=plan_sha,
        completion_sha256=transport.digest((root / "COMPLETE.json").read_bytes()),
        output=tmp_path / "output")
    assert result["invalid_rows"] == 1 and result["observed_rows"] == 397
    assert result["observed_bar_schema_accepted"] is result["training_ready"] is False
