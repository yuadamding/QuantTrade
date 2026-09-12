"""LSF-only minute transport/window regressions; no fabricated economic proof."""

import gzip
import json
from datetime import datetime
from pathlib import Path

import pytest

from rl_quant.data_sources.massive import qt200_minute_capture_v1 as minute
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport


def test_exact_panel_plus_source_observed_alias_intervals():
    queries = minute.minute_queries()
    assert len(queries) == 203 and tuple(q.ticker for q in queries[:200]) == minute.SYMBOLS
    assert all(q.validate_url(q.url) == q.url for q in queries)
    assert queries[-1].ticker == "HONI" and queries[-1].start == "2026-06-15"
    for ticker in ("GOOG", "BRK-B", "CASH", "../AAPL"):
        with pytest.raises(transport.ResearchCaptureError):
            minute.plan_fields(ticker)


@pytest.mark.parametrize("day", ["2017-01-03", "2026-03-06", "2026-03-09"])
@pytest.mark.parametrize("clock,inside", [("09:34", False), ("09:35", True), ("09:44", True), ("09:45", False)])
def test_execution_window_uses_new_york_clock_across_dst(day, clock, inside):
    stamp = int(datetime.fromisoformat(day + "T" + clock).replace(tzinfo=transport.ET).timestamp()) * 1000
    assert minute.in_execution_window(stamp) is inside


def test_paged_capture_preserves_all_rows_but_window_is_only_a_view(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    root.mkdir()
    query = minute.minute_queries()[0]
    plan = {**minute.plan_fields("AAPL"),
        "minute_implementation_sha256": transport.digest(Path(minute.__file__).read_bytes()),
        "capture_implementation_sha256": transport.digest(Path(transport.__file__).read_bytes())}
    raw = transport.canonical(plan)
    transport.write_once(root / "plan.json", raw)
    calls = []

    class Response:
        code = 200
        headers = {"Content-Type": "application/json"}

        def __init__(self, url):
            self.url = url

        def geturl(self):
            return self.url

        def read(self, cap):
            second = "cursor=" in self.url
            times = ["09:45"] if second else ["09:34", "09:35", "09:44"]
            rows = [{"t": int(datetime.fromisoformat("2017-01-03T" + t).replace(tzinfo=transport.ET).timestamp()) * 1000,
                     "o": 10, "h": 12, "l": 9, "c": 11, "v": 100} for t in times]
            body = dict(status="OK", request_id="synthetic", ticker="AAPL", adjusted=False,
                        results=rows, resultsCount=len(rows))
            if not second:
                body["next_url"] = query.url + "&cursor=next"
            return transport.canonical(body)[:cap]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Opener:
        def open(self, req, timeout):
            assert req.get_header("Authorization") == "Bearer synthetic-only" and timeout == 45
            calls.append(req.full_url)
            return Response(req.full_url)

    monkeypatch.setattr(transport.request, "build_opener", lambda *_: Opener())
    monkeypatch.setattr(transport.time, "sleep", lambda *_: None)
    plan_sha = transport.digest(raw)
    minute.capture_minute(root=root, ticker="AAPL", api_key="synthetic-only", plan_sha256=plan_sha)
    complete_sha = transport.digest((root / "COMPLETE.json").read_bytes())
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "window"
    result = minute.normalize_execution_inputs(root=root, ticker="AAPL", plan_sha256=plan_sha,
                                               completion_sha256=complete_sha, output=output)
    assert result["raw_rows"] == 4 and result["selected_window_rows"] == 2
    assert result["dates_with_window_gaps"] == 1 and result["vwap_observed_rows"] == 0
    assert result["modeled_fills_created"] is result["training_ready"] is False
    bars = [json.loads(row) for row in gzip.decompress((output / "execution-window-bars.jsonl.gz").read_bytes()).splitlines()]
    assert len(bars) == 2 and all(row["vw"] is None and row["n"] is None for row in bars)
    assert len(calls) == 2 and before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    coverage = json.loads(gzip.decompress((output / "coverage.json.gz").read_bytes()))
    assert coverage["missing_window_slots"] == {"2017-01-03": list(range(36, 44))}
