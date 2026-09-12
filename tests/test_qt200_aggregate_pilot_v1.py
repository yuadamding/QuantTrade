"""Remote-only aggregate ingestion regressions, not investment evidence."""

import gzip
import json

import pytest

from rl_quant.data_sources.massive import qt200_aggregate_pilot_v1 as pilot
from rl_quant.data_sources.massive import qt200_research_capture_v1 as capture
from test_qt200_research_capture_v1 import _Response, _plan


def _bar(**changes):
    return {"t": 1483419600000, "o": 10, "h": 12, "l": 9, "c": 11, "v": 100, **changes}


def test_bar_optional_fields_are_not_zero_filled():
    value = pilot.normalize_bar(_bar(), capture.pilot_queries()[0], received_at_ns=1)
    assert value["vw"] is value["n"] is value["historical_available_at_ns"] is None
    assert not value["vwap_observed"] and not value["transaction_count_observed"]
    assert value["adjusted"] is False and value["v"] == "100"
    assert value["session_date_et"] == "2017-01-03"


@pytest.mark.parametrize("changes", [
    {"o": True}, {"h": 8}, {"l": 13}, {"c": 0}, {"v": -1},
    {"vw": None}, {"vw": float("inf")}, {"n": 1.5}, {"n": True},
    {"n": -1}, {"t": True}, {"t": 1483419600000.0},
    {"t": 1483419600001}, {"t": 0}, {"otc": True}, {"otc": 0},
])
def test_bad_bars_rejected(changes):
    with pytest.raises(capture.ResearchCaptureError):
        pilot.normalize_bar(_bar(**changes), capture.pilot_queries()[0], received_at_ns=1)


def test_fractional_volume_and_optional_vwap_preserved():
    value = pilot.normalize_bar(_bar(v=0.125, vw=13.01, n=0), capture.pilot_queries()[0], received_at_ns=1)
    assert value["v"] == "0.125" and value["vw"] == "13.01" and value["n"] == 0
    assert value["vwap_observed"] and value["transaction_count_observed"]


def test_dst_and_minute_alignment():
    # Local midnight is UTC04 after the March DST boundary, UTC05 before it.
    query = capture.pilot_queries()[0]
    assert pilot.normalize_bar(_bar(t=1489377600000), query, received_at_ns=1)["session_date_et"] == "2017-03-13"
    with pytest.raises(capture.ResearchCaptureError):
        pilot.normalize_bar(_bar(t=1489381200000), query, received_at_ns=1)
    minute = capture.PilotQuery("minute", "AAPL", "2017-01-03", "2017-01-03")
    assert pilot.normalize_bar(_bar(t=1483454100000), minute, received_at_ns=1)["bar_start_ms"] == 1483454100000
    with pytest.raises(capture.ResearchCaptureError):
        pilot.normalize_bar(_bar(t=1483454100001), minute, received_at_ns=1)


def test_sequence_match_does_not_manufacture_id_or_link():
    original = {"ticker": "WMT", "id": "", "sequence_number": "7", "correction": "12", "size": "57200"}
    row = {"id": "", "sequence_number": 7, "size": 52700}
    result = pilot.compare_original(original, [(row, {"page": 0, "row": 0}), (row, {"page": 0, "row": 1})])
    assert result["candidate_count"] == 2 and result["correction_target_reference"] is None
    for candidate in result["candidates"]:
        assert candidate["rest_row"]["id"] == ""
        assert candidate["absent_rest_fields"] == ["correction"]
        assert candidate["different_fields"] == ["size"]
        assert candidate["correction_link_established"] is False


@pytest.fixture
def captured(tmp_path, monkeypatch, request):
    root, plan_sha = _plan(tmp_path)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["original_trade_object_sha256"] = "a" * 64
    plan_path.chmod(0o600)
    plan_path.write_bytes(capture.canonical(plan))
    plan_sha = capture.digest(plan_path.read_bytes())
    shape = getattr(request, "param", "ordinary")
    calls = []

    class Opener:
        def open(self, req, timeout):
            calls.append(req.full_url)
            query = next(q for q in capture.pilot_queries() if q.url == req.full_url)
            if query.product == "trades":
                rows = [{"id": "", "sequence_number": 7, "size": 0, "correction": 12}]
            else:
                rows = [_bar(t=1483454100000 if query.product == "minute" else 1483419600000)]
                if shape == "duplicate":
                    rows *= 2
                elif shape == "invalid":
                    rows[0]["l"] = 100
                elif shape == "empty":
                    rows = []
            body = {"status": "OK", "request_id": "synthetic-only", "results": rows}
            if query.product != "trades":
                body.update(ticker=query.ticker, adjusted=False, resultsCount=len(rows))
            return _Response(req.full_url, capture.canonical(body))

    # Only HTTP is synthetic; persistence, replay, analysis and all guards run.
    monkeypatch.setattr(capture.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(capture.time, "sleep", lambda *_: None)
    capture.capture_pilot(root=root, api_key="synthetic-transport-only", plan_sha256=plan_sha)
    csv = tmp_path / "samples.csv"
    csv.write_text("ticker,id,sequence_number,correction,size\nWMT,,7,12,0\n")
    pairs = tmp_path / "pairs.json"
    pairs.write_bytes(capture.canonical({"original_source_compressed_sha256": "a" * 64, "groups": []}))
    return dict(capture_root=root, plan_sha256=plan_sha,
                completion_sha256=capture.digest((root / "COMPLETE.json").read_bytes()),
                samples_csv=csv, samples_sha256=capture.digest(csv.read_bytes()),
                pairs_json=pairs, pairs_sha256=capture.digest(pairs.read_bytes()),
                output_root=tmp_path / "analysis"), calls


def test_persisted_capture_to_compact_bars_is_nonpromoting(captured):
    arguments, calls = captured
    root = arguments["capture_root"]
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = pilot.analyze_pilot(**arguments)
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after and len(calls) == 9
    assert result["observed_bar_schema_accepted"] and result["bar_rows"] == 6
    assert result["correction_links_established"] == 0
    assert result["training_ready"] is result["point_in_time_qualified"] is False
    records = [json.loads(line) for line in gzip.decompress(
        (arguments["output_root"] / result["bars"]["path"]).read_bytes()).splitlines()]
    assert len(records) == 6 and all(r["historical_available_at_ns"] is None for r in records)
    with pytest.raises(FileExistsError):
        pilot.analyze_pilot(**arguments)
    assert before == after


@pytest.mark.parametrize("target", ["page", "sample", "pair"])
def test_tamper_fails_before_output_publication(captured, target):
    arguments, _ = captured
    path = {"page": arguments["capture_root"] / "day-ABBV/page-0000.json.gz",
            "sample": arguments["samples_csv"], "pair": arguments["pairs_json"]}[target]
    path.chmod(0o600)
    path.write_bytes(b"tampered")
    with pytest.raises(capture.ResearchCaptureError):
        pilot.analyze_pilot(**arguments)
    assert not arguments["output_root"].exists()


@pytest.mark.parametrize("captured", ["duplicate", "invalid", "empty"], indirect=True)
def test_invalid_or_absent_bars_are_reported_without_promotion(captured):
    arguments, _ = captured
    result = pilot.analyze_pilot(**arguments)
    assert result["observed_bar_schema_accepted"] is False
    assert result["training_ready"] is False
    assert (arguments["output_root"] / "COMPLETE.json").is_file()
