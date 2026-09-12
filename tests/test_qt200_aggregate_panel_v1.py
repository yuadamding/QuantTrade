"""LSF-only regression of lossless observed-panel integration, not training."""

import gzip
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rl_quant.data_sources.massive import qt200_aggregate_panel_v1 as panel
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_pilot_v1 import normalize_bar
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import DailyQuery, END, START
from rl_quant.data_sources.massive.qt200_minute_capture_v1 import minute_queries


def _bar(clock="09:35", day="2017-01-03", volume="100.001"):
    query = minute_queries()[0]
    stamp = int(datetime.fromisoformat(day + "T" + clock).replace(tzinfo=transport.ET).timestamp()) * 1000
    row = normalize_bar(dict(t=stamp, o="10.123456789123456789", h="12", l="9", c="11", v=volume), query,
                        received_at_ns=1_790_000_000_000_000_000)
    row["source"] = dict(page=0, row=0, page_sha256="a" * 64, received_at_ns=row["observed_at_ns"])
    return row


def _resolution(assertions=()):
    return dict(dated_assertions=list(assertions), current_reference_evidence=dict(gaps=[],
                active_exact_reference_count=1, ticker_event_observations=[]))


def test_exact_decimal_and_optional_nulls_survive_parquet(tmp_path):
    original = _bar()
    row = panel._observed_bar(original, minute_queries()[0])
    path = tmp_path / "bars.parquet"
    pq.write_table(pa.Table.from_pylist([row], schema=panel._bar_schema()), path, compression="zstd")
    assert pq.read_table(path).to_pylist() == [row]
    assert row["o"] == "10.123456789123456789"
    assert row["vw"] is row["n"] is row["historical_available_at_ns"] is None
    assert row["raw_row_sha256"] == original["raw_row_sha256"]


@pytest.mark.parametrize("key,value", [
    ("adjusted", True), ("historical_available_at_ns", 1), ("observed_at_ns", 2),
    ("vwap_observed", True), ("transaction_count_observed", True), ("vwap_observed", 0),
    ("session_date_et", "2017-01-04"), ("ticker", "MSFT"), ("raw_row_sha256", "fake"),
    ("o", "0"), ("v", "-1"), ("c", "NaN"),
])
def test_no_adjustment_availability_optional_or_source_fabrication(key, value):
    row = _bar()
    row[key] = value
    with pytest.raises(transport.ResearchCaptureError):
        panel._observed_bar(row, minute_queries()[0])


def test_missing_window_is_null_volume_not_zero_liquidity():
    facts = panel._window_facts([], "2017-01-03")
    assert facts["observed_volume_sum"] is None
    assert facts["missing_slot_mask"] == 1023 and facts["missing_slots"] == list(range(35, 45))
    observed = panel._window_facts([_bar(volume="0")], "2017-01-03")
    assert observed["observed_volume_sum"] == "0" and observed["observed_slots"] == 1
    assert not observed["complete"]


@pytest.mark.parametrize("day", ["2017-01-03", "2026-03-06", "2026-03-09"])
def test_window_uses_all_ten_real_slots_across_dst(day):
    rows = [_bar(f"09:{minute}", day) for minute in range(35, 45)]
    facts = panel._window_facts(rows, day)
    assert facts["complete"] and facts["observed_volume_sum"] == "1000.010"
    assert facts["missing_slot_mask"] == 0 and facts["observed_vwap_slots"] == 0


@pytest.mark.parametrize("clocks", [("09:34",), ("09:45",), ("09:35", "09:35"), ("09:36", "09:35")])
def test_invalid_duplicate_and_out_of_order_slots_fail(clocks):
    with pytest.raises(transport.ResearchCaptureError):
        panel._window_facts([_bar(clock) for clock in clocks], "2017-01-03")


def test_sparse_identity_match_does_not_propagate_or_stitch_ticker():
    rows = [dict(query_date="2017-01-03", provider_row=dict(ticker="AAPL"),
                 classification="exact_date_issue_identifiers_match_only")]
    assert panel._identity_evidence(_resolution(), "AAPL", "2017-01-03", rows) == "exact_date_issue_identifier_match_only"
    assert panel._identity_evidence(_resolution(), "AAPL", "2017-01-04", rows) == "outside_or_missing_issue_event_bracket"
    assert panel._identity_evidence(_resolution(), "MSFT", "2017-01-03", rows) == "outside_or_missing_issue_event_bracket"


def test_conflict_wins_but_another_issuer_share_class_does_not_veto():
    rows = [dict(query_date="2017-01-03", provider_row=dict(ticker="GOOG"), classification="conflicting_issue_identifiers")]
    assert panel._identity_evidence(_resolution(), "GOOGL", "2017-01-03", rows) != "dated_issue_conflict"
    rows[0]["provider_row"]["ticker"] = "GOOGL"
    assert panel._identity_evidence(_resolution(), "GOOGL", "2017-01-03", rows) == "dated_issue_conflict"


def test_wrong_historical_issue_survives_as_conflict_not_qualified():
    assertions = [dict(query_ticker="SNOW", query_date="2017-01-03",
                       classification="conflicts_with_current_observed_issue_identifiers")]
    extra = [dict(query_date="2017-01-03", provider_row=dict(ticker="SNOW"),
                  classification="exact_date_issue_identifiers_match_only")]
    assert panel._identity_evidence(_resolution(assertions), "SNOW", "2017-01-03", extra) == "dated_issue_conflict"


def test_parent_receipt_cannot_be_replaced_or_promoted(tmp_path):
    raw = transport.canonical(dict(schema="example", training_ready=False))
    transport.write_once(tmp_path / "COMPLETE.json", raw)
    digest = transport.digest(raw)
    assert panel._complete(tmp_path, digest, "example")[1] == raw
    for wrong_hash, wrong_schema in (("f" * 64, "example"), (digest, "wrong")):
        with pytest.raises(transport.ResearchCaptureError):
            panel._complete(tmp_path, wrong_hash, wrong_schema)
    (tmp_path / "COMPLETE.json").write_bytes(transport.canonical(dict(schema="example", training_ready=True)))
    changed = (tmp_path / "COMPLETE.json").read_bytes()
    with pytest.raises(transport.ResearchCaptureError):
        panel._complete(tmp_path, transport.digest(changed), "example")


def test_source_position_is_not_discarded_or_fabricated():
    row = _bar()
    row["source"]["row"] = -1
    with pytest.raises(transport.ResearchCaptureError):
        panel._observed_bar(row, minute_queries()[0])


def test_daily_and_execution_inputs_are_distinct():
    with pytest.raises(transport.ResearchCaptureError):
        panel._observed_bar(_bar(), DailyQuery("day", "AAPL", START, END))
    fields = panel._panel_schema().names
    assert "daily_row" in fields and "feature_daily_row" in fields and "execution_session" in fields
    assert "return" not in fields and "fill" not in fields


def test_normalized_jsonl_preserves_duplicate_visible_rows():
    row = {"t": 1}
    packed = gzip.compress(transport.canonical(row) * 2)
    assert list(panel._rows(packed)) == [row, row]
