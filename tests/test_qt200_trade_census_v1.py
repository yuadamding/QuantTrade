"""Run only on the approved remote regression worker; no economic promotion."""

from dataclasses import replace
import base64
import csv
import gzip
import hashlib
import io
import json
import resource

import pytest

from rl_quant.data_sources.massive import qt200_trade_census_v1 as census
from rl_quant.data_sources.massive.corrections import build_massive_correction_authority
from rl_quant.data_sources.massive.selected_trade_scan_v1 import scan_massive_selected_trade_file_v1
from rl_quant.protocol.canonical_artifact import file_sha256
from test_massive_selected_trade_scan_v1 import HEADER, _source


def _line(*, ticker="AAA", trade_id="", size="100", price="10.00", sequence="1",
          correction="0", conditions="", trf="", end="\r\n"):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator=end).writerow([
        ticker, conditions, correction, "4", trade_id, "1483455600000000000",
        price, sequence, "1483455600000000001", size, "1", trf, "",
    ])
    return stream.getvalue().encode()


def _arguments(tmp_path, content, *, compressed=None):
    return dict(source_request=_source(tmp_path / "raw", content, compressed=compressed),
        source_date="2017-01-03", qt200_tickers=("AAA",), alias_tickers=("BRK.B",),
        correction_authority=build_massive_correction_authority(
            ((0, "new-trade"), (10, "cancellation"), (12, "replacement")),
            canary_receipt_sha256="c" * 64),
        output_directory=tmp_path / "census", limits=census.CensusLimits())


def _run(args):
    result = census.scan_and_publish_qt200_trade_census_v1(**args)
    report = json.loads((args["output_directory"] / "census.json").read_bytes())
    assert file_sha256(args["output_directory"] / "census.json") == result["files"][0]["sha256"]
    return result, report


def test_blank_identity_and_zero_size_are_independent_not_first_error_counts(tmp_path):
    args = _arguments(tmp_path, HEADER + _line(size="0", conditions="38,41") + _line(sequence="2"))
    before = {p: file_sha256(p) for p in (tmp_path / "raw").rglob("*") if p.is_file()}
    result, report = _run(args)
    assert report["marginal_counts"]["provider_id_status"] == {"blank": 2}
    assert report["marginal_counts"]["size_status"] == {"positive": 1, "zero": 1}
    zero = next(row for row in report["intersections"] if row["values"][2] == "zero")
    assert zero["values"][1] == "blank" and zero["row_count"] == 1
    assert report["samples"][0]["provider_trade_id"] is None
    assert report["samples"][0]["provider_trade_id_raw"] == ""
    assert report["source_scan"]["selected_canonicalization_error_count"] == 2
    assert result["diagnostic_scan_complete"] is True
    assert {p: file_sha256(p) for p in before} == before


@pytest.mark.parametrize("raw,status", [
    ("100.00", "positive"), ("0", "zero"), ("-0.0", "zero"), ("-1", "negative"),
    ("", "absent"), ("bad", "malformed"), ("NaN", "nonfinite"), ("Infinity", "nonfinite"),
])
def test_quantity_classes_preserve_original_cells_without_fill_eligibility(tmp_path, raw, status):
    args = _arguments(tmp_path, HEADER + _line(size=raw, price=raw))
    result, report = _run(args)
    assert report["marginal_counts"]["size_status"] == {status: 1}
    assert report["marginal_counts"]["price_status"] == {status: 1}
    assert report["samples"][0]["original_values"][9] == raw
    assert result["bars_tape_fills_qualified"] is False


def test_identical_rows_remain_distinct_source_records_and_exact_raw_bytes(tmp_path):
    line = _line()
    args = _arguments(tmp_path, HEADER + line * 2 + _line(size="101"))
    _, report = _run(args)
    first, second = report["samples"]
    assert first["record_id"] != second["record_id"]
    assert first["raw_line_sha256"] == second["raw_line_sha256"] == hashlib.sha256(line).hexdigest()
    assert base64.b64decode(first["raw_line_base64"]) == line
    assert first["source_row_number"] == 2 and second["source_row_number"] == 3
    assert {row["status"]: row["row_count"] for row in report["sequence_counts"]} == {
        "first-observation": 1, "repeated-same-line-hash": 1, "repeated-different-line-hash": 1}
    assert report["selected_rows"] == 3 and report["sequence_deduplication_applied"] is False


def test_record_identity_binds_source_transaction_not_just_row_hash(tmp_path):
    _, left = _run(_arguments(tmp_path / "left", HEADER + _line()))
    _, right = _run(_arguments(tmp_path / "right", HEADER + _line() + _line(ticker="OTHER")))
    assert left["samples"][0]["raw_line_sha256"] == right["samples"][0]["raw_line_sha256"]
    assert left["samples"][0]["record_id"] != right["samples"][0]["record_id"]


def test_alias_counts_unknown_corrections_conditions_and_unresolved_dependencies(tmp_path):
    args = _arguments(tmp_path, HEADER + _line(correction="10") + _line(correction="12")
        + _line(ticker="BRK.B", correction="99", conditions="true", trf="0")
        + _line(ticker="OTHER"))
    result, report = _run(args)
    assert result["source_rows"] == 4 and result["selected_rows"] == 3
    assert report["marginal_counts"]["membership_scope"] == {"alias_candidate": 1, "qt200": 2}
    assert report["marginal_counts"]["native_candidate_kind"] == {
        "cancellation": 1, "replacement": 1, "unsupported": 1}
    assert report["marginal_counts"]["raw_trf_id"] == {"": 2, "0": 1}
    assert report["correction_resolution"] == dict(status="not-assessed", resolved_count=None, unresolved_count=None)
    assert len(report["unresolved_dependency_scopes"]) == 2
    assert report["correction_inventory_historical_applicability"] == "not-established"
    assert all(result[key] is value and report[key] is value for key, value in census.CLAIMS.items())
    assert {p.name for p in args["output_directory"].iterdir()} == {"census.json", "COMPLETE.json"}


def test_sequence_scope_is_ticker_and_day_without_correction_link_inference(tmp_path):
    args = _arguments(tmp_path, HEADER + _line() + _line(ticker="BRK.B") + _line(sequence="")
        + _line(sequence="-1") + _line(sequence=str(2**64)))
    _, report = _run(args)
    assert report["sequence_unique_keys"] == 2
    assert sum(row["row_count"] for row in report["sequence_counts"]) == 5
    assert {row["status"] for row in report["sequence_counts"]} == {
        "first-observation", "absent", "malformed", "outside-uint64"}
    assert all(sample["correction_target_reference"] is None for sample in report["samples"])


@pytest.mark.parametrize("limit", ["maximum_groups", "maximum_group_key_bytes", "maximum_sequence_keys",
    "maximum_selected_rows", "maximum_sample_bytes", "output_budget_bytes"])
def test_budget_exhaustion_never_publishes_completion(tmp_path, limit):
    args = _arguments(tmp_path, HEADER + _line() + _line(size="0", sequence="2"))
    args["limits"] = replace(args["limits"], **{limit: 1})
    with pytest.raises(ValueError, match="allocation"):
        _run(args)
    assert not (args["output_directory"] / "COMPLETE.json").exists()


@pytest.mark.parametrize("damage", ["crc", "truncated", "receipt", "source-mutation"])
def test_source_failures_do_not_promote_provisional_rows(tmp_path, damage):
    content = HEADER + _line()
    compressed = gzip.compress(content, mtime=0)
    if damage == "crc":
        compressed = compressed[:-8] + bytes([compressed[-8] ^ 1]) + compressed[-7:]
    if damage == "truncated":
        compressed = compressed[:-4]
    args = _arguments(tmp_path, content, compressed=compressed)
    if damage == "receipt":
        args["source_request"]["expected_receipt_file_sha256"] = "f" * 64
    if damage == "source-mutation":
        def change(_physical, _selected):
            path = args["source_request"]["root"] / "trades/day.csv.gz"
            path.chmod(0o644)
            path.write_bytes(compressed)
        args["progress_callback"] = change
    with pytest.raises((ValueError, OSError, EOFError)):
        _run(args)
    assert not (args["output_directory"] / "COMPLETE.json").exists()


def test_exact_membership_and_source_date_required_before_output(tmp_path):
    args = _arguments(tmp_path, HEADER + _line())
    args["source_date"] = "2017-01-04"
    with pytest.raises(ValueError, match="Exact date"):
        _run(args)
    assert not args["output_directory"].exists()
    args["source_date"] = "2017-01-03"
    args["alias_tickers"] = ("AAA",)
    with pytest.raises(ValueError, match="membership"):
        _run(args)


def test_original_line_callback_only_receives_exact_selected_source_bytes(tmp_path):
    raw = _line(size="0")
    args = _source(tmp_path, HEADER + raw + _line(ticker="OTHER"))
    observed = []
    scan_massive_selected_trade_file_v1(**args, row_sink=lambda _: None,
        original_line_sink=lambda row, line: observed.append((row, line)))
    assert len(observed) == 1 and observed[0][1] == raw
    assert observed[0][0].extracted_row is None


def test_all_tables_reconcile_and_existing_attempt_is_never_overwritten(tmp_path):
    args = _arguments(tmp_path, HEADER + _line() + _line(ticker="BRK.B", size="0"))
    _, report = _run(args)
    assert all(sum(counts.values()) == 2 for counts in report["marginal_counts"].values())
    assert sum(row["row_count"] for row in report["groups"]) == 2
    assert sum(row["row_count"] for row in report["intersections"]) == 2
    before = {p: p.read_bytes() for p in args["output_directory"].iterdir()}
    with pytest.raises(FileExistsError):
        _run(args)
    assert {p: p.read_bytes() for p in before} == before


def test_full_pilot_sequence_allocation_fits_application_memory_guard(record_property):
    # Actual worst-count index, not an extrapolation from a small sample. Runs
    # remotely on H100 host RAM; this is explicitly not GPU optimization proof.
    size = 8_000_000
    index = census._SequenceIndex(size)
    for value in range(size):
        index.observe(value % 206, value, value.to_bytes(32, "big").hex())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    record_property("sequence_keys", len(index.first))
    record_property("process_peak_rss_bytes", peak)
    print(json.dumps(dict(sequence_keys=len(index.first), process_peak_rss_bytes=peak)))
    assert len(index.first) == size and peak < 6 * 1024**3
    with pytest.raises(census.Qt200CensusError, match="Sequence-key"):
        index.observe(0, size, "0" * 64)
