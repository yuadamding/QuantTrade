"""Remote-only source interpretation regressions; no training promotion."""

from dataclasses import replace
import csv
import hashlib
import io

import pytest

from rl_quant.data_sources.massive.conditions import (
    MASSIVE_STOCK_TRADE_CONDITION_QUERY, build_massive_condition_authority,
)
from rl_quant.data_sources.massive.qt200_historical_message_adapter_v1 import (
    Qt200HistoricalMessageError,
    historical_message_category_v1,
    interpret_selected_historical_message_v1,
)
from rl_quant.data_sources.massive.selected_trade_scan_v1 import (
    scan_massive_selected_trade_file_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from test_massive_selected_trade_scan_v1 import HEADER, _source


RAW_AAPL_CLOSE = (
    b'AAPL,"38,41",0,12,34110,1483478100016366686,116.15,2303896,'
    b'1483478100016390377,0,3,0,0\n'
)


def _line(**changes):
    values = dict(ticker="AAPL", conditions="38,41", correction="0", exchange="12",
        id="34110", participant_timestamp="1483478100016366686", price="116.15",
        sequence_number="2303896", sip_timestamp="1483478100016390377", size="0",
        tape="3", trf_id="0", trf_timestamp="0")
    values.update(changes)
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerow(values.values())
    return stream.getvalue().encode()


def _read(tmp_path, lines):
    request = _source(tmp_path, HEADER + b"".join(lines))
    request["tickers"] = ("AAPL",)
    rows = []
    scan = scan_massive_selected_trade_file_v1(**request, row_sink=rows.append)
    identity = dict(source_object_key=scan.loaded_source.receipt.source_object_key,
        compressed_sha256=scan.compressed_sha256,
        source_receipt_sha256=scan.loaded_source.receipt.receipt_sha256,
        receipt_file_sha256=scan.receipt_file_sha256,
        commit_file_sha256=scan.commit_file_sha256)
    return rows, scan, identity


def _conditions(*, volume38=False, omit38=False):
    return build_massive_condition_authority([
        dict(id=code, name=name, asset_class="stocks", data_types=["trade"],
            update_rules=dict(consolidated=dict(updates_high_low=True,
                updates_open_close=True, updates_volume=volume)))
        for code, name, volume in (
            (38, "Corrected Consolidated Close (per listing market)", volume38),
            (41, "Trade Thru Exempt", True)) if code != 38 or not omit38
    ], source_object_receipt_sha256="c" * 64,
        source_query_path=MASSIVE_STOCK_TRADE_CONDITION_QUERY)


def test_exact_preserved_aapl_zero_volume_close_is_not_an_executable_trade(tmp_path):
    assert hashlib.sha256(RAW_AAPL_CLOSE).hexdigest() == (
        "9fc27d3a292101f11995af052cabde52b858ff3b06c13c88193459512bd5cf8d")
    rows, scan, identity = _read(tmp_path, [RAW_AAPL_CLOSE])
    row = rows[0]
    result = interpret_selected_historical_message_v1(
        row, source_identity=identity, condition_authority=_conditions())
    assert result["message_category"] == "zero-volume-corrected-close-candidate"
    assert result["original_values"] == row.original_values
    assert result["provider_trade_id"] == "34110"
    assert result["candidate_executable_share_volume"] == "0"
    assert result["candidate_consolidated_price_rules"] == dict(
        updates_high_low=True, updates_open_close=True, updates_volume=False)
    assert row.extracted_row is None and scan.selected_canonicalization_error_count == 1
    assert "size must be finite and positive" in result["canonicalization_error"]
    for claim in ("fill_eligible", "volume_update_authorized", "bar_price_update_authorized",
            "training_ready", "historical_condition_mapping_qualified",
            "correction_replay_qualified", "strategy_availability_qualified"):
        assert result[claim] is False
    assert result["receipt_sha256"] == semantic_sha256({
        key: value for key, value in result.items() if key != "receipt_sha256"})


def test_after_close_message_retains_sip_time_without_backdating_or_availability_claim(tmp_path):
    rows, _, identity = _read(tmp_path, [RAW_AAPL_CLOSE])
    result = interpret_selected_historical_message_v1(rows[0], source_identity=identity)
    regular_close_ns = 1483477200000000000
    assert result["participant_timestamp_ns"] == 1483478100016366686
    assert result["sip_timestamp_ns"] == 1483478100016390377
    assert result["vendor_clock_lower_bound_ns"] > regular_close_ns
    assert result["vendor_clock_lower_bound_ns"] == result["sip_timestamp_ns"]
    assert result["strategy_available_at_ns"] is None
    assert result["strategy_availability_qualified"] is False


def test_cross_clock_order_is_preserved_and_cannot_reduce_causal_lower_bound(tmp_path):
    rows, _, identity = _read(tmp_path, [_line(participant_timestamp="1483478100016390999")])
    result = interpret_selected_historical_message_v1(rows[0], source_identity=identity)
    assert result["participant_timestamp_ns"] > result["sip_timestamp_ns"]
    assert result["vendor_clock_lower_bound_ns"] == 1483478100016390999


def test_blank_trade_ids_receive_distinct_source_record_ids_not_invented_trade_ids(tmp_path):
    raw = _line(id="", size="100", conditions="", tape="1")
    rows, _, identity = _read(tmp_path, [raw, raw])
    left, right = (interpret_selected_historical_message_v1(row, source_identity=identity)
                   for row in rows)
    assert left["message_category"] == "positive-volume-provider-id-unresolved"
    assert left["provider_trade_id"] is right["provider_trade_id"] is None
    assert left["provider_trade_id_raw"] == right["provider_trade_id_raw"] == ""
    assert left["raw_line_sha256"] == right["raw_line_sha256"]
    assert left["record_id"] != right["record_id"]
    assert left["correction_target_reference"] is right["correction_target_reference"] is None
    assert left["provider_sequence_number"] == right["provider_sequence_number"] == 2303896


def test_source_record_identity_binds_transaction_and_keeps_whitespace_id_raw(tmp_path):
    rows, _, identity = _read(tmp_path, [_line(id=" ", size="100", conditions="")])
    left = interpret_selected_historical_message_v1(rows[0], source_identity=identity)
    right = interpret_selected_historical_message_v1(rows[0],
        source_identity=dict(identity, commit_file_sha256="f" * 64))
    assert left["record_id"] != right["record_id"]
    assert left["provider_trade_id"] is None and left["provider_trade_id_raw"] == " "


@pytest.mark.parametrize("code,category", [
    ("1", "late-corrected-snapshot-candidate"),
    ("7", "error-original-data-candidate"),
    ("8", "cancellation-original-data-candidate"),
    ("10", "cancellation-message-candidate"),
    ("11", "error-message-candidate"),
    ("12", "correction-original-data-candidate"),
    ("99", "historical-correction-lifecycle-unresolved"),
    ("", "historical-correction-lifecycle-unresolved"),
])
def test_correction_codes_never_infer_target_or_replacement_orientation(tmp_path, code, category):
    rows, _, identity = _read(tmp_path, [_line(correction=code, conditions="", size="100")])
    result = interpret_selected_historical_message_v1(rows[0], source_identity=identity)
    assert result["message_category"] == category
    assert result["correction_target_reference"] is None
    assert result["correction_resolution_status"] == "historical-contract-unresolved"
    assert result["correction_replay_qualified"] is False
    assert result["historical_correction_mapping_qualified"] is False


@pytest.mark.parametrize("changes,category", [
    ({"size": "-1"}, "negative-quantity-unresolved"),
    ({"size": "NaN"}, "unsupported-quantity"),
    ({"size": "bad"}, "unsupported-quantity"),
    ({"price": "0"}, "unsupported-price"),
    ({"price": "Infinity"}, "unsupported-price"),
    ({"conditions": "true"}, "unsupported-condition-encoding"),
    ({"conditions": "38,38,41"}, "unsupported-condition-encoding"),
    ({"conditions": "41"}, "zero-quantity-message-unresolved"),
    ({"conditions": "38,41,52"}, "zero-quantity-message-unresolved"),
    ({"tape": "1"}, "zero-quantity-message-unresolved"),
    ({"exchange": "4"}, "zero-quantity-message-unresolved"),
    ({"id": ""}, "zero-quantity-message-unresolved"),
    ({"trf_id": "12"}, "zero-quantity-message-unresolved"),
    ({"correction": "12"}, "zero-quantity-message-unresolved"),
    ({"sip_timestamp": "0"}, "zero-quantity-message-unresolved"),
    ({"conditions": "41,38"}, "zero-volume-corrected-close-candidate"),
    ({"size": "100", "conditions": ""}, "positive-volume-trade-candidate"),
])
def test_message_kinds_do_not_blanket_accept_nonpositive_sizes(tmp_path, changes, category):
    rows, _, identity = _read(tmp_path, [_line(**changes)])
    result = interpret_selected_historical_message_v1(rows[0], source_identity=identity)
    assert historical_message_category_v1(rows[0].original_values) == category
    assert result["message_category"] == category
    assert result["original_values"] == rows[0].original_values
    assert result["fill_eligible"] is False


@pytest.mark.parametrize("authority,category", [
    (None, "zero-volume-corrected-close-candidate"),
    (_conditions(omit38=True), "zero-volume-condition-reference-unresolved"),
    (_conditions(volume38=True), "zero-volume-condition-reference-conflict"),
])
def test_missing_or_conflicting_condition_reference_never_qualifies_history(tmp_path, authority, category):
    rows, _, identity = _read(tmp_path, [RAW_AAPL_CLOSE])
    result = interpret_selected_historical_message_v1(rows[0], source_identity=identity,
        condition_authority=authority)
    assert result["message_category"] == category
    assert result["historical_condition_mapping_qualified"] is False
    assert result["bar_price_update_authorized"] is False


def test_corrupt_selected_row_or_source_identity_is_rejected(tmp_path):
    rows, _, identity = _read(tmp_path, [RAW_AAPL_CLOSE])
    with pytest.raises(ValueError, match="receipt differs"):
        interpret_selected_historical_message_v1(
            replace(rows[0], raw_line_sha256="f" * 64), source_identity=identity)
    with pytest.raises(Qt200HistoricalMessageError, match="exact source"):
        interpret_selected_historical_message_v1(rows[0], source_identity={})
    with pytest.raises(Qt200HistoricalMessageError, match="canonical"):
        interpret_selected_historical_message_v1(rows[0],
            source_identity=dict(identity, source_object_key="../outside.csv.gz"))
    with pytest.raises(Qt200HistoricalMessageError, match="SHA-256"):
        interpret_selected_historical_message_v1(rows[0],
            source_identity=dict(identity, compressed_sha256="not-a-hash"))


def test_bad_clock_encoding_cannot_become_a_strategy_availability_time(tmp_path):
    rows, _, identity = _read(tmp_path, [_line(sip_timestamp=str(2**64))])
    result = interpret_selected_historical_message_v1(rows[0], source_identity=identity)
    assert result["message_category"] == "zero-quantity-message-unresolved"
    assert result["sip_timestamp_ns"] is None
    assert result["vendor_clock_lower_bound_ns"] is None
    assert result["strategy_available_at_ns"] is None


def test_interpretation_has_no_public_economic_qualification_override():
    import inspect
    assert tuple(inspect.signature(interpret_selected_historical_message_v1).parameters) == (
        "row", "source_identity", "condition_authority")
