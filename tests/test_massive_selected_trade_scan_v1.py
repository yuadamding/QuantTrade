"""Source-only regression cases; these tests do not run a model or authorize data."""

from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.selected_trade_scan_v1 import (
    MassiveSelectedTradeScanError,
    scan_massive_selected_trade_file_v1,
)
from rl_quant.data_sources.massive.source_receipts import publish_massive_source_object
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_COLUMNS,
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    _parse_conditions as extract_conditions,
)
from rl_quant.data_sources.massive.finalized_daily_scan import _parse_conditions as scan_conditions
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)


HEADER = (",".join(MASSIVE_FLAT_TRADE_COLUMNS) + "\r\n").encode()
SOURCE_KEY = "us_stocks_sip/trades_v1/2017/01/2017-01-03.csv.gz"


def _line(ticker="AAA", *, correction="0", price="10.000", conditions="", end="\r\n"):
    # Exact text spelling and physical CRLF must survive via values plus line SHA.
    return (f'{ticker},"{conditions}",{correction},4,T1,1483455600000000000,'
            f'{price},1,1483455600000000001,100.00,1,,{end}').encode()


def _source(tmp_path: Path, content: bytes, *, compressed: bytes | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(content, mtime=0) if compressed is None else compressed
    publish_massive_source_object(
        stream=BytesIO(payload), root=tmp_path, relative_payload_path="trades/day.csv.gz",
        dataset_id="us_stocks_sip/trades_v1", source_object_key=SOURCE_KEY,
        requested_at_ms=1, downloaded_at_ms=2, committed_at_ms=3,
        schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
        entitlement_receipt_sha256="e" * 64,
    )
    return {
        "root": tmp_path,
        "payload_relative_path": "trades/day.csv.gz",
        "expected_source_object_key": SOURCE_KEY,
        "expected_receipt_file_sha256": file_sha256(tmp_path / "trades/day.csv.gz.receipt.json"),
        "expected_commit_file_sha256": file_sha256(tmp_path / "trades/day.csv.gz.commit.json"),
        "expected_entitlement_receipt_sha256": "e" * 64,
        "expected_compressed_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_compressed_bytes": len(payload),
        "tickers": ("AAA", "BRK.B"),
        "maximum_line_bytes": 1024,
        "verified_at_ms": 4,
    }


def test_original_crlf_fields_unknown_codes_and_exact_selection(tmp_path: Path) -> None:
    original = _line(correction="99", conditions="[999,7]")
    other = _line("BRK/B")
    valid = _line("BRK.B", correction="11")
    content = HEADER + original + other + valid
    rows = []
    evidence = scan_massive_selected_trade_file_v1(**_source(tmp_path, content), row_sink=rows.append)

    assert len(rows) == 2
    assert [row.source_row_number for row in rows] == [2, 4]
    assert rows[0].raw_line_sha256 == hashlib.sha256(original).hexdigest()
    assert rows[0].raw_line_sha256 != hashlib.sha256(original.replace(b"\r\n", b"\n")).hexdigest()
    assert rows[0].original_values[1:3] == ("[999,7]", "99")
    assert rows[0].original_values[6] == "10.000"
    assert rows[0].original_values[9] == "100.00"
    assert rows[0].original_values[-2:] == ("", "")
    canonical = rows[0].extracted_row.canonical_record
    assert canonical.conditions == (7, 999)
    assert canonical.correction_code == 99
    assert canonical.raw_source_record_sha256 == rows[0].raw_line_sha256
    assert rows[1].extracted_row.canonical_record.correction_code == 11
    assert evidence.source_row_count == 3
    assert evidence.selected_row_count == 2
    assert evidence.unselected_row_count == 1
    assert evidence.selected_canonicalization_error_count == 0
    assert evidence.selected_ticker_counts == (("AAA", 1), ("BRK.B", 1))
    assert evidence.decompressed_sha256 == hashlib.sha256(content).hexdigest()
    assert evidence.selected_row_inventory_sha256 == semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    assert evidence.complete_exact_ticker_selection
    assert not evidence.whole_source_canonical_scan_qualified
    assert not evidence.identity_qualified
    assert not evidence.training_ready
    evidence.validate()


def test_canonicalization_failure_is_retained_not_repaired(tmp_path: Path) -> None:
    raw = _line(correction="", price="", conditions="[1,1]")
    rows = []
    evidence = scan_massive_selected_trade_file_v1(**_source(tmp_path, HEADER + raw), row_sink=rows.append)
    assert len(rows) == 1
    assert rows[0].original_values[1:3] == ("[1,1]", "")
    assert rows[0].original_values[6] == ""
    assert rows[0].raw_line_sha256 == hashlib.sha256(raw).hexdigest()
    assert rows[0].extracted_row is None
    assert rows[0].canonicalization_error
    assert evidence.selected_canonical_row_count == 0
    assert evidence.selected_canonicalization_error_count == 1
    assert evidence.selected_ticker_canonicalization_error_counts == (("AAA", 1), ("BRK.B", 0))


@pytest.mark.parametrize("cell,expected", [
    ("", ()), ("  ", ()), ("0", (0,)), ("12", (12,)), (" 12 ", (12,)),
    ("12,37", (12, 37)), ("12, 37", (12, 37)), ("[]", ()),
    ("[12]", (12,)), ("[12,37]", (12, 37)), ('["12","37"]', (12, 37)),
])
def test_provider_condition_cell_formats_are_identical_in_both_readers(cell, expected):
    assert extract_conditions(cell) == scan_conditions(cell) == expected


@pytest.mark.parametrize("cell", [
    "true", "false", "null", "12.0", "1.2", "1e2", "-1", '"12"', "{}",
    "[true]", "[12.5]", "[null]", "[-1]", "12,,37", "12,", ",12",
    "[12", "12]", "[12,]", "[[12]]", '["1.0"]', "NaN", "[NaN]",
])
def test_invalid_condition_cells_are_never_dropped_or_coerced(cell):
    for parser in (extract_conditions, scan_conditions):
        with pytest.raises(ValueError, match="conditions are malformed"):
            parser(cell)


def test_actual_retained_singleton_condition_record_keeps_original_provenance(tmp_path: Path):
    # Original AAPL row 103939 of 2017-01-03. The old parser rejected "12".
    raw = (b"AAPL,12,0,12,3,1483434234152687361,116.02,1046,"
           b"1483434234152763543,100,3,0,0\n")
    assert hashlib.sha256(raw).hexdigest() == "a6030dddc4dff8bb19995263337707a8e4a513d577c8395b3fd40c787e70c3d9"
    args = dict(_source(tmp_path, HEADER + raw), tickers=("AAPL",))
    rows = []
    evidence = scan_massive_selected_trade_file_v1(**args, row_sink=rows.append)
    assert evidence.selected_canonicalization_error_count == 0
    assert rows[0].original_values[1] == "12"
    assert rows[0].raw_line_sha256 == hashlib.sha256(raw).hexdigest()
    trade = rows[0].extracted_row.canonical_record
    assert trade.conditions == (12,)
    assert trade.price_decimal == "116.02" and trade.size_decimal == "100"
    assert trade.participant_timestamp_ns == 1483434234152687361
    assert trade.sip_timestamp_ns == 1483434234152763543


def test_all_gzip_members_and_last_unterminated_physical_row_are_read(tmp_path: Path) -> None:
    first, last = _line(), _line("BBB", end="")
    compressed = gzip.compress(HEADER + first, mtime=0) + gzip.compress(last, mtime=0)
    rows = []
    evidence = scan_massive_selected_trade_file_v1(
        **_source(tmp_path, HEADER + first + last, compressed=compressed), row_sink=rows.append,
    )
    assert evidence.source_row_count == 2
    assert evidence.compressed_bytes == len(compressed)
    assert evidence.compressed_sha256 == hashlib.sha256(compressed).hexdigest()
    assert evidence.decompressed_sha256 == hashlib.sha256(HEADER + first + last).hexdigest()


@pytest.mark.parametrize("damage", ["truncated", "crc", "trailing"])
def test_bad_gzip_never_returns_completion(tmp_path: Path, damage: str) -> None:
    content = HEADER + _line()
    compressed = gzip.compress(content, mtime=0)
    if damage == "truncated":
        compressed = compressed[:-4]
    elif damage == "crc":
        compressed = compressed[:-8] + bytes([compressed[-8] ^ 1]) + compressed[-7:]
    else:
        compressed += b"not-a-gzip-member"
    # Even a native transaction faithfully binding damaged gzip cannot pass CRC.
    args = _source(tmp_path, content, compressed=compressed)
    with pytest.raises((EOFError, OSError, MassiveSelectedTradeScanError)):
        scan_massive_selected_trade_file_v1(**args, row_sink=lambda _: None)


def test_wrong_expected_receipt_fails_before_callback(tmp_path: Path) -> None:
    args = _source(tmp_path, HEADER + _line())
    args["expected_receipt_file_sha256"] = "f" * 64
    rows = []
    with pytest.raises(MassiveSelectedTradeScanError, match="metadata"):
        scan_massive_selected_trade_file_v1(**args, row_sink=rows.append)
    assert not rows


def test_gzip_header_change_fails_full_hash_despite_valid_crc(tmp_path: Path) -> None:
    args = _source(tmp_path, HEADER + _line())
    path = tmp_path / "trades/day.csv.gz"
    payload = path.read_bytes()
    # Header mtime is not protected by the ordinary gzip payload CRC. Keep the
    # actual persisted receipt and commit unchanged; do not manufacture a match.
    changed = payload[:4] + bytes([payload[4] ^ 1]) + payload[5:]
    path.chmod(0o644)
    path.write_bytes(changed)
    provisional_rows = []
    with pytest.raises(MassiveSelectedTradeScanError, match="compressed source hash"):
        scan_massive_selected_trade_file_v1(**args, row_sink=provisional_rows.append)
    assert len(provisional_rows) == 1
    assert provisional_rows[0].original_values[0] == "AAA"


def test_oversized_physical_line_cannot_complete(tmp_path: Path) -> None:
    args = _source(tmp_path, HEADER + _line(price="1" * 2048))
    with pytest.raises(MassiveSelectedTradeScanError, match="line exceeds"):
        scan_massive_selected_trade_file_v1(**args, row_sink=lambda _: None)


def test_payload_symlink_is_rejected(tmp_path: Path) -> None:
    args = _source(tmp_path, HEADER + _line())
    path = tmp_path / "trades/day.csv.gz"
    target = path.with_name("preserved-original.csv.gz")
    path.rename(target)
    path.symlink_to(target.name)
    with pytest.raises(OSError):
        scan_massive_selected_trade_file_v1(**args, row_sink=lambda _: None)


def test_validly_resealed_commit_with_wrong_crosslink_is_rejected(tmp_path: Path) -> None:
    args = _source(tmp_path, HEADER + _line())
    path = tmp_path / "trades/day.csv.gz.commit.json"
    value = json.loads(path.read_bytes())
    value["payload_file_sha256"] = "a" * 64
    del value["receipt_sha256"]
    value["receipt_sha256"] = semantic_sha256(value)
    path.chmod(0o644)
    path.write_bytes(canonical_json_file_bytes(value))
    args["expected_commit_file_sha256"] = file_sha256(path)
    with pytest.raises(MassiveSelectedTradeScanError, match="transaction"):
        scan_massive_selected_trade_file_v1(**args, row_sink=lambda _: None)


def test_callback_failure_cannot_return_completion_or_write_source(tmp_path: Path) -> None:
    args = _source(tmp_path, HEADER + _line())
    paths = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    before = {path: file_sha256(path) for path in paths}

    def fail(_row):
        raise RuntimeError("caller staging failed")

    with pytest.raises(RuntimeError, match="caller staging failed"):
        scan_massive_selected_trade_file_v1(**args, row_sink=fail)
    assert before == {path: file_sha256(path) for path in paths}
    assert paths == sorted(path for path in tmp_path.rglob("*") if path.is_file())


def test_unselected_malformed_row_is_not_silently_skipped(tmp_path: Path) -> None:
    args = _source(tmp_path, HEADER + _line() + b"BBB,too,few,columns\n")
    with pytest.raises(MassiveSelectedTradeScanError, match="field count"):
        scan_massive_selected_trade_file_v1(**args, row_sink=lambda _: None)


def test_source_mutation_in_final_progress_callback_has_no_completion(tmp_path: Path) -> None:
    args = _source(tmp_path, HEADER + _line())
    receipt = tmp_path / "trades/day.csv.gz.receipt.json"

    def mutate(_source_count, _selected_count):
        receipt.chmod(0o644)
        receipt.write_bytes(receipt.read_bytes() + b" ")

    with pytest.raises(MassiveSelectedTradeScanError, match="metadata"):
        scan_massive_selected_trade_file_v1(**args, row_sink=lambda _: None, progress_callback=mutate)


def test_changed_selection_or_authorization_flags_do_not_validate(tmp_path: Path) -> None:
    evidence = scan_massive_selected_trade_file_v1(**_source(tmp_path, HEADER + _line()), row_sink=lambda _: None)
    changed = replace(evidence, ordered_tickers=tuple(reversed(evidence.ordered_tickers)))
    with pytest.raises(MassiveSelectedTradeScanError, match="selection"):
        changed.validate()
    promoted = replace(evidence, training_ready=True)
    promoted = replace(promoted, receipt_sha256=semantic_sha256(promoted.unsigned()))
    with pytest.raises(MassiveSelectedTradeScanError, match="authorize"):
        promoted.validate()
