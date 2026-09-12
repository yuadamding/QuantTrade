"""Source-processing parity/failure cases, not qualification or model tests."""

from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive.conditions import MASSIVE_STOCK_TRADE_CONDITION_QUERY, build_massive_condition_authority
from rl_quant.data_sources.massive.corrections import build_massive_correction_authority
from rl_quant.data_sources.massive.qt200_market_day_v1 import Qt200MarketDayError, Qt200MarketDaySpoolV1
from rl_quant.data_sources.massive.selected_trade_scan_v1 import scan_massive_selected_trade_file_v1
from rl_quant.data_sources.massive.session_calendar import MassiveExchangeSession, build_massive_session_authority
from rl_quant.features.massive_daily_bars_v0 import _row as native_bars
from rl_quant.features.massive_daily_tape_v0 import _row as native_tape
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from test_massive_selected_trade_scan_v1 import HEADER, _source


def _at(hour: int, minute: int) -> int:
    return int(datetime(2017, 1, 3, hour, minute, tzinfo=ZoneInfo("America/New_York")).timestamp()) * 1_000_000_000


def _trade(trade_id, minute, price, size, *, ticker="AAA", correction=0, conditions=(),
           hour=9, sip_hour=None, sip_minute=None, sequence=1, exchange=4, trf="", tape=1):
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    participant = _at(hour, minute)
    sip = participant if sip_hour is None else _at(sip_hour, sip_minute)
    writer.writerow([ticker, "[" + ",".join(str(x) for x in conditions) + "]", correction,
                     exchange, trade_id, participant, price, sequence, sip, size, tape, trf, ""])
    return output.getvalue().encode()


def _authorities():
    condition_rows = []
    for code, flags in ((1, (True, True, True)), (2, (False, False, True)),
                        (3, (False, True, False)), (4, (True, False, False))):
        condition_rows.append({"id": code, "name": "Synthetic condition " + str(code), "asset_class": "stocks",
                               "data_types": ["trade"], "update_rules": {"consolidated": {
                                   "updates_open_close": flags[0], "updates_high_low": flags[1], "updates_volume": flags[2]}}})
    conditions = build_massive_condition_authority(condition_rows, source_object_receipt_sha256="a" * 64,
                                                   source_query_path=MASSIVE_STOCK_TRADE_CONDITION_QUERY)
    # Entirely synthetic forward-event contract. Code 2 exercises replacement
    # mechanics, not any provider mapping. The legacy 12 rule remains present
    # specifically to prove that it cannot override the historical guard.
    corrections = build_massive_correction_authority(
        ((0, "new-trade"), (1, "new-trade"), (2, "replacement"), (7, "new-trade"),
         (11, "cancellation"), (12, "replacement"), (9, "late-report")),
        canary_receipt_sha256="b" * 64,
    )
    session = MassiveExchangeSession(session_date="2017-01-03", exchange="XNYS", regular_open_ns=_at(9, 30),
                                     regular_close_ns=_at(16, 0), scheduled_five_minute_intervals=78,
                                     special_session_reason=None, calendar_source_receipt_sha256="d" * 64)
    sessions = build_massive_session_authority((session,), calendar_source_receipt_sha256="d" * 64)
    return sessions, session, conditions, corrections


def _spool(tmp_path: Path, args, *, allocation_bytes=32 * 1024 * 1024):
    directory = tmp_path / "spool"
    directory.mkdir()
    sessions, session, conditions, corrections = _authorities()
    return Qt200MarketDaySpoolV1(database_path=directory / "day.sqlite3", ordered_tickers=args["tickers"],
                                session_authority=sessions, session=session, condition_authority=conditions,
                                correction_authority=corrections, allocation_bytes=allocation_bytes)


def _run(tmp_path: Path, trades):
    args = _source(tmp_path / "raw", HEADER + b"".join(trades))
    selected = []
    with _spool(tmp_path, args) as spool:
        def sink(row):
            selected.append(row)
            spool.append(row)
        scan = scan_massive_selected_trade_file_v1(**args, row_sink=sink)
        result = spool.finalize(scan)
    return result, selected, args


def _legacy_full_indexes(spool) -> None:
    # Recreate only the predecessor's index layout, before any rows are stored.
    assert spool.count == 0
    spool.connection.executescript("""
        DROP INDEX active_order;
        DROP INDEX size_order;
        DROP INDEX venue_order;
        CREATE INDEX active_order ON events(ticker,active,regular,participant,sip,sequence,ordinal);
        CREATE INDEX size_order ON events(ticker,active,regular,size_digits,size_whole,size_fraction,ordinal);
        CREATE INDEX venue_order ON events(ticker,active,regular,exchange_id,participant,sip,sequence,ordinal);
    """)
    spool.connection.commit()


@pytest.mark.parametrize("bad_ticker_day", [False, True])
def test_partial_indexes_preserve_complete_day_against_legacy_full_indexes(tmp_path: Path, bad_ticker_day: bool) -> None:
    trades = [
        _trade("CANCEL", 36, "10", "10.01", correction=7, sequence=1),
        _trade("REPLACE", 37, "20", "2.001", sequence=2),
        _trade("VOLUME", 38, "99", "0.12", conditions=(2,), exchange=9, sequence=3),
        _trade("HIGHLOW", 39, "30", "0.009", conditions=(3,), exchange=2, sequence=4),
        _trade("OPENCLOSE", 40, "15", "2.0001", conditions=(4,), exchange=1, sequence=5),
        _trade("PRE", 29, "9", "9.999", sequence=6),
        _trade("EARLY", 34, "12", "4", sequence=7),
        _trade("ENDPOINT", 45, "30", "20", sequence=8),
        _trade("AFTER", 1, "1000", "1000", hour=16, sequence=9),
        _trade("CANCEL", 36, "10", "10.01", correction=11, sip_hour=17, sip_minute=0, sequence=10),
        _trade("REPLACE", 37, "21", "3.5", correction=2, sip_hour=17, sip_minute=1, sequence=11),
        _trade("LATE", 41, "21", "7", correction=9, sip_hour=17, sip_minute=2, sequence=12),
        _trade("DOT", 36, "25", "100", ticker="BRK.B", trf=1, tape=2, sequence=13),
        _trade("UNSELECTED", 36, "1", "1", ticker="OTHER", sequence=14),
    ]
    if bad_ticker_day:
        trades.extend([
            _trade("UNKNOWN", 42, "10", "1", conditions=(999,), sequence=15),
            _trade("UNPARSED", 43, "", "1", sequence=16),
            _trade("ABSENT", 44, "10", "1", correction=11, sequence=17),
        ])
    args = _source(tmp_path / "raw", HEADER + b"".join(trades))
    optimized_root, legacy_root = tmp_path / "optimized", tmp_path / "legacy"
    optimized_root.mkdir()
    legacy_root.mkdir()
    with _spool(optimized_root, args) as optimized, _spool(legacy_root, args) as legacy:
        _legacy_full_indexes(legacy)
        orders = {
            "active_order": "participant,sip,sequence,ordinal",
            "size_order": "size_digits,size_whole,size_fraction,ordinal",
            "venue_order": "exchange_id,participant,sip,sequence,ordinal",
        }
        for spool, partial in ((optimized, 1), (legacy, 0)):
            metadata = {row[1]: row[4] for row in spool.connection.execute("PRAGMA index_list(events)")}
            assert all(metadata[name] == partial for name in orders)
            for name, order in orders.items():
                query = (f"SELECT ordinal FROM events INDEXED BY {name} "
                         f"WHERE ticker=? AND active=1 AND regular=1 ORDER BY {order}")
                plans = tuple(row[-1] for row in spool.connection.execute("EXPLAIN QUERY PLAN " + query, ("AAA",)))
                assert any(name in plan for plan in plans)
                assert all("TEMP B-TREE" not in plan.upper() and "AUTOMATIC" not in plan.upper() for plan in plans)

        def sink(row):
            optimized.append(row)
            legacy.append(row)

        scan = scan_massive_selected_trade_file_v1(**args, row_sink=sink)
        assert scan.source_row_count == len(trades)
        assert scan.selected_row_count == len(trades) - 1
        optimized_report, legacy_report = optimized.finalize(scan), legacy.finalize(scan)
    # File layout/path/hash and their enclosing receipt necessarily differ;
    # all source, correction, economic, mask, and semantic row hashes must not.
    omitted = {"spool", "receipt_sha256"}
    assert {k: v for k, v in optimized_report.items() if k not in omitted} == {
        k: v for k, v in legacy_report.items() if k not in omitted
    }
    assert optimized_report["rows"][0]["source_selected_market_day_valid"] is not bad_ticker_day
    assert optimized_report["rows"][1]["source_selected_market_day_valid"]
    assert file_sha256(args["root"] / args["payload_relative_path"]) == args["expected_compressed_sha256"]
    for report in (optimized_report, legacy_report):
        assert file_sha256(Path(report["spool"]["path"])) == report["spool"]["sha256"]
        assert not report["training_ready"] and not report["identity_qualified"]


def test_synthetic_terminal_replay_parity_is_not_decision_time_fill(tmp_path: Path) -> None:
    # Synthetic post-close corrections change terminal morning statistics.
    # These are explicitly not executable fills or decision-time observations.
    trades = [
        _trade("CANCEL", 36, "10", "100", correction=7, sequence=1),
        _trade("REPLACE", 37, "20", "100", sequence=2),
        _trade("VOL", 38, "99", "10", conditions=(2,), exchange=9, sequence=3),
        _trade("HL", 39, "30", "3", conditions=(3,), exchange=2, sequence=4),
        _trade("OC", 40, "15", "5", conditions=(4,), exchange=1, sequence=5),
        _trade("EARLY", 34, "9", "10", sequence=6),
        _trade("ENDPOINT", 45, "30", "20", sequence=7),
        _trade("AFTER", 1, "1000", "1000", hour=16, sequence=8),
        _trade("CANCEL", 36, "10", "100", correction=11, sip_hour=17, sip_minute=0, sequence=9),
        _trade("REPLACE", 37, "21", "120", correction=2, sip_hour=17, sip_minute=1, sequence=10),
    ]
    report, rows, args = _run(tmp_path, trades)
    observed = report["rows"][0]
    by_id = {}
    corrections = []
    authority = _authorities()[3]
    for row in sorted((item.extracted_row for item in rows), key=lambda r: (
            r.canonical_record.sip_timestamp_ns, r.canonical_record.sequence_number,
            r.canonical_record.exchange_id, -1 if r.canonical_record.trf_id is None else r.canonical_record.trf_id,
            r.canonical_record.trade_id, r.source_row_number)):
        record = row.canonical_record
        kind = authority.resolve(record.correction_code)
        key = (record.ticker, record.exchange_id, -1 if record.trf_id is None else record.trf_id, record.trade_id)
        if kind == "cancellation":
            del by_id[key]
        else:
            by_id[key] = row
        if kind in {"replacement", "cancellation", "late-report"}:
            corrections.append({"source_row_number": row.source_row_number, "correction_kind": kind,
                                "event_key": key, "canonical_record_receipt_sha256": record.receipt_sha256})
    active = tuple(row for row in by_id.values() if _at(9, 30) <= row.canonical_record.participant_timestamp_ns < _at(16, 0))
    condition_authority = _authorities()[2]
    expected_bars = native_bars("TEST-AAA", active, condition_authority)
    expected_tape = native_tape("TEST-AAA", active, tuple(corrections), condition_authority)
    assert observed["bars_values"] == list(expected_bars.values)
    assert observed["bars_valid"] == list(expected_bars.valid)
    assert observed["legacy_tape_values"] == list(expected_tape.values)
    assert observed["native_correction_inventory_sha256"] == semantic_sha256(tuple(corrections))
    assert observed["terminal_active_regular_native_inventory_sha256"] == expected_bars.source_active_inventory_sha256
    assert observed["fill"]["valid"]
    assert observed["fill"]["vwap"] == 21.0
    assert observed["fill"]["share_volume"] == 120.0
    assert observed["fill"]["trade_count"] == 1
    for view in (report, observed, observed["fill"]):
        assert view["view"] == "terminal_corrected_diagnostic"
        assert view["decision_time_qualified"] is False
    assert observed["fill"]["execution_eligible"] is False
    assert report["historical_correction_applicability_qualified"] is False
    assert observed["volume_forming_flow"]["share_volume"] == 160.0
    assert observed["price_volume_forming_flow"]["share_volume"] == 150.0
    assert observed["event_counts"]["cancellation"] == 1
    assert observed["observed_terminal_active_regular_rows"] == 6
    assert observed["security_id"] is None and not observed["identity_qualified"]
    assert not report["training_ready"] and not report["whole_source_canonical_scan_qualified"]
    assert not report["cold_replay_ready_without_original_gzip"]
    assert report["selection_scan"]["selected_row_count"] == 10
    assert file_sha256(Path(report["spool"]["path"])) == report["spool"]["sha256"]
    assert file_sha256(args["root"] / args["payload_relative_path"]) == args["expected_compressed_sha256"]
    missing = report["rows"][1]
    assert not any(missing["bars_valid"]) and not missing["fill"]["valid"]
    assert missing["bars_values"] == [0.0] * 8


@pytest.mark.parametrize("case", ["unknown-condition", "unknown-correction", "missing-cancel",
                                   "missing-replace", "conflicting-duplicate", "canonical-error"])
def test_bad_ticker_day_keeps_source_rows_but_cannot_make_usable_data(tmp_path: Path, case: str) -> None:
    if case == "unknown-condition":
        rows = [_trade("BAD", 36, "10", "1", conditions=(999,)),
                _trade("BAD", 36, "10", "1", correction=11, sip_hour=17, sip_minute=0, sequence=2)]
    elif case == "unknown-correction":
        rows = [_trade("BAD", 36, "10", "1", correction=99)]
    elif case == "missing-cancel":
        rows = [_trade("BAD", 36, "10", "1", correction=11)]
    elif case == "missing-replace":
        rows = [_trade("BAD", 36, "10", "1", correction=2)]
    elif case == "conflicting-duplicate":
        rows = [_trade("BAD", 36, "10", "1"), _trade("BAD", 36, "20", "1", sequence=2)]
    else:
        rows = [_trade("BAD", 36, "", "1")]
    rows += [_trade("GOOD", 40, "11", "10", sequence=3)]
    report, selected, _ = _run(tmp_path, rows)
    day = report["rows"][0]
    assert len(selected) == len(rows)
    assert report["selection_scan"]["selected_row_count"] == len(rows)
    assert day["errors"]
    assert not day["source_selected_market_day_valid"]
    assert day["bars_values"] == [0.0] * 8 and not any(day["bars_valid"])
    assert day["legacy_tape_values"] == [0.0] * 15 and not any(day["legacy_tape_valid"])
    assert day["fill"]["vwap"] == 0 and not day["fill"]["valid"]
    assert not day["volume_forming_flow"]["valid"]


@pytest.mark.parametrize("code", ["1", "01", "12", "012"])
@pytest.mark.parametrize("trade_id", ["MATCHING-ID", ""])
def test_generic_canary_cannot_authorize_historical_corrections(tmp_path: Path, code: str, trade_id: str) -> None:
    import sqlite3

    trades = [
        _trade("MATCHING-ID", 36, "100", "10", sequence=1),
        _trade(trade_id, 36, "101", "20", correction=code, sip_hour=17, sip_minute=0, sequence=2),
        _trade("OTHER-ISSUE", 37, "25", "5", ticker="BRK.B", sequence=3),
    ]
    report, selected, args = _run(tmp_path, trades)
    day = report["rows"][0]
    assert len(selected) == report["selection_scan"]["selected_row_count"] == 3
    assert selected[1].original_values[2] == code
    issue = "historical_correction_applicability_unresolved"
    if not trade_id:
        issue += "_and_canonicalization_error"
    assert day["errors"][issue] == 1
    assert not day["source_selected_market_day_valid"]
    assert not day["terminal_active_state_valid"]
    assert not any(day["bars_valid"]) and not any(day["legacy_tape_valid"])
    assert day["bars_values"] == [0.0] * 8
    assert not day["fill"]["valid"] and not day["fill"]["execution_eligible"]
    assert not day["volume_forming_flow"]["valid"]
    assert not day["price_volume_forming_flow"]["valid"]
    assert report["rows"][1]["source_selected_market_day_valid"]
    assert not report["training_ready"] and not report["decision_time_qualified"]
    # Prove preservation and non-application in the actual persisted spool,
    # not only through the report's flags. The old payload is not replaced.
    connection = sqlite3.connect(f"file:{report['spool']['path']}?mode=ro", uri=True)
    try:
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 3
        assert connection.execute("SELECT kind,active FROM events WHERE ordinal=3").fetchone() == (None, 0)
        assert connection.execute("SELECT price FROM events WHERE ticker='AAA' AND active=1").fetchall() == [("100",)]
    finally:
        connection.close()
    assert file_sha256(Path(report["spool"]["path"])) == report["spool"]["sha256"]
    assert file_sha256(args["root"] / args["payload_relative_path"]) == args["expected_compressed_sha256"]


def test_volume_only_trade_does_not_invent_price_or_fill(tmp_path: Path) -> None:
    report, _, _ = _run(tmp_path, [_trade("V", 36, "40", "10", conditions=(2,))])
    day = report["rows"][0]
    assert day["source_selected_market_day_valid"]
    assert day["bars_values"][:4] == [0.0] * 4
    assert day["bars_valid"][:4] == [False] * 4
    assert day["bars_values"][4:6] == [10.0, 400.0]
    assert day["bars_valid"][4:6] == [True, True]
    assert all(day["legacy_tape_valid"])
    assert day["volume_forming_flow"]["valid"]
    assert not day["price_volume_forming_flow"]["valid"]
    assert not day["fill"]["valid"]


def test_exact_fractional_size_quantiles_do_not_use_sqlite_float_sort(tmp_path: Path) -> None:
    sizes = ("10.01", "2.001", "0.12", "0.009", "2.0001", "9.999")
    report, selected, _ = _run(tmp_path, [_trade(str(i), 35 + i, "10", size, sequence=i + 1) for i, size in enumerate(sizes)])
    expected = native_tape("TEST-AAA", tuple(row.extracted_row for row in selected), (), _authorities()[2])
    assert report["rows"][0]["legacy_tape_values"] == list(expected.values)


def test_incomplete_spool_cannot_finalize_from_another_complete_receipt(tmp_path: Path) -> None:
    args = _source(tmp_path / "raw", HEADER + _trade("A", 36, "10", "1") + _trade("B", 37, "10", "1"))
    selected = []
    scan = scan_massive_selected_trade_file_v1(**args, row_sink=selected.append)
    with _spool(tmp_path, args) as spool:
        spool.append(selected[0])
        with pytest.raises(Qt200MarketDayError, match="reconcile"):
            spool.finalize(scan)
        assert spool.path.exists()


@pytest.mark.parametrize("legacy_full_indexes", [False, True])
def test_derived_spool_field_tamper_is_rejected_before_replay(tmp_path: Path, legacy_full_indexes: bool) -> None:
    args = _source(tmp_path / "raw", HEADER + _trade("A", 36, "10", "1"))
    with _spool(tmp_path, args) as spool:
        if legacy_full_indexes:
            _legacy_full_indexes(spool)
        scan = scan_massive_selected_trade_file_v1(**args, row_sink=spool.append)
        spool.connection.execute("UPDATE events SET price='999'")
        with pytest.raises(Qt200MarketDayError, match="original bound row"):
            spool.finalize(scan)


def test_append_and_finalize_are_not_repeatable_publication_paths(tmp_path: Path) -> None:
    args = _source(tmp_path / "raw", HEADER + _trade("A", 36, "10", "1"))
    selected = []
    scan = scan_massive_selected_trade_file_v1(**args, row_sink=selected.append)
    with _spool(tmp_path, args) as spool:
        spool.append(selected[0])
        with pytest.raises(Qt200MarketDayError, match="physical order"):
            spool.append(selected[0])
        report = spool.finalize(scan)
        with pytest.raises(Qt200MarketDayError, match="once"):
            spool.finalize(scan)
        assert Path(report["spool"]["path"]).exists()
        assert report["spool"]["bytes"] < report["spool"]["allocation_bytes"]
