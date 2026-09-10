"""Preserve independent vendor clocks without weakening causal availability."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.trade_canonicalization import (
    MassiveTradeCanonicalizationError,
    canonicalize_massive_flat_file_trade,
    canonicalize_massive_rest_trade,
)
from rl_quant.data_sources.massive.trade_replay import (
    MassiveTradeReplayError,
    normalize_massive_trade_event,
    replay_massive_trades,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from test_massive_selected_trade_scan_v1 import HEADER
from test_massive_trade_replay import _decision_clock, _normalization_authorities, _session
from test_qt200_native_scan_v1 import _arguments, _publish, _reconstruct


def _record(offset_ns=23_205_468):
    session, _ = _session()
    sip = session.regular_open_ns + 1_000_000_000
    return dict(ticker="AAA", id="CLOCK", exchange=4, sequence_number=1,
                participant_timestamp=sip + offset_ns, sip_timestamp=sip,
                trf_id=12, trf_timestamp=sip - 10, tape=3,
                price="115.79", size="100", conditions=[1], correction=0)


@pytest.mark.parametrize("parser", [canonicalize_massive_rest_trade, canonicalize_massive_flat_file_trade])
@pytest.mark.parametrize("offset_ns", [-23_205_468, 0, 23_205_468])
def test_vendor_clock_offset_is_retained_exactly(parser, offset_ns):
    source = _record(offset_ns)
    before = dict(source)
    record = parser(source)
    record.validate()
    assert source == before
    assert record.participant_timestamp_ns == source["participant_timestamp"]
    assert record.sip_timestamp_ns == source["sip_timestamp"]
    assert record.trf_timestamp_ns == source["trf_timestamp"]
    assert record.participant_timestamp_ns - record.sip_timestamp_ns == offset_ns
    assert record.raw_source_record_sha256 == semantic_sha256(before)
    assert record.receipt_sha256 == semantic_sha256(record.unsigned())


@pytest.mark.parametrize("field", ["participant_timestamp", "sip_timestamp"])
@pytest.mark.parametrize("invalid", [True, -1, 1.5, None, "not-a-timestamp"])
def test_invalid_timestamp_values_still_fail(field, invalid):
    with pytest.raises(MassiveTradeCanonicalizationError):
        canonicalize_massive_flat_file_trade(dict(_record(), **{field: invalid}))


@pytest.mark.parametrize("field", ["participant_timestamp_ns", "sip_timestamp_ns"])
def test_canonical_timestamp_fields_cannot_remain_strings(field):
    record = canonicalize_massive_flat_file_trade(_record())
    altered = replace(record, **{field: str(getattr(record, field))})
    altered = replace(altered, receipt_sha256=semantic_sha256(altered.unsigned()))
    with pytest.raises(MassiveTradeCanonicalizationError, match="canonical timestamp"):
        altered.validate()


def test_clock_contract_change_invalidates_old_canonical_receipts():
    record = canonicalize_massive_rest_trade(_record())
    old_spec = semantic_sha256(dict(
        schema="rl-quant.massive-canonical-trade-source-record-v2",
        websocket_timestamp_unit="milliseconds", rest_timestamp_unit="nanoseconds",
        flat_file_timestamp_unit="nanoseconds", price_representation="canonical-decimal-string",
        size_representation="canonical-decimal-string", condition_order="sorted-unique",
        tape="integer-or-null", delayed_receive_time="clock-error-upper-bound",
    ))
    assert record.canonicalization_spec_sha256 != old_spec
    altered = replace(record, canonicalization_spec_sha256=old_spec)
    altered = replace(altered, receipt_sha256=semantic_sha256(altered.unsigned()))
    with pytest.raises(MassiveTradeCanonicalizationError, match="spec drifted"):
        altered.validate()


def test_positive_clock_offset_preserves_qualified_sip_delay_and_replay():
    authorities = _normalization_authorities()
    source = _record()
    event = normalize_massive_trade_event(source, source_row_number=1, **authorities)
    delay = authorities["entitlement_authority"].entitlement_delay_minutes * 60 * 1_000_000_000
    assert event.participant_timestamp_ns == source["participant_timestamp"]
    assert event.sip_timestamp_ns == source["sip_timestamp"]
    assert event.strategy_available_timestamp_ns == source["sip_timestamp"] + delay
    assert event.strategy_available_timestamp_ns >= event.participant_timestamp_ns
    assert event.availability_kind == "qualified-sip-delay"
    replay = replay_massive_trades((event,), decision_clock=_decision_clock(), **authorities)
    assert replay.active_events == (event,)


def test_clock_offset_cannot_make_a_future_participant_event_available():
    authorities = _normalization_authorities()
    delay = authorities["entitlement_authority"].entitlement_delay_minutes * 60 * 1_000_000_000
    with pytest.raises(MassiveTradeReplayError, match="availability precedes participant"):
        normalize_massive_trade_event(_record(delay + 1), source_row_number=1, **authorities)


def test_positive_clock_offset_cannot_bypass_post_cutoff_replay():
    authorities = _normalization_authorities()
    clock = _decision_clock()
    source = _record()
    source["sip_timestamp"] = clock.decision_at_ns
    source["participant_timestamp"] = clock.decision_at_ns + 23_205_468
    event = normalize_massive_trade_event(source, source_row_number=1, **authorities)
    replay = replay_massive_trades((event,), decision_clock=clock, **authorities)
    assert replay.active_events == ()
    assert replay.post_cutoff_event_count == 1


def test_positive_clock_offset_does_not_authorize_forged_availability():
    authorities = _normalization_authorities()
    event = normalize_massive_trade_event(_record(), source_row_number=1, **authorities)
    too_early = replace(event, strategy_available_timestamp_ns=event.participant_timestamp_ns - 1)
    with pytest.raises(MassiveTradeReplayError, match="availability precedes participant"):
        replay_massive_trades((too_early,), decision_clock=_decision_clock(), **authorities)


def test_actual_failed_clock_rows_complete_native_scan_without_timestamp_repair(tmp_path: Path):
    # Original 2017-01-03 AAPL source rows 105118, 108017, 108018, 108019.
    rows = (
        b'AAPL,"53,35,41",0,4,86,1483453802357000000,115.79,8209,1483453802333794532,100,3,12,1483453802333712156\n',
        b'AAPL,"53,35,41",0,4,916,1483453930151000000,116.27,25364,1483453930128059689,600,3,12,1483453930127997881\n',
        b'AAPL,"53,35,41",0,4,917,1483453930157000000,116.27,25365,1483453930134758002,400,3,12,1483453930134714476\n',
        b'AAPL,"53,35,41",0,4,918,1483453930164000000,116.27,25366,1483453930142003584,500,3,12,1483453930141951529\n',
    )
    hashes = (
        "1b1566a96b9d7e5f25d57b9f88c44085f863e2ccbfdff1e6c079e005d6cdaeb6",
        "682f50afc91185ee550bb7070396a068a710db0582348be7a0fef5d1249dc735",
        "b6c704ffb7a6a239b375a445a731e29d1b01443acffe9d402cbd3d708a266733",
        "69060f2524563d24f782c21527fff5259b3b755e092d51c3a6b3999f23e9580f",
    )
    assert tuple(hashlib.sha256(row).hexdigest() for row in rows) == hashes
    args = _arguments(tmp_path, HEADER + b"".join(rows))
    args["source_request"]["tickers"] = ("AAPL",)
    completion = _publish(args)
    before = {p.name: p.read_bytes() for p in args["output_directory"].iterdir()}
    prerequisite = json.loads(before["prerequisite.json"])
    assert prerequisite["selection_preflight"]["selected_canonicalization_error_count"] == 0
    assert completion["source_rows"] == completion["selected_rows"] == 4
    assert completion["training_ready"] is False
    native = _reconstruct(args)
    assert native.minimum_participant_timestamp_ns == 1483453802357000000
    assert native.minimum_sip_timestamp_ns == 1483453802333794532
    assert native.maximum_participant_timestamp_ns == 1483453930164000000
    assert native.maximum_sip_timestamp_ns == 1483453930142003584
    assert {p.name: p.read_bytes() for p in args["output_directory"].iterdir()} == before
