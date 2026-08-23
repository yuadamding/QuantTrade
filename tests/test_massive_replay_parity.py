from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import gzip
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive.decision_clock import build_massive_decision_clock_authority
from rl_quant.data_sources.massive.recorder_clock import MassiveRecorderClockAuthority
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADES_DATASET_ID,
    MASSIVE_FLAT_TRADE_COLUMNS,
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    extract_massive_flat_file_security_session,
)
from rl_quant.data_sources.massive.trade_replay import (
    MassiveResolvedSecurityIdentity,
    normalize_massive_canonical_trade_event,
    normalize_massive_delayed_websocket_trade,
    replay_massive_trades,
)
from rl_quant.data_sources.massive.websocket_capture import (
    MASSIVE_DELAYED_CAPTURE_DATASET_ID,
    MASSIVE_MAXIMUM_SILENT_INTERVAL_NS,
    MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
    MASSIVE_WEBSOCKET_CAPTURE_SOURCE_SCHEMA_SHA256,
    MassiveDelayedWebSocketEvent,
    MassiveWebSocketCaptureError,
    build_massive_delayed_websocket_capture_authority,
    parse_massive_delayed_websocket_capture,
)
from rl_quant.evaluation.massive_replay_artifacts import (
    MassiveReplayArtifactError,
    MassiveReplayFeatureSpec,
    MassiveTradeExtractionManifest,
    materialize_massive_replay_features,
)
from rl_quant.evaluation.massive_replay_parity import (
    MassiveReplayParityError,
    MassiveReplayParityInput,
    build_massive_delayed_replay_authority,
)
from rl_quant.protocol.canonical_artifact import canonical_json_payload, semantic_sha256
from test_massive_trade_replay import _conditions, _corrections, _entitlement


def _loaded_source(
    root: Path,
    *,
    relative: str,
    payload: bytes,
    dataset_id: str,
    schema_sha256: str,
):
    entitlement = _entitlement()
    root.mkdir(parents=True, exist_ok=True)
    publish_massive_source_object(
        stream=BytesIO(payload),
        root=root,
        relative_payload_path=relative,
        dataset_id=dataset_id,
        source_object_key=relative,
        requested_at_ms=0,
        downloaded_at_ms=1,
        schema_sha256=schema_sha256,
        entitlement_receipt_sha256=entitlement.receipt_sha256,
        committed_at_ms=2,
    )
    return load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=3
    )


def _clock() -> MassiveRecorderClockAuthority:
    return MassiveRecorderClockAuthority.build(
        host_id="station001",
        clock_source="chrony-tracking",
        synchronization_protocol="ntp",
        measured_before_capture_offset_ns=10_000_000,
        measured_after_capture_offset_ns=15_000_000,
        maximum_absolute_offset_ns=20_000_000,
        maximum_drift_ns=5_000_000,
        measurement_source_receipts=("a" * 64, "b" * 64),
    )


def _capture_rows(
    *,
    clock,
    recorder_clock: MassiveRecorderClockAuthority,
    trades: tuple[tuple[dict[str, object], int], ...],
    subscriptions: tuple[str, ...],
    disconnected: bool,
) -> tuple[dict[str, object], ...]:
    generation = "generation-0001"

    def row(kind: str, recorded_at_ns: int, **extra: object) -> dict[str, object]:
        return {
            "schema": MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
            "kind": kind,
            "recorded_at_ns": recorded_at_ns,
            "connection_generation_id": generation,
            **extra,
        }

    start = clock.source_day_start_ns
    rows = [
        row(
            "recorder-start",
            start - 4,
            recorder_source_sha256="c" * 64,
            recorder_image_receipt_sha256="d" * 64,
            recorder_clock_authority_receipt_sha256=recorder_clock.receipt_sha256,
        ),
        row(
            "server-batch",
            start - 3,
            payload=[{"ev": "status", "status": "connected", "message": "ok"}],
        ),
        row(
            "server-batch",
            start - 2,
            payload=[{"ev": "status", "status": "auth_success", "message": "ok"}],
        ),
        row(
            "subscription-ack",
            start - 1,
            tickers=list(subscriptions),
            payload={"ev": "status", "status": "success", "message": "subscribed"},
        ),
    ]
    heartbeat_at = start
    heartbeat_index = 0
    heartbeat_end = max(
        clock.decision_at_ns,
        *(received_at_ns for _, received_at_ns in trades),
    )
    while heartbeat_at < heartbeat_end:
        rows.append(
            row(
                "transport-heartbeat",
                heartbeat_at,
                transport_receipt_sha256=semantic_sha256(
                    (generation, heartbeat_index, heartbeat_at)
                ),
            )
        )
        heartbeat_index += 1
        heartbeat_at = min(
            heartbeat_at + MASSIVE_MAXIMUM_SILENT_INTERVAL_NS,
            heartbeat_end,
        )
    rows.append(
        row(
            "transport-heartbeat",
            heartbeat_end,
            transport_receipt_sha256=semantic_sha256(
                (generation, heartbeat_index, heartbeat_end)
            ),
        )
    )
    for payload, received_at_ns in trades:
        rows.append(row("server-batch", received_at_ns, payload=[payload]))
    if disconnected:
        rows.extend(
            (
                row("disconnect-start", clock.regular_open_ns + 1),
                row("disconnect-end", clock.regular_open_ns + 2),
            )
        )
    rows.append(row("recorder-checkpoint", heartbeat_end + 1))
    return tuple(sorted(rows, key=lambda item: int(item["recorded_at_ns"])))


def _flat_payload(
    *, ticker: str, sip_ns: int, participant_ns: int, trade_id: str = "T1",
    sequence: int = 1,
) -> bytes:
    header = ",".join(MASSIVE_FLAT_TRADE_COLUMNS)
    values = (
        ticker, "[1]", "0", "4", trade_id, str(participant_ns), "10.0",
        str(sequence), str(sip_ns), "100", "1", "12", str(sip_ns),
    )
    return gzip.compress(f"{header}\n{','.join(values)}\n".encode(), mtime=0)


def _parity_input(
    tmp_path: Path,
    *,
    disconnected: bool = False,
    received_after_decision: bool = False,
    extra_capture_ticker: str | None = None,
    premarket: bool = False,
):
    eastern = ZoneInfo("America/New_York")
    session_date = "2026-08-20"
    open_at = datetime(2026, 8, 20, 9, 30, tzinfo=eastern)
    close_at = datetime(2026, 8, 20, 16, 0, tzinfo=eastern)
    open_ns = int(open_at.timestamp() * 1_000_000_000)
    close_ns = int(close_at.timestamp() * 1_000_000_000)
    session = MassiveExchangeSession(
        session_date=session_date,
        exchange="XNYS",
        regular_open_ns=open_ns,
        regular_close_ns=close_ns,
        scheduled_five_minute_intervals=78,
        special_session_reason=None,
        calendar_source_receipt_sha256="3" * 64,
    )
    session_authority = build_massive_session_authority(
        (session,), calendar_source_receipt_sha256="3" * 64
    )
    decision_clock = build_massive_decision_clock_authority(
        session_authority=session_authority, session=session
    )
    recorder_clock = _clock()
    sip_at = (
        datetime(2026, 8, 20, 8, 0, tzinfo=eastern)
        if premarket
        else open_at + timedelta(minutes=30)
    )
    sip_ms = int(sip_at.timestamp() * 1_000)
    sip_ns = sip_ms * 1_000_000
    participant_ms = sip_ms - 1
    participant_ns = participant_ms * 1_000_000
    received_at_ns = (
        decision_clock.decision_at_ns + 1_000_000_000
        if received_after_decision
        else sip_ns + 15 * 60 * 1_000_000_000
    )
    trade = {
        "ev": "T", "sym": "AAA", "x": 4, "i": "T1", "p": 10.0,
        "s": 100, "c": [1], "t": sip_ms, "pt": participant_ms, "q": 1,
        "trfi": 12, "trft": sip_ms, "z": 1,
    }
    captures: list[tuple[dict[str, object], int]] = [(trade, received_at_ns)]
    subscriptions = ["AAA"]
    if extra_capture_ticker is not None:
        captures.append(
            ({**trade, "sym": extra_capture_ticker, "i": "T2", "q": 2}, received_at_ns + 1)
        )
        subscriptions.append(extra_capture_ticker)
    capture_rows = _capture_rows(
        clock=decision_clock,
        recorder_clock=recorder_clock,
        trades=tuple(captures),
        subscriptions=tuple(subscriptions),
        disconnected=disconnected,
    )
    capture_payload = b"\n".join(canonical_json_payload(item) for item in capture_rows) + b"\n"
    capture_key = f"{MASSIVE_DELAYED_CAPTURE_DATASET_ID}/2026/08/{session_date}.jsonl"
    raw_source = _loaded_source(
        tmp_path / "capture",
        relative=capture_key,
        payload=capture_payload,
        dataset_id=MASSIVE_DELAYED_CAPTURE_DATASET_ID,
        schema_sha256=MASSIVE_WEBSOCKET_CAPTURE_SOURCE_SCHEMA_SHA256,
    )
    flat_key = f"{MASSIVE_FLAT_TRADES_DATASET_ID}/2026/08/{session_date}.csv.gz"
    final_source = _loaded_source(
        tmp_path / "flat",
        relative=flat_key,
        payload=_flat_payload(ticker="AAA", sip_ns=sip_ns, participant_ns=participant_ns),
        dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
        schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    )
    identity = MassiveResolvedSecurityIdentity.build(
        security_id="SEC-A",
        source_ticker="AAA",
        primary_exchange="XNYS",
        session_date=session_date,
        valid_from_ns=decision_clock.source_day_start_ns - 1,
        valid_to_ns=None,
        identity_authority_receipt_sha256="4" * 64,
        ticker_history_receipt_sha256="5" * 64,
    )
    common = {
        "entitlement_authority": _entitlement(),
        "session_authority": session_authority,
        "session": session,
        "condition_authority": _conditions(),
        "correction_authority": _corrections(),
        "identity_resolution": identity,
    }
    captured_events, capture = parse_massive_delayed_websocket_capture(
        root=tmp_path / "capture",
        loaded_source=raw_source,
        decision_clock=decision_clock,
        recorder_clock_authority=recorder_clock,
        entitlement_authority=common["entitlement_authority"],
    )
    selected_capture_events = tuple(event for event in captured_events if event.ticker == "AAA")
    delayed_events = tuple(
        normalize_massive_delayed_websocket_trade(
            event,
            source_object_receipt=raw_source.receipt,
            source_row_number=index,
            recorder_clock_authority=recorder_clock,
            **common,
        )
        for index, event in enumerate(selected_capture_events, start=1)
    )
    delayed_replay = replay_massive_trades(
        delayed_events,
        decision_clock=decision_clock,
        source_object_receipt=raw_source.receipt,
        recorder_clock_authority=recorder_clock,
        **common,
    )
    flat_rows, flat_evidence = extract_massive_flat_file_security_session(
        root=tmp_path / "flat",
        loaded_source=final_source,
        identity_resolution=identity,
    )
    finalized_events = tuple(
        normalize_massive_canonical_trade_event(
            item.canonical_record,
            source_object_receipt=final_source.receipt,
            source_row_number=item.source_row_number,
            **common,
        )
        for item in flat_rows
    )
    finalized_replay = replay_massive_trades(
        finalized_events,
        decision_clock=decision_clock,
        source_object_receipt=final_source.receipt,
        **common,
    )
    feature_spec = MassiveReplayFeatureSpec.canonical()
    row = MassiveReplayParityInput(
        canary_kind="normal-session",
        capture=capture,
        delayed_source=raw_source,
        finalized_source=final_source,
        delayed_extraction=MassiveTradeExtractionManifest.from_delayed_capture(
            loaded_source=raw_source,
            capture=capture,
            capture_events=captured_events,
            replay=delayed_replay,
            source_ticker="AAA",
        ),
        finalized_extraction=MassiveTradeExtractionManifest.from_flat_file_evidence(
            evidence=flat_evidence, replay=finalized_replay
        ),
        finalized_flat_extraction_evidence=flat_evidence,
        decision_clock=decision_clock,
        recorder_clock_authority=recorder_clock,
        session=session,
        delayed_replay=delayed_replay,
        finalized_replay=finalized_replay,
        delayed_features=materialize_massive_replay_features(
            delayed_replay, specification=feature_spec
        ),
        finalized_features=materialize_massive_replay_features(
            finalized_replay, specification=feature_spec
        ),
    )
    return row, common


def _build(row, common):
    return build_massive_delayed_replay_authority(
        (row,),
        entitlement_authority=common["entitlement_authority"],
        session_authority=common["session_authority"],
        condition_authority=common["condition_authority"],
        correction_authority=common["correction_authority"],
    )


def test_committed_parity_is_development_only_without_runtime_entitlement(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path)
    authority = _build(row, common)
    assert authority.development_asof_replay_authorized
    assert not authority.runtime_entitlement_qualified
    assert authority.canonical_source_parsers_qualified
    assert not authority.historical_asof_replay_authorized
    assert not authority.predictive_training_authorized


def test_actual_receipt_after_decision_fails_event_and_feature_parity(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path, received_after_decision=True)
    authority = _build(row, common)
    assert row.delayed_replay.post_cutoff_event_count == 1
    assert row.finalized_replay.visible_event_count == 1
    assert authority.failed_event_symbol_days
    assert authority.failed_feature_symbol_days


def test_capture_disconnect_is_evidence_not_self_asserted_completeness(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path, disconnected=True)
    authority = _build(row, common)
    assert not row.capture.capture_complete
    assert authority.failed_event_symbol_days


def test_parity_rejects_forged_feature_payload(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path)
    bad = replace(row.delayed_features, canonical_feature_payload_json="{}")
    bad = replace(
        bad,
        output_feature_receipt_sha256=semantic_sha256({}),
        receipt_sha256=semantic_sha256(bad.unsigned()),
    )
    with pytest.raises((MassiveReplayArtifactError, MassiveReplayParityError)):
        _build(replace(row, delayed_features=bad), common)


def test_multi_ticker_capture_supports_one_security_parity(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path, extra_capture_ticker="BBB")
    authority = _build(row, common)
    assert row.capture.event_count == 2
    assert row.delayed_replay.input_event_count == 1
    assert authority.canonical_source_parsers_qualified


def test_full_source_day_capture_reconciles_premarket_trade(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path, premarket=True)
    authority = _build(row, common)
    assert row.capture.lifecycle.required_capture_start_ns == (
        row.decision_clock.source_day_start_ns
    )
    assert not row.delayed_replay.active_events[0].regular_session
    assert not authority.failed_event_symbol_days


def test_generic_capture_builder_cannot_claim_parser_qualification(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path)
    event = row.delayed_replay.active_events[0]
    unqualified = build_massive_delayed_websocket_capture_authority(
        (
            MassiveDelayedWebSocketEvent.from_payload(
                {
                    "ev": "T", "sym": event.source_ticker, "x": event.exchange_id,
                    "i": event.trade_id, "p": event.price, "s": int(event.decimal_size),
                    "c": list(event.conditions), "t": event.sip_timestamp_ns // 1_000_000,
                    "pt": event.participant_timestamp_ns // 1_000_000,
                    "q": event.sequence_number, "z": event.tape_id,
                },
                received_at_ns=event.actual_received_at_ns or 0,
            ),
        ),
        lifecycle=row.capture.lifecycle,
        subscribed_tickers=("AAA",),
        entitlement_authority=common["entitlement_authority"],
        raw_capture_source_receipt=row.delayed_source.receipt,
    )
    assert not unqualified.capture_file_parser_qualified
    assert unqualified.loaded_source_receipt_sha256 is None


def test_real_websocket_event_requires_official_trade_fields() -> None:
    with pytest.raises(MassiveWebSocketCaptureError, match="exchange"):
        MassiveDelayedWebSocketEvent.from_payload(
            {"ev": "T", "sym": "AAA", "t": 1, "q": 1}, received_at_ns=2
        )


def test_recorder_clock_uses_conservative_upper_receive_bound(tmp_path: Path) -> None:
    row, _ = _parity_input(tmp_path)
    event = row.delayed_replay.active_events[0]
    assert event.canonical_received_at_ns == (
        event.actual_received_at_ns
        + row.recorder_clock_authority.maximum_positive_clock_error_ns
    )
