from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive.decision_clock import (
    build_massive_decision_clock_authority,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.data_sources.massive.trade_replay import (
    MassiveResolvedSecurityIdentity,
    normalize_massive_delayed_websocket_trade,
    normalize_massive_trade_event,
    replay_massive_trades,
)
from rl_quant.data_sources.massive.websocket_capture import (
    MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
    MassiveDelayedWebSocketEvent,
    MassiveWebSocketCaptureError,
    MassiveWebSocketCaptureLifecycle,
    build_massive_delayed_websocket_capture_authority,
    parse_massive_delayed_websocket_capture,
)
from rl_quant.evaluation.massive_replay_artifacts import (
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


def _loaded_source(root: Path, *, relative: str, payload: bytes):
    entitlement = _entitlement()
    root.mkdir(parents=True, exist_ok=True)
    publish_massive_source_object(
        stream=BytesIO(payload),
        root=root,
        relative_payload_path=relative,
        dataset_id="massive-parity-test",
        source_object_key=relative,
        requested_at_ms=0,
        downloaded_at_ms=1,
        schema_sha256="e" * 64,
        entitlement_receipt_sha256=entitlement.receipt_sha256,
        committed_at_ms=2,
    )
    return load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=3
    )


def _parity_input(
    tmp_path: Path,
    *,
    disconnected: bool = False,
    received_after_decision: bool = False,
):
    eastern = ZoneInfo("America/New_York")
    session_date = "2026-08-20"
    open_at = datetime(2026, 8, 20, 9, 30, tzinfo=eastern)
    close_at = datetime(2026, 8, 20, 16, 0, tzinfo=eastern)
    open_ns = int(open_at.timestamp() * 1_000_000_000)
    close_ns = int(close_at.timestamp() * 1_000_000_000)
    sip_at = open_at + timedelta(minutes=30)
    sip_ms = int(sip_at.timestamp() * 1_000)
    participant_ms = sip_ms - 1
    raw_payload = {
        "ev": "T",
        "sym": "AAA",
        "x": 4,
        "i": "T1",
        "p": 10.0,
        "s": 100,
        "c": [1],
        "t": sip_ms,
        "pt": participant_ms,
        "q": 1,
        "trfi": 12,
        "trft": sip_ms,
    }
    captured_event = MassiveDelayedWebSocketEvent.from_payload(
        raw_payload,
        received_at_ns=(close_ns + 61 * 60 * 1_000_000_000)
        if received_after_decision
        else sip_ms * 1_000_000 + 15 * 60 * 1_000_000_000,
    )
    raw_source = _loaded_source(
        tmp_path,
        relative="captures/2026-08-20.jsonl",
        payload=captured_event.canonical_payload_json.encode(),
    )
    final_source = _loaded_source(
        tmp_path,
        relative="trades/2026-08-20.csv.gz",
        payload=b"finalized-row",
    )
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
    identity = MassiveResolvedSecurityIdentity.build(
        security_id="SEC-A",
        source_ticker="AAA",
        primary_exchange="XNYS",
        session_date=session_date,
        valid_from_ns=open_ns - 1,
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
    delayed_event = normalize_massive_delayed_websocket_trade(
        captured_event,
        source_object_receipt=raw_source.receipt,
        source_row_number=1,
        **common,
    )
    finalized_event = normalize_massive_trade_event(
        {
            "ticker": "AAA",
            "id": "T1",
            "exchange": 4,
            "sequence_number": 1,
            "participant_timestamp": participant_ms * 1_000_000,
            "sip_timestamp": sip_ms * 1_000_000,
            "price": 10.0,
            "decimal_size": "100",
            "conditions": [1],
            "correction": 0,
            "trf_id": 12,
            "trf_timestamp": sip_ms * 1_000_000,
        },
        source_object_receipt=final_source.receipt,
        source_row_number=1,
        **common,
    )
    delayed_replay = replay_massive_trades(
        (delayed_event,),
        decision_clock=decision_clock,
        source_object_receipt=raw_source.receipt,
        **common,
    )
    finalized_replay = replay_massive_trades(
        (finalized_event,),
        decision_clock=decision_clock,
        source_object_receipt=final_source.receipt,
        **common,
    )
    lifecycle = MassiveWebSocketCaptureLifecycle.build(
        decision_clock=decision_clock,
        connected_at_ns=open_ns - 10,
        authenticated_at_ns=open_ns - 9,
        subscribed_at_ns=open_ns - 8,
        last_heartbeat_at_ns=max(
            decision_clock.decision_at_ns, captured_event.received_at_ns
        ),
        disconnected_intervals=((open_ns + 1, open_ns + 2),) if disconnected else (),
        authentication_ack_sha256="6" * 64,
        subscription_ack_sha256="7" * 64,
        raw_capture_source_receipt_sha256=raw_source.receipt.receipt_sha256,
        subscription_universe_receipt_sha256=semantic_sha256(("AAA",)),
    )
    capture = build_massive_delayed_websocket_capture_authority(
        (captured_event,),
        lifecycle=lifecycle,
        subscribed_tickers=("AAA",),
        entitlement_authority=_entitlement(),
        raw_capture_source_receipt=raw_source.receipt,
    )
    feature_spec = MassiveReplayFeatureSpec.canonical()
    delayed_features = materialize_massive_replay_features(
        delayed_replay, specification=feature_spec
    )
    finalized_features = materialize_massive_replay_features(
        finalized_replay, specification=feature_spec
    )
    parser_spec = semantic_sha256("test-parser-v1")
    row = MassiveReplayParityInput(
        canary_kind="normal-session",
        capture=capture,
        delayed_source=raw_source,
        finalized_source=final_source,
        delayed_extraction=MassiveTradeExtractionManifest.from_loaded_replay(
            loaded_source=raw_source,
            replay=delayed_replay,
            parser_spec_sha256=parser_spec,
        ),
        finalized_extraction=MassiveTradeExtractionManifest.from_loaded_replay(
            loaded_source=final_source,
            replay=finalized_replay,
            parser_spec_sha256=parser_spec,
        ),
        decision_clock=decision_clock,
        session=session,
        delayed_replay=delayed_replay,
        finalized_replay=finalized_replay,
        delayed_features=delayed_features,
        finalized_features=finalized_features,
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


def test_committed_parity_is_development_only_without_runtime_entitlement(
    tmp_path: Path,
) -> None:
    row, common = _parity_input(tmp_path)

    authority = _build(row, common)

    assert authority.development_asof_replay_authorized
    assert not authority.runtime_entitlement_qualified
    assert not authority.canonical_source_parsers_qualified
    assert not authority.historical_asof_replay_authorized
    assert not authority.predictive_training_authorized


def test_actual_receipt_after_decision_fails_event_and_feature_parity(
    tmp_path: Path,
) -> None:
    row, common = _parity_input(tmp_path, received_after_decision=True)

    authority = _build(row, common)

    assert row.delayed_replay.post_cutoff_event_count == 1
    assert row.finalized_replay.visible_event_count == 1
    assert authority.failed_event_symbol_days
    assert authority.failed_feature_symbol_days


def test_capture_disconnect_is_evidence_not_self_asserted_completeness(
    tmp_path: Path,
) -> None:
    row, common = _parity_input(tmp_path, disconnected=True)

    authority = _build(row, common)

    assert not row.capture.capture_complete
    assert authority.failed_event_symbol_days


def test_parity_rejects_feature_artifact_from_another_replay(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path)
    bad = replace(row.delayed_features, input_replay_receipt_sha256="f" * 64)
    bad = replace(bad, receipt_sha256=semantic_sha256(bad.unsigned()))

    with pytest.raises(MassiveReplayParityError, match="another replay"):
        _build(replace(row, delayed_features=bad), common)


def test_real_websocket_event_requires_official_trade_fields() -> None:
    with pytest.raises(MassiveWebSocketCaptureError, match="exchange"):
        MassiveDelayedWebSocketEvent.from_payload(
            {"ev": "T", "sym": "AAA", "t": 1, "q": 1}, received_at_ns=2
        )


def test_capture_lifecycle_is_derived_from_committed_jsonl(tmp_path: Path) -> None:
    row, common = _parity_input(tmp_path / "base")
    clock = row.decision_clock
    sip_ms = (clock.regular_open_ns + 30 * 60 * 1_000_000_000) // 1_000_000
    trade = {
        "ev": "T",
        "sym": "AAA",
        "x": 4,
        "i": "T1",
        "p": 10,
        "s": 100,
        "c": [1],
        "t": sip_ms,
        "pt": sip_ms - 1,
        "q": 1,
    }
    rows = (
        {
            "schema": MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
            "kind": "server-batch",
            "recorded_at_ns": clock.regular_open_ns - 3,
            "payload": [
                {"ev": "status", "status": "connected", "message": "ok"}
            ],
        },
        {
            "schema": MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
            "kind": "server-batch",
            "recorded_at_ns": clock.regular_open_ns - 2,
            "payload": [
                {"ev": "status", "status": "auth_success", "message": "ok"}
            ],
        },
        {
            "schema": MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
            "kind": "subscription-ack",
            "recorded_at_ns": clock.regular_open_ns - 1,
            "tickers": ["AAA"],
            "payload": {"ev": "status", "status": "success", "message": "subscribed"},
        },
        {
            "schema": MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
            "kind": "server-batch",
            "recorded_at_ns": sip_ms * 1_000_000 + 15 * 60 * 1_000_000_000,
            "payload": [trade],
        },
        {
            "schema": MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
            "kind": "recorder-checkpoint",
            "recorded_at_ns": clock.decision_at_ns,
        },
    )
    payload = b"\n".join(canonical_json_payload(item) for item in rows) + b"\n"
    loaded = _loaded_source(
        tmp_path / "parsed", relative="captures/raw.jsonl", payload=payload
    )

    events, capture = parse_massive_delayed_websocket_capture(
        root=tmp_path / "parsed",
        loaded_source=loaded,
        decision_clock=clock,
        entitlement_authority=common["entitlement_authority"],
    )

    assert len(events) == 1
    assert capture.capture_complete
    assert capture.capture_file_parser_qualified
    assert capture.loaded_source_receipt_sha256 == loaded.receipt_sha256
    assert capture.subscribed_tickers == ("AAA",)

    normalized = normalize_massive_delayed_websocket_trade(
        events[0],
        source_object_receipt=loaded.receipt,
        source_row_number=1,
        **common,
    )
    replay = replay_massive_trades(
        (normalized,),
        decision_clock=clock,
        source_object_receipt=loaded.receipt,
        **common,
    )
    manifest = MassiveTradeExtractionManifest.from_delayed_capture(
        loaded_source=loaded,
        capture=capture,
        capture_events=events,
        replay=replay,
        source_ticker="AAA",
    )
    assert manifest.canonical_parser_qualified
