from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import MassiveSourceObjectReceipt
from rl_quant.data_sources.massive.trade_replay import (
    MassiveResolvedSecurityIdentity,
    normalize_massive_trade_event,
    replay_massive_trades,
)
from rl_quant.data_sources.massive.websocket_capture import (
    MassiveDelayedWebSocketEvent,
    MassiveWebSocketCaptureError,
    MassiveWebSocketCaptureLifecycle,
    build_massive_delayed_websocket_capture_authority,
)
from rl_quant.evaluation.massive_replay_parity import (
    MassiveReplayFeatureArtifact,
    MassiveReplayParityError,
    MassiveReplayParityInput,
    build_massive_delayed_replay_authority,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from test_massive_trade_replay import (
    _conditions,
    _corrections,
    _entitlement,
)


def _source(*, physical: str, key: str) -> MassiveSourceObjectReceipt:
    entitlement = _entitlement()
    body = {
        "schema": "rl-quant.massive-source-object-v1",
        "dataset_id": "massive-parity-test",
        "source_object_key": key,
        "requested_at_ms": 0,
        "downloaded_at_ms": 1,
        "content_length": 1,
        "etag": None,
        "request_id": None,
        "physical_sha256": physical,
        "schema_sha256": "e" * 64,
        "entitlement_receipt_sha256": entitlement.receipt_sha256,
    }
    return MassiveSourceObjectReceipt(
        receipt_sha256=semantic_sha256(body), **body
    )


def _parity_input(*, disconnected: bool = False) -> tuple[MassiveReplayParityInput, dict]:
    eastern = ZoneInfo("America/New_York")
    session_date = "2026-08-20"
    open_at = datetime(2026, 8, 20, 9, 30, tzinfo=eastern)
    close_at = datetime(2026, 8, 20, 16, 0, tzinfo=eastern)
    open_ns = int(open_at.timestamp() * 1_000_000_000)
    close_ns = int(close_at.timestamp() * 1_000_000_000)
    sip_at = open_at + timedelta(minutes=30)
    sip_ns = int(sip_at.timestamp() * 1_000_000_000)
    sip_ms = sip_ns // 1_000_000
    participant_ns = sip_ns - 1_000
    decision_ns = close_ns + 60 * 60 * 1_000_000_000
    raw_source = _source(physical="1" * 64, key="captures/2026-08-20.jsonl")
    final_source = _source(physical="2" * 64, key="trades/2026-08-20.csv.gz")
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
    record = {
        "ev": "T",
        "sym": "AAA",
        "t": sip_ms,
        "q": 1,
        "id": "T1",
        "exchange": 4,
        "sequence_number": 1,
        "participant_timestamp": participant_ns,
        "sip_timestamp": sip_ns,
        "price": 10.0,
        "decimal_size": "100",
        "conditions": [1],
        "correction": 0,
        "trf_id": 12,
    }
    common = {
        "entitlement_authority": _entitlement(),
        "session_authority": session_authority,
        "session": session,
        "condition_authority": _conditions(),
        "correction_authority": _corrections(),
        "identity_resolution": identity,
    }
    delayed_event = normalize_massive_trade_event(
        record,
        source_object_receipt=raw_source,
        source_row_number=1,
        **common,
    )
    finalized_event = normalize_massive_trade_event(
        record,
        source_object_receipt=final_source,
        source_row_number=1,
        **common,
    )
    delayed_replay = replay_massive_trades(
        (delayed_event,),
        decision_at_ns=decision_ns,
        source_object_receipt=raw_source,
        **common,
    )
    finalized_replay = replay_massive_trades(
        (finalized_event,),
        decision_at_ns=decision_ns,
        source_object_receipt=final_source,
        **common,
    )
    captured_event = MassiveDelayedWebSocketEvent.from_payload(
        record,
        received_at_ns=sip_ns + 15 * 60 * 1_000_000_000,
    )
    lifecycle = MassiveWebSocketCaptureLifecycle.build(
        session_date=session_date,
        connected_at_ns=open_ns - 10,
        authenticated_at_ns=open_ns - 9,
        subscribed_at_ns=open_ns - 8,
        required_capture_start_ns=open_ns,
        required_capture_end_ns=decision_ns,
        last_heartbeat_at_ns=decision_ns,
        disconnected_intervals=((open_ns + 1, open_ns + 2),) if disconnected else (),
        authentication_ack_sha256="6" * 64,
        subscription_ack_sha256="7" * 64,
        clock_authority_receipt_sha256="8" * 64,
        raw_capture_source_receipt_sha256=raw_source.receipt_sha256,
        subscription_universe_receipt_sha256="9" * 64,
    )
    capture = build_massive_delayed_websocket_capture_authority(
        (captured_event,),
        lifecycle=lifecycle,
        subscribed_tickers=("AAA",),
        entitlement_authority=_entitlement(),
        raw_capture_source_receipt=raw_source,
    )
    delayed_features = MassiveReplayFeatureArtifact.build(
        security_id="SEC-A",
        session_date=session_date,
        source_replay_receipt_sha256=delayed_replay.receipt_sha256,
        feature_payload={"trade_count": 1, "volume": 100},
    )
    finalized_features = MassiveReplayFeatureArtifact.build(
        security_id="SEC-A",
        session_date=session_date,
        source_replay_receipt_sha256=finalized_replay.receipt_sha256,
        feature_payload={"trade_count": 1, "volume": 100},
    )
    parity_input = MassiveReplayParityInput(
        canary_kind="normal-session",
        capture=capture,
        finalized_source=final_source,
        session=session,
        delayed_replay=delayed_replay,
        finalized_replay=finalized_replay,
        delayed_features=delayed_features,
        finalized_features=finalized_features,
    )
    return parity_input, common


def test_typed_parity_is_derived_but_one_canary_cannot_authorize_history() -> None:
    row, common = _parity_input()

    authority = build_massive_delayed_replay_authority(
        (row,),
        entitlement_authority=common["entitlement_authority"],
        session_authority=common["session_authority"],
        condition_authority=common["condition_authority"],
        correction_authority=common["correction_authority"],
    )

    assert authority.development_asof_replay_authorized
    assert not authority.historical_asof_replay_authorized
    assert not authority.predictive_training_authorized
    assert authority.canary_kinds_present == ("normal-session",)


def test_capture_disconnect_is_evidence_not_self_asserted_completeness() -> None:
    row, common = _parity_input(disconnected=True)

    authority = build_massive_delayed_replay_authority(
        (row,),
        entitlement_authority=common["entitlement_authority"],
        session_authority=common["session_authority"],
        condition_authority=common["condition_authority"],
        correction_authority=common["correction_authority"],
    )

    assert not row.capture.capture_complete
    assert not authority.historical_asof_replay_authorized
    assert authority.failed_event_symbol_days


def test_capture_rejects_event_outside_subscription() -> None:
    row, _ = _parity_input()

    with pytest.raises(MassiveWebSocketCaptureError, match="not subscribed"):
        build_massive_delayed_websocket_capture_authority(
            (MassiveDelayedWebSocketEvent.from_payload(
                {
                    "ev": "T",
                    "sym": "OTHER",
                    "t": row.capture.lifecycle.required_capture_start_ns // 1_000_000,
                    "q": 2,
                },
                received_at_ns=row.capture.lifecycle.required_capture_start_ns,
            ),),
            lifecycle=row.capture.lifecycle,
            subscribed_tickers=("AAA",),
            entitlement_authority=_entitlement(),
            raw_capture_source_receipt=_source(
                physical="1" * 64, key="captures/2026-08-20.jsonl"
            ),
        )


def test_parity_rejects_feature_artifact_from_another_replay() -> None:
    row, common = _parity_input()
    bad_feature = replace(
        row.delayed_features,
        source_replay_receipt_sha256="f" * 64,
    )
    bad_feature = replace(
        bad_feature, receipt_sha256=semantic_sha256(bad_feature.unsigned())
    )

    with pytest.raises(MassiveReplayParityError, match="another replay"):
        build_massive_delayed_replay_authority(
            (replace(row, delayed_features=bad_feature),),
            entitlement_authority=common["entitlement_authority"],
            session_authority=common["session_authority"],
            condition_authority=common["condition_authority"],
            correction_authority=common["correction_authority"],
        )
