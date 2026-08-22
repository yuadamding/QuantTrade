from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.data_sources.massive.conditions import (
    MassiveConditionError,
    build_massive_condition_authority,
)
from rl_quant.data_sources.massive.corrections import (
    MassiveCorrectionError,
    build_massive_correction_authority,
)
from rl_quant.data_sources.massive.entitlement import (
    build_massive_developer_entitlement_authority,
    documented_massive_surface,
)
from rl_quant.data_sources.massive.session_calendar import (
    FIVE_MINUTES_NS,
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import MassiveSourceObjectReceipt
from rl_quant.data_sources.massive.trade_replay import (
    MassiveResolvedSecurityIdentity,
    MassiveTradeReplayError,
    normalize_massive_trade_event,
    replay_massive_trades,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


def _conditions():
    return build_massive_condition_authority(
        (
            {
                "id": 1,
                "name": "Regular Sale",
                "data_types": ["trade"],
                "update_rules": {
                    "consolidated": {
                        "updates_high_low": True,
                        "updates_open_close": True,
                        "updates_volume": True,
                    }
                },
            },
            {
                "id": 2,
                "name": "Volume Only",
                "data_types": ["trade"],
                "update_rules": {
                    "consolidated": {
                        "updates_high_low": False,
                        "updates_open_close": False,
                        "updates_volume": True,
                    }
                },
            },
        ),
        source_object_receipt_sha256="a" * 64,
    )


def _corrections():
    return build_massive_correction_authority(
        (
            (0, "new-trade"),
            (1, "replacement"),
            (2, "cancellation"),
            (3, "late-report"),
        ),
        canary_receipt_sha256="b" * 64,
    )


def _entitlement():
    observed_at_ms = 1
    observations = tuple(
        documented_massive_surface(
            surface_id=surface,
            request_path=f"/{surface}",
            observed_at_ms=observed_at_ms,
        )
        for surface in (
            "delayed-websocket",
            "financials-and-ratios",
            "flat-files",
            "historical-quotes",
            "reference-rest",
            "trades-rest",
        )
    )
    return build_massive_developer_entitlement_authority(
        observations, observed_at_ms=observed_at_ms
    )


def _session():
    row = MassiveExchangeSession(
        session_date="2026-08-20",
        exchange="XNYS",
        regular_open_ns=0,
        regular_close_ns=78 * FIVE_MINUTES_NS,
        scheduled_five_minute_intervals=78,
        special_session_reason=None,
        calendar_source_receipt_sha256="d" * 64,
    )
    return row, build_massive_session_authority(
        (row,), calendar_source_receipt_sha256="d" * 64
    )


def _source():
    entitlement = _entitlement()
    body = {
        "schema": "rl-quant.massive-source-object-v1",
        "dataset_id": "massive-test",
        "source_object_key": "trades/2026-08-20.csv.gz",
        "requested_at_ms": 0,
        "downloaded_at_ms": 1,
        "content_length": 1,
        "etag": None,
        "request_id": None,
        "physical_sha256": "c" * 64,
        "schema_sha256": "e" * 64,
        "entitlement_receipt_sha256": entitlement.receipt_sha256,
    }
    return entitlement, MassiveSourceObjectReceipt(
        receipt_sha256=semantic_sha256(body), **body
    )


def _identity():
    return MassiveResolvedSecurityIdentity.build(
        security_id="SEC-A",
        source_ticker="AAA",
        primary_exchange="XNYS",
        session_date="2026-08-20",
        valid_from_ns=0,
        valid_to_ns=None,
        identity_authority_receipt_sha256="f" * 64,
        ticker_history_receipt_sha256="0" * 64,
    )


def _normalization_authorities():
    entitlement, source = _source()
    session, session_authority = _session()
    return {
        "entitlement_authority": entitlement,
        "session_authority": session_authority,
        "session": session,
        "condition_authority": _conditions(),
        "correction_authority": _corrections(),
        "source_object_receipt": source,
        "identity_resolution": _identity(),
    }


def _event(
    *, trade_id: str, sequence: int, correction: int, sip: int, price: float
):
    record = {
        "id": trade_id,
        "exchange": 4,
        "sequence_number": sequence,
        "participant_timestamp": sip - 5,
        "sip_timestamp": sip,
        "price": price,
        "decimal_size": "100.5",
        "conditions": [1],
        "correction": correction,
        "trf_id": 12,
        "trf_timestamp": sip - 2,
    }
    return normalize_massive_trade_event(
        record,
        source_row_number=sequence,
        **_normalization_authorities(),
    )


def test_replay_is_permutation_invariant_and_applies_replacement_cancellation() -> None:
    events = (
        _event(trade_id="T1", sequence=1, correction=0, sip=100, price=10.0),
        _event(trade_id="T1", sequence=2, correction=1, sip=110, price=11.0),
        _event(trade_id="T1", sequence=3, correction=2, sip=120, price=11.0),
        _event(trade_id="T2", sequence=4, correction=3, sip=130, price=12.0),
    )
    forward = replay_massive_trades(
        events,
        decision_at_ns=1_000_000_000_000,
        **_normalization_authorities(),
    )
    reverse = replay_massive_trades(
        tuple(reversed(events)),
        decision_at_ns=1_000_000_000_000,
        **_normalization_authorities(),
    )

    assert forward.receipt_sha256 == reverse.receipt_sha256
    assert [event.trade_id for event in forward.active_events] == ["T2"]
    assert forward.cancelled_event_keys == (events[0].event_key,)


def test_post_cutoff_mutation_does_not_change_visible_replay() -> None:
    visible = _event(trade_id="T1", sequence=1, correction=0, sip=100, price=10.0)
    future = _event(
        trade_id="T2", sequence=2, correction=0, sip=2_000_000_000_000, price=20.0
    )
    first = replay_massive_trades(
        (visible, future),
        decision_at_ns=1_000_000_000_000,
        **_normalization_authorities(),
    )
    second = replay_massive_trades(
        (visible, replace(future, price=999.0)),
        decision_at_ns=1_000_000_000_000,
        **_normalization_authorities(),
    )

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.active_events == second.active_events == (visible,)


def test_unknown_condition_or_correction_fails_closed() -> None:
    with pytest.raises(MassiveConditionError, match="unknown"):
        _conditions().resolve((999,))
    with pytest.raises(MassiveCorrectionError, match="unqualified"):
        _corrections().resolve(999)


def test_volume_only_condition_is_not_price_forming() -> None:
    assert _conditions().resolve((2,)) == (False, False, True)


def test_replay_rejects_authority_substitution_and_availability_forgery() -> None:
    event = _event(trade_id="T1", sequence=1, correction=0, sip=100, price=10.0)
    authorities = _normalization_authorities()

    with pytest.raises(MassiveTradeReplayError, match="condition_authority"):
        replay_massive_trades(
            (replace(event, condition_authority_receipt_sha256="9" * 64),),
            decision_at_ns=1_000_000_000_000,
            **authorities,
        )
    with pytest.raises(MassiveTradeReplayError, match="availability"):
        replay_massive_trades(
            (replace(event, strategy_available_timestamp_ns=event.sip_timestamp_ns),),
            decision_at_ns=1_000_000_000_000,
            **authorities,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("session_authority_receipt_sha256", "1" * 64),
        ("correction_authority_receipt_sha256", "2" * 64),
        ("source_object_receipt_sha256", "3" * 64),
        ("identity_authority_receipt_sha256", "4" * 64),
        ("ticker_history_receipt_sha256", "5" * 64),
        ("identity_resolution_receipt_sha256", "6" * 64),
    ),
)
def test_every_normalized_event_authority_is_reconciled(
    field: str, value: str
) -> None:
    event = _event(trade_id="T1", sequence=1, correction=0, sip=100, price=10.0)

    with pytest.raises(MassiveTradeReplayError, match=field):
        replay_massive_trades(
            (replace(event, **{field: value}),),
            decision_at_ns=1_000_000_000_000,
            **_normalization_authorities(),
        )


def test_caller_cannot_supply_an_entitlement_delay_or_ticker_identity() -> None:
    record = {
        "id": "T1",
        "ticker": "WRONG",
        "exchange": 4,
        "sequence_number": 1,
        "participant_timestamp": 95,
        "sip_timestamp": 100,
        "price": 10.0,
        "decimal_size": "100",
        "conditions": [1],
        "correction": 0,
    }
    with pytest.raises(MassiveTradeReplayError, match="ticker differs"):
        normalize_massive_trade_event(
            record,
            source_row_number=1,
            **_normalization_authorities(),
        )
    with pytest.raises(TypeError, match="entitlement_delay_ns"):
        normalize_massive_trade_event(
            {**record, "ticker": "AAA"},
            source_row_number=1,
            entitlement_delay_ns=True,  # type: ignore[call-arg]
            **_normalization_authorities(),
        )


def test_replay_reresolves_condition_and_correction_semantics() -> None:
    event = _event(trade_id="T1", sequence=1, correction=0, sip=100, price=10.0)
    with pytest.raises(MassiveTradeReplayError, match="condition eligibility"):
        replay_massive_trades(
            (replace(event, updates_open_close=False),),
            decision_at_ns=1_000_000_000_000,
            **_normalization_authorities(),
        )
    with pytest.raises(MassiveTradeReplayError, match="correction semantic"):
        replay_massive_trades(
            (replace(event, correction_kind="late-report"),),
            decision_at_ns=1_000_000_000_000,
            **_normalization_authorities(),
        )


def test_session_and_identity_are_derived_not_caller_flags() -> None:
    event = _event(
        trade_id="T1",
        sequence=1,
        correction=0,
        sip=78 * FIVE_MINUTES_NS + 100,
        price=10.0,
    )

    assert not event.regular_session
    with pytest.raises(MassiveTradeReplayError, match="session eligibility"):
        replay_massive_trades(
            (replace(event, regular_session=True),),
            decision_at_ns=100 * FIVE_MINUTES_NS,
            **_normalization_authorities(),
        )
