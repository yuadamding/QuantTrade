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
from rl_quant.data_sources.massive.trade_replay import (
    normalize_massive_trade_event,
    replay_massive_trades,
)


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
        security_id="SEC-A",
        source_ticker="AAA",
        session_date="2026-08-20",
        entitlement_delay_ns=100,
        regular_session=True,
        condition_authority=_conditions(),
        correction_authority=_corrections(),
        source_file_sha256="c" * 64,
        source_row_number=sequence,
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
        decision_at_ns=1_000,
        condition_authority=_conditions(),
        correction_authority=_corrections(),
    )
    reverse = replay_massive_trades(
        tuple(reversed(events)),
        decision_at_ns=1_000,
        condition_authority=_conditions(),
        correction_authority=_corrections(),
    )

    assert forward.receipt_sha256 == reverse.receipt_sha256
    assert [event.trade_id for event in forward.active_events] == ["T2"]
    assert forward.cancelled_event_keys == (events[0].event_key,)


def test_post_cutoff_mutation_does_not_change_visible_replay() -> None:
    visible = _event(trade_id="T1", sequence=1, correction=0, sip=100, price=10.0)
    future = _event(trade_id="T2", sequence=2, correction=0, sip=2_000, price=20.0)
    first = replay_massive_trades(
        (visible, future),
        decision_at_ns=1_000,
        condition_authority=_conditions(),
        correction_authority=_corrections(),
    )
    second = replay_massive_trades(
        (visible, replace(future, price=999.0)),
        decision_at_ns=1_000,
        condition_authority=_conditions(),
        correction_authority=_corrections(),
    )

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.active_events == second.active_events == (visible,)


def test_unknown_condition_or_correction_fails_closed() -> None:
    with pytest.raises(MassiveConditionError, match="unknown"):
        _conditions().resolve((999,))
    with pytest.raises(MassiveCorrectionError, match="unqualified"):
        _corrections().resolve(999)


def test_volume_only_condition_is_not_price_forming() -> None:
    assert _conditions().resolve((2,)) == (False, True)
