from __future__ import annotations

from dataclasses import replace

from rl_quant.data_sources.massive.aggregate_reconciliation import (
    reconcile_massive_aggregate_bars,
    reconstruct_massive_five_minute_bars,
)
from rl_quant.data_sources.massive.conditions import build_massive_condition_authority
from rl_quant.data_sources.massive.corrections import build_massive_correction_authority
from rl_quant.data_sources.massive.session_calendar import MassiveExchangeSession
from rl_quant.data_sources.massive.trade_replay import (
    normalize_massive_trade_event,
    replay_massive_trades,
)


FIVE_MINUTES_NS = 300_000_000_000


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
        ),
        source_object_receipt_sha256="a" * 64,
    )


def _corrections():
    return build_massive_correction_authority(
        ((0, "new-trade"),), canary_receipt_sha256="b" * 64
    )


def _event(*, trade_id: str, sequence: int, sip: int, price: float):
    return normalize_massive_trade_event(
        {
            "id": trade_id,
            "exchange": 4,
            "sequence_number": sequence,
            "participant_timestamp": sip - 5,
            "sip_timestamp": sip,
            "price": price,
            "decimal_size": "100",
            "conditions": [1],
            "correction": 0,
        },
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


def _session(*, intervals: int = 78) -> MassiveExchangeSession:
    return MassiveExchangeSession(
        session_date="2026-08-20",
        exchange="XNYS",
        regular_open_ns=0,
        regular_close_ns=intervals * FIVE_MINUTES_NS,
        scheduled_five_minute_intervals=intervals,
        special_session_reason=None if intervals == 78 else "early-close",
        calendar_source_receipt_sha256="d" * 64,
    )


def test_early_close_is_structural_session_length_not_missing_intervals() -> None:
    early = _session(intervals=42)

    early.validate()
    assert early.scheduled_five_minute_intervals == 42
    assert early.is_regular(early.regular_close_ns - 1)
    assert not early.is_regular(early.regular_close_ns)


def test_trade_reconstruction_keeps_sparse_support_and_reconciles() -> None:
    first = replace(
        _event(trade_id="T1", sequence=1, sip=100, price=10.0),
        participant_timestamp_ns=100,
        sip_timestamp_ns=105,
        strategy_available_timestamp_ns=205,
    )
    second = replace(
        _event(trade_id="T2", sequence=2, sip=FIVE_MINUTES_NS + 100, price=12.0),
        participant_timestamp_ns=FIVE_MINUTES_NS + 100,
        sip_timestamp_ns=FIVE_MINUTES_NS + 105,
        strategy_available_timestamp_ns=FIVE_MINUTES_NS + 205,
    )
    replay = replay_massive_trades(
        (first, second),
        decision_at_ns=10 * FIVE_MINUTES_NS,
        condition_authority=_conditions(),
        correction_authority=_corrections(),
    )
    bars = reconstruct_massive_five_minute_bars(replay, session=_session())
    parity = reconcile_massive_aggregate_bars(
        bars, bars, vendor_aggregate_receipt_sha256="e" * 64
    )

    assert [bar.interval_index for bar in bars] == [0, 1]
    assert parity.exact_support
    assert parity.all_values_match


def test_aggregate_value_mutation_fails_parity() -> None:
    event = replace(
        _event(trade_id="T1", sequence=1, sip=100, price=10.0),
        participant_timestamp_ns=100,
        sip_timestamp_ns=105,
        strategy_available_timestamp_ns=205,
    )
    replay = replay_massive_trades(
        (event,),
        decision_at_ns=FIVE_MINUTES_NS,
        condition_authority=_conditions(),
        correction_authority=_corrections(),
    )
    bars = reconstruct_massive_five_minute_bars(replay, session=_session())
    vendor = (replace(bars[0], close=11.0, high=11.0),)
    parity = reconcile_massive_aggregate_bars(
        bars, vendor, vendor_aggregate_receipt_sha256="e" * 64
    )

    assert not parity.all_values_match
    assert parity.mismatched_intervals == (0,)
