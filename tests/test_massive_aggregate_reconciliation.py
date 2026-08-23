from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.data_sources.massive.aggregate_reconciliation import (
    MassiveAggregateReconciliationError,
    MassiveAggregateReconciliationSpec,
    MassiveVendorAggregateBar,
    reconcile_massive_aggregate_bars,
    reconstruct_massive_five_minute_bars,
)
from rl_quant.data_sources.massive.conditions import (
    MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    build_massive_condition_authority,
)
from rl_quant.data_sources.massive.session_calendar import MassiveExchangeSession
from rl_quant.data_sources.massive.trade_replay import (
    normalize_massive_trade_event,
    replay_massive_trades,
)
from test_massive_trade_replay import _decision_clock, _normalization_authorities


FIVE_MINUTES_NS = 300_000_000_000


def _conditions():
    return build_massive_condition_authority(
        (
            {
                "id": 1,
                "name": "Regular Sale",
                "asset_class": "stocks",
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
        source_query_path=MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    )


def _event(
    *,
    trade_id: str,
    sequence: int,
    sip: int,
    price: float,
    authorities=None,
    condition: int = 1,
):
    resolved = _normalization_authorities() if authorities is None else authorities
    origin = resolved["session"].regular_open_ns
    return normalize_massive_trade_event(
        {
            "ticker": "AAA",
            "id": trade_id,
            "exchange": 4,
            "sequence_number": sequence,
            "participant_timestamp": origin + sip - 5,
            "sip_timestamp": origin + sip,
            "price": price,
            "decimal_size": "100",
            "conditions": [condition],
            "correction": 0,
        },
        source_row_number=sequence,
        **resolved,
    )


def _session(*, intervals: int = 78) -> MassiveExchangeSession:
    regular_open_ns = _normalization_authorities()["session"].regular_open_ns
    return MassiveExchangeSession(
        session_date="2026-08-20",
        exchange="XNYS",
        regular_open_ns=regular_open_ns,
        regular_close_ns=regular_open_ns + intervals * FIVE_MINUTES_NS,
        scheduled_five_minute_intervals=intervals,
        special_session_reason=None if intervals == 78 else "early-close",
        calendar_source_receipt_sha256="d" * 64,
    )


def _vendor(bars):
    return tuple(
        MassiveVendorAggregateBar(
            security_id=bar.security_id,
            session_date=bar.session_date,
            interval_index=bar.interval_index,
            interval_start_ns=bar.interval_start_ns,
            interval_end_ns=bar.interval_end_ns,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            vwap=bar.vwap,
            transaction_count=bar.transaction_count,
            vendor_source_object_receipt_sha256="e" * 64,
            vendor_schema_receipt_sha256="f" * 64,
            adjusted=False,
            timespan="minute",
            multiplier=5,
        )
        for bar in bars
    )


def test_early_close_is_structural_session_length_not_missing_intervals() -> None:
    early = _session(intervals=42)

    early.validate()
    assert early.scheduled_five_minute_intervals == 42
    assert early.is_regular(early.regular_close_ns - 1)
    assert not early.is_regular(early.regular_close_ns)


def test_trade_reconstruction_keeps_sparse_support_and_reconciles() -> None:
    first = _event(trade_id="T1", sequence=1, sip=100, price=10.0)
    second = _event(
        trade_id="T2", sequence=2, sip=FIVE_MINUTES_NS + 100, price=12.0
    )
    replay = replay_massive_trades(
        (first, second),
        decision_clock=_decision_clock(),
        **_normalization_authorities(),
    )
    bars = reconstruct_massive_five_minute_bars(replay, session=_session())
    parity = reconcile_massive_aggregate_bars(
        bars, _vendor(bars), specification=MassiveAggregateReconciliationSpec.build()
    )

    assert [bar.interval_index for bar in bars] == [0, 1]
    assert parity.exact_support
    assert parity.all_values_match


def test_reconciliation_receipt_is_input_order_invariant() -> None:
    first = _event(trade_id="T1", sequence=1, sip=100, price=10.0)
    second = _event(
        trade_id="T2", sequence=2, sip=FIVE_MINUTES_NS + 100, price=12.0
    )
    replay = replay_massive_trades(
        (first, second),
        decision_clock=_decision_clock(),
        **_normalization_authorities(),
    )
    bars = reconstruct_massive_five_minute_bars(replay, session=_session())
    specification = MassiveAggregateReconciliationSpec.build()

    forward = reconcile_massive_aggregate_bars(
        bars, _vendor(bars), specification=specification
    )
    reverse = reconcile_massive_aggregate_bars(
        tuple(reversed(bars)),
        tuple(reversed(_vendor(bars))),
        specification=specification,
    )

    assert forward.receipt_sha256 == reverse.receipt_sha256


def test_aggregate_value_mutation_fails_parity() -> None:
    event = _event(trade_id="T1", sequence=1, sip=100, price=10.0)
    replay = replay_massive_trades(
        (event,),
        decision_clock=_decision_clock(),
        **_normalization_authorities(),
    )
    bars = reconstruct_massive_five_minute_bars(replay, session=_session())
    vendor = (replace(_vendor(bars)[0], close=11.0, high=11.0),)
    parity = reconcile_massive_aggregate_bars(
        bars, vendor, specification=MassiveAggregateReconciliationSpec.build()
    )

    assert not parity.all_values_match
    assert parity.mismatched_intervals == (0,)


def test_aggregate_identity_mismatch_fails_before_numeric_comparison() -> None:
    event = _event(trade_id="T1", sequence=1, sip=100, price=10.0)
    replay = replay_massive_trades(
        (event,),
        decision_clock=_decision_clock(),
        **_normalization_authorities(),
    )
    bars = reconstruct_massive_five_minute_bars(replay, session=_session())

    with pytest.raises(MassiveAggregateReconciliationError, match="identities differ"):
        reconcile_massive_aggregate_bars(
            bars,
            (replace(_vendor(bars)[0], security_id="SEC-B"),),
            specification=MassiveAggregateReconciliationSpec.build(),
        )


@pytest.mark.parametrize("tolerance", ("NaN", "Infinity", "-1"))
def test_reconciliation_rejects_nonfinite_or_negative_tolerance(
    tolerance: str,
) -> None:
    with pytest.raises(
        MassiveAggregateReconciliationError, match="finite and nonnegative"
    ):
        MassiveAggregateReconciliationSpec.build(
            price_absolute_tolerance=tolerance
        )


def test_reconciliation_rejects_empty_inputs() -> None:
    with pytest.raises(MassiveAggregateReconciliationError, match="nonempty"):
        reconcile_massive_aggregate_bars(
            (), (), specification=MassiveAggregateReconciliationSpec.build()
        )


@pytest.mark.parametrize(
    "mutation",
    (
        {"adjusted": True},
        {"volume": float("nan")},
    ),
)
def test_vendor_bar_rejects_adjusted_or_nonfinite_activity(
    mutation: dict[str, object],
) -> None:
    event = _event(trade_id="T1", sequence=1, sip=100, price=10.0)
    replay = replay_massive_trades(
        (event,),
        decision_clock=_decision_clock(),
        **_normalization_authorities(),
    )
    bars = reconstruct_massive_five_minute_bars(replay, session=_session())

    with pytest.raises(MassiveAggregateReconciliationError):
        replace(_vendor(bars)[0], **mutation).validate()


def test_split_condition_families_build_distinct_ohlc_and_volume() -> None:
    conditions = build_massive_condition_authority(
        (
            {
                "id": 10,
                "name": "Open Close",
                "asset_class": "stocks",
                "data_types": ["trade"],
                "update_rules": {"consolidated": {
                    "updates_high_low": False,
                    "updates_open_close": True,
                    "updates_volume": False,
                }},
            },
            {
                "id": 11,
                "name": "High Low",
                "asset_class": "stocks",
                "data_types": ["trade"],
                "update_rules": {"consolidated": {
                    "updates_high_low": True,
                    "updates_open_close": False,
                    "updates_volume": False,
                }},
            },
            {
                "id": 12,
                "name": "Volume",
                "asset_class": "stocks",
                "data_types": ["trade"],
                "update_rules": {"consolidated": {
                    "updates_high_low": False,
                    "updates_open_close": False,
                    "updates_volume": True,
                }},
            },
        ),
        source_object_receipt_sha256="a" * 64,
        source_query_path=MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    )
    authorities = _normalization_authorities()
    authorities["condition_authority"] = conditions
    events = (
        _event(trade_id="O1", sequence=1, sip=100, price=10, authorities=authorities, condition=10),
        _event(trade_id="O2", sequence=2, sip=200, price=11, authorities=authorities, condition=10),
        _event(trade_id="H1", sequence=3, sip=300, price=5, authorities=authorities, condition=11),
        _event(trade_id="H2", sequence=4, sip=400, price=20, authorities=authorities, condition=11),
        _event(trade_id="V1", sequence=5, sip=500, price=8, authorities=authorities, condition=12),
    )
    replay = replay_massive_trades(
        events, decision_clock=_decision_clock(), **authorities
    )

    bars = reconstruct_massive_five_minute_bars(replay, session=_session())

    assert len(bars) == 1
    assert (bars[0].open, bars[0].close) == (10, 11)
    assert (bars[0].low, bars[0].high) == (5, 20)
    assert bars[0].vwap == 8
    assert bars[0].transaction_count == 1
