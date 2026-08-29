from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
import inspect
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from rl_quant.alpha.contracts import (
    CorporateActionKind,
    CorporateActionRecord,
    TerminalEventKind,
    TerminalEventRecord,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import (
    MassiveAdaptiveFillRowV1,
    MassiveAdaptiveFillSourceV1Error,
    adaptive_fill_clock_v1,
    build_massive_adaptive_fill_source_v1,
)
from rl_quant.features.massive_adaptive_source_targets_v1 import (
    _build_path,
    build_massive_adaptive_source_targets_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.protocol.canonical_artifact import semantic_sha256

_EASTERN = ZoneInfo("America/New_York")


def _ms(day: str, value: time) -> int:
    return int(datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp() * 1_000)


@dataclass(frozen=True)
class _TradeRecord:
    participant_timestamp_ns: int
    price_decimal: str
    size_decimal: str
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class _Trade:
    canonical_record: _TradeRecord
    receipt_sha256: str


def _trade(day: str, at: time, *, price: float, size: float, condition: str) -> _Trade:
    body = (day, at.isoformat(), price, size, condition)
    return _Trade(
        canonical_record=_TradeRecord(
            participant_timestamp_ns=_ms(day, at) * 1_000_000,
            price_decimal=str(price),
            size_decimal=str(size),
            conditions=(condition,),
        ),
        receipt_sha256=semantic_sha256(body),
    )


def test_adaptive_fill_uses_exact_morning_price_and_volume_population(monkeypatch) -> None:
    day = "2024-01-03"
    session_receipt = semantic_sha256("sessions")
    condition_receipt = semantic_sha256("conditions")
    manifest_receipt = semantic_sha256("manifest")
    partition_receipt = semantic_sha256("partition")
    daily_receipt = semantic_sha256("daily")
    daily_row_receipt = semantic_sha256("daily-row")
    session_authority = SimpleNamespace(
        receipt_sha256=session_receipt,
        validate=lambda: None,
    )
    condition_authority = SimpleNamespace(
        receipt_sha256=condition_receipt,
        validate=lambda: None,
        resolve=lambda values: (
            values == ("ok",),
            False,
            values == ("ok",),
            False,
        ),
    )
    daily_session = SimpleNamespace(
        source_session_date=day,
        persisted_partition_manifest_receipt_sha256=manifest_receipt,
    )
    daily_row = SimpleNamespace(receipt_sha256=daily_row_receipt)
    daily = SimpleNamespace(
        session_authority_receipt_sha256=session_receipt,
        condition_authority_receipt_sha256=condition_receipt,
        supported_security_ids=("SEC-A",),
        sessions=(daily_session,),
        semantic_receipt_sha256=daily_receipt,
        daily_input_data_qualified=False,
        validate=lambda: None,
        row=lambda **_: daily_row,
    )
    partition = SimpleNamespace(
        security_id="SEC-A", receipt_sha256=partition_receipt
    )
    manifest = SimpleNamespace(
        source_session_date=day,
        receipt_sha256=manifest_receipt,
        partitions=(partition,),
        validate=lambda: None,
    )
    trades = (
        _trade(day, time(9, 34, 59), price=90.0, size=10.0, condition="ok"),
        _trade(day, time(9, 35), price=100.0, size=2.0, condition="ok"),
        _trade(day, time(9, 44, 59), price=110.0, size=1.0, condition="ok"),
        _trade(day, time(9, 40), price=999.0, size=50.0, condition="excluded"),
        _trade(day, time(9, 45), price=120.0, size=10.0, condition="ok"),
    )
    monkeypatch.setattr(
        "rl_quant.features.massive_adaptive_fill_source_v1.load_massive_persisted_security_rows_v2",
        lambda **_: ((), trades, ()),
    )

    result = build_massive_adaptive_fill_source_v1(
        persisted_root="/unused",
        session_authority=session_authority,
        condition_authority=condition_authority,
        daily_input_authority=daily,
        persisted_partition_manifests=(manifest,),
        required_session_dates=(day,),
        supported_security_ids=("SEC-A",),
    )

    row = result.rows[0]
    assert (row.fill_start_at_ms, row.fill_end_at_ms) == adaptive_fill_clock_v1(day)
    assert row.qualifying_trade_count == 2
    assert row.qualifying_share_volume == 3.0
    assert row.fill_vwap == pytest.approx(310.0 / 3.0)
    assert result.source_paths_replayed
    assert not result.predictive_training_authorized


def _sessions() -> tuple[SimpleNamespace, ...]:
    start = date(2024, 1, 2)
    days = []
    current = start
    while len(days) < 127:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(
        SimpleNamespace(
            session_date=day,
            regular_close_ns=_ms(day, time(16)) * 1_000_000,
        )
        for day in days
    )


def _fill_row(day: str, price: float, *, valid: bool = True) -> MassiveAdaptiveFillRowV1:
    start, end = adaptive_fill_clock_v1(day)
    body = {
        "session_date": day,
        "security_id": "SEC-A",
        "fill_start_at_ms": start,
        "fill_end_at_ms": end,
        "fill_vwap": price if valid else 0.0,
        "qualifying_share_volume": 1.0 if valid else 0.0,
        "qualifying_dollar_volume": price if valid else 0.0,
        "qualifying_trade_count": 1 if valid else 0,
        "valid": valid,
        "qualifying_trade_inventory_sha256": semantic_sha256((day, "trades")),
        "persisted_partition_receipt_sha256": semantic_sha256((day, "partition")),
        "daily_input_row_receipt_sha256": semantic_sha256((day, "daily")),
    }
    return MassiveAdaptiveFillRowV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )


def test_path_uses_boundary_fills_intermediate_closes_and_terminal_cash_carry() -> None:
    sessions = _sessions()
    fills = {
        row.session_date: row
        for offset in (0, 1, 5, 10, 21, 42, 63, 126)
        for row in (_fill_row(sessions[offset].session_date, 100.0 + 2.0 * offset),)
    }
    fill_source = SimpleNamespace(
        row=lambda *, session_date, security_id: fills[session_date]
    )
    daily_rows = {
        session.session_date: SimpleNamespace(
            bars_valid=(True,) * len(MASSIVE_DAILY_BARS_V0_FIELDS),
            bars_values=tuple(
                100.0 + offset if name == "close" else 1.0
                for name in MASSIVE_DAILY_BARS_V0_FIELDS
            ),
            daily_bar_row_receipt_sha256=semantic_sha256((session.session_date, "bar")),
        )
        for offset, session in enumerate(sessions)
    }
    daily = SimpleNamespace(
        sessions=tuple(
            SimpleNamespace(
                source_session_date=session.session_date,
                authenticated_get_completed_at_ms=(
                    session.regular_close_ns // 1_000_000 + 1
                ),
            )
            for session in sessions
        ),
        row=lambda *, session_date, security_id: daily_rows[session_date],
    )
    identity = SimpleNamespace(
        security_master=(
            SimpleNamespace(
                security_id="SEC-A",
                listing_at_ms=_ms(sessions[0].session_date, time(9, 30)) - 1,
                delisting_at_ms=_ms(sessions[5].session_date, time(9, 40)),
            ),
        )
    )
    dividend = CorporateActionRecord(
        event_id="DIV",
        security_id="SEC-A",
        kind=CorporateActionKind.CASH_DIVIDEND,
        effective_at_ms=_ms(sessions[2].session_date, time(10)),
        available_at_ms=_ms(sessions[2].session_date, time(10)),
        cash_per_share=1.0,
    )
    fallback = TerminalEventRecord(
        event_id="FALLBACK:DELIST",
        security_id="SEC-A",
        kind=TerminalEventKind.WORTHLESS,
        effective_at_ms=_ms(sessions[5].session_date, time(9, 40)),
        available_at_ms=_ms(sessions[5].session_date, time(9, 40)),
    )

    path = _build_path(
        security_id="SEC-A",
        decision_at_ms=_ms("2024-01-01", time(17)),
        sessions=sessions,
        fill_source=fill_source,
        daily_input=daily,
        identity=identity,
        events=(dividend, fallback),
        source_root_receipt=semantic_sha256("root"),
    )

    assert path.values[0] == 100.0
    assert path.values[1] == 102.0  # boundary VWAP, not the 101 close
    assert path.values[2] == 103.0  # 102 close plus the dividend cash
    assert path.unresolved_terminal_fallback_session_offset == 5
    assert path.values[5:] == (1.0,) * 122
    assert all(path.terminal[5:])

    bad_fills = fills | {
        sessions[10].session_date: _fill_row(
            sessions[10].session_date, 0.0, valid=False
        )
    }
    missing = _build_path(
        security_id="SEC-A",
        decision_at_ms=_ms("2024-01-01", time(17)),
        sessions=sessions,
        fill_source=SimpleNamespace(
            row=lambda *, session_date, security_id: bad_fills[session_date]
        ),
        daily_input=daily,
        identity=SimpleNamespace(
            security_master=(
                SimpleNamespace(
                    security_id="SEC-A",
                    listing_at_ms=identity.security_master[0].listing_at_ms,
                    delisting_at_ms=None,
                ),
            )
        ),
        events=(dividend,),
        source_root_receipt=semantic_sha256("root-missing"),
    )
    assert not missing.valid[10]
    assert missing.values[10] == 0.0
    assert missing.mark_kinds[10] == "missing"


def test_fill_row_rejects_close_window_clock() -> None:
    row = _fill_row("2024-01-03", 100.0)
    with pytest.raises(MassiveAdaptiveFillSourceV1Error, match="clock"):
        replace(row, fill_start_at_ms=_ms("2024-01-03", time(15, 50))).validate()


def test_source_target_builder_accepts_only_package_owned_origin() -> None:
    parameters = inspect.signature(
        build_massive_adaptive_source_targets_v1
    ).parameters

    assert "origin_authority" in parameters
    assert "security_ids" not in parameters
    assert "exposure_panel" not in parameters
