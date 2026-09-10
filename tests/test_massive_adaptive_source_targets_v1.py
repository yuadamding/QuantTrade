from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
import inspect
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from rl_quant.alpha.contracts import (
    CorporateActionKind,
    CorporateActionRecord,
    TerminalEventKind,
    TerminalEventRecord,
)
from rl_quant.data_sources.massive.conditions import (
    MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    build_massive_condition_authority,
)
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    build_massive_finalized_feature_domain_spec_v0,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    load_massive_persisted_security_rows_v2,
    stream_and_persist_massive_daily_trade_partitions_v1,
)
from rl_quant.data_sources.massive.session_calendar import (
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MASSIVE_FLAT_TRADES_DATASET_ID,
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
from test_massive_finalized_whole_file_v0 import (
    CALENDAR_RECEIPT,
    ENTITLEMENT_RECEIPT,
    _correction_authority,
    _flat_payload,
    _identity_authority,
    _ns,
    _publish,
    _session,
    _trade_row,
)

_EASTERN = ZoneInfo("America/New_York")


def _ms(day: str, value: time) -> int:
    return int(datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp() * 1_000)


def _fill_conditions():
    return build_massive_condition_authority(
        tuple(
            {
                "id": code,
                "name": name,
                "asset_class": "stocks",
                "data_types": ["trade"],
                "update_rules": {
                    "consolidated": {
                        "updates_open_close": price,
                        "updates_high_low": high_low,
                        "updates_volume": volume,
                    }
                },
            }
            for code, name, price, high_low, volume in (
                (1, "Price and volume, no high-low", True, False, True),
                (2, "Volume only", False, True, True),
                (3, "Price only", True, True, False),
            )
        ),
        source_object_receipt_sha256=semantic_sha256("native-fill-conditions"),
        source_query_path=MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    )


@pytest.mark.parametrize(
    ("day", "include_eligible", "utc_hour"),
    (
        ("2024-01-03", True, 14),
        ("2024-07-02", True, 13),
        ("2024-01-03", False, 14),
    ),
)
def test_adaptive_fill_uses_exact_morning_price_and_volume_population(
    tmp_path: Path, day: str, include_eligible: bool, utc_hour: int
) -> None:
    session = _session(day)
    session_authority = build_massive_session_authority(
        (session,), calendar_source_receipt_sha256=CALENDAR_RECEIPT
    )
    condition_authority = _fill_conditions()
    corrections = _correction_authority()
    feature_spec = build_massive_finalized_feature_domain_spec_v0(
        condition_authority=condition_authority, correction_authority=corrections
    )
    definitions = (
        ("BEFORE", time(9, 34, 59), "90", "10", "", 0, None),
        ("OPEN", time(9, 35), "100", "2", "1" if include_eligible else "2", 0, None),
        ("CANCELLED", time(9, 38), "777", "50", "", 0, None),
        ("VOLUME", time(9, 40), "999", "50", "2", 0, None),
        ("PRICE", time(9, 41), "888", "50", "3", 0, None),
        (
            "LAST",
            time(9, 44, 59, 999000),
            "110",
            "1",
            "" if include_eligible else "3",
            0,
            None,
        ),
        ("END", time(9, 45), "120", "10", "", 0, None),
        ("CANCELLED", time(9, 38), "777", "50", "", 2, time(9, 46)),
    )
    raw_rows = []
    for sequence, (trade_id, at, price, size, condition, correction, reported) in enumerate(
        definitions, start=1
    ):
        raw = list(
            _trade_row(
                ticker="AAA",
                trade_id=trade_id,
                participant_ns=_ns(day, at),
                sip_ns=_ns(day, reported or at),
                price=price,
                size=size,
                correction=correction,
                sequence=sequence,
            )
        )
        raw[1] = "[]" if not condition else f"[{condition}]"
        raw_rows.append(tuple(raw))
    loaded = _publish(
        root=tmp_path / "source",
        key=canonical_massive_trade_object_key(day),
        payload=_flat_payload(tuple(raw_rows)),
        dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
        schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
        downloaded_at_ms=_ms(day, time(17)),
        etag="native-morning-fill-fixture",
    )
    _, _, manifest = stream_and_persist_massive_daily_trade_partitions_v1(
        source_root=tmp_path / "source",
        loaded_source=loaded,
        spool_root=tmp_path / "spool",
        persisted_root=tmp_path / "persisted",
        session_authority=session_authority,
        session=session,
        identity_authority=_identity_authority(day, ("AAA",)),
        condition_authority=condition_authority,
        correction_authority=corrections,
        feature_domain_spec=feature_spec,
        entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
        published_at_ms=_ms(day, time(17, 1)),
    )
    partition = manifest.partitions[0]
    _, active, _ = load_massive_persisted_security_rows_v2(
        root=tmp_path / "persisted", partition=partition
    )
    assert active
    assert all(row.canonical_record.trade_id != "CANCELLED" for row in active)
    # Only the daily receipt binding is a nonauthorizing unit-test stand-in.
    # Conditions, gzip parsing, correction replay and persisted trades are native.
    daily_receipt = semantic_sha256("daily")
    daily_row_receipt = semantic_sha256("daily-row")
    daily_session = SimpleNamespace(
        source_session_date=day,
        persisted_partition_manifest_receipt_sha256=manifest.receipt_sha256,
    )
    daily_row = SimpleNamespace(receipt_sha256=daily_row_receipt)
    daily = SimpleNamespace(
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        supported_security_ids=("SEC-AAA",),
        sessions=(daily_session,),
        semantic_receipt_sha256=daily_receipt,
        daily_input_data_qualified=False,
        validate=lambda: None,
        row=lambda **_: daily_row,
    )
    result = build_massive_adaptive_fill_source_v1(
        persisted_root=tmp_path / "persisted",
        session_authority=session_authority,
        condition_authority=condition_authority,
        daily_input_authority=daily,
        persisted_partition_manifests=(manifest,),
        required_session_dates=(day,),
        supported_security_ids=("SEC-AAA",),
    )

    row = result.rows[0]
    assert (row.fill_start_at_ms, row.fill_end_at_ms) == adaptive_fill_clock_v1(day)
    assert row.fill_start_at_ms == _ms(day, time(9, 35))
    assert (
        datetime.fromtimestamp(row.fill_start_at_ms / 1_000, ZoneInfo("UTC")).hour
        == utc_hour
    )
    assert row.valid is include_eligible
    assert row.qualifying_trade_count == (2 if include_eligible else 0)
    assert row.qualifying_share_volume == (3.0 if include_eligible else 0.0)
    assert row.qualifying_dollar_volume == (310.0 if include_eligible else 0.0)
    assert row.fill_vwap == pytest.approx(310.0 / 3.0 if include_eligible else 0.0)
    expected_trades = tuple(
        trade.receipt_sha256
        for trade in active
        if include_eligible and trade.canonical_record.trade_id in {"OPEN", "LAST"}
    )
    assert row.qualifying_trade_inventory_sha256 == semantic_sha256(expected_trades)
    assert row.persisted_partition_receipt_sha256 == partition.receipt_sha256
    assert row.daily_input_row_receipt_sha256 == daily_row_receipt
    assert result.source_paths_replayed
    assert not result.source_data_qualified
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
