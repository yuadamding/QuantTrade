from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, date, datetime, time, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import rl_quant.features.massive_session_panel_v1 as session_panel_module
from rl_quant.alpha.contracts import CorporateActionKind, TerminalEventKind
from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    PITUniverseRuleSpec,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    MassiveSourceObjectError,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.features.massive_economic_event_source_v4 import (
    MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V4,
    MASSIVE_ECONOMIC_EVENT_SOURCE_V4_DATASETS,
    MASSIVE_ECONOMIC_EVENT_SOURCE_V4_SCHEMA,
    MASSIVE_ECONOMIC_EVENT_SOURCE_V4_SOURCE_SCHEMA_SHA256,
    MassiveEconomicEventSourceV4Error,
    build_massive_economic_event_authority_v4,
    economic_event_source_row_receipt_v4,
    expected_effective_timestamp_contract_v4,
    parse_massive_economic_event_source_v4,
)
from rl_quant.features.massive_economic_history_v4 import (
    build_massive_economic_history_origin_v4,
    build_massive_economic_history_rows_v4,
    materialize_massive_economic_history_v4,
    validate_massive_economic_history_v4,
)
from rl_quant.features.massive_session_panel_v1 import (
    MASSIVE_SESSION_PANEL_V1_DATASET,
    MASSIVE_SESSION_PANEL_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
    MassiveSessionInputReceiptV1,
    MassiveSessionPanelArtifactV1,
    MassiveSessionPanelRowV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)

NY = ZoneInfo("America/New_York")
DIGEST = "a" * 64


def _business_dates(count: int) -> tuple[date, ...]:
    output: list[date] = []
    current = date(2024, 1, 2)
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current)
        current += timedelta(days=1)
    return tuple(output)


def _sessions(count: int = 10) -> MassiveSessionAuthority:
    calendar_receipt = "1" * 64
    rows = []
    for session_date in _business_dates(count):
        open_ns = int(
            datetime.combine(session_date, time(9, 30), tzinfo=NY).timestamp()
            * 1_000_000_000
        )
        close_ns = int(
            datetime.combine(session_date, time(16), tzinfo=NY).timestamp()
            * 1_000_000_000
        )
        rows.append(
            MassiveExchangeSession(
                session_date=session_date.isoformat(),
                exchange="XNYS",
                regular_open_ns=open_ns,
                regular_close_ns=close_ns,
                scheduled_five_minute_intervals=78,
                special_session_reason=None,
                calendar_source_receipt_sha256=calendar_receipt,
            )
        )
    return build_massive_session_authority(
        tuple(rows), calendar_source_receipt_sha256=calendar_receipt
    )


def _identity(
    sessions: MassiveSessionAuthority, *, listing_at_ms: int | None = None
) -> PITSecurityUniverseAuthority:
    listed = (
        sessions.sessions[0].regular_open_ns // 1_000_000
        if listing_at_ms is None
        else listing_at_ms
    )
    effective = sessions.sessions[2].regular_open_ns // 1_000_000
    rule = PITUniverseRuleSpec.build(
        rule_id="economic-history-v4-test",
        target_size=2,
        ranking_lookback_sessions=3,
        ranking_lag_sessions=1,
        minimum_observed_sessions=2,
        minimum_close_price=1.0,
        minimum_average_dollar_volume=0.0,
        rebalance_frequency="monthly",
    )
    masters = tuple(
        SourcedSecurityMasterRecord(
            security_id=security_id,
            issuer_id=f"ISS-{security_id}",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=listed,
            delisting_at_ms=None,
            successor_security_id=None,
            corporate_action_chain_id="CHAIN-TEST",
            identity_source_receipt_sha256=semantic_sha256((security_id, "master")),
        )
        for security_id in ("SEC-A", "SEC-B")
    )
    tickers = tuple(
        SourcedTickerHistoryRecord(
            security_id=security_id,
            ticker=ticker,
            valid_from_ms=listed,
            valid_to_ms=None,
            available_at_ms=listed,
            primary_exchange="XNYS",
            source_receipt_sha256=semantic_sha256((security_id, "ticker")),
        )
        for security_id, ticker in (("SEC-A", "AAA"), ("SEC-B", "BBB"))
    )
    listings = tuple(
        ListingEventRecord(
            event_id=f"LIST-{security_id}",
            security_id=security_id,
            effective_at_ms=listed,
            available_at_ms=listed,
            primary_exchange="XNYS",
            ticker=ticker,
            source_receipt_sha256=semantic_sha256((security_id, "listing")),
        )
        for security_id, ticker in (("SEC-A", "AAA"), ("SEC-B", "BBB"))
    )
    ranks = tuple(
        UniverseRankInputRecord(
            security_id=security_id,
            effective_at_ms=effective,
            effective_session_index=10,
            available_at_ms=effective - 1,
            observation_start_ms=listed,
            observation_end_ms=effective - 2,
            observation_start_session_index=7,
            observation_end_session_index=9,
            observed_session_count=3,
            average_dollar_volume=1_000_000.0,
            close_price=100.0,
            source_receipt_sha256=semantic_sha256((security_id, "rank")),
        )
        for security_id in ("SEC-A", "SEC-B")
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=(),
        rank_inputs=ranks,
    )


def _panel_row(
    session_index: int,
    session: MassiveExchangeSession,
    security_id: str,
    *,
    close: float,
) -> MassiveSessionPanelRowV1:
    bars = {
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "share_volume": 10_000.0,
        "dollar_volume": close * 10_000.0,
        "high_low_range": 0.0,
        "close_location": 0.5,
    }
    tape = {
        "log_trade_count": 1.0,
        "median_trade_size": 100.0,
        "p90_trade_size": 200.0,
        "large_trade_dollar_fraction": 0.0,
        "quote_free_signed_dollar_flow_proxy": 0.0,
        "absolute_signed_flow_imbalance": 0.0,
        "price_response_per_signed_dollar": 0.0,
        "trf_off_exchange_dollar_fraction": 0.0,
        "venue_entropy": 0.0,
        "largest_venue_share": 1.0,
        "tape_a_fraction": 1.0,
        "tape_b_fraction": 0.0,
        "tape_c_fraction": 0.0,
        "special_condition_fraction": 0.0,
        "correction_replacement_fraction": 0.0,
    }
    provisional = MassiveSessionPanelRowV1(
        source_session_index=session_index,
        source_session_date=session.session_date,
        regular_open_ns=session.regular_open_ns,
        regular_close_ns=session.regular_close_ns,
        security_id=security_id,
        pit_member=True,
        listed=True,
        tradable=True,
        observed_regular_trade=True,
        halt_or_no_print=False,
        bars_values=tuple(bars[name] for name in MASSIVE_DAILY_BARS_V0_FIELDS),
        bars_valid=(True,) * len(MASSIVE_DAILY_BARS_V0_FIELDS),
        tape_values=tuple(tape[name] for name in MASSIVE_DAILY_TAPE_V0_FIELDS),
        tape_valid=(True,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS),
        event_timeline_count=1,
        replacement_event_count=0,
        cancellation_event_count=0,
        late_report_event_count=0,
        daily_bars_row_receipt_sha256="2" * 64,
        daily_tape_row_receipt_sha256="3" * 64,
        event_counts_receipt_sha256="4" * 64,
        receipt_sha256="0" * 64,
    )
    result = provisional.__class__(
        **{
            **provisional.unsigned(),
            "receipt_sha256": semantic_sha256(provisional.unsigned()),
        }
    )
    result.validate()
    return result


def _panel(sessions: MassiveSessionAuthority) -> tuple[MassiveSessionPanelRowV1, ...]:
    rows = []
    # The panel deliberately begins two sessions after the true listing session.
    for local_index, session in enumerate(sessions.sessions[2:7]):
        rows.extend(
            (
                _panel_row(local_index, session, "SEC-A", close=100.0),
                _panel_row(local_index, session, "SEC-B", close=20.0),
            )
        )
    return tuple(rows)


def _panel_artifact(
    root: Path,
    *,
    sessions: MassiveSessionAuthority,
    identity: PITSecurityUniverseAuthority,
) -> MassiveSessionPanelArtifactV1:
    rows = _panel(sessions)
    inputs = []
    for session in sessions.sessions[2:7]:
        body = {
            "source_session_date": session.session_date,
            "daily_bars_artifact_receipt_sha256": semantic_sha256(
                (session.session_date, "bars")
            ),
            "daily_tape_artifact_receipt_sha256": semantic_sha256(
                (session.session_date, "tape")
            ),
            "persisted_partition_manifest_receipt_sha256": semantic_sha256(
                (session.session_date, "partitions")
            ),
        }
        inputs.append(
            MassiveSessionInputReceiptV1(
                **body,
                receipt_sha256=semantic_sha256(body),
            )
        )
    relative = "panel/session-panel.json"
    payload = {
        "schema": session_panel_module.MASSIVE_SESSION_PANEL_V1_SCHEMA,
        "exchange": "XNYS",
        "start_session_date": sessions.sessions[2].session_date,
        "end_session_date": sessions.sessions[6].session_date,
        "session_authority_receipt_sha256": sessions.receipt_sha256,
        "identity_authority_receipt_sha256": identity.receipt_sha256,
        "condition_authority_receipt_sha256": "8" * 64,
        "input_receipts": tuple(asdict(row) for row in inputs),
        "rows": tuple(asdict(row) for row in rows),
        "session_count": 5,
        "security_count": 2,
        "row_count": len(rows),
        "member_row_count": len(rows),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "feature_spec_receipt_sha256": MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
        "feature_source_sha256": (
            session_panel_module.MASSIVE_SESSION_PANEL_V1_SOURCE_SHA256
        ),
    }
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_SESSION_PANEL_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_SESSION_PANEL_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=DIGEST,
        committed_at_ms=1,
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=1
    )
    provisional = MassiveSessionPanelArtifactV1(
        exchange="XNYS",
        start_session_date=sessions.sessions[2].session_date,
        end_session_date=sessions.sessions[6].session_date,
        session_authority_receipt_sha256=sessions.receipt_sha256,
        identity_authority_receipt_sha256=identity.receipt_sha256,
        condition_authority_receipt_sha256="8" * 64,
        input_receipts=tuple(inputs),
        rows=rows,
        session_count=5,
        security_count=2,
        row_count=len(rows),
        member_row_count=len(rows),
        row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        feature_spec_receipt_sha256=MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
        feature_source_sha256=(
            session_panel_module.MASSIVE_SESSION_PANEL_V1_SOURCE_SHA256
        ),
        loaded_source=loaded,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


def _record(source_kind: str, **values: object) -> dict[str, object]:
    row = dict(values)
    row["source_row_receipt_sha256"] = economic_event_source_row_receipt_v4(
        source_kind=source_kind, record=row
    )
    return row


def _corporate(
    event_id: str,
    security_id: str,
    kind: CorporateActionKind,
    effective_at_ms: int,
    *,
    available_at_ms: int | None = None,
    global_economic_sequence: int = 0,
    cash_per_share: float = 0.0,
    share_ratio: float = 1.0,
    successor_security_id: str | None = None,
    successor_ratio: float = 0.0,
    affected_fraction: float = 1.0,
) -> dict[str, object]:
    return _record(
        "corporate-actions",
        event_id=event_id,
        security_id=security_id,
        kind=kind.value,
        effective_at_ms=effective_at_ms,
        available_at_ms=(
            effective_at_ms if available_at_ms is None else available_at_ms
        ),
        cash_per_share=cash_per_share,
        share_ratio=share_ratio,
        successor_security_id=successor_security_id,
        successor_ratio=successor_ratio,
        affected_fraction=affected_fraction,
        global_economic_sequence=global_economic_sequence,
        effective_timestamp_contract=expected_effective_timestamp_contract_v4(
            source_kind="corporate-actions", event_kind=kind.value
        ),
        upstream_source_receipt_sha256=semantic_sha256((event_id, "upstream")),
    )


def _cash_return(
    event_id: str,
    effective_at_ms: int,
    *,
    one_step_return: float,
    global_economic_sequence: int = 0,
) -> dict[str, object]:
    return _record(
        "cash-returns",
        event_id=event_id,
        effective_at_ms=effective_at_ms,
        available_at_ms=effective_at_ms,
        one_step_return=one_step_return,
        global_economic_sequence=global_economic_sequence,
        effective_timestamp_contract=expected_effective_timestamp_contract_v4(
            source_kind="cash-returns", event_kind="cash-return"
        ),
        upstream_source_receipt_sha256=semantic_sha256((event_id, "upstream")),
    )


def _publish_source(
    root: Path,
    *,
    source_kind: str,
    identity_receipt: str,
    records: list[dict[str, object]],
    suffix: str = "",
) -> LoadedMassiveSourceObject:
    relative = f"economic/{source_kind}{suffix}.json"
    payload = {
        "schema": MASSIVE_ECONOMIC_EVENT_SOURCE_V4_SCHEMA,
        "source_kind": source_kind,
        "identity_authority_receipt_sha256": identity_receipt,
        "records": records,
    }
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ECONOMIC_EVENT_SOURCE_V4_DATASETS[source_kind],
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_ECONOMIC_EVENT_SOURCE_V4_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=DIGEST,
        committed_at_ms=1,
    )
    return load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=1
    )


def _authority(
    tmp_path: Path,
    *,
    identity: PITSecurityUniverseAuthority,
    corporate: list[dict[str, object]],
    cash_returns: list[dict[str, object]] | None = None,
) -> object:
    terminal = _record(
        "terminal-outcomes",
        event_id="FUTURE-TERMINAL",
        security_id="SEC-B",
        kind=TerminalEventKind.WORTHLESS.value,
        effective_at_ms=9_999_999_999_999,
        available_at_ms=9_999_999_999_999,
        cash_per_share=0.0,
        successor_security_id=None,
        successor_ratio=0.0,
        global_economic_sequence=0,
        effective_timestamp_contract=expected_effective_timestamp_contract_v4(
            source_kind="terminal-outcomes",
            event_kind=TerminalEventKind.WORTHLESS.value,
        ),
        upstream_source_receipt_sha256="5" * 64,
    )
    cash = _cash_return("CASH-ZERO", 1, one_step_return=0.0)
    loaded = (
        _publish_source(
            tmp_path,
            source_kind="corporate-actions",
            identity_receipt=identity.receipt_sha256,
            records=corporate,
        ),
        _publish_source(
            tmp_path,
            source_kind="terminal-outcomes",
            identity_receipt=identity.receipt_sha256,
            records=[terminal],
        ),
        _publish_source(
            tmp_path,
            source_kind="cash-returns",
            identity_receipt=identity.receipt_sha256,
            records=[cash, *(cash_returns or [])],
        ),
    )
    return build_massive_economic_event_authority_v4(
        root=tmp_path,
        loaded_sources=loaded,
        identity_authority=identity,
    )


def _run(
    tmp_path: Path,
    corporate: list[dict[str, object]],
    *,
    sessions: MassiveSessionAuthority | None = None,
    identity: PITSecurityUniverseAuthority | None = None,
    cash_returns: list[dict[str, object]] | None = None,
    panel_rows: tuple[MassiveSessionPanelRowV1, ...] | None = None,
) -> tuple[object, tuple[object, ...], tuple[str, ...], tuple[str, ...]]:
    sessions = _sessions() if sessions is None else sessions
    identity = _identity(sessions) if identity is None else identity
    authority = _authority(
        tmp_path,
        identity=identity,
        corporate=corporate,
        cash_returns=cash_returns,
    )
    origin = build_massive_economic_history_origin_v4(
        session_authority=sessions,
        source_session_date=sessions.sessions[6].session_date,
        decision_origin_receipt_sha256="7" * 64,
    )
    rows, admitted, unavailable = build_massive_economic_history_rows_v4(
        panel_rows=_panel(sessions) if panel_rows is None else panel_rows,
        economic_authority=authority,
        identity_authority=identity,
        session_authority=sessions,
        origin=origin,
    )
    return authority, rows, admitted, unavailable


def test_pre_base_actions_are_permanently_ineligible(tmp_path: Path) -> None:
    sessions = _sessions()
    before_panel = sessions.sessions[1].regular_close_ns // 1_000_000 - 1
    known_after_panel_base = sessions.sessions[7].regular_close_ns // 1_000_000
    events = [
        _corporate(
            "OLD-DIVIDEND",
            "SEC-A",
            CorporateActionKind.CASH_DIVIDEND,
            before_panel,
            available_at_ms=known_after_panel_base,
            cash_per_share=10.0,
            global_economic_sequence=0,
        ),
        _corporate(
            "OLD-SPIN",
            "SEC-A",
            CorporateActionKind.SPIN_OFF,
            before_panel,
            successor_security_id="SEC-B",
            successor_ratio=1.0,
            global_economic_sequence=1,
        ),
        _corporate(
            "OLD-CASH-MERGER",
            "SEC-A",
            CorporateActionKind.MERGER_CASH,
            before_panel,
            cash_per_share=500.0,
            global_economic_sequence=2,
        ),
    ]
    _, rows, _, _ = _run(tmp_path, events)
    first_a = next(row for row in rows if row.security_id == "SEC-A")

    assert first_a.economic_value == pytest.approx(100.0)
    assert first_a.position is not None
    assert first_a.position.cash == 0.0
    assert tuple(holding.security_id for holding in first_a.position.holdings) == (
        "SEC-A",
    )
    assert set(first_a.position.excluded_event_ids) >= {
        "OLD-DIVIDEND",
        "OLD-SPIN",
        "OLD-CASH-MERGER",
    }


def test_successor_does_not_receive_pre_acquisition_actions(tmp_path: Path) -> None:
    sessions = _sessions()
    old_b_time = sessions.sessions[3].regular_close_ns // 1_000_000 - 1
    spin_time = sessions.sessions[4].regular_close_ns // 1_000_000 - 1
    known_after_acquisition = sessions.sessions[7].regular_close_ns // 1_000_000
    events = [
        _corporate(
            "B-OLD-SPLIT",
            "SEC-B",
            CorporateActionKind.SPLIT,
            old_b_time,
            share_ratio=2.0,
        ),
        _corporate(
            "B-OLD-DIVIDEND",
            "SEC-B",
            CorporateActionKind.CASH_DIVIDEND,
            old_b_time,
            available_at_ms=known_after_acquisition,
            cash_per_share=5.0,
            global_economic_sequence=1,
        ),
        _corporate(
            "A-SPIN",
            "SEC-A",
            CorporateActionKind.SPIN_OFF,
            spin_time,
            successor_security_id="SEC-B",
            successor_ratio=1.0,
        ),
    ]
    _, rows, _, _ = _run(tmp_path, events)
    source_a = [row for row in rows if row.security_id == "SEC-A"][-1]

    assert source_a.economic_value == pytest.approx(120.0)
    assert source_a.position is not None
    assert source_a.position.cash == 0.0
    assert set(source_a.position.excluded_event_ids) >= {
        "B-OLD-SPLIT",
        "B-OLD-DIVIDEND",
    }


def test_late_available_event_replays_at_effective_time(tmp_path: Path) -> None:
    sessions = _sessions()
    effective = sessions.sessions[3].regular_close_ns // 1_000_000 - 1
    available = sessions.sessions[7].regular_close_ns // 1_000_000
    after_decision = sessions.sessions[8].regular_close_ns // 1_000_000
    events = [
        _corporate(
            "LATE-KNOWN-DIVIDEND",
            "SEC-A",
            CorporateActionKind.CASH_DIVIDEND,
            effective,
            available_at_ms=available,
            cash_per_share=10.0,
        ),
        _corporate(
            "UNAVAILABLE-DIVIDEND",
            "SEC-A",
            CorporateActionKind.CASH_DIVIDEND,
            effective,
            available_at_ms=after_decision,
            cash_per_share=99.0,
            global_economic_sequence=1,
        ),
    ]
    _, rows, admitted, unavailable = _run(tmp_path, events)
    a = {row.source_session_date: row for row in rows if row.security_id == "SEC-A"}

    assert a[sessions.sessions[2].session_date].economic_value == pytest.approx(100.0)
    assert a[sessions.sessions[3].session_date].economic_value == pytest.approx(110.0)
    assert "LATE-KNOWN-DIVIDEND" in admitted
    assert "UNAVAILABLE-DIVIDEND" in unavailable


def test_explicit_global_sequence_controls_same_time_successor_event(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    effective = sessions.sessions[4].regular_close_ns // 1_000_000 - 1
    events = [
        _corporate(
            "A-SPIN-FIRST",
            "SEC-A",
            CorporateActionKind.SPIN_OFF,
            effective,
            global_economic_sequence=0,
            successor_security_id="SEC-B",
            successor_ratio=1.0,
        ),
        _corporate(
            "B-SPLIT-SECOND",
            "SEC-B",
            CorporateActionKind.SPLIT,
            effective,
            global_economic_sequence=1,
            share_ratio=2.0,
        ),
    ]
    _, rows, _, _ = _run(tmp_path, events)
    source_a = [row for row in rows if row.security_id == "SEC-A"][-1]

    assert source_a.economic_value == pytest.approx(140.0)
    assert source_a.position is not None
    b_lot = next(
        holding
        for holding in source_a.position.holdings
        if holding.security_id == "SEC-B"
    )
    assert b_lot.shares == pytest.approx(2.0)
    assert b_lot.acquisition_order == (effective, 0)
    assert b_lot.acquired_event_id == "A-SPIN-FIRST"


def test_duplicate_global_order_fails_closed_within_one_source(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    effective = sessions.sessions[4].regular_close_ns // 1_000_000 - 1
    events = [
        _corporate(
            "A-SPIN-FIRST",
            "SEC-A",
            CorporateActionKind.SPIN_OFF,
            effective,
            global_economic_sequence=0,
            successor_security_id="SEC-B",
            successor_ratio=1.0,
        ),
        _corporate(
            "B-SPLIT-SECOND",
            "SEC-B",
            CorporateActionKind.SPLIT,
            effective,
            global_economic_sequence=0,
            share_ratio=2.0,
        ),
    ]

    with pytest.raises(
        MassiveEconomicEventSourceV4Error,
        match="rows are not ordered and unique",
    ):
        _run(tmp_path, events)


def test_event_id_renaming_does_not_change_economic_value(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    effective = sessions.sessions[4].regular_close_ns // 1_000_000 - 1
    cases = (
        ("names-forward", "A-DIVIDEND", "Z-CASH-RETURN"),
        ("names-reversed", "Z-DIVIDEND", "A-CASH-RETURN"),
    )
    values = []
    for case_name, dividend_id, cash_id in cases:
        case_root = tmp_path / case_name
        case_root.mkdir()
        _, rows, _, _ = _run(
            case_root,
            [
                _corporate(
                    dividend_id,
                    "SEC-A",
                    CorporateActionKind.CASH_DIVIDEND,
                    effective,
                    global_economic_sequence=0,
                    cash_per_share=10.0,
                )
            ],
            sessions=sessions,
            identity=_identity(sessions),
            cash_returns=[
                _cash_return(
                    cash_id,
                    effective,
                    global_economic_sequence=1,
                    one_step_return=0.1,
                )
            ],
        )
        source_a = [row for row in rows if row.security_id == "SEC-A"][-1]
        values.append(source_a.economic_value)

    assert values == pytest.approx((111.0, 111.0))


def test_duplicate_global_order_fails_closed_across_source_roles(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    effective = sessions.sessions[4].regular_close_ns // 1_000_000 - 1

    with pytest.raises(
        MassiveEconomicEventSourceV4Error,
        match="global cash-event order is duplicated",
    ):
        _run(
            tmp_path,
            [
                _corporate(
                    "DIVIDEND",
                    "SEC-A",
                    CorporateActionKind.CASH_DIVIDEND,
                    effective,
                    global_economic_sequence=4,
                    cash_per_share=10.0,
                )
            ],
            sessions=sessions,
            identity=_identity(sessions),
            cash_returns=[
                _cash_return(
                    "CASH-RETURN",
                    effective,
                    global_economic_sequence=4,
                    one_step_return=0.1,
                )
            ],
        )


def test_partial_tender_preserves_multiple_acquisition_vintages(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    events = [
        _corporate(
            "A-SPIN-B",
            "SEC-A",
            CorporateActionKind.SPIN_OFF,
            sessions.sessions[3].regular_close_ns // 1_000_000 - 1,
            successor_security_id="SEC-B",
            successor_ratio=1.0,
        ),
        _corporate(
            "B-SPIN-A",
            "SEC-B",
            CorporateActionKind.SPIN_OFF,
            sessions.sessions[4].regular_close_ns // 1_000_000 - 1,
            successor_security_id="SEC-A",
            successor_ratio=0.5,
        ),
        _corporate(
            "A-PARTIAL-TENDER",
            "SEC-A",
            CorporateActionKind.TENDER_OFFER,
            sessions.sessions[5].regular_close_ns // 1_000_000 - 1,
            cash_per_share=10.0,
            affected_fraction=0.5,
        ),
    ]

    _, rows, _, _ = _run(tmp_path, events)
    source_a = [row for row in rows if row.security_id == "SEC-A"][-1]
    assert source_a.position is not None
    a_lots = tuple(
        holding
        for holding in source_a.position.holdings
        if holding.security_id == "SEC-A"
    )

    assert tuple(lot.shares for lot in a_lots) == pytest.approx((0.5, 0.25))
    assert {lot.acquired_event_id for lot in a_lots} == {
        f"BASE:SEC-A:{sessions.sessions[2].session_date}",
        "B-SPIN-A",
    }
    assert source_a.position.cash == pytest.approx(7.5)
    assert source_a.economic_value == pytest.approx(102.5)


def test_global_economic_sequence_cannot_reach_base_sentinel(tmp_path: Path) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    row = _corporate(
        "INVALID-SEQUENCE",
        "SEC-A",
        CorporateActionKind.CASH_DIVIDEND,
        sessions.sessions[3].regular_close_ns // 1_000_000 - 1,
        global_economic_sequence=MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V4,
        cash_per_share=1.0,
    )
    loaded = _publish_source(
        tmp_path,
        source_kind="corporate-actions",
        identity_receipt=identity.receipt_sha256,
        records=[row],
        suffix="-invalid-sequence",
    )

    with pytest.raises(MassiveEconomicEventSourceV4Error, match="base sentinel"):
        parse_massive_economic_event_source_v4(root=tmp_path, loaded_source=loaded)


def test_kind_specific_effective_timestamp_contract_is_exact(tmp_path: Path) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    row = _corporate(
        "BAD-TIMESTAMP-CONTRACT",
        "SEC-A",
        CorporateActionKind.CASH_DIVIDEND,
        sessions.sessions[3].regular_close_ns // 1_000_000 - 1,
        cash_per_share=1.0,
    )
    row["effective_timestamp_contract"] = "payment-date"
    row.pop("source_row_receipt_sha256")
    row["source_row_receipt_sha256"] = economic_event_source_row_receipt_v4(
        source_kind="corporate-actions", record=row
    )
    loaded = _publish_source(
        tmp_path,
        source_kind="corporate-actions",
        identity_receipt=identity.receipt_sha256,
        records=[row],
        suffix="-bad-timestamp-contract",
    )

    with pytest.raises(
        MassiveEconomicEventSourceV4Error,
        match="was not kind-derived",
    ):
        parse_massive_economic_event_source_v4(root=tmp_path, loaded_source=loaded)


def test_listing_age_uses_full_calendar_before_panel_start(tmp_path: Path) -> None:
    _, rows, _, _ = _run(tmp_path, [])
    first_a = next(row for row in rows if row.security_id == "SEC-A")
    last_a = [row for row in rows if row.security_id == "SEC-A"][-1]

    assert first_a.listing_session_ordinal == 0
    assert first_a.listing_age_sessions == 3
    assert not first_a.listing_age_left_censored
    assert last_a.listing_age_sessions == 7


def test_midnight_listing_timestamp_resolves_by_calendar_date(tmp_path: Path) -> None:
    sessions = _sessions()
    listing_date = datetime.fromisoformat(sessions.sessions[1].session_date).date()
    listing_at_ms = int(
        datetime.combine(listing_date, time(0, 0), tzinfo=UTC).timestamp() * 1000
    )
    identity = _identity(sessions, listing_at_ms=listing_at_ms)

    _, rows, _, _ = _run(
        tmp_path,
        [],
        sessions=sessions,
        identity=identity,
    )
    first_a = next(row for row in rows if row.security_id == "SEC-A")

    assert first_a.listing_calendar_date == listing_date.isoformat()
    assert first_a.listing_session_ordinal == 1
    assert first_a.listing_age_sessions == 2
    assert not first_a.listing_age_left_censored


def test_listing_before_calendar_is_explicitly_left_censored(tmp_path: Path) -> None:
    sessions = _sessions()
    listing_date = date(2020, 1, 2)
    listing_at_ms = int(
        datetime.combine(listing_date, time(0, 0), tzinfo=NY).timestamp() * 1000
    )
    identity = _identity(sessions, listing_at_ms=listing_at_ms)

    _, rows, _, _ = _run(
        tmp_path,
        [],
        sessions=sessions,
        identity=identity,
    )
    first_a = next(row for row in rows if row.security_id == "SEC-A")

    assert first_a.listing_calendar_date == listing_date.isoformat()
    assert first_a.listing_session_ordinal == 0
    assert first_a.listing_age_sessions == 3
    assert first_a.listing_age_left_censored


def test_unlisted_successor_close_cannot_mark_an_ordinary_holding(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    panel_rows = list(_panel(sessions))
    source_date = sessions.sessions[6].session_date
    target_index = next(
        index
        for index, row in enumerate(panel_rows)
        if row.source_session_date == source_date and row.security_id == "SEC-B"
    )
    stale_mark = replace(
        panel_rows[target_index],
        listed=False,
        tradable=False,
        receipt_sha256="0" * 64,
    )
    stale_mark = replace(
        stale_mark,
        receipt_sha256=semantic_sha256(stale_mark.unsigned()),
    )
    stale_mark.validate()
    panel_rows[target_index] = stale_mark
    spin_time = sessions.sessions[4].regular_close_ns // 1_000_000 - 1

    _, rows, _, _ = _run(
        tmp_path,
        [
            _corporate(
                "A-SPIN-B",
                "SEC-A",
                CorporateActionKind.SPIN_OFF,
                spin_time,
                successor_security_id="SEC-B",
                successor_ratio=1.0,
            )
        ],
        sessions=sessions,
        identity=identity,
        panel_rows=tuple(panel_rows),
    )
    source_a = next(
        row
        for row in rows
        if row.source_session_date == source_date and row.security_id == "SEC-A"
    )

    assert source_a.position is not None
    assert any(holding.security_id == "SEC-B" for holding in source_a.position.holdings)
    assert not source_a.economic_value_valid
    assert source_a.economic_value == 0.0


def test_committed_source_row_cannot_be_changed_under_retained_receipt(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    effective = sessions.sessions[3].regular_close_ns // 1_000_000 - 1
    row = _corporate(
        "DIVIDEND",
        "SEC-A",
        CorporateActionKind.CASH_DIVIDEND,
        effective,
        cash_per_share=1.0,
    )
    row["cash_per_share"] = 999.0
    loaded = _publish_source(
        tmp_path,
        source_kind="corporate-actions",
        identity_receipt=identity.receipt_sha256,
        records=[row],
        suffix="-tampered",
    )

    with pytest.raises(
        MassiveEconomicEventSourceV4Error,
        match="was not derived from committed fields",
    ):
        parse_massive_economic_event_source_v4(root=tmp_path, loaded_source=loaded)


def test_materialized_history_round_trips_and_detects_byte_tampering(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    panel = _panel_artifact(tmp_path, sessions=sessions, identity=identity)
    effective = sessions.sessions[4].regular_close_ns // 1_000_000 - 1
    authority = _authority(
        tmp_path,
        identity=identity,
        corporate=[
            _corporate(
                "ROUNDTRIP-DIVIDEND",
                "SEC-A",
                CorporateActionKind.CASH_DIVIDEND,
                effective,
                cash_per_share=2.0,
            )
        ],
    )
    origin = build_massive_economic_history_origin_v4(
        session_authority=sessions,
        source_session_date=sessions.sessions[6].session_date,
        decision_origin_receipt_sha256="7" * 64,
    )
    output_root = tmp_path / "history-output"
    output_root.mkdir()

    artifact = materialize_massive_economic_history_v4(
        session_panel_root=tmp_path,
        economic_source_root=tmp_path,
        output_root=output_root,
        session_panel=panel,
        economic_authority=authority,
        identity_authority=identity,
        session_authority=sessions,
        origin=origin,
        entitlement_receipt_sha256=DIGEST,
        published_at_ms=2,
    )

    validate_massive_economic_history_v4(root=output_root, artifact=artifact)
    assert artifact.row_count == 10
    assert artifact.valid_row_count == 10
    assert "ROUNDTRIP-DIVIDEND" in tuple(
        event_id
        for row in artifact.rows
        if row.position is not None
        for event_id in row.position.applied_event_ids
    )

    payload_path = output_root / artifact.loaded_source.payload_relative_path
    payload_path.chmod(0o644)
    payload_path.write_bytes(payload_path.read_bytes() + b"\n")

    with pytest.raises(MassiveSourceObjectError, match="was replaced"):
        validate_massive_economic_history_v4(root=output_root, artifact=artifact)
