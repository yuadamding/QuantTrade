from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import pytest

from rl_quant.alpha.contracts import CorporateActionKind, CorporateActionRecord
from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
    build_massive_session_authority,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.features.massive_economic_return_index_v1 import (
    MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_bounded_artifacts_v1 import (
    materialize_massive_profitability_bounded_artifact_v1,
    parse_massive_profitability_bounded_artifact_v1,
)
from rl_quant.features.massive_profitability_data_gate_v1 import (
    MASSIVE_PROFITABILITY_DATA_GATE_V1_LOCKBOX_ACCESS_AUTHORIZED,
    MASSIVE_PROFITABILITY_DATA_GATE_V1_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_PROFITABILITY_DATA_GATE_V1_PROFITABILITY_REPORTING_AUTHORIZED,
    MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA,
    MASSIVE_PROFITABILITY_DATA_GATE_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_DATA_GATE_V1_SPEC_SHA256,
    MassiveProfitabilityDataGateV1,
    MassiveProfitabilityDateSupportGateV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_feature_accounting_v1 import (
    MassiveProfitabilityFeatureAccountingV1Error,
    build_massive_profitability_feature_accounting_v1,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
    MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
    TAPE_MIN_V2_FIELDS,
    MassiveProfitabilityOriginFeaturesV2Error,
    MassiveProfitabilityTapePopulationRowV2,
    build_massive_profitability_origin_features_v2,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MASSIVE_PROFITABILITY_DECISION_ORIGIN_V1_SCHEMA,
    MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256,
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MassiveProfitabilityQualifyingFillTradeV1,
    build_massive_profitability_fill_window_v1,
    build_massive_profitability_target_accounting_v1,
    replay_massive_profitability_economic_inputs_v1,
)
from rl_quant.features.massive_profitability_targets_v1 import (
    MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
    build_massive_profitability_targets_v1,
)
from rl_quant.features.massive_session_panel_v1 import (
    MASSIVE_SESSION_PANEL_V1_SCHEMA,
    MassiveSessionPanelRowV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

_EASTERN = timezone(timedelta(hours=-4))
_DIGEST = "a" * 64


def _ms(day: str, value: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp()
        * 1_000
    )


def _session_dates(count: int = 130) -> tuple[str, ...]:
    values: list[str] = []
    current = date(2020, 1, 2)
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


def _sessions() -> MassiveSessionAuthority:
    source = semantic_sha256("bounded-feature-calendar")
    rows = tuple(
        MassiveExchangeSession(
            session_date=day,
            exchange="XNYS",
            regular_open_ns=_ms(day, time(9, 30)) * 1_000_000,
            regular_close_ns=_ms(day, time(16, 0)) * 1_000_000,
            scheduled_five_minute_intervals=78,
            special_session_reason=None,
            calendar_source_receipt_sha256=source,
        )
        for day in _session_dates()
    )
    return build_massive_session_authority(rows, calendar_source_receipt_sha256=source)


def _identity(sessions: MassiveSessionAuthority) -> PITSecurityUniverseAuthority:
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    listing = sessions.sessions[0].regular_open_ns // 1_000_000
    effective_index = 64
    effective = sessions.sessions[effective_index].regular_open_ns // 1_000_000
    rank = UniverseRankInputRecord(
        security_id="SEC-A",
        effective_at_ms=effective,
        effective_session_index=effective_index,
        available_at_ms=sessions.sessions[63].regular_close_ns // 1_000_000,
        observation_start_ms=sessions.sessions[1].regular_close_ns // 1_000_000,
        observation_end_ms=sessions.sessions[63].regular_close_ns // 1_000_000,
        observation_start_session_index=1,
        observation_end_session_index=63,
        observed_session_count=63,
        average_dollar_volume=10_000_000.0,
        close_price=100.0,
        source_receipt_sha256=semantic_sha256("rank-a"),
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=(
            SourcedSecurityMasterRecord(
                security_id="SEC-A",
                issuer_id="ISS-A",
                primary_exchange="XNYS",
                share_class="COMMON",
                security_type="common-stock",
                listing_at_ms=listing,
                delisting_at_ms=None,
                successor_security_id=None,
                corporate_action_chain_id="CHAIN-A",
                identity_source_receipt_sha256=semantic_sha256("master-a"),
            ),
        ),
        ticker_history=(
            SourcedTickerHistoryRecord(
                security_id="SEC-A",
                ticker="AAA",
                valid_from_ms=listing,
                valid_to_ms=None,
                available_at_ms=listing,
                primary_exchange="XNYS",
                source_receipt_sha256=semantic_sha256("ticker-a"),
            ),
        ),
        listing_events=(
            ListingEventRecord(
                event_id="LIST-A",
                security_id="SEC-A",
                effective_at_ms=listing,
                available_at_ms=listing,
                primary_exchange="XNYS",
                ticker="AAA",
                source_receipt_sha256=semantic_sha256("listing-a"),
            ),
        ),
        delisting_events=(),
        rank_inputs=(rank,),
    )


def _origin(sessions: MassiveSessionAuthority) -> MassiveProfitabilityDecisionOriginV1:
    source = sessions.sessions[63]
    decision = sessions.sessions[65]
    decision_at = _ms(decision.session_date, time(12, 30))
    semantic = {
        "source_session_date": source.session_date,
        "decision_session_date": decision.session_date,
        "decision_at_ms": decision_at,
        "fill_start_at_ms": _ms(decision.session_date, time(15, 50)),
        "fill_end_at_ms": _ms(decision.session_date, time(16, 0)),
        "feature_cutoff_at_ms": source.regular_close_ns // 1_000_000,
        "source_staleness_sessions": 2,
        "vendor_last_modified_at_ms": decision_at - 19 * 3_600_000,
        "vendor_lead_time_ms": 19 * 3_600_000,
        "source_object_key": canonical_massive_trade_object_key(source.session_date),
        "source_evidence_receipt_sha256": semantic_sha256("source-row"),
        "source_evidence_artifact_semantic_receipt_sha256": semantic_sha256(
            "source-artifact"
        ),
        "scheduled_rebalance_session_date": sessions.sessions[64].session_date,
        "membership_age_sessions": 1,
        "membership_effective_at_ms": (
            sessions.sessions[64].regular_open_ns // 1_000_000
        ),
        "decision_member_security_ids": ("SEC-A",),
        "decision_member_universe_ranks": (1,),
        "membership_group_semantic_receipt_sha256": semantic_sha256("members"),
        "membership_schedule_semantic_receipt_sha256": semantic_sha256("schedule"),
        "origin_available_identity_receipt_sha256": semantic_sha256("identity"),
        "session_authority_receipt_sha256": sessions.receipt_sha256,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "origin_spec_receipt_sha256": MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256,
        "schema": MASSIVE_PROFITABILITY_DECISION_ORIGIN_V1_SCHEMA,
    }
    receipt = semantic_sha256(semantic)
    identity_audit = semantic_sha256("identity-audit")
    result = MassiveProfitabilityDecisionOriginV1(
        **semantic,
        receipt_sha256=receipt,
        identity_authority_audit_receipt_sha256=identity_audit,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": receipt,
                "identity_authority_audit_receipt_sha256": identity_audit,
            }
        ),
    )
    result.validate()
    return result


def _panel_row(
    session: MassiveExchangeSession,
    index: int,
    *,
    observed: bool = True,
) -> MassiveSessionPanelRowV1:
    close = 100.0 + index
    dollar = 1_000_000.0 + index
    bars = {
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "share_volume": dollar / close,
        "dollar_volume": dollar,
        "high_low_range": 0.02,
        "close_location": 0.5,
    }
    tape = {
        "log_trade_count": math.log1p(100),
        "median_trade_size": 100.0,
        "p90_trade_size": 500.0,
        "large_trade_dollar_fraction": 0.2,
        "quote_free_signed_dollar_flow_proxy": 900_000.0,
        "absolute_signed_flow_imbalance": 0.9,
        "price_response_per_signed_dollar": 0.0,
        "trf_off_exchange_dollar_fraction": 0.3,
        "venue_entropy": 1.2,
        "largest_venue_share": 0.4,
        "tape_a_fraction": 0.5,
        "tape_b_fraction": 0.3,
        "tape_c_fraction": 0.2,
        "special_condition_fraction": 0.0,
        "correction_replacement_fraction": 0.0,
    }
    bars_values = tuple(bars[name] for name in MASSIVE_DAILY_BARS_V0_FIELDS)
    tape_values = tuple(tape[name] for name in MASSIVE_DAILY_TAPE_V0_FIELDS)
    if not observed:
        bars_values = (0.0,) * len(bars_values)
        tape_values = (0.0,) * len(tape_values)
    body = {
        "source_session_index": index,
        "source_session_date": session.session_date,
        "regular_open_ns": session.regular_open_ns,
        "regular_close_ns": session.regular_close_ns,
        "security_id": "SEC-A",
        "pit_member": True,
        "listed": True,
        "tradable": observed,
        "observed_regular_trade": observed,
        "halt_or_no_print": not observed,
        "bars_values": bars_values,
        "bars_valid": (observed,) * len(bars_values),
        "tape_values": tape_values,
        "tape_valid": (observed,) * len(tape_values),
        "event_timeline_count": 100 if observed else 0,
        "replacement_event_count": 2 if observed else 0,
        "cancellation_event_count": 1 if observed else 0,
        "late_report_event_count": 3 if observed else 0,
        "daily_bars_row_receipt_sha256": _DIGEST if observed else None,
        "daily_tape_row_receipt_sha256": "b" * 64 if observed else None,
        "event_counts_receipt_sha256": "c" * 64 if observed else None,
    }
    provisional = MassiveSessionPanelRowV1(**body, receipt_sha256="0" * 64)
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _feature_fixture() -> tuple[
    MassiveSessionAuthority,
    MassiveProfitabilityDecisionOriginV1,
    PITSecurityUniverseAuthority,
    tuple[MassiveSessionPanelRowV1, ...],
    object,
]:
    sessions = _sessions()
    origin = _origin(sessions)
    identity = _identity(sessions)
    selected = sessions.sessions[:64]
    panels = tuple(_panel_row(session, index) for index, session in enumerate(selected))
    values = {
        (session.session_date, "SEC-A"): 100.0 * 1.001**index
        for index, session in enumerate(selected)
    }
    marks = {key: semantic_sha256(key) for key in values}
    accounting = build_massive_profitability_feature_accounting_v1(
        origin=origin,
        session_authority=sessions,
        economic_values=values,
        mark_receipts=marks,
        economic_coverage_semantic_receipt_sha256=semantic_sha256("economic"),
        economic_coverage_audit_receipt_sha256=semantic_sha256("economic-audit"),
    )
    return sessions, origin, identity, panels, accounting


def _field(row: object, fields: tuple[str, ...], name: str) -> tuple[float, bool]:
    values = getattr(
        row, "bars_values" if fields is BARS_MIN_V2_FIELDS else "tape_values"
    )
    valid = getattr(row, "bars_valid" if fields is BARS_MIN_V2_FIELDS else "tape_valid")
    index = fields.index(name)
    return values[index], valid[index]


def test_bounded_feature_cross_section_uses_same_population_tape_semantics() -> None:
    sessions, origin, identity, panels, accounting = _feature_fixture()
    population = MassiveProfitabilityTapePopulationRowV2.build(
        source_session_date=origin.source_session_date,
        security_id="SEC-A",
        signed_dollar_flow=50.0,
        dollar_volume=100.0,
        regular_session_event_count=10,
        replacement_event_count=2,
        cancellation_event_count=1,
        late_report_event_count=3,
        population_receipt_sha256=semantic_sha256("same-population"),
    )
    artifact = build_massive_profitability_origin_features_v2(
        origin=origin,
        session_authority=sessions,
        identity_authority=identity,
        panel_rows=panels,
        session_panel_receipt_sha256=semantic_sha256("panel"),
        feature_accounting=accounting,
        tape_population_rows=(population,),
    )
    row = artifact.rows[0]
    assert len(artifact.input_session_dates) == 64
    assert artifact.input_session_dates[-1] == origin.source_session_date
    assert _field(row, TAPE_MIN_V2_FIELDS, "signed_dollar_flow_fraction") == (
        0.5,
        True,
    )
    assert _field(row, TAPE_MIN_V2_FIELDS, "replacement_event_fraction") == (
        0.2,
        True,
    )
    assert _field(row, BARS_MIN_V2_FIELDS, "listing_age_left_censored") == (
        0.0,
        True,
    )
    rebuilt = build_massive_profitability_origin_features_v2(
        origin=origin,
        session_authority=sessions,
        identity_authority=identity,
        panel_rows=panels,
        session_panel_receipt_sha256=semantic_sha256("panel"),
        feature_accounting=accounting,
        tape_population_rows=(population,),
    )
    assert rebuilt.semantic_receipt_sha256 == artifact.semantic_receipt_sha256
    assert artifact.tape_population_data_qualified is False
    assert artifact.feature_accounting_data_qualified is False
    assert artifact.predictive_training_authorized is False


def test_missing_source_observation_remains_zero_with_false_masks() -> None:
    sessions, origin, identity, panels, accounting = _feature_fixture()
    panels = panels[:-1] + (_panel_row(sessions.sessions[63], 63, observed=False),)
    artifact = build_massive_profitability_origin_features_v2(
        origin=origin,
        session_authority=sessions,
        identity_authority=identity,
        panel_rows=panels,
        session_panel_receipt_sha256=semantic_sha256("panel-missing"),
        feature_accounting=accounting,
        tape_population_rows=(),
    )
    row = artifact.rows[0]
    assert _field(row, BARS_MIN_V2_FIELDS, "high_low_range") == (0.0, False)
    assert _field(row, TAPE_MIN_V2_FIELDS, "log_trade_count") == (0.0, False)
    assert _field(row, TAPE_MIN_V2_FIELDS, "signed_dollar_flow_fraction") == (
        0.0,
        False,
    )


def test_feature_accounting_rejects_target_period_events() -> None:
    sessions, origin, _, _, _ = _feature_fixture()
    values = {
        (session.session_date, "SEC-A"): 100.0 for session in sessions.sessions[:64]
    }
    marks = {key: semantic_sha256(key) for key in values}
    from rl_quant.features.massive_economic_coverage_v8 import (
        MassiveNativeEconomicObservationV8,
    )

    body = {
        "surface_id": "massive-dividends-v1",
        "provider_event_key": "DIV-FUTURE",
        "logical_event_key": semantic_sha256("DIV-FUTURE"),
        "security_id": "SEC-A",
        "ticker": "AAA",
        "kind": "cash-dividend",
        "classification": "recurring",
        "effective_at_ms": origin.feature_cutoff_at_ms + 1,
        "research_captured_at_ms": origin.feature_cutoff_at_ms + 2,
        "accounting_lane": "finalized-accounting-research",
        "predictive_feature_eligible": False,
        "currency": "USD",
        "cash_per_share": 1.0,
        "split_adjusted_cash_per_share": 1.0,
        "share_ratio": 1.0,
        "historical_adjustment_factor": 1.0,
        "raw_provider_request_id": "REQ",
        "raw_provider_row_locator": "results=0",
        "raw_provider_row_sha256": semantic_sha256("raw"),
        "identity_mapping_receipt_sha256": semantic_sha256("map"),
    }
    event = MassiveNativeEconomicObservationV8(
        **body, receipt_sha256=semantic_sha256(body)
    )
    event.validate()
    with pytest.raises(
        MassiveProfitabilityFeatureAccountingV1Error,
        match="target-period event",
    ):
        build_massive_profitability_feature_accounting_v1(
            origin=origin,
            session_authority=sessions,
            economic_values=values,
            mark_receipts=marks,
            selected_events=(event,),
            economic_coverage_semantic_receipt_sha256=semantic_sha256("economic"),
            economic_coverage_audit_receipt_sha256=semantic_sha256("audit"),
        )


def _fill(
    *,
    session_date: str,
    security_ids: tuple[str, ...],
    prices: dict[str, float],
) -> object:
    start = _ms(session_date, time(15, 50))
    end = _ms(session_date, time(16, 0))
    trades = tuple(
        MassiveProfitabilityQualifyingFillTradeV1(
            security_id=security_id,
            participant_at_ms=_ms(session_date, time(15, 55)),
            price=price,
            size=100.0,
            terminal_active=True,
            volume_forming=True,
            source_trade_receipt_sha256=semantic_sha256((session_date, security_id)),
        )
        for security_id, price in prices.items()
    )
    return build_massive_profitability_fill_window_v1(
        session_date=session_date,
        fill_start_at_ms=start,
        fill_end_at_ms=end,
        security_ids=security_ids,
        trades=trades,
        source_partition_receipts=(semantic_sha256((session_date, "partition")),),
    )


def test_fill_to_fill_targets_carry_conservative_total_loss() -> None:
    sessions = _sessions()
    origin = _origin(sessions)
    security_ids = ("SEC-A",)
    decision_index = 65
    entry = _fill(
        session_date=origin.decision_session_date,
        security_ids=security_ids,
        prices={"SEC-A": 100.0},
    )
    exits = {
        offset: _fill(
            session_date=sessions.sessions[decision_index + offset].session_date,
            security_ids=security_ids,
            prices={"SEC-A": 100.0 + offset},
        )
        for offset in (1, 5, 21, 63)
    }
    values = {("SEC-A", offset): 100.0 + offset for offset in range(64)}
    kinds = {key: "market" for key in values}
    marks = {key: semantic_sha256(key) for key in values}
    accounting = build_massive_profitability_target_accounting_v1(
        origin=origin,
        session_authority=sessions,
        entry_fill=entry,
        exit_fills_by_offset=exits,
        economic_values=values,
        mark_kinds=kinds,
        mark_receipts=marks,
        unresolved_delisting_offsets={"SEC-A": 3},
        terminal_inventory_sha256=semantic_sha256("terminal"),
        economic_coverage_semantic_receipt_sha256=semantic_sha256("economic"),
        economic_coverage_audit_receipt_sha256=semantic_sha256("economic-audit"),
    )
    targets = build_massive_profitability_targets_v1(accounting=accounting)
    row = targets.rows[0]
    assert row.simple_returns[0] == pytest.approx(0.01)
    assert row.simple_returns[1:] == (-1.0, -1.0, -1.0)
    assert row.terminal_zero_value == (False, True, True, True)
    assert row.conservative_total_loss_fallback is True
    assert accounting.fill_sources_qualified is False
    assert targets.fill_sources_qualified is False
    assert targets.economic_values_data_qualified is False
    rebuilt = build_massive_profitability_targets_v1(accounting=accounting)
    assert rebuilt.semantic_receipt_sha256 == targets.semantic_receipt_sha256
    assert targets.predictive_training_authorized is False


def test_split_adjusted_position_value_prevents_false_target_loss() -> None:
    sessions = _sessions()
    origin = _origin(sessions)
    decision_index = 65
    session_times = tuple(
        row.regular_close_ns // 1_000_000
        for row in sessions.sessions[decision_index : decision_index + 64]
    )
    split = CorporateActionRecord(
        event_id="SPLIT-A",
        security_id="SEC-A",
        kind=CorporateActionKind.SPLIT,
        effective_at_ms=session_times[2],
        available_at_ms=session_times[2],
        share_ratio=2.0,
    )
    marks = {(offset, "SEC-A"): 100.0 if offset < 2 else 50.0 for offset in range(64)}
    mark_receipts = {key: semantic_sha256(key) for key in marks}
    values, kinds, receipts, terminal_offsets = (
        replay_massive_profitability_economic_inputs_v1(
            origin_security_ids=("SEC-A",),
            economic_at_ms=session_times,
            marks=marks,
            mark_receipts=mark_receipts,
            events=(split,),
        )
    )
    assert values[("SEC-A", 1)] == 100.0
    assert values[("SEC-A", 2)] == 100.0
    assert terminal_offsets == {}
    entry = _fill(
        session_date=origin.decision_session_date,
        security_ids=("SEC-A",),
        prices={"SEC-A": 100.0},
    )
    exits = {
        offset: _fill(
            session_date=sessions.sessions[decision_index + offset].session_date,
            security_ids=("SEC-A",),
            prices={"SEC-A": 100.0 if offset < 2 else 50.0},
        )
        for offset in (1, 5, 21, 63)
    }
    accounting = build_massive_profitability_target_accounting_v1(
        origin=origin,
        session_authority=sessions,
        entry_fill=entry,
        exit_fills_by_offset=exits,
        economic_values=values,
        mark_kinds=kinds,
        mark_receipts=receipts,
        terminal_offsets=terminal_offsets,
        event_receipts=(semantic_sha256("SPLIT-A"),),
        terminal_inventory_sha256=semantic_sha256("no-terminal"),
        economic_coverage_semantic_receipt_sha256=semantic_sha256("economic"),
        economic_coverage_audit_receipt_sha256=semantic_sha256("economic-audit"),
    )
    targets = build_massive_profitability_targets_v1(accounting=accounting)
    assert targets.rows[0].simple_returns == (0.0, 0.0, 0.0, 0.0)
    assert targets.rows[0].valid == (True, True, True, True)


def test_legacy_feature_generation_is_rejected() -> None:
    sessions, origin, identity, panels, accounting = _feature_fixture()
    with pytest.raises(
        (MassiveProfitabilityOriginFeaturesV2Error, ValueError),
        match="legacy profitability generation",
    ):
        build_massive_profitability_origin_features_v2(
            origin=origin,
            session_authority=sessions,
            identity_authority=identity,
            panel_rows=panels,
            session_panel_receipt_sha256=semantic_sha256("panel"),
            feature_accounting=accounting,
            tape_population_rows=(),
            input_schemas=(
                MASSIVE_SESSION_PANEL_V1_SCHEMA,
                MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA,
            ),
        )


def test_bounded_component_materializes_and_reloads_exact_bytes(tmp_path) -> None:
    _, _, _, _, accounting = _feature_fixture()
    artifact = materialize_massive_profitability_bounded_artifact_v1(
        root=tmp_path,
        component=accounting,
        component_semantic_receipt_sha256=accounting.semantic_receipt_sha256,
        artifact_id="feature-accounting-canary",
        committed_at_ms=1_700_000_000_000,
        entitlement_receipt_sha256=semantic_sha256("entitlement"),
    )
    reloaded = parse_massive_profitability_bounded_artifact_v1(
        root=tmp_path, loaded_source=artifact.loaded_source
    )
    assert reloaded.receipt_sha256 == artifact.receipt_sha256
    assert (
        reloaded.component_semantic_receipt_sha256 == accounting.semantic_receipt_sha256
    )
    assert reloaded.runtime_authorizing is False


def test_data_gate_can_pass_data_but_never_authorizes_training_or_lockbox() -> None:
    support_body = {
        "decision_session_date": "2020-04-02",
        "decision_member_count": 500,
        "required_common_valid_count": 400,
        "feature_row_count": 500,
        "target_common_valid_count": 400,
        "common_security_inventory_sha256": semantic_sha256("common"),
        "passed": True,
    }
    support = MassiveProfitabilityDateSupportGateV1(
        **support_body, receipt_sha256=semantic_sha256(support_body)
    )
    component = {
        "source_transport_qualified": True,
        "rank_bar_data_qualified": True,
        "origin_plan_v2_only": True,
        "exact_rank_window_complete": True,
        "exact_feature_cutoff_complete": True,
        "exact_source_staleness_complete": True,
        "exact_64_session_rectangles_complete": True,
        "tape_population_data_qualified": True,
        "fill_source_complete": True,
        "economic_accounting_data_qualified": True,
        "target_path_complete": True,
        "terminal_accounting_complete": True,
        "common_model_support_complete": True,
        "reproducible_materialization_complete": True,
        "future_mutation_invariance_complete": True,
        "legacy_generations_rejected": True,
    }
    semantic = {
        "coverage_semantic_receipt_sha256": semantic_sha256("coverage"),
        "candidate_session_dates": ("2020-04-02",),
        "feature_receipts": (semantic_sha256("features"),),
        "target_receipts": (semantic_sha256("targets"),),
        "date_support_gates": (
            support_body | {"receipt_sha256": support.receipt_sha256},
        ),
        "input_schemas": tuple(
            sorted(
                (
                    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
                    MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
                    MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
                )
            )
        ),
        **component,
        "data_gate_passed": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_DATA_GATE_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_DATA_GATE_V1_SOURCE_SHA256,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "schema": MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA,
    }
    receipt = semantic_sha256(semantic)
    gate = MassiveProfitabilityDataGateV1(
        coverage_semantic_receipt_sha256=semantic["coverage_semantic_receipt_sha256"],
        candidate_session_dates=("2020-04-02",),
        feature_receipts=semantic["feature_receipts"],
        target_receipts=semantic["target_receipts"],
        date_support_gates=(support,),
        input_schemas=semantic["input_schemas"],
        **component,
        data_gate_passed=True,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_DATA_GATE_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_DATA_GATE_V1_SOURCE_SHA256,
        semantic_receipt_sha256=receipt,
        audit_receipt_sha256=semantic_sha256("gate-audit"),
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    gate.validate()
    assert gate.data_gate_passed is True
    assert MASSIVE_PROFITABILITY_DATA_GATE_V1_PREDICTIVE_TRAINING_AUTHORIZED is False
    assert (
        MASSIVE_PROFITABILITY_DATA_GATE_V1_PROFITABILITY_REPORTING_AUTHORIZED is False
    )
    assert MASSIVE_PROFITABILITY_DATA_GATE_V1_LOCKBOX_ACCESS_AUTHORIZED is False
    with pytest.raises(ValueError, match="legacy"):
        replace(
            gate,
            input_schemas=(MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA,),
        ).validate()
