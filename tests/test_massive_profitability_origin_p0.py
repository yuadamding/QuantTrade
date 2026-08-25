from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.alpha.pit_universe import (
    DelistingEventRecord,
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
    MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
    MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
    canonical_massive_trade_object_key,
    parse_massive_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MASSIVE_FLAT_TRADES_DATASET_ID,
)
from rl_quant.features.massive_profitability_origin_p0 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SESSIONS,
    MASSIVE_PROFITABILITY_FEATURE_LOOKBACK_SESSIONS,
    MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS,
    MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS,
    MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS,
    MASSIVE_PROFITABILITY_P0_LOCKBOX_ACCESS_AUTHORIZED,
    MASSIVE_PROFITABILITY_P0_PANEL_MATERIALIZATION_AUTHORIZED,
    MASSIVE_PROFITABILITY_P0_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_PROFITABILITY_P0_PROFITABILITY_REPORTING_AUTHORIZED,
    MassiveProfitabilityOriginP0Error,
    MassiveProfitabilitySourceEvidenceP0,
    build_massive_economic_coverage_scope_from_profitability_experiment_v8,
    build_massive_profitability_decision_origin_plan_p0,
    build_massive_profitability_experiment_coverage_v1,
    build_massive_profitability_security_support_v1,
    build_massive_profitability_source_evidence_p0,
    validate_massive_profitability_experiment_coverage_v1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

_EASTERN = ZoneInfo("America/New_York")
_DIGEST = "a" * 64
_FIRST_CANDIDATE_INDEX = 65
_SESSION_COUNT = (
    _FIRST_CANDIDATE_INDEX
    + MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS
    + MASSIVE_PROFITABILITY_FEATURE_LOOKBACK_SESSIONS
)


def _ms(day: str, value: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp()
        * 1_000
    )


def _session_dates(count: int) -> tuple[str, ...]:
    output: list[str] = []
    current = date(2015, 1, 2)
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(output)


def _sessions(
    count: int = _SESSION_COUNT, *, early_close_index: int | None = None
) -> MassiveSessionAuthority:
    calendar_receipt = semantic_sha256(("p0-calendar", count, early_close_index))
    rows = []
    for index, session_date in enumerate(_session_dates(count)):
        close = time(13, 0) if index == early_close_index else time(16, 0)
        open_ms = _ms(session_date, time(9, 30))
        close_ms = _ms(session_date, close)
        rows.append(
            MassiveExchangeSession(
                session_date=session_date,
                exchange="XNYS",
                regular_open_ns=open_ms * 1_000_000,
                regular_close_ns=close_ms * 1_000_000,
                scheduled_five_minute_intervals=(close_ms - open_ms) // 300_000,
                special_session_reason=(
                    "scheduled-early-close" if index == early_close_index else None
                ),
                calendar_source_receipt_sha256=calendar_receipt,
            )
        )
    return build_massive_session_authority(
        tuple(rows), calendar_source_receipt_sha256=calendar_receipt
    )


def _identity(sessions: MassiveSessionAuthority) -> PITSecurityUniverseAuthority:
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    start_ms = sessions.sessions[0].regular_open_ns // 1_000_000
    successor_ms = sessions.sessions[-42].regular_open_ns // 1_000_000
    effective_index = _FIRST_CANDIDATE_INDEX - 1
    effective_ms = sessions.sessions[effective_index].regular_open_ns // 1_000_000
    observation_start_index = effective_index - rule.ranking_lookback_sessions
    observation_end_index = effective_index - rule.ranking_lag_sessions
    observation_start_ms = (
        sessions.sessions[observation_start_index].regular_close_ns // 1_000_000
    )
    observation_end_ms = (
        sessions.sessions[observation_end_index].regular_close_ns // 1_000_000
    )
    masters = (
        SourcedSecurityMasterRecord(
            security_id="SEC-A",
            issuer_id="ISS-A",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=start_ms,
            delisting_at_ms=successor_ms,
            successor_security_id="SEC-C",
            corporate_action_chain_id="CHAIN-A",
            identity_source_receipt_sha256=semantic_sha256("master-a"),
        ),
        SourcedSecurityMasterRecord(
            security_id="SEC-B",
            issuer_id="ISS-B",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=start_ms,
            delisting_at_ms=None,
            successor_security_id=None,
            corporate_action_chain_id="CHAIN-B",
            identity_source_receipt_sha256=semantic_sha256("master-b"),
        ),
        SourcedSecurityMasterRecord(
            security_id="SEC-C",
            issuer_id="ISS-A",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=successor_ms,
            delisting_at_ms=None,
            successor_security_id=None,
            corporate_action_chain_id="CHAIN-A",
            identity_source_receipt_sha256=semantic_sha256("master-c"),
        ),
    )
    tickers = (
        SourcedTickerHistoryRecord(
            security_id="SEC-A",
            ticker="AAA",
            valid_from_ms=start_ms,
            valid_to_ms=successor_ms,
            available_at_ms=start_ms,
            primary_exchange="XNYS",
            source_receipt_sha256=semantic_sha256("ticker-a"),
        ),
        SourcedTickerHistoryRecord(
            security_id="SEC-B",
            ticker="BBB",
            valid_from_ms=start_ms,
            valid_to_ms=None,
            available_at_ms=start_ms,
            primary_exchange="XNYS",
            source_receipt_sha256=semantic_sha256("ticker-b"),
        ),
        SourcedTickerHistoryRecord(
            security_id="SEC-C",
            ticker="AAC",
            valid_from_ms=successor_ms,
            valid_to_ms=None,
            available_at_ms=successor_ms,
            primary_exchange="XNYS",
            source_receipt_sha256=semantic_sha256("ticker-c"),
        ),
    )
    listings = tuple(
        ListingEventRecord(
            event_id=f"LIST-{security_id}",
            security_id=security_id,
            effective_at_ms=listing_at_ms,
            available_at_ms=listing_at_ms,
            primary_exchange="XNYS",
            ticker=ticker,
            source_receipt_sha256=semantic_sha256(("listing", security_id)),
        )
        for security_id, ticker, listing_at_ms in (
            ("SEC-A", "AAA", start_ms),
            ("SEC-B", "BBB", start_ms),
            ("SEC-C", "AAC", successor_ms),
        )
    )
    delistings = (
        DelistingEventRecord(
            event_id="DELIST-A",
            security_id="SEC-A",
            effective_at_ms=successor_ms,
            available_at_ms=successor_ms,
            reason="stock-merger",
            successor_security_id="SEC-C",
            source_receipt_sha256=semantic_sha256("delisting-a"),
        ),
    )
    ranks = tuple(
        UniverseRankInputRecord(
            security_id=security_id,
            effective_at_ms=effective_ms,
            effective_session_index=effective_index,
            available_at_ms=observation_end_ms,
            observation_start_ms=observation_start_ms,
            observation_end_ms=observation_end_ms,
            observation_start_session_index=observation_start_index,
            observation_end_session_index=observation_end_index,
            observed_session_count=rule.ranking_lookback_sessions,
            average_dollar_volume=dollar_volume,
            close_price=100.0,
            source_receipt_sha256=semantic_sha256(("rank", security_id)),
        )
        for security_id, dollar_volume in (
            ("SEC-A", 20_000_000.0),
            ("SEC-B", 10_000_000.0),
        )
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=delistings,
        rank_inputs=ranks,
    )


def _source_evidence(
    *, source: MassiveExchangeSession, decision: MassiveExchangeSession
) -> MassiveProfitabilitySourceEvidenceP0:
    decision_at = _ms(decision.session_date, time(12, 30))
    last_modified = decision_at - 19 * 60 * 60 * 1_000
    provisional = MassiveProfitabilitySourceEvidenceP0(
        source_session_date=source.session_date,
        source_object_key=canonical_massive_trade_object_key(source.session_date),
        vendor_last_modified_at_ms=last_modified,
        listing_observed_at_ms=decision_at + 1,
        research_downloaded_at_ms=decision_at + 2,
        research_verified_at_ms=decision_at + 3,
        content_length=100,
        etag=f"etag-{source.session_date}",
        listing_entry_receipt_sha256=semantic_sha256(
            ("listing-entry", source.session_date)
        ),
        committed_listing_receipt_sha256=semantic_sha256(
            ("committed-listing", source.session_date[:7])
        ),
        loaded_listing_receipt_sha256=semantic_sha256(
            ("loaded-listing", source.session_date[:7])
        ),
        loaded_source_receipt_sha256=semantic_sha256(
            ("loaded-source", source.session_date)
        ),
        source_object_receipt_sha256=semantic_sha256(
            ("source-object", source.session_date)
        ),
        source_commit_receipt_sha256=semantic_sha256(
            ("source-commit", source.session_date)
        ),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _source_inventory(
    sessions: MassiveSessionAuthority,
    *,
    first_candidate_index: int = _FIRST_CANDIDATE_INDEX,
    origin_count: int = MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS,
) -> tuple[MassiveProfitabilitySourceEvidenceP0, ...]:
    return tuple(
        _source_evidence(
            source=sessions.sessions[decision_index - 2],
            decision=sessions.sessions[decision_index],
        )
        for decision_index in range(
            first_candidate_index, first_candidate_index + origin_count
        )
    )


@pytest.fixture(scope="module")
def experiment():
    sessions = _sessions()
    identity = _identity(sessions)
    sources = _source_inventory(sessions)
    first = sessions.sessions[_FIRST_CANDIDATE_INDEX].session_date
    last = sessions.sessions[
        _FIRST_CANDIDATE_INDEX + MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS - 1
    ].session_date
    plan = build_massive_profitability_decision_origin_plan_p0(
        session_authority=sessions,
        identity_authority=identity,
        source_evidences=sources,
        first_candidate_decision_session_date=first,
        last_candidate_decision_session_date=last,
    )
    support = build_massive_profitability_security_support_v1(plan=plan)
    coverage = build_massive_profitability_experiment_coverage_v1(
        plan=plan, support=support
    )
    return sessions, identity, sources, plan, support, coverage


def test_origin_plan_enforces_two_sessions_lead_fill_and_membership(experiment):
    sessions, _, _, plan, _, _ = experiment
    assert len(plan.origins) == MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS
    assert plan.skipped_decisions == ()
    origin = plan.origins[0]
    decision_index = _FIRST_CANDIDATE_INDEX
    assert (
        origin.source_session_date == sessions.sessions[decision_index - 2].session_date
    )
    assert (
        origin.decision_session_date == sessions.sessions[decision_index].session_date
    )
    assert origin.source_staleness_sessions == 2
    assert origin.vendor_lead_time_ms == 19 * 60 * 60 * 1_000
    assert origin.vendor_lead_time_ms >= MASSIVE_PROFITABILITY_MINIMUM_VENDOR_LEAD_MS
    assert origin.feature_maximum_session_date == origin.source_session_date
    assert origin.decision_member_security_ids == ("SEC-A", "SEC-B")
    assert origin.decision_member_universe_ranks == (1, 2)
    assert origin.decision_at_ms == _ms(origin.decision_session_date, time(12, 30))
    assert origin.fill_start_at_ms == _ms(origin.decision_session_date, time(15, 50))
    assert origin.fill_end_at_ms == _ms(origin.decision_session_date, time(16, 0))


def test_origin_plan_exhaustively_emits_vendor_lead_and_missing_source_skips(
    experiment,
):
    sessions, identity, sources, _, _, _ = experiment
    decision_index = _FIRST_CANDIDATE_INDEX
    decision = sessions.sessions[decision_index]
    original = sources[0]
    late = replace(
        original,
        vendor_last_modified_at_ms=(
            _ms(decision.session_date, time(12, 30)) - 17 * 60 * 60 * 1_000
        ),
        receipt_sha256="0" * 64,
    )
    late = replace(late, receipt_sha256=semantic_sha256(late.unsigned()))
    late.validate()
    lead_plan = build_massive_profitability_decision_origin_plan_p0(
        session_authority=sessions,
        identity_authority=identity,
        source_evidences=(late,),
        first_candidate_decision_session_date=decision.session_date,
        last_candidate_decision_session_date=decision.session_date,
    )
    assert lead_plan.origins == ()
    assert lead_plan.skipped_decisions[0].reason == "vendor-lead-below-18-hours"

    missing_plan = build_massive_profitability_decision_origin_plan_p0(
        session_authority=sessions,
        identity_authority=identity,
        source_evidences=(),
        first_candidate_decision_session_date=decision.session_date,
        last_candidate_decision_session_date=decision.session_date,
    )
    assert missing_plan.origins == ()
    assert missing_plan.skipped_decisions[0].reason == "missing-source-object-evidence"


def test_early_close_is_an_explicit_skip(experiment):
    _, _, sources, _, _, _ = experiment
    decision_index = _FIRST_CANDIDATE_INDEX
    early_sessions = _sessions(early_close_index=decision_index)
    early_identity = _identity(early_sessions)
    early_source = _source_evidence(
        source=early_sessions.sessions[decision_index - 2],
        decision=early_sessions.sessions[decision_index],
    )
    plan = build_massive_profitability_decision_origin_plan_p0(
        session_authority=early_sessions,
        identity_authority=early_identity,
        source_evidences=(early_source,),
        first_candidate_decision_session_date=(
            early_sessions.sessions[decision_index].session_date
        ),
        last_candidate_decision_session_date=(
            early_sessions.sessions[decision_index].session_date
        ),
    )
    assert plan.origins == ()
    assert (
        plan.skipped_decisions[0].reason
        == "decision-session-cannot-support-frozen-clock"
    )
    assert sources[0].source_session_date == early_source.source_session_date


def test_security_support_adds_accounting_successors(experiment):
    _, _, _, plan, support, _ = experiment
    assert support.decision_member_security_ids == ("SEC-A", "SEC-B")
    assert support.accounting_chain_security_ids == ("SEC-C",)
    assert support.all_supported_security_ids == ("SEC-A", "SEC-B", "SEC-C")
    assert support.origin_plan_receipt_sha256 == plan.receipt_sha256


def test_experiment_coverage_is_calendar_derived_and_freezes_the_lockbox(experiment):
    sessions, _, _, plan, support, coverage = experiment
    first_origin = plan.origins[0]
    last_origin = plan.origins[-1]
    source_position = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == first_origin.source_session_date
    )
    last_decision_position = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == last_origin.decision_session_date
    )
    assert coverage.earliest_feature_base_session_date == (
        sessions.sessions[
            source_position - MASSIVE_PROFITABILITY_FEATURE_LOOKBACK_SESSIONS
        ].session_date
    )
    assert coverage.latest_h63_endpoint_session_date == (
        sessions.sessions[
            last_decision_position + MASSIVE_PROFITABILITY_FEATURE_LOOKBACK_SESSIONS
        ].session_date
    )
    assert coverage.economic_coverage_start_date == (
        coverage.earliest_feature_base_session_date
    )
    assert coverage.economic_coverage_end_date == (
        coverage.latest_h63_endpoint_session_date
    )
    assert coverage.development_interval.origin_count == (
        MASSIVE_PROFITABILITY_MINIMUM_ELIGIBLE_ORIGINS
        - MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS
    )
    assert (
        coverage.confirmation_interval.origin_count
        == MASSIVE_PROFITABILITY_CONFIRMATION_SESSIONS
    )
    assert (
        coverage.lockbox_interval.origin_count == MASSIVE_PROFITABILITY_LOCKBOX_SESSIONS
    )
    economic_scope = (
        build_massive_economic_coverage_scope_from_profitability_experiment_v8(
            plan=plan, support=support, coverage=coverage
        )
    )
    assert economic_scope.coverage_start_date == coverage.economic_coverage_start_date
    assert economic_scope.coverage_end_date == coverage.economic_coverage_end_date
    validate_massive_profitability_experiment_coverage_v1(
        plan=plan, support=support, coverage=coverage
    )
    with pytest.raises(MassiveProfitabilityOriginP0Error, match="rederived"):
        validate_massive_profitability_experiment_coverage_v1(
            plan=plan,
            support=support,
            coverage=replace(
                coverage,
                latest_h63_endpoint_session_date=(coverage.latest_fill_session_date),
                economic_coverage_end_date=coverage.latest_fill_session_date,
                receipt_sha256=semantic_sha256(
                    replace(
                        coverage,
                        latest_h63_endpoint_session_date=(
                            coverage.latest_fill_session_date
                        ),
                        economic_coverage_end_date=coverage.latest_fill_session_date,
                        receipt_sha256="0" * 64,
                    ).unsigned()
                ),
            ),
        )


def test_experiment_coverage_rejects_an_underfilled_history(experiment):
    sessions, identity, sources, _, _, _ = experiment
    decision = sessions.sessions[_FIRST_CANDIDATE_INDEX]
    plan = build_massive_profitability_decision_origin_plan_p0(
        session_authority=sessions,
        identity_authority=identity,
        source_evidences=(sources[0],),
        first_candidate_decision_session_date=decision.session_date,
        last_candidate_decision_session_date=decision.session_date,
    )
    support = build_massive_profitability_security_support_v1(plan=plan)
    with pytest.raises(MassiveProfitabilityOriginP0Error, match="minimum"):
        build_massive_profitability_experiment_coverage_v1(plan=plan, support=support)


def test_source_evidence_reopens_listing_and_source_bytes(tmp_path: Path):
    source_date = "2024-05-20"
    key = canonical_massive_trade_object_key(source_date)
    source_bytes = b"committed finalized trade source"
    vendor_modified = _ms(source_date, time(18, 0))
    observed = vendor_modified + 1_000
    publish_massive_source_object(
        stream=BytesIO(source_bytes),
        root=tmp_path,
        relative_payload_path=key,
        dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
        source_object_key=key,
        requested_at_ms=observed,
        downloaded_at_ms=observed,
        schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_DIGEST,
        committed_at_ms=observed,
        etag="source-etag",
        request_id="SOURCE-P0",
    )
    loaded_source = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=key, verified_at_ms=observed
    )
    listing_payload = {
        "schema": MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
        "observed_at_ms": observed,
        "entries": [
            {
                "dataset_id": MASSIVE_FLAT_TRADES_DATASET_ID,
                "source_object_key": key,
                "etag": "source-etag",
                "content_length": len(source_bytes),
                "last_modified_at_ms": vendor_modified,
            }
        ],
    }
    listing_path = "profitability-p0/listing.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(listing_payload)),
        root=tmp_path,
        relative_payload_path=listing_path,
        dataset_id=MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
        source_object_key=listing_path,
        requested_at_ms=observed,
        downloaded_at_ms=observed,
        schema_sha256=MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_DIGEST,
        committed_at_ms=observed,
        request_id="LISTING-P0",
    )
    loaded_listing = load_massive_source_bundle(
        root=tmp_path,
        relative_payload_path=listing_path,
        verified_at_ms=observed,
    )
    committed_listing = parse_massive_flat_file_listing_v0(
        root=tmp_path, loaded_listing=loaded_listing
    )
    result = build_massive_profitability_source_evidence_p0(
        root=tmp_path,
        loaded_source=loaded_source,
        loaded_listing=loaded_listing,
        committed_listing=committed_listing,
        listing_entry=committed_listing.entries[0],
    )
    assert result.source_session_date == source_date
    assert result.source_object_receipt_sha256 == loaded_source.receipt.receipt_sha256
    with pytest.raises(MassiveProfitabilityOriginP0Error, match="entry differs"):
        build_massive_profitability_source_evidence_p0(
            root=tmp_path,
            loaded_source=loaded_source,
            loaded_listing=loaded_listing,
            committed_listing=committed_listing,
            listing_entry=replace(
                committed_listing.entries[0],
                etag="changed",
                receipt_sha256=semantic_sha256(
                    replace(
                        committed_listing.entries[0],
                        etag="changed",
                        receipt_sha256="0" * 64,
                    ).unsigned()
                ),
            ),
        )


def test_p0_origin_generation_keeps_every_performance_authorization_false():
    assert MASSIVE_PROFITABILITY_P0_PANEL_MATERIALIZATION_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_P0_PREDICTIVE_TRAINING_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_P0_PROFITABILITY_REPORTING_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_P0_LOCKBOX_ACCESS_AUTHORIZED is False
