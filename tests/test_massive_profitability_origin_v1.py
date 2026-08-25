from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pytest

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
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MassiveCapturedFlatFileListingV0,
    capture_massive_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MASSIVE_FLAT_TRADES_DATASET_ID,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MASSIVE_PROFITABILITY_ORIGIN_V1_LOCKBOX_ACCESS_AUTHORIZED,
    MASSIVE_PROFITABILITY_ORIGIN_V1_PANEL_MATERIALIZATION_AUTHORIZED,
    MASSIVE_PROFITABILITY_ORIGIN_V1_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_PROFITABILITY_ORIGIN_V1_PROFITABILITY_REPORTING_AUTHORIZED,
    MassiveProfitabilityOriginV1Error,
    build_massive_profitability_decision_origin_plan_v1,
    build_massive_profitability_monthly_membership_schedule_v1,
    materialize_massive_profitability_acquired_source_evidence_v1,
    parse_massive_profitability_acquired_source_evidence_v1,
    validate_massive_profitability_decision_origin_plan_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

_EASTERN = timezone(timedelta(hours=-4))
_ENTITLEMENT = "a" * 64


def _ms(day: str, value: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp()
        * 1_000
    )


def _session_dates(count: int = 180) -> tuple[str, ...]:
    output: list[str] = []
    current = date(2020, 1, 2)
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(output)


def _sessions() -> MassiveSessionAuthority:
    source_receipt = semantic_sha256("origin-v1-calendar")
    rows = tuple(
        MassiveExchangeSession(
            session_date=session_date,
            exchange="XNYS",
            regular_open_ns=_ms(session_date, time(9, 30)) * 1_000_000,
            regular_close_ns=_ms(session_date, time(16, 0)) * 1_000_000,
            scheduled_five_minute_intervals=78,
            special_session_reason=None,
            calendar_source_receipt_sha256=source_receipt,
        )
        for session_date in _session_dates()
    )
    return build_massive_session_authority(
        rows, calendar_source_receipt_sha256=source_receipt
    )


def _first_session_for_month(
    sessions: MassiveSessionAuthority, month: str
) -> tuple[int, MassiveExchangeSession]:
    return next(
        (index, row)
        for index, row in enumerate(sessions.sessions)
        if row.session_date.startswith(month)
    )


def _identity(
    sessions: MassiveSessionAuthority, *, membership_months: tuple[str, ...]
) -> PITSecurityUniverseAuthority:
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    listing_at_ms = sessions.sessions[0].regular_open_ns // 1_000_000
    ranks = []
    for month in membership_months:
        effective_index, effective_session = _first_session_for_month(sessions, month)
        observation_end_index = effective_index - rule.ranking_lag_sessions
        observation_start_index = (
            observation_end_index - rule.ranking_lookback_sessions + 1
        )
        assert observation_start_index >= 0
        ranks.append(
            UniverseRankInputRecord(
                security_id="SEC-A",
                effective_at_ms=(effective_session.regular_open_ns // 1_000_000),
                effective_session_index=effective_index,
                available_at_ms=(
                    sessions.sessions[observation_end_index].regular_close_ns
                    // 1_000_000
                ),
                observation_start_ms=(
                    sessions.sessions[observation_start_index].regular_close_ns
                    // 1_000_000
                ),
                observation_end_ms=(
                    sessions.sessions[observation_end_index].regular_close_ns
                    // 1_000_000
                ),
                observation_start_session_index=observation_start_index,
                observation_end_session_index=observation_end_index,
                observed_session_count=rule.ranking_lookback_sessions,
                average_dollar_volume=10_000_000.0,
                close_price=100.0,
                source_receipt_sha256=semantic_sha256(("rank", month)),
            )
        )
    master = SourcedSecurityMasterRecord(
        security_id="SEC-A",
        issuer_id="ISS-A",
        primary_exchange="XNYS",
        share_class="COMMON",
        security_type="common-stock",
        listing_at_ms=listing_at_ms,
        delisting_at_ms=None,
        successor_security_id=None,
        corporate_action_chain_id="CHAIN-A",
        identity_source_receipt_sha256=semantic_sha256("master-a"),
    )
    ticker = SourcedTickerHistoryRecord(
        security_id="SEC-A",
        ticker="AAA",
        valid_from_ms=listing_at_ms,
        valid_to_ms=None,
        available_at_ms=listing_at_ms,
        primary_exchange="XNYS",
        source_receipt_sha256=semantic_sha256("ticker-a"),
    )
    listing = ListingEventRecord(
        event_id="LIST-A",
        security_id="SEC-A",
        effective_at_ms=listing_at_ms,
        available_at_ms=listing_at_ms,
        primary_exchange="XNYS",
        ticker="AAA",
        source_receipt_sha256=semantic_sha256("listing-a"),
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=(master,),
        ticker_history=(ticker,),
        listing_events=(listing,),
        delisting_events=(),
        rank_inputs=tuple(ranks),
    )


class _Paginator:
    def __init__(self, page: dict[str, object]) -> None:
        self.page = page

    def paginate(self, **_: object) -> tuple[dict[str, object], ...]:
        return (self.page,)


class _S3Client:
    def __init__(self, page: dict[str, object]) -> None:
        self.page = page

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self.page)


def _captured_source(
    *,
    tmp_path: Path,
    source_session_date: str,
    artifact_id: str,
) -> tuple[
    MassiveCapturedFlatFileListingV0,
    LoadedMassiveSourceObject,
    object,
]:
    key = canonical_massive_trade_object_key(source_session_date)
    source_bytes = f"finalized source {source_session_date}".encode()
    vendor_modified = _ms(source_session_date, time(17, 0))
    observed = _ms("2020-08-25", time(12, 0))
    page = {
        "ResponseMetadata": {"RequestId": f"LIST-{source_session_date}"},
        "IsTruncated": False,
        "Contents": [
            {
                "Key": key,
                "ETag": '"source-etag"',
                "Size": len(source_bytes),
                "LastModified": datetime.fromtimestamp(vendor_modified / 1_000, tz=UTC),
            }
        ],
    }
    ticks = iter((observed - 1, observed))
    captured = capture_massive_flat_file_listing_v0(
        s3_client=_S3Client(page),
        root=tmp_path,
        year=int(source_session_date[:4]),
        month=int(source_session_date[5:7]),
        entitlement_receipt_sha256=_ENTITLEMENT,
        now_ms=lambda: next(ticks),
    )
    publish_massive_source_object(
        stream=BytesIO(source_bytes),
        root=tmp_path,
        relative_payload_path=key,
        dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
        source_object_key=key,
        requested_at_ms=observed + 1,
        downloaded_at_ms=observed + 1,
        schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_ENTITLEMENT,
        committed_at_ms=observed + 1,
        etag="source-etag",
        request_id=f"GET-{source_session_date}",
    )
    loaded = load_massive_source_bundle(
        root=tmp_path,
        relative_payload_path=key,
        verified_at_ms=observed + 1,
    )
    artifact = materialize_massive_profitability_acquired_source_evidence_v1(
        root=tmp_path,
        captured_listings=(captured,),
        loaded_trade_sources=(loaded,),
        artifact_id=artifact_id,
        committed_at_ms=observed + 2,
        entitlement_receipt_sha256=_ENTITLEMENT,
    )
    return captured, loaded, artifact


def _candidate_and_source(
    sessions: MassiveSessionAuthority, candidate_date: str
) -> tuple[MassiveExchangeSession, MassiveExchangeSession]:
    index = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == candidate_date
    )
    return sessions.sessions[index], sessions.sessions[index - 2]


def test_acquired_source_artifact_reloads_but_generic_reload_is_nonauthorizing(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    _, source = _candidate_and_source(sessions, "2020-05-04")
    _, _, artifact = _captured_source(
        tmp_path=tmp_path,
        source_session_date=source.session_date,
        artifact_id="roundtrip",
    )
    reloaded = parse_massive_profitability_acquired_source_evidence_v1(
        root=tmp_path, loaded_source=artifact.loaded_source
    )
    assert reloaded == artifact
    assert reloaded.acquisition_qualified is False
    assert reloaded.rows[0].source_session_date == source.session_date
    assert (
        "source_evidences"
        not in inspect.signature(
            build_massive_profitability_decision_origin_plan_v1
        ).parameters
    )


def test_plan_binds_authenticated_listing_and_scheduled_monthly_membership(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    candidate, source = _candidate_and_source(sessions, "2020-05-04")
    captured, loaded, artifact = _captured_source(
        tmp_path=tmp_path,
        source_session_date=source.session_date,
        artifact_id="eligible",
    )
    identity = _identity(sessions, membership_months=("2020-05",))
    plan = build_massive_profitability_decision_origin_plan_v1(
        root=tmp_path,
        session_authority=sessions,
        identity_authority=identity,
        source_evidence_artifact=artifact,
        captured_listings=(captured,),
        loaded_trade_sources=(loaded,),
        first_candidate_decision_session_date=candidate.session_date,
        last_candidate_decision_session_date=candidate.session_date,
    )
    assert len(plan.origins) == 1
    assert plan.skipped_decisions == ()
    origin = plan.origins[0]
    assert origin.source_session_date == source.session_date
    assert origin.scheduled_rebalance_session_date == "2020-05-01"
    assert origin.membership_age_sessions == 1
    assert origin.decision_member_security_ids == ("SEC-A",)
    assert origin.vendor_lead_time_ms >= 18 * 60 * 60 * 1_000
    validate_massive_profitability_decision_origin_plan_v1(
        root=tmp_path,
        plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        source_evidence_artifact=artifact,
        captured_listings=(captured,),
        loaded_trade_sources=(loaded,),
    )
    with pytest.raises(MassiveProfitabilityOriginV1Error, match="listings and trade"):
        build_massive_profitability_decision_origin_plan_v1(
            root=tmp_path,
            session_authority=sessions,
            identity_authority=identity,
            source_evidence_artifact=artifact,
            captured_listings=(),
            loaded_trade_sources=(loaded,),
            first_candidate_decision_session_date=candidate.session_date,
            last_candidate_decision_session_date=candidate.session_date,
        )


def test_one_membership_group_cannot_cover_a_later_scheduled_month(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    identity = _identity(sessions, membership_months=("2020-05",))
    schedule = build_massive_profitability_monthly_membership_schedule_v1(
        session_authority=sessions,
        identity_authority=identity,
        first_candidate_decision_session_date="2020-05-04",
        last_candidate_decision_session_date="2020-06-02",
    )
    assert tuple(row.calendar_month for row in schedule.rows) == (
        "2020-05",
        "2020-06",
    )
    assert schedule.rows[0].membership_group_present
    assert not schedule.rows[1].membership_group_present

    candidate, source = _candidate_and_source(sessions, "2020-06-02")
    captured, loaded, artifact = _captured_source(
        tmp_path=tmp_path,
        source_session_date=source.session_date,
        artifact_id="missing-june-membership",
    )
    plan = build_massive_profitability_decision_origin_plan_v1(
        root=tmp_path,
        session_authority=sessions,
        identity_authority=identity,
        source_evidence_artifact=artifact,
        captured_listings=(captured,),
        loaded_trade_sources=(loaded,),
        first_candidate_decision_session_date=candidate.session_date,
        last_candidate_decision_session_date=candidate.session_date,
    )
    assert plan.origins == ()
    assert plan.skipped_decisions[0].reason == ("missing-scheduled-monthly-membership")
    assert plan.skipped_decisions[0].scheduled_rebalance_session_date == "2020-06-01"


def test_future_membership_changes_audit_not_earlier_origin_semantics(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    candidate, source = _candidate_and_source(sessions, "2020-05-04")
    captured, loaded, artifact = _captured_source(
        tmp_path=tmp_path,
        source_session_date=source.session_date,
        artifact_id="future-invariance",
    )
    earlier = _identity(sessions, membership_months=("2020-05",))
    with_future = _identity(sessions, membership_months=("2020-05", "2020-06"))
    assert earlier.receipt_sha256 != with_future.receipt_sha256
    common = {
        "root": tmp_path,
        "session_authority": sessions,
        "source_evidence_artifact": artifact,
        "captured_listings": (captured,),
        "loaded_trade_sources": (loaded,),
        "first_candidate_decision_session_date": candidate.session_date,
        "last_candidate_decision_session_date": candidate.session_date,
    }
    earlier_plan = build_massive_profitability_decision_origin_plan_v1(
        identity_authority=earlier, **common
    )
    future_plan = build_massive_profitability_decision_origin_plan_v1(
        identity_authority=with_future, **common
    )
    assert earlier_plan.origins[0].receipt_sha256 == (
        future_plan.origins[0].receipt_sha256
    )
    assert earlier_plan.semantic_receipt_sha256 == future_plan.semantic_receipt_sha256
    assert earlier_plan.origins[0].audit_receipt_sha256 != (
        future_plan.origins[0].audit_receipt_sha256
    )
    assert earlier_plan.audit_receipt_sha256 != future_plan.audit_receipt_sha256


def test_committed_evidence_cannot_be_replaced_by_self_consistent_rows(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    candidate, source = _candidate_and_source(sessions, "2020-05-04")
    captured, loaded, artifact = _captured_source(
        tmp_path=tmp_path,
        source_session_date=source.session_date,
        artifact_id="substitution",
    )
    identity = _identity(sessions, membership_months=("2020-05",))
    changed_row = replace(
        artifact.rows[0],
        vendor_last_modified_at_ms=artifact.rows[0].vendor_last_modified_at_ms + 1,
        receipt_sha256="0" * 64,
    )
    changed_row = replace(
        changed_row, receipt_sha256=semantic_sha256(changed_row.unsigned())
    )
    forged = replace(
        artifact,
        rows=(changed_row,),
        source_inventory_sha256=semantic_sha256(
            ((changed_row.source_session_date, changed_row.receipt_sha256),)
        ),
        semantic_receipt_sha256="0" * 64,
    )
    forged = replace(
        forged,
        semantic_receipt_sha256=semantic_sha256(forged.semantic_unsigned()),
    )
    with pytest.raises(MassiveProfitabilityOriginV1Error, match="committed bytes"):
        build_massive_profitability_decision_origin_plan_v1(
            root=tmp_path,
            session_authority=sessions,
            identity_authority=identity,
            source_evidence_artifact=forged,
            captured_listings=(captured,),
            loaded_trade_sources=(loaded,),
            first_candidate_decision_session_date=candidate.session_date,
            last_candidate_decision_session_date=candidate.session_date,
        )


def test_origin_v1_keeps_every_performance_authorization_false() -> None:
    assert MASSIVE_PROFITABILITY_ORIGIN_V1_PANEL_MATERIALIZATION_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V1_PREDICTIVE_TRAINING_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V1_PROFITABILITY_REPORTING_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V1_LOCKBOX_ACCESS_AUTHORIZED is False
