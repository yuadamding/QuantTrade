from __future__ import annotations

from dataclasses import asdict, replace
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
    canonical_massive_trade_object_key,
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
from rl_quant.evaluation.massive_profitability_selection_guard_v1 import (
    MassiveProfitabilitySelectedPositionV1,
    MassiveProfitabilitySelectionGuardV1Error,
    guard_massive_profitability_selected_positions_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.features.massive_economic_coverage_v8 import (
    MASSIVE_ECONOMIC_COVERAGE_V8_DATASET,
    MASSIVE_ECONOMIC_COVERAGE_V8_OBJECT_PREFIX,
    MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA,
    MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SCHEMA_SHA256,
    MassiveEconomicCoverageScopeV8,
    MassiveNativeEconomicObservationV8,
    MassiveZeroCashPolicyV8,
    parse_massive_economic_origin_coverage_v8,
)
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    materialize_massive_profitability_accounting_freeze_for_test_v1,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256,
    MassiveProfitabilityDailyInputAuthorityV1,
    MassiveProfitabilityDailyInputSessionV1,
    MassiveProfitabilityDailySecurityInputV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    massive_profitability_identity_semantic_receipt_v2,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    build_massive_profitability_feature_accounting_authority_v2,
)
from rl_quant.features.massive_profitability_fill_source_authority_v2 import (
    MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SPEC_SHA256,
    MassiveProfitabilityFillSourceAuthorityV2,
    MassiveProfitabilityFillSourceRowV2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    build_massive_profitability_origin_features_v3,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MASSIVE_PROFITABILITY_DECISION_ORIGIN_V1_SCHEMA,
    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V1_SCHEMA,
    MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256,
    MassiveProfitabilityDecisionOriginPlanV1,
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
    MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    build_massive_profitability_target_accounting_authority_v2,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    build_massive_profitability_targets_v2,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SPEC_SHA256,
    MassiveProfitabilityTerminalCoverageAuthorityV1,
    MassiveProfitabilityTerminalSupportRowV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

_EASTERN = ZoneInfo("America/New_York")
_ENTITLEMENT = "e" * 64


def _ms(day: str, value: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp()
        * 1_000
    )


def _sessions() -> MassiveSessionAuthority:
    days: list[str] = []
    current = date(2020, 1, 2)
    while len(days) < 130:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    source = semantic_sha256("source-derived-accounting-calendar")
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
        for day in days
    )
    return build_massive_session_authority(rows, calendar_source_receipt_sha256=source)


def _identity(sessions: MassiveSessionAuthority) -> PITSecurityUniverseAuthority:
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    effective_index = 64
    listing = sessions.sessions[0].regular_open_ns // 1_000_000
    delisting = sessions.sessions[68].regular_close_ns // 1_000_000
    rank = UniverseRankInputRecord(
        security_id="SEC-A",
        effective_at_ms=sessions.sessions[effective_index].regular_open_ns // 1_000_000,
        effective_session_index=effective_index,
        available_at_ms=sessions.sessions[63].regular_close_ns // 1_000_000,
        observation_start_ms=sessions.sessions[1].regular_close_ns // 1_000_000,
        observation_end_ms=sessions.sessions[63].regular_close_ns // 1_000_000,
        observation_start_session_index=1,
        observation_end_session_index=63,
        observed_session_count=63,
        average_dollar_volume=10_000_000.0,
        close_price=163.0,
        source_receipt_sha256=semantic_sha256("source-derived-rank"),
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
                delisting_at_ms=delisting,
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
                valid_to_ms=delisting,
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
        delisting_events=(
            DelistingEventRecord(
                event_id="DELIST-A",
                security_id="SEC-A",
                effective_at_ms=delisting,
                available_at_ms=delisting,
                reason="unresolved",
                successor_security_id=None,
                source_receipt_sha256=semantic_sha256("delisting-a"),
            ),
        ),
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


def _origin_plan(
    origin: MassiveProfitabilityDecisionOriginV1,
) -> MassiveProfitabilityDecisionOriginPlanV2:
    source_semantic = semantic_sha256("source-artifact")
    source_audit = semantic_sha256("source-artifact-audit")
    identity_audit = semantic_sha256("identity-audit")
    provisional_v1 = MassiveProfitabilityDecisionOriginPlanV1(
        first_candidate_decision_session_date=origin.decision_session_date,
        last_candidate_decision_session_date=origin.decision_session_date,
        candidate_decision_session_dates=(origin.decision_session_date,),
        origins=(origin,),
        skipped_decisions=(),
        source_evidence_artifact_semantic_receipt_sha256=source_semantic,
        membership_schedule_semantic_receipt_sha256=(
            origin.membership_schedule_semantic_receipt_sha256
        ),
        session_authority_receipt_sha256=origin.session_authority_receipt_sha256,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        origin_spec_receipt_sha256=MASSIVE_PROFITABILITY_ORIGIN_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_ORIGIN_V1_SOURCE_SHA256,
        semantic_receipt_sha256="0" * 64,
        identity_authority_audit_receipt_sha256=identity_audit,
        source_evidence_artifact_audit_receipt_sha256=source_audit,
        audit_receipt_sha256="0" * 64,
        panel_materialization_authorized=False,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        schema=MASSIVE_PROFITABILITY_ORIGIN_PLAN_V1_SCHEMA,
    )
    v1_semantic = semantic_sha256(provisional_v1.semantic_unsigned())
    v1 = replace(
        provisional_v1,
        semantic_receipt_sha256=v1_semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": v1_semantic,
                "identity_authority_audit_receipt_sha256": identity_audit,
                "source_evidence_artifact_audit_receipt_sha256": source_audit,
            }
        ),
    )
    v1.validate()
    provisional_v2 = MassiveProfitabilityDecisionOriginPlanV2(
        origin_plan_v1=v1,
        production_acquisition_receipt_sha256=semantic_sha256("acquisition"),
        monthly_rank_authority_semantic_receipt_sha256=semantic_sha256("rank"),
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256,
        semantic_receipt_sha256="0" * 64,
        acquisition_audit_receipt_sha256=semantic_sha256("acquisition-audit"),
        monthly_rank_audit_receipt_sha256=semantic_sha256("rank-audit"),
        audit_receipt_sha256="0" * 64,
        panel_materialization_authorized=False,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        schema=MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
    )
    v2_semantic = semantic_sha256(provisional_v2.semantic_unsigned())
    result = replace(
        provisional_v2,
        semantic_receipt_sha256=v2_semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": v2_semantic,
                "acquisition_audit_receipt_sha256": (
                    provisional_v2.acquisition_audit_receipt_sha256
                ),
                "monthly_rank_audit_receipt_sha256": (
                    provisional_v2.monthly_rank_audit_receipt_sha256
                ),
                "origin_plan_v1_audit_receipt_sha256": v1.audit_receipt_sha256,
            }
        ),
    )
    result.validate()
    return result


def _daily(
    sessions: MassiveSessionAuthority, identity: PITSecurityUniverseAuthority
) -> MassiveProfitabilityDailyInputAuthorityV1:
    rows = []
    session_rows = []
    tape_values = tuple(1.0 for _ in MASSIVE_DAILY_TAPE_V0_FIELDS)
    for index, session in enumerate(sessions.sessions):
        price = 100.0 + index
        bars = {
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "share_volume": 1_000.0,
            "dollar_volume": price * 1_000.0,
            "high_low_range": 2.0 / price,
            "close_location": 0.5,
        }
        body = {
            "source_session_date": session.session_date,
            "security_id": "SEC-A",
            "bars_values": tuple(bars[name] for name in MASSIVE_DAILY_BARS_V0_FIELDS),
            "bars_valid": (True,) * len(MASSIVE_DAILY_BARS_V0_FIELDS),
            "tape_values": tape_values,
            "tape_valid": (True,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS),
            "signed_dollar_flow": 50.0,
            "same_population_dollar_volume": 100.0,
            "absolute_signed_flow_imbalance": 0.5,
            "same_population_valid": True,
            "regular_session_event_count": 10,
            "replacement_event_count": 1,
            "cancellation_event_count": 1,
            "late_report_event_count": 1,
            "daily_bar_row_receipt_sha256": semantic_sha256(
                (session.session_date, "bar")
            ),
            "daily_tape_row_receipt_sha256": semantic_sha256(
                (session.session_date, "tape")
            ),
            "tape_population_row_receipt_sha256": semantic_sha256(
                (session.session_date, "population")
            ),
            "persisted_partition_receipt_sha256": semantic_sha256(
                (session.session_date, "partition")
            ),
        }
        row = MassiveProfitabilityDailySecurityInputV1(
            **body, receipt_sha256=semantic_sha256(body)
        )
        row.validate()
        rows.append(row)
        session_body = {
            "source_session_date": session.session_date,
            "regular_open_at_ms": session.regular_open_ns // 1_000_000,
            "regular_close_at_ms": session.regular_close_ns // 1_000_000,
            "vendor_last_modified_at_ms": session.regular_close_ns // 1_000_000,
            "authenticated_get_completed_at_ms": (
                session.regular_close_ns // 1_000_000
            ),
            "authenticated_download_receipt_sha256": semantic_sha256(
                (session.session_date, "download")
            ),
            "whole_file_scan_receipt_sha256": semantic_sha256(
                (session.session_date, "scan")
            ),
            "semantic_partition_manifest_receipt_sha256": semantic_sha256(
                (session.session_date, "semantic")
            ),
            "persisted_partition_manifest_receipt_sha256": semantic_sha256(
                (session.session_date, "persisted")
            ),
            "daily_bars_artifact_receipt_sha256": semantic_sha256(
                (session.session_date, "bars")
            ),
            "daily_tape_artifact_receipt_sha256": semantic_sha256(
                (session.session_date, "tapes")
            ),
            "supported_security_row_inventory_sha256": semantic_sha256(
                (row.receipt_sha256,)
            ),
        }
        session_row = MassiveProfitabilityDailyInputSessionV1(
            **session_body, receipt_sha256=semantic_sha256(session_body)
        )
        session_row.validate()
        session_rows.append(session_row)
    provisional = MassiveProfitabilityDailyInputAuthorityV1(
        coverage_start_session_date=sessions.sessions[0].session_date,
        coverage_end_session_date=sessions.sessions[-1].session_date,
        data_freeze_at_ms=sessions.sessions[-1].regular_close_ns // 1_000_000,
        supported_security_ids=("SEC-A",),
        sessions=tuple(session_rows),
        rows=tuple(rows),
        archive_freeze_semantic_receipt_sha256=semantic_sha256("archive-freeze"),
        security_support_semantic_receipt_sha256=None,
        session_authority_receipt_sha256=sessions.receipt_sha256,
        normalized_identity_semantic_receipt_sha256=(
            massive_profitability_identity_semantic_receipt_v2(identity)
        ),
        condition_authority_receipt_sha256=semantic_sha256("conditions"),
        correction_authority_receipt_sha256=semantic_sha256("corrections"),
        event_domain_spec_receipt_sha256=semantic_sha256("domain"),
        session_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in session_rows)
        ),
        row_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        source_transport_qualified=False,
        daily_input_data_qualified=False,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256,
        semantic_receipt_sha256="0" * 64,
        acquisition_audit_receipt_sha256=semantic_sha256("daily-acquisition"),
        audit_receipt_sha256="0" * 64,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    semantic = semantic_sha256(provisional.semantic_unsigned())
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "acquisition_audit_receipt_sha256": (
                    provisional.acquisition_audit_receipt_sha256
                ),
            }
        ),
    )
    result.validate()
    return result


def _terminal(
    sessions: MassiveSessionAuthority, identity: PITSecurityUniverseAuthority
) -> MassiveProfitabilityTerminalCoverageAuthorityV1:
    delisting = identity.delisting_events[0]
    body = {
        "security_id": "SEC-A",
        "listing_delisting_event_id": delisting.event_id,
        "effective_at_ms": delisting.effective_at_ms,
        "provider_available_at_ms": delisting.available_at_ms,
        "resolution_kind": "conservative-total-loss",
        "cash_per_share": 0.0,
        "successor_security_id": None,
        "successor_ratio": 0.0,
        "conservative_total_loss": True,
        "identity_delisting_receipt_sha256": delisting.source_receipt_sha256,
        "provider_disposition_receipt_sha256": None,
    }
    row = MassiveProfitabilityTerminalSupportRowV1(
        **body, receipt_sha256=semantic_sha256(body)
    )
    row.validate()
    provisional = MassiveProfitabilityTerminalCoverageAuthorityV1(
        coverage_start_date=sessions.sessions[0].session_date,
        coverage_end_date=sessions.sessions[-1].session_date,
        supported_security_ids=("SEC-A",),
        rows=(row,),
        known_disposition_count=0,
        exact_provider_disposition_count=0,
        conservative_total_loss_count=1,
        terminal_accounting_mode="conservative-lower-bound",
        support_semantic_receipt_sha256=semantic_sha256("support"),
        normalized_identity_semantic_receipt_sha256=(
            massive_profitability_identity_semantic_receipt_v2(identity)
        ),
        terminal_source_semantic_receipt_sha256=semantic_sha256("terminal-source"),
        row_inventory_sha256=semantic_sha256((row.receipt_sha256,)),
        structural_terminal_coverage_complete=True,
        terminal_source_runtime_qualified=False,
        terminal_evidence_data_qualified=False,
        conservative_lower_bound_complete=True,
        terminal_accounting_data_qualified=True,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=(
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SPEC_SHA256
        ),
        implementation_source_sha256=(
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SOURCE_SHA256
        ),
        semantic_receipt_sha256="0" * 64,
        terminal_source_audit_receipt_sha256=semantic_sha256("terminal-audit"),
        audit_receipt_sha256="0" * 64,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    semantic = semantic_sha256(provisional.semantic_unsigned())
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "terminal_source_audit_receipt_sha256": (
                    provisional.terminal_source_audit_receipt_sha256
                ),
            }
        ),
    )
    result.validate()
    return result


def _fills(
    *,
    sessions: MassiveSessionAuthority,
    origin: MassiveProfitabilityDecisionOriginV1,
    daily: MassiveProfitabilityDailyInputAuthorityV1,
) -> MassiveProfitabilityFillSourceAuthorityV2:
    decision_index = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == origin.decision_session_date
    )
    selected = tuple(
        sessions.sessions[decision_index + offset]
        for offset in (0, 1, 5, 21, 63)
    )
    rows = []
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    for session in selected:
        daily_row = daily.row(session_date=session.session_date, security_id="SEC-A")
        price = daily_row.bars_values[close_index]
        body = {
            "session_date": session.session_date,
            "security_id": "SEC-A",
            "fill_start_at_ms": session.regular_close_ns // 1_000_000 - 600_000,
            "fill_end_at_ms": session.regular_close_ns // 1_000_000,
            "fill_vwap": price,
            "qualifying_share_volume": 100.0,
            "qualifying_dollar_volume": price * 100.0,
            "qualifying_trade_count": 1,
            "valid": True,
            "qualifying_trade_inventory_sha256": semantic_sha256(
                (session.session_date, "fill-trade")
            ),
            "persisted_partition_receipt_sha256": (
                daily_row.persisted_partition_receipt_sha256
            ),
            "daily_input_row_receipt_sha256": daily_row.receipt_sha256,
        }
        row = MassiveProfitabilityFillSourceRowV2(
            **body, receipt_sha256=semantic_sha256(body)
        )
        row.validate()
        rows.append(row)
    dates = tuple(row.session_date for row in selected)
    provisional = MassiveProfitabilityFillSourceAuthorityV2(
        session_dates=dates,
        supported_security_ids=("SEC-A",),
        rows=tuple(rows),
        daily_input_authority_semantic_receipt_sha256=daily.semantic_receipt_sha256,
        origin_plan_semantic_receipt_sha256=None,
        security_support_semantic_receipt_sha256=None,
        condition_authority_receipt_sha256=semantic_sha256("conditions"),
        persisted_manifest_inventory_sha256=semantic_sha256("fill-manifests"),
        row_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        source_data_qualified=False,
        fill_source_data_qualified=False,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SOURCE_SHA256,
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256="0" * 64,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    semantic = semantic_sha256(provisional.semantic_unsigned())
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "daily_input_audit_receipt_sha256": daily.semantic_receipt_sha256,
            }
        ),
    )
    result.validate()
    return result


def _event(
    *,
    sessions: MassiveSessionAuthority,
    session_index: int = 66,
    provider_event_key: str = "DIV-A",
) -> MassiveNativeEconomicObservationV8:
    effective = sessions.sessions[session_index].regular_open_ns // 1_000_000
    body = {
        "surface_id": "massive-dividends-v1",
        "provider_event_key": provider_event_key,
        "logical_event_key": semantic_sha256(provider_event_key),
        "security_id": "SEC-A",
        "ticker": "AAA",
        "kind": "cash-dividend",
        "classification": "recurring",
        "effective_at_ms": effective,
        "research_captured_at_ms": effective + 1,
        "accounting_lane": "finalized-accounting-research",
        "predictive_feature_eligible": False,
        "currency": "USD",
        "cash_per_share": 1.0,
        "split_adjusted_cash_per_share": 1.0,
        "share_ratio": 1.0,
        "historical_adjustment_factor": 1.0,
        "raw_provider_request_id": "REQ",
        "raw_provider_row_locator": "page=0/results=0",
        "raw_provider_row_sha256": semantic_sha256(
            ("raw-dividend", provider_event_key)
        ),
        "identity_mapping_receipt_sha256": semantic_sha256("map-dividend"),
    }
    result = MassiveNativeEconomicObservationV8(
        **body, receipt_sha256=semantic_sha256(body)
    )
    result.validate()
    return result


def _coverage(
    *,
    root: Path,
    sessions: MassiveSessionAuthority,
    terminal: MassiveProfitabilityTerminalCoverageAuthorityV1,
    origin: MassiveProfitabilityDecisionOriginV1,
    events: tuple[MassiveNativeEconomicObservationV8, ...],
    artifact_id: str,
):
    scope = MassiveEconomicCoverageScopeV8.build(
        coverage_start_date=sessions.sessions[0].session_date,
        coverage_end_date=sessions.sessions[-1].session_date,
    )
    cash = MassiveZeroCashPolicyV8.build()
    semantic_body = {
        "schema": MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA,
        "scope": asdict(scope),
        "decision_at_ms": origin.decision_at_ms,
        "accounting_lane": "finalized-accounting-research",
        "selected_events": tuple(asdict(row) for row in events),
        "terminal_inventory_sha256": semantic_sha256("terminal-inventory"),
        "cash_policy": asdict(cash),
        "ambiguous_interaction_group_receipts": (),
        "coverage_qualified": True,
    }
    semantic_receipt = semantic_sha256(semantic_body)
    audit_receipt = semantic_sha256(
        {
            "semantic_receipt_sha256": semantic_receipt,
            "capture_receipts": (),
            "future_capture_receipts": (),
            "terminal_source_receipt_sha256": (
                terminal.terminal_source_semantic_receipt_sha256
            ),
        }
    )
    payload = {
        "schema": MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA,
        "scope": asdict(scope),
        "decision_at_ms": origin.decision_at_ms,
        "accounting_lane": "finalized-accounting-research",
        "capture_receipts": (),
        "selected_events": tuple(asdict(row) for row in events),
        "future_capture_receipts": (),
        "terminal_source_receipt_sha256": (
            terminal.terminal_source_semantic_receipt_sha256
        ),
        "terminal_inventory_sha256": semantic_sha256("terminal-inventory"),
        "cash_policy": asdict(cash),
        "ambiguous_interaction_group_receipts": (),
        "coverage_qualified": True,
        "semantic_receipt_sha256": semantic_receipt,
        "audit_receipt_sha256": audit_receipt,
    }
    relative = f"{MASSIVE_ECONOMIC_COVERAGE_V8_OBJECT_PREFIX}{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ECONOMIC_COVERAGE_V8_DATASET,
        source_object_key=relative,
        requested_at_ms=origin.decision_at_ms,
        downloaded_at_ms=origin.decision_at_ms,
        schema_sha256=MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_ENTITLEMENT,
        committed_at_ms=origin.decision_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=origin.decision_at_ms
    )
    return parse_massive_economic_origin_coverage_v8(
        root=root, loaded_source=loaded
    )


def test_feature_semantics_exclude_future_target_period_event(tmp_path: Path) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    origin = _origin(sessions)
    plan = _origin_plan(origin)
    daily = _daily(sessions, identity)
    terminal = _terminal(sessions, identity)
    without_future = _coverage(
        root=tmp_path,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(),
        artifact_id="without-future",
    )
    with_future = _coverage(
        root=tmp_path,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(_event(sessions=sessions),),
        artifact_id="with-future",
    )
    first = build_massive_profitability_feature_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        economic_coverage=without_future,
        terminal_authority=terminal,
    )
    second = build_massive_profitability_feature_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        economic_coverage=with_future,
        terminal_authority=terminal,
    )
    assert first.semantic_receipt_sha256 == second.semantic_receipt_sha256
    assert first.audit_receipt_sha256 != second.audit_receipt_sha256
    assert len(first.accounting.session_dates) == 64
    assert first.maximum_actual_input_at_ms == origin.feature_cutoff_at_ms
    assert first.economic_values_data_qualified is False


def test_target_replay_preserves_cash_then_conservatively_loses_security(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    origin = _origin(sessions)
    plan = _origin_plan(origin)
    daily = _daily(sessions, identity)
    fills = _fills(sessions=sessions, origin=origin, daily=daily)
    terminal = _terminal(sessions, identity)
    coverage = _coverage(
        root=tmp_path,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(_event(sessions=sessions),),
        artifact_id="target",
    )
    accounting = build_massive_profitability_target_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        fill_source_authority=fills,
        economic_coverage=coverage,
        terminal_authority=terminal,
    )
    assert accounting.target(security_id="SEC-A", horizon_sessions=1) == pytest.approx(
        (166.0 + 1.0) / 165.0 - 1.0
    )
    assert accounting.target(security_id="SEC-A", horizon_sessions=5) == pytest.approx(
        1.0 / 165.0 - 1.0
    )
    assert accounting.conservative_total_loss_target_count == 1
    assert accounting.rows[0].unresolved_terminal_fallback_session_offset == 3
    assert accounting.fill_sources_qualified is False
    assert accounting.economic_values_data_qualified is False
    assert accounting.predictive_training_authorized is False

    no_cash_coverage = _coverage(
        root=tmp_path,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(),
        artifact_id="target-no-cash",
    )
    no_cash = build_massive_profitability_target_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        fill_source_authority=fills,
        economic_coverage=no_cash_coverage,
        terminal_authority=terminal,
    )
    assert no_cash.target(security_id="SEC-A", horizon_sessions=5) == -1.0


def test_target_semantics_exclude_events_after_h63(tmp_path: Path) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    origin = _origin(sessions)
    plan = _origin_plan(origin)
    daily = _daily(sessions, identity)
    fills = _fills(sessions=sessions, origin=origin, daily=daily)
    terminal = _terminal(sessions, identity)
    without_future = _coverage(
        root=tmp_path,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(),
        artifact_id="target-without-post-h63",
    )
    with_future = _coverage(
        root=tmp_path,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(
            _event(
                sessions=sessions,
                session_index=129,
                provider_event_key="DIV-POST-H63",
            ),
        ),
        artifact_id="target-with-post-h63",
    )
    first = build_massive_profitability_target_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        fill_source_authority=fills,
        economic_coverage=without_future,
        terminal_authority=terminal,
    )
    second = build_massive_profitability_target_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        fill_source_authority=fills,
        economic_coverage=with_future,
        terminal_authority=terminal,
    )
    assert first.semantic_receipt_sha256 == second.semantic_receipt_sha256
    assert first.audit_receipt_sha256 != second.audit_receipt_sha256


def test_source_owned_accounting_drives_v3_features_and_v2_targets(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    origin = _origin(sessions)
    plan = _origin_plan(origin)
    daily = _daily(sessions, identity)
    terminal = _terminal(sessions, identity)
    fills = _fills(sessions=sessions, origin=origin, daily=daily)
    coverage = _coverage(
        root=tmp_path,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(_event(sessions=sessions),),
        artifact_id="integrated-source-owned",
    )
    accounting_freeze = (
        materialize_massive_profitability_accounting_freeze_for_test_v1(
            root=tmp_path,
            archive_freeze_semantic_receipt_sha256=(
                daily.archive_freeze_semantic_receipt_sha256
            ),
            origin_plan=plan,
            terminal_authority=terminal,
            economic_coverages=(coverage,),
            accounting_freeze_at_ms=origin.decision_at_ms + 1_000,
            entitlement_receipt_sha256=_ENTITLEMENT,
            artifact_id="integrated-source-owned",
        )
    )
    feature_accounting = build_massive_profitability_feature_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        economic_coverage=coverage,
        terminal_authority=terminal,
    )
    features = build_massive_profitability_origin_features_v3(
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        feature_accounting=feature_accounting,
        accounting_freeze=accounting_freeze,
        terminal_authority=terminal,
    )
    assert features.input_session_dates == tuple(
        row.session_date for row in sessions.sessions[:64]
    )
    assert len(features.rows) == 1
    assert features.maximum_economic_input_at_ms == origin.feature_cutoff_at_ms
    assert features.source_inputs_data_qualified is False
    assert "rl-quant.massive-session-panel-v1" not in features.input_schemas
    assert "rl-quant.massive-profitability-feature-accounting-v1" not in (
        features.input_schemas
    )

    target_accounting = build_massive_profitability_target_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        fill_source_authority=fills,
        economic_coverage=coverage,
        terminal_authority=terminal,
    )
    targets = build_massive_profitability_targets_v2(
        accounting=target_accounting,
        origin_plan=plan,
        accounting_freeze=accounting_freeze,
        terminal_authority=terminal,
    )
    assert targets.rows[0].simple_returns[0] == pytest.approx(
        (166.0 + 1.0) / 165.0 - 1.0
    )
    assert targets.rows[0].simple_returns[1] == pytest.approx(1.0 / 165.0 - 1.0)
    assert targets.terminal_accounting_mode == "conservative-lower-bound"
    assert targets.exact_provider_disposition_count == 0
    assert targets.conservative_total_loss_count == 1
    assert targets.source_inputs_data_qualified is False
    assert "rl-quant.massive-profitability-target-accounting-v1" not in (
        targets.input_schemas
    )


def test_selected_positions_fail_on_missing_exit_and_short_terminal_windfall(
    tmp_path: Path,
) -> None:
    sessions = _sessions()
    identity = _identity(sessions)
    origin = _origin(sessions)
    plan = _origin_plan(origin)
    daily = _daily(sessions, identity)
    terminal = _terminal(sessions, identity)
    fills = _fills(sessions=sessions, origin=origin, daily=daily)
    coverage = _coverage(
        root=tmp_path,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(),
        artifact_id="selection-guard",
    )
    accounting_freeze = materialize_massive_profitability_accounting_freeze_for_test_v1(
        root=tmp_path,
        archive_freeze_semantic_receipt_sha256=(
            daily.archive_freeze_semantic_receipt_sha256
        ),
        origin_plan=plan,
        terminal_authority=terminal,
        economic_coverages=(coverage,),
        accounting_freeze_at_ms=origin.decision_at_ms + 1_000,
        entitlement_receipt_sha256=_ENTITLEMENT,
        artifact_id="selection-guard",
    )
    accounting = build_massive_profitability_target_accounting_authority_v2(
        root=tmp_path,
        origin=origin,
        origin_plan=plan,
        session_authority=sessions,
        identity_authority=identity,
        daily_input_authority=daily,
        fill_source_authority=fills,
        economic_coverage=coverage,
        terminal_authority=terminal,
    )
    targets = build_massive_profitability_targets_v2(
        accounting=accounting,
        origin_plan=plan,
        accounting_freeze=accounting_freeze,
        terminal_authority=terminal,
    )
    selected_long = MassiveProfitabilitySelectedPositionV1(
        decision_session_date=origin.decision_session_date,
        security_id="SEC-A",
        horizon_sessions=5,
        side="long",
    )
    support = guard_massive_profitability_selected_positions_v1(
        selected_positions=(selected_long,), targets=(targets,)
    )
    assert support.selected_exit_support_complete is True
    assert support.direction_safe_terminal_support_complete is True
    assert support.profitability_reporting_authorized is False

    selected_short = replace(selected_long, side="short")
    with pytest.raises(
        MassiveProfitabilitySelectionGuardV1Error,
        match="cannot credit a selected short",
    ):
        guard_massive_profitability_selected_positions_v1(
            selected_positions=(selected_short,), targets=(targets,)
        )

    original_row = targets.rows[0]
    missing_body = original_row.unsigned()
    simple_returns = list(original_row.simple_returns)
    valid = list(original_row.valid)
    terminal_zero = list(original_row.terminal_zero_value)
    simple_returns[0] = 0.0
    valid[0] = False
    terminal_zero[0] = False
    missing_body.update(
        simple_returns=tuple(simple_returns),
        valid=tuple(valid),
        terminal_zero_value=tuple(terminal_zero),
    )
    missing_row = replace(
        original_row,
        simple_returns=tuple(simple_returns),
        valid=tuple(valid),
        terminal_zero_value=tuple(terminal_zero),
        receipt_sha256=semantic_sha256(missing_body),
    )
    missing_artifact = replace(
        targets,
        rows=(missing_row,),
        valid_counts_by_horizon=tuple(int(value) for value in missing_row.valid),
        row_inventory_sha256=semantic_sha256((missing_row.receipt_sha256,)),
        semantic_receipt_sha256="0" * 64,
    )
    missing_artifact = replace(
        missing_artifact,
        semantic_receipt_sha256=semantic_sha256(missing_artifact.semantic_unsigned()),
    )
    missing_artifact.validate()

    # Missing support for an unselected horizon is retained as missingness.
    guard_massive_profitability_selected_positions_v1(
        selected_positions=(selected_long,), targets=(missing_artifact,)
    )
    selected_missing = replace(selected_long, horizon_sessions=1)
    with pytest.raises(
        MassiveProfitabilitySelectionGuardV1Error,
        match="lacks its scheduled exit fill",
    ):
        guard_massive_profitability_selected_positions_v1(
            selected_positions=(selected_missing,), targets=(missing_artifact,)
        )
