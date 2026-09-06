"""Persisted synthetic source package for the authoritative V5 qualification.

The fixture deliberately uses the production authority classes, source graph,
dependency reconstruction, trainers, and V5 root.  Only the market observations
are synthetic.  No action, transition, return, selection, seal, report, or
authorization result is supplied to the experiment runner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from io import BytesIO
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast
from zoneinfo import ZoneInfo

from rl_quant.workflows.massive_adaptive_rl_vertical_qualification_runner_v1 import (
    MASSIVE_ADAPTIVE_RL_FRESH_REPLAY_TIMEOUT_SECONDS_V1,
)
from rl_quant.alpha.massive_universe_adapter import checked_pit_universe_rule
from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.decision_clock import (
    MassiveDecisionClockAuthority,
    build_massive_decision_clock_authorities,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MASSIVE_PERSISTED_ACTIVE_DATASET_V1,
    MASSIVE_PERSISTED_CORRECTION_SCHEMA_SHA256,
    MASSIVE_PERSISTED_CORRECTIONS_DATASET_V1,
    MASSIVE_PERSISTED_EVENTS_DATASET_V1,
    MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
    MASSIVE_PERSISTED_PARTITION_SOURCE_SHA256,
    MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
    MassivePersistedPartitionManifestV1,
    MassivePersistedSecurityPartitionV1,
)
from rl_quant.data_sources.massive.session_calendar import (
    FIVE_MINUTES_NS,
    MassiveExchangeSession,
    MassiveSessionAuthority,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v1 import (
    materialize_massive_adaptive_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
    materialize_massive_adaptive_forecast_calibration_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    MassiveAdaptiveRLFitForecastArchiveV1,
    materialize_massive_adaptive_rl_fit_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_inference_plan_v1 import (
    build_massive_adaptive_rl_fit_inference_plan_v1,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
    build_massive_adaptive_context_origin_authority_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    authorize_massive_adaptive_decision_tensor_v1,
    materialize_massive_adaptive_decision_tensor_v1,
    parse_massive_adaptive_decision_tensor_v1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import (
    MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA,
    MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256,
    MassiveAdaptiveFillRowV1,
    MassiveAdaptiveFillSourceV1,
    adaptive_fill_clock_v1,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_source_targets_v1 import (
    MassiveAdaptiveSourceTargetsV1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256,
    MassiveProfitabilityDailyInputAuthorityV1,
    MassiveProfitabilityDailyInputSessionV1,
    MassiveProfitabilityDailySecurityInputV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    massive_profitability_identity_semantic_receipt_v2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
    MassiveAdaptiveAlphaModelSpecV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training import massive_adaptive_supervised_trainer_v1 as supervised
from rl_quant.training.massive_adaptive_checkpoint_v1 import MassiveAdaptiveCheckpointV1
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptivePPOConfigV1
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    MassiveAdaptiveCausalCheckpointChoiceV1,
    build_massive_adaptive_causal_checkpoint_choice_v1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1,
    build_massive_adaptive_split_plan_v1,
)
from rl_quant.training.massive_adaptive_training_authority_v1 import (
    build_massive_adaptive_training_authority_v1,
)
from rl_quant.training.massive_adaptive_supervised_trainer_v1 import (
    MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import MassiveAdaptiveWindowPlanV1
from rl_quant.workflows import (
    massive_adaptive_rl_runtime_source_reconstruction_v1 as reconstruction,
)
from rl_quant.workflows.massive_adaptive_rl_deterministic_runtime_v1 import (
    MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT,
    configure_massive_adaptive_rl_deterministic_runtime_v1,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v5 import (
    MassiveAdaptiveRLPrequentialRunV5,
    run_massive_adaptive_rl_experiment_v5,
    verify_massive_adaptive_rl_experiment_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MassiveAdaptiveRLExperimentManifestV5,
    build_massive_adaptive_rl_experiment_manifest_v5,
    write_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    MassiveAdaptiveRLTypedAuthorityInventoryV1,
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    build_massive_adaptive_rl_typed_authority_inventory_v1,
    load_massive_adaptive_rl_runtime_source_graph_authority_v1,
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1,
    MassiveAdaptiveRLRoleBoundSourceAuthorityV1,
    authorize_massive_adaptive_rl_source_bundle_v1,
    bind_massive_adaptive_rl_source_authority_v1,
    load_massive_adaptive_rl_source_bundle_v1,
    materialize_massive_adaptive_rl_source_bundle_v1,
)
from rl_quant.workflows.massive_adaptive_rl_vertical_qualification_scope_v1 import (
    MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_ENVIRONMENT_V1,
    massive_adaptive_rl_vertical_qualification_scope_v1,
)
from test_massive_adaptive_decision_tensor_v1 import _origin
from test_massive_adaptive_profitability_v1_vertical_slice import (
    _empty_event_archive,
)
from test_massive_adaptive_source_authorized_training_v1 import _source_target
from test_massive_profitability_v6_vertical_slice import _feature_and_target
from test_massive_trade_replay import _conditions


_EASTERN = ZoneInfo("America/New_York")
# The registered origin exposure panel has six columns (including the
# intercept) and requires residual degrees of freedom. Eight securities retain
# the smallest numerically stable cross-section used by the compiler tests.
_SECURITY_IDS = tuple(f"SEC-{index:02d}" for index in range(8))
_ENTITLEMENT = semantic_sha256("v5-persisted-qualification-entitlement")
_MODEL_CONTEXT_SESSIONS = 1


def _digest(value: object) -> str:
    return semantic_sha256(("v5-persisted-qualification", value))


def _reseal(value: Any, /, **changes: object) -> Any:
    provisional = replace(value, **changes, semantic_receipt_sha256="0" * 64)
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _business_dates(start: date, count: int) -> tuple[str, ...]:
    result: list[str] = []
    cursor = start
    while len(result) < count:
        if cursor.weekday() < 5:
            result.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(result)


def _sessions() -> MassiveSessionAuthority:
    source = _digest("session-calendar")
    rows = []
    for session_date in _business_dates(
        date(2015, 1, 2), MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1
    ):
        day = date.fromisoformat(session_date)
        regular_open = int(
            datetime.combine(day, time(9, 30), tzinfo=_EASTERN).timestamp()
            * 1_000_000_000
        )
        rows.append(
            MassiveExchangeSession(
                session_date=session_date,
                exchange="XNYS",
                regular_open_ns=regular_open,
                regular_close_ns=regular_open + 78 * FIVE_MINUTES_NS,
                scheduled_five_minute_intervals=78,
                special_session_reason=None,
                calendar_source_receipt_sha256=source,
            )
        )
    return build_massive_session_authority(
        tuple(rows), calendar_source_receipt_sha256=source
    )


def _context_identity(sessions: MassiveSessionAuthority) -> PITSecurityUniverseAuthority:
    rule = checked_pit_universe_rule(
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.context_universe_rule
    )
    effective_index = 100
    end_index = effective_index - rule.ranking_lag_sessions
    start_index = end_index - rule.ranking_lookback_sessions + 1
    effective = sessions.sessions[effective_index]
    listing_at_ms = sessions.sessions[0].regular_open_ns // 1_000_000 - 1
    masters = tuple(
        SourcedSecurityMasterRecord(
            security_id=security_id,
            issuer_id=f"ISSUER-{index:02d}",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=listing_at_ms,
            delisting_at_ms=None,
            successor_security_id=None,
            corporate_action_chain_id=f"CHAIN-{index:02d}",
            identity_source_receipt_sha256=_digest((security_id, "master")),
        )
        for index, security_id in enumerate(_SECURITY_IDS)
    )
    tickers = tuple(
        SourcedTickerHistoryRecord(
            security_id=security_id,
            ticker=f"Q{index:02d}",
            valid_from_ms=listing_at_ms,
            valid_to_ms=None,
            available_at_ms=listing_at_ms,
            primary_exchange="XNYS",
            source_receipt_sha256=_digest((security_id, "ticker")),
        )
        for index, security_id in enumerate(_SECURITY_IDS)
    )
    listings = tuple(
        ListingEventRecord(
            event_id=f"LIST-{security_id}",
            security_id=security_id,
            effective_at_ms=listing_at_ms,
            available_at_ms=listing_at_ms,
            primary_exchange="XNYS",
            ticker=f"Q{index:02d}",
            source_receipt_sha256=_digest((security_id, "listing")),
        )
        for index, security_id in enumerate(_SECURITY_IDS)
    )
    ranks = tuple(
        UniverseRankInputRecord(
            security_id=security_id,
            effective_at_ms=effective.regular_open_ns // 1_000_000,
            effective_session_index=effective_index,
            available_at_ms=effective.regular_open_ns // 1_000_000 - 1,
            observation_start_ms=(
                sessions.sessions[start_index].regular_close_ns // 1_000_000
            ),
            observation_end_ms=(
                sessions.sessions[end_index].regular_close_ns // 1_000_000
            ),
            observation_start_session_index=start_index,
            observation_end_session_index=end_index,
            observed_session_count=rule.ranking_lookback_sessions,
            average_dollar_volume=100_000_000.0 - index,
            close_price=100.0 + index,
            source_receipt_sha256=_digest((security_id, "rank")),
        )
        for index, security_id in enumerate(_SECURITY_IDS)
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=(),
        rank_inputs=ranks,
    )


def _publish_partition_source(
    *, root: Path, name: str, dataset: str, schema_sha256: str
):
    relative = f"qualification-source/{name}.jsonl"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes({"source": name})),
        root=root,
        relative_payload_path=relative,
        dataset_id=dataset,
        source_object_key=relative,
        requested_at_ms=1_000,
        downloaded_at_ms=1_000,
        schema_sha256=schema_sha256,
        entitlement_receipt_sha256=_ENTITLEMENT,
        committed_at_ms=1_000,
    )
    return load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=1_000
    )


def _partitions(
    *, root: Path, sessions: MassiveSessionAuthority, identity: PITSecurityUniverseAuthority
) -> tuple[MassivePersistedPartitionManifestV1, ...]:
    event_source = _publish_partition_source(
        root=root,
        name="event-timeline",
        dataset=MASSIVE_PERSISTED_EVENTS_DATASET_V1,
        schema_sha256=MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
    )
    active_source = _publish_partition_source(
        root=root,
        name="active-regular",
        dataset=MASSIVE_PERSISTED_ACTIVE_DATASET_V1,
        schema_sha256=MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
    )
    correction_source = _publish_partition_source(
        root=root,
        name="correction-timeline",
        dataset=MASSIVE_PERSISTED_CORRECTIONS_DATASET_V1,
        schema_sha256=MASSIVE_PERSISTED_CORRECTION_SCHEMA_SHA256,
    )
    partition_body = {
        "security_id": _SECURITY_IDS[0],
        "event_timeline": event_source,
        "active_regular": active_source,
        "correction_timeline": correction_source,
        "event_row_count": 1,
        "active_regular_row_count": 1,
        "correction_event_count": 0,
        "event_inventory_sha256": _digest("event-inventory"),
        "active_inventory_sha256": _digest("active-inventory"),
        "correction_inventory_sha256": _digest("correction-inventory"),
    }
    provisional_partition = MassivePersistedSecurityPartitionV1(
        **partition_body, receipt_sha256="0" * 64
    )
    security_partition = replace(
        provisional_partition,
        receipt_sha256=semantic_sha256(provisional_partition.unsigned()),
    )
    security_partition.validate()
    result = []
    for session in sessions.sessions:
        body = {
            "source_session_date": session.session_date,
            "source_file_scan_receipt_sha256": _digest(
                (session.session_date, "scan")
            ),
            "semantic_partition_manifest_receipt_sha256": _digest(
                (session.session_date, "semantic-partition")
            ),
            "identity_authority_receipt_sha256": identity.receipt_sha256,
            "correction_authority_receipt_sha256": _digest("corrections"),
            "partition_spec_sha256": MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
            "partition_source_sha256": MASSIVE_PERSISTED_PARTITION_SOURCE_SHA256,
            "partitions": (security_partition,),
            "source_row_count": 1,
            "persisted_event_row_count": 1,
            "active_event_key_count": 1,
            "correction_event_count": 0,
            "security_partition_count": 1,
            "partition_inventory_sha256": semantic_sha256(
                (security_partition.receipt_sha256,)
            ),
        }
        provisional_manifest = MassivePersistedPartitionManifestV1(
            **body, receipt_sha256="0" * 64
        )
        manifest = replace(
            provisional_manifest,
            receipt_sha256=semantic_sha256(provisional_manifest.unsigned()),
        )
        manifest.validate()
        result.append(manifest)
    return tuple(result)


def _daily_input(
    *,
    sessions: MassiveSessionAuthority,
    identity: PITSecurityUniverseAuthority,
    conditions: object,
    partitions: tuple[MassivePersistedPartitionManifestV1, ...],
) -> MassiveProfitabilityDailyInputAuthorityV1:
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    partition_by_date = {row.source_session_date: row for row in partitions}
    rows = []
    session_rows = []
    for date_index, session in enumerate(sessions.sessions):
        current_rows = []
        for security_index, security_id in enumerate(_SECURITY_IDS):
            bars = [1.0] * len(MASSIVE_DAILY_BARS_V0_FIELDS)
            bars[close_index] = 100.0 + security_index + 0.01 * date_index
            # Keep compiler capacity comfortably above the slowly accumulated
            # benchmark book.  Actual executions remain limited to one share
            # per name by the independent qualifying-fill volume below.
            bars[dollar_index] = 100_000_000.0
            body = {
                "source_session_date": session.session_date,
                "security_id": security_id,
                "bars_values": tuple(bars),
                "bars_valid": (True,) * len(bars),
                "tape_values": (1.0,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS),
                "tape_valid": (True,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS),
                "signed_dollar_flow": 0.0,
                "same_population_dollar_volume": 100_000_000.0,
                "absolute_signed_flow_imbalance": 0.0,
                "same_population_valid": True,
                "regular_session_event_count": 1,
                "replacement_event_count": 0,
                "cancellation_event_count": 0,
                "late_report_event_count": 0,
                "daily_bar_row_receipt_sha256": _digest(
                    (session.session_date, security_id, "bar")
                ),
                "daily_tape_row_receipt_sha256": _digest(
                    (session.session_date, security_id, "tape")
                ),
                "tape_population_row_receipt_sha256": _digest(
                    (session.session_date, security_id, "population")
                ),
                "persisted_partition_receipt_sha256": partition_by_date[
                    session.session_date
                ].receipt_sha256,
            }
            row = MassiveProfitabilityDailySecurityInputV1(
                **body, receipt_sha256=semantic_sha256(body)
            )
            row.validate()
            rows.append(row)
            current_rows.append(row)
        open_ms = session.regular_open_ns // 1_000_000
        close_ms = session.regular_close_ns // 1_000_000
        session_body = {
            "source_session_date": session.session_date,
            "regular_open_at_ms": open_ms,
            "regular_close_at_ms": close_ms,
            "vendor_last_modified_at_ms": close_ms,
            "authenticated_get_completed_at_ms": close_ms,
            "authenticated_download_receipt_sha256": _digest(
                (session.session_date, "download")
            ),
            "whole_file_scan_receipt_sha256": _digest(
                (session.session_date, "whole-file")
            ),
            "semantic_partition_manifest_receipt_sha256": _digest(
                (session.session_date, "semantic-partition")
            ),
            "persisted_partition_manifest_receipt_sha256": partition_by_date[
                session.session_date
            ].receipt_sha256,
            "daily_bars_artifact_receipt_sha256": _digest(
                (session.session_date, "bars")
            ),
            "daily_tape_artifact_receipt_sha256": _digest(
                (session.session_date, "daily-tape")
            ),
            "supported_security_row_inventory_sha256": semantic_sha256(
                tuple(row.receipt_sha256 for row in current_rows)
            ),
        }
        session_row = MassiveProfitabilityDailyInputSessionV1(
            **session_body, receipt_sha256=semantic_sha256(session_body)
        )
        session_row.validate()
        session_rows.append(session_row)
    acquisition = _digest("daily-acquisition")
    body = {
        "schema": MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
        "coverage_start_session_date": sessions.sessions[0].session_date,
        "coverage_end_session_date": sessions.sessions[-1].session_date,
        "data_freeze_at_ms": session_rows[-1].authenticated_get_completed_at_ms,
        "supported_security_ids": _SECURITY_IDS,
        "sessions": tuple(session_rows),
        "rows": tuple(rows),
        "archive_freeze_semantic_receipt_sha256": _digest("archive-freeze"),
        "security_support_semantic_receipt_sha256": _digest("security-support"),
        "session_authority_receipt_sha256": sessions.receipt_sha256,
        "normalized_identity_semantic_receipt_sha256": (
            massive_profitability_identity_semantic_receipt_v2(identity)
        ),
        "condition_authority_receipt_sha256": cast(Any, conditions).receipt_sha256,
        "correction_authority_receipt_sha256": _digest("corrections"),
        "event_domain_spec_receipt_sha256": _digest("event-domain"),
        "session_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in session_rows)
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "source_transport_qualified": True,
        "daily_input_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256
        ),
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    provisional = MassiveProfitabilityDailyInputAuthorityV1(
        **body,
        semantic_receipt_sha256="0" * 64,
        acquisition_audit_receipt_sha256=acquisition,
        audit_receipt_sha256="0" * 64,
    )
    semantic = semantic_sha256(provisional.semantic_unsigned())
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "acquisition_audit_receipt_sha256": acquisition,
            }
        ),
    )
    result.validate()
    return result


def _fill_source(
    *,
    sessions: MassiveSessionAuthority,
    conditions: object,
    daily: MassiveProfitabilityDailyInputAuthorityV1,
) -> MassiveAdaptiveFillSourceV1:
    rows = []
    daily_sessions = {row.source_session_date: row for row in daily.sessions}
    for date_index, session in enumerate(sessions.sessions):
        start, end = adaptive_fill_clock_v1(session.session_date)
        for security_index, security_id in enumerate(_SECURITY_IDS):
            price = 100.005 + security_index + 0.01 * date_index
            body = {
                "session_date": session.session_date,
                "security_id": security_id,
                "fill_start_at_ms": start,
                "fill_end_at_ms": end,
                "fill_vwap": price,
                "qualifying_share_volume": 50.0,
                "qualifying_dollar_volume": price * 50.0,
                "qualifying_trade_count": 10,
                "valid": True,
                "qualifying_trade_inventory_sha256": _digest(
                    (session.session_date, security_id, "fills")
                ),
                "persisted_partition_receipt_sha256": daily_sessions[
                    session.session_date
                ].persisted_partition_manifest_receipt_sha256,
                "daily_input_row_receipt_sha256": daily.row(
                    session_date=session.session_date,
                    security_id=security_id,
                ).receipt_sha256,
            }
            row = MassiveAdaptiveFillRowV1(
                **body, receipt_sha256=semantic_sha256(body)
            )
            row.validate()
            rows.append(row)
    body = {
        "schema": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SCHEMA,
        "session_dates": tuple(row.session_date for row in sessions.sessions),
        "supported_security_ids": _SECURITY_IDS,
        "rows": tuple(rows),
        "daily_input_authority_semantic_receipt_sha256": (
            daily.semantic_receipt_sha256
        ),
        "session_authority_receipt_sha256": sessions.receipt_sha256,
        "condition_authority_receipt_sha256": cast(Any, conditions).receipt_sha256,
        "persisted_manifest_inventory_sha256": semantic_sha256(
            tuple(
                row.persisted_partition_manifest_receipt_sha256
                for row in daily.sessions
            )
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "source_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SOURCE_SHA256,
        "source_paths_replayed": True,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    provisional = MassiveAdaptiveFillSourceV1(
        **body,
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256="0" * 64,
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


@dataclass(frozen=True)
class _PredictorRoots:
    features: Mapping[str, MassiveProfitabilityOriginFeaturesV3]
    actions: Mapping[str, MassiveAdaptiveOriginAuthorityV1]
    contexts: Mapping[str, MassiveAdaptiveContextOriginAuthorityV1]
    decisions: Mapping[str, MassiveAdaptiveDecisionRootV1]
    clocks: Mapping[str, MassiveDecisionClockAuthority]


def _predictor_roots(
    *,
    sessions: MassiveSessionAuthority,
    identity: PITSecurityUniverseAuthority,
    daily: MassiveProfitabilityDailyInputAuthorityV1,
    required_dates: Iterable[str],
) -> _PredictorRoots:
    candidate_dates = tuple(row.session_date for row in sessions.sessions)
    sessions_by_date = {row.session_date: row for row in sessions.sessions}
    dates = tuple(sorted(set(required_dates)))
    clocks = build_massive_decision_clock_authorities(
        session_authority=sessions,
        sessions=tuple(sessions_by_date[value] for value in dates),
    )
    clock_by_date = {row.session_date: row for row in clocks}
    features: dict[str, MassiveProfitabilityOriginFeaturesV3] = {}
    actions: dict[str, MassiveAdaptiveOriginAuthorityV1] = {}
    contexts: dict[str, MassiveAdaptiveContextOriginAuthorityV1] = {}
    decisions: dict[str, MassiveAdaptiveDecisionRootV1] = {}
    for session_date in dates:
        date_index = candidate_dates.index(session_date)
        feature, _unused = _feature_and_target(
            decision_session_date=session_date,
            source_session_date=candidate_dates[date_index - 1],
            input_session_dates=candidate_dates[date_index - 64 : date_index],
            date_index=date_index,
        )
        feature_rows = list(feature.rows[: len(_SECURITY_IDS)])
        for security_index, security_id in enumerate(
            _SECURITY_IDS[len(feature_rows) :], start=len(feature_rows)
        ):
            provisional_row = replace(
                feature.rows[security_index % len(feature.rows)],
                security_id=security_id,
                decision_membership_rank=security_index + 1,
                receipt_sha256="0" * 64,
            )
            feature_rows.append(
                replace(
                    provisional_row,
                    receipt_sha256=semantic_sha256(provisional_row.unsigned()),
                )
            )
        feature = _reseal(
            feature,
            rows=tuple(feature_rows),
            row_inventory_sha256=semantic_sha256(
                tuple(row.receipt_sha256 for row in feature_rows)
            ),
        )
        feature = _reseal(
            feature,
            daily_input_authority_semantic_receipt_sha256=(
                daily.semantic_receipt_sha256
            ),
        )
        clock = clock_by_date[session_date]
        action = _origin(
            feature,
            action_ids=_SECURITY_IDS,
            session_authority_receipt_sha256=sessions.receipt_sha256,
        )
        action = _reseal(
            action,
            decision_at_ms=clock.decision_at_ns // 1_000_000,
            exposure_panel=replace(
                action.exposure_panel,
                origin_at_ms=clock.decision_at_ns // 1_000_000,
                available_at_ms=clock.decision_at_ns // 1_000_000,
            ),
            decision_clock_receipt_sha256=clock.receipt_sha256,
        )
        context = build_massive_adaptive_context_origin_authority_v1(
            decision_clock=clock,
            session_authority=sessions,
            identity_authority=identity,
            features=feature,
        )
        decision = build_massive_adaptive_decision_root_v1(
            context_origin=context,
            action_origin=action,
            features=feature,
        )
        features[session_date] = feature
        actions[session_date] = action
        contexts[session_date] = context
        decisions[session_date] = decision
    return _PredictorRoots(
        features=features,
        actions=actions,
        contexts=contexts,
        decisions=decisions,
        clocks=clock_by_date,
    )


def _target_for_training_date(
    *,
    action: MassiveAdaptiveOriginAuthorityV1,
    identity: PITSecurityUniverseAuthority,
    daily: MassiveProfitabilityDailyInputAuthorityV1,
    fills: MassiveAdaptiveFillSourceV1,
) -> MassiveAdaptiveSourceTargetsV1:
    source_target = _source_target(action)
    target = replace(
        source_target.targets,
        fill_source_receipt_sha256=fills.semantic_receipt_sha256,
        semantic_receipt_sha256="0" * 64,
    )
    target = replace(
        target,
        semantic_receipt_sha256=semantic_sha256(target.semantic_unsigned()),
    )
    target.validate()
    result = replace(
        source_target,
        targets=target,
        identity_authority_receipt_sha256=identity.receipt_sha256,
        daily_input_authority_receipt_sha256=daily.semantic_receipt_sha256,
        fill_source_receipt_sha256=fills.semantic_receipt_sha256,
        target_receipt_sha256=target.semantic_receipt_sha256,
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        result,
        semantic_receipt_sha256=semantic_sha256(result.semantic_unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class _SupervisedLineage:
    window: MassiveAdaptiveWindowPlanV1
    checkpoint: MassiveAdaptiveCheckpointV1
    choice: MassiveAdaptiveCausalCheckpointChoiceV1
    calibration: MassiveAdaptiveForecastCalibrationV2
    model_spec: MassiveAdaptiveAlphaModelSpecV1


def _training_lineages(
    *,
    root: Path,
    sessions: MassiveSessionAuthority,
    identity: PITSecurityUniverseAuthority,
    split_plan: MassiveAdaptiveSplitPlanV1,
    daily: MassiveProfitabilityDailyInputAuthorityV1,
    fills: MassiveAdaptiveFillSourceV1,
    roots: _PredictorRoots,
    objects: dict[str, object],
) -> tuple[_SupervisedLineage, ...]:
    candidate_dates = split_plan.candidate_session_dates
    training_dates = candidate_dates[200:201]
    features = tuple(roots.features[value] for value in training_dates)
    actions = tuple(roots.actions[value] for value in training_dates)
    contexts = tuple(roots.contexts[value] for value in training_dates)
    decisions = tuple(roots.decisions[value] for value in training_dates)
    targets = tuple(
        _target_for_training_date(
            action=action,
            identity=identity,
            daily=daily,
            fills=fills,
        )
        for action in actions
    )
    committed_tensor = materialize_massive_adaptive_decision_tensor_v1(
        root=root,
        artifact_id="qualification-supervised-training",
        features=features,
        action_origins=actions,
        committed_at_ms=20_000,
    )
    training_tensor = authorize_massive_adaptive_decision_tensor_v1(
        root=root,
        tensor=parse_massive_adaptive_decision_tensor_v1(
            root=root, loaded_source=committed_tensor.loaded_source
        ),
        features=features,
        action_origins=actions,
    )
    model_spec = replace(
        MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
        token_dimension=8,
        fast_window_sessions=1,
        maximum_context_sessions=_MODEL_CONTEXT_SESSIONS,
        maximum_intraday_intervals=1,
        market_latent_count=2,
        attention_heads=2,
        dropout_probability=0.0,
    )
    model_spec.validate()
    config = replace(
        MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1,
        seed=5,
        scheduler_total_updates=8,
    )
    config.validate()
    lineages = []
    for source_fold_index in range(4):
        prepared = supervised._prepare_training(
            root=root,
            artifact_id=f"qualification-supervised-{source_fold_index}",
            decision_tensor=training_tensor,
            features=features,
            context_origins=contexts,
            action_origins=actions,
            source_targets=targets,
            target_archive=None,
            target_source_runtimes=(),
            session_authority=sessions,
            split_plan=replace(
                split_plan,
                candidate_source_data_qualified=False,
                semantic_receipt_sha256=semantic_sha256(
                    replace(
                        split_plan,
                        candidate_source_data_qualified=False,
                        semantic_receipt_sha256="0" * 64,
                    ).semantic_unsigned()
                ),
            ),
            fold_index=source_fold_index,
            split_role="training",
            archive_freeze=None,
        )
        window = _reseal(
            prepared.window_plan,
            split_plan_receipt_sha256=split_plan.semantic_receipt_sha256,
        )
        target_archive = _reseal(
            prepared.target_archive,
            committed_source_data_qualified=True,
            development_training_authorized=True,
        )
        training_authority = build_massive_adaptive_training_authority_v1(
            decision_tensor=prepared.decision_tensor,
            decision_roots=prepared.decision_roots,
            target_archive=target_archive,
            split_plan=split_plan,
            window_plan=window,
        )
        qualified_prepared = replace(
            prepared,
            target_archive=target_archive,
            split_plan=split_plan,
            window_plan=window,
            training_authority=training_authority,
        )
        checkpoint = supervised._run_and_publish(
            root=root,
            artifact_id=f"qualification-supervised-{source_fold_index}",
            prepared=qualified_prepared,
            model_spec=model_spec,
            config=config,
            updates=1,
            committed_at_ms=30_000 + source_fold_index,
            resume_checkpoint=None,
            require_authorized=False,
        )
        training_forecast = materialize_massive_adaptive_forecast_archive_v1(
            root=root,
            artifact_id=f"qualification-training-forecast-{source_fold_index}",
            checkpoint=checkpoint,
            decision_tensor=training_tensor,
            decision_roots=decisions,
            window_plan=window,
            model_spec=model_spec,
            committed_at_ms=40_000 + source_fold_index,
        )
        calibration = materialize_massive_adaptive_forecast_calibration_v2(
            root=root,
            artifact_id=f"qualification-calibration-{source_fold_index}",
            checkpoint=checkpoint,
            training_forecasts=training_forecast,
            training_targets=target_archive,
            training_window_plan=window,
            committed_at_ms=50_000 + source_fold_index,
        )
        choice = build_massive_adaptive_causal_checkpoint_choice_v1(
            checkpoints=(checkpoint,), training_window_plan=window
        )
        lineages.append(
            _SupervisedLineage(
                window=window,
                checkpoint=checkpoint,
                choice=choice,
                calibration=calibration,
                model_spec=model_spec,
            )
        )
        for value in (
            training_authority,
            target_archive,
            *target_archive.runtime_target_roots,
            *targets,
            checkpoint,
            window,
            choice,
            training_forecast,
            calibration,
        ):
            objects[reconstruction._receipt(value)] = value
    for value in (
        training_tensor,
        model_spec,
        config,
        *features,
        *actions,
        *contexts,
        *decisions,
        *roots.clocks.values(),
    ):
        objects[reconstruction._receipt(value)] = value
    return tuple(lineages)


def _required_predictor_dates(split_plan: MassiveAdaptiveSplitPlanV1) -> tuple[str, ...]:
    candidates = split_plan.candidate_session_dates
    result = set(candidates[200:201])
    for fold_index, fold in enumerate(split_plan.outer_folds):
        fit_count = 126 * (fold_index + 1)
        result.update(fold.fit_session_dates[-fit_count:])
        for role_dates in (
            fold.inner_validation_session_dates,
            fold.outer_test_session_dates,
        ):
            start = candidates.index(role_dates[0])
            stop = candidates.index(role_dates[-1]) + 1
            result.update(
                candidates[
                    start - MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1 + 1 : stop
                ]
            )
    return tuple(sorted(result))


def _fit_archives(
    *,
    root: Path,
    split_plan: MassiveAdaptiveSplitPlanV1,
    roots: _PredictorRoots,
    lineages: tuple[_SupervisedLineage, ...],
    objects: dict[str, object],
    block_sessions: int,
) -> tuple[
    tuple[MassiveAdaptiveRLFitForecastArchiveV1, ...], ...
]:
    result = []
    for outer_fold_index, fold in enumerate(split_plan.outer_folds):
        fit_count = 126 * (outer_fold_index + 1)
        fit_dates = fold.fit_session_dates[-fit_count:]
        features = tuple(roots.features[value] for value in fit_dates)
        actions = tuple(roots.actions[value] for value in fit_dates)
        decisions = tuple(roots.decisions[value] for value in fit_dates)
        committed_tensor = materialize_massive_adaptive_decision_tensor_v1(
            root=root,
            artifact_id=f"qualification-fit-tensor-{outer_fold_index}",
            features=features,
            action_origins=actions,
            committed_at_ms=60_000 + outer_fold_index,
        )
        tensor = authorize_massive_adaptive_decision_tensor_v1(
            root=root,
            tensor=parse_massive_adaptive_decision_tensor_v1(
                root=root, loaded_source=committed_tensor.loaded_source
            ),
            features=features,
            action_origins=actions,
        )
        objects[reconstruction._receipt(tensor)] = tensor
        archives = []
        block_count = fit_count // block_sessions
        blocks_per_source = 126 // block_sessions
        for block_index in range(block_count):
            lineage = lineages[block_index // blocks_per_source]
            plan = build_massive_adaptive_rl_fit_inference_plan_v1(
                decision_tensor=tensor,
                decision_roots=decisions,
                split_plan=split_plan,
                outer_fold_index=outer_fold_index,
                block_index=block_index,
                block_sessions=block_sessions,
                model_spec=lineage.model_spec,
            )
            archive = materialize_massive_adaptive_rl_fit_forecast_archive_v1(
                root=root,
                artifact_id=f"qualification-fit-{outer_fold_index}-{block_index}",
                checkpoint=lineage.checkpoint,
                training_window_plan=lineage.window,
                inference_tensor=tensor,
                inference_decision_roots=decisions,
                inference_plan=plan,
                split_plan=split_plan,
                model_spec=lineage.model_spec,
                committed_at_ms=70_000 + outer_fold_index * 100 + block_index,
            )
            archives.append(archive)
            objects[reconstruction._receipt(plan)] = plan
            objects[reconstruction._receipt(archive)] = archive
        result.append(tuple(archives))
    return tuple(result)


def _inventory(
    *, role: str, fold_index: int | None, items: Iterable[object]
) -> MassiveAdaptiveRLTypedAuthorityInventoryV1:
    return build_massive_adaptive_rl_typed_authority_inventory_v1(
        role=role,
        fold_index=fold_index,
        items=cast(tuple[Any, ...], tuple(items)),
    )


def _bound(
    *, role: str, fold_index: int | None, authority: object
) -> MassiveAdaptiveRLRoleBoundSourceAuthorityV1:
    return bind_massive_adaptive_rl_source_authority_v1(
        role=role,
        fold_index=fold_index,
        authority=cast(Any, authority),
        source_data_qualified=True,
        runtime_source_replayed=True,
    )


def _persist_fixed_source_artifacts(
    *, root: Path, runtime_sources: Mapping[tuple[str, int | None], object]
) -> None:
    from rl_quant.workflows import massive_adaptive_rl_source_bundle_v1 as source_bundle

    paths = source_bundle._expected_paths()
    for key, relative in paths.items():
        bound = cast(MassiveAdaptiveRLRoleBoundSourceAuthorityV1, runtime_sources[key])
        authority = bound.authority
        if isinstance(authority, MassiveAdaptiveRLTypedAuthorityInventoryV1):
            payload = {
                **authority.semantic_unsigned(),
                "semantic_receipt_sha256": authority.semantic_receipt_sha256,
            }
        else:
            receipt_name = (
                "semantic_receipt_sha256"
                if hasattr(authority, "semantic_receipt_sha256")
                else "receipt_sha256"
            )
            payload = {receipt_name: getattr(authority, receipt_name)}
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_file_bytes(payload))


def _replay_dependencies(
    *,
    runtime_sources: Mapping[tuple[str, int | None], object],
    objects: Mapping[str, object],
) -> tuple[object, ...]:
    from rl_quant.workflows import (
        massive_adaptive_rl_runtime_source_graph_authority_v1 as graph_module,
    )

    primary_rows = []
    for role, specification in MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1.items():
        for fold_index in range(4) if specification.fold_scoped else (None,):
            bound = cast(
                MassiveAdaptiveRLRoleBoundSourceAuthorityV1,
                runtime_sources[(role, fold_index)],
            )
            authority = bound.authority
            if isinstance(authority, MassiveAdaptiveRLTypedAuthorityInventoryV1):
                if authority.runtime_items is None:
                    raise AssertionError("qualification runtime inventory is absent")
                primary_rows.extend(
                    (
                        role,
                        fold_index,
                        graph_module._item_logical_key(role=role, item=item),
                        authority.semantic_receipt_sha256,
                        item,
                    )
                    for item in authority.runtime_items
                )
            else:
                primary_rows.append(
                    (
                        role,
                        fold_index,
                        "root",
                        reconstruction._receipt(authority),
                        authority,
                    )
                )
    primary = tuple(primary_rows)
    primary_receipts = {reconstruction._receipt(row[4]) for row in primary}
    decisions: dict[str, MassiveAdaptiveDecisionRootV1] = {}
    for value in objects.values():
        if isinstance(value, MassiveAdaptiveDecisionRootV1):
            previous = decisions.setdefault(value.decision_session_date, value)
            if previous.semantic_receipt_sha256 != value.semantic_receipt_sha256:
                raise AssertionError("qualification decision roots diverged")
    queue = list(primary_receipts)
    visited: set[str] = set()
    while queue:
        receipt = queue.pop()
        if receipt in visited:
            continue
        visited.add(receipt)
        value = objects.get(receipt)
        if value is None:
            raise AssertionError(f"qualification dependency is absent: {receipt}")
        expected = reconstruction._expected_dependencies(
            value=value,
            objects_by_receipt=objects,
            primary_decisions_by_date=decisions,
        )
        for dependency_receipt, expected_type in expected.items():
            dependency = objects.get(dependency_receipt)
            if dependency is None or type(dependency) is not expected_type:
                raise AssertionError(
                    "qualification dependency type differs: "
                    f"{dependency_receipt} expected {expected_type.__name__}, "
                    f"observed {type(dependency).__name__}"
                )
            queue.append(dependency_receipt)
    return tuple(objects[value] for value in sorted(visited - primary_receipts))


def prepare_persisted_v5_source_package(
    *, root: Path, manifest: MassiveAdaptiveRLExperimentManifestV5
) -> None:
    sessions = _sessions()
    conditions = _conditions()
    identity = _context_identity(sessions)
    partitions = _partitions(root=root, sessions=sessions, identity=identity)
    daily = _daily_input(
        sessions=sessions,
        identity=identity,
        conditions=conditions,
        partitions=partitions,
    )
    fills = _fill_source(sessions=sessions, conditions=conditions, daily=daily)
    split_plan = _reseal(
        build_massive_adaptive_split_plan_v1(
            candidate_session_dates=tuple(row.session_date for row in sessions.sessions),
            session_authority=sessions,
        ),
        candidate_source_data_qualified=True,
    )
    required_dates = _required_predictor_dates(split_plan)
    roots = _predictor_roots(
        sessions=sessions,
        identity=identity,
        daily=daily,
        required_dates=required_dates,
    )
    event_root = root / "qualification-economic-events"
    event_root.mkdir()
    events = _empty_event_archive(
        event_root,
        identity=identity,
        observed_at_ms=sessions.sessions[-1].regular_close_ns // 1_000_000 + 1,
    )
    objects: dict[str, object] = {}
    lineages = _training_lineages(
        root=root,
        sessions=sessions,
        identity=identity,
        split_plan=split_plan,
        daily=daily,
        fills=fills,
        roots=roots,
        objects=objects,
    )
    block_sessions = manifest.base_manifest.base_manifest.base_manifest.prequential_block_sessions
    archives_by_fold = _fit_archives(
        root=root,
        split_plan=split_plan,
        roots=roots,
        lineages=lineages,
        objects=objects,
        block_sessions=block_sessions,
    )
    runtime_sources: dict[tuple[str, int | None], object] = {}
    direct = {
        "session-authority": sessions,
        "condition-authority": conditions,
        "identity-authority": identity,
        "economic-event-archive": events,
        "daily-input-authority": daily,
        "fill-source-authority": fills,
        "split-plan": split_plan,
    }
    for role, authority in direct.items():
        runtime_sources[(role, None)] = _bound(
            role=role, fold_index=None, authority=authority
        )
        objects[reconstruction._receipt(authority)] = authority
    partition_inventory = _inventory(
        role="persisted-partition-inventory", fold_index=None, items=partitions
    )
    runtime_sources[("persisted-partition-inventory", None)] = _bound(
        role="persisted-partition-inventory",
        fold_index=None,
        authority=partition_inventory,
    )
    for value in partitions:
        objects[reconstruction._receipt(value)] = value

    candidate_dates = split_plan.candidate_session_dates
    development_date_set: set[str] = set()
    for fold in split_plan.outer_folds:
        start = candidate_dates.index(fold.outer_test_session_dates[0])
        stop = candidate_dates.index(fold.outer_test_session_dates[-1]) + 1
        development_date_set.update(
            candidate_dates[
                start - MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1 + 1 : stop
            ]
        )
    development_dates = tuple(
        value for value in candidate_dates if value in development_date_set
    )
    for role, values in (
        (
            "development-origin-feature-inventory",
            tuple(roots.features[value] for value in development_dates),
        ),
        (
            "development-origin-action-inventory",
            tuple(roots.actions[value] for value in development_dates),
        ),
    ):
        inventory = _inventory(role=role, fold_index=None, items=values)
        runtime_sources[(role, None)] = _bound(
            role=role, fold_index=None, authority=inventory
        )

    for outer_fold_index, fold in enumerate(split_plan.outer_folds):
        fit_count = 126 * (outer_fold_index + 1)
        fit_dates = fold.fit_session_dates[-fit_count:]
        validation_start = candidate_dates.index(fold.inner_validation_session_dates[0])
        validation_stop = (
            candidate_dates.index(fold.inner_validation_session_dates[-1]) + 1
        )
        validation_dates = candidate_dates[
            validation_start
            - MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1
            + 1 : validation_stop
        ]
        fold_items = {
            "training-window-inventory": tuple(
                row.window for row in lineages[: outer_fold_index + 1]
            ),
            "supervised-checkpoint-inventory": tuple(
                row.choice for row in lineages[: outer_fold_index + 1]
            ),
            "calibration-inventory": tuple(
                row.calibration for row in lineages[: outer_fold_index + 1]
            ),
            "fit-forecast-archive-inventory": archives_by_fold[outer_fold_index],
            "decision-root-inventory": tuple(
                roots.decisions[value] for value in fit_dates
            ),
            "context-origin-inventory": tuple(
                roots.contexts[value] for value in fit_dates
            ),
            "validation-origin-feature-inventory": tuple(
                roots.features[value] for value in validation_dates
            ),
            "validation-origin-action-inventory": tuple(
                roots.actions[value] for value in validation_dates
            ),
        }
        for role, values in fold_items.items():
            inventory = _inventory(
                role=role, fold_index=outer_fold_index, items=values
            )
            runtime_sources[(role, outer_fold_index)] = _bound(
                role=role, fold_index=outer_fold_index, authority=inventory
            )
    for value in (
        *roots.features.values(),
        *roots.actions.values(),
        *roots.contexts.values(),
        *roots.decisions.values(),
        *roots.clocks.values(),
    ):
        objects[reconstruction._receipt(value)] = value
    _persist_fixed_source_artifacts(root=root, runtime_sources=runtime_sources)
    manifest_v3 = manifest.base_manifest.base_manifest
    manifest_v2 = manifest_v3.base_manifest
    materialize_massive_adaptive_rl_source_bundle_v1(
        source_root=root,
        manifest=manifest_v2,
        runtime_sources=cast(Any, runtime_sources),
    )
    loaded_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=root, manifest=manifest_v2
    )
    authorized_bundle = authorize_massive_adaptive_rl_source_bundle_v1(
        source_bundle=loaded_bundle,
        runtime_sources=cast(Any, runtime_sources),
    )
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=root,
        manifest=manifest_v3,
        source_bundle=authorized_bundle,
        runtime_sources=cast(Any, runtime_sources),
    )
    loaded_graph = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=root,
        manifest=manifest_v3,
        source_bundle=authorized_bundle,
    )
    graph = authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        authority=loaded_graph,
        source_bundle=authorized_bundle,
        runtime_sources=cast(Any, runtime_sources),
    )
    dependencies = _replay_dependencies(
        runtime_sources=runtime_sources,
        objects=objects,
    )
    reconstruction.materialize_massive_adaptive_rl_replay_dependency_index_v1(
        source_root=root,
        manifest=manifest_v3,
        runtime_source_graph_authority=graph,
        replay_dependencies=dependencies,
    )


@dataclass(frozen=True)
class PersistedV5QualificationRun:
    manifest_path: Path
    source_root: Path
    artifact_root: Path
    first_run: MassiveAdaptiveRLPrequentialRunV5
    completed_run: MassiveAdaptiveRLPrequentialRunV5
    verified_run: MassiveAdaptiveRLPrequentialRunV5
    fresh_process_verified_result: dict[str, Any]


def _persisted_evidence_inventory(roots: tuple[Path, ...]) -> tuple[object, ...]:
    rows: list[object] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise AssertionError(f"synthetic evidence contains a symlink: {path}")
            content = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            rows.append((str(root), path.relative_to(root).as_posix(), content))
    return tuple(rows)


def run_persisted_v5_qualification(root: Path) -> PersistedV5QualificationRun:
    for name, value in MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT:
        os.environ[name] = value
    configure_massive_adaptive_rl_deterministic_runtime_v1()
    experiment_id = "v5-vertical-qualification-persisted"
    ppo_config = MassiveAdaptivePPOConfigV1(
        epochs_per_rollout=1,
        rollout_length=63,
        minibatch_size=63,
        seed=17,
    )
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id=experiment_id,
        prequential_block_sessions=63,
        ppo_config=ppo_config,
    )
    manifest_path = root / "manifest-v5.json"
    source_root = root / "source"
    artifact_root = root / "artifacts"
    source_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    write_massive_adaptive_rl_experiment_manifest_v5(
        path=manifest_path, manifest=manifest
    )
    print("persisted V5: committing synthetic sources", flush=True)
    prepare_persisted_v5_source_package(root=source_root, manifest=manifest)
    with massive_adaptive_rl_vertical_qualification_scope_v1():
        print("persisted V5: starting authoritative causal training", flush=True)
        first = run_massive_adaptive_rl_experiment_v5(
            manifest_path=manifest_path,
            source_root=source_root,
            artifact_root=artifact_root,
            device="cpu",
            resume=False,
        )
        if not isinstance(first, MassiveAdaptiveRLPrequentialRunV5):
            raise AssertionError(f"V5 training did not reach its boundary: {first!r}")
        print("persisted V5: training complete; registering implementation", flush=True)
        registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
            root=artifact_root,
            manifest=manifest,
            allow_materialize=False,
        )
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=artifact_root,
            manifest=manifest,
            manifest_registration=registration,
            allow_materialize=True,
        )
        print("persisted V5: executing validation, outer folds, and report", flush=True)
        completed = run_massive_adaptive_rl_experiment_v5(
            manifest_path=manifest_path,
            source_root=source_root,
            artifact_root=artifact_root,
            device="cpu",
            resume=True,
        )
        print("persisted V5: replaying completed experiment", flush=True)
        verified = verify_massive_adaptive_rl_experiment_v5(
            manifest_path=manifest_path,
            source_root=source_root,
            artifact_root=artifact_root,
            device="cpu",
        )
    if not isinstance(completed, MassiveAdaptiveRLPrequentialRunV5) or not isinstance(
        verified, MassiveAdaptiveRLPrequentialRunV5
    ):
        raise AssertionError("V5 qualification did not return its stable result envelope")
    print("persisted V5: verifying through CLI in a fresh process", flush=True)
    roots = (source_root, artifact_root)
    before = _persisted_evidence_inventory(roots)
    child_environment = dict(os.environ)
    child_environment[MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_ENVIRONMENT_V1] = "1"
    # Exercise the installed launcher's re-exec and deterministic startup, not
    # inherited notebook/PyTorch state. The reserved namespace remains unable
    # to issue positive production authorization.
    child_environment.pop("QUANTTRADE_ADAPTIVE_RL_RUNTIME_V1", None)
    child = subprocess.run(
        (
            sys.executable,
            "-m",
            "rl_quant.workflows.massive_adaptive_rl_cli_v1",
            "verify",
            "--manifest", str(manifest_path),
            "--source-root", str(source_root),
            "--artifact-root", str(artifact_root),
            "--device", "cpu",
        ),
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=MASSIVE_ADAPTIVE_RL_FRESH_REPLAY_TIMEOUT_SECONDS_V1,
        check=False,
    )
    if child.returncode != 0:
        raise AssertionError(
            f"fresh-process V5 verification exited {child.returncode}: "
            f"{child.stderr[-10000:]}\n{child.stdout[-10000:]}"
        )
    fresh_process_result = json.loads(child.stdout)
    assert _persisted_evidence_inventory(roots) == before
    assert fresh_process_result == json.loads(json.dumps(asdict(verified)))
    assert fresh_process_result["end_to_end_profitability_execution_complete"] is True
    print("persisted V5: fresh-process verification passed without evidence writes", flush=True)
    return PersistedV5QualificationRun(
        manifest_path=manifest_path,
        source_root=source_root,
        artifact_root=artifact_root,
        first_run=first,
        completed_run=completed,
        verified_run=verified,
        fresh_process_verified_result=fresh_process_result,
    )
