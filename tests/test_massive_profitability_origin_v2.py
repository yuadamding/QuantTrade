from __future__ import annotations

import gzip
import inspect
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.conditions import (
    MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    build_massive_condition_authority,
)
from rl_quant.data_sources.massive.corrections import (
    build_massive_correction_authority,
)
from rl_quant.data_sources.massive.finalized_daily_scan import (
    scan_massive_daily_trade_file_v0,
)
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_BUCKET,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    build_massive_daily_trade_partition_manifest_v0,
    build_massive_finalized_feature_domain_spec_v0,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    persist_massive_daily_trade_partitions_v1,
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
from rl_quant.data_sources.massive.trade_extraction import MASSIVE_FLAT_TRADE_COLUMNS
from rl_quant.features import massive_profitability_origin_v2 as origin_v2
from rl_quant.features.massive_daily_bars_v0 import (
    MASSIVE_DAILY_BARS_V0_DATASET,
    MASSIVE_DAILY_BARS_V0_FIELDS,
    MASSIVE_DAILY_BARS_V0_SCHEMA,
    MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256,
    MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
    MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
    MassiveDailyBarsArtifactV0,
    MassiveDailyBarsRowV0,
    materialize_massive_daily_bars_v0,
    validate_massive_daily_bars_v0,
)
from rl_quant.features.massive_monthly_rank_bar_authority_v1 import (
    MassiveMonthlyRankBarAuthorityV1Error,
    build_massive_monthly_rank_bar_authority_for_test_v1,
    build_massive_monthly_rank_bar_authority_v1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    massive_profitability_identity_semantic_receipt_v2,
    materialize_massive_profitability_security_support_v2,
    parse_massive_profitability_security_support_v2,
)
from rl_quant.features.massive_profitability_frozen_authorities_v1 import (
    materialize_massive_monthly_rank_bar_authority_v1,
    materialize_massive_monthly_rank_input_authority_v2,
    materialize_massive_profitability_origin_plan_v2,
    parse_massive_profitability_frozen_authority_v1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_V2_LOCKBOX_ACCESS_AUTHORIZED,
    MASSIVE_PROFITABILITY_ORIGIN_V2_PANEL_MATERIALIZATION_AUTHORIZED,
    MASSIVE_PROFITABILITY_ORIGIN_V2_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_PROFITABILITY_ORIGIN_V2_PROFITABILITY_REPORTING_AUTHORIZED,
    MassiveProfitabilityOriginV2Error,
    build_massive_monthly_rank_input_authority_v2,
    build_massive_profitability_acquisition_for_test_v2,
    build_massive_profitability_decision_origin_plan_v2,
    capture_massive_profitability_production_acquisition_v2,
    materialize_massive_profitability_acquired_source_evidence_from_acquisition_v2,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

_EASTERN = ZoneInfo("America/New_York")
_ENTITLEMENT = "a" * 64


def _ms(day: str, value: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp()
        * 1_000
    )


def _sessions() -> MassiveSessionAuthority:
    days: list[str] = []
    current = date(2020, 1, 2)
    while current <= date(2020, 5, 8):
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    source_receipt = semantic_sha256("origin-v2-calendar")
    rows = tuple(
        MassiveExchangeSession(
            session_date=day,
            exchange="XNYS",
            regular_open_ns=_ms(day, time(9, 30)) * 1_000_000,
            regular_close_ns=_ms(day, time(16, 0)) * 1_000_000,
            scheduled_five_minute_intervals=78,
            special_session_reason=None,
            calendar_source_receipt_sha256=source_receipt,
        )
        for day in days
    )
    return build_massive_session_authority(
        rows, calendar_source_receipt_sha256=source_receipt
    )


class _Paginator:
    def __init__(self, client: _S3Client) -> None:
        self.client = client

    def paginate(self, **kwargs: object) -> tuple[dict[str, object], ...]:
        assert kwargs["Bucket"] == MASSIVE_FLAT_FILE_BUCKET
        prefix = str(kwargs["Prefix"])
        contents = []
        for key, payload in sorted(self.client.payloads.items()):
            if not key.startswith(prefix):
                continue
            day = key.rsplit("/", 1)[-1].removesuffix(".csv.gz")
            contents.append(
                {
                    "Key": key,
                    "ETag": f'"etag-{day}"',
                    "Size": len(payload),
                    "LastModified": datetime.fromtimestamp(
                        _ms(day, time(18, 0)) / 1_000, tz=UTC
                    ),
                }
            )
        return (
            {
                "ResponseMetadata": {"RequestId": f"LIST-{prefix.replace('/', '-')}"},
                "IsTruncated": False,
                "Contents": contents,
            },
        )


class _S3Client:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return _Paginator(self)

    def get_object(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["Bucket"] == MASSIVE_FLAT_FILE_BUCKET
        key = str(kwargs["Key"])
        payload = self.payloads[key]
        day = key.rsplit("/", 1)[-1].removesuffix(".csv.gz")
        return {
            "ResponseMetadata": {"RequestId": f"GET-{day}"},
            "Body": BytesIO(payload),
            "ETag": f'"etag-{day}"',
            "ContentLength": len(payload),
            "VersionId": f"VERSION-{day}",
        }


def _flat_trade_payload(*, session_date: str, ordinal: int) -> bytes:
    participant = _ms(session_date, time(15, 59)) * 1_000_000
    row = (
        "AAA",
        "[1]",
        "0",
        "4",
        f"TRADE-{ordinal}",
        str(participant),
        f"{100 + ordinal}.00",
        "1",
        str(participant),
        "100000",
        "1",
        "12",
        str(participant),
    )
    text = ",".join(MASSIVE_FLAT_TRADE_COLUMNS) + "\n" + ",".join(row) + "\n"
    return gzip.compress(text.encode(), mtime=0)


def _routing_identity(
    sessions: MassiveSessionAuthority,
) -> PITSecurityUniverseAuthority:
    """Identity-only authority used by persisted partition routing."""

    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    effective_index = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == "2020-05-01"
    )
    effective = sessions.sessions[effective_index]
    end_index = effective_index - rule.ranking_lag_sessions
    start_index = end_index - rule.ranking_lookback_sessions + 1
    listing_at = sessions.sessions[0].regular_open_ns // 1_000_000
    rank = UniverseRankInputRecord(
        security_id="SEC-A",
        effective_at_ms=effective.regular_open_ns // 1_000_000,
        effective_session_index=effective_index,
        available_at_ms=effective.regular_open_ns // 1_000_000 - 1,
        observation_start_ms=(
            sessions.sessions[start_index].regular_close_ns // 1_000_000
        ),
        observation_end_ms=(sessions.sessions[end_index].regular_close_ns // 1_000_000),
        observation_start_session_index=start_index,
        observation_end_session_index=end_index,
        observed_session_count=rule.ranking_lookback_sessions,
        average_dollar_volume=10_000_000.0,
        close_price=100.0,
        source_receipt_sha256=semantic_sha256("routing-rank-input"),
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
                listing_at_ms=listing_at,
                delisting_at_ms=None,
                successor_security_id=None,
                corporate_action_chain_id="CHAIN-A",
                identity_source_receipt_sha256=semantic_sha256("routing-master-a"),
            ),
        ),
        ticker_history=(
            SourcedTickerHistoryRecord(
                security_id="SEC-A",
                ticker="AAA",
                valid_from_ms=listing_at,
                valid_to_ms=None,
                available_at_ms=listing_at,
                primary_exchange="XNYS",
                source_receipt_sha256=semantic_sha256("routing-ticker-a"),
            ),
        ),
        listing_events=(
            ListingEventRecord(
                event_id="LIST-A",
                security_id="SEC-A",
                effective_at_ms=listing_at,
                available_at_ms=listing_at,
                primary_exchange="XNYS",
                ticker="AAA",
                source_receipt_sha256=semantic_sha256("routing-listing-a"),
            ),
        ),
        delisting_events=(),
        rank_inputs=(rank,),
    )


def _republish_bar_artifact(
    *,
    root: Path,
    artifact: MassiveDailyBarsArtifactV0,
    ordinal: int,
    mutate_dollar_volume: bool,
) -> MassiveDailyBarsArtifactV0:
    source_row = artifact.rows[0]
    values = list(source_row.values)
    if mutate_dollar_volume:
        values[MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")] += 1_000_000.0
    row_body = {
        "security_id": source_row.security_id,
        "values": tuple(values),
        "valid": source_row.valid,
        "source_active_inventory_sha256": (source_row.source_active_inventory_sha256),
    }
    row = MassiveDailyBarsRowV0(
        **row_body,
        receipt_sha256=semantic_sha256(row_body),
    )
    row_inventory = semantic_sha256((row.receipt_sha256,))
    payload = {
        "schema": MASSIVE_DAILY_BARS_V0_SCHEMA,
        "source_session_date": artifact.source_session_date,
        "persisted_partition_manifest_receipt_sha256": (
            artifact.persisted_partition_manifest_receipt_sha256
        ),
        "condition_authority_receipt_sha256": (
            artifact.condition_authority_receipt_sha256
        ),
        "feature_spec_receipt_sha256": MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
        "feature_source_sha256": MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
        "feature_names": MASSIVE_DAILY_BARS_V0_FIELDS,
        "rows": (asdict(row),),
        "row_inventory_sha256": row_inventory,
    }
    relative = (
        "massive-finalized-v0-tampered/"
        f"session={artifact.source_session_date}/daily-bars.json"
    )
    published = _ms("2026-08-25", time(14, 0)) + ordinal
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_DAILY_BARS_V0_DATASET,
        source_object_key=relative,
        requested_at_ms=published,
        downloaded_at_ms=published,
        schema_sha256=MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_ENTITLEMENT,
        committed_at_ms=published,
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=published
    )
    body = {
        "source_session_date": artifact.source_session_date,
        "persisted_partition_manifest_receipt_sha256": (
            artifact.persisted_partition_manifest_receipt_sha256
        ),
        "condition_authority_receipt_sha256": (
            artifact.condition_authority_receipt_sha256
        ),
        "feature_spec_receipt_sha256": MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
        "feature_source_sha256": MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
        "rows": (row,),
        "row_inventory_sha256": row_inventory,
        "loaded_source": loaded,
        "schema": MASSIVE_DAILY_BARS_V0_SCHEMA,
    }
    provisional = MassiveDailyBarsArtifactV0(**body, receipt_sha256="0" * 64)
    result = MassiveDailyBarsArtifactV0(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    validate_massive_daily_bars_v0(root=root, artifact=result)
    return result


def _rank_source_receipt(
    *,
    effective_at_ms: int,
    window_dates: tuple[str, ...],
    bars: tuple[MassiveDailyBarsArtifactV0, ...],
    acquisition: object,
) -> str:
    listing_by_acquisition = {
        row.acquisition_evidence.receipt_sha256: row
        for row in acquisition.captured_listings
    }
    downloads = {
        row.source_object_key: row for row in acquisition.authenticated_downloads
    }
    bar_by_date = {row.source_session_date: row for row in bars}
    return semantic_sha256(
        {
            "specification_sha256": origin_v2.MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
            "security_id": "SEC-A",
            "effective_at_ms": effective_at_ms,
            "window": tuple(
                {
                    "session_date": session_date,
                    "daily_bars_artifact_receipt_sha256": (
                        bar_by_date[session_date].receipt_sha256
                    ),
                    "daily_bar_row_receipt_sha256": (
                        bar_by_date[session_date].rows[0].receipt_sha256
                    ),
                    "listing_entry_receipt_sha256": listing_by_acquisition[
                        downloads[
                            canonical_massive_trade_object_key(session_date)
                        ].listing_acquisition_receipt_sha256
                    ]
                    .committed_listing.resolve(
                        source_object_key=canonical_massive_trade_object_key(
                            session_date
                        )
                    )
                    .receipt_sha256,
                    "authenticated_download_receipt_sha256": downloads[
                        canonical_massive_trade_object_key(session_date)
                    ].receipt_sha256,
                }
                for session_date in window_dates
            ),
        }
    )


def _identity(
    *,
    sessions: MassiveSessionAuthority,
    bars: tuple[MassiveDailyBarsArtifactV0, ...],
    acquisition: object,
    stale_window: bool = False,
) -> PITSecurityUniverseAuthority:
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    effective_index = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == "2020-05-01"
    )
    effective = sessions.sessions[effective_index]
    end_index = effective_index - rule.ranking_lag_sessions
    start_index = end_index - rule.ranking_lookback_sessions + 1
    canonical_dates = tuple(
        row.session_date for row in sessions.sessions[start_index : end_index + 1]
    )
    source_receipt = _rank_source_receipt(
        effective_at_ms=effective.regular_open_ns // 1_000_000,
        window_dates=canonical_dates,
        bars=bars,
        acquisition=acquisition,
    )
    if stale_window:
        start_index -= 1
        end_index -= 1
        source_receipt = semantic_sha256("causal-but-stale-rank-window")
    window_bars = bars[: rule.ranking_lookback_sessions]
    average = sum(row.rows[0].values[5] for row in window_bars) / len(window_bars)
    listing_by_acquisition = {
        row.acquisition_evidence.receipt_sha256: row
        for row in acquisition.captured_listings
    }
    available = max(
        listing_by_acquisition[row.listing_acquisition_receipt_sha256]
        .committed_listing.resolve(source_object_key=row.source_object_key)
        .vendor_last_modified_at_ms
        for row in acquisition.authenticated_downloads
        if row.source_object_key
        in {canonical_massive_trade_object_key(value) for value in canonical_dates}
    )
    rank = UniverseRankInputRecord(
        security_id="SEC-A",
        effective_at_ms=effective.regular_open_ns // 1_000_000,
        effective_session_index=effective_index,
        available_at_ms=available,
        observation_start_ms=(
            sessions.sessions[start_index].regular_close_ns // 1_000_000
        ),
        observation_end_ms=(sessions.sessions[end_index].regular_close_ns // 1_000_000),
        observation_start_session_index=start_index,
        observation_end_session_index=end_index,
        observed_session_count=rule.ranking_lookback_sessions,
        average_dollar_volume=average,
        close_price=bars[-1].rows[0].values[3],
        source_receipt_sha256=source_receipt,
    )
    listing_at = sessions.sessions[0].regular_open_ns // 1_000_000
    master = SourcedSecurityMasterRecord(
        security_id="SEC-A",
        issuer_id="ISS-A",
        primary_exchange="XNYS",
        share_class="COMMON",
        security_type="common-stock",
        listing_at_ms=listing_at,
        delisting_at_ms=None,
        successor_security_id=None,
        corporate_action_chain_id="CHAIN-A",
        identity_source_receipt_sha256=semantic_sha256("master-a"),
    )
    ticker = SourcedTickerHistoryRecord(
        security_id="SEC-A",
        ticker="AAA",
        valid_from_ms=listing_at,
        valid_to_ms=None,
        available_at_ms=listing_at,
        primary_exchange="XNYS",
        source_receipt_sha256=semantic_sha256("ticker-a"),
    )
    listing = ListingEventRecord(
        event_id="LIST-A",
        security_id="SEC-A",
        effective_at_ms=listing_at,
        available_at_ms=listing_at,
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
        rank_inputs=(rank,),
    )


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    sessions = _sessions()
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    effective_index = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == "2020-05-01"
    )
    end_index = effective_index - rule.ranking_lag_sessions
    start_index = end_index - rule.ranking_lookback_sessions + 1
    window_dates = tuple(
        row.session_date for row in sessions.sessions[start_index : end_index + 1]
    )
    payloads = {
        canonical_massive_trade_object_key(day): _flat_trade_payload(
            session_date=day, ordinal=index
        )
        for index, day in enumerate(window_dates)
    }
    client = _S3Client(payloads)
    monkeypatch.setattr(
        origin_v2,
        "_fixed_massive_s3_client_v2",
        lambda **_: client,
    )
    acquisition = capture_massive_profitability_production_acquisition_v2(
        root=tmp_path,
        source_object_keys=tuple(payloads),
        entitlement_receipt_sha256=_ENTITLEMENT,
    )
    routing_identity = _routing_identity(sessions)
    conditions = build_massive_condition_authority(
        (
            {
                "id": 1,
                "name": "Regular",
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
        source_object_receipt_sha256=semantic_sha256("rank-condition-source"),
        source_query_path=MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    )
    corrections = build_massive_correction_authority(
        (
            (0, "new-trade"),
            (1, "replacement"),
            (2, "cancellation"),
            (3, "late-report"),
        ),
        canary_receipt_sha256=semantic_sha256("rank-correction-canary"),
    )
    feature_domain = build_massive_finalized_feature_domain_spec_v0(
        condition_authority=conditions,
        correction_authority=corrections,
    )
    download_by_date = {
        row.source_object_key.rsplit("/", 1)[-1].removesuffix(".csv.gz"): row
        for row in acquisition.authenticated_downloads
    }
    scans = []
    semantic_manifests = []
    persisted_manifests = []
    bars_list = []
    for index, day in enumerate(window_dates):
        session = sessions.resolve(exchange="XNYS", session_date=day)
        rows, scan = scan_massive_daily_trade_file_v0(
            root=tmp_path,
            loaded_source=download_by_date[day].loaded_source,
            session_authority=sessions,
            session=session,
            correction_authority=corrections,
        )
        semantic_manifest = build_massive_daily_trade_partition_manifest_v0(
            rows=rows,
            scan_evidence=scan,
            identity_authority=routing_identity,
            condition_authority=conditions,
            correction_authority=corrections,
            feature_domain_spec=feature_domain,
        )
        persisted = persist_massive_daily_trade_partitions_v1(
            root=tmp_path,
            rows=rows,
            scan_evidence=scan,
            semantic_partition_manifest=semantic_manifest,
            identity_authority=routing_identity,
            correction_authority=corrections,
            entitlement_receipt_sha256=_ENTITLEMENT,
            published_at_ms=_ms("2026-08-25", time(12, 0)) + index,
        )
        bar = materialize_massive_daily_bars_v0(
            persisted_root=tmp_path,
            output_root=tmp_path,
            manifest=persisted,
            condition_authority=conditions,
            entitlement_receipt_sha256=_ENTITLEMENT,
            published_at_ms=_ms("2026-08-25", time(13, 0)) + index,
        )
        scans.append(scan)
        semantic_manifests.append(semantic_manifest)
        persisted_manifests.append(persisted)
        bars_list.append(bar)
    bars = tuple(bars_list)
    identity = _identity(
        sessions=sessions,
        bars=bars,
        acquisition=acquisition,
    )
    rank_authority = build_massive_monthly_rank_input_authority_v2(
        root=tmp_path,
        session_authority=sessions,
        identity_authority=identity,
        acquisition=acquisition,
        daily_bars=bars,
        first_candidate_decision_session_date="2020-05-01",
        last_candidate_decision_session_date="2020-05-01",
    )
    return {
        "sessions": sessions,
        "window_dates": window_dates,
        "acquisition": acquisition,
        "bars": bars,
        "identity": identity,
        "routing_identity": routing_identity,
        "conditions": conditions,
        "corrections": corrections,
        "scans": tuple(scans),
        "semantic_manifests": tuple(semantic_manifests),
        "persisted_manifests": tuple(persisted_manifests),
        "rank_authority": rank_authority,
    }


def test_production_entry_point_is_noninjectable_and_binds_authenticated_gets(
    stack: dict[str, object],
) -> None:
    signature = inspect.signature(
        capture_massive_profitability_production_acquisition_v2
    )
    assert "s3_client" not in signature.parameters
    assert "now_ms" not in signature.parameters
    assert "monotonic_ns" not in signature.parameters
    acquisition = stack["acquisition"]
    assert acquisition.fixed_runtime_captured is True
    assert acquisition.fixed_runtime_capture_receipt_sha256 is not None
    assert all(
        row.provider_request_id.startswith("GET-")
        for row in acquisition.authenticated_downloads
    )


def test_exact_t_minus_one_rank_authority_reparses_bars_and_sources(
    stack: dict[str, object],
) -> None:
    authority = stack["rank_authority"]
    group = authority.groups[0]
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    row = group.rank_inputs[0]
    assert row.observation_end_session_index == (
        row.effective_session_index - rule.ranking_lag_sessions
    )
    assert row.observation_start_session_index == (
        row.observation_end_session_index - rule.ranking_lookback_sessions + 1
    )
    assert group.observation_session_dates == stack["window_dates"]
    assert row.observed_session_count == 63
    assert authority.source_data_qualified is True


def test_rank_bar_authority_rederives_authenticated_partitions(
    stack: dict[str, object], tmp_path: Path
) -> None:
    authority = build_massive_monthly_rank_bar_authority_v1(
        source_root=tmp_path,
        persisted_root=tmp_path,
        daily_bars_root=tmp_path,
        session_authority=stack["sessions"],
        identity_authority=stack["routing_identity"],
        condition_authority=stack["conditions"],
        correction_authority=stack["corrections"],
        acquisition=stack["acquisition"],
        rank_input_authority=stack["rank_authority"],
        scan_evidence=stack["scans"],
        semantic_partition_manifests=stack["semantic_manifests"],
        persisted_partition_manifests=stack["persisted_manifests"],
        daily_bars=stack["bars"],
    )
    assert len(authority.sessions) == 63
    assert authority.source_transport_qualified is True
    assert authority.rank_bar_data_qualified is True
    assert authority.panel_materialization_authorized is False
    assert authority.predictive_training_authorized is False

    nonauthorizing = build_massive_monthly_rank_bar_authority_for_test_v1(
        source_root=tmp_path,
        persisted_root=tmp_path,
        daily_bars_root=tmp_path,
        session_authority=stack["sessions"],
        identity_authority=stack["routing_identity"],
        condition_authority=stack["conditions"],
        correction_authority=stack["corrections"],
        acquisition=stack["acquisition"],
        rank_input_authority=stack["rank_authority"],
        scan_evidence=stack["scans"],
        semantic_partition_manifests=stack["semantic_manifests"],
        persisted_partition_manifests=stack["persisted_manifests"],
        daily_bars=stack["bars"],
    )
    assert nonauthorizing.source_transport_qualified is False
    assert nonauthorizing.rank_bar_data_qualified is False


def test_rank_bar_authority_rejects_self_consistent_fake_dollar_volume(
    stack: dict[str, object], tmp_path: Path
) -> None:
    tampered_bars = tuple(
        _republish_bar_artifact(
            root=tmp_path,
            artifact=artifact,
            ordinal=index,
            mutate_dollar_volume=index == 0,
        )
        for index, artifact in enumerate(stack["bars"])
    )
    tampered_identity = _identity(
        sessions=stack["sessions"],
        bars=tampered_bars,
        acquisition=stack["acquisition"],
    )
    tampered_rank = build_massive_monthly_rank_input_authority_v2(
        root=tmp_path,
        session_authority=stack["sessions"],
        identity_authority=tampered_identity,
        acquisition=stack["acquisition"],
        daily_bars=tampered_bars,
        first_candidate_decision_session_date="2020-05-01",
        last_candidate_decision_session_date="2020-05-01",
    )
    with pytest.raises(
        MassiveMonthlyRankBarAuthorityV1Error,
        match="daily bars differ from authenticated partition rederivation",
    ):
        build_massive_monthly_rank_bar_authority_v1(
            source_root=tmp_path,
            persisted_root=tmp_path,
            daily_bars_root=tmp_path,
            session_authority=stack["sessions"],
            identity_authority=stack["routing_identity"],
            condition_authority=stack["conditions"],
            correction_authority=stack["corrections"],
            acquisition=stack["acquisition"],
            rank_input_authority=tampered_rank,
            scan_evidence=stack["scans"],
            semantic_partition_manifests=stack["semantic_manifests"],
            persisted_partition_manifests=stack["persisted_manifests"],
            daily_bars=tampered_bars,
        )


def test_causal_but_staler_rank_window_is_rejected(
    stack: dict[str, object], tmp_path: Path
) -> None:
    stale_identity = _identity(
        sessions=stack["sessions"],
        bars=stack["bars"],
        acquisition=stack["acquisition"],
        stale_window=True,
    )
    with pytest.raises(
        MassiveProfitabilityOriginV2Error,
        match="differ from committed daily-bar derivation",
    ):
        build_massive_monthly_rank_input_authority_v2(
            root=tmp_path,
            session_authority=stack["sessions"],
            identity_authority=stale_identity,
            acquisition=stack["acquisition"],
            daily_bars=stack["bars"],
            first_candidate_decision_session_date="2020-05-01",
            last_candidate_decision_session_date="2020-05-01",
        )


def test_synthetic_acquisition_cannot_authorize_plan(
    stack: dict[str, object], tmp_path: Path
) -> None:
    production = stack["acquisition"]
    synthetic = build_massive_profitability_acquisition_for_test_v2(
        captured_listings=production.captured_listings,
        authenticated_downloads=production.authenticated_downloads,
        entitlement_receipt_sha256=_ENTITLEMENT,
    )
    assert synthetic.fixed_runtime_captured is False
    rank_authority = build_massive_monthly_rank_input_authority_v2(
        root=tmp_path,
        session_authority=stack["sessions"],
        identity_authority=stack["identity"],
        acquisition=synthetic,
        daily_bars=stack["bars"],
        first_candidate_decision_session_date="2020-05-01",
        last_candidate_decision_session_date="2020-05-01",
    )
    assert rank_authority.source_data_qualified is False
    with pytest.raises(
        MassiveProfitabilityOriginV2Error,
        match="package-owned production acquisition",
    ):
        build_massive_profitability_decision_origin_plan_v2(
            root=tmp_path,
            session_authority=stack["sessions"],
            identity_authority=stack["identity"],
            acquisition=synthetic,
            source_evidence_artifact=object(),
            monthly_rank_authority=rank_authority,
            daily_bars=stack["bars"],
            first_candidate_decision_session_date="2020-05-01",
            last_candidate_decision_session_date="2020-05-01",
        )


def test_production_acquisition_and_exact_ranks_bind_v2_origin_plan(
    stack: dict[str, object], tmp_path: Path
) -> None:
    sessions = stack["sessions"]
    candidate_index = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == "2020-05-01"
    )
    source_date = sessions.sessions[candidate_index - 2].session_date
    artifact = (
        materialize_massive_profitability_acquired_source_evidence_from_acquisition_v2(
            root=tmp_path,
            acquisition=stack["acquisition"],
            source_session_dates=(source_date,),
            artifact_id="origin-v2-source",
            committed_at_ms=_ms("2026-08-25", time(13, 0)),
        )
    )
    plan = build_massive_profitability_decision_origin_plan_v2(
        root=tmp_path,
        session_authority=sessions,
        identity_authority=stack["identity"],
        acquisition=stack["acquisition"],
        source_evidence_artifact=artifact,
        monthly_rank_authority=stack["rank_authority"],
        daily_bars=stack["bars"],
        first_candidate_decision_session_date="2020-05-01",
        last_candidate_decision_session_date="2020-05-01",
    )
    assert len(plan.origin_plan_v1.origins) == 1
    origin = plan.origin_plan_v1.origins[0]
    assert origin.source_session_date == source_date
    assert origin.scheduled_rebalance_session_date == "2020-05-01"
    assert (
        plan.production_acquisition_receipt_sha256
        == stack["acquisition"].receipt_sha256
    )
    assert (
        plan.monthly_rank_authority_semantic_receipt_sha256
        == stack["rank_authority"].semantic_receipt_sha256
    )
    assert plan.panel_materialization_authorized is False
    assert plan.predictive_training_authorized is False


def test_v2_origin_rank_and_rank_bar_authorities_freeze_and_support_round_trip(
    stack: dict[str, object], tmp_path: Path
) -> None:
    sessions = stack["sessions"]
    candidate_index = next(
        index
        for index, row in enumerate(sessions.sessions)
        if row.session_date == "2020-05-01"
    )
    source_date = sessions.sessions[candidate_index - 2].session_date
    source_artifact = (
        materialize_massive_profitability_acquired_source_evidence_from_acquisition_v2(
            root=tmp_path,
            acquisition=stack["acquisition"],
            source_session_dates=(source_date,),
            artifact_id="frozen-origin-source",
            committed_at_ms=_ms("2026-08-25", time(13, 30)),
        )
    )
    origin_plan = build_massive_profitability_decision_origin_plan_v2(
        root=tmp_path,
        session_authority=sessions,
        identity_authority=stack["identity"],
        acquisition=stack["acquisition"],
        source_evidence_artifact=source_artifact,
        monthly_rank_authority=stack["rank_authority"],
        daily_bars=stack["bars"],
        first_candidate_decision_session_date="2020-05-01",
        last_candidate_decision_session_date="2020-05-01",
    )
    rank_bar = build_massive_monthly_rank_bar_authority_v1(
        source_root=tmp_path,
        persisted_root=tmp_path,
        daily_bars_root=tmp_path,
        session_authority=sessions,
        identity_authority=stack["routing_identity"],
        condition_authority=stack["conditions"],
        correction_authority=stack["corrections"],
        acquisition=stack["acquisition"],
        rank_input_authority=stack["rank_authority"],
        scan_evidence=stack["scans"],
        semantic_partition_manifests=stack["semantic_manifests"],
        persisted_partition_manifests=stack["persisted_manifests"],
        daily_bars=stack["bars"],
    )
    origin_frozen = materialize_massive_profitability_origin_plan_v2(
        root=tmp_path,
        authority=origin_plan,
        acquisition=stack["acquisition"],
        monthly_rank_authority=stack["rank_authority"],
        artifact_id="origin",
        committed_at_ms=_ms("2026-08-25", time(14, 0)),
        entitlement_receipt_sha256=_ENTITLEMENT,
    )
    rank_frozen = materialize_massive_monthly_rank_input_authority_v2(
        root=tmp_path,
        authority=stack["rank_authority"],
        acquisition=stack["acquisition"],
        artifact_id="rank",
        committed_at_ms=_ms("2026-08-25", time(14, 1)),
        entitlement_receipt_sha256=_ENTITLEMENT,
    )
    rank_bar_frozen = materialize_massive_monthly_rank_bar_authority_v1(
        root=tmp_path,
        authority=rank_bar,
        acquisition=stack["acquisition"],
        artifact_id="rank-bar",
        committed_at_ms=_ms("2026-08-25", time(14, 2)),
        entitlement_receipt_sha256=_ENTITLEMENT,
    )

    assert origin_frozen.runtime_qualified is True
    assert rank_frozen.runtime_qualified is True
    assert rank_bar_frozen.runtime_qualified is True
    assert origin_frozen.authority_semantic_payload_sha256 != (
        origin_frozen.authority_semantic_receipt_sha256
    )
    assert (
        parse_massive_profitability_frozen_authority_v1(
            root=tmp_path, loaded_source=rank_bar_frozen.loaded_source
        ).runtime_qualified
        is False
    )
    assert massive_profitability_identity_semantic_receipt_v2(
        stack["routing_identity"]
    ) == massive_profitability_identity_semantic_receipt_v2(stack["identity"])

    support = materialize_massive_profitability_security_support_v2(
        root=tmp_path,
        origin_plan=origin_plan,
        monthly_rank_authority=stack["rank_authority"],
        monthly_rank_bar_authority=rank_bar,
        routing_identity_authority=stack["routing_identity"],
        rank_identity_authority=stack["identity"],
        frozen_origin_artifact=origin_frozen,
        frozen_rank_artifact=rank_frozen,
        frozen_rank_bar_artifact=rank_bar_frozen,
        artifact_id="support",
        committed_at_ms=_ms("2026-08-25", time(14, 3)),
        entitlement_receipt_sha256=_ENTITLEMENT,
    )
    assert support.decision_member_security_ids == ("SEC-A",)
    assert support.all_supported_security_ids == ("SEC-A",)
    assert support.components_runtime_qualified is True
    reloaded = parse_massive_profitability_security_support_v2(
        root=tmp_path, loaded_source=support.loaded_source
    )
    assert reloaded.semantic_receipt_sha256 == support.semantic_receipt_sha256
    assert reloaded.components_runtime_qualified is False


def test_performance_authorizations_remain_false() -> None:
    assert MASSIVE_PROFITABILITY_ORIGIN_V2_PANEL_MATERIALIZATION_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V2_PREDICTIVE_TRAINING_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V2_PROFITABILITY_REPORTING_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V2_LOCKBOX_ACCESS_AUTHORIZED is False
