from __future__ import annotations

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
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_BUCKET,
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
    validate_massive_daily_bars_v0,
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


def _bar_artifact(
    *, root: Path, session_date: str, ordinal: int
) -> MassiveDailyBarsArtifactV0:
    values = (
        100.0 + ordinal,
        101.0 + ordinal,
        99.0 + ordinal,
        100.0 + ordinal,
        10_000.0,
        10_000_000.0 + ordinal,
        0.02,
        0.5,
    )
    row_body = {
        "security_id": "SEC-A",
        "values": values,
        "valid": (True,) * len(MASSIVE_DAILY_BARS_V0_FIELDS),
        "source_active_inventory_sha256": semantic_sha256(("active", session_date)),
    }
    row = MassiveDailyBarsRowV0(
        **row_body,
        receipt_sha256=semantic_sha256(row_body),
    )
    persisted_receipt = semantic_sha256(("persisted", session_date))
    condition_receipt = semantic_sha256("condition-authority")
    row_inventory = semantic_sha256((row.receipt_sha256,))
    payload = {
        "schema": MASSIVE_DAILY_BARS_V0_SCHEMA,
        "source_session_date": session_date,
        "persisted_partition_manifest_receipt_sha256": persisted_receipt,
        "condition_authority_receipt_sha256": condition_receipt,
        "feature_spec_receipt_sha256": MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
        "feature_source_sha256": MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
        "feature_names": MASSIVE_DAILY_BARS_V0_FIELDS,
        "rows": (asdict(row),),
        "row_inventory_sha256": row_inventory,
    }
    relative = f"massive-finalized-v0/session={session_date}/daily-bars.json"
    published = _ms("2026-08-25", time(12, 0)) + ordinal
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
        "source_session_date": session_date,
        "persisted_partition_manifest_receipt_sha256": persisted_receipt,
        "condition_authority_receipt_sha256": condition_receipt,
        "feature_spec_receipt_sha256": MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
        "feature_source_sha256": MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
        "rows": (row,),
        "row_inventory_sha256": row_inventory,
        "loaded_source": loaded,
        "schema": MASSIVE_DAILY_BARS_V0_SCHEMA,
    }
    provisional = MassiveDailyBarsArtifactV0(
        **body,
        receipt_sha256="0" * 64,
    )
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
        canonical_massive_trade_object_key(day): f"source-{day}".encode()
        for day in window_dates
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
    bars = tuple(
        _bar_artifact(root=tmp_path, session_date=day, ordinal=index)
        for index, day in enumerate(window_dates)
    )
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


def test_performance_authorizations_remain_false() -> None:
    assert MASSIVE_PROFITABILITY_ORIGIN_V2_PANEL_MATERIALIZATION_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V2_PREDICTIVE_TRAINING_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V2_PROFITABILITY_REPORTING_AUTHORIZED is False
    assert MASSIVE_PROFITABILITY_ORIGIN_V2_LOCKBOX_ACCESS_AUTHORIZED is False
