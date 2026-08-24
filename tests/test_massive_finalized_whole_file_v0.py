from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, time
import gzip
import inspect
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    PITUniverseRuleSpec,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.conditions import (
    MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    build_massive_condition_authority,
)
from rl_quant.data_sources.massive.corrections import build_massive_correction_authority
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanError,
    scan_massive_daily_trade_file_v0,
)
from rl_quant.data_sources.massive.finalized_artifact_readiness import (
    MASSIVE_ARTIFACT_EXECUTION_DATASET_V1,
    MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256,
    MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
    MASSIVE_ARTIFACT_READINESS_STAGE_CONTRACTS_V1,
    MassiveArtifactReadinessError,
    measure_massive_artifact_readiness_v1,
    parse_massive_artifact_execution_authority_v1,
)
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_BUCKET,
    MassiveFlatFileListingAcquisitionError,
    capture_massive_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
)
from rl_quant.data_sources.massive.finalized_origin_authority import (
    MassiveQualifiedFinalizedOriginError,
    build_massive_qualified_finalized_daily_source_v0,
    build_massive_qualified_finalized_origin_plan_v0,
)
from rl_quant.data_sources.massive.finalized_origin_policy import (
    MASSIVE_FINALIZED_ORIGIN_POLICY_V0,
    MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256,
    MASSIVE_FINALIZED_ORIGIN_POLICY_V1,
    MASSIVE_FINALIZED_ORIGIN_POLICY_V1_RECEIPT_SHA256,
    MassiveFinalizedOriginPolicyError,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionError,
    build_massive_finalized_feature_domain_spec_v0,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    persist_massive_daily_trade_partitions_v1,
    stream_and_persist_massive_daily_trade_partitions_v1,
    validate_massive_persisted_partitions_v1,
)
from rl_quant.data_sources.massive.finalized_readiness import (
    MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0,
    MassiveFinalizedReadinessError,
    build_massive_finalized_readiness_capability_v0,
    build_massive_finalized_readiness_panel_spec_v0,
    build_massive_finalized_readiness_stage_artifact_v0,
    measure_massive_finalized_full_readiness_v0,
)
from rl_quant.data_sources.massive.processing_capability import (
    build_massive_finalized_processing_capability_v0,
    measure_massive_finalized_source_processing_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADES_DATASET_ID,
    MASSIVE_FLAT_TRADE_COLUMNS,
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


EASTERN = ZoneInfo("America/New_York")
CALENDAR_RECEIPT = "a" * 64
ENTITLEMENT_RECEIPT = "b" * 64
CONDITION_RECEIPT = "c" * 64
CORRECTION_RECEIPT = "d" * 64
HARDWARE_RECEIPT = "e" * 64
SOFTWARE_COMMIT = "f" * 64


def _ns(day: str, local_time: time) -> int:
    return int(
        datetime.combine(
            date.fromisoformat(day), local_time, tzinfo=EASTERN
        ).timestamp()
        * 1_000_000_000
    )


def _ms(day: str, local_time: time) -> int:
    return _ns(day, local_time) // 1_000_000


def _session(
    day: str,
    *,
    opened: time = time(9, 30),
    closed: time = time(16, 0),
    reason: str | None = None,
) -> MassiveExchangeSession:
    open_ns = _ns(day, opened)
    close_ns = _ns(day, closed)
    return MassiveExchangeSession(
        session_date=day,
        exchange="XNYS",
        regular_open_ns=open_ns,
        regular_close_ns=close_ns,
        scheduled_five_minute_intervals=(close_ns - open_ns) // (300 * 1_000_000_000),
        special_session_reason=reason,
        calendar_source_receipt_sha256=CALENDAR_RECEIPT,
    )


def _trade_row(
    *,
    ticker: str,
    trade_id: str,
    participant_ns: int,
    sip_ns: int,
    price: str = "10.00",
    size: str = "100",
    correction: int = 0,
    sequence: int = 1,
) -> tuple[str, ...]:
    return (
        ticker,
        "[1]",
        str(correction),
        "4",
        trade_id,
        str(participant_ns),
        price,
        str(sequence),
        str(sip_ns),
        size,
        "1",
        "12",
        str(sip_ns),
    )


def _flat_payload(rows: tuple[tuple[str, ...], ...]) -> bytes:
    text = ",".join(MASSIVE_FLAT_TRADE_COLUMNS) + "\n"
    text += "".join(",".join(row) + "\n" for row in rows)
    return gzip.compress(text.encode(), mtime=0)


def _publish(
    *,
    root: Path,
    key: str,
    payload: bytes,
    dataset_id: str,
    schema_sha256: str,
    downloaded_at_ms: int,
    etag: str,
):
    root.mkdir(parents=True, exist_ok=True)
    publish_massive_source_object(
        stream=BytesIO(payload),
        root=root,
        relative_payload_path=key,
        dataset_id=dataset_id,
        source_object_key=key,
        requested_at_ms=downloaded_at_ms - 1,
        downloaded_at_ms=downloaded_at_ms,
        schema_sha256=schema_sha256,
        entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
        committed_at_ms=downloaded_at_ms + 1,
        etag=etag,
    )
    return load_massive_source_bundle(
        root=root,
        relative_payload_path=key,
        verified_at_ms=downloaded_at_ms + 2,
    )


def _condition_authority():
    return build_massive_condition_authority(
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
        source_object_receipt_sha256=CONDITION_RECEIPT,
        source_query_path=MASSIVE_STOCK_TRADE_CONDITION_QUERY,
    )


def _correction_authority():
    return build_massive_correction_authority(
        ((0, "new-trade"), (1, "replacement"), (2, "cancellation"), (3, "late-report")),
        canary_receipt_sha256=CORRECTION_RECEIPT,
    )


def _identity_authority(source_day: str, tickers: tuple[str, ...]):
    listed = _ms(source_day, time(0, 0)) - 86_400_000
    effective = _ms(source_day, time(9, 0))
    rule = PITUniverseRuleSpec.build(
        rule_id="massive-whole-file-test-v0",
        target_size=max(1, len(tickers)),
        ranking_lookback_sessions=3,
        ranking_lag_sessions=1,
        minimum_observed_sessions=2,
        minimum_close_price=1.0,
        minimum_average_dollar_volume=0.0,
        rebalance_frequency="monthly",
    )
    masters = tuple(
        SourcedSecurityMasterRecord(
            f"SEC-{ticker}",
            f"ISS-{ticker}",
            "XNYS",
            "COMMON",
            "common-stock",
            listed,
            None,
            None,
            None,
            semantic_sha256((ticker, "master")),
        )
        for ticker in tickers
    )
    histories = tuple(
        SourcedTickerHistoryRecord(
            f"SEC-{ticker}",
            ticker,
            listed,
            None,
            listed,
            "XNYS",
            semantic_sha256((ticker, "history")),
        )
        for ticker in tickers
    )
    listings = tuple(
        ListingEventRecord(
            f"LIST-{ticker}",
            f"SEC-{ticker}",
            listed,
            listed,
            "XNYS",
            ticker,
            semantic_sha256((ticker, "listing")),
        )
        for ticker in tickers
    )
    rank_inputs = tuple(
        UniverseRankInputRecord(
            security_id=f"SEC-{ticker}",
            effective_at_ms=effective,
            effective_session_index=10,
            available_at_ms=effective - 1,
            observation_start_ms=listed,
            observation_end_ms=effective - 2,
            observation_start_session_index=7,
            observation_end_session_index=9,
            observed_session_count=3,
            average_dollar_volume=1_000_000.0,
            close_price=10.0,
            source_receipt_sha256=semantic_sha256((ticker, "rank")),
        )
        for ticker in tickers
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=histories,
        listing_events=listings,
        delisting_events=(),
        rank_inputs=rank_inputs,
    )


def _qualified_sources(
    tmp_path: Path,
    *,
    sessions: tuple[MassiveExchangeSession, ...],
    source_rows: dict[str, tuple[tuple[str, ...], ...]],
    last_modified: dict[str, int],
):
    session_authority = build_massive_session_authority(
        sessions, calendar_source_receipt_sha256=CALENDAR_RECEIPT
    )
    conditions = _condition_authority()
    corrections = _correction_authority()
    feature_spec = build_massive_finalized_feature_domain_spec_v0(
        condition_authority=conditions, correction_authority=corrections
    )
    tickers = tuple(sorted({row[0] for rows in source_rows.values() for row in rows}))
    identity = _identity_authority(min(source_rows), tickers)
    observed_at_ms = max(last_modified.values()) + 7_200_000
    loaded_by_day = {}
    listing_entries = []
    for index, (day, rows) in enumerate(sorted(source_rows.items())):
        key = canonical_massive_trade_object_key(day)
        payload = _flat_payload(rows)
        etag = f"trade-etag-{index}"
        loaded_by_day[day] = _publish(
            root=tmp_path / "trades",
            key=key,
            payload=payload,
            dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
            schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
            downloaded_at_ms=observed_at_ms + 100 + index,
            etag=etag,
        )
        listing_entries.append(
            {
                "dataset_id": MASSIVE_FLAT_TRADES_DATASET_ID,
                "source_object_key": key,
                "etag": etag,
                "content_length": len(payload),
                "last_modified_at_ms": last_modified[day],
            }
        )
    pages = (
        {
            "ResponseMetadata": {"RequestId": "qualified-listing-request"},
            "IsTruncated": False,
            "Contents": [
                {
                    "Key": row["source_object_key"],
                    "ETag": f'"{row["etag"]}"',
                    "Size": row["content_length"],
                    "LastModified": datetime.fromtimestamp(
                        row["last_modified_at_ms"] / 1_000, tz=UTC
                    ),
                }
                for row in listing_entries
            ],
        },
    )

    class FakePaginator:
        def paginate(self, **kwargs):
            return pages

    class FakeClient:
        def get_paginator(self, operation: str):
            return FakePaginator()

    (tmp_path / "listing").mkdir(parents=True, exist_ok=True)
    capture_times = iter((observed_at_ms - 1, observed_at_ms))
    captured_listing = capture_massive_flat_file_listing_v0(
        s3_client=FakeClient(),
        root=tmp_path / "listing",
        year=int(min(source_rows)[:4]),
        month=int(min(source_rows)[5:7]),
        entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
        now_ms=lambda: next(capture_times),
    )
    listing = captured_listing.committed_listing
    stacks = []
    benchmarks = []
    for day, loaded in sorted(loaded_by_day.items()):
        session = session_authority.resolve(exchange="XNYS", session_date=day)
        _, scan, partition, benchmark = measure_massive_finalized_source_processing_v0(
            root=tmp_path / "trades",
            loaded_source=loaded,
            session_authority=session_authority,
            session=session,
            identity_authority=identity,
            condition_authority=conditions,
            correction_authority=corrections,
            feature_domain_spec=feature_spec,
            hardware_contract_receipt_sha256=HARDWARE_RECEIPT,
            software_commit_sha256=SOFTWARE_COMMIT,
        )
        stacks.append((day, loaded, session, scan, partition))
        benchmarks.append(benchmark)
    legacy_capability = build_massive_finalized_processing_capability_v0(benchmarks)
    assert legacy_capability.readiness_authorizing is False

    def stage_runner(stage_id: str, inputs: tuple[str, ...]):
        output = _publish(
            root=tmp_path / "readiness",
            key=f"readiness-v0/{stage_id}.json",
            payload=(semantic_sha256((stage_id, inputs)) + "\n").encode(),
            dataset_id=f"massive-finalized-readiness-{stage_id}-v0",
            schema_sha256=semantic_sha256(("readiness-stage-schema", stage_id)),
            downloaded_at_ms=observed_at_ms + 20,
            etag=f"readiness-{stage_id}",
        )
        return build_massive_finalized_readiness_stage_artifact_v0(
            stage_id=stage_id,
            input_artifact_receipts=inputs,
            output_loaded_source=output,
            implementation_source_sha256=semantic_sha256(
                ("readiness-implementation", stage_id)
            ),
        )

    monotonic_values = iter((1_000_000_000, 1_010_000_000))
    wall_values = iter((observed_at_ms, observed_at_ms + 10))
    base_run = measure_massive_finalized_full_readiness_v0(
        scan_evidence=stacks[0][3],
        partition_manifest=stacks[0][4],
        listing_acquisition_receipt_sha256=(
            captured_listing.acquisition_evidence.receipt_sha256
        ),
        hardware_contract_receipt_sha256=HARDWARE_RECEIPT,
        software_commit_sha256=SOFTWARE_COMMIT,
        pipeline_implementation_source_sha256=semantic_sha256(
            "test-full-readiness-pipeline"
        ),
        stage_runner=stage_runner,
        monotonic_ns=lambda: next(monotonic_values),
        wall_ms=lambda: next(wall_values),
    )
    readiness_dates = tuple(
        f"{year}-{month:02d}-{day:02d}"
        for year in (2024, 2025, 2026)
        for month, day in ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8))
    )[:20]
    readiness_runs = []
    for index, session_date in enumerate(readiness_dates):
        run = replace(
            base_run,
            source_session_date=session_date,
            source_object_receipt_sha256=semantic_sha256(
                ("readiness-source", session_date)
            ),
            compressed_bytes=base_run.compressed_bytes + index,
            source_row_count=base_run.source_row_count + index,
            ticker_count=base_run.ticker_count + index,
            post_close_correction_row_count=1 if index == 19 else 0,
        )
        run = replace(run, receipt_sha256=semantic_sha256(run.unsigned()))
        run.validate()
        readiness_runs.append(run)
    largest_run = readiness_runs[-1]
    panel_spec = build_massive_finalized_readiness_panel_spec_v0(
        source_session_dates=readiness_dates,
        largest_compressed_source_receipt_sha256=(
            largest_run.source_object_receipt_sha256
        ),
        largest_row_count_source_receipt_sha256=(
            largest_run.source_object_receipt_sha256
        ),
        correction_activity_session_dates=(largest_run.source_session_date,),
        high_ticker_count_session_dates=(largest_run.source_session_date,),
    )
    capability = build_massive_finalized_readiness_capability_v0(
        panel_spec=panel_spec, runs=readiness_runs
    )
    qualified = []
    for day, loaded, session, scan, partition in stacks:
        qualified.append(
            build_massive_qualified_finalized_daily_source_v0(
                listing_root=tmp_path / "listing",
                loaded_source=loaded,
                captured_listing=captured_listing,
                scan_evidence=scan,
                partition_manifest=partition,
                feature_domain_spec=feature_spec,
                readiness_capability=capability,
                session_authority=session_authority,
                source_session=session,
            )
        )
    return {
        "session_authority": session_authority,
        "conditions": conditions,
        "corrections": corrections,
        "feature_spec": feature_spec,
        "identity": identity,
        "listing": listing,
        "captured_listing": captured_listing,
        "loaded": loaded_by_day,
        "stacks": stacks,
        "benchmarks": tuple(benchmarks),
        "legacy_capability": legacy_capability,
        "capability": capability,
        "qualified": tuple(qualified),
    }


def test_whole_file_scan_rejects_future_row_for_another_ticker(tmp_path: Path) -> None:
    source_day = "2026-08-20"
    next_day = "2026-08-21"
    sessions = (_session(source_day), _session(next_day))
    authority = build_massive_session_authority(
        sessions, calendar_source_receipt_sha256=CALENDAR_RECEIPT
    )
    rows = (
        _trade_row(
            ticker="AAA",
            trade_id="A1",
            participant_ns=_ns(source_day, time(15, 59)),
            sip_ns=_ns(source_day, time(15, 59)),
        ),
        _trade_row(
            ticker="BBB",
            trade_id="B1",
            participant_ns=_ns(next_day, time(9, 31)),
            sip_ns=_ns(next_day, time(9, 31)),
            sequence=2,
        ),
    )
    key = canonical_massive_trade_object_key(source_day)
    loaded = _publish(
        root=tmp_path,
        key=key,
        payload=_flat_payload(rows),
        dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
        schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
        downloaded_at_ms=_ms(next_day, time(12, 0)),
        etag="future-row",
    )
    with pytest.raises(MassiveDailyTradeFileScanError, match="outside source"):
        scan_massive_daily_trade_file_v0(
            root=tmp_path,
            loaded_source=loaded,
            session_authority=authority,
            session=sessions[0],
            correction_authority=_correction_authority(),
        )


def test_authenticated_listing_capture_exhausts_pages_and_persists_no_secrets(
    tmp_path: Path,
) -> None:
    modified_20 = datetime(2026, 8, 21, 15, 0, tzinfo=UTC)
    modified_21 = datetime(2026, 8, 22, 15, 0, tzinfo=UTC)
    pages = (
        {
            "ResponseMetadata": {"RequestId": "request-1"},
            "IsTruncated": True,
            "NextContinuationToken": "opaque-token",
            "Contents": [
                {
                    "Key": canonical_massive_trade_object_key("2026-08-20"),
                    "ETag": '"etag-20"',
                    "Size": 100,
                    "LastModified": modified_20,
                }
            ],
        },
        {
            "ResponseMetadata": {"RequestId": "request-2"},
            "IsTruncated": False,
            "Contents": [
                {
                    "Key": canonical_massive_trade_object_key("2026-08-21"),
                    "ETag": '"etag-21"',
                    "Size": 200,
                    "LastModified": modified_21,
                }
            ],
        },
    )

    class FakePaginator:
        def paginate(self, **kwargs):
            assert kwargs == {
                "Bucket": MASSIVE_FLAT_FILE_BUCKET,
                "Prefix": "us_stocks_sip/trades_v1/2026/08/",
            }
            return pages

    class FakeClient:
        def get_paginator(self, operation: str):
            assert operation == "list_objects_v2"
            return FakePaginator()

    completed_ms = int(datetime(2026, 8, 22, 17, 0, tzinfo=UTC).timestamp() * 1_000)
    times = iter((completed_ms - 1_000, completed_ms))
    result = capture_massive_flat_file_listing_v0(
        s3_client=FakeClient(),
        root=tmp_path,
        year=2026,
        month=8,
        entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
        access_key_environment_variable="TEST_ACCESS_KEY_ENV",
        secret_key_environment_variable="TEST_SECRET_KEY_ENV",
        now_ms=lambda: next(times),
    )
    assert result.acquisition_evidence.page_count == 2
    assert result.acquisition_evidence.provider_request_ids == (
        "request-1",
        "request-2",
    )
    assert tuple(row.etag for row in result.committed_listing.entries) == (
        "etag-20",
        "etag-21",
    )
    acquisition_bytes = read_loaded_massive_source_bytes(
        root=tmp_path, loaded_source=result.loaded_acquisition
    )
    assert b"TEST_ACCESS_KEY_ENV" in acquisition_bytes
    assert b"real-access-secret" not in acquisition_bytes
    result.validate()
    tampered_acquisition = replace(
        result.acquisition_evidence,
        object_inventory_sha256=semantic_sha256("substituted-object-inventory"),
    )
    tampered_acquisition = replace(
        tampered_acquisition,
        receipt_sha256=semantic_sha256(tampered_acquisition.unsigned()),
    )
    tampered_acquisition.validate()
    with pytest.raises(
        MassiveFlatFileListingAcquisitionError,
        match="committed-listing inventories differ",
    ):
        replace(result, acquisition_evidence=tampered_acquisition).validate()


def test_listing_capture_rejects_incomplete_pagination(tmp_path: Path) -> None:
    class FakePaginator:
        def paginate(self, **kwargs):
            return (
                {
                    "ResponseMetadata": {"RequestId": "request-1"},
                    "IsTruncated": True,
                    "Contents": (),
                },
            )

    class FakeClient:
        def get_paginator(self, operation: str):
            return FakePaginator()

    with pytest.raises(MassiveFlatFileListingAcquisitionError, match="close exactly"):
        capture_massive_flat_file_listing_v0(
            s3_client=FakeClient(),
            root=tmp_path,
            year=2026,
            month=8,
            entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
            now_ms=lambda: _ms("2026-08-22", time(17, 0)),
        )


def test_participant_time_domain_retains_late_corrections_and_excludes_after_hours(
    tmp_path: Path,
) -> None:
    day = "2026-08-20"
    rows = (
        _trade_row(
            ticker="AAA",
            trade_id="A1",
            participant_ns=_ns(day, time(15, 58)),
            sip_ns=_ns(day, time(15, 58)),
            sequence=1,
        ),
        _trade_row(
            ticker="AAA",
            trade_id="A1",
            participant_ns=_ns(day, time(15, 58)),
            sip_ns=_ns(day, time(16, 5)),
            price="11.00",
            correction=1,
            sequence=2,
        ),
        _trade_row(
            ticker="AAA",
            trade_id="A2",
            participant_ns=_ns(day, time(15, 59)),
            sip_ns=_ns(day, time(16, 6)),
            correction=3,
            sequence=3,
        ),
        _trade_row(
            ticker="AAA",
            trade_id="A3",
            participant_ns=_ns(day, time(18, 0)),
            sip_ns=_ns(day, time(18, 0)),
            sequence=4,
        ),
    )
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(day), _session("2026-08-21")),
        source_rows={day: rows},
        last_modified={day: _ms("2026-08-21", time(11, 0))},
    )
    scan = stack["stacks"][0][3]
    partition = stack["stacks"][0][4].security_partitions[0]
    assert scan.regular_session_row_count == 3
    assert scan.after_hours_row_count == 1
    assert scan.late_report_row_count == 2
    assert scan.post_close_correction_row_count == 1
    assert partition.source_row_count == 4
    assert partition.active_regular_session_row_count == 2
    assert partition.after_hours_row_count == 1
    qualified = stack["qualified"][0]
    assert qualified.listing_acquisition_receipt_sha256 == (
        stack["captured_listing"].acquisition_evidence.receipt_sha256
    )
    assert qualified.measured_feature_ready_upper_bound_at_ms == (
        qualified.vendor_last_modified_at_ms
        + qualified.publication_safety_margin_ms
        + stack["capability"].maximum_runtime_ms
    )
    assert not hasattr(stack["qualified"][0], "panel_materialization_authorized")
    _, repeated_scan, repeated_partition, _ = (
        measure_massive_finalized_source_processing_v0(
            root=tmp_path / "trades",
            loaded_source=stack["loaded"][day],
            session_authority=stack["session_authority"],
            session=stack["session_authority"].resolve(
                exchange="XNYS", session_date=day
            ),
            identity_authority=stack["identity"],
            condition_authority=stack["conditions"],
            correction_authority=stack["corrections"],
            feature_domain_spec=stack["feature_spec"],
            hardware_contract_receipt_sha256=HARDWARE_RECEIPT,
            software_commit_sha256=SOFTWARE_COMMIT,
        )
    )
    assert repeated_scan.receipt_sha256 == scan.receipt_sha256
    assert repeated_partition.receipt_sha256 == stack["stacks"][0][4].receipt_sha256


def test_fake_partition_receipt_cannot_enter_qualified_source(tmp_path: Path) -> None:
    day = "2026-08-20"
    rows = (
        _trade_row(
            ticker="AAA",
            trade_id="A1",
            participant_ns=_ns(day, time(15, 59)),
            sip_ns=_ns(day, time(15, 59)),
        ),
    )
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(day), _session("2026-08-21")),
        source_rows={day: rows},
        last_modified={day: _ms("2026-08-21", time(11, 0))},
    )
    _, loaded, session, scan, partition = stack["stacks"][0]
    forged = replace(partition, receipt_sha256="0" * 64)
    with pytest.raises(MassiveDailyTradePartitionError, match="receipt differs"):
        build_massive_qualified_finalized_daily_source_v0(
            listing_root=tmp_path / "listing",
            loaded_source=loaded,
            captured_listing=stack["captured_listing"],
            scan_evidence=scan,
            partition_manifest=forged,
            feature_domain_spec=stack["feature_spec"],
            readiness_capability=stack["capability"],
            session_authority=stack["session_authority"],
            source_session=session,
        )


def test_processing_over_55_minutes_blocks_source(tmp_path: Path) -> None:
    day = "2026-08-20"
    rows = (
        _trade_row(
            ticker="AAA",
            trade_id="A1",
            participant_ns=_ns(day, time(15, 59)),
            sip_ns=_ns(day, time(15, 59)),
        ),
    )
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(day), _session("2026-08-21")),
        source_rows={day: rows},
        last_modified={day: _ms("2026-08-21", time(11, 0))},
    )
    too_slow = MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS + 1
    slow_runs = list(stack["capability"].runs)
    slow_run = replace(
        slow_runs[-1],
        monotonic_finished_ns=(
            slow_runs[-1].monotonic_started_ns + too_slow * 1_000_000
        ),
        observed_full_pipeline_runtime_ms=too_slow,
    )
    slow_run = replace(slow_run, receipt_sha256=semantic_sha256(slow_run.unsigned()))
    slow_run.validate()
    slow_runs[-1] = slow_run
    capability = build_massive_finalized_readiness_capability_v0(
        panel_spec=stack["capability"].panel_spec,
        runs=slow_runs,
    )
    assert capability.capability_passed is False
    _, loaded, session, scan, partition = stack["stacks"][0]
    with pytest.raises(MassiveQualifiedFinalizedOriginError, match="does not cover"):
        build_massive_qualified_finalized_daily_source_v0(
            listing_root=tmp_path / "listing",
            loaded_source=loaded,
            captured_listing=stack["captured_listing"],
            scan_evidence=scan,
            partition_manifest=partition,
            feature_domain_spec=stack["feature_spec"],
            readiness_capability=capability,
            session_authority=stack["session_authority"],
            source_session=session,
        )


def test_plan_skips_early_close_and_late_open_but_accepts_fill_at_close(
    tmp_path: Path,
) -> None:
    source_day = "2026-08-20"
    rows = (
        _trade_row(
            ticker="AAA",
            trade_id="A1",
            participant_ns=_ns(source_day, time(15, 59)),
            sip_ns=_ns(source_day, time(15, 59)),
        ),
    )
    sessions = (
        _session(source_day),
        _session("2026-08-21", closed=time(13, 0), reason="early-close"),
        _session("2026-08-24", opened=time(13, 0), reason="late-open"),
        _session("2026-08-25"),
    )
    stack = _qualified_sources(
        tmp_path,
        sessions=sessions,
        source_rows={source_day: rows},
        last_modified={source_day: _ms("2026-08-21", time(10, 0))},
    )
    plan = build_massive_qualified_finalized_origin_plan_v0(
        session_authority=stack["session_authority"],
        exchange="XNYS",
        daily_sources=stack["qualified"],
        first_decision_session_date="2026-08-21",
        last_decision_session_date="2026-08-25",
    )
    assert tuple(row.reason for row in plan.skipped_decisions) == (
        "session-does-not-support-decision-and-fill",
        "session-does-not-support-decision-and-fill",
    )
    assert plan.origins[0].decision_session_date == "2026-08-25"
    assert plan.origins[0].fill_end_at_ms == plan.origins[0].regular_close_at_ms
    assert plan.origins[0].source_staleness_context_value == 3
    assert plan.panel_materialization_authorized is False
    forged_origin = replace(
        plan.origins[0],
        source_staleness_sessions=2,
        source_staleness_context_value=2,
    )
    forged_origin = replace(
        forged_origin, receipt_sha256=semantic_sha256(forged_origin.unsigned())
    )
    forged_plan = replace(plan, origins=(forged_origin,))
    forged_plan = replace(
        forged_plan, receipt_sha256=semantic_sha256(forged_plan.unsigned())
    )
    with pytest.raises(MassiveQualifiedFinalizedOriginError, match="independently"):
        forged_plan.validate()


def test_different_feature_spec_receipts_across_dates_fail_plan(tmp_path: Path) -> None:
    days = ("2026-08-20", "2026-08-21")
    source_rows = {
        day: (
            _trade_row(
                ticker="AAA",
                trade_id=f"A-{day}",
                participant_ns=_ns(day, time(15, 59)),
                sip_ns=_ns(day, time(15, 59)),
            ),
        )
        for day in days
    }
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(days[0]), _session(days[1]), _session("2026-08-24")),
        source_rows=source_rows,
        last_modified={
            days[0]: _ms(days[1], time(10, 0)),
            days[1]: _ms("2026-08-22", time(10, 0)),
        },
    )
    changed = replace(
        stack["qualified"][1], feature_domain_spec_receipt_sha256="9" * 64
    )
    changed = replace(changed, receipt_sha256=semantic_sha256(changed.unsigned()))
    changed.validate()
    with pytest.raises(MassiveQualifiedFinalizedOriginError, match="feature-domain"):
        build_massive_qualified_finalized_origin_plan_v0(
            session_authority=stack["session_authority"],
            exchange="XNYS",
            daily_sources=(stack["qualified"][0], changed),
            first_decision_session_date="2026-08-24",
            last_decision_session_date="2026-08-24",
        )


def test_origin_policy_receipt_rejects_staleness_drift() -> None:
    assert (
        MASSIVE_FINALIZED_ORIGIN_POLICY_V0.receipt_sha256
        == MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256
    )
    assert (
        MASSIVE_FINALIZED_ORIGIN_POLICY_V1.receipt_sha256
        == MASSIVE_FINALIZED_ORIGIN_POLICY_V1_RECEIPT_SHA256
    )
    assert (
        MASSIVE_FINALIZED_ORIGIN_POLICY_V1.readiness_capability_scope
        == "full-feature-to-order-readiness"
    )
    drifted = replace(
        MASSIVE_FINALIZED_ORIGIN_POLICY_V0,
        maximum_source_staleness_sessions=5,
    )
    drifted = replace(drifted, receipt_sha256=semantic_sha256(drifted.unsigned()))
    with pytest.raises(MassiveFinalizedOriginPolicyError, match="staleness"):
        drifted.validate()


def test_manual_listing_and_partition_timing_paths_are_nonauthorizing() -> None:
    parameters = inspect.signature(
        build_massive_qualified_finalized_daily_source_v0
    ).parameters
    assert "captured_listing" in parameters
    assert "listing_root" in parameters
    assert "committed_listing" not in parameters
    assert "listing_entry" not in parameters
    assert "metadata" not in parameters
    assert "readiness_capability" in parameters
    assert "processing_capability" not in parameters


def test_legacy_partition_timing_cannot_authorize_qualified_source(
    tmp_path: Path,
) -> None:
    day = "2026-08-20"
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(day), _session("2026-08-21")),
        source_rows={
            day: (
                _trade_row(
                    ticker="AAA",
                    trade_id="A1",
                    participant_ns=_ns(day, time(15, 59)),
                    sip_ns=_ns(day, time(15, 59)),
                ),
            )
        },
        last_modified={day: _ms("2026-08-21", time(11, 0))},
    )
    _, loaded, session, scan, partition = stack["stacks"][0]
    with pytest.raises(
        MassiveQualifiedFinalizedOriginError,
        match="full feature-to-order readiness",
    ):
        build_massive_qualified_finalized_daily_source_v0(
            listing_root=tmp_path / "listing",
            loaded_source=loaded,
            captured_listing=stack["captured_listing"],
            scan_evidence=scan,
            partition_manifest=partition,
            feature_domain_spec=stack["feature_spec"],
            readiness_capability=stack["legacy_capability"],
            session_authority=stack["session_authority"],
            source_session=session,
        )


def test_readiness_panel_requires_twenty_sessions_across_three_years() -> None:
    too_small = tuple(
        f"2026-01-{day:02d}"
        for day in range(1, MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0)
    )
    with pytest.raises(MassiveFinalizedReadinessError, match="session minimum"):
        build_massive_finalized_readiness_panel_spec_v0(
            source_session_dates=too_small,
            largest_compressed_source_receipt_sha256="1" * 64,
            largest_row_count_source_receipt_sha256="2" * 64,
            correction_activity_session_dates=(too_small[0],),
            high_ticker_count_session_dates=(too_small[-1],),
        )


def test_persisted_partition_v1_reloads_event_active_and_correction_bytes(
    tmp_path: Path,
) -> None:
    day = "2026-08-20"
    rows = (
        _trade_row(
            ticker="AAA",
            trade_id="A1",
            participant_ns=_ns(day, time(15, 55)),
            sip_ns=_ns(day, time(15, 55)),
            price="9.00",
            sequence=1,
        ),
        _trade_row(
            ticker="AAA",
            trade_id="A1",
            participant_ns=_ns(day, time(15, 55)),
            sip_ns=_ns(day, time(16, 5)),
            price="10.00",
            correction=1,
            sequence=2,
        ),
    )
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(day), _session("2026-08-21")),
        source_rows={day: rows},
        last_modified={day: _ms("2026-08-21", time(11, 0))},
    )
    _, _, _, scan, semantic_partition = stack["stacks"][0]
    scanned_rows, repeated_scan = scan_massive_daily_trade_file_v0(
        root=tmp_path / "trades",
        loaded_source=stack["loaded"][day],
        session_authority=stack["session_authority"],
        session=stack["session_authority"].resolve(exchange="XNYS", session_date=day),
        correction_authority=stack["corrections"],
    )
    assert repeated_scan == scan
    streamed_rows = []
    omitted_rows, streamed_scan = scan_massive_daily_trade_file_v0(
        root=tmp_path / "trades",
        loaded_source=stack["loaded"][day],
        session_authority=stack["session_authority"],
        session=stack["session_authority"].resolve(exchange="XNYS", session_date=day),
        correction_authority=stack["corrections"],
        row_sink=streamed_rows.append,
        retain_rows=False,
    )
    assert omitted_rows == ()
    assert tuple(streamed_rows) == scanned_rows
    assert streamed_scan == repeated_scan
    persisted = persist_massive_daily_trade_partitions_v1(
        root=tmp_path / "persisted",
        rows=scanned_rows,
        scan_evidence=scan,
        semantic_partition_manifest=semantic_partition,
        identity_authority=stack["identity"],
        correction_authority=stack["corrections"],
        entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
        published_at_ms=_ms("2026-08-21", time(11, 5)),
    )
    validate_massive_persisted_partitions_v1(
        root=tmp_path / "persisted", manifest=persisted
    )
    assert persisted.source_row_count == 2
    assert persisted.active_event_key_count == 1
    assert persisted.correction_event_count == 1
    assert persisted.partitions[0].active_regular_row_count == 1
    bounded_scan, bounded_semantic, bounded_persisted = (
        stream_and_persist_massive_daily_trade_partitions_v1(
            source_root=tmp_path / "trades",
            loaded_source=stack["loaded"][day],
            spool_root=tmp_path / "spool",
            persisted_root=tmp_path / "bounded-persisted",
            session_authority=stack["session_authority"],
            session=stack["session_authority"].resolve(
                exchange="XNYS", session_date=day
            ),
            identity_authority=stack["identity"],
            condition_authority=stack["conditions"],
            correction_authority=stack["corrections"],
            feature_domain_spec=stack["feature_spec"],
            entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
            published_at_ms=_ms("2026-08-21", time(11, 6)),
        )
    )
    assert bounded_scan == scan
    assert bounded_semantic == semantic_partition
    assert bounded_persisted.source_row_count == persisted.source_row_count
    assert bounded_persisted.active_event_key_count == persisted.active_event_key_count
    assert bounded_persisted.correction_event_count == persisted.correction_event_count
    validate_massive_persisted_partitions_v1(
        root=tmp_path / "bounded-persisted", manifest=bounded_persisted
    )
    active_path = (
        tmp_path
        / "persisted"
        / persisted.partitions[0].active_regular.payload_relative_path
    )
    active_path.chmod(0o644)
    active_path.write_text("{}\n")
    with pytest.raises(Exception):
        validate_massive_persisted_partitions_v1(
            root=tmp_path / "persisted", manifest=persisted
        )


def test_artifact_readiness_v1_measures_source_through_orders(
    tmp_path: Path,
) -> None:
    day = "2026-08-20"
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(day), _session("2026-08-21")),
        source_rows={
            day: (
                _trade_row(
                    ticker="AAA",
                    trade_id="A1",
                    participant_ns=_ns(day, time(15, 59)),
                    sip_ns=_ns(day, time(15, 59)),
                ),
            )
        },
        last_modified={day: _ms("2026-08-21", time(11, 0))},
    )
    listing_loaded = stack["loaded"][day]
    execution_payload = {
        "hardware_contract_receipt_sha256": HARDWARE_RECEIPT,
        "software_source_archive_sha256": SOFTWARE_COMMIT,
        "container_image_receipt_sha256": "1" * 64,
        "python_environment_receipt_sha256": "2" * 64,
    }
    execution_loaded = _publish(
        root=tmp_path / "execution",
        key="artifact-readiness-v1/execution-environment.json",
        payload=canonical_json_file_bytes(execution_payload),
        dataset_id=MASSIVE_ARTIFACT_EXECUTION_DATASET_V1,
        schema_sha256=MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256,
        downloaded_at_ms=listing_loaded.receipt.requested_at_ms - 20,
        etag="execution-environment",
    )
    execution_authority = parse_massive_artifact_execution_authority_v1(
        root=tmp_path / "execution", loaded_source=execution_loaded
    )
    source_payload = read_loaded_massive_source_bytes(
        root=tmp_path / "trades", loaded_source=listing_loaded
    )

    def source_loader(stage_started_at_ms: int):
        root = tmp_path / "measured-source"
        root.mkdir(parents=True, exist_ok=True)
        publish_massive_source_object(
            stream=BytesIO(source_payload),
            root=root,
            relative_payload_path=listing_loaded.receipt.source_object_key,
            dataset_id=listing_loaded.receipt.dataset_id,
            source_object_key=listing_loaded.receipt.source_object_key,
            requested_at_ms=stage_started_at_ms,
            downloaded_at_ms=stage_started_at_ms,
            schema_sha256=listing_loaded.receipt.schema_sha256,
            entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
            committed_at_ms=stage_started_at_ms,
            etag=listing_loaded.receipt.etag,
        )
        measured = load_massive_source_bundle(
            root=root,
            relative_payload_path=listing_loaded.receipt.source_object_key,
            verified_at_ms=stage_started_at_ms,
        )
        return root, measured

    def downstream_runner(
        stage_id: str,
        inputs: tuple[str, ...],
        source_session_date: str,
        stage_started_at_ms: int,
    ):
        semantic_receipt = semantic_sha256((stage_id, inputs, source_session_date))
        output_rows = 1
        payload = {
            "schema": "rl-quant.massive-finalized-readiness-stage-output-v1",
            "stage_id": stage_id,
            "source_session_date": source_session_date,
            "input_artifact_receipts": inputs,
            "semantic_output_receipt_sha256": semantic_receipt,
            "output_row_count": output_rows,
            "implementation_source_sha256": MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
            "protocol_receipt_sha256": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        }
        contract = MASSIVE_ARTIFACT_READINESS_STAGE_CONTRACTS_V1[stage_id]
        root = tmp_path / "downstream"
        root.mkdir(parents=True, exist_ok=True)
        key = f"artifact-v1/{stage_id}.json"
        publish_massive_source_object(
            stream=BytesIO(canonical_json_file_bytes(payload)),
            root=root,
            relative_payload_path=key,
            dataset_id=contract["dataset_id"],
            source_object_key=key,
            requested_at_ms=stage_started_at_ms,
            downloaded_at_ms=stage_started_at_ms,
            schema_sha256=contract["schema_sha256"],
            entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
            committed_at_ms=stage_started_at_ms,
        )
        output = load_massive_source_bundle(
            root=root,
            relative_payload_path=key,
            verified_at_ms=stage_started_at_ms,
        )
        return root, output, semantic_receipt, output_rows

    run = measure_massive_artifact_readiness_v1(
        source_loader=source_loader,
        artifact_root=tmp_path / "artifact-evidence",
        persisted_partition_root=tmp_path / "artifact-partitions",
        execution_authority_root=tmp_path / "execution",
        execution_authority=execution_authority,
        captured_listing=stack["captured_listing"],
        session_authority=stack["session_authority"],
        source_session=stack["session_authority"].resolve(
            exchange="XNYS", session_date=day
        ),
        identity_authority=stack["identity"],
        condition_authority=stack["conditions"],
        correction_authority=stack["corrections"],
        feature_domain_spec=stack["feature_spec"],
        downstream_stage_runner=downstream_runner,
    )
    run.validate()
    assert tuple(stage.stage_id for stage in run.stages) == (
        "source-download-and-commit",
        "whole-file-scan",
        "pit-route-and-finalized-replay",
        "persisted-trade-partitions",
        "daily-features",
        "rolling-features",
        "pit500-decision-tensor",
        "frozen-model-inference",
        "requested-orders",
    )
    assert run.wall_started_at_ms <= run.loaded_source.receipt.requested_at_ms
    assert run.wall_finished_at_ms >= run.stages[-1].stage_finished_at_ms
    cloned = replace(run, source_session=_session("2026-08-19"))
    cloned = replace(cloned, receipt_sha256=semantic_sha256(cloned.unsigned()))
    with pytest.raises(MassiveArtifactReadinessError, match="authorities differ"):
        cloned.validate()
    preexisting = replace(
        run.stages[4],
        stage_started_at_ms=(
            run.stages[4].output_loaded_source.receipt.requested_at_ms + 1
        ),
    )
    preexisting = replace(
        preexisting, receipt_sha256=semantic_sha256(preexisting.unsigned())
    )
    with pytest.raises(
        MassiveArtifactReadinessError,
        match="not created and verified inside",
    ):
        preexisting.validate()
