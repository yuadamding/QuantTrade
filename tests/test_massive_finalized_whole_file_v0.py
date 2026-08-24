from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, time
import gzip
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
from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
    MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
    MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
    canonical_massive_trade_object_key,
    parse_massive_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_BUCKET,
    MassiveFlatFileListingAcquisitionError,
    capture_massive_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
    build_massive_vendor_object_metadata_from_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin_authority import (
    MassiveQualifiedFinalizedOriginError,
    build_massive_qualified_finalized_daily_source_v0,
    build_massive_qualified_finalized_origin_plan_v0,
)
from rl_quant.data_sources.massive.finalized_origin_policy import (
    MASSIVE_FINALIZED_ORIGIN_POLICY_V0,
    MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256,
    MassiveFinalizedOriginPolicyError,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionError,
    build_massive_finalized_feature_domain_spec_v0,
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
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, semantic_sha256


EASTERN = ZoneInfo("America/New_York")
CALENDAR_RECEIPT = "a" * 64
ENTITLEMENT_RECEIPT = "b" * 64
CONDITION_RECEIPT = "c" * 64
CORRECTION_RECEIPT = "d" * 64
HARDWARE_RECEIPT = "e" * 64
SOFTWARE_COMMIT = "f" * 64


def _ns(day: str, local_time: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(day), local_time, tzinfo=EASTERN).timestamp()
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
    listing_payload = canonical_json_file_bytes(
        {
            "schema": MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
            "observed_at_ms": observed_at_ms,
            "entries": listing_entries,
        }
    )
    listing_key = "massive-flat-file-listing-v0/2026/08/28/listing.json"
    loaded_listing = _publish(
        root=tmp_path / "listing",
        key=listing_key,
        payload=listing_payload,
        dataset_id=MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
        schema_sha256=MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
        downloaded_at_ms=observed_at_ms,
        etag="listing-etag",
    )
    listing = parse_massive_flat_file_listing_v0(
        root=tmp_path / "listing", loaded_listing=loaded_listing
    )
    stacks = []
    benchmarks = []
    for day, loaded in sorted(loaded_by_day.items()):
        session = session_authority.resolve(exchange="XNYS", session_date=day)
        _, scan, partition, benchmark = (
            measure_massive_finalized_source_processing_v0(
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
        )
        stacks.append((day, loaded, session, scan, partition))
        benchmarks.append(benchmark)
    capability = build_massive_finalized_processing_capability_v0(benchmarks)
    qualified = []
    for day, loaded, session, scan, partition in stacks:
        listing_entry = listing.resolve(source_object_key=loaded.receipt.source_object_key)
        metadata = build_massive_vendor_object_metadata_from_listing_v0(
            committed_listing=listing,
            listing_entry=listing_entry,
            loaded_source=loaded,
        )
        qualified.append(
            build_massive_qualified_finalized_daily_source_v0(
                loaded_source=loaded,
                committed_listing=listing,
                listing_entry=listing_entry,
                metadata=metadata,
                scan_evidence=scan,
                partition_manifest=partition,
                feature_domain_spec=feature_spec,
                processing_capability=capability,
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
        "loaded": loaded_by_day,
        "stacks": stacks,
        "benchmarks": tuple(benchmarks),
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
            ticker="AAA", trade_id="A1", participant_ns=_ns(day, time(15, 58)),
            sip_ns=_ns(day, time(15, 58)), sequence=1,
        ),
        _trade_row(
            ticker="AAA", trade_id="A1", participant_ns=_ns(day, time(15, 58)),
            sip_ns=_ns(day, time(16, 5)), price="11.00", correction=1, sequence=2,
        ),
        _trade_row(
            ticker="AAA", trade_id="A2", participant_ns=_ns(day, time(15, 59)),
            sip_ns=_ns(day, time(16, 6)), correction=3, sequence=3,
        ),
        _trade_row(
            ticker="AAA", trade_id="A3", participant_ns=_ns(day, time(18, 0)),
            sip_ns=_ns(day, time(18, 0)), sequence=4,
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
    rows = (_trade_row(ticker="AAA", trade_id="A1", participant_ns=_ns(day, time(15, 59)), sip_ns=_ns(day, time(15, 59))),)
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(day), _session("2026-08-21")),
        source_rows={day: rows},
        last_modified={day: _ms("2026-08-21", time(11, 0))},
    )
    _, loaded, session, scan, partition = stack["stacks"][0]
    forged = replace(partition, receipt_sha256="0" * 64)
    listing_entry = stack["listing"].resolve(source_object_key=loaded.receipt.source_object_key)
    metadata = build_massive_vendor_object_metadata_from_listing_v0(
        committed_listing=stack["listing"], listing_entry=listing_entry, loaded_source=loaded
    )
    with pytest.raises(MassiveDailyTradePartitionError, match="receipt differs"):
        build_massive_qualified_finalized_daily_source_v0(
            loaded_source=loaded,
            committed_listing=stack["listing"],
            listing_entry=listing_entry,
            metadata=metadata,
            scan_evidence=scan,
            partition_manifest=forged,
            feature_domain_spec=stack["feature_spec"],
            processing_capability=stack["capability"],
            session_authority=stack["session_authority"],
            source_session=session,
        )


def test_processing_over_55_minutes_blocks_source(tmp_path: Path) -> None:
    day = "2026-08-20"
    rows = (_trade_row(ticker="AAA", trade_id="A1", participant_ns=_ns(day, time(15, 59)), sip_ns=_ns(day, time(15, 59))),)
    stack = _qualified_sources(
        tmp_path,
        sessions=(_session(day), _session("2026-08-21")),
        source_rows={day: rows},
        last_modified={day: _ms("2026-08-21", time(11, 0))},
    )
    too_slow = MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS + 1
    capability = replace(
        stack["capability"],
        observed_runtime_ms=(too_slow,),
        p95_runtime_ms=too_slow,
        p99_runtime_ms=too_slow,
        maximum_runtime_ms=too_slow,
        capability_passed=False,
    )
    capability = replace(capability, receipt_sha256=semantic_sha256(capability.unsigned()))
    capability.validate()
    _, loaded, session, scan, partition = stack["stacks"][0]
    listing_entry = stack["listing"].resolve(source_object_key=loaded.receipt.source_object_key)
    metadata = build_massive_vendor_object_metadata_from_listing_v0(
        committed_listing=stack["listing"], listing_entry=listing_entry, loaded_source=loaded
    )
    with pytest.raises(MassiveQualifiedFinalizedOriginError, match="does not cover"):
        build_massive_qualified_finalized_daily_source_v0(
            loaded_source=loaded,
            committed_listing=stack["listing"],
            listing_entry=listing_entry,
            metadata=metadata,
            scan_evidence=scan,
            partition_manifest=partition,
            feature_domain_spec=stack["feature_spec"],
            processing_capability=capability,
            session_authority=stack["session_authority"],
            source_session=session,
        )


def test_plan_skips_early_close_and_late_open_but_accepts_fill_at_close(
    tmp_path: Path,
) -> None:
    source_day = "2026-08-20"
    rows = (_trade_row(ticker="AAA", trade_id="A1", participant_ns=_ns(source_day, time(15, 59)), sip_ns=_ns(source_day, time(15, 59))),)
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
        day: (_trade_row(ticker="AAA", trade_id=f"A-{day}", participant_ns=_ns(day, time(15, 59)), sip_ns=_ns(day, time(15, 59))),)
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
    changed = replace(stack["qualified"][1], feature_domain_spec_receipt_sha256="9" * 64)
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
    drifted = replace(
        MASSIVE_FINALIZED_ORIGIN_POLICY_V0,
        maximum_source_staleness_sessions=5,
    )
    drifted = replace(drifted, receipt_sha256=semantic_sha256(drifted.unsigned()))
    with pytest.raises(MassiveFinalizedOriginPolicyError, match="staleness"):
        drifted.validate()
