from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time
import gzip
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
    MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
    MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
    MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
    MassiveFinalizedListingError,
    canonical_massive_trade_object_key,
    parse_massive_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_PROCESSING_SPEC_V0,
    MASSIVE_FINALIZED_TOTAL_READINESS_ALLOWANCE_MS,
    MassiveFinalizedOriginError,
    build_massive_feature_input_cutoff_evidence_v0,
    build_massive_finalized_daily_source_evidence_v0,
    build_massive_finalized_decision_origin_plan_v0,
    build_massive_finalized_source_coverage_v0,
    build_massive_vendor_object_metadata_from_listing_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADES_DATASET_ID,
    MASSIVE_FLAT_TRADE_COLUMNS,
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MassiveTradeExtractionError,
    extract_massive_flat_file_security_session,
)
from rl_quant.data_sources.massive.trade_replay import MassiveResolvedSecurityIdentity
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)


EASTERN = ZoneInfo("America/New_York")
CALENDAR_RECEIPT = "a" * 64
ENTITLEMENT_RECEIPT = "b" * 64
FEATURE_SPEC_RECEIPT = "c" * 64


def _ms(day: str, local_time: time) -> int:
    return int(
        datetime.combine(
            date.fromisoformat(day), local_time, tzinfo=EASTERN
        ).timestamp()
        * 1_000
    )


def _session(day: str, *, close: time = time(16, 0)) -> MassiveExchangeSession:
    opened = _ms(day, time(9, 30)) * 1_000_000
    closed = _ms(day, close) * 1_000_000
    return MassiveExchangeSession(
        session_date=day,
        exchange="XNYS",
        regular_open_ns=opened,
        regular_close_ns=closed,
        scheduled_five_minute_intervals=(closed - opened) // (300 * 1_000_000_000),
        special_session_reason=(None if close == time(16, 0) else "early-close"),
        calendar_source_receipt_sha256=CALENDAR_RECEIPT,
    )


def _flat_payload(day: str) -> bytes:
    sip_ns = _ms(day, time(15, 59)) * 1_000_000
    header = ",".join(MASSIVE_FLAT_TRADE_COLUMNS)
    values = (
        "AAA",
        "[1]",
        "0",
        "4",
        f"trade-{day}",
        str(sip_ns - 1_000_000),
        "10.00",
        "1",
        str(sip_ns),
        "100",
        "1",
        "12",
        str(sip_ns),
    )
    return gzip.compress(f"{header}\n{','.join(values)}\n".encode(), mtime=0)


def _publish_loaded(
    *,
    root: Path,
    relative: str,
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
        relative_payload_path=relative,
        dataset_id=dataset_id,
        source_object_key=relative,
        requested_at_ms=downloaded_at_ms - 1,
        downloaded_at_ms=downloaded_at_ms,
        schema_sha256=schema_sha256,
        entitlement_receipt_sha256=ENTITLEMENT_RECEIPT,
        committed_at_ms=downloaded_at_ms + 1,
        etag=etag,
    )
    return load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=downloaded_at_ms + 2,
    )


def _source_stack(
    tmp_path: Path,
    *,
    source_last_modified: dict[str, int],
    sessions: tuple[MassiveExchangeSession, ...],
):
    session_authority = build_massive_session_authority(
        sessions,
        calendar_source_receipt_sha256=CALENDAR_RECEIPT,
    )
    trade_root = tmp_path / "trades"
    loaded_trades = {}
    listing_entries = []
    observed_at = max(source_last_modified.values()) + 2 * 60 * 60 * 1_000
    for index, (source_day, modified_at) in enumerate(
        sorted(source_last_modified.items())
    ):
        key = canonical_massive_trade_object_key(source_day)
        payload = _flat_payload(source_day)
        etag = f"etag-{index}"
        loaded = _publish_loaded(
            root=trade_root,
            relative=key,
            payload=payload,
            dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
            schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
            downloaded_at_ms=observed_at + 1_000 + index,
            etag=etag,
        )
        loaded_trades[source_day] = loaded
        listing_entries.append(
            {
                "dataset_id": MASSIVE_FLAT_TRADES_DATASET_ID,
                "source_object_key": key,
                "etag": etag,
                "content_length": len(payload),
                "last_modified_at_ms": modified_at,
            }
        )
    listing_payload = {
        "schema": MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
        "observed_at_ms": observed_at,
        "entries": sorted(
            listing_entries, key=lambda entry: str(entry["source_object_key"])
        ),
    }
    listing_key = "massive-flat-file-listing-v0/2026/08/25/listing.json"
    loaded_listing = _publish_loaded(
        root=tmp_path / "listing",
        relative=listing_key,
        payload=canonical_json_file_bytes(listing_payload),
        dataset_id=MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
        schema_sha256=MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
        downloaded_at_ms=observed_at,
        etag="listing-etag",
    )
    listing = parse_massive_flat_file_listing_v0(
        root=tmp_path / "listing",
        loaded_listing=loaded_listing,
    )
    daily_sources = []
    for source_day, loaded in sorted(loaded_trades.items()):
        source_session = session_authority.resolve(
            exchange="XNYS", session_date=source_day
        )
        identity = MassiveResolvedSecurityIdentity.build(
            security_id="SEC-A",
            source_ticker="AAA",
            primary_exchange="XNYS",
            session_date=source_day,
            valid_from_ns=source_session.regular_open_ns - 1,
            valid_to_ns=None,
            identity_authority_receipt_sha256="d" * 64,
            ticker_history_receipt_sha256="e" * 64,
        )
        rows, extraction = extract_massive_flat_file_security_session(
            root=trade_root,
            loaded_source=loaded,
            identity_resolution=identity,
        )
        coverage = build_massive_finalized_source_coverage_v0(
            loaded_source=loaded,
            extraction_evidence=extraction,
        )
        cutoff = build_massive_feature_input_cutoff_evidence_v0(
            extracted_rows=rows,
            extraction_evidence=extraction,
            source_partition_receipts=(semantic_sha256((source_day, "partition")),),
            feature_spec_receipt_sha256=FEATURE_SPEC_RECEIPT,
        )
        entry = listing.resolve(source_object_key=loaded.receipt.source_object_key)
        metadata = build_massive_vendor_object_metadata_from_listing_v0(
            committed_listing=listing,
            listing_entry=entry,
            loaded_source=loaded,
        )
        daily_sources.append(
            build_massive_finalized_daily_source_evidence_v0(
                loaded_source=loaded,
                committed_listing=listing,
                listing_entry=entry,
                metadata=metadata,
                coverage=coverage,
                feature_input_cutoff=cutoff,
                session_authority=session_authority,
                source_session=source_session,
            )
        )
    return session_authority, tuple(daily_sources), listing, loaded_trades


def test_committed_listing_derives_exact_key_etag_size_and_last_modified(
    tmp_path: Path,
) -> None:
    sessions = (_session("2026-08-20"), _session("2026-08-21"))
    modified = _ms("2026-08-21", time(11, 29))
    _, sources, listing, loaded = _source_stack(
        tmp_path,
        source_last_modified={"2026-08-20": modified},
        sessions=sessions,
    )

    entry = listing.entries[0]
    source = loaded["2026-08-20"]
    assert entry.source_role == "finalized-trades-v1"
    assert entry.source_object_key == (
        "us_stocks_sip/trades_v1/2026/08/2026-08-20.csv.gz"
    )
    assert entry.coverage_session_date == "2026-08-20"
    assert entry.etag == source.receipt.etag
    assert entry.content_length == source.receipt.content_length
    assert entry.vendor_last_modified_at_ms == modified
    assert sources[0].feature_ready_at_ms == (
        modified + MASSIVE_FINALIZED_TOTAL_READINESS_ALLOWANCE_MS
    )


def test_wrong_date_or_noncanonical_trade_object_key_is_rejected() -> None:
    with pytest.raises(MassiveFinalizedListingError, match="invalid"):
        canonical_massive_trade_object_key("not-a-date")
    with pytest.raises(MassiveFinalizedListingError, match="layout"):
        from rl_quant.data_sources.massive.finalized_listing import (
            coverage_session_from_massive_trade_key,
        )

        coverage_session_from_massive_trade_key("2026/08/2026-08-20.csv.gz")


def test_committed_trade_object_cannot_be_paired_with_another_source_date(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wrong-date"
    key = canonical_massive_trade_object_key("2026-08-19")
    loaded = _publish_loaded(
        root=root,
        relative=key,
        payload=_flat_payload("2026-08-20"),
        dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
        schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
        downloaded_at_ms=_ms("2026-08-21", time(12, 0)),
        etag="wrong-date-etag",
    )
    source_session = _session("2026-08-20")
    identity = MassiveResolvedSecurityIdentity.build(
        security_id="SEC-A",
        source_ticker="AAA",
        primary_exchange="XNYS",
        session_date="2026-08-20",
        valid_from_ns=source_session.regular_open_ns - 1,
        valid_to_ns=None,
        identity_authority_receipt_sha256="d" * 64,
        ticker_history_receipt_sha256="e" * 64,
    )

    with pytest.raises(MassiveTradeExtractionError, match="object key differs"):
        extract_massive_flat_file_security_session(
            root=root,
            loaded_source=loaded,
            identity_resolution=identity,
        )


def test_listing_entry_last_modified_cannot_be_forged(tmp_path: Path) -> None:
    sessions = (_session("2026-08-20"), _session("2026-08-21"))
    _, _, listing, loaded = _source_stack(
        tmp_path,
        source_last_modified={"2026-08-20": _ms("2026-08-21", time(11, 29))},
        sessions=sessions,
    )
    original = listing.entries[0]
    forged = replace(
        original,
        vendor_last_modified_at_ms=original.vendor_last_modified_at_ms - 60_000,
    )
    forged = replace(forged, receipt_sha256=semantic_sha256(forged.unsigned()))

    with pytest.raises(MassiveFinalizedOriginError, match="not resolved"):
        build_massive_vendor_object_metadata_from_listing_v0(
            committed_listing=listing,
            listing_entry=forged,
            loaded_source=loaded["2026-08-20"],
        )


def test_feature_cutoff_is_derived_from_parser_rows(tmp_path: Path) -> None:
    sessions = (_session("2026-08-20"), _session("2026-08-21"))
    session_authority, sources, _, _ = _source_stack(
        tmp_path,
        source_last_modified={"2026-08-20": _ms("2026-08-21", time(11, 29))},
        sessions=sessions,
    )

    source = sources[0]
    assert source.maximum_input_timestamp_ms == _ms("2026-08-20", time(15, 59))
    assert source.maximum_input_timestamp_ms < source.source_feature_cutoff_at_ms
    assert source.session_authority_receipt_sha256 == session_authority.receipt_sha256


def test_decision_plan_is_exhaustive_unique_and_uses_latest_ready_source(
    tmp_path: Path,
) -> None:
    sessions = (
        _session("2026-08-20"),
        _session("2026-08-21"),
        _session("2026-08-24"),
        _session("2026-08-25"),
    )
    authority, sources, _, _ = _source_stack(
        tmp_path,
        source_last_modified={
            "2026-08-20": _ms("2026-08-21", time(11, 31)),
            "2026-08-21": _ms("2026-08-22", time(11, 0)),
        },
        sessions=sessions,
    )
    plan = build_massive_finalized_decision_origin_plan_v0(
        session_authority=authority,
        exchange="XNYS",
        daily_sources=sources,
        first_decision_session_date="2026-08-21",
        last_decision_session_date="2026-08-25",
    )

    assert plan.candidate_decision_session_dates == (
        "2026-08-21",
        "2026-08-24",
        "2026-08-25",
    )
    assert tuple(row.decision_session_date for row in plan.origins) == (
        "2026-08-24",
        "2026-08-25",
    )
    assert plan.origins[0].source_session_date == "2026-08-21"
    assert plan.origins[0].source_staleness_sessions == 1
    assert plan.origins[0].required_source_roles == (
        MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES
    )
    assert tuple(row.decision_session_date for row in plan.skipped_decisions) == (
        "2026-08-21",
    )
    assert len({row.decision_session_date for row in plan.origins}) == len(plan.origins)
    assert plan.source_origin_authority_closed is True
    assert plan.panel_materialization_authorized is False

    omitted = replace(
        plan,
        candidate_decision_session_dates=("2026-08-21", "2026-08-25"),
    )
    omitted = replace(omitted, receipt_sha256=semantic_sha256(omitted.unsigned()))
    with pytest.raises(MassiveFinalizedOriginError, match="omitted"):
        omitted.validate()

    duplicated = replace(plan, origins=plan.origins + (plan.origins[0],))
    duplicated = replace(
        duplicated, receipt_sha256=semantic_sha256(duplicated.unsigned())
    )
    with pytest.raises(MassiveFinalizedOriginError, match="not derived"):
        duplicated.validate()


def test_unknown_or_duplicate_source_roles_cannot_enter_plan(tmp_path: Path) -> None:
    sessions = (_session("2026-08-20"), _session("2026-08-21"))
    authority, sources, _, _ = _source_stack(
        tmp_path,
        source_last_modified={"2026-08-20": _ms("2026-08-21", time(11, 29))},
        sessions=sessions,
    )
    unknown = replace(sources[0], source_role="unexpected-source")
    unknown = replace(unknown, receipt_sha256=semantic_sha256(unknown.unsigned()))

    with pytest.raises(MassiveFinalizedOriginError, match="source role"):
        build_massive_finalized_decision_origin_plan_v0(
            session_authority=authority,
            exchange="XNYS",
            daily_sources=(unknown,),
            first_decision_session_date="2026-08-21",
            last_decision_session_date="2026-08-21",
        )
    with pytest.raises(MassiveFinalizedOriginError, match="sorted and unique"):
        build_massive_finalized_decision_origin_plan_v0(
            session_authority=authority,
            exchange="XNYS",
            daily_sources=(sources[0], sources[0]),
            first_decision_session_date="2026-08-21",
            last_decision_session_date="2026-08-21",
        )
    with pytest.raises(MassiveFinalizedOriginError, match="inventory is empty"):
        build_massive_finalized_decision_origin_plan_v0(
            session_authority=authority,
            exchange="XNYS",
            daily_sources=(),
            first_decision_session_date="2026-08-21",
            last_decision_session_date="2026-08-21",
        )


def test_processing_budget_gives_1129_readiness_and_rejects_1131(
    tmp_path: Path,
) -> None:
    sessions = (_session("2026-08-20"), _session("2026-08-21"))
    authority, early_sources, _, _ = _source_stack(
        tmp_path / "early",
        source_last_modified={"2026-08-20": _ms("2026-08-21", time(11, 29))},
        sessions=sessions,
    )
    early = build_massive_finalized_decision_origin_plan_v0(
        session_authority=authority,
        exchange="XNYS",
        daily_sources=early_sources,
        first_decision_session_date="2026-08-21",
        last_decision_session_date="2026-08-21",
    )
    late_authority, late_sources, _, _ = _source_stack(
        tmp_path / "late",
        source_last_modified={"2026-08-20": _ms("2026-08-21", time(11, 31))},
        sessions=sessions,
    )
    late = build_massive_finalized_decision_origin_plan_v0(
        session_authority=late_authority,
        exchange="XNYS",
        daily_sources=late_sources,
        first_decision_session_date="2026-08-21",
        last_decision_session_date="2026-08-21",
    )

    assert MASSIVE_FINALIZED_PROCESSING_SPEC_V0.total_readiness_allowance_ms == (
        60 * 60 * 1_000
    )
    assert len(early.origins) == 1
    assert not early.skipped_decisions
    assert not late.origins
    assert late.skipped_decisions[0].reason == (
        "no-ready-source-within-staleness-bound"
    )


def test_source_staleness_is_bounded_at_three_trading_sessions(
    tmp_path: Path,
) -> None:
    sessions = (
        _session("2026-08-20"),
        _session("2026-08-21"),
        _session("2026-08-24"),
        _session("2026-08-25"),
        _session("2026-08-26"),
        _session("2026-08-27"),
    )
    authority, sources, _, _ = _source_stack(
        tmp_path,
        source_last_modified={"2026-08-20": _ms("2026-08-21", time(11, 0))},
        sessions=sessions,
    )
    plan = build_massive_finalized_decision_origin_plan_v0(
        session_authority=authority,
        exchange="XNYS",
        daily_sources=sources,
        first_decision_session_date="2026-08-25",
        last_decision_session_date="2026-08-26",
    )

    assert plan.origins[0].decision_session_date == "2026-08-25"
    assert plan.origins[0].source_staleness_sessions == 3
    assert plan.skipped_decisions[0].decision_session_date == "2026-08-26"
    assert plan.skipped_decisions[0].reason == (
        "no-ready-source-within-staleness-bound"
    )
