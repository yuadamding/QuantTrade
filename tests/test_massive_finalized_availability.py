from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive.finalized_availability import (
    MassiveFinalizedAvailabilityError,
    build_massive_finalized_origin_availability_authority_v0,
    build_massive_finalized_source_availability_authority_v0,
    build_massive_vendor_object_metadata_v0,
    select_first_eligible_massive_finalized_origin_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.data_sources.massive.source_receipts import (
    MASSIVE_SOURCE_OBJECT_SCHEMA,
    MassiveSourceObjectReceipt,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


EASTERN = ZoneInfo("America/New_York")
CALENDAR_RECEIPT = "a" * 64
LISTING_RECEIPT = "b" * 64


def _ms(day: str, local_time: time) -> int:
    return int(
        datetime.combine(
            date.fromisoformat(day), local_time, tzinfo=EASTERN
        ).timestamp()
        * 1_000
    )


def _session(day: str, *, close: time = time(16, 0)) -> MassiveExchangeSession:
    regular_open_ns = _ms(day, time(9, 30)) * 1_000_000
    regular_close_ns = _ms(day, close) * 1_000_000
    return MassiveExchangeSession(
        session_date=day,
        exchange="XNYS",
        regular_open_ns=regular_open_ns,
        regular_close_ns=regular_close_ns,
        scheduled_five_minute_intervals=(regular_close_ns - regular_open_ns)
        // (300 * 1_000_000_000),
        special_session_reason=(None if close == time(16, 0) else "early-close"),
        calendar_source_receipt_sha256=CALENDAR_RECEIPT,
    )


def _source_receipt(*, downloaded_at_ms: int) -> MassiveSourceObjectReceipt:
    body = {
        "schema": MASSIVE_SOURCE_OBJECT_SCHEMA,
        "dataset_id": "us_stocks_sip/trades_v1",
        "source_object_key": "2026/08/2026-08-20.csv.gz",
        "requested_at_ms": downloaded_at_ms - 1_000,
        "downloaded_at_ms": downloaded_at_ms,
        "content_length": 123_456,
        "etag": "vendor-etag",
        "request_id": "request-1",
        "physical_sha256": "c" * 64,
        "schema_sha256": "d" * 64,
        "entitlement_receipt_sha256": "e" * 64,
    }
    return MassiveSourceObjectReceipt(
        dataset_id=body["dataset_id"],
        source_object_key=body["source_object_key"],
        requested_at_ms=body["requested_at_ms"],
        downloaded_at_ms=body["downloaded_at_ms"],
        content_length=body["content_length"],
        etag=body["etag"],
        request_id=body["request_id"],
        physical_sha256=body["physical_sha256"],
        schema_sha256=body["schema_sha256"],
        entitlement_receipt_sha256=body["entitlement_receipt_sha256"],
        receipt_sha256=semantic_sha256(body),
    )


def _availability(
    *,
    vendor_time: time,
    decision_day: str = "2026-08-21",
    decision_close: time = time(16, 0),
):
    source_session = _session("2026-08-20")
    decision_session = _session(decision_day, close=decision_close)
    sessions = (source_session, decision_session)
    authority = build_massive_session_authority(
        sessions,
        calendar_source_receipt_sha256=CALENDAR_RECEIPT,
    )
    vendor_at = _ms("2026-08-21", vendor_time)
    source = _source_receipt(downloaded_at_ms=_ms("2026-08-22", time(9, 0)))
    metadata = build_massive_vendor_object_metadata_v0(
        source_object_receipt=source,
        vendor_last_modified_at_ms=vendor_at,
        metadata_observed_at_ms=_ms("2026-08-22", time(8, 0)),
        listing_source_receipt_sha256=LISTING_RECEIPT,
    )
    row = build_massive_finalized_source_availability_authority_v0(
        source_object_receipt=source,
        source_metadata=metadata,
        session_authority=authority,
        source_session=source_session,
        decision_session=decision_session,
        latest_input_observation_at_ms=_ms("2026-08-20", time(16, 0)),
    )
    return row, source, metadata


def test_vendor_available_at_1129_is_eligible_for_1230_decision() -> None:
    row, source, metadata = _availability(vendor_time=time(11, 29))

    assert row.origin_eligible is True
    assert row.ineligibility_reason is None
    assert row.vendor_available_at_ms == _ms("2026-08-21", time(11, 29))
    assert row.vendor_available_at_ms == row.vendor_last_modified_at_ms
    assert row.vendor_available_at_ms == metadata.vendor_available_at_ms
    assert row.vendor_available_at_ms != source.downloaded_at_ms
    assert row.availability_cutoff_at_ms == _ms("2026-08-21", time(11, 30))
    assert row.decision_at_ms == _ms("2026-08-21", time(12, 30))
    assert row.fill_start_at_ms == _ms("2026-08-21", time(15, 50))
    assert row.fill_end_at_ms == _ms("2026-08-21", time(16, 0))
    row.validate()
    assert row.training_gate_eligible is False
    assert metadata.training_gate_eligible is False


def test_vendor_available_at_1131_skips_origin_and_moves_to_next_session() -> None:
    skipped, _, _ = _availability(vendor_time=time(11, 31))
    next_session, _, _ = _availability(
        vendor_time=time(11, 31),
        decision_day="2026-08-24",
    )

    assert skipped.origin_eligible is False
    assert skipped.ineligibility_reason == "vendor-available-after-cutoff"
    assert next_session.origin_eligible is True
    skipped_origin = build_massive_finalized_origin_availability_authority_v0(
        (skipped,)
    )
    next_origin = build_massive_finalized_origin_availability_authority_v0(
        (next_session,)
    )
    selected = select_first_eligible_massive_finalized_origin_v0(
        (skipped_origin, next_origin)
    )
    assert selected == next_origin
    assert selected.decision_session_date == "2026-08-24"


def test_decision_session_observation_cannot_enter_source_features() -> None:
    source_session = _session("2026-08-20")
    decision_session = _session("2026-08-21")
    authority = build_massive_session_authority(
        (source_session, decision_session),
        calendar_source_receipt_sha256=CALENDAR_RECEIPT,
    )
    source = _source_receipt(downloaded_at_ms=_ms("2026-08-22", time(9, 0)))
    metadata = build_massive_vendor_object_metadata_v0(
        source_object_receipt=source,
        vendor_last_modified_at_ms=_ms("2026-08-21", time(11, 0)),
        metadata_observed_at_ms=_ms("2026-08-22", time(8, 0)),
        listing_source_receipt_sha256=LISTING_RECEIPT,
    )

    with pytest.raises(
        MassiveFinalizedAvailabilityError,
        match="decision-session observations",
    ):
        build_massive_finalized_source_availability_authority_v0(
            source_object_receipt=source,
            source_metadata=metadata,
            session_authority=authority,
            source_session=source_session,
            decision_session=decision_session,
            latest_input_observation_at_ms=_ms("2026-08-21", time(9, 31)),
        )


def test_fill_can_never_precede_or_overlap_decision() -> None:
    row, _, _ = _availability(vendor_time=time(11, 29))

    with pytest.raises(
        MassiveFinalizedAvailabilityError,
        match="chronology",
    ):
        replace(row, fill_start_at_ms=row.decision_at_ms).validate()


def test_early_close_session_cannot_host_the_frozen_fill_window() -> None:
    with pytest.raises(
        MassiveFinalizedAvailabilityError,
        match="cannot support",
    ):
        _availability(vendor_time=time(11, 29), decision_close=time(13, 0))


def test_all_required_source_objects_must_be_available() -> None:
    eligible, _, _ = _availability(vendor_time=time(11, 29))
    late, _, _ = _availability(vendor_time=time(11, 31))
    late = replace(
        late,
        source_object_receipt_sha256="f" * 64,
        source_metadata_receipt_sha256="1" * 64,
        source_object_key="2026/08/2026-08-20-aggs.csv.gz",
    )
    late = replace(
        late,
        availability_authority_receipt_sha256=semantic_sha256(late.unsigned()),
    )

    origin = build_massive_finalized_origin_availability_authority_v0((eligible, late))
    assert origin.origin_eligible is False
    assert origin.ineligibility_reasons == ("vendor-available-after-cutoff",)
