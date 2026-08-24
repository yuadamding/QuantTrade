"""Authenticated, paginated S3 listing capture for Massive flat files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
    MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
    MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
    MassiveCommittedFlatFileListingV0,
    canonical_massive_trade_object_key,
    coverage_session_from_massive_trade_key,
    parse_massive_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.data_sources.massive.trade_extraction import MASSIVE_FLAT_TRADES_DATASET_ID
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, semantic_sha256


MASSIVE_FLAT_FILE_ENDPOINT = "https://files.massive.com"
MASSIVE_FLAT_FILE_BUCKET = "flatfiles"
MASSIVE_FLAT_FILE_LISTING_ACQUISITION_DATASET_ID = (
    "massive-flat-file-listing-acquisition-v0"
)
MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA = (
    "rl-quant.massive-flat-file-listing-acquisition-v0"
)
MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA,
        "transport": "S3-compatible-list-objects-v2-s3v4",
        "endpoint": MASSIVE_FLAT_FILE_ENDPOINT,
        "bucket": MASSIVE_FLAT_FILE_BUCKET,
        "pagination": "complete-provider-paginator",
        "credentials": "environment-variable-names-only",
    }
)


class MassiveFlatFileListingAcquisitionError(ValueError):
    """The provider listing was incomplete, unauthenticated, or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFlatFileListingAcquisitionError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveFlatFileListingPageEvidenceV0:
    page_index: int
    request_id: str
    is_truncated: bool
    key_count: int
    key_inventory_sha256: str
    next_continuation_token_sha256: str | None

    def validate(self, *, page_count: int) -> None:
        if (
            isinstance(self.page_index, bool)
            or not isinstance(self.page_index, int)
            or not 0 <= self.page_index < page_count
            or not self.request_id
            or self.request_id != self.request_id.strip()
            or isinstance(self.key_count, bool)
            or not isinstance(self.key_count, int)
            or self.key_count < 0
        ):
            raise MassiveFlatFileListingAcquisitionError("page evidence is invalid")
        if self.is_truncated is not (self.page_index < page_count - 1):
            raise MassiveFlatFileListingAcquisitionError("page truncation chain differs")
        _digest("page key inventory", self.key_inventory_sha256)
        if self.next_continuation_token_sha256 is not None:
            _digest(
                "continuation token inventory",
                self.next_continuation_token_sha256,
            )
        if self.is_truncated is not (self.next_continuation_token_sha256 is not None):
            raise MassiveFlatFileListingAcquisitionError(
                "continuation token evidence differs from truncation"
            )


@dataclass(frozen=True, slots=True)
class MassiveFlatFileListingAcquisitionEvidenceV0:
    endpoint_url: str
    bucket: str
    prefix: str
    requested_at_ms: int
    completed_at_ms: int
    page_count: int
    provider_request_ids: tuple[str, ...]
    page_metadata: tuple[MassiveFlatFileListingPageEvidenceV0, ...]
    page_metadata_inventory_sha256: str
    object_count: int
    object_inventory_sha256: str
    access_key_environment_variable: str
    secret_key_environment_variable: str
    signature_version: str
    receipt_sha256: str
    schema: str = MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA:
            raise MassiveFlatFileListingAcquisitionError("acquisition schema drifted")
        if self.endpoint_url != MASSIVE_FLAT_FILE_ENDPOINT or self.bucket != MASSIVE_FLAT_FILE_BUCKET:
            raise MassiveFlatFileListingAcquisitionError("flat-file endpoint/bucket drifted")
        if not self.prefix.startswith(f"{MASSIVE_FLAT_TRADES_DATASET_ID}/"):
            raise MassiveFlatFileListingAcquisitionError("listing prefix drifted")
        if (
            self.requested_at_ms < 0
            or self.completed_at_ms < self.requested_at_ms
            or self.page_count <= 0
            or self.object_count <= 0
        ):
            raise MassiveFlatFileListingAcquisitionError("acquisition chronology/counts are invalid")
        if (
            len(self.provider_request_ids) != self.page_count
            or len(set(self.provider_request_ids)) != self.page_count
            or any(not value or value != value.strip() for value in self.provider_request_ids)
        ):
            raise MassiveFlatFileListingAcquisitionError("provider request inventory differs")
        if len(self.page_metadata) != self.page_count:
            raise MassiveFlatFileListingAcquisitionError("page metadata count differs")
        for index, row in enumerate(self.page_metadata):
            row.validate(page_count=self.page_count)
            if row.page_index != index or row.request_id != self.provider_request_ids[index]:
                raise MassiveFlatFileListingAcquisitionError("page metadata order differs")
        if self.page_metadata_inventory_sha256 != semantic_sha256(
            tuple(asdict(row) for row in self.page_metadata)
        ):
            raise MassiveFlatFileListingAcquisitionError("page metadata inventory differs")
        for name in (
            "page_metadata_inventory_sha256",
            "object_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in (
            self.access_key_environment_variable,
            self.secret_key_environment_variable,
        ):
            if not name or name != name.strip() or any(character.isspace() for character in name):
                raise MassiveFlatFileListingAcquisitionError("credential source name is invalid")
        if self.signature_version != "s3v4":
            raise MassiveFlatFileListingAcquisitionError("signature version drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFlatFileListingAcquisitionError("acquisition receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveCapturedFlatFileListingV0:
    acquisition_evidence: MassiveFlatFileListingAcquisitionEvidenceV0
    loaded_acquisition: LoadedMassiveSourceObject
    loaded_listing: LoadedMassiveSourceObject
    committed_listing: MassiveCommittedFlatFileListingV0

    def validate(self) -> None:
        self.acquisition_evidence.validate()
        self.loaded_acquisition.validate()
        self.loaded_listing.validate()
        self.committed_listing.validate()
        if (
            self.loaded_acquisition.receipt.dataset_id
            != MASSIVE_FLAT_FILE_LISTING_ACQUISITION_DATASET_ID
            or self.loaded_acquisition.receipt.schema_sha256
            != MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA_SHA256
            or self.loaded_listing.receipt.dataset_id
            != MASSIVE_FLAT_FILE_LISTING_DATASET_ID
            or self.loaded_listing.receipt.schema_sha256
            != MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256
            or self.committed_listing.loaded_listing_receipt_sha256
            != self.loaded_listing.receipt_sha256
        ):
            raise MassiveFlatFileListingAcquisitionError("captured listing links differ")


def _last_modified_ms(value: object) -> int:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise MassiveFlatFileListingAcquisitionError(
            "provider LastModified must be timezone-aware"
        )
    return int(value.timestamp() * 1_000)


def capture_massive_flat_file_listing_v0(
    *,
    s3_client: Any,
    root: str | Path,
    year: int,
    month: int,
    entitlement_receipt_sha256: str,
    access_key_environment_variable: str = "MASSIVE_S3_ACCESS_KEY_ID",
    secret_key_environment_variable: str = "MASSIVE_S3_SECRET_ACCESS_KEY",
    now_ms: Callable[[], int] | None = None,
) -> MassiveCapturedFlatFileListingV0:
    """Exhaust the provider paginator and publish exact secret-free listing evidence."""

    entitlement = _digest("entitlement receipt", entitlement_receipt_sha256)
    if year < 2000 or not 1 <= month <= 12:
        raise MassiveFlatFileListingAcquisitionError("listing year/month is invalid")
    clock = now_ms or (lambda: time.time_ns() // 1_000_000)
    requested_at_ms = clock()
    prefix = f"{MASSIVE_FLAT_TRADES_DATASET_ID}/{year:04d}/{month:02d}/"
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = tuple(
            paginator.paginate(Bucket=MASSIVE_FLAT_FILE_BUCKET, Prefix=prefix)
        )
    except Exception as exc:  # pragma: no cover - SDK exception classes are optional
        raise MassiveFlatFileListingAcquisitionError("provider listing request failed") from exc
    completed_at_ms = clock()
    if not pages:
        raise MassiveFlatFileListingAcquisitionError("provider paginator returned no pages")
    provider_request_ids: list[str] = []
    page_metadata: list[MassiveFlatFileListingPageEvidenceV0] = []
    entries: list[dict[str, object]] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            raise MassiveFlatFileListingAcquisitionError("provider page is malformed")
        response = page.get("ResponseMetadata")
        request_id = response.get("RequestId") if isinstance(response, Mapping) else None
        if not isinstance(request_id, str) or not request_id:
            raise MassiveFlatFileListingAcquisitionError("provider request ID is absent")
        provider_request_ids.append(request_id)
        truncated = page.get("IsTruncated") is True
        if truncated is not (page_index < len(pages) - 1):
            raise MassiveFlatFileListingAcquisitionError("provider pagination did not close exactly")
        contents = page.get("Contents", ())
        if not isinstance(contents, (list, tuple)):
            raise MassiveFlatFileListingAcquisitionError("provider contents are malformed")
        page_keys: list[str] = []
        for item in contents:
            if not isinstance(item, Mapping):
                raise MassiveFlatFileListingAcquisitionError("provider object is malformed")
            key = item.get("Key")
            etag = item.get("ETag")
            size = item.get("Size")
            if (
                not isinstance(key, str)
                or not key.startswith(prefix)
                or not isinstance(etag, str)
                or not etag
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise MassiveFlatFileListingAcquisitionError("provider object metadata is invalid")
            session_date = coverage_session_from_massive_trade_key(key)
            if key != canonical_massive_trade_object_key(session_date):
                raise MassiveFlatFileListingAcquisitionError("provider object key drifted")
            page_keys.append(key)
            entries.append(
                {
                    "dataset_id": MASSIVE_FLAT_TRADES_DATASET_ID,
                    "source_object_key": key,
                    "etag": etag.removeprefix('"').removesuffix('"'),
                    "content_length": size,
                    "last_modified_at_ms": _last_modified_ms(item.get("LastModified")),
                }
            )
        token = page.get("NextContinuationToken")
        page_metadata.append(
            MassiveFlatFileListingPageEvidenceV0(
                page_index=page_index,
                request_id=request_id,
                is_truncated=truncated,
                key_count=len(page_keys),
                key_inventory_sha256=semantic_sha256(tuple(sorted(page_keys))),
                next_continuation_token_sha256=None
                if token is None
                else semantic_sha256(str(token)),
            )
        )
    ordered_entries = tuple(sorted(entries, key=lambda row: str(row["source_object_key"])))
    keys = tuple(str(row["source_object_key"]) for row in ordered_entries)
    if not keys or keys != tuple(sorted(set(keys))):
        raise MassiveFlatFileListingAcquisitionError("provider objects are empty or duplicate")
    page_rows = tuple(page_metadata)
    acquisition_body: dict[str, object] = {
        "schema": MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA,
        "endpoint_url": MASSIVE_FLAT_FILE_ENDPOINT,
        "bucket": MASSIVE_FLAT_FILE_BUCKET,
        "prefix": prefix,
        "requested_at_ms": requested_at_ms,
        "completed_at_ms": completed_at_ms,
        "page_count": len(pages),
        "provider_request_ids": tuple(provider_request_ids),
        "page_metadata": page_rows,
        "page_metadata_inventory_sha256": semantic_sha256(
            tuple(asdict(row) for row in page_rows)
        ),
        "object_count": len(ordered_entries),
        "object_inventory_sha256": semantic_sha256(ordered_entries),
        "access_key_environment_variable": access_key_environment_variable,
        "secret_key_environment_variable": secret_key_environment_variable,
        "signature_version": "s3v4",
    }
    provisional = MassiveFlatFileListingAcquisitionEvidenceV0(
        **acquisition_body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,
    )
    acquisition = MassiveFlatFileListingAcquisitionEvidenceV0(
        **acquisition_body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    acquisition.validate()
    root_path = Path(root)
    capture_day = datetime.fromtimestamp(
        completed_at_ms / 1_000, tz=ZoneInfo("America/New_York")
    ).date()
    base = f"massive-flat-file-listing-v0/{capture_day:%Y/%m/%d}"
    acquisition_key = f"{base}/acquisition-{year:04d}-{month:02d}.json"
    listing_key = f"{base}/listing-{year:04d}-{month:02d}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(acquisition.unsigned() | {"receipt_sha256": acquisition.receipt_sha256})),
        root=root_path,
        relative_payload_path=acquisition_key,
        dataset_id=MASSIVE_FLAT_FILE_LISTING_ACQUISITION_DATASET_ID,
        source_object_key=acquisition_key,
        requested_at_ms=requested_at_ms,
        downloaded_at_ms=completed_at_ms,
        schema_sha256=MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement,
        committed_at_ms=completed_at_ms,
    )
    listing_payload = {
        "schema": MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA,
        "observed_at_ms": completed_at_ms,
        "entries": ordered_entries,
    }
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(listing_payload)),
        root=root_path,
        relative_payload_path=listing_key,
        dataset_id=MASSIVE_FLAT_FILE_LISTING_DATASET_ID,
        source_object_key=listing_key,
        requested_at_ms=requested_at_ms,
        downloaded_at_ms=completed_at_ms,
        schema_sha256=MASSIVE_FLAT_FILE_LISTING_CAPTURE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement,
        committed_at_ms=completed_at_ms,
    )
    loaded_acquisition = load_massive_source_bundle(
        root=root_path,
        relative_payload_path=acquisition_key,
        verified_at_ms=completed_at_ms,
    )
    loaded_listing = load_massive_source_bundle(
        root=root_path,
        relative_payload_path=listing_key,
        verified_at_ms=completed_at_ms,
    )
    committed_listing = parse_massive_flat_file_listing_v0(
        root=root_path, loaded_listing=loaded_listing
    )
    result = MassiveCapturedFlatFileListingV0(
        acquisition_evidence=acquisition,
        loaded_acquisition=loaded_acquisition,
        loaded_listing=loaded_listing,
        committed_listing=committed_listing,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_FLAT_FILE_BUCKET",
    "MASSIVE_FLAT_FILE_ENDPOINT",
    "MASSIVE_FLAT_FILE_LISTING_ACQUISITION_DATASET_ID",
    "MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA",
    "MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA_SHA256",
    "MassiveCapturedFlatFileListingV0",
    "MassiveFlatFileListingAcquisitionError",
    "MassiveFlatFileListingAcquisitionEvidenceV0",
    "MassiveFlatFileListingPageEvidenceV0",
    "capture_massive_flat_file_listing_v0",
]
