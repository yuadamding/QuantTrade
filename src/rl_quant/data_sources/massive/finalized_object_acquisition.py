"""Authenticated Massive flat-file object GET and immutable publication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping

from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_BUCKET,
    MASSIVE_FLAT_FILE_ENDPOINT,
    MassiveCapturedFlatFileListingV0,
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
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SCHEMA = (
    "rl-quant.massive-authenticated-flat-file-object-get-v1"
)
MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256 = semantic_sha256(
    {
        "endpoint": MASSIVE_FLAT_FILE_ENDPOINT,
        "bucket": MASSIVE_FLAT_FILE_BUCKET,
        "operation": "authenticated-s3v4-get-object",
        "listing_binding": ("key", "etag", "content-length"),
        "publication": "streaming-create-only-source-transaction",
        "request_evidence": ("provider-request-id", "version-id-if-present"),
    }
)


class MassiveAuthenticatedObjectGetError(ValueError):
    """Authenticated object GET evidence is incomplete or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAuthenticatedObjectGetError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAuthenticatedFlatFileDownloadV1:
    endpoint_url: str
    bucket: str
    source_object_key: str
    listing_acquisition_receipt_sha256: str
    listing_entry_receipt_sha256: str
    provider_request_id: str
    provider_version_id: str | None
    requested_at_ms: int
    completed_at_ms: int
    etag: str
    content_length: int
    loaded_source: LoadedMassiveSourceObject
    acquisition_spec_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SCHEMA
            or self.endpoint_url != MASSIVE_FLAT_FILE_ENDPOINT
            or self.bucket != MASSIVE_FLAT_FILE_BUCKET
            or not self.source_object_key
            or not self.provider_request_id
            or not self.etag
            or isinstance(self.content_length, bool)
            or self.content_length <= 0
            or self.requested_at_ms > self.completed_at_ms
        ):
            raise MassiveAuthenticatedObjectGetError(
                "authenticated object GET metadata differs"
            )
        for name in (
            "listing_acquisition_receipt_sha256",
            "listing_entry_receipt_sha256",
            "acquisition_spec_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.acquisition_spec_sha256
            != MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256
        ):
            raise MassiveAuthenticatedObjectGetError(
                "authenticated object GET specification drifted"
            )
        self.loaded_source.validate()
        receipt = self.loaded_source.receipt
        if (
            receipt.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID
            or receipt.source_object_key != self.source_object_key
            or receipt.schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256
            or receipt.requested_at_ms != self.requested_at_ms
            or receipt.downloaded_at_ms != self.completed_at_ms
            or receipt.request_id != self.provider_request_id
            or receipt.etag != self.etag
            or receipt.content_length != self.content_length
        ):
            raise MassiveAuthenticatedObjectGetError(
                "authenticated object GET differs from committed payload"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAuthenticatedObjectGetError(
                "authenticated object GET receipt differs"
            )


def download_massive_flat_file_object_v1(
    *,
    s3_client: Any,
    captured_listing: MassiveCapturedFlatFileListingV0,
    source_object_key: str,
    destination_root: str | Path,
    entitlement_receipt_sha256: str,
    now_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
) -> MassiveAuthenticatedFlatFileDownloadV1:
    """Perform the fixed authenticated GET used by readiness qualification."""

    captured_listing.validate()
    entry = captured_listing.committed_listing.resolve(
        source_object_key=source_object_key
    )
    requested_at_ms = now_ms()
    try:
        response = s3_client.get_object(
            Bucket=MASSIVE_FLAT_FILE_BUCKET,
            Key=source_object_key,
        )
    except Exception as exc:  # pragma: no cover - SDK exception classes are optional
        raise MassiveAuthenticatedObjectGetError("provider object GET failed") from exc
    if not isinstance(response, Mapping):
        raise MassiveAuthenticatedObjectGetError(
            "provider object response is malformed"
        )
    metadata = response.get("ResponseMetadata")
    request_id = metadata.get("RequestId") if isinstance(metadata, Mapping) else None
    body = response.get("Body")
    etag = response.get("ETag")
    length = response.get("ContentLength")
    version_id = response.get("VersionId")
    if (
        not isinstance(request_id, str)
        or not request_id
        or body is None
        or not hasattr(body, "read")
        or not isinstance(etag, str)
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or (version_id is not None and not isinstance(version_id, str))
    ):
        raise MassiveAuthenticatedObjectGetError(
            "provider object response metadata is incomplete"
        )
    clean_etag = etag.removeprefix('"').removesuffix('"')
    if clean_etag != entry.etag or length != entry.content_length:
        raise MassiveAuthenticatedObjectGetError(
            "provider object differs from captured listing"
        )
    Path(destination_root).mkdir(parents=True, exist_ok=True)
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as spool:
        shutil.copyfileobj(body, spool, length=1024 * 1024)
        completed_at_ms = now_ms()
        spool.seek(0)
        publish_massive_source_object(
            stream=spool,
            root=destination_root,
            relative_payload_path=source_object_key,
            dataset_id=MASSIVE_FLAT_TRADES_DATASET_ID,
            source_object_key=source_object_key,
            requested_at_ms=requested_at_ms,
            downloaded_at_ms=completed_at_ms,
            schema_sha256=MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
            entitlement_receipt_sha256=entitlement_receipt_sha256,
            committed_at_ms=completed_at_ms,
            etag=clean_etag,
            request_id=request_id,
        )
    loaded = load_massive_source_bundle(
        root=destination_root,
        relative_payload_path=source_object_key,
        verified_at_ms=completed_at_ms,
    )
    body_values = {
        "schema": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SCHEMA,
        "endpoint_url": MASSIVE_FLAT_FILE_ENDPOINT,
        "bucket": MASSIVE_FLAT_FILE_BUCKET,
        "source_object_key": source_object_key,
        "listing_acquisition_receipt_sha256": captured_listing.acquisition_evidence.receipt_sha256,
        "listing_entry_receipt_sha256": entry.receipt_sha256,
        "provider_request_id": request_id,
        "provider_version_id": version_id,
        "requested_at_ms": requested_at_ms,
        "completed_at_ms": completed_at_ms,
        "etag": clean_etag,
        "content_length": length,
        "loaded_source": loaded,
        "acquisition_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
    }
    provisional = MassiveAuthenticatedFlatFileDownloadV1(
        **body_values,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveAuthenticatedFlatFileDownloadV1(
        **body_values,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256",
    "MassiveAuthenticatedFlatFileDownloadV1",
    "MassiveAuthenticatedObjectGetError",
    "download_massive_flat_file_object_v1",
]
