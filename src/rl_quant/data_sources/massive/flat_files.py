"""Streaming transport boundary for large Massive flat-file objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from rl_quant.data_sources.massive.source_receipts import (
    MassiveSourceCommit,
    MassiveSourceObjectReceipt,
    publish_massive_source_object,
)


@dataclass(frozen=True, slots=True)
class MassiveFlatFileMetadata:
    dataset_id: str
    source_object_key: str
    requested_at_ms: int
    downloaded_at_ms: int
    schema_sha256: str
    entitlement_receipt_sha256: str
    committed_at_ms: int
    etag: str | None = None
    request_id: str | None = None
    expected_physical_sha256: str | None = None


def mirror_massive_flat_file_stream(
    *,
    stream: BinaryIO,
    destination_root: str | Path,
    relative_payload_path: str,
    metadata: MassiveFlatFileMetadata,
) -> tuple[MassiveSourceObjectReceipt, MassiveSourceCommit]:
    """Mirror an authenticated stream without loading it into memory.

    Authentication and retry belong to the calling transport. This function
    owns immutable byte publication and never accepts an API credential.
    """

    return publish_massive_source_object(
        stream=stream,
        root=destination_root,
        relative_payload_path=relative_payload_path,
        dataset_id=metadata.dataset_id,
        source_object_key=metadata.source_object_key,
        requested_at_ms=metadata.requested_at_ms,
        downloaded_at_ms=metadata.downloaded_at_ms,
        schema_sha256=metadata.schema_sha256,
        entitlement_receipt_sha256=metadata.entitlement_receipt_sha256,
        committed_at_ms=metadata.committed_at_ms,
        etag=metadata.etag,
        request_id=metadata.request_id,
        expected_physical_sha256=metadata.expected_physical_sha256,
    )


__all__ = ["MassiveFlatFileMetadata", "mirror_massive_flat_file_stream"]
