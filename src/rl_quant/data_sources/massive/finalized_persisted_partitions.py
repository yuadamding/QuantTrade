"""Committed, reloadable finalized-trade partitions for validation V0.

This is a new evidence generation.  The V0 semantic partition manifest remains
immutable and useful for development, while this module publishes the rows that
feature materializers actually consume.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
    scan_massive_daily_trade_file_v0,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MASSIVE_DAILY_TRADE_SECURITY_PARTITION_V0_SCHEMA,
    MassiveDailyTradePartitionManifestV0,
    MassiveDailyTradeSecurityPartitionV0,
    MassiveFinalizedFeatureDomainSpecV0,
    build_massive_daily_trade_partition_manifest_from_security_partitions_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.data_sources.massive.trade_canonicalization import (
    MassiveCanonicalTradeSourceRecord,
)
from rl_quant.data_sources.massive.trade_extraction import MassiveExtractedTradeRow
from rl_quant.protocol.canonical_artifact import (
    canonical_json_payload,
    file_sha256,
    semantic_sha256,
)


MASSIVE_PERSISTED_PARTITION_V1_SCHEMA = (
    "rl-quant.massive-finalized-persisted-security-partition-v1"
)
MASSIVE_PERSISTED_PARTITION_MANIFEST_V1_SCHEMA = (
    "rl-quant.massive-finalized-persisted-partition-manifest-v1"
)
MASSIVE_PERSISTED_EVENTS_DATASET_V1 = "massive-finalized-trade-events-v1"
MASSIVE_PERSISTED_ACTIVE_DATASET_V1 = "massive-finalized-active-regular-v1"
MASSIVE_PERSISTED_CORRECTIONS_DATASET_V1 = "massive-finalized-corrections-v1"
MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256 = semantic_sha256(
    {
        "format": "canonical-jsonl",
        "record": "MassiveExtractedTradeRow",
        "newline": "LF",
        "decimal_authority": "canonical-decimal-text",
    }
)
MASSIVE_PERSISTED_CORRECTION_SCHEMA_SHA256 = semantic_sha256(
    {
        "format": "canonical-jsonl",
        "fields": (
            "source_row_number",
            "correction_kind",
            "event_key",
            "canonical_record_receipt_sha256",
        ),
    }
)
MASSIVE_PERSISTED_PARTITION_SPEC_SHA256 = semantic_sha256(
    {
        "input": "complete-whole-file-scan-rows",
        "identity": "indexed-PIT-ticker-interval-at-participant-time",
        "ordering": "sip-sequence-exchange-trf-trade-source-row",
        "outputs": (
            "complete-visible-event-timeline",
            "terminal-active-regular-session-events",
            "complete-correction-event-timeline",
        ),
        "publication": "create-only-source-transaction-per-output",
    }
)
MASSIVE_PERSISTED_PARTITION_SOURCE_SHA256 = file_sha256(Path(__file__))


class MassivePersistedPartitionError(ValueError):
    """Persisted partitions are incomplete or differ from committed bytes."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassivePersistedPartitionError(f"{name} must be a lowercase SHA-256")
    return value


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassivePersistedPartitionError(f"{name} must be nonnegative")
    return value


def _jsonl(rows: Sequence[object]) -> bytes:
    return b"".join(canonical_json_payload(row) + b"\n" for row in rows)


def _event_key(row: MassiveExtractedTradeRow) -> tuple[str, int, int, str]:
    record = row.canonical_record
    return (
        record.ticker,
        record.exchange_id,
        -1 if record.trf_id is None else record.trf_id,
        record.trade_id,
    )


def _order_key(row: MassiveExtractedTradeRow) -> tuple[object, ...]:
    record = row.canonical_record
    return (
        record.sip_timestamp_ns,
        record.sequence_number,
        record.exchange_id,
        -1 if record.trf_id is None else record.trf_id,
        record.trade_id,
        row.source_row_number,
    )


def _identity_index(
    authority: PITSecurityUniverseAuthority,
) -> dict[str, tuple[object, ...]]:
    result: dict[str, list[object]] = defaultdict(list)
    for row in authority.ticker_history:
        result[row.ticker].append(row)
    return {
        ticker: tuple(sorted(rows, key=lambda value: value.valid_from_ms))
        for ticker, rows in result.items()
    }


def _resolve_security(
    index: dict[str, tuple[object, ...]], record: MassiveCanonicalTradeSourceRecord
) -> str:
    observed_at_ms = record.participant_timestamp_ns // 1_000_000
    matches = tuple(
        row
        for row in index.get(record.ticker, ())
        if row.valid_from_ms <= observed_at_ms
        and (row.valid_to_ms is None or observed_at_ms < row.valid_to_ms)
        and row.available_at_ms <= observed_at_ms
    )
    if len(matches) != 1:
        raise MassivePersistedPartitionError(
            f"ticker {record.ticker} does not resolve uniquely"
        )
    return matches[0].security_id


class MassiveDiskTradeRowSpoolV1(AbstractContextManager["MassiveDiskTradeRowSpoolV1"]):
    """Bounded-memory SQLite spool keyed by PIT permanent security identity."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        identity_authority: PITSecurityUniverseAuthority,
    ) -> None:
        identity_authority.validate()
        self._identity_index = _identity_index(identity_authority)
        self._connection = sqlite3.connect(str(database_path))
        self._connection.execute(
            """
            CREATE TABLE trade_rows (
                security_id TEXT NOT NULL,
                source_row_number INTEGER NOT NULL PRIMARY KEY,
                sip_timestamp_ns INTEGER NOT NULL,
                sequence_number INTEGER NOT NULL,
                exchange_id INTEGER NOT NULL,
                trf_id INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                row_json TEXT NOT NULL
            )
            """
        )
        self._sealed = False

    def append(self, row: MassiveExtractedTradeRow) -> None:
        if self._sealed:
            raise MassivePersistedPartitionError("trade spool is already sealed")
        row.validate()
        record = row.canonical_record
        security_id = _resolve_security(self._identity_index, record)
        self._connection.execute(
            "INSERT INTO trade_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                security_id,
                row.source_row_number,
                record.sip_timestamp_ns,
                record.sequence_number,
                record.exchange_id,
                -1 if record.trf_id is None else record.trf_id,
                record.trade_id,
                canonical_json_payload(asdict(row)).decode("ascii"),
            ),
        )

    def seal(self) -> None:
        if not self._sealed:
            self._connection.commit()
            self._connection.execute(
                "CREATE INDEX trade_rows_security_order ON trade_rows "
                "(security_id, sip_timestamp_ns, sequence_number, exchange_id, "
                "trf_id, trade_id, source_row_number)"
            )
            self._connection.commit()
            self._sealed = True

    def security_ids(self) -> tuple[str, ...]:
        self.seal()
        return tuple(
            row[0]
            for row in self._connection.execute(
                "SELECT DISTINCT security_id FROM trade_rows ORDER BY security_id"
            )
        )

    def rows_for_security(
        self, security_id: str
    ) -> tuple[MassiveExtractedTradeRow, ...]:
        self.seal()
        payloads = self._connection.execute(
            "SELECT row_json FROM trade_rows WHERE security_id = ? "
            "ORDER BY sip_timestamp_ns, sequence_number, exchange_id, trf_id, "
            "trade_id, source_row_number",
            (security_id,),
        )
        result: list[MassiveExtractedTradeRow] = []
        for (payload,) in payloads:
            value = json.loads(payload)
            canonical_payload = value["canonical_record"]
            canonical_payload["conditions"] = tuple(canonical_payload["conditions"])
            canonical = MassiveCanonicalTradeSourceRecord(**canonical_payload)
            row = MassiveExtractedTradeRow(
                source_row_number=value["source_row_number"],
                raw_row_sha256=value["raw_row_sha256"],
                canonical_record=canonical,
                receipt_sha256=value["receipt_sha256"],
                schema=value["schema"],
            )
            row.validate()
            result.append(row)
        if not result:
            raise MassivePersistedPartitionError("security spool partition is empty")
        return tuple(result)

    def close(self) -> None:
        self._connection.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _publish(
    *,
    root: str | Path,
    relative_path: str,
    dataset_id: str,
    schema_sha256: str,
    payload: bytes,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> LoadedMassiveSourceObject:
    Path(root).mkdir(parents=True, exist_ok=True)
    publish_massive_source_object(
        stream=BytesIO(payload),
        root=root,
        relative_payload_path=relative_path,
        dataset_id=dataset_id,
        source_object_key=relative_path,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=schema_sha256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    return load_massive_source_bundle(
        root=root,
        relative_payload_path=relative_path,
        verified_at_ms=published_at_ms,
    )


@dataclass(frozen=True, slots=True)
class MassivePersistedSecurityPartitionV1:
    security_id: str
    event_timeline: LoadedMassiveSourceObject
    active_regular: LoadedMassiveSourceObject
    correction_timeline: LoadedMassiveSourceObject
    event_row_count: int
    active_regular_row_count: int
    correction_event_count: int
    event_inventory_sha256: str
    active_inventory_sha256: str
    correction_inventory_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_PERSISTED_PARTITION_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PERSISTED_PARTITION_V1_SCHEMA or not self.security_id:
            raise MassivePersistedPartitionError("persisted security identity drifted")
        expected = (
            (
                self.event_timeline,
                MASSIVE_PERSISTED_EVENTS_DATASET_V1,
                MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
            ),
            (
                self.active_regular,
                MASSIVE_PERSISTED_ACTIVE_DATASET_V1,
                MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
            ),
            (
                self.correction_timeline,
                MASSIVE_PERSISTED_CORRECTIONS_DATASET_V1,
                MASSIVE_PERSISTED_CORRECTION_SCHEMA_SHA256,
            ),
        )
        for loaded, dataset, schema in expected:
            loaded.validate()
            if (
                loaded.receipt.dataset_id != dataset
                or loaded.receipt.schema_sha256 != schema
            ):
                raise MassivePersistedPartitionError(
                    "persisted partition dataset/schema differs"
                )
        for name in (
            "event_row_count",
            "active_regular_row_count",
            "correction_event_count",
        ):
            _count(name, getattr(self, name))
        if (
            self.event_row_count <= 0
            or self.active_regular_row_count > self.event_row_count
        ):
            raise MassivePersistedPartitionError(
                "persisted partition row counts differ"
            )
        for name in (
            "event_inventory_sha256",
            "active_inventory_sha256",
            "correction_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassivePersistedPartitionError("persisted security receipt differs")


@dataclass(frozen=True, slots=True)
class MassivePersistedPartitionManifestV1:
    source_session_date: str
    source_file_scan_receipt_sha256: str
    semantic_partition_manifest_receipt_sha256: str
    identity_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    partition_spec_sha256: str
    partition_source_sha256: str
    partitions: tuple[MassivePersistedSecurityPartitionV1, ...]
    source_row_count: int
    persisted_event_row_count: int
    active_event_key_count: int
    correction_event_count: int
    security_partition_count: int
    partition_inventory_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_PERSISTED_PARTITION_MANIFEST_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PERSISTED_PARTITION_MANIFEST_V1_SCHEMA:
            raise MassivePersistedPartitionError("persisted manifest schema drifted")
        for name in (
            "source_file_scan_receipt_sha256",
            "semantic_partition_manifest_receipt_sha256",
            "identity_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "partition_spec_sha256",
            "partition_source_sha256",
            "partition_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.partition_spec_sha256 != MASSIVE_PERSISTED_PARTITION_SPEC_SHA256
            or self.partition_source_sha256 != MASSIVE_PERSISTED_PARTITION_SOURCE_SHA256
        ):
            raise MassivePersistedPartitionError(
                "persisted implementation identity drifted"
            )
        for name in (
            "source_row_count",
            "persisted_event_row_count",
            "active_event_key_count",
            "correction_event_count",
            "security_partition_count",
        ):
            _count(name, getattr(self, name))
        keys = tuple(row.security_id for row in self.partitions)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassivePersistedPartitionError(
                "persisted partitions are not canonical"
            )
        for row in self.partitions:
            row.validate()
        if (
            self.security_partition_count != len(self.partitions)
            or self.persisted_event_row_count
            != sum(row.event_row_count for row in self.partitions)
            or self.active_event_key_count
            != sum(row.active_regular_row_count for row in self.partitions)
            or self.correction_event_count
            != sum(row.correction_event_count for row in self.partitions)
            or self.persisted_event_row_count != self.source_row_count
        ):
            raise MassivePersistedPartitionError(
                "persisted global counts do not reconcile"
            )
        if self.partition_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.partitions)
        ):
            raise MassivePersistedPartitionError("persisted inventory differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassivePersistedPartitionError("persisted manifest receipt differs")


def _publish_security_partition(
    *,
    root: str | Path,
    security_id: str,
    values: Sequence[MassiveExtractedTradeRow],
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    correction_authority: MassiveCorrectionAuthority,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> tuple[
    MassivePersistedSecurityPartitionV1,
    MassiveDailyTradeSecurityPartitionV0,
]:
    ordered = tuple(sorted(values, key=_order_key))
    if not ordered:
        raise MassivePersistedPartitionError("security partition is empty")
    active: dict[tuple[str, int, int, str], MassiveExtractedTradeRow] = {}
    cancelled: set[tuple[str, int, int, str]] = set()
    corrections: list[dict[str, object]] = []
    for row in ordered:
        code = (
            0
            if row.canonical_record.correction_code is None
            else row.canonical_record.correction_code
        )
        kind = correction_authority.resolve(code)
        key = _event_key(row)
        if kind in {"new-trade", "late-report"}:
            existing = active.get(key)
            if existing is not None and existing.receipt_sha256 != row.receipt_sha256:
                raise MassivePersistedPartitionError("conflicting finalized duplicate")
            active[key] = row
            cancelled.discard(key)
        elif kind == "replacement":
            if key not in active:
                raise MassivePersistedPartitionError("replacement lacks predecessor")
            active[key] = row
            cancelled.discard(key)
        elif kind == "cancellation":
            if key not in active:
                raise MassivePersistedPartitionError("cancellation lacks predecessor")
            del active[key]
            cancelled.add(key)
        if kind in {"replacement", "cancellation", "late-report"}:
            corrections.append(
                {
                    "source_row_number": row.source_row_number,
                    "correction_kind": kind,
                    "event_key": key,
                    "canonical_record_receipt_sha256": (
                        row.canonical_record.receipt_sha256
                    ),
                }
            )
    regular_inputs = tuple(
        row
        for row in ordered
        if scan_evidence.regular_open_ns
        <= row.canonical_record.participant_timestamp_ns
        < scan_evidence.regular_close_ns
    )
    active_regular = tuple(
        sorted(
            (
                row
                for row in active.values()
                if scan_evidence.regular_open_ns
                <= row.canonical_record.participant_timestamp_ns
                < scan_evidence.regular_close_ns
            ),
            key=_event_key,
        )
    )
    premarket = sum(
        row.canonical_record.participant_timestamp_ns < scan_evidence.regular_open_ns
        for row in ordered
    )
    after_hours = sum(
        row.canonical_record.participant_timestamp_ns >= scan_evidence.regular_close_ns
        for row in ordered
    )
    safe_security = security_id.replace("/", "_")
    prefix = (
        f"massive-finalized-v1/session={scan_evidence.source_session_date}/"
        f"security={safe_security}"
    )
    event_payload_rows = tuple(asdict(row) for row in ordered)
    active_payload_rows = tuple(asdict(row) for row in active_regular)
    correction_payload_rows = tuple(corrections)
    events = _publish(
        root=root,
        relative_path=f"{prefix}/events.jsonl",
        dataset_id=MASSIVE_PERSISTED_EVENTS_DATASET_V1,
        schema_sha256=MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
        payload=_jsonl(event_payload_rows),
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    active_loaded = _publish(
        root=root,
        relative_path=f"{prefix}/active_regular.jsonl",
        dataset_id=MASSIVE_PERSISTED_ACTIVE_DATASET_V1,
        schema_sha256=MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
        payload=_jsonl(active_payload_rows),
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    corrections_loaded = _publish(
        root=root,
        relative_path=f"{prefix}/corrections.jsonl",
        dataset_id=MASSIVE_PERSISTED_CORRECTIONS_DATASET_V1,
        schema_sha256=MASSIVE_PERSISTED_CORRECTION_SCHEMA_SHA256,
        payload=_jsonl(correction_payload_rows),
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    body: dict[str, object] = {
        "schema": MASSIVE_PERSISTED_PARTITION_V1_SCHEMA,
        "security_id": security_id,
        "event_timeline": events,
        "active_regular": active_loaded,
        "correction_timeline": corrections_loaded,
        "event_row_count": len(ordered),
        "active_regular_row_count": len(active_regular),
        "correction_event_count": len(corrections),
        "event_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
        "active_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in active_regular)
        ),
        "correction_inventory_sha256": semantic_sha256(correction_payload_rows),
    }
    provisional = MassivePersistedSecurityPartitionV1(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    persisted = MassivePersistedSecurityPartitionV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    persisted.validate()

    semantic_body: dict[str, object] = {
        "schema": MASSIVE_DAILY_TRADE_SECURITY_PARTITION_V0_SCHEMA,
        "security_id": security_id,
        "source_tickers": tuple(
            sorted({row.canonical_record.ticker for row in ordered})
        ),
        "source_row_count": len(ordered),
        "premarket_row_count": premarket,
        "regular_session_input_row_count": len(regular_inputs),
        "after_hours_row_count": after_hours,
        "active_regular_session_row_count": len(active_regular),
        "cancelled_event_count": len(cancelled),
        "all_row_inventory_sha256": semantic_sha256(
            tuple(
                (
                    row.source_row_number,
                    row.raw_row_sha256,
                    row.canonical_record.receipt_sha256,
                )
                for row in sorted(ordered, key=lambda value: value.source_row_number)
            )
        ),
        "active_regular_row_inventory_sha256": semantic_sha256(
            tuple(row.canonical_record.receipt_sha256 for row in active_regular)
        ),
    }
    semantic = MassiveDailyTradeSecurityPartitionV0(
        **semantic_body,  # type: ignore[arg-type]
        partition_receipt_sha256=semantic_sha256(semantic_body),
    )
    semantic.validate()
    return persisted, semantic


def persist_massive_daily_trade_partitions_v1(
    *,
    root: str | Path,
    rows: Sequence[MassiveExtractedTradeRow],
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    semantic_partition_manifest: MassiveDailyTradePartitionManifestV0,
    identity_authority: PITSecurityUniverseAuthority,
    correction_authority: MassiveCorrectionAuthority,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassivePersistedPartitionManifestV1:
    """Publish exact feature-facing event, active, and correction artifacts."""

    scan_evidence.validate()
    semantic_partition_manifest.validate()
    identity_authority.validate()
    correction_authority.validate()
    source_rows = tuple(sorted(rows, key=lambda row: row.source_row_number))
    if (
        len(source_rows) != scan_evidence.source_row_count
        or semantic_partition_manifest.source_file_scan_receipt_sha256
        != scan_evidence.receipt_sha256
        or semantic_partition_manifest.identity_authority_receipt_sha256
        != identity_authority.receipt_sha256
        or semantic_partition_manifest.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
    ):
        raise MassivePersistedPartitionError("persisted partition inputs differ")
    index = _identity_index(identity_authority)
    grouped: dict[str, list[MassiveExtractedTradeRow]] = defaultdict(list)
    for row in source_rows:
        row.validate()
        grouped[_resolve_security(index, row.canonical_record)].append(row)

    partitions: list[MassivePersistedSecurityPartitionV1] = []
    base = f"massive-finalized-v1/session={scan_evidence.source_session_date}"
    for security_id, values in sorted(grouped.items()):
        ordered = tuple(sorted(values, key=_order_key))
        active: dict[tuple[str, int, int, str], MassiveExtractedTradeRow] = {}
        corrections: list[dict[str, object]] = []
        for row in ordered:
            code = (
                0
                if row.canonical_record.correction_code is None
                else row.canonical_record.correction_code
            )
            kind = correction_authority.resolve(code)
            key = _event_key(row)
            if kind in {"new-trade", "late-report"}:
                existing = active.get(key)
                if (
                    existing is not None
                    and existing.receipt_sha256 != row.receipt_sha256
                ):
                    raise MassivePersistedPartitionError(
                        "conflicting finalized duplicate"
                    )
                active[key] = row
            elif kind == "replacement":
                if key not in active:
                    raise MassivePersistedPartitionError(
                        "replacement lacks predecessor"
                    )
                active[key] = row
            elif kind == "cancellation":
                if key not in active:
                    raise MassivePersistedPartitionError(
                        "cancellation lacks predecessor"
                    )
                del active[key]
            if kind in {"replacement", "cancellation", "late-report"}:
                corrections.append(
                    {
                        "source_row_number": row.source_row_number,
                        "correction_kind": kind,
                        "event_key": key,
                        "canonical_record_receipt_sha256": row.canonical_record.receipt_sha256,
                    }
                )
        active_regular = tuple(
            sorted(
                (
                    row
                    for row in active.values()
                    if scan_evidence.regular_open_ns
                    <= row.canonical_record.participant_timestamp_ns
                    < scan_evidence.regular_close_ns
                ),
                key=_event_key,
            )
        )
        safe_security = security_id.replace("/", "_")
        prefix = f"{base}/security={safe_security}"
        event_payload_rows = tuple(asdict(row) for row in ordered)
        active_payload_rows = tuple(asdict(row) for row in active_regular)
        correction_payload_rows = tuple(corrections)
        events = _publish(
            root=root,
            relative_path=f"{prefix}/events.jsonl",
            dataset_id=MASSIVE_PERSISTED_EVENTS_DATASET_V1,
            schema_sha256=MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
            payload=_jsonl(event_payload_rows),
            entitlement_receipt_sha256=entitlement_receipt_sha256,
            published_at_ms=published_at_ms,
        )
        active_loaded = _publish(
            root=root,
            relative_path=f"{prefix}/active_regular.jsonl",
            dataset_id=MASSIVE_PERSISTED_ACTIVE_DATASET_V1,
            schema_sha256=MASSIVE_PERSISTED_JSONL_SCHEMA_SHA256,
            payload=_jsonl(active_payload_rows),
            entitlement_receipt_sha256=entitlement_receipt_sha256,
            published_at_ms=published_at_ms,
        )
        corrections_loaded = _publish(
            root=root,
            relative_path=f"{prefix}/corrections.jsonl",
            dataset_id=MASSIVE_PERSISTED_CORRECTIONS_DATASET_V1,
            schema_sha256=MASSIVE_PERSISTED_CORRECTION_SCHEMA_SHA256,
            payload=_jsonl(correction_payload_rows),
            entitlement_receipt_sha256=entitlement_receipt_sha256,
            published_at_ms=published_at_ms,
        )
        body: dict[str, object] = {
            "schema": MASSIVE_PERSISTED_PARTITION_V1_SCHEMA,
            "security_id": security_id,
            "event_timeline": events,
            "active_regular": active_loaded,
            "correction_timeline": corrections_loaded,
            "event_row_count": len(ordered),
            "active_regular_row_count": len(active_regular),
            "correction_event_count": len(corrections),
            "event_inventory_sha256": semantic_sha256(
                tuple(row.receipt_sha256 for row in ordered)
            ),
            "active_inventory_sha256": semantic_sha256(
                tuple(row.receipt_sha256 for row in active_regular)
            ),
            "correction_inventory_sha256": semantic_sha256(correction_payload_rows),
        }
        provisional = MassivePersistedSecurityPartitionV1(
            **body,
            receipt_sha256="0" * 64,  # type: ignore[arg-type]
        )
        partition = MassivePersistedSecurityPartitionV1(
            **body,
            receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
        )
        partition.validate()
        partitions.append(partition)

    partition_rows = tuple(partitions)
    body = {
        "schema": MASSIVE_PERSISTED_PARTITION_MANIFEST_V1_SCHEMA,
        "source_session_date": scan_evidence.source_session_date,
        "source_file_scan_receipt_sha256": scan_evidence.receipt_sha256,
        "semantic_partition_manifest_receipt_sha256": semantic_partition_manifest.receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "partition_spec_sha256": MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
        "partition_source_sha256": MASSIVE_PERSISTED_PARTITION_SOURCE_SHA256,
        "partitions": partition_rows,
        "source_row_count": len(source_rows),
        "persisted_event_row_count": sum(row.event_row_count for row in partition_rows),
        "active_event_key_count": sum(
            row.active_regular_row_count for row in partition_rows
        ),
        "correction_event_count": sum(
            row.correction_event_count for row in partition_rows
        ),
        "security_partition_count": len(partition_rows),
        "partition_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in partition_rows)
        ),
    }
    provisional_manifest = MassivePersistedPartitionManifestV1(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassivePersistedPartitionManifestV1(
        **body,
        receipt_sha256=semantic_sha256(provisional_manifest.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    return result


def stream_and_persist_massive_daily_trade_partitions_v1(
    *,
    source_root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    spool_root: str | Path,
    persisted_root: str | Path,
    session_authority: MassiveSessionAuthority,
    session: MassiveExchangeSession,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> tuple[
    MassiveDailyTradeFileScanEvidenceV0,
    MassiveDailyTradePartitionManifestV0,
    MassivePersistedPartitionManifestV1,
]:
    """Scan once into a disk spool, then publish one security at a time."""

    identity_authority.validate()
    condition_authority.validate()
    correction_authority.validate()
    feature_domain_spec.validate()
    if (
        feature_domain_spec.condition_authority_receipt_sha256
        != condition_authority.receipt_sha256
        or feature_domain_spec.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
    ):
        raise MassivePersistedPartitionError(
            "bounded partition feature authorities differ"
        )
    Path(spool_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".massive-finalized-spool-", dir=spool_root
    ) as temporary:
        with MassiveDiskTradeRowSpoolV1(
            database_path=Path(temporary) / "trade-rows.sqlite3",
            identity_authority=identity_authority,
        ) as spool:
            retained, scan = scan_massive_daily_trade_file_v0(
                root=source_root,
                loaded_source=loaded_source,
                session_authority=session_authority,
                session=session,
                correction_authority=correction_authority,
                row_sink=spool.append,
                retain_rows=False,
            )
            if retained:
                raise MassivePersistedPartitionError(
                    "bounded scan unexpectedly retained source rows"
                )
            persisted_rows: list[MassivePersistedSecurityPartitionV1] = []
            semantic_rows: list[MassiveDailyTradeSecurityPartitionV0] = []
            for security_id in spool.security_ids():
                security_rows = spool.rows_for_security(security_id)
                for row in security_rows:
                    condition_authority.resolve(row.canonical_record.conditions)
                persisted, semantic = _publish_security_partition(
                    root=persisted_root,
                    security_id=security_id,
                    values=security_rows,
                    scan_evidence=scan,
                    correction_authority=correction_authority,
                    entitlement_receipt_sha256=entitlement_receipt_sha256,
                    published_at_ms=published_at_ms,
                )
                persisted_rows.append(persisted)
                semantic_rows.append(semantic)

    semantic_manifest = (
        build_massive_daily_trade_partition_manifest_from_security_partitions_v0(
            scan_evidence=scan,
            identity_authority=identity_authority,
            condition_authority=condition_authority,
            correction_authority=correction_authority,
            feature_domain_spec=feature_domain_spec,
            security_partitions=semantic_rows,
        )
    )

    partition_rows = tuple(persisted_rows)
    persisted_body: dict[str, object] = {
        "schema": MASSIVE_PERSISTED_PARTITION_MANIFEST_V1_SCHEMA,
        "source_session_date": scan.source_session_date,
        "source_file_scan_receipt_sha256": scan.receipt_sha256,
        "semantic_partition_manifest_receipt_sha256": semantic_manifest.receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "partition_spec_sha256": MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
        "partition_source_sha256": MASSIVE_PERSISTED_PARTITION_SOURCE_SHA256,
        "partitions": partition_rows,
        "source_row_count": scan.source_row_count,
        "persisted_event_row_count": sum(row.event_row_count for row in partition_rows),
        "active_event_key_count": sum(
            row.active_regular_row_count for row in partition_rows
        ),
        "correction_event_count": sum(
            row.correction_event_count for row in partition_rows
        ),
        "security_partition_count": len(partition_rows),
        "partition_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in partition_rows)
        ),
    }
    persisted_provisional = MassivePersistedPartitionManifestV1(
        **persisted_body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    persisted_manifest = MassivePersistedPartitionManifestV1(
        **persisted_body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(persisted_provisional.unsigned()),
    )
    persisted_manifest.validate()
    return scan, semantic_manifest, persisted_manifest


def validate_massive_persisted_partitions_v1(
    *, root: str | Path, manifest: MassivePersistedPartitionManifestV1
) -> None:
    """Reopen every committed row file and verify its semantic inventory."""

    manifest.validate()
    for partition in manifest.partitions:
        event_lines = tuple(
            json.loads(line)
            for line in read_loaded_massive_source_bytes(
                root=root, loaded_source=partition.event_timeline
            ).splitlines()
            if line
        )
        active_lines = tuple(
            json.loads(line)
            for line in read_loaded_massive_source_bytes(
                root=root, loaded_source=partition.active_regular
            ).splitlines()
            if line
        )
        correction_lines = tuple(
            json.loads(line)
            for line in read_loaded_massive_source_bytes(
                root=root, loaded_source=partition.correction_timeline
            ).splitlines()
            if line
        )
        if (
            len(event_lines) != partition.event_row_count
            or len(active_lines) != partition.active_regular_row_count
            or len(correction_lines) != partition.correction_event_count
            or semantic_sha256(tuple(row["receipt_sha256"] for row in event_lines))
            != partition.event_inventory_sha256
            or semantic_sha256(tuple(row["receipt_sha256"] for row in active_lines))
            != partition.active_inventory_sha256
            or semantic_sha256(correction_lines)
            != partition.correction_inventory_sha256
        ):
            raise MassivePersistedPartitionError("persisted partition bytes differ")


def _parse_extracted_trade_row_v2(value: object) -> MassiveExtractedTradeRow:
    """Reconstruct one extracted row rather than trusting nested receipt text."""

    if not isinstance(value, dict):
        raise MassivePersistedPartitionError("persisted event row is not an object")
    expected = {
        "schema",
        "source_row_number",
        "raw_row_sha256",
        "canonical_record",
        "receipt_sha256",
    }
    if set(value) != expected or not isinstance(value["canonical_record"], dict):
        raise MassivePersistedPartitionError("persisted event field inventory drifted")
    raw_record = dict(value["canonical_record"])
    raw_record["conditions"] = tuple(raw_record.get("conditions", ()))
    try:
        canonical = MassiveCanonicalTradeSourceRecord(**raw_record)
        row = MassiveExtractedTradeRow(
            schema=value["schema"],
            source_row_number=value["source_row_number"],
            raw_row_sha256=value["raw_row_sha256"],
            canonical_record=canonical,
            receipt_sha256=value["receipt_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MassivePersistedPartitionError(
            "persisted event values are malformed"
        ) from exc
    row.validate()
    return row


def _parse_canonical_jsonl_v2(raw: bytes) -> tuple[object, ...]:
    if raw and not raw.endswith(b"\n"):
        raise MassivePersistedPartitionError("persisted JSONL lacks a final newline")
    try:
        rows = tuple(json.loads(line) for line in raw.splitlines())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassivePersistedPartitionError("persisted JSONL is malformed") from exc
    if raw != _jsonl(rows):
        raise MassivePersistedPartitionError("persisted JSONL is not canonical")
    return rows


def _parse_correction_row_v2(value: object) -> dict[str, object]:
    expected = {
        "source_row_number",
        "correction_kind",
        "event_key",
        "canonical_record_receipt_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or not isinstance(value["event_key"], list)
        or len(value["event_key"]) != 4
    ):
        raise MassivePersistedPartitionError(
            "persisted correction field inventory drifted"
        )
    result = dict(value)
    result["event_key"] = tuple(value["event_key"])
    return result


def validate_massive_persisted_partitions_semantically_v2(
    *,
    root: str | Path,
    manifest: MassivePersistedPartitionManifestV1,
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    semantic_partition_manifest: MassiveDailyTradePartitionManifestV0,
    identity_authority: PITSecurityUniverseAuthority,
    correction_authority: MassiveCorrectionAuthority,
) -> None:
    """Reparse event bytes and independently rederive every cached partition.

    V1's physical validator remains immutable.  This V2 qualification treats
    the event timeline as the sole authority and regards the active/correction
    files as deterministic caches.
    """

    manifest.validate()
    scan_evidence.validate()
    semantic_partition_manifest.validate()
    identity_authority.validate()
    correction_authority.validate()
    if (
        manifest.source_file_scan_receipt_sha256 != scan_evidence.receipt_sha256
        or manifest.semantic_partition_manifest_receipt_sha256
        != semantic_partition_manifest.receipt_sha256
        or manifest.identity_authority_receipt_sha256
        != identity_authority.receipt_sha256
        or manifest.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
    ):
        raise MassivePersistedPartitionError(
            "semantic persisted-partition authorities differ"
        )
    semantic_by_security = {
        row.security_id: row for row in semantic_partition_manifest.security_partitions
    }
    if set(semantic_by_security) != {row.security_id for row in manifest.partitions}:
        raise MassivePersistedPartitionError(
            "semantic and persisted security inventories differ"
        )
    identity_index = _identity_index(identity_authority)
    global_source_rows: set[int] = set()
    for partition in manifest.partitions:
        event_raw = read_loaded_massive_source_bytes(
            root=root, loaded_source=partition.event_timeline
        )
        active_raw = read_loaded_massive_source_bytes(
            root=root, loaded_source=partition.active_regular
        )
        correction_raw = read_loaded_massive_source_bytes(
            root=root, loaded_source=partition.correction_timeline
        )
        event_values = _parse_canonical_jsonl_v2(event_raw)
        active_values = _parse_canonical_jsonl_v2(active_raw)
        correction_values = tuple(
            _parse_correction_row_v2(value)
            for value in _parse_canonical_jsonl_v2(correction_raw)
        )
        events = tuple(_parse_extracted_trade_row_v2(value) for value in event_values)
        active_rows = tuple(
            _parse_extracted_trade_row_v2(value) for value in active_values
        )
        if events != tuple(sorted(events, key=_order_key)):
            raise MassivePersistedPartitionError("persisted event order differs")
        source_rows = tuple(row.source_row_number for row in events)
        if len(set(source_rows)) != len(source_rows) or global_source_rows.intersection(
            source_rows
        ):
            raise MassivePersistedPartitionError(
                "persisted source-row provenance is not unique"
            )
        global_source_rows.update(source_rows)
        for row in events:
            if (
                _resolve_security(identity_index, row.canonical_record)
                != partition.security_id
            ):
                raise MassivePersistedPartitionError(
                    "persisted event is routed to the wrong security"
                )

        active: dict[tuple[str, int, int, str], MassiveExtractedTradeRow] = {}
        cancelled: set[tuple[str, int, int, str]] = set()
        expected_corrections: list[dict[str, object]] = []
        for row in events:
            record = row.canonical_record
            code = 0 if record.correction_code is None else record.correction_code
            kind = correction_authority.resolve(code)
            key = _event_key(row)
            if kind in {"new-trade", "late-report"}:
                previous = active.get(key)
                if (
                    previous is not None
                    and previous.receipt_sha256 != row.receipt_sha256
                ):
                    raise MassivePersistedPartitionError(
                        "persisted event timeline has a conflicting duplicate"
                    )
                active[key] = row
                cancelled.discard(key)
            elif kind == "replacement":
                if key not in active:
                    raise MassivePersistedPartitionError(
                        "persisted replacement lacks a predecessor"
                    )
                active[key] = row
                cancelled.discard(key)
            elif kind == "cancellation":
                if key not in active:
                    raise MassivePersistedPartitionError(
                        "persisted cancellation lacks a predecessor"
                    )
                del active[key]
                cancelled.add(key)
            if kind in {"replacement", "cancellation", "late-report"}:
                expected_corrections.append(
                    {
                        "source_row_number": row.source_row_number,
                        "correction_kind": kind,
                        "event_key": key,
                        "canonical_record_receipt_sha256": record.receipt_sha256,
                    }
                )
        expected_active = tuple(
            sorted(
                (
                    row
                    for row in active.values()
                    if scan_evidence.regular_open_ns
                    <= row.canonical_record.participant_timestamp_ns
                    < scan_evidence.regular_close_ns
                ),
                key=_event_key,
            )
        )
        expected_correction_rows = tuple(expected_corrections)
        if (
            active_rows != expected_active
            or correction_values != expected_correction_rows
            or event_raw != _jsonl(tuple(asdict(row) for row in events))
            or active_raw != _jsonl(tuple(asdict(row) for row in expected_active))
            or correction_raw != _jsonl(expected_correction_rows)
        ):
            raise MassivePersistedPartitionError(
                "persisted derived caches differ from the event timeline"
            )
        semantic = semantic_by_security[partition.security_id]
        regular_input_count = sum(
            scan_evidence.regular_open_ns
            <= row.canonical_record.participant_timestamp_ns
            < scan_evidence.regular_close_ns
            for row in events
        )
        if (
            partition.event_row_count != len(events)
            or partition.active_regular_row_count != len(expected_active)
            or partition.correction_event_count != len(expected_correction_rows)
            or partition.event_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in events))
            or partition.active_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in expected_active))
            or partition.correction_inventory_sha256
            != semantic_sha256(expected_correction_rows)
            or semantic.source_row_count != len(events)
            or semantic.regular_session_input_row_count != regular_input_count
            or semantic.active_regular_session_row_count != len(expected_active)
            or semantic.cancelled_event_count != len(cancelled)
        ):
            raise MassivePersistedPartitionError(
                "persisted rows do not reconcile to the semantic partition"
            )
    if len(global_source_rows) != scan_evidence.source_row_count:
        raise MassivePersistedPartitionError(
            "persisted global source-row inventory is incomplete"
        )


def load_massive_persisted_security_rows_v2(
    *,
    root: str | Path,
    partition: MassivePersistedSecurityPartitionV1,
) -> tuple[
    tuple[MassiveExtractedTradeRow, ...],
    tuple[MassiveExtractedTradeRow, ...],
    tuple[dict[str, object], ...],
]:
    """Reload the three canonical security artifacts with nested validation."""

    partition.validate()
    event_values = _parse_canonical_jsonl_v2(
        read_loaded_massive_source_bytes(
            root=root, loaded_source=partition.event_timeline
        )
    )
    active_values = _parse_canonical_jsonl_v2(
        read_loaded_massive_source_bytes(
            root=root, loaded_source=partition.active_regular
        )
    )
    correction_values = tuple(
        _parse_correction_row_v2(value)
        for value in _parse_canonical_jsonl_v2(
            read_loaded_massive_source_bytes(
                root=root, loaded_source=partition.correction_timeline
            )
        )
    )
    return (
        tuple(_parse_extracted_trade_row_v2(value) for value in event_values),
        tuple(_parse_extracted_trade_row_v2(value) for value in active_values),
        correction_values,
    )


__all__ = [
    "MASSIVE_PERSISTED_ACTIVE_DATASET_V1",
    "MASSIVE_PERSISTED_CORRECTIONS_DATASET_V1",
    "MASSIVE_PERSISTED_EVENTS_DATASET_V1",
    "MASSIVE_PERSISTED_PARTITION_MANIFEST_V1_SCHEMA",
    "MASSIVE_PERSISTED_PARTITION_SPEC_SHA256",
    "MASSIVE_PERSISTED_PARTITION_V1_SCHEMA",
    "MassivePersistedPartitionError",
    "MassivePersistedPartitionManifestV1",
    "MassivePersistedSecurityPartitionV1",
    "MassiveDiskTradeRowSpoolV1",
    "persist_massive_daily_trade_partitions_v1",
    "load_massive_persisted_security_rows_v2",
    "stream_and_persist_massive_daily_trade_partitions_v1",
    "validate_massive_persisted_partitions_v1",
    "validate_massive_persisted_partitions_semantically_v2",
]
