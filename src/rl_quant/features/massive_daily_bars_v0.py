"""Typed daily bar artifacts for finalized Massive validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Sequence

from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    load_massive_persisted_security_rows_v2,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_DAILY_BARS_V0_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "share_volume",
    "dollar_volume",
    "high_low_range",
    "close_location",
)
MASSIVE_DAILY_BARS_V0_SCHEMA = "rl-quant.massive-daily-bars-v0"
MASSIVE_DAILY_BARS_V0_DATASET = "massive-finalized-daily-bars-v0"
MASSIVE_DAILY_BARS_V0_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_DAILY_BARS_V0_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "input": "terminal-active-regular-session-events",
        "event_order": "participant-sip-sequence-source-row",
        "fields": MASSIVE_DAILY_BARS_V0_FIELDS,
        "open_close_domain": "condition-authority-updates-open-close",
        "high_low_domain": "condition-authority-updates-high-low",
        "volume_domain": "condition-authority-updates-volume",
        "open": "first-open-close-eligible-price",
        "high": "maximum-high-low-eligible-price",
        "low": "minimum-high-low-eligible-price",
        "close": "last-open-close-eligible-price",
        "share_volume": "sum-volume-eligible-decimal-size",
        "dollar_volume": "sum-volume-eligible-price-times-size",
        "high_low_range": "(high-low)/close",
        "close_location": "(close-low)/(high-low);0.5-when-flat",
    }
)
MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_DAILY_BARS_V0_SCHEMA,
        "fields": MASSIVE_DAILY_BARS_V0_FIELDS,
        "value_type": "finite-float64",
        "mask_type": "boolean",
    }
)


class MassiveDailyBarsV0Error(ValueError):
    """Daily bar bytes or semantics differ from the frozen V0 contract."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveDailyBarsV0Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveDailyBarsRowV0:
    security_id: str
    values: tuple[float, ...]
    valid: tuple[bool, ...]
    source_active_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.security_id or self.security_id != self.security_id.strip():
            raise MassiveDailyBarsV0Error("bar security identity is invalid")
        if (
            len(self.values) != len(MASSIVE_DAILY_BARS_V0_FIELDS)
            or len(self.valid) != len(self.values)
            or any(not isinstance(value, bool) for value in self.valid)
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in self.values
            )
        ):
            raise MassiveDailyBarsV0Error("bar values or masks are malformed")
        _digest("bar source inventory", self.source_active_inventory_sha256)
        _digest("bar row receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveDailyBarsV0Error("bar row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveDailyBarsArtifactV0:
    source_session_date: str
    persisted_partition_manifest_receipt_sha256: str
    condition_authority_receipt_sha256: str
    feature_spec_receipt_sha256: str
    feature_source_sha256: str
    rows: tuple[MassiveDailyBarsRowV0, ...]
    row_inventory_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_DAILY_BARS_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_DAILY_BARS_V0_SCHEMA:
            raise MassiveDailyBarsV0Error("daily bars schema drifted")
        for name in (
            "persisted_partition_manifest_receipt_sha256",
            "condition_authority_receipt_sha256",
            "feature_spec_receipt_sha256",
            "feature_source_sha256",
            "row_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.feature_spec_receipt_sha256 != MASSIVE_DAILY_BARS_V0_SPEC_SHA256
            or self.feature_source_sha256 != MASSIVE_DAILY_BARS_V0_SOURCE_SHA256
        ):
            raise MassiveDailyBarsV0Error("daily bars implementation drifted")
        security_ids = tuple(row.security_id for row in self.rows)
        if not security_ids or security_ids != tuple(sorted(set(security_ids))):
            raise MassiveDailyBarsV0Error("daily bar rows are not canonical")
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveDailyBarsV0Error("daily bar inventory differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_DAILY_BARS_V0_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveDailyBarsV0Error("daily bars source contract differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveDailyBarsV0Error("daily bars artifact receipt differs")


def _row(
    security_id: str,
    active_rows: Sequence[object],
    condition_authority: MassiveConditionAuthority,
) -> MassiveDailyBarsRowV0:
    rows = tuple(active_rows)
    if not rows:
        raise MassiveDailyBarsV0Error("daily bars require active regular trades")
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.canonical_record.participant_timestamp_ns,
                row.canonical_record.sip_timestamp_ns,
                row.canonical_record.sequence_number,
                row.source_row_number,
            ),
        )
    )
    eligibility = tuple(
        condition_authority.resolve(row.canonical_record.conditions) for row in ordered
    )
    open_close_rows = tuple(
        row for row, flags in zip(ordered, eligibility, strict=True) if flags[0]
    )
    high_low_rows = tuple(
        row for row, flags in zip(ordered, eligibility, strict=True) if flags[1]
    )
    volume_rows = tuple(
        row for row, flags in zip(ordered, eligibility, strict=True) if flags[2]
    )
    open_close_prices = tuple(
        Decimal(row.canonical_record.price_decimal) for row in open_close_rows
    )
    high_low_prices = tuple(
        Decimal(row.canonical_record.price_decimal) for row in high_low_rows
    )
    volume_prices = tuple(
        Decimal(row.canonical_record.price_decimal) for row in volume_rows
    )
    volume_sizes = tuple(
        Decimal(row.canonical_record.size_decimal) for row in volume_rows
    )
    opening = open_close_prices[0] if open_close_prices else Decimal(0)
    closing = open_close_prices[-1] if open_close_prices else Decimal(0)
    high = max(high_low_prices) if high_low_prices else Decimal(0)
    low = min(high_low_prices) if high_low_prices else Decimal(0)
    shares = sum(volume_sizes, Decimal(0))
    dollars = sum(
        (
            price * size
            for price, size in zip(volume_prices, volume_sizes, strict=True)
        ),
        Decimal(0),
    )
    combined_price_valid = bool(open_close_prices and high_low_prices)
    high_low_range = (
        Decimal(0)
        if not combined_price_valid or closing == 0
        else (high - low) / closing
    )
    close_location = (
        Decimal(0)
        if not combined_price_valid
        else Decimal("0.5")
        if high == low
        else (closing - low) / (high - low)
    )
    values = tuple(
        float(value)
        for value in (
            opening,
            high,
            low,
            closing,
            shares,
            dollars,
            high_low_range,
            close_location,
        )
    )
    body = {
        "security_id": security_id,
        "values": values,
        "valid": (
            bool(open_close_prices),
            bool(high_low_prices),
            bool(high_low_prices),
            bool(open_close_prices),
            bool(volume_rows),
            bool(volume_rows),
            combined_price_valid and closing != 0,
            combined_price_valid,
        ),
        "source_active_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
    }
    result = MassiveDailyBarsRowV0(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _payload(
    *,
    source_session_date: str,
    persisted_receipt: str,
    condition_authority_receipt: str,
    rows: tuple[MassiveDailyBarsRowV0, ...],
) -> dict[str, object]:
    return {
        "schema": MASSIVE_DAILY_BARS_V0_SCHEMA,
        "source_session_date": source_session_date,
        "persisted_partition_manifest_receipt_sha256": persisted_receipt,
        "condition_authority_receipt_sha256": condition_authority_receipt,
        "feature_spec_receipt_sha256": MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
        "feature_source_sha256": MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
        "feature_names": MASSIVE_DAILY_BARS_V0_FIELDS,
        "rows": tuple(asdict(row) for row in rows),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
    }


def materialize_massive_daily_bars_v0(
    *,
    persisted_root: str | Path,
    output_root: str | Path,
    manifest: MassivePersistedPartitionManifestV1,
    condition_authority: MassiveConditionAuthority,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveDailyBarsArtifactV0:
    manifest.validate()
    condition_authority.validate()
    rows = []
    for partition in manifest.partitions:
        _, active, _ = load_massive_persisted_security_rows_v2(
            root=persisted_root, partition=partition
        )
        if active:
            rows.append(_row(partition.security_id, active, condition_authority))
    ordered = tuple(sorted(rows, key=lambda row: row.security_id))
    if not ordered:
        raise MassiveDailyBarsV0Error("daily bars contain no valid securities")
    payload = _payload(
        source_session_date=manifest.source_session_date,
        persisted_receipt=manifest.receipt_sha256,
        condition_authority_receipt=condition_authority.receipt_sha256,
        rows=ordered,
    )
    relative = (
        f"massive-finalized-v0/session={manifest.source_session_date}/daily-bars.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_DAILY_BARS_V0_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=output_root, relative_payload_path=relative, verified_at_ms=published_at_ms
    )
    body = {
        "source_session_date": manifest.source_session_date,
        "persisted_partition_manifest_receipt_sha256": manifest.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "feature_spec_receipt_sha256": MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
        "feature_source_sha256": MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
        "rows": ordered,
        "row_inventory_sha256": payload["row_inventory_sha256"],
        "loaded_source": loaded,
        "schema": MASSIVE_DAILY_BARS_V0_SCHEMA,
    }
    provisional = MassiveDailyBarsArtifactV0(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveDailyBarsArtifactV0(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    validate_massive_daily_bars_v0(root=output_root, artifact=result)
    return result


def validate_massive_daily_bars_v0(
    *, root: str | Path, artifact: MassiveDailyBarsArtifactV0
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveDailyBarsV0Error("daily bars source is not JSON") from exc
    expected = _payload(
        source_session_date=artifact.source_session_date,
        persisted_receipt=artifact.persisted_partition_manifest_receipt_sha256,
        condition_authority_receipt=artifact.condition_authority_receipt_sha256,
        rows=artifact.rows,
    )
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        expected
    ):
        raise MassiveDailyBarsV0Error("daily bars bytes differ")


__all__ = [
    "MASSIVE_DAILY_BARS_V0_DATASET",
    "MASSIVE_DAILY_BARS_V0_FIELDS",
    "MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256",
    "MASSIVE_DAILY_BARS_V0_SPEC_SHA256",
    "MassiveDailyBarsArtifactV0",
    "MassiveDailyBarsRowV0",
    "MassiveDailyBarsV0Error",
    "materialize_massive_daily_bars_v0",
    "validate_massive_daily_bars_v0",
]
