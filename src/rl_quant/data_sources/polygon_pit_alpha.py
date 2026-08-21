"""Guarded conversion of the organized Polygon cache into PIT-alpha staging data.

This module intentionally stops short of issuing :class:`PITAlphaDatasetAuthority`.
The existing organized cache is future-selected and does not contain enough
terminal-event, membership, cash-return, or independent reconciliation evidence
to become reportable by transformation alone.  It can still supply audited raw
inputs, identity observations, corporate-action candidates, and ordered
five-minute bars for development and reconciliation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePath
from typing import Any
from zoneinfo import ZoneInfo

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)

PIT_ALPHA_CONVERSION_AUDIT_SCHEMA = "rl-quant.pit-alpha-conversion-readiness-v2"
PIT_ALPHA_FIVE_MINUTE_STAGING_SCHEMA = "rl-quant.pit-alpha-five-minute-staging-v2"
PIT_ALPHA_STAGING_COMMIT_SCHEMA = "rl-quant.pit-alpha-staging-commit-v1"
POLYGON_SYMBOL_DAY_SOURCE_SCHEMA = "rl-quant.polygon-symbol-day-source-v1"
POLYGON_SESSION_AUTHORITY_SCHEMA = "rl-quant.polygon-exchange-session-v1"
EASTERN = ZoneInfo("America/New_York")
BAR_SECONDS = 300
EXPECTED_NORMAL_SESSION_INTERVALS = 78
_SAFE_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,15}$")
_FILE_STATUSES = frozenset({"downloaded", "exists"})
_EXPECTED_SOURCE_COLUMNS = frozenset(
    {"symbol", "timestamp_ms", "open", "high", "low", "close", "volume"}
)
_SOURCE_VALUE_COLUMNS = (
    "symbol",
    "timestamp_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "adjusted",
    "timespan",
    "multiplier",
)
_OUTPUT_SCHEMA = (
    ("symbol", "utf8", False),
    ("session_date", "utf8", False),
    ("exchange", "utf8", False),
    ("interval_index", "int16", False),
    ("interval_start_ms", "int64", False),
    ("economic_interval_end_ms", "int64", False),
    ("assumed_strategy_available_at_ms", "int64", False),
    ("open", "float64", False),
    ("high", "float64", False),
    ("low", "float64", False),
    ("close", "float64", False),
    ("volume", "float64", False),
    ("vwap", "float64", True),
    ("transactions", "int64", False),
    ("source_row_count", "int64", False),
    ("observed", "bool", False),
)
_PARQUET_COMPRESSION = "zstd"
_PARQUET_COMPRESSION_LEVEL = 9
_PARQUET_ROW_GROUP_SIZE = EXPECTED_NORMAL_SESSION_INTERVALS


class PolygonPITAlphaConversionError(ValueError):
    """The organized Polygon cache cannot satisfy a conversion invariant."""


def _canonical_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PolygonPITAlphaConversionError(f"{name} must be a canonical string")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolygonPITAlphaConversionError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolygonPITAlphaConversionError(f"{name} must be a nonnegative integer")
    return value


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise PolygonPITAlphaConversionError(f"{name} must be Boolean")
    return value


def _optional_manifest_int(name: str, value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise PolygonPITAlphaConversionError(
            f"manifest {name} must be an integer"
        ) from exc
    return _nonnegative_int(f"manifest {name}", parsed)


@dataclass(frozen=True, slots=True)
class OrganizedPolygonShard:
    """One organized bars/covariates/universe shard."""

    name: str
    second_aggs_root: Path
    covariates_root: Path
    universe_tickers_file: Path
    universe_asof: str

    @property
    def manifest_csv(self) -> Path:
        return self.second_aggs_root / "manifest.csv"

    @property
    def dataset_manifest_json(self) -> Path:
        return self.second_aggs_root / "dataset_manifest.json"

    def validate(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise PolygonPITAlphaConversionError("shard name must be canonical")
        try:
            date.fromisoformat(self.universe_asof)
        except ValueError as exc:
            raise PolygonPITAlphaConversionError(
                "universe_asof must be an ISO date"
            ) from exc
        for path in (
            self.second_aggs_root,
            self.covariates_root,
            self.universe_tickers_file,
            self.manifest_csv,
            self.dataset_manifest_json,
        ):
            if not path.exists():
                raise PolygonPITAlphaConversionError(
                    f"organized source is missing: {path}"
                )


@dataclass(frozen=True, slots=True)
class OrganizedPolygonShardAudit:
    name: str
    dataset_start: str
    dataset_end_exclusive: str
    universe_asof: str
    expected_symbol_count: int
    manifest_row_count: int
    valid_manifest_symbol_count: int
    malformed_manifest_rows: int
    unexpected_manifest_symbols: tuple[str, ...]
    missing_manifest_symbols: tuple[str, ...]
    status_counts: tuple[tuple[str, int], ...]
    blank_sha256_rows: int
    duplicate_manifest_rows: int
    stale_output_path_rows: int
    canonical_missing_file_rows: int | None
    canonical_nonregular_file_rows: int | None
    canonical_hash_mismatch_rows: int | None
    canonical_size_mismatch_rows: int | None
    canonical_row_count_mismatch_rows: int | None
    canonical_schema_invalid_rows: int | None
    unexpected_file_for_empty_rows: int | None
    observed_source_hash_rows: int | None
    manifest_hash_verified_rows: int | None
    manifest_declared_remaining_symbol_days: int
    manifest_derived_incomplete_symbol_days: int
    covariate_symbol_count: int
    unexpected_covariate_symbols: tuple[str, ...]
    missing_covariate_symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolygonManifestRow:
    shard_id: str
    manifest_file_sha256: str
    line_number: int
    symbol: str
    session_date: str
    status: str
    recorded_output_path: str
    manifest_rows: int | None
    manifest_size_bytes: int | None
    manifest_sha256: str | None


@dataclass(frozen=True, slots=True)
class PolygonSymbolDaySourceAuthority:
    """Manifest-aware identity for one immutable organized symbol-day."""

    schema: str
    shard_id: str
    symbol: str
    session_date: str
    canonical_path: str
    manifest_path: str
    manifest_file_sha256: str
    manifest_line_number: int
    manifest_status: str
    manifest_rows: int | None
    manifest_size_bytes: int | None
    manifest_sha256: str | None
    observed_row_count: int
    observed_size_bytes: int
    observed_sha256: str
    parquet_schema_sha256: str
    manifest_hash_verified: bool
    manifest_size_verified: bool | None
    manifest_rows_verified: bool | None
    qualifies_for_staging: bool
    source_receipt_sha256: str

    def unsigned(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "source_receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != POLYGON_SYMBOL_DAY_SOURCE_SCHEMA:
            raise PolygonPITAlphaConversionError("source authority schema drifted")
        _canonical_text("shard ID", self.shard_id)
        if _SAFE_SYMBOL.fullmatch(self.symbol) is None:
            raise PolygonPITAlphaConversionError("source authority symbol is unsafe")
        try:
            date.fromisoformat(self.session_date)
        except ValueError as exc:
            raise PolygonPITAlphaConversionError(
                "source authority session date is invalid"
            ) from exc
        source = Path(self.canonical_path)
        if not source.is_absolute() or ".." in PurePath(source).parts:
            raise PolygonPITAlphaConversionError(
                "source authority path must be normalized and absolute"
            )
        manifest = Path(self.manifest_path)
        if not manifest.is_absolute() or ".." in PurePath(manifest).parts:
            raise PolygonPITAlphaConversionError(
                "source manifest path must be normalized and absolute"
            )
        _digest("manifest file SHA", self.manifest_file_sha256)
        _nonnegative_int("manifest line number", self.manifest_line_number)
        if self.manifest_line_number == 0 or self.manifest_status not in _FILE_STATUSES:
            raise PolygonPITAlphaConversionError(
                "source authority lacks an accepted manifest row"
            )
        for name in ("observed_row_count", "observed_size_bytes"):
            if _nonnegative_int(name, getattr(self, name)) == 0:
                raise PolygonPITAlphaConversionError(f"{name} must be positive")
        _digest("observed source SHA", self.observed_sha256)
        _digest("Parquet schema SHA", self.parquet_schema_sha256)
        if self.manifest_sha256 is not None:
            _digest("manifest SHA", self.manifest_sha256)
            if not self.manifest_hash_verified:
                raise PolygonPITAlphaConversionError(
                    "declared manifest SHA was not verified"
                )
        elif self.manifest_hash_verified:
            raise PolygonPITAlphaConversionError(
                "blank manifest SHA cannot be marked verified"
            )
        for declared, verified, label in (
            (self.manifest_size_bytes, self.manifest_size_verified, "size"),
            (self.manifest_rows, self.manifest_rows_verified, "row count"),
        ):
            if declared is not None:
                _nonnegative_int(f"manifest {label}", declared)
            if declared is None and verified is not None:
                raise PolygonPITAlphaConversionError(
                    f"missing manifest {label} cannot have verification state"
                )
            if declared is not None and verified is not True:
                raise PolygonPITAlphaConversionError(
                    f"declared manifest {label} was not verified"
                )
        _bool("manifest hash verified", self.manifest_hash_verified)
        _bool("qualifies for staging", self.qualifies_for_staging)
        if not self.qualifies_for_staging:
            raise PolygonPITAlphaConversionError(
                "source authority does not qualify for staging"
            )
        if self.source_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise PolygonPITAlphaConversionError("source authority receipt drifted")


@dataclass(frozen=True, slots=True)
class PolygonExchangeSessionAuthority:
    """One byte-bound exchange session supplied by an external calendar authority."""

    schema: str
    session_date: str
    exchange: str
    open_at_ms: int
    close_at_ms: int
    scheduled_interval_count: int
    special_session_reason: str | None
    assumed_availability_lag_ms: int
    calendar_source_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != POLYGON_SESSION_AUTHORITY_SCHEMA:
            raise PolygonPITAlphaConversionError("session authority schema drifted")
        try:
            date.fromisoformat(self.session_date)
        except ValueError as exc:
            raise PolygonPITAlphaConversionError(
                "session authority date is invalid"
            ) from exc
        _canonical_text("exchange", self.exchange)
        opened = _nonnegative_int("session open", self.open_at_ms)
        closed = _nonnegative_int("session close", self.close_at_ms)
        if closed <= opened or (closed - opened) % (BAR_SECONDS * 1_000) != 0:
            raise PolygonPITAlphaConversionError(
                "session duration must be a positive whole number of five-minute bins"
            )
        expected = (closed - opened) // (BAR_SECONDS * 1_000)
        _nonnegative_int("scheduled interval count", self.scheduled_interval_count)
        if self.scheduled_interval_count != expected:
            raise PolygonPITAlphaConversionError(
                "scheduled interval count differs from session duration"
            )
        _nonnegative_int("assumed availability lag", self.assumed_availability_lag_ms)
        if self.special_session_reason is not None:
            _canonical_text("special session reason", self.special_session_reason)
        _digest("calendar source receipt", self.calendar_source_receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise PolygonPITAlphaConversionError("session authority receipt drifted")


@dataclass(frozen=True, slots=True)
class PolygonIdentityObservation:
    source_symbol: str
    asof_date: str
    observed_ticker: str
    cik: str | None
    composite_figi: str | None
    share_class_figi: str | None
    primary_exchange: str | None
    security_type: str | None
    source_file_sha256: str
    source_line_number: int


@dataclass(frozen=True, slots=True)
class PolygonCorporateActionCandidate:
    source_symbol: str
    source_dataset: str
    vendor_event_id: str
    candidate_kind: str
    effective_date: str
    announced_date: str | None
    declaration_date: str | None
    ex_date: str | None
    record_date: str | None
    payment_date: str | None
    execution_date: str | None
    causal_availability_known: bool
    cash_per_share: float | None
    share_ratio: float | None
    source_file_sha256: str
    source_line_number: int
    source_record_sha256: str


@dataclass(frozen=True, slots=True)
class PolygonPITAlphaConversionAudit:
    schema: str
    shards: tuple[OrganizedPolygonShardAudit, ...]
    expected_universe_symbol_count: int
    observed_covariate_symbol_count: int
    overlapping_covariate_symbols: tuple[str, ...]
    unexpected_covariate_symbols: tuple[str, ...]
    source_symbol_identity_transition_count: int
    source_symbol_identity_transitions: tuple[str, ...]
    cross_ticker_share_class_figi_count: int
    cross_ticker_share_class_figis: tuple[str, ...]
    identity_observation_count: int
    dividend_candidate_count: int
    split_candidate_count: int
    corporate_actions_missing_announcement_count: int
    staging_conversion_possible: bool
    bar_source_inventory_verified: bool
    pit_alpha_training_ready: bool
    reportable_pit_authority_ready: bool
    blockers: tuple[str, ...]
    source_receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_organized_polygon_shards(
    data_root: Path,
) -> tuple[OrganizedPolygonShard, ...]:
    """Resolve the two existing organized TOP2000 source shards."""

    polygon = data_root / "polygon"
    universes = polygon / "universes"
    return (
        OrganizedPolygonShard(
            name="top500",
            second_aggs_root=(
                polygon
                / "second_aggs"
                / "top500_common_stocks_2022-01-01_to_2026-06-15"
            ),
            covariates_root=polygon / "stock_covariates" / "top500_2022_to_present",
            universe_tickers_file=(
                universes / "top_500_s3_volume_common_stocks_2026-06-12_tickers.txt"
            ),
            universe_asof="2026-06-12",
        ),
        OrganizedPolygonShard(
            name="top501_2000",
            second_aggs_root=(
                polygon
                / "second_aggs"
                / "top501_2000_common_stocks_2022-01-01_to_2026-06-15"
            ),
            covariates_root=(
                polygon / "stock_covariates" / "top501_2000_2022_to_present"
            ),
            universe_tickers_file=(
                universes
                / "top_501_2000_s3_volume_common_stocks_2026-06-12_tickers.txt"
            ),
            universe_asof="2026-06-12",
        ),
    )


def build_exchange_session_authority(
    *,
    session_date: str,
    exchange: str,
    open_at_ms: int,
    close_at_ms: int,
    calendar_source_receipt_sha256: str,
    special_session_reason: str | None = None,
    assumed_availability_lag_ms: int = 1_000,
) -> PolygonExchangeSessionAuthority:
    """Construct one validated exchange session from an external calendar row."""

    unsigned = {
        "schema": POLYGON_SESSION_AUTHORITY_SCHEMA,
        "session_date": session_date,
        "exchange": exchange,
        "open_at_ms": open_at_ms,
        "close_at_ms": close_at_ms,
        "scheduled_interval_count": (close_at_ms - open_at_ms) // (BAR_SECONDS * 1_000),
        "special_session_reason": special_session_reason,
        "assumed_availability_lag_ms": assumed_availability_lag_ms,
        "calendar_source_receipt_sha256": calendar_source_receipt_sha256,
    }
    authority = PolygonExchangeSessionAuthority(
        schema=POLYGON_SESSION_AUTHORITY_SCHEMA,
        session_date=session_date,
        exchange=exchange,
        open_at_ms=open_at_ms,
        close_at_ms=close_at_ms,
        scheduled_interval_count=(close_at_ms - open_at_ms) // (BAR_SECONDS * 1_000),
        special_session_reason=special_session_reason,
        assumed_availability_lag_ms=assumed_availability_lag_ms,
        calendar_source_receipt_sha256=calendar_source_receipt_sha256,
        receipt_sha256=semantic_sha256(unsigned),
    )
    authority.validate()
    return authority


def write_exchange_session_authority(
    path: Path, authority: PolygonExchangeSessionAuthority
) -> str:
    """Publish one canonical, create-only session authority."""

    authority.validate()
    return _write_new_bytes(path, canonical_json_file_bytes(asdict(authority)))


def load_exchange_session_authority(
    path: Path, *, expected_file_sha256: str
) -> PolygonExchangeSessionAuthority:
    """Load one exact canonical session authority."""

    raw = _read_regular_file(path, expected_sha256=expected_file_sha256)
    try:
        value = json.loads(raw)
        authority = PolygonExchangeSessionAuthority(**value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PolygonPITAlphaConversionError(
            "session authority file is malformed"
        ) from exc
    if raw != canonical_json_file_bytes(value):
        raise PolygonPITAlphaConversionError("session authority is not canonical")
    authority.validate()
    return authority


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise PolygonPITAlphaConversionError(f"expected a JSON object: {path}")
    return payload


def _read_regular_file(
    path: Path, *, expected_sha256: str | None = None, maximum_bytes: int | None = None
) -> bytes:
    """Read one stable regular non-symlink through a no-follow descriptor."""

    if expected_sha256 is not None:
        _digest("expected file SHA", expected_sha256)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise PolygonPITAlphaConversionError(
            f"source file is unavailable: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise PolygonPITAlphaConversionError(
                f"source must be a nonempty regular file: {path}"
            )
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise PolygonPITAlphaConversionError(f"source file is too large: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise PolygonPITAlphaConversionError(f"source changed while read: {path}")
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise PolygonPITAlphaConversionError(f"source file SHA drifted: {path}")
    return raw


def _stream_regular_file_identity(path: Path) -> tuple[int, str]:
    """Hash one stable regular non-symlink without retaining its bytes."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise PolygonPITAlphaConversionError(
            f"source file is unavailable: {path}"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise PolygonPITAlphaConversionError(
                f"source must be a nonempty regular file: {path}"
            )
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PolygonPITAlphaConversionError(f"source changed while hashed: {path}")
    return before.st_size, digest.hexdigest()


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise PolygonPITAlphaConversionError(
            "pyarrow is required for byte-authoritative Parquet staging"
        ) from exc
    return pa, pq


def _parquet_metadata(path: Path) -> tuple[int, str]:
    """Validate Parquet readability and return row count plus schema identity."""

    _, pq = _require_pyarrow()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise PolygonPITAlphaConversionError(
            f"Parquet source is unavailable: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise PolygonPITAlphaConversionError(
                f"Parquet source must be a nonempty regular file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            parquet = pq.ParquetFile(stream)
            row_count = int(parquet.metadata.num_rows)
            arrow_schema = parquet.schema_arrow
        after = os.fstat(descriptor)
    except Exception as exc:
        if isinstance(exc, PolygonPITAlphaConversionError):
            raise
        raise PolygonPITAlphaConversionError(
            f"Parquet source is unreadable: {path}"
        ) from exc
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PolygonPITAlphaConversionError(
            f"Parquet source changed during metadata read: {path}"
        )
    names = set(arrow_schema.names)
    if not _EXPECTED_SOURCE_COLUMNS.issubset(names) or row_count <= 0:
        raise PolygonPITAlphaConversionError(
            f"Parquet source schema or row count is invalid: {path}"
        )
    schema_body = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in arrow_schema
    ]
    return row_count, semantic_sha256(schema_body)


def _read_expected_symbols(path: Path) -> tuple[str, ...]:
    symbols = tuple(
        line.strip().upper() for line in path.read_text().splitlines() if line.strip()
    )
    if len(symbols) != len(set(symbols)):
        raise PolygonPITAlphaConversionError(
            f"universe ticker file contains duplicates: {path}"
        )
    if any(_SAFE_SYMBOL.fullmatch(symbol) is None for symbol in symbols):
        raise PolygonPITAlphaConversionError(
            f"universe ticker file contains unsafe symbols: {path}"
        )
    return symbols


def _canonical_symbol_day_path(root: Path, symbol: str, date_value: str) -> Path:
    try:
        parsed = date.fromisoformat(date_value)
    except ValueError as exc:
        raise PolygonPITAlphaConversionError(
            f"invalid manifest date {date_value!r}"
        ) from exc
    return (
        root
        / symbol
        / f"{parsed.year:04d}"
        / f"{parsed.month:02d}"
        / f"{date_value}.parquet"
    )


def _parse_manifest_row(
    shard: OrganizedPolygonShard,
    manifest_file_sha256: str,
    line_number: int,
    row: Mapping[str, Any],
) -> PolygonManifestRow:
    symbol = str(row.get("symbol") or "").strip().upper()
    session_date = str(row.get("date") or "").strip()
    if _SAFE_SYMBOL.fullmatch(symbol) is None:
        raise PolygonPITAlphaConversionError(
            f"manifest row has an unsafe symbol: {shard.manifest_csv}:{line_number}"
        )
    try:
        date.fromisoformat(session_date)
    except ValueError as exc:
        raise PolygonPITAlphaConversionError(
            f"manifest row has an invalid date: {shard.manifest_csv}:{line_number}"
        ) from exc
    manifest_sha = str(row.get("sha256") or "").strip().lower() or None
    if manifest_sha is not None:
        _digest("manifest SHA", manifest_sha)
    return PolygonManifestRow(
        shard_id=shard.name,
        manifest_file_sha256=manifest_file_sha256,
        line_number=line_number,
        symbol=symbol,
        session_date=session_date,
        status=str(row.get("status") or "").strip().lower(),
        recorded_output_path=str(row.get("output_path") or "").strip(),
        manifest_rows=_optional_manifest_int("rows", row.get("rows")),
        manifest_size_bytes=_optional_manifest_int(
            "output_size", row.get("output_size")
        ),
        manifest_sha256=manifest_sha,
    )


def _iter_manifest_rows(
    shard: OrganizedPolygonShard,
) -> Iterator[PolygonManifestRow]:
    raw = _read_regular_file(shard.manifest_csv, maximum_bytes=512 * 1024 * 1024)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PolygonPITAlphaConversionError(
            f"organized manifest is not UTF-8: {shard.manifest_csv}"
        ) from exc
    manifest_file_sha = hashlib.sha256(raw).hexdigest()
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {
        "symbol",
        "date",
        "status",
        "rows",
        "output_path",
        "output_size",
        "sha256",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise PolygonPITAlphaConversionError(
            f"organized manifest is missing required fields: {shard.manifest_csv}"
        )
    for line_number, row in enumerate(reader, start=2):
        yield _parse_manifest_row(shard, manifest_file_sha, line_number, row)


def _source_authority_from_row(
    shard: OrganizedPolygonShard, row: PolygonManifestRow
) -> PolygonSymbolDaySourceAuthority:
    if row.status not in _FILE_STATUSES:
        raise PolygonPITAlphaConversionError(
            f"manifest status {row.status!r} does not qualify for staging"
        )
    unresolved = _canonical_symbol_day_path(
        shard.second_aggs_root, row.symbol, row.session_date
    )
    try:
        metadata = unresolved.lstat()
    except FileNotFoundError as exc:
        raise PolygonPITAlphaConversionError("manifest source is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or unresolved.is_symlink():
        raise PolygonPITAlphaConversionError(
            "manifest source must be a regular non-symlink file"
        )
    canonical = unresolved.resolve(strict=True)
    observed_size, observed_sha = _stream_regular_file_identity(canonical)
    observed_rows, schema_sha = _parquet_metadata(canonical)
    hash_verified = (
        row.manifest_sha256 is not None and row.manifest_sha256 == observed_sha
    )
    size_verified = (
        None
        if row.manifest_size_bytes is None
        else row.manifest_size_bytes == observed_size
    )
    rows_verified = (
        None if row.manifest_rows is None else row.manifest_rows == observed_rows
    )
    if row.manifest_sha256 is not None and not hash_verified:
        raise PolygonPITAlphaConversionError("manifest SHA differs from source bytes")
    if size_verified is False:
        raise PolygonPITAlphaConversionError("manifest size differs from source bytes")
    if rows_verified is False:
        raise PolygonPITAlphaConversionError(
            "manifest rows differ from Parquet metadata"
        )
    unsigned = {
        "schema": POLYGON_SYMBOL_DAY_SOURCE_SCHEMA,
        "shard_id": row.shard_id,
        "manifest_file_sha256": row.manifest_file_sha256,
        "symbol": row.symbol,
        "session_date": row.session_date,
        "canonical_path": str(canonical),
        "manifest_path": str(shard.manifest_csv.resolve(strict=True)),
        "manifest_line_number": row.line_number,
        "manifest_status": row.status,
        "manifest_rows": row.manifest_rows,
        "manifest_size_bytes": row.manifest_size_bytes,
        "manifest_sha256": row.manifest_sha256,
        "observed_row_count": observed_rows,
        "observed_size_bytes": observed_size,
        "observed_sha256": observed_sha,
        "parquet_schema_sha256": schema_sha,
        "manifest_hash_verified": hash_verified,
        "manifest_size_verified": size_verified,
        "manifest_rows_verified": rows_verified,
        "qualifies_for_staging": True,
    }
    authority = PolygonSymbolDaySourceAuthority(
        schema=POLYGON_SYMBOL_DAY_SOURCE_SCHEMA,
        shard_id=row.shard_id,
        manifest_file_sha256=row.manifest_file_sha256,
        symbol=row.symbol,
        session_date=row.session_date,
        canonical_path=str(canonical),
        manifest_path=str(shard.manifest_csv.resolve(strict=True)),
        manifest_line_number=row.line_number,
        manifest_status=row.status,
        manifest_rows=row.manifest_rows,
        manifest_size_bytes=row.manifest_size_bytes,
        manifest_sha256=row.manifest_sha256,
        observed_row_count=observed_rows,
        observed_size_bytes=observed_size,
        observed_sha256=observed_sha,
        parquet_schema_sha256=schema_sha,
        manifest_hash_verified=hash_verified,
        manifest_size_verified=size_verified,
        manifest_rows_verified=rows_verified,
        qualifies_for_staging=True,
        source_receipt_sha256=semantic_sha256(unsigned),
    )
    authority.validate()
    return authority


def resolve_symbol_day_source(
    shards: Sequence[OrganizedPolygonShard], symbol: str, day: str
) -> PolygonSymbolDaySourceAuthority:
    """Resolve exactly one accepted manifest row and verify its current bytes."""

    normalized = symbol.strip().upper()
    if _SAFE_SYMBOL.fullmatch(normalized) is None:
        raise PolygonPITAlphaConversionError("symbol is unsafe")
    try:
        date.fromisoformat(day)
    except ValueError as exc:
        raise PolygonPITAlphaConversionError("session date is invalid") from exc
    matches: list[tuple[OrganizedPolygonShard, PolygonManifestRow]] = []
    for shard in shards:
        shard.validate()
        matches.extend(
            (shard, row)
            for row in _iter_manifest_rows(shard)
            if row.symbol == normalized and row.session_date == day
        )
    if len(matches) != 1:
        raise PolygonPITAlphaConversionError(
            f"expected one manifest source for {normalized} {day}, found {len(matches)}"
        )
    shard, row = matches[0]
    return _source_authority_from_row(shard, row)


def _audit_shard(
    shard: OrganizedPolygonShard,
    *,
    verify_canonical_files: bool,
) -> OrganizedPolygonShardAudit:
    shard.validate()
    expected = set(_read_expected_symbols(shard.universe_tickers_file))
    dataset = _read_json(shard.dataset_manifest_json)
    statuses: Counter[str] = Counter()
    manifest_symbols: set[str] = set()
    malformed = blank_sha = stale_paths = canonical_missing = 0
    duplicate_rows = nonregular = hash_mismatch = size_mismatch = 0
    row_count_mismatch = schema_invalid = unexpected_empty = 0
    observed_hashes = verified_hashes = 0
    manifest_rows = 0
    manifest_keys: set[tuple[str, str]] = set()

    with shard.manifest_csv.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "symbol",
            "date",
            "status",
            "rows",
            "output_path",
            "output_size",
            "sha256",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PolygonPITAlphaConversionError(
                f"organized manifest is missing required fields: {shard.manifest_csv}"
            )
        for line_number, row in enumerate(reader, start=2):
            manifest_rows += 1
            symbol = str(row.get("symbol") or "").strip().upper()
            date_value = str(row.get("date") or "").strip()
            status = str(row.get("status") or "").strip().lower()
            statuses[status] += 1
            manifest_sha = str(row.get("sha256") or "").strip().lower()
            if status in _FILE_STATUSES and not manifest_sha:
                blank_sha += 1
            if _SAFE_SYMBOL.fullmatch(symbol) is None:
                malformed += 1
                continue
            try:
                date.fromisoformat(date_value)
                declared_rows = _optional_manifest_int("rows", row.get("rows"))
                declared_size = _optional_manifest_int(
                    "output_size", row.get("output_size")
                )
                if manifest_sha:
                    _digest("manifest SHA", manifest_sha)
            except (PolygonPITAlphaConversionError, ValueError):
                malformed += 1
                continue
            key = (symbol, date_value)
            if key in manifest_keys:
                duplicate_rows += 1
            manifest_keys.add(key)
            manifest_symbols.add(symbol)
            canonical = _canonical_symbol_day_path(
                shard.second_aggs_root, symbol, date_value
            )
            recorded = Path(str(row.get("output_path") or ""))
            if tuple(recorded.parts[-5:]) != tuple(canonical.parts[-5:]):
                stale_paths += 1
            if not verify_canonical_files:
                continue
            if status == "empty":
                if canonical.exists() or canonical.is_symlink():
                    unexpected_empty += 1
                continue
            if status not in _FILE_STATUSES:
                continue
            try:
                metadata = canonical.lstat()
            except FileNotFoundError:
                canonical_missing += 1
                continue
            if not stat.S_ISREG(metadata.st_mode) or canonical.is_symlink():
                nonregular += 1
                continue
            try:
                observed_size, observed_sha = _stream_regular_file_identity(canonical)
            except PolygonPITAlphaConversionError:
                nonregular += 1
                continue
            observed_hashes += 1
            if manifest_sha:
                if observed_sha == manifest_sha:
                    verified_hashes += 1
                else:
                    hash_mismatch += 1
            if declared_size is not None and declared_size != observed_size:
                size_mismatch += 1
            try:
                observed_rows, _ = _parquet_metadata(canonical)
            except PolygonPITAlphaConversionError:
                schema_invalid += 1
                continue
            if declared_rows is not None and declared_rows != observed_rows:
                row_count_mismatch += 1

    covariate_symbols = {
        path.name.upper()
        for path in shard.covariates_root.iterdir()
        if path.is_dir()
        and path.name != "reference"
        and _SAFE_SYMBOL.fullmatch(path.name.upper())
    }
    completed = sum(statuses[status] for status in ("downloaded", "exists", "empty"))
    declared_symbols = int(dataset.get("symbols") or len(expected))
    declared_days = int(dataset.get("market_weekdays") or 0)
    derived_incomplete = max(declared_symbols * declared_days - completed, 0)
    return OrganizedPolygonShardAudit(
        name=shard.name,
        dataset_start=str(dataset.get("start") or ""),
        dataset_end_exclusive=str(dataset.get("end_exclusive") or ""),
        universe_asof=shard.universe_asof,
        expected_symbol_count=len(expected),
        manifest_row_count=manifest_rows,
        valid_manifest_symbol_count=len(manifest_symbols),
        malformed_manifest_rows=malformed,
        unexpected_manifest_symbols=tuple(sorted(manifest_symbols - expected)),
        missing_manifest_symbols=tuple(sorted(expected - manifest_symbols)),
        status_counts=tuple(sorted(statuses.items())),
        blank_sha256_rows=blank_sha,
        duplicate_manifest_rows=duplicate_rows,
        stale_output_path_rows=stale_paths,
        canonical_missing_file_rows=canonical_missing
        if verify_canonical_files
        else None,
        canonical_nonregular_file_rows=nonregular if verify_canonical_files else None,
        canonical_hash_mismatch_rows=hash_mismatch if verify_canonical_files else None,
        canonical_size_mismatch_rows=size_mismatch if verify_canonical_files else None,
        canonical_row_count_mismatch_rows=row_count_mismatch
        if verify_canonical_files
        else None,
        canonical_schema_invalid_rows=schema_invalid
        if verify_canonical_files
        else None,
        unexpected_file_for_empty_rows=unexpected_empty
        if verify_canonical_files
        else None,
        observed_source_hash_rows=observed_hashes if verify_canonical_files else None,
        manifest_hash_verified_rows=verified_hashes if verify_canonical_files else None,
        manifest_declared_remaining_symbol_days=int(
            dataset.get("remaining_symbol_days") or 0
        ),
        manifest_derived_incomplete_symbol_days=derived_incomplete,
        covariate_symbol_count=len(covariate_symbols),
        unexpected_covariate_symbols=tuple(sorted(covariate_symbols - expected)),
        missing_covariate_symbols=tuple(sorted(expected - covariate_symbols)),
    )


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise PolygonPITAlphaConversionError(
                    f"JSONL record must be an object: {path}:{line_number}"
                )
            yield line_number, payload


def iter_identity_observations(
    covariates_root: Path,
    *,
    symbols: Iterable[str] | None = None,
) -> Iterator[PolygonIdentityObservation]:
    """Yield source identity observations without asserting permanent-ID closure."""

    selected = None if symbols is None else {symbol.upper() for symbol in symbols}
    for symbol_dir in sorted(covariates_root.iterdir()):
        symbol = symbol_dir.name.upper()
        if (
            not symbol_dir.is_dir()
            or symbol == "REFERENCE"
            or _SAFE_SYMBOL.fullmatch(symbol) is None
            or (selected is not None and symbol not in selected)
        ):
            continue
        path = symbol_dir / "overview_snapshots.jsonl"
        if not path.is_file():
            continue
        source_digest = file_sha256(path)
        for line_number, payload in _iter_jsonl(path):
            asof = str(payload.get("asof_date") or "")
            ticker = str(payload.get("ticker") or symbol).upper()
            if not asof:
                raise PolygonPITAlphaConversionError(
                    f"overview observation has no asof_date: {path}:{line_number}"
                )
            yield PolygonIdentityObservation(
                source_symbol=symbol,
                asof_date=asof,
                observed_ticker=ticker,
                cik=str(payload["cik"]) if payload.get("cik") else None,
                composite_figi=(
                    str(payload["composite_figi"])
                    if payload.get("composite_figi")
                    else None
                ),
                share_class_figi=(
                    str(payload["share_class_figi"])
                    if payload.get("share_class_figi")
                    else None
                ),
                primary_exchange=(
                    str(payload["primary_exchange"])
                    if payload.get("primary_exchange")
                    else None
                ),
                security_type=str(payload["type"]) if payload.get("type") else None,
                source_file_sha256=source_digest,
                source_line_number=line_number,
            )


def _effective_date(source_dataset: str, payload: Mapping[str, Any]) -> str:
    keys = (
        ("ex_dividend_date", "pay_date", "record_date", "declaration_date")
        if source_dataset == "dividends"
        else ("execution_date", "split_date", "effective_date", "ex_date")
    )
    for key in keys:
        if payload.get(key):
            return str(payload[key])[:10]
    raise PolygonPITAlphaConversionError(
        f"{source_dataset} record has no effective date"
    )


def iter_corporate_action_candidates(
    covariates_root: Path,
    *,
    symbols: Iterable[str] | None = None,
) -> Iterator[PolygonCorporateActionCandidate]:
    """Yield dividend/split candidates while preserving missing causal evidence."""

    selected = None if symbols is None else {symbol.upper() for symbol in symbols}
    for symbol_dir in sorted(covariates_root.iterdir()):
        symbol = symbol_dir.name.upper()
        if (
            not symbol_dir.is_dir()
            or symbol == "REFERENCE"
            or _SAFE_SYMBOL.fullmatch(symbol) is None
            or (selected is not None and symbol not in selected)
        ):
            continue
        for dataset in ("dividends", "splits"):
            path = symbol_dir / f"{dataset}.jsonl"
            if not path.is_file():
                continue
            source_file_sha = file_sha256(path)
            for line_number, payload in _iter_jsonl(path):
                declaration_date = next(
                    (
                        str(payload[key])[:10]
                        for key in (
                            "declaration_date",
                            "announcement_date",
                            "declared_date",
                        )
                        if payload.get(key)
                    ),
                    None,
                )
                announced = next(
                    (
                        str(payload[key])[:10]
                        for key in (
                            "declaration_date",
                            "announcement_date",
                            "declared_date",
                        )
                        if payload.get(key)
                    ),
                    None,
                )
                vendor_id = str(
                    payload.get("id") or f"{symbol}:{dataset}:{line_number}"
                )
                if dataset == "dividends":
                    dividend_type = str(payload.get("dividend_type") or "").lower()
                    kind = (
                        "special-dividend"
                        if "special" in dividend_type
                        else "cash-dividend"
                    )
                    cash = (
                        float(payload["cash_amount"])
                        if payload.get("cash_amount") is not None
                        else None
                    )
                    ratio = None
                else:
                    split_from = float(payload.get("split_from") or 0.0)
                    split_to = float(payload.get("split_to") or 0.0)
                    ratio = (
                        split_to / split_from
                        if split_from > 0.0 and split_to > 0.0
                        else None
                    )
                    kind = (
                        "split"
                        if ratio is not None and ratio >= 1.0
                        else "reverse-split"
                    )
                    cash = None
                yield PolygonCorporateActionCandidate(
                    source_symbol=symbol,
                    source_dataset=dataset,
                    vendor_event_id=vendor_id,
                    candidate_kind=kind,
                    effective_date=_effective_date(dataset, payload),
                    announced_date=announced,
                    declaration_date=declaration_date,
                    ex_date=next(
                        (
                            str(payload[key])[:10]
                            for key in ("ex_dividend_date", "ex_date")
                            if payload.get(key)
                        ),
                        None,
                    ),
                    record_date=(
                        str(payload["record_date"])[:10]
                        if payload.get("record_date")
                        else None
                    ),
                    payment_date=(
                        str(payload["pay_date"])[:10]
                        if payload.get("pay_date")
                        else None
                    ),
                    execution_date=next(
                        (
                            str(payload[key])[:10]
                            for key in ("execution_date", "split_date")
                            if payload.get(key)
                        ),
                        None,
                    ),
                    causal_availability_known=announced is not None,
                    cash_per_share=cash,
                    share_ratio=ratio,
                    source_file_sha256=source_file_sha,
                    source_line_number=line_number,
                    source_record_sha256=semantic_sha256(payload),
                )


def audit_organized_polygon_for_pit_alpha(
    shards: Sequence[OrganizedPolygonShard],
    *,
    verify_canonical_files: bool = False,
) -> PolygonPITAlphaConversionAudit:
    """Audit organized inputs and explain why reportable issuance is blocked."""

    if not shards:
        raise PolygonPITAlphaConversionError("at least one organized shard is required")
    shard_audits = tuple(
        _audit_shard(shard, verify_canonical_files=verify_canonical_files)
        for shard in shards
    )
    expected_by_shard = [
        set(_read_expected_symbols(shard.universe_tickers_file)) for shard in shards
    ]
    expected_union = set().union(*expected_by_shard)
    if sum(len(symbols) for symbols in expected_by_shard) != len(expected_union):
        raise PolygonPITAlphaConversionError("organized universe shards overlap")

    covariate_sets = [
        {
            path.name.upper()
            for path in shard.covariates_root.iterdir()
            if path.is_dir()
            and path.name != "reference"
            and _SAFE_SYMBOL.fullmatch(path.name.upper())
        }
        for shard in shards
    ]
    symbol_frequency = Counter(
        symbol for symbols in covariate_sets for symbol in symbols
    )
    overlaps = tuple(
        sorted(symbol for symbol, count in symbol_frequency.items() if count > 1)
    )
    observed_union = set(symbol_frequency)

    observations_by_key: dict[tuple[str, str, int], PolygonIdentityObservation] = {}
    for shard in shards:
        for observation in iter_identity_observations(shard.covariates_root):
            observation_key = (
                observation.source_symbol,
                observation.source_file_sha256,
                observation.source_line_number,
            )
            observations_by_key[observation_key] = observation
    observations = tuple(observations_by_key.values())

    figis_by_symbol: defaultdict[str, set[str]] = defaultdict(set)
    symbols_by_figi: defaultdict[str, set[str]] = defaultdict(set)
    for observation in observations:
        if observation.share_class_figi:
            figis_by_symbol[observation.source_symbol].add(observation.share_class_figi)
            symbols_by_figi[observation.share_class_figi].add(observation.source_symbol)
    transitions = tuple(
        sorted(symbol for symbol, figis in figis_by_symbol.items() if len(figis) > 1)
    )
    cross_ticker_figis = tuple(
        sorted(figi for figi, symbols in symbols_by_figi.items() if len(symbols) > 1)
    )

    action_receipts: dict[tuple[str, str], str] = {}
    actions: list[PolygonCorporateActionCandidate] = []
    for shard in shards:
        for action in iter_corporate_action_candidates(shard.covariates_root):
            action_key = (action.source_dataset, action.vendor_event_id)
            prior_receipt = action_receipts.get(action_key)
            if prior_receipt is None:
                action_receipts[action_key] = action.source_record_sha256
                actions.append(action)
            elif prior_receipt != action.source_record_sha256:
                raise PolygonPITAlphaConversionError(
                    "conflicting corporate-action records share one vendor ID"
                )

    manifest_byte_authority_missing = any(row.blank_sha256_rows for row in shard_audits)
    manifest_inconsistent = any(
        row.malformed_manifest_rows
        or row.duplicate_manifest_rows
        or row.unexpected_manifest_symbols
        or row.missing_manifest_symbols
        or row.unexpected_covariate_symbols
        or row.missing_covariate_symbols
        or row.stale_output_path_rows
        or (
            row.canonical_missing_file_rows is not None
            and row.canonical_missing_file_rows > 0
        )
        or (
            row.canonical_nonregular_file_rows is not None
            and row.canonical_nonregular_file_rows > 0
        )
        or (
            row.canonical_hash_mismatch_rows is not None
            and row.canonical_hash_mismatch_rows > 0
        )
        or (
            row.canonical_size_mismatch_rows is not None
            and row.canonical_size_mismatch_rows > 0
        )
        or (
            row.canonical_row_count_mismatch_rows is not None
            and row.canonical_row_count_mismatch_rows > 0
        )
        or (
            row.canonical_schema_invalid_rows is not None
            and row.canonical_schema_invalid_rows > 0
        )
        or (
            row.unexpected_file_for_empty_rows is not None
            and row.unexpected_file_for_empty_rows > 0
        )
        or row.manifest_declared_remaining_symbol_days
        != row.manifest_derived_incomplete_symbol_days
        for row in shard_audits
    )
    missing_announcements = sum(
        not action.causal_availability_known for action in actions
    )
    blockers = [
        "future_selected_universe",
        "missing_point_in_time_membership_history",
        "missing_terminal_event_and_successor_ledger",
        "missing_causal_cash_return_series",
        "missing_independent_total_return_reconciliation",
        "missing_exchange_session_calendar_authority",
        "missing_permanent_security_id_authority",
        "missing_fold_specific_universe_materialization",
        "missing_training_tensor_authority",
        "adjusted_bar_split_accounting_policy_unresolved",
    ]
    if transitions or cross_ticker_figis:
        blockers.append("unresolved_permanent_security_identity_transitions")
    if missing_announcements:
        blockers.append("corporate_action_announcement_timestamps_incomplete")
    if manifest_byte_authority_missing:
        blockers.append("source_manifests_are_not_byte_authoritative")
    if manifest_inconsistent:
        blockers.append("source_manifest_inventory_is_inconsistent")

    staging_possible = all(
        sum(count for status, count in row.status_counts if status in _FILE_STATUSES)
        > 0
        and (
            not verify_canonical_files
            or (
                (row.observed_source_hash_rows or 0) > 0
                and (row.canonical_schema_invalid_rows or 0) == 0
            )
        )
        for row in shard_audits
    )
    bar_inventory_verified = verify_canonical_files and all(
        not row.blank_sha256_rows
        and not row.duplicate_manifest_rows
        and not row.stale_output_path_rows
        and not row.malformed_manifest_rows
        and not row.missing_manifest_symbols
        and not row.unexpected_manifest_symbols
        and (row.canonical_missing_file_rows or 0) == 0
        and (row.canonical_nonregular_file_rows or 0) == 0
        and (row.canonical_hash_mismatch_rows or 0) == 0
        and (row.canonical_size_mismatch_rows or 0) == 0
        and (row.canonical_row_count_mismatch_rows or 0) == 0
        and (row.canonical_schema_invalid_rows or 0) == 0
        and (row.unexpected_file_for_empty_rows or 0) == 0
        and (row.manifest_hash_verified_rows or 0)
        == sum(count for status, count in row.status_counts if status in _FILE_STATUSES)
        and row.manifest_declared_remaining_symbol_days
        == row.manifest_derived_incomplete_symbol_days
        for row in shard_audits
    )
    if not bar_inventory_verified:
        blockers.append("bar_source_inventory_not_verified")
    body: dict[str, Any] = {
        "schema": PIT_ALPHA_CONVERSION_AUDIT_SCHEMA,
        "shards": [asdict(row) for row in shard_audits],
        "expected_universe_symbol_count": len(expected_union),
        "observed_covariate_symbol_count": len(observed_union),
        "overlapping_covariate_symbols": overlaps,
        "unexpected_covariate_symbols": tuple(sorted(observed_union - expected_union)),
        "source_symbol_identity_transition_count": len(transitions),
        "source_symbol_identity_transitions": transitions,
        "cross_ticker_share_class_figi_count": len(cross_ticker_figis),
        "cross_ticker_share_class_figis": cross_ticker_figis,
        "identity_observation_count": len(observations),
        "dividend_candidate_count": sum(
            action.source_dataset == "dividends" for action in actions
        ),
        "split_candidate_count": sum(
            action.source_dataset == "splits" for action in actions
        ),
        "corporate_actions_missing_announcement_count": missing_announcements,
        "staging_conversion_possible": staging_possible,
        "bar_source_inventory_verified": bar_inventory_verified,
        "pit_alpha_training_ready": False,
        "reportable_pit_authority_ready": False,
        "blockers": tuple(blockers),
    }
    return PolygonPITAlphaConversionAudit(
        schema=PIT_ALPHA_CONVERSION_AUDIT_SCHEMA,
        shards=shard_audits,
        expected_universe_symbol_count=len(expected_union),
        observed_covariate_symbol_count=len(observed_union),
        overlapping_covariate_symbols=overlaps,
        unexpected_covariate_symbols=tuple(sorted(observed_union - expected_union)),
        source_symbol_identity_transition_count=len(transitions),
        source_symbol_identity_transitions=transitions,
        cross_ticker_share_class_figi_count=len(cross_ticker_figis),
        cross_ticker_share_class_figis=cross_ticker_figis,
        identity_observation_count=len(observations),
        dividend_candidate_count=sum(
            action.source_dataset == "dividends" for action in actions
        ),
        split_candidate_count=sum(
            action.source_dataset == "splits" for action in actions
        ),
        corporate_actions_missing_announcement_count=missing_announcements,
        staging_conversion_possible=staging_possible,
        bar_source_inventory_verified=bar_inventory_verified,
        pit_alpha_training_ready=False,
        reportable_pit_authority_ready=False,
        blockers=tuple(blockers),
        source_receipt_sha256=semantic_sha256(body),
    )


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise PolygonPITAlphaConversionError(
            "pandas and pyarrow are required for five-minute conversion"
        ) from exc
    return pd


def _semantic_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise PolygonPITAlphaConversionError(
                "semantic table contains a nonfinite value"
            )
        return {"float64_hex": value.hex()}
    if isinstance(value, str):
        return {"utf8": value}
    raise PolygonPITAlphaConversionError(
        f"unsupported semantic-table value type: {type(value).__name__}"
    )


def _table_semantic_sha256(
    frame: Any,
    *,
    columns: Sequence[str],
    schema: Sequence[tuple[str, str, bool]],
    sort_by: Sequence[str],
) -> str:
    if tuple(columns) != tuple(name for name, _, _ in schema):
        raise PolygonPITAlphaConversionError(
            "semantic table columns differ from their frozen schema"
        )
    ordered = frame.loc[:, list(columns)].sort_values(list(sort_by), kind="mergesort")
    rows = [
        [_semantic_scalar(value) for value in row]
        for row in ordered.itertuples(index=False, name=None)
    ]
    return semantic_sha256(
        {
            "schema": [
                {"name": name, "type": kind, "nullable": nullable}
                for name, kind, nullable in schema
            ],
            "rows": rows,
        }
    )


def _source_table_semantic_sha256(frame: Any) -> str:
    columns = tuple(column for column in _SOURCE_VALUE_COLUMNS if column in frame)
    kinds = {
        "symbol": ("utf8", False),
        "timestamp_ms": ("int64", False),
        "open": ("float64", False),
        "high": ("float64", False),
        "low": ("float64", False),
        "close": ("float64", False),
        "volume": ("float64", False),
        "vwap": ("float64", True),
        "transactions": ("int64", True),
        "adjusted": ("bool", True),
        "timespan": ("utf8", True),
        "multiplier": ("int64", True),
    }
    schema = tuple((column, *kinds[column]) for column in columns)
    return _table_semantic_sha256(
        frame,
        columns=columns,
        schema=schema,
        sort_by=("timestamp_ms",),
    )


def aggregate_polygon_second_bars_to_five_minutes(
    frame: Any, session: PolygonExchangeSessionAuthority
) -> Any:
    """Aggregate one regular-session symbol-day without zero-filling missing bins."""

    pd = _require_pandas()
    session.validate()
    required = {
        "symbol",
        "timestamp_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = required - set(frame.columns)
    if missing:
        raise PolygonPITAlphaConversionError(
            f"second-bar frame is missing required columns: {sorted(missing)}"
        )
    work = frame.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    if len(set(work["symbol"])) != 1:
        raise PolygonPITAlphaConversionError(
            "five-minute conversion requires one symbol-day"
        )
    work["timestamp_ms"] = pd.to_numeric(work["timestamp_ms"], errors="raise").astype(
        "int64"
    )
    if bool(work["timestamp_ms"].duplicated().any()):
        raise PolygonPITAlphaConversionError(
            "second-bar frame contains duplicate timestamps"
        )
    if "adjusted" in work and not bool(work["adjusted"].fillna(False).all()):
        raise PolygonPITAlphaConversionError(
            "organized staging conversion requires adjusted bars"
        )
    if "timespan" in work and bool(
        (work["timespan"].astype(str).str.lower() != "second").any()
    ):
        raise PolygonPITAlphaConversionError(
            "organized staging conversion requires second bars"
        )
    if "multiplier" in work and bool(
        (pd.to_numeric(work["multiplier"], errors="raise") != 1).any()
    ):
        raise PolygonPITAlphaConversionError(
            "organized staging conversion requires multiplier=1"
        )
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_ms"], unit="ms", utc=True)
    work["timestamp_exchange"] = work["timestamp_utc"].dt.tz_convert(EASTERN)
    observed_dates = {value.isoformat() for value in work["timestamp_exchange"].dt.date}
    if observed_dates != {session.session_date}:
        raise PolygonPITAlphaConversionError(
            "five-minute conversion differs from the authorized exchange date"
        )
    work = work.loc[
        (work["timestamp_ms"] >= session.open_at_ms)
        & (work["timestamp_ms"] < session.close_at_ms)
    ].copy()
    if work.empty:
        raise PolygonPITAlphaConversionError(
            "symbol-day contains no rows inside the authorized exchange session"
        )
    work = work.sort_values("timestamp_ms")
    work["interval_index"] = (work["timestamp_ms"] - session.open_at_ms) // (
        BAR_SECONDS * 1_000
    )
    if bool(
        (
            (work["interval_index"] < 0)
            | (work["interval_index"] >= session.scheduled_interval_count)
        ).any()
    ):
        raise PolygonPITAlphaConversionError(
            "regular-session row mapped outside the authorized session grid"
        )
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not bool(work[column].map(math.isfinite).all()):
            raise PolygonPITAlphaConversionError(
                f"second-bar {column} values must be finite"
            )
    if bool((work["volume"] < 0).any()):
        raise PolygonPITAlphaConversionError("second-bar volume cannot be negative")
    if bool(
        (
            (work["high"] < work[["open", "close"]].max(axis=1))
            | (work["low"] > work[["open", "close"]].min(axis=1))
            | (work["high"] < work["low"])
        ).any()
    ):
        raise PolygonPITAlphaConversionError("second-bar OHLC geometry is invalid")
    if "transactions" not in work:
        work["transactions"] = 0
    work["transactions"] = pd.to_numeric(work["transactions"], errors="coerce").fillna(
        0
    )
    if "vwap" not in work:
        work["vwap"] = float("nan")
    work["vwap"] = pd.to_numeric(work["vwap"], errors="coerce")
    work["vwap_notional"] = work["vwap"] * work["volume"]
    work["vwap_volume"] = work["volume"].where(work["vwap"].notna(), 0.0)

    grouped = work.groupby("interval_index", sort=True, observed=True)
    output = grouped.agg(
        symbol=("symbol", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        transactions=("transactions", "sum"),
        source_row_count=("timestamp_ms", "size"),
        latest_source_timestamp_ms=("timestamp_ms", "max"),
        vwap_notional=("vwap_notional", "sum"),
        vwap_volume=("vwap_volume", "sum"),
    ).reset_index()
    output["vwap"] = output["vwap_notional"] / output["vwap_volume"].where(
        output["vwap_volume"] > 0
    )
    output["session_date"] = session.session_date
    output["exchange"] = session.exchange
    output["interval_start_ms"] = (
        session.open_at_ms + output["interval_index"] * BAR_SECONDS * 1_000
    )
    output["economic_interval_end_ms"] = (
        output["interval_start_ms"] + BAR_SECONDS * 1_000
    )
    output["assumed_strategy_available_at_ms"] = (
        pd.concat(
            (
                output["economic_interval_end_ms"],
                output["latest_source_timestamp_ms"] + 1_000,
            ),
            axis=1,
        ).max(axis=1)
        + session.assumed_availability_lag_ms
    )
    output["interval_index"] = output["interval_index"].astype("int16")
    output["transactions"] = output["transactions"].astype("int64")
    output["source_row_count"] = output["source_row_count"].astype("int64")
    output["observed"] = True
    return output[[name for name, _, _ in _OUTPUT_SCHEMA]]


def resolve_symbol_day_path(
    shards: Sequence[OrganizedPolygonShard], symbol: str, day: str
) -> Path:
    """Return the path only after manifest-aware source authority succeeds."""

    return Path(resolve_symbol_day_source(shards, symbol, day).canonical_path)


def _read_authorized_source_frame(
    source: PolygonSymbolDaySourceAuthority,
) -> Any:
    source.validate()
    pd = _require_pandas()
    path = Path(source.canonical_path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise PolygonPITAlphaConversionError(
            "authorized source is unavailable"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != source.observed_size_bytes
        ):
            raise PolygonPITAlphaConversionError(
                "authorized source type or size drifted"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
            stream.seek(0)
            frame = pd.read_parquet(stream)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or digest.hexdigest() != source.observed_sha256
        or len(frame) != source.observed_row_count
    ):
        raise PolygonPITAlphaConversionError(
            "authorized source changed before conversion"
        )
    return frame


def _arrow_output_table(frame: Any) -> Any:
    pa, _ = _require_pyarrow()
    arrow_types = {
        "utf8": pa.string(),
        "int16": pa.int16(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }
    schema = pa.schema(
        [
            pa.field(name, arrow_types[kind], nullable=nullable)
            for name, kind, nullable in _OUTPUT_SCHEMA
        ]
    )
    values: dict[str, list[Any]] = {}
    for name, kind, nullable in _OUTPUT_SCHEMA:
        column: list[Any] = []
        for value in frame[name].tolist():
            if hasattr(value, "item"):
                value = value.item()
            if kind == "float64" and isinstance(value, float) and math.isnan(value):
                value = None
            if value is None and not nullable:
                raise PolygonPITAlphaConversionError(
                    f"output column {name} contains an unexpected null"
                )
            column.append(value)
        values[name] = column
    return pa.Table.from_pydict(values, schema=schema).replace_schema_metadata(None)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(parent: Path, name: str) -> tuple[int, Path]:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=parent
    )
    return descriptor, Path(raw_path)


def _write_temporary_bytes(parent: Path, name: str, raw: bytes) -> Path:
    descriptor, temporary = _temporary_path(parent, name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_temporary_parquet(parent: Path, name: str, table: Any) -> Path:
    _, pq = _require_pyarrow()
    descriptor, temporary = _temporary_path(parent, name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pq.write_table(
                table,
                stream,
                compression=_PARQUET_COMPRESSION,
                compression_level=_PARQUET_COMPRESSION_LEVEL,
                use_dictionary=False,
                write_statistics=True,
                row_group_size=_PARQUET_ROW_GROUP_SIZE,
                data_page_version="2.0",
                version="2.6",
            )
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _link_no_replace(source: Path, target: Path) -> None:
    try:
        os.link(source, target, follow_symlinks=False)
    except FileExistsError as exc:
        raise PolygonPITAlphaConversionError(
            f"refusing to replace staging artifact: {target}"
        ) from exc


def _write_new_bytes(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PolygonPITAlphaConversionError(f"output already exists: {path}")
    temporary = _write_temporary_bytes(path.parent, path.name, raw)
    try:
        _link_no_replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _rollback_owned_publications(
    publications: Sequence[tuple[Path, str]],
) -> None:
    for path, expected_sha in reversed(publications):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or file_sha256(path) != expected_sha
        ):
            raise PolygonPITAlphaConversionError(
                "cannot safely roll back a drifted staging publication"
            )
        path.unlink()


def convert_symbol_day_to_five_minute_staging(
    source: PolygonSymbolDaySourceAuthority,
    session: PolygonExchangeSessionAuthority,
    output_path: Path,
) -> dict[str, Any]:
    """Transactionally publish one nonreportable five-minute staging bundle."""

    source.validate()
    session.validate()
    if source.session_date != session.session_date:
        raise PolygonPITAlphaConversionError("source and exchange-session dates differ")
    _read_regular_file(
        Path(source.manifest_path), expected_sha256=source.manifest_file_sha256
    )
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    commit_path = output_path.with_suffix(output_path.suffix + ".commit.json")
    finals = (output_path, receipt_path, commit_path)
    if any(path.exists() or path.is_symlink() for path in finals):
        raise PolygonPITAlphaConversionError("staging output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = _read_authorized_source_frame(source)
    source_semantic_sha = _source_table_semantic_sha256(frame)
    converted = aggregate_polygon_second_bars_to_five_minutes(frame, session)
    output_semantic_sha = _table_semantic_sha256(
        converted,
        columns=tuple(name for name, _, _ in _OUTPUT_SCHEMA),
        schema=_OUTPUT_SCHEMA,
        sort_by=("interval_index",),
    )
    table = _arrow_output_table(converted)
    pa, _ = _require_pyarrow()
    pd = _require_pandas()

    temporaries: list[Path] = []
    publications: list[tuple[Path, str]] = []
    try:
        output_temporary = _write_temporary_parquet(
            output_path.parent, output_path.name, table
        )
        temporaries.append(output_temporary)
        output_file_sha = file_sha256(output_temporary)
        observed_indexes = tuple(int(value) for value in converted["interval_index"])
        observed_set = set(observed_indexes)
        missing_indexes = tuple(
            index
            for index in range(session.scheduled_interval_count)
            if index not in observed_set
        )
        body = {
            "schema": PIT_ALPHA_FIVE_MINUTE_STAGING_SCHEMA,
            "source_authority": asdict(source),
            "source_file_sha256": source.observed_sha256,
            "source_table_semantic_sha256": source_semantic_sha,
            "session_authority": asdict(session),
            "output_path": str(output_path),
            "output_file_sha256": output_file_sha,
            "output_table_semantic_sha256": output_semantic_sha,
            "output_schema_sha256": semantic_sha256(
                [
                    {"name": name, "type": kind, "nullable": nullable}
                    for name, kind, nullable in _OUTPUT_SCHEMA
                ]
            ),
            "symbol": source.symbol,
            "session_date": source.session_date,
            "exchange": session.exchange,
            "observed_interval_count": len(converted),
            "scheduled_interval_count": session.scheduled_interval_count,
            "observed_interval_indexes": observed_indexes,
            "scheduled_missing_interval_indexes": missing_indexes,
            "structurally_closed_intervals_are_not_missing": True,
            "missing_intervals_are_not_zero_filled": True,
            "source_prices_adjusted": bool(frame["adjusted"].all())
            if "adjusted" in frame
            else None,
            "availability_convention": "bar-end-plus-authorized-fixed-lag",
            "pandas_version": pd.__version__,
            "pyarrow_version": pa.__version__,
            "parquet_compression": _PARQUET_COMPRESSION,
            "parquet_compression_level": _PARQUET_COMPRESSION_LEVEL,
            "parquet_row_group_size": _PARQUET_ROW_GROUP_SIZE,
            "parquet_dictionary_encoding": False,
            "reportable_pit_authority": False,
            "pit_alpha_training_ready": False,
            "blocking_reason": (
                "permanent identity, PIT universe, terminal economics, and "
                "adjustment authorities are not closed"
            ),
        }
        receipt = {**body, "receipt_sha256": semantic_sha256(body)}
        receipt_raw = canonical_json_file_bytes(receipt)
        receipt_temporary = _write_temporary_bytes(
            output_path.parent, receipt_path.name, receipt_raw
        )
        temporaries.append(receipt_temporary)
        receipt_file_sha = hashlib.sha256(receipt_raw).hexdigest()
        commit_body = {
            "schema": PIT_ALPHA_STAGING_COMMIT_SCHEMA,
            "output_path": str(output_path),
            "output_file_sha256": output_file_sha,
            "output_table_semantic_sha256": output_semantic_sha,
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": receipt_file_sha,
            "staging_receipt_sha256": receipt["receipt_sha256"],
            "source_receipt_sha256": source.source_receipt_sha256,
            "session_receipt_sha256": session.receipt_sha256,
            "transaction_complete": True,
        }
        commit = {
            **commit_body,
            "commit_receipt_sha256": semantic_sha256(commit_body),
        }
        commit_raw = canonical_json_file_bytes(commit)
        commit_temporary = _write_temporary_bytes(
            output_path.parent, commit_path.name, commit_raw
        )
        temporaries.append(commit_temporary)
        commit_file_sha = hashlib.sha256(commit_raw).hexdigest()

        for temporary, final, expected_sha in (
            (output_temporary, output_path, output_file_sha),
            (receipt_temporary, receipt_path, receipt_file_sha),
            (commit_temporary, commit_path, commit_file_sha),
        ):
            _link_no_replace(temporary, final)
            publications.append((final, expected_sha))
        _fsync_directory(output_path.parent)
    except Exception:
        _rollback_owned_publications(publications)
        if publications:
            _fsync_directory(output_path.parent)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)

    return {
        "output_path": str(output_path),
        "output_file_sha256": output_file_sha,
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": receipt_file_sha,
        "receipt_sha256": receipt["receipt_sha256"],
        "commit_path": str(commit_path),
        "commit_file_sha256": commit_file_sha,
        "commit_receipt_sha256": commit["commit_receipt_sha256"],
        "reportable_pit_authority": False,
        "pit_alpha_training_ready": False,
    }


def load_five_minute_staging_publication(
    output_path: Path, *, expected_commit_file_sha256: str
) -> dict[str, Any]:
    """Validate the commit marker and both exact files of one staging transaction."""

    commit_path = output_path.with_suffix(output_path.suffix + ".commit.json")
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    commit_raw = _read_regular_file(
        commit_path, expected_sha256=expected_commit_file_sha256
    )
    try:
        commit = json.loads(commit_raw)
    except json.JSONDecodeError as exc:
        raise PolygonPITAlphaConversionError("staging commit is malformed") from exc
    if (
        not isinstance(commit, dict)
        or commit_raw != canonical_json_file_bytes(commit)
        or commit.get("schema") != PIT_ALPHA_STAGING_COMMIT_SCHEMA
        or commit.get("transaction_complete") is not True
        or commit.get("output_path") != str(output_path)
        or commit.get("receipt_path") != str(receipt_path)
    ):
        raise PolygonPITAlphaConversionError("staging commit contract drifted")
    commit_unsigned = {
        key: value for key, value in commit.items() if key != "commit_receipt_sha256"
    }
    if commit.get("commit_receipt_sha256") != semantic_sha256(commit_unsigned):
        raise PolygonPITAlphaConversionError("staging commit receipt drifted")
    output_raw = _read_regular_file(
        output_path,
        expected_sha256=_digest(
            "committed output SHA", commit.get("output_file_sha256")
        ),
    )
    receipt_raw = _read_regular_file(
        receipt_path,
        expected_sha256=_digest(
            "committed receipt file SHA", commit.get("receipt_file_sha256")
        ),
    )
    try:
        receipt = json.loads(receipt_raw)
    except json.JSONDecodeError as exc:
        raise PolygonPITAlphaConversionError("staging receipt is malformed") from exc
    if (
        not isinstance(receipt, dict)
        or receipt_raw != canonical_json_file_bytes(receipt)
        or receipt.get("schema") != PIT_ALPHA_FIVE_MINUTE_STAGING_SCHEMA
        or receipt.get("output_file_sha256") != hashlib.sha256(output_raw).hexdigest()
        or receipt.get("receipt_sha256") != commit.get("staging_receipt_sha256")
        or receipt.get("reportable_pit_authority") is not False
        or receipt.get("pit_alpha_training_ready") is not False
    ):
        raise PolygonPITAlphaConversionError("staging receipt contract drifted")
    receipt_unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if receipt.get("receipt_sha256") != semantic_sha256(receipt_unsigned):
        raise PolygonPITAlphaConversionError("staging receipt semantic hash drifted")
    try:
        source = PolygonSymbolDaySourceAuthority(**receipt["source_authority"])
        session = PolygonExchangeSessionAuthority(**receipt["session_authority"])
    except (KeyError, TypeError) as exc:
        raise PolygonPITAlphaConversionError(
            "staging receipt authorities are malformed"
        ) from exc
    source.validate()
    session.validate()
    expected_output_schema_sha = semantic_sha256(
        [
            {"name": name, "type": kind, "nullable": nullable}
            for name, kind, nullable in _OUTPUT_SCHEMA
        ]
    )
    if (
        commit.get("output_table_semantic_sha256")
        != receipt.get("output_table_semantic_sha256")
        or commit.get("source_receipt_sha256") != source.source_receipt_sha256
        or commit.get("session_receipt_sha256") != session.receipt_sha256
        or receipt.get("source_file_sha256") != source.observed_sha256
        or receipt.get("output_path") != str(output_path)
        or receipt.get("output_schema_sha256") != expected_output_schema_sha
        or receipt.get("symbol") != source.symbol
        or receipt.get("session_date") != source.session_date
        or source.session_date != session.session_date
        or receipt.get("exchange") != session.exchange
        or receipt.get("scheduled_interval_count") != session.scheduled_interval_count
        or receipt.get("availability_convention") != "bar-end-plus-authorized-fixed-lag"
        or receipt.get("missing_intervals_are_not_zero_filled") is not True
        or receipt.get("structurally_closed_intervals_are_not_missing") is not True
    ):
        raise PolygonPITAlphaConversionError("staging authority linkage drifted")
    pd = _require_pandas()
    try:
        output_frame = pd.read_parquet(io.BytesIO(output_raw))
    except Exception as exc:
        raise PolygonPITAlphaConversionError(
            "committed staging Parquet is unreadable"
        ) from exc
    output_semantic_sha = _table_semantic_sha256(
        output_frame,
        columns=tuple(name for name, _, _ in _OUTPUT_SCHEMA),
        schema=_OUTPUT_SCHEMA,
        sort_by=("interval_index",),
    )
    observed_indexes = tuple(int(value) for value in output_frame["interval_index"])
    expected_missing = tuple(
        index
        for index in range(session.scheduled_interval_count)
        if index not in set(observed_indexes)
    )
    if (
        output_semantic_sha != receipt.get("output_table_semantic_sha256")
        or len(output_frame) != receipt.get("observed_interval_count")
        or observed_indexes != tuple(receipt.get("observed_interval_indexes") or ())
        or expected_missing
        != tuple(receipt.get("scheduled_missing_interval_indexes") or ())
        or bool(output_frame["observed"].all()) is not True
    ):
        raise PolygonPITAlphaConversionError(
            "committed staging table differs from its semantic receipt"
        )
    return {
        "commit": commit,
        "receipt": receipt,
        "output_file_sha256": hashlib.sha256(output_raw).hexdigest(),
        "commit_file_sha256": hashlib.sha256(commit_raw).hexdigest(),
    }


def write_conversion_audit(path: Path, audit: PolygonPITAlphaConversionAudit) -> str:
    """Write one canonical, no-clobber conversion-readiness report."""

    return _write_new_bytes(path, canonical_json_file_bytes(audit.to_dict()))


__all__ = [
    "EXPECTED_NORMAL_SESSION_INTERVALS",
    "PIT_ALPHA_CONVERSION_AUDIT_SCHEMA",
    "PIT_ALPHA_FIVE_MINUTE_STAGING_SCHEMA",
    "PIT_ALPHA_STAGING_COMMIT_SCHEMA",
    "POLYGON_SESSION_AUTHORITY_SCHEMA",
    "POLYGON_SYMBOL_DAY_SOURCE_SCHEMA",
    "OrganizedPolygonShard",
    "OrganizedPolygonShardAudit",
    "PolygonCorporateActionCandidate",
    "PolygonExchangeSessionAuthority",
    "PolygonIdentityObservation",
    "PolygonManifestRow",
    "PolygonPITAlphaConversionAudit",
    "PolygonPITAlphaConversionError",
    "PolygonSymbolDaySourceAuthority",
    "aggregate_polygon_second_bars_to_five_minutes",
    "audit_organized_polygon_for_pit_alpha",
    "build_exchange_session_authority",
    "convert_symbol_day_to_five_minute_staging",
    "default_organized_polygon_shards",
    "iter_corporate_action_candidates",
    "iter_identity_observations",
    "load_exchange_session_authority",
    "load_five_minute_staging_publication",
    "resolve_symbol_day_path",
    "resolve_symbol_day_source",
    "write_conversion_audit",
    "write_exchange_session_authority",
]
