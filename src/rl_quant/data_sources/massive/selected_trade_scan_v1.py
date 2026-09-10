"""One-pass original-gzip provenance bridge for an exact source-ticker selection.

This is an engineering source bridge, not a finalized whole-market canonical
scan, historical identity authority, feature builder, or V5 training promotion.
Every callback row is PROVISIONAL: a caller must stage its output and publish
only after this function returns validated completion evidence. In particular,
gzip CRC, trailing bytes, source hashes, and transaction mutations can fail
after selected rows have already reached the callback.

No source bytes are written. Memory is bounded by one caller-limited physical
line, gzip buffers, one selected row, and counters for the explicit selection.
The callback owns its own bounded storage, cancellation, and output budget.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from decimal import DecimalException
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import BinaryIO

from rl_quant.data_sources.massive.finalized_daily_scan import (
    MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256,
    _parse_conditions,
)
from rl_quant.data_sources.massive.source_receipts import (
    MASSIVE_LOADED_SOURCE_OBJECT_SCHEMA,
    LoadedMassiveSourceObject,
    MassiveSourceCommit,
    MassiveSourceObjectReceipt,
    _open_parent_directory,
    open_loaded_massive_source_stream,
)
from rl_quant.data_sources.massive.trade_canonicalization import (
    MASSIVE_TRADE_CANONICALIZATION_SPEC_SHA256,
    canonicalize_massive_flat_file_trade,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_COLUMNS,
    MASSIVE_FLAT_TRADE_SCHEMA_SHA256,
    MASSIVE_FLAT_TRADES_DATASET_ID,
    MassiveExtractedTradeRow,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_payload,
    file_sha256,
    semantic_sha256,
)


SELECTED_TRADE_ROW_V1_SCHEMA = "rl-quant.massive-selected-original-trade-row-v1"
SELECTED_TRADE_SCAN_V1_SCHEMA = "rl-quant.massive-selected-trade-file-scan-v1"
SELECTED_TRADE_SCAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
SELECTED_TRADE_SCAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "columns": MASSIVE_FLAT_TRADE_COLUMNS,
        "parser": "strict-utf8-csv-one-record-per-original-physical-line",
        "provenance": "sha256-original-decompressed-line-bytes-including-terminator",
        "selection": "exact-source-ticker-only;ordered-caller-selection",
        "source_row_number": "physical-line-number;header=1;first-data=2",
        "canonicalization": MASSIVE_TRADE_CANONICALIZATION_SPEC_SHA256,
        "condition_parser_source": MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256,
        "canonicalization_failure": "retain-original-selected-row-with-explicit-error",
        "conditions_and_corrections": "retain-all;no-authority-resolution-or-filter",
        "completion": "native-transaction-plus-full-compressed-sha-size-crc-eof-stat",
        "whole_source_canonical_scan_qualified": False,
        "identity_qualified": False,
        "training_ready": False,
    }
)
_METADATA_BYTES = 1024 * 1024
_IDENTITY_FIELDS = (
    "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
    "st_uid", "st_gid", "st_mode", "st_nlink",
)


class MassiveSelectedTradeScanError(ValueError):
    """The selected stream cannot receive source-completion evidence."""


def _digest(name: str, value: object) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise MassiveSelectedTradeScanError(f"{name} must be a lowercase SHA-256")
    return value


def _count(name: str, value: object, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise MassiveSelectedTradeScanError(f"{name} has an invalid integer value")
    return value


def _relative(value: str) -> str:
    if not isinstance(value, str):
        raise MassiveSelectedTradeScanError("source path must be a string")
    path = PurePosixPath(value)
    if (not value or not path.parts or path.is_absolute() or str(path) != value
            or any(part in {".", ".."} for part in path.parts)):
        raise MassiveSelectedTradeScanError("source path must be exactly canonical and relative")
    return value


def _identity(info: os.stat_result) -> tuple[int, ...]:
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise MassiveSelectedTradeScanError("source must be a regular worker-owned file")
    return tuple(getattr(info, field) for field in _IDENTITY_FIELDS)


def _metadata(parent_fd: int, name: str, expected_sha: str):
    """Bound only metadata; never call the native eager payload loader."""
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        before = _identity(os.fstat(fd))
        if before[2] > _METADATA_BYTES:
            raise MassiveSelectedTradeScanError("source metadata exceeds its byte bound")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(_METADATA_BYTES + 1)
        if (_identity(os.fstat(fd)) != before or len(data) != before[2]
                or hashlib.sha256(data).hexdigest() != expected_sha):
            raise MassiveSelectedTradeScanError("source metadata hash or identity differs")
    finally:
        os.close(fd)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MassiveSelectedTradeScanError("duplicate source metadata JSON key")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise MassiveSelectedTradeScanError("source metadata must be a JSON object")
    return value, before


class _SequenceDigest:
    """The native semantic SHA of a sequence without retaining its members."""

    def __init__(self):
        self._hash = hashlib.sha256(b"[")
        self._count = 0

    def add(self, value: object) -> None:
        if self._count:
            self._hash.update(b",")
        self._hash.update(canonical_json_payload(value))
        self._count += 1

    def hexdigest(self) -> str:
        final = self._hash.copy()
        final.update(b"]")
        return final.hexdigest()


class _CompressedReader:
    def __init__(self, source: BinaryIO):
        self.source = source
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise MassiveSelectedTradeScanError("unbounded compressed read is forbidden")
        data = self.source.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data


def _csv_fields(raw_line: bytes, row_number: int) -> tuple[str, ...]:
    try:
        rows = list(csv.reader((raw_line.decode("utf-8"),), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise MassiveSelectedTradeScanError(
            f"physical CSV line {row_number} is malformed; no selection completion"
        ) from exc
    if len(rows) != 1 or len(rows[0]) != len(MASSIVE_FLAT_TRADE_COLUMNS):
        raise MassiveSelectedTradeScanError(
            f"physical CSV line {row_number} has the wrong field count"
        )
    return tuple(rows[0])


@dataclass(frozen=True, slots=True)
class MassiveSelectedOriginalTradeRowV1:
    source_row_number: int
    raw_line_sha256: str
    original_values: tuple[str, ...]
    extracted_row: MassiveExtractedTradeRow | None
    canonicalization_error: str | None
    receipt_sha256: str
    schema: str = SELECTED_TRADE_ROW_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        if self.schema != SELECTED_TRADE_ROW_V1_SCHEMA or self.source_row_number < 2:
            raise MassiveSelectedTradeScanError("selected original row identity differs")
        _count("physical source row number", self.source_row_number)
        _digest("original line SHA", self.raw_line_sha256)
        if (len(self.original_values) != len(MASSIVE_FLAT_TRADE_COLUMNS)
                or any(not isinstance(value, str) for value in self.original_values)):
            raise MassiveSelectedTradeScanError("original selected values are malformed")
        if self.extracted_row is None:
            if not isinstance(self.canonicalization_error, str) or not self.canonicalization_error:
                raise MassiveSelectedTradeScanError("invalid canonical row lacks an explicit error")
        else:
            self.extracted_row.validate()
            if (self.canonicalization_error is not None
                    or self.extracted_row.source_row_number != self.source_row_number
                    or self.extracted_row.raw_row_sha256 != self.raw_line_sha256
                    or self.extracted_row.canonical_record.ticker != self.original_values[0]):
                raise MassiveSelectedTradeScanError("canonical row lost selected original provenance")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSelectedTradeScanError("selected row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveSelectedTradeFileScanEvidenceV1:
    loaded_source: LoadedMassiveSourceObject
    receipt_file_sha256: str
    commit_file_sha256: str
    source_identity: tuple[int, ...]
    ordered_tickers: tuple[str, ...]
    selector_receipt_sha256: str
    maximum_line_bytes: int
    source_row_count: int
    selected_row_count: int
    unselected_row_count: int
    selected_canonical_row_count: int
    selected_canonicalization_error_count: int
    selected_ticker_counts: tuple[tuple[str, int], ...]
    selected_ticker_canonicalization_error_counts: tuple[tuple[str, int], ...]
    compressed_bytes: int
    compressed_sha256: str
    decompressed_bytes: int
    decompressed_sha256: str
    selected_row_inventory_sha256: str
    selected_correction_condition_inventory_sha256: str
    complete_exact_ticker_selection: bool
    whole_source_canonical_scan_qualified: bool
    identity_qualified: bool
    training_ready: bool
    parser_spec_sha256: str
    parser_source_sha256: str
    receipt_sha256: str
    schema: str = SELECTED_TRADE_SCAN_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        self.loaded_source.validate()
        receipt, commit = self.loaded_source.receipt, self.loaded_source.commit
        if self.schema != SELECTED_TRADE_SCAN_V1_SCHEMA:
            raise MassiveSelectedTradeScanError("selected scan schema differs")
        if (receipt.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID
                or receipt.schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256
                or commit.payload_file_sha256 != receipt.physical_sha256
                or commit.source_receipt_sha256 != receipt.receipt_sha256
                or commit.receipt_file_sha256 != self.receipt_file_sha256
                or commit.committed_at_ms < receipt.downloaded_at_ms
                or self.loaded_source.verified_at_ms < commit.committed_at_ms):
            raise MassiveSelectedTradeScanError("selected scan transaction links differ")
        if (not self.ordered_tickers or len(set(self.ordered_tickers)) != len(self.ordered_tickers)
                or any(not isinstance(t, str) or not t or t != t.strip() for t in self.ordered_tickers)
                or self.selector_receipt_sha256 != _selector_sha(self.ordered_tickers)):
            raise MassiveSelectedTradeScanError("ordered exact ticker selection differs")
        for name in ("source_row_count", "selected_row_count", "unselected_row_count",
                     "selected_canonical_row_count", "selected_canonicalization_error_count",
                     "compressed_bytes", "decompressed_bytes"):
            _count(name, getattr(self, name))
        _count("line byte bound", self.maximum_line_bytes, positive=True)
        if (self.source_row_count <= 0
                or self.selected_row_count + self.unselected_row_count != self.source_row_count
                or self.selected_canonical_row_count + self.selected_canonicalization_error_count != self.selected_row_count):
            raise MassiveSelectedTradeScanError("selected scan row counts do not reconcile")
        for counts, total in ((self.selected_ticker_counts, self.selected_row_count),
                              (self.selected_ticker_canonicalization_error_counts, self.selected_canonicalization_error_count)):
            if tuple(t for t, _ in counts) != self.ordered_tickers:
                raise MassiveSelectedTradeScanError("ticker count order differs")
            if sum(_count("ticker count", n) for _, n in counts) != total:
                raise MassiveSelectedTradeScanError("ticker counts do not reconcile")
        if any(errors > selected for (_, errors), (_, selected) in zip(
                self.selected_ticker_canonicalization_error_counts,
                self.selected_ticker_counts, strict=True)):
            raise MassiveSelectedTradeScanError("ticker canonicalization errors exceed selected rows")
        if (len(self.source_identity) != len(_IDENTITY_FIELDS)
                or self.source_identity[0] != self.loaded_source.payload_device
                or self.source_identity[1] != self.loaded_source.payload_inode
                or self.source_identity[2] != self.compressed_bytes
                or self.source_identity[4] != self.loaded_source.payload_ctime_ns
                or self.compressed_bytes != receipt.content_length
                or self.compressed_sha256 != receipt.physical_sha256):
            raise MassiveSelectedTradeScanError("source byte or inode identity differs")
        if (self.complete_exact_ticker_selection is not True
                or self.whole_source_canonical_scan_qualified is not False
                or self.identity_qualified is not False or self.training_ready is not False):
            raise MassiveSelectedTradeScanError("source selection cannot authorize canonical or training qualification")
        if (self.parser_spec_sha256 != SELECTED_TRADE_SCAN_V1_SPEC_SHA256
                or self.parser_source_sha256 != SELECTED_TRADE_SCAN_V1_SOURCE_SHA256):
            raise MassiveSelectedTradeScanError("selected parser implementation differs")
        for name in ("receipt_file_sha256", "commit_file_sha256", "selector_receipt_sha256",
                     "compressed_sha256", "decompressed_sha256", "selected_row_inventory_sha256",
                     "selected_correction_condition_inventory_sha256", "receipt_sha256"):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSelectedTradeScanError("selected scan receipt differs")


def _selector_sha(tickers: tuple[str, ...]) -> str:
    return semantic_sha256({"selection": "exact-source-ticker-only", "ordered_tickers": tickers})


def _selected_row(values: tuple[str, ...], raw_line: bytes, row_number: int):
    raw_sha = hashlib.sha256(raw_line).hexdigest()
    extracted, error = None, None
    source: dict[str, object] = dict(zip(MASSIVE_FLAT_TRADE_COLUMNS, values, strict=True))
    try:
        # Reuse native JSON-list/comma fallback parsing, not a new condition
        # qualification rule. The untouched original string remains in the row.
        source["conditions"] = _parse_conditions(values[1])
        for nullable in ("tape", "trf_id", "trf_timestamp"):
            if source[nullable] == "":
                source[nullable] = None
        canonical = canonicalize_massive_flat_file_trade(source, raw_source_record_sha256=raw_sha)
        extracted = MassiveExtractedTradeRow.build(
            source_row_number=row_number, raw_row_sha256=raw_sha, canonical_record=canonical,
        )
    except (ValueError, TypeError, OverflowError, DecimalException) as exc:
        error = type(exc).__name__ + ": " + str(exc)[:1000]
    body = {"schema": SELECTED_TRADE_ROW_V1_SCHEMA, "source_row_number": row_number,
            "raw_line_sha256": raw_sha, "original_values": values,
            "extracted_row": None if extracted is None else asdict(extracted),
            "canonicalization_error": error}
    row = MassiveSelectedOriginalTradeRowV1(
        source_row_number=row_number, raw_line_sha256=raw_sha, original_values=values,
        extracted_row=extracted, canonicalization_error=error, receipt_sha256=semantic_sha256(body),
    )
    row.validate()
    return row


def scan_massive_selected_trade_file_v1(
    *, root: str | Path, payload_relative_path: str,
    expected_source_object_key: str, expected_receipt_file_sha256: str,
    expected_commit_file_sha256: str, expected_entitlement_receipt_sha256: str,
    expected_compressed_sha256: str, expected_compressed_bytes: int,
    tickers: Sequence[str], row_sink: Callable[[MassiveSelectedOriginalTradeRowV1], None],
    maximum_line_bytes: int, verified_at_ms: int,
    progress_callback: Callable[[int, int], None] | None = None,
    original_line_sink: Callable[[MassiveSelectedOriginalTradeRowV1, bytes], None] | None = None,
) -> MassiveSelectedTradeFileScanEvidenceV1:
    """Scan one actual persisted source; return evidence only after full validation.

    The callback receives every exact-ticker row, including pre/post-market,
    unknown numeric correction/condition codes, and canonicalization failures.
    It must not publish or use rows economically before successful return.
    ``maximum_line_bytes`` is an explicit parser allocation, not a dataset cap.
    The optional progress callback receives physical data-row and selected-row
    counts every 8192 rows and after EOF, so the caller can enforce its budget.
    ``original_line_sink`` optionally receives that same provisional row and
    its exact original bytes (including the line terminator), never reserialized
    CSV. It has the same completion and caller-owned allocation requirements.
    """
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or root.stat().st_uid != os.getuid():
        raise MassiveSelectedTradeScanError("source root must be canonical, absolute, and worker-owned")
    relative = _relative(payload_relative_path)
    _relative(expected_source_object_key)
    for name, value in (("receipt file", expected_receipt_file_sha256), ("commit file", expected_commit_file_sha256),
                        ("entitlement", expected_entitlement_receipt_sha256), ("payload", expected_compressed_sha256)):
        _digest(name, value)
    _count("expected compressed bytes", expected_compressed_bytes, positive=True)
    _count("physical line byte allocation", maximum_line_bytes, positive=True)
    _count("verification timestamp", verified_at_ms)
    ordered = tuple(tickers)
    if (isinstance(tickers, (str, bytes)) or not ordered or len(set(ordered)) != len(ordered)
            or any(not isinstance(t, str) or not t or t != t.strip() for t in ordered)
            or not callable(row_sink)
            or (original_line_sink is not None and not callable(original_line_sink))):
        raise MassiveSelectedTradeScanError("exact selection or provisional sink is malformed")
    selected_set = frozenset(ordered)
    ticker_counts = dict.fromkeys(ordered, 0)
    invalid_counts = dict.fromkeys(ordered, 0)
    parent_fd, payload_name = _open_parent_directory(root, relative, create=False)
    receipt_name, commit_name = payload_name + ".receipt.json", payload_name + ".commit.json"
    source_rows = selected_rows = invalid_rows = decompressed_bytes = 0
    rows_hash, corrections_hash = _SequenceDigest(), _SequenceDigest()
    plain_hash = hashlib.sha256()
    try:
        receipt_value, receipt_stat = _metadata(parent_fd, receipt_name, expected_receipt_file_sha256)
        commit_value, commit_stat = _metadata(parent_fd, commit_name, expected_commit_file_sha256)
        receipt, commit = MassiveSourceObjectReceipt(**receipt_value), MassiveSourceCommit(**commit_value)
        receipt.validate()
        commit.validate()
        expected_receipt_path = str(PurePosixPath(relative).with_name(receipt_name))
        if (receipt.dataset_id != MASSIVE_FLAT_TRADES_DATASET_ID
                or receipt.source_object_key != expected_source_object_key
                or receipt.schema_sha256 != MASSIVE_FLAT_TRADE_SCHEMA_SHA256
                or receipt.entitlement_receipt_sha256 != expected_entitlement_receipt_sha256
                or receipt.physical_sha256 != expected_compressed_sha256
                or receipt.content_length != expected_compressed_bytes
                or commit.receipt_file_sha256 != expected_receipt_file_sha256
                or commit.payload_file_sha256 != receipt.physical_sha256
                or commit.source_receipt_sha256 != receipt.receipt_sha256
                or commit.payload_relative_path != relative or commit.receipt_relative_path != expected_receipt_path
                or commit.committed_at_ms < receipt.downloaded_at_ms or verified_at_ms < commit.committed_at_ms):
            raise MassiveSelectedTradeScanError("persisted source transaction differs from exact expected scope")
        fd = os.open(payload_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            before = _identity(os.fstat(fd))
            if before[2] != receipt.content_length:
                raise MassiveSelectedTradeScanError("source initial compressed size differs")
            with os.fdopen(fd, "rb", closefd=False) as source:
                compressed = _CompressedReader(source)
                with gzip.GzipFile(fileobj=compressed, mode="rb") as decoded:
                    header = decoded.readline(maximum_line_bytes + 1)
                    if len(header) > maximum_line_bytes or _csv_fields(header, 1) != MASSIVE_FLAT_TRADE_COLUMNS:
                        raise MassiveSelectedTradeScanError("original trade header or line bound differs")
                    plain_hash.update(header)
                    decompressed_bytes += len(header)
                    while True:
                        raw_line = decoded.readline(maximum_line_bytes + 1)
                        if not raw_line:
                            break
                        if len(raw_line) > maximum_line_bytes:
                            raise MassiveSelectedTradeScanError("physical CSV line exceeds caller allocation")
                        source_rows += 1
                        values = _csv_fields(raw_line, source_rows + 1)
                        plain_hash.update(raw_line)
                        decompressed_bytes += len(raw_line)
                        if values[0] in selected_set:
                            row = _selected_row(values, raw_line, source_rows + 1)
                            selected_rows += 1
                            ticker_counts[values[0]] += 1
                            if row.extracted_row is None:
                                invalid_rows += 1
                                invalid_counts[values[0]] += 1
                            rows_hash.add(row.receipt_sha256)
                            corrections_hash.add((row.source_row_number, values[0], values[1], values[2]))
                            row_sink(row)
                            if original_line_sink is not None:
                                original_line_sink(row, raw_line)
                        if progress_callback is not None and source_rows % 8192 == 0:
                            progress_callback(source_rows, selected_rows)
                # Never drain unparsed trailing compressed bytes to make a hash match.
                if compressed.read(1):
                    raise MassiveSelectedTradeScanError("gzip left unparsed trailing compressed bytes")
                if (compressed.bytes_read != receipt.content_length
                        or compressed.digest.hexdigest() != receipt.physical_sha256):
                    raise MassiveSelectedTradeScanError("complete compressed source hash or size differs")
            if _identity(os.fstat(fd)) != before:
                raise MassiveSelectedTradeScanError("source worker-local identity changed during scan")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    if source_rows <= 0:
        raise MassiveSelectedTradeScanError("source has no physical data rows")
    if progress_callback is not None:
        progress_callback(source_rows, selected_rows)
    # Reopen through the supplied root again, not only through an old directory FD.
    if root.resolve() != root:
        raise MassiveSelectedTradeScanError("source root ancestry changed")
    parent_fd, final_name = _open_parent_directory(root, relative, create=False)
    try:
        final_fd = os.open(final_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            if _identity(os.fstat(final_fd)) != before:
                raise MassiveSelectedTradeScanError("source path was replaced during scan")
        finally:
            os.close(final_fd)
        if (_metadata(parent_fd, receipt_name, expected_receipt_file_sha256)[1] != receipt_stat
                or _metadata(parent_fd, commit_name, expected_commit_file_sha256)[1] != commit_stat):
            raise MassiveSelectedTradeScanError("source transaction metadata identity changed")
    finally:
        os.close(parent_fd)
    # A native Loaded object is created ONLY now: the real payload was fully hashed.
    loaded_body = {"schema": MASSIVE_LOADED_SOURCE_OBJECT_SCHEMA, "receipt": asdict(receipt),
                   "commit": asdict(commit), "payload_relative_path": relative,
                   "payload_device": before[0], "payload_inode": before[1], "payload_ctime_ns": before[4],
                   "verified_at_ms": verified_at_ms}
    loaded = LoadedMassiveSourceObject(
        receipt=receipt, commit=commit, payload_relative_path=relative,
        payload_device=before[0], payload_inode=before[1], payload_ctime_ns=before[4],
        verified_at_ms=verified_at_ms, receipt_sha256=semantic_sha256(loaded_body),
    )
    loaded.validate()
    with open_loaded_massive_source_stream(root=root, loaded_source=loaded) as final_source:
        if _identity(os.fstat(final_source.fileno())) != before:
            raise MassiveSelectedTradeScanError("verified loaded source identity differs")
    body = {
        "schema": SELECTED_TRADE_SCAN_V1_SCHEMA, "loaded_source": asdict(loaded),
        "receipt_file_sha256": expected_receipt_file_sha256, "commit_file_sha256": expected_commit_file_sha256,
        "source_identity": before, "ordered_tickers": ordered, "selector_receipt_sha256": _selector_sha(ordered),
        "maximum_line_bytes": maximum_line_bytes, "source_row_count": source_rows,
        "selected_row_count": selected_rows, "unselected_row_count": source_rows - selected_rows,
        "selected_canonical_row_count": selected_rows - invalid_rows,
        "selected_canonicalization_error_count": invalid_rows,
        "selected_ticker_counts": tuple(ticker_counts.items()),
        "selected_ticker_canonicalization_error_counts": tuple(invalid_counts.items()),
        "compressed_bytes": compressed.bytes_read, "compressed_sha256": compressed.digest.hexdigest(),
        "decompressed_bytes": decompressed_bytes, "decompressed_sha256": plain_hash.hexdigest(),
        "selected_row_inventory_sha256": rows_hash.hexdigest(),
        "selected_correction_condition_inventory_sha256": corrections_hash.hexdigest(),
        "complete_exact_ticker_selection": True, "whole_source_canonical_scan_qualified": False,
        "identity_qualified": False, "training_ready": False,
        "parser_spec_sha256": SELECTED_TRADE_SCAN_V1_SPEC_SHA256,
        "parser_source_sha256": SELECTED_TRADE_SCAN_V1_SOURCE_SHA256,
    }
    evidence = MassiveSelectedTradeFileScanEvidenceV1(
        **{**body, "loaded_source": loaded}, receipt_sha256=semantic_sha256(body),
    )
    evidence.validate()
    return evidence


__all__ = [
    "MassiveSelectedOriginalTradeRowV1", "MassiveSelectedTradeFileScanEvidenceV1",
    "MassiveSelectedTradeScanError", "SELECTED_TRADE_ROW_V1_SCHEMA", "SELECTED_TRADE_SCAN_V1_SCHEMA",
    "SELECTED_TRADE_SCAN_V1_SOURCE_SHA256", "SELECTED_TRADE_SCAN_V1_SPEC_SHA256",
    "scan_massive_selected_trade_file_v1",
]
