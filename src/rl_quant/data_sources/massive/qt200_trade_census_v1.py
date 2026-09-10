"""Bounded historical-message census, never a corrected trade or training source.

Read the original transaction once through the lossless selected-row scanner.
The native correction inventory supplies *candidate labels*, not evidence that
its lifecycle semantics apply to this historical export. Nothing is discarded,
deduplicated, linked to a predecessor, assigned a provider ID, or made fillable.
Only completed original-file validation permits diagnostic publication.
"""

from __future__ import annotations

import base64
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.qt200_native_scan_v1 import (
    _directory, _publish, _REQUEST_FIELDS, _snapshot,
)
from rl_quant.data_sources.massive.selected_trade_scan_v1 import (
    MassiveSelectedOriginalTradeRowV1, scan_massive_selected_trade_file_v1,
)
from rl_quant.data_sources.massive.trade_extraction import MASSIVE_FLAT_TRADE_COLUMNS, _parse_conditions
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, file_sha256, semantic_sha256


SCHEMA = "rl-quant.qt200-historical-message-census-v1"
SOURCE_SHA256 = file_sha256(Path(__file__))
CLAIMS = dict(diagnostic_only=True, identity_qualified=False,
              message_semantics_qualified=False, correction_replay_qualified=False,
              bars_tape_fills_qualified=False, training_ready=False)
GROUP_COLUMNS = (
    "membership_scope", "source_ticker", "raw_tape", "raw_exchange", "raw_trf_id",
    "raw_correction", "native_candidate_kind", "raw_conditions", "condition_parse_status",
    "provider_id_status", "size_status", "price_status",
)


class Qt200CensusError(ValueError):
    """The diagnostic cannot complete; keep the failed attempt and original source."""


@dataclass(frozen=True)
class CensusLimits:
    maximum_groups: int = 65536
    maximum_group_key_bytes: int = 32 * 1024**2
    maximum_sequence_keys: int = 8_000_000
    maximum_selected_rows: int = 8_000_000
    maximum_samples: int = 256
    samples_per_category: int = 2
    maximum_sample_bytes: int = 8 * 1024**2
    output_budget_bytes: int = 256 * 1024**2

    def validate(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise Qt200CensusError("Census allocations must be explicit positive integers")


def _number_status(raw: str) -> str:
    if not raw.strip():
        return "absent"
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return "malformed"
    if not value.is_finite():
        return "nonfinite"
    return "zero" if value == 0 else "positive" if value > 0 else "negative"


def _unsigned_integer(raw: str) -> tuple[str, int | None]:
    if not raw.strip():
        return "absent", None
    if not raw.isascii() or not raw.isdecimal():
        return "malformed", None
    # Avoid arbitrarily large int conversion while retaining the original cell.
    stripped = raw.lstrip("0") or "0"
    if len(stripped) > 20 or int(stripped) > 2**64 - 1:
        return "outside-uint64", None
    return "valid", int(stripped)


class _SequenceIndex:
    """Hard-capped in-memory index, one packed (ticker, uint64 sequence) key.

    Values are 32-byte original-line hashes. Equal/different means relative to
    the first occurrence, not a proven retransmission/transaction identity.
    No spill, on-disk database, or silent eviction is permitted.
    """

    def __init__(self, maximum: int):
        self.maximum = maximum
        self.first: dict[int, bytes] = {}

    def observe(self, ticker_index: int, sequence: int, raw_sha256: str) -> str:
        key = (ticker_index << 64) | sequence
        digest = bytes.fromhex(raw_sha256)
        first = self.first.get(key)
        if first is not None:
            return "repeated-same-line-hash" if first == digest else "repeated-different-line-hash"
        if len(self.first) >= self.maximum:
            raise Qt200CensusError("Sequence-key allocation exhausted; no complete census")
        self.first[key] = digest
        return "first-observation"


class _Census:
    def __init__(self, tickers, qt200, corrections, limits):
        self.limits = limits
        self.ticker_indexes = {ticker: index for index, ticker in enumerate(tickers)}
        self.qt200 = frozenset(qt200)
        self.corrections = {rule.correction_code: rule.semantic_kind for rule in corrections.rules}
        self.groups: Counter[tuple[str, ...]] = Counter()
        self.key_bytes = self.rows = self.sample_bytes = 0
        self.sequence = _SequenceIndex(limits.maximum_sequence_keys)
        self.sequence_counts: Counter[tuple[str, str]] = Counter()
        self.samples: list[dict[str, Any]] = []
        self.sample_categories: Counter[tuple[str, ...]] = Counter()

    def observe(self, row: MassiveSelectedOriginalTradeRowV1, raw_line: bytes) -> None:
        # The scanner already validates the row and computes its original hash.
        values = row.original_values
        ticker, conditions, correction = values[:3]
        scope = "qt200" if ticker in self.qt200 else "alias_candidate"
        size, price = _number_status(values[9]), _number_status(values[6])
        identity = "blank" if not values[4].strip() else "present"
        _, code = _unsigned_integer(correction)
        kind = self.corrections.get(code, "unsupported")
        try:
            _parse_conditions(conditions)
            condition_status = "parsed-not-qualified"
        except ValueError:
            condition_status = "malformed"
        key = (scope, ticker, values[10], values[3], values[11], correction, kind,
               conditions, condition_status, identity, size, price)
        if key not in self.groups:
            allocated = sum(len(value.encode("utf-8")) for value in key)
            if (len(self.groups) >= self.limits.maximum_groups
                    or self.key_bytes + allocated > self.limits.maximum_group_key_bytes):
                raise Qt200CensusError("Group allocation exhausted; no complete census")
            self.key_bytes += allocated
        if self.rows >= self.limits.maximum_selected_rows:
            raise Qt200CensusError("Selected-row allocation exhausted; no complete census")
        self.groups[key] += 1
        self.rows += 1
        sequence_status, sequence = _unsigned_integer(values[7])
        if sequence is not None:
            sequence_status = self.sequence.observe(self.ticker_indexes[ticker], sequence, row.raw_line_sha256)
        self.sequence_counts[ticker, sequence_status] += 1
        category = (scope, identity, size, price, kind, condition_status)
        if (len(self.samples) < self.limits.maximum_samples
                and self.sample_categories[category] < self.limits.samples_per_category):
            if self.sample_bytes + len(raw_line) > self.limits.maximum_sample_bytes:
                raise Qt200CensusError("Original-sample byte allocation exhausted")
            self.sample_bytes += len(raw_line)
            self.sample_categories[category] += 1
            self.samples.append(dict(source_row_number=row.source_row_number,
                raw_line_sha256=row.raw_line_sha256, original_values=values,
                raw_line_base64=base64.b64encode(raw_line).decode("ascii"),
                provider_trade_id=None if identity == "blank" else values[4],
                provider_trade_id_raw=values[4], correction_target_reference=None,
                correction_resolution_status="not-assessed", canonicalization_error=row.canonicalization_error,
                membership_scope=scope))


def scan_and_publish_qt200_trade_census_v1(
    *, source_request: Mapping[str, Any], source_date: str,
    qt200_tickers: Sequence[str], alias_tickers: Sequence[str],
    correction_authority: MassiveCorrectionAuthority, output_directory: str | Path,
    limits: CensusLimits, progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Publish two diagnostic files only after exact full-file validation.

    The caller additionally owns scheduler, wall-clock and independent RSS guards.
    Output must be a new absent attempt. Test selections may be small; the real
    pilot wrapper must bind the exact frozen 200-symbol file and six aliases.
    """
    limits.validate()
    correction_authority.validate()
    selected = tuple(qt200_tickers) + tuple(alias_tickers)
    if (set(source_request) != _REQUEST_FIELDS or not qt200_tickers
            or any(isinstance(x, (str, bytes)) for x in (qt200_tickers, alias_tickers))
            or tuple(source_request["tickers"]) != selected or len(set(selected)) != len(selected)
            or date.fromisoformat(source_date).isoformat() != source_date
            or Path(source_request["expected_source_object_key"]).name != source_date + ".csv.gz"):
        raise Qt200CensusError("Exact date, transaction and disjoint ordered membership required")
    root, directory = Path(source_request["root"]), Path(output_directory)
    _directory(directory.parent)
    if (not directory.is_absolute() or directory.resolve() != directory
            or directory.is_relative_to(root) or root.is_relative_to(directory)):
        raise Qt200CensusError("Diagnostic output overlaps the source or is not canonical")
    directory.mkdir(mode=0o700)
    census = _Census(selected, qt200_tickers, correction_authority, limits)
    scan = scan_massive_selected_trade_file_v1(
        **source_request, row_sink=lambda _: None,
        original_line_sink=census.observe, progress_callback=progress_callback,
    )
    baseline = _snapshot(root, scan)
    if (scan.selected_row_count != census.rows or sum(census.groups.values()) != census.rows
            or sum(census.sequence_counts.values()) != census.rows):
        raise Qt200CensusError("Diagnostic counts do not reconcile to complete selected scan")
    source_identity = dict(source_object_key=scan.loaded_source.receipt.source_object_key,
        compressed_sha256=scan.compressed_sha256, source_receipt_sha256=scan.loaded_source.receipt.receipt_sha256,
        receipt_file_sha256=scan.receipt_file_sha256, commit_file_sha256=scan.commit_file_sha256)
    for sample in census.samples:
        sample["record_id"] = semantic_sha256(dict(source_identity=source_identity,
            source_row_number=sample["source_row_number"], raw_line_sha256=sample["raw_line_sha256"]))
    marginal = {name: Counter() for name in GROUP_COLUMNS}
    intersections: Counter[tuple[str, ...]] = Counter()
    groups = []
    for key, count in sorted(census.groups.items()):
        group = dict(zip(GROUP_COLUMNS, key, strict=True))
        for name, value in group.items():
            marginal[name][value] += count
        intersections[key[0], key[9], key[10], key[11], key[6]] += count
        group.update(row_count=count, source_date=source_date, stable_security_id=None,
                     security_identity_status="not-assessed", feed_applicability_status="undocumented",
                     parsed_conditions=None if key[8] == "malformed" else _parse_conditions(key[7]))
        groups.append(group)
    ticker_counts = Counter()
    for key, count in census.groups.items():
        ticker_counts[key[1]] += count
    if dict(ticker_counts) != {ticker: count for ticker, count in scan.selected_ticker_counts if count}:
        raise Qt200CensusError("Ticker census differs from source selection")
    report = dict(schema=SCHEMA, implementation_sha256=SOURCE_SHA256, source_date=source_date,
        source_identity=source_identity, source_scan=asdict(scan),
        ordered_qt200_tickers=list(qt200_tickers), ordered_alias_candidates=list(alias_tickers),
        selection_is_historical_security_identity=False, limits=asdict(limits),
        correction_inventory=asdict(correction_authority), correction_inventory_historical_applicability="not-established",
        source_representation="undocumented-event-stream-versus-corrected-snapshot",
        exported_columns=list(MASSIVE_FLAT_TRADE_COLUMNS),
        explicit_original_reference_columns=[],
        original_reference_mapping="not-documented; sequence_number is not assumed to be an original reference",
        field_presence={name: "supplied-but-historical-meaning-not-qualified" for name in MASSIVE_FLAT_TRADE_COLUMNS},
        source_rows=scan.source_row_count, selected_rows=census.rows,
        marginal_counts={name: dict(sorted(counts.items())) for name, counts in marginal.items()},
        groups=groups, intersection_columns=["membership_scope", "provider_id_status", "size_status", "price_status", "native_candidate_kind"],
        intersections=[dict(values=key, row_count=count) for key, count in sorted(intersections.items())],
        sequence_key="source_date|source_ticker|uint64_sequence; comparison-to-first-occurrence",
        sequence_unique_keys=len(census.sequence.first), sequence_deduplication_applied=False,
        sequence_counts=[dict(source_ticker=key[0], status=key[1], row_count=count)
                         for key, count in sorted(census.sequence_counts.items())],
        sample_policy="first samples_per_category by scope/id/size/price/kind/condition-status up to maximum_samples",
        sample_bytes=census.sample_bytes, samples=census.samples,
        correction_resolution=dict(status="not-assessed", resolved_count=None, unresolved_count=None),
        unresolved_dependency_scopes=[dict(source_date=source_date, source_ticker=ticker,
            membership_scope="qt200" if ticker in census.qt200 else "alias_candidate",
            scope="all-dependent-bars-tape-fills-features-for-this-symbol-day",
            reasons=["historical-identity-unqualified", "export-representation-and-correction-linkage-undocumented"])
            for ticker, count in scan.selected_ticker_counts if count], **CLAIMS)
    report["receipt_sha256"] = semantic_sha256(report)
    # Preflight serialization before any final report appears. No full tick copy.
    if len(canonical_json_file_bytes(report)) + 16384 > limits.output_budget_bytes:
        raise Qt200CensusError("Diagnostic output allocation exhausted")
    file = _publish(directory, "census.json", report, limits.output_budget_bytes - 16384)
    if _snapshot(root, scan) != baseline:
        raise Qt200CensusError("Original source changed before completion")
    completion = dict(schema=SCHEMA + "-completion", implementation_sha256=SOURCE_SHA256,
        files=[file], source_identity=source_identity, source_scan_receipt_sha256=scan.receipt_sha256,
        source_rows=scan.source_row_count, selected_rows=census.rows,
        diagnostic_scan_complete=True, output_budget_bytes=limits.output_budget_bytes, **CLAIMS)
    completion["receipt_sha256"] = semantic_sha256(completion)
    _publish(directory, "COMPLETE.json", completion, limits.output_budget_bytes - file["bytes"])
    return completion
