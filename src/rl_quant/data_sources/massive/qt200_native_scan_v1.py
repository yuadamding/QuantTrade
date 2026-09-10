"""Persist the existing native full-file scan prerequisite without trade copies.

The first pass authenticates the actual compressed transaction and its complete
original byte stream. The second uses the unchanged native scanner with
retain_rows=False. Neither pass resolves permanent identities or promotes an
acquisition, partition, daily-input, or training authority. Original gzip files
remain mandatory replay dependencies. Failed output attempts are never removed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.qt200_historical_message_adapter_v1 import historical_message_category_v1
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
    scan_massive_daily_trade_file_v0,
)
from rl_quant.data_sources.massive.selected_trade_scan_v1 import (
    _identity, _metadata, _SequenceDigest,
    MassiveSelectedTradeFileScanEvidenceV1,
    scan_massive_selected_trade_file_v1,
)
from rl_quant.data_sources.massive.session_calendar import MassiveExchangeSession, MassiveSessionAuthority
from rl_quant.data_sources.massive.source_receipts import (
    _open_parent_directory, LoadedMassiveSourceObject, MassiveSourceCommit, MassiveSourceObjectReceipt,
)
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, file_sha256, semantic_sha256


QT200_NATIVE_SCAN_V1_SCHEMA = "rl-quant.qt200-native-scan-prerequisite-v1"
QT200_NATIVE_SCAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_REQUEST_FIELDS = frozenset((
    "root", "payload_relative_path", "expected_source_object_key", "expected_receipt_file_sha256",
    "expected_commit_file_sha256", "expected_entitlement_receipt_sha256", "expected_compressed_sha256",
    "expected_compressed_bytes", "tickers", "maximum_line_bytes", "verified_at_ms",
))
_CLAIMS = dict(native_acquisition_authority_qualified=False, identity_qualified=False,
               correction_chains_replayed=False, condition_eligibility_qualified=False,
               persisted_partitions_created=False, native_daily_input_authority=False,
               training_ready=False, trade_rows_retained=False)
Progress = Callable[[str, int, int], None]


class Qt200NativeScanError(ValueError):
    """The source prerequisite cannot be published or reconstructed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Qt200NativeScanError(message)


def _directory(path: Path) -> None:
    info = path.lstat()
    _require(path.is_absolute() and path.resolve() == path and stat.S_ISDIR(info.st_mode)
             and info.st_uid == os.getuid(), "directory must be absolute, owned and nonsymlink")


def _native_semantics(scan: MassiveDailyTradeFileScanEvidenceV0) -> dict:
    # A fresh Loaded object has a fresh verification time and runtime identity.
    # All native canonical/provenance, source, parser and support fields remain.
    return {k: v for k, v in scan.unsigned().items() if k != "loaded_source_receipt_sha256"}


def _selection_semantics(scan) -> dict:
    return {k: v for k, v in asdict(scan).items()
            if k not in {"loaded_source", "source_identity", "receipt_sha256"}}


def _snapshot(root: Path, selected) -> tuple:
    _directory(root)
    parent, name = _open_parent_directory(root, selected.loaded_source.payload_relative_path, create=False)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            identity = _identity(os.fstat(fd))
        finally:
            os.close(fd)
        _require(identity == selected.source_identity, "source identity changed between complete passes")
        _, receipt = _metadata(parent, name + ".receipt.json", selected.receipt_file_sha256)
        _, commit = _metadata(parent, name + ".commit.json", selected.commit_file_sha256)
        return identity, receipt, commit
    finally:
        os.close(parent)


def _scan(request: Mapping[str, Any], sessions: MassiveSessionAuthority,
          session: MassiveExchangeSession, corrections: MassiveCorrectionAuthority,
          progress: Progress | None):
    _require(set(request) == _REQUEST_FIELDS, "exact selected-scanner transaction arguments required")
    sessions.validate()
    session.validate()
    corrections.validate()
    _require(session.exchange == "XNYS" and sessions.resolve(
        exchange="XNYS", session_date=session.session_date) == session, "session authority differs")
    selected_hash, second_selected_hash = _SequenceDigest(), _SequenceDigest()
    first_selected_count = second_selected_count = native_count = 0
    error_samples = []
    error_counts: Counter[str] = Counter()
    retrospective_counts: Counter[str] = Counter()

    def first(row):
        nonlocal first_selected_count
        # Massive's NYSE glossary defines 01 as corrected values at the
        # original time, and 12 as ORIGINAL incorrect values at correction
        # time. The legacy new-trade/replacement mapping cannot establish a
        # causal historical stream, even when both rows have provider IDs.
        # Do not modify the source, synthesize a correction key, or call the
        # underlying canonical parser permissive to get past this boundary.
        code = row.original_values[2].lstrip("0") or "0"
        if code in {"1", "12"}:
            retrospective_counts[code] += 1
        if row.extracted_row is not None:
            selected_hash.add(row.extracted_row.receipt_sha256)
            first_selected_count += 1
        else:
            reason = row.canonicalization_error[:80]
            error_counts[reason if reason in error_counts or len(error_counts) < 4 else "other-errors"] += 1
            if len(error_samples) < 4:
                error_samples.append(dict(source_row_number=row.source_row_number,
                                          ticker=row.original_values[0][:32],
                                          raw_line_sha256=row.raw_line_sha256,
                                          message_category=historical_message_category_v1(row.original_values),
                                          error=row.canonicalization_error[:160]))

    selected = scan_massive_selected_trade_file_v1(
        **request, row_sink=first,
        progress_callback=(None if progress is None else lambda rows, chosen: progress("physical-source", rows, chosen)),
    )
    _require(selected.selected_canonicalization_error_count == 0,
             "selected canonicalization errors prohibit a native whole-file completion; "
             f"count={selected.selected_canonicalization_error_count}; "
             f"error_counts={dict(sorted(error_counts.items()))}; "
             f"retrospective_correction_counts={dict(sorted(retrospective_counts.items()))}; "
             f"first_errors={error_samples}")
    _require(not retrospective_counts,
             "historical retrospective correction payloads require a qualified "
             "01/12 orientation, target and revision-availability adapter; "
             f"counts={dict(sorted(retrospective_counts.items()))}; "
             "code 12 must not be replayed as new replacement values")
    root = Path(request["root"])
    baseline = _snapshot(root, selected)
    tickers = frozenset(selected.ordered_tickers)
    codes: Counter[str] = Counter()

    def second(row):
        nonlocal native_count, second_selected_count
        native_count += 1
        _require(row.source_row_number == native_count + 1, "native physical row ordinal differs")
        record = row.canonical_record
        _require(record.correction_code not in {1, 12},
                 "historical retrospective correction payload outside the selected "
                 "panel requires a qualified 01/12 revision-availability adapter")
        codes["null" if record.correction_code is None else str(record.correction_code)] += 1
        if record.ticker in tickers:
            second_selected_hash.add(row.receipt_sha256)
            second_selected_count += 1
        if progress is not None and native_count % 8192 == 0:
            progress("native-canonical", native_count, second_selected_count)

    retained, native = scan_massive_daily_trade_file_v0(
        root=root, loaded_source=selected.loaded_source, session_authority=sessions,
        session=session, correction_authority=corrections, row_sink=second, retain_rows=False,
    )
    native.validate()
    _require(not retained and native_count == native.source_row_count == selected.source_row_count,
             "native scan omitted rows or retained the source in memory")
    _require(first_selected_count == second_selected_count == selected.selected_row_count
             and selected_hash.hexdigest() == second_selected_hash.hexdigest(),
             "native selected rows differ from original-line canonical provenance")
    _require(_snapshot(root, selected) == baseline, "source transaction changed during native scan")
    if progress is not None:
        progress("native-complete", native_count, second_selected_count)
    _require(_snapshot(root, selected) == baseline, "source transaction changed in completion callback")
    return native, selected, selected_hash.hexdigest(), dict(sorted(codes.items())), baseline


def _read(path: Path, expected: str, cap: int) -> tuple[dict, bytes]:
    _directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = _identity(os.fstat(fd))
        _require(stat.S_IMODE(before[7]) == 0o444 and before[8] == 1 and before[2] <= cap,
                 "persisted prerequisite is unsealed, linked or oversized: "
                 f"mode={oct(stat.S_IMODE(before[7]))}, nlink={before[8]}, bytes={before[2]}, cap={cap}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            body = stream.read(cap + 1)
        _require(_identity(os.fstat(fd)) == before and _identity(path.lstat()) == before
                 and len(body) == before[2] and hashlib.sha256(body).hexdigest() == expected,
                 "persisted prerequisite bytes or identity changed")
    finally:
        os.close(fd)
    value = json.loads(body)
    _require(isinstance(value, dict) and canonical_json_file_bytes(value) == body,
             "persisted prerequisite is not canonical JSON")
    return value, body


def _publish(directory: Path, name: str, value: dict, remaining: int) -> dict:
    body = canonical_json_file_bytes(value)
    _require(len(body) <= remaining, "prerequisite output allocation exceeded")
    partial, final = directory / (name + ".partial"), directory / name
    fd = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(body)
            stream.flush()
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        before = _identity(os.fstat(fd))
        _require(_identity(partial.lstat()) == before, "partial publication inode differs")
        os.link(partial, final, follow_symlinks=False)
        # Only remove the newly owned publication hardlink, never old evidence.
        linked = _identity(os.fstat(fd))
        _require(_identity(partial.lstat()) == linked == _identity(final.lstat()) and linked[8] == 2,
                 "publication hardlink identity differs")
    finally:
        os.close(fd)
    # Close BEFORE unlink: NFS can otherwise retain an open-unlinked temporary
    # name (sillyrename), exposing nlink=2 after unlink returns. Fsync alone does
    # not close that handle. Remove only the exact newly owned hardlink, then
    # durably settle the directory and retain the strict nlink=1 replay check.
    _require(_identity(partial.lstat()) == linked == _identity(final.lstat()),
             "publication hardlink changed before closed-handle unlink")
    partial.unlink()
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    digest = hashlib.sha256(body).hexdigest()
    _read(final, digest, len(body))
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return dict(path=name, bytes=len(body), sha256=digest)


def scan_and_publish_qt200_native_trade_file_v1(
    *, source_request: Mapping[str, Any], output_directory: str | Path, output_budget_bytes: int,
    session_authority: MassiveSessionAuthority, session: MassiveExchangeSession,
    correction_authority: MassiveCorrectionAuthority, progress_callback: Progress | None = None,
) -> dict:
    """Create one absent output directory and three sealed metadata files.

    The caller owns wall/RSS/project guards through progress_callback and must
    preserve the original transaction. No trade rows or partitions are stored.
    A failure leaves the absent-completion attempt in place; never retry into it.
    """
    directory = Path(output_directory)
    _directory(directory.parent)
    _require(type(output_budget_bytes) is int and output_budget_bytes > 0,
             "positive explicit metadata allocation required")
    root = Path(source_request["root"])
    _require(directory.is_absolute() and directory.resolve() == directory
             and not directory.is_relative_to(root), "output cannot overlap the original source root")
    directory.mkdir(mode=0o700)
    native, selected, selected_native_hash, codes, baseline = _scan(
        source_request, session_authority, session, correction_authority, progress_callback,
    )
    request = dict(source_request, root=str(root), tickers=list(selected.ordered_tickers))
    budget = min(output_budget_bytes, _MAX_METADATA_BYTES)
    native_file = _publish(directory, "native-scan.json", asdict(native), budget)
    prerequisite = dict(
        schema=QT200_NATIVE_SCAN_V1_SCHEMA, implementation_sha256=QT200_NATIVE_SCAN_V1_SOURCE_SHA256,
        source_request=request, selection_preflight=asdict(selected), native_scan=native_file,
        native_semantic_sha256=semantic_sha256(_native_semantics(native)),
        selection_semantic_sha256=semantic_sha256(_selection_semantics(selected)),
        selected_native_row_inventory_sha256=selected_native_hash,
        all_source_canonical_correction_code_counts=codes,
        retained_native_row_count=0, original_gzip_required_for_replay=True,
        physical_preflight_and_native_scan_completed=True,
        original_source_preserved=True, output_budget_bytes=output_budget_bytes, **_CLAIMS,
    )
    prerequisite["receipt_sha256"] = semantic_sha256(prerequisite)
    proof_file = _publish(directory, "prerequisite.json", prerequisite,
                          budget - native_file["bytes"])
    _require(_snapshot(root, selected) == baseline, "source changed before completion publication")
    completion = dict(schema=QT200_NATIVE_SCAN_V1_SCHEMA + "-completion",
                      implementation_sha256=QT200_NATIVE_SCAN_V1_SOURCE_SHA256,
                      files=[native_file, proof_file], source_rows=native.source_row_count,
                      selected_rows=selected.selected_row_count, output_budget_bytes=output_budget_bytes,
                      original_source_preserved=True, native_scan_prerequisite_complete=True, **_CLAIMS)
    completion["receipt_sha256"] = semantic_sha256(completion)
    _publish(directory, "COMPLETE.json", completion,
             budget - native_file["bytes"] - proof_file["bytes"])
    return completion


def reconstruct_qt200_native_trade_file_v1(
    *, root: str | Path, output_directory: str | Path, expected_completion_sha256: str,
    session_authority: MassiveSessionAuthority, session: MassiveExchangeSession,
    correction_authority: MassiveCorrectionAuthority, verified_at_ms: int,
    progress_callback: Progress | None = None,
) -> MassiveDailyTradeFileScanEvidenceV0:
    """Read-only full-source reconstruction; missing artifacts are never rebuilt.

    Return freshly loaded native evidence. Its loaded-source receipt may differ
    with verification time or mount identity; the original evidence is untouched.
    All other native source/canonical/provenance/support fields must agree.
    """
    directory = Path(output_directory)
    _directory(directory)
    _require(set(path.name for path in directory.iterdir())
             == {"native-scan.json", "prerequisite.json", "COMPLETE.json"}, "incomplete prerequisite file set")
    # The completion itself is small; its bound allocation limits child metadata.
    completion, completion_body = _read(directory / "COMPLETE.json", expected_completion_sha256, 1024 * 1024)
    _require(completion.get("schema") == QT200_NATIVE_SCAN_V1_SCHEMA + "-completion"
             and completion.get("native_scan_prerequisite_complete") is True
             and completion.get("implementation_sha256") == QT200_NATIVE_SCAN_V1_SOURCE_SHA256
             and completion.get("original_source_preserved") is True
             and all(completion.get(k) is v for k, v in _CLAIMS.items()), "completion contract differs")
    allocation = completion.get("output_budget_bytes")
    _require(type(allocation) is int and allocation > 0, "invalid recorded metadata allocation")
    budget = min(allocation, _MAX_METADATA_BYTES)
    unsigned = {k: v for k, v in completion.items() if k != "receipt_sha256"}
    _require(completion["receipt_sha256"] == semantic_sha256(unsigned), "completion semantic digest differs")
    entries = completion["files"]
    _require([entry["path"] for entry in entries] == ["native-scan.json", "prerequisite.json"],
             "prerequisite inventory differs")
    values, consumed = [], len(completion_body)
    for entry in entries:
        value, body = _read(directory / entry["path"], entry["sha256"], budget)
        _require(len(body) == entry["bytes"], "prerequisite inventory size differs")
        consumed += len(body)
        values.append(value)
    _require(consumed <= budget, "persisted metadata exceeds its allocation")
    native_value = dict(values[0])
    for field in ("observed_participant_calendar_dates", "observed_sip_calendar_dates"):
        native_value[field] = tuple(native_value[field])
    stored, prerequisite = MassiveDailyTradeFileScanEvidenceV0(**native_value), values[1]
    stored.validate()
    _require(prerequisite.get("schema") == QT200_NATIVE_SCAN_V1_SCHEMA
             and prerequisite.get("implementation_sha256") == QT200_NATIVE_SCAN_V1_SOURCE_SHA256
             and prerequisite.get("output_budget_bytes") == allocation
             and prerequisite.get("retained_native_row_count") == 0
             and prerequisite.get("original_gzip_required_for_replay") is True
             and prerequisite.get("physical_preflight_and_native_scan_completed") is True
             and prerequisite.get("original_source_preserved") is True
             and all(prerequisite.get(k) is v for k, v in _CLAIMS.items()), "prerequisite contract differs")
    _require(prerequisite["receipt_sha256"] == semantic_sha256({
        k: v for k, v in prerequisite.items() if k != "receipt_sha256"})
        and prerequisite["native_scan"] == entries[0]
        and prerequisite["source_request"]["root"] == str(Path(root))
        and prerequisite["native_semantic_sha256"] == semantic_sha256(_native_semantics(stored)),
        "stored source/native prerequisite crosslink differs")
    # Restore the existing typed validators, rather than trust nested JSON facts.
    selection_value = dict(prerequisite["selection_preflight"])
    loaded_value = dict(selection_value["loaded_source"])
    loaded_value["receipt"] = MassiveSourceObjectReceipt(**loaded_value["receipt"])
    loaded_value["commit"] = MassiveSourceCommit(**loaded_value["commit"])
    selection_value["loaded_source"] = LoadedMassiveSourceObject(**loaded_value)
    for field in ("source_identity", "ordered_tickers"):
        selection_value[field] = tuple(selection_value[field])
    for field in ("selected_ticker_counts", "selected_ticker_canonicalization_error_counts"):
        selection_value[field] = tuple(tuple(pair) for pair in selection_value[field])
    original_selection = MassiveSelectedTradeFileScanEvidenceV1(**selection_value)
    original_selection.validate()
    loaded = original_selection.loaded_source
    _require(prerequisite["selection_semantic_sha256"] == semantic_sha256(_selection_semantics(original_selection))
             and stored.loaded_source_receipt_sha256 == loaded.receipt_sha256
             and stored.source_object_receipt_sha256 == loaded.receipt.receipt_sha256
             and stored.source_commit_receipt_sha256 == loaded.commit.receipt_sha256,
             "stored native scan and physical preflight differ")
    _require(prerequisite["source_request"] == dict(
        root=str(Path(root)), payload_relative_path=loaded.payload_relative_path,
        expected_source_object_key=loaded.receipt.source_object_key,
        expected_receipt_file_sha256=original_selection.receipt_file_sha256,
        expected_commit_file_sha256=original_selection.commit_file_sha256,
        expected_entitlement_receipt_sha256=loaded.receipt.entitlement_receipt_sha256,
        expected_compressed_sha256=original_selection.compressed_sha256,
        expected_compressed_bytes=original_selection.compressed_bytes,
        tickers=list(original_selection.ordered_tickers), maximum_line_bytes=original_selection.maximum_line_bytes,
        verified_at_ms=loaded.verified_at_ms,
    ), "stored original request differs from the physical preflight")
    request = dict(prerequisite["source_request"], root=Path(root), verified_at_ms=verified_at_ms)
    fresh, selected, selected_native_hash, codes, _ = _scan(
        request, session_authority, session, correction_authority, progress_callback,
    )
    _require(_native_semantics(fresh) == _native_semantics(stored)
             and semantic_sha256(_selection_semantics(selected)) == prerequisite["selection_semantic_sha256"]
             and selected_native_hash == prerequisite["selected_native_row_inventory_sha256"]
             and codes == prerequisite["all_source_canonical_correction_code_counts"]
             and fresh.source_row_count == completion["source_rows"]
             and selected.selected_row_count == completion["selected_rows"],
             "fresh reconstruction differs from persisted original-source facts")
    return fresh
