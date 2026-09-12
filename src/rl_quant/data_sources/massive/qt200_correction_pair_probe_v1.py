"""Bounded, read-only observations of historical correction-code neighbours.

This probe consumes a hash-bound *lossless* raw-string Parquet, not economic
trades. Matching ticker/sequence values are an investigative grouping only:
they are neither provider transaction IDs nor authenticated correction links.
In particular, codes 1 and 12 are retained without assuming which row contains
the original or corrected values. Original CSV quoting/line hashes cannot be
reconstructed from Parquet strings and are deliberately not claimed here.
"""

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, BinaryIO, Iterator


COLUMNS = (
    "ticker", "conditions", "correction", "exchange", "id",
    "participant_timestamp", "price", "sequence_number", "sip_timestamp",
    "size", "tape", "trf_id", "trf_timestamp",
)
NAMES = (*COLUMNS, "source_row_number")
CANDIDATE_CODE_FAMILIES = {
    "correction": (1, 12), "cancellation": (8, 10), "error": (7, 11),
}
CANDIDATE_CODES = frozenset(code for pair in CANDIDATE_CODE_FAMILIES.values() for code in pair)
CLAIMS = {
    "source_data_qualified": False,
    "training_ready": False,
    "correction_replay_qualified": False,
    "correction_links_inferred": False,
    "sequence_used_as_provider_trade_id": False,
    "original_csv_line_hash_reconstructed": False,
    "source_writes": False,
}


@dataclass(frozen=True)
class CorrectionPairProbeLimits:
    maximum_rows: int = 8_000_000
    maximum_keys: int = 4_096
    maximum_candidate_rows: int = 16_384
    maximum_candidate_bytes: int = 16 * 1024 * 1024
    maximum_report_bytes: int = 32 * 1024 * 1024
    batch_rows: int = 65_536

    def validate(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Probe allocations must be positive integers")
        if self.batch_rows > 65_536:
            raise ValueError("Probe batch allocation exceeds 65,536 rows")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA-256")


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode, value.st_nlink, value.st_uid, value.st_gid)


def _path_metadata(path: Path, *, metadata_only: bool = False) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Probe input must be an absolute, non-traversing path")
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("Probe input has a symlink ancestor")
    value = path.lstat()
    regular = (stat.S_ISREG(value.st_mode) and value.st_nlink == 1
               and value.st_uid == os.geteuid())
    # The legacy pilot published its JSON completion as owner-mode 0700.
    # Never chmod historical evidence. That metadata is opened O_RDONLY only,
    # never executed, and remains exact-hash/FD/path bound before and after.
    # This exception does not apply to the Parquet or grant source authority.
    permissions_ok = (stat.S_IMODE(value.st_mode) in (0o400, 0o444, 0o600, 0o700)
                      if metadata_only else not value.st_mode & 0o222)
    if not regular or not permissions_ok:
        raise ValueError("Probe requires an owned regular single-link input with approved permissions")
    return value


def _hash_stream(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(8 * 1024 * 1024):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


@contextmanager
def _bound_file(path: Path, expected_sha256: str, expected_bytes: int,
                *, metadata_only: bool = False) -> Iterator[BinaryIO]:
    _sha(expected_sha256)
    before = _path_metadata(path, metadata_only=metadata_only)
    if before.st_size != expected_bytes:
        raise ValueError("Probe input byte count differs")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError("Probe input changed before opening")
        if _hash_stream(stream) != expected_sha256:
            raise ValueError("Probe input SHA-256 differs")
        yield stream
        if (_hash_stream(stream) != expected_sha256
                or _identity(os.fstat(stream.fileno())) != _identity(before)
                or _identity(_path_metadata(path, metadata_only=metadata_only)) != _identity(before)):
            raise ValueError("Probe input changed during read-only inspection")


def _integer(raw: str) -> int | None:
    # Preserve the original lexeme separately; parsing selects candidates,
    # never manufactures a trade ID or normalizes the grouping key. Bound the
    # integer conversion so malformed very long fields cannot abort the census.
    normalized = raw.lstrip("0") or "0"
    if not raw or not raw.isascii() or not raw.isdecimal() or len(normalized) > 20:
        return None
    value = int(normalized)
    return value if value < 2**64 else None


def _group_observation(key: tuple[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    codes = [row["original_fields"]["correction"] for row in rows]
    parsed_codes = [row["correction_code"] for row in rows]
    status = "ambiguous-candidate-group"
    if not key[0] or _integer(key[1]) is None:
        status = "missing-or-malformed-group-key"
    elif len(rows) == 1:
        status = "unmatched-candidate"
    elif len(rows) == 2 and any(set(parsed_codes) == set(pair) for pair in CANDIDATE_CODE_FAMILIES.values()):
        status = "observed-code-pair-not-a-qualified-link"
    comparisons = []
    # Only a two-row comparison has a unique observed counterpart. More rows
    # remain visible but do not trigger an arbitrary matching algorithm.
    if len(rows) == 2:
        left, right = (row["original_fields"] for row in rows)
        deltas = {}
        for name in ("participant_timestamp", "sip_timestamp", "trf_timestamp"):
            first, second = _integer(left[name]), _integer(right[name])
            deltas[name] = None if first is None or second is None else second - first
        comparisons.append({
            "left_source_row_number": rows[0]["source_row_number"],
            "right_source_row_number": rows[1]["source_row_number"],
            "code_order_in_source": codes,
            "parsed_code_order_in_source": parsed_codes,
            "different_original_fields": [name for name in COLUMNS if left[name] != right[name]],
            "right_minus_left_timestamp_ns": deltas,
            "same_provider_trade_id_text": left["id"] == right["id"],
            "nonblank_same_provider_trade_id": bool(left["id"]) and left["id"] == right["id"],
            "economic_role_assignment": "not-established",
        })
    return {
        "ticker_raw": key[0], "sequence_number_raw": key[1],
        "status": status, "row_count": len(rows), "code_counts": dict(sorted(Counter(codes).items())),
        "rows": rows, "comparisons": comparisons,
        "correction_target_reference": None,
    }


def probe_qt200_correction_pairs_v1(
    *, parquet_path: Path, expected_sha256: str, expected_bytes: int,
    expected_rows: int, source_date: str,
    expected_completion_path: Path | None = None,
    expected_completion_sha256: str | None = None,
    limits: CorrectionPairProbeLimits = CorrectionPairProbeLimits(),
) -> dict[str, Any]:
    """Return bounded diagnostic evidence; no files are created or modified.

    Pass 1 inventories every code and retains candidate raw ticker/sequence
    keys. Pass 2 inspects every row and retains *all* rows sharing those keys,
    including code 0 and unknown codes, so collisions cannot be hidden by the
    initial filter. The caller owns create-only report publication and binding
    this extraction transaction to the separately preserved gzip source.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    limits.validate()
    if (type(expected_bytes) is not int or expected_bytes <= 0
            or type(expected_rows) is not int or not 0 < expected_rows <= limits.maximum_rows):
        raise ValueError("Probe expected rows/bytes exceed the declared allocation")
    if type(source_date) is not str or date.fromisoformat(source_date).isoformat() != source_date:
        raise ValueError("Probe source date must be an exact ISO date")
    if (expected_completion_path is None) != (expected_completion_sha256 is None):
        raise ValueError("Completion path and SHA-256 must be supplied together")
    completion = None
    if expected_completion_path is not None:
        assert expected_completion_sha256 is not None
        meta = _path_metadata(expected_completion_path, metadata_only=True)
        if meta.st_size > 1024 * 1024:
            raise ValueError("Completion metadata exceeds allocation")
        with _bound_file(expected_completion_path, expected_completion_sha256, meta.st_size,
                         metadata_only=True) as stream:
            completion = json.loads(stream.read())
        if not isinstance(completion, dict):
            raise ValueError("Completion metadata must be an object")
        body = {key: value for key, value in completion.items() if key != "receipt_sha256"}
        if (completion.get("schema") != "quanttrade-qt200-full-history-extraction-pilot-completion-v1"
                or completion.get("receipt_sha256") != hashlib.sha256(_canonical(body)).hexdigest()
                or completion.get("output_sha256") != expected_sha256
                or completion.get("output_bytes") != expected_bytes
                or completion.get("selected_row_count") != expected_rows
                or completion.get("source_columns") != list(COLUMNS)
                or completion.get("output_file") != "trades.parquet"
                or completion.get("source_object_key") != (
                    f"us_stocks_sip/trades_v1/{source_date[:4]}/{source_date[5:7]}/{source_date}.csv.gz")
                or completion.get("ordered_values_reopen_verified") is not True
                or completion.get("full_gzip_integrity_verified") is not True
                or completion.get("training_ready") is not False):
            raise ValueError("Completion extraction binding or semantic receipt differs")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    code_counts: Counter[str] = Counter()
    key_bytes = 0
    candidate_code_rows = 0
    retained_bytes = 0
    retained_count = 0
    with _bound_file(parquet_path, expected_sha256, expected_bytes) as stream:
        parquet = pq.ParquetFile(stream)
        schema = parquet.schema_arrow
        if (tuple(schema.names) != NAMES
                or any(schema.field(i).type != (pa.string() if i < 13 else pa.uint64()) for i in range(14))):
            raise ValueError("Probe requires the exact original 13-string/uint64-ordinal schema")
        if parquet.metadata.num_rows != expected_rows:
            raise ValueError("Probe footer row count differs")
        schema_sha = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
        first_pass_rows = 0
        previous_ordinal = 1
        for batch in parquet.iter_batches(batch_size=limits.batch_rows,
                columns=["ticker", "correction", "sequence_number", "source_row_number"], use_threads=False):
            batch.validate(full=True)
            if any(column.null_count for column in batch.columns):
                raise ValueError("Probe original selected fields must be nonnull")
            for ticker, code, sequence, ordinal in zip(*(column.to_pylist() for column in batch.columns)):
                if ordinal <= previous_ordinal:
                    raise ValueError("Probe source ordinals are not strictly increasing")
                previous_ordinal = ordinal
                first_pass_rows += 1
                code_counts[code] += 1
                if len(code_counts) > 256:
                    raise ValueError("Probe raw correction-code allocation exceeded")
                if _integer(code) not in CANDIDATE_CODES:
                    continue
                candidate_code_rows += 1
                key = (ticker, sequence)
                if key not in groups:
                    key_bytes += len(_canonical(key))
                    if len(groups) >= limits.maximum_keys or key_bytes > limits.maximum_candidate_bytes:
                        raise ValueError("Probe candidate-key allocation exceeded")
                    groups[key] = []
        if first_pass_rows != expected_rows:
            raise ValueError("Probe first-pass row count differs")

        second_pass_rows = 0
        # Filter both marginal key components in Arrow before converting rows
        # to Python objects, then check the exact tuple. Do not manufacture a
        # Cartesian-product link or materialize millions of ordinary rows.
        ticker_values = pa.array(sorted({key[0] for key in groups}), type=pa.string())
        sequence_values = pa.array(sorted({key[1] for key in groups}), type=pa.string())
        for batch in parquet.iter_batches(batch_size=limits.batch_rows, use_threads=False):
            batch.validate(full=True)
            if any(column.null_count for column in batch.columns):
                raise ValueError("Probe original selected fields must be nonnull")
            second_pass_rows += batch.num_rows
            selected = batch.filter(pc.and_(
                pc.is_in(batch.column(0), value_set=ticker_values),
                pc.is_in(batch.column(7), value_set=sequence_values)))
            for raw in selected.to_pylist():
                key = (raw["ticker"], raw["sequence_number"])
                if key not in groups:
                    continue
                original = {name: raw[name] for name in COLUMNS}
                content_sha = hashlib.sha256(_canonical(original)).hexdigest()
                row = {
                    "source_row_number": raw["source_row_number"],
                    "source_row_number_semantics": "CSV record ordinal; header=1; first data=2",
                    "original_fields": original,
                    "original_fields_sha256": content_sha,
                    "correction_code": _integer(original["correction"]),
                    "provider_trade_id": original["id"] or None,
                    "provider_trade_id_raw": original["id"],
                    "correction_target_reference": None,
                    "raw_line_sha256": None,
                }
                retained_count += 1
                retained_bytes += len(_canonical(row))
                if retained_count > limits.maximum_candidate_rows or retained_bytes > limits.maximum_candidate_bytes:
                    raise ValueError("Probe retained-candidate allocation exceeded")
                groups[key].append(row)
        if second_pass_rows != expected_rows or any(not rows for rows in groups.values()):
            raise ValueError("Probe second-pass row/key coverage differs")

    observations = [_group_observation(key, rows) for key, rows in sorted(groups.items())]
    report = {
        "schema": "quanttrade-qt200-correction-pair-probe-v1",
        "source_date": source_date, "input_parquet_sha256": expected_sha256,
        "input_parquet_bytes": expected_bytes, "input_schema_sha256": schema_sha,
        "completion_file_sha256": expected_completion_sha256,
        "completion_metadata_reloaded": completion is not None,
        "original_source_compressed_sha256": None if completion is None else completion.get("source_compressed_sha256"),
        "original_source_receipt_sha256": None if completion is None else completion.get("source_object_receipt_sha256"),
        "source_rows": expected_rows, "passes": 2,
        "raw_correction_code_counts": dict(sorted(code_counts.items())),
        "candidate_code_families": CANDIDATE_CODE_FAMILIES,
        "candidate_selection": "parsed-uint64-code;original-code-and-group-key-lexemes-preserved",
        "candidate_code_rows": candidate_code_rows,
        "candidate_key_count": len(groups), "retained_rows": retained_count,
        "retained_original_row_bytes": retained_bytes,
        "status_counts": dict(sorted(Counter(row["status"] for row in observations).items())),
        "grouping_semantics": "raw ticker/sequence equality for observation only, not a trade identity",
        "source_provenance": "hash-bound lossless Parquet plus original CSV record ordinal",
        "correction_resolution_status": "not-assessed; historical provider mapping required",
        "groups": observations, "limits": asdict(limits), **CLAIMS,
    }
    if len(_canonical(report)) > limits.maximum_report_bytes:
        raise ValueError("Probe report allocation exceeded")
    report["receipt_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report
