"""Official FF5+Momentum retrieval and materialization for 2026-YTD.

The official path starts with a package-owned HTTPS retrieval under Python's
default TLS verification and produces a content-bound receipt.  The parser
accepts only that typed evidence, requires complete exact-date coverage of the
unchanged score window, converts selected source percentages to decimal
returns, and publishes a replayable canonical artifact.

Current Kenneth French containers may contain rows after the frozen
2026-06-23 cutoff.  Their container bytes remain bound for reproducibility,
but return values are parsed only for exact frozen score dates.  Later rows
are counted as unused metadata and can never enter evaluator arrays.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from rl_quant.evaluation.top2000_m03r_v7_2026 import (
    TOP2000_M03R_V7_2026_FACTOR_NAMES,
    Top2000M03RV72026FactorManifest,
    build_top2000_m03r_v7_2026_factor_manifest,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
)

TOP2000_M03R_V7_2026_FACTOR_DATA_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-ff5-momentum-data-v1"
)
TOP2000_M03R_V7_2026_FACTOR_SOURCE_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-factor-source-v1"
)
TOP2000_M03R_V7_2026_FACTOR_COVERAGE_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-factor-coverage-v1"
)
TOP2000_M03R_V7_2026_FACTOR_ARRAY_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-factor-arrays-v1"
)
TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-official-factor-https-retrieval-v1"
)
TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_METHOD = (
    "package-owned-https-default-tls-v1"
)
TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER = (
    "F-F_Research_Data_5_Factors_2x3_daily.CSV"
)
TOP2000_M03R_V7_2026_MOMENTUM_MEMBER = "F-F_Momentum_Factor_daily.CSV"
TOP2000_M03R_V7_2026_FIVE_FACTOR_ARCHIVE = (
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
TOP2000_M03R_V7_2026_MOMENTUM_ARCHIVE = "F-F_Momentum_Factor_daily_CSV.zip"
TOP2000_M03R_V7_2026_MAX_FACTOR_ARCHIVE_BYTES = 64 * 1024 * 1024
TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_USER_AGENT = (
    "QuantTrade-M03R-v7-2026-factor-retrieval/1"
)


class Top2000M03RV72026FactorDataError(ValueError):
    """The official factor source, score coverage, or array identity drifted."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026FactorDataError(
            "factor evidence is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype == object:
        raw = _canonical_json(array.tolist())
        dtype = "canonical-utf8-string"
    else:
        normalized = np.ascontiguousarray(array, dtype=">f8")
        raw = normalized.tobytes(order="C")
        dtype = "big-endian-float64"
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {"name": name, "shape": list(array.shape), "normalized_dtype": dtype}
        )
    )
    digest.update(raw)
    return digest.hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026FactorDataError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _validate_embedded_receipt(
    payload: dict[str, Any],
    *,
    expected_schema: str,
    label: str,
) -> None:
    if payload.get("schema") != expected_schema:
        raise Top2000M03RV72026FactorDataError(f"{label} schema drifted")
    unsigned = dict(payload)
    observed = unsigned.pop("receipt_sha256", None)
    if _require_digest(f"{label} receipt_sha256", observed) != _sha256(unsigned):
        raise Top2000M03RV72026FactorDataError(f"{label} hash drifted")


def _canonical_score_dates(values: Sequence[str]) -> tuple[str, ...]:
    dates = tuple(values)
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
    if (
        len(dates) <= 30
        or len(set(dates)) != len(dates)
        or tuple(sorted(dates)) != dates
        or dates[0] != contract.window.first_scored_date
        or dates[-1] != contract.window.last_scored_date
        or any(not isinstance(value, str) or len(value) != 10 for value in dates)
    ):
        raise Top2000M03RV72026FactorDataError(
            "factor score dates must be the complete frozen 2026-YTD chronology"
        )
    return dates


def _read_zip_member_bytes(raw_zip: bytes, expected_member: str) -> tuple[bytes, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
            members = tuple(archive.infolist())
            if (
                len(members) != 1
                or members[0].is_dir()
                or members[0].filename != expected_member
                or PurePosixPath(members[0].filename).name != expected_member
            ):
                raise Top2000M03RV72026FactorDataError(
                    "factor ZIP member inventory drifted"
                )
            raw = archive.read(members[0])
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise Top2000M03RV72026FactorDataError(
            "factor archive cannot be read"
        ) from exc
    return raw, hashlib.sha256(raw).hexdigest()


def _read_zip_member(path: Path, expected_member: str) -> tuple[bytes, str, str]:
    if not path.is_file() or path.is_symlink():
        raise Top2000M03RV72026FactorDataError(
            f"factor archive is absent or unsafe: {path}"
        )
    try:
        raw_zip = path.read_bytes()
    except OSError as exc:
        raise Top2000M03RV72026FactorDataError(
            "factor archive cannot be read"
        ) from exc
    raw, member_sha256 = _read_zip_member_bytes(raw_zip, expected_member)
    return raw, hashlib.sha256(raw_zip).hexdigest(), member_sha256


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _download_official_archive(url: str) -> tuple[bytes, int, str]:
    if urlsplit(url).scheme != "https":
        raise Top2000M03RV72026FactorDataError(
            "official factor retrieval requires HTTPS"
        )
    request = Request(
        url,
        headers={
            "Accept": "application/zip,application/octet-stream",
            "User-Agent": TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_USER_AGENT,
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            status = int(response.status)
            final_url = str(response.geturl())
            raw = response.read(TOP2000_M03R_V7_2026_MAX_FACTOR_ARCHIVE_BYTES + 1)
    except (OSError, URLError, ValueError) as exc:
        raise Top2000M03RV72026FactorDataError(
            "official factor HTTPS retrieval failed"
        ) from exc
    if status != 200:
        raise Top2000M03RV72026FactorDataError(
            "official factor HTTPS retrieval did not return status 200"
        )
    if final_url != url:
        raise Top2000M03RV72026FactorDataError(
            "official factor HTTPS retrieval redirected away from the frozen URL"
        )
    if not raw or len(raw) > TOP2000_M03R_V7_2026_MAX_FACTOR_ARCHIVE_BYTES:
        raise Top2000M03RV72026FactorDataError(
            "official factor archive is empty or exceeds the frozen byte limit"
        )
    return raw, status, final_url


def _write_immutable_bytes(path: Path, raw: bytes, *, label: str) -> str:
    if path.exists():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == raw:
            return hashlib.sha256(raw).hexdigest()
        raise Top2000M03RV72026FactorDataError(
            f"refusing to overwrite immutable {label} {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as sink:
        sink.write(raw)
        sink.flush()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026OfficialFactorRetrieval:
    """Replayable evidence from the package-owned two-URL HTTPS retrieval."""

    frozen_plan_file_sha256: str
    frozen_plan_receipt_sha256: str
    retrieved_at_utc: str
    five_factor_url: str
    momentum_url: str
    five_factor_response_url: str
    momentum_response_url: str
    five_factor_http_status: int
    momentum_http_status: int
    five_factor_archive_path: str
    momentum_archive_path: str
    five_factor_archive_sha256: str
    momentum_archive_sha256: str
    five_factor_archive_bytes: int
    momentum_archive_bytes: int
    five_factor_member: str
    momentum_member: str
    five_factor_member_sha256: str
    momentum_member_sha256: str
    receipt_sha256: str
    schema: str = TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_RECEIPT_SCHEMA
    retrieval_method: str = TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_METHOD
    user_agent: str = TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_USER_AGENT
    default_tls_verification: bool = True
    redirects_followed: bool = False
    caller_staged_archives: bool = False
    official_source_verified: bool = True
    factor_archives_opened: bool = True
    source_containers_may_include_unused_post_end_rows: bool = True
    extraction_deferred_to_exact_frozen_score_dates: bool = True
    development_only: bool = True
    retrospective_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
        for name in (
            "frozen_plan_file_sha256",
            "frozen_plan_receipt_sha256",
            "five_factor_archive_sha256",
            "momentum_archive_sha256",
            "five_factor_member_sha256",
            "momentum_member_sha256",
            "receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        try:
            retrieved_at = dt.datetime.fromisoformat(self.retrieved_at_utc)
        except (TypeError, ValueError) as exc:
            raise Top2000M03RV72026FactorDataError(
                "factor retrieval timestamp must be UTC to whole seconds"
            ) from exc
        if (
            self.schema
            != TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_RECEIPT_SCHEMA
            or retrieved_at.tzinfo != dt.UTC
            or retrieved_at.microsecond != 0
            or self.retrieved_at_utc != retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            or self.retrieval_method != contract.official_source_transport
            or self.user_agent
            != TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_USER_AGENT
            or self.five_factor_url != contract.five_factor_download_url
            or self.momentum_url != contract.momentum_download_url
            or self.five_factor_response_url != self.five_factor_url
            or self.momentum_response_url != self.momentum_url
            or self.five_factor_http_status != 200
            or self.momentum_http_status != 200
            or self.five_factor_member
            != TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER
            or self.momentum_member != TOP2000_M03R_V7_2026_MOMENTUM_MEMBER
            or Path(self.five_factor_archive_path).name
            != TOP2000_M03R_V7_2026_FIVE_FACTOR_ARCHIVE
            or Path(self.momentum_archive_path).name
            != TOP2000_M03R_V7_2026_MOMENTUM_ARCHIVE
            or type(self.five_factor_archive_bytes) is not int
            or type(self.momentum_archive_bytes) is not int
            or not 0
            < self.five_factor_archive_bytes
            <= TOP2000_M03R_V7_2026_MAX_FACTOR_ARCHIVE_BYTES
            or not 0
            < self.momentum_archive_bytes
            <= TOP2000_M03R_V7_2026_MAX_FACTOR_ARCHIVE_BYTES
            or not self.default_tls_verification
            or self.redirects_followed
            or self.caller_staged_archives
            or not self.official_source_verified
            or not self.factor_archives_opened
            or not self.source_containers_may_include_unused_post_end_rows
            or not self.extraction_deferred_to_exact_frozen_score_dates
            or not self.development_only
            or not self.retrospective_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026FactorDataError(
                "official factor retrieval evidence drifted"
            )
        self.validate_archives()
        unsigned = asdict(self)
        unsigned.pop("receipt_sha256")
        if self.receipt_sha256 != _sha256(unsigned):
            raise Top2000M03RV72026FactorDataError(
                "official factor retrieval receipt hash drifted"
            )

    def validate_archives(self) -> None:
        """Re-hash the exact immutable containers and their sole CSV members."""

        five_raw, five_archive_sha, five_member_sha = _read_zip_member(
            Path(self.five_factor_archive_path), self.five_factor_member
        )
        momentum_raw, momentum_archive_sha, momentum_member_sha = _read_zip_member(
            Path(self.momentum_archive_path), self.momentum_member
        )
        if (
            five_archive_sha != self.five_factor_archive_sha256
            or momentum_archive_sha != self.momentum_archive_sha256
            or len(Path(self.five_factor_archive_path).read_bytes())
            != self.five_factor_archive_bytes
            or len(Path(self.momentum_archive_path).read_bytes())
            != self.momentum_archive_bytes
            or hashlib.sha256(five_raw).hexdigest()
            != self.five_factor_member_sha256
            or hashlib.sha256(momentum_raw).hexdigest()
            != self.momentum_member_sha256
            or five_member_sha != self.five_factor_member_sha256
            or momentum_member_sha != self.momentum_member_sha256
        ):
            raise Top2000M03RV72026FactorDataError(
                "retrieved factor archive or member hash drifted"
            )


def load_top2000_m03r_v7_2026_official_factor_retrieval(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Top2000M03RV72026OfficialFactorRetrieval:
    """Load package-owned HTTPS retrieval evidence and re-hash both archives."""

    _require_digest("expected_file_sha256", expected_file_sha256)
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise Top2000M03RV72026FactorDataError(
            "factor retrieval receipt must be a regular non-symlink file"
        )
    if _file_sha256(source) != expected_file_sha256:
        raise Top2000M03RV72026FactorDataError(
            "factor retrieval receipt file SHA-256 drifted"
        )
    try:
        payload = json.loads(source.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV72026FactorDataError(
            "factor retrieval receipt cannot be read"
        ) from exc
    expected_keys = {field.name for field in fields(Top2000M03RV72026OfficialFactorRetrieval)}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise Top2000M03RV72026FactorDataError(
            "factor retrieval receipt fields drifted"
        )
    try:
        return Top2000M03RV72026OfficialFactorRetrieval(**payload)
    except TypeError as exc:
        raise Top2000M03RV72026FactorDataError(
            "factor retrieval receipt cannot be reconstructed"
        ) from exc


def retrieve_top2000_m03r_v7_2026_official_factor_archives(
    *,
    output_directory: str | Path,
    output_receipt_path: str | Path,
    frozen_plan_file_sha256: str,
    frozen_plan_receipt_sha256: str,
) -> tuple[Top2000M03RV72026OfficialFactorRetrieval, str]:
    """Retrieve only the two frozen official URLs using default TLS checks."""

    _require_digest("frozen_plan_file_sha256", frozen_plan_file_sha256)
    _require_digest("frozen_plan_receipt_sha256", frozen_plan_receipt_sha256)
    directory = Path(output_directory)
    receipt_path = Path(output_receipt_path)
    if receipt_path.exists():
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise Top2000M03RV72026FactorDataError(
                "factor retrieval receipt path is unsafe"
            )
        receipt_file_sha256 = _file_sha256(receipt_path)
        existing = load_top2000_m03r_v7_2026_official_factor_retrieval(
            receipt_path,
            expected_file_sha256=receipt_file_sha256,
        )
        if (
            existing.frozen_plan_file_sha256 != frozen_plan_file_sha256
            or existing.frozen_plan_receipt_sha256 != frozen_plan_receipt_sha256
            or Path(existing.five_factor_archive_path).parent != directory
            or Path(existing.momentum_archive_path).parent != directory
        ):
            raise Top2000M03RV72026FactorDataError(
                "existing factor retrieval does not bind the requested frozen plan"
            )
        return existing, receipt_file_sha256
    if directory.exists() and (not directory.is_dir() or directory.is_symlink()):
        raise Top2000M03RV72026FactorDataError(
            "factor retrieval output directory is unsafe"
        )
    directory.mkdir(parents=True, exist_ok=True)
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
    five_zip, five_status, five_response_url = _download_official_archive(
        contract.five_factor_download_url
    )
    momentum_zip, momentum_status, momentum_response_url = (
        _download_official_archive(contract.momentum_download_url)
    )
    _five_member, five_member_sha = _read_zip_member_bytes(
        five_zip, TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER
    )
    _momentum_member, momentum_member_sha = _read_zip_member_bytes(
        momentum_zip, TOP2000_M03R_V7_2026_MOMENTUM_MEMBER
    )
    five_path = directory / TOP2000_M03R_V7_2026_FIVE_FACTOR_ARCHIVE
    momentum_path = directory / TOP2000_M03R_V7_2026_MOMENTUM_ARCHIVE
    five_archive_sha = _write_immutable_bytes(
        five_path, five_zip, label="five-factor archive"
    )
    momentum_archive_sha = _write_immutable_bytes(
        momentum_path, momentum_zip, label="momentum archive"
    )
    evidence_fields: dict[str, Any] = {
        "frozen_plan_file_sha256": frozen_plan_file_sha256,
        "frozen_plan_receipt_sha256": frozen_plan_receipt_sha256,
        "retrieved_at_utc": _utc_now(),
        "five_factor_url": contract.five_factor_download_url,
        "momentum_url": contract.momentum_download_url,
        "five_factor_response_url": five_response_url,
        "momentum_response_url": momentum_response_url,
        "five_factor_http_status": five_status,
        "momentum_http_status": momentum_status,
        "five_factor_archive_path": str(five_path),
        "momentum_archive_path": str(momentum_path),
        "five_factor_archive_sha256": five_archive_sha,
        "momentum_archive_sha256": momentum_archive_sha,
        "five_factor_archive_bytes": len(five_zip),
        "momentum_archive_bytes": len(momentum_zip),
        "five_factor_member": TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
        "momentum_member": TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
        "five_factor_member_sha256": five_member_sha,
        "momentum_member_sha256": momentum_member_sha,
        "schema": TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_RECEIPT_SCHEMA,
        "retrieval_method": TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_METHOD,
        "user_agent": TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_USER_AGENT,
        "default_tls_verification": True,
        "redirects_followed": False,
        "caller_staged_archives": False,
        "official_source_verified": True,
        "factor_archives_opened": True,
        "source_containers_may_include_unused_post_end_rows": True,
        "extraction_deferred_to_exact_frozen_score_dates": True,
        "development_only": True,
        "retrospective_only": True,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
    }
    evidence = Top2000M03RV72026OfficialFactorRetrieval(
        **evidence_fields,
        receipt_sha256=_sha256(evidence_fields),
    )
    receipt_file_sha256 = _write_immutable_bytes(
        receipt_path,
        _canonical_json(asdict(evidence)),
        label="factor retrieval receipt",
    )
    return evidence, receipt_file_sha256


@dataclass(frozen=True, slots=True)
class _ParsedDailyCSV:
    selected_returns: dict[str, tuple[float, ...]]
    first_source_date: str
    last_source_date: str
    source_daily_row_count: int
    unselected_source_row_count: int
    unused_post_end_source_row_count: int


def _parse_daily_csv(
    raw: bytes,
    *,
    expected_columns: tuple[str, ...],
    selected_dates: frozenset[str],
    last_score_date: str,
) -> _ParsedDailyCSV:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise Top2000M03RV72026FactorDataError(
            "factor CSV must be UTF-8/ASCII"
        ) from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    header_index: int | None = None
    for index, row in enumerate(rows):
        normalized = tuple(value.strip() for value in row)
        if normalized and normalized[0] == "" and normalized[1:] == expected_columns:
            header_index = index
            break
    if header_index is None:
        raise Top2000M03RV72026FactorDataError(
            "factor CSV header or column order drifted"
        )
    selected: dict[str, tuple[float, ...]] = {}
    first_source_date: str | None = None
    last_source_date: str | None = None
    source_daily_row_count = 0
    unused_post_end_source_row_count = 0
    daily_rows_started = False
    for row in rows[header_index + 1 :]:
        normalized = tuple(value.strip() for value in row)
        if not normalized or not normalized[0].isdigit() or len(normalized[0]) != 8:
            if daily_rows_started:
                break
            continue
        daily_rows_started = True
        if len(normalized) != len(expected_columns) + 1:
            raise Top2000M03RV72026FactorDataError(
                "factor CSV data width drifted"
            )
        try:
            source_date = dt.date.fromisoformat(
                f"{normalized[0][:4]}-{normalized[0][4:6]}-{normalized[0][6:]}"
            )
        except ValueError as exc:
            raise Top2000M03RV72026FactorDataError(
                "factor CSV contains an invalid daily date"
            ) from exc
        date_value = source_date.isoformat()
        if last_source_date is not None and date_value <= last_source_date:
            raise Top2000M03RV72026FactorDataError(
                "factor CSV daily dates are duplicate or out of order"
            )
        if first_source_date is None:
            first_source_date = date_value
        last_source_date = date_value
        source_daily_row_count += 1
        if date_value > last_score_date:
            unused_post_end_source_row_count += 1
        if date_value not in selected_dates:
            continue
        try:
            values = tuple(float(value) / 100.0 for value in normalized[1:])
        except ValueError as exc:
            raise Top2000M03RV72026FactorDataError(
                "selected factor CSV row contains a nonnumeric return"
            ) from exc
        if any(not math.isfinite(value) or abs(value) >= 0.9 for value in values):
            raise Top2000M03RV72026FactorDataError(
                "selected factor CSV row contains missing sentinels or "
                "implausible daily values"
            )
        selected[date_value] = values
    if first_source_date is None or last_source_date is None:
        raise Top2000M03RV72026FactorDataError("factor CSV contains no daily data")
    return _ParsedDailyCSV(
        selected_returns=selected,
        first_source_date=first_source_date,
        last_source_date=last_source_date,
        source_daily_row_count=source_daily_row_count,
        unselected_source_row_count=source_daily_row_count - len(selected),
        unused_post_end_source_row_count=unused_post_end_source_row_count,
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026FactorData:
    score_dates: tuple[str, ...]
    risk_free_returns: tuple[float, ...]
    market_excess_returns: tuple[float, ...]
    factor_returns: tuple[tuple[float, ...], ...]
    source_receipt: dict[str, Any]
    coverage_receipt: dict[str, Any]
    exact_array_receipt: dict[str, Any]
    manifest: Top2000M03RV72026FactorManifest
    receipt_sha256: str
    schema: str = TOP2000_M03R_V7_2026_FACTOR_DATA_SCHEMA
    development_only: bool = True
    retrospective_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        dates = _canonical_score_dates(self.score_dates)
        rows = len(dates)
        rf = np.asarray(self.risk_free_returns, dtype=np.float64)
        market = np.asarray(self.market_excess_returns, dtype=np.float64)
        factors = np.asarray(self.factor_returns, dtype=np.float64)
        if (
            self.schema != TOP2000_M03R_V7_2026_FACTOR_DATA_SCHEMA
            or rf.shape != (rows,)
            or market.shape != (rows,)
            or factors.shape != (rows, len(TOP2000_M03R_V7_2026_FACTOR_NAMES))
            or not np.isfinite(rf).all()
            or not np.isfinite(market).all()
            or not np.isfinite(factors).all()
            or not self.development_only
            or not self.retrospective_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026FactorDataError(
                "factor data arrays or retrospective labels drifted"
            )
        _validate_embedded_receipt(
            self.source_receipt,
            expected_schema=TOP2000_M03R_V7_2026_FACTOR_SOURCE_RECEIPT_SCHEMA,
            label="factor source receipt",
        )
        _validate_embedded_receipt(
            self.coverage_receipt,
            expected_schema=TOP2000_M03R_V7_2026_FACTOR_COVERAGE_RECEIPT_SCHEMA,
            label="factor coverage receipt",
        )
        _validate_embedded_receipt(
            self.exact_array_receipt,
            expected_schema=TOP2000_M03R_V7_2026_FACTOR_ARRAY_RECEIPT_SCHEMA,
            label="factor array receipt",
        )
        factor_contract = (
            M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
        )
        count_fields = (
            "five_factor_source_daily_row_count",
            "momentum_source_daily_row_count",
            "five_factor_unselected_source_row_count",
            "momentum_unselected_source_row_count",
            "five_factor_unused_post_end_source_row_count",
            "momentum_unused_post_end_source_row_count",
            "post_end_source_rows_used",
            "imputed_value_count",
        )
        if any(
            type(self.coverage_receipt.get(name)) is not int
            or int(self.coverage_receipt[name]) < 0
            for name in count_fields
        ):
            raise Top2000M03RV72026FactorDataError(
                "factor coverage receipt contains invalid row counts"
            )
        counts = {
            name: int(self.coverage_receipt[name]) for name in count_fields
        }
        _require_digest(
            "retrieval_receipt_sha256",
            self.source_receipt.get("retrieval_receipt_sha256"),
        )
        _require_digest(
            "retrieval frozen_plan_file_sha256",
            self.source_receipt.get("frozen_plan_file_sha256"),
        )
        _require_digest(
            "retrieval frozen_plan_receipt_sha256",
            self.source_receipt.get("frozen_plan_receipt_sha256"),
        )
        if (
            self.source_receipt.get("source") != factor_contract.source_library
            or self.source_receipt.get("five_factor_url")
            != factor_contract.five_factor_download_url
            or self.source_receipt.get("momentum_url")
            != factor_contract.momentum_download_url
            or self.source_receipt.get("retrieval_method")
            != factor_contract.official_source_transport
            or self.source_receipt.get("official_source_verified") is not True
            or self.source_receipt.get("caller_staged_archives") is not False
            or self.source_receipt.get(
                "source_containers_may_include_unused_post_end_rows"
            )
            is not True
            or self.coverage_receipt.get("extraction_rule")
            != factor_contract.extraction_rule
            or self.coverage_receipt.get("exact_date_join") is not True
            or self.coverage_receipt.get("score_row_count") != rows
            or self.coverage_receipt.get("first_score_date") != dates[0]
            or self.coverage_receipt.get("last_score_date") != dates[-1]
            or self.coverage_receipt.get("missing_five_factor_dates") != []
            or self.coverage_receipt.get("missing_momentum_dates") != []
            or self.coverage_receipt.get(
                "source_containers_may_include_unused_post_end_rows"
            )
            is not True
            or self.coverage_receipt.get("post_end_source_rows_used") != 0
            or self.coverage_receipt.get("post_end_source_values_parsed")
            is not False
            or self.coverage_receipt.get(
                "post_end_source_rows_may_enter_evaluator_arrays"
            )
            is not False
            or self.coverage_receipt.get("imputed_value_count") != 0
            or self.coverage_receipt.get("score_window_shortened") is not False
            or counts["five_factor_unselected_source_row_count"]
            != counts["five_factor_source_daily_row_count"] - rows
            or counts["momentum_unselected_source_row_count"]
            != counts["momentum_source_daily_row_count"] - rows
            or counts["five_factor_unused_post_end_source_row_count"]
            > counts["five_factor_unselected_source_row_count"]
            or counts["momentum_unused_post_end_source_row_count"]
            > counts["momentum_unselected_source_row_count"]
            or self.exact_array_receipt.get("extraction_rule")
            != factor_contract.extraction_rule
            or self.exact_array_receipt.get("post_end_source_rows_used") != 0
        ):
            raise Top2000M03RV72026FactorDataError(
                "official retrieval or exact score-date extraction semantics drifted"
            )
        if (
            self.source_receipt.get("receipt_sha256")
            != self.manifest.source_receipt_sha256
            or self.coverage_receipt.get("receipt_sha256")
            != self.manifest.coverage_receipt_sha256
            or self.exact_array_receipt.get("receipt_sha256")
            != self.manifest.exact_array_receipt_sha256
        ):
            raise Top2000M03RV72026FactorDataError(
                "factor manifest does not bind the exact receipts"
            )
        if (
            self.source_receipt.get("five_factor_zip_sha256")
            != self.manifest.five_factor_source_file_sha256
            or self.source_receipt.get("momentum_zip_sha256")
            != self.manifest.momentum_source_file_sha256
            or self.coverage_receipt.get("score_dates_sha256")
            != _array_sha256("score_dates", np.asarray(dates, dtype=object))
            or self.exact_array_receipt.get("score_dates_sha256")
            != self.coverage_receipt.get("score_dates_sha256")
            or self.exact_array_receipt.get("risk_free_returns_sha256")
            != _array_sha256("risk_free_returns", rf)
            or self.exact_array_receipt.get("market_excess_returns_sha256")
            != _array_sha256("market_excess_returns", market)
            or self.exact_array_receipt.get("factor_returns_sha256")
            != _array_sha256("factor_returns", factors)
        ):
            raise Top2000M03RV72026FactorDataError(
                "factor receipt arrays or source archives drifted"
            )
        unsigned = asdict(self)
        unsigned.pop("receipt_sha256")
        if _require_digest("receipt_sha256", self.receipt_sha256) != _sha256(unsigned):
            raise Top2000M03RV72026FactorDataError("factor data receipt hash drifted")


def build_top2000_m03r_v7_2026_factor_data(
    *,
    retrieval_evidence: Top2000M03RV72026OfficialFactorRetrieval,
    score_dates: Sequence[str],
) -> Top2000M03RV72026FactorData:
    """Extract exact score dates from package-retrieved official archives."""

    dates = _canonical_score_dates(score_dates)
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
    if not isinstance(
        retrieval_evidence, Top2000M03RV72026OfficialFactorRetrieval
    ):
        raise Top2000M03RV72026FactorDataError(
            "package-owned official retrieval evidence is required; "
            "caller-staged archives are unverified"
        )
    ff_raw, ff_zip_sha, ff_member_sha = _read_zip_member(
        Path(retrieval_evidence.five_factor_archive_path),
        TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
    )
    mom_raw, mom_zip_sha, mom_member_sha = _read_zip_member(
        Path(retrieval_evidence.momentum_archive_path),
        TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
    )
    if (
        ff_zip_sha != retrieval_evidence.five_factor_archive_sha256
        or mom_zip_sha != retrieval_evidence.momentum_archive_sha256
        or ff_member_sha != retrieval_evidence.five_factor_member_sha256
        or mom_member_sha != retrieval_evidence.momentum_member_sha256
    ):
        raise Top2000M03RV72026FactorDataError(
            "factor archives do not match the package-owned retrieval evidence"
        )
    selected_dates = frozenset(dates)
    ff = _parse_daily_csv(
        ff_raw,
        expected_columns=contract.five_factor_source_columns,
        selected_dates=selected_dates,
        last_score_date=dates[-1],
    )
    mom = _parse_daily_csv(
        mom_raw,
        expected_columns=(contract.momentum_source_column,),
        selected_dates=selected_dates,
        last_score_date=dates[-1],
    )
    missing_ff = tuple(
        value for value in dates if value not in ff.selected_returns
    )
    missing_mom = tuple(
        value for value in dates if value not in mom.selected_returns
    )
    if missing_ff or missing_mom:
        raise Top2000M03RV72026FactorDataError(
            "official factors do not cover every scored date; score window is unchanged"
        )
    risk_free = np.asarray(
        [ff.selected_returns[value][5] for value in dates], dtype=np.float64
    )
    market = np.asarray(
        [ff.selected_returns[value][0] for value in dates], dtype=np.float64
    )
    factors = np.asarray(
        [
            (
                *ff.selected_returns[value][1:5],
                mom.selected_returns[value][0],
            )
            for value in dates
        ],
        dtype=np.float64,
    )
    source_unsigned = {
        "schema": TOP2000_M03R_V7_2026_FACTOR_SOURCE_RECEIPT_SCHEMA,
        "source": contract.source_library,
        "five_factor_url": contract.five_factor_download_url,
        "momentum_url": contract.momentum_download_url,
        "five_factor_zip_sha256": ff_zip_sha,
        "momentum_zip_sha256": mom_zip_sha,
        "five_factor_member": TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
        "momentum_member": TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
        "five_factor_member_sha256": ff_member_sha,
        "momentum_member_sha256": mom_member_sha,
        "retrieval_receipt_sha256": retrieval_evidence.receipt_sha256,
        "retrieval_method": retrieval_evidence.retrieval_method,
        "retrieved_at_utc": retrieval_evidence.retrieved_at_utc,
        "frozen_plan_file_sha256": (
            retrieval_evidence.frozen_plan_file_sha256
        ),
        "frozen_plan_receipt_sha256": (
            retrieval_evidence.frozen_plan_receipt_sha256
        ),
        "official_source_verified": True,
        "caller_staged_archives": False,
        "source_containers_may_include_unused_post_end_rows": True,
    }
    source_receipt = {**source_unsigned, "receipt_sha256": _sha256(source_unsigned)}
    coverage_unsigned = {
        "schema": TOP2000_M03R_V7_2026_FACTOR_COVERAGE_RECEIPT_SCHEMA,
        "score_dates_sha256": _array_sha256(
            "score_dates", np.asarray(dates, dtype=object)
        ),
        "score_row_count": len(dates),
        "first_score_date": dates[0],
        "last_score_date": dates[-1],
        "missing_five_factor_dates": [],
        "missing_momentum_dates": [],
        "exact_date_join": True,
        "extraction_rule": contract.extraction_rule,
        "five_factor_first_source_date": ff.first_source_date,
        "five_factor_last_source_date": ff.last_source_date,
        "momentum_first_source_date": mom.first_source_date,
        "momentum_last_source_date": mom.last_source_date,
        "five_factor_source_daily_row_count": ff.source_daily_row_count,
        "momentum_source_daily_row_count": mom.source_daily_row_count,
        "five_factor_unselected_source_row_count": (
            ff.unselected_source_row_count
        ),
        "momentum_unselected_source_row_count": mom.unselected_source_row_count,
        "five_factor_unused_post_end_source_row_count": (
            ff.unused_post_end_source_row_count
        ),
        "momentum_unused_post_end_source_row_count": (
            mom.unused_post_end_source_row_count
        ),
        "source_containers_may_include_unused_post_end_rows": True,
        "post_end_source_rows_used": 0,
        "post_end_source_values_parsed": False,
        "post_end_source_rows_may_enter_evaluator_arrays": False,
        "imputed_value_count": 0,
        "score_window_shortened": False,
    }
    coverage_receipt = {
        **coverage_unsigned,
        "receipt_sha256": _sha256(coverage_unsigned),
    }
    array_unsigned = {
        "schema": TOP2000_M03R_V7_2026_FACTOR_ARRAY_RECEIPT_SCHEMA,
        "score_dates_sha256": coverage_unsigned["score_dates_sha256"],
        "risk_free_returns_sha256": _array_sha256("risk_free_returns", risk_free),
        "market_excess_returns_sha256": _array_sha256(
            "market_excess_returns", market
        ),
        "factor_returns_sha256": _array_sha256("factor_returns", factors),
        "factor_names": list(TOP2000_M03R_V7_2026_FACTOR_NAMES),
        "source_unit": "percent",
        "evaluator_unit": "decimal-return",
        "conversion": "divide-by-100",
        "extraction_rule": contract.extraction_rule,
        "post_end_source_rows_used": 0,
    }
    array_receipt = {**array_unsigned, "receipt_sha256": _sha256(array_unsigned)}
    source_receipt_sha256 = _require_digest(
        "source_receipt_sha256", source_receipt["receipt_sha256"]
    )
    coverage_receipt_sha256 = _require_digest(
        "coverage_receipt_sha256", coverage_receipt["receipt_sha256"]
    )
    array_receipt_sha256 = _require_digest(
        "array_receipt_sha256", array_receipt["receipt_sha256"]
    )
    manifest = build_top2000_m03r_v7_2026_factor_manifest(
        five_factor_source_file_sha256=ff_zip_sha,
        momentum_source_file_sha256=mom_zip_sha,
        source_receipt_sha256=source_receipt_sha256,
        coverage_receipt_sha256=coverage_receipt_sha256,
        exact_array_receipt_sha256=array_receipt_sha256,
    )
    fields: dict[str, Any] = {
        "score_dates": dates,
        "risk_free_returns": tuple(float(value) for value in risk_free),
        "market_excess_returns": tuple(float(value) for value in market),
        "factor_returns": tuple(
            tuple(float(value) for value in row) for row in factors
        ),
        "source_receipt": source_receipt,
        "coverage_receipt": coverage_receipt,
        "exact_array_receipt": array_receipt,
        "manifest": manifest,
        "schema": TOP2000_M03R_V7_2026_FACTOR_DATA_SCHEMA,
        "development_only": True,
        "retrospective_only": True,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
    }
    receipt_payload = {**fields, "manifest": asdict(manifest)}
    return Top2000M03RV72026FactorData(
        **fields,
        receipt_sha256=_sha256(receipt_payload),
    )


def write_top2000_m03r_v7_2026_factor_data(
    data: Top2000M03RV72026FactorData,
    path: str | Path,
) -> str:
    """Publish the small canonical array artifact without overwrite."""

    if not isinstance(data, Top2000M03RV72026FactorData):
        raise Top2000M03RV72026FactorDataError("typed factor data is required")
    payload = asdict(data)
    destination = Path(path)
    raw = _canonical_json(payload)
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == raw
        ):
            return hashlib.sha256(raw).hexdigest()
        raise Top2000M03RV72026FactorDataError(
            f"refusing to overwrite factor data artifact {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as sink:
        sink.write(raw)
        sink.flush()
    return hashlib.sha256(raw).hexdigest()


def load_top2000_m03r_v7_2026_factor_data(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Top2000M03RV72026FactorData:
    """Load and replay every embedded receipt from a canonical artifact."""

    _require_digest("expected_file_sha256", expected_file_sha256)
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise Top2000M03RV72026FactorDataError(
            "factor data artifact must be a regular non-symlink file"
        )
    if _file_sha256(source) != expected_file_sha256:
        raise Top2000M03RV72026FactorDataError("factor data file SHA-256 drifted")
    try:
        payload = json.loads(source.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV72026FactorDataError(
            "factor data artifact cannot be read"
        ) from exc
    if not isinstance(payload, dict):
        raise Top2000M03RV72026FactorDataError("factor data artifact must be an object")
    expected_keys = {
        "score_dates",
        "risk_free_returns",
        "market_excess_returns",
        "factor_returns",
        "source_receipt",
        "coverage_receipt",
        "exact_array_receipt",
        "manifest",
        "receipt_sha256",
        "schema",
        "development_only",
        "retrospective_only",
        "scientific_reporting_eligible",
        "promotion_eligible",
    }
    if set(payload) != expected_keys or not isinstance(payload.get("manifest"), dict):
        raise Top2000M03RV72026FactorDataError(
            "factor data artifact fields drifted"
        )
    fields = dict(payload)
    try:
        manifest_payload = dict(payload["manifest"])
        manifest_payload["factor_names"] = tuple(manifest_payload["factor_names"])
        fields["manifest"] = Top2000M03RV72026FactorManifest(
            **manifest_payload
        )
        fields["score_dates"] = tuple(payload["score_dates"])
        fields["risk_free_returns"] = tuple(payload["risk_free_returns"])
        fields["market_excess_returns"] = tuple(payload["market_excess_returns"])
        fields["factor_returns"] = tuple(
            tuple(row) for row in payload["factor_returns"]
        )
        return Top2000M03RV72026FactorData(**fields)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, Top2000M03RV72026FactorDataError):
            raise
        raise Top2000M03RV72026FactorDataError(
            "factor data artifact cannot be reconstructed"
        ) from exc


__all__ = [
    "TOP2000_M03R_V7_2026_FACTOR_ARRAY_RECEIPT_SCHEMA",
    "TOP2000_M03R_V7_2026_FACTOR_COVERAGE_RECEIPT_SCHEMA",
    "TOP2000_M03R_V7_2026_FACTOR_DATA_SCHEMA",
    "TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_METHOD",
    "TOP2000_M03R_V7_2026_FACTOR_RETRIEVAL_RECEIPT_SCHEMA",
    "TOP2000_M03R_V7_2026_FACTOR_SOURCE_RECEIPT_SCHEMA",
    "TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER",
    "TOP2000_M03R_V7_2026_MOMENTUM_MEMBER",
    "Top2000M03RV72026FactorData",
    "Top2000M03RV72026FactorDataError",
    "Top2000M03RV72026OfficialFactorRetrieval",
    "build_top2000_m03r_v7_2026_factor_data",
    "load_top2000_m03r_v7_2026_factor_data",
    "load_top2000_m03r_v7_2026_official_factor_retrieval",
    "retrieve_top2000_m03r_v7_2026_official_factor_archives",
    "write_top2000_m03r_v7_2026_factor_data",
]
