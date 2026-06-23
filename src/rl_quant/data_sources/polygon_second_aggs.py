from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from rl_quant.research_protocol import stable_json_hash, utc_now_iso


RTH_START = time(9, 30)
RTH_END = time(16, 0)
SECOND_TIMESPAN = "second"
SECOND_MULTIPLIER = 1
DEFAULT_BAR_LATENCY_MS = 1_000
REPORTABILITY_METADATA_KEYS = (
    "provider",
    "asset_class",
    "bar_type",
    "adjusted",
    "download_started_at_utc",
    "download_status",
    "universe_asof",
)
COMPLETED_STATUSES = {"downloaded", "exists", "empty"}
FILE_STATUSES = {"downloaded", "exists"}


ManifestRows = list[dict[str, str]]


@dataclass(frozen=True)
class PolygonSecondAggConfig:
    root: Path
    manifest_csv: Path
    dataset_manifest_json: Path
    rth_only: bool = True
    include_extended_hours: bool = False
    require_adjusted: bool = True
    min_symbol_day_coverage: float = 0.95
    max_failed_symbol_days: int = 0
    bar_latency_ms: int = DEFAULT_BAR_LATENCY_MS
    ingestion_latency_ms: int = 0

    def validate(self) -> None:
        if self.min_symbol_day_coverage < 0 or self.min_symbol_day_coverage > 1:
            raise ValueError("min_symbol_day_coverage must be between 0 and 1.")
        if self.max_failed_symbol_days < 0:
            raise ValueError("max_failed_symbol_days must be non-negative.")
        if self.bar_latency_ms < DEFAULT_BAR_LATENCY_MS:
            raise ValueError("bar_latency_ms must be at least 1000 for second aggregates.")
        if self.ingestion_latency_ms < 0:
            raise ValueError("ingestion_latency_ms must be non-negative.")
        if self.rth_only and self.include_extended_hours:
            raise ValueError("rth_only and include_extended_hours cannot both be true.")


@dataclass(frozen=True)
class PolygonSecondAggManifest:
    root: Path
    symbols: list[str]
    start: str
    end_exclusive: str
    market_weekdays: int
    source: str
    source_access: str
    adjusted: bool
    timespan: str
    multiplier: int
    source_download_complete: bool
    universe_asof: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["root"] = str(self.root)
        return payload


@dataclass(frozen=True)
class PolygonSecondAggQualityReport:
    created_at_utc: str
    root: str
    expected_symbol_days: int
    completed_symbol_days: int
    downloaded_symbol_days: int
    existing_symbol_days: int
    empty_symbol_days: int
    failed_symbol_days: int
    pending_symbol_days: int
    missing_file_symbol_days: int
    coverage_ratio: float
    row_count: int
    symbols_expected: int
    symbols_seen: int
    dates_expected: int
    dates_seen: int
    source: str
    source_access: str
    source_download_complete: bool
    reportable: bool
    reportability_errors: list[str] = field(default_factory=list)
    issue_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return stable_json_hash(self.to_dict())


def load_manifest(path: Path) -> ManifestRows:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        rows: ManifestRows = []
        for row in reader:
            normalized = {key: (value or "").strip() for key, value in row.items()}
            if normalized.get("symbol"):
                normalized["symbol"] = normalized["symbol"].upper()
            if normalized.get("status"):
                normalized["status"] = normalized["status"].lower()
            rows.append(normalized)
    return rows


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _source_access_from_source(source: str) -> str:
    text = source.lower()
    if "s3" in text or "flat" in text:
        return "AWS S3"
    if "rest" in text or "range endpoint" in text:
        return "REST"
    return "unknown"


def normalize_source_metadata(
    payload: Mapping[str, Any],
    *,
    source_access: str | None = None,
) -> dict[str, Any]:
    access = source_access or str(payload.get("source_access") or "").strip() or _source_access_from_source(
        str(payload.get("source", ""))
    )
    if access.upper().replace("_", " ") in {"AWS S3", "S3"}:
        source = "Polygon flat files / S3"
        access = "AWS S3"
    elif access.upper() == "REST":
        source = "Polygon REST aggregate range endpoint"
        access = "REST"
    else:
        source = str(payload.get("source") or "Polygon aggregate bars")
    return {
        "source": source,
        "source_access": access,
        "provider": str(payload.get("provider") or "polygon"),
        "asset_class": str(payload.get("asset_class") or "stocks"),
        "bar_type": str(payload.get("bar_type") or "second_aggregate"),
        "adjusted": bool(payload.get("adjusted", True)),
        "timespan": str(payload.get("timespan") or SECOND_TIMESPAN),
        "multiplier": int(payload.get("multiplier", SECOND_MULTIPLIER)),
        "download_started_at_utc": payload.get("download_started_at_utc") or payload.get("created_at_utc"),
        "download_completed_at_utc": payload.get("download_completed_at_utc"),
        "download_status": str(payload.get("download_status") or _infer_download_status(payload)),
        "universe_asof": payload.get("universe_asof"),
    }


def _infer_download_status(payload: Mapping[str, Any]) -> str:
    remaining = int(float(payload.get("remaining_symbol_days", 0) or 0))
    if remaining > 0:
        return "running"
    return "complete"


def _status_counts(rows: Iterable[Mapping[str, str]]) -> Counter[str]:
    return Counter(str(row.get("status", "")).lower() for row in rows)


def _manifest_symbols(rows: Iterable[Mapping[str, str]]) -> list[str]:
    return sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})


def _manifest_dates(rows: Iterable[Mapping[str, str]]) -> list[str]:
    return sorted({str(row.get("date", "")) for row in rows if row.get("date")})


def _row_count(rows: Iterable[Mapping[str, str]]) -> int:
    total = 0
    for row in rows:
        try:
            total += int(float(str(row.get("rows", "0") or 0)))
        except ValueError:
            continue
    return total


def _expected_symbol_days(rows: ManifestRows, payload: Mapping[str, Any]) -> int:
    symbols = int(float(payload.get("symbols", 0) or 0))
    weekdays = int(float(payload.get("market_weekdays", 0) or 0))
    if symbols > 0 and weekdays > 0:
        return symbols * weekdays
    manifest_symbols = _manifest_symbols(rows)
    manifest_dates = _manifest_dates(rows)
    if manifest_symbols and manifest_dates:
        return len(manifest_symbols) * len(manifest_dates)
    return len(rows)


def _missing_file_count(rows: Iterable[Mapping[str, str]], statuses: set[str] = FILE_STATUSES) -> int:
    missing = 0
    for row in rows:
        if str(row.get("status", "")).lower() not in statuses:
            continue
        output_path = str(row.get("output_path", ""))
        if output_path and not Path(output_path).exists():
            missing += 1
    return missing


def validate_manifest(
    manifest: ManifestRows,
    config: PolygonSecondAggConfig,
    *,
    source_access: str | None = None,
) -> PolygonSecondAggQualityReport:
    config.validate()
    payload = load_dataset_manifest(config.dataset_manifest_json)
    source_meta = normalize_source_metadata(payload, source_access=source_access)
    counts = _status_counts(manifest)
    expected = max(_expected_symbol_days(manifest, payload), 1)
    completed = sum(counts.get(status, 0) for status in COMPLETED_STATUSES)
    coverage_ratio = min(completed / float(expected), 1.0)
    failed = counts.get("failed", 0) + counts.get("error", 0)
    pending = max(expected - completed - failed, 0)
    missing_files = _missing_file_count(manifest)
    remaining = int(float(payload.get("remaining_symbol_days", pending) or 0))
    source_download_complete = (
        remaining == 0
        and pending == 0
        and source_meta["download_status"].lower() in {"complete", "completed", "done"}
    )

    errors: list[str] = []
    if not source_download_complete:
        errors.append("source_download_incomplete")
    if failed > config.max_failed_symbol_days:
        errors.append("failed_symbol_days_exceed_limit")
    if coverage_ratio < config.min_symbol_day_coverage:
        errors.append("symbol_day_coverage_below_minimum")
    if missing_files:
        errors.append("manifest_references_missing_files")
    for key in REPORTABILITY_METADATA_KEYS:
        # source_meta is the NORMALIZED dict, which always contains every key (some default to None),
        # so a key-presence test can never fire. Check the resolved VALUE instead: None or blank string
        # is "missing". bool/int values (e.g. adjusted=False) are present and never flagged.
        value = source_meta.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"source_metadata_missing_{key}")
    if str(source_meta["source_access"]).lower() == "unknown":
        errors.append("source_access_unknown")

    return PolygonSecondAggQualityReport(
        created_at_utc=utc_now_iso(),
        root=str(config.root),
        expected_symbol_days=expected,
        completed_symbol_days=completed,
        downloaded_symbol_days=counts.get("downloaded", 0),
        existing_symbol_days=counts.get("exists", 0),
        empty_symbol_days=counts.get("empty", 0),
        failed_symbol_days=failed,
        pending_symbol_days=pending,
        missing_file_symbol_days=missing_files,
        coverage_ratio=coverage_ratio,
        row_count=_row_count(manifest),
        symbols_expected=int(float(payload.get("symbols", 0) or len(_manifest_symbols(manifest)))),
        symbols_seen=len(_manifest_symbols(manifest)),
        dates_expected=int(float(payload.get("market_weekdays", 0) or len(_manifest_dates(manifest)))),
        dates_seen=len(_manifest_dates(manifest)),
        source=str(source_meta["source"]),
        source_access=str(source_meta["source_access"]),
        source_download_complete=source_download_complete,
        reportable=not errors,
        reportability_errors=list(dict.fromkeys(errors)),
        issue_counts=dict(counts),
    )


def iter_symbol_day_files(
    manifest: ManifestRows,
    *,
    statuses: Iterable[str] = FILE_STATUSES,
) -> Iterator[Path]:
    allowed = {status.lower() for status in statuses}
    for row in manifest:
        if str(row.get("status", "")).lower() not in allowed:
            continue
        output_path = str(row.get("output_path", ""))
        if not output_path:
            continue
        path = Path(output_path)
        if path.suffix == ".parquet":
            yield path


def available_timestamp_ms(
    timestamp_ms: int,
    *,
    bar_latency_ms: int = DEFAULT_BAR_LATENCY_MS,
    ingestion_latency_ms: int = 0,
) -> int:
    if bar_latency_ms < DEFAULT_BAR_LATENCY_MS:
        raise ValueError("bar_latency_ms must be at least 1000 for one-second aggregate bars.")
    if ingestion_latency_ms < 0:
        raise ValueError("ingestion_latency_ms must be non-negative.")
    return int(timestamp_ms) + int(bar_latency_ms) + int(ingestion_latency_ms)


def iso_to_timestamp_ms(value: str) -> int:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and pyarrow are required to read Polygon second aggregate Parquet files.") from exc
    return pd


def _ensure_timestamp_ms(frame: Any) -> Any:
    pd = _require_pandas()
    out = frame.copy()
    if "timestamp_ms" in out.columns:
        out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce").astype("Int64")
    elif "timestamp_utc" in out.columns:
        out["timestamp_ms"] = (pd.to_datetime(out["timestamp_utc"], utc=True).astype("int64") // 1_000_000).astype("Int64")
    else:
        raise ValueError("Polygon second aggregate frame must include timestamp_ms or timestamp_utc.")
    out = out.dropna(subset=["timestamp_ms"]).copy()
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["timestamp"] = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.upper()
    return out


def _regular_session_mask(frame: Any) -> Any:
    pd = _require_pandas()
    if "timestamp_exchange" in frame.columns:
        # Parse with utc=True then convert to Eastern: a mixed-offset column otherwise yields object dtype,
        # on which `.dt.time` raises. This matches the canonical-timestamp branch's tz handling below.
        exchange_ts = pd.to_datetime(frame["timestamp_exchange"], utc=True, errors="coerce").dt.tz_convert(
            "America/New_York"
        )
        exchange_time = exchange_ts.dt.time
        return (exchange_time >= RTH_START) & (exchange_time < RTH_END)
    eastern = frame["timestamp"].dt.tz_convert("America/New_York")
    exchange_time = eastern.dt.time
    return (exchange_time >= RTH_START) & (exchange_time < RTH_END)


def normalize_symbol_day_frame(
    frame: Any,
    *,
    symbol_hint: str | None = None,
    rth_only: bool = True,
    include_extended_hours: bool = False,
    require_adjusted: bool = True,
) -> Any:
    pd = _require_pandas()
    out = _ensure_timestamp_ms(frame)
    if symbol_hint and "symbol" not in out.columns:
        out["symbol"] = symbol_hint.upper()
    if require_adjusted and "adjusted" in out.columns and not bool(out["adjusted"].fillna(False).all()):
        raise ValueError("Expected adjusted Polygon second aggregate bars.")
    if "timespan" in out.columns:
        bad_timespan = out["timespan"].astype(str).str.lower() != SECOND_TIMESPAN
        if bool(bad_timespan.any()):
            raise ValueError("Expected Polygon second aggregate timespan='second'.")
    if "multiplier" in out.columns:
        # Coerce non-numeric multipliers to NaN (-> default) instead of letting .astype(int) raise an opaque
        # error; a genuinely-wrong numeric multiplier still trips the descriptive ValueError below.
        multiplier = pd.to_numeric(out["multiplier"], errors="coerce").fillna(SECOND_MULTIPLIER).astype("int64")
        if bool((multiplier != SECOND_MULTIPLIER).any()):
            raise ValueError("Expected Polygon second aggregate multiplier=1.")
    if rth_only and not include_extended_hours:
        out = out.loc[_regular_session_mask(out)].copy()
    out = out.sort_values("timestamp_ms").drop_duplicates(["timestamp_ms"], keep="last").reset_index(drop=True)
    return out


def load_symbol_day(
    path: Path,
    *,
    rth_only: bool = True,
    include_extended_hours: bool = False,
    require_adjusted: bool = True,
) -> Any:
    pd = _require_pandas()
    symbol_hint = path.parents[2].name if len(path.parents) >= 3 else path.stem.upper()
    frame = pd.read_parquet(path)
    return normalize_symbol_day_frame(
        frame,
        symbol_hint=symbol_hint,
        rth_only=rth_only,
        include_extended_hours=include_extended_hours,
        require_adjusted=require_adjusted,
    )


def _safe_numeric(frame: Any, column: str) -> Any:
    pd = _require_pandas()
    if column not in frame.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def audit_symbol_day_frame(frame: Any) -> dict[str, int]:
    normalized = _ensure_timestamp_ms(frame)
    issues: dict[str, int] = {}
    issues["rows"] = int(len(normalized))
    issues["duplicate_timestamps"] = int(normalized["timestamp_ms"].duplicated().sum())
    issues["non_monotonic_timestamps"] = int((normalized["timestamp_ms"].diff().dropna() < 0).sum())
    if {"open", "high", "low", "close"}.issubset(normalized.columns):
        open_value = _safe_numeric(normalized, "open")
        high = _safe_numeric(normalized, "high")
        low = _safe_numeric(normalized, "low")
        close = _safe_numeric(normalized, "close")
        issues["bad_ohlc_rows"] = int(((high < open_value) | (high < close) | (low > open_value) | (low > close)).sum())
        pct_jump = close.pct_change().abs()
        issues["large_price_jump_rows"] = int((pct_jump > 0.50).sum())
        if len(close) and float(close.iloc[0]) > 0:
            daily_move = abs(float(close.iloc[-1]) / float(close.iloc[0]) - 1.0)
            issues["large_open_to_close_move_days"] = int(daily_move > 1.0)
    if {"vwap", "high", "low"}.issubset(normalized.columns):
        vwap = _safe_numeric(normalized, "vwap")
        high = _safe_numeric(normalized, "high")
        low = _safe_numeric(normalized, "low")
        issues["bad_vwap_rows"] = int(((vwap < low) | (vwap > high)).sum())
    if "volume" in normalized.columns:
        issues["negative_volume_rows"] = int((_safe_numeric(normalized, "volume") < 0).sum())
    if "transactions" in normalized.columns:
        issues["negative_transaction_rows"] = int((_safe_numeric(normalized, "transactions") < 0).sum())
    if "adjusted" in normalized.columns:
        issues["bad_adjusted_rows"] = int((~normalized["adjusted"].fillna(False).astype(bool)).sum())
    if "timespan" in normalized.columns:
        issues["bad_timespan_rows"] = int((normalized["timespan"].astype(str).str.lower() != SECOND_TIMESPAN).sum())
    if "multiplier" in normalized.columns:
        issues["bad_multiplier_rows"] = int((_safe_numeric(normalized, "multiplier").fillna(SECOND_MULTIPLIER) != SECOND_MULTIPLIER).sum())
    rth_mask = _regular_session_mask(normalized)
    issues["rth_rows"] = int(rth_mask.sum())
    issues["extended_hours_rows"] = int((~rth_mask).sum())
    return issues


def audit_symbol_day_files(
    paths: Iterable[Path],
    *,
    max_files: int | None = None,
) -> dict[str, int]:
    pd = _require_pandas()
    totals: defaultdict[str, int] = defaultdict(int)
    scanned = 0
    for path in paths:
        if max_files is not None and scanned >= max_files:
            break
        if not path.exists():
            totals["missing_files"] += 1
            continue
        frame = pd.read_parquet(path)
        file_issues = audit_symbol_day_frame(frame)
        for key, value in file_issues.items():
            totals[key] += int(value)
        scanned += 1
    totals["scanned_files"] = scanned
    return dict(totals)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


