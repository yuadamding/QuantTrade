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
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)

PIT_ALPHA_CONVERSION_AUDIT_SCHEMA = "rl-quant.pit-alpha-conversion-readiness-v1"
PIT_ALPHA_FIVE_MINUTE_STAGING_SCHEMA = "rl-quant.pit-alpha-five-minute-staging-v1"
EASTERN = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
BAR_SECONDS = 300
EXPECTED_NORMAL_SESSION_INTERVALS = 78
_SAFE_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,15}$")
_FILE_STATUSES = frozenset({"downloaded", "exists"})


class PolygonPITAlphaConversionError(ValueError):
    """The organized Polygon cache cannot satisfy a conversion invariant."""


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
    stale_output_path_rows: int
    canonical_missing_file_rows: int | None
    manifest_declared_remaining_symbol_days: int
    manifest_derived_incomplete_symbol_days: int
    covariate_symbol_count: int
    unexpected_covariate_symbols: tuple[str, ...]
    missing_covariate_symbols: tuple[str, ...]


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
    causal_availability_known: bool
    cash_per_share: float | None
    share_ratio: float | None
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
    development_five_minute_conversion_ready: bool
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise PolygonPITAlphaConversionError(f"expected a JSON object: {path}")
    return payload


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
    manifest_rows = 0

    with shard.manifest_csv.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"symbol", "date", "status", "output_path", "sha256"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PolygonPITAlphaConversionError(
                f"organized manifest is missing required fields: {shard.manifest_csv}"
            )
        for row in reader:
            manifest_rows += 1
            symbol = str(row.get("symbol") or "").strip().upper()
            date_value = str(row.get("date") or "").strip()
            status = str(row.get("status") or "").strip().lower()
            statuses[status] += 1
            if not str(row.get("sha256") or "").strip():
                blank_sha += 1
            if _SAFE_SYMBOL.fullmatch(symbol) is None:
                malformed += 1
                continue
            manifest_symbols.add(symbol)
            canonical = _canonical_symbol_day_path(
                shard.second_aggs_root, symbol, date_value
            )
            recorded = Path(str(row.get("output_path") or ""))
            if tuple(recorded.parts[-5:]) != tuple(canonical.parts[-5:]):
                stale_paths += 1
            if (
                verify_canonical_files
                and status in _FILE_STATUSES
                and not canonical.is_file()
            ):
                canonical_missing += 1

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
        stale_output_path_rows=stale_paths,
        canonical_missing_file_rows=canonical_missing
        if verify_canonical_files
        else None,
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
            for line_number, payload in _iter_jsonl(path):
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
                    causal_availability_known=announced is not None,
                    cash_per_share=cash,
                    share_ratio=ratio,
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

    action_keys: set[tuple[str, str]] = set()
    actions: list[PolygonCorporateActionCandidate] = []
    for shard in shards:
        for action in iter_corporate_action_candidates(shard.covariates_root):
            action_key = (action.source_dataset, action.vendor_event_id)
            if action_key not in action_keys:
                action_keys.add(action_key)
                actions.append(action)

    manifest_byte_authority_missing = any(row.blank_sha256_rows for row in shard_audits)
    manifest_inconsistent = any(
        row.malformed_manifest_rows
        or row.unexpected_manifest_symbols
        or row.missing_manifest_symbols
        or row.unexpected_covariate_symbols
        or row.missing_covariate_symbols
        or row.stale_output_path_rows
        or (
            row.canonical_missing_file_rows is not None
            and row.canonical_missing_file_rows > 0
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

    development_ready = all(
        not row.missing_manifest_symbols
        and (
            row.canonical_missing_file_rows is None
            or row.canonical_missing_file_rows == 0
        )
        for row in shard_audits
    )
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
        "development_five_minute_conversion_ready": development_ready,
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
        development_five_minute_conversion_ready=development_ready,
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


def aggregate_polygon_second_bars_to_five_minutes(frame: Any) -> Any:
    """Aggregate one regular-session symbol-day without zero-filling missing bins."""

    pd = _require_pandas()
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
    if len(set(work["timestamp_exchange"].dt.date)) != 1:
        raise PolygonPITAlphaConversionError(
            "five-minute conversion crosses exchange dates"
        )
    exchange_time = work["timestamp_exchange"].dt.time
    work = work.loc[(exchange_time >= RTH_OPEN) & (exchange_time < RTH_CLOSE)].copy()
    if work.empty:
        raise PolygonPITAlphaConversionError(
            "symbol-day contains no regular-session rows"
        )
    work = work.sort_values("timestamp_ms")
    open_offset_seconds = RTH_OPEN.hour * 3_600 + RTH_OPEN.minute * 60
    opened = work["timestamp_exchange"].dt.normalize() + pd.Timedelta(
        open_offset_seconds,
        unit="s",
    )
    seconds_from_open = (
        (work["timestamp_exchange"] - opened).dt.total_seconds().astype("int64")
    )
    work["interval_index"] = seconds_from_open // BAR_SECONDS
    if bool(
        (
            (work["interval_index"] < 0)
            | (work["interval_index"] >= EXPECTED_NORMAL_SESSION_INTERVALS)
        ).any()
    ):
        raise PolygonPITAlphaConversionError(
            "regular-session row mapped outside the 78-bin grid"
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
    session_day = work["timestamp_exchange"].iloc[0].date().isoformat()
    local_open = datetime.combine(
        date.fromisoformat(session_day), RTH_OPEN, tzinfo=EASTERN
    )
    open_ms = int(local_open.astimezone(UTC).timestamp() * 1000)
    output["session_date"] = session_day
    output["interval_start_ms"] = (
        open_ms + output["interval_index"] * BAR_SECONDS * 1000
    )
    output["interval_end_ms"] = output["interval_start_ms"] + BAR_SECONDS * 1000
    output["available_at_ms"] = pd.concat(
        (
            output["interval_end_ms"],
            output["latest_source_timestamp_ms"] + 1_000,
        ),
        axis=1,
    ).max(axis=1)
    output["observed"] = True
    return output[
        [
            "symbol",
            "session_date",
            "interval_index",
            "interval_start_ms",
            "interval_end_ms",
            "available_at_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "transactions",
            "source_row_count",
            "observed",
        ]
    ]


def resolve_symbol_day_path(
    shards: Sequence[OrganizedPolygonShard], symbol: str, day: str
) -> Path:
    """Resolve exactly one canonical symbol-day file across the organized shards."""

    normalized = symbol.strip().upper()
    if _SAFE_SYMBOL.fullmatch(normalized) is None:
        raise PolygonPITAlphaConversionError("symbol is unsafe")
    matches = [
        path
        for shard in shards
        if (
            path := _canonical_symbol_day_path(shard.second_aggs_root, normalized, day)
        ).is_file()
    ]
    if len(matches) != 1:
        raise PolygonPITAlphaConversionError(
            f"expected one canonical source file for {normalized} {day}, found {len(matches)}"
        )
    return matches[0]


def convert_symbol_day_to_five_minute_staging(
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create one no-clobber development Parquet and an immutable receipt."""

    pd = _require_pandas()
    if (
        output_path.exists()
        or output_path.with_suffix(output_path.suffix + ".receipt.json").exists()
    ):
        raise PolygonPITAlphaConversionError("staging output already exists")
    frame = pd.read_parquet(source_path)
    converted = aggregate_polygon_second_bars_to_five_minutes(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(output_path, index=False)
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    body = {
        "schema": PIT_ALPHA_FIVE_MINUTE_STAGING_SCHEMA,
        "source_path": str(source_path),
        "source_file_sha256": file_sha256(source_path),
        "output_path": str(output_path),
        "output_file_sha256": file_sha256(output_path),
        "symbol": str(converted["symbol"].iloc[0]),
        "session_date": str(converted["session_date"].iloc[0]),
        "observed_interval_count": len(converted),
        "expected_normal_session_interval_count": EXPECTED_NORMAL_SESSION_INTERVALS,
        "missing_intervals_are_not_zero_filled": True,
        "source_prices_adjusted": bool(frame["adjusted"].all())
        if "adjusted" in frame
        else None,
        "reportable_pit_authority": False,
        "blocking_reason": "permanent identity and PIT economic authorities are not closed",
    }
    payload = {**body, "receipt_sha256": semantic_sha256(body)}
    receipt_path.write_bytes(canonical_json_file_bytes(payload))
    return payload


def write_conversion_audit(path: Path, audit: PolygonPITAlphaConversionAudit) -> str:
    """Write one canonical, no-clobber conversion-readiness report."""

    if path.exists():
        raise PolygonPITAlphaConversionError(f"audit output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_file_bytes(audit.to_dict()))
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "EXPECTED_NORMAL_SESSION_INTERVALS",
    "PIT_ALPHA_CONVERSION_AUDIT_SCHEMA",
    "PIT_ALPHA_FIVE_MINUTE_STAGING_SCHEMA",
    "OrganizedPolygonShard",
    "OrganizedPolygonShardAudit",
    "PolygonCorporateActionCandidate",
    "PolygonIdentityObservation",
    "PolygonPITAlphaConversionAudit",
    "PolygonPITAlphaConversionError",
    "aggregate_polygon_second_bars_to_five_minutes",
    "audit_organized_polygon_for_pit_alpha",
    "convert_symbol_day_to_five_minute_staging",
    "default_organized_polygon_shards",
    "iter_corporate_action_candidates",
    "iter_identity_observations",
    "resolve_symbol_day_path",
    "write_conversion_audit",
]
