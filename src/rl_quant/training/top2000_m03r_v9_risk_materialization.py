"""Causal local risk-source materialization for TOP2000 M03R-v9.

The raw Polygon archive is retrospective provider-as-of evidence.  This module
turns only records causally available by each pre-2026 cache decision into a
frozen SIC-sector surface, then derives past-only active-beta and style-risk
controls from the already SHA-verified TOP2000 daily cache.  It never treats
the future-selected TOP2000 universe as reportable or promotion evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from itertools import pairwise
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch

from rl_quant.data_sources.polygon_stock_covariates import (
    infer_covariate_timestamps_ms,
    iter_jsonl_payloads,
)
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_PROTOCOL_SHA256,
    M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    M03RV9OriginRiskExposures,
    qualify_m03r_v9_origin_risk_exposures,
)
from rl_quant.training.top2000_m03r_v9_risk_source import (
    M03RV9RiskSourceInventory,
    M03RV9RiskSourceReadiness,
    audit_m03r_v9_risk_source,
)

M03R_V9_RISK_MATERIALIZATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-risk-materialization-v1"
)
M03R_V9_RISK_ARTIFACT_SCHEMA = "rl-quant.top2000-dev.m03r-v9-risk-artifact-v1"
M03R_V9_RISK_ARTIFACT_MANIFEST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v9-risk-artifact-manifest-v1"
)
M03R_V9_POLYGON_VALIDATION_SCHEMA = (
    "rl-quant.polygon-stock-covariate-delta-validation-v1"
)

M03R_V9_SECTOR_EXPOSURE_NAMES = (
    "sector-agriculture",
    "sector-mining",
    "sector-construction",
    "sector-manufacturing",
    "sector-transport-communications-utilities",
    "sector-wholesale",
    "sector-retail",
    "sector-finance-insurance-real-estate",
    "sector-services",
    "sector-public-administration",
    "sector-unknown-other",
)
M03R_V9_ACTIVE_BETA_EXPOSURE_NAME = "active-beta-63-session-to-c1-proxy"
M03R_V9_STYLE_RISK_EXPOSURE_NAMES = (
    "style-log-return-63-session",
    "style-volatility-63-session",
    "style-mean-log-volume-63-session",
)
M03R_V9_PROJECTOR_EXPOSURE_NAMES = (
    *M03R_V9_SECTOR_EXPOSURE_NAMES,
    M03R_V9_ACTIVE_BETA_EXPOSURE_NAME,
    *M03R_V9_STYLE_RISK_EXPOSURE_NAMES,
)
M03R_V9_PROJECTOR_EXPOSURE_FAMILIES = (
    *("sector" for _ in M03R_V9_SECTOR_EXPOSURE_NAMES),
    "active-beta",
    *("style-risk" for _ in M03R_V9_STYLE_RISK_EXPOSURE_NAMES),
)
M03R_V9_LOOKBACK_SESSIONS = 63
M03R_V9_MINIMUM_HISTORY_SESSIONS = 20
M03R_V9_DECISION_TIME = time(16, 0)
M03R_V9_EXCHANGE_TIMEZONE = ZoneInfo("America/New_York")


class M03RV9RiskMaterializationError(ValueError):
    """Raw risk inputs or a materialized artifact failed qualification."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _axis_sha256(values: tuple[str, ...]) -> str:
    encoded = (
        json.dumps(
            list(values),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV9RiskMaterializationError(f"{name} must be a lowercase SHA-256")
    return value


def _require_regular(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise M03RV9RiskMaterializationError(
            f"{name} must be a regular non-symlink file"
        )
    return resolved


def _require_directory(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not path.exists() or not path.is_dir() or path.is_symlink():
        raise M03RV9RiskMaterializationError(
            f"{name} must be a directory and not a symbolic link"
        )
    return resolved


def _read_universe(path: Path) -> tuple[str, ...]:
    regular = _require_regular(path, name="universe")
    symbols = tuple(
        line.strip().upper()
        for line in regular.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not symbols or len(set(symbols)) != len(symbols):
        raise M03RV9RiskMaterializationError("universe is empty or contains duplicates")
    return symbols


@dataclass(frozen=True, slots=True)
class M03RV9PolygonRiskSlice:
    label: str
    universe_path: Path
    base_root: Path
    delta_root: Path


@dataclass(frozen=True, slots=True)
class M03RV9PolygonRiskInputs:
    slices: tuple[M03RV9PolygonRiskSlice, ...]
    validation_receipt_path: Path
    validation_receipt_file_sha256: str


@dataclass(frozen=True, slots=True)
class M03RV9MaterializedRiskSource:
    exposures: M03RV9OriginRiskExposures
    inventory: M03RV9RiskSourceInventory
    readiness: M03RV9RiskSourceReadiness
    cache_sha256: str
    cache_identity: str
    action_hash: str
    first_exchange_date: str
    last_exchange_date: str
    polygon_validation_receipt_file_sha256: str
    polygon_validation_receipt_payload_sha256: str
    sector_receipt_sha256: str
    active_beta_receipt_sha256: str
    style_risk_receipt_sha256: str
    origin_availability_receipt_sha256: str
    selected_overview_inventory_sha256: str
    source_overview_file_inventory_sha256: str
    raw_symbol_count: int
    mapped_risky_action_count: int
    unused_raw_symbols: tuple[str, ...]
    selected_snapshot_count: int
    explicit_unknown_state_asset_count: int
    future_delta_rows_consumed: int
    receipt_sha256: str
    schema: str = M03R_V9_RISK_MATERIALIZATION_SCHEMA

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_sha256": M03R_V9_PROTOCOL_SHA256,
            "cache_sha256": self.cache_sha256,
            "cache_identity": self.cache_identity,
            "action_hash": self.action_hash,
            "first_exchange_date": self.first_exchange_date,
            "last_exchange_date": self.last_exchange_date,
            "polygon_validation_receipt_file_sha256": (
                self.polygon_validation_receipt_file_sha256
            ),
            "polygon_validation_receipt_payload_sha256": (
                self.polygon_validation_receipt_payload_sha256
            ),
            "sector_receipt_sha256": self.sector_receipt_sha256,
            "active_beta_receipt_sha256": self.active_beta_receipt_sha256,
            "style_risk_receipt_sha256": self.style_risk_receipt_sha256,
            "origin_availability_receipt_sha256": (
                self.origin_availability_receipt_sha256
            ),
            "selected_overview_inventory_sha256": (
                self.selected_overview_inventory_sha256
            ),
            "source_overview_file_inventory_sha256": (
                self.source_overview_file_inventory_sha256
            ),
            "raw_symbol_count": self.raw_symbol_count,
            "mapped_risky_action_count": self.mapped_risky_action_count,
            "unused_raw_symbols": list(self.unused_raw_symbols),
            "selected_snapshot_count": self.selected_snapshot_count,
            "explicit_unknown_state_asset_count": (
                self.explicit_unknown_state_asset_count
            ),
            "future_delta_rows_consumed": self.future_delta_rows_consumed,
            "exposure_receipt_sha256": self.exposures.receipt_sha256,
            "source_inventory_sha256": self.readiness.source_inventory_sha256,
            "readiness_receipt_sha256": self.readiness.receipt_sha256,
            "readiness_blockers": list(self.readiness.blocker_codes),
            "research_only": True,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }

    def validate(self) -> None:
        self.exposures.validate()
        self.inventory.validate()
        self.readiness.validate()
        for name in (
            "cache_sha256",
            "cache_identity",
            "action_hash",
            "polygon_validation_receipt_file_sha256",
            "polygon_validation_receipt_payload_sha256",
            "sector_receipt_sha256",
            "active_beta_receipt_sha256",
            "style_risk_receipt_sha256",
            "origin_availability_receipt_sha256",
            "selected_overview_inventory_sha256",
            "source_overview_file_inventory_sha256",
            "receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.schema != M03R_V9_RISK_MATERIALIZATION_SCHEMA
            or self.inventory.asset_axis_sha256 != self.action_hash
            or self.inventory.point_in_time_sector_receipt_sha256
            != self.sector_receipt_sha256
            or self.inventory.point_in_time_style_risk_receipt_sha256
            != self.style_risk_receipt_sha256
            or self.inventory.point_in_time_active_beta_receipt_sha256
            != self.active_beta_receipt_sha256
            or self.inventory.origin_availability_receipt_sha256
            != self.origin_availability_receipt_sha256
            or self.exposures.asset_axis_sha256 != self.action_hash
            or self.mapped_risky_action_count
            != self.exposures.exposure_loadings.shape[1] - 1
            or self.future_delta_rows_consumed != 0
            or self.receipt_sha256 != _canonical_sha256(self.canonical_payload())
        ):
            raise M03RV9RiskMaterializationError(
                "materialized risk-source identity or semantics drifted"
            )


@dataclass(frozen=True, slots=True)
class M03RV9WrittenRiskSource:
    artifact_path: Path
    artifact_file_sha256: str
    manifest_path: Path
    manifest_file_sha256: str


@dataclass(frozen=True, slots=True)
class _OverviewRecord:
    asof_date: str
    available_timestamp_ms: int
    sic_code: str | None
    record_sha256: str
    source_layer: str


@dataclass(frozen=True, slots=True)
class _RawSymbolSource:
    label: str
    base_root: Path
    delta_root: Path
    delta_member: bool


def _load_polygon_validation(
    inputs: M03RV9PolygonRiskInputs,
) -> tuple[dict[str, Any], dict[str, _RawSymbolSource]]:
    receipt_path = _require_regular(
        inputs.validation_receipt_path, name="Polygon validation receipt"
    )
    expected = _require_digest(
        "validation_receipt_file_sha256", inputs.validation_receipt_file_sha256
    )
    if _file_sha256(receipt_path) != expected:
        raise M03RV9RiskMaterializationError("Polygon validation receipt hash mismatch")
    payload = json.loads(receipt_path.read_text())
    if (
        payload.get("schema") != M03R_V9_POLYGON_VALIDATION_SCHEMA
        or payload.get("checks", {}).get("manifest_key_coverage") is not True
        or payload.get("checks", {}).get("file_sha256_match") is not True
        or payload.get("checks", {}).get("row_counts_match") is not True
        or payload.get("checks", {}).get("regular_non_symlink_files") is not True
        or payload.get("checks", {}).get("event_dates_within_requested_window")
        is not True
        or payload.get("checks", {}).get("download_failures") != 0
        or payload.get("checks", {}).get("transient_api_failures") != 0
        or payload.get("research_only") is not True
        or payload.get("development_only") is not True
        or payload.get("reportable") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise M03RV9RiskMaterializationError("Polygon validation receipt is incomplete")
    unsigned = dict(payload)
    claimed_payload_hash = unsigned.pop("canonical_payload_sha256", None)
    if claimed_payload_hash != _canonical_sha256(unsigned):
        raise M03RV9RiskMaterializationError(
            "Polygon validation receipt payload hash mismatch"
        )

    bound: dict[str, _RawSymbolSource] = {}
    validated_slices: list[tuple[M03RV9PolygonRiskSlice, Path, Path]] = []
    receipt_roots = {
        row.get("slice"): row
        for row in payload.get("roots", [])
        if isinstance(row, dict) and isinstance(row.get("slice"), str)
    }
    all_symbols: set[str] = set()
    for item in inputs.slices:
        if not item.label or item.label in bound:
            raise M03RV9RiskMaterializationError("Polygon slice labels must be unique")
        universe = _require_regular(item.universe_path, name=f"{item.label} universe")
        base = _require_directory(item.base_root, name=f"{item.label} base root")
        delta = _require_directory(item.delta_root, name=f"{item.label} delta root")
        symbols = _read_universe(universe)
        overlap = all_symbols.intersection(symbols)
        if overlap:
            raise M03RV9RiskMaterializationError(
                f"Polygon universe slices overlap: {sorted(overlap)[:3]}"
            )
        all_symbols.update(symbols)
        receipt_row = receipt_roots.get(item.label)
        if (
            receipt_row is None
            or receipt_row.get("symbols") != len(symbols)
            or receipt_row.get("universe_sha256") != _file_sha256(universe)
            or Path(str(receipt_row.get("base_root"))).resolve() != base
            or Path(str(receipt_row.get("delta_root"))).resolve() != delta
            or receipt_row.get("base_dataset_manifest_sha256")
            != _file_sha256(base / "dataset_manifest.json")
            or receipt_row.get("base_manifest_sha256")
            != _file_sha256(base / "manifest.csv")
            or receipt_row.get("delta_dataset_manifest_sha256")
            != _file_sha256(delta / "dataset_manifest.json")
            or receipt_row.get("delta_manifest_sha256")
            != _file_sha256(delta / "manifest.csv")
        ):
            raise M03RV9RiskMaterializationError(
                f"Polygon slice {item.label!r} disagrees with its validation receipt"
            )
        for symbol in symbols:
            bound[symbol] = _RawSymbolSource(item.label, base, delta, True)
        validated_slices.append((item, base, delta))
    if payload.get("combined_symbols") != len(all_symbols) or payload.get(
        "combined_unique_symbols"
    ) != len(all_symbols):
        raise M03RV9RiskMaterializationError("combined Polygon symbol count drifted")
    # The verified cache may retain a listed equity that left the current
    # future-selected universe.  Its historical base rows remain required for
    # the pre-2026 action axis, while a post-cache delta is neither required nor
    # permitted to fill that historical surface.  Discover only overview rows
    # named in the already hash-bound base manifest; never infer from a broad
    # directory scan.
    for item, base, delta in validated_slices:
        with (base / "manifest.csv").open(newline="") as source:
            for row in csv.DictReader(source):
                symbol = str(row.get("symbol") or "").upper()
                if (
                    row.get("dataset") == "overview_snapshots"
                    and row.get("status") == "downloaded"
                    and symbol
                    and not symbol.startswith("__")
                    and symbol not in bound
                ):
                    bound[symbol] = _RawSymbolSource(item.label, base, delta, False)
    return payload, bound


def _action_symbol_candidates(action: str) -> tuple[str, ...]:
    source_action = action.removeprefix("EQUITY:")
    return tuple(
        dict.fromkeys(
            (
                source_action,
                source_action.replace(".", "-"),
                source_action.replace("-", "."),
            )
        )
    )


def _resolve_action_symbols(
    action_ids: tuple[str, ...],
    raw_symbols: Mapping[str, _RawSymbolSource],
) -> tuple[dict[str, str], tuple[str, ...]]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for action in action_ids[1:]:
        candidates = [
            candidate
            for candidate in _action_symbol_candidates(action)
            if candidate in raw_symbols
        ]
        if len(candidates) != 1:
            raise M03RV9RiskMaterializationError(
                f"action {action!r} resolves to {len(candidates)} Polygon symbols"
            )
        mapping[action] = candidates[0]
        used.add(candidates[0])
    return mapping, tuple(sorted(set(raw_symbols) - used))


def _read_overview_records(
    *,
    action: str,
    raw_symbol: str,
    base_root: Path,
    delta_root: Path,
    delta_member: bool,
) -> tuple[list[_OverviewRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[_OverviewRecord] = []
    file_inventory: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    allowed_tickers = set(_action_symbol_candidates(raw_symbol)) | set(
        _action_symbol_candidates(action)
    )
    layers = [("base", base_root)]
    if delta_member:
        layers.append(("delta", delta_root))
    for source_layer, root in layers:
        path = root / raw_symbol / "overview_snapshots.jsonl"
        regular = _require_regular(path, name=f"{action} {source_layer} overview")
        file_inventory.append(
            {
                "action": action,
                "raw_symbol": raw_symbol,
                "source_layer": source_layer,
                "file_sha256": _file_sha256(regular),
                "size_bytes": regular.stat().st_size,
            }
        )
        for line_number, payload in iter_jsonl_payloads(regular):
            asof = payload.get("asof_date")
            if not isinstance(asof, str):
                raise M03RV9RiskMaterializationError(
                    f"{action} overview row {line_number} lacks asof_date"
                )
            if payload.get("record_available") is not True:
                unavailable.append(
                    {
                        "action": action,
                        "source_layer": source_layer,
                        "asof_date": asof,
                        "error_sha256": _canonical_sha256(
                            str(payload.get("error", ""))
                        ),
                    }
                )
                continue
            ticker = str(payload.get("ticker") or raw_symbol).upper()
            if ticker not in allowed_tickers:
                raise M03RV9RiskMaterializationError(
                    f"{action} overview ticker {ticker!r} disagrees with its source axis"
                )
            _event, available = infer_covariate_timestamps_ms(
                "overview_snapshots", payload
            )
            sic = str(payload.get("sic_code") or "").strip()
            records.append(
                _OverviewRecord(
                    asof_date=asof,
                    available_timestamp_ms=available,
                    sic_code=sic or None,
                    record_sha256=_canonical_sha256(payload),
                    source_layer=source_layer,
                )
            )
    records.sort(
        key=lambda row: (row.available_timestamp_ms, row.asof_date, row.record_sha256)
    )
    if any(
        left.available_timestamp_ms >= right.available_timestamp_ms
        for left, right in pairwise(records)
    ):
        raise M03RV9RiskMaterializationError(
            f"{action} overview availability timestamps are not strictly increasing"
        )
    return records, file_inventory, unavailable


def _sic_sector_index(value: str | None) -> int:
    if value is None or not value.isdigit():
        return len(M03R_V9_SECTOR_EXPOSURE_NAMES) - 1
    code = int(value)
    ranges = (
        (100, 999),
        (1000, 1499),
        (1500, 1799),
        (2000, 3999),
        (4000, 4999),
        (5000, 5199),
        (5200, 5999),
        (6000, 6799),
        (7000, 8999),
        (9100, 9999),
    )
    for index, (lower, upper) in enumerate(ranges):
        if lower <= code <= upper:
            return index
    return len(M03R_V9_SECTOR_EXPOSURE_NAMES) - 1


def _decision_timestamps(exchange_dates: tuple[str, ...]) -> torch.Tensor:
    values: list[int] = []
    parsed_dates: list[date] = []
    for value in exchange_dates:
        parsed = date.fromisoformat(value)
        parsed_dates.append(parsed)
        timestamp = datetime.combine(
            parsed, M03R_V9_DECISION_TIME, tzinfo=M03R_V9_EXCHANGE_TIMEZONE
        )
        values.append(int(timestamp.astimezone(UTC).timestamp() * 1000))
    if any(left >= right for left, right in pairwise(parsed_dates)):
        raise M03RV9RiskMaterializationError("cache exchange dates are not increasing")
    return torch.tensor(values, dtype=torch.int64)


def _rolling_sum(values: torch.Tensor, window: int) -> torch.Tensor:
    if values.ndim != 2 or window <= 0:
        raise M03RV9RiskMaterializationError("rolling input or window is invalid")
    prefix = torch.cat(
        (torch.zeros((1, values.shape[1]), dtype=values.dtype), values.cumsum(dim=0)),
        dim=0,
    )
    stop = torch.arange(1, values.shape[0] + 1, dtype=torch.long)
    start = (stop - window).clamp_min(0)
    return prefix.index_select(0, stop) - prefix.index_select(0, start)


def _standardize_cross_section(
    values: torch.Tensor, qualified: torch.Tensor
) -> torch.Tensor:
    result = torch.zeros_like(values, dtype=torch.float32)
    for row in range(values.shape[0]):
        selected = qualified[row]
        if int(selected.sum()) < 2:
            continue
        current = values[row, selected]
        mean = current.mean()
        scale = current.std(unbiased=False)
        if not bool(torch.isfinite(scale)) or float(scale) < 1.0e-12:
            continue
        result[row, selected] = ((current - mean) / scale).to(torch.float32)
    return result


def _cache_factor_loadings(
    cache: Top2000VerifiedDevelopmentCache,
    decision_timestamp_ms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache.validate_unmodified()
    daily = cache.daily_ohlcv.to(torch.float64)
    available = cache.availability
    close = daily[..., 3]
    volume = daily[..., 4].clamp_min(0.0)
    pair_valid = available[:-1] & available[1:] & (close[:-1] > 0.0) & (close[1:] > 0.0)
    returns = torch.zeros_like(close)
    returns[1:] = torch.where(
        pair_valid,
        close[1:] / close[:-1] - 1.0,
        torch.zeros_like(close[1:]),
    )
    returns[:, 0] = 0.0
    valid = torch.zeros_like(available)
    valid[1:] = pair_valid
    valid[:, 0] = False
    risky_valid = valid[:, 1:]
    risky_returns = returns[:, 1:]
    market_count = risky_valid.sum(dim=1).clamp_min(1)
    market = (risky_returns * risky_valid).sum(dim=1) / market_count

    valid_float = valid.to(torch.float64)
    count = _rolling_sum(valid_float, M03R_V9_LOOKBACK_SESSIONS)
    sum_x = _rolling_sum(returns * valid_float, M03R_V9_LOOKBACK_SESSIONS)
    sum_x2 = _rolling_sum(returns.square() * valid_float, M03R_V9_LOOKBACK_SESSIONS)
    market_grid = market[:, None]
    sum_m = _rolling_sum(market_grid * valid_float, M03R_V9_LOOKBACK_SESSIONS)
    sum_m2 = _rolling_sum(market_grid.square() * valid_float, M03R_V9_LOOKBACK_SESSIONS)
    sum_xm = _rolling_sum(
        returns * market_grid * valid_float, M03R_V9_LOOKBACK_SESSIONS
    )
    safe_count = count.clamp_min(1.0)
    covariance_numerator = sum_xm - sum_x * sum_m / safe_count
    market_variance_numerator = sum_m2 - sum_m.square() / safe_count
    beta = torch.where(
        market_variance_numerator > 1.0e-12,
        covariance_numerator / market_variance_numerator.clamp_min(1.0e-12),
        torch.zeros_like(covariance_numerator),
    )
    momentum = _rolling_sum(
        torch.log1p(returns.clamp_min(-0.999999)) * valid_float,
        M03R_V9_LOOKBACK_SESSIONS,
    )
    variance = (sum_x2 / safe_count - (sum_x / safe_count).square()).clamp_min(0.0)
    volatility = variance.sqrt()

    volume_valid = available & torch.isfinite(volume)
    volume_valid[:, 0] = False
    volume_count = _rolling_sum(
        volume_valid.to(torch.float64), M03R_V9_LOOKBACK_SESSIONS
    )
    volume_mean = _rolling_sum(
        torch.log1p(volume) * volume_valid,
        M03R_V9_LOOKBACK_SESSIONS,
    ) / volume_count.clamp_min(1.0)
    qualified = (
        available
        & (count >= M03R_V9_MINIMUM_HISTORY_SESSIONS)
        & (volume_count >= M03R_V9_MINIMUM_HISTORY_SESSIONS)
        & (market_variance_numerator > 1.0e-12)
    )
    qualified[:, 0] = False
    beta_z = _standardize_cross_section(beta, qualified)
    momentum_z = _standardize_cross_section(momentum, qualified)
    volatility_z = _standardize_cross_section(volatility, qualified)
    volume_z = _standardize_cross_section(volume_mean, qualified)
    factors = torch.stack((beta_z, momentum_z, volatility_z, volume_z), dim=-1)
    factor_available = torch.where(
        qualified,
        decision_timestamp_ms[:, None],
        torch.zeros_like(decision_timestamp_ms[:, None]),
    )
    return factors, qualified, factor_available


def materialize_top2000_m03r_v9_risk_source(
    cache: Top2000VerifiedDevelopmentCache,
    inputs: M03RV9PolygonRiskInputs,
) -> M03RV9MaterializedRiskSource:
    """Build one pre-2026, asset-aligned, causal risk-source surface."""

    if not isinstance(cache, Top2000VerifiedDevelopmentCache):
        raise M03RV9RiskMaterializationError(
            "risk materialization needs a verified cache"
        )
    cache.validate_unmodified()
    if _axis_sha256(cache.action_ids) != cache.action_hash:
        raise M03RV9RiskMaterializationError(
            "verified cache action axis and action hash disagree"
        )
    validation, raw_symbols = _load_polygon_validation(inputs)
    action_map, unused_raw_symbols = _resolve_action_symbols(
        cache.action_ids, raw_symbols
    )
    decisions = _decision_timestamps(cache.exchange_dates)
    states = len(cache.exchange_dates)
    assets = len(cache.action_ids)
    sector_count = len(M03R_V9_SECTOR_EXPOSURE_NAMES)
    exposure_count = len(M03R_V9_PROJECTOR_EXPOSURE_NAMES)
    loadings = torch.zeros((states, assets, 1 + exposure_count), dtype=torch.float32)
    loadings[:, 1:, 0] = 1.0
    sector_available = torch.zeros((states, assets), dtype=torch.int64)
    selected_inventory_digest = hashlib.sha256()
    source_file_inventory: list[dict[str, Any]] = []
    unavailable_inventory: list[dict[str, Any]] = []
    explicit_unknown = 0
    selected_snapshot_keys: set[tuple[str, str, str]] = set()
    future_delta_rows_consumed = 0

    unknown_index = sector_count - 1
    for asset_index, action in enumerate(cache.action_ids[1:], start=1):
        raw_symbol = action_map[action]
        source = raw_symbols[raw_symbol]
        records, file_rows, unavailable_rows = _read_overview_records(
            action=action,
            raw_symbol=raw_symbol,
            base_root=source.base_root,
            delta_root=source.delta_root,
            delta_member=source.delta_member,
        )
        source_file_inventory.extend(file_rows)
        unavailable_inventory.extend(unavailable_rows)
        cursor = 0
        selected: _OverviewRecord | None = None
        for state_index, decision in enumerate(decisions.tolist()):
            while (
                cursor < len(records)
                and records[cursor].available_timestamp_ms <= decision
            ):
                selected = records[cursor]
                cursor += 1
            sector_index = unknown_index
            available_time = 0
            if selected is not None:
                sector_index = _sic_sector_index(selected.sic_code)
                available_time = selected.available_timestamp_ms
                selected_snapshot_keys.add(
                    (action, selected.asof_date, selected.record_sha256)
                )
                if selected.source_layer == "delta":
                    future_delta_rows_consumed += 1
                selected_inventory_digest.update(
                    json.dumps(
                        {
                            "state_index": state_index,
                            "action": action,
                            "asof_date": selected.asof_date,
                            "available_timestamp_ms": available_time,
                            "sic_code": selected.sic_code,
                            "sector_name": M03R_V9_SECTOR_EXPOSURE_NAMES[sector_index],
                            "record_sha256": selected.record_sha256,
                            "source_layer": selected.source_layer,
                        },
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                )
                selected_inventory_digest.update(b"\n")
            if sector_index == unknown_index:
                explicit_unknown += 1
            loadings[state_index, asset_index, 1 + sector_index] = 1.0
            sector_available[state_index, asset_index] = available_time

    factor_values, qualified, factor_available = _cache_factor_loadings(
        cache, decisions
    )
    active_beta_column = 1 + sector_count
    style_start = active_beta_column + 1
    loadings[:, :, active_beta_column] = factor_values[:, :, 0]
    loadings[:, :, style_start : style_start + 3] = factor_values[:, :, 1:]
    regression_weights = qualified.to(torch.float32)
    family_available = torch.zeros(
        (states, assets, len(M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES)),
        dtype=torch.int64,
    )
    family_available[:, :, 0] = sector_available
    family_available[:, :, 1] = factor_available
    family_available[:, :, 2] = factor_available

    selected_inventory_hash = selected_inventory_digest.hexdigest()
    source_file_inventory_hash = _canonical_sha256(
        sorted(
            source_file_inventory, key=lambda row: (row["action"], row["source_layer"])
        )
    )
    sector_tensor_hash = _tensor_sha256(loadings[:, :, 1 : 1 + sector_count])
    beta_tensor_hash = _tensor_sha256(loadings[:, :, active_beta_column])
    style_tensor_hash = _tensor_sha256(loadings[:, :, style_start : style_start + 3])
    sector_receipt = _canonical_sha256(
        {
            "schema": "rl-quant.top2000-dev.m03r-v9-sic-sector-source-v1",
            "taxonomy": "SEC-SIC-major-division-v1-with-explicit-unknown",
            "sector_exposure_names": M03R_V9_SECTOR_EXPOSURE_NAMES,
            "source_validation_receipt_file_sha256": (
                inputs.validation_receipt_file_sha256
            ),
            "source_overview_file_inventory_sha256": source_file_inventory_hash,
            "selected_overview_inventory_sha256": selected_inventory_hash,
            "sector_tensor_sha256": sector_tensor_hash,
            "sector_availability_sha256": _tensor_sha256(sector_available),
            "unknown_state_asset_count": explicit_unknown,
            "unavailable_record_inventory_sha256": _canonical_sha256(
                unavailable_inventory
            ),
            "future_backfill_forbidden": True,
        }
    )
    active_beta_receipt = _canonical_sha256(
        {
            "schema": "rl-quant.top2000-dev.m03r-v9-active-beta-source-v1",
            "name": M03R_V9_ACTIVE_BETA_EXPOSURE_NAME,
            "cache_sha256": cache.cache_sha256,
            "action_hash": cache.action_hash,
            "lookback_sessions": M03R_V9_LOOKBACK_SESSIONS,
            "minimum_history_sessions": M03R_V9_MINIMUM_HISTORY_SESSIONS,
            "benchmark": "same-transition-equal-weight-available-risky-C1-proxy",
            "cross_sectional_standardization": "population-zscore-per-origin",
            "tensor_sha256": beta_tensor_hash,
        }
    )
    style_receipt = _canonical_sha256(
        {
            "schema": "rl-quant.top2000-dev.m03r-v9-style-risk-source-v1",
            "names": M03R_V9_STYLE_RISK_EXPOSURE_NAMES,
            "cache_sha256": cache.cache_sha256,
            "action_hash": cache.action_hash,
            "lookback_sessions": M03R_V9_LOOKBACK_SESSIONS,
            "minimum_history_sessions": M03R_V9_MINIMUM_HISTORY_SESSIONS,
            "formulas": (
                "sum-log1p-close-to-close-returns",
                "population-standard-deviation-close-to-close-returns",
                "mean-log1p-volume",
            ),
            "cross_sectional_standardization": "population-zscore-per-origin",
            "tensor_sha256": style_tensor_hash,
        }
    )
    origin_receipt = _canonical_sha256(
        {
            "schema": "rl-quant.top2000-dev.m03r-v9-origin-availability-v1",
            "decision_timestamp_rule": "exchange-date-16:00-America/New_York",
            "decision_timestamp_sha256": _tensor_sha256(decisions),
            "availability_family_names": M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES,
            "family_availability_sha256": _tensor_sha256(family_available),
            "all_available_no_later_than_decision": bool(
                (family_available <= decisions[:, None, None]).all()
            ),
        }
    )
    source_receipt = _canonical_sha256(
        {
            "schema": M03R_V9_RISK_MATERIALIZATION_SCHEMA,
            "protocol_sha256": M03R_V9_PROTOCOL_SHA256,
            "cache_sha256": cache.cache_sha256,
            "cache_identity": cache.cache_identity,
            "action_hash": cache.action_hash,
            "sector_receipt_sha256": sector_receipt,
            "active_beta_receipt_sha256": active_beta_receipt,
            "style_risk_receipt_sha256": style_receipt,
            "origin_availability_receipt_sha256": origin_receipt,
        }
    )
    exposures = qualify_m03r_v9_origin_risk_exposures(
        state_start_index=0,
        cash_index=0,
        projector_exposure_names=M03R_V9_PROJECTOR_EXPOSURE_NAMES,
        projector_exposure_families=M03R_V9_PROJECTOR_EXPOSURE_FAMILIES,
        availability_family_names=M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES,
        asset_axis_sha256=cache.action_hash,
        source_receipt_sha256=source_receipt,
        exposure_loadings=loadings,
        regression_weights=regression_weights,
        decision_timestamp_ms=decisions,
        exposure_available_timestamp_ms=family_available,
    )
    inventory = M03RV9RiskSourceInventory(
        source_id="polygon-sic-plus-verified-top2000-daily-cache-v1",
        source_schema_sha256=_canonical_sha256(
            {
                "sector_names": M03R_V9_SECTOR_EXPOSURE_NAMES,
                "active_beta_name": M03R_V9_ACTIVE_BETA_EXPOSURE_NAME,
                "style_names": M03R_V9_STYLE_RISK_EXPOSURE_NAMES,
                "availability_family_names": M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES,
                "lookback_sessions": M03R_V9_LOOKBACK_SESSIONS,
                "minimum_history_sessions": M03R_V9_MINIMUM_HISTORY_SESSIONS,
            }
        ),
        asset_axis_sha256=cache.action_hash,
        source_columns=(
            "overview_snapshots.asof_date",
            "overview_snapshots.record_available",
            "overview_snapshots.sic_code",
            "cache.daily_ohlcv.close",
            "cache.daily_ohlcv.volume",
            "cache.availability",
        ),
        sector_exposure_names=M03R_V9_SECTOR_EXPOSURE_NAMES,
        style_risk_exposure_names=M03R_V9_STYLE_RISK_EXPOSURE_NAMES,
        active_beta_exposure_name=M03R_V9_ACTIVE_BETA_EXPOSURE_NAME,
        point_in_time_sector_receipt_sha256=sector_receipt,
        point_in_time_style_risk_receipt_sha256=style_receipt,
        point_in_time_active_beta_receipt_sha256=active_beta_receipt,
        origin_availability_receipt_sha256=origin_receipt,
        projector_manifest_sha256=None,
        target_projector_exposure_names_match=False,
    )
    readiness = audit_m03r_v9_risk_source(inventory)
    unsigned = {
        "exposures": exposures,
        "inventory": inventory,
        "readiness": readiness,
        "cache_sha256": cache.cache_sha256,
        "cache_identity": cache.cache_identity,
        "action_hash": cache.action_hash,
        "first_exchange_date": cache.exchange_dates[0],
        "last_exchange_date": cache.exchange_dates[-1],
        "polygon_validation_receipt_file_sha256": inputs.validation_receipt_file_sha256,
        "polygon_validation_receipt_payload_sha256": validation[
            "canonical_payload_sha256"
        ],
        "sector_receipt_sha256": sector_receipt,
        "active_beta_receipt_sha256": active_beta_receipt,
        "style_risk_receipt_sha256": style_receipt,
        "origin_availability_receipt_sha256": origin_receipt,
        "selected_overview_inventory_sha256": selected_inventory_hash,
        "source_overview_file_inventory_sha256": source_file_inventory_hash,
        "raw_symbol_count": len(raw_symbols),
        "mapped_risky_action_count": len(action_map),
        "unused_raw_symbols": unused_raw_symbols,
        "selected_snapshot_count": len(selected_snapshot_keys),
        "explicit_unknown_state_asset_count": explicit_unknown,
        "future_delta_rows_consumed": future_delta_rows_consumed,
    }
    provisional = M03RV9MaterializedRiskSource(
        **unsigned,
        receipt_sha256="0" * 64,
    )
    result = M03RV9MaterializedRiskSource(
        **unsigned,
        receipt_sha256=_canonical_sha256(provisional.canonical_payload()),
    )
    result.validate()
    return result


def _artifact_payload(result: M03RV9MaterializedRiskSource) -> dict[str, Any]:
    result.validate()
    return {
        "schema": M03R_V9_RISK_ARTIFACT_SCHEMA,
        "materialization": result.canonical_payload(),
        "materialization_receipt_sha256": result.receipt_sha256,
        "inventory": asdict(result.inventory),
        "readiness": asdict(result.readiness),
        "exposures": {
            "schema": result.exposures.schema,
            "state_start_index": result.exposures.state_start_index,
            "cash_index": result.exposures.cash_index,
            "projector_exposure_names": result.exposures.projector_exposure_names,
            "projector_exposure_families": result.exposures.projector_exposure_families,
            "availability_family_names": result.exposures.availability_family_names,
            "asset_axis_sha256": result.exposures.asset_axis_sha256,
            "source_receipt_sha256": result.exposures.source_receipt_sha256,
            "exposure_loadings": result.exposures.exposure_loadings,
            "regression_weights": result.exposures.regression_weights,
            "decision_timestamp_ms": result.exposures.decision_timestamp_ms,
            "exposure_available_timestamp_ms": (
                result.exposures.exposure_available_timestamp_ms
            ),
            "tensor_sha256": result.exposures.tensor_sha256,
            "receipt_sha256": result.exposures.receipt_sha256,
        },
    }


def write_top2000_m03r_v9_risk_source(
    result: M03RV9MaterializedRiskSource,
    output_root: Path,
) -> M03RV9WrittenRiskSource:
    """Publish one no-clobber tensor artifact and its external file binding."""

    result.validate()
    output_root.mkdir(mode=0o750, parents=True, exist_ok=False)
    artifact = output_root / "risk-exposures.pt"
    artifact_tmp = output_root / ".risk-exposures.pt.tmp"
    manifest = output_root / "risk-source-manifest.json"
    torch.save(_artifact_payload(result), artifact_tmp)
    os.chmod(artifact_tmp, 0o640)
    os.replace(artifact_tmp, artifact)
    artifact_sha = _file_sha256(artifact)
    manifest_payload = {
        "schema": M03R_V9_RISK_ARTIFACT_MANIFEST_SCHEMA,
        "artifact_name": artifact.name,
        "artifact_file_sha256": artifact_sha,
        "artifact_size_bytes": artifact.stat().st_size,
        "materialization_receipt_sha256": result.receipt_sha256,
        "exposure_receipt_sha256": result.exposures.receipt_sha256,
        "source_inventory_sha256": result.readiness.source_inventory_sha256,
        "readiness_receipt_sha256": result.readiness.receipt_sha256,
        "readiness_blockers": list(result.readiness.blocker_codes),
        "predictive_worker_authorized": result.readiness.predictive_worker_authorized,
        "research_only": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    manifest_payload["receipt_sha256"] = _canonical_sha256(manifest_payload)
    with manifest.open("x") as sink:
        json.dump(manifest_payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.chmod(manifest, 0o640)
    return M03RV9WrittenRiskSource(
        artifact_path=artifact,
        artifact_file_sha256=artifact_sha,
        manifest_path=manifest,
        manifest_file_sha256=_file_sha256(manifest),
    )


def load_top2000_m03r_v9_risk_source(
    manifest_path: Path,
    *,
    expected_manifest_file_sha256: str,
) -> tuple[M03RV9MaterializedRiskSource, M03RV9WrittenRiskSource]:
    """Fail closed while loading one externally hash-bound local artifact."""

    manifest = _require_regular(manifest_path, name="risk-source manifest")
    expected = _require_digest(
        "expected_manifest_file_sha256", expected_manifest_file_sha256
    )
    manifest_file_sha = _file_sha256(manifest)
    if manifest_file_sha != expected:
        raise M03RV9RiskMaterializationError("risk-source manifest file hash mismatch")
    manifest_payload = json.loads(manifest.read_text())
    claimed_manifest_receipt = manifest_payload.pop("receipt_sha256", None)
    if (
        manifest_payload.get("schema") != M03R_V9_RISK_ARTIFACT_MANIFEST_SCHEMA
        or claimed_manifest_receipt != _canonical_sha256(manifest_payload)
    ):
        raise M03RV9RiskMaterializationError("risk-source manifest receipt drifted")
    artifact_name = manifest_payload.get("artifact_name")
    if (
        not isinstance(artifact_name, str)
        or Path(artifact_name).name != artifact_name
        or artifact_name != "risk-exposures.pt"
    ):
        raise M03RV9RiskMaterializationError("risk artifact name is unsafe")
    artifact = _require_regular(manifest.parent / artifact_name, name="risk artifact")
    artifact_sha = _file_sha256(artifact)
    if artifact_sha != manifest_payload.get(
        "artifact_file_sha256"
    ) or artifact.stat().st_size != manifest_payload.get("artifact_size_bytes"):
        raise M03RV9RiskMaterializationError("risk artifact file binding drifted")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != M03R_V9_RISK_ARTIFACT_SCHEMA
    ):
        raise M03RV9RiskMaterializationError("risk artifact payload schema drifted")
    exposure_payload = payload.get("exposures")
    if not isinstance(exposure_payload, dict):
        raise M03RV9RiskMaterializationError("risk artifact omits exposure payload")
    exposures = M03RV9OriginRiskExposures(**exposure_payload)
    inventory_payload = payload.get("inventory")
    readiness_payload = payload.get("readiness")
    if not isinstance(inventory_payload, dict) or not isinstance(
        readiness_payload, dict
    ):
        raise M03RV9RiskMaterializationError("risk artifact omits source receipts")
    inventory = M03RV9RiskSourceInventory(**inventory_payload)
    readiness = M03RV9RiskSourceReadiness(**readiness_payload)
    materialization = payload.get("materialization")
    if not isinstance(materialization, dict):
        raise M03RV9RiskMaterializationError(
            "risk artifact omits materialization receipt"
        )
    result = M03RV9MaterializedRiskSource(
        exposures=exposures,
        inventory=inventory,
        readiness=readiness,
        cache_sha256=materialization["cache_sha256"],
        cache_identity=materialization["cache_identity"],
        action_hash=materialization["action_hash"],
        first_exchange_date=materialization["first_exchange_date"],
        last_exchange_date=materialization["last_exchange_date"],
        polygon_validation_receipt_file_sha256=materialization[
            "polygon_validation_receipt_file_sha256"
        ],
        polygon_validation_receipt_payload_sha256=materialization[
            "polygon_validation_receipt_payload_sha256"
        ],
        sector_receipt_sha256=materialization["sector_receipt_sha256"],
        active_beta_receipt_sha256=materialization["active_beta_receipt_sha256"],
        style_risk_receipt_sha256=materialization["style_risk_receipt_sha256"],
        origin_availability_receipt_sha256=materialization[
            "origin_availability_receipt_sha256"
        ],
        selected_overview_inventory_sha256=materialization[
            "selected_overview_inventory_sha256"
        ],
        source_overview_file_inventory_sha256=materialization[
            "source_overview_file_inventory_sha256"
        ],
        raw_symbol_count=materialization["raw_symbol_count"],
        mapped_risky_action_count=materialization["mapped_risky_action_count"],
        unused_raw_symbols=tuple(materialization["unused_raw_symbols"]),
        selected_snapshot_count=materialization["selected_snapshot_count"],
        explicit_unknown_state_asset_count=materialization[
            "explicit_unknown_state_asset_count"
        ],
        future_delta_rows_consumed=materialization["future_delta_rows_consumed"],
        receipt_sha256=payload["materialization_receipt_sha256"],
    )
    result.validate()
    if (
        result.receipt_sha256 != manifest_payload["materialization_receipt_sha256"]
        or result.exposures.receipt_sha256
        != manifest_payload["exposure_receipt_sha256"]
        or result.readiness.source_inventory_sha256
        != manifest_payload["source_inventory_sha256"]
        or result.readiness.receipt_sha256
        != manifest_payload["readiness_receipt_sha256"]
        or list(result.readiness.blocker_codes)
        != manifest_payload["readiness_blockers"]
        or result.readiness.predictive_worker_authorized
        != manifest_payload["predictive_worker_authorized"]
    ):
        raise M03RV9RiskMaterializationError("risk artifact and manifest disagree")
    written = M03RV9WrittenRiskSource(
        artifact_path=artifact,
        artifact_file_sha256=artifact_sha,
        manifest_path=manifest,
        manifest_file_sha256=manifest_file_sha,
    )
    return result, written


__all__ = [
    "M03R_V9_ACTIVE_BETA_EXPOSURE_NAME",
    "M03R_V9_LOOKBACK_SESSIONS",
    "M03R_V9_MINIMUM_HISTORY_SESSIONS",
    "M03R_V9_PROJECTOR_EXPOSURE_FAMILIES",
    "M03R_V9_PROJECTOR_EXPOSURE_NAMES",
    "M03R_V9_RISK_ARTIFACT_MANIFEST_SCHEMA",
    "M03R_V9_RISK_ARTIFACT_SCHEMA",
    "M03R_V9_RISK_MATERIALIZATION_SCHEMA",
    "M03R_V9_SECTOR_EXPOSURE_NAMES",
    "M03R_V9_STYLE_RISK_EXPOSURE_NAMES",
    "M03RV9MaterializedRiskSource",
    "M03RV9PolygonRiskInputs",
    "M03RV9PolygonRiskSlice",
    "M03RV9RiskMaterializationError",
    "M03RV9WrittenRiskSource",
    "load_top2000_m03r_v9_risk_source",
    "materialize_top2000_m03r_v9_risk_source",
    "write_top2000_m03r_v9_risk_source",
]
