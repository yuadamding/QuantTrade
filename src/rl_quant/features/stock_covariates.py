from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from rl_quant.data_sources.polygon_stock_covariates import RawCovariateRecord, canonical_symbol
from rl_quant.research_protocol import stable_json_hash


DAY_MS = 86_400_000
ACTION_COVARIATE_FEATURE_NAMES = [
    "log_market_cap",
    "log_share_class_shares_outstanding",
    "days_since_listed",
    "is_common_stock",
    "is_adr_or_foreign",
    "is_active_reference_record",
    "overview_age_seconds",
    "overview_missing_flag",
    "days_since_last_financial_filing",
    "financial_records_last_365d",
    "revenue_yoy_growth",
    "net_income_margin",
    "debt_to_assets",
    "cash_to_assets",
    "operating_cashflow_to_assets",
    "financial_age_seconds",
    "financial_missing_flag",
    "days_since_last_dividend",
    "trailing_12m_dividend_count",
    "trailing_12m_dividend_cash",
    "dividend_age_seconds",
    "dividend_missing_flag",
    "days_since_last_split",
    "split_events_last_365d",
    "split_age_seconds",
    "split_missing_flag",
    "news_count_1h",
    "news_count_1d",
    "news_count_7d",
    "news_count_30d",
    "news_publisher_count_1d",
    "news_publisher_count_7d",
    "news_age_seconds",
    "news_missing_flag",
    "covariate_group_coverage_fraction",
    "covariate_max_age_seconds",
]
ACTION_COVARIATE_SCHEMA_HASH = stable_json_hash(ACTION_COVARIATE_FEATURE_NAMES)
COVARIATE_GROUPS = ("overview_snapshots", "financials", "dividends", "splits", "news")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _log1p_nonnegative(value: Any) -> float:
    return math.log1p(max(_finite(value), 0.0))


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    denom = _finite(denominator)
    if abs(denom) <= 1e-12:
        return 0.0
    return max(min(_finite(numerator) / denom, 10.0), -10.0)


def _nested_financial_value(payload: Mapping[str, Any], statement: str, key: str) -> float:
    financials = payload.get("financials", {})
    if not isinstance(financials, Mapping):
        return 0.0
    statement_payload = financials.get(statement, {})
    if not isinstance(statement_payload, Mapping):
        return 0.0
    item = statement_payload.get(key, {})
    if not isinstance(item, Mapping):
        return 0.0
    return _finite(item.get("value"))


def _publisher_id(payload: Mapping[str, Any]) -> str:
    publisher = payload.get("publisher")
    if isinstance(publisher, Mapping):
        name = publisher.get("name")
        if name:
            return str(name)
    return ""


def _list_date_ms(payload: Mapping[str, Any]) -> int:
    value = payload.get("list_date")
    if not value:
        return -1
    from rl_quant.data_sources.polygon_stock_covariates import regular_session_open_ms_on_or_after

    return regular_session_open_ms_on_or_after(str(value))


def build_symbol_silver_rows(records: list[RawCovariateRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (item.available_timestamp_ms, item.source_dataset)):
        payload = record.payload
        row: dict[str, Any] = {
            "symbol": record.symbol,
            "source_dataset": record.source_dataset,
            "source_record_id": record.source_record_id,
            "source_record_hash": record.source_record_hash,
            "raw_payload_hash": record.source_record_hash,
            "event_timestamp_ms": int(record.event_timestamp_ms),
            "available_timestamp_ms": int(record.available_timestamp_ms),
        }
        if record.source_dataset == "overview_snapshots":
            row.update(
                {
                    "market_cap": _finite(payload.get("market_cap")),
                    "share_class_shares_outstanding": _finite(
                        payload.get("share_class_shares_outstanding", payload.get("weighted_shares_outstanding"))
                    ),
                    "list_date_ms": _list_date_ms(payload),
                    "is_common_stock": float(str(payload.get("type", "")).upper() == "CS"),
                    "is_adr_or_foreign": float(
                        "ADR" in str(payload.get("type", "")).upper()
                        or str(payload.get("locale", "")).lower() not in {"", "us"}
                    ),
                    "is_active_reference_record": float(bool(payload.get("record_available", payload.get("active", False)))),
                }
            )
        elif record.source_dataset == "financials":
            revenue = _nested_financial_value(payload, "income_statement", "revenues")
            net_income = _nested_financial_value(payload, "income_statement", "net_income_loss")
            assets = _nested_financial_value(payload, "balance_sheet", "assets")
            liabilities = _nested_financial_value(payload, "balance_sheet", "liabilities")
            cash = _nested_financial_value(payload, "balance_sheet", "cash_and_cash_equivalents")
            operating_cashflow = _nested_financial_value(
                payload,
                "cash_flow_statement",
                "net_cash_flow_from_operating_activities",
            )
            row.update(
                {
                    "financial_revenue": revenue,
                    "financial_net_income": net_income,
                    "financial_assets": assets,
                    "financial_liabilities": liabilities,
                    "financial_cash": cash,
                    "financial_operating_cashflow": operating_cashflow,
                    "financial_fiscal_year": int(_finite(payload.get("fiscal_year"), default=0.0)),
                    "financial_fiscal_period": str(payload.get("fiscal_period", "")),
                }
            )
        elif record.source_dataset == "dividends":
            row["dividend_cash_amount"] = _finite(payload.get("cash_amount"))
        elif record.source_dataset == "splits":
            split_from = _finite(payload.get("split_from"), default=0.0)
            split_to = _finite(payload.get("split_to"), default=0.0)
            row["split_ratio"] = _safe_ratio(split_to, split_from) if split_from else 0.0
        elif record.source_dataset == "news":
            row["news_publisher_id"] = _publisher_id(payload)
        rows.append(row)
    return rows


def read_silver_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict("records")


def load_silver_rows_by_symbol(root: Path, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        if symbol.upper() == "CASH":
            continue
        candidates = [root / f"{symbol}.parquet", root / f"{symbol.replace('.', '-')}.parquet"]
        for path in candidates:
            if path.exists():
                out[canonical_symbol(symbol)] = read_silver_rows(path)
                break
        else:
            out[canonical_symbol(symbol)] = []
    return out


def empty_source_coverage() -> dict[str, bool]:
    return {group: False for group in COVARIATE_GROUPS}


def read_covariate_coverage_manifest(path: Path) -> dict[str, dict[str, bool]]:
    if not path.exists():
        return {}
    coverage_by_symbol: dict[str, dict[str, bool]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            symbol = canonical_symbol(str(row.get("symbol", "")))
            if not symbol:
                continue
            available = {
                item.strip()
                for item in str(row.get("datasets_available", "")).split(",")
                if item.strip()
            }
            missing = {
                item.strip()
                for item in str(row.get("datasets_missing", "")).split(",")
                if item.strip()
            }
            coverage = {group: group in available for group in COVARIATE_GROUPS}
            for group in missing:
                if group in coverage:
                    coverage[group] = False
            coverage_by_symbol[symbol] = coverage
    return coverage_by_symbol


def validate_action_covariate_feature_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text())
    feature_names = list(schema.get("action_covariate_feature_names", []))
    schema_hash = schema.get("action_covariate_schema_hash")
    if feature_names != ACTION_COVARIATE_FEATURE_NAMES:
        raise ValueError("action covariate feature_schema.json feature names do not match ACTION_COVARIATE_FEATURE_NAMES.")
    if schema_hash != ACTION_COVARIATE_SCHEMA_HASH:
        raise ValueError("action covariate feature_schema.json hash does not match ACTION_COVARIATE_SCHEMA_HASH.")
    return schema


def _latest_before(
    rows: list[Mapping[str, Any]],
    dataset: str,
    decision_ms: int,
    *,
    max_age_ms: int | None = None,
) -> Mapping[str, Any] | None:
    selected = [
        row
        for row in rows
        if row.get("source_dataset") == dataset and int(row.get("available_timestamp_ms", -1)) <= decision_ms
        and (max_age_ms is None or int(row.get("available_timestamp_ms", -1)) >= decision_ms - max_age_ms)
    ]
    if not selected:
        return None
    return max(selected, key=lambda row: int(row.get("available_timestamp_ms", -1)))


def _rows_before(rows: list[Mapping[str, Any]], dataset: str, decision_ms: int) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("source_dataset") == dataset and int(row.get("available_timestamp_ms", -1)) <= decision_ms
    ]


def _count_recent(rows: list[Mapping[str, Any]], dataset: str, decision_ms: int, window_ms: int) -> list[Mapping[str, Any]]:
    lower = decision_ms - window_ms
    return [
        row
        for row in _rows_before(rows, dataset, decision_ms)
        if lower <= int(row.get("event_timestamp_ms", -1)) <= decision_ms
    ]


class _CovariateRowBuilder:
    def __init__(self, decision_ms: int) -> None:
        self.decision_ms = int(decision_ms)
        self.values: list[float] = []
        self.mask: list[bool] = []
        self.available: list[int] = []
        self.age_seconds: list[float] = []

    def add(self, value: float, available_ms: int, *, valid: bool) -> None:
        if valid:
            age = max(0.0, (self.decision_ms - int(available_ms)) / 1000.0)
            self.values.append(float(value))
            self.mask.append(True)
            self.available.append(int(available_ms))
            self.age_seconds.append(age)
        else:
            self.values.append(0.0)
            self.mask.append(False)
            self.available.append(-1)
            self.age_seconds.append(-1.0)

    def add_known_now(self, value: float) -> None:
        self.values.append(float(value))
        self.mask.append(True)
        self.available.append(self.decision_ms)
        self.age_seconds.append(0.0)


def _add_overview(
    builder: _CovariateRowBuilder,
    row: Mapping[str, Any] | None,
    source_available: bool,
) -> tuple[bool, float]:
    valid = source_available and row is not None
    missing = not valid
    available = -1 if row is None else int(row.get("available_timestamp_ms", -1))
    builder.add(_log1p_nonnegative(row.get("market_cap") if row else 0.0), available, valid=valid)
    builder.add(_log1p_nonnegative(row.get("share_class_shares_outstanding") if row else 0.0), available, valid=valid)
    list_ms = int(row.get("list_date_ms", -1)) if row is not None else -1
    builder.add((builder.decision_ms - list_ms) / DAY_MS if list_ms >= 0 else 0.0, available, valid=valid and list_ms >= 0)
    builder.add(_finite(row.get("is_common_stock") if row else 0.0), available, valid=valid)
    builder.add(_finite(row.get("is_adr_or_foreign") if row else 0.0), available, valid=valid)
    builder.add(_finite(row.get("is_active_reference_record") if row else 0.0), available, valid=valid)
    age = max(0.0, (builder.decision_ms - available) / 1000.0) if valid else 0.0
    builder.add(age, available, valid=valid)
    builder.add_known_now(float(missing))
    return not missing, age


def _add_financials(
    builder: _CovariateRowBuilder,
    rows: list[Mapping[str, Any]],
    latest: Mapping[str, Any] | None,
    source_available: bool,
) -> tuple[bool, float]:
    valid = source_available and latest is not None
    available = -1 if latest is None else int(latest.get("available_timestamp_ms", -1))
    recent = [
        row
        for row in rows
        if row.get("source_dataset") == "financials"
        and int(row.get("available_timestamp_ms", -1)) <= builder.decision_ms
        and int(row.get("available_timestamp_ms", -1)) >= builder.decision_ms - 365 * DAY_MS
    ]
    builder.add((builder.decision_ms - available) / DAY_MS if valid else 0.0, available, valid=valid)
    builder.add(float(len(recent)), builder.decision_ms, valid=source_available)
    yoy = 0.0
    if valid:
        fiscal_year = int(_finite(latest.get("financial_fiscal_year"), default=0.0))
        period = str(latest.get("financial_fiscal_period", ""))
        previous = [
            row
            for row in rows
            if row.get("source_dataset") == "financials"
            and int(_finite(row.get("financial_fiscal_year"), default=0.0)) == fiscal_year - 1
            and str(row.get("financial_fiscal_period", "")) == period
            and int(row.get("available_timestamp_ms", -1)) <= builder.decision_ms
        ]
        if previous:
            prev_revenue = _finite(max(previous, key=lambda row: int(row.get("available_timestamp_ms", -1))).get("financial_revenue"))
            yoy = _safe_ratio(_finite(latest.get("financial_revenue")) - prev_revenue, prev_revenue)
    builder.add(yoy, available, valid=valid)
    revenue = latest.get("financial_revenue") if latest else 0.0
    assets = latest.get("financial_assets") if latest else 0.0
    builder.add(_safe_ratio(latest.get("financial_net_income") if latest else 0.0, revenue), available, valid=valid)
    builder.add(_safe_ratio(latest.get("financial_liabilities") if latest else 0.0, assets), available, valid=valid)
    builder.add(_safe_ratio(latest.get("financial_cash") if latest else 0.0, assets), available, valid=valid)
    builder.add(_safe_ratio(latest.get("financial_operating_cashflow") if latest else 0.0, assets), available, valid=valid)
    age = max(0.0, (builder.decision_ms - available) / 1000.0) if valid else 0.0
    builder.add(age, available, valid=valid)
    builder.add_known_now(float(not source_available or not valid))
    return valid, age


def _add_dividends(
    builder: _CovariateRowBuilder,
    rows: list[Mapping[str, Any]],
    source_available: bool,
) -> tuple[bool, float]:
    recent = _count_recent(rows, "dividends", builder.decision_ms, 365 * DAY_MS)
    latest = max(recent, key=lambda row: int(row.get("event_timestamp_ms", -1))) if recent else None
    known_zero = source_available and latest is None
    available = int(latest.get("available_timestamp_ms", builder.decision_ms)) if latest else builder.decision_ms
    if latest is not None:
        days_since = (builder.decision_ms - int(latest.get("event_timestamp_ms", builder.decision_ms))) / DAY_MS
        age = max(0.0, (builder.decision_ms - available) / 1000.0)
    else:
        days_since = 0.0
        age = 0.0
    builder.add(days_since, available, valid=latest is not None)
    builder.add(float(len(recent)), builder.decision_ms, valid=source_available)
    builder.add(sum(_finite(row.get("dividend_cash_amount")) for row in recent), builder.decision_ms, valid=source_available)
    builder.add(age, available, valid=latest is not None or known_zero)
    builder.add_known_now(float(not source_available))
    return source_available, age


def _add_splits(
    builder: _CovariateRowBuilder,
    rows: list[Mapping[str, Any]],
    source_available: bool,
) -> tuple[bool, float]:
    recent = _count_recent(rows, "splits", builder.decision_ms, 365 * DAY_MS)
    latest = max(recent, key=lambda row: int(row.get("event_timestamp_ms", -1))) if recent else None
    available = int(latest.get("available_timestamp_ms", builder.decision_ms)) if latest else builder.decision_ms
    if latest is not None:
        days_since = (builder.decision_ms - int(latest.get("event_timestamp_ms", builder.decision_ms))) / DAY_MS
        age = max(0.0, (builder.decision_ms - available) / 1000.0)
    else:
        days_since = 0.0
        age = 0.0
    builder.add(days_since, available, valid=latest is not None)
    builder.add(float(len(recent)), builder.decision_ms, valid=source_available)
    builder.add(age, available, valid=latest is not None or source_available)
    builder.add_known_now(float(not source_available))
    return source_available, age


def _add_news(
    builder: _CovariateRowBuilder,
    rows: list[Mapping[str, Any]],
    source_available: bool,
) -> tuple[bool, float]:
    news_1h = _count_recent(rows, "news", builder.decision_ms, 3_600_000)
    news_1d = _count_recent(rows, "news", builder.decision_ms, DAY_MS)
    news_7d = _count_recent(rows, "news", builder.decision_ms, 7 * DAY_MS)
    news_30d = _count_recent(rows, "news", builder.decision_ms, 30 * DAY_MS)
    latest = max(_rows_before(rows, "news", builder.decision_ms), key=lambda row: int(row.get("event_timestamp_ms", -1)), default=None)
    available = int(latest.get("available_timestamp_ms", builder.decision_ms)) if latest else builder.decision_ms
    age = max(0.0, (builder.decision_ms - available) / 1000.0) if latest else 0.0
    builder.add(float(len(news_1h)), builder.decision_ms, valid=source_available)
    builder.add(float(len(news_1d)), builder.decision_ms, valid=source_available)
    builder.add(float(len(news_7d)), builder.decision_ms, valid=source_available)
    builder.add(float(len(news_30d)), builder.decision_ms, valid=source_available)
    builder.add(float(len({str(row.get("news_publisher_id", "")) for row in news_1d if row.get("news_publisher_id")})), builder.decision_ms, valid=source_available)
    builder.add(float(len({str(row.get("news_publisher_id", "")) for row in news_7d if row.get("news_publisher_id")})), builder.decision_ms, valid=source_available)
    builder.add(age, available, valid=latest is not None or source_available)
    builder.add_known_now(float(not source_available))
    return source_available, age


def build_action_covariate_tensor(
    *,
    silver_rows_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    action_names: list[str],
    decision_timestamps_ms: list[int] | torch.Tensor,
    source_coverage_by_symbol: Mapping[str, Mapping[str, bool]] | None = None,
    source_manifest_hash: str | None = None,
    max_age_days: int = 0,
) -> dict[str, Any]:
    decisions = [int(value) for value in torch.as_tensor(decision_timestamps_ms, dtype=torch.long).tolist()]
    max_age_ms = None if max_age_days <= 0 else int(max_age_days) * DAY_MS
    rows_by_symbol = {canonical_symbol(symbol): list(rows) for symbol, rows in silver_rows_by_symbol.items()}
    for action in action_names:
        symbol = canonical_symbol(action)
        if symbol != "CASH":
            rows_by_symbol.setdefault(symbol, [])
    normalized_source_coverage = {
        canonical_symbol(symbol): dict(coverage)
        for symbol, coverage in (source_coverage_by_symbol or {}).items()
    }
    coverage_missing_symbols = [
        canonical_symbol(action)
        for action in action_names
        if canonical_symbol(action) != "CASH" and canonical_symbol(action) not in normalized_source_coverage
    ]
    coverage_by_symbol = {
        symbol: normalized_source_coverage.get(symbol, empty_source_coverage())
        for symbol, rows in rows_by_symbol.items()
    }
    value_rows: list[list[list[float]]] = []
    mask_rows: list[list[list[bool]]] = []
    available_rows: list[list[list[int]]] = []
    age_rows: list[list[list[float]]] = []
    coverage_rows: list[list[float]] = []
    for decision_ms in decisions:
        decision_values: list[list[float]] = []
        decision_masks: list[list[bool]] = []
        decision_available: list[list[int]] = []
        decision_ages: list[list[float]] = []
        decision_group_coverage: list[float] = []
        for action in action_names:
            symbol = canonical_symbol(action)
            builder = _CovariateRowBuilder(decision_ms)
            if symbol == "CASH":
                for _name in ACTION_COVARIATE_FEATURE_NAMES:
                    builder.add(0.0, decision_ms, valid=False)
                decision_group_coverage.append(0.0)
            else:
                rows = rows_by_symbol.get(symbol, [])
                coverage = coverage_by_symbol.get(symbol, empty_source_coverage())
                group_ok: list[bool] = []
                group_ages: list[float] = []
                ok, age = _add_overview(
                    builder,
                    _latest_before(rows, "overview_snapshots", decision_ms, max_age_ms=max_age_ms),
                    bool(coverage.get("overview_snapshots", False)),
                )
                group_ok.append(ok)
                group_ages.append(age)
                ok, age = _add_financials(
                    builder,
                    rows,
                    _latest_before(rows, "financials", decision_ms, max_age_ms=max_age_ms),
                    bool(coverage.get("financials", False)),
                )
                group_ok.append(ok)
                group_ages.append(age)
                ok, age = _add_dividends(builder, rows, bool(coverage.get("dividends", False)))
                group_ok.append(ok)
                group_ages.append(age)
                ok, age = _add_splits(builder, rows, bool(coverage.get("splits", False)))
                group_ok.append(ok)
                group_ages.append(age)
                ok, age = _add_news(builder, rows, bool(coverage.get("news", False)))
                group_ok.append(ok)
                group_ages.append(age)
                coverage_fraction = sum(group_ok) / float(len(group_ok))
                max_age = max([age for ok, age in zip(group_ok, group_ages) if ok] or [0.0])
                builder.add_known_now(coverage_fraction)
                builder.add_known_now(max_age)
                decision_group_coverage.append(coverage_fraction)
            if len(builder.values) != len(ACTION_COVARIATE_FEATURE_NAMES):
                raise ValueError("Internal covariate feature width mismatch.")
            decision_values.append(builder.values)
            decision_masks.append(builder.mask)
            decision_available.append(builder.available)
            decision_ages.append(builder.age_seconds)
        value_rows.append(decision_values)
        mask_rows.append(decision_masks)
        available_rows.append(decision_available)
        age_rows.append(decision_ages)
        coverage_rows.append(decision_group_coverage)
    coverage_summary = {
        "mean_action_group_coverage_fraction": float(
            sum(sum(row) for row in coverage_rows) / max(sum(len(row) for row in coverage_rows), 1)
        ),
        "source_manifest_hash": source_manifest_hash,
        "source_coverage_explicit": source_coverage_by_symbol is not None,
        "symbols_missing_source_coverage": sorted(set(coverage_missing_symbols)),
    }
    reportability_errors = [] if source_manifest_hash else ["covariate_source_manifest_hash_missing"]
    if source_coverage_by_symbol is None:
        reportability_errors.append("covariate_source_coverage_manifest_missing")
    elif coverage_missing_symbols:
        reportability_errors.append("covariate_source_coverage_missing_for_selected_actions")
    return {
        "action_covariates": torch.tensor(value_rows, dtype=torch.float32),
        "action_covariate_mask": torch.tensor(mask_rows, dtype=torch.bool),
        "action_covariate_available_timestamps_ms": torch.tensor(available_rows, dtype=torch.long),
        "action_covariate_age_seconds": torch.tensor(age_rows, dtype=torch.float32),
        "action_covariate_feature_names": list(ACTION_COVARIATE_FEATURE_NAMES),
        "action_covariate_schema_hash": ACTION_COVARIATE_SCHEMA_HASH,
        "action_covariate_source_manifest_hash": source_manifest_hash,
        "action_covariate_coverage_report": coverage_summary,
        "action_covariate_reportability_errors": list(dict.fromkeys(reportability_errors)),
    }


def append_action_covariates_to_payload(
    payload: Mapping[str, Any],
    covariates: Mapping[str, Any],
    *,
    append_to_action_features: bool = True,
    append_mask_features: bool = True,
) -> dict[str, Any]:
    out = dict(payload)
    action_covariates = covariates["action_covariates"].float()
    action_covariate_mask = covariates["action_covariate_mask"].bool()
    action_covariate_available = covariates["action_covariate_available_timestamps_ms"].long()
    if tuple(action_covariates.shape[:2]) != tuple(out["action_returns"].shape):
        raise ValueError("action_covariates first two dimensions must match action_returns.")
    if tuple(action_covariate_mask.shape) != tuple(action_covariates.shape):
        raise ValueError("action_covariate_mask shape must match action_covariates.")
    if tuple(action_covariate_available.shape) != tuple(action_covariates.shape):
        raise ValueError("action_covariate_available_timestamps_ms shape must match action_covariates.")
    decisions = out.get("decision_timestamps_ms")
    if decisions is None:
        raise ValueError("decision_timestamps_ms is required before appending action covariates.")
    decision_ms = torch.as_tensor(decisions, dtype=torch.long).view(-1, 1, 1)
    decision_covariate_ms = decision_ms.expand_as(action_covariate_available)
    known_covariates = action_covariate_mask & (action_covariate_available >= 0)
    if bool((action_covariate_available[known_covariates] > decision_covariate_ms[known_covariates]).any().item()):
        raise ValueError("action covariate availability timestamp exceeds decision timestamp.")

    out.update(dict(covariates))
    feature_names = {key: list(value) for key, value in dict(out["feature_names"]).items()}
    feature_names["action_covariates"] = list(covariates["action_covariate_feature_names"])
    if append_to_action_features:
        base_features = out["action_features"].float()
        base_width = int(base_features.shape[-1])
        appended_features = [action_covariates]
        appended_names = [
            *[f"stock_covariates_v1.{name}" for name in covariates["action_covariate_feature_names"]],
        ]
        if append_mask_features:
            appended_features.append(action_covariate_mask.float())
            appended_names.extend(
                f"stock_covariates_v1_mask.{name}"
                for name in covariates["action_covariate_feature_names"]
            )
        appended_feature_tensor = torch.cat(appended_features, dim=-1)
        out["action_features"] = torch.cat([base_features, appended_feature_tensor], dim=-1)
        feature_names["action_features"] = [
            *feature_names.get("action_features", []),
            *appended_names,
        ]
        base_available = out.get("action_feature_available_timestamps_ms")
        if base_available is None:
            base_available = out.get("action_features_available_timestamps_ms")
        if base_available is None:
            raise ValueError("Base action feature availability is required before appending covariates.")
        base_available = base_available.long()
        if base_available.ndim == 2:
            base_available = base_available.unsqueeze(-1).expand(*base_available.shape, base_width)
        appended_available = [action_covariate_available]
        if append_mask_features:
            appended_available.append(decision_ms.expand_as(action_covariate_available))
        per_feature_available = torch.cat([base_available, *appended_available], dim=-1)
        decision_feature_ms = decision_ms.expand_as(per_feature_available)
        known_features = per_feature_available >= 0
        if bool((per_feature_available[known_features] > decision_feature_ms[known_features]).any().item()):
            raise ValueError("action feature availability timestamp exceeds decision timestamp after covariate append.")
        out["action_feature_available_timestamps_ms"] = per_feature_available
        known = per_feature_available >= 0
        row_level = torch.where(known, per_feature_available, torch.full_like(per_feature_available, -1)).amax(dim=-1)
        out["action_features_available_timestamps_ms"] = row_level
        out["action_features_augmented_with_covariates"] = True
        out["action_covariate_mask_appended_to_action_features"] = bool(append_mask_features)
        out["action_feature_groups"] = {
            "base_action_features": [0, base_width],
            "stock_covariates_v1": [base_width, base_width + int(action_covariates.shape[-1])],
        }
        if append_mask_features:
            mask_start = base_width + int(action_covariates.shape[-1])
            out["action_feature_groups"]["stock_covariates_v1_mask"] = [
                mask_start,
                mask_start + int(action_covariates.shape[-1]),
            ]
    else:
        out["action_features_augmented_with_covariates"] = False
        out["action_covariate_mask_appended_to_action_features"] = False
    out["feature_names"] = feature_names
    out["feature_names_by_tensor"] = feature_names
    out["feature_schema_hash"] = stable_json_hash(feature_names)
    base_payload_hash = out.get("payload_hash")
    payload_hash = stable_json_hash(
        {
            "base_payload_hash": base_payload_hash,
            "decision_timestamps": list(out.get("decision_timestamps", [])),
            "action_names": list(out.get("action_names", [])),
            "config": out.get("config", {}),
            "feature_schema_hash": out["feature_schema_hash"],
            "action_covariate_schema_hash": covariates["action_covariate_schema_hash"],
            "action_covariate_source_manifest_hash": covariates.get("action_covariate_source_manifest_hash"),
            "action_covariate_feature_schema_file_hash": covariates.get("action_covariate_feature_schema_file_hash"),
            "action_feature_groups": out.get("action_feature_groups", {}),
            "action_covariate_mask_appended_to_action_features": out.get("action_covariate_mask_appended_to_action_features"),
            "dataset_manifest": {
                key: value
                for key, value in dict(out.get("dataset_manifest", {})).items()
                if key
                not in {
                    "payload_hash",
                    "base_payload_hash_before_action_covariates",
                    "covariate_augmented_payload_hash",
                }
            },
        }
    )
    out["base_payload_hash_before_action_covariates"] = base_payload_hash
    out["payload_hash"] = payload_hash

    manifest = dict(out.get("dataset_manifest", {}))
    errors = list(manifest.get("reportability_errors", []))
    errors.extend(covariates.get("action_covariate_reportability_errors", []))
    manifest.update(
        {
            "action_covariate_schema_hash": covariates["action_covariate_schema_hash"],
            "action_covariate_source_manifest_hash": covariates.get("action_covariate_source_manifest_hash"),
            "action_covariate_feature_schema_file_hash": covariates.get("action_covariate_feature_schema_file_hash"),
            "action_features_augmented_with_covariates": bool(append_to_action_features),
            "action_covariate_mask_appended_to_action_features": bool(append_to_action_features and append_mask_features),
            "action_feature_groups": out.get("action_feature_groups", {}),
            "feature_schema_hash": out["feature_schema_hash"],
            "base_payload_hash_before_action_covariates": base_payload_hash,
            "payload_hash": payload_hash,
            "covariate_augmented_payload_hash": payload_hash,
            "action_covariate_cash_semantics": "CASH covariate values are zero-imputed and mask=false; mask channels are model-facing when appended.",
            "reportability_errors": list(dict.fromkeys(errors)),
        }
    )
    manifest["reportable"] = bool(manifest.get("reportable", True)) and not manifest["reportability_errors"]
    out["dataset_manifest"] = manifest
    return out


def write_silver_outputs(
    *,
    rows_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    output_root: Path,
    coverage_by_symbol: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    import pandas as pd

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for symbol, rows in rows_by_symbol.items():
        path = output_root / f"{symbol}.parquet"
        frame = pd.DataFrame.from_records(rows)
        frame.to_parquet(path, index=False)
        coverage = dict(coverage_by_symbol.get(symbol, {}))
        manifest_rows.append(
            {
                "symbol": symbol,
                "path": str(path),
                "rows": len(rows),
                "datasets_available": ",".join(sorted(dataset for dataset, present in coverage.items() if present)),
                "datasets_missing": ",".join(sorted(dataset for dataset, present in coverage.items() if not present)),
            }
        )
    manifest_path = output_root / "manifest.csv"
    pd.DataFrame.from_records(manifest_rows).to_csv(manifest_path, index=False)
    feature_schema = {
        "schema_version": "stock_covariates_silver_v1",
        "action_covariate_feature_names": ACTION_COVARIATE_FEATURE_NAMES,
        "action_covariate_schema_hash": ACTION_COVARIATE_SCHEMA_HASH,
    }
    (output_root / "feature_schema.json").write_text(
        stable_json_dump(feature_schema)
    )
    coverage_report = {
        "symbols": len(manifest_rows),
        "total_silver_rows": int(sum(int(row["rows"]) for row in manifest_rows)),
        "manifest_hash": stable_json_hash(manifest_rows),
    }
    (output_root / "coverage_report.json").write_text(stable_json_dump(coverage_report))
    return coverage_report


def stable_json_dump(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
