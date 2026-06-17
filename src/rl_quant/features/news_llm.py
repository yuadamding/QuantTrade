from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Mapping

import torch

from rl_quant.data_sources.polygon_stock_covariates import (
    canonical_symbol,
    covariate_jsonl_path,
    iter_jsonl_payloads,
    parse_utc_timestamp_ms,
)
from rl_quant.research_protocol import stable_json_hash, utc_now_iso


DAY_MS = 86_400_000
HOUR_MS = 3_600_000
NEWS_LLM_PROTOCOL_VERSION = "stock_news_llm_v1"
NEWS_LLM_ARTICLE_SCHEMA_VERSION = "stock_news_article_table_v1"
NEWS_LLM_EXTRACT_SCHEMA_VERSION = "stock_news_llm_article_ticker_v1"
NEWS_LLM_ACTION_SIDECAR_SCHEMA_VERSION = "hour_from_second_action_news_llm_v1"
NEWS_LLM_FLAT_APPEND_MODE = "flat_append_with_news_llm_v1"
DETERMINISTIC_NEWS_LLM_MODEL_ID = "deterministic_news_llm_v1_baseline"
NEWS_LLM_ANALYST_MODEL_POLICY_SCHEMA_VERSION = "llm_analyst_model_stack_v1"
DEFAULT_NEWS_LLM_PRIMARY_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_NEWS_LLM_SECONDARY_MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_NEWS_LLM_FALLBACK_MODEL_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
DEFAULT_NEWS_LLM_SERVING_ENGINE = "local_transformers"
# The current local extractor produces prompted JSON that is extracted, range-clamped, and
# strictly validated post-hoc -- NOT constrained-decoding JSON-schema output. The default must
# describe that honestly so manifests do not over-claim structured decoding.
DEFAULT_NEWS_LLM_STRUCTURED_OUTPUT = "prompted_json_posthoc_extract_clamp_validate"
DETERMINISTIC_NEWS_LLM_PROMPT_HASH = stable_json_hash(
    {
        "schema": NEWS_LLM_EXTRACT_SCHEMA_VERSION,
        "extractor": "deterministic_keyword_baseline",
        "retrieval": "disabled",
        "temperature": 0,
    }
)
NEWS_LLM_EVENT_FLAGS = [
    "event_earnings",
    "event_guidance",
    "event_product",
    "event_ai_or_technology",
    "event_analyst_rating",
    "event_mna",
    "event_regulatory",
    "event_litigation",
    "event_macro",
    "event_sector",
    "event_management",
    "event_capital_return",
]
NEWS_LLM_ARTICLE_TICKER_FIELDS = [
    "article_id",
    "ticker",
    "published_utc",
    "published_timestamp_ms",
    "source_available_timestamp_ms",
    "llm_feature_available_timestamp_ms",
    "ticker_relevance",
    "is_primary_ticker",
    "company_specificity",
    "is_broad_market_or_sector",
    "sentiment_score",
    "positive_score",
    "negative_score",
    "neutral_score",
    "uncertainty_score",
    "materiality_score",
    "novelty_score",
    "time_horizon",
    *NEWS_LLM_EVENT_FLAGS,
    "confidence",
    "llm_valid",
    "llm_model_id",
    "llm_prompt_hash",
    "llm_schema_version",
    "llm_schema_hash",
    "extractor_provider",
    "extractor_temperature",
    "extractor_no_retrieval",
    "model_available_timestamp_ms",
    "model_training_cutoff_utc",
    "article_weight",
    "ticker_count",
]
NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH = stable_json_hash(NEWS_LLM_ARTICLE_TICKER_FIELDS)
NEWS_LLM_AGGREGATE_FEATURE_NAMES = [
    "log1p_llm_weighted_news_count_1h",
    "log1p_llm_weighted_news_count_1d",
    "log1p_llm_weighted_news_count_7d",
    "log1p_llm_weighted_news_count_30d",
    "llm_positive_intensity_1d",
    "llm_negative_intensity_1d",
    "llm_net_sentiment_1d",
    "llm_net_sentiment_7d",
    "llm_net_sentiment_30d",
    "log1p_llm_material_positive_count_7d",
    "log1p_llm_material_negative_count_7d",
    "llm_company_specific_fraction_1d",
    "llm_broad_market_fraction_1d",
    "llm_multi_ticker_fraction_7d",
    "log1p_llm_earnings_event_count_30d",
    "log1p_llm_guidance_event_count_30d",
    "log1p_llm_analyst_rating_event_count_7d",
    "log1p_llm_regulatory_negative_event_count_30d",
    "log1p_llm_litigation_negative_event_count_30d",
    "log1p_llm_product_or_ai_positive_event_count_30d",
    "llm_novelty_weighted_sentiment_1d",
    "llm_novelty_weighted_sentiment_7d",
    "llm_avg_confidence_7d",
    "llm_low_confidence_fraction_7d",
    "log1p_llm_invalid_output_count_7d",
    "llm_time_since_last_material_news_seconds",
    "llm_time_since_last_negative_news_seconds",
    "llm_news_missing_flag",
]
NEWS_LLM_AGGREGATE_SCHEMA_HASH = stable_json_hash(NEWS_LLM_AGGREGATE_FEATURE_NAMES)


def default_news_llm_analyst_model_policy(
    *,
    primary_model_id: str = DEFAULT_NEWS_LLM_PRIMARY_MODEL_ID,
    secondary_model_id: str = DEFAULT_NEWS_LLM_SECONDARY_MODEL_ID,
    fallback_model_id: str = DEFAULT_NEWS_LLM_FALLBACK_MODEL_ID,
    serving_engine: str = DEFAULT_NEWS_LLM_SERVING_ENGINE,
    structured_output: str = DEFAULT_NEWS_LLM_STRUCTURED_OUTPUT,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    return {
        "schema_version": NEWS_LLM_ANALYST_MODEL_POLICY_SCHEMA_VERSION,
        "llm_feature_group": NEWS_LLM_PROTOCOL_VERSION,
        "primary_model_id": primary_model_id,
        "primary_model_role": "main_extractor",
        "secondary_model_id": secondary_model_id,
        "secondary_model_role": "validator_or_fallback",
        "fallback_model_id": fallback_model_id,
        "fallback_model_role": "structured_output_fallback",
        "serving_engine": serving_engine,
        "structured_output": structured_output,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "no_external_retrieval": True,
        "cached_outputs_only": True,
        "recommended_feature_groups": [
            NEWS_LLM_PROTOCOL_VERSION,
            "stock_fundamental_llm_v1",
        ],
        "retrospective_historical_policy": {
            "reportable_for_2023_to_2026_backtest": False,
            "reportability_errors": ["llm_extractor_not_available_at_backtest_start"],
            "note": "Treat retrospective LLM-assisted 2023-2026 runs as diagnostic unless extractor availability is proven at the first decision.",
        },
    }


def stable_json_dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def timestamp_ms_to_utc_iso(value: int) -> str:
    return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc).replace(microsecond=0).isoformat()


def parse_timestamp_ms(value: str | int | float | None, *, default: int | None = None) -> int:
    if value is None or value == "":
        if default is None:
            raise ValueError("timestamp value is required")
        return int(default)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return parse_utc_timestamp_ms(text)


def json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def json_list_text(values: Iterable[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"), default=str)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publisher_name(payload: Mapping[str, Any]) -> str:
    publisher = payload.get("publisher")
    if isinstance(publisher, Mapping):
        name = publisher.get("name") or publisher.get("homepage_url") or publisher.get("favicon_url")
        return "" if name is None else str(name)
    return "" if publisher is None else str(publisher)


def _news_tickers(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("tickers", [])
    if isinstance(raw, str):
        raw_items = [raw]
    elif isinstance(raw, Iterable):
        raw_items = list(raw)
    else:
        raw_items = []
    return list(dict.fromkeys(canonical_symbol(str(item)) for item in raw_items if str(item).strip()))


def _news_article_id(payload: Mapping[str, Any]) -> str:
    for key in ("id", "article_url", "amp_url"):
        value = payload.get(key)
        if value:
            return str(value)
    return stable_json_hash(
        {
            "publisher": _publisher_name(payload),
            "published_utc": payload.get("published_utc", payload.get("published_at", payload.get("created_at", ""))),
            "title": payload.get("title", ""),
        }
    )


def _url_hash(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    return "" if not value else stable_json_hash(str(value))


def _raw_article_row(
    symbol: str, payload: Mapping[str, Any], line_number: int, *, source_latency_seconds: int = 0
) -> dict[str, Any]:
    if source_latency_seconds < 0:
        # Fail closed rather than clamp: a negative latency would imply availability BEFORE publish
        # (a look-ahead), and silently clamping it to 0 would hand a non-CLI caller optimistic
        # point-in-time availability. The CLI rejects this too; the library must not be more lenient.
        raise ValueError(f"source_latency_seconds must be non-negative; got {source_latency_seconds}.")
    published = payload.get("published_utc", payload.get("published_at", payload.get("created_at")))
    published_ms = parse_timestamp_ms(published)
    # Source availability defaults to publish time, but publish time is not necessarily pipeline
    # availability. A non-zero --source-latency-seconds yields a more conservative point-in-time
    # availability (published + latency); the policy/latency is recorded in the article manifest.
    source_available_ms = int(published_ms) + int(source_latency_seconds) * 1000
    tickers = _news_tickers(payload)
    primary = canonical_symbol(str(payload.get("primary_ticker", tickers[0] if tickers else symbol)))
    record_hash = stable_json_hash(dict(payload))
    return {
        "article_id": _news_article_id(payload),
        "published_utc": timestamp_ms_to_utc_iso(published_ms),
        "published_timestamp_ms": int(published_ms),
        "source_available_timestamp_ms": int(source_available_ms),
        "publisher_name": _publisher_name(payload),
        "title": str(payload.get("title", "") or ""),
        "description": str(payload.get("description", "") or payload.get("summary", "") or ""),
        "tickers_json": json_list_text(tickers),
        "primary_ticker": primary,
        "ticker_count": len(tickers),
        "source_symbols_json": json_list_text([canonical_symbol(symbol)]),
        "source_record_hashes_json": json_list_text([record_hash]),
        "source_record_count": 1,
        "source_line_count": 1,
        "article_url_hash": _url_hash(payload, "article_url"),
        "amp_url_hash": _url_hash(payload, "amp_url"),
        "first_source_line": int(line_number),
    }


def _merge_article(existing: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    if int(incoming["published_timestamp_ms"]) < int(merged["published_timestamp_ms"]):
        merged["published_timestamp_ms"] = int(incoming["published_timestamp_ms"])
        merged["published_utc"] = incoming["published_utc"]
    merged["source_available_timestamp_ms"] = min(
        int(merged["source_available_timestamp_ms"]),
        int(incoming["source_available_timestamp_ms"]),
    )
    for key in ("publisher_name", "title", "description", "primary_ticker", "article_url_hash", "amp_url_hash"):
        if not merged.get(key) and incoming.get(key):
            merged[key] = incoming[key]
    tickers = sorted(
        set(canonical_symbol(str(item)) for item in json_list(merged.get("tickers_json")))
        | set(canonical_symbol(str(item)) for item in json_list(incoming.get("tickers_json")))
    )
    source_symbols = sorted(
        set(canonical_symbol(str(item)) for item in json_list(merged.get("source_symbols_json")))
        | set(canonical_symbol(str(item)) for item in json_list(incoming.get("source_symbols_json")))
    )
    hashes = sorted(
        set(str(item) for item in json_list(merged.get("source_record_hashes_json")))
        | set(str(item) for item in json_list(incoming.get("source_record_hashes_json")))
    )
    merged["tickers_json"] = json_list_text(tickers)
    merged["source_symbols_json"] = json_list_text(source_symbols)
    merged["source_record_hashes_json"] = json_list_text(hashes)
    merged["ticker_count"] = len(tickers)
    merged["source_record_count"] = len(hashes)
    merged["source_line_count"] = int(merged.get("source_line_count", 0)) + int(incoming.get("source_line_count", 1))
    return merged


def build_news_article_rows(
    *,
    raw_root: Path,
    symbols: list[str],
    strict: bool = False,
    source_latency_seconds: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    if source_latency_seconds < 0:
        # Validate up front (not per row): the per-row try/except below would otherwise bury this
        # config error in the errors list instead of failing closed.
        raise ValueError(f"source_latency_seconds must be non-negative; got {source_latency_seconds}.")
    articles: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for symbol in symbols:
        normalized = canonical_symbol(symbol)
        path = covariate_jsonl_path(raw_root, normalized, "news")
        if path is None:
            errors.append(f"{normalized}: missing news.jsonl")
            continue
        for line_number, payload in iter_jsonl_payloads(path):
            try:
                row = _raw_article_row(normalized, payload, line_number, source_latency_seconds=source_latency_seconds)
            except Exception as exc:  # noqa: BLE001 - keep batch build moving unless strict.
                errors.append(f"{normalized}:{line_number}: {exc}")
                continue
            key = str(row["article_id"])
            articles[key] = _merge_article(articles[key], row) if key in articles else row
    if strict and errors:
        preview = "; ".join(errors[:10])
        raise ValueError(f"News article table build failed: {preview}")
    rows = sorted(
        articles.values(),
        key=lambda row: (int(row["source_available_timestamp_ms"]), int(row["published_timestamp_ms"]), str(row["article_id"])),
    )
    return rows, errors


def discover_news_source_symbols(*, raw_root: Path, symbols: list[str]) -> list[str]:
    return [
        canonical_symbol(symbol)
        for symbol in symbols
        if covariate_jsonl_path(raw_root, canonical_symbol(symbol), "news") is not None
    ]


def write_news_article_outputs(
    *,
    rows: list[Mapping[str, Any]],
    output_root: Path,
    raw_root: Path,
    symbols: list[str],
    source_symbols: list[str] | None = None,
    errors: list[str],
    source_latency_seconds: int = 0,
) -> dict[str, Any]:
    if source_latency_seconds < 0:
        raise ValueError(f"source_latency_seconds must be non-negative; got {source_latency_seconds}.")
    import pandas as pd

    output_root.mkdir(parents=True, exist_ok=True)
    article_path = output_root / "news_articles.parquet"
    frame = pd.DataFrame.from_records(rows)
    frame.to_parquet(article_path, index=False)
    symbols_with_source = (
        sorted(set(canonical_symbol(symbol) for symbol in source_symbols))
        if source_symbols is not None
        else sorted(
            set(
                canonical_symbol(str(symbol))
                for row in rows
                for symbol in json_list(row.get("source_symbols_json"))
            )
        )
    )
    manifest = {
        "schema_version": NEWS_LLM_ARTICLE_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "raw_root": str(raw_root),
        "source_availability_policy": (
            "published_plus_source_latency_seconds" if source_latency_seconds else "published_timestamp"
        ),
        "source_latency_seconds": int(source_latency_seconds),
        "selected_symbol_count": len(symbols),
        "selected_symbols": list(symbols),
        "symbols_with_source_news": symbols_with_source,
        "article_count": len(rows),
        "first_published_utc": rows[0]["published_utc"] if rows else None,
        "last_published_utc": rows[-1]["published_utc"] if rows else None,
        "article_table_path": str(article_path),
        "article_table_hash": stable_json_hash(
            [
                {
                    "article_id": row.get("article_id"),
                    "published_timestamp_ms": row.get("published_timestamp_ms"),
                    "source_record_hashes_json": row.get("source_record_hashes_json"),
                }
                for row in rows
            ]
        ),
        "error_count": len(errors),
        "errors_preview": errors[:50],
        "reportable": not errors,
        "reportability_errors": [] if not errors else ["news_article_table_source_errors"],
    }
    (output_root / "manifest.json").write_text(stable_json_dump(manifest))
    return manifest


def read_news_article_rows(root_or_path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    path = root_or_path / "news_articles.parquet" if root_or_path.is_dir() else root_or_path
    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict("records")


def _keyword_hits(text: str, keywords: Iterable[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _bounded_score(hits: int, scale: float = 3.0) -> float:
    return max(0.0, min(1.0, float(hits) / scale))


def _event_flags(text: str) -> dict[str, float]:
    return {
        "event_earnings": float(_keyword_hits(text, ("earnings", "eps", "revenue", "profit", "quarter")) > 0),
        "event_guidance": float(_keyword_hits(text, ("guidance", "forecast", "outlook", "expects", "projection")) > 0),
        "event_product": float(_keyword_hits(text, ("product", "launch", "unveils", "release", "orders")) > 0),
        "event_ai_or_technology": float(
            _keyword_hits(text, (" ai ", "artificial intelligence", "chip", "semiconductor", "software", "data center")) > 0
        ),
        "event_analyst_rating": float(
            _keyword_hits(text, ("upgrade", "downgrade", "price target", "initiates", "maintains", "rating")) > 0
        ),
        "event_mna": float(_keyword_hits(text, ("merger", "acquisition", "acquire", "buyout", "takeover")) > 0),
        "event_regulatory": float(_keyword_hits(text, ("sec", "regulator", "regulatory", "antitrust", "probe")) > 0),
        "event_litigation": float(_keyword_hits(text, ("lawsuit", "litigation", "court", "settlement", "trial")) > 0),
        "event_macro": float(_keyword_hits(text, ("fed", "inflation", "rates", "jobs", "gdp", "economy", "tariff")) > 0),
        "event_sector": float(_keyword_hits(text, ("sector", "industry", "semiconductor", "bank", "energy")) > 0),
        "event_management": float(_keyword_hits(text, ("ceo", "cfo", "executive", "resigns", "appoints")) > 0),
        "event_capital_return": float(_keyword_hits(text, ("dividend", "buyback", "repurchase", "split")) > 0),
    }


def _time_horizon(text: str) -> str:
    if _keyword_hits(text, ("today", "intraday", "pre-market", "after hours", "this morning")):
        return "intraday"
    if _keyword_hits(text, ("week", "short term", "near term", "days")):
        return "days_to_weeks"
    if _keyword_hits(text, ("year", "long term", "multi-year", "2027", "2028")):
        return "months_to_years"
    return "unknown"


def _article_text(row: Mapping[str, Any]) -> str:
    return f"{row.get('title', '')} {row.get('description', '')}".strip().lower()


def deterministic_article_ticker_features(
    article: Mapping[str, Any],
    *,
    ticker: str,
    model_id: str,
    model_available_timestamp_ms: int,
    model_training_cutoff_utc: str,
    vendor_latency_seconds: int,
    processing_latency_seconds: int,
    provider: str = "deterministic_baseline",
) -> dict[str, Any]:
    text = f" {_article_text(article)} "
    positive_hits = _keyword_hits(
        text,
        (
            "beat",
            "upgrade",
            "raises",
            "growth",
            "record",
            "bullish",
            "profit",
            "strong",
            "wins",
            "surges",
        ),
    )
    negative_hits = _keyword_hits(
        text,
        (
            "miss",
            "downgrade",
            "cuts",
            "lawsuit",
            "investigation",
            "probe",
            "weak",
            "bearish",
            "falls",
            "loss",
        ),
    )
    uncertainty_hits = _keyword_hits(text, ("may", "could", "risk", "uncertain", "warn", "possible", "volatile"))
    positive = _bounded_score(positive_hits)
    negative = _bounded_score(negative_hits)
    sentiment = max(-1.0, min(1.0, positive - negative))
    neutral = max(0.0, 1.0 - min(1.0, positive + negative))
    uncertainty = _bounded_score(uncertainty_hits, scale=4.0)
    flags = _event_flags(text)
    event_count = sum(flags.values())
    materiality = max(
        _bounded_score(int(event_count), scale=2.0),
        _bounded_score(_keyword_hits(text, ("material", "major", "significant", "deal", "guidance")), scale=2.0),
        min(1.0, abs(sentiment) * 0.75),
    )
    tickers = [canonical_symbol(str(item)) for item in json_list(article.get("tickers_json"))]
    ticker_count = max(len(tickers), 1)
    primary = canonical_symbol(str(article.get("primary_ticker", tickers[0] if tickers else ticker)))
    normalized_ticker = canonical_symbol(ticker)
    company_specificity = max(0.0, min(1.0, 1.0 - (ticker_count - 1) / 10.0))
    broad_market = float(ticker_count > 3 or flags["event_macro"] > 0 or flags["event_sector"] > 0)
    ticker_relevance = 1.0 if normalized_ticker == primary else 1.0 / float(ticker_count)
    novelty = 1.0
    title_or_description_present = bool(str(article.get("title", "") or article.get("description", "")).strip())
    confidence = 0.0 if not title_or_description_present else min(0.75, 0.45 + 0.03 * event_count + 0.04 * (positive_hits + negative_hits))
    source_available = int(article.get("source_available_timestamp_ms", article.get("published_timestamp_ms", -1)))
    feature_available = max(int(source_available), int(model_available_timestamp_ms)) + int(vendor_latency_seconds + processing_latency_seconds) * 1000
    row = {
        "article_id": str(article.get("article_id", "")),
        "ticker": normalized_ticker,
        "published_utc": str(article.get("published_utc", "")),
        "published_timestamp_ms": int(article.get("published_timestamp_ms", -1)),
        "source_available_timestamp_ms": source_available,
        "llm_feature_available_timestamp_ms": int(feature_available),
        "ticker_relevance": ticker_relevance,
        "is_primary_ticker": float(normalized_ticker == primary),
        "company_specificity": company_specificity,
        "is_broad_market_or_sector": broad_market,
        "sentiment_score": sentiment,
        "positive_score": positive,
        "negative_score": negative,
        "neutral_score": neutral,
        "uncertainty_score": uncertainty,
        "materiality_score": materiality,
        "novelty_score": novelty,
        "time_horizon": _time_horizon(text),
        **flags,
        "confidence": confidence,
        "llm_valid": bool(title_or_description_present and normalized_ticker),
        "llm_model_id": model_id,
        "llm_prompt_hash": DETERMINISTIC_NEWS_LLM_PROMPT_HASH,
        "llm_schema_version": NEWS_LLM_EXTRACT_SCHEMA_VERSION,
        "llm_schema_hash": NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH,
        "extractor_provider": provider,
        "extractor_temperature": 0.0,
        "extractor_no_retrieval": True,
        "model_available_timestamp_ms": int(model_available_timestamp_ms),
        "model_training_cutoff_utc": model_training_cutoff_utc,
        "article_weight": max(0.0, ticker_relevance * company_specificity * novelty),
        "ticker_count": float(ticker_count),
    }
    return {field: row.get(field) for field in NEWS_LLM_ARTICLE_TICKER_FIELDS}


def build_deterministic_news_llm_rows(
    article_rows: list[Mapping[str, Any]],
    *,
    model_id: str = DETERMINISTIC_NEWS_LLM_MODEL_ID,
    model_available_timestamp_ms: int = 0,
    model_training_cutoff_utc: str = "not_applicable_deterministic_baseline",
    vendor_latency_seconds: int = 300,
    processing_latency_seconds: int = 60,
    allowed_tickers: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed = None if allowed_tickers is None else {canonical_symbol(symbol) for symbol in allowed_tickers}
    for article in article_rows:
        for ticker in json_list(article.get("tickers_json")):
            normalized = canonical_symbol(str(ticker))
            if normalized and (allowed is None or normalized in allowed):
                rows.append(
                    deterministic_article_ticker_features(
                        article,
                        ticker=normalized,
                        model_id=model_id,
                        model_available_timestamp_ms=model_available_timestamp_ms,
                        model_training_cutoff_utc=model_training_cutoff_utc,
                        vendor_latency_seconds=vendor_latency_seconds,
                        processing_latency_seconds=processing_latency_seconds,
                    )
                )
    return sorted(
        rows,
        key=lambda row: (
            int(row["llm_feature_available_timestamp_ms"]),
            str(row["ticker"]),
            str(row["article_id"]),
        ),
    )


_NEWS_LLM_UNIT_INTERVAL_FIELDS = (
    "ticker_relevance",
    "company_specificity",
    "positive_score",
    "negative_score",
    "neutral_score",
    "uncertainty_score",
    "materiality_score",
    "novelty_score",
    "confidence",
)
_NEWS_LLM_BINARY_FIELDS = ("is_primary_ticker", "is_broad_market_or_sector", "llm_valid", *NEWS_LLM_EVENT_FLAGS)
NEWS_LLM_TIME_HORIZONS = ("intraday", "days_to_weeks", "months_to_years", "unknown")
_NEWS_LLM_NONEMPTY_STRING_FIELDS = (
    "article_id",
    "ticker",
    "llm_model_id",
    "llm_prompt_hash",
    "extractor_provider",
)


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def validate_news_llm_rows(rows: list[Mapping[str, Any]]) -> list[str]:
    """Strict validation for a reportable, model-facing LLM feature table.

    Beyond schema/identity, this enforces finiteness and declared score ranges, point-in-time
    timestamp ordering (published <= source <= feature, model <= feature), nonempty provenance
    strings, and duplicate-key rejection. These guards matter most on the imported/precomputed
    path, where rows are produced outside this module.
    """
    errors: list[str] = []
    required = set(NEWS_LLM_ARTICLE_TICKER_FIELDS)
    seen_keys: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            errors.append(f"row {index}: missing fields {sorted(missing)}")
            continue
        if row.get("llm_schema_hash") != NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH:
            errors.append(f"row {index}: llm_schema_hash mismatch")
        if row.get("llm_schema_version") != NEWS_LLM_EXTRACT_SCHEMA_VERSION:
            errors.append(f"row {index}: llm_schema_version mismatch")
        if _finite(row.get("extractor_temperature"), default=-1.0) != 0.0:
            errors.append(f"row {index}: extractor_temperature must be 0")
        if not _truthy(row.get("extractor_no_retrieval")):
            errors.append(f"row {index}: extractor_no_retrieval must be true")
        for field in _NEWS_LLM_NONEMPTY_STRING_FIELDS:
            if not str(row.get(field) or "").strip():
                errors.append(f"row {index}: {field} must be a nonempty string")
        # Continuous scores must be real floats, not booleans coerced to 0.0/1.0 (a bool here
        # usually signals a schema/extraction error), so reject bool explicitly for these fields.
        sentiment_raw = row.get("sentiment_score")
        sentiment = None if isinstance(sentiment_raw, bool) else _as_float_or_none(sentiment_raw)
        if sentiment is None or not math.isfinite(sentiment) or not (-1.0 <= sentiment <= 1.0):
            errors.append(f"row {index}: sentiment_score must be a finite float in [-1, 1]")
        for field in _NEWS_LLM_UNIT_INTERVAL_FIELDS:
            raw = row.get(field)
            value = None if isinstance(raw, bool) else _as_float_or_none(raw)
            if value is None or not math.isfinite(value) or not (0.0 <= value <= 1.0):
                errors.append(f"row {index}: {field} must be a finite float in [0, 1]")
        for field in _NEWS_LLM_BINARY_FIELDS:
            value = _as_float_or_none(row.get(field))
            if value is None or value not in (0.0, 1.0):
                errors.append(f"row {index}: {field} must be 0 or 1")
        if row.get("time_horizon") not in NEWS_LLM_TIME_HORIZONS:
            errors.append(f"row {index}: time_horizon must be one of {NEWS_LLM_TIME_HORIZONS}")
        published = int(_finite(row.get("published_timestamp_ms"), default=-1.0))
        source_available = int(_finite(row.get("source_available_timestamp_ms"), default=-1.0))
        feature_available = int(_finite(row.get("llm_feature_available_timestamp_ms"), default=-1.0))
        model_available = int(_finite(row.get("model_available_timestamp_ms"), default=0.0))
        if published < 0 or source_available < 0 or feature_available < 0:
            errors.append(f"row {index}: published/source/feature timestamps must be non-negative ms integers")
        if published > source_available:
            errors.append(f"row {index}: published_timestamp_ms must be <= source_available_timestamp_ms")
        if source_available > feature_available:
            errors.append(f"row {index}: source availability must be <= llm feature availability")
        if model_available > feature_available:
            errors.append(f"row {index}: model availability must be <= llm feature availability")
        key = (str(row.get("article_id")), str(row.get("ticker")), str(row.get("llm_schema_hash")))
        if key in seen_keys:
            errors.append(f"row {index}: duplicate (article_id, ticker, llm_schema_hash) {key}")
        else:
            seen_keys.add(key)
    return errors


def write_news_llm_feature_outputs(
    *,
    rows: list[Mapping[str, Any]],
    output_root: Path,
    article_manifest: Mapping[str, Any] | None,
    model_id: str,
    model_available_timestamp_ms: int,
    model_training_cutoff_utc: str,
    provider: str,
    errors: list[str] | None = None,
    analyst_model_policy: Mapping[str, Any] | None = None,
    generation_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import pandas as pd

    output_root.mkdir(parents=True, exist_ok=True)
    # Validate BEFORE writing, and only ever materialize a validated table at the canonical path.
    # An invalid table is quarantined under a clearly non-canonical filename so a downstream
    # consumer cannot mistake it for a usable, reportable feature table.
    validation_errors = validate_news_llm_rows([dict(row) for row in rows])
    # Provenance must be derived from the rows, not hard-coded, so imported/precomputed tables are
    # described honestly. Mixed provenance (more than one prompt hash / model id / provider) is a
    # reportability failure unless explicitly reconciled upstream.
    distinct_prompt_hashes = sorted({str(r.get("llm_prompt_hash")) for r in rows if str(r.get("llm_prompt_hash") or "").strip()})
    distinct_model_ids = sorted({str(r.get("llm_model_id")) for r in rows if str(r.get("llm_model_id") or "").strip()})
    distinct_providers = sorted({str(r.get("extractor_provider")) for r in rows if str(r.get("extractor_provider") or "").strip()})
    distinct_temperatures = sorted({round(_finite(r.get("extractor_temperature"), default=0.0), 6) for r in rows})
    mixed_provenance = (
        len(distinct_prompt_hashes) > 1
        or len(distinct_model_ids) > 1
        or len(distinct_providers) > 1
        or len(distinct_temperatures) > 1
    )
    provenance_errors = ["mixed_provenance_in_feature_table"] if mixed_provenance else []
    all_errors = list(errors or []) + validation_errors + provenance_errors
    reportable = not all_errors
    canonical_path = output_root / "news_article_ticker_llm.parquet"
    feature_path = canonical_path if reportable else output_root / "news_article_ticker_llm.nonreportable.parquet"
    frame = pd.DataFrame.from_records(rows, columns=NEWS_LLM_ARTICLE_TICKER_FIELDS)
    tmp_path = feature_path.with_name(f".{feature_path.name}.tmp.{os.getpid()}")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(feature_path)
    # On a non-reportable build, remove any stale canonical table from a prior valid build so a
    # reader keyed on the canonical path cannot consume an out-of-date, now-superseded table.
    if not reportable and canonical_path.exists():
        canonical_path.unlink()
    resolved_prompt_hash = distinct_prompt_hashes[0] if len(distinct_prompt_hashes) == 1 else DETERMINISTIC_NEWS_LLM_PROMPT_HASH
    symbols = sorted(set(canonical_symbol(str(row.get("ticker", ""))) for row in rows if row.get("ticker")))
    model_policy = dict(analyst_model_policy or default_news_llm_analyst_model_policy())
    manifest = {
        "schema_version": NEWS_LLM_EXTRACT_SCHEMA_VERSION,
        "protocol_version": NEWS_LLM_PROTOCOL_VERSION,
        "llm_feature_group": NEWS_LLM_PROTOCOL_VERSION,
        "created_at_utc": utc_now_iso(),
        "feature_table_path": str(feature_path),
        "feature_table_file_name": feature_path.name,
        "article_manifest_hash": stable_json_hash(article_manifest or {}),
        "article_table_hash": None if article_manifest is None else article_manifest.get("article_table_hash"),
        "row_count": len(rows),
        "symbol_count": len(symbols),
        "symbols_with_news_llm": symbols,
        "llm_model_id": model_id,
        "llm_prompt_hash": resolved_prompt_hash,
        "llm_schema_hash": NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH,
        "llm_schema_version": NEWS_LLM_EXTRACT_SCHEMA_VERSION,
        "extractor_provider": provider,
        "extractor_temperature": 0.0,
        "extractor_no_retrieval": True,
        "model_available_timestamp_ms": int(model_available_timestamp_ms),
        "model_available_utc": timestamp_ms_to_utc_iso(model_available_timestamp_ms),
        "model_training_cutoff_utc": model_training_cutoff_utc,
        "primary_model_id": model_policy.get("primary_model_id"),
        "primary_model_role": model_policy.get("primary_model_role"),
        "secondary_model_id": model_policy.get("secondary_model_id"),
        "secondary_model_role": model_policy.get("secondary_model_role"),
        "fallback_model_id": model_policy.get("fallback_model_id"),
        "fallback_model_role": model_policy.get("fallback_model_role"),
        "serving_engine": model_policy.get("serving_engine"),
        "structured_output": model_policy.get("structured_output"),
        "temperature": model_policy.get("temperature"),
        "top_p": model_policy.get("top_p"),
        "no_external_retrieval": model_policy.get("no_external_retrieval"),
        "cached_outputs_only": model_policy.get("cached_outputs_only"),
        "llm_analyst_model_policy": model_policy,
        # Identity-only hash kept for backward compatibility. The CONTENT hash below covers every
        # feature field so a change to any score/provenance value invalidates reportability.
        "feature_table_hash": stable_json_hash(
            [
                {
                    "article_id": row.get("article_id"),
                    "ticker": row.get("ticker"),
                    "llm_feature_available_timestamp_ms": row.get("llm_feature_available_timestamp_ms"),
                    "llm_schema_hash": row.get("llm_schema_hash"),
                }
                for row in rows
            ]
        ),
        "feature_table_content_hash": stable_json_hash(
            [
                {field: row.get(field) for field in NEWS_LLM_ARTICLE_TICKER_FIELDS}
                for row in sorted(
                    rows,
                    key=lambda r: (
                        str(r.get("ticker")),
                        str(r.get("article_id")),
                        int(_finite(r.get("llm_feature_available_timestamp_ms"), default=0.0)),
                    ),
                )
            ]
        ),
        "feature_table_file_sha256": _file_sha256(feature_path),
        "distinct_extractor_temperatures": distinct_temperatures,
        "distinct_llm_prompt_hashes": distinct_prompt_hashes,
        "distinct_llm_model_ids": distinct_model_ids,
        "distinct_extractor_providers": distinct_providers,
        "mixed_provenance": bool(mixed_provenance),
        "generation_diagnostics": dict(generation_diagnostics) if generation_diagnostics is not None else None,
        "reportability_errors": list(dict.fromkeys(all_errors)),
    }
    manifest["reportable"] = not manifest["reportability_errors"]
    (output_root / "manifest.json").write_text(stable_json_dump(manifest))
    return manifest


def read_news_llm_rows(root_or_path: Path, *, allow_nonreportable: bool = False) -> list[dict[str, Any]]:
    import pandas as pd

    if root_or_path.is_dir():
        # The manifest is the source of truth for which feature table is current. Fail CLOSED by
        # default on a non-reportable manifest: raise rather than return [] so a caller cannot
        # confuse "non-reportable feature table" with "known zero LLM rows". Pass
        # allow_nonreportable=True for explicit diagnostic reads.
        manifest = read_manifest(root_or_path)
        if manifest.get("reportability_errors") and not allow_nonreportable:
            raise ValueError(
                "Refusing to read non-reportable news LLM feature table: "
                + "; ".join(str(error) for error in list(manifest["reportability_errors"])[:20])
            )
        # Resolve the table path portably: prefer the file NAME inside this artifact dir, then a
        # relative path resolved against the dir, then an absolute path, so the manifest reads
        # correctly from any working directory.
        file_name = manifest.get("feature_table_file_name")
        raw_path = manifest.get("feature_table_path")
        if file_name:
            path = root_or_path / str(file_name)
        elif raw_path:
            candidate = Path(str(raw_path))
            path = candidate if candidate.is_absolute() else root_or_path / candidate.name
        else:
            path = root_or_path / "news_article_ticker_llm.parquet"
    else:
        path = root_or_path
    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict("records")


def load_news_llm_rows_by_symbol(
    root_or_path: Path, symbols: list[str], *, allow_nonreportable: bool = False
) -> dict[str, list[dict[str, Any]]]:
    selected = {canonical_symbol(symbol) for symbol in symbols if canonical_symbol(symbol) != "CASH"}
    rows_by_symbol = {symbol: [] for symbol in selected}
    for row in read_news_llm_rows(root_or_path, allow_nonreportable=allow_nonreportable):
        ticker = canonical_symbol(str(row.get("ticker", "")))
        if ticker in selected:
            rows_by_symbol.setdefault(ticker, []).append(row)
    for ticker in rows_by_symbol:
        rows_by_symbol[ticker].sort(
            key=lambda row: (
                int(row.get("llm_feature_available_timestamp_ms", -1)),
                int(row.get("published_timestamp_ms", -1)),
                str(row.get("article_id", "")),
            )
        )
    return rows_by_symbol


def _known_rows(rows: list[Mapping[str, Any]], decision_ms: int) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if int(_finite(row.get("llm_feature_available_timestamp_ms"), default=-1.0)) <= decision_ms
        and int(_finite(row.get("published_timestamp_ms"), default=-1.0)) <= decision_ms
    ]


def _window_rows(rows: list[Mapping[str, Any]], decision_ms: int, window_ms: int) -> list[Mapping[str, Any]]:
    lower = decision_ms - window_ms
    deduped: dict[str, Mapping[str, Any]] = {}
    for row in _known_rows(rows, decision_ms):
        published = int(_finite(row.get("published_timestamp_ms"), default=-1.0))
        if lower <= published <= decision_ms:
            key = str(row.get("article_id", ""))
            existing = deduped.get(key)
            if existing is None or int(row.get("llm_feature_available_timestamp_ms", -1)) > int(
                existing.get("llm_feature_available_timestamp_ms", -1)
            ):
                deduped[key] = row
    return list(deduped.values())


def _row_weight(row: Mapping[str, Any]) -> float:
    if not _truthy(row.get("llm_valid")):
        return 0.0
    explicit = _finite(row.get("article_weight"), default=float("nan"))
    if math.isfinite(explicit):
        return max(0.0, explicit)
    return max(
        0.0,
        _finite(row.get("ticker_relevance"), default=0.0)
        * _finite(row.get("company_specificity"), default=0.0)
        * _finite(row.get("novelty_score"), default=1.0),
    )


def _weighted_sum(rows: list[Mapping[str, Any]], predicate: str | None = None) -> float:
    if predicate is None:
        return sum(_row_weight(row) for row in rows)
    return sum(_row_weight(row) for row in rows if _finite(row.get(predicate), default=0.0) > 0.0)


def _weighted_mean(rows: list[Mapping[str, Any]], field: str) -> float:
    total_weight = sum(_row_weight(row) for row in rows)
    if total_weight <= 0.0:
        return 0.0
    return sum(_finite(row.get(field), default=0.0) * _row_weight(row) for row in rows) / total_weight


def _fraction(rows: list[Mapping[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    weight_total = sum(_row_weight(row) for row in rows)
    if weight_total <= 0.0:
        return 0.0
    return sum(_row_weight(row) for row in rows if predicate(row)) / weight_total


def _novelty_weighted_sentiment(rows: list[Mapping[str, Any]]) -> float:
    total = sum(_row_weight(row) * _finite(row.get("novelty_score"), default=1.0) for row in rows)
    if total <= 0.0:
        return 0.0
    return (
        sum(
            _finite(row.get("sentiment_score"), default=0.0)
            * _row_weight(row)
            * _finite(row.get("novelty_score"), default=1.0)
            for row in rows
        )
        / total
    )


def _last_input_available_ms(rows: list[Mapping[str, Any]]) -> int | None:
    """Most recent availability among contributing (weighted) rows, or None if there are none."""
    avails = [
        int(_finite(row.get("llm_feature_available_timestamp_ms"), default=-1.0))
        for row in rows
        if _row_weight(row) > 0.0
    ]
    avails = [value for value in avails if value >= 0]
    return max(avails) if avails else None


class _NewsLlmAggregateBuilder:
    def __init__(self, decision_ms: int, source_available: bool) -> None:
        self.decision_ms = int(decision_ms)
        self.source_available = bool(source_available)
        self.values: list[float] = []
        self.mask: list[bool] = []
        self.available: list[int] = []
        self.age_seconds: list[float] = []

    def _append(self, value: float, *, mask: bool, available_ms: int) -> None:
        if mask:
            self.values.append(float(value))
            self.mask.append(True)
            self.available.append(int(available_ms))
            self.age_seconds.append(max(0.0, (self.decision_ms - int(available_ms)) / 1000.0))
        else:
            self.values.append(0.0)
            self.mask.append(False)
            self.available.append(-1)
            self.age_seconds.append(-1.0)

    def add_decision_value(self, value: float) -> None:
        self._append(value, mask=self.source_available, available_ms=self.decision_ms)

    def add_count_value(self, value: float, *, window_rows: list[Mapping[str, Any]]) -> None:
        # Count features are valid (value meaningful, including 0 = "no news") whenever source
        # coverage exists. Freshness reflects the most recent contributing row when present,
        # otherwise the decision time (a true zero-count computed now).
        last = _last_input_available_ms(window_rows)
        self._append(
            value,
            mask=self.source_available,
            available_ms=last if last is not None else self.decision_ms,
        )

    def add_mean_value(self, value: float, *, window_rows: list[Mapping[str, Any]]) -> None:
        # Mean/fraction/sentiment features are only valid when at least one weighted row exists;
        # otherwise a 0.0 would be indistinguishable from "no news in the window", so mask False.
        # Freshness reflects the most recent contributing row.
        last = _last_input_available_ms(window_rows)
        valid = self.source_available and last is not None
        self._append(value, mask=valid, available_ms=last if last is not None else -1)

    def add_event_age(self, value: float, available_ms: int | None) -> None:
        if self.source_available and available_ms is not None and available_ms >= 0:
            self.values.append(float(value))
            self.mask.append(True)
            self.available.append(int(available_ms))
            self.age_seconds.append(max(0.0, (self.decision_ms - int(available_ms)) / 1000.0))
        else:
            self.values.append(0.0)
            self.mask.append(False)
            self.available.append(-1)
            self.age_seconds.append(-1.0)

    def add_missing_flag(self) -> None:
        self.values.append(float(not self.source_available))
        self.mask.append(True)
        self.available.append(self.decision_ms)
        self.age_seconds.append(0.0)


def aggregate_news_llm_features_for_symbol(
    *,
    rows: list[Mapping[str, Any]],
    decision_ms: int,
    source_available: bool,
) -> tuple[list[float], list[bool], list[int], list[float]]:
    rows_1h = _window_rows(rows, decision_ms, HOUR_MS)
    rows_1d = _window_rows(rows, decision_ms, DAY_MS)
    rows_7d = _window_rows(rows, decision_ms, 7 * DAY_MS)
    rows_30d = _window_rows(rows, decision_ms, 30 * DAY_MS)
    known = _known_rows(rows, decision_ms)
    builder = _NewsLlmAggregateBuilder(decision_ms, source_available=source_available)
    for window_rows in (rows_1h, rows_1d, rows_7d, rows_30d):
        builder.add_count_value(math.log1p(_weighted_sum(window_rows)), window_rows=window_rows)
    builder.add_mean_value(_weighted_mean(rows_1d, "positive_score"), window_rows=rows_1d)
    builder.add_mean_value(_weighted_mean(rows_1d, "negative_score"), window_rows=rows_1d)
    builder.add_mean_value(_weighted_mean(rows_1d, "sentiment_score"), window_rows=rows_1d)
    builder.add_mean_value(_weighted_mean(rows_7d, "sentiment_score"), window_rows=rows_7d)
    builder.add_mean_value(_weighted_mean(rows_30d, "sentiment_score"), window_rows=rows_30d)
    builder.add_count_value(
        math.log1p(
            sum(
                _row_weight(row)
                for row in rows_7d
                if _finite(row.get("materiality_score"), default=0.0) >= 0.5
                and _finite(row.get("sentiment_score"), default=0.0) > 0.0
            )
        ),
        window_rows=rows_7d,
    )
    builder.add_count_value(
        math.log1p(
            sum(
                _row_weight(row)
                for row in rows_7d
                if _finite(row.get("materiality_score"), default=0.0) >= 0.5
                and _finite(row.get("sentiment_score"), default=0.0) < 0.0
            )
        ),
        window_rows=rows_7d,
    )
    builder.add_mean_value(_weighted_mean(rows_1d, "company_specificity"), window_rows=rows_1d)
    builder.add_mean_value(_weighted_mean(rows_1d, "is_broad_market_or_sector"), window_rows=rows_1d)
    builder.add_mean_value(
        _fraction(rows_7d, lambda row: _finite(row.get("ticker_count"), default=1.0) > 1.0), window_rows=rows_7d
    )
    builder.add_count_value(math.log1p(_weighted_sum(rows_30d, "event_earnings")), window_rows=rows_30d)
    builder.add_count_value(math.log1p(_weighted_sum(rows_30d, "event_guidance")), window_rows=rows_30d)
    builder.add_count_value(math.log1p(_weighted_sum(rows_7d, "event_analyst_rating")), window_rows=rows_7d)
    builder.add_count_value(
        math.log1p(
            sum(
                _row_weight(row)
                for row in rows_30d
                if _finite(row.get("event_regulatory"), default=0.0) > 0.0
                and _finite(row.get("sentiment_score"), default=0.0) < 0.0
            )
        ),
        window_rows=rows_30d,
    )
    builder.add_count_value(
        math.log1p(
            sum(
                _row_weight(row)
                for row in rows_30d
                if _finite(row.get("event_litigation"), default=0.0) > 0.0
                and _finite(row.get("sentiment_score"), default=0.0) < 0.0
            )
        ),
        window_rows=rows_30d,
    )
    builder.add_count_value(
        math.log1p(
            sum(
                _row_weight(row)
                for row in rows_30d
                if (
                    _finite(row.get("event_product"), default=0.0) > 0.0
                    or _finite(row.get("event_ai_or_technology"), default=0.0) > 0.0
                )
                and _finite(row.get("sentiment_score"), default=0.0) > 0.0
            )
        ),
        window_rows=rows_30d,
    )
    builder.add_mean_value(_novelty_weighted_sentiment(rows_1d), window_rows=rows_1d)
    builder.add_mean_value(_novelty_weighted_sentiment(rows_7d), window_rows=rows_7d)
    builder.add_mean_value(_weighted_mean(rows_7d, "confidence"), window_rows=rows_7d)
    builder.add_mean_value(
        _fraction(rows_7d, lambda row: _finite(row.get("confidence"), default=0.0) < 0.5), window_rows=rows_7d
    )
    builder.add_count_value(
        math.log1p(sum(1 for row in rows_7d if not _truthy(row.get("llm_valid")))), window_rows=rows_7d
    )

    material_rows = [
        row
        for row in known
        if _finite(row.get("materiality_score"), default=0.0) >= 0.5 and _truthy(row.get("llm_valid"))
    ]
    last_material = max(
        material_rows,
        key=lambda row: int(row.get("llm_feature_available_timestamp_ms", -1)),
        default=None,
    )
    material_available = (
        int(last_material.get("llm_feature_available_timestamp_ms", -1)) if last_material is not None else None
    )
    builder.add_event_age(
        (decision_ms - material_available) / 1000.0 if material_available is not None else 0.0,
        material_available,
    )
    negative_rows = [
        row
        for row in known
        if _finite(row.get("sentiment_score"), default=0.0) < 0.0 and _truthy(row.get("llm_valid"))
    ]
    last_negative = max(
        negative_rows,
        key=lambda row: int(row.get("llm_feature_available_timestamp_ms", -1)),
        default=None,
    )
    negative_available = (
        int(last_negative.get("llm_feature_available_timestamp_ms", -1)) if last_negative is not None else None
    )
    builder.add_event_age(
        (decision_ms - negative_available) / 1000.0 if negative_available is not None else 0.0,
        negative_available,
    )
    builder.add_missing_flag()
    if len(builder.values) != len(NEWS_LLM_AGGREGATE_FEATURE_NAMES):
        raise ValueError("Internal news LLM aggregate feature width mismatch.")
    return builder.values, builder.mask, builder.available, builder.age_seconds


def build_action_news_llm_tensor(
    *,
    news_llm_rows_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    action_names: list[str],
    decision_timestamps_ms: list[int] | torch.Tensor,
    source_symbols: Iterable[str] | None = None,
    source_manifest_hash: str | None = None,
) -> dict[str, Any]:
    decisions = [int(value) for value in torch.as_tensor(decision_timestamps_ms, dtype=torch.long).tolist()]
    rows_by_symbol = {canonical_symbol(symbol): list(rows) for symbol, rows in news_llm_rows_by_symbol.items()}
    explicit_source_symbols = None if source_symbols is None else {canonical_symbol(symbol) for symbol in source_symbols}
    value_rows: list[list[list[float]]] = []
    mask_rows: list[list[list[bool]]] = []
    available_rows: list[list[list[int]]] = []
    age_rows: list[list[list[float]]] = []
    for decision_ms in decisions:
        decision_values: list[list[float]] = []
        decision_masks: list[list[bool]] = []
        decision_available: list[list[int]] = []
        decision_ages: list[list[float]] = []
        for action in action_names:
            symbol = canonical_symbol(action)
            if symbol == "CASH":
                width = len(NEWS_LLM_AGGREGATE_FEATURE_NAMES)
                decision_values.append([0.0] * width)
                decision_masks.append([False] * width)
                decision_available.append([-1] * width)
                decision_ages.append([-1.0] * width)
                continue
            rows = rows_by_symbol.get(symbol, [])
            if explicit_source_symbols is None:
                source_available = bool(rows)
            else:
                source_available = symbol in explicit_source_symbols
            values, mask, available, ages = aggregate_news_llm_features_for_symbol(
                rows=rows,
                decision_ms=decision_ms,
                source_available=source_available,
            )
            decision_values.append(values)
            decision_masks.append(mask)
            decision_available.append(available)
            decision_ages.append(ages)
        value_rows.append(decision_values)
        mask_rows.append(decision_masks)
        available_rows.append(decision_available)
        age_rows.append(decision_ages)
    first_decision = min(decisions) if decisions else None
    model_available_values = [
        int(_finite(row.get("model_available_timestamp_ms"), default=0.0))
        for rows in rows_by_symbol.values()
        for row in rows
    ]
    model_available_timestamp_ms = max(model_available_values) if model_available_values else 0
    reportability_errors: list[str] = [] if source_manifest_hash else ["news_llm_source_manifest_hash_missing"]
    if explicit_source_symbols is None:
        reportability_errors.append("news_llm_source_coverage_missing")
    if first_decision is not None and model_available_timestamp_ms > first_decision:
        reportability_errors.append("news_llm_model_available_after_first_decision")
    features = torch.tensor(value_rows, dtype=torch.float32)
    mask = torch.tensor(mask_rows, dtype=torch.bool)
    available = torch.tensor(available_rows, dtype=torch.long)
    decision_tensor = torch.tensor(decisions, dtype=torch.long).view(-1, 1, 1).expand_as(available)
    known = mask & (available >= 0)
    if bool((available[known] > decision_tensor[known]).any().item()):
        raise ValueError("News LLM feature availability timestamp exceeds decision timestamp.")
    return {
        "action_news_llm_features": features,
        "action_news_llm_mask": mask,
        "action_news_llm_available_timestamps_ms": available,
        "action_news_llm_age_seconds": torch.tensor(age_rows, dtype=torch.float32),
        "action_news_llm_feature_names": list(NEWS_LLM_AGGREGATE_FEATURE_NAMES),
        "action_news_llm_schema_hash": NEWS_LLM_AGGREGATE_SCHEMA_HASH,
        "action_news_llm_protocol_version": NEWS_LLM_PROTOCOL_VERSION,
        "action_news_llm_source_manifest_hash": source_manifest_hash,
        "action_news_llm_model_available_timestamp_ms": model_available_timestamp_ms,
        "action_news_llm_reportability_errors": list(dict.fromkeys(reportability_errors)),
    }


def read_manifest(path_or_root: Path) -> dict[str, Any]:
    path = path_or_root / "manifest.json" if path_or_root.is_dir() else path_or_root
    if not path.exists():
        return {}
    return json.loads(path.read_text())
