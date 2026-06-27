#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.features.news_llm import (  # noqa: E402
    NEWS_LLM_ARTICLE_TICKER_FIELDS,
    NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH,
    NEWS_LLM_EVENT_FLAGS,
    NEWS_LLM_EXTRACT_SCHEMA_VERSION,
    canonical_symbol,
    deterministic_article_ticker_features,
    json_list,
    parse_timestamp_ms,
    read_news_article_rows,
    stable_json_hash,
)


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_ARTICLE_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "news_articles_v1" / "top500_2023_to_present"
DEFAULT_PROTOCOL_ROOT = DATA_ROOT / "protocol" / "polygon_second_top500_2023_to_2026-06-15"
DEFAULT_PARTITIONS_ROOT = DEFAULT_PROTOCOL_ROOT / "hour_from_second_1s_top50" / "partitions"
DEFAULT_OUTPUT_JSONL = (
    DATA_ROOT
    / "polygon"
    / "stock_covariates"
    / "news_llm_v1"
    / "qwen3_1_7b_top16_2023_to_present"
    / "precomputed_qwen3_1_7b.jsonl"
)
DEFAULT_LOCAL_MODEL = Path("../LLM/Qwen3-1.7B")
PROMPT_VERSION = "qwen_news_llm_article_ticker_prompt_v1"


def prompt_hash_for(
    temperature: float,
    *,
    top_p: float = 1.0,
    max_new_tokens: int = 384,
    no_retrieval: bool = True,
) -> str:
    # Bind ALL generation-affecting parameters (not just temperature) so the hash changes when
    # sampling, nucleus, or length settings change -- a sampled or differently-configured run
    # cannot share an identity with the deterministic temperature=0 extractor.
    return stable_json_hash(
        {
            "prompt_version": PROMPT_VERSION,
            "schema": NEWS_LLM_EXTRACT_SCHEMA_VERSION,
            "extractor": "local_qwen3_1_7b",
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(temperature != 0.0),
            "structured_output": "prompted_json_posthoc_extract_clamp_validate",
            "no_retrieval": bool(no_retrieval),
        }
    )


DETERMINISTIC_PROMPT_HASH = prompt_hash_for(0.0)
TIME_HORIZONS = {"intraday", "days_to_weeks", "months_to_years", "unknown"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate precomputed stock_news_llm_v1 article-ticker rows with local Qwen3-1.7B."
    )
    parser.add_argument("--article-root", type=Path, default=DEFAULT_ARTICLE_ROOT)
    parser.add_argument("--partitions-root", type=Path, default=DEFAULT_PARTITIONS_ROOT)
    parser.add_argument("--dataset-file-name", default="hour_from_second_dataset.pt")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--local-model", type=Path, default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--tickers", help="Comma-separated tickers. Defaults to non-cash actions from --partitions-root.")
    parser.add_argument(
        "--model-available-timestamp-utc",
        default=None,
        help=(
            "When the pretrained extractor became available (UTC ISO). Defaults to the local model "
            "manifest's downloaded_at_utc; a 1970/epoch default is intentionally NOT used because it "
            "would make a modern model appear available before any backtest. Required if neither is present."
        ),
    )
    parser.add_argument("--model-training-cutoff-utc", default="unknown_for_downloaded_pretrained_model")
    parser.add_argument("--vendor-latency-seconds", type=int, default=300)
    parser.add_argument("--processing-latency-seconds", type=int, default=60)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all selected article-ticker rows.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_action_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return [canonical_symbol(symbol) for symbol in args.tickers.split(",") if canonical_symbol(symbol)]
    paths = sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))
    if not paths:
        raise FileNotFoundError(f"No partition datasets found below {args.partitions_root}")
    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    return [canonical_symbol(symbol) for symbol in payload["action_names"] if canonical_symbol(symbol) != "CASH"]


def read_completed_keys(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open() as source:
        for line in source:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            article_id = str(row.get("article_id", ""))
            ticker = canonical_symbol(str(row.get("ticker", "")))
            if article_id and ticker:
                completed.add((article_id, ticker))
    return completed


def selected_article_ticker_pairs(article_rows: list[Mapping[str, Any]], tickers: list[str]) -> list[tuple[Mapping[str, Any], str]]:
    selected = set(canonical_symbol(ticker) for ticker in tickers)
    pairs: list[tuple[Mapping[str, Any], str]] = []
    for article in article_rows:
        article_tickers = [canonical_symbol(str(item)) for item in json_list(article.get("tickers_json"))]
        for ticker in article_tickers:
            if ticker in selected:
                pairs.append((article, ticker))
    return sorted(
        pairs,
        key=lambda pair: (
            int(pair[0].get("source_available_timestamp_ms", -1)),
            str(pair[1]),
            str(pair[0].get("article_id", "")),
        ),
    )


def load_local_model_manifest(local_model: Path) -> dict[str, Any]:
    manifest_path = local_model / "download_manifest.json"
    if not manifest_path.exists():
        return {"repo_id": local_model.name, "revision": "local"}
    return json.loads(manifest_path.read_text())


def model_id_from_manifest(manifest: Mapping[str, Any]) -> str:
    repo_id = str(manifest.get("repo_id", "Qwen/Qwen3-1.7B"))
    revision = str(manifest.get("revision", "local"))
    return f"{repo_id}@{revision}" if revision else repo_id


def dtype_from_args(value: str) -> torch.dtype:
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_model(args: argparse.Namespace) -> tuple[Any, Any, str]:
    local_model = resolve_project_path(args.local_model)
    dtype = dtype_from_args(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        local_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "dtype": dtype,
    }
    if args.device == "auto":
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(local_model, **{k: v for k, v in kwargs.items() if v is not None})
    if args.device in {"cuda", "cpu"}:
        model = model.to(args.device)
    model.eval()
    device = str(next(model.parameters()).device)
    return tokenizer, model, device


def article_prompt(article: Mapping[str, Any], ticker: str) -> list[dict[str, str]]:
    tickers = [canonical_symbol(str(item)) for item in json_list(article.get("tickers_json"))]
    schema = {
        "ticker_relevance": "float 0..1",
        "company_specificity": "float 0..1",
        "is_broad_market_or_sector": "0 or 1",
        "sentiment_score": "float -1..1",
        "positive_score": "float 0..1",
        "negative_score": "float 0..1",
        "neutral_score": "float 0..1",
        "uncertainty_score": "float 0..1",
        "materiality_score": "float 0..1",
        "novelty_score": "float 0..1",
        "time_horizon": "intraday | days_to_weeks | months_to_years | unknown",
        **{flag: "0 or 1" for flag in NEWS_LLM_EVENT_FLAGS},
        "confidence": "float 0..1",
        "llm_valid": "boolean",
    }
    user = {
        "task": "Extract point-in-time trading covariates for exactly one stock ticker from this news item.",
        "ticker": ticker,
        "published_utc": article.get("published_utc", ""),
        "primary_ticker": article.get("primary_ticker", ""),
        "article_tickers": tickers,
        "publisher": article.get("publisher_name", ""),
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "schema": schema,
        "rules": [
            "Use only the supplied title and description.",
            "Do not infer facts from outside knowledge.",
            "Return one compact JSON object and no markdown.",
            "All numeric scores must stay inside the requested ranges.",
        ],
    }
    return [
        {"role": "system", "content": "You are a deterministic financial news feature extractor. Return only JSON."},
        {"role": "user", "content": json.dumps(user, sort_keys=True, ensure_ascii=False)},
    ]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(cleaned[start : end + 1])


def clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not torch.isfinite(torch.tensor(number)):
        number = default
    return max(low, min(high, number))


def binary(value: Any) -> float:
    if isinstance(value, str):
        return float(value.strip().lower() in {"1", "true", "yes", "y"})
    return float(bool(value))


def normalize_qwen_payload(parsed: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    positive = clamp(parsed.get("positive_score"), 0.0, 1.0, float(fallback["positive_score"]))
    negative = clamp(parsed.get("negative_score"), 0.0, 1.0, float(fallback["negative_score"]))
    sentiment = clamp(parsed.get("sentiment_score"), -1.0, 1.0, positive - negative)
    neutral = clamp(parsed.get("neutral_score"), 0.0, 1.0, max(0.0, 1.0 - min(1.0, positive + negative)))
    horizon = str(parsed.get("time_horizon", fallback["time_horizon"])).strip()
    if horizon not in TIME_HORIZONS:
        horizon = "unknown"
    row = {
        "ticker_relevance": clamp(parsed.get("ticker_relevance"), 0.0, 1.0, float(fallback["ticker_relevance"])),
        "company_specificity": clamp(
            parsed.get("company_specificity"), 0.0, 1.0, float(fallback["company_specificity"])
        ),
        "is_broad_market_or_sector": binary(parsed.get("is_broad_market_or_sector", fallback["is_broad_market_or_sector"])),
        "sentiment_score": sentiment,
        "positive_score": positive,
        "negative_score": negative,
        "neutral_score": neutral,
        "uncertainty_score": clamp(parsed.get("uncertainty_score"), 0.0, 1.0, float(fallback["uncertainty_score"])),
        "materiality_score": clamp(parsed.get("materiality_score"), 0.0, 1.0, float(fallback["materiality_score"])),
        "novelty_score": clamp(parsed.get("novelty_score"), 0.0, 1.0, float(fallback["novelty_score"])),
        "time_horizon": horizon,
        "confidence": clamp(parsed.get("confidence"), 0.0, 1.0, 0.5),
        "llm_valid": bool(parsed.get("llm_valid", True)),
    }
    for flag in NEWS_LLM_EVENT_FLAGS:
        row[flag] = binary(parsed.get(flag, fallback[flag]))
    return row


def build_output_row(
    *,
    article: Mapping[str, Any],
    ticker: str,
    parsed: Mapping[str, Any] | None,
    fallback: Mapping[str, Any],
    model_id: str,
    model_available_timestamp_ms: int,
    model_training_cutoff_utc: str,
    vendor_latency_seconds: int,
    processing_latency_seconds: int,
    parse_error: str | None,
    extractor_temperature: float,
    prompt_hash: str,
) -> dict[str, Any]:
    row = dict(fallback)
    if parsed is not None:
        row.update(normalize_qwen_payload(parsed, fallback))
    else:
        row["llm_valid"] = False
        row["confidence"] = 0.0
    source_available = int(article.get("source_available_timestamp_ms", article.get("published_timestamp_ms", -1)))
    feature_available = max(source_available, model_available_timestamp_ms) + (
        int(vendor_latency_seconds) + int(processing_latency_seconds)
    ) * 1000
    row.update(
        {
            "article_id": str(article.get("article_id", "")),
            "ticker": canonical_symbol(ticker),
            "published_utc": str(article.get("published_utc", "")),
            "published_timestamp_ms": int(article.get("published_timestamp_ms", -1)),
            "source_available_timestamp_ms": source_available,
            "llm_feature_available_timestamp_ms": int(feature_available),
            "llm_model_id": model_id,
            "llm_prompt_hash": prompt_hash,
            "llm_schema_version": NEWS_LLM_EXTRACT_SCHEMA_VERSION,
            "llm_schema_hash": NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH,
            "extractor_provider": "local_transformers_qwen3_1_7b",
            # Record the ACTUAL generation temperature. A nonzero (sampled) run is therefore
            # honestly non-deterministic and downstream validate_news_llm_rows marks it
            # non-reportable, instead of every row claiming temperature 0.
            "extractor_temperature": float(extractor_temperature),
            "extractor_no_retrieval": True,
            "model_available_timestamp_ms": int(model_available_timestamp_ms),
            "model_training_cutoff_utc": model_training_cutoff_utc,
            "ticker_count": float(article.get("ticker_count", fallback.get("ticker_count", 1.0)) or 1.0),
        }
    )
    row["article_weight"] = max(
        0.0,
        float(row["ticker_relevance"]) * float(row["company_specificity"]) * float(row["novelty_score"]),
    )
    if parse_error:
        row["llm_valid"] = False
    return {field: row.get(field) for field in NEWS_LLM_ARTICLE_TICKER_FIELDS}


@torch.inference_mode()
def generate_batch(
    *,
    tokenizer: Any,
    model: Any,
    pairs: list[tuple[Mapping[str, Any], str]],
    args: argparse.Namespace,
    model_id: str,
    model_available_timestamp_ms: int,
    prompt_hash: str,
) -> tuple[list[dict[str, Any]], int]:
    fallbacks = [
        deterministic_article_ticker_features(
            article,
            ticker=ticker,
            model_id=model_id,
            model_available_timestamp_ms=model_available_timestamp_ms,
            model_training_cutoff_utc=args.model_training_cutoff_utc,
            vendor_latency_seconds=args.vendor_latency_seconds,
            processing_latency_seconds=args.processing_latency_seconds,
            provider="local_transformers_qwen3_1_7b_fallback",
        )
        for article, ticker in pairs
    ]
    prompts = [render_prompt(tokenizer, article_prompt(article, ticker)) for article, ticker in pairs]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    do_sample = args.temperature != 0.0
    generation_kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    generated = model.generate(
        **generation_kwargs,
    )
    prompt_width = encoded["input_ids"].shape[-1]
    texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    rows: list[dict[str, Any]] = []
    parse_errors = 0
    for (article, ticker), fallback, text in zip(pairs, fallbacks, texts, strict=True):
        parsed: dict[str, Any] | None
        parse_error: str | None = None
        try:
            parsed = extract_json_object(text)
        except Exception as exc:  # noqa: BLE001 - failed structured output becomes an invalid audited row.
            parsed = None
            parse_error = f"{type(exc).__name__}: {exc}"
            parse_errors += 1
        rows.append(
            build_output_row(
                article=article,
                ticker=ticker,
                parsed=parsed,
                fallback=fallback,
                model_id=model_id,
                model_available_timestamp_ms=model_available_timestamp_ms,
                model_training_cutoff_utc=args.model_training_cutoff_utc,
                vendor_latency_seconds=args.vendor_latency_seconds,
                processing_latency_seconds=args.processing_latency_seconds,
                parse_error=parse_error,
                extractor_temperature=args.temperature,
                prompt_hash=prompt_hash,
            )
        )
    return rows, parse_errors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.article_root = resolve_project_path(args.article_root)
    args.partitions_root = resolve_project_path(args.partitions_root)
    args.local_model = resolve_project_path(args.local_model)
    args.output_jsonl = resolve_project_path(args.output_jsonl)
    model_manifest = load_local_model_manifest(args.local_model)
    model_id = model_id_from_manifest(model_manifest)
    model_available_utc = args.model_available_timestamp_utc or str(model_manifest.get("downloaded_at_utc") or "")
    if not model_available_utc:
        raise SystemExit(
            "Model availability is required: pass --model-available-timestamp-utc or provide a local "
            "download_manifest.json with downloaded_at_utc. Refusing to default to the epoch, which "
            "would make a modern pretrained model appear available before any backtest."
        )
    model_available_timestamp_ms = parse_timestamp_ms(model_available_utc)
    prompt_hash = prompt_hash_for(args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    tickers = load_action_tickers(args)
    articles = read_news_article_rows(args.article_root)
    pairs = selected_article_ticker_pairs(articles, tickers)
    if args.max_rows > 0:
        pairs = pairs[: args.max_rows]
    completed = read_completed_keys(args.output_jsonl) if args.resume else set()
    todo = [
        (article, ticker)
        for article, ticker in pairs
        if (str(article.get("article_id", "")), canonical_symbol(ticker)) not in completed
    ]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "output_jsonl": str(args.output_jsonl),
                "model_id": model_id,
                "device": args.device,
                "selected_tickers": tickers,
                "total_selected_pairs": len(pairs),
                "completed_pairs": len(completed),
                "todo_pairs": len(todo),
                "prompt_hash": prompt_hash,
                "extractor_temperature": float(args.temperature),
                "model_available_utc": model_available_utc,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not todo:
        return 0
    tokenizer, model, device = load_model(args)
    print(f"Loaded {model_id} on {device}", flush=True)
    written = 0
    parse_error_total = 0
    invalid_total = 0
    started = time.monotonic()
    batch_size = max(1, int(args.batch_size))
    next_report = 1
    with args.output_jsonl.open("a") as sink:
        index = 0
        while index < len(todo):
            batch = todo[index : index + batch_size]
            try:
                rows, batch_parse_errors = generate_batch(
                    tokenizer=tokenizer,
                    model=model,
                    pairs=batch,
                    args=args,
                    model_id=model_id,
                    model_available_timestamp_ms=model_available_timestamp_ms,
                    prompt_hash=prompt_hash,
                )
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                print(f"CUDA OOM; reducing batch_size to {batch_size}", flush=True)
                continue
            for row in rows:
                sink.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            written += len(rows)
            parse_error_total += batch_parse_errors
            invalid_total += sum(1 for row in rows if not row.get("llm_valid"))
            index += len(rows)
            if written >= next_report or written == len(todo):
                sink.flush()
                elapsed = max(time.monotonic() - started, 1e-6)
                rows_per_second = written / elapsed
                remaining = max(0, len(todo) - written)
                eta_hours = (remaining / rows_per_second / 3600.0) if rows_per_second > 0 else float("inf")
                last = rows[-1]
                print(
                    f"[{written}/{len(todo)}] wrote {last['ticker']} {last['article_id']} "
                    f"batch_size={batch_size} rows_per_sec={rows_per_second:.4f} eta_hours={eta_hours:.2f}",
                    flush=True,
                )
                if next_report == 1:
                    next_report = 25
                while next_report <= written:
                    next_report += 25
    # Generation diagnostics: surface parse failures / invalid rows so a broken extractor run is
    # not mistaken for clean output. A high parse_error_fraction should gate downstream reportable use.
    diagnostics = {
        "output_jsonl": str(args.output_jsonl),
        "model_id": model_id,
        "prompt_hash": prompt_hash,
        "extractor_temperature": float(args.temperature),
        "model_available_utc": model_available_utc,
        "rows_written_this_run": written,
        "parse_error_count": parse_error_total,
        "parse_error_fraction": (parse_error_total / written) if written else 0.0,
        "invalid_llm_row_count": invalid_total,
        "invalid_llm_row_fraction": (invalid_total / written) if written else 0.0,
    }
    diagnostics_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".generation_diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True), flush=True)
    print(f"Done. wrote={written} output={args.output_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
