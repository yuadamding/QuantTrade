#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.features.news_llm import (  # noqa: E402
    DEFAULT_NEWS_LLM_FALLBACK_MODEL_ID,
    DEFAULT_NEWS_LLM_PRIMARY_MODEL_ID,
    DEFAULT_NEWS_LLM_SECONDARY_MODEL_ID,
    DEFAULT_NEWS_LLM_SERVING_ENGINE,
    DEFAULT_NEWS_LLM_STRUCTURED_OUTPUT,
    DETERMINISTIC_NEWS_LLM_MODEL_ID,
    NEWS_LLM_ARTICLE_TICKER_FIELDS,
    NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH,
    NEWS_LLM_EXTRACT_SCHEMA_VERSION,
    build_deterministic_news_llm_rows,
    canonical_symbol,
    default_news_llm_analyst_model_policy,
    parse_timestamp_ms,
    read_manifest,
    read_news_article_rows,
    stable_json_hash,
    validate_news_llm_rows,
    write_news_llm_feature_outputs,
)


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_ARTICLE_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "news_articles_v1" / "top500_2023_to_present"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "news_llm_v1" / "top500_2023_to_present"
LOCAL_MODEL_PRESETS = {
    "qwen3_6_27b": Path("../LLM/Qwen3.6-27B/download_manifest.json"),
    "qwen3_1_7b": Path("../LLM/Qwen3-1.7B/download_manifest.json"),
    "gemma4_26b_a4b_it": Path("../LLM/gemma-4-26B-A4B-it/download_manifest.json"),
    "mistral_small_3_2_24b": Path("../LLM/Mistral-Small-3.2-24B-Instruct-2506/download_manifest.json"),
}
DEFAULT_LOCAL_MODEL_PRESET = "qwen3_6_27b"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build audited article-ticker news_llm_v1 features. The default provider is a deterministic "
            "baseline; pass --precomputed-jsonl to import externally generated structured LLM outputs."
        )
    )
    parser.add_argument("--article-root", type=Path, default=DEFAULT_ARTICLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--precomputed-jsonl", type=Path)
    parser.add_argument("--model-id", default=DETERMINISTIC_NEWS_LLM_MODEL_ID)
    parser.add_argument("--provider", default="deterministic_baseline")
    parser.add_argument("--model-available-timestamp-utc")
    parser.add_argument("--model-training-cutoff-utc", default="not_applicable_deterministic_baseline")
    parser.add_argument("--primary-model-id", default=DEFAULT_NEWS_LLM_PRIMARY_MODEL_ID)
    parser.add_argument("--secondary-model-id", default=DEFAULT_NEWS_LLM_SECONDARY_MODEL_ID)
    parser.add_argument("--fallback-model-id", default=DEFAULT_NEWS_LLM_FALLBACK_MODEL_ID)
    parser.add_argument("--serving-engine", default=DEFAULT_NEWS_LLM_SERVING_ENGINE)
    parser.add_argument("--structured-output", default=DEFAULT_NEWS_LLM_STRUCTURED_OUTPUT)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--local-model-preset",
        choices=sorted(LOCAL_MODEL_PRESETS),
        default=DEFAULT_LOCAL_MODEL_PRESET,
        help="Named local pretrained LLM manifest preset used when importing precomputed outputs.",
    )
    parser.add_argument(
        "--local-model-manifest",
        type=Path,
        help="Override the preset manifest used to pin local pretrained LLM metadata.",
    )
    parser.add_argument(
        "--include-external-article-tickers",
        action="store_true",
        help="Keep article-ticker rows outside the article table source universe. Default restricts to the selected universe.",
    )
    parser.add_argument("--vendor-latency-seconds", type=int, default=300)
    parser.add_argument("--processing-latency-seconds", type=int, default=60)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def read_precomputed_rows(
    path: Path,
    *,
    allowed_tickers: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"line {line_number}: expected a JSON object")
                continue
            row = {field: payload.get(field) for field in NEWS_LLM_ARTICLE_TICKER_FIELDS}
            if row.get("ticker") is not None:
                row["ticker"] = canonical_symbol(str(row["ticker"]))
            if allowed_tickers is not None and canonical_symbol(str(row.get("ticker", ""))) not in allowed_tickers:
                continue
            if row.get("llm_schema_version") in (None, ""):
                row["llm_schema_version"] = NEWS_LLM_EXTRACT_SCHEMA_VERSION
            if row.get("llm_schema_hash") in (None, ""):
                row["llm_schema_hash"] = NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH
            rows.append(row)
    errors.extend(validate_news_llm_rows(rows))
    return rows, errors


def local_model_manifest_path(args: argparse.Namespace) -> Path:
    return args.local_model_manifest or LOCAL_MODEL_PRESETS[args.local_model_preset]


def resolve_project_relative_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def workspace_relative_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return (Path("..") / resolved.relative_to(PROJECT_ROOT.parent)).as_posix()
    except ValueError:
        return path.name


def sanitize_local_model_manifest(manifest: dict[str, object]) -> dict[str, object]:
    sanitized = dict(manifest)
    local_path = sanitized.get("local_path")
    if local_path not in (None, ""):
        sanitized["local_path"] = workspace_relative_path(Path(str(local_path)))
    return sanitized


def analyst_model_policy_from_args(args: argparse.Namespace) -> dict[str, object]:
    return default_news_llm_analyst_model_policy(
        primary_model_id=args.primary_model_id,
        secondary_model_id=args.secondary_model_id,
        fallback_model_id=args.fallback_model_id,
        serving_engine=args.serving_engine,
        structured_output=args.structured_output,
        temperature=0.0,
        top_p=args.top_p,
    )


def resolve_model_metadata(args: argparse.Namespace) -> tuple[str, int, str, str, dict[str, object] | None]:
    manifest_path = local_model_manifest_path(args)
    readable_manifest_path = resolve_project_relative_path(manifest_path)
    if args.precomputed_jsonl and readable_manifest_path.exists():
        manifest = sanitize_local_model_manifest(json.loads(readable_manifest_path.read_text()))
        model_id = args.model_id
        if model_id == DETERMINISTIC_NEWS_LLM_MODEL_ID:
            model_id = f"{manifest['repo_id']}@{manifest['revision']}"
        available_utc = args.model_available_timestamp_utc or str(manifest.get("downloaded_at_utc", ""))
        provider = args.provider if args.provider != "deterministic_baseline" else "local_transformers"
        training_cutoff = args.model_training_cutoff_utc
        if training_cutoff == "not_applicable_deterministic_baseline":
            training_cutoff = "unknown_for_downloaded_pretrained_model"
        return model_id, parse_timestamp_ms(available_utc), training_cutoff, provider, manifest
    if args.precomputed_jsonl and not args.model_available_timestamp_utc:
        raise SystemExit(
            "Precomputed LLM rows require --model-available-timestamp-utc or an existing "
            f"--local-model-manifest/--local-model-preset manifest. Missing: {manifest_path}"
        )
    model_id = args.model_id
    if args.precomputed_jsonl and model_id == DETERMINISTIC_NEWS_LLM_MODEL_ID:
        model_id = args.primary_model_id
    available_utc = args.model_available_timestamp_utc or "1970-01-01T00:00:00+00:00"
    provider = args.provider
    if args.precomputed_jsonl and provider == "deterministic_baseline":
        provider = args.serving_engine
    return model_id, parse_timestamp_ms(available_utc), args.model_training_cutoff_utc, provider, None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    article_manifest = read_manifest(args.article_root / "manifest.json")
    analyst_model_policy = analyst_model_policy_from_args(args)
    source_symbols = {
        canonical_symbol(str(symbol))
        for symbol in article_manifest.get("symbols_with_source_news", [])
    }
    allowed_tickers = None if args.include_external_article_tickers or not source_symbols else source_symbols
    model_id, model_available_ms, training_cutoff, provider, local_model_manifest = resolve_model_metadata(args)
    if args.precomputed_jsonl:
        rows, errors = read_precomputed_rows(args.precomputed_jsonl, allowed_tickers=allowed_tickers)
    else:
        articles = read_news_article_rows(args.article_root)
        rows = build_deterministic_news_llm_rows(
            articles,
            model_id=model_id,
            model_available_timestamp_ms=model_available_ms,
            model_training_cutoff_utc=training_cutoff,
            vendor_latency_seconds=args.vendor_latency_seconds,
            processing_latency_seconds=args.processing_latency_seconds,
            allowed_tickers=allowed_tickers,
        )
        errors = []
        provider = "deterministic_baseline"
    if args.strict and errors:
        preview = "; ".join(errors[:10])
        raise SystemExit(f"news_llm_v1 feature build failed: {preview}")
    manifest = write_news_llm_feature_outputs(
        rows=rows,
        output_root=args.output_root,
        article_manifest=article_manifest,
        model_id=model_id,
        model_available_timestamp_ms=model_available_ms,
        model_training_cutoff_utc=training_cutoff,
        provider=provider,
        errors=errors,
        analyst_model_policy=analyst_model_policy,
    )
    if local_model_manifest is not None:
        model_manifest_path = args.output_root / "local_model_manifest.json"
        model_manifest_path.write_text(json.dumps(local_model_manifest, indent=2, sort_keys=True) + "\n")
        manifest["local_model_manifest_path"] = workspace_relative_path(model_manifest_path)
        manifest["local_model_manifest_hash"] = stable_json_hash(local_model_manifest)
        (args.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
        )
    manifest_hash = stable_json_hash(manifest)
    print(f"News LLM rows: {manifest['row_count']} | symbols: {manifest['symbol_count']}")
    print(f"News LLM manifest hash: {manifest_hash}")
    print(f"News LLM output -> {args.output_root}")
    if manifest["reportability_errors"]:
        print(f"Reportability errors: {manifest['reportability_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
