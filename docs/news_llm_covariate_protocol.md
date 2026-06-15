# News LLM Covariate Protocol

This project may use a pretrained language model only as an offline,
point-in-time analyst feature extractor. It must not be used as the trading
policy, as a direct action reasoner, or as an extractor for structured Polygon
fields such as financial statements, dividends, splits, or reference snapshots.
Those structured fields should be computed deterministically; a later
`stock_fundamental_llm_v1` layer may interpret quality/risk from those
deterministic ratios.

## Scope

The `stock_news_llm_v1` layer converts raw Polygon news into a typed
article-ticker feature table and then into decision-aligned action covariates.
It is separate from the existing `stock_covariates_v1` flat covariates.

Pipeline:

```text
raw Polygon news JSONL
-> deduplicated article table
-> article-ticker news_llm_v1 feature table
-> action-level news LLM sidecar
-> optional model-facing action_features append
```

No script fetches current article URLs or article bodies. The extractor uses
only the raw Polygon news payload already downloaded.

The currently downloaded local model is a smoke-test and small experiment
checkpoint:

```text
../LLM/Qwen3-1.7B
repo_id: Qwen/Qwen3-1.7B
revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
manifest: ../LLM/Qwen3-1.7B/download_manifest.json
```

The recommended under-30B production analyst stack is:

```text
Primary extractor:   Qwen/Qwen3.6-27B
Validator/fallback:  google/gemma-4-26B-A4B-it
Structured fallback: mistralai/Mistral-Small-3.2-24B-Instruct-2506
Serving engine:      vLLM
Output mode:         JSON schema
Temperature:         0.0
Top-p:               1.0
```

This stack is for frozen, cached feature extraction only. Any historical run
using a model before its model availability/training cutoff must be marked
non-reportable unless a stricter point-in-time model policy is documented.

## Artifacts

Article table:

```text
scripts/build_news_article_table.py
data/polygon/stock_covariates/news_articles_v1/top500_2023_to_present/
  news_articles.parquet
  manifest.json
```

Feature table:

```text
scripts/build_news_llm_features.py
data/polygon/stock_covariates/news_llm_v1/top500_2023_to_present/
  news_article_ticker_llm.parquet
  manifest.json
```

Decision sidecars:

```text
scripts/build_news_llm_aggregates.py
.../hour_from_second_1s/partitions/<partition>/action_news_llm_covariates.pt
```

Training does not consume the news LLM sidecar by default. Use:

```bash
conda run -n ml1 python scripts/train_hourly_from_second_protocol_partitions.py \
  --news-llm-sidecar required
```

## Article-Ticker Schema

Each row is one unique `(article_id, ticker)` pair. Important fields include:

- `source_available_timestamp_ms`
- `llm_feature_available_timestamp_ms`
- `ticker_relevance`
- `is_primary_ticker`
- `company_specificity`
- `is_broad_market_or_sector`
- sentiment and uncertainty scores
- materiality and novelty scores
- event flags for earnings, guidance, analyst rating, M&A, regulatory,
  litigation, macro, sector, management, capital return, product, and AI/tech
- `confidence`
- `llm_valid`
- `llm_model_id`
- `llm_prompt_hash`
- `llm_schema_version`
- `llm_schema_hash`

The current in-repo implementation supports a deterministic baseline extractor
and importing externally generated structured JSONL. A real pretrained LLM
output should be imported as precomputed JSONL after the extraction process has
recorded model id, prompt hash, schema hash, temperature, retrieval policy, and
training cutoff.

For top-500 stock experiments, `scripts/build_news_llm_features.py` restricts
article-ticker rows to the article-table source universe by default. Pass
`--include-external-article-tickers` only for a deliberate cross-universe
diagnostic.

When importing rows produced by the downloaded local Qwen model, keep the model
manifest attached:

```bash
conda run -n ml1 python scripts/build_news_llm_features.py \
  --article-root ../data/polygon/stock_covariates/news_articles_v1/top500_2023_to_present \
  --precomputed-jsonl data/examples/frozen_qwen_news_outputs.jsonl \
  --local-model-preset qwen3_6_27b \
  --provider vllm \
  --model-available-timestamp-utc 2026-06-15T00:00:00+00:00 \
  --strict
```

Available local manifest presets are:

```text
qwen3_6_27b
gemma4_26b_a4b_it
mistral_small_3_2_24b
qwen3_1_7b
```

The `qwen3_1_7b` preset points to the downloaded smoke-test checkpoint. The
larger presets are expected relative locations for frozen local manifests after
those models are downloaded or served. Use `--local-model-manifest` only when
importing outputs from a different frozen checkpoint.

Each `stock_news_llm_v1` feature manifest records:

```json
{
  "llm_feature_group": "stock_news_llm_v1",
  "primary_model_id": "Qwen/Qwen3.6-27B",
  "primary_model_role": "main_extractor",
  "secondary_model_id": "google/gemma-4-26B-A4B-it",
  "secondary_model_role": "validator_or_fallback",
  "fallback_model_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
  "fallback_model_role": "structured_output_fallback",
  "serving_engine": "vllm",
  "structured_output": "json_schema",
  "temperature": 0.0,
  "top_p": 1.0,
  "no_external_retrieval": true,
  "cached_outputs_only": true
}
```

The same policy object includes
`retrospective_historical_policy.reportable_for_2023_to_2026_backtest = false`
unless the extractor model was actually available at the first decision time.

Local inference dependencies are optional:

```bash
conda run -n ml1 python -m pip install -e ".[llm]"
```

## Reportability Rules

A backtest using `stock_news_llm_v1` is reportable only if:

- `llm_feature_available_timestamp_ms <= decision_timestamp_ms` for every
  model-facing value.
- The extractor model availability timestamp is at or before the first decision
  timestamp in the evaluated dataset.
- The extraction uses temperature `0`.
- No external retrieval is used during extraction.
- Prompt hash, schema hash, and model id are recorded.
- The model training cutoff is recorded, or the extractor is the deterministic
  non-trained baseline.
- Source coverage is explicit, so missing news files are not confused with
  known zero news.

If any condition fails, the sidecar records the issue in
`action_news_llm_reportability_errors`.

## Model Input Format

The sidecar stores a typed tensor group:

```text
action_news_llm_features: [decisions, actions, news_llm_feature_count]
action_news_llm_mask: [decisions, actions, news_llm_feature_count]
action_news_llm_available_timestamps_ms: [decisions, actions, news_llm_feature_count]
action_news_llm_age_seconds: [decisions, actions, news_llm_feature_count]
```

For optional model-facing use, the loader appends:

```text
stock_news_llm_v1.<feature>
stock_news_llm_v1_mask.<feature>
```

Mask channels and missing flags are kept raw during normalization.

`stock_fundamental_llm_v1` is intentionally not model-facing yet. It remains a
planned typed group until there is a separate cached silver builder with the
same manifest, mask, age, and reportability contract.

## Aggregate Features

The compact aggregate vector includes weighted counts over 1 hour, 1 day,
7 days, and 30 days; net sentiment over 1, 7, and 30 days; material positive
and negative counts; company-specific and broad-market fractions; event counts;
novelty-weighted sentiment; confidence diagnostics; invalid-output count; and
time since last material or negative news.

Count-like features use `log1p` compression. Multi-ticker articles are deduped
and weighted by ticker relevance and company specificity.
