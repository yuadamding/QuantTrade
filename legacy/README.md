# `legacy/` — quarantined engineered-feature producers (NOT part of the audited path)

This directory holds the **offline engineered-feature producers** that were moved OUT of the `rl_quant` package
on 2026-06-26 for the **no-feature-engineering** audit. They are kept for data provenance and history, but they
are **not importable from `rl_quant`** and are **never used by the audited training/evaluation path**.

The audited Phase-1 path (`rl_quant.datasets` / `models` / `training` / `evaluation.statistical`, driven by
`../../training/train_phase1.py`) consumes only raw inputs — raw 1-second OHLCV bars, raw as-of fundamental
covariates, raw per-article LLM news scores, and raw-price forward-return labels. `tests/test_no_feature_engineering.py`
fails CI if that path ever imports `rl_quant.features` or references any engineered artifact below.

## Contents

| Path | What it is | Why quarantined |
|---|---|---|
| `features/stock_covariates.py` | `ACTION_COVARIATE_FEATURE_NAMES` — engineered covariate ratios (log_market_cap, revenue_yoy_growth, margins, debt/assets, recency, news-count windows…), `action_covariates` tensors, `feature_schema.json` writer | Engineered features |
| `features/news_llm.py` | `NEWS_LLM_AGGREGATE_FEATURE_NAMES` — windowed news aggregates (1h/1d/7d/30d weighted counts, net sentiment, novelty-weighted sentiment, …) | Engineered features |
| `scripts/build_news_llm_aggregates.py` | builds the `action_features` engineered news sidecar | Feature builder |
| `scripts/build_news_llm_features.py` | per-article LLM feature extraction | Feature builder |
| `scripts/build_news_article_table.py` | `news_article_ticker_llm` table | Feature builder |
| `scripts/generate_qwen_news_precomputed.py` | Qwen3 "financial news feature extractor" (sentiment/materiality/novelty/event flags). Also the upstream producer of the per-article `sentiment_score` the model uses as a raw input | Feature extractor (`trust_remote_code=True` — pin model/revision before any reuse) |
| `scripts/build_stock_covariate_silver_features.py` | thin wrapper for the silver-covariate builder | Feature builder |
| `workflows_commands/build_stock_covariate_silver_features.py` | the silver-covariate command (heavy logic) | Feature builder |

## Notes
- **Kept in the package:** `rl_quant.features.action_risk` — that is action-universe / risk *metadata* (tickers,
  asset classes, risk flags), not a predictive feature; the decision-framework / reportability subsystem uses it.
- These files still contain `from rl_quant... import ...` lines and are **not runnable as-is** from here; they are
  reference/provenance code. To run one, restore it into the package deliberately (and expect the compliance test
  to flag the audited path only — these scripts are never on it).
- The per-article `sentiment_score` already materialized into the TOP50/TOP2000 `news.jsonl` partitions remains a
  raw model input; its **anachronistic scorer** caveat (the `model_available_timestamp_ms = 1000` sentinel) is
  documented in the top-level `README.md` — news-driven results are not point-in-time-clean for a reportable
  backtest until re-scored with a period-correct model.
