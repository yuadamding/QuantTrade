"""Feature builders for model-ready RL datasets.

Model 1 (second_to_hour) uses ``news_llm`` (LLM news aggregates), ``stock_covariates`` (point-in-time
covariates), and ``action_risk``. The per-second market-context reducer lives in the dataset-build script
``scripts/build_hourly_transformer_dataset.py`` (``aggregate_stock_features``), not here. Import submodules
directly (e.g. ``from rl_quant.features.news_llm import ...``)."""
