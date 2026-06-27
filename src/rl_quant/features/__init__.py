"""Action-universe / risk metadata for the action set.

Only ``action_risk`` (action-universe + risk metadata: tickers, asset classes, risk flags) lives here now. It is
NOT a predictive feature and is used by the (non-Phase-1) decision-framework / reportability subsystem. The
engineered-FEATURE producers that used to live here -- ``stock_covariates`` (engineered covariate ratios) and
``news_llm`` (windowed news aggregates) -- were quarantined OUT of the package to ``legacy/`` for the
no-feature-engineering audit; they are never imported by the training/evaluation path (enforced by
``tests/test_no_feature_engineering.py``). Import the submodule directly: ``from rl_quant.features.action_risk
import ...``."""
