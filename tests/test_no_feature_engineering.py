"""Compliance guard for the "no feature engineering" rule on the AUDITED Phase-1 path.

The two-stage Phase-1 framework (the driver's import set) must consume ONLY raw inputs -- raw 1-second OHLCV
bars, raw as-of fundamental covariates, raw per-article LLM news scores, and raw-price forward-return labels --
and must NEVER import the engineered-feature producers (``rl_quant.features.*``) or load any engineered
sidecar/schema artifact. The feature producers (``features/stock_covariates.py`` engineered ratio schema,
``features/news_llm.py`` aggregate schema, the ``scripts/build_*`` / ``generate_qwen_news_precomputed`` builders)
exist in the repo for offline data provenance but are DEAD relative to the audited path. These tests make that
boundary a CI failure rather than a claim, so a future edit that wires an engineered feature into training or
evaluation fails loudly.

What "audited path" means here: exactly the modules the Phase-1 driver (../training/train_phase1.py) imports --
``rl_quant.datasets`` (raw_window/daily/splits), ``rl_quant.models`` (context_encoder/decision_policy),
``rl_quant.training`` (the two-stage trainers + designs), and ``rl_quant.evaluation.statistical`` (the verdict
battery). The older ``evaluation.decision_framework`` / ``reportability`` / ``protocol`` subsystem is NOT part of
this path and is out of scope for this guard.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
_PKG = _SRC_ROOT / "rl_quant"

# The exact module set the Phase-1 driver imports (kept in sync with train_phase1.py's imports).
AUDITED_IMPORTS = [
    "rl_quant.datasets",
    "rl_quant.datasets.raw_window",
    "rl_quant.datasets.daily",
    "rl_quant.datasets.splits",
    "rl_quant.models",
    "rl_quant.models.context_encoder",
    "rl_quant.models.decision_policy",
    "rl_quant.training",
    "rl_quant.training.context_pretrain",
    "rl_quant.training.decision_policy",
    "rl_quant.training.designs",
    "rl_quant.evaluation.statistical",
]

# Source files that constitute the audited path (whole datasets/models/training packages + statistical.py).
def _audited_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for pkg in ("datasets", "models", "training"):
        files += sorted((_PKG / pkg).rglob("*.py"))
    files.append(_PKG / "evaluation" / "statistical.py")
    return [f for f in files if "__pycache__" not in f.parts]


# Engineered-feature artifact names that must NEVER appear in the audited path (from the audit's remediation #13).
FORBIDDEN_TOKENS = [
    "rl_quant.features",                    # importing the engineered-feature producers
    "action_covariates",                    # stock_covariates.py model-facing covariate tensor
    "action_covariate_feature_names",
    "ACTION_COVARIATE_FEATURE_NAMES",
    "action_features",                      # build_news_llm_aggregates.py engineered news sidecar
    "action_feature_names",
    "feature_schema",                       # feature_schema.json engineered-schema artifact
    "NEWS_LLM_AGGREGATE_FEATURE_NAMES",     # news_llm.py windowed aggregate features
    "stock_news_llm_v1",                    # aggregate feature namespace
    "news_article_ticker_llm",              # engineered news article table
    "_sidecar",                             # any *_sidecar* feature artifact
]

# Raw as-of fundamental/corporate-action covariate columns the encoder is ALLOWED to consume (NOT engineered
# ratios/growth like log_market_cap / revenue_yoy_growth / net_income_margin / debt_to_assets / days_since_listed).
ALLOWED_COV_FIELDS = {
    "market_cap", "share_class_shares_outstanding", "financial_revenue", "financial_net_income",
    "financial_assets", "financial_liabilities", "financial_cash", "financial_operating_cashflow",
    "dividend_cash_amount", "split_ratio", "is_common_stock", "is_adr_or_foreign",
}
# Tell-tale substrings of ENGINEERED covariate features (ratios, growth, recency, counts) -- never allowed.
ENGINEERED_COV_MARKERS = ("yoy", "growth", "margin", "_to_", "ratio_", "days_since", "log_", "_count",
                          "novelty", "sentiment_count", "publisher", "_zscore", "_pct", "rolling")


class NoFeatureEngineeringTests(unittest.TestCase):
    def test_audited_imports_pull_in_no_engineered_feature_module(self) -> None:
        """Import the audited Phase-1 module set in a FRESH interpreter; assert it transitively loads none of the
        engineered-feature producers. A subprocess avoids sys.modules pollution from other tests."""
        probe = (
            "import importlib, sys, json\n"
            f"mods = {AUDITED_IMPORTS!r}\n"
            "for m in mods:\n"
            "    importlib.import_module(m)\n"
            "bad = sorted(k for k in sys.modules if k == 'rl_quant.features' "
            "or k.startswith('rl_quant.features.') "
            "or k in ('rl_quant.action_risk',) "
            "or k.endswith('.stock_covariates') or k.endswith('.news_llm'))\n"
            "print(json.dumps(bad))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(out.returncode, 0, f"probe failed:\nSTDOUT={out.stdout}\nSTDERR={out.stderr}")
        leaked = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "[]"
        self.assertEqual(
            leaked, "[]",
            "The audited Phase-1 import set transitively loaded engineered-feature module(s): "
            f"{leaked}. The training/evaluation path must consume only raw inputs -- move the offending import "
            "out of the audited path (datasets/models/training/evaluation.statistical).",
        )

    def test_audited_source_references_no_engineered_artifact(self) -> None:
        """No audited source file may name an engineered-feature artifact (tensor/schema/sidecar/aggregate)."""
        hits: list[str] = []
        for path in _audited_files():
            text = path.read_text()
            for token in FORBIDDEN_TOKENS:
                if token in text:
                    rel = path.relative_to(_SRC_ROOT.parent)
                    hits.append(f"{rel}: references forbidden engineered-feature token '{token}'")
        self.assertEqual(
            hits, [],
            "Engineered-feature artifacts referenced inside the audited Phase-1 path:\n" + "\n".join(hits)
            + "\nThe audited path must not load action_covariates/action_features/feature_schema/aggregate "
            "news features or any *_sidecar*. Keep those in the offline (quarantined) feature pipeline.",
        )

    def test_covariate_fields_are_raw_not_engineered(self) -> None:
        """The encoder's covariate columns must be the raw as-of fundamental allowlist, not engineered ratios."""
        sys.path.insert(0, str(_SRC_ROOT))
        try:
            from rl_quant.datasets import COV_FIELDS, NEWS_RAW_DIM
        finally:
            sys.path.remove(str(_SRC_ROOT))
        unexpected = set(COV_FIELDS) - ALLOWED_COV_FIELDS
        self.assertEqual(
            unexpected, set(),
            f"COV_FIELDS introduced columns outside the raw as-of fundamental allowlist: {sorted(unexpected)}. "
            "Engineered covariates (ratios/growth/recency/counts) are not permitted as model inputs.",
        )
        for f in COV_FIELDS:
            for marker in ENGINEERED_COV_MARKERS:
                self.assertNotIn(
                    marker, f.lower(),
                    f"COV_FIELDS column '{f}' looks engineered (marker '{marker}'); only raw fields are allowed.",
                )
        # Raw per-article news score only -- NOT a precomputed count/mean/aggregate (those would inflate this dim).
        self.assertEqual(NEWS_RAW_DIM, 1,
                         "NEWS_RAW_DIM must stay 1 (one raw per-article score); >1 implies precomputed news aggregates.")


if __name__ == "__main__":
    unittest.main()
