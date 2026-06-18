"""Backward-compatibility shim. The statistical-credibility metrics moved to the evaluation layer
(``rl_quant.evaluation.statistical``) in the protocol-first reorganization; this re-export keeps the old
import path working (see architecture_migration_plan.md)."""

from rl_quant.evaluation.statistical import (
    DSR_PROMOTION_CONFIDENCE,
    PSR_MIN_CREDIBLE_OBSERVATIONS,
    PromotionVerdict,
    block_bootstrap_confidence_interval,
    deflated_sharpe_promotion_verdict,
    deflated_sharpe_ratio,
    effective_sample_size,
    expected_maximum_sharpe,
    hansens_spa,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    psr_is_credible,
    walk_forward_degradation_ratio,
    white_reality_check,
)

__all__ = [
    "DSR_PROMOTION_CONFIDENCE",
    "PSR_MIN_CREDIBLE_OBSERVATIONS",
    "PromotionVerdict",
    "block_bootstrap_confidence_interval",
    "deflated_sharpe_promotion_verdict",
    "deflated_sharpe_ratio",
    "effective_sample_size",
    "expected_maximum_sharpe",
    "hansens_spa",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "psr_is_credible",
    "walk_forward_degradation_ratio",
    "white_reality_check",
]
