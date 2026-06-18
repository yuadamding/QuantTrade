"""Evaluation layer: statistical credibility, baselines, and sequential-evaluation helpers.

Part of the protocol-first layered architecture (see architecture_migration_plan.md). Currently re-exports the
statistical-credibility metrics; baseline/sequential-eval helpers still live in their workflow modules and
are migrated in later phases."""

from rl_quant.evaluation.ranking import (
    information_coefficient,
    rank_information_coefficient,
    selection_regret,
    top_k_mean_return,
)
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
    statistical_credibility_report,
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
    "information_coefficient",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "psr_is_credible",
    "rank_information_coefficient",
    "selection_regret",
    "statistical_credibility_report",
    "top_k_mean_return",
    "walk_forward_degradation_ratio",
    "white_reality_check",
]
