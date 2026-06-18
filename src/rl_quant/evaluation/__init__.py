"""Evaluation layer: statistical credibility, baselines, and sequential-evaluation helpers.

Part of the protocol-first layered architecture (see architecture_migration_plan.md). Currently re-exports the
statistical-credibility metrics; baseline/sequential-eval helpers still live in their workflow modules and
are migrated in later phases."""

from rl_quant.evaluation.statistical import (
    DSR_PROMOTION_CONFIDENCE,
    PSR_MIN_CREDIBLE_OBSERVATIONS,
    PromotionVerdict,
    deflated_sharpe_promotion_verdict,
    deflated_sharpe_ratio,
    effective_sample_size,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    psr_is_credible,
)

__all__ = [
    "DSR_PROMOTION_CONFIDENCE",
    "PSR_MIN_CREDIBLE_OBSERVATIONS",
    "PromotionVerdict",
    "deflated_sharpe_promotion_verdict",
    "deflated_sharpe_ratio",
    "effective_sample_size",
    "expected_maximum_sharpe",
    "probabilistic_sharpe_ratio",
    "psr_is_credible",
]
