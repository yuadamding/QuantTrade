"""Evaluation layer: statistical credibility, baselines, and sequential-evaluation helpers.

Part of the protocol-first layered architecture (see architecture_migration_plan.md). Currently re-exports the
statistical-credibility metrics; baseline/sequential-eval helpers still live in their workflow modules and
are migrated in later phases."""

from rl_quant.evaluation.statistical import (
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "probabilistic_sharpe_ratio",
]
