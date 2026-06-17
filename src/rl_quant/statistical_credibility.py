"""Backward-compatibility shim. The statistical-credibility metrics moved to the evaluation layer
(``rl_quant.evaluation.statistical``) in the protocol-first reorganization; this re-export keeps the old
import path working (see architecture_migration_plan.md)."""

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
