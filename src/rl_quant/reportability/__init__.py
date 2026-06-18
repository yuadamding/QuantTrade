"""Reportability layer: judges whether a run may CLAIM mechanical / real-executable reportability, from
config flags AND persisted/streamed decision-log rows -- separate from statistical credibility (evaluation
layer). Part of the protocol-first layered architecture (see architecture_migration_plan.md).

``rl_quant.reportability`` is now a package; the gate lives in ``reportability.decision_log`` and is
re-exported here so the old import path is unchanged."""

from rl_quant.reportability.baselines import (
    QUOTE_CONDITIONAL_STRESS,
    REQUIRED_BASELINES,
    REQUIRED_STRESS,
    assert_baseline_stress_coverage,
    validate_baseline_stress_coverage,
)
from rl_quant.reportability.decision_log import (
    REQUIRED_DECISION_LOG_FIELDS,
    ReportabilityIssue,
    ReportabilityVerdict,
    evaluate_decision_log_reportability,
)

__all__ = [
    "QUOTE_CONDITIONAL_STRESS",
    "REQUIRED_BASELINES",
    "REQUIRED_DECISION_LOG_FIELDS",
    "REQUIRED_STRESS",
    "ReportabilityIssue",
    "ReportabilityVerdict",
    "assert_baseline_stress_coverage",
    "evaluate_decision_log_reportability",
    "validate_baseline_stress_coverage",
]
