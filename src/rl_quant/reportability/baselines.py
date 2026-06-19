"""Baseline / stress-grid coverage gate -- a reportable run must be COMPARED, not reported in isolation.

The canonical contract (the required baseline/stress IDs + the pure coverage check) now lives ONE layer down
in ``rl_quant.protocol.reportability_contract`` so that BOTH this ``reportability`` gate AND the lower
``evaluation`` layer can import a single source -- evaluation sits below reportability and cannot import from
it, which is what previously forced it to re-declare the names and drift (the review's #6). This module is a
pure re-export shim kept for the existing ``rl_quant.reportability`` import path; the definitions are
unchanged.
"""

from __future__ import annotations

from rl_quant.protocol.reportability_contract import (
    QUOTE_CONDITIONAL_STRESS,
    REQUIRED_BASELINE_SPECS,
    REQUIRED_BASELINES,
    REQUIRED_STRESS,
    REQUIRED_STRESS_SPECS,
    BaselineSpec,
    StressScenarioSpec,
    assert_baseline_stress_coverage,
    validate_baseline_stress_coverage,
)

__all__ = [
    "QUOTE_CONDITIONAL_STRESS",
    "REQUIRED_BASELINES",
    "REQUIRED_BASELINE_SPECS",
    "REQUIRED_STRESS",
    "REQUIRED_STRESS_SPECS",
    "BaselineSpec",
    "StressScenarioSpec",
    "assert_baseline_stress_coverage",
    "validate_baseline_stress_coverage",
]
