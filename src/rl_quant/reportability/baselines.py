"""Baseline / stress-grid coverage gate -- a reportable run must be COMPARED, not reported in isolation.

The README already states a reportable run should include cash / buy-and-hold / random-action-distribution /
same-turnover baselines and cost & latency stress tests, and that a run is non-reportable when "required
baselines or stress tests are missing" -- but nothing enforced it in code (the review's #13/#17). This lifts
that contract into a fail-closed gate: given the baselines and stress scenarios a run ACTUALLY produced, it
fails closed if any required member is absent.

The SAME-TURNOVER-RANDOM baseline matters most: many policies "win" only by changing turnover, not by
selecting better -- comparing against a random policy matched on turnover isolates selection skill from
turnover. Pure, stdlib only; changes no backtest number. The run/report layer supplies the included sets.
"""

from __future__ import annotations

from collections.abc import Sequence

# The minimum baseline grid for a reportable run (README + review). buy_and_hold is "where applicable"
# (pass buy_and_hold_applicable=False for a strategy with no natural long-only benchmark).
REQUIRED_BASELINES: tuple[str, ...] = (
    "cash",
    "buy_and_hold",
    "random_action_distribution",
    "same_turnover_random",
)
# The minimum cost/latency stress scenarios.
REQUIRED_STRESS: tuple[str, ...] = (
    "cost_doubled",
    "cost_tripled",
    "latency_plus_one_bar",
)
# Additional stress required ONLY when crossable quote data exists (so spread/impact can be modelled).
QUOTE_CONDITIONAL_STRESS: tuple[str, ...] = ("spread_impact",)


def validate_baseline_stress_coverage(
    included_baselines: Sequence[str],
    included_stress: Sequence[str],
    *,
    buy_and_hold_applicable: bool = True,
    quote_data_available: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Validate that a run's INCLUDED baselines/stress cover the required grid. Returns (ok, issues). Names
    are compared case-insensitively. ``buy_and_hold_applicable=False`` drops the buy-and-hold requirement (no
    natural long-only benchmark); ``quote_data_available=True`` additionally requires the spread/impact stress
    (only meaningful with crossable quotes). A reportable run must satisfy this -- otherwise a result is
    reported without the comparisons that distinguish skill from turnover/cost/latency luck."""
    included_b = {str(b).lower() for b in included_baselines}
    included_s = {str(s).lower() for s in included_stress}
    required_b = set(REQUIRED_BASELINES)
    if not buy_and_hold_applicable:
        required_b.discard("buy_and_hold")
    required_s = set(REQUIRED_STRESS)
    if quote_data_available:
        required_s |= set(QUOTE_CONDITIONAL_STRESS)
    issues = [f"missing required baseline: {name}" for name in sorted(required_b - included_b)]
    issues += [f"missing required stress scenario: {name}" for name in sorted(required_s - included_s)]
    return (not issues, tuple(issues))


def assert_baseline_stress_coverage(
    included_baselines: Sequence[str],
    included_stress: Sequence[str],
    *,
    buy_and_hold_applicable: bool = True,
    quote_data_available: bool = False,
) -> None:
    """Fail closed: raise ValueError if the run does not cover the required baseline/stress grid."""
    ok, issues = validate_baseline_stress_coverage(
        included_baselines, included_stress,
        buy_and_hold_applicable=buy_and_hold_applicable, quote_data_available=quote_data_available,
    )
    if not ok:
        raise ValueError("reportability contract violation (baseline/stress grid): " + "; ".join(issues))
