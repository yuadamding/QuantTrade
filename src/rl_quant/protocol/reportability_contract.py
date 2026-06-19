"""Protocol layer: the CANONICAL baseline / stress-grid contract -- the single source of truth for which
baselines and stress scenarios a reportable run must be COMPARED against.

Why this lives in the ``protocol`` layer (the lowest): the baseline/stress IDs are consumed by BOTH the
``reportability`` gate AND the ``evaluation`` layer (which produces the comparisons and validates run
summaries). Under the protocol-first layering, ``evaluation`` is BELOW ``reportability``, so it cannot import
from it -- which is exactly why ``evaluation.decision_framework`` was forced to hard-code its own baseline/
stress names, drifting from the ``reportability`` gate (the review's #6 name drift). Hoisting the canonical
IDs + the pure coverage check down to ``protocol`` gives every higher layer ONE importable source, so the two
can no longer disagree.

This contract COMPLEMENTS, and does not replace, ``ModelManifest``'s reportability validation: the manifest
governs which artifacts a run must carry; this governs which comparison baselines/stress members it must
include. Pure, stdlib only; changes no backtest number.

OUTSTANDING RECONCILIATION (intentionally not done here -- it MOVES verdicts, so it needs sign-off): the
producers/summary validators currently emit DIFFERENT on-the-wire names than these logical IDs --
``evaluation.second_context`` emits CamelCase (``CASH``, ``RandomSameTurnover``, ``BuyAndHold_<symbol>``) and
``evaluation.decision_framework.validate_reportable_summary`` checks dotted summary paths
(``baselines.CASH``, ``cost_stress.fixed_rollout``). Wiring the gate against real run summaries requires a
name map (logical ID -> produced/summary key), which is the natural home for the structured
baseline/stress records. Until then this module is the single DEFINITION; the mapping is a follow-up.
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
