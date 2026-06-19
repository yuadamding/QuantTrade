"""Protocol layer: the CANONICAL baseline / stress-grid contract -- the single source of truth for which
baselines and stress scenarios a reportable run must be COMPARED against.

Why this lives in the ``protocol`` layer (the lowest): the baseline/stress IDs are consumed by BOTH the
``reportability`` gate AND the ``evaluation`` layer (which produces the comparisons and validates run
summaries). Under the protocol-first layering, ``evaluation`` is BELOW ``reportability``, so it cannot import
from it -- which is exactly why ``evaluation.decision_framework`` was forced to hard-code its own baseline/
stress names, drifting from the ``reportability`` gate (the review's #6 name drift). Hoisting the canonical
IDs + the pure coverage check down to ``protocol`` gives every higher layer ONE importable source, so the two
can no longer disagree.

The contract is expressed as STRUCTURED records (``BaselineSpec`` / ``StressScenarioSpec``), each carrying its
logical id, the RATIONALE for requiring it, and the applicability gate (a named boolean flag that, when set,
makes the member conditionally required) -- instead of bare string tuples with the applicability rules baked
into ``if`` branches in the validator. The legacy ``REQUIRED_BASELINES`` / ``REQUIRED_STRESS`` /
``QUOTE_CONDITIONAL_STRESS`` tuples are DERIVED from the records, so every existing import path is unchanged.

This contract COMPLEMENTS, and does not replace, ``ModelManifest``'s reportability validation: the manifest
governs which artifacts a run must carry; this governs which comparison baselines/stress members it must
include. Pure, stdlib only; changes no backtest number.

OUTSTANDING RECONCILIATION (intentionally not encoded here -- it MOVES verdicts, so it needs sign-off): the
producers/summary validators currently emit DIFFERENT on-the-wire names than these logical ids --
``evaluation.second_context`` emits CamelCase (``CASH``, ``RandomSameTurnover``, ``BuyAndHold_<symbol>``) and
``evaluation.decision_framework.validate_reportable_summary`` checks dotted summary paths
(``baselines.CASH``, ``cost_stress.fixed_rollout``). Wiring the gate against real run summaries needs a map
(logical id -> produced/summary key) AND a domain decision on whether the produced stress axis
(fixed_rollout / adaptive, cost multipliers [1.0, 2.0]) actually covers the contract's cost_doubled /
cost_tripled / latency_plus_one_bar members. A future ``produced_key`` field on these records is the natural
home for that map.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    """One required comparison baseline in the reportable grid. ``gate_flag``: if set, the baseline is
    required ONLY when the named boolean flag (an argument of ``validate_baseline_stress_coverage``) is True;
    if None, it is always required."""

    id: str
    rationale: str
    gate_flag: str | None = None


@dataclass(frozen=True)
class StressScenarioSpec:
    """One required stress scenario in the reportable grid. ``gate_flag``: if set, the scenario is required
    ONLY when the named boolean flag is True (e.g. spread/impact needs crossable quotes); if None, always."""

    id: str
    rationale: str
    gate_flag: str | None = None


# The canonical reportable grid as structured records. buy_and_hold is gated on ``buy_and_hold_applicable``
# (a strategy with no natural long-only benchmark waives it); spread_impact is gated on
# ``quote_data_available`` (only meaningful with crossable quotes).
REQUIRED_BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec("cash", "Holding cash is the do-nothing floor every strategy must clear.", None),
    BaselineSpec("buy_and_hold", "A passive long-only benchmark, where one exists.", "buy_and_hold_applicable"),
    BaselineSpec("random_action_distribution",
                 "A policy matched on the action distribution isolates timing/selection from the mix.", None),
    BaselineSpec("same_turnover_random",
                 "A random policy matched on TURNOVER isolates selection skill from turnover -- the baseline "
                 "that matters most, since many policies 'win' only by trading more.", None),
)
REQUIRED_STRESS_SPECS: tuple[StressScenarioSpec, ...] = (
    StressScenarioSpec("cost_doubled", "Robustness to a 2x transaction-cost shock.", None),
    StressScenarioSpec("cost_tripled", "Robustness to a 3x transaction-cost shock.", None),
    StressScenarioSpec("latency_plus_one_bar", "Robustness to one extra bar of execution latency.", None),
    StressScenarioSpec("spread_impact", "Spread/impact stress -- requires crossable quote data to model.",
                       "quote_data_available"),
)

# Legacy tuples, DERIVED from the records so the two cannot drift. REQUIRED_BASELINES is the default-required
# set (buy_and_hold included; its gate defaults required). REQUIRED_STRESS is the unconditional stress set;
# QUOTE_CONDITIONAL_STRESS is the subset gated on crossable-quote availability.
REQUIRED_BASELINES: tuple[str, ...] = tuple(spec.id for spec in REQUIRED_BASELINE_SPECS)
REQUIRED_STRESS: tuple[str, ...] = tuple(spec.id for spec in REQUIRED_STRESS_SPECS if spec.gate_flag is None)
QUOTE_CONDITIONAL_STRESS: tuple[str, ...] = tuple(
    spec.id for spec in REQUIRED_STRESS_SPECS if spec.gate_flag == "quote_data_available"
)


def _required_ids(specs: Sequence[BaselineSpec | StressScenarioSpec], flags: Mapping[str, bool]) -> set[str]:
    """The ids required under the given applicability flags: a member with no gate is always required; a
    gated member is required iff its flag is True. A spec naming an unknown flag is a programming error and
    raises (fail closed) rather than silently dropping the requirement."""
    return {spec.id for spec in specs if spec.gate_flag is None or flags[spec.gate_flag]}


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
    flags = {"buy_and_hold_applicable": buy_and_hold_applicable, "quote_data_available": quote_data_available}
    included_b = {str(b).lower() for b in included_baselines}
    included_s = {str(s).lower() for s in included_stress}
    required_b = _required_ids(REQUIRED_BASELINE_SPECS, flags)
    required_s = _required_ids(REQUIRED_STRESS_SPECS, flags)
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
