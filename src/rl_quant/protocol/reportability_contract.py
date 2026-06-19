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

STRESS GRID -- RECONCILED to production reality (signed off 2026-06): the produced stress axis is the
cost-stress grid's 2x multiplier (cost_doubled) only; there is no 3x multiplier, latency, or spread/impact
stress. REQUIRED_STRESS is therefore just ``cost_doubled`` (+ quote-conditional ``spread_impact``, never
required while quote data is absent); ``cost_tripled`` / ``latency_plus_one_bar`` are retained as
ASPIRATIONAL_STRESS_SPECS to promote when the producer emits them. See the comment above REQUIRED_STRESS_SPECS.

OUTSTANDING RECONCILIATION (intentionally not encoded here -- it MOVES verdicts, so it needs sign-off): the
producers/summary validators still emit DIFFERENT on-the-wire NAMES than these logical ids --
``evaluation.second_context`` emits CamelCase (``CASH``, ``RandomSameTurnover``, ``BuyAndHold_<symbol>``) and
``evaluation.decision_framework.validate_reportable_summary`` checks dotted summary paths
(``baselines.CASH``, ``cost_stress.fixed_rollout``). Wiring the gate against real run summaries still needs a
name map (logical id -> produced/summary key); a future ``produced_key`` field on these records is the natural
home for it. (The stress-COVERAGE question above is now resolved; only the name mapping remains.)
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
# The REQUIRED stress grid is RELAXED to what the production pipeline ACTUALLY produces (verified against the
# evaluation protocol + producers, 2026-06): the cost-stress grid emits a 2x multiplier (-> cost_doubled). It
# does NOT produce a 3x multiplier, an execution-latency stress, or a spread/impact stress, so requiring those
# would (once this gate is wired to real summaries) mark every real run non-reportable. cost_doubled is
# required; spread_impact stays quote-conditional (never required while quote data is absent -- the production
# reality). The 3x-cost and +1-bar-latency scenarios are retained as ASPIRATIONAL_STRESS_SPECS below: promote
# them back into REQUIRED_STRESS_SPECS once the producer emits them (a deliberate, reportability-TIGHTENING
# change, not a silent one).
REQUIRED_STRESS_SPECS: tuple[StressScenarioSpec, ...] = (
    StressScenarioSpec("cost_doubled",
                       "Robustness to a 2x transaction-cost shock (produced by the cost-stress grid).", None),
    StressScenarioSpec("spread_impact", "Spread/impact stress -- requires crossable quote data to model.",
                       "quote_data_available"),
)
# Documented but NOT YET REQUIRED -- the production pipeline does not produce these (no 3x cost multiplier, no
# execution-latency stress). They are the intended next bar; promote into REQUIRED_STRESS_SPECS when produced.
ASPIRATIONAL_STRESS_SPECS: tuple[StressScenarioSpec, ...] = (
    StressScenarioSpec("cost_tripled", "Robustness to a 3x transaction-cost shock (not yet produced).", None),
    StressScenarioSpec("latency_plus_one_bar",
                       "Robustness to one extra bar of execution latency (not yet produced).", None),
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


# Explicit map from a PRODUCED baseline name (what evaluation.second_context emits and
# decision_framework.validate_reportable_summary checks) to its canonical logical id. This is the BASELINE
# half of the documented name map (the stress half awaits the cost_doubled<->produced-key pairing decision).
# Exact, case-insensitive match -- NOT fuzzy substring -- except BuyAndHold_<symbol>, where the symbol is a
# parameter and matches by prefix. The producer's more-specific random variants (RandomSameTurnoverSameTiming,
# RandomSameSegments) are intentionally NOT mapped: they are extra baselines, not the canonical required four.
# NOTE: this is the tested FOUNDATION; it is deliberately NOT yet wired into validate_reportable_summary (that
# rewrite + the stress mapping move reportability verdicts and need the pairing sign-off).
_BASELINE_PRODUCED_ALIASES: dict[str, str] = {
    "cash": "cash",
    "randomsameactiondistribution": "random_action_distribution",
    "random_action_distribution": "random_action_distribution",
    "randomsameturnover": "same_turnover_random",
    "same_turnover_random": "same_turnover_random",
}


def canonicalize_baseline_id(produced_name: str) -> str | None:
    """Map a produced/summary baseline name to its canonical logical id in REQUIRED_BASELINES, or None if it is
    not one of the canonical required baselines (e.g. a more-specific random variant). Case-insensitive;
    BuyAndHold_<symbol> -> buy_and_hold by prefix (the symbol is a parameter, not part of the id)."""
    key = str(produced_name).strip().lower()
    if key in _BASELINE_PRODUCED_ALIASES:
        return _BASELINE_PRODUCED_ALIASES[key]
    if key.startswith("buyandhold"):
        return "buy_and_hold"
    return None
