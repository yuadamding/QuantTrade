"""Decision-log reportability gate -- additive, LABEL-ONLY (moves NO P&L).

Computes whether a sequential evaluation may *claim* to be reportable / a real executable trade, from its
config flags AND its decision-log rows -- never from config alone. Enforces the honesty line: real executable
trading requires crossable quote-side fills, latency P&L, **applied impact**, AND real fill-price logs;
anything short of that is causal-research / backtest only.

Two tiers (aligned to docs/decision_tensor_protocol.md, the "decision_logs.jsonl" required-fields list):
- BASE / mechanical reportability: every protocol field present AND semantically valid (finite numerics,
  non-negative costs/turnover, positive equity, ordered timestamps when numeric). This is what a CAUSAL
  close-based backtest can satisfy.
- STRICT / real-executable: base PLUS crossable-fill flags (real_executable_fill_model, valuation_complete,
  execution_complete, impact_applied) and real fill prices (entry_price on traded rows, exit_price when
  required). The close-only path legitimately has no fill prices, so entry/exit_price are STRICT-tier, not
  base -- a causal backtest is base-reportable but not real-executable.

Pure functions, no model/trainer dependency; changes no return/cost/equity number. Intended consumers: the
second_context / hourly decision-log emitters (stamp the verdict into the run manifest) and the future
leg-engine wiring (design PR-B2). Deletion criterion: fold into the report layer once that layer owns
reportability end-to-end. KNOWN LIMITATIONS (deferred): net_return==gross-cost consistency is not asserted
(the cost convention is path-specific); requires_exit_price is honored as logged rather than derived from
position/terminal state; statistical reportability (overfitting/multiple-testing) is out of scope here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# BASE tier: protocol-required fields a causal backtest must log (docs/decision_tensor_protocol.md). The
# real fill prices (entry_price/exit_price) are intentionally STRICT-tier, not base.
REQUIRED_DECISION_LOG_FIELDS: tuple[str, ...] = (
    "decision_ts",
    "context_available_until",
    "entry_execution_ts",
    "reward_end_ts",
    "exit_execution_ts",
    "previous_action",
    "selected_action",
    "target_weight",
    "order_legs",
    "traded_notional",
    "q_values",
    "q_edge_vs_cash",
    "q_edge_vs_current",
    "action_mask",
    "mask_reasons",
    "data_quality_score",
    "readiness_score",
    "gross_return",
    "cost_bps",
    "net_return",
    "equity_after",
)

# STRICT real-executable flags every row must carry True (the leg-engine reportability axes).
_REAL_EXECUTABLE_FLAGS: tuple[str, ...] = (
    "real_executable_fill_model",
    "valuation_complete",
    "execution_complete",
    "impact_applied",
)
_FINITE_NUMERIC_FIELDS = ("target_weight", "order_legs", "traded_notional", "gross_return", "cost_bps", "net_return", "equity_after")
_NONNEGATIVE_FIELDS = ("order_legs", "traded_notional", "cost_bps")
_TIMESTAMP_ORDER_FIELDS = ("decision_ts", "entry_execution_ts", "reward_end_ts", "exit_execution_ts")


@dataclass(frozen=True)
class ReportabilityIssue:
    row_index: int | None
    field: str | None
    category: str  # "missing" | "malformed" | "negative" | "nonpositive_equity" | "ordering" | "strict"
    message: str


@dataclass(frozen=True)
class ReportabilityVerdict:
    reportable: bool  # overall gate: base valid (+ strict when require_real_executable)
    real_executable_trade_reportable: bool  # strict claim, always computed
    issues: tuple[ReportabilityIssue, ...]  # structured per-row/field issues (no string parsing needed)
    missing_reportability_reasons: tuple[str, ...]  # distinct "category:field" tokens, for manifests/labels


def _is_finite_number(value: object) -> bool:
    # Reject None, bool, strings, NaN, inf. (bool is an int subclass, so exclude it explicitly.)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _is_traded_row(row: Mapping) -> bool:
    # A row trades iff it has positive turnover. A malformed turnover value is handled by the numeric checks
    # (which fail base reportability); here it is treated conservatively as TRADED so the strict entry-price
    # requirement still applies rather than being silently skipped.
    legs = row.get("order_legs")
    notional = row.get("traded_notional")
    if (legs is not None and not _is_finite_number(legs)) or (notional is not None and not _is_finite_number(notional)):
        return True
    return (_is_finite_number(legs) and float(legs) > 0.0) or (_is_finite_number(notional) and abs(float(notional)) > 0.0)


def evaluate_decision_log_reportability(
    rows: Sequence[Mapping],
    *,
    require_real_executable: bool,
) -> ReportabilityVerdict:
    """Judge decision-log rows for (base) mechanical reportability and (strict) real-executable reportability.

    Validates presence AND semantics: required fields non-None, finite numerics (no NaN/inf/bool/str),
    non-negative costs/turnover, positive equity, and -- when timestamps are numeric -- non-decreasing
    decision/entry/reward-end/exit ordering. All checks are defensive (they never raise on a malformed field;
    they record a structured issue). ``reportable`` is base validity (+ strict when required);
    ``real_executable_trade_reportable`` is always the strict claim."""
    base_issues: list[ReportabilityIssue] = []
    strict_issues: list[ReportabilityIssue] = []

    if not rows:
        base_issues.append(ReportabilityIssue(None, None, "missing", "no decision rows"))

    for i, row in enumerate(rows):
        for field in REQUIRED_DECISION_LOG_FIELDS:
            if row.get(field) is None:
                base_issues.append(ReportabilityIssue(i, field, "missing", f"row {i}: missing {field}"))
        for field in _FINITE_NUMERIC_FIELDS:
            value = row.get(field)
            if value is not None and not _is_finite_number(value):
                base_issues.append(ReportabilityIssue(i, field, "malformed", f"row {i}: {field} is not a finite number ({value!r})"))
        for field in _NONNEGATIVE_FIELDS:
            value = row.get(field)
            if _is_finite_number(value) and float(value) < 0.0:
                base_issues.append(ReportabilityIssue(i, field, "negative", f"row {i}: {field} is negative ({value!r})"))
        equity = row.get("equity_after")
        if _is_finite_number(equity) and float(equity) <= 0.0:
            base_issues.append(ReportabilityIssue(i, "equity_after", "nonpositive_equity", f"row {i}: equity_after must be > 0 ({equity!r})"))
        # Timestamp ordering only when ALL four are numeric (ISO-string logs skip this defensively, never fail).
        stamps = [row.get(field) for field in _TIMESTAMP_ORDER_FIELDS]
        if all(_is_finite_number(s) for s in stamps) and not (stamps[0] <= stamps[1] <= stamps[2] <= stamps[3]):
            base_issues.append(ReportabilityIssue(i, None, "ordering", f"row {i}: timestamps not non-decreasing {stamps}"))

        # STRICT real-executable tier.
        for field in _REAL_EXECUTABLE_FLAGS:
            if row.get(field) is not True:
                strict_issues.append(ReportabilityIssue(i, field, "strict", f"row {i}: {field} is not True"))
        if _is_traded_row(row) and row.get("entry_price") is None:
            strict_issues.append(ReportabilityIssue(i, "entry_price", "strict", f"row {i}: traded row missing entry_price"))
        if row.get("requires_exit_price") and row.get("exit_price") is None:
            strict_issues.append(ReportabilityIssue(i, "exit_price", "strict", f"row {i}: missing required exit_price"))

    base_ok = not base_issues
    real_ok = base_ok and not strict_issues
    reportable = base_ok and (real_ok if require_real_executable else True)
    all_issues = (*base_issues, *strict_issues)
    reasons = tuple(sorted({f"{issue.category}:{issue.field}" for issue in all_issues}))
    return ReportabilityVerdict(
        reportable=reportable,
        real_executable_trade_reportable=real_ok,
        issues=all_issues,
        missing_reportability_reasons=reasons,
    )
