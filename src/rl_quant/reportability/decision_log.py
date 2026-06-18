"""Decision-log reportability gate -- additive, LABEL-ONLY (moves NO P&L).

Computes whether a sequential evaluation may *claim* to be reportable / a real executable trade, from its
config flags AND its decision-log rows -- never from config alone. Enforces the honesty line: real executable
trading requires crossable quote-side fills, latency P&L, **applied impact**, AND real fill-price logs;
anything short of that is causal-research / backtest only.

Two tiers (aligned to docs/decision_tensor_protocol.md, the "decision_logs.jsonl" required-fields list):
- BASE / mechanical reportability: every protocol field present AND semantically valid -- finite numerics,
  non-negative costs/turnover, positive equity, the SELECTED action allowed by the (ex-ante) action mask, and
  point-in-time-causal, ordered timestamps: context_available_until <= decision_ts <= entry_execution_ts <=
  reward_end_ts <= exit_execution_ts (parsed from numeric epochs, ISO-8601 strings, or datetimes). This is
  what a CAUSAL close-based backtest can satisfy.
- STRICT / real-executable: base PLUS crossable-fill flags (real_executable_fill_model, valuation_complete,
  execution_complete, impact_applied) and real fill prices that are FINITE POSITIVE numbers (entry_price on
  traded rows, exit_price when required). The close-only path has no fills, so entry/exit_price are
  STRICT-tier -- a causal backtest is base-reportable but not real-executable.

Pure functions, no model/trainer dependency; changes no return/cost/equity number. Consumers: the
second_context / hourly decision-log emitters (stamp the verdict into the run manifest) and the future
leg-engine wiring (design PR-B2).

KNOWN LIMITATIONS (deferred; mostly need NEW decision-log fields, not validator logic): the policy-intent
chain (raw_policy/requested/constraint-adjusted action) is not required because the protocol doc requires
only selected_action; requires_exit_price is honored as logged rather than derived from position/terminal
state; row-level impact decomposition (impact_cost_bps) is not required (the leg engine already carries
ExecutionLeg.impact_bps); net_return==gross-cost / equity recurrence is not asserted (the cost convention is
path-specific); and STATISTICAL credibility (overfitting / multiple-testing: PBO / deflated-Sharpe /
reality-check) is a SEPARATE axis this mechanical gate intentionally does not cover.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

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
# Point-in-time-causal chain: must be non-decreasing left to right.
_TIMESTAMP_CHAIN = ("context_available_until", "decision_ts", "entry_execution_ts", "reward_end_ts", "exit_execution_ts")


@dataclass(frozen=True)
class ReportabilityIssue:
    row_index: int | None
    field: str | None
    category: str  # "missing" | "malformed" | "negative" | "nonpositive_equity" | "ordering" | "mask" | "strict" | "ledger"
    message: str


@dataclass(frozen=True)
class ReportabilityVerdict:
    reportable: bool  # overall gate: base valid (+ strict when require_real_executable)
    base_reportable: bool  # mechanical/base reportability, independent of the require flag
    real_executable_trade_reportable: bool  # strict claim, always computed
    issues: tuple[ReportabilityIssue, ...]  # structured per-row/field issues (no string parsing needed)
    missing_reportability_reasons: tuple[str, ...]  # distinct "category:field" tokens, for manifests/labels


def _is_finite_number(value: object) -> bool:
    # Reject None, bool, strings, NaN, inf. (bool is an int subclass, so exclude it explicitly.)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _is_positive_finite_number(value: object) -> bool:
    return _is_finite_number(value) and float(value) > 0.0


def _parse_timestamp(value: object) -> float | None:
    """Best-effort timestamp -> comparable epoch seconds. Never raises; returns None if unrecognized.
    Accepts finite numeric epochs, tz-AWARE ISO-8601 strings (datetime.fromisoformat), and tz-aware datetime
    objects. A tz-NAIVE datetime/string is rejected (returns None -> flagged malformed by the caller): it has
    no absolute instant, so ``.timestamp()`` would silently assume the local system timezone, making the
    parsed epoch -- and the causal-ordering verdict that compares these stamps -- machine-dependent."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, datetime):
        return value.timestamp() if value.utcoffset() is not None else None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed.timestamp() if parsed.utcoffset() is not None else None
    return None


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

    Validates presence AND semantics: required fields non-None; finite numerics (no NaN/inf/bool/str);
    non-negative costs/turnover; positive equity; the selected action allowed by the ex-ante action mask;
    parseable, point-in-time-causal, non-decreasing timestamps; and (strict tier) finite-positive fill prices.
    All checks are defensive -- they never raise on a malformed field/row; they record a structured issue.
    ``base_reportable`` is mechanical validity; ``reportable`` adds the strict tier when required;
    ``real_executable_trade_reportable`` is always the strict claim."""
    base_issues: list[ReportabilityIssue] = []
    strict_issues: list[ReportabilityIssue] = []

    if not rows:
        base_issues.append(ReportabilityIssue(None, "decision_rows", "missing", "no decision rows"))

    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            base_issues.append(ReportabilityIssue(i, None, "malformed", f"row {i}: not a mapping ({type(row).__name__})"))
            continue

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

        # Point-in-time causality: parse the timestamp chain (numeric / ISO / datetime) and require it
        # non-decreasing. A present-but-unparseable timestamp is malformed; ordering runs once all parse.
        parsed: dict[str, float] = {}
        for field in _TIMESTAMP_CHAIN:
            value = row.get(field)
            if value is None:
                continue  # already a 'missing' issue
            stamp = _parse_timestamp(value)
            if stamp is None:
                base_issues.append(ReportabilityIssue(i, field, "malformed", f"row {i}: {field} is not a parseable timestamp ({value!r})"))
            else:
                parsed[field] = stamp
        if len(parsed) == len(_TIMESTAMP_CHAIN):
            seq = [parsed[field] for field in _TIMESTAMP_CHAIN]
            if any(seq[k] > seq[k + 1] for k in range(len(seq) - 1)):
                base_issues.append(ReportabilityIssue(i, "execution_timestamps", "ordering", f"row {i}: timestamps not non-decreasing {seq}"))

        # The selected action must be allowed by the (ex-ante) action mask. Two logged shapes: a name->bool
        # MAP (selected_action is the name) or a positional ARRAY/list of bools (selected_action is the index).
        action_mask = row.get("action_mask")
        selected = row.get("selected_action")
        if isinstance(action_mask, Mapping) and selected is not None:
            try:
                allowed = action_mask.get(selected)
            except TypeError:
                allowed = None  # unhashable selected_action
            if allowed is not True:
                base_issues.append(ReportabilityIssue(i, "action_mask", "mask", f"row {i}: selected_action {selected!r} not allowed by action_mask"))
        elif isinstance(action_mask, Sequence) and not isinstance(action_mask, (str, bytes)) and selected is not None:
            # Array/list mask: selected_action must be an in-bounds index whose entry is True. A non-integer
            # selected_action cannot be resolved against an unnamed array, so it FAILS CLOSED -- a required
            # field we cannot validate the selection against must not silently pass (the prior code, which
            # only handled the Mapping shape, skipped array masks entirely).
            index = selected if isinstance(selected, int) and not isinstance(selected, bool) else None
            if index is None or not (0 <= index < len(action_mask)) or action_mask[index] is not True:
                base_issues.append(ReportabilityIssue(i, "action_mask", "mask", f"row {i}: selected_action {selected!r} not an allowed index of the array action_mask"))

        # STRICT real-executable tier.
        for field in _REAL_EXECUTABLE_FLAGS:
            if row.get(field) is not True:
                strict_issues.append(ReportabilityIssue(i, field, "strict", f"row {i}: {field} is not True"))
        if _is_traded_row(row) and not _is_positive_finite_number(row.get("entry_price")):
            strict_issues.append(ReportabilityIssue(i, "entry_price", "strict", f"row {i}: traded row needs a finite positive entry_price ({row.get('entry_price')!r})"))
        if row.get("requires_exit_price") and not _is_positive_finite_number(row.get("exit_price")):
            strict_issues.append(ReportabilityIssue(i, "exit_price", "strict", f"row {i}: needs a finite positive exit_price ({row.get('exit_price')!r})"))

    # Report-only LEDGER check (NON-gating): the equity curve must compound by net_return row-to-row,
    # equity_after[i] == equity_after[i-1] * (1 + net_return[i]) (the eval keeps a single continuous,
    # full-precision equity that is never reset mid-log -- only the position resets on a path break). It is
    # surfaced as a "ledger" issue for diagnostics but does NOT gate base/real reportability: the cost
    # convention is path-specific and not every emitter is confirmed to maintain this invariant. Row 0 is
    # skipped (equity_before is not a logged field). Tolerance is generous (full-precision -> ~exact match).
    ledger_issues: list[ReportabilityIssue] = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if not (isinstance(prev, Mapping) and isinstance(cur, Mapping)):
            continue
        eq_prev, eq_cur, net = prev.get("equity_after"), cur.get("equity_after"), cur.get("net_return")
        if not (_is_finite_number(eq_prev) and _is_finite_number(eq_cur) and _is_finite_number(net)):
            continue  # missing/malformed numerics are already base issues
        expected = float(eq_prev) * (1.0 + float(net))
        if abs(expected - float(eq_cur)) > 1e-6 * max(1.0, abs(float(eq_cur))):
            ledger_issues.append(ReportabilityIssue(
                i, "equity_after", "ledger",
                f"row {i}: equity_after {float(eq_cur)!r} != equity_after[{i - 1}]*(1+net_return) = {expected!r}",
            ))

    base_reportable = not base_issues
    real_ok = base_reportable and not strict_issues
    reportable = base_reportable and (real_ok if require_real_executable else True)
    gating_issues = (*base_issues, *strict_issues)
    all_issues = (*gating_issues, *ledger_issues)
    # reasons = GATING categories only; the non-gating ledger diagnostic must not read as a reportability fail.
    reasons = tuple(sorted({f"{issue.category}:{issue.field}" for issue in gating_issues}))
    return ReportabilityVerdict(
        reportable=reportable,
        base_reportable=base_reportable,
        real_executable_trade_reportable=real_ok,
        issues=all_issues,
        missing_reportability_reasons=reasons,
    )
