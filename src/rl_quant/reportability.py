"""Decision-log reportability gate -- additive, LABEL-ONLY (moves NO P&L).

Computes whether a sequential evaluation may *claim* to be a real executable trade, from its config flags AND
its decision-log rows -- never from config alone. This enforces the honesty line: real executable trading
requires crossable quote-side fills, latency P&L, **applied impact**, AND real fill-price logs; anything short
of that is causal-research / backtest only.

Pure functions, no model/trainer dependency. Intended consumers: (1) the second_context / hourly decision-log
emitters, which should stamp the verdict into their run manifest (the close-only path will honestly report
``real_executable_trade_reportable=False``); (2) the future leg-engine wiring (design PR-B2). It deliberately
changes no return/cost/equity number -- it only governs what a result is *allowed to claim*. Deletion
criterion: fold into the report layer once that layer owns reportability end-to-end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Fields every reportable sequential evaluation must log per decision row (docs/decision_tensor_protocol.md).
REQUIRED_DECISION_LOG_FIELDS: tuple[str, ...] = (
    "decision_ts",
    "entry_execution_ts",
    "reward_end_ts",
    "exit_execution_ts",
    "previous_action",
    "requested_action",
    "selected_action",
    "target_weight",
    "order_legs",
    "traded_notional",
    "gross_return",
    "cost_bps",
    "net_return",
    "equity_after",
)


@dataclass(frozen=True)
class ReportabilityVerdict:
    reportable: bool  # overall gate: required fields present (+ the strict claim when it is required)
    real_executable_trade_reportable: bool  # the STRICT claim, always computed independent of require flag
    missing_reportability_reasons: tuple[str, ...]  # every gap found (base + strict), informative regardless


def _is_traded_row(row: Mapping) -> bool:
    try:
        legs = float(row.get("order_legs") or 0.0)
        notional = float(row.get("traded_notional") or 0.0)
    except (TypeError, ValueError):
        return True  # malformed turnover fields -> treat as traded so the missing-field checks still fire
    return legs > 0.0 or abs(notional) > 0.0


def evaluate_decision_log_reportability(
    rows: Sequence[Mapping],
    *,
    require_real_executable: bool,
) -> ReportabilityVerdict:
    """Judge a list of decision-log row dicts.

    ``reportable`` is the overall gate: all REQUIRED_DECISION_LOG_FIELDS present on every row, plus -- when
    ``require_real_executable`` is True -- the strict real-executable claim. ``real_executable_trade_reportable``
    is the strict claim itself, always computed so a caller can see *why* a run cannot claim real execution
    even when it did not require it. ``missing_reportability_reasons`` is the union of all gaps found."""
    base_missing: list[str] = []
    strict_missing: list[str] = []

    if not rows:
        base_missing.append("no_decision_rows")

    for i, row in enumerate(rows):
        for key in REQUIRED_DECISION_LOG_FIELDS:
            if row.get(key) is None:
                base_missing.append(f"row{i}:missing_{key}")

        # Strict real-executable claim (always evaluated; only gates `reportable` when required).
        if row.get("real_executable_fill_model") is not True:
            strict_missing.append(f"row{i}:not_crossable_quote_fill_model")
        if row.get("valuation_complete") is not True:
            strict_missing.append(f"row{i}:valuation_incomplete")
        if row.get("execution_complete") is not True:
            strict_missing.append(f"row{i}:execution_incomplete")
        if row.get("impact_applied") is not True:
            strict_missing.append(f"row{i}:impact_not_applied")
        if _is_traded_row(row) and row.get("entry_price") is None:
            strict_missing.append(f"row{i}:missing_entry_price")
        if row.get("requires_exit_price") and row.get("exit_price") is None:
            strict_missing.append(f"row{i}:missing_exit_price")

    base_ok = not base_missing
    real_ok = base_ok and not strict_missing
    reportable = base_ok and (real_ok if require_real_executable else True)
    return ReportabilityVerdict(
        reportable=reportable,
        real_executable_trade_reportable=real_ok,
        missing_reportability_reasons=tuple(base_missing + strict_missing),
    )
