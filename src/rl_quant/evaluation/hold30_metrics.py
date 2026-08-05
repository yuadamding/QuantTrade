"""Receipt-stable telemetry for the Hold-30 chronological runtime.

The aggregator consumes a detached :class:`Hold30CanonicalTrace`; it never
reruns the policy and never infers holding duration from a gate. Retention and
the primary sale age use score-origin return-neutral entry units; economic
notional sale age is retained only as a secondary diagnostic. Turnover uses
the runtime's disjoint cause accounting over the supplied scored-row mask.

All ratios carry their numerator and denominator.  An undefined ratio is a
JSON ``null`` with an explicit reason rather than NaN or an invented zero.
The returned SHA-256 covers the canonical JSON payload excluding the digest
field itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch

from rl_quant.envs.hold30 import (
    AGE_BIN_COUNT,
    MAX_EXACT_AGE,
    TURNOVER_CAUSES,
    CohortTradeAccounting,
    TurnoverCause,
)
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace, Hold30Transition


HOLD30_TELEMETRY_SCHEMA = "hold30-telemetry-v2"
HOLD30_SURVIVAL_HORIZONS = (5, 10, 20, 30, 60)
HOLD30_OVERLAP_LAGS = (5, 10, 20, 30)
_FORCED_CAUSES = (
    TurnoverCause.MEMBERSHIP_FORCED,
    TurnoverCause.AVAILABILITY_FORCED,
    TurnoverCause.RISK_FORCED,
)
_CAUSE_ORDER = (
    TurnoverCause.MEMBERSHIP_FORCED,
    TurnoverCause.AVAILABILITY_FORCED,
    TurnoverCause.RISK_FORCED,
    TurnoverCause.DISCRETIONARY,
)
_TOLERANCE = 1e-6
_ABSENT_TOLERANCE = 1e-12


class Hold30TelemetryError(ValueError):
    """The persisted trace cannot support scientifically valid telemetry."""


def _float(value: float | torch.Tensor) -> float:
    result = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise Hold30TelemetryError("telemetry contains a non-finite scalar")
    return 0.0 if result == 0.0 else result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hold30_metrics_digest(payload: dict[str, Any]) -> str:
    """Return the canonical digest, ignoring an existing ``sha256`` field."""

    material = dict(payload)
    material.pop("sha256", None)
    return hashlib.sha256(_canonical_json(material)).hexdigest()


def verify_hold30_metrics_digest(payload: dict[str, Any]) -> bool:
    """Verify a serialized telemetry receipt without mutating it."""

    digest = payload.get("sha256")
    return isinstance(digest, str) and digest == hold30_metrics_digest(payload)


def _ratio(
    numerator: float | torch.Tensor,
    denominator: float | torch.Tensor,
    *,
    null_reason: str,
) -> dict[str, float | str | None]:
    top = _float(numerator)
    bottom = _float(denominator)
    if bottom < 0.0:
        raise Hold30TelemetryError("a metric denominator cannot be negative")
    if bottom == 0.0:
        return {
            "value": None,
            "numerator": top,
            "denominator": bottom,
            "null_reason": null_reason,
        }
    return {
        "value": _float(top / bottom),
        "numerator": top,
        "denominator": bottom,
        "null_reason": None,
    }


def _mean(
    total: float | torch.Tensor,
    observations: int,
    *,
    null_reason: str,
) -> dict[str, float | int | str | None]:
    if observations < 0:
        raise Hold30TelemetryError("observation count cannot be negative")
    value = _ratio(total, observations, null_reason=null_reason)
    return {
        "value": value["value"],
        "total": value["numerator"],
        "observations": observations,
        "null_reason": value["null_reason"],
    }


def _require_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    nonnegative: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise Hold30TelemetryError(f"{name} must have shape {shape}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise Hold30TelemetryError(f"{name} must be a finite floating-point tensor")
    if nonnegative and bool((value < -_TOLERANCE).any()):
        raise Hold30TelemetryError(f"{name} cannot contain negative mass")


def _validate_accounting(
    transition_index: int,
    cause: TurnoverCause,
    accounting: CohortTradeAccounting,
    *,
    batch: int,
    assets: int,
) -> None:
    prefix = f"transitions[{transition_index}].accounting[{cause.value}]"
    if not isinstance(accounting, CohortTradeAccounting) or accounting.cause is not cause:
        raise Hold30TelemetryError(f"{prefix} has the wrong turnover cause")
    _require_tensor(f"{prefix}.turnover", accounting.turnover, (batch,), nonnegative=True)
    for name in ("net_buys", "net_sells", "entry_units_added"):
        _require_tensor(
            f"{prefix}.{name}", getattr(accounting, name), (batch, assets), nonnegative=True
        )
    for name in ("sold_value_by_age", "sold_units_by_age"):
        _require_tensor(
            f"{prefix}.{name}",
            getattr(accounting, name),
            (batch, assets, AGE_BIN_COUNT),
            nonnegative=True,
        )
    for name in ("early_exit_notional", "early_exit_units"):
        _require_tensor(f"{prefix}.{name}", getattr(accounting, name), (batch,), nonnegative=True)


def _validate_trace(trace: Hold30CanonicalTrace) -> tuple[int, int, int, int]:
    if not isinstance(trace, Hold30CanonicalTrace):
        raise Hold30TelemetryError("trace must be a Hold30CanonicalTrace")
    transitions = trace.transitions
    count = len(transitions)
    if count == 0:
        raise Hold30TelemetryError("telemetry requires at least one transition")
    if len(trace.boundary_states) != count + 1:
        raise Hold30TelemetryError("boundary_states must contain one more row than transitions")
    if len(trace.decision_states) != count or len(trace.pending_intents) != count:
        raise Hold30TelemetryError(
            "decision_states and pending_intents must contain one row per transition"
        )

    first = trace.boundary_states[0]
    batch = first.ledger.batch_size
    assets = first.ledger.num_assets
    cash = first.ledger.cash_index
    expected_position = first.position_index
    axis_ids: set[str] = set()
    for index, transition in enumerate(transitions):
        before = trace.boundary_states[index]
        after = trace.boundary_states[index + 1]
        pending = trace.pending_intents[index]
        if before.position_index != expected_position or after.position_index != expected_position + 1:
            raise Hold30TelemetryError("boundary states are not consecutive")
        if transition.decision_index != expected_position or transition.fill_index != expected_position + 1:
            raise Hold30TelemetryError("transition decision/fill indices are not consecutive")
        if pending.decision_index != expected_position or pending.fill_index != expected_position + 1:
            raise Hold30TelemetryError("pending-intent indices do not match the transition")
        axis_ids.add(pending.axis_id)
        if before.pending_intent is not None or after.pending_intent is not None:
            raise Hold30TelemetryError("canonical boundary states cannot retain a pending intent")
        if (
            before.ledger.batch_size != batch
            or after.ledger.batch_size != batch
            or before.ledger.num_assets != assets
            or after.ledger.num_assets != assets
            or before.ledger.cash_index != cash
            or after.ledger.cash_index != cash
        ):
            raise Hold30TelemetryError("ledger axes changed within the trace")
        before.ledger.assert_reconciles(before.ledger.weights)
        after.ledger.assert_reconciles(after.ledger.weights)

        matrix_names = (
            "decision_weights",
            "execution_pretrade_weights",
            "membership_repaired_weights",
            "availability_repaired_weights",
            "risk_repaired_weights",
            "requested_delta",
            "constructed_delta",
            "filled_delta",
            "pre_cost_weights",
            "post_cost_weights",
            "cost_financing",
        )
        for name in matrix_names:
            _require_tensor(f"transitions[{index}].{name}", getattr(transition, name), (batch, assets))
        for name in ("retention_units_before_membership", "retention_units_after_forced"):
            _require_tensor(
                f"transitions[{index}].{name}",
                getattr(transition, name),
                (batch, assets, AGE_BIN_COUNT),
                nonnegative=True,
            )
        for name in (
            "holding_return",
            "cost",
            "net_return",
            "benchmark_net_return",
            "utility",
            "projection_distance",
            "equity_before",
            "equity_after",
        ):
            _require_tensor(f"transitions[{index}].{name}", getattr(transition, name), (batch,))

        if set(transition.turnover_by_cause) != set(TURNOVER_CAUSES):
            raise Hold30TelemetryError("turnover_by_cause is not exhaustive")
        if set(transition.accounting_by_cause) != set(TURNOVER_CAUSES):
            raise Hold30TelemetryError("accounting_by_cause is not exhaustive")
        for cause in TURNOVER_CAUSES:
            _validate_accounting(
                index,
                cause,
                transition.accounting_by_cause[cause],
                batch=batch,
                assets=assets,
            )
            _require_tensor(
                f"transitions[{index}].turnover[{cause.value}]",
                transition.turnover_by_cause[cause],
                (batch,),
                nonnegative=True,
            )
            if not bool(
                torch.allclose(
                    transition.turnover_by_cause[cause],
                    transition.accounting_by_cause[cause].turnover,
                    atol=_TOLERANCE,
                    rtol=_TOLERANCE,
                )
            ):
                raise Hold30TelemetryError("turnover and cohort accounting disagree")
        discretionary = transition.accounting_by_cause[TurnoverCause.DISCRETIONARY]
        for name in (
            "turnover",
            "net_buys",
            "net_sells",
            "sold_value_by_age",
            "sold_units_by_age",
            "entry_units_added",
        ):
            if not bool(
                torch.allclose(
                    getattr(transition.discretionary_accounting, name),
                    getattr(discretionary, name),
                    atol=_TOLERANCE,
                    rtol=_TOLERANCE,
                )
            ):
                raise Hold30TelemetryError("discretionary accounting is internally inconsistent")
        if not bool(
            torch.allclose(
                transition.decision_weights,
                before.ledger.weights,
                atol=_TOLERANCE,
                rtol=_TOLERANCE,
            )
        ):
            raise Hold30TelemetryError("decision weights do not match their boundary ledger")
        if not bool(
            torch.allclose(
                transition.post_cost_weights,
                after.ledger.weights,
                atol=_TOLERANCE,
                rtol=_TOLERANCE,
            )
        ):
            raise Hold30TelemetryError("post-cost weights do not match the next boundary ledger")
        expected_position += 1
    if len(axis_ids) != 1:
        raise Hold30TelemetryError("pending intents do not share one axis_id")
    return count, batch, assets, cash


def _current_age(trace: Hold30CanonicalTrace, cash: int) -> dict[str, Any]:
    value = trace.terminal_state.ledger.economic_value.detach()
    risky = value.clone()
    risky[:, cash] = 0.0
    ages = torch.arange(AGE_BIN_COUNT, dtype=value.dtype, device=value.device)
    numerator = (risky * ages).sum()
    denominator = risky.sum()
    result = _ratio(
        numerator,
        denominator,
        null_reason="terminal_book_has_no_risky_notional",
    )
    return {
        "notional_weighted_sessions_capped_60": result,
        "age_60_plus_is_capped": True,
    }


@dataclass(frozen=True, slots=True)
class _RetentionEvents:
    entry_units: torch.Tensor
    at_risk_before_forced: torch.Tensor
    forced_sold_units: torch.Tensor
    at_risk_before_discretionary: torch.Tensor
    discretionary_sold_units: torch.Tensor
    terminal_censored_units: torch.Tensor


def _origin_mask(
    trace: Hold30CanonicalTrace,
    score_origin_mask: torch.Tensor | None,
) -> torch.Tensor:
    count = len(trace.transitions)
    if score_origin_mask is None:
        return torch.ones(count, dtype=torch.bool)
    if (
        not isinstance(score_origin_mask, torch.Tensor)
        or score_origin_mask.dtype != torch.bool
        or tuple(score_origin_mask.shape) != (count,)
    ):
        raise Hold30TelemetryError(
            "score_origin_mask must be a boolean row for every transition"
        )
    return score_origin_mask.detach().to(device="cpu").clone()


def _score_origin_events(
    trace: Hold30CanonicalTrace,
    *,
    cash: int,
    score_origin_mask: torch.Tensor,
) -> _RetentionEvents:
    """Replay the tracked-unit ledger while retaining only score-origin buys.

    Before age 60, an age bin identifies one entry date. At age 60+, multiple
    origins merge, so filtering ``sold_units_by_age`` after the fact is not
    valid. This proportional sub-ledger follows the exact cause-by-cause
    removal fractions and remains exact after the terminal-bin merge.
    """

    reference = trace.transitions[0].decision_weights
    tagged = trace.boundary_states[0].ledger.retention_units.detach().new_zeros(
        trace.boundary_states[0].ledger.retention_units.shape
    )
    entry = reference.new_zeros(())
    at_risk_all = reference.new_zeros(AGE_BIN_COUNT)
    forced_sold = reference.new_zeros(AGE_BIN_COUNT)
    at_risk_disc = reference.new_zeros(AGE_BIN_COUNT)
    discretionary_sold = reference.new_zeros(AGE_BIN_COUNT)

    for index, transition in enumerate(trace.transitions):
        aged = torch.zeros_like(tagged)
        aged[..., 1:MAX_EXACT_AGE] = tagged[..., : MAX_EXACT_AGE - 1]
        aged[..., MAX_EXACT_AGE] = (
            tagged[..., MAX_EXACT_AGE - 1] + tagged[..., MAX_EXACT_AGE]
        )
        total_stage = transition.retention_units_before_membership.detach().clone()
        if bool((aged - total_stage > _TOLERANCE).any()):
            raise Hold30TelemetryError("score-origin units exceed the canonical unit ledger")
        aged[:, cash] = 0.0
        total_stage[:, cash] = 0.0
        at_risk_all += aged.sum(dim=(0, 1))
        tagged_stage = aged

        for cause in _FORCED_CAUSES:
            sold_total = (
                transition.accounting_by_cause[cause].sold_units_by_age.detach().clone()
            )
            sold_total[:, cash] = 0.0
            fraction = torch.where(
                total_stage > 0.0,
                sold_total / total_stage.clamp_min(torch.finfo(total_stage.dtype).eps),
                torch.zeros_like(total_stage),
            )
            if bool((fraction < -_TOLERANCE).any()) or bool(
                (fraction > 1.0 + _TOLERANCE).any()
            ):
                raise Hold30TelemetryError("forced unit-removal fraction lies outside [0, 1]")
            sold_tagged = tagged_stage * fraction.clamp(0.0, 1.0)
            forced_sold += sold_tagged.sum(dim=(0, 1))
            tagged_stage = (tagged_stage - sold_tagged).clamp_min(0.0)
            total_stage = (total_stage - sold_total).clamp_min(0.0)

        at_risk_disc += tagged_stage.sum(dim=(0, 1))
        discretionary = transition.accounting_by_cause[
            TurnoverCause.DISCRETIONARY
        ]
        sold_total = discretionary.sold_units_by_age.detach().clone()
        sold_total[:, cash] = 0.0
        fraction = torch.where(
            total_stage > 0.0,
            sold_total / total_stage.clamp_min(torch.finfo(total_stage.dtype).eps),
            torch.zeros_like(total_stage),
        )
        if bool((fraction < -_TOLERANCE).any()) or bool(
            (fraction > 1.0 + _TOLERANCE).any()
        ):
            raise Hold30TelemetryError(
                "discretionary unit-removal fraction lies outside [0, 1]"
            )
        sold_tagged = tagged_stage * fraction.clamp(0.0, 1.0)
        discretionary_sold += sold_tagged.sum(dim=(0, 1))
        tagged_stage = (tagged_stage - sold_tagged).clamp_min(0.0)

        if bool(score_origin_mask[index]):
            added = discretionary.entry_units_added.detach().clone()
            added[:, cash] = 0.0
            tagged_stage = tagged_stage.clone()
            tagged_stage[..., 0] = tagged_stage[..., 0] + added
            entry += added.sum()
        tagged = tagged_stage
        canonical = trace.boundary_states[index + 1].ledger.retention_units.detach()
        if bool((tagged - canonical > _TOLERANCE).any()):
            raise Hold30TelemetryError(
                "score-origin replay exceeds the canonical boundary unit ledger"
            )

    tagged = tagged.clone()
    tagged[:, cash] = 0.0
    return _RetentionEvents(
        entry_units=entry,
        at_risk_before_forced=at_risk_all,
        forced_sold_units=forced_sold,
        at_risk_before_discretionary=at_risk_disc,
        discretionary_sold_units=discretionary_sold,
        terminal_censored_units=tagged.sum(dim=(0, 1)),
    )


def _weighted_age_payload(
    sold: torch.Tensor,
    *,
    weighting: str,
    denominator_name: str,
) -> dict[str, Any]:
    total = sold.sum()
    total_float = _float(total)
    null_reason = f"no_discretionary_{denominator_name}"
    if total_float == 0.0:
        median: dict[str, float | str | None] = {
            "value": None,
            denominator_name: 0.0,
            "null_reason": null_reason,
        }
    else:
        threshold = total * 0.5
        median_age = int(torch.searchsorted(sold.cumsum(0), threshold, right=False).item())
        median = {
            "value": float(median_age),
            denominator_name: total_float,
            "null_reason": None,
        }
    quantiles: dict[str, float | None] = {}
    for probability in (0.10, 0.25, 0.50, 0.75, 0.90):
        key = f"{probability:.2f}"
        if total_float == 0.0:
            quantiles[key] = None
        else:
            threshold = total * probability
            quantiles[key] = float(
                int(torch.searchsorted(sold.cumsum(0), threshold, right=False).item())
            )
    return {
        "weighting": weighting,
        "median_sessions_capped_60": median,
        "quantiles_sessions_capped_60": quantiles,
        "young_sell_fraction": {
            "lt_10": _ratio(sold[:10].sum(), total, null_reason=null_reason),
            "lt_20": _ratio(sold[:20].sum(), total, null_reason=null_reason),
            "lt_30": _ratio(sold[:30].sum(), total, null_reason=null_reason),
        },
        "age_60_plus_is_capped": True,
    }


def _sale_age(
    trace: Hold30CanonicalTrace,
    cash: int,
    events: _RetentionEvents,
) -> dict[str, Any]:
    reference = trace.transitions[0].decision_weights
    economic_sold = reference.new_zeros(AGE_BIN_COUNT)
    for transition in trace.transitions:
        by_age = transition.discretionary_accounting.sold_value_by_age.detach().clone()
        by_age[:, cash] = 0.0
        economic_sold += by_age.sum(dim=(0, 1))
    primary = _weighted_age_payload(
        events.discretionary_sold_units,
        weighting="score_origin_return_neutral_entry_notional_units",
        denominator_name="sold_entry_units",
    )
    primary["economic_value_weighted_secondary"] = _weighted_age_payload(
        economic_sold,
        weighting="all_tracked_discretionary_sold_economic_notional",
        denominator_name="sold_notional",
    )
    return primary


def _product_limit(events: _RetentionEvents) -> dict[str, Any]:
    discretionary = 1.0
    all_cause = 1.0
    horizons: dict[str, Any] = {}
    for age in range(MAX_EXACT_AGE):
        risk_disc = _float(events.at_risk_before_discretionary[age])
        sold_disc = _float(events.discretionary_sold_units[age])
        risk_all = _float(events.at_risk_before_forced[age])
        sold_forced = _float(events.forced_sold_units[age])
        if sold_disc > risk_disc + _TOLERANCE or sold_forced + sold_disc > risk_all + _TOLERANCE:
            raise Hold30TelemetryError("product-limit events exceed their at-risk units")
        if risk_disc > 0.0:
            discretionary *= max(0.0, 1.0 - sold_disc / risk_disc)
        if risk_all > 0.0:
            all_cause *= max(0.0, 1.0 - (sold_forced + sold_disc) / risk_all)
        horizon = age + 1
        if horizon in HOLD30_SURVIVAL_HORIZONS:
            horizons[str(horizon)] = {
                "discretionary_survival": discretionary,
                "all_cause_survival": all_cause,
            }
    return {
        "weighting": "score_origin_return_neutral_entry_notional_units",
        "risk_event_table": {
            "age": list(range(AGE_BIN_COUNT)),
            "at_risk_before_forced": [
                _float(value) for value in events.at_risk_before_forced
            ],
            "forced_exit_events": [_float(value) for value in events.forced_sold_units],
            "at_risk_before_discretionary": [
                _float(value) for value in events.at_risk_before_discretionary
            ],
            "discretionary_exit_events": [
                _float(value) for value in events.discretionary_sold_units
            ],
            "administrative_censor_units": [
                _float(value) for value in events.terminal_censored_units
            ],
        },
        "entry_units": _float(events.entry_units),
        "horizons": horizons,
        "pooling_rule": "sum fold risk/events at each age before applying product",
    }


def _turnover_lifecycle(
    transition: Hold30Transition,
    cash: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = transition.risk_repaired_weights
    post = transition.pre_cost_weights
    delta = post - pre
    risky = torch.ones_like(delta, dtype=torch.bool)
    risky[:, cash] = False
    entering = risky & (pre <= _ABSENT_TOLERANCE) & (delta > 0.0)
    exiting = risky & (pre > _ABSENT_TOLERANCE) & (post <= _ABSENT_TOLERANCE) & (delta < 0.0)
    resizing = risky & (delta != 0.0) & ~entering & ~exiting
    half = 0.5 * delta.abs()
    entry = (half * entering).sum(-1)
    exit_ = (half * exiting).sum(-1)
    resize = (half * resizing).sum(-1)

    cash_half = half[:, cash]
    cash_delta = delta[:, cash]
    risky_buys = delta.clamp_min(0.0) * risky
    risky_sells = (-delta).clamp_min(0.0) * risky
    entry_buys = (risky_buys * entering).sum(-1)
    resize_buys = (risky_buys * resizing).sum(-1)
    exit_sells = (risky_sells * exiting).sum(-1)
    resize_sells = (risky_sells * resizing).sum(-1)

    deploying = cash_delta < 0.0
    deploy_total = entry_buys + resize_buys
    entry_share = torch.where(deploy_total > 0.0, entry_buys / deploy_total, torch.zeros_like(entry))
    entry = entry + torch.where(deploying, cash_half * entry_share, torch.zeros_like(entry))
    resize = resize + torch.where(
        deploying,
        cash_half * (1.0 - entry_share),
        torch.zeros_like(resize),
    )

    withdrawing = cash_delta > 0.0
    withdraw_total = exit_sells + resize_sells
    exit_share = torch.where(
        withdraw_total > 0.0,
        exit_sells / withdraw_total,
        torch.zeros_like(exit_),
    )
    exit_ = exit_ + torch.where(withdrawing, cash_half * exit_share, torch.zeros_like(exit_))
    resize = resize + torch.where(
        withdrawing,
        cash_half * (1.0 - exit_share),
        torch.zeros_like(resize),
    )
    classified = entry + exit_ + resize
    expected = transition.turnover_by_cause[TurnoverCause.DISCRETIONARY]
    if not bool(torch.allclose(classified, expected, atol=_TOLERANCE, rtol=_TOLERANCE)):
        raise Hold30TelemetryError("entry/exit/resize turnover does not reconcile")
    return entry, exit_, resize


def _turnover(
    trace: Hold30CanonicalTrace,
    *,
    batch: int,
    cash: int,
    decision_mask: torch.Tensor,
) -> dict[str, Any]:
    selected = tuple(
        transition
        for index, transition in enumerate(trace.transitions)
        if bool(decision_mask[index])
    )
    decisions = len(selected)
    if decisions == 0:
        raise Hold30TelemetryError("turnover telemetry requires at least one selected decision")
    cells = decisions * batch
    totals = {
        cause: sum(
            (transition.turnover_by_cause[cause].detach().sum() for transition in selected),
            start=selected[0].decision_weights.new_zeros(()),
        )
        for cause in TURNOVER_CAUSES
    }
    cause_payload = {
        cause.value: {
            "total_one_way_mean_per_path": _float(total / batch),
            "mean_one_way_per_decision": _float(total / cells),
        }
        for cause, total in totals.items()
    }
    forced = sum((totals[cause] for cause in _FORCED_CAUSES), start=totals[_FORCED_CAUSES[0]].new_zeros(()))
    entry_total = totals[TurnoverCause.DISCRETIONARY].new_zeros(())
    exit_total = entry_total.clone()
    resize_total = entry_total.clone()
    for transition in selected:
        entry, exit_, resize = _turnover_lifecycle(transition, cash)
        entry_total += entry.sum()
        exit_total += exit_.sum()
        resize_total += resize.sum()
    discretionary = totals[TurnoverCause.DISCRETIONARY]
    implied = _ratio(
        1.0,
        discretionary / cells,
        null_reason="zero_mean_discretionary_turnover",
    )
    return {
        "causes": cause_payload,
        "forced_all_causes": {
            "total_one_way_mean_per_path": _float(forced / batch),
            "mean_one_way_per_decision": _float(forced / cells),
        },
        "discretionary_lifecycle_partition": {
            "definition": "one-way turnover, including the allocated CASH financing leg",
            "entry": {
                "total_one_way_mean_per_path": _float(entry_total / batch),
                "mean_one_way_per_decision": _float(entry_total / cells),
            },
            "exit": {
                "total_one_way_mean_per_path": _float(exit_total / batch),
                "mean_one_way_per_decision": _float(exit_total / cells),
            },
            "resize": {
                "total_one_way_mean_per_path": _float(resize_total / batch),
                "mean_one_way_per_decision": _float(resize_total / cells),
            },
            "maximum_reconciliation_error": _float(
                (entry_total + exit_total + resize_total - discretionary).abs()
            ),
        },
        "turnover_implied_horizon_sessions_approx": {
            **implied,
            "label": "approximate inverse of mean discretionary one-way turnover",
        },
    }


def _origin_at_horizon(
    trace: Hold30CanonicalTrace,
    origin: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    target = origin + horizon
    transition = trace.transitions[target]
    if horizon < MAX_EXACT_AGE:
        before = transition.retention_units_before_membership[..., horizon]
        after_forced = transition.retention_units_after_forced[..., horizon]
        after_all = trace.boundary_states[target + 1].ledger.retention_units[..., horizon]
        losses = {
            "forced": sum(
                (
                    trace.transitions[origin + age]
                    .accounting_by_cause[cause]
                    .sold_units_by_age[..., age]
                    for age in range(1, horizon + 1)
                    for cause in _FORCED_CAUSES
                ),
                start=before.new_zeros(before.shape),
            ),
            "discretionary": sum(
                (
                    trace.transitions[origin + age]
                    .accounting_by_cause[TurnoverCause.DISCRETIONARY]
                    .sold_units_by_age[..., age]
                    for age in range(1, horizon + 1)
                ),
                start=before.new_zeros(before.shape),
            ),
        }
        return before, after_forced, after_all, losses

    # At age 60, the ledger merges the newly matured origin with older units.
    # The trade ledger removes every unit in that asset/bin with the same
    # economic removal fraction, so sequential proportional attribution is
    # exact under the accounting contract and avoids counting older cohorts.
    before = trace.boundary_states[target].ledger.retention_units[..., MAX_EXACT_AGE - 1]
    origin_remaining = before.clone()
    stage_total = transition.retention_units_before_membership[..., MAX_EXACT_AGE].clone()
    forced_loss = before.new_zeros(before.shape)
    discretionary_loss = before.new_zeros(before.shape)
    for cause in _CAUSE_ORDER:
        sold_total = transition.accounting_by_cause[cause].sold_units_by_age[..., MAX_EXACT_AGE]
        fraction = torch.where(stage_total > 0.0, sold_total / stage_total, torch.zeros_like(stage_total))
        if bool((fraction < -_TOLERANCE).any()) or bool((fraction > 1.0 + _TOLERANCE).any()):
            raise Hold30TelemetryError("age-60 cohort removal fraction is outside [0, 1]")
        origin_sold = origin_remaining * fraction.clamp(0.0, 1.0)
        if cause in _FORCED_CAUSES:
            forced_loss += origin_sold
        else:
            discretionary_loss += origin_sold
        origin_remaining = (origin_remaining - origin_sold).clamp_min(0.0)
        stage_total = (stage_total - sold_total).clamp_min(0.0)
    after_forced = before - forced_loss
    after_all = origin_remaining

    for age in range(1, MAX_EXACT_AGE):
        aged_transition = trace.transitions[origin + age]
        forced_loss += sum(
            (
                aged_transition.accounting_by_cause[cause].sold_units_by_age[..., age]
                for cause in _FORCED_CAUSES
            ),
            start=before.new_zeros(before.shape),
        )
        discretionary_loss += aged_transition.accounting_by_cause[
            TurnoverCause.DISCRETIONARY
        ].sold_units_by_age[..., age]
    return before, after_forced, after_all, {
        "forced": forced_loss,
        "discretionary": discretionary_loss,
    }


def _survival(trace: Hold30CanonicalTrace, cash: int) -> dict[str, Any]:
    if bool((trace.boundary_states[0].ledger.retention_units.abs() > _TOLERANCE).any()):
        return {
            str(horizon): {
                "eligible_entry_units": 0.0,
                **{
                    name: _ratio(
                        0.0,
                        0.0,
                        null_reason="initial_endowment_contains_tracked_units",
                    )
                    for name in (
                        "before_membership",
                        "after_forced_repairs",
                        "after_discretionary_trade",
                        "cumulative_forced_exit_fraction",
                        "cumulative_discretionary_exit_fraction",
                    )
                },
            }
            for horizon in HOLD30_SURVIVAL_HORIZONS
        }

    count = len(trace.transitions)
    result: dict[str, Any] = {}
    for horizon in HOLD30_SURVIVAL_HORIZONS:
        eligible_origins = range(max(0, count - horizon))
        denominator = trace.transitions[0].decision_weights.new_zeros(())
        before_total = denominator.clone()
        after_forced_total = denominator.clone()
        after_all_total = denominator.clone()
        forced_loss_total = denominator.clone()
        discretionary_loss_total = denominator.clone()
        for origin in eligible_origins:
            entries = (
                trace.transitions[origin]
                .accounting_by_cause[TurnoverCause.DISCRETIONARY]
                .entry_units_added.detach()
                .clone()
            )
            entries[:, cash] = 0.0
            denominator += entries.sum()
            before, after_forced, after_all, losses = _origin_at_horizon(
                trace, origin, horizon
            )
            before = before.detach().clone()
            after_forced = after_forced.detach().clone()
            after_all = after_all.detach().clone()
            before[:, cash] = 0.0
            after_forced[:, cash] = 0.0
            after_all[:, cash] = 0.0
            before_total += before.sum()
            after_forced_total += after_forced.sum()
            after_all_total += after_all.sum()
            forced = losses["forced"].detach().clone()
            discretionary = losses["discretionary"].detach().clone()
            forced[:, cash] = 0.0
            discretionary[:, cash] = 0.0
            forced_loss_total += forced.sum()
            discretionary_loss_total += discretionary.sum()

        null_reason = f"no_entry_units_with_{horizon}_session_followup"
        if _float(denominator) > 0.0:
            reconciliation = (
                denominator - after_all_total - forced_loss_total - discretionary_loss_total
            ).abs()
            if _float(reconciliation) > _TOLERANCE * max(1.0, _float(denominator)):
                raise Hold30TelemetryError(
                    f"retention accounting does not reconcile at horizon {horizon}"
                )
        result[str(horizon)] = {
            "eligible_entry_units": _float(denominator),
            "before_membership": _ratio(before_total, denominator, null_reason=null_reason),
            "after_forced_repairs": _ratio(
                after_forced_total, denominator, null_reason=null_reason
            ),
            "after_discretionary_trade": _ratio(
                after_all_total, denominator, null_reason=null_reason
            ),
            "cumulative_forced_exit_fraction": _ratio(
                forced_loss_total, denominator, null_reason=null_reason
            ),
            "cumulative_discretionary_exit_fraction": _ratio(
                discretionary_loss_total, denominator, null_reason=null_reason
            ),
        }
    return result


def _overlap(
    trace: Hold30CanonicalTrace,
    cash: int,
    batch: int,
    decision_mask: torch.Tensor,
) -> dict[str, Any]:
    selected = torch.where(decision_mask)[0].tolist()
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise Hold30TelemetryError("portfolio-overlap decision mask must be one contiguous block")
    books = [
        state.ledger.weights.detach()
        for state in trace.boundary_states[selected[0] : selected[-1] + 2]
    ]
    result: dict[str, Any] = {}
    for lag in HOLD30_OVERLAP_LAGS:
        pair_count = max(0, len(books) - lag)
        if pair_count == 0:
            result[str(lag)] = _mean(
                0.0,
                0,
                null_reason=f"trace_has_no_{lag}_session_pairs",
            )
            continue
        total = books[0].new_zeros(())
        for left, right in zip(books[:-lag], books[lag:]):
            overlap = torch.minimum(left, right)
            overlap = overlap.clone()
            overlap[:, cash] = 0.0
            total += overlap.sum()
        result[str(lag)] = _mean(
            total,
            pair_count * batch,
            null_reason=f"trace_has_no_{lag}_session_pairs",
        )
    return result


def _pnl_by_age(
    trace: Hold30CanonicalTrace,
    *,
    batch: int,
    cash: int,
    decision_mask: torch.Tensor,
) -> dict[str, Any]:
    buckets = {
        "0_9": range(0, 10),
        "10_19": range(10, 20),
        "20_29": range(20, 30),
        "30_59": range(30, 60),
        "60_plus": range(60, 61),
    }
    reference = trace.transitions[0].decision_weights
    gross_pnl = {name: reference.new_zeros(()) for name in buckets}
    return_contribution = {name: reference.new_zeros(()) for name in buckets}
    cash_pnl = reference.new_zeros(())
    transaction_cost_pnl = reference.new_zeros(())
    for index, transition in enumerate(trace.transitions):
        if not bool(decision_mask[index]):
            continue
        ledger = trace.boundary_states[index].ledger
        old_weights = ledger.weights
        gross = 1.0 + transition.holding_return
        pretrade = transition.execution_pretrade_weights
        positive = old_weights > torch.finfo(old_weights.dtype).eps
        if bool((pretrade.masked_select(~positive).abs() > _TOLERANCE).any()):
            raise Hold30TelemetryError("a zero-weight asset acquired value before the trade stage")
        asset_return = torch.where(
            positive,
            pretrade * gross.unsqueeze(-1) / old_weights.clamp_min(torch.finfo(old_weights.dtype).eps)
            - 1.0,
            torch.zeros_like(old_weights),
        )
        contribution = ledger.economic_value * asset_return.unsqueeze(-1)
        contribution = contribution.clone()
        contribution[:, cash] = 0.0
        equity_contribution = contribution * transition.equity_before[:, None, None]
        for name, ages in buckets.items():
            age_indices = tuple(ages)
            return_contribution[name] += contribution[..., age_indices].sum()
            gross_pnl[name] += equity_contribution[..., age_indices].sum()
        cash_return_contribution = old_weights[:, cash] * asset_return[:, cash]
        cash_pnl += (cash_return_contribution * transition.equity_before).sum()
        transaction_cost_pnl += (transition.cost * transition.equity_before).sum()
        reconstructed = contribution.sum(dim=(-1, -2)) + cash_return_contribution
        if not bool(
            torch.allclose(
                reconstructed,
                transition.holding_return,
                atol=_TOLERANCE,
                rtol=_TOLERANCE,
            )
        ):
            raise Hold30TelemetryError("age-attributed gross return does not reconcile")
    decisions = int(decision_mask.sum().item())
    return {
        "age_convention": "economic notional age at the start of the holding-return row",
        "cost_attribution": "transaction costs are reported separately, not assigned to an age bucket",
        "buckets": {
            name: {
                "gross_pnl_mean_per_path": _float(gross_pnl[name] / batch),
                "gross_return_contribution_mean_per_decision": _float(
                    return_contribution[name] / (batch * decisions)
                ),
            }
            for name in buckets
        },
        "cash_gross_pnl_mean_per_path": _float(cash_pnl / batch),
        "transaction_cost_pnl_mean_per_path": _float(transaction_cost_pnl / batch),
    }


def aggregate_hold30_metrics(
    trace: Hold30CanonicalTrace,
    *,
    score_origin_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Validate and aggregate one exact canonical Hold-30 trace.

    The output contains only JSON primitives and has a deterministic digest.
    Batch paths are pooled for rate statistics; turnover and P&L totals are
    explicitly labeled as means per path.
    """

    count, batch, assets, cash = _validate_trace(trace)
    origin_mask = _origin_mask(trace, score_origin_mask)
    events = _score_origin_events(
        trace,
        cash=cash,
        score_origin_mask=origin_mask,
    )
    payload: dict[str, Any] = {
        "schema_version": HOLD30_TELEMETRY_SCHEMA,
        "trace": {
            "transition_count": count,
            "boundary_count": count + 1,
            "batch_size": batch,
            "asset_count_including_cash": assets,
            "cash_index": cash,
            "axis_id": trace.pending_intents[0].axis_id,
            "score_origin_mask_sha256": hashlib.sha256(
                _canonical_json([bool(value) for value in origin_mask.tolist()])
            ).hexdigest(),
        },
        "current_holding_age": _current_age(trace, cash),
        "sale_age": _sale_age(trace, cash, events),
        "product_limit_survival": _product_limit(events),
        "retention_survival": _survival(trace, cash),
        "turnover": _turnover(
            trace,
            batch=batch,
            cash=cash,
            decision_mask=origin_mask,
        ),
        "risky_notional_portfolio_overlap": _overlap(
            trace,
            cash,
            batch,
            origin_mask,
        ),
        "pnl_contribution_by_position_age": _pnl_by_age(
            trace,
            batch=batch,
            cash=cash,
            decision_mask=origin_mask,
        ),
    }
    payload["sha256"] = hold30_metrics_digest(payload)
    return payload


def pool_hold30_product_limit(
    fold_reports: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    """Pool six fold risk/event tables before applying the product limit."""

    if not isinstance(fold_reports, Mapping) or set(fold_reports) != set(range(6)):
        raise Hold30TelemetryError(
            "product-limit pooling requires exact fold keys 0 through 5"
        )
    table_names = (
        "at_risk_before_forced",
        "forced_exit_events",
        "at_risk_before_discretionary",
        "discretionary_exit_events",
        "administrative_censor_units",
    )
    totals = {name: torch.zeros(AGE_BIN_COUNT, dtype=torch.float64) for name in table_names}
    entry_units = 0.0
    receipt_hashes: list[str] = []
    for fold_index in range(6):
        report = fold_reports[fold_index]
        if not isinstance(report, dict) or not verify_hold30_metrics_digest(report):
            raise Hold30TelemetryError(f"fold {fold_index} telemetry digest is invalid")
        product = report.get("product_limit_survival")
        if not isinstance(product, dict) or product.get("weighting") != (
            "score_origin_return_neutral_entry_notional_units"
        ):
            raise Hold30TelemetryError(f"fold {fold_index} lacks the primary survival table")
        table = product.get("risk_event_table")
        if not isinstance(table, dict) or table.get("age") != list(range(AGE_BIN_COUNT)):
            raise Hold30TelemetryError(f"fold {fold_index} risk/event ages are invalid")
        for name in table_names:
            values = torch.tensor(table.get(name), dtype=torch.float64)
            if values.shape != (AGE_BIN_COUNT,) or not bool(torch.isfinite(values).all()) or bool(
                (values < 0.0).any()
            ):
                raise Hold30TelemetryError(f"fold {fold_index} {name} is invalid")
            totals[name] += values
        entry = product.get("entry_units")
        if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(entry):
            raise Hold30TelemetryError(f"fold {fold_index} entry_units is invalid")
        entry_units += float(entry)
        receipt_hashes.append(report["sha256"])
    if len(set(receipt_hashes)) != 6:
        raise Hold30TelemetryError("fold telemetry receipts must be distinct")

    events = _RetentionEvents(
        entry_units=torch.tensor(entry_units, dtype=torch.float64),
        at_risk_before_forced=totals["at_risk_before_forced"],
        forced_sold_units=totals["forced_exit_events"],
        at_risk_before_discretionary=totals["at_risk_before_discretionary"],
        discretionary_sold_units=totals["discretionary_exit_events"],
        terminal_censored_units=totals["administrative_censor_units"],
    )
    payload: dict[str, Any] = {
        "schema_version": "hold30-pooled-product-limit-v1",
        "fold_count": 6,
        "fold_telemetry_sha256": receipt_hashes,
        "pooled": _product_limit(events),
    }
    payload["sha256"] = hold30_metrics_digest(payload)
    return payload


__all__ = [
    "HOLD30_OVERLAP_LAGS",
    "HOLD30_SURVIVAL_HORIZONS",
    "HOLD30_TELEMETRY_SCHEMA",
    "Hold30TelemetryError",
    "aggregate_hold30_metrics",
    "hold30_metrics_digest",
    "pool_hold30_product_limit",
    "verify_hold30_metrics_digest",
]
