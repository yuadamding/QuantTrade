"""Trace-owned 2026 telemetry for the one-seed TOP2000 retrospective.

The chronological runtime is the only owner of portfolio mutation, costs, and
cohort ages.  This adapter therefore consumes one *completed* detached runtime
trace and never reconstructs a policy path from checkpoint outputs.  The warm
252-state prefix remains in the trace, but only the retrospective data
adapter's exact ``score_transition_slice`` enters evaluation arrays.

The current TOP2000 universe is future-selected.  Every receipt emitted here
is consequently development-only, nonreportable, and nonpromotable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import torch

from rl_quant.envs.hold30 import (
    AGE_BIN_COUNT,
    TURNOVER_CAUSES,
    CohortTradeAccounting,
    TurnoverCause,
)
from rl_quant.evaluation.top2000_m03r_v7_2026 import Top2000M03RV72026Telemetry
from rl_quant.evaluation.top2000_m03r_v7_2026_execution_view import (
    Top2000M03RV72026EconomicExecutionView,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_COST_RATE,
    Top2000M03RV72026RetrospectiveData,
)
from rl_quant.models.daily_policy import hold30_release_hazard
from rl_quant.models.hold30_exit_action_v6 import (
    M03R_V6_CONTINUOUS_ACTION_INDEX,
    M03R_V6_EXIT_ACTION_INDEX,
    M03R_V6_HOLD_ACTION_INDEX,
)
from rl_quant.models.hold30_hazard import HOLD30_HAZARD_MAX, HOLD30_HAZARD_MIN
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
    runtime_setting_id,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    resolve_m03r_top2000_dev_setting,
)
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace, Hold30Transition

TOP2000_M03R_V7_2026_TRACE_TELEMETRY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-trace-telemetry-v1"
)
TOP2000_M03R_V7_2026_TRACE_ACTIONS = ("HOLD", "CONTINUOUS", "EXIT")
_ACTION_INDEX = {
    "HOLD": M03R_V6_HOLD_ACTION_INDEX,
    "CONTINUOUS": M03R_V6_CONTINUOUS_ACTION_INDEX,
    "EXIT": M03R_V6_EXIT_ACTION_INDEX,
}
_FORCED_EXIT_CAUSES = (
    TurnoverCause.MEMBERSHIP_FORCED,
    TurnoverCause.AVAILABILITY_FORCED,
    TurnoverCause.RISK_FORCED,
    TurnoverCause.TERMINAL,
)
_STAGE_CAUSES = (
    TurnoverCause.MEMBERSHIP_FORCED,
    TurnoverCause.AVAILABILITY_FORCED,
    TurnoverCause.RISK_FORCED,
    TurnoverCause.DISCRETIONARY,
)
_STAGE_WEIGHTS = (
    "execution_pretrade_weights",
    "membership_repaired_weights",
    "availability_repaired_weights",
    "risk_repaired_weights",
    "pre_cost_weights",
)
_TOLERANCE = 2.0e-6


class Top2000M03RV72026TraceTelemetryError(ValueError):
    """A trace cannot support the frozen retrospective telemetry semantics."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026TraceTelemetryError(
            "trace telemetry receipt is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _cpu_array(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            f"{name} must be finite floating with shape {shape}"
        )
    if nonnegative and bool((value < -_TOLERANCE).any()):
        raise Top2000M03RV72026TraceTelemetryError(f"{name} contains negative mass")
    result = value.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
    if nonnegative:
        result[result < 0.0] = 0.0
    return result


def _torch_close(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    if tuple(left.shape) != tuple(right.shape) or not bool(
        torch.allclose(left, right, atol=_TOLERANCE, rtol=_TOLERANCE)
    ):
        maximum = (
            math.inf
            if tuple(left.shape) != tuple(right.shape)
            else float((left - right).detach().abs().max().to(device="cpu"))
        )
        raise Top2000M03RV72026TraceTelemetryError(
            f"{name} does not reconcile (maximum difference {maximum:g})"
        )


def _validate_accounting(
    transition: Hold30Transition,
    cause: TurnoverCause,
    *,
    batch: int,
    assets: int,
) -> CohortTradeAccounting:
    try:
        accounting = transition.accounting_by_cause[cause]
        turnover = transition.turnover_by_cause[cause]
    except KeyError as exc:
        raise Top2000M03RV72026TraceTelemetryError(
            "runtime turnover causes are incomplete"
        ) from exc
    if not isinstance(accounting, CohortTradeAccounting) or accounting.cause is not cause:
        raise Top2000M03RV72026TraceTelemetryError(
            f"accounting cause {cause.value!r} is mislabeled"
        )
    _cpu_array(f"{cause.value}.turnover", turnover, (batch,), nonnegative=True)
    _cpu_array(
        f"{cause.value}.accounting.turnover",
        accounting.turnover,
        (batch,),
        nonnegative=True,
    )
    _torch_close(f"{cause.value} turnover", turnover, accounting.turnover)
    for name in ("net_buys", "net_sells"):
        _cpu_array(
            f"{cause.value}.{name}",
            getattr(accounting, name),
            (batch, assets),
            nonnegative=True,
        )
    for name in ("sold_value_by_age", "sold_units_by_age"):
        _cpu_array(
            f"{cause.value}.{name}",
            getattr(accounting, name),
            (batch, assets, AGE_BIN_COUNT),
            nonnegative=True,
        )
    _torch_close(
        f"{cause.value} sold value",
        accounting.sold_value_by_age.sum(dim=-1),
        accounting.net_sells,
    )
    if bool(((accounting.net_buys > _TOLERANCE) & (accounting.net_sells > _TOLERANCE)).any()):
        raise Top2000M03RV72026TraceTelemetryError(
            f"{cause.value} has overlapping same-name buy and sell legs"
        )
    # CASH is implicit in the cohort ledger and absent from net_buys/net_sells.
    # The complete simplex trade therefore has one-way turnover equal to the
    # larger risky buy or risky sell leg, with the difference financed by CASH.
    expected_turnover = torch.maximum(
        accounting.net_buys.sum(dim=-1),
        accounting.net_sells.sum(dim=-1),
    )
    _torch_close(f"{cause.value} accounting turnover", accounting.turnover, expected_turnover)
    return accounting


def _reverse_trade_age_ledger(
    after: torch.Tensor,
    accounting: CohortTradeAccounting,
    *,
    expected_before_weights: torch.Tensor,
    cash_index: int,
    label: str,
) -> torch.Tensor:
    entry = torch.zeros_like(after)
    entry[..., 0] = accounting.net_buys
    before = after - entry + accounting.sold_value_by_age
    if bool((before < -_TOLERANCE).any()):
        raise Top2000M03RV72026TraceTelemetryError(
            f"{label} cause partition overlaps more notional than the age ledger holds"
        )
    before = before.clamp_min(0.0)
    if bool((accounting.sold_value_by_age - before > _TOLERANCE).any()):
        raise Top2000M03RV72026TraceTelemetryError(
            f"{label} sold-notional cause overlaps its pre-trade age ledger"
        )
    risky_weights = before.sum(dim=-1)
    risky_weights = risky_weights.clone()
    risky_weights[:, cash_index] = 0.0
    expected = expected_before_weights.clone()
    expected[:, cash_index] = 0.0
    _torch_close(f"{label} reconstructed pre-trade age ledger", risky_weights, expected)
    return before


def _post_forced_pre_discretionary_age_notional(
    trace: Hold30CanonicalTrace,
    transition_index: int,
) -> torch.Tensor:
    """Recover the exact economic age ledger at the requested stage.

    The runtime persists the post-discretionary cohort ledger plus exact
    discretionary buys and age-attributed sales.  Reversing that one trade is
    algebraically exact and avoids substituting return-neutral retention units
    for economic notional.
    """

    transition = trace.transitions[transition_index]
    after = trace.boundary_states[transition_index + 1].ledger.economic_value.detach()
    cash_index = trace.boundary_states[0].ledger.cash_index
    stage = after
    for cause, expected_name in zip(
        reversed(_STAGE_CAUSES),
        reversed(_STAGE_WEIGHTS[:-1]),
        strict=True,
    ):
        accounting = transition.accounting_by_cause[cause]
        stage = _reverse_trade_age_ledger(
            stage,
            accounting,
            expected_before_weights=getattr(transition, expected_name),
            cash_index=cash_index,
            label=cause.value,
        )
        if cause is TurnoverCause.DISCRETIONARY:
            post_forced = stage.clone()
    return post_forced


def _validate_scored_transition(
    trace: Hold30CanonicalTrace,
    transition_index: int,
    *,
    batch: int,
    assets: int,
) -> torch.Tensor:
    transition = trace.transitions[transition_index]
    if set(transition.turnover_by_cause) != set(TURNOVER_CAUSES) or set(
        transition.accounting_by_cause
    ) != set(TURNOVER_CAUSES):
        raise Top2000M03RV72026TraceTelemetryError(
            "runtime cause inventory must be exhaustive and exact"
        )
    accountings = {
        cause: _validate_accounting(
            transition,
            cause,
            batch=batch,
            assets=assets,
        )
        for cause in TURNOVER_CAUSES
    }
    cash_index = trace.boundary_states[0].ledger.cash_index
    for index, cause in enumerate(_STAGE_CAUSES):
        before = getattr(transition, _STAGE_WEIGHTS[index])
        after = getattr(transition, _STAGE_WEIGHTS[index + 1])
        expected_delta = accountings[cause].net_buys - accountings[cause].net_sells
        expected_delta = expected_delta.clone()
        expected_delta[:, cash_index] = -expected_delta.sum(dim=-1)
        _torch_close(f"{cause.value} stage delta", after - before, expected_delta)
    for cause in (TurnoverCause.STARTUP, TurnoverCause.TERMINAL):
        accounting = accountings[cause]
        if any(
            bool((getattr(accounting, name).abs() > _TOLERANCE).any())
            for name in (
                "turnover",
                "net_buys",
                "net_sells",
                "sold_value_by_age",
                "sold_units_by_age",
                "entry_units_added",
                "early_exit_notional",
                "early_exit_units",
            )
        ):
            raise Top2000M03RV72026TraceTelemetryError(
                f"{cause.value} cannot carry scored-path turnover or exits"
            )
    discretionary = accountings[TurnoverCause.DISCRETIONARY]
    for name in (
        "turnover",
        "net_buys",
        "net_sells",
        "sold_value_by_age",
        "sold_units_by_age",
        "entry_units_added",
        "early_exit_notional",
        "early_exit_units",
    ):
        _torch_close(
            f"discretionary_accounting.{name}",
            getattr(transition.discretionary_accounting, name),
            getattr(discretionary, name),
        )
    requested_to_executed = 0.5 * (
        transition.filled_delta - transition.requested_delta
    ).abs().sum(dim=-1)
    construction_to_fill = 0.5 * (
        transition.filled_delta - transition.constructed_delta
    ).abs().sum(dim=-1)
    _torch_close(
        "runtime construction-to-fill projection distance",
        transition.projection_distance,
        construction_to_fill,
    )
    total_turnover = sum(
        transition.turnover_by_cause.values(),
        start=torch.zeros_like(transition.cost),
    )
    _torch_close(
        "20-bp runtime cost",
        transition.cost,
        TOP2000_M03R_V7_2026_COST_RATE * total_turnover,
    )
    _torch_close(
        "runtime net return",
        transition.net_return,
        transition.holding_return - transition.cost,
    )
    if bool((requested_to_executed < -_TOLERANCE).any()):
        raise AssertionError("absolute projection distance became negative")
    return _post_forced_pre_discretionary_age_notional(trace, transition_index)


def _validate_complete_trace(
    trace: Hold30CanonicalTrace,
    retrospective: Top2000M03RV72026RetrospectiveData,
    economic_execution_view: Top2000M03RV72026EconomicExecutionView | None,
) -> tuple[int, int, int, int, slice, int, str | None]:
    if not isinstance(trace, Hold30CanonicalTrace):
        raise Top2000M03RV72026TraceTelemetryError(
            "trace must be a Hold30CanonicalTrace"
        )
    if not isinstance(retrospective, Top2000M03RV72026RetrospectiveData):
        raise Top2000M03RV72026TraceTelemetryError(
            "retrospective must use the immutable 2026 data adapter"
        )
    identity = retrospective.identity
    source = retrospective.source_evidence
    if (
        not identity.single_continuous_chronology
        or identity.state_reset_count_within_2026 != 0
        or not identity.development_only
        or not identity.future_selected_universe
        or identity.scientific_reporting_eligible
        or identity.promotion_eligible
        or not source.development_only
        or source.dataset_reportable
        or source.scientific_reporting_eligible
        or source.promotion_eligible
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "2026 trace must remain one no-reset, future-selected, nonreportable chronology"
        )
    if economic_execution_view is None:
        sequence = retrospective.sequence
        execution_start = 0
        local_score = retrospective.score_transition_slice
        execution_receipt_sha256: str | None = None
        expected_axis_id = identity.axis_id
    else:
        view = economic_execution_view
        receipt = view.receipt
        if (
            receipt.chronology_receipt_sha256 != identity.receipt_sha256
            or receipt.pre2026_cache_sha256 != identity.pre2026_cache_sha256
            or receipt.pre2026_cache_identity != identity.pre2026_cache_identity
            or receipt.global_score_transition_start
            != identity.score_transition_start
            or receipt.global_score_transition_stop_exclusive
            != identity.score_transition_stop_exclusive
            or receipt.training_fold_index not in range(6)
            or receipt.in_sample_origin_holdings_enter_2026
            or receipt.learned_policy_actions_before_execution_start != 0
        ):
            raise Top2000M03RV72026TraceTelemetryError(
                "economic execution view does not match the retrospective chronology"
            )
        sequence = view.sequence
        execution_start = receipt.economic_execution_start
        local_score = slice(
            receipt.local_score_transition_start,
            receipt.local_score_transition_stop_exclusive,
        )
        execution_receipt_sha256 = receipt.receipt_sha256
        expected_axis_id = receipt.execution_axis_id
    transitions = sequence.n_positions - 1
    batch = sequence.batch_size
    assets = sequence.num_assets
    cash_index = sequence.cash_index
    if (
        batch != 1
        or len(trace.transitions) != transitions
        or len(trace.boundary_states) != sequence.n_positions
        or len(trace.decision_states) != transitions
        or len(trace.pending_intents) != transitions
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "trace must be completed once over the entire retrospective sequence"
        )
    first = trace.boundary_states[0]
    if first.position_index != 0 or trace.terminal_state.position_index != transitions:
        raise Top2000M03RV72026TraceTelemetryError(
            "trace boundary positions do not span the complete chronology"
        )
    _torch_close(
        "initial economic ledger",
        first.ledger.economic_value,
        sequence.initial_ledger.economic_value,
    )
    _torch_close(
        "initial retention ledger",
        first.ledger.retention_units,
        sequence.initial_ledger.retention_units,
    )
    benchmark_net_rows: list[torch.Tensor] = []
    for index, (before, transition, pending, after) in enumerate(
        zip(
            trace.boundary_states[:-1],
            trace.transitions,
            trace.pending_intents,
            trace.boundary_states[1:],
            strict=True,
        )
    ):
        if (
            before.position_index != index
            or transition.decision_index != index
            or pending.decision_index != index
            or transition.fill_index != index + 1
            or pending.fill_index != index + 1
            or after.position_index != index + 1
            or pending.axis_id != expected_axis_id
            or before.pending_intent is not None
            or after.pending_intent is not None
            or before.ledger.batch_size != batch
            or before.ledger.num_assets != assets
            or before.ledger.cash_index != cash_index
        ):
            raise Top2000M03RV72026TraceTelemetryError(
                "trace geometry contains a reset, axis drift, or delayed-intent mismatch"
            )
        benchmark_net_rows.append(transition.benchmark_net_return)
    benchmark_net = torch.stack(benchmark_net_rows, dim=0)
    _torch_close(
        "trace/data benchmark net chronology",
        benchmark_net,
        sequence.benchmark_net_returns,
    )
    score = local_score
    if (
        score.step not in (None, 1)
        or not isinstance(score.start, int)
        or not isinstance(score.stop, int)
        or not 0 <= score.start < score.stop <= transitions
        or score.stop - score.start != len(retrospective.score_return_dates)
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "retrospective score_transition_slice is inconsistent"
        )
    return (
        transitions,
        batch,
        assets,
        cash_index,
        score,
        execution_start,
        execution_receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026TraceTelemetryReceipt:
    """Content identity and non-authorizing interpretation of one trace."""

    setting_id: str
    runtime_setting_id: str
    checkpoint_sha256: str
    checkpoint_fold_index: int
    chronology_receipt_sha256: str
    trace_axis_id: str
    scored_array_sha256s: tuple[tuple[str, str], ...]
    score_transition_start: int
    score_transition_stop_exclusive: int
    completed_transition_rows: int
    scored_transition_rows: int
    action_count: int
    economic_execution_receipt_sha256: str | None = None
    economic_execution_start: int = 0
    global_score_transition_start: int = 0
    global_score_transition_stop_exclusive: int = 0
    batch_size: int = 1
    score_rule: str = "only-retrospective.score_transition_slice"
    age_risk_set_stage: str = "post-forced-pre-discretionary-economic-notional"
    continuous_hazard_statistic: str = (
        "age-notional-weighted-mean-on-chosen-continuous-positive-held-risky-names"
    )
    benchmark_cause_mapping: tuple[tuple[str, str], ...] = (
        ("monthly_rebalance_one_way_turnover", "discretionary"),
        ("availability_forced_one_way_turnover", "availability_forced"),
        ("risk_forced_one_way_turnover", "risk_forced"),
    )
    single_continuous_trace: bool = True
    state_reset_count_within_2026: int = 0
    future_selected_universe: bool = True
    development_only: bool = True
    dataset_reportable: bool = False
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_TRACE_TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        _require_digest("checkpoint_sha256", self.checkpoint_sha256)
        _require_digest("chronology_receipt_sha256", self.chronology_receipt_sha256)
        if self.economic_execution_receipt_sha256 is not None:
            _require_digest(
                "economic_execution_receipt_sha256",
                self.economic_execution_receipt_sha256,
            )
        if self.setting_id not in M03R_SEED17_TOP2000_SETTING_IDS:
            raise Top2000M03RV72026TraceTelemetryError("unknown seed-17 setting")
        if self.runtime_setting_id != runtime_setting_id(self.setting_id):
            raise Top2000M03RV72026TraceTelemetryError(
                "seed-17 and runtime setting identities do not match"
            )
        if (
            isinstance(self.checkpoint_fold_index, bool)
            or self.checkpoint_fold_index not in range(6)
            or not self.trace_axis_id
            or self.score_transition_start < 0
            or self.score_transition_stop_exclusive <= self.score_transition_start
            or self.completed_transition_rows < self.score_transition_stop_exclusive
            or self.scored_transition_rows
            != self.score_transition_stop_exclusive - self.score_transition_start
            or self.scored_transition_rows <= 0
            or self.economic_execution_start < 0
            or self.global_score_transition_start < self.economic_execution_start
            or self.global_score_transition_stop_exclusive
            <= self.global_score_transition_start
            or self.score_transition_start
            != self.global_score_transition_start - self.economic_execution_start
            or self.score_transition_stop_exclusive
            != self.global_score_transition_stop_exclusive
            - self.economic_execution_start
            or self.action_count < 2
            or self.batch_size != 1
            or self.schema != TOP2000_M03R_V7_2026_TRACE_TELEMETRY_SCHEMA
            or self.score_rule != "only-retrospective.score_transition_slice"
            or self.age_risk_set_stage
            != "post-forced-pre-discretionary-economic-notional"
            or not self.single_continuous_trace
            or self.state_reset_count_within_2026 != 0
            or not self.future_selected_universe
            or not self.development_only
            or self.dataset_reportable
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026TraceTelemetryError(
                "trace telemetry receipt geometry or research-only semantics drifted"
            )
        if self.economic_execution_receipt_sha256 is None and (
            self.economic_execution_start != 0
            or self.score_transition_start != self.global_score_transition_start
            or self.score_transition_stop_exclusive
            != self.global_score_transition_stop_exclusive
        ):
            raise Top2000M03RV72026TraceTelemetryError(
                "an offset trace requires a content-bound economic execution receipt"
            )
        names = tuple(name for name, _digest in self.scored_array_sha256s)
        if not names or len(set(names)) != len(names) or names != tuple(sorted(names)):
            raise Top2000M03RV72026TraceTelemetryError(
                "scored array hashes must be nonempty, unique, and sorted"
            )
        for name, digest in self.scored_array_sha256s:
            if not name:
                raise Top2000M03RV72026TraceTelemetryError(
                    "scored array hash names cannot be empty"
                )
            _require_digest(f"scored_array_sha256s[{name}]", digest)

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026TraceEvaluationInputs:
    """One setting's evaluator-ready arrays plus their replayable receipt."""

    score_dates: tuple[str, ...]
    portfolio_gross_returns: np.ndarray
    benchmark_gross_returns: np.ndarray
    portfolio_net_returns_20bp: np.ndarray
    benchmark_net_returns_20bp: np.ndarray
    portfolio_turnover_by_cause: Mapping[str, np.ndarray]
    benchmark_turnover_by_cause: Mapping[str, np.ndarray]
    telemetry: Top2000M03RV72026Telemetry
    construction_to_fill_safety_projection_distance: np.ndarray
    receipt: Top2000M03RV72026TraceTelemetryReceipt

    def __post_init__(self) -> None:
        validate_top2000_m03r_v7_2026_trace_evaluation_inputs(self)


def _result_arrays(
    result: Top2000M03RV72026TraceEvaluationInputs,
) -> dict[str, np.ndarray]:
    telemetry = result.telemetry
    arrays = {
        "benchmark_gross_returns": result.benchmark_gross_returns,
        "benchmark_net_returns_20bp": result.benchmark_net_returns_20bp,
        "construction_to_fill_safety_projection_distance": (
            result.construction_to_fill_safety_projection_distance
        ),
        "portfolio_gross_returns": result.portfolio_gross_returns,
        "portfolio_net_returns_20bp": result.portfolio_net_returns_20bp,
        "score_dates": np.frombuffer(
            _canonical_json(list(result.score_dates)),
            dtype=np.uint8,
        ).copy(),
        "telemetry/age_notional_at_risk": np.asarray(
            telemetry.age_notional_at_risk
        ),
        "telemetry/continuous_hazard": np.asarray(telemetry.continuous_hazard),
        "telemetry/continuous_hazard_observed": np.asarray(
            telemetry.continuous_hazard_observed
        ),
        "telemetry/discretionary_exit_notional_by_age": np.asarray(
            telemetry.discretionary_exit_notional_by_age
        ),
        "telemetry/requested_to_executed_projection_distance": np.asarray(
            telemetry.requested_to_executed_projection_distance
        ),
    }
    arrays.update(
        {
            f"portfolio_turnover_by_cause/{cause}": np.asarray(value)
            for cause, value in result.portfolio_turnover_by_cause.items()
        }
    )
    arrays.update(
        {
            f"benchmark_turnover_by_cause/{cause}": np.asarray(value)
            for cause, value in result.benchmark_turnover_by_cause.items()
        }
    )
    arrays.update(
        {
            f"telemetry/forced_exit_notional_by_cause_and_age/{cause}": np.asarray(
                value
            )
            for cause, value in telemetry.forced_exit_notional_by_cause_and_age.items()
        }
    )
    arrays.update(
        {
            f"telemetry/action_counts_by_type/{action}": np.asarray(value)
            for action, value in telemetry.action_counts_by_type.items()
        }
    )
    return arrays


def validate_top2000_m03r_v7_2026_trace_evaluation_inputs(
    result: Top2000M03RV72026TraceEvaluationInputs,
) -> None:
    """Recompute every content hash and reject mutable-array drift."""

    if not isinstance(result, Top2000M03RV72026TraceEvaluationInputs):
        raise Top2000M03RV72026TraceTelemetryError(
            "result must use the typed trace-evaluation contract"
        )
    receipt = result.receipt
    if not isinstance(receipt, Top2000M03RV72026TraceTelemetryReceipt):
        raise Top2000M03RV72026TraceTelemetryError(
            "result requires a typed trace telemetry receipt"
        )
    receipt.__post_init__()
    rows = receipt.scored_transition_rows
    if (
        len(result.score_dates) != rows
        or any(
            type(value) is not str
            or len(value) != 10
            or not value.startswith("2026-")
            for value in result.score_dates
        )
        or any(
            current <= previous
            for previous, current in pairwise(result.score_dates)
        )
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "score dates must be an aligned strictly increasing 2026 chronology"
        )
    required_causes = {cause.value for cause in TURNOVER_CAUSES}
    if set(result.portfolio_turnover_by_cause) != required_causes or set(
        result.benchmark_turnover_by_cause
    ) != required_causes:
        raise Top2000M03RV72026TraceTelemetryError(
            "turnover outputs must contain every authoritative cause exactly once"
        )
    telemetry = result.telemetry
    if not isinstance(telemetry, Top2000M03RV72026Telemetry):
        raise Top2000M03RV72026TraceTelemetryError(
            "telemetry output must use the evaluator's typed raw-array contract"
        )
    expected_vector = (rows,)
    expected_panel_vector = (1, rows)
    expected_age = (1, rows, AGE_BIN_COUNT)
    expected_asset = (1, rows, receipt.action_count)
    if any(
        np.asarray(value).shape != expected_vector
        for value in (
            result.portfolio_gross_returns,
            result.benchmark_gross_returns,
            result.portfolio_net_returns_20bp,
            result.benchmark_net_returns_20bp,
            result.construction_to_fill_safety_projection_distance,
            *result.portfolio_turnover_by_cause.values(),
            *result.benchmark_turnover_by_cause.values(),
        )
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "return, projection, and turnover vectors must align with score dates"
        )
    if (
        np.asarray(telemetry.requested_to_executed_projection_distance).shape
        != expected_panel_vector
        or np.asarray(telemetry.age_notional_at_risk).shape != expected_age
        or np.asarray(telemetry.discretionary_exit_notional_by_age).shape
        != expected_age
        or set(telemetry.forced_exit_notional_by_cause_and_age)
        != {cause.value for cause in _FORCED_EXIT_CAUSES}
        or any(
            np.asarray(value).shape != expected_age
            for value in telemetry.forced_exit_notional_by_cause_and_age.values()
        )
        or set(telemetry.action_counts_by_type)
        != set(TOP2000_M03R_V7_2026_TRACE_ACTIONS)
        or any(
            np.asarray(value).shape != expected_panel_vector
            for value in telemetry.action_counts_by_type.values()
        )
        or np.asarray(telemetry.continuous_hazard).shape != expected_asset
        or np.asarray(telemetry.continuous_hazard_observed).shape != expected_asset
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "raw telemetry arrays do not match the one-setting score geometry"
        )
    arrays = _result_arrays(result)
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or not np.isfinite(array).all():
            raise Top2000M03RV72026TraceTelemetryError(
                f"output array {name!r} is non-finite or not an ndarray"
            )
    portfolio_total_turnover = sum(
        result.portfolio_turnover_by_cause.values(),
        start=np.zeros(rows, dtype=np.float64),
    )
    benchmark_total_turnover = sum(
        result.benchmark_turnover_by_cause.values(),
        start=np.zeros(rows, dtype=np.float64),
    )
    if not np.allclose(
        result.portfolio_net_returns_20bp,
        result.portfolio_gross_returns
        - TOP2000_M03R_V7_2026_COST_RATE * portfolio_total_turnover,
        atol=_TOLERANCE,
        rtol=_TOLERANCE,
    ) or not np.allclose(
        result.benchmark_net_returns_20bp,
        result.benchmark_gross_returns
        - TOP2000_M03R_V7_2026_COST_RATE * benchmark_total_turnover,
        atol=_TOLERANCE,
        rtol=_TOLERANCE,
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "20-bp net returns do not reconcile with gross returns and cause turnover"
        )
    expected = tuple(sorted((name, _array_sha256(value)) for name, value in arrays.items()))
    if expected != receipt.scored_array_sha256s:
        raise Top2000M03RV72026TraceTelemetryError(
            "scored trace arrays do not match their content-bound receipt"
        )


def adapt_top2000_m03r_v7_2026_trace(
    trace: Hold30CanonicalTrace,
    retrospective: Top2000M03RV72026RetrospectiveData,
    *,
    setting_id: str,
    checkpoint_sha256: str,
    checkpoint_fold_index: int,
    economic_execution_view: Top2000M03RV72026EconomicExecutionView | None = None,
) -> Top2000M03RV72026TraceEvaluationInputs:
    """Map one completed runtime trace into exact 2026 evaluation inputs."""

    _require_digest("checkpoint_sha256", checkpoint_sha256)
    if setting_id not in M03R_SEED17_TOP2000_SETTING_IDS:
        raise Top2000M03RV72026TraceTelemetryError("unknown seed-17 setting")
    runtime_id = runtime_setting_id(setting_id)
    setting = resolve_m03r_top2000_dev_setting(runtime_id)
    (
        transitions,
        batch,
        assets,
        cash_index,
        score,
        execution_start,
        execution_receipt_sha256,
    ) = _validate_complete_trace(
        trace,
        retrospective,
        economic_execution_view,
    )
    assert isinstance(score.start, int) and isinstance(score.stop, int)
    selected_indices = tuple(range(score.start, score.stop))
    rows = len(selected_indices)

    portfolio_gross = np.zeros(rows, dtype=np.float64)
    portfolio_net = np.zeros(rows, dtype=np.float64)
    requested_projection = np.zeros(rows, dtype=np.float64)
    safety_projection = np.zeros(rows, dtype=np.float64)
    portfolio_turnover = {
        cause.value: np.zeros(rows, dtype=np.float64) for cause in TURNOVER_CAUSES
    }
    age_at_risk = np.zeros((1, rows, AGE_BIN_COUNT), dtype=np.float64)
    discretionary_exit = np.zeros((1, rows, AGE_BIN_COUNT), dtype=np.float64)
    forced_exit = {
        cause.value: np.zeros((1, rows, AGE_BIN_COUNT), dtype=np.float64)
        for cause in _FORCED_EXIT_CAUSES
    }
    actions = {
        action: np.zeros((1, rows), dtype=np.float64)
        for action in TOP2000_M03R_V7_2026_TRACE_ACTIONS
    }
    continuous_hazard = np.zeros((1, rows, assets), dtype=np.float64)
    continuous_observed = np.zeros((1, rows, assets), dtype=np.bool_)
    ages = torch.arange(AGE_BIN_COUNT)

    for row, transition_index in enumerate(selected_indices):
        transition = trace.transitions[transition_index]
        risk_age = _validate_scored_transition(
            trace,
            transition_index,
            batch=batch,
            assets=assets,
        )
        risk_age = risk_age.clone()
        risk_age[:, cash_index] = 0.0
        risk_total = risk_age.sum(dim=-1)
        age_at_risk[0, row] = (
            risk_age.sum(dim=(0, 1)).detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        discretionary = transition.accounting_by_cause[TurnoverCause.DISCRETIONARY]
        discretionary_sold = discretionary.sold_value_by_age.clone()
        discretionary_sold[:, cash_index] = 0.0
        discretionary_exit[0, row] = (
            discretionary_sold
            .sum(dim=(0, 1))
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        if bool((discretionary_sold - risk_age > _TOLERANCE).any()):
            raise Top2000M03RV72026TraceTelemetryError(
                "discretionary exits exceed the post-forced age risk set"
            )
        for cause in _FORCED_EXIT_CAUSES:
            sold = transition.accounting_by_cause[cause].sold_value_by_age.clone()
            sold[:, cash_index] = 0.0
            forced_exit[cause.value][0, row] = (
                sold.sum(dim=(0, 1))
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .numpy()
            )

        portfolio_gross[row] = float(transition.holding_return.detach().cpu().item())
        portfolio_net[row] = float(transition.net_return.detach().cpu().item())
        requested_projection[row] = float(
            (
                0.5
                * (transition.filled_delta - transition.requested_delta)
                .abs()
                .sum(dim=-1)
            )
            .detach()
            .cpu()
            .item()
        )
        safety_projection[row] = float(transition.projection_distance.detach().cpu().item())
        for cause in TURNOVER_CAUSES:
            portfolio_turnover[cause.value][row] = float(
                transition.turnover_by_cause[cause].detach().cpu().item()
            )

        intent = transition.raw_intent
        hazard_residual = intent.hazard_residual
        if (
            hazard_residual is None
            or tuple(hazard_residual.shape) != (batch, assets)
            or not hazard_residual.is_floating_point()
            or not bool(torch.isfinite(hazard_residual).all())
            or bool(
                (
                    (hazard_residual < HOLD30_HAZARD_MIN)
                    | (hazard_residual > HOLD30_HAZARD_MAX)
                ).any()
            )
        ):
            raise Top2000M03RV72026TraceTelemetryError(
                "scored intent requires one bounded hazard residual per asset"
            )
        action = intent.exit_action_v6
        execution_sequence = (
            retrospective.sequence
            if economic_execution_view is None
            else economic_execution_view.sequence
        )
        expected_risky = execution_sequence.decision_available[transition_index].clone()
        expected_risky[:, cash_index] = False
        if setting.exit_hazard_mode == "learned-age-aware":
            if action is None:
                raise Top2000M03RV72026TraceTelemetryError(
                    "learned-hazard setting omitted its three-way exit action"
                )
            action.validate()
            if not torch.equal(action.risky_available, expected_risky):
                raise Top2000M03RV72026TraceTelemetryError(
                    "exit action risky-availability mask drifted from the decision"
                )
            decision = action.decision_st.detach()
        else:
            if action is not None:
                raise Top2000M03RV72026TraceTelemetryError(
                    "fixed-prior A08 must not emit a learned three-way exit action"
                )
            decision = torch.zeros(
                (batch, assets, 3),
                dtype=hazard_residual.dtype,
                device=hazard_residual.device,
            )
            decision[..., M03R_V6_CONTINUOUS_ACTION_INDEX] = expected_risky.to(
                dtype=hazard_residual.dtype
            )
        for action_name, action_index in _ACTION_INDEX.items():
            actions[action_name][0, row] = float(
                decision[..., action_index][expected_risky]
                .sum()
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
        if int(sum(value[0, row] for value in actions.values())) != int(
            expected_risky.sum()
        ):
            raise Top2000M03RV72026TraceTelemetryError(
                "HOLD/CONTINUOUS/EXIT do not partition risky available names"
            )

        age_axis = ages.to(
            device=hazard_residual.device,
            dtype=hazard_residual.dtype,
        )
        hazard_by_age = hold30_release_hazard(
            age_axis,
            hazard_residual.unsqueeze(-1),
        ).to(dtype=risk_age.dtype)
        numerator = (risk_age * hazard_by_age).sum(dim=-1)
        positive_held = risk_total > 0.0
        chosen_continuous = (
            decision[..., M03R_V6_CONTINUOUS_ACTION_INDEX].detach() == 1.0
        )
        observed = expected_risky & chosen_continuous & positive_held
        mean_hazard = torch.where(
            positive_held,
            numerator / risk_total.clamp_min(torch.finfo(risk_total.dtype).tiny),
            torch.zeros_like(numerator),
        )
        mean_hazard = torch.where(observed, mean_hazard, torch.zeros_like(mean_hazard))
        if bool(((mean_hazard < 0.0) | (mean_hazard > 1.0)).any()):
            raise Top2000M03RV72026TraceTelemetryError(
                "held-notional-weighted continuous hazard lies outside [0,1]"
            )
        continuous_hazard[0, row] = (
            mean_hazard[0].detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        continuous_observed[0, row] = observed[0].detach().to(device="cpu").numpy()

    benchmark = retrospective.benchmark
    benchmark_gross = (
        benchmark.gross_returns[score]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
        .copy()
    )
    benchmark_net = (
        benchmark.net_returns[score]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
        .copy()
    )
    benchmark_turnover = {
        cause.value: np.zeros(rows, dtype=np.float64) for cause in TURNOVER_CAUSES
    }
    benchmark_turnover[TurnoverCause.DISCRETIONARY.value] = (
        benchmark.monthly_rebalance_one_way_turnover[score]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
        .copy()
    )
    benchmark_turnover[TurnoverCause.AVAILABILITY_FORCED.value] = (
        benchmark.availability_forced_one_way_turnover[score]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
        .copy()
    )
    benchmark_turnover[TurnoverCause.RISK_FORCED.value] = (
        benchmark.risk_forced_one_way_turnover[score]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
        .copy()
    )
    benchmark_total = sum(
        benchmark_turnover.values(),
        start=np.zeros(rows, dtype=np.float64),
    )
    if not np.allclose(
        benchmark_total,
        benchmark.total_one_way_turnover[score].detach().cpu().numpy(),
        atol=_TOLERANCE,
        rtol=_TOLERANCE,
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "benchmark cause mapping does not reconcile to total turnover"
        )
    if not np.allclose(
        benchmark_net,
        benchmark_gross - TOP2000_M03R_V7_2026_COST_RATE * benchmark_total,
        atol=_TOLERANCE,
        rtol=_TOLERANCE,
    ):
        raise Top2000M03RV72026TraceTelemetryError(
            "benchmark 20-bp net return does not reconcile after cause mapping"
        )

    telemetry = Top2000M03RV72026Telemetry(
        requested_to_executed_projection_distance=requested_projection[None, :],
        age_notional_at_risk=age_at_risk,
        discretionary_exit_notional_by_age=discretionary_exit,
        forced_exit_notional_by_cause_and_age=forced_exit,
        action_counts_by_type=actions,
        continuous_hazard=continuous_hazard,
        continuous_hazard_observed=continuous_observed,
    )
    provisional = object.__new__(Top2000M03RV72026TraceEvaluationInputs)
    for name, value in {
        "score_dates": retrospective.score_return_dates,
        "portfolio_gross_returns": portfolio_gross,
        "benchmark_gross_returns": benchmark_gross,
        "portfolio_net_returns_20bp": portfolio_net,
        "benchmark_net_returns_20bp": benchmark_net,
        "portfolio_turnover_by_cause": portfolio_turnover,
        "benchmark_turnover_by_cause": benchmark_turnover,
        "telemetry": telemetry,
        "construction_to_fill_safety_projection_distance": safety_projection,
    }.items():
        object.__setattr__(provisional, name, value)
    array_hashes = tuple(
        sorted(
            (name, _array_sha256(value))
            for name, value in _result_arrays(provisional).items()
        )
    )
    receipt = Top2000M03RV72026TraceTelemetryReceipt(
        setting_id=setting_id,
        runtime_setting_id=runtime_id,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_fold_index=checkpoint_fold_index,
        chronology_receipt_sha256=retrospective.identity.receipt_sha256,
        trace_axis_id=retrospective.identity.axis_id,
        scored_array_sha256s=array_hashes,
        score_transition_start=score.start,
        score_transition_stop_exclusive=score.stop,
        completed_transition_rows=transitions,
        scored_transition_rows=rows,
        action_count=assets,
        economic_execution_receipt_sha256=execution_receipt_sha256,
        economic_execution_start=execution_start,
        global_score_transition_start=retrospective.identity.score_transition_start,
        global_score_transition_stop_exclusive=(
            retrospective.identity.score_transition_stop_exclusive
        ),
    )
    return Top2000M03RV72026TraceEvaluationInputs(
        score_dates=retrospective.score_return_dates,
        portfolio_gross_returns=portfolio_gross,
        benchmark_gross_returns=benchmark_gross,
        portfolio_net_returns_20bp=portfolio_net,
        benchmark_net_returns_20bp=benchmark_net,
        portfolio_turnover_by_cause=portfolio_turnover,
        benchmark_turnover_by_cause=benchmark_turnover,
        telemetry=telemetry,
        construction_to_fill_safety_projection_distance=safety_projection,
        receipt=receipt,
    )


__all__ = [
    "TOP2000_M03R_V7_2026_TRACE_ACTIONS",
    "TOP2000_M03R_V7_2026_TRACE_TELEMETRY_SCHEMA",
    "Top2000M03RV72026TraceEvaluationInputs",
    "Top2000M03RV72026TraceTelemetryError",
    "Top2000M03RV72026TraceTelemetryReceipt",
    "adapt_top2000_m03r_v7_2026_trace",
    "validate_top2000_m03r_v7_2026_trace_evaluation_inputs",
]
