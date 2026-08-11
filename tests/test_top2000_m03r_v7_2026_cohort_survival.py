"""Focused tests for complete score-origin holding trajectories."""

from __future__ import annotations

import copy
import datetime as dt
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from rl_quant.envs.hold30 import AGE_BIN_COUNT, TURNOVER_CAUSES, TurnoverCause
from rl_quant.evaluation import top2000_m03r_v7_2026_cohort_survival as cohort
from rl_quant.evaluation.top2000_m03r_v7_2026_cohort_survival import (
    TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES,
    Top2000M03RV72026CohortSurvivalError,
    Top2000M03RV72026CohortTrajectories,
    Top2000M03RV72026CohortTrajectoryReceipt,
    build_top2000_m03r_v7_2026_cohort_trajectories,
    evaluate_top2000_m03r_v7_2026_cohort_rmst60,
    validate_top2000_m03r_v7_2026_cohort_rmst60_receipt,
    validate_top2000_m03r_v7_2026_cohort_trajectories,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_execution_view import (
    Top2000M03RV72026EconomicExecutionReceipt,
    Top2000M03RV72026EconomicExecutionView,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    Top2000M03RV72026RetrospectiveData,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace


def _digest(character: str) -> str:
    return character * 64


def _origin_dates(rows: int = 42) -> tuple[str, ...]:
    start = dt.date(2026, 1, 2)
    return tuple((start + dt.timedelta(days=index)).isoformat() for index in range(rows))


def _trajectory_from_arrays(
    setting_id: str,
    *,
    entry: np.ndarray,
    events: np.ndarray,
    forced: dict[str, np.ndarray],
    terminal: np.ndarray,
) -> Top2000M03RV72026CohortTrajectories:
    dates = _origin_dates(int(entry.size))
    forced_hashes = tuple(
        (
            cause.value,
            cohort._array_sha256(
                f"forced_censor/{cause.value}",
                forced[cause.value],
            ),
        )
        for cause in (
            TurnoverCause.MEMBERSHIP_FORCED,
            TurnoverCause.AVAILABILITY_FORCED,
            TurnoverCause.RISK_FORCED,
            TurnoverCause.TERMINAL,
        )
    )
    receipt = Top2000M03RV72026CohortTrajectoryReceipt(
        setting_id=setting_id,
        checkpoint_sha256=_digest("a"),
        checkpoint_fold_index=5,
        chronology_receipt_sha256=_digest("b"),
        economic_execution_receipt_sha256=cohort._sha256(
            {"setting_id": setting_id, "artifact": "economic-execution"}
        ),
        score_dates_sha256=cohort._sha256(list(dates)),
        entry_units_sha256=cohort._array_sha256("entry_units", entry),
        discretionary_event_units_by_age_sha256=cohort._array_sha256(
            "discretionary_event_units_by_age", events
        ),
        forced_censor_units_by_cause_and_age_sha256=forced_hashes,
        terminal_censor_units_by_age_sha256=cohort._array_sha256(
            "terminal_censor_units_by_age", terminal
        ),
        origin_rows=len(dates),
    )
    return Top2000M03RV72026CohortTrajectories(
        origin_dates=dates,
        entry_units=entry,
        discretionary_event_units_by_age=events,
        forced_censor_units_by_cause_and_age=forced,
        terminal_censor_units_by_age=terminal,
        receipt=receipt,
    )


def _scenario(
    setting_id: str,
    name: str,
) -> Top2000M03RV72026CohortTrajectories:
    rows = len(_origin_dates())
    entry = np.ones(rows, dtype=np.float64)
    events = np.zeros((rows, AGE_BIN_COUNT), dtype=np.float64)
    forced = {
        cause.value: np.zeros_like(events)
        for cause in (
            TurnoverCause.MEMBERSHIP_FORCED,
            TurnoverCause.AVAILABILITY_FORCED,
            TurnoverCause.RISK_FORCED,
            TurnoverCause.TERMINAL,
        )
    }
    terminal = np.zeros_like(events)
    if name == "partial":
        events[:, 10] = 0.25
        events[:, 20] = 0.75
    elif name == "forced":
        forced[TurnoverCause.AVAILABILITY_FORCED.value][:, 5] = 0.5
        events[:, 10] = 0.5
    elif name == "terminal60":
        terminal[:, 60] = 1.0
    elif name == "terminal30":
        terminal[:, 30] = 1.0
    elif name == "event1":
        events[:, 1] = 1.0
    elif name == "missing":
        entry.fill(0.0)
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unknown scenario {name}")
    return _trajectory_from_arrays(
        setting_id,
        entry=entry,
        events=events,
        forced=forced,
        terminal=terminal,
    )


def test_partial_sales_forced_censoring_terminal_censoring_and_joint_draws() -> None:
    scenarios = (
        "partial",
        "forced",
        "terminal60",
        "terminal30",
        "missing",
        "partial",
        "event1",
        *("partial" for _ in range(5)),
    )
    panel = tuple(
        _scenario(setting_id, scenario)
        for setting_id, scenario in zip(
            M03R_SEED17_TOP2000_SETTING_IDS,
            scenarios,
            strict=True,
        )
    )
    result = evaluate_top2000_m03r_v7_2026_cohort_rmst60(panel)
    validate_top2000_m03r_v7_2026_cohort_rmst60_receipt(result)

    rows = result["rows"]
    assert rows[0]["rmst60_sessions"] == pytest.approx(17.5)
    assert rows[1]["rmst60_sessions"] == pytest.approx(10.0)
    assert rows[2]["rmst60_sessions"] == pytest.approx(60.0)
    assert rows[3]["status"] == "unavailable"
    assert rows[3]["rmst60_sessions"] is None
    assert rows[4]["status"] == "unavailable"
    assert rows[4]["entry_units"] == 0.0
    assert rows[0]["uncertainty"] == rows[5]["uncertainty"]
    assert rows[6]["rmst60_sessions"] == pytest.approx(1.0)
    assert rows[0]["uncertainty"]["valid_replicates"] == (
        TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES
    )
    assert result["common_origin_draws_across_settings"] is True
    assert len(result["origin_block_schedule_sha256"]) == 64

    invalid_rmst = copy.deepcopy(result)
    invalid_rmst["rows"][0]["rmst60_sessions"] = 61.0
    invalid_rmst["receipt_sha256"] = cohort._sha256(
        {key: value for key, value in invalid_rmst.items() if key != "receipt_sha256"}
    )
    with pytest.raises(Top2000M03RV72026CohortSurvivalError, match="available"):
        validate_top2000_m03r_v7_2026_cohort_rmst60_receipt(invalid_rmst)

    negative_standard_error = copy.deepcopy(result)
    negative_standard_error["rows"][0]["uncertainty"][
        "bootstrap_standard_error"
    ] = -1.0
    negative_standard_error["receipt_sha256"] = cohort._sha256(
        {
            key: value
            for key, value in negative_standard_error.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(Top2000M03RV72026CohortSurvivalError, match="available"):
        validate_top2000_m03r_v7_2026_cohort_rmst60_receipt(
            negative_standard_error
        )

    reversed_interval = copy.deepcopy(result)
    reversed_interval["rows"][0]["uncertainty"][
        "two_sided_95_percent_interval"
    ] = [40.0, 20.0]
    reversed_interval["receipt_sha256"] = cohort._sha256(
        {
            key: value
            for key, value in reversed_interval.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(Top2000M03RV72026CohortSurvivalError, match="available"):
        validate_top2000_m03r_v7_2026_cohort_rmst60_receipt(reversed_interval)

    negative_entry = copy.deepcopy(result)
    negative_entry["rows"][0]["entry_units"] = -1.0
    negative_entry["receipt_sha256"] = cohort._sha256(
        {key: value for key, value in negative_entry.items() if key != "receipt_sha256"}
    )
    with pytest.raises(Top2000M03RV72026CohortSurvivalError, match="entry units"):
        validate_top2000_m03r_v7_2026_cohort_rmst60_receipt(negative_entry)


def test_exact_reconciliation_and_post_construction_mutation_fail_closed() -> None:
    value = _scenario(M03R_SEED17_TOP2000_SETTING_IDS[0], "partial")
    validate_top2000_m03r_v7_2026_cohort_trajectories(value)
    value.discretionary_event_units_by_age[0, 10] += 0.01
    with pytest.raises(
        Top2000M03RV72026CohortSurvivalError,
        match="reconcile|receipt|risk set",
    ):
        validate_top2000_m03r_v7_2026_cohort_trajectories(value)

    rows = len(_origin_dates())
    entry = np.ones(rows, dtype=np.float64)
    empty = np.zeros((rows, AGE_BIN_COUNT), dtype=np.float64)
    forced = {
        cause.value: empty.copy()
        for cause in (
            TurnoverCause.MEMBERSHIP_FORCED,
            TurnoverCause.AVAILABILITY_FORCED,
            TurnoverCause.RISK_FORCED,
            TurnoverCause.TERMINAL,
        )
    }
    with pytest.raises(
        Top2000M03RV72026CohortSurvivalError,
        match="reconcile",
    ):
        _trajectory_from_arrays(
            M03R_SEED17_TOP2000_SETTING_IDS[0],
            entry=entry,
            events=empty.copy(),
            forced=forced,
            terminal=empty.copy(),
        )


def _accounting(
    *,
    entry: float = 0.0,
    sold_age: int | None = None,
    sold: float = 0.0,
) -> Any:
    sold_units = torch.zeros((1, 2, AGE_BIN_COUNT), dtype=torch.float64)
    if sold_age is not None:
        sold_units[0, 1, sold_age] = sold
    entry_units = torch.tensor([[0.0, entry]], dtype=torch.float64)
    return SimpleNamespace(
        entry_units_added=entry_units,
        sold_units_by_age=sold_units,
    )


def _fake_execution_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    origins: int,
    transition_rows: int,
) -> tuple[
    Top2000M03RV72026RetrospectiveData,
    Top2000M03RV72026EconomicExecutionView,
]:
    monkeypatch.setattr(
        Top2000M03RV72026RetrospectiveData,
        "__post_init__",
        lambda self: None,
    )
    monkeypatch.setattr(
        Top2000M03RV72026EconomicExecutionView,
        "__post_init__",
        lambda self: None,
    )
    monkeypatch.setattr(
        Top2000M03RV72026EconomicExecutionReceipt,
        "__post_init__",
        lambda self: None,
    )
    monkeypatch.setattr(
        Top2000M03RV72026EconomicExecutionReceipt,
        "receipt_sha256",
        property(lambda self: _digest("d")),
    )

    retrospective = object.__new__(Top2000M03RV72026RetrospectiveData)
    object.__setattr__(retrospective, "score_return_dates", _origin_dates(origins))
    object.__setattr__(
        retrospective,
        "identity",
        SimpleNamespace(receipt_sha256=_digest("b")),
    )
    execution_receipt = object.__new__(Top2000M03RV72026EconomicExecutionReceipt)
    for name, value in {
        "executed_transition_rows": transition_rows,
        "executed_state_rows": transition_rows + 1,
        "training_fold_index": 5,
        "local_score_transition_start": 0,
        "local_score_transition_stop_exclusive": origins,
        "chronology_receipt_sha256": _digest("b"),
    }.items():
        object.__setattr__(execution_receipt, name, value)
    execution_view = object.__new__(Top2000M03RV72026EconomicExecutionView)
    object.__setattr__(execution_view, "receipt", execution_receipt)
    object.__setattr__(
        execution_view,
        "sequence",
        SimpleNamespace(
            n_positions=transition_rows + 1,
            batch_size=1,
            cash_index=0,
            num_assets=2,
        ),
    )
    return retrospective, execution_view


def test_transition_major_trace_attribution_preserves_partial_and_forced_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origins = 35
    transitions: list[Any] = []
    remaining_first = 1.0
    for index in range(origins + 2):
        before = torch.zeros((1, 2, AGE_BIN_COUNT), dtype=torch.float64)
        for origin in range(min(index, origins)):
            amount = remaining_first if origin == 0 else 1.0
            before[0, 1, min(index - origin, AGE_BIN_COUNT - 1)] += amount
        accounting = {
            cause: _accounting(
                entry=1.0
                if cause is TurnoverCause.DISCRETIONARY and index < origins
                else 0.0
            )
            for cause in TURNOVER_CAUSES
        }
        if index == 1:
            accounting[TurnoverCause.DISCRETIONARY] = _accounting(
                entry=1.0,
                sold_age=1,
                sold=0.25,
            )
            remaining_first = 0.75
        elif index == 2:
            accounting[TurnoverCause.AVAILABILITY_FORCED] = _accounting(
                sold_age=2,
                sold=0.75,
            )
            remaining_first = 0.0
        transitions.append(
            SimpleNamespace(
                retention_units_before_membership=before,
                accounting_by_cause=accounting,
            )
        )
    trace = Hold30CanonicalTrace(
        boundary_states=cast(Any, tuple(None for _ in range(len(transitions) + 1))),
        decision_states=(),
        pending_intents=(),
        transitions=cast(Any, tuple(transitions)),
    )

    retrospective, execution_view = _fake_execution_inputs(
        monkeypatch,
        origins=origins,
        transition_rows=len(transitions),
    )

    result = build_top2000_m03r_v7_2026_cohort_trajectories(
        trace,
        retrospective,
        execution_view,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        checkpoint_sha256=_digest("a"),
        checkpoint_fold_index=5,
    )
    assert result.entry_units[0] == pytest.approx(1.0)
    assert result.discretionary_event_units_by_age[0, 1] == pytest.approx(0.25)
    assert result.forced_censor_units_by_cause_and_age[
        TurnoverCause.AVAILABILITY_FORCED.value
    ][0, 2] == pytest.approx(0.75)
    assert result.terminal_censor_units_by_age[0].sum() == pytest.approx(0.0)
    assert result.terminal_censor_units_by_age[-1, 2] == pytest.approx(1.0)
    validate_top2000_m03r_v7_2026_cohort_trajectories(result)


def test_capped_age60_origins_receive_proportional_cause_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origins = 62
    transitions: list[Any] = []
    for index in range(origins + 1):
        before = torch.zeros((1, 2, AGE_BIN_COUNT), dtype=torch.float64)
        for origin in range(min(index, origins)):
            before[0, 1, min(index - origin, AGE_BIN_COUNT - 1)] += 1.0
        accounting = {
            cause: _accounting(
                entry=1.0
                if cause is TurnoverCause.DISCRETIONARY and index < origins
                else 0.0
            )
            for cause in TURNOVER_CAUSES
        }
        if index == origins:
            accounting[TurnoverCause.AVAILABILITY_FORCED] = _accounting(
                sold_age=AGE_BIN_COUNT - 1,
                sold=0.6,
            )
            accounting[TurnoverCause.DISCRETIONARY] = _accounting(
                sold_age=AGE_BIN_COUNT - 1,
                sold=1.2,
            )
        transitions.append(
            SimpleNamespace(
                retention_units_before_membership=before,
                accounting_by_cause=accounting,
            )
        )
    trace = Hold30CanonicalTrace(
        boundary_states=cast(Any, tuple(None for _ in range(len(transitions) + 1))),
        decision_states=(),
        pending_intents=(),
        transitions=cast(Any, tuple(transitions)),
    )
    retrospective, execution_view = _fake_execution_inputs(
        monkeypatch,
        origins=origins,
        transition_rows=len(transitions),
    )

    result = build_top2000_m03r_v7_2026_cohort_trajectories(
        trace,
        retrospective,
        execution_view,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        checkpoint_sha256=_digest("a"),
        checkpoint_fold_index=5,
    )
    age60 = AGE_BIN_COUNT - 1
    availability = result.forced_censor_units_by_cause_and_age[
        TurnoverCause.AVAILABILITY_FORCED.value
    ]
    np.testing.assert_allclose(availability[:3, age60], 0.2)
    np.testing.assert_allclose(
        result.discretionary_event_units_by_age[:3, age60],
        0.4,
    )
    np.testing.assert_allclose(
        result.terminal_censor_units_by_age[:3, age60],
        0.4,
    )
    assert availability[:, age60].sum() == pytest.approx(0.6)
    assert result.discretionary_event_units_by_age[:, age60].sum() == pytest.approx(
        1.2
    )
    removed = (
        result.discretionary_event_units_by_age
        + result.terminal_censor_units_by_age
        + sum(
            result.forced_censor_units_by_cause_and_age.values(),
            start=np.zeros_like(result.discretionary_event_units_by_age),
        )
    )
    np.testing.assert_allclose(removed.sum(axis=1), result.entry_units)
    validate_top2000_m03r_v7_2026_cohort_trajectories(result)
