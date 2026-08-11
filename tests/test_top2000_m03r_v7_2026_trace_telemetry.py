"""Focused tests for runtime-owned 2026 retrospective telemetry."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
import torch

from rl_quant.envs.hold30 import TURNOVER_CAUSES, TurnoverCause
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    Top2000M03RV72026RetrospectiveData,
    Top2000M03RV72026RetrospectiveSourceEvidence,
    compose_top2000_m03r_v7_2026_retrospective_data,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_trace_telemetry import (
    Top2000M03RV72026TraceTelemetryError,
    adapt_top2000_m03r_v7_2026_trace,
    validate_top2000_m03r_v7_2026_trace_evaluation_inputs,
)
from rl_quant.models.daily_policy import Hold30Intent, hold30_release_hazard
from rl_quant.models.hold30_exit_action_v6 import (
    M03R_V6_CONTINUOUS_ACTION_INDEX,
    M03R_V6_EXIT_ACTION_COUNT,
    M03R_V6_EXIT_ACTION_INDEX,
    M03R_V6_HOLD_ACTION_INDEX,
    M03RV6ExitAction,
    straight_through_m03r_v6_exit_action,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.training.hold30 import Hold30ReplayGeometry
from rl_quant.training.hold30_runtime import (
    Hold30CanonicalTrace,
    Hold30ChronologicalRuntime,
    Hold30Policy,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _axis_digest(values: tuple[str, ...]) -> str:
    payload = (
        json.dumps(
            list(values),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _weekdays(start: dt.date, stop: dt.date) -> tuple[str, ...]:
    values: list[str] = []
    current = start
    while current <= stop:
        if current.weekday() < 5:
            values.append(current.isoformat())
        current += dt.timedelta(days=1)
    return tuple(values)


def _daily(
    dates: tuple[str, ...],
    actions: tuple[str, ...],
    *,
    start_close: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = len(dates)
    bars = torch.zeros((rows, len(actions), 5), dtype=torch.float64)
    close = start_close + torch.arange(rows, dtype=torch.float64) * 0.01
    bars[:, 1:, 0] = close.unsqueeze(-1)
    bars[:, 1:, 1] = close.unsqueeze(-1) + 1.0
    bars[:, 1:, 2] = close.unsqueeze(-1) - 1.0
    bars[:, 1:, 3] = close.unsqueeze(-1)
    bars[:, 1:, 4] = 1_000_000.0
    return bars, torch.ones((rows, len(actions)), dtype=torch.bool)


def _retrospective() -> Top2000M03RV72026RetrospectiveData:
    actions = ("CASH", "A1", "A2", "A3")
    pre_dates = _weekdays(dt.date(2024, 12, 2), dt.date(2025, 12, 31))
    pre_bars, pre_available = _daily(pre_dates, actions, start_close=100.0)
    pre = Top2000VerifiedDevelopmentCache(
        daily_ohlcv=pre_bars,
        availability=pre_available,
        exchange_dates=pre_dates,
        action_ids=actions,
        cache_sha256=_digest("pre-cache"),
        cache_identity=_digest("pre-identity"),
        search_identity=_digest("search"),
        action_hash=_axis_digest(actions),
        bar_seconds=300,
        acknowledgement="I acknowledge TOP2000 results are development-only",
        development_only=True,
        bars_only=True,
    )
    raw_dates = (
        pre_dates[-1],
        "2026-01-02",
        "2026-01-05",
        "2026-06-22",
        "2026-06-23",
    )
    raw_bars, raw_available = _daily(raw_dates, actions, start_close=102.0)
    raw_bars[0] = pre_bars[-1]
    raw_available[0] = pre_available[-1]
    source = Top2000M03RV72026RetrospectiveSourceEvidence(
        base_dataset_identity=_digest("base"),
        search_identity=_digest("search"),
        lockbox_partition_names_hash=_digest("lockbox"),
        test_identity=_digest("test"),
        test_partition_inventory_sha256=_digest("partitions"),
        manifest_sha256=_digest("manifest"),
        universe_sha256=_digest("universe"),
        training_completion_receipt_sha256=_digest("complete"),
        evaluation_contract_sha256=_digest("contract"),
        raw_first_exchange_date=raw_dates[0],
        raw_last_exchange_date=raw_dates[-1],
    )
    return compose_top2000_m03r_v7_2026_retrospective_data(
        pre,
        retrospective_daily_ohlcv=raw_bars,
        retrospective_availability=raw_available,
        retrospective_exchange_dates=raw_dates,
        retrospective_action_ids=actions,
        source_evidence=source,
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )


class _ThreeWayPolicy(torch.nn.Module):
    def __init__(self, score_start: int) -> None:
        super().__init__()
        self.score_start = score_start
        self.calls = 0

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del state_t, age_summaries
        batch, assets = prev_weights.shape
        logits = prev_weights.new_full((batch, assets, M03R_V6_EXIT_ACTION_COUNT), -4.0)
        selected = [M03R_V6_HOLD_ACTION_INDEX] * assets
        if self.calls >= self.score_start:
            selected = [
                M03R_V6_HOLD_ACTION_INDEX,
                M03R_V6_HOLD_ACTION_INDEX,
                M03R_V6_CONTINUOUS_ACTION_INDEX,
                M03R_V6_EXIT_ACTION_INDEX,
            ]
        for asset, action_index in enumerate(selected):
            logits[:, asset, action_index] = 4.0
        risky = available.bool().clone()
        risky[:, 0] = False
        soft, decision = straight_through_m03r_v6_exit_action(logits)
        hold = torch.zeros_like(soft)
        hold[..., M03R_V6_HOLD_ACTION_INDEX] = 1.0
        soft = torch.where(risky.unsqueeze(-1), soft, hold)
        decision = torch.where(risky.unsqueeze(-1), decision, hold)
        action = M03RV6ExitAction(
            logits=logits,
            soft_probabilities=soft,
            decision_st=decision,
            risky_available=risky,
            exact_hold_atom_enabled=True,
        )
        hazard = prev_weights.new_tensor([[-12.0, -3.0, 0.75, 2.0]])
        self.calls += 1
        return Hold30Intent(
            entry_scores=torch.zeros_like(prev_weights),
            hazard_residual=hazard,
            exposure_residual=prev_weights.new_zeros(batch),
            exit_action_v6=action,
        )


class _FixedPriorPolicy(torch.nn.Module):
    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del state_t, available, age_summaries
        return Hold30Intent(
            entry_scores=torch.zeros_like(prev_weights),
            hazard_residual=torch.zeros_like(prev_weights),
            exposure_residual=prev_weights.new_zeros(prev_weights.shape[0]),
        )


def _trace(
    retrospective: Top2000M03RV72026RetrospectiveData,
    policy: Hold30Policy,
) -> Hold30CanonicalTrace:
    runtime = Hold30ChronologicalRuntime("H2")
    roles = Hold30ReplayGeometry().roles(retrospective.sequence.n_positions)
    with torch.no_grad():
        trace, _rows = runtime.canonical_pass(policy, retrospective.sequence, roles)
    return trace


@pytest.fixture(scope="module", autouse=True)
def _bounded_torch_threads() -> Iterator[None]:
    """Tiny tensor accounting is slower with the host's large thread pool."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def retrospective_data() -> Top2000M03RV72026RetrospectiveData:
    return _retrospective()


@pytest.fixture(scope="module")
def learned_trace(
    retrospective_data: Top2000M03RV72026RetrospectiveData,
) -> Hold30CanonicalTrace:
    return _trace(
        retrospective_data,
        _ThreeWayPolicy(retrospective_data.identity.score_transition_start),
    )


@pytest.fixture(scope="module")
def fixed_trace(
    retrospective_data: Top2000M03RV72026RetrospectiveData,
) -> Hold30CanonicalTrace:
    return _trace(retrospective_data, _FixedPriorPolicy())


def test_adapter_scores_only_exact_suffix_and_maps_runtime_and_benchmark_arrays(
    retrospective_data: Top2000M03RV72026RetrospectiveData,
    learned_trace: Hold30CanonicalTrace,
) -> None:
    retrospective = retrospective_data
    trace = learned_trace
    result = adapt_top2000_m03r_v7_2026_trace(
        trace,
        retrospective,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        checkpoint_sha256="a" * 64,
        checkpoint_fold_index=5,
    )

    score = retrospective.score_transition_slice
    assert result.score_dates == retrospective.score_return_dates
    assert result.portfolio_gross_returns.shape == (4,)
    np.testing.assert_allclose(
        result.portfolio_gross_returns,
        np.asarray(
            [float(row.holding_return) for row in trace.transitions[score]],
        ),
    )
    np.testing.assert_allclose(
        result.benchmark_gross_returns,
        retrospective.benchmark.gross_returns[score].numpy(),
    )
    for cause in TURNOVER_CAUSES:
        np.testing.assert_allclose(
            result.portfolio_turnover_by_cause[cause.value],
            np.asarray(
                [
                    float(row.turnover_by_cause[cause])
                    for row in trace.transitions[score]
                ]
            ),
        )
    np.testing.assert_allclose(
        result.benchmark_turnover_by_cause[
            TurnoverCause.DISCRETIONARY.value
        ],
        retrospective.benchmark.monthly_rebalance_one_way_turnover[score].numpy(),
    )
    np.testing.assert_allclose(
        result.benchmark_turnover_by_cause[
            TurnoverCause.AVAILABILITY_FORCED.value
        ],
        retrospective.benchmark.availability_forced_one_way_turnover[score].numpy(),
    )
    np.testing.assert_allclose(
        result.benchmark_turnover_by_cause[TurnoverCause.RISK_FORCED.value],
        retrospective.benchmark.risk_forced_one_way_turnover[score].numpy(),
    )
    assert np.all(
        result.benchmark_turnover_by_cause[TurnoverCause.STARTUP.value] == 0.0
    )
    assert result.receipt.completed_transition_rows == len(trace.transitions)
    assert result.receipt.scored_transition_rows == 4
    assert result.receipt.development_only
    assert not result.receipt.dataset_reportable
    assert not result.receipt.scientific_reporting_eligible
    assert not result.receipt.promotion_eligible


def test_age_risk_set_actions_and_continuous_hazard_use_exact_requested_stage(
    retrospective_data: Top2000M03RV72026RetrospectiveData,
    learned_trace: Hold30CanonicalTrace,
) -> None:
    retrospective = retrospective_data
    trace = learned_trace
    result = adapt_top2000_m03r_v7_2026_trace(
        trace,
        retrospective,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        checkpoint_sha256="b" * 64,
        checkpoint_fold_index=5,
    )
    first = retrospective.identity.score_transition_start
    transition = trace.transitions[first]
    discretionary = transition.accounting_by_cause[TurnoverCause.DISCRETIONARY]
    after = trace.boundary_states[first + 1].ledger.economic_value
    entry = torch.zeros_like(after)
    entry[..., 0] = discretionary.net_buys
    exact_risk = after - entry + discretionary.sold_value_by_age
    exact_risk[:, retrospective.sequence.cash_index] = 0.0
    np.testing.assert_allclose(
        np.asarray(result.telemetry.age_notional_at_risk)[0, 0],
        exact_risk.sum(dim=(0, 1)).numpy(),
        atol=1.0e-12,
    )
    expected_discretionary = discretionary.sold_value_by_age.clone()
    expected_discretionary[:, retrospective.sequence.cash_index] = 0.0
    np.testing.assert_allclose(
        np.asarray(result.telemetry.discretionary_exit_notional_by_age)[0, 0],
        expected_discretionary.sum(dim=(0, 1)).numpy(),
        atol=1.0e-12,
    )
    action_counts = result.telemetry.action_counts_by_type
    np.testing.assert_array_equal(action_counts["HOLD"], np.ones((1, 4)))
    np.testing.assert_array_equal(action_counts["CONTINUOUS"], np.ones((1, 4)))
    np.testing.assert_array_equal(action_counts["EXIT"], np.ones((1, 4)))

    observed = np.asarray(result.telemetry.continuous_hazard_observed)
    assert observed[0, 0].tolist() == [False, False, True, False]
    hazard = np.asarray(result.telemetry.continuous_hazard)
    residual = transition.raw_intent.hazard_residual
    assert residual is not None
    by_age = hold30_release_hazard(
        torch.arange(61, dtype=residual.dtype),
        residual[0, 2],
    )
    held = exact_risk[0, 2]
    expected_hazard = float((held * by_age).sum() / held.sum())
    assert hazard[0, 0, 2] == pytest.approx(expected_hazard)
    assert np.all(hazard[0, 0, [0, 1, 3]] == 0.0)


def test_fixed_prior_is_counted_as_continuous_without_fabricating_exact_actions(
    retrospective_data: Top2000M03RV72026RetrospectiveData,
    fixed_trace: Hold30CanonicalTrace,
) -> None:
    retrospective = retrospective_data
    trace = fixed_trace
    result = adapt_top2000_m03r_v7_2026_trace(
        trace,
        retrospective,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[3],
        checkpoint_sha256="c" * 64,
        checkpoint_fold_index=4,
    )

    actions = result.telemetry.action_counts_by_type
    assert np.all(actions["HOLD"] == 0.0)
    assert np.all(actions["EXIT"] == 0.0)
    assert np.all(actions["CONTINUOUS"] == 3.0)
    observed = np.asarray(result.telemetry.continuous_hazard_observed)
    assert np.all(observed[..., 1:])
    assert not np.any(observed[..., 0])


def test_receipt_revalidates_content_and_trace_geometry_fails_closed(
    retrospective_data: Top2000M03RV72026RetrospectiveData,
    learned_trace: Hold30CanonicalTrace,
) -> None:
    retrospective = retrospective_data
    trace = learned_trace
    result = adapt_top2000_m03r_v7_2026_trace(
        trace,
        retrospective,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        checkpoint_sha256="d" * 64,
        checkpoint_fold_index=5,
    )
    validate_top2000_m03r_v7_2026_trace_evaluation_inputs(result)
    result.portfolio_gross_returns[0] += 1.0e-6
    with pytest.raises(Top2000M03RV72026TraceTelemetryError, match="content-bound"):
        validate_top2000_m03r_v7_2026_trace_evaluation_inputs(result)

    incomplete = replace(trace, transitions=trace.transitions[:-1])
    with pytest.raises(Top2000M03RV72026TraceTelemetryError, match="completed once"):
        adapt_top2000_m03r_v7_2026_trace(
            incomplete,
            retrospective,
            setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
            checkpoint_sha256="e" * 64,
            checkpoint_fold_index=5,
        )


def test_overlapping_cause_legs_are_rejected(
    retrospective_data: Top2000M03RV72026RetrospectiveData,
    learned_trace: Hold30CanonicalTrace,
) -> None:
    retrospective = retrospective_data
    trace = learned_trace
    index = retrospective.identity.score_transition_start
    transition = trace.transitions[index]
    accounting = transition.accounting_by_cause[TurnoverCause.DISCRETIONARY]
    sold_asset = int(accounting.net_sells[0].argmax())
    assert float(accounting.net_sells[0, sold_asset]) > 0.0
    overlapping_buys = accounting.net_buys.clone()
    overlapping_buys[0, sold_asset] = 1.0e-4
    overlapping = replace(accounting, net_buys=overlapping_buys)
    accounting_by_cause = dict(transition.accounting_by_cause)
    accounting_by_cause[TurnoverCause.DISCRETIONARY] = overlapping
    corrupted_transition = replace(
        transition,
        accounting_by_cause=accounting_by_cause,
        discretionary_accounting=overlapping,
    )
    transitions = list(trace.transitions)
    transitions[index] = corrupted_transition
    corrupted = replace(trace, transitions=tuple(transitions))

    with pytest.raises(Top2000M03RV72026TraceTelemetryError, match="overlapping"):
        adapt_top2000_m03r_v7_2026_trace(
            corrupted,
            retrospective,
            setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
            checkpoint_sha256="f" * 64,
            checkpoint_fold_index=5,
        )
