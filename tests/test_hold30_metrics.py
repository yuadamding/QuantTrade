from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger, TurnoverCause
from rl_quant.evaluation.hold30_metrics import (
    Hold30TelemetryError,
    aggregate_hold30_metrics,
    hold30_metrics_digest,
    pool_hold30_product_limit,
    verify_hold30_metrics_digest,
)
from rl_quant.execution.hold30 import Hold30BuiltAction
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.training.hold30_runtime import (
    Hold30CanonicalTrace,
    Hold30ChronologicalRuntime,
    Hold30Sequence,
)


def _sequence(
    positions: int,
    *,
    initial_weights: torch.Tensor | None = None,
) -> Hold30Sequence:
    dtype = torch.float64
    weights = (
        torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype)
        if initial_weights is None
        else initial_weights.to(dtype=dtype)
    )
    batch, assets = weights.shape
    state = torch.zeros((positions, batch, assets, 1), dtype=dtype)
    state[..., 0] = torch.arange(positions, dtype=dtype).view(-1, 1, 1)
    masks = torch.ones((positions, batch, assets), dtype=torch.bool)
    return Hold30Sequence(
        decision_state=state,
        asset_returns=torch.zeros((positions - 1, batch, assets), dtype=dtype),
        decision_available=masks.clone(),
        fill_membership=masks.clone(),
        fill_availability=masks.clone(),
        benchmark_weights=weights.unsqueeze(0).expand(positions, -1, -1).clone(),
        risk_asset_caps=torch.ones((positions, batch, assets), dtype=dtype),
        risk_gross_max=torch.ones((positions, batch), dtype=dtype),
        benchmark_net_returns=torch.zeros((positions - 1, batch), dtype=dtype),
        initial_ledger=CohortLedger.from_weights(
            weights,
            cash_index=0,
            initial_age=0,
            track_initial_units=False,
        ),
        cost_rate=0.0,
        axis_id="metrics-fixture-v1",
    )


class _ScheduledPolicy(torch.nn.Module):
    def __init__(self, schedule: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.schedule = schedule

    def hold30_intent(self, state_t, prev_weights, available, age_summaries=None):
        del available, age_summaries
        decision = int(state_t[0, 0, 0].item())
        target = self.schedule.get(decision, prev_weights).to(prev_weights)
        return Hold30Intent(
            target_logits=target.expand_as(prev_weights),
            gate=torch.ones(prev_weights.shape[0], dtype=prev_weights.dtype),
        )


class _TargetBuilder:
    def __call__(self, intent, repaired_ledger, benchmark, trade_mask, caps, gross):
        del benchmark, trade_mask, caps, gross
        assert intent.target_logits is not None
        target = intent.target_logits
        delta = target - repaired_ledger.weights
        turnover = 0.5 * delta.abs().sum(-1)
        return Hold30BuiltAction(
            target_weights=target,
            requested_delta=delta,
            constructed_delta=delta,
            requested_turnover=turnover,
            constructed_turnover=turnover,
            desired_risky_exposure=target[:, 1:].sum(-1),
            proposed_release_by_age=target.new_zeros((*target.shape, 61)),
            proposed_release=torch.zeros_like(target),
            capacity_shortfall=torch.zeros_like(turnover),
        )


def _trace(policy: torch.nn.Module, sequence: Hold30Sequence) -> Hold30CanonicalTrace:
    runtime = Hold30ChronologicalRuntime("H0", action_builder=_TargetBuilder())
    state = runtime.initial_state(sequence)
    boundaries = [state.detach()]
    decision_states = []
    pendings = []
    transitions = []
    with torch.no_grad():
        while state.position_index < sequence.n_positions - 1:
            decision_states.append(sequence.decision_state[state.position_index].detach().clone())
            state = runtime.decide(policy, sequence, state)
            assert state.pending_intent is not None
            pendings.append(state.pending_intent.detach())
            state, transition = runtime.advance(sequence, state)
            transitions.append(transition)
            boundaries.append(state.detach())
    return Hold30CanonicalTrace(
        boundary_states=tuple(boundaries),
        decision_states=tuple(decision_states),
        pending_intents=tuple(pendings),
        transitions=tuple(transitions),
    )


def test_exact_entry_forced_exit_discretionary_exit_and_survival() -> None:
    sequence = _sequence(66)
    sequence.fill_membership[6, 0, 1] = False
    policy = _ScheduledPolicy(
        {
            0: torch.tensor([0.98, 0.01, 0.01], dtype=torch.float64),
            5: torch.tensor([0.99, 0.0, 0.01], dtype=torch.float64),
            10: torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        }
    )

    report = aggregate_hold30_metrics(_trace(policy, sequence))

    survival5 = report["retention_survival"]["5"]
    assert survival5["eligible_entry_units"] == pytest.approx(0.02)
    assert survival5["before_membership"]["value"] == pytest.approx(1.0)
    assert survival5["after_forced_repairs"]["value"] == pytest.approx(0.5)
    assert survival5["after_discretionary_trade"]["value"] == pytest.approx(0.5)
    assert survival5["cumulative_forced_exit_fraction"]["value"] == pytest.approx(0.5)

    survival10 = report["retention_survival"]["10"]
    assert survival10["before_membership"]["value"] == pytest.approx(0.5)
    assert survival10["after_forced_repairs"]["value"] == pytest.approx(0.5)
    assert survival10["after_discretionary_trade"]["value"] == pytest.approx(0.0)
    assert survival10["cumulative_discretionary_exit_fraction"]["value"] == pytest.approx(0.5)
    assert report["retention_survival"]["60"]["before_membership"]["value"] == 0.0

    sale = report["sale_age"]
    assert sale["weighting"] == "score_origin_return_neutral_entry_notional_units"
    assert sale["median_sessions_capped_60"]["value"] == 10.0
    assert sale["median_sessions_capped_60"]["sold_entry_units"] == pytest.approx(0.01)
    assert sale["quantiles_sessions_capped_60"]["0.50"] == 10.0
    assert sale["young_sell_fraction"]["lt_10"]["value"] == 0.0
    assert sale["young_sell_fraction"]["lt_20"]["value"] == 1.0
    product = report["product_limit_survival"]
    assert product["entry_units"] == pytest.approx(0.02)
    assert product["horizons"]["5"]["discretionary_survival"] == pytest.approx(1.0)
    assert product["horizons"]["10"]["all_cause_survival"] == pytest.approx(0.5)
    assert product["horizons"]["20"]["discretionary_survival"] == pytest.approx(0.0)
    turnover = report["turnover"]
    assert turnover["causes"][TurnoverCause.MEMBERSHIP_FORCED.value][
        "total_one_way_mean_per_path"
    ] == pytest.approx(0.01)
    lifecycle = turnover["discretionary_lifecycle_partition"]
    assert lifecycle["entry"]["total_one_way_mean_per_path"] == pytest.approx(0.02)
    assert lifecycle["exit"]["total_one_way_mean_per_path"] == pytest.approx(0.01)
    assert lifecycle["resize"]["total_one_way_mean_per_path"] == pytest.approx(0.0)
    assert lifecycle["maximum_reconciliation_error"] == pytest.approx(0.0)


def test_zero_denominators_are_explicit_json_nulls_and_digest_is_stable() -> None:
    trace = _trace(_ScheduledPolicy({}), _sequence(3))

    first = aggregate_hold30_metrics(trace)
    second = aggregate_hold30_metrics(trace)

    assert first == second
    assert verify_hold30_metrics_digest(first)
    assert first["sha256"] == hold30_metrics_digest(first)
    json.dumps(first, sort_keys=True, allow_nan=False)
    assert first["current_holding_age"]["notional_weighted_sessions_capped_60"] == {
        "value": None,
        "numerator": 0.0,
        "denominator": 0.0,
        "null_reason": "terminal_book_has_no_risky_notional",
    }
    assert first["sale_age"]["young_sell_fraction"]["lt_30"]["value"] is None
    assert first["retention_survival"]["5"]["before_membership"]["value"] is None
    assert first["risky_notional_portfolio_overlap"]["5"]["value"] is None
    assert first["turnover"]["turnover_implied_horizon_sessions_approx"]["value"] is None

    tampered = dict(first)
    tampered["schema_version"] = "tampered"
    assert not verify_hold30_metrics_digest(tampered)


def test_current_age_and_gross_pnl_are_attributed_to_starting_age() -> None:
    weights = torch.tensor([[0.99, 0.01, 0.0]], dtype=torch.float64)
    sequence = _sequence(2, initial_weights=weights)
    sequence = replace(
        sequence,
        initial_ledger=CohortLedger.from_weights(
            weights,
            cash_index=0,
            initial_age=9,
            track_initial_units=False,
        ),
    )
    sequence.asset_returns[0, 0, 1] = 0.1

    report = aggregate_hold30_metrics(_trace(_ScheduledPolicy({}), sequence))

    assert report["current_holding_age"]["notional_weighted_sessions_capped_60"][
        "value"
    ] == pytest.approx(10.0)
    pnl = report["pnl_contribution_by_position_age"]
    assert pnl["buckets"]["0_9"]["gross_pnl_mean_per_path"] == pytest.approx(0.001)
    assert pnl["buckets"]["10_19"]["gross_pnl_mean_per_path"] == 0.0
    assert pnl["cash_gross_pnl_mean_per_path"] == pytest.approx(0.0, abs=1e-14)


def test_malformed_nonconsecutive_trace_fails_closed() -> None:
    trace = _trace(_ScheduledPolicy({}), _sequence(3))
    broken_transition = replace(trace.transitions[1], decision_index=99)
    broken = replace(trace, transitions=(trace.transitions[0], broken_transition))

    with pytest.raises(Hold30TelemetryError, match="not consecutive"):
        aggregate_hold30_metrics(broken)


def test_score_origin_mask_excludes_warmup_entries_from_holding_gate_population() -> None:
    sequence = _sequence(12)
    policy = _ScheduledPolicy(
        {
            0: torch.tensor([0.99, 0.01, 0.0], dtype=torch.float64),
            5: torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        }
    )
    trace = _trace(policy, sequence)
    score_mask = torch.zeros(len(trace.transitions), dtype=torch.bool)
    score_mask[6:] = True
    report = aggregate_hold30_metrics(
        trace,
        score_origin_mask=score_mask,
    )

    assert report["product_limit_survival"]["entry_units"] == 0.0
    assert report["sale_age"]["median_sessions_capped_60"]["value"] is None
    assert report["sale_age"]["economic_value_weighted_secondary"][
        "median_sessions_capped_60"
    ]["value"] == 5.0


def test_six_fold_product_limit_pools_risk_and_events_before_product() -> None:
    sequence = _sequence(66)
    policy = _ScheduledPolicy(
        {
            0: torch.tensor([0.99, 0.01, 0.0], dtype=torch.float64),
            10: torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        }
    )
    reports = {
        fold: aggregate_hold30_metrics(
            _trace(policy, replace(sequence, axis_id=f"metrics-fold-{fold}"))
        )
        for fold in range(6)
    }

    pooled = pool_hold30_product_limit(reports)

    assert verify_hold30_metrics_digest(pooled)
    assert pooled["pooled"]["entry_units"] == pytest.approx(0.06)
    assert pooled["pooled"]["horizons"]["10"]["discretionary_survival"] == 1.0
    assert pooled["pooled"]["horizons"]["20"]["discretionary_survival"] == 0.0
