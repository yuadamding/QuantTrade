from __future__ import annotations

import copy

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger, TurnoverCause
from rl_quant.evaluation.hold30_controls import Hold30ControlGrossTrace
from rl_quant.evaluation.hold30_endpoints import (
    Hold30EndpointError,
    evaluate_hold30_endpoints,
    verify_hold30_endpoint_receipt,
)
from rl_quant.execution.hold30 import Hold30BuiltAction
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30_freeze import sha256_payload
from rl_quant.training.hold30_runtime import (
    Hold30CanonicalTrace,
    Hold30ChronologicalRuntime,
    Hold30Sequence,
)


def _digest(value: object) -> str:
    return sha256_payload(value)


class _Policy(torch.nn.Module):
    def hold30_intent(self, state_t, prev_weights, available, age_summaries=None):
        del available, age_summaries
        decision = int(state_t[0, 0, 0].item())
        target = prev_weights
        if decision == 0:
            target = prev_weights.new_tensor([[0.99, 0.01, 0.0]])
        return Hold30Intent(
            target_logits=target,
            gate=torch.ones(prev_weights.shape[0], dtype=prev_weights.dtype),
        )


class _Builder:
    def __call__(self, intent, repaired_ledger, benchmark, trade_mask, caps, gross):
        del benchmark, trade_mask, caps, gross
        target = intent.target_logits
        assert target is not None
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


def _trace_and_c1():
    dtype = torch.float64
    positions = 66
    assets = 3
    axis = _digest("endpoint-axis")
    initial_weights = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype)
    state_tensor = torch.zeros((positions, 1, assets, 1), dtype=dtype)
    state_tensor[..., 0] = torch.arange(positions, dtype=dtype).view(-1, 1, 1)
    masks = torch.ones((positions, 1, assets), dtype=torch.bool)
    returns = torch.zeros((positions - 1, 1, assets), dtype=dtype)
    returns[:, 0, 1] = 0.001 + 0.0002 * torch.sin(torch.arange(positions - 1, dtype=dtype))
    benchmark = initial_weights.unsqueeze(0).expand(positions, -1, -1).clone()
    sequence = Hold30Sequence(
        decision_state=state_tensor,
        asset_returns=returns,
        decision_available=masks.clone(),
        fill_membership=masks.clone(),
        fill_availability=masks.clone(),
        benchmark_weights=benchmark,
        risk_asset_caps=torch.ones((positions, 1, assets), dtype=dtype),
        risk_gross_max=torch.ones((positions, 1), dtype=dtype),
        benchmark_net_returns=torch.zeros((positions - 1, 1), dtype=dtype),
        initial_ledger=CohortLedger.from_weights(initial_weights, cash_index=0),
        cost_rate=0.002,
        axis_id=axis,
    )
    runtime = Hold30ChronologicalRuntime("H0", action_builder=_Builder())
    runtime_state = runtime.initial_state(sequence)
    boundaries = [runtime_state.detach()]
    states = []
    pendings = []
    transitions = []
    with torch.no_grad():
        while runtime_state.position_index < sequence.n_positions - 1:
            states.append(sequence.decision_state[runtime_state.position_index].clone())
            runtime_state = runtime.decide(_Policy(), sequence, runtime_state)
            assert runtime_state.pending_intent is not None
            pendings.append(runtime_state.pending_intent.detach())
            runtime_state, transition = runtime.advance(sequence, runtime_state)
            transitions.append(transition)
            boundaries.append(runtime_state.detach())
    trace = Hold30CanonicalTrace(
        boundary_states=tuple(boundaries),
        decision_states=tuple(states),
        pending_intents=tuple(pendings),
        transitions=tuple(transitions),
    )

    rows = positions - 1
    matrix = torch.zeros((rows, 1, assets), dtype=dtype)
    c1_weights = benchmark
    score = torch.zeros(rows, dtype=torch.bool)
    score[:63] = True
    c1 = Hold30ControlGrossTrace(
        control_id="C1",
        axis_id=axis,
        asset_ids=("CASH", "A", "B"),
        weights=c1_weights,
        pretrade_weights=c1_weights[:-1].clone(),
        gross_returns=torch.zeros((rows, 1), dtype=dtype),
        startup_delta=matrix.clone(),
        membership_forced_delta=matrix.clone(),
        availability_forced_delta=matrix.clone(),
        risk_forced_delta=matrix.clone(),
        discretionary_delta=matrix.clone(),
        terminal_delta=matrix.clone(),
        score_mask=score,
        outer_start=0,
        fitting_rows=(),
        source_receipt_sha256=_digest("c1-source"),
        strategy_inputs_sha256=_digest("c1-inputs"),
    )
    return trace, c1, score


def test_endpoint_cost_ladder_continuing_and_liquidated_are_separate() -> None:
    trace, c1, score = _trace_and_c1()
    receipt = evaluate_hold30_endpoints(
        trace,
        c1,
        score_mask=score,
        learned_source_receipt_sha256=_digest("ensemble"),
    )

    verify_hold30_endpoint_receipt(
        receipt,
        trace=trace,
        c1_trace=c1,
        score_mask=score,
        learned_source_receipt_sha256=_digest("ensemble"),
    )
    assert receipt["score_indices"] == list(range(63))
    assert receipt["rungs"]["20"]["continuing_active_log_wealth"] > 0.0
    assert receipt["rungs"]["40"]["continuing_active_log_wealth"] < receipt["rungs"][
        "10"
    ]["continuing_active_log_wealth"]
    assert receipt["rungs"]["20"]["liquidated_active_log_wealth"] < receipt["rungs"][
        "20"
    ]["continuing_active_log_wealth"]
    assert receipt["actions"]["constructed_to_filled_one_way_max"] == 0.0
    assert receipt["holding_telemetry"]["product_limit_survival"]["entry_units"] == pytest.approx(
        0.01
    )


def test_endpoint_receipt_and_score_alignment_fail_closed() -> None:
    trace, c1, score = _trace_and_c1()
    receipt = evaluate_hold30_endpoints(
        trace,
        c1,
        score_mask=score,
        learned_source_receipt_sha256=_digest("ensemble"),
    )
    tampered = copy.deepcopy(receipt)
    tampered["rungs"]["20"]["continuing_active_log_wealth"] = 9.0
    with pytest.raises(Hold30EndpointError, match="self-hash"):
        verify_hold30_endpoint_receipt(tampered)

    short = score.clone()
    short[62] = False
    with pytest.raises(Hold30EndpointError, match="63 scored"):
        evaluate_hold30_endpoints(
            trace,
            c1,
            score_mask=short,
            learned_source_receipt_sha256=_digest("ensemble"),
        )


def test_c1_cause_accounting_remains_zero_in_fixture() -> None:
    _trace, c1, _score = _trace_and_c1()
    assert all(
        float(c1.turnover_by_cause[cause].sum()) == 0.0 for cause in TurnoverCause
    )
