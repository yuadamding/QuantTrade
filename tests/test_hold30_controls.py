from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from rl_quant.datasets.hold30 import (
    HOLD30_CASH_RETURN_RULE,
    HOLD30_UNIVERSE_MODE,
    Hold30AsOfEvidence,
    Hold30DatasetSequence,
    Hold30PointInTimeProvenance,
)
from rl_quant.envs.hold30 import TurnoverCause
from rl_quant.evaluation.hold30_controls import (
    HOLD30_COST_RUNGS_BPS,
    Hold30ControlError,
    accept_c1_bound_trace,
    c4_momentum_scores,
    construct_c0_cash,
    construct_c2_daily_equal_weight,
    construct_c3_initial_universe_hold,
    construct_c4_momentum,
    price_hold30_cost_ladder,
)

POSITIONS = 96
ASSETS = 301  # CASH plus exactly 300 PIT active names.
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
HISTORY = 21


def _digest(character: str) -> str:
    return character * 64


@dataclass(frozen=True)
class _Fixture:
    sequence: Hold30DatasetSequence
    c1_pretrade: torch.Tensor
    c1_gross: torch.Tensor
    c1_membership_delta: torch.Tensor
    c1_availability_delta: torch.Tensor
    c1_risk_delta: torch.Tensor
    c1_discretionary_delta: torch.Tensor
    monthly: torch.Tensor
    closes: torch.Tensor
    close_valid: torch.Tensor
    close_known: torch.Tensor


def _provenance() -> Hold30PointInTimeProvenance:
    return Hold30PointInTimeProvenance(
        data_snapshot_sha256=_digest("a"),
        raw_market_data_sha256=_digest("b"),
        universe_events_sha256=_digest("c"),
        tradability_events_sha256=_digest("d"),
        corporate_actions_sha256=_digest("e"),
        identifier_events_sha256=_digest("f"),
        c1_benchmark_trace_sha256=_digest("1"),
        risk_limits_sha256=_digest("2"),
        universe_mode=HOLD30_UNIVERSE_MODE,
        universe_rule_id="pit-active300-controls-test-v1",
        stable_asset_id_namespace="perm-id-v1",
        benchmark_id="C1",
        cash_asset_id="CASH",
        cash_return_rule=HOLD30_CASH_RETURN_RULE,
    )


def _repair_cash(weights: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    target = torch.where(allowed, weights, torch.zeros_like(weights))
    target[:, 0] = 0.0
    target[:, 0] = (1.0 - target.sum(dim=-1)).clamp_min(0.0)
    return target


def _fixture() -> _Fixture:
    dtype = torch.float64
    batch = 1
    first_decision = 1_704_067_200_000
    decision_ts = first_decision + torch.arange(POSITIONS, dtype=torch.int64) * DAY_MS
    fill_ts = decision_ts - 6 * HOUR_MS
    fill_ts[0] = decision_ts[0] - HOUR_MS
    shape = (POSITIONS, batch, ASSETS)

    membership = torch.ones(shape, dtype=torch.bool)
    decision_tradability = torch.ones(shape, dtype=torch.bool)
    fill_tradability = torch.ones(shape, dtype=torch.bool)
    # Decision 69 legally knows the fill-70 outage. It becomes available again
    # at fill 75; C1/C3 must leave the forced proceeds in cash thereafter.
    decision_tradability[69:75, 0, 2] = False
    fill_tradability[70:75, 0, 2] = False

    returns = torch.zeros((POSITIONS - 1, batch, ASSETS), dtype=dtype)
    returns[..., 0] = 0.0001
    cross_section = (torch.arange(ASSETS - 1, dtype=dtype) - 149.5) * 1e-6
    returns[..., 1:] = cross_section
    mandatory = torch.zeros_like(returns, dtype=torch.bool)
    ordinary = membership[:-1].clone()
    ordinary[..., 0] = False

    caps = torch.full(shape, 0.01, dtype=dtype)
    caps[..., 0] = 1.0
    caps[~fill_tradability] = 0.0
    gross_max = torch.ones((POSITIONS, batch), dtype=dtype)
    cost_rate = torch.full((POSITIONS - 1, batch), 0.002, dtype=dtype)

    rows = POSITIONS - 1
    c1_weights = torch.zeros(shape, dtype=dtype)
    c1_weights[0, :, 1:] = 1.0 / 300.0
    c1_pretrade = torch.zeros((rows, batch, ASSETS), dtype=dtype)
    c1_gross = torch.zeros((rows, batch), dtype=dtype)
    membership_delta = torch.zeros_like(c1_pretrade)
    availability_delta = torch.zeros_like(c1_pretrade)
    risk_delta = torch.zeros_like(c1_pretrade)
    discretionary_delta = torch.zeros_like(c1_pretrade)
    c1_net = torch.zeros((rows, batch), dtype=dtype)
    for row in range(rows):
        current = c1_weights[row]
        holding = (current * returns[row]).sum(dim=-1)
        pretrade = current * (1.0 + returns[row]) / (1.0 + holding).unsqueeze(-1)
        c1_pretrade[row] = pretrade
        c1_gross[row] = holding
        membership_book = _repair_cash(pretrade, membership[row + 1])
        membership_delta[row] = membership_book - pretrade
        availability_book = _repair_cash(membership_book, fill_tradability[row + 1])
        availability_delta[row] = availability_book - membership_book
        # The tiny synthetic dispersion never reaches the common 1% name cap.
        risk_book = availability_book
        risk_delta[row] = risk_book - availability_book
        c1_weights[row + 1] = risk_book
        turnover = 0.5 * (
            membership_delta[row].abs().sum(-1)
            + availability_delta[row].abs().sum(-1)
            + risk_delta[row].abs().sum(-1)
        )
        c1_net[row] = holding - cost_rate[row] * turnover

    decision_state = torch.zeros((*shape, 1), dtype=dtype)
    known_decision = decision_ts.view(-1, 1, 1).expand(shape).clone()
    known_fill = fill_ts.view(-1, 1, 1).expand(shape).clone()
    versions = torch.zeros(shape, dtype=torch.int64)
    absent = torch.full(shape, -1, dtype=torch.int64)
    evidence = Hold30AsOfEvidence(
        decision_membership_known_at_ms=known_decision.clone(),
        decision_tradability_known_at_ms=known_decision.clone(),
        fill_membership_known_at_ms=known_fill.clone(),
        fill_tradability_known_at_ms=known_fill.clone(),
        corporate_action_factor=torch.ones(shape, dtype=dtype),
        corporate_action_version=versions.clone(),
        corporate_action_known_at_ms=absent.clone(),
        identifier_version=versions.clone(),
        identifier_known_at_ms=absent.clone(),
    )
    sequence = Hold30DatasetSequence(
        decision_timestamps_ms=decision_ts,
        fill_timestamps_ms=fill_ts,
        asset_ids=("CASH", *(f"PERM-{index:03d}" for index in range(1, ASSETS))),
        decision_state=decision_state,
        decision_membership=membership.clone(),
        decision_tradability=decision_tradability,
        fill_membership=membership.clone(),
        fill_tradability=fill_tradability,
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
        c1_benchmark_weights=c1_weights,
        c1_benchmark_net_returns=c1_net,
        risk_asset_caps=caps,
        risk_gross_max=gross_max,
        cost_rate=cost_rate,
        asof_evidence=evidence,
        provenance=_provenance(),
    )
    monthly = torch.zeros(POSITIONS, dtype=torch.bool)
    monthly[0] = True
    history_shape = (POSITIONS + HISTORY, batch, ASSETS)
    closes = torch.full(history_shape, 100.0, dtype=dtype)
    history_return = returns[0]
    for position in range(1, POSITIONS + HISTORY):
        closes[position] = closes[position - 1] * (1.0 + history_return)
    close_valid = torch.ones(history_shape, dtype=torch.bool)
    history_times = (
        decision_ts[0]
        + (torch.arange(POSITIONS + HISTORY, dtype=torch.int64) - HISTORY) * DAY_MS
    )
    close_known = history_times.view(-1, 1, 1).expand(history_shape).clone()
    return _Fixture(
        sequence,
        c1_pretrade,
        c1_gross,
        membership_delta,
        availability_delta,
        risk_delta,
        discretionary_delta,
        monthly,
        closes,
        close_valid,
        close_known,
    )


def _accept_c1(fixture: _Fixture):
    return accept_c1_bound_trace(
        fixture.sequence,
        monthly_rebalance=fixture.monthly,
        bound_receipt_sha256=fixture.sequence.provenance.c1_benchmark_trace_sha256,
        outer_start=63,
    )


def test_c0_uses_only_frozen_cash_returns_and_has_no_turnover() -> None:
    fixture = _fixture()
    trace = construct_c0_cash(fixture.sequence, outer_start=63)
    ladder = price_hold30_cost_ladder(trace)

    torch.testing.assert_close(
        trace.gross_returns,
        fixture.sequence.asset_returns[..., fixture.sequence.cash_index],
        rtol=0.0,
        atol=0.0,
    )
    assert not bool((trace.total_turnover != 0).any())
    assert tuple(rung.cost_bps for rung in ladder.rungs) == HOLD30_COST_RUNGS_BPS
    for rung in ladder.rungs:
        torch.testing.assert_close(rung.net_returns, trace.gross_returns)
        torch.testing.assert_close(rung.total_cost, torch.zeros_like(rung.total_cost))


def test_c1_accepts_bound_buy_drift_and_replays_mandatory_repairs_only() -> None:
    fixture = _fixture()
    trace = _accept_c1(fixture)
    ladder = price_hold30_cost_ladder(trace)

    assert trace.control_id == "C1"
    assert trace.turnover_by_cause[TurnoverCause.AVAILABILITY_FORCED][69, 0] > 0
    # Re-derived float64 drift can leave sub-machine-economic residuals.  They
    # remain in the accounting trace and are only tolerance-classified here;
    # the evaluator must never round economic deltas away.
    assert float(trace.turnover_by_cause[TurnoverCause.DISCRETIONARY].max()) < 1e-12
    torch.testing.assert_close(
        ladder.rungs[1].net_returns,
        fixture.sequence.c1_benchmark_net_returns,
        atol=1e-12,
        rtol=1e-12,
    )

    with pytest.raises(Hold30ControlError, match="bound benchmark receipt"):
        accept_c1_bound_trace(
            fixture.sequence,
            monthly_rebalance=fixture.monthly,
            bound_receipt_sha256="9" * 64,
            outer_start=63,
        )


def test_c1_rejects_a_bound_primary_trace_not_priced_at_exactly_20bp() -> None:
    fixture = _fixture()
    wrong_rate = replace(
        fixture.sequence,
        cost_rate=torch.full_like(fixture.sequence.cost_rate, 0.001),
    )
    with pytest.raises(Hold30ControlError, match="exactly 20 bp"):
        accept_c1_bound_trace(
            wrong_rate,
            monthly_rebalance=fixture.monthly,
            bound_receipt_sha256=wrong_rate.provenance.c1_benchmark_trace_sha256,
            outer_start=63,
        )


def test_c2_daily_rebalances_while_c3_forced_proceeds_remain_cash() -> None:
    fixture = _fixture()
    c2 = construct_c2_daily_equal_weight(fixture.sequence, outer_start=63)
    c3 = construct_c3_initial_universe_hold(fixture.sequence, outer_start=63)

    assert c2.turnover_by_cause[TurnoverCause.AVAILABILITY_FORCED][69, 0] > 0
    assert c3.turnover_by_cause[TurnoverCause.AVAILABILITY_FORCED][69, 0] > 0
    assert c2.turnover_by_cause[TurnoverCause.DISCRETIONARY][75, 0] > 0
    assert not bool((c3.turnover_by_cause[TurnoverCause.DISCRETIONARY] != 0).any())
    assert c2.weights[76, 0, 2] > 0
    assert c3.weights[76, 0, 2] == 0
    assert bool(
        (c2.turnover_by_cause[TurnoverCause.DISCRETIONARY] <= 0.10 + 1e-12).all()
    )


def test_cost_ladder_reprices_one_trace_and_separates_all_causes() -> None:
    trace = construct_c2_daily_equal_weight(_fixture().sequence, outer_start=63)
    first = price_hold30_cost_ladder(trace)
    second = price_hold30_cost_ladder(trace)

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.trace_sha256 == trace.trace_sha256
    turnover = trace.total_turnover
    for rung in first.rungs:
        torch.testing.assert_close(
            rung.total_cost,
            turnover * (rung.cost_bps / 10_000.0),
        )
        assert set(rung.costs_by_cause) == set(trace.turnover_by_cause)
        assert rung.continuing_wealth.shape == (POSITIONS, 1)
    torch.testing.assert_close(
        first.rungs[2].total_cost, first.rungs[0].total_cost * 4.0
    )
    assert not bool((trace.turnover_by_cause[TurnoverCause.STARTUP] != 0).any())
    assert not bool((trace.turnover_by_cause[TurnoverCause.TERMINAL] != 0).any())


def test_c4_scores_use_legal_trailing_closes_zscores_and_stable_id_ties() -> None:
    fixture = _fixture()
    closes = fixture.closes.clone()
    # Exact momentum tie; stable ID must deterministically rank PERM-001 first.
    closes[:, 0, 2] = closes[:, 0, 1]
    result = c4_momentum_scores(
        fixture.sequence,
        closes,
        fixture.close_valid,
        fixture.close_known,
    )

    position = 63
    valid_scores = result.values[position].masked_select(result.valid[position])
    assert valid_scores.numel() == 300
    assert float(valid_scores.abs().max()) <= 2.0
    assert result.stable_rank[position, 0, 1] < result.stable_rank[position, 0, 2]
    assert result.values[position, 0, 1] == result.values[position, 0, 2]
    assert bool(result.valid[0, 0, 1:].all())

    future_known = fixture.close_known.clone()
    future_known[HISTORY + 63, 0, 1] = fixture.sequence.decision_timestamps_ms[63] + 1
    with pytest.raises(Hold30ControlError, match="legally available"):
        c4_momentum_scores(
            fixture.sequence,
            closes,
            fixture.close_valid,
            future_known,
        )


def test_c4_uses_canonical_h2_action_path_and_is_deterministic() -> None:
    fixture = _fixture()
    first = construct_c4_momentum(
        fixture.sequence,
        fixture.closes,
        fixture.close_valid,
        fixture.close_known,
        outer_start=63,
    )
    second = construct_c4_momentum(
        fixture.sequence,
        fixture.closes,
        fixture.close_valid,
        fixture.close_known,
        outer_start=63,
    )

    assert first.trace_sha256 == second.trace_sha256
    assert first.control_id == "C4"
    assert bool(
        (first.turnover_by_cause[TurnoverCause.DISCRETIONARY] <= 0.10 + 1e-12).all()
    )
    assert not bool((first.turnover_by_cause[TurnoverCause.STARTUP] != 0).any())
    assert not bool((first.turnover_by_cause[TurnoverCause.TERMINAL] != 0).any())


def test_outer_rows_are_forbidden_from_control_fitting() -> None:
    fixture = _fixture()
    with pytest.raises(Hold30ControlError, match="outer data are forbidden"):
        construct_c0_cash(
            fixture.sequence,
            outer_start=63,
            fitting_rows=(62, 63),
        )
    with pytest.raises(Hold30ControlError, match="first true row"):
        construct_c0_cash(fixture.sequence, outer_start=64)
