from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.datasets.hold30 import (
    HOLD30_CASH_RETURN_RULE,
    HOLD30_UNIVERSE_MODE,
    Hold30AsOfEvidence,
    Hold30DatasetError,
    Hold30DatasetSequence,
    Hold30NullDomain,
    Hold30PointInTimeProvenance,
)
from rl_quant.datasets.hold30_null_rebuild import (
    HOLD30_C1_ACTIVE_COUNT,
    HOLD30_NULL_REBUILDER_VERSION,
    rebuild_hold30_null_outcomes,
)


DAY_MS = 86_400_000
HOUR_MS = 3_600_000
POSITIONS = 187
ASSETS = 302  # CASH + 300 active names + one future monthly replacement.


def _digest(character: str) -> str:
    return character * 64


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
        universe_rule_id="pit-active300-test-v1",
        stable_asset_id_namespace="perm-id-v1",
        benchmark_id="C1",
        cash_asset_id="CASH",
        cash_return_rule=HOLD30_CASH_RETURN_RULE,
    )


def _base_sequence() -> tuple[Hold30DatasetSequence, torch.Tensor]:
    dtype = torch.float64
    batch = 1
    first_decision = 1_735_776_000_000
    decision_ts = first_decision + torch.arange(POSITIONS, dtype=torch.int64) * DAY_MS
    fill_ts = decision_ts - 6 * HOUR_MS
    fill_ts[0] = decision_ts[0] - HOUR_MS
    shape = (POSITIONS, batch, ASSETS)

    decision_membership = torch.zeros(shape, dtype=torch.bool)
    fill_membership = torch.zeros(shape, dtype=torch.bool)
    decision_membership[..., 0] = True
    fill_membership[..., 0] = True
    decision_membership[..., 1:301] = True
    fill_membership[..., 1:301] = True
    # The rotation is known at decision 39 and becomes effective at fill 40.
    decision_membership[39:, 0, 1] = False
    decision_membership[39:, 0, 301] = True
    fill_membership[40:, 0, 1] = False
    fill_membership[40:, 0, 301] = True

    decision_tradability = torch.ones(shape, dtype=torch.bool)
    fill_tradability = torch.ones(shape, dtype=torch.bool)
    # Asset two is forced out at fill 10, becomes tradeable at fill 15, and is
    # not bought again until the scheduled reconstitution at fill 20.
    decision_tradability[9:15, 0, 2] = False
    fill_tradability[10:15, 0, 2] = False

    decision_state = torch.zeros((*shape, 1), dtype=dtype)
    returns = torch.zeros((POSITIONS - 1, batch, ASSETS), dtype=dtype)
    returns[..., 0] = 0.0001
    risky_levels = torch.arange(1, ASSETS, dtype=dtype) * 1e-6
    returns[..., 1:] = risky_levels
    mandatory = torch.zeros_like(returns, dtype=torch.bool)
    # Asset one earns its delisting outcome before the effective fill-40
    # membership repair. The null transform must leave this cell fixed.
    mandatory[39, 0, 1] = True
    returns[39, 0, 1] = -0.40
    ordinary = fill_membership[:-1].clone()
    ordinary[..., 0] = False
    ordinary &= ~mandatory

    fill_trade = fill_membership & fill_tradability
    caps = torch.zeros(shape, dtype=dtype)
    caps[..., 0] = 1.0
    caps[..., 1:] = torch.where(
        fill_trade[..., 1:],
        torch.full_like(caps[..., 1:], 0.01),
        torch.zeros_like(caps[..., 1:]),
    )
    gross = torch.ones((POSITIONS, batch), dtype=dtype)
    costs = torch.full((POSITIONS - 1, batch), 0.002, dtype=dtype)

    # Deliberately stale cash-only benchmark fields. The rebuilder must not
    # copy these into its transformed sequence.
    stale_c1 = torch.zeros(shape, dtype=dtype)
    stale_c1[..., 0] = 1.0
    stale_c1_return = returns[..., 0].clone()

    decision_known = decision_ts.view(-1, 1, 1).expand(shape).clone()
    fill_known = fill_ts.view(-1, 1, 1).expand(shape).clone()
    versions = torch.zeros(shape, dtype=torch.int64)
    absent = torch.full(shape, -1, dtype=torch.int64)
    evidence = Hold30AsOfEvidence(
        decision_membership_known_at_ms=decision_known.clone(),
        decision_tradability_known_at_ms=decision_known.clone(),
        fill_membership_known_at_ms=fill_known.clone(),
        fill_tradability_known_at_ms=fill_known.clone(),
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
        decision_membership=decision_membership,
        decision_tradability=decision_tradability,
        fill_membership=fill_membership,
        fill_tradability=fill_tradability,
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
        c1_benchmark_weights=stale_c1,
        c1_benchmark_net_returns=stale_c1_return,
        risk_asset_caps=caps,
        risk_gross_max=gross,
        cost_rate=costs,
        asof_evidence=evidence,
        provenance=_provenance(),
    )
    monthly = torch.zeros(POSITIONS, dtype=torch.bool)
    monthly[::20] = True
    monthly[40] = True
    return sequence, monthly


def _domains() -> tuple[Hold30NullDomain, ...]:
    return (
        Hold30NullDomain("train", 0, 62),
        Hold30NullDomain("validation", 62, 124),
        Hold30NullDomain("outer", 124, 186),
    )


def _rebuilt():
    base, monthly = _base_sequence()
    view = base.n_xs(29, domains=_domains())
    return base, view, monthly, rebuild_hold30_null_outcomes(
        base,
        view,
        monthly_rebalance=monthly,
    )


def test_rebuild_is_deterministic_replaces_stale_economics_and_preserves_state() -> None:
    base, view, monthly, first = _rebuilt()
    second = rebuild_hold30_null_outcomes(base, view, monthly_rebalance=monthly)

    assert first.receipt == second.receipt
    torch.testing.assert_close(first.c1.weights, second.c1.weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.c5.values, second.c5.values, rtol=0.0, atol=0.0)
    assert first.receipt.builder_version == HOLD30_NULL_REBUILDER_VERSION
    assert first.receipt.source_axis_id == base.axis_id
    assert first.receipt.null_transform_id == view.receipt.transform_id
    assert first.receipt.c1_trace_sha256 == first.c1.trace_sha256
    assert first.receipt.c5_labels_sha256 == first.c5.labels_sha256
    assert first.sequence.provenance.receipt_id == (
        first.receipt.transformed_provenance_receipt_id
    )
    assert first.sequence.axis_id == first.receipt.transformed_axis_id

    # The actor and every legal-date fact are reused exactly; only outcomes and
    # their dependent benchmark/labels/provenance are rebuilt.
    assert first.sequence.decision_state is base.decision_state
    assert first.sequence.fill_membership is base.fill_membership
    assert first.sequence.fill_tradability is base.fill_tradability
    assert first.sequence.risk_asset_caps is base.risk_asset_caps
    assert first.sequence.cost_rate is base.cost_rate
    assert first.sequence.mandatory_return_mask is base.mandatory_return_mask
    assert first.sequence.asset_returns is view.asset_returns
    assert not torch.equal(first.sequence.c1_benchmark_weights, base.c1_benchmark_weights)
    assert not torch.equal(first.sequence.c1_benchmark_net_returns, base.c1_benchmark_net_returns)
    assert first.sequence.provenance.receipt_id != base.provenance.receipt_id


def test_c1_buy_drift_forced_exit_monthly_rebalance_and_cost_chronology() -> None:
    _base, _view, _monthly, rebuilt = _rebuilt()
    c1 = rebuilt.c1

    expected = 1.0 / HOLD30_C1_ACTIVE_COUNT
    assert c1.weights[0, 0, 1].item() == pytest.approx(expected)
    assert c1.weights[0, 0, 300].item() == pytest.approx(expected)
    assert c1.weights[0, 0, 301].item() == 0.0

    # Availability repair is charged on row 9 at fill 10. Re-availability is
    # not a buy event; the next frozen monthly event at fill 20 restores it.
    assert c1.weights[10, 0, 2].item() == 0.0
    assert c1.availability_turnover[9, 0].item() > 0.0
    assert c1.costs[9, 0].item() > 0.0
    assert c1.weights[15, 0, 2].item() == 0.0
    assert c1.weights[19, 0, 2].item() == 0.0
    assert c1.weights[20, 0, 2].item() == pytest.approx(expected)
    assert c1.scheduled_turnover[19, 0].item() > 0.0

    # The fill-40 rotation first forces the deletion, then the scheduled event
    # buys the newly active coordinate under the same row's common cost.
    assert c1.weights[40, 0, 1].item() == 0.0
    assert c1.weights[40, 0, 301].item() == pytest.approx(expected)
    assert c1.membership_turnover[39, 0].item() > 0.0
    assert c1.scheduled_turnover[39, 0].item() > 0.0
    torch.testing.assert_close(
        c1.net_returns,
        c1.holding_returns - c1.costs,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        c1.costs,
        rebuilt.sequence.cost_rate * c1.total_turnover,
        rtol=0.0,
        atol=0.0,
    )


def test_c5_uses_post_fill_30_return_path_forced_cash_and_domain_censoring() -> None:
    base, _view, _monthly, rebuilt = _rebuilt()
    c5 = rebuilt.c5

    # Origin 8 legally fills at 9. Asset two earns return row 9, is forced to
    # cash at fill 10, then earns CASH on rows 10..38. No future re-entry is
    # permitted inside the buy-and-hold label path.
    origin = 8
    asset = 2
    stock_log = torch.log1p(rebuilt.sequence.asset_returns[9, 0, asset])
    stock_log = stock_log + torch.log1p(
        rebuilt.sequence.asset_returns[10:39, 0, base.cash_index]
    ).sum()
    benchmark_log = torch.log1p(rebuilt.c1.net_returns[9:39, 0]).sum()
    assert bool(c5.valid[origin, 0, asset])
    assert not bool(c5.censored[origin, 0, asset])
    assert c5.values[origin, 0, asset].item() == pytest.approx(
        (stock_log - benchmark_log).item()
    )

    # Origin 31 has support through row 61 and is valid; origin 32 would cross
    # from train into validation and is explicitly right-censored.
    assert bool(c5.valid[31, 0, 3])
    assert not bool(c5.censored[31, 0, 3])
    assert not bool(c5.valid[32, 0, 3])
    assert bool(c5.censored[32, 0, 3])
    assert not bool((c5.valid & c5.censored).any())
    assert not bool(c5.valid[..., base.cash_index].any())


def test_tampered_null_receipt_or_unmarked_membership_event_fails_closed() -> None:
    base, monthly = _base_sequence()
    view = base.n_xs(29, domains=_domains())

    changed_returns = view.asset_returns.clone()
    changed_returns[0, 0, 1] += 0.001
    with pytest.raises(Hold30DatasetError, match="transformed outcomes"):
        rebuild_hold30_null_outcomes(
            base,
            replace(view, asset_returns=changed_returns),
            monthly_rebalance=monthly,
        )

    changed_mapping = view.source_index.clone()
    changed_mapping[0, 0, 1] = 1
    with pytest.raises(Hold30DatasetError, match="source mapping"):
        rebuild_hold30_null_outcomes(
            base,
            replace(view, source_index=changed_mapping),
            monthly_rebalance=monthly,
        )

    unmarked = monthly.clone()
    unmarked[40] = False
    with pytest.raises(Hold30DatasetError, match="outside the frozen monthly"):
        rebuild_hold30_null_outcomes(
            base,
            view,
            monthly_rebalance=unmarked,
        )
