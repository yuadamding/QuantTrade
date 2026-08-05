from __future__ import annotations

import torch

from rl_quant.envs import (
    CohortLedger,
    HistoricalMarketData,
    PortfolioCostModel,
    TurnoverCause,
    VectorPortfolioEnv,
    net_trade_legs,
)


def _market(
    returns: torch.Tensor,
    *,
    availability: torch.Tensor | None = None,
) -> HistoricalMarketData:
    batch_size, horizon, num_assets = returns.shape
    if availability is None:
        availability = torch.ones(
            (batch_size, horizon + 1, num_assets), dtype=torch.bool
        )
    return HistoricalMarketData(
        features={"signal": torch.zeros((batch_size, horizon + 1, num_assets - 1))},
        asset_returns=returns,
        availability=availability,
    )


def test_economic_and_return_neutral_cohorts_conserve_through_drift_and_sale() -> None:
    ledger = CohortLedger.from_weights(
        torch.tensor([[0.5, 0.5]]),
        cash_index=0,
        initial_age=0,
        track_initial_units=True,
    )

    drifted = ledger.age_and_drift(torch.tensor([[0.0, 1.0]]))

    # Economic value doubles and normalizes to 2/3 of the portfolio.  The
    # return-neutral entry units remain the original 1/2 notional.
    torch.testing.assert_close(drifted.weights, torch.tensor([[1.0 / 3.0, 2.0 / 3.0]]))
    torch.testing.assert_close(drifted.economic_value[0, 1, 1], torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(drifted.retention_units[0, 1, 1], torch.tensor(0.5))

    reduced, accounting = drifted.trade_to(
        torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]),
        cause=TurnoverCause.DISCRETIONARY,
    )

    # Selling half the current cohort removes half its return-neutral units.
    torch.testing.assert_close(
        accounting.sold_value_by_age[0, 1, 1], torch.tensor(1.0 / 3.0)
    )
    torch.testing.assert_close(
        accounting.sold_units_by_age[0, 1, 1], torch.tensor(0.25)
    )
    torch.testing.assert_close(reduced.retention_units[0, 1, 1], torch.tensor(0.25))
    reduced.assert_reconciles(torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]))


def test_age_59_advances_into_a_stable_60_plus_bin() -> None:
    ledger = CohortLedger.from_weights(
        torch.tensor([[0.0, 1.0]]),
        cash_index=0,
        initial_age=59,
        track_initial_units=True,
    )

    age_60 = ledger.age_and_drift(torch.zeros((1, 2)))
    age_61 = age_60.age_and_drift(torch.zeros((1, 2)))

    torch.testing.assert_close(age_60.economic_value[0, 1, 60], torch.tensor(1.0))
    torch.testing.assert_close(age_61.economic_value[0, 1, 60], torch.tensor(1.0))
    torch.testing.assert_close(age_61.retention_units[0, 1, 60], torch.tensor(1.0))
    assert torch.count_nonzero(age_61.economic_value[..., :60]).item() == 0


def test_common_endowment_is_evenly_staggered_over_ages_zero_through_29() -> None:
    ledger = CohortLedger.from_staggered_endowment(
        torch.tensor([[0.7, 0.3]]),
        cash_index=0,
    )
    torch.testing.assert_close(
        ledger.economic_value[0, 1, :30],
        torch.full((30,), 0.01),
    )
    torch.testing.assert_close(ledger.economic_value[..., 30:], torch.zeros(1, 2, 31))
    torch.testing.assert_close(ledger.retention_units, torch.zeros_like(ledger.retention_units))
    ledger.assert_reconciles(torch.tensor([[0.7, 0.3]]))


def test_same_name_buy_and_sell_are_netted_without_resetting_existing_age() -> None:
    ledger = CohortLedger.from_weights(
        torch.tensor([[0.6, 0.4]]),
        cash_index=0,
        initial_age=20,
        track_initial_units=True,
    )
    gross_buys = torch.tensor([[0.0, 0.3]])
    gross_sells = torch.tensor([[0.0, 0.2]])
    net_buys, net_sells = net_trade_legs(gross_buys, gross_sells)
    target = ledger.weights + net_buys - net_sells
    target[:, 0] = 1.0 - target[:, 1]

    updated, accounting = ledger.trade_to(target, cause=TurnoverCause.DISCRETIONARY)

    # Only the net +10% purchase is new.  The existing 40% position keeps age 20.
    torch.testing.assert_close(updated.economic_value[0, 1, 20], torch.tensor(0.4))
    torch.testing.assert_close(updated.economic_value[0, 1, 0], torch.tensor(0.1))
    torch.testing.assert_close(updated.retention_units[0, 1, 20], torch.tensor(0.4))
    torch.testing.assert_close(accounting.entry_units_added, torch.tensor([[0.0, 0.1]]))
    torch.testing.assert_close(
        accounting.net_sells, torch.zeros_like(accounting.net_sells)
    )


def test_forced_sales_are_exempt_from_early_exit_accounting() -> None:
    young = CohortLedger.from_weights(
        torch.tensor([[0.5, 0.5]]),
        cash_index=0,
        initial_age=5,
        track_initial_units=True,
    )
    target = torch.tensor([[1.0, 0.0]])

    _discretionary, ordinary = young.trade_to(target, cause=TurnoverCause.DISCRETIONARY)
    _forced, unavailable = young.trade_to(
        target, cause=TurnoverCause.AVAILABILITY_FORCED
    )
    _risk, risk = young.trade_to(target, cause=TurnoverCause.RISK_FORCED)

    assert ordinary.early_exit_notional.item() > 0
    assert ordinary.early_exit_units.item() > 0
    torch.testing.assert_close(unavailable.early_exit_notional, torch.zeros(1))
    torch.testing.assert_close(unavailable.early_exit_units, torch.zeros(1))
    torch.testing.assert_close(risk.early_exit_notional, torch.zeros(1))
    assert unavailable.early_exit_exempt
    assert risk.early_exit_exempt


def test_hazard_release_is_consumed_before_proportional_residual_sale() -> None:
    value = torch.zeros((1, 2, 61))
    units = torch.zeros_like(value)
    value[0, 1, 5] = 0.25
    value[0, 1, 40] = 0.25
    units.copy_(value)
    ledger = CohortLedger(value, units, cash_index=0)
    proposed = torch.zeros_like(value)
    proposed[0, 1, 5] = 0.10

    updated, accounting = ledger.trade_to(
        torch.tensor([[0.7, 0.3]]),
        cause=TurnoverCause.DISCRETIONARY,
        proposed_release=proposed,
    )

    # Consume the 10% proposed young release first.  The remaining 10% sale is
    # split pro rata over the 15% young and 25% old residual cohorts.
    torch.testing.assert_close(
        accounting.sold_value_by_age[0, 1, 5], torch.tensor(0.1375)
    )
    torch.testing.assert_close(
        accounting.sold_value_by_age[0, 1, 40], torch.tensor(0.0625)
    )
    torch.testing.assert_close(updated.weights, torch.tensor([[0.7, 0.3]]))
    torch.testing.assert_close(
        accounting.sold_units_by_age, accounting.sold_value_by_age
    )


def test_environment_classifies_availability_forced_turnover_separately() -> None:
    returns = torch.zeros((1, 2, 2))
    availability = torch.ones((1, 3, 2), dtype=torch.bool)
    availability[:, 1, 1] = False
    env = VectorPortfolioEnv(
        _market(returns, availability=availability),
        initial_weights=torch.tensor([0.0, 1.0]),
        terminal_liquidate=False,
    )
    env.reset()

    env.step(torch.tensor([[0.0, 1.0]]))
    forced = env.step(torch.tensor([[1.0, 0.0]]))

    torch.testing.assert_close(
        forced.info["turnover_availability_forced"], torch.ones(1)
    )
    torch.testing.assert_close(forced.info["turnover_discretionary"], torch.zeros(1))
    torch.testing.assert_close(forced.info["turnover_risk_forced"], torch.zeros(1))
    torch.testing.assert_close(forced.info["early_exit_notional"], torch.zeros(1))
    torch.testing.assert_close(
        forced.info["turnover_reconciliation_error"], torch.zeros(1)
    )
    env.age_ledger.assert_reconciles(env.weights)


def test_checkpoint_restore_carries_book_age_units_and_turnover_exactly() -> None:
    data = _market(torch.tensor([[[0.0, 0.10], [0.0, -0.05], [0.0, 0.02]]]))
    initial = torch.tensor([0.5, 0.5])
    uninterrupted = VectorPortfolioEnv(
        data,
        initial_weights=initial,
        initial_position_age=7,
        track_initial_cohort_units=True,
        terminal_liquidate=False,
    )
    resumed = VectorPortfolioEnv(
        data,
        initial_weights=initial,
        initial_position_age=7,
        track_initial_cohort_units=True,
        terminal_liquidate=False,
    )
    uninterrupted.reset()
    resumed.reset()
    uninterrupted.step(torch.tensor([[0.4, 0.6]]))
    checkpoint = uninterrupted.capture_state()

    resumed.restore_state(checkpoint)
    expected = uninterrupted.step(torch.tensor([[0.6, 0.4]]))
    actual = resumed.step(torch.tensor([[0.6, 0.4]]))

    torch.testing.assert_close(actual.reward, expected.reward)
    torch.testing.assert_close(resumed.weights, uninterrupted.weights)
    torch.testing.assert_close(resumed.equity, uninterrupted.equity)
    torch.testing.assert_close(
        resumed.age_ledger.economic_value,
        uninterrupted.age_ledger.economic_value,
    )
    torch.testing.assert_close(
        resumed.age_ledger.retention_units,
        uninterrupted.age_ledger.retention_units,
    )
    for cause in TurnoverCause:
        torch.testing.assert_close(
            resumed.cumulative_turnover_by_cause[cause],
            uninterrupted.cumulative_turnover_by_cause[cause],
        )


def test_continuing_terminal_does_not_liquidate_or_reset_cohorts() -> None:
    env = VectorPortfolioEnv(
        _market(torch.zeros((1, 1, 2))),
        costs=PortfolioCostModel(spread_bps=10.0),
        initial_weights=torch.tensor([0.0, 1.0]),
        initial_position_age=12,
        track_initial_cohort_units=True,
        terminal_liquidate=False,
    )
    env.reset()

    terminal = env.step(torch.tensor([[0.0, 1.0]]))

    assert terminal.terminated.item()
    torch.testing.assert_close(terminal.rewards.liquidation_cost, torch.zeros(1))
    torch.testing.assert_close(terminal.info["turnover_terminal"], torch.zeros(1))
    torch.testing.assert_close(env.weights, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(
        env.age_ledger.economic_value[0, 1, 13], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        env.age_ledger.retention_units[0, 1, 13], torch.tensor(1.0)
    )


def test_truncation_carries_cohorts_without_artificial_liquidation() -> None:
    env = VectorPortfolioEnv(
        _market(torch.zeros((1, 3, 2))),
        initial_weights=torch.tensor([0.0, 1.0]),
        initial_position_age=3,
        track_initial_cohort_units=True,
        max_episode_steps=1,
    )
    env.reset()

    truncated = env.step(torch.tensor([[0.0, 1.0]]))

    assert truncated.truncated.item()
    assert not truncated.terminated.item()
    torch.testing.assert_close(truncated.info["turnover_terminal"], torch.zeros(1))
    torch.testing.assert_close(env.weights, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(
        env.age_ledger.economic_value[0, 1, 4], torch.tensor(1.0)
    )


def test_cohort_trade_remains_differentiable() -> None:
    ledger = CohortLedger.from_weights(torch.tensor([[0.5, 0.5]]), cash_index=0)
    logits = torch.tensor([[0.0, 0.2]], requires_grad=True)
    target = torch.softmax(logits, dim=-1)

    updated, accounting = ledger.trade_to(target, cause=TurnoverCause.DISCRETIONARY)
    objective = (
        updated.economic_value[..., 0].sum() - accounting.early_exit_notional.sum()
    )
    objective.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0
