from __future__ import annotations

import pytest
import torch

from rl_quant.envs import (
    HistoricalMarketData,
    PortfolioConstraints,
    PortfolioCostModel,
    VectorPortfolioEnv,
)
from rl_quant.execution import (
    ImmediateTargetWeightExecution,
    TargetWeightExecutionResult,
)
from rl_quant.rl import ActionBatch, ObservationBatch, VectorEnvironment


def _market(
    returns: torch.Tensor,
    *,
    availability: torch.Tensor | None = None,
    signal: torch.Tensor | None = None,
) -> HistoricalMarketData:
    batch_size, horizon, num_assets = returns.shape
    if availability is None:
        availability = torch.ones((batch_size, horizon + 1, num_assets), dtype=torch.bool)
    if signal is None:
        signal = torch.zeros((batch_size, horizon + 1, num_assets - 1))
    return HistoricalMarketData(
        features={"signal": signal},
        asset_returns=returns,
        availability=availability,
    )


def test_planted_signal_policy_grows_equity_chronologically() -> None:
    # At every state, the positive signal names the asset with +10% next return.
    returns = torch.tensor(
        [[[0.0, 0.10, -0.10], [0.0, -0.10, 0.10], [0.0, 0.10, -0.10]]]
    )
    signal = torch.tensor(
        [[[[1.0, -1.0]], [[-1.0, 1.0]], [[1.0, -1.0]], [[0.0, 0.0]]]]
    ).squeeze(2)
    env = VectorPortfolioEnv(_market(returns, signal=signal), discount=0.97)
    observation, _ = env.reset()
    for _step in range(3):
        selected = observation.tensors["signal"].argmax(dim=-1) + 1
        action = torch.zeros((1, 3))
        action.scatter_(1, selected.unsqueeze(-1), 1.0)
        transition = env.step(ActionBatch(action=action, log_prob=torch.zeros(1)))
        observation = transition.next_observation

    torch.testing.assert_close(env.equity, torch.tensor([1.1**3]))
    assert transition.terminated.tolist() == [True]
    assert transition.truncated.tolist() == [False]
    assert transition.discount.tolist() == [0.0]
    assert env.weights.tolist() == [[1.0, 0.0, 0.0]]


def test_environment_steps_independent_episodes_as_one_tensor_batch() -> None:
    returns = torch.tensor([[[0.0, 0.10]], [[0.0, -0.20]]])
    env = VectorPortfolioEnv(_market(returns))
    assert isinstance(env, VectorEnvironment)
    observation, _ = env.reset()
    assert observation.batch_size == 2
    transition = env.step(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    torch.testing.assert_close(transition.reward, torch.tensor([0.10, 0.0]))
    torch.testing.assert_close(env.equity, torch.tensor([1.10, 1.0]))
    assert transition.terminated.tolist() == [True, True]


def test_execution_drift_and_terminal_liquidation_share_one_ledger() -> None:
    returns = torch.tensor([[[0.0, 0.10], [0.0, 0.0]]])
    env = VectorPortfolioEnv(
        _market(returns),
        costs=PortfolioCostModel(spread_bps=10.0),
    )
    env.reset()
    stock = torch.tensor([[0.0, 1.0]])

    first = env.step(stock)
    torch.testing.assert_close(first.rewards.gross_return, torch.tensor([0.10]))
    torch.testing.assert_close(first.rewards.execution_cost, torch.tensor([0.001]))
    torch.testing.assert_close(first.reward, torch.tensor([0.099]))
    torch.testing.assert_close(env.equity, torch.tensor([1.099]))

    final = env.step(stock)
    assert final.terminated.item()
    torch.testing.assert_close(final.rewards.execution_cost, torch.tensor([0.0]))
    torch.testing.assert_close(final.rewards.liquidation_cost, torch.tensor([0.001]))
    torch.testing.assert_close(final.reward, torch.tensor([-0.001]))
    torch.testing.assert_close(env.equity, torch.tensor([1.099 * 0.999]))
    assert env.weights.tolist() == [[1.0, 0.0]]
    with pytest.raises(RuntimeError, match="done"):
        env.step(stock)


def test_terminal_liquidation_cost_is_scaled_from_post_return_equity() -> None:
    env = VectorPortfolioEnv(
        _market(torch.tensor([[[0.0, 0.50]]])),
        costs=PortfolioCostModel(spread_bps=10.0),
    )
    env.reset()

    transition = env.step(torch.tensor([[0.0, 1.0]]))

    # Entry cost is 0.1% of starting equity. Liquidation is 0.1% of
    # post-return/post-entry equity: (1 + .50 - .001) * (1 - .001).
    expected_liquidation = torch.tensor([0.001499])
    expected_equity = torch.tensor([1.499 * 0.999])
    torch.testing.assert_close(transition.rewards.liquidation_cost, expected_liquidation)
    torch.testing.assert_close(transition.info["liquidation_cost_return_units"], expected_liquidation)
    torch.testing.assert_close(env.equity, expected_equity)


def test_holdings_drift_after_asset_returns() -> None:
    returns = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    env = VectorPortfolioEnv(_market(returns))
    env.reset()
    transition = env.step(torch.tensor([[0.5, 0.5]]))

    torch.testing.assert_close(transition.reward, torch.tensor([0.5]))
    torch.testing.assert_close(env.weights, torch.tensor([[1.0 / 3.0, 2.0 / 3.0]]))


def test_projection_enforces_availability_weight_leverage_and_turnover() -> None:
    returns = torch.zeros((1, 2, 4))
    availability = torch.ones((1, 3, 4), dtype=torch.bool)
    availability[:, 0, 1] = False
    env = VectorPortfolioEnv(
        _market(returns, availability=availability),
        constraints=PortfolioConstraints(max_asset_weight=0.25, max_leverage=0.5),
    )
    env.reset()
    transition = env.step(torch.tensor([[0.0, 0.6, 0.3, 0.1]]))
    torch.testing.assert_close(transition.executed_action, torch.tensor([[0.65, 0.0, 0.25, 0.10]]))
    assert transition.info["gross_exposure"].item() <= 0.5

    capped = VectorPortfolioEnv(
        _market(torch.zeros((1, 2, 2))),
        constraints=PortfolioConstraints(max_turnover=0.2),
    )
    capped.reset()
    limited = capped.step(torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(limited.executed_action, torch.tensor([[0.8, 0.2]]))
    torch.testing.assert_close(limited.info["one_way_turnover"], torch.tensor([0.2]))


def test_truncation_bootstraps_and_does_not_force_liquidation() -> None:
    env = VectorPortfolioEnv(
        _market(torch.zeros((1, 3, 2))),
        costs=PortfolioCostModel(spread_bps=10.0),
        discount=0.95,
        max_episode_steps=2,
    )
    env.reset()
    stock = torch.tensor([[0.0, 1.0]])
    env.step(stock)
    transition = env.step(stock)

    assert transition.truncated.tolist() == [True]
    assert transition.terminated.tolist() == [False]
    torch.testing.assert_close(transition.discount, torch.tensor([0.95]))
    torch.testing.assert_close(transition.rewards.liquidation_cost, torch.tensor([0.0]))
    torch.testing.assert_close(env.weights, stock)


def test_hard_availability_can_override_an_impossible_turnover_cap() -> None:
    returns = torch.zeros((1, 3, 2))
    returns[:, 0, 1] = 1.0
    availability = torch.ones((1, 4, 2), dtype=torch.bool)
    availability[:, 1, 1] = False
    env = VectorPortfolioEnv(
        _market(returns, availability=availability),
        constraints=PortfolioConstraints(max_turnover=0.1),
    )
    env.reset()
    # The first target is turnover-capped at 10% stock, which then appreciates
    # to 18.18% of the portfolio before becoming unavailable.
    env.step(torch.tensor([[0.0, 1.0]]))
    # At the next state the held stock is unavailable, so it must be liquidated.
    transition = env.step(torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(transition.executed_action, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(transition.info["forced_turnover"], torch.tensor([2.0 / 11.0]))
    torch.testing.assert_close(transition.info["forced_turnover_excess"], torch.tensor([2.0 / 11.0 - 0.1]))


def test_market_data_and_cash_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="greater than -1"):
        _market(torch.tensor([[[0.0, -1.0]]]))

    data = _market(torch.tensor([[[0.01, 0.0]]]))
    with pytest.raises(ValueError, match="CASH return"):
        VectorPortfolioEnv(data)


def test_decision_ids_survive_the_environment_transition_for_exact_alignment() -> None:
    data = HistoricalMarketData(
        features={"signal": torch.zeros(1, 3, 1)},
        asset_returns=torch.zeros(1, 2, 2),
        availability=torch.ones(1, 3, 2, dtype=torch.bool),
        decision_ids=torch.tensor([[2026010201, 2026010301]]),
    )
    env = VectorPortfolioEnv(data)
    env.reset()
    first = env.step(torch.tensor([[1.0, 0.0]]))
    assert first.info["decision_id"].tolist() == [2026010201]
    assert first.info["decision_index"].tolist() == [0]
    assert first.info["environment_index"].tolist() == [0]

    with pytest.raises(ValueError, match="unique"):
        HistoricalMarketData(
            features={"signal": torch.zeros(1, 3, 1)},
            asset_returns=torch.zeros(1, 2, 2),
            availability=torch.ones(1, 3, 2, dtype=torch.bool),
            decision_ids=torch.tensor([[7, 7]]),
        )


def test_max_drawdown_latches_hard_cash_fallback_and_exposes_risk_state() -> None:
    returns = torch.tensor([[[0.0, -0.20], [0.0, 0.50], [0.0, 0.50]]])
    env = VectorPortfolioEnv(
        _market(returns),
        constraints=PortfolioConstraints(max_drawdown=0.10),
        costs=PortfolioCostModel(spread_bps=10.0),
    )
    observation, reset_info = env.reset()
    assert not reset_info["risk_halted"].item()
    expected_state = {
        "portfolio_peak_equity",
        "portfolio_drawdown",
        "portfolio_recent_turnover",
        "portfolio_gross_exposure",
        "constraint_max_asset_weight",
        "constraint_max_gross_exposure",
        "constraint_max_turnover",
        "constraint_max_drawdown",
        "constraint_risk_halted",
        "constraint_valid_action_fraction",
    }
    assert expected_state.issubset(observation.tensors)

    stock = torch.tensor([[0.0, 1.0]])
    breached = env.step(stock)
    assert breached.info["risk_halt_triggered"].tolist() == [True]
    assert env.risk_halted.tolist() == [True]
    torch.testing.assert_close(env.weights, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(env.peak_equity, torch.tensor([1.0]))
    torch.testing.assert_close(breached.rewards.execution_cost, torch.tensor([0.001]))
    torch.testing.assert_close(breached.rewards.liquidation_cost, torch.tensor([0.000799]))
    torch.testing.assert_close(breached.reward, torch.tensor([-0.201799]))
    torch.testing.assert_close(env.drawdown, torch.tensor([0.201799]))
    torch.testing.assert_close(env.recent_turnover, torch.tensor([2.0]))
    torch.testing.assert_close(
        breached.next_observation.tensors["portfolio_drawdown"],
        torch.tensor([[0.201799]]),
    )
    assert breached.next_observation.action_mask is not None
    assert breached.next_observation.action_mask.tolist() == [[True, False]]

    # A malicious/nonconforming policy can ignore the observation mask, but the
    # environment-owned hard halt still makes CASH the only executed target.
    bypass_attempt = env.step(stock)
    torch.testing.assert_close(bypass_attempt.executed_action, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(bypass_attempt.reward, torch.tensor([0.0]))
    assert bypass_attempt.info["risk_override"].tolist() == [True]
    assert env.risk_halted.tolist() == [True]
    torch.testing.assert_close(env.weights, torch.tensor([[1.0, 0.0]]))


class _AuthoritativeCostModel:
    def __init__(self) -> None:
        self.targets: list[torch.Tensor] = []

    def execute(
        self,
        current_weights: torch.Tensor,
        target_weights: torch.Tensor,
        *,
        cash_index: int,
    ) -> TargetWeightExecutionResult:
        del cash_index
        self.targets.append(target_weights.clone())
        changed = ~torch.eq(current_weights, target_weights).all(dim=-1)
        execution_cost = torch.where(
            changed,
            torch.full_like(changed, 0.02, dtype=target_weights.dtype),
            torch.zeros_like(changed, dtype=target_weights.dtype),
        )
        modeled_impact_cost = torch.where(
            changed,
            torch.full_like(changed, 0.005, dtype=target_weights.dtype),
            torch.zeros_like(changed, dtype=target_weights.dtype),
        )
        traded = (target_weights - current_weights).abs()
        traded[:, 0] = 0.0
        return TargetWeightExecutionResult(
            execution_cost=execution_cost,
            modeled_impact_cost=modeled_impact_cost,
            traded_notional=traded.sum(dim=-1),
            diagnostics={"authority_marker": torch.ones_like(execution_cost)},
        )


def test_environment_projects_before_execution_and_uses_execution_costs_authoritatively() -> None:
    returns = torch.zeros((1, 2, 4))
    availability = torch.ones((1, 3, 4), dtype=torch.bool)
    availability[:, 0, 1] = False
    execution_model = _AuthoritativeCostModel()
    env = VectorPortfolioEnv(
        _market(returns, availability=availability),
        constraints=PortfolioConstraints(max_asset_weight=0.25, max_leverage=0.5),
        execution_model=execution_model,
    )
    env.reset()
    transition = env.step(torch.tensor([[0.0, 0.6, 0.3, 0.1]]))

    feasible = torch.tensor([[0.65, 0.0, 0.25, 0.10]])
    torch.testing.assert_close(execution_model.targets[0], feasible)
    torch.testing.assert_close(transition.executed_action, feasible)
    torch.testing.assert_close(transition.rewards.execution_cost, torch.tensor([0.02]))
    torch.testing.assert_close(transition.rewards.impact_cost, torch.tensor([0.005]))
    torch.testing.assert_close(transition.reward, torch.tensor([-0.025]))
    torch.testing.assert_close(env.equity, torch.tensor([0.975]))
    torch.testing.assert_close(transition.info["execution_authority_marker"], torch.tensor([1.0]))
    assert transition.info["gross_exposure"].item() <= 0.5

    with pytest.raises(ValueError, match="either costs or execution_model"):
        VectorPortfolioEnv(
            _market(returns),
            costs=PortfolioCostModel(spread_bps=1.0),
            execution_model=execution_model,
        )


class _LegacyObservationAdapter:
    """Old adapter signature: it deliberately knows nothing about risk state."""

    def build(
        self,
        data: HistoricalMarketData,
        *,
        time_index: int,
        weights: torch.Tensor,
        equity: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> ObservationBatch:
        del weights, equity
        return ObservationBatch(
            tensors={"legacy_signal": data.features["signal"][:, time_index]},
            episode_start=episode_start,
        )


def test_environment_enriches_existing_observation_adapters_without_signature_changes() -> None:
    env = VectorPortfolioEnv(
        _market(torch.zeros((1, 2, 2))),
        observation_adapter=_LegacyObservationAdapter(),
    )
    observation, _ = env.reset()

    assert "legacy_signal" in observation.tensors
    assert "portfolio_peak_equity" in observation.tensors
    assert "portfolio_recent_turnover" in observation.tensors
    assert "constraint_risk_halted" in observation.tensors
    assert observation.action_mask is not None
    assert observation.action_mask.tolist() == [[True, True]]
    assert ImmediateTargetWeightExecution.models_market_fills is False
