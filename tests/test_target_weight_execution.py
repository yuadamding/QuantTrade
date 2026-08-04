from __future__ import annotations

import pytest
import torch

from rl_quant.envs import HistoricalMarketData, VectorPortfolioEnv
from rl_quant.execution import (
    FixedTurnoverTargetWeightExecution,
    TargetWeightExecutionModel,
    fixed_turnover_cost,
    one_way_turnover,
)


def test_fixed_turnover_execution_matches_canonical_legacy_cost_basis() -> None:
    current = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.3, 0.5],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.4, 0.2, 0.4],
        ],
        dtype=torch.float64,
    )
    model = FixedTurnoverTargetWeightExecution(cost_bps=12.5)

    result = model.execute(current, target, cash_index=0)
    turnover = one_way_turnover(target, current)

    assert isinstance(model, TargetWeightExecutionModel)
    assert model.models_market_fills is False
    torch.testing.assert_close(turnover, torch.tensor([1.0, 1.0, 0.2], dtype=torch.float64))
    torch.testing.assert_close(result.traded_notional, turnover)
    torch.testing.assert_close(result.diagnostics["one_way_turnover"], turnover)
    torch.testing.assert_close(
        result.execution_cost,
        fixed_turnover_cost(target, current, 12.5 / 10_000.0),
    )
    torch.testing.assert_close(result.modeled_impact_cost, torch.zeros_like(turnover))


def test_fixed_turnover_execution_zero_cost_preserves_turnover_diagnostics() -> None:
    current = torch.tensor([[0.0, 1.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])

    result = FixedTurnoverTargetWeightExecution().execute(current, target, cash_index=0)

    torch.testing.assert_close(result.execution_cost, torch.zeros(1))
    torch.testing.assert_close(result.modeled_impact_cost, torch.zeros(1))
    torch.testing.assert_close(result.traded_notional, torch.ones(1))


@pytest.mark.parametrize("cost_bps", [True, -0.1, float("nan"), float("inf"), "not-a-number"])
def test_fixed_turnover_execution_rejects_invalid_cost_bps(cost_bps: object) -> None:
    with pytest.raises(ValueError, match="cost_bps"):
        FixedTurnoverTargetWeightExecution(cost_bps=cost_bps)  # type: ignore[arg-type]


def test_fixed_turnover_execution_coerces_numeric_cost_bps() -> None:
    model = FixedTurnoverTargetWeightExecution(cost_bps="2.5")  # type: ignore[arg-type]
    assert model.cost_bps == 2.5
    assert isinstance(model.cost_bps, float)


@pytest.mark.parametrize(
    ("current", "target", "cash_index", "message"),
    [
        (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 0, "identical"),
        (torch.ones(1, 2), torch.ones(2, 2), 0, "identical"),
        (torch.ones(1, 2, dtype=torch.int64), torch.ones(1, 2, dtype=torch.int64), 0, "floating dtype"),
        (torch.ones(1, 2), torch.ones(1, 2, dtype=torch.float64), 0, "floating dtype"),
        (torch.tensor([[float("nan"), 0.0]]), torch.tensor([[1.0, 0.0]]), 0, "finite"),
        (torch.ones(1, 2), torch.ones(1, 2), True, "integer, not bool"),
        (torch.ones(1, 2), torch.ones(1, 2), 2, "outside"),
    ],
)
def test_fixed_turnover_execution_validates_transition_inputs(
    current: torch.Tensor,
    target: torch.Tensor,
    cash_index: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FixedTurnoverTargetWeightExecution(cost_bps=1.0).execute(
            current,
            target,
            cash_index=cash_index,  # type: ignore[arg-type]
        )


def test_fixed_turnover_cost_uses_environment_additive_liquidation_ledger() -> None:
    data = HistoricalMarketData(
        features={"signal": torch.zeros(1, 2, 1)},
        asset_returns=torch.tensor([[[0.0, 0.10]]]),
        availability=torch.ones(1, 2, 2, dtype=torch.bool),
    )
    env = VectorPortfolioEnv(
        data,
        execution_model=FixedTurnoverTargetWeightExecution(cost_bps=10.0),
    )
    env.reset()

    transition = env.step(torch.tensor([[0.0, 1.0]]))

    # Entry costs 0.1% of opening equity. The terminal CASH transition costs
    # 0.1% of post-return/post-entry equity, then is converted once into the
    # beginning-of-step units required by the additive reward ledger.
    expected_pre_liquidation_growth = torch.tensor([1.0 + 0.10 - 0.001])
    expected_liquidation_cost = expected_pre_liquidation_growth * 0.001
    expected_equity = expected_pre_liquidation_growth * (1.0 - 0.001)
    torch.testing.assert_close(transition.rewards.execution_cost, torch.tensor([0.001]))
    torch.testing.assert_close(transition.rewards.impact_cost, torch.tensor([0.0]))
    torch.testing.assert_close(transition.rewards.liquidation_cost, expected_liquidation_cost)
    torch.testing.assert_close(transition.reward, expected_equity - 1.0)
    torch.testing.assert_close(env.equity, expected_equity)
    torch.testing.assert_close(transition.info["execution_one_way_turnover"], torch.tensor([1.0]))
    torch.testing.assert_close(transition.info["liquidation_execution_one_way_turnover"], torch.tensor([1.0]))

