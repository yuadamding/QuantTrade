from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.execution import (
    AgeAwareNoTradeConfig,
    AlphaExecutionCostError,
    ExecutionCostObservation,
    ForecastDistribution,
    SquareRootImpactConfig,
    estimate_execution_cost,
    evaluate_capacity,
    evaluate_replacement,
)
from rl_quant.models import (
    AlphaDistribution,
    AlphaDistributionHead,
    CrossDayAlphaConfig,
    CrossDayAlphaEncoder,
    HierarchicalAlphaModel,
    MarketLatentConfig,
    MarketLatentEncoder,
    OrderedFiveMinuteConfig,
    OrderedFiveMinuteEncoder,
)
from rl_quant.training.alpha_supervised import (
    AlphaObjectiveConfig,
    AlphaSupervisedBatch,
    alpha_supervised_loss,
)


def _intraday_encoder() -> OrderedFiveMinuteEncoder:
    torch.manual_seed(1)
    model = OrderedFiveMinuteEncoder(
        OrderedFiveMinuteConfig(
            feature_dimension=5,
            token_dimension=16,
            layers=2,
            heads=4,
            feedforward_dimension=32,
            dropout=0.0,
            intervals_per_session=8,
        )
    )
    model.eval()
    return model


def test_ordered_intraday_tokens_are_causal() -> None:
    encoder = _intraday_encoder()
    raw = torch.randn(2, 8, 5)
    valid = torch.ones(2, 8, dtype=torch.bool)
    changed = raw.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 100.0

    first = encoder.forward_sequence(raw, valid)
    second = encoder.forward_sequence(changed, valid)

    torch.testing.assert_close(first[:, :5], second[:, :5], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first[:, -1], second[:, -1])


def test_invalid_intraday_payload_is_ignored_not_imputed_from_batch() -> None:
    encoder = _intraday_encoder()
    raw = torch.randn(1, 8, 5)
    valid = torch.tensor([[True, True, False, True, True, False, True, True]])
    changed = raw.clone()
    changed[:, ~valid[0]] = torch.nan

    first = encoder(raw, valid)
    second = encoder(changed, valid)

    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_cross_day_tokens_are_causal() -> None:
    torch.manual_seed(2)
    encoder = CrossDayAlphaEncoder(
        CrossDayAlphaConfig(
            token_dimension=16,
            layers=2,
            heads=4,
            feedforward_dimension=32,
            dropout=0.0,
            context_sessions=6,
        )
    ).eval()
    tokens = torch.randn(2, 6, 16)
    valid = torch.ones(2, 6, dtype=torch.bool)
    changed = tokens.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 50.0

    first = encoder.forward_sequence(tokens, valid)
    second = encoder.forward_sequence(changed, valid)

    torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-6, rtol=1e-6)


def test_market_latent_encoder_is_stock_permutation_equivariant() -> None:
    torch.manual_seed(3)
    encoder = MarketLatentEncoder(
        MarketLatentConfig(token_dimension=16, latent_count=4, heads=4, dropout=0.0)
    ).eval()
    stocks = torch.randn(2, 7, 16)
    valid = torch.tensor(
        [[True, True, True, True, False, True, True], [True] * 7]
    )
    permutation = torch.tensor((4, 2, 6, 0, 5, 1, 3))
    inverse = torch.argsort(permutation)

    original_stocks, original_market = encoder(stocks, valid)
    permuted_stocks, permuted_market = encoder(
        stocks[:, permutation], valid[:, permutation]
    )

    torch.testing.assert_close(
        original_stocks,
        permuted_stocks[:, inverse],
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(original_market, permuted_market, atol=2e-6, rtol=2e-6)


def test_distribution_head_orders_quantiles_and_positive_scale() -> None:
    head = AlphaDistributionHead(16, 4)
    output = head(torch.randn(3, 5, 16))

    assert output.mean.shape == (3, 5, 4)
    assert bool((output.downside_quantile <= output.median).all())
    assert bool((output.median <= output.upside_quantile).all())
    assert bool((output.scale > 0.0).all())


def test_hierarchical_model_shapes_and_backpropagates() -> None:
    torch.manual_seed(4)
    model = HierarchicalAlphaModel(
        OrderedFiveMinuteConfig(5, 16, 1, 4, 32, 0.0, 4),
        CrossDayAlphaConfig(16, 1, 4, 32, 0.0, 3),
        MarketLatentConfig(16, 4, 4, 0.0),
        horizons=2,
    )
    raw = torch.randn(1, 3, 3, 4, 5)
    interval_valid = torch.ones(1, 3, 3, 4, dtype=torch.bool)
    interval_valid[:, 0, 2] = False
    stock_day_valid = interval_valid.any(dim=-1)
    stock_valid = stock_day_valid.any(dim=1)

    output = model(raw, interval_valid, stock_day_valid, stock_valid)
    output.distribution.mean.sum().backward()

    assert output.stock_context.shape == (1, 3, 16)
    assert output.market_context.shape == (1, 4, 16)
    assert output.distribution.mean.shape == (1, 3, 2)
    assert model.intraday.input_projection.weight.grad is not None


def _distribution(mean: torch.Tensor, *, scale: float = 1.0) -> AlphaDistribution:
    return AlphaDistribution(
        mean=mean,
        downside_quantile=mean - 0.5,
        median=mean,
        upside_quantile=mean + 0.5,
        scale=torch.full_like(mean, scale),
    )


def test_alpha_loss_balances_dates_not_asset_counts() -> None:
    target = torch.zeros(2, 4, 1)
    mean = torch.zeros_like(target, requires_grad=True)
    mean.data[0, :2] = 1.0
    valid = torch.tensor(
        [[[True], [True], [False], [False]], [[True], [True], [True], [True]]]
    )
    batch = AlphaSupervisedBatch(
        distribution=_distribution(mean),
        target=target,
        valid=valid,
        executable_score=mean,
    )
    loss = alpha_supervised_loss(
        batch,
        AlphaObjectiveConfig(
            rank_weight=0.0,
            quantile_weight=0.0,
            calibration_weight=0.0,
            residual_ssl_weight=0.0,
        ),
    )

    assert loss.robust_mean.item() == pytest.approx(0.25)
    loss.total.backward()
    assert mean.grad is not None


def test_rank_loss_rewards_correct_tail_order() -> None:
    target = torch.tensor([[[-2.0], [-1.0], [1.0], [2.0]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    aligned = target.clone().requires_grad_()
    reversed_score = (-target).clone().requires_grad_()
    config = AlphaObjectiveConfig(
        huber_weight=1.0,
        rank_weight=1.0,
        quantile_weight=0.0,
        calibration_weight=0.0,
        residual_ssl_weight=0.0,
    )

    aligned_loss = alpha_supervised_loss(
        AlphaSupervisedBatch(_distribution(aligned), target, valid, aligned), config
    )
    reversed_loss = alpha_supervised_loss(
        AlphaSupervisedBatch(
            _distribution(reversed_score), target, valid, reversed_score
        ),
        config,
    )

    assert aligned_loss.rank < reversed_loss.rank


def test_age_preference_is_soft_not_a_mandatory_hold() -> None:
    config = AgeAwareNoTradeConfig(
        preferred_holding_sessions=30,
        young_position_penalty_return=0.01,
        downside_penalty=0.0,
        epistemic_penalty=0.0,
    )
    held = ForecastDistribution(-0.20, -0.20, 0.0)
    candidate = ForecastDistribution(0.10, 0.10, 0.0)
    decision = evaluate_replacement(
        held=held,
        candidate=candidate,
        held_age_sessions=1,
        sell_cost_return=0.001,
        buy_cost_return=0.001,
        config=config,
    )

    assert decision.action == "replace"
    assert decision.young_sale_penalty > 0.0


def test_age_penalty_decays_and_forced_exit_is_exempt() -> None:
    config = AgeAwareNoTradeConfig(30, 0.01, 0.0, 0.0)
    held = ForecastDistribution(0.01, 0.01, 0.0)
    candidate = ForecastDistribution(0.015, 0.015, 0.0)
    young = evaluate_replacement(
        held=held,
        candidate=candidate,
        held_age_sessions=0,
        sell_cost_return=0.0,
        buy_cost_return=0.0,
        config=config,
    )
    old = evaluate_replacement(
        held=held,
        candidate=candidate,
        held_age_sessions=30,
        sell_cost_return=0.0,
        buy_cost_return=0.0,
        config=config,
    )
    forced = evaluate_replacement(
        held=held,
        candidate=candidate,
        held_age_sessions=0,
        sell_cost_return=0.01,
        buy_cost_return=0.01,
        config=config,
        forced_exit=True,
    )

    assert young.action == "keep"
    assert old.action == "replace"
    assert forced.action == "forced-exit"
    assert forced.young_sale_penalty == 0.0


def _cost_observation(order: float) -> ExecutionCostObservation:
    return ExecutionCostObservation(
        decision_at_ms=1_000,
        available_at_ms=999,
        spread_basis_points=8.0,
        daily_volatility=0.02,
        order_notional=order,
        trailing_adv_notional=1_000_000.0,
        delay_cost_basis_points=1.0,
        fee_basis_points=0.2,
    )


def test_execution_cost_increases_with_participation() -> None:
    config = SquareRootImpactConfig(0.1, 0.02, 0.20)
    small = estimate_execution_cost(_cost_observation(10_000.0), config)
    large = estimate_execution_cost(_cost_observation(100_000.0), config)

    assert large.participation > small.participation
    assert large.total_one_way_cost_basis_points > small.total_one_way_cost_basis_points


def test_execution_cost_rejects_future_available_inputs() -> None:
    with pytest.raises(AlphaExecutionCostError, match="unavailable"):
        estimate_execution_cost(
            replace(_cost_observation(10_000.0), available_at_ms=1_001),
            SquareRootImpactConfig(0.1, 0.02, 0.20),
        )


def test_capacity_cost_and_clipping_increase_with_capital() -> None:
    config = SquareRootImpactConfig(0.1, 0.02, 0.05)
    sources = (_cost_observation(0.0), _cost_observation(0.0))
    small = evaluate_capacity(
        capital=1_000_000.0,
        order_weight_changes=(0.01, -0.01),
        observations=sources,
        config=config,
    )
    large = evaluate_capacity(
        capital=20_000_000.0,
        order_weight_changes=(0.01, -0.01),
        observations=sources,
        config=config,
    )

    assert large.weighted_cost_basis_points > small.weighted_cost_basis_points
    assert large.clipped_order_fraction == 1.0
    assert large.lost_notional_fraction > 0.0
