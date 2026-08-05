from __future__ import annotations

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.models.daily_policy import DailyCrossSectionConfig, DailyCrossSectionPolicy
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.hold30_state import (
    Hold30DailyPolicyInputs,
    Hold30DailyPolicyStateProvider,
)


def _fixture() -> tuple[
    DailyCrossSectionPolicy,
    Hold30Sequence,
    Hold30DailyPolicyStateProvider,
]:
    torch.manual_seed(19)
    dtype = torch.float32
    decisions, positions, batch, assets = 4, 5, 1, 3
    token_dim = 8
    config = DailyCrossSectionConfig(
        context_dim=8,
        bar_feature_dim=5,
        raw_policy_dim=8,
        raw_policy_layers=1,
        raw_policy_heads=2,
        raw_block_seconds=2,
        session_seconds=4,
        news_raw_dim=1,
        news_embed_dim=4,
        token_dim=token_dim,
        temporal_layers=1,
        temporal_heads=2,
        daily_lookback=3,
        max_days=5,
        alloc_layers=1,
        alloc_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        raw_recent_days=2,
        raw_stock_chunk=0,
        hold30_setting="hold30-m02-age-hazard",
    )
    policy = DailyCrossSectionPolicy(config)
    axis_id = "a" * 64
    masks = torch.ones((positions, batch, assets), dtype=torch.bool)
    weights = torch.tensor([[0.98, 0.01, 0.01]], dtype=dtype)
    benchmark = weights.unsqueeze(0).expand(positions, -1, -1).clone()
    sequence = Hold30Sequence(
        decision_state=torch.zeros((positions, batch, assets, token_dim), dtype=dtype),
        asset_returns=torch.zeros((decisions, batch, assets), dtype=dtype),
        decision_available=masks,
        fill_membership=masks.clone(),
        fill_availability=masks.clone(),
        benchmark_weights=benchmark,
        risk_asset_caps=torch.tensor([[[1.0, 0.01, 0.01]]], dtype=dtype).expand_as(benchmark).clone(),
        risk_gross_max=torch.ones((positions, batch), dtype=dtype),
        benchmark_net_returns=torch.zeros((decisions, batch), dtype=dtype),
        initial_ledger=CohortLedger.from_weights(weights, cash_index=0),
        cost_rate=0.0,
        axis_id=axis_id,
    )

    market = torch.randn(batch, decisions, 8, dtype=dtype)
    stock = torch.randn(batch, decisions, assets, 8, dtype=dtype)
    news = torch.zeros(batch, decisions, assets, 1, 1, dtype=dtype)
    news_mask = torch.zeros(batch, decisions, assets, 1, dtype=torch.bool)
    past_return = torch.randn(batch, decisions, assets, dtype=dtype) * 0.01
    past_valid = torch.ones(batch, decisions, assets, dtype=torch.bool)
    bars_by_day: list[torch.Tensor] = []
    masks_by_day: list[torch.Tensor] = []
    for _ in range(decisions):
        bars = torch.rand(batch, assets, 4, 5, dtype=dtype)
        bars[..., :4] = 50.0 + bars[..., :4]
        bars_by_day.append(bars)
        masks_by_day.append(torch.ones(batch, assets, 4, dtype=torch.bool))

    inputs = Hold30DailyPolicyInputs(
        market_context=market,
        stock_context=stock,
        news_raw=news,
        news_mask=news_mask,
        available=masks[:-1].permute(1, 0, 2).clone(),
        past_return=past_return,
        past_return_valid=past_valid,
        day_bars_fn=lambda index: (bars_by_day[index], masks_by_day[index]),
        source_axis_id=axis_id,
        raw_bars_sha256="b" * 64,
        frozen_context_sha256="c" * 64,
    )
    return policy, sequence, Hold30DailyPolicyStateProvider(inputs)


def _long_fixture(
    *,
    decisions: int,
    daily_lookback: int,
    raw_recent_days: int,
    day_loads: list[int] | None = None,
) -> tuple[
    DailyCrossSectionPolicy,
    Hold30Sequence,
    Hold30DailyPolicyStateProvider,
]:
    """Small tensors with production-like rolling-window geometry."""

    torch.manual_seed(23)
    dtype = torch.float32
    positions, batch, assets, token_dim = decisions + 1, 1, 3, 8
    config = DailyCrossSectionConfig(
        context_dim=8,
        bar_feature_dim=5,
        raw_policy_dim=8,
        raw_policy_layers=1,
        raw_policy_heads=2,
        raw_block_seconds=2,
        session_seconds=4,
        news_raw_dim=1,
        news_embed_dim=4,
        token_dim=token_dim,
        temporal_layers=1,
        temporal_heads=2,
        daily_lookback=daily_lookback,
        max_days=max(decisions + 1, daily_lookback + 1),
        alloc_layers=1,
        alloc_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        raw_recent_days=raw_recent_days,
        raw_stock_chunk=0,
        hold30_setting="hold30-m02-age-hazard",
    )
    policy = DailyCrossSectionPolicy(config)
    masks = torch.ones((positions, batch, assets), dtype=torch.bool)
    weights = torch.tensor([[0.98, 0.01, 0.01]], dtype=dtype)
    benchmark = weights.unsqueeze(0).expand(positions, -1, -1).clone()
    axis_id = "e" * 64
    sequence = Hold30Sequence(
        decision_state=torch.zeros((positions, batch, assets, token_dim), dtype=dtype),
        asset_returns=torch.zeros((decisions, batch, assets), dtype=dtype),
        decision_available=masks,
        fill_membership=masks.clone(),
        fill_availability=masks.clone(),
        benchmark_weights=benchmark,
        risk_asset_caps=torch.tensor([[[1.0, 0.01, 0.01]]], dtype=dtype)
        .expand_as(benchmark)
        .clone(),
        risk_gross_max=torch.ones((positions, batch), dtype=dtype),
        benchmark_net_returns=torch.zeros((decisions, batch), dtype=dtype),
        initial_ledger=CohortLedger.from_weights(weights, cash_index=0),
        cost_rate=0.0,
        axis_id=axis_id,
    )
    market = torch.randn(batch, decisions, 8, dtype=dtype)
    stock = torch.randn(batch, decisions, assets, 8, dtype=dtype)
    news = torch.zeros(batch, decisions, assets, 1, 1, dtype=dtype)
    news_mask = torch.zeros(batch, decisions, assets, 1, dtype=torch.bool)
    past_return = torch.randn(batch, decisions, assets, dtype=dtype) * 0.01
    past_valid = torch.ones(batch, decisions, assets, dtype=torch.bool)
    bars_by_day = [torch.rand(batch, assets, 4, 5, dtype=dtype) for _ in range(decisions)]
    bar_masks_by_day = [torch.ones(batch, assets, 4, dtype=torch.bool) for _ in range(decisions)]

    def day_bars(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if day_loads is not None:
            day_loads.append(index)
        return bars_by_day[index], bar_masks_by_day[index]

    inputs = Hold30DailyPolicyInputs(
        market_context=market,
        stock_context=stock,
        news_raw=news,
        news_mask=news_mask,
        available=masks[:-1].permute(1, 0, 2).clone(),
        past_return=past_return,
        past_return_valid=past_valid,
        day_bars_fn=day_bars,
        source_axis_id=axis_id,
        raw_bars_sha256="f" * 64,
        frozen_context_sha256="1" * 64,
    )
    return policy, sequence, Hold30DailyPolicyStateProvider(inputs)


def test_daily_policy_provider_replays_exact_state_with_upstream_gradients() -> None:
    policy, sequence, provider = _fixture()
    policy.train()
    with torch.no_grad():
        canonical = provider.canonical_states(policy, sequence)

    assert isinstance(canonical, torch.Tensor)
    assert canonical.shape == (4, 1, 3, 8)
    for origin in range(4):
        replay = provider.replay_origin_state(policy, sequence, origin)
        torch.testing.assert_close(replay, canonical[origin], atol=2e-6, rtol=2e-6)

    policy.zero_grad(set_to_none=True)
    replay = provider.replay_origin_state(policy, sequence, 3)
    replay.square().mean().backward()
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in policy.raw_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in policy.temporal.parameters()
    )


def test_origin_batch_matches_scalar_states_and_upstream_gradients() -> None:
    policy, sequence, provider = _long_fixture(
        decisions=8,
        daily_lookback=5,
        raw_recent_days=3,
    )
    origins = torch.tensor([1, 4, 7])

    scalar = torch.stack(
        [provider.replay_origin_state(policy, sequence, int(origin)) for origin in origins]
    )
    scalar_parameters = tuple(
        parameter
        for name, parameter in policy.named_parameters()
        if name.startswith(("raw_encoder.", "token_proj.", "temporal."))
    )
    scalar_gradients = torch.autograd.grad(scalar.square().sum(), scalar_parameters)

    batched = provider.replay_origin_states(policy, sequence, origins)
    batched_gradients = torch.autograd.grad(batched.square().sum(), scalar_parameters)

    torch.testing.assert_close(batched, scalar, atol=2e-6, rtol=2e-6)
    for batched_gradient, scalar_gradient in zip(
        batched_gradients,
        scalar_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            batched_gradient,
            scalar_gradient,
            atol=2e-5,
            rtol=2e-5,
        )


def test_contiguous_origin_batch_loads_raw_union_once_and_batches_temporal() -> None:
    day_loads: list[int] = []
    policy, sequence, provider = _long_fixture(
        decisions=95,
        daily_lookback=63,
        raw_recent_days=42,
        day_loads=day_loads,
    )
    origins = torch.arange(63, 95)
    temporal_shapes: list[tuple[int, ...]] = []
    original_temporal_state = policy.temporal_state

    def observed_temporal_state(tokens: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        temporal_shapes.append(tuple(tokens.shape))
        return original_temporal_state(tokens, available)

    policy.temporal_state = observed_temporal_state  # type: ignore[method-assign]
    states = provider.replay_origin_states(policy, sequence, origins)

    assert states.shape == (32, 1, 3, 8)
    # Origin 63 needs raw days 22..63; origin 94 needs 53..94.  Their
    # contiguous union is exactly 73 days, not 32 x 63 window loads.
    assert day_loads == list(range(22, 95))
    assert temporal_shapes == [(32, 63, 3, 8)]


def test_daily_policy_provider_binding_is_content_addressed_and_fail_closed() -> None:
    policy, sequence, provider = _fixture()
    binding = provider.binding_config
    assert binding["provider"].endswith("Hold30DailyPolicyStateProvider")
    assert binding["schema_version"] == 2
    assert binding["replay_batching"] == "union-raw-days-v1"
    assert len(str(binding["binding_sha256"])) == 64

    bad_inputs = Hold30DailyPolicyInputs(
        **{
            **provider.inputs.__dict__,
            "source_axis_id": "d" * 64,
        }
    )
    bad_provider = Hold30DailyPolicyStateProvider(bad_inputs)
    try:
        bad_provider.canonical_states(policy, sequence)
    except ValueError as error:
        assert "source axis" in str(error)
    else:
        raise AssertionError("a mismatched provider axis must fail closed")
