from __future__ import annotations

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.training.top2000_m03r_v7_dev import (
    Top2000M03RV7ActionBuilder,
    Top2000M03RV7DevelopmentPolicy,
    build_top2000_m03r_v7_development_optimizers,
    top2000_m03r_v7_direct_sharpe_vjp_coefficients,
    top2000_m03r_v7_factor_neutral_executed_weights,
    top2000_m03r_v7_total_excess_sharpe,
)


def _a06_policy() -> Top2000M03RV7DevelopmentPolicy:
    return Top2000M03RV7DevelopmentPolicy(
        "A06-sharpe-overlay-top2000-dev-v1",
        token_dim=16,
        raw_stock_chunk=32,
    )


def test_a06_uses_disjoint_optimizer_states_and_stop_gradient_routes() -> None:
    policy = _a06_policy()
    core_optimizer, overlay_optimizer = (
        build_top2000_m03r_v7_development_optimizers(
            policy,
            learning_rate=1.0e-4,
            weight_decay=1.0e-4,
        )
    )
    assert overlay_optimizer is not None
    core_parameters = policy.alpha_core_parameters()
    overlay_parameters = policy.total_risk_overlay_parameters()
    assert {id(value) for value in core_parameters}.isdisjoint(
        {id(value) for value in overlay_parameters}
    )
    assert {
        id(value)
        for group in core_optimizer.param_groups
        for value in group["params"]
    } == {id(value) for value in core_parameters}
    assert {
        id(value)
        for group in overlay_optimizer.param_groups
        for value in group["params"]
    } == {id(value) for value in overlay_parameters}

    batch, assets = 2, 9
    state = torch.randn(batch, assets, policy.token_dim)
    previous = torch.full((batch, assets), 1.0 / assets)
    available = torch.ones((batch, assets), dtype=torch.bool)
    ages = torch.zeros((batch, assets, 5))

    policy.zero_grad(set_to_none=True)
    policy.set_gradient_route("alpha-core")
    alpha_intent = policy.hold30_intent(state, previous, available, ages)
    assert alpha_intent.entry_scores is not None
    assert alpha_intent.total_risk_overlay is not None
    assert not alpha_intent.total_risk_overlay.requires_grad
    alpha_intent.entry_scores.sum().backward()  # type: ignore[no-untyped-call]
    assert all(value.grad is None for value in overlay_parameters)

    policy.zero_grad(set_to_none=True)
    policy.set_gradient_route("total-risk-overlay")
    overlay_intent = policy.hold30_intent(state, previous, available, ages)
    assert overlay_intent.entry_scores is not None
    assert overlay_intent.total_risk_overlay is not None
    assert not overlay_intent.entry_scores.requires_grad
    assert overlay_intent.total_risk_overlay.requires_grad
    overlay_intent.total_risk_overlay.sum().backward()  # type: ignore[no-untyped-call]
    assert all(value.grad is None for value in core_parameters)
    assert any(value.grad is not None for value in overlay_parameters)


def test_a07_two_pass_coefficients_are_total_excess_return_sharpe_gradient() -> None:
    active = torch.tensor(
        [0.0010, -0.0004, 0.0007, 0.0002, -0.0001, 0.0009],
        dtype=torch.float64,
    )
    benchmark = torch.tensor(
        [0.010, -0.006, 0.004, 0.002, -0.003, 0.008],
        dtype=torch.float64,
    )
    cash = torch.tensor(
        [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        dtype=torch.float64,
    )
    rows = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    coefficients, reported = top2000_m03r_v7_direct_sharpe_vjp_coefficients(
        active,
        benchmark,
        cash,
        rows,
    )

    direct = active.clone().requires_grad_(True)
    expected = top2000_m03r_v7_total_excess_sharpe(
        direct.index_select(0, rows),
        benchmark.index_select(0, rows),
        cash.index_select(0, rows),
    )
    expected.backward()  # type: ignore[no-untyped-call]
    assert direct.grad is not None
    assert torch.allclose(coefficients, direct.grad)
    assert torch.allclose(reported, expected.detach())
    assert coefficients[0] == 0
    assert coefficients[5] == 0

    # An active-return Sharpe would be invariant to this benchmark path.  The
    # A07 statistic intentionally is not: it measures investor total return.
    different_benchmark = benchmark.flip(0)
    different = top2000_m03r_v7_total_excess_sharpe(
        active.index_select(0, rows),
        different_benchmark.index_select(0, rows),
        cash.index_select(0, rows),
    )
    assert not torch.allclose(reported, different)


def test_factor_neutrality_is_applied_to_executed_active_weights_with_gradient() -> None:
    torch.manual_seed(9)
    assets = 121
    benchmark = torch.zeros((1, assets), dtype=torch.float64)
    benchmark[:, 1:] = 1.0 / (assets - 1)
    requested = benchmark + 5.0e-4 * torch.randn_like(benchmark)
    requested = requested.clone().requires_grad_(True)
    loadings = torch.randn((assets, 4), dtype=torch.float64)
    loadings[0] = 0.0
    trade_mask = torch.ones((1, assets), dtype=torch.bool)
    caps = torch.full((1, assets), 0.01, dtype=torch.float64)
    caps[:, 0] = 0.0
    gross = torch.ones(1, dtype=torch.float64)

    projected = top2000_m03r_v7_factor_neutral_executed_weights(
        requested,
        benchmark,
        loadings,
        trade_mask,
        caps,
        gross,
    )
    active = projected - benchmark
    assert torch.allclose(projected.sum(-1), torch.ones(1, dtype=torch.float64))
    assert float((active @ loadings).abs().max()) < 1.0e-9
    assert float(projected.min()) >= -1.0e-10
    assert float(projected[:, 1:].max()) <= 0.01 + 1.0e-10

    objective = (projected * torch.linspace(0.0, 1.0, assets)).sum()
    objective.backward()  # type: ignore[no-untyped-call]
    assert requested.grad is not None
    assert float(requested.grad.abs().sum()) > 0.0
    # A pure factor direction is removed, demonstrating that repair acts on
    # the executed book rather than rewriting the model's input scores.
    factor_direction = loadings[:, 0].clone()
    factor_direction -= factor_direction.mean()
    factor_direction[0] = -factor_direction[1:].sum()
    shifted = top2000_m03r_v7_factor_neutral_executed_weights(
        requested.detach() + 1.0e-5 * factor_direction,
        benchmark,
        loadings,
        trade_mask,
        caps,
        gross,
    )
    assert torch.allclose(projected, shifted, atol=2.0e-6, rtol=2.0e-6)


def test_action_builder_applies_projection_except_for_a10() -> None:
    torch.manual_seed(19)
    assets = 121
    benchmark = torch.zeros((1, assets), dtype=torch.float32)
    benchmark[:, 1:] = 1.0 / (assets - 1)
    economic_value = torch.zeros((1, assets, 61), dtype=torch.float32)
    economic_value[:, 1:, 30] = benchmark[:, 1:]
    ledger = CohortLedger(economic_value, economic_value.clone(), cash_index=0)
    loadings = torch.randn((assets, 3), dtype=torch.float32)
    loadings[0] = 0.0
    intent = Hold30Intent(
        entry_scores=loadings[:, 0].unsqueeze(0),
        hazard_residual=torch.zeros((1, assets)),
        raw_hazard_residual=torch.zeros((1, assets)),
        active_risk_scale=torch.full((1,), 0.04),
        signal_confidence=torch.ones(1),
        uncalibrated_signal_confidence_logit=torch.ones(1),
    )
    trade_mask = torch.ones((1, assets), dtype=torch.bool)
    caps = torch.full((1, assets), 0.01, dtype=torch.float32)
    caps[:, 0] = 1.0
    gross = torch.ones(1, dtype=torch.float32)

    canonical = Top2000M03RV7DevelopmentPolicy(
        "M03R-soft-persistence-active-alpha-hold30-top2000-dev-v1",
        token_dim=16,
        raw_stock_chunk=32,
    )
    a10 = Top2000M03RV7DevelopmentPolicy(
        "A10-no-factor-neutral-projection-top2000-dev-v1",
        token_dim=16,
        raw_stock_chunk=32,
    )
    canonical.bind_episode_factor_loadings(loadings)
    a10.bind_episode_factor_loadings(loadings)
    canonical_target = Top2000M03RV7ActionBuilder(canonical)(
        intent,
        ledger,
        benchmark,
        trade_mask,
        caps,
        gross,
    ).target_weights
    a10_target = Top2000M03RV7ActionBuilder(a10)(
        intent,
        ledger,
        benchmark,
        trade_mask,
        caps,
        gross,
    ).target_weights
    canonical_exposure = (canonical_target - benchmark) @ loadings
    a10_exposure = (a10_target - benchmark) @ loadings
    assert float(canonical_exposure.abs().max()) < 2.0e-4
    assert float(a10_exposure.abs().max()) > 1.0e-3
