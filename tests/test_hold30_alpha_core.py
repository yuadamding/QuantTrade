from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_quant.execution.hold30 import build_alpha_hold30_action
from rl_quant.models.hold30_alpha import (
    Hold30AlphaHead,
    Hold30AlphaHeadConfig,
    Hold30AlphaModelError,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_TE_TARGET_ANNUAL,
    HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
)
from rl_quant.training.hold30_alpha import (
    HOLD30_ALPHA_ANNUALIZATION,
    Hold30AlphaBatch,
    Hold30AlphaMomentSums,
    Hold30AlphaObjectiveConfig,
    Hold30AlphaTrainingError,
    Hold30AlphaUnresolvedCoefficientError,
    Hold30AlphaValidationMetrics,
    aggregate_hold30_alpha_moments,
    distributed_sum_hold30_alpha_moments,
    hold30_alpha_global_metrics,
    hold30_alpha_two_pass_objective,
)

M02 = "hold30a-m02-active-te"
M03 = "hold30a-m03-alpha-core"
A04 = "hold30a-a04-no-uncertainty"
A05 = "hold30a-a05-no-te-floor"
A06 = "hold30a-a06-sharpe-overlay"
A07 = "hold30a-a07-direct-sharpe"


def _batch(
    policy: torch.Tensor,
    *,
    alpha_heads: bool,
    uncertainty: bool = True,
    benchmark: torch.Tensor | None = None,
    market: torch.Tensor | None = None,
    risk_free: torch.Tensor | None = None,
) -> Hold30AlphaBatch:
    n = policy.numel()
    dtype = policy.dtype
    device = policy.device
    benchmark = (
        torch.linspace(0.0002, 0.0010, n, dtype=dtype, device=device)
        if benchmark is None
        else benchmark.to(device=device, dtype=dtype)
    ).detach()
    market = (
        torch.tensor(
            [-0.0020, -0.0005, 0.0010, 0.0025, 0.0002, 0.0018],
            dtype=dtype,
            device=device,
        )[:n]
        if market is None
        else market.to(device=device, dtype=dtype)
    ).detach()
    risk_free = (
        torch.linspace(0.00005, 0.00010, n, dtype=dtype, device=device)
        if risk_free is None
        else risk_free.to(device=device, dtype=dtype)
    ).detach()
    auxiliary_prediction = None
    auxiliary_target = None
    auxiliary_valid = None
    downside = None
    if alpha_heads:
        auxiliary_prediction = torch.zeros(
            (n, 2, 4), dtype=dtype, device=device
        )
        auxiliary_target = torch.zeros_like(auxiliary_prediction)
        auxiliary_valid = torch.ones_like(auxiliary_prediction, dtype=torch.bool)
        if uncertainty:
            downside = torch.ones((n, 2), dtype=dtype, device=device)
    return Hold30AlphaBatch(
        binding_kind="qualification-math-fixture",
        source_axis_id="synthetic-axis",
        objective_inputs_id="c" * 64,
        role="qualification-math-fixture",
        stream_id="primary",
        origin_row_ids=torch.arange(n, dtype=torch.int64, device=device),
        global_path_ids=torch.zeros(n, dtype=torch.int64, device=device),
        evaluation_point_id="e" * 64,
        policy_net_return=policy,
        benchmark_net_return=benchmark,
        market_return=market,
        risk_free_return=risk_free,
        discretionary_turnover=torch.zeros_like(policy),
        early_exit_mass=torch.zeros_like(policy),
        auxiliary_prediction=auxiliary_prediction,
        auxiliary_target=auxiliary_target,
        auxiliary_valid=auxiliary_valid,
        downside_30d=downside,
    )


def _config(setting_id: str, **updates: object) -> Hold30AlphaObjectiveConfig:
    values: dict[str, object] = {
        "setting_id": setting_id,
        "lambda_te_floor": 0.7,
        "lambda_te_ceiling": 0.9,
        "lambda_turnover": 0.0,
        "lambda_early_exit": 0.0,
        "qualification_math_test_only": True,
    }
    if setting_id != M02:
        values.update(
            {
                "active_log_scale_bounds": (-2.0, 2.0),
                "auxiliary_horizon_weights": (0.15, 0.20, 0.50, 0.15),
                "auxiliary_horizon_scales": (1.0, 1.0, 1.0, 1.0),
                "lambda_beta": 0.6,
                "lambda_auxiliary_alpha": 0.0,
            }
        )
        if setting_id != A04:
            values.update(
                {
                    "downside_penalty_kappa": 0.75,
                    "uncertainty_log_scale_bounds": (-4.0, 2.0),
                    "lambda_uncertainty": 0.0,
                }
            )
    if setting_id == A05:
        values["lambda_te_floor"] = None
    if setting_id == A07:
        values.update(
            {
                "lambda_direct_sharpe": 0.4,
                "direct_sharpe_epsilon": 1e-8,
            }
        )
    values.update(updates)
    return Hold30AlphaObjectiveConfig(**values)


def _central_difference(
    function,
    values: torch.Tensor,
    *,
    step: float = 1e-7,
) -> torch.Tensor:
    result = torch.empty_like(values)
    for index in range(values.numel()):
        plus = values.detach().clone()
        minus = values.detach().clone()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2.0 * step)
    return result


def _two_pass_gradient(
    values: torch.Tensor,
    config: Hold30AlphaObjectiveConfig,
    *,
    benchmark: torch.Tensor | None = None,
    market: torch.Tensor | None = None,
    risk_free: torch.Tensor | None = None,
) -> torch.Tensor:
    uncertainty = config.setting_id != A04
    pass_a = _batch(
        values.detach(),
        alpha_heads=config.setting_id != M02,
        uncertainty=uncertainty,
        benchmark=benchmark,
        market=market,
        risk_free=risk_free,
    )
    differentiable = values.detach().clone().requires_grad_(True)
    pass_b = _batch(
        differentiable,
        alpha_heads=config.setting_id != M02,
        uncertainty=uncertainty,
        benchmark=benchmark,
        market=market,
        risk_free=risk_free,
    )
    objective, _metrics = hold30_alpha_two_pass_objective(
        (pass_a,), (pass_b,), config
    )
    objective.backward()
    assert differentiable.grad is not None
    return differentiable.grad.detach()


def test_alpha_head_is_m03_only_and_uses_mu_minus_kappa_downside() -> None:
    with pytest.raises(Hold30AlphaModelError, match="m00-m02"):
        Hold30AlphaHeadConfig(
            setting_id=M02,
            hidden_dim=8,
            active_log_scale_bounds=(-2.0, 2.0),
        )
    with pytest.raises(Hold30AlphaModelError, match="downside_penalty_kappa"):
        Hold30AlphaHeadConfig(
            setting_id=M03,
            hidden_dim=8,
            active_log_scale_bounds=(-2.0, 2.0),
        )

    head = Hold30AlphaHead(
        Hold30AlphaHeadConfig(
            setting_id=M03,
            hidden_dim=8,
            downside_penalty_kappa=0.75,
            active_log_scale_bounds=(-2.0, 2.0),
            uncertainty_log_scale_bounds=(-4.0, 2.0),
        )
    )
    hidden = torch.randn(2, 4, 8)
    available = torch.ones(2, 4, dtype=torch.bool)
    output = head(
        hidden,
        torch.zeros(2, 4),
        torch.zeros(2, 4, 5),
        available,
    )
    assert output.downside_30d is not None
    torch.testing.assert_close(
        output.risk_adjusted_score,
        output.mean_30d - 0.75 * output.downside_30d,
    )
    horizon_30 = (5, 21, 30, 63).index(30)
    assert torch.equal(output.auxiliary_mean[..., horizon_30], output.mean_30d)
    score_gradient = torch.autograd.grad(
        output.risk_adjusted_score.sum(),
        head.auxiliary_head[-1].weight,
        retain_graph=True,
    )[0]
    assert bool((score_gradient[horizon_30] != 0).any())
    head.zero_grad(set_to_none=True)
    output.auxiliary_mean[..., horizon_30].square().sum().backward()
    supervised_gradient = head.auxiliary_head[-1].weight.grad
    assert supervised_gradient is not None
    assert bool((supervised_gradient[horizon_30] != 0).any())

    mean_only = Hold30AlphaHead(
        Hold30AlphaHeadConfig(
            setting_id=A04,
            hidden_dim=8,
            active_log_scale_bounds=(-2.0, 2.0),
        )
    )(hidden, torch.zeros(2, 4), torch.zeros(2, 4, 5), available)
    assert mean_only.downside_30d is None
    torch.testing.assert_close(mean_only.risk_adjusted_score, mean_only.mean_30d)


def test_active_risk_bounds_are_explicit_and_have_deterministic_boundaries() -> None:
    with pytest.raises(Hold30AlphaModelError, match="active_log_scale_bounds"):
        Hold30AlphaHeadConfig(
            setting_id=A04,
            hidden_dim=4,
        )
    head = Hold30AlphaHead(
        Hold30AlphaHeadConfig(
            setting_id=A04,
            hidden_dim=4,
            active_log_scale_bounds=(-1.25, 0.75),
        )
    )
    hidden = torch.zeros(1, 3, 4)
    available = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        head.active_risk_head[-1].bias.fill_(100.0)
    high = head(hidden, torch.zeros(1, 3), torch.zeros(1, 3, 5), available)
    assert high.active_risk_scale.item() == pytest.approx(
        HOLD30_ALPHA_TE_TARGET_ANNUAL * math.exp(0.75)
    )
    with torch.no_grad():
        head.active_risk_head[-1].bias.fill_(-100.0)
    low = head(hidden, torch.zeros(1, 3), torch.zeros(1, 3, 5), available)
    assert low.active_risk_scale.item() == pytest.approx(
        HOLD30_ALPHA_TE_TARGET_ANNUAL * math.exp(-1.25)
    )


def test_model_cash_and_unavailable_downside_zeros_are_safe_in_batch_nll() -> None:
    head = Hold30AlphaHead(
        Hold30AlphaHeadConfig(
            setting_id=M03,
            hidden_dim=8,
            downside_penalty_kappa=0.75,
            active_log_scale_bounds=(-2.0, 2.0),
            uncertainty_log_scale_bounds=(-4.0, 2.0),
        )
    )
    hidden = torch.randn(6, 4, 8)
    available = torch.ones(6, 4, dtype=torch.bool)
    available[::2, 3] = False
    output = head(
        hidden,
        torch.zeros(6, 4),
        torch.zeros(6, 4, 5),
        available,
    )
    assert output.downside_30d is not None
    valid_assets = available.clone()
    valid_assets[:, 0] = False
    assert torch.equal(
        output.downside_30d.masked_select(~valid_assets),
        torch.zeros_like(output.downside_30d.masked_select(~valid_assets)),
    )
    valid = valid_assets.unsqueeze(-1).expand(6, 4, 4)
    policy = torch.linspace(-0.001, 0.002, 6, dtype=torch.float64).requires_grad_(
        True
    )
    base = _batch(policy, alpha_heads=True)
    target = torch.full_like(output.auxiliary_mean, 0.01)
    pass_b = replace(
        base,
        auxiliary_prediction=output.auxiliary_mean.to(torch.float64),
        auxiliary_target=target.to(torch.float64),
        auxiliary_valid=valid,
        downside_30d=output.downside_30d.to(torch.float64),
    )
    pass_a = replace(
        pass_b,
        policy_net_return=policy.detach(),
        auxiliary_prediction=pass_b.auxiliary_prediction.detach(),
        downside_30d=pass_b.downside_30d.detach(),
    )
    objective, _metrics = hold30_alpha_two_pass_objective(
        (pass_a,),
        (pass_b,),
        _config(M03, lambda_auxiliary_alpha=1.0, lambda_uncertainty=1.0),
    )
    assert torch.isfinite(objective)
    objective.backward()
    gradients = [
        parameter.grad
        for parameter in head.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_a06_overlay_stream_is_one_way_isolated_and_config_resolves() -> None:
    head = Hold30AlphaHead(
        Hold30AlphaHeadConfig(
            setting_id=A06,
            hidden_dim=6,
            downside_penalty_kappa=0.5,
            active_log_scale_bounds=(-2.0, 2.0),
            uncertainty_log_scale_bounds=(-4.0, 2.0),
        )
    )
    hidden = torch.randn(1, 3, 6, requires_grad=True)
    output = head(
        hidden,
        torch.zeros(1, 3),
        torch.zeros(1, 3, 5),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert output.total_risk_overlay is not None
    output.total_risk_overlay.sum().backward()
    assert hidden.grad is None or not bool((hidden.grad != 0).any())
    assert head.total_risk_head is not None
    assert any(parameter.grad is not None for parameter in head.total_risk_head.parameters())
    assert all(parameter.grad is None for parameter in head.auxiliary_head.parameters())

    config = _config(
        A06,
        lambda_total_excess_mean=0.2,
        lambda_total_sharpe_overlay=0.3,
        total_sharpe_epsilon=1e-8,
        lambda_volatility_ratio=0.4,
        target_volatility_ratio=1.0,
        lambda_drawdown=0.5,
        drawdown_limit=0.2,
        a06_total_risk_step=0.1,
        alpha_core_parameter_selector="alpha-core-only",
        overlay_parameter_selector="a06-overlay-only",
        stop_gradient_core_to_overlay=True,
        stop_gradient_overlay_to_core=True,
        separate_optimizer_spec_receipt_sha256="a" * 64,
    )
    config.require_resolved()


def test_canonical_alpha_action_cannot_time_gross_market_exposure() -> None:
    repaired = torch.tensor([[0.20, 0.30, 0.25, 0.25]], dtype=torch.float64)
    age = torch.zeros((1, 4, 61), dtype=torch.float64)
    age[:, :, 30] = repaired
    benchmark = torch.tensor([[0.20, 0.20, 0.30, 0.30]], dtype=torch.float64)
    built = build_alpha_hold30_action(
        repaired,
        age,
        torch.tensor([[0.0, -1.0, 0.5, 1.5]], dtype=torch.float64),
        torch.zeros_like(repaired),
        torch.tensor([0.04], dtype=torch.float64),
        benchmark,
        torch.ones_like(repaired, dtype=torch.bool),
        torch.ones_like(repaired),
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        built.desired_risky_exposure,
        repaired[:, 1:].sum(-1),
    )
    with pytest.raises(ValueError, match="forbidden"):
        build_alpha_hold30_action(
            repaired,
            age,
            torch.zeros_like(repaired),
            torch.zeros_like(repaired),
            torch.tensor([0.04], dtype=torch.float64),
            benchmark,
            torch.ones_like(repaired, dtype=torch.bool),
            torch.ones_like(repaired),
            torch.ones(1, dtype=torch.float64),
            total_risk_step=0.1,
        )


def test_m02_has_only_active_te_turnover_and_early_exit_terms() -> None:
    config = _config(M02)
    config.require_resolved()
    plain = _batch(torch.linspace(0.0, 0.002, 6, dtype=torch.float64), alpha_heads=False)
    config.validate_batch(plain)
    with pytest.raises(Hold30AlphaTrainingError, match="beta loss is exclusive"):
        replace(config, lambda_beta=0.1).require_resolved()
    supervised = _batch(
        plain.policy_net_return,
        alpha_heads=True,
    )
    with pytest.raises(Hold30AlphaTrainingError, match="forbids supervised"):
        config.validate_batch(supervised)


def test_te_squared_hinge_two_pass_matches_monolithic_finite_difference() -> None:
    benchmark = torch.tensor(
        [0.0010, 0.0008, 0.0011, 0.0009, 0.0012, 0.0010],
        dtype=torch.float64,
    )
    values = benchmark + torch.tensor(
        [-0.00015, -0.00005, 0.00002, 0.00008, 0.00012, -0.00002],
        dtype=torch.float64,
    )
    config = _config(M03, lambda_beta=0.0, lambda_te_floor=0.8, lambda_te_ceiling=1.1)

    def monolithic(policy: torch.Tensor) -> torch.Tensor:
        active = torch.log1p(policy) - torch.log1p(benchmark)
        te = active.std(unbiased=True) * math.sqrt(HOLD30_ALPHA_ANNUALIZATION)
        return (
            active.mean()
            - 0.8 * torch.relu(policy.new_tensor(0.02) - te).square()
            - 1.1 * torch.relu(te - policy.new_tensor(0.06)).square()
        )

    expected = _central_difference(monolithic, values)
    actual = _two_pass_gradient(values, config, benchmark=benchmark)
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-6)


def test_beta_uses_pit_market_excess_not_c1_and_matches_finite_difference() -> None:
    benchmark = torch.tensor(
        [0.01, -0.008, 0.006, -0.004, 0.003, -0.002],
        dtype=torch.float64,
    )
    market = torch.tensor(
        [-0.003, -0.001, 0.0005, 0.002, 0.0035, 0.001],
        dtype=torch.float64,
    )
    risk_free = torch.tensor(
        [0.0001, 0.0001, 0.0002, 0.0001, 0.0002, 0.0001],
        dtype=torch.float64,
    )
    market_excess = market - risk_free
    values = risk_free + 1.45 * market_excess + torch.tensor(
        [0.0001, -0.0001, 0.00005, 0.0, -0.00005, 0.00008],
        dtype=torch.float64,
    )
    config = _config(
        M03,
        lambda_te_floor=0.0,
        lambda_te_ceiling=0.0,
        lambda_beta=0.7,
    )

    def monolithic(policy: torch.Tensor) -> torch.Tensor:
        active = torch.log1p(policy) - torch.log1p(benchmark)
        policy_excess = policy - risk_free
        centered_market = market_excess - market_excess.mean()
        beta = (
            (policy_excess - policy_excess.mean()) * centered_market
        ).sum() / centered_market.square().sum()
        return active.mean() - 0.7 * (beta - 1.0).square()

    expected = _central_difference(monolithic, values)
    actual = _two_pass_gradient(
        values,
        config,
        benchmark=benchmark,
        market=market,
        risk_free=risk_free,
    )
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-6)
    metrics = hold30_alpha_global_metrics(
        aggregate_hold30_alpha_moments(
            (
                _batch(
                    values,
                    alpha_heads=True,
                    benchmark=benchmark,
                    market=market,
                    risk_free=risk_free,
                ),
            )
        )
    )
    assert metrics.beta == pytest.approx(1.45, abs=0.05)


def test_a07_population_sharpe_two_pass_matches_finite_difference() -> None:
    values = torch.tensor(
        [-0.0015, 0.0003, 0.0012, 0.0020, -0.0004, 0.0010],
        dtype=torch.float64,
    )
    benchmark = torch.tensor(
        [-0.0010, 0.0002, 0.0008, 0.0011, -0.0002, 0.0007],
        dtype=torch.float64,
    )
    risk_free = torch.tensor(
        [0.0001, 0.0001, 0.00015, 0.0001, 0.00015, 0.0001],
        dtype=torch.float64,
    )
    config = _config(
        A07,
        lambda_te_floor=0.0,
        lambda_te_ceiling=0.0,
        lambda_beta=0.0,
        lambda_direct_sharpe=0.4,
        direct_sharpe_epsilon=2e-8,
    )

    def monolithic(policy: torch.Tensor) -> torch.Tensor:
        active = torch.log1p(policy) - torch.log1p(benchmark)
        excess = policy - risk_free
        mean = excess.mean()
        variance = (excess - mean).square().mean()
        sharpe = math.sqrt(HOLD30_ALPHA_ANNUALIZATION) * mean / torch.sqrt(
            variance + 2e-8
        )
        return active.mean() + 0.4 * sharpe

    expected = _central_difference(monolithic, values, step=2e-8)
    actual = _two_pass_gradient(
        values,
        config,
        benchmark=benchmark,
        risk_free=risk_free,
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    with pytest.raises(Hold30AlphaUnresolvedCoefficientError, match="epsilon"):
        replace(config, direct_sharpe_epsilon=None).require_resolved()


def test_a05_training_ablation_does_not_remove_common_selection_te_floor() -> None:
    contract = replace(
        HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
        projection_distance_max=0.1,
        forced_turnover_fraction_max=0.1,
    )
    metrics = Hold30AlphaValidationMetrics(
        update=32,
        coverage_complete=True,
        active_return_20bp=0.01,
        active_return_40bp=0.005,
        tracking_error=0.019,
        beta=1.0,
        median_sale_age=30.0,
        projection_distance=0.01,
        forced_turnover_fraction=0.01,
        median_active_return_20bp=0.01,
        information_ratio_20bp=0.5,
        total_sharpe_20bp=1.0,
        max_drawdown_20bp=0.1,
        turnover_cost_20bp=0.01,
        trace_sha256="b" * 64,
    )
    assert not metrics.eligible(A05, contract=contract)
    assert replace(metrics, tracking_error=0.02).eligible(A05, contract=contract)


def _distributed_moment_worker(
    rank: int,
    init_file: str,
    queue,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        values = torch.tensor(
            [-0.001, 0.0002, 0.0010, 0.0018, -0.0003, 0.0007],
            dtype=torch.float64,
        )
        benchmark = torch.linspace(0.0002, 0.0010, 6, dtype=torch.float64)
        market = torch.tensor(
            [-0.0020, -0.0005, 0.0010, 0.0025, 0.0002, 0.0018],
            dtype=torch.float64,
        )
        risk_free = torch.linspace(0.00005, 0.00010, 6, dtype=torch.float64)
        rows = slice(rank * 3, (rank + 1) * 3)
        local = Hold30AlphaMomentSums.from_batch(
            _batch(
                values[rows],
                alpha_heads=False,
                benchmark=benchmark[rows],
                market=market[rows],
                risk_free=risk_free[rows],
            )
        )
        reduced = distributed_sum_hold30_alpha_moments(local, device="cpu")
        if rank == 0:
            queue.put(reduced.packed().tolist())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_rank_global_moment_sum_matches_one_rank(tmp_path: Path) -> None:
    values = torch.tensor(
        [-0.001, 0.0002, 0.0010, 0.0018, -0.0003, 0.0007],
        dtype=torch.float64,
    )
    expected = Hold30AlphaMomentSums.from_batch(
        _batch(values, alpha_heads=False)
    ).packed()
    context = mp.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "gloo-init")
    processes = [
        context.Process(
            target=_distributed_moment_worker,
            args=(rank, init_file, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    actual = torch.tensor(queue.get(timeout=30), dtype=torch.float64)
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=0.0)
