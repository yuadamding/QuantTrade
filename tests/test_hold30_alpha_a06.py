from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_quant.training.hold30_alpha import (
    Hold30A06OptimizerStateReceipt,
    Hold30AlphaBatch,
    Hold30AlphaObjectiveConfig,
    Hold30AlphaTrainingError,
    bind_hold30_alpha_global_moments,
    build_hold30_a06_optimizer_spec_receipt,
    build_hold30_a06_optimizer_state_receipt,
    hold30_a06_overlay_two_pass_objective,
    hold30_alpha_evaluation_point_id,
    partition_hold30_a06_parameters,
    train_hold30_a06_two_optimizer_update,
)

A06 = "hold30a-a06-sharpe-overlay"
ROWS = 6


class _A06Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha_core = torch.nn.Parameter(
            torch.linspace(-0.002, 0.002, ROWS, dtype=torch.float64)
        )
        self.total_risk_head = torch.nn.ParameterDict(
            {
                "offset": torch.nn.Parameter(
                    torch.linspace(0.0003, -0.0002, ROWS, dtype=torch.float64)
                )
            }
        )

    @property
    def overlay(self) -> torch.nn.Parameter:
        return self.total_risk_head["offset"]


def _config(receipt_id: str) -> Hold30AlphaObjectiveConfig:
    return Hold30AlphaObjectiveConfig(
        setting_id=A06,
        downside_penalty_kappa=0.75,
        active_log_scale_bounds=(-2.0, 2.0),
        uncertainty_log_scale_bounds=(-4.0, 2.0),
        auxiliary_horizon_weights=(0.15, 0.20, 0.50, 0.15),
        auxiliary_horizon_scales=(1.0, 1.0, 1.0, 1.0),
        a06_total_risk_step=0.05,
        alpha_core_parameter_selector="alpha-core-only",
        overlay_parameter_selector="a06-overlay-only",
        stop_gradient_core_to_overlay=True,
        stop_gradient_overlay_to_core=True,
        separate_optimizer_spec_receipt_sha256=receipt_id,
        lambda_te_floor=0.2,
        lambda_te_ceiling=0.2,
        lambda_beta=0.2,
        lambda_turnover=0.2,
        lambda_early_exit=0.2,
        lambda_auxiliary_alpha=0.2,
        lambda_uncertainty=0.2,
        lambda_total_excess_mean=0.2,
        lambda_total_sharpe_overlay=0.2,
        total_sharpe_epsilon=1e-8,
        lambda_volatility_ratio=0.2,
        target_volatility_ratio=1.0,
        lambda_drawdown=0.2,
        drawdown_limit=0.001,
        qualification_math_test_only=True,
    )


def _batch(
    policy_return: torch.Tensor,
    *,
    stream_id: str,
    evaluation_point_id: str,
) -> Hold30AlphaBatch:
    device = policy_return.device
    dtype = policy_return.dtype
    benchmark = torch.tensor(
        [-0.0015, -0.0002, 0.0008, 0.0016, -0.0004, 0.0011],
        dtype=dtype,
        device=device,
    )
    market = torch.tensor(
        [-0.0020, -0.0005, 0.0010, 0.0025, 0.0002, 0.0018],
        dtype=dtype,
        device=device,
    )
    risk_free = torch.linspace(0.00005, 0.00010, ROWS, dtype=dtype, device=device)
    auxiliary = torch.zeros((ROWS, 2, 4), dtype=dtype, device=device)
    return Hold30AlphaBatch(
        binding_kind="qualification-math-fixture",
        source_axis_id="a06-qualification-axis",
        objective_inputs_id="c" * 64,
        role="qualification-math-fixture",
        stream_id=stream_id,
        origin_row_ids=torch.arange(ROWS, dtype=torch.int64, device=device),
        global_path_ids=torch.zeros(ROWS, dtype=torch.int64, device=device),
        evaluation_point_id=evaluation_point_id,
        policy_net_return=policy_return,
        benchmark_net_return=benchmark,
        market_return=market,
        risk_free_return=risk_free,
        discretionary_turnover=torch.full_like(policy_return, 0.04),
        early_exit_mass=torch.full_like(policy_return, 0.01),
        auxiliary_prediction=auxiliary,
        auxiliary_target=torch.full_like(auxiliary, 0.001),
        auxiliary_valid=torch.ones_like(auxiliary, dtype=torch.bool),
        downside_30d=torch.full((ROWS, 2), 0.01, dtype=dtype, device=device),
    )


def _streams(
    model: _A06Policy,
) -> tuple[
    Hold30AlphaBatch,
    Hold30AlphaBatch,
    Hold30AlphaBatch,
    Hold30AlphaBatch,
]:
    evaluation_point = hold30_alpha_evaluation_point_id(model)
    combined = model.alpha_core + model.overlay
    alpha_core_pass_a = _batch(
        model.alpha_core.detach(),
        stream_id="a06-alpha-core",
        evaluation_point_id=evaluation_point,
    )
    alpha_core_pass_b = _batch(
        model.alpha_core,
        stream_id="a06-alpha-core",
        evaluation_point_id=evaluation_point,
    )
    executed_pass_a = _batch(
        combined.detach(),
        stream_id="a06-executed-overlay",
        evaluation_point_id=evaluation_point,
    )
    overlay_pass_b = _batch(
        model.alpha_core.detach() + model.overlay,
        stream_id="a06-executed-overlay",
        evaluation_point_id=evaluation_point,
    )
    return (
        alpha_core_pass_a,
        alpha_core_pass_b,
        executed_pass_a,
        overlay_pass_b,
    )


def _select_rows(
    batch: Hold30AlphaBatch,
    row_indices: torch.Tensor,
) -> Hold30AlphaBatch:
    updates: dict[str, torch.Tensor | None] = {}
    for name in (
        "origin_row_ids",
        "global_path_ids",
        "policy_net_return",
        "benchmark_net_return",
        "market_return",
        "risk_free_return",
        "discretionary_turnover",
        "early_exit_mass",
        "valid",
        "auxiliary_prediction",
        "auxiliary_target",
        "auxiliary_valid",
        "downside_30d",
    ):
        value = getattr(batch, name)
        updates[name] = (
            None if value is None else value.index_select(0, row_indices)
        )
    return replace(batch, **updates)


def _a06_optimizers(
    model: _A06Policy,
    config: Hold30AlphaObjectiveConfig,
):
    partition = partition_hold30_a06_parameters(model, config)
    alpha_core_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in partition.alpha_core),
        lr=0.01,
    )
    overlay_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in partition.overlay),
        lr=0.01,
    )
    spec = build_hold30_a06_optimizer_spec_receipt(
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
    )
    config = replace(
        config,
        separate_optimizer_spec_receipt_sha256=spec.receipt_id,
    )
    state = build_hold30_a06_optimizer_state_receipt(
        model,
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
        spec,
        update_index=0,
        parent_state_receipt_sha256=None,
    )
    return (
        config,
        alpha_core_optimizer,
        overlay_optimizer,
        spec,
        state,
    )
def test_a06_overlay_objective_and_disjoint_optimizer_update_are_receipt_bound() -> None:
    model = _A06Policy()
    provisional = _config("0" * 64)
    partition = partition_hold30_a06_parameters(model, provisional)
    assert partition.alpha_core_names == ("alpha_core",)
    assert partition.overlay_names == ("total_risk_head.offset",)

    alpha_core_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in partition.alpha_core),
        lr=0.01,
    )
    overlay_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in partition.overlay),
        lr=0.01,
    )
    spec = build_hold30_a06_optimizer_spec_receipt(
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
    )
    config = replace(
        provisional,
        separate_optimizer_spec_receipt_sha256=spec.receipt_id,
    )
    config.require_resolved()
    with pytest.raises(Hold30AlphaTrainingError, match="total_sharpe_epsilon"):
        replace(config, total_sharpe_epsilon=None).require_resolved()
    spec_payload = spec.manifest_payload()
    assert spec_payload["alpha_core_optimizer_parameter_names"] == ["alpha_core"]
    assert spec_payload["overlay_optimizer_parameter_names"] == [
        "total_risk_head.offset"
    ]
    state = build_hold30_a06_optimizer_state_receipt(
        model,
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
        spec,
        update_index=0,
        parent_state_receipt_sha256=None,
    )

    (
        _alpha_core_pass_a,
        _alpha_core_pass_b,
        executed_pass_a,
        overlay_pass_b,
    ) = _streams(model)
    binding = bind_hold30_alpha_global_moments((executed_pass_a,))
    overlay_objective, _metrics = hold30_a06_overlay_two_pass_objective(
        (executed_pass_a,),
        (overlay_pass_b,),
        config,
        global_moments=binding,
    )
    (-overlay_objective).backward()
    assert model.alpha_core.grad is None
    assert model.overlay.grad is not None

    alpha_core_optimizer.zero_grad(set_to_none=True)
    overlay_optimizer.zero_grad(set_to_none=True)
    before_core = model.alpha_core.detach().clone()
    before_overlay = model.overlay.detach().clone()
    (
        alpha_core_pass_a,
        alpha_core_pass_b,
        executed_pass_a,
        overlay_pass_b,
    ) = _streams(model)
    result = train_hold30_a06_two_optimizer_update(
        model,
        alpha_core_optimizer,
        overlay_optimizer,
        (alpha_core_pass_a,),
        (alpha_core_pass_b,),
        (executed_pass_a,),
        (overlay_pass_b,),
        config,
        optimizer_spec_receipt=spec,
        optimizer_state_receipt=state,
    )
    assert result["gradient_isolation_verified"] is True
    assert result["gradient_reduction"] == "SUM"
    assert result["alpha_core_optimizer_steps"] == 1
    assert result["overlay_optimizer_steps"] == 1
    assert result["three_stream_contract_verified"] is True
    assert result["optimizer_spec_receipt_sha256"] == spec.receipt_id
    assert result["pre_update_optimizer_state_receipt_sha256"] == state.receipt_id
    assert result["post_update_optimizer_state_receipt_sha256"] != state.receipt_id
    assert result["post_update_evaluation_point_id"] != (
        result["pre_update_evaluation_point_id"]
    )
    post_receipt = result["post_update_optimizer_state_receipt"]
    assert post_receipt["update_index"] == 1
    assert post_receipt["parent_state_receipt_sha256"] == state.receipt_id
    assert result["alpha_core_global_moment_receipt_sha256"] != (
        result["executed_global_moment_receipt_sha256"]
    )
    assert not torch.equal(
        alpha_core_pass_a.policy_net_return,
        executed_pass_a.policy_net_return,
    )
    assert not torch.equal(model.alpha_core.detach(), before_core)
    assert not torch.equal(model.overlay.detach(), before_overlay)
    assert build_hold30_a06_optimizer_spec_receipt(
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
    ) == spec

    (
        alpha_core_pass_a,
        alpha_core_pass_b,
        executed_pass_a,
        overlay_pass_b,
    ) = _streams(model)
    with pytest.raises(Hold30AlphaTrainingError, match="ledger receipt"):
        train_hold30_a06_two_optimizer_update(
            model,
            alpha_core_optimizer,
            overlay_optimizer,
            (alpha_core_pass_a,),
            (alpha_core_pass_b,),
            (executed_pass_a,),
            (overlay_pass_b,),
            config,
            optimizer_spec_receipt=spec,
            optimizer_state_receipt=state,
        )


def test_a06_state_ledger_restart_matches_uninterrupted_second_update() -> None:
    model = _A06Policy()
    provisional = _config("0" * 64)
    partition = partition_hold30_a06_parameters(model, provisional)
    alpha_core_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in partition.alpha_core),
        lr=0.01,
    )
    overlay_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in partition.overlay),
        lr=0.01,
    )
    spec = build_hold30_a06_optimizer_spec_receipt(
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
    )
    config = replace(
        provisional,
        separate_optimizer_spec_receipt_sha256=spec.receipt_id,
    )
    initial_state = build_hold30_a06_optimizer_state_receipt(
        model,
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
        spec,
        update_index=0,
        parent_state_receipt_sha256=None,
    )
    streams = _streams(model)
    first = train_hold30_a06_two_optimizer_update(
        model,
        alpha_core_optimizer,
        overlay_optimizer,
        *((stream,) for stream in streams),
        config,
        optimizer_spec_receipt=spec,
        optimizer_state_receipt=initial_state,
    )
    checkpoint_model = copy.deepcopy(model.state_dict())
    checkpoint_core_optimizer = copy.deepcopy(alpha_core_optimizer.state_dict())
    checkpoint_overlay_optimizer = copy.deepcopy(overlay_optimizer.state_dict())
    post_first = Hold30A06OptimizerStateReceipt(
        **first["post_update_optimizer_state_receipt"]
    )

    uninterrupted_streams = _streams(model)
    uninterrupted = train_hold30_a06_two_optimizer_update(
        model,
        alpha_core_optimizer,
        overlay_optimizer,
        *((stream,) for stream in uninterrupted_streams),
        config,
        optimizer_spec_receipt=spec,
        optimizer_state_receipt=post_first,
    )

    restarted_model = _A06Policy()
    restarted_model.load_state_dict(checkpoint_model)
    restarted_partition = partition_hold30_a06_parameters(
        restarted_model,
        config,
    )
    restarted_core_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in restarted_partition.alpha_core),
        lr=0.01,
    )
    restarted_overlay_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in restarted_partition.overlay),
        lr=0.01,
    )
    restarted_core_optimizer.load_state_dict(checkpoint_core_optimizer)
    restarted_overlay_optimizer.load_state_dict(checkpoint_overlay_optimizer)
    assert build_hold30_a06_optimizer_spec_receipt(
        restarted_partition,
        restarted_core_optimizer,
        restarted_overlay_optimizer,
    ) == spec
    restarted_streams = _streams(restarted_model)
    restarted = train_hold30_a06_two_optimizer_update(
        restarted_model,
        restarted_core_optimizer,
        restarted_overlay_optimizer,
        *((stream,) for stream in restarted_streams),
        config,
        optimizer_spec_receipt=spec,
        optimizer_state_receipt=post_first,
    )
    assert restarted["parameter_sha256"] == uninterrupted["parameter_sha256"]
    assert restarted["alpha_core_gradient_sha256"] == (
        uninterrupted["alpha_core_gradient_sha256"]
    )
    assert restarted["overlay_gradient_sha256"] == (
        uninterrupted["overlay_gradient_sha256"]
    )
    assert restarted["post_update_optimizer_state_receipt_sha256"] == (
        uninterrupted["post_update_optimizer_state_receipt_sha256"]
    )


def _distributed_a06_worker(rank: int, init_file: str, queue) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.set_num_threads(1)
        model = _A06Policy()
        (
            config,
            alpha_core_optimizer,
            overlay_optimizer,
            spec,
            state,
        ) = _a06_optimizers(model, _config("0" * 64))
        row_indices = torch.arange(rank, ROWS, 2, dtype=torch.int64)
        streams = tuple(
            _select_rows(stream, row_indices) for stream in _streams(model)
        )
        result = train_hold30_a06_two_optimizer_update(
            model,
            alpha_core_optimizer,
            overlay_optimizer,
            *((stream,) for stream in streams),
            config,
            optimizer_spec_receipt=spec,
            optimizer_state_receipt=state,
            group=dist.group.WORLD,
            moment_device="cpu",
        )
        if rank == 0:
            queue.put(
                {
                    "alpha_core": model.alpha_core.detach().tolist(),
                    "overlay": model.overlay.detach().tolist(),
                    "alpha_core_gradient_sha256": result[
                        "alpha_core_gradient_sha256"
                    ],
                    "overlay_gradient_sha256": result[
                        "overlay_gradient_sha256"
                    ],
                    "parameter_sha256": result["parameter_sha256"],
                    "post_state_sha256": result[
                        "post_update_optimizer_state_receipt_sha256"
                    ],
                    "alpha_core_objective": result["alpha_core_objective"],
                    "overlay_objective": result["overlay_objective"],
                    "core_moments": result["alpha_core_global_moment_receipt"],
                    "executed_moments": result["executed_global_moment_receipt"],
                }
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch distributed Gloo is unavailable",
)
def test_two_rank_a06_nonzero_objectives_match_one_rank_exactly(
    tmp_path: Path,
) -> None:
    model = _A06Policy()
    (
        config,
        alpha_core_optimizer,
        overlay_optimizer,
        spec,
        state,
    ) = _a06_optimizers(model, _config("0" * 64))
    expected = train_hold30_a06_two_optimizer_update(
        model,
        alpha_core_optimizer,
        overlay_optimizer,
        *((stream,) for stream in _streams(model)),
        config,
        optimizer_spec_receipt=spec,
        optimizer_state_receipt=state,
    )
    expected_core = model.alpha_core.detach().clone()
    expected_overlay = model.overlay.detach().clone()

    context = mp.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "hold30-a06-gloo-init")
    processes = [
        context.Process(
            target=_distributed_a06_worker,
            args=(rank, init_file, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    actual = queue.get(timeout=30)
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert torch.equal(
        torch.tensor(actual["alpha_core"], dtype=torch.float64),
        expected_core,
    )
    assert torch.equal(
        torch.tensor(actual["overlay"], dtype=torch.float64),
        expected_overlay,
    )
    for name in (
        "alpha_core_gradient_sha256",
        "overlay_gradient_sha256",
        "parameter_sha256",
    ):
        assert actual[name] == expected[name]
    assert actual["post_state_sha256"] == (
        expected["post_update_optimizer_state_receipt_sha256"]
    )
    assert actual["alpha_core_objective"] == pytest.approx(
        expected["alpha_core_objective"],
        rel=0.0,
        abs=1e-12,
    )
    assert actual["overlay_objective"] == pytest.approx(
        expected["overlay_objective"],
        rel=0.0,
        abs=1e-12,
    )
    for name in (
        "row_identity_sha256",
        "pass_a_content_sha256",
        "moments_sha256",
    ):
        assert actual["core_moments"][name] == (
            expected["alpha_core_global_moment_receipt"][name]
        )
        assert actual["executed_moments"][name] == (
            expected["executed_global_moment_receipt"][name]
        )


def test_a06_optimizer_receipt_rejects_overlapping_ownership() -> None:
    model = _A06Policy()
    config = _config("0" * 64)
    partition = partition_hold30_a06_parameters(model, config)
    wrong_core_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    overlay_optimizer = torch.optim.Adam(
        (parameter for _name, parameter in partition.overlay),
        lr=0.01,
    )
    with pytest.raises(Hold30AlphaTrainingError, match="does not exactly own"):
        build_hold30_a06_optimizer_spec_receipt(
            partition,
            wrong_core_optimizer,
            overlay_optimizer,
        )
