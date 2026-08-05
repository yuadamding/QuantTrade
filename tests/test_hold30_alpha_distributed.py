from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_quant.training.hold30_alpha import (
    Hold30AlphaBatch,
    Hold30AlphaMomentSums,
    Hold30AlphaObjectiveConfig,
    Hold30AlphaTrainingError,
    bind_hold30_alpha_global_moments,
    hold30_alpha_two_pass_objective,
    train_hold30_alpha_two_pass_update,
)

M02 = "hold30a-m02-active-te"
ROWS = 6
ORIGIN_ROWS = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
GLOBAL_PATHS = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)


class _RowPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.returns = torch.nn.Parameter(torch.zeros(ROWS, dtype=torch.float64))


def _config() -> Hold30AlphaObjectiveConfig:
    return Hold30AlphaObjectiveConfig(
        setting_id=M02,
        lambda_te_floor=0.0,
        lambda_te_ceiling=0.0,
        lambda_turnover=0.0,
        lambda_early_exit=0.0,
        qualification_math_test_only=True,
    )


def _batch(policy: torch.Tensor, row_indices: torch.Tensor) -> Hold30AlphaBatch:
    dtype = policy.dtype
    market = torch.tensor(
        [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0],
        dtype=dtype,
        device=policy.device,
    ).mul_(2.0**-12)
    selected_market = market.index_select(0, row_indices.to(device=policy.device))
    zeros = torch.zeros_like(policy).detach()
    return Hold30AlphaBatch(
        binding_kind="qualification-math-fixture",
        source_axis_id="distributed-qualification-axis",
        objective_inputs_id="c" * 64,
        role="qualification-math-fixture",
        stream_id="primary",
        origin_row_ids=ORIGIN_ROWS.index_select(0, row_indices).to(
            device=policy.device
        ),
        global_path_ids=GLOBAL_PATHS.index_select(0, row_indices).to(
            device=policy.device
        ),
        evaluation_point_id="e" * 64,
        policy_net_return=policy,
        benchmark_net_return=zeros,
        market_return=selected_market.detach(),
        risk_free_return=zeros,
        discretionary_turnover=zeros,
        early_exit_mass=zeros,
    )


def _batches(
    policy: _RowPolicy,
    row_indices: torch.Tensor,
) -> tuple[Hold30AlphaBatch, Hold30AlphaBatch]:
    selected = policy.returns.index_select(0, row_indices)
    return _batch(selected.detach(), row_indices), _batch(selected, row_indices)


def _optimizer(policy: _RowPolicy) -> torch.optim.SGD:
    return torch.optim.SGD(policy.parameters(), lr=0.125)


def test_external_global_moments_require_exact_row_and_evaluation_receipt() -> None:
    policy = _RowPolicy()
    row_ids = torch.arange(ROWS, dtype=torch.int64)
    pass_a, pass_b = _batches(policy, row_ids)
    binding = bind_hold30_alpha_global_moments((pass_a,))

    objective, metrics = hold30_alpha_two_pass_objective(
        (pass_a,),
        (pass_b,),
        _config(),
        global_moments=binding,
    )
    assert metrics.count == ROWS
    assert binding.world_size == 1
    assert binding.manifest_payload()["row_count"] == ROWS
    assert len(binding.receipt_id) == 64
    objective.backward()
    assert policy.returns.grad is not None

    raw = Hold30AlphaMomentSums.from_batch(pass_a)
    with pytest.raises(Hold30AlphaTrainingError, match="typed row/evaluation receipt"):
        hold30_alpha_two_pass_objective(
            (pass_a,),
            (pass_b,),
            _config(),
            global_moments=raw,  # type: ignore[arg-type]
        )

    other_point_a = replace(pass_a, evaluation_point_id="f" * 64)
    other_point_b = replace(pass_b, evaluation_point_id="f" * 64)
    with pytest.raises(Hold30AlphaTrainingError, match="another data or evaluation"):
        hold30_alpha_two_pass_objective(
            (other_point_a,),
            (other_point_b,),
            _config(),
            global_moments=binding,
        )

    changed_a = replace(
        pass_a,
        policy_net_return=pass_a.policy_net_return + 2.0**-20,
    )
    changed_b = replace(
        pass_b,
        policy_net_return=pass_b.policy_net_return + 2.0**-20,
    )
    with pytest.raises(Hold30AlphaTrainingError, match="exact local.*row content"):
        hold30_alpha_two_pass_objective(
            (changed_a,),
            (changed_b,),
            _config(),
            global_moments=binding,
        )


def _distributed_update_worker(
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
        torch.set_num_threads(1)
        policy = _RowPolicy()
        # Shard by globally bound path columns, not by dates.  Each rank sees
        # every origin but owns one stable global path ID.
        row_indices = torch.arange(rank, ROWS, 2, dtype=torch.int64)
        pass_a, pass_b = _batches(policy, row_indices)
        result = train_hold30_alpha_two_pass_update(
            policy,
            _optimizer(policy),
            (pass_a,),
            (pass_b,),
            _config(),
            moment_device="cpu",
        )
        if rank == 0:
            assert policy.returns.grad is not None
            queue.put(
                {
                    "parameters": policy.returns.detach().tolist(),
                    "gradients": policy.returns.grad.detach().tolist(),
                    "objective": result["objective"],
                    "row_identity_sha256": result["row_identity_sha256"],
                    "pass_a_content_sha256": result["pass_a_content_sha256"],
                    "moments_sha256": result["moments_sha256"],
                    "gradient_sha256": result["gradient_sha256"],
                    "parameter_sha256": result["parameter_sha256"],
                    "world_size": result["distributed_world_size"],
                    "gradient_reduction": result["gradient_reduction"],
                }
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch distributed Gloo is unavailable",
)
def test_two_rank_pass_b_gradient_and_parameter_update_match_one_rank(
    tmp_path: Path,
) -> None:
    policy = _RowPolicy()
    row_ids = torch.arange(ROWS, dtype=torch.int64)
    pass_a, pass_b = _batches(policy, row_ids)
    expected = train_hold30_alpha_two_pass_update(
        policy,
        _optimizer(policy),
        (pass_a,),
        (pass_b,),
        _config(),
    )
    assert policy.returns.grad is not None
    expected_parameters = policy.returns.detach().clone()
    expected_gradients = policy.returns.grad.detach().clone()

    context = mp.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "hold30-alpha-gloo-init")
    processes = [
        context.Process(
            target=_distributed_update_worker,
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

    actual_parameters = torch.tensor(actual["parameters"], dtype=torch.float64)
    actual_gradients = torch.tensor(actual["gradients"], dtype=torch.float64)
    assert torch.equal(actual_gradients, expected_gradients)
    assert torch.equal(actual_parameters, expected_parameters)
    assert actual["objective"] == expected["objective"]
    assert actual["row_identity_sha256"] == expected["row_identity_sha256"]
    assert actual["pass_a_content_sha256"] == expected["pass_a_content_sha256"]
    assert actual["moments_sha256"] == expected["moments_sha256"]
    assert actual["gradient_sha256"] == expected["gradient_sha256"]
    assert actual["parameter_sha256"] == expected["parameter_sha256"]
    assert actual["world_size"] == 2
    assert actual["gradient_reduction"] == "SUM"
