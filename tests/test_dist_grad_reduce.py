"""Real-process regression tests for the external Phase-1 gradient reducer."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
import queue
import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DIST_UTILS_PATH = Path(__file__).resolve().parents[2] / "training" / "dist_utils.py"
SPEC = importlib.util.spec_from_file_location("phase1_dist_utils", DIST_UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
dist_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dist_utils)


def _rank_gradients(rank: int) -> list[torch.Tensor | None]:
    """Dense gradients chosen to force several buckets and one asymmetric parameter."""
    if rank == 0:
        return [
            torch.tensor([1.0, 2.0, 3.0]),
            None,
            None,
            torch.tensor([-1.0, 2.0, 0.0, 4.0]),
            torch.tensor([1.5, -0.5], dtype=torch.float64),
        ]
    return [
        torch.tensor([5.0, 6.0, 7.0]),
        torch.tensor([2.0, 4.0]),
        None,
        torch.tensor([3.0, 0.0, 2.0, 8.0]),
        torch.tensor([2.5, 4.5], dtype=torch.float64),
    ]


def _gloo_grad_worker(rank: int, world: int, init_path: str, result_queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(init_path).as_uri(),
            rank=rank,
            world_size=world,
            timeout=dt.timedelta(seconds=20),
        )
        parameters = [
            torch.nn.Parameter(torch.zeros(3)),
            torch.nn.Parameter(torch.zeros(2)),
            torch.nn.Parameter(torch.zeros(1)),
            torch.nn.Parameter(torch.zeros(4)),
            torch.nn.Parameter(torch.zeros(2, dtype=torch.float64)),
        ]
        for parameter, gradient in zip(parameters, _rank_gradients(rank), strict=True):
            parameter.grad = gradient

        collective_trace: list[tuple[str, int, str]] = []
        original_all_reduce = dist.all_reduce

        def traced_all_reduce(
            tensor: torch.Tensor,
            op=dist.ReduceOp.SUM,
            group=None,
            async_op: bool = False,
        ):
            if op == dist.ReduceOp.MAX:
                op_name = "max"
            elif op == dist.ReduceOp.SUM:
                op_name = "sum"
            else:
                op_name = str(op)
            collective_trace.append((op_name, tensor.numel(), str(tensor.dtype)))
            return original_all_reduce(tensor, op=op, group=group, async_op=async_op)

        dist.all_reduce = traced_all_reduce
        try:
            # 16 bytes yields float32 buckets of 3, 2, and 4 elements; the final float64 parameter gets its own
            # dtype-separated bucket. Every rank must launch the same presence + four payload collectives.
            dist_utils.make_grad_reduce(world, bucket_bytes=16)(parameters)
        finally:
            dist.all_reduce = original_all_reduce

        result_queue.put({
            "rank": rank,
            "gradients": [None if parameter.grad is None else parameter.grad.tolist() for parameter in parameters],
            "dtypes": [None if parameter.grad is None else str(parameter.grad.dtype) for parameter in parameters],
            "collectives": collective_trace,
        })
        # This final rendezvous makes completion itself evidence that both ranks left the reducer in the same
        # collective state. It is deliberately outside the traced all_reduce sequence asserted by the parent.
        dist.barrier()
    except BaseException:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_bucketed_grad_reduce_handles_asymmetric_and_globally_unused_parameters(tmp_path: Path) -> None:
    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / "gloo_init"
    processes = [
        context.Process(target=_gloo_grad_worker, args=(rank, world, str(init_path), result_queue))
        for rank in range(world)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + 30.0
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    stuck = [process for process in processes if process.is_alive()]
    if stuck:
        for process in stuck:
            process.terminate()
        for process in stuck:
            process.join(5.0)
        pytest.fail("Gloo gradient reduction deadlocked: ranks launched different collectives")

    assert [process.exitcode for process in processes] == [0, 0]
    results = []
    try:
        for _ in range(world):
            results.append(result_queue.get(timeout=5.0))
    except queue.Empty:
        pytest.fail("A Gloo worker exited without returning its gradient-reduction result")
    results.sort(key=lambda result: result["rank"])
    errors = [result["error"] for result in results if "error" in result]
    assert not errors, "\n".join(errors)

    expected = [
        [3.0, 4.0, 5.0],
        [1.0, 2.0],  # rank 0 contributes zeros; rank 1 contributes [2, 4]
        None,
        [1.0, 1.0, 1.0, 6.0],
        [2.0, 2.0],
    ]
    for result in results:
        assert result["gradients"] == expected
        assert result["gradients"][1] is not None  # locally unused on rank 0, globally used
        assert result["gradients"][2] is None      # globally unused stays None (no weight decay)
        assert result["dtypes"] == ["torch.float32", "torch.float32", None, "torch.float32", "torch.float64"]

    expected_collectives = [
        ("max", 5, "torch.uint8"),
        ("sum", 3, "torch.float32"),
        ("sum", 2, "torch.float32"),
        ("sum", 4, "torch.float32"),
        ("sum", 2, "torch.float64"),
    ]
    assert results[0]["collectives"] == expected_collectives
    assert results[1]["collectives"] == expected_collectives
