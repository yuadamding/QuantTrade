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


def test_broadcast_bool_uses_one_backend_tensor_and_obeys_source(monkeypatch) -> None:
    calls: list[tuple[torch.dtype, str, int]] = []
    monkeypatch.setattr(dist_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(dist_utils.dist, "is_initialized", lambda: True)

    def fake_broadcast(signal: torch.Tensor, src: int) -> None:
        calls.append((signal.dtype, signal.device.type, src))
        signal.fill_(1)

    monkeypatch.setattr(dist_utils.dist, "broadcast", fake_broadcast)

    assert dist_utils.broadcast_bool(False, torch.device("cpu"), src=0)
    assert calls == [(torch.uint8, "cpu", 0)]


def test_broadcast_bool_requires_initialized_group(monkeypatch) -> None:
    monkeypatch.setattr(dist_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(dist_utils.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="initialized process group"):
        dist_utils.broadcast_bool(True, torch.device("cpu"))


class _ScalarPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))


class _TinyContextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, bars, _mask, cov, _cov_valid=None):
        batch, actions = bars.shape[:2]
        blocks = cov.shape[1]
        per_stock = self.weight.expand(batch, blocks, actions, 1)
        market = self.weight.expand(batch, blocks, 1)
        return per_stock, market


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


def _gloo_sampler_worker(rank: int, world: int, init_path: str, result_queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(init_path).as_uri(),
            rank=rank,
            world_size=world,
            timeout=dt.timedelta(seconds=20),
        )
        # Deliberately distinct local RNGs: only rank 0 owns global-batch
        # sampling, so rank 1's stream cannot clone or perturb the draw.
        torch.manual_seed(123 if rank == 0 else 999)
        sampler = dist_utils.make_distributed_index_sampler(rank, world, torch.device("cpu"))
        batches = [sampler(10, 2) for _ in range(3)]
        try:
            sampler(3, 2)
        except ValueError as exc:
            capacity_error = str(exc)
        else:
            capacity_error = ""
        result_queue.put({
            "rank": rank,
            "batches": batches,
            "capacity_error": capacity_error,
        })
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
def test_distributed_sampler_partitions_one_unique_global_batch_per_step(tmp_path: Path) -> None:
    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / "gloo_sampler_init"
    processes = [
        context.Process(target=_gloo_sampler_worker, args=(rank, world, str(init_path), result_queue))
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
        pytest.fail("Gloo distributed index sampling deadlocked")

    assert [process.exitcode for process in processes] == [0, 0]
    results = []
    try:
        for _ in range(world):
            results.append(result_queue.get(timeout=5.0))
    except queue.Empty:
        pytest.fail("A Gloo worker exited without returning its sampler result")
    results.sort(key=lambda result: result["rank"])
    errors = [result["error"] for result in results if "error" in result]
    assert not errors, "\n".join(errors)

    generator = torch.Generator().manual_seed(123)
    expected_global = [torch.randperm(10, generator=generator)[:4].tolist() for _ in range(3)]
    for step, expected in enumerate(expected_global):
        combined = results[0]["batches"][step] + results[1]["batches"][step]
        assert combined == expected
        assert len(combined) == len(set(combined)) == 4
        assert results[0]["batches"][step] != results[1]["batches"][step]
    assert "global batch 4 exceeds 3 available episodes" in results[0]["capacity_error"]
    assert results[0]["capacity_error"] == results[1]["capacity_error"]


def _gloo_rng_worker(rank: int, world: int, init_path: str, result_queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(init_path).as_uri(),
            rank=rank,
            world_size=world,
            timeout=dt.timedelta(seconds=20),
        )
        torch.manual_seed(1000 + rank)
        expected = torch.get_rng_state().clone()
        gathered = dist_utils.gather_rank_rng_states(rank, world, torch.device("cpu"))
        container = [gathered]
        dist.broadcast_object_list(container, src=0)
        rank_states = container[0]
        torch.rand(17)  # move away from the checkpoint boundary
        dist_utils.restore_local_rng_state(rank_states[rank], torch.device("cpu"))
        result_queue.put({
            "rank": rank,
            "restored": torch.equal(torch.get_rng_state(), expected),
            "distinct": not torch.equal(rank_states[0]["cpu"], rank_states[1]["cpu"]),
        })
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
def test_rank_rng_checkpoint_gathers_and_restores_each_distinct_rank(tmp_path: Path) -> None:
    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / "gloo_rng_init"
    processes = [
        context.Process(target=_gloo_rng_worker, args=(rank, world, str(init_path), result_queue))
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
        pytest.fail("per-rank RNG gathering deadlocked")

    assert [process.exitcode for process in processes] == [0, 0]
    results = [result_queue.get(timeout=5.0) for _ in range(world)]
    errors = [result["error"] for result in results if "error" in result]
    assert not errors, "\n".join(errors)
    assert all(result["restored"] and result["distinct"] for result in results)


def test_local_rng_snapshot_restores_selected_cuda_generator(monkeypatch) -> None:
    device = torch.device("cuda:2")
    cuda_state = torch.tensor([9, 8, 7, 6], dtype=torch.uint8)
    get_calls: list[torch.device] = []
    set_calls: list[tuple[torch.Tensor, torch.device]] = []
    monkeypatch.setattr(
        dist_utils.torch.cuda,
        "get_rng_state",
        lambda selected: (get_calls.append(selected), cuda_state.clone())[1],
    )
    monkeypatch.setattr(
        dist_utils.torch.cuda,
        "set_rng_state",
        lambda state, selected: set_calls.append((state.clone(), selected)),
    )
    torch.manual_seed(321)
    expected_cpu = torch.get_rng_state().clone()

    snapshot = dist_utils.capture_local_rng_state(device)
    torch.rand(11)
    dist_utils.restore_local_rng_state(snapshot, device)

    assert torch.equal(torch.get_rng_state(), expected_cpu)
    assert get_calls == [device]
    assert len(set_calls) == 1
    assert torch.equal(set_calls[0][0], cuda_state)
    assert set_calls[0][1] == device


def test_init_distributed_uses_long_configurable_collective_timeout(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist_utils.dist, "init_process_group", lambda **kwargs: calls.append(kwargs))

    assert dist_utils.init_distributed() == (0, 2, 0, True)
    assert calls[-1]["timeout"] == dt.timedelta(hours=2)
    explicit = dt.timedelta(minutes=7)
    assert dist_utils.init_distributed(timeout=explicit) == (0, 2, 0, True)
    assert calls[-1]["timeout"] == explicit


def _gloo_rank0_work_sync_worker(
    rank: int,
    world: int,
    init_path: str,
    daily_ready_path: str,
    ssl_ready_path: str,
    result_queue,
) -> None:
    """Exercise both trainer rendezvous hooks with real Gloo collectives."""

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(init_path).as_uri(),
            rank=rank,
            world_size=world,
            timeout=dt.timedelta(seconds=20),
        )
        # Import inside the spawned process so replacing the two local helper
        # functions cannot affect the parent pytest process.
        import rl_quant.training.daily_policy as daily_training
        from rl_quant.training.context_pretrain import train_context_encoder

        def fake_stack(_episodes, indices, _device):
            batch = len(indices)
            return {
                "ret_valid": torch.ones(batch, 1, 2, dtype=torch.bool),
                "score_mask": torch.ones(batch, 1, dtype=torch.bool),
            }

        def fake_rollout(policy, batch, _cost, **_kwargs):
            rows = batch["score_mask"].shape[0]
            value = policy.weight.expand(rows, 1)
            gate = policy.weight.sigmoid().expand(rows, 1)
            zero = value * 0.0
            return value, gate, zero, None, None, zero, None

        daily_training._stack = fake_stack
        daily_training._daily_rollout = fake_rollout
        daily_ready = Path(daily_ready_path)
        base_sampler = dist_utils.make_distributed_index_sampler(rank, world, torch.device("cpu"))
        sample_calls = 0
        daily_entered_early = False

        def sampler(n_items: int, local_batch: int) -> list[int]:
            nonlocal sample_calls, daily_entered_early
            sample_calls += 1
            if rank == 1 and sample_calls == 2:
                daily_entered_early = not daily_ready.exists()
            return base_sampler(n_items, local_batch)

        def on_eval(step, *_args) -> None:
            if rank == 0 and step == 1:
                time.sleep(0.35)
                daily_ready.touch()

        policy = _ScalarPolicy()
        daily_training.train_daily_policy(
            policy,
            [{}, {}, {}, {}],
            steps=2,
            batch_days=1,
            eval_every=1,
            val_eps=[],
            device=torch.device("cpu"),
            on_eval=on_eval,
            episode_sampler=sampler,
            grad_reduce=dist_utils.make_grad_reduce(world),
            is_main=(rank == 0),
            sync_after_eval=dist.barrier,
        )

        # Stage 1 has the same rank-0-only pause at checkpoint boundaries. The
        # second gradient reduction is the next distributed training
        # collective and therefore must not begin until rank 0 finishes I/O.
        ssl_ready = Path(ssl_ready_path)
        reduce_calls = 0
        ssl_entered_early = False
        base_reduce = dist_utils.make_grad_reduce(world)

        def checked_reduce(parameters) -> None:
            nonlocal reduce_calls, ssl_entered_early
            reduce_calls += 1
            if rank == 1 and reduce_calls == 2:
                ssl_entered_early = not ssl_ready.exists()
            base_reduce(parameters)

        def on_checkpoint(step, _optimizer) -> None:
            if rank == 0 and step == 1:
                time.sleep(0.35)
                ssl_ready.touch()

        day = {
            "bars": torch.zeros(2, 1, 1),
            "bar_mask": torch.ones(2, 1, dtype=torch.bool),
            "cov_blocks": torch.zeros(1, 2, 1),
            "ret": torch.tensor([[0.0, 0.01]]),
            "ret_valid": torch.ones(1, 2, dtype=torch.bool),
        }
        encoder = _TinyContextEncoder()
        head = torch.nn.Linear(1, 2, bias=False)
        train_context_encoder(
            encoder,
            head,
            [day],
            device=torch.device("cpu"),
            steps=2,
            batch_size=1,
            accum_steps=1,
            checkpoint_every=1,
            on_checkpoint=on_checkpoint,
            grad_reduce=checked_reduce,
            sync_after_checkpoint=dist.barrier,
        )
        dist.barrier()
        result_queue.put({
            "rank": rank,
            "daily_entered_early": daily_entered_early,
            "ssl_entered_early": ssl_entered_early,
        })
    except BaseException:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_rank0_validation_and_checkpoint_work_rendezvous_before_next_collective(tmp_path: Path) -> None:
    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / "gloo_rank0_work_init"
    daily_ready = tmp_path / "daily_validation_complete"
    ssl_ready = tmp_path / "ssl_checkpoint_complete"
    processes = [
        context.Process(
            target=_gloo_rank0_work_sync_worker,
            args=(rank, world, str(init_path), str(daily_ready), str(ssl_ready), result_queue),
        )
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
        pytest.fail("rank-0-only validation/checkpoint rendezvous deadlocked")

    assert [process.exitcode for process in processes] == [0, 0]
    results = []
    try:
        for _ in range(world):
            results.append(result_queue.get(timeout=5.0))
    except queue.Empty:
        pytest.fail("A Gloo worker exited without returning its synchronization result")
    results.sort(key=lambda result: result["rank"])
    errors = [result["error"] for result in results if "error" in result]
    assert not errors, "\n".join(errors)
    assert daily_ready.exists()
    assert ssl_ready.exists()
    assert all(not result["daily_entered_early"] for result in results)
    assert all(not result["ssl_entered_early"] for result in results)
