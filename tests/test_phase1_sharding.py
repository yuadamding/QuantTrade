"""Fail-closed geometry checks for the external Phase-1 distributed driver."""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
import queue
import time
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DRIVER_PATH = Path(__file__).resolve().parents[2] / "training" / "train_phase1.py"
SPEC = importlib.util.spec_from_file_location("phase1_sharding_driver", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def _daily_raw(*, episode_len: int = 252, exec_delay: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        horizon_mode="daily_raw",
        episode_len=episode_len,
        exec_delay=exec_delay,
    )


def test_daily_raw_rejects_distributed_training_without_full_context_reassembly() -> None:
    with pytest.raises(SystemExit, match=r"shorten the chronological history to as few as 210") as exc:
        driver._validate_daily_raw_rank_history(_daily_raw(), n_train_days=840, world=4)

    message = str(exc.value)
    assert "requires full-context reassembly" in message
    assert "refusing date-sharded policy training" in message


def test_daily_raw_rank_history_accepts_global_reassembly_and_checks_full_split_capacity() -> None:
    # TOP2000/TOP50 have 840 train days. Reassembly restores all 840 on each
    # policy rank, so the local Stage-1 shard length is irrelevant to L=252.
    driver._validate_daily_raw_rank_history(
        _daily_raw(), n_train_days=840, world=4, full_context_reassembly=True
    )
    driver._validate_daily_raw_rank_history(_daily_raw(), n_train_days=254, world=1)
    with pytest.raises(SystemExit, match=r"full training split has only 253"):
        driver._validate_daily_raw_rank_history(_daily_raw(), n_train_days=253, world=1)


def test_non_daily_mode_is_not_subject_to_daily_raw_history_guard() -> None:
    intraday = SimpleNamespace(horizon_mode="intraday", episode_len=252, exec_delay=1)
    driver._validate_daily_raw_rank_history(intraday, n_train_days=4, world=4)


def test_single_gpu_reassembly_is_an_exact_no_op() -> None:
    source = [{"date": "d0"}]
    encoded = [{"date": "d0", "market": torch.tensor([1.0])}]

    rebuilt = driver._reassemble_daily_raw_training_context(
        source, encoded, shard_lo=0, device=torch.device("cpu")
    )

    assert rebuilt is encoded


def _encoded_day(index: int) -> dict:
    return {
        "date": f"d{index}",
        "market": torch.tensor([index, index + 0.5], dtype=torch.bfloat16),
        "per_stock": torch.full((3, 2), float(index), dtype=torch.bfloat16),
        "avail": torch.tensor([True, index % 2 == 0, True]),
        "news_raw": torch.full((3, 1, 1), float(index)),
        "news_mask": torch.tensor([[True], [False], [True]]),
        "day_close": torch.tensor([1.0, 10.0 + index, 20.0 + index]),
    }


def _valid_context_fields(n_days: int, *, actions: int = 3, context_dim: int = 2) -> dict[str, torch.Tensor]:
    day_index = torch.arange(n_days, dtype=torch.float32)
    return {
        "market": day_index[:, None].expand(n_days, context_dim).clone(),
        "per_stock": day_index[:, None, None].expand(n_days, actions, context_dim).clone(),
        "avail": torch.ones(n_days, actions, dtype=torch.bool),
        "news_raw": torch.zeros(n_days, actions, driver.MAX_NEWS, driver.NEWS_RAW_DIM),
        "news_mask": torch.zeros(n_days, actions, driver.MAX_NEWS, dtype=torch.bool),
        "day_close": torch.arange(n_days * actions, dtype=torch.float32).reshape(n_days, actions) + 1.0,
    }


def test_daily_raw_context_views_preserve_full_train_and_causal_oos_prefixes() -> None:
    days = [{"date": f"d{index}"} for index in range(10)]
    fields = _valid_context_fields(len(days))

    train, val, test = driver._slice_daily_raw_context(
        days[:6], days[6:8], days[8:], fields, daily_lookback=4, is_main=True
    )

    assert [day["date"] for day in train] == [f"d{index}" for index in range(6)]
    assert [day["date"] for day in val] == [f"d{index}" for index in range(3, 8)]
    assert [day["date"] for day in test] == [f"d{index}" for index in range(5, 10)]
    assert [float(day["market"][0]) for day in val] == list(map(float, range(3, 8)))

    worker_train, worker_val, worker_test = driver._slice_daily_raw_context(
        days[:6], days[6:8], days[8:], fields, daily_lookback=4, is_main=False
    )
    assert [day["date"] for day in worker_train] == [f"d{index}" for index in range(6)]
    assert worker_val == worker_test == []


def test_daily_eod_context_cache_round_trip_and_identity_validation(tmp_path: Path) -> None:
    dates = [f"d{index}" for index in range(4)]
    fields = _valid_context_fields(len(dates))
    path = driver._daily_eod_context_cache_path(tmp_path, "ctx-123", "float32")

    driver._save_daily_eod_context_cache(
        path,
        context_hash="ctx-123",
        dates=dates,
        storage_dtype="float32",
        fields=fields,
    )
    loaded, reason = driver._load_daily_eod_context_cache(
        path,
        context_hash="ctx-123",
        dates=dates,
        context_dim=2,
        storage_dtype="float32",
    )

    assert reason == "ok"
    assert loaded is not None
    assert set(loaded) == set(driver._DAILY_RAW_REASSEMBLY_FIELDS)
    assert torch.equal(loaded["per_stock"], fields["per_stock"])

    bad_hash, reason = driver._load_daily_eod_context_cache(
        path,
        context_hash="other-context",
        dates=dates,
        context_dim=2,
        storage_dtype="float32",
    )
    assert bad_hash is None and reason == "context hash mismatch"

    bad_dates, reason = driver._load_daily_eod_context_cache(
        path,
        context_hash="ctx-123",
        dates=[*dates[:-1], "different-date"],
        context_dim=2,
        storage_dtype="float32",
    )
    assert bad_dates is None and reason == "ordered date sequence mismatch"

    payload = torch.load(path, weights_only=True, map_location="cpu")
    payload["fields"] = dict(payload["fields"])
    payload["fields"].pop("avail")
    bad_fields, reason = driver._validate_daily_eod_context_payload(
        payload,
        context_hash="ctx-123",
        dates=dates,
        context_dim=2,
        storage_dtype="float32",
    )
    assert bad_fields is None and reason == "field set mismatch"


def test_disabled_news_eod_cache_and_overlay_keep_scalar_storage(tmp_path: Path) -> None:
    assert set(driver._daily_raw_gather_fields(no_news=False)) == set(driver._DAILY_RAW_REASSEMBLY_FIELDS)
    assert set(driver._daily_raw_gather_fields(no_news=True)) == {
        "market", "per_stock", "avail", "day_close",
    }

    dates = [f"d{index}" for index in range(6)]
    fields = _valid_context_fields(len(dates), actions=5)
    fields.update(driver._scalar_disabled_news_fields(len(dates), actions=5))
    path = driver._daily_eod_context_cache_path(tmp_path, "ctx-no-news", "float32")

    driver._save_daily_eod_context_cache(
        path,
        context_hash="ctx-no-news",
        dates=dates,
        storage_dtype="float32",
        fields=fields,
    )
    loaded, reason = driver._load_daily_eod_context_cache(
        path,
        context_hash="ctx-no-news",
        dates=dates,
        context_dim=2,
        storage_dtype="float32",
    )

    assert reason == "ok"
    assert loaded is not None
    for key in ("news_raw", "news_mask"):
        assert loaded[key].untyped_storage().nbytes() == loaded[key].element_size()
        assert not bool(loaded[key].any())

    overlaid = driver._overlay_daily_raw_context([{"date": date} for date in dates], loaded)
    for day in overlaid:
        assert day["news_raw"].shape == (5, driver.MAX_NEWS, driver.NEWS_RAW_DIM)
        assert day["news_mask"].shape == (5, driver.MAX_NEWS)
        assert day["news_raw"].untyped_storage().data_ptr() == loaded["news_raw"].untyped_storage().data_ptr()
        assert day["news_mask"].untyped_storage().data_ptr() == loaded["news_mask"].untyped_storage().data_ptr()


def test_stage2_policy_initialization_is_independent_of_prior_cache_rng_use() -> None:
    torch.manual_seed(1)
    torch.rand(7)
    driver._seed_stage2_policy_initialization(19)
    cache_miss_init = torch.nn.Linear(11, 5).state_dict()

    torch.manual_seed(999)
    torch.rand(10_000)
    driver._seed_stage2_policy_initialization(19)
    cache_hit_init = torch.nn.Linear(11, 5).state_dict()

    assert cache_miss_init.keys() == cache_hit_init.keys()
    assert all(torch.equal(cache_miss_init[key], cache_hit_init[key]) for key in cache_miss_init)


def _gloo_context_worker(rank: int, world: int, init_path: str, result_queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(init_path).as_uri(),
            rank=rank,
            world_size=world,
            timeout=dt.timedelta(seconds=20),
        )
        driver._DIST.update(rank=rank, world=world, local_rank=rank, is_dist=True, grad_reduce=None)
        source = [{"date": f"d{index}", "source_index": index} for index in range(8)]
        lo, hi = len(source) * rank // world, len(source) * (rank + 1) // world
        rebuilt = driver._reassemble_daily_raw_training_context(
            source,
            [_encoded_day(index) for index in range(lo, hi)],
            shard_lo=lo,
            device=torch.device("cpu"),
        )
        result_queue.put({
            "rank": rank,
            "dates": [day["date"] for day in rebuilt],
            "source_indices": [day["source_index"] for day in rebuilt],
            "market": [day["market"].float().tolist() for day in rebuilt],
            "per_stock_dtype": str(rebuilt[0]["per_stock"].dtype),
            "day_close": [day["day_close"].tolist() for day in rebuilt],
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
def test_distributed_reassembly_restores_identical_full_chronology_on_every_rank(tmp_path: Path) -> None:
    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / "gloo_context_init"
    processes = [
        context.Process(target=_gloo_context_worker, args=(rank, world, str(init_path), result_queue))
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
        pytest.fail("Gloo daily-context reassembly deadlocked")

    assert [process.exitcode for process in processes] == [0, 0]
    results = []
    try:
        for _ in range(world):
            results.append(result_queue.get(timeout=5.0))
    except queue.Empty:
        pytest.fail("A Gloo worker exited without returning its reassembly result")
    results.sort(key=lambda result: result["rank"])
    errors = [result["error"] for result in results if "error" in result]
    assert not errors, "\n".join(errors)

    assert results[0] == {**results[1], "rank": 0}
    assert results[0]["dates"] == [f"d{index}" for index in range(8)]
    assert results[0]["source_indices"] == list(range(8))
    assert results[0]["per_stock_dtype"] == "torch.bfloat16"
    assert results[0]["market"][6] == [6.0, 6.5]
    assert results[0]["day_close"][7] == [1.0, 17.0, 27.0]
