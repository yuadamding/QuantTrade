"""Concurrency and cache-reuse checks for the external Phase-1 driver."""
from __future__ import annotations

import importlib.util
import multiprocessing as multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch


DRIVER_PATH = Path(__file__).resolve().parents[2] / "training" / "train_phase1.py"
SPEC = importlib.util.spec_from_file_location("phase1_shared_artifact_driver", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def _claim_worker(path: str, hold_seconds: float, ready, results) -> None:
    driver._DIST.update(rank=0, world=1, local_rank=0, is_dist=False, grad_reduce=None)
    with driver._exclusive_artifact_claim(Path(path), "test artifact"):
        entered = time.monotonic()
        ready.set()
        time.sleep(hold_seconds)
        leaving = time.monotonic()
    results.put((entered, leaving))


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires POSIX flock/fork")
def test_shared_artifact_claim_serializes_independent_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    results = context.Queue()
    artifact = tmp_path / "shared.pt"
    first = context.Process(target=_claim_worker, args=(str(artifact), 0.25, ready, results))
    first.start()
    assert ready.wait(timeout=5)
    second = context.Process(target=_claim_worker, args=(str(artifact), 0.0, context.Event(), results))
    second.start()
    intervals = [results.get(timeout=5), results.get(timeout=5)]
    first.join(timeout=5)
    second.join(timeout=5)
    assert first.exitcode == 0
    assert second.exitcode == 0
    intervals.sort()
    assert intervals[1][0] >= intervals[0][1]


def test_shared_artifact_claim_cleans_only_dead_owned_temps_for_exact_path(tmp_path: Path) -> None:
    driver._DIST.update(rank=0, world=1, local_rank=0, is_dist=False, grad_reduce=None)
    artifact = tmp_path / "shared.pt"
    host = driver._artifact_host_label()
    dead_pid = 999_999_999
    assert not driver._local_pid_is_alive(dead_pid)
    orphan = tmp_path / f".{artifact.name}.{host}.{dead_pid}.orphan.tmp"
    live = tmp_path / f".{artifact.name}.{host}.{os.getpid()}.live.tmp"
    other = tmp_path / f".other.pt.{host}.{dead_pid}.other.tmp"
    for path in (orphan, live, other):
        path.write_bytes(b"partial")

    with driver._exclusive_artifact_claim(artifact, "test artifact"):
        assert not orphan.exists()
        assert live.exists()
        assert other.exists()


def _encoded_days(days: list[dict], context_dim: int = 2, actions: int = 3) -> list[dict]:
    return [
        {
            "market": torch.full((context_dim,), float(index)),
            "per_stock": torch.full((actions, context_dim), float(index)),
            "avail": torch.ones(actions, dtype=torch.bool),
            "day_close": torch.full((actions,), 100.0 + index),
        }
        for index, _day in enumerate(days)
    ]


def test_single_process_daily_context_builds_reuses_and_recovers_invalid_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver._DIST.update(rank=0, world=1, local_rank=0, is_dist=False, grad_reduce=None)
    train = [{"date": "d0"}, {"date": "d1"}]
    validation = [{"date": "d2"}]
    test = [{"date": "d3"}]
    all_days = [*train, *validation, *test]
    calls = {"count": 0}

    def encode(*_args, **_kwargs):
        calls["count"] += 1
        return _encoded_days(all_days)

    monkeypatch.setattr(driver, "encode_days", encode)
    args = SimpleNamespace(
        context_storage_dtype="float32",
        d_model=2,
        no_news=True,
        amp=False,
        daily_lookback=2,
    )
    kwargs = dict(
        a=args,
        device=torch.device("cpu"),
        encode_batch=2,
        context_hash="ctx",
        context_artifact_id="encoder-generation-1",
        context_dir=tmp_path,
    )

    first = driver._single_process_daily_raw_context(
        object(), train, validation, test, use_cached=False, **kwargs
    )
    second = driver._single_process_daily_raw_context(
        object(), train, validation, test, use_cached=True, **kwargs
    )
    assert calls["count"] == 1
    assert [len(split) for split in first] == [2, 2, 2]
    assert [len(split) for split in second] == [2, 2, 2]

    cache_path = driver._daily_eod_context_cache_path(tmp_path, "ctx", "float32")
    cache_path.write_bytes(b"truncated")
    driver._single_process_daily_raw_context(
        object(), train, validation, test, use_cached=True, **kwargs
    )
    assert calls["count"] == 2
    loaded, reason = driver._load_daily_eod_context_cache(
        cache_path,
        context_hash="ctx",
        dates=[day["date"] for day in all_days],
        context_dim=2,
        storage_dtype="float32",
        context_artifact_id="encoder-generation-1",
    )
    assert reason == "ok"
    assert loaded is not None
    assert not list(tmp_path.glob("*.tmp"))


def test_eod_cache_is_bound_to_exact_encoder_artifact_identity(tmp_path: Path) -> None:
    fields = driver._stack_single_process_daily_fields(
        _encoded_days([{"date": "d0"}]), no_news=True
    )
    path = tmp_path / "context.daily_eod.float32.pt"
    driver._save_daily_eod_context_cache(
        path,
        context_hash="ctx",
        dates=["d0"],
        storage_dtype="float32",
        fields=fields,
        context_artifact_id="generation-a",
    )
    loaded, reason = driver._load_daily_eod_context_cache(
        path,
        context_hash="ctx",
        dates=["d0"],
        context_dim=2,
        storage_dtype="float32",
        context_artifact_id="generation-b",
    )
    assert loaded is None
    assert reason == "encoder artifact identity mismatch"


def test_policy_resume_rejects_a_different_encoder_generation(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"different frozen features.*--fresh"):
        driver._require_matching_policy_context(
            {"phase": "policy", "context_hash": "ctx", "context_artifact_id": "generation-a"},
            "generation-b",
            context_hash="ctx",
            context_path=tmp_path / "ctx.pt",
        )

    driver._require_matching_policy_context(
        {"phase": "policy", "context_hash": "ctx", "context_artifact_id": "generation-a"},
        "generation-a",
        context_hash="ctx",
        context_path=tmp_path / "ctx.pt",
    )


def test_policy_resume_recovers_exact_immutable_generation_and_republishes_alias(tmp_path: Path) -> None:
    context_path = tmp_path / "ctx.pt"
    original = torch.nn.Linear(3, 2)
    driver.save_context(context_path, original, seed=17, ctx_hash="ctx", artifact_id="generation-a")
    generation_path = driver._context_generation_path(context_path, "generation-a")
    context_path.unlink()
    assert generation_path.exists() and not context_path.exists()

    restored = torch.nn.Linear(3, 2)
    resume = {"phase": "policy", "context_hash": "ctx", "context_artifact_id": "generation-a"}
    state, artifact_id, republish, reason = driver._resolve_policy_resume_context_artifact(
        resume,
        restored,
        context_hash="ctx",
        context_path=context_path,
        cached_state_dict=None,
        cached_artifact_id=None,
    )
    assert reason == "ok" and state is not None
    assert artifact_id == "generation-a" and republish
    restored.load_state_dict(state)
    with driver._exclusive_artifact_claim(context_path, "encoder context"):
        driver.save_context(context_path, restored, seed=17, ctx_hash="ctx", artifact_id=artifact_id)

    alias_state, alias_id, alias_reason = driver._load_context_artifact(context_path, "ctx")
    assert alias_reason == "ok" and alias_id == "generation-a" and alias_state is not None
    assert torch.equal(alias_state["weight"], original.state_dict()["weight"])


def test_policy_resume_never_bypasses_valid_different_alias_with_old_generation(tmp_path: Path) -> None:
    context_path = tmp_path / "ctx.pt"
    old = torch.nn.Linear(3, 2)
    current = torch.nn.Linear(3, 2)
    driver.save_context(context_path, old, seed=17, ctx_hash="ctx", artifact_id="generation-a")
    driver.save_context(context_path, current, seed=17, ctx_hash="ctx", artifact_id="generation-b")
    cached_state, cached_id, reason = driver._load_context_artifact(context_path, "ctx")
    assert reason == "ok" and cached_state is not None and cached_id == "generation-b"

    with pytest.raises(SystemExit, match=r"different frozen features.*--fresh"):
        driver._resolve_policy_resume_context_artifact(
            {"phase": "policy", "context_hash": "ctx", "context_artifact_id": "generation-a"},
            torch.nn.Linear(3, 2),
            context_hash="ctx",
            context_path=context_path,
            cached_state_dict=cached_state,
            cached_artifact_id=cached_id,
        )


def test_policy_resume_generation_recovery_requires_hash_and_architecture_match(tmp_path: Path) -> None:
    context_path = tmp_path / "ctx.pt"
    original = torch.nn.Linear(3, 2)
    driver.save_context(context_path, original, seed=17, ctx_hash="ctx", artifact_id="generation-a")
    context_path.unlink()
    resume = {"phase": "policy", "context_hash": "ctx", "context_artifact_id": "generation-a"}

    state, artifact_id, republish, reason = driver._resolve_policy_resume_context_artifact(
        resume,
        torch.nn.Linear(4, 2),
        context_hash="ctx",
        context_path=context_path,
        cached_state_dict=None,
        cached_artifact_id=None,
    )
    assert state is None and artifact_id is None and not republish
    assert "architecture mismatch" in reason

    state, artifact_id, republish, reason = driver._resolve_policy_resume_context_artifact(
        {**resume, "context_hash": "other"},
        torch.nn.Linear(3, 2),
        context_hash="ctx",
        context_path=context_path,
        cached_state_dict=None,
        cached_artifact_id=None,
    )
    assert state is None and artifact_id is None and not republish
    assert "context hash" in reason


def test_context_refresh_preserves_immutable_encoder_generations(tmp_path: Path) -> None:
    context_path = tmp_path / "ctx.pt"
    first = torch.nn.Linear(3, 2)
    second = torch.nn.Linear(3, 2)

    driver.save_context(context_path, first, seed=17, ctx_hash="ctx", artifact_id="generation-a")
    first_path = driver._context_generation_path(context_path, "generation-a")
    assert first_path.exists()
    first_state, first_id, first_reason = driver._load_context_artifact(first_path, "ctx")
    assert first_reason == "ok" and first_id == "generation-a" and first_state is not None

    driver.save_context(context_path, second, seed=17, ctx_hash="ctx", artifact_id="generation-b")
    second_path = driver._context_generation_path(context_path, "generation-b")
    current_state, current_id, current_reason = driver._load_context_artifact(context_path, "ctx")

    assert first_path.exists() and second_path.exists()
    assert current_reason == "ok" and current_id == "generation-b" and current_state is not None
    assert not torch.equal(first_state["weight"], current_state["weight"])


def test_context_generation_identity_cannot_be_overwritten(tmp_path: Path) -> None:
    context_path = tmp_path / "ctx.pt"
    first = torch.nn.Linear(3, 2)
    different = torch.nn.Linear(3, 2)
    driver.save_context(context_path, first, seed=17, ctx_hash="ctx", artifact_id="fixed-generation")

    with pytest.raises(RuntimeError, match="refusing to overwrite immutable encoder generation"):
        driver.save_context(
            context_path,
            different,
            seed=17,
            ctx_hash="ctx",
            artifact_id="fixed-generation",
        )

    stored, artifact_id, reason = driver._load_context_artifact(
        driver._context_generation_path(context_path, "fixed-generation"), "ctx"
    )
    assert reason == "ok" and artifact_id == "fixed-generation" and stored is not None
    assert torch.equal(stored["weight"], first.state_dict()["weight"])
