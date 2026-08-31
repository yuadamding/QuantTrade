"""Auditability regressions for the external Phase-1 training driver."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch


DRIVER_PATH = Path(__file__).resolve().parents[2] / "training" / "train_phase1.py"
SWEEP_PATH = Path(__file__).resolve().parents[2] / "training" / "sweep_phase1.py"
if not DRIVER_PATH.is_file() or not SWEEP_PATH.is_file():
    pytest.skip(
        "external Phase-1 reporting drivers are not packaged in a clean checkout",
        allow_module_level=True,
    )
SPEC = importlib.util.spec_from_file_location("phase1_reporting_driver", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

SWEEP_SPEC = importlib.util.spec_from_file_location("phase1_sweep_driver", SWEEP_PATH)
assert SWEEP_SPEC is not None and SWEEP_SPEC.loader is not None
sweep = importlib.util.module_from_spec(SWEEP_SPEC)
SWEEP_SPEC.loader.exec_module(sweep)
TOP2000_LAUNCHER = Path(__file__).resolve().parents[2] / "training" / "run_phase1_top2000_4xh100.sh"
ROOT_LAUNCHER = Path(__file__).resolve().parents[2] / "run.sh"


def _result(seed: int, values: list[float], *, reportable: int | None = None,
            decision_ids: list[str] | None = None) -> dict:
    count = len(values)
    reportable = count if reportable is None else reportable
    return {
        "schema_version": 1,
        "seed": seed,
        "returns": values,
        "best_validation_mean": 0.001 * seed,
        "evaluation": {
            "label_blocks": count,
            "reportable_blocks": reportable,
            "label_reportable_fraction": reportable / count,
            "decision_ids": decision_ids or [f"d{index}" for index in range(count)],
        },
        "telemetry": {},
        "selection_source": "validation_only",
        "test_used_for_selection": False,
    }


def test_seed_returns_are_paired_by_decision_not_pooled() -> None:
    paired, normalized, certified, note = driver._paired_seed_returns(
        {11: _result(11, [1.0, 3.0, 5.0]), 7: _result(7, [3.0, 1.0, 7.0])}
    )

    assert paired == [2.0, 2.0, 6.0]
    assert [result["seed"] for result in normalized] == [7, 11]
    assert certified
    assert note == "complete aligned fixed-label coverage"

    paired, _, certified, note = driver._paired_seed_returns(
        {1: _result(1, [1.0, 2.0]), 2: _result(2, [1.0])}
    )
    assert paired == []
    assert not certified
    assert "counts differ" in note

    paired, _, certified, note = driver._paired_seed_returns(
        {
            1: _result(1, [1.0, 2.0], decision_ids=["a", "b"]),
            2: _result(2, [4.0, 3.0], decision_ids=["b", "a"]),
        }
    )
    assert paired == [2.0, 3.0]
    assert certified
    assert note == "complete aligned fixed-label coverage"

    paired, _, certified, note = driver._paired_seed_returns(
        {
            1: _result(1, [1.0, 2.0], decision_ids=["a", "b"]),
            2: _result(2, [3.0, 4.0], decision_ids=["a", "c"]),
        }
    )
    assert paired == []
    assert not certified
    assert "identifier set differs" in note


def test_incomplete_or_legacy_alignment_cannot_be_certified() -> None:
    paired, _, certified, note = driver._paired_seed_returns(
        {1: _result(1, [0.1, 0.2, 0.3], reportable=2), 2: [0.2, 0.1, 0.4]}
    )

    assert paired == pytest.approx([0.15, 0.15, 0.35])
    assert not certified
    assert "legacy" in note


def test_low_reportable_fraction_is_a_verdict_gate_not_an_alignment_filter(monkeypatch) -> None:
    results = {
        1: _result(1, [0.010, 0.020, 0.015], reportable=2),
        2: _result(2, [0.012, 0.018, 0.014], reportable=2),
    }
    paired, _, certified, _ = driver._paired_seed_returns(results)
    assert len(paired) == 3
    assert certified

    baseline = [{"ret": torch.zeros(3, 2), "ret_valid": torch.ones(3, 2, dtype=torch.bool),
                 "availability": torch.ones(3, 2, dtype=torch.bool)}]
    monkeypatch.setattr(driver, "block_bootstrap_confidence_interval", lambda *args, **kwargs: (0.001, 0.02))
    monkeypatch.setattr(driver, "effective_sample_size", lambda values: float(len(values)))
    monkeypatch.setattr(driver, "deflated_sharpe_ratio", lambda *args, **kwargs: 0.99)

    summary = driver.verdict("low-coverage", results, baseline, n_seeds=2, n_trials=2, cost=0.0)

    assert summary["alignment_certified"]
    assert not summary["test_coverage_eligible"]
    assert not summary["statistically_positive"]


def test_buy_and_hold_baseline_pays_each_round_trip() -> None:
    item = {
        "ret": torch.tensor([[0.0, 0.10], [0.0, 0.10], [0.0, 0.10]]),
        "ret_valid": torch.tensor([[True, True], [True, False], [True, True]]),
        "availability": torch.ones(3, 2, dtype=torch.bool),
    }

    cash, buy_hold = driver.round_trip_cost_paid_baselines([item], cost=0.01)

    # One continuously held portfolio: gross .20 minus one .01 entry and one .01 exit, over two labeled marks.
    assert cash == 0.0
    assert abs(buy_hold - 0.09) < 1e-7


def test_buy_and_hold_baseline_is_one_fixed_date_portfolio_not_per_asset_means() -> None:
    item = {
        "ret": torch.tensor([[0.0, 0.20, -0.10], [0.0, 0.00, 0.10]]),
        "ret_valid": torch.tensor([[True, True, True], [True, False, True]]),
        "availability": torch.ones(2, 3, dtype=torch.bool),
    }

    _, buy_hold = driver.round_trip_cost_paid_baselines([item], cost=0.0)

    # Equal 50/50 entry earns .05, then drifts to 4/7 and 3/7. The missing
    # stock receives zero credit on date two while the surviving stock earns
    # .10, so both policy-aligned labeled dates remain in the denominator.
    assert buy_hold == pytest.approx((0.05 + 3.0 / 7.0 * 0.10) / 2.0)


def test_daily_raw_baseline_requires_decision_time_eod_availability() -> None:
    # Stock 1 has a +100% forward price move for decision d0, but it is not in
    # the feasible EOD action set until d1.  A future finite price must not make
    # the pre-IPO/unavailable d0 observation a benchmark candidate.
    closes = torch.tensor([
        [float("nan"), 1.0, 1.0],
        [float("nan"), 1.0, 1.0],
        [float("nan"), 2.0, 1.0],
        [float("nan"), 2.0, 1.0],
    ])
    days = []
    for index in range(len(closes)):
        eod = torch.tensor([True, index >= 1, True])
        days.append({
            "day_close": closes[index],
            # Exercise a half-day-like padded suffix: the session-close block is
            # block 0 and the post-close row is all unavailable.  Baseline
            # feasibility must select the declared close, not tensor[-1].
            "avail": torch.stack((eod, torch.zeros(3, dtype=torch.bool))),
            "session_close_block": torch.tensor(0),
        })

    items = driver.baseline_items(days, "daily_raw", episode_len=252, horizon=21, exec_delay=1)
    valid = items[0]["ret_valid"]
    availability = items[0]["availability"]

    assert valid.shape == (2, 3)
    assert bool(valid[0, 1])
    assert bool(valid[1, 1])
    assert not bool(availability[0, 1])
    _, buy_hold = driver.round_trip_cost_paid_baselines(items, cost=0.0)
    assert buy_hold == 0.0


def test_verdict_uses_paired_series_and_never_selects_test_seed(monkeypatch, capsys) -> None:
    results = {
        1: _result(1, [0.010, -0.005, 0.020, 0.001]),
        2: _result(2, [0.020, -0.001, 0.010, 0.005]),
    }
    baseline = [{"ret": torch.zeros(4, 2), "ret_valid": torch.ones(4, 2, dtype=torch.bool),
                 "availability": torch.ones(4, 2, dtype=torch.bool)}]
    monkeypatch.setattr(driver, "block_bootstrap_confidence_interval", lambda *args, **kwargs: (0.001, 0.02))
    monkeypatch.setattr(driver, "effective_sample_size", lambda values: float(len(values)))
    monkeypatch.setattr(driver, "deflated_sharpe_ratio", lambda *args, **kwargs: 0.99)

    summary = driver.verdict("test", results, baseline, n_seeds=2, n_trials=3, cost=0.0)
    output = capsys.readouterr().out.lower()

    expected = [(a + b) / 2 for a, b in zip(results[1]["returns"], results[2]["returns"])]
    assert abs(summary["paired_mean"] - sum(expected) / len(expected)) < 1e-12
    assert summary["positive"]
    assert "test results selected no seed" in output
    assert "best seed" not in output


def test_missing_requested_seed_replication_cannot_be_promoted(monkeypatch) -> None:
    results = {
        17: _result(17, [0.010, 0.020, 0.015, 0.012]),
        18: _result(18, [0.012, 0.018, 0.014, 0.011]),
    }
    baseline = [{"ret": torch.zeros(4, 2), "ret_valid": torch.ones(4, 2, dtype=torch.bool),
                 "availability": torch.ones(4, 2, dtype=torch.bool)}]
    monkeypatch.setattr(driver, "block_bootstrap_confidence_interval", lambda *args, **kwargs: (0.001, 0.02))
    monkeypatch.setattr(driver, "effective_sample_size", lambda values: float(len(values)))
    monkeypatch.setattr(driver, "deflated_sharpe_ratio", lambda *args, **kwargs: 0.99)

    summary = driver.verdict("incomplete", results, baseline, n_seeds=3, n_trials=3, cost=0.0)

    assert not summary["replications_complete"]
    assert not summary["positive"]
    assert summary["requested_seed_count"] == 3
    assert "completed 2/3" in summary["alignment_note"]


def test_statistical_success_cannot_override_failed_data_provenance(monkeypatch, capsys) -> None:
    results = {
        1: _result(1, [0.010, 0.020, 0.015, 0.012]),
        2: _result(2, [0.012, 0.018, 0.014, 0.011]),
    }
    baseline = [{"ret": torch.zeros(4, 2), "ret_valid": torch.ones(4, 2, dtype=torch.bool),
                 "availability": torch.ones(4, 2, dtype=torch.bool)}]
    monkeypatch.setattr(driver, "block_bootstrap_confidence_interval", lambda *args, **kwargs: (0.001, 0.02))
    monkeypatch.setattr(driver, "effective_sample_size", lambda values: float(len(values)))
    monkeypatch.setattr(driver, "deflated_sharpe_ratio", lambda *args, **kwargs: 0.99)

    summary = driver.verdict(
        "development-only",
        results,
        baseline,
        n_seeds=2,
        n_trials=2,
        cost=0.0,
        promotion_eligible=False,
        promotion_note="future-selected universe",
    )

    assert summary["statistically_positive"]
    assert not summary["positive"]
    assert not summary["promotion_eligible"]
    output = capsys.readouterr().out
    assert "PROVENANCE GATE FAILED" in output
    assert "statistically positive but NOT PROMOTABLE" in output


def test_aggregate_selects_only_the_explicit_requested_seed_range(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    for seed in (17, 18, 19, 98, 99):
        state_path = tmp_path / f"seed_{seed}" / "state.pt"
        state_path.parent.mkdir()
        torch.save(
            {
                "cfg": "stale" if seed == 98 else "cfg",
                "promotion_eligible": seed != 19,
                "promotion_note": (
                    "future-selected universe" if seed == 19 else "provenance gate passed"
                ),
                "seed_results": {seed: _result(seed, [0.01, 0.02, 0.03])},
            },
            state_path,
        )

    captured: dict[str, object] = {}

    def capture_verdict(
        _label: str,
        seed_results: dict[int, dict],
        _test_days: list,
        n_seeds: int,
        **_kwargs: object,
    ) -> None:
        captured["seeds"] = tuple(seed_results)
        captured["requested"] = n_seeds

    monkeypatch.setattr(driver, "build_windows", lambda *_args: (None, []))
    monkeypatch.setattr(driver, "split_days", lambda *_args: ([], [], []))
    monkeypatch.setattr(driver, "baseline_items", lambda *_args: [])
    monkeypatch.setattr(driver, "data_dependency_hash", lambda *_args: "data")
    monkeypatch.setattr(driver, "resolved_cfg_hash", lambda *_args: "cfg")
    monkeypatch.setattr(driver, "verdict", capture_verdict)
    args = SimpleNamespace(
        seed_base=17,
        seeds=3,
        design="tiny",
        horizon_mode="daily_raw",
        episode_len=1,
        label_horizon_days=1,
        exec_delay=1,
        n_trials=1,
        cost=0.0,
        max_windows=0,
        data_root="TOP50",
        no_news=True,
        session_seconds=60,
        block_seconds=10,
        bar_seconds=1,
    )

    assert driver.aggregate(tmp_path, args, tmp_path / "cache") == 0

    assert captured == {"seeds": (17, 18), "requested": 3}
    output = capsys.readouterr().out
    assert "ignoring checkpoint with stale code/data/config identity" in output
    assert "ignoring checkpoint with a different training-time provenance gate" in output
    assert "ignoring seed artifacts outside requested range 17..19: [99]" in output


def test_run_identity_changes_when_same_path_raw_source_changes(tmp_path: Path) -> None:
    (tmp_path / "universe.json").write_text(
        json.dumps({"actions": ["CASH", "AAA"], "cash_index": 0})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"universe_selection_date": "2019-12-31"}))
    partition = tmp_path / "partitions" / "2020-01-01_to_2020-01-03"
    partition.mkdir(parents=True)
    bars = partition / "bars.parquet"
    bars.write_bytes(b"first source version")

    first = driver.data_dependency_hash(str(tmp_path), 0, 23400, 300, 1, True)
    bars.write_bytes(b"second source version with a new size")
    second = driver.data_dependency_hash(str(tmp_path), 0, 23400, 300, 1, True)

    assert first != second
    manifest.write_text(json.dumps({"universe_selection_date": "2026-01-01"}))
    third = driver.data_dependency_hash(str(tmp_path), 0, 23400, 300, 1, True)
    assert second != third
    design = driver.DESIGNS["tiny"]
    resolved = SimpleNamespace(
        design="tiny",
        data_root=str(tmp_path),
        no_news=True,
        max_windows=0,
        normalization_days=64,
        **{field: getattr(design, field) for field in driver.DESIGN_FIELDS},
    )
    design_data = driver.data_dependency_hash(
        str(tmp_path),
        0,
        design.session_seconds,
        design.block_seconds,
        design.bar_seconds,
        True,
    )
    assert driver.design_cfg_hash("tiny", 0, str(tmp_path), True) == driver.resolved_cfg_hash(
        resolved, design_data
    )


def test_manifest_jsonl_and_final_policy_are_durable(tmp_path, monkeypatch) -> None:
    times = iter(["t0", "t1", "t2"])
    monkeypatch.setattr(driver, "_utc_now", lambda: next(times))
    manifest_path = tmp_path / "run_manifest.json"
    driver._publish_run_manifest(manifest_path, {"cfg_hash": "abc", "value": float("nan")})
    driver._publish_run_manifest(manifest_path, {"cfg_hash": "abc", "value": 2})
    manifest = json.loads(manifest_path.read_text())
    assert manifest["created_at_utc"] == "t0"
    assert manifest["updated_at_utc"] == "t1"
    assert manifest["value"] == 2
    assert not list(tmp_path.glob("*.tmp"))

    metrics = tmp_path / "metrics.jsonl"
    driver._append_jsonl(metrics, {"event": "one", "value": float("inf")})
    driver._append_jsonl(metrics, {"event": "two", "value": torch.tensor(3.0)})
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert rows == [{"event": "one", "value": None}, {"event": "two", "value": 3.0}]

    policy = torch.nn.Linear(2, 1)
    state = {"cfg": "abc"}
    checkpoint = driver._save_final_policy(
        tmp_path / "state.pt", policy, seed=9, best_val=0.25, state=state, mode="daily",
        context_hash="ctx-hash", context_artifact_id="generation-a", context_path=tmp_path / "ctx.pt",
    )
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["seed"] == 9
    assert saved["selection_source"] == "validation_mean_net_return"
    assert saved["context"] == {
        "hash": "ctx-hash",
        "artifact_id": "generation-a",
        "encoder_path": str(driver._context_generation_path(tmp_path / "ctx.pt", "generation-a")),
    }
    assert set(saved["policy_state_dict"]) == set(policy.state_dict())


def test_daily_raw_without_eligible_validation_checkpoint_never_publishes_or_tests(
    tmp_path: Path, monkeypatch
) -> None:
    class FakePolicy:
        def __init__(self, _config) -> None:
            pass

        def to(self, _device):
            return self

    def fake_build(_records, episode_len, **_kwargs):
        return [{"n_blocks": episode_len}]

    calls = {"publish": 0, "test": 0}

    def forbidden_publish(*_args, **_kwargs):
        calls["publish"] += 1
        raise AssertionError("invalid policy weights were published")

    def forbidden_test(*_args, **_kwargs):
        calls["test"] += 1
        raise AssertionError("test split was opened without a validation checkpoint")

    monkeypatch.setattr(driver, "to_daily_raw_records", lambda values: list(values))
    monkeypatch.setattr(driver, "build_daily_raw_episodes", fake_build)
    monkeypatch.setattr(driver, "DailyCrossSectionPolicy", FakePolicy)
    monkeypatch.setattr(driver, "train_daily_policy", lambda *_args, **_kwargs: (None, -1e9, None))
    monkeypatch.setattr(driver, "_begin_stage_runtime", lambda _device: 0.0)
    monkeypatch.setattr(driver, "_finish_stage_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "_save_final_policy", forbidden_publish)
    monkeypatch.setattr(driver, "evaluate_daily_detailed", forbidden_test)
    monkeypatch.setattr(driver, "_is_main", lambda: True)

    args = SimpleNamespace(
        label_horizon_days=1,
        exec_delay=1,
        episode_stride=2,
        episode_len=2,
        batch_days=1,
        policy_accum=1,
        d_model=8,
        raw_policy_dim=8,
        raw_policy_layers=1,
        raw_policy_heads=2,
        block_seconds=2,
        bar_seconds=1,
        session_seconds=4,
        policy_token_dim=8,
        policy_layers=1,
        policy_heads=2,
        daily_lookback=2,
        dropout=0.0,
        temperature=1.0,
        max_stock_weight=1.0,
        gate_init_bias=0.0,
        grad_checkpoint=False,
        raw_norm="level",
        raw_recent_days=0,
        raw_stock_chunk=0,
        pol_lr=1e-3,
        pol_weight_decay=0.0,
        policy_steps=1,
        cost=0.0,
        risk_lambda=0.0,
        entropy_coef=0.0,
        max_actions_per_day=1.0,
        budget_lambda=0.0,
        gate_entropy_coef=0.0,
        missing_label_penalty=0.0,
        bptt_window=1,
        pol_warmup_frac=0.0,
        schedule="constant",
        grad_clip=0.0,
        amp=False,
        vram_ceiling_gib=0.0,
    )
    values = [{"date": "d0"}, {"date": "d1"}, {"date": "d2"}]

    with pytest.raises(SystemExit, match=r"no validation checkpoint.*refusing to publish weights or evaluate"):
        driver._run_daily_raw(
            values,
            values,
            values,
            values,
            values,
            values,
            args,
            torch.device("cpu"),
            17,
            None,
            {"cfg": "test"},
            tmp_path / "state.pt",
        )

    assert calls == {"publish": 0, "test": 0}


def test_daily_raw_validation_checkpoint_abort_is_broadcast_to_every_rank(monkeypatch) -> None:
    calls: list[tuple[bool, str, int]] = []

    monkeypatch.setitem(driver._DIST, "is_dist", True)
    monkeypatch.setattr(driver, "_is_main", lambda: False)

    def fake_broadcast(value: bool, device: torch.device, src: int = 0) -> bool:
        calls.append((value, device.type, src))
        return False

    monkeypatch.setattr(driver, "broadcast_bool", fake_broadcast)

    with pytest.raises(SystemExit, match=r"no validation checkpoint.*refusing to publish weights or evaluate"):
        driver._require_validation_checkpoint(None, torch.device("cpu"), seed=17, mode="daily_raw")

    # A non-main rank contributes a placeholder; rank 0's broadcast verdict
    # controls whether all ranks continue or abort together.
    assert calls == [(False, "cpu", 0)]


def test_vram_ceiling_is_validated_and_applied_as_an_allocator_fraction(monkeypatch) -> None:
    calls: list[tuple[float, torch.device]] = []
    monkeypatch.setattr(
        driver.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=80 * 1024**3),
    )
    monkeypatch.setattr(
        driver.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: calls.append((fraction, device)),
    )
    device = torch.device("cuda:2")

    control = driver._configure_vram_ceiling(device, 75.0)

    assert control == {
        "enabled": True,
        "ceiling_gib": 75.0,
        "device_total_gib": 80.0,
        "allocator_fraction": 0.9375,
    }
    assert calls == [(0.9375, device)]
    with pytest.raises(ValueError, match="exceeds"):
        driver._configure_vram_ceiling(device, 81.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        driver._configure_vram_ceiling(device, float("nan"))
    with pytest.raises(ValueError, match="requires a CUDA"):
        driver._configure_vram_ceiling(torch.device("cpu"), 1.0)


def test_runtime_stage_payload_reports_segment_step_time_and_measured_peak() -> None:
    payload = driver._runtime_stage_payload(
        "stage1_ssl",
        elapsed_seconds=25.0,
        start_step=10,
        end_step=20,
        memory={
            "allocated_gb": 12.0,
            "reserved_gb": 15.0,
            "peak_allocated_gb": 68.0,
            "peak_reserved_gb": 70.0,
            "free_gb": 5.0,
            "total_gb": 80.0,
        },
        vram_ceiling_gib=75.0,
    )

    assert payload["optimizer_steps_completed"] == 10
    assert payload["seconds_per_optimizer_step"] == 2.5
    assert payload["cuda_peak_allocated_gib"] == 68.0
    assert payload["cuda_peak_reserved_gib"] == 70.0
    assert payload["peak_reserved_fraction_of_ceiling"] == pytest.approx(70 / 75)


def test_driver_restores_rank_specific_rng_and_supports_legacy_checkpoint(monkeypatch) -> None:
    original_dist = dict(driver._DIST)
    try:
        rank_states = []
        for rank_seed in (101, 202):
            torch.manual_seed(rank_seed)
            rank_states.append({"cpu": torch.get_rng_state().clone()})
        driver._DIST.update(rank=1, world=2, local_rank=1, is_dist=True, grad_reduce=None)
        torch.manual_seed(999)
        driver._restore_rng(None, 17, 25, 7919, torch.device("cpu"), rank_states=rank_states)
        assert torch.equal(torch.get_rng_state(), rank_states[1]["cpu"])

        # Old checkpoints have only rank 0's CPU state. They remain readable;
        # the absent CUDA stream is reconstructed deterministically.
        driver._DIST.update(rank=0, world=1, local_rank=0, is_dist=False, grad_reduce=None)
        torch.manual_seed(303)
        legacy_cpu = torch.get_rng_state().clone()
        cuda_seeds: list[int] = []
        monkeypatch.setattr(driver.torch.cuda, "manual_seed", cuda_seeds.append)
        torch.rand(9)
        driver._restore_rng(legacy_cpu, 17, 25, 7919, torch.device("cuda:0"))
        assert torch.equal(torch.get_rng_state(), legacy_cpu)
        assert cuda_seeds == [17 * 7919 + 26 * 104729]
    finally:
        driver._DIST.clear()
        driver._DIST.update(original_dist)


def test_named_top50_sweep_aliases_are_one_seed_one_gpu_protocols() -> None:
    assert sweep.resolve_designs("top50-core") == list(sweep.TOP50_H100_CORE_SWEEP)
    assert sweep.resolve_designs("top50-wide") == list(sweep.TOP50_H100_WIDE_SWEEP)
    assert all(sweep.DESIGNS[name].min_gpus == 1 for name in sweep.resolve_designs("top50-wide"))
    sweep.validate_screening_request("top50-wide", seeds=1, gpus_per_job=0)
    sweep.validate_screening_request("top50-core", seeds=1, gpus_per_job=1)
    with pytest.raises(ValueError, match="one-seed screening protocol"):
        sweep.validate_screening_request("top50-wide", seeds=2, gpus_per_job=1)
    with pytest.raises(ValueError, match="one GPU per setting"):
        sweep.validate_screening_request("top50-core", seeds=1, gpus_per_job=2)


def test_named_top2000_study_is_one_seed_per_independent_gpu_setting() -> None:
    expected = list(sweep.TOP2000_H100_WIDE_SWEEP)
    core = list(sweep.TOP2000_H100_CORE_SWEEP)
    assert expected
    assert expected[:len(core)] == core
    assert len(expected) == len(set(expected))
    assert sweep.resolve_designs("top2000-wide") == expected
    assert sweep.resolve_designs("top2000") == expected
    assert sweep.resolve_designs("sweep") == expected
    assert all(sweep.DESIGNS[name].min_gpus == 1 for name in expected)
    sweep.validate_screening_request("top2000-wide", seeds=1, gpus_per_job=0)
    sweep.validate_screening_request("sweep", seeds=1, gpus_per_job=1)
    sweep.validate_screening_request("top2000-wide", seeds=1, gpus_per_job=1)
    with pytest.raises(ValueError, match="one-seed-per-setting"):
        sweep.validate_screening_request("top2000-wide", seeds=2, gpus_per_job=1)
    with pytest.raises(ValueError, match="one GPU per independent setting"):
        sweep.validate_screening_request("top2000-wide", seeds=1, gpus_per_job=2)


@pytest.mark.parametrize(
    ("trailing_override", "message"),
    [
        (["--designs", "daily_raw_top2000"], "requires --designs"),
        (["--gpus-per-job", "4"], "one GPU per independent setting"),
        (["--seeds", "2"], "one-seed-per-setting"),
    ],
)
def test_sticky_top2000_protocol_rejects_trailing_launcher_overrides(
    trailing_override: list[str], message: str
) -> None:
    argv = [
        "--independent-top2000-protocol",
        "--designs", "top2000-wide",
        "--gpus-per-job", "1",
        "--seeds", "1",
        *trailing_override,
    ]

    with pytest.raises(SystemExit, match=message):
        sweep.main(argv)


def test_explicit_sweep_rejects_duplicate_design_settings() -> None:
    with pytest.raises(ValueError, match="duplicate settings"):
        sweep.resolve_designs("daily_raw_top2000,daily_raw_top2000")


def test_top2000_wide_has_one_shared_and_four_distinct_context_groups() -> None:
    groups: dict[tuple[object, ...], list[str]] = {}
    for name in sweep.TOP2000_H100_WIDE_SWEEP:
        design = sweep.DESIGNS[name]
        key = tuple(getattr(design, field) for field in driver.CONTEXT_FIELDS)
        groups.setdefault(key, []).append(name)

    sizes = sorted(len(names) for names in groups.values())
    assert sizes == [1, 1, 1, 1, 23]
    shared = next(names for names in groups.values() if len(names) == 23)
    assert list(sweep.TOP2000_H100_CORE_SWEEP) == shared[:10]


def test_unpublished_context_blocks_dependants_only_for_current_invocation(tmp_path) -> None:
    context_path = tmp_path / "contexts" / "shared.pt"
    groups = {context_path: [0, 1, 2, 3]}
    launched = {0, 3}

    blocked = sweep.block_unpublished_context_dependants(
        context_path, 0, groups, launched, artifacts_ready=False
    )

    assert blocked == [1, 2]
    assert launched == {0, 1, 2, 3}

    # Validated publication before producer exit leaves all policy dependants
    # runnable. A new process invocation also reconstructs its own launched set,
    # so nothing durable marks dependants skipped across resumable runs.
    relaunched = {0}
    assert sweep.block_unpublished_context_dependants(
        context_path, 0, groups, relaunched, artifacts_ready=True
    ) == []
    assert relaunched == {0}


def test_context_artifact_readiness_validates_identity_and_rechecks_atomic_replacement(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "ctx-expected.pt"
    wrong_state = {"weight": torch.ones(1)}
    torch.save({
        "schema_version": 1,
        "ctx": "wrong",
        "artifact_id": "generation-wrong",
        "state_schema_sha256": driver._context_state_schema_sha256(wrong_state),
        "enc": wrong_state,
    }, path)
    cache: sweep.ArtifactReadinessCache = {}
    real_load = sweep.torch.load
    loads: list[Path] = []

    def tracked_load(candidate, *args, **kwargs):
        loads.append(Path(candidate))
        return real_load(candidate, *args, **kwargs)

    monkeypatch.setattr(sweep.torch, "load", tracked_load)
    assert not sweep.context_encoder_artifact_ready(path, cache)
    assert not sweep.context_encoder_artifact_ready(path, cache)
    assert loads == [path]

    replacement = tmp_path / "replacement.pt"
    replacement_state = {"weight": torch.ones(1)}
    torch.save({
        "schema_version": 1,
        "ctx": path.stem,
        "artifact_id": "generation-correct",
        "state_schema_sha256": driver._context_state_schema_sha256(replacement_state),
        "enc": replacement_state,
    }, replacement)
    os.replace(replacement, path)
    assert sweep.context_encoder_artifact_ready(path, cache)
    assert loads == [path, path]


def test_daily_eod_artifact_readiness_validates_payload_and_hash(tmp_path) -> None:
    context_hash = "ctx-eod"
    storage_dtype = "bfloat16"
    path = tmp_path / f"{context_hash}.daily_eod.{storage_dtype}.pt"
    fields = {
        "market": torch.zeros(1, 2, dtype=torch.bfloat16),
        "per_stock": torch.zeros(1, 3, 2, dtype=torch.bfloat16),
        "avail": torch.ones(1, 3, dtype=torch.bool),
        "news_raw": torch.zeros(1, 3, driver.MAX_NEWS, driver.NEWS_RAW_DIM),
        "news_mask": torch.zeros(1, 3, driver.MAX_NEWS, dtype=torch.bool),
        "day_close": torch.ones(1, 3),
    }
    torch.save({
        "schema_version": 1,
        "context_hash": context_hash,
        "context_artifact_id": "encoder-generation",
        "storage_dtype": storage_dtype,
        "dates": ["2024-01-02"],
        "fields": fields,
    }, path)
    cache: sweep.ArtifactReadinessCache = {}

    assert sweep.daily_eod_artifact_ready(
        path,
        context_hash=context_hash,
        context_artifact_id="encoder-generation",
        context_dim=2,
        storage_dtype=storage_dtype,
        cache=cache,
    )
    # The file itself is unchanged, but a new encoder generation must force
    # revalidation instead of reusing the memoized True verdict.
    assert not sweep.daily_eod_artifact_ready(
        path,
        context_hash=context_hash,
        context_artifact_id="new-encoder-generation",
        context_dim=2,
        storage_dtype=storage_dtype,
        cache=cache,
    )
    assert not sweep.daily_eod_artifact_ready(
        path,
        context_hash="other",
        context_artifact_id="encoder-generation",
        context_dim=2,
        storage_dtype=storage_dtype,
        cache={},
    )


def test_sweep_device_parser_canonicalizes_and_rejects_ambiguous_pools() -> None:
    assert sweep.parse_devices("0,cuda:1,02,3") == ["0", "1", "2", "3"]
    assert sweep.parse_devices("cpu") == ["cpu"]
    for invalid in ("", "0,00", "0,cpu", "-1", "gpu:0", "GPU-deadbeef"):
        with pytest.raises(ValueError):
            sweep.parse_devices(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cache_windows", 0, "cache-windows"),
        ("cpu_workers_per_gpu", 0, "cpu-workers-per-gpu"),
        ("build_workers", -1, "build-workers"),
        ("max_windows", -1, "max-windows"),
        ("poll_sec", float("nan"), "poll-sec"),
        ("poll_sec", 0.0, "poll-sec"),
        ("distributed_timeout_minutes", float("nan"), "distributed-timeout-minutes"),
        ("distributed_timeout_minutes", 0.0, "distributed-timeout-minutes"),
        ("runs_root", " ", "runs-root"),
        ("master_port_base", 65_530, "TCP ports"),
    ],
)
def test_sweep_runtime_controls_fail_before_launch(field: str, value: object, message: str) -> None:
    controls = {
        "cache_windows": 4,
        "cpu_workers_per_gpu": 8,
        "build_workers": 28,
        "max_windows": 0,
        "poll_sec": 2.0,
        "distributed_timeout_minutes": 120.0,
        "master_port_base": None,
        "max_launches": 26,
        "runs_root": "runs/top2000",
    }
    controls[field] = value
    with pytest.raises(ValueError, match=message):
        sweep.validate_runtime_controls(**controls)


def test_torchrun_defaults_to_a_collision_free_standalone_rendezvous() -> None:
    automatic = sweep.torchrun_prefix("python", 4, None)
    assert automatic == [
        "python", "-m", "torch.distributed.run", "--nproc_per_node", "4", "--standalone",
    ]
    fixed = sweep.torchrun_prefix("python", 4, 29_500)
    assert fixed[-4:] == ["--master_addr", "127.0.0.1", "--master_port", "29500"]
    with pytest.raises(ValueError, match="at least two"):
        sweep.torchrun_prefix("python", 1, None)


def test_sweep_lock_prevents_duplicate_checkpoint_writers(tmp_path) -> None:
    first = sweep.acquire_sweep_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another sweep already owns"):
            sweep.acquire_sweep_lock(tmp_path)
    finally:
        first.close()

    resumed = sweep.acquire_sweep_lock(tmp_path)
    resumed.close()


def test_sweep_resume_skips_only_matching_completed_seed(tmp_path) -> None:
    state_path = tmp_path / "state.pt"
    torch.save({"cfg": "expected", "seed_results": {"17": {"returns": []}}}, state_path)
    assert sweep.seed_done(state_path, 17, "expected")
    assert not sweep.seed_done(state_path, 18, "expected")
    assert not sweep.seed_done(state_path, 17, "stale")

    # A durable in-progress checkpoint must resume, not be mistaken for a
    # completed seed and skipped by the scheduler.
    torch.save({"cfg": "expected", "seed_results": {}, "cur": {"seed": 17, "step": 500}}, state_path)
    assert not sweep.seed_done(state_path, 17, "expected")
    state_path.write_bytes(b"truncated")
    assert not sweep.seed_done(state_path, 17, "expected")


def test_guarded_gpu_preflight_is_strict_and_requires_valid_nvidia_smi(monkeypatch) -> None:
    def healthy(*_args, **kwargs):
        assert kwargs["check"] is True
        rows = [f"{index}, 80000, 81920" for index in range(4)]
        return SimpleNamespace(stdout="\n".join(rows))

    monkeypatch.setattr(sweep.subprocess, "run", healthy)
    sweep.gpu_preflight(["0", "1", "2", "3"], vram_ceiling_gib=75.0, strict_free=True)

    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, 70000, 81920"),
    )
    with pytest.raises(SystemExit, match="Refusing the guarded TOP2000 launch"):
        sweep.gpu_preflight(["0"], vram_ceiling_gib=75.0, strict_free=True)

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(sweep.subprocess, "run", unavailable)
    with pytest.raises(SystemExit, match="refusing to launch"):
        sweep.gpu_preflight(["0"], vram_ceiling_gib=75.0, strict_free=True)


def test_guarded_top2000_rechecks_claim_immediately_before_launch(tmp_path, monkeypatch) -> None:
    preflight_calls: list[tuple[tuple[str, ...], float, bool]] = []
    monkeypatch.setattr(
        sweep,
        "gpu_preflight",
        lambda devices, need_gib=40.0, vram_ceiling_gib=0.0, *, strict_free=False: preflight_calls.append(
            (tuple(devices), vram_ceiling_gib, strict_free)
        ),
    )
    monkeypatch.setattr(sweep, "resolve_designs", lambda _spec: ["daily_raw_top2000"])
    monkeypatch.setattr(sweep, "seed_done", lambda *_args, **_kwargs: False)
    context_path = tmp_path / "cache" / "contexts" / "shared.pt"
    context_path.parent.mkdir(parents=True)
    shared_state = {"weight": torch.ones(1)}
    torch.save({
        "schema_version": 1,
        "ctx": context_path.stem,
        "artifact_id": "shared-generation",
        "state_schema_sha256": driver._context_state_schema_sha256(shared_state),
        "enc": shared_state,
    }, context_path)
    monkeypatch.setattr(sweep, "_context_cache_path", lambda *_args, **_kwargs: context_path)
    monkeypatch.setattr(sweep, "design_cfg_hash", lambda *_args, **_kwargs: "cfg")
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    class FinishedProcess:
        def poll(self) -> int:
            return 0

    launched_commands: list[list[str]] = []

    def finished_popen(command, *_args, **_kwargs):
        launched_commands.append(command)
        return FinishedProcess()

    monkeypatch.setattr(sweep.subprocess, "Popen", finished_popen)
    monkeypatch.setattr(sweep.time, "sleep", lambda _seconds: None)

    assert sweep.main([
        "--designs", "top2000-core",
        "--data-root", "TOP2000",
        "--devices", "0,1,2,3",
        "--runs-root", str(tmp_path),
        "--distributed-timeout-minutes", "37.5",
        "--python", "python",
    ]) == 0

    # Once before cache construction and once again after the scheduler claims
    # the cards, directly adjacent to Popen.
    assert preflight_calls == [
        (("0", "1", "2", "3"), 75.0, True),
        (("0",), 75.0, True),
    ]
    assert len(launched_commands) == 1
    command = launched_commands[0]
    assert command[command.index("--distributed-timeout-minutes") + 1] == "37.5"
    assert "torch.distributed.run" not in command
    assert command[command.index("--device") + 1] == "cuda:0"


def test_named_top2000_scheduler_replenishes_four_direct_gpus_across_all_settings(
    tmp_path, monkeypatch
) -> None:
    designs = list(sweep.TOP2000_H100_WIDE_SWEEP)
    assert len(designs) == 27
    monkeypatch.setattr(sweep, "seed_done", lambda *_args, **_kwargs: False)

    context_paths_by_key: dict[tuple[object, ...], Path] = {}
    context_paths_by_design: dict[str, Path] = {}
    for name in designs:
        design = sweep.DESIGNS[name]
        key = tuple(getattr(design, field) for field in driver.CONTEXT_FIELDS)
        if key not in context_paths_by_key:
            context_paths_by_key[key] = tmp_path / "cache" / "contexts" / f"group-{len(context_paths_by_key)}.pt"
        context_paths_by_design[name] = context_paths_by_key[key]

    monkeypatch.setattr(
        sweep,
        "_context_cache_path",
        lambda design_name, _seed, **_kwargs: context_paths_by_design[design_name],
    )
    ready_contexts: set[Path] = set()

    def ready_context(path, _cache, artifact_ids):
        if path not in ready_contexts:
            artifact_ids.pop(path, None)
            return False
        artifact_ids[path] = f"generation-{path.stem}"
        return True

    monkeypatch.setattr(sweep, "context_encoder_artifact_ready", ready_context)
    monkeypatch.setattr(sweep, "daily_eod_artifact_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sweep, "design_cfg_hash", lambda *_args, **_kwargs: "cfg")
    for key in sweep.TORCH_DISTRIBUTED_ENV_VARS:
        monkeypatch.setenv(key, "hostile-parent-value")

    preflight_claims: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        sweep,
        "gpu_preflight",
        lambda devices, *_args, **_kwargs: preflight_claims.append(tuple(devices)),
    )
    direct_run_envs: list[dict[str, str]] = []

    def finished_run(*_args, **kwargs):
        if kwargs.get("env") is not None:
            direct_run_envs.append(dict(kwargs["env"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", finished_run)
    monkeypatch.setattr(sweep.time, "sleep", lambda _seconds: None)

    counters = {"active": 0, "peak": 0}
    launches: list[tuple[list[str], str, dict[str, str]]] = []

    class FinishedProcess:
        def __init__(self, context_path: Path) -> None:
            self.finished = False
            self.context_path = context_path

        def poll(self) -> int:
            if not self.finished:
                self.finished = True
                counters["active"] -= 1
                ready_contexts.add(self.context_path)
            return 0

    def finished_popen(command, *_args, **kwargs):
        design_name = command[command.index("--design") + 1]
        counters["active"] += 1
        counters["peak"] = max(counters["peak"], counters["active"])
        launches.append((list(command), kwargs["env"]["CUDA_VISIBLE_DEVICES"], dict(kwargs["env"])))
        return FinishedProcess(context_paths_by_design[design_name])

    monkeypatch.setattr(sweep.subprocess, "Popen", finished_popen)

    assert sweep.main([
        "--independent-top2000-protocol",
        "--designs", "top2000-wide",
        "--data-root", "TOP2000",
        "--devices", "0,1,2,3",
        "--runs-root", str(tmp_path),
        "--python", "python",
    ]) == 0

    assert counters["peak"] == 4
    assert len(launches) == 27
    launched_designs = [command[command.index("--design") + 1] for command, _visible, _env in launches]
    assert len(set(launched_designs)) == 27
    assert set(launched_designs) == set(designs)
    assert len({context_paths_by_design[name] for name in launched_designs[:4]}) == 4
    visible_devices = [visible for _command, visible, _env in launches]
    assert visible_devices[:4] == ["0", "1", "2", "3"]
    assert sorted(visible_devices.count(device) for device in ("0", "1", "2", "3")) == [6, 7, 7, 7]
    assert preflight_claims[0] == ("0", "1", "2", "3")
    assert len(preflight_claims) == 28
    assert all(len(claim) == 1 for claim in preflight_claims[1:])
    for command, visible, env in launches:
        assert command[:2] == ["python", str(sweep.DRIVER)]
        assert "torch.distributed.run" not in command
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--seeds") + 1] == "1"
        assert command[command.index("--seed-base") + 1] == "17"
        assert env["CUDA_VISIBLE_DEVICES"] == visible
        assert all(key not in env for key in sweep.TORCH_DISTRIBUTED_ENV_VARS)
    assert direct_run_envs
    assert all(
        key not in env
        for env in direct_run_envs
        for key in sweep.TORCH_DISTRIBUTED_ENV_VARS
    )


@pytest.mark.parametrize(("build_only", "stream"), [(True, True), (False, True), (False, False)])
def test_window_builder_refuses_partial_requested_chronology(
    tmp_path, monkeypatch, build_only: bool, stream: bool
) -> None:
    monkeypatch.setattr(driver, "_resolve_root", lambda _root: tmp_path)
    monkeypatch.setattr(driver, "load_universe", lambda _root: (["CASH", "AAA"], 0))
    monkeypatch.setattr(driver, "source_symbol_to_action_index", lambda _root: {"AAA": 1})
    monkeypatch.setattr(driver, "list_windows", lambda _root: ["w0", "w1"])
    monkeypatch.setattr(driver, "_window_key", lambda _root, window, _cfg, _sig: f"{window}.pt")
    monkeypatch.setattr(driver, "load_or_build_window", lambda *_args, **_kwargs: None)
    args = SimpleNamespace(
        data_root="TOP2000",
        session_seconds=300,
        block_seconds=300,
        bar_seconds=60,
        no_news=True,
        max_windows=0,
        build_workers=1,
        build_only=build_only,
        stream=stream,
    )

    with pytest.raises(SystemExit, match="Refusing to train on a partial date universe"):
        driver.build_windows(args, tmp_path / "cache")


def _fake_python(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_top2000_launcher_passes_guarded_four_h100_protocol_without_implicit_ack(tmp_path) -> None:
    capture = tmp_path / "args.txt"
    fake_python = _fake_python(tmp_path / "python", 'printf "%s\\n" "$@" > "$CAPTURE_PATH"')
    env = {**os.environ, "PYTHON": str(fake_python), "CAPTURE_PATH": str(capture)}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("QUANTTRADE_DEVICES", None)

    completed = subprocess.run(["bash", str(TOP2000_LAUNCHER)], env=env, capture_output=True, text=True, timeout=5)

    assert completed.returncode == 0, completed.stderr
    args = capture.read_text(encoding="utf-8").splitlines()
    assert args[0].endswith("/training/sweep_phase1.py")
    expected_values = {
        "--designs": "top2000-wide",
        "--data-root": "TOP2000",
        "--devices": "0,1,2,3",
        "--gpus-per-job": "1",
        "--vram-ceiling-gib": "75",
        "--cache-windows": "4",
        "--cpu-workers-per-gpu": "8",
        "--build-workers": "28",
        "--distributed-timeout-minutes": "120",
        "--seeds": "1",
        "--seed-base": "17",
        "--runs-root": "runs/top2000",
    }
    for flag, expected in expected_values.items():
        assert args[args.index(flag) + 1] == expected
    assert args.count("--independent-top2000-protocol") == 1
    assert args.count("--stream") == 1
    assert "--allow-unreportable" not in args

    env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    subprocess.run(["bash", str(TOP2000_LAUNCHER)], env=env, check=True, timeout=5)
    inherited_args = capture.read_text(encoding="utf-8").splitlines()
    assert inherited_args[inherited_args.index("--devices") + 1] == "4,5,6,7"

    subprocess.run(
        [
            "bash", str(TOP2000_LAUNCHER),
            "--designs", "daily_raw_top2000",
            "--gpus-per-job", "4",
            "--seeds", "2",
        ],
        env=env,
        check=True,
        timeout=5,
    )
    hostile_args = capture.read_text(encoding="utf-8").splitlines()
    assert hostile_args.count("--independent-top2000-protocol") == 1
    with pytest.raises(SystemExit, match="requires --designs"):
        sweep.main(hostile_args[1:])

    assert "--independent-top2000-protocol" in ROOT_LAUNCHER.read_text(encoding="utf-8")


def test_cache_build_defaults_use_28_single_threaded_workers(monkeypatch) -> None:
    assert driver.DEFAULT_CACHE_BUILD_WORKERS == 28
    assert sweep.DEFAULT_CACHE_BUILD_WORKERS == 28
    assert "--build-workers 28" in ROOT_LAUNCHER.read_text(encoding="utf-8")

    thread_calls: list[tuple[str, int]] = []
    fake_arrow = SimpleNamespace(
        set_cpu_count=lambda count: thread_calls.append(("arrow-cpu", count)),
        set_io_thread_count=lambda count: thread_calls.append(("arrow-io", count)),
    )
    monkeypatch.setitem(__import__("sys").modules, "pyarrow", fake_arrow)
    monkeypatch.setattr(driver.os, "environ", {})
    monkeypatch.setattr(driver.torch, "set_num_threads", lambda count: thread_calls.append(("torch", count)))
    monkeypatch.setattr(
        driver.torch,
        "set_num_interop_threads",
        lambda count: thread_calls.append(("torch-interop", count)),
    )

    driver._configure_cache_worker()

    assert set(driver.os.environ.values()) == {"1"}
    assert thread_calls == [("torch", 1), ("torch-interop", 1), ("arrow-cpu", 1), ("arrow-io", 1)]


def test_cache_build_pool_caps_28_worker_default_to_uncached_work(monkeypatch, tmp_path) -> None:
    pool_config: dict[str, object] = {}
    inherited_thread_env: dict[str, str | None] = {}

    class FakePool:
        def __init__(self, **kwargs) -> None:
            pool_config.update(kwargs)
            inherited_thread_env.update(
                {name: driver.os.environ.get(name) for name in driver.CACHE_WORKER_THREAD_ENV_VARS}
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        @staticmethod
        def map(_fn, args):
            return [item[1] for item in args]

    monkeypatch.setattr(driver, "_resolve_root", lambda _root: tmp_path)
    monkeypatch.setattr(driver, "load_universe", lambda _root: (["CASH", "AAA"], 0))
    monkeypatch.setattr(driver, "source_symbol_to_action_index", lambda _root: {"AAA": 1})
    monkeypatch.setattr(driver, "list_windows", lambda _root: ["w0", "w1"])
    monkeypatch.setattr(driver, "_window_key", lambda _root, window, _cfg, _sig: f"{window}.pt")
    monkeypatch.setattr(driver, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(driver, "_ensure_complete_window_cache", lambda *args: args[1])
    original_thread_env = {
        name: f"original-{index}" for index, name in enumerate(driver.CACHE_WORKER_THREAD_ENV_VARS)
    }
    for name, value in original_thread_env.items():
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(
        data_root="TOP2000",
        session_seconds=300,
        block_seconds=300,
        bar_seconds=60,
        no_news=True,
        max_windows=0,
        build_workers=driver.DEFAULT_CACHE_BUILD_WORKERS,
        build_only=True,
        stream=True,
    )

    _, windows = driver.build_windows(args, tmp_path / "cache")

    assert windows == ["w0", "w1"]
    assert pool_config["max_workers"] == 2
    assert pool_config["initializer"] is driver._configure_cache_worker
    assert set(inherited_thread_env.values()) == {"1"}
    assert {name: driver.os.environ.get(name) for name in driver.CACHE_WORKER_THREAD_ENV_VARS} == original_thread_env


def test_root_launcher_stops_tailing_and_propagates_sweep_failure(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    launcher = project / "run.sh"
    launcher.write_text(ROOT_LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")
    fake_python = _fake_python(tmp_path / "python", "exit 7")

    completed = subprocess.run(
        ["bash", str(launcher)],
        env={**os.environ, "PYTHON": str(fake_python)},
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 7
    assert "sweep started" in completed.stdout
