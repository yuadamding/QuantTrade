"""Auditability regressions for the external Phase-1 training driver."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


DRIVER_PATH = Path(__file__).resolve().parents[2] / "training" / "train_phase1.py"
SPEC = importlib.util.spec_from_file_location("phase1_reporting_driver", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


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
    assert note == "complete aligned coverage"

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
    assert note == "complete aligned coverage"

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
    assert "reportable" in note and "legacy" in note


def test_buy_and_hold_baseline_pays_each_round_trip() -> None:
    item = {
        "ret": torch.tensor([[0.0, 0.10], [0.0, 0.10], [0.0, 0.10]]),
        "ret_valid": torch.tensor([[True, True], [True, False], [True, True]]),
    }

    cash, buy_hold = driver.round_trip_cost_paid_baselines([item], cost=0.01)

    # Two one-row valid holding intervals: gross .20 - two entries/exits * .02, divided by two marks.
    assert cash == 0.0
    assert abs(buy_hold - 0.08) < 1e-7


def test_verdict_uses_paired_series_and_never_selects_test_seed(monkeypatch, capsys) -> None:
    results = {
        1: _result(1, [0.010, -0.005, 0.020, 0.001]),
        2: _result(2, [0.020, -0.001, 0.010, 0.005]),
    }
    baseline = [{"ret": torch.zeros(4, 2), "ret_valid": torch.ones(4, 2, dtype=torch.bool)}]
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
    baseline = [{"ret": torch.zeros(4, 2), "ret_valid": torch.ones(4, 2, dtype=torch.bool)}]
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
    baseline = [{"ret": torch.zeros(4, 2), "ret_valid": torch.ones(4, 2, dtype=torch.bool)}]
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
        tmp_path / "state.pt", policy, seed=9, best_val=0.25, state=state, mode="daily"
    )
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["seed"] == 9
    assert saved["selection_source"] == "validation_mean_net_return"
    assert set(saved["policy_state_dict"]) == set(policy.state_dict())
