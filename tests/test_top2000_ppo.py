from __future__ import annotations

import hashlib
import json
from pathlib import Path
import datetime as dt

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import rl_quant.workflows.top2000_ppo as workflow
from rl_quant.envs import HistoricalMarketData


def _write_dataset_layout(root: Path) -> None:
    (root / "manifest.json").write_text(
        json.dumps({"universe_selection_date": "2026-06-12"})
    )
    (root / "universe.json").write_text(
        json.dumps({"actions": ["CASH", "AAA"], "cash_index": 0})
    )
    for name in (
        "2024-12-27_to_2024-12-31",
        "2025-01-02_to_2025-01-06",
        "2026-01-02_to_2026-01-06",
    ):
        partition = root / "partitions" / name
        partition.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "symbol": ["AAA"],
                "timestamp_ms": [1],
                "date_exchange": [name[:10]],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }),
            partition / "bars.parquet",
        )


def test_search_plan_physically_excludes_test_bars_and_identity(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_layout(tmp_path)
    touched: list[str] = []

    def signature(_root, name, _cfg, **_kwargs):
        touched.append(name)
        return hashlib.sha256(name.encode()).hexdigest()[:16]

    monkeypatch.setattr(workflow, "_portable_bars_signature", signature)
    plan = workflow.build_search_plan(tmp_path)
    serialized = json.dumps(plan.public_dict(), sort_keys=True)

    assert touched == [
        "2024-12-27_to_2024-12-31",
        "2025-01-02_to_2025-01-06",
    ]
    assert "2026-01-02_to_2026-01-06" not in serialized
    assert plan.development_only
    assert any("after sample start" in reason for reason in plan.development_reasons)

    # Changing lockbox contents cannot change search identity: search never
    # stats/hashes those bars.  It is bound later by the evaluation identity.
    search_identity = plan.search_identity
    test_file = tmp_path / "partitions/2026-01-02_to_2026-01-06/bars.parquet"
    test_file.write_bytes(b"changed test bars")
    assert workflow.build_search_plan(tmp_path).search_identity == search_identity


def test_evaluation_plan_is_the_only_plan_that_resolves_test_sources(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_layout(tmp_path)
    touched: list[str] = []

    def signature(_root, name, _cfg, **_kwargs):
        touched.append(name)
        return hashlib.sha256(name.encode()).hexdigest()[:16]

    monkeypatch.setattr(workflow, "_portable_bars_signature", signature)
    plan = workflow.build_evaluation_plan(tmp_path)

    assert touched[-1] == "2026-01-02_to_2026-01-06"
    assert [value.name for value in plan.test] == ["2026-01-02_to_2026-01-06"]
    assert plan.development_only


def test_2026_crossing_partition_is_reserved_wholly_for_lockbox(tmp_path: Path, monkeypatch) -> None:
    _write_dataset_layout(tmp_path)
    crossing = tmp_path / "partitions/2025-12-30_to_2026-01-02"
    crossing.mkdir()
    lockbox_bars = crossing / "bars.parquet"
    lockbox_bars.write_bytes(b"must-not-be-opened")
    original_stat = Path.stat
    original_open = Path.open

    def guarded_stat(self, *args, **kwargs):
        if self == lockbox_bars:
            raise AssertionError("search stat'ed lockbox bars")
        return original_stat(self, *args, **kwargs)

    def guarded_open(self, *args, **kwargs):
        if self == lockbox_bars:
            raise AssertionError("search opened lockbox bars")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "open", guarded_open)
    plan = workflow.build_search_plan(tmp_path)

    assert "2025-12-30_to_2026-01-02" not in json.dumps(plan.public_dict(), sort_keys=True)


def _market(actions: int, dates: int = 6) -> HistoricalMarketData:
    bars = torch.zeros((dates, actions, 5))
    close = torch.arange(dates, dtype=torch.float32).view(-1, 1) + 100.0
    bars[:, 1:, 0] = close
    bars[:, 1:, 1] = close + 1.0
    bars[:, 1:, 2] = close - 1.0
    bars[:, 1:, 3] = close
    bars[:, 1:, 4] = 1_000_000.0
    availability = torch.ones((dates, actions), dtype=torch.bool)
    exchange_dates = [f"2024-01-{day:02d}" for day in range(2, 2 + dates)]
    return workflow.market_data_from_daily_ohlcv(bars, availability, exchange_dates)


def test_shared_asset_actor_parameter_count_does_not_grow_with_action_count() -> None:
    small = workflow.SharedAssetRecurrentActorCritic(
        observation_key="asset_features", asset_feature_dim=8, hidden_dim=16, action_dim=5
    )
    top2000 = workflow.SharedAssetRecurrentActorCritic(
        observation_key="asset_features", asset_feature_dim=8, hidden_dim=16, action_dim=1999
    )
    assert sum(value.numel() for value in small.parameters()) == sum(
        value.numel() for value in top2000.parameters()
    )


def test_1999_action_stack_collects_and_updates_with_existing_ppo_components() -> None:
    data = _market(1999)
    trial = workflow.TrialConfig(
        hidden_dim=8,
        ppo_epochs=1,
        rollout_horizon=2,
        sequence_length=2,
        burn_in=0,
        updates=1,
        max_asset_weight=0.01,
    )
    stack = workflow.build_ppo_stack(data, trial)
    result = stack.coordinator.collect()
    recurrent = result.buffer.recurrent_sequences(sequence_length=2)
    metrics = stack.algorithm.update(recurrent)

    assert result.buffer.as_batch().actions.shape == (2, 1, 1999)
    assert torch.isfinite(torch.tensor(float(metrics["loss"])))


def test_bars_only_adapter_rejects_extra_feature_channels() -> None:
    data = _market(3)
    contaminated = HistoricalMarketData(
        features={**data.features, "news": torch.zeros((1, data.horizon + 1, 3, 1))},
        asset_returns=data.asset_returns,
        availability=data.availability,
        decision_ids=data.decision_ids,
    )
    adapter = workflow.BarsOnlyObservationAdapter()

    with pytest.raises(ValueError, match="news/covariate fields are forbidden"):
        adapter.build(
            contaminated,
            time_index=0,
            weights=torch.tensor([[1.0, 0.0, 0.0]]),
            equity=torch.ones(1),
            episode_start=torch.ones(1, dtype=torch.bool),
        )


def test_training_requires_exact_development_ack_before_loading_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Training requires"):
        workflow.run_trial(
            tmp_path,
            tmp_path / "trial",
            workflow.TrialConfig(updates=1),
            bar_seconds=300,
            device="cpu",
            acknowledgement="yes",
        )


def test_indexed_worker_rejects_plan_digest_before_dataset_access(tmp_path: Path) -> None:
    frozen = tmp_path / "plan.json"
    frozen.write_text(json.dumps({"schema_version": 1, "trials": []}))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        workflow.run_indexed_worker(
            tmp_path / "cache.pt",
            "1" * 64,
            frozen,
            "0" * 64,
            tmp_path / "out",
            index=0,
            device="cpu",
            acknowledgement=workflow.DEVELOPMENT_ACK,
        )


def _tiny_cache(tmp_path: Path, *, days: int = 1000, actions: int = 4):
    dates = tuple(
        (dt.date(2022, 1, 1) + dt.timedelta(days=index)).isoformat()
        for index in range(days)
    )
    action_names = tuple(["CASH", *[f"A{index}" for index in range(1, actions)]])
    bars = torch.zeros((days, actions, 5))
    trend = torch.arange(days, dtype=torch.float32).view(-1, 1) * 0.01 + 100.0
    bars[:, 1:, 0] = trend
    bars[:, 1:, 1] = trend + 1.0
    bars[:, 1:, 2] = trend - 1.0
    bars[:, 1:, 3] = trend
    bars[:, 1:, 4] = 1_000_000.0
    availability = torch.ones((days, actions), dtype=torch.bool)
    date_hash = hashlib.sha256(workflow._canonical_json(list(dates))).hexdigest()
    action_hash = hashlib.sha256(workflow._canonical_json(list(action_names))).hexdigest()
    payload = {
        "schema_version": 1,
        "feature_cache_version": workflow.FEATURE_CACHE_VERSION,
        "label": workflow.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "search_identity": "search-a",
        "base_dataset_identity": "base-a",
        "lockbox_partition_names_hash": "lockbox-a",
        "cache_identity": "cache-a",
        "actions": action_names,
        "action_hash": action_hash,
        "exchange_dates": dates,
        "date_hash": date_hash,
        "daily_ohlcv": bars,
        "availability": availability,
    }
    path = tmp_path / "daily.pt"
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_cache_walk_forward_geometry_and_parallel_episodes_are_exact(tmp_path: Path) -> None:
    path, digest = _tiny_cache(tmp_path)
    cache = workflow.load_daily_cache(path, expected_sha256=digest)
    folds = workflow.walk_forward_folds(cache)

    assert len(folds) == 3
    assert folds[0].train.size == 378
    assert folds[0].validation.size == 63
    assert folds[0].test.size == 63
    assert folds[0].config.label_horizon == folds[0].config.purge_size == 22
    assert folds[0].config.embargo_size == 5

    data, ranges = workflow.sampled_parallel_market_data(
        cache,
        start=folds[0].train.start_position,
        stop=folds[0].train.stop_position,
        num_envs=8,
        max_episode_steps=63,
        seed=17,
    )
    assert data.batch_size == 8
    assert data.horizon == 63
    assert len({left for left, _right in ranges}) == 8
    assert data.decision_ids is None
    _other_data, other_ranges = workflow.sampled_parallel_market_data(
        cache,
        start=folds[0].train.start_position,
        stop=folds[0].train.stop_position,
        num_envs=8,
        max_episode_steps=63,
        seed=18,
    )
    assert other_ranges != ranges


def test_sequence_length_21_and_burn_in_21_is_valid_contract() -> None:
    trial = workflow.TrialConfig(sequence_length=21, burn_in=21)
    assert trial.sequence_length == trial.burn_in == 21


def test_screen_worker_runs_three_folds_serially_from_one_pinned_cache(tmp_path: Path) -> None:
    path, digest = _tiny_cache(tmp_path)
    cache = workflow.load_daily_cache(path, expected_sha256=digest)
    folds = [workflow.fold_descriptor(value) for value in workflow.walk_forward_folds(cache)]
    config = workflow.TrialConfig(
        seed=17,
        hidden_dim=8,
        shared_mlp_layers=1,
        ppo_epochs=1,
        minibatch_sequences=2,
        rollout_horizon=4,
        num_envs=8,
        sequence_length=2,
        burn_in=2,
        updates=1,
        max_asset_weight=0.5,
    )
    rows = [
        {
            "global_index": index,
            "setting_id": f"S{index}",
            "fold_indexes": [0, 1, 2],
            "config": {**workflow.asdict(config), "learning_rate": config.learning_rate + index * 1e-6},
        }
        for index in range(8)
    ]
    plan = {
        "schema_version": 1,
        "search_identity": cache["search_identity"],
        "cache_identity": cache["cache_identity"],
        "cache_sha256": digest,
        "image_ref": "hpcharbor.mdanderson.edu/yding41/ml2@sha256:" + "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "orchestration_manifest_sha256": "3" * 64,
        "folds": folds,
        "trials": rows,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(workflow._canonical_json(plan))
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    receipt = workflow.run_indexed_worker(
        path,
        digest,
        plan_path,
        plan_hash,
        tmp_path / "out",
        index=0,
        device="cpu",
        acknowledgement=workflow.DEVELOPMENT_ACK,
    )

    assert receipt["setting_id"] == "S0"
    assert len(receipt["folds"]) == 3
    assert receipt["folds"][0]["sampling"]["updates"] == 1
    assert receipt["folds"][0]["fold_test_status"] == "sealed-for-post-selection-confirmation"
    assert "screen_test_metrics" not in receipt["folds"][0]
    assert (tmp_path / "out/trial-0000/screen-receipt.json").is_file()


def test_raw_lockbox_loader_filters_crossing_rows_by_exchange_date(tmp_path: Path, monkeypatch) -> None:
    dates = ["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"]
    bars = torch.zeros((4, 2, 1, 5))
    for index in range(4):
        price = 100.0 + index
        bars[index, 1, 0] = torch.tensor([price, price + 1, price - 1, price, 10.0])
    built = {
        "bars": bars,
        "bar_mask": torch.tensor([[[False], [True]]] * 4),
        "avail": torch.ones((4, 1, 2), dtype=torch.bool),
        "session_close_block": torch.zeros(4, dtype=torch.long),
        "dates": dates,
    }
    monkeypatch.setattr(workflow, "declared_universe_actions", lambda _root: ["CASH", "AAA"])
    monkeypatch.setattr(workflow, "source_symbol_to_action_index", lambda _root: {"AAA": 1})
    monkeypatch.setattr(workflow, "build_window", lambda *_args, **_kwargs: built)
    ref = workflow.PartitionRef(
        name="2025-12-30_to_2026-01-05",
        start="2025-12-30",
        end="2026-01-05",
        source_signature="x",
    )

    data, selected = workflow.load_market_data(
        tmp_path,
        [ref],
        bar_seconds=300,
        device="cpu",
        date_start=workflow.TEST_START,
    )

    assert selected == ("2026-01-02", "2026-01-05")
    assert data.decision_ids.tolist() == [[20260102]]


def test_single_model_winner_and_test_shortcuts_are_not_exposed() -> None:
    assert not hasattr(workflow, "seal_winner")
    assert not hasattr(workflow, "evaluate_test")
