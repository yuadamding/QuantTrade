from __future__ import annotations

from dataclasses import asdict, replace
import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from rl_quant.workflows import top2000_pipeline as pipeline
from rl_quant.workflows import top2000_ppo as ppo


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _runtime() -> dict[str, str]:
    return {
        "image_ref": "hpcharbor.mdanderson.edu/yding41/ml2@sha256:" + "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "orchestration_manifest_sha256": "3" * 64,
    }


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pipeline._canonical_json(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_cache(tmp_path: Path, *, days: int = 1000, actions: int = 4) -> tuple[Path, str]:
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
    payload = {
        "schema_version": 1,
        "feature_cache_version": ppo.FEATURE_CACHE_VERSION,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "search_identity": _digest("search"),
        "base_dataset_identity": _digest("base"),
        "lockbox_partition_names_hash": _digest("lockbox"),
        "cache_identity": _digest("cache"),
        "actions": action_names,
        "action_hash": hashlib.sha256(ppo._canonical_json(list(action_names))).hexdigest(),
        "exchange_dates": dates,
        "date_hash": hashlib.sha256(ppo._canonical_json(list(dates))).hexdigest(),
        "daily_ohlcv": bars,
        "availability": availability,
    }
    path = tmp_path / "cache.pt"
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _folds(cache_path: Path, cache_sha256: str) -> list[dict[str, Any]]:
    cache = ppo.load_daily_cache(cache_path, expected_sha256=cache_sha256)
    return [ppo.fold_descriptor(value) for value in ppo.walk_forward_folds(cache)]


def _metric(mean: float, cost_bps: float, *, observations: int = 63) -> dict[str, Any]:
    returns = [mean + (0.00005 if index % 2 else -0.00005) for index in range(observations)]
    return pipeline._series_metrics(
        returns,
        [0.05] * observations,
        risky_available=[True] * observations,
        cost_bps=cost_bps,
    )


def _cash_metric(cost_bps: float, *, observations: int = 63) -> dict[str, Any]:
    return pipeline._series_metrics(
        [0.0] * observations,
        [0.0] * observations,
        risky_available=[True] * observations,
        cost_bps=cost_bps,
    )


def _ladder(mean: float, *, observations: int = 63) -> dict[str, dict[str, Any]]:
    return {
        "gross_0bp": _metric(mean + 0.00015, 0.0, observations=observations),
        "base": _metric(mean + 0.00010, 10.0, observations=observations),
        "stress_20bp": _metric(mean, 20.0, observations=observations),
        "stress_40bp": _metric(mean - 0.00010, 40.0, observations=observations),
    }


def _cash_ladder(*, observations: int = 63) -> dict[str, dict[str, Any]]:
    return {
        key: _cash_metric(cost, observations=observations)
        for key, cost in pipeline._COST_BY_KEY.items()
    }


def _selection(folds: list[dict[str, Any]]) -> dict[str, Any]:
    settings = []
    for index, hidden in ((6, 128), (7, 192)):
        settings.append(
            {
                "setting_index": index,
                "setting_id": f"S{index}",
                "trial_config": asdict(ppo.TrialConfig(hidden_dim=hidden)),
                "ranking_metrics": {},
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": pipeline.SCREEN_SELECTION_KIND,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "screen_plan_sha256": _digest("screen-plan"),
        "cache_identity": _digest("cache"),
        "cache_sha256": _digest("cache-file"),
        "search_identity": _digest("search"),
        "base_dataset_identity": _digest("base"),
        "lockbox_partition_names_hash": _digest("lockbox"),
        "runtime": _runtime(),
        "folds": folds,
        "receipt_sha256s": [],
        "eligibility_gates": {},
        "ranking_rule": [],
        "candidate_summaries": [],
        "selected_settings": settings,
    }
    payload["selection_identity"] = pipeline._artifact_identity(payload, "selection_identity")
    return payload


def _winner(selection: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": pipeline.CONFIRMATION_WINNER_KIND,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "confirmation_plan_sha256": _digest("confirmation-plan"),
        "receipt_set_manifest_sha256": _digest("confirmation-set"),
        "cache_identity": selection["cache_identity"],
        "cache_sha256": selection["cache_sha256"],
        "search_identity": selection["search_identity"],
        "base_dataset_identity": selection["base_dataset_identity"],
        "lockbox_partition_names_hash": selection["lockbox_partition_names_hash"],
        "screen_selection_sha256": _digest("selection-file"),
        "selection_identity": selection["selection_identity"],
        "receipt_sha256s": [],
        "robust_gates": {},
        "ranking_rule": [],
        "candidate_summaries": [],
        "winning_setting": {
            **selection["selected_settings"][0],
            "confirmation_metrics": {},
        },
    }
    payload["winner_identity"] = pipeline._artifact_identity(payload, "winner_identity")
    return payload


def test_trial_config_rejects_bool_nonfinite_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        ppo.TrialConfig(hidden_dim=True)
    with pytest.raises(ValueError, match="learning_rate"):
        ppo.TrialConfig(learning_rate=float("nan"))
    with pytest.raises(ValueError, match="discount"):
        ppo.TrialConfig(discount=1.01)
    with pytest.raises(ValueError, match="sequence_length"):
        ppo.TrialConfig(sequence_length=64, rollout_horizon=63)


def test_metric_summaries_must_recompute_from_daily_series() -> None:
    metric = _metric(0.001, 20.0)
    pipeline._validate_metric_payload(metric, "metric")
    tampered = {**metric, "net_total_return": metric["net_total_return"] + 0.1}
    with pytest.raises(pipeline.PipelineValidationError, match="does not recompute"):
        pipeline._validate_metric_payload(tampered, "metric")
    availability_tampered = {**metric, "decision_coverage": 0.5}
    with pytest.raises(pipeline.PipelineValidationError, match="does not recompute"):
        pipeline._validate_metric_payload(availability_tampered, "metric")


def test_confirmation_and_refit_plan_builders_freeze_exact_seed_matrix() -> None:
    folds = [
        {"fold_index": index, "fold_id": f"fold-{index}"}
        for index in range(3)
    ]
    selection = _selection(folds)
    confirmation = pipeline.build_confirmation_plan(
        selection,
        screen_selection_sha256=_digest("selection-file"),
        runtime=_runtime(),
    )

    assert [row["seed"] for row in confirmation["trials"]] == [17, 29, 43, 71] * 2
    assert [row["setting_id"] for row in confirmation["trials"]] == ["S6"] * 4 + ["S7"] * 4
    assert confirmation["runtime"] == _runtime()

    winner = _winner(selection)
    refit = pipeline.build_refit_plan(
        winner,
        confirmation_winner_sha256=_digest("winner-file"),
        runtime=_runtime(),
    )
    assert [row["seed"] for row in refit["trials"]] == [17, 29, 43, 71]
    broken = {**refit, "trials": [*refit["trials"]]}
    broken["trials"][0] = {**broken["trials"][0], "seed": True}
    with pytest.raises(pipeline.PipelineValidationError, match="ordering"):
        pipeline.validate_refit_plan(broken)


def test_receipt_set_uses_explicit_paths_same_bytes_and_detects_mutation(tmp_path: Path) -> None:
    plan_sha256 = _digest("confirmation-plan")
    paths: list[str] = []
    for index in range(8):
        relative = f"index-{index:04d}/trial-{index:04d}/confirmation-receipt.json"
        paths.append(relative)
        _write_json(
            tmp_path / relative,
            {
                "schema_version": 1,
                "artifact_kind": pipeline.CONFIRMATION_RECEIPT_KIND,
                "label": ppo.DEVELOPMENT_LABEL,
                "development_only": True,
                "bars_only": True,
                "confirmation_plan_sha256": plan_sha256,
                "global_index": index,
            },
        )
    manifest = pipeline.build_receipt_set_manifest(
        "confirmation", plan_sha256, tmp_path, paths
    )
    manifest_path = tmp_path / "receipt-set.json"
    manifest_sha256 = _write_json(manifest_path, manifest)
    loaded_paths, _loaded = pipeline._load_receipt_set(
        manifest_path,
        manifest_sha256,
        root=tmp_path,
        kind=pipeline.CONFIRMATION_RECEIPT_SET_KIND,
        plan_sha256=plan_sha256,
        expected_count=8,
    )
    original_payload = loaded_paths[0].payload
    (tmp_path / paths[0]).write_text("changed")
    assert loaded_paths[0].payload == original_payload
    with pytest.raises(pipeline.PipelineValidationError, match="SHA-256 mismatch"):
        pipeline.load_receipt_set_manifest(
            manifest_path,
            expected_sha256=manifest_sha256,
            stage="confirmation",
            plan_sha256=plan_sha256,
            receipts_root=tmp_path,
        )
    with pytest.raises(pipeline.PipelineValidationError, match="unsafe"):
        pipeline.build_receipt_set_manifest(
            "confirmation",
            plan_sha256,
            tmp_path,
            ["../escape.json", *paths[1:]],
        )
    with pytest.raises(pipeline.PipelineValidationError, match="non-canonical"):
        pipeline.build_receipt_set_manifest(
            "confirmation",
            plan_sha256,
            tmp_path,
            [paths[0].replace("/", "//", 1), *paths[1:]],
        )


def test_screen_aggregation_applies_fail_closed_gates_and_deterministic_ranking(
    tmp_path: Path,
) -> None:
    cache_path, cache_sha256 = _tiny_cache(tmp_path)
    cache = ppo.load_daily_cache(cache_path, expected_sha256=cache_sha256)
    folds = [ppo.fold_descriptor(value) for value in ppo.walk_forward_folds(cache)]
    base_trial = ppo.TrialConfig(hidden_dim=16)
    rows = [
        {
            "global_index": index,
            "setting_id": f"S{index}",
            "fold_indexes": [0, 1, 2],
            "config": {
                **asdict(base_trial),
                "learning_rate": base_trial.learning_rate + index * 1e-6,
            },
        }
        for index in range(8)
    ]
    plan = {
        "schema_version": 1,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "cache_identity": cache["cache_identity"],
        "cache_sha256": cache_sha256,
        "search_identity": cache["search_identity"],
        "data_evidence": {"base_dataset_identity": cache["base_dataset_identity"]},
        **_runtime(),
        "folds": folds,
        "trials": rows,
    }
    plan_path = tmp_path / "screen-plan.json"
    plan_sha256 = _write_json(plan_path, plan)
    relative_paths: list[str] = []
    for index, row in enumerate(rows):
        directory = tmp_path / "receipts" / f"trial-{index:04d}"
        directory.mkdir(parents=True)
        fold_receipts = []
        for fold_index, descriptor in enumerate(folds):
            ladder = _ladder(0.0002 + index * 0.00002)
            checkpoint = {
                "schema_version": 1,
                "label": ppo.DEVELOPMENT_LABEL,
                "development_only": True,
                "screen_plan_sha256": plan_sha256,
                "runtime": _runtime(),
                "cache_identity": cache["cache_identity"],
                "setting_index": index,
                "setting_id": row["setting_id"],
                "fold": descriptor,
                "trial_config": row["config"],
                "model_state_dict": {"dummy": torch.tensor([1.0])},
            }
            checkpoint_path = directory / f"fold-{fold_index:02d}.pt"
            torch.save(checkpoint, checkpoint_path)
            fold_receipts.append(
                {
                    "fold": descriptor,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    "validation_metrics": ladder["base"],
                    "validation_cost_ladder": ladder,
                    "fold_test_status": "sealed-for-post-selection-confirmation",
                }
            )
        receipt = {
            "schema_version": 1,
            "artifact_kind": "screen-validation-only",
            "label": ppo.DEVELOPMENT_LABEL,
            "development_only": True,
            "bars_only": True,
            "screen_plan_sha256": plan_sha256,
            "runtime": _runtime(),
            "search_identity": cache["search_identity"],
            "base_dataset_identity": cache["base_dataset_identity"],
            "lockbox_partition_names_hash": cache["lockbox_partition_names_hash"],
            "cache_identity": cache["cache_identity"],
            "cache_sha256": cache_sha256,
            "setting_index": index,
            "setting_id": row["setting_id"],
            "trial_config": row["config"],
            "folds": fold_receipts,
            "aggregate": {},
        }
        relative = f"trial-{index:04d}/screen-receipt.json"
        relative_paths.append(relative)
        _write_json(tmp_path / "receipts" / relative, receipt)
    receipt_set = pipeline.build_receipt_set_manifest(
        "screen", plan_sha256, tmp_path / "receipts", relative_paths
    )
    receipt_set_path = tmp_path / "screen-receipt-set.json"
    receipt_set_sha256 = _write_json(receipt_set_path, receipt_set)

    result = pipeline.aggregate_screen_receipts(
        cache_path,
        cache_sha256,
        plan_path,
        plan_sha256,
        tmp_path / "receipts",
        receipt_set_path,
        receipt_set_sha256,
        tmp_path / "selection.json",
        acknowledgement=ppo.DEVELOPMENT_ACK,
    )

    assert [value["setting_id"] for value in result["selected_settings"]] == ["S7", "S6"]
    assert all(value["eligible"] for value in result["candidate_summaries"])
    assert result["receipt_set_manifest_sha256"] == receipt_set_sha256


def _confirmation_setting_rows(
    *,
    setting_id: str,
    setting_index: int,
    mean: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    base = ppo.TrialConfig(hidden_dim=16)
    for seed in pipeline.CONFIRMATION_SEEDS:
        config = asdict(replace(base, seed=seed))
        row = {
            "global_index": len(rows),
            "setting_index": setting_index,
            "setting_id": setting_id,
            "seed": seed,
            "config": config,
        }
        folds = []
        for _fold in range(3):
            model = _ladder(mean)
            equal_weight = _ladder(0.00005)
            cash = _cash_ladder()
            folds.append(
                {
                    "test_cost_ladder": model,
                    "baselines": {
                        "cash": cash,
                        "equal_weight_available_assets": equal_weight,
                    },
                }
            )
        receipt = {"folds": folds, "aggregate": pipeline._confirmation_aggregate(folds)}
        rows.append((row, receipt))
    return rows


def test_confirmation_candidate_has_bounded_robust_gates_and_can_fail_closed() -> None:
    positive = pipeline._confirmation_candidate(
        _confirmation_setting_rows(setting_id="S1", setting_index=1, mean=0.0005)
    )
    assert positive["eligible"]
    assert positive["metrics"]["positive_seed_count_20bp"] == 4
    assert positive["metrics"]["bootstrap_95pct_mean_daily_return_20bp"]["samples"] == 2000

    negative = pipeline._confirmation_candidate(
        _confirmation_setting_rows(setting_id="S2", setting_index=2, mean=-0.0005)
    )
    assert not negative["eligible"]
    assert not negative["gates"]["seed_averaged_pooled_return_20bp_positive"]


def _sealed_ensemble(tmp_path: Path, *, actions: int = 4) -> tuple[Path, str, ppo.TrialConfig]:
    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    runtime = _runtime()
    trial = ppo.TrialConfig(hidden_dim=8, max_asset_weight=0.5)
    template = ppo.SharedAssetRecurrentActorCritic(
        observation_key=ppo.BarsOnlyObservationAdapter.observation_key,
        asset_feature_dim=ppo.BarsOnlyObservationAdapter.asset_feature_dim,
        hidden_dim=trial.hidden_dim,
        action_dim=actions,
        shared_mlp_layers=trial.shared_mlp_layers,
    )
    members = []
    for index, seed in enumerate(pipeline.CONFIRMATION_SEEDS):
        config = asdict(replace(trial, seed=seed))
        checkpoint = {
            "schema_version": 1,
            "artifact_kind": pipeline.REFIT_MEMBER_KIND,
            "label": ppo.DEVELOPMENT_LABEL,
            "development_only": True,
            "bars_only": True,
            "refit_plan_sha256": _digest("refit-plan"),
            "confirmation_winner_sha256": _digest("winner"),
            "winner_identity": _digest("winner-identity"),
            "cache_identity": _digest("cache"),
            "cache_sha256": _digest("cache-file"),
            "search_identity": _digest("search"),
            "base_dataset_identity": _digest("base"),
            "lockbox_partition_names_hash": _digest("lockbox"),
            "runtime": runtime,
            "global_index": index,
            "setting_index": 6,
            "setting_id": "S6",
            "seed": seed,
            "trial_config": config,
            "model_state_dict": template.state_dict(),
        }
        name = f"member-{index:02d}-seed-{seed}.pt"
        path = ensemble / name
        torch.save(checkpoint, path)
        members.append(
            {
                "member_index": index,
                "name": name,
                "seed": seed,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "refit_receipt_sha256": _digest(f"receipt-{index}"),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": pipeline.ENSEMBLE_KIND,
        "label": ppo.DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "not_reportable": True,
        "refit_plan_sha256": _digest("refit-plan"),
        "receipt_set_manifest_sha256": _digest("refit-set"),
        "confirmation_winner_sha256": _digest("winner"),
        "winner_identity": _digest("winner-identity"),
        "cache_identity": _digest("cache"),
        "cache_sha256": _digest("cache-file"),
        "search_identity": _digest("search"),
        "base_dataset_identity": _digest("base"),
        "lockbox_partition_names_hash": _digest("lockbox"),
        "setting_index": 6,
        "setting_id": "S6",
        "trial_config": asdict(trial),
        "seeds": list(pipeline.CONFIRMATION_SEEDS),
        "runtime": runtime,
        "members": members,
        "action_combination": "arithmetic-mean-of-four-deterministic-requested-weight-vectors",
    }
    manifest["ensemble_identity"] = pipeline._artifact_identity(manifest, "ensemble_identity")
    digest = _write_json(ensemble / "ensemble-manifest.json", manifest)
    return ensemble, digest, trial


def test_four_model_evaluation_averages_deterministic_requested_weights() -> None:
    data = ppo.synthetic_market(actions=4, dates=12)
    trial = ppo.TrialConfig(hidden_dim=8, max_asset_weight=0.5, cost_bps=10.0)
    model = ppo.SharedAssetRecurrentActorCritic(
        observation_key=ppo.BarsOnlyObservationAdapter.observation_key,
        asset_feature_dim=ppo.BarsOnlyObservationAdapter.asset_feature_dim,
        hidden_dim=trial.hidden_dim,
        action_dim=data.num_assets,
        shared_mlp_layers=trial.shared_mlp_layers,
    )
    trials = [replace(trial, seed=seed) for seed in pipeline.CONFIRMATION_SEEDS]
    ensemble = pipeline._evaluate_requested_weight_ensemble(
        [model, model, model, model],
        trials,
        data,
        cost_bps=10.0,
    )
    single = ppo.evaluate_model(model, data, trial)
    assert ensemble["daily_net_returns"] == pytest.approx(single["daily_net_returns"])
    assert ensemble["net_total_return"] == pytest.approx(single["net_total_return"])


def test_refit_receipts_are_validated_before_exact_four_member_seal(tmp_path: Path) -> None:
    folds = [{"fold_index": index, "fold_id": f"fold-{index}"} for index in range(3)]
    winner = _winner(_selection(folds))
    winner_path = tmp_path / "winner.json"
    winner_sha256 = _write_json(winner_path, winner)
    plan = pipeline.build_refit_plan(
        winner,
        confirmation_winner_sha256=winner_sha256,
        runtime=_runtime(),
    )
    plan_path = tmp_path / "refit-plan.json"
    plan_sha256 = _write_json(plan_path, plan)
    receipts_root = tmp_path / "refits"
    relative_paths: list[str] = []
    for index, row in enumerate(plan["trials"]):
        directory = receipts_root / f"trial-{index:04d}"
        directory.mkdir(parents=True)
        checkpoint = {
            "schema_version": 1,
            "artifact_kind": pipeline.REFIT_MEMBER_KIND,
            "label": ppo.DEVELOPMENT_LABEL,
            "development_only": True,
            "bars_only": True,
            "refit_plan_sha256": plan_sha256,
            "confirmation_winner_sha256": winner_sha256,
            "winner_identity": winner["winner_identity"],
            "cache_identity": plan["cache_identity"],
            "cache_sha256": plan["cache_sha256"],
            "search_identity": plan["search_identity"],
            "base_dataset_identity": plan["base_dataset_identity"],
            "lockbox_partition_names_hash": plan["lockbox_partition_names_hash"],
            "runtime": _runtime(),
            "global_index": index,
            "setting_index": row["setting_index"],
            "setting_id": row["setting_id"],
            "seed": row["seed"],
            "trial_config": row["config"],
            "model_state_dict": {"dummy": torch.tensor([float(index)])},
        }
        checkpoint_path = directory / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        receipt = {
            "schema_version": 1,
            "artifact_kind": pipeline.REFIT_MEMBER_KIND,
            "label": ppo.DEVELOPMENT_LABEL,
            "development_only": True,
            "bars_only": True,
            **{
                key: checkpoint[key]
                for key in (
                    "refit_plan_sha256",
                    "confirmation_winner_sha256",
                    "winner_identity",
                    "cache_identity",
                    "cache_sha256",
                    "search_identity",
                    "base_dataset_identity",
                    "lockbox_partition_names_hash",
                    "runtime",
                    "global_index",
                    "setting_index",
                    "setting_id",
                    "seed",
                    "trial_config",
                )
            },
            "full_pre2026_training_range": {
                "decision_start": 0,
                "decision_stop": 999,
                "decision_count": 999,
                "first_decision_date": "2022-01-03",
                "last_decision_date": "2025-12-26",
                "final_label_support_date": "2025-12-29",
                "cutoff_exclusive": "2026-01-01",
            },
            "sampling": {},
            "checkpoint": "checkpoint.pt",
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "last_training_metrics": {},
        }
        relative = f"trial-{index:04d}/refit-receipt.json"
        relative_paths.append(relative)
        _write_json(receipts_root / relative, receipt)
    receipt_set = pipeline.build_receipt_set_manifest(
        "refit", plan_sha256, receipts_root, relative_paths
    )
    receipt_set_path = tmp_path / "refit-receipt-set.json"
    receipt_set_sha256 = _write_json(receipt_set_path, receipt_set)

    manifest = pipeline.seal_refit_ensemble(
        plan_path,
        plan_sha256,
        winner_path,
        winner_sha256,
        receipts_root,
        receipt_set_path,
        receipt_set_sha256,
        tmp_path / "sealed",
        acknowledgement=ppo.DEVELOPMENT_ACK,
    )

    assert manifest["seeds"] == [17, 29, 43, 71]
    assert len(manifest["members"]) == 4
    assert (tmp_path / "sealed/ensemble-manifest.json").is_file()
    assert not any(path.name.startswith(".sealed.tmp") for path in tmp_path.iterdir())


def test_2026_access_marker_precedes_test_plan_and_is_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensemble, manifest_sha256, _trial = _sealed_ensemble(tmp_path)
    root = tmp_path / "data"
    root.mkdir()
    search = ppo.SearchPlan(
        protocol_version=1,
        label=ppo.DEVELOPMENT_LABEL,
        development_only=True,
        development_reasons=("development",),
        base_dataset_identity=_digest("base"),
        search_identity=_digest("search"),
        lockbox_partition_names_hash=_digest("lockbox"),
        train=(),
        validation=(),
        bar_seconds=300,
    )
    evaluation = ppo.EvaluationPlan(
        protocol_version=1,
        label=ppo.DEVELOPMENT_LABEL,
        development_only=True,
        base_dataset_identity=_digest("base"),
        search_identity=_digest("search"),
        lockbox_partition_names_hash=_digest("lockbox"),
        test_identity=_digest("test"),
        test=(),
        bar_seconds=300,
    )
    opened: list[bool] = []

    def build_evaluation(*_args, **_kwargs):
        opened.append((ensemble / "test-accessed.json").is_file())
        return evaluation

    monkeypatch.setattr(ppo, "declared_universe_actions", lambda _root: ["CASH", "A", "B", "C"])
    monkeypatch.setattr(ppo, "build_search_plan", lambda *_args, **_kwargs: search)
    monkeypatch.setattr(ppo, "build_evaluation_plan", build_evaluation)
    monkeypatch.setattr(
        ppo,
        "load_market_data",
        lambda *_args, **_kwargs: (
            ppo.synthetic_market(actions=4, dates=12),
            tuple(f"2026-01-{day:02d}" for day in range(1, 13)),
        ),
    )

    result = pipeline.evaluate_2026_ensemble(
        root,
        ensemble,
        manifest_sha256,
        tmp_path / "test-result.json",
        bar_seconds=300,
        device="cpu",
        acknowledgement=ppo.DEVELOPMENT_ACK,
    )
    assert opened == [True]
    assert result["not_reportable"]
    assert result["baselines"]["cash"]["stress_40bp"]["net_total_return"] == 0.0
    with pytest.raises(FileExistsError):
        pipeline.evaluate_2026_ensemble(
            root,
            ensemble,
            manifest_sha256,
            tmp_path / "second-result.json",
            bar_seconds=300,
            device="cpu",
            acknowledgement=ppo.DEVELOPMENT_ACK,
        )
    assert opened == [True]
