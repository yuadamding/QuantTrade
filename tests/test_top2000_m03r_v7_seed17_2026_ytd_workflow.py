"""Pre-access lineage tests for the retrospective 2026-YTD workflow."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest
import torch

from rl_quant.evaluation import top2000_m03r_v7_2026_factor_data as factor_module
from rl_quant.evaluation.top2000_m03r_v7_2026_factor_data import (
    TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
    TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_COMPLETION_SCHEMA,
    M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA,
    M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA,
)
from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03RV7Top2000ArtifactBindings,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_package import (
    build_m03r_v7_seed17_top2000_package_plan,
)
from rl_quant.workflows import top2000_m03r_v7_seed17_2026_ytd as workflow_module
from rl_quant.workflows.top2000_m03r_v7_dev import CELL_RECEIPT_SCHEMA
from rl_quant.workflows.top2000_m03r_v7_seed17_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_SOURCE_FILES,
    M03R_SEED17_TOP2000_2026_YTD_SOURCE_RUN_ID,
    Top2000M03RV7Seed172026YTDWorkflowError,
    freeze_top2000_m03r_v7_seed17_2026_ytd_plan,
    load_top2000_m03r_v7_seed17_2026_ytd_plan,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _axis_digest(values: tuple[str, ...]) -> str:
    return hashlib.sha256(
        (json.dumps(list(values), separators=(",", ":"), sort_keys=True) + "\n").encode()
    ).hexdigest()


def _rank_proof() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "device": f"cuda:{rank}",
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_total_memory_bytes": 80 * 1024**3,
            "compute_capability": [9, 0],
            "allocator_oom_count": 0,
            "torchrun_restart_count": 0,
        }
        for rank in range(2)
    ]


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _artifacts(cache_sha256: str) -> M03RV7Top2000ArtifactBindings:
    image = _digest("image")
    return M03RV7Top2000ArtifactBindings(
        source_archive_sha256=_digest("source"),
        source_manifest_sha256=_digest("source-manifest"),
        dependency_lock_sha256=_digest("lock"),
        cache_artifact_sha256=cache_sha256,
        cache_manifest_sha256=_digest("cache-manifest"),
        data_manifest_sha256=_digest("data-manifest"),
        execution_model_sha256=_digest("execution-model"),
        image_reference=f"example.invalid/quanttrade@sha256:{image}",
        image_digest_sha256=image,
    )


def _dataset(root: Path, actions: tuple[str, ...]) -> None:
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "top2000_raw_time_partitioned_v1",
            "built_at_utc": "2026-06-23",
            "universe_selection_date": "2026-06-12",
            "universe_selection_method": (
                "one-day S3 dollar-volume rank intersected with common stocks "
                "active on selection date"
            ),
            "membership_mode": "static",
            "dataset_reportable": False,
            "reportability_errors": [
                "universe was selected on 2026-06-12 after the 2022-01-03 sample start",
                "static future-selected universe omits point-in-time membership and delisting history",
            ],
            "last_window": "2026-06-18_to_2026-06-24",
            "coverage": "2022-01-03 -> 2026-06-23",
            "universe": {"action_count": 1999, "cash_index": 0, "stocks": 1998},
        },
    )
    _write_json(
        root / "universe.json",
        {"action_count": 1999, "cash_index": 0, "actions": list(actions)},
    )
    # A freeze-plan implementation must not enumerate or open this namespace.
    (root / "partitions").mkdir()
    (root / "partitions" / "DO-NOT-OPEN").write_text("outcome", encoding="utf-8")


def _training_fixture(tmp_path: Path) -> dict[str, object]:
    training_root = tmp_path / M03R_SEED17_TOP2000_2026_YTD_SOURCE_RUN_ID
    pre2026_cache_path = tmp_path / "cache.pt"
    weekdays: list[str] = []
    current = date(2022, 1, 3)
    stop = date(2025, 12, 29)
    while current <= stop:
        if current.weekday() < 5:
            weekdays.append(current.isoformat())
        current += timedelta(days=1)
    dates = (*weekdays[:1000], stop.isoformat())
    actions = ("CASH", *(f"A{index}" for index in range(1, 1999)))
    daily_ohlcv = torch.zeros((len(dates), len(actions), 5), dtype=torch.float16)
    availability = torch.zeros((len(dates), len(actions)), dtype=torch.bool)
    availability[:, 0] = True
    cache_payload = {
        "schema_version": 1,
        "feature_cache_version": 1,
        "development_only": True,
        "bars_only": True,
        "bar_seconds": 300,
        "cache_identity": _digest("cache-identity"),
        "base_dataset_identity": _digest("base-dataset"),
        "search_identity": _digest("search"),
        "lockbox_partition_names_hash": _digest("lockbox-names"),
        "action_hash": _axis_digest(actions),
        "date_hash": _axis_digest(dates),
        "exchange_dates": dates,
        "actions": actions,
        "daily_ohlcv": daily_ohlcv,
        "availability": availability,
    }
    torch.save(cache_payload, pre2026_cache_path)
    cache_sha256 = _digest(pre2026_cache_path.read_bytes())
    package = build_m03r_v7_seed17_top2000_package_plan(
        artifacts=_artifacts(cache_sha256),
        plan_artifact_path="/mnt/package/package-plan.json",
        benchmark_preflight_sha256=_digest("benchmark-preflight"),
    )
    package_path = tmp_path / "package-plan.json"
    _write_json(
        package_path,
        {**asdict(package), "schema": M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA},
    )
    completion_hashes: dict[str, str] = {}
    runtime_proof: dict[str, dict[str, object]] = {}
    for row in package.indices:
        setting_root = training_root / (
            f"completion-{row.completion_index:02d}-setting-{row.setting_index:02d}"
        )
        run_root = setting_root / "training"
        plan_file_sha = _digest(f"plan-file-{row.setting_index}")
        plan_receipt_sha = _digest(f"plan-receipt-{row.setting_index}")
        cell_hashes: dict[str, str] = {}
        validation_hashes: dict[str, str] = {}
        execution_hashes: dict[str, str] = {}
        for fold in range(6):
            model_relative = (
                f"cells/fold-{fold:02d}-seed-17/model.rank-00.pt"
            )
            model_path = run_root / model_relative
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(
                f"model-setting-{row.setting_index}-fold-{fold}".encode()
            )
            model_sha = _digest(model_path.read_bytes())
            state_sha = _digest(f"state-{row.setting_index}-{fold}")
            rank_one_sha = _digest(f"rank-one-{row.setting_index}-{fold}")
            fold_sha = _digest(f"fold-{fold}")
            validation = {
                "schema": M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA,
                "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
                "setting_index": row.setting_index,
                "setting_id": row.setting_id,
                "fold_index": fold,
                "seed": 17,
                "fold_receipt_sha256": fold_sha,
                "checkpoint_file_sha256": model_sha,
                "model_state_sha256": state_sha,
                "checkpoint_selection_rule": (
                    "frozen-final-optimizer-update-no-validation-selection-v1"
                ),
                "metrics": {"decision_count": 63},
                "development_only": True,
                "outer_evaluation_authorized": False,
                "promotion_eligible": False,
            }
            validation_path = run_root / (
                f"receipts/seed-validation/fold-{fold:02d}-seed-17.json"
            )
            validation_sha = _write_json(validation_path, validation)
            validation_hashes[
                f"receipts/seed-validation/fold-{fold:02d}-seed-17.json"
            ] = validation_sha
            execution = {
                "schema": M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA,
                "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
                "setting_index": row.setting_index,
                "setting_id": row.setting_id,
                "fold_index": fold,
                "fold_receipt_sha256": fold_sha,
                "ordered_seeds": [17],
                "member_count": 1,
                "chronological_return_path_count": 1,
                "seed_validation_receipt_sha256s": [validation_sha],
                "member_checkpoint_file_sha256s": [model_sha],
                "member_model_state_sha256s": [state_sha],
                "one_member_fold_execution": True,
                "output_space_ensemble": False,
                "five_seed_ensemble_eligible": False,
                "outer_evaluation_authorized": False,
                "development_only": True,
                "promotion_eligible": False,
            }
            execution_sha = _write_json(
                run_root / f"receipts/fold-execution/fold-{fold:02d}.json",
                execution,
            )
            execution_hashes[
                f"receipts/fold-execution/fold-{fold:02d}.json"
            ] = execution_sha
            cell = {
                "schema": CELL_RECEIPT_SCHEMA,
                "mode": "full-seed17",
                "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
                "plan_file_sha256": plan_file_sha,
                "plan_receipt_sha256": plan_receipt_sha,
                "setting_index": row.setting_index,
                "setting_id": row.setting_id,
                "fold_index": fold,
                "seed": 17,
                "fold_receipt_sha256": fold_sha,
                "optimizer_steps": 64,
                "rank_model_sha256": [model_sha, rank_one_sha],
                "rank_model_state_sha256": [state_sha, state_sha],
                "seed_validation_receipt_sha256": validation_sha,
                "rank_peak_cuda_memory": _rank_proof(),
                "development_only": True,
                "promotion_eligible": False,
            }
            cell_sha = _write_json(
                run_root / f"receipts/fold-{fold:02d}-seed-17.json",
                cell,
            )
            cell_hashes[f"fold-{fold:02d}-seed-17.json"] = cell_sha
        completion = {
            "schema": M03R_SEED17_TOP2000_COMPLETION_SCHEMA,
            "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
            "setting_index": row.setting_index,
            "setting_id": row.setting_id,
            "runtime_setting_id": row.runtime_setting_id,
            "plan_file_sha256": plan_file_sha,
            "plan_receipt_sha256": plan_receipt_sha,
            "cache_sha256": cache_sha256,
            "cache_identity": cache_payload["cache_identity"],
            "search_identity": cache_payload["search_identity"],
            "action_hash": cache_payload["action_hash"],
            "world_size": 2,
            "fold_count": 6,
            "paired_seeds": [17],
            "completed_cells": 6,
            "optimizer_steps_per_cell": 64,
            "cell_receipt_sha256": cell_hashes,
            "seed_validation_receipt_sha256": validation_hashes,
            "fold_execution_receipt_sha256": execution_hashes,
            "seed_validation_receipt_count": 6,
            "fold_ensemble_receipt_count": 0,
            "fold_execution_receipt_count": 6,
            "inference_path_count": 6,
            "one_member_fold_execution_required": True,
            "five_seed_ensemble_eligible": False,
            "output_space_ensemble_required": False,
            "rank_peak_cuda_memory": _rank_proof(),
            "complete": True,
            "development_only": True,
            "future_selected_universe": True,
            "outer_evaluation_authorized": False,
            "promotion_eligible": False,
        }
        completion_path = run_root / "completion-receipt.json"
        completion_sha = _write_json(completion_path, completion)
        completion_hashes[
            str(completion_path.relative_to(training_root))
        ] = completion_sha
        runtime_proof[str(row.completion_index)] = {
            "pod_name": f"worker-{row.completion_index}",
            "pod_uid": f"uid-{row.completion_index}",
            "setting_index": row.setting_index,
            "setting_id": row.setting_id,
            "rank_runtime": _rank_proof(),
        }
    coverage = {
        "schema": "rl-quant.top2000-m03r-v7-one-seed-coverage-v1",
        "package_plan_sha256": package.package_plan_sha256,
        "source_archive_sha256": package.artifacts.source_archive_sha256,
        "expected_seed": 17,
        "expected_fold_count": 6,
        "completion_count": 12,
        "receipt_sha256": completion_hashes,
        "worker_runtime_proof": runtime_proof,
        "development_only": True,
        "promotion_eligible": False,
    }
    coverage["coverage_sha256"] = hashlib.sha256(_canonical(coverage)).hexdigest()
    coverage_path = tmp_path / "completion-coverage.json"
    _write_json(coverage_path, coverage)
    dataset_root = tmp_path / "TOP2000"
    dataset_root.mkdir()
    _dataset(dataset_root, actions)
    source_root = tmp_path / "evaluation-source"
    for relative in M03R_SEED17_TOP2000_2026_YTD_SOURCE_FILES:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen {relative}\n", encoding="utf-8")
    return {
        "training_root": training_root,
        "package": package,
        "package_path": package_path,
        "coverage_path": coverage_path,
        "dataset_root": dataset_root,
        "source_root": source_root,
        "pre2026_cache_path": pre2026_cache_path,
    }


def _freeze(fixture: dict[str, object]):
    package = fixture["package"]
    return freeze_top2000_m03r_v7_seed17_2026_ytd_plan(
        training_output_root=fixture["training_root"],
        package_plan_path=fixture["package_path"],
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[union-attr]
        pre2026_cache_path=fixture["pre2026_cache_path"],
        completion_coverage_receipt_path=fixture["coverage_path"],
        dataset_root=fixture["dataset_root"],
        evaluation_source_root=fixture["source_root"],
    )


def test_freeze_plan_binds_all_checkpoints_without_opening_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _training_fixture(tmp_path)
    protected = Path(fixture["dataset_root"]) / "partitions"
    original_open = Path.open
    original_iterdir = Path.iterdir
    original_glob = Path.glob
    original_rglob = Path.rglob

    def _protected(path: Path) -> bool:
        return path == protected or protected in path.parents

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if _protected(path):
            raise AssertionError("freeze-plan opened outcome partition contents")
        return original_open(path, *args, **kwargs)

    def guarded_iterdir(path: Path):
        if _protected(path):
            raise AssertionError("freeze-plan enumerated outcome partitions")
        return original_iterdir(path)

    def guarded_glob(
        path: Path,
        pattern: str | Path,
        *args: object,
        **kwargs: object,
    ):
        if _protected(path):
            raise AssertionError("freeze-plan globbed outcome partitions")
        return original_glob(path, pattern, *args, **kwargs)

    def guarded_rglob(
        path: Path,
        pattern: str | Path,
        *args: object,
        **kwargs: object,
    ):
        if _protected(path):
            raise AssertionError("freeze-plan recursively scanned outcome partitions")
        return original_rglob(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    monkeypatch.setattr(Path, "glob", guarded_glob)
    monkeypatch.setattr(Path, "rglob", guarded_rglob)
    plan = _freeze(fixture)

    assert len(plan.checkpoints) == 72
    assert [row.checkpoint_role for row in plan.checkpoints[:6]] == [
        "cutoff-sensitivity",
        "cutoff-sensitivity",
        "cutoff-sensitivity",
        "cutoff-sensitivity",
        "cutoff-sensitivity",
        "headline",
    ]
    # Scientific setting 7 is completion 8 in the immutable admission order.
    assert plan.checkpoints[7 * 6].completion_index == 8
    assert plan.data_namespace.partition_contents_opened is False
    assert plan.outcome_partition_contents_opened_while_freezing is False
    assert plan.factor_archives_opened_while_freezing is False
    assert plan.evaluation_source_sha256 == plan.evaluation_source.inventory_sha256
    assert len(plan.evaluation_source.files) == len(
        M03R_SEED17_TOP2000_2026_YTD_SOURCE_FILES
    )
    assert not plan.scientific_reporting_eligible
    assert not plan.promotion_eligible


def test_factor_workflow_cli_binds_retrieval_to_plan_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _training_fixture(tmp_path)
    frozen_plan = _freeze(fixture)
    plan_path = tmp_path / "frozen-plan.json"
    plan_file_sha256 = _write_json(plan_path, asdict(frozen_plan))
    score_dates: list[str] = []
    current = date(2026, 1, 2)
    stop = date(2026, 6, 23)
    while current <= stop:
        if current.weekday() < 5:
            score_dates.append(current.isoformat())
        current += timedelta(days=1)

    def zipped(member: str, lines: list[str]) -> bytes:
        target = io.BytesIO()
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr(member, "\n".join(lines))
        return target.getvalue()

    factor_contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
    payloads = {
        factor_contract.five_factor_download_url: zipped(
            TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
            [
                "Official mocked fixture",
                ",Mkt-RF,SMB,HML,RMW,CMA,RF",
                *(
                    f"{value.replace('-', '')},1.0,0.2,-0.1,0.3,0.4,0.01"
                    for value in score_dates
                ),
                "20260624,UNUSED,UNUSED,UNUSED,UNUSED,UNUSED,UNUSED",
                "",
            ],
        ),
        factor_contract.momentum_download_url: zipped(
            TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
            [
                "Official mocked fixture",
                ",Mom",
                *(f"{value.replace('-', '')},0.5" for value in score_dates),
                "20260624,UNUSED",
                "",
            ],
        ),
    }

    class MockResponse:
        def __init__(self, raw: bytes, url: str) -> None:
            self._stream = io.BytesIO(raw)
            self._url = url
            self.status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return self._url

        def read(self, size: int) -> bytes:
            return self._stream.read(size)

    def fake_urlopen(request: Any, *, timeout: int) -> MockResponse:
        assert timeout == 30
        url = str(request.full_url)
        return MockResponse(payloads[url], url)

    chronology_sha256 = _digest("mocked-2026-chronology")
    cache_sha256 = _digest("mocked-2026-cache")
    cache_receipt_path = tmp_path / "cache-stage-receipt.json"
    cache_receipt_file_sha256 = _write_json(
        cache_receipt_path,
        {"cache_sha256": cache_sha256},
    )

    def fake_load_cache_stage(
        *,
        receipt_path: str | Path,
        expected_receipt_file_sha256: str,
        plan: Any,
    ) -> tuple[dict[str, str], SimpleNamespace]:
        assert Path(receipt_path) == cache_receipt_path
        assert expected_receipt_file_sha256 == cache_receipt_file_sha256
        assert plan.receipt_sha256 == frozen_plan.receipt_sha256
        data = SimpleNamespace(
            score_return_dates=tuple(score_dates),
            identity=SimpleNamespace(receipt_sha256=chronology_sha256),
        )
        return {"cache_sha256": cache_sha256}, data

    monkeypatch.setattr(factor_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        factor_module,
        "_utc_now",
        lambda: "2026-06-25T12:00:00Z",
    )
    monkeypatch.setattr(
        workflow_module,
        "_load_verified_2026_cache_stage",
        fake_load_cache_stage,
    )
    archives = tmp_path / "official-factors"
    retrieval_receipt = tmp_path / "factor-retrieval.json"
    assert (
        workflow_module.main(
            [
                "retrieve-factors",
                "--plan",
                str(plan_path),
                "--plan-file-sha256",
                plan_file_sha256,
                "--plan-receipt-sha256",
                frozen_plan.receipt_sha256,
                "--output-directory",
                str(archives),
                "--output-receipt",
                str(retrieval_receipt),
            ]
        )
        == 0
    )
    retrieval_file_sha256 = _digest(retrieval_receipt.read_bytes())
    retrieval = json.loads(retrieval_receipt.read_bytes())
    assert retrieval["frozen_plan_file_sha256"] == plan_file_sha256
    assert retrieval["frozen_plan_receipt_sha256"] == frozen_plan.receipt_sha256
    assert retrieval["official_source_verified"] is True
    assert retrieval["caller_staged_archives"] is False

    factor_data_path = tmp_path / "factor-data.json"
    factor_stage_receipt_path = tmp_path / "factor-stage.json"
    build_arguments = [
        "build-factors",
        "--plan",
        str(plan_path),
        "--plan-file-sha256",
        plan_file_sha256,
        "--plan-receipt-sha256",
        frozen_plan.receipt_sha256,
        "--cache-receipt",
        str(cache_receipt_path),
        "--cache-receipt-file-sha256",
        cache_receipt_file_sha256,
        "--retrieval-receipt",
        str(retrieval_receipt),
        "--retrieval-receipt-file-sha256",
        retrieval_file_sha256,
        "--output-factor-data",
        str(factor_data_path),
        "--output-receipt",
        str(factor_stage_receipt_path),
    ]
    assert workflow_module.main(build_arguments) == 0
    factor_stage = json.loads(factor_stage_receipt_path.read_bytes())
    assert factor_stage["frozen_plan_file_sha256"] == plan_file_sha256
    assert factor_stage["frozen_plan_receipt_sha256"] == frozen_plan.receipt_sha256
    assert factor_stage["cache_receipt_file_sha256"] == cache_receipt_file_sha256
    assert factor_stage["chronology_receipt_sha256"] == chronology_sha256
    assert factor_stage["retrieval_receipt_file_sha256"] == retrieval_file_sha256
    assert factor_stage["retrieval_receipt_sha256"] == retrieval["receipt_sha256"]
    assert factor_stage["factor_data_file_sha256"] == _digest(
        factor_data_path.read_bytes()
    )
    assert factor_stage["post_end_source_rows_used"] == 0

    capsys.readouterr()
    retrieval["five_factor_http_status"] = 201
    retrieval_receipt.write_bytes(_canonical(retrieval))
    assert workflow_module.main(build_arguments) == 2
    assert "factor retrieval receipt file SHA-256 drifted" in capsys.readouterr().err


def test_freeze_plan_rejects_checkpoint_and_coverage_drift(tmp_path: Path) -> None:
    fixture = _training_fixture(tmp_path)
    plan = _freeze(fixture)
    first = plan.checkpoints[0]
    model = Path(fixture["training_root"]) / first.model_relative_path
    model.write_bytes(b"mutated")

    with pytest.raises(
        Top2000M03RV7Seed172026YTDWorkflowError,
        match="checkpoint",
    ):
        _freeze(fixture)

    model.write_bytes(b"model-setting-0-fold-0")
    coverage_path = Path(fixture["coverage_path"])
    coverage = json.loads(coverage_path.read_bytes())
    key = min(coverage["receipt_sha256"])
    coverage["receipt_sha256"][key] = "0" * 64
    unsigned = dict(coverage)
    unsigned.pop("coverage_sha256")
    coverage["coverage_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    coverage_path.write_bytes(_canonical(coverage))
    with pytest.raises(
        Top2000M03RV7Seed172026YTDWorkflowError,
        match="completion receipt",
    ):
        _freeze(fixture)


def test_frozen_plan_rechecks_bound_evaluation_source(tmp_path: Path) -> None:
    fixture = _training_fixture(tmp_path)
    plan = _freeze(fixture)
    plan_path = tmp_path / "frozen-plan.json"
    plan_file_sha256 = _write_json(plan_path, asdict(plan))

    loaded = load_top2000_m03r_v7_seed17_2026_ytd_plan(
        plan_path,
        expected_file_sha256=plan_file_sha256,
        expected_receipt_sha256=plan.receipt_sha256,
    )
    assert loaded == plan

    source = Path(fixture["source_root"]) / (
        M03R_SEED17_TOP2000_2026_YTD_SOURCE_FILES[0]
    )
    source.write_text("# drifted after plan freeze\n", encoding="utf-8")
    with pytest.raises(
        Top2000M03RV7Seed172026YTDWorkflowError,
        match="evaluation source file drifted",
    ):
        load_top2000_m03r_v7_seed17_2026_ytd_plan(
            plan_path,
            expected_file_sha256=plan_file_sha256,
            expected_receipt_sha256=plan.receipt_sha256,
        )


def test_freeze_plan_rejects_universe_action_axis_drift(tmp_path: Path) -> None:
    fixture = _training_fixture(tmp_path)
    universe_path = Path(fixture["dataset_root"]) / "universe.json"
    universe = json.loads(universe_path.read_bytes())
    universe["actions"][-1] = "DRIFTED"
    universe_path.write_bytes(_canonical(universe))
    with pytest.raises(
        Top2000M03RV7Seed172026YTDWorkflowError,
        match="action identities differ",
    ):
        _freeze(fixture)


def test_freeze_plan_rejects_unqualified_worker_runtime(tmp_path: Path) -> None:
    fixture = _training_fixture(tmp_path)
    coverage_path = Path(fixture["coverage_path"])
    coverage = json.loads(coverage_path.read_bytes())
    coverage["worker_runtime_proof"]["0"]["rank_runtime"][0]["gpu_name"] = (
        "NVIDIA A100-SXM4-80GB"
    )
    unsigned = dict(coverage)
    unsigned.pop("coverage_sha256")
    coverage["coverage_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    coverage_path.write_bytes(_canonical(coverage))
    with pytest.raises(
        Top2000M03RV7Seed172026YTDWorkflowError,
        match="qualified H100",
    ):
        _freeze(fixture)


def test_freeze_plan_reconciles_completion_child_inventory(tmp_path: Path) -> None:
    fixture = _training_fixture(tmp_path)
    package = fixture["package"]
    row = next(item for item in package.indices if item.completion_index == 0)  # type: ignore[union-attr]
    relative = (
        f"completion-00-setting-{row.setting_index:02d}/"
        "training/completion-receipt.json"
    )
    completion_path = Path(fixture["training_root"]) / relative
    completion = json.loads(completion_path.read_bytes())
    completion["cell_receipt_sha256"]["fold-00-seed-17.json"] = "0" * 64
    completion_sha256 = _digest(_canonical(completion))
    completion_path.write_bytes(_canonical(completion))

    coverage_path = Path(fixture["coverage_path"])
    coverage = json.loads(coverage_path.read_bytes())
    coverage["receipt_sha256"][relative] = completion_sha256
    unsigned = dict(coverage)
    unsigned.pop("coverage_sha256")
    coverage["coverage_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    coverage_path.write_bytes(_canonical(coverage))

    with pytest.raises(
        Top2000M03RV7Seed172026YTDWorkflowError,
        match="checkpoint",
    ):
        _freeze(fixture)


def test_freeze_plan_rejects_reportable_or_pit_metadata(tmp_path: Path) -> None:
    fixture = _training_fixture(tmp_path)
    manifest_path = Path(fixture["dataset_root"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["dataset_reportable"] = True
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(
        Top2000M03RV7Seed172026YTDWorkflowError,
        match="pre-access namespace",
    ):
        _freeze(fixture)
