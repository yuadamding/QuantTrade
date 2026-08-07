from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA,
    M03R_SEED17_TOP2000_PANEL,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA,
)
from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03RV7Top2000ArtifactBindings,
    M03RV7Top2000RuntimeProfile,
    build_m03r_v7_top2000_package_plan,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_package import (
    M03RV7Seed17PackageError,
    build_m03r_v7_seed17_top2000_package_plan,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.workflows import top2000_m03r_v7_dev as base_worker
from rl_quant.workflows import top2000_m03r_v7_seed17_dev as seed17_worker

_PACKAGE_PLAN_PATH = "/" + "mnt/package/package-plan.json"


def _artifacts() -> M03RV7Top2000ArtifactBindings:
    return M03RV7Top2000ArtifactBindings(
        source_archive_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        dependency_lock_sha256="3" * 64,
        cache_artifact_sha256="4" * 64,
        cache_manifest_sha256="5" * 64,
        data_manifest_sha256="6" * 64,
        execution_model_sha256="7" * 64,
        image_reference=f"example.invalid/quanttrade@sha256:{'8' * 64}",
        image_digest_sha256="8" * 64,
    )


def test_seed17_package_is_disjoint_and_owns_exactly_six_cells() -> None:
    seed17 = build_m03r_v7_seed17_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path=_PACKAGE_PLAN_PATH,
        benchmark_preflight_sha256="b" * 64,
    )
    legacy = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path=_PACKAGE_PLAN_PATH,
    )

    assert M03R_SEED17_TOP2000_PANEL.total_cells == 72
    assert seed17.package_plan_sha256 != legacy.package_plan_sha256
    assert len(seed17.indices) == 12
    assert all(row.fold_seed_cell_count == 6 for row in seed17.indices)
    assert all(row.paired_seeds == (17,) for row in seed17.indices)
    assert seed17.runtime_profile == M03RV7Top2000RuntimeProfile()

    with pytest.raises(M03RV7Seed17PackageError, match="six-cell inventory"):
        replace(seed17.indices[0], paired_seeds=(17, 29))


def test_one_member_fold_execution_binds_seed_validation_without_ensemble(
    tmp_path: Path,
) -> None:
    fold = render_top2000_m03r_v7_development_folds(1001)[0]
    plan = SimpleNamespace(
        protocol_sha256=M03R_SEED17_TOP2000_PROTOCOL_SHA256,
        receipt_sha256="9" * 64,
        cache_sha256="4" * 64,
        setting_index=0,
        setting_id=(
            "M03R-soft-persistence-active-alpha-hold30-top2000-seed17-dev-v1"
        ),
        runtime_setting_id=(
            "M03R-soft-persistence-active-alpha-hold30-top2000-dev-v1"
        ),
        paired_seeds=(17,),
    )
    seed_path = base_worker._seed_validation_receipt_path(
        tmp_path,
        fold_index=0,
        seed=17,
    )
    seed_payload = {
        "schema": M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA,
        "protocol_sha256": M03R_SEED17_TOP2000_PROTOCOL_SHA256,
        "setting_index": 0,
        "setting_id": plan.setting_id,
        "fold_index": 0,
        "seed": 17,
        "fold_receipt_sha256": fold.receipt_sha256,
        "sequence_receipt_sha256": "a" * 64,
        "checkpoint_file_sha256": "b" * 64,
        "model_state_sha256": "c" * 64,
        "validation_trace_artifact_sha256": "d" * 64,
        "validation_trace_sha256": "e" * 64,
        "array_sha256": {"net_returns": "f" * 64},
        "metrics": {"decision_count": 63},
        "validation_global_decision_start": 408,
        "validation_global_decision_stop_exclusive": 471,
        "first_validation_date": "2020-01-02",
        "last_validation_date": "2020-04-01",
    }
    base_worker._write_immutable_json(seed_path, seed_payload)

    path, digest = base_worker._publish_seed17_fold_execution(
        fold,
        plan=plan,
        run_root=tmp_path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(digest) == 64
    assert payload["schema"] == M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA
    assert payload["ordered_seeds"] == [17]
    assert payload["member_count"] == 1
    assert payload["chronological_return_path_count"] == 1
    assert payload["one_member_fold_execution"] is True
    assert payload["output_space_ensemble"] is False
    assert payload["five_seed_ensemble_eligible"] is False


def test_seed17_package_file_round_trip_preserves_preflight_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "package-plan.json"
    plan = build_m03r_v7_seed17_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path=str(path),
        benchmark_preflight_sha256="b" * 64,
    )
    path.write_text(
        json.dumps(
            {
                **asdict(plan),
                "schema": (
                    "rl-quant.top2000-dev.m03r-v7-seed17-"
                    "package-plan-file-v1"
                ),
            }
        ),
        encoding="utf-8",
    )

    loaded = seed17_worker.load_package_plan(
        path,
        expected_package_plan_sha256=plan.package_plan_sha256,
    )

    assert loaded == plan
    assert loaded.benchmark_preflight_sha256 == "b" * 64
    assert all(row.paired_seeds == (17,) for row in loaded.indices)
