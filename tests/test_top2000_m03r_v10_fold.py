from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import M03R_V10_SETTING_IDS
from rl_quant.training.top2000_m03r_v10_fold import (
    M03RV10FoldError,
    deterministic_m03r_v10_episode_start,
    evaluate_m03r_v10_qualification_fold,
    evaluate_m03r_v10_untouched_tail_diagnostics,
    render_m03r_v10_fold_geometry,
    run_m03r_v10_pretraining_fold_update,
)
from rl_quant.training.top2000_m03r_v10_predictive_worker import (
    M03RV10PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)


def _worker(setting: int = 0) -> M03RV10PredictiveWorkerPlan:
    return M03RV10PredictiveWorkerPlan(
        setting_index=setting,
        setting_id=M03R_V10_SETTING_IDS[setting],
        cache_path="/approved/cache.pt",
        cache_sha256="a" * 64,
        risk_source_manifest_path="/approved/risk.json",
        risk_source_manifest_file_sha256="b" * 64,
        projector_manifest_path="/approved/projector.json",
        projector_manifest_file_sha256="c" * 64,
        projector_manifest_sha256="d" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256="1" * 64,
        output_root=f"/approved/output/{setting}",
    )


def test_v10_wraps_but_does_not_relabel_the_six_v9_fold_geometries() -> None:
    folds = render_top2000_m03r_v7_development_folds(1001)
    receipts = set()
    imported = set()
    for fold in folds:
        geometry = render_m03r_v10_fold_geometry(fold)
        assert (
            geometry.qualification_origin_stop_exclusive
            - (geometry.qualification_start_inclusive)
            == 63
        )
        assert geometry.optimizer_target_stop_exclusive == (
            geometry.qualification_start_inclusive
        )
        receipts.add(geometry.receipt_sha256)
        imported.add(geometry.imported_v9_geometry_sha256)
    assert len(receipts) == len(imported) == 6
    assert receipts.isdisjoint(imported)


def test_v10_episode_schedule_is_setting_bound_and_update64_rejects() -> None:
    fold = render_top2000_m03r_v7_development_folds(1001)[5]
    first = deterministic_m03r_v10_episode_start(
        _worker(0),
        fold,
        completed_updates=17,
    )
    assert first == deterministic_m03r_v10_episode_start(
        _worker(0),
        fold,
        completed_updates=17,
    )
    assert first != deterministic_m03r_v10_episode_start(
        replace(
            _worker(0),
            setting_index=1,
            setting_id=M03R_V10_SETTING_IDS[1],
            output_root="/approved/output/1",
        ),
        fold,
        completed_updates=17,
    )
    with pytest.raises(M03RV10FoldError, match="outside"):
        deterministic_m03r_v10_episode_start(
            _worker(),
            fold,
            completed_updates=64,
        )


def test_v10_fold_routes_only_v10_steps_and_untouched_diagnostics() -> None:
    update_source = inspect.getsource(run_m03r_v10_pretraining_fold_update)
    diagnostic_source = inspect.getsource(evaluate_m03r_v10_untouched_tail_diagnostics)
    assert "train_m03r_v10_alpha_pretraining_update" in update_source
    assert "train_m03r_v9_alpha_pretraining_update" not in update_source
    assert 'split="training"' in update_source
    assert 'split="qualification"' in diagnostic_source
    assert "build_m03r_v10_fold_diagnostics" in diagnostic_source


def test_v10_qualification_pairs_diagnostics_and_wrapped_sleeve() -> None:
    source = inspect.getsource(evaluate_m03r_v10_qualification_fold)
    assert source.index("build_m03r_v10_fold_diagnostics") < source.index(
        "run_m03r_v10_simple_sleeve"
    )
    assert source.index("run_m03r_v10_simple_sleeve") < source.index(
        "build_m03r_v10_sleeve_fold_evidence"
    )
    assert "run_m03r_v9_simple_sleeve" not in source
