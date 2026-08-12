from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03R_V9_SETTING_IDS
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v9_fold import (
    M03R_V9_QUALIFICATION_ORIGINS,
    M03RV9FoldError,
    deterministic_m03r_v9_episode_start,
    render_m03r_v9_fold_geometry,
)
from rl_quant.training.top2000_m03r_v9_predictive_worker import (
    M03RV9PredictiveWorkerPlan,
)
from rl_quant.workflows.top2000_m03r_v9_predictive import (
    run_m03r_v9_predictive_worker,
)


def _worker() -> M03RV9PredictiveWorkerPlan:
    return M03RV9PredictiveWorkerPlan(
        setting_index=0,
        setting_id=M03R_V9_SETTING_IDS[0],
        cache_path="/mnt/package/cache/top2000-daily-bars.pt",
        cache_sha256="a" * 64,
        risk_source_manifest_path="/mnt/package/risk/risk-source-manifest.json",
        risk_source_manifest_file_sha256="b" * 64,
        projector_manifest_path="/mnt/package/risk/projector-manifest.json",
        projector_manifest_file_sha256="c" * 64,
        projector_manifest_sha256="d" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256="1" * 64,
        output_root="/mnt/output/completion-00-setting-00",
    )


def test_v9_fold_has_one_untouched_63_origin_update64_tail() -> None:
    folds = render_top2000_m03r_v7_development_folds(1001)
    receipts: set[str] = set()
    for fold in folds:
        geometry = render_m03r_v9_fold_geometry(fold)
        assert (
            geometry.qualification_origin_stop_exclusive
            - geometry.qualification_start_inclusive
            == M03R_V9_QUALIFICATION_ORIGINS
        )
        assert (
            geometry.qualification_target_stop_exclusive
            - geometry.qualification_origin_stop_exclusive
            == 64
        )
        assert geometry.optimizer_target_stop_exclusive == (
            geometry.qualification_start_inclusive
        )
        receipts.add(geometry.receipt_sha256)
    assert len(receipts) == 6


def test_episode_schedule_is_deterministic_and_worker_bound() -> None:
    worker = _worker()
    # Fold zero has exactly one 378-state training episode, so use a later
    # expanding fold to prove the content-bound schedule changes by worker.
    fold = render_top2000_m03r_v7_development_folds(1001)[5]
    observed = deterministic_m03r_v9_episode_start(
        worker,
        fold,
        completed_updates=17,
    )
    assert observed == deterministic_m03r_v9_episode_start(
        worker,
        fold,
        completed_updates=17,
    )
    assert observed != deterministic_m03r_v9_episode_start(
        replace(
            worker,
            setting_index=1,
            setting_id=M03R_V9_SETTING_IDS[1],
            output_root="/mnt/output/completion-01-setting-01",
        ),
        fold,
        completed_updates=17,
    )
    with pytest.raises(M03RV9FoldError, match="outside"):
        deterministic_m03r_v9_episode_start(worker, fold, completed_updates=64)


def test_worker_freezes_both_horizons_then_builds_one_common_fold_risk_state() -> None:
    source = inspect.getsource(run_m03r_v9_predictive_worker)
    checkpoint = source.index("write_immutable_m03r_v9_alpha_checkpoint")
    risk = source.index("build_m03r_v9_qualification_risk_state")
    qualification = source.index("evaluate_m03r_v9_qualification_fold")
    assert checkpoint < risk < qualification
    assert source.count("build_m03r_v9_qualification_risk_state") == 1
