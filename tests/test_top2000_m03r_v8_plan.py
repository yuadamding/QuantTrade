from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_TOP2000_DEV_SETTINGS,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v8_plan import (
    M03RV8DevelopmentTrainingPlan,
    M03RV8TrainingPlanError,
    deterministic_m03r_v8_pretraining_episode_start,
    render_m03r_v8_fold_pretraining_geometry,
)


def _plan(setting_index: int = 0) -> M03RV8DevelopmentTrainingPlan:
    return M03RV8DevelopmentTrainingPlan(
        setting_index=setting_index,
        setting_id=M03R_V8_TOP2000_DEV_SETTINGS[setting_index].setting_id,
        cache_path="/immutable/cache.pt",
        cache_sha256="a" * 64,
        output_root="/immutable/output",
        source_manifest_sha256="b" * 64,
    )


def test_all_fold_splits_stay_inside_training_and_leave_outer_score_untouched() -> None:
    folds = render_top2000_m03r_v7_development_folds(
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
    )
    for fold in folds:
        geometry = render_m03r_v8_fold_pretraining_geometry(fold)
        assert geometry.inner_validation_origin_stop_exclusive - geometry.inner_validation_start_inclusive == 63
        assert geometry.inner_validation_target_stop_exclusive == fold.training_state_stop_exclusive
        assert geometry.inner_validation_target_stop_exclusive < fold.purge_stop_exclusive
        assert geometry.validation_episode_state_stop_exclusive - geometry.validation_episode_state_start == 378
        assert geometry.receipt_sha256


def test_schedule_is_paired_across_settings_and_bounded() -> None:
    folds = render_top2000_m03r_v7_development_folds(
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
    )
    first = _plan(0)
    other = _plan(7)
    assert first.episode_schedule_sha256 == other.episode_schedule_sha256
    assert not _plan(1).alpha_pretraining_required
    assert first.alpha_pretraining_required
    for fold in folds:
        starts = [
            deterministic_m03r_v8_pretraining_episode_start(
                first, fold, completed_updates=update
            )
            for update in range(64)
        ]
        assert min(starts) >= 0
        assert max(starts) <= fold.training_state_stop_exclusive - 378
    assert {
        deterministic_m03r_v8_pretraining_episode_start(
            first, folds[-1], completed_updates=update
        )
        for update in range(64)
    } != {0}


def test_plan_identity_and_cursor_fail_closed() -> None:
    plan = _plan()
    assert plan.receipt_sha256
    with pytest.raises(M03RV8TrainingPlanError, match="drifted"):
        replace(plan, expected_world_size=1).validate()
    with pytest.raises(M03RV8TrainingPlanError, match="cursor"):
        deterministic_m03r_v8_pretraining_episode_start(
            plan,
            render_top2000_m03r_v7_development_folds(1001)[0],
            completed_updates=64,
        )
