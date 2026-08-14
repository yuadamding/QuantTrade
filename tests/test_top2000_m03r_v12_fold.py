from __future__ import annotations

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v12_fold import (
    M03R_V12_MAX_LOCAL_ORIGIN,
    M03R_V12_MINIMUM_RESIDUAL_ORIGIN_STATE_INDEX,
    render_m03r_v12_training_shard_plan,
)
from rl_quant.training.top2000_m03r_v12_predictive_worker import (
    M03RV12PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v12_schedule import (
    M03RV12PanelEpisodeSchedule,
)


def test_every_v12_setting_gets_identical_episode_and_rank_shards() -> None:
    folds = render_top2000_m03r_v7_development_folds(1001)
    schedule = M03RV12PanelEpisodeSchedule(
        protocol_common_data_sha256="a" * 64,
        cache_sha256="b" * 64,
        fold_geometry_sha256=tuple(
            render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
        ),
    )
    plans = []
    for setting in range(3):
        worker = M03RV12PredictiveWorkerPlan(
            setting_index=setting,
            setting_id=M03R_V12_SETTING_IDS[setting],
            output_root=f"/approved/v12/{setting}",
            cache_path="/approved/cache.pt",
            initial_parameter_state_path="/approved/common-initial-state.pt",
            panel_episode_schedule_sha256=schedule.receipt_sha256,
            initial_parameter_state_file_sha256="9" * 64,
            initial_parameter_state_sha256="c" * 64,
            cache_sha256="b" * 64,
            risk_source_manifest_path="/approved/risk.json",
            risk_source_manifest_file_sha256="d" * 64,
            projector_manifest_path="/approved/projector.json",
            projector_manifest_file_sha256="2" * 64,
            projector_manifest_sha256="3" * 64,
            projector_binding_sha256="e" * 64,
            source_manifest_sha256="f" * 64,
            source_archive_sha256="1" * 64,
            structural_preflight_path="/approved/preflight.json",
            structural_preflight_file_sha256="4" * 64,
            structural_preflight_receipt_sha256="5" * 64,
        )
        plans.append(
            render_m03r_v12_training_shard_plan(
                worker, schedule, folds[4], completed_update=37
            )
        )
    assert len({row.episode_start for row in plans}) == 1
    assert min(plans[0].global_origins) >= (
        M03R_V12_MINIMUM_RESIDUAL_ORIGIN_STATE_INDEX
    )
    assert len({row.global_origins for row in plans}) == 1
    assert len({row.rank_origins for row in plans}) == 1


def test_fold_zero_never_schedules_pre_risk_history_origins() -> None:
    folds = render_top2000_m03r_v7_development_folds(1001)
    schedule = M03RV12PanelEpisodeSchedule(
        protocol_common_data_sha256="a" * 64,
        cache_sha256="b" * 64,
        fold_geometry_sha256=tuple(
            render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
        ),
    )
    worker = M03RV12PredictiveWorkerPlan(
        setting_index=0,
        setting_id=M03R_V12_SETTING_IDS[0],
        output_root="/approved/v12/0",
        cache_path="/approved/cache.pt",
        initial_parameter_state_path="/approved/common-initial-state.pt",
        panel_episode_schedule_sha256=schedule.receipt_sha256,
        initial_parameter_state_file_sha256="9" * 64,
        initial_parameter_state_sha256="c" * 64,
        cache_sha256="b" * 64,
        risk_source_manifest_path="/approved/risk.json",
        risk_source_manifest_file_sha256="d" * 64,
        projector_manifest_path="/approved/projector.json",
        projector_manifest_file_sha256="2" * 64,
        projector_manifest_sha256="3" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256="1" * 64,
        structural_preflight_path="/approved/preflight.json",
        structural_preflight_file_sha256="4" * 64,
        structural_preflight_receipt_sha256="5" * 64,
    )
    plans = tuple(
        render_m03r_v12_training_shard_plan(
            worker, schedule, folds[0], completed_update=update
        )
        for update in range(64)
    )
    assert min(min(plan.global_origins) for plan in plans) >= (
        M03R_V12_MINIMUM_RESIDUAL_ORIGIN_STATE_INDEX
    )


def test_every_fold_update_origin_has_all_four_horizons_inside_episode() -> None:
    folds = render_top2000_m03r_v7_development_folds(1001)
    schedule = M03RV12PanelEpisodeSchedule(
        protocol_common_data_sha256="a" * 64,
        cache_sha256="b" * 64,
        fold_geometry_sha256=tuple(
            render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
        ),
    )
    for setting_index, setting_id in enumerate(M03R_V12_SETTING_IDS):
        worker = M03RV12PredictiveWorkerPlan(
            setting_index=setting_index,
            setting_id=setting_id,
            output_root=f"/approved/v12/{setting_index}",
            cache_path="/approved/cache.pt",
            initial_parameter_state_path="/approved/common-initial-state.pt",
            panel_episode_schedule_sha256=schedule.receipt_sha256,
            initial_parameter_state_file_sha256="9" * 64,
            initial_parameter_state_sha256="c" * 64,
            cache_sha256="b" * 64,
            risk_source_manifest_path="/approved/risk.json",
            risk_source_manifest_file_sha256="d" * 64,
            projector_manifest_path="/approved/projector.json",
            projector_manifest_file_sha256="2" * 64,
            projector_manifest_sha256="3" * 64,
            projector_binding_sha256="e" * 64,
            source_manifest_sha256="f" * 64,
            source_archive_sha256="1" * 64,
            structural_preflight_path="/approved/preflight.json",
            structural_preflight_file_sha256="4" * 64,
            structural_preflight_receipt_sha256="5" * 64,
        )
        for fold in folds:
            geometry = render_m03r_v10_fold_geometry(fold)
            for completed_update in range(64):
                plan = render_m03r_v12_training_shard_plan(
                    worker,
                    schedule,
                    fold,
                    completed_update=completed_update,
                )
                assert max(plan.global_origins) - plan.episode_start <= (
                    M03R_V12_MAX_LOCAL_ORIGIN
                )
                assert max(plan.global_origins) + 63 + 1 <= (
                    geometry.optimizer_target_stop_exclusive
                )
