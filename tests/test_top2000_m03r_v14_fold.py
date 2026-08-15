from __future__ import annotations

from collections import Counter

from rl_quant.training.top2000_m03r_v14_fold import (
    M03R_V14_MAXIMUM_LOCAL_TRAINING_ORIGIN,
    M03R_V14_MINIMUM_LOCAL_ORIGIN,
    M03RV14PanelEpisodeSchedule,
    render_m03r_v14_fold_geometries,
    render_m03r_v14_training_update_plan,
)


def _schedule() -> M03RV14PanelEpisodeSchedule:
    geometries = render_m03r_v14_fold_geometries(1001)
    return M03RV14PanelEpisodeSchedule(
        protocol_common_data_sha256="1" * 64,
        cache_sha256="2" * 64,
        asset_axis_sha256="3" * 64,
        fold_geometry_sha256=tuple(row.receipt_sha256 for row in geometries),
    )


def test_v14_geometry_matches_context_and_consumes_pre2026_boundary() -> None:
    geometries = render_m03r_v14_fold_geometries(1001)
    assert len(geometries) == 6
    assert geometries[0].qualification_origin_start_inclusive == 469
    assert geometries[-1].qualification_origin_start_inclusive == 934
    assert geometries[-1].qualification_target_stop_exclusive == 1001
    assert [len(row.eligible_training_origins) for row in geometries] == [
        184,
        277,
        370,
        463,
        556,
        649,
    ]
    assert [row.optimizer_updates for row in geometries] == [24, 40, 48, 64, 72, 88]
    for row in geometries:
        assert (
            row.qualification_origin_start_inclusive
            - row.qualification_episode_state_start
            == 311
        )
        assert (
            row.qualification_origin_stop_exclusive
            - 1
            - row.qualification_episode_state_start
            == 373
        )


def test_v14_every_origin_appears_once_per_epoch_with_full_context() -> None:
    schedule = _schedule()
    for geometry in render_m03r_v14_fold_geometries(1001):
        for setting_index in range(2):
            plans = tuple(
                render_m03r_v14_training_update_plan(
                    schedule,
                    geometry,
                    setting_index=setting_index,
                    completed_update=update,
                )
                for update in range(geometry.optimizer_updates)
            )
            for epoch_index in range(8):
                epoch = tuple(
                    plan for plan in plans if plan.epoch_index == epoch_index
                )
                counts = Counter(
                    origin for plan in epoch for origin in plan.global_origins
                )
                assert tuple(sorted(counts)) == geometry.eligible_training_origins
                assert set(counts.values()) == {1}
            for plan in plans:
                local = tuple(origin - plan.episode_start for origin in plan.global_origins)
                assert min(local) >= M03R_V14_MINIMUM_LOCAL_ORIGIN
                assert max(local) <= M03R_V14_MAXIMUM_LOCAL_TRAINING_ORIGIN
                assert max(local) + 1 + 3 <= 377
                assert plan.episode_stop_exclusive <= geometry.training_target_stop_exclusive
                assert tuple(sorted((*plan.rank_origins[0], *plan.rank_origins[1]))) == (
                    plan.global_origins
                )


def test_v14_loss_only_setting_change_preserves_all_scheduled_inputs() -> None:
    schedule = _schedule()
    for geometry in render_m03r_v14_fold_geometries(1001):
        for update in range(geometry.optimizer_updates):
            ranked = render_m03r_v14_training_update_plan(
                schedule,
                geometry,
                setting_index=0,
                completed_update=update,
            )
            control = render_m03r_v14_training_update_plan(
                schedule,
                geometry,
                setting_index=1,
                completed_update=update,
            )
            assert ranked.episode_start == control.episode_start
            assert ranked.episode_stop_exclusive == control.episode_stop_exclusive
            assert ranked.global_origins == control.global_origins
            assert ranked.rank_origins == control.rank_origins
