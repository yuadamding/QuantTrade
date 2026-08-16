from __future__ import annotations

from collections import Counter

from rl_quant.training.top2000_m03r_v16_fold import (
    M03R_V16_MAXIMUM_LOCAL_ORIGIN,
    M03R_V16_MINIMUM_LOCAL_ORIGIN,
    M03RV16PanelSchedule,
    balanced_m03r_v16_training_blocks,
    render_m03r_v16_fold_geometries,
    render_m03r_v16_training_update_plan,
)


def _schedule() -> M03RV16PanelSchedule:
    geometries = render_m03r_v16_fold_geometries(1001)
    return M03RV16PanelSchedule(
        protocol_common_data_sha256="1" * 64,
        cache_sha256="2" * 64,
        asset_axis_sha256="3" * 64,
        fold_geometry_sha256=tuple(row.receipt_sha256 for row in geometries),
    )


def test_v16_fold_geometry_uses_long_support_and_all_pre2026_states() -> None:
    geometries = render_m03r_v16_fold_geometries(1001)
    assert [row.qualification_origin_start_inclusive for row in geometries] == [
        535,
        628,
        721,
        814,
        907,
    ]
    assert geometries[-1].qualification_target_stop_exclusive == 1001
    assert [len(row.eligible_training_origins) for row in geometries] == [
        129,
        222,
        315,
        408,
        501,
    ]
    assert all(
        row.inner_validation_origin_stop_exclusive
        - row.inner_validation_origin_start_inclusive
        == 63
        for row in geometries
    )
    supports = [
        set(
            range(
                row.qualification_origin_start_inclusive + 1,
                row.qualification_origin_stop_exclusive + 30,
            )
        )
        for row in geometries
    ]
    assert all(
        not left.intersection(right) for left, right in zip(supports, supports[1:])
    )


def test_v16_every_origin_appears_once_per_epoch_and_settings_are_paired() -> None:
    schedule = _schedule()
    for geometry in render_m03r_v16_fold_geometries(1001):
        for epoch in range(8):
            plans = tuple(
                render_m03r_v16_training_update_plan(
                    schedule,
                    geometry,
                    setting_index=0,
                    completed_update=epoch * geometry.training_block_count + block,
                )
                for block in range(geometry.training_block_count)
            )
            counts = Counter(origin for plan in plans for origin in plan.global_origins)
            assert tuple(sorted(counts)) == geometry.eligible_training_origins
            assert set(counts.values()) == {1}
            for plan in plans:
                local = tuple(
                    origin - plan.episode_start for origin in plan.global_origins
                )
                assert min(local) >= M03R_V16_MINIMUM_LOCAL_ORIGIN
                assert max(local) <= M03R_V16_MAXIMUM_LOCAL_ORIGIN
                assert max(local) + 31 <= 344
                assert plan.episode_stop_exclusive <= (
                    geometry.inner_validation_origin_start_inclusive
                )
        for update in range(geometry.maximum_optimizer_updates):
            plans = tuple(
                render_m03r_v16_training_update_plan(
                    schedule,
                    geometry,
                    setting_index=setting,
                    completed_update=update,
                )
                for setting in range(3)
            )
            assert len({plan.episode_start for plan in plans}) == 1
            assert len({plan.global_origins for plan in plans}) == 1
            assert len({plan.rank_origins for plan in plans}) == 1


def test_v16_training_blocks_are_balanced_with_equal_origin_coverage() -> None:
    expected = (
        (43, 43, 43),
        (56, 56, 55, 55),
        (63, 63, 63, 63, 63),
        (59, 59, 58, 58, 58, 58, 58),
        (63, 63, 63, 63, 63, 62, 62, 62),
    )
    for geometry, sizes in zip(
        render_m03r_v16_fold_geometries(1001), expected, strict=True
    ):
        blocks = balanced_m03r_v16_training_blocks(geometry)
        assert tuple(map(len, blocks)) == sizes
        assert max(map(len, blocks)) - min(map(len, blocks)) <= 1
        assert tuple(origin for block in blocks for origin in block) == (
            geometry.eligible_training_origins
        )
