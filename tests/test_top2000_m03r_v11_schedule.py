from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.training.top2000_m03r_v11_fold import M03RV11TrainingShardPlan
from rl_quant.training.top2000_m03r_v11_schedule import (
    M03RV11PanelEpisodeSchedule,
    M03RV11ScheduleError,
    build_m03r_v11_paired_input_receipt,
    deterministic_m03r_v11_episode_start,
    m03r_v11_complementary_rank_shards,
)


def _schedule() -> M03RV11PanelEpisodeSchedule:
    return M03RV11PanelEpisodeSchedule(
        protocol_common_data_sha256="a" * 64,
        cache_sha256="b" * 64,
        fold_geometry_sha256=tuple(f"{index + 1:x}" * 64 for index in range(6)),
    )


def test_episode_schedule_is_paired_across_every_setting() -> None:
    schedule = _schedule()
    starts_by_setting = {
        setting: tuple(
            deterministic_m03r_v11_episode_start(
                schedule,
                fold_index=fold,
                completed_updates=update,
                admissible_start_count=257,
            )
            for fold in range(6)
            for update in range(64)
        )
        for setting in range(3)
    }
    assert starts_by_setting[0] == starts_by_setting[1] == starts_by_setting[2]
    assert len(set(starts_by_setting[0])) > 1


def test_rank_shards_are_complementary_and_schedule_identity_matters() -> None:
    origins = tuple(range(12, 24))
    rank_zero, rank_one = m03r_v11_complementary_rank_shards(origins)
    assert set(rank_zero).isdisjoint(rank_one)
    assert tuple(sorted((*rank_zero, *rank_one))) == origins
    first = deterministic_m03r_v11_episode_start(
        _schedule(), fold_index=2, completed_updates=9, admissible_start_count=251
    )
    changed = deterministic_m03r_v11_episode_start(
        replace(_schedule(), cache_sha256="c" * 64),
        fold_index=2,
        completed_updates=9,
        admissible_start_count=251,
    )
    assert first != changed
    with pytest.raises(M03RV11ScheduleError):
        m03r_v11_complementary_rank_shards((1,))


def test_paired_input_receipt_is_setting_neutral_and_tensor_bound() -> None:
    tensor = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    receipts = []
    for setting in range(3):
        shard = M03RV11TrainingShardPlan(
            setting_index=setting,
            fold_index=0,
            completed_update=0,
            episode_start=1,
            global_origins=(1, 2, 3, 4, 5, 6),
            rank_origins=((1, 3, 5), (2, 4, 6)),
            panel_episode_schedule_sha256="d" * 64,
            fold_geometry_sha256="e" * 64,
        )
        receipts.append(
            build_m03r_v11_paired_input_receipt(
                shard,
                (tensor,),
                source_array_sha256="a" * 64,
                asset_axis_sha256="b" * 64,
            )
        )
    assert len({receipt.receipt_sha256 for receipt in receipts}) == 1
    changed = build_m03r_v11_paired_input_receipt(
        M03RV11TrainingShardPlan(
            setting_index=0,
            fold_index=0,
            completed_update=0,
            episode_start=1,
            global_origins=(1, 2, 3, 4, 5, 6),
            rank_origins=((1, 3, 5), (2, 4, 6)),
            panel_episode_schedule_sha256="d" * 64,
            fold_geometry_sha256="e" * 64,
        ),
        (tensor + 1.0,),
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
    )
    assert changed.receipt_sha256 != receipts[0].receipt_sha256
