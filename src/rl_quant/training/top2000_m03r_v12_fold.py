"""Paired fold-origin planning for M03R-v12 predictive training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    Top2000M03RV7DevelopmentFold,
)
from rl_quant.training.top2000_m03r_v9_fold import (
    M03R_V9_MAX_TARGET_HORIZON,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03R_V9_MINIMUM_HISTORY_SESSIONS,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v12_predictive_worker import (
    M03RV12PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v12_schedule import (
    M03RV12PanelEpisodeSchedule,
    deterministic_m03r_v12_episode_start,
    m03r_v12_complementary_rank_shards,
)

M03R_V12_TRAINING_SHARD_SCHEMA = "rl-quant.top2000-dev.m03r-v12-training-shard-v1"
M03R_V12_MINIMUM_RESIDUAL_ORIGIN_STATE_INDEX = M03R_V9_MINIMUM_HISTORY_SESSIONS
M03R_V12_MAX_LOCAL_ORIGIN = (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS - M03R_V9_MAX_TARGET_HORIZON - 2
)


class M03RV12FoldError(ValueError):
    """The v12 paired fold schedule or rank shard drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV12TrainingShardPlan:
    setting_index: int
    fold_index: int
    completed_update: int
    episode_start: int
    global_origins: tuple[int, ...]
    rank_origins: tuple[tuple[int, ...], tuple[int, ...]]
    panel_episode_schedule_sha256: str
    fold_geometry_sha256: str
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    schema: str = M03R_V12_TRAINING_SHARD_SCHEMA

    def validate(self) -> None:
        expected_shards = m03r_v12_complementary_rank_shards(self.global_origins)
        if (
            self.setting_index not in range(3)
            or self.fold_index not in range(6)
            or self.completed_update not in range(64)
            or self.episode_start < 0
            or self.rank_origins != expected_shards
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.schema != M03R_V12_TRAINING_SHARD_SCHEMA
        ):
            raise M03RV12FoldError("v12 training shard plan drifted")
        for value in (
            self.panel_episode_schedule_sha256,
            self.fold_geometry_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise M03RV12FoldError("v12 shard identity is not a SHA-256")
        if any(
            origin - self.episode_start > M03R_V12_MAX_LOCAL_ORIGIN
            for origin in self.global_origins
        ):
            raise M03RV12FoldError(
                "v12 shard origin lacks a complete maximum-horizon return path"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def render_m03r_v12_training_shard_plan(
    worker: M03RV12PredictiveWorkerPlan,
    schedule: M03RV12PanelEpisodeSchedule,
    fold: Top2000M03RV7DevelopmentFold,
    *,
    completed_update: int,
) -> M03RV12TrainingShardPlan:
    worker.validate()
    schedule.validate()
    imported_geometry = render_m03r_v10_fold_geometry(fold)
    if (
        worker.panel_episode_schedule_sha256 != schedule.receipt_sha256
        or worker.cache_sha256 != schedule.cache_sha256
        or schedule.fold_geometry_sha256[fold.fold_index]
        != imported_geometry.receipt_sha256
    ):
        raise M03RV12FoldError("v12 worker and panel schedule are not aligned")
    maximum_start = (
        imported_geometry.qualification_target_stop_exclusive
        - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
    )
    start = deterministic_m03r_v12_episode_start(
        schedule,
        fold_index=fold.fold_index,
        completed_updates=completed_update,
        admissible_start_count=maximum_start + 1,
    )
    first_origin = max(start, M03R_V12_MINIMUM_RESIDUAL_ORIGIN_STATE_INDEX)
    last_origin = min(
        imported_geometry.optimizer_target_stop_exclusive
        - M03R_V9_MAX_TARGET_HORIZON
        - 1,
        start + M03R_V12_MAX_LOCAL_ORIGIN,
    )
    origins = tuple(range(first_origin, last_origin + 1))
    shards = m03r_v12_complementary_rank_shards(origins)
    result = M03RV12TrainingShardPlan(
        setting_index=worker.setting_index,
        fold_index=fold.fold_index,
        completed_update=completed_update,
        episode_start=start,
        global_origins=origins[: len(origins) - len(origins) % 2],
        rank_origins=(shards[0], shards[1]),
        panel_episode_schedule_sha256=schedule.receipt_sha256,
        fold_geometry_sha256=imported_geometry.receipt_sha256,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V12_MAX_LOCAL_ORIGIN",
    "M03R_V12_MINIMUM_RESIDUAL_ORIGIN_STATE_INDEX",
    "M03R_V12_TRAINING_SHARD_SCHEMA",
    "M03RV12FoldError",
    "M03RV12TrainingShardPlan",
    "render_m03r_v12_training_shard_plan",
]
