"""Origin-aligned folds and paired every-origin schedules for M03R-v16."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_EPISODE_SCHEDULE_RULE,
    M03R_V16_FOLD_GEOMETRY_RULE,
    M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)

M03R_V16_FOLD_SCHEMA = "rl-quant.top2000-dev.m03r-v16-fold-geometry-v1"
M03R_V16_PANEL_SCHEDULE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-panel-episode-schedule-v1"
)
M03R_V16_TRAINING_UPDATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-update-v1"
)
M03R_V16_REQUIRED_STATE_ROWS = 1001
M03R_V16_FOLD_ADVANCE = 80
M03R_V16_FINAL_QUALIFICATION_START = 907
M03R_V16_FIRST_QUALIFICATION_START = (
    M03R_V16_FINAL_QUALIFICATION_START
    - M03R_V16_FOLD_ADVANCE * (M03R_V16_PREDICTIVE_SPEC.chronological_fold_count - 1)
)
M03R_V16_MINIMUM_LOCAL_ORIGIN = (
    M03R_V16_PREDICTIVE_SPEC.observation_context_sessions - 1
)
M03R_V16_MAXIMUM_LOCAL_ORIGIN = (
    M03R_V16_PREDICTIVE_SPEC.episode_state_rows
    - M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS
    - 2
)


class M03RV16FoldError(ValueError):
    """The V16 fold or paired schedule drifted."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _rank_shards(origins: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        len(origins) < 2
        or any(right <= left for left, right in pairwise(origins))
    ):
        raise M03RV16FoldError("V16 origins cannot form paired rank shards")
    shards = (origins[::2], origins[1::2])
    if (
        not shards[0]
        or not shards[1]
        or set(shards[0]).intersection(shards[1])
        or tuple(sorted((*shards[0], *shards[1]))) != origins
    ):
        raise M03RV16FoldError("V16 rank shards do not cover the origin block")
    return shards


@dataclass(frozen=True, slots=True)
class M03RV16FoldGeometry:
    fold_index: int
    cache_state_rows: int
    training_origin_start_inclusive: int
    training_origin_stop_exclusive: int
    inner_validation_origin_start_inclusive: int
    inner_validation_origin_stop_exclusive: int
    training_target_stop_exclusive: int
    qualification_origin_start_inclusive: int
    qualification_origin_stop_exclusive: int
    qualification_target_stop_exclusive: int
    geometry_rule: str = M03R_V16_FOLD_GEOMETRY_RULE
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_FOLD_SCHEMA

    def validate(self) -> None:
        spec = M03R_V16_PREDICTIVE_SPEC
        expected_qualification_start = (
            M03R_V16_FIRST_QUALIFICATION_START
            + self.fold_index * M03R_V16_FOLD_ADVANCE
        )
        if (
            self.fold_index not in range(spec.chronological_fold_count)
            or self.cache_state_rows != M03R_V16_REQUIRED_STATE_ROWS
            or self.training_origin_start_inclusive != M03R_V16_MINIMUM_LOCAL_ORIGIN
            or self.qualification_origin_start_inclusive
            != expected_qualification_start
            or self.qualification_origin_stop_exclusive
            - self.qualification_origin_start_inclusive
            != spec.qualification_origins_per_fold
            or self.qualification_target_stop_exclusive
            != self.qualification_origin_stop_exclusive
            + M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS
            + 1
            or self.training_target_stop_exclusive
            != self.qualification_origin_start_inclusive
            - spec.qualification_purge_sessions
            or self.inner_validation_origin_stop_exclusive
            != self.training_target_stop_exclusive
            - M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS
            - 1
            or self.inner_validation_origin_stop_exclusive
            - self.inner_validation_origin_start_inclusive
            != spec.inner_validation_origins_per_fold
            or self.training_origin_stop_exclusive
            != self.inner_validation_origin_start_inclusive
            - spec.optimizer_to_validation_embargo_sessions
            or self.training_origin_stop_exclusive
            <= self.training_origin_start_inclusive
            or self.qualification_target_stop_exclusive > self.cache_state_rows
            or self.geometry_rule != M03R_V16_FOLD_GEOMETRY_RULE
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_FOLD_SCHEMA
        ):
            raise M03RV16FoldError("V16 fold geometry drifted")
        if (
            self.fold_index == spec.chronological_fold_count - 1
            and self.qualification_target_stop_exclusive != self.cache_state_rows
        ):
            raise M03RV16FoldError("V16 final fold does not consume the cache boundary")

    @property
    def eligible_training_origins(self) -> tuple[int, ...]:
        self.validate()
        return tuple(
            range(
                self.training_origin_start_inclusive,
                self.training_origin_stop_exclusive,
            )
        )

    @property
    def training_block_count(self) -> int:
        return math.ceil(
            len(self.eligible_training_origins)
            / M03R_V16_PREDICTIVE_SPEC.origins_per_update
        )

    @property
    def maximum_optimizer_updates(self) -> int:
        return (
            self.training_block_count
            * M03R_V16_PREDICTIVE_SPEC.maximum_score_training_epochs
        )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def render_m03r_v16_fold_geometries(
    state_rows: int,
) -> tuple[M03RV16FoldGeometry, ...]:
    if state_rows != M03R_V16_REQUIRED_STATE_ROWS:
        raise M03RV16FoldError("V16 requires the exact 1001-state pre-2026 cache")
    spec = M03R_V16_PREDICTIVE_SPEC
    rows: list[M03RV16FoldGeometry] = []
    for fold_index in range(spec.chronological_fold_count):
        qualification_start = (
            M03R_V16_FIRST_QUALIFICATION_START + fold_index * M03R_V16_FOLD_ADVANCE
        )
        qualification_stop = qualification_start + spec.qualification_origins_per_fold
        qualification_target_stop = (
            qualification_stop + M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS + 1
        )
        training_target_stop = qualification_start - spec.qualification_purge_sessions
        validation_stop = (
            training_target_stop - M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS - 1
        )
        validation_start = validation_stop - spec.inner_validation_origins_per_fold
        row = M03RV16FoldGeometry(
            fold_index=fold_index,
            cache_state_rows=state_rows,
            training_origin_start_inclusive=M03R_V16_MINIMUM_LOCAL_ORIGIN,
            training_origin_stop_exclusive=(
                validation_start - spec.optimizer_to_validation_embargo_sessions
            ),
            inner_validation_origin_start_inclusive=validation_start,
            inner_validation_origin_stop_exclusive=validation_stop,
            training_target_stop_exclusive=training_target_stop,
            qualification_origin_start_inclusive=qualification_start,
            qualification_origin_stop_exclusive=qualification_stop,
            qualification_target_stop_exclusive=qualification_target_stop,
        )
        row.validate()
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class M03RV16PanelSchedule:
    protocol_common_data_sha256: str
    cache_sha256: str
    asset_axis_sha256: str
    fold_geometry_sha256: tuple[str, ...]
    seed: int = M03R_V16_PREDICTIVE_SPEC.seed
    schedule_rule: str = M03R_V16_EPISODE_SCHEDULE_RULE
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_PANEL_SCHEDULE_SCHEMA

    def validate(self) -> None:
        geometries = render_m03r_v16_fold_geometries(M03R_V16_REQUIRED_STATE_ROWS)
        if (
            self.fold_geometry_sha256
            != tuple(row.receipt_sha256 for row in geometries)
            or self.seed != M03R_V16_PREDICTIVE_SPEC.seed
            or self.schedule_rule != M03R_V16_EPISODE_SCHEDULE_RULE
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_PANEL_SCHEDULE_SCHEMA
            or not all(
                _digest(value)
                for value in (
                    self.protocol_common_data_sha256,
                    self.cache_sha256,
                    self.asset_axis_sha256,
                )
            )
        ):
            raise M03RV16FoldError("V16 panel schedule drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV16TrainingUpdatePlan:
    setting_index: int
    fold_index: int
    completed_update: int
    epoch_index: int
    epoch_block_index: int
    episode_start: int
    episode_stop_exclusive: int
    global_origins: tuple[int, ...]
    rank_origins: tuple[tuple[int, ...], tuple[int, ...]]
    panel_schedule_sha256: str
    fold_geometry_sha256: str
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_TRAINING_UPDATE_SCHEMA

    def validate(self) -> None:
        geometries = render_m03r_v16_fold_geometries(M03R_V16_REQUIRED_STATE_ROWS)
        geometry = geometries[self.fold_index] if self.fold_index in range(6) else None
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or geometry is None
            or self.completed_update not in range(geometry.maximum_optimizer_updates)
            or self.epoch_index
            != self.completed_update // geometry.training_block_count
            or self.epoch_block_index not in range(geometry.training_block_count)
            or self.episode_stop_exclusive - self.episode_start
            != M03R_V16_PREDICTIVE_SPEC.episode_state_rows
            or not self.global_origins
            or min(self.global_origins) - self.episode_start
            < M03R_V16_MINIMUM_LOCAL_ORIGIN
            or max(self.global_origins) - self.episode_start
            > M03R_V16_MAXIMUM_LOCAL_ORIGIN
            or any(
                origin + M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS + 1
                > geometry.inner_validation_origin_start_inclusive
                for origin in self.global_origins
            )
            or self.rank_origins != _rank_shards(self.global_origins)
            or self.fold_geometry_sha256 != geometry.receipt_sha256
            or not _digest(self.panel_schedule_sha256)
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_TRAINING_UPDATE_SCHEMA
        ):
            raise M03RV16FoldError("V16 training update plan drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _epoch_block_order(
    schedule: M03RV16PanelSchedule,
    geometry: M03RV16FoldGeometry,
    epoch_index: int,
) -> tuple[int, ...]:
    if epoch_index not in range(M03R_V16_PREDICTIVE_SPEC.maximum_score_training_epochs):
        raise M03RV16FoldError("V16 epoch cursor drifted")
    return tuple(
        sorted(
            range(geometry.training_block_count),
            key=lambda block: hashlib.sha256(
                (
                    f"{schedule.receipt_sha256}:{geometry.receipt_sha256}:"
                    f"{epoch_index}:{block}:paired-v16"
                ).encode("ascii")
            ).digest(),
        )
    )


def render_m03r_v16_training_update_plan(
    schedule: M03RV16PanelSchedule,
    geometry: M03RV16FoldGeometry,
    *,
    setting_index: int,
    completed_update: int,
) -> M03RV16TrainingUpdatePlan:
    schedule.validate()
    geometry.validate()
    if (
        setting_index not in range(len(M03R_V16_SETTINGS))
        or completed_update not in range(geometry.maximum_optimizer_updates)
        or schedule.fold_geometry_sha256[geometry.fold_index]
        != geometry.receipt_sha256
    ):
        raise M03RV16FoldError("V16 update request drifted")
    epoch_index, block_slot = divmod(completed_update, geometry.training_block_count)
    block = _epoch_block_order(schedule, geometry, epoch_index)[block_slot]
    first = block * M03R_V16_PREDICTIVE_SPEC.origins_per_update
    origins = geometry.eligible_training_origins[
        first : first + M03R_V16_PREDICTIVE_SPEC.origins_per_update
    ]
    if not origins:
        raise M03RV16FoldError("V16 selected an empty training block")
    lower_start = origins[-1] - M03R_V16_MAXIMUM_LOCAL_ORIGIN
    preferred_start = origins[0] - M03R_V16_MINIMUM_LOCAL_ORIGIN
    boundary_start = (
        geometry.inner_validation_origin_start_inclusive
        - M03R_V16_PREDICTIVE_SPEC.episode_state_rows
    )
    episode_start = max(0, lower_start, min(preferred_start, boundary_start))
    result = M03RV16TrainingUpdatePlan(
        setting_index=setting_index,
        fold_index=geometry.fold_index,
        completed_update=completed_update,
        epoch_index=epoch_index,
        epoch_block_index=block,
        episode_start=episode_start,
        episode_stop_exclusive=(
            episode_start + M03R_V16_PREDICTIVE_SPEC.episode_state_rows
        ),
        global_origins=origins,
        rank_origins=_rank_shards(origins),
        panel_schedule_sha256=schedule.receipt_sha256,
        fold_geometry_sha256=geometry.receipt_sha256,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V16_FINAL_QUALIFICATION_START",
    "M03R_V16_FIRST_QUALIFICATION_START",
    "M03R_V16_FOLD_ADVANCE",
    "M03R_V16_FOLD_SCHEMA",
    "M03R_V16_MAXIMUM_LOCAL_ORIGIN",
    "M03R_V16_MINIMUM_LOCAL_ORIGIN",
    "M03R_V16_PANEL_SCHEDULE_SCHEMA",
    "M03R_V16_REQUIRED_STATE_ROWS",
    "M03R_V16_TRAINING_UPDATE_SCHEMA",
    "M03RV16FoldError",
    "M03RV16FoldGeometry",
    "M03RV16PanelSchedule",
    "M03RV16TrainingUpdatePlan",
    "render_m03r_v16_fold_geometries",
    "render_m03r_v16_training_update_plan",
]
