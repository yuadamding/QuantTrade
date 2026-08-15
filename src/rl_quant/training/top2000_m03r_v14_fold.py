"""Context-matched fold geometry and paired epoch schedule for M03R-v14."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v14_top2000_dev import (
    M03R_V14_EPISODE_SCHEDULE_RULE,
    M03R_V14_FOLD_GEOMETRY_RULE,
    M03R_V14_PREDICTIVE_SPEC,
    M03R_V14_PROTOCOL_SHA256,
    M03R_V14_SELECTED_HORIZON_SESSIONS,
    M03R_V14_SETTINGS,
)
M03R_V14_FOLD_SCHEMA = "rl-quant.top2000-dev.m03r-v14-fold-geometry-v1"
M03R_V14_PANEL_SCHEDULE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-panel-episode-schedule-v1"
)
M03R_V14_TRAINING_UPDATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-training-update-v1"
)
M03R_V14_PAIRED_INPUT_BINDING_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v14-paired-input-binding-v1"
)
M03R_V14_REQUIRED_STATE_ROWS = 1001
M03R_V14_FIRST_QUALIFICATION_START = 469
M03R_V14_FOLD_ADVANCE = 93
M03R_V14_MINIMUM_LOCAL_ORIGIN = (
    M03R_V14_PREDICTIVE_SPEC.observation_context_sessions - 1
)
M03R_V14_MAXIMUM_LOCAL_TRAINING_ORIGIN = (
    M03R_V14_PREDICTIVE_SPEC.episode_state_rows
    - M03R_V14_SELECTED_HORIZON_SESSIONS
    - 2
)


class M03RV14FoldError(ValueError):
    """The v14 full-context fold or paired update schedule drifted."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV14FoldError(f"{name} is not a lowercase SHA-256")
    return value


def _complementary_rank_shards(
    origins: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        len(origins) < M03R_V14_PREDICTIVE_SPEC.expected_world_size
        or any(later <= earlier for earlier, later in pairwise(origins))
    ):
        raise M03RV14FoldError("v14 origins cannot form complementary rank shards")
    first = origins[::2]
    second = origins[1::2]
    if not first or not second:
        raise M03RV14FoldError("v14 paired origin set is too small")
    if (
        set(first).intersection(second)
        or tuple(sorted((*first, *second))) != origins
    ):
        raise M03RV14FoldError("v14 rank shards do not cover the global origins")
    return first, second


@dataclass(frozen=True, slots=True)
class M03RV14FoldGeometry:
    fold_index: int
    cache_state_rows: int
    training_state_start: int
    training_target_stop_exclusive: int
    training_origin_start_inclusive: int
    training_origin_stop_exclusive: int
    purge_start_inclusive: int
    qualification_origin_start_inclusive: int
    qualification_origin_stop_exclusive: int
    qualification_target_stop_exclusive: int
    qualification_episode_state_start: int
    qualification_episode_state_stop_exclusive: int
    geometry_rule: str = M03R_V14_FOLD_GEOMETRY_RULE
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_FOLD_SCHEMA

    def validate(self) -> None:
        spec = M03R_V14_PREDICTIVE_SPEC
        expected_qualification_start = (
            M03R_V14_FIRST_QUALIFICATION_START
            + M03R_V14_FOLD_ADVANCE * self.fold_index
        )
        if (
            isinstance(self.fold_index, bool)
            or self.fold_index not in range(spec.chronological_fold_count)
            or self.cache_state_rows != M03R_V14_REQUIRED_STATE_ROWS
            or self.training_state_start != 0
            or self.qualification_origin_start_inclusive
            != expected_qualification_start
            or self.training_target_stop_exclusive
            != self.qualification_origin_start_inclusive - spec.purge_sessions
            or self.purge_start_inclusive != self.training_target_stop_exclusive
            or self.training_origin_start_inclusive != M03R_V14_MINIMUM_LOCAL_ORIGIN
            or self.training_origin_stop_exclusive
            != self.training_target_stop_exclusive
            - M03R_V14_SELECTED_HORIZON_SESSIONS
            - 1
            or self.training_origin_stop_exclusive
            <= self.training_origin_start_inclusive
            or self.qualification_origin_stop_exclusive
            - self.qualification_origin_start_inclusive
            != spec.qualification_origins_per_fold
            or self.qualification_target_stop_exclusive
            - self.qualification_origin_stop_exclusive
            != M03R_V14_SELECTED_HORIZON_SESSIONS + 1
            or self.qualification_episode_state_stop_exclusive
            != self.qualification_target_stop_exclusive
            or self.qualification_episode_state_start
            != self.qualification_episode_state_stop_exclusive
            - spec.episode_state_rows
            or self.qualification_origin_start_inclusive
            - self.qualification_episode_state_start
            < M03R_V14_MINIMUM_LOCAL_ORIGIN
            or self.qualification_origin_stop_exclusive
            - 1
            - self.qualification_episode_state_start
            > M03R_V14_MAXIMUM_LOCAL_TRAINING_ORIGIN
            or self.qualification_target_stop_exclusive > self.cache_state_rows
            or self.geometry_rule != M03R_V14_FOLD_GEOMETRY_RULE
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_FOLD_SCHEMA
        ):
            raise M03RV14FoldError("v14 fold geometry drifted")
        if self.fold_index == spec.chronological_fold_count - 1 and (
            self.qualification_target_stop_exclusive != self.cache_state_rows
        ):
            raise M03RV14FoldError("v14 final fold does not consume the cache boundary")

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
            / M03R_V14_PREDICTIVE_SPEC.origins_per_update
        )

    @property
    def optimizer_updates(self) -> int:
        return self.training_block_count * M03R_V14_PREDICTIVE_SPEC.training_epochs

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def render_m03r_v14_fold_geometries(
    state_rows: int,
) -> tuple[M03RV14FoldGeometry, ...]:
    if state_rows != M03R_V14_REQUIRED_STATE_ROWS:
        raise M03RV14FoldError("v14 requires the exact 1001-state pre-2026 cache")
    spec = M03R_V14_PREDICTIVE_SPEC
    rows: list[M03RV14FoldGeometry] = []
    for fold_index in range(spec.chronological_fold_count):
        qualification_start = (
            M03R_V14_FIRST_QUALIFICATION_START
            + M03R_V14_FOLD_ADVANCE * fold_index
        )
        qualification_stop = qualification_start + spec.qualification_origins_per_fold
        target_stop = (
            qualification_stop + M03R_V14_SELECTED_HORIZON_SESSIONS + 1
        )
        training_stop = qualification_start - spec.purge_sessions
        row = M03RV14FoldGeometry(
            fold_index=fold_index,
            cache_state_rows=state_rows,
            training_state_start=0,
            training_target_stop_exclusive=training_stop,
            training_origin_start_inclusive=M03R_V14_MINIMUM_LOCAL_ORIGIN,
            training_origin_stop_exclusive=(
                training_stop - M03R_V14_SELECTED_HORIZON_SESSIONS
                - 1
            ),
            purge_start_inclusive=training_stop,
            qualification_origin_start_inclusive=qualification_start,
            qualification_origin_stop_exclusive=qualification_stop,
            qualification_target_stop_exclusive=target_stop,
            qualification_episode_state_start=target_stop - spec.episode_state_rows,
            qualification_episode_state_stop_exclusive=target_stop,
        )
        row.validate()
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class M03RV14PanelEpisodeSchedule:
    protocol_common_data_sha256: str
    cache_sha256: str
    asset_axis_sha256: str
    fold_geometry_sha256: tuple[str, ...]
    seed: int = M03R_V14_PREDICTIVE_SPEC.seed
    training_epochs: int = M03R_V14_PREDICTIVE_SPEC.training_epochs
    origins_per_update: int = M03R_V14_PREDICTIVE_SPEC.origins_per_update
    schedule_rule: str = M03R_V14_EPISODE_SCHEDULE_RULE
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_PANEL_SCHEDULE_SCHEMA

    def validate(self) -> None:
        geometries = render_m03r_v14_fold_geometries(
            M03R_V14_REQUIRED_STATE_ROWS
        )
        if (
            self.fold_geometry_sha256
            != tuple(geometry.receipt_sha256 for geometry in geometries)
            or self.seed != M03R_V14_PREDICTIVE_SPEC.seed
            or self.training_epochs != M03R_V14_PREDICTIVE_SPEC.training_epochs
            or self.origins_per_update != M03R_V14_PREDICTIVE_SPEC.origins_per_update
            or self.schedule_rule != M03R_V14_EPISODE_SCHEDULE_RULE
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_PANEL_SCHEDULE_SCHEMA
        ):
            raise M03RV14FoldError("v14 panel episode schedule drifted")
        for name, value in (
            ("protocol_common_data_sha256", self.protocol_common_data_sha256),
            ("cache_sha256", self.cache_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
        ):
            _digest(name, value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def _epoch_block_order(
    schedule: M03RV14PanelEpisodeSchedule,
    geometry: M03RV14FoldGeometry,
    *,
    epoch_index: int,
) -> tuple[int, ...]:
    if epoch_index not in range(M03R_V14_PREDICTIVE_SPEC.training_epochs):
        raise M03RV14FoldError("v14 epoch cursor drifted")
    return tuple(
        sorted(
            range(geometry.training_block_count),
            key=lambda block_index: hashlib.sha256(
                (
                    f"{schedule.receipt_sha256}:{geometry.receipt_sha256}:"
                    f"{epoch_index}:{block_index}:paired-v14"
                ).encode("ascii")
            ).digest(),
        )
    )


@dataclass(frozen=True, slots=True)
class M03RV14TrainingUpdatePlan:
    setting_index: int
    fold_index: int
    completed_update: int
    epoch_index: int
    epoch_block_index: int
    episode_start: int
    episode_stop_exclusive: int
    global_origins: tuple[int, ...]
    rank_origins: tuple[tuple[int, ...], tuple[int, ...]]
    panel_episode_schedule_sha256: str
    fold_geometry_sha256: str
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_TRAINING_UPDATE_SCHEMA

    def validate(self) -> None:
        geometries = render_m03r_v14_fold_geometries(
            M03R_V14_REQUIRED_STATE_ROWS
        )
        geometry = geometries[self.fold_index] if self.fold_index in range(6) else None
        if (
            self.setting_index not in range(len(M03R_V14_SETTINGS))
            or geometry is None
            or self.completed_update not in range(geometry.optimizer_updates)
            or self.epoch_index
            != self.completed_update // geometry.training_block_count
            or self.epoch_block_index not in range(geometry.training_block_count)
            or self.episode_stop_exclusive - self.episode_start
            != M03R_V14_PREDICTIVE_SPEC.episode_state_rows
            or self.episode_start < 0
            or self.episode_stop_exclusive > geometry.training_target_stop_exclusive
            or not self.global_origins
            or min(self.global_origins) - self.episode_start
            < M03R_V14_MINIMUM_LOCAL_ORIGIN
            or max(self.global_origins) - self.episode_start
            > M03R_V14_MAXIMUM_LOCAL_TRAINING_ORIGIN
            or any(
                origin + M03R_V14_SELECTED_HORIZON_SESSIONS + 1
                > geometry.training_target_stop_exclusive
                for origin in self.global_origins
            )
            or self.rank_origins != _complementary_rank_shards(self.global_origins)
            or self.fold_geometry_sha256 != geometry.receipt_sha256
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_TRAINING_UPDATE_SCHEMA
        ):
            raise M03RV14FoldError("v14 training update plan drifted")
        _digest("panel_episode_schedule_sha256", self.panel_episode_schedule_sha256)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV14PairedInputBinding:
    fold_index: int
    completed_update: int
    epoch_index: int
    epoch_block_index: int
    episode_start: int
    episode_stop_exclusive: int
    global_origins: tuple[int, ...]
    rank_origins: tuple[tuple[int, ...], tuple[int, ...]]
    panel_episode_schedule_sha256: str
    fold_geometry_sha256: str
    cache_sha256: str
    source_array_sha256: str
    asset_axis_sha256: str
    protocol_sha256: str = M03R_V14_PROTOCOL_SHA256
    schema: str = M03R_V14_PAIRED_INPUT_BINDING_SCHEMA

    def validate(self) -> None:
        if (
            self.fold_index not in range(6)
            or self.completed_update < 0
            or self.epoch_index < 0
            or self.epoch_block_index < 0
            or self.episode_start < 0
            or self.episode_stop_exclusive - self.episode_start
            != M03R_V14_PREDICTIVE_SPEC.episode_state_rows
            or not self.global_origins
            or self.rank_origins != _complementary_rank_shards(self.global_origins)
            or self.protocol_sha256 != M03R_V14_PROTOCOL_SHA256
            or self.schema != M03R_V14_PAIRED_INPUT_BINDING_SCHEMA
        ):
            raise M03RV14FoldError("v14 paired input binding drifted")
        for name, value in (
            ("panel_episode_schedule_sha256", self.panel_episode_schedule_sha256),
            ("fold_geometry_sha256", self.fold_geometry_sha256),
            ("cache_sha256", self.cache_sha256),
            ("source_array_sha256", self.source_array_sha256),
            ("asset_axis_sha256", self.asset_axis_sha256),
        ):
            _digest(name, value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def build_m03r_v14_paired_input_binding(
    update_plan: M03RV14TrainingUpdatePlan,
    *,
    cache_sha256: str,
    source_array_sha256: str,
    asset_axis_sha256: str,
) -> M03RV14PairedInputBinding:
    """Bind common immutable episode inputs without rehashing tensors per setting."""

    update_plan.validate()
    result = M03RV14PairedInputBinding(
        fold_index=update_plan.fold_index,
        completed_update=update_plan.completed_update,
        epoch_index=update_plan.epoch_index,
        epoch_block_index=update_plan.epoch_block_index,
        episode_start=update_plan.episode_start,
        episode_stop_exclusive=update_plan.episode_stop_exclusive,
        global_origins=update_plan.global_origins,
        rank_origins=update_plan.rank_origins,
        panel_episode_schedule_sha256=update_plan.panel_episode_schedule_sha256,
        fold_geometry_sha256=update_plan.fold_geometry_sha256,
        cache_sha256=cache_sha256,
        source_array_sha256=source_array_sha256,
        asset_axis_sha256=asset_axis_sha256,
    )
    result.validate()
    return result


def render_m03r_v14_training_update_plan(
    schedule: M03RV14PanelEpisodeSchedule,
    geometry: M03RV14FoldGeometry,
    *,
    setting_index: int,
    completed_update: int,
) -> M03RV14TrainingUpdatePlan:
    schedule.validate()
    geometry.validate()
    if (
        setting_index not in range(len(M03R_V14_SETTINGS))
        or completed_update not in range(geometry.optimizer_updates)
        or schedule.fold_geometry_sha256[geometry.fold_index]
        != geometry.receipt_sha256
    ):
        raise M03RV14FoldError("v14 update request drifted")
    epoch_index, block_slot = divmod(
        completed_update, geometry.training_block_count
    )
    epoch_block_index = _epoch_block_order(
        schedule, geometry, epoch_index=epoch_index
    )[block_slot]
    eligible = geometry.eligible_training_origins
    first = epoch_block_index * schedule.origins_per_update
    origins = eligible[first : first + schedule.origins_per_update]
    if not origins:
        raise M03RV14FoldError("v14 update selected an empty origin block")
    episode_start = min(
        origins[0] - M03R_V14_MINIMUM_LOCAL_ORIGIN,
        geometry.training_target_stop_exclusive
        - M03R_V14_PREDICTIVE_SPEC.episode_state_rows,
    )
    result = M03RV14TrainingUpdatePlan(
        setting_index=setting_index,
        fold_index=geometry.fold_index,
        completed_update=completed_update,
        epoch_index=epoch_index,
        epoch_block_index=epoch_block_index,
        episode_start=episode_start,
        episode_stop_exclusive=(
            episode_start + M03R_V14_PREDICTIVE_SPEC.episode_state_rows
        ),
        global_origins=origins,
        rank_origins=_complementary_rank_shards(origins),
        panel_episode_schedule_sha256=schedule.receipt_sha256,
        fold_geometry_sha256=geometry.receipt_sha256,
    )
    result.validate()
    return result


__all__ = [
    "M03R_V14_FIRST_QUALIFICATION_START",
    "M03R_V14_FOLD_ADVANCE",
    "M03R_V14_FOLD_SCHEMA",
    "M03R_V14_MAXIMUM_LOCAL_TRAINING_ORIGIN",
    "M03R_V14_MINIMUM_LOCAL_ORIGIN",
    "M03R_V14_PANEL_SCHEDULE_SCHEMA",
    "M03R_V14_PAIRED_INPUT_BINDING_SCHEMA",
    "M03R_V14_TRAINING_UPDATE_SCHEMA",
    "M03RV14FoldError",
    "M03RV14FoldGeometry",
    "M03RV14PanelEpisodeSchedule",
    "M03RV14PairedInputBinding",
    "M03RV14TrainingUpdatePlan",
    "build_m03r_v14_paired_input_binding",
    "render_m03r_v14_fold_geometries",
    "render_m03r_v14_training_update_plan",
]
