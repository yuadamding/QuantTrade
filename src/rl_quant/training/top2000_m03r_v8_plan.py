"""Frozen worker geometry for TOP2000 M03R-v8 alpha discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
    M03R_V8_TOP2000_DEV_DESIGN_ID,
    M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_V8_TOP2000_DEV_SETTINGS,
    resolve_m03r_v8_top2000_dev_setting,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_FOLD_COUNT,
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    Top2000M03RV7DevelopmentFold,
    render_top2000_m03r_v7_development_folds,
)

M03R_V8_TRAINING_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v8-training-plan-v1"
M03R_V8_FOLD_GEOMETRY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-alpha-fold-geometry-v1"
)
M03R_V8_INNER_VALIDATION_ORIGINS = 63
M03R_V8_MAX_TARGET_HORIZON = max(
    M03R_V8_ALPHA_PRETRAINING.horizons_trading_sessions
)
M03R_V8_TARGET_SUPPORT_STATES = M03R_V8_MAX_TARGET_HORIZON + 1


class M03RV8TrainingPlanError(ValueError):
    """The v8 worker plan or chronological split geometry drifted."""


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


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise M03RV8TrainingPlanError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class M03RV8FoldPretrainingGeometry:
    """One training-only optimizer split plus tail inner-validation split."""

    fold_index: int
    training_state_start: int
    optimizer_target_stop_exclusive: int
    inner_validation_start_inclusive: int
    inner_validation_origin_stop_exclusive: int
    inner_validation_target_stop_exclusive: int
    validation_episode_state_start: int
    validation_episode_state_stop_exclusive: int
    schema: str = M03R_V8_FOLD_GEOMETRY_SCHEMA

    def validate(self) -> None:
        fold = render_top2000_m03r_v7_development_folds(
            TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        )[self.fold_index]
        if (
            self.schema != M03R_V8_FOLD_GEOMETRY_SCHEMA
            or self.training_state_start != fold.training_state_start
            or self.optimizer_target_stop_exclusive
            != fold.training_state_stop_exclusive
            - M03R_V8_INNER_VALIDATION_ORIGINS
            - M03R_V8_TARGET_SUPPORT_STATES
            or self.inner_validation_start_inclusive
            != self.optimizer_target_stop_exclusive
            or self.inner_validation_origin_stop_exclusive
            != fold.training_state_stop_exclusive - M03R_V8_MAX_TARGET_HORIZON
            - 1
            or self.inner_validation_target_stop_exclusive
            != fold.training_state_stop_exclusive
            or self.inner_validation_origin_stop_exclusive
            - self.inner_validation_start_inclusive
            != M03R_V8_INNER_VALIDATION_ORIGINS
            or self.validation_episode_state_start
            != fold.training_state_stop_exclusive
            - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
            or self.validation_episode_state_stop_exclusive
            != fold.training_state_stop_exclusive
            or self.validation_episode_state_stop_exclusive
            - self.validation_episode_state_start
            != TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
        ):
            raise M03RV8TrainingPlanError("v8 pretraining fold geometry drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def render_m03r_v8_fold_pretraining_geometry(
    fold: Top2000M03RV7DevelopmentFold,
) -> M03RV8FoldPretrainingGeometry:
    """Carve inner validation only from the tail of the training fold."""

    if not isinstance(fold, Top2000M03RV7DevelopmentFold):
        raise M03RV8TrainingPlanError("v8 geometry requires a reviewed v7 fold")
    result = M03RV8FoldPretrainingGeometry(
        fold_index=fold.fold_index,
        training_state_start=fold.training_state_start,
        optimizer_target_stop_exclusive=(
            fold.training_state_stop_exclusive
            - M03R_V8_INNER_VALIDATION_ORIGINS
            - M03R_V8_TARGET_SUPPORT_STATES
        ),
        inner_validation_start_inclusive=(
            fold.training_state_stop_exclusive
            - M03R_V8_INNER_VALIDATION_ORIGINS
            - M03R_V8_TARGET_SUPPORT_STATES
        ),
        inner_validation_origin_stop_exclusive=(
            fold.training_state_stop_exclusive - M03R_V8_TARGET_SUPPORT_STATES
        ),
        inner_validation_target_stop_exclusive=fold.training_state_stop_exclusive,
        validation_episode_state_start=(
            fold.training_state_stop_exclusive
            - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
        ),
        validation_episode_state_stop_exclusive=fold.training_state_stop_exclusive,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class M03RV8DevelopmentTrainingPlan:
    """Immutable one-setting worker plan; all eight rows share seed/folds."""

    setting_index: int
    setting_id: str
    cache_path: str
    cache_sha256: str
    output_root: str
    source_manifest_sha256: str
    seed: int = 17
    fold_count: int = TOP2000_M03R_V7_DEV_FOLD_COUNT
    expected_world_size: int = 2
    token_dim: int = 128
    raw_stock_chunk: int = 512
    activation_checkpointing: bool = True
    mixed_precision: str = "bfloat16"
    alpha_pretraining_updates: int = M03R_V8_ALPHA_PRETRAINING.maximum_optimizer_updates
    economic_optimizer_updates: int = 64
    protocol_generation: str = M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION
    design_id: str = M03R_V8_TOP2000_DEV_DESIGN_ID
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    development_only: bool = True
    promotion_eligible: bool = False
    schema: str = M03R_V8_TRAINING_PLAN_SCHEMA

    def validate(self) -> None:
        setting = resolve_m03r_v8_top2000_dev_setting(self.setting_index)
        _digest("cache_sha256", self.cache_sha256)
        _digest("source_manifest_sha256", self.source_manifest_sha256)
        if (
            self.setting_id != setting.setting_id
            or self.setting_id != M03R_V8_TOP2000_DEV_SETTINGS[self.setting_index].setting_id
            or not self.cache_path
            or not self.output_root
            or self.seed != 17
            or self.fold_count != 6
            or self.expected_world_size != 2
            or self.token_dim != 128
            or self.raw_stock_chunk != 512
            or not self.activation_checkpointing
            or self.mixed_precision != "bfloat16"
            or self.alpha_pretraining_updates != 64
            or self.economic_optimizer_updates != 64
            or self.protocol_generation != M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION
            or self.design_id != M03R_V8_TOP2000_DEV_DESIGN_ID
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or not self.development_only
            or self.promotion_eligible
            or self.schema != M03R_V8_TRAINING_PLAN_SCHEMA
        ):
            raise M03RV8TrainingPlanError("v8 training plan drifted")

    @property
    def alpha_pretraining_required(self) -> bool:
        self.validate()
        return (
            resolve_m03r_v8_top2000_dev_setting(self.setting_index).alpha_pretraining_mode
            == "training-fold-pretrained"
        )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))

    @property
    def episode_schedule_sha256(self) -> str:
        return _sha256(
            {
                "schema": "rl-quant.top2000-dev.m03r-v8-alpha-episode-schedule-v1",
                "protocol_sha256": self.protocol_sha256,
                "cache_sha256": self.cache_sha256,
                "seed": self.seed,
                "episode_state_rows": TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
                "inner_validation_origins": M03R_V8_INNER_VALIDATION_ORIGINS,
                "maximum_target_horizon": M03R_V8_MAX_TARGET_HORIZON,
                "target_support_states": M03R_V8_TARGET_SUPPORT_STATES,
                "fold_geometry_sha256": [
                    render_m03r_v8_fold_pretraining_geometry(fold).receipt_sha256
                    for fold in render_top2000_m03r_v7_development_folds(
                        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
                    )
                ],
            }
        )


def deterministic_m03r_v8_pretraining_episode_start(
    plan: M03RV8DevelopmentTrainingPlan,
    fold: Top2000M03RV7DevelopmentFold,
    *,
    completed_updates: int,
) -> int:
    """Select one paired causal 378-state window without RNG state."""

    plan.validate()
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or not 0 <= completed_updates < plan.alpha_pretraining_updates
    ):
        raise M03RV8TrainingPlanError("pretraining schedule cursor is invalid")
    geometry = render_m03r_v8_fold_pretraining_geometry(fold)
    maximum_start = (
        geometry.inner_validation_target_stop_exclusive
        - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
    )
    if maximum_start < 0:
        raise M03RV8TrainingPlanError("fold cannot supply one training episode")
    digest = hashlib.sha256(
        f"{plan.episode_schedule_sha256}:{fold.fold_index}:{completed_updates}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (maximum_start + 1)


__all__ = [
    "M03R_V8_FOLD_GEOMETRY_SCHEMA",
    "M03R_V8_INNER_VALIDATION_ORIGINS",
    "M03R_V8_MAX_TARGET_HORIZON",
    "M03R_V8_TARGET_SUPPORT_STATES",
    "M03R_V8_TRAINING_PLAN_SCHEMA",
    "M03RV8DevelopmentTrainingPlan",
    "M03RV8FoldPretrainingGeometry",
    "M03RV8TrainingPlanError",
    "deterministic_m03r_v8_pretraining_episode_start",
    "render_m03r_v8_fold_pretraining_geometry",
]
