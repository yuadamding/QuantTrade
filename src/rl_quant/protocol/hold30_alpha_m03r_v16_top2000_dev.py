"""Holding-aligned selection research after the completed V15 h3 screen.

V16 is predictive-only.  It compares three long-horizon selection targets
under one paired daily representation while retaining a separately supervised
three-session timing output.  It cannot reuse V15 model/optimizer state,
authorize an economic/RL optimizer, or access 2026 outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_PROTOCOL_SHA256,
)

M03R_V16_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v16-holding-aligned-selection-v1"
)
M03R_V16_DESIGN_ID = (
    "daily-ohlcv-paired-long-horizon-selection-with-h3-timing-v1"
)
M03R_V16_TIMING_HORIZON_SESSIONS = 3
M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS = 30
M03R_V16_SELECTION_TARGET_IDS = (
    "h21-cumulative-factor-residual",
    "h30-cumulative-factor-residual",
    "survival-weighted-1-30-mean-factor-residual",
)
M03R_V16_SETTING_IDS = (
    "V16-P0-h21-selection-h3-timing",
    "V16-P1-h30-selection-h3-timing",
    "V16-P2-survival30-selection-h3-timing",
)
M03R_V16_SCORE_COMPONENT_WEIGHTS = (0.85, 0.15)
M03R_V16_SURVIVAL_DAILY_HAZARD = 1.0 / 30.0
M03R_V16_EPISODE_SCHEDULE_RULE = (
    "origin-aligned-252-context-63-origin-block-paired-v1"
)
M03R_V16_FOLD_GEOMETRY_RULE = (
    "six-expanding-inner32-selected-63-origin-max30-support-pre2026-v1"
)
M03R_V16_EXECUTABLE_SCORE_RULE = (
    "v15-operator-action-projected-selection-and-timing-score-v1"
)
M03R_V16_LABEL_SUPPORT_RULE = (
    "origin-action-eligible-intersect-target-specific-future-support-v1"
)
M03R_V16_FILL_RULE = (
    "observe-close-t-fill-next-close-t-plus-1-earn-post-fill-transition-v1"
)
M03R_V16_OPTIMIZER_RULE = (
    "score-encoder-and-two-means-then-frozen-mean-two-scale-calibration-v1"
)
M03R_V16_CHECKPOINT_SELECTION_RULE = (
    "inner-validation-max-selection-ic-then-spread-then-min-robust-v1"
)


class M03RV16ProtocolError(ValueError):
    """The immutable V16 predictive contract drifted."""


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


def m03r_v16_survival_weights() -> tuple[float, ...]:
    """Return the frozen normalized geometric survival weights for days 1..30."""

    raw = tuple(
        (1.0 - M03R_V16_SURVIVAL_DAILY_HAZARD) ** offset
        for offset in range(M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS)
    )
    total = math.fsum(raw)
    return tuple(value / total for value in raw)


M03R_V16_SURVIVAL_WEIGHTS = m03r_v16_survival_weights()


@dataclass(frozen=True, slots=True)
class M03RV16PredictiveSetting:
    setting_index: int
    setting_id: str
    selection_target: Literal[
        "h21-cumulative-factor-residual",
        "h30-cumulative-factor-residual",
        "survival-weighted-1-30-mean-factor-residual",
    ]
    selection_support_sessions: int
    timing_horizon_sessions: int = M03R_V16_TIMING_HORIZON_SESSIONS
    score_component_weights: tuple[float, float] = M03R_V16_SCORE_COMPONENT_WEIGHTS
    uncertainty_calibration_is_separate: bool = True

    def __post_init__(self) -> None:
        expected = (
            (0, M03R_V16_SETTING_IDS[0], M03R_V16_SELECTION_TARGET_IDS[0], 21),
            (1, M03R_V16_SETTING_IDS[1], M03R_V16_SELECTION_TARGET_IDS[1], 30),
            (2, M03R_V16_SETTING_IDS[2], M03R_V16_SELECTION_TARGET_IDS[2], 30),
        )
        if (
            isinstance(self.setting_index, bool)
            or self.setting_index not in range(len(expected))
            or (
                self.setting_index,
                self.setting_id,
                self.selection_target,
                self.selection_support_sessions,
            )
            != expected[self.setting_index]
            or self.timing_horizon_sessions != M03R_V16_TIMING_HORIZON_SESSIONS
            or self.score_component_weights != M03R_V16_SCORE_COMPONENT_WEIGHTS
            or not self.uncertainty_calibration_is_separate
        ):
            raise M03RV16ProtocolError("V16 predictive setting drifted")

    @property
    def receipt_sha256(self) -> str:
        self.__post_init__()
        return _sha256(asdict(self))


M03R_V16_SETTINGS = (
    M03RV16PredictiveSetting(
        0,
        M03R_V16_SETTING_IDS[0],
        "h21-cumulative-factor-residual",
        21,
    ),
    M03RV16PredictiveSetting(
        1,
        M03R_V16_SETTING_IDS[1],
        "h30-cumulative-factor-residual",
        30,
    ),
    M03RV16PredictiveSetting(
        2,
        M03R_V16_SETTING_IDS[2],
        "survival-weighted-1-30-mean-factor-residual",
        30,
    ),
)


def resolve_m03r_v16_setting(setting: int | str) -> M03RV16PredictiveSetting:
    if isinstance(setting, bool):
        raise M03RV16ProtocolError("V16 setting identity is invalid")
    if isinstance(setting, int) and setting in range(len(M03R_V16_SETTINGS)):
        return M03R_V16_SETTINGS[setting]
    if isinstance(setting, str):
        for row in M03R_V16_SETTINGS:
            if row.setting_id == setting:
                return row
    raise M03RV16ProtocolError("unknown V16 predictive setting")


@dataclass(frozen=True, slots=True)
class M03RV16PredictiveSpec:
    seed: int = 17
    chronological_fold_count: int = 6
    qualification_origins_per_fold: int = 63
    inner_validation_origins_per_fold: int = 32
    optimizer_to_validation_embargo_sessions: int = 31
    qualification_purge_sessions: int = 30
    observation_context_sessions: int = 252
    episode_state_rows: int = 345
    origins_per_update: int = 63
    minimum_score_training_epochs: int = 4
    maximum_score_training_epochs: int = 24
    checkpoint_patience_epochs: int = 4
    scale_calibration_epochs: int = 4
    score_learning_rates: tuple[float, float] = (2.0e-5, 1.0e-4)
    scale_calibration_learning_rate: float = 1.0e-4
    learning_rate_warmup_fraction: float = 0.05
    minimum_learning_rate_multiplier: float = 0.10
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    expected_world_size: int = 2
    maximum_workers: int = 3
    maximum_h100_requests: int = 6
    minimum_mean_spearman_rank_ic: float = 0.020
    minimum_positive_mean_ic_fold_count: int = 4
    minimum_positive_spread_fold_count: int = 4
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 160_017
    bootstrap_primary_block_sessions: int = 21
    minimum_gross_active_return_lcb: float = 0.0
    minimum_net_10bp_active_return_lcb: float = 0.0
    minimum_spread_lcb: float = 0.0
    minimum_break_even_one_way_cost_basis_points: float = 10.0
    evaluation_cost_basis_points: tuple[float, ...] = (
        0.0,
        1.0,
        2.0,
        3.0,
        5.0,
        10.0,
        20.0,
        40.0,
    )
    economic_optimizer_updates: int = 0
    reinforcement_learning_updates: int = 0
    outer_2026_access_authorized: bool = False
    v15_model_or_optimizer_state_reuse_authorized: bool = False
    future_selected_universe: bool = True
    development_only: bool = True
    reportable: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        if (
            self.seed != 17
            or self.chronological_fold_count != 6
            or self.qualification_origins_per_fold != 63
            or self.inner_validation_origins_per_fold != 32
            or self.optimizer_to_validation_embargo_sessions
            != M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS + 1
            or self.qualification_purge_sessions != 30
            or self.observation_context_sessions != 252
            or self.episode_state_rows != 345
            or self.origins_per_update != 63
            or self.minimum_score_training_epochs != 4
            or self.maximum_score_training_epochs != 24
            or self.checkpoint_patience_epochs != 4
            or self.scale_calibration_epochs != 4
            or self.score_learning_rates != (2.0e-5, 1.0e-4)
            or self.scale_calibration_learning_rate != 1.0e-4
            or self.learning_rate_warmup_fraction != 0.05
            or self.minimum_learning_rate_multiplier != 0.10
            or self.weight_decay != 1.0e-4
            or self.gradient_clip_norm != 1.0
            or self.expected_world_size != 2
            or self.maximum_workers != len(M03R_V16_SETTINGS)
            or self.maximum_h100_requests != 6
            or self.minimum_mean_spearman_rank_ic != 0.020
            or self.minimum_positive_mean_ic_fold_count != 4
            or self.minimum_positive_spread_fold_count != 4
            or self.bootstrap_replicates != 10_000
            or self.bootstrap_seed != 160_017
            or self.bootstrap_primary_block_sessions != 21
            or self.minimum_gross_active_return_lcb != 0.0
            or self.minimum_net_10bp_active_return_lcb != 0.0
            or self.minimum_spread_lcb != 0.0
            or self.minimum_break_even_one_way_cost_basis_points != 10.0
            or self.evaluation_cost_basis_points
            != (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0)
            or self.economic_optimizer_updates != 0
            or self.reinforcement_learning_updates != 0
            or self.outer_2026_access_authorized
            or self.v15_model_or_optimizer_state_reuse_authorized
            or not self.future_selected_universe
            or not self.development_only
            or self.reportable
            or self.promotable
        ):
            raise M03RV16ProtocolError("V16 predictive specification drifted")

    @property
    def receipt_sha256(self) -> str:
        self.__post_init__()
        return _sha256(asdict(self))


M03R_V16_PREDICTIVE_SPEC = M03RV16PredictiveSpec()
M03R_V16_PROTOCOL_SHA256 = _sha256(
    {
        "generation": M03R_V16_PROTOCOL_GENERATION,
        "design_id": M03R_V16_DESIGN_ID,
        "parent_protocol_sha256": M03R_V15_PROTOCOL_SHA256,
        "selection_targets": M03R_V16_SELECTION_TARGET_IDS,
        "timing_horizon_sessions": M03R_V16_TIMING_HORIZON_SESSIONS,
        "maximum_target_support_sessions": M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS,
        "survival_daily_hazard": M03R_V16_SURVIVAL_DAILY_HAZARD,
        "survival_weights": M03R_V16_SURVIVAL_WEIGHTS,
        "episode_schedule_rule": M03R_V16_EPISODE_SCHEDULE_RULE,
        "fold_geometry_rule": M03R_V16_FOLD_GEOMETRY_RULE,
        "executable_score_rule": M03R_V16_EXECUTABLE_SCORE_RULE,
        "label_support_rule": M03R_V16_LABEL_SUPPORT_RULE,
        "fill_rule": M03R_V16_FILL_RULE,
        "optimizer_rule": M03R_V16_OPTIMIZER_RULE,
        "checkpoint_selection_rule": M03R_V16_CHECKPOINT_SELECTION_RULE,
        "settings": tuple(setting.receipt_sha256 for setting in M03R_V16_SETTINGS),
        "predictive_spec": M03R_V16_PREDICTIVE_SPEC.receipt_sha256,
    }
)


__all__ = [
    "M03R_V16_DESIGN_ID",
    "M03R_V16_EPISODE_SCHEDULE_RULE",
    "M03R_V16_EXECUTABLE_SCORE_RULE",
    "M03R_V16_FOLD_GEOMETRY_RULE",
    "M03R_V16_FILL_RULE",
    "M03R_V16_LABEL_SUPPORT_RULE",
    "M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS",
    "M03R_V16_OPTIMIZER_RULE",
    "M03R_V16_PREDICTIVE_SPEC",
    "M03R_V16_PROTOCOL_GENERATION",
    "M03R_V16_PROTOCOL_SHA256",
    "M03R_V16_SCORE_COMPONENT_WEIGHTS",
    "M03R_V16_CHECKPOINT_SELECTION_RULE",
    "M03R_V16_SELECTION_TARGET_IDS",
    "M03R_V16_SETTINGS",
    "M03R_V16_SETTING_IDS",
    "M03R_V16_SURVIVAL_DAILY_HAZARD",
    "M03R_V16_SURVIVAL_WEIGHTS",
    "M03R_V16_TIMING_HORIZON_SESSIONS",
    "M03RV16PredictiveSetting",
    "M03RV16PredictiveSpec",
    "M03RV16ProtocolError",
    "m03r_v16_survival_weights",
    "resolve_m03r_v16_setting",
]
