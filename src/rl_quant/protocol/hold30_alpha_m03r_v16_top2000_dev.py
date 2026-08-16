"""Paired holding-aligned selection research after the V15 h3 screen.

V16 is a selection-only, predictive development experiment.  It compares
three long-horizon targets on the same 30-session asset/date support and the
same action/label residual operators.  The Hold-30-prior target is the sole
predeclared primary hypothesis; h21 and h30 are explanatory controls.  V16
cannot train timing, uncertainty, an economic controller, or RL, and it cannot
access 2026 outcomes.
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
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC

M03R_V16_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v16-holding-aligned-selection-v5"
)
M03R_V16_DESIGN_ID = "derived-repair-terminal-schedule-authority-hold30-primary-v5"
M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS = 30
M03R_V16_COMMON_LABEL_SUPPORT_SESSIONS = 30
M03RV16SelectionTarget = Literal[
    "h21-cumulative-common30-factor-residual",
    "h30-cumulative-common30-factor-residual",
    "hold30-prior-truncated-1-30-cumulative-common30-factor-residual",
]
M03R_V16_SELECTION_TARGET_IDS: tuple[M03RV16SelectionTarget, ...] = (
    "h21-cumulative-common30-factor-residual",
    "h30-cumulative-common30-factor-residual",
    "hold30-prior-truncated-1-30-cumulative-common30-factor-residual",
)
M03R_V16_SETTING_IDS = (
    "V16-R0-h21-selection-control",
    "V16-R1-h30-selection-control",
    "V16-R2-hold30-prior-selection-primary",
)
M03R_V16_PRIMARY_SETTING_INDEX = 2
M03R_V16_SCORE_EPOCHS = 8
M03R_V16_EPISODE_SCHEDULE_RULE = (
    "origin-aligned-252-context-balanced-block-paired-selection-only-v3"
)
M03R_V16_FOLD_GEOMETRY_RULE = (
    "five-disjoint-93-advance-inner63-diagnostic-63-origin-common30-pre2026-v2"
)
M03R_V16_EXECUTABLE_SCORE_RULE = "dimensionless-action-projected-terminal-checkpoint-recomputed-score-used-everywhere-v4"
M03R_V16_LABEL_SUPPORT_RULE = (
    "common-origin-action-eligible-intersect-full-30-session-future-support-v2"
)
M03R_V16_FILL_RULE = (
    "observe-close-t-fill-next-close-t-plus-1-development-diagnostic-v1"
)
M03R_V16_OPTIMIZER_RULE = (
    "selection-only-encoder-and-dimensionless-mean-module-aware-decay-v2"
)
M03R_V16_CHECKPOINT_SELECTION_RULE = (
    "fixed-terminal-epoch-8-inner-validation-diagnostics-only-v2"
)
M03R_V16_TARGET_SELECTION_RULE = (
    "hold30-prior-setting-2-primary-h21-h30-explanatory-controls-v1"
)
M03R_V16_COHORT_SLEEVE_RULE = "self-financing-signal-cohorts-fresh-derived-risk-repair-truncated30-terminal-close-v4"


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


def _sigmoid(value: float) -> float:
    bounded = min(20.0, max(-20.0, value))
    return 1.0 / (1.0 + math.exp(-bounded))


def m03r_v16_hold30_reference_release_hazard(age: int) -> float:
    """Return the frozen neutral Hold-30 release probability at one age."""

    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
        raise M03RV16ProtocolError("Hold-30 release age must be nonnegative")
    clamped = min(age, 60)
    beta = -2.0 + (clamped - 30.0) / 4.0
    minimum = _sigmoid(beta - 12.0)
    release = _sigmoid(beta)
    return (release - minimum) / (1.0 - minimum)


def m03r_v16_hold30_reference_release_hazards() -> tuple[float, ...]:
    """Return the neutral Hold-30 age-clock hazards for earned days 1..30.

    This is the scalar equivalent of ``hold30_release_hazard(age, 0)``.  The
    subtraction of the ``-12`` endpoint probability is part of the frozen
    normalized release mechanism, not an approximation.
    """

    return tuple(
        m03r_v16_hold30_reference_release_hazard(age)
        for age in range(1, M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS + 1)
    )


M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS = m03r_v16_hold30_reference_release_hazards()
M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS_60 = tuple(
    m03r_v16_hold30_reference_release_hazard(age) for age in range(1, 61)
)


def m03r_v16_survival_weights() -> tuple[float, ...]:
    """Return unnormalized survival value weights for earned days 1..30.

    The finite 30-session target is explicitly a truncated cumulative value,
    not a normalized alpha rate and not a claim that the position expires at
    day 30.  Remaining survival is published in the protocol for attribution.
    """

    survival = 1.0
    weights: list[float] = []
    for hazard in M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS:
        weights.append(survival)
        survival *= 1.0 - hazard
    return tuple(weights)


M03R_V16_SURVIVAL_WEIGHTS = m03r_v16_survival_weights()
M03R_V16_SURVIVAL_AFTER_DAY_30 = math.prod(
    1.0 - value for value in M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS
)


def m03r_v16_selection_target_scale_from_id(target_id: str) -> float:
    """Return the frozen economic-unit scale used to expose a z-score head."""

    weights = m03r_v16_selection_target_weights_from_id(target_id)
    return 0.02 * math.sqrt(math.fsum(value * value for value in weights))


def m03r_v16_selection_target_weights_from_id(
    target_id: str,
) -> tuple[float, ...]:
    """Return the economic-unit aggregation weights bound to one target."""

    if target_id == M03R_V16_SELECTION_TARGET_IDS[0]:
        return (1.0,) * 21
    if target_id == M03R_V16_SELECTION_TARGET_IDS[1]:
        return (1.0,) * 30
    if target_id == M03R_V16_SELECTION_TARGET_IDS[2]:
        return M03R_V16_SURVIVAL_WEIGHTS
    raise M03RV16ProtocolError("unknown V16 target weights")


@dataclass(frozen=True, slots=True)
class M03RV16PredictiveSetting:
    setting_index: int
    setting_id: str
    selection_target: M03RV16SelectionTarget
    numerical_target_support_sessions: int
    common_label_support_sessions: int = M03R_V16_COMMON_LABEL_SUPPORT_SESSIONS
    selection_loss_weight: float = 1.0
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        expected = (
            (0, M03R_V16_SETTING_IDS[0], M03R_V16_SELECTION_TARGET_IDS[0], 21, False),
            (1, M03R_V16_SETTING_IDS[1], M03R_V16_SELECTION_TARGET_IDS[1], 30, False),
            (2, M03R_V16_SETTING_IDS[2], M03R_V16_SELECTION_TARGET_IDS[2], 30, True),
        )
        if (
            isinstance(self.setting_index, bool)
            or self.setting_index not in range(len(expected))
            or (
                self.setting_index,
                self.setting_id,
                self.selection_target,
                self.numerical_target_support_sessions,
                self.promotion_eligible,
            )
            != expected[self.setting_index]
            or self.common_label_support_sessions
            != M03R_V16_COMMON_LABEL_SUPPORT_SESSIONS
            or self.selection_loss_weight != 1.0
        ):
            raise M03RV16ProtocolError("V16 predictive setting drifted")

    @property
    def selection_target_scale(self) -> float:
        self.__post_init__()
        return m03r_v16_selection_target_scale_from_id(self.selection_target)

    @property
    def receipt_sha256(self) -> str:
        self.__post_init__()
        return _sha256(asdict(self))


M03R_V16_SETTINGS = (
    M03RV16PredictiveSetting(
        0, M03R_V16_SETTING_IDS[0], M03R_V16_SELECTION_TARGET_IDS[0], 21
    ),
    M03RV16PredictiveSetting(
        1, M03R_V16_SETTING_IDS[1], M03R_V16_SELECTION_TARGET_IDS[1], 30
    ),
    M03RV16PredictiveSetting(
        2,
        M03R_V16_SETTING_IDS[2],
        M03R_V16_SELECTION_TARGET_IDS[2],
        30,
        promotion_eligible=True,
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
    chronological_fold_count: int = 5
    qualification_origins_per_fold: int = 63
    inner_validation_origins_per_fold: int = 63
    optimizer_to_validation_embargo_sessions: int = 31
    qualification_purge_sessions: int = 30
    observation_context_sessions: int = 252
    episode_state_rows: int = 345
    origins_per_update: int = 63
    score_training_epochs: int = M03R_V16_SCORE_EPOCHS
    score_learning_rates: tuple[float, float] = (2.0e-5, 1.0e-4)
    learning_rate_warmup_fraction: float = 0.05
    minimum_learning_rate_multiplier: float = 0.10
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    expected_world_size: int = 2
    maximum_workers: int = 3
    maximum_h100_requests: int = 6
    primary_setting_index: int = M03R_V16_PRIMARY_SETTING_INDEX
    minimum_mean_spearman_rank_ic: float = 0.020
    minimum_positive_mean_ic_fold_count: int = 4
    minimum_positive_spread_fold_count: int = 4
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 160_017
    bootstrap_primary_block_sessions: int = 42
    bootstrap_sensitivity_block_sessions: tuple[int, int] = (30, 63)
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
    cohort_total_active_one_way_mass: float = 0.0025
    hold_target_sessions: int = LEGACY_HOLD30_TARGET_SPEC.target_sessions
    hold_age_cap_sessions: int = LEGACY_HOLD30_TARGET_SPEC.age_cap_sessions
    hold_prior_family: str = LEGACY_HOLD30_TARGET_SPEC.prior_family
    hold_target_spec_sha256: str = LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    # The final cohort earns its first return on its decision/fill step, so a
    # 30-return horizon needs 29 additional no-new-decision transitions.
    cohort_no_new_decision_tail_sessions: int = 29
    timing_optimizer_updates: int = 0
    uncertainty_calibration_updates: int = 0
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
            or self.chronological_fold_count != 5
            or self.qualification_origins_per_fold != 63
            or self.inner_validation_origins_per_fold != 63
            or self.optimizer_to_validation_embargo_sessions != 31
            or self.qualification_purge_sessions != 30
            or self.observation_context_sessions != 252
            or self.episode_state_rows != 345
            or self.origins_per_update != 63
            or self.score_training_epochs != 8
            or self.score_learning_rates != (2.0e-5, 1.0e-4)
            or self.learning_rate_warmup_fraction != 0.05
            or self.minimum_learning_rate_multiplier != 0.10
            or self.weight_decay != 1.0e-4
            or self.gradient_clip_norm != 1.0
            or self.expected_world_size != 2
            or self.maximum_workers != len(M03R_V16_SETTINGS)
            or self.maximum_h100_requests != 6
            or self.primary_setting_index != M03R_V16_PRIMARY_SETTING_INDEX
            or self.minimum_mean_spearman_rank_ic != 0.020
            or self.minimum_positive_mean_ic_fold_count != 4
            or self.minimum_positive_spread_fold_count != 4
            or self.bootstrap_replicates != 10_000
            or self.bootstrap_seed != 160_017
            or self.bootstrap_primary_block_sessions != 42
            or self.bootstrap_sensitivity_block_sessions != (30, 63)
            or self.minimum_gross_active_return_lcb != 0.0
            or self.minimum_net_10bp_active_return_lcb != 0.0
            or self.minimum_spread_lcb != 0.0
            or self.minimum_break_even_one_way_cost_basis_points != 10.0
            or self.evaluation_cost_basis_points
            != (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0)
            or self.cohort_total_active_one_way_mass != 0.0025
            or self.hold_target_sessions != 30
            or self.hold_age_cap_sessions != 60
            or self.hold_prior_family != "legacy-hold30-v1"
            or self.hold_target_spec_sha256 != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
            or self.cohort_no_new_decision_tail_sessions != 29
            or self.timing_optimizer_updates != 0
            or self.uncertainty_calibration_updates != 0
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
        "maximum_target_support_sessions": M03R_V16_MAXIMUM_TARGET_SUPPORT_SESSIONS,
        "common_label_support_sessions": M03R_V16_COMMON_LABEL_SUPPORT_SESSIONS,
        "hold30_reference_release_hazards": M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS,
        "hold30_reference_release_hazards_60": (
            M03R_V16_HOLD30_REFERENCE_RELEASE_HAZARDS_60
        ),
        "survival_weights": M03R_V16_SURVIVAL_WEIGHTS,
        "survival_after_day_30": M03R_V16_SURVIVAL_AFTER_DAY_30,
        "hold_target_spec_sha256": LEGACY_HOLD30_TARGET_SPEC.receipt_sha256,
        "episode_schedule_rule": M03R_V16_EPISODE_SCHEDULE_RULE,
        "fold_geometry_rule": M03R_V16_FOLD_GEOMETRY_RULE,
        "executable_score_rule": M03R_V16_EXECUTABLE_SCORE_RULE,
        "label_support_rule": M03R_V16_LABEL_SUPPORT_RULE,
        "fill_rule": M03R_V16_FILL_RULE,
        "optimizer_rule": M03R_V16_OPTIMIZER_RULE,
        "checkpoint_selection_rule": M03R_V16_CHECKPOINT_SELECTION_RULE,
        "target_selection_rule": M03R_V16_TARGET_SELECTION_RULE,
        "cohort_sleeve_rule": M03R_V16_COHORT_SLEEVE_RULE,
        "settings": tuple(setting.receipt_sha256 for setting in M03R_V16_SETTINGS),
        "predictive_spec": M03R_V16_PREDICTIVE_SPEC.receipt_sha256,
    }
)


__all__ = [name for name in globals() if name.startswith("M03R_V16_")] + [
    "M03RV16PredictiveSetting",
    "M03RV16PredictiveSpec",
    "M03RV16ProtocolError",
    "M03RV16SelectionTarget",
    "m03r_v16_hold30_reference_release_hazard",
    "m03r_v16_hold30_reference_release_hazards",
    "m03r_v16_selection_target_scale_from_id",
    "m03r_v16_survival_weights",
    "resolve_m03r_v16_setting",
]
