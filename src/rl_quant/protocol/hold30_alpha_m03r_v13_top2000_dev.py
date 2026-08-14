"""Context-matched direct-h3 TOP2000 M03R predictive research contract.

V13 is a fresh predictive-only generation.  It does not reinterpret or resume
V12 model/optimizer state, authorize an economic optimizer, or permit access
to 2026 outcomes.  The future-selected TOP2000 surface remains a mechanism
diagnostic rather than reportable or promotable evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_PROTOCOL_SHA256,
)

M03R_V13_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v13-context-matched-direct-h3-v1"
)
M03R_V13_DESIGN_ID = "daily-ohlcv-top2000-dev-m03r-v13-direct-h3-control-v1"
M03R_V13_SELECTED_HORIZON_SESSIONS = 3
M03R_V13_HORIZONS = (M03R_V13_SELECTED_HORIZON_SESSIONS,)
M03R_V13_SETTING_IDS = (
    "V13-P0-direct-rank-gaussian-economic-h3",
    "V13-P1-direct-economic-h3-control",
)
M03R_V13_EPISODE_SCHEDULE_RULE = (
    "full-252-context-every-origin-once-per-epoch-paired-v1"
)
M03R_V13_FOLD_GEOMETRY_RULE = (
    "six-expanding-63-origin-h3-qualification-through-pre2026-cache-end-v1"
)
M03R_V13_SIMPLE_SLEEVE_RULE = (
    "causal-action-operator-fixed-25bp-rank-sleeve-postfill-return-v1"
)


class M03RV13ProtocolError(ValueError):
    """The immutable v13 predictive research contract drifted."""


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


@dataclass(frozen=True, slots=True)
class M03RV13PredictiveSetting:
    setting_index: int
    setting_id: str
    ranking_objective: Literal["rank-gaussian-correlation", "none"]
    component_weights: tuple[float, float, float]
    selected_horizon_sessions: int = M03R_V13_SELECTED_HORIZON_SESSIONS
    target_mode: Literal["factor-residual"] = "factor-residual"
    rank_score_is_economic_mean: bool = True

    def __post_init__(self) -> None:
        expected = (
            (
                0,
                M03R_V13_SETTING_IDS[0],
                "rank-gaussian-correlation",
                (0.25, 0.45, 0.30),
            ),
            (1, M03R_V13_SETTING_IDS[1], "none", (0.0, 0.60, 0.40)),
        )
        if (
            isinstance(self.setting_index, bool)
            or not isinstance(self.setting_index, int)
            or self.setting_index not in range(len(expected))
            or (
                self.setting_index,
                self.setting_id,
                self.ranking_objective,
                self.component_weights,
            )
            != expected[self.setting_index]
            or abs(sum(self.component_weights) - 1.0) > 1.0e-12
            or self.selected_horizon_sessions
            != M03R_V13_SELECTED_HORIZON_SESSIONS
            or self.target_mode != "factor-residual"
            or not self.rank_score_is_economic_mean
        ):
            raise M03RV13ProtocolError("v13 predictive setting drifted")

    @property
    def receipt_sha256(self) -> str:
        self.__post_init__()
        return _sha256(asdict(self))


M03R_V13_SETTINGS = (
    M03RV13PredictiveSetting(
        0,
        M03R_V13_SETTING_IDS[0],
        "rank-gaussian-correlation",
        (0.25, 0.45, 0.30),
    ),
    M03RV13PredictiveSetting(
        1,
        M03R_V13_SETTING_IDS[1],
        "none",
        (0.0, 0.60, 0.40),
    ),
)


@dataclass(frozen=True, slots=True)
class M03RV13PredictiveSpec:
    seed: int = 17
    chronological_fold_count: int = 6
    qualification_origins_per_fold: int = 63
    purge_sessions: int = 30
    observation_context_sessions: int = 252
    episode_state_rows: int = 378
    origins_per_update: int = 64
    training_epochs: int = 8
    expected_world_size: int = 2
    maximum_workers: int = 2
    maximum_h100_requests: int = 4
    early_stopping_enabled: bool = False
    minimum_mean_spearman_rank_ic: float = 0.020
    minimum_positive_mean_ic_fold_count: int = 4
    minimum_positive_median_ic_fold_count: int = 4
    minimum_positive_date_fraction_fold_count: int = 4
    minimum_positive_ic_date_fraction: float = 0.50
    minimum_positive_spread_fold_count: int = 4
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 130_017
    bootstrap_primary_block_sessions: int = 21
    bootstrap_sensitivity_block_sessions: tuple[int, int] = (10, 30)
    minimum_gross_active_return_lcb: float = 0.0
    minimum_net_10bp_active_return_lcb: float = 0.0
    minimum_spread_lcb: float = 0.0
    minimum_break_even_one_way_cost_basis_points: float = 10.0
    minimum_median_signal_projection_retention: float = 0.50
    minimum_fold_median_signal_projection_retention: float = 0.20
    minimum_median_risk_projection_retention: float = 0.50
    minimum_fold_median_risk_projection_retention: float = 0.20
    simple_sleeve_maximum_active_one_way_mass: float = 0.0025
    evaluation_cost_basis_points: tuple[float, float, float, float] = (
        0.0,
        10.0,
        20.0,
        40.0,
    )
    economic_optimizer_updates: int = 0
    outer_2026_access_authorized: bool = False
    v12_model_or_optimizer_state_reuse_authorized: bool = False
    future_selected_universe: bool = True
    development_only: bool = True
    reportable: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        if (
            self.seed != 17
            or self.chronological_fold_count != 6
            or self.qualification_origins_per_fold != 63
            or self.purge_sessions != 30
            or self.observation_context_sessions != 252
            or self.episode_state_rows != 378
            or self.origins_per_update != 64
            or self.training_epochs != 8
            or self.expected_world_size != 2
            or self.maximum_workers != len(M03R_V13_SETTINGS)
            or self.maximum_h100_requests != 4
            or self.early_stopping_enabled
            or self.minimum_mean_spearman_rank_ic != 0.020
            or self.minimum_positive_mean_ic_fold_count != 4
            or self.minimum_positive_median_ic_fold_count != 4
            or self.minimum_positive_date_fraction_fold_count != 4
            or self.minimum_positive_ic_date_fraction != 0.50
            or self.minimum_positive_spread_fold_count != 4
            or self.bootstrap_replicates != 10_000
            or self.bootstrap_seed != 130_017
            or self.bootstrap_primary_block_sessions != 21
            or self.bootstrap_sensitivity_block_sessions != (10, 30)
            or self.minimum_gross_active_return_lcb != 0.0
            or self.minimum_net_10bp_active_return_lcb != 0.0
            or self.minimum_spread_lcb != 0.0
            or self.minimum_break_even_one_way_cost_basis_points != 10.0
            or self.minimum_median_signal_projection_retention != 0.50
            or self.minimum_fold_median_signal_projection_retention != 0.20
            or self.minimum_median_risk_projection_retention != 0.50
            or self.minimum_fold_median_risk_projection_retention != 0.20
            or self.simple_sleeve_maximum_active_one_way_mass != 0.0025
            or self.evaluation_cost_basis_points != (0.0, 10.0, 20.0, 40.0)
            or self.economic_optimizer_updates != 0
            or self.outer_2026_access_authorized
            or self.v12_model_or_optimizer_state_reuse_authorized
            or not self.future_selected_universe
            or not self.development_only
            or self.reportable
            or self.promotable
        ):
            raise M03RV13ProtocolError("v13 predictive specification drifted")

    @property
    def receipt_sha256(self) -> str:
        self.__post_init__()
        return _sha256(asdict(self))


M03R_V13_PREDICTIVE_SPEC = M03RV13PredictiveSpec()
M03R_V13_PROTOCOL_SHA256 = _sha256(
    {
        "generation": M03R_V13_PROTOCOL_GENERATION,
        "design_id": M03R_V13_DESIGN_ID,
        "parent_protocol_sha256": M03R_V12_PROTOCOL_SHA256,
        "horizons": M03R_V13_HORIZONS,
        "episode_schedule_rule": M03R_V13_EPISODE_SCHEDULE_RULE,
        "fold_geometry_rule": M03R_V13_FOLD_GEOMETRY_RULE,
        "simple_sleeve_rule": M03R_V13_SIMPLE_SLEEVE_RULE,
        "settings": tuple(setting.receipt_sha256 for setting in M03R_V13_SETTINGS),
        "predictive_spec": M03R_V13_PREDICTIVE_SPEC.receipt_sha256,
    }
)


def resolve_m03r_v13_setting(value: int | str) -> M03RV13PredictiveSetting:
    if isinstance(value, bool):
        raise M03RV13ProtocolError("boolean is not a v13 setting")
    for setting in M03R_V13_SETTINGS:
        if value in {setting.setting_index, setting.setting_id}:
            setting.__post_init__()
            return setting
    raise M03RV13ProtocolError(f"unknown v13 setting: {value!r}")


__all__ = [
    "M03R_V13_DESIGN_ID",
    "M03R_V13_EPISODE_SCHEDULE_RULE",
    "M03R_V13_FOLD_GEOMETRY_RULE",
    "M03R_V13_HORIZONS",
    "M03R_V13_PREDICTIVE_SPEC",
    "M03R_V13_PROTOCOL_GENERATION",
    "M03R_V13_PROTOCOL_SHA256",
    "M03R_V13_SELECTED_HORIZON_SESSIONS",
    "M03R_V13_SETTING_IDS",
    "M03R_V13_SIMPLE_SLEEVE_RULE",
    "M03R_V13_SETTINGS",
    "M03RV13PredictiveSetting",
    "M03RV13PredictiveSpec",
    "M03RV13ProtocolError",
    "resolve_m03r_v13_setting",
]
