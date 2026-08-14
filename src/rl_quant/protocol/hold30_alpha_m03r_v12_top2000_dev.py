"""Rank/scale-decoupled TOP2000 M03R predictive research contract.

V12 is a fresh predictive-only generation motivated by the completed v11 a15
inference audit.  It does not reinterpret v11 artifacts, authorize economic
optimization, or permit access to 2026 outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
)

M03R_V12_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v12-rank-scale-decoupled-v1"
)
M03R_V12_DESIGN_ID = "daily-ohlcv-top2000-dev-m03r-v12-paired-rank-economic-scale-v1"
M03R_V12_HORIZONS = (3, 5, 21, 30, 63)
M03R_V12_SELECTED_HORIZON_SESSIONS = 3
M03R_V12_ELIGIBLE_EXECUTION_HORIZONS = (M03R_V12_SELECTED_HORIZON_SESSIONS,)
M03R_V12_SETTING_IDS = (
    "V12-P0-separate-listwise-rank-economic-scale",
    "V12-P1-separate-rank-gaussian-economic-scale",
    "V12-P2-economic-scale-no-rank-control",
)
M03R_V12_RANK_TO_ECONOMIC_ENCODER_GRADIENT_RATIO_MAX = 0.25
M03R_V12_TURNOVER_UTILIZATION_RULE = "tanh-rms-cost-clearing-zscore-temperature-2-v1"
M03R_V12_EPISODE_SCHEDULE_RULE = "paired-across-settings-v1"
M03R_V12_BOOTSTRAP_RULE = "fold-bounded-circular-moving-block-common-draws-v1"


class M03RV12ProtocolError(ValueError):
    """The immutable v12 predictive research contract drifted."""


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
class M03RV12PredictiveSetting:
    setting_index: int
    setting_id: str
    ranking_objective: Literal[
        "standardized-return-listwise",
        "rank-gaussian-correlation",
        "none",
    ]
    component_weights: tuple[float, float, float]
    horizon_loss_weights: tuple[float, float, float, float, float] = (
        0.10,
        0.10,
        0.30,
        0.35,
        0.15,
    )
    target_mode: Literal["factor-residual"] = "factor-residual"
    separate_rank_score_head: bool = True
    separate_economic_mean_scale_heads: bool = True

    def __post_init__(self) -> None:
        expected = (
            (
                0,
                M03R_V12_SETTING_IDS[0],
                "standardized-return-listwise",
                (0.25, 0.45, 0.30),
            ),
            (
                1,
                M03R_V12_SETTING_IDS[1],
                "rank-gaussian-correlation",
                (0.25, 0.45, 0.30),
            ),
            (2, M03R_V12_SETTING_IDS[2], "none", (0.0, 0.60, 0.40)),
        )
        observed = (
            self.setting_index,
            self.setting_id,
            self.ranking_objective,
            self.component_weights,
        )
        if (
            isinstance(self.setting_index, bool)
            or not isinstance(self.setting_index, int)
            or self.setting_index not in range(len(expected))
            or observed != expected[self.setting_index]
            or self.horizon_loss_weights != (0.10, 0.10, 0.30, 0.35, 0.15)
            or abs(sum(self.component_weights) - 1.0) > 1.0e-12
            or abs(sum(self.horizon_loss_weights) - 1.0) > 1.0e-12
            or self.target_mode != "factor-residual"
            or not self.separate_rank_score_head
            or not self.separate_economic_mean_scale_heads
        ):
            raise M03RV12ProtocolError("v12 predictive setting drifted")

    @property
    def receipt_sha256(self) -> str:
        self.__post_init__()
        return _sha256(asdict(self))


M03R_V12_SETTINGS = (
    M03RV12PredictiveSetting(
        0,
        M03R_V12_SETTING_IDS[0],
        "standardized-return-listwise",
        (0.25, 0.45, 0.30),
    ),
    M03RV12PredictiveSetting(
        1,
        M03R_V12_SETTING_IDS[1],
        "rank-gaussian-correlation",
        (0.25, 0.45, 0.30),
    ),
    M03RV12PredictiveSetting(
        2,
        M03R_V12_SETTING_IDS[2],
        "none",
        (0.0, 0.60, 0.40),
    ),
)


@dataclass(frozen=True, slots=True)
class M03RV12PredictiveSpec:
    optimizer_updates: int = 64
    seed: int = 17
    chronological_fold_count: int = 6
    expected_world_size: int = 2
    maximum_workers: int = 3
    maximum_h100_requests: int = 6
    qualification_updates: tuple[int, ...] = (64,)
    early_stopping_enabled: bool = False
    maximum_rank_to_economic_encoder_gradient_ratio: float = (
        M03R_V12_RANK_TO_ECONOMIC_ENCODER_GRADIENT_RATIO_MAX
    )
    rank_head_gradient_clip_norm: float = 0.25
    encoder_gradient_clip_norm: float = 1.0
    economic_head_gradient_clip_norm: float = 1.0
    turnover_utilization_temperature: float = 2.0
    minimum_mean_spearman_rank_ic: float = 0.020
    minimum_positive_mean_ic_fold_count: int = 4
    minimum_positive_median_ic_fold_count: int = 4
    minimum_positive_date_fraction: float = 0.50
    minimum_positive_date_fraction_fold_count: int = 4
    minimum_positive_spread_fold_count: int = 4
    minimum_prediction_cross_sectional_std: float = 1.0e-5
    prediction_target_dispersion_ratio_range: tuple[float, float] = (0.05, 20.0)
    bootstrap_replicates: int = 10_000
    bootstrap_primary_block_sessions: int = 21
    bootstrap_sensitivity_block_sessions: tuple[int, int] = (10, 30)
    minimum_gross_active_return_lcb: float = 0.0
    minimum_spread_lcb: float = 0.0
    minimum_break_even_one_way_cost_basis_points: float = 10.0
    minimum_median_projection_retention: float = 0.50
    minimum_fold_projection_retention: float = 0.20
    economic_optimizer_updates: int = 0
    outer_2026_access_authorized: bool = False
    v11_model_or_optimizer_state_reuse_authorized: bool = False
    future_selected_universe: bool = True
    development_only: bool = True
    reportable: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        if (
            self.optimizer_updates != 64
            or self.seed != 17
            or self.chronological_fold_count != 6
            or self.expected_world_size != 2
            or self.maximum_workers != 3
            or self.maximum_h100_requests != 6
            or self.qualification_updates != (64,)
            or self.early_stopping_enabled
            or self.maximum_rank_to_economic_encoder_gradient_ratio
            != M03R_V12_RANK_TO_ECONOMIC_ENCODER_GRADIENT_RATIO_MAX
            or self.rank_head_gradient_clip_norm != 0.25
            or self.encoder_gradient_clip_norm != 1.0
            or self.economic_head_gradient_clip_norm != 1.0
            or self.turnover_utilization_temperature != 2.0
            or self.minimum_mean_spearman_rank_ic != 0.020
            or self.minimum_positive_mean_ic_fold_count != 4
            or self.minimum_positive_median_ic_fold_count != 4
            or self.minimum_positive_date_fraction != 0.50
            or self.minimum_positive_date_fraction_fold_count != 4
            or self.minimum_positive_spread_fold_count != 4
            or self.minimum_prediction_cross_sectional_std != 1.0e-5
            or self.prediction_target_dispersion_ratio_range != (0.05, 20.0)
            or self.bootstrap_replicates != 10_000
            or self.bootstrap_primary_block_sessions != 21
            or self.bootstrap_sensitivity_block_sessions != (10, 30)
            or self.minimum_gross_active_return_lcb != 0.0
            or self.minimum_spread_lcb != 0.0
            or self.minimum_break_even_one_way_cost_basis_points != 10.0
            or self.minimum_median_projection_retention != 0.50
            or self.minimum_fold_projection_retention != 0.20
            or self.economic_optimizer_updates != 0
            or self.outer_2026_access_authorized
            or self.v11_model_or_optimizer_state_reuse_authorized
            or not self.future_selected_universe
            or not self.development_only
            or self.reportable
            or self.promotable
        ):
            raise M03RV12ProtocolError("v12 predictive specification drifted")


M03R_V12_PREDICTIVE_SPEC = M03RV12PredictiveSpec()
M03R_V12_PROTOCOL_SHA256 = _sha256(
    {
        "generation": M03R_V12_PROTOCOL_GENERATION,
        "design_id": M03R_V12_DESIGN_ID,
        "horizons": M03R_V12_HORIZONS,
        "eligible_execution_horizons": M03R_V12_ELIGIBLE_EXECUTION_HORIZONS,
        "settings": tuple(asdict(row) for row in M03R_V12_SETTINGS),
        "spec": asdict(M03R_V12_PREDICTIVE_SPEC),
        "turnover_utilization_rule": M03R_V12_TURNOVER_UTILIZATION_RULE,
        "episode_schedule_rule": M03R_V12_EPISODE_SCHEDULE_RULE,
        "bootstrap_rule": M03R_V12_BOOTSTRAP_RULE,
        "superseded_predictive_protocol_sha256": M03R_V11_PROTOCOL_SHA256,
    }
)


def resolve_m03r_v12_setting(value: int | str) -> M03RV12PredictiveSetting:
    if isinstance(value, bool):
        raise M03RV12ProtocolError("boolean is not a v12 setting identity")
    for row in M03R_V12_SETTINGS:
        if value in {row.setting_index, row.setting_id}:
            return row
    raise M03RV12ProtocolError(f"unknown v12 setting: {value!r}")


__all__ = [
    "M03R_V12_DESIGN_ID",
    "M03R_V12_BOOTSTRAP_RULE",
    "M03R_V12_ELIGIBLE_EXECUTION_HORIZONS",
    "M03R_V12_EPISODE_SCHEDULE_RULE",
    "M03R_V12_HORIZONS",
    "M03R_V12_PREDICTIVE_SPEC",
    "M03R_V12_PROTOCOL_GENERATION",
    "M03R_V12_PROTOCOL_SHA256",
    "M03R_V12_RANK_TO_ECONOMIC_ENCODER_GRADIENT_RATIO_MAX",
    "M03R_V12_SELECTED_HORIZON_SESSIONS",
    "M03R_V12_SETTINGS",
    "M03R_V12_SETTING_IDS",
    "M03R_V12_TURNOVER_UTILIZATION_RULE",
    "M03RV12PredictiveSetting",
    "M03RV12ProtocolError",
    "resolve_m03r_v12_setting",
]
