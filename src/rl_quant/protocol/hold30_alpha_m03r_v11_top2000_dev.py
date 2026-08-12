"""Corrected paired rank-geometry protocol for TOP2000 M03R research.

V11 supersedes the unlaunched v10 proposal after a local correctness review.
It is development-only, uses no v9/v10 model or optimizer state, authorizes no
economic optimization, and forbids access to 2026 outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import (
    M03R_V10_PROTOCOL_SHA256,
)

M03R_V11_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v11-rank-geometry-corrected-v1"
)
M03R_V11_DESIGN_ID = (
    "daily-ohlcv-top2000-dev-m03r-v11-paired-qualified-residual-rank-v1"
)
M03R_V11_HORIZONS = (5, 21, 30, 63)
M03R_V11_ELIGIBLE_EXECUTION_HORIZONS = (21, 30)
M03R_V11_SETTING_IDS = (
    "V11-P0-factor-residual-standardized-listwise-control",
    "V11-P1-factor-residual-rank-gaussian-correlation",
    "V11-P2-factor-residual-rank-gaussian-21-30-only",
)
M03R_V11_EPISODE_SCHEDULE_RULE = "paired-across-settings-v1"
M03R_V11_RESIDUAL_OPERATOR_RULE = (
    "weighted-qr-reference-sector-drop-shared-target-signal-v1"
)
M03R_V11_BOOTSTRAP_RULE = "fold-bounded-circular-moving-block-common-draws-v1"


class M03RV11ProtocolError(ValueError):
    """The immutable v11 research contract drifted."""


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
class M03RV11PredictiveSetting:
    setting_index: int
    setting_id: str
    ranking_objective: Literal[
        "standardized-return-listwise",
        "rank-gaussian-correlation",
    ]
    horizon_loss_weights: tuple[float, float, float, float]
    component_weights: tuple[float, float, float] = (0.50, 0.30, 0.20)
    target_mode: Literal["factor-residual"] = "factor-residual"

    def __post_init__(self) -> None:
        expected = (
            (
                0,
                M03R_V11_SETTING_IDS[0],
                "standardized-return-listwise",
                (0.10, 0.35, 0.40, 0.15),
            ),
            (
                1,
                M03R_V11_SETTING_IDS[1],
                "rank-gaussian-correlation",
                (0.10, 0.35, 0.40, 0.15),
            ),
            (
                2,
                M03R_V11_SETTING_IDS[2],
                "rank-gaussian-correlation",
                (0.0, 7.0 / 15.0, 8.0 / 15.0, 0.0),
            ),
        )
        observed = (
            self.setting_index,
            self.setting_id,
            self.ranking_objective,
            self.horizon_loss_weights,
        )
        if (
            isinstance(self.setting_index, bool)
            or not isinstance(self.setting_index, int)
            or not 0 <= self.setting_index < len(expected)
            or observed != expected[self.setting_index]
            or self.component_weights != (0.50, 0.30, 0.20)
            or self.target_mode != "factor-residual"
            or abs(sum(self.horizon_loss_weights) - 1.0) > 1.0e-12
        ):
            raise M03RV11ProtocolError("v11 predictive setting drifted")

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


M03R_V11_SETTINGS = (
    M03RV11PredictiveSetting(
        0,
        M03R_V11_SETTING_IDS[0],
        "standardized-return-listwise",
        (0.10, 0.35, 0.40, 0.15),
    ),
    M03RV11PredictiveSetting(
        1,
        M03R_V11_SETTING_IDS[1],
        "rank-gaussian-correlation",
        (0.10, 0.35, 0.40, 0.15),
    ),
    M03RV11PredictiveSetting(
        2,
        M03R_V11_SETTING_IDS[2],
        "rank-gaussian-correlation",
        (0.0, 7.0 / 15.0, 8.0 / 15.0, 0.0),
    ),
)


@dataclass(frozen=True, slots=True)
class M03RV11PredictiveSpec:
    optimizer_updates: int = 64
    seed: int = 17
    chronological_fold_count: int = 6
    expected_world_size: int = 2
    maximum_workers: int = 3
    maximum_h100_requests: int = 6
    early_stopping_enabled: bool = False
    qualification_updates: tuple[int, ...] = (64,)
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
    minimum_net_active_return_10bp_lcb: float = 0.0
    minimum_gross_active_return_lcb: float = 0.0
    minimum_spread_lcb: float = 0.0
    minimum_break_even_one_way_cost_basis_points: float = 10.0
    minimum_median_projection_retention: float = 0.50
    minimum_fold_projection_retention: float = 0.20
    economic_optimizer_updates: int = 0
    v9_or_v10_state_reuse_authorized: bool = False
    outer_2026_access_authorized: bool = False
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
            or self.early_stopping_enabled
            or self.qualification_updates != (64,)
            or self.minimum_mean_spearman_rank_ic != 0.020
            or self.minimum_positive_mean_ic_fold_count != 4
            or self.minimum_positive_median_ic_fold_count != 4
            or self.minimum_positive_date_fraction != 0.50
            or self.minimum_positive_date_fraction_fold_count != 4
            or self.minimum_positive_spread_fold_count != 4
            or self.bootstrap_replicates != 10_000
            or self.bootstrap_primary_block_sessions != 21
            or self.bootstrap_sensitivity_block_sessions != (10, 30)
            or self.minimum_break_even_one_way_cost_basis_points != 10.0
            or self.minimum_median_projection_retention != 0.50
            or self.minimum_fold_projection_retention != 0.20
            or self.economic_optimizer_updates != 0
            or self.v9_or_v10_state_reuse_authorized
            or self.outer_2026_access_authorized
            or not self.future_selected_universe
            or not self.development_only
            or self.reportable
            or self.promotable
        ):
            raise M03RV11ProtocolError("v11 predictive specification drifted")


M03R_V11_PREDICTIVE_SPEC = M03RV11PredictiveSpec()
M03R_V11_PROTOCOL_SHA256 = _sha256(
    {
        "generation": M03R_V11_PROTOCOL_GENERATION,
        "design_id": M03R_V11_DESIGN_ID,
        "horizons": M03R_V11_HORIZONS,
        "eligible_execution_horizons": M03R_V11_ELIGIBLE_EXECUTION_HORIZONS,
        "settings": tuple(asdict(row) for row in M03R_V11_SETTINGS),
        "spec": asdict(M03R_V11_PREDICTIVE_SPEC),
        "episode_schedule_rule": M03R_V11_EPISODE_SCHEDULE_RULE,
        "residual_operator_rule": M03R_V11_RESIDUAL_OPERATOR_RULE,
        "bootstrap_rule": M03R_V11_BOOTSTRAP_RULE,
        "superseded_unlaunched_protocol_sha256": M03R_V10_PROTOCOL_SHA256,
    }
)


def resolve_m03r_v11_setting(value: int | str) -> M03RV11PredictiveSetting:
    if isinstance(value, bool):
        raise M03RV11ProtocolError("boolean is not a v11 setting identity")
    for row in M03R_V11_SETTINGS:
        if value in {row.setting_index, row.setting_id}:
            return row
    raise M03RV11ProtocolError(f"unknown v11 setting: {value!r}")


__all__ = [
    "M03R_V11_BOOTSTRAP_RULE",
    "M03R_V11_DESIGN_ID",
    "M03R_V11_ELIGIBLE_EXECUTION_HORIZONS",
    "M03R_V11_EPISODE_SCHEDULE_RULE",
    "M03R_V11_HORIZONS",
    "M03R_V11_PREDICTIVE_SPEC",
    "M03R_V11_PROTOCOL_GENERATION",
    "M03R_V11_PROTOCOL_SHA256",
    "M03R_V11_RESIDUAL_OPERATOR_RULE",
    "M03R_V11_SETTINGS",
    "M03R_V11_SETTING_IDS",
    "M03RV11PredictiveSetting",
    "M03RV11ProtocolError",
    "resolve_m03r_v11_setting",
]
