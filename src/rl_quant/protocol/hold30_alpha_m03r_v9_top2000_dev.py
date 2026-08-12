"""Frozen predictive-development contract for TOP2000 M03R-v9.

V9 is a source-homogeneous successor to the failed V8 predictive panel.  It
does not authorize economic policy optimization.  The only permitted next
stage is a three-setting, six-fold predictive study whose selected alpha
horizon, factor-residual target, predictive distribution, and deterministic
simple-sleeve evidence agree exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

M03R_V9_PROTOCOL_GENERATION = "top2000-dev-hold30-active-alpha-m03r-v9-predictive-v1"
M03R_V9_DESIGN_ID = "daily-ohlcv-top2000-dev-m03r-v9-factor-residual-predictive-v1"
M03R_V9_SCHEMA_VERSION = 1
M03R_V9_SETTING_IDS = (
    "V9-P0-factor-residual-ranked",
    "V9-P1-factor-residual-no-ranking",
    "V9-P2-benchmark-relative-ranked",
)
M03R_V9_HORIZONS = (5, 21, 30, 63)
M03R_V9_ELIGIBLE_EXECUTION_HORIZONS = (21, 30)
M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES = (
    "sector",
    "active-beta",
    "style-risk",
)


class M03RV9ProtocolError(ValueError):
    """The frozen V9 predictive contract or one of its identities drifted."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M03RV9ProtocolError("V9 contract is not canonical-JSON safe") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV9PredictiveSpec:
    """One-shot predictive training and tradeability qualification."""

    horizons_trading_sessions: tuple[int, ...] = M03R_V9_HORIZONS
    horizon_loss_weights: tuple[float, ...] = (0.10, 0.35, 0.40, 0.15)
    ranked_component_weights: tuple[float, float, float] = (0.50, 0.30, 0.20)
    no_ranking_component_weights: tuple[float, float, float] = (0.0, 0.60, 0.40)
    listwise_horizon_scale_rule: str = "cumulative-return-divided-by-0.02-sqrt-h-v1"
    target_ridge_lambda: float = 1.0e-6
    required_risk_exposure_families: tuple[str, ...] = (
        M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES
    )
    risk_exposures_available_at_decision_origin: bool = True
    risk_asset_axis_must_match_sequence: bool = True
    maximum_optimizer_updates: int = 64
    early_stopping_enabled: bool = False
    checkpoint_selection_update: int = 64
    qualification_evaluation_updates: tuple[int, ...] = (64,)
    minimum_mean_spearman_rank_ic: float = 0.020
    minimum_positive_rank_ic_fold_count: int = 4
    minimum_mean_top_bottom_spread: float = 0.0
    minimum_positive_spread_fold_count: int = 4
    minimum_simple_sleeve_gross_active_return: float = 0.0
    minimum_simple_sleeve_net_active_return_10bp: float = 0.0
    minimum_gross_positive_fold_count: int = 4
    minimum_break_even_one_way_cost_basis_points: float = 10.0
    chronological_fold_count: int = 6
    seed: int = 17
    h100s_per_worker: int = 2
    maximum_workers: int = 3
    maximum_h100_requests: int = 6
    economic_optimizer_updates: int = 0
    outer_lockbox_access_authorized: bool = False
    future_selected_universe: bool = True
    development_only: bool = True
    reportable: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        if (
            self.horizons_trading_sessions != M03R_V9_HORIZONS
            or self.horizon_loss_weights != (0.10, 0.35, 0.40, 0.15)
            or self.ranked_component_weights != (0.50, 0.30, 0.20)
            or self.no_ranking_component_weights != (0.0, 0.60, 0.40)
            or self.required_risk_exposure_families
            != M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES
            or not self.risk_exposures_available_at_decision_origin
            or not self.risk_asset_axis_must_match_sequence
            or abs(sum(self.ranked_component_weights) - 1.0) > 1.0e-12
            or abs(sum(self.no_ranking_component_weights) - 1.0) > 1.0e-12
            or self.maximum_optimizer_updates != 64
            or self.early_stopping_enabled
            or self.checkpoint_selection_update != 64
            or self.qualification_evaluation_updates != (64,)
            or self.minimum_mean_spearman_rank_ic != 0.020
            or self.minimum_positive_rank_ic_fold_count != 4
            or self.minimum_positive_spread_fold_count != 4
            or self.minimum_gross_positive_fold_count != 4
            or self.minimum_break_even_one_way_cost_basis_points != 10.0
            or self.chronological_fold_count != 6
            or self.seed != 17
            or self.h100s_per_worker != 2
            or self.maximum_workers != 3
            or self.maximum_h100_requests != 6
            or self.economic_optimizer_updates != 0
            or self.outer_lockbox_access_authorized
            or not self.future_selected_universe
            or not self.development_only
            or self.reportable
            or self.promotable
        ):
            raise M03RV9ProtocolError("V9 predictive specification drifted")


M03R_V9_PREDICTIVE_SPEC = M03RV9PredictiveSpec()
M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256 = _sha256(
    {
        "schema": "rl-quant.top2000-dev.m03r-v9-alpha-distribution-v1",
        "horizons": M03R_V9_HORIZONS,
        "eligible_execution_horizons": M03R_V9_ELIGIBLE_EXECUTION_HORIZONS,
        "fields": (
            "mean_by_horizon",
            "log_scale_by_horizon",
            "selected_horizon_sessions",
            "selected_mean",
            "selected_scale",
        ),
        "old_alpha_downside_30d_forbidden": True,
    }
)


@dataclass(frozen=True, slots=True)
class M03RV9PredictiveSetting:
    setting_index: int
    setting_id: str
    target_mode: Literal["factor-residual", "benchmark-relative"]
    ranking_enabled: bool
    factor_exposure_names_must_match_projector: bool
    actor_inputs_are_raw_ohlcv_only: bool = True
    factors_are_actor_inputs: bool = False
    development_only: bool = True
    reportable: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.setting_index, bool)
            or not isinstance(self.setting_index, int)
            or not 0 <= self.setting_index < len(M03R_V9_SETTING_IDS)
        ):
            raise M03RV9ProtocolError("V9 predictive setting index drifted")
        expected = (
            (0, M03R_V9_SETTING_IDS[0], "factor-residual", True, True),
            (1, M03R_V9_SETTING_IDS[1], "factor-residual", False, True),
            (2, M03R_V9_SETTING_IDS[2], "benchmark-relative", True, False),
        )
        observed = (
            self.setting_index,
            self.setting_id,
            self.target_mode,
            self.ranking_enabled,
            self.factor_exposure_names_must_match_projector,
        )
        if (
            observed != expected[self.setting_index]
            or not self.actor_inputs_are_raw_ohlcv_only
            or self.factors_are_actor_inputs
            or not self.development_only
            or self.reportable
            or self.promotable
        ):
            raise M03RV9ProtocolError("V9 predictive setting drifted")

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


M03R_V9_SETTINGS = (
    M03RV9PredictiveSetting(0, M03R_V9_SETTING_IDS[0], "factor-residual", True, True),
    M03RV9PredictiveSetting(1, M03R_V9_SETTING_IDS[1], "factor-residual", False, True),
    M03RV9PredictiveSetting(
        2, M03R_V9_SETTING_IDS[2], "benchmark-relative", True, False
    ),
)


def resolve_m03r_v9_setting(value: int | str) -> M03RV9PredictiveSetting:
    if isinstance(value, bool):
        raise M03RV9ProtocolError("boolean is not a V9 setting identity")
    for row in M03R_V9_SETTINGS:
        if value == row.setting_index or value == row.setting_id:
            return row
    raise M03RV9ProtocolError(f"unknown V9 predictive setting: {value!r}")


@dataclass(frozen=True, slots=True)
class M03RV9HorizonBinding:
    """One eligible horizon shared by selection, qualification, and execution."""

    checkpoint_selection_horizon: int
    qualification_horizon: int
    economic_execution_horizon: int

    def __post_init__(self) -> None:
        if (
            self.checkpoint_selection_horizon != self.qualification_horizon
            or self.qualification_horizon != self.economic_execution_horizon
            or self.economic_execution_horizon
            not in M03R_V9_ELIGIBLE_EXECUTION_HORIZONS
        ):
            raise M03RV9ProtocolError(
                "checkpoint, qualification, and execution horizons must be identical"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


M03R_V9_PROTOCOL_SHA256 = _sha256(
    {
        "generation": M03R_V9_PROTOCOL_GENERATION,
        "design_id": M03R_V9_DESIGN_ID,
        "schema_version": M03R_V9_SCHEMA_VERSION,
        "predictive_spec": asdict(M03R_V9_PREDICTIVE_SPEC),
        "settings": tuple(asdict(row) for row in M03R_V9_SETTINGS),
        "alpha_distribution_contract_sha256": (
            M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256
        ),
    }
)


__all__ = [
    "M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256",
    "M03R_V9_DESIGN_ID",
    "M03R_V9_ELIGIBLE_EXECUTION_HORIZONS",
    "M03R_V9_HORIZONS",
    "M03R_V9_PREDICTIVE_SPEC",
    "M03R_V9_PROTOCOL_GENERATION",
    "M03R_V9_PROTOCOL_SHA256",
    "M03R_V9_REQUIRED_RISK_EXPOSURE_FAMILIES",
    "M03R_V9_SETTINGS",
    "M03R_V9_SETTING_IDS",
    "M03RV9HorizonBinding",
    "M03RV9PredictiveSetting",
    "M03RV9PredictiveSpec",
    "M03RV9ProtocolError",
    "resolve_m03r_v9_setting",
]
