"""Frozen development contract for the M03R-v8 alpha-discovery panel.

This generation responds only to the completed TOP2000 seed-17 Phase-0
findings.  It is disjoint from canonical point-in-time M03R-v7 and cannot
create reportable or promotable evidence.  The central change is structural:
prediction is qualified before economic fine-tuning, and confidence bounds a
cost-aware incremental move from a learned-exit anchor instead of changing the
temperature of a full replacement book.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v8-discovery-v1"
)
M03R_V8_TOP2000_DEV_DESIGN_ID = (
    "daily-ohlcv-top2000-dev-m03r-v8-pretrained-incremental-costgate-v1"
)
M03R_V8_TOP2000_DEV_SCHEMA_VERSION = 1
M03R_V8_TOP2000_DEV_REFERENCE_SETTING_ID = (
    "V8-0-pretrained-alpha-costgate-top2000-dev-v1"
)
M03R_V8_TOP2000_DEV_SETTING_IDS = (
    M03R_V8_TOP2000_DEV_REFERENCE_SETTING_ID,
    "V8-1-no-alpha-pretraining-top2000-dev-v1",
    "V8-2-no-ranking-loss-top2000-dev-v1",
    "V8-3-softer-exact-hold-top2000-dev-v1",
    "V8-4-no-cost-gate-top2000-dev-v1",
    "V8-5-strong-cost-gate-top2000-dev-v1",
    "V8-6-fixed-exit-hazard-top2000-dev-v1",
    "V8-7-relaxed-factor-bounds-top2000-dev-v1",
)
M03R_V8_TOP2000_DEV_CAUSAL_FIELDS = (
    "alpha_pretraining_mode",
    "ranking_loss_weight",
    "exact_hold_action_temperature",
    "cost_gate_mode",
    "exit_hazard_mode",
    "factor_sector_bound_multiplier",
)


class M03RV8Top2000DevProtocolError(ValueError):
    """The v8 development identity or a causal row drifted."""


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
        raise M03RV8Top2000DevProtocolError(
            "v8 development payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV8AlphaPretrainingSpec:
    """Training-fold-only raw-encoder and residual-alpha qualification."""

    horizons_trading_sessions: tuple[int, ...] = (5, 21, 30, 63)
    horizon_loss_weights: tuple[float, ...] = (0.10, 0.35, 0.40, 0.15)
    ranking_loss_weight: float = 0.50
    huber_loss_weight: float = 0.30
    distributional_loss_weight: float = 0.20
    early_stop_horizons_trading_sessions: tuple[int, ...] = (21, 30)
    minimum_mean_spearman_rank_ic: float = 0.02
    minimum_positive_rank_ic_fold_count: int = 4
    chronological_fold_count: int = 6
    training_fold_only: bool = True
    inner_validation_only_for_early_stop_and_calibration: bool = True
    benchmark_relative_or_factor_residual_targets: bool = True
    factors_are_actor_inputs: bool = False
    raw_ohlcv_only_actor_inputs: bool = True
    encoder_learning_rate_below_policy_head_learning_rate: bool = True
    encoder_learning_rate: float = 2.0e-5
    prediction_head_learning_rate: float = 1.0e-4
    adamw_weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    maximum_optimizer_updates: int = 64
    inner_validation_interval_updates: int = 4
    minimum_optimizer_updates_before_early_stop: int = 16
    early_stopping_patience_evaluations: int = 4
    checkpoint_selection_rule: str = (
        "max-of-21d-30d-date-balanced-mean-spearman-ic-tie-earliest-v1"
    )

    def __post_init__(self) -> None:
        if asdict(self) != {
            "horizons_trading_sessions": (5, 21, 30, 63),
            "horizon_loss_weights": (0.10, 0.35, 0.40, 0.15),
            "ranking_loss_weight": 0.50,
            "huber_loss_weight": 0.30,
            "distributional_loss_weight": 0.20,
            "early_stop_horizons_trading_sessions": (21, 30),
            "minimum_mean_spearman_rank_ic": 0.02,
            "minimum_positive_rank_ic_fold_count": 4,
            "chronological_fold_count": 6,
            "training_fold_only": True,
            "inner_validation_only_for_early_stop_and_calibration": True,
            "benchmark_relative_or_factor_residual_targets": True,
            "factors_are_actor_inputs": False,
            "raw_ohlcv_only_actor_inputs": True,
            "encoder_learning_rate_below_policy_head_learning_rate": True,
            "encoder_learning_rate": 2.0e-5,
            "prediction_head_learning_rate": 1.0e-4,
            "adamw_weight_decay": 1.0e-4,
            "gradient_clip_norm": 1.0,
            "maximum_optimizer_updates": 64,
            "inner_validation_interval_updates": 4,
            "minimum_optimizer_updates_before_early_stop": 16,
            "early_stopping_patience_evaluations": 4,
            "checkpoint_selection_rule": (
                "max-of-21d-30d-date-balanced-mean-spearman-ic-tie-earliest-v1"
            ),
        }:
            raise M03RV8Top2000DevProtocolError("v8 alpha-pretraining contract drifted")


@dataclass(frozen=True, slots=True)
class M03RV8ActivePolicySpec:
    """Shared incremental action and risk-control contract."""

    recent_raw_context_trading_sessions: int = 42
    learned_temporal_context_trading_sessions: int = 252
    persistence_coefficient_basis_points: float = 2.0
    persistence_preference_horizon_sessions: int = 30
    persistence_is_soft_and_one_sided: bool = True
    exact_hold_action_available: bool = True
    reference_exact_hold_action_temperature: float = 1.0
    softened_exact_hold_action_temperature: float = 1.5
    policy_operates_on_incremental_active_weights: bool = True
    learned_exit_hazard_precedes_cost_gate: bool = True
    confidence_controls_new_and_expanding_risk_only: bool = True
    confidence_does_not_suppress_learned_exits: bool = True
    maximum_incremental_one_way_turnover: float = 0.02
    entry_hurdle_multiplier: float = 1.0
    retention_hurdle_multiplier: float = 0.5
    uncertainty_multiplier: float = 1.0
    disabled_cost_gate_hurdle_multiplier: float = 0.0
    strong_cost_gate_cost_multiplier: float = 1.5
    strong_cost_gate_uncertainty_multiplier: float = 1.5
    training_one_way_cost_basis_points: int = 20
    evaluation_one_way_cost_basis_points: tuple[int, ...] = (0, 10, 20, 40)
    annual_tracking_error_floor: float | None = None
    annual_tracking_error_ceiling: float = 0.06
    active_beta_equivalence_absolute_upper_bound: float = 0.10
    maximum_stock_weight_fraction: float = 0.01
    factor_sector_projection_required: bool = True
    nonzero_content_bound_factor_sector_slabs_required: bool = True
    projection_mode: str = "benchmark-radial-factor-beta-te-v1"
    relaxed_factor_sector_bound_multiplier: float = 1.5
    benchmark_anchoring_required: bool = True

    def __post_init__(self) -> None:
        if asdict(self) != {
            "recent_raw_context_trading_sessions": 42,
            "learned_temporal_context_trading_sessions": 252,
            "persistence_coefficient_basis_points": 2.0,
            "persistence_preference_horizon_sessions": 30,
            "persistence_is_soft_and_one_sided": True,
            "exact_hold_action_available": True,
            "reference_exact_hold_action_temperature": 1.0,
            "softened_exact_hold_action_temperature": 1.5,
            "policy_operates_on_incremental_active_weights": True,
            "learned_exit_hazard_precedes_cost_gate": True,
            "confidence_controls_new_and_expanding_risk_only": True,
            "confidence_does_not_suppress_learned_exits": True,
            "maximum_incremental_one_way_turnover": 0.02,
            "entry_hurdle_multiplier": 1.0,
            "retention_hurdle_multiplier": 0.5,
            "uncertainty_multiplier": 1.0,
            "disabled_cost_gate_hurdle_multiplier": 0.0,
            "strong_cost_gate_cost_multiplier": 1.5,
            "strong_cost_gate_uncertainty_multiplier": 1.5,
            "training_one_way_cost_basis_points": 20,
            "evaluation_one_way_cost_basis_points": (0, 10, 20, 40),
            "annual_tracking_error_floor": None,
            "annual_tracking_error_ceiling": 0.06,
            "active_beta_equivalence_absolute_upper_bound": 0.10,
            "maximum_stock_weight_fraction": 0.01,
            "factor_sector_projection_required": True,
            "nonzero_content_bound_factor_sector_slabs_required": True,
            "projection_mode": "benchmark-radial-factor-beta-te-v1",
            "relaxed_factor_sector_bound_multiplier": 1.5,
            "benchmark_anchoring_required": True,
        }:
            raise M03RV8Top2000DevProtocolError(
                "v8 incremental active-policy contract drifted"
            )


M03R_V8_ALPHA_PRETRAINING = M03RV8AlphaPretrainingSpec()
M03R_V8_ACTIVE_POLICY = M03RV8ActivePolicySpec()


@dataclass(frozen=True, slots=True)
class M03RV8Top2000DevSetting:
    """One high-information row in the eight-setting discovery panel."""

    setting_index: int
    setting_id: str
    alpha_pretraining_mode: str
    ranking_loss_weight: float
    exact_hold_action_temperature: float
    cost_gate_mode: str
    exit_hazard_mode: str
    factor_sector_bound_multiplier: float
    ablation_of: str | None
    declared_causal_field: str | None
    shared_alpha_pretraining_sha256: str
    shared_active_policy_sha256: str
    development_only: bool = True
    future_selected_universe: bool = True
    reportable: bool = False
    promotable: bool = False
    outer_lockbox_evaluation_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            not 0 <= self.setting_index < len(M03R_V8_TOP2000_DEV_SETTING_IDS)
            or self.setting_id != M03R_V8_TOP2000_DEV_SETTING_IDS[self.setting_index]
            or "top2000-dev" not in self.setting_id
            or self.shared_alpha_pretraining_sha256
            != _sha256(asdict(M03R_V8_ALPHA_PRETRAINING))
            or self.shared_active_policy_sha256
            != _sha256(asdict(M03R_V8_ACTIVE_POLICY))
        ):
            raise M03RV8Top2000DevProtocolError(
                "v8 setting identity or shared-contract binding drifted"
            )
        if self.alpha_pretraining_mode not in {
            "training-fold-pretrained",
            "joint-random-initialization",
        }:
            raise M03RV8Top2000DevProtocolError("unknown v8 pretraining mode")
        if self.ranking_loss_weight not in {0.0, 0.50}:
            raise M03RV8Top2000DevProtocolError("unknown v8 ranking-loss weight")
        if self.exact_hold_action_temperature not in {
            M03R_V8_ACTIVE_POLICY.reference_exact_hold_action_temperature,
            M03R_V8_ACTIVE_POLICY.softened_exact_hold_action_temperature,
        }:
            raise M03RV8Top2000DevProtocolError(
                "unknown v8 exact-HOLD action temperature"
            )
        if self.cost_gate_mode not in {"reference", "disabled", "strong"}:
            raise M03RV8Top2000DevProtocolError("unknown v8 cost-gate mode")
        if self.exit_hazard_mode not in {
            "learned",
            "fixed-structural-30-session-prior",
        }:
            raise M03RV8Top2000DevProtocolError("unknown v8 exit-hazard mode")
        if self.factor_sector_bound_multiplier not in {1.0, 1.5}:
            raise M03RV8Top2000DevProtocolError(
                "v8 factor bounds may be canonical or relaxed 1.5x"
            )
        if self.setting_index == 0:
            if self.ablation_of is not None or self.declared_causal_field is not None:
                raise M03RV8Top2000DevProtocolError(
                    "the v8 reference row is not an ablation"
                )
        elif (
            self.ablation_of != M03R_V8_TOP2000_DEV_REFERENCE_SETTING_ID
            or self.declared_causal_field not in M03R_V8_TOP2000_DEV_CAUSAL_FIELDS
        ):
            raise M03RV8Top2000DevProtocolError(
                "v8 ablation lineage or causal field drifted"
            )
        if (
            not self.development_only
            or not self.future_selected_universe
            or self.reportable
            or self.promotable
            or self.outer_lockbox_evaluation_authorized
        ):
            raise M03RV8Top2000DevProtocolError(
                "v8 TOP2000 discovery rows must remain nonreportable"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


_ALPHA_SPEC_SHA256 = _sha256(asdict(M03R_V8_ALPHA_PRETRAINING))
_ACTIVE_POLICY_SHA256 = _sha256(asdict(M03R_V8_ACTIVE_POLICY))

_REFERENCE = M03RV8Top2000DevSetting(
    setting_index=0,
    setting_id=M03R_V8_TOP2000_DEV_REFERENCE_SETTING_ID,
    alpha_pretraining_mode="training-fold-pretrained",
    ranking_loss_weight=0.50,
    exact_hold_action_temperature=(
        M03R_V8_ACTIVE_POLICY.reference_exact_hold_action_temperature
    ),
    cost_gate_mode="reference",
    exit_hazard_mode="learned",
    factor_sector_bound_multiplier=1.0,
    ablation_of=None,
    declared_causal_field=None,
    shared_alpha_pretraining_sha256=_ALPHA_SPEC_SHA256,
    shared_active_policy_sha256=_ACTIVE_POLICY_SHA256,
)


def _ablation(
    setting_index: int,
    causal_field: str,
    **change: Any,
) -> M03RV8Top2000DevSetting:
    if set(change) != {causal_field}:
        raise M03RV8Top2000DevProtocolError(
            "each v8 discovery row must change exactly its declared causal field"
        )
    return replace(
        _REFERENCE,
        setting_index=setting_index,
        setting_id=M03R_V8_TOP2000_DEV_SETTING_IDS[setting_index],
        ablation_of=M03R_V8_TOP2000_DEV_REFERENCE_SETTING_ID,
        declared_causal_field=causal_field,
        **change,
    )


M03R_V8_TOP2000_DEV_SETTINGS = (
    _REFERENCE,
    _ablation(
        1,
        "alpha_pretraining_mode",
        alpha_pretraining_mode="joint-random-initialization",
    ),
    _ablation(2, "ranking_loss_weight", ranking_loss_weight=0.0),
    _ablation(
        3,
        "exact_hold_action_temperature",
        exact_hold_action_temperature=(
            M03R_V8_ACTIVE_POLICY.softened_exact_hold_action_temperature
        ),
    ),
    _ablation(4, "cost_gate_mode", cost_gate_mode="disabled"),
    _ablation(5, "cost_gate_mode", cost_gate_mode="strong"),
    _ablation(
        6,
        "exit_hazard_mode",
        exit_hazard_mode="fixed-structural-30-session-prior",
    ),
    _ablation(
        7,
        "factor_sector_bound_multiplier",
        factor_sector_bound_multiplier=1.5,
    ),
)
M03R_V8_TOP2000_DEV_SETTINGS_BY_ID = {
    row.setting_id: row for row in M03R_V8_TOP2000_DEV_SETTINGS
}


def m03r_v8_top2000_dev_protocol_payload() -> dict[str, Any]:
    payload = {
        "schema_version": M03R_V8_TOP2000_DEV_SCHEMA_VERSION,
        "protocol_generation": M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION,
        "design_id": M03R_V8_TOP2000_DEV_DESIGN_ID,
        "alpha_pretraining": asdict(M03R_V8_ALPHA_PRETRAINING),
        "active_policy": asdict(M03R_V8_ACTIVE_POLICY),
        "settings": [asdict(row) for row in M03R_V8_TOP2000_DEV_SETTINGS],
        "setting_receipt_sha256": [
            row.receipt_sha256 for row in M03R_V8_TOP2000_DEV_SETTINGS
        ],
        "chronological_fold_count": 6,
        "discovery_seed": 17,
        "initial_optimizer_update_count": 64,
        "conditional_extension_optimizer_update_count": 128,
        "automatic_update_extension_forbidden": True,
        "phase0_distinct_policy_gate_must_pass_before_remote_training": True,
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    return {**payload, "receipt_sha256": _sha256(payload)}


M03R_V8_TOP2000_DEV_PROTOCOL = m03r_v8_top2000_dev_protocol_payload()
M03R_V8_TOP2000_DEV_PROTOCOL_SHA256 = M03R_V8_TOP2000_DEV_PROTOCOL["receipt_sha256"]


def resolve_m03r_v8_top2000_dev_setting(
    setting: int | str,
) -> M03RV8Top2000DevSetting:
    if isinstance(setting, bool):
        raise M03RV8Top2000DevProtocolError("boolean is not a v8 setting index")
    if isinstance(setting, int):
        try:
            return M03R_V8_TOP2000_DEV_SETTINGS[setting]
        except IndexError as exc:
            raise M03RV8Top2000DevProtocolError("unknown v8 setting index") from exc
    if isinstance(setting, str):
        try:
            return M03R_V8_TOP2000_DEV_SETTINGS_BY_ID[setting]
        except KeyError as exc:
            raise M03RV8Top2000DevProtocolError("unknown v8 setting ID") from exc
    raise M03RV8Top2000DevProtocolError("v8 setting must be an index or ID")


__all__ = [
    "M03R_V8_ACTIVE_POLICY",
    "M03R_V8_ALPHA_PRETRAINING",
    "M03R_V8_TOP2000_DEV_CAUSAL_FIELDS",
    "M03R_V8_TOP2000_DEV_DESIGN_ID",
    "M03R_V8_TOP2000_DEV_PROTOCOL",
    "M03R_V8_TOP2000_DEV_PROTOCOL_GENERATION",
    "M03R_V8_TOP2000_DEV_PROTOCOL_SHA256",
    "M03R_V8_TOP2000_DEV_REFERENCE_SETTING_ID",
    "M03R_V8_TOP2000_DEV_SETTINGS",
    "M03R_V8_TOP2000_DEV_SETTINGS_BY_ID",
    "M03R_V8_TOP2000_DEV_SETTING_IDS",
    "M03RV8ActivePolicySpec",
    "M03RV8AlphaPretrainingSpec",
    "M03RV8Top2000DevProtocolError",
    "M03RV8Top2000DevSetting",
    "m03r_v8_top2000_dev_protocol_payload",
    "resolve_m03r_v8_top2000_dev_setting",
]
