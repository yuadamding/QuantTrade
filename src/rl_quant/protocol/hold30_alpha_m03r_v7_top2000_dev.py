"""Development-only TOP2000 compatibility identity for the reviewed M03R v7 panel.

This protocol does not alter, alias, or supersede the canonical point-in-time
Active-300 v7 protocol.  It mirrors the twelve reviewed causal rows under a
disjoint ``top2000-dev`` identity so future-selected TOP2000 cache experiments
cannot be mistaken for promotable or scientifically reportable evidence.

The cache contract is intentionally unbound here.  A later, content-addressed
cache binding may qualify data access, but it cannot change the development-only
or non-reportable status frozen by this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_DESIGN_ID,
    M03R_V7_PRIMARY_SETTINGS,
    M03R_V7_PRIMARY_SETTINGS_BY_ID,
    M03R_V7_PROTOCOL_GENERATION,
    M03RV7Setting,
)

M03R_TOP2000_DEV_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v7-compat-v1"
)
M03R_TOP2000_DEV_DESIGN_ID = (
    "daily_ohlcv_aggregated_top2000_dev_hold30_m03r_v7_compat_v1"
)
M03R_TOP2000_DEV_SCHEMA_VERSION = 1
M03R_TOP2000_DEV_UNIVERSE_ID = "future-selected-top2000-development-only"
M03R_TOP2000_DEV_DATA_ROLE = "development-only-nonreportable"
M03R_TOP2000_DEV_CACHE_CONTRACT_SCHEMA = (
    "rl-quant.top2000-dev-cache-contract-unbound-v1"
)
M03R_TOP2000_DEV_RAW_SIGMOID_ACTIVE_RISK_BUDGET_MODE = (
    "uncalibrated-sigmoid-0-to-4pct-development-only"
)
M03R_TOP2000_DEV_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE = (
    "fixed-2pct-development-only"
)
M03R_TOP2000_DEV_REFERENCE_SETTING_ID = (
    "M03R-soft-persistence-active-alpha-hold30-top2000-dev-v1"
)

M03R_TOP2000_DEV_SETTING_IDS = (
    M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
    "P00-no-soft-persistence-top2000-dev-v1",
    "P10-soft-persistence-10bp-top2000-dev-v1",
    "A08-fixed-exit-hazard-top2000-dev-v1",
    "A11-no-exact-hold-atom-top2000-dev-v1",
    "A09-no-long-context-top2000-dev-v1",
    "M02-active-risk-no-alpha-heads-top2000-dev-v1",
    "A04-no-downside-score-adjustment-top2000-dev-v1",
    "A12-fixed-2pct-active-risk-budget-top2000-dev-v1",
    "A10-no-factor-neutral-projection-top2000-dev-v1",
    "A06-sharpe-overlay-top2000-dev-v1",
    "A07-direct-sharpe-top2000-dev-v1",
)


class M03RTop2000DevProtocolError(ValueError):
    """A TOP2000 development compatibility identity is inconsistent."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M03RTop2000DevProtocolError(
            "TOP2000 development protocol payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RTop2000DevGeometry:
    """Frozen compatibility geometry; holding preference is not a hard rule."""

    input_observation_contract: str = "daily-ohlcv-aggregated-from-300s-source"
    daily_ohlcv_feature_count: int = 5
    recent_daily_ohlcv_context_trading_sessions: int = 42
    canonical_learned_temporal_context_trading_sessions: int = 252
    lightweight_daily_ohlcv_token_for_all_context_sessions: bool = True
    rollout_trading_sessions: int = 63
    economic_credit_post_fill_return_count: int = 30
    maximum_auxiliary_label_horizon_return_count: int = 63
    training_replay_state_rows: int = 378
    training_observation_warmup_decisions: int = 251
    training_loss_bearing_origin_count: int = 63
    validation_score_transition_count: int = 63
    age_state_bin_count: int = 61
    minimum_age_sessions: int = 0
    maximum_age_sessions: int = 60

    def __post_init__(self) -> None:
        if asdict(self) != {
            "input_observation_contract": "daily-ohlcv-aggregated-from-300s-source",
            "daily_ohlcv_feature_count": 5,
            "recent_daily_ohlcv_context_trading_sessions": 42,
            "canonical_learned_temporal_context_trading_sessions": 252,
            "lightweight_daily_ohlcv_token_for_all_context_sessions": True,
            "rollout_trading_sessions": 63,
            "economic_credit_post_fill_return_count": 30,
            "maximum_auxiliary_label_horizon_return_count": 63,
            "training_replay_state_rows": 378,
            "training_observation_warmup_decisions": 251,
            "training_loss_bearing_origin_count": 63,
            "validation_score_transition_count": 63,
            "age_state_bin_count": 61,
            "minimum_age_sessions": 0,
            "maximum_age_sessions": 60,
        }:
            raise M03RTop2000DevProtocolError(
                "TOP2000 development geometry must remain 42/252/63/30 with 61 age bins"
            )


@dataclass(frozen=True, slots=True)
class M03RTop2000DevCacheRequirement:
    """Unbound cache requirement that may be satisfied by a later artifact."""

    schema: str = M03R_TOP2000_DEV_CACHE_CONTRACT_SCHEMA
    expected_daily_ohlcv_tensor_shape: tuple[int, int, int] = (1001, 1999, 5)
    cache_contract_required: bool = True
    cache_contract_bound: bool = False
    cache_contract_sha256: str | None = None
    cache_manifest_sha256: str | None = None
    later_content_addressed_binding_allowed: bool = True

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_TOP2000_DEV_CACHE_CONTRACT_SCHEMA
            or self.expected_daily_ohlcv_tensor_shape != (1001, 1999, 5)
            or not self.cache_contract_required
            or self.cache_contract_bound
            or self.cache_contract_sha256 is not None
            or self.cache_manifest_sha256 is not None
            or not self.later_content_addressed_binding_allowed
        ):
            raise M03RTop2000DevProtocolError(
                "the compatibility cache contract must remain explicitly unbound"
            )


M03R_TOP2000_DEV_GEOMETRY = M03RTop2000DevGeometry()
M03R_TOP2000_DEV_CACHE_REQUIREMENT = M03RTop2000DevCacheRequirement()


@dataclass(frozen=True, slots=True)
class M03RTop2000DevSetting:
    """One disjoint development row bound to one reviewed canonical-v7 row."""

    setting_index: int
    setting_id: str
    reviewed_v7_setting_id: str
    persistence_coefficient_basis_points: float
    exit_hazard_mode: str
    exact_hold_action_supported: bool
    learned_temporal_context_trading_sessions: int
    residual_alpha_head_mode: str
    active_risk_budget_mode: str
    development_active_risk_budget_execution_mode: str
    factor_sector_neutral_projection: bool
    sharpe_mode: str
    ablation_of: str | None
    declared_causal_field: str | None
    development_only: bool = True
    promotion_eligible: bool = False
    scientific_reporting_eligible: bool = False
    outer_lockbox_evaluation_eligible: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.setting_index < len(M03R_TOP2000_DEV_SETTING_IDS):
            raise M03RTop2000DevProtocolError("invalid TOP2000 development index")
        if self.setting_id != M03R_TOP2000_DEV_SETTING_IDS[self.setting_index]:
            raise M03RTop2000DevProtocolError(
                "TOP2000 development setting ID/index drifted"
            )
        if "top2000-dev" not in self.setting_id:
            raise M03RTop2000DevProtocolError(
                "every compatibility setting ID must visibly contain top2000-dev"
            )
        try:
            reviewed = M03R_V7_PRIMARY_SETTINGS_BY_ID[self.reviewed_v7_setting_id]
        except KeyError as exc:
            raise M03RTop2000DevProtocolError(
                "compatibility row must bind an exact reviewed v7 setting"
            ) from exc
        if reviewed.setting_index != self.setting_index:
            raise M03RTop2000DevProtocolError(
                "TOP2000 development and reviewed-v7 setting order must match"
            )
        expected_semantics = (
            reviewed.persistence_coefficient_basis_points,
            reviewed.exit_hazard_mode,
            reviewed.exact_hold_action_supported,
            reviewed.learned_temporal_context_trading_sessions,
            reviewed.residual_alpha_head_mode,
            reviewed.active_risk_budget_mode,
            reviewed.factor_sector_neutral_projection,
            reviewed.sharpe_mode,
        )
        observed_semantics = (
            self.persistence_coefficient_basis_points,
            self.exit_hazard_mode,
            self.exact_hold_action_supported,
            self.learned_temporal_context_trading_sessions,
            self.residual_alpha_head_mode,
            self.active_risk_budget_mode,
            self.factor_sector_neutral_projection,
            self.sharpe_mode,
        )
        if observed_semantics != expected_semantics:
            raise M03RTop2000DevProtocolError(
                "TOP2000 development row changed reviewed v7 scientific semantics"
            )
        expected_development_budget_mode = (
            M03R_TOP2000_DEV_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE
            if self.setting_index == 8
            else M03R_TOP2000_DEV_RAW_SIGMOID_ACTIVE_RISK_BUDGET_MODE
        )
        if (
            self.development_active_risk_budget_execution_mode
            != expected_development_budget_mode
        ):
            raise M03RTop2000DevProtocolError(
                "TOP2000 development risk sizing must truthfully bind raw-sigmoid or fixed-2% execution"
            )
        if self.setting_index == 0:
            if self.ablation_of is not None or self.declared_causal_field is not None:
                raise M03RTop2000DevProtocolError(
                    "the TOP2000 development reference row is not an ablation"
                )
        elif (
            self.ablation_of != M03R_TOP2000_DEV_REFERENCE_SETTING_ID
            or self.declared_causal_field != reviewed.declared_causal_field
        ):
            raise M03RTop2000DevProtocolError(
                "TOP2000 development ablations must preserve the reviewed causal field"
            )
        if (
            not self.development_only
            or self.promotion_eligible
            or self.scientific_reporting_eligible
            or self.outer_lockbox_evaluation_eligible
        ):
            raise M03RTop2000DevProtocolError(
                "TOP2000 compatibility rows are development-only and nonreportable"
            )

    @property
    def reviewed_semantics_sha256(self) -> str:
        return _sha256(self.semantic_payload())

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "reviewed_v7_setting_id": self.reviewed_v7_setting_id,
            "persistence_coefficient_basis_points": (
                self.persistence_coefficient_basis_points
            ),
            "exit_hazard_mode": self.exit_hazard_mode,
            "exact_hold_action_supported": self.exact_hold_action_supported,
            "learned_temporal_context_trading_sessions": (
                self.learned_temporal_context_trading_sessions
            ),
            "residual_alpha_head_mode": self.residual_alpha_head_mode,
            "active_risk_budget_mode": self.active_risk_budget_mode,
            "development_active_risk_budget_execution_mode": (
                self.development_active_risk_budget_execution_mode
            ),
            "factor_sector_neutral_projection": (self.factor_sector_neutral_projection),
            "sharpe_mode": self.sharpe_mode,
        }


def _build_setting(
    development_setting_id: str,
    reviewed: M03RV7Setting,
) -> M03RTop2000DevSetting:
    return M03RTop2000DevSetting(
        setting_index=reviewed.setting_index,
        setting_id=development_setting_id,
        reviewed_v7_setting_id=reviewed.setting_id,
        persistence_coefficient_basis_points=(
            reviewed.persistence_coefficient_basis_points
        ),
        exit_hazard_mode=reviewed.exit_hazard_mode,
        exact_hold_action_supported=reviewed.exact_hold_action_supported,
        learned_temporal_context_trading_sessions=(
            reviewed.learned_temporal_context_trading_sessions
        ),
        residual_alpha_head_mode=reviewed.residual_alpha_head_mode,
        active_risk_budget_mode=reviewed.active_risk_budget_mode,
        development_active_risk_budget_execution_mode=(
            M03R_TOP2000_DEV_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE
            if reviewed.setting_index == 8
            else M03R_TOP2000_DEV_RAW_SIGMOID_ACTIVE_RISK_BUDGET_MODE
        ),
        factor_sector_neutral_projection=(reviewed.factor_sector_neutral_projection),
        sharpe_mode=reviewed.sharpe_mode,
        ablation_of=(
            None
            if reviewed.setting_index == 0
            else M03R_TOP2000_DEV_REFERENCE_SETTING_ID
        ),
        declared_causal_field=reviewed.declared_causal_field,
    )


M03R_TOP2000_DEV_SETTINGS = tuple(
    _build_setting(development_id, reviewed)
    for development_id, reviewed in zip(
        M03R_TOP2000_DEV_SETTING_IDS,
        M03R_V7_PRIMARY_SETTINGS,
        strict=True,
    )
)
M03R_TOP2000_DEV_SETTINGS_BY_ID = {
    setting.setting_id: setting for setting in M03R_TOP2000_DEV_SETTINGS
}
M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID = {
    setting.setting_id: setting.reviewed_v7_setting_id
    for setting in M03R_TOP2000_DEV_SETTINGS
}


def resolve_m03r_top2000_dev_setting(setting_id: str) -> M03RTop2000DevSetting:
    """Resolve only an exact disjoint TOP2000 development setting identity."""

    try:
        return M03R_TOP2000_DEV_SETTINGS_BY_ID[setting_id]
    except KeyError as exc:
        valid = ", ".join(M03R_TOP2000_DEV_SETTING_IDS)
        raise M03RTop2000DevProtocolError(
            f"unknown TOP2000 development setting {setting_id!r}; expected: {valid}"
        ) from exc


def validate_m03r_top2000_dev_artifact_identity(
    *,
    protocol_generation: str,
    design_id: str,
    setting_id: str,
) -> M03RTop2000DevSetting:
    """Reject canonical-v7 or other identities at the development boundary."""

    if protocol_generation != M03R_TOP2000_DEV_PROTOCOL_GENERATION:
        if protocol_generation == M03R_V7_PROTOCOL_GENERATION:
            raise M03RTop2000DevProtocolError(
                "canonical PIT-300 v7 cannot identify a TOP2000 development artifact"
            )
        raise M03RTop2000DevProtocolError(
            "invalid TOP2000 development protocol generation"
        )
    if design_id != M03R_TOP2000_DEV_DESIGN_ID:
        if design_id == M03R_V7_DESIGN_ID:
            raise M03RTop2000DevProtocolError(
                "canonical PIT-300 v7 design cannot identify TOP2000 development work"
            )
        raise M03RTop2000DevProtocolError("invalid TOP2000 development design ID")
    return resolve_m03r_top2000_dev_setting(setting_id)


def m03r_top2000_dev_protocol_payload() -> dict[str, Any]:
    """Return the deterministic non-authorizing compatibility payload."""

    return {
        "schema_version": M03R_TOP2000_DEV_SCHEMA_VERSION,
        "protocol_generation": M03R_TOP2000_DEV_PROTOCOL_GENERATION,
        "design_id": M03R_TOP2000_DEV_DESIGN_ID,
        "canonical_pit300_v7_protocol_generation": M03R_V7_PROTOCOL_GENERATION,
        "canonical_pit300_v7_design_id": M03R_V7_DESIGN_ID,
        "canonical_pit300_v7_identity_is_immutable": True,
        "universe_id": M03R_TOP2000_DEV_UNIVERSE_ID,
        "data_role": M03R_TOP2000_DEV_DATA_ROLE,
        "geometry": asdict(M03R_TOP2000_DEV_GEOMETRY),
        "cache_requirement": asdict(M03R_TOP2000_DEV_CACHE_REQUIREMENT),
        "settings": [asdict(setting) for setting in M03R_TOP2000_DEV_SETTINGS],
        "development_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "scientific_reporting_authorized": False,
        "outer_lockbox_evaluation_authorized": False,
        "authorization_blockers": (
            "top2000-development-cache-contract-unbound",
            "top2000-development-model-route-not-a-training-driver",
        ),
    }


M03R_TOP2000_DEV_PROTOCOL_SHA256 = _sha256(m03r_top2000_dev_protocol_payload())


__all__ = [
    "M03R_TOP2000_DEV_CACHE_CONTRACT_SCHEMA",
    "M03R_TOP2000_DEV_CACHE_REQUIREMENT",
    "M03R_TOP2000_DEV_DATA_ROLE",
    "M03R_TOP2000_DEV_DESIGN_ID",
    "M03R_TOP2000_DEV_GEOMETRY",
    "M03R_TOP2000_DEV_PROTOCOL_GENERATION",
    "M03R_TOP2000_DEV_PROTOCOL_SHA256",
    "M03R_TOP2000_DEV_REFERENCE_SETTING_ID",
    "M03R_TOP2000_DEV_SCHEMA_VERSION",
    "M03R_TOP2000_DEV_SETTINGS",
    "M03R_TOP2000_DEV_SETTINGS_BY_ID",
    "M03R_TOP2000_DEV_SETTING_IDS",
    "M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID",
    "M03R_TOP2000_DEV_UNIVERSE_ID",
    "M03RTop2000DevCacheRequirement",
    "M03RTop2000DevGeometry",
    "M03RTop2000DevProtocolError",
    "M03RTop2000DevSetting",
    "m03r_top2000_dev_protocol_payload",
    "resolve_m03r_top2000_dev_setting",
    "validate_m03r_top2000_dev_artifact_identity",
]
