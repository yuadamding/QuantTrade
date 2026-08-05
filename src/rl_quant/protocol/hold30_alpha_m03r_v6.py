"""Immutable M03R v6 soft-persistence active-alpha protocol.

V6 interprets 30 trading sessions only as a soft inductive bias.  It never
creates a minimum holding period, a pre-horizon sell mask, a forced expiry, or
a holding-duration promotion gate.  V4 and v5 retain their original identities
and semantics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_ALPHA_HORIZONS_TRADING_SESSIONS,
    M03R_PRIMARY_BENCHMARK_ID,
    M03R_PROTOCOL_GENERATION as M03R_V5_PROTOCOL_GENERATION,
    M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS,
    M03R_VALIDATION_ONE_WAY_COSTS_BASIS_POINTS,
    M03RActiveRiskContract,
    M03REnsembleExecutionContract,
    M03RFactorSectorProjectionContract,
)

M03R_PROTOCOL_GENERATION = "prelockbox-hold30-active-alpha-m03r-v6"
M03R_V6_PROTOCOL_GENERATION = M03R_PROTOCOL_GENERATION
M03R_V6_SCHEMA_VERSION = 6
M03R_SUPERSEDED_PROTOCOL_GENERATION = M03R_V5_PROTOCOL_GENERATION
M03R_DESIGN_ID = "daily_raw_pit300_hold30_m03r_v6"
M03R_CANONICAL_SETTING_ID = "M03R-soft-persistence-active-alpha-hold30"


class M03RProtocolError(ValueError):
    """A v6 identity or soft-persistence invariant is inconsistent."""


@dataclass(frozen=True, slots=True)
class M03RTemporalContract:
    """Separate context, rollout, economic-credit, and evaluation spans.

    The 30 post-fill returns support the economic objective.  They do not
    prescribe how long any position must remain open.
    """

    decisions_per_trading_session: int = 1
    fast_raw_context_trading_sessions: int = 42
    learned_temporal_context_trading_sessions: int = 252
    rollout_trading_sessions: int = 63
    economic_origin_post_fill_return_count: int = 30
    maximum_auxiliary_label_horizon_trading_sessions: int = 63
    evaluation_warmup_trading_sessions: int = 63
    evaluation_score_trading_sessions: int = 63

    def __post_init__(self) -> None:
        expected = {
            "decisions_per_trading_session": 1,
            "fast_raw_context_trading_sessions": 42,
            "learned_temporal_context_trading_sessions": 252,
            "rollout_trading_sessions": 63,
            "economic_origin_post_fill_return_count": 30,
            "maximum_auxiliary_label_horizon_trading_sessions": 63,
            "evaluation_warmup_trading_sessions": 63,
            "evaluation_score_trading_sessions": 63,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise M03RProtocolError(
                    f"M03R v6 {name} must be exactly {required}; temporal "
                    "spans cannot be substituted for persistence semantics"
                )

    @property
    def economic_origin_state_row_count(self) -> int:
        """One decision state plus exactly 30 post-fill transitions."""

        return self.economic_origin_post_fill_return_count + 1


@dataclass(frozen=True, slots=True)
class M03RSoftPersistenceContract:
    """Content-bound soft preference for persistent, freely revisable books."""

    holding_preference_horizon_sessions: int = 30
    holding_preference_is_inductive_bias_only: bool = True
    minimum_holding_period_sessions: int | None = None
    sell_mask_before_preference_horizon: bool = False
    forced_expiry_at_preference_horizon: bool = False
    holding_duration_is_promotion_gate: bool = False
    turnover_target_is_holding_duration_proxy: bool = False
    turnover_is_hard_holding_constraint: bool = False
    early_exit_always_allowed: bool = True
    exact_hold_action_supported: bool = True
    exact_hold_action_required: bool = False
    bounded_hazard_residual_minimum: float = -12.0
    bounded_hazard_residual_maximum: float = 12.0
    maximum_hazard_endpoint_means_full_discretionary_exit: bool = True
    early_exit_penalty_shape: str = "quadratic-one-sided"
    early_exit_penalty_bp_per_unit_at_age_zero: float = 5.0
    early_exit_penalty_inner_development_grid_bp_per_unit_at_age_zero: tuple[
        float, ...
    ] = (2.0, 5.0, 10.0)
    early_exit_penalty_warmup_shape: str = "linear-from-zero"
    early_exit_penalty_linear_warmup_fraction: float = 0.10
    early_exit_sold_notional_epsilon: float = 1e-12
    age_weight_formula: str = "max(0,1-age/30)^2"
    holding_to_economic_gradient_norm_ratio_diagnostic_band: tuple[float, float] = (
        0.05,
        0.15,
    )
    gradient_norm_ratio_is_promotion_gate: bool = False

    def __post_init__(self) -> None:
        if self.holding_preference_horizon_sessions != 30:
            raise M03RProtocolError(
                "M03R v6 holding preference must be exactly 30 sessions"
            )
        if not self.holding_preference_is_inductive_bias_only:
            raise M03RProtocolError("30 sessions must remain an inductive bias only")
        if self.minimum_holding_period_sessions is not None:
            raise M03RProtocolError("M03R v6 cannot impose a minimum holding period")
        if (
            self.sell_mask_before_preference_horizon
            or self.forced_expiry_at_preference_horizon
            or self.holding_duration_is_promotion_gate
            or self.turnover_target_is_holding_duration_proxy
            or self.turnover_is_hard_holding_constraint
            or not self.early_exit_always_allowed
        ):
            raise M03RProtocolError(
                "M03R v6 cannot mask sales, force expiry, gate promotion on age, "
                "or prohibit early exits"
            )
        if not self.exact_hold_action_supported or self.exact_hold_action_required:
            raise M03RProtocolError(
                "canonical v6 supports but never requires the exact-hold action"
            )
        if (
            self.bounded_hazard_residual_minimum != -12.0
            or self.bounded_hazard_residual_maximum != 12.0
            or not self.maximum_hazard_endpoint_means_full_discretionary_exit
        ):
            raise M03RProtocolError(
                "v6 hazard endpoints must permit exact hold and full discretionary exit"
            )
        if self.early_exit_penalty_shape != "quadratic-one-sided":
            raise M03RProtocolError("M03R v6 early-exit penalty shape drifted")
        if self.early_exit_penalty_bp_per_unit_at_age_zero != 5.0:
            raise M03RProtocolError(
                "canonical age-zero early-exit penalty must be 5 bp"
            )
        if self.early_exit_penalty_inner_development_grid_bp_per_unit_at_age_zero != (
            2.0,
            5.0,
            10.0,
        ):
            raise M03RProtocolError("M03R v6 inner-development penalty grid drifted")
        if (
            self.early_exit_penalty_warmup_shape != "linear-from-zero"
            or self.early_exit_penalty_linear_warmup_fraction != 0.10
        ):
            raise M03RProtocolError("M03R v6 requires a frozen 10% linear warmup")
        if (
            not math.isfinite(self.early_exit_sold_notional_epsilon)
            or self.early_exit_sold_notional_epsilon <= 0.0
            or self.early_exit_sold_notional_epsilon != 1e-12
        ):
            raise M03RProtocolError("M03R v6 sold-notional epsilon must be 1e-12")
        if self.age_weight_formula != "max(0,1-age/30)^2":
            raise M03RProtocolError("M03R v6 age-weight formula drifted")
        if (
            self.holding_to_economic_gradient_norm_ratio_diagnostic_band != (0.05, 0.15)
            or self.gradient_norm_ratio_is_promotion_gate
        ):
            raise M03RProtocolError(
                "v6 gradient balance must remain a diagnostic 5%-15% target"
            )

    def age_weight(self, age_sessions: float) -> float:
        """Return the one-sided quadratic penalty weight for a sale age."""

        if not math.isfinite(age_sessions) or age_sessions < 0.0:
            raise M03RProtocolError("age_sessions must be finite and nonnegative")
        remaining_fraction = max(
            0.0,
            1.0 - age_sessions / self.holding_preference_horizon_sessions,
        )
        return remaining_fraction * remaining_fraction

    def warmup_scale(self, completed_training_fraction: float) -> float:
        """Return the frozen linear coefficient ramp over the first 10%."""

        if (
            not math.isfinite(completed_training_fraction)
            or not 0.0 <= completed_training_fraction <= 1.0
        ):
            raise M03RProtocolError(
                "completed_training_fraction must be finite and in [0,1]"
            )
        return min(
            1.0,
            completed_training_fraction
            / self.early_exit_penalty_linear_warmup_fraction,
        )


M03R_SOFT_PERSISTENCE = M03RSoftPersistenceContract()


@dataclass(frozen=True, slots=True)
class M03RModelContract:
    """V6 capacity and action-support contract."""

    optimizer_family: str = "direct-differentiable-trajectory"
    maximum_trainable_parameters: int = 7_000_000
    initial_search_minimum_trainable_parameters: int = 1_000_000
    initial_search_maximum_trainable_parameters: int = 3_000_000
    trainable_raw_fast_branch_required: bool = True
    trainable_raw_fast_branch_trading_sessions: int = 42
    learned_temporal_context_required: bool = True
    learned_temporal_context_trading_sessions: int = 252
    trainable_raw_slow_branch_required: bool = False
    age_cohort_state_required: bool = True
    entry_head_required: bool = True
    exit_hazard_head_required: bool = True
    exact_hold_action_supported: bool = True
    exact_hold_action_required: bool = False
    uncertainty_head_required: bool = True
    confidence_calibration_manifest_sha256_required: bool = True
    confidence_calibration_schema: str = "rl-quant.m03r-confidence-calibration-v2"
    confidence_target_definition: str = (
        "probability-30-session-net-active-log-return-versus-C1-is-positive"
    )
    confidence_calibration_method: str = "temperature-sigmoid-v1"
    separate_sharpe_overlay_in_canonical: bool = False

    def __post_init__(self) -> None:
        if self.optimizer_family != "direct-differentiable-trajectory":
            raise M03RProtocolError("M03R v6 optimizer family drifted")
        if not (
            0
            < self.initial_search_minimum_trainable_parameters
            <= self.initial_search_maximum_trainable_parameters
            <= self.maximum_trainable_parameters
            == 7_000_000
        ):
            raise M03RProtocolError("M03R v6 model-size contract drifted")
        if not all(
            (
                self.trainable_raw_fast_branch_required,
                self.learned_temporal_context_required,
                self.age_cohort_state_required,
                self.entry_head_required,
                self.exit_hazard_head_required,
                self.uncertainty_head_required,
                self.confidence_calibration_manifest_sha256_required,
            )
        ):
            raise M03RProtocolError("M03R v6 canonical components are mandatory")
        if (
            self.trainable_raw_fast_branch_trading_sessions != 42
            or self.learned_temporal_context_trading_sessions != 252
            or self.trainable_raw_slow_branch_required
        ):
            raise M03RProtocolError("M03R v6 temporal model contract drifted")
        if not self.exact_hold_action_supported or self.exact_hold_action_required:
            raise M03RProtocolError("M03R v6 exact hold must be available but optional")
        if (
            self.confidence_calibration_schema
            != "rl-quant.m03r-confidence-calibration-v2"
            or self.confidence_target_definition
            != "probability-30-session-net-active-log-return-versus-C1-is-positive"
            or self.confidence_calibration_method != "temperature-sigmoid-v1"
        ):
            raise M03RProtocolError("M03R v6 confidence contract drifted")
        if self.separate_sharpe_overlay_in_canonical:
            raise M03RProtocolError("Sharpe overlay is not canonical M03R v6")


@dataclass(frozen=True, slots=True)
class M03RDesign:
    """Stable v6 design payload."""

    design_id: str
    primary_benchmark_id: str
    temporal: M03RTemporalContract
    soft_persistence: M03RSoftPersistenceContract
    active_risk: M03RActiveRiskContract
    factor_sector_projection: M03RFactorSectorProjectionContract
    model: M03RModelContract
    ensemble_execution: M03REnsembleExecutionContract
    auxiliary_alpha_horizons_trading_sessions: tuple[int, ...]
    primary_auxiliary_alpha_horizon_trading_sessions: int
    training_one_way_cost_basis_points: int
    validation_one_way_costs_basis_points: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.design_id != M03R_DESIGN_ID:
            raise M03RProtocolError(f"design_id must be {M03R_DESIGN_ID!r}")
        if self.primary_benchmark_id != M03R_PRIMARY_BENCHMARK_ID:
            raise M03RProtocolError("M03R v6 primary benchmark must remain C1")
        if self.soft_persistence != M03R_SOFT_PERSISTENCE:
            raise M03RProtocolError("M03R v6 soft-persistence contract drifted")
        if self.auxiliary_alpha_horizons_trading_sessions != (
            M03R_ALPHA_HORIZONS_TRADING_SESSIONS
        ):
            raise M03RProtocolError("M03R v6 auxiliary horizons drifted")
        if self.primary_auxiliary_alpha_horizon_trading_sessions != 30:
            raise M03RProtocolError("M03R v6 primary auxiliary horizon must be 30")
        if self.training_one_way_cost_basis_points != 20:
            raise M03RProtocolError("M03R v6 training cost must be 20 bp")
        if self.validation_one_way_costs_basis_points != (10, 20, 40):
            raise M03RProtocolError("M03R v6 validation costs must be 10/20/40 bp")


M03R_DESIGN = M03RDesign(
    design_id=M03R_DESIGN_ID,
    primary_benchmark_id=M03R_PRIMARY_BENCHMARK_ID,
    temporal=M03RTemporalContract(),
    soft_persistence=M03R_SOFT_PERSISTENCE,
    active_risk=M03RActiveRiskContract(),
    factor_sector_projection=M03RFactorSectorProjectionContract(),
    model=M03RModelContract(),
    ensemble_execution=M03REnsembleExecutionContract(),
    auxiliary_alpha_horizons_trading_sessions=M03R_ALPHA_HORIZONS_TRADING_SESSIONS,
    primary_auxiliary_alpha_horizon_trading_sessions=30,
    training_one_way_cost_basis_points=M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS,
    validation_one_way_costs_basis_points=M03R_VALIDATION_ONE_WAY_COSTS_BASIS_POINTS,
)


@dataclass(frozen=True, slots=True)
class M03RSetting:
    """One immutable v6 causal mechanism setting."""

    setting_index: int
    setting_id: str
    objective_mode: str
    age_aware_holding: bool
    soft_persistence: bool
    residual_alpha_heads: bool
    use_downside_adjusted_stock_score: bool
    use_confidence_scaled_active_risk_budget: bool
    annual_tracking_error_floor: float | None
    annual_tracking_error_ceiling: float | None
    confidence_preferred_tracking_error: bool
    active_beta_neutrality: bool
    factor_sector_projection: bool
    exit_hazard_mode: str
    slow_context_trading_sessions: int
    sharpe_mode: str
    exact_hold_action_supported: bool
    promotion_eligible: bool
    ablation_of: str | None
    description: str
    gradient_null_control_of: str | None = None

    def __post_init__(self) -> None:
        if not self.setting_id or not self.description:
            raise M03RProtocolError("v6 setting identity and description are required")
        if self.objective_mode not in {
            "absolute-net-log-return",
            "c1-active-net-log-return",
        }:
            raise M03RProtocolError("unknown v6 objective mode")
        if self.exit_hazard_mode not in {"learned-age-aware", "fixed-hold30-prior"}:
            raise M03RProtocolError("unknown v6 exit-hazard mode")
        if self.sharpe_mode not in {
            "none",
            "separate-total-risk-overlay",
            "direct-two-pass-gradient",
        }:
            raise M03RProtocolError("unknown v6 Sharpe mode")
        if self.slow_context_trading_sessions not in {63, 252}:
            raise M03RProtocolError("v6 slow context must be 63 or 252 sessions")
        if not self.soft_persistence:
            raise M03RProtocolError("all v6 settings use the soft-persistence contract")
        if self.promotion_eligible and self.setting_id != M03R_CANONICAL_SETTING_ID:
            raise M03RProtocolError("only canonical M03R v6 may be promotion eligible")
        if self.setting_id == "M01-benchmark-subtraction-v6":
            if self.gradient_null_control_of != "M00-absolute-return-v6":
                raise M03RProtocolError("M01 v6 must bind its M00 gradient-null parent")
        elif self.gradient_null_control_of is not None:
            raise M03RProtocolError("only M01 v6 may bind a gradient-null parent")

    @property
    def emits_exact_hold_action(self) -> bool:
        """Whether this route instantiates the optional learned action atom."""

        return bool(
            self.exact_hold_action_supported
            and self.exit_hazard_mode == "learned-age-aware"
        )


_CANONICAL = M03RSetting(
    setting_index=3,
    setting_id=M03R_CANONICAL_SETTING_ID,
    objective_mode="c1-active-net-log-return",
    age_aware_holding=True,
    soft_persistence=True,
    residual_alpha_heads=True,
    use_downside_adjusted_stock_score=True,
    use_confidence_scaled_active_risk_budget=True,
    annual_tracking_error_floor=0.0,
    annual_tracking_error_ceiling=0.06,
    confidence_preferred_tracking_error=True,
    active_beta_neutrality=True,
    factor_sector_projection=True,
    exit_hazard_mode="learned-age-aware",
    slow_context_trading_sessions=252,
    sharpe_mode="none",
    exact_hold_action_supported=True,
    promotion_eligible=True,
    ablation_of=None,
    description="Canonical v6 active-alpha policy with a soft 30-session persistence bias.",
)

M03R_SETTINGS: tuple[M03RSetting, ...] = (
    M03RSetting(
        0,
        "M00-absolute-return-v6",
        "absolute-net-log-return",
        True,
        True,
        False,
        False,
        False,
        None,
        None,
        False,
        False,
        False,
        "learned-age-aware",
        252,
        "none",
        True,
        False,
        None,
        "V6 soft-persistence architecture with absolute net log return only.",
    ),
    M03RSetting(
        1,
        "M01-benchmark-subtraction-v6",
        "c1-active-net-log-return",
        True,
        True,
        False,
        False,
        False,
        None,
        None,
        False,
        False,
        False,
        "learned-age-aware",
        252,
        "none",
        True,
        False,
        None,
        "Gradient-null v6 governance control: M00 plus detached C1 subtraction.",
        "M00-absolute-return-v6",
    ),
    M03RSetting(
        2,
        "M02-active-risk-no-alpha-heads-v6",
        "c1-active-net-log-return",
        True,
        True,
        False,
        False,
        True,
        0.0,
        0.06,
        True,
        True,
        True,
        "learned-age-aware",
        252,
        "none",
        True,
        False,
        None,
        "V6 active-risk controls without residual-alpha heads.",
    ),
    _CANONICAL,
    replace(
        _CANONICAL,
        setting_index=4,
        setting_id="A04-no-downside-score-adjustment-v6",
        use_downside_adjusted_stock_score=False,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="V6 canonical without downside-adjusted stock scores.",
    ),
    replace(
        _CANONICAL,
        setting_index=5,
        setting_id="A05-fixed-te-floor-v6",
        annual_tracking_error_floor=0.02,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="V6 canonical with the rejected 2% compulsory TE floor.",
    ),
    replace(
        _CANONICAL,
        setting_index=6,
        setting_id="A06-sharpe-overlay-v6",
        sharpe_mode="separate-total-risk-overlay",
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="V6 canonical plus a separate total-risk Sharpe overlay.",
    ),
    replace(
        _CANONICAL,
        setting_index=7,
        setting_id="A07-direct-sharpe-v6",
        sharpe_mode="direct-two-pass-gradient",
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="V6 canonical plus the direct two-pass Sharpe ablation.",
    ),
    replace(
        _CANONICAL,
        setting_index=8,
        setting_id="A08-fixed-exit-hazard-v6",
        exit_hazard_mode="fixed-hold30-prior",
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="V6 canonical with the soft 30-session exit prior frozen.",
    ),
    replace(
        _CANONICAL,
        setting_index=9,
        setting_id="A09-no-long-context-v6",
        slow_context_trading_sessions=63,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="V6 canonical with learned context truncated to 63 sessions.",
    ),
    replace(
        _CANONICAL,
        setting_index=10,
        setting_id="A10-no-factor-neutral-projection-v6",
        factor_sector_projection=False,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="V6 canonical without factor/sector-neutral projection.",
    ),
    replace(
        _CANONICAL,
        setting_index=11,
        setting_id="A11-no-exact-hold-atom",
        exact_hold_action_supported=False,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="V6 canonical without the optional exact-hold action atom.",
    ),
)

M03R_SETTING_IDS = tuple(setting.setting_id for setting in M03R_SETTINGS)
M03R_SETTINGS_BY_ID = {setting.setting_id: setting for setting in M03R_SETTINGS}


def _validate_inventory() -> None:
    if tuple(row.setting_index for row in M03R_SETTINGS) != tuple(range(12)):
        raise RuntimeError("M03R v6 setting indexes must be contiguous 0 through 11")
    if len(M03R_SETTINGS_BY_ID) != 12:
        raise RuntimeError("M03R v6 setting IDs must be unique")
    if [row.setting_id for row in M03R_SETTINGS if row.promotion_eligible] != [
        M03R_CANONICAL_SETTING_ID
    ]:
        raise RuntimeError("M03R v6 must have exactly one promotion candidate")

    causal_fields = (
        "objective_mode",
        "age_aware_holding",
        "soft_persistence",
        "residual_alpha_heads",
        "use_downside_adjusted_stock_score",
        "use_confidence_scaled_active_risk_budget",
        "annual_tracking_error_floor",
        "annual_tracking_error_ceiling",
        "confidence_preferred_tracking_error",
        "active_beta_neutrality",
        "factor_sector_projection",
        "exit_hazard_mode",
        "slow_context_trading_sessions",
        "sharpe_mode",
        "exact_hold_action_supported",
    )
    expected = {
        "A04-no-downside-score-adjustment-v6": "use_downside_adjusted_stock_score",
        "A05-fixed-te-floor-v6": "annual_tracking_error_floor",
        "A06-sharpe-overlay-v6": "sharpe_mode",
        "A07-direct-sharpe-v6": "sharpe_mode",
        "A08-fixed-exit-hazard-v6": "exit_hazard_mode",
        "A09-no-long-context-v6": "slow_context_trading_sessions",
        "A10-no-factor-neutral-projection-v6": "factor_sector_projection",
        "A11-no-exact-hold-atom": "exact_hold_action_supported",
    }
    for setting_id, expected_field in expected.items():
        row = M03R_SETTINGS_BY_ID[setting_id]
        changed = tuple(
            field
            for field in causal_fields
            if getattr(row, field) != getattr(_CANONICAL, field)
        )
        if changed != (expected_field,):
            raise RuntimeError(
                f"{setting_id} must differ only in {expected_field}; observed {changed}"
            )


_validate_inventory()


def resolve_m03r_v6_setting(setting_id: str) -> M03RSetting:
    """Resolve an exact v6 setting ID; v4/v5 aliases fail closed."""

    try:
        return M03R_SETTINGS_BY_ID[setting_id]
    except KeyError as exc:
        valid = ", ".join(M03R_SETTING_IDS)
        raise M03RProtocolError(
            f"unknown M03R v6 setting {setting_id!r}; expected one of: {valid}"
        ) from exc


def validate_m03r_v6_artifact_identity(
    *, protocol_generation: str, design_id: str, setting_id: str
) -> M03RSetting:
    """Validate the complete v6 artifact identity without relabeling history."""

    if protocol_generation != M03R_PROTOCOL_GENERATION:
        if protocol_generation == M03R_V5_PROTOCOL_GENERATION:
            raise M03RProtocolError(
                "M03R v5 remains immutable and cannot identify a v6 artifact"
            )
        raise M03RProtocolError(
            f"protocol_generation must be {M03R_PROTOCOL_GENERATION!r}"
        )
    if design_id != M03R_DESIGN_ID:
        raise M03RProtocolError(f"design_id must be {M03R_DESIGN_ID!r}")
    return resolve_m03r_v6_setting(setting_id)


def m03r_v6_design_payload() -> dict[str, Any]:
    """Return the stable JSON-ready v6 design and causal inventory."""

    return {
        "schema_version": M03R_V6_SCHEMA_VERSION,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "supersedes_protocol_generation": M03R_SUPERSEDED_PROTOCOL_GENERATION,
        "v5_artifacts_retain_their_original_identity": True,
        "canonical_setting_id": M03R_CANONICAL_SETTING_ID,
        "design": asdict(M03R_DESIGN),
        "settings": [asdict(setting) for setting in M03R_SETTINGS],
        "launch_authorized": False,
        "launch_blockers": (
            "governed-v6-all-setting-production-driver-not-implemented",
            "authoritative-cause-typed-chronological-ledger-adapter-not-implemented",
            "v6-seed-ensemble-and-risk-projection-receipts-not-implemented",
            "point-in-time-real-data-and-inference-family-not-sealed",
            "cuda-two-rank-restart-and-h100-capacity-not-qualified",
        ),
        "required_launch_bindings": (
            "soft_persistence_contract_sha256",
            "point_in_time_factor_manifest_sha256",
            "point_in_time_sector_manifest_sha256",
            "factor_sector_exposure_bounds_manifest_sha256",
            "m03r_risk_manifest_sha256",
            "m03r_seed_checkpoint_ensemble_manifest_sha256",
            "per_seed_confidence_calibration_manifest_sha256s",
            "projection_execution_contract_sha256",
            "inference_contract_sha256",
            "source_archive_sha256",
            "container_image_digest",
            "data_manifest_sha256",
        ),
    }


__all__ = [
    "M03R_ALPHA_HORIZONS_TRADING_SESSIONS",
    "M03R_CANONICAL_SETTING_ID",
    "M03R_DESIGN",
    "M03R_DESIGN_ID",
    "M03R_PRIMARY_BENCHMARK_ID",
    "M03R_PROTOCOL_GENERATION",
    "M03R_SETTINGS",
    "M03R_SETTINGS_BY_ID",
    "M03R_SETTING_IDS",
    "M03R_SOFT_PERSISTENCE",
    "M03R_SUPERSEDED_PROTOCOL_GENERATION",
    "M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS",
    "M03R_V5_PROTOCOL_GENERATION",
    "M03R_V6_PROTOCOL_GENERATION",
    "M03R_V6_SCHEMA_VERSION",
    "M03R_VALIDATION_ONE_WAY_COSTS_BASIS_POINTS",
    "M03RActiveRiskContract",
    "M03RDesign",
    "M03REnsembleExecutionContract",
    "M03RFactorSectorProjectionContract",
    "M03RModelContract",
    "M03RProtocolError",
    "M03RSetting",
    "M03RSoftPersistenceContract",
    "M03RTemporalContract",
    "m03r_v6_design_payload",
    "resolve_m03r_v6_setting",
    "validate_m03r_v6_artifact_identity",
]
