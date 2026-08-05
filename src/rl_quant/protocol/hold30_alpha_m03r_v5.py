"""Immutable M03R v5 Hold-30 active-alpha scientific protocol.

This is a new protocol generation. It supersedes the design intent of M03R v4
without changing, aliasing, or accepting any v4 or v3 artifact identity.
The module is deliberately declarative: trainers, launch preflights, and
evaluators may bind to these values, but must not reinterpret them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

M03R_PROTOCOL_GENERATION = "prelockbox-hold30-active-alpha-m03r-v5"
M03R_V5_PROTOCOL_GENERATION = M03R_PROTOCOL_GENERATION
M03R_V5_SCHEMA_VERSION = 5
M03R_SUPERSEDED_PROTOCOL_GENERATION = "prelockbox-hold30-active-alpha-m03r-v4"
M03R_V3_AUDIT_PROTOCOL_GENERATION = "prelockbox-hold30-alpha-mech8-v3"
M03R_DESIGN_ID = "daily_raw_pit300_hold30_m03r_v5"
M03R_CANONICAL_SETTING_ID = "M03R-active-alpha-hold30"
M03R_PRIMARY_BENCHMARK_ID = "C1-monthly-pit-active300-equal-weight-buy-and-drift"

M03R_ALPHA_HORIZONS_TRADING_SESSIONS = (5, 21, 30, 63)
M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS = 20
M03R_VALIDATION_ONE_WAY_COSTS_BASIS_POINTS = (10, 20, 40)


class M03RProtocolError(ValueError):
    """An M03R identity or scientific invariant is absent or inconsistent."""


@dataclass(frozen=True, slots=True)
class M03RTemporalContract:
    """Disjoint context, rollout, reward, holding, and evaluation horizons.

    The verbose field names and units are intentional.  In particular, no
    downstream manifest may collapse these values into an ambiguous
    ``credit_span`` or ``horizon`` field.
    """

    decisions_per_trading_session: int = 1
    fast_raw_context_trading_sessions: int = 42
    learned_temporal_context_trading_sessions: int = 252
    rollout_trading_sessions: int = 63
    economic_origin_post_fill_return_count: int = 30
    maximum_auxiliary_label_horizon_trading_sessions: int = 63
    target_holding_trading_sessions: int = 30
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
            "target_holding_trading_sessions": 30,
            "evaluation_warmup_trading_sessions": 63,
            "evaluation_score_trading_sessions": 63,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise M03RProtocolError(
                    f"M03R {name} must be exactly {required}; temporal fields "
                    "cannot be substituted for one another"
                )

    @property
    def economic_origin_state_row_count(self) -> int:
        """One decision state plus exactly 30 post-fill return transitions."""

        return self.economic_origin_post_fill_return_count + 1


@dataclass(frozen=True, slots=True)
class M03RActiveRiskContract:
    """Canonical active-risk controls, with no compulsory active variance."""

    annual_tracking_error_floor: float = 0.0
    annual_tracking_error_ceiling: float = 0.06
    confidence_preferred_annual_tracking_error_minimum: float = 0.0
    confidence_preferred_annual_tracking_error_maximum: float = 0.04
    confidence_preferred_tracking_error_rule: str = (
        "linear:preferred_annual_te=0.04*calibrated_confidence;confidence-in-[0,1]"
    )
    active_market_beta_target: float = 0.0
    absolute_active_market_beta_maximum: float = 0.10
    total_portfolio_market_beta_is_secondary_diagnostic: bool = True
    maximum_asset_weight_fraction: float = 0.01
    target_daily_one_way_discretionary_turnover: float = 1.0 / 30.0
    maximum_confidence_incremental_one_way_turnover: float = 1.0
    calibrated_confidence_controls_new_active_risk_only: bool = True
    learned_hazard_exits_are_confidence_independent: bool = True
    confidence_budget_applies_to_replacement_entry_only: bool = True
    calibrated_confidence_scales_entry_scores: bool = False
    zero_confidence_forces_benchmark_derisk: bool = False
    benchmark_derisk_request_is_separate: bool = True
    canonical_benchmark_derisk_request: float = 0.0

    def __post_init__(self) -> None:
        exact = (
            self.annual_tracking_error_floor,
            self.annual_tracking_error_ceiling,
            self.confidence_preferred_annual_tracking_error_minimum,
            self.confidence_preferred_annual_tracking_error_maximum,
            self.active_market_beta_target,
            self.absolute_active_market_beta_maximum,
            self.maximum_asset_weight_fraction,
            self.target_daily_one_way_discretionary_turnover,
            self.maximum_confidence_incremental_one_way_turnover,
        )
        required = (
            0.0,
            0.06,
            0.0,
            0.04,
            0.0,
            0.10,
            0.01,
            1.0 / 30.0,
            1.0,
        )
        if exact != required:
            raise M03RProtocolError("M03R canonical active-risk values drifted")
        if self.confidence_preferred_tracking_error_rule != (
            "linear:preferred_annual_te=0.04*calibrated_confidence;confidence-in-[0,1]"
        ):
            raise M03RProtocolError("M03R confidence-dependent TE rule drifted")
        if not self.total_portfolio_market_beta_is_secondary_diagnostic:
            raise M03RProtocolError(
                "M03R must constrain active beta; total beta is only secondary"
            )
        if not self.calibrated_confidence_controls_new_active_risk_only:
            raise M03RProtocolError(
                "M03R confidence must control only new or enlarged active risk"
            )
        if (
            not self.learned_hazard_exits_are_confidence_independent
            or not self.confidence_budget_applies_to_replacement_entry_only
        ):
            raise M03RProtocolError(
                "M03R learned exits must precede confidence-limited replacement entry"
            )
        if self.calibrated_confidence_scales_entry_scores:
            raise M03RProtocolError(
                "M03R must not apply confidence a second time to entry scores"
            )
        if self.zero_confidence_forces_benchmark_derisk:
            raise M03RProtocolError(
                "zero confidence is a no-new-risk signal, not a C1 liquidation order"
            )
        if (
            not self.benchmark_derisk_request_is_separate
            or self.canonical_benchmark_derisk_request != 0.0
        ):
            raise M03RProtocolError(
                "benchmark de-risking must be a separate, canonically inactive request"
            )

    def preferred_annual_tracking_error(self, calibrated_confidence: float) -> float:
        """Return the frozen confidence-scaled preferred annual tracking error."""

        if (
            not math.isfinite(calibrated_confidence)
            or not 0.0 <= calibrated_confidence <= 1.0
        ):
            raise M03RProtocolError("calibrated_confidence must be finite and in [0,1]")
        return (
            self.confidence_preferred_annual_tracking_error_maximum
            * calibrated_confidence
        )

    def maximum_incremental_one_way_turnover(
        self,
        calibrated_confidence: float,
    ) -> float:
        """Return the nondegenerate confidence-scaled L1 safety budget."""

        if (
            not math.isfinite(calibrated_confidence)
            or not 0.0 <= calibrated_confidence <= 1.0
        ):
            raise M03RProtocolError("calibrated_confidence must be finite and in [0,1]")
        return (
            self.maximum_confidence_incremental_one_way_turnover
            * calibrated_confidence
        )


@dataclass(frozen=True, slots=True)
class M03RFactorSectorProjectionContract:
    """Risk-layer projection; exposure identities never become actor features.

    Numerical sector/factor bands depend on a point-in-time exposure model and
    are result-moving.  They must therefore arrive through the named,
    content-addressed launch binding rather than an implementation default.
    """

    requested_quantity: str = "active-weight-delta-versus-C1"
    projection_objective: str = (
        "linear-minimum-l2-then-benchmark-radial-tracking-error-scaling"
    )
    linear_projection_objective: str = (
        "minimum-l2-distance-over-affine-box-and-linear-exposure-sets"
    )
    tracking_error_operation: str = "benchmark-radial-scaling-after-linear-projection"
    joint_covariance_ellipsoid_minimum_l2_claimed: bool = False
    exposure_families: tuple[str, ...] = (
        "market",
        "sector",
        "size",
        "momentum",
        "value",
        "volatility",
        "liquidity",
    )
    exposure_loadings_are_point_in_time: bool = True
    exposure_loadings_actor_feature_access: bool = False
    active_weight_sum_target: float = 0.0
    long_only_post_projection: bool = True
    projection_applied_after_seed_ensemble: bool = True
    deterministic_solver_tie_break: str = "lexical-point-in-time-asset-id"
    numerical_exposure_bounds_binding_schema: str = (
        "rl-quant.m03r-factor-sector-exposure-bounds-v1"
    )
    numerical_exposure_bounds_manifest_sha256_required: bool = True
    infeasible_projection_behavior: str = "fail-closed-no-artifact"

    def __post_init__(self) -> None:
        if self.requested_quantity != "active-weight-delta-versus-C1":
            raise M03RProtocolError("M03R projection must operate on C1 active weights")
        if self.projection_objective != (
            "linear-minimum-l2-then-benchmark-radial-tracking-error-scaling"
        ):
            raise M03RProtocolError("M03R projection objective drifted")
        if (
            self.linear_projection_objective
            != "minimum-l2-distance-over-affine-box-and-linear-exposure-sets"
            or self.tracking_error_operation
            != "benchmark-radial-scaling-after-linear-projection"
            or self.joint_covariance_ellipsoid_minimum_l2_claimed
        ):
            raise M03RProtocolError(
                "M03R v5 must state its deterministic two-stage projection exactly"
            )
        if self.exposure_families != (
            "market",
            "sector",
            "size",
            "momentum",
            "value",
            "volatility",
            "liquidity",
        ):
            raise M03RProtocolError("M03R factor/sector exposure family drifted")
        if not self.exposure_loadings_are_point_in_time:
            raise M03RProtocolError("M03R exposure loadings must be point in time")
        if self.exposure_loadings_actor_feature_access:
            raise M03RProtocolError(
                "M03R actor cannot consume factor/sector identities"
            )
        if self.active_weight_sum_target != 0.0 or not self.long_only_post_projection:
            raise M03RProtocolError(
                "M03R projection must preserve zero-sum active weights and long-only holdings"
            )
        if not self.projection_applied_after_seed_ensemble:
            raise M03RProtocolError("M03R constraints apply once after seed ensembling")
        if self.deterministic_solver_tie_break != "lexical-point-in-time-asset-id":
            raise M03RProtocolError("M03R projection tie break drifted")
        if (
            self.numerical_exposure_bounds_binding_schema
            != ("rl-quant.m03r-factor-sector-exposure-bounds-v1")
            or not self.numerical_exposure_bounds_manifest_sha256_required
        ):
            raise M03RProtocolError("M03R must content-bind numerical exposure bands")
        if self.infeasible_projection_behavior != "fail-closed-no-artifact":
            raise M03RProtocolError("M03R infeasible projections must fail closed")


@dataclass(frozen=True, slots=True)
class M03RModelContract:
    """Capacity and information-flow contract for the revised candidate."""

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
    exact_hold_branch_required: bool = True
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
            raise M03RProtocolError("M03R canonical optimizer is direct differentiable")
        if not (
            0
            < self.initial_search_minimum_trainable_parameters
            <= self.initial_search_maximum_trainable_parameters
            <= self.maximum_trainable_parameters
            == 7_000_000
        ):
            raise M03RProtocolError("M03R model-size contract drifted")
        if not all(
            (
                self.trainable_raw_fast_branch_required,
                self.learned_temporal_context_required,
                self.age_cohort_state_required,
                self.entry_head_required,
                self.exit_hazard_head_required,
                self.exact_hold_branch_required,
                self.uncertainty_head_required,
                self.confidence_calibration_manifest_sha256_required,
            )
        ):
            raise M03RProtocolError("M03R canonical model components are mandatory")
        if (
            self.trainable_raw_fast_branch_trading_sessions != 42
            or self.learned_temporal_context_trading_sessions != 252
            or self.trainable_raw_slow_branch_required
        ):
            raise M03RProtocolError(
                "M03R implements 42 trainable raw sessions plus 252 learned temporal sessions"
            )
        if (
            self.confidence_calibration_schema
            != "rl-quant.m03r-confidence-calibration-v2"
            or self.confidence_target_definition
            != "probability-30-session-net-active-log-return-versus-C1-is-positive"
            or self.confidence_calibration_method != "temperature-sigmoid-v1"
        ):
            raise M03RProtocolError("M03R confidence-calibration contract drifted")
        if self.separate_sharpe_overlay_in_canonical:
            raise M03RProtocolError("Sharpe overlay is an ablation, not canonical M03R")


@dataclass(frozen=True, slots=True)
class M03REnsembleExecutionContract:
    """Frozen seed aggregation and post-ensemble projection semantics."""

    ensemble_rule_id: str = "five-seed-m03r-output-ensemble-v1"
    ensemble_member_count: int = 5
    member_seed_and_checkpoint_manifest_sha256_required: bool = True
    aggregation_order: str = "ascending-integer-seed"
    post_ensemble_projection_application_count: int = 2
    hazard_anchor_projection_application_count: int = 1
    replacement_proposal_projection_application_count: int = 1
    risk_manifest_schema: str = "rl-quant.m03r-factor-sector-risk-manifest-v2"
    projection_solver: str = "deterministic-dykstra-euclidean"
    projection_tolerance: float = 1e-10
    projection_maximum_iterations: int = 4_000
    cash_asset_id_launch_binding_required: bool = True
    maximum_one_way_turnover_launch_binding_required: bool = True

    def __post_init__(self) -> None:
        if self.ensemble_rule_id != "five-seed-m03r-output-ensemble-v1":
            raise M03RProtocolError("M03R ensemble rule identity drifted")
        if self.ensemble_member_count != 5:
            raise M03RProtocolError("M03R requires exactly five seed members")
        if not self.member_seed_and_checkpoint_manifest_sha256_required:
            raise M03RProtocolError("M03R seed/checkpoint inventory must be bound")
        if self.aggregation_order != "ascending-integer-seed":
            raise M03RProtocolError("M03R aggregation order drifted")
        if (
            self.post_ensemble_projection_application_count != 2
            or self.hazard_anchor_projection_application_count != 1
            or self.replacement_proposal_projection_application_count != 1
        ):
            raise M03RProtocolError(
                "M03R must separately project the hazard anchor and replacement proposal"
            )
        if self.risk_manifest_schema != (
            "rl-quant.m03r-factor-sector-risk-manifest-v2"
        ):
            raise M03RProtocolError("M03R risk-manifest schema drifted")
        if (
            self.projection_solver != "deterministic-dykstra-euclidean"
            or self.projection_tolerance != 1e-10
            or self.projection_maximum_iterations != 4_000
        ):
            raise M03RProtocolError("M03R projection numerics drifted")
        if not (
            self.cash_asset_id_launch_binding_required
            and self.maximum_one_way_turnover_launch_binding_required
        ):
            raise M03RProtocolError("M03R execution bindings must fail closed")


@dataclass(frozen=True, slots=True)
class M03RDesign:
    """Stable downstream API for the canonical M03R design."""

    design_id: str
    primary_benchmark_id: str
    temporal: M03RTemporalContract
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
            raise M03RProtocolError(f"M03R design_id must be {M03R_DESIGN_ID!r}")
        if self.primary_benchmark_id != M03R_PRIMARY_BENCHMARK_ID:
            raise M03RProtocolError("M03R primary benchmark must be frozen C1")
        if self.auxiliary_alpha_horizons_trading_sessions != (
            M03R_ALPHA_HORIZONS_TRADING_SESSIONS
        ):
            raise M03RProtocolError("M03R auxiliary alpha horizons drifted")
        if self.primary_auxiliary_alpha_horizon_trading_sessions != 30:
            raise M03RProtocolError("M03R primary auxiliary alpha horizon must be 30")
        if self.training_one_way_cost_basis_points != 20:
            raise M03RProtocolError("M03R training cost must be 20 one-way bp")
        if self.validation_one_way_costs_basis_points != (10, 20, 40):
            raise M03RProtocolError("M03R validation costs must be 10/20/40 one-way bp")


M03R_DESIGN = M03RDesign(
    design_id=M03R_DESIGN_ID,
    primary_benchmark_id=M03R_PRIMARY_BENCHMARK_ID,
    temporal=M03RTemporalContract(),
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
    """One immutable mechanism or causal ablation in the M03R inventory."""

    setting_index: int
    setting_id: str
    objective_mode: str
    age_aware_holding: bool
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
    promotion_eligible: bool
    ablation_of: str | None
    description: str
    gradient_null_control_of: str | None = None

    def __post_init__(self) -> None:
        if not self.setting_id or not self.description:
            raise M03RProtocolError(
                "M03R setting identity and description are required"
            )
        if self.objective_mode not in {
            "absolute-net-log-return",
            "c1-active-net-log-return",
        }:
            raise M03RProtocolError("unknown M03R objective mode")
        if self.exit_hazard_mode not in {"learned-age-aware", "fixed-hold30-prior"}:
            raise M03RProtocolError("unknown M03R exit-hazard mode")
        if self.sharpe_mode not in {
            "none",
            "separate-total-risk-overlay",
            "direct-two-pass-gradient",
        }:
            raise M03RProtocolError("unknown M03R Sharpe mode")
        if self.slow_context_trading_sessions not in {63, 252}:
            raise M03RProtocolError(
                "M03R slow context must be 252 or A09's 63 sessions"
            )
        for name in (
            "annual_tracking_error_floor",
            "annual_tracking_error_ceiling",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise M03RProtocolError(f"{name} must be finite, nonnegative, or None")
        if (
            self.annual_tracking_error_floor is not None
            and self.annual_tracking_error_ceiling is not None
            and self.annual_tracking_error_floor > self.annual_tracking_error_ceiling
        ):
            raise M03RProtocolError("tracking-error floor cannot exceed ceiling")
        if self.promotion_eligible and self.setting_id != M03R_CANONICAL_SETTING_ID:
            raise M03RProtocolError("only M03R canonical may be promotion eligible")
        if self.promotion_eligible and self.sharpe_mode != "none":
            raise M03RProtocolError("Sharpe variants are causal ablations only")
        if self.setting_id == "M01-benchmark-subtraction":
            if self.gradient_null_control_of != "M00-absolute-return":
                raise M03RProtocolError(
                    "M01 must be identified as M00's gradient-null governance control"
                )
        elif self.gradient_null_control_of is not None:
            raise M03RProtocolError(
                "only M01 may identify a gradient-null governance parent"
            )


_M03R_CANONICAL_SETTING = M03RSetting(
    setting_index=3,
    setting_id=M03R_CANONICAL_SETTING_ID,
    objective_mode="c1-active-net-log-return",
    age_aware_holding=True,
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
    promotion_eligible=True,
    ablation_of=None,
    description=(
        "Canonical age-aware Hold-30 active-alpha policy with downside-adjusted "
        "stock scores and confidence-budgeted new active risk, "
        "no TE floor, active-beta neutrality, factor projection, and 252-session context."
    ),
)


M03R_SETTINGS: tuple[M03RSetting, ...] = (
    M03RSetting(
        setting_index=0,
        setting_id="M00-absolute-return",
        objective_mode="absolute-net-log-return",
        age_aware_holding=True,
        residual_alpha_heads=False,
        use_downside_adjusted_stock_score=False,
        use_confidence_scaled_active_risk_budget=False,
        annual_tracking_error_floor=None,
        annual_tracking_error_ceiling=None,
        confidence_preferred_tracking_error=False,
        active_beta_neutrality=False,
        factor_sector_projection=False,
        exit_hazard_mode="learned-age-aware",
        slow_context_trading_sessions=252,
        sharpe_mode="none",
        promotion_eligible=False,
        ablation_of=None,
        description="Hold-30 architecture with the absolute-return objective only.",
    ),
    M03RSetting(
        setting_index=1,
        setting_id="M01-benchmark-subtraction",
        objective_mode="c1-active-net-log-return",
        age_aware_holding=True,
        residual_alpha_heads=False,
        use_downside_adjusted_stock_score=False,
        use_confidence_scaled_active_risk_budget=False,
        annual_tracking_error_floor=None,
        annual_tracking_error_ceiling=None,
        confidence_preferred_tracking_error=False,
        active_beta_neutrality=False,
        factor_sector_projection=False,
        exit_hazard_mode="learned-age-aware",
        slow_context_trading_sessions=252,
        sharpe_mode="none",
        promotion_eligible=False,
        ablation_of=None,
        description=(
            "Gradient-null governance control: M00 plus detached C1 subtraction. "
            "It must have identical training gradients and optimizer updates to M00."
        ),
        gradient_null_control_of="M00-absolute-return",
    ),
    M03RSetting(
        setting_index=2,
        setting_id="M02-active-risk-no-alpha-heads",
        objective_mode="c1-active-net-log-return",
        age_aware_holding=True,
        residual_alpha_heads=False,
        use_downside_adjusted_stock_score=False,
        use_confidence_scaled_active_risk_budget=True,
        annual_tracking_error_floor=0.0,
        annual_tracking_error_ceiling=0.06,
        confidence_preferred_tracking_error=True,
        active_beta_neutrality=True,
        factor_sector_projection=True,
        exit_hazard_mode="learned-age-aware",
        slow_context_trading_sessions=252,
        sharpe_mode="none",
        promotion_eligible=False,
        ablation_of=None,
        description="M01 plus active-risk controls and factor/sector projection, without alpha heads.",
    ),
    _M03R_CANONICAL_SETTING,
    replace(
        _M03R_CANONICAL_SETTING,
        setting_index=4,
        setting_id="A04-no-downside-score-adjustment",
        use_downside_adjusted_stock_score=False,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description=(
            "Canonical M03R without downside adjustment of stock scores; "
            "confidence-scaled new-active-risk budgeting remains enabled."
        ),
    ),
    replace(
        _M03R_CANONICAL_SETTING,
        setting_index=5,
        setting_id="A05-fixed-te-floor",
        annual_tracking_error_floor=0.02,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="Canonical M03R with the rejected 2% compulsory annual TE floor restored.",
    ),
    replace(
        _M03R_CANONICAL_SETTING,
        setting_index=6,
        setting_id="A06-sharpe-overlay",
        sharpe_mode="separate-total-risk-overlay",
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="Canonical M03R plus a separately optimized total-risk/Sharpe overlay.",
    ),
    replace(
        _M03R_CANONICAL_SETTING,
        setting_index=7,
        setting_id="A07-direct-sharpe",
        sharpe_mode="direct-two-pass-gradient",
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="Canonical M03R plus the direct two-pass Sharpe-gradient ablation.",
    ),
    replace(
        _M03R_CANONICAL_SETTING,
        setting_index=8,
        setting_id="A08-fixed-exit-hazard",
        exit_hazard_mode="fixed-hold30-prior",
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="Canonical M03R with the exit hazard frozen to the 30-session structural prior.",
    ),
    replace(
        _M03R_CANONICAL_SETTING,
        setting_index=9,
        setting_id="A09-no-long-context",
        slow_context_trading_sessions=63,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="Canonical M03R with learned temporal context truncated from 252 to 63 sessions.",
    ),
    replace(
        _M03R_CANONICAL_SETTING,
        setting_index=10,
        setting_id="A10-no-factor-neutral-projection",
        factor_sector_projection=False,
        promotion_eligible=False,
        ablation_of=M03R_CANONICAL_SETTING_ID,
        description="Canonical M03R without factor/sector-neutral risk projection.",
    ),
)

M03R_SETTING_IDS = tuple(setting.setting_id for setting in M03R_SETTINGS)
M03R_SETTINGS_BY_ID = {setting.setting_id: setting for setting in M03R_SETTINGS}


def _validate_inventory() -> None:
    if tuple(setting.setting_index for setting in M03R_SETTINGS) != tuple(range(11)):
        raise RuntimeError("M03R setting indexes must be contiguous from 0 through 10")
    if len(M03R_SETTINGS_BY_ID) != 11:
        raise RuntimeError("M03R setting IDs must be unique")
    if [row.setting_id for row in M03R_SETTINGS if row.promotion_eligible] != [
        M03R_CANONICAL_SETTING_ID
    ]:
        raise RuntimeError("M03R must have exactly one promotion candidate")

    canonical = M03R_SETTINGS_BY_ID[M03R_CANONICAL_SETTING_ID]
    causal_fields = (
        "objective_mode",
        "age_aware_holding",
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
    )
    expected_single_difference = {
        "A04-no-downside-score-adjustment": "use_downside_adjusted_stock_score",
        "A05-fixed-te-floor": "annual_tracking_error_floor",
        "A06-sharpe-overlay": "sharpe_mode",
        "A07-direct-sharpe": "sharpe_mode",
        "A08-fixed-exit-hazard": "exit_hazard_mode",
        "A09-no-long-context": "slow_context_trading_sessions",
        "A10-no-factor-neutral-projection": "factor_sector_projection",
    }
    for setting_id, expected_field in expected_single_difference.items():
        setting = M03R_SETTINGS_BY_ID[setting_id]
        changed = tuple(
            field
            for field in causal_fields
            if getattr(setting, field) != getattr(canonical, field)
        )
        if changed != (expected_field,):
            raise RuntimeError(
                f"{setting_id} must differ from canonical only in {expected_field}; "
                f"observed {changed}"
            )


_validate_inventory()


def resolve_m03r_v5_setting(setting_id: str) -> M03RSetting:
    """Resolve only an exact M03R v5 setting identity; aliases fail closed.

    Callers outside this generation must use this generation-qualified name.
    V4 intentionally retains its historical unsuffixed resolver in the frozen
    :mod:`hold30_alpha_m03r` module, so importing an unqualified resolver into
    shared model code is ambiguous and prohibited for new code.
    """

    try:
        return M03R_SETTINGS_BY_ID[setting_id]
    except KeyError as exc:
        if setting_id.startswith("hold30a-"):
            raise M03RProtocolError(
                f"V3 setting {setting_id!r} cannot identify an M03R artifact"
            ) from exc
        valid = ", ".join(M03R_SETTING_IDS)
        raise M03RProtocolError(
            f"unknown M03R setting {setting_id!r}; expected one of: {valid}"
        ) from exc


def validate_m03r_artifact_identity(
    *, protocol_generation: str, design_id: str, setting_id: str
) -> M03RSetting:
    """Reject V3 or aliased identities before an M03R artifact is produced."""

    if protocol_generation != M03R_PROTOCOL_GENERATION:
        if protocol_generation == M03R_SUPERSEDED_PROTOCOL_GENERATION:
            raise M03RProtocolError(
                "M03R v4 remains an immutable audit generation and cannot identify v5"
            )
        if protocol_generation == M03R_V3_AUDIT_PROTOCOL_GENERATION:
            raise M03RProtocolError(
                "V3 remains an immutable audit generation and cannot identify M03R"
            )
        raise M03RProtocolError(
            f"protocol_generation must be {M03R_PROTOCOL_GENERATION!r}"
        )
    if design_id != M03R_DESIGN_ID:
        raise M03RProtocolError(f"design_id must be {M03R_DESIGN_ID!r}")
    return resolve_m03r_v5_setting(setting_id)


def m03r_design_payload() -> dict[str, Any]:
    """Return the stable JSON-ready M03R design and causal inventory."""

    return {
        "schema_version": M03R_V5_SCHEMA_VERSION,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "supersedes_protocol_generation": M03R_SUPERSEDED_PROTOCOL_GENERATION,
        "v4_artifacts_retain_their_original_identity": True,
        "v3_audit_protocol_generation": M03R_V3_AUDIT_PROTOCOL_GENERATION,
        "v3_artifacts_retain_their_original_identity": True,
        "canonical_setting_id": M03R_CANONICAL_SETTING_ID,
        "design": asdict(M03R_DESIGN),
        "settings": [asdict(setting) for setting in M03R_SETTINGS],
        "required_launch_bindings": (
            "point_in_time_factor_manifest_sha256",
            "point_in_time_sector_manifest_sha256",
            "factor_sector_exposure_bounds_manifest_sha256",
            "m03r_risk_manifest_sha256",
            "m03r_seed_checkpoint_ensemble_manifest_sha256",
            "per_seed_confidence_calibration_manifest_sha256s",
            "per_seed_confidence_source_score_array_sha256s",
            "per_seed_confidence_source_target_array_sha256s",
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
    "M03R_SUPERSEDED_PROTOCOL_GENERATION",
    "M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS",
    "M03R_V3_AUDIT_PROTOCOL_GENERATION",
    "M03R_V5_PROTOCOL_GENERATION",
    "M03R_V5_SCHEMA_VERSION",
    "M03R_VALIDATION_ONE_WAY_COSTS_BASIS_POINTS",
    "M03RActiveRiskContract",
    "M03RDesign",
    "M03REnsembleExecutionContract",
    "M03RFactorSectorProjectionContract",
    "M03RModelContract",
    "M03RProtocolError",
    "M03RSetting",
    "M03RTemporalContract",
    "m03r_design_payload",
    "resolve_m03r_v5_setting",
    "validate_m03r_artifact_identity",
]
