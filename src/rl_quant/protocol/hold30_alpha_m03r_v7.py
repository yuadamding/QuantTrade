"""Immutable M03R v7 active-alpha soft-persistence experiment protocol.

V7 freezes the twelve-setting development panel separately from M03R v6.  It
does not relabel v6 artifacts and it does not authorize training or H100 use.
Every primary row is the canonical policy with exactly one declared causal
field changed.  The gradient-null M01 check and rejected fixed-TE-floor A05
diagnostic are deliberately outside the primary panel.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_PROTOCOL_GENERATION as M03R_V6_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_freeze import HOLD30_FOLDS, HOLD30_SEEDS

M03R_V7_PROTOCOL_GENERATION = "prelockbox-hold30-active-alpha-m03r-v7"
M03R_PROTOCOL_GENERATION = M03R_V7_PROTOCOL_GENERATION
M03R_V7_SCHEMA_VERSION = 7
M03R_V7_SUPERSEDED_PROTOCOL_GENERATION = M03R_V6_PROTOCOL_GENERATION
M03R_V7_DESIGN_ID = "daily_raw_pit300_hold30_m03r_v7"
M03R_DESIGN_ID = M03R_V7_DESIGN_ID
M03R_V7_CANONICAL_SETTING_ID = "M03R-soft-persistence-active-alpha-hold30-v7"
M03R_CANONICAL_SETTING_ID = M03R_V7_CANONICAL_SETTING_ID
M03R_V7_PRIMARY_BENCHMARK_ID = "C1-monthly-pit-active300-equal-weight-buy-and-drift"
M03R_V7_UNIVERSE_ID = "point-in-time-active-300"

M03R_V7_PERSISTENCE_OBJECTIVE_SCHEMA = (
    "rl-quant.hold30.m03r-v7-nav-session-proportional-soft-persistence-v1"
)
M03R_V7_PERSISTENCE_NORMALIZATION = "valid-decision-session-count-only"
M03R_V7_PERSISTENCE_AGE_WEIGHT_FORMULA = "max(0,1-age/30)^2"

M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE = "calibrated-confidence-0-to-4pct"
M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE = "fixed-2pct"

M03R_V7_PRIMARY_SETTING_IDS = (
    M03R_V7_CANONICAL_SETTING_ID,
    "P00-no-soft-persistence-v7",
    "P10-soft-persistence-10bp-v7",
    "A08-fixed-exit-hazard-v7",
    "A11-no-exact-hold-atom-v7",
    "A09-no-long-context-v7",
    "M02-active-risk-no-alpha-heads-v7",
    "A04-no-downside-score-adjustment-v7",
    "A12-fixed-2pct-active-risk-budget-v7",
    "A10-no-factor-neutral-projection-v7",
    "A06-sharpe-overlay-v7",
    "A07-direct-sharpe-v7",
)
M03R_V7_SETTING_IDS = M03R_V7_PRIMARY_SETTING_IDS
M03R_SETTING_IDS = M03R_V7_PRIMARY_SETTING_IDS

M03R_V7_M00_QUALIFICATION_SETTING_ID = "M00-absolute-return-v7"
M03R_V7_M01_QUALIFICATION_SETTING_ID = "M01-benchmark-subtraction-v7"
M03R_V7_A05_RESERVE_SETTING_ID = "A05-fixed-te-floor-v7"


class M03RV7ProtocolError(ValueError):
    """A v7 identity or scientific invariant is inconsistent."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV7PersistenceObjectiveContract:
    """One-sided persistence cost proportional to young sold NAV.

    The denominator is the valid decision-session count, never total sold
    notional.  Mature sales therefore have exactly zero value and gradient and
    cannot dilute a young-sale penalty.
    """

    objective_schema: str = M03R_V7_PERSISTENCE_OBJECTIVE_SCHEMA
    age_ledger_bin_count: int = 61
    minimum_age_sessions: int = 0
    maximum_age_sessions: int = 60
    maximum_bin_accumulates_older_notional: bool = True
    preference_horizon_sessions: int = 30
    age_weight_formula: str = M03R_V7_PERSISTENCE_AGE_WEIGHT_FORMULA
    normalization: str = M03R_V7_PERSISTENCE_NORMALIZATION
    denominator_uses_total_sold_notional: bool = False
    forced_and_unavailable_exits_exempt: bool = True
    warmup_shape: str = "linear-from-zero"
    warmup_fraction_of_optimizer_updates: float = 0.10
    canonical_age_zero_coefficient_basis_points: float = 5.0
    primary_panel_coefficients_basis_points: tuple[float, ...] = (0.0, 5.0, 10.0)
    minimum_holding_period_sessions: int | None = None
    pre_horizon_sell_mask: bool = False
    forced_expiry_at_preference_horizon: bool = False
    holding_duration_is_promotion_gate: bool = False

    def __post_init__(self) -> None:
        if self.objective_schema != M03R_V7_PERSISTENCE_OBJECTIVE_SCHEMA:
            raise M03RV7ProtocolError("v7 persistence objective schema drifted")
        if (
            self.age_ledger_bin_count != 61
            or self.minimum_age_sessions != 0
            or self.maximum_age_sessions != 60
            or not self.maximum_bin_accumulates_older_notional
        ):
            raise M03RV7ProtocolError("v7 requires exactly 61 age bins, ages 0-60")
        if (
            self.preference_horizon_sessions != 30
            or self.age_weight_formula != M03R_V7_PERSISTENCE_AGE_WEIGHT_FORMULA
        ):
            raise M03RV7ProtocolError("v7 soft-persistence age rule drifted")
        if (
            self.normalization != M03R_V7_PERSISTENCE_NORMALIZATION
            or self.denominator_uses_total_sold_notional
        ):
            raise M03RV7ProtocolError(
                "v7 persistence must be NAV/session proportional without a sold-"
                "notional denominator"
            )
        if not self.forced_and_unavailable_exits_exempt:
            raise M03RV7ProtocolError("forced and unavailable exits must be exempt")
        if (
            self.warmup_shape != "linear-from-zero"
            or self.warmup_fraction_of_optimizer_updates != 0.10
        ):
            raise M03RV7ProtocolError("v7 requires a frozen 10% linear warmup")
        if (
            self.canonical_age_zero_coefficient_basis_points != 5.0
            or self.primary_panel_coefficients_basis_points != (0.0, 5.0, 10.0)
        ):
            raise M03RV7ProtocolError("v7 persistence coefficient inventory drifted")
        if (
            self.minimum_holding_period_sessions is not None
            or self.pre_horizon_sell_mask
            or self.forced_expiry_at_preference_horizon
            or self.holding_duration_is_promotion_gate
        ):
            raise M03RV7ProtocolError(
                "30 sessions is a soft preference, never a sell rule or promotion gate"
            )

    def age_weight(self, age_sessions: float) -> float:
        """Return the frozen one-sided quadratic sale-age weight."""

        if not math.isfinite(age_sessions) or age_sessions < 0.0:
            raise M03RV7ProtocolError("age_sessions must be finite and nonnegative")
        fraction = max(0.0, 1.0 - age_sessions / self.preference_horizon_sessions)
        return fraction * fraction

    def warmup_multiplier(self, completed_update_fraction: float) -> float:
        """Return the frozen linear warmup multiplier."""

        if (
            not math.isfinite(completed_update_fraction)
            or not 0.0 <= completed_update_fraction <= 1.0
        ):
            raise M03RV7ProtocolError(
                "completed_update_fraction must be finite and in [0,1]"
            )
        return min(
            1.0,
            completed_update_fraction / self.warmup_fraction_of_optimizer_updates,
        )

    @staticmethod
    def coefficient_as_return(coefficient_basis_points: float) -> float:
        """Convert a bound age-zero basis-point coefficient to return units."""

        if coefficient_basis_points not in (0.0, 5.0, 10.0):
            raise M03RV7ProtocolError(
                "v7 primary persistence coefficient must be 0, 5, or 10 bp"
            )
        return coefficient_basis_points / 10_000.0


@dataclass(frozen=True, slots=True)
class M03RV7SharedConfiguration:
    """Result-moving values shared by every primary v7 setting."""

    universe_id: str = M03R_V7_UNIVERSE_ID
    primary_benchmark_id: str = M03R_V7_PRIMARY_BENCHMARK_ID
    decisions_per_trading_session: int = 1
    recent_raw_context_trading_sessions: int = 42
    canonical_learned_temporal_context_trading_sessions: int = 252
    rollout_trading_sessions: int = 63
    economic_credit_post_fill_return_count: int = 30
    auxiliary_horizons_trading_sessions: tuple[int, ...] = (5, 21, 30, 63)
    training_one_way_cost_basis_points: int = 20
    validation_one_way_costs_basis_points: tuple[int, ...] = (10, 20, 40)
    annual_tracking_error_floor: float | None = None
    annual_tracking_error_ceiling: float = 0.06
    active_market_beta_target: float = 0.0
    active_beta_equivalence_absolute_upper_bound: float = 0.10
    maximum_stock_weight_fraction: float = 0.01
    calibrated_active_risk_budget_range_annualized: tuple[float, float] = (0.0, 0.04)
    fixed_active_risk_ablation_budget_annualized: float = 0.02
    age_state_bin_count: int = 61
    exact_hold_action_available_but_optional: bool = True
    canonical_sharpe_mode: str = "none"
    validation_fold_count: int = HOLD30_FOLDS
    paired_seeds: tuple[int, ...] = HOLD30_SEEDS
    seed_outputs_ensembled_before_chronological_inference: bool = True
    identical_setting_order_within_each_fold_seed: bool = True
    identical_data_stream_across_settings: bool = True
    identical_initialization_convention_across_settings: bool = True
    identical_calibration_procedure_across_settings: bool = True
    identical_optimizer_schedule_across_settings: bool = True
    persistence: M03RV7PersistenceObjectiveContract = (
        M03RV7PersistenceObjectiveContract()
    )

    def __post_init__(self) -> None:
        expected = (
            self.universe_id,
            self.primary_benchmark_id,
            self.decisions_per_trading_session,
            self.recent_raw_context_trading_sessions,
            self.canonical_learned_temporal_context_trading_sessions,
            self.rollout_trading_sessions,
            self.economic_credit_post_fill_return_count,
            self.auxiliary_horizons_trading_sessions,
            self.training_one_way_cost_basis_points,
            self.validation_one_way_costs_basis_points,
            self.annual_tracking_error_floor,
            self.annual_tracking_error_ceiling,
            self.active_market_beta_target,
            self.active_beta_equivalence_absolute_upper_bound,
            self.maximum_stock_weight_fraction,
            self.calibrated_active_risk_budget_range_annualized,
            self.fixed_active_risk_ablation_budget_annualized,
            self.age_state_bin_count,
            self.exact_hold_action_available_but_optional,
            self.canonical_sharpe_mode,
            self.validation_fold_count,
            self.paired_seeds,
            self.seed_outputs_ensembled_before_chronological_inference,
            self.identical_setting_order_within_each_fold_seed,
            self.identical_data_stream_across_settings,
            self.identical_initialization_convention_across_settings,
            self.identical_calibration_procedure_across_settings,
            self.identical_optimizer_schedule_across_settings,
        )
        required = (
            M03R_V7_UNIVERSE_ID,
            M03R_V7_PRIMARY_BENCHMARK_ID,
            1,
            42,
            252,
            63,
            30,
            (5, 21, 30, 63),
            20,
            (10, 20, 40),
            None,
            0.06,
            0.0,
            0.10,
            0.01,
            (0.0, 0.04),
            0.02,
            61,
            True,
            "none",
            6,
            (17, 29, 43, 71, 101),
            True,
            True,
            True,
            True,
            True,
            True,
        )
        if expected != required:
            raise M03RV7ProtocolError("v7 shared canonical configuration drifted")


M03R_V7_SHARED_CONFIGURATION = M03RV7SharedConfiguration()
M03R_V7_SHARED_CONFIGURATION_SHA256 = _sha256_payload(
    asdict(M03R_V7_SHARED_CONFIGURATION)
)


@dataclass(frozen=True, slots=True)
class M03RV7Setting:
    """One primary-panel setting with exactly one causal delta from canonical."""

    setting_index: int
    setting_id: str
    persistence_coefficient_basis_points: float
    exit_hazard_mode: str
    exact_hold_action_supported: bool
    learned_temporal_context_trading_sessions: int
    residual_alpha_head_mode: str
    active_risk_budget_mode: str
    factor_sector_neutral_projection: bool
    sharpe_mode: str
    promotion_eligible: bool
    ablation_of: str | None
    declared_causal_field: str | None
    scientific_question: str

    def __post_init__(self) -> None:
        if self.setting_id not in M03R_V7_PRIMARY_SETTING_IDS:
            raise M03RV7ProtocolError("unknown v7 primary setting ID")
        M03R_V7_SHARED_CONFIGURATION.persistence.coefficient_as_return(
            self.persistence_coefficient_basis_points
        )
        if self.exit_hazard_mode not in {
            "learned-age-aware",
            "fixed-structural-30-session-prior",
        }:
            raise M03RV7ProtocolError("unknown v7 exit-hazard mode")
        if self.learned_temporal_context_trading_sessions not in {63, 252}:
            raise M03RV7ProtocolError("v7 learned context must be 63 or 252 sessions")
        if self.residual_alpha_head_mode not in {
            "none",
            "mean-only",
            "mean-and-downside",
        }:
            raise M03RV7ProtocolError("unknown v7 residual-alpha-head mode")
        if self.active_risk_budget_mode not in {
            M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE,
            M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE,
        }:
            raise M03RV7ProtocolError("unknown v7 active-risk-budget mode")
        if self.sharpe_mode not in {
            "none",
            "separate-total-risk-overlay",
            "direct-full-batch-two-pass-gradient",
        }:
            raise M03RV7ProtocolError("unknown v7 Sharpe mode")
        if not self.scientific_question:
            raise M03RV7ProtocolError("v7 scientific question is required")
        if self.promotion_eligible != (self.setting_id == M03R_V7_CANONICAL_SETTING_ID):
            raise M03RV7ProtocolError("only canonical v7 may be promotion eligible")
        if self.setting_id == M03R_V7_CANONICAL_SETTING_ID:
            if self.ablation_of is not None or self.declared_causal_field is not None:
                raise M03RV7ProtocolError("canonical v7 cannot declare an ablation")
        elif (
            self.ablation_of != M03R_V7_CANONICAL_SETTING_ID
            or self.declared_causal_field is None
        ):
            raise M03RV7ProtocolError(
                "every noncanonical v7 primary row must bind its canonical ablation"
            )

    @property
    def residual_alpha_heads(self) -> bool:
        return self.residual_alpha_head_mode != "none"

    @property
    def downside_score_adjustment(self) -> bool:
        return self.residual_alpha_head_mode == "mean-and-downside"

    @property
    def emits_exact_hold_action(self) -> bool:
        return bool(
            self.exact_hold_action_supported
            and self.exit_hazard_mode == "learned-age-aware"
        )

    def preferred_active_risk_budget_annualized(
        self,
        calibrated_confidence: float,
    ) -> float:
        if (
            not math.isfinite(calibrated_confidence)
            or not 0.0 <= calibrated_confidence <= 1.0
        ):
            raise M03RV7ProtocolError(
                "calibrated_confidence must be finite and in [0,1]"
            )
        if self.active_risk_budget_mode == (M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE):
            return 0.02
        return 0.04 * calibrated_confidence


M03R_V7_CAUSAL_FIELDS = (
    "persistence_coefficient_basis_points",
    "exit_hazard_mode",
    "exact_hold_action_supported",
    "learned_temporal_context_trading_sessions",
    "residual_alpha_head_mode",
    "active_risk_budget_mode",
    "factor_sector_neutral_projection",
    "sharpe_mode",
)

_CANONICAL = M03RV7Setting(
    setting_index=0,
    setting_id=M03R_V7_CANONICAL_SETTING_ID,
    persistence_coefficient_basis_points=5.0,
    exit_hazard_mode="learned-age-aware",
    exact_hold_action_supported=True,
    learned_temporal_context_trading_sessions=252,
    residual_alpha_head_mode="mean-and-downside",
    active_risk_budget_mode=M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE,
    factor_sector_neutral_projection=True,
    sharpe_mode="none",
    promotion_eligible=True,
    ablation_of=None,
    declared_causal_field=None,
    scientific_question=(
        "Can the complete policy produce cost-adjusted active alpha, acceptable "
        "Sharpe, and persistent but freely revisable positions?"
    ),
)

M03R_V7_PRIMARY_SETTINGS: tuple[M03RV7Setting, ...] = (
    _CANONICAL,
    replace(
        _CANONICAL,
        setting_index=1,
        setting_id="P00-no-soft-persistence-v7",
        persistence_coefficient_basis_points=0.0,
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="persistence_coefficient_basis_points",
        scientific_question=(
            "Does the explicit persistence penalty add value beyond costs, age "
            "state, and the structural hazard prior?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=2,
        setting_id="P10-soft-persistence-10bp-v7",
        persistence_coefficient_basis_points=10.0,
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="persistence_coefficient_basis_points",
        scientific_question=(
            "Does stronger persistence improve holding behavior or suppress useful "
            "early exits and alpha?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=3,
        setting_id="A08-fixed-exit-hazard-v7",
        exit_hazard_mode="fixed-structural-30-session-prior",
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="exit_hazard_mode",
        scientific_question=(
            "Does learned exit timing add value relative to the fixed structural "
            "30-session prior?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=4,
        setting_id="A11-no-exact-hold-atom-v7",
        exact_hold_action_supported=False,
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="exact_hold_action_supported",
        scientific_question=(
            "Does the exact no-trade atom add value beyond continuous hazard, "
            "costs, and the persistence penalty?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=5,
        setting_id="A09-no-long-context-v7",
        learned_temporal_context_trading_sessions=63,
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="learned_temporal_context_trading_sessions",
        scientific_question=(
            "Does 252-session context improve alpha, regime recognition, and "
            "one-month holding decisions?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=6,
        setting_id="M02-active-risk-no-alpha-heads-v7",
        residual_alpha_head_mode="none",
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="residual_alpha_head_mode",
        scientific_question=(
            "Do multi-horizon residual heads add stock-selection skill beyond the "
            "common encoder and risk controls?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=7,
        setting_id="A04-no-downside-score-adjustment-v7",
        residual_alpha_head_mode="mean-only",
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="residual_alpha_head_mode",
        scientific_question=(
            "Does downside-aware scoring improve Sharpe and drawdown without "
            "attenuating useful alpha?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=8,
        setting_id="A12-fixed-2pct-active-risk-budget-v7",
        active_risk_budget_mode=M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE,
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="active_risk_budget_mode",
        scientific_question=(
            "Does calibrated confidence sizing improve risk-adjusted return over "
            "a fixed 2% active-risk budget?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=9,
        setting_id="A10-no-factor-neutral-projection-v7",
        factor_sector_neutral_projection=False,
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="factor_sector_neutral_projection",
        scientific_question=(
            "How much apparent performance comes from systematic factor and "
            "sector tilts rather than stock selection?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=10,
        setting_id="A06-sharpe-overlay-v7",
        sharpe_mode="separate-total-risk-overlay",
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="sharpe_mode",
        scientific_question=(
            "Can a separate total-risk overlay improve investor-facing Sharpe "
            "without contaminating the alpha core?"
        ),
    ),
    replace(
        _CANONICAL,
        setting_index=11,
        setting_id="A07-direct-sharpe-v7",
        sharpe_mode="direct-full-batch-two-pass-gradient",
        promotion_eligible=False,
        ablation_of=M03R_V7_CANONICAL_SETTING_ID,
        declared_causal_field="sharpe_mode",
        scientific_question=(
            "Does direct Sharpe optimization add value or create unstable, "
            "overfit updates?"
        ),
    ),
)
M03R_V7_SETTINGS = M03R_V7_PRIMARY_SETTINGS
M03R_SETTINGS = M03R_V7_PRIMARY_SETTINGS
M03R_V7_PRIMARY_SETTINGS_BY_ID = {
    row.setting_id: row for row in M03R_V7_PRIMARY_SETTINGS
}
M03R_V7_SETTINGS_BY_ID = M03R_V7_PRIMARY_SETTINGS_BY_ID
M03R_SETTINGS_BY_ID = M03R_V7_PRIMARY_SETTINGS_BY_ID


@dataclass(frozen=True, slots=True)
class M03RV7GradientNullQualification:
    """Short deterministic M00/M01 parity check outside the primary panel."""

    reference_setting_id: str = M03R_V7_M00_QUALIFICATION_SETTING_ID
    benchmark_subtraction_setting_id: str = M03R_V7_M01_QUALIFICATION_SETTING_ID
    reference_objective: str = "negative-portfolio-net-log-return"
    benchmark_subtraction_objective: str = (
        "negative-portfolio-net-log-return-plus-detached-C1-net-log-return"
    )
    benchmark_term_is_parameter_independent_and_detached: bool = True
    maximum_optimizer_updates: int = 4
    minimum_optimizer_updates: int = 2
    same_seed_required: bool = True
    same_minibatches_required: bool = True
    same_checkpoint_initialization_required: bool = True
    same_optimizer_state_required: bool = True
    compare_gradients: bool = True
    compare_parameter_updates: bool = True
    compare_model_state_hashes: bool = True
    compare_optimizer_state_hashes: bool = True
    complete_checkpoint_or_receipt_hash_equality_required: bool = False
    primary_panel_member: bool = False
    full_fold_seed_study_authorized: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if (
            self.reference_setting_id != M03R_V7_M00_QUALIFICATION_SETTING_ID
            or self.benchmark_subtraction_setting_id
            != M03R_V7_M01_QUALIFICATION_SETTING_ID
            or self.minimum_optimizer_updates != 2
            or self.maximum_optimizer_updates != 4
            or self.reference_objective != "negative-portfolio-net-log-return"
            or self.benchmark_subtraction_objective
            != "negative-portfolio-net-log-return-plus-detached-C1-net-log-return"
            or not self.benchmark_term_is_parameter_independent_and_detached
        ):
            raise M03RV7ProtocolError("v7 M00/M01 qualification identity drifted")
        if not all(
            (
                self.same_seed_required,
                self.same_minibatches_required,
                self.same_checkpoint_initialization_required,
                self.same_optimizer_state_required,
                self.compare_gradients,
                self.compare_parameter_updates,
                self.compare_model_state_hashes,
                self.compare_optimizer_state_hashes,
            )
        ):
            raise M03RV7ProtocolError("v7 M00/M01 parity evidence is incomplete")
        if self.complete_checkpoint_or_receipt_hash_equality_required:
            raise M03RV7ProtocolError(
                "M00/M01 setting identity and logged loss intentionally make complete "
                "checkpoint and receipt hashes distinct"
            )
        if (
            self.primary_panel_member
            or self.full_fold_seed_study_authorized
            or self.promotion_eligible
        ):
            raise M03RV7ProtocolError("M01 is short qualification only")


@dataclass(frozen=True, slots=True)
class M03RV7FixedTEFloorReserve:
    """Rejected compulsory-risk mechanism retained as an inactive reserve."""

    setting_id: str = M03R_V7_A05_RESERVE_SETTING_ID
    ablation_of: str = M03R_V7_CANONICAL_SETTING_ID
    annual_tracking_error_floor: float = 0.02
    activation_condition: str = (
        "canonical-collapses-to-near-zero-active-risk-on-inner-development-folds"
    )
    primary_panel_member: bool = False
    automatically_scheduled: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if (
            self.setting_id != M03R_V7_A05_RESERVE_SETTING_ID
            or self.ablation_of != M03R_V7_CANONICAL_SETTING_ID
            or self.annual_tracking_error_floor != 0.02
            or self.activation_condition
            != "canonical-collapses-to-near-zero-active-risk-on-inner-development-folds"
        ):
            raise M03RV7ProtocolError("v7 A05 reserve identity drifted")
        if (
            self.primary_panel_member
            or self.automatically_scheduled
            or self.promotion_eligible
        ):
            raise M03RV7ProtocolError("A05 must remain an inactive reserve diagnostic")


M03R_V7_GRADIENT_NULL_QUALIFICATION = M03RV7GradientNullQualification()
M03R_V7_FIXED_TE_FLOOR_RESERVE = M03RV7FixedTEFloorReserve()


def _validate_primary_inventory() -> None:
    if tuple(row.setting_id for row in M03R_V7_PRIMARY_SETTINGS) != (
        M03R_V7_PRIMARY_SETTING_IDS
    ):
        raise RuntimeError("v7 primary setting order drifted")
    if tuple(row.setting_index for row in M03R_V7_PRIMARY_SETTINGS) != tuple(range(12)):
        raise RuntimeError("v7 primary setting indexes must be 0 through 11")
    if len(M03R_V7_PRIMARY_SETTINGS_BY_ID) != 12:
        raise RuntimeError("v7 primary setting IDs must be unique")
    if [
        row.setting_id for row in M03R_V7_PRIMARY_SETTINGS if row.promotion_eligible
    ] != [M03R_V7_CANONICAL_SETTING_ID]:
        raise RuntimeError("v7 must have exactly one promotion candidate")

    for row in M03R_V7_PRIMARY_SETTINGS[1:]:
        changed = tuple(
            field
            for field in M03R_V7_CAUSAL_FIELDS
            if getattr(row, field) != getattr(_CANONICAL, field)
        )
        if changed != (row.declared_causal_field,):
            raise RuntimeError(
                f"{row.setting_id} must differ only in "
                f"{row.declared_causal_field}; observed {changed}"
            )

    excluded = {
        M03R_V7_M00_QUALIFICATION_SETTING_ID,
        M03R_V7_M01_QUALIFICATION_SETTING_ID,
        M03R_V7_A05_RESERVE_SETTING_ID,
    }
    if excluded.intersection(M03R_V7_PRIMARY_SETTINGS_BY_ID):
        raise RuntimeError("v7 qualification/reserve rows entered the primary panel")


_validate_primary_inventory()

M03R_V7_PRIMARY_PANEL_SHA256 = _sha256_payload(
    {
        "protocol_generation": M03R_V7_PROTOCOL_GENERATION,
        "design_id": M03R_V7_DESIGN_ID,
        "shared_configuration_sha256": M03R_V7_SHARED_CONFIGURATION_SHA256,
        "primary_settings": [asdict(row) for row in M03R_V7_PRIMARY_SETTINGS],
    }
)
M03R_V7_AUXILIARY_CONTROLS_SHA256 = _sha256_payload(
    {
        "gradient_null_qualification": asdict(M03R_V7_GRADIENT_NULL_QUALIFICATION),
        "fixed_te_floor_reserve": asdict(M03R_V7_FIXED_TE_FLOOR_RESERVE),
    }
)


def resolve_m03r_v7_setting(setting_id: str) -> M03RV7Setting:
    """Resolve only a primary v7 setting; qualification/reserve IDs fail closed."""

    try:
        return M03R_V7_PRIMARY_SETTINGS_BY_ID[setting_id]
    except KeyError as exc:
        raise M03RV7ProtocolError(
            f"unknown primary M03R v7 setting {setting_id!r}"
        ) from exc


def validate_m03r_v7_artifact_identity(
    *,
    protocol_generation: str,
    design_id: str,
    setting_id: str,
) -> M03RV7Setting:
    """Validate a primary v7 artifact without accepting v6 identities."""

    if protocol_generation != M03R_V7_PROTOCOL_GENERATION:
        if protocol_generation == M03R_V6_PROTOCOL_GENERATION:
            raise M03RV7ProtocolError(
                "M03R v6 remains immutable and cannot identify a v7 artifact"
            )
        raise M03RV7ProtocolError(
            f"protocol_generation must be {M03R_V7_PROTOCOL_GENERATION!r}"
        )
    if design_id != M03R_V7_DESIGN_ID:
        raise M03RV7ProtocolError(f"design_id must be {M03R_V7_DESIGN_ID!r}")
    return resolve_m03r_v7_setting(setting_id)


def m03r_v7_protocol_payload() -> dict[str, Any]:
    """Return the stable JSON-ready v7 protocol and fail-closed launch state."""

    return {
        "schema_version": M03R_V7_SCHEMA_VERSION,
        "protocol_generation": M03R_V7_PROTOCOL_GENERATION,
        "design_id": M03R_V7_DESIGN_ID,
        "supersedes_protocol_generation": M03R_V7_SUPERSEDED_PROTOCOL_GENERATION,
        "v6_artifacts_retain_their_original_identity": True,
        "canonical_setting_id": M03R_V7_CANONICAL_SETTING_ID,
        "shared_configuration": asdict(M03R_V7_SHARED_CONFIGURATION),
        "shared_configuration_sha256": M03R_V7_SHARED_CONFIGURATION_SHA256,
        "primary_settings": [asdict(row) for row in M03R_V7_PRIMARY_SETTINGS],
        "primary_panel_sha256": M03R_V7_PRIMARY_PANEL_SHA256,
        "gradient_null_qualification": asdict(M03R_V7_GRADIENT_NULL_QUALIFICATION),
        "fixed_te_floor_reserve": asdict(M03R_V7_FIXED_TE_FLOOR_RESERVE),
        "auxiliary_controls_sha256": M03R_V7_AUXILIARY_CONTROLS_SHA256,
        "launch_authorized": False,
        "h100_training_authorized": False,
        "launch_blockers": (
            "governed-v7-twelve-setting-production-driver-not-implemented",
            "v7-setting-model-and-training-routes-not-implemented",
            "v7-ensemble-projection-selection-and-evaluator-adapters-not-qualified",
            "point-in-time-active300-data-and-inference-family-not-sealed",
            "cuda-two-rank-restart-and-h100-capacity-not-qualified",
        ),
    }


__all__ = [
    "M03R_CANONICAL_SETTING_ID",
    "M03R_DESIGN_ID",
    "M03R_PROTOCOL_GENERATION",
    "M03R_SETTINGS",
    "M03R_SETTINGS_BY_ID",
    "M03R_SETTING_IDS",
    "M03R_V6_PROTOCOL_GENERATION",
    "M03R_V7_A05_RESERVE_SETTING_ID",
    "M03R_V7_AUXILIARY_CONTROLS_SHA256",
    "M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE",
    "M03R_V7_CANONICAL_SETTING_ID",
    "M03R_V7_CAUSAL_FIELDS",
    "M03R_V7_DESIGN_ID",
    "M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE",
    "M03R_V7_FIXED_TE_FLOOR_RESERVE",
    "M03R_V7_GRADIENT_NULL_QUALIFICATION",
    "M03R_V7_M00_QUALIFICATION_SETTING_ID",
    "M03R_V7_M01_QUALIFICATION_SETTING_ID",
    "M03R_V7_PERSISTENCE_AGE_WEIGHT_FORMULA",
    "M03R_V7_PERSISTENCE_NORMALIZATION",
    "M03R_V7_PERSISTENCE_OBJECTIVE_SCHEMA",
    "M03R_V7_PRIMARY_PANEL_SHA256",
    "M03R_V7_PRIMARY_SETTINGS",
    "M03R_V7_PRIMARY_SETTINGS_BY_ID",
    "M03R_V7_PRIMARY_SETTING_IDS",
    "M03R_V7_PROTOCOL_GENERATION",
    "M03R_V7_SCHEMA_VERSION",
    "M03R_V7_SETTINGS",
    "M03R_V7_SETTINGS_BY_ID",
    "M03R_V7_SETTING_IDS",
    "M03R_V7_SHARED_CONFIGURATION",
    "M03R_V7_SHARED_CONFIGURATION_SHA256",
    "M03R_V7_SUPERSEDED_PROTOCOL_GENERATION",
    "M03R_V7_UNIVERSE_ID",
    "M03RV7FixedTEFloorReserve",
    "M03RV7GradientNullQualification",
    "M03RV7PersistenceObjectiveContract",
    "M03RV7ProtocolError",
    "M03RV7Setting",
    "M03RV7SharedConfiguration",
    "m03r_v7_protocol_payload",
    "resolve_m03r_v7_setting",
    "validate_m03r_v7_artifact_identity",
]
