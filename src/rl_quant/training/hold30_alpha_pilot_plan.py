"""Pilot-only coefficient freeze for the Hold-30 alpha V3 mechanism screen.

The canonical protocol deliberately keeps result-moving coefficients unset.
This module leaves that audit record unchanged and supplies one separately
content-addressed, pre-lockbox pilot profile.  It keeps A06 non-executable
unless every result-moving field validates against this prelaunch scientific
profile; no pilot coefficient may be selected from outcome evidence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Final

from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_HORIZONS,
    HOLD30_ALPHA_TE_MAX_ANNUAL,
    HOLD30_ALPHA_TE_MIN_ANNUAL,
    HOLD30_ALPHA_TE_TARGET_ANNUAL,
    HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
)
from rl_quant.protocol.hold30_freeze import sha256_payload
from rl_quant.training.hold30_alpha import (
    Hold30AlphaObjectiveConfig,
    Hold30AlphaTrainingError,
)
from rl_quant.training.hold30_alpha_plan import (
    HOLD30_ALPHA_CONFIG_SETTING_IDS,
    Hold30AlphaTrainingPlan,
)

HOLD30_ALPHA_PILOT_PROFILE_SCHEMA = "rl-quant.hold30-alpha-pilot-profile-v1"
HOLD30_ALPHA_PILOT_QUALIFICATION_SCHEMA = (
    "rl-quant.hold30-alpha-pilot-plan-qualification-v1"
)
(
    _M02,
    _M03,
    _A04,
    _A05,
    _A06,
    _A07,
) = HOLD30_ALPHA_CONFIG_SETTING_IDS


class Hold30AlphaPilotPlanError(ValueError):
    """The pilot profile, runtime binding, or qualification has drifted."""


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30AlphaPilotPlanError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


@dataclass(frozen=True, slots=True)
class Hold30AlphaPilotProfile:
    """Exact scientific choices for the non-promotional pre-lockbox pilot."""

    auxiliary_horizon_weights: tuple[float, float, float, float]
    auxiliary_horizon_scales: tuple[float, float, float, float]
    active_log_scale_bounds: tuple[float, float]
    uncertainty_log_scale_bounds: tuple[float, float]
    downside_penalty_kappa: float
    lambda_te_floor: float
    lambda_te_ceiling: float
    lambda_beta: float
    lambda_turnover: float
    lambda_early_exit: float
    lambda_auxiliary_alpha: float
    lambda_uncertainty: float
    projection_distance_max: float
    projection_distance_statistic: str
    forced_turnover_fraction_max: float
    forced_turnover_fraction_statistic: str
    lambda_total_excess_mean: float
    lambda_total_sharpe_overlay: float
    lambda_volatility_ratio: float
    target_volatility_ratio: float
    lambda_drawdown: float
    drawdown_limit_log: float
    a06_total_risk_step: float
    total_sharpe_epsilon: float
    lambda_direct_sharpe: float
    direct_sharpe_epsilon: float
    assumptions: tuple[str, ...]
    schema: str = HOLD30_ALPHA_PILOT_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HOLD30_ALPHA_PILOT_PROFILE_SCHEMA:
            raise Hold30AlphaPilotPlanError("pilot profile schema drifted")
        if (
            not isinstance(self.auxiliary_horizon_weights, tuple)
            or len(self.auxiliary_horizon_weights) != len(HOLD30_ALPHA_HORIZONS)
            or any(value <= 0.0 for value in self.auxiliary_horizon_weights)
            or not math.isclose(
                sum(self.auxiliary_horizon_weights), 1.0, abs_tol=1e-12
            )
        ):
            raise Hold30AlphaPilotPlanError(
                "pilot horizon weights must be four positive values summing to one"
            )
        primary = HOLD30_ALPHA_HORIZONS.index(30)
        if any(
            self.auxiliary_horizon_weights[primary] <= value
            for index, value in enumerate(self.auxiliary_horizon_weights)
            if index != primary
        ):
            raise Hold30AlphaPilotPlanError(
                "the pilot 30-session horizon weight must be uniquely largest"
            )
        if (
            not isinstance(self.auxiliary_horizon_scales, tuple)
            or len(self.auxiliary_horizon_scales) != len(HOLD30_ALPHA_HORIZONS)
            or any(value <= 0.0 for value in self.auxiliary_horizon_scales)
            or any(
                right <= left
                for left, right in zip(
                    self.auxiliary_horizon_scales,
                    self.auxiliary_horizon_scales[1:],
                )
            )
        ):
            raise Hold30AlphaPilotPlanError(
                "pilot horizon scales must be four strictly increasing positives"
            )
        expected_action_bounds = (
            math.log(HOLD30_ALPHA_TE_MIN_ANNUAL / HOLD30_ALPHA_TE_TARGET_ANNUAL),
            math.log(HOLD30_ALPHA_TE_MAX_ANNUAL / HOLD30_ALPHA_TE_TARGET_ANNUAL),
        )
        if self.active_log_scale_bounds != expected_action_bounds:
            raise Hold30AlphaPilotPlanError(
                "active action bounds must map the 4% scale exactly into the 2%-6% TE band"
            )
        if (
            self.uncertainty_log_scale_bounds[0]
            != math.log(0.01)
            or self.uncertainty_log_scale_bounds[1] != 0.0
        ):
            raise Hold30AlphaPilotPlanError(
                "uncertainty log-scale bounds must encode a 1%-100% 30d scale"
            )
        scalar_values = (
            self.downside_penalty_kappa,
            self.lambda_te_floor,
            self.lambda_te_ceiling,
            self.lambda_beta,
            self.lambda_turnover,
            self.lambda_early_exit,
            self.lambda_auxiliary_alpha,
            self.lambda_uncertainty,
            self.projection_distance_max,
            self.forced_turnover_fraction_max,
            self.lambda_total_excess_mean,
            self.lambda_total_sharpe_overlay,
            self.lambda_volatility_ratio,
            self.target_volatility_ratio,
            self.lambda_drawdown,
            self.drawdown_limit_log,
            self.a06_total_risk_step,
            self.total_sharpe_epsilon,
            self.lambda_direct_sharpe,
            self.direct_sharpe_epsilon,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in scalar_values
        ):
            raise Hold30AlphaPilotPlanError(
                "all pilot coefficients, thresholds, and scales must be finite positives"
            )
        if self.projection_distance_max > 1.0:
            raise Hold30AlphaPilotPlanError("projection threshold cannot exceed NAV")
        if self.forced_turnover_fraction_max > 1.0:
            raise Hold30AlphaPilotPlanError(
                "forced-turnover fraction threshold cannot exceed one"
            )
        if self.projection_distance_statistic != (
            "mean-requested-to-constructed-one-way-weight-distance-over-scored-decisions"
        ):
            raise Hold30AlphaPilotPlanError("pilot projection statistic drifted")
        if self.forced_turnover_fraction_statistic != (
            "forced-one-way-turnover/(forced-one-way-turnover+discretionary-one-way-turnover);"
            "zero-denominator-is-zero"
        ):
            raise Hold30AlphaPilotPlanError(
                "pilot forced-turnover statistic drifted"
            )
        if (
            not isinstance(self.assumptions, tuple)
            or not self.assumptions
            or len(set(self.assumptions)) != len(self.assumptions)
            or any(not isinstance(value, str) or not value for value in self.assumptions)
        ):
            raise Hold30AlphaPilotPlanError(
                "pilot assumptions must be explicit, nonempty, and unique"
            )

    def manifest_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_id(self) -> str:
        return sha256_payload(self.manifest_payload())


HOLD30_ALPHA_PILOT_PROFILE: Final = Hold30AlphaPilotProfile(
    auxiliary_horizon_weights=(0.10, 0.20, 0.50, 0.20),
    auxiliary_horizon_scales=tuple(
        0.02 * math.sqrt(horizon) for horizon in HOLD30_ALPHA_HORIZONS
    ),
    active_log_scale_bounds=(
        math.log(HOLD30_ALPHA_TE_MIN_ANNUAL / HOLD30_ALPHA_TE_TARGET_ANNUAL),
        math.log(HOLD30_ALPHA_TE_MAX_ANNUAL / HOLD30_ALPHA_TE_TARGET_ANNUAL),
    ),
    uncertainty_log_scale_bounds=(math.log(0.01), 0.0),
    downside_penalty_kappa=0.25,
    # A full 2%-point TE-band breach costs one/two reference bp per day.
    lambda_te_floor=0.25,
    lambda_te_ceiling=0.50,
    # A 0.10 beta miss costs one reference bp per day.
    lambda_beta=0.01,
    # Two percentage points above 1/30 turnover costs one reference bp/day.
    lambda_turnover=0.25,
    # Five percent of NAV in fully weighted early exits costs one bp/day.
    lambda_early_exit=0.002,
    lambda_auxiliary_alpha=0.0001,
    lambda_uncertainty=0.00005,
    projection_distance_max=0.01,
    projection_distance_statistic=(
        "mean-requested-to-constructed-one-way-weight-distance-over-scored-decisions"
    ),
    forced_turnover_fraction_max=0.10,
    forced_turnover_fraction_statistic=(
        "forced-one-way-turnover/(forced-one-way-turnover+discretionary-one-way-turnover);"
        "zero-denominator-is-zero"
    ),
    lambda_total_excess_mean=1.0,
    lambda_total_sharpe_overlay=0.00005,
    # A 0.10 volatility-ratio miss costs one reference bp/day.
    lambda_volatility_ratio=0.01,
    target_volatility_ratio=1.0,
    # A 0.05 log-drawdown breach costs one reference bp/day.
    lambda_drawdown=0.04,
    drawdown_limit_log=-math.log(0.85),
    a06_total_risk_step=0.05,
    # Same 10-bp daily-volatility floor used by A07. This is negligible at
    # ordinary equity volatility but stabilizes near-zero Sharpe denominators.
    total_sharpe_epsilon=1e-6,
    lambda_direct_sharpe=0.00005,
    # Variance floor corresponding to 10 bp daily standard deviation.
    direct_sharpe_epsilon=1e-6,
    assumptions=(
        "auxiliary labels are unstandardized cumulative benchmark-relative log returns",
        "a 2% daily idiosyncratic-volatility prior sets sqrt-horizon target scales",
        "one basis point of daily objective value is the penalty calibration reference",
        "discretionary turnover excludes startup, forced, and terminal transactions",
        "early-exit mass is age-weighted sold notional normalized by portfolio NAV",
        "checkpoint thresholds use inner-validation evidence only and never outer returns",
        "the A06 optimizer-spec digest freezes topology and hyperparameters, not mutable state",
        "A06 and A07 use the same 1e-6 variance floor, equivalent to a 10 bp daily-volatility floor",
        "the A06 floor is negligible at normal equity volatility and stabilizes near-zero Sharpe denominators",
        "the A06 floor was frozen before launch without inspecting training or evaluation outcomes",
    ),
)
HOLD30_ALPHA_PILOT_PROFILE_RECEIPT_SHA256: Final = (
    "7cb98970c93bc4e8cd59c49cc09b1b7883025ff700acec3783453585c7084752"
)
if HOLD30_ALPHA_PILOT_PROFILE.receipt_id != (
    HOLD30_ALPHA_PILOT_PROFILE_RECEIPT_SHA256
):
    raise RuntimeError(
        "Hold30 alpha pilot scientific profile changed without a receipt update"
    )


def _pilot_objective_configs(
    *, a06_optimizer_spec_receipt_sha256: str
) -> tuple[Hold30AlphaObjectiveConfig, ...]:
    profile = HOLD30_ALPHA_PILOT_PROFILE
    common = {
        "lambda_te_ceiling": profile.lambda_te_ceiling,
        "lambda_turnover": profile.lambda_turnover,
        "lambda_early_exit": profile.lambda_early_exit,
    }
    alpha = {
        **common,
        "lambda_beta": profile.lambda_beta,
        "lambda_auxiliary_alpha": profile.lambda_auxiliary_alpha,
        "active_log_scale_bounds": profile.active_log_scale_bounds,
        "auxiliary_horizon_weights": profile.auxiliary_horizon_weights,
        "auxiliary_horizon_scales": profile.auxiliary_horizon_scales,
    }
    uncertainty = {
        **alpha,
        "downside_penalty_kappa": profile.downside_penalty_kappa,
        "lambda_uncertainty": profile.lambda_uncertainty,
        "uncertainty_log_scale_bounds": profile.uncertainty_log_scale_bounds,
    }
    return (
        Hold30AlphaObjectiveConfig(
            setting_id=_M02,
            lambda_te_floor=profile.lambda_te_floor,
            **common,
        ),
        Hold30AlphaObjectiveConfig(
            setting_id=_M03,
            lambda_te_floor=profile.lambda_te_floor,
            **uncertainty,
        ),
        Hold30AlphaObjectiveConfig(
            setting_id=_A04,
            lambda_te_floor=profile.lambda_te_floor,
            **alpha,
        ),
        Hold30AlphaObjectiveConfig(setting_id=_A05, **uncertainty),
        Hold30AlphaObjectiveConfig(
            setting_id=_A06,
            lambda_te_floor=profile.lambda_te_floor,
            lambda_total_excess_mean=profile.lambda_total_excess_mean,
            lambda_total_sharpe_overlay=profile.lambda_total_sharpe_overlay,
            total_sharpe_epsilon=profile.total_sharpe_epsilon,
            lambda_volatility_ratio=profile.lambda_volatility_ratio,
            target_volatility_ratio=profile.target_volatility_ratio,
            lambda_drawdown=profile.lambda_drawdown,
            drawdown_limit=profile.drawdown_limit_log,
            a06_total_risk_step=profile.a06_total_risk_step,
            alpha_core_parameter_selector="alpha-core-only",
            overlay_parameter_selector="a06-overlay-only",
            stop_gradient_core_to_overlay=True,
            stop_gradient_overlay_to_core=True,
            separate_optimizer_spec_receipt_sha256=(
                a06_optimizer_spec_receipt_sha256
            ),
            **uncertainty,
        ),
        Hold30AlphaObjectiveConfig(
            setting_id=_A07,
            lambda_te_floor=profile.lambda_te_floor,
            lambda_direct_sharpe=profile.lambda_direct_sharpe,
            direct_sharpe_epsilon=profile.direct_sharpe_epsilon,
            **uncertainty,
        ),
    )


def build_hold30_alpha_pilot_training_plan(
    *, a06_optimizer_spec_receipt_sha256: str
) -> Hold30AlphaTrainingPlan:
    """Build the exact pilot plan without mutating the canonical unresolved plan."""

    digest = _require_digest(
        "a06_optimizer_spec_receipt_sha256",
        a06_optimizer_spec_receipt_sha256,
    )
    checkpoint = replace(
        HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
        projection_distance_max=HOLD30_ALPHA_PILOT_PROFILE.projection_distance_max,
        forced_turnover_fraction_max=(
            HOLD30_ALPHA_PILOT_PROFILE.forced_turnover_fraction_max
        ),
    )
    return Hold30AlphaTrainingPlan(
        objective_configs=_pilot_objective_configs(
            a06_optimizer_spec_receipt_sha256=digest
        ),
        checkpoint_contract=checkpoint,
        scientific_decision_receipt_sha256=HOLD30_ALPHA_PILOT_PROFILE.receipt_id,
    )


@dataclass(frozen=True, slots=True)
class Hold30AlphaPilotQualification:
    """Qualification outcome for the exact typed pilot plan."""

    training_plan_receipt_sha256: str
    scientific_decision_receipt_sha256: str
    numerical_profile_complete: bool
    checkpoint_thresholds_complete: bool
    remaining_implementation_blockers: tuple[str, ...]
    executable_eight_setting_run: bool
    promotion_authorized: bool = False
    schema: str = HOLD30_ALPHA_PILOT_QUALIFICATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "training_plan_receipt_sha256",
            "scientific_decision_receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.schema != HOLD30_ALPHA_PILOT_QUALIFICATION_SCHEMA:
            raise Hold30AlphaPilotPlanError("pilot qualification schema drifted")
        if not self.numerical_profile_complete or not self.checkpoint_thresholds_complete:
            raise Hold30AlphaPilotPlanError(
                "pilot qualification requires complete numerical choices"
            )
        if self.executable_eight_setting_run != (
            not self.remaining_implementation_blockers
        ):
            raise Hold30AlphaPilotPlanError(
                "pilot executability must agree with remaining blockers"
            )
        if self.promotion_authorized:
            raise Hold30AlphaPilotPlanError(
                "a pre-lockbox pilot qualification cannot authorize promotion"
            )

    def manifest_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_id(self) -> str:
        return sha256_payload(self.manifest_payload())


def qualify_hold30_alpha_pilot_training_plan(
    plan: Hold30AlphaTrainingPlan,
) -> Hold30AlphaPilotQualification:
    """Verify the exact freeze and report only genuine implementation blockers."""

    if not isinstance(plan, Hold30AlphaTrainingPlan):
        raise Hold30AlphaPilotPlanError("typed Hold30AlphaTrainingPlan is required")
    a06 = next(
        (config for config in plan.objective_configs if config.setting_id == _A06),
        None,
    )
    if a06 is None or a06.separate_optimizer_spec_receipt_sha256 is None:
        raise Hold30AlphaPilotPlanError(
            "pilot plan lacks the immutable A06 optimizer-spec binding"
        )
    expected = build_hold30_alpha_pilot_training_plan(
        a06_optimizer_spec_receipt_sha256=(
            a06.separate_optimizer_spec_receipt_sha256
        )
    )
    if plan.receipt_id != expected.receipt_id:
        raise Hold30AlphaPilotPlanError(
            "training plan differs from the exact pilot coefficient/profile freeze"
        )

    blockers: list[str] = []
    for config in plan.objective_configs:
        try:
            config.require_resolved()
        except Hold30AlphaTrainingError as exc:
            raise Hold30AlphaPilotPlanError(
                f"invalid pilot config for {config.setting_id}: {exc}"
            ) from exc

    return Hold30AlphaPilotQualification(
        training_plan_receipt_sha256=plan.receipt_id,
        scientific_decision_receipt_sha256=HOLD30_ALPHA_PILOT_PROFILE.receipt_id,
        numerical_profile_complete=True,
        checkpoint_thresholds_complete=(
            plan.checkpoint_contract.result_moving_thresholds_complete
        ),
        remaining_implementation_blockers=tuple(blockers),
        executable_eight_setting_run=not blockers,
    )


__all__ = [
    "HOLD30_ALPHA_PILOT_PROFILE",
    "HOLD30_ALPHA_PILOT_PROFILE_RECEIPT_SHA256",
    "HOLD30_ALPHA_PILOT_PROFILE_SCHEMA",
    "HOLD30_ALPHA_PILOT_QUALIFICATION_SCHEMA",
    "Hold30AlphaPilotPlanError",
    "Hold30AlphaPilotProfile",
    "Hold30AlphaPilotQualification",
    "build_hold30_alpha_pilot_training_plan",
    "qualify_hold30_alpha_pilot_training_plan",
]
