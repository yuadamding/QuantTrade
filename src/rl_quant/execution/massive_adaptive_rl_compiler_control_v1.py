"""Execution-layer adapter for bounded adaptive RL compiler controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import numpy as np

from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
    MassiveAdaptivePortfolioCompilerInputsV1,
    MassiveAdaptivePortfolioDecisionV1,
    compile_massive_adaptive_portfolio_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import MassiveAdaptiveRLActionV1


MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-compiler-control-v1"
)
MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SPEC_SHA256 = semantic_sha256(
    {
        "bucket_multiplier": "one-plus-one-half-control",
        "uncertainty_multiplier": "exp-point-seven-control",
        "risk_multiplier": "exp-point-seven-control",
        "turnover_multiplier": "one-minus-one-half-one-sided-control",
        "hard_constraints": "never-relaxed",
        "neutral_action": "exact-original-input-config-and-decision",
        "profitability_reporting": False,
        "outer": False,
        "lockbox": False,
        "rl_training": False,
    }
)


class MassiveAdaptiveRLCompilerControlV1Error(ValueError):
    """A compiler control changed an identity or hard feasibility limit."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLCompilerControlV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLCompilerControlV1:
    action_receipt_sha256: str
    base_input_receipt_sha256: str
    adjusted_input_receipt_sha256: str
    base_config_receipt_sha256: str
    adjusted_config_receipt_sha256: str
    bucket_multipliers: tuple[float, ...]
    uncertainty_multiplier: float
    risk_multiplier: float
    turnover_multiplier: float
    neutral_equivalence: bool
    hard_constraints_unchanged: bool
    semantic_receipt_sha256: str
    compiler_control_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        multipliers = (
            *self.bucket_multipliers,
            self.uncertainty_multiplier,
            self.risk_multiplier,
            self.turnover_multiplier,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SCHEMA
            or len(self.bucket_multipliers)
            != len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
            or any(not math.isfinite(value) or value <= 0.0 for value in multipliers)
            or any(not 0.5 <= value <= 1.5 for value in self.bucket_multipliers)
            or not math.exp(-0.7)
            <= self.uncertainty_multiplier
            <= math.exp(0.7)
            or not math.exp(-0.7) <= self.risk_multiplier <= math.exp(0.7)
            or not 0.5 <= self.turnover_multiplier <= 1.0
            or not self.hard_constraints_unchanged
            or self.compiler_control_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLCompilerControlV1Error(
                "adaptive RL compiler control differs"
            )
        for value in (
            self.action_receipt_sha256,
            self.base_input_receipt_sha256,
            self.adjusted_input_receipt_sha256,
            self.base_config_receipt_sha256,
            self.adjusted_config_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL compiler control", value)
        if self.neutral_equivalence and (
            self.base_input_receipt_sha256 != self.adjusted_input_receipt_sha256
            or self.base_config_receipt_sha256 != self.adjusted_config_receipt_sha256
            or any(value != 1.0 for value in multipliers)
        ):
            raise MassiveAdaptiveRLCompilerControlV1Error(
                "neutral adaptive RL action changed the compiler route"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def apply_massive_adaptive_rl_action_v1(
    *,
    inputs: MassiveAdaptivePortfolioCompilerInputsV1,
    config: MassiveAdaptivePortfolioCompilerConfigV1,
    action: MassiveAdaptiveRLActionV1,
) -> tuple[
    MassiveAdaptivePortfolioCompilerInputsV1,
    MassiveAdaptivePortfolioCompilerConfigV1,
    MassiveAdaptiveRLCompilerControlV1,
]:
    """Apply bounded controls without changing the hard feasible region."""

    inputs.validate()
    config.validate()
    action.validate()
    bucket_multipliers = tuple(1.0 + 0.5 * value for value in action.bucket_controls)
    uncertainty_multiplier = math.exp(0.7 * action.uncertainty_control)
    risk_multiplier = math.exp(0.7 * action.risk_control)
    turnover_multiplier = 1.0 - 0.5 * action.turnover_control
    if action.is_neutral:
        adjusted_inputs = inputs
        adjusted_config = config
    else:
        expected = np.asarray(
            inputs.bucket_expected_residual_returns, dtype=np.float64
        ) * np.asarray(bucket_multipliers, dtype=np.float64)[None, :]
        expected_values = tuple(
            tuple(float(value) for value in row) for row in expected.tolist()
        )
        adjusted_inputs = replace(
            inputs,
            bucket_expected_residual_returns=expected_values,
            forecast_receipt_sha256=semantic_sha256(
                {
                    "base_forecast": inputs.forecast_receipt_sha256,
                    "rl_action": action.semantic_receipt_sha256,
                    "adjusted_bucket_expected_residual_returns": expected_values,
                }
            ),
        )
        adjusted_config = replace(
            config,
            uncertainty_standard_deviations=(
                config.uncertainty_standard_deviations * uncertainty_multiplier
            ),
            risk_aversion=config.risk_aversion * risk_multiplier,
            maximum_daily_one_way_turnover=(
                config.maximum_daily_one_way_turnover * turnover_multiplier
            ),
        )
        adjusted_inputs.validate()
        adjusted_config.validate()
    hard_names = (
        "maximum_security_weight",
        "maximum_issuer_weight",
        "tracking_error_limit_annualized",
        "absolute_active_beta_limit",
        "maximum_adv_participation",
    )
    hard_unchanged = all(
        getattr(config, name) == getattr(adjusted_config, name) for name in hard_names
    ) and (
        adjusted_config.maximum_daily_one_way_turnover
        <= config.maximum_daily_one_way_turnover
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SCHEMA,
        "action_receipt_sha256": action.semantic_receipt_sha256,
        "base_input_receipt_sha256": inputs.receipt_sha256,
        "adjusted_input_receipt_sha256": adjusted_inputs.receipt_sha256,
        "base_config_receipt_sha256": config.receipt_sha256,
        "adjusted_config_receipt_sha256": adjusted_config.receipt_sha256,
        "bucket_multipliers": bucket_multipliers,
        "uncertainty_multiplier": uncertainty_multiplier,
        "risk_multiplier": risk_multiplier,
        "turnover_multiplier": turnover_multiplier,
        "neutral_equivalence": action.is_neutral,
        "hard_constraints_unchanged": hard_unchanged,
        "compiler_control_authorized": False,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SOURCE_SHA256
        ),
    }
    application = MassiveAdaptiveRLCompilerControlV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    application.validate()
    return adjusted_inputs, adjusted_config, application


def compile_massive_adaptive_rl_control_v1(
    *,
    inputs: MassiveAdaptivePortfolioCompilerInputsV1,
    config: MassiveAdaptivePortfolioCompilerConfigV1,
    action: MassiveAdaptiveRLActionV1,
) -> tuple[MassiveAdaptiveRLCompilerControlV1, MassiveAdaptivePortfolioDecisionV1]:
    """Run the sole compiler path after applying one bounded control action."""

    adjusted_inputs, adjusted_config, application = (
        apply_massive_adaptive_rl_action_v1(
            inputs=inputs,
            config=config,
            action=action,
        )
    )
    decision = compile_massive_adaptive_portfolio_v1(
        adjusted_inputs, config=adjusted_config
    )
    return application, decision


__all__ = [
    "MASSIVE_ADAPTIVE_RL_COMPILER_CONTROL_V1_SCHEMA",
    "MassiveAdaptiveRLCompilerControlV1",
    "MassiveAdaptiveRLCompilerControlV1Error",
    "apply_massive_adaptive_rl_action_v1",
    "compile_massive_adaptive_rl_control_v1",
]
