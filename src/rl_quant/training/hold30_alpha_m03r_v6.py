"""Soft-persistence objective primitives for immutable M03R v6.

V6 changes only the learned discretionary exit preference.  It does not add a
minimum holding age, a sell mask, or a forced 30-session expiry.  The economic
active-return objective remains external to this module so persistence and
economic gradients can be inspected independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_AGE_LEDGER_BIN_COUNT,
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SOFT_PERSISTENCE,
    validate_m03r_v6_artifact_identity,
)

M03R_V6_SOFT_PERSISTENCE_OBJECTIVE_SCHEMA = (
    "rl-quant.hold30.m03r-v6-soft-persistence-objective-v2"
)
M03R_V6_TRAINING_PLAN_SCHEMA = "rl-quant.hold30.m03r-v6-training-plan-v1"
_BASIS_POINT_AS_RETURN = 1.0e-4
_GRADIENT_NORM_EPSILON = 1.0e-12


class M03RV6ObjectiveError(ValueError):
    """The v6 soft-persistence objective is unbound or malformed."""


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_nonnegative_notional(name: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.numel() != M03R_AGE_LEDGER_BIN_COUNT
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool((value < 0.0).any())
    ):
        raise M03RV6ObjectiveError(
            f"{name} must be finite nonnegative floating notional in exactly "
            f"{M03R_AGE_LEDGER_BIN_COUNT} age bins"
        )


@dataclass(frozen=True, slots=True)
class M03RV6ExitNotionalByAge:
    """Cause-typed sold notional; only policy-discretionary exits enter the loss."""

    discretionary_policy: torch.Tensor
    other_forced: torch.Tensor
    unavailable: torch.Tensor
    risk_repair: torch.Tensor
    corporate_action: torch.Tensor
    terminal: torch.Tensor
    valid_decision_session_count: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate mutable tensor contents before every objective call."""

        values = (
            ("discretionary_policy", self.discretionary_policy),
            ("other_forced", self.other_forced),
            ("unavailable", self.unavailable),
            ("risk_repair", self.risk_repair),
            ("corporate_action", self.corporate_action),
            ("terminal", self.terminal),
        )
        for name, value in values:
            _require_nonnegative_notional(name, value)
        if (
            isinstance(self.valid_decision_session_count, bool)
            or not isinstance(self.valid_decision_session_count, int)
            or self.valid_decision_session_count <= 0
        ):
            raise M03RV6ObjectiveError(
                "valid_decision_session_count must be a positive integer bound "
                "to the cause-typed exit inventory"
            )
        reference = self.discretionary_policy
        for name, value in values[1:]:
            if (
                tuple(value.shape) != tuple(reference.shape)
                or value.dtype != reference.dtype
                or value.device != reference.device
            ):
                raise M03RV6ObjectiveError(
                    f"{name} must align exactly with discretionary_policy"
                )


@dataclass(frozen=True, slots=True)
class M03RV6TrainingPlan:
    """Content-addressed denominator for the frozen warm-up schedule."""

    total_optimizer_steps: int
    setting_id: str = M03R_CANONICAL_SETTING_ID
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID
    schema: str = M03R_V6_TRAINING_PLAN_SCHEMA

    def __post_init__(self) -> None:
        try:
            validate_m03r_v6_artifact_identity(
                protocol_generation=self.protocol_generation,
                design_id=self.design_id,
                setting_id=self.setting_id,
            )
        except ValueError as exc:
            raise M03RV6ObjectiveError(str(exc)) from exc
        if (
            isinstance(self.total_optimizer_steps, bool)
            or not isinstance(self.total_optimizer_steps, int)
            or self.total_optimizer_steps <= 0
            or self.schema != M03R_V6_TRAINING_PLAN_SCHEMA
        ):
            raise M03RV6ObjectiveError(
                "v6 training plan needs a positive total optimizer-step count"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "setting_id": self.setting_id,
            "total_optimizer_steps": self.total_optimizer_steps,
            "early_exit_penalty_linear_warmup_fraction": (
                M03R_SOFT_PERSISTENCE.early_exit_penalty_linear_warmup_fraction
            ),
            "soft_persistence_objective_schema": (
                M03R_V6_SOFT_PERSISTENCE_OBJECTIVE_SCHEMA
            ),
        }

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class M03RV6TrainingProgress:
    """Completed optimizer steps bound to one immutable v6 training plan."""

    completed_optimizer_steps: int
    training_plan: M03RV6TrainingPlan

    def __post_init__(self) -> None:
        if (
            not isinstance(self.training_plan, M03RV6TrainingPlan)
            or isinstance(self.completed_optimizer_steps, bool)
            or not isinstance(self.completed_optimizer_steps, int)
            or not 0
            <= self.completed_optimizer_steps
            <= self.training_plan.total_optimizer_steps
        ):
            raise M03RV6ObjectiveError(
                "optimizer progress must bind one plan and satisfy 0 <= completed <= total"
            )

    @property
    def total_optimizer_steps(self) -> int:
        return self.training_plan.total_optimizer_steps

    @property
    def early_exit_penalty_warmup_multiplier(self) -> float:
        warmup_steps = max(
            1,
            math.ceil(
                self.total_optimizer_steps
                * M03R_SOFT_PERSISTENCE.early_exit_penalty_linear_warmup_fraction
            ),
        )
        return min(1.0, self.completed_optimizer_steps / warmup_steps)


@dataclass(frozen=True, slots=True)
class M03RV6SoftPersistenceConfig:
    """Identity-bound v6 coefficient selected by the frozen protocol."""

    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID
    schema: str = M03R_V6_SOFT_PERSISTENCE_OBJECTIVE_SCHEMA
    early_exit_penalty_bp_per_unit_at_age_zero: float = (
        M03R_SOFT_PERSISTENCE.early_exit_penalty_bp_per_unit_at_age_zero
    )
    early_exit_penalty_linear_warmup_fraction: float = (
        M03R_SOFT_PERSISTENCE.early_exit_penalty_linear_warmup_fraction
    )

    def __post_init__(self) -> None:
        if (
            self.protocol_generation != M03R_PROTOCOL_GENERATION
            or self.design_id != M03R_DESIGN_ID
            or self.schema != M03R_V6_SOFT_PERSISTENCE_OBJECTIVE_SCHEMA
        ):
            raise M03RV6ObjectiveError("v6 soft-persistence identity drifted")
        coefficient = self.early_exit_penalty_bp_per_unit_at_age_zero
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            or not math.isfinite(float(coefficient))
            or float(coefficient)
            != M03R_SOFT_PERSISTENCE.early_exit_penalty_bp_per_unit_at_age_zero
        ):
            raise M03RV6ObjectiveError(
                "canonical v6 early-exit coefficient must be exactly 5 bp; "
                "alternatives require a new protocol generation"
            )
        if self.early_exit_penalty_linear_warmup_fraction != (
            M03R_SOFT_PERSISTENCE.early_exit_penalty_linear_warmup_fraction
        ):
            raise M03RV6ObjectiveError("v6 early-exit warmup fraction drifted")


@dataclass(frozen=True, slots=True)
class M03RV6SoftPersistenceDiagnostics:
    """Detached telemetry; it cannot feed back into the economic loss."""

    early_exit_penalty_paid: torch.Tensor
    discretionary_exit_notional_by_age: torch.Tensor
    weighted_early_exit_notional: torch.Tensor
    weighted_early_exit_notional_per_valid_session: torch.Tensor
    total_discretionary_exit_notional: torch.Tensor
    valid_decision_session_count: int
    warmup_multiplier: float
    coefficient_bp_per_unit_at_age_zero: float


@dataclass(frozen=True, slots=True)
class M03RV6GradientNormTelemetry:
    """Optional hold/economic gradient telemetry, never a loss component."""

    hold_gradient_l2_norm: float
    economic_gradient_l2_norm: float

    @property
    def holding_to_economic_gradient_norm_ratio(self) -> float:
        """Diagnostic balance; it never rescales either loss automatically."""

        return self.hold_gradient_l2_norm / max(
            self.economic_gradient_l2_norm,
            _GRADIENT_NORM_EPSILON,
        )


def m03r_v6_early_exit_age_weight(age_sessions: int) -> float:
    """Return ``((30-age)_+/30)^2`` for one integer trading-session age."""

    if (
        isinstance(age_sessions, bool)
        or not isinstance(age_sessions, int)
        or age_sessions < 0
    ):
        raise M03RV6ObjectiveError("age_sessions must be a nonnegative integer")
    horizon = M03R_SOFT_PERSISTENCE.holding_preference_horizon_sessions
    remaining_fraction = max(0.0, (horizon - age_sessions) / horizon)
    return remaining_fraction * remaining_fraction


def m03r_v6_soft_persistence_objective(
    exits: M03RV6ExitNotionalByAge,
    progress: M03RV6TrainingProgress,
    config: M03RV6SoftPersistenceConfig | None = None,
) -> tuple[torch.Tensor, M03RV6SoftPersistenceDiagnostics]:
    """Return the proportional young-sale penalty and cause-safe telemetry.

    ``discretionary_policy`` is an age histogram of policy-originated sold
    notional expressed as fractions of portfolio NAV and aggregated across the
    scored decisions. The authoritative adapter includes learned-hazard,
    explicit policy de-risk, and policy-induced projection sales. The loss is
    normalized only by the exact count of valid decisions. It is deliberately
    *not* normalized by total sold notional: mature sales must have zero value
    and zero gradient in this persistence term, and cannot dilute young sales.
    """

    if not isinstance(exits, M03RV6ExitNotionalByAge):
        raise M03RV6ObjectiveError("exits must use the typed v6 cause inventory")
    if not isinstance(progress, M03RV6TrainingProgress):
        raise M03RV6ObjectiveError("progress must use typed v6 optimizer progress")
    if config is None:
        config = M03RV6SoftPersistenceConfig()
    elif not isinstance(config, M03RV6SoftPersistenceConfig):
        raise M03RV6ObjectiveError("config must use the immutable v6 objective type")
    # Revalidate because tensor contents remain mutable after dataclass creation.
    exits.validate()
    valid_decision_session_count = exits.valid_decision_session_count
    sold = exits.discretionary_policy
    horizon = M03R_SOFT_PERSISTENCE.holding_preference_horizon_sessions
    ages = torch.arange(sold.numel(), device=sold.device, dtype=sold.dtype)
    weights = ((float(horizon) - ages).clamp_min(0.0) / float(horizon)).square()
    total = sold.sum()
    weighted_notional = (sold * weights).sum()
    weighted_notional_per_valid_session = (
        weighted_notional / valid_decision_session_count
    )
    warmup_multiplier = progress.early_exit_penalty_warmup_multiplier
    coefficient_as_return = (
        config.early_exit_penalty_bp_per_unit_at_age_zero * _BASIS_POINT_AS_RETURN
    )
    penalty = (
        coefficient_as_return * warmup_multiplier * weighted_notional_per_valid_session
    )
    diagnostics = M03RV6SoftPersistenceDiagnostics(
        early_exit_penalty_paid=penalty.detach().clone(),
        discretionary_exit_notional_by_age=sold.detach().clone(),
        weighted_early_exit_notional=weighted_notional.detach().clone(),
        weighted_early_exit_notional_per_valid_session=(
            weighted_notional_per_valid_session.detach().clone()
        ),
        total_discretionary_exit_notional=total.detach().clone(),
        valid_decision_session_count=valid_decision_session_count,
        warmup_multiplier=warmup_multiplier,
        coefficient_bp_per_unit_at_age_zero=(
            config.early_exit_penalty_bp_per_unit_at_age_zero
        ),
    )
    return penalty, diagnostics


def m03r_v6_gradient_norm_telemetry(
    hold_loss: torch.Tensor,
    economic_loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
) -> M03RV6GradientNormTelemetry:
    """Measure separate gradient norms without combining or accumulating losses."""

    for name, loss in (("hold_loss", hold_loss), ("economic_loss", economic_loss)):
        if (
            not isinstance(loss, torch.Tensor)
            or loss.numel() != 1
            or not loss.is_floating_point()
            or not bool(torch.isfinite(loss).all())
            or not loss.requires_grad
        ):
            raise M03RV6ObjectiveError(
                f"{name} must be one finite gradient-enabled scalar"
            )
    bound_parameters = tuple(parameters)
    if not bound_parameters or any(
        not isinstance(parameter, torch.Tensor)
        or not parameter.is_floating_point()
        or not parameter.requires_grad
        for parameter in bound_parameters
    ):
        raise M03RV6ObjectiveError(
            "gradient telemetry needs gradient-enabled floating parameters"
        )
    if len({id(parameter) for parameter in bound_parameters}) != len(bound_parameters):
        raise M03RV6ObjectiveError("gradient telemetry parameters must be unique")

    hold_gradients = torch.autograd.grad(
        hold_loss,
        bound_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    economic_gradients = torch.autograd.grad(
        economic_loss,
        bound_parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def l2_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
        terms = [
            gradient.detach().to(dtype=torch.float64).square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        if not terms:
            return 0.0
        devices = {term.device for term in terms}
        if len(devices) != 1:
            raise M03RV6ObjectiveError(
                "gradient telemetry parameters must share one device"
            )
        return float(torch.stack(terms).sum().sqrt())

    return M03RV6GradientNormTelemetry(
        hold_gradient_l2_norm=l2_norm(hold_gradients),
        economic_gradient_l2_norm=l2_norm(economic_gradients),
    )


__all__ = [
    "M03R_V6_SOFT_PERSISTENCE_OBJECTIVE_SCHEMA",
    "M03R_V6_TRAINING_PLAN_SCHEMA",
    "M03RV6ExitNotionalByAge",
    "M03RV6GradientNormTelemetry",
    "M03RV6ObjectiveError",
    "M03RV6SoftPersistenceConfig",
    "M03RV6SoftPersistenceDiagnostics",
    "M03RV6TrainingPlan",
    "M03RV6TrainingProgress",
    "m03r_v6_early_exit_age_weight",
    "m03r_v6_gradient_norm_telemetry",
    "m03r_v6_soft_persistence_objective",
]
