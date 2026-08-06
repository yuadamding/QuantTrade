"""Generation-qualified soft-persistence objective for the M03R v7 panel.

V7 keeps the corrected NAV/session normalization introduced during v6
qualification while making the age-zero coefficient a setting identity:
P00 uses 0 bp, the canonical row uses 5 bp, and P10 uses 10 bp.  No caller may
override the coefficient independently of the registered setting.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_DESIGN_ID,
    M03R_V7_PERSISTENCE_OBJECTIVE_SCHEMA,
    M03R_V7_PROTOCOL_GENERATION,
    M03R_V7_SHARED_CONFIGURATION,
    resolve_m03r_v7_setting,
    validate_m03r_v7_artifact_identity,
)

M03R_V7_TRAINING_PLAN_SCHEMA = "rl-quant.m03r-v7-training-plan-v1"
M03R_V7_AGE_LEDGER_BIN_COUNT = 61
_BASIS_POINT_AS_RETURN = 1.0e-4


class M03RV7ObjectiveError(ValueError):
    """A v7 persistence input or generation identity is malformed."""


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_age_notional(name: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.numel() != M03R_V7_AGE_LEDGER_BIN_COUNT
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool((value < 0.0).any())
    ):
        raise M03RV7ObjectiveError(
            f"{name} must be finite nonnegative floating notional in exactly "
            f"{M03R_V7_AGE_LEDGER_BIN_COUNT} age bins"
        )


@dataclass(frozen=True, slots=True)
class M03RV7ExitNotionalByAge:
    """Cause-typed aggregate; only policy-discretionary sales enter the loss."""

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
        """Revalidate tensor contents because frozen dataclasses do not freeze tensors."""

        rows = (
            ("discretionary_policy", self.discretionary_policy),
            ("other_forced", self.other_forced),
            ("unavailable", self.unavailable),
            ("risk_repair", self.risk_repair),
            ("corporate_action", self.corporate_action),
            ("terminal", self.terminal),
        )
        for name, value in rows:
            _require_age_notional(name, value)
        reference = self.discretionary_policy
        for name, value in rows[1:]:
            if (
                value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
            ):
                raise M03RV7ObjectiveError(
                    f"{name} must align exactly with discretionary_policy"
                )
        if (
            isinstance(self.valid_decision_session_count, bool)
            or not isinstance(self.valid_decision_session_count, int)
            or self.valid_decision_session_count <= 0
        ):
            raise M03RV7ObjectiveError(
                "valid_decision_session_count must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class M03RV7TrainingPlan:
    """Content-bound optimizer geometry and setting-specific coefficient."""

    setting_id: str
    total_optimizer_steps: int
    protocol_generation: str = M03R_V7_PROTOCOL_GENERATION
    design_id: str = M03R_V7_DESIGN_ID
    schema: str = M03R_V7_TRAINING_PLAN_SCHEMA

    def __post_init__(self) -> None:
        try:
            validate_m03r_v7_artifact_identity(
                protocol_generation=self.protocol_generation,
                design_id=self.design_id,
                setting_id=self.setting_id,
            )
        except ValueError as exc:
            raise M03RV7ObjectiveError(str(exc)) from exc
        if (
            isinstance(self.total_optimizer_steps, bool)
            or not isinstance(self.total_optimizer_steps, int)
            or self.total_optimizer_steps <= 0
            or self.schema != M03R_V7_TRAINING_PLAN_SCHEMA
        ):
            raise M03RV7ObjectiveError(
                "v7 training plan requires a positive optimizer-step count"
            )

    @property
    def persistence_coefficient_basis_points(self) -> float:
        return resolve_m03r_v7_setting(
            self.setting_id
        ).persistence_coefficient_basis_points

    def canonical_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "persistence_objective_schema": M03R_V7_PERSISTENCE_OBJECTIVE_SCHEMA,
            "persistence_coefficient_basis_points": (
                self.persistence_coefficient_basis_points
            ),
            "warmup_fraction_of_optimizer_updates": (
                M03R_V7_SHARED_CONFIGURATION.persistence
                .warmup_fraction_of_optimizer_updates
            ),
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class M03RV7TrainingProgress:
    """Completed steps bound to the exact setting-specific training plan."""

    plan: M03RV7TrainingPlan
    completed_optimizer_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.plan, M03RV7TrainingPlan):
            raise M03RV7ObjectiveError("progress must bind a typed v7 plan")
        if (
            isinstance(self.completed_optimizer_steps, bool)
            or not isinstance(self.completed_optimizer_steps, int)
            or not 0
            <= self.completed_optimizer_steps
            <= self.plan.total_optimizer_steps
        ):
            raise M03RV7ObjectiveError(
                "completed optimizer steps must lie in [0,total]"
            )

    @property
    def warmup_multiplier(self) -> float:
        warmup_steps = max(
            1,
            math.ceil(
                self.plan.total_optimizer_steps
                * M03R_V7_SHARED_CONFIGURATION.persistence
                .warmup_fraction_of_optimizer_updates
            ),
        )
        return min(1.0, self.completed_optimizer_steps / warmup_steps)


@dataclass(frozen=True, slots=True)
class M03RV7PersistenceDiagnostics:
    """Detached setting- and cause-aware persistence telemetry."""

    setting_id: str
    coefficient_basis_points: float
    warmup_multiplier: float
    valid_decision_session_count: int
    weighted_early_exit_notional: torch.Tensor
    weighted_early_exit_notional_per_valid_session: torch.Tensor
    total_discretionary_exit_notional: torch.Tensor
    penalty_paid: torch.Tensor


def m03r_v7_soft_persistence_objective(
    exits: M03RV7ExitNotionalByAge,
    progress: M03RV7TrainingProgress,
) -> tuple[torch.Tensor, M03RV7PersistenceDiagnostics]:
    """Return the registered coefficient times young sold NAV per valid session.

    There is deliberately no denominator based on total sold notional.  Sales
    aged 30 sessions or more have exactly zero value and gradient in this term,
    so mature turnover cannot dilute the cost of a young sale.
    """

    if not isinstance(exits, M03RV7ExitNotionalByAge):
        raise M03RV7ObjectiveError("exits must use the typed v7 cause inventory")
    if not isinstance(progress, M03RV7TrainingProgress):
        raise M03RV7ObjectiveError("progress must use the typed v7 plan")
    exits.validate()

    sold = exits.discretionary_policy
    horizon = (
        M03R_V7_SHARED_CONFIGURATION.persistence.preference_horizon_sessions
    )
    ages = torch.arange(sold.numel(), device=sold.device, dtype=sold.dtype)
    weights = ((float(horizon) - ages).clamp_min(0.0) / float(horizon)).square()
    weighted = (sold * weights).sum()
    per_session = weighted / exits.valid_decision_session_count
    coefficient_bp = progress.plan.persistence_coefficient_basis_points
    penalty = (
        coefficient_bp
        * _BASIS_POINT_AS_RETURN
        * progress.warmup_multiplier
        * per_session
    )
    diagnostics = M03RV7PersistenceDiagnostics(
        setting_id=progress.plan.setting_id,
        coefficient_basis_points=coefficient_bp,
        warmup_multiplier=progress.warmup_multiplier,
        valid_decision_session_count=exits.valid_decision_session_count,
        weighted_early_exit_notional=weighted.detach().clone(),
        weighted_early_exit_notional_per_valid_session=(
            per_session.detach().clone()
        ),
        total_discretionary_exit_notional=sold.sum().detach().clone(),
        penalty_paid=penalty.detach().clone(),
    )
    return penalty, diagnostics


__all__ = [
    "M03R_V7_AGE_LEDGER_BIN_COUNT",
    "M03R_V7_TRAINING_PLAN_SCHEMA",
    "M03RV7ExitNotionalByAge",
    "M03RV7ObjectiveError",
    "M03RV7PersistenceDiagnostics",
    "M03RV7TrainingPlan",
    "M03RV7TrainingProgress",
    "m03r_v7_soft_persistence_objective",
]
