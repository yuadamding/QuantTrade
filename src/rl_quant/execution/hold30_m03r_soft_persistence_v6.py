"""Cause-typed cohort release semantics for M03R v6 soft persistence.

This module qualifies only the holding/release mechanism.  It is not the
missing real-data portfolio adapter or post-ensemble risk projection.  A
learned discretionary exit is never masked by age; forced exits are applied
first and are never attenuated by either the 30-session prior or exact hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_quant.models.daily_policy import HOLD30_AGE_CAP, hold30_release_hazard
from rl_quant.protocol.hold30_alpha_m03r_v6 import M03R_SOFT_PERSISTENCE


class M03RV6ReleaseError(ValueError):
    """A cohort release request is malformed or violates v6 semantics."""


def _finite_float_tensor(name: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise M03RV6ReleaseError(f"{name} must be a finite floating tensor")


@dataclass(frozen=True, slots=True)
class M03RV6CohortReleaseResult:
    """One conservation-complete split of held cohort notional."""

    discretionary_release_by_age: torch.Tensor
    forced_release_by_age: torch.Tensor
    remaining_notional_by_age: torch.Tensor
    discretionary_release_hazard_by_age: torch.Tensor

    @property
    def discretionary_release_by_asset(self) -> torch.Tensor:
        return self.discretionary_release_by_age.sum(dim=-1)

    @property
    def forced_release_by_asset(self) -> torch.Tensor:
        return self.forced_release_by_age.sum(dim=-1)


def m03r_v6_release_cohorts(
    age_notional: torch.Tensor,
    hazard_residual: torch.Tensor,
    *,
    exact_hold_decision_st: torch.Tensor | None = None,
    forced_exit_fraction: torch.Tensor | None = None,
) -> M03RV6CohortReleaseResult:
    """Release held cohorts without an age mask or a day-30 expiry.

    ``hazard_residual`` is the already bounded learned residual.  Its upper
    endpoint is an exact full discretionary exit, symmetrically with the
    lower endpoint's exact zero release.  The optional exact-hold atom can
    suppress ordinary release, but cannot suppress a forced repair.
    """

    _finite_float_tensor("age_notional", age_notional)
    _finite_float_tensor("hazard_residual", hazard_residual)
    if (
        age_notional.ndim < 2
        or age_notional.shape[-1] != HOLD30_AGE_CAP + 1
        or tuple(hazard_residual.shape) != tuple(age_notional.shape[:-1])
        or bool((age_notional < 0.0).any())
    ):
        raise M03RV6ReleaseError(
            "age_notional must be nonnegative [...,asset,61] and hazard_residual "
            "must match its leading shape"
        )
    minimum = M03R_SOFT_PERSISTENCE.bounded_hazard_residual_minimum
    maximum = M03R_SOFT_PERSISTENCE.bounded_hazard_residual_maximum
    if bool(((hazard_residual < minimum) | (hazard_residual > maximum)).any()):
        raise M03RV6ReleaseError("hazard_residual must lie in the bound v6 interval")

    if exact_hold_decision_st is not None:
        _finite_float_tensor("exact_hold_decision_st", exact_hold_decision_st)
        if tuple(exact_hold_decision_st.shape) != tuple(hazard_residual.shape) or bool(
            ((exact_hold_decision_st != 0.0) & (exact_hold_decision_st != 1.0)).any()
        ):
            raise M03RV6ReleaseError(
                "exact_hold_decision_st must be a hard 0/1 tensor aligned by asset"
            )

    if forced_exit_fraction is None:
        forced_exit_fraction = torch.zeros_like(hazard_residual)
    else:
        _finite_float_tensor("forced_exit_fraction", forced_exit_fraction)
        if tuple(forced_exit_fraction.shape) != tuple(hazard_residual.shape) or bool(
            ((forced_exit_fraction < 0.0) | (forced_exit_fraction > 1.0)).any()
        ):
            raise M03RV6ReleaseError(
                "forced_exit_fraction must align by asset and lie in [0,1]"
            )

    forced = age_notional * forced_exit_fraction.unsqueeze(-1)
    after_forced = age_notional - forced
    ages = torch.arange(
        HOLD30_AGE_CAP + 1,
        device=age_notional.device,
        dtype=age_notional.dtype,
    )
    hazard = hold30_release_hazard(
        ages,
        hazard_residual.to(dtype=age_notional.dtype).unsqueeze(-1),
        exact_hold_probability=(
            None
            if exact_hold_decision_st is None
            else exact_hold_decision_st.to(dtype=age_notional.dtype).unsqueeze(-1)
        ),
    )
    # The historical normalized sigmoid makes -12 exactly zero but leaves the
    # +12 endpoint infinitesimally below one.  V6 makes the opposite endpoint
    # economically exact so an adverse signal can close even a young cohort.
    full_exit = hazard_residual >= maximum
    if exact_hold_decision_st is not None:
        full_exit = full_exit & (exact_hold_decision_st == 0.0)
    hazard = torch.where(full_exit.unsqueeze(-1), torch.ones_like(hazard), hazard)
    discretionary = after_forced * hazard
    remaining = after_forced - discretionary
    if not bool(
        torch.allclose(
            forced + discretionary + remaining,
            age_notional,
            atol=1e-12,
            rtol=1e-10,
        )
    ):
        raise M03RV6ReleaseError("cohort release failed notional conservation")
    return M03RV6CohortReleaseResult(
        discretionary_release_by_age=discretionary,
        forced_release_by_age=forced,
        remaining_notional_by_age=remaining,
        discretionary_release_hazard_by_age=hazard,
    )


__all__ = [
    "M03RV6CohortReleaseResult",
    "M03RV6ReleaseError",
    "m03r_v6_release_cohorts",
]
