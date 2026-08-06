"""V6-only mutually exclusive HOLD / CONTINUOUS / EXIT action head.

The legacy smooth hazard is open at its upper endpoint for every finite model
logit, so it can approximate but cannot represent a literal full exit.  This
isolated v6 surface adds two exact atoms around that continuous action without
changing v4/v5 tensors, state dictionaries, or execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rl_quant.protocol.hold30_alpha_m03r_v6 import M03R_DESIGN

M03R_V6_CONTINUOUS_ACTION_INDEX = 0
M03R_V6_HOLD_ACTION_INDEX = 1
M03R_V6_EXIT_ACTION_INDEX = 2
M03R_V6_EXIT_ACTION_COUNT = 3
M03R_V6_EXIT_ACTION_SCHEMA = "rl-quant.m03r-v6-three-way-exit-action-v1"


class M03RV6ExitActionError(ValueError):
    """A v6 exit-action tensor or model input is malformed."""


def validate_m03r_v6_exit_action_protocol() -> None:
    """Fail closed if the immutable v6 action semantics drift from this head."""

    persistence = M03R_DESIGN.soft_persistence
    model = M03R_DESIGN.model
    if (
        not persistence.exact_exit_action_supported
        or not persistence.exact_exit_action_required_for_learned_hazard_settings
        or persistence.exit_action_parameterization
        != "mutually-exclusive-straight-through-hold-continuous-exit-v1"
        or persistence.continuous_hazard_upper_endpoint_is_exact_exit
        or not model.exact_exit_action_supported
        or not model.exact_exit_action_required_for_learned_hazard_settings
    ):
        raise M03RV6ExitActionError(
            "M03R v6 protocol must require a distinct reachable exact EXIT atom "
            "alongside the non-exact continuous hazard"
        )


def straight_through_m03r_v6_exit_action(
    logits: torch.Tensor,
    *,
    allow_exact_hold_atom: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft probabilities and an exact one-hot forward decision.

    Action index zero is CONTINUOUS, so a zero-initialized head preserves the
    existing hazard path under deterministic ``argmax`` tie-breaking.  The
    hard forward action uses a softmax straight-through gradient surrogate.
    """

    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim < 1
        or logits.shape[-1] != M03R_V6_EXIT_ACTION_COUNT
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
    ):
        raise M03RV6ExitActionError(
            "exit-action logits must be finite floating [...,3]"
        )
    if not isinstance(allow_exact_hold_atom, bool):
        raise M03RV6ExitActionError("allow_exact_hold_atom must be boolean")
    if allow_exact_hold_atom:
        soft = torch.softmax(logits, dim=-1)
        hard_index = soft.argmax(dim=-1)
    else:
        # A11 removes only the exact HOLD atom. CONTINUOUS and exact EXIT stay
        # trainable and mutually exclusive, with no finite-logit approximation
        # or post-hoc masking that could leak probability into HOLD.
        continuous_exit_logits = logits[
            ...,
            (M03R_V6_CONTINUOUS_ACTION_INDEX, M03R_V6_EXIT_ACTION_INDEX),
        ]
        continuous_exit_soft = torch.softmax(continuous_exit_logits, dim=-1)
        soft = torch.zeros_like(logits)
        soft[..., M03R_V6_CONTINUOUS_ACTION_INDEX] = continuous_exit_soft[..., 0]
        soft[..., M03R_V6_EXIT_ACTION_INDEX] = continuous_exit_soft[..., 1]
        reduced_index = continuous_exit_soft.argmax(dim=-1)
        hard_index = torch.where(
            reduced_index == 0,
            torch.full_like(reduced_index, M03R_V6_CONTINUOUS_ACTION_INDEX),
            torch.full_like(reduced_index, M03R_V6_EXIT_ACTION_INDEX),
        )
    hard = F.one_hot(
        hard_index,
        num_classes=M03R_V6_EXIT_ACTION_COUNT,
    ).to(dtype=soft.dtype)
    return soft, (soft - soft.detach()) + hard


@dataclass(frozen=True, slots=True)
class M03RV6ExitAction:
    """One model-emitted three-way action with an explicit risky-asset mask."""

    logits: torch.Tensor
    soft_probabilities: torch.Tensor
    decision_st: torch.Tensor
    risky_available: torch.Tensor
    exact_hold_atom_enabled: bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate tensors because tensor contents remain mutable."""

        if (
            not isinstance(self.logits, torch.Tensor)
            or self.logits.ndim != 3
            or self.logits.shape[-1] != M03R_V6_EXIT_ACTION_COUNT
            or not self.logits.is_floating_point()
            or not bool(torch.isfinite(self.logits).all())
        ):
            raise M03RV6ExitActionError(
                "exit action logits must be finite floating [batch,asset,3]"
            )
        if (
            not isinstance(self.risky_available, torch.Tensor)
            or self.risky_available.dtype != torch.bool
            or tuple(self.risky_available.shape) != tuple(self.logits.shape[:-1])
            or self.risky_available.device != self.logits.device
        ):
            raise M03RV6ExitActionError(
                "risky_available must be boolean [batch,asset] on the logits device"
            )
        if not isinstance(self.exact_hold_atom_enabled, bool):
            raise M03RV6ExitActionError("exact_hold_atom_enabled must be boolean")
        for name, value in (
            ("soft_probabilities", self.soft_probabilities),
            ("decision_st", self.decision_st),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(self.logits.shape)
                or value.dtype != self.logits.dtype
                or value.device != self.logits.device
                or not bool(torch.isfinite(value).all())
            ):
                raise M03RV6ExitActionError(
                    f"{name} must align exactly with exit-action logits"
                )
        if not bool(
            torch.allclose(
                self.soft_probabilities.sum(dim=-1),
                torch.ones_like(self.soft_probabilities[..., 0]),
                atol=1e-7,
                rtol=1e-7,
            )
        ) or bool(
            ((self.soft_probabilities < 0.0) | (self.soft_probabilities > 1.0)).any()
        ):
            raise M03RV6ExitActionError(
                "exit-action soft probabilities must lie on the simplex"
            )
        detached_decision = self.decision_st.detach()
        if bool(
            ((detached_decision != 0.0) & (detached_decision != 1.0)).any()
        ) or not bool(
            torch.equal(
                detached_decision.sum(dim=-1),
                torch.ones_like(detached_decision[..., 0]),
            )
        ):
            raise M03RV6ExitActionError(
                "exit-action forward decision must be exactly one-hot"
            )
        risky = self.risky_available
        if bool(risky.any()):
            expected_soft, expected_decision = straight_through_m03r_v6_exit_action(
                self.logits[risky],
                allow_exact_hold_atom=self.exact_hold_atom_enabled,
            )
            if not bool(
                torch.allclose(
                    self.soft_probabilities[risky],
                    expected_soft,
                    atol=1e-7,
                    rtol=1e-7,
                )
            ):
                raise M03RV6ExitActionError(
                    "risky soft probabilities do not match model logits"
                )
            if not bool(
                torch.equal(
                    detached_decision[risky],
                    expected_decision.detach(),
                )
            ):
                raise M03RV6ExitActionError(
                    "risky forward decisions do not match deterministic logits"
                )
            if not self.exact_hold_atom_enabled and (
                bool(
                    (
                        self.soft_probabilities[..., M03R_V6_HOLD_ACTION_INDEX][risky]
                        != 0.0
                    ).any()
                )
                or bool(
                    (
                        detached_decision[..., M03R_V6_HOLD_ACTION_INDEX][risky] != 0.0
                    ).any()
                )
            ):
                raise M03RV6ExitActionError(
                    "A11 risky assets must have exactly zero HOLD probability and decision"
                )
        unavailable = ~risky
        if bool(unavailable.any()):
            hold = torch.zeros(
                M03R_V6_EXIT_ACTION_COUNT,
                device=self.logits.device,
                dtype=self.logits.dtype,
            )
            hold[M03R_V6_HOLD_ACTION_INDEX] = 1.0
            expected = hold.expand(int(unavailable.sum()), -1)
            if not bool(torch.equal(self.soft_probabilities[unavailable], expected)):
                raise M03RV6ExitActionError(
                    "CASH/unavailable assets require an exact HOLD soft sentinel"
                )
            if not bool(torch.equal(detached_decision[unavailable], expected)):
                raise M03RV6ExitActionError(
                    "CASH/unavailable assets require an exact HOLD decision"
                )

    def clone(self, *, detach: bool = False) -> M03RV6ExitAction:
        """Copy the complete typed action across delayed/runtime boundaries.

        The straight-through decision and its soft surrogate must travel
        together.  Copying only the hard decision would silently sever the
        training gradient, while copying only logits would resample the
        decision at the future fill boundary.
        """

        if not isinstance(detach, bool):
            raise M03RV6ExitActionError("detach must be boolean")

        def copy_float(value: torch.Tensor) -> torch.Tensor:
            return value.detach().clone() if detach else value.clone()

        risky_available = (
            self.risky_available.detach().clone()
            if detach
            else self.risky_available.clone()
        )
        return M03RV6ExitAction(
            logits=copy_float(self.logits),
            soft_probabilities=copy_float(self.soft_probabilities),
            decision_st=copy_float(self.decision_st),
            risky_available=risky_available,
            exact_hold_atom_enabled=self.exact_hold_atom_enabled,
        )

    @property
    def continuous_decision_st(self) -> torch.Tensor:
        return self.decision_st[..., M03R_V6_CONTINUOUS_ACTION_INDEX]

    @property
    def hold_decision_st(self) -> torch.Tensor:
        return self.decision_st[..., M03R_V6_HOLD_ACTION_INDEX]

    @property
    def exit_decision_st(self) -> torch.Tensor:
        return self.decision_st[..., M03R_V6_EXIT_ACTION_INDEX]


class M03RV6ExitActionHead(nn.Module):
    """Small shared head producing a reachable exact action for every asset."""

    def __init__(self, hidden_dim: int, *, allow_exact_hold_atom: bool = True) -> None:
        super().__init__()
        validate_m03r_v6_exit_action_protocol()
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise M03RV6ExitActionError("hidden_dim must be a positive integer")
        if not isinstance(allow_exact_hold_atom, bool):
            raise M03RV6ExitActionError("allow_exact_hold_atom must be boolean")
        self.hidden_dim = hidden_dim
        self.allow_exact_hold_atom = allow_exact_hold_atom
        self.action_logits = nn.Linear(hidden_dim, M03R_V6_EXIT_ACTION_COUNT)
        # Equal zero logits choose CONTINUOUS (index zero), preserving the
        # existing smooth-hazard action at initialization without a new bias.
        nn.init.zeros_(self.action_logits.weight)
        nn.init.zeros_(self.action_logits.bias)

    def forward(
        self,
        hazard_hidden: torch.Tensor,
        available: torch.Tensor,
        *,
        cash_index: int = 0,
    ) -> M03RV6ExitAction:
        if (
            not isinstance(hazard_hidden, torch.Tensor)
            or hazard_hidden.ndim != 3
            or hazard_hidden.shape[-1] != self.hidden_dim
            or not hazard_hidden.is_floating_point()
            or not bool(torch.isfinite(hazard_hidden).all())
        ):
            raise M03RV6ExitActionError(
                "hazard_hidden must be finite floating [batch,asset,hidden_dim]"
            )
        batch, assets, _width = hazard_hidden.shape
        if (
            not isinstance(available, torch.Tensor)
            or available.dtype != torch.bool
            or tuple(available.shape) != (batch, assets)
            or available.device != hazard_hidden.device
        ):
            raise M03RV6ExitActionError(
                "available must be boolean [batch,asset] on the hidden-state device"
            )
        if (
            isinstance(cash_index, bool)
            or not isinstance(cash_index, int)
            or not 0 <= cash_index < assets
        ):
            raise M03RV6ExitActionError("cash_index is outside the asset axis")
        risky = available.clone()
        risky[:, cash_index] = False
        raw_logits = self.action_logits(hazard_hidden)
        soft, decision = straight_through_m03r_v6_exit_action(
            raw_logits,
            allow_exact_hold_atom=self.allow_exact_hold_atom,
        )
        hold = torch.zeros_like(soft)
        hold[..., M03R_V6_HOLD_ACTION_INDEX] = 1.0
        soft = torch.where(risky.unsqueeze(-1), soft, hold)
        decision = torch.where(risky.unsqueeze(-1), decision, hold)
        logits = torch.where(
            risky.unsqueeze(-1), raw_logits, torch.zeros_like(raw_logits)
        )
        return M03RV6ExitAction(
            logits=logits,
            soft_probabilities=soft,
            decision_st=decision,
            risky_available=risky,
            exact_hold_atom_enabled=self.allow_exact_hold_atom,
        )


__all__ = [
    "M03R_V6_CONTINUOUS_ACTION_INDEX",
    "M03R_V6_EXIT_ACTION_COUNT",
    "M03R_V6_EXIT_ACTION_INDEX",
    "M03R_V6_EXIT_ACTION_SCHEMA",
    "M03R_V6_HOLD_ACTION_INDEX",
    "M03RV6ExitAction",
    "M03RV6ExitActionError",
    "M03RV6ExitActionHead",
    "straight_through_m03r_v6_exit_action",
    "validate_m03r_v6_exit_action_protocol",
]
