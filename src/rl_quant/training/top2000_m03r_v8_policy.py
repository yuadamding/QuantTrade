"""Generation-qualified policy adapter for TOP2000 M03R-v8 development.

The v8 panel reuses the reviewed v7 raw daily encoder and alpha/hazard core,
but it owns a disjoint model identity and a four-horizon distributional scale
head for training-fold-only alpha pretraining.  The exact-HOLD ablation is a
frozen temperature transform of the three-way straight-through action: it
changes the training surrogate while preserving deterministic action labels
for a fixed set of logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.models.hold30_exit_action_v6 import (
    M03R_V6_HOLD_ACTION_INDEX,
    M03RV6ExitAction,
    straight_through_m03r_v6_exit_action,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_REFERENCE_SETTING_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ACTIVE_POLICY,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
    M03RV8Top2000DevSetting,
    resolve_m03r_v8_top2000_dev_setting,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    Top2000M03RV7DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03R_V8_ALPHA_HORIZON_SCALES,
)

M03R_V8_FIXED_HAZARD_SOURCE_SETTING_ID = "A08-fixed-exit-hazard-top2000-dev-v1"
M03R_V8_POLICY_SCHEMA = "rl-quant.top2000-dev.m03r-v8-policy-v1"


class M03RV8PolicyError(ValueError):
    """The v8 policy or its exact-HOLD transformation is malformed."""


def apply_m03r_v8_exit_action_temperature(
    action: M03RV6ExitAction,
    *,
    temperature: float,
) -> M03RV6ExitAction:
    """Apply the frozen temperature to risky three-way action logits.

    CASH and unavailable assets remain the exact HOLD sentinel.  Temperature
    one returns the original typed object exactly, avoiding needless numeric
    drift in the seven reference-temperature rows.
    """

    action.validate()
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)):
        raise M03RV8PolicyError("exact-HOLD action temperature must be finite")
    value = float(temperature)
    if value not in {
        M03R_V8_ACTIVE_POLICY.reference_exact_hold_action_temperature,
        M03R_V8_ACTIVE_POLICY.softened_exact_hold_action_temperature,
    }:
        raise M03RV8PolicyError("exact-HOLD action temperature is not frozen")
    if value == M03R_V8_ACTIVE_POLICY.reference_exact_hold_action_temperature:
        return action

    risky = action.risky_available
    logits = torch.where(
        risky.unsqueeze(-1),
        action.logits / value,
        torch.zeros_like(action.logits),
    )
    soft, decision = straight_through_m03r_v6_exit_action(
        logits,
        allow_exact_hold_atom=action.exact_hold_atom_enabled,
    )
    unavailable_hold = torch.zeros_like(soft)
    unavailable_hold[..., M03R_V6_HOLD_ACTION_INDEX] = 1.0
    soft = torch.where(risky.unsqueeze(-1), soft, unavailable_hold)
    decision = torch.where(risky.unsqueeze(-1), decision, unavailable_hold)
    return M03RV6ExitAction(
        logits=logits,
        soft_probabilities=soft,
        decision_st=decision,
        risky_available=risky,
        exact_hold_atom_enabled=action.exact_hold_atom_enabled,
    )


@dataclass(frozen=True, slots=True)
class M03RV8AlphaDistribution:
    """Four-horizon alpha means and log scales from one decision state."""

    predicted_mean: torch.Tensor
    predicted_log_scale: torch.Tensor

    def validate(self) -> None:
        if (
            not isinstance(self.predicted_mean, torch.Tensor)
            or self.predicted_mean.ndim != 3
            or self.predicted_mean.shape[-1] != 4
            or not self.predicted_mean.is_floating_point()
            or not bool(torch.isfinite(self.predicted_mean).all())
            or not isinstance(self.predicted_log_scale, torch.Tensor)
            or self.predicted_log_scale.shape != self.predicted_mean.shape
            or self.predicted_log_scale.dtype != self.predicted_mean.dtype
            or self.predicted_log_scale.device != self.predicted_mean.device
            or not bool(torch.isfinite(self.predicted_log_scale).all())
        ):
            raise M03RV8PolicyError(
                "v8 alpha distribution must be finite aligned [batch,asset,4]"
            )


class Top2000M03RV8DevelopmentPolicy(nn.Module):
    """Disjoint v8 adapter over the compact reviewed v7 model core."""

    state_provider_compatibility_id = (
        Top2000M03RV7DevelopmentPolicy.state_provider_compatibility_id
    )
    protocol_sha256 = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    schema = M03R_V8_POLICY_SCHEMA

    def __init__(
        self,
        setting: int | str,
        *,
        token_dim: int = 128,
        raw_stock_chunk: int = 512,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.setting: M03RV8Top2000DevSetting = resolve_m03r_v8_top2000_dev_setting(
            setting
        )
        self.source_setting_id = (
            M03R_V8_FIXED_HAZARD_SOURCE_SETTING_ID
            if self.setting.exit_hazard_mode == "fixed-structural-30-session-prior"
            else M03R_TOP2000_DEV_REFERENCE_SETTING_ID
        )
        self.source_policy = Top2000M03RV7DevelopmentPolicy(
            self.source_setting_id,
            token_dim=token_dim,
            raw_stock_chunk=raw_stock_chunk,
            activation_checkpointing=activation_checkpointing,
        )
        self.alpha_log_scale_head = nn.Linear(token_dim, 4)
        nn.init.zeros_(self.alpha_log_scale_head.weight)
        with torch.no_grad():
            self.alpha_log_scale_head.bias.copy_(
                torch.tensor(
                    [math.log(value) for value in M03R_V8_ALPHA_HORIZON_SCALES],
                    dtype=self.alpha_log_scale_head.bias.dtype,
                )
            )

    @property
    def token_dim(self) -> int:
        return self.source_policy.token_dim

    def encode_episode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.source_policy.encode_episode(*args, **kwargs)

    def alpha_pretraining_distribution(
        self,
        state_t: torch.Tensor,
        available: torch.Tensor,
    ) -> M03RV8AlphaDistribution:
        """Emit the four-horizon distribution used only by pretraining."""

        if (
            not isinstance(state_t, torch.Tensor)
            or state_t.ndim != 3
            or state_t.shape[-1] != self.token_dim
            or not state_t.is_floating_point()
            or not bool(torch.isfinite(state_t).all())
            or not isinstance(available, torch.Tensor)
            or tuple(available.shape) != tuple(state_t.shape[:2])
            or available.dtype != torch.bool
            or available.device != state_t.device
        ):
            raise M03RV8PolicyError(
                "pretraining state/availability must be aligned [batch,asset]"
            )
        with torch.autocast(
            device_type=state_t.device.type,
            dtype=torch.bfloat16,
            enabled=state_t.device.type == "cuda" or state_t.dtype == torch.bfloat16,
        ):
            market_hidden, _mask = self.source_policy.core._allocator_hidden(  # type: ignore[no-untyped-call]
                state_t,
                torch.zeros(
                    state_t.shape[:2], device=state_t.device, dtype=state_t.dtype
                ),
                available,
            )
            alpha_head = self.source_policy.core.alpha_head
            if alpha_head is None:
                raise M03RV8PolicyError("v8 source policy omitted its alpha head")
            mean = alpha_head.auxiliary_head(market_hidden).float()
            log_scale = (
                self.alpha_log_scale_head(market_hidden).float().clamp(-8.0, 2.0)
            )
        risky = available.clone()
        risky[:, 0] = False
        result = M03RV8AlphaDistribution(
            predicted_mean=torch.where(
                risky.unsqueeze(-1), mean, torch.zeros_like(mean)
            ),
            predicted_log_scale=torch.where(
                risky.unsqueeze(-1), log_scale, torch.zeros_like(log_scale)
            ),
        )
        result.validate()
        return result

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        intent = self.source_policy.hold30_intent(
            state_t,
            prev_weights,
            available,
            age_summaries,
        )
        action = intent.exit_action_v6
        if action is None:
            raise M03RV8PolicyError(
                "v8 source policy omitted its three-way exit action"
            )
        transformed = apply_m03r_v8_exit_action_temperature(
            action,
            temperature=self.setting.exact_hold_action_temperature,
        )
        return replace(intent, exit_action_v6=transformed)


__all__ = [
    "M03R_V8_FIXED_HAZARD_SOURCE_SETTING_ID",
    "M03R_V8_POLICY_SCHEMA",
    "M03RV8AlphaDistribution",
    "M03RV8PolicyError",
    "Top2000M03RV8DevelopmentPolicy",
    "apply_m03r_v8_exit_action_temperature",
]
