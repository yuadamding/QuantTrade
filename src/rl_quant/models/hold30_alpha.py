"""Compact benchmark-anchored heads for the Hold-30 alpha v3 screen.

The module is deliberately independent of portfolio accounting.  It turns a
shared per-stock market representation into economically named raw outputs and
constructs the pre-constraint benchmark tilt.  The execution layer remains
responsible for the age ledger, fills, turnover, caps, and costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import torch
from torch import nn

from rl_quant.models.hold30_hazard import (
    HOLD30_HAZARD_BOUND_MODES,
    HOLD30_HAZARD_MAX,
    HOLD30_HAZARD_MIN,
    Hold30HazardBoundMode,
    bound_hold30_hazard_residual,
    straight_through_exact_hold_decision,
)
from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_ALPHA_HORIZONS_TRADING_SESSIONS,
    M03R_DESIGN,
    M03R_SETTING_IDS,
    resolve_m03r_setting,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_HORIZONS,
    HOLD30_ALPHA_MECH8_IDS,
    HOLD30_ALPHA_TE_TARGET_ANNUAL,
    resolve_hold30_alpha_setting,
)

HOLD30_ALPHA_HEAD_PARAMETER_CAP: Final[int] = 1_000_000
_DIGEST_CHARS = frozenset("0123456789abcdef")


class Hold30AlphaModelError(ValueError):
    """A v3 alpha-head input or setting violates the frozen interface."""


def _alpha_setting_flags(setting_id: str) -> tuple[bool, bool, str]:
    """Resolve exact V3 or M03R identity without cross-generation aliases."""

    if setting_id in M03R_SETTING_IDS:
        setting = resolve_m03r_setting(setting_id)
        return (
            setting.residual_alpha_heads,
            setting.uncertainty_scaled_sizing,
            setting.sharpe_mode,
        )
    setting = resolve_hold30_alpha_setting(setting_id)
    return (
        setting.supervised_residual_alpha_heads,
        setting.uncertainty_downside_heads,
        setting.sharpe_mode,
    )


@dataclass(frozen=True, slots=True)
class Hold30AlphaOutput:
    """Raw v3 alpha outputs before fill-time portfolio construction."""

    mean_30d: torch.Tensor
    downside_30d: torch.Tensor | None
    risk_adjusted_score: torch.Tensor
    auxiliary_mean: torch.Tensor
    hazard_residual: torch.Tensor
    raw_hazard_residual: torch.Tensor
    exact_hold_probability: torch.Tensor | None
    active_risk_scale: torch.Tensor
    signal_confidence: torch.Tensor | None
    total_risk_overlay: torch.Tensor | None
    auxiliary_horizons_trading_sessions: tuple[int, ...] = HOLD30_ALPHA_HORIZONS

    def validate(self) -> None:
        if self.mean_30d.ndim != 2:
            raise Hold30AlphaModelError("mean_30d must be [batch,asset]")
        matrix = tuple(self.mean_30d.shape)
        for name in (
            "risk_adjusted_score",
            "hazard_residual",
            "raw_hazard_residual",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != matrix:
                raise Hold30AlphaModelError(f"{name} must have shape {matrix}")
        if self.downside_30d is not None and tuple(self.downside_30d.shape) != matrix:
            raise Hold30AlphaModelError(f"downside_30d must have shape {matrix}")
        if (
            self.exact_hold_probability is not None
            and tuple(self.exact_hold_probability.shape) != matrix
        ):
            raise Hold30AlphaModelError(
                f"exact_hold_probability must have shape {matrix}"
            )
        if tuple(self.auxiliary_mean.shape) != (
            *matrix,
            len(self.auxiliary_horizons_trading_sessions),
        ):
            raise Hold30AlphaModelError(
                "auxiliary_mean must be [batch,asset,four_horizons]"
            )
        if tuple(self.active_risk_scale.shape) != (matrix[0],):
            raise Hold30AlphaModelError("active_risk_scale must be [batch]")
        if self.signal_confidence is not None and tuple(
            self.signal_confidence.shape
        ) != (matrix[0],):
            raise Hold30AlphaModelError("signal_confidence must be [batch]")
        if self.total_risk_overlay is not None and tuple(
            self.total_risk_overlay.shape
        ) != (matrix[0],):
            raise Hold30AlphaModelError("total_risk_overlay must be [batch]")
        values = (
            self.mean_30d,
            self.risk_adjusted_score,
            self.auxiliary_mean,
            self.hazard_residual,
            self.raw_hazard_residual,
            self.active_risk_scale,
        )
        if self.exact_hold_probability is not None:
            values = (*values, self.exact_hold_probability)
        if self.signal_confidence is not None:
            values = (*values, self.signal_confidence)
        if self.total_risk_overlay is not None:
            values = (*values, self.total_risk_overlay)
        if self.downside_30d is not None:
            values = (*values, self.downside_30d)
        if not all(value.is_floating_point() and bool(torch.isfinite(value).all()) for value in values):
            raise Hold30AlphaModelError("alpha output contains non-finite values")
        if (
            self.downside_30d is not None
            and bool((self.downside_30d < 0).any())
        ) or bool((self.active_risk_scale < 0).any()):
            raise Hold30AlphaModelError(
                "downside and active-risk scale must be nonnegative"
            )
        if self.exact_hold_probability is not None and bool(
            (
                (self.exact_hold_probability < 0) | (self.exact_hold_probability > 1)
            ).any()
        ):
            raise Hold30AlphaModelError("exact_hold_probability must lie in [0,1]")
        if self.signal_confidence is not None and bool(
            ((self.signal_confidence < 0) | (self.signal_confidence > 1)).any()
        ):
            raise Hold30AlphaModelError("signal_confidence must lie in [0,1]")
        if bool(
            (self.hazard_residual < HOLD30_HAZARD_MIN).any()
            or (self.hazard_residual > HOLD30_HAZARD_MAX).any()
        ):
            raise Hold30AlphaModelError("hazard_residual must lie in [-12,12]")


@dataclass(frozen=True, slots=True)
class Hold30AlphaHeadConfig:
    setting_id: str
    hidden_dim: int
    age_summary_dim: int = 5
    downside_penalty_kappa: float | None = None
    active_log_scale_bounds: tuple[float, float] | None = None
    uncertainty_log_scale_bounds: tuple[float, float] | None = None
    te_target: float = HOLD30_ALPHA_TE_TARGET_ANNUAL
    parameter_cap: int = HOLD30_ALPHA_HEAD_PARAMETER_CAP
    # Explicitly opt-in for post-v3 generations. These defaults are the frozen
    # v3 behavior and therefore do not change existing checkpoint identities.
    mechanism_generation: Literal["v3-frozen", "m03r-v1"] = "v3-frozen"
    hazard_bound_mode: Hold30HazardBoundMode = "hard_clip"
    exact_hold_mixture: bool = False
    exact_hold_logit_bias: float | None = None
    fixed_hazard_residual: float | None = None
    confidence_calibration_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        use_alpha, use_uncertainty, _sharpe_mode = _alpha_setting_flags(self.setting_id)
        is_m03r = self.setting_id in M03R_SETTING_IDS
        m03r_setting = resolve_m03r_setting(self.setting_id) if is_m03r else None
        if is_m03r != (self.mechanism_generation == "m03r-v1"):
            raise Hold30AlphaModelError(
                "M03R head options require an exact M03R setting identity and "
                "V3 heads require mechanism_generation='v3-frozen'"
            )
        if not use_alpha:
            raise Hold30AlphaModelError(
                "supervised alpha heads are exclusive to m03/a04-a07 or exact "
                "M03R settings with residual heads; m00-m02 must not instantiate them"
            )
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or self.hidden_dim <= 0
        ):
            raise Hold30AlphaModelError("hidden_dim must be a positive integer")
        if (
            isinstance(self.age_summary_dim, bool)
            or not isinstance(self.age_summary_dim, int)
            or self.age_summary_dim <= 0
        ):
            raise Hold30AlphaModelError("age_summary_dim must be a positive integer")
        if not 0 < float(self.te_target) < 1:
            raise Hold30AlphaModelError("te_target must lie in (0,1)")
        expected_risk_reference = (
            M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
            if is_m03r
            else HOLD30_ALPHA_TE_TARGET_ANNUAL
        )
        if float(self.te_target) != float(expected_risk_reference):
            raise Hold30AlphaModelError(
                "te_target must match the exact protocol generation risk reference"
            )
        if use_uncertainty:
            if self.downside_penalty_kappa is None:
                raise Hold30AlphaModelError(
                    "downside_penalty_kappa is an unresolved result-moving "
                    "coefficient for uncertainty settings"
                )
            if (
                isinstance(self.downside_penalty_kappa, bool)
                or not math.isfinite(float(self.downside_penalty_kappa))
                or float(self.downside_penalty_kappa) <= 0
            ):
                raise Hold30AlphaModelError(
                    "downside_penalty_kappa must be finite and strictly positive"
                )
            if self.uncertainty_log_scale_bounds is None:
                raise Hold30AlphaModelError(
                    "uncertainty_log_scale_bounds are unresolved result-moving bounds"
                )
            lower, upper = self.uncertainty_log_scale_bounds
            if (
                any(isinstance(value, bool) for value in (lower, upper))
                or not all(math.isfinite(float(value)) for value in (lower, upper))
                or float(lower) >= float(upper)
            ):
                raise Hold30AlphaModelError(
                    "uncertainty_log_scale_bounds must be finite and ordered"
                )
        elif self.downside_penalty_kappa not in (None, 0, 0.0):
            raise Hold30AlphaModelError(
                "the no-uncertainty ablation cannot carry a downside penalty"
            )
        elif self.uncertainty_log_scale_bounds is not None:
            raise Hold30AlphaModelError(
                "the no-uncertainty ablation cannot carry uncertainty bounds"
            )
        if is_m03r:
            if self.active_log_scale_bounds is not None:
                raise Hold30AlphaModelError(
                    "M03R active risk is confidence-scaled; V3 active-log-scale "
                    "bounds are forbidden"
                )
            digest = self.confidence_calibration_manifest_sha256
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in _DIGEST_CHARS for character in digest)
            ):
                raise Hold30AlphaModelError(
                    "M03R requires a content-bound confidence calibration manifest"
                )
        else:
            if self.confidence_calibration_manifest_sha256 is not None:
                raise Hold30AlphaModelError(
                    "V3 heads cannot bind an M03R confidence calibration manifest"
                )
            if self.active_log_scale_bounds is None:
                raise Hold30AlphaModelError(
                    "active_log_scale_bounds are unresolved result-moving action bounds"
                )
            lower, upper = self.active_log_scale_bounds
            if (
                any(isinstance(value, bool) for value in (lower, upper))
                or not all(math.isfinite(float(value)) for value in (lower, upper))
                or float(lower) >= float(upper)
            ):
                raise Hold30AlphaModelError(
                    "active_log_scale_bounds must be finite and strictly ordered"
                )
        if (
            isinstance(self.parameter_cap, bool)
            or not isinstance(self.parameter_cap, int)
            or self.parameter_cap <= 0
        ):
            raise Hold30AlphaModelError("parameter_cap must be a positive integer")
        if self.hazard_bound_mode not in HOLD30_HAZARD_BOUND_MODES:
            raise Hold30AlphaModelError(
                "hazard_bound_mode must be 'hard_clip' or 'smooth_tanh'"
            )
        if self.mechanism_generation not in {"v3-frozen", "m03r-v1"}:
            raise Hold30AlphaModelError(
                "mechanism_generation must be 'v3-frozen' or 'm03r-v1'"
            )
        if not isinstance(self.exact_hold_mixture, bool):
            raise Hold30AlphaModelError("exact_hold_mixture must be boolean")
        if self.exact_hold_mixture:
            if (
                self.exact_hold_logit_bias is None
                or isinstance(self.exact_hold_logit_bias, bool)
                or not math.isfinite(float(self.exact_hold_logit_bias))
            ):
                raise Hold30AlphaModelError(
                    "exact_hold_mixture requires a finite exact_hold_logit_bias"
                )
        elif self.exact_hold_logit_bias is not None:
            raise Hold30AlphaModelError(
                "exact_hold_logit_bias is forbidden without exact_hold_mixture"
            )
        if self.fixed_hazard_residual is not None:
            fixed = self.fixed_hazard_residual
            if (
                isinstance(fixed, bool)
                or not math.isfinite(float(fixed))
                or not HOLD30_HAZARD_MIN <= float(fixed) <= HOLD30_HAZARD_MAX
            ):
                raise Hold30AlphaModelError(
                    "fixed_hazard_residual must be finite and lie in [-12,12]"
                )
            if self.exact_hold_mixture:
                raise Hold30AlphaModelError(
                    "fixed-hazard comparator cannot learn an exact-hold mixture"
                )
        if (
            m03r_setting is not None
            and m03r_setting.exit_hazard_mode == "learned-age-aware"
            and not self.exact_hold_mixture
        ):
            raise Hold30AlphaModelError(
                "learned M03R exit settings require the hard exact-hold branch"
            )
        if self.mechanism_generation == "v3-frozen":
            if (
                self.hazard_bound_mode != "hard_clip"
                or self.exact_hold_mixture
                or self.fixed_hazard_residual is not None
            ):
                raise Hold30AlphaModelError(
                    "post-v3 hazard options require mechanism_generation='m03r-v1'"
                )
        elif self.hazard_bound_mode != "smooth_tanh":
            raise Hold30AlphaModelError("m03r-v1 requires the smooth_tanh hazard bound")

    @property
    def use_uncertainty(self) -> bool:
        return _alpha_setting_flags(self.setting_id)[1]

    @property
    def use_total_risk_overlay(self) -> bool:
        return _alpha_setting_flags(self.setting_id)[2] == "separate-total-risk-overlay"

    @property
    def use_confidence_scaled_risk(self) -> bool:
        return self.setting_id in M03R_SETTING_IDS

    @property
    def auxiliary_horizons(self) -> tuple[int, ...]:
        return (
            M03R_ALPHA_HORIZONS_TRADING_SESSIONS
            if self.setting_id in M03R_SETTING_IDS
            else HOLD30_ALPHA_HORIZONS
        )


class Hold30AlphaHead(nn.Module):
    """Small shared alpha/hazard/risk head over per-stock market states.

    ``market_hidden`` must not contain holdings or ages.  Those variables enter
    only the hazard branch.  This separation blocks beta/market-timing leakage
    from the canonical stock alpha score.  The optional a06 total-risk overlay
    has its own scalar head and is absent from every other setting.
    """

    def __init__(self, config: Hold30AlphaHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.auxiliary_horizons = config.auxiliary_horizons
        width = config.hidden_dim
        self.downside_head: nn.Module | None = None
        if config.use_uncertainty:
            self.downside_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 1)
            )
        self.auxiliary_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, len(self.auxiliary_horizons))
        )
        self.hazard_features = nn.Sequential(
            nn.Linear(width + 1 + config.age_summary_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.hazard_head = nn.Linear(width, 1)
        self.exact_hold_head: nn.Linear | None = None
        if config.exact_hold_mixture:
            self.exact_hold_head = nn.Linear(width, 1)
        self.active_risk_head: nn.Module | None = None
        self.confidence_head: nn.Module | None = None
        if config.use_confidence_scaled_risk:
            self.confidence_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 1)
            )
        else:
            self.active_risk_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 1)
            )
        self.total_risk_head: nn.Module | None = None
        if config.use_total_risk_overlay:
            self.total_risk_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 1)
            )
        for module in (
            None if self.downside_head is None else self.downside_head[-1],
            self.auxiliary_head[-1],
            self.hazard_head,
            self.exact_hold_head,
            None if self.active_risk_head is None else self.active_risk_head[-1],
            None if self.confidence_head is None else self.confidence_head[-1],
            None if self.total_risk_head is None else self.total_risk_head[-1],
        ):
            if module is None:
                continue
            nn.init.orthogonal_(module.weight, gain=1e-3)
            nn.init.zeros_(module.bias)
        if self.exact_hold_head is not None:
            assert config.exact_hold_logit_bias is not None
            nn.init.constant_(
                self.exact_hold_head.bias,
                float(config.exact_hold_logit_bias),
            )
        if config.fixed_hazard_residual is not None:
            for parameter in self.hazard_head.parameters():
                parameter.requires_grad_(False)
        count = sum(parameter.numel() for parameter in self.parameters())
        if count > config.parameter_cap:
            raise Hold30AlphaModelError(
                f"alpha head has {count:,} parameters, exceeding cap "
                f"{config.parameter_cap:,}"
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        market_hidden: torch.Tensor,
        prev_weights: torch.Tensor,
        age_summaries: torch.Tensor,
        available: torch.Tensor,
    ) -> Hold30AlphaOutput:
        if market_hidden.ndim != 3:
            raise Hold30AlphaModelError("market_hidden must be [batch,asset,hidden]")
        batch, assets, width = market_hidden.shape
        if width != self.config.hidden_dim:
            raise Hold30AlphaModelError("market_hidden width differs from head config")
        if tuple(prev_weights.shape) != (batch, assets):
            raise Hold30AlphaModelError("prev_weights must be [batch,asset]")
        if tuple(age_summaries.shape) != (
            batch,
            assets,
            self.config.age_summary_dim,
        ):
            raise Hold30AlphaModelError("age_summaries has the wrong shape")
        if tuple(available.shape) != (batch, assets) or available.dtype != torch.bool:
            raise Hold30AlphaModelError("available must be boolean [batch,asset]")
        risky = available.clone()
        risky[:, 0] = False

        auxiliary = self.auxiliary_head(market_hidden)
        mean = auxiliary[..., self.auxiliary_horizons.index(30)]
        downside: torch.Tensor | None = None
        if self.downside_head is not None:
            raw_downside = self.downside_head(market_hidden).squeeze(-1)
            assert self.config.uncertainty_log_scale_bounds is not None
            lower, upper = self.config.uncertainty_log_scale_bounds
            downside = torch.exp(raw_downside.clamp(float(lower), float(upper)))
        score = (
            mean
            - float(self.config.downside_penalty_kappa) * downside
            if downside is not None
            else mean
        )
        zero = torch.zeros_like(mean)
        mean = torch.where(risky, mean, zero)
        if downside is not None:
            downside = torch.where(risky, downside, zero)
        score = torch.where(risky, score, zero)
        auxiliary = torch.where(risky.unsqueeze(-1), auxiliary, torch.zeros_like(auxiliary))

        hazard_input = torch.cat(
            (
                market_hidden,
                prev_weights.to(dtype=market_hidden.dtype).unsqueeze(-1),
                age_summaries.to(dtype=market_hidden.dtype),
            ),
            dim=-1,
        )
        hazard_hidden = self.hazard_features(hazard_input)
        learned_raw_hazard = self.hazard_head(hazard_hidden).squeeze(-1)
        if self.config.fixed_hazard_residual is None:
            raw_hazard = learned_raw_hazard
            hazard = bound_hold30_hazard_residual(
                raw_hazard,
                mode=self.config.hazard_bound_mode,
            )
        else:
            raw_hazard = torch.full_like(
                learned_raw_hazard,
                float(self.config.fixed_hazard_residual),
            )
            hazard = raw_hazard
        exact_hold: torch.Tensor | None = None
        if self.exact_hold_head is not None:
            exact_hold = straight_through_exact_hold_decision(
                self.exact_hold_head(hazard_hidden).squeeze(-1)
            )
            exact_hold = torch.where(
                risky,
                exact_hold,
                torch.ones_like(exact_hold),
            )
        # Deliberate CASH/unavailable sentinels are kept out of raw-hazard
        # saturation telemetry by the separate availability mask.
        raw_hazard = torch.where(risky, raw_hazard, torch.zeros_like(raw_hazard))
        hazard = torch.where(
            risky,
            hazard,
            torch.full_like(hazard, HOLD30_HAZARD_MIN),
        )

        mask = risky.to(dtype=market_hidden.dtype).unsqueeze(-1)
        pooled = (market_hidden * mask).sum(dim=1, dtype=torch.float32) / mask.sum(
            dim=1, dtype=torch.float32
        ).clamp_min(1.0)
        signal_confidence: torch.Tensor | None = None
        if self.confidence_head is not None:
            signal_confidence = torch.sigmoid(self.confidence_head(pooled).squeeze(-1))
            active_risk = (
                float(
                    M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
                )
                * signal_confidence
            )
        else:
            assert self.active_risk_head is not None
            assert self.config.active_log_scale_bounds is not None
            lower, upper = self.config.active_log_scale_bounds
            log_scale = (
                self.active_risk_head(pooled)
                .squeeze(-1)
                .clamp(float(lower), float(upper))
            )
            active_risk = float(self.config.te_target) * torch.exp(log_scale)
        total_overlay = (
            None
            if self.total_risk_head is None
            # A06 is architecturally one-way isolated: overlay-only losses may
            # update the overlay head but cannot update the alpha encoder/core.
            # The reverse route is absent because core outputs never consume
            # overlay parameters. The training runtime verifies the same
            # isolation dynamically before its two disjoint optimizer steps.
            else self.total_risk_head(pooled.detach()).squeeze(-1)
        )
        output = Hold30AlphaOutput(
            mean_30d=mean.float(),
            downside_30d=None if downside is None else downside.float(),
            risk_adjusted_score=score.float(),
            auxiliary_mean=auxiliary.float(),
            hazard_residual=hazard.float(),
            raw_hazard_residual=raw_hazard.float(),
            exact_hold_probability=(None if exact_hold is None else exact_hold.float()),
            active_risk_scale=active_risk.float(),
            signal_confidence=(
                None if signal_confidence is None else signal_confidence.float()
            ),
            total_risk_overlay=(
                None if total_overlay is None else total_overlay.float()
            ),
            auxiliary_horizons_trading_sessions=self.auxiliary_horizons,
        )
        output.validate()
        return output


__all__ = [
    "HOLD30_ALPHA_HEAD_PARAMETER_CAP",
    "HOLD30_ALPHA_HORIZONS",
    "HOLD30_ALPHA_MECH8_IDS",
    "Hold30AlphaHead",
    "Hold30AlphaHeadConfig",
    "Hold30AlphaModelError",
    "Hold30AlphaOutput",
]
