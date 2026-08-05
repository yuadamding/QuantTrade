"""Compact benchmark-anchored heads for the Hold-30 alpha v3 screen.

The module is deliberately independent of portfolio accounting.  It turns a
shared per-stock market representation into economically named raw outputs and
constructs the pre-constraint benchmark tilt.  The execution layer remains
responsible for the age ledger, fills, turnover, caps, and costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_HORIZONS,
    HOLD30_ALPHA_MECH8_IDS,
    HOLD30_ALPHA_TE_TARGET_ANNUAL,
    resolve_hold30_alpha_setting,
)

HOLD30_ALPHA_HEAD_PARAMETER_CAP: Final[int] = 1_000_000


class Hold30AlphaModelError(ValueError):
    """A v3 alpha-head input or setting violates the frozen interface."""


@dataclass(frozen=True, slots=True)
class Hold30AlphaOutput:
    """Raw v3 alpha outputs before fill-time portfolio construction."""

    mean_30d: torch.Tensor
    downside_30d: torch.Tensor | None
    risk_adjusted_score: torch.Tensor
    auxiliary_mean: torch.Tensor
    hazard_residual: torch.Tensor
    active_risk_scale: torch.Tensor
    total_risk_overlay: torch.Tensor | None

    def validate(self) -> None:
        if self.mean_30d.ndim != 2:
            raise Hold30AlphaModelError("mean_30d must be [batch,asset]")
        matrix = tuple(self.mean_30d.shape)
        for name in ("risk_adjusted_score", "hazard_residual"):
            value = getattr(self, name)
            if tuple(value.shape) != matrix:
                raise Hold30AlphaModelError(f"{name} must have shape {matrix}")
        if self.downside_30d is not None and tuple(self.downside_30d.shape) != matrix:
            raise Hold30AlphaModelError(f"downside_30d must have shape {matrix}")
        if tuple(self.auxiliary_mean.shape) != (*matrix, len(HOLD30_ALPHA_HORIZONS)):
            raise Hold30AlphaModelError(
                "auxiliary_mean must be [batch,asset,four_horizons]"
            )
        if tuple(self.active_risk_scale.shape) != (matrix[0],):
            raise Hold30AlphaModelError("active_risk_scale must be [batch]")
        if self.total_risk_overlay is not None and tuple(
            self.total_risk_overlay.shape
        ) != (matrix[0],):
            raise Hold30AlphaModelError("total_risk_overlay must be [batch]")
        values = (
            self.mean_30d,
            self.risk_adjusted_score,
            self.auxiliary_mean,
            self.hazard_residual,
            self.active_risk_scale,
        )
        if self.total_risk_overlay is not None:
            values = (*values, self.total_risk_overlay)
        if self.downside_30d is not None:
            values = (*values, self.downside_30d)
        if not all(value.is_floating_point() and bool(torch.isfinite(value).all()) for value in values):
            raise Hold30AlphaModelError("alpha output contains non-finite values")
        if (
            self.downside_30d is not None
            and bool((self.downside_30d < 0).any())
        ) or bool((self.active_risk_scale <= 0).any()):
            raise Hold30AlphaModelError(
                "downside must be nonnegative and active-risk scale strictly positive"
            )


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

    def __post_init__(self) -> None:
        setting = resolve_hold30_alpha_setting(self.setting_id)
        if not setting.supervised_residual_alpha_heads:
            raise Hold30AlphaModelError(
                "supervised alpha heads are exclusive to m03/a04-a07; "
                "m00-m02 must not instantiate them"
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
        if setting.uncertainty_downside_heads:
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

    @property
    def use_uncertainty(self) -> bool:
        return resolve_hold30_alpha_setting(
            self.setting_id
        ).uncertainty_downside_heads

    @property
    def use_total_risk_overlay(self) -> bool:
        return (
            resolve_hold30_alpha_setting(self.setting_id).sharpe_mode
            == "separate-total-risk-overlay"
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
        width = config.hidden_dim
        self.downside_head: nn.Module | None = None
        if config.use_uncertainty:
            self.downside_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 1)
            )
        self.auxiliary_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, len(HOLD30_ALPHA_HORIZONS))
        )
        self.hazard_features = nn.Sequential(
            nn.Linear(width + 1 + config.age_summary_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.hazard_head = nn.Linear(width, 1)
        self.active_risk_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.total_risk_head: nn.Module | None = None
        if config.use_total_risk_overlay:
            self.total_risk_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 1)
            )
        for module in (
            None if self.downside_head is None else self.downside_head[-1],
            self.auxiliary_head[-1],
            self.hazard_head,
            self.active_risk_head[-1],
            None if self.total_risk_head is None else self.total_risk_head[-1],
        ):
            if module is None:
                continue
            nn.init.orthogonal_(module.weight, gain=1e-3)
            nn.init.zeros_(module.bias)
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
        mean = auxiliary[..., HOLD30_ALPHA_HORIZONS.index(30)]
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
        raw_hazard = self.hazard_head(self.hazard_features(hazard_input)).squeeze(-1)
        hazard = torch.where(
            raw_hazard <= -12.0,
            torch.full_like(raw_hazard, -12.0),
            torch.where(
                raw_hazard >= 12.0,
                torch.full_like(raw_hazard, 12.0),
                raw_hazard,
            ),
        )
        hazard = torch.where(risky, hazard, torch.full_like(hazard, -12.0))

        mask = risky.to(dtype=market_hidden.dtype).unsqueeze(-1)
        pooled = (market_hidden * mask).sum(dim=1, dtype=torch.float32) / mask.sum(
            dim=1, dtype=torch.float32
        ).clamp_min(1.0)
        assert self.config.active_log_scale_bounds is not None
        lower, upper = self.config.active_log_scale_bounds
        log_scale = self.active_risk_head(pooled).squeeze(-1).clamp(
            float(lower), float(upper)
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
            active_risk_scale=active_risk.float(),
            total_risk_overlay=(
                None if total_overlay is None else total_overlay.float()
            ),
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
