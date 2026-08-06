"""Causal pre-score factor controls for TOP2000 M03R-v7 development.

Factor controls are estimated from the first 63 transitions of the 251-row
observation-only warmup.  The calibration slice may share those context rows
with the episode, but it ends long before the first loss-bearing decision.
The helper accepts typed cache-adapter results rather than bare tensors so
cache, action-axis, date, and slice identities are checked before use.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch

from rl_quant.training.hold30_top2000_development import (
    Top2000Hold30DevelopmentSequence,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_WARMUP_DECISIONS,
)

TOP2000_M03R_V7_FACTOR_CALIBRATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-pre-score-factor-calibration-v2"
)
TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS = 63
TOP2000_M03R_V7_FACTOR_NAMES = (
    "beta-to-c1",
    "63-session-log-return",
    "63-session-volatility",
    "mean-log-volume",
)


class Top2000M03RV7FactorCalibrationError(ValueError):
    """The factor window is not the causal observation-warmup prefix."""


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV7FactorCalibrationError(
            f"{name} must be a lowercase SHA-256 digest"
        )


@dataclass(frozen=True, slots=True)
class Top2000M03RV7WarmupFactorCalibration:
    """Four standardized controls fitted before every loss-bearing origin."""

    loadings: torch.Tensor
    action_ids: tuple[str, ...]
    factor_names: tuple[str, ...]
    cache_sha256: str
    action_hash: str
    calibration_sequence_receipt_sha256: str
    episode_sequence_receipt_sha256: str
    calibration_state_start_index: int
    calibration_state_stop_index_exclusive: int
    episode_state_start_index: int
    episode_state_stop_index_exclusive: int
    first_loss_origin_state_index: int
    calibration_first_date: str
    calibration_last_date: str
    first_loss_origin_date: str
    loadings_sha256: str
    calibration_transition_count: int = TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS
    fit_role: str = "observation-warmup-pre-score-development-calibration-only"
    development_only: bool = True
    outer_evaluation_authorized: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_FACTOR_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "cache_sha256",
            "action_hash",
            "calibration_sequence_receipt_sha256",
            "episode_sequence_receipt_sha256",
            "loadings_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.schema != TOP2000_M03R_V7_FACTOR_CALIBRATION_SCHEMA
            or self.factor_names != TOP2000_M03R_V7_FACTOR_NAMES
            or self.calibration_transition_count
            != TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS
            or self.fit_role
            != "observation-warmup-pre-score-development-calibration-only"
            or not self.development_only
            or self.outer_evaluation_authorized
            or self.promotion_eligible
        ):
            raise Top2000M03RV7FactorCalibrationError(
                "factor calibration must remain fixed, development-only, and pre-score"
            )
        if (
            not isinstance(self.loadings, torch.Tensor)
            or self.loadings.ndim != 2
            or tuple(self.loadings.shape)
            != (len(self.action_ids), len(self.factor_names))
            or not self.loadings.is_floating_point()
            or not bool(torch.isfinite(self.loadings).all())
            or self.loadings.requires_grad
        ):
            raise Top2000M03RV7FactorCalibrationError(
                "loadings must be detached finite floating [action,factor]"
            )
        if not self.action_ids or self.action_ids[0] != "CASH":
            raise Top2000M03RV7FactorCalibrationError(
                "factor action axis must begin with CASH"
            )
        if bool((self.loadings[0] != 0).any()):
            raise Top2000M03RV7FactorCalibrationError(
                "synthetic CASH factor loadings must be zero"
            )
        if _tensor_sha256(self.loadings) != self.loadings_sha256:
            raise Top2000M03RV7FactorCalibrationError(
                "factor-loading content hash mismatch"
            )
        if (
            self.calibration_state_stop_index_exclusive
            - self.calibration_state_start_index
            != self.calibration_transition_count + 1
            or self.calibration_state_start_index != self.episode_state_start_index
            or self.episode_state_stop_index_exclusive
            - self.episode_state_start_index
            != TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
            or self.first_loss_origin_state_index
            != self.episode_state_start_index
            + TOP2000_M03R_V7_DEV_WARMUP_DECISIONS
            or self.calibration_state_stop_index_exclusive
            > self.first_loss_origin_state_index
        ):
            raise Top2000M03RV7FactorCalibrationError(
                "factor calibration must be the 63-transition warmup prefix "
                "and end before the first score origin"
            )
        if not (
            dt.date.fromisoformat(self.calibration_first_date)
            <= dt.date.fromisoformat(self.calibration_last_date)
            < dt.date.fromisoformat(self.first_loss_origin_date)
        ):
            raise Top2000M03RV7FactorCalibrationError(
                "factor calibration dates must precede the first score origin"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "factor_names": list(self.factor_names),
            "cache_sha256": self.cache_sha256,
            "action_hash": self.action_hash,
            "action_ids": list(self.action_ids),
            "calibration_sequence_receipt_sha256": (
                self.calibration_sequence_receipt_sha256
            ),
            "episode_sequence_receipt_sha256": self.episode_sequence_receipt_sha256,
            "calibration_state_start_index": self.calibration_state_start_index,
            "calibration_state_stop_index_exclusive": (
                self.calibration_state_stop_index_exclusive
            ),
            "episode_state_start_index": self.episode_state_start_index,
            "episode_state_stop_index_exclusive": (
                self.episode_state_stop_index_exclusive
            ),
            "first_loss_origin_state_index": self.first_loss_origin_state_index,
            "calibration_first_date": self.calibration_first_date,
            "calibration_last_date": self.calibration_last_date,
            "first_loss_origin_date": self.first_loss_origin_date,
            "calibration_transition_count": self.calibration_transition_count,
            "loadings_sha256": self.loadings_sha256,
            "fit_role": self.fit_role,
            "development_only": self.development_only,
            "outer_evaluation_authorized": self.outer_evaluation_authorized,
            "promotion_eligible": self.promotion_eligible,
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


def _validate_pre_score_identity(
    calibration: Top2000Hold30DevelopmentSequence,
    episode: Top2000Hold30DevelopmentSequence,
) -> None:
    if not isinstance(calibration, Top2000Hold30DevelopmentSequence) or not isinstance(
        episode, Top2000Hold30DevelopmentSequence
    ):
        raise Top2000M03RV7FactorCalibrationError(
            "factor fitting requires typed TOP2000 cache-adapter sequences"
        )
    if (
        calibration.identity.cache_sha256 != episode.identity.cache_sha256
        or calibration.identity.cache_identity != episode.identity.cache_identity
        or calibration.identity.search_identity != episode.identity.search_identity
        or calibration.identity.action_hash != episode.identity.action_hash
        or calibration.action_ids != episode.action_ids
    ):
        raise Top2000M03RV7FactorCalibrationError(
            "calibration and episode must share one verified cache and action axis"
        )
    if (
        calibration.identity.transition_rows
        != TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS
    ):
        raise Top2000M03RV7FactorCalibrationError(
            "factor calibration requires exactly 63 transitions"
        )
    calibration_start = calibration.identity.state_start_index
    calibration_stop = calibration.identity.state_stop_index_exclusive
    episode_start = episode.identity.state_start_index
    episode_stop = episode.identity.state_stop_index_exclusive
    first_loss_origin = episode_start + TOP2000_M03R_V7_DEV_WARMUP_DECISIONS
    if episode_stop - episode_start != TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS:
        raise Top2000M03RV7FactorCalibrationError(
            "factor calibration requires the exact 378-state training episode"
        )
    if calibration_start != episode_start:
        raise Top2000M03RV7FactorCalibrationError(
            "calibration must be the episode's exact observation-warmup prefix"
        )
    if calibration_stop > first_loss_origin:
        raise Top2000M03RV7FactorCalibrationError(
            "future calibration at or after the first score origin is forbidden"
        )
    if (
        len(episode.exchange_dates) <= TOP2000_M03R_V7_DEV_WARMUP_DECISIONS
        or dt.date.fromisoformat(
        calibration.exchange_dates[-1]
        )
        >= dt.date.fromisoformat(
            episode.exchange_dates[TOP2000_M03R_V7_DEV_WARMUP_DECISIONS]
        )
    ):
        raise Top2000M03RV7FactorCalibrationError(
            "calibration date range must end before the first score origin"
        )


def fit_top2000_m03r_v7_warmup_factor_calibration(
    calibration: Top2000Hold30DevelopmentSequence,
    episode: Top2000Hold30DevelopmentSequence,
) -> Top2000M03RV7WarmupFactorCalibration:
    """Fit four controls using only the bound warmup-prefix window."""

    _validate_pre_score_identity(calibration, episode)
    sequence = calibration.sequence
    returns = sequence.asset_returns[:, 0]
    available = sequence.decision_available[:, 0].all(0)
    available[0] = False
    benchmark = sequence.benchmark_net_returns[:, 0]
    centered_market = benchmark - benchmark.mean()
    market_variance = centered_market.square().sum().clamp_min(1.0e-12)
    centered_assets = returns - returns.mean(0, keepdim=True)
    beta = (centered_assets * centered_market.unsqueeze(-1)).sum(0) / market_variance
    momentum = torch.log1p(returns.clamp_min(-0.999999)).sum(0)
    volatility = returns.std(0, unbiased=False)
    daily_ohlcv = sequence.decision_state[:, 0]
    if daily_ohlcv.ndim != 3 or daily_ohlcv.shape[-1] != 5:
        raise Top2000M03RV7FactorCalibrationError(
            "calibration decision state must retain daily OHLCV"
        )
    volume = torch.log1p(daily_ohlcv[..., 4].clamp_min(0.0)).mean(0)
    values = torch.stack((beta, momentum, volatility, volume), dim=-1)
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    selected = values[available]
    if selected.shape[0] <= values.shape[1]:
        raise Top2000M03RV7FactorCalibrationError(
            "factor calibration needs more continuously available risky assets than factors"
        )
    mean = selected.mean(0)
    scale = selected.std(0, unbiased=False).clamp_min(1.0e-6)
    standardized = ((values - mean) / scale).detach()
    standardized[~available] = 0.0
    standardized[0] = 0.0
    loadings_sha256 = _tensor_sha256(standardized)
    return Top2000M03RV7WarmupFactorCalibration(
        loadings=standardized,
        action_ids=calibration.action_ids,
        factor_names=TOP2000_M03R_V7_FACTOR_NAMES,
        cache_sha256=calibration.identity.cache_sha256,
        action_hash=calibration.identity.action_hash,
        calibration_sequence_receipt_sha256=calibration.identity.receipt_sha256,
        episode_sequence_receipt_sha256=episode.identity.receipt_sha256,
        calibration_state_start_index=calibration.identity.state_start_index,
        calibration_state_stop_index_exclusive=(
            calibration.identity.state_stop_index_exclusive
        ),
        episode_state_start_index=episode.identity.state_start_index,
        episode_state_stop_index_exclusive=(
            episode.identity.state_stop_index_exclusive
        ),
        first_loss_origin_state_index=(
            episode.identity.state_start_index
            + TOP2000_M03R_V7_DEV_WARMUP_DECISIONS
        ),
        calibration_first_date=calibration.exchange_dates[0],
        calibration_last_date=calibration.exchange_dates[-1],
        first_loss_origin_date=episode.exchange_dates[
            TOP2000_M03R_V7_DEV_WARMUP_DECISIONS
        ],
        loadings_sha256=loadings_sha256,
    )


__all__ = [
    "TOP2000_M03R_V7_FACTOR_CALIBRATION_SCHEMA",
    "TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS",
    "TOP2000_M03R_V7_FACTOR_NAMES",
    "Top2000M03RV7FactorCalibrationError",
    "Top2000M03RV7WarmupFactorCalibration",
    "fit_top2000_m03r_v7_warmup_factor_calibration",
]
