"""Frozen Hold-30 registry and package-owned model construction."""
from __future__ import annotations

from dataclasses import dataclass

from rl_quant.models.context_encoder import ContextEncoder, ContextEncoderConfig
from rl_quant.models.daily_policy import DailyCrossSectionConfig
from rl_quant.models.daily_policy import DailyCrossSectionPolicy
from rl_quant.protocol.hold30 import (
    HOLD30_BASE_DESIGN,
    HOLD30_CANONICAL_ID,
    HOLD30_MECH8_BY_ID,
    HOLD30_MECH8_IDS,
    HOLD30_MECH8_SETTINGS,
    HOLD30_PROTOCOL_GENERATION,
    Hold30Setting,
    resolve_hold30_setting,
)
from rl_quant.training.designs import DESIGNS


HOLD30_BAR_FEATURE_DIM = 5
HOLD30_COVARIATE_DIM = 0


@dataclass(frozen=True)
class Hold30ParameterCounts:
    """Unique trainable-parameter inventory required by the frozen manifest.

    The Stage-1 context encoder is optimized before it is frozen and cached,
    while the actor remains trainable during the mechanism screen.  They are
    therefore separate actor-path and shared-pretraining rows, but both count
    toward the experiment's total unique trainable capacity.
    """

    context_encoder: int
    actor_path: int
    total_unique: int


def build_hold30_context_config() -> ContextEncoderConfig:
    """Construct the exact raw-OHLCV-only shared context encoder."""

    design = DESIGNS[HOLD30_BASE_DESIGN]
    return ContextEncoderConfig(
        bar_feature_dim=HOLD30_BAR_FEATURE_DIM,
        covariate_dim=HOLD30_COVARIATE_DIM,
        d_model=design.d_model,
        n_heads=design.enc_heads,
        n_layers=design.enc_layers,
        feedforward_dim=256,
        dropout=design.dropout,
        max_seconds=design.session_seconds // design.bar_seconds,
        block_seconds=design.block_seconds // design.bar_seconds,
        grad_checkpoint=design.grad_checkpoint,
        stock_chunk=design.enc_stock_chunk,
    )


def build_hold30_policy_config(setting_id: str) -> DailyCrossSectionConfig:
    """Construct the exact compact actor configuration for one stable ID."""

    setting = resolve_hold30_setting(setting_id)
    design = DESIGNS[HOLD30_BASE_DESIGN]
    return DailyCrossSectionConfig(
        context_dim=design.d_model,
        raw_policy_dim=design.raw_policy_dim,
        raw_policy_layers=design.raw_policy_layers,
        raw_policy_heads=design.raw_policy_heads,
        raw_block_seconds=design.block_seconds // design.bar_seconds,
        session_seconds=design.session_seconds // design.bar_seconds,
        token_dim=design.policy_token_dim,
        temporal_layers=design.policy_layers,
        temporal_heads=design.policy_heads,
        daily_lookback=design.daily_lookback,
        max_days=design.episode_len,
        alloc_layers=design.policy_layers,
        alloc_heads=design.policy_heads,
        feedforward_dim=256,
        dropout=design.dropout,
        temperature=design.temperature,
        max_stock_weight=design.max_stock_weight,
        gate_init_bias=design.gate_init_bias,
        grad_checkpoint=design.grad_checkpoint,
        raw_norm=design.raw_norm,
        raw_recent_days=design.raw_recent_days,
        raw_stock_chunk=design.raw_stock_chunk,
        hold30_setting=setting.setting_id,
    )


def hold30_parameter_counts(setting_id: str) -> Hold30ParameterCounts:
    """Return exact non-overlapping counts for a frozen mechanism model."""

    context = ContextEncoder(build_hold30_context_config())
    actor = DailyCrossSectionPolicy(build_hold30_policy_config(setting_id))
    context_count = sum(parameter.numel() for parameter in context.parameters() if parameter.requires_grad)
    actor_count = sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad)
    return Hold30ParameterCounts(
        context_encoder=context_count,
        actor_path=actor_count,
        total_unique=context_count + actor_count,
    )


__all__ = [
    "HOLD30_BASE_DESIGN",
    "HOLD30_CANONICAL_ID",
    "HOLD30_MECH8_BY_ID",
    "HOLD30_MECH8_IDS",
    "HOLD30_MECH8_SETTINGS",
    "HOLD30_PROTOCOL_GENERATION",
    "Hold30Setting",
    "Hold30ParameterCounts",
    "HOLD30_BAR_FEATURE_DIM",
    "HOLD30_COVARIATE_DIM",
    "build_hold30_context_config",
    "build_hold30_policy_config",
    "hold30_parameter_counts",
    "resolve_hold30_setting",
]
