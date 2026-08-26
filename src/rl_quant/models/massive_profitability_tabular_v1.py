"""Capacity-controlled tabular models for the Massive P0 tape experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from rl_quant.models.alpha_hierarchical import AlphaDistribution, AlphaDistributionHead
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_TABULAR_V1_SCHEMA = (
    "rl-quant.massive-profitability-tabular-v1"
)
MASSIVE_PROFITABILITY_TABULAR_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1 = (
    "MV02",
    "MV04",
    "MV04-SHUFFLE",
)
MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1 = (
    "MV00",
    *MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1,
)
MASSIVE_PROFITABILITY_HORIZONS_V1 = (1, 5, 21, 63)
MASSIVE_PROFITABILITY_BARS_DIMENSION_V1 = 19
MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1 = 15


class MassiveProfitabilityTabularV1Error(ValueError):
    """A P0 tabular model or input tensor differs from the frozen contract."""


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTabularSpecV1:
    bars_dimension: int = MASSIVE_PROFITABILITY_BARS_DIMENSION_V1
    tape_dimension: int = MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1
    bars_projection_dimension: int = 64
    tape_projection_dimension: int = 64
    staleness_projection_dimension: int = 8
    hidden_dimension: int = 128
    horizon_count: int = len(MASSIVE_PROFITABILITY_HORIZONS_V1)
    dropout_probability: float = 0.05
    protocol_receipt_sha256: str = (
        MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_PROFITABILITY_TABULAR_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_PROFITABILITY_TABULAR_V1_SCHEMA

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_TABULAR_V1_SCHEMA
            or self.bars_dimension != MASSIVE_PROFITABILITY_BARS_DIMENSION_V1
            or self.tape_dimension != MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1
            or self.bars_projection_dimension != 64
            or self.tape_projection_dimension != 64
            or self.staleness_projection_dimension != 8
            or self.hidden_dimension != 128
            or self.horizon_count != len(MASSIVE_PROFITABILITY_HORIZONS_V1)
            or self.dropout_probability != 0.05
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TABULAR_V1_SOURCE_SHA256
        ):
            raise MassiveProfitabilityTabularV1Error(
                "tabular model specification differs"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(
            {
                "schema": self.schema,
                "bars_dimension": self.bars_dimension,
                "tape_dimension": self.tape_dimension,
                "bars_projection_dimension": self.bars_projection_dimension,
                "tape_projection_dimension": self.tape_projection_dimension,
                "staleness_projection_dimension": self.staleness_projection_dimension,
                "hidden_dimension": self.hidden_dimension,
                "horizon_count": self.horizon_count,
                "dropout_probability": self.dropout_probability,
                "protocol_receipt_sha256": self.protocol_receipt_sha256,
                "implementation_source_sha256": self.implementation_source_sha256,
            }
        )


MASSIVE_PROFITABILITY_TABULAR_SPEC_V1 = MassiveProfitabilityTabularSpecV1()


def _validate_inputs(
    *,
    bars_values: torch.Tensor,
    bars_valid: torch.Tensor,
    tape_values: torch.Tensor,
    tape_valid: torch.Tensor,
    source_staleness: torch.Tensor,
) -> None:
    if (
        bars_values.ndim < 2
        or bars_values.shape[-1] != MASSIVE_PROFITABILITY_BARS_DIMENSION_V1
        or tape_values.shape[:-1] != bars_values.shape[:-1]
        or tape_values.shape[-1] != MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1
        or bars_valid.shape != bars_values.shape
        or tape_valid.shape != tape_values.shape
        or bars_valid.dtype != torch.bool
        or tape_valid.dtype != torch.bool
        or source_staleness.shape != bars_values.shape[:-1] + (1,)
        or bars_values.dtype != tape_values.dtype
        or bars_values.dtype != source_staleness.dtype
        or bars_values.device != tape_values.device
        or bars_values.device != bars_valid.device
        or bars_values.device != tape_valid.device
        or bars_values.device != source_staleness.device
        or not bars_values.is_floating_point()
        or not bool(torch.isfinite(bars_values).all())
        or not bool(torch.isfinite(tape_values).all())
        or not bool(torch.isfinite(source_staleness).all())
        or bool((~bars_valid & (bars_values != 0.0)).any())
        or bool((~tape_valid & (tape_values != 0.0)).any())
    ):
        raise MassiveProfitabilityTabularV1Error(
            "tabular model inputs are malformed"
        )


class MassiveProfitabilityTabularModelV1(nn.Module):
    """Two-branch MLP shared by bars-only, real-tape, and tape-placebo runs."""

    def __init__(
        self,
        *,
        setting_id: str,
        spec: MassiveProfitabilityTabularSpecV1 = MASSIVE_PROFITABILITY_TABULAR_SPEC_V1,
    ) -> None:
        super().__init__()
        spec.validate()
        if setting_id not in MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1:
            raise MassiveProfitabilityTabularV1Error(
                "tabular model setting is not trainable"
            )
        self.setting_id = setting_id
        self.spec = spec
        self.bars_projection = nn.Linear(
            2 * spec.bars_dimension, spec.bars_projection_dimension
        )
        self.tape_projection = nn.Linear(
            2 * spec.tape_dimension, spec.tape_projection_dimension
        )
        self.staleness_projection = nn.Linear(
            1, spec.staleness_projection_dimension
        )
        fusion_dimension = (
            spec.bars_projection_dimension
            + spec.tape_projection_dimension
            + spec.staleness_projection_dimension
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dimension, spec.hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(spec.hidden_dimension),
            nn.Dropout(spec.dropout_probability),
            nn.Linear(spec.hidden_dimension, spec.hidden_dimension),
            nn.GELU(),
        )
        self.distribution_head = AlphaDistributionHead(
            spec.hidden_dimension, spec.horizon_count
        )

    def forward(
        self,
        *,
        bars_values: torch.Tensor,
        bars_valid: torch.Tensor,
        tape_values: torch.Tensor,
        tape_valid: torch.Tensor,
        source_staleness: torch.Tensor,
    ) -> AlphaDistribution:
        _validate_inputs(
            bars_values=bars_values,
            bars_valid=bars_valid,
            tape_values=tape_values,
            tape_valid=tape_valid,
            source_staleness=source_staleness,
        )
        if self.setting_id == "MV02":
            tape_values = torch.zeros_like(tape_values)
            tape_valid = torch.zeros_like(tape_valid)
        bars = torch.cat((bars_values, bars_valid.to(bars_values.dtype)), dim=-1)
        tape = torch.cat((tape_values, tape_valid.to(tape_values.dtype)), dim=-1)
        hidden = torch.cat(
            (
                self.bars_projection(bars),
                self.tape_projection(tape),
                self.staleness_projection(source_staleness),
            ),
            dim=-1,
        )
        return self.distribution_head(self.fusion(hidden))


def massive_profitability_mv00_scores_v1(
    *,
    bars_values: torch.Tensor,
    bars_valid: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed bars sanity score on all four equal-status horizons."""

    if (
        bars_values.ndim < 2
        or bars_values.shape[-1] != MASSIVE_PROFITABILITY_BARS_DIMENSION_V1
        or bars_valid.shape != bars_values.shape
        or bars_valid.dtype != torch.bool
        or not bool(torch.isfinite(bars_values).all())
        or bool((~bars_valid & (bars_values != 0.0)).any())
    ):
        raise MassiveProfitabilityTabularV1Error("MV00 bars inputs are malformed")
    indices_and_signs = (
        (6, 1.0),
        (7, 1.0),
        (5, -1.0),
        (14, 0.25),
        (15, -0.25),
    )
    numerator = torch.zeros_like(bars_values[..., 0])
    denominator = torch.zeros_like(numerator)
    for index, sign in indices_and_signs:
        valid = bars_valid[..., index]
        numerator = numerator + torch.where(
            valid, sign * bars_values[..., index], torch.zeros_like(numerator)
        )
        denominator = denominator + valid.to(numerator.dtype) * abs(sign)
    score = numerator / denominator.clamp_min(1.0)
    return score.unsqueeze(-1).expand(*score.shape, len(MASSIVE_PROFITABILITY_HORIZONS_V1))


__all__ = [
    "MASSIVE_PROFITABILITY_BARS_DIMENSION_V1",
    "MASSIVE_PROFITABILITY_HORIZONS_V1",
    "MASSIVE_PROFITABILITY_TABULAR_SPEC_V1",
    "MASSIVE_PROFITABILITY_TABULAR_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1",
    "MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1",
    "MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1",
    "MassiveProfitabilityTabularModelV1",
    "MassiveProfitabilityTabularSpecV1",
    "MassiveProfitabilityTabularV1Error",
    "massive_profitability_mv00_scores_v1",
]
