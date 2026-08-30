"""Bounded adaptive RL controls with exact neutral compiler equivalence.

This module defines the policy-facing action surface only.  The action may
rescale forecast buckets, change uncertainty/risk aversion bidirectionally,
or tighten discretionary turnover.  It cannot relax any frozen hard compiler limit,
emit security weights, execute trades, or authorize RL training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_RL_ACTION_V1_SCHEMA = "rl-quant.massive-adaptive-rl-action-v1"
MASSIVE_ADAPTIVE_RL_ACTION_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_ACTION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "bucket_controls": "seven-values-in-minus-one-to-one",
        "bidirectional_scalar_controls": ("uncertainty", "risk"),
        "one_sided_scalar_controls": ("turnover-tightening",),
        "neutral_action": "all-controls-exactly-zero",
        "security_weights": False,
        "profitability_reporting": False,
        "outer": False,
        "lockbox": False,
        "rl_training": False,
    }
)


class MassiveAdaptiveRLActionV1Error(ValueError):
    """Adaptive RL control is out of range or changes a hard constraint."""


def _bounded(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveAdaptiveRLActionV1Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise MassiveAdaptiveRLActionV1Error(f"{name} must lie in [-1, 1]")
    return result


def _unit_interval(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveAdaptiveRLActionV1Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise MassiveAdaptiveRLActionV1Error(f"{name} must lie in [0, 1]")
    return result


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLActionV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLActionV1:
    bucket_controls: tuple[float, ...]
    uncertainty_control: float
    risk_control: float
    turnover_control: float
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_ACTION_V1_SPEC_SHA256
    implementation_source_sha256: str = MASSIVE_ADAPTIVE_RL_ACTION_V1_SOURCE_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_ACTION_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    @property
    def is_neutral(self) -> bool:
        return all(value == 0.0 for value in self.bucket_controls) and all(
            value == 0.0
            for value in (
                self.uncertainty_control,
                self.risk_control,
                self.turnover_control,
            )
        )

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_ACTION_V1_SCHEMA
            or len(self.bucket_controls) != len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_ACTION_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_ACTION_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLActionV1Error(
                "adaptive RL action identity or authorization differs"
            )
        for value in (
            *self.bucket_controls,
            self.uncertainty_control,
            self.risk_control,
        ):
            _bounded("adaptive RL control", value)
        _unit_interval("adaptive RL turnover control", self.turnover_control)
        for digest_value in (
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL action", digest_value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_action_v1(
    *,
    bucket_controls: tuple[float, ...],
    uncertainty_control: float,
    risk_control: float,
    turnover_control: float,
) -> MassiveAdaptiveRLActionV1:
    """Construct one bounded, nonauthorizing compiler-control action."""

    body = {
        "schema": MASSIVE_ADAPTIVE_RL_ACTION_V1_SCHEMA,
        "bucket_controls": tuple(float(value) for value in bucket_controls),
        "uncertainty_control": float(uncertainty_control),
        "risk_control": float(risk_control),
        "turnover_control": float(turnover_control),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_ACTION_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_ACTION_V1_SOURCE_SHA256,
    }
    result = MassiveAdaptiveRLActionV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def neutral_massive_adaptive_rl_action_v1() -> MassiveAdaptiveRLActionV1:
    """Return the unique action that leaves the compiler route unchanged."""

    return build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS),
        uncertainty_control=0.0,
        risk_control=0.0,
        turnover_control=0.0,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_ACTION_V1_SCHEMA",
    "MassiveAdaptiveRLActionV1",
    "MassiveAdaptiveRLActionV1Error",
    "build_massive_adaptive_rl_action_v1",
    "neutral_massive_adaptive_rl_action_v1",
]
