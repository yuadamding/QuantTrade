"""Typed, fail-closed training-plan receipt for Hold-30 alpha V3.

This receipt carries the exact :class:`Hold30AlphaObjectiveConfig` objects used
by training and invokes their own setting-specific validator.  A manifest hash
without this typed payload is not a resolved scientific plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_MECH8_SETTINGS,
    HOLD30_ALPHA_PROTOCOL_GENERATION,
    HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
    Hold30AlphaCheckpointContract,
)
from rl_quant.protocol.hold30_freeze import sha256_payload
from rl_quant.training.hold30_alpha import (
    Hold30AlphaObjectiveConfig,
    Hold30AlphaTrainingError,
)

HOLD30_ALPHA_TRAINING_PLAN_SCHEMA = "rl-quant.hold30-alpha-training-plan-v3"
HOLD30_ALPHA_CONFIG_SETTING_IDS = tuple(
    setting.setting_id
    for setting in HOLD30_ALPHA_MECH8_SETTINGS
    if setting.objective_mode != "absolute-net-log-return"
)
HOLD30_ALPHA_ABSOLUTE_CONTROLS = tuple(
    (setting.setting_id, setting.objective_mode)
    for setting in HOLD30_ALPHA_MECH8_SETTINGS
    if setting.objective_mode == "absolute-net-log-return"
)


class Hold30AlphaTrainingPlanError(ValueError):
    """The typed V3 training plan is structurally invalid or unresolved."""


def _require_digest(name: str, value: str | None) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30AlphaTrainingPlanError(
            f"{name} must be an explicit lowercase SHA-256 digest"
        )
    return value


@dataclass(frozen=True, slots=True)
class Hold30AlphaTrainingPlan:
    """Exact setting configs, checkpoint contract, and decision provenance."""

    objective_configs: tuple[Hold30AlphaObjectiveConfig, ...]
    checkpoint_contract: Hold30AlphaCheckpointContract = (
        HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT
    )
    scientific_decision_receipt_sha256: str | None = None
    protocol_generation: str = HOLD30_ALPHA_PROTOCOL_GENERATION
    schema: str = HOLD30_ALPHA_TRAINING_PLAN_SCHEMA
    absolute_controls: tuple[tuple[str, str], ...] = HOLD30_ALPHA_ABSOLUTE_CONTROLS

    def __post_init__(self) -> None:
        if self.protocol_generation != HOLD30_ALPHA_PROTOCOL_GENERATION:
            raise Hold30AlphaTrainingPlanError("training plan rejects another generation")
        if self.schema != HOLD30_ALPHA_TRAINING_PLAN_SCHEMA:
            raise Hold30AlphaTrainingPlanError("training-plan schema drifted")
        if self.absolute_controls != HOLD30_ALPHA_ABSOLUTE_CONTROLS:
            raise Hold30AlphaTrainingPlanError(
                "m00/m01 absolute-control objective identities drifted"
            )
        if not isinstance(self.checkpoint_contract, Hold30AlphaCheckpointContract):
            raise Hold30AlphaTrainingPlanError("typed checkpoint contract is required")
        if not isinstance(self.objective_configs, tuple) or any(
            not isinstance(config, Hold30AlphaObjectiveConfig)
            for config in self.objective_configs
        ):
            raise Hold30AlphaTrainingPlanError(
                "objective_configs must contain typed Hold30AlphaObjectiveConfig rows"
            )
        ids = tuple(config.setting_id for config in self.objective_configs)
        if ids != HOLD30_ALPHA_CONFIG_SETTING_IDS:
            raise Hold30AlphaTrainingPlanError(
                "objective configs must match ordered m02/m03/a04/a05/a06/a07 inventory"
            )
        if self.scientific_decision_receipt_sha256 is not None:
            _require_digest(
                "scientific_decision_receipt_sha256",
                self.scientific_decision_receipt_sha256,
            )

    def require_resolved(self) -> None:
        """Invoke the executable config validator for every objective row."""

        if not self.checkpoint_contract.result_moving_thresholds_complete:
            raise Hold30AlphaTrainingPlanError(
                "projection-distance and forced-turnover thresholds are unresolved"
            )
        _require_digest(
            "scientific_decision_receipt_sha256",
            self.scientific_decision_receipt_sha256,
        )
        for config in self.objective_configs:
            if config.qualification_math_test_only:
                raise Hold30AlphaTrainingPlanError(
                    f"{config.setting_id} uses a qualification-only math config"
                )
            try:
                config.require_resolved()
            except Hold30AlphaTrainingError as exc:
                raise Hold30AlphaTrainingPlanError(
                    f"{config.setting_id} objective config is unresolved: {exc}"
                ) from exc

    @property
    def resolved_for_executable(self) -> bool:
        try:
            self.require_resolved()
        except Hold30AlphaTrainingPlanError:
            return False
        return True

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_generation": self.protocol_generation,
            "absolute_controls": [
                {"setting_id": setting_id, "objective_mode": objective_mode}
                for setting_id, objective_mode in self.absolute_controls
            ],
            "objective_configs": [asdict(config) for config in self.objective_configs],
            "checkpoint_contract": asdict(self.checkpoint_contract),
            "scientific_decision_receipt_sha256": (
                self.scientific_decision_receipt_sha256
            ),
            "resolved_for_executable": self.resolved_for_executable,
        }

    @property
    def receipt_id(self) -> str:
        return sha256_payload(self.manifest_payload())


def unresolved_hold30_alpha_training_plan() -> Hold30AlphaTrainingPlan:
    """Build the explicit unresolved plan used by dry-run qualification."""

    return Hold30AlphaTrainingPlan(
        objective_configs=tuple(
            Hold30AlphaObjectiveConfig(setting_id=setting_id)
            for setting_id in HOLD30_ALPHA_CONFIG_SETTING_IDS
        )
    )


__all__ = [
    "HOLD30_ALPHA_ABSOLUTE_CONTROLS",
    "HOLD30_ALPHA_CONFIG_SETTING_IDS",
    "HOLD30_ALPHA_TRAINING_PLAN_SCHEMA",
    "Hold30AlphaTrainingPlan",
    "Hold30AlphaTrainingPlanError",
    "unresolved_hold30_alpha_training_plan",
]
