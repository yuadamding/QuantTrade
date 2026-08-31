"""Shared runtime contract for adaptive RL training-forecast authorities.

The V1 synthetic-compatibility authority and the V2 historical fit-only
authority intentionally have different semantic identities.  PPO orchestration
uses only the small common surface declared here, so accepting V2 must not
require an unsafe nominal cast to V1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    MassiveAdaptiveRLTrainingForecastBlockV1,
)


@runtime_checkable
class MassiveAdaptiveRLTrainingForecastAuthorityProtocol(Protocol):
    """Structural contract consumed by PPO and fixed-control orchestration."""

    @property
    def outer_fold_index(self) -> int: ...

    @property
    def block_sessions(self) -> int: ...

    @property
    def blocks(self) -> tuple[MassiveAdaptiveRLTrainingForecastBlockV1, ...]: ...

    @property
    def origin_session_dates(self) -> tuple[str, ...]: ...

    @property
    def block_inventory_sha256(self) -> str: ...

    @property
    def source_data_qualified(self) -> bool: ...

    @property
    def semantic_receipt_sha256(self) -> str: ...

    @property
    def reinforcement_learning_authorized(self) -> bool: ...

    def validate(self) -> None:
        """Fail closed when the concrete authority differs from its protocol."""


__all__ = ["MassiveAdaptiveRLTrainingForecastAuthorityProtocol"]
