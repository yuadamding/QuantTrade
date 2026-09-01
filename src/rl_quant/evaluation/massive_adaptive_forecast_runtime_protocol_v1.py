"""Structural runtime contracts shared by adaptive forecast environments.

The profitability environment consumes the same causal forecast-row and
inference-row surface during RL fitting, inner validation, and sealed outer
evaluation.  Concrete archive generations retain their distinct semantic
identities; these protocols describe only the small runtime interface shared
by those generations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastRowV2,
)


class MassiveAdaptiveForecastRuntimeProtocol(Protocol):
    """Runtime forecast rows and immutable lineage consumed by economics."""

    @property
    def semantic_receipt_sha256(self) -> str: ...

    @property
    def row_receipts(self) -> tuple[str, ...]: ...

    @property
    def runtime_rows(self) -> tuple[MassiveAdaptiveForecastRowV2, ...] | None: ...

    @property
    def runtime_forecasts_replayed(self) -> bool: ...

    @property
    def model_state_receipt_sha256(self) -> str: ...

    @property
    def training_window_plan_receipt_sha256(self) -> str: ...

    @property
    def fold_index(self) -> int: ...

    def validate(self) -> None: ...


class MassiveAdaptiveInferenceRowRuntimeProtocol(Protocol):
    """One causal decision row shared by fit, validation, and outer plans."""

    @property
    def decision_session_date(self) -> str: ...

    @property
    def context_session_dates(self) -> tuple[str, ...]: ...

    @property
    def next_session_date(self) -> str: ...

    @property
    def receipt_sha256(self) -> str: ...

    def validate(self, *, maximum_context_sessions: int) -> None: ...


class MassiveAdaptiveInferencePlanRuntimeProtocol(Protocol):
    """Chronological runtime plan consumed by the stateful environment."""

    @property
    def fold_index(self) -> int: ...

    @property
    def semantic_receipt_sha256(self) -> str: ...

    @property
    def maximum_context_sessions(self) -> int: ...

    @property
    def rows(self) -> Sequence[MassiveAdaptiveInferenceRowRuntimeProtocol]: ...

    def validate(self) -> None: ...


__all__ = [
    "MassiveAdaptiveForecastRuntimeProtocol",
    "MassiveAdaptiveInferencePlanRuntimeProtocol",
    "MassiveAdaptiveInferenceRowRuntimeProtocol",
]
