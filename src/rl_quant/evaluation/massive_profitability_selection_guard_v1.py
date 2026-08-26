"""Pre-P&L execution guards for the Massive P0 recovery canary.

This module deliberately does not calculate or authorize profitability.  It
freezes the two pieces of future execution support that must be checked after
a score has selected a position: the scheduled exit must exist, and an
unresolved terminal fallback must never become a windfall for a short.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.features.massive_profitability_targets_v2 import (
    MassiveProfitabilityTargetsV2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SCHEMA = (
    "rl-quant.massive-profitability-selection-guard-v1"
)
MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_HORIZONS = (1, 5, 21, 63)
MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "selected_missing_exit": "hard-failure",
        "unselected_missing_exit": "permitted",
        "unresolved_terminal_long": "full-minus-one-underlying-return",
        "unresolved_terminal_short": "hard-failure-no-windfall-credit",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveProfitabilitySelectionGuardV1Error(ValueError):
    """A selected position cannot be evaluated without ex-post optimism."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilitySelectionGuardV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySelectedPositionV1:
    decision_session_date: str
    security_id: str
    horizon_sessions: int
    side: str

    def validate(self) -> None:
        if (
            not self.decision_session_date
            or not self.security_id
            or self.horizon_sessions
            not in MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_HORIZONS
            or self.side not in {"long", "short"}
        ):
            raise MassiveProfitabilitySelectionGuardV1Error(
                "selected position identity differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySelectedPositionSupportV1:
    selected_positions: tuple[MassiveProfitabilitySelectedPositionV1, ...]
    target_semantic_receipts: tuple[str, ...]
    selected_target_row_receipts: tuple[str, ...]
    selected_exit_support_complete: bool
    direction_safe_terminal_support_complete: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        keys = tuple(
            (
                row.decision_session_date,
                row.security_id,
                row.horizon_sessions,
                row.side,
            )
            for row in self.selected_positions
        )
        economic_keys = tuple(value[:3] for value in keys)
        if (
            self.schema != MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SOURCE_SHA256
            or not self.selected_positions
            or keys != tuple(sorted(set(keys)))
            or len(economic_keys) != len(set(economic_keys))
            or len(self.target_semantic_receipts)
            != len({row.decision_session_date for row in self.selected_positions})
            or len(self.selected_target_row_receipts) != len(self.selected_positions)
            or not self.selected_exit_support_complete
            or not self.direction_safe_terminal_support_complete
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilitySelectionGuardV1Error(
                "selected-position support or authorization differs"
            )
        for row in self.selected_positions:
            row.validate()
        for value in (
            *self.target_semantic_receipts,
            *self.selected_target_row_receipts,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("selection guard", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilitySelectionGuardV1Error(
                "selection guard semantic receipt differs"
            )


def guard_massive_profitability_selected_positions_v1(
    *,
    selected_positions: Sequence[MassiveProfitabilitySelectedPositionV1],
    targets: Sequence[MassiveProfitabilityTargetsV2],
) -> MassiveProfitabilitySelectedPositionSupportV1:
    """Fail closed on missing selected exits or optimistic short fallbacks."""

    ordered_positions = tuple(
        sorted(
            selected_positions,
            key=lambda row: (
                row.decision_session_date,
                row.security_id,
                row.horizon_sessions,
                row.side,
            ),
        )
    )
    if not ordered_positions:
        raise MassiveProfitabilitySelectionGuardV1Error(
            "selection guard requires at least one selected position"
        )
    for row in ordered_positions:
        row.validate()
    target_by_date: dict[str, MassiveProfitabilityTargetsV2] = {}
    for artifact in targets:
        artifact.validate()
        if artifact.decision_session_date in target_by_date:
            raise MassiveProfitabilitySelectionGuardV1Error(
                "selection guard received duplicate target dates"
            )
        target_by_date[artifact.decision_session_date] = artifact

    selected_receipts: list[str] = []
    horizon_index = {
        value: index
        for index, value in enumerate(MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_HORIZONS)
    }
    for position in ordered_positions:
        artifact = target_by_date.get(position.decision_session_date)
        if artifact is None:
            raise MassiveProfitabilitySelectionGuardV1Error(
                "selected position lacks its target date"
            )
        target_row = next(
            (row for row in artifact.rows if row.security_id == position.security_id),
            None,
        )
        if (
            target_row is None
            or not target_row.valid[horizon_index[position.horizon_sessions]]
        ):
            raise MassiveProfitabilitySelectionGuardV1Error(
                "selected position lacks its scheduled exit fill"
            )
        if position.side == "short" and target_row.conservative_total_loss_fallback:
            raise MassiveProfitabilitySelectionGuardV1Error(
                "unresolved terminal fallback cannot credit a selected short"
            )
        selected_receipts.append(target_row.receipt_sha256)

    target_receipts = tuple(
        target_by_date[value].semantic_receipt_sha256
        for value in sorted({row.decision_session_date for row in ordered_positions})
    )
    body = {
        "schema": MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SCHEMA,
        "selected_positions": tuple(asdict(row) for row in ordered_positions),
        "target_semantic_receipts": target_receipts,
        "selected_target_row_receipts": tuple(selected_receipts),
        "selected_exit_support_complete": True,
        "direction_safe_terminal_support_complete": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SOURCE_SHA256
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilitySelectedPositionSupportV1(
        selected_positions=ordered_positions,
        target_semantic_receipts=target_receipts,
        selected_target_row_receipts=tuple(selected_receipts),
        selected_exit_support_complete=True,
        direction_safe_terminal_support_complete=True,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SPEC_SHA256,
        implementation_source_sha256=(
            MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SOURCE_SHA256
        ),
        semantic_receipt_sha256=semantic_sha256(body),
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_HORIZONS",
    "MASSIVE_PROFITABILITY_SELECTION_GUARD_V1_SCHEMA",
    "MassiveProfitabilitySelectedPositionSupportV1",
    "MassiveProfitabilitySelectedPositionV1",
    "MassiveProfitabilitySelectionGuardV1Error",
    "guard_massive_profitability_selected_positions_v1",
]
