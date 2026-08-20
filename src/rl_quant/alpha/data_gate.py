"""Independent reconciliation gate for a materialized PIT alpha dataset."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from rl_quant.alpha.contracts import PITAlphaDataError, PITAlphaDatasetAuthority
from rl_quant.protocol.canonical_artifact import semantic_sha256


PIT_ALPHA_DATA_GATE_SCHEMA = "rl-quant.pit-alpha-data-gate-v1"


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PITAlphaDataError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class IndependentEconomicReconciliation:
    event_id: str
    security_id: str
    internal_value_change: float
    independent_value_change: float
    absolute_tolerance: float
    independent_source_receipt_sha256: str

    def validate(self) -> None:
        for name in ("event_id", "security_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise PITAlphaDataError(f"reconciliation {name} is invalid")
        for name in (
            "internal_value_change",
            "independent_value_change",
            "absolute_tolerance",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise PITAlphaDataError(f"reconciliation {name} is nonfinite")
        if self.absolute_tolerance < 0.0:
            raise PITAlphaDataError("reconciliation tolerance cannot be negative")
        _digest(
            "independent reconciliation source",
            self.independent_source_receipt_sha256,
        )

    @property
    def matched(self) -> bool:
        self.validate()
        return (
            abs(self.internal_value_change - self.independent_value_change)
            <= self.absolute_tolerance
        )

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "event_id": self.event_id,
            "security_id": self.security_id,
            "internal_value_change": self.internal_value_change,
            "independent_value_change": self.independent_value_change,
            "absolute_tolerance": self.absolute_tolerance,
            "independent_source_receipt_sha256": (
                self.independent_source_receipt_sha256
            ),
            "matched": self.matched,
        }


@dataclass(frozen=True, slots=True)
class PITAlphaDataGateEvidence:
    dataset_receipt_sha256: str
    reloaded_dataset_receipt_sha256: str
    first_tensor_materialization_receipt_sha256: str
    second_tensor_materialization_receipt_sha256: str
    reconciliations: tuple[IndependentEconomicReconciliation, ...]
    missing_event_ids: tuple[str, ...]
    mismatched_event_ids: tuple[str, ...]
    independent_source_overlap_event_ids: tuple[str, ...]
    risky_action_count: int
    terminal_event_count: int
    corporate_action_count: int
    passed: bool
    receipt_sha256: str
    schema: str = PIT_ALPHA_DATA_GATE_SCHEMA

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
            "reloaded_dataset_receipt_sha256": self.reloaded_dataset_receipt_sha256,
            "first_tensor_materialization_receipt_sha256": (
                self.first_tensor_materialization_receipt_sha256
            ),
            "second_tensor_materialization_receipt_sha256": (
                self.second_tensor_materialization_receipt_sha256
            ),
            "reconciliations": tuple(row.payload() for row in self.reconciliations),
            "missing_event_ids": self.missing_event_ids,
            "mismatched_event_ids": self.mismatched_event_ids,
            "independent_source_overlap_event_ids": (
                self.independent_source_overlap_event_ids
            ),
            "risky_action_count": self.risky_action_count,
            "terminal_event_count": self.terminal_event_count,
            "corporate_action_count": self.corporate_action_count,
            "passed": self.passed,
        }

    def validate(self) -> None:
        for name in (
            "dataset_receipt_sha256",
            "reloaded_dataset_receipt_sha256",
            "first_tensor_materialization_receipt_sha256",
            "second_tensor_materialization_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for row in self.reconciliations:
            row.validate()
        for name in (
            "missing_event_ids",
            "mismatched_event_ids",
            "independent_source_overlap_event_ids",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise PITAlphaDataError(f"{name} must be sorted and unique")
        for name in (
            "risky_action_count",
            "terminal_event_count",
            "corporate_action_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PITAlphaDataError(f"{name} must be nonnegative")
        expected_pass = (
            self.dataset_receipt_sha256 == self.reloaded_dataset_receipt_sha256
            and self.first_tensor_materialization_receipt_sha256
            == self.second_tensor_materialization_receipt_sha256
            and not self.missing_event_ids
            and not self.mismatched_event_ids
            and not self.independent_source_overlap_event_ids
            and self.risky_action_count > 0
        )
        if self.schema != PIT_ALPHA_DATA_GATE_SCHEMA or self.passed != expected_pass:
            raise PITAlphaDataError("PIT alpha data-gate decision drifted")
        if self.receipt_sha256 != semantic_sha256(self._payload()):
            raise PITAlphaDataError("PIT alpha data-gate receipt drifted")


def evaluate_pit_alpha_data_gate(
    authority: PITAlphaDatasetAuthority,
    *,
    reloaded_dataset_receipt_sha256: str,
    first_tensor_materialization_receipt_sha256: str,
    second_tensor_materialization_receipt_sha256: str,
    reconciliations: Sequence[IndependentEconomicReconciliation],
) -> PITAlphaDataGateEvidence:
    """Require exact reload/tensor identity and independent event reconciliation."""

    authority.validate()
    expected_events = {
        row.event_id: row.security_id for row in authority.corporate_actions
    }
    expected_events.update(
        {row.event_id: row.security_id for row in authority.terminal_events}
    )
    by_event: dict[str, IndependentEconomicReconciliation] = {}
    mismatched: list[str] = []
    overlapping: list[str] = []
    for row in reconciliations:
        row.validate()
        if row.event_id in by_event:
            raise PITAlphaDataError("economic reconciliation event is duplicated")
        by_event[row.event_id] = row
        if expected_events.get(row.event_id) != row.security_id or not row.matched:
            mismatched.append(row.event_id)
        if row.independent_source_receipt_sha256 in authority.manifest.source_receipts:
            overlapping.append(row.event_id)
    extra = set(by_event).difference(expected_events)
    if extra:
        raise PITAlphaDataError("economic reconciliation references an unknown event")
    missing = sorted(set(expected_events).difference(by_event))
    payload_without_receipt = {
        "schema": PIT_ALPHA_DATA_GATE_SCHEMA,
        "dataset_receipt_sha256": authority.manifest.receipt_sha256,
        "reloaded_dataset_receipt_sha256": reloaded_dataset_receipt_sha256,
        "first_tensor_materialization_receipt_sha256": (
            first_tensor_materialization_receipt_sha256
        ),
        "second_tensor_materialization_receipt_sha256": (
            second_tensor_materialization_receipt_sha256
        ),
        "reconciliations": tuple(row.payload() for row in reconciliations),
        "missing_event_ids": tuple(missing),
        "mismatched_event_ids": tuple(sorted(mismatched)),
        "independent_source_overlap_event_ids": tuple(sorted(overlapping)),
        "risky_action_count": len(authority.manifest.action_axis) - 1,
        "terminal_event_count": len(authority.terminal_events),
        "corporate_action_count": len(authority.corporate_actions),
        "passed": (
            authority.manifest.receipt_sha256 == reloaded_dataset_receipt_sha256
            and first_tensor_materialization_receipt_sha256
            == second_tensor_materialization_receipt_sha256
            and not missing
            and not mismatched
            and not overlapping
            and len(authority.manifest.action_axis) > 1
        ),
    }
    result = PITAlphaDataGateEvidence(
        dataset_receipt_sha256=authority.manifest.receipt_sha256,
        reloaded_dataset_receipt_sha256=reloaded_dataset_receipt_sha256,
        first_tensor_materialization_receipt_sha256=(
            first_tensor_materialization_receipt_sha256
        ),
        second_tensor_materialization_receipt_sha256=(
            second_tensor_materialization_receipt_sha256
        ),
        reconciliations=tuple(reconciliations),
        missing_event_ids=tuple(missing),
        mismatched_event_ids=tuple(sorted(mismatched)),
        independent_source_overlap_event_ids=tuple(sorted(overlapping)),
        risky_action_count=len(authority.manifest.action_axis) - 1,
        terminal_event_count=len(authority.terminal_events),
        corporate_action_count=len(authority.corporate_actions),
        passed=bool(payload_without_receipt["passed"]),
        receipt_sha256=semantic_sha256(payload_without_receipt),
    )
    result.validate()
    return result


__all__ = [
    "PIT_ALPHA_DATA_GATE_SCHEMA",
    "IndependentEconomicReconciliation",
    "PITAlphaDataGateEvidence",
    "evaluate_pit_alpha_data_gate",
]
