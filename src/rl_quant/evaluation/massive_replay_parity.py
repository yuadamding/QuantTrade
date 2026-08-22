"""Delayed-stream versus finalized-file replay qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA = "rl-quant.massive-delayed-replay-v1"
REQUIRED_MASSIVE_REPLAY_CANARIES = (
    "correction-activity",
    "early-close-session",
    "normal-session",
    "special-condition",
    "ticker-change-identity",
    "trf-trades",
)


class MassiveReplayParityError(ValueError):
    """Delayed capture and finalized replay do not establish parity."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveReplayParityError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MassiveReplayParityRow:
    security_id: str
    session_date: str
    delayed_event_inventory_sha256: str
    finalized_replay_inventory_sha256: str
    delayed_feature_sha256: str
    finalized_feature_sha256: str
    event_exact: bool
    feature_exact: bool
    failure_reason: str | None

    def validate(self) -> None:
        if not self.security_id or not self.session_date:
            raise MassiveReplayParityError("parity row identity is absent")
        for name in (
            "delayed_event_inventory_sha256",
            "finalized_replay_inventory_sha256",
            "delayed_feature_sha256",
            "finalized_feature_sha256",
        ):
            _digest(name, getattr(self, name))
        if any(not isinstance(value, bool) for value in (self.event_exact, self.feature_exact)):
            raise MassiveReplayParityError("parity flags must be Boolean")
        if self.event_exact != (
            self.delayed_event_inventory_sha256
            == self.finalized_replay_inventory_sha256
        ):
            raise MassiveReplayParityError("event parity flag differs from identities")
        if self.feature_exact != (
            self.delayed_feature_sha256 == self.finalized_feature_sha256
        ):
            raise MassiveReplayParityError("feature parity flag differs from identities")
        if self.event_exact and self.feature_exact:
            if self.failure_reason is not None:
                raise MassiveReplayParityError("exact parity cannot have a failure reason")
        elif not self.failure_reason or self.failure_reason != self.failure_reason.strip():
            raise MassiveReplayParityError("failed parity needs a canonical reason")


@dataclass(frozen=True, slots=True)
class MassiveDelayedReplayAuthority:
    entitlement_receipt_sha256: str
    websocket_capture_receipts: tuple[str, ...]
    finalized_flat_file_receipts: tuple[str, ...]
    correction_semantics_receipt_sha256: str
    condition_authority_receipt_sha256: str
    parity_rows: tuple[MassiveReplayParityRow, ...]
    canary_kinds_present: tuple[str, ...]
    compared_session_count: int
    exact_event_symbol_day_count: int
    failed_event_symbol_days: tuple[str, ...]
    exact_feature_symbol_day_count: int
    failed_feature_symbol_days: tuple[str, ...]
    development_asof_replay_authorized: bool
    historical_asof_replay_authorized: bool
    predictive_training_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "entitlement_receipt_sha256": self.entitlement_receipt_sha256,
            "websocket_capture_receipts": self.websocket_capture_receipts,
            "finalized_flat_file_receipts": self.finalized_flat_file_receipts,
            "correction_semantics_receipt_sha256": self.correction_semantics_receipt_sha256,
            "condition_authority_receipt_sha256": self.condition_authority_receipt_sha256,
            "parity_rows": [asdict(row) for row in self.parity_rows],
            "canary_kinds_present": self.canary_kinds_present,
            "compared_session_count": self.compared_session_count,
            "exact_event_symbol_day_count": self.exact_event_symbol_day_count,
            "failed_event_symbol_days": self.failed_event_symbol_days,
            "exact_feature_symbol_day_count": self.exact_feature_symbol_day_count,
            "failed_feature_symbol_days": self.failed_feature_symbol_days,
            "development_asof_replay_authorized": self.development_asof_replay_authorized,
            "historical_asof_replay_authorized": self.historical_asof_replay_authorized,
            "predictive_training_authorized": self.predictive_training_authorized,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA:
            raise MassiveReplayParityError("delayed replay schema drifted")
        for name in (
            "entitlement_receipt_sha256",
            "correction_semantics_receipt_sha256",
            "condition_authority_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for inventory_name in (
            "websocket_capture_receipts",
            "finalized_flat_file_receipts",
        ):
            inventory = getattr(self, inventory_name)
            if not inventory or inventory != tuple(sorted(set(inventory))):
                raise MassiveReplayParityError(
                    f"{inventory_name} must be sorted, unique, and nonempty"
                )
            for value in inventory:
                _digest(inventory_name, value)
        if not self.parity_rows:
            raise MassiveReplayParityError("delayed replay has no parity rows")
        row_keys = tuple(f"{row.security_id}:{row.session_date}" for row in self.parity_rows)
        if row_keys != tuple(sorted(set(row_keys))):
            raise MassiveReplayParityError("parity rows must be sorted and unique")
        for row in self.parity_rows:
            row.validate()
        sessions = {row.session_date for row in self.parity_rows}
        if self.compared_session_count != len(sessions):
            raise MassiveReplayParityError("compared session count drifted")
        exact_events = sum(row.event_exact for row in self.parity_rows)
        exact_features = sum(row.feature_exact for row in self.parity_rows)
        failed_events = tuple(
            key for key, row in zip(row_keys, self.parity_rows, strict=True) if not row.event_exact
        )
        failed_features = tuple(
            key for key, row in zip(row_keys, self.parity_rows, strict=True) if not row.feature_exact
        )
        if (
            self.exact_event_symbol_day_count != exact_events
            or self.exact_feature_symbol_day_count != exact_features
            or self.failed_event_symbol_days != failed_events
            or self.failed_feature_symbol_days != failed_features
        ):
            raise MassiveReplayParityError("parity inventories do not reconcile")
        if self.canary_kinds_present != tuple(sorted(set(self.canary_kinds_present))):
            raise MassiveReplayParityError("canary inventory is not canonical")
        canaries_complete = set(REQUIRED_MASSIVE_REPLAY_CANARIES).issubset(
            self.canary_kinds_present
        )
        exact = not failed_events and not failed_features and canaries_complete
        if self.development_asof_replay_authorized is not True:
            raise MassiveReplayParityError("validated contracts must allow development replay")
        if self.historical_asof_replay_authorized is not exact:
            raise MassiveReplayParityError("historical replay authority differs from parity")
        if self.predictive_training_authorized:
            raise MassiveReplayParityError(
                "replay parity alone cannot authorize predictive training"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayParityError("delayed replay receipt differs")


def build_massive_delayed_replay_authority(
    rows: Sequence[MassiveReplayParityRow],
    *,
    entitlement_receipt_sha256: str,
    websocket_capture_receipts: Sequence[str],
    finalized_flat_file_receipts: Sequence[str],
    correction_semantics_receipt_sha256: str,
    condition_authority_receipt_sha256: str,
    canary_kinds_present: Sequence[str],
) -> MassiveDelayedReplayAuthority:
    """Issue historical as-of replay only when events, features, and canaries pass."""

    ordered = tuple(sorted(rows, key=lambda row: (row.security_id, row.session_date)))
    for row in ordered:
        row.validate()
    row_keys = tuple(f"{row.security_id}:{row.session_date}" for row in ordered)
    failed_events = tuple(
        key for key, row in zip(row_keys, ordered, strict=True) if not row.event_exact
    )
    failed_features = tuple(
        key for key, row in zip(row_keys, ordered, strict=True) if not row.feature_exact
    )
    canaries = tuple(sorted(set(canary_kinds_present)))
    historical = (
        not failed_events
        and not failed_features
        and set(REQUIRED_MASSIVE_REPLAY_CANARIES).issubset(canaries)
    )
    body = {
        "schema": MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA,
        "entitlement_receipt_sha256": _digest(
            "entitlement receipt", entitlement_receipt_sha256
        ),
        "websocket_capture_receipts": tuple(sorted(set(websocket_capture_receipts))),
        "finalized_flat_file_receipts": tuple(sorted(set(finalized_flat_file_receipts))),
        "correction_semantics_receipt_sha256": _digest(
            "correction semantics receipt", correction_semantics_receipt_sha256
        ),
        "condition_authority_receipt_sha256": _digest(
            "condition authority receipt", condition_authority_receipt_sha256
        ),
        "parity_rows": [asdict(row) for row in ordered],
        "canary_kinds_present": canaries,
        "compared_session_count": len({row.session_date for row in ordered}),
        "exact_event_symbol_day_count": sum(row.event_exact for row in ordered),
        "failed_event_symbol_days": failed_events,
        "exact_feature_symbol_day_count": sum(row.feature_exact for row in ordered),
        "failed_feature_symbol_days": failed_features,
        "development_asof_replay_authorized": True,
        "historical_asof_replay_authorized": historical,
        "predictive_training_authorized": False,
    }
    authority = MassiveDelayedReplayAuthority(
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        websocket_capture_receipts=tuple(sorted(set(websocket_capture_receipts))),
        finalized_flat_file_receipts=tuple(
            sorted(set(finalized_flat_file_receipts))
        ),
        correction_semantics_receipt_sha256=correction_semantics_receipt_sha256,
        condition_authority_receipt_sha256=condition_authority_receipt_sha256,
        parity_rows=ordered,
        canary_kinds_present=canaries,
        compared_session_count=len({row.session_date for row in ordered}),
        exact_event_symbol_day_count=sum(row.event_exact for row in ordered),
        failed_event_symbol_days=failed_events,
        exact_feature_symbol_day_count=sum(row.feature_exact for row in ordered),
        failed_feature_symbol_days=failed_features,
        development_asof_replay_authorized=True,
        historical_asof_replay_authorized=historical,
        predictive_training_authorized=False,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


__all__ = [
    "MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA",
    "REQUIRED_MASSIVE_REPLAY_CANARIES",
    "MassiveDelayedReplayAuthority",
    "MassiveReplayParityError",
    "MassiveReplayParityRow",
    "build_massive_delayed_replay_authority",
]
