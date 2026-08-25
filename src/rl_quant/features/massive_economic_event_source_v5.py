"""Committed, byte-derived economic-event inputs for profitability P0 V5."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rl_quant.alpha.contracts import (
    CashReturnRecord,
    CorporateActionKind,
    CorporateActionRecord,
    TerminalEventKind,
    TerminalEventRecord,
)
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SCHEMA = "rl-quant.massive-economic-event-source-v5"
MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS = (
    "corporate-actions",
    "terminal-outcomes",
    "cash-returns",
    "economic-order-evidence",
)
MASSIVE_ECONOMIC_EVENT_SOURCE_V5_DATASETS = {
    "corporate-actions": "massive-economic-corporate-actions-v5",
    "terminal-outcomes": "massive-economic-terminal-outcomes-v5",
    "cash-returns": "massive-economic-cash-returns-v5",
    "economic-order-evidence": "massive-economic-order-evidence-v5",
}
MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5 = 2**63 - 1
MASSIVE_ECONOMIC_REVISION_STATUSES_V5 = ("active", "corrected", "cancelled")
MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V5_RECEIPT_SHA256 = semantic_sha256(
    {
        "rule": "single-logical-event-at-effective-time",
        "derived_global_economic_sequence": 0,
        "ambiguous_tie": "requires-committed-provider-order-evidence",
    }
)
MASSIVE_ECONOMIC_EFFECTIVE_TIMESTAMP_CONTRACTS_V5 = {
    "corporate-actions": {
        CorporateActionKind.CASH_DIVIDEND.value: "ex-distribution-economic-time",
        CorporateActionKind.SPECIAL_DIVIDEND.value: "ex-distribution-economic-time",
        CorporateActionKind.RETURN_OF_CAPITAL.value: "ex-distribution-economic-time",
        CorporateActionKind.SPLIT.value: "first-post-split-trading-economic-time",
        CorporateActionKind.REVERSE_SPLIT.value: "first-post-split-trading-economic-time",
        CorporateActionKind.SPIN_OFF.value: "first-separate-entitlement-economic-time",
        CorporateActionKind.MERGER_CASH.value: "legal-disposition-economic-time",
        CorporateActionKind.MERGER_STOCK.value: "legal-disposition-economic-time",
        CorporateActionKind.TENDER_OFFER.value: "accepted-settlement-disposition-time",
        CorporateActionKind.RIGHTS_DISTRIBUTION.value: "rights-entitlement-economic-time",
        CorporateActionKind.TICKER_EXCHANGE_CHANGE.value: "first-new-reference-trading-economic-time",
    },
    "terminal-outcomes": {
        TerminalEventKind.DELISTING_CASH.value: "final-economic-disposition-time",
        TerminalEventKind.MERGER_CASH.value: "legal-disposition-economic-time",
        TerminalEventKind.MERGER_STOCK.value: "legal-disposition-economic-time",
        TerminalEventKind.BANKRUPTCY_RECOVERY.value: "recovery-economic-disposition-time",
        TerminalEventKind.WORTHLESS.value: "authoritative-zero-value-effective-time",
    },
    "cash-returns": {"cash-return": "cash-accrual-period-end-economic-time"},
}
MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "source_kinds": MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS,
        "roles": "exactly-one-committed-source-per-kind",
        "rows": "canonical-json-exact-field-inventory",
        "row_receipt": "derived-from-complete-row-and-upstream-source-receipt",
        "event_time": "kind-specific-economic-time-distinct-from-strategy-availability",
        "effective_timestamp_contracts": MASSIVE_ECONOMIC_EFFECTIVE_TIMESTAMP_CONTRACTS_V5,
        "event_order": (
            "derived-from-committed-provider-order-evidence-or-frozen-single-event-rule"
        ),
        "event_id_role": "audit-identity-only-never-economic-order",
        "event_rows": "contain-no-authorizing-economic-sequence",
        "logical_events": "linear-origin-vintage-revision-and-cancellation-chain",
        "semantic_duplicates": "rejected-across-logical-event-keys",
        "provider_order": (
            "sequence-derived-from-separate-committed-provider-observation-bytes"
        ),
        "single_event_order_rule": (
            MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V5_RECEIPT_SHA256
        ),
        "base_global_economic_sequence": MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5,
        "identity": "permanent-security-authority-receipt",
        "historical_panel_authorized": False,
        "predictive_training_authorized": False,
    }
)
MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SCHEMA,
        "source_kinds": MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS,
    }
)


class MassiveEconomicEventSourceV5Error(ValueError):
    """Economic-event source bytes or their authority chain differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicEventSourceV5Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicEventSourceV5Error(f"{name} must be nonnegative")
    return value


def _global_economic_sequence(name: str, value: object) -> int:
    sequence = _nonnegative_int(name, value)
    if sequence >= MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5:
        raise MassiveEconomicEventSourceV5Error(
            f"{name} must be below the base sentinel"
        )
    return sequence


def _optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _text(name, value)


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEconomicEventSourceV5Error(f"{name} must be canonical text")
    return value


def _event_id(name: str, value: object) -> str:
    event_id = _text(name, value)
    if event_id.startswith("BASE:"):
        raise MassiveEconomicEventSourceV5Error(
            f"{name} uses the reserved base-position prefix"
        )
    return event_id


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveEconomicEventSourceV5Error(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MassiveEconomicEventSourceV5Error(f"{name} must be finite")
    return normalized


def _revision_status(value: object) -> str:
    status = _text("economic revision status", value)
    if status not in MASSIVE_ECONOMIC_REVISION_STATUSES_V5:
        raise MassiveEconomicEventSourceV5Error(
            "economic revision status is unsupported"
        )
    return status


def corporate_economic_fingerprint_v5(event: CorporateActionRecord) -> str:
    """Hash only economically operative corporate-action terms."""

    event.validate()
    return semantic_sha256(
        {
            "family": "security-event",
            "security_id": event.security_id,
            "kind": event.kind.value,
            "effective_at_ms": event.effective_at_ms,
            "cash_per_share": event.cash_per_share,
            "share_ratio": event.share_ratio,
            "successor_security_id": event.successor_security_id,
            "successor_ratio": event.successor_ratio,
            "affected_fraction": event.affected_fraction,
        }
    )


def terminal_economic_fingerprint_v5(event: TerminalEventRecord) -> str:
    """Hash terminal terms in the same semantic namespace as corporate events."""

    event.validate()
    return semantic_sha256(
        {
            "family": "security-event",
            "security_id": event.security_id,
            "kind": event.kind.value,
            "effective_at_ms": event.effective_at_ms,
            "cash_per_share": event.cash_per_share,
            "share_ratio": 1.0,
            "successor_security_id": event.successor_security_id,
            "successor_ratio": event.successor_ratio,
            "affected_fraction": 1.0,
        }
    )


def cash_economic_fingerprint_v5(
    *, effective_at_ms: int, one_step_return: float
) -> str:
    return semantic_sha256(
        {
            "family": "cash-return",
            "effective_at_ms": effective_at_ms,
            "one_step_return": one_step_return,
        }
    )


def expected_effective_timestamp_contract_v5(
    *, source_kind: str, event_kind: str
) -> str:
    try:
        return MASSIVE_ECONOMIC_EFFECTIVE_TIMESTAMP_CONTRACTS_V5[source_kind][
            event_kind
        ]
    except KeyError as exc:
        raise MassiveEconomicEventSourceV5Error(
            "economic effective-timestamp contract is unsupported"
        ) from exc


def economic_event_source_row_receipt_v5(
    *, source_kind: str, record: Mapping[str, object]
) -> str:
    """Derive a row receipt; *record* must omit its own receipt field."""

    if source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS:
        raise MassiveEconomicEventSourceV5Error("economic source kind is unsupported")
    if "source_row_receipt_sha256" in record:
        raise MassiveEconomicEventSourceV5Error(
            "economic source row receipt is recursively present"
        )
    return semantic_sha256({"source_kind": source_kind, "record": dict(record)})


@dataclass(frozen=True, slots=True)
class MassiveSourcedCorporateActionV5:
    event: CorporateActionRecord
    provider_event_key: str
    logical_event_key: str
    revision_id: str
    supersedes_revision_id: str | None
    revision_status: str
    economic_fingerprint_sha256: str
    effective_timestamp_contract: str
    upstream_source_receipt_sha256: str
    source_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.event.validate()
        _event_id("corporate event ID", self.event.event_id)
        _text("corporate provider event key", self.provider_event_key)
        _text("corporate logical event key", self.logical_event_key)
        _text("corporate revision ID", self.revision_id)
        _optional_text("corporate superseded revision ID", self.supersedes_revision_id)
        _revision_status(self.revision_status)
        _digest("corporate economic fingerprint", self.economic_fingerprint_sha256)
        if self.economic_fingerprint_sha256 != corporate_economic_fingerprint_v5(
            self.event
        ):
            raise MassiveEconomicEventSourceV5Error(
                "corporate economic fingerprint differs"
            )
        if (
            self.effective_timestamp_contract
            != expected_effective_timestamp_contract_v5(
                source_kind="corporate-actions", event_kind=self.event.kind.value
            )
        ):
            raise MassiveEconomicEventSourceV5Error(
                "corporate effective-timestamp contract differs"
            )
        for value in (
            self.upstream_source_receipt_sha256,
            self.source_row_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("corporate-action source digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV5Error(
                "sourced corporate-action receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveSourcedTerminalEventV5:
    event: TerminalEventRecord
    provider_event_key: str
    logical_event_key: str
    revision_id: str
    supersedes_revision_id: str | None
    revision_status: str
    economic_fingerprint_sha256: str
    effective_timestamp_contract: str
    upstream_source_receipt_sha256: str
    source_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.event.validate()
        _event_id("terminal event ID", self.event.event_id)
        _text("terminal provider event key", self.provider_event_key)
        _text("terminal logical event key", self.logical_event_key)
        _text("terminal revision ID", self.revision_id)
        _optional_text("terminal superseded revision ID", self.supersedes_revision_id)
        _revision_status(self.revision_status)
        _digest("terminal economic fingerprint", self.economic_fingerprint_sha256)
        if self.economic_fingerprint_sha256 != terminal_economic_fingerprint_v5(
            self.event
        ):
            raise MassiveEconomicEventSourceV5Error(
                "terminal economic fingerprint differs"
            )
        if (
            self.effective_timestamp_contract
            != expected_effective_timestamp_contract_v5(
                source_kind="terminal-outcomes", event_kind=self.event.kind.value
            )
        ):
            raise MassiveEconomicEventSourceV5Error(
                "terminal effective-timestamp contract differs"
            )
        for value in (
            self.upstream_source_receipt_sha256,
            self.source_row_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("terminal-event source digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV5Error(
                "sourced terminal-event receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveSourcedCashReturnV5:
    event_id: str
    cash_return: CashReturnRecord
    provider_event_key: str
    logical_event_key: str
    revision_id: str
    supersedes_revision_id: str | None
    revision_status: str
    economic_fingerprint_sha256: str
    effective_timestamp_contract: str
    source_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _event_id("cash-return event ID", self.event_id)
        self.cash_return.validate()
        _text("cash provider event key", self.provider_event_key)
        _text("cash logical event key", self.logical_event_key)
        _text("cash revision ID", self.revision_id)
        _optional_text("cash superseded revision ID", self.supersedes_revision_id)
        _revision_status(self.revision_status)
        _digest("cash economic fingerprint", self.economic_fingerprint_sha256)
        if self.economic_fingerprint_sha256 != cash_economic_fingerprint_v5(
            effective_at_ms=self.cash_return.effective_at_ms,
            one_step_return=self.cash_return.one_step_return,
        ):
            raise MassiveEconomicEventSourceV5Error("cash economic fingerprint differs")
        if (
            self.effective_timestamp_contract
            != expected_effective_timestamp_contract_v5(
                source_kind="cash-returns", event_kind="cash-return"
            )
        ):
            raise MassiveEconomicEventSourceV5Error(
                "cash effective-timestamp contract differs"
            )
        _digest("cash-return row receipt", self.source_row_receipt_sha256)
        _digest("sourced cash-return receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV5Error(
                "sourced cash-return receipt differs"
            )


MassiveSourcedEconomicEventV5 = (
    MassiveSourcedCorporateActionV5
    | MassiveSourcedTerminalEventV5
    | MassiveSourcedCashReturnV5
)


def sourced_economic_event_identity_v5(
    row: MassiveSourcedEconomicEventV5,
) -> str:
    """Return the audit identity, which never resolves an economic-order tie."""

    if isinstance(row, MassiveSourcedCashReturnV5):
        return row.event_id
    return row.event.event_id


def sourced_logical_event_key_v5(row: MassiveSourcedEconomicEventV5) -> str:
    return row.logical_event_key


def sourced_revision_id_v5(row: MassiveSourcedEconomicEventV5) -> str:
    return row.revision_id


def sourced_effective_at_ms_v5(row: MassiveSourcedEconomicEventV5) -> int:
    if isinstance(row, MassiveSourcedCashReturnV5):
        return row.cash_return.effective_at_ms
    return row.event.effective_at_ms


def sourced_available_at_ms_v5(row: MassiveSourcedEconomicEventV5) -> int:
    if isinstance(row, MassiveSourcedCashReturnV5):
        return row.cash_return.available_at_ms
    return row.event.available_at_ms


@dataclass(frozen=True, slots=True)
class MassiveProviderEconomicOrderRowV5:
    logical_event_key: str
    effective_at_ms: int
    provider_global_economic_sequence: int
    provider_order_source_receipt_sha256: str
    provider_row_locator: str
    source_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _text("provider-order logical event key", self.logical_event_key)
        _nonnegative_int("provider-order effective time", self.effective_at_ms)
        _global_economic_sequence(
            "provider global economic sequence",
            self.provider_global_economic_sequence,
        )
        _digest(
            "provider order source receipt",
            self.provider_order_source_receipt_sha256,
        )
        _text("provider order row locator", self.provider_row_locator)
        _digest("provider order row receipt", self.source_row_receipt_sha256)
        _digest("provider order receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV5Error(
                "provider economic-order row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveEconomicOrderEvidenceV5:
    logical_event_key: str
    effective_at_ms: int
    derived_global_economic_sequence: int
    derivation_kind: str
    provider_order_source_receipt_sha256: str | None
    canonical_order_rule_receipt_sha256: str | None
    input_event_inventory_sha256: str
    provider_order_row_receipt_sha256: str | None
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _text("order-evidence logical event key", self.logical_event_key)
        _nonnegative_int("order-evidence effective time", self.effective_at_ms)
        _global_economic_sequence(
            "derived global economic sequence",
            self.derived_global_economic_sequence,
        )
        if self.derivation_kind not in {"provider-sequence", "frozen-canonical-rule"}:
            raise MassiveEconomicEventSourceV5Error(
                "economic order derivation kind is unsupported"
            )
        _digest("order input inventory", self.input_event_inventory_sha256)
        _digest("economic order evidence receipt", self.receipt_sha256)
        if self.derivation_kind == "provider-sequence":
            if (
                self.provider_order_source_receipt_sha256 is None
                or self.provider_order_row_receipt_sha256 is None
                or self.canonical_order_rule_receipt_sha256 is not None
            ):
                raise MassiveEconomicEventSourceV5Error(
                    "provider order evidence is incomplete"
                )
            _digest(
                "provider order source receipt",
                self.provider_order_source_receipt_sha256,
            )
            _digest(
                "provider order row receipt",
                self.provider_order_row_receipt_sha256,
            )
        elif (
            self.provider_order_source_receipt_sha256 is not None
            or self.provider_order_row_receipt_sha256 is not None
            or self.canonical_order_rule_receipt_sha256
            != MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V5_RECEIPT_SHA256
            or self.derived_global_economic_sequence != 0
        ):
            raise MassiveEconomicEventSourceV5Error(
                "frozen canonical order evidence differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV5Error(
                "economic order evidence receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveOrderedEconomicEventV5:
    source_event: MassiveSourcedEconomicEventV5
    order_evidence: MassiveEconomicOrderEvidenceV5
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.source_event.validate()
        self.order_evidence.validate()
        if (
            self.source_event.logical_event_key != self.order_evidence.logical_event_key
            or sourced_effective_at_ms_v5(self.source_event)
            != self.order_evidence.effective_at_ms
        ):
            raise MassiveEconomicEventSourceV5Error(
                "ordered event differs from its order evidence"
            )
        _digest("ordered economic event receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV5Error(
                "ordered economic event receipt differs"
            )


def ordered_economic_event_order_v5(
    row: MassiveOrderedEconomicEventV5,
) -> tuple[int, int]:
    return (
        row.order_evidence.effective_at_ms,
        row.order_evidence.derived_global_economic_sequence,
    )


def _sourced_revision_canonical_key(
    row: MassiveSourcedEconomicEventV5,
) -> tuple[str, int, str]:
    return (
        row.logical_event_key,
        sourced_available_at_ms_v5(row),
        row.revision_id,
    )


@dataclass(frozen=True, slots=True)
class MassiveEconomicEventSourceV5:
    source_kind: str
    identity_authority_receipt_sha256: str
    corporate_actions: tuple[MassiveSourcedCorporateActionV5, ...]
    terminal_events: tuple[MassiveSourcedTerminalEventV5, ...]
    cash_returns: tuple[MassiveSourcedCashReturnV5, ...]
    provider_order_rows: tuple[MassiveProviderEconomicOrderRowV5, ...]
    row_inventory_sha256: str
    parser_source_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SCHEMA
            or self.source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS
        ):
            raise MassiveEconomicEventSourceV5Error("economic source identity drifted")
        for value in (
            self.identity_authority_receipt_sha256,
            self.row_inventory_sha256,
            self.parser_source_sha256,
            self.receipt_sha256,
        ):
            _digest("economic source digest", value)
        if self.parser_source_sha256 != MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SOURCE_SHA256:
            raise MassiveEconomicEventSourceV5Error("economic source parser drifted")
        if (
            (self.source_kind != "corporate-actions" and self.corporate_actions)
            or (self.source_kind != "terminal-outcomes" and self.terminal_events)
            or (self.source_kind != "cash-returns" and self.cash_returns)
            or (
                self.source_kind != "economic-order-evidence"
                and self.provider_order_rows
            )
        ):
            raise MassiveEconomicEventSourceV5Error(
                "economic source contains the wrong record role"
            )
        records: tuple[
            MassiveSourcedCorporateActionV5
            | MassiveSourcedTerminalEventV5
            | MassiveSourcedCashReturnV5
            | MassiveProviderEconomicOrderRowV5,
            ...,
        ] = (
            *self.corporate_actions,
            *self.terminal_events,
            *self.cash_returns,
            *self.provider_order_rows,
        )
        for row in records:
            row.validate()
        revisions: tuple[MassiveSourcedEconomicEventV5, ...] = (
            *self.corporate_actions,
            *self.terminal_events,
            *self.cash_returns,
        )
        revision_keys = tuple(_sourced_revision_canonical_key(row) for row in revisions)
        order_keys = tuple(
            (
                row.effective_at_ms,
                row.provider_global_economic_sequence,
                row.logical_event_key,
            )
            for row in self.provider_order_rows
        )
        if revision_keys != tuple(sorted(set(revision_keys))) or order_keys != tuple(
            sorted(set(order_keys))
        ):
            raise MassiveEconomicEventSourceV5Error(
                "economic source rows are not canonical and unique"
            )
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in records)
        ):
            raise MassiveEconomicEventSourceV5Error(
                "economic source row inventory differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V5_DATASETS[self.source_kind]
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveEconomicEventSourceV5Error(
                "economic committed source contract differs"
            )
        if self.source_kind == "economic-order-evidence" and any(
            row.provider_order_source_receipt_sha256
            != self.loaded_source.receipt_sha256
            for row in self.provider_order_rows
        ):
            raise MassiveEconomicEventSourceV5Error(
                "provider order rows are not bound to their committed observation bytes"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV5Error(
                "economic source artifact receipt differs"
            )


def _revision_source_role(row: MassiveSourcedEconomicEventV5) -> str:
    if isinstance(row, MassiveSourcedCorporateActionV5):
        return "corporate-actions"
    if isinstance(row, MassiveSourcedTerminalEventV5):
        return "terminal-outcomes"
    return "cash-returns"


def _validate_revision_chains(
    revisions: Sequence[MassiveSourcedEconomicEventV5],
) -> dict[str, tuple[MassiveSourcedEconomicEventV5, ...]]:
    event_ids: set[str] = set()
    revision_ids: set[str] = set()
    fingerprints: dict[str, str] = {}
    grouped: dict[str, list[MassiveSourcedEconomicEventV5]] = {}
    for row in revisions:
        row.validate()
        event_id = sourced_economic_event_identity_v5(row)
        if event_id in event_ids or row.revision_id in revision_ids:
            raise MassiveEconomicEventSourceV5Error(
                "economic event or revision IDs are not globally unique"
            )
        event_ids.add(event_id)
        revision_ids.add(row.revision_id)
        if row.revision_status != "cancelled":
            prior_key = fingerprints.get(row.economic_fingerprint_sha256)
            if prior_key is not None and prior_key != row.logical_event_key:
                raise MassiveEconomicEventSourceV5Error(
                    "semantic economic event is duplicated across logical keys"
                )
            fingerprints[row.economic_fingerprint_sha256] = row.logical_event_key
        grouped.setdefault(row.logical_event_key, []).append(row)

    output: dict[str, tuple[MassiveSourcedEconomicEventV5, ...]] = {}
    for logical_key, candidates in grouped.items():
        provider_keys = {row.provider_event_key for row in candidates}
        roles = {_revision_source_role(row) for row in candidates}
        effective_times = {sourced_effective_at_ms_v5(row) for row in candidates}
        security_ids = {
            row.event.security_id
            for row in candidates
            if isinstance(
                row,
                (MassiveSourcedCorporateActionV5, MassiveSourcedTerminalEventV5),
            )
        }
        if (
            len(provider_keys) != 1
            or len(roles) != 1
            or len(effective_times) != 1
            or len(security_ids) > 1
        ):
            raise MassiveEconomicEventSourceV5Error(
                "logical event revisions change provider identity, role, security, or effective time"
            )
        by_id = {row.revision_id: row for row in candidates}
        roots = tuple(row for row in candidates if row.supersedes_revision_id is None)
        if len(roots) != 1 or roots[0].revision_status != "active":
            raise MassiveEconomicEventSourceV5Error(
                "logical event revision chain lacks one active root"
            )
        child_by_parent: dict[str, MassiveSourcedEconomicEventV5] = {}
        for row in candidates:
            parent = row.supersedes_revision_id
            if parent is None:
                continue
            if parent not in by_id or parent in child_by_parent:
                raise MassiveEconomicEventSourceV5Error(
                    "logical event revision chain branches or has an unknown parent"
                )
            child_by_parent[parent] = row
        chain: list[MassiveSourcedEconomicEventV5] = []
        current = roots[0]
        seen_fingerprints: set[str] = set()
        while True:
            chain.append(current)
            if (
                current.revision_status != "cancelled"
                and current.economic_fingerprint_sha256 in seen_fingerprints
            ):
                raise MassiveEconomicEventSourceV5Error(
                    "logical event revision repeats identical economic terms"
                )
            if current.revision_status != "cancelled":
                seen_fingerprints.add(current.economic_fingerprint_sha256)
            child = child_by_parent.get(current.revision_id)
            if child is None:
                break
            if (
                current.revision_status == "cancelled"
                or child.revision_status not in {"corrected", "cancelled"}
                or sourced_available_at_ms_v5(child)
                <= sourced_available_at_ms_v5(current)
            ):
                raise MassiveEconomicEventSourceV5Error(
                    "logical event revision chronology is invalid"
                )
            current = child
        if len(chain) != len(candidates):
            raise MassiveEconomicEventSourceV5Error(
                "logical event revision chain is disconnected"
            )
        output[logical_key] = tuple(chain)
    return output


def _derive_order_evidence(
    *,
    chains: Mapping[str, tuple[MassiveSourcedEconomicEventV5, ...]],
    provider_rows: Sequence[MassiveProviderEconomicOrderRowV5],
) -> tuple[
    tuple[MassiveEconomicOrderEvidenceV5, ...],
    tuple[MassiveOrderedEconomicEventV5, ...],
]:
    by_time: dict[int, list[str]] = {}
    for logical_key, chain in chains.items():
        by_time.setdefault(sourced_effective_at_ms_v5(chain[0]), []).append(logical_key)
    provider_by_key = {row.logical_event_key: row for row in provider_rows}
    if len(provider_by_key) != len(provider_rows):
        raise MassiveEconomicEventSourceV5Error(
            "provider economic-order logical keys are duplicated"
        )
    evidence: list[MassiveEconomicOrderEvidenceV5] = []
    used_provider_keys: set[str] = set()
    for effective_at_ms, logical_keys_raw in sorted(by_time.items()):
        logical_keys = tuple(sorted(logical_keys_raw))
        input_inventory = semantic_sha256(
            tuple(
                (
                    logical_key,
                    tuple(row.receipt_sha256 for row in chains[logical_key]),
                )
                for logical_key in logical_keys
            )
        )
        if len(logical_keys) == 1:
            logical_key = logical_keys[0]
            if logical_key in provider_by_key:
                raise MassiveEconomicEventSourceV5Error(
                    "unambiguous event supplied unnecessary provider order evidence"
                )
            evidence_provisional = MassiveEconomicOrderEvidenceV5(
                logical_event_key=logical_key,
                effective_at_ms=effective_at_ms,
                derived_global_economic_sequence=0,
                derivation_kind="frozen-canonical-rule",
                provider_order_source_receipt_sha256=None,
                canonical_order_rule_receipt_sha256=(
                    MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V5_RECEIPT_SHA256
                ),
                input_event_inventory_sha256=input_inventory,
                provider_order_row_receipt_sha256=None,
                receipt_sha256="0" * 64,
            )
            evidence.append(
                replace(
                    evidence_provisional,
                    receipt_sha256=semantic_sha256(evidence_provisional.unsigned()),
                )
            )
            continue
        group_provider = tuple(provider_by_key.get(key) for key in logical_keys)
        if any(row is None for row in group_provider):
            raise MassiveEconomicEventSourceV5Error(
                "ambiguous noncommuting tie lacks provider order evidence"
            )
        typed_provider = tuple(
            row
            for row in group_provider
            if isinstance(row, MassiveProviderEconomicOrderRowV5)
        )
        sequences = tuple(
            row.provider_global_economic_sequence for row in typed_provider
        )
        source_receipts = {
            row.provider_order_source_receipt_sha256 for row in typed_provider
        }
        if (
            len(typed_provider) != len(logical_keys)
            or len(set(sequences)) != len(sequences)
            or len(source_receipts) != 1
            or any(row.effective_at_ms != effective_at_ms for row in typed_provider)
        ):
            raise MassiveEconomicEventSourceV5Error(
                "provider order evidence does not uniquely resolve the tie"
            )
        for row in typed_provider:
            used_provider_keys.add(row.logical_event_key)
            evidence_provisional = MassiveEconomicOrderEvidenceV5(
                logical_event_key=row.logical_event_key,
                effective_at_ms=effective_at_ms,
                derived_global_economic_sequence=(
                    row.provider_global_economic_sequence
                ),
                derivation_kind="provider-sequence",
                provider_order_source_receipt_sha256=(
                    row.provider_order_source_receipt_sha256
                ),
                canonical_order_rule_receipt_sha256=None,
                input_event_inventory_sha256=input_inventory,
                provider_order_row_receipt_sha256=row.receipt_sha256,
                receipt_sha256="0" * 64,
            )
            evidence.append(
                replace(
                    evidence_provisional,
                    receipt_sha256=semantic_sha256(evidence_provisional.unsigned()),
                )
            )
    if used_provider_keys != set(provider_by_key):
        raise MassiveEconomicEventSourceV5Error(
            "provider order evidence contains unused or out-of-scope rows"
        )
    evidence_by_key = {row.logical_event_key: row for row in evidence}
    ordered: list[MassiveOrderedEconomicEventV5] = []
    for logical_key, chain in chains.items():
        for source_event in chain:
            ordered_provisional = MassiveOrderedEconomicEventV5(
                source_event=source_event,
                order_evidence=evidence_by_key[logical_key],
                receipt_sha256="0" * 64,
            )
            ordered.append(
                replace(
                    ordered_provisional,
                    receipt_sha256=semantic_sha256(ordered_provisional.unsigned()),
                )
            )
    result_evidence = tuple(
        sorted(
            evidence,
            key=lambda row: (row.effective_at_ms, row.derived_global_economic_sequence),
        )
    )
    result_ordered = tuple(
        sorted(
            ordered,
            key=lambda row: (
                *ordered_economic_event_order_v5(row),
                row.source_event.logical_event_key,
                sourced_available_at_ms_v5(row.source_event),
            ),
        )
    )
    for evidence_row in result_evidence:
        evidence_row.validate()
    for ordered_row in result_ordered:
        ordered_row.validate()
    return result_evidence, result_ordered


@dataclass(frozen=True, slots=True)
class MassiveEconomicEventAuthorityV5:
    identity_authority_receipt_sha256: str
    security_ids: tuple[str, ...]
    sources: tuple[MassiveEconomicEventSourceV5, ...]
    corporate_actions: tuple[MassiveSourcedCorporateActionV5, ...]
    terminal_events: tuple[MassiveSourcedTerminalEventV5, ...]
    cash_returns: tuple[MassiveSourcedCashReturnV5, ...]
    provider_order_rows: tuple[MassiveProviderEconomicOrderRowV5, ...]
    order_evidence: tuple[MassiveEconomicOrderEvidenceV5, ...]
    ordered_revisions: tuple[MassiveOrderedEconomicEventV5, ...]
    event_inventory_sha256: str
    global_event_order_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        for value in (
            self.identity_authority_receipt_sha256,
            self.event_inventory_sha256,
            self.global_event_order_inventory_sha256,
            self.receipt_sha256,
        ):
            _digest("economic authority digest", value)
        if (
            not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or tuple(source.source_kind for source in self.sources)
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS
        ):
            raise MassiveEconomicEventSourceV5Error(
                "economic authority inventory is not canonical"
            )
        for source in self.sources:
            source.validate()
            if (
                source.identity_authority_receipt_sha256
                != self.identity_authority_receipt_sha256
            ):
                raise MassiveEconomicEventSourceV5Error(
                    "economic source identity authorities differ"
                )
        expected_corporate = tuple(
            row for source in self.sources for row in source.corporate_actions
        )
        expected_terminal = tuple(
            row for source in self.sources for row in source.terminal_events
        )
        expected_cash = tuple(
            row for source in self.sources for row in source.cash_returns
        )
        expected_provider = tuple(
            row for source in self.sources for row in source.provider_order_rows
        )
        if (
            self.corporate_actions != expected_corporate
            or self.terminal_events != expected_terminal
            or self.cash_returns != expected_cash
            or self.provider_order_rows != expected_provider
        ):
            raise MassiveEconomicEventSourceV5Error(
                "economic authority rows differ from committed sources"
            )
        revisions: tuple[MassiveSourcedEconomicEventV5, ...] = (
            *self.corporate_actions,
            *self.terminal_events,
            *self.cash_returns,
        )
        security_events: tuple[
            MassiveSourcedCorporateActionV5 | MassiveSourcedTerminalEventV5, ...
        ] = (*self.corporate_actions, *self.terminal_events)
        for row in security_events:
            event = row.event
            if event.security_id not in self.security_ids or (
                event.successor_security_id is not None
                and event.successor_security_id not in self.security_ids
            ):
                raise MassiveEconomicEventSourceV5Error(
                    "economic event references an unknown security"
                )
        chains = _validate_revision_chains(revisions)
        expected_evidence, expected_ordered = _derive_order_evidence(
            chains=chains,
            provider_rows=self.provider_order_rows,
        )
        if (
            self.order_evidence != expected_evidence
            or self.ordered_revisions != expected_ordered
        ):
            raise MassiveEconomicEventSourceV5Error(
                "economic order was not independently evidence-derived"
            )
        event_inventory = semantic_sha256(
            tuple(row.receipt_sha256 for row in (*revisions, *self.provider_order_rows))
        )
        order_inventory = semantic_sha256(
            tuple(
                (
                    *ordered_economic_event_order_v5(row),
                    row.source_event.logical_event_key,
                    row.source_event.revision_id,
                    row.receipt_sha256,
                )
                for row in self.ordered_revisions
            )
        )
        if (
            self.event_inventory_sha256 != event_inventory
            or self.global_event_order_inventory_sha256 != order_inventory
        ):
            raise MassiveEconomicEventSourceV5Error(
                "economic authority event or order inventory differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV5Error(
                "economic event authority receipt differs"
            )


def resolve_massive_economic_events_at_origin_v5(
    *,
    authority: MassiveEconomicEventAuthorityV5,
    decision_at_ms: int,
) -> tuple[
    tuple[MassiveOrderedEconomicEventV5, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Select one active revision per logical event at a decision vintage."""

    authority.validate()
    _nonnegative_int("economic resolution decision time", decision_at_ms)
    revisions: tuple[MassiveSourcedEconomicEventV5, ...] = (
        *authority.corporate_actions,
        *authority.terminal_events,
        *authority.cash_returns,
    )
    chains = _validate_revision_chains(revisions)
    ordered_by_revision = {
        row.source_event.revision_id: row for row in authority.ordered_revisions
    }
    selected: list[MassiveOrderedEconomicEventV5] = []
    future_revision_ids: list[str] = []
    cancelled_logical_keys: list[str] = []
    for logical_key, chain in sorted(chains.items()):
        available = tuple(
            row for row in chain if sourced_available_at_ms_v5(row) <= decision_at_ms
        )
        future_revision_ids.extend(
            row.revision_id
            for row in chain
            if sourced_available_at_ms_v5(row) > decision_at_ms
        )
        if not available:
            continue
        latest = available[-1]
        if latest.revision_status == "cancelled":
            cancelled_logical_keys.append(logical_key)
            continue
        selected.append(ordered_by_revision[latest.revision_id])
    selected_rows = tuple(
        sorted(
            selected,
            key=lambda row: (
                *ordered_economic_event_order_v5(row),
                row.source_event.logical_event_key,
            ),
        )
    )
    return (
        selected_rows,
        tuple(sorted(future_revision_ids)),
        tuple(sorted(cancelled_logical_keys)),
    )


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveEconomicEventSourceV5Error(f"{name} field inventory differs")


def _parse_corporate(record: Mapping[str, object]) -> MassiveSourcedCorporateActionV5:
    expected = {
        "event_id",
        "security_id",
        "kind",
        "effective_at_ms",
        "available_at_ms",
        "cash_per_share",
        "share_ratio",
        "successor_security_id",
        "successor_ratio",
        "affected_fraction",
        "provider_event_key",
        "logical_event_key",
        "revision_id",
        "supersedes_revision_id",
        "revision_status",
        "economic_fingerprint_sha256",
        "effective_timestamp_contract",
        "upstream_source_receipt_sha256",
        "source_row_receipt_sha256",
    }
    _exact_keys(record, expected, name="corporate-action row")
    source_row_receipt = record["source_row_receipt_sha256"]
    _digest("corporate-action row receipt", source_row_receipt)
    unsigned = {
        key: value
        for key, value in record.items()
        if key != "source_row_receipt_sha256"
    }
    if source_row_receipt != economic_event_source_row_receipt_v5(
        source_kind="corporate-actions", record=unsigned
    ):
        raise MassiveEconomicEventSourceV5Error(
            "corporate-action row was not derived from committed fields"
        )
    try:
        raw_successor = record["successor_security_id"]
        if raw_successor is not None and not isinstance(raw_successor, str):
            raise MassiveEconomicEventSourceV5Error(
                "corporate successor security ID is malformed"
            )
        kind = CorporateActionKind(_text("corporate kind", record["kind"]))
        timestamp_contract = _text(
            "corporate effective-timestamp contract",
            record["effective_timestamp_contract"],
        )
        if timestamp_contract != expected_effective_timestamp_contract_v5(
            source_kind="corporate-actions", event_kind=kind.value
        ):
            raise MassiveEconomicEventSourceV5Error(
                "corporate effective-timestamp contract was not kind-derived"
            )
        event = CorporateActionRecord(
            event_id=_event_id("corporate event ID", record["event_id"]),
            security_id=_text("corporate security ID", record["security_id"]),
            kind=kind,
            effective_at_ms=_nonnegative_int(
                "corporate effective time", record["effective_at_ms"]
            ),
            available_at_ms=_nonnegative_int(
                "corporate available time", record["available_at_ms"]
            ),
            cash_per_share=_number("corporate cash", record["cash_per_share"]),
            share_ratio=_number("corporate share ratio", record["share_ratio"]),
            successor_security_id=raw_successor,
            successor_ratio=_number(
                "corporate successor ratio", record["successor_ratio"]
            ),
            affected_fraction=_number(
                "corporate affected fraction", record["affected_fraction"]
            ),
        )
        provider_event_key = _text(
            "corporate provider event key", record["provider_event_key"]
        )
        logical_event_key = _text(
            "corporate logical event key", record["logical_event_key"]
        )
        revision_id = _text("corporate revision ID", record["revision_id"])
        supersedes_revision_id = _optional_text(
            "corporate superseded revision ID", record["supersedes_revision_id"]
        )
        revision_status = _revision_status(record["revision_status"])
        economic_fingerprint = _digest(
            "corporate economic fingerprint", record["economic_fingerprint_sha256"]
        )
        upstream = _digest(
            "corporate upstream source", record["upstream_source_receipt_sha256"]
        )
    except MassiveEconomicEventSourceV5Error:
        raise
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV5Error(
            "corporate-action row is malformed"
        ) from exc
    provisional = MassiveSourcedCorporateActionV5(
        event=event,
        provider_event_key=provider_event_key,
        logical_event_key=logical_event_key,
        revision_id=revision_id,
        supersedes_revision_id=supersedes_revision_id,
        revision_status=revision_status,
        economic_fingerprint_sha256=economic_fingerprint,
        effective_timestamp_contract=timestamp_contract,
        upstream_source_receipt_sha256=upstream,
        source_row_receipt_sha256=source_row_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _parse_terminal(record: Mapping[str, object]) -> MassiveSourcedTerminalEventV5:
    expected = {
        "event_id",
        "security_id",
        "kind",
        "effective_at_ms",
        "available_at_ms",
        "cash_per_share",
        "successor_security_id",
        "successor_ratio",
        "provider_event_key",
        "logical_event_key",
        "revision_id",
        "supersedes_revision_id",
        "revision_status",
        "economic_fingerprint_sha256",
        "effective_timestamp_contract",
        "upstream_source_receipt_sha256",
        "source_row_receipt_sha256",
    }
    _exact_keys(record, expected, name="terminal-event row")
    source_row_receipt = record["source_row_receipt_sha256"]
    _digest("terminal-event row receipt", source_row_receipt)
    unsigned = {
        key: value
        for key, value in record.items()
        if key != "source_row_receipt_sha256"
    }
    if source_row_receipt != economic_event_source_row_receipt_v5(
        source_kind="terminal-outcomes", record=unsigned
    ):
        raise MassiveEconomicEventSourceV5Error(
            "terminal-event row was not derived from committed fields"
        )
    try:
        raw_successor = record["successor_security_id"]
        if raw_successor is not None and not isinstance(raw_successor, str):
            raise MassiveEconomicEventSourceV5Error(
                "terminal successor security ID is malformed"
            )
        kind = TerminalEventKind(_text("terminal kind", record["kind"]))
        timestamp_contract = _text(
            "terminal effective-timestamp contract",
            record["effective_timestamp_contract"],
        )
        if timestamp_contract != expected_effective_timestamp_contract_v5(
            source_kind="terminal-outcomes", event_kind=kind.value
        ):
            raise MassiveEconomicEventSourceV5Error(
                "terminal effective-timestamp contract was not kind-derived"
            )
        event = TerminalEventRecord(
            event_id=_event_id("terminal event ID", record["event_id"]),
            security_id=_text("terminal security ID", record["security_id"]),
            kind=kind,
            effective_at_ms=_nonnegative_int(
                "terminal effective time", record["effective_at_ms"]
            ),
            available_at_ms=_nonnegative_int(
                "terminal available time", record["available_at_ms"]
            ),
            cash_per_share=_number("terminal cash", record["cash_per_share"]),
            successor_security_id=raw_successor,
            successor_ratio=_number(
                "terminal successor ratio", record["successor_ratio"]
            ),
        )
        provider_event_key = _text(
            "terminal provider event key", record["provider_event_key"]
        )
        logical_event_key = _text(
            "terminal logical event key", record["logical_event_key"]
        )
        revision_id = _text("terminal revision ID", record["revision_id"])
        supersedes_revision_id = _optional_text(
            "terminal superseded revision ID", record["supersedes_revision_id"]
        )
        revision_status = _revision_status(record["revision_status"])
        economic_fingerprint = _digest(
            "terminal economic fingerprint", record["economic_fingerprint_sha256"]
        )
        upstream = _digest(
            "terminal upstream source", record["upstream_source_receipt_sha256"]
        )
    except MassiveEconomicEventSourceV5Error:
        raise
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV5Error(
            "terminal-event row is malformed"
        ) from exc
    provisional = MassiveSourcedTerminalEventV5(
        event=event,
        provider_event_key=provider_event_key,
        logical_event_key=logical_event_key,
        revision_id=revision_id,
        supersedes_revision_id=supersedes_revision_id,
        revision_status=revision_status,
        economic_fingerprint_sha256=economic_fingerprint,
        effective_timestamp_contract=timestamp_contract,
        upstream_source_receipt_sha256=upstream,
        source_row_receipt_sha256=source_row_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _parse_cash(record: Mapping[str, object]) -> MassiveSourcedCashReturnV5:
    expected = {
        "event_id",
        "effective_at_ms",
        "available_at_ms",
        "one_step_return",
        "provider_event_key",
        "logical_event_key",
        "revision_id",
        "supersedes_revision_id",
        "revision_status",
        "economic_fingerprint_sha256",
        "effective_timestamp_contract",
        "upstream_source_receipt_sha256",
        "source_row_receipt_sha256",
    }
    _exact_keys(record, expected, name="cash-return row")
    source_row_receipt = record["source_row_receipt_sha256"]
    _digest("cash-return row receipt", source_row_receipt)
    unsigned = {
        key: value
        for key, value in record.items()
        if key != "source_row_receipt_sha256"
    }
    if source_row_receipt != economic_event_source_row_receipt_v5(
        source_kind="cash-returns", record=unsigned
    ):
        raise MassiveEconomicEventSourceV5Error(
            "cash-return row was not derived from committed fields"
        )
    try:
        event_id = _event_id("cash-return event ID", record["event_id"])
        provider_event_key = _text(
            "cash provider event key", record["provider_event_key"]
        )
        logical_event_key = _text("cash logical event key", record["logical_event_key"])
        revision_id = _text("cash revision ID", record["revision_id"])
        supersedes_revision_id = _optional_text(
            "cash superseded revision ID", record["supersedes_revision_id"]
        )
        revision_status = _revision_status(record["revision_status"])
        economic_fingerprint = _digest(
            "cash economic fingerprint", record["economic_fingerprint_sha256"]
        )
        timestamp_contract = _text(
            "cash effective-timestamp contract",
            record["effective_timestamp_contract"],
        )
        if timestamp_contract != expected_effective_timestamp_contract_v5(
            source_kind="cash-returns", event_kind="cash-return"
        ):
            raise MassiveEconomicEventSourceV5Error(
                "cash effective-timestamp contract was not kind-derived"
            )
        upstream = _digest(
            "cash upstream source", record["upstream_source_receipt_sha256"]
        )
        cash = CashReturnRecord(
            effective_at_ms=_nonnegative_int(
                "cash-return effective time", record["effective_at_ms"]
            ),
            available_at_ms=_nonnegative_int(
                "cash-return available time", record["available_at_ms"]
            ),
            one_step_return=_number(
                "cash-return one-step return", record["one_step_return"]
            ),
            source_receipt_sha256=upstream,
        )
    except MassiveEconomicEventSourceV5Error:
        raise
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV5Error("cash-return row is malformed") from exc
    provisional = MassiveSourcedCashReturnV5(
        event_id=event_id,
        cash_return=cash,
        provider_event_key=provider_event_key,
        logical_event_key=logical_event_key,
        revision_id=revision_id,
        supersedes_revision_id=supersedes_revision_id,
        revision_status=revision_status,
        economic_fingerprint_sha256=economic_fingerprint,
        effective_timestamp_contract=timestamp_contract,
        source_row_receipt_sha256=source_row_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _parse_provider_order(
    record: Mapping[str, object],
    *,
    provider_order_source_receipt_sha256: str,
) -> MassiveProviderEconomicOrderRowV5:
    expected = {
        "logical_event_key",
        "effective_at_ms",
        "provider_global_economic_sequence",
        "provider_row_locator",
        "source_row_receipt_sha256",
    }
    _exact_keys(record, expected, name="provider economic-order row")
    source_row_receipt = _digest(
        "provider order row receipt", record["source_row_receipt_sha256"]
    )
    unsigned = {
        key: value
        for key, value in record.items()
        if key != "source_row_receipt_sha256"
    }
    if source_row_receipt != economic_event_source_row_receipt_v5(
        source_kind="economic-order-evidence", record=unsigned
    ):
        raise MassiveEconomicEventSourceV5Error(
            "provider order row was not derived from committed fields"
        )
    provisional = MassiveProviderEconomicOrderRowV5(
        logical_event_key=_text(
            "provider-order logical event key", record["logical_event_key"]
        ),
        effective_at_ms=_nonnegative_int(
            "provider-order effective time", record["effective_at_ms"]
        ),
        provider_global_economic_sequence=_global_economic_sequence(
            "provider global economic sequence",
            record["provider_global_economic_sequence"],
        ),
        provider_order_source_receipt_sha256=_digest(
            "provider order source receipt",
            provider_order_source_receipt_sha256,
        ),
        provider_row_locator=_text(
            "provider order row locator", record["provider_row_locator"]
        ),
        source_row_receipt_sha256=source_row_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def parse_massive_economic_event_source_v5(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveEconomicEventSourceV5:
    """Reopen and parse one exact committed economic-event source."""

    loaded_source.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicEventSourceV5Error(
            "economic event source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicEventSourceV5Error(
            "economic event source is not canonical JSON"
        )
    _exact_keys(
        payload,
        {"schema", "source_kind", "identity_authority_receipt_sha256", "records"},
        name="economic event source",
    )
    if payload["schema"] != MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SCHEMA:
        raise MassiveEconomicEventSourceV5Error("economic event source schema drifted")
    source_kind = payload["source_kind"]
    if (
        not isinstance(source_kind, str)
        or source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS
    ):
        raise MassiveEconomicEventSourceV5Error("economic event source kind is invalid")
    identity_receipt = _digest(
        "economic source identity", payload["identity_authority_receipt_sha256"]
    )
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise MassiveEconomicEventSourceV5Error(
            "economic event source records are malformed"
        )
    if any(not isinstance(record, dict) for record in raw_records):
        raise MassiveEconomicEventSourceV5Error("economic source record is malformed")
    corporate = (
        tuple(
            sorted(
                (_parse_corporate(record) for record in raw_records),
                key=_sourced_revision_canonical_key,
            )
        )
        if source_kind == "corporate-actions"
        else ()
    )
    terminal = (
        tuple(
            sorted(
                (_parse_terminal(record) for record in raw_records),
                key=_sourced_revision_canonical_key,
            )
        )
        if source_kind == "terminal-outcomes"
        else ()
    )
    cash = (
        tuple(
            sorted(
                (_parse_cash(record) for record in raw_records),
                key=_sourced_revision_canonical_key,
            )
        )
        if source_kind == "cash-returns"
        else ()
    )
    provider_order = (
        tuple(
            sorted(
                (
                    _parse_provider_order(
                        record,
                        provider_order_source_receipt_sha256=(
                            loaded_source.receipt_sha256
                        ),
                    )
                    for record in raw_records
                ),
                key=lambda row: (
                    row.effective_at_ms,
                    row.provider_global_economic_sequence,
                    row.logical_event_key,
                ),
            )
        )
        if source_kind == "economic-order-evidence"
        else ()
    )
    records: list[
        MassiveSourcedCorporateActionV5
        | MassiveSourcedTerminalEventV5
        | MassiveSourcedCashReturnV5
        | MassiveProviderEconomicOrderRowV5
    ] = [*corporate, *terminal, *cash, *provider_order]
    provisional = MassiveEconomicEventSourceV5(
        source_kind=source_kind,
        identity_authority_receipt_sha256=identity_receipt,
        corporate_actions=corporate,
        terminal_events=terminal,
        cash_returns=cash,
        provider_order_rows=provider_order,
        row_inventory_sha256=semantic_sha256(
            tuple(record.receipt_sha256 for record in records)
        ),
        parser_source_sha256=MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SOURCE_SHA256,
        loaded_source=loaded_source,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def build_massive_economic_event_authority_v5(
    *,
    root: str | Path,
    loaded_sources: Sequence[LoadedMassiveSourceObject],
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveEconomicEventAuthorityV5:
    """Reparse event revisions and independently derive every economic order."""

    identity_authority.validate()
    parsed = tuple(
        sorted(
            (
                parse_massive_economic_event_source_v5(
                    root=root, loaded_source=loaded_source
                )
                for loaded_source in loaded_sources
            ),
            key=lambda source: MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS.index(
                source.source_kind
            ),
        )
    )
    if (
        tuple(source.source_kind for source in parsed)
        != MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS
    ):
        raise MassiveEconomicEventSourceV5Error(
            "economic authority requires exactly four source roles"
        )
    corporate = tuple(row for source in parsed for row in source.corporate_actions)
    terminal = tuple(row for source in parsed for row in source.terminal_events)
    cash = tuple(row for source in parsed for row in source.cash_returns)
    provider_order = tuple(
        row for source in parsed for row in source.provider_order_rows
    )
    revisions: tuple[MassiveSourcedEconomicEventV5, ...] = (
        *corporate,
        *terminal,
        *cash,
    )
    chains = _validate_revision_chains(revisions)
    order_evidence, ordered_revisions = _derive_order_evidence(
        chains=chains,
        provider_rows=provider_order,
    )
    event_inventory = semantic_sha256(
        tuple(
            row.receipt_sha256
            for row in (
                *revisions,
                *provider_order,
            )
        )
    )
    order_inventory = semantic_sha256(
        tuple(
            (
                *ordered_economic_event_order_v5(row),
                row.source_event.logical_event_key,
                row.source_event.revision_id,
                row.receipt_sha256,
            )
            for row in ordered_revisions
        )
    )
    provisional = MassiveEconomicEventAuthorityV5(
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        security_ids=tuple(
            sorted(row.security_id for row in identity_authority.security_master)
        ),
        sources=parsed,
        corporate_actions=corporate,
        terminal_events=terminal,
        cash_returns=cash,
        provider_order_rows=provider_order,
        order_evidence=order_evidence,
        ordered_revisions=ordered_revisions,
        event_inventory_sha256=event_inventory,
        global_event_order_inventory_sha256=order_inventory,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5",
    "MASSIVE_ECONOMIC_EFFECTIVE_TIMESTAMP_CONTRACTS_V5",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V5_DATASETS",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V5_KINDS",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SCHEMA",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SPEC_SHA256",
    "MASSIVE_ECONOMIC_REVISION_STATUSES_V5",
    "MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V5_RECEIPT_SHA256",
    "MassiveEconomicEventAuthorityV5",
    "MassiveEconomicOrderEvidenceV5",
    "MassiveEconomicEventSourceV5",
    "MassiveEconomicEventSourceV5Error",
    "MassiveOrderedEconomicEventV5",
    "MassiveProviderEconomicOrderRowV5",
    "MassiveSourcedCashReturnV5",
    "MassiveSourcedCorporateActionV5",
    "MassiveSourcedEconomicEventV5",
    "MassiveSourcedTerminalEventV5",
    "build_massive_economic_event_authority_v5",
    "cash_economic_fingerprint_v5",
    "corporate_economic_fingerprint_v5",
    "economic_event_source_row_receipt_v5",
    "expected_effective_timestamp_contract_v5",
    "parse_massive_economic_event_source_v5",
    "ordered_economic_event_order_v5",
    "resolve_massive_economic_events_at_origin_v5",
    "sourced_available_at_ms_v5",
    "sourced_effective_at_ms_v5",
    "sourced_economic_event_identity_v5",
    "sourced_logical_event_key_v5",
    "sourced_revision_id_v5",
    "terminal_economic_fingerprint_v5",
]
