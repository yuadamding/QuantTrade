"""Committed, byte-derived economic-event inputs for profitability P0 V3."""

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

MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SCHEMA = "rl-quant.massive-economic-event-source-v3"
MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS = (
    "corporate-actions",
    "terminal-outcomes",
    "cash-returns",
)
MASSIVE_ECONOMIC_EVENT_SOURCE_V3_DATASETS = {
    "corporate-actions": "massive-economic-corporate-actions-v3",
    "terminal-outcomes": "massive-economic-terminal-outcomes-v3",
    "cash-returns": "massive-economic-cash-returns-v3",
}
MASSIVE_ECONOMIC_BASE_SEQUENCE_V3 = 2**63 - 1
MASSIVE_ECONOMIC_EFFECTIVE_TIMESTAMP_CONTRACTS_V3 = {
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
MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "source_kinds": MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS,
        "roles": "exactly-one-committed-source-per-kind",
        "rows": "canonical-json-exact-field-inventory",
        "row_receipt": "derived-from-complete-row-and-upstream-source-receipt",
        "event_time": "kind-specific-economic-time-distinct-from-strategy-availability",
        "effective_timestamp_contracts": MASSIVE_ECONOMIC_EFFECTIVE_TIMESTAMP_CONTRACTS_V3,
        "event_order": "globally-unique-effective-time-sequence-event-id",
        "event_sequence": "nonnegative-and-strictly-below-base-sentinel",
        "base_sequence": MASSIVE_ECONOMIC_BASE_SEQUENCE_V3,
        "identity": "permanent-security-authority-receipt",
    }
)
MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SCHEMA,
        "source_kinds": MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS,
    }
)


class MassiveEconomicEventSourceV3Error(ValueError):
    """Economic-event source bytes or their authority chain differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicEventSourceV3Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicEventSourceV3Error(f"{name} must be nonnegative")
    return value


def _event_sequence(name: str, value: object) -> int:
    sequence = _nonnegative_int(name, value)
    if sequence >= MASSIVE_ECONOMIC_BASE_SEQUENCE_V3:
        raise MassiveEconomicEventSourceV3Error(
            f"{name} must be below the base sentinel"
        )
    return sequence


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEconomicEventSourceV3Error(f"{name} must be canonical text")
    return value


def _event_id(name: str, value: object) -> str:
    event_id = _text(name, value)
    if event_id.startswith("BASE:"):
        raise MassiveEconomicEventSourceV3Error(
            f"{name} uses the reserved base-position prefix"
        )
    return event_id


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveEconomicEventSourceV3Error(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MassiveEconomicEventSourceV3Error(f"{name} must be finite")
    return normalized


def expected_effective_timestamp_contract_v3(
    *, source_kind: str, event_kind: str
) -> str:
    try:
        return MASSIVE_ECONOMIC_EFFECTIVE_TIMESTAMP_CONTRACTS_V3[source_kind][
            event_kind
        ]
    except KeyError as exc:
        raise MassiveEconomicEventSourceV3Error(
            "economic effective-timestamp contract is unsupported"
        ) from exc


def economic_event_source_row_receipt_v3(
    *, source_kind: str, record: Mapping[str, object]
) -> str:
    """Derive a row receipt; *record* must omit its own receipt field."""

    if source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS:
        raise MassiveEconomicEventSourceV3Error("economic source kind is unsupported")
    if "source_row_receipt_sha256" in record:
        raise MassiveEconomicEventSourceV3Error(
            "economic source row receipt is recursively present"
        )
    return semantic_sha256({"source_kind": source_kind, "record": dict(record)})


@dataclass(frozen=True, slots=True)
class MassiveSourcedCorporateActionV3:
    event: CorporateActionRecord
    event_sequence: int
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
        _event_sequence("corporate-action sequence", self.event_sequence)
        if (
            self.effective_timestamp_contract
            != expected_effective_timestamp_contract_v3(
                source_kind="corporate-actions", event_kind=self.event.kind.value
            )
        ):
            raise MassiveEconomicEventSourceV3Error(
                "corporate effective-timestamp contract differs"
            )
        for value in (
            self.upstream_source_receipt_sha256,
            self.source_row_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("corporate-action source digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV3Error(
                "sourced corporate-action receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveSourcedTerminalEventV3:
    event: TerminalEventRecord
    event_sequence: int
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
        _event_sequence("terminal-event sequence", self.event_sequence)
        if (
            self.effective_timestamp_contract
            != expected_effective_timestamp_contract_v3(
                source_kind="terminal-outcomes", event_kind=self.event.kind.value
            )
        ):
            raise MassiveEconomicEventSourceV3Error(
                "terminal effective-timestamp contract differs"
            )
        for value in (
            self.upstream_source_receipt_sha256,
            self.source_row_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("terminal-event source digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV3Error(
                "sourced terminal-event receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveSourcedCashReturnV3:
    event_id: str
    cash_return: CashReturnRecord
    event_sequence: int
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
        _event_sequence("cash-return sequence", self.event_sequence)
        if (
            self.effective_timestamp_contract
            != expected_effective_timestamp_contract_v3(
                source_kind="cash-returns", event_kind="cash-return"
            )
        ):
            raise MassiveEconomicEventSourceV3Error(
                "cash effective-timestamp contract differs"
            )
        _digest("cash-return row receipt", self.source_row_receipt_sha256)
        _digest("sourced cash-return receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV3Error(
                "sourced cash-return receipt differs"
            )


MassiveSourcedEconomicEventV3 = (
    MassiveSourcedCorporateActionV3
    | MassiveSourcedTerminalEventV3
    | MassiveSourcedCashReturnV3
)


def sourced_economic_event_order_v3(
    row: MassiveSourcedEconomicEventV3,
) -> tuple[int, int, str]:
    """Return the one globally authoritative economic-event order."""

    if isinstance(row, MassiveSourcedCashReturnV3):
        return (
            row.cash_return.effective_at_ms,
            row.event_sequence,
            row.event_id,
        )
    return (row.event.effective_at_ms, row.event_sequence, row.event.event_id)


@dataclass(frozen=True, slots=True)
class MassiveEconomicEventSourceV3:
    source_kind: str
    identity_authority_receipt_sha256: str
    corporate_actions: tuple[MassiveSourcedCorporateActionV3, ...]
    terminal_events: tuple[MassiveSourcedTerminalEventV3, ...]
    cash_returns: tuple[MassiveSourcedCashReturnV3, ...]
    row_inventory_sha256: str
    parser_source_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SCHEMA
            or self.source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS
        ):
            raise MassiveEconomicEventSourceV3Error("economic source identity drifted")
        for value in (
            self.identity_authority_receipt_sha256,
            self.row_inventory_sha256,
            self.parser_source_sha256,
            self.receipt_sha256,
        ):
            _digest("economic source digest", value)
        if self.parser_source_sha256 != MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SOURCE_SHA256:
            raise MassiveEconomicEventSourceV3Error("economic source parser drifted")
        if (
            (self.source_kind != "corporate-actions" and self.corporate_actions)
            or (self.source_kind != "terminal-outcomes" and self.terminal_events)
            or (self.source_kind != "cash-returns" and self.cash_returns)
        ):
            raise MassiveEconomicEventSourceV3Error(
                "economic source contains the wrong record role"
            )
        records: tuple[
            MassiveSourcedCorporateActionV3
            | MassiveSourcedTerminalEventV3
            | MassiveSourcedCashReturnV3,
            ...,
        ] = (
            *self.corporate_actions,
            *self.terminal_events,
            *self.cash_returns,
        )
        keys: list[tuple[int, int, str]] = []
        for row in records:
            row.validate()
            if isinstance(
                row,
                (MassiveSourcedCorporateActionV3, MassiveSourcedTerminalEventV3),
            ):
                keys.append(
                    (row.event.effective_at_ms, row.event_sequence, row.event.event_id)
                )
            else:
                keys.append(
                    (
                        row.cash_return.effective_at_ms,
                        row.event_sequence,
                        row.event_id,
                    )
                )
        if tuple(keys) != tuple(sorted(set(keys))):
            raise MassiveEconomicEventSourceV3Error(
                "economic source rows are not ordered and unique"
            )
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in records)
        ):
            raise MassiveEconomicEventSourceV3Error(
                "economic source row inventory differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V3_DATASETS[self.source_kind]
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveEconomicEventSourceV3Error(
                "economic committed source contract differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV3Error(
                "economic source artifact receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveEconomicEventAuthorityV3:
    identity_authority_receipt_sha256: str
    security_ids: tuple[str, ...]
    sources: tuple[MassiveEconomicEventSourceV3, ...]
    corporate_actions: tuple[MassiveSourcedCorporateActionV3, ...]
    terminal_events: tuple[MassiveSourcedTerminalEventV3, ...]
    cash_returns: tuple[MassiveSourcedCashReturnV3, ...]
    event_inventory_sha256: str
    global_event_order_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _digest("economic authority identity", self.identity_authority_receipt_sha256)
        _digest("economic authority inventory", self.event_inventory_sha256)
        _digest(
            "economic authority global order",
            self.global_event_order_inventory_sha256,
        )
        _digest("economic authority receipt", self.receipt_sha256)
        if (
            not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or tuple(source.source_kind for source in self.sources)
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS
        ):
            raise MassiveEconomicEventSourceV3Error(
                "economic authority inventory is not canonical"
            )
        for source in self.sources:
            source.validate()
            if (
                source.identity_authority_receipt_sha256
                != self.identity_authority_receipt_sha256
            ):
                raise MassiveEconomicEventSourceV3Error(
                    "economic source identity authorities differ"
                )
        if (
            self.corporate_actions
            != tuple(row for source in self.sources for row in source.corporate_actions)
            or self.terminal_events
            != tuple(row for source in self.sources for row in source.terminal_events)
            or self.cash_returns
            != tuple(row for source in self.sources for row in source.cash_returns)
        ):
            raise MassiveEconomicEventSourceV3Error(
                "economic authority rows differ from committed sources"
            )
        event_ids: set[str] = set()
        order_keys: set[tuple[int, int, str]] = set()
        receipts: list[str] = []
        security_events: list[
            MassiveSourcedCorporateActionV3 | MassiveSourcedTerminalEventV3
        ] = [*self.corporate_actions, *self.terminal_events]
        for row in security_events:
            event = row.event
            if event.event_id in event_ids:
                raise MassiveEconomicEventSourceV3Error(
                    "economic event IDs are not globally unique"
                )
            event_ids.add(event.event_id)
            key = sourced_economic_event_order_v3(row)
            if key in order_keys:
                raise MassiveEconomicEventSourceV3Error(
                    "global economic event order is duplicated"
                )
            order_keys.add(key)
            if event.security_id not in self.security_ids or (
                event.successor_security_id is not None
                and event.successor_security_id not in self.security_ids
            ):
                raise MassiveEconomicEventSourceV3Error(
                    "economic event references an unknown security"
                )
            receipts.append(row.receipt_sha256)
        for cash_row in self.cash_returns:
            if cash_row.event_id in event_ids:
                raise MassiveEconomicEventSourceV3Error(
                    "cash event ID is not globally unique"
                )
            event_ids.add(cash_row.event_id)
            key = sourced_economic_event_order_v3(cash_row)
            if key in order_keys:
                raise MassiveEconomicEventSourceV3Error(
                    "global cash-event order is duplicated"
                )
            order_keys.add(key)
            receipts.append(cash_row.receipt_sha256)
        global_rows: tuple[MassiveSourcedEconomicEventV3, ...] = (
            *self.corporate_actions,
            *self.terminal_events,
            *self.cash_returns,
        )
        global_inventory = tuple(
            (
                *sourced_economic_event_order_v3(row),
                row.receipt_sha256,
            )
            for row in sorted(global_rows, key=sourced_economic_event_order_v3)
        )
        if self.event_inventory_sha256 != semantic_sha256(
            tuple(receipts)
        ) or self.global_event_order_inventory_sha256 != semantic_sha256(
            global_inventory
        ):
            raise MassiveEconomicEventSourceV3Error(
                "economic authority event or global-order inventory differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV3Error(
                "economic event authority receipt differs"
            )


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveEconomicEventSourceV3Error(f"{name} field inventory differs")


def _parse_corporate(record: Mapping[str, object]) -> MassiveSourcedCorporateActionV3:
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
        "event_sequence",
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
    if source_row_receipt != economic_event_source_row_receipt_v3(
        source_kind="corporate-actions", record=unsigned
    ):
        raise MassiveEconomicEventSourceV3Error(
            "corporate-action row was not derived from committed fields"
        )
    try:
        raw_successor = record["successor_security_id"]
        if raw_successor is not None and not isinstance(raw_successor, str):
            raise MassiveEconomicEventSourceV3Error(
                "corporate successor security ID is malformed"
            )
        kind = CorporateActionKind(_text("corporate kind", record["kind"]))
        timestamp_contract = _text(
            "corporate effective-timestamp contract",
            record["effective_timestamp_contract"],
        )
        if timestamp_contract != expected_effective_timestamp_contract_v3(
            source_kind="corporate-actions", event_kind=kind.value
        ):
            raise MassiveEconomicEventSourceV3Error(
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
        sequence = _event_sequence("corporate sequence", record["event_sequence"])
        upstream = _digest(
            "corporate upstream source", record["upstream_source_receipt_sha256"]
        )
    except MassiveEconomicEventSourceV3Error:
        raise
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV3Error(
            "corporate-action row is malformed"
        ) from exc
    provisional = MassiveSourcedCorporateActionV3(
        event=event,
        event_sequence=sequence,
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


def _parse_terminal(record: Mapping[str, object]) -> MassiveSourcedTerminalEventV3:
    expected = {
        "event_id",
        "security_id",
        "kind",
        "effective_at_ms",
        "available_at_ms",
        "cash_per_share",
        "successor_security_id",
        "successor_ratio",
        "event_sequence",
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
    if source_row_receipt != economic_event_source_row_receipt_v3(
        source_kind="terminal-outcomes", record=unsigned
    ):
        raise MassiveEconomicEventSourceV3Error(
            "terminal-event row was not derived from committed fields"
        )
    try:
        raw_successor = record["successor_security_id"]
        if raw_successor is not None and not isinstance(raw_successor, str):
            raise MassiveEconomicEventSourceV3Error(
                "terminal successor security ID is malformed"
            )
        kind = TerminalEventKind(_text("terminal kind", record["kind"]))
        timestamp_contract = _text(
            "terminal effective-timestamp contract",
            record["effective_timestamp_contract"],
        )
        if timestamp_contract != expected_effective_timestamp_contract_v3(
            source_kind="terminal-outcomes", event_kind=kind.value
        ):
            raise MassiveEconomicEventSourceV3Error(
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
        sequence = _event_sequence("terminal sequence", record["event_sequence"])
        upstream = _digest(
            "terminal upstream source", record["upstream_source_receipt_sha256"]
        )
    except MassiveEconomicEventSourceV3Error:
        raise
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV3Error(
            "terminal-event row is malformed"
        ) from exc
    provisional = MassiveSourcedTerminalEventV3(
        event=event,
        event_sequence=sequence,
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


def _parse_cash(record: Mapping[str, object]) -> MassiveSourcedCashReturnV3:
    expected = {
        "event_id",
        "effective_at_ms",
        "available_at_ms",
        "one_step_return",
        "event_sequence",
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
    if source_row_receipt != economic_event_source_row_receipt_v3(
        source_kind="cash-returns", record=unsigned
    ):
        raise MassiveEconomicEventSourceV3Error(
            "cash-return row was not derived from committed fields"
        )
    try:
        event_id = _event_id("cash-return event ID", record["event_id"])
        sequence = _event_sequence("cash-return sequence", record["event_sequence"])
        timestamp_contract = _text(
            "cash effective-timestamp contract",
            record["effective_timestamp_contract"],
        )
        if timestamp_contract != expected_effective_timestamp_contract_v3(
            source_kind="cash-returns", event_kind="cash-return"
        ):
            raise MassiveEconomicEventSourceV3Error(
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
    except MassiveEconomicEventSourceV3Error:
        raise
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV3Error("cash-return row is malformed") from exc
    provisional = MassiveSourcedCashReturnV3(
        event_id=event_id,
        cash_return=cash,
        event_sequence=sequence,
        effective_timestamp_contract=timestamp_contract,
        source_row_receipt_sha256=source_row_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def parse_massive_economic_event_source_v3(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveEconomicEventSourceV3:
    """Reopen and parse one exact committed economic-event source."""

    loaded_source.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicEventSourceV3Error(
            "economic event source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicEventSourceV3Error(
            "economic event source is not canonical JSON"
        )
    _exact_keys(
        payload,
        {"schema", "source_kind", "identity_authority_receipt_sha256", "records"},
        name="economic event source",
    )
    if payload["schema"] != MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SCHEMA:
        raise MassiveEconomicEventSourceV3Error("economic event source schema drifted")
    source_kind = payload["source_kind"]
    if (
        not isinstance(source_kind, str)
        or source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS
    ):
        raise MassiveEconomicEventSourceV3Error("economic event source kind is invalid")
    identity_receipt = _digest(
        "economic source identity", payload["identity_authority_receipt_sha256"]
    )
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise MassiveEconomicEventSourceV3Error(
            "economic event source records are malformed"
        )
    if any(not isinstance(record, dict) for record in raw_records):
        raise MassiveEconomicEventSourceV3Error("economic source record is malformed")
    corporate = (
        tuple(_parse_corporate(record) for record in raw_records)
        if source_kind == "corporate-actions"
        else ()
    )
    terminal = (
        tuple(_parse_terminal(record) for record in raw_records)
        if source_kind == "terminal-outcomes"
        else ()
    )
    cash = (
        tuple(_parse_cash(record) for record in raw_records)
        if source_kind == "cash-returns"
        else ()
    )
    records: list[
        MassiveSourcedCorporateActionV3
        | MassiveSourcedTerminalEventV3
        | MassiveSourcedCashReturnV3
    ] = [*corporate, *terminal, *cash]
    provisional = MassiveEconomicEventSourceV3(
        source_kind=source_kind,
        identity_authority_receipt_sha256=identity_receipt,
        corporate_actions=corporate,
        terminal_events=terminal,
        cash_returns=cash,
        row_inventory_sha256=semantic_sha256(
            tuple(record.receipt_sha256 for record in records)
        ),
        parser_source_sha256=MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SOURCE_SHA256,
        loaded_source=loaded_source,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def build_massive_economic_event_authority_v3(
    *,
    root: str | Path,
    loaded_sources: Sequence[LoadedMassiveSourceObject],
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveEconomicEventAuthorityV3:
    """Build the authority only by reparsing the three committed source roles."""

    identity_authority.validate()
    parsed = tuple(
        sorted(
            (
                parse_massive_economic_event_source_v3(
                    root=root, loaded_source=loaded_source
                )
                for loaded_source in loaded_sources
            ),
            key=lambda source: MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS.index(
                source.source_kind
            ),
        )
    )
    if (
        tuple(source.source_kind for source in parsed)
        != MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS
    ):
        raise MassiveEconomicEventSourceV3Error(
            "economic authority requires exactly three source roles"
        )
    corporate = tuple(row for source in parsed for row in source.corporate_actions)
    terminal = tuple(row for source in parsed for row in source.terminal_events)
    cash = tuple(row for source in parsed for row in source.cash_returns)
    receipts = tuple(
        [row.receipt_sha256 for row in corporate]
        + [row.receipt_sha256 for row in terminal]
        + [row.receipt_sha256 for row in cash]
    )
    global_rows: tuple[MassiveSourcedEconomicEventV3, ...] = (
        *corporate,
        *terminal,
        *cash,
    )
    global_inventory = tuple(
        (
            *sourced_economic_event_order_v3(row),
            row.receipt_sha256,
        )
        for row in sorted(global_rows, key=sourced_economic_event_order_v3)
    )
    provisional = MassiveEconomicEventAuthorityV3(
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        security_ids=tuple(
            sorted(row.security_id for row in identity_authority.security_master)
        ),
        sources=parsed,
        corporate_actions=corporate,
        terminal_events=terminal,
        cash_returns=cash,
        event_inventory_sha256=semantic_sha256(receipts),
        global_event_order_inventory_sha256=semantic_sha256(global_inventory),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ECONOMIC_BASE_SEQUENCE_V3",
    "MASSIVE_ECONOMIC_EFFECTIVE_TIMESTAMP_CONTRACTS_V3",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V3_DATASETS",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V3_KINDS",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SCHEMA",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V3_SPEC_SHA256",
    "MassiveEconomicEventAuthorityV3",
    "MassiveEconomicEventSourceV3",
    "MassiveEconomicEventSourceV3Error",
    "MassiveSourcedCashReturnV3",
    "MassiveSourcedCorporateActionV3",
    "MassiveSourcedEconomicEventV3",
    "MassiveSourcedTerminalEventV3",
    "build_massive_economic_event_authority_v3",
    "economic_event_source_row_receipt_v3",
    "expected_effective_timestamp_contract_v3",
    "parse_massive_economic_event_source_v3",
    "sourced_economic_event_order_v3",
]
