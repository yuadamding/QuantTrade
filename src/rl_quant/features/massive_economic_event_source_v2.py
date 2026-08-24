"""Committed, byte-derived economic-event inputs for profitability P0 V2."""

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

MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SCHEMA = "rl-quant.massive-economic-event-source-v2"
MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS = (
    "corporate-actions",
    "terminal-outcomes",
    "cash-returns",
)
MASSIVE_ECONOMIC_EVENT_SOURCE_V2_DATASETS = {
    "corporate-actions": "massive-economic-corporate-actions-v2",
    "terminal-outcomes": "massive-economic-terminal-outcomes-v2",
    "cash-returns": "massive-economic-cash-returns-v2",
}
MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "source_kinds": MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS,
        "roles": "exactly-one-committed-source-per-kind",
        "rows": "canonical-json-exact-field-inventory",
        "row_receipt": "derived-from-complete-row-and-upstream-source-receipt",
        "event_time": "economic-effective-time-distinct-from-strategy-availability",
        "event_order": "nonnegative-source-sequence-breaks-effective-time-ties",
        "identity": "permanent-security-authority-receipt",
    }
)
MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SCHEMA,
        "source_kinds": MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS,
    }
)


class MassiveEconomicEventSourceV2Error(ValueError):
    """Economic-event source bytes or their authority chain differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicEventSourceV2Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicEventSourceV2Error(f"{name} must be nonnegative")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEconomicEventSourceV2Error(f"{name} must be canonical text")
    return value


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveEconomicEventSourceV2Error(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MassiveEconomicEventSourceV2Error(f"{name} must be finite")
    return normalized


def economic_event_source_row_receipt_v2(
    *, source_kind: str, record: Mapping[str, object]
) -> str:
    """Derive a row receipt; *record* must omit its own receipt field."""

    if source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS:
        raise MassiveEconomicEventSourceV2Error("economic source kind is unsupported")
    if "source_row_receipt_sha256" in record:
        raise MassiveEconomicEventSourceV2Error(
            "economic source row receipt is recursively present"
        )
    return semantic_sha256({"source_kind": source_kind, "record": dict(record)})


@dataclass(frozen=True, slots=True)
class MassiveSourcedCorporateActionV2:
    event: CorporateActionRecord
    event_sequence: int
    upstream_source_receipt_sha256: str
    source_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.event.validate()
        _nonnegative_int("corporate-action sequence", self.event_sequence)
        for value in (
            self.upstream_source_receipt_sha256,
            self.source_row_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("corporate-action source digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV2Error(
                "sourced corporate-action receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveSourcedTerminalEventV2:
    event: TerminalEventRecord
    event_sequence: int
    upstream_source_receipt_sha256: str
    source_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.event.validate()
        _nonnegative_int("terminal-event sequence", self.event_sequence)
        for value in (
            self.upstream_source_receipt_sha256,
            self.source_row_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("terminal-event source digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV2Error(
                "sourced terminal-event receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveSourcedCashReturnV2:
    event_id: str
    cash_return: CashReturnRecord
    event_sequence: int
    source_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.event_id or self.event_id != self.event_id.strip():
            raise MassiveEconomicEventSourceV2Error("cash-return event ID is invalid")
        self.cash_return.validate()
        _nonnegative_int("cash-return sequence", self.event_sequence)
        _digest("cash-return row receipt", self.source_row_receipt_sha256)
        _digest("sourced cash-return receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV2Error(
                "sourced cash-return receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveEconomicEventSourceV2:
    source_kind: str
    identity_authority_receipt_sha256: str
    corporate_actions: tuple[MassiveSourcedCorporateActionV2, ...]
    terminal_events: tuple[MassiveSourcedTerminalEventV2, ...]
    cash_returns: tuple[MassiveSourcedCashReturnV2, ...]
    row_inventory_sha256: str
    parser_source_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SCHEMA
            or self.source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS
        ):
            raise MassiveEconomicEventSourceV2Error("economic source identity drifted")
        for value in (
            self.identity_authority_receipt_sha256,
            self.row_inventory_sha256,
            self.parser_source_sha256,
            self.receipt_sha256,
        ):
            _digest("economic source digest", value)
        if self.parser_source_sha256 != MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SOURCE_SHA256:
            raise MassiveEconomicEventSourceV2Error("economic source parser drifted")
        if (
            (self.source_kind != "corporate-actions" and self.corporate_actions)
            or (self.source_kind != "terminal-outcomes" and self.terminal_events)
            or (self.source_kind != "cash-returns" and self.cash_returns)
        ):
            raise MassiveEconomicEventSourceV2Error(
                "economic source contains the wrong record role"
            )
        records: tuple[
            MassiveSourcedCorporateActionV2
            | MassiveSourcedTerminalEventV2
            | MassiveSourcedCashReturnV2,
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
                (MassiveSourcedCorporateActionV2, MassiveSourcedTerminalEventV2),
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
            raise MassiveEconomicEventSourceV2Error(
                "economic source rows are not ordered and unique"
            )
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in records)
        ):
            raise MassiveEconomicEventSourceV2Error(
                "economic source row inventory differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V2_DATASETS[self.source_kind]
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveEconomicEventSourceV2Error(
                "economic committed source contract differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV2Error(
                "economic source artifact receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveEconomicEventAuthorityV2:
    identity_authority_receipt_sha256: str
    security_ids: tuple[str, ...]
    sources: tuple[MassiveEconomicEventSourceV2, ...]
    corporate_actions: tuple[MassiveSourcedCorporateActionV2, ...]
    terminal_events: tuple[MassiveSourcedTerminalEventV2, ...]
    cash_returns: tuple[MassiveSourcedCashReturnV2, ...]
    event_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _digest("economic authority identity", self.identity_authority_receipt_sha256)
        _digest("economic authority inventory", self.event_inventory_sha256)
        _digest("economic authority receipt", self.receipt_sha256)
        if (
            not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or tuple(source.source_kind for source in self.sources)
            != MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS
        ):
            raise MassiveEconomicEventSourceV2Error(
                "economic authority inventory is not canonical"
            )
        for source in self.sources:
            source.validate()
            if (
                source.identity_authority_receipt_sha256
                != self.identity_authority_receipt_sha256
            ):
                raise MassiveEconomicEventSourceV2Error(
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
            raise MassiveEconomicEventSourceV2Error(
                "economic authority rows differ from committed sources"
            )
        event_ids: set[str] = set()
        order_keys: set[tuple[str, int, int]] = set()
        receipts: list[str] = []
        security_events: list[
            MassiveSourcedCorporateActionV2 | MassiveSourcedTerminalEventV2
        ] = [*self.corporate_actions, *self.terminal_events]
        for row in security_events:
            event = row.event
            if event.event_id in event_ids:
                raise MassiveEconomicEventSourceV2Error(
                    "economic event IDs are not globally unique"
                )
            event_ids.add(event.event_id)
            key = (event.security_id, event.effective_at_ms, row.event_sequence)
            if key in order_keys:
                raise MassiveEconomicEventSourceV2Error(
                    "economic event ordering is ambiguous"
                )
            order_keys.add(key)
            if event.security_id not in self.security_ids or (
                event.successor_security_id is not None
                and event.successor_security_id not in self.security_ids
            ):
                raise MassiveEconomicEventSourceV2Error(
                    "economic event references an unknown security"
                )
            receipts.append(row.receipt_sha256)
        for cash_row in self.cash_returns:
            if cash_row.event_id in event_ids:
                raise MassiveEconomicEventSourceV2Error(
                    "cash event ID is not globally unique"
                )
            event_ids.add(cash_row.event_id)
            receipts.append(cash_row.receipt_sha256)
        if self.event_inventory_sha256 != semantic_sha256(tuple(receipts)):
            raise MassiveEconomicEventSourceV2Error(
                "economic authority event inventory differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicEventSourceV2Error(
                "economic event authority receipt differs"
            )


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveEconomicEventSourceV2Error(f"{name} field inventory differs")


def _parse_corporate(record: Mapping[str, object]) -> MassiveSourcedCorporateActionV2:
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
    if source_row_receipt != economic_event_source_row_receipt_v2(
        source_kind="corporate-actions", record=unsigned
    ):
        raise MassiveEconomicEventSourceV2Error(
            "corporate-action row was not derived from committed fields"
        )
    try:
        raw_successor = record["successor_security_id"]
        if raw_successor is not None and not isinstance(raw_successor, str):
            raise MassiveEconomicEventSourceV2Error(
                "corporate successor security ID is malformed"
            )
        event = CorporateActionRecord(
            event_id=_text("corporate event ID", record["event_id"]),
            security_id=_text("corporate security ID", record["security_id"]),
            kind=CorporateActionKind(_text("corporate kind", record["kind"])),
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
        sequence = _nonnegative_int("corporate sequence", record["event_sequence"])
        upstream = _digest(
            "corporate upstream source", record["upstream_source_receipt_sha256"]
        )
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV2Error(
            "corporate-action row is malformed"
        ) from exc
    provisional = MassiveSourcedCorporateActionV2(
        event=event,
        event_sequence=sequence,
        upstream_source_receipt_sha256=upstream,
        source_row_receipt_sha256=source_row_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _parse_terminal(record: Mapping[str, object]) -> MassiveSourcedTerminalEventV2:
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
    if source_row_receipt != economic_event_source_row_receipt_v2(
        source_kind="terminal-outcomes", record=unsigned
    ):
        raise MassiveEconomicEventSourceV2Error(
            "terminal-event row was not derived from committed fields"
        )
    try:
        raw_successor = record["successor_security_id"]
        if raw_successor is not None and not isinstance(raw_successor, str):
            raise MassiveEconomicEventSourceV2Error(
                "terminal successor security ID is malformed"
            )
        event = TerminalEventRecord(
            event_id=_text("terminal event ID", record["event_id"]),
            security_id=_text("terminal security ID", record["security_id"]),
            kind=TerminalEventKind(_text("terminal kind", record["kind"])),
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
        sequence = _nonnegative_int("terminal sequence", record["event_sequence"])
        upstream = _digest(
            "terminal upstream source", record["upstream_source_receipt_sha256"]
        )
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV2Error(
            "terminal-event row is malformed"
        ) from exc
    provisional = MassiveSourcedTerminalEventV2(
        event=event,
        event_sequence=sequence,
        upstream_source_receipt_sha256=upstream,
        source_row_receipt_sha256=source_row_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _parse_cash(record: Mapping[str, object]) -> MassiveSourcedCashReturnV2:
    expected = {
        "event_id",
        "effective_at_ms",
        "available_at_ms",
        "one_step_return",
        "event_sequence",
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
    if source_row_receipt != economic_event_source_row_receipt_v2(
        source_kind="cash-returns", record=unsigned
    ):
        raise MassiveEconomicEventSourceV2Error(
            "cash-return row was not derived from committed fields"
        )
    try:
        event_id = _text("cash-return event ID", record["event_id"])
        sequence = _nonnegative_int("cash-return sequence", record["event_sequence"])
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
    except (TypeError, ValueError) as exc:
        raise MassiveEconomicEventSourceV2Error("cash-return row is malformed") from exc
    provisional = MassiveSourcedCashReturnV2(
        event_id=event_id,
        cash_return=cash,
        event_sequence=sequence,
        source_row_receipt_sha256=source_row_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def parse_massive_economic_event_source_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveEconomicEventSourceV2:
    """Reopen and parse one exact committed economic-event source."""

    loaded_source.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicEventSourceV2Error(
            "economic event source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicEventSourceV2Error(
            "economic event source is not canonical JSON"
        )
    _exact_keys(
        payload,
        {"schema", "source_kind", "identity_authority_receipt_sha256", "records"},
        name="economic event source",
    )
    if payload["schema"] != MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SCHEMA:
        raise MassiveEconomicEventSourceV2Error("economic event source schema drifted")
    source_kind = payload["source_kind"]
    if (
        not isinstance(source_kind, str)
        or source_kind not in MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS
    ):
        raise MassiveEconomicEventSourceV2Error("economic event source kind is invalid")
    identity_receipt = _digest(
        "economic source identity", payload["identity_authority_receipt_sha256"]
    )
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise MassiveEconomicEventSourceV2Error(
            "economic event source records are malformed"
        )
    if any(not isinstance(record, dict) for record in raw_records):
        raise MassiveEconomicEventSourceV2Error("economic source record is malformed")
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
        MassiveSourcedCorporateActionV2
        | MassiveSourcedTerminalEventV2
        | MassiveSourcedCashReturnV2
    ] = [*corporate, *terminal, *cash]
    provisional = MassiveEconomicEventSourceV2(
        source_kind=source_kind,
        identity_authority_receipt_sha256=identity_receipt,
        corporate_actions=corporate,
        terminal_events=terminal,
        cash_returns=cash,
        row_inventory_sha256=semantic_sha256(
            tuple(record.receipt_sha256 for record in records)
        ),
        parser_source_sha256=MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SOURCE_SHA256,
        loaded_source=loaded_source,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def build_massive_economic_event_authority_v2(
    *,
    root: str | Path,
    loaded_sources: Sequence[LoadedMassiveSourceObject],
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveEconomicEventAuthorityV2:
    """Build the authority only by reparsing the three committed source roles."""

    identity_authority.validate()
    parsed = tuple(
        sorted(
            (
                parse_massive_economic_event_source_v2(
                    root=root, loaded_source=loaded_source
                )
                for loaded_source in loaded_sources
            ),
            key=lambda source: MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS.index(
                source.source_kind
            ),
        )
    )
    if (
        tuple(source.source_kind for source in parsed)
        != MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS
    ):
        raise MassiveEconomicEventSourceV2Error(
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
    provisional = MassiveEconomicEventAuthorityV2(
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        security_ids=tuple(
            sorted(row.security_id for row in identity_authority.security_master)
        ),
        sources=parsed,
        corporate_actions=corporate,
        terminal_events=terminal,
        cash_returns=cash,
        event_inventory_sha256=semantic_sha256(receipts),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V2_DATASETS",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V2_KINDS",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SCHEMA",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_EVENT_SOURCE_V2_SPEC_SHA256",
    "MassiveEconomicEventAuthorityV2",
    "MassiveEconomicEventSourceV2",
    "MassiveEconomicEventSourceV2Error",
    "MassiveSourcedCashReturnV2",
    "MassiveSourcedCorporateActionV2",
    "MassiveSourcedTerminalEventV2",
    "build_massive_economic_event_authority_v2",
    "economic_event_source_row_receipt_v2",
    "parse_massive_economic_event_source_v2",
]
