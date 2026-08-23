"""Source-specific Massive trade adapters into one canonical source record."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
from typing import Literal, Mapping, Sequence

from rl_quant.data_sources.massive.websocket_capture import (
    MassiveDelayedWebSocketEvent,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_CANONICAL_TRADE_SOURCE_SCHEMA = (
    "rl-quant.massive-canonical-trade-source-record-v1"
)
MASSIVE_TRADE_CANONICALIZATION_SPEC_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_CANONICAL_TRADE_SOURCE_SCHEMA,
        "websocket_timestamp_unit": "milliseconds",
        "rest_timestamp_unit": "nanoseconds",
        "flat_file_timestamp_unit": "nanoseconds",
        "price_representation": "canonical-decimal-string",
        "size_representation": "canonical-decimal-string",
        "condition_order": "sorted-unique",
    }
)

MassiveTradeSourceKind = Literal[
    "delayed-websocket",
    "rest-trades",
    "flat-file-trades",
]


class MassiveTradeCanonicalizationError(ValueError):
    """A vendor trade row cannot be mapped without ambiguity."""


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise MassiveTradeCanonicalizationError(f"{name} must be an integer")
    try:
        observed = int(value)
    except ValueError as exc:
        raise MassiveTradeCanonicalizationError(f"{name} must be an integer") from exc
    if observed < 0:
        raise MassiveTradeCanonicalizationError(f"{name} must be nonnegative")
    return observed


def _decimal(name: str, value: object) -> str:
    if isinstance(value, bool):
        raise MassiveTradeCanonicalizationError(f"{name} must be decimal")
    try:
        observed = Decimal(str(value))
    except InvalidOperation as exc:
        raise MassiveTradeCanonicalizationError(f"{name} must be decimal") from exc
    if not observed.is_finite() or observed <= 0:
        raise MassiveTradeCanonicalizationError(f"{name} must be finite and positive")
    return format(observed.normalize(), "f")


def _conditions(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MassiveTradeCanonicalizationError("conditions must be a sequence")
    observed = tuple(sorted(_integer("condition ID", item) for item in value))
    if observed != tuple(sorted(set(observed))):
        raise MassiveTradeCanonicalizationError("conditions must be unique")
    return observed


@dataclass(frozen=True, slots=True)
class MassiveCanonicalTradeSourceRecord:
    ticker: str
    exchange_id: int
    trade_id: str
    sequence_number: int
    participant_timestamp_ns: int
    sip_timestamp_ns: int
    trf_id: int | None
    trf_timestamp_ns: int | None
    price_decimal: str
    size_decimal: str
    conditions: tuple[int, ...]
    correction_code: int | None
    source_kind: MassiveTradeSourceKind
    raw_source_record_sha256: str
    canonicalization_spec_sha256: str
    received_at_ns: int | None
    receipt_sha256: str
    schema: str = MASSIVE_CANONICAL_TRADE_SOURCE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_CANONICAL_TRADE_SOURCE_SCHEMA:
            raise MassiveTradeCanonicalizationError("canonical trade schema drifted")
        if not self.ticker or not self.trade_id:
            raise MassiveTradeCanonicalizationError("canonical trade identity is absent")
        for name in (
            "exchange_id",
            "sequence_number",
            "participant_timestamp_ns",
            "sip_timestamp_ns",
        ):
            _integer(name, getattr(self, name))
        if self.participant_timestamp_ns > self.sip_timestamp_ns:
            raise MassiveTradeCanonicalizationError(
                "participant timestamp exceeds SIP timestamp"
            )
        if self.trf_id is not None:
            _integer("TRF ID", self.trf_id)
        if self.trf_timestamp_ns is not None:
            _integer("TRF timestamp", self.trf_timestamp_ns)
        _decimal("price", self.price_decimal)
        _decimal("size", self.size_decimal)
        if self.conditions != tuple(sorted(set(self.conditions))):
            raise MassiveTradeCanonicalizationError("condition inventory is not canonical")
        if self.correction_code is not None:
            _integer("correction code", self.correction_code)
        if self.source_kind not in {
            "delayed-websocket",
            "rest-trades",
            "flat-file-trades",
        }:
            raise MassiveTradeCanonicalizationError("trade source kind is unsupported")
        if self.source_kind == "delayed-websocket":
            if self.received_at_ns is None:
                raise MassiveTradeCanonicalizationError(
                    "delayed trade lacks actual receive time"
                )
            _integer("received timestamp", self.received_at_ns)
            if self.received_at_ns < self.sip_timestamp_ns:
                raise MassiveTradeCanonicalizationError(
                    "delayed receive time predates SIP dissemination"
                )
        elif self.received_at_ns is not None:
            raise MassiveTradeCanonicalizationError(
                "non-WebSocket source cannot claim a local receive time"
            )
        for name in (
            "raw_source_record_sha256",
            "canonicalization_spec_sha256",
            "receipt_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise MassiveTradeCanonicalizationError(f"{name} is not a SHA-256")
        if self.canonicalization_spec_sha256 != (
            MASSIVE_TRADE_CANONICALIZATION_SPEC_SHA256
        ):
            raise MassiveTradeCanonicalizationError("canonicalization spec drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTradeCanonicalizationError("canonical trade receipt differs")


def _build_canonical_record(
    *,
    ticker: object,
    exchange_id: object,
    trade_id: object,
    sequence_number: object,
    participant_timestamp_ns: int,
    sip_timestamp_ns: int,
    trf_id: object,
    trf_timestamp_ns: int | None,
    price: object,
    size: object,
    conditions: object,
    correction_code: object,
    source_kind: MassiveTradeSourceKind,
    raw_source_record_sha256: str,
    received_at_ns: int | None,
) -> MassiveCanonicalTradeSourceRecord:
    canonical_exchange_id = _integer("exchange ID", exchange_id)
    canonical_sequence_number = _integer("sequence number", sequence_number)
    canonical_trf_id = None if trf_id is None else _integer("TRF ID", trf_id)
    canonical_price = _decimal("price", price)
    canonical_size = _decimal("size", size)
    canonical_conditions = _conditions(conditions)
    canonical_correction = (
        None
        if correction_code is None
        else _integer("correction code", correction_code)
    )
    body = {
        "schema": MASSIVE_CANONICAL_TRADE_SOURCE_SCHEMA,
        "ticker": str(ticker),
        "exchange_id": canonical_exchange_id,
        "trade_id": str(trade_id),
        "sequence_number": canonical_sequence_number,
        "participant_timestamp_ns": participant_timestamp_ns,
        "sip_timestamp_ns": sip_timestamp_ns,
        "trf_id": canonical_trf_id,
        "trf_timestamp_ns": trf_timestamp_ns,
        "price_decimal": canonical_price,
        "size_decimal": canonical_size,
        "conditions": canonical_conditions,
        "correction_code": canonical_correction,
        "source_kind": source_kind,
        "raw_source_record_sha256": raw_source_record_sha256,
        "canonicalization_spec_sha256": MASSIVE_TRADE_CANONICALIZATION_SPEC_SHA256,
        "received_at_ns": received_at_ns,
    }
    value = MassiveCanonicalTradeSourceRecord(
        ticker=str(ticker),
        exchange_id=canonical_exchange_id,
        trade_id=str(trade_id),
        sequence_number=canonical_sequence_number,
        participant_timestamp_ns=participant_timestamp_ns,
        sip_timestamp_ns=sip_timestamp_ns,
        trf_id=canonical_trf_id,
        trf_timestamp_ns=trf_timestamp_ns,
        price_decimal=canonical_price,
        size_decimal=canonical_size,
        conditions=canonical_conditions,
        correction_code=canonical_correction,
        source_kind=source_kind,
        raw_source_record_sha256=raw_source_record_sha256,
        canonicalization_spec_sha256=MASSIVE_TRADE_CANONICALIZATION_SPEC_SHA256,
        received_at_ns=received_at_ns,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


def canonicalize_massive_websocket_trade(
    event: MassiveDelayedWebSocketEvent,
) -> MassiveCanonicalTradeSourceRecord:
    """Canonicalize the actual compact delayed WebSocket stock-trade schema."""

    event.validate()
    payload = json.loads(event.canonical_payload_json)
    return _build_canonical_record(
        ticker=payload["sym"],
        exchange_id=payload["x"],
        trade_id=payload["i"],
        sequence_number=payload["q"],
        participant_timestamp_ns=_integer("participant timestamp", payload["pt"])
        * 1_000_000,
        sip_timestamp_ns=_integer("SIP timestamp", payload["t"]) * 1_000_000,
        trf_id=payload.get("trfi"),
        trf_timestamp_ns=None
        if payload.get("trft") is None
        else _integer("TRF timestamp", payload["trft"]) * 1_000_000,
        price=payload["p"],
        size=payload.get("ds", payload.get("s")),
        conditions=payload.get("c", ()),
        correction_code=None,
        source_kind="delayed-websocket",
        raw_source_record_sha256=event.payload_sha256,
        received_at_ns=event.received_at_ns,
    )


def _canonicalize_long_trade(
    record: Mapping[str, object], *, source_kind: MassiveTradeSourceKind
) -> MassiveCanonicalTradeSourceRecord:
    raw_size = record.get("decimal_size", record.get("size"))
    return _build_canonical_record(
        ticker=record["ticker"],
        exchange_id=record["exchange"],
        trade_id=record["id"],
        sequence_number=record["sequence_number"],
        participant_timestamp_ns=_integer(
            "participant timestamp", record["participant_timestamp"]
        ),
        sip_timestamp_ns=_integer("SIP timestamp", record["sip_timestamp"]),
        trf_id=record.get("trf_id"),
        trf_timestamp_ns=None
        if record.get("trf_timestamp") is None
        else _integer("TRF timestamp", record["trf_timestamp"]),
        price=record["price"],
        size=raw_size,
        conditions=record.get("conditions", ()),
        correction_code=record.get("correction", 0),
        source_kind=source_kind,
        raw_source_record_sha256=semantic_sha256(
            {key: record[key] for key in sorted(record)}
        ),
        received_at_ns=None,
    )


def canonicalize_massive_rest_trade(
    record: Mapping[str, object],
) -> MassiveCanonicalTradeSourceRecord:
    return _canonicalize_long_trade(record, source_kind="rest-trades")


def canonicalize_massive_flat_file_trade(
    record: Mapping[str, object],
) -> MassiveCanonicalTradeSourceRecord:
    return _canonicalize_long_trade(record, source_kind="flat-file-trades")


__all__ = [
    "MASSIVE_CANONICAL_TRADE_SOURCE_SCHEMA",
    "MASSIVE_TRADE_CANONICALIZATION_SPEC_SHA256",
    "MassiveCanonicalTradeSourceRecord",
    "MassiveTradeCanonicalizationError",
    "MassiveTradeSourceKind",
    "canonicalize_massive_flat_file_trade",
    "canonicalize_massive_rest_trade",
    "canonicalize_massive_websocket_trade",
]
