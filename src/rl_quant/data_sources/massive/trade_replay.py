"""Causal, correction-aware replay of normalized Massive stock trades."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import math
from typing import Mapping, Sequence

from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import (
    MassiveCorrectionAuthority,
    MassiveCorrectionKind,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_TRADE_EVENT_SCHEMA = "rl-quant.massive-trade-event-v1"
MASSIVE_TRADE_REPLAY_SCHEMA = "rl-quant.massive-trade-replay-v1"


class MassiveTradeReplayError(ValueError):
    """A normalized event or correction replay is ambiguous."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveTradeReplayError(f"{name} must be a canonical nonempty string")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveTradeReplayError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveTradeReplayError(f"{name} must be a nonnegative integer")
    return value


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise MassiveTradeReplayError(f"{name} must be an integer")
    try:
        observed = int(value)
    except ValueError as exc:
        raise MassiveTradeReplayError(f"{name} must be an integer") from exc
    return observed


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise MassiveTradeReplayError(f"{name} must be numeric")
    try:
        observed = float(value)
    except ValueError as exc:
        raise MassiveTradeReplayError(f"{name} must be numeric") from exc
    if not math.isfinite(observed) or observed <= 0:
        raise MassiveTradeReplayError(f"{name} must be finite and positive")
    return observed


def _object_sequence(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MassiveTradeReplayError(f"{name} must be a sequence")
    return value


@dataclass(frozen=True, slots=True)
class MassiveTradeEventV1:
    security_id: str
    source_ticker: str
    session_date: str
    trade_id: str
    exchange_id: int
    trf_id: int | None
    sequence_number: int
    participant_timestamp_ns: int
    sip_timestamp_ns: int
    trf_timestamp_ns: int | None
    strategy_available_timestamp_ns: int
    price: float
    decimal_size: str
    conditions: tuple[int, ...]
    correction_code: int
    correction_kind: MassiveCorrectionKind
    price_forming: bool
    volume_forming: bool
    regular_session: bool
    source_file_sha256: str
    source_row_number: int
    source_record_sha256: str
    schema: str = MASSIVE_TRADE_EVENT_SCHEMA

    @property
    def event_key(self) -> str:
        trf = "NONE" if self.trf_id is None else str(self.trf_id)
        return f"{self.security_id}|{self.exchange_id}|{trf}|{self.trade_id}"

    @property
    def replay_order(self) -> tuple[int, int, int, int, int, str]:
        return (
            self.strategy_available_timestamp_ns,
            self.sip_timestamp_ns,
            self.sequence_number,
            self.exchange_id,
            -1 if self.trf_id is None else self.trf_id,
            self.trade_id,
        )

    def economic_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("source_row_number")
        payload.pop("source_file_sha256")
        return payload

    def validate(self) -> None:
        if self.schema != MASSIVE_TRADE_EVENT_SCHEMA:
            raise MassiveTradeReplayError("normalized trade schema drifted")
        for name in ("security_id", "source_ticker", "session_date", "trade_id"):
            _text(name, getattr(self, name))
        for name in (
            "exchange_id",
            "sequence_number",
            "participant_timestamp_ns",
            "sip_timestamp_ns",
            "strategy_available_timestamp_ns",
            "correction_code",
            "source_row_number",
        ):
            _nonnegative_int(name, getattr(self, name))
        if self.trf_id is not None:
            _nonnegative_int("TRF ID", self.trf_id)
        if self.trf_timestamp_ns is not None:
            _nonnegative_int("TRF timestamp", self.trf_timestamp_ns)
        if self.participant_timestamp_ns > self.sip_timestamp_ns:
            raise MassiveTradeReplayError("participant timestamp exceeds SIP timestamp")
        if self.strategy_available_timestamp_ns < self.sip_timestamp_ns:
            raise MassiveTradeReplayError("strategy availability precedes SIP dissemination")
        if not isinstance(self.price, (int, float)) or not math.isfinite(float(self.price)) or self.price <= 0:
            raise MassiveTradeReplayError("trade price must be finite and positive")
        try:
            size = Decimal(self.decimal_size)
        except InvalidOperation as exc:
            raise MassiveTradeReplayError("decimal trade size is invalid") from exc
        if not size.is_finite() or size <= 0:
            raise MassiveTradeReplayError("trade size must be finite and positive")
        if self.conditions != tuple(sorted(set(self.conditions))):
            raise MassiveTradeReplayError("condition IDs must be sorted and unique")
        if self.correction_kind not in {
            "new-trade",
            "replacement",
            "cancellation",
            "late-report",
        }:
            raise MassiveTradeReplayError("correction kind is unsupported")
        if any(
            not isinstance(value, bool)
            for value in (self.price_forming, self.volume_forming, self.regular_session)
        ):
            raise MassiveTradeReplayError("trade eligibility fields must be Boolean")
        _digest("source file SHA", self.source_file_sha256)
        _digest("source record SHA", self.source_record_sha256)


@dataclass(frozen=True, slots=True)
class MassiveTradeReplayResult:
    security_id: str
    session_date: str
    decision_at_ns: int
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    input_event_count: int
    visible_event_count: int
    active_events: tuple[MassiveTradeEventV1, ...]
    cancelled_event_keys: tuple[str, ...]
    post_cutoff_event_count: int
    receipt_sha256: str
    schema: str = MASSIVE_TRADE_REPLAY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "security_id": self.security_id,
            "session_date": self.session_date,
            "decision_at_ns": self.decision_at_ns,
            "condition_authority_receipt_sha256": self.condition_authority_receipt_sha256,
            "correction_authority_receipt_sha256": self.correction_authority_receipt_sha256,
            "input_event_count": self.input_event_count,
            "visible_event_count": self.visible_event_count,
            "active_events": [event.economic_payload() for event in self.active_events],
            "cancelled_event_keys": list(self.cancelled_event_keys),
            "post_cutoff_event_count": self.post_cutoff_event_count,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TRADE_REPLAY_SCHEMA:
            raise MassiveTradeReplayError("trade replay schema drifted")
        _text("security ID", self.security_id)
        _text("session date", self.session_date)
        _nonnegative_int("decision timestamp", self.decision_at_ns)
        for name in (
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in (
            "input_event_count",
            "visible_event_count",
            "post_cutoff_event_count",
        ):
            _nonnegative_int(name, getattr(self, name))
        if self.visible_event_count + self.post_cutoff_event_count != self.input_event_count:
            raise MassiveTradeReplayError("replay event inventory does not reconcile")
        keys = tuple(event.event_key for event in self.active_events)
        if keys != tuple(sorted(set(keys))):
            raise MassiveTradeReplayError("active event inventory is not canonical")
        for event in self.active_events:
            event.validate()
            if event.security_id != self.security_id or event.session_date != self.session_date:
                raise MassiveTradeReplayError("replay mixed security-day identities")
            if event.strategy_available_timestamp_ns > self.decision_at_ns:
                raise MassiveTradeReplayError("post-cutoff trade entered replay")
        if self.cancelled_event_keys != tuple(sorted(set(self.cancelled_event_keys))):
            raise MassiveTradeReplayError("cancelled event keys are not canonical")
        if set(keys) & set(self.cancelled_event_keys):
            raise MassiveTradeReplayError("cancelled event remains active")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTradeReplayError("trade replay receipt differs")


def normalize_massive_trade_event(
    record: Mapping[str, object],
    *,
    security_id: str,
    source_ticker: str,
    session_date: str,
    entitlement_delay_ns: int,
    regular_session: bool,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    source_file_sha256: str,
    source_row_number: int,
) -> MassiveTradeEventV1:
    """Normalize one vendor record using separately qualified semantics."""

    raw_conditions = record.get("conditions", ())
    condition_ids = tuple(
        sorted(
            _integer("condition ID", value)
            for value in _object_sequence("conditions", raw_conditions)
        )
    )
    price_forming, volume_forming = condition_authority.resolve(condition_ids)
    correction_code = _integer("correction code", record.get("correction", 0))
    correction_kind = correction_authority.resolve(correction_code)
    sip_timestamp_ns = _integer("SIP timestamp", record["sip_timestamp"])
    raw_size = record.get("decimal_size")
    if raw_size is None:
        raw_size = record["size"]
    payload_for_source_hash = {
        key: record[key] for key in sorted(record)
    }
    event = MassiveTradeEventV1(
        security_id=security_id,
        source_ticker=source_ticker,
        session_date=session_date,
        trade_id=str(record["id"]),
        exchange_id=_integer("exchange ID", record["exchange"]),
        trf_id=None
        if record.get("trf_id") is None
        else _integer("TRF ID", record["trf_id"]),
        sequence_number=_integer("sequence number", record["sequence_number"]),
        participant_timestamp_ns=_integer(
            "participant timestamp", record["participant_timestamp"]
        ),
        sip_timestamp_ns=sip_timestamp_ns,
        trf_timestamp_ns=None
        if record.get("trf_timestamp") is None
        else _integer("TRF timestamp", record["trf_timestamp"]),
        strategy_available_timestamp_ns=sip_timestamp_ns + entitlement_delay_ns,
        price=_positive_float("price", record["price"]),
        decimal_size=str(raw_size),
        conditions=condition_ids,
        correction_code=correction_code,
        correction_kind=correction_kind,
        price_forming=price_forming,
        volume_forming=volume_forming,
        regular_session=regular_session,
        source_file_sha256=source_file_sha256,
        source_row_number=source_row_number,
        source_record_sha256=semantic_sha256(payload_for_source_hash),
    )
    event.validate()
    return event


def replay_massive_trades(
    events: Sequence[MassiveTradeEventV1],
    *,
    decision_at_ns: int,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
) -> MassiveTradeReplayResult:
    """Replay all strategy-visible events in deterministic timestamp order."""

    _nonnegative_int("decision timestamp", decision_at_ns)
    condition_authority.validate()
    correction_authority.validate()
    if not events:
        raise MassiveTradeReplayError("trade replay requires one symbol-day event")
    for event in events:
        event.validate()
    identities = {(event.security_id, event.session_date) for event in events}
    if len(identities) != 1:
        raise MassiveTradeReplayError("trade replay mixed symbol-day identities")
    security_id, session_date = next(iter(identities))
    visible = tuple(
        sorted(
            (
                event
                for event in events
                if event.strategy_available_timestamp_ns <= decision_at_ns
            ),
            key=lambda event: event.replay_order,
        )
    )
    for left, right in zip(visible, visible[1:], strict=False):
        if left.replay_order == right.replay_order and left.economic_payload() != right.economic_payload():
            raise MassiveTradeReplayError("ambiguous events share one replay order")

    active: dict[str, MassiveTradeEventV1] = {}
    cancelled: set[str] = set()
    for event in visible:
        key = event.event_key
        kind = event.correction_kind
        if kind in {"new-trade", "late-report"}:
            existing = active.get(key)
            if existing is not None and existing.economic_payload() != event.economic_payload():
                raise MassiveTradeReplayError("conflicting duplicate new trade")
            active[key] = event
            cancelled.discard(key)
        elif kind == "replacement":
            if key not in active:
                raise MassiveTradeReplayError("replacement lacks a visible predecessor")
            active[key] = event
            cancelled.discard(key)
        elif kind == "cancellation":
            if key not in active:
                raise MassiveTradeReplayError("cancellation lacks a visible predecessor")
            del active[key]
            cancelled.add(key)
        else:  # pragma: no cover - authority validation makes this unreachable
            raise MassiveTradeReplayError("unqualified correction semantic")
    ordered_active = tuple(active[key] for key in sorted(active))
    body = {
        "schema": MASSIVE_TRADE_REPLAY_SCHEMA,
        "security_id": security_id,
        "session_date": session_date,
        "decision_at_ns": decision_at_ns,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "input_event_count": len(events),
        "visible_event_count": len(visible),
        "active_events": [event.economic_payload() for event in ordered_active],
        "cancelled_event_keys": sorted(cancelled),
        "post_cutoff_event_count": len(events) - len(visible),
    }
    result = MassiveTradeReplayResult(
        security_id=security_id,
        session_date=session_date,
        decision_at_ns=decision_at_ns,
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        correction_authority_receipt_sha256=correction_authority.receipt_sha256,
        input_event_count=len(events),
        visible_event_count=len(visible),
        active_events=ordered_active,
        cancelled_event_keys=tuple(sorted(cancelled)),
        post_cutoff_event_count=len(events) - len(visible),
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_TRADE_EVENT_SCHEMA",
    "MASSIVE_TRADE_REPLAY_SCHEMA",
    "MassiveTradeEventV1",
    "MassiveTradeReplayError",
    "MassiveTradeReplayResult",
    "normalize_massive_trade_event",
    "replay_massive_trades",
]
