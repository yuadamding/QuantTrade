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
from rl_quant.data_sources.massive.entitlement import MassiveEntitlementAuthority
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import MassiveSourceObjectReceipt
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_RESOLVED_SECURITY_IDENTITY_SCHEMA = (
    "rl-quant.massive-resolved-security-identity-v1"
)
MASSIVE_TRADE_EVENT_SCHEMA = "rl-quant.massive-trade-event-v2"
MASSIVE_TRADE_REPLAY_SCHEMA = "rl-quant.massive-trade-replay-v2"


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
class MassiveResolvedSecurityIdentity:
    """One PIT ticker-to-security resolution backed by identity authorities."""

    security_id: str
    source_ticker: str
    primary_exchange: str
    session_date: str
    valid_from_ns: int
    valid_to_ns: int | None
    identity_authority_receipt_sha256: str
    ticker_history_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_RESOLVED_SECURITY_IDENTITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_RESOLVED_SECURITY_IDENTITY_SCHEMA:
            raise MassiveTradeReplayError("resolved identity schema drifted")
        for name in (
            "security_id",
            "source_ticker",
            "primary_exchange",
            "session_date",
        ):
            _text(name, getattr(self, name))
        _nonnegative_int("identity valid-from", self.valid_from_ns)
        if self.valid_to_ns is not None:
            _nonnegative_int("identity valid-to", self.valid_to_ns)
            if self.valid_to_ns <= self.valid_from_ns:
                raise MassiveTradeReplayError("identity validity interval is empty")
        for name in (
            "identity_authority_receipt_sha256",
            "ticker_history_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTradeReplayError("resolved identity receipt differs")

    def contains(self, timestamp_ns: int) -> bool:
        self.validate()
        return self.valid_from_ns <= timestamp_ns and (
            self.valid_to_ns is None or timestamp_ns < self.valid_to_ns
        )

    @classmethod
    def build(
        cls,
        *,
        security_id: str,
        source_ticker: str,
        primary_exchange: str,
        session_date: str,
        valid_from_ns: int,
        valid_to_ns: int | None,
        identity_authority_receipt_sha256: str,
        ticker_history_receipt_sha256: str,
    ) -> MassiveResolvedSecurityIdentity:
        body = {
            "schema": MASSIVE_RESOLVED_SECURITY_IDENTITY_SCHEMA,
            "security_id": security_id,
            "source_ticker": source_ticker,
            "primary_exchange": primary_exchange,
            "session_date": session_date,
            "valid_from_ns": valid_from_ns,
            "valid_to_ns": valid_to_ns,
            "identity_authority_receipt_sha256": identity_authority_receipt_sha256,
            "ticker_history_receipt_sha256": ticker_history_receipt_sha256,
        }
        value = cls(
            security_id=security_id,
            source_ticker=source_ticker,
            primary_exchange=primary_exchange,
            session_date=session_date,
            valid_from_ns=valid_from_ns,
            valid_to_ns=valid_to_ns,
            identity_authority_receipt_sha256=identity_authority_receipt_sha256,
            ticker_history_receipt_sha256=ticker_history_receipt_sha256,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveTradeEventV2:
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
    updates_open_close: bool
    updates_high_low: bool
    updates_volume: bool
    regular_session: bool
    source_file_sha256: str
    source_row_number: int
    source_record_sha256: str
    entitlement_authority_receipt_sha256: str
    session_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    source_object_receipt_sha256: str
    identity_authority_receipt_sha256: str
    ticker_history_receipt_sha256: str
    identity_resolution_receipt_sha256: str
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
            for value in (
                self.updates_open_close,
                self.updates_high_low,
                self.updates_volume,
                self.regular_session,
            )
        ):
            raise MassiveTradeReplayError("trade eligibility fields must be Boolean")
        for name in (
            "source_file_sha256",
            "source_record_sha256",
            "entitlement_authority_receipt_sha256",
            "session_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "source_object_receipt_sha256",
            "identity_authority_receipt_sha256",
            "ticker_history_receipt_sha256",
            "identity_resolution_receipt_sha256",
        ):
            _digest(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class MassiveTradeReplayResult:
    security_id: str
    session_date: str
    decision_at_ns: int
    entitlement_authority_receipt_sha256: str
    session_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    source_object_receipt_sha256: str
    identity_authority_receipt_sha256: str
    ticker_history_receipt_sha256: str
    identity_resolution_receipt_sha256: str
    input_source_record_inventory_sha256: str
    input_event_count: int
    visible_event_count: int
    active_events: tuple[MassiveTradeEventV2, ...]
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
            "entitlement_authority_receipt_sha256": self.entitlement_authority_receipt_sha256,
            "session_authority_receipt_sha256": self.session_authority_receipt_sha256,
            "condition_authority_receipt_sha256": self.condition_authority_receipt_sha256,
            "correction_authority_receipt_sha256": self.correction_authority_receipt_sha256,
            "source_object_receipt_sha256": self.source_object_receipt_sha256,
            "identity_authority_receipt_sha256": self.identity_authority_receipt_sha256,
            "ticker_history_receipt_sha256": self.ticker_history_receipt_sha256,
            "identity_resolution_receipt_sha256": self.identity_resolution_receipt_sha256,
            "input_source_record_inventory_sha256": self.input_source_record_inventory_sha256,
            "input_event_count": self.input_event_count,
            "visible_event_count": self.visible_event_count,
            "active_events": [event.economic_payload() for event in self.active_events],
            "cancelled_event_keys": list(self.cancelled_event_keys),
            "post_cutoff_event_count": self.post_cutoff_event_count,
        }

    @property
    def active_state_inventory_sha256(self) -> str:
        """Hash economic replay state without source-transport provenance."""

        rows = []
        for event in self.active_events:
            rows.append(
                {
                    "event_key": event.event_key,
                    "security_id": event.security_id,
                    "source_ticker": event.source_ticker,
                    "session_date": event.session_date,
                    "trade_id": event.trade_id,
                    "exchange_id": event.exchange_id,
                    "trf_id": event.trf_id,
                    "sequence_number": event.sequence_number,
                    "participant_timestamp_ns": event.participant_timestamp_ns,
                    "sip_timestamp_ns": event.sip_timestamp_ns,
                    "trf_timestamp_ns": event.trf_timestamp_ns,
                    "price": event.price,
                    "decimal_size": event.decimal_size,
                    "conditions": event.conditions,
                    "correction_code": event.correction_code,
                    "correction_kind": event.correction_kind,
                    "updates_open_close": event.updates_open_close,
                    "updates_high_low": event.updates_high_low,
                    "updates_volume": event.updates_volume,
                    "regular_session": event.regular_session,
                }
            )
        return semantic_sha256(
            {
                "active_events": rows,
                "cancelled_event_keys": self.cancelled_event_keys,
            }
        )

    def validate(self) -> None:
        if self.schema != MASSIVE_TRADE_REPLAY_SCHEMA:
            raise MassiveTradeReplayError("trade replay schema drifted")
        _text("security ID", self.security_id)
        _text("session date", self.session_date)
        _nonnegative_int("decision timestamp", self.decision_at_ns)
        for name in (
            "entitlement_authority_receipt_sha256",
            "session_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "source_object_receipt_sha256",
            "identity_authority_receipt_sha256",
            "ticker_history_receipt_sha256",
            "identity_resolution_receipt_sha256",
            "input_source_record_inventory_sha256",
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
            for field in (
                "entitlement_authority_receipt_sha256",
                "session_authority_receipt_sha256",
                "condition_authority_receipt_sha256",
                "correction_authority_receipt_sha256",
                "source_object_receipt_sha256",
                "identity_authority_receipt_sha256",
                "ticker_history_receipt_sha256",
                "identity_resolution_receipt_sha256",
            ):
                if getattr(event, field) != getattr(self, field):
                    raise MassiveTradeReplayError(
                        f"active trade {field} differs from replay authority"
                    )
        if self.cancelled_event_keys != tuple(sorted(set(self.cancelled_event_keys))):
            raise MassiveTradeReplayError("cancelled event keys are not canonical")
        if set(keys) & set(self.cancelled_event_keys):
            raise MassiveTradeReplayError("cancelled event remains active")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTradeReplayError("trade replay receipt differs")


def normalize_massive_trade_event(
    record: Mapping[str, object],
    *,
    entitlement_authority: MassiveEntitlementAuthority,
    session_authority: MassiveSessionAuthority,
    session: MassiveExchangeSession,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    source_object_receipt: MassiveSourceObjectReceipt,
    identity_resolution: MassiveResolvedSecurityIdentity,
    source_row_number: int,
) -> MassiveTradeEventV2:
    """Normalize one vendor record using separately qualified semantics."""

    entitlement_authority.validate()
    session_authority.validate()
    session.validate()
    condition_authority.validate()
    correction_authority.validate()
    source_object_receipt.validate()
    identity_resolution.validate()
    if source_object_receipt.entitlement_receipt_sha256 != entitlement_authority.receipt_sha256:
        raise MassiveTradeReplayError("source object used another entitlement authority")
    if session_authority.resolve(
        exchange=identity_resolution.primary_exchange,
        session_date=identity_resolution.session_date,
    ) != session:
        raise MassiveTradeReplayError("session was not resolved by the supplied authority")
    if identity_resolution.session_date != session.session_date:
        raise MassiveTradeReplayError("identity and session dates differ")
    record_ticker = record.get("ticker", record.get("sym"))
    if record_ticker is not None and str(record_ticker) != identity_resolution.source_ticker:
        raise MassiveTradeReplayError("trade ticker differs from the PIT identity")

    raw_conditions = record.get("conditions", ())
    condition_ids = tuple(
        sorted(
            _integer("condition ID", value)
            for value in _object_sequence("conditions", raw_conditions)
        )
    )
    updates_open_close, updates_high_low, updates_volume = condition_authority.resolve(
        condition_ids
    )
    correction_code = _integer("correction code", record.get("correction", 0))
    correction_kind = correction_authority.resolve(correction_code)
    sip_timestamp_ns = _integer("SIP timestamp", record["sip_timestamp"])
    participant_timestamp_ns = _integer(
        "participant timestamp", record["participant_timestamp"]
    )
    if not identity_resolution.contains(participant_timestamp_ns):
        raise MassiveTradeReplayError("trade lies outside the PIT ticker identity interval")
    entitlement_delay_ns = (
        entitlement_authority.entitlement_delay_minutes * 60 * 1_000_000_000
    )
    raw_size = record.get("decimal_size")
    if raw_size is None:
        raw_size = record["size"]
    payload_for_source_hash = {
        key: record[key] for key in sorted(record)
    }
    event = MassiveTradeEventV2(
        security_id=identity_resolution.security_id,
        source_ticker=identity_resolution.source_ticker,
        session_date=session.session_date,
        trade_id=str(record["id"]),
        exchange_id=_integer("exchange ID", record["exchange"]),
        trf_id=None
        if record.get("trf_id") is None
        else _integer("TRF ID", record["trf_id"]),
        sequence_number=_integer("sequence number", record["sequence_number"]),
        participant_timestamp_ns=participant_timestamp_ns,
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
        updates_open_close=updates_open_close,
        updates_high_low=updates_high_low,
        updates_volume=updates_volume,
        regular_session=session.is_regular(participant_timestamp_ns),
        source_file_sha256=source_object_receipt.physical_sha256,
        source_row_number=source_row_number,
        source_record_sha256=semantic_sha256(payload_for_source_hash),
        entitlement_authority_receipt_sha256=entitlement_authority.receipt_sha256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        correction_authority_receipt_sha256=correction_authority.receipt_sha256,
        source_object_receipt_sha256=source_object_receipt.receipt_sha256,
        identity_authority_receipt_sha256=identity_resolution.identity_authority_receipt_sha256,
        ticker_history_receipt_sha256=identity_resolution.ticker_history_receipt_sha256,
        identity_resolution_receipt_sha256=identity_resolution.receipt_sha256,
    )
    event.validate()
    return event


def replay_massive_trades(
    events: Sequence[MassiveTradeEventV2],
    *,
    decision_at_ns: int,
    entitlement_authority: MassiveEntitlementAuthority,
    session_authority: MassiveSessionAuthority,
    session: MassiveExchangeSession,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    source_object_receipt: MassiveSourceObjectReceipt,
    identity_resolution: MassiveResolvedSecurityIdentity,
) -> MassiveTradeReplayResult:
    """Replay all strategy-visible events in deterministic timestamp order."""

    _nonnegative_int("decision timestamp", decision_at_ns)
    entitlement_authority.validate()
    session_authority.validate()
    session.validate()
    condition_authority.validate()
    correction_authority.validate()
    source_object_receipt.validate()
    identity_resolution.validate()
    if source_object_receipt.entitlement_receipt_sha256 != entitlement_authority.receipt_sha256:
        raise MassiveTradeReplayError("source object used another entitlement authority")
    if session_authority.resolve(
        exchange=identity_resolution.primary_exchange,
        session_date=identity_resolution.session_date,
    ) != session:
        raise MassiveTradeReplayError("session was not resolved by the supplied authority")
    if identity_resolution.session_date != session.session_date:
        raise MassiveTradeReplayError("identity and session dates differ")
    if not events:
        raise MassiveTradeReplayError("trade replay requires one symbol-day event")
    for event in events:
        event.validate()
        expected_conditions = condition_authority.resolve(event.conditions)
        if expected_conditions != (
            event.updates_open_close,
            event.updates_high_low,
            event.updates_volume,
        ):
            raise MassiveTradeReplayError("trade condition eligibility differs from authority")
        if correction_authority.resolve(event.correction_code) != event.correction_kind:
            raise MassiveTradeReplayError("trade correction semantic differs from authority")
        expected_receipts = {
            "entitlement_authority_receipt_sha256": entitlement_authority.receipt_sha256,
            "session_authority_receipt_sha256": session_authority.receipt_sha256,
            "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
            "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
            "source_object_receipt_sha256": source_object_receipt.receipt_sha256,
            "identity_authority_receipt_sha256": identity_resolution.identity_authority_receipt_sha256,
            "ticker_history_receipt_sha256": identity_resolution.ticker_history_receipt_sha256,
            "identity_resolution_receipt_sha256": identity_resolution.receipt_sha256,
        }
        for field, expected in expected_receipts.items():
            if getattr(event, field) != expected:
                raise MassiveTradeReplayError(f"trade {field} differs from authority")
        if event.source_file_sha256 != source_object_receipt.physical_sha256:
            raise MassiveTradeReplayError("trade source bytes differ from source authority")
        if event.security_id != identity_resolution.security_id or event.source_ticker != identity_resolution.source_ticker:
            raise MassiveTradeReplayError("trade identity differs from PIT resolution")
        if event.session_date != session.session_date:
            raise MassiveTradeReplayError("trade session differs from session authority")
        if event.strategy_available_timestamp_ns != (
            event.sip_timestamp_ns
            + entitlement_authority.entitlement_delay_minutes * 60 * 1_000_000_000
        ):
            raise MassiveTradeReplayError("trade availability differs from entitlement delay")
        if event.regular_session is not session.is_regular(event.participant_timestamp_ns):
            raise MassiveTradeReplayError("trade session eligibility differs from calendar")
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

    active: dict[str, MassiveTradeEventV2] = {}
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
        "entitlement_authority_receipt_sha256": entitlement_authority.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "source_object_receipt_sha256": source_object_receipt.receipt_sha256,
        "identity_authority_receipt_sha256": identity_resolution.identity_authority_receipt_sha256,
        "ticker_history_receipt_sha256": identity_resolution.ticker_history_receipt_sha256,
        "identity_resolution_receipt_sha256": identity_resolution.receipt_sha256,
        "input_source_record_inventory_sha256": semantic_sha256(
            tuple(
                sorted(
                    (
                        event.source_ticker,
                        event.sequence_number,
                        event.source_record_sha256,
                    )
                    for event in events
                )
            )
        ),
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
        entitlement_authority_receipt_sha256=entitlement_authority.receipt_sha256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        correction_authority_receipt_sha256=correction_authority.receipt_sha256,
        source_object_receipt_sha256=source_object_receipt.receipt_sha256,
        identity_authority_receipt_sha256=identity_resolution.identity_authority_receipt_sha256,
        ticker_history_receipt_sha256=identity_resolution.ticker_history_receipt_sha256,
        identity_resolution_receipt_sha256=identity_resolution.receipt_sha256,
        input_source_record_inventory_sha256=semantic_sha256(
            tuple(
                sorted(
                    (
                        event.source_ticker,
                        event.sequence_number,
                        event.source_record_sha256,
                    )
                    for event in events
                )
            )
        ),
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
    "MASSIVE_RESOLVED_SECURITY_IDENTITY_SCHEMA",
    "MASSIVE_TRADE_EVENT_SCHEMA",
    "MASSIVE_TRADE_REPLAY_SCHEMA",
    "MassiveResolvedSecurityIdentity",
    "MassiveTradeEventV2",
    "MassiveTradeReplayError",
    "MassiveTradeReplayResult",
    "normalize_massive_trade_event",
    "replay_massive_trades",
]
