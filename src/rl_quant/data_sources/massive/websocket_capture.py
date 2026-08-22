"""Immutable records for actual delayed Massive WebSocket observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Sequence

from rl_quant.protocol.canonical_artifact import canonical_json_payload, semantic_sha256


MASSIVE_DELAYED_WEBSOCKET_EVENT_SCHEMA = (
    "rl-quant.massive-delayed-websocket-event-v1"
)
MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA = (
    "rl-quant.massive-delayed-websocket-capture-v1"
)


class MassiveWebSocketCaptureError(ValueError):
    """A delayed WebSocket capture is incomplete or non-canonical."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveWebSocketCaptureError(f"{name} must be canonical text")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveWebSocketCaptureError(f"{name} must be a SHA-256 digest")
    return value


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise MassiveWebSocketCaptureError(f"{name} must be an integer")
    try:
        observed = int(value)
    except ValueError as exc:
        raise MassiveWebSocketCaptureError(f"{name} must be an integer") from exc
    if observed < 0:
        raise MassiveWebSocketCaptureError(f"{name} must be nonnegative")
    return observed


@dataclass(frozen=True, slots=True)
class MassiveDelayedWebSocketEvent:
    received_at_ns: int
    ticker: str
    sip_timestamp_ms: int
    sequence_number: int
    canonical_payload_json: str
    payload_sha256: str
    schema: str = MASSIVE_DELAYED_WEBSOCKET_EVENT_SCHEMA

    def validate(self) -> None:
        if self.schema != MASSIVE_DELAYED_WEBSOCKET_EVENT_SCHEMA:
            raise MassiveWebSocketCaptureError("delayed event schema drifted")
        for name in ("received_at_ns", "sip_timestamp_ms", "sequence_number"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveWebSocketCaptureError(f"{name} must be nonnegative")
        _text("ticker", self.ticker)
        _text("canonical payload", self.canonical_payload_json)
        payload = json.loads(self.canonical_payload_json)
        canonical = canonical_json_payload(payload).decode("ascii")
        if canonical != self.canonical_payload_json:
            raise MassiveWebSocketCaptureError("captured payload is not canonical JSON")
        _digest("payload SHA", self.payload_sha256)
        if self.payload_sha256 != semantic_sha256(payload):
            raise MassiveWebSocketCaptureError("captured payload SHA differs")

    @classmethod
    def from_payload(
        cls, payload: dict[str, object], *, received_at_ns: int
    ) -> MassiveDelayedWebSocketEvent:
        canonical = canonical_json_payload(payload).decode("ascii")
        value = cls(
            received_at_ns=received_at_ns,
            ticker=str(payload["sym"]),
            sip_timestamp_ms=_integer("SIP timestamp", payload["t"]),
            sequence_number=_integer("sequence number", payload["q"]),
            canonical_payload_json=canonical,
            payload_sha256=semantic_sha256(payload),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveDelayedWebSocketCaptureAuthority:
    session_date: str
    endpoint: str
    subscribed_tickers: tuple[str, ...]
    event_count: int
    first_received_at_ns: int
    last_received_at_ns: int
    event_inventory_sha256: str
    entitlement_receipt_sha256: str
    capture_complete: bool
    secret_material_persisted: bool
    receipt_sha256: str
    schema: str = MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA:
            raise MassiveWebSocketCaptureError("capture authority schema drifted")
        _text("session date", self.session_date)
        if self.endpoint != "wss://delayed.massive.com/stocks":
            raise MassiveWebSocketCaptureError("capture did not use delayed endpoint")
        if self.subscribed_tickers != tuple(sorted(set(self.subscribed_tickers))):
            raise MassiveWebSocketCaptureError("subscriptions are not canonical")
        for ticker in self.subscribed_tickers:
            _text("subscribed ticker", ticker)
        for name in (
            "event_count",
            "first_received_at_ns",
            "last_received_at_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveWebSocketCaptureError(f"{name} must be nonnegative")
        if self.event_count <= 0 or self.last_received_at_ns < self.first_received_at_ns:
            raise MassiveWebSocketCaptureError("capture event chronology is invalid")
        for name in (
            "event_inventory_sha256",
            "entitlement_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.capture_complete is not True or self.secret_material_persisted:
            raise MassiveWebSocketCaptureError(
                "capture must be complete and contain no credential"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveWebSocketCaptureError("capture authority receipt differs")


def build_massive_delayed_websocket_capture_authority(
    events: Sequence[MassiveDelayedWebSocketEvent],
    *,
    session_date: str,
    subscribed_tickers: Sequence[str],
    entitlement_receipt_sha256: str,
) -> MassiveDelayedWebSocketCaptureAuthority:
    """Seal one already-recorded delayed session without retaining a key."""

    if not events:
        raise MassiveWebSocketCaptureError("cannot seal an empty delayed capture")
    for event in events:
        event.validate()
    ordered_inventory = tuple(
        sorted(
            (
                event.received_at_ns,
                event.ticker,
                event.sequence_number,
                event.payload_sha256,
            )
            for event in events
        )
    )
    subscriptions = tuple(sorted(set(subscribed_tickers)))
    body = {
        "schema": MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA,
        "session_date": session_date,
        "endpoint": "wss://delayed.massive.com/stocks",
        "subscribed_tickers": subscriptions,
        "event_count": len(events),
        "first_received_at_ns": min(event.received_at_ns for event in events),
        "last_received_at_ns": max(event.received_at_ns for event in events),
        "event_inventory_sha256": semantic_sha256(ordered_inventory),
        "entitlement_receipt_sha256": _digest(
            "entitlement receipt", entitlement_receipt_sha256
        ),
        "capture_complete": True,
        "secret_material_persisted": False,
    }
    authority = MassiveDelayedWebSocketCaptureAuthority(
        session_date=session_date,
        endpoint="wss://delayed.massive.com/stocks",
        subscribed_tickers=subscriptions,
        event_count=len(events),
        first_received_at_ns=min(event.received_at_ns for event in events),
        last_received_at_ns=max(event.received_at_ns for event in events),
        event_inventory_sha256=semantic_sha256(ordered_inventory),
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        capture_complete=True,
        secret_material_persisted=False,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


__all__ = [
    "MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA",
    "MASSIVE_DELAYED_WEBSOCKET_EVENT_SCHEMA",
    "MassiveDelayedWebSocketCaptureAuthority",
    "MassiveDelayedWebSocketEvent",
    "MassiveWebSocketCaptureError",
    "build_massive_delayed_websocket_capture_authority",
]
