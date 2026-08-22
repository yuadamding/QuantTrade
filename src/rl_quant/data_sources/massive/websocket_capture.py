"""Immutable records for actual delayed Massive WebSocket observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from typing import Sequence
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.entitlement import MassiveEntitlementAuthority
from rl_quant.data_sources.massive.source_receipts import MassiveSourceObjectReceipt
from rl_quant.protocol.canonical_artifact import canonical_json_payload, semantic_sha256


MASSIVE_DELAYED_WEBSOCKET_EVENT_SCHEMA = (
    "rl-quant.massive-delayed-websocket-event-v2"
)
MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA = (
    "rl-quant.massive-delayed-websocket-capture-v2"
)
MASSIVE_WEBSOCKET_CAPTURE_LIFECYCLE_SCHEMA = (
    "rl-quant.massive-websocket-capture-lifecycle-v1"
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
    session_date: str
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
        _text("session date", self.session_date)
        _text("ticker", self.ticker)
        _text("canonical payload", self.canonical_payload_json)
        payload = json.loads(self.canonical_payload_json)
        canonical = canonical_json_payload(payload).decode("ascii")
        if canonical != self.canonical_payload_json:
            raise MassiveWebSocketCaptureError("captured payload is not canonical JSON")
        if payload.get("ev") != "T":
            raise MassiveWebSocketCaptureError("capture payload is not a stock trade")
        if payload.get("sym") != self.ticker:
            raise MassiveWebSocketCaptureError("capture ticker differs from payload")
        if _integer("SIP timestamp", payload.get("t")) != self.sip_timestamp_ms:
            raise MassiveWebSocketCaptureError("capture timestamp differs from payload")
        if _integer("sequence number", payload.get("q")) != self.sequence_number:
            raise MassiveWebSocketCaptureError("capture sequence differs from payload")
        observed_session = datetime.fromtimestamp(
            self.sip_timestamp_ms / 1_000,
            tz=ZoneInfo("America/New_York"),
        ).date().isoformat()
        if observed_session != self.session_date:
            raise MassiveWebSocketCaptureError(
                "capture session differs from its SIP timestamp"
            )
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
            session_date=datetime.fromtimestamp(
                _integer("SIP timestamp", payload["t"]) / 1_000,
                tz=ZoneInfo("America/New_York"),
            ).date().isoformat(),
            ticker=str(payload["sym"]),
            sip_timestamp_ms=_integer("SIP timestamp", payload["t"]),
            sequence_number=_integer("sequence number", payload["q"]),
            canonical_payload_json=canonical,
            payload_sha256=semantic_sha256(payload),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveWebSocketCaptureLifecycle:
    session_date: str
    connected_at_ns: int
    authenticated_at_ns: int
    subscribed_at_ns: int
    required_capture_start_ns: int
    required_capture_end_ns: int
    last_heartbeat_at_ns: int
    disconnected_intervals: tuple[tuple[int, int], ...]
    authentication_ack_sha256: str
    subscription_ack_sha256: str
    clock_authority_receipt_sha256: str
    raw_capture_source_receipt_sha256: str
    subscription_universe_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_WEBSOCKET_CAPTURE_LIFECYCLE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_WEBSOCKET_CAPTURE_LIFECYCLE_SCHEMA:
            raise MassiveWebSocketCaptureError("capture lifecycle schema drifted")
        _text("session date", self.session_date)
        for name in (
            "connected_at_ns",
            "authenticated_at_ns",
            "subscribed_at_ns",
            "required_capture_start_ns",
            "required_capture_end_ns",
            "last_heartbeat_at_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveWebSocketCaptureError(f"{name} must be nonnegative")
        if self.required_capture_end_ns <= self.required_capture_start_ns:
            raise MassiveWebSocketCaptureError("required capture window is empty")
        if self.disconnected_intervals != tuple(sorted(set(self.disconnected_intervals))):
            raise MassiveWebSocketCaptureError(
                "capture disconnect intervals are not canonical"
            )
        previous_end = -1
        for start, end in self.disconnected_intervals:
            if start < 0 or end <= start or start < previous_end:
                raise MassiveWebSocketCaptureError(
                    "capture disconnect interval is invalid"
                )
            previous_end = end
        for name in (
            "authentication_ack_sha256",
            "subscription_ack_sha256",
            "clock_authority_receipt_sha256",
            "raw_capture_source_receipt_sha256",
            "subscription_universe_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveWebSocketCaptureError("capture lifecycle receipt differs")

    @property
    def covers_required_window(self) -> bool:
        self.validate()
        prerequisites = (
            self.connected_at_ns <= self.required_capture_start_ns
            and self.authenticated_at_ns <= self.required_capture_start_ns
            and self.subscribed_at_ns <= self.required_capture_start_ns
            and self.last_heartbeat_at_ns >= self.required_capture_end_ns
        )
        interrupted = any(
            start < self.required_capture_end_ns
            and end > self.required_capture_start_ns
            for start, end in self.disconnected_intervals
        )
        return prerequisites and not interrupted

    @classmethod
    def build(
        cls,
        *,
        session_date: str,
        connected_at_ns: int,
        authenticated_at_ns: int,
        subscribed_at_ns: int,
        required_capture_start_ns: int,
        required_capture_end_ns: int,
        last_heartbeat_at_ns: int,
        disconnected_intervals: Sequence[tuple[int, int]],
        authentication_ack_sha256: str,
        subscription_ack_sha256: str,
        clock_authority_receipt_sha256: str,
        raw_capture_source_receipt_sha256: str,
        subscription_universe_receipt_sha256: str,
    ) -> MassiveWebSocketCaptureLifecycle:
        body = {
            "schema": MASSIVE_WEBSOCKET_CAPTURE_LIFECYCLE_SCHEMA,
            "session_date": session_date,
            "connected_at_ns": connected_at_ns,
            "authenticated_at_ns": authenticated_at_ns,
            "subscribed_at_ns": subscribed_at_ns,
            "required_capture_start_ns": required_capture_start_ns,
            "required_capture_end_ns": required_capture_end_ns,
            "last_heartbeat_at_ns": last_heartbeat_at_ns,
            "disconnected_intervals": tuple(sorted(set(disconnected_intervals))),
            "authentication_ack_sha256": authentication_ack_sha256,
            "subscription_ack_sha256": subscription_ack_sha256,
            "clock_authority_receipt_sha256": clock_authority_receipt_sha256,
            "raw_capture_source_receipt_sha256": raw_capture_source_receipt_sha256,
            "subscription_universe_receipt_sha256": subscription_universe_receipt_sha256,
        }
        value = cls(
            session_date=session_date,
            connected_at_ns=connected_at_ns,
            authenticated_at_ns=authenticated_at_ns,
            subscribed_at_ns=subscribed_at_ns,
            required_capture_start_ns=required_capture_start_ns,
            required_capture_end_ns=required_capture_end_ns,
            last_heartbeat_at_ns=last_heartbeat_at_ns,
            disconnected_intervals=tuple(sorted(set(disconnected_intervals))),
            authentication_ack_sha256=authentication_ack_sha256,
            subscription_ack_sha256=subscription_ack_sha256,
            clock_authority_receipt_sha256=clock_authority_receipt_sha256,
            raw_capture_source_receipt_sha256=raw_capture_source_receipt_sha256,
            subscription_universe_receipt_sha256=subscription_universe_receipt_sha256,
            receipt_sha256=semantic_sha256(body),
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
    payload_inventory_sha256: str
    entitlement_receipt_sha256: str
    raw_capture_source_receipt_sha256: str
    lifecycle: MassiveWebSocketCaptureLifecycle
    capture_complete: bool
    secret_material_persisted: bool
    receipt_sha256: str
    schema: str = MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate(self) -> None:
        if self.schema != MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA:
            raise MassiveWebSocketCaptureError("capture authority schema drifted")
        _text("session date", self.session_date)
        if self.endpoint != "wss://delayed.massive.com/stocks":
            raise MassiveWebSocketCaptureError("capture did not use delayed endpoint")
        if (
            not self.subscribed_tickers
            or self.subscribed_tickers != tuple(sorted(set(self.subscribed_tickers)))
        ):
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
            "payload_inventory_sha256",
            "entitlement_receipt_sha256",
            "raw_capture_source_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        self.lifecycle.validate()
        if self.lifecycle.session_date != self.session_date:
            raise MassiveWebSocketCaptureError(
                "capture and lifecycle sessions differ"
            )
        if (
            self.lifecycle.raw_capture_source_receipt_sha256
            != self.raw_capture_source_receipt_sha256
        ):
            raise MassiveWebSocketCaptureError(
                "capture raw source differs from lifecycle"
            )
        if self.capture_complete is not self.lifecycle.covers_required_window:
            raise MassiveWebSocketCaptureError(
                "capture completeness differs from lifecycle"
            )
        if self.secret_material_persisted:
            raise MassiveWebSocketCaptureError(
                "capture authority cannot contain credential material"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveWebSocketCaptureError("capture authority receipt differs")


def build_massive_delayed_websocket_capture_authority(
    events: Sequence[MassiveDelayedWebSocketEvent],
    *,
    lifecycle: MassiveWebSocketCaptureLifecycle,
    subscribed_tickers: Sequence[str],
    entitlement_authority: MassiveEntitlementAuthority,
    raw_capture_source_receipt: MassiveSourceObjectReceipt,
) -> MassiveDelayedWebSocketCaptureAuthority:
    """Seal one already-recorded delayed session without retaining a key."""

    entitlement_authority.validate()
    raw_capture_source_receipt.validate()
    lifecycle.validate()
    if (
        raw_capture_source_receipt.entitlement_receipt_sha256
        != entitlement_authority.receipt_sha256
    ):
        raise MassiveWebSocketCaptureError(
            "raw capture used another entitlement authority"
        )
    if (
        lifecycle.raw_capture_source_receipt_sha256
        != raw_capture_source_receipt.receipt_sha256
    ):
        raise MassiveWebSocketCaptureError(
            "capture lifecycle used another raw source object"
        )
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
    if not subscriptions:
        raise MassiveWebSocketCaptureError("capture subscription inventory is empty")
    for event in events:
        if event.session_date != lifecycle.session_date:
            raise MassiveWebSocketCaptureError(
                "captured event lies outside the lifecycle session"
            )
        if event.ticker not in subscriptions:
            raise MassiveWebSocketCaptureError(
                "captured event ticker was not subscribed"
            )
        if event.received_at_ns < lifecycle.connected_at_ns:
            raise MassiveWebSocketCaptureError(
                "captured event predates the connection"
            )
        if event.received_at_ns > lifecycle.last_heartbeat_at_ns:
            raise MassiveWebSocketCaptureError(
                "captured event follows the final heartbeat"
            )
    body = {
        "schema": MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA,
        "session_date": lifecycle.session_date,
        "endpoint": "wss://delayed.massive.com/stocks",
        "subscribed_tickers": subscriptions,
        "event_count": len(events),
        "first_received_at_ns": min(event.received_at_ns for event in events),
        "last_received_at_ns": max(event.received_at_ns for event in events),
        "event_inventory_sha256": semantic_sha256(ordered_inventory),
        "payload_inventory_sha256": semantic_sha256(
            tuple(
                sorted(
                    (event.ticker, event.sequence_number, event.payload_sha256)
                    for event in events
                )
            )
        ),
        "entitlement_receipt_sha256": entitlement_authority.receipt_sha256,
        "raw_capture_source_receipt_sha256": raw_capture_source_receipt.receipt_sha256,
        "lifecycle": asdict(lifecycle),
        "capture_complete": lifecycle.covers_required_window,
        "secret_material_persisted": False,
    }
    authority = MassiveDelayedWebSocketCaptureAuthority(
        session_date=lifecycle.session_date,
        endpoint="wss://delayed.massive.com/stocks",
        subscribed_tickers=subscriptions,
        event_count=len(events),
        first_received_at_ns=min(event.received_at_ns for event in events),
        last_received_at_ns=max(event.received_at_ns for event in events),
        event_inventory_sha256=semantic_sha256(ordered_inventory),
        payload_inventory_sha256=semantic_sha256(
            tuple(
                sorted(
                    (event.ticker, event.sequence_number, event.payload_sha256)
                    for event in events
                )
            )
        ),
        entitlement_receipt_sha256=entitlement_authority.receipt_sha256,
        raw_capture_source_receipt_sha256=raw_capture_source_receipt.receipt_sha256,
        lifecycle=lifecycle,
        capture_complete=lifecycle.covers_required_window,
        secret_material_persisted=False,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


__all__ = [
    "MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA",
    "MASSIVE_DELAYED_WEBSOCKET_EVENT_SCHEMA",
    "MASSIVE_WEBSOCKET_CAPTURE_LIFECYCLE_SCHEMA",
    "MassiveDelayedWebSocketCaptureAuthority",
    "MassiveDelayedWebSocketEvent",
    "MassiveWebSocketCaptureError",
    "MassiveWebSocketCaptureLifecycle",
    "build_massive_delayed_websocket_capture_authority",
]
