"""Immutable records for actual delayed Massive WebSocket observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.decision_clock import (
    MassiveDecisionClockAuthority,
)
from rl_quant.data_sources.massive.entitlement import MassiveEntitlementAuthority
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    MassiveSourceObjectReceipt,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import canonical_json_payload, semantic_sha256


MASSIVE_DELAYED_WEBSOCKET_EVENT_SCHEMA = (
    "rl-quant.massive-delayed-websocket-event-v2"
)
MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA = (
    "rl-quant.massive-delayed-websocket-capture-v3"
)
MASSIVE_WEBSOCKET_CAPTURE_LIFECYCLE_SCHEMA = (
    "rl-quant.massive-websocket-capture-lifecycle-v2"
)
MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA = "rl-quant.massive-websocket-capture-file-v1"


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
        for key, name in (("x", "exchange"), ("pt", "participant timestamp")):
            _integer(name, payload.get(key))
        trade_id = payload.get("i")
        if not isinstance(trade_id, str) or not trade_id:
            raise MassiveWebSocketCaptureError("capture trade ID is absent")
        try:
            price = Decimal(str(payload.get("p")))
            size = Decimal(str(payload.get("ds", payload.get("s"))))
        except InvalidOperation as exc:
            raise MassiveWebSocketCaptureError(
                "capture price or size is not decimal"
            ) from exc
        if not price.is_finite() or price <= 0 or not size.is_finite() or size <= 0:
            raise MassiveWebSocketCaptureError(
                "capture price and size must be finite and positive"
            )
        conditions = payload.get("c", ())
        if not isinstance(conditions, list):
            raise MassiveWebSocketCaptureError("capture conditions must be a list")
        normalized_conditions = tuple(_integer("condition ID", row) for row in conditions)
        if normalized_conditions != tuple(sorted(set(normalized_conditions))):
            raise MassiveWebSocketCaptureError(
                "capture conditions must be sorted and unique"
            )
        if payload.get("trfi") is not None:
            _integer("TRF ID", payload["trfi"])
        if payload.get("trft") is not None:
            _integer("TRF timestamp", payload["trft"])
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
    decision_clock_receipt_sha256: str
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
            "decision_clock_receipt_sha256",
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
        decision_clock: MassiveDecisionClockAuthority,
        connected_at_ns: int,
        authenticated_at_ns: int,
        subscribed_at_ns: int,
        last_heartbeat_at_ns: int,
        disconnected_intervals: Sequence[tuple[int, int]],
        authentication_ack_sha256: str,
        subscription_ack_sha256: str,
        raw_capture_source_receipt_sha256: str,
        subscription_universe_receipt_sha256: str,
    ) -> MassiveWebSocketCaptureLifecycle:
        decision_clock.validate()
        body = {
            "schema": MASSIVE_WEBSOCKET_CAPTURE_LIFECYCLE_SCHEMA,
            "session_date": decision_clock.session_date,
            "connected_at_ns": connected_at_ns,
            "authenticated_at_ns": authenticated_at_ns,
            "subscribed_at_ns": subscribed_at_ns,
            "required_capture_start_ns": decision_clock.regular_open_ns,
            "required_capture_end_ns": decision_clock.decision_at_ns,
            "last_heartbeat_at_ns": last_heartbeat_at_ns,
            "disconnected_intervals": tuple(sorted(set(disconnected_intervals))),
            "authentication_ack_sha256": authentication_ack_sha256,
            "subscription_ack_sha256": subscription_ack_sha256,
            "decision_clock_receipt_sha256": decision_clock.receipt_sha256,
            "raw_capture_source_receipt_sha256": raw_capture_source_receipt_sha256,
            "subscription_universe_receipt_sha256": subscription_universe_receipt_sha256,
        }
        value = cls(
            session_date=decision_clock.session_date,
            connected_at_ns=connected_at_ns,
            authenticated_at_ns=authenticated_at_ns,
            subscribed_at_ns=subscribed_at_ns,
            required_capture_start_ns=decision_clock.regular_open_ns,
            required_capture_end_ns=decision_clock.decision_at_ns,
            last_heartbeat_at_ns=last_heartbeat_at_ns,
            disconnected_intervals=tuple(sorted(set(disconnected_intervals))),
            authentication_ack_sha256=authentication_ack_sha256,
            subscription_ack_sha256=subscription_ack_sha256,
            decision_clock_receipt_sha256=decision_clock.receipt_sha256,
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
    loaded_source_receipt_sha256: str | None
    capture_file_parser_qualified: bool
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
        if not isinstance(self.capture_file_parser_qualified, bool):
            raise MassiveWebSocketCaptureError(
                "capture parser qualification must be Boolean"
            )
        if self.capture_file_parser_qualified:
            _digest("loaded capture source", self.loaded_source_receipt_sha256)
        elif self.loaded_source_receipt_sha256 is not None:
            raise MassiveWebSocketCaptureError(
                "unqualified capture cannot claim a loaded source receipt"
            )
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
    loaded_source: LoadedMassiveSourceObject | None = None,
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
    if loaded_source is not None:
        loaded_source.validate()
        if loaded_source.receipt != raw_capture_source_receipt:
            raise MassiveWebSocketCaptureError(
                "loaded capture source differs from source receipt"
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
    if lifecycle.subscription_universe_receipt_sha256 != semantic_sha256(
        subscriptions
    ):
        raise MassiveWebSocketCaptureError(
            "subscription inventory differs from lifecycle evidence"
        )
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
        "loaded_source_receipt_sha256": None
        if loaded_source is None
        else loaded_source.receipt_sha256,
        "capture_file_parser_qualified": loaded_source is not None,
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
        loaded_source_receipt_sha256=None
        if loaded_source is None
        else loaded_source.receipt_sha256,
        capture_file_parser_qualified=loaded_source is not None,
        lifecycle=lifecycle,
        capture_complete=lifecycle.covers_required_window,
        secret_material_persisted=False,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


def parse_massive_delayed_websocket_capture(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    decision_clock: MassiveDecisionClockAuthority,
    entitlement_authority: MassiveEntitlementAuthority,
) -> tuple[
    tuple[MassiveDelayedWebSocketEvent, ...],
    MassiveDelayedWebSocketCaptureAuthority,
]:
    """Derive lifecycle and trade events from one committed recorder JSONL file.

    Recorder rows are canonical JSON objects with ``schema``, ``kind``, and
    ``recorded_at_ns``. ``server-batch`` rows contain the unmodified Massive
    message array. ``subscription-ack`` rows bind the safe ticker inventory to
    the corresponding server status payload. Disconnect and checkpoint rows
    are local recorder evidence; credential-bearing client auth messages are
    prohibited.
    """

    loaded_source.validate()
    decision_clock.validate()
    entitlement_authority.validate()
    if (
        loaded_source.receipt.entitlement_receipt_sha256
        != entitlement_authority.receipt_sha256
    ):
        raise MassiveWebSocketCaptureError(
            "committed capture used another entitlement authority"
        )
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line:
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise MassiveWebSocketCaptureError(
                f"capture row {line_number} is not JSON"
            ) from exc
        if not isinstance(row, dict):
            raise MassiveWebSocketCaptureError("capture rows must be objects")
        if canonical_json_payload(row) != raw_line:
            raise MassiveWebSocketCaptureError("capture row is not canonical JSON")
        if row.get("schema") != MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA:
            raise MassiveWebSocketCaptureError("capture file schema drifted")
        if "apiKey" in raw_line.decode("utf-8") or row.get("action") == "auth":
            raise MassiveWebSocketCaptureError("capture file contains credential material")
        _integer("recorded timestamp", row.get("recorded_at_ns"))
        rows.append(row)
    if not rows:
        raise MassiveWebSocketCaptureError("committed capture file is empty")

    connected_at: int | None = None
    authenticated_at: int | None = None
    authentication_payload: dict[str, object] | None = None
    subscription_at: int | None = None
    subscription_payload: dict[str, object] | None = None
    subscriptions: tuple[str, ...] = ()
    checkpoint_at: int | None = None
    disconnect_start: int | None = None
    disconnects: list[tuple[int, int]] = []
    events: list[MassiveDelayedWebSocketEvent] = []
    for row in rows:
        kind = row.get("kind")
        recorded_at = _integer("recorded timestamp", row["recorded_at_ns"])
        if kind == "server-batch":
            messages = row.get("payload")
            if not isinstance(messages, list) or not messages:
                raise MassiveWebSocketCaptureError("server batch is empty")
            for message in messages:
                if not isinstance(message, dict):
                    raise MassiveWebSocketCaptureError("server message is not an object")
                if message.get("ev") == "status":
                    if message.get("status") == "connected":
                        connected_at = recorded_at
                    elif message.get("status") == "auth_success":
                        authenticated_at = recorded_at
                        authentication_payload = message
                elif message.get("ev") == "T":
                    events.append(
                        MassiveDelayedWebSocketEvent.from_payload(
                            message, received_at_ns=recorded_at
                        )
                    )
        elif kind == "subscription-ack":
            tickers = row.get("tickers")
            payload = row.get("payload")
            if not isinstance(tickers, list) or not tickers:
                raise MassiveWebSocketCaptureError("subscription inventory is absent")
            if not isinstance(payload, dict) or payload.get("ev") != "status":
                raise MassiveWebSocketCaptureError("subscription acknowledgement is absent")
            subscriptions = tuple(sorted(set(str(ticker) for ticker in tickers)))
            subscription_at = recorded_at
            subscription_payload = payload
        elif kind == "disconnect-start":
            if disconnect_start is not None:
                raise MassiveWebSocketCaptureError("nested disconnect interval")
            disconnect_start = recorded_at
        elif kind == "disconnect-end":
            if disconnect_start is None or recorded_at <= disconnect_start:
                raise MassiveWebSocketCaptureError("disconnect end lacks a start")
            disconnects.append((disconnect_start, recorded_at))
            disconnect_start = None
        elif kind == "recorder-checkpoint":
            checkpoint_at = max(checkpoint_at or 0, recorded_at)
        else:
            raise MassiveWebSocketCaptureError("capture row kind is unsupported")
    if disconnect_start is not None:
        disconnects.append((disconnect_start, checkpoint_at or disconnect_start + 1))
    if (
        connected_at is None
        or authenticated_at is None
        or authentication_payload is None
        or subscription_at is None
        or subscription_payload is None
        or checkpoint_at is None
    ):
        raise MassiveWebSocketCaptureError("capture lifecycle evidence is incomplete")
    lifecycle = MassiveWebSocketCaptureLifecycle.build(
        decision_clock=decision_clock,
        connected_at_ns=connected_at,
        authenticated_at_ns=authenticated_at,
        subscribed_at_ns=subscription_at,
        last_heartbeat_at_ns=checkpoint_at,
        disconnected_intervals=disconnects,
        authentication_ack_sha256=semantic_sha256(authentication_payload),
        subscription_ack_sha256=semantic_sha256(subscription_payload),
        raw_capture_source_receipt_sha256=loaded_source.receipt.receipt_sha256,
        subscription_universe_receipt_sha256=semantic_sha256(subscriptions),
    )
    capture = build_massive_delayed_websocket_capture_authority(
        events,
        lifecycle=lifecycle,
        subscribed_tickers=subscriptions,
        entitlement_authority=entitlement_authority,
        raw_capture_source_receipt=loaded_source.receipt,
        loaded_source=loaded_source,
    )
    return tuple(events), capture


__all__ = [
    "MASSIVE_DELAYED_WEBSOCKET_CAPTURE_SCHEMA",
    "MASSIVE_DELAYED_WEBSOCKET_EVENT_SCHEMA",
    "MASSIVE_WEBSOCKET_CAPTURE_LIFECYCLE_SCHEMA",
    "MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA",
    "MassiveDelayedWebSocketCaptureAuthority",
    "MassiveDelayedWebSocketEvent",
    "MassiveWebSocketCaptureError",
    "MassiveWebSocketCaptureLifecycle",
    "build_massive_delayed_websocket_capture_authority",
    "parse_massive_delayed_websocket_capture",
]
