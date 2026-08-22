"""Content-addressed exchange-session times for Massive feature replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_SESSION_AUTHORITY_SCHEMA = "rl-quant.massive-session-authority-v1"
FIVE_MINUTES_NS = 300 * 1_000_000_000


class MassiveSessionError(ValueError):
    """A session boundary is absent or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveSessionError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MassiveExchangeSession:
    session_date: str
    exchange: str
    regular_open_ns: int
    regular_close_ns: int
    scheduled_five_minute_intervals: int
    special_session_reason: str | None
    calendar_source_receipt_sha256: str

    def validate(self) -> None:
        if not self.session_date or not self.exchange:
            raise MassiveSessionError("session identity is absent")
        for name in ("regular_open_ns", "regular_close_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveSessionError(f"{name} must be nonnegative")
        if self.regular_close_ns <= self.regular_open_ns:
            raise MassiveSessionError("session close must follow its open")
        duration = self.regular_close_ns - self.regular_open_ns
        if duration % FIVE_MINUTES_NS:
            raise MassiveSessionError("session duration is not five-minute aligned")
        expected = duration // FIVE_MINUTES_NS
        if self.scheduled_five_minute_intervals != expected:
            raise MassiveSessionError("scheduled interval count differs from calendar")
        if self.special_session_reason is not None and (
            not self.special_session_reason
            or self.special_session_reason != self.special_session_reason.strip()
        ):
            raise MassiveSessionError("special-session reason is not canonical")
        _digest("calendar source receipt", self.calendar_source_receipt_sha256)

    def is_regular(self, timestamp_ns: int) -> bool:
        self.validate()
        return self.regular_open_ns <= timestamp_ns < self.regular_close_ns

    def five_minute_interval(self, timestamp_ns: int) -> int:
        if not self.is_regular(timestamp_ns):
            raise MassiveSessionError("timestamp lies outside the regular session")
        return (timestamp_ns - self.regular_open_ns) // FIVE_MINUTES_NS


@dataclass(frozen=True, slots=True)
class MassiveSessionAuthority:
    sessions: tuple[MassiveExchangeSession, ...]
    calendar_source_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_SESSION_AUTHORITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sessions": [asdict(row) for row in self.sessions],
            "calendar_source_receipt_sha256": self.calendar_source_receipt_sha256,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_SESSION_AUTHORITY_SCHEMA:
            raise MassiveSessionError("session authority schema drifted")
        keys = tuple((row.exchange, row.session_date) for row in self.sessions)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveSessionError("sessions must be sorted and unique")
        for row in self.sessions:
            row.validate()
            if row.calendar_source_receipt_sha256 != self.calendar_source_receipt_sha256:
                raise MassiveSessionError("session calendar identities differ")
        _digest("calendar source receipt", self.calendar_source_receipt_sha256)
        _digest("session authority receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveSessionError("session authority receipt differs")

    def resolve(self, *, exchange: str, session_date: str) -> MassiveExchangeSession:
        self.validate()
        for row in self.sessions:
            if row.exchange == exchange and row.session_date == session_date:
                return row
        raise MassiveSessionError("session is absent from the calendar authority")


def build_massive_session_authority(
    sessions: tuple[MassiveExchangeSession, ...],
    *,
    calendar_source_receipt_sha256: str,
) -> MassiveSessionAuthority:
    source = _digest("calendar source receipt", calendar_source_receipt_sha256)
    ordered = tuple(sorted(sessions, key=lambda row: (row.exchange, row.session_date)))
    body = {
        "schema": MASSIVE_SESSION_AUTHORITY_SCHEMA,
        "sessions": [asdict(row) for row in ordered],
        "calendar_source_receipt_sha256": source,
    }
    authority = MassiveSessionAuthority(
        sessions=ordered,
        calendar_source_receipt_sha256=source,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


__all__ = [
    "FIVE_MINUTES_NS",
    "MASSIVE_SESSION_AUTHORITY_SCHEMA",
    "MassiveExchangeSession",
    "MassiveSessionAuthority",
    "MassiveSessionError",
    "build_massive_session_authority",
]
