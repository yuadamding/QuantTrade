"""Protocol-bound decision timestamps for Massive adaptive-alpha replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL_ID,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)


MASSIVE_DECISION_CLOCK_SCHEMA = "rl-quant.massive-decision-clock-v1"
MASSIVE_DECISION_DELAY_NS = 60 * 60 * 1_000_000_000


class MassiveDecisionClockError(ValueError):
    """A replay decision clock differs from the frozen protocol or calendar."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveDecisionClockError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MassiveDecisionClockAuthority:
    protocol_id: str
    protocol_receipt_sha256: str
    session_authority_receipt_sha256: str
    exchange: str
    session_date: str
    regular_open_ns: int
    regular_close_ns: int
    decision_delay_ns: int
    decision_at_ns: int
    receipt_sha256: str
    schema: str = MASSIVE_DECISION_CLOCK_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_DECISION_CLOCK_SCHEMA:
            raise MassiveDecisionClockError("decision clock schema drifted")
        if self.protocol_id != MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL_ID:
            raise MassiveDecisionClockError("decision clock protocol ID drifted")
        if self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256:
            raise MassiveDecisionClockError("decision clock protocol receipt drifted")
        if MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.decision_rule != (
            "once-daily-close-plus-60-minutes"
        ):
            raise MassiveDecisionClockError("adaptive protocol decision rule drifted")
        if not self.exchange or not self.session_date:
            raise MassiveDecisionClockError("decision clock identity is absent")
        for name in ("regular_open_ns", "regular_close_ns", "decision_at_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveDecisionClockError(f"{name} must be nonnegative")
        if self.regular_close_ns <= self.regular_open_ns:
            raise MassiveDecisionClockError("decision clock session is empty")
        if self.decision_delay_ns != MASSIVE_DECISION_DELAY_NS:
            raise MassiveDecisionClockError("decision clock delay drifted")
        if self.decision_at_ns != self.regular_close_ns + self.decision_delay_ns:
            raise MassiveDecisionClockError("decision timestamp was not derived from close")
        _digest("session authority receipt", self.session_authority_receipt_sha256)
        _digest("decision clock receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveDecisionClockError("decision clock receipt differs")


def build_massive_decision_clock_authority(
    *,
    session_authority: MassiveSessionAuthority,
    session: MassiveExchangeSession,
) -> MassiveDecisionClockAuthority:
    """Derive the only valid decision timestamp for one exchange session."""

    session_authority.validate()
    session.validate()
    if session_authority.resolve(
        exchange=session.exchange, session_date=session.session_date
    ) != session:
        raise MassiveDecisionClockError("session was not resolved by its authority")
    body = {
        "schema": MASSIVE_DECISION_CLOCK_SCHEMA,
        "protocol_id": MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL_ID,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "exchange": session.exchange,
        "session_date": session.session_date,
        "regular_open_ns": session.regular_open_ns,
        "regular_close_ns": session.regular_close_ns,
        "decision_delay_ns": MASSIVE_DECISION_DELAY_NS,
        "decision_at_ns": session.regular_close_ns + MASSIVE_DECISION_DELAY_NS,
    }
    value = MassiveDecisionClockAuthority(
        protocol_id=MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL_ID,
        protocol_receipt_sha256=MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        exchange=session.exchange,
        session_date=session.session_date,
        regular_open_ns=session.regular_open_ns,
        regular_close_ns=session.regular_close_ns,
        decision_delay_ns=MASSIVE_DECISION_DELAY_NS,
        decision_at_ns=session.regular_close_ns + MASSIVE_DECISION_DELAY_NS,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


__all__ = [
    "MASSIVE_DECISION_CLOCK_SCHEMA",
    "MASSIVE_DECISION_DELAY_NS",
    "MassiveDecisionClockAuthority",
    "MassiveDecisionClockError",
    "build_massive_decision_clock_authority",
]
