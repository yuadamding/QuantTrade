"""Authority-derived decision origins for the finalized validation typed lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.finalized_archive_scope import (
    MassiveFinalizedArchiveScopeV2,
)
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_object_acquisition import (
    MassiveAuthenticatedFlatFileDownloadV1,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)

MASSIVE_TYPED_DECISION_ORIGIN_V1_SCHEMA = (
    "rl-quant.massive-finalized-typed-decision-origin-v1"
)
MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "exchange": "XNYS",
        "source_selection": "immediately-prior-exchange-session",
        "primary_source_staleness_sessions": 1,
        "source_availability": "authenticated-download-complete-by-decision",
        "decision_local_time": "12:30:00 America/New_York",
        "diagnostic_fill_window": "[15:50:00,16:00:00) America/New_York",
    }
)
_EASTERN = ZoneInfo("America/New_York")


class MassiveTypedDecisionOriginError(ValueError):
    """The typed decision origin is not derived from its authorities."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveTypedDecisionOriginError(f"{name} must be a lowercase SHA-256")
    return value


def _local_ms(session_date: str, value: time) -> int:
    return int(
        datetime.combine(
            date.fromisoformat(session_date),
            value,
            tzinfo=_EASTERN,
        ).timestamp()
        * 1_000
    )


@dataclass(frozen=True, slots=True)
class MassiveTypedDecisionOriginV1:
    source_session_date: str
    decision_session_date: str
    exchange: str
    source_staleness_sessions: int
    decision_at_ms: int
    fill_start_at_ms: int
    fill_end_at_ms: int
    decision_regular_open_at_ms: int
    decision_regular_close_at_ms: int
    authenticated_download_receipt_sha256: str
    source_object_receipt_sha256: str
    archive_scope_receipt_sha256: str
    session_authority_receipt_sha256: str
    origin_spec_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_TYPED_DECISION_ORIGIN_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_TYPED_DECISION_ORIGIN_V1_SCHEMA
            or self.exchange != "XNYS"
            or self.source_staleness_sessions != 1
            or not (
                self.decision_regular_open_at_ms
                <= self.decision_at_ms
                < self.fill_start_at_ms
                < self.fill_end_at_ms
                <= self.decision_regular_close_at_ms
            )
        ):
            raise MassiveTypedDecisionOriginError("typed decision chronology differs")
        if (
            self.decision_at_ms
            != _local_ms(self.decision_session_date, time(12, 30))
            or self.fill_start_at_ms
            != _local_ms(self.decision_session_date, time(15, 50))
            or self.fill_end_at_ms
            != _local_ms(self.decision_session_date, time(16, 0))
        ):
            raise MassiveTypedDecisionOriginError("typed decision clock drifted")
        for name in (
            "authenticated_download_receipt_sha256",
            "source_object_receipt_sha256",
            "archive_scope_receipt_sha256",
            "session_authority_receipt_sha256",
            "origin_spec_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.origin_spec_receipt_sha256
            != MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256
        ):
            raise MassiveTypedDecisionOriginError("typed origin specification drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTypedDecisionOriginError("typed origin receipt differs")


def _session_position(
    *, authority: MassiveSessionAuthority, session: MassiveExchangeSession
) -> int:
    rows = tuple(row for row in authority.sessions if row.exchange == session.exchange)
    try:
        return rows.index(session)
    except ValueError as exc:  # pragma: no cover - guarded by resolve
        raise MassiveTypedDecisionOriginError("session is absent from authority") from exc


def build_massive_typed_decision_origin_v1(
    *,
    session_authority: MassiveSessionAuthority,
    source_session: MassiveExchangeSession,
    decision_session: MassiveExchangeSession,
    authenticated_download: MassiveAuthenticatedFlatFileDownloadV1,
    archive_scope: MassiveFinalizedArchiveScopeV2,
) -> MassiveTypedDecisionOriginV1:
    """Derive the primary V0 estimand's origin without free chronology fields."""

    session_authority.validate()
    source_session.validate()
    decision_session.validate()
    authenticated_download.validate()
    archive_scope.validate()
    if not archive_scope.qualification_complete:
        raise MassiveTypedDecisionOriginError("archive scope is incomplete")
    resolved_source = session_authority.resolve(
        exchange="XNYS", session_date=source_session.session_date
    )
    resolved_decision = session_authority.resolve(
        exchange="XNYS", session_date=decision_session.session_date
    )
    if resolved_source != source_session or resolved_decision != decision_session:
        raise MassiveTypedDecisionOriginError("session authority resolution differs")
    source_position = _session_position(
        authority=session_authority, session=source_session
    )
    decision_position = _session_position(
        authority=session_authority, session=decision_session
    )
    if decision_position != source_position + 1:
        raise MassiveTypedDecisionOriginError(
            "primary typed origin requires the immediately prior source session"
        )
    expected_key = canonical_massive_trade_object_key(source_session.session_date)
    if (
        source_session.session_date not in archive_scope.expected_source_session_dates
        or expected_key not in archive_scope.observed_in_scope_object_keys
        or authenticated_download.source_object_key != expected_key
        or authenticated_download.loaded_source.receipt.source_object_key != expected_key
        or authenticated_download.listing_acquisition_receipt_sha256
        not in archive_scope.captured_listing_receipts
        or archive_scope.session_authority_receipt_sha256
        != session_authority.receipt_sha256
    ):
        raise MassiveTypedDecisionOriginError("typed origin source authority differs")
    decision_at_ms = _local_ms(decision_session.session_date, time(12, 30))
    fill_start_at_ms = _local_ms(decision_session.session_date, time(15, 50))
    fill_end_at_ms = _local_ms(decision_session.session_date, time(16, 0))
    regular_open_at_ms = decision_session.regular_open_ns // 1_000_000
    regular_close_at_ms = decision_session.regular_close_ns // 1_000_000
    if authenticated_download.completed_at_ms > decision_at_ms:
        raise MassiveTypedDecisionOriginError(
            "authenticated source completed after the decision"
        )
    body = {
        "schema": MASSIVE_TYPED_DECISION_ORIGIN_V1_SCHEMA,
        "source_session_date": source_session.session_date,
        "decision_session_date": decision_session.session_date,
        "exchange": "XNYS",
        "source_staleness_sessions": 1,
        "decision_at_ms": decision_at_ms,
        "fill_start_at_ms": fill_start_at_ms,
        "fill_end_at_ms": fill_end_at_ms,
        "decision_regular_open_at_ms": regular_open_at_ms,
        "decision_regular_close_at_ms": regular_close_at_ms,
        "authenticated_download_receipt_sha256": authenticated_download.receipt_sha256,
        "source_object_receipt_sha256": authenticated_download.loaded_source.receipt.receipt_sha256,
        "archive_scope_receipt_sha256": archive_scope.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "origin_spec_receipt_sha256": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
    }
    result = MassiveTypedDecisionOriginV1(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256",
    "MassiveTypedDecisionOriginError",
    "MassiveTypedDecisionOriginV1",
    "build_massive_typed_decision_origin_v1",
]
