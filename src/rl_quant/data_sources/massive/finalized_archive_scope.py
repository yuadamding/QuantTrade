"""Immutable historical archive scope for finalized validation readiness."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date

from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MassiveCapturedFlatFileListingV0,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.protocol.canonical_artifact import semantic_sha256

MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SCHEMA = (
    "rl-quant.massive-finalized-archive-scope-v1"
)
MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "exchange": "XNYS",
        "dates": "inclusive-session-authority-range",
        "expected_objects": "one-canonical-trades-v1-object-per-session",
        "listing_coverage": "every-calendar-month-in-range",
        "missing_objects": "explicit-and-disqualifying",
    }
)
MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SCHEMA = (
    "rl-quant.massive-finalized-archive-scope-v2"
)
MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "exchange": "XNYS",
        "dates": "inclusive-session-authority-range",
        "expected_objects": "one-canonical-trades-v1-object-per-session",
        "listing_coverage": "every-calendar-month-in-range",
        "membership": "expected-keys-subset-of-monthly-listing",
        "same_prefix_out_of_range": "inventoried-not-rejected",
        "missing_objects": "explicit-and-disqualifying",
    }
)


class MassiveFinalizedArchiveScopeError(ValueError):
    """Archive listings do not cover the frozen exchange-session range."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFinalizedArchiveScopeError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedArchiveScopeV1:
    start_session_date: str
    end_session_date: str
    exchange: str
    expected_month_prefixes: tuple[str, ...]
    captured_listing_receipts: tuple[str, ...]
    expected_source_session_dates: tuple[str, ...]
    expected_source_object_keys: tuple[str, ...]
    missing_source_objects: tuple[str, ...]
    session_authority_receipt_sha256: str
    scope_spec_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SCHEMA
            or self.exchange != "XNYS"
            or date.fromisoformat(self.start_session_date)
            > date.fromisoformat(self.end_session_date)
        ):
            raise MassiveFinalizedArchiveScopeError("archive scope identity differs")
        inventories = (
            self.expected_month_prefixes,
            self.captured_listing_receipts,
            self.expected_source_session_dates,
            self.expected_source_object_keys,
        )
        if any(
            not values or values != tuple(sorted(set(values))) for values in inventories
        ):
            raise MassiveFinalizedArchiveScopeError(
                "archive scope inventories are not canonical"
            )
        if len(self.expected_source_session_dates) != len(
            self.expected_source_object_keys
        ):
            raise MassiveFinalizedArchiveScopeError(
                "archive scope session/object counts differ"
            )
        expected_keys = tuple(
            canonical_massive_trade_object_key(value)
            for value in self.expected_source_session_dates
        )
        if self.expected_source_object_keys != expected_keys:
            raise MassiveFinalizedArchiveScopeError(
                "archive object key inventory differs"
            )
        if self.missing_source_objects != tuple(
            sorted(set(self.missing_source_objects))
        ):
            raise MassiveFinalizedArchiveScopeError("archive missing inventory differs")
        for name in (
            "session_authority_receipt_sha256",
            "scope_spec_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for value in self.captured_listing_receipts:
            _digest("captured listing receipt", value)
        if (
            self.scope_spec_receipt_sha256
            != MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SPEC_SHA256
        ):
            raise MassiveFinalizedArchiveScopeError(
                "archive scope specification drifted"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedArchiveScopeError("archive scope receipt differs")

    @property
    def qualification_complete(self) -> bool:
        return not self.missing_source_objects


@dataclass(frozen=True, slots=True)
class MassiveFinalizedArchiveScopeV2:
    start_session_date: str
    end_session_date: str
    exchange: str
    expected_month_prefixes: tuple[str, ...]
    captured_listing_receipts: tuple[str, ...]
    expected_source_session_dates: tuple[str, ...]
    expected_source_object_keys: tuple[str, ...]
    observed_in_scope_object_keys: tuple[str, ...]
    observed_out_of_scope_object_keys: tuple[str, ...]
    missing_source_objects: tuple[str, ...]
    session_authority_receipt_sha256: str
    scope_spec_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SCHEMA
            or self.exchange != "XNYS"
            or date.fromisoformat(self.start_session_date)
            > date.fromisoformat(self.end_session_date)
        ):
            raise MassiveFinalizedArchiveScopeError("archive scope v2 identity differs")
        required = (
            self.expected_month_prefixes,
            self.captured_listing_receipts,
            self.expected_source_session_dates,
            self.expected_source_object_keys,
            self.observed_in_scope_object_keys,
        )
        optional = (
            self.observed_out_of_scope_object_keys,
            self.missing_source_objects,
        )
        if any(
            not values or values != tuple(sorted(set(values))) for values in required
        ) or any(values != tuple(sorted(set(values))) for values in optional):
            raise MassiveFinalizedArchiveScopeError(
                "archive scope v2 inventories are not canonical"
            )
        expected_keys = tuple(
            canonical_massive_trade_object_key(value)
            for value in self.expected_source_session_dates
        )
        if self.expected_source_object_keys != expected_keys:
            raise MassiveFinalizedArchiveScopeError(
                "archive scope v2 object inventory differs"
            )
        if set(self.observed_in_scope_object_keys) - set(expected_keys):
            raise MassiveFinalizedArchiveScopeError(
                "archive scope v2 in-scope inventory escaped the range"
            )
        if set(self.observed_out_of_scope_object_keys) & set(expected_keys):
            raise MassiveFinalizedArchiveScopeError(
                "archive scope v2 inventories overlap"
            )
        if self.missing_source_objects != tuple(
            sorted(set(expected_keys) - set(self.observed_in_scope_object_keys))
        ):
            raise MassiveFinalizedArchiveScopeError(
                "archive scope v2 missing inventory differs"
            )
        for name in (
            "session_authority_receipt_sha256",
            "scope_spec_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for value in self.captured_listing_receipts:
            _digest("captured listing receipt", value)
        if self.scope_spec_receipt_sha256 != MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256:
            raise MassiveFinalizedArchiveScopeError(
                "archive scope v2 specification drifted"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedArchiveScopeError("archive scope v2 receipt differs")

    @property
    def qualification_complete(self) -> bool:
        return not self.missing_source_objects


def build_massive_finalized_archive_scope_v1(
    *,
    session_authority: MassiveSessionAuthority,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    start_session_date: str,
    end_session_date: str,
    exchange: str = "XNYS",
) -> MassiveFinalizedArchiveScopeV1:
    session_authority.validate()
    start = date.fromisoformat(start_session_date)
    end = date.fromisoformat(end_session_date)
    if start > end:
        raise MassiveFinalizedArchiveScopeError("archive scope dates are inverted")
    sessions = tuple(
        row
        for row in session_authority.sessions
        if row.exchange == exchange
        and start <= date.fromisoformat(row.session_date) <= end
    )
    if not sessions:
        raise MassiveFinalizedArchiveScopeError(
            "archive scope has no exchange sessions"
        )
    expected_dates = tuple(row.session_date for row in sessions)
    expected_keys = tuple(
        canonical_massive_trade_object_key(value) for value in expected_dates
    )
    expected_prefixes = tuple(
        sorted(
            {
                f"us_stocks_sip/trades_v1/{value[:4]}/{value[5:7]}/"
                for value in expected_dates
            }
        )
    )
    listings = tuple(captured_listings)
    if not listings:
        raise MassiveFinalizedArchiveScopeError("archive listing inventory is absent")
    for listing in listings:
        listing.validate()
    prefixes = tuple(
        sorted(listing.acquisition_evidence.prefix for listing in listings)
    )
    if prefixes != expected_prefixes:
        raise MassiveFinalizedArchiveScopeError(
            "archive listing months do not exhaust the requested scope"
        )
    observed_keys = {
        entry.source_object_key
        for listing in listings
        for entry in listing.committed_listing.entries
    }
    missing = tuple(sorted(set(expected_keys) - observed_keys))
    unexpected = tuple(sorted(observed_keys - set(expected_keys)))
    if unexpected:
        raise MassiveFinalizedArchiveScopeError(
            "archive listings contain objects outside the requested scope"
        )
    body = {
        "schema": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SCHEMA,
        "start_session_date": start_session_date,
        "end_session_date": end_session_date,
        "exchange": exchange,
        "expected_month_prefixes": expected_prefixes,
        "captured_listing_receipts": tuple(
            sorted(listing.acquisition_evidence.receipt_sha256 for listing in listings)
        ),
        "expected_source_session_dates": expected_dates,
        "expected_source_object_keys": expected_keys,
        "missing_source_objects": missing,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "scope_spec_receipt_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SPEC_SHA256,
    }
    result = MassiveFinalizedArchiveScopeV1(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


def build_massive_finalized_archive_scope_v2(
    *,
    session_authority: MassiveSessionAuthority,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    start_session_date: str,
    end_session_date: str,
    exchange: str = "XNYS",
) -> MassiveFinalizedArchiveScopeV2:
    """Build a range scope while accepting normal full-month listing payloads."""

    session_authority.validate()
    start = date.fromisoformat(start_session_date)
    end = date.fromisoformat(end_session_date)
    if start > end:
        raise MassiveFinalizedArchiveScopeError("archive scope v2 dates are inverted")
    sessions = tuple(
        row
        for row in session_authority.sessions
        if row.exchange == exchange
        and start <= date.fromisoformat(row.session_date) <= end
    )
    if not sessions:
        raise MassiveFinalizedArchiveScopeError("archive scope v2 has no sessions")
    expected_dates = tuple(row.session_date for row in sessions)
    expected_keys = tuple(
        canonical_massive_trade_object_key(value) for value in expected_dates
    )
    expected_prefixes = tuple(
        sorted(
            {
                f"us_stocks_sip/trades_v1/{value[:4]}/{value[5:7]}/"
                for value in expected_dates
            }
        )
    )
    listings = tuple(captured_listings)
    if not listings:
        raise MassiveFinalizedArchiveScopeError("archive scope v2 listings are absent")
    for listing in listings:
        listing.validate()
    prefixes = tuple(sorted(listing.acquisition_evidence.prefix for listing in listings))
    if prefixes != expected_prefixes:
        raise MassiveFinalizedArchiveScopeError(
            "archive scope v2 listing months are incomplete"
        )
    observed = {
        entry.source_object_key
        for listing in listings
        for entry in listing.committed_listing.entries
    }
    if any(not any(key.startswith(prefix) for prefix in expected_prefixes) for key in observed):
        raise MassiveFinalizedArchiveScopeError(
            "archive scope v2 listing contains a wrong-prefix object"
        )
    expected_set = set(expected_keys)
    body = {
        "schema": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SCHEMA,
        "start_session_date": start_session_date,
        "end_session_date": end_session_date,
        "exchange": exchange,
        "expected_month_prefixes": expected_prefixes,
        "captured_listing_receipts": tuple(
            sorted(listing.acquisition_evidence.receipt_sha256 for listing in listings)
        ),
        "expected_source_session_dates": expected_dates,
        "expected_source_object_keys": expected_keys,
        "observed_in_scope_object_keys": tuple(sorted(observed & expected_set)),
        "observed_out_of_scope_object_keys": tuple(sorted(observed - expected_set)),
        "missing_source_objects": tuple(sorted(expected_set - observed)),
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "scope_spec_receipt_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256,
    }
    result = MassiveFinalizedArchiveScopeV2(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SPEC_SHA256",
    "MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256",
    "MassiveFinalizedArchiveScopeError",
    "MassiveFinalizedArchiveScopeV1",
    "MassiveFinalizedArchiveScopeV2",
    "build_massive_finalized_archive_scope_v1",
    "build_massive_finalized_archive_scope_v2",
]
