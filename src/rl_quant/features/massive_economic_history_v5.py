"""Origin-vintage, holding-vintage economic history for profitability P0."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, time
from io import BytesIO
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from rl_quant.alpha.contracts import (
    CorporateActionKind,
    CorporateActionRecord,
    TerminalEventKind,
)
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_economic_event_source_v5 import (
    MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5,
    MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SPEC_SHA256,
    MassiveEconomicEventAuthorityV5,
    MassiveOrderedEconomicEventV5,
    MassiveSourcedCashReturnV5,
    MassiveSourcedCorporateActionV5,
    MassiveSourcedTerminalEventV5,
    build_massive_economic_event_authority_v5,
    ordered_economic_event_order_v5,
    resolve_massive_economic_events_at_origin_v5,
)
from rl_quant.features.massive_session_panel_v1 import (
    MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
    MassiveSessionPanelArtifactV1,
    MassiveSessionPanelRowV1,
    validate_massive_session_panel_v1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_ECONOMIC_HISTORY_V5_SCHEMA = "rl-quant.massive-economic-history-at-origin-v5"
MASSIVE_ECONOMIC_HISTORY_V5_DATASET = "massive-economic-history-at-origin-v5"
MASSIVE_ECONOMIC_HISTORY_V5_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_HISTORY_V5_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "session_panel_spec": MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
        "economic_event_source_spec": MASSIVE_ECONOMIC_EVENT_SOURCE_V5_SPEC_SHA256,
        "origin": "source-session-plus-two-XNYS-sessions-at-12:30-America/New_York",
        "availability_vintage": (
            "select-latest-active-linear-revision-available-by-decision-origin"
        ),
        "replay": "admitted-events-replayed-at-economic-effective-time",
        "base": "one-share-at-first-valid-origin-security-mark",
        "pre_base": "all-events-at-or-before-base-order-permanently-ineligible",
        "holding_vintage": (
            "event-applies-only-to-lots-acquired-before-global-economic-order"
        ),
        "successor_vintage": "effective-time-and-global-economic-sequence",
        "event_id_role": "audit-identity-only-never-economic-order",
        "tie_policy": "order-derived-from-source-evidence-or-fails-closed",
        "revision_policy": "corrected-replaces-prior-cancelled-removes-logical-event",
        "base_global_economic_sequence": MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5,
        "missing_mark": "invalid-never-zero-imputation",
        "listing_age": "calendar-date-resolved-with-explicit-left-censor-indicator",
        "training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
)
MASSIVE_ECONOMIC_HISTORY_V5_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_HISTORY_V5_SCHEMA,
        "row_key": ("source_session_index", "security_id"),
    }
)
_NY = ZoneInfo("America/New_York")


class MassiveEconomicHistoryV5Error(ValueError):
    """Origin-vintage economic history or its source evidence differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicHistoryV5Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicHistoryV5Error(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class MassiveEconomicHistoryOriginV5:
    source_session_date: str
    decision_session_date: str
    decision_at_ms: int
    source_staleness_sessions: int
    session_authority_receipt_sha256: str
    decision_origin_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        try:
            source_date = datetime.fromisoformat(self.source_session_date).date()
            decision_date = datetime.fromisoformat(self.decision_session_date).date()
        except ValueError as exc:
            raise MassiveEconomicHistoryV5Error(
                "economic origin date is invalid"
            ) from exc
        if source_date >= decision_date or self.source_staleness_sessions != 2:
            raise MassiveEconomicHistoryV5Error("economic origin chronology drifted")
        _nonnegative_int("decision timestamp", self.decision_at_ms)
        for value in (
            self.session_authority_receipt_sha256,
            self.decision_origin_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("economic origin digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicHistoryV5Error("economic origin receipt differs")


def build_massive_economic_history_origin_v5(
    *,
    session_authority: MassiveSessionAuthority,
    source_session_date: str,
    decision_origin_receipt_sha256: str,
) -> MassiveEconomicHistoryOriginV5:
    """Derive the exact P0 decision two sessions after the source session."""

    session_authority.validate()
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    source_matches = tuple(
        index
        for index, session in enumerate(sessions)
        if session.session_date == source_session_date
    )
    if len(source_matches) != 1 or source_matches[0] + 2 >= len(sessions):
        raise MassiveEconomicHistoryV5Error(
            "source session lacks its exact two-session decision"
        )
    decision_session = sessions[source_matches[0] + 2]
    decision_at_ms = int(
        datetime.combine(
            datetime.fromisoformat(decision_session.session_date).date(),
            time(12, 30),
            tzinfo=_NY,
        ).timestamp()
        * 1000
    )
    if not (
        decision_session.regular_open_ns // 1_000_000
        <= decision_at_ms
        < decision_session.regular_close_ns // 1_000_000
    ):
        raise MassiveEconomicHistoryV5Error(
            "decision time lies outside its exchange session"
        )
    provisional = MassiveEconomicHistoryOriginV5(
        source_session_date=source_session_date,
        decision_session_date=decision_session.session_date,
        decision_at_ms=decision_at_ms,
        source_staleness_sessions=2,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        decision_origin_receipt_sha256=_digest(
            "decision origin receipt", decision_origin_receipt_sha256
        ),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveVintageHoldingV5:
    security_id: str
    shares: float
    acquired_effective_at_ms: int
    acquired_global_economic_sequence: int
    acquired_event_id: str

    def validate(self) -> None:
        if (
            not self.security_id
            or not self.acquired_event_id
            or not math.isfinite(self.shares)
            or self.shares <= 0.0
        ):
            raise MassiveEconomicHistoryV5Error("vintage holding is invalid")
        _nonnegative_int("holding acquisition time", self.acquired_effective_at_ms)
        sequence = _nonnegative_int(
            "holding acquisition sequence", self.acquired_global_economic_sequence
        )
        if sequence > MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5:
            raise MassiveEconomicHistoryV5Error(
                "holding acquisition sequence exceeds the base sentinel"
            )
        is_base_id = self.acquired_event_id.startswith("BASE:")
        if (sequence == MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5) != is_base_id:
            raise MassiveEconomicHistoryV5Error(
                "base acquisition identity and sequence differ"
            )

    @property
    def acquisition_order(self) -> tuple[int, int]:
        return (
            self.acquired_effective_at_ms,
            self.acquired_global_economic_sequence,
        )


@dataclass(frozen=True, slots=True)
class MassiveVintageEconomicPositionV5:
    holdings: tuple[MassiveVintageHoldingV5, ...]
    cash: float
    applied_event_ids: tuple[str, ...]
    excluded_event_ids: tuple[str, ...]

    def validate(self) -> None:
        keys: list[tuple[str, int, int, str]] = []
        for holding in self.holdings:
            holding.validate()
            keys.append(
                (
                    holding.security_id,
                    holding.acquired_effective_at_ms,
                    holding.acquired_global_economic_sequence,
                    holding.acquired_event_id,
                )
            )
        if keys != sorted(set(keys)):
            raise MassiveEconomicHistoryV5Error("vintage holdings are not canonical")
        if not math.isfinite(self.cash) or self.cash < 0.0:
            raise MassiveEconomicHistoryV5Error("vintage position cash is invalid")
        if (
            self.applied_event_ids != tuple(sorted(set(self.applied_event_ids)))
            or self.excluded_event_ids != tuple(sorted(set(self.excluded_event_ids)))
            or set(self.applied_event_ids) & set(self.excluded_event_ids)
        ):
            raise MassiveEconomicHistoryV5Error(
                "vintage event disposition is not canonical"
            )


@dataclass(frozen=True, slots=True)
class MassiveEconomicHistoryRowV5:
    source_session_index: int
    source_session_date: str
    security_id: str
    listed: bool
    listing_at_ms: int
    listing_calendar_date: str
    listing_session_ordinal: int
    listing_age_sessions: int
    listing_age_left_censored: bool
    economic_value: float
    economic_value_valid: bool
    terminal: bool
    position: MassiveVintageEconomicPositionV5 | None
    session_panel_row_receipt_sha256: str
    economic_authority_receipt_sha256: str
    origin_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _nonnegative_int("economic session index", self.source_session_index)
        _nonnegative_int("listing time", self.listing_at_ms)
        _nonnegative_int("listing session ordinal", self.listing_session_ordinal)
        _nonnegative_int("listing age", self.listing_age_sessions)
        try:
            datetime.fromisoformat(self.listing_calendar_date).date()
        except ValueError as exc:
            raise MassiveEconomicHistoryV5Error(
                "listing calendar date is invalid"
            ) from exc
        if self.listing_age_left_censored and self.listing_session_ordinal != 0:
            raise MassiveEconomicHistoryV5Error(
                "left-censored listing ordinal must use the calendar boundary"
            )
        if not self.source_session_date or not self.security_id:
            raise MassiveEconomicHistoryV5Error(
                "economic history row identity is absent"
            )
        if any(
            not isinstance(value, bool)
            for value in (
                self.listed,
                self.listing_age_left_censored,
                self.economic_value_valid,
                self.terminal,
            )
        ):
            raise MassiveEconomicHistoryV5Error("economic history flags are invalid")
        if not math.isfinite(self.economic_value) or self.economic_value < 0.0:
            raise MassiveEconomicHistoryV5Error("economic history value is invalid")
        if not self.economic_value_valid and self.economic_value != 0.0:
            raise MassiveEconomicHistoryV5Error(
                "invalid economic history value is not a zero placeholder"
            )
        if self.position is None:
            if self.economic_value_valid or self.terminal:
                raise MassiveEconomicHistoryV5Error(
                    "absent position carries an economic value"
                )
        else:
            self.position.validate()
            if self.terminal != (not self.position.holdings):
                raise MassiveEconomicHistoryV5Error(
                    "terminal state differs from vintage holdings"
                )
        for value in (
            self.session_panel_row_receipt_sha256,
            self.economic_authority_receipt_sha256,
            self.origin_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("economic history row digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicHistoryV5Error("economic history row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveEconomicHistoryAtOriginV5:
    origin: MassiveEconomicHistoryOriginV5
    session_panel_receipt_sha256: str
    economic_authority_receipt_sha256: str
    identity_authority_receipt_sha256: str
    session_authority_receipt_sha256: str
    admitted_event_inventory_sha256: str
    unavailable_event_inventory_sha256: str
    cancelled_logical_event_inventory_sha256: str
    excluded_event_inventory_sha256: str
    rows: tuple[MassiveEconomicHistoryRowV5, ...]
    row_count: int
    valid_row_count: int
    terminal_row_count: int
    row_inventory_sha256: str
    feature_spec_receipt_sha256: str
    feature_source_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_ECONOMIC_HISTORY_V5_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ECONOMIC_HISTORY_V5_SCHEMA:
            raise MassiveEconomicHistoryV5Error("economic history schema drifted")
        self.origin.validate()
        for value in (
            self.session_panel_receipt_sha256,
            self.economic_authority_receipt_sha256,
            self.identity_authority_receipt_sha256,
            self.session_authority_receipt_sha256,
            self.admitted_event_inventory_sha256,
            self.unavailable_event_inventory_sha256,
            self.cancelled_logical_event_inventory_sha256,
            self.excluded_event_inventory_sha256,
            self.row_inventory_sha256,
            self.feature_spec_receipt_sha256,
            self.feature_source_sha256,
            self.receipt_sha256,
        ):
            _digest("economic history artifact digest", value)
        if (
            self.feature_spec_receipt_sha256 != MASSIVE_ECONOMIC_HISTORY_V5_SPEC_SHA256
            or self.feature_source_sha256 != MASSIVE_ECONOMIC_HISTORY_V5_SOURCE_SHA256
            or self.origin.session_authority_receipt_sha256
            != self.session_authority_receipt_sha256
        ):
            raise MassiveEconomicHistoryV5Error(
                "economic history implementation or origin drifted"
            )
        keys = tuple((row.source_session_index, row.security_id) for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveEconomicHistoryV5Error(
                "economic history rows are not canonical"
            )
        for row in self.rows:
            row.validate()
            if (
                row.origin_receipt_sha256 != self.origin.receipt_sha256
                or row.economic_authority_receipt_sha256
                != self.economic_authority_receipt_sha256
                or row.source_session_date > self.origin.source_session_date
            ):
                raise MassiveEconomicHistoryV5Error(
                    "economic history row authority differs"
                )
        excluded = tuple(
            sorted(
                (row.security_id, event_id)
                for row in self.rows
                if row.position is not None
                for event_id in row.position.excluded_event_ids
            )
        )
        if (
            self.row_count != len(self.rows)
            or self.valid_row_count
            != sum(row.economic_value_valid for row in self.rows)
            or self.terminal_row_count != sum(row.terminal for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.excluded_event_inventory_sha256 != semantic_sha256(excluded)
        ):
            raise MassiveEconomicHistoryV5Error(
                "economic history counts or inventories differ"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_ECONOMIC_HISTORY_V5_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_HISTORY_V5_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveEconomicHistoryV5Error(
                "economic history source contract differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicHistoryV5Error(
                "economic history artifact receipt differs"
            )


EconomicSourceEventV5 = MassiveOrderedEconomicEventV5


def _event_identity(row: EconomicSourceEventV5) -> str:
    return row.source_event.logical_event_key


def _event_order(row: EconomicSourceEventV5) -> tuple[int, int]:
    return ordered_economic_event_order_v5(row)


def _normalized_position(
    *,
    holdings: Sequence[MassiveVintageHoldingV5],
    cash: float,
    applied: Sequence[str],
    excluded: Sequence[str],
) -> MassiveVintageEconomicPositionV5:
    grouped: dict[tuple[str, int, int, str], float] = {}
    for holding in holdings:
        key = (
            holding.security_id,
            holding.acquired_effective_at_ms,
            holding.acquired_global_economic_sequence,
            holding.acquired_event_id,
        )
        grouped[key] = grouped.get(key, 0.0) + holding.shares
    position = MassiveVintageEconomicPositionV5(
        holdings=tuple(
            MassiveVintageHoldingV5(
                security_id=key[0],
                acquired_effective_at_ms=key[1],
                acquired_global_economic_sequence=key[2],
                acquired_event_id=key[3],
                shares=shares,
            )
            for key, shares in sorted(grouped.items())
            if shares > 1e-15
        ),
        cash=float(cash),
        applied_event_ids=tuple(sorted(set(applied))),
        excluded_event_ids=tuple(sorted(set(excluded))),
    )
    position.validate()
    return position


def _exclude_event(
    position: MassiveVintageEconomicPositionV5, event_id: str
) -> MassiveVintageEconomicPositionV5:
    return _normalized_position(
        holdings=position.holdings,
        cash=position.cash,
        applied=position.applied_event_ids,
        excluded=(*position.excluded_event_ids, event_id),
    )


def _apply_cash_return(
    position: MassiveVintageEconomicPositionV5,
    row: MassiveOrderedEconomicEventV5,
) -> MassiveVintageEconomicPositionV5:
    source_event = row.source_event
    if not isinstance(source_event, MassiveSourcedCashReturnV5):
        raise MassiveEconomicHistoryV5Error("ordered cash event has the wrong role")
    logical_event_key = source_event.logical_event_key
    return _normalized_position(
        holdings=position.holdings,
        cash=position.cash * (1.0 + source_event.cash_return.one_step_return),
        applied=(*position.applied_event_ids, logical_event_key),
        excluded=position.excluded_event_ids,
    )


def _apply_security_event(
    position: MassiveVintageEconomicPositionV5,
    row: MassiveOrderedEconomicEventV5,
) -> MassiveVintageEconomicPositionV5:
    source_event = row.source_event
    if not isinstance(
        source_event,
        (MassiveSourcedCorporateActionV5, MassiveSourcedTerminalEventV5),
    ):
        raise MassiveEconomicHistoryV5Error("ordered security event has the wrong role")
    event = source_event.event
    event_id = source_event.logical_event_key
    order = _event_order(row)
    eligible = tuple(
        holding
        for holding in position.holdings
        if holding.security_id == event.security_id
        and holding.acquisition_order < order
    )
    if not eligible:
        return _exclude_event(position, event_id)
    eligible_keys = {
        (
            holding.security_id,
            holding.acquired_effective_at_ms,
            holding.acquired_global_economic_sequence,
            holding.acquired_event_id,
        )
        for holding in eligible
    }
    holdings: list[MassiveVintageHoldingV5] = [
        holding
        for holding in position.holdings
        if (
            holding.security_id,
            holding.acquired_effective_at_ms,
            holding.acquired_global_economic_sequence,
            holding.acquired_event_id,
        )
        not in eligible_keys
    ]
    cash = position.cash
    total_eligible = sum(holding.shares for holding in eligible)

    if isinstance(source_event, MassiveSourcedTerminalEventV5):
        if event.kind is TerminalEventKind.MERGER_STOCK:
            assert event.successor_security_id is not None
            holdings.append(
                MassiveVintageHoldingV5(
                    security_id=event.successor_security_id,
                    shares=total_eligible * event.successor_ratio,
                    acquired_effective_at_ms=event.effective_at_ms,
                    acquired_global_economic_sequence=(
                        row.order_evidence.derived_global_economic_sequence
                    ),
                    acquired_event_id=event_id,
                )
            )
            cash += total_eligible * event.cash_per_share
        elif event.kind is not TerminalEventKind.WORTHLESS:
            cash += total_eligible * event.cash_per_share
    else:
        corporate_event = cast(CorporateActionRecord, event)
        return _apply_corporate_event(
            position=position,
            row=row,
            event=corporate_event,
            event_id=event_id,
            eligible=eligible,
            holdings=holdings,
            cash=cash,
            total_eligible=total_eligible,
        )
    return _normalized_position(
        holdings=holdings,
        cash=cash,
        applied=(*position.applied_event_ids, event_id),
        excluded=position.excluded_event_ids,
    )


def _apply_corporate_event(
    *,
    position: MassiveVintageEconomicPositionV5,
    row: MassiveOrderedEconomicEventV5,
    event: CorporateActionRecord,
    event_id: str,
    eligible: tuple[MassiveVintageHoldingV5, ...],
    holdings: list[MassiveVintageHoldingV5],
    cash: float,
    total_eligible: float,
) -> MassiveVintageEconomicPositionV5:
    if event.kind in {CorporateActionKind.SPLIT, CorporateActionKind.REVERSE_SPLIT}:
        holdings.extend(
            replace(holding, shares=holding.shares * event.share_ratio)
            for holding in eligible
        )
    elif event.kind in {
        CorporateActionKind.CASH_DIVIDEND,
        CorporateActionKind.SPECIAL_DIVIDEND,
        CorporateActionKind.RETURN_OF_CAPITAL,
    }:
        holdings.extend(eligible)
        cash += total_eligible * event.cash_per_share
    elif event.kind is CorporateActionKind.SPIN_OFF:
        assert event.successor_security_id is not None
        holdings.extend(eligible)
        holdings.append(
            MassiveVintageHoldingV5(
                security_id=event.successor_security_id,
                shares=total_eligible * event.successor_ratio,
                acquired_effective_at_ms=event.effective_at_ms,
                acquired_global_economic_sequence=(
                    row.order_evidence.derived_global_economic_sequence
                ),
                acquired_event_id=event_id,
            )
        )
        cash += total_eligible * event.cash_per_share
    elif event.kind is CorporateActionKind.MERGER_STOCK:
        assert event.successor_security_id is not None
        for holding in eligible:
            unaffected = holding.shares * (1.0 - event.affected_fraction)
            if unaffected > 0.0:
                holdings.append(replace(holding, shares=unaffected))
        affected = total_eligible * event.affected_fraction
        holdings.append(
            MassiveVintageHoldingV5(
                security_id=event.successor_security_id,
                shares=affected * event.successor_ratio,
                acquired_effective_at_ms=event.effective_at_ms,
                acquired_global_economic_sequence=(
                    row.order_evidence.derived_global_economic_sequence
                ),
                acquired_event_id=event_id,
            )
        )
        cash += affected * event.cash_per_share
    elif event.kind in {
        CorporateActionKind.MERGER_CASH,
        CorporateActionKind.TENDER_OFFER,
    }:
        for holding in eligible:
            unaffected = holding.shares * (1.0 - event.affected_fraction)
            if unaffected > 0.0:
                holdings.append(replace(holding, shares=unaffected))
        cash += total_eligible * event.affected_fraction * event.cash_per_share
    elif event.kind is CorporateActionKind.RIGHTS_DISTRIBUTION:
        holdings.extend(eligible)
        cash += total_eligible * event.cash_per_share
        if event.successor_ratio > 0.0:
            assert event.successor_security_id is not None
            holdings.append(
                MassiveVintageHoldingV5(
                    security_id=event.successor_security_id,
                    shares=total_eligible * event.successor_ratio,
                    acquired_effective_at_ms=event.effective_at_ms,
                    acquired_global_economic_sequence=(
                        row.order_evidence.derived_global_economic_sequence
                    ),
                    acquired_event_id=event_id,
                )
            )
    elif event.kind is CorporateActionKind.TICKER_EXCHANGE_CHANGE:
        holdings.extend(eligible)
    else:
        raise MassiveEconomicHistoryV5Error("economic event kind is unsupported")
    return _normalized_position(
        holdings=holdings,
        cash=cash,
        applied=(*position.applied_event_ids, event_id),
        excluded=position.excluded_event_ids,
    )


def _mark_position(
    position: MassiveVintageEconomicPositionV5, marks: Mapping[str, float]
) -> float:
    value = position.cash
    for holding in position.holdings:
        if holding.security_id not in marks:
            raise MassiveEconomicHistoryV5Error("held successor has no economic mark")
        mark = float(marks[holding.security_id])
        if not math.isfinite(mark) or mark < 0.0:
            raise MassiveEconomicHistoryV5Error("economic mark is invalid")
        value += holding.shares * mark
    if not math.isfinite(value) or value < 0.0:
        raise MassiveEconomicHistoryV5Error("marked economic value is invalid")
    return value


def _listing_coordinates(
    *,
    identity_authority: PITSecurityUniverseAuthority,
    session_authority: MassiveSessionAuthority,
) -> dict[str, tuple[int, str, int, bool]]:
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    if not sessions:
        raise MassiveEconomicHistoryV5Error("XNYS session authority is empty")
    session_dates = tuple(
        datetime.fromisoformat(session.session_date).date() for session in sessions
    )
    result: dict[str, tuple[int, str, int, bool]] = {}
    for master in identity_authority.security_master:
        utc_listing = datetime.fromtimestamp(master.listing_at_ms / 1000, tz=UTC)
        local_listing = utc_listing.astimezone(_NY)
        listing_date = (
            utc_listing.date()
            if utc_listing.time() == time(0, 0)
            else local_listing.date()
        )
        if listing_date < session_dates[0]:
            result[master.security_id] = (
                master.listing_at_ms,
                listing_date.isoformat(),
                0,
                True,
            )
            continue
        exact_date = tuple(
            index
            for index, session_date in enumerate(session_dates)
            if session_date == listing_date
        )
        matches = tuple(
            index
            for index, session in enumerate(sessions)
            if session.regular_close_ns // 1_000_000 >= master.listing_at_ms
        )
        if exact_date:
            listing_ordinal = exact_date[0]
        elif matches:
            listing_ordinal = matches[0]
        else:
            raise MassiveEconomicHistoryV5Error(
                "listing lies beyond the session authority"
            )
        result[master.security_id] = (
            master.listing_at_ms,
            listing_date.isoformat(),
            listing_ordinal,
            False,
        )
    return result


def _validate_origin(
    *,
    origin: MassiveEconomicHistoryOriginV5,
    session_authority: MassiveSessionAuthority,
) -> None:
    expected = build_massive_economic_history_origin_v5(
        session_authority=session_authority,
        source_session_date=origin.source_session_date,
        decision_origin_receipt_sha256=origin.decision_origin_receipt_sha256,
    )
    if origin != expected:
        raise MassiveEconomicHistoryV5Error(
            "economic history origin was not authority-derived"
        )


def build_massive_economic_history_rows_v5(
    *,
    panel_rows: Sequence[MassiveSessionPanelRowV1],
    economic_authority: MassiveEconomicEventAuthorityV5,
    identity_authority: PITSecurityUniverseAuthority,
    session_authority: MassiveSessionAuthority,
    origin: MassiveEconomicHistoryOriginV5,
) -> tuple[
    tuple[MassiveEconomicHistoryRowV5, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Replay one origin-vintage history on exact session coordinates."""

    economic_authority.validate()
    identity_authority.validate()
    session_authority.validate()
    origin.validate()
    _validate_origin(origin=origin, session_authority=session_authority)
    if (
        economic_authority.identity_authority_receipt_sha256
        != identity_authority.receipt_sha256
    ):
        raise MassiveEconomicHistoryV5Error("economic and identity authorities differ")
    rows = tuple(
        row
        for row in panel_rows
        if row.source_session_date <= origin.source_session_date
    )
    for row in rows:
        row.validate()
    keys = tuple((row.source_session_index, row.security_id) for row in rows)
    if not keys or keys != tuple(sorted(set(keys))):
        raise MassiveEconomicHistoryV5Error("economic panel rows are not canonical")
    security_ids = tuple(sorted({row.security_id for row in rows}))
    if security_ids != economic_authority.security_ids:
        raise MassiveEconomicHistoryV5Error(
            "economic and panel security inventories differ"
        )
    by_index: dict[int, dict[str, MassiveSessionPanelRowV1]] = {}
    for row in rows:
        by_index.setdefault(row.source_session_index, {})[row.security_id] = row
    indices = tuple(by_index)
    if indices != tuple(range(indices[0], indices[-1] + 1)) or any(
        set(group) != set(security_ids) for group in by_index.values()
    ):
        raise MassiveEconomicHistoryV5Error(
            "economic panel is not an exact session rectangle"
        )
    if {row.source_session_date for row in by_index[indices[-1]].values()} != {
        origin.source_session_date
    }:
        raise MassiveEconomicHistoryV5Error(
            "economic panel does not terminate at the origin source session"
        )
    listing_coordinates = _listing_coordinates(
        identity_authority=identity_authority,
        session_authority=session_authority,
    )
    admitted, unavailable_revision_ids, cancelled_logical_keys = (
        resolve_massive_economic_events_at_origin_v5(
            authority=economic_authority,
            decision_at_ms=origin.decision_at_ms,
        )
    )
    admitted_ids = tuple(_event_identity(row) for row in admitted)
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    output: list[MassiveEconomicHistoryRowV5] = []
    session_ordinal = {
        session.session_date: index
        for index, session in enumerate(
            row for row in session_authority.sessions if row.exchange == "XNYS"
        )
    }
    for origin_security_id in security_ids:
        position: MassiveVintageEconomicPositionV5 | None = None
        for session_index, group in by_index.items():
            panel = group[origin_security_id]
            coordinates = {
                (row.source_session_date, row.regular_open_ns, row.regular_close_ns)
                for row in group.values()
            }
            if len(coordinates) != 1:
                raise MassiveEconomicHistoryV5Error(
                    "economic panel session coordinates differ"
                )
            _, _, regular_close_ns = next(iter(coordinates))
            close_ms = regular_close_ns // 1_000_000
            marks = {
                security_id: candidate.bars_values[close_index]
                for security_id, candidate in group.items()
                if candidate.listed and candidate.bars_valid[close_index]
            }
            if position is None and panel.listed and origin_security_id in marks:
                base_event_id = f"BASE:{origin_security_id}:{panel.source_session_date}"
                base_order = (
                    close_ms,
                    MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5,
                )
                pre_base = tuple(
                    _event_identity(event)
                    for event in admitted
                    if _event_order(event) <= base_order
                )
                position = _normalized_position(
                    holdings=(
                        MassiveVintageHoldingV5(
                            security_id=origin_security_id,
                            shares=1.0,
                            acquired_effective_at_ms=close_ms,
                            acquired_global_economic_sequence=MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5,
                            acquired_event_id=base_event_id,
                        ),
                    ),
                    cash=0.0,
                    applied=(),
                    excluded=pre_base,
                )
            if position is not None:
                for event in admitted:
                    event_id = _event_identity(event)
                    if (
                        event_id in position.applied_event_ids
                        or event_id in position.excluded_event_ids
                        or _event_order(event)[0] > close_ms
                    ):
                        continue
                    if isinstance(event.source_event, MassiveSourcedCashReturnV5):
                        position = _apply_cash_return(position, event)
                    else:
                        position = _apply_security_event(position, event)
            value = 0.0
            valid = False
            if position is not None:
                try:
                    value = _mark_position(position, marks)
                    valid = True
                except MassiveEconomicHistoryV5Error:
                    value = 0.0
            (
                listing_at_ms,
                listing_calendar_date,
                listing_ordinal,
                listing_age_left_censored,
            ) = listing_coordinates[origin_security_id]
            current_ordinal = session_ordinal[panel.source_session_date]
            listing_age = max(0, current_ordinal - listing_ordinal + 1)
            provisional = MassiveEconomicHistoryRowV5(
                source_session_index=session_index,
                source_session_date=panel.source_session_date,
                security_id=origin_security_id,
                listed=panel.listed,
                listing_at_ms=listing_at_ms,
                listing_calendar_date=listing_calendar_date,
                listing_session_ordinal=listing_ordinal,
                listing_age_sessions=listing_age,
                listing_age_left_censored=listing_age_left_censored,
                economic_value=value,
                economic_value_valid=valid,
                terminal=position is not None and not position.holdings,
                position=position,
                session_panel_row_receipt_sha256=panel.receipt_sha256,
                economic_authority_receipt_sha256=economic_authority.receipt_sha256,
                origin_receipt_sha256=origin.receipt_sha256,
                receipt_sha256="0" * 64,
            )
            result = replace(
                provisional,
                receipt_sha256=semantic_sha256(provisional.unsigned()),
            )
            result.validate()
            output.append(result)
    return (
        tuple(
            sorted(output, key=lambda row: (row.source_session_index, row.security_id))
        ),
        admitted_ids,
        unavailable_revision_ids,
        cancelled_logical_keys,
    )


def _payload(artifact: MassiveEconomicHistoryAtOriginV5) -> dict[str, object]:
    return {
        "schema": artifact.schema,
        "origin": asdict(artifact.origin),
        "session_panel_receipt_sha256": artifact.session_panel_receipt_sha256,
        "economic_authority_receipt_sha256": artifact.economic_authority_receipt_sha256,
        "identity_authority_receipt_sha256": artifact.identity_authority_receipt_sha256,
        "session_authority_receipt_sha256": artifact.session_authority_receipt_sha256,
        "admitted_event_inventory_sha256": artifact.admitted_event_inventory_sha256,
        "unavailable_event_inventory_sha256": artifact.unavailable_event_inventory_sha256,
        "cancelled_logical_event_inventory_sha256": (
            artifact.cancelled_logical_event_inventory_sha256
        ),
        "excluded_event_inventory_sha256": artifact.excluded_event_inventory_sha256,
        "rows": tuple(asdict(row) for row in artifact.rows),
        "row_count": artifact.row_count,
        "valid_row_count": artifact.valid_row_count,
        "terminal_row_count": artifact.terminal_row_count,
        "row_inventory_sha256": artifact.row_inventory_sha256,
        "feature_spec_receipt_sha256": artifact.feature_spec_receipt_sha256,
        "feature_source_sha256": artifact.feature_source_sha256,
    }


def materialize_massive_economic_history_v5(
    *,
    session_panel_root: str | Path,
    economic_source_root: str | Path,
    output_root: str | Path,
    session_panel: MassiveSessionPanelArtifactV1,
    economic_authority: MassiveEconomicEventAuthorityV5,
    identity_authority: PITSecurityUniverseAuthority,
    session_authority: MassiveSessionAuthority,
    origin: MassiveEconomicHistoryOriginV5,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveEconomicHistoryAtOriginV5:
    """Reparse every source and publish one origin-specific economic history."""

    validate_massive_session_panel_v1(root=session_panel_root, artifact=session_panel)
    if (
        session_panel.identity_authority_receipt_sha256
        != identity_authority.receipt_sha256
        or session_panel.session_authority_receipt_sha256
        != session_authority.receipt_sha256
    ):
        raise MassiveEconomicHistoryV5Error(
            "session panel identity or calendar authority differs"
        )
    expected_authority = build_massive_economic_event_authority_v5(
        root=economic_source_root,
        loaded_sources=tuple(
            source.loaded_source for source in economic_authority.sources
        ),
        identity_authority=identity_authority,
    )
    if economic_authority != expected_authority:
        raise MassiveEconomicHistoryV5Error(
            "economic authority was not derived from committed source bytes"
        )
    rows, admitted, unavailable, cancelled = build_massive_economic_history_rows_v5(
        panel_rows=session_panel.rows,
        economic_authority=economic_authority,
        identity_authority=identity_authority,
        session_authority=session_authority,
        origin=origin,
    )
    excluded = tuple(
        sorted(
            (row.security_id, event_id)
            for row in rows
            if row.position is not None
            for event_id in row.position.excluded_event_ids
        )
    )
    relative = (
        "massive-profitability-p0/economic-history-v5/"
        f"{origin.source_session_date}-{origin.decision_session_date}.json"
    )
    placeholder = MassiveEconomicHistoryAtOriginV5(
        origin=origin,
        session_panel_receipt_sha256=session_panel.receipt_sha256,
        economic_authority_receipt_sha256=economic_authority.receipt_sha256,
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        admitted_event_inventory_sha256=semantic_sha256(admitted),
        unavailable_event_inventory_sha256=semantic_sha256(unavailable),
        cancelled_logical_event_inventory_sha256=semantic_sha256(cancelled),
        excluded_event_inventory_sha256=semantic_sha256(excluded),
        rows=rows,
        row_count=len(rows),
        valid_row_count=sum(row.economic_value_valid for row in rows),
        terminal_row_count=sum(row.terminal for row in rows),
        row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        feature_spec_receipt_sha256=MASSIVE_ECONOMIC_HISTORY_V5_SPEC_SHA256,
        feature_source_sha256=MASSIVE_ECONOMIC_HISTORY_V5_SOURCE_SHA256,
        loaded_source=session_panel.loaded_source,
        receipt_sha256="0" * 64,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(placeholder))),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ECONOMIC_HISTORY_V5_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_ECONOMIC_HISTORY_V5_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=output_root,
        relative_payload_path=relative,
        verified_at_ms=published_at_ms,
    )
    provisional = replace(placeholder, loaded_source=loaded)
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    validate_massive_economic_history_v5(root=output_root, artifact=result)
    return result


def validate_massive_economic_history_v5(
    *, root: str | Path, artifact: MassiveEconomicHistoryAtOriginV5
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicHistoryV5Error(
            "economic history source is not JSON"
        ) from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(artifact)
    ):
        raise MassiveEconomicHistoryV5Error("economic history bytes differ")


__all__ = [
    "MASSIVE_ECONOMIC_HISTORY_V5_DATASET",
    "MASSIVE_ECONOMIC_HISTORY_V5_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_HISTORY_V5_SPEC_SHA256",
    "MassiveEconomicHistoryAtOriginV5",
    "MassiveEconomicHistoryOriginV5",
    "MassiveEconomicHistoryRowV5",
    "MassiveEconomicHistoryV5Error",
    "MassiveVintageEconomicPositionV5",
    "MassiveVintageHoldingV5",
    "build_massive_economic_history_origin_v5",
    "build_massive_economic_history_rows_v5",
    "materialize_massive_economic_history_v5",
    "validate_massive_economic_history_v5",
]
