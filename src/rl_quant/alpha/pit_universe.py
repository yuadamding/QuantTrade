"""Permanent-security and historical point-in-time universe authority.

This module materializes the identity and universe files that must exist before
any PIT alpha training tensor may be constructed. Polygon staging contributes
only byte-bound market observations; it is not an identity, membership, or
economic authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

from rl_quant.alpha.contracts import (
    MembershipEvent,
    PITAlphaDataError,
    SecurityMasterRecord,
    TickerHistoryRecord,
    UniverseRule,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)

PIT_SECURITY_UNIVERSE_SCHEMA = "rl-quant.pit-security-universe-v1"
PIT_UNIVERSE_RULE_SCHEMA = "rl-quant.pit-universe-rule-v1"
PIT_UNIVERSE_BUNDLE_SCHEMA = "rl-quant.pit-universe-bundle-v1"
PIT_POLYGON_COVERAGE_SCHEMA = "rl-quant.pit-polygon-coverage-v1"

SECURITY_MASTER_FILE = "security_master.parquet"
TICKER_HISTORY_FILE = "ticker_history.parquet"
LISTING_EVENTS_FILE = "listing_events.parquet"
DELISTING_EVENTS_FILE = "delisting_events.parquet"
MEMBERSHIP_EVENTS_FILE = "membership_events.parquet"
UNIVERSE_RANK_INPUTS_FILE = "universe_rank_inputs.parquet"
UNIVERSE_RULE_FILE = "universe_rule.json"
AUTHORITY_FILE = "identity_universe_authority.json"

_PARQUET_COMPRESSION = "zstd"
_PARQUET_COMPRESSION_LEVEL = 9
_PARQUET_ROW_GROUP_SIZE = 65_536
_SAFE_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,15}$")


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PITAlphaDataError(f"{name} must be a canonical nonempty string")
    return value


def _timestamp(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PITAlphaDataError(f"{name} must be a nonnegative integer timestamp")
    return value


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PITAlphaDataError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PITAlphaDataError(f"{name} must be numeric")
    observed = float(value)
    if not math.isfinite(observed) or (minimum is not None and observed < minimum):
        raise PITAlphaDataError(f"{name} is outside its finite domain")
    return observed


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PITAlphaDataError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _optional_text(name: str, value: str | None) -> None:
    if value is not None:
        _text(name, value)


def _ticker(value: object) -> str:
    observed = _text("ticker", value)
    if _SAFE_TICKER.fullmatch(observed) is None:
        raise PITAlphaDataError("ticker is not canonical for the organized market data")
    return observed


def _iso_date(name: str, value: object) -> str:
    observed = _text(name, value)
    try:
        date.fromisoformat(observed)
    except ValueError as exc:
        raise PITAlphaDataError(f"{name} must be an ISO calendar date") from exc
    return observed


@dataclass(frozen=True, slots=True)
class SourcedSecurityMasterRecord:
    security_id: str
    issuer_id: str
    primary_exchange: str
    share_class: str
    security_type: str
    listing_at_ms: int
    delisting_at_ms: int | None
    successor_security_id: str | None
    corporate_action_chain_id: str | None
    identity_source_receipt_sha256: str

    def base(self) -> SecurityMasterRecord:
        return SecurityMasterRecord(
            security_id=self.security_id,
            issuer_id=self.issuer_id,
            primary_exchange=self.primary_exchange,
            share_class=self.share_class,
            security_type=self.security_type,
            listing_at_ms=self.listing_at_ms,
            delisting_at_ms=self.delisting_at_ms,
            successor_security_id=self.successor_security_id,
            corporate_action_chain_id=self.corporate_action_chain_id,
        )

    def validate(self) -> None:
        self.base().validate()
        _digest("identity source receipt", self.identity_source_receipt_sha256)


@dataclass(frozen=True, slots=True)
class SourcedTickerHistoryRecord:
    security_id: str
    ticker: str
    valid_from_ms: int
    valid_to_ms: int | None
    available_at_ms: int
    primary_exchange: str
    source_receipt_sha256: str

    def base(self) -> TickerHistoryRecord:
        return TickerHistoryRecord(
            security_id=self.security_id,
            ticker=self.ticker,
            valid_from_ms=self.valid_from_ms,
            valid_to_ms=self.valid_to_ms,
            available_at_ms=self.available_at_ms,
        )

    def validate(self) -> None:
        self.base().validate()
        _ticker(self.ticker)
        _text("ticker exchange", self.primary_exchange)
        _digest("ticker source receipt", self.source_receipt_sha256)


@dataclass(frozen=True, slots=True)
class ListingEventRecord:
    event_id: str
    security_id: str
    effective_at_ms: int
    available_at_ms: int
    primary_exchange: str
    ticker: str
    source_receipt_sha256: str

    def validate(self) -> None:
        for name in ("listing event ID", "security ID", "exchange", "ticker"):
            attribute = {
                "listing event ID": self.event_id,
                "security ID": self.security_id,
                "exchange": self.primary_exchange,
                "ticker": self.ticker,
            }[name]
            _text(name, attribute)
        _ticker(self.ticker)
        effective = _timestamp("listing effective time", self.effective_at_ms)
        available = _timestamp("listing availability time", self.available_at_ms)
        if available > effective:
            raise PITAlphaDataError("listing event was unavailable when effective")
        _digest("listing source receipt", self.source_receipt_sha256)


@dataclass(frozen=True, slots=True)
class DelistingEventRecord:
    event_id: str
    security_id: str
    effective_at_ms: int
    available_at_ms: int
    reason: str
    successor_security_id: str | None
    source_receipt_sha256: str

    def validate(self) -> None:
        _text("delisting event ID", self.event_id)
        _text("security ID", self.security_id)
        _text("delisting reason", self.reason)
        _optional_text("successor security ID", self.successor_security_id)
        effective = _timestamp("delisting effective time", self.effective_at_ms)
        available = _timestamp("delisting availability time", self.available_at_ms)
        if available > effective:
            raise PITAlphaDataError("delisting event was unavailable when effective")
        if self.successor_security_id == self.security_id:
            raise PITAlphaDataError("delisting successor must be a different security")
        _digest("delisting source receipt", self.source_receipt_sha256)


@dataclass(frozen=True, slots=True)
class PITUniverseRuleSpec:
    rule_id: str
    target_size: int
    ranking_metric: str
    ranking_lookback_sessions: int
    ranking_lag_sessions: int
    minimum_observed_sessions: int
    minimum_close_price: float
    minimum_average_dollar_volume: float
    eligible_security_types: tuple[str, ...]
    rebalance_frequency: str
    tie_breaker: str
    uses_future_survival: bool
    receipt_sha256: str
    schema: str = PIT_UNIVERSE_RULE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rule_id": self.rule_id,
            "target_size": self.target_size,
            "ranking_metric": self.ranking_metric,
            "ranking_lookback_sessions": self.ranking_lookback_sessions,
            "ranking_lag_sessions": self.ranking_lag_sessions,
            "minimum_observed_sessions": self.minimum_observed_sessions,
            "minimum_close_price": self.minimum_close_price,
            "minimum_average_dollar_volume": self.minimum_average_dollar_volume,
            "eligible_security_types": list(self.eligible_security_types),
            "rebalance_frequency": self.rebalance_frequency,
            "tie_breaker": self.tie_breaker,
            "uses_future_survival": self.uses_future_survival,
        }

    def validate(self) -> None:
        if self.schema != PIT_UNIVERSE_RULE_SCHEMA:
            raise PITAlphaDataError("PIT universe rule schema is unsupported")
        _text("universe rule ID", self.rule_id)
        _positive_int("target universe size", self.target_size)
        if self.ranking_metric != "trailing-mean-dollar-volume":
            raise PITAlphaDataError("universe ranking metric is unsupported")
        _positive_int("ranking lookback", self.ranking_lookback_sessions)
        if _positive_int("ranking lag", self.ranking_lag_sessions) < 1:
            raise PITAlphaDataError("universe rank inputs must end by t-1")
        observed = _positive_int(
            "minimum observed sessions", self.minimum_observed_sessions
        )
        if observed > self.ranking_lookback_sessions:
            raise PITAlphaDataError("minimum observations exceed the ranking lookback")
        _finite("minimum close price", self.minimum_close_price, minimum=0.0)
        _finite(
            "minimum average dollar volume",
            self.minimum_average_dollar_volume,
            minimum=0.0,
        )
        if (
            not self.eligible_security_types
            or tuple(sorted(set(self.eligible_security_types)))
            != self.eligible_security_types
        ):
            raise PITAlphaDataError("eligible security types must be sorted and unique")
        for security_type in self.eligible_security_types:
            _text("eligible security type", security_type)
        if self.rebalance_frequency not in {"daily", "weekly", "monthly"}:
            raise PITAlphaDataError("universe rebalance frequency is unsupported")
        if self.tie_breaker != "security-id-ascending":
            raise PITAlphaDataError("universe tie breaker must be deterministic")
        if not isinstance(self.uses_future_survival, bool):
            raise PITAlphaDataError("future-survival flag must be Boolean")
        if self.uses_future_survival:
            raise PITAlphaDataError("future survival cannot enter universe membership")
        _digest("universe rule receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise PITAlphaDataError("universe rule receipt differs from its payload")

    def base(self) -> UniverseRule:
        self.validate()
        return UniverseRule(
            rule_id=self.rule_id,
            rule_sha256=self.receipt_sha256,
            membership_mode="point-in-time-events",
            ranking_lookback_sessions=self.ranking_lookback_sessions,
            ranking_lag_sessions=self.ranking_lag_sessions,
            uses_future_survival=False,
        )

    @classmethod
    def build(
        cls,
        *,
        rule_id: str,
        target_size: int = 500,
        ranking_lookback_sessions: int = 63,
        ranking_lag_sessions: int = 1,
        minimum_observed_sessions: int = 50,
        minimum_close_price: float = 1.0,
        minimum_average_dollar_volume: float = 0.0,
        eligible_security_types: Sequence[str] = ("common-stock",),
        rebalance_frequency: str = "monthly",
    ) -> PITUniverseRuleSpec:
        body: dict[str, object] = {
            "schema": PIT_UNIVERSE_RULE_SCHEMA,
            "rule_id": rule_id,
            "target_size": target_size,
            "ranking_metric": "trailing-mean-dollar-volume",
            "ranking_lookback_sessions": ranking_lookback_sessions,
            "ranking_lag_sessions": ranking_lag_sessions,
            "minimum_observed_sessions": minimum_observed_sessions,
            "minimum_close_price": float(minimum_close_price),
            "minimum_average_dollar_volume": float(minimum_average_dollar_volume),
            "eligible_security_types": sorted(set(eligible_security_types)),
            "rebalance_frequency": rebalance_frequency,
            "tie_breaker": "security-id-ascending",
            "uses_future_survival": False,
        }
        value = cls(
            schema=PIT_UNIVERSE_RULE_SCHEMA,
            rule_id=rule_id,
            target_size=target_size,
            ranking_metric="trailing-mean-dollar-volume",
            ranking_lookback_sessions=ranking_lookback_sessions,
            ranking_lag_sessions=ranking_lag_sessions,
            minimum_observed_sessions=minimum_observed_sessions,
            minimum_close_price=float(minimum_close_price),
            minimum_average_dollar_volume=float(minimum_average_dollar_volume),
            eligible_security_types=tuple(sorted(set(eligible_security_types))),
            rebalance_frequency=rebalance_frequency,
            tie_breaker="security-id-ascending",
            uses_future_survival=False,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class UniverseRankInputRecord:
    security_id: str
    effective_at_ms: int
    effective_session_index: int
    available_at_ms: int
    observation_start_ms: int
    observation_end_ms: int
    observation_start_session_index: int
    observation_end_session_index: int
    observed_session_count: int
    average_dollar_volume: float | None
    close_price: float | None
    source_receipt_sha256: str

    def validate_for(self, rule: PITUniverseRuleSpec) -> None:
        rule.validate()
        _text("rank-input security ID", self.security_id)
        effective = _timestamp("rank effective time", self.effective_at_ms)
        available = _timestamp("rank-input availability time", self.available_at_ms)
        start = _timestamp("rank observation start", self.observation_start_ms)
        end = _timestamp("rank observation end", self.observation_end_ms)
        effective_index = _positive_int(
            "effective session index", self.effective_session_index, allow_zero=True
        )
        start_index = _positive_int(
            "observation start session index",
            self.observation_start_session_index,
            allow_zero=True,
        )
        end_index = _positive_int(
            "observation end session index",
            self.observation_end_session_index,
            allow_zero=True,
        )
        observed = _positive_int(
            "observed session count", self.observed_session_count, allow_zero=True
        )
        if start > end or end > available or available > effective:
            raise PITAlphaDataError("rank input violates causal timestamp ordering")
        if end_index - start_index + 1 != rule.ranking_lookback_sessions:
            raise PITAlphaDataError(
                "rank input window differs from the frozen lookback"
            )
        if end_index > effective_index - rule.ranking_lag_sessions:
            raise PITAlphaDataError("rank input includes t or future sessions")
        if observed > rule.ranking_lookback_sessions:
            raise PITAlphaDataError("observed sessions exceed the ranking window")
        if observed == 0:
            if self.average_dollar_volume is not None or self.close_price is not None:
                raise PITAlphaDataError("empty rank input cannot carry market values")
        else:
            _finite("average dollar volume", self.average_dollar_volume, minimum=0.0)
            _finite("rank-input close price", self.close_price, minimum=0.0)
        _digest("rank-input source receipt", self.source_receipt_sha256)


@dataclass(frozen=True, slots=True)
class HistoricalMembershipRecord:
    security_id: str
    effective_at_ms: int
    available_at_ms: int
    observation_end_ms: int
    is_member: bool
    universe_rank: int | None
    eligibility_reason: str
    rank_input_group_receipt_sha256: str
    universe_rule_receipt_sha256: str

    def base(self) -> MembershipEvent:
        return MembershipEvent(
            security_id=self.security_id,
            effective_at_ms=self.effective_at_ms,
            available_at_ms=self.available_at_ms,
            observation_end_ms=self.observation_end_ms,
            is_member=self.is_member,
            universe_rank=self.universe_rank,
        )

    def validate(self) -> None:
        self.base().validate()
        _text("membership eligibility reason", self.eligibility_reason)
        _digest("rank-input group receipt", self.rank_input_group_receipt_sha256)
        _digest("universe rule receipt", self.universe_rule_receipt_sha256)


def _rank_group_receipt(
    rule: PITUniverseRuleSpec, rows: Sequence[UniverseRankInputRecord]
) -> str:
    return semantic_sha256(
        {
            "rule_receipt_sha256": rule.receipt_sha256,
            "rank_inputs": [
                asdict(row) for row in sorted(rows, key=lambda value: value.security_id)
            ],
        }
    )


def _active_at(master: SourcedSecurityMasterRecord, effective_at_ms: int) -> bool:
    return master.listing_at_ms <= effective_at_ms and (
        master.delisting_at_ms is None or effective_at_ms < master.delisting_at_ms
    )


def build_historical_membership(
    *,
    rule: PITUniverseRuleSpec,
    security_master: Sequence[SourcedSecurityMasterRecord],
    listing_events: Sequence[ListingEventRecord],
    delisting_events: Sequence[DelistingEventRecord],
    rank_inputs: Sequence[UniverseRankInputRecord],
) -> tuple[HistoricalMembershipRecord, ...]:
    """Derive membership independently from complete causal candidate groups."""

    rule.validate()
    masters = {row.security_id: row for row in security_master}
    if len(masters) != len(security_master):
        raise PITAlphaDataError("security master contains duplicate permanent IDs")
    for master_row in masters.values():
        master_row.validate()
    listings = {row.security_id: row for row in listing_events}
    delistings = {row.security_id: row for row in delisting_events}
    if len(listings) != len(listing_events) or set(listings) != set(masters):
        raise PITAlphaDataError(
            "membership construction requires one listing per security"
        )
    expected_delistings = {
        row.security_id for row in masters.values() if row.delisting_at_ms is not None
    }
    if (
        len(delistings) != len(delisting_events)
        or set(delistings) != expected_delistings
    ):
        raise PITAlphaDataError(
            "membership construction requires every historical delisting"
        )
    for security_id, listing_row in listings.items():
        listing_row.validate()
        if listing_row.effective_at_ms != masters[security_id].listing_at_ms:
            raise PITAlphaDataError("listing event differs from security master")
    for security_id, delisting_row in delistings.items():
        delisting_row.validate()
        if delisting_row.effective_at_ms != masters[security_id].delisting_at_ms:
            raise PITAlphaDataError("delisting event differs from security master")
    grouped: defaultdict[int, list[UniverseRankInputRecord]] = defaultdict(list)
    for rank_row in rank_inputs:
        rank_row.validate_for(rule)
        if rank_row.security_id not in masters:
            raise PITAlphaDataError("rank input references an unknown security")
        grouped[rank_row.effective_at_ms].append(rank_row)
    if not grouped:
        raise PITAlphaDataError("historical universe requires rank-input groups")

    output: list[HistoricalMembershipRecord] = []
    previously_observed: set[str] = set()
    for effective_at_ms in sorted(grouped):
        group = grouped[effective_at_ms]
        by_security = {row.security_id: row for row in group}
        if len(by_security) != len(group):
            raise PITAlphaDataError("rank-input group contains duplicate securities")
        active = {
            security_id
            for security_id, master in masters.items()
            if _active_at(master, effective_at_ms)
        }
        if set(by_security) != active:
            missing = sorted(active - set(by_security))
            unexpected = sorted(set(by_security) - active)
            raise PITAlphaDataError(
                "rank-input group is not the complete active candidate set: "
                f"missing={missing}, unexpected={unexpected}"
            )
        group_receipt = _rank_group_receipt(rule, group)
        eligible: list[UniverseRankInputRecord] = []
        reasons: dict[str, str] = {}
        for security_id, candidate_row in by_security.items():
            master = masters[security_id]
            if master.security_type not in rule.eligible_security_types:
                reasons[security_id] = "security-type-ineligible"
            elif candidate_row.observed_session_count < rule.minimum_observed_sessions:
                reasons[security_id] = "insufficient-observations"
            elif (
                candidate_row.close_price is None
                or candidate_row.close_price < rule.minimum_close_price
            ):
                reasons[security_id] = "price-ineligible"
            elif (
                candidate_row.average_dollar_volume is None
                or candidate_row.average_dollar_volume
                < rule.minimum_average_dollar_volume
            ):
                reasons[security_id] = "liquidity-ineligible"
            else:
                eligible.append(candidate_row)
        eligible.sort(
            key=lambda row: (
                -float(row.average_dollar_volume or 0.0),
                row.security_id,
            )
        )
        ranks = {row.security_id: index for index, row in enumerate(eligible, start=1)}
        group_available = max(row.available_at_ms for row in group)
        group_observation_end = max(row.observation_end_ms for row in group)
        known = {
            security_id
            for security_id, master in masters.items()
            if master.listing_at_ms <= effective_at_ms
        }
        for security_id in sorted(known):
            selected_input = by_security.get(security_id)
            if selected_input is None:
                rank = None
                is_member = False
                reason = "delisted"
                membership_available = max(
                    group_available, delistings[security_id].available_at_ms
                )
            else:
                rank = ranks.get(security_id)
                is_member = rank is not None and rank <= rule.target_size
                reason = (
                    "top-k"
                    if is_member
                    else "below-cutoff"
                    if rank is not None
                    else reasons[security_id]
                )
                membership_available = max(
                    group_available, listings[security_id].available_at_ms
                )
            output.append(
                HistoricalMembershipRecord(
                    security_id=security_id,
                    effective_at_ms=effective_at_ms,
                    available_at_ms=membership_available,
                    observation_end_ms=group_observation_end,
                    is_member=is_member,
                    universe_rank=rank,
                    eligibility_reason=reason,
                    rank_input_group_receipt_sha256=group_receipt,
                    universe_rule_receipt_sha256=rule.receipt_sha256,
                )
            )
        previously_observed.update(known)
    if previously_observed != {
        security_id
        for security_id, master in masters.items()
        if master.listing_at_ms <= max(grouped)
    }:
        raise PITAlphaDataError("membership history did not cover known securities")
    result = tuple(output)
    for membership_row in result:
        membership_row.validate()
    return result


@dataclass(frozen=True, slots=True)
class PITSecurityUniverseAuthority:
    rule: PITUniverseRuleSpec
    security_master: tuple[SourcedSecurityMasterRecord, ...]
    ticker_history: tuple[SourcedTickerHistoryRecord, ...]
    listing_events: tuple[ListingEventRecord, ...]
    delisting_events: tuple[DelistingEventRecord, ...]
    rank_inputs: tuple[UniverseRankInputRecord, ...]
    membership_events: tuple[HistoricalMembershipRecord, ...]
    receipt_sha256: str
    schema: str = PIT_SECURITY_UNIVERSE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rule": self.rule.unsigned() | {"receipt_sha256": self.rule.receipt_sha256},
            "security_master": [asdict(row) for row in self.security_master],
            "ticker_history": [asdict(row) for row in self.ticker_history],
            "listing_events": [asdict(row) for row in self.listing_events],
            "delisting_events": [asdict(row) for row in self.delisting_events],
            "rank_inputs": [asdict(row) for row in self.rank_inputs],
            "membership_events": [asdict(row) for row in self.membership_events],
        }

    def validate(self) -> None:
        if self.schema != PIT_SECURITY_UNIVERSE_SCHEMA:
            raise PITAlphaDataError("PIT security-universe schema is unsupported")
        self.rule.validate()
        masters = _validate_identity_graph(
            self.security_master,
            self.ticker_history,
            self.listing_events,
            self.delisting_events,
        )
        for row in self.rank_inputs:
            row.validate_for(self.rule)
            if row.security_id not in masters:
                raise PITAlphaDataError("rank input references an unknown security")
        expected_membership = build_historical_membership(
            rule=self.rule,
            security_master=self.security_master,
            listing_events=self.listing_events,
            delisting_events=self.delisting_events,
            rank_inputs=self.rank_inputs,
        )
        if self.membership_events != expected_membership:
            raise PITAlphaDataError(
                "membership events differ from independent PIT rank reconstruction"
            )
        _digest("security-universe authority receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise PITAlphaDataError(
                "security-universe authority receipt differs from its payload"
            )

    @classmethod
    def build(
        cls,
        *,
        rule: PITUniverseRuleSpec,
        security_master: Sequence[SourcedSecurityMasterRecord],
        ticker_history: Sequence[SourcedTickerHistoryRecord],
        listing_events: Sequence[ListingEventRecord],
        delisting_events: Sequence[DelistingEventRecord],
        rank_inputs: Sequence[UniverseRankInputRecord],
    ) -> PITSecurityUniverseAuthority:
        masters = tuple(sorted(security_master, key=lambda row: row.security_id))
        tickers = tuple(
            sorted(
                ticker_history,
                key=lambda row: (row.security_id, row.valid_from_ms),
            )
        )
        listings = tuple(
            sorted(listing_events, key=lambda row: (row.effective_at_ms, row.event_id))
        )
        delistings = tuple(
            sorted(
                delisting_events,
                key=lambda row: (row.effective_at_ms, row.event_id),
            )
        )
        ranks = tuple(
            sorted(
                rank_inputs,
                key=lambda row: (row.effective_at_ms, row.security_id),
            )
        )
        memberships = build_historical_membership(
            rule=rule,
            security_master=masters,
            listing_events=listings,
            delisting_events=delistings,
            rank_inputs=ranks,
        )
        provisional = cls(
            rule=rule,
            security_master=masters,
            ticker_history=tickers,
            listing_events=listings,
            delisting_events=delistings,
            rank_inputs=ranks,
            membership_events=memberships,
            receipt_sha256="0" * 64,
        )
        result = cls(
            rule=rule,
            security_master=masters,
            ticker_history=tickers,
            listing_events=listings,
            delisting_events=delistings,
            rank_inputs=ranks,
            membership_events=memberships,
            receipt_sha256=semantic_sha256(provisional.unsigned()),
        )
        result.validate()
        return result


def _validate_identity_graph(
    security_master: Sequence[SourcedSecurityMasterRecord],
    ticker_history: Sequence[SourcedTickerHistoryRecord],
    listing_events: Sequence[ListingEventRecord],
    delisting_events: Sequence[DelistingEventRecord],
) -> dict[str, SourcedSecurityMasterRecord]:
    masters: dict[str, SourcedSecurityMasterRecord] = {}
    for master_row in security_master:
        master_row.validate()
        if master_row.security_id == "CASH" or master_row.security_id in masters:
            raise PITAlphaDataError("security master contains a duplicate/reserved ID")
        masters[master_row.security_id] = master_row
    if not masters:
        raise PITAlphaDataError("security master cannot be empty")
    for master_row in masters.values():
        if (
            master_row.successor_security_id is not None
            and master_row.successor_security_id not in masters
        ):
            raise PITAlphaDataError("security master successor is unknown")
    for security_id in masters:
        visited: set[str] = set()
        current: str | None = security_id
        while current is not None:
            if current in visited:
                raise PITAlphaDataError("security successor graph contains a cycle")
            visited.add(current)
            current = masters[current].successor_security_id

    by_security: defaultdict[str, list[SourcedTickerHistoryRecord]] = defaultdict(list)
    ticker_intervals: defaultdict[tuple[str, str], list[SourcedTickerHistoryRecord]] = (
        defaultdict(list)
    )
    for ticker_row in ticker_history:
        ticker_row.validate()
        master = masters.get(ticker_row.security_id)
        if master is None:
            raise PITAlphaDataError("ticker history references an unknown security")
        if ticker_row.primary_exchange != master.primary_exchange:
            raise PITAlphaDataError("ticker exchange differs from security master")
        if ticker_row.valid_from_ms < master.listing_at_ms or (
            master.delisting_at_ms is not None
            and (ticker_row.valid_to_ms or master.delisting_at_ms)
            > master.delisting_at_ms
        ):
            raise PITAlphaDataError("ticker interval lies outside security lifetime")
        by_security[ticker_row.security_id].append(ticker_row)
        ticker_intervals[(ticker_row.primary_exchange, ticker_row.ticker)].append(
            ticker_row
        )
    if set(by_security) != set(masters):
        raise PITAlphaDataError("every security requires ticker history")
    for security_id, rows in by_security.items():
        ordered = sorted(rows, key=lambda row: row.valid_from_ms)
        master = masters[security_id]
        if ordered[0].valid_from_ms != master.listing_at_ms:
            raise PITAlphaDataError("ticker history does not begin at listing")
        for earlier, later in pairwise(ordered):
            if earlier.valid_to_ms != later.valid_from_ms:
                raise PITAlphaDataError("ticker history has a gap or overlap")
        expected_end = master.delisting_at_ms
        if ordered[-1].valid_to_ms != expected_end:
            raise PITAlphaDataError(
                "ticker history does not cover the security lifetime"
            )
    for rows in ticker_intervals.values():
        ordered = sorted(rows, key=lambda row: row.valid_from_ms)
        for earlier, later in pairwise(ordered):
            if earlier.valid_to_ms is None or earlier.valid_to_ms > later.valid_from_ms:
                raise PITAlphaDataError("ticker history intervals overlap")

    listings: dict[str, ListingEventRecord] = {}
    event_ids: set[str] = set()
    for listing_row in listing_events:
        listing_row.validate()
        master = masters.get(listing_row.security_id)
        if master is None or listing_row.security_id in listings:
            raise PITAlphaDataError("listing events must map one-to-one to securities")
        if listing_row.event_id in event_ids:
            raise PITAlphaDataError("identity event IDs must be globally unique")
        event_ids.add(listing_row.event_id)
        if (
            listing_row.effective_at_ms != master.listing_at_ms
            or listing_row.primary_exchange != master.primary_exchange
        ):
            raise PITAlphaDataError("listing event differs from security master")
        candidates = by_security[listing_row.security_id]
        if not any(
            ticker.ticker == listing_row.ticker
            and ticker.valid_from_ms == listing_row.effective_at_ms
            for ticker in candidates
        ):
            raise PITAlphaDataError("listing event lacks matching ticker history")
        listings[listing_row.security_id] = listing_row
    if set(listings) != set(masters):
        raise PITAlphaDataError("every security requires exactly one listing event")

    delistings: dict[str, DelistingEventRecord] = {}
    for delisting_row in delisting_events:
        delisting_row.validate()
        master = masters.get(delisting_row.security_id)
        if master is None or delisting_row.security_id in delistings:
            raise PITAlphaDataError("delisting events must be unique by security")
        if delisting_row.event_id in event_ids:
            raise PITAlphaDataError("identity event IDs must be globally unique")
        event_ids.add(delisting_row.event_id)
        if master.delisting_at_ms != delisting_row.effective_at_ms:
            raise PITAlphaDataError("delisting event differs from security master")
        if master.successor_security_id != delisting_row.successor_security_id:
            raise PITAlphaDataError("delisting successor differs from security master")
        if (
            delisting_row.successor_security_id is not None
            and delisting_row.successor_security_id not in masters
        ):
            raise PITAlphaDataError("delisting successor is unknown")
        delistings[delisting_row.security_id] = delisting_row
    expected_delistings = {
        row.security_id for row in masters.values() if row.delisting_at_ms is not None
    }
    if set(delistings) != expected_delistings:
        raise PITAlphaDataError(
            "every delisted security requires exactly one delisting event"
        )
    return masters


@dataclass(frozen=True, slots=True)
class PolygonStagingInventoryRecord:
    symbol: str
    session_date: str
    source_receipt_sha256: str
    output_file_sha256: str
    commit_receipt_sha256: str

    def validate(self) -> None:
        _ticker(self.symbol)
        _iso_date("staged session date", self.session_date)
        for name in (
            "source_receipt_sha256",
            "output_file_sha256",
            "commit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class PITPolygonCoverageAudit:
    required_symbol_day_count: int
    covered_symbol_day_count: int
    missing_symbol_days: tuple[str, ...]
    unresolved_ticker_symbol_days: tuple[str, ...]
    unused_inventory_symbol_days: tuple[str, ...]
    coverage_fraction: float
    bar_source_inventory_complete: bool
    pit_alpha_training_ready: bool
    reportable_pit_authority_ready: bool
    receipt_sha256: str
    schema: str = PIT_POLYGON_COVERAGE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != PIT_POLYGON_COVERAGE_SCHEMA:
            raise PITAlphaDataError("Polygon coverage schema is unsupported")
        required = _positive_int(
            "required symbol-day count", self.required_symbol_day_count, allow_zero=True
        )
        covered = _positive_int(
            "covered symbol-day count", self.covered_symbol_day_count, allow_zero=True
        )
        if covered > required:
            raise PITAlphaDataError("covered symbol-days exceed requirements")
        fraction = _finite("coverage fraction", self.coverage_fraction, minimum=0.0)
        expected = 1.0 if required == 0 else covered / required
        if fraction != expected or fraction > 1.0:
            raise PITAlphaDataError("coverage fraction differs from counts")
        for rows in (
            self.missing_symbol_days,
            self.unresolved_ticker_symbol_days,
            self.unused_inventory_symbol_days,
        ):
            if tuple(sorted(set(rows))) != rows:
                raise PITAlphaDataError(
                    "coverage inventories must be sorted and unique"
                )
        expected_complete = (
            not self.missing_symbol_days
            and not self.unresolved_ticker_symbol_days
            and covered == required
        )
        if self.bar_source_inventory_complete is not expected_complete:
            raise PITAlphaDataError("coverage-complete flag differs from evidence")
        if self.pit_alpha_training_ready or self.reportable_pit_authority_ready:
            raise PITAlphaDataError(
                "coverage alone cannot authorize PIT alpha training"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise PITAlphaDataError("coverage receipt differs from its payload")


def audit_polygon_staging_coverage(
    *,
    authority: PITSecurityUniverseAuthority,
    required_sessions: Sequence[tuple[str, int]],
    inventory: Sequence[PolygonStagingInventoryRecord],
) -> PITPolygonCoverageAudit:
    """Measure source coverage without treating Polygon as identity authority."""

    authority.validate()
    sessions = tuple(required_sessions)
    if not sessions or tuple(sorted(set(sessions), key=lambda row: row[1])) != sessions:
        raise PITAlphaDataError("required sessions must be unique and chronological")
    for session_date, effective_at_ms in sessions:
        _iso_date("required session date", session_date)
        _timestamp("required session time", effective_at_ms)
    inventory_by_key: dict[tuple[str, str], PolygonStagingInventoryRecord] = {}
    for inventory_row in inventory:
        inventory_row.validate()
        key = (inventory_row.symbol, inventory_row.session_date)
        if key in inventory_by_key:
            raise PITAlphaDataError("Polygon staging inventory contains duplicate keys")
        inventory_by_key[key] = inventory_row

    ticker_by_security: defaultdict[str, list[SourcedTickerHistoryRecord]] = (
        defaultdict(list)
    )
    for ticker_row in authority.ticker_history:
        ticker_by_security[ticker_row.security_id].append(ticker_row)
    required_keys: set[tuple[str, str]] = set()
    missing: list[str] = []
    unresolved: list[str] = []
    covered = 0
    required = 0
    for session_date, effective_at_ms in sessions:
        for master in authority.security_master:
            if not _active_at(master, effective_at_ms):
                continue
            required += 1
            matches = [
                row
                for row in ticker_by_security[master.security_id]
                if row.valid_from_ms <= effective_at_ms
                and (row.valid_to_ms is None or effective_at_ms < row.valid_to_ms)
                and row.available_at_ms <= effective_at_ms
            ]
            if len(matches) != 1:
                unresolved.append(f"{master.security_id}|{session_date}")
                continue
            key = (matches[0].ticker, session_date)
            required_keys.add(key)
            if key in inventory_by_key:
                covered += 1
            else:
                missing.append(f"{master.security_id}|{key[0]}|{session_date}")
    unused = tuple(
        sorted(
            f"{symbol}|{session_date}"
            for symbol, session_date in set(inventory_by_key) - required_keys
        )
    )
    body: dict[str, object] = {
        "schema": PIT_POLYGON_COVERAGE_SCHEMA,
        "required_symbol_day_count": required,
        "covered_symbol_day_count": covered,
        "missing_symbol_days": sorted(set(missing)),
        "unresolved_ticker_symbol_days": sorted(set(unresolved)),
        "unused_inventory_symbol_days": list(unused),
        "coverage_fraction": 1.0 if required == 0 else covered / required,
        "bar_source_inventory_complete": not missing
        and not unresolved
        and covered == required,
        "pit_alpha_training_ready": False,
        "reportable_pit_authority_ready": False,
    }
    result = PITPolygonCoverageAudit(
        required_symbol_day_count=required,
        covered_symbol_day_count=covered,
        missing_symbol_days=tuple(sorted(set(missing))),
        unresolved_ticker_symbol_days=tuple(sorted(set(unresolved))),
        unused_inventory_symbol_days=unused,
        coverage_fraction=1.0 if required == 0 else covered / required,
        bar_source_inventory_complete=not missing
        and not unresolved
        and covered == required,
        pit_alpha_training_ready=False,
        reportable_pit_authority_ready=False,
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


_TABLE_SPECS: dict[str, tuple[tuple[str, str, bool], ...]] = {
    SECURITY_MASTER_FILE: (
        ("security_id", "string", False),
        ("issuer_id", "string", False),
        ("primary_exchange", "string", False),
        ("share_class", "string", False),
        ("security_type", "string", False),
        ("listing_at_ms", "int64", False),
        ("delisting_at_ms", "int64", True),
        ("successor_security_id", "string", True),
        ("corporate_action_chain_id", "string", True),
        ("identity_source_receipt_sha256", "string", False),
    ),
    TICKER_HISTORY_FILE: (
        ("security_id", "string", False),
        ("ticker", "string", False),
        ("valid_from_ms", "int64", False),
        ("valid_to_ms", "int64", True),
        ("available_at_ms", "int64", False),
        ("primary_exchange", "string", False),
        ("source_receipt_sha256", "string", False),
    ),
    LISTING_EVENTS_FILE: (
        ("event_id", "string", False),
        ("security_id", "string", False),
        ("effective_at_ms", "int64", False),
        ("available_at_ms", "int64", False),
        ("primary_exchange", "string", False),
        ("ticker", "string", False),
        ("source_receipt_sha256", "string", False),
    ),
    DELISTING_EVENTS_FILE: (
        ("event_id", "string", False),
        ("security_id", "string", False),
        ("effective_at_ms", "int64", False),
        ("available_at_ms", "int64", False),
        ("reason", "string", False),
        ("successor_security_id", "string", True),
        ("source_receipt_sha256", "string", False),
    ),
    UNIVERSE_RANK_INPUTS_FILE: (
        ("security_id", "string", False),
        ("effective_at_ms", "int64", False),
        ("effective_session_index", "int64", False),
        ("available_at_ms", "int64", False),
        ("observation_start_ms", "int64", False),
        ("observation_end_ms", "int64", False),
        ("observation_start_session_index", "int64", False),
        ("observation_end_session_index", "int64", False),
        ("observed_session_count", "int64", False),
        ("average_dollar_volume", "float64", True),
        ("close_price", "float64", True),
        ("source_receipt_sha256", "string", False),
    ),
    MEMBERSHIP_EVENTS_FILE: (
        ("security_id", "string", False),
        ("effective_at_ms", "int64", False),
        ("available_at_ms", "int64", False),
        ("observation_end_ms", "int64", False),
        ("is_member", "bool", False),
        ("universe_rank", "int64", True),
        ("eligibility_reason", "string", False),
        ("rank_input_group_receipt_sha256", "string", False),
        ("universe_rule_receipt_sha256", "string", False),
    ),
}


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise PITAlphaDataError(
            "pyarrow is required for PIT authority materialization"
        ) from exc
    return pa, pq


def _table_from_rows(file_name: str, rows: Sequence[Any]) -> Any:
    pa, _ = _require_pyarrow()
    kinds = {
        "string": pa.string(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }
    spec = _TABLE_SPECS[file_name]
    schema = pa.schema(
        [
            pa.field(name, kinds[kind], nullable=nullable)
            for name, kind, nullable in spec
        ]
    )
    payload = [asdict(row) for row in rows]
    values = {name: [row[name] for row in payload] for name, _, _ in spec}
    return pa.Table.from_pydict(values, schema=schema).replace_schema_metadata(None)


def _semantic_table_sha(file_name: str, rows: Sequence[Any]) -> str:
    spec = _TABLE_SPECS[file_name]
    return semantic_sha256(
        {
            "schema": [
                {"name": name, "type": kind, "nullable": nullable}
                for name, kind, nullable in spec
            ],
            "rows": [asdict(row) for row in rows],
        }
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_file(parent: Path, name: str) -> tuple[int, Path]:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=parent
    )
    return descriptor, Path(raw_path)


def _write_temp_bytes(parent: Path, name: str, raw: bytes) -> Path:
    descriptor, path = _temporary_file(parent, name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _write_temp_parquet(parent: Path, name: str, table: Any) -> Path:
    _, pq = _require_pyarrow()
    descriptor, path = _temporary_file(parent, name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pq.write_table(
                table,
                stream,
                compression=_PARQUET_COMPRESSION,
                compression_level=_PARQUET_COMPRESSION_LEVEL,
                use_dictionary=False,
                write_statistics=True,
                row_group_size=_PARQUET_ROW_GROUP_SIZE,
                data_page_version="2.0",
                version="2.6",
            )
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, *, expected_sha256: str) -> bytes:
    _digest("expected file SHA", expected_sha256)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise PITAlphaDataError(f"authority file is unavailable: {path}") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise PITAlphaDataError("authority member must be a nonempty regular file")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PITAlphaDataError("authority member changed while it was read")
    if digest.hexdigest() != expected_sha256:
        raise PITAlphaDataError("authority member SHA-256 drifted")
    return b"".join(chunks)


def materialize_pit_security_universe(
    root: Path, authority: PITSecurityUniverseAuthority
) -> dict[str, object]:
    """Publish the seven required files plus one immutable authority receipt."""

    authority.validate()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or any(root.iterdir()):
        raise PITAlphaDataError("PIT security-universe output root must be empty")
    rows_by_file: dict[str, Sequence[object]] = {
        SECURITY_MASTER_FILE: authority.security_master,
        TICKER_HISTORY_FILE: authority.ticker_history,
        LISTING_EVENTS_FILE: authority.listing_events,
        DELISTING_EVENTS_FILE: authority.delisting_events,
        MEMBERSHIP_EVENTS_FILE: authority.membership_events,
        UNIVERSE_RANK_INPUTS_FILE: authority.rank_inputs,
    }
    temporaries: list[Path] = []
    linked: list[Path] = []
    records: list[dict[str, object]] = []
    try:
        for file_name, rows in rows_by_file.items():
            temporary = _write_temp_parquet(
                root, file_name, _table_from_rows(file_name, rows)
            )
            temporaries.append(temporary)
            records.append(
                {
                    "relative_path": file_name,
                    "file_sha256": _file_sha(temporary),
                    "size_bytes": temporary.stat().st_size,
                    "semantic_sha256": _semantic_table_sha(file_name, rows),
                    "row_count": len(rows),
                }
            )
        rule_raw = canonical_json_file_bytes(
            authority.rule.unsigned()
            | {"receipt_sha256": authority.rule.receipt_sha256}
        )
        rule_temp = _write_temp_bytes(root, UNIVERSE_RULE_FILE, rule_raw)
        temporaries.append(rule_temp)
        records.append(
            {
                "relative_path": UNIVERSE_RULE_FILE,
                "file_sha256": hashlib.sha256(rule_raw).hexdigest(),
                "size_bytes": len(rule_raw),
                "semantic_sha256": authority.rule.receipt_sha256,
                "row_count": 1,
            }
        )
        records.sort(key=lambda row: str(row["relative_path"]))
        publication_body: dict[str, object] = {
            "schema": PIT_UNIVERSE_BUNDLE_SCHEMA,
            "authority_receipt_sha256": authority.receipt_sha256,
            "files": records,
            "identity_authority_ready": True,
            "historical_universe_authority_ready": True,
            "pit_alpha_training_ready": False,
            "reportable_pit_authority_ready": False,
            "blocking_reason": (
                "terminal economics, causal cash, availability, total-return "
                "reconciliation, and training tensors remain unissued"
            ),
        }
        publication = publication_body | {
            "receipt_sha256": semantic_sha256(publication_body)
        }
        authority_raw = canonical_json_file_bytes(publication)
        authority_temp = _write_temp_bytes(root, AUTHORITY_FILE, authority_raw)
        temporaries.append(authority_temp)

        ordered_temps = temporaries[:-1]
        ordered_names = list(rows_by_file) + [UNIVERSE_RULE_FILE]
        for temporary, file_name in zip(ordered_temps, ordered_names, strict=True):
            target = root / file_name
            os.link(temporary, target, follow_symlinks=False)
            linked.append(target)
        os.link(authority_temp, root / AUTHORITY_FILE, follow_symlinks=False)
        linked.append(root / AUTHORITY_FILE)
        _fsync_directory(root)
    except BaseException:
        for target in reversed(linked):
            target.unlink(missing_ok=True)
        if linked:
            _fsync_directory(root)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return publication | {
        "authority_file_sha256": hashlib.sha256(authority_raw).hexdigest()
    }


def _rule_from_dict(value: object) -> PITUniverseRuleSpec:
    if not isinstance(value, dict):
        raise PITAlphaDataError("universe rule file must contain one object")
    expected = {
        "schema",
        "rule_id",
        "target_size",
        "ranking_metric",
        "ranking_lookback_sessions",
        "ranking_lag_sessions",
        "minimum_observed_sessions",
        "minimum_close_price",
        "minimum_average_dollar_volume",
        "eligible_security_types",
        "rebalance_frequency",
        "tie_breaker",
        "uses_future_survival",
        "receipt_sha256",
    }
    if set(value) != expected or not isinstance(value["eligible_security_types"], list):
        raise PITAlphaDataError("universe rule file has an unexpected shape")
    try:
        rule = PITUniverseRuleSpec(
            schema=value["schema"],
            rule_id=value["rule_id"],
            target_size=value["target_size"],
            ranking_metric=value["ranking_metric"],
            ranking_lookback_sessions=value["ranking_lookback_sessions"],
            ranking_lag_sessions=value["ranking_lag_sessions"],
            minimum_observed_sessions=value["minimum_observed_sessions"],
            minimum_close_price=value["minimum_close_price"],
            minimum_average_dollar_volume=value["minimum_average_dollar_volume"],
            eligible_security_types=tuple(value["eligible_security_types"]),
            rebalance_frequency=value["rebalance_frequency"],
            tie_breaker=value["tie_breaker"],
            uses_future_survival=value["uses_future_survival"],
            receipt_sha256=value["receipt_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise PITAlphaDataError("universe rule file is malformed") from exc
    rule.validate()
    return rule


def _parquet_rows(file_name: str, raw: bytes) -> list[dict[str, object]]:
    _, pq = _require_pyarrow()
    try:
        table = pq.read_table(io.BytesIO(raw))
    except Exception as exc:
        raise PITAlphaDataError(
            f"authority Parquet is unreadable: {file_name}"
        ) from exc
    observed_schema = tuple(
        (
            field.name,
            "float64" if str(field.type) == "double" else str(field.type),
            field.nullable,
        )
        for field in table.schema
    )
    if observed_schema != _TABLE_SPECS[file_name]:
        raise PITAlphaDataError(f"authority Parquet schema drifted: {file_name}")
    rows = table.to_pylist()
    if any(not isinstance(row, dict) for row in rows):
        raise PITAlphaDataError(f"authority Parquet rows are malformed: {file_name}")
    return rows


def _construct_rows(
    constructor: Any, values: Sequence[dict[str, object]], *, label: str
) -> tuple[Any, ...]:
    try:
        return tuple(constructor(**row) for row in values)
    except (TypeError, ValueError) as exc:
        raise PITAlphaDataError(f"{label} rows are malformed") from exc


def load_pit_security_universe(
    root: Path, *, expected_authority_file_sha256: str
) -> PITSecurityUniverseAuthority:
    """Reopen exact files and independently reconstruct the universe authority."""

    if not root.is_dir() or root.is_symlink():
        raise PITAlphaDataError(
            "PIT security-universe root must be a non-symlink directory"
        )
    allowed = set(_TABLE_SPECS) | {UNIVERSE_RULE_FILE, AUTHORITY_FILE}
    observed: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise PITAlphaDataError("PIT security-universe tree contains a non-file")
        observed.add(path.name)
    if observed != allowed:
        raise PITAlphaDataError("PIT security-universe file inventory drifted")
    authority_raw = _read_regular_bytes(
        root / AUTHORITY_FILE, expected_sha256=expected_authority_file_sha256
    )
    try:
        publication = json.loads(authority_raw)
    except json.JSONDecodeError as exc:
        raise PITAlphaDataError(
            "identity-universe authority JSON is malformed"
        ) from exc
    if (
        not isinstance(publication, dict)
        or authority_raw != canonical_json_file_bytes(publication)
        or publication.get("schema") != PIT_UNIVERSE_BUNDLE_SCHEMA
        or publication.get("identity_authority_ready") is not True
        or publication.get("historical_universe_authority_ready") is not True
        or publication.get("pit_alpha_training_ready") is not False
        or publication.get("reportable_pit_authority_ready") is not False
    ):
        raise PITAlphaDataError("identity-universe authority contract drifted")
    unsigned = {
        key: value for key, value in publication.items() if key != "receipt_sha256"
    }
    if publication.get("receipt_sha256") != semantic_sha256(unsigned):
        raise PITAlphaDataError("identity-universe authority receipt drifted")
    raw_records = publication.get("files")
    if not isinstance(raw_records, list):
        raise PITAlphaDataError("identity-universe file inventory is malformed")
    records: dict[str, dict[str, object]] = {}
    for record in raw_records:
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "file_sha256",
            "size_bytes",
            "semantic_sha256",
            "row_count",
        }:
            raise PITAlphaDataError("identity-universe file record is malformed")
        relative = record["relative_path"]
        if not isinstance(relative, str) or relative in records:
            raise PITAlphaDataError("identity-universe file paths are invalid")
        records[relative] = record
    expected_members = set(_TABLE_SPECS) | {UNIVERSE_RULE_FILE}
    if set(records) != expected_members:
        raise PITAlphaDataError("identity-universe authority omits required files")

    table_values: dict[str, list[dict[str, object]]] = {}
    rule: PITUniverseRuleSpec | None = None
    for file_name in sorted(expected_members):
        record = records[file_name]
        file_sha = _digest("authority file SHA", record["file_sha256"])
        raw = _read_regular_bytes(root / file_name, expected_sha256=file_sha)
        if record["size_bytes"] != len(raw):
            raise PITAlphaDataError("authority file size differs from its receipt")
        if file_name == UNIVERSE_RULE_FILE:
            try:
                rule_value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PITAlphaDataError("universe rule JSON is malformed") from exc
            if raw != canonical_json_file_bytes(rule_value):
                raise PITAlphaDataError("universe rule file is not canonical JSON")
            rule = _rule_from_dict(rule_value)
            if (
                record["row_count"] != 1
                or record["semantic_sha256"] != rule.receipt_sha256
            ):
                raise PITAlphaDataError("universe rule file receipt drifted")
            continue
        rows = _parquet_rows(file_name, raw)
        if record["row_count"] != len(rows):
            raise PITAlphaDataError("authority Parquet row count drifted")
        table_values[file_name] = rows

    if rule is None:
        raise PITAlphaDataError("identity-universe authority lacks its rule")

    masters = _construct_rows(
        SourcedSecurityMasterRecord,
        table_values[SECURITY_MASTER_FILE],
        label="security master",
    )
    tickers = _construct_rows(
        SourcedTickerHistoryRecord,
        table_values[TICKER_HISTORY_FILE],
        label="ticker history",
    )
    listings = _construct_rows(
        ListingEventRecord,
        table_values[LISTING_EVENTS_FILE],
        label="listing event",
    )
    delistings = _construct_rows(
        DelistingEventRecord,
        table_values[DELISTING_EVENTS_FILE],
        label="delisting event",
    )
    rank_inputs = _construct_rows(
        UniverseRankInputRecord,
        table_values[UNIVERSE_RANK_INPUTS_FILE],
        label="universe rank input",
    )
    memberships = _construct_rows(
        HistoricalMembershipRecord,
        table_values[MEMBERSHIP_EVENTS_FILE],
        label="membership event",
    )
    typed_by_file: dict[str, Sequence[Any]] = {
        SECURITY_MASTER_FILE: masters,
        TICKER_HISTORY_FILE: tickers,
        LISTING_EVENTS_FILE: listings,
        DELISTING_EVENTS_FILE: delistings,
        UNIVERSE_RANK_INPUTS_FILE: rank_inputs,
        MEMBERSHIP_EVENTS_FILE: memberships,
    }
    for file_name, typed_rows in typed_by_file.items():
        if records[file_name]["semantic_sha256"] != _semantic_table_sha(
            file_name, typed_rows
        ):
            raise PITAlphaDataError("authority semantic table receipt drifted")
    reconstructed = PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=delistings,
        rank_inputs=rank_inputs,
    )
    if (
        memberships != reconstructed.membership_events
        or publication.get("authority_receipt_sha256") != reconstructed.receipt_sha256
    ):
        raise PITAlphaDataError("materialized membership or authority receipt drifted")
    return reconstructed


__all__ = [
    "AUTHORITY_FILE",
    "DELISTING_EVENTS_FILE",
    "LISTING_EVENTS_FILE",
    "MEMBERSHIP_EVENTS_FILE",
    "SECURITY_MASTER_FILE",
    "TICKER_HISTORY_FILE",
    "UNIVERSE_RANK_INPUTS_FILE",
    "UNIVERSE_RULE_FILE",
    "DelistingEventRecord",
    "HistoricalMembershipRecord",
    "ListingEventRecord",
    "PITPolygonCoverageAudit",
    "PITSecurityUniverseAuthority",
    "PITUniverseRuleSpec",
    "PolygonStagingInventoryRecord",
    "SourcedSecurityMasterRecord",
    "SourcedTickerHistoryRecord",
    "UniverseRankInputRecord",
    "audit_polygon_staging_coverage",
    "build_historical_membership",
    "load_pit_security_universe",
    "materialize_pit_security_universe",
]
