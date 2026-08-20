"""Point-in-time dataset authority for reportable alpha research.

The contracts in this module deliberately avoid dataframe and tensor
dependencies.  They validate permanent security identity, causal universe
membership, distinct decision/fill/mark states, terminal economic outcomes,
and an exact byte inventory before any model adapter is allowed to materialize
tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable, Mapping, Sequence

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)


PIT_ALPHA_DATASET_SCHEMA = "rl-quant.pit-alpha-dataset-v1"
PIT_ALPHA_CASH_ID = "CASH"


class PITAlphaDataError(ValueError):
    """A point-in-time or economic data invariant is missing."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PITAlphaDataError(f"{name} must be a non-empty canonical string")
    return value


def _timestamp(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PITAlphaDataError(f"{name} must be a nonnegative epoch-millisecond integer")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PITAlphaDataError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PITAlphaDataError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or (minimum is not None and normalized < minimum):
        raise PITAlphaDataError(f"{name} is outside its finite domain")
    return normalized


def _safe_relative_path(value: object) -> str:
    path = PurePosixPath(_text("dataset file path", value))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PITAlphaDataError("dataset file paths must be normalized relative paths")
    return path.as_posix()


class CorporateActionKind(str, Enum):
    CASH_DIVIDEND = "cash-dividend"
    SPECIAL_DIVIDEND = "special-dividend"
    SPLIT = "split"
    REVERSE_SPLIT = "reverse-split"
    SPIN_OFF = "spin-off"
    RIGHTS_DISTRIBUTION = "rights-distribution"
    TENDER_OFFER = "tender-offer"
    MERGER_CASH = "merger-cash"
    MERGER_STOCK = "merger-stock"
    RETURN_OF_CAPITAL = "return-of-capital"
    TICKER_EXCHANGE_CHANGE = "ticker-exchange-change"


class TerminalEventKind(str, Enum):
    DELISTING_CASH = "delisting-cash"
    MERGER_CASH = "merger-cash"
    MERGER_STOCK = "merger-stock"
    BANKRUPTCY_RECOVERY = "bankruptcy-recovery"
    WORTHLESS = "worthless"


@dataclass(frozen=True, slots=True)
class SecurityMasterRecord:
    security_id: str
    issuer_id: str
    primary_exchange: str
    share_class: str
    security_type: str
    listing_at_ms: int
    delisting_at_ms: int | None = None
    successor_security_id: str | None = None
    corporate_action_chain_id: str | None = None

    def validate(self) -> None:
        for name in (
            "security_id",
            "issuer_id",
            "primary_exchange",
            "share_class",
            "security_type",
        ):
            _text(name, getattr(self, name))
        _timestamp("listing_at_ms", self.listing_at_ms)
        if self.delisting_at_ms is not None:
            _timestamp("delisting_at_ms", self.delisting_at_ms)
            if self.delisting_at_ms <= self.listing_at_ms:
                raise PITAlphaDataError("delisting must occur after listing")
        for name in ("successor_security_id", "corporate_action_chain_id"):
            value = getattr(self, name)
            if value is not None:
                _text(name, value)
        if self.successor_security_id == self.security_id:
            raise PITAlphaDataError("a security cannot be its own successor")


@dataclass(frozen=True, slots=True)
class TickerHistoryRecord:
    security_id: str
    ticker: str
    valid_from_ms: int
    valid_to_ms: int | None
    available_at_ms: int

    def validate(self) -> None:
        _text("security_id", self.security_id)
        _text("ticker", self.ticker)
        _timestamp("valid_from_ms", self.valid_from_ms)
        _timestamp("available_at_ms", self.available_at_ms)
        if self.available_at_ms > self.valid_from_ms:
            raise PITAlphaDataError("ticker mapping was unavailable when it became effective")
        if self.valid_to_ms is not None:
            _timestamp("valid_to_ms", self.valid_to_ms)
            if self.valid_to_ms <= self.valid_from_ms:
                raise PITAlphaDataError("ticker validity interval is empty")


@dataclass(frozen=True, slots=True)
class UniverseRule:
    rule_id: str
    rule_sha256: str
    membership_mode: str
    ranking_lookback_sessions: int
    ranking_lag_sessions: int
    uses_future_survival: bool = False

    def validate(self) -> None:
        _text("universe rule ID", self.rule_id)
        _digest("universe rule SHA", self.rule_sha256)
        if self.membership_mode != "point-in-time-events":
            raise PITAlphaDataError("universe membership must be event-sourced point in time")
        for name in ("ranking_lookback_sessions", "ranking_lag_sessions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PITAlphaDataError(f"{name} must be a nonnegative integer")
        if self.ranking_lookback_sessions == 0:
            raise PITAlphaDataError("universe ranking requires a trailing history")
        if self.uses_future_survival:
            raise PITAlphaDataError("future survival cannot enter universe membership")


@dataclass(frozen=True, slots=True)
class MembershipEvent:
    security_id: str
    effective_at_ms: int
    available_at_ms: int
    observation_end_ms: int
    is_member: bool
    universe_rank: int | None = None

    def validate(self) -> None:
        _text("security_id", self.security_id)
        for name in ("effective_at_ms", "available_at_ms", "observation_end_ms"):
            _timestamp(name, getattr(self, name))
        if self.observation_end_ms > self.available_at_ms:
            raise PITAlphaDataError("membership ranking uses observations from the future")
        if self.available_at_ms > self.effective_at_ms:
            raise PITAlphaDataError("membership was unavailable when it became effective")
        if not isinstance(self.is_member, bool):
            raise PITAlphaDataError("membership state must be boolean")
        if self.universe_rank is not None and (
            isinstance(self.universe_rank, bool)
            or not isinstance(self.universe_rank, int)
            or self.universe_rank <= 0
        ):
            raise PITAlphaDataError("universe rank must be a positive integer")
        if self.is_member and self.universe_rank is None:
            raise PITAlphaDataError("positive membership requires its point-in-time rank")


@dataclass(frozen=True, slots=True)
class AvailabilitySnapshot:
    observable: bool
    decision_eligible: bool
    fill_eligible: bool
    markable: bool
    terminal_event: bool

    def validate(self) -> None:
        values = (
            self.observable,
            self.decision_eligible,
            self.fill_eligible,
            self.markable,
            self.terminal_event,
        )
        if any(not isinstance(value, bool) for value in values):
            raise PITAlphaDataError("availability fields must be boolean")
        if self.decision_eligible and not self.observable:
            raise PITAlphaDataError("decision eligibility requires observable inputs")
        if self.fill_eligible and not self.markable:
            raise PITAlphaDataError("fillable securities must be markable")
        if self.terminal_event and (
            self.decision_eligible or self.fill_eligible or not self.markable
        ):
            raise PITAlphaDataError(
                "a terminal event must be markable and unavailable for ordinary actions"
            )


@dataclass(frozen=True, slots=True)
class AvailabilityRecord:
    security_id: str
    effective_at_ms: int
    available_at_ms: int
    state: AvailabilitySnapshot
    reason: str

    def validate(self) -> None:
        _text("security_id", self.security_id)
        _timestamp("effective_at_ms", self.effective_at_ms)
        _timestamp("available_at_ms", self.available_at_ms)
        _text("availability reason", self.reason)
        self.state.validate()


@dataclass(frozen=True, slots=True)
class CorporateActionRecord:
    event_id: str
    security_id: str
    kind: CorporateActionKind
    effective_at_ms: int
    available_at_ms: int
    cash_per_share: float = 0.0
    share_ratio: float = 1.0
    successor_security_id: str | None = None
    successor_ratio: float = 0.0
    affected_fraction: float = 1.0

    def validate(self) -> None:
        _text("corporate-action event ID", self.event_id)
        _text("security_id", self.security_id)
        if not isinstance(self.kind, CorporateActionKind):
            raise PITAlphaDataError("corporate-action kind is unsupported")
        _timestamp("effective_at_ms", self.effective_at_ms)
        _timestamp("available_at_ms", self.available_at_ms)
        cash = _finite("cash_per_share", self.cash_per_share, minimum=0.0)
        share_ratio = _finite("share_ratio", self.share_ratio, minimum=0.0)
        successor_ratio = _finite("successor_ratio", self.successor_ratio, minimum=0.0)
        fraction = _finite("affected_fraction", self.affected_fraction, minimum=0.0)
        if not 0.0 < fraction <= 1.0:
            raise PITAlphaDataError("affected_fraction must lie in (0, 1]")
        if self.kind in {CorporateActionKind.SPLIT, CorporateActionKind.REVERSE_SPLIT}:
            if share_ratio <= 0.0 or cash != 0.0 or self.successor_security_id is not None:
                raise PITAlphaDataError("split actions require only a positive share ratio")
        elif self.kind in {
            CorporateActionKind.CASH_DIVIDEND,
            CorporateActionKind.SPECIAL_DIVIDEND,
            CorporateActionKind.RETURN_OF_CAPITAL,
        }:
            if cash <= 0.0 or self.successor_security_id is not None:
                raise PITAlphaDataError("cash distributions require positive cash per share")
        elif self.kind in {
            CorporateActionKind.SPIN_OFF,
            CorporateActionKind.MERGER_STOCK,
        }:
            _text("successor_security_id", self.successor_security_id)
            if self.successor_security_id == self.security_id or successor_ratio <= 0.0:
                raise PITAlphaDataError("stock distributions require a distinct successor")
        elif self.kind in {
            CorporateActionKind.MERGER_CASH,
            CorporateActionKind.TENDER_OFFER,
        } and cash <= 0.0:
            raise PITAlphaDataError("cash acquisitions require positive cash per share")
        elif self.kind is CorporateActionKind.RIGHTS_DISTRIBUTION:
            if cash <= 0.0 and successor_ratio <= 0.0:
                raise PITAlphaDataError("rights must have a cash value or distributed security")
            if successor_ratio > 0.0:
                _text("successor_security_id", self.successor_security_id)


@dataclass(frozen=True, slots=True)
class TerminalEventRecord:
    event_id: str
    security_id: str
    kind: TerminalEventKind
    effective_at_ms: int
    available_at_ms: int
    cash_per_share: float = 0.0
    successor_security_id: str | None = None
    successor_ratio: float = 0.0

    def validate(self) -> None:
        _text("terminal event ID", self.event_id)
        _text("security_id", self.security_id)
        if not isinstance(self.kind, TerminalEventKind):
            raise PITAlphaDataError("terminal-event kind is unsupported")
        _timestamp("effective_at_ms", self.effective_at_ms)
        _timestamp("available_at_ms", self.available_at_ms)
        cash = _finite("cash_per_share", self.cash_per_share, minimum=0.0)
        successor_ratio = _finite("successor_ratio", self.successor_ratio, minimum=0.0)
        if self.kind is TerminalEventKind.MERGER_STOCK:
            _text("successor_security_id", self.successor_security_id)
            if self.successor_security_id == self.security_id or successor_ratio <= 0.0:
                raise PITAlphaDataError("stock merger requires a distinct successor and ratio")
        elif self.kind is TerminalEventKind.WORTHLESS:
            if cash != 0.0 or self.successor_security_id is not None:
                raise PITAlphaDataError("worthless disposition cannot create value")
        elif cash < 0.0 or self.successor_security_id is not None:
            raise PITAlphaDataError("cash terminal events cannot create successor shares")


@dataclass(frozen=True, slots=True)
class CashReturnRecord:
    effective_at_ms: int
    available_at_ms: int
    one_step_return: float
    source_receipt_sha256: str

    def validate(self) -> None:
        _timestamp("effective_at_ms", self.effective_at_ms)
        _timestamp("available_at_ms", self.available_at_ms)
        if self.available_at_ms > self.effective_at_ms:
            raise PITAlphaDataError("cash return was unavailable when it became effective")
        value = _finite("one_step_return", self.one_step_return)
        if value <= -1.0:
            raise PITAlphaDataError("cash gross return must remain positive")
        _digest("cash return source receipt", self.source_receipt_sha256)


@dataclass(frozen=True, slots=True)
class DatasetFileRecord:
    relative_path: str
    size_bytes: int
    file_sha256: str
    media_type: str

    def validate(self) -> None:
        if _safe_relative_path(self.relative_path) != self.relative_path:
            raise PITAlphaDataError("dataset file path is not canonical")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise PITAlphaDataError("dataset file size must be a nonnegative integer")
        _digest("dataset file SHA", self.file_sha256)
        _text("dataset file media type", self.media_type)

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "file_sha256": self.file_sha256,
            "media_type": self.media_type,
        }


@dataclass(frozen=True, slots=True)
class PITAlphaDatasetManifest:
    dataset_id: str
    action_axis: tuple[str, ...]
    universe_rule: UniverseRule
    files: tuple[DatasetFileRecord, ...]
    source_receipts: tuple[str, ...]
    receipt_sha256: str
    schema: str = PIT_ALPHA_DATASET_SCHEMA

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "action_axis": list(self.action_axis),
            "universe_rule": {
                "rule_id": self.universe_rule.rule_id,
                "rule_sha256": self.universe_rule.rule_sha256,
                "membership_mode": self.universe_rule.membership_mode,
                "ranking_lookback_sessions": self.universe_rule.ranking_lookback_sessions,
                "ranking_lag_sessions": self.universe_rule.ranking_lag_sessions,
                "uses_future_survival": self.universe_rule.uses_future_survival,
            },
            "files": [row.to_dict() for row in self.files],
            "source_receipts": list(self.source_receipts),
        }

    def validate(self) -> None:
        if self.schema != PIT_ALPHA_DATASET_SCHEMA:
            raise PITAlphaDataError("PIT dataset schema is unsupported")
        _text("dataset ID", self.dataset_id)
        self.universe_rule.validate()
        if (
            not self.action_axis
            or self.action_axis[0] != PIT_ALPHA_CASH_ID
            or len(set(self.action_axis)) != len(self.action_axis)
        ):
            raise PITAlphaDataError("action axis must be unique with CASH fixed at index zero")
        for security_id in self.action_axis:
            _text("action-axis security ID", security_id)
        if not self.files:
            raise PITAlphaDataError("dataset manifest must bind at least one file")
        paths: list[str] = []
        for row in self.files:
            row.validate()
            paths.append(row.relative_path)
        if paths != sorted(paths) or len(set(paths)) != len(paths):
            raise PITAlphaDataError("dataset file inventory must be sorted and unique")
        if not self.source_receipts:
            raise PITAlphaDataError("dataset manifest must bind source receipts")
        for receipt in self.source_receipts:
            _digest("source receipt SHA", receipt)
        if (
            tuple(sorted(self.source_receipts)) != self.source_receipts
            or len(set(self.source_receipts)) != len(self.source_receipts)
        ):
            raise PITAlphaDataError("source receipts must be sorted and unique")
        _digest("dataset manifest receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned_payload()):
            raise PITAlphaDataError("dataset manifest receipt differs from its payload")

    @classmethod
    def build(
        cls,
        *,
        dataset_id: str,
        action_axis: Sequence[str],
        universe_rule: UniverseRule,
        files: Sequence[DatasetFileRecord],
        source_receipts: Sequence[str],
    ) -> "PITAlphaDatasetManifest":
        provisional = cls(
            dataset_id=dataset_id,
            action_axis=tuple(action_axis),
            universe_rule=universe_rule,
            files=tuple(sorted(files, key=lambda row: row.relative_path)),
            source_receipts=tuple(sorted(source_receipts)),
            receipt_sha256="0" * 64,
        )
        value = cls(
            dataset_id=provisional.dataset_id,
            action_axis=provisional.action_axis,
            universe_rule=provisional.universe_rule,
            files=provisional.files,
            source_receipts=provisional.source_receipts,
            receipt_sha256=semantic_sha256(provisional.unsigned_payload()),
        )
        value.validate()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PITAlphaDatasetManifest":
        try:
            expected_keys = {
                "schema",
                "dataset_id",
                "action_axis",
                "universe_rule",
                "files",
                "source_receipts",
                "receipt_sha256",
            }
            if set(value) != expected_keys:
                raise TypeError
            rule_value = value["universe_rule"]
            files_value = value["files"]
            if not isinstance(rule_value, Mapping) or not isinstance(files_value, list):
                raise TypeError
            expected_rule_keys = {
                "rule_id",
                "rule_sha256",
                "membership_mode",
                "ranking_lookback_sessions",
                "ranking_lag_sessions",
                "uses_future_survival",
            }
            if set(rule_value) != expected_rule_keys or not (
                isinstance(rule_value["rule_id"], str)
                and isinstance(rule_value["rule_sha256"], str)
                and isinstance(rule_value["membership_mode"], str)
                and isinstance(rule_value["ranking_lookback_sessions"], int)
                and not isinstance(rule_value["ranking_lookback_sessions"], bool)
                and isinstance(rule_value["ranking_lag_sessions"], int)
                and not isinstance(rule_value["ranking_lag_sessions"], bool)
                and isinstance(rule_value["uses_future_survival"], bool)
            ):
                raise TypeError
            rule = UniverseRule(
                rule_id=rule_value["rule_id"],
                rule_sha256=rule_value["rule_sha256"],
                membership_mode=rule_value["membership_mode"],
                ranking_lookback_sessions=rule_value["ranking_lookback_sessions"],
                ranking_lag_sessions=rule_value["ranking_lag_sessions"],
                uses_future_survival=rule_value["uses_future_survival"],
            )
            files_list: list[DatasetFileRecord] = []
            for row in files_value:
                if not isinstance(row, Mapping) or set(row) != {
                    "relative_path",
                    "size_bytes",
                    "file_sha256",
                    "media_type",
                }:
                    raise TypeError
                if not (
                    isinstance(row["relative_path"], str)
                    and isinstance(row["size_bytes"], int)
                    and not isinstance(row["size_bytes"], bool)
                    and isinstance(row["file_sha256"], str)
                    and isinstance(row["media_type"], str)
                ):
                    raise TypeError
                files_list.append(
                    DatasetFileRecord(
                        relative_path=row["relative_path"],
                        size_bytes=row["size_bytes"],
                        file_sha256=row["file_sha256"],
                        media_type=row["media_type"],
                    )
                )
            files = tuple(files_list)
            action_axis = value["action_axis"]
            source_receipts = value["source_receipts"]
            if (
                not isinstance(action_axis, list)
                or any(not isinstance(row, str) for row in action_axis)
                or not isinstance(source_receipts, list)
                or any(not isinstance(row, str) for row in source_receipts)
                or not isinstance(value["schema"], str)
                or not isinstance(value["dataset_id"], str)
                or not isinstance(value["receipt_sha256"], str)
            ):
                raise TypeError
            result = cls(
                schema=value["schema"],
                dataset_id=value["dataset_id"],
                action_axis=tuple(action_axis),
                universe_rule=rule,
                files=files,
                source_receipts=tuple(source_receipts),
                receipt_sha256=value["receipt_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PITAlphaDataError("PIT dataset manifest is malformed") from exc
        result.validate()
        return result

    def to_dict(self) -> dict[str, object]:
        return {**self.unsigned_payload(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PITAlphaDatasetAuthority:
    manifest: PITAlphaDatasetManifest
    security_master: tuple[SecurityMasterRecord, ...]
    ticker_history: tuple[TickerHistoryRecord, ...]
    membership_events: tuple[MembershipEvent, ...]
    availability: tuple[AvailabilityRecord, ...]
    corporate_actions: tuple[CorporateActionRecord, ...]
    terminal_events: tuple[TerminalEventRecord, ...]
    cash_returns: tuple[CashReturnRecord, ...]

    def validate(self) -> None:
        self.manifest.validate()
        masters: dict[str, SecurityMasterRecord] = {}
        for master_row in self.security_master:
            master_row.validate()
            if (
                master_row.security_id in masters
                or master_row.security_id == PIT_ALPHA_CASH_ID
            ):
                raise PITAlphaDataError("security master contains a duplicate/reserved ID")
            masters[master_row.security_id] = master_row
        if set(self.manifest.action_axis[1:]) != set(masters):
            raise PITAlphaDataError("every risky model action must map to one permanent security")
        _validate_ticker_history(self.ticker_history, masters)
        event_ids: set[str] = set()
        covered_event_securities: list[set[str]] = []
        for event_rows in (self.membership_events, self.availability):
            seen: set[tuple[str, int, int]] = set()
            covered: set[str] = set()
            for event_row in event_rows:
                event_row.validate()
                if event_row.security_id not in masters:
                    raise PITAlphaDataError("event references an unknown permanent security")
                covered.add(event_row.security_id)
                key = (
                    event_row.security_id,
                    event_row.effective_at_ms,
                    event_row.available_at_ms,
                )
                if key in seen:
                    raise PITAlphaDataError("event authority contains a duplicate timestamp")
                seen.add(key)
            covered_event_securities.append(covered)
        if covered_event_securities[0] != set(masters):
            raise PITAlphaDataError("every risky action requires membership history")
        if covered_event_securities[1] != set(masters):
            raise PITAlphaDataError("every risky action requires availability history")
        positive_membership = {
            row.security_id for row in self.membership_events if row.is_member
        }
        if positive_membership != set(masters):
            raise PITAlphaDataError("every risky action requires positive PIT membership")
        for action_row in self.corporate_actions:
            action_row.validate()
            if action_row.event_id in event_ids:
                raise PITAlphaDataError("economic event IDs must be globally unique")
            event_ids.add(action_row.event_id)
            _validate_economic_security_references(action_row, masters)
        terminal_by_security: dict[str, list[TerminalEventRecord]] = {}
        for terminal_row in self.terminal_events:
            terminal_row.validate()
            if terminal_row.event_id in event_ids:
                raise PITAlphaDataError("economic event IDs must be globally unique")
            event_ids.add(terminal_row.event_id)
            _validate_economic_security_references(terminal_row, masters)
            terminal_by_security.setdefault(terminal_row.security_id, []).append(
                terminal_row
            )
        for security_id, master in masters.items():
            terminals = terminal_by_security.get(security_id, [])
            if master.delisting_at_ms is not None:
                if len(terminals) != 1 or terminals[0].effective_at_ms != master.delisting_at_ms:
                    raise PITAlphaDataError(
                        "every delisted security requires one disposition at the delisting time"
                    )
            elif terminals:
                raise PITAlphaDataError("a terminal disposition requires a delisting timestamp")
        if not self.cash_returns:
            raise PITAlphaDataError("dataset authority requires causal cash returns")
        for cash_row in self.cash_returns:
            cash_row.validate()
        cash_effective = [cash_row.effective_at_ms for cash_row in self.cash_returns]
        if cash_effective != sorted(set(cash_effective)):
            raise PITAlphaDataError("cash returns must have unique chronological timestamps")


def _validate_economic_security_references(
    row: CorporateActionRecord | TerminalEventRecord,
    masters: Mapping[str, SecurityMasterRecord],
) -> None:
    if row.security_id not in masters:
        raise PITAlphaDataError("economic event references an unknown security")
    if row.successor_security_id is not None and row.successor_security_id not in masters:
        raise PITAlphaDataError("economic event references an unknown successor")


def _validate_ticker_history(
    rows: Sequence[TickerHistoryRecord],
    masters: Mapping[str, SecurityMasterRecord],
) -> None:
    by_security: dict[str, list[TickerHistoryRecord]] = {}
    for row in rows:
        row.validate()
        master = masters.get(row.security_id)
        if master is None:
            raise PITAlphaDataError("ticker history references an unknown security")
        if row.valid_from_ms < master.listing_at_ms or (
            master.delisting_at_ms is not None
            and (row.valid_to_ms or master.delisting_at_ms) > master.delisting_at_ms
        ):
            raise PITAlphaDataError("ticker interval lies outside the security lifetime")
        by_security.setdefault(row.security_id, []).append(row)
    if set(by_security) != set(masters):
        raise PITAlphaDataError("every security requires ticker history")
    for security_rows in by_security.values():
        ordered = sorted(security_rows, key=lambda row: row.valid_from_ms)
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier.valid_to_ms is None or earlier.valid_to_ms > later.valid_from_ms:
                raise PITAlphaDataError("ticker history intervals overlap")


def resolve_ticker(
    rows: Iterable[TickerHistoryRecord],
    *,
    security_id: str,
    effective_at_ms: int,
    knowledge_at_ms: int,
) -> str | None:
    """Resolve only a ticker mapping available by the requested knowledge time."""

    _text("security_id", security_id)
    _timestamp("effective_at_ms", effective_at_ms)
    _timestamp("knowledge_at_ms", knowledge_at_ms)
    matches = [
        row
        for row in rows
        if row.security_id == security_id
        and row.valid_from_ms <= effective_at_ms
        and (row.valid_to_ms is None or effective_at_ms < row.valid_to_ms)
        and row.available_at_ms <= knowledge_at_ms
    ]
    if len(matches) > 1:
        raise PITAlphaDataError("ticker authority resolves more than one mapping")
    return None if not matches else matches[0].ticker


def membership_at(
    rows: Iterable[MembershipEvent],
    *,
    security_ids: Sequence[str],
    effective_at_ms: int,
    knowledge_at_ms: int,
) -> dict[str, bool]:
    """Materialize membership without consulting later-available events."""

    return _latest_event_state(
        rows,
        security_ids=security_ids,
        effective_at_ms=effective_at_ms,
        knowledge_at_ms=knowledge_at_ms,
        default=False,
        value=lambda row: row.is_member,
    )


def availability_at(
    rows: Iterable[AvailabilityRecord],
    *,
    security_ids: Sequence[str],
    effective_at_ms: int,
    knowledge_at_ms: int,
) -> dict[str, AvailabilitySnapshot]:
    """Materialize the five availability dimensions from causal records."""

    empty = AvailabilitySnapshot(False, False, False, False, False)
    return _latest_event_state(
        rows,
        security_ids=security_ids,
        effective_at_ms=effective_at_ms,
        knowledge_at_ms=knowledge_at_ms,
        default=empty,
        value=lambda row: row.state,
    )


def _latest_event_state(
    rows: Iterable[Any],
    *,
    security_ids: Sequence[str],
    effective_at_ms: int,
    knowledge_at_ms: int,
    default: Any,
    value: Any,
) -> dict[str, Any]:
    _timestamp("effective_at_ms", effective_at_ms)
    _timestamp("knowledge_at_ms", knowledge_at_ms)
    if len(set(security_ids)) != len(security_ids):
        raise PITAlphaDataError("requested security IDs must be unique")
    result = {security_id: default for security_id in security_ids}
    latest: dict[str, tuple[int, int]] = {}
    for row in rows:
        if row.security_id not in result:
            continue
        row.validate()
        if row.effective_at_ms > effective_at_ms or row.available_at_ms > knowledge_at_ms:
            continue
        key = (row.effective_at_ms, row.available_at_ms)
        previous = latest.get(row.security_id)
        if previous is not None and key == previous:
            raise PITAlphaDataError("point-in-time authority has conflicting duplicate events")
        if previous is None or key > previous:
            latest[row.security_id] = key
            result[row.security_id] = value(row)
    return result


def _read_exact_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PITAlphaDataError("dataset inventory member is not a regular file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
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
            raise PITAlphaDataError("dataset file changed while it was read")
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def validate_manifest_files(
    root: str | Path,
    manifest: PITAlphaDatasetManifest,
    *,
    allowed_unlisted: Sequence[str] = ("manifest.json",),
) -> None:
    """Verify the exact regular-file inventory and byte hashes under *root*."""

    manifest.validate()
    dataset_root = Path(root)
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise PITAlphaDataError("dataset root must be a non-symlink directory")
    expected = {row.relative_path: row for row in manifest.files}
    allowed = {_safe_relative_path(row) for row in allowed_unlisted}
    observed: set[str] = set()
    for path in dataset_root.rglob("*"):
        relative = path.relative_to(dataset_root).as_posix()
        if path.is_symlink():
            raise PITAlphaDataError("dataset tree contains a symbolic link")
        if path.is_dir():
            continue
        if relative not in expected and relative not in allowed:
            raise PITAlphaDataError(f"dataset tree contains unmanifested file {relative!r}")
        observed.add(relative)
    missing = set(expected) - observed
    if missing:
        raise PITAlphaDataError(f"dataset tree is missing files: {sorted(missing)}")
    for relative, record in expected.items():
        raw, metadata = _read_exact_regular_file(dataset_root / relative)
        if metadata.st_size != record.size_bytes or hashlib.sha256(raw).hexdigest() != record.file_sha256:
            raise PITAlphaDataError(f"dataset file bytes drifted: {relative}")


def write_pit_alpha_manifest(
    path: str | Path,
    manifest: PITAlphaDatasetManifest,
) -> str:
    """Publish one canonical manifest with create-only semantics."""

    manifest.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_file_bytes(manifest.to_dict())
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def load_pit_alpha_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PITAlphaDatasetManifest:
    """Load a canonical manifest from one no-follow descriptor."""

    _digest("expected manifest file SHA", expected_file_sha256)
    raw, _ = _read_exact_regular_file(Path(path))
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise PITAlphaDataError("manifest file SHA differs")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PITAlphaDataError("manifest JSON is malformed") from exc
    if not isinstance(value, dict) or raw != canonical_json_file_bytes(value):
        raise PITAlphaDataError("manifest file is not canonical JSON")
    return PITAlphaDatasetManifest.from_dict(value)


__all__ = [
    "AvailabilityRecord",
    "AvailabilitySnapshot",
    "CashReturnRecord",
    "CorporateActionKind",
    "CorporateActionRecord",
    "DatasetFileRecord",
    "MembershipEvent",
    "PIT_ALPHA_CASH_ID",
    "PIT_ALPHA_DATASET_SCHEMA",
    "PITAlphaDataError",
    "PITAlphaDatasetAuthority",
    "PITAlphaDatasetManifest",
    "SecurityMasterRecord",
    "TerminalEventKind",
    "TerminalEventRecord",
    "TickerHistoryRecord",
    "UniverseRule",
    "availability_at",
    "load_pit_alpha_manifest",
    "membership_at",
    "resolve_ticker",
    "validate_manifest_files",
    "write_pit_alpha_manifest",
]
