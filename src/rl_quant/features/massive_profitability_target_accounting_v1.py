"""Diagnostic fill and terminal-complete target accounting for profitability P0."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.alpha.accounting import (
    EconomicPosition,
    EconomicValuePoint,
    apply_corporate_action,
    compute_post_fill_total_return,
    mark_position,
)
from rl_quant.alpha.contracts import CorporateActionRecord, TerminalEventRecord
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    load_massive_persisted_security_rows_v2,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_profitability_origin_v1 import (
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_FILL_WINDOW_V1_SCHEMA = (
    "rl-quant.massive-profitability-fill-window-v1"
)
MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SCHEMA = (
    "rl-quant.massive-profitability-target-accounting-v1"
)
MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "entry_and_exit": "qualifying-trade-VWAP-[15:50,16:00)-ET",
        "path": "complete-fill-to-fill-economic-values-through-H63",
        "terminal": "explicit-carry",
        "unresolved_delisting": "conservative-total-loss",
        "cash": "zero-return",
        "missing_live_fill": "invalid-never-shifted",
        "horizons": (1, 5, 21, 63),
        "production_equivalence": False,
    }
)


class MassiveProfitabilityTargetAccountingV1Error(ValueError):
    """Fill or target accounting differs from the frozen P0 contract."""


_EASTERN = ZoneInfo("America/New_York")


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTargetAccountingV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(name: str, value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise MassiveProfitabilityTargetAccountingV1Error(f"{name} must be finite")
    return float(value)


def _validate_fill_clock(session_date: str, start_at_ms: int, end_at_ms: int) -> None:
    start = datetime.fromtimestamp(start_at_ms / 1_000, tz=_EASTERN)
    end = datetime.fromtimestamp(end_at_ms / 1_000, tz=_EASTERN)
    if (
        start.date().isoformat() != session_date
        or end.date().isoformat() != session_date
        or start.timetz().replace(tzinfo=None) != time(15, 50)
        or end.timetz().replace(tzinfo=None) != time(16, 0)
    ):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "fill window is not [15:50,16:00) America/New_York"
        )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityQualifyingFillTradeV1:
    security_id: str
    participant_at_ms: int
    price: float
    size: float
    terminal_active: bool
    volume_forming: bool
    source_trade_receipt_sha256: str

    def validate(self) -> None:
        if (
            not self.security_id
            or isinstance(self.participant_at_ms, bool)
            or not isinstance(self.participant_at_ms, int)
            or self.participant_at_ms < 0
            or _finite("fill price", self.price) <= 0.0
            or _finite("fill size", self.size) <= 0.0
            or not isinstance(self.terminal_active, bool)
            or not isinstance(self.volume_forming, bool)
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "qualifying fill trade differs"
            )
        _digest("fill trade", self.source_trade_receipt_sha256)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFillWindowRowV1:
    session_date: str
    security_id: str
    fill_start_at_ms: int
    fill_end_at_ms: int
    fill_vwap: float
    qualifying_share_volume: float
    qualifying_dollar_volume: float
    qualifying_trade_count: int
    valid: bool
    trade_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _validate_fill_clock(
            self.session_date, self.fill_start_at_ms, self.fill_end_at_ms
        )
        if (
            not self.session_date
            or not self.security_id
            or isinstance(self.fill_start_at_ms, bool)
            or not isinstance(self.fill_start_at_ms, int)
            or isinstance(self.fill_end_at_ms, bool)
            or not isinstance(self.fill_end_at_ms, int)
            or self.fill_end_at_ms <= self.fill_start_at_ms
            or not isinstance(self.valid, bool)
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "fill-window row identity differs"
            )
        vwap = _finite("fill VWAP", self.fill_vwap)
        shares = _finite("fill share volume", self.qualifying_share_volume)
        dollars = _finite("fill dollar volume", self.qualifying_dollar_volume)
        if (
            isinstance(self.qualifying_trade_count, bool)
            or not isinstance(self.qualifying_trade_count, int)
            or self.qualifying_trade_count < 0
            or (
                self.valid
                and (
                    self.qualifying_trade_count <= 0
                    or shares <= 0.0
                    or dollars <= 0.0
                    or vwap <= 0.0
                    or not math.isclose(
                        vwap, dollars / shares, rel_tol=0.0, abs_tol=1e-12
                    )
                )
            )
            or (
                not self.valid
                and any(
                    value != 0
                    for value in (vwap, shares, dollars, self.qualifying_trade_count)
                )
            )
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "fill-window values or mask differ"
            )
        _digest("fill trade inventory", self.trade_inventory_sha256)
        _digest("fill-window row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "fill-window row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFillWindowV1:
    session_date: str
    fill_start_at_ms: int
    fill_end_at_ms: int
    security_ids: tuple[str, ...]
    rows: tuple[MassiveProfitabilityFillWindowRowV1, ...]
    row_inventory_sha256: str
    source_partition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_FILL_WINDOW_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_FILL_WINDOW_V1_SCHEMA
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or tuple(row.security_id for row in self.rows) != self.security_ids
            or not isinstance(self.source_data_qualified, bool)
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "fill-window support differs"
            )
        for row in self.rows:
            row.validate()
            if (
                row.session_date != self.session_date
                or row.fill_start_at_ms != self.fill_start_at_ms
                or row.fill_end_at_ms != self.fill_end_at_ms
            ):
                raise MassiveProfitabilityTargetAccountingV1Error(
                    "fill-window row chronology differs"
                )
        for value in (
            self.row_inventory_sha256,
            self.source_partition_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("fill-window digest", value)
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "fill-window receipt differs"
            )


def _build_massive_profitability_fill_window_v1(
    *,
    session_date: str,
    fill_start_at_ms: int,
    fill_end_at_ms: int,
    security_ids: Sequence[str],
    trades: Sequence[MassiveProfitabilityQualifyingFillTradeV1],
    source_partition_receipts: Sequence[str],
    source_data_qualified: bool,
) -> MassiveProfitabilityFillWindowV1:
    """Compute a fixed-window VWAP without moving an unavailable fill."""

    members = tuple(sorted(set(security_ids)))
    if not members:
        raise MassiveProfitabilityTargetAccountingV1Error(
            "fill window requires security support"
        )
    by_security: dict[str, list[MassiveProfitabilityQualifyingFillTradeV1]] = {
        security_id: [] for security_id in members
    }
    for trade in trades:
        trade.validate()
        if trade.security_id not in by_security:
            continue
        if (
            fill_start_at_ms <= trade.participant_at_ms < fill_end_at_ms
            and trade.terminal_active
            and trade.volume_forming
        ):
            by_security[trade.security_id].append(trade)
    rows: list[MassiveProfitabilityFillWindowRowV1] = []
    for security_id in members:
        selected = tuple(
            sorted(
                by_security[security_id],
                key=lambda row: (
                    row.participant_at_ms,
                    row.source_trade_receipt_sha256,
                ),
            )
        )
        shares = sum((trade.size for trade in selected), 0.0)
        dollars = sum((trade.price * trade.size for trade in selected), 0.0)
        valid = bool(selected) and shares > 0.0 and dollars > 0.0
        body = {
            "session_date": session_date,
            "security_id": security_id,
            "fill_start_at_ms": fill_start_at_ms,
            "fill_end_at_ms": fill_end_at_ms,
            "fill_vwap": dollars / shares if valid else 0.0,
            "qualifying_share_volume": shares if valid else 0.0,
            "qualifying_dollar_volume": dollars if valid else 0.0,
            "qualifying_trade_count": len(selected) if valid else 0,
            "valid": valid,
            "trade_inventory_sha256": semantic_sha256(
                tuple(trade.source_trade_receipt_sha256 for trade in selected)
            ),
        }
        row = MassiveProfitabilityFillWindowRowV1(
            **body, receipt_sha256=semantic_sha256(body)
        )
        row.validate()
        rows.append(row)
    partitions = tuple(
        sorted(_digest("fill partition", value) for value in source_partition_receipts)
    )
    semantic = {
        "session_date": session_date,
        "fill_start_at_ms": fill_start_at_ms,
        "fill_end_at_ms": fill_end_at_ms,
        "security_ids": members,
        "rows": tuple(asdict(row) for row in rows),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "source_partition_inventory_sha256": semantic_sha256(partitions),
        "source_data_qualified": source_data_qualified,
        "schema": MASSIVE_PROFITABILITY_FILL_WINDOW_V1_SCHEMA,
    }
    result = MassiveProfitabilityFillWindowV1(
        session_date=session_date,
        fill_start_at_ms=fill_start_at_ms,
        fill_end_at_ms=fill_end_at_ms,
        security_ids=members,
        rows=tuple(rows),
        row_inventory_sha256=semantic["row_inventory_sha256"],  # type: ignore[arg-type]
        source_partition_inventory_sha256=semantic["source_partition_inventory_sha256"],  # type: ignore[arg-type]
        source_data_qualified=source_data_qualified,
        semantic_receipt_sha256=semantic_sha256(semantic),
    )
    result.validate()
    return result


def build_massive_profitability_fill_window_v1(
    *,
    session_date: str,
    fill_start_at_ms: int,
    fill_end_at_ms: int,
    security_ids: Sequence[str],
    trades: Sequence[MassiveProfitabilityQualifyingFillTradeV1],
    source_partition_receipts: Sequence[str],
) -> MassiveProfitabilityFillWindowV1:
    """Build a deterministic canary fill; caller rows never authorize data."""

    return _build_massive_profitability_fill_window_v1(
        session_date=session_date,
        fill_start_at_ms=fill_start_at_ms,
        fill_end_at_ms=fill_end_at_ms,
        security_ids=security_ids,
        trades=trades,
        source_partition_receipts=source_partition_receipts,
        source_data_qualified=False,
    )


def derive_massive_profitability_fill_window_v1(
    *,
    persisted_root: str | Path,
    manifest: MassivePersistedPartitionManifestV1,
    condition_authority: MassiveConditionAuthority,
    session_date: str,
    fill_start_at_ms: int,
    fill_end_at_ms: int,
    security_ids: Sequence[str],
) -> MassiveProfitabilityFillWindowV1:
    """Rederive the VWAP from committed terminal-active regular partitions."""

    manifest.validate()
    condition_authority.validate()
    if manifest.source_session_date != session_date:
        raise MassiveProfitabilityTargetAccountingV1Error(
            "fill manifest and requested session differ"
        )
    support = set(security_ids)
    trades: list[MassiveProfitabilityQualifyingFillTradeV1] = []
    partition_receipts: list[str] = []
    for partition in manifest.partitions:
        if partition.security_id not in support:
            continue
        _, active, _ = load_massive_persisted_security_rows_v2(
            root=persisted_root, partition=partition
        )
        partition_receipts.append(partition.receipt_sha256)
        for row in active:
            trades.append(
                MassiveProfitabilityQualifyingFillTradeV1(
                    security_id=partition.security_id,
                    participant_at_ms=(
                        row.canonical_record.participant_timestamp_ns // 1_000_000
                    ),
                    price=float(row.canonical_record.price_decimal),
                    size=float(row.canonical_record.size_decimal),
                    terminal_active=True,
                    volume_forming=condition_authority.resolve(
                        row.canonical_record.conditions
                    )[2],
                    source_trade_receipt_sha256=row.receipt_sha256,
                )
            )
    return _build_massive_profitability_fill_window_v1(
        session_date=session_date,
        fill_start_at_ms=fill_start_at_ms,
        fill_end_at_ms=fill_end_at_ms,
        security_ids=security_ids,
        trades=trades,
        source_partition_receipts=partition_receipts,
        source_data_qualified=True,
    )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTargetEconomicPathRowV1:
    security_id: str
    economic_at_ms: tuple[int, ...]
    available_at_ms: tuple[int, ...]
    values: tuple[float, ...]
    valid: tuple[bool, ...]
    terminal: tuple[bool, ...]
    mark_kinds: tuple[str, ...]
    mark_receipts: tuple[str, ...]
    unresolved_terminal_fallback_session_offset: int | None
    conservative_total_loss_fallback: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.security_id or any(
            len(values) != 64
            for values in (
                self.values,
                self.valid,
                self.terminal,
                self.mark_kinds,
                self.mark_receipts,
                self.economic_at_ms,
                self.available_at_ms,
            )
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target path shape differs"
            )
        if any(not isinstance(value, bool) for value in self.valid + self.terminal):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target path masks differ"
            )
        if (
            self.economic_at_ms != tuple(sorted(set(self.economic_at_ms)))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.economic_at_ms + self.available_at_ms
            )
            or any(
                available < economic
                for economic, available in zip(
                    self.economic_at_ms, self.available_at_ms, strict=True
                )
            )
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target path economic or availability times differ"
            )
        if any(
            not math.isfinite(float(value))
            or value < 0.0
            or (not valid and value != 0.0)
            for value, valid in zip(self.values, self.valid, strict=True)
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target path values differ"
            )
        terminal_seen = False
        terminal_value = 0.0
        for value, valid, terminal, kind, receipt in zip(
            self.values,
            self.valid,
            self.terminal,
            self.mark_kinds,
            self.mark_receipts,
            strict=True,
        ):
            _digest("target path mark", receipt)
            if terminal:
                if not valid or kind != "terminal-disposition":
                    raise MassiveProfitabilityTargetAccountingV1Error(
                        "target terminal point differs"
                    )
                if not terminal_seen:
                    terminal_seen = True
                    terminal_value = value
                elif value != terminal_value:
                    raise MassiveProfitabilityTargetAccountingV1Error(
                        "target terminal carry differs"
                    )
            elif terminal_seen:
                raise MassiveProfitabilityTargetAccountingV1Error(
                    "target path stops carrying a terminal disposition"
                )
            elif valid and kind not in {"market", "validated-fallback"}:
                raise MassiveProfitabilityTargetAccountingV1Error(
                    "target live mark kind differs"
                )
        fallback = self.unresolved_terminal_fallback_session_offset
        if fallback is not None and (
            isinstance(fallback, bool)
            or not isinstance(fallback, int)
            or not 1 <= fallback <= 63
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "conservative terminal fallback offset differs"
            )
        if self.conservative_total_loss_fallback != (fallback is not None):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "conservative terminal fallback state differs"
            )
        _digest("target path", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target path receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTargetAccountingV1:
    origin_receipt_sha256: str
    decision_session_date: str
    session_dates: tuple[str, ...]
    entry_fill_receipt_sha256: str
    exit_fill_receipts: tuple[str, ...]
    rows: tuple[MassiveProfitabilityTargetEconomicPathRowV1, ...]
    event_receipts: tuple[str, ...]
    terminal_inventory_sha256: str
    economic_coverage_semantic_receipt_sha256: str
    row_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    economic_coverage_audit_receipt_sha256: str
    audit_receipt_sha256: str
    fill_sources_qualified: bool
    economic_values_data_qualified: bool
    schema: str = MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "economic_coverage_audit_receipt_sha256",
                "audit_receipt_sha256",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SOURCE_SHA256
            or len(self.session_dates) != 64
            or self.session_dates != tuple(sorted(set(self.session_dates)))
            or self.session_dates[0] != self.decision_session_date
            or not isinstance(self.fill_sources_qualified, bool)
            or not isinstance(self.economic_values_data_qualified, bool)
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target accounting identity or interval differs"
            )
        keys = tuple(row.security_id for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target accounting support differs"
            )
        for row in self.rows:
            row.validate()
        if self.event_receipts != tuple(sorted(set(self.event_receipts))):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target event inventory is not canonical"
            )
        for value in (
            self.origin_receipt_sha256,
            self.entry_fill_receipt_sha256,
            self.terminal_inventory_sha256,
            self.economic_coverage_semantic_receipt_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.economic_coverage_audit_receipt_sha256,
            self.audit_receipt_sha256,
            *self.exit_fill_receipts,
            *self.event_receipts,
        ):
            _digest("target accounting digest", value)
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target accounting semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "economic_coverage_audit_receipt_sha256": (
                    self.economic_coverage_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target accounting audit receipt differs"
            )


def build_massive_profitability_target_accounting_v1(
    *,
    origin: MassiveProfitabilityDecisionOriginV1,
    session_authority: MassiveSessionAuthority,
    entry_fill: MassiveProfitabilityFillWindowV1,
    exit_fills_by_offset: Mapping[int, MassiveProfitabilityFillWindowV1],
    economic_values: Mapping[tuple[str, int], float | None],
    mark_kinds: Mapping[tuple[str, int], str],
    mark_receipts: Mapping[tuple[str, int], str],
    terminal_offsets: Mapping[str, int] | None = None,
    unresolved_delisting_offsets: Mapping[str, int] | None = None,
    event_receipts: Sequence[str] = (),
    terminal_inventory_sha256: str,
    economic_coverage_semantic_receipt_sha256: str,
    economic_coverage_audit_receipt_sha256: str,
) -> MassiveProfitabilityTargetAccountingV1:
    """Build a complete decision-fill through H63 economic path."""

    origin.validate()
    session_authority.validate()
    entry_fill.validate()
    if (
        entry_fill.session_date != origin.decision_session_date
        or entry_fill.fill_start_at_ms != origin.fill_start_at_ms
        or entry_fill.fill_end_at_ms != origin.fill_end_at_ms
    ):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "entry fill differs from the decision origin"
        )
    sessions = tuple(session_authority.sessions)
    by_date = {row.session_date: index for index, row in enumerate(sessions)}
    decision_index = by_date.get(origin.decision_session_date)
    if decision_index is None or decision_index + 63 >= len(sessions):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "decision does not have a complete H63 session path"
        )
    selected_sessions = sessions[decision_index : decision_index + 64]
    session_dates = tuple(row.session_date for row in selected_sessions)
    required_exit_offsets = (1, 5, 21, 63)
    if set(exit_fills_by_offset) != set(required_exit_offsets):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "target accounting requires H1/H5/H21/H63 exit fills"
        )
    for offset, fill in exit_fills_by_offset.items():
        fill.validate()
        if fill.session_date != session_dates[offset]:
            raise MassiveProfitabilityTargetAccountingV1Error(
                "exit fill is not on the exact horizon session"
            )
    security_ids = entry_fill.security_ids
    if any(fill.security_ids != security_ids for fill in exit_fills_by_offset.values()):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "entry and exit fill support differs"
        )
    expected = {
        (security_id, offset) for security_id in security_ids for offset in range(64)
    }
    if (
        set(economic_values) != expected
        or set(mark_kinds) != expected
        or set(mark_receipts) != expected
    ):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "target economic paths do not exactly cover fill through H63"
        )
    terminal_offsets = dict(terminal_offsets or {})
    unresolved_offsets = dict(unresolved_delisting_offsets or {})
    if set(terminal_offsets) & set(unresolved_offsets) or not (
        set(terminal_offsets) | set(unresolved_offsets)
    ) <= set(security_ids):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "terminal path inventories differ"
        )
    entry_by_security = {row.security_id: row for row in entry_fill.rows}
    exit_by_offset = {
        offset: {row.security_id: row for row in fill.rows}
        for offset, fill in exit_fills_by_offset.items()
    }
    rows: list[MassiveProfitabilityTargetEconomicPathRowV1] = []
    for security_id in security_ids:
        fallback = unresolved_offsets.get(security_id)
        known_terminal = terminal_offsets.get(security_id)
        if fallback is not None and not 1 <= fallback <= 63:
            raise MassiveProfitabilityTargetAccountingV1Error(
                "unresolved terminal fallback is outside the target interval"
            )
        if known_terminal is not None and not 1 <= known_terminal <= 63:
            raise MassiveProfitabilityTargetAccountingV1Error(
                "known terminal disposition is outside the target interval"
            )
        values: list[float] = []
        valid: list[bool] = []
        terminal: list[bool] = []
        kinds: list[str] = []
        receipts: list[str] = []
        terminal_start = fallback if fallback is not None else known_terminal
        terminal_value: float | None = None
        for offset in range(64):
            key = (security_id, offset)
            value = economic_values[key]
            kind = mark_kinds[key]
            receipt = _digest("target mark", mark_receipts[key])
            if offset == 0:
                fill_row = entry_by_security[security_id]
                if fill_row.valid:
                    value = fill_row.fill_vwap
                    kind = "market"
                    receipt = fill_row.receipt_sha256
                else:
                    value = None
            elif offset in exit_by_offset:
                fill_row = exit_by_offset[offset][security_id]
                if terminal_start is None or offset < terminal_start:
                    if not fill_row.valid:
                        value = None
                    elif value is not None:
                        # The economic value may include split-adjusted share
                        # quantities, successor holdings, and cash.  The raw
                        # exit VWAP is evidence for the mark, never a direct
                        # replacement for that complete position value.
                        kind = "market"
            is_terminal = terminal_start is not None and offset >= terminal_start
            if is_terminal:
                if fallback is not None:
                    value = 0.0
                elif terminal_value is None:
                    if value is None:
                        raise MassiveProfitabilityTargetAccountingV1Error(
                            "known terminal disposition lacks an economic value"
                        )
                    terminal_value = float(value)
                else:
                    value = terminal_value
                kind = "terminal-disposition"
            values.append(0.0 if value is None else float(value))
            valid.append(value is not None)
            terminal.append(is_terminal)
            kinds.append(kind)
            receipts.append(receipt)
        body = {
            "security_id": security_id,
            "economic_at_ms": tuple(
                session.regular_close_ns // 1_000_000 for session in selected_sessions
            ),
            "available_at_ms": tuple(
                session.regular_close_ns // 1_000_000 for session in selected_sessions
            ),
            "values": tuple(values),
            "valid": tuple(valid),
            "terminal": tuple(terminal),
            "mark_kinds": tuple(kinds),
            "mark_receipts": tuple(receipts),
            "unresolved_terminal_fallback_session_offset": fallback,
            "conservative_total_loss_fallback": fallback is not None,
        }
        row = MassiveProfitabilityTargetEconomicPathRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        rows.append(row)
    ordered_events = tuple(
        sorted({_digest("target event", value) for value in event_receipts})
    )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    semantic = {
        "origin_receipt_sha256": origin.receipt_sha256,
        "decision_session_date": origin.decision_session_date,
        "session_dates": session_dates,
        "entry_fill_receipt_sha256": entry_fill.semantic_receipt_sha256,
        "exit_fill_receipts": tuple(
            exit_fills_by_offset[offset].semantic_receipt_sha256
            for offset in required_exit_offsets
        ),
        "rows": tuple(asdict(row) for row in rows),
        "event_receipts": ordered_events,
        "terminal_inventory_sha256": _digest(
            "terminal inventory", terminal_inventory_sha256
        ),
        "economic_coverage_semantic_receipt_sha256": _digest(
            "target economic coverage", economic_coverage_semantic_receipt_sha256
        ),
        "row_inventory_sha256": row_inventory,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SOURCE_SHA256,
        "fill_sources_qualified": entry_fill.source_data_qualified
        and all(fill.source_data_qualified for fill in exit_fills_by_offset.values()),
        "economic_values_data_qualified": False,
        "schema": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SCHEMA,
    }
    semantic_receipt = semantic_sha256(semantic)
    audit = _digest("target economic audit", economic_coverage_audit_receipt_sha256)
    result = MassiveProfitabilityTargetAccountingV1(
        origin_receipt_sha256=origin.receipt_sha256,
        decision_session_date=origin.decision_session_date,
        session_dates=session_dates,
        entry_fill_receipt_sha256=entry_fill.semantic_receipt_sha256,
        exit_fill_receipts=semantic["exit_fill_receipts"],  # type: ignore[arg-type]
        rows=tuple(rows),
        event_receipts=ordered_events,
        terminal_inventory_sha256=terminal_inventory_sha256,
        economic_coverage_semantic_receipt_sha256=economic_coverage_semantic_receipt_sha256,
        row_inventory_sha256=row_inventory,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        economic_coverage_audit_receipt_sha256=audit,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "economic_coverage_audit_receipt_sha256": audit,
            }
        ),
        fill_sources_qualified=semantic["fill_sources_qualified"],  # type: ignore[arg-type]
        economic_values_data_qualified=False,
    )
    result.validate()
    return result


def replay_massive_profitability_economic_inputs_v1(
    *,
    origin_security_ids: Sequence[str],
    economic_at_ms: Sequence[int],
    marks: Mapping[tuple[int, str], float],
    mark_receipts: Mapping[tuple[int, str], str],
    events: Sequence[CorporateActionRecord | TerminalEventRecord] = (),
) -> tuple[
    dict[tuple[str, int], float | None],
    dict[tuple[str, int], str],
    dict[tuple[str, int], str],
    dict[str, int],
]:
    """Replay lower-level position accounting into complete per-origin paths.

    Events are processed only when their affected security is held at that
    chronological point.  An earlier successor action is therefore excluded
    before a later acquisition of that successor.  Same-time ties fail closed
    because the lower-level records do not carry an economic sequence.
    """

    origins = tuple(sorted(set(origin_security_ids)))
    times = tuple(economic_at_ms)
    if (
        not origins
        or len(times) != 64
        or times != tuple(sorted(set(times)))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in times
        )
    ):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "economic replay support or time coordinates differ"
        )
    ordered_events = tuple(
        sorted(events, key=lambda row: (row.effective_at_ms, row.event_id))
    )
    event_times = tuple(row.effective_at_ms for row in ordered_events)
    if len(event_times) != len(set(event_times)):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "same-time economic events require an explicit qualified order"
        )
    for event in ordered_events:
        event.validate()
        if event.effective_at_ms <= times[0] or event.effective_at_ms > times[-1]:
            raise MassiveProfitabilityTargetAccountingV1Error(
                "target event lies outside the post-fill economic interval"
            )
    if set(mark_receipts) != set(marks):
        raise MassiveProfitabilityTargetAccountingV1Error(
            "economic marks and mark receipts differ"
        )
    for receipt in mark_receipts.values():
        _digest("economic replay mark", receipt)
    values: dict[tuple[str, int], float | None] = {}
    kinds: dict[tuple[str, int], str] = {}
    receipts: dict[tuple[str, int], str] = {}
    terminal_offsets: dict[str, int] = {}
    for origin in origins:
        position = EconomicPosition.from_mapping({origin: 1.0})
        event_index = 0
        excluded_events: list[str] = []
        applied_events: list[str] = []
        terminal_value: float | None = None
        for offset, economic_time in enumerate(times):
            while (
                event_index < len(ordered_events)
                and ordered_events[event_index].effective_at_ms <= economic_time
            ):
                event = ordered_events[event_index]
                if event.security_id in position.as_mapping():
                    position = apply_corporate_action(position, event)
                    applied_events.append(event.event_id)
                else:
                    excluded_events.append(event.event_id)
                event_index += 1
            held = position.as_mapping()
            mark_map = {
                security_id: float(marks[(offset, security_id)])
                for security_id in held
                if (offset, security_id) in marks
            }
            key = (origin, offset)
            missing = set(held) - set(mark_map)
            if missing:
                values[key] = None
                kinds[key] = "market"
            else:
                value = mark_position(position, mark_map)
                values[key] = value
                if not held:
                    if terminal_value is None:
                        terminal_value = value
                        terminal_offsets[origin] = offset
                    elif value != terminal_value:
                        raise MassiveProfitabilityTargetAccountingV1Error(
                            "cash terminal value did not carry exactly"
                        )
                    kinds[key] = "terminal-disposition"
                else:
                    kinds[key] = "market"
            receipts[key] = semantic_sha256(
                {
                    "origin_security_id": origin,
                    "session_offset": offset,
                    "position": asdict(position),
                    "mark_receipts": tuple(
                        sorted(
                            mark_receipts[(offset, security_id)]
                            for security_id in held
                            if (offset, security_id) in mark_receipts
                        )
                    ),
                    "applied_event_ids": tuple(applied_events),
                    "excluded_event_ids": tuple(excluded_events),
                }
            )
    return values, kinds, receipts, terminal_offsets


def target_points_for_security_v1(
    *,
    accounting: MassiveProfitabilityTargetAccountingV1,
    security_id: str,
) -> tuple[EconomicValuePoint, ...]:
    """Expose a validated lower-level accounting path for target generation."""

    accounting.validate()
    row = next(
        (
            candidate
            for candidate in accounting.rows
            if candidate.security_id == security_id
        ),
        None,
    )
    if row is None:
        raise MassiveProfitabilityTargetAccountingV1Error(
            "security is absent from target accounting"
        )
    # Fill-window times are monotonically encoded by session position.  The
    # values are finalized accounting and therefore may become available after
    # their economic times without becoming predictive inputs.
    points = tuple(
        EconomicValuePoint(
            session_index=offset,
            economic_at_ms=row.economic_at_ms[offset],
            available_at_ms=row.available_at_ms[offset],
            value=row.values[offset],
            mark_kind=row.mark_kinds[offset] if row.valid[offset] else "market",
            terminal=row.terminal[offset],
        )
        for offset in range(64)
        if row.valid[offset]
    )
    for point in points:
        point.validate()
    return points


def compute_massive_profitability_target_v1(
    *,
    accounting: MassiveProfitabilityTargetAccountingV1,
    security_id: str,
    horizon_sessions: int,
) -> float:
    """Compute one target with the repository's terminal-carry implementation."""

    target = compute_post_fill_total_return(
        target_points_for_security_v1(accounting=accounting, security_id=security_id),
        fill_session_index=0,
        horizon_sessions=horizon_sessions,
    )
    return target.simple_return


__all__ = [
    "MASSIVE_PROFITABILITY_FILL_WINDOW_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SPEC_SHA256",
    "MassiveProfitabilityFillWindowRowV1",
    "MassiveProfitabilityFillWindowV1",
    "MassiveProfitabilityQualifyingFillTradeV1",
    "MassiveProfitabilityTargetAccountingV1",
    "MassiveProfitabilityTargetAccountingV1Error",
    "MassiveProfitabilityTargetEconomicPathRowV1",
    "build_massive_profitability_fill_window_v1",
    "build_massive_profitability_target_accounting_v1",
    "compute_massive_profitability_target_v1",
    "derive_massive_profitability_fill_window_v1",
    "replay_massive_profitability_economic_inputs_v1",
    "target_points_for_security_v1",
]
