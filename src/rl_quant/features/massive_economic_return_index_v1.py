"""Causal corporate-action-complete value index for profitability P0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from rl_quant.alpha.accounting import (
    EconomicPosition,
    apply_cash_return,
    apply_corporate_action,
    mark_position,
)
from rl_quant.alpha.contracts import (
    CashReturnRecord,
    CorporateActionRecord,
    TerminalEventRecord,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
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


MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA = "rl-quant.massive-economic-return-index-v1"
MASSIVE_ECONOMIC_RETURN_INDEX_V1_DATASET = "massive-finalized-economic-return-index-v1"
MASSIVE_ECONOMIC_RETURN_INDEX_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_RETURN_INDEX_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "session_panel_spec": MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
        "initial_position": "one-share-of-origin-permanent-security",
        "event_rule": "effective-and-available-by-session-close-exactly-once",
        "events": (
            "cash-dividends",
            "special-dividends",
            "splits-and-reverse-splits",
            "spin-offs",
            "cash-and-stock-mergers",
            "tenders",
            "rights",
            "return-of-capital",
            "terminal-dispositions",
            "causal-cash-return",
        ),
        "mark": "terminal-active-regular-close-by-permanent-security",
        "missing_mark": "invalid-never-zero-imputation",
        "terminal": "cash-or-zero-carried-without-future-survival",
    }
)
MASSIVE_ECONOMIC_RETURN_INDEX_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA,
        "row_key": ("source_session_index", "security_id"),
    }
)


class MassiveEconomicReturnIndexV1Error(ValueError):
    """Economic authority, chronology, values, or committed bytes differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicReturnIndexV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveEconomicEventAuthorityV1:
    security_ids: tuple[str, ...]
    corporate_actions: tuple[CorporateActionRecord, ...]
    terminal_events: tuple[TerminalEventRecord, ...]
    cash_returns: tuple[CashReturnRecord, ...]
    source_receipts: tuple[str, ...]
    event_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or any(not value or value != value.strip() for value in self.security_ids)
        ):
            raise MassiveEconomicReturnIndexV1Error(
                "economic security inventory is not canonical"
            )
        event_ids: set[str] = set()
        event_receipts: list[str] = []
        events: list[CorporateActionRecord | TerminalEventRecord] = [
            *self.corporate_actions,
            *self.terminal_events,
        ]
        for event in events:
            event.validate()
            if event.event_id in event_ids:
                raise MassiveEconomicReturnIndexV1Error(
                    "economic event IDs are not unique"
                )
            event_ids.add(event.event_id)
            if event.security_id not in self.security_ids or (
                event.successor_security_id is not None
                and event.successor_security_id not in self.security_ids
            ):
                raise MassiveEconomicReturnIndexV1Error(
                    "economic event references an unknown permanent security"
                )
            event_receipts.append(semantic_sha256(asdict(event)))
        cash_times: list[int] = []
        for cash in self.cash_returns:
            cash.validate()
            cash_times.append(cash.effective_at_ms)
            event_receipts.append(semantic_sha256(asdict(cash)))
        if cash_times != sorted(set(cash_times)):
            raise MassiveEconomicReturnIndexV1Error(
                "cash-return observations are not chronological and unique"
            )
        if not self.source_receipts or self.source_receipts != tuple(
            sorted(set(self.source_receipts))
        ):
            raise MassiveEconomicReturnIndexV1Error(
                "economic source receipts are not canonical"
            )
        for receipt in (*self.source_receipts, self.event_inventory_sha256):
            _digest("economic authority digest", receipt)
        if self.event_inventory_sha256 != semantic_sha256(tuple(event_receipts)):
            raise MassiveEconomicReturnIndexV1Error("economic event inventory differs")
        _digest("economic authority receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicReturnIndexV1Error(
                "economic authority receipt differs"
            )


def build_massive_economic_event_authority_v1(
    *,
    security_ids: Sequence[str],
    corporate_actions: Sequence[CorporateActionRecord],
    terminal_events: Sequence[TerminalEventRecord],
    cash_returns: Sequence[CashReturnRecord],
    source_receipts: Sequence[str],
) -> MassiveEconomicEventAuthorityV1:
    securities = tuple(sorted(security_ids))
    corporate = tuple(
        sorted(corporate_actions, key=lambda row: (row.effective_at_ms, row.event_id))
    )
    terminal = tuple(
        sorted(terminal_events, key=lambda row: (row.effective_at_ms, row.event_id))
    )
    cash = tuple(sorted(cash_returns, key=lambda row: row.effective_at_ms))
    event_inventory = tuple(
        [semantic_sha256(asdict(row)) for row in corporate]
        + [semantic_sha256(asdict(row)) for row in terminal]
        + [semantic_sha256(asdict(row)) for row in cash]
    )
    provisional = MassiveEconomicEventAuthorityV1(
        security_ids=securities,
        corporate_actions=corporate,
        terminal_events=terminal,
        cash_returns=cash,
        source_receipts=tuple(sorted(source_receipts)),
        event_inventory_sha256=semantic_sha256(event_inventory),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveEconomicReturnRowV1:
    source_session_index: int
    source_session_date: str
    security_id: str
    listed: bool
    economic_value: float
    economic_value_valid: bool
    terminal: bool
    position: EconomicPosition | None
    applied_cash_return_receipts: tuple[str, ...]
    session_panel_row_receipt_sha256: str
    economic_authority_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            isinstance(self.source_session_index, bool)
            or not isinstance(self.source_session_index, int)
            or self.source_session_index < 0
            or not self.source_session_date
            or not self.security_id
        ):
            raise MassiveEconomicReturnIndexV1Error("economic row identity is invalid")
        if (
            not isinstance(self.listed, bool)
            or not isinstance(self.economic_value_valid, bool)
            or not isinstance(self.terminal, bool)
        ):
            raise MassiveEconomicReturnIndexV1Error(
                "economic row state must be Boolean"
            )
        if not math.isfinite(self.economic_value) or self.economic_value < 0.0:
            raise MassiveEconomicReturnIndexV1Error("economic value is invalid")
        if not self.economic_value_valid and self.economic_value != 0.0:
            raise MassiveEconomicReturnIndexV1Error(
                "an invalid economic value must use its zero placeholder"
            )
        if self.position is None:
            if self.listed or self.economic_value_valid or self.terminal:
                raise MassiveEconomicReturnIndexV1Error(
                    "an absent position cannot carry an economic state"
                )
        else:
            self.position.validate()
            if self.terminal != (not self.position.holdings):
                raise MassiveEconomicReturnIndexV1Error(
                    "terminal state differs from risky holdings"
                )
        if self.applied_cash_return_receipts != tuple(
            sorted(set(self.applied_cash_return_receipts))
        ):
            raise MassiveEconomicReturnIndexV1Error(
                "applied cash-return inventory is not canonical"
            )
        for value in (
            *self.applied_cash_return_receipts,
            self.session_panel_row_receipt_sha256,
            self.economic_authority_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("economic row digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicReturnIndexV1Error("economic row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveEconomicReturnIndexArtifactV1:
    session_panel_receipt_sha256: str
    economic_authority_receipt_sha256: str
    rows: tuple[MassiveEconomicReturnRowV1, ...]
    row_count: int
    valid_row_count: int
    terminal_row_count: int
    row_inventory_sha256: str
    feature_spec_receipt_sha256: str
    feature_source_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA:
            raise MassiveEconomicReturnIndexV1Error("economic index schema drifted")
        for value in (
            self.session_panel_receipt_sha256,
            self.economic_authority_receipt_sha256,
            self.row_inventory_sha256,
            self.feature_spec_receipt_sha256,
            self.feature_source_sha256,
            self.receipt_sha256,
        ):
            _digest("economic artifact digest", value)
        if (
            self.feature_spec_receipt_sha256
            != MASSIVE_ECONOMIC_RETURN_INDEX_V1_SPEC_SHA256
            or self.feature_source_sha256
            != MASSIVE_ECONOMIC_RETURN_INDEX_V1_SOURCE_SHA256
        ):
            raise MassiveEconomicReturnIndexV1Error(
                "economic index implementation drifted"
            )
        keys = tuple((row.source_session_index, row.security_id) for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveEconomicReturnIndexV1Error(
                "economic index rows are not canonical"
            )
        for row in self.rows:
            row.validate()
            if (
                row.economic_authority_receipt_sha256
                != self.economic_authority_receipt_sha256
            ):
                raise MassiveEconomicReturnIndexV1Error(
                    "economic row authority differs"
                )
        if (
            self.row_count != len(self.rows)
            or self.valid_row_count
            != sum(row.economic_value_valid for row in self.rows)
            or self.terminal_row_count != sum(row.terminal for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
        ):
            raise MassiveEconomicReturnIndexV1Error("economic index counts differ")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ECONOMIC_RETURN_INDEX_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_RETURN_INDEX_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveEconomicReturnIndexV1Error(
                "economic index source contract differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicReturnIndexV1Error(
                "economic index artifact receipt differs"
            )


def build_massive_economic_return_rows_v1(
    *,
    panel_rows: Sequence[MassiveSessionPanelRowV1],
    economic_authority: MassiveEconomicEventAuthorityV1,
) -> tuple[MassiveEconomicReturnRowV1, ...]:
    """Build causal economic values on the exact panel session grid."""

    economic_authority.validate()
    rows = tuple(panel_rows)
    for row in rows:
        row.validate()
    keys = tuple((row.source_session_index, row.security_id) for row in rows)
    if not keys or keys != tuple(sorted(set(keys))):
        raise MassiveEconomicReturnIndexV1Error("panel rows are not canonical")
    security_ids = tuple(sorted({row.security_id for row in rows}))
    if security_ids != economic_authority.security_ids:
        raise MassiveEconomicReturnIndexV1Error(
            "economic and panel security inventories differ"
        )
    by_index: dict[int, dict[str, MassiveSessionPanelRowV1]] = {}
    for row in rows:
        by_index.setdefault(row.source_session_index, {})[row.security_id] = row
    if tuple(by_index) != tuple(range(len(by_index))) or any(
        set(group) != set(security_ids) for group in by_index.values()
    ):
        raise MassiveEconomicReturnIndexV1Error(
            "economic input is not a complete session rectangle"
        )
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    output: list[MassiveEconomicReturnRowV1] = []
    event_list: list[CorporateActionRecord | TerminalEventRecord] = [
        *economic_authority.corporate_actions,
        *economic_authority.terminal_events,
    ]
    events = tuple(
        sorted(
            event_list,
            key=lambda event_row: (
                event_row.effective_at_ms,
                event_row.available_at_ms,
                event_row.event_id,
            ),
        )
    )
    cash_rows = tuple(economic_authority.cash_returns)
    for origin_security_id in security_ids:
        position: EconomicPosition | None = None
        applied_cash: set[str] = set()
        for session_index, group in by_index.items():
            panel = group[origin_security_id]
            session_close_ms = _session_close_ms(group)
            if position is None and panel.listed:
                position = EconomicPosition.from_mapping({origin_security_id: 1.0})
            if position is not None:
                operations: list[
                    tuple[
                        int,
                        int,
                        str,
                        CorporateActionRecord | TerminalEventRecord | CashReturnRecord,
                    ]
                ] = []
                for event in events:
                    if (
                        event.event_id not in position.applied_event_ids
                        and event.effective_at_ms <= session_close_ms
                        and event.available_at_ms <= session_close_ms
                    ):
                        operations.append(
                            (event.effective_at_ms, 0, event.event_id, event)
                        )
                for cash in cash_rows:
                    cash_receipt = semantic_sha256(asdict(cash))
                    if (
                        cash_receipt not in applied_cash
                        and cash.effective_at_ms <= session_close_ms
                        and cash.available_at_ms <= session_close_ms
                    ):
                        operations.append((cash.effective_at_ms, 1, cash_receipt, cash))
                for _, _, operation_id, operation in sorted(operations):
                    if isinstance(operation, CashReturnRecord):
                        position = apply_cash_return(
                            position, operation.one_step_return
                        )
                        applied_cash.add(operation_id)
                    elif operation.security_id in position.as_mapping():
                        position = apply_corporate_action(position, operation)
            valid = False
            value = 0.0
            if position is not None:
                marks = {
                    security_id: candidate.bars_values[close_index]
                    for security_id, candidate in group.items()
                    if candidate.bars_valid[close_index]
                }
                try:
                    value = mark_position(position, marks)
                    valid = True
                except ValueError:
                    value = 0.0
            provisional = MassiveEconomicReturnRowV1(
                source_session_index=session_index,
                source_session_date=panel.source_session_date,
                security_id=origin_security_id,
                listed=panel.listed,
                economic_value=value,
                economic_value_valid=valid,
                terminal=position is not None and not position.holdings,
                position=position,
                applied_cash_return_receipts=tuple(sorted(applied_cash)),
                session_panel_row_receipt_sha256=panel.receipt_sha256,
                economic_authority_receipt_sha256=(economic_authority.receipt_sha256),
                receipt_sha256="0" * 64,
            )
            result = replace(
                provisional,
                receipt_sha256=semantic_sha256(provisional.unsigned()),
            )
            result.validate()
            output.append(result)
    return tuple(
        sorted(output, key=lambda row: (row.source_session_index, row.security_id))
    )


def _session_close_ms(
    group: Mapping[str, MassiveSessionPanelRowV1],
) -> int:
    """Return the exact exchange-session close bound carried by every grid row."""

    coordinates = {
        (row.source_session_date, row.regular_open_ns, row.regular_close_ns)
        for row in group.values()
    }
    if len(coordinates) != 1:
        raise MassiveEconomicReturnIndexV1Error("panel session coordinates differ")
    return next(iter(coordinates))[2] // 1_000_000


def _payload(artifact: MassiveEconomicReturnIndexArtifactV1) -> dict[str, object]:
    return {
        "schema": artifact.schema,
        "session_panel_receipt_sha256": artifact.session_panel_receipt_sha256,
        "economic_authority_receipt_sha256": artifact.economic_authority_receipt_sha256,
        "rows": tuple(asdict(row) for row in artifact.rows),
        "row_count": artifact.row_count,
        "valid_row_count": artifact.valid_row_count,
        "terminal_row_count": artifact.terminal_row_count,
        "row_inventory_sha256": artifact.row_inventory_sha256,
        "feature_spec_receipt_sha256": artifact.feature_spec_receipt_sha256,
        "feature_source_sha256": artifact.feature_source_sha256,
    }


def materialize_massive_economic_return_index_v1(
    *,
    session_panel_root: str | Path,
    output_root: str | Path,
    session_panel: MassiveSessionPanelArtifactV1,
    economic_authority: MassiveEconomicEventAuthorityV1,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveEconomicReturnIndexArtifactV1:
    validate_massive_session_panel_v1(root=session_panel_root, artifact=session_panel)
    economic_authority.validate()
    rows = build_massive_economic_return_rows_v1(
        panel_rows=session_panel.rows,
        economic_authority=economic_authority,
    )
    relative = (
        "massive-profitability-p0/economic-return-index-v1/"
        f"{session_panel.start_session_date}-{session_panel.end_session_date}.json"
    )
    placeholder = MassiveEconomicReturnIndexArtifactV1(
        session_panel_receipt_sha256=session_panel.receipt_sha256,
        economic_authority_receipt_sha256=economic_authority.receipt_sha256,
        rows=rows,
        row_count=len(rows),
        valid_row_count=sum(row.economic_value_valid for row in rows),
        terminal_row_count=sum(row.terminal for row in rows),
        row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        feature_spec_receipt_sha256=MASSIVE_ECONOMIC_RETURN_INDEX_V1_SPEC_SHA256,
        feature_source_sha256=MASSIVE_ECONOMIC_RETURN_INDEX_V1_SOURCE_SHA256,
        loaded_source=session_panel.loaded_source,
        receipt_sha256="0" * 64,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(placeholder))),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ECONOMIC_RETURN_INDEX_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_ECONOMIC_RETURN_INDEX_V1_SOURCE_SCHEMA_SHA256,
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
    validate_massive_economic_return_index_v1(root=output_root, artifact=result)
    return result


def validate_massive_economic_return_index_v1(
    *,
    root: str | Path,
    artifact: MassiveEconomicReturnIndexArtifactV1,
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicReturnIndexV1Error(
            "economic return index source is not JSON"
        ) from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(artifact)
    ):
        raise MassiveEconomicReturnIndexV1Error("economic return index bytes differ")


__all__ = [
    "MASSIVE_ECONOMIC_RETURN_INDEX_V1_DATASET",
    "MASSIVE_ECONOMIC_RETURN_INDEX_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_RETURN_INDEX_V1_SPEC_SHA256",
    "MassiveEconomicEventAuthorityV1",
    "MassiveEconomicReturnIndexArtifactV1",
    "MassiveEconomicReturnIndexV1Error",
    "MassiveEconomicReturnRowV1",
    "build_massive_economic_event_authority_v1",
    "build_massive_economic_return_rows_v1",
    "materialize_massive_economic_return_index_v1",
    "validate_massive_economic_return_index_v1",
]
