"""Bounded, origin-semantic economic histories for P0 feature construction.

This generation deliberately contains only the 64 exchange sessions ending at
the source cutoff.  Capture and target-period provenance is audit-only; a
future economic event cannot enter the feature semantic receipt.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_economic_coverage_v8 import (
    MassiveNativeEconomicObservationV8,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA = (
    "rl-quant.massive-profitability-feature-accounting-v1"
)
MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "history": "exact-source-minus-63-through-source-XNYS-sessions",
        "events": "economic-effective-at-or-before-source-close-only",
        "future_capture": "audit-only",
        "missing": "zero-value-plus-independent-false-mask",
        "target_period_events": "prohibited",
        "production_equivalence": False,
    }
)


class MassiveProfitabilityFeatureAccountingV1Error(ValueError):
    """A bounded feature-accounting history differs from the P0 contract."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityFeatureAccountingV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _date(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise MassiveProfitabilityFeatureAccountingV1Error(f"{name} must be a date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise MassiveProfitabilityFeatureAccountingV1Error(
            f"{name} must be a date"
        ) from exc
    if parsed.isoformat() != value:
        raise MassiveProfitabilityFeatureAccountingV1Error(f"{name} must be canonical")
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFeatureEconomicValueRowV1:
    source_session_offset: int
    source_session_date: str
    security_id: str
    economic_value: float
    valid: bool
    terminal: bool
    mark_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            isinstance(self.source_session_offset, bool)
            or not isinstance(self.source_session_offset, int)
            or not -63 <= self.source_session_offset <= 0
            or not self.security_id
            or self.security_id != self.security_id.strip()
        ):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature economic row identity differs"
            )
        _date("feature economic session", self.source_session_date)
        if (
            not isinstance(self.valid, bool)
            or not isinstance(self.terminal, bool)
            or not isinstance(self.economic_value, (int, float))
            or isinstance(self.economic_value, bool)
            or not math.isfinite(float(self.economic_value))
            or float(self.economic_value) < 0.0
            or (not self.valid and float(self.economic_value) != 0.0)
            or (self.terminal and not self.valid)
        ):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature economic value or mask differs"
            )
        _digest("feature mark inventory", self.mark_inventory_sha256)
        _digest("feature economic row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature economic row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFeatureAccountingV1:
    origin_receipt_sha256: str
    source_session_date: str
    feature_cutoff_at_ms: int
    session_dates: tuple[str, ...]
    decision_member_security_ids: tuple[str, ...]
    rows: tuple[MassiveProfitabilityFeatureEconomicValueRowV1, ...]
    selected_event_receipts: tuple[str, ...]
    maximum_selected_event_effective_at_ms: int | None
    economic_coverage_semantic_receipt_sha256: str
    row_inventory_sha256: str
    event_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    economic_coverage_audit_receipt_sha256: str
    audit_receipt_sha256: str
    economic_values_data_qualified: bool
    schema: str = MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        excluded = {
            "semantic_receipt_sha256",
            "economic_coverage_audit_receipt_sha256",
            "audit_receipt_sha256",
        }
        return {
            key: value for key, value in asdict(self).items() if key not in excluded
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SOURCE_SHA256
            or not isinstance(self.economic_values_data_qualified, bool)
        ):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature accounting identity differs"
            )
        _date("feature accounting source", self.source_session_date)
        if (
            isinstance(self.feature_cutoff_at_ms, bool)
            or not isinstance(self.feature_cutoff_at_ms, int)
            or self.feature_cutoff_at_ms < 0
            or len(self.session_dates) != 64
            or self.session_dates != tuple(sorted(set(self.session_dates)))
            or self.session_dates[-1] != self.source_session_date
            or not self.decision_member_security_ids
            or self.decision_member_security_ids
            != tuple(sorted(set(self.decision_member_security_ids)))
        ):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature accounting interval or support differs"
            )
        for value in self.session_dates:
            _date("feature accounting session", value)
        keys = tuple((row.source_session_offset, row.security_id) for row in self.rows)
        expected_keys = tuple(
            (offset, security_id)
            for offset in range(-63, 1)
            for security_id in self.decision_member_security_ids
        )
        if keys != expected_keys:
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature accounting does not contain one exact 64-session rectangle"
            )
        for row in self.rows:
            row.validate()
            if (
                self.session_dates[row.source_session_offset + 63]
                != row.source_session_date
            ):
                raise MassiveProfitabilityFeatureAccountingV1Error(
                    "feature accounting offset and date differ"
                )
        if self.selected_event_receipts != tuple(
            sorted(set(self.selected_event_receipts))
        ):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature accounting events are not canonical"
            )
        for value in (
            self.origin_receipt_sha256,
            self.economic_coverage_semantic_receipt_sha256,
            self.row_inventory_sha256,
            self.event_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.economic_coverage_audit_receipt_sha256,
            self.audit_receipt_sha256,
            *self.selected_event_receipts,
        ):
            _digest("feature accounting digest", value)
        if self.maximum_selected_event_effective_at_ms is not None and (
            isinstance(self.maximum_selected_event_effective_at_ms, bool)
            or not isinstance(self.maximum_selected_event_effective_at_ms, int)
            or self.maximum_selected_event_effective_at_ms > self.feature_cutoff_at_ms
        ):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "post-cutoff event entered feature accounting"
            )
        if (
            self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.event_inventory_sha256
            != semantic_sha256(self.selected_event_receipts)
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature accounting semantic inventory differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "economic_coverage_audit_receipt_sha256": (
                    self.economic_coverage_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "feature accounting audit receipt differs"
            )


def build_massive_profitability_feature_accounting_v1(
    *,
    origin: MassiveProfitabilityDecisionOriginV1,
    session_authority: MassiveSessionAuthority,
    economic_values: Mapping[tuple[str, str], float | None],
    mark_receipts: Mapping[tuple[str, str], str],
    terminal_keys: Sequence[tuple[str, str]] = (),
    selected_events: Sequence[MassiveNativeEconomicObservationV8] = (),
    economic_coverage_semantic_receipt_sha256: str,
    economic_coverage_audit_receipt_sha256: str,
) -> MassiveProfitabilityFeatureAccountingV1:
    """Build an exact 64-session feature history at one decision origin."""

    origin.validate()
    session_authority.validate()
    sessions = tuple(session_authority.sessions)
    by_date = {row.session_date: index for index, row in enumerate(sessions)}
    source_index = by_date.get(origin.source_session_date)
    if source_index is None or source_index < 63:
        raise MassiveProfitabilityFeatureAccountingV1Error(
            "source does not have 63 authority sessions of prehistory"
        )
    selected_sessions = sessions[source_index - 63 : source_index + 1]
    session_dates = tuple(row.session_date for row in selected_sessions)
    if (
        selected_sessions[-1].regular_close_ns // 1_000_000
        != origin.feature_cutoff_at_ms
    ):
        raise MassiveProfitabilityFeatureAccountingV1Error(
            "feature cutoff is not the source-session regular close"
        )
    members = tuple(sorted(origin.decision_member_security_ids))
    allowed_keys = {
        (session_date, security_id)
        for session_date in session_dates
        for security_id in members
    }
    if set(economic_values) != allowed_keys or set(mark_receipts) != allowed_keys:
        raise MassiveProfitabilityFeatureAccountingV1Error(
            "feature economic values do not exactly match bounded support"
        )
    terminal = set(terminal_keys)
    if not terminal <= allowed_keys:
        raise MassiveProfitabilityFeatureAccountingV1Error(
            "feature terminal inventory exceeds bounded support"
        )
    rows: list[MassiveProfitabilityFeatureEconomicValueRowV1] = []
    for offset, session_date in zip(range(-63, 1), session_dates, strict=True):
        for security_id in members:
            key = (session_date, security_id)
            value = economic_values[key]
            mark = _digest("feature mark", mark_receipts[key])
            body = {
                "source_session_offset": offset,
                "source_session_date": session_date,
                "security_id": security_id,
                "economic_value": 0.0 if value is None else float(value),
                "valid": value is not None,
                "terminal": key in terminal,
                "mark_inventory_sha256": mark,
            }
            row = MassiveProfitabilityFeatureEconomicValueRowV1(
                **body,
                receipt_sha256=semantic_sha256(body),
            )
            row.validate()
            rows.append(row)
    event_rows = tuple(selected_events)
    for event in event_rows:
        event.validate()
        if event.effective_at_ms > origin.feature_cutoff_at_ms:
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "target-period event cannot enter feature accounting"
            )
        if event.predictive_feature_eligible:
            raise MassiveProfitabilityFeatureAccountingV1Error(
                "corporate-action fields cannot be predictive inputs"
            )
    event_receipts = tuple(sorted(event.receipt_sha256 for event in event_rows))
    maximum_effective = max(
        (event.effective_at_ms for event in event_rows), default=None
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA,
        "origin_receipt_sha256": origin.receipt_sha256,
        "source_session_date": origin.source_session_date,
        "feature_cutoff_at_ms": origin.feature_cutoff_at_ms,
        "session_dates": session_dates,
        "decision_member_security_ids": members,
        "rows": tuple(asdict(row) for row in rows),
        "selected_event_receipts": event_receipts,
        "maximum_selected_event_effective_at_ms": maximum_effective,
        "economic_coverage_semantic_receipt_sha256": _digest(
            "economic coverage semantics", economic_coverage_semantic_receipt_sha256
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "event_inventory_sha256": semantic_sha256(event_receipts),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SOURCE_SHA256,
        "economic_values_data_qualified": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    audit_receipt = _digest(
        "economic coverage audit", economic_coverage_audit_receipt_sha256
    )
    result = MassiveProfitabilityFeatureAccountingV1(
        origin_receipt_sha256=origin.receipt_sha256,
        source_session_date=origin.source_session_date,
        feature_cutoff_at_ms=origin.feature_cutoff_at_ms,
        session_dates=session_dates,
        decision_member_security_ids=members,
        rows=tuple(rows),
        selected_event_receipts=event_receipts,
        maximum_selected_event_effective_at_ms=maximum_effective,
        economic_coverage_semantic_receipt_sha256=economic_coverage_semantic_receipt_sha256,
        row_inventory_sha256=semantic["row_inventory_sha256"],  # type: ignore[arg-type]
        event_inventory_sha256=semantic["event_inventory_sha256"],  # type: ignore[arg-type]
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        economic_coverage_audit_receipt_sha256=audit_receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "economic_coverage_audit_receipt_sha256": audit_receipt,
            }
        ),
        economic_values_data_qualified=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SPEC_SHA256",
    "MassiveProfitabilityFeatureAccountingV1",
    "MassiveProfitabilityFeatureAccountingV1Error",
    "MassiveProfitabilityFeatureEconomicValueRowV1",
    "build_massive_profitability_feature_accounting_v1",
]
