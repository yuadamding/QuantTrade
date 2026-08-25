"""Cash-vintage and raw-provider economic authority for profitability P0 V7.

V7 is deliberately not a history materializer.  It converts exact raw Massive
REST response bytes into origin-visible dividend and split observations, keeps
future malformed rows in a separate audit inventory, resolves tied security
events from one complete provider snapshot, and models cash return as an
accrual interval over cash lots held before the interval began.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.alpha.contracts import CorporateActionKind
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.economic_provider_capture_v7 import (
    MASSIVE_ECONOMIC_RAW_CAPTURE_V7_DATASET,
    MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SPEC_SHA256,
    MassiveEconomicRawRestCaptureV7,
    parse_massive_economic_raw_rest_capture_v7,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_ECONOMIC_AUTHORITY_V7_SCHEMA = (
    "rl-quant.massive-resolved-economic-authority-at-origin-v7"
)
MASSIVE_ECONOMIC_AUTHORITY_V7_DATASET = (
    "massive-resolved-economic-authority-at-origin-v7"
)
MASSIVE_ECONOMIC_AUTHORITY_V7_OBJECT_PREFIX = (
    "massive-profitability-p0/resolved-economic-authority-v7/"
)
MASSIVE_ECONOMIC_AUTHORITY_V7_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_AUTHORITY_V7_SCHEMA,
        "semantic_receipt": "origin-visible-events-cash-and-order",
        "audit_receipt": "semantic-receipt-plus-complete-raw-capture-inventory",
        "publication": "immutable-create-only-source-transaction",
    }
)
MASSIVE_CASH_ACCRUAL_CONVENTION_V7 = (
    "cash-held-strictly-before-period-start-earns-period-return;"
    "intraperiod-cash-starts-next-period"
)
MASSIVE_CASH_ACCRUAL_CONVENTION_V7_RECEIPT_SHA256 = semantic_sha256(
    MASSIVE_CASH_ACCRUAL_CONVENTION_V7
)
MASSIVE_SINGLE_SECURITY_EVENT_ORDER_RULE_V7_RECEIPT_SHA256 = semantic_sha256(
    {
        "rule": "one-selected-event-in-origin-interaction-group",
        "local_sequence": 0,
        "snapshot": "prohibited-as-unnecessary",
    }
)
MASSIVE_NATIVE_AVAILABILITY_RULE_V7_RECEIPT_SHA256 = semantic_sha256(
    {
        "preferred": "raw-provider-update-timestamp",
        "current_massive-dividends-splits": "conservative-response-completion",
        "historical_backdating": "prohibited",
    }
)
MASSIVE_ECONOMIC_AUTHORITY_V7_HISTORICAL_PANEL_AUTHORIZED = False
MASSIVE_ECONOMIC_AUTHORITY_V7_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_ECONOMIC_AUTHORITY_V7_PROFITABILITY_REPORTING_AUTHORIZED = False

_NEW_YORK = ZoneInfo("America/New_York")
_MASSIVE_DATE_FIELDS = {
    "massive-dividends": "ex_dividend_date",
    "massive-splits": "execution_date",
}


class MassiveEconomicAuthorityV7Error(ValueError):
    """Raw provider, cash-vintage, order, or origin evidence differs."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEconomicAuthorityV7Error(f"{name} must be canonical text")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicAuthorityV7Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicAuthorityV7Error(f"{name} must be nonnegative")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveEconomicAuthorityV7Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MassiveEconomicAuthorityV7Error(f"{name} must be finite")
    return result


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveEconomicAuthorityV7Error(f"{name} fields differ")


def _date_at_market_open_ms(name: str, value: object) -> int:
    raw = _text(name, value)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise MassiveEconomicAuthorityV7Error(
            f"{name} must be an ISO calendar date"
        ) from exc
    if parsed.isoformat() != raw:
        raise MassiveEconomicAuthorityV7Error(f"{name} is not canonical")
    return int(
        datetime.combine(parsed, time(hour=9, minute=30), tzinfo=_NEW_YORK).timestamp()
        * 1000
    )


def _security_for_ticker(
    *,
    identity_authority: PITSecurityUniverseAuthority,
    ticker: str,
    effective_at_ms: int,
) -> tuple[str, str]:
    candidates = tuple(
        row
        for row in identity_authority.ticker_history
        if row.ticker == ticker
        and row.valid_from_ms <= effective_at_ms
        and (row.valid_to_ms is None or effective_at_ms < row.valid_to_ms)
        and row.available_at_ms <= effective_at_ms
    )
    if len(candidates) != 1:
        raise MassiveEconomicAuthorityV7Error(
            "native provider ticker does not resolve to one causal security"
        )
    return candidates[0].security_id, candidates[0].source_receipt_sha256


@dataclass(frozen=True, slots=True)
class MassiveCashLotV7:
    lot_id: str
    amount: float
    acquired_at_ms: int
    acquisition_event_key: str
    applied_accrual_period_receipts: tuple[str, ...]
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _text("cash lot ID", self.lot_id)
        _finite("cash lot amount", self.amount)
        _nonnegative_int("cash lot acquisition time", self.acquired_at_ms)
        _text("cash lot acquisition event", self.acquisition_event_key)
        if self.applied_accrual_period_receipts != tuple(
            sorted(set(self.applied_accrual_period_receipts))
        ):
            raise MassiveEconomicAuthorityV7Error(
                "cash lot accrual inventory is not canonical"
            )
        for value in self.applied_accrual_period_receipts:
            _digest("cash lot accrual receipt", value)
        _digest("cash lot receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV7Error("cash lot receipt differs")

    @classmethod
    def build(
        cls,
        *,
        lot_id: str,
        amount: float,
        acquired_at_ms: int,
        acquisition_event_key: str,
        applied_accrual_period_receipts: Sequence[str] = (),
    ) -> MassiveCashLotV7:
        provisional = cls(
            lot_id=lot_id,
            amount=float(amount),
            acquired_at_ms=acquired_at_ms,
            acquisition_event_key=acquisition_event_key,
            applied_accrual_period_receipts=tuple(
                sorted(set(applied_accrual_period_receipts))
            ),
            receipt_sha256="0" * 64,
        )
        result = replace(
            provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class MassiveCashAccrualPeriodV7:
    period_id: str
    period_start_at_ms: int
    period_end_at_ms: int
    one_period_return: float
    available_at_ms: int
    provider_id: str
    provider_dataset: str
    raw_provider_source_receipt_sha256: str
    raw_provider_row_receipt_sha256: str
    availability_rule_receipt_sha256: str
    convention_receipt_sha256: str
    provider_runtime_qualified: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _text("cash accrual period ID", self.period_id)
        start = _nonnegative_int("cash accrual start", self.period_start_at_ms)
        end = _nonnegative_int("cash accrual end", self.period_end_at_ms)
        if end <= start:
            raise MassiveEconomicAuthorityV7Error("cash accrual interval is empty")
        if _finite("cash one-period return", self.one_period_return) <= -1.0:
            raise MassiveEconomicAuthorityV7Error(
                "cash accrual return loses more than the cash balance"
            )
        _nonnegative_int("cash accrual availability", self.available_at_ms)
        _text("cash provider", self.provider_id)
        _text("cash provider dataset", self.provider_dataset)
        for value in (
            self.raw_provider_source_receipt_sha256,
            self.raw_provider_row_receipt_sha256,
            self.availability_rule_receipt_sha256,
            self.convention_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("cash accrual digest", value)
        if (
            self.convention_receipt_sha256
            != MASSIVE_CASH_ACCRUAL_CONVENTION_V7_RECEIPT_SHA256
        ):
            raise MassiveEconomicAuthorityV7Error("cash accrual convention differs")
        if not isinstance(self.provider_runtime_qualified, bool):
            raise MassiveEconomicAuthorityV7Error(
                "cash provider qualification is not boolean"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV7Error("cash accrual period receipt differs")

    @classmethod
    def build(
        cls,
        *,
        period_id: str,
        period_start_at_ms: int,
        period_end_at_ms: int,
        one_period_return: float,
        available_at_ms: int,
        provider_id: str,
        provider_dataset: str,
        raw_provider_source_receipt_sha256: str,
        raw_provider_row_receipt_sha256: str,
        availability_rule_receipt_sha256: str,
    ) -> MassiveCashAccrualPeriodV7:
        provisional = cls(
            period_id=period_id,
            period_start_at_ms=period_start_at_ms,
            period_end_at_ms=period_end_at_ms,
            one_period_return=float(one_period_return),
            available_at_ms=available_at_ms,
            provider_id=provider_id,
            provider_dataset=provider_dataset,
            raw_provider_source_receipt_sha256=raw_provider_source_receipt_sha256,
            raw_provider_row_receipt_sha256=raw_provider_row_receipt_sha256,
            availability_rule_receipt_sha256=availability_rule_receipt_sha256,
            convention_receipt_sha256=(
                MASSIVE_CASH_ACCRUAL_CONVENTION_V7_RECEIPT_SHA256
            ),
            provider_runtime_qualified=False,
            receipt_sha256="0" * 64,
        )
        result = replace(
            provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
        )
        result.validate()
        return result


def accrue_massive_cash_lots_v7(
    *,
    cash_lots: Sequence[MassiveCashLotV7],
    period: MassiveCashAccrualPeriodV7,
) -> tuple[MassiveCashLotV7, ...]:
    """Accrue only cash that existed strictly before the interval began."""

    period.validate()
    if len({row.lot_id for row in cash_lots}) != len(cash_lots):
        raise MassiveEconomicAuthorityV7Error("cash lots are duplicated")
    output: list[MassiveCashLotV7] = []
    for row in cash_lots:
        row.validate()
        amount = row.amount
        applied = row.applied_accrual_period_receipts
        if period.receipt_sha256 in applied:
            raise MassiveEconomicAuthorityV7Error(
                "cash accrual period was already applied to this lot"
            )
        if row.acquired_at_ms < period.period_start_at_ms:
            amount *= 1.0 + period.one_period_return
            applied = tuple(sorted((*applied, period.receipt_sha256)))
        output.append(
            MassiveCashLotV7.build(
                lot_id=row.lot_id,
                amount=amount,
                acquired_at_ms=row.acquired_at_ms,
                acquisition_event_key=row.acquisition_event_key,
                applied_accrual_period_receipts=applied,
            )
        )
    return tuple(sorted(output, key=lambda row: row.lot_id))


@dataclass(frozen=True, slots=True)
class MassiveNativeEconomicObservationV7:
    surface_id: str
    provider_event_key: str
    provider_revision_id: str
    logical_event_key: str
    security_id: str
    ticker: str
    kind: str
    effective_at_ms: int
    available_at_ms: int
    availability_derivation_kind: str
    availability_field_name: str
    availability_raw_value: str
    availability_rule_receipt_sha256: str
    raw_page_index: int
    raw_provider_request_id: str
    raw_provider_row_locator: str
    raw_provider_row_sha256: str
    identity_mapping_receipt_sha256: str
    cash_per_share: float
    share_ratio: float
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.surface_id not in {"massive-dividends", "massive-splits"}:
            raise MassiveEconomicAuthorityV7Error(
                "native economic surface is unsupported"
            )
        for name in (
            "provider_event_key",
            "provider_revision_id",
            "security_id",
            "ticker",
            "kind",
            "availability_derivation_kind",
            "availability_field_name",
            "availability_raw_value",
            "raw_provider_request_id",
            "raw_provider_row_locator",
        ):
            _text(name, getattr(self, name))
        for value in (
            self.logical_event_key,
            self.availability_rule_receipt_sha256,
            self.raw_provider_row_sha256,
            self.identity_mapping_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("native economic observation digest", value)
        _nonnegative_int("native event effective time", self.effective_at_ms)
        _nonnegative_int("native event availability", self.available_at_ms)
        _nonnegative_int("native event page index", self.raw_page_index)
        _finite("native cash per share", self.cash_per_share)
        share_ratio = _finite("native share ratio", self.share_ratio)
        if self.surface_id == "massive-dividends":
            if (
                self.kind
                not in {
                    CorporateActionKind.CASH_DIVIDEND.value,
                    CorporateActionKind.SPECIAL_DIVIDEND.value,
                }
                or self.cash_per_share < 0.0
                or share_ratio != 1.0
            ):
                raise MassiveEconomicAuthorityV7Error("native dividend terms differ")
        elif (
            self.kind
            not in {
                CorporateActionKind.SPLIT.value,
                CorporateActionKind.REVERSE_SPLIT.value,
            }
            or self.cash_per_share != 0.0
            or share_ratio <= 0.0
        ):
            raise MassiveEconomicAuthorityV7Error("native split terms differ")
        if (
            self.availability_derivation_kind != "conservative-response-completion"
            or self.availability_field_name != "response.completed_at_ms"
            or self.availability_raw_value != str(self.available_at_ms)
            or self.availability_rule_receipt_sha256
            != MASSIVE_NATIVE_AVAILABILITY_RULE_V7_RECEIPT_SHA256
        ):
            raise MassiveEconomicAuthorityV7Error(
                "native event availability derivation differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV7Error(
                "native economic observation receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveOriginProviderAuditV7:
    raw_capture_receipts: tuple[str, ...]
    raw_page_receipts: tuple[str, ...]
    future_page_receipts: tuple[str, ...]
    supplemental_source_receipts: tuple[str, ...]
    issue_inventory: tuple[tuple[str, str], ...]
    raw_source_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        for inventory in (
            self.raw_capture_receipts,
            self.raw_page_receipts,
            self.future_page_receipts,
            self.supplemental_source_receipts,
        ):
            if inventory != tuple(sorted(set(inventory))):
                raise MassiveEconomicAuthorityV7Error(
                    "provider audit inventory is not canonical"
                )
            for value in inventory:
                _digest("provider audit inventory", value)
        if self.issue_inventory != tuple(sorted(set(self.issue_inventory))):
            raise MassiveEconomicAuthorityV7Error(
                "provider audit issues are not canonical"
            )
        for locator, issue in self.issue_inventory:
            _text("provider audit issue locator", locator)
            _digest("provider audit issue", issue)
        _digest("raw source inventory", self.raw_source_inventory_sha256)
        _digest("provider audit receipt", self.receipt_sha256)
        if self.raw_source_inventory_sha256 != semantic_sha256(
            (self.raw_capture_receipts, self.supplemental_source_receipts)
        ):
            raise MassiveEconomicAuthorityV7Error("raw source audit inventory differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV7Error("provider audit receipt differs")


def _native_observation(
    *,
    capture: MassiveEconomicRawRestCaptureV7,
    page_index: int,
    row_index: int,
    row: Mapping[str, object],
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveNativeEconomicObservationV7:
    page = capture.pages[page_index]
    surface = capture.surface_id
    event_key = _text("native provider event ID", row.get("id"))
    ticker = _text("native provider ticker", row.get("ticker"))
    effective_field = _MASSIVE_DATE_FIELDS[surface]
    effective_at_ms = _date_at_market_open_ms(
        f"native {effective_field}", row.get(effective_field)
    )
    security_id, identity_receipt = _security_for_ticker(
        identity_authority=identity_authority,
        ticker=ticker,
        effective_at_ms=effective_at_ms,
    )
    raw_row_sha = semantic_sha256(dict(row))
    provider_revision = semantic_sha256(
        (capture.provider_id, surface, event_key, raw_row_sha)
    )
    logical = semantic_sha256((capture.provider_id, surface, event_key))
    if surface == "massive-dividends":
        cash = _finite("native cash amount", row.get("cash_amount"))
        dividend_type = row.get("dividend_type")
        kind = (
            CorporateActionKind.SPECIAL_DIVIDEND.value
            if dividend_type in {"SC", "special", "special-cash"}
            else CorporateActionKind.CASH_DIVIDEND.value
        )
        ratio = 1.0
    else:
        split_from = _finite("native split from", row.get("split_from"))
        split_to = _finite("native split to", row.get("split_to"))
        if split_from <= 0.0 or split_to <= 0.0:
            raise MassiveEconomicAuthorityV7Error("native split ratio is nonpositive")
        ratio = split_to / split_from
        cash = 0.0
        kind = (
            CorporateActionKind.SPLIT.value
            if ratio >= 1.0
            else CorporateActionKind.REVERSE_SPLIT.value
        )
    provisional = MassiveNativeEconomicObservationV7(
        surface_id=surface,
        provider_event_key=event_key,
        provider_revision_id=provider_revision,
        logical_event_key=logical,
        security_id=security_id,
        ticker=ticker,
        kind=kind,
        effective_at_ms=effective_at_ms,
        available_at_ms=page.completed_at_ms,
        availability_derivation_kind="conservative-response-completion",
        availability_field_name="response.completed_at_ms",
        availability_raw_value=str(page.completed_at_ms),
        availability_rule_receipt_sha256=(
            MASSIVE_NATIVE_AVAILABILITY_RULE_V7_RECEIPT_SHA256
        ),
        raw_page_index=page_index,
        raw_provider_request_id=page.provider_request_id,
        raw_provider_row_locator=f"page={page_index}/results={row_index}",
        raw_provider_row_sha256=raw_row_sha,
        identity_mapping_receipt_sha256=identity_receipt,
        cash_per_share=cash,
        share_ratio=ratio,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def adapt_massive_raw_economic_captures_at_origin_v7(
    *,
    captures: Sequence[MassiveEconomicRawRestCaptureV7],
    identity_authority: PITSecurityUniverseAuthority,
    decision_at_ms: int,
) -> tuple[
    tuple[MassiveNativeEconomicObservationV7, ...],
    MassiveOriginProviderAuditV7,
]:
    """Parse origin-visible rows strictly; audit future malformed rows separately."""

    identity_authority.validate()
    decision = _nonnegative_int("native economic decision time", decision_at_ms)
    if len({capture.surface_id for capture in captures}) != len(captures):
        raise MassiveEconomicAuthorityV7Error(
            "raw economic capture surfaces are duplicated"
        )
    selected: list[MassiveNativeEconomicObservationV7] = []
    issues: list[tuple[str, str]] = []
    page_receipts: list[str] = []
    future_pages: list[str] = []
    for capture in sorted(captures, key=lambda row: row.surface_id):
        capture.validate()
        if capture.loaded_source.receipt.dataset_id != (
            MASSIVE_ECONOMIC_RAW_CAPTURE_V7_DATASET
        ):
            raise MassiveEconomicAuthorityV7Error(
                "raw economic capture dataset differs"
            )
        if capture.surface_id not in _MASSIVE_DATE_FIELDS:
            continue
        for page in capture.pages:
            page_receipts.append(page.receipt_sha256)
            raw_rows = page.parsed_body()["results"]
            if not isinstance(raw_rows, list):
                raise MassiveEconomicAuthorityV7Error(
                    "provider result inventory is not a list"
                )
            rows: list[object] = raw_rows
            if page.completed_at_ms > decision:
                future_pages.append(page.receipt_sha256)
                for index, row in enumerate(rows):
                    try:
                        if not isinstance(row, dict):
                            raise MassiveEconomicAuthorityV7Error(
                                "future provider result is not an object"
                            )
                        _native_observation(
                            capture=capture,
                            page_index=page.page_index,
                            row_index=index,
                            row=row,
                            identity_authority=identity_authority,
                        )
                    except (KeyError, MassiveEconomicAuthorityV7Error) as exc:
                        locator = (
                            f"{capture.surface_id}/page={page.page_index}/"
                            f"results={index}"
                        )
                        issues.append(
                            (
                                locator,
                                semantic_sha256(
                                    {
                                        "error_type": type(exc).__name__,
                                        "message": str(exc),
                                        "row": row,
                                    }
                                ),
                            )
                        )
                continue
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise MassiveEconomicAuthorityV7Error(
                        "origin-visible provider result is not an object"
                    )
                selected.append(
                    _native_observation(
                        capture=capture,
                        page_index=page.page_index,
                        row_index=index,
                        row=row,
                        identity_authority=identity_authority,
                    )
                )
    by_logical: dict[str, MassiveNativeEconomicObservationV7] = {}
    for row in selected:
        previous = by_logical.get(row.logical_event_key)
        if previous is not None and previous.receipt_sha256 != row.receipt_sha256:
            raise MassiveEconomicAuthorityV7Error(
                "native provider event has unresolved multiple revisions"
            )
        by_logical[row.logical_event_key] = row
    observations = tuple(
        sorted(
            by_logical.values(),
            key=lambda row: (
                row.effective_at_ms,
                row.security_id,
                row.logical_event_key,
            ),
        )
    )
    capture_receipts = tuple(sorted(capture.receipt_sha256 for capture in captures))
    audit_provisional = MassiveOriginProviderAuditV7(
        raw_capture_receipts=capture_receipts,
        raw_page_receipts=tuple(sorted(set(page_receipts))),
        future_page_receipts=tuple(sorted(set(future_pages))),
        supplemental_source_receipts=(),
        issue_inventory=tuple(sorted(set(issues))),
        raw_source_inventory_sha256=semantic_sha256((capture_receipts, ())),
        receipt_sha256="0" * 64,
    )
    audit = replace(
        audit_provisional,
        receipt_sha256=semantic_sha256(audit_provisional.unsigned()),
    )
    audit.validate()
    return observations, audit


@dataclass(frozen=True, slots=True)
class MassiveProviderEconomicOrderSnapshotV7:
    order_group_id: str
    order_snapshot_id: str
    interaction_domain_sha256: str
    effective_at_ms: int
    snapshot_available_at_ms: int
    complete_logical_event_keys: tuple[str, ...]
    ordered_logical_event_keys: tuple[str, ...]
    raw_provider_source_receipt_sha256: str
    raw_provider_snapshot_receipt_sha256: str
    provider_runtime_qualified: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _text("order group ID", self.order_group_id)
        _text("order snapshot ID", self.order_snapshot_id)
        _digest("order interaction domain", self.interaction_domain_sha256)
        _nonnegative_int("order effective time", self.effective_at_ms)
        _nonnegative_int("order snapshot availability", self.snapshot_available_at_ms)
        if (
            self.complete_logical_event_keys
            != tuple(sorted(set(self.complete_logical_event_keys)))
            or len(self.complete_logical_event_keys) < 2
            or set(self.ordered_logical_event_keys)
            != set(self.complete_logical_event_keys)
            or len(self.ordered_logical_event_keys)
            != len(self.complete_logical_event_keys)
        ):
            raise MassiveEconomicAuthorityV7Error(
                "provider order snapshot is not one complete group"
            )
        for value in (
            *self.complete_logical_event_keys,
            *self.ordered_logical_event_keys,
            self.raw_provider_source_receipt_sha256,
            self.raw_provider_snapshot_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("provider order snapshot digest", value)
        if not isinstance(self.provider_runtime_qualified, bool):
            raise MassiveEconomicAuthorityV7Error(
                "order snapshot qualification is not boolean"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV7Error(
                "provider order snapshot receipt differs"
            )


def select_massive_order_snapshot_at_origin_v7(
    *,
    interaction_domain_sha256: str,
    effective_at_ms: int,
    logical_event_keys: Sequence[str],
    snapshots: Sequence[MassiveProviderEconomicOrderSnapshotV7],
    decision_at_ms: int,
) -> MassiveProviderEconomicOrderSnapshotV7:
    """Select one latest complete snapshot; never mix per-event observations."""

    domain = _digest("interaction domain", interaction_domain_sha256)
    effective = _nonnegative_int("interaction effective time", effective_at_ms)
    decision = _nonnegative_int("interaction decision time", decision_at_ms)
    inventory = tuple(sorted(set(logical_event_keys)))
    if len(inventory) < 2 or len(inventory) != len(logical_event_keys):
        raise MassiveEconomicAuthorityV7Error(
            "order snapshot selection requires one tied event inventory"
        )
    for value in inventory:
        _digest("interaction logical event key", value)
    candidates: list[MassiveProviderEconomicOrderSnapshotV7] = []
    for snapshot in snapshots:
        snapshot.validate()
        if (
            snapshot.interaction_domain_sha256 == domain
            and snapshot.effective_at_ms == effective
            and snapshot.complete_logical_event_keys == inventory
            and snapshot.snapshot_available_at_ms <= decision
        ):
            candidates.append(snapshot)
    if not candidates:
        raise MassiveEconomicAuthorityV7Error(
            "no origin-available complete order snapshot resolves the tie"
        )
    latest_at = max(row.snapshot_available_at_ms for row in candidates)
    latest = tuple(
        row for row in candidates if row.snapshot_available_at_ms == latest_at
    )
    if len(latest) != 1:
        raise MassiveEconomicAuthorityV7Error(
            "order snapshot has an unresolved same-vintage conflict"
        )
    return latest[0]


def _selected_identity_scope_receipt_v7(
    *,
    events: Sequence[MassiveNativeEconomicObservationV7],
    identity_authority: PITSecurityUniverseAuthority,
) -> str:
    masters = {row.security_id: row for row in identity_authority.security_master}
    return semantic_sha256(
        tuple(
            (
                security_id,
                masters[security_id].corporate_action_chain_id,
                masters[security_id].identity_source_receipt_sha256,
                tuple(
                    sorted(
                        row.identity_mapping_receipt_sha256
                        for row in events
                        if row.security_id == security_id
                    )
                ),
            )
            for security_id in sorted({row.security_id for row in events})
        )
    )


def _required_order_snapshots_v7(
    *,
    events: Sequence[MassiveNativeEconomicObservationV7],
    identity_authority: PITSecurityUniverseAuthority,
    snapshots: Sequence[MassiveProviderEconomicOrderSnapshotV7],
    decision_at_ms: int,
) -> tuple[MassiveProviderEconomicOrderSnapshotV7, ...]:
    masters = {row.security_id: row for row in identity_authority.security_master}
    groups: dict[tuple[str, int], list[str]] = {}
    for row in events:
        master = masters[row.security_id]
        interaction_identity = (
            master.corporate_action_chain_id or f"singleton:{row.security_id}"
        )
        domain = semantic_sha256(
            ("massive-native-economic-interaction-v7", interaction_identity)
        )
        groups.setdefault((domain, row.effective_at_ms), []).append(
            row.logical_event_key
        )
    selected: list[MassiveProviderEconomicOrderSnapshotV7] = []
    for (domain, effective_at_ms), logical_keys in sorted(groups.items()):
        if len(logical_keys) == 1:
            continue
        selected.append(
            select_massive_order_snapshot_at_origin_v7(
                interaction_domain_sha256=domain,
                effective_at_ms=effective_at_ms,
                logical_event_keys=logical_keys,
                snapshots=snapshots,
                decision_at_ms=decision_at_ms,
            )
        )
    expected = tuple(sorted(selected, key=lambda row: row.receipt_sha256))
    supplied = tuple(sorted(snapshots, key=lambda row: row.receipt_sha256))
    if tuple(row.receipt_sha256 for row in expected) != tuple(
        row.receipt_sha256 for row in supplied
    ):
        raise MassiveEconomicAuthorityV7Error(
            "supplied order snapshots are not the exact origin-required inventory"
        )
    return expected


@dataclass(frozen=True, slots=True)
class MassiveResolvedEconomicAuthorityAtOriginV7:
    decision_at_ms: int
    identity_scope_receipt_sha256: str
    selected_native_events: tuple[MassiveNativeEconomicObservationV7, ...]
    cash_accrual_periods: tuple[MassiveCashAccrualPeriodV7, ...]
    selected_order_snapshot_receipts: tuple[str, ...]
    semantic_event_inventory_sha256: str
    semantic_receipt_sha256: str
    provider_runtime_qualified: bool
    provider_audit: MassiveOriginProviderAuditV7
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    schema: str = MASSIVE_ECONOMIC_AUTHORITY_V7_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "decision_at_ms": self.decision_at_ms,
            "identity_scope_receipt_sha256": self.identity_scope_receipt_sha256,
            "selected_native_events": tuple(
                asdict(row) for row in self.selected_native_events
            ),
            "cash_accrual_periods": tuple(
                asdict(row) for row in self.cash_accrual_periods
            ),
            "selected_order_snapshot_receipts": (self.selected_order_snapshot_receipts),
            "semantic_event_inventory_sha256": (self.semantic_event_inventory_sha256),
            "provider_runtime_qualified": self.provider_runtime_qualified,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ECONOMIC_AUTHORITY_V7_SCHEMA:
            raise MassiveEconomicAuthorityV7Error(
                "resolved V7 authority schema differs"
            )
        _nonnegative_int("resolved V7 decision time", self.decision_at_ms)
        for value in (
            self.identity_scope_receipt_sha256,
            self.semantic_event_inventory_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("resolved V7 authority digest", value)
        if not isinstance(self.provider_runtime_qualified, bool):
            raise MassiveEconomicAuthorityV7Error(
                "resolved provider qualification is not boolean"
            )
        logical = tuple(row.logical_event_key for row in self.selected_native_events)
        if len(logical) != len(set(logical)):
            raise MassiveEconomicAuthorityV7Error(
                "resolved native events are duplicated"
            )
        for row in self.selected_native_events:
            row.validate()
            if row.available_at_ms > self.decision_at_ms:
                raise MassiveEconomicAuthorityV7Error(
                    "future native event entered resolved authority"
                )
        periods = tuple(
            sorted(
                self.cash_accrual_periods,
                key=lambda row: (row.period_start_at_ms, row.period_id),
            )
        )
        if self.cash_accrual_periods != periods:
            raise MassiveEconomicAuthorityV7Error(
                "cash accrual periods are not canonical"
            )
        previous_end = -1
        for period in periods:
            period.validate()
            if (
                period.available_at_ms > self.decision_at_ms
                or period.period_start_at_ms < previous_end
            ):
                raise MassiveEconomicAuthorityV7Error(
                    "cash periods overlap or use future evidence"
                )
            previous_end = period.period_end_at_ms
        if self.selected_order_snapshot_receipts != tuple(
            sorted(set(self.selected_order_snapshot_receipts))
        ):
            raise MassiveEconomicAuthorityV7Error(
                "selected order snapshots are not canonical"
            )
        expected_inventory = semantic_sha256(
            {
                "events": tuple(
                    (row.logical_event_key, row.receipt_sha256)
                    for row in self.selected_native_events
                ),
                "cash_periods": tuple(
                    (row.period_id, row.receipt_sha256)
                    for row in self.cash_accrual_periods
                ),
                "order_snapshots": self.selected_order_snapshot_receipts,
            }
        )
        if self.semantic_event_inventory_sha256 != expected_inventory:
            raise MassiveEconomicAuthorityV7Error(
                "resolved V7 semantic inventory differs"
            )
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveEconomicAuthorityV7Error(
                "resolved V7 semantic receipt differs"
            )
        self.provider_audit.validate()
        expected_audit = semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "provider_audit_receipt_sha256": (self.provider_audit.receipt_sha256),
                "raw_source_inventory_sha256": (
                    self.provider_audit.raw_source_inventory_sha256
                ),
            }
        )
        if self.audit_receipt_sha256 != expected_audit:
            raise MassiveEconomicAuthorityV7Error("resolved V7 audit receipt differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ECONOMIC_AUTHORITY_V7_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_AUTHORITY_V7_SOURCE_SCHEMA_SHA256
            or not self.loaded_source.receipt.source_object_key.startswith(
                MASSIVE_ECONOMIC_AUTHORITY_V7_OBJECT_PREFIX
            )
        ):
            raise MassiveEconomicAuthorityV7Error(
                "resolved V7 immutable source differs"
            )


def _resolved_payload(
    *,
    decision_at_ms: int,
    identity_scope_receipt_sha256: str,
    selected_native_events: tuple[MassiveNativeEconomicObservationV7, ...],
    cash_accrual_periods: tuple[MassiveCashAccrualPeriodV7, ...],
    selected_order_snapshot_receipts: tuple[str, ...],
    semantic_event_inventory_sha256: str,
    semantic_receipt_sha256: str,
    provider_runtime_qualified: bool,
    provider_audit: MassiveOriginProviderAuditV7,
    audit_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_ECONOMIC_AUTHORITY_V7_SCHEMA,
        "decision_at_ms": decision_at_ms,
        "identity_scope_receipt_sha256": identity_scope_receipt_sha256,
        "selected_native_events": tuple(asdict(row) for row in selected_native_events),
        "cash_accrual_periods": tuple(asdict(row) for row in cash_accrual_periods),
        "selected_order_snapshot_receipts": selected_order_snapshot_receipts,
        "semantic_event_inventory_sha256": semantic_event_inventory_sha256,
        "semantic_receipt_sha256": semantic_receipt_sha256,
        "provider_runtime_qualified": provider_runtime_qualified,
        "provider_audit": asdict(provider_audit),
        "audit_receipt_sha256": audit_receipt_sha256,
    }


def materialize_massive_resolved_economic_authority_at_origin_v7(
    *,
    root: str | Path,
    capture_loaded_sources: Sequence[LoadedMassiveSourceObject],
    identity_authority: PITSecurityUniverseAuthority,
    decision_at_ms: int,
    cash_accrual_periods: Sequence[MassiveCashAccrualPeriodV7],
    selected_order_snapshots: Sequence[MassiveProviderEconomicOrderSnapshotV7],
    entitlement_receipt_sha256: str,
    artifact_id: str,
    committed_at_ms: int,
) -> MassiveResolvedEconomicAuthorityAtOriginV7:
    """Reparse raw captures and publish one dual-receipt origin authority."""

    captures = tuple(
        parse_massive_economic_raw_rest_capture_v7(root=root, loaded_source=loaded)
        for loaded in capture_loaded_sources
    )
    events, audit = adapt_massive_raw_economic_captures_at_origin_v7(
        captures=captures,
        identity_authority=identity_authority,
        decision_at_ms=decision_at_ms,
    )
    periods = tuple(
        sorted(
            cash_accrual_periods,
            key=lambda row: (row.period_start_at_ms, row.period_id),
        )
    )
    previous_period_end = -1
    for period in periods:
        period.validate()
        if (
            period.available_at_ms > decision_at_ms
            or period.period_start_at_ms < previous_period_end
        ):
            raise MassiveEconomicAuthorityV7Error(
                "future or overlapping cash period entered origin authority"
            )
        previous_period_end = period.period_end_at_ms
    required_snapshots = _required_order_snapshots_v7(
        events=events,
        identity_authority=identity_authority,
        snapshots=selected_order_snapshots,
        decision_at_ms=decision_at_ms,
    )
    snapshot_receipts = tuple(
        snapshot.receipt_sha256 for snapshot in required_snapshots
    )
    for snapshot in required_snapshots:
        snapshot.validate()
        if snapshot.snapshot_available_at_ms > decision_at_ms:
            raise MassiveEconomicAuthorityV7Error(
                "future order snapshot entered origin authority"
            )
    supplemental_receipts = tuple(
        sorted(
            {
                *(row.raw_provider_source_receipt_sha256 for row in periods),
                *(row.raw_provider_source_receipt_sha256 for row in required_snapshots),
            }
        )
    )
    audit_provisional = replace(
        audit,
        supplemental_source_receipts=supplemental_receipts,
        raw_source_inventory_sha256=semantic_sha256(
            (audit.raw_capture_receipts, supplemental_receipts)
        ),
        receipt_sha256="0" * 64,
    )
    audit = replace(
        audit_provisional,
        receipt_sha256=semantic_sha256(audit_provisional.unsigned()),
    )
    audit.validate()
    inventory = semantic_sha256(
        {
            "events": tuple(
                (row.logical_event_key, row.receipt_sha256) for row in events
            ),
            "cash_periods": tuple(
                (row.period_id, row.receipt_sha256) for row in periods
            ),
            "order_snapshots": snapshot_receipts,
        }
    )
    provider_qualified = (
        bool(captures)
        and all(capture.provider_runtime_qualified for capture in captures)
        and all(period.provider_runtime_qualified for period in periods)
        and all(snapshot.provider_runtime_qualified for snapshot in required_snapshots)
    )
    identity_scope_receipt = _selected_identity_scope_receipt_v7(
        events=events,
        identity_authority=identity_authority,
    )
    semantic_unsigned = {
        "schema": MASSIVE_ECONOMIC_AUTHORITY_V7_SCHEMA,
        "decision_at_ms": decision_at_ms,
        "identity_scope_receipt_sha256": identity_scope_receipt,
        "selected_native_events": tuple(asdict(row) for row in events),
        "cash_accrual_periods": tuple(asdict(row) for row in periods),
        "selected_order_snapshot_receipts": snapshot_receipts,
        "semantic_event_inventory_sha256": inventory,
        "provider_runtime_qualified": provider_qualified,
    }
    semantic_receipt = semantic_sha256(semantic_unsigned)
    audit_receipt = semantic_sha256(
        {
            "semantic_receipt_sha256": semantic_receipt,
            "provider_audit_receipt_sha256": audit.receipt_sha256,
            "raw_source_inventory_sha256": audit.raw_source_inventory_sha256,
        }
    )
    payload = _resolved_payload(
        decision_at_ms=decision_at_ms,
        identity_scope_receipt_sha256=identity_scope_receipt,
        selected_native_events=events,
        cash_accrual_periods=periods,
        selected_order_snapshot_receipts=snapshot_receipts,
        semantic_event_inventory_sha256=inventory,
        semantic_receipt_sha256=semantic_receipt,
        provider_runtime_qualified=provider_qualified,
        provider_audit=audit,
        audit_receipt_sha256=audit_receipt,
    )
    relative = (
        f"{MASSIVE_ECONOMIC_AUTHORITY_V7_OBJECT_PREFIX}"
        f"{_text('resolved authority artifact ID', artifact_id)}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ECONOMIC_AUTHORITY_V7_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ECONOMIC_AUTHORITY_V7_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "resolved authority entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"MASSIVE-ECO-V7-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return parse_massive_resolved_economic_authority_at_origin_v7(
        root=root, loaded_source=loaded
    )


def parse_massive_resolved_economic_authority_at_origin_v7(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveResolvedEconomicAuthorityAtOriginV7:
    """Reopen and independently validate one committed V7 origin authority."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicAuthorityV7Error(
            "resolved V7 authority is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicAuthorityV7Error(
            "resolved V7 authority is not canonical JSON"
        )
    _exact_keys(
        payload,
        {
            "schema",
            "decision_at_ms",
            "identity_scope_receipt_sha256",
            "selected_native_events",
            "cash_accrual_periods",
            "selected_order_snapshot_receipts",
            "semantic_event_inventory_sha256",
            "semantic_receipt_sha256",
            "provider_runtime_qualified",
            "provider_audit",
            "audit_receipt_sha256",
        },
        name="resolved V7 authority",
    )
    raw_events = payload["selected_native_events"]
    raw_periods = payload["cash_accrual_periods"]
    raw_snapshots = payload["selected_order_snapshot_receipts"]
    raw_audit = payload["provider_audit"]
    if (
        not isinstance(raw_events, list)
        or any(not isinstance(row, dict) for row in raw_events)
        or not isinstance(raw_periods, list)
        or any(not isinstance(row, dict) for row in raw_periods)
        or not isinstance(raw_snapshots, list)
        or not isinstance(raw_audit, dict)
    ):
        raise MassiveEconomicAuthorityV7Error(
            "resolved V7 nested inventory is malformed"
        )
    events = tuple(MassiveNativeEconomicObservationV7(**row) for row in raw_events)
    periods = tuple(MassiveCashAccrualPeriodV7(**row) for row in raw_periods)
    audit_fields = dict(raw_audit)
    for name in (
        "raw_capture_receipts",
        "raw_page_receipts",
        "future_page_receipts",
        "supplemental_source_receipts",
        "issue_inventory",
    ):
        value = audit_fields[name]
        if not isinstance(value, list):
            raise MassiveEconomicAuthorityV7Error(
                "provider audit nested inventory is malformed"
            )
        audit_fields[name] = tuple(
            tuple(row) if isinstance(row, list) else row for row in value
        )
    audit = MassiveOriginProviderAuditV7(**audit_fields)
    result = MassiveResolvedEconomicAuthorityAtOriginV7(
        schema=payload["schema"],
        decision_at_ms=payload["decision_at_ms"],
        identity_scope_receipt_sha256=payload["identity_scope_receipt_sha256"],
        selected_native_events=events,
        cash_accrual_periods=periods,
        selected_order_snapshot_receipts=tuple(raw_snapshots),
        semantic_event_inventory_sha256=payload["semantic_event_inventory_sha256"],
        semantic_receipt_sha256=payload["semantic_receipt_sha256"],
        provider_runtime_qualified=payload["provider_runtime_qualified"],
        provider_audit=audit,
        audit_receipt_sha256=payload["audit_receipt_sha256"],
        loaded_source=loaded_source,
    )
    regenerated = _resolved_payload(
        decision_at_ms=result.decision_at_ms,
        identity_scope_receipt_sha256=result.identity_scope_receipt_sha256,
        selected_native_events=result.selected_native_events,
        cash_accrual_periods=result.cash_accrual_periods,
        selected_order_snapshot_receipts=(result.selected_order_snapshot_receipts),
        semantic_event_inventory_sha256=(result.semantic_event_inventory_sha256),
        semantic_receipt_sha256=result.semantic_receipt_sha256,
        provider_runtime_qualified=result.provider_runtime_qualified,
        provider_audit=result.provider_audit,
        audit_receipt_sha256=result.audit_receipt_sha256,
    )
    if raw != canonical_json_file_bytes(regenerated):
        raise MassiveEconomicAuthorityV7Error("resolved V7 regenerated payload differs")
    result.validate()
    return result


MASSIVE_ECONOMIC_AUTHORITY_V7_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_AUTHORITY_V7_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "raw_capture_spec": MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SPEC_SHA256,
        "native_surfaces": tuple(sorted(_MASSIVE_DATE_FIELDS)),
        "availability": "raw-provider-field-or-conservative-response-completion",
        "cash": MASSIVE_CASH_ACCRUAL_CONVENTION_V7,
        "security_ties": "one-latest-complete-origin-available-group-snapshot",
        "future_invalid_rows": "audit-only-after-origin-availability-boundary",
        "receipts": "future-invariant-semantic-plus-complete-source-audit",
        "committed_origin_artifact": MASSIVE_ECONOMIC_AUTHORITY_V7_SCHEMA,
        "source_sha256": MASSIVE_ECONOMIC_AUTHORITY_V7_SOURCE_SHA256,
        "historical_panel_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
    }
)


__all__ = [
    "MASSIVE_CASH_ACCRUAL_CONVENTION_V7",
    "MASSIVE_CASH_ACCRUAL_CONVENTION_V7_RECEIPT_SHA256",
    "MASSIVE_ECONOMIC_AUTHORITY_V7_DATASET",
    "MASSIVE_ECONOMIC_AUTHORITY_V7_HISTORICAL_PANEL_AUTHORIZED",
    "MASSIVE_ECONOMIC_AUTHORITY_V7_OBJECT_PREFIX",
    "MASSIVE_ECONOMIC_AUTHORITY_V7_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_ECONOMIC_AUTHORITY_V7_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_ECONOMIC_AUTHORITY_V7_SCHEMA",
    "MASSIVE_ECONOMIC_AUTHORITY_V7_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_AUTHORITY_V7_SPEC_SHA256",
    "MASSIVE_NATIVE_AVAILABILITY_RULE_V7_RECEIPT_SHA256",
    "MASSIVE_SINGLE_SECURITY_EVENT_ORDER_RULE_V7_RECEIPT_SHA256",
    "MassiveCashAccrualPeriodV7",
    "MassiveCashLotV7",
    "MassiveEconomicAuthorityV7Error",
    "MassiveNativeEconomicObservationV7",
    "MassiveOriginProviderAuditV7",
    "MassiveProviderEconomicOrderSnapshotV7",
    "MassiveResolvedEconomicAuthorityAtOriginV7",
    "accrue_massive_cash_lots_v7",
    "adapt_massive_raw_economic_captures_at_origin_v7",
    "materialize_massive_resolved_economic_authority_at_origin_v7",
    "parse_massive_resolved_economic_authority_at_origin_v7",
    "select_massive_order_snapshot_at_origin_v7",
]
