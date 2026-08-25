"""Provider-bound, origin-scoped economic authority for profitability P0 V6.

V6 is deliberately an authority boundary, not another full-history materializer.
It reparses exact raw provider-observation bytes, derives logical/revision identity,
effective timestamps, interaction domains, and point-in-time order, then emits a
semantic origin receipt that is independent of future-unavailable archive rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import Any

from rl_quant.alpha.contracts import (
    CashReturnRecord,
    CorporateActionKind,
    CorporateActionRecord,
    TerminalEventKind,
    TerminalEventRecord,
)
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_economic_event_source_v5 import (
    MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5,
    MassiveSourcedCashReturnV5,
    MassiveSourcedCorporateActionV5,
    MassiveSourcedEconomicEventV5,
    MassiveSourcedTerminalEventV5,
    cash_economic_fingerprint_v5,
    corporate_economic_fingerprint_v5,
    expected_effective_timestamp_contract_v5,
    sourced_available_at_ms_v5,
    sourced_effective_at_ms_v5,
    terminal_economic_fingerprint_v5,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)


MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA = (
    "rl-quant.massive-raw-provider-economic-source-v6"
)
MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS = (
    "corporate-actions",
    "terminal-outcomes",
    "cash-returns",
    "economic-order-observations",
)
MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_DATASETS = {
    role: f"massive-raw-provider-{role}-v6"
    for role in MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS
}
MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_OBJECT_PREFIX = (
    "massive-profitability-p0/raw-provider-economic-v6/"
)
MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA,
        "roles": MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS,
        "payload": "exact-canonical-raw-provider-observation-pages",
    }
)
MASSIVE_ECONOMIC_REVISION_STATUSES_V6 = ("active", "corrected", "cancelled")
MASSIVE_ECONOMIC_AUTHORITY_V6_HISTORICAL_PANEL_AUTHORIZED = False
MASSIVE_ECONOMIC_AUTHORITY_V6_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_ECONOMIC_AUTHORITY_V6_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_ECONOMIC_CASH_INTERACTION_DOMAIN_V6 = semantic_sha256(
    ("massive-profitability-p0-v6", "cash-accrual-ledger")
)
MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V6_RECEIPT_SHA256 = semantic_sha256(
    {
        "rule": "one-origin-selected-logical-event-in-interaction-domain-at-time",
        "local_sequence": 0,
        "provider_order_observation": "prohibited-as-unnecessary",
    }
)

_CORPORATE_EFFECTIVE_FIELDS = {
    CorporateActionKind.CASH_DIVIDEND.value: "ex_dividend_at_ms",
    CorporateActionKind.SPECIAL_DIVIDEND.value: "ex_dividend_at_ms",
    CorporateActionKind.RETURN_OF_CAPITAL.value: "ex_dividend_at_ms",
    CorporateActionKind.SPLIT.value: "execution_at_ms",
    CorporateActionKind.REVERSE_SPLIT.value: "execution_at_ms",
    CorporateActionKind.SPIN_OFF.value: "entitlement_at_ms",
    CorporateActionKind.MERGER_CASH.value: "legal_disposition_at_ms",
    CorporateActionKind.MERGER_STOCK.value: "legal_disposition_at_ms",
    CorporateActionKind.TENDER_OFFER.value: "accepted_settlement_at_ms",
    CorporateActionKind.RIGHTS_DISTRIBUTION.value: "entitlement_at_ms",
    CorporateActionKind.TICKER_EXCHANGE_CHANGE.value: "first_new_reference_at_ms",
}
_TERMINAL_EFFECTIVE_FIELDS = {
    TerminalEventKind.DELISTING_CASH.value: "final_disposition_at_ms",
    TerminalEventKind.MERGER_CASH.value: "legal_disposition_at_ms",
    TerminalEventKind.MERGER_STOCK.value: "legal_disposition_at_ms",
    TerminalEventKind.BANKRUPTCY_RECOVERY.value: "recovery_disposition_at_ms",
    TerminalEventKind.WORTHLESS.value: "zero_value_effective_at_ms",
}
_CASH_EFFECTIVE_FIELD = "accrual_period_end_at_ms"


class MassiveEconomicAuthorityV6Error(ValueError):
    """Raw provider evidence or its origin-scoped resolution differs."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEconomicAuthorityV6Error(f"{name} must be canonical text")
    return value


def _optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _text(name, value)


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicAuthorityV6Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicAuthorityV6Error(f"{name} must be nonnegative")
    return value


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveEconomicAuthorityV6Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MassiveEconomicAuthorityV6Error(f"{name} must be finite")
    return result


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveEconomicAuthorityV6Error(f"{name} field inventory differs")


def _revision_status(value: object) -> str:
    result = _text("provider revision status", value)
    if result not in MASSIVE_ECONOMIC_REVISION_STATUSES_V6:
        raise MassiveEconomicAuthorityV6Error("provider revision status is unsupported")
    return result


def _logical_event_key(
    *,
    source_kind: str,
    provider_id: str,
    provider_dataset: str,
    provider_event_key: str,
) -> str:
    return semantic_sha256(
        {
            "provider_id": provider_id,
            "provider_dataset": provider_dataset,
            "source_kind": source_kind,
            "provider_event_key": provider_event_key,
        }
    )


def _revision_id(*, logical_event_key: str, provider_revision_id: str) -> str:
    return semantic_sha256((logical_event_key, provider_revision_id))


def _event_id(*, logical_event_key: str, provider_revision_id: str) -> str:
    return f"PROVIDER:{semantic_sha256((logical_event_key, provider_revision_id, 'event'))}"


def _raw_record_receipt(
    *,
    source_kind: str,
    provider_id: str,
    provider_dataset: str,
    provider_request_id: str,
    provider_row_locator: str,
    record: Mapping[str, object],
) -> str:
    return semantic_sha256(
        {
            "source_kind": source_kind,
            "provider_id": provider_id,
            "provider_dataset": provider_dataset,
            "provider_request_id": provider_request_id,
            "provider_row_locator": provider_row_locator,
            "raw_provider_record": dict(record),
        }
    )


@dataclass(frozen=True, slots=True)
class MassiveRawProviderEventObservationV6:
    source_kind: str
    provider_id: str
    provider_dataset: str
    provider_request_id: str
    provider_event_key: str
    provider_revision_id: str
    supersedes_provider_revision_id: str | None
    revision_status: str
    provider_available_at_ms: int
    provider_row_locator: str
    raw_provider_record_sha256: str
    observation_receipt_sha256: str
    source_event: MassiveSourcedEconomicEventV5

    def validate(self) -> None:
        if self.source_kind not in MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS[:3]:
            raise MassiveEconomicAuthorityV6Error("provider event source role differs")
        for name in (
            "provider_id",
            "provider_dataset",
            "provider_request_id",
            "provider_event_key",
            "provider_revision_id",
            "provider_row_locator",
        ):
            _text(name, getattr(self, name))
        _optional_text(
            "superseded provider revision", self.supersedes_provider_revision_id
        )
        _revision_status(self.revision_status)
        _nonnegative_int("provider event availability", self.provider_available_at_ms)
        _digest("raw provider record", self.raw_provider_record_sha256)
        _digest("provider observation", self.observation_receipt_sha256)
        self.source_event.validate()
        expected_logical = _logical_event_key(
            source_kind=self.source_kind,
            provider_id=self.provider_id,
            provider_dataset=self.provider_dataset,
            provider_event_key=self.provider_event_key,
        )
        if (
            self.source_event.provider_event_key != self.provider_event_key
            or self.source_event.logical_event_key != expected_logical
            or self.source_event.revision_id
            != _revision_id(
                logical_event_key=expected_logical,
                provider_revision_id=self.provider_revision_id,
            )
            or self.source_event.supersedes_revision_id
            != (
                None
                if self.supersedes_provider_revision_id is None
                else _revision_id(
                    logical_event_key=expected_logical,
                    provider_revision_id=self.supersedes_provider_revision_id,
                )
            )
            or self.source_event.revision_status != self.revision_status
            or sourced_available_at_ms_v5(self.source_event)
            != self.provider_available_at_ms
        ):
            raise MassiveEconomicAuthorityV6Error(
                "canonical event was not derived from its provider identity"
            )


@dataclass(frozen=True, slots=True)
class MassiveRawProviderOrderObservationV6:
    provider_id: str
    provider_dataset: str
    provider_request_id: str
    event_provider_id: str
    event_provider_dataset: str
    event_source_kind: str
    provider_event_key: str
    logical_event_key: str
    effective_at_ms: int
    provider_local_economic_sequence: int
    provider_order_available_at_ms: int
    provider_row_locator: str
    raw_provider_record_sha256: str
    observation_receipt_sha256: str

    def validate(self) -> None:
        for name in (
            "provider_id",
            "provider_dataset",
            "provider_request_id",
            "event_provider_id",
            "event_provider_dataset",
            "provider_event_key",
            "provider_row_locator",
        ):
            _text(name, getattr(self, name))
        if (
            self.event_source_kind
            not in MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS[:3]
        ):
            raise MassiveEconomicAuthorityV6Error("provider-order event role differs")
        if self.logical_event_key != _logical_event_key(
            source_kind=self.event_source_kind,
            provider_id=self.event_provider_id,
            provider_dataset=self.event_provider_dataset,
            provider_event_key=self.provider_event_key,
        ):
            raise MassiveEconomicAuthorityV6Error("provider-order logical key differs")
        _nonnegative_int("provider-order effective time", self.effective_at_ms)
        sequence = _nonnegative_int(
            "provider local economic sequence",
            self.provider_local_economic_sequence,
        )
        if sequence >= MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5:
            raise MassiveEconomicAuthorityV6Error(
                "provider sequence reaches base sentinel"
            )
        _nonnegative_int(
            "provider-order availability", self.provider_order_available_at_ms
        )
        _digest("raw provider-order record", self.raw_provider_record_sha256)
        _digest("provider-order observation", self.observation_receipt_sha256)


@dataclass(frozen=True, slots=True)
class MassiveRawProviderEconomicSourceV6:
    source_kind: str
    provider_id: str
    provider_dataset: str
    provider_endpoint: str
    query_start_at_ms: int
    query_end_at_ms: int
    provider_observed_at_ms: int
    provider_request_ids: tuple[str, ...]
    pagination_complete: bool
    page_count: int
    event_observations: tuple[MassiveRawProviderEventObservationV6, ...]
    order_observations: tuple[MassiveRawProviderOrderObservationV6, ...]
    row_inventory_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA
            or self.source_kind not in MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS
        ):
            raise MassiveEconomicAuthorityV6Error(
                "raw provider source identity differs"
            )
        for name in ("provider_id", "provider_dataset", "provider_endpoint"):
            _text(name, getattr(self, name))
        if not self.provider_endpoint.startswith("https://"):
            raise MassiveEconomicAuthorityV6Error("provider endpoint must use HTTPS")
        start = _nonnegative_int("provider query start", self.query_start_at_ms)
        end = _nonnegative_int("provider query end", self.query_end_at_ms)
        observed = _nonnegative_int(
            "provider observed time", self.provider_observed_at_ms
        )
        if end < start or observed < end:
            raise MassiveEconomicAuthorityV6Error("provider query chronology differs")
        if (
            not self.pagination_complete
            or self.page_count <= 0
            or len(self.provider_request_ids) != self.page_count
            or len(set(self.provider_request_ids)) != self.page_count
        ):
            raise MassiveEconomicAuthorityV6Error("provider pagination is incomplete")
        for request_id in self.provider_request_ids:
            _text("provider request ID", request_id)
        if self.source_kind == "economic-order-observations":
            if self.event_observations:
                raise MassiveEconomicAuthorityV6Error(
                    "provider-order source role differs"
                )
            records: Sequence[
                MassiveRawProviderEventObservationV6
                | MassiveRawProviderOrderObservationV6
            ] = self.order_observations
        else:
            if self.order_observations:
                raise MassiveEconomicAuthorityV6Error(
                    "provider event source role differs"
                )
            records = self.event_observations
        for row in records:
            row.validate()
            if (
                row.provider_id != self.provider_id
                or row.provider_dataset != self.provider_dataset
                or row.provider_request_id not in self.provider_request_ids
            ):
                raise MassiveEconomicAuthorityV6Error(
                    "provider observation differs from its capture"
                )
            row_available_at_ms = (
                row.provider_available_at_ms
                if isinstance(row, MassiveRawProviderEventObservationV6)
                else row.provider_order_available_at_ms
            )
            if row_available_at_ms > observed:
                raise MassiveEconomicAuthorityV6Error(
                    "provider observation claims future availability"
                )
        row_keys = tuple(
            (row.provider_row_locator, row.observation_receipt_sha256)
            for row in records
        )
        if row_keys != tuple(sorted(set(row_keys))):
            raise MassiveEconomicAuthorityV6Error(
                "provider observations are not canonical and unique"
            )
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.observation_receipt_sha256 for row in records)
        ):
            raise MassiveEconomicAuthorityV6Error("provider row inventory differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_DATASETS[self.source_kind]
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256
            or not self.loaded_source.receipt.source_object_key.startswith(
                MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_OBJECT_PREFIX
            )
            or self.loaded_source.receipt.request_id != self.provider_request_ids[-1]
            or self.loaded_source.receipt.etag is None
            or self.loaded_source.receipt.downloaded_at_ms
            != self.provider_observed_at_ms
        ):
            raise MassiveEconomicAuthorityV6Error(
                "provider capture source transaction differs"
            )
        _digest("raw provider source receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV6Error("raw provider source receipt differs")


def _provider_record(
    record: Mapping[str, object],
    *,
    expected_effective_field: str,
) -> tuple[int, Mapping[str, object]]:
    raw = record.get("provider_record")
    if not isinstance(raw, dict):
        raise MassiveEconomicAuthorityV6Error("raw provider record is malformed")
    if expected_effective_field not in raw:
        raise MassiveEconomicAuthorityV6Error(
            "required provider effective-time field is absent"
        )
    return _nonnegative_int(
        "provider-derived effective time", raw[expected_effective_field]
    ), raw


def _parse_event_observation(
    *,
    source_kind: str,
    provider_id: str,
    provider_dataset: str,
    record: Mapping[str, object],
) -> MassiveRawProviderEventObservationV6:
    _exact_keys(
        record,
        {
            "provider_request_id",
            "provider_event_key",
            "provider_revision_id",
            "supersedes_provider_revision_id",
            "revision_status",
            "provider_available_at_ms",
            "provider_row_locator",
            "provider_record",
        },
        name="raw provider event observation",
    )
    request_id = _text("provider request ID", record["provider_request_id"])
    provider_key = _text("provider event key", record["provider_event_key"])
    provider_revision = _text("provider revision ID", record["provider_revision_id"])
    supersedes_provider = _optional_text(
        "superseded provider revision", record["supersedes_provider_revision_id"]
    )
    status = _revision_status(record["revision_status"])
    available_at_ms = _nonnegative_int(
        "provider availability", record["provider_available_at_ms"]
    )
    locator = _text("provider row locator", record["provider_row_locator"])
    raw_value = record["provider_record"]
    assert isinstance(raw_value, dict)
    event_kind = _text("provider event kind", raw_value.get("kind"))
    if source_kind == "corporate-actions":
        try:
            kind = CorporateActionKind(event_kind)
            effective_field = _CORPORATE_EFFECTIVE_FIELDS[event_kind]
        except (ValueError, KeyError) as exc:
            raise MassiveEconomicAuthorityV6Error(
                "provider corporate-action kind is unsupported"
            ) from exc
        expected = {
            "kind",
            "security_id",
            effective_field,
            "cash_per_share",
            "share_ratio",
            "successor_security_id",
            "successor_ratio",
            "affected_fraction",
        }
        _exact_keys(raw_value, expected, name="raw provider corporate record")
        effective_at_ms, _ = _provider_record(
            record, expected_effective_field=effective_field
        )
        logical_key = _logical_event_key(
            source_kind=source_kind,
            provider_id=provider_id,
            provider_dataset=provider_dataset,
            provider_event_key=provider_key,
        )
        event = CorporateActionRecord(
            event_id=_event_id(
                logical_event_key=logical_key,
                provider_revision_id=provider_revision,
            ),
            security_id=_text("provider security ID", raw_value["security_id"]),
            kind=kind,
            effective_at_ms=effective_at_ms,
            available_at_ms=available_at_ms,
            cash_per_share=_number(
                "provider cash per share", raw_value["cash_per_share"]
            ),
            share_ratio=_number("provider share ratio", raw_value["share_ratio"]),
            successor_security_id=_optional_text(
                "provider successor security", raw_value["successor_security_id"]
            ),
            successor_ratio=_number(
                "provider successor ratio", raw_value["successor_ratio"]
            ),
            affected_fraction=_number(
                "provider affected fraction", raw_value["affected_fraction"]
            ),
        )
        event.validate()
        fingerprint = corporate_economic_fingerprint_v5(event)
        provisional_source: MassiveSourcedEconomicEventV5 = (
            MassiveSourcedCorporateActionV5(
                event=event,
                provider_event_key=provider_key,
                logical_event_key=logical_key,
                revision_id=_revision_id(
                    logical_event_key=logical_key,
                    provider_revision_id=provider_revision,
                ),
                supersedes_revision_id=None
                if supersedes_provider is None
                else _revision_id(
                    logical_event_key=logical_key,
                    provider_revision_id=supersedes_provider,
                ),
                revision_status=status,
                economic_fingerprint_sha256=fingerprint,
                effective_timestamp_contract=expected_effective_timestamp_contract_v5(
                    source_kind=source_kind, event_kind=kind.value
                ),
                upstream_source_receipt_sha256="0" * 64,
                source_row_receipt_sha256="0" * 64,
                receipt_sha256="0" * 64,
            )
        )
    elif source_kind == "terminal-outcomes":
        try:
            terminal_kind = TerminalEventKind(event_kind)
            effective_field = _TERMINAL_EFFECTIVE_FIELDS[event_kind]
        except (ValueError, KeyError) as exc:
            raise MassiveEconomicAuthorityV6Error(
                "provider terminal-event kind is unsupported"
            ) from exc
        expected = {
            "kind",
            "security_id",
            effective_field,
            "cash_per_share",
            "successor_security_id",
            "successor_ratio",
        }
        _exact_keys(raw_value, expected, name="raw provider terminal record")
        effective_at_ms, _ = _provider_record(
            record, expected_effective_field=effective_field
        )
        logical_key = _logical_event_key(
            source_kind=source_kind,
            provider_id=provider_id,
            provider_dataset=provider_dataset,
            provider_event_key=provider_key,
        )
        terminal = TerminalEventRecord(
            event_id=_event_id(
                logical_event_key=logical_key,
                provider_revision_id=provider_revision,
            ),
            security_id=_text("provider security ID", raw_value["security_id"]),
            kind=terminal_kind,
            effective_at_ms=effective_at_ms,
            available_at_ms=available_at_ms,
            cash_per_share=_number(
                "provider cash per share", raw_value["cash_per_share"]
            ),
            successor_security_id=_optional_text(
                "provider successor security", raw_value["successor_security_id"]
            ),
            successor_ratio=_number(
                "provider successor ratio", raw_value["successor_ratio"]
            ),
        )
        terminal.validate()
        fingerprint = terminal_economic_fingerprint_v5(terminal)
        provisional_source = MassiveSourcedTerminalEventV5(
            event=terminal,
            provider_event_key=provider_key,
            logical_event_key=logical_key,
            revision_id=_revision_id(
                logical_event_key=logical_key,
                provider_revision_id=provider_revision,
            ),
            supersedes_revision_id=None
            if supersedes_provider is None
            else _revision_id(
                logical_event_key=logical_key,
                provider_revision_id=supersedes_provider,
            ),
            revision_status=status,
            economic_fingerprint_sha256=fingerprint,
            effective_timestamp_contract=expected_effective_timestamp_contract_v5(
                source_kind=source_kind, event_kind=terminal_kind.value
            ),
            upstream_source_receipt_sha256="0" * 64,
            source_row_receipt_sha256="0" * 64,
            receipt_sha256="0" * 64,
        )
    else:
        if source_kind != "cash-returns" or event_kind != "cash-return":
            raise MassiveEconomicAuthorityV6Error("provider cash event kind differs")
        expected = {"kind", _CASH_EFFECTIVE_FIELD, "one_step_return"}
        _exact_keys(raw_value, expected, name="raw provider cash record")
        effective_at_ms, _ = _provider_record(
            record, expected_effective_field=_CASH_EFFECTIVE_FIELD
        )
        logical_key = _logical_event_key(
            source_kind=source_kind,
            provider_id=provider_id,
            provider_dataset=provider_dataset,
            provider_event_key=provider_key,
        )
        cash = CashReturnRecord(
            effective_at_ms=effective_at_ms,
            available_at_ms=available_at_ms,
            one_step_return=_number(
                "provider one-step cash return", raw_value["one_step_return"]
            ),
            source_receipt_sha256="0" * 64,
        )
        # Source receipt is filled with the row observation below.
        fingerprint = cash_economic_fingerprint_v5(
            effective_at_ms=effective_at_ms,
            one_step_return=cash.one_step_return,
        )
        provisional_source = MassiveSourcedCashReturnV5(
            event_id=_event_id(
                logical_event_key=logical_key,
                provider_revision_id=provider_revision,
            ),
            cash_return=cash,
            provider_event_key=provider_key,
            logical_event_key=logical_key,
            revision_id=_revision_id(
                logical_event_key=logical_key,
                provider_revision_id=provider_revision,
            ),
            supersedes_revision_id=None
            if supersedes_provider is None
            else _revision_id(
                logical_event_key=logical_key,
                provider_revision_id=supersedes_provider,
            ),
            revision_status=status,
            economic_fingerprint_sha256=fingerprint,
            effective_timestamp_contract=expected_effective_timestamp_contract_v5(
                source_kind=source_kind, event_kind="cash-return"
            ),
            source_row_receipt_sha256="0" * 64,
            receipt_sha256="0" * 64,
        )
    raw_sha = semantic_sha256(raw_value)
    observation_receipt = _raw_record_receipt(
        source_kind=source_kind,
        provider_id=provider_id,
        provider_dataset=provider_dataset,
        provider_request_id=request_id,
        provider_row_locator=locator,
        record=record,
    )
    source_provisional: MassiveSourcedEconomicEventV5
    if isinstance(provisional_source, MassiveSourcedCashReturnV5):
        cash_with_receipt = replace(
            provisional_source.cash_return,
            source_receipt_sha256=observation_receipt,
        )
        source_provisional = replace(
            provisional_source,
            cash_return=cash_with_receipt,
            source_row_receipt_sha256=observation_receipt,
        )
    else:
        source_provisional = replace(
            provisional_source,
            upstream_source_receipt_sha256=observation_receipt,
            source_row_receipt_sha256=observation_receipt,
        )
    source_event = replace(
        source_provisional,
        receipt_sha256=semantic_sha256(source_provisional.unsigned()),
    )
    source_event.validate()
    result = MassiveRawProviderEventObservationV6(
        source_kind=source_kind,
        provider_id=provider_id,
        provider_dataset=provider_dataset,
        provider_request_id=request_id,
        provider_event_key=provider_key,
        provider_revision_id=provider_revision,
        supersedes_provider_revision_id=supersedes_provider,
        revision_status=status,
        provider_available_at_ms=available_at_ms,
        provider_row_locator=locator,
        raw_provider_record_sha256=raw_sha,
        observation_receipt_sha256=observation_receipt,
        source_event=source_event,
    )
    result.validate()
    return result


def _parse_order_observation(
    *, provider_id: str, provider_dataset: str, record: Mapping[str, object]
) -> MassiveRawProviderOrderObservationV6:
    _exact_keys(
        record,
        {
            "provider_request_id",
            "event_provider_id",
            "event_provider_dataset",
            "event_source_kind",
            "provider_event_key",
            "provider_order_available_at_ms",
            "provider_row_locator",
            "provider_record",
        },
        name="raw provider order observation",
    )
    request_id = _text("provider request ID", record["provider_request_id"])
    event_provider_id = _text("event provider ID", record["event_provider_id"])
    event_provider_dataset = _text(
        "event provider dataset", record["event_provider_dataset"]
    )
    source_kind = _text("ordered event source kind", record["event_source_kind"])
    provider_key = _text("ordered provider event key", record["provider_event_key"])
    available_at_ms = _nonnegative_int(
        "provider order availability", record["provider_order_available_at_ms"]
    )
    locator = _text("provider-order row locator", record["provider_row_locator"])
    raw = record["provider_record"]
    if not isinstance(raw, dict):
        raise MassiveEconomicAuthorityV6Error("raw provider-order record is malformed")
    _exact_keys(
        raw,
        {"provider_effective_at_ms", "provider_local_economic_sequence"},
        name="raw provider-order record",
    )
    result = MassiveRawProviderOrderObservationV6(
        provider_id=provider_id,
        provider_dataset=provider_dataset,
        provider_request_id=request_id,
        event_provider_id=event_provider_id,
        event_provider_dataset=event_provider_dataset,
        event_source_kind=source_kind,
        provider_event_key=provider_key,
        logical_event_key=_logical_event_key(
            source_kind=source_kind,
            provider_id=event_provider_id,
            provider_dataset=event_provider_dataset,
            provider_event_key=provider_key,
        ),
        effective_at_ms=_nonnegative_int(
            "provider-order effective time", raw["provider_effective_at_ms"]
        ),
        provider_local_economic_sequence=_nonnegative_int(
            "provider local sequence", raw["provider_local_economic_sequence"]
        ),
        provider_order_available_at_ms=available_at_ms,
        provider_row_locator=locator,
        raw_provider_record_sha256=semantic_sha256(raw),
        observation_receipt_sha256=_raw_record_receipt(
            source_kind="economic-order-observations",
            provider_id=provider_id,
            provider_dataset=provider_dataset,
            provider_request_id=request_id,
            provider_row_locator=locator,
            record=record,
        ),
    )
    result.validate()
    return result


def parse_massive_raw_provider_economic_source_v6(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveRawProviderEconomicSourceV6:
    """Reopen one exact raw provider-observation capture."""

    loaded_source.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicAuthorityV6Error(
            "raw provider source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicAuthorityV6Error(
            "raw provider source is not canonical JSON"
        )
    _exact_keys(
        payload,
        {
            "schema",
            "source_kind",
            "provider_id",
            "provider_dataset",
            "provider_endpoint",
            "query_start_at_ms",
            "query_end_at_ms",
            "provider_observed_at_ms",
            "provider_request_ids",
            "pagination_complete",
            "page_count",
            "records",
        },
        name="raw provider source",
    )
    if payload["schema"] != MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA:
        raise MassiveEconomicAuthorityV6Error("raw provider source schema differs")
    source_kind = _text("raw provider source kind", payload["source_kind"])
    if source_kind not in MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS:
        raise MassiveEconomicAuthorityV6Error("raw provider source kind is unsupported")
    provider_id = _text("provider ID", payload["provider_id"])
    provider_dataset = _text("provider dataset", payload["provider_dataset"])
    request_ids_raw = payload["provider_request_ids"]
    if not isinstance(request_ids_raw, list) or any(
        not isinstance(value, str) for value in request_ids_raw
    ):
        raise MassiveEconomicAuthorityV6Error("provider request inventory is malformed")
    raw_records = payload["records"]
    if not isinstance(raw_records, list) or any(
        not isinstance(record, dict) for record in raw_records
    ):
        raise MassiveEconomicAuthorityV6Error("raw provider records are malformed")
    if source_kind == "economic-order-observations":
        events: tuple[MassiveRawProviderEventObservationV6, ...] = ()
        orders = tuple(
            sorted(
                (
                    _parse_order_observation(
                        provider_id=provider_id,
                        provider_dataset=provider_dataset,
                        record=record,
                    )
                    for record in raw_records
                ),
                key=lambda row: (
                    row.provider_row_locator,
                    row.observation_receipt_sha256,
                ),
            )
        )
        records: Sequence[
            MassiveRawProviderEventObservationV6 | MassiveRawProviderOrderObservationV6
        ] = orders
    else:
        events = tuple(
            sorted(
                (
                    _parse_event_observation(
                        source_kind=source_kind,
                        provider_id=provider_id,
                        provider_dataset=provider_dataset,
                        record=record,
                    )
                    for record in raw_records
                ),
                key=lambda row: (
                    row.provider_row_locator,
                    row.observation_receipt_sha256,
                ),
            )
        )
        orders = ()
        records = events
    provisional = MassiveRawProviderEconomicSourceV6(
        source_kind=source_kind,
        provider_id=provider_id,
        provider_dataset=provider_dataset,
        provider_endpoint=_text("provider endpoint", payload["provider_endpoint"]),
        query_start_at_ms=_nonnegative_int("query start", payload["query_start_at_ms"]),
        query_end_at_ms=_nonnegative_int("query end", payload["query_end_at_ms"]),
        provider_observed_at_ms=_nonnegative_int(
            "provider observed time", payload["provider_observed_at_ms"]
        ),
        provider_request_ids=tuple(request_ids_raw),
        pagination_complete=payload["pagination_complete"] is True,
        page_count=_nonnegative_int("provider page count", payload["page_count"]),
        event_observations=events,
        order_observations=orders,
        row_inventory_sha256=semantic_sha256(
            tuple(row.observation_receipt_sha256 for row in records)
        ),
        loaded_source=loaded_source,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def capture_massive_raw_provider_economic_source_v6(
    *,
    provider_client: Any,
    root: str | Path,
    source_kind: str,
    provider_id: str,
    provider_dataset: str,
    provider_endpoint: str,
    query_start_at_ms: int,
    query_end_at_ms: int,
    entitlement_receipt_sha256: str,
    capture_id: str,
    now_ms: Callable[[], int] | None = None,
) -> MassiveRawProviderEconomicSourceV6:
    """Exhaust a provider paginator and commit its exact raw observations.

    The client boundary is intentionally narrow: it must return pages with
    ``ResponseMetadata.RequestId``, ``Records``, ``IsTruncated``, and (for
    nonterminal pages) ``NextToken``.  Authentication remains owned by the
    supplied provider client; no credential values enter the artifact.
    """

    if source_kind not in MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS:
        raise MassiveEconomicAuthorityV6Error("provider capture role is unsupported")
    provider_name = _text("provider ID", provider_id)
    dataset = _text("provider dataset", provider_dataset)
    endpoint = _text("provider endpoint", provider_endpoint)
    capture_name = _text("provider capture ID", capture_id)
    start = _nonnegative_int("provider query start", query_start_at_ms)
    end = _nonnegative_int("provider query end", query_end_at_ms)
    if end < start:
        raise MassiveEconomicAuthorityV6Error("provider query interval is inverted")
    entitlement = _digest("provider entitlement receipt", entitlement_receipt_sha256)
    clock = now_ms or (lambda: time.time_ns() // 1_000_000)
    requested_at_ms = clock()
    try:
        pages_raw = provider_client.paginate_economic_observations(
            source_kind=source_kind,
            query_start_at_ms=start,
            query_end_at_ms=end,
        )
        pages = tuple(pages_raw)
    except Exception as exc:  # pragma: no cover - provider SDK exceptions are optional
        raise MassiveEconomicAuthorityV6Error(
            "provider economic observation request failed"
        ) from exc
    observed_at_ms = clock()
    if not pages:
        raise MassiveEconomicAuthorityV6Error("provider returned no pages")
    request_ids: list[str] = []
    records: list[dict[str, object]] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            raise MassiveEconomicAuthorityV6Error("provider page is malformed")
        response = page.get("ResponseMetadata")
        request_id = (
            response.get("RequestId") if isinstance(response, Mapping) else None
        )
        request = _text("provider request ID", request_id)
        request_ids.append(request)
        truncated = page.get("IsTruncated") is True
        if truncated is not (page_index < len(pages) - 1):
            raise MassiveEconomicAuthorityV6Error(
                "provider pagination did not close exactly"
            )
        token = page.get("NextToken")
        if truncated != isinstance(token, str):
            raise MassiveEconomicAuthorityV6Error(
                "provider continuation token chain differs"
            )
        page_records = page.get("Records")
        if not isinstance(page_records, (list, tuple)):
            raise MassiveEconomicAuthorityV6Error("provider page records are malformed")
        for row in page_records:
            if not isinstance(row, Mapping) or "provider_request_id" in row:
                raise MassiveEconomicAuthorityV6Error(
                    "provider raw record is malformed or shadows request identity"
                )
            records.append({"provider_request_id": request, **dict(row)})
    if len(set(request_ids)) != len(request_ids):
        raise MassiveEconomicAuthorityV6Error("provider request IDs are duplicated")
    payload = {
        "schema": MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA,
        "source_kind": source_kind,
        "provider_id": provider_name,
        "provider_dataset": dataset,
        "provider_endpoint": endpoint,
        "query_start_at_ms": start,
        "query_end_at_ms": end,
        "provider_observed_at_ms": observed_at_ms,
        "provider_request_ids": request_ids,
        "pagination_complete": True,
        "page_count": len(pages),
        "records": records,
    }
    relative = (
        f"{MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_OBJECT_PREFIX}"
        f"{source_kind}-{capture_name}.json"
    )
    payload_bytes = canonical_json_file_bytes(payload)
    publish_massive_source_object(
        stream=BytesIO(payload_bytes),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_DATASETS[source_kind],
        source_object_key=relative,
        requested_at_ms=requested_at_ms,
        downloaded_at_ms=observed_at_ms,
        schema_sha256=MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement,
        committed_at_ms=observed_at_ms,
        etag=semantic_sha256(payload),
        request_id=request_ids[-1],
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=observed_at_ms,
    )
    return parse_massive_raw_provider_economic_source_v6(
        root=root, loaded_source=loaded
    )


def _source_role(row: MassiveSourcedEconomicEventV5) -> str:
    if isinstance(row, MassiveSourcedCorporateActionV5):
        return "corporate-actions"
    if isinstance(row, MassiveSourcedTerminalEventV5):
        return "terminal-outcomes"
    return "cash-returns"


def _validate_revision_chains_v6(
    observations: Sequence[MassiveRawProviderEventObservationV6],
) -> dict[str, tuple[MassiveRawProviderEventObservationV6, ...]]:
    grouped: dict[str, list[MassiveRawProviderEventObservationV6]] = {}
    provider_to_logical: dict[tuple[str, str, str, str], str] = {}
    revision_ids: set[tuple[str, str]] = set()
    fingerprints: dict[str, str] = {}
    for row in observations:
        row.validate()
        source = row.source_event
        logical = source.logical_event_key
        provider_identity = (
            row.source_kind,
            row.provider_id,
            row.provider_dataset,
            row.provider_event_key,
        )
        prior_logical = provider_to_logical.get(provider_identity)
        if prior_logical is not None and prior_logical != logical:
            raise MassiveEconomicAuthorityV6Error(
                "one provider event key maps to multiple logical events"
            )
        provider_to_logical[provider_identity] = logical
        revision_key = (logical, row.provider_revision_id)
        if revision_key in revision_ids:
            raise MassiveEconomicAuthorityV6Error("provider revision is duplicated")
        revision_ids.add(revision_key)
        if source.revision_status != "cancelled":
            prior = fingerprints.get(source.economic_fingerprint_sha256)
            if prior is not None and prior != logical:
                raise MassiveEconomicAuthorityV6Error(
                    "semantic economic event is duplicated across logical keys"
                )
            fingerprints[source.economic_fingerprint_sha256] = logical
        grouped.setdefault(logical, []).append(row)
    output: dict[str, tuple[MassiveRawProviderEventObservationV6, ...]] = {}
    for logical, candidates in grouped.items():
        security_ids = {
            row.source_event.event.security_id
            for row in candidates
            if isinstance(
                row.source_event,
                (MassiveSourcedCorporateActionV5, MassiveSourcedTerminalEventV5),
            )
        }
        event_kinds = {
            row.source_event.event.kind.value
            for row in candidates
            if isinstance(
                row.source_event,
                (MassiveSourcedCorporateActionV5, MassiveSourcedTerminalEventV5),
            )
        }
        if len(security_ids) > 1 or len(event_kinds) > 1:
            raise MassiveEconomicAuthorityV6Error(
                "provider revisions change security or economic event kind"
            )
        by_provider_revision = {row.provider_revision_id: row for row in candidates}
        roots = tuple(
            row for row in candidates if row.supersedes_provider_revision_id is None
        )
        if len(roots) != 1 or roots[0].revision_status != "active":
            raise MassiveEconomicAuthorityV6Error(
                "provider revision chain lacks one active root"
            )
        child_by_parent: dict[str, MassiveRawProviderEventObservationV6] = {}
        for row in candidates:
            parent = row.supersedes_provider_revision_id
            if parent is None:
                continue
            if parent not in by_provider_revision or parent in child_by_parent:
                raise MassiveEconomicAuthorityV6Error(
                    "provider revision chain branches or has an unknown parent"
                )
            child_by_parent[parent] = row
        chain: list[MassiveRawProviderEventObservationV6] = []
        current = roots[0]
        seen_fingerprints: set[str] = set()
        while True:
            chain.append(current)
            source = current.source_event
            if (
                source.revision_status != "cancelled"
                and source.economic_fingerprint_sha256 in seen_fingerprints
            ):
                raise MassiveEconomicAuthorityV6Error(
                    "provider revision repeats identical economic terms"
                )
            if source.revision_status != "cancelled":
                seen_fingerprints.add(source.economic_fingerprint_sha256)
            child = child_by_parent.get(current.provider_revision_id)
            if child is None:
                break
            if (
                current.revision_status == "cancelled"
                or child.revision_status not in {"corrected", "cancelled"}
                or child.provider_available_at_ms <= current.provider_available_at_ms
            ):
                raise MassiveEconomicAuthorityV6Error(
                    "provider revision chronology is invalid"
                )
            current = child
        if len(chain) != len(candidates):
            raise MassiveEconomicAuthorityV6Error(
                "provider revision chain is disconnected"
            )
        output[logical] = tuple(chain)
    return output


def _security_interaction_domains(
    identity_authority: PITSecurityUniverseAuthority,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in identity_authority.security_master:
        chain = row.corporate_action_chain_id
        if chain is None:
            chain = f"singleton:{row.security_id}"
        result[row.security_id] = semantic_sha256(
            ("economic-interaction-domain-v6", chain)
        )
    return result


def _interaction_domain(
    row: MassiveSourcedEconomicEventV5,
    *,
    domains: Mapping[str, str],
) -> str:
    if isinstance(row, MassiveSourcedCashReturnV5):
        return MASSIVE_ECONOMIC_CASH_INTERACTION_DOMAIN_V6
    event = row.event
    try:
        domain = domains[event.security_id]
    except KeyError as exc:
        raise MassiveEconomicAuthorityV6Error(
            "economic event references an unknown security"
        ) from exc
    if event.successor_security_id is not None:
        successor_domain = domains.get(event.successor_security_id)
        if successor_domain is None or successor_domain != domain:
            raise MassiveEconomicAuthorityV6Error(
                "predecessor and successor interaction domains differ"
            )
    return domain


@dataclass(frozen=True, slots=True)
class MassiveProviderEconomicArchiveAuthorityV6:
    identity_authority_receipt_sha256: str
    sources: tuple[MassiveRawProviderEconomicSourceV6, ...]
    event_observations: tuple[MassiveRawProviderEventObservationV6, ...]
    order_observations: tuple[MassiveRawProviderOrderObservationV6, ...]
    interaction_domains: tuple[tuple[str, str], ...]
    revision_chain_inventory_sha256: str
    raw_archive_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        for value in (
            self.identity_authority_receipt_sha256,
            self.revision_chain_inventory_sha256,
            self.raw_archive_inventory_sha256,
            self.receipt_sha256,
        ):
            _digest("provider archive digest", value)
        if tuple(source.source_kind for source in self.sources) != (
            MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS
        ):
            raise MassiveEconomicAuthorityV6Error(
                "provider archive requires exactly four raw source roles"
            )
        for source in self.sources:
            source.validate()
        expected_events = tuple(
            row for source in self.sources for row in source.event_observations
        )
        expected_orders = tuple(
            row for source in self.sources for row in source.order_observations
        )
        if (
            self.event_observations != expected_events
            or self.order_observations != expected_orders
        ):
            raise MassiveEconomicAuthorityV6Error("provider archive rows differ")
        chains = _validate_revision_chains_v6(self.event_observations)
        expected_chain_inventory = semantic_sha256(
            tuple(
                (
                    logical,
                    tuple(row.observation_receipt_sha256 for row in chain),
                )
                for logical, chain in sorted(chains.items())
            )
        )
        if self.revision_chain_inventory_sha256 != expected_chain_inventory:
            raise MassiveEconomicAuthorityV6Error("provider revision inventory differs")
        if self.raw_archive_inventory_sha256 != semantic_sha256(
            tuple(source.receipt_sha256 for source in self.sources)
        ):
            raise MassiveEconomicAuthorityV6Error("raw archive inventory differs")
        if self.interaction_domains != tuple(sorted(set(self.interaction_domains))):
            raise MassiveEconomicAuthorityV6Error(
                "interaction domains are not canonical"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV6Error("provider archive receipt differs")


def build_massive_provider_economic_archive_authority_v6(
    *,
    root: str | Path,
    loaded_sources: Sequence[LoadedMassiveSourceObject],
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveProviderEconomicArchiveAuthorityV6:
    """Reparse the complete raw provider archive and derive stable event identity."""

    identity_authority.validate()
    parsed = tuple(
        sorted(
            (
                parse_massive_raw_provider_economic_source_v6(
                    root=root, loaded_source=source
                )
                for source in loaded_sources
            ),
            key=lambda row: MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS.index(
                row.source_kind
            ),
        )
    )
    if tuple(row.source_kind for row in parsed) != (
        MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS
    ):
        raise MassiveEconomicAuthorityV6Error(
            "provider archive requires one exact source per role"
        )
    events = tuple(row for source in parsed for row in source.event_observations)
    orders = tuple(row for source in parsed for row in source.order_observations)
    chains = _validate_revision_chains_v6(events)
    domains = _security_interaction_domains(identity_authority)
    for row in events:
        _interaction_domain(row.source_event, domains=domains)
    provisional = MassiveProviderEconomicArchiveAuthorityV6(
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        sources=parsed,
        event_observations=events,
        order_observations=orders,
        interaction_domains=tuple(sorted(domains.items())),
        revision_chain_inventory_sha256=semantic_sha256(
            tuple(
                (
                    logical,
                    tuple(row.observation_receipt_sha256 for row in chain),
                )
                for logical, chain in sorted(chains.items())
            )
        ),
        raw_archive_inventory_sha256=semantic_sha256(
            tuple(source.receipt_sha256 for source in parsed)
        ),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveOriginEconomicOrderEvidenceV6:
    interaction_domain_sha256: str
    logical_event_key: str
    effective_at_ms: int
    local_economic_sequence: int
    derivation_kind: str
    origin_selected_event_inventory_sha256: str
    provider_order_observation_receipt_sha256: str | None
    provider_order_available_at_ms: int | None
    canonical_rule_receipt_sha256: str | None
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _digest("interaction domain", self.interaction_domain_sha256)
        _digest("logical event key", self.logical_event_key)
        _nonnegative_int("origin order effective time", self.effective_at_ms)
        sequence = _nonnegative_int(
            "origin local economic sequence", self.local_economic_sequence
        )
        if sequence >= MASSIVE_ECONOMIC_BASE_GLOBAL_SEQUENCE_V5:
            raise MassiveEconomicAuthorityV6Error("origin order reaches base sentinel")
        _digest(
            "origin selected event inventory",
            self.origin_selected_event_inventory_sha256,
        )
        if self.derivation_kind == "provider-order-observation":
            if (
                self.provider_order_observation_receipt_sha256 is None
                or self.provider_order_available_at_ms is None
                or self.canonical_rule_receipt_sha256 is not None
            ):
                raise MassiveEconomicAuthorityV6Error(
                    "provider origin order is incomplete"
                )
            _digest(
                "provider order observation",
                self.provider_order_observation_receipt_sha256,
            )
            _nonnegative_int(
                "provider order availability", self.provider_order_available_at_ms
            )
        elif self.derivation_kind == "single-event-canonical-rule":
            if (
                self.provider_order_observation_receipt_sha256 is not None
                or self.provider_order_available_at_ms is not None
                or self.canonical_rule_receipt_sha256
                != MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V6_RECEIPT_SHA256
                or self.local_economic_sequence != 0
            ):
                raise MassiveEconomicAuthorityV6Error("canonical origin order differs")
        else:
            raise MassiveEconomicAuthorityV6Error(
                "origin order derivation is unsupported"
            )
        _digest("origin order evidence receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV6Error(
                "origin order evidence receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveOrderedEconomicEventV6:
    source_event: MassiveSourcedEconomicEventV5
    selected_observation_receipt_sha256: str
    interaction_domain_sha256: str
    order_evidence: MassiveOriginEconomicOrderEvidenceV6
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.source_event.validate()
        _digest(
            "selected provider observation", self.selected_observation_receipt_sha256
        )
        _digest("ordered interaction domain", self.interaction_domain_sha256)
        self.order_evidence.validate()
        if (
            self.source_event.logical_event_key != self.order_evidence.logical_event_key
            or sourced_effective_at_ms_v5(self.source_event)
            != self.order_evidence.effective_at_ms
            or self.interaction_domain_sha256
            != self.order_evidence.interaction_domain_sha256
        ):
            raise MassiveEconomicAuthorityV6Error("ordered origin event differs")
        _digest("ordered origin event receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicAuthorityV6Error(
                "ordered origin event receipt differs"
            )


def ordered_economic_event_order_v6(
    row: MassiveOrderedEconomicEventV6,
) -> tuple[int, int, str]:
    return (
        row.order_evidence.effective_at_ms,
        row.order_evidence.local_economic_sequence,
        row.interaction_domain_sha256,
    )


@dataclass(frozen=True, slots=True)
class MassiveResolvedEconomicAuthorityAtOriginV6:
    decision_at_ms: int
    identity_scope_receipt_sha256: str
    selected_events: tuple[MassiveOrderedEconomicEventV6, ...]
    cancelled_logical_event_keys: tuple[str, ...]
    selected_revision_inventory_sha256: str
    selected_order_inventory_sha256: str
    semantic_event_inventory_sha256: str
    receipt_sha256: str
    archive_audit_receipt_sha256: str
    raw_source_audit_inventory_sha256: str

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "decision_at_ms": self.decision_at_ms,
            "identity_scope_receipt_sha256": self.identity_scope_receipt_sha256,
            "selected_events": tuple(asdict(row) for row in self.selected_events),
            "cancelled_logical_event_keys": self.cancelled_logical_event_keys,
            "selected_revision_inventory_sha256": self.selected_revision_inventory_sha256,
            "selected_order_inventory_sha256": self.selected_order_inventory_sha256,
            "semantic_event_inventory_sha256": self.semantic_event_inventory_sha256,
        }

    def validate(self) -> None:
        _nonnegative_int("resolved authority decision time", self.decision_at_ms)
        for value in (
            self.identity_scope_receipt_sha256,
            self.selected_revision_inventory_sha256,
            self.selected_order_inventory_sha256,
            self.semantic_event_inventory_sha256,
            self.receipt_sha256,
            self.archive_audit_receipt_sha256,
            self.raw_source_audit_inventory_sha256,
        ):
            _digest("resolved authority digest", value)
        keys = tuple(row.source_event.logical_event_key for row in self.selected_events)
        if len(keys) != len(set(keys)):
            raise MassiveEconomicAuthorityV6Error(
                "resolved logical events are duplicated"
            )
        for row in self.selected_events:
            row.validate()
            if sourced_available_at_ms_v5(row.source_event) > self.decision_at_ms:
                raise MassiveEconomicAuthorityV6Error(
                    "future event entered origin authority"
                )
            if (
                row.order_evidence.provider_order_available_at_ms is not None
                and row.order_evidence.provider_order_available_at_ms
                > self.decision_at_ms
            ):
                raise MassiveEconomicAuthorityV6Error(
                    "future order entered origin authority"
                )
        canonical = tuple(
            sorted(
                self.selected_events,
                key=lambda row: (
                    *ordered_economic_event_order_v6(row),
                    row.source_event.logical_event_key,
                ),
            )
        )
        if self.selected_events != canonical:
            raise MassiveEconomicAuthorityV6Error("resolved events are not canonical")
        if self.cancelled_logical_event_keys != tuple(
            sorted(set(self.cancelled_logical_event_keys))
        ):
            raise MassiveEconomicAuthorityV6Error(
                "cancelled logical keys are not canonical"
            )
        revision_inventory = semantic_sha256(
            tuple(
                (
                    row.source_event.logical_event_key,
                    row.source_event.revision_id,
                    row.selected_observation_receipt_sha256,
                    row.source_event.receipt_sha256,
                )
                for row in self.selected_events
            )
        )
        order_inventory = semantic_sha256(
            tuple(
                (
                    row.interaction_domain_sha256,
                    row.source_event.logical_event_key,
                    row.order_evidence.local_economic_sequence,
                    row.order_evidence.receipt_sha256,
                )
                for row in self.selected_events
            )
        )
        semantic_inventory = semantic_sha256(
            tuple(
                (
                    row.source_event.logical_event_key,
                    row.source_event.revision_id,
                    row.source_event.economic_fingerprint_sha256,
                    row.order_evidence.receipt_sha256,
                )
                for row in self.selected_events
            )
        )
        if (
            self.selected_revision_inventory_sha256 != revision_inventory
            or self.selected_order_inventory_sha256 != order_inventory
            or self.semantic_event_inventory_sha256 != semantic_inventory
        ):
            raise MassiveEconomicAuthorityV6Error("resolved origin inventories differ")
        if self.receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveEconomicAuthorityV6Error("resolved semantic receipt differs")


def resolve_massive_economic_authority_at_origin_v6(
    *,
    archive: MassiveProviderEconomicArchiveAuthorityV6,
    identity_authority: PITSecurityUniverseAuthority,
    decision_at_ms: int,
) -> MassiveResolvedEconomicAuthorityAtOriginV6:
    """Resolve revisions and interaction-local order using origin-available bytes."""

    archive.validate()
    identity_authority.validate()
    decision = _nonnegative_int("economic origin decision time", decision_at_ms)
    if archive.identity_authority_receipt_sha256 != identity_authority.receipt_sha256:
        raise MassiveEconomicAuthorityV6Error("archive and identity authorities differ")
    if archive.interaction_domains != tuple(
        sorted(_security_interaction_domains(identity_authority).items())
    ):
        raise MassiveEconomicAuthorityV6Error(
            "archive interaction domains were not identity-derived"
        )
    chains = _validate_revision_chains_v6(archive.event_observations)
    selected_observations: list[MassiveRawProviderEventObservationV6] = []
    cancelled: list[str] = []
    for logical, chain in sorted(chains.items()):
        available = tuple(
            row for row in chain if row.provider_available_at_ms <= decision
        )
        if not available:
            continue
        latest = available[-1]
        if latest.revision_status == "cancelled":
            cancelled.append(logical)
        else:
            selected_observations.append(latest)
    domains = dict(archive.interaction_domains)
    selected_domains = {
        row.source_event.logical_event_key: _interaction_domain(
            row.source_event, domains=domains
        )
        for row in selected_observations
    }
    groups: dict[tuple[str, int], list[MassiveRawProviderEventObservationV6]] = {}
    for row in selected_observations:
        source = row.source_event
        groups.setdefault(
            (
                selected_domains[source.logical_event_key],
                sourced_effective_at_ms_v5(source),
            ),
            [],
        ).append(row)
    order_by_event: dict[str, MassiveOriginEconomicOrderEvidenceV6] = {}
    for (interaction_domain, effective_at_ms), raw_group in sorted(groups.items()):
        group = tuple(
            sorted(raw_group, key=lambda row: row.source_event.logical_event_key)
        )
        inventory = semantic_sha256(
            tuple(
                (
                    row.source_event.logical_event_key,
                    row.source_event.revision_id,
                    row.observation_receipt_sha256,
                )
                for row in group
            )
        )
        if len(group) == 1:
            row = group[0]
            canonical_order_provisional = MassiveOriginEconomicOrderEvidenceV6(
                interaction_domain_sha256=interaction_domain,
                logical_event_key=row.source_event.logical_event_key,
                effective_at_ms=effective_at_ms,
                local_economic_sequence=0,
                derivation_kind="single-event-canonical-rule",
                origin_selected_event_inventory_sha256=inventory,
                provider_order_observation_receipt_sha256=None,
                provider_order_available_at_ms=None,
                canonical_rule_receipt_sha256=(
                    MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V6_RECEIPT_SHA256
                ),
                receipt_sha256="0" * 64,
            )
            order_by_event[row.source_event.logical_event_key] = replace(
                canonical_order_provisional,
                receipt_sha256=semantic_sha256(canonical_order_provisional.unsigned()),
            )
            continue
        observations_by_key: dict[str, list[MassiveRawProviderOrderObservationV6]] = {}
        for observation in archive.order_observations:
            if (
                observation.logical_event_key
                in {row.source_event.logical_event_key for row in group}
                and observation.effective_at_ms == effective_at_ms
                and observation.provider_order_available_at_ms <= decision
            ):
                observations_by_key.setdefault(
                    observation.logical_event_key, []
                ).append(observation)
        selected_orders: list[MassiveRawProviderOrderObservationV6] = []
        for row in group:
            candidates = tuple(
                sorted(
                    observations_by_key.get(row.source_event.logical_event_key, ()),
                    key=lambda candidate: (
                        candidate.provider_order_available_at_ms,
                        candidate.observation_receipt_sha256,
                    ),
                )
            )
            if not candidates:
                raise MassiveEconomicAuthorityV6Error(
                    "origin-available provider order does not resolve interaction tie"
                )
            latest_at = candidates[-1].provider_order_available_at_ms
            latest_order_candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.provider_order_available_at_ms == latest_at
            )
            if len(latest_order_candidates) != 1:
                raise MassiveEconomicAuthorityV6Error(
                    "provider order has an unresolved same-vintage revision"
                )
            selected_orders.append(latest_order_candidates[0])
        sequences = tuple(
            row.provider_local_economic_sequence for row in selected_orders
        )
        if len(sequences) != len(set(sequences)):
            raise MassiveEconomicAuthorityV6Error(
                "provider order does not uniquely resolve interaction tie"
            )
        for observation in selected_orders:
            provider_order_provisional = MassiveOriginEconomicOrderEvidenceV6(
                interaction_domain_sha256=interaction_domain,
                logical_event_key=observation.logical_event_key,
                effective_at_ms=effective_at_ms,
                local_economic_sequence=observation.provider_local_economic_sequence,
                derivation_kind="provider-order-observation",
                origin_selected_event_inventory_sha256=inventory,
                provider_order_observation_receipt_sha256=(
                    observation.observation_receipt_sha256
                ),
                provider_order_available_at_ms=(
                    observation.provider_order_available_at_ms
                ),
                canonical_rule_receipt_sha256=None,
                receipt_sha256="0" * 64,
            )
            order_by_event[observation.logical_event_key] = replace(
                provider_order_provisional,
                receipt_sha256=semantic_sha256(provider_order_provisional.unsigned()),
            )
    ordered: list[MassiveOrderedEconomicEventV6] = []
    for selected_observation in selected_observations:
        source = selected_observation.source_event
        ordered_provisional = MassiveOrderedEconomicEventV6(
            source_event=source,
            selected_observation_receipt_sha256=(
                selected_observation.observation_receipt_sha256
            ),
            interaction_domain_sha256=selected_domains[source.logical_event_key],
            order_evidence=order_by_event[source.logical_event_key],
            receipt_sha256="0" * 64,
        )
        ordered.append(
            replace(
                ordered_provisional,
                receipt_sha256=semantic_sha256(ordered_provisional.unsigned()),
            )
        )
    selected = tuple(
        sorted(
            ordered,
            key=lambda row: (
                *ordered_economic_event_order_v6(row),
                row.source_event.logical_event_key,
            ),
        )
    )
    security_scope: set[str] = set()
    masters = {row.security_id: row for row in identity_authority.security_master}
    for selected_row in selected:
        selected_source = selected_row.source_event
        if isinstance(
            selected_source,
            (MassiveSourcedCorporateActionV5, MassiveSourcedTerminalEventV5),
        ):
            security_scope.add(selected_source.event.security_id)
            if selected_source.event.successor_security_id is not None:
                security_scope.add(selected_source.event.successor_security_id)
    identity_scope = semantic_sha256(
        tuple(
            (
                security_id,
                masters[security_id].corporate_action_chain_id,
                masters[security_id].identity_source_receipt_sha256,
            )
            for security_id in sorted(security_scope)
        )
    )
    revision_inventory = semantic_sha256(
        tuple(
            (
                row.source_event.logical_event_key,
                row.source_event.revision_id,
                row.selected_observation_receipt_sha256,
                row.source_event.receipt_sha256,
            )
            for row in selected
        )
    )
    order_inventory = semantic_sha256(
        tuple(
            (
                row.interaction_domain_sha256,
                row.source_event.logical_event_key,
                row.order_evidence.local_economic_sequence,
                row.order_evidence.receipt_sha256,
            )
            for row in selected
        )
    )
    semantic_inventory = semantic_sha256(
        tuple(
            (
                row.source_event.logical_event_key,
                row.source_event.revision_id,
                row.source_event.economic_fingerprint_sha256,
                row.order_evidence.receipt_sha256,
            )
            for row in selected
        )
    )
    provisional_result = MassiveResolvedEconomicAuthorityAtOriginV6(
        decision_at_ms=decision,
        identity_scope_receipt_sha256=identity_scope,
        selected_events=selected,
        cancelled_logical_event_keys=tuple(sorted(cancelled)),
        selected_revision_inventory_sha256=revision_inventory,
        selected_order_inventory_sha256=order_inventory,
        semantic_event_inventory_sha256=semantic_inventory,
        receipt_sha256="0" * 64,
        archive_audit_receipt_sha256=archive.receipt_sha256,
        raw_source_audit_inventory_sha256=archive.raw_archive_inventory_sha256,
    )
    result = replace(
        provisional_result,
        receipt_sha256=semantic_sha256(provisional_result.semantic_unsigned()),
    )
    result.validate()
    return result


def build_massive_resolved_economic_authority_from_provider_files_v6(
    *,
    root: str | Path,
    loaded_sources: Sequence[LoadedMassiveSourceObject],
    identity_authority: PITSecurityUniverseAuthority,
    decision_at_ms: int,
) -> MassiveResolvedEconomicAuthorityAtOriginV6:
    """Reparse raw provider bytes and resolve one semantic origin authority."""

    archive = build_massive_provider_economic_archive_authority_v6(
        root=root,
        loaded_sources=loaded_sources,
        identity_authority=identity_authority,
    )
    return resolve_massive_economic_authority_at_origin_v6(
        archive=archive,
        identity_authority=identity_authority,
        decision_at_ms=decision_at_ms,
    )


MASSIVE_ECONOMIC_AUTHORITY_V6_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_AUTHORITY_V6_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "raw_provider_sources": MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS,
        "raw_provider_schema": MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256,
        "provider_capture": "request-id-etag-pagination-and-exact-raw-row-bound",
        "logical_event_key": "derived-from-provider-id-role-and-provider-event-key",
        "provider_key_bijection": "one-role-provider-key-one-logical-key",
        "revision": "linear-provider-revision-chain-origin-latest-available",
        "effective_time": "derived-from-kind-specific-raw-provider-field",
        "interaction_domain": "identity-corporate-action-chain-or-cash-ledger",
        "order": "interaction-local-and-origin-available",
        "semantic_receipt": "selected-revisions-and-orders-only-future-row-invariant",
        "full_archive": "audit-only-excluded-from-semantic-origin-receipt",
        "historical_panel_authorized": (
            MASSIVE_ECONOMIC_AUTHORITY_V6_HISTORICAL_PANEL_AUTHORIZED
        ),
        "predictive_training_authorized": (
            MASSIVE_ECONOMIC_AUTHORITY_V6_PREDICTIVE_TRAINING_AUTHORIZED
        ),
        "profitability_reporting_authorized": (
            MASSIVE_ECONOMIC_AUTHORITY_V6_PROFITABILITY_REPORTING_AUTHORIZED
        ),
    }
)


__all__ = [
    "MASSIVE_ECONOMIC_AUTHORITY_V6_SPEC_SHA256",
    "MASSIVE_ECONOMIC_AUTHORITY_V6_SOURCE_SHA256",
    "MASSIVE_ECONOMIC_AUTHORITY_V6_HISTORICAL_PANEL_AUTHORIZED",
    "MASSIVE_ECONOMIC_AUTHORITY_V6_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_ECONOMIC_AUTHORITY_V6_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_ECONOMIC_CASH_INTERACTION_DOMAIN_V6",
    "MASSIVE_ECONOMIC_SINGLE_EVENT_ORDER_RULE_V6_RECEIPT_SHA256",
    "MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_DATASETS",
    "MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS",
    "MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_OBJECT_PREFIX",
    "MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA",
    "MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256",
    "MassiveEconomicAuthorityV6Error",
    "MassiveOrderedEconomicEventV6",
    "MassiveOriginEconomicOrderEvidenceV6",
    "MassiveProviderEconomicArchiveAuthorityV6",
    "MassiveRawProviderEconomicSourceV6",
    "MassiveRawProviderEventObservationV6",
    "MassiveRawProviderOrderObservationV6",
    "MassiveResolvedEconomicAuthorityAtOriginV6",
    "build_massive_provider_economic_archive_authority_v6",
    "build_massive_resolved_economic_authority_from_provider_files_v6",
    "capture_massive_raw_provider_economic_source_v6",
    "ordered_economic_event_order_v6",
    "parse_massive_raw_provider_economic_source_v6",
    "resolve_massive_economic_authority_at_origin_v6",
]
