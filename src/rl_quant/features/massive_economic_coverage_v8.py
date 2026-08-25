"""Complete current-surface economic coverage authority for profitability P0 V8."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.alpha.contracts import CorporateActionKind, TerminalEventKind
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.economic_provider_capture_v8 import (
    MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SPEC_SHA256,
    MASSIVE_ECONOMIC_REST_DATE_FIELDS_V8,
    MASSIVE_ECONOMIC_REST_SURFACES_V8,
    MassiveEconomicRawRestCaptureV8,
    parse_massive_economic_raw_rest_capture_v8,
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

MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA = "rl-quant.massive-economic-origin-coverage-v8"
MASSIVE_ECONOMIC_COVERAGE_V8_DATASET = "massive-economic-origin-coverage-v8"
MASSIVE_ECONOMIC_COVERAGE_V8_OBJECT_PREFIX = (
    "massive-profitability-p0/economic-origin-coverage-v8/"
)
MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA,
        "captures": tuple(sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8)),
        "terminal": "exact-delisting-disposition-coverage",
        "cash": "explicit-zero-return-no-borrowing",
        "origin_cutoff": "complete-capture-or-no-capture",
        "generic_reload": "transport-and-economic-authority-false",
    }
)
MASSIVE_TERMINAL_COVERAGE_V8_SCHEMA = "rl-quant.massive-terminal-coverage-source-v8"
MASSIVE_TERMINAL_COVERAGE_V8_DATASET = "massive-terminal-coverage-source-v8"
MASSIVE_TERMINAL_COVERAGE_V8_OBJECT_PREFIX = (
    "massive-profitability-p0/terminal-coverage-v8/"
)
MASSIVE_TERMINAL_COVERAGE_V8_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_TERMINAL_COVERAGE_V8_SCHEMA,
        "rows": "one-provider-bound-disposition-per-covered-delisting",
        "generic_parse": "always-fixed-runtime-captured-false",
    }
)
MASSIVE_ECONOMIC_ACCOUNTING_LANES_V8 = (
    "strict-pit-capture",
    "finalized-accounting-research",
)
MASSIVE_DIVIDEND_DISTRIBUTION_TYPES_V8 = (
    "irregular",
    "recurring",
    "special",
    "supplemental",
    "unknown",
)
MASSIVE_SPLIT_ADJUSTMENT_TYPES_V8 = (
    "forward_split",
    "reverse_split",
    "stock_dividend",
)
MASSIVE_ZERO_CASH_POLICY_V8 = "explicit-zero-return-no-borrowing"
MASSIVE_ZERO_CASH_POLICY_V8_RECEIPT_SHA256 = semantic_sha256(
    {
        "policy": MASSIVE_ZERO_CASH_POLICY_V8,
        "currency": "USD",
        "cash_return": 0.0,
        "periods": (),
        "negative_cash": False,
    }
)
MASSIVE_ECONOMIC_COVERAGE_V8_HISTORICAL_PANEL_AUTHORIZED = False
MASSIVE_ECONOMIC_COVERAGE_V8_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_ECONOMIC_COVERAGE_V8_PROFITABILITY_REPORTING_AUTHORIZED = False

_NEW_YORK = ZoneInfo("America/New_York")


class MassiveEconomicCoverageV8Error(ValueError):
    """Current-surface economic data is incomplete or noncausal."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEconomicCoverageV8Error(f"{name} must be canonical text")
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
        raise MassiveEconomicCoverageV8Error(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicCoverageV8Error(f"{name} must be nonnegative")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveEconomicCoverageV8Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MassiveEconomicCoverageV8Error(f"{name} must be finite")
    return result


def _canonical_date(name: str, value: object) -> str:
    raw = _text(name, value)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise MassiveEconomicCoverageV8Error(
            f"{name} must be an ISO calendar date"
        ) from exc
    if parsed.isoformat() != raw:
        raise MassiveEconomicCoverageV8Error(f"{name} is not canonical")
    return raw


def _date_at_market_open_ms(name: str, value: object) -> int:
    parsed = date.fromisoformat(_canonical_date(name, value))
    return int(
        datetime.combine(parsed, time(hour=9, minute=30), tzinfo=_NEW_YORK).timestamp()
        * 1000
    )


def _date_for_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=_NEW_YORK).date().isoformat()


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveEconomicCoverageV8Error(f"{name} fields differ")


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
        raise MassiveEconomicCoverageV8Error(
            "current Massive ticker does not resolve to one causal security"
        )
    return candidates[0].security_id, candidates[0].source_receipt_sha256


@dataclass(frozen=True, slots=True)
class MassiveEconomicCoverageScopeV8:
    coverage_start_date: str
    coverage_end_date: str
    required_surface_ids: tuple[str, ...]
    terminal_coverage_required: bool
    cash_policy_id: str
    full_market_query_required: bool
    initial_cursor_prohibited: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        start = _canonical_date("coverage scope start", self.coverage_start_date)
        end = _canonical_date("coverage scope end", self.coverage_end_date)
        if end < start:
            raise MassiveEconomicCoverageV8Error("coverage scope interval is inverted")
        if self.required_surface_ids != tuple(
            sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8)
        ):
            raise MassiveEconomicCoverageV8Error(
                "coverage scope does not require both current Massive surfaces"
            )
        if (
            self.terminal_coverage_required is not True
            or self.cash_policy_id != MASSIVE_ZERO_CASH_POLICY_V8
            or self.full_market_query_required is not True
            or self.initial_cursor_prohibited is not True
        ):
            raise MassiveEconomicCoverageV8Error("coverage scope policy differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicCoverageV8Error("coverage scope receipt differs")

    @classmethod
    def build(
        cls, *, coverage_start_date: str, coverage_end_date: str
    ) -> MassiveEconomicCoverageScopeV8:
        provisional = cls(
            coverage_start_date=coverage_start_date,
            coverage_end_date=coverage_end_date,
            required_surface_ids=tuple(sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8)),
            terminal_coverage_required=True,
            cash_policy_id=MASSIVE_ZERO_CASH_POLICY_V8,
            full_market_query_required=True,
            initial_cursor_prohibited=True,
            receipt_sha256="0" * 64,
        )
        result = replace(
            provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class MassiveZeroCashPolicyV8:
    policy_id: str
    currency: str
    one_period_return: float
    accrual_period_receipts: tuple[str, ...]
    borrowing_allowed: bool
    negative_cash_allowed: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.policy_id != MASSIVE_ZERO_CASH_POLICY_V8
            or self.currency != "USD"
            or self.one_period_return != 0.0
            or self.accrual_period_receipts
            or self.borrowing_allowed
            or self.negative_cash_allowed
        ):
            raise MassiveEconomicCoverageV8Error("zero-cash policy differs")
        if self.receipt_sha256 != MASSIVE_ZERO_CASH_POLICY_V8_RECEIPT_SHA256:
            raise MassiveEconomicCoverageV8Error("zero-cash policy receipt differs")

    @classmethod
    def build(cls) -> MassiveZeroCashPolicyV8:
        result = cls(
            policy_id=MASSIVE_ZERO_CASH_POLICY_V8,
            currency="USD",
            one_period_return=0.0,
            accrual_period_receipts=(),
            borrowing_allowed=False,
            negative_cash_allowed=False,
            receipt_sha256=MASSIVE_ZERO_CASH_POLICY_V8_RECEIPT_SHA256,
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class MassiveTerminalDispositionV8:
    security_id: str
    delisting_event_id: str
    terminal_kind: str
    effective_at_ms: int
    provider_available_at_ms: int
    cash_per_share: float
    successor_security_id: str | None
    successor_ratio: float
    delisting_source_receipt_sha256: str
    raw_provider_source_receipt_sha256: str
    raw_provider_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        _text("terminal security ID", self.security_id)
        _text("terminal delisting event ID", self.delisting_event_id)
        if self.terminal_kind not in {row.value for row in TerminalEventKind}:
            raise MassiveEconomicCoverageV8Error("terminal kind is unsupported")
        effective = _nonnegative_int("terminal effective time", self.effective_at_ms)
        available = _nonnegative_int(
            "terminal provider availability", self.provider_available_at_ms
        )
        if available < effective:
            raise MassiveEconomicCoverageV8Error(
                "terminal outcome availability predates economic disposition"
            )
        cash = _finite("terminal cash per share", self.cash_per_share)
        ratio = _finite("terminal successor ratio", self.successor_ratio)
        if cash < 0.0 or ratio < 0.0:
            raise MassiveEconomicCoverageV8Error("terminal terms are negative")
        successor = _optional_text(
            "terminal successor security", self.successor_security_id
        )
        if self.terminal_kind in {
            TerminalEventKind.MERGER_STOCK.value,
        }:
            if successor is None or ratio <= 0.0:
                raise MassiveEconomicCoverageV8Error(
                    "stock terminal outcome lacks successor terms"
                )
        elif successor is not None or ratio != 0.0:
            raise MassiveEconomicCoverageV8Error(
                "nonstock terminal outcome contains successor terms"
            )
        for value in (
            self.delisting_source_receipt_sha256,
            self.raw_provider_source_receipt_sha256,
            self.raw_provider_row_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("terminal disposition digest", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicCoverageV8Error("terminal disposition receipt differs")

    @classmethod
    def build(
        cls,
        *,
        security_id: str,
        delisting_event_id: str,
        terminal_kind: str,
        effective_at_ms: int,
        provider_available_at_ms: int,
        cash_per_share: float,
        successor_security_id: str | None,
        successor_ratio: float,
        delisting_source_receipt_sha256: str,
        raw_provider_source_receipt_sha256: str,
        raw_provider_row_receipt_sha256: str,
    ) -> MassiveTerminalDispositionV8:
        provisional = cls(
            security_id=security_id,
            delisting_event_id=delisting_event_id,
            terminal_kind=terminal_kind,
            effective_at_ms=effective_at_ms,
            provider_available_at_ms=provider_available_at_ms,
            cash_per_share=float(cash_per_share),
            successor_security_id=successor_security_id,
            successor_ratio=float(successor_ratio),
            delisting_source_receipt_sha256=delisting_source_receipt_sha256,
            raw_provider_source_receipt_sha256=raw_provider_source_receipt_sha256,
            raw_provider_row_receipt_sha256=raw_provider_row_receipt_sha256,
            receipt_sha256="0" * 64,
        )
        result = replace(
            provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class MassiveTerminalCoverageSourceV8:
    coverage_start_date: str
    coverage_end_date: str
    dispositions: tuple[MassiveTerminalDispositionV8, ...]
    provider_id: str
    provider_dataset: str
    capture_kind: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    fixed_runtime_captured: bool = False
    schema: str = MASSIVE_TERMINAL_COVERAGE_V8_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"receipt_sha256", "fixed_runtime_captured"}
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TERMINAL_COVERAGE_V8_SCHEMA:
            raise MassiveEconomicCoverageV8Error("terminal source schema differs")
        start = _canonical_date("terminal coverage start", self.coverage_start_date)
        end = _canonical_date("terminal coverage end", self.coverage_end_date)
        if end < start:
            raise MassiveEconomicCoverageV8Error(
                "terminal coverage interval is inverted"
            )
        _text("terminal provider", self.provider_id)
        _text("terminal provider dataset", self.provider_dataset)
        if self.capture_kind not in {
            "fixed-terminal-provider-production-v8",
            "synthetic-terminal-test-v8",
        }:
            raise MassiveEconomicCoverageV8Error("terminal capture kind is unsupported")
        if not isinstance(self.fixed_runtime_captured, bool):
            raise MassiveEconomicCoverageV8Error(
                "terminal fixed-runtime flag is not boolean"
            )
        if self.fixed_runtime_captured and self.capture_kind != (
            "fixed-terminal-provider-production-v8"
        ):
            raise MassiveEconomicCoverageV8Error(
                "synthetic terminal source cannot be runtime-qualified"
            )
        canonical = tuple(
            sorted(
                self.dispositions,
                key=lambda row: (row.effective_at_ms, row.security_id),
            )
        )
        if self.dispositions != canonical or len(
            {row.security_id for row in self.dispositions}
        ) != len(self.dispositions):
            raise MassiveEconomicCoverageV8Error(
                "terminal dispositions are not one-per-security canonical rows"
            )
        for row in self.dispositions:
            row.validate()
            row_date = _date_for_ms(row.effective_at_ms)
            if not start <= row_date <= end:
                raise MassiveEconomicCoverageV8Error(
                    "terminal disposition is outside coverage"
                )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_TERMINAL_COVERAGE_V8_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_TERMINAL_COVERAGE_V8_SOURCE_SCHEMA_SHA256
            or not self.loaded_source.receipt.source_object_key.startswith(
                MASSIVE_TERMINAL_COVERAGE_V8_OBJECT_PREFIX
            )
        ):
            raise MassiveEconomicCoverageV8Error("terminal immutable source differs")
        if self.receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveEconomicCoverageV8Error("terminal source receipt differs")


def _terminal_payload(
    *,
    coverage_start_date: str,
    coverage_end_date: str,
    dispositions: tuple[MassiveTerminalDispositionV8, ...],
    provider_id: str,
    provider_dataset: str,
    capture_kind: str,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_TERMINAL_COVERAGE_V8_SCHEMA,
        "coverage_start_date": coverage_start_date,
        "coverage_end_date": coverage_end_date,
        "dispositions": tuple(asdict(row) for row in dispositions),
        "provider_id": provider_id,
        "provider_dataset": provider_dataset,
        "capture_kind": capture_kind,
    }


def publish_massive_terminal_coverage_source_for_test_v8(
    *,
    root: str | Path,
    coverage_start_date: str,
    coverage_end_date: str,
    dispositions: Sequence[MassiveTerminalDispositionV8],
    entitlement_receipt_sha256: str,
    source_id: str,
    committed_at_ms: int,
) -> MassiveTerminalCoverageSourceV8:
    """Publish synthetic terminal rows; generic/test sources never authorize."""

    rows = tuple(
        sorted(dispositions, key=lambda row: (row.effective_at_ms, row.security_id))
    )
    for row in rows:
        row.validate()
    relative = (
        f"{MASSIVE_TERMINAL_COVERAGE_V8_OBJECT_PREFIX}"
        f"{_text('terminal source ID', source_id)}.json"
    )
    payload = _terminal_payload(
        coverage_start_date=coverage_start_date,
        coverage_end_date=coverage_end_date,
        dispositions=rows,
        provider_id="synthetic-terminal-test-provider",
        provider_dataset="synthetic-terminal-dispositions",
        capture_kind="synthetic-terminal-test-v8",
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_TERMINAL_COVERAGE_V8_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_TERMINAL_COVERAGE_V8_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "terminal entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"TERMINAL-V8-{source_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return parse_massive_terminal_coverage_source_v8(root=root, loaded_source=loaded)


def parse_massive_terminal_coverage_source_v8(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveTerminalCoverageSourceV8:
    """Reopen terminal bytes; generic parsing is always nonauthorizing."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicCoverageV8Error("terminal source is not JSON") from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicCoverageV8Error("terminal source is not canonical JSON")
    _exact_keys(
        payload,
        {
            "schema",
            "coverage_start_date",
            "coverage_end_date",
            "dispositions",
            "provider_id",
            "provider_dataset",
            "capture_kind",
        },
        name="terminal source",
    )
    raw_rows = payload["dispositions"]
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, dict) for row in raw_rows
    ):
        raise MassiveEconomicCoverageV8Error("terminal rows are malformed")
    rows = tuple(MassiveTerminalDispositionV8(**row) for row in raw_rows)
    provisional = MassiveTerminalCoverageSourceV8(
        schema=payload["schema"],
        coverage_start_date=payload["coverage_start_date"],
        coverage_end_date=payload["coverage_end_date"],
        dispositions=rows,
        provider_id=payload["provider_id"],
        provider_dataset=payload["provider_dataset"],
        capture_kind=payload["capture_kind"],
        loaded_source=loaded_source,
        receipt_sha256="0" * 64,
        fixed_runtime_captured=False,
    )
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveNativeEconomicObservationV8:
    surface_id: str
    provider_event_key: str
    logical_event_key: str
    security_id: str
    ticker: str
    kind: str
    classification: str
    effective_at_ms: int
    research_captured_at_ms: int
    accounting_lane: str
    predictive_feature_eligible: bool
    currency: str | None
    cash_per_share: float
    split_adjusted_cash_per_share: float | None
    share_ratio: float
    historical_adjustment_factor: float
    raw_provider_request_id: str
    raw_provider_row_locator: str
    raw_provider_row_sha256: str
    identity_mapping_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.surface_id not in MASSIVE_ECONOMIC_REST_SURFACES_V8:
            raise MassiveEconomicCoverageV8Error(
                "native current-surface event is unsupported"
            )
        for name in (
            "provider_event_key",
            "security_id",
            "ticker",
            "kind",
            "classification",
            "accounting_lane",
            "raw_provider_request_id",
            "raw_provider_row_locator",
        ):
            _text(name, getattr(self, name))
        for value in (
            self.logical_event_key,
            self.raw_provider_row_sha256,
            self.identity_mapping_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("native V8 event digest", value)
        _nonnegative_int("native V8 effective time", self.effective_at_ms)
        _nonnegative_int(
            "native V8 research capture time", self.research_captured_at_ms
        )
        if self.accounting_lane not in MASSIVE_ECONOMIC_ACCOUNTING_LANES_V8:
            raise MassiveEconomicCoverageV8Error("accounting lane is unsupported")
        if self.predictive_feature_eligible is not False:
            raise MassiveEconomicCoverageV8Error(
                "corporate-action terms cannot enter predictive features"
            )
        cash = _finite("native V8 cash", self.cash_per_share)
        ratio = _finite("native V8 share ratio", self.share_ratio)
        factor = _finite(
            "native V8 historical adjustment factor",
            self.historical_adjustment_factor,
        )
        if factor <= 0.0:
            raise MassiveEconomicCoverageV8Error(
                "historical adjustment factor is nonpositive"
            )
        if self.surface_id == "massive-dividends-v1":
            if (
                self.currency != "USD"
                or self.classification not in MASSIVE_DIVIDEND_DISTRIBUTION_TYPES_V8
                or self.kind
                not in {
                    CorporateActionKind.CASH_DIVIDEND.value,
                    CorporateActionKind.SPECIAL_DIVIDEND.value,
                }
                or cash < 0.0
                or self.split_adjusted_cash_per_share is None
                or _finite(
                    "split-adjusted dividend",
                    self.split_adjusted_cash_per_share,
                )
                < 0.0
                or ratio != 1.0
            ):
                raise MassiveEconomicCoverageV8Error("USD dividend terms differ")
        elif (
            self.currency is not None
            or cash != 0.0
            or self.split_adjusted_cash_per_share is not None
            or self.classification not in MASSIVE_SPLIT_ADJUSTMENT_TYPES_V8
            or self.kind
            not in {
                CorporateActionKind.SPLIT.value,
                CorporateActionKind.REVERSE_SPLIT.value,
            }
            or ratio <= 0.0
        ):
            raise MassiveEconomicCoverageV8Error("split terms differ")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicCoverageV8Error("native V8 event receipt differs")


def _native_observation_v8(
    *,
    capture: MassiveEconomicRawRestCaptureV8,
    page_index: int,
    row_index: int,
    row: Mapping[str, object],
    identity_authority: PITSecurityUniverseAuthority,
    accounting_lane: str,
) -> MassiveNativeEconomicObservationV8:
    surface = capture.surface_id
    page = capture.pages[page_index]
    event_key = _text("current provider event ID", row.get("id"))
    ticker = _text("current provider ticker", row.get("ticker"))
    date_field = MASSIVE_ECONOMIC_REST_DATE_FIELDS_V8[surface]
    effective_at_ms = _date_at_market_open_ms(
        f"current {date_field}", row.get(date_field)
    )
    if (
        not capture.coverage_start_date
        <= _date_for_ms(effective_at_ms)
        <= (capture.coverage_end_date)
    ):
        raise MassiveEconomicCoverageV8Error(
            "provider returned an event outside the frozen query interval"
        )
    security_id, identity_receipt = _security_for_ticker(
        identity_authority=identity_authority,
        ticker=ticker,
        effective_at_ms=effective_at_ms,
    )
    if surface == "massive-dividends-v1":
        currency = _text("dividend currency", row.get("currency"))
        if currency != "USD":
            raise MassiveEconomicCoverageV8Error(
                "non-USD dividend requires a PIT FX authority"
            )
        classification = _text(
            "dividend distribution type", row.get("distribution_type")
        )
        if classification not in MASSIVE_DIVIDEND_DISTRIBUTION_TYPES_V8:
            raise MassiveEconomicCoverageV8Error(
                "dividend distribution type is unsupported"
            )
        kind = (
            CorporateActionKind.SPECIAL_DIVIDEND.value
            if classification in {"special", "supplemental", "irregular"}
            else CorporateActionKind.CASH_DIVIDEND.value
        )
        cash = _finite("dividend cash amount", row.get("cash_amount"))
        split_adjusted = _finite(
            "split-adjusted dividend", row.get("split_adjusted_cash_amount")
        )
        ratio = 1.0
    else:
        currency = None
        classification = _text("split adjustment type", row.get("adjustment_type"))
        if classification not in MASSIVE_SPLIT_ADJUSTMENT_TYPES_V8:
            raise MassiveEconomicCoverageV8Error("split adjustment type is unsupported")
        split_from = _finite("split from", row.get("split_from"))
        split_to = _finite("split to", row.get("split_to"))
        if split_from <= 0.0 or split_to <= 0.0:
            raise MassiveEconomicCoverageV8Error("split ratio is nonpositive")
        ratio = split_to / split_from
        if (classification in {"forward_split", "stock_dividend"} and ratio <= 1.0) or (
            classification == "reverse_split" and ratio >= 1.0
        ):
            raise MassiveEconomicCoverageV8Error(
                "split classification and ratio disagree"
            )
        kind = (
            CorporateActionKind.REVERSE_SPLIT.value
            if classification == "reverse_split"
            else CorporateActionKind.SPLIT.value
        )
        cash = 0.0
        split_adjusted = None
    factor = _finite(
        "historical adjustment factor", row.get("historical_adjustment_factor")
    )
    raw_row_sha = semantic_sha256(dict(row))
    logical = semantic_sha256(("massive", surface, event_key))
    provisional = MassiveNativeEconomicObservationV8(
        surface_id=surface,
        provider_event_key=event_key,
        logical_event_key=logical,
        security_id=security_id,
        ticker=ticker,
        kind=kind,
        classification=classification,
        effective_at_ms=effective_at_ms,
        research_captured_at_ms=capture.completed_at_ms,
        accounting_lane=accounting_lane,
        predictive_feature_eligible=False,
        currency=currency,
        cash_per_share=cash,
        split_adjusted_cash_per_share=split_adjusted,
        share_ratio=ratio,
        historical_adjustment_factor=factor,
        raw_provider_request_id=page.provider_request_id,
        raw_provider_row_locator=f"page={page_index}/results={row_index}",
        raw_provider_row_sha256=raw_row_sha,
        identity_mapping_receipt_sha256=identity_receipt,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def adapt_massive_current_economic_captures_at_origin_v8(
    *,
    captures: Sequence[MassiveEconomicRawRestCaptureV8],
    identity_authority: PITSecurityUniverseAuthority,
    decision_at_ms: int,
    accounting_lane: str,
) -> tuple[
    tuple[MassiveNativeEconomicObservationV8, ...],
    tuple[str, ...],
]:
    """Apply a capture-level cutoff; a partial page prefix is never admitted."""

    identity_authority.validate()
    decision = _nonnegative_int("economic decision time", decision_at_ms)
    if accounting_lane not in MASSIVE_ECONOMIC_ACCOUNTING_LANES_V8:
        raise MassiveEconomicCoverageV8Error("accounting lane is unsupported")
    selected: list[MassiveNativeEconomicObservationV8] = []
    future_captures: list[str] = []
    for capture in sorted(captures, key=lambda row: row.surface_id):
        capture.validate()
        if (
            accounting_lane == "strict-pit-capture"
            and capture.completed_at_ms > decision
        ):
            future_captures.append(capture.receipt_sha256)
            continue
        for page in capture.pages:
            raw_rows = page.parsed_body().get("results")
            if not isinstance(raw_rows, list):
                raise MassiveEconomicCoverageV8Error(
                    "current provider result inventory is malformed"
                )
            for index, raw_row in enumerate(raw_rows):
                if not isinstance(raw_row, dict):
                    raise MassiveEconomicCoverageV8Error(
                        "current provider result is not an object"
                    )
                selected.append(
                    _native_observation_v8(
                        capture=capture,
                        page_index=page.page_index,
                        row_index=index,
                        row=raw_row,
                        identity_authority=identity_authority,
                        accounting_lane=accounting_lane,
                    )
                )
    logical = tuple(row.logical_event_key for row in selected)
    if len(logical) != len(set(logical)):
        raise MassiveEconomicCoverageV8Error(
            "current provider event inventory contains duplicate IDs"
        )
    return (
        tuple(
            sorted(
                selected,
                key=lambda row: (
                    row.effective_at_ms,
                    row.security_id,
                    row.logical_event_key,
                ),
            )
        ),
        tuple(sorted(future_captures)),
    )


def _validate_terminal_completeness_v8(
    *,
    terminal_source: MassiveTerminalCoverageSourceV8,
    identity_authority: PITSecurityUniverseAuthority,
    scope: MassiveEconomicCoverageScopeV8,
) -> None:
    expected = {
        row.security_id: row
        for row in identity_authority.delisting_events
        if scope.coverage_start_date
        <= _date_for_ms(row.effective_at_ms)
        <= scope.coverage_end_date
    }
    observed = {row.security_id: row for row in terminal_source.dispositions}
    if set(expected) != set(observed):
        raise MassiveEconomicCoverageV8Error(
            "terminal source does not exactly cover interval delistings"
        )
    masters = {row.security_id: row for row in identity_authority.security_master}
    for security_id, delisting in expected.items():
        disposition = observed[security_id]
        if (
            disposition.delisting_event_id != delisting.event_id
            or disposition.effective_at_ms != delisting.effective_at_ms
            or disposition.delisting_source_receipt_sha256
            != delisting.source_receipt_sha256
            or disposition.successor_security_id != delisting.successor_security_id
            or (
                disposition.successor_security_id is not None
                and disposition.successor_security_id not in masters
            )
        ):
            raise MassiveEconomicCoverageV8Error(
                "terminal disposition and identity delisting differ"
            )


def _ambiguous_interaction_groups_v8(
    *,
    events: Sequence[MassiveNativeEconomicObservationV8],
    identity_authority: PITSecurityUniverseAuthority,
) -> tuple[str, ...]:
    masters = {row.security_id: row for row in identity_authority.security_master}
    groups: dict[tuple[str, int], list[str]] = {}
    for row in events:
        chain = masters[row.security_id].corporate_action_chain_id
        domain = semantic_sha256(
            ("massive-economic-interaction-v8", chain or row.security_id)
        )
        groups.setdefault((domain, row.effective_at_ms), []).append(
            row.logical_event_key
        )
    return tuple(
        sorted(
            semantic_sha256((domain, effective, tuple(sorted(keys))))
            for (domain, effective), keys in groups.items()
            if len(keys) > 1
        )
    )


@dataclass(frozen=True, slots=True)
class MassiveEconomicOriginCoverageV8:
    scope: MassiveEconomicCoverageScopeV8
    decision_at_ms: int
    accounting_lane: str
    capture_receipts: tuple[str, ...]
    selected_events: tuple[MassiveNativeEconomicObservationV8, ...]
    future_capture_receipts: tuple[str, ...]
    terminal_source_receipt_sha256: str
    terminal_inventory_sha256: str
    cash_policy: MassiveZeroCashPolicyV8
    ambiguous_interaction_group_receipts: tuple[str, ...]
    transport_qualified: bool
    coverage_qualified: bool
    economic_authority_qualified: bool
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    schema: str = MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "scope": asdict(self.scope),
            "decision_at_ms": self.decision_at_ms,
            "accounting_lane": self.accounting_lane,
            "selected_events": tuple(asdict(row) for row in self.selected_events),
            "terminal_inventory_sha256": self.terminal_inventory_sha256,
            "cash_policy": asdict(self.cash_policy),
            "ambiguous_interaction_group_receipts": (
                self.ambiguous_interaction_group_receipts
            ),
            "coverage_qualified": self.coverage_qualified,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA:
            raise MassiveEconomicCoverageV8Error("origin coverage schema differs")
        self.scope.validate()
        _nonnegative_int("origin coverage decision", self.decision_at_ms)
        if self.accounting_lane not in MASSIVE_ECONOMIC_ACCOUNTING_LANES_V8:
            raise MassiveEconomicCoverageV8Error("accounting lane is unsupported")
        for inventory in (
            self.capture_receipts,
            self.future_capture_receipts,
            self.ambiguous_interaction_group_receipts,
        ):
            if inventory != tuple(sorted(set(inventory))):
                raise MassiveEconomicCoverageV8Error(
                    "origin coverage inventory is not canonical"
                )
            for value in inventory:
                _digest("origin coverage inventory", value)
        for value in (
            self.terminal_source_receipt_sha256,
            self.terminal_inventory_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("origin coverage digest", value)
        self.cash_policy.validate()
        for row in self.selected_events:
            row.validate()
            if row.accounting_lane != self.accounting_lane:
                raise MassiveEconomicCoverageV8Error(
                    "selected event accounting lane differs"
                )
        if not all(
            isinstance(value, bool)
            for value in (
                self.transport_qualified,
                self.coverage_qualified,
                self.economic_authority_qualified,
            )
        ):
            raise MassiveEconomicCoverageV8Error(
                "coverage qualification flags are not Boolean"
            )
        if self.transport_qualified or self.economic_authority_qualified:
            raise MassiveEconomicCoverageV8Error(
                "generic V8 coverage reload cannot authorize runtime evidence"
            )
        expected_coverage = not self.ambiguous_interaction_group_receipts and (
            self.accounting_lane == "finalized-accounting-research"
            or not self.future_capture_receipts
        )
        if self.coverage_qualified != expected_coverage:
            raise MassiveEconomicCoverageV8Error(
                "structural coverage qualification differs"
            )
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveEconomicCoverageV8Error(
                "origin coverage semantic receipt differs"
            )
        expected_audit = semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "capture_receipts": self.capture_receipts,
                "future_capture_receipts": self.future_capture_receipts,
                "terminal_source_receipt_sha256": (self.terminal_source_receipt_sha256),
            }
        )
        if self.audit_receipt_sha256 != expected_audit:
            raise MassiveEconomicCoverageV8Error(
                "origin coverage audit receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ECONOMIC_COVERAGE_V8_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SCHEMA_SHA256
            or not self.loaded_source.receipt.source_object_key.startswith(
                MASSIVE_ECONOMIC_COVERAGE_V8_OBJECT_PREFIX
            )
        ):
            raise MassiveEconomicCoverageV8Error(
                "origin coverage immutable source differs"
            )


def _origin_payload(
    *,
    scope: MassiveEconomicCoverageScopeV8,
    decision_at_ms: int,
    accounting_lane: str,
    capture_receipts: tuple[str, ...],
    selected_events: tuple[MassiveNativeEconomicObservationV8, ...],
    future_capture_receipts: tuple[str, ...],
    terminal_source_receipt_sha256: str,
    terminal_inventory_sha256: str,
    cash_policy: MassiveZeroCashPolicyV8,
    ambiguous_interaction_group_receipts: tuple[str, ...],
    coverage_qualified: bool,
    semantic_receipt_sha256: str,
    audit_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA,
        "scope": asdict(scope),
        "decision_at_ms": decision_at_ms,
        "accounting_lane": accounting_lane,
        "capture_receipts": capture_receipts,
        "selected_events": tuple(asdict(row) for row in selected_events),
        "future_capture_receipts": future_capture_receipts,
        "terminal_source_receipt_sha256": terminal_source_receipt_sha256,
        "terminal_inventory_sha256": terminal_inventory_sha256,
        "cash_policy": asdict(cash_policy),
        "ambiguous_interaction_group_receipts": (ambiguous_interaction_group_receipts),
        "coverage_qualified": coverage_qualified,
        "semantic_receipt_sha256": semantic_receipt_sha256,
        "audit_receipt_sha256": audit_receipt_sha256,
    }


def materialize_massive_economic_origin_coverage_v8(
    *,
    root: str | Path,
    capture_objects: Sequence[MassiveEconomicRawRestCaptureV8],
    identity_authority: PITSecurityUniverseAuthority,
    terminal_loaded_source: LoadedMassiveSourceObject,
    scope: MassiveEconomicCoverageScopeV8,
    decision_at_ms: int,
    accounting_lane: str,
    entitlement_receipt_sha256: str,
    artifact_id: str,
    committed_at_ms: int,
) -> MassiveEconomicOriginCoverageV8:
    """Reparse exact required sources and publish nonauthorizing V8 coverage."""

    identity_authority.validate()
    scope.validate()
    if tuple(sorted(row.surface_id for row in capture_objects)) != (
        scope.required_surface_ids
    ):
        raise MassiveEconomicCoverageV8Error(
            "origin coverage requires exactly one capture per current surface"
        )
    reparsed: list[MassiveEconomicRawRestCaptureV8] = []
    for supplied in capture_objects:
        supplied.validate()
        parsed = parse_massive_economic_raw_rest_capture_v8(
            root=root, loaded_source=supplied.loaded_source
        )
        if parsed.semantic_unsigned() != supplied.semantic_unsigned():
            raise MassiveEconomicCoverageV8Error(
                "supplied capture differs from committed bytes"
            )
        if (
            parsed.coverage_start_date != scope.coverage_start_date
            or parsed.coverage_end_date != scope.coverage_end_date
        ):
            raise MassiveEconomicCoverageV8Error(
                "capture and coverage scope intervals differ"
            )
        reparsed.append(parsed)
    captures = tuple(sorted(reparsed, key=lambda row: row.surface_id))
    terminal = parse_massive_terminal_coverage_source_v8(
        root=root, loaded_source=terminal_loaded_source
    )
    if (
        terminal.coverage_start_date != scope.coverage_start_date
        or terminal.coverage_end_date != scope.coverage_end_date
    ):
        raise MassiveEconomicCoverageV8Error(
            "terminal and coverage scope intervals differ"
        )
    _validate_terminal_completeness_v8(
        terminal_source=terminal,
        identity_authority=identity_authority,
        scope=scope,
    )
    events, future = adapt_massive_current_economic_captures_at_origin_v8(
        captures=captures,
        identity_authority=identity_authority,
        decision_at_ms=decision_at_ms,
        accounting_lane=accounting_lane,
    )
    ambiguous = _ambiguous_interaction_groups_v8(
        events=events, identity_authority=identity_authority
    )
    cash_policy = MassiveZeroCashPolicyV8.build()
    capture_receipts = tuple(sorted(row.receipt_sha256 for row in captures))
    terminal_inventory = semantic_sha256(
        tuple((row.security_id, row.receipt_sha256) for row in terminal.dispositions)
    )
    coverage_qualified = not ambiguous and (
        accounting_lane == "finalized-accounting-research" or not future
    )
    semantic_unsigned = {
        "schema": MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA,
        "scope": asdict(scope),
        "decision_at_ms": decision_at_ms,
        "accounting_lane": accounting_lane,
        "selected_events": tuple(asdict(row) for row in events),
        "terminal_inventory_sha256": terminal_inventory,
        "cash_policy": asdict(cash_policy),
        "ambiguous_interaction_group_receipts": ambiguous,
        "coverage_qualified": coverage_qualified,
    }
    semantic_receipt = semantic_sha256(semantic_unsigned)
    audit_receipt = semantic_sha256(
        {
            "semantic_receipt_sha256": semantic_receipt,
            "capture_receipts": capture_receipts,
            "future_capture_receipts": future,
            "terminal_source_receipt_sha256": terminal.receipt_sha256,
        }
    )
    payload = _origin_payload(
        scope=scope,
        decision_at_ms=decision_at_ms,
        accounting_lane=accounting_lane,
        capture_receipts=capture_receipts,
        selected_events=events,
        future_capture_receipts=future,
        terminal_source_receipt_sha256=terminal.receipt_sha256,
        terminal_inventory_sha256=terminal_inventory,
        cash_policy=cash_policy,
        ambiguous_interaction_group_receipts=ambiguous,
        coverage_qualified=coverage_qualified,
        semantic_receipt_sha256=semantic_receipt,
        audit_receipt_sha256=audit_receipt,
    )
    relative = (
        f"{MASSIVE_ECONOMIC_COVERAGE_V8_OBJECT_PREFIX}"
        f"{_text('coverage artifact ID', artifact_id)}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ECONOMIC_COVERAGE_V8_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "coverage entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"ECONOMIC-COVERAGE-V8-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return parse_massive_economic_origin_coverage_v8(root=root, loaded_source=loaded)


def parse_massive_economic_origin_coverage_v8(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveEconomicOriginCoverageV8:
    """Reopen V8 coverage; generic reload always clears runtime authority."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicCoverageV8Error("coverage source is not JSON") from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicCoverageV8Error("coverage source is not canonical JSON")
    _exact_keys(
        payload,
        {
            "schema",
            "scope",
            "decision_at_ms",
            "accounting_lane",
            "capture_receipts",
            "selected_events",
            "future_capture_receipts",
            "terminal_source_receipt_sha256",
            "terminal_inventory_sha256",
            "cash_policy",
            "ambiguous_interaction_group_receipts",
            "coverage_qualified",
            "semantic_receipt_sha256",
            "audit_receipt_sha256",
        },
        name="origin coverage source",
    )
    raw_scope = payload["scope"]
    raw_events = payload["selected_events"]
    raw_cash = payload["cash_policy"]
    if (
        not isinstance(raw_scope, dict)
        or not isinstance(raw_events, list)
        or any(not isinstance(row, dict) for row in raw_events)
        or not isinstance(raw_cash, dict)
    ):
        raise MassiveEconomicCoverageV8Error("coverage nested inventory is malformed")
    scope_fields = dict(raw_scope)
    raw_required_surfaces = scope_fields.get("required_surface_ids")
    if not isinstance(raw_required_surfaces, list):
        raise MassiveEconomicCoverageV8Error(
            "coverage scope surface inventory is malformed"
        )
    scope_fields["required_surface_ids"] = tuple(raw_required_surfaces)
    scope = MassiveEconomicCoverageScopeV8(**scope_fields)
    events = tuple(MassiveNativeEconomicObservationV8(**row) for row in raw_events)
    cash_fields = dict(raw_cash)
    raw_periods = cash_fields.get("accrual_period_receipts")
    if not isinstance(raw_periods, list):
        raise MassiveEconomicCoverageV8Error("cash policy periods are malformed")
    cash_fields["accrual_period_receipts"] = tuple(raw_periods)
    cash_policy = MassiveZeroCashPolicyV8(**cash_fields)
    result = MassiveEconomicOriginCoverageV8(
        schema=payload["schema"],
        scope=scope,
        decision_at_ms=payload["decision_at_ms"],
        accounting_lane=payload["accounting_lane"],
        capture_receipts=tuple(payload["capture_receipts"]),
        selected_events=events,
        future_capture_receipts=tuple(payload["future_capture_receipts"]),
        terminal_source_receipt_sha256=payload["terminal_source_receipt_sha256"],
        terminal_inventory_sha256=payload["terminal_inventory_sha256"],
        cash_policy=cash_policy,
        ambiguous_interaction_group_receipts=tuple(
            payload["ambiguous_interaction_group_receipts"]
        ),
        transport_qualified=False,
        coverage_qualified=payload["coverage_qualified"],
        economic_authority_qualified=False,
        semantic_receipt_sha256=payload["semantic_receipt_sha256"],
        audit_receipt_sha256=payload["audit_receipt_sha256"],
        loaded_source=loaded_source,
    )
    regenerated = _origin_payload(
        scope=result.scope,
        decision_at_ms=result.decision_at_ms,
        accounting_lane=result.accounting_lane,
        capture_receipts=result.capture_receipts,
        selected_events=result.selected_events,
        future_capture_receipts=result.future_capture_receipts,
        terminal_source_receipt_sha256=result.terminal_source_receipt_sha256,
        terminal_inventory_sha256=result.terminal_inventory_sha256,
        cash_policy=result.cash_policy,
        ambiguous_interaction_group_receipts=(
            result.ambiguous_interaction_group_receipts
        ),
        coverage_qualified=result.coverage_qualified,
        semantic_receipt_sha256=result.semantic_receipt_sha256,
        audit_receipt_sha256=result.audit_receipt_sha256,
    )
    if raw != canonical_json_file_bytes(regenerated):
        raise MassiveEconomicCoverageV8Error("coverage regenerated payload differs")
    result.validate()
    return result


MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_COVERAGE_V8_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "capture_spec": MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SPEC_SHA256,
        "required_surfaces": tuple(sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8)),
        "origin_cutoff": "complete-capture-or-none",
        "dividend_currency": "USD-only-without-PIT-FX",
        "terminal": "exact-delisting-disposition-coverage-required",
        "cash": MASSIVE_ZERO_CASH_POLICY_V8,
        "ties": "invalidate-origin-no-supplemental-order-snapshot",
        "lanes": MASSIVE_ECONOMIC_ACCOUNTING_LANES_V8,
        "generic_reload": "transport-and-economic-authority-false",
        "source_sha256": MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SHA256,
        "historical_panel_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
    }
)


__all__ = [
    "MASSIVE_DIVIDEND_DISTRIBUTION_TYPES_V8",
    "MASSIVE_ECONOMIC_ACCOUNTING_LANES_V8",
    "MASSIVE_ECONOMIC_COVERAGE_V8_DATASET",
    "MASSIVE_ECONOMIC_COVERAGE_V8_HISTORICAL_PANEL_AUTHORIZED",
    "MASSIVE_ECONOMIC_COVERAGE_V8_OBJECT_PREFIX",
    "MASSIVE_ECONOMIC_COVERAGE_V8_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_ECONOMIC_COVERAGE_V8_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_ECONOMIC_COVERAGE_V8_SCHEMA",
    "MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_COVERAGE_V8_SPEC_SHA256",
    "MASSIVE_SPLIT_ADJUSTMENT_TYPES_V8",
    "MASSIVE_TERMINAL_COVERAGE_V8_DATASET",
    "MASSIVE_TERMINAL_COVERAGE_V8_OBJECT_PREFIX",
    "MASSIVE_TERMINAL_COVERAGE_V8_SCHEMA",
    "MASSIVE_TERMINAL_COVERAGE_V8_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ZERO_CASH_POLICY_V8",
    "MASSIVE_ZERO_CASH_POLICY_V8_RECEIPT_SHA256",
    "MassiveEconomicCoverageScopeV8",
    "MassiveEconomicCoverageV8Error",
    "MassiveEconomicOriginCoverageV8",
    "MassiveNativeEconomicObservationV8",
    "MassiveTerminalCoverageSourceV8",
    "MassiveTerminalDispositionV8",
    "MassiveZeroCashPolicyV8",
    "adapt_massive_current_economic_captures_at_origin_v8",
    "materialize_massive_economic_origin_coverage_v8",
    "parse_massive_economic_origin_coverage_v8",
    "parse_massive_terminal_coverage_source_v8",
    "publish_massive_terminal_coverage_source_for_test_v8",
]
