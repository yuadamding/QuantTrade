"""Immutable accounting-source freeze for the Massive P0 experiment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.economic_provider_capture_v8 import (
    MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SHA256,
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
from rl_quant.features.massive_economic_coverage_v8 import (
    MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SHA256,
    MassiveEconomicOriginCoverageV8,
    MassiveTerminalCoverageSourceV8,
    parse_massive_economic_origin_coverage_v8,
    parse_massive_terminal_coverage_source_v8,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MassiveProfitabilityArchiveFreezeV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SOURCE_SHA256,
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA = (
    "rl-quant.massive-profitability-accounting-freeze-v1"
)
MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_DATASET = (
    "massive-profitability-accounting-freeze-v1"
)
MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
        "format": "canonical-json-newline",
        "fields": "exact",
    }
)
MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "lane": "finalized-accounting-research",
        "captures": tuple(sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8)),
        "cutoff": "capture-and-terminal-and-derived-coverage-committed-by-freeze",
        "cash": "zero-return",
        "terminal": "exact-provider-or-conservative-lower-bound",
        "coverage": "one-origin-coverage-per-frozen-v2-origin",
        "performance_authorization": False,
    }
)


class MassiveProfitabilityAccountingFreezeV1Error(ValueError):
    """Accounting sources do not equal the pretraining frozen inventory."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityAccountingFreezeV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityAccountingCaptureFreezeRowV1:
    surface_id: str
    capture_receipt_sha256: str
    capture_completed_at_ms: int
    loaded_source_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.surface_id not in MASSIVE_ECONOMIC_REST_SURFACES_V8
            or self.capture_completed_at_ms < 0
        ):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting capture freeze row differs"
            )
        for value in (
            self.capture_receipt_sha256,
            self.loaded_source_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("accounting capture row", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting capture freeze row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityAccountingCoverageFreezeRowV1:
    decision_at_ms: int
    origin_receipt_sha256: str
    economic_coverage_semantic_receipt_sha256: str
    economic_coverage_audit_receipt_sha256: str
    economic_coverage_loaded_source_receipt_sha256: str
    committed_at_ms: int
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.decision_at_ms < 0 or self.committed_at_ms < 0:
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting coverage freeze chronology differs"
            )
        for value in (
            self.origin_receipt_sha256,
            self.economic_coverage_semantic_receipt_sha256,
            self.economic_coverage_audit_receipt_sha256,
            self.economic_coverage_loaded_source_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("accounting coverage row", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting coverage freeze row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityAccountingFreezeV1:
    accounting_freeze_at_ms: int
    accounting_lane: str
    coverage_start_date: str
    coverage_end_date: str
    archive_freeze_semantic_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    capture_rows: tuple[MassiveProfitabilityAccountingCaptureFreezeRowV1, ...]
    coverage_rows: tuple[MassiveProfitabilityAccountingCoverageFreezeRowV1, ...]
    terminal_source_semantic_receipt_sha256: str
    terminal_source_loaded_receipt_sha256: str
    terminal_source_committed_at_ms: int
    terminal_authority_semantic_receipt_sha256: str
    terminal_accounting_mode: str
    zero_cash_policy_receipt_sha256: str
    capture_inventory_sha256: str
    coverage_inventory_sha256: str
    implementation_inventory_sha256: str
    accounting_sources_frozen: bool
    capture_transport_qualified: bool
    conservative_lower_bound_frozen: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    development_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "audit_receipt_sha256",
                "loaded_source",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA
            or self.accounting_freeze_at_ms < 0
            or self.accounting_lane != "finalized-accounting-research"
            or self.coverage_end_date < self.coverage_start_date
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SOURCE_SHA256
        ):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting freeze identity or interval differs"
            )
        if tuple(row.surface_id for row in self.capture_rows) != tuple(
            sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8)
        ):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting freeze capture surfaces differ"
            )
        for capture_row in self.capture_rows:
            capture_row.validate()
            if capture_row.capture_completed_at_ms > self.accounting_freeze_at_ms:
                raise MassiveProfitabilityAccountingFreezeV1Error(
                    "accounting capture completed after the freeze"
                )
        decisions = tuple(row.decision_at_ms for row in self.coverage_rows)
        if not decisions or decisions != tuple(sorted(set(decisions))):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting coverage decisions are not canonical"
            )
        for coverage_row in self.coverage_rows:
            coverage_row.validate()
            if coverage_row.committed_at_ms > self.accounting_freeze_at_ms:
                raise MassiveProfitabilityAccountingFreezeV1Error(
                    "economic coverage committed after the freeze"
                )
        if (
            self.terminal_source_committed_at_ms > self.accounting_freeze_at_ms
            or self.terminal_accounting_mode
            not in {"exact-provider", "conservative-lower-bound"}
            or self.conservative_lower_bound_frozen
            != (self.terminal_accounting_mode == "conservative-lower-bound")
            or self.accounting_sources_frozen is not True
            or not isinstance(self.capture_transport_qualified, bool)
            or any(
                (
                    self.development_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting freeze qualification or authorization differs"
            )
        if self.capture_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.capture_rows)
        ) or self.coverage_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.coverage_rows)
        ):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting freeze inventory differs"
            )
        for value in (
            self.archive_freeze_semantic_receipt_sha256,
            self.origin_plan_semantic_receipt_sha256,
            self.terminal_source_semantic_receipt_sha256,
            self.terminal_source_loaded_receipt_sha256,
            self.terminal_authority_semantic_receipt_sha256,
            self.zero_cash_policy_receipt_sha256,
            self.capture_inventory_sha256,
            self.coverage_inventory_sha256,
            self.implementation_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("accounting freeze", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting freeze semantic receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SOURCE_SCHEMA_SHA256
            or self.audit_receipt_sha256
            != semantic_sha256(
                {
                    "semantic_receipt_sha256": self.semantic_receipt_sha256,
                    "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
                }
            )
        ):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting freeze committed source differs"
            )


def _payload(
    *,
    accounting_freeze_at_ms: int,
    accounting_lane: str,
    coverage_start_date: str,
    coverage_end_date: str,
    archive_freeze_semantic_receipt_sha256: str,
    origin_plan_semantic_receipt_sha256: str,
    capture_rows: Sequence[MassiveProfitabilityAccountingCaptureFreezeRowV1],
    coverage_rows: Sequence[MassiveProfitabilityAccountingCoverageFreezeRowV1],
    terminal_source_semantic_receipt_sha256: str,
    terminal_source_loaded_receipt_sha256: str,
    terminal_source_committed_at_ms: int,
    terminal_authority_semantic_receipt_sha256: str,
    terminal_accounting_mode: str,
    zero_cash_policy_receipt_sha256: str,
    capture_transport_qualified: bool,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
        "accounting_freeze_at_ms": accounting_freeze_at_ms,
        "accounting_lane": accounting_lane,
        "coverage_start_date": coverage_start_date,
        "coverage_end_date": coverage_end_date,
        "archive_freeze_semantic_receipt_sha256": (
            archive_freeze_semantic_receipt_sha256
        ),
        "origin_plan_semantic_receipt_sha256": origin_plan_semantic_receipt_sha256,
        "capture_rows": tuple(asdict(row) for row in capture_rows),
        "coverage_rows": tuple(asdict(row) for row in coverage_rows),
        "terminal_source_semantic_receipt_sha256": (
            terminal_source_semantic_receipt_sha256
        ),
        "terminal_source_loaded_receipt_sha256": terminal_source_loaded_receipt_sha256,
        "terminal_source_committed_at_ms": terminal_source_committed_at_ms,
        "terminal_authority_semantic_receipt_sha256": (
            terminal_authority_semantic_receipt_sha256
        ),
        "terminal_accounting_mode": terminal_accounting_mode,
        "zero_cash_policy_receipt_sha256": zero_cash_policy_receipt_sha256,
        "capture_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in capture_rows)
        ),
        "coverage_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in coverage_rows)
        ),
        "implementation_inventory_sha256": semantic_sha256(
            (
                MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SHA256,
                MASSIVE_ECONOMIC_COVERAGE_V8_SOURCE_SHA256,
                MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SOURCE_SHA256,
            )
        ),
        "accounting_sources_frozen": True,
        "capture_transport_qualified": capture_transport_qualified,
        "conservative_lower_bound_frozen": (
            terminal_accounting_mode == "conservative-lower-bound"
        ),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SOURCE_SHA256
        ),
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }


def materialize_massive_profitability_accounting_freeze_v1(
    *,
    root: str | Path,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    capture_objects: Sequence[MassiveEconomicRawRestCaptureV8],
    terminal_source: MassiveTerminalCoverageSourceV8,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
    economic_coverages: Sequence[MassiveEconomicOriginCoverageV8],
    accounting_freeze_at_ms: int,
    entitlement_receipt_sha256: str,
    artifact_id: str,
) -> MassiveProfitabilityAccountingFreezeV1:
    """Freeze exact economic captures and one derived coverage per V2 origin."""

    archive_freeze.validate()
    origin_plan.validate()
    terminal_authority.validate()
    captures = tuple(sorted(capture_objects, key=lambda row: row.surface_id))
    if tuple(row.surface_id for row in captures) != tuple(
        sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8)
    ):
        raise MassiveProfitabilityAccountingFreezeV1Error(
            "accounting freeze requires exactly the current economic surfaces"
        )
    capture_rows: list[MassiveProfitabilityAccountingCaptureFreezeRowV1] = []
    for capture in captures:
        capture.validate()
        reparsed = parse_massive_economic_raw_rest_capture_v8(
            root=root, loaded_source=capture.loaded_source
        )
        if (
            reparsed.receipt_sha256 != capture.receipt_sha256
            or not capture.fixed_runtime_captured
            or capture.completed_at_ms > accounting_freeze_at_ms
        ):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "accounting capture is not fixed-runtime evidence frozen in time"
            )
        body = {
            "surface_id": capture.surface_id,
            "capture_receipt_sha256": capture.receipt_sha256,
            "capture_completed_at_ms": capture.completed_at_ms,
            "loaded_source_receipt_sha256": capture.loaded_source.receipt_sha256,
        }
        capture_rows.append(
            MassiveProfitabilityAccountingCaptureFreezeRowV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    reparsed_terminal = parse_massive_terminal_coverage_source_v8(
        root=root, loaded_source=terminal_source.loaded_source
    )
    if (
        reparsed_terminal.receipt_sha256 != terminal_source.receipt_sha256
        or terminal_source.loaded_source.commit.committed_at_ms
        > accounting_freeze_at_ms
        or terminal_authority.terminal_source_semantic_receipt_sha256
        != terminal_source.receipt_sha256
    ):
        raise MassiveProfitabilityAccountingFreezeV1Error(
            "terminal source differs from the frozen accounting authority"
        )
    expected_origins = {
        row.decision_at_ms: row
        for row in origin_plan.origin_plan_v1.origins
    }
    coverage_by_decision = {row.decision_at_ms: row for row in economic_coverages}
    if (
        len(expected_origins) != len(origin_plan.origin_plan_v1.origins)
        or len(coverage_by_decision) != len(tuple(economic_coverages))
        or set(expected_origins) != set(coverage_by_decision)
    ):
        raise MassiveProfitabilityAccountingFreezeV1Error(
            "accounting freeze lacks one economic coverage per V2 origin"
        )
    capture_receipts = tuple(
        sorted(row.capture_receipt_sha256 for row in capture_rows)
    )
    coverage_rows: list[MassiveProfitabilityAccountingCoverageFreezeRowV1] = []
    zero_cash_receipt: str | None = None
    scope: tuple[str, str] | None = None
    for decision_at_ms, origin in sorted(expected_origins.items()):
        coverage = coverage_by_decision[decision_at_ms]
        coverage.validate()
        reparsed = parse_massive_economic_origin_coverage_v8(
            root=root, loaded_source=coverage.loaded_source
        )
        if (
            reparsed.semantic_receipt_sha256 != coverage.semantic_receipt_sha256
            or coverage.accounting_lane != "finalized-accounting-research"
            or coverage.capture_receipts != capture_receipts
            or coverage.terminal_source_receipt_sha256
            != terminal_source.receipt_sha256
            or coverage.loaded_source.commit.committed_at_ms
            > accounting_freeze_at_ms
        ):
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "origin economic coverage differs from frozen accounting sources"
            )
        current_scope = (
            coverage.scope.coverage_start_date,
            coverage.scope.coverage_end_date,
        )
        scope = current_scope if scope is None else scope
        if current_scope != scope:
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "origin economic coverage scopes differ"
            )
        current_cash = coverage.cash_policy.receipt_sha256
        zero_cash_receipt = current_cash if zero_cash_receipt is None else zero_cash_receipt
        if current_cash != zero_cash_receipt:
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "origin economic coverages use different cash policies"
            )
        body = {
            "decision_at_ms": decision_at_ms,
            "origin_receipt_sha256": origin.receipt_sha256,
            "economic_coverage_semantic_receipt_sha256": (
                coverage.semantic_receipt_sha256
            ),
            "economic_coverage_audit_receipt_sha256": coverage.audit_receipt_sha256,
            "economic_coverage_loaded_source_receipt_sha256": (
                coverage.loaded_source.receipt_sha256
            ),
            "committed_at_ms": coverage.loaded_source.commit.committed_at_ms,
        }
        coverage_rows.append(
            MassiveProfitabilityAccountingCoverageFreezeRowV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    assert scope is not None and zero_cash_receipt is not None
    payload = _payload(
        accounting_freeze_at_ms=accounting_freeze_at_ms,
        accounting_lane="finalized-accounting-research",
        coverage_start_date=scope[0],
        coverage_end_date=scope[1],
        archive_freeze_semantic_receipt_sha256=archive_freeze.semantic_receipt_sha256,
        origin_plan_semantic_receipt_sha256=origin_plan.semantic_receipt_sha256,
        capture_rows=tuple(capture_rows),
        coverage_rows=tuple(coverage_rows),
        terminal_source_semantic_receipt_sha256=terminal_source.receipt_sha256,
        terminal_source_loaded_receipt_sha256=terminal_source.loaded_source.receipt_sha256,
        terminal_source_committed_at_ms=(
            terminal_source.loaded_source.commit.committed_at_ms
        ),
        terminal_authority_semantic_receipt_sha256=(
            terminal_authority.semantic_receipt_sha256
        ),
        terminal_accounting_mode=terminal_authority.terminal_accounting_mode,
        zero_cash_policy_receipt_sha256=zero_cash_receipt,
        capture_transport_qualified=all(row.fixed_runtime_captured for row in captures),
    )
    relative = f"massive-profitability-accounting-freeze-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=accounting_freeze_at_ms,
        downloaded_at_ms=accounting_freeze_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "accounting freeze entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=accounting_freeze_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=accounting_freeze_at_ms,
    )
    return parse_massive_profitability_accounting_freeze_v1(
        root=root, loaded_source=loaded
    )


def materialize_massive_profitability_accounting_freeze_for_test_v1(
    *,
    root: str | Path,
    archive_freeze_semantic_receipt_sha256: str,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
    economic_coverages: Sequence[MassiveEconomicOriginCoverageV8],
    accounting_freeze_at_ms: int,
    entitlement_receipt_sha256: str,
    artifact_id: str,
) -> MassiveProfitabilityAccountingFreezeV1:
    """Persist deterministic fixture semantics without transport qualification."""

    origin_plan.validate()
    terminal_authority.validate()
    capture_rows = []
    for surface_id in sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8):
        body = {
            "surface_id": surface_id,
            "capture_receipt_sha256": semantic_sha256(("test-capture", surface_id)),
            "capture_completed_at_ms": accounting_freeze_at_ms,
            "loaded_source_receipt_sha256": semantic_sha256(
                ("test-capture-source", surface_id)
            ),
        }
        capture_rows.append(
            MassiveProfitabilityAccountingCaptureFreezeRowV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    origins = {row.decision_at_ms: row for row in origin_plan.origin_plan_v1.origins}
    coverages = {row.decision_at_ms: row for row in economic_coverages}
    if set(origins) != set(coverages):
        raise MassiveProfitabilityAccountingFreezeV1Error(
            "test accounting freeze lacks one coverage per origin"
        )
    coverage_rows = []
    scope: tuple[str, str] | None = None
    zero_cash: str | None = None
    for decision_at_ms, origin in sorted(origins.items()):
        coverage = coverages[decision_at_ms]
        coverage.validate()
        current_scope = (
            coverage.scope.coverage_start_date,
            coverage.scope.coverage_end_date,
        )
        scope = current_scope if scope is None else scope
        zero_cash = (
            coverage.cash_policy.receipt_sha256 if zero_cash is None else zero_cash
        )
        if current_scope != scope or coverage.cash_policy.receipt_sha256 != zero_cash:
            raise MassiveProfitabilityAccountingFreezeV1Error(
                "test accounting coverage scope or cash policy differs"
            )
        body = {
            "decision_at_ms": decision_at_ms,
            "origin_receipt_sha256": origin.receipt_sha256,
            "economic_coverage_semantic_receipt_sha256": (
                coverage.semantic_receipt_sha256
            ),
            "economic_coverage_audit_receipt_sha256": coverage.audit_receipt_sha256,
            "economic_coverage_loaded_source_receipt_sha256": (
                coverage.loaded_source.receipt_sha256
            ),
            "committed_at_ms": coverage.loaded_source.commit.committed_at_ms,
        }
        coverage_rows.append(
            MassiveProfitabilityAccountingCoverageFreezeRowV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    assert scope is not None and zero_cash is not None
    payload = _payload(
        accounting_freeze_at_ms=accounting_freeze_at_ms,
        accounting_lane="finalized-accounting-research",
        coverage_start_date=scope[0],
        coverage_end_date=scope[1],
        archive_freeze_semantic_receipt_sha256=_digest(
            "test archive freeze", archive_freeze_semantic_receipt_sha256
        ),
        origin_plan_semantic_receipt_sha256=origin_plan.semantic_receipt_sha256,
        capture_rows=tuple(capture_rows),
        coverage_rows=tuple(coverage_rows),
        terminal_source_semantic_receipt_sha256=(
            terminal_authority.terminal_source_semantic_receipt_sha256
        ),
        terminal_source_loaded_receipt_sha256=semantic_sha256(
            "test-terminal-loaded-source"
        ),
        terminal_source_committed_at_ms=accounting_freeze_at_ms,
        terminal_authority_semantic_receipt_sha256=(
            terminal_authority.semantic_receipt_sha256
        ),
        terminal_accounting_mode=terminal_authority.terminal_accounting_mode,
        zero_cash_policy_receipt_sha256=zero_cash,
        capture_transport_qualified=False,
    )
    relative = f"massive-profitability-accounting-freeze-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=accounting_freeze_at_ms,
        downloaded_at_ms=accounting_freeze_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=accounting_freeze_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=accounting_freeze_at_ms,
    )
    return parse_massive_profitability_accounting_freeze_v1(
        root=root, loaded_source=loaded
    )


def parse_massive_profitability_accounting_freeze_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityAccountingFreezeV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityAccountingFreezeV1Error(
            "accounting freeze is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityAccountingFreezeV1Error(
            "accounting freeze is not canonical JSON"
        )
    capture_rows = tuple(
        MassiveProfitabilityAccountingCaptureFreezeRowV1(**row)
        for row in payload["capture_rows"]
    )
    coverage_rows = tuple(
        MassiveProfitabilityAccountingCoverageFreezeRowV1(**row)
        for row in payload["coverage_rows"]
    )
    semantic = dict(payload)
    semantic["capture_rows"] = tuple(asdict(row) for row in capture_rows)
    semantic["coverage_rows"] = tuple(asdict(row) for row in coverage_rows)
    receipt = semantic_sha256(semantic)
    runtime = dict(payload)
    runtime.pop("capture_rows")
    runtime.pop("coverage_rows")
    result = MassiveProfitabilityAccountingFreezeV1(
        **runtime,  # type: ignore[arg-type]
        capture_rows=capture_rows,
        coverage_rows=coverage_rows,
        semantic_receipt_sha256=receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": receipt,
                "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
            }
        ),
        loaded_source=loaded_source,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA",
    "MassiveProfitabilityAccountingCaptureFreezeRowV1",
    "MassiveProfitabilityAccountingCoverageFreezeRowV1",
    "MassiveProfitabilityAccountingFreezeV1",
    "MassiveProfitabilityAccountingFreezeV1Error",
    "materialize_massive_profitability_accounting_freeze_for_test_v1",
    "materialize_massive_profitability_accounting_freeze_v1",
    "parse_massive_profitability_accounting_freeze_v1",
]
