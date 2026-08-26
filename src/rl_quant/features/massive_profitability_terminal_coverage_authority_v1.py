"""Support-scoped terminal outcomes for Massive P0 accounting.

The authority reopens the committed V8 terminal source, reconciles every
experiment-supported delisting against permanent identity, and assigns a
conservative total-loss fallback when no exact provider disposition exists.
No delisted security may silently disappear from a feature or target path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.contracts import TerminalEventKind
from rl_quant.alpha.pit_universe import DelistingEventRecord, PITSecurityUniverseAuthority
from rl_quant.features.massive_economic_coverage_v8 import (
    MassiveTerminalCoverageSourceV8,
    MassiveTerminalDispositionV8,
    parse_massive_terminal_coverage_source_v8,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    MassiveProfitabilitySecuritySupportV2,
    massive_profitability_identity_semantic_receipt_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-profitability-terminal-coverage-authority-v1"
)
MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "support": "frozen-V2-experiment-security-support",
        "known_disposition": "exact-provider-terms",
        "known_successor": "carry-successor-position",
        "worthless": "cash-zero-successor-none-ratio-zero",
        "unresolved_delisting": "conservative-total-loss",
        "availability": "retained-for-origin-specific-strict-PIT-filtering",
        "silent_drop": "prohibited",
        "performance_authorization": False,
    }
)


class MassiveProfitabilityTerminalCoverageAuthorityV1Error(ValueError):
    """Terminal support or source evidence is incomplete or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTerminalSupportRowV1:
    security_id: str
    listing_delisting_event_id: str | None
    effective_at_ms: int | None
    provider_available_at_ms: int | None
    resolution_kind: str
    cash_per_share: float
    successor_security_id: str | None
    successor_ratio: float
    conservative_total_loss: bool
    identity_delisting_receipt_sha256: str | None
    provider_disposition_receipt_sha256: str | None
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.security_id or self.resolution_kind not in {
            "live-through-coverage",
            *(row.value for row in TerminalEventKind),
            "conservative-total-loss",
        }:
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal support row identity differs"
            )
        live = self.resolution_kind == "live-through-coverage"
        if live:
            if any(
                value is not None
                for value in (
                    self.listing_delisting_event_id,
                    self.effective_at_ms,
                    self.provider_available_at_ms,
                    self.successor_security_id,
                    self.identity_delisting_receipt_sha256,
                    self.provider_disposition_receipt_sha256,
                )
            ) or any(
                value != 0.0 for value in (self.cash_per_share, self.successor_ratio)
            ):
                raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                    "live terminal support contains disposition terms"
                )
        else:
            if (
                not self.listing_delisting_event_id
                or self.effective_at_ms is None
                or self.provider_available_at_ms is None
                or self.effective_at_ms < 0
                or self.provider_available_at_ms < self.effective_at_ms
                or self.identity_delisting_receipt_sha256 is None
            ):
                raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                    "terminal disposition chronology or identity differs"
                )
        if self.cash_per_share < 0.0 or self.successor_ratio < 0.0:
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal payout terms are negative"
            )
        fallback = self.resolution_kind == "conservative-total-loss"
        if self.conservative_total_loss != fallback:
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal fallback flag differs"
            )
        if self.resolution_kind in {
            "worthless",
            "conservative-total-loss",
        } and (
            self.cash_per_share != 0.0
            or self.successor_security_id is not None
            or self.successor_ratio != 0.0
        ):
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "worthless or fallback terminal outcome creates value"
            )
        if self.resolution_kind == TerminalEventKind.MERGER_STOCK.value:
            if self.successor_security_id is None or self.successor_ratio <= 0.0:
                raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                    "stock terminal outcome lacks successor terms"
                )
        elif self.successor_security_id is not None or self.successor_ratio != 0.0:
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "nonstock terminal outcome creates successor shares"
            )
        for value in (
            self.identity_delisting_receipt_sha256,
            self.provider_disposition_receipt_sha256,
        ):
            if value is not None:
                _digest("terminal row source", value)
        _digest("terminal support row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal support row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTerminalCoverageAuthorityV1:
    coverage_start_date: str
    coverage_end_date: str
    supported_security_ids: tuple[str, ...]
    rows: tuple[MassiveProfitabilityTerminalSupportRowV1, ...]
    known_disposition_count: int
    exact_provider_disposition_count: int
    conservative_total_loss_count: int
    terminal_accounting_mode: str
    support_semantic_receipt_sha256: str
    normalized_identity_semantic_receipt_sha256: str
    terminal_source_semantic_receipt_sha256: str
    row_inventory_sha256: str
    structural_terminal_coverage_complete: bool
    terminal_source_runtime_qualified: bool
    terminal_evidence_data_qualified: bool
    conservative_lower_bound_complete: bool
    terminal_accounting_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    terminal_source_audit_receipt_sha256: str
    audit_receipt_sha256: str
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "terminal_source_audit_receipt_sha256",
                "audit_receipt_sha256",
            }
        }

    def validate(self) -> None:
        if (
            self.schema
            != MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SOURCE_SHA256
            or self.coverage_end_date < self.coverage_start_date
            or self.supported_security_ids
            != tuple(sorted(set(self.supported_security_ids)))
        ):
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal authority identity or interval differs"
            )
        keys = tuple(row.security_id for row in self.rows)
        if keys != self.supported_security_ids:
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal authority does not exactly cover experiment support"
            )
        for row in self.rows:
            row.validate()
        known = sum(
            row.resolution_kind
            not in {"live-through-coverage", "conservative-total-loss"}
            for row in self.rows
        )
        fallback = sum(row.conservative_total_loss for row in self.rows)
        exact = sum(
            row.provider_disposition_receipt_sha256 is not None for row in self.rows
        )
        expected_mode = (
            "exact-provider"
            if self.terminal_source_runtime_qualified and fallback == 0
            else "conservative-lower-bound"
        )
        if (
            self.known_disposition_count != known
            or self.exact_provider_disposition_count != exact
            or self.conservative_total_loss_count != fallback
            or self.terminal_accounting_mode != expected_mode
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.structural_terminal_coverage_complete is not True
            or not isinstance(self.terminal_source_runtime_qualified, bool)
            or not isinstance(self.terminal_evidence_data_qualified, bool)
            or self.terminal_evidence_data_qualified
            != self.terminal_source_runtime_qualified
            or self.conservative_lower_bound_complete
            != (self.terminal_accounting_mode == "conservative-lower-bound")
            or self.terminal_accounting_data_qualified is not True
            or any(
                (
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal authority completeness or qualification differs"
            )
        for name in (
            "support_semantic_receipt_sha256",
            "normalized_identity_semantic_receipt_sha256",
            "terminal_source_semantic_receipt_sha256",
            "row_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "terminal_source_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal authority semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "terminal_source_audit_receipt_sha256": (
                    self.terminal_source_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                "terminal authority audit receipt differs"
            )

    def row(self, security_id: str) -> MassiveProfitabilityTerminalSupportRowV1:
        for value in self.rows:
            if value.security_id == security_id:
                return value
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            "terminal support does not contain the requested security"
        )


def _disposition_body(
    *,
    disposition: MassiveTerminalDispositionV8,
    delisting_event_id: str,
    delisting_receipt: str,
) -> dict[str, object]:
    if disposition.delisting_event_id != delisting_event_id:
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            "terminal provider row and identity delisting differ"
        )
    if disposition.terminal_kind == TerminalEventKind.WORTHLESS.value and (
        disposition.cash_per_share != 0.0
        or disposition.successor_security_id is not None
        or disposition.successor_ratio != 0.0
    ):
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            "provider worthless outcome creates economic value"
        )
    return {
        "security_id": disposition.security_id,
        "listing_delisting_event_id": delisting_event_id,
        "effective_at_ms": disposition.effective_at_ms,
        "provider_available_at_ms": disposition.provider_available_at_ms,
        "resolution_kind": disposition.terminal_kind,
        "cash_per_share": disposition.cash_per_share,
        "successor_security_id": disposition.successor_security_id,
        "successor_ratio": disposition.successor_ratio,
        "conservative_total_loss": False,
        "identity_delisting_receipt_sha256": delisting_receipt,
        "provider_disposition_receipt_sha256": disposition.receipt_sha256,
    }


def build_massive_profitability_terminal_coverage_authority_v1(
    *,
    root: str | Path,
    identity_authority: PITSecurityUniverseAuthority,
    security_support: MassiveProfitabilitySecuritySupportV2,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    terminal_source: MassiveTerminalCoverageSourceV8,
) -> MassiveProfitabilityTerminalCoverageAuthorityV1:
    """Reopen and reconcile all terminal outcomes on experiment support."""

    identity_authority.validate()
    security_support.validate()
    daily_input_authority.validate()
    reloaded = parse_massive_terminal_coverage_source_v8(
        root=root, loaded_source=terminal_source.loaded_source
    )
    if reloaded.receipt_sha256 != terminal_source.receipt_sha256:
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            "terminal source differs after committed-byte reparse"
        )
    identity_semantics = massive_profitability_identity_semantic_receipt_v2(
        identity_authority
    )
    if (
        identity_semantics
        != security_support.normalized_identity_semantic_receipt_sha256
        or identity_semantics
        != daily_input_authority.normalized_identity_semantic_receipt_sha256
        or security_support.all_supported_security_ids
        != daily_input_authority.supported_security_ids
        or reloaded.coverage_start_date
        > daily_input_authority.coverage_start_session_date
        or reloaded.coverage_end_date < daily_input_authority.coverage_end_session_date
    ):
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            "terminal source, identity, support, or daily interval differs"
        )
    delistings_by_security: dict[str, list[DelistingEventRecord]] = {}
    for row in identity_authority.delisting_events:
        delistings_by_security.setdefault(row.security_id, []).append(row)
    if any(len(rows) > 1 for rows in delistings_by_security.values()):
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            "identity contains multiple delistings for one permanent security"
        )
    source_dispositions = {row.security_id: row for row in reloaded.dispositions}
    support_set = set(security_support.all_supported_security_ids)
    if any(
        security_id in support_set and security_id not in delistings_by_security
        for security_id in source_dispositions
    ):
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            "provider terminal disposition lacks one identity delisting"
        )
    if any(
        row.successor_security_id is not None
        and row.successor_security_id not in support_set
        for row in source_dispositions.values()
        if row.security_id in support_set
    ):
        raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
            "terminal successor lies outside experiment accounting support"
        )
    dispositions = (
        source_dispositions if terminal_source.fixed_runtime_captured else {}
    )
    rows: list[MassiveProfitabilityTerminalSupportRowV1] = []
    for security_id in security_support.all_supported_security_ids:
        delisting_rows = delistings_by_security.get(security_id, [])
        if not delisting_rows:
            body: dict[str, object] = {
                "security_id": security_id,
                "listing_delisting_event_id": None,
                "effective_at_ms": None,
                "provider_available_at_ms": None,
                "resolution_kind": "live-through-coverage",
                "cash_per_share": 0.0,
                "successor_security_id": None,
                "successor_ratio": 0.0,
                "conservative_total_loss": False,
                "identity_delisting_receipt_sha256": None,
                "provider_disposition_receipt_sha256": None,
            }
        else:
            delisting = delisting_rows[0]
            disposition = dispositions.get(security_id)
            if disposition is None:
                body = {
                    "security_id": security_id,
                    "listing_delisting_event_id": delisting.event_id,
                    "effective_at_ms": delisting.effective_at_ms,
                    "provider_available_at_ms": max(
                        delisting.effective_at_ms, delisting.available_at_ms
                    ),
                    "resolution_kind": "conservative-total-loss",
                    "cash_per_share": 0.0,
                    "successor_security_id": None,
                    "successor_ratio": 0.0,
                    "conservative_total_loss": True,
                    "identity_delisting_receipt_sha256": (
                        delisting.source_receipt_sha256
                    ),
                    "provider_disposition_receipt_sha256": None,
                }
            else:
                if (
                    disposition.effective_at_ms != delisting.effective_at_ms
                    or disposition.delisting_source_receipt_sha256
                    != delisting.source_receipt_sha256
                    or disposition.successor_security_id
                    != delisting.successor_security_id
                ):
                    raise MassiveProfitabilityTerminalCoverageAuthorityV1Error(
                        "terminal disposition does not reconcile to identity"
                    )
                body = _disposition_body(
                    disposition=disposition,
                    delisting_event_id=delisting.event_id,
                    delisting_receipt=delisting.source_receipt_sha256,
                )
        result_row = MassiveProfitabilityTerminalSupportRowV1(
            **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
        )
        result_row.validate()
        rows.append(result_row)
    known = sum(
        row.resolution_kind
        not in {"live-through-coverage", "conservative-total-loss"} for row in rows
    )
    fallback = sum(row.conservative_total_loss for row in rows)
    exact = sum(
        row.provider_disposition_receipt_sha256 is not None for row in rows
    )
    accounting_mode = (
        "exact-provider"
        if terminal_source.fixed_runtime_captured and fallback == 0
        else "conservative-lower-bound"
    )
    semantic: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
        "coverage_start_date": daily_input_authority.coverage_start_session_date,
        "coverage_end_date": daily_input_authority.coverage_end_session_date,
        "supported_security_ids": security_support.all_supported_security_ids,
        "rows": tuple(asdict(row) for row in rows),
        "known_disposition_count": known,
        "exact_provider_disposition_count": exact,
        "conservative_total_loss_count": fallback,
        "terminal_accounting_mode": accounting_mode,
        "support_semantic_receipt_sha256": security_support.semantic_receipt_sha256,
        "normalized_identity_semantic_receipt_sha256": identity_semantics,
        "terminal_source_semantic_receipt_sha256": reloaded.receipt_sha256,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "structural_terminal_coverage_complete": True,
        "terminal_source_runtime_qualified": terminal_source.fixed_runtime_captured,
        "terminal_evidence_data_qualified": terminal_source.fixed_runtime_captured,
        "conservative_lower_bound_complete": (
            accounting_mode == "conservative-lower-bound"
        ),
        "terminal_accounting_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SOURCE_SHA256
        ),
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    terminal_audit = terminal_source.loaded_source.receipt.receipt_sha256
    result = MassiveProfitabilityTerminalCoverageAuthorityV1(
        coverage_start_date=daily_input_authority.coverage_start_session_date,
        coverage_end_date=daily_input_authority.coverage_end_session_date,
        supported_security_ids=security_support.all_supported_security_ids,
        rows=tuple(rows),
        known_disposition_count=known,
        exact_provider_disposition_count=exact,
        conservative_total_loss_count=fallback,
        terminal_accounting_mode=accounting_mode,
        support_semantic_receipt_sha256=security_support.semantic_receipt_sha256,
        normalized_identity_semantic_receipt_sha256=identity_semantics,
        terminal_source_semantic_receipt_sha256=reloaded.receipt_sha256,
        row_inventory_sha256=semantic["row_inventory_sha256"],  # type: ignore[arg-type]
        structural_terminal_coverage_complete=True,
        terminal_source_runtime_qualified=terminal_source.fixed_runtime_captured,
        terminal_evidence_data_qualified=terminal_source.fixed_runtime_captured,
        conservative_lower_bound_complete=(
            accounting_mode == "conservative-lower-bound"
        ),
        terminal_accounting_data_qualified=True,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        terminal_source_audit_receipt_sha256=terminal_audit,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "terminal_source_audit_receipt_sha256": terminal_audit,
            }
        ),
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA",
    "MassiveProfitabilityTerminalCoverageAuthorityV1",
    "MassiveProfitabilityTerminalCoverageAuthorityV1Error",
    "MassiveProfitabilityTerminalSupportRowV1",
    "build_massive_profitability_terminal_coverage_authority_v1",
]
