"""Frozen-source H1/H5/H21/H63 targets for Massive profitability P0."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.contracts import PITAlphaDataError
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
    MassiveProfitabilityAccountingFreezeV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    reject_massive_profitability_legacy_generation_v2,
)
from rl_quant.features.massive_profitability_fill_source_authority_v2 import (
    MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
    MassiveProfitabilityTargetAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_targets_v1 import (
    MassiveProfitabilityTargetRowV1,
    MassiveProfitabilityTargetSpecV1,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA = "rl-quant.massive-profitability-targets-v2"
MASSIVE_PROFITABILITY_TARGETS_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TARGETS_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "origin": MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
        "accounting": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
        "fill": MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
        "terminal": MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
        "accounting_freeze": MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
        "horizons": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
        "primary_horizon": None,
        "return": "source-owned-fill-to-fill-economic-total-simple-return",
        "performance_authorization": False,
    }
)


class MassiveProfitabilityTargetsV2Error(ValueError):
    """A V2 target artifact is detached from frozen source-owned accounting."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTargetsV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTargetsV2:
    decision_session_date: str
    origin_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    target_spec: MassiveProfitabilityTargetSpecV1
    target_accounting_authority_semantic_receipt_sha256: str
    accounting_freeze_semantic_receipt_sha256: str
    terminal_authority_semantic_receipt_sha256: str
    terminal_accounting_mode: str
    rows: tuple[MassiveProfitabilityTargetRowV1, ...]
    valid_counts_by_horizon: tuple[int, ...]
    exact_provider_disposition_count: int
    conservative_total_loss_count: int
    row_inventory_sha256: str
    input_schemas: tuple[str, ...]
    fill_sources_qualified: bool
    economic_values_data_qualified: bool
    terminal_accounting_data_qualified: bool
    source_inputs_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    development_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "audit_receipt_sha256"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_PROFITABILITY_TARGETS_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TARGETS_V2_SOURCE_SHA256
            or self.terminal_accounting_mode
            not in {"exact-provider", "conservative-lower-bound"}
            or not all(
                isinstance(value, bool)
                for value in (
                    self.fill_sources_qualified,
                    self.economic_values_data_qualified,
                    self.terminal_accounting_data_qualified,
                    self.source_inputs_data_qualified,
                )
            )
            or self.source_inputs_data_qualified
            != (
                self.fill_sources_qualified
                and self.economic_values_data_qualified
                and self.terminal_accounting_data_qualified
            )
            or any(
                (
                    self.development_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityTargetsV2Error(
                "targets V2 identity, qualification, or authorization differs"
            )
        self.target_spec.validate()
        for schema in self.input_schemas:
            reject_massive_profitability_legacy_generation_v2(schema)
        if set(self.input_schemas) != {
            MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
        }:
            raise MassiveProfitabilityTargetsV2Error(
                "targets V2 input generations differ"
            )
        keys = tuple(row.security_id for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityTargetsV2Error(
                "targets V2 cross-section differs"
            )
        for row in self.rows:
            row.validate()
            if row.decision_session_date != self.decision_session_date:
                raise MassiveProfitabilityTargetsV2Error(
                    "targets V2 row chronology differs"
                )
        counts = tuple(sum(row.valid[index] for row in self.rows) for index in range(4))
        if (
            self.valid_counts_by_horizon != counts
            or self.conservative_total_loss_count
            != sum(row.conservative_total_loss_fallback for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
        ):
            raise MassiveProfitabilityTargetsV2Error(
                "targets V2 inventory differs"
            )
        for value in (
            self.origin_receipt_sha256,
            self.origin_plan_semantic_receipt_sha256,
            self.target_accounting_authority_semantic_receipt_sha256,
            self.accounting_freeze_semantic_receipt_sha256,
            self.terminal_authority_semantic_receipt_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("targets V2", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityTargetsV2Error(
                "targets V2 semantic receipt differs"
            )


def build_massive_profitability_targets_v2(
    *,
    accounting: MassiveProfitabilityTargetAccountingAuthorityV2,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
) -> MassiveProfitabilityTargetsV2:
    """Compute equal-status targets only from source-owned V2 accounting."""

    accounting.validate()
    origin_plan.validate()
    accounting_freeze.validate()
    terminal_authority.validate()
    if (
        accounting.origin_receipt_sha256
        not in {row.receipt_sha256 for row in origin_plan.origin_plan_v1.origins}
        or accounting.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or accounting.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
        or accounting_freeze.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or accounting_freeze.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityTargetsV2Error(
            "targets V2 authorities do not share one frozen experiment"
        )
    frozen_coverage = next(
        (
            row
            for row in accounting_freeze.coverage_rows
            if row.origin_receipt_sha256 == accounting.origin_receipt_sha256
        ),
        None,
    )
    if (
        frozen_coverage is None
        or frozen_coverage.economic_coverage_audit_receipt_sha256
        != accounting.economic_archive_audit_receipt_sha256
    ):
        raise MassiveProfitabilityTargetsV2Error(
            "target accounting coverage is absent from the accounting freeze"
        )
    spec = MassiveProfitabilityTargetSpecV1.build()
    rows: list[MassiveProfitabilityTargetRowV1] = []
    for path in accounting.rows:
        values: list[float] = []
        valid: list[bool] = []
        terminal_zero: list[bool] = []
        for horizon in MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS:
            try:
                value = accounting.target(
                    security_id=path.security_id, horizon_sessions=horizon
                )
            except PITAlphaDataError:
                value = 0.0
                valid.append(False)
                terminal_zero.append(False)
            else:
                if not math.isfinite(value) or value < -1.0:
                    raise MassiveProfitabilityTargetsV2Error(
                        "source-owned target return is invalid"
                    )
                valid.append(True)
                terminal_zero.append(
                    path.terminal[horizon] and path.values[horizon] == 0.0
                )
            values.append(value)
        body = {
            "decision_session_date": accounting.decision_session_date,
            "security_id": path.security_id,
            "simple_returns": tuple(values),
            "valid": tuple(valid),
            "terminal_zero_value": tuple(terminal_zero),
            "conservative_total_loss_fallback": (
                path.conservative_total_loss_fallback
            ),
            "target_accounting_row_receipt_sha256": path.receipt_sha256,
        }
        row = MassiveProfitabilityTargetRowV1(
            **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
        )
        row.validate()
        rows.append(row)
    counts = tuple(sum(row.valid[index] for row in rows) for index in range(4))
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    schemas = tuple(
        sorted(
            (
                MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
                MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
                MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
                MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
                MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
            )
        )
    )
    qualified = (
        accounting.fill_sources_qualified
        and accounting.economic_values_data_qualified
        and accounting.terminal_accounting_complete
        and terminal_authority.terminal_accounting_data_qualified
        and accounting_freeze.accounting_sources_frozen
        and accounting_freeze.capture_transport_qualified
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA,
        "decision_session_date": accounting.decision_session_date,
        "origin_receipt_sha256": accounting.origin_receipt_sha256,
        "origin_plan_semantic_receipt_sha256": origin_plan.semantic_receipt_sha256,
        "target_spec": asdict(spec),
        "target_accounting_authority_semantic_receipt_sha256": (
            accounting.semantic_receipt_sha256
        ),
        "accounting_freeze_semantic_receipt_sha256": (
            accounting_freeze.semantic_receipt_sha256
        ),
        "terminal_authority_semantic_receipt_sha256": (
            terminal_authority.semantic_receipt_sha256
        ),
        "terminal_accounting_mode": terminal_authority.terminal_accounting_mode,
        "rows": tuple(asdict(row) for row in rows),
        "valid_counts_by_horizon": counts,
        "exact_provider_disposition_count": (
            terminal_authority.exact_provider_disposition_count
        ),
        "conservative_total_loss_count": sum(
            row.conservative_total_loss_fallback for row in rows
        ),
        "row_inventory_sha256": row_inventory,
        "input_schemas": schemas,
        "fill_sources_qualified": accounting.fill_sources_qualified,
        "economic_values_data_qualified": accounting.economic_values_data_qualified,
        "terminal_accounting_data_qualified": (
            terminal_authority.terminal_accounting_data_qualified
        ),
        "source_inputs_data_qualified": qualified,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TARGETS_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TARGETS_V2_SOURCE_SHA256,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    receipt = semantic_sha256(semantic)
    runtime = dict(semantic)
    runtime.pop("target_spec")
    runtime.pop("rows")
    result = MassiveProfitabilityTargetsV2(
        **runtime,  # type: ignore[arg-type]
        target_spec=spec,
        rows=tuple(rows),
        semantic_receipt_sha256=receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": receipt,
                "target_accounting_audit_receipt_sha256": (
                    accounting.audit_receipt_sha256
                ),
                "accounting_freeze_audit_receipt_sha256": (
                    accounting_freeze.audit_receipt_sha256
                ),
                "terminal_audit_receipt_sha256": terminal_authority.audit_receipt_sha256,
            }
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA",
    "MassiveProfitabilityTargetsV2",
    "MassiveProfitabilityTargetsV2Error",
    "build_massive_profitability_targets_v2",
]
