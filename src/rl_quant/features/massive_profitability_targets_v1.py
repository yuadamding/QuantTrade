"""Equal-status H1/H5/H21/H63 economic targets for Massive profitability P0."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.contracts import PITAlphaDataError
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MassiveProfitabilityTargetAccountingV1,
    compute_massive_profitability_target_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_TARGET_HORIZONS_V1 = (1, 5, 21, 63)
MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA = "rl-quant.massive-profitability-targets-v1"
MASSIVE_PROFITABILITY_TARGETS_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TARGETS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "horizons": MASSIVE_PROFITABILITY_TARGET_HORIZONS_V1,
        "primary_horizon": None,
        "return": "fill-to-fill-economic-total-simple-return",
        "entry": "decision-session-[15:50,16:00)-VWAP",
        "exit": "same-window-H-exchange-sessions-later",
        "terminal": "mandatory-carry-or-conservative-total-loss",
        "survival_required": False,
        "training_scale": "fit-only-median-and-1.4826-MAD",
        "performance_authorization": False,
    }
)

MASSIVE_PROFITABILITY_TARGETS_V1_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_TARGETS_V1_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_TARGETS_V1_LOCKBOX_ACCESS_AUTHORIZED = False


class MassiveProfitabilityTargetsV1Error(ValueError):
    """A target artifact differs from the equal-status P0 target contract."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTargetsV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTargetSpecV1:
    horizons_sessions: tuple[int, ...]
    primary_horizon_sessions: int | None
    return_kind: str
    entry_benchmark: str
    exit_benchmark: str
    target_begins_strictly_after_fill: bool
    terminal_outcomes_included: bool
    future_survival_required: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.horizons_sessions != MASSIVE_PROFITABILITY_TARGET_HORIZONS_V1
            or self.primary_horizon_sessions is not None
            or self.return_kind != "economic-total-simple-return"
            or self.entry_benchmark != "decision-fill-vwap-1550-1600-et"
            or self.exit_benchmark != "horizon-fill-vwap-1550-1600-et"
            or self.target_begins_strictly_after_fill is not True
            or self.terminal_outcomes_included is not True
            or self.future_survival_required is not False
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityTargetsV1Error("target specification differs")

    @classmethod
    def build(cls) -> MassiveProfitabilityTargetSpecV1:
        body = {
            "horizons_sessions": MASSIVE_PROFITABILITY_TARGET_HORIZONS_V1,
            "primary_horizon_sessions": None,
            "return_kind": "economic-total-simple-return",
            "entry_benchmark": "decision-fill-vwap-1550-1600-et",
            "exit_benchmark": "horizon-fill-vwap-1550-1600-et",
            "target_begins_strictly_after_fill": True,
            "terminal_outcomes_included": True,
            "future_survival_required": False,
        }
        result = cls(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTargetRowV1:
    decision_session_date: str
    security_id: str
    simple_returns: tuple[float, ...]
    valid: tuple[bool, ...]
    terminal_zero_value: tuple[bool, ...]
    conservative_total_loss_fallback: bool
    target_accounting_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.decision_session_date
            or not self.security_id
            or len(self.simple_returns) != 4
            or len(self.valid) != 4
            or len(self.terminal_zero_value) != 4
            or any(not isinstance(value, bool) for value in self.valid)
            or any(not isinstance(value, bool) for value in self.terminal_zero_value)
            or not isinstance(self.conservative_total_loss_fallback, bool)
            or any(
                not math.isfinite(float(value))
                or value < -1.0
                or (not valid and value != 0.0)
                for value, valid in zip(self.simple_returns, self.valid, strict=True)
            )
            or any(
                terminal and (not valid or value != -1.0)
                for terminal, valid, value in zip(
                    self.terminal_zero_value,
                    self.valid,
                    self.simple_returns,
                    strict=True,
                )
            )
        ):
            raise MassiveProfitabilityTargetsV1Error(
                "target row values or masks differ"
            )
        _digest("target accounting row", self.target_accounting_row_receipt_sha256)
        _digest("target row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityTargetsV1Error("target row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTargetsV1:
    decision_session_date: str
    origin_receipt_sha256: str
    target_spec: MassiveProfitabilityTargetSpecV1
    target_accounting_semantic_receipt_sha256: str
    rows: tuple[MassiveProfitabilityTargetRowV1, ...]
    valid_counts_by_horizon: tuple[int, ...]
    conservative_total_loss_count: int
    row_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    fill_sources_qualified: bool
    economic_values_data_qualified: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "audit_receipt_sha256"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_PROFITABILITY_TARGETS_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TARGETS_V1_SOURCE_SHA256
            or any(
                (
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
            or not isinstance(self.fill_sources_qualified, bool)
            or not isinstance(self.economic_values_data_qualified, bool)
        ):
            raise MassiveProfitabilityTargetsV1Error(
                "target artifact identity or authorization differs"
            )
        self.target_spec.validate()
        keys = tuple(row.security_id for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityTargetsV1Error(
                "target rows are not one canonical cross-section"
            )
        for row in self.rows:
            row.validate()
            if row.decision_session_date != self.decision_session_date:
                raise MassiveProfitabilityTargetsV1Error(
                    "target row decision date differs"
                )
        expected_counts = tuple(
            sum(row.valid[index] for row in self.rows) for index in range(4)
        )
        if (
            self.valid_counts_by_horizon != expected_counts
            or self.conservative_total_loss_count
            != sum(row.conservative_total_loss_fallback for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityTargetsV1Error(
                "target artifact inventory differs"
            )
        for value in (
            self.origin_receipt_sha256,
            self.target_accounting_semantic_receipt_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("target artifact digest", value)


def build_massive_profitability_targets_v1(
    *, accounting: MassiveProfitabilityTargetAccountingV1
) -> MassiveProfitabilityTargetsV1:
    """Compute all equal-status horizons, retaining explicit invalid masks."""

    accounting.validate()
    spec = MassiveProfitabilityTargetSpecV1.build()
    rows: list[MassiveProfitabilityTargetRowV1] = []
    for path in accounting.rows:
        values: list[float] = []
        valid: list[bool] = []
        terminal_zero: list[bool] = []
        for horizon in MASSIVE_PROFITABILITY_TARGET_HORIZONS_V1:
            try:
                return_value = compute_massive_profitability_target_v1(
                    accounting=accounting,
                    security_id=path.security_id,
                    horizon_sessions=horizon,
                )
            except PITAlphaDataError:
                return_value = 0.0
                valid.append(False)
                terminal_zero.append(False)
            else:
                valid.append(True)
                terminal_zero.append(return_value == -1.0)
            values.append(return_value)
        body = {
            "decision_session_date": accounting.decision_session_date,
            "security_id": path.security_id,
            "simple_returns": tuple(values),
            "valid": tuple(valid),
            "terminal_zero_value": tuple(terminal_zero),
            "conservative_total_loss_fallback": (path.conservative_total_loss_fallback),
            "target_accounting_row_receipt_sha256": path.receipt_sha256,
        }
        row = MassiveProfitabilityTargetRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        rows.append(row)
    counts = tuple(sum(row.valid[index] for row in rows) for index in range(4))
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    semantic = {
        "decision_session_date": accounting.decision_session_date,
        "origin_receipt_sha256": accounting.origin_receipt_sha256,
        "target_spec": asdict(spec),
        "target_accounting_semantic_receipt_sha256": accounting.semantic_receipt_sha256,
        "rows": tuple(asdict(row) for row in rows),
        "valid_counts_by_horizon": counts,
        "conservative_total_loss_count": sum(
            row.conservative_total_loss_fallback for row in rows
        ),
        "row_inventory_sha256": row_inventory,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TARGETS_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TARGETS_V1_SOURCE_SHA256,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "fill_sources_qualified": accounting.fill_sources_qualified,
        "economic_values_data_qualified": accounting.economic_values_data_qualified,
        "schema": MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
    }
    semantic_receipt = semantic_sha256(semantic)
    artifact = MassiveProfitabilityTargetsV1(
        decision_session_date=accounting.decision_session_date,
        origin_receipt_sha256=accounting.origin_receipt_sha256,
        target_spec=spec,
        target_accounting_semantic_receipt_sha256=accounting.semantic_receipt_sha256,
        rows=tuple(rows),
        valid_counts_by_horizon=counts,
        conservative_total_loss_count=semantic["conservative_total_loss_count"],  # type: ignore[arg-type]
        row_inventory_sha256=row_inventory,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_TARGETS_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_TARGETS_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "target_accounting_audit_receipt_sha256": accounting.audit_receipt_sha256,
            }
        ),
        fill_sources_qualified=accounting.fill_sources_qualified,
        economic_values_data_qualified=accounting.economic_values_data_qualified,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    artifact.validate()
    return artifact


__all__ = [
    "MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_TARGETS_V1_SPEC_SHA256",
    "MASSIVE_PROFITABILITY_TARGET_HORIZONS_V1",
    "MassiveProfitabilityTargetRowV1",
    "MassiveProfitabilityTargetSpecV1",
    "MassiveProfitabilityTargetsV1",
    "MassiveProfitabilityTargetsV1Error",
    "build_massive_profitability_targets_v1",
]
