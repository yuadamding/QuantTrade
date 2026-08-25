"""Fail-closed data gate for the bounded Massive P0 feature/target panel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
    MassiveProfitabilityExperimentCoverageV2,
    reject_massive_profitability_legacy_generation_v2,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
    MassiveProfitabilityOriginFeaturesV2,
)
from rl_quant.features.massive_profitability_targets_v1 import (
    MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
    MassiveProfitabilityTargetsV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA = (
    "rl-quant.massive-profitability-data-gate-v1"
)
MASSIVE_PROFITABILITY_DATA_GATE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_DATA_GATE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "required_inputs": (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
        ),
        "dates": "all-frozen-candidate-dates-no-outer-or-lockbox-shifting",
        "common_support": "max-300-or-80-percent-of-decision-members",
        "model_support": "identical-MV02-MV04-MV04-SHUFFLE",
        "reproducibility": "exact-semantic-receipt-equality",
        "future_mutation": "earlier-feature-semantics-invariant",
        "legacy": "hard-rejected",
        "training_implementation": "not-yet-authorized",
        "reporting": False,
        "lockbox": False,
    }
)

MASSIVE_PROFITABILITY_DATA_GATE_V1_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_DATA_GATE_V1_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_DATA_GATE_V1_LOCKBOX_ACCESS_AUTHORIZED = False


class MassiveProfitabilityDataGateV1Error(ValueError):
    """The bounded profitability dataset does not satisfy its frozen gate."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityDataGateV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDateSupportGateV1:
    decision_session_date: str
    decision_member_count: int
    required_common_valid_count: int
    feature_row_count: int
    target_common_valid_count: int
    common_security_inventory_sha256: str
    passed: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        counts = (
            self.decision_member_count,
            self.required_common_valid_count,
            self.feature_row_count,
            self.target_common_valid_count,
        )
        if (
            not self.decision_session_date
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            )
            or not isinstance(self.passed, bool)
            or self.passed
            != (
                self.feature_row_count == self.decision_member_count
                and self.target_common_valid_count >= self.required_common_valid_count
            )
        ):
            raise MassiveProfitabilityDataGateV1Error("date support gate differs")
        _digest("common support inventory", self.common_security_inventory_sha256)
        _digest("date support gate", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityDataGateV1Error(
                "date support gate receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDataGateV1:
    coverage_semantic_receipt_sha256: str
    candidate_session_dates: tuple[str, ...]
    feature_receipts: tuple[str, ...]
    target_receipts: tuple[str, ...]
    date_support_gates: tuple[MassiveProfitabilityDateSupportGateV1, ...]
    input_schemas: tuple[str, ...]
    source_transport_qualified: bool
    rank_bar_data_qualified: bool
    origin_plan_v2_only: bool
    exact_rank_window_complete: bool
    exact_feature_cutoff_complete: bool
    exact_source_staleness_complete: bool
    exact_64_session_rectangles_complete: bool
    tape_population_data_qualified: bool
    fill_source_complete: bool
    economic_accounting_data_qualified: bool
    target_path_complete: bool
    terminal_accounting_complete: bool
    common_model_support_complete: bool
    reproducible_materialization_complete: bool
    future_mutation_invariance_complete: bool
    legacy_generations_rejected: bool
    data_gate_passed: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "audit_receipt_sha256"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_DATA_GATE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_DATA_GATE_V1_SOURCE_SHA256
            or self.candidate_session_dates
            != tuple(sorted(set(self.candidate_session_dates)))
            or any(
                (
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityDataGateV1Error(
                "data gate identity or authorization differs"
            )
        for schema in self.input_schemas:
            reject_massive_profitability_legacy_generation_v2(schema)
        if self.input_schemas != tuple(sorted(set(self.input_schemas))) or set(
            self.input_schemas
        ) != {
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
        }:
            raise MassiveProfitabilityDataGateV1Error("data gate input schemas differ")
        for row in self.date_support_gates:
            row.validate()
        if (
            tuple(row.decision_session_date for row in self.date_support_gates)
            != self.candidate_session_dates
        ):
            raise MassiveProfitabilityDataGateV1Error(
                "data gate does not bind every frozen candidate date"
            )
        component_flags = (
            self.source_transport_qualified,
            self.rank_bar_data_qualified,
            self.origin_plan_v2_only,
            self.exact_rank_window_complete,
            self.exact_feature_cutoff_complete,
            self.exact_source_staleness_complete,
            self.exact_64_session_rectangles_complete,
            self.tape_population_data_qualified,
            self.fill_source_complete,
            self.economic_accounting_data_qualified,
            self.target_path_complete,
            self.terminal_accounting_complete,
            self.common_model_support_complete,
            self.reproducible_materialization_complete,
            self.future_mutation_invariance_complete,
            self.legacy_generations_rejected,
        )
        if any(
            not isinstance(value, bool) for value in component_flags
        ) or self.data_gate_passed != all(component_flags):
            raise MassiveProfitabilityDataGateV1Error(
                "data gate component result differs"
            )
        for value in (
            self.coverage_semantic_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
            *self.feature_receipts,
            *self.target_receipts,
        ):
            _digest("data gate digest", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityDataGateV1Error(
                "data gate semantic receipt differs"
            )


def build_massive_profitability_data_gate_v1(
    *,
    coverage: MassiveProfitabilityExperimentCoverageV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV2],
    targets: Sequence[MassiveProfitabilityTargetsV1],
    rematerialized_feature_receipts: Mapping[str, str],
    rematerialized_target_receipts: Mapping[str, str],
    future_mutation_invariance_by_date: Mapping[str, bool],
    input_schemas: Sequence[str] = (
        MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
        MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
        MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
    ),
) -> MassiveProfitabilityDataGateV1:
    """Evaluate the immutable source-to-target data contract, never model results."""

    coverage.validate()
    feature_by_date = {row.decision_session_date: row for row in features}
    target_by_date = {row.decision_session_date: row for row in targets}
    candidates = coverage.candidate_session_dates
    if (
        len(feature_by_date) != len(tuple(features))
        or len(target_by_date) != len(tuple(targets))
        or set(feature_by_date) != set(candidates)
        or set(target_by_date) != set(candidates)
    ):
        raise MassiveProfitabilityDataGateV1Error(
            "features and targets must cover every frozen candidate date exactly once"
        )
    schemas = tuple(sorted(input_schemas))
    legacy_rejected = True
    for schema in schemas:
        try:
            reject_massive_profitability_legacy_generation_v2(schema)
        except ValueError:
            legacy_rejected = False
            raise
    required_by_date = {
        session_date: (members, required)
        for session_date, members, required in coverage.common_support_requirements
    }
    gates: list[MassiveProfitabilityDateSupportGateV1] = []
    feature_cutoff_complete = True
    staleness_complete = True
    rectangle_complete = True
    tape_population_qualified = True
    fill_complete = True
    economic_accounting_qualified = True
    target_complete = True
    terminal_complete = True
    model_support_complete = True
    reproducible = True
    future_invariant = True
    for session_date in candidates:
        feature = feature_by_date[session_date]
        target = target_by_date[session_date]
        feature.validate()
        target.validate()
        feature_ids = tuple(row.security_id for row in feature.rows)
        target_map = {row.security_id: row for row in target.rows}
        if feature.origin_receipt_sha256 != target.origin_receipt_sha256:
            raise MassiveProfitabilityDataGateV1Error(
                "feature and target origins differ"
            )
        common = tuple(
            security_id
            for security_id in feature_ids
            if security_id in target_map and all(target_map[security_id].valid)
        )
        members, required = required_by_date.get(
            session_date, (len(feature_ids), max(300, (4 * len(feature_ids) + 4) // 5))
        )
        body = {
            "decision_session_date": session_date,
            "decision_member_count": members,
            "required_common_valid_count": required,
            "feature_row_count": len(feature_ids),
            "target_common_valid_count": len(common),
            "common_security_inventory_sha256": semantic_sha256(common),
            "passed": len(feature_ids) == members and len(common) >= required,
        }
        gate = MassiveProfitabilityDateSupportGateV1(
            **body, receipt_sha256=semantic_sha256(body)
        )
        gate.validate()
        gates.append(gate)
        feature_cutoff_complete &= feature.feature_cutoff_at_ms >= 0
        staleness_complete &= feature.source_staleness_sessions == 2
        rectangle_complete &= len(feature.input_session_dates) == 64
        tape_population_qualified &= feature.tape_population_data_qualified
        economic_accounting_qualified &= (
            feature.feature_accounting_data_qualified
            and target.economic_values_data_qualified
        )
        fill_complete &= target.fill_sources_qualified and all(
            count > 0 for count in target.valid_counts_by_horizon
        )
        target_complete &= all(
            row.valid[index]
            for row in target.rows
            for index in range(len(row.valid))
            if row.security_id in common
        )
        terminal_complete &= all(
            not terminal or valid
            for row in target.rows
            for terminal, valid in zip(row.terminal_zero_value, row.valid, strict=True)
        )
        model_support_complete &= gate.passed
        reproducible &= (
            rematerialized_feature_receipts.get(session_date)
            == feature.semantic_receipt_sha256
            and rematerialized_target_receipts.get(session_date)
            == target.semantic_receipt_sha256
        )
        future_invariant &= future_mutation_invariance_by_date.get(session_date) is True
    component = {
        "source_transport_qualified": coverage.source_transport_qualified,
        "rank_bar_data_qualified": coverage.rank_bar_data_qualified,
        "origin_plan_v2_only": True,
        "exact_rank_window_complete": coverage.rank_bar_data_qualified,
        "exact_feature_cutoff_complete": feature_cutoff_complete,
        "exact_source_staleness_complete": staleness_complete,
        "exact_64_session_rectangles_complete": rectangle_complete,
        "tape_population_data_qualified": tape_population_qualified,
        "fill_source_complete": fill_complete,
        "economic_accounting_data_qualified": economic_accounting_qualified,
        "target_path_complete": target_complete,
        "terminal_accounting_complete": terminal_complete,
        "common_model_support_complete": model_support_complete,
        "reproducible_materialization_complete": reproducible,
        "future_mutation_invariance_complete": future_invariant,
        "legacy_generations_rejected": legacy_rejected,
    }
    data_gate_passed = all(component.values())
    semantic = {
        "coverage_semantic_receipt_sha256": coverage.semantic_receipt_sha256,
        "candidate_session_dates": candidates,
        "feature_receipts": tuple(
            feature_by_date[value].semantic_receipt_sha256 for value in candidates
        ),
        "target_receipts": tuple(
            target_by_date[value].semantic_receipt_sha256 for value in candidates
        ),
        "date_support_gates": tuple(asdict(row) for row in gates),
        "input_schemas": schemas,
        **component,
        "data_gate_passed": data_gate_passed,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_DATA_GATE_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_DATA_GATE_V1_SOURCE_SHA256,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "schema": MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA,
    }
    semantic_receipt = semantic_sha256(semantic)
    result = MassiveProfitabilityDataGateV1(
        coverage_semantic_receipt_sha256=coverage.semantic_receipt_sha256,
        candidate_session_dates=candidates,
        feature_receipts=semantic["feature_receipts"],  # type: ignore[arg-type]
        target_receipts=semantic["target_receipts"],  # type: ignore[arg-type]
        date_support_gates=tuple(gates),
        input_schemas=schemas,
        source_transport_qualified=coverage.source_transport_qualified,
        rank_bar_data_qualified=coverage.rank_bar_data_qualified,
        origin_plan_v2_only=True,
        exact_rank_window_complete=coverage.rank_bar_data_qualified,
        exact_feature_cutoff_complete=feature_cutoff_complete,
        exact_source_staleness_complete=staleness_complete,
        exact_64_session_rectangles_complete=rectangle_complete,
        tape_population_data_qualified=tape_population_qualified,
        fill_source_complete=fill_complete,
        economic_accounting_data_qualified=economic_accounting_qualified,
        target_path_complete=target_complete,
        terminal_accounting_complete=terminal_complete,
        common_model_support_complete=model_support_complete,
        reproducible_materialization_complete=reproducible,
        future_mutation_invariance_complete=future_invariant,
        legacy_generations_rejected=legacy_rejected,
        data_gate_passed=data_gate_passed,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_DATA_GATE_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_DATA_GATE_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "coverage_audit_receipt_sha256": coverage.audit_receipt_sha256,
                "feature_audit_receipts": tuple(
                    feature_by_date[value].audit_receipt_sha256 for value in candidates
                ),
                "target_audit_receipts": tuple(
                    target_by_date[value].audit_receipt_sha256 for value in candidates
                ),
            }
        ),
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_DATA_GATE_V1_SPEC_SHA256",
    "MassiveProfitabilityDataGateV1",
    "MassiveProfitabilityDataGateV1Error",
    "MassiveProfitabilityDateSupportGateV1",
    "build_massive_profitability_data_gate_v1",
]
