"""Executable source-to-panel gate for Massive P0 development and outer folds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.economic_provider_capture_v8 import (
    MASSIVE_ECONOMIC_REST_SURFACES_V8,
    MassiveEconomicRawRestCaptureV8,
    parse_massive_economic_raw_rest_capture_v8,
)
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionManifestV0,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_daily_bars_v0 import MassiveDailyBarsArtifactV0
from rl_quant.features.massive_daily_tape_v0 import MassiveDailyTapeArtifactV0
from rl_quant.features.massive_economic_coverage_v8 import (
    MassiveEconomicOriginCoverageV8,
    MassiveTerminalCoverageSourceV8,
)
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
    MassiveProfitabilityAccountingFreezeV1,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA,
    MassiveProfitabilityArchiveFreezeV1,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
    MassiveProfitabilityDailyInputAuthorityV1,
    build_massive_profitability_daily_input_authority_v1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
    MassiveProfitabilityExperimentCoverageV2,
    MassiveProfitabilitySecuritySupportV2,
    reject_massive_profitability_legacy_generation_v2,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
    MassiveProfitabilityFeatureAccountingAuthorityV2,
    build_massive_profitability_feature_accounting_authority_v2,
)
from rl_quant.features.massive_profitability_fill_source_authority_v2 import (
    MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
    build_massive_profitability_fill_source_authority_v2,
)
from rl_quant.features.massive_profitability_lockbox_target_seal_v1 import (
    MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA,
    MassiveProfitabilityLockboxTargetSealV1,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA,
    MassiveProfitabilityOriginFeaturesV3,
    build_massive_profitability_origin_features_v3,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
    MassiveProfitabilityDecisionOriginPlanV2,
    MassiveProfitabilityProductionAcquisitionV2,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
    build_massive_profitability_target_accounting_authority_v2,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA,
    MassiveProfitabilityTargetsV2,
    build_massive_profitability_targets_v2,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
    build_massive_profitability_terminal_coverage_authority_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_DATA_GATE_V2_SCHEMA = "rl-quant.massive-profitability-data-gate-v2"
MASSIVE_PROFITABILITY_DATA_GATE_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
_MASSIVE_PROFITABILITY_DATA_GATE_V2_INPUT_SCHEMAS = tuple(
    sorted(
        (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
            MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA,
            MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA,
            MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA,
        )
    )
)
MASSIVE_PROFITABILITY_DATA_GATE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "inputs": (
            MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA,
            MASSIVE_PROFITABILITY_TARGETS_V2_SCHEMA,
            MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA,
        ),
        "attestations": "none-caller-supplied",
        "rematerialization": "package-owned-daily-fill-terminal-accounting-features-targets",
        "training_scope": "development-and-outer-folds-only",
        "lockbox": "commitment-only-targets-excluded",
        "reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityDataGateV2Error(ValueError):
    """The executable source reconstruction does not pass the P0 data gate."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityDataGateV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDataGateSourceBundleV2:
    source_root: str | Path
    persisted_root: str | Path
    daily_bars_root: str | Path
    daily_tape_root: str | Path
    session_authority: MassiveSessionAuthority
    identity_authority: PITSecurityUniverseAuthority
    condition_authority: MassiveConditionAuthority
    correction_authority: MassiveCorrectionAuthority
    acquisition: MassiveProfitabilityProductionAcquisitionV2
    scan_evidence: tuple[MassiveDailyTradeFileScanEvidenceV0, ...]
    semantic_partition_manifests: tuple[MassiveDailyTradePartitionManifestV0, ...]
    persisted_partition_manifests: tuple[MassivePersistedPartitionManifestV1, ...]
    daily_bars: tuple[MassiveDailyBarsArtifactV0, ...]
    daily_tape: tuple[MassiveDailyTapeArtifactV0, ...]
    economic_captures: tuple[MassiveEconomicRawRestCaptureV8, ...]
    economic_coverages: tuple[MassiveEconomicOriginCoverageV8, ...]
    terminal_source: MassiveTerminalCoverageSourceV8


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDateSupportGateV2:
    decision_session_date: str
    decision_member_count: int
    required_common_valid_count: int
    feature_row_count: int
    target_common_valid_count: int
    common_security_inventory_sha256: str
    phase: str
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
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            )
            or self.phase not in {"development", "outer-test"}
            or not isinstance(self.passed, bool)
            or self.passed
            != (
                self.feature_row_count == self.decision_member_count
                and self.target_common_valid_count >= self.required_common_valid_count
            )
        ):
            raise MassiveProfitabilityDataGateV2Error(
                "data gate V2 date support differs"
            )
        _digest("data gate V2 common support", self.common_security_inventory_sha256)
        _digest("data gate V2 support row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityDataGateV2Error(
                "data gate V2 support row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDataGateV2:
    coverage_semantic_receipt_sha256: str
    archive_freeze_semantic_receipt_sha256: str
    accounting_freeze_semantic_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    daily_input_authority_semantic_receipt_sha256: str
    fill_source_authority_semantic_receipt_sha256: str
    terminal_authority_semantic_receipt_sha256: str
    lockbox_seal_semantic_receipt_sha256: str
    gated_session_dates: tuple[str, ...]
    outer_test_session_dates: tuple[str, ...]
    excluded_lockbox_session_dates: tuple[str, ...]
    feature_receipts: tuple[str, ...]
    target_receipts: tuple[str, ...]
    date_support_gates: tuple[MassiveProfitabilityDateSupportGateV2, ...]
    input_schemas: tuple[str, ...]
    source_transport_qualified: bool
    rank_bar_data_qualified: bool
    exact_frozen_acquisition_complete: bool
    exact_accounting_freeze_complete: bool
    exact_origin_plan_membership_complete: bool
    exact_feature_cutoff_complete: bool
    exact_source_staleness_complete: bool
    exact_64_session_rectangles_complete: bool
    fill_source_complete: bool
    economic_accounting_data_qualified: bool
    terminal_accounting_complete: bool
    common_model_support_complete: bool
    package_rematerialization_complete: bool
    future_mutation_invariance_complete: bool
    lockbox_targets_sealed_and_excluded: bool
    legacy_generations_rejected: bool
    data_gate_passed: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    component_audit_inventory_sha256: str
    audit_receipt_sha256: str
    development_training_authorized: bool
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_DATA_GATE_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "component_audit_inventory_sha256",
                "audit_receipt_sha256",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_DATA_GATE_V2_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_DATA_GATE_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_DATA_GATE_V2_SOURCE_SHA256
            or self.gated_session_dates
            != tuple(sorted(set(self.gated_session_dates)))
            or self.outer_test_session_dates
            != tuple(sorted(set(self.outer_test_session_dates)))
            or self.excluded_lockbox_session_dates
            != tuple(sorted(set(self.excluded_lockbox_session_dates)))
            or set(self.gated_session_dates) & set(self.excluded_lockbox_session_dates)
            or self.input_schemas
            != _MASSIVE_PROFITABILITY_DATA_GATE_V2_INPUT_SCHEMAS
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityDataGateV2Error(
                "data gate V2 identity, dates, or authorization differs"
            )
        for row in self.date_support_gates:
            row.validate()
        if tuple(row.decision_session_date for row in self.date_support_gates) != (
            self.gated_session_dates
        ):
            raise MassiveProfitabilityDataGateV2Error(
                "data gate V2 support inventory differs"
            )
        components = (
            self.source_transport_qualified,
            self.rank_bar_data_qualified,
            self.exact_frozen_acquisition_complete,
            self.exact_accounting_freeze_complete,
            self.exact_origin_plan_membership_complete,
            self.exact_feature_cutoff_complete,
            self.exact_source_staleness_complete,
            self.exact_64_session_rectangles_complete,
            self.fill_source_complete,
            self.economic_accounting_data_qualified,
            self.terminal_accounting_complete,
            self.common_model_support_complete,
            self.package_rematerialization_complete,
            self.future_mutation_invariance_complete,
            self.lockbox_targets_sealed_and_excluded,
            self.legacy_generations_rejected,
        )
        if (
            any(not isinstance(value, bool) for value in components)
            or self.data_gate_passed != all(components)
            or self.development_training_authorized != self.data_gate_passed
            or self.outer_prediction_authorized != self.data_gate_passed
        ):
            raise MassiveProfitabilityDataGateV2Error(
                "data gate V2 component or limited authorization differs"
            )
        for schema in self.input_schemas:
            reject_massive_profitability_legacy_generation_v2(schema)
        for value in (
            self.coverage_semantic_receipt_sha256,
            self.archive_freeze_semantic_receipt_sha256,
            self.accounting_freeze_semantic_receipt_sha256,
            self.origin_plan_semantic_receipt_sha256,
            self.daily_input_authority_semantic_receipt_sha256,
            self.fill_source_authority_semantic_receipt_sha256,
            self.terminal_authority_semantic_receipt_sha256,
            self.lockbox_seal_semantic_receipt_sha256,
            *self.feature_receipts,
            *self.target_receipts,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.component_audit_inventory_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("data gate V2", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityDataGateV2Error(
                "data gate V2 semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "component_audit_inventory_sha256": (
                    self.component_audit_inventory_sha256
                ),
            }
        ):
            raise MassiveProfitabilityDataGateV2Error(
                "data gate V2 audit receipt differs"
            )


def _future_audit_isolation(
    accounting: MassiveProfitabilityFeatureAccountingAuthorityV2,
) -> bool:
    mutated_audit = semantic_sha256(
        ("future-economic-audit-canary", accounting.audit_receipt_sha256)
    )
    candidate = replace(
        accounting,
        economic_archive_audit_receipt_sha256=mutated_audit,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": accounting.semantic_receipt_sha256,
                "economic_archive_audit_receipt_sha256": mutated_audit,
            }
        ),
    )
    candidate.validate()
    return candidate.semantic_receipt_sha256 == accounting.semantic_receipt_sha256


def _rematerialize_daily(
    *,
    bundle: MassiveProfitabilityDataGateSourceBundleV2,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    security_support: MassiveProfitabilitySecuritySupportV2,
) -> MassiveProfitabilityDailyInputAuthorityV1:
    return build_massive_profitability_daily_input_authority_v1(
        source_root=bundle.source_root,
        persisted_root=bundle.persisted_root,
        daily_bars_root=bundle.daily_bars_root,
        daily_tape_root=bundle.daily_tape_root,
        session_authority=bundle.session_authority,
        identity_authority=bundle.identity_authority,
        condition_authority=bundle.condition_authority,
        correction_authority=bundle.correction_authority,
        acquisition=bundle.acquisition,
        archive_freeze=archive_freeze,
        security_support=security_support,
        scan_evidence=bundle.scan_evidence,
        semantic_partition_manifests=bundle.semantic_partition_manifests,
        persisted_partition_manifests=bundle.persisted_partition_manifests,
        daily_bars=bundle.daily_bars,
        daily_tape=bundle.daily_tape,
    )


def _accounting_freeze_matches_bundle(
    *,
    bundle: MassiveProfitabilityDataGateSourceBundleV2,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
) -> bool:
    captures = tuple(sorted(bundle.economic_captures, key=lambda row: row.surface_id))
    if tuple(row.surface_id for row in captures) != tuple(
        sorted(MASSIVE_ECONOMIC_REST_SURFACES_V8)
    ):
        return False
    frozen_capture_rows = {
        row.surface_id: row for row in accounting_freeze.capture_rows
    }
    if len(frozen_capture_rows) != len(accounting_freeze.capture_rows):
        return False
    for capture in captures:
        capture.validate()
        reparsed = parse_massive_economic_raw_rest_capture_v8(
            root=bundle.source_root, loaded_source=capture.loaded_source
        )
        frozen_capture = frozen_capture_rows.get(capture.surface_id)
        if (
            frozen_capture is None
            or not capture.fixed_runtime_captured
            or reparsed.receipt_sha256 != capture.receipt_sha256
            or capture.completed_at_ms > accounting_freeze.accounting_freeze_at_ms
            or frozen_capture.capture_receipt_sha256 != capture.receipt_sha256
            or frozen_capture.capture_completed_at_ms != capture.completed_at_ms
            or frozen_capture.loaded_source_receipt_sha256
            != capture.loaded_source.receipt_sha256
        ):
            return False
    terminal = bundle.terminal_source
    terminal.validate()
    if (
        accounting_freeze.terminal_source_semantic_receipt_sha256
        != terminal.receipt_sha256
        or accounting_freeze.terminal_source_loaded_receipt_sha256
        != terminal.loaded_source.receipt_sha256
        or accounting_freeze.terminal_source_committed_at_ms
        != terminal.loaded_source.commit.committed_at_ms
        or terminal.loaded_source.commit.committed_at_ms
        > accounting_freeze.accounting_freeze_at_ms
    ):
        return False
    origins = {
        row.decision_at_ms: row for row in origin_plan.origin_plan_v1.origins
    }
    coverages = {
        row.decision_at_ms: row for row in bundle.economic_coverages
    }
    frozen_coverages = {
        row.decision_at_ms: row for row in accounting_freeze.coverage_rows
    }
    if set(origins) != set(coverages) or set(origins) != set(frozen_coverages):
        return False
    for decision_at_ms, origin in origins.items():
        coverage = coverages[decision_at_ms]
        coverage.validate()
        frozen_coverage = frozen_coverages[decision_at_ms]
        if (
            frozen_coverage.origin_receipt_sha256 != origin.receipt_sha256
            or frozen_coverage.economic_coverage_semantic_receipt_sha256
            != coverage.semantic_receipt_sha256
            or frozen_coverage.economic_coverage_audit_receipt_sha256
            != coverage.audit_receipt_sha256
            or frozen_coverage.economic_coverage_loaded_source_receipt_sha256
            != coverage.loaded_source.receipt_sha256
            or frozen_coverage.committed_at_ms
            != coverage.loaded_source.commit.committed_at_ms
            or frozen_coverage.committed_at_ms
            > accounting_freeze.accounting_freeze_at_ms
        ):
            return False
    return accounting_freeze.capture_transport_qualified


def build_massive_profitability_data_gate_v2(
    *,
    coverage: MassiveProfitabilityExperimentCoverageV2,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    security_support: MassiveProfitabilitySecuritySupportV2,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    lockbox_seal: MassiveProfitabilityLockboxTargetSealV1,
    source_bundle: MassiveProfitabilityDataGateSourceBundleV2,
) -> MassiveProfitabilityDataGateV2:
    """Reconstruct the complete nonlockbox panel; accept no caller attestations."""

    coverage.validate()
    archive_freeze.validate()
    origin_plan.validate()
    security_support.validate()
    accounting_freeze.validate()
    lockbox_seal.validate()
    if (
        coverage.archive_freeze_semantic_receipt_sha256
        != archive_freeze.semantic_receipt_sha256
        or coverage.security_support_semantic_receipt_sha256
        != security_support.semantic_receipt_sha256
        or security_support.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or accounting_freeze.archive_freeze_semantic_receipt_sha256
        != archive_freeze.semantic_receipt_sha256
        or accounting_freeze.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or lockbox_seal.archive_freeze_semantic_receipt_sha256
        != archive_freeze.semantic_receipt_sha256
        or lockbox_seal.accounting_freeze_semantic_receipt_sha256
        != accounting_freeze.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityDataGateV2Error(
            "data gate V2 frozen experiment components differ"
        )
    daily = _rematerialize_daily(
        bundle=source_bundle,
        archive_freeze=archive_freeze,
        security_support=security_support,
    )
    daily_again = _rematerialize_daily(
        bundle=source_bundle,
        archive_freeze=archive_freeze,
        security_support=security_support,
    )
    fill = build_massive_profitability_fill_source_authority_v2(
        persisted_root=source_bundle.persisted_root,
        session_authority=source_bundle.session_authority,
        condition_authority=source_bundle.condition_authority,
        daily_input_authority=daily,
        persisted_partition_manifests=source_bundle.persisted_partition_manifests,
        origin_plan=origin_plan,
        security_support=security_support,
    )
    fill_again = build_massive_profitability_fill_source_authority_v2(
        persisted_root=source_bundle.persisted_root,
        session_authority=source_bundle.session_authority,
        condition_authority=source_bundle.condition_authority,
        daily_input_authority=daily_again,
        persisted_partition_manifests=source_bundle.persisted_partition_manifests,
        origin_plan=origin_plan,
        security_support=security_support,
    )
    terminal = build_massive_profitability_terminal_coverage_authority_v1(
        root=source_bundle.source_root,
        identity_authority=source_bundle.identity_authority,
        security_support=security_support,
        daily_input_authority=daily,
        terminal_source=source_bundle.terminal_source,
    )
    terminal_again = build_massive_profitability_terminal_coverage_authority_v1(
        root=source_bundle.source_root,
        identity_authority=source_bundle.identity_authority,
        security_support=security_support,
        daily_input_authority=daily_again,
        terminal_source=source_bundle.terminal_source,
    )
    if (
        terminal.semantic_receipt_sha256
        != accounting_freeze.terminal_authority_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityDataGateV2Error(
            "rematerialized terminal authority differs from accounting freeze"
        )
    coverage_by_decision = {
        row.decision_at_ms: row for row in source_bundle.economic_coverages
    }
    if len(coverage_by_decision) != len(source_bundle.economic_coverages):
        raise MassiveProfitabilityDataGateV2Error(
            "economic coverages duplicate a decision origin"
        )
    lockbox_dates = set(archive_freeze.fixed_lockbox_session_dates)
    outer_dates = tuple(
        value
        for inventory in archive_freeze.fixed_outer_test_session_inventories
        for value in inventory
    )
    origins = tuple(
        row
        for row in origin_plan.origin_plan_v1.origins
        if row.decision_session_date not in lockbox_dates
    )
    if not set(outer_dates) <= {row.decision_session_date for row in origins}:
        raise MassiveProfitabilityDataGateV2Error(
            "outer-test phase contains a skipped decision origin"
        )
    features: list[MassiveProfitabilityOriginFeaturesV3] = []
    targets: list[MassiveProfitabilityTargetsV2] = []
    reproducible = (
        daily.semantic_receipt_sha256 == daily_again.semantic_receipt_sha256
        and fill.semantic_receipt_sha256 == fill_again.semantic_receipt_sha256
        and terminal.semantic_receipt_sha256 == terminal_again.semantic_receipt_sha256
    )
    future_invariant = True
    for origin in origins:
        economic = coverage_by_decision.get(origin.decision_at_ms)
        if economic is None:
            raise MassiveProfitabilityDataGateV2Error(
                "origin lacks frozen economic coverage"
            )
        feature_accounting = build_massive_profitability_feature_accounting_authority_v2(
            root=source_bundle.source_root,
            origin=origin,
            origin_plan=origin_plan,
            session_authority=source_bundle.session_authority,
            identity_authority=source_bundle.identity_authority,
            daily_input_authority=daily,
            economic_coverage=economic,
            terminal_authority=terminal,
        )
        target_accounting = build_massive_profitability_target_accounting_authority_v2(
            root=source_bundle.source_root,
            origin=origin,
            origin_plan=origin_plan,
            session_authority=source_bundle.session_authority,
            identity_authority=source_bundle.identity_authority,
            daily_input_authority=daily,
            fill_source_authority=fill,
            economic_coverage=economic,
            terminal_authority=terminal,
        )
        feature = build_massive_profitability_origin_features_v3(
            origin=origin,
            origin_plan=origin_plan,
            session_authority=source_bundle.session_authority,
            identity_authority=source_bundle.identity_authority,
            daily_input_authority=daily,
            feature_accounting=feature_accounting,
            accounting_freeze=accounting_freeze,
            terminal_authority=terminal,
        )
        target = build_massive_profitability_targets_v2(
            accounting=target_accounting,
            origin_plan=origin_plan,
            accounting_freeze=accounting_freeze,
            terminal_authority=terminal,
        )
        second_feature_accounting = (
            build_massive_profitability_feature_accounting_authority_v2(
                root=source_bundle.source_root,
                origin=origin,
                origin_plan=origin_plan,
                session_authority=source_bundle.session_authority,
                identity_authority=source_bundle.identity_authority,
                daily_input_authority=daily_again,
                economic_coverage=economic,
                terminal_authority=terminal_again,
            )
        )
        second_target_accounting = (
            build_massive_profitability_target_accounting_authority_v2(
                root=source_bundle.source_root,
                origin=origin,
                origin_plan=origin_plan,
                session_authority=source_bundle.session_authority,
                identity_authority=source_bundle.identity_authority,
                daily_input_authority=daily_again,
                fill_source_authority=fill_again,
                economic_coverage=economic,
                terminal_authority=terminal_again,
            )
        )
        second_feature = build_massive_profitability_origin_features_v3(
            origin=origin,
            origin_plan=origin_plan,
            session_authority=source_bundle.session_authority,
            identity_authority=source_bundle.identity_authority,
            daily_input_authority=daily_again,
            feature_accounting=second_feature_accounting,
            accounting_freeze=accounting_freeze,
            terminal_authority=terminal_again,
        )
        second_target = build_massive_profitability_targets_v2(
            accounting=second_target_accounting,
            origin_plan=origin_plan,
            accounting_freeze=accounting_freeze,
            terminal_authority=terminal_again,
        )
        reproducible &= (
            feature.semantic_receipt_sha256 == second_feature.semantic_receipt_sha256
            and target.semantic_receipt_sha256 == second_target.semantic_receipt_sha256
        )
        future_invariant &= _future_audit_isolation(feature_accounting)
        features.append(feature)
        targets.append(target)
    feature_by_date = {row.decision_session_date: row for row in features}
    target_by_date = {row.decision_session_date: row for row in targets}
    requirements = {
        session_date: (members, required)
        for session_date, members, required in coverage.common_support_requirements
    }
    gates: list[MassiveProfitabilityDateSupportGateV2] = []
    for session_date in sorted(feature_by_date):
        feature = feature_by_date[session_date]
        target = target_by_date[session_date]
        target_map = {row.security_id: row for row in target.rows}
        common = tuple(
            row.security_id
            for row in feature.rows
            if row.security_id in target_map and all(target_map[row.security_id].valid)
        )
        members, required = requirements[session_date]
        body = {
            "decision_session_date": session_date,
            "decision_member_count": members,
            "required_common_valid_count": required,
            "feature_row_count": len(feature.rows),
            "target_common_valid_count": len(common),
            "common_security_inventory_sha256": semantic_sha256(common),
            "phase": "outer-test" if session_date in set(outer_dates) else "development",
            "passed": len(feature.rows) == members and len(common) >= required,
        }
        gates.append(
            MassiveProfitabilityDateSupportGateV2(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    gated_dates = tuple(row.decision_session_date for row in gates)
    schemas = _MASSIVE_PROFITABILITY_DATA_GATE_V2_INPUT_SCHEMAS
    for schema in schemas:
        reject_massive_profitability_legacy_generation_v2(schema)
    accounting_freeze_complete = _accounting_freeze_matches_bundle(
        bundle=source_bundle,
        accounting_freeze=accounting_freeze,
        origin_plan=origin_plan,
    )
    components = {
        "source_transport_qualified": daily.source_transport_qualified,
        "rank_bar_data_qualified": coverage.rank_bar_data_qualified,
        "exact_frozen_acquisition_complete": (
            daily.archive_freeze_semantic_receipt_sha256
            == archive_freeze.semantic_receipt_sha256
        ),
        "exact_accounting_freeze_complete": accounting_freeze_complete,
        "exact_origin_plan_membership_complete": all(
            row.origin_plan_semantic_receipt_sha256
            == origin_plan.semantic_receipt_sha256
            for row in features
        )
        and all(
            row.origin_plan_semantic_receipt_sha256
            == origin_plan.semantic_receipt_sha256
            for row in targets
        ),
        "exact_feature_cutoff_complete": all(
            row.maximum_economic_input_at_ms == row.feature_cutoff_at_ms
            for row in features
        ),
        "exact_source_staleness_complete": all(
            row.source_staleness_sessions == 2 for row in features
        ),
        "exact_64_session_rectangles_complete": all(
            len(row.input_session_dates) == 64 for row in features
        ),
        "fill_source_complete": fill.fill_source_data_qualified
        and all(all(value > 0 for value in row.valid_counts_by_horizon) for row in targets),
        "economic_accounting_data_qualified": all(
            row.source_inputs_data_qualified for row in features
        )
        and all(row.source_inputs_data_qualified for row in targets),
        "terminal_accounting_complete": terminal.terminal_accounting_data_qualified,
        "common_model_support_complete": all(row.passed for row in gates),
        "package_rematerialization_complete": reproducible,
        "future_mutation_invariance_complete": future_invariant,
        "lockbox_targets_sealed_and_excluded": (
            lockbox_seal.lockbox_session_dates
            == archive_freeze.fixed_lockbox_session_dates
            and not set(gated_dates) & set(lockbox_seal.lockbox_session_dates)
        ),
        "legacy_generations_rejected": True,
    }
    passed = all(components.values())
    semantic = {
        "schema": MASSIVE_PROFITABILITY_DATA_GATE_V2_SCHEMA,
        "coverage_semantic_receipt_sha256": coverage.semantic_receipt_sha256,
        "archive_freeze_semantic_receipt_sha256": archive_freeze.semantic_receipt_sha256,
        "accounting_freeze_semantic_receipt_sha256": (
            accounting_freeze.semantic_receipt_sha256
        ),
        "origin_plan_semantic_receipt_sha256": origin_plan.semantic_receipt_sha256,
        "daily_input_authority_semantic_receipt_sha256": daily.semantic_receipt_sha256,
        "fill_source_authority_semantic_receipt_sha256": fill.semantic_receipt_sha256,
        "terminal_authority_semantic_receipt_sha256": terminal.semantic_receipt_sha256,
        "lockbox_seal_semantic_receipt_sha256": lockbox_seal.semantic_receipt_sha256,
        "gated_session_dates": gated_dates,
        "outer_test_session_dates": outer_dates,
        "excluded_lockbox_session_dates": archive_freeze.fixed_lockbox_session_dates,
        "feature_receipts": tuple(row.semantic_receipt_sha256 for row in features),
        "target_receipts": tuple(row.semantic_receipt_sha256 for row in targets),
        "date_support_gates": tuple(asdict(row) for row in gates),
        "input_schemas": schemas,
        **components,
        "data_gate_passed": passed,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_DATA_GATE_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_DATA_GATE_V2_SOURCE_SHA256,
        "development_training_authorized": passed,
        "outer_prediction_authorized": passed,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    receipt = semantic_sha256(semantic)
    component_audit_inventory = semantic_sha256(
        {
            "coverage_audit_receipt_sha256": coverage.audit_receipt_sha256,
            "archive_audit_receipt_sha256": archive_freeze.audit_receipt_sha256,
            "accounting_freeze_audit_receipt_sha256": (
                accounting_freeze.audit_receipt_sha256
            ),
            "daily_input_audit_receipt_sha256": daily.audit_receipt_sha256,
            "fill_source_audit_receipt_sha256": fill.audit_receipt_sha256,
            "terminal_audit_receipt_sha256": terminal.audit_receipt_sha256,
            "feature_audit_receipts": tuple(row.audit_receipt_sha256 for row in features),
            "target_audit_receipts": tuple(row.audit_receipt_sha256 for row in targets),
            "lockbox_seal_audit_receipt_sha256": lockbox_seal.audit_receipt_sha256,
        }
    )
    runtime = dict(semantic)
    runtime.pop("date_support_gates")
    result = MassiveProfitabilityDataGateV2(
        **runtime,  # type: ignore[arg-type]
        date_support_gates=tuple(gates),
        semantic_receipt_sha256=receipt,
        component_audit_inventory_sha256=component_audit_inventory,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": receipt,
                "component_audit_inventory_sha256": component_audit_inventory,
            }
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_DATA_GATE_V2_SCHEMA",
    "MassiveProfitabilityDataGateSourceBundleV2",
    "MassiveProfitabilityDataGateV2",
    "MassiveProfitabilityDataGateV2Error",
    "MassiveProfitabilityDateSupportGateV2",
    "build_massive_profitability_data_gate_v2",
]
