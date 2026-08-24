"""Immutable subordinate origin policy for finalized validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.data_sources.massive.finalized_archive_scope import (
    MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SPEC_SHA256,
    MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_artifact_readiness import (
    MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256,
    MASSIVE_ARTIFACT_READINESS_PANEL_SELECTION_SPEC_SHA256,
    MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1,
)
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_execution_authority import (
    MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256,
    MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
    MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_runtime_authority import (
    MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256,
    MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
    MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
    MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA_SHA256,
)
from rl_quant.data_sources.massive.finalized_object_acquisition import (
    MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
    MASSIVE_FINALIZED_PROCESSING_SPEC_V0,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_readiness import (
    MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0,
    MASSIVE_FINALIZED_MINIMUM_READINESS_YEARS_V0,
    MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0,
)
from rl_quant.data_sources.massive.finalized_typed_decision_origin import (
    MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256

MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v0"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V1_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v1"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V2_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v2"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V3_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v3"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V4_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v4"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V5_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v5"
)
MASSIVE_TYPED_PIPELINE_STAGE_IDS_V0 = (
    "daily-features",
    "rolling-features",
    "pit500-decision-tensor",
    "frozen-model-inference",
    "requested-orders",
)
# Frozen from the workflow-owned stage implementation registry.  The workflow
# regression test independently requires this value to equal its live registry.
MASSIVE_TYPED_STAGE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "7a511c54752111545cb694ed22350049c80cea2bc9bf39a8795262e404b1f049"
)
# Frozen from the workflow and feature layers without importing those layers
# into the lower-level data-source namespace.  Regression tests compare each
# literal with the live implementation.
MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256_FROZEN = (
    "935f78a3514a46bd5443b4fa53f04d66a43fe7dd95128f93e094ad2658ca6f79"
)
MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V2_FROZEN = (
    "2181bbe853d437788a8822f94812cfd62052d27651c459e8a12fc39495e38700"
)
MASSIVE_HISTORICAL_READINESS_V1_SPEC_SHA256_FROZEN = (
    "ced255229a056b8b9af4809c748f1a3f1204ecc14e592f9de61a2ef6756f5d3d"
)
MASSIVE_TYPED_READINESS_CAPABILITY_V1_SPEC_SHA256_FROZEN = (
    "1ed6ff402e2f349152a30a1d98eb39c2ea099de8d2be654a590cc0bc815c1cea"
)
MASSIVE_TYPED_READINESS_PANEL_V1_SPEC_SHA256_FROZEN = (
    "c4c4c4969469710404b62df1ea619ce64c2cc5cf9a43e60fb4429ec65cc260be"
)
MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256_FROZEN = (
    "ed4ff5003cd5ba135a49481ceb1454b5e40a95e39598ff2c0f299115ecda1be0"
)
MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256_FROZEN = (
    "957f3656ec4e455d896faddb9582a12d032463e99f4f83b1beb55c44a4021fdc"
)
MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256_FROZEN = (
    "539d59b171940d0dd767bc9de205b8f53e4f980a16dfe68f0d869480b4cd1a48"
)
MASSIVE_PRODUCTION_TYPED_RUN_V3_SPEC_SHA256_FROZEN = (
    "42c46aed70ff07edeeb1c6d89afacdcb8d87f3f7e2751518c6933d65f31cc5e4"
)
MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V3_FROZEN = (
    "7a2738d8c321f5c87efb54745e1c9cdea6daa98336e45b42d46f44248fcccf76"
)
MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256_PREDECESSOR_FROZEN = (
    "89e3afd74af3e263f441fc535ac395f8a0d021e54f26cf11737e93224363694d"
)
MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256_PREDECESSOR_FROZEN = (
    "25ccff5ad3623bf9acc18f5cf5ffeb41a9078b139d5a4ceeae799ce3c819ba71"
)
MASSIVE_PRODUCTION_TYPED_RUN_V3_HARDENED_SPEC_SHA256_FROZEN = (
    "a64c7475c162e86120083abf77f162bbd44e44c1a5d9880823815d8cc1c5e668"
)
MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V3_HARDENED_FROZEN = (
    "c95eebbd60ac04460765383c9d88a9906e4144ffed3f2772396065a6b05b9e4f"
)
MASSIVE_RUNTIME_IMPLEMENTATION_SOURCE_RELATIVE_PATHS_V5 = (
    "src/rl_quant/data_sources/massive/finalized_runtime_authority.py",
    "src/rl_quant/workflows/massive_production_typed_run_v2.py",
    "src/rl_quant/data_sources/massive/finalized_typed_decision_origin.py",
    "src/rl_quant/data_sources/massive/finalized_archive_scope.py",
    "src/rl_quant/workflows/massive_historical_readiness_v1.py",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MASSIVE_RUNTIME_IMPLEMENTATION_SOURCE_INVENTORY_V5 = semantic_sha256(
    tuple(
        (
            relative_path,
            file_sha256(_REPOSITORY_ROOT / relative_path),
        )
        for relative_path in MASSIVE_RUNTIME_IMPLEMENTATION_SOURCE_RELATIVE_PATHS_V5
    )
)


class MassiveFinalizedOriginPolicyError(ValueError):
    """The immutable finalized V0 source-origin policy drifted."""


@dataclass(frozen=True, slots=True)
class MassiveFinalizedOriginPolicyV0:
    required_daily_source_roles: tuple[str, ...]
    listing_parser_spec_sha256: str
    trade_file_scan_spec_sha256: str
    trade_partition_spec_sha256: str
    participant_time_domain: str
    processing_spec_receipt_sha256: str
    processing_capability_scope: str
    maximum_source_staleness_sessions: int
    primary_estimand_source_staleness_sessions: int
    nonprimary_staleness_context_required: bool
    source_selection_rule: str
    decision_local_time: str
    fill_window: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA:
            raise MassiveFinalizedOriginPolicyError("origin policy schema drifted")
        expected: dict[str, object] = {
            "required_daily_source_roles": MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
            "listing_parser_spec_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
            "trade_file_scan_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
            "trade_partition_spec_sha256": MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
            "participant_time_domain": "[regular-open,regular-close)",
            "processing_spec_receipt_sha256": MASSIVE_FINALIZED_PROCESSING_SPEC_V0.receipt_sha256,
            "processing_capability_scope": "source-scan-and-partition-only",
            "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
            "primary_estimand_source_staleness_sessions": 1,
            "nonprimary_staleness_context_required": True,
            "source_selection_rule": "latest-prior-source-ready-by-decision",
            "decision_local_time": "12:30:00-America/New_York",
            "fill_window": "[15:50:00,16:00:00]-America/New_York",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveFinalizedOriginPolicyError(f"{name} drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginPolicyError("origin policy receipt differs")
        if self.receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256:
            raise MassiveFinalizedOriginPolicyError(
                "origin policy frozen receipt drifted"
            )


@dataclass(frozen=True, slots=True)
class MassiveFinalizedOriginPolicyV1:
    required_daily_source_roles: tuple[str, ...]
    listing_acquisition_required: bool
    listing_acquisition_schema_sha256: str
    listing_parser_spec_sha256: str
    trade_file_scan_spec_sha256: str
    trade_partition_spec_sha256: str
    participant_time_domain: str
    processing_spec_receipt_sha256: str
    readiness_capability_scope: str
    readiness_stage_ids: tuple[str, ...]
    minimum_readiness_sessions: int
    minimum_readiness_years: int
    maximum_source_staleness_sessions: int
    primary_estimand_source_staleness_sessions: int
    nonprimary_staleness_context_required: bool
    source_selection_rule: str
    decision_local_time: str
    fill_window: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ORIGIN_POLICY_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_ORIGIN_POLICY_V1_SCHEMA:
            raise MassiveFinalizedOriginPolicyError("origin policy v1 schema drifted")
        expected: dict[str, object] = {
            "required_daily_source_roles": MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
            "listing_acquisition_required": True,
            "listing_acquisition_schema_sha256": MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA_SHA256,
            "listing_parser_spec_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
            "trade_file_scan_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
            "trade_partition_spec_sha256": MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
            "participant_time_domain": "[regular-open,regular-close)",
            "processing_spec_receipt_sha256": MASSIVE_FINALIZED_PROCESSING_SPEC_V0.receipt_sha256,
            "readiness_capability_scope": "full-feature-to-order-readiness",
            "readiness_stage_ids": MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0,
            "minimum_readiness_sessions": MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0,
            "minimum_readiness_years": MASSIVE_FINALIZED_MINIMUM_READINESS_YEARS_V0,
            "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
            "primary_estimand_source_staleness_sessions": 1,
            "nonprimary_staleness_context_required": True,
            "source_selection_rule": "latest-prior-source-ready-by-decision",
            "decision_local_time": "12:30:00-America/New_York",
            "fill_window": "[15:50:00,16:00:00]-America/New_York",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveFinalizedOriginPolicyError(f"{name} drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginPolicyError("origin policy v1 receipt differs")
        if self.receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V1_RECEIPT_SHA256:
            raise MassiveFinalizedOriginPolicyError(
                "origin policy v1 frozen receipt drifted"
            )


@dataclass(frozen=True, slots=True)
class MassiveFinalizedOriginPolicyV2:
    required_daily_source_roles: tuple[str, ...]
    listing_acquisition_required: bool
    authenticated_object_get_spec_sha256: str
    archive_scope_spec_sha256: str
    trade_file_scan_spec_sha256: str
    persisted_partition_spec_sha256: str
    artifact_readiness_stage_ids: tuple[str, ...]
    typed_pipeline_stage_ids: tuple[str, ...]
    typed_stage_implementation_inventory_sha256: str
    artifact_panel_selection_spec_sha256: str
    execution_authority_source_schema_sha256: str
    participant_time_domain: str
    maximum_source_staleness_sessions: int
    source_selection_rule: str
    decision_local_time: str
    fill_window: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ORIGIN_POLICY_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_ORIGIN_POLICY_V2_SCHEMA:
            raise MassiveFinalizedOriginPolicyError("origin policy v2 schema drifted")
        expected: dict[str, object] = {
            "required_daily_source_roles": MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
            "listing_acquisition_required": True,
            "authenticated_object_get_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
            "archive_scope_spec_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SPEC_SHA256,
            "trade_file_scan_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
            "persisted_partition_spec_sha256": MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
            "artifact_readiness_stage_ids": MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1,
            "typed_pipeline_stage_ids": MASSIVE_TYPED_PIPELINE_STAGE_IDS_V0,
            "typed_stage_implementation_inventory_sha256": MASSIVE_TYPED_STAGE_IMPLEMENTATION_INVENTORY_SHA256,
            "artifact_panel_selection_spec_sha256": MASSIVE_ARTIFACT_READINESS_PANEL_SELECTION_SPEC_SHA256,
            "execution_authority_source_schema_sha256": MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256,
            "participant_time_domain": "[regular-open,regular-close)",
            "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
            "source_selection_rule": "latest-prior-source-ready-by-decision",
            "decision_local_time": "12:30:00-America/New_York",
            "fill_window": "[15:50:00,16:00:00)-America/New_York",
            "panel_materialization_authorized": False,
            "predictive_training_authorized": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveFinalizedOriginPolicyError(f"{name} drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginPolicyError("origin policy v2 receipt differs")
        if self.receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V2_RECEIPT_SHA256:
            raise MassiveFinalizedOriginPolicyError(
                "origin policy v2 frozen receipt drifted"
            )


@dataclass(frozen=True, slots=True)
class MassiveFinalizedOriginPolicyV3:
    authenticated_object_get_spec_sha256: str
    archive_scope_spec_sha256: str
    typed_decision_origin_spec_sha256: str
    production_typed_run_spec_sha256: str
    production_implementation_inventory_sha256: str
    execution_clock_spec_sha256: str
    execution_environment_spec_sha256: str
    input_availability_spec_sha256: str
    typed_readiness_panel_spec_sha256: str
    typed_readiness_capability_spec_sha256: str
    historical_readiness_spec_sha256: str
    pit500_tensor_spec_sha256: str
    inference_spec_sha256: str
    orders_spec_sha256: str
    timing_source_kind: str
    historical_readiness_rule: str
    source_selection_rule: str
    decision_local_time: str
    fill_window: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ORIGIN_POLICY_V3_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_ORIGIN_POLICY_V3_SCHEMA:
            raise MassiveFinalizedOriginPolicyError("origin policy v3 schema drifted")
        expected: dict[str, object] = {
            "authenticated_object_get_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
            "archive_scope_spec_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256,
            "typed_decision_origin_spec_sha256": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
            "production_typed_run_spec_sha256": MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256_FROZEN,
            "production_implementation_inventory_sha256": MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V2_FROZEN,
            "execution_clock_spec_sha256": MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256,
            "execution_environment_spec_sha256": MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
            "input_availability_spec_sha256": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
            "typed_readiness_panel_spec_sha256": MASSIVE_TYPED_READINESS_PANEL_V1_SPEC_SHA256_FROZEN,
            "typed_readiness_capability_spec_sha256": MASSIVE_TYPED_READINESS_CAPABILITY_V1_SPEC_SHA256_FROZEN,
            "historical_readiness_spec_sha256": MASSIVE_HISTORICAL_READINESS_V1_SPEC_SHA256_FROZEN,
            "pit500_tensor_spec_sha256": MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256_FROZEN,
            "inference_spec_sha256": MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256_FROZEN,
            "orders_spec_sha256": MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256_FROZEN,
            "timing_source_kind": "production-system-clocks-with-committed-clock-authority",
            "historical_readiness_rule": "vendor-last-modified+five-minute-safety+qualified-maximum-runtime",
            "source_selection_rule": "immediately-prior-exchange-session",
            "decision_local_time": "12:30:00-America/New_York",
            "fill_window": "[15:50:00,16:00:00)-America/New_York",
            "panel_materialization_authorized": False,
            "predictive_training_authorized": False,
            "portfolio_evaluation_authorized": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveFinalizedOriginPolicyError(f"{name} drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginPolicyError("origin policy v3 receipt differs")
        if self.receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V3_RECEIPT_SHA256:
            raise MassiveFinalizedOriginPolicyError(
                "origin policy v3 frozen receipt drifted"
            )


@dataclass(frozen=True, slots=True)
class MassiveFinalizedOriginPolicyV4:
    authenticated_object_get_spec_sha256: str
    archive_scope_spec_sha256: str
    typed_decision_origin_spec_sha256: str
    production_typed_run_spec_sha256: str
    production_implementation_inventory_sha256: str
    host_execution_spec_sha256: str
    execution_clock_spec_sha256: str
    execution_environment_spec_sha256: str
    input_availability_spec_sha256: str
    timing_source_kind: str
    historical_capability_authorized: bool
    source_selection_rule: str
    decision_local_time: str
    fill_window: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ORIGIN_POLICY_V4_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        expected: dict[str, object] = {
            "authenticated_object_get_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
            "archive_scope_spec_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256,
            "typed_decision_origin_spec_sha256": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
            "production_typed_run_spec_sha256": MASSIVE_PRODUCTION_TYPED_RUN_V3_SPEC_SHA256_FROZEN,
            "production_implementation_inventory_sha256": MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V3_FROZEN,
            "host_execution_spec_sha256": MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
            "execution_clock_spec_sha256": MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256_PREDECESSOR_FROZEN,
            "execution_environment_spec_sha256": MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256_PREDECESSOR_FROZEN,
            "input_availability_spec_sha256": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
            "timing_source_kind": "fixed-host-chrony-and-runtime-environment-capture",
            "historical_capability_authorized": False,
            "source_selection_rule": "immediately-prior-exchange-session",
            "decision_local_time": "12:30:00-America/New_York",
            "fill_window": "[15:50:00,16:00:00)-America/New_York",
            "panel_materialization_authorized": False,
            "predictive_training_authorized": False,
            "portfolio_evaluation_authorized": False,
        }
        if self.schema != MASSIVE_FINALIZED_ORIGIN_POLICY_V4_SCHEMA:
            raise MassiveFinalizedOriginPolicyError("origin policy v4 schema drifted")
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveFinalizedOriginPolicyError(f"{name} drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginPolicyError("origin policy v4 receipt differs")
        if self.receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V4_RECEIPT_SHA256:
            raise MassiveFinalizedOriginPolicyError(
                "origin policy v4 frozen receipt drifted"
            )


@dataclass(frozen=True, slots=True)
class MassiveFinalizedOriginPolicyV5:
    authenticated_object_get_spec_sha256: str
    archive_scope_spec_sha256: str
    typed_decision_origin_spec_sha256: str
    production_typed_run_spec_sha256: str
    production_implementation_inventory_sha256: str
    runtime_implementation_source_inventory_sha256: str
    host_execution_spec_sha256: str
    execution_clock_spec_sha256: str
    execution_environment_spec_sha256: str
    input_availability_spec_sha256: str
    timing_source_kind: str
    clock_interval_rule: str
    container_identity_rule: str
    imported_source_rule: str
    historical_capability_authorized: bool
    source_selection_rule: str
    decision_local_time: str
    fill_window: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ORIGIN_POLICY_V5_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        expected: dict[str, object] = {
            "authenticated_object_get_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
            "archive_scope_spec_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256,
            "typed_decision_origin_spec_sha256": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
            "production_typed_run_spec_sha256": MASSIVE_PRODUCTION_TYPED_RUN_V3_HARDENED_SPEC_SHA256_FROZEN,
            "production_implementation_inventory_sha256": MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V3_HARDENED_FROZEN,
            "runtime_implementation_source_inventory_sha256": MASSIVE_RUNTIME_IMPLEMENTATION_SOURCE_INVENTORY_V5,
            "host_execution_spec_sha256": MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
            "execution_clock_spec_sha256": MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256,
            "execution_environment_spec_sha256": MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256,
            "input_availability_spec_sha256": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
            "timing_source_kind": "fixed-host-chrony-and-runtime-environment-capture",
            "clock_interval_rule": "measurement-upper<=run-start-lower;run-finish-upper<=qualification-end-lower",
            "container_identity_rule": "fixed-read-only-runtime-metadata-cross-checked-to-proc-cgroup",
            "imported_source_rule": "executing-module-root+HEAD-blob-equality",
            "historical_capability_authorized": False,
            "source_selection_rule": "immediately-prior-exchange-session",
            "decision_local_time": "12:30:00-America/New_York",
            "fill_window": "[15:50:00,16:00:00)-America/New_York",
            "panel_materialization_authorized": False,
            "predictive_training_authorized": False,
            "portfolio_evaluation_authorized": False,
        }
        if self.schema != MASSIVE_FINALIZED_ORIGIN_POLICY_V5_SCHEMA:
            raise MassiveFinalizedOriginPolicyError("origin policy v5 schema drifted")
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveFinalizedOriginPolicyError(f"{name} drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginPolicyError("origin policy v5 receipt differs")
        if self.receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V5_RECEIPT_SHA256:
            raise MassiveFinalizedOriginPolicyError(
                "origin policy v5 frozen receipt drifted"
            )


def _build_policy() -> MassiveFinalizedOriginPolicyV0:
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA,
        "required_daily_source_roles": MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
        "listing_parser_spec_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
        "trade_file_scan_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
        "trade_partition_spec_sha256": MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
        "participant_time_domain": "[regular-open,regular-close)",
        "processing_spec_receipt_sha256": MASSIVE_FINALIZED_PROCESSING_SPEC_V0.receipt_sha256,
        "processing_capability_scope": "source-scan-and-partition-only",
        "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
        "primary_estimand_source_staleness_sessions": 1,
        "nonprimary_staleness_context_required": True,
        "source_selection_rule": "latest-prior-source-ready-by-decision",
        "decision_local_time": "12:30:00-America/New_York",
        "fill_window": "[15:50:00,16:00:00]-America/New_York",
    }
    return MassiveFinalizedOriginPolicyV0(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


def _build_policy_v1() -> MassiveFinalizedOriginPolicyV1:
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_ORIGIN_POLICY_V1_SCHEMA,
        "required_daily_source_roles": MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
        "listing_acquisition_required": True,
        "listing_acquisition_schema_sha256": MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA_SHA256,
        "listing_parser_spec_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
        "trade_file_scan_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
        "trade_partition_spec_sha256": MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
        "participant_time_domain": "[regular-open,regular-close)",
        "processing_spec_receipt_sha256": MASSIVE_FINALIZED_PROCESSING_SPEC_V0.receipt_sha256,
        "readiness_capability_scope": "full-feature-to-order-readiness",
        "readiness_stage_ids": MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0,
        "minimum_readiness_sessions": MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0,
        "minimum_readiness_years": MASSIVE_FINALIZED_MINIMUM_READINESS_YEARS_V0,
        "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
        "primary_estimand_source_staleness_sessions": 1,
        "nonprimary_staleness_context_required": True,
        "source_selection_rule": "latest-prior-source-ready-by-decision",
        "decision_local_time": "12:30:00-America/New_York",
        "fill_window": "[15:50:00,16:00:00]-America/New_York",
    }
    return MassiveFinalizedOriginPolicyV1(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


def _build_policy_v2() -> MassiveFinalizedOriginPolicyV2:
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_ORIGIN_POLICY_V2_SCHEMA,
        "required_daily_source_roles": MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
        "listing_acquisition_required": True,
        "authenticated_object_get_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
        "archive_scope_spec_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SPEC_SHA256,
        "trade_file_scan_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
        "persisted_partition_spec_sha256": MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
        "artifact_readiness_stage_ids": MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1,
        "typed_pipeline_stage_ids": MASSIVE_TYPED_PIPELINE_STAGE_IDS_V0,
        "typed_stage_implementation_inventory_sha256": MASSIVE_TYPED_STAGE_IMPLEMENTATION_INVENTORY_SHA256,
        "artifact_panel_selection_spec_sha256": MASSIVE_ARTIFACT_READINESS_PANEL_SELECTION_SPEC_SHA256,
        "execution_authority_source_schema_sha256": MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256,
        "participant_time_domain": "[regular-open,regular-close)",
        "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
        "source_selection_rule": "latest-prior-source-ready-by-decision",
        "decision_local_time": "12:30:00-America/New_York",
        "fill_window": "[15:50:00,16:00:00)-America/New_York",
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
    }
    return MassiveFinalizedOriginPolicyV2(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


def _build_policy_v3() -> MassiveFinalizedOriginPolicyV3:
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_ORIGIN_POLICY_V3_SCHEMA,
        "authenticated_object_get_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
        "archive_scope_spec_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256,
        "typed_decision_origin_spec_sha256": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
        "production_typed_run_spec_sha256": MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256_FROZEN,
        "production_implementation_inventory_sha256": MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V2_FROZEN,
        "execution_clock_spec_sha256": MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256,
        "execution_environment_spec_sha256": MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
        "input_availability_spec_sha256": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
        "typed_readiness_panel_spec_sha256": MASSIVE_TYPED_READINESS_PANEL_V1_SPEC_SHA256_FROZEN,
        "typed_readiness_capability_spec_sha256": MASSIVE_TYPED_READINESS_CAPABILITY_V1_SPEC_SHA256_FROZEN,
        "historical_readiness_spec_sha256": MASSIVE_HISTORICAL_READINESS_V1_SPEC_SHA256_FROZEN,
        "pit500_tensor_spec_sha256": MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256_FROZEN,
        "inference_spec_sha256": MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256_FROZEN,
        "orders_spec_sha256": MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256_FROZEN,
        "timing_source_kind": "production-system-clocks-with-committed-clock-authority",
        "historical_readiness_rule": "vendor-last-modified+five-minute-safety+qualified-maximum-runtime",
        "source_selection_rule": "immediately-prior-exchange-session",
        "decision_local_time": "12:30:00-America/New_York",
        "fill_window": "[15:50:00,16:00:00)-America/New_York",
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
    return MassiveFinalizedOriginPolicyV3(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


def _build_policy_v4() -> MassiveFinalizedOriginPolicyV4:
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_ORIGIN_POLICY_V4_SCHEMA,
        "authenticated_object_get_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
        "archive_scope_spec_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256,
        "typed_decision_origin_spec_sha256": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
        "production_typed_run_spec_sha256": MASSIVE_PRODUCTION_TYPED_RUN_V3_SPEC_SHA256_FROZEN,
        "production_implementation_inventory_sha256": MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V3_FROZEN,
        "host_execution_spec_sha256": MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
        "execution_clock_spec_sha256": MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256_PREDECESSOR_FROZEN,
        "execution_environment_spec_sha256": MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256_PREDECESSOR_FROZEN,
        "input_availability_spec_sha256": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
        "timing_source_kind": "fixed-host-chrony-and-runtime-environment-capture",
        "historical_capability_authorized": False,
        "source_selection_rule": "immediately-prior-exchange-session",
        "decision_local_time": "12:30:00-America/New_York",
        "fill_window": "[15:50:00,16:00:00)-America/New_York",
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
    return MassiveFinalizedOriginPolicyV4(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


def _build_policy_v5() -> MassiveFinalizedOriginPolicyV5:
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_ORIGIN_POLICY_V5_SCHEMA,
        "authenticated_object_get_spec_sha256": MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
        "archive_scope_spec_sha256": MASSIVE_FINALIZED_ARCHIVE_SCOPE_V2_SPEC_SHA256,
        "typed_decision_origin_spec_sha256": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
        "production_typed_run_spec_sha256": MASSIVE_PRODUCTION_TYPED_RUN_V3_HARDENED_SPEC_SHA256_FROZEN,
        "production_implementation_inventory_sha256": MASSIVE_PRODUCTION_TYPED_IMPLEMENTATION_INVENTORY_V3_HARDENED_FROZEN,
        "runtime_implementation_source_inventory_sha256": MASSIVE_RUNTIME_IMPLEMENTATION_SOURCE_INVENTORY_V5,
        "host_execution_spec_sha256": MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
        "execution_clock_spec_sha256": MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256,
        "execution_environment_spec_sha256": MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256,
        "input_availability_spec_sha256": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
        "timing_source_kind": "fixed-host-chrony-and-runtime-environment-capture",
        "clock_interval_rule": "measurement-upper<=run-start-lower;run-finish-upper<=qualification-end-lower",
        "container_identity_rule": "fixed-read-only-runtime-metadata-cross-checked-to-proc-cgroup",
        "imported_source_rule": "executing-module-root+HEAD-blob-equality",
        "historical_capability_authorized": False,
        "source_selection_rule": "immediately-prior-exchange-session",
        "decision_local_time": "12:30:00-America/New_York",
        "fill_window": "[15:50:00,16:00:00)-America/New_York",
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
    return MassiveFinalizedOriginPolicyV5(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


# This literal changes only with a new origin-policy generation.
MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256 = (
    "ce58430a6abea44032da299bc7b8580e0aeeb6dcbac24c6c3f9ba30c768b67a5"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V0 = _build_policy()
MASSIVE_FINALIZED_ORIGIN_POLICY_V0.validate()
# This literal changes only with a new acquired/readiness origin-policy generation.
MASSIVE_FINALIZED_ORIGIN_POLICY_V1_RECEIPT_SHA256 = (
    "95ffdd7a710dd49672cf695e3d3c304313fe23a458392175ce7875c8b3a292d1"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V1 = _build_policy_v1()
MASSIVE_FINALIZED_ORIGIN_POLICY_V1.validate()
# This literal changes only with a new typed artifact origin-policy generation.
MASSIVE_FINALIZED_ORIGIN_POLICY_V2_RECEIPT_SHA256 = (
    "cd0f11e022d39e9c38d32443f4f5737353b12d9936c77c3c46709b49d4f7dfca"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V2 = _build_policy_v2()
MASSIVE_FINALIZED_ORIGIN_POLICY_V2.validate()
# This literal changes only with the production-clock/historical-readiness generation.
MASSIVE_FINALIZED_ORIGIN_POLICY_V3_RECEIPT_SHA256 = (
    "2ff75b9c3a0aafc72f63661c75eea050ae9c98b6524096cce0a1a92ad01d9bea"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V3 = _build_policy_v3()
MASSIVE_FINALIZED_ORIGIN_POLICY_V3.validate()
# This literal changes only with the raw host/clock/runtime authority generation.
MASSIVE_FINALIZED_ORIGIN_POLICY_V4_RECEIPT_SHA256 = (
    "90a24c87e91e3488f8e534887b70f475220b80a8c493d5346504d1577ee89c6e"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V4 = _build_policy_v4()
MASSIVE_FINALIZED_ORIGIN_POLICY_V4.validate()
MASSIVE_FINALIZED_ORIGIN_POLICY_V5_RECEIPT_SHA256 = (
    "6e8655c150dbb61bf608332c6bb1892c91b2e2c78cb630c1819967501f318dc3"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V5 = _build_policy_v5()
MASSIVE_FINALIZED_ORIGIN_POLICY_V5.validate()


__all__ = [
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V0",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V1",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V1_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V1_SCHEMA",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V2",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V2_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V2_SCHEMA",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V3",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V3_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V3_SCHEMA",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V4",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V4_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V4_SCHEMA",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V5",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V5_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V5_SCHEMA",
    "MASSIVE_RUNTIME_IMPLEMENTATION_SOURCE_INVENTORY_V5",
    "MassiveFinalizedOriginPolicyError",
    "MassiveFinalizedOriginPolicyV0",
    "MassiveFinalizedOriginPolicyV1",
    "MassiveFinalizedOriginPolicyV2",
    "MassiveFinalizedOriginPolicyV3",
    "MassiveFinalizedOriginPolicyV4",
    "MassiveFinalizedOriginPolicyV5",
]
