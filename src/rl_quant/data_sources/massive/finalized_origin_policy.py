"""Immutable subordinate origin policy for finalized validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rl_quant.data_sources.massive.finalized_daily_scan import (
    MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
    MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_LISTING_ACQUISITION_SCHEMA_SHA256,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
    MASSIVE_FINALIZED_PROCESSING_SPEC_V0,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_readiness import (
    MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0,
    MASSIVE_FINALIZED_MINIMUM_READINESS_YEARS_V0,
    MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0,
)
from rl_quant.data_sources.massive.finalized_archive_scope import (
    MASSIVE_FINALIZED_ARCHIVE_SCOPE_V1_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_artifact_readiness import (
    MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256,
    MASSIVE_ARTIFACT_READINESS_PANEL_SELECTION_SPEC_SHA256,
    MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1,
)
from rl_quant.data_sources.massive.finalized_object_acquisition import (
    MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v0"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V1_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v1"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V2_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v2"
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
    "MassiveFinalizedOriginPolicyError",
    "MassiveFinalizedOriginPolicyV0",
    "MassiveFinalizedOriginPolicyV1",
    "MassiveFinalizedOriginPolicyV2",
]
