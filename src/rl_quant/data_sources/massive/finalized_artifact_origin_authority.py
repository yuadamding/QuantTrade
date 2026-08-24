"""Artifact-derived finalized daily-source authority.

This generation is intentionally separate from the immutable development V0
source authority.  It accepts only a measured artifact-readiness run and its
deterministically selected capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from rl_quant.data_sources.massive.finalized_artifact_readiness import (
    MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1,
    MassiveArtifactReadinessCapabilityV1,
    MassiveArtifactReadinessRunV1,
    parse_massive_artifact_execution_authority_v1,
    validate_massive_artifact_readiness_stage_bytes_v1,
)
from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FINALIZED_V0_SOURCE_ROLE,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    validate_massive_captured_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_PROCESSING_SPEC_V0,
    build_massive_vendor_object_metadata_from_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin_policy import (
    MASSIVE_FINALIZED_ORIGIN_POLICY_V1,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    validate_massive_persisted_partitions_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_ARTIFACT_QUALIFIED_DAILY_SOURCE_V1_SCHEMA = (
    "rl-quant.massive-artifact-qualified-finalized-daily-source-v1"
)


class MassiveArtifactQualifiedOriginError(ValueError):
    """An artifact-derived finalized daily source cannot be established."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveArtifactQualifiedOriginError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveArtifactQualifiedDailySourceV1:
    source_role: str
    source_session_date: str
    source_object_receipt_sha256: str
    source_commit_receipt_sha256: str
    listing_acquisition_receipt_sha256: str
    listing_entry_receipt_sha256: str
    vendor_last_modified_at_ms: int
    artifact_readiness_run_receipt_sha256: str
    artifact_readiness_capability_receipt_sha256: str
    whole_file_scan_receipt_sha256: str
    semantic_partition_manifest_receipt_sha256: str
    persisted_partition_manifest_receipt_sha256: str
    identity_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    session_authority_receipt_sha256: str
    origin_policy_receipt_sha256: str
    publication_safety_margin_ms: int
    maximum_measured_runtime_ms: int
    measured_order_ready_upper_bound_at_ms: int
    panel_materialization_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_ARTIFACT_QUALIFIED_DAILY_SOURCE_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ARTIFACT_QUALIFIED_DAILY_SOURCE_V1_SCHEMA:
            raise MassiveArtifactQualifiedOriginError(
                "artifact-qualified source schema drifted"
            )
        if self.source_role != MASSIVE_FINALIZED_V0_SOURCE_ROLE:
            raise MassiveArtifactQualifiedOriginError(
                "artifact-qualified source role drifted"
            )
        for name in (
            "source_object_receipt_sha256",
            "source_commit_receipt_sha256",
            "listing_acquisition_receipt_sha256",
            "listing_entry_receipt_sha256",
            "artifact_readiness_run_receipt_sha256",
            "artifact_readiness_capability_receipt_sha256",
            "whole_file_scan_receipt_sha256",
            "semantic_partition_manifest_receipt_sha256",
            "persisted_partition_manifest_receipt_sha256",
            "identity_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "session_authority_receipt_sha256",
            "origin_policy_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.publication_safety_margin_ms
            != MASSIVE_FINALIZED_PROCESSING_SPEC_V0.publication_safety_margin_ms
            or self.maximum_measured_runtime_ms <= 0
            or self.measured_order_ready_upper_bound_at_ms
            != self.vendor_last_modified_at_ms
            + self.publication_safety_margin_ms
            + self.maximum_measured_runtime_ms
        ):
            raise MassiveArtifactQualifiedOriginError(
                "artifact-qualified readiness bound differs"
            )
        if (
            self.origin_policy_receipt_sha256
            != MASSIVE_FINALIZED_ORIGIN_POLICY_V1.receipt_sha256
        ):
            raise MassiveArtifactQualifiedOriginError(
                "artifact-qualified origin policy drifted"
            )
        if self.panel_materialization_authorized is not False:
            raise MassiveArtifactQualifiedOriginError(
                "artifact readiness cannot authorize panel materialization"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveArtifactQualifiedOriginError(
                "artifact-qualified source receipt differs"
            )


def build_massive_artifact_qualified_daily_source_v1(
    *,
    listing_root: str | Path,
    persisted_partition_root: str | Path,
    execution_authority_root: str | Path,
    stage_roots: Mapping[str, str | Path],
    readiness_run: MassiveArtifactReadinessRunV1,
    readiness_capability: MassiveArtifactReadinessCapabilityV1,
) -> MassiveArtifactQualifiedDailySourceV1:
    """Reopen every stage and issue a non-training artifact source receipt."""

    readiness_run.validate()
    readiness_capability.validate()
    if (
        readiness_run.execution_authority
        != parse_massive_artifact_execution_authority_v1(
            root=execution_authority_root,
            loaded_source=readiness_run.execution_authority.loaded_source,
        )
    ):
        raise MassiveArtifactQualifiedOriginError(
            "execution authority differs from committed bytes"
        )
    if not readiness_capability.covers(readiness_run):
        raise MassiveArtifactQualifiedOriginError(
            "artifact readiness capability does not cover the source run"
        )
    if set(stage_roots) != set(MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1):
        raise MassiveArtifactQualifiedOriginError("artifact stage roots are incomplete")
    validate_massive_captured_flat_file_listing_v0(
        root=listing_root, captured_listing=readiness_run.captured_listing
    )
    validate_massive_persisted_partitions_v1(
        root=persisted_partition_root,
        manifest=readiness_run.persisted_partition_manifest,
    )
    for stage in readiness_run.stages:
        validate_massive_artifact_readiness_stage_bytes_v1(
            root=stage_roots[stage.stage_id],
            stage=stage,
            source_session_date=readiness_run.source_session.session_date,
        )
    listing = readiness_run.captured_listing.committed_listing
    entry = listing.resolve(
        source_object_key=readiness_run.loaded_source.receipt.source_object_key
    )
    metadata = build_massive_vendor_object_metadata_from_listing_v0(
        committed_listing=listing,
        listing_entry=entry,
        loaded_source=readiness_run.loaded_source,
    )
    body: dict[str, object] = {
        "schema": MASSIVE_ARTIFACT_QUALIFIED_DAILY_SOURCE_V1_SCHEMA,
        "source_role": MASSIVE_FINALIZED_V0_SOURCE_ROLE,
        "source_session_date": readiness_run.source_session.session_date,
        "source_object_receipt_sha256": readiness_run.loaded_source.receipt.receipt_sha256,
        "source_commit_receipt_sha256": readiness_run.loaded_source.commit.receipt_sha256,
        "listing_acquisition_receipt_sha256": readiness_run.captured_listing.acquisition_evidence.receipt_sha256,
        "listing_entry_receipt_sha256": entry.receipt_sha256,
        "vendor_last_modified_at_ms": metadata.vendor_last_modified_at_ms,
        "artifact_readiness_run_receipt_sha256": readiness_run.receipt_sha256,
        "artifact_readiness_capability_receipt_sha256": readiness_capability.receipt_sha256,
        "whole_file_scan_receipt_sha256": readiness_run.scan_evidence.receipt_sha256,
        "semantic_partition_manifest_receipt_sha256": readiness_run.semantic_partition_manifest.receipt_sha256,
        "persisted_partition_manifest_receipt_sha256": readiness_run.persisted_partition_manifest.receipt_sha256,
        "identity_authority_receipt_sha256": readiness_run.identity_authority_receipt_sha256,
        "condition_authority_receipt_sha256": readiness_run.condition_authority_receipt_sha256,
        "correction_authority_receipt_sha256": readiness_run.correction_authority_receipt_sha256,
        "session_authority_receipt_sha256": readiness_run.session_authority_receipt_sha256,
        "origin_policy_receipt_sha256": MASSIVE_FINALIZED_ORIGIN_POLICY_V1.receipt_sha256,
        "publication_safety_margin_ms": MASSIVE_FINALIZED_PROCESSING_SPEC_V0.publication_safety_margin_ms,
        "maximum_measured_runtime_ms": readiness_capability.maximum_runtime_ms,
        "measured_order_ready_upper_bound_at_ms": (
            metadata.vendor_last_modified_at_ms
            + MASSIVE_FINALIZED_PROCESSING_SPEC_V0.publication_safety_margin_ms
            + readiness_capability.maximum_runtime_ms
        ),
        "panel_materialization_authorized": False,
    }
    provisional = MassiveArtifactQualifiedDailySourceV1(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveArtifactQualifiedDailySourceV1(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ARTIFACT_QUALIFIED_DAILY_SOURCE_V1_SCHEMA",
    "MassiveArtifactQualifiedDailySourceV1",
    "MassiveArtifactQualifiedOriginError",
    "build_massive_artifact_qualified_daily_source_v1",
]
