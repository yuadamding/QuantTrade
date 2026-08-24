"""Fixed typed feature-to-order pipeline for finalized validation V0.

This generation replaces unrestricted downstream callbacks with package-owned
materializers.  It remains a readiness canary and cannot authorize training or
performance reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

from rl_quant.features.massive_pit500_tensor_v0 import (
    MASSIVE_PIT500_TENSOR_V0_SOURCE_SCHEMA_SHA256,
    MASSIVE_PIT500_TENSOR_V0_SOURCE_SHA256,
    MASSIVE_PIT500_TENSOR_V0_SPEC_SHA256,
    MassivePIT500DecisionTensorV0,
    materialize_massive_pit500_tensor_v0,
    validate_massive_pit500_tensor_v0,
)
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_archive_scope import (
    MassiveFinalizedArchiveScopeV1,
)
from rl_quant.data_sources.massive.finalized_artifact_origin_authority import (
    MassiveArtifactQualifiedOriginError,
    build_massive_artifact_qualified_daily_source_v1,
)
from rl_quant.data_sources.massive.finalized_artifact_readiness import (
    MassiveArtifactReadinessCapabilityV1,
    MassiveArtifactReadinessRunV1,
)
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionManifestV0,
)
from rl_quant.data_sources.massive.finalized_object_acquisition import (
    MassiveAuthenticatedFlatFileDownloadV1,
)
from rl_quant.data_sources.massive.finalized_origin_policy import (
    MASSIVE_FINALIZED_ORIGIN_POLICY_V2,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    validate_massive_persisted_partitions_semantically_v2,
)
from rl_quant.evaluation.massive_validation_inference_v0 import (
    MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SCHEMA_SHA256,
    MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SHA256,
    MASSIVE_VALIDATION_INFERENCE_V0_SPEC_SHA256,
    MassiveValidationCheckpointV0,
    MassiveValidationInferenceArtifactV0,
    materialize_massive_validation_inference_v0,
    parse_massive_validation_checkpoint_v0,
    validate_massive_validation_inference_v0,
)
from rl_quant.evaluation.massive_validation_orders_v0 import (
    MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SCHEMA_SHA256,
    MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SHA256,
    MASSIVE_VALIDATION_ORDERS_V0_SPEC_SHA256,
    MassiveRequestedOrdersArtifactV0,
    materialize_massive_requested_orders_v0,
    validate_massive_requested_orders_v0,
)
from rl_quant.features.massive_daily_bars_v0 import (
    MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256,
    MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
    MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
    MassiveDailyBarsArtifactV0,
    materialize_massive_daily_bars_v0,
    validate_massive_daily_bars_v0,
)
from rl_quant.features.massive_daily_tape_v0 import (
    MASSIVE_DAILY_TAPE_V0_SOURCE_SCHEMA_SHA256,
    MASSIVE_DAILY_TAPE_V0_SOURCE_SHA256,
    MASSIVE_DAILY_TAPE_V0_SPEC_SHA256,
    MassiveDailyTapeArtifactV0,
    materialize_massive_daily_tape_v0,
    validate_massive_daily_tape_v0,
)
from rl_quant.features.massive_rolling_features_v0 import (
    MASSIVE_ROLLING_FEATURES_V0_SOURCE_SCHEMA_SHA256,
    MASSIVE_ROLLING_FEATURES_V0_SOURCE_SHA256,
    MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256,
    MassiveRollingFeatureArtifactV0,
    materialize_massive_rolling_features_v0,
    validate_massive_rolling_features_v0,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_TYPED_PIPELINE_STAGE_IDS_V0 = (
    "daily-features",
    "rolling-features",
    "pit500-decision-tensor",
    "frozen-model-inference",
    "requested-orders",
)
MASSIVE_TYPED_STAGE_IMPLEMENTATION_V0_SCHEMA = (
    "rl-quant.massive-typed-stage-implementation-v0"
)
MASSIVE_TYPED_PIPELINE_STAGE_V0_SCHEMA = "rl-quant.massive-typed-pipeline-stage-v0"
MASSIVE_TYPED_PIPELINE_RUN_V0_SCHEMA = "rl-quant.massive-typed-pipeline-run-v0"
MASSIVE_TYPED_ARTIFACT_QUALIFIED_DAILY_SOURCE_V2_SCHEMA = (
    "rl-quant.massive-typed-artifact-qualified-finalized-daily-source-v2"
)


class MassiveTypedPipelineV0Error(ValueError):
    """The fixed typed pipeline or its evidence chain differs."""


@dataclass(frozen=True, slots=True)
class MassiveTypedArtifactQualifiedDailySourceV2:
    source_session_date: str
    authenticated_download_receipt_sha256: str
    archive_scope_receipt_sha256: str
    mechanical_source_receipt_sha256: str
    typed_pipeline_run_receipt_sha256: str
    daily_bars_receipt_sha256: str
    daily_tape_receipt_sha256: str
    rolling_features_receipt_sha256: str
    decision_tensor_receipt_sha256: str
    inference_receipt_sha256: str
    requested_orders_receipt_sha256: str
    origin_policy_receipt_sha256: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_TYPED_ARTIFACT_QUALIFIED_DAILY_SOURCE_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TYPED_ARTIFACT_QUALIFIED_DAILY_SOURCE_V2_SCHEMA:
            raise MassiveArtifactQualifiedOriginError(
                "typed artifact-qualified source schema drifted"
            )
        for name in (
            "authenticated_download_receipt_sha256",
            "archive_scope_receipt_sha256",
            "mechanical_source_receipt_sha256",
            "typed_pipeline_run_receipt_sha256",
            "daily_bars_receipt_sha256",
            "daily_tape_receipt_sha256",
            "rolling_features_receipt_sha256",
            "decision_tensor_receipt_sha256",
            "inference_receipt_sha256",
            "requested_orders_receipt_sha256",
            "origin_policy_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.origin_policy_receipt_sha256
            != MASSIVE_FINALIZED_ORIGIN_POLICY_V2.receipt_sha256
        ):
            raise MassiveArtifactQualifiedOriginError(
                "typed artifact-qualified origin policy drifted"
            )
        if (
            self.panel_materialization_authorized
            or self.predictive_training_authorized
            or self.portfolio_evaluation_authorized
        ):
            raise MassiveArtifactQualifiedOriginError(
                "typed readiness cannot authorize performance work"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveArtifactQualifiedOriginError(
                "typed artifact-qualified source receipt differs"
            )


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveTypedPipelineV0Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveTypedStageImplementationV0:
    stage_id: str
    module_source_sha256: str
    input_schema_receipts: tuple[str, ...]
    output_schema_receipts: tuple[str, ...]
    configuration_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_TYPED_STAGE_IMPLEMENTATION_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_TYPED_STAGE_IMPLEMENTATION_V0_SCHEMA
            or self.stage_id not in MASSIVE_TYPED_PIPELINE_STAGE_IDS_V0
            or not self.input_schema_receipts
            or not self.output_schema_receipts
        ):
            raise MassiveTypedPipelineV0Error("typed stage implementation drifted")
        for value in (
            self.module_source_sha256,
            self.configuration_receipt_sha256,
            self.receipt_sha256,
            *self.input_schema_receipts,
            *self.output_schema_receipts,
        ):
            _digest("typed stage implementation receipt", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTypedPipelineV0Error(
                "typed stage implementation receipt differs"
            )


def _implementation(
    stage_id: str,
    module_source: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    configuration: str,
) -> MassiveTypedStageImplementationV0:
    body = {
        "schema": MASSIVE_TYPED_STAGE_IMPLEMENTATION_V0_SCHEMA,
        "stage_id": stage_id,
        "module_source_sha256": module_source,
        "input_schema_receipts": inputs,
        "output_schema_receipts": outputs,
        "configuration_receipt_sha256": configuration,
    }
    result = MassiveTypedStageImplementationV0(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


MASSIVE_TYPED_STAGE_IMPLEMENTATIONS_V0 = (
    _implementation(
        "daily-features",
        semantic_sha256(
            (MASSIVE_DAILY_BARS_V0_SOURCE_SHA256, MASSIVE_DAILY_TAPE_V0_SOURCE_SHA256)
        ),
        (semantic_sha256("persisted-partition-manifest-v1"),),
        (
            MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256,
            MASSIVE_DAILY_TAPE_V0_SOURCE_SCHEMA_SHA256,
        ),
        semantic_sha256(
            (MASSIVE_DAILY_BARS_V0_SPEC_SHA256, MASSIVE_DAILY_TAPE_V0_SPEC_SHA256)
        ),
    ),
    _implementation(
        "rolling-features",
        MASSIVE_ROLLING_FEATURES_V0_SOURCE_SHA256,
        (
            MASSIVE_DAILY_BARS_V0_SOURCE_SCHEMA_SHA256,
            MASSIVE_DAILY_TAPE_V0_SOURCE_SCHEMA_SHA256,
        ),
        (MASSIVE_ROLLING_FEATURES_V0_SOURCE_SCHEMA_SHA256,),
        MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256,
    ),
    _implementation(
        "pit500-decision-tensor",
        MASSIVE_PIT500_TENSOR_V0_SOURCE_SHA256,
        (MASSIVE_ROLLING_FEATURES_V0_SOURCE_SCHEMA_SHA256,),
        (MASSIVE_PIT500_TENSOR_V0_SOURCE_SCHEMA_SHA256,),
        MASSIVE_PIT500_TENSOR_V0_SPEC_SHA256,
    ),
    _implementation(
        "frozen-model-inference",
        MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SHA256,
        (MASSIVE_PIT500_TENSOR_V0_SOURCE_SCHEMA_SHA256,),
        (MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SCHEMA_SHA256,),
        MASSIVE_VALIDATION_INFERENCE_V0_SPEC_SHA256,
    ),
    _implementation(
        "requested-orders",
        MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SHA256,
        (MASSIVE_VALIDATION_INFERENCE_V0_SOURCE_SCHEMA_SHA256,),
        (MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SCHEMA_SHA256,),
        MASSIVE_VALIDATION_ORDERS_V0_SPEC_SHA256,
    ),
)
MASSIVE_TYPED_STAGE_IMPLEMENTATION_INVENTORY_SHA256 = semantic_sha256(
    tuple(row.receipt_sha256 for row in MASSIVE_TYPED_STAGE_IMPLEMENTATIONS_V0)
)


@dataclass(frozen=True, slots=True)
class MassiveTypedPipelineStageV0:
    stage_id: str
    implementation_receipt_sha256: str
    input_artifact_receipts: tuple[str, ...]
    output_artifact_receipts: tuple[str, ...]
    started_at_ms: int
    finished_at_ms: int
    started_monotonic_ns: int
    finished_monotonic_ns: int
    receipt_sha256: str
    schema: str = MASSIVE_TYPED_PIPELINE_STAGE_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_TYPED_PIPELINE_STAGE_V0_SCHEMA
            or self.stage_id not in MASSIVE_TYPED_PIPELINE_STAGE_IDS_V0
            or not self.input_artifact_receipts
            or not self.output_artifact_receipts
            or self.started_at_ms > self.finished_at_ms
            or self.started_monotonic_ns >= self.finished_monotonic_ns
        ):
            raise MassiveTypedPipelineV0Error("typed stage chronology differs")
        for value in (
            self.implementation_receipt_sha256,
            self.receipt_sha256,
            *self.input_artifact_receipts,
            *self.output_artifact_receipts,
        ):
            _digest("typed stage receipt", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTypedPipelineV0Error("typed stage receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveTypedPipelineRunV0:
    source_session_date: str
    persisted_partition_manifest_receipt_sha256: str
    daily_bars: MassiveDailyBarsArtifactV0
    daily_tape: MassiveDailyTapeArtifactV0
    rolling_features: MassiveRollingFeatureArtifactV0
    decision_tensor: MassivePIT500DecisionTensorV0
    inference: MassiveValidationInferenceArtifactV0
    requested_orders: MassiveRequestedOrdersArtifactV0
    stage_implementations: tuple[MassiveTypedStageImplementationV0, ...]
    stages: tuple[MassiveTypedPipelineStageV0, ...]
    protocol_receipt_sha256: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_TYPED_PIPELINE_RUN_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TYPED_PIPELINE_RUN_V0_SCHEMA:
            raise MassiveTypedPipelineV0Error("typed pipeline schema drifted")
        self.daily_bars.validate()
        self.daily_tape.validate()
        self.rolling_features.validate()
        self.decision_tensor.validate()
        self.inference.validate()
        self.requested_orders.validate()
        for implementation in self.stage_implementations:
            implementation.validate()
        if self.stage_implementations != MASSIVE_TYPED_STAGE_IMPLEMENTATIONS_V0:
            raise MassiveTypedPipelineV0Error("typed implementation registry differs")
        if (
            tuple(stage.stage_id for stage in self.stages)
            != MASSIVE_TYPED_PIPELINE_STAGE_IDS_V0
        ):
            raise MassiveTypedPipelineV0Error("typed pipeline stage inventory differs")
        for stage, implementation in zip(
            self.stages, self.stage_implementations, strict=True
        ):
            stage.validate()
            if stage.implementation_receipt_sha256 != implementation.receipt_sha256:
                raise MassiveTypedPipelineV0Error("typed stage implementation differs")
        if (
            self.daily_bars.persisted_partition_manifest_receipt_sha256
            != self.persisted_partition_manifest_receipt_sha256
            or self.daily_tape.persisted_partition_manifest_receipt_sha256
            != self.persisted_partition_manifest_receipt_sha256
            or self.rolling_features.source_session_date != self.source_session_date
            or self.decision_tensor.rolling_feature_artifact_receipt_sha256
            != self.rolling_features.receipt_sha256
            or self.inference.tensor_receipt_sha256
            != self.decision_tensor.receipt_sha256
            or self.requested_orders.inference_receipt_sha256
            != self.inference.receipt_sha256
        ):
            raise MassiveTypedPipelineV0Error("typed pipeline authority chain differs")
        if (
            self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256
        ):
            raise MassiveTypedPipelineV0Error("typed pipeline protocol differs")
        if (
            self.panel_materialization_authorized
            or self.predictive_training_authorized
            or self.portfolio_evaluation_authorized
        ):
            raise MassiveTypedPipelineV0Error(
                "typed readiness cannot authorize performance work"
            )
        _digest("typed pipeline receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveTypedPipelineV0Error("typed pipeline receipt differs")


def _stage(
    *,
    stage_id: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    started_ms: int,
    finished_ms: int,
    started_ns: int,
    finished_ns: int,
) -> MassiveTypedPipelineStageV0:
    implementation = next(
        row
        for row in MASSIVE_TYPED_STAGE_IMPLEMENTATIONS_V0
        if row.stage_id == stage_id
    )
    body = {
        "schema": MASSIVE_TYPED_PIPELINE_STAGE_V0_SCHEMA,
        "stage_id": stage_id,
        "implementation_receipt_sha256": implementation.receipt_sha256,
        "input_artifact_receipts": inputs,
        "output_artifact_receipts": outputs,
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
    }
    result = MassiveTypedPipelineStageV0(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


def materialize_massive_finalized_typed_pipeline_v0(
    *,
    persisted_root: str | Path,
    output_root: str | Path,
    checkpoint_root: str | Path,
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    semantic_partition_manifest: MassiveDailyTradePartitionManifestV0,
    persisted_partition_manifest: MassivePersistedPartitionManifestV1,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    prior_daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    prior_daily_tape: Sequence[MassiveDailyTapeArtifactV0],
    checkpoint: MassiveValidationCheckpointV0,
    decision_session_date: str,
    decision_at_ms: int,
    source_staleness_sessions: int,
    entitlement_receipt_sha256: str,
    wall_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> MassiveTypedPipelineRunV0:
    """Run the fixed feature-to-order materializers and bind actual artifacts."""

    condition_authority.validate()
    if (
        semantic_partition_manifest.condition_authority_receipt_sha256
        != condition_authority.receipt_sha256
    ):
        raise MassiveTypedPipelineV0Error(
            "typed feature condition authority differs from source partitions"
        )
    validate_massive_persisted_partitions_semantically_v2(
        root=persisted_root,
        manifest=persisted_partition_manifest,
        scan_evidence=scan_evidence,
        semantic_partition_manifest=semantic_partition_manifest,
        identity_authority=identity_authority,
        correction_authority=correction_authority,
    )
    checkpoint.validate()
    if checkpoint != parse_massive_validation_checkpoint_v0(
        root=checkpoint_root, loaded_source=checkpoint.loaded_source
    ):
        raise MassiveTypedPipelineV0Error(
            "typed pipeline checkpoint was not parsed from committed bytes"
        )
    Path(output_root).mkdir(parents=True, exist_ok=True)
    stages = []

    started_ms, started_ns = wall_ms(), monotonic_ns()
    daily_bars = materialize_massive_daily_bars_v0(
        persisted_root=persisted_root,
        output_root=output_root,
        manifest=persisted_partition_manifest,
        condition_authority=condition_authority,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=started_ms,
    )
    daily_tape = materialize_massive_daily_tape_v0(
        persisted_root=persisted_root,
        output_root=output_root,
        manifest=persisted_partition_manifest,
        condition_authority=condition_authority,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=started_ms,
    )
    finished_ns, finished_ms = monotonic_ns(), wall_ms()
    stages.append(
        _stage(
            stage_id="daily-features",
            inputs=(persisted_partition_manifest.receipt_sha256,),
            outputs=(daily_bars.receipt_sha256, daily_tape.receipt_sha256),
            started_ms=started_ms,
            finished_ms=finished_ms,
            started_ns=started_ns,
            finished_ns=finished_ns,
        )
    )

    started_ms, started_ns = wall_ms(), monotonic_ns()
    rolling = materialize_massive_rolling_features_v0(
        daily_feature_root=output_root,
        output_root=output_root,
        bars_artifacts=tuple(prior_daily_bars) + (daily_bars,),
        tape_artifacts=tuple(prior_daily_tape) + (daily_tape,),
        identity_authority=identity_authority,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=started_ms,
    )
    finished_ns, finished_ms = monotonic_ns(), wall_ms()
    stages.append(
        _stage(
            stage_id="rolling-features",
            inputs=(daily_bars.receipt_sha256, daily_tape.receipt_sha256),
            outputs=(rolling.receipt_sha256,),
            started_ms=started_ms,
            finished_ms=finished_ms,
            started_ns=started_ns,
            finished_ns=finished_ns,
        )
    )

    started_ms, started_ns = wall_ms(), monotonic_ns()
    tensor = materialize_massive_pit500_tensor_v0(
        rolling_root=output_root,
        output_root=output_root,
        rolling=rolling,
        identity_authority=identity_authority,
        decision_session_date=decision_session_date,
        decision_at_ms=decision_at_ms,
        source_staleness_sessions=source_staleness_sessions,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=started_ms,
    )
    finished_ns, finished_ms = monotonic_ns(), wall_ms()
    stages.append(
        _stage(
            stage_id="pit500-decision-tensor",
            inputs=(rolling.receipt_sha256,),
            outputs=(tensor.receipt_sha256,),
            started_ms=started_ms,
            finished_ms=finished_ms,
            started_ns=started_ns,
            finished_ns=finished_ns,
        )
    )

    started_ms, started_ns = wall_ms(), monotonic_ns()
    inference = materialize_massive_validation_inference_v0(
        tensor_root=output_root,
        output_root=output_root,
        tensor=tensor,
        checkpoint=checkpoint,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=started_ms,
    )
    finished_ns, finished_ms = monotonic_ns(), wall_ms()
    stages.append(
        _stage(
            stage_id="frozen-model-inference",
            inputs=(tensor.receipt_sha256, checkpoint.receipt_sha256),
            outputs=(inference.receipt_sha256,),
            started_ms=started_ms,
            finished_ms=finished_ms,
            started_ns=started_ns,
            finished_ns=finished_ns,
        )
    )

    started_ms, started_ns = wall_ms(), monotonic_ns()
    orders = materialize_massive_requested_orders_v0(
        tensor_root=output_root,
        inference_root=output_root,
        output_root=output_root,
        tensor=tensor,
        inference=inference,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=started_ms,
    )
    finished_ns, finished_ms = monotonic_ns(), wall_ms()
    stages.append(
        _stage(
            stage_id="requested-orders",
            inputs=(inference.receipt_sha256,),
            outputs=(orders.receipt_sha256,),
            started_ms=started_ms,
            finished_ms=finished_ms,
            started_ns=started_ns,
            finished_ns=finished_ns,
        )
    )

    body = {
        "schema": MASSIVE_TYPED_PIPELINE_RUN_V0_SCHEMA,
        "source_session_date": persisted_partition_manifest.source_session_date,
        "persisted_partition_manifest_receipt_sha256": persisted_partition_manifest.receipt_sha256,
        "daily_bars": daily_bars,
        "daily_tape": daily_tape,
        "rolling_features": rolling,
        "decision_tensor": tensor,
        "inference": inference,
        "requested_orders": orders,
        "stage_implementations": MASSIVE_TYPED_STAGE_IMPLEMENTATIONS_V0,
        "stages": tuple(stages),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
    provisional = MassiveTypedPipelineRunV0(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveTypedPipelineRunV0(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    validate_massive_finalized_typed_pipeline_v0(output_root=output_root, result=result)
    return result


def validate_massive_finalized_typed_pipeline_v0(
    *, output_root: str | Path, result: MassiveTypedPipelineRunV0
) -> None:
    result.validate()
    validate_massive_daily_bars_v0(root=output_root, artifact=result.daily_bars)
    validate_massive_daily_tape_v0(root=output_root, artifact=result.daily_tape)
    validate_massive_rolling_features_v0(
        root=output_root, artifact=result.rolling_features
    )
    validate_massive_pit500_tensor_v0(root=output_root, tensor=result.decision_tensor)
    validate_massive_validation_inference_v0(
        root=output_root, artifact=result.inference
    )
    validate_massive_requested_orders_v0(
        root=output_root, artifact=result.requested_orders
    )


def build_massive_typed_artifact_qualified_daily_source_v2(
    *,
    listing_root: str | Path,
    persisted_partition_root: str | Path,
    execution_authority_root: str | Path,
    stage_roots: Mapping[str, str | Path],
    typed_output_root: str | Path,
    readiness_run: MassiveArtifactReadinessRunV1,
    readiness_capability: MassiveArtifactReadinessCapabilityV1,
    authenticated_download: MassiveAuthenticatedFlatFileDownloadV1,
    archive_scope: MassiveFinalizedArchiveScopeV1,
    typed_pipeline_run: MassiveTypedPipelineRunV0,
) -> MassiveTypedArtifactQualifiedDailySourceV2:
    """Bind actual typed outputs while retaining every performance block."""

    mechanical = build_massive_artifact_qualified_daily_source_v1(
        listing_root=listing_root,
        persisted_partition_root=persisted_partition_root,
        execution_authority_root=execution_authority_root,
        stage_roots=stage_roots,
        readiness_run=readiness_run,
        readiness_capability=readiness_capability,
    )
    authenticated_download.validate()
    archive_scope.validate()
    validate_massive_finalized_typed_pipeline_v0(
        output_root=typed_output_root, result=typed_pipeline_run
    )
    source_key = readiness_run.loaded_source.receipt.source_object_key
    if (
        not archive_scope.qualification_complete
        or source_key not in archive_scope.expected_source_object_keys
        or authenticated_download.loaded_source != readiness_run.loaded_source
        or authenticated_download.source_object_key != source_key
        or typed_pipeline_run.source_session_date
        != readiness_run.source_session.session_date
        or typed_pipeline_run.persisted_partition_manifest_receipt_sha256
        != readiness_run.persisted_partition_manifest.receipt_sha256
    ):
        raise MassiveArtifactQualifiedOriginError(
            "typed artifact-qualified authorities differ"
        )
    body = {
        "schema": MASSIVE_TYPED_ARTIFACT_QUALIFIED_DAILY_SOURCE_V2_SCHEMA,
        "source_session_date": readiness_run.source_session.session_date,
        "authenticated_download_receipt_sha256": authenticated_download.receipt_sha256,
        "archive_scope_receipt_sha256": archive_scope.receipt_sha256,
        "mechanical_source_receipt_sha256": mechanical.receipt_sha256,
        "typed_pipeline_run_receipt_sha256": typed_pipeline_run.receipt_sha256,
        "daily_bars_receipt_sha256": typed_pipeline_run.daily_bars.receipt_sha256,
        "daily_tape_receipt_sha256": typed_pipeline_run.daily_tape.receipt_sha256,
        "rolling_features_receipt_sha256": typed_pipeline_run.rolling_features.receipt_sha256,
        "decision_tensor_receipt_sha256": typed_pipeline_run.decision_tensor.receipt_sha256,
        "inference_receipt_sha256": typed_pipeline_run.inference.receipt_sha256,
        "requested_orders_receipt_sha256": typed_pipeline_run.requested_orders.receipt_sha256,
        "origin_policy_receipt_sha256": MASSIVE_FINALIZED_ORIGIN_POLICY_V2.receipt_sha256,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
    result = MassiveTypedArtifactQualifiedDailySourceV2(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_TYPED_ARTIFACT_QUALIFIED_DAILY_SOURCE_V2_SCHEMA",
    "MASSIVE_TYPED_PIPELINE_STAGE_IDS_V0",
    "MASSIVE_TYPED_STAGE_IMPLEMENTATIONS_V0",
    "MASSIVE_TYPED_STAGE_IMPLEMENTATION_INVENTORY_SHA256",
    "MassiveTypedPipelineRunV0",
    "MassiveTypedPipelineStageV0",
    "MassiveTypedPipelineV0Error",
    "MassiveTypedStageImplementationV0",
    "MassiveTypedArtifactQualifiedDailySourceV2",
    "build_massive_typed_artifact_qualified_daily_source_v2",
    "materialize_massive_finalized_typed_pipeline_v0",
    "validate_massive_finalized_typed_pipeline_v0",
]
