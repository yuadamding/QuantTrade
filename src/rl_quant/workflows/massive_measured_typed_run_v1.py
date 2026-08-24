"""One measured authenticated source-to-order typed run for validation V1.

This workflow is deliberately fixed and package-owned, but its injectable
clocks make it development evidence only.  Production timing qualification is
issued exclusively by the V2 production wrapper.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_archive_scope import (
    MassiveFinalizedArchiveScopeV2,
)
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256,
    MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
    MassiveDailyTradeFileScanEvidenceV0,
)
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MassiveCapturedFlatFileListingV0,
)
from rl_quant.data_sources.massive.finalized_object_acquisition import (
    MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SOURCE_SHA256,
    MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
    MassiveAuthenticatedFlatFileDownloadV1,
    download_massive_flat_file_object_v1,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256,
    MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
    MassiveDailyTradePartitionManifestV0,
    MassiveFinalizedFeatureDomainSpecV0,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MASSIVE_PERSISTED_PARTITION_SOURCE_SHA256,
    MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
    MassivePersistedPartitionManifestV1,
    stream_and_persist_massive_daily_trade_partitions_v1,
    validate_massive_persisted_partitions_semantically_v2,
)
from rl_quant.data_sources.massive.finalized_typed_decision_origin import (
    MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
    MassiveTypedDecisionOriginV1,
    build_massive_typed_decision_origin_v1,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.evaluation.massive_validation_inference_v1 import (
    MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SHA256,
    MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256,
    MassiveValidationCheckpointV1,
    MassiveValidationInferenceArtifactV1,
    materialize_massive_validation_inference_v1,
    validate_massive_validation_inference_v1,
)
from rl_quant.evaluation.massive_validation_orders_v1 import (
    MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SHA256,
    MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256,
    MassiveRequestedOrdersArtifactV1,
    materialize_massive_requested_orders_v1,
    validate_massive_requested_orders_v1,
)
from rl_quant.features.massive_daily_bars_v0 import (
    MASSIVE_DAILY_BARS_V0_SOURCE_SHA256,
    MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
    MassiveDailyBarsArtifactV0,
    materialize_massive_daily_bars_v0,
    validate_massive_daily_bars_v0,
)
from rl_quant.features.massive_daily_tape_v0 import (
    MASSIVE_DAILY_TAPE_V0_SOURCE_SHA256,
    MASSIVE_DAILY_TAPE_V0_SPEC_SHA256,
    MassiveDailyTapeArtifactV0,
    materialize_massive_daily_tape_v0,
    validate_massive_daily_tape_v0,
)
from rl_quant.features.massive_pit500_tensor_v1 import (
    MASSIVE_PIT500_TENSOR_V1_SOURCE_SHA256,
    MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256,
    MassivePIT500DecisionTensorV1,
    materialize_massive_pit500_tensor_v1,
    validate_massive_pit500_tensor_v1,
)
from rl_quant.features.massive_rolling_features_v0 import (
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

MASSIVE_MEASURED_TYPED_STAGE_IDS_V1 = (
    "authenticated-object-get",
    "source-scan-route-replay-persist",
    "daily-features",
    "rolling-features",
    "pit500-decision-tensor",
    "frozen-model-inference",
    "requested-orders",
)
MASSIVE_MEASURED_TYPED_STAGE_IMPLEMENTATIONS_V1 = (
    MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SOURCE_SHA256,
    semantic_sha256(
        (
            MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256,
            MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256,
            MASSIVE_PERSISTED_PARTITION_SOURCE_SHA256,
        )
    ),
    semantic_sha256(
        (MASSIVE_DAILY_BARS_V0_SOURCE_SHA256, MASSIVE_DAILY_TAPE_V0_SOURCE_SHA256)
    ),
    MASSIVE_ROLLING_FEATURES_V0_SOURCE_SHA256,
    MASSIVE_PIT500_TENSOR_V1_SOURCE_SHA256,
    MASSIVE_VALIDATION_INFERENCE_V1_SOURCE_SHA256,
    MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SHA256,
)
MASSIVE_MEASURED_TYPED_STAGE_CONFIGURATION_V1 = (
    MASSIVE_AUTHENTICATED_OBJECT_GET_V1_SPEC_SHA256,
    semantic_sha256(
        (
            MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
            MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
            MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
        )
    ),
    semantic_sha256(
        (MASSIVE_DAILY_BARS_V0_SPEC_SHA256, MASSIVE_DAILY_TAPE_V0_SPEC_SHA256)
    ),
    MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256,
    MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256,
    MASSIVE_VALIDATION_INFERENCE_V1_SPEC_SHA256,
    MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256,
)
MASSIVE_MEASURED_TYPED_RUN_V1_SCHEMA = "rl-quant.massive-measured-typed-run-v1"
MASSIVE_MEASURED_TYPED_RUN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "stage_ids": MASSIVE_MEASURED_TYPED_STAGE_IDS_V1,
        "stage_implementations": MASSIVE_MEASURED_TYPED_STAGE_IMPLEMENTATIONS_V1,
        "stage_configurations": MASSIVE_MEASURED_TYPED_STAGE_CONFIGURATION_V1,
        "outer_timer": "before-authenticated-get-through-requested-orders",
        "maximum_runtime_ms": 55 * 60 * 1_000,
        "decision_origin_spec": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
        "timing_source": "injectable-development-clock",
        "timing_qualification": False,
        "output_commits": "created-and-verified-inside-stage-wall-interval",
        "performance_authorization": False,
    }
)


class MassiveMeasuredTypedRunV1Error(ValueError):
    """The measured typed run is not its declared source-to-order workflow."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveMeasuredTypedRunV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveMeasuredTypedStageV1:
    stage_id: str
    implementation_source_sha256: str
    configuration_receipt_sha256: str
    input_artifact_receipts: tuple[str, ...]
    output_artifact_receipts: tuple[str, ...]
    output_commit_receipts: tuple[str, ...]
    output_committed_at_ms: tuple[int, ...]
    started_at_ms: int
    finished_at_ms: int
    started_monotonic_ns: int
    finished_monotonic_ns: int
    receipt_sha256: str
    schema: str = "rl-quant.massive-measured-typed-stage-v1"

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.stage_id not in MASSIVE_MEASURED_TYPED_STAGE_IDS_V1:
            raise MassiveMeasuredTypedRunV1Error("measured typed stage ID differs")
        index = MASSIVE_MEASURED_TYPED_STAGE_IDS_V1.index(self.stage_id)
        if (
            self.schema != "rl-quant.massive-measured-typed-stage-v1"
            or self.implementation_source_sha256
            != MASSIVE_MEASURED_TYPED_STAGE_IMPLEMENTATIONS_V1[index]
            or self.configuration_receipt_sha256
            != MASSIVE_MEASURED_TYPED_STAGE_CONFIGURATION_V1[index]
            or not self.input_artifact_receipts
            or not self.output_artifact_receipts
            or not self.output_commit_receipts
            or len(self.output_commit_receipts) != len(self.output_committed_at_ms)
            or self.started_at_ms > self.finished_at_ms
            or self.started_monotonic_ns >= self.finished_monotonic_ns
            or any(
                not self.started_at_ms <= value <= self.finished_at_ms
                for value in self.output_committed_at_ms
            )
        ):
            raise MassiveMeasuredTypedRunV1Error("measured typed stage differs")
        for value in (
            self.implementation_source_sha256,
            self.configuration_receipt_sha256,
            self.receipt_sha256,
            *self.input_artifact_receipts,
            *self.output_artifact_receipts,
            *self.output_commit_receipts,
        ):
            _digest("measured typed stage receipt", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveMeasuredTypedRunV1Error("measured typed stage receipt differs")


def _stage(
    *,
    stage_id: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    commits: Sequence[tuple[str, int]],
    started_at_ms: int,
    finished_at_ms: int,
    started_monotonic_ns: int,
    finished_monotonic_ns: int,
) -> MassiveMeasuredTypedStageV1:
    index = MASSIVE_MEASURED_TYPED_STAGE_IDS_V1.index(stage_id)
    body = {
        "schema": "rl-quant.massive-measured-typed-stage-v1",
        "stage_id": stage_id,
        "implementation_source_sha256": MASSIVE_MEASURED_TYPED_STAGE_IMPLEMENTATIONS_V1[
            index
        ],
        "configuration_receipt_sha256": MASSIVE_MEASURED_TYPED_STAGE_CONFIGURATION_V1[
            index
        ],
        "input_artifact_receipts": tuple(inputs),
        "output_artifact_receipts": tuple(outputs),
        "output_commit_receipts": tuple(value[0] for value in commits),
        "output_committed_at_ms": tuple(value[1] for value in commits),
        "started_at_ms": started_at_ms,
        "finished_at_ms": finished_at_ms,
        "started_monotonic_ns": started_monotonic_ns,
        "finished_monotonic_ns": finished_monotonic_ns,
    }
    result = MassiveMeasuredTypedStageV1(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveMeasuredTypedRunV1:
    source_session_date: str
    decision_session_date: str
    session_authority: MassiveSessionAuthority
    source_session: MassiveExchangeSession
    decision_session: MassiveExchangeSession
    captured_listing_receipt_sha256: str
    archive_scope: MassiveFinalizedArchiveScopeV2
    authenticated_download: MassiveAuthenticatedFlatFileDownloadV1
    decision_origin: MassiveTypedDecisionOriginV1
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0
    semantic_partition_manifest: MassiveDailyTradePartitionManifestV0
    persisted_partition_manifest: MassivePersistedPartitionManifestV1
    daily_bars: MassiveDailyBarsArtifactV0
    daily_tape: MassiveDailyTapeArtifactV0
    rolling_features: MassiveRollingFeatureArtifactV0
    decision_tensor: MassivePIT500DecisionTensorV1
    checkpoint: MassiveValidationCheckpointV1
    inference: MassiveValidationInferenceArtifactV1
    requested_orders: MassiveRequestedOrdersArtifactV1
    stages: tuple[MassiveMeasuredTypedStageV1, ...]
    outer_started_at_ms: int
    outer_finished_at_ms: int
    outer_started_monotonic_ns: int
    outer_finished_monotonic_ns: int
    runtime_ms: int
    run_spec_receipt_sha256: str
    typed_timing_qualified: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_MEASURED_TYPED_RUN_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_MEASURED_TYPED_RUN_V1_SCHEMA
            or self.run_spec_receipt_sha256 != MASSIVE_MEASURED_TYPED_RUN_V1_SPEC_SHA256
            or self.source_session_date != self.decision_origin.source_session_date
            or self.decision_session_date != self.decision_origin.decision_session_date
            or self.outer_started_at_ms > self.outer_finished_at_ms
            or self.outer_started_monotonic_ns >= self.outer_finished_monotonic_ns
            or self.runtime_ms
            != (self.outer_finished_monotonic_ns - self.outer_started_monotonic_ns)
            // 1_000_000
            or self.runtime_ms > 55 * 60 * 1_000
            or self.outer_finished_at_ms > self.decision_origin.decision_at_ms
            or self.typed_timing_qualified
        ):
            raise MassiveMeasuredTypedRunV1Error(
                "measured typed run chronology differs"
            )
        self.archive_scope.validate()
        self.authenticated_download.validate()
        self.decision_origin.validate()
        self.session_authority.validate()
        self.source_session.validate()
        self.decision_session.validate()
        expected_origin = build_massive_typed_decision_origin_v1(
            session_authority=self.session_authority,
            source_session=self.source_session,
            decision_session=self.decision_session,
            authenticated_download=self.authenticated_download,
            archive_scope=self.archive_scope,
        )
        if self.decision_origin != expected_origin:
            raise MassiveMeasuredTypedRunV1Error(
                "measured typed decision origin was not authority-derived"
            )
        self.scan_evidence.validate()
        self.semantic_partition_manifest.validate()
        self.persisted_partition_manifest.validate()
        self.daily_bars.validate()
        self.daily_tape.validate()
        self.rolling_features.validate()
        self.decision_tensor.validate()
        self.checkpoint.validate()
        self.inference.validate()
        self.requested_orders.validate()
        if (
            tuple(stage.stage_id for stage in self.stages)
            != MASSIVE_MEASURED_TYPED_STAGE_IDS_V1
        ):
            raise MassiveMeasuredTypedRunV1Error(
                "measured typed stage inventory differs"
            )
        for stage in self.stages:
            stage.validate()
        if any(
            stage.started_at_ms < self.outer_started_at_ms
            or stage.finished_at_ms > self.outer_finished_at_ms
            or stage.started_monotonic_ns < self.outer_started_monotonic_ns
            or stage.finished_monotonic_ns > self.outer_finished_monotonic_ns
            for stage in self.stages
        ):
            raise MassiveMeasuredTypedRunV1Error(
                "measured typed stage escaped the outer timer"
            )
        if any(
            left.finished_monotonic_ns > right.started_monotonic_ns
            or left.finished_at_ms > right.started_at_ms
            for left, right in zip(self.stages, self.stages[1:])
        ):
            raise MassiveMeasuredTypedRunV1Error("measured typed stages overlap")
        expected_inputs_outputs = (
            (
                (self.captured_listing_receipt_sha256,),
                (
                    self.authenticated_download.receipt_sha256,
                    self.authenticated_download.loaded_source.receipt.receipt_sha256,
                ),
            ),
            (
                (
                    self.authenticated_download.receipt_sha256,
                    self.authenticated_download.loaded_source.receipt.receipt_sha256,
                ),
                (
                    self.scan_evidence.receipt_sha256,
                    self.semantic_partition_manifest.receipt_sha256,
                    self.persisted_partition_manifest.receipt_sha256,
                ),
            ),
            (
                (self.persisted_partition_manifest.receipt_sha256,),
                (self.daily_bars.receipt_sha256, self.daily_tape.receipt_sha256),
            ),
            (
                self.rolling_features.daily_bars_artifact_receipts
                + self.rolling_features.daily_tape_artifact_receipts,
                (self.rolling_features.receipt_sha256,),
            ),
            (
                (
                    self.rolling_features.receipt_sha256,
                    self.decision_origin.receipt_sha256,
                    self.decision_tensor.identity_authority_receipt_sha256,
                ),
                (self.decision_tensor.receipt_sha256,),
            ),
            (
                (self.decision_tensor.receipt_sha256, self.checkpoint.receipt_sha256),
                (self.inference.receipt_sha256,),
            ),
            (
                (
                    self.decision_tensor.receipt_sha256,
                    self.inference.receipt_sha256,
                    self.decision_origin.receipt_sha256,
                ),
                (self.requested_orders.receipt_sha256,),
            ),
        )
        if any(
            (stage.input_artifact_receipts, stage.output_artifact_receipts) != expected
            for stage, expected in zip(
                self.stages, expected_inputs_outputs, strict=True
            )
        ):
            raise MassiveMeasuredTypedRunV1Error("measured typed stage bindings differ")
        expected_loaded_outputs = (
            (self.authenticated_download.loaded_source,),
            tuple(
                source
                for partition in self.persisted_partition_manifest.partitions
                for source in (
                    partition.event_timeline,
                    partition.active_regular,
                    partition.correction_timeline,
                )
            ),
            (self.daily_bars.loaded_source, self.daily_tape.loaded_source),
            (self.rolling_features.loaded_source,),
            (self.decision_tensor.loaded_source,),
            (self.inference.loaded_source,),
            (self.requested_orders.loaded_source,),
        )
        for stage, loaded_outputs in zip(
            self.stages, expected_loaded_outputs, strict=True
        ):
            expected_commits = tuple(_commit(value) for value in loaded_outputs)
            if stage.output_commit_receipts != tuple(
                value[0] for value in expected_commits
            ) or stage.output_committed_at_ms != tuple(
                value[1] for value in expected_commits
            ):
                raise MassiveMeasuredTypedRunV1Error(
                    "measured typed stage commit inventory differs"
                )
            for loaded in loaded_outputs:
                if not (
                    stage.started_at_ms
                    <= loaded.receipt.requested_at_ms
                    <= loaded.receipt.downloaded_at_ms
                    <= loaded.commit.committed_at_ms
                    <= loaded.verified_at_ms
                    <= stage.finished_at_ms
                ):
                    raise MassiveMeasuredTypedRunV1Error(
                        "measured typed output escaped its stage interval"
                    )
        if (
            self.captured_listing_receipt_sha256
            != self.authenticated_download.listing_acquisition_receipt_sha256
            or self.authenticated_download.receipt_sha256
            != self.decision_origin.authenticated_download_receipt_sha256
            or self.scan_evidence.source_object_receipt_sha256
            != self.authenticated_download.loaded_source.receipt.receipt_sha256
            or self.persisted_partition_manifest.source_file_scan_receipt_sha256
            != self.scan_evidence.receipt_sha256
            or self.daily_bars.persisted_partition_manifest_receipt_sha256
            != self.persisted_partition_manifest.receipt_sha256
            or self.daily_tape.persisted_partition_manifest_receipt_sha256
            != self.persisted_partition_manifest.receipt_sha256
            or self.decision_tensor.rolling_feature_artifact_receipt_sha256
            != self.rolling_features.receipt_sha256
            or self.decision_tensor.decision_origin_receipt_sha256
            != self.decision_origin.receipt_sha256
            or self.inference.tensor_receipt_sha256
            != self.decision_tensor.receipt_sha256
            or self.inference.checkpoint_receipt_sha256
            != self.checkpoint.receipt_sha256
            or self.inference.setting_id != self.checkpoint.setting_id
            or self.inference.seed != self.checkpoint.seed
            or self.requested_orders.inference_receipt_sha256
            != self.inference.receipt_sha256
            or self.requested_orders.tensor_receipt_sha256
            != self.decision_tensor.receipt_sha256
            or self.requested_orders.decision_origin_receipt_sha256
            != self.decision_origin.receipt_sha256
            or self.requested_orders.setting_id != self.inference.setting_id
            or self.requested_orders.seed != self.checkpoint.seed
            or self.requested_orders.decision_session_date
            != self.decision_origin.decision_session_date
            or self.requested_orders.decision_at_ms
            != self.decision_origin.decision_at_ms
        ):
            raise MassiveMeasuredTypedRunV1Error(
                "measured typed authority chain differs"
            )
        if (
            self.panel_materialization_authorized
            or self.predictive_training_authorized
            or self.portfolio_evaluation_authorized
        ):
            raise MassiveMeasuredTypedRunV1Error(
                "measured typed run cannot authorize performance work"
            )
        _digest("measured typed run specification", self.run_spec_receipt_sha256)
        _digest("measured typed run receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveMeasuredTypedRunV1Error("measured typed run receipt differs")


def _commit(loaded: Any) -> tuple[str, int]:
    return loaded.commit.receipt_sha256, loaded.commit.committed_at_ms


def measure_massive_typed_finalized_run_v1(
    *,
    s3_client: Any,
    captured_listing: MassiveCapturedFlatFileListingV0,
    archive_scope: MassiveFinalizedArchiveScopeV2,
    session_authority: MassiveSessionAuthority,
    source_session: MassiveExchangeSession,
    decision_session: MassiveExchangeSession,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    prior_daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    prior_daily_tape: Sequence[MassiveDailyTapeArtifactV0],
    checkpoint: MassiveValidationCheckpointV1,
    source_root: str | Path,
    spool_root: str | Path,
    persisted_root: str | Path,
    artifact_root: str | Path,
    entitlement_receipt_sha256: str,
    now_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> MassiveMeasuredTypedRunV1:
    """Measure a development run; injected clocks can never authorize timing."""

    checkpoint.validate()
    captured_listing.validate()
    archive_scope.validate()
    outer_started_at_ms = now_ms()
    outer_started_monotonic_ns = monotonic_ns()
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    stages: list[MassiveMeasuredTypedStageV1] = []

    started_at_ms, started_mono = now_ms(), monotonic_ns()
    download = download_massive_flat_file_object_v1(
        s3_client=s3_client,
        captured_listing=captured_listing,
        source_object_key=canonical_massive_trade_object_key(
            source_session.session_date
        ),
        destination_root=source_root,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        now_ms=now_ms,
    )
    finished_at_ms, finished_mono = now_ms(), monotonic_ns()
    stages.append(
        _stage(
            stage_id="authenticated-object-get",
            inputs=(captured_listing.acquisition_evidence.receipt_sha256,),
            outputs=(
                download.receipt_sha256,
                download.loaded_source.receipt.receipt_sha256,
            ),
            commits=(_commit(download.loaded_source),),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            started_monotonic_ns=started_mono,
            finished_monotonic_ns=finished_mono,
        )
    )
    decision_origin = build_massive_typed_decision_origin_v1(
        session_authority=session_authority,
        source_session=source_session,
        decision_session=decision_session,
        authenticated_download=download,
        archive_scope=archive_scope,
    )

    started_at_ms, started_mono = now_ms(), monotonic_ns()
    published_at_ms = now_ms()
    scan, semantic_partition, persisted = (
        stream_and_persist_massive_daily_trade_partitions_v1(
            source_root=source_root,
            loaded_source=download.loaded_source,
            spool_root=spool_root,
            persisted_root=persisted_root,
            session_authority=session_authority,
            session=source_session,
            identity_authority=identity_authority,
            condition_authority=condition_authority,
            correction_authority=correction_authority,
            feature_domain_spec=feature_domain_spec,
            entitlement_receipt_sha256=entitlement_receipt_sha256,
            published_at_ms=published_at_ms,
        )
    )
    validate_massive_persisted_partitions_semantically_v2(
        root=persisted_root,
        manifest=persisted,
        scan_evidence=scan,
        semantic_partition_manifest=semantic_partition,
        identity_authority=identity_authority,
        correction_authority=correction_authority,
    )
    finished_at_ms, finished_mono = now_ms(), monotonic_ns()
    partition_commits = tuple(
        _commit(source)
        for partition in persisted.partitions
        for source in (
            partition.event_timeline,
            partition.active_regular,
            partition.correction_timeline,
        )
    )
    stages.append(
        _stage(
            stage_id="source-scan-route-replay-persist",
            inputs=(
                download.receipt_sha256,
                download.loaded_source.receipt.receipt_sha256,
            ),
            outputs=(
                scan.receipt_sha256,
                semantic_partition.receipt_sha256,
                persisted.receipt_sha256,
            ),
            commits=partition_commits,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            started_monotonic_ns=started_mono,
            finished_monotonic_ns=finished_mono,
        )
    )

    started_at_ms, started_mono = now_ms(), monotonic_ns()
    published_at_ms = now_ms()
    bars = materialize_massive_daily_bars_v0(
        persisted_root=persisted_root,
        output_root=artifact_root,
        manifest=persisted,
        condition_authority=condition_authority,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    tape = materialize_massive_daily_tape_v0(
        persisted_root=persisted_root,
        output_root=artifact_root,
        manifest=persisted,
        condition_authority=condition_authority,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    finished_at_ms, finished_mono = now_ms(), monotonic_ns()
    stages.append(
        _stage(
            stage_id="daily-features",
            inputs=(persisted.receipt_sha256,),
            outputs=(bars.receipt_sha256, tape.receipt_sha256),
            commits=(_commit(bars.loaded_source), _commit(tape.loaded_source)),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            started_monotonic_ns=started_mono,
            finished_monotonic_ns=finished_mono,
        )
    )

    bars_history = tuple(prior_daily_bars) + (bars,)
    tape_history = tuple(prior_daily_tape) + (tape,)
    started_at_ms, started_mono = now_ms(), monotonic_ns()
    published_at_ms = now_ms()
    rolling = materialize_massive_rolling_features_v0(
        daily_feature_root=artifact_root,
        output_root=artifact_root,
        bars_artifacts=bars_history,
        tape_artifacts=tape_history,
        identity_authority=identity_authority,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    finished_at_ms, finished_mono = now_ms(), monotonic_ns()
    stages.append(
        _stage(
            stage_id="rolling-features",
            inputs=rolling.daily_bars_artifact_receipts
            + rolling.daily_tape_artifact_receipts,
            outputs=(rolling.receipt_sha256,),
            commits=(_commit(rolling.loaded_source),),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            started_monotonic_ns=started_mono,
            finished_monotonic_ns=finished_mono,
        )
    )

    started_at_ms, started_mono = now_ms(), monotonic_ns()
    published_at_ms = now_ms()
    tensor = materialize_massive_pit500_tensor_v1(
        rolling_root=artifact_root,
        output_root=artifact_root,
        rolling=rolling,
        identity_authority=identity_authority,
        decision_origin=decision_origin,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    finished_at_ms, finished_mono = now_ms(), monotonic_ns()
    stages.append(
        _stage(
            stage_id="pit500-decision-tensor",
            inputs=(
                rolling.receipt_sha256,
                decision_origin.receipt_sha256,
                identity_authority.receipt_sha256,
            ),
            outputs=(tensor.receipt_sha256,),
            commits=(_commit(tensor.loaded_source),),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            started_monotonic_ns=started_mono,
            finished_monotonic_ns=finished_mono,
        )
    )

    started_at_ms, started_mono = now_ms(), monotonic_ns()
    published_at_ms = now_ms()
    inference = materialize_massive_validation_inference_v1(
        tensor_root=artifact_root,
        output_root=artifact_root,
        tensor=tensor,
        checkpoint=checkpoint,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    finished_at_ms, finished_mono = now_ms(), monotonic_ns()
    stages.append(
        _stage(
            stage_id="frozen-model-inference",
            inputs=(tensor.receipt_sha256, checkpoint.receipt_sha256),
            outputs=(inference.receipt_sha256,),
            commits=(_commit(inference.loaded_source),),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            started_monotonic_ns=started_mono,
            finished_monotonic_ns=finished_mono,
        )
    )

    started_at_ms, started_mono = now_ms(), monotonic_ns()
    published_at_ms = now_ms()
    orders = materialize_massive_requested_orders_v1(
        tensor_root=artifact_root,
        inference_root=artifact_root,
        output_root=artifact_root,
        tensor=tensor,
        inference=inference,
        decision_origin=decision_origin,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        published_at_ms=published_at_ms,
    )
    finished_at_ms, finished_mono = now_ms(), monotonic_ns()
    stages.append(
        _stage(
            stage_id="requested-orders",
            inputs=(
                tensor.receipt_sha256,
                inference.receipt_sha256,
                decision_origin.receipt_sha256,
            ),
            outputs=(orders.receipt_sha256,),
            commits=(_commit(orders.loaded_source),),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            started_monotonic_ns=started_mono,
            finished_monotonic_ns=finished_mono,
        )
    )

    validate_massive_daily_bars_v0(root=artifact_root, artifact=bars)
    validate_massive_daily_tape_v0(root=artifact_root, artifact=tape)
    validate_massive_rolling_features_v0(root=artifact_root, artifact=rolling)
    validate_massive_pit500_tensor_v1(root=artifact_root, tensor=tensor)
    validate_massive_validation_inference_v1(root=artifact_root, artifact=inference)
    validate_massive_requested_orders_v1(root=artifact_root, artifact=orders)
    outer_finished_at_ms = now_ms()
    outer_finished_monotonic_ns = monotonic_ns()
    runtime_ms = (outer_finished_monotonic_ns - outer_started_monotonic_ns) // 1_000_000
    body = {
        "schema": MASSIVE_MEASURED_TYPED_RUN_V1_SCHEMA,
        "source_session_date": source_session.session_date,
        "decision_session_date": decision_session.session_date,
        "session_authority": session_authority,
        "source_session": source_session,
        "decision_session": decision_session,
        "captured_listing_receipt_sha256": captured_listing.acquisition_evidence.receipt_sha256,
        "archive_scope": archive_scope,
        "authenticated_download": download,
        "decision_origin": decision_origin,
        "scan_evidence": scan,
        "semantic_partition_manifest": semantic_partition,
        "persisted_partition_manifest": persisted,
        "daily_bars": bars,
        "daily_tape": tape,
        "rolling_features": rolling,
        "decision_tensor": tensor,
        "checkpoint": checkpoint,
        "inference": inference,
        "requested_orders": orders,
        "stages": tuple(stages),
        "outer_started_at_ms": outer_started_at_ms,
        "outer_finished_at_ms": outer_finished_at_ms,
        "outer_started_monotonic_ns": outer_started_monotonic_ns,
        "outer_finished_monotonic_ns": outer_finished_monotonic_ns,
        "runtime_ms": runtime_ms,
        "run_spec_receipt_sha256": MASSIVE_MEASURED_TYPED_RUN_V1_SPEC_SHA256,
        "typed_timing_qualified": False,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
    provisional = MassiveMeasuredTypedRunV1(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveMeasuredTypedRunV1(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    return result


measure_massive_typed_finalized_run_for_test_v1 = measure_massive_typed_finalized_run_v1


__all__ = [
    "MASSIVE_MEASURED_TYPED_RUN_V1_SPEC_SHA256",
    "MASSIVE_MEASURED_TYPED_STAGE_IDS_V1",
    "MassiveMeasuredTypedRunV1",
    "MassiveMeasuredTypedRunV1Error",
    "MassiveMeasuredTypedStageV1",
    "measure_massive_typed_finalized_run_for_test_v1",
    "measure_massive_typed_finalized_run_v1",
]
