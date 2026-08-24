"""Artifact-derived readiness evidence for finalized validation V0.

The older readiness-v0 records are retained as immutable development evidence.
This generation starts its timer before the source transaction, derives scan and
partition evidence inside the measured workflow, and accepts only stage outputs
whose committed bytes satisfy frozen stage contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import Callable, Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MassiveCapturedFlatFileListingV0,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionManifestV0,
    MassiveFinalizedFeatureDomainSpecV0,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    stream_and_persist_massive_daily_trade_partitions_v1,
    validate_massive_persisted_partitions_v1,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1 = (
    "source-download-and-commit",
    "whole-file-scan",
    "pit-route-and-finalized-replay",
    "persisted-trade-partitions",
    "daily-features",
    "rolling-features",
    "pit500-decision-tensor",
    "frozen-model-inference",
    "requested-orders",
)
MASSIVE_ARTIFACT_READINESS_STAGE_V1_SCHEMA = (
    "rl-quant.massive-finalized-artifact-readiness-stage-v1"
)
MASSIVE_ARTIFACT_READINESS_RUN_V1_SCHEMA = (
    "rl-quant.massive-finalized-artifact-readiness-run-v1"
)
MASSIVE_ARTIFACT_READINESS_PANEL_V1_SCHEMA = (
    "rl-quant.massive-finalized-artifact-readiness-panel-v1"
)
MASSIVE_ARTIFACT_READINESS_CAPABILITY_V1_SCHEMA = (
    "rl-quant.massive-finalized-artifact-readiness-capability-v1"
)
MASSIVE_ARTIFACT_EXECUTION_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-finalized-artifact-execution-authority-v1"
)
MASSIVE_ARTIFACT_EXECUTION_DATASET_V1 = (
    "massive-finalized-artifact-execution-environment-v1"
)
MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": "rl-quant.massive-finalized-execution-environment-source-v1",
        "fields": (
            "hardware_contract_receipt_sha256",
            "software_source_archive_sha256",
            "container_image_receipt_sha256",
            "python_environment_receipt_sha256",
        ),
    }
)
MASSIVE_ARTIFACT_READINESS_MINIMUM_SESSIONS_V1 = 20
MASSIVE_ARTIFACT_READINESS_MINIMUM_YEARS_V1 = 3
MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256 = file_sha256(Path(__file__))


def _stage_schema(stage_id: str) -> str:
    return semantic_sha256(
        {
            "generation": "massive-finalized-artifact-readiness-v1",
            "stage_id": stage_id,
            "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
            "required_fields": (
                "schema",
                "stage_id",
                "source_session_date",
                "input_artifact_receipts",
                "semantic_output_receipt_sha256",
                "output_row_count",
                "implementation_source_sha256",
                "protocol_receipt_sha256",
            ),
        }
    )


MASSIVE_ARTIFACT_READINESS_STAGE_CONTRACTS_V1 = {
    stage_id: {
        "dataset_id": f"massive-finalized-readiness-{stage_id}-v1",
        "schema_sha256": _stage_schema(stage_id),
    }
    for stage_id in MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1[1:]
}


class MassiveArtifactReadinessError(ValueError):
    """Artifact-derived readiness evidence is incomplete or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveArtifactReadinessError(f"{name} must be a lowercase SHA-256")
    return value


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveArtifactReadinessError(f"{name} must be positive")
    return value


def _nonnegative(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveArtifactReadinessError(f"{name} must be nonnegative")
    return value


def _canonical_stage_payload(
    *,
    stage_id: str,
    source_session_date: str,
    input_artifact_receipts: tuple[str, ...],
    semantic_output_receipt_sha256: str,
    output_row_count: int,
    implementation_source_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "rl-quant.massive-finalized-readiness-stage-output-v1",
        "stage_id": stage_id,
        "source_session_date": source_session_date,
        "input_artifact_receipts": input_artifact_receipts,
        "semantic_output_receipt_sha256": semantic_output_receipt_sha256,
        "output_row_count": output_row_count,
        "implementation_source_sha256": implementation_source_sha256,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
    }


@dataclass(frozen=True, slots=True)
class MassiveArtifactExecutionAuthorityV1:
    loaded_source: LoadedMassiveSourceObject
    hardware_contract_receipt_sha256: str
    software_source_archive_sha256: str
    container_image_receipt_sha256: str
    python_environment_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_ARTIFACT_EXECUTION_AUTHORITY_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ARTIFACT_EXECUTION_AUTHORITY_V1_SCHEMA:
            raise MassiveArtifactReadinessError("execution authority schema drifted")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ARTIFACT_EXECUTION_DATASET_V1
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveArtifactReadinessError(
                "execution authority source dataset/schema differ"
            )
        for name in (
            "hardware_contract_receipt_sha256",
            "software_source_archive_sha256",
            "container_image_receipt_sha256",
            "python_environment_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveArtifactReadinessError("execution authority receipt differs")


def parse_massive_artifact_execution_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveArtifactExecutionAuthorityV1:
    """Issue an execution identity only from committed canonical bytes."""

    loaded_source.validate()
    if (
        loaded_source.receipt.dataset_id != MASSIVE_ARTIFACT_EXECUTION_DATASET_V1
        or loaded_source.receipt.schema_sha256
        != MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256
    ):
        raise MassiveArtifactReadinessError(
            "execution authority source dataset/schema differ"
        )
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveArtifactReadinessError(
            "execution authority source is not JSON"
        ) from exc
    expected_keys = {
        "hardware_contract_receipt_sha256",
        "software_source_archive_sha256",
        "container_image_receipt_sha256",
        "python_environment_receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or raw != canonical_json_file_bytes(payload)
    ):
        raise MassiveArtifactReadinessError(
            "execution authority source is not canonical"
        )
    for name in expected_keys:
        _digest(name, payload[name])
    body: dict[str, object] = {
        "schema": MASSIVE_ARTIFACT_EXECUTION_AUTHORITY_V1_SCHEMA,
        "loaded_source": loaded_source,
        **payload,
    }
    provisional = MassiveArtifactExecutionAuthorityV1(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveArtifactExecutionAuthorityV1(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveArtifactReadinessStageV1:
    stage_id: str
    input_artifact_receipts: tuple[str, ...]
    output_loaded_source: LoadedMassiveSourceObject
    semantic_output_receipt_sha256: str
    output_row_count: int
    implementation_source_sha256: str
    stage_started_at_ms: int
    stage_finished_at_ms: int
    stage_started_monotonic_ns: int
    stage_finished_monotonic_ns: int
    receipt_sha256: str
    schema: str = MASSIVE_ARTIFACT_READINESS_STAGE_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ARTIFACT_READINESS_STAGE_V1_SCHEMA:
            raise MassiveArtifactReadinessError(
                "artifact-readiness stage schema drifted"
            )
        if self.stage_id not in MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1:
            raise MassiveArtifactReadinessError(
                "artifact-readiness stage identity drifted"
            )
        if not self.input_artifact_receipts or self.input_artifact_receipts != tuple(
            sorted(set(self.input_artifact_receipts))
        ):
            raise MassiveArtifactReadinessError("stage inputs are not canonical")
        for value in self.input_artifact_receipts:
            _digest("stage input", value)
        self.output_loaded_source.validate()
        for name in (
            "semantic_output_receipt_sha256",
            "implementation_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        _nonnegative("stage output rows", self.output_row_count)
        _positive("stage wall start", self.stage_started_at_ms)
        _positive("stage wall finish", self.stage_finished_at_ms)
        _positive("stage monotonic start", self.stage_started_monotonic_ns)
        _positive("stage monotonic finish", self.stage_finished_monotonic_ns)
        if (
            self.stage_finished_at_ms < self.stage_started_at_ms
            or self.stage_finished_monotonic_ns <= self.stage_started_monotonic_ns
        ):
            raise MassiveArtifactReadinessError("stage chronology is invalid")
        receipt = self.output_loaded_source.receipt
        commit = self.output_loaded_source.commit
        payload_ctime_ms = self.output_loaded_source.payload_ctime_ns // 1_000_000
        if not (
            self.stage_started_at_ms
            <= receipt.requested_at_ms
            <= receipt.downloaded_at_ms
            <= commit.committed_at_ms
            <= self.output_loaded_source.verified_at_ms
            <= self.stage_finished_at_ms
            and self.stage_started_at_ms - 1
            <= payload_ctime_ms
            <= self.stage_finished_at_ms + 1
        ):
            raise MassiveArtifactReadinessError(
                "stage output was not created and verified inside its measured interval"
            )
        if self.stage_id != "source-download-and-commit":
            contract = MASSIVE_ARTIFACT_READINESS_STAGE_CONTRACTS_V1[self.stage_id]
            if (
                receipt.dataset_id != contract["dataset_id"]
                or receipt.schema_sha256 != contract["schema_sha256"]
            ):
                raise MassiveArtifactReadinessError(
                    "stage dataset/schema contract differs"
                )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveArtifactReadinessError(
                "artifact-readiness stage receipt differs"
            )


def validate_massive_artifact_readiness_stage_bytes_v1(
    *,
    root: str | Path,
    stage: MassiveArtifactReadinessStageV1,
    source_session_date: str,
) -> None:
    stage.validate()
    if stage.stage_id == "source-download-and-commit":
        read_loaded_massive_source_bytes(
            root=root, loaded_source=stage.output_loaded_source
        )
        return
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=stage.output_loaded_source
    )
    try:
        json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveArtifactReadinessError(
            "stage output is not canonical JSON"
        ) from exc
    expected = _canonical_stage_payload(
        stage_id=stage.stage_id,
        source_session_date=source_session_date,
        input_artifact_receipts=stage.input_artifact_receipts,
        semantic_output_receipt_sha256=stage.semantic_output_receipt_sha256,
        output_row_count=stage.output_row_count,
        implementation_source_sha256=stage.implementation_source_sha256,
    )
    if raw != canonical_json_file_bytes(expected):
        raise MassiveArtifactReadinessError(
            "stage output bytes differ from its contract"
        )


@dataclass(frozen=True, slots=True)
class MassiveArtifactReadinessRunV1:
    captured_listing: MassiveCapturedFlatFileListingV0
    source_session: MassiveExchangeSession
    session_authority_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0
    semantic_partition_manifest: MassiveDailyTradePartitionManifestV0
    persisted_partition_manifest: MassivePersistedPartitionManifestV1
    execution_authority: MassiveArtifactExecutionAuthorityV1
    stages: tuple[MassiveArtifactReadinessStageV1, ...]
    stage_receipt_inventory_sha256: str
    identity_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    wall_started_at_ms: int
    wall_finished_at_ms: int
    monotonic_started_ns: int
    monotonic_finished_ns: int
    observed_full_pipeline_runtime_ms: int
    compressed_bytes: int
    source_row_count: int
    ticker_count: int
    correction_event_count: int
    active_event_key_count: int
    security_partition_count: int
    daily_feature_row_count: int
    rolling_feature_row_count: int
    decision_tensor_bytes: int
    receipt_sha256: str
    schema: str = MASSIVE_ARTIFACT_READINESS_RUN_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ARTIFACT_READINESS_RUN_V1_SCHEMA:
            raise MassiveArtifactReadinessError("artifact-readiness run schema drifted")
        self.captured_listing.validate()
        self.source_session.validate()
        self.loaded_source.validate()
        self.scan_evidence.validate()
        self.semantic_partition_manifest.validate()
        self.persisted_partition_manifest.validate()
        self.execution_authority.validate()
        for name in (
            "session_authority_receipt_sha256",
            "stage_receipt_inventory_sha256",
            "identity_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.source_session.session_date != self.scan_evidence.source_session_date
            or self.scan_evidence.source_object_receipt_sha256
            != self.loaded_source.receipt.receipt_sha256
            or self.scan_evidence.session_authority_receipt_sha256
            != self.session_authority_receipt_sha256
            or self.semantic_partition_manifest.source_file_scan_receipt_sha256
            != self.scan_evidence.receipt_sha256
            or self.persisted_partition_manifest.source_file_scan_receipt_sha256
            != self.scan_evidence.receipt_sha256
            or self.persisted_partition_manifest.semantic_partition_manifest_receipt_sha256
            != self.semantic_partition_manifest.receipt_sha256
            or self.semantic_partition_manifest.identity_authority_receipt_sha256
            != self.identity_authority_receipt_sha256
            or self.semantic_partition_manifest.condition_authority_receipt_sha256
            != self.condition_authority_receipt_sha256
            or self.semantic_partition_manifest.correction_authority_receipt_sha256
            != self.correction_authority_receipt_sha256
        ):
            raise MassiveArtifactReadinessError("typed readiness authorities differ")
        entry = self.captured_listing.committed_listing.resolve(
            source_object_key=self.loaded_source.receipt.source_object_key
        )
        if (
            entry.content_length != self.loaded_source.receipt.content_length
            or entry.etag != self.loaded_source.receipt.etag
        ):
            raise MassiveArtifactReadinessError(
                "run source differs from acquired listing"
            )
        if (
            tuple(stage.stage_id for stage in self.stages)
            != MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1
        ):
            raise MassiveArtifactReadinessError("run stage inventory drifted")
        expected_inputs = (self.captured_listing.acquisition_evidence.receipt_sha256,)
        for stage in self.stages:
            stage.validate()
            if stage.input_artifact_receipts != expected_inputs:
                raise MassiveArtifactReadinessError("run stage chain is discontinuous")
            expected_inputs = (stage.semantic_output_receipt_sha256,)
        if self.stages[0].output_loaded_source != self.loaded_source:
            raise MassiveArtifactReadinessError(
                "source stage did not produce the bound source"
            )
        if (
            self.stages[1].semantic_output_receipt_sha256
            != self.scan_evidence.receipt_sha256
            or self.stages[2].semantic_output_receipt_sha256
            != self.semantic_partition_manifest.receipt_sha256
            or self.stages[3].semantic_output_receipt_sha256
            != self.persisted_partition_manifest.receipt_sha256
        ):
            raise MassiveArtifactReadinessError("derived stage outputs differ")
        if self.stage_receipt_inventory_sha256 != semantic_sha256(
            tuple(stage.receipt_sha256 for stage in self.stages)
        ):
            raise MassiveArtifactReadinessError("run stage receipt inventory differs")
        for prior, current in zip(self.stages, self.stages[1:]):
            if (
                current.stage_started_at_ms < prior.stage_finished_at_ms
                or current.stage_started_monotonic_ns
                < prior.stage_finished_monotonic_ns
            ):
                raise MassiveArtifactReadinessError(
                    "readiness stages overlap or regress"
                )
        if (
            self.stages[0].stage_started_at_ms < self.wall_started_at_ms
            or self.stages[-1].stage_finished_at_ms > self.wall_finished_at_ms
            or self.stages[0].stage_started_monotonic_ns < self.monotonic_started_ns
            or self.stages[-1].stage_finished_monotonic_ns > self.monotonic_finished_ns
            or self.wall_finished_at_ms < self.wall_started_at_ms
            or self.monotonic_finished_ns <= self.monotonic_started_ns
        ):
            raise MassiveArtifactReadinessError("outer readiness chronology differs")
        expected_runtime = max(
            1,
            (self.monotonic_finished_ns - self.monotonic_started_ns + 999_999)
            // 1_000_000,
        )
        if self.observed_full_pipeline_runtime_ms != expected_runtime:
            raise MassiveArtifactReadinessError(
                "outer runtime was not monotonic-derived"
            )
        derived = {
            "compressed_bytes": self.loaded_source.receipt.content_length,
            "source_row_count": self.scan_evidence.source_row_count,
            "ticker_count": self.scan_evidence.ticker_count,
            "correction_event_count": self.persisted_partition_manifest.correction_event_count,
            "active_event_key_count": self.persisted_partition_manifest.active_event_key_count,
            "security_partition_count": self.persisted_partition_manifest.security_partition_count,
            "daily_feature_row_count": self.stages[4].output_row_count,
            "rolling_feature_row_count": self.stages[5].output_row_count,
            "decision_tensor_bytes": self.stages[
                6
            ].output_loaded_source.receipt.content_length,
        }
        for name, value in derived.items():
            if getattr(self, name) != value:
                raise MassiveArtifactReadinessError(f"run workload {name} differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveArtifactReadinessError(
                "artifact-readiness run receipt differs"
            )


StageOutputRunnerV1 = Callable[
    [str, tuple[str, ...], str, int],
    tuple[str | Path, LoadedMassiveSourceObject, str, int],
]
SourceLoaderV1 = Callable[[int], tuple[str | Path, LoadedMassiveSourceObject]]


def _publish_evidence_stage(
    *,
    artifact_root: str | Path,
    stage_id: str,
    source_session_date: str,
    input_receipts: tuple[str, ...],
    semantic_receipt: str,
    output_rows: int,
    implementation_source_sha256: str,
    timestamp_ms: int,
) -> LoadedMassiveSourceObject:
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    payload = _canonical_stage_payload(
        stage_id=stage_id,
        source_session_date=source_session_date,
        input_artifact_receipts=input_receipts,
        semantic_output_receipt_sha256=semantic_receipt,
        output_row_count=output_rows,
        implementation_source_sha256=implementation_source_sha256,
    )
    relative = f"artifact-readiness-v1/session={source_session_date}/{stage_id}.json"
    contract = MASSIVE_ARTIFACT_READINESS_STAGE_CONTRACTS_V1[stage_id]
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=artifact_root,
        relative_payload_path=relative,
        dataset_id=str(contract["dataset_id"]),
        source_object_key=relative,
        requested_at_ms=timestamp_ms,
        downloaded_at_ms=timestamp_ms,
        schema_sha256=str(contract["schema_sha256"]),
        entitlement_receipt_sha256=semantic_sha256("local-artifact-readiness-v1"),
        committed_at_ms=timestamp_ms,
    )
    return load_massive_source_bundle(
        root=artifact_root,
        relative_payload_path=relative,
        verified_at_ms=timestamp_ms,
    )


def _stage(
    *,
    stage_id: str,
    inputs: tuple[str, ...],
    loaded: LoadedMassiveSourceObject,
    semantic_receipt: str,
    output_rows: int,
    implementation_source_sha256: str,
    started_ms: int,
    finished_ms: int,
    started_ns: int,
    finished_ns: int,
) -> MassiveArtifactReadinessStageV1:
    body: dict[str, object] = {
        "schema": MASSIVE_ARTIFACT_READINESS_STAGE_V1_SCHEMA,
        "stage_id": stage_id,
        "input_artifact_receipts": inputs,
        "output_loaded_source": loaded,
        "semantic_output_receipt_sha256": semantic_receipt,
        "output_row_count": output_rows,
        "implementation_source_sha256": implementation_source_sha256,
        "stage_started_at_ms": started_ms,
        "stage_finished_at_ms": finished_ms,
        "stage_started_monotonic_ns": started_ns,
        "stage_finished_monotonic_ns": finished_ns,
    }
    provisional = MassiveArtifactReadinessStageV1(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveArtifactReadinessStageV1(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    return result


def measure_massive_artifact_readiness_v1(
    *,
    source_loader: SourceLoaderV1,
    artifact_root: str | Path,
    persisted_partition_root: str | Path,
    execution_authority_root: str | Path,
    execution_authority: MassiveArtifactExecutionAuthorityV1,
    captured_listing: MassiveCapturedFlatFileListingV0,
    session_authority: MassiveSessionAuthority,
    source_session: MassiveExchangeSession,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    downstream_stage_runner: StageOutputRunnerV1,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wall_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
) -> MassiveArtifactReadinessRunV1:
    """Measure source transaction through requested orders in one outer timer."""

    captured_listing.validate()
    session_authority.validate()
    source_session.validate()
    identity_authority.validate()
    condition_authority.validate()
    correction_authority.validate()
    feature_domain_spec.validate()
    expected_execution = parse_massive_artifact_execution_authority_v1(
        root=execution_authority_root,
        loaded_source=execution_authority.loaded_source,
    )
    if execution_authority != expected_execution:
        raise MassiveArtifactReadinessError(
            "execution authority was not parsed from committed bytes"
        )
    wall_started = wall_ms()
    mono_started = monotonic_ns()
    stages: list[MassiveArtifactReadinessStageV1] = []
    inputs = (captured_listing.acquisition_evidence.receipt_sha256,)

    stage_wall_start = wall_ms()
    stage_mono_start = monotonic_ns()
    source_root, loaded_source = source_loader(stage_wall_start)
    stage_mono_finish = monotonic_ns()
    stage_wall_finish = wall_ms()
    source_stage = _stage(
        stage_id="source-download-and-commit",
        inputs=inputs,
        loaded=loaded_source,
        semantic_receipt=loaded_source.receipt.receipt_sha256,
        output_rows=0,
        implementation_source_sha256=MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
        started_ms=stage_wall_start,
        finished_ms=stage_wall_finish,
        started_ns=stage_mono_start,
        finished_ns=stage_mono_finish,
    )
    stages.append(source_stage)
    inputs = (source_stage.semantic_output_receipt_sha256,)

    stage_wall_start = wall_ms()
    stage_mono_start = monotonic_ns()
    scan, semantic_partition, persisted = (
        stream_and_persist_massive_daily_trade_partitions_v1(
            source_root=source_root,
            loaded_source=loaded_source,
            spool_root=Path(artifact_root) / ".spool",
            persisted_root=persisted_partition_root,
            session_authority=session_authority,
            session=source_session,
            identity_authority=identity_authority,
            condition_authority=condition_authority,
            correction_authority=correction_authority,
            feature_domain_spec=feature_domain_spec,
            entitlement_receipt_sha256=(
                loaded_source.receipt.entitlement_receipt_sha256
            ),
            published_at_ms=stage_wall_start,
        )
    )
    stage_publish_at = wall_ms()
    scan_loaded = _publish_evidence_stage(
        artifact_root=artifact_root,
        stage_id="whole-file-scan",
        source_session_date=source_session.session_date,
        input_receipts=inputs,
        semantic_receipt=scan.receipt_sha256,
        output_rows=scan.source_row_count,
        implementation_source_sha256=MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
        timestamp_ms=stage_publish_at,
    )
    stage_mono_finish = monotonic_ns()
    stage_wall_finish = wall_ms()
    scan_stage = _stage(
        stage_id="whole-file-scan",
        inputs=inputs,
        loaded=scan_loaded,
        semantic_receipt=scan.receipt_sha256,
        output_rows=scan.source_row_count,
        implementation_source_sha256=MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
        started_ms=stage_wall_start,
        finished_ms=stage_wall_finish,
        started_ns=stage_mono_start,
        finished_ns=stage_mono_finish,
    )
    stages.append(scan_stage)
    inputs = (scan.receipt_sha256,)

    stage_wall_start = wall_ms()
    stage_mono_start = monotonic_ns()
    semantic_partition.validate()
    stage_publish_at = wall_ms()
    route_loaded = _publish_evidence_stage(
        artifact_root=artifact_root,
        stage_id="pit-route-and-finalized-replay",
        source_session_date=source_session.session_date,
        input_receipts=inputs,
        semantic_receipt=semantic_partition.receipt_sha256,
        output_rows=semantic_partition.partitioned_row_count,
        implementation_source_sha256=MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
        timestamp_ms=stage_publish_at,
    )
    stage_mono_finish = monotonic_ns()
    stage_wall_finish = wall_ms()
    route_stage = _stage(
        stage_id="pit-route-and-finalized-replay",
        inputs=inputs,
        loaded=route_loaded,
        semantic_receipt=semantic_partition.receipt_sha256,
        output_rows=semantic_partition.partitioned_row_count,
        implementation_source_sha256=MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
        started_ms=stage_wall_start,
        finished_ms=stage_wall_finish,
        started_ns=stage_mono_start,
        finished_ns=stage_mono_finish,
    )
    stages.append(route_stage)
    inputs = (semantic_partition.receipt_sha256,)

    stage_wall_start = wall_ms()
    stage_mono_start = monotonic_ns()
    validate_massive_persisted_partitions_v1(
        root=persisted_partition_root, manifest=persisted
    )
    stage_publish_at = wall_ms()
    persisted_loaded = _publish_evidence_stage(
        artifact_root=artifact_root,
        stage_id="persisted-trade-partitions",
        source_session_date=source_session.session_date,
        input_receipts=inputs,
        semantic_receipt=persisted.receipt_sha256,
        output_rows=persisted.persisted_event_row_count,
        implementation_source_sha256=MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
        timestamp_ms=stage_publish_at,
    )
    stage_mono_finish = monotonic_ns()
    stage_wall_finish = wall_ms()
    persisted_stage = _stage(
        stage_id="persisted-trade-partitions",
        inputs=inputs,
        loaded=persisted_loaded,
        semantic_receipt=persisted.receipt_sha256,
        output_rows=persisted.persisted_event_row_count,
        implementation_source_sha256=MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
        started_ms=stage_wall_start,
        finished_ms=stage_wall_finish,
        started_ns=stage_mono_start,
        finished_ns=stage_mono_finish,
    )
    stages.append(persisted_stage)
    inputs = (persisted.receipt_sha256,)

    roots: dict[str, str | Path] = {
        "source-download-and-commit": source_root,
        "whole-file-scan": artifact_root,
        "pit-route-and-finalized-replay": artifact_root,
        "persisted-trade-partitions": artifact_root,
    }
    for stage_id in MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1[4:]:
        stage_wall_start = wall_ms()
        stage_mono_start = monotonic_ns()
        root, loaded, semantic_receipt, output_rows = downstream_stage_runner(
            stage_id, inputs, source_session.session_date, stage_wall_start
        )
        stage_mono_finish = monotonic_ns()
        stage_wall_finish = wall_ms()
        stage = _stage(
            stage_id=stage_id,
            inputs=inputs,
            loaded=loaded,
            semantic_receipt=semantic_receipt,
            output_rows=output_rows,
            implementation_source_sha256=MASSIVE_ARTIFACT_READINESS_SOURCE_SHA256,
            started_ms=stage_wall_start,
            finished_ms=stage_wall_finish,
            started_ns=stage_mono_start,
            finished_ns=stage_mono_finish,
        )
        validate_massive_artifact_readiness_stage_bytes_v1(
            root=root, stage=stage, source_session_date=source_session.session_date
        )
        stages.append(stage)
        roots[stage_id] = root
        inputs = (semantic_receipt,)

    mono_finished = monotonic_ns()
    wall_finished = wall_ms()
    stage_rows = tuple(stages)
    body: dict[str, object] = {
        "schema": MASSIVE_ARTIFACT_READINESS_RUN_V1_SCHEMA,
        "captured_listing": captured_listing,
        "source_session": source_session,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "loaded_source": loaded_source,
        "scan_evidence": scan,
        "semantic_partition_manifest": semantic_partition,
        "persisted_partition_manifest": persisted,
        "execution_authority": execution_authority,
        "stages": stage_rows,
        "stage_receipt_inventory_sha256": semantic_sha256(
            tuple(stage.receipt_sha256 for stage in stage_rows)
        ),
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "wall_started_at_ms": wall_started,
        "wall_finished_at_ms": wall_finished,
        "monotonic_started_ns": mono_started,
        "monotonic_finished_ns": mono_finished,
        "observed_full_pipeline_runtime_ms": max(
            1, (mono_finished - mono_started + 999_999) // 1_000_000
        ),
        "compressed_bytes": loaded_source.receipt.content_length,
        "source_row_count": scan.source_row_count,
        "ticker_count": scan.ticker_count,
        "correction_event_count": persisted.correction_event_count,
        "active_event_key_count": persisted.active_event_key_count,
        "security_partition_count": persisted.security_partition_count,
        "daily_feature_row_count": stage_rows[4].output_row_count,
        "rolling_feature_row_count": stage_rows[5].output_row_count,
        "decision_tensor_bytes": stage_rows[
            6
        ].output_loaded_source.receipt.content_length,
    }
    provisional = MassiveArtifactReadinessRunV1(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveArtifactReadinessRunV1(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveArtifactReadinessPanelV1:
    archive_run_receipts: tuple[str, ...]
    selected_run_receipts: tuple[str, ...]
    selected_session_dates: tuple[str, ...]
    selection_spec_sha256: str
    archive_inventory_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_ARTIFACT_READINESS_PANEL_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ARTIFACT_READINESS_PANEL_V1_SCHEMA:
            raise MassiveArtifactReadinessError("artifact panel schema drifted")
        if (
            len(self.selected_run_receipts)
            < MASSIVE_ARTIFACT_READINESS_MINIMUM_SESSIONS_V1
            or self.selected_run_receipts
            != tuple(sorted(set(self.selected_run_receipts)))
            or not set(self.selected_run_receipts).issubset(self.archive_run_receipts)
        ):
            raise MassiveArtifactReadinessError("artifact panel selection differs")
        if len(self.selected_session_dates) != len(
            self.selected_run_receipts
        ) or self.selected_session_dates != tuple(
            sorted(set(self.selected_session_dates))
        ):
            raise MassiveArtifactReadinessError("artifact panel sessions differ")
        if (
            len(
                {
                    date.fromisoformat(value).year
                    for value in self.selected_session_dates
                }
            )
            < MASSIVE_ARTIFACT_READINESS_MINIMUM_YEARS_V1
        ):
            raise MassiveArtifactReadinessError("artifact panel year coverage differs")
        for name in (
            "selection_spec_sha256",
            "archive_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveArtifactReadinessError("artifact panel receipt differs")


MASSIVE_ARTIFACT_READINESS_PANEL_SELECTION_SPEC_SHA256 = semantic_sha256(
    {
        "selection": (
            "top5-compressed-bytes",
            "top5-source-rows",
            "top5-ticker-count",
            "top5-correction-events",
            "deterministic-receipt-order-fill-to-20",
        ),
        "minimum_sessions": MASSIVE_ARTIFACT_READINESS_MINIMUM_SESSIONS_V1,
        "minimum_years": MASSIVE_ARTIFACT_READINESS_MINIMUM_YEARS_V1,
        "manual-replacement": False,
    }
)


def build_massive_artifact_readiness_panel_v1(
    runs: Sequence[MassiveArtifactReadinessRunV1],
) -> MassiveArtifactReadinessPanelV1:
    """Deterministically select a panel from the complete supplied archive."""

    archive = tuple(
        sorted(
            runs, key=lambda row: (row.source_session.session_date, row.receipt_sha256)
        )
    )
    if len(archive) < MASSIVE_ARTIFACT_READINESS_MINIMUM_SESSIONS_V1:
        raise MassiveArtifactReadinessError(
            "artifact archive is smaller than panel minimum"
        )
    for run in archive:
        run.validate()
    listed_source_keys = {
        entry.source_object_key
        for run in archive
        for entry in run.captured_listing.committed_listing.entries
    }
    measured_source_keys = {
        run.loaded_source.receipt.source_object_key for run in archive
    }
    if measured_source_keys != listed_source_keys:
        raise MassiveArtifactReadinessError(
            "artifact archive does not exhaust its committed listing inventories"
        )
    uniqueness = (
        tuple(run.loaded_source.receipt.receipt_sha256 for run in archive),
        tuple(run.scan_evidence.receipt_sha256 for run in archive),
        tuple(run.semantic_partition_manifest.receipt_sha256 for run in archive),
        tuple(run.persisted_partition_manifest.receipt_sha256 for run in archive),
        tuple(run.stage_receipt_inventory_sha256 for run in archive),
    )
    if any(len(set(values)) != len(values) for values in uniqueness):
        raise MassiveArtifactReadinessError("artifact archive contains cloned evidence")
    selected: dict[str, MassiveArtifactReadinessRunV1] = {}
    metrics = (
        "compressed_bytes",
        "source_row_count",
        "ticker_count",
        "correction_event_count",
    )
    for metric in metrics:
        ranked = sorted(
            archive,
            key=lambda row: (-getattr(row, metric), row.receipt_sha256),
        )
        for run in ranked[:5]:
            selected[run.receipt_sha256] = run
    for run in sorted(archive, key=lambda row: row.receipt_sha256):
        if len(selected) >= MASSIVE_ARTIFACT_READINESS_MINIMUM_SESSIONS_V1:
            break
        selected[run.receipt_sha256] = run
    chosen = tuple(
        sorted(selected.values(), key=lambda row: row.source_session.session_date)
    )
    if (
        len({row.source_session.session_date[:4] for row in chosen})
        < MASSIVE_ARTIFACT_READINESS_MINIMUM_YEARS_V1
    ):
        for year in sorted({row.source_session.session_date[:4] for row in archive}):
            candidate = next(
                row
                for row in archive
                if row.source_session.session_date.startswith(year)
            )
            selected[candidate.receipt_sha256] = candidate
        chosen = tuple(
            sorted(selected.values(), key=lambda row: row.source_session.session_date)
        )
    body = {
        "schema": MASSIVE_ARTIFACT_READINESS_PANEL_V1_SCHEMA,
        "archive_run_receipts": tuple(sorted(run.receipt_sha256 for run in archive)),
        "selected_run_receipts": tuple(sorted(run.receipt_sha256 for run in chosen)),
        "selected_session_dates": tuple(
            sorted(run.source_session.session_date for run in chosen)
        ),
        "selection_spec_sha256": MASSIVE_ARTIFACT_READINESS_PANEL_SELECTION_SPEC_SHA256,
        "archive_inventory_sha256": semantic_sha256(
            tuple(run.receipt_sha256 for run in archive)
        ),
    }
    result = MassiveArtifactReadinessPanelV1(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveArtifactReadinessCapabilityV1:
    panel: MassiveArtifactReadinessPanelV1
    runs: tuple[MassiveArtifactReadinessRunV1, ...]
    execution_authority_receipt_sha256: str
    maximum_workload: tuple[tuple[str, int], ...]
    observed_runtime_ms: tuple[int, ...]
    p95_runtime_ms: int
    p99_runtime_ms: int
    maximum_runtime_ms: int
    allowed_processing_ms: int
    capability_passed: bool
    receipt_sha256: str
    schema: str = MASSIVE_ARTIFACT_READINESS_CAPABILITY_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ARTIFACT_READINESS_CAPABILITY_V1_SCHEMA:
            raise MassiveArtifactReadinessError("artifact capability schema drifted")
        self.panel.validate()
        for run in self.runs:
            run.validate()
        if (
            tuple(sorted(run.receipt_sha256 for run in self.runs))
            != self.panel.selected_run_receipts
        ):
            raise MassiveArtifactReadinessError(
                "artifact capability runs differ from panel"
            )
        execution_receipts = {
            run.execution_authority.receipt_sha256 for run in self.runs
        }
        if execution_receipts != {self.execution_authority_receipt_sha256}:
            raise MassiveArtifactReadinessError(
                "artifact capability execution authorities differ"
            )
        expected_workload = _maximum_workload(self.runs)
        if self.maximum_workload != expected_workload:
            raise MassiveArtifactReadinessError("artifact workload envelope differs")
        runtimes = tuple(
            sorted(run.observed_full_pipeline_runtime_ms for run in self.runs)
        )
        if self.observed_runtime_ms != runtimes:
            raise MassiveArtifactReadinessError("artifact runtime inventory differs")
        if (
            self.p95_runtime_ms != _nearest_rank(runtimes, 0.95)
            or self.p99_runtime_ms != _nearest_rank(runtimes, 0.99)
            or self.maximum_runtime_ms != max(runtimes)
            or self.allowed_processing_ms != MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS
        ):
            raise MassiveArtifactReadinessError("artifact runtime aggregates differ")
        expected_pass = self.maximum_runtime_ms <= self.allowed_processing_ms
        if self.capability_passed is not expected_pass:
            raise MassiveArtifactReadinessError("artifact capability outcome differs")
        _digest("artifact capability receipt", self.receipt_sha256)
        _digest(
            "artifact capability execution authority",
            self.execution_authority_receipt_sha256,
        )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveArtifactReadinessError("artifact capability receipt differs")

    @property
    def readiness_authorizing(self) -> bool:
        return self.capability_passed

    def covers(self, run: MassiveArtifactReadinessRunV1) -> bool:
        self.validate()
        run.validate()
        limits = dict(self.maximum_workload)
        return (
            self.capability_passed
            and run.execution_authority.receipt_sha256
            == self.execution_authority_receipt_sha256
            and all(getattr(run, name) <= limit for name, limit in limits.items())
        )


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


_WORKLOAD_FIELDS = (
    "compressed_bytes",
    "source_row_count",
    "ticker_count",
    "correction_event_count",
    "active_event_key_count",
    "security_partition_count",
    "daily_feature_row_count",
    "rolling_feature_row_count",
    "decision_tensor_bytes",
)


def _maximum_workload(
    runs: Sequence[MassiveArtifactReadinessRunV1],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, max(getattr(run, name) for run in runs)) for name in _WORKLOAD_FIELDS
    )


def build_massive_artifact_readiness_capability_v1(
    *,
    panel: MassiveArtifactReadinessPanelV1,
    archive_runs: Sequence[MassiveArtifactReadinessRunV1],
) -> MassiveArtifactReadinessCapabilityV1:
    panel.validate()
    expected_panel = build_massive_artifact_readiness_panel_v1(archive_runs)
    if panel != expected_panel:
        raise MassiveArtifactReadinessError(
            "capability panel was not deterministically derived"
        )
    by_receipt = {run.receipt_sha256: run for run in archive_runs}
    if set(by_receipt) != set(panel.archive_run_receipts):
        raise MassiveArtifactReadinessError(
            "capability did not receive the complete archive"
        )
    selected = tuple(by_receipt[value] for value in panel.selected_run_receipts)
    for run in selected:
        run.validate()
    runtimes = tuple(sorted(run.observed_full_pipeline_runtime_ms for run in selected))
    execution_receipts = {run.execution_authority.receipt_sha256 for run in selected}
    if len(execution_receipts) != 1:
        raise MassiveArtifactReadinessError(
            "artifact capability execution authorities differ"
        )
    body = {
        "schema": MASSIVE_ARTIFACT_READINESS_CAPABILITY_V1_SCHEMA,
        "panel": panel,
        "runs": selected,
        "execution_authority_receipt_sha256": next(iter(execution_receipts)),
        "maximum_workload": _maximum_workload(selected),
        "observed_runtime_ms": runtimes,
        "p95_runtime_ms": _nearest_rank(runtimes, 0.95),
        "p99_runtime_ms": _nearest_rank(runtimes, 0.99),
        "maximum_runtime_ms": max(runtimes),
        "allowed_processing_ms": MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
        "capability_passed": max(runtimes) <= MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
    }
    provisional = MassiveArtifactReadinessCapabilityV1(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveArtifactReadinessCapabilityV1(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ARTIFACT_EXECUTION_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ARTIFACT_EXECUTION_DATASET_V1",
    "MASSIVE_ARTIFACT_EXECUTION_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ARTIFACT_READINESS_CAPABILITY_V1_SCHEMA",
    "MASSIVE_ARTIFACT_READINESS_PANEL_SELECTION_SPEC_SHA256",
    "MASSIVE_ARTIFACT_READINESS_PANEL_V1_SCHEMA",
    "MASSIVE_ARTIFACT_READINESS_RUN_V1_SCHEMA",
    "MASSIVE_ARTIFACT_READINESS_STAGE_CONTRACTS_V1",
    "MASSIVE_ARTIFACT_READINESS_STAGE_IDS_V1",
    "MASSIVE_ARTIFACT_READINESS_STAGE_V1_SCHEMA",
    "MassiveArtifactExecutionAuthorityV1",
    "MassiveArtifactReadinessCapabilityV1",
    "MassiveArtifactReadinessError",
    "MassiveArtifactReadinessPanelV1",
    "MassiveArtifactReadinessRunV1",
    "MassiveArtifactReadinessStageV1",
    "build_massive_artifact_readiness_capability_v1",
    "build_massive_artifact_readiness_panel_v1",
    "measure_massive_artifact_readiness_v1",
    "parse_massive_artifact_execution_authority_v1",
    "validate_massive_artifact_readiness_stage_bytes_v1",
]
