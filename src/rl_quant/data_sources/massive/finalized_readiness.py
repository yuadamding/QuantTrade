"""Measured full feature-to-order readiness for finalized validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
import time
from typing import Callable, Sequence

from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionManifestV0,
)
from rl_quant.data_sources.massive.source_receipts import LoadedMassiveSourceObject
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0 = (
    "persisted-trade-partitions",
    "daily-features",
    "rolling-features",
    "pit500-decision-tensor",
    "frozen-model-inference",
    "requested-orders",
)
MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0 = 20
MASSIVE_FINALIZED_MINIMUM_READINESS_YEARS_V0 = 3
MASSIVE_FINALIZED_READINESS_STAGE_ARTIFACT_V0_SCHEMA = (
    "rl-quant.massive-finalized-readiness-stage-artifact-v0"
)
MASSIVE_MEASURED_FINALIZED_READINESS_RUN_V0_SCHEMA = (
    "rl-quant.massive-measured-finalized-readiness-run-v0"
)
MASSIVE_FINALIZED_READINESS_PANEL_SPEC_V0_SCHEMA = (
    "rl-quant.massive-finalized-readiness-panel-spec-v0"
)
MASSIVE_FINALIZED_READINESS_CAPABILITY_V0_SCHEMA = (
    "rl-quant.massive-finalized-readiness-capability-v0"
)


class MassiveFinalizedReadinessError(ValueError):
    """A full finalized feature-to-order readiness claim is invalid."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFinalizedReadinessError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveFinalizedReadinessError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedReadinessStageArtifactV0:
    stage_id: str
    input_artifact_receipts: tuple[str, ...]
    output_loaded_source: LoadedMassiveSourceObject
    output_artifact_receipt_sha256: str
    output_commit_receipt_sha256: str
    implementation_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_READINESS_STAGE_ARTIFACT_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_READINESS_STAGE_ARTIFACT_V0_SCHEMA:
            raise MassiveFinalizedReadinessError("readiness stage schema drifted")
        if self.stage_id not in MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0:
            raise MassiveFinalizedReadinessError("readiness stage identity drifted")
        if (
            not self.input_artifact_receipts
            or self.input_artifact_receipts
            != tuple(sorted(set(self.input_artifact_receipts)))
        ):
            raise MassiveFinalizedReadinessError(
                "readiness stage inputs must be sorted and unique"
            )
        for value in self.input_artifact_receipts:
            _digest("readiness stage input", value)
        self.output_loaded_source.validate()
        for name in (
            "output_artifact_receipt_sha256",
            "output_commit_receipt_sha256",
            "implementation_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.output_artifact_receipt_sha256
            != self.output_loaded_source.receipt.receipt_sha256
            or self.output_commit_receipt_sha256
            != self.output_loaded_source.commit.receipt_sha256
        ):
            raise MassiveFinalizedReadinessError(
                "readiness stage output is not bound to a committed artifact"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedReadinessError("readiness stage receipt differs")


def build_massive_finalized_readiness_stage_artifact_v0(
    *,
    stage_id: str,
    input_artifact_receipts: Sequence[str],
    output_loaded_source: LoadedMassiveSourceObject,
    implementation_source_sha256: str,
) -> MassiveFinalizedReadinessStageArtifactV0:
    """Bind one committed output in the measured full-readiness chain."""

    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_READINESS_STAGE_ARTIFACT_V0_SCHEMA,
        "stage_id": stage_id,
        "input_artifact_receipts": tuple(sorted(input_artifact_receipts)),
        "output_loaded_source": output_loaded_source,
        "output_artifact_receipt_sha256": (
            output_loaded_source.receipt.receipt_sha256
        ),
        "output_commit_receipt_sha256": output_loaded_source.commit.receipt_sha256,
        "implementation_source_sha256": implementation_source_sha256,
    }
    provisional = MassiveFinalizedReadinessStageArtifactV0(
        **body, receipt_sha256="0" * 64  # type: ignore[arg-type]
    )
    result = MassiveFinalizedReadinessStageArtifactV0(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveMeasuredFinalizedReadinessRunV0:
    source_session_date: str
    source_file_scan_receipt_sha256: str
    partition_manifest_receipt_sha256: str
    source_object_receipt_sha256: str
    listing_acquisition_receipt_sha256: str
    hardware_contract_receipt_sha256: str
    software_commit_sha256: str
    pipeline_implementation_source_sha256: str
    stages: tuple[MassiveFinalizedReadinessStageArtifactV0, ...]
    stage_receipt_inventory_sha256: str
    wall_started_at_ms: int
    wall_finished_at_ms: int
    monotonic_started_ns: int
    monotonic_finished_ns: int
    observed_full_pipeline_runtime_ms: int
    compressed_bytes: int
    source_row_count: int
    ticker_count: int
    post_close_correction_row_count: int
    measurement_kind: str
    receipt_sha256: str
    schema: str = MASSIVE_MEASURED_FINALIZED_READINESS_RUN_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_MEASURED_FINALIZED_READINESS_RUN_V0_SCHEMA:
            raise MassiveFinalizedReadinessError("measured readiness schema drifted")
        try:
            date.fromisoformat(self.source_session_date)
        except (TypeError, ValueError) as exc:
            raise MassiveFinalizedReadinessError(
                "measured readiness session is invalid"
            ) from exc
        for name in (
            "source_file_scan_receipt_sha256",
            "partition_manifest_receipt_sha256",
            "source_object_receipt_sha256",
            "listing_acquisition_receipt_sha256",
            "hardware_contract_receipt_sha256",
            "software_commit_sha256",
            "pipeline_implementation_source_sha256",
            "stage_receipt_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in (
            "wall_started_at_ms",
            "wall_finished_at_ms",
            "monotonic_started_ns",
            "monotonic_finished_ns",
            "observed_full_pipeline_runtime_ms",
            "compressed_bytes",
            "source_row_count",
            "ticker_count",
        ):
            _positive(name, getattr(self, name))
        if (
            isinstance(self.post_close_correction_row_count, bool)
            or not isinstance(self.post_close_correction_row_count, int)
            or self.post_close_correction_row_count < 0
        ):
            raise MassiveFinalizedReadinessError(
                "measured correction count is invalid"
            )
        if (
            self.wall_finished_at_ms < self.wall_started_at_ms
            or self.monotonic_finished_ns <= self.monotonic_started_ns
        ):
            raise MassiveFinalizedReadinessError(
                "measured readiness chronology is invalid"
            )
        expected_runtime = max(
            1,
            (
                self.monotonic_finished_ns
                - self.monotonic_started_ns
                + 999_999
            )
            // 1_000_000,
        )
        if self.observed_full_pipeline_runtime_ms != expected_runtime:
            raise MassiveFinalizedReadinessError(
                "measured readiness runtime was not monotonic-clock-derived"
            )
        if self.measurement_kind != "package-monotonic-full-feature-to-order":
            raise MassiveFinalizedReadinessError("readiness measurement kind drifted")
        if tuple(stage.stage_id for stage in self.stages) != (
            MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0
        ):
            raise MassiveFinalizedReadinessError("readiness stage inventory drifted")
        expected_inputs = (self.partition_manifest_receipt_sha256,)
        for stage in self.stages:
            stage.validate()
            if stage.input_artifact_receipts != expected_inputs:
                raise MassiveFinalizedReadinessError(
                    "readiness stage chain is discontinuous"
                )
            expected_inputs = (stage.output_artifact_receipt_sha256,)
        if self.stage_receipt_inventory_sha256 != semantic_sha256(
            tuple(stage.receipt_sha256 for stage in self.stages)
        ):
            raise MassiveFinalizedReadinessError(
                "readiness stage receipt inventory differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedReadinessError("measured readiness receipt differs")


ReadinessStageRunner = Callable[
    [str, tuple[str, ...]], MassiveFinalizedReadinessStageArtifactV0
]


def measure_massive_finalized_full_readiness_v0(
    *,
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    partition_manifest: MassiveDailyTradePartitionManifestV0,
    listing_acquisition_receipt_sha256: str,
    hardware_contract_receipt_sha256: str,
    software_commit_sha256: str,
    pipeline_implementation_source_sha256: str,
    stage_runner: ReadinessStageRunner,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wall_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
) -> MassiveMeasuredFinalizedReadinessRunV0:
    """Time the exact persisted-partition through requested-order sequence."""

    scan_evidence.validate()
    partition_manifest.validate()
    if partition_manifest.source_file_scan_receipt_sha256 != scan_evidence.receipt_sha256:
        raise MassiveFinalizedReadinessError(
            "readiness scan and partition evidence differ"
        )
    acquisition = _digest(
        "listing acquisition receipt", listing_acquisition_receipt_sha256
    )
    hardware = _digest("hardware contract receipt", hardware_contract_receipt_sha256)
    commit = _digest("software commit", software_commit_sha256)
    implementation = _digest(
        "pipeline implementation source", pipeline_implementation_source_sha256
    )
    wall_started = wall_ms()
    monotonic_started = monotonic_ns()
    inputs = (partition_manifest.receipt_sha256,)
    stages: list[MassiveFinalizedReadinessStageArtifactV0] = []
    for stage_id in MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0:
        stage = stage_runner(stage_id, inputs)
        stage.validate()
        if stage.stage_id != stage_id or stage.input_artifact_receipts != inputs:
            raise MassiveFinalizedReadinessError(
                "readiness runner returned a different stage or input chain"
            )
        stages.append(stage)
        inputs = (stage.output_artifact_receipt_sha256,)
    monotonic_finished = monotonic_ns()
    wall_finished = wall_ms()
    stage_rows = tuple(stages)
    body: dict[str, object] = {
        "schema": MASSIVE_MEASURED_FINALIZED_READINESS_RUN_V0_SCHEMA,
        "source_session_date": scan_evidence.source_session_date,
        "source_file_scan_receipt_sha256": scan_evidence.receipt_sha256,
        "partition_manifest_receipt_sha256": partition_manifest.receipt_sha256,
        "source_object_receipt_sha256": scan_evidence.source_object_receipt_sha256,
        "listing_acquisition_receipt_sha256": acquisition,
        "hardware_contract_receipt_sha256": hardware,
        "software_commit_sha256": commit,
        "pipeline_implementation_source_sha256": implementation,
        "stages": stage_rows,
        "stage_receipt_inventory_sha256": semantic_sha256(
            tuple(stage.receipt_sha256 for stage in stage_rows)
        ),
        "wall_started_at_ms": wall_started,
        "wall_finished_at_ms": wall_finished,
        "monotonic_started_ns": monotonic_started,
        "monotonic_finished_ns": monotonic_finished,
        "observed_full_pipeline_runtime_ms": max(
            1, (monotonic_finished - monotonic_started + 999_999) // 1_000_000
        ),
        "compressed_bytes": scan_evidence.compressed_bytes,
        "source_row_count": scan_evidence.source_row_count,
        "ticker_count": scan_evidence.ticker_count,
        "post_close_correction_row_count": (
            scan_evidence.post_close_correction_row_count
        ),
        "measurement_kind": "package-monotonic-full-feature-to-order",
    }
    provisional = MassiveMeasuredFinalizedReadinessRunV0(
        **body, receipt_sha256="0" * 64  # type: ignore[arg-type]
    )
    result = MassiveMeasuredFinalizedReadinessRunV0(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveFinalizedReadinessPanelSpecV0:
    source_session_dates: tuple[str, ...]
    largest_compressed_source_receipt_sha256: str
    largest_row_count_source_receipt_sha256: str
    correction_activity_session_dates: tuple[str, ...]
    high_ticker_count_session_dates: tuple[str, ...]
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_READINESS_PANEL_SPEC_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_READINESS_PANEL_SPEC_V0_SCHEMA:
            raise MassiveFinalizedReadinessError("readiness panel schema drifted")
        if (
            len(self.source_session_dates)
            < MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0
            or self.source_session_dates
            != tuple(sorted(set(self.source_session_dates)))
        ):
            raise MassiveFinalizedReadinessError(
                "readiness panel does not contain the frozen session minimum"
            )
        years: set[int] = set()
        try:
            years = {date.fromisoformat(value).year for value in self.source_session_dates}
        except (TypeError, ValueError) as exc:
            raise MassiveFinalizedReadinessError(
                "readiness panel session inventory is invalid"
            ) from exc
        if len(years) < MASSIVE_FINALIZED_MINIMUM_READINESS_YEARS_V0:
            raise MassiveFinalizedReadinessError(
                "readiness panel does not span the frozen year minimum"
            )
        for name in (
            "largest_compressed_source_receipt_sha256",
            "largest_row_count_source_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        sessions = set(self.source_session_dates)
        for inventory_name in (
            "correction_activity_session_dates",
            "high_ticker_count_session_dates",
        ):
            inventory = getattr(self, inventory_name)
            if (
                not inventory
                or inventory != tuple(sorted(set(inventory)))
                or not set(inventory).issubset(sessions)
            ):
                raise MassiveFinalizedReadinessError(
                    f"{inventory_name} is not a canonical panel subset"
                )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedReadinessError("readiness panel receipt differs")


def build_massive_finalized_readiness_panel_spec_v0(
    *,
    source_session_dates: Sequence[str],
    largest_compressed_source_receipt_sha256: str,
    largest_row_count_source_receipt_sha256: str,
    correction_activity_session_dates: Sequence[str],
    high_ticker_count_session_dates: Sequence[str],
) -> MassiveFinalizedReadinessPanelSpecV0:
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_READINESS_PANEL_SPEC_V0_SCHEMA,
        "source_session_dates": tuple(sorted(source_session_dates)),
        "largest_compressed_source_receipt_sha256": (
            largest_compressed_source_receipt_sha256
        ),
        "largest_row_count_source_receipt_sha256": (
            largest_row_count_source_receipt_sha256
        ),
        "correction_activity_session_dates": tuple(
            sorted(correction_activity_session_dates)
        ),
        "high_ticker_count_session_dates": tuple(
            sorted(high_ticker_count_session_dates)
        ),
    }
    result = MassiveFinalizedReadinessPanelSpecV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


@dataclass(frozen=True, slots=True)
class MassiveFinalizedReadinessCapabilityV0:
    panel_spec: MassiveFinalizedReadinessPanelSpecV0
    panel_spec_receipt_sha256: str
    runs: tuple[MassiveMeasuredFinalizedReadinessRunV0, ...]
    run_receipts: tuple[str, ...]
    hardware_contract_receipt_sha256: str
    software_commit_sha256: str
    pipeline_implementation_source_sha256: str
    maximum_compressed_bytes: int
    maximum_source_rows: int
    observed_runtime_ms: tuple[int, ...]
    p95_runtime_ms: int
    p99_runtime_ms: int
    maximum_runtime_ms: int
    allowed_processing_ms: int
    capability_scope: str
    representative_panel_passed: bool
    capability_passed: bool
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_READINESS_CAPABILITY_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_READINESS_CAPABILITY_V0_SCHEMA:
            raise MassiveFinalizedReadinessError("readiness capability schema drifted")
        self.panel_spec.validate()
        if self.panel_spec_receipt_sha256 != self.panel_spec.receipt_sha256:
            raise MassiveFinalizedReadinessError("readiness panel receipt differs")
        if not self.runs:
            raise MassiveFinalizedReadinessError("readiness runs are absent")
        for run in self.runs:
            run.validate()
        if tuple(run.source_session_date for run in self.runs) != (
            self.panel_spec.source_session_dates
        ):
            raise MassiveFinalizedReadinessError(
                "readiness runs do not exhaust the predeclared panel"
            )
        expected_receipts = tuple(run.receipt_sha256 for run in self.runs)
        source_receipts = tuple(
            run.source_object_receipt_sha256 for run in self.runs
        )
        if (
            self.run_receipts != expected_receipts
            or len(set(self.run_receipts)) != len(self.run_receipts)
            or len(set(source_receipts)) != len(source_receipts)
        ):
            raise MassiveFinalizedReadinessError("readiness run receipts differ")
        identities = {
            (
                run.hardware_contract_receipt_sha256,
                run.software_commit_sha256,
                run.pipeline_implementation_source_sha256,
            )
            for run in self.runs
        }
        if len(identities) != 1:
            raise MassiveFinalizedReadinessError(
                "readiness run implementations differ"
            )
        identity = next(iter(identities))
        if identity != (
            self.hardware_contract_receipt_sha256,
            self.software_commit_sha256,
            self.pipeline_implementation_source_sha256,
        ):
            raise MassiveFinalizedReadinessError(
                "readiness capability implementation differs"
            )
        runtimes = tuple(sorted(run.observed_full_pipeline_runtime_ms for run in self.runs))
        if self.observed_runtime_ms != runtimes:
            raise MassiveFinalizedReadinessError("readiness runtime inventory differs")
        if (
            self.maximum_compressed_bytes != max(run.compressed_bytes for run in self.runs)
            or self.maximum_source_rows != max(run.source_row_count for run in self.runs)
            or self.maximum_runtime_ms != max(runtimes)
            or self.p95_runtime_ms != _nearest_rank(runtimes, 0.95)
            or self.p99_runtime_ms != _nearest_rank(runtimes, 0.99)
        ):
            raise MassiveFinalizedReadinessError(
                "readiness capability aggregates differ"
            )
        if self.allowed_processing_ms != MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS:
            raise MassiveFinalizedReadinessError("readiness allowance drifted")
        if self.capability_scope != "full-feature-to-order-readiness":
            raise MassiveFinalizedReadinessError("readiness capability scope drifted")
        largest_compressed = max(self.runs, key=lambda run: run.compressed_bytes)
        largest_rows = max(self.runs, key=lambda run: run.source_row_count)
        maximum_ticker_count = max(run.ticker_count for run in self.runs)
        run_by_date = {run.source_session_date: run for run in self.runs}
        representative = (
            largest_compressed.source_object_receipt_sha256
            == self.panel_spec.largest_compressed_source_receipt_sha256
            and largest_rows.source_object_receipt_sha256
            == self.panel_spec.largest_row_count_source_receipt_sha256
            and all(
                run_by_date[value].post_close_correction_row_count > 0
                for value in self.panel_spec.correction_activity_session_dates
            )
            and all(
                run_by_date[value].ticker_count == maximum_ticker_count
                for value in self.panel_spec.high_ticker_count_session_dates
            )
        )
        if self.representative_panel_passed is not representative:
            raise MassiveFinalizedReadinessError(
                "readiness representative-panel outcome differs"
            )
        expected_pass = representative and self.maximum_runtime_ms <= self.allowed_processing_ms
        if self.capability_passed is not expected_pass:
            raise MassiveFinalizedReadinessError("readiness capability outcome differs")
        for name in (
            "panel_spec_receipt_sha256",
            "hardware_contract_receipt_sha256",
            "software_commit_sha256",
            "pipeline_implementation_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedReadinessError("readiness capability receipt differs")

    def covers(self, scan: MassiveDailyTradeFileScanEvidenceV0) -> bool:
        self.validate()
        scan.validate()
        return (
            self.capability_passed
            and scan.compressed_bytes <= self.maximum_compressed_bytes
            and scan.source_row_count <= self.maximum_source_rows
        )


def build_massive_finalized_readiness_capability_v0(
    *,
    panel_spec: MassiveFinalizedReadinessPanelSpecV0,
    runs: Sequence[MassiveMeasuredFinalizedReadinessRunV0],
) -> MassiveFinalizedReadinessCapabilityV0:
    panel_spec.validate()
    rows = tuple(sorted(runs, key=lambda run: run.source_session_date))
    if not rows:
        raise MassiveFinalizedReadinessError("readiness runs are absent")
    for run in rows:
        run.validate()
    if len({run.source_object_receipt_sha256 for run in rows}) != len(rows):
        raise MassiveFinalizedReadinessError(
            "readiness runs must use distinct committed source objects"
        )
    identities = {
        (
            run.hardware_contract_receipt_sha256,
            run.software_commit_sha256,
            run.pipeline_implementation_source_sha256,
        )
        for run in rows
    }
    if len(identities) != 1:
        raise MassiveFinalizedReadinessError("readiness run implementations differ")
    hardware, commit, implementation = next(iter(identities))
    runtimes = tuple(sorted(run.observed_full_pipeline_runtime_ms for run in rows))
    largest_compressed = max(rows, key=lambda run: run.compressed_bytes)
    largest_rows = max(rows, key=lambda run: run.source_row_count)
    maximum_ticker_count = max(run.ticker_count for run in rows)
    run_by_date = {run.source_session_date: run for run in rows}
    representative = (
        tuple(run_by_date) == panel_spec.source_session_dates
        and largest_compressed.source_object_receipt_sha256
        == panel_spec.largest_compressed_source_receipt_sha256
        and largest_rows.source_object_receipt_sha256
        == panel_spec.largest_row_count_source_receipt_sha256
        and all(
            run_by_date[value].post_close_correction_row_count > 0
            for value in panel_spec.correction_activity_session_dates
        )
        and all(
            run_by_date[value].ticker_count == maximum_ticker_count
            for value in panel_spec.high_ticker_count_session_dates
        )
    )
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_READINESS_CAPABILITY_V0_SCHEMA,
        "panel_spec": panel_spec,
        "panel_spec_receipt_sha256": panel_spec.receipt_sha256,
        "runs": rows,
        "run_receipts": tuple(run.receipt_sha256 for run in rows),
        "hardware_contract_receipt_sha256": hardware,
        "software_commit_sha256": commit,
        "pipeline_implementation_source_sha256": implementation,
        "maximum_compressed_bytes": max(run.compressed_bytes for run in rows),
        "maximum_source_rows": max(run.source_row_count for run in rows),
        "observed_runtime_ms": runtimes,
        "p95_runtime_ms": _nearest_rank(runtimes, 0.95),
        "p99_runtime_ms": _nearest_rank(runtimes, 0.99),
        "maximum_runtime_ms": max(runtimes),
        "allowed_processing_ms": MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
        "capability_scope": "full-feature-to-order-readiness",
        "representative_panel_passed": representative,
        "capability_passed": (
            representative
            and max(runtimes) <= MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS
        ),
    }
    provisional = MassiveFinalizedReadinessCapabilityV0(
        **body, receipt_sha256="0" * 64  # type: ignore[arg-type]
    )
    result = MassiveFinalizedReadinessCapabilityV0(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_FINALIZED_MINIMUM_READINESS_SESSIONS_V0",
    "MASSIVE_FINALIZED_MINIMUM_READINESS_YEARS_V0",
    "MASSIVE_FINALIZED_READINESS_CAPABILITY_V0_SCHEMA",
    "MASSIVE_FINALIZED_READINESS_PANEL_SPEC_V0_SCHEMA",
    "MASSIVE_FINALIZED_READINESS_STAGE_ARTIFACT_V0_SCHEMA",
    "MASSIVE_FINALIZED_READINESS_STAGE_IDS_V0",
    "MASSIVE_MEASURED_FINALIZED_READINESS_RUN_V0_SCHEMA",
    "MassiveFinalizedReadinessCapabilityV0",
    "MassiveFinalizedReadinessError",
    "MassiveFinalizedReadinessPanelSpecV0",
    "MassiveFinalizedReadinessStageArtifactV0",
    "MassiveMeasuredFinalizedReadinessRunV0",
    "build_massive_finalized_readiness_capability_v0",
    "build_massive_finalized_readiness_panel_spec_v0",
    "build_massive_finalized_readiness_stage_artifact_v0",
    "measure_massive_finalized_full_readiness_v0",
]
