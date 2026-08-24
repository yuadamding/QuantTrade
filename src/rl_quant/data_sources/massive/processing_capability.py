"""Measured source-scan and partition capability for finalized validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256,
    MassiveDailyTradeFileScanEvidenceV0,
    scan_massive_daily_trade_file_v0,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256,
    MassiveDailyTradePartitionManifestV0,
    MassiveFinalizedFeatureDomainSpecV0,
    build_massive_daily_trade_partition_manifest_v0,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import LoadedMassiveSourceObject
from rl_quant.data_sources.massive.trade_extraction import MassiveExtractedTradeRow
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_FINALIZED_PROCESSING_BENCHMARK_V0_SCHEMA = (
    "rl-quant.massive-finalized-processing-benchmark-v0"
)
MASSIVE_FINALIZED_PROCESSING_CAPABILITY_V0_SCHEMA = (
    "rl-quant.massive-finalized-processing-capability-v0"
)


class MassiveFinalizedProcessingCapabilityError(ValueError):
    """Measured finalized-source processing does not meet the frozen budget."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFinalizedProcessingCapabilityError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveFinalizedProcessingCapabilityError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedProcessingBenchmarkV0:
    source_file_scan_receipt_sha256: str
    partition_manifest_receipt_sha256: str
    source_object_receipt_sha256: str
    hardware_contract_receipt_sha256: str
    software_commit_sha256: str
    parser_source_sha256: str
    feature_materializer_source_sha256: str
    compressed_bytes: int
    decompressed_rows: int
    observed_scan_runtime_ms: int
    observed_partition_runtime_ms: int
    observed_total_runtime_ms: int
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_PROCESSING_BENCHMARK_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_PROCESSING_BENCHMARK_V0_SCHEMA:
            raise MassiveFinalizedProcessingCapabilityError("benchmark schema drifted")
        for name in (
            "source_file_scan_receipt_sha256",
            "partition_manifest_receipt_sha256",
            "source_object_receipt_sha256",
            "hardware_contract_receipt_sha256",
            "software_commit_sha256",
            "parser_source_sha256",
            "feature_materializer_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in (
            "compressed_bytes",
            "decompressed_rows",
            "observed_scan_runtime_ms",
            "observed_partition_runtime_ms",
            "observed_total_runtime_ms",
        ):
            _positive(name, getattr(self, name))
        if self.parser_source_sha256 != MASSIVE_DAILY_TRADE_FILE_SCAN_SOURCE_SHA256:
            raise MassiveFinalizedProcessingCapabilityError("benchmark parser drifted")
        if (
            self.feature_materializer_source_sha256
            != MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256
        ):
            raise MassiveFinalizedProcessingCapabilityError("benchmark partitioner drifted")
        if self.observed_total_runtime_ms != (
            self.observed_scan_runtime_ms + self.observed_partition_runtime_ms
        ):
            raise MassiveFinalizedProcessingCapabilityError("benchmark runtimes differ")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedProcessingCapabilityError("benchmark receipt differs")


def build_massive_finalized_processing_benchmark_v0(
    *,
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    partition_manifest: MassiveDailyTradePartitionManifestV0,
    hardware_contract_receipt_sha256: str,
    software_commit_sha256: str,
    observed_scan_runtime_ms: int,
    observed_partition_runtime_ms: int,
) -> MassiveFinalizedProcessingBenchmarkV0:
    scan_evidence.validate()
    partition_manifest.validate()
    hardware = _digest("hardware contract receipt", hardware_contract_receipt_sha256)
    commit = _digest("software commit", software_commit_sha256)
    scan_runtime = _positive("observed scan runtime", observed_scan_runtime_ms)
    partition_runtime = _positive(
        "observed partition runtime", observed_partition_runtime_ms
    )
    if partition_manifest.source_file_scan_receipt_sha256 != scan_evidence.receipt_sha256:
        raise MassiveFinalizedProcessingCapabilityError("benchmark artifacts differ")
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_PROCESSING_BENCHMARK_V0_SCHEMA,
        "source_file_scan_receipt_sha256": scan_evidence.receipt_sha256,
        "partition_manifest_receipt_sha256": partition_manifest.receipt_sha256,
        "source_object_receipt_sha256": scan_evidence.source_object_receipt_sha256,
        "hardware_contract_receipt_sha256": hardware,
        "software_commit_sha256": commit,
        "parser_source_sha256": scan_evidence.parser_source_sha256,
        "feature_materializer_source_sha256": partition_manifest.partition_source_sha256,
        "compressed_bytes": scan_evidence.compressed_bytes,
        "decompressed_rows": scan_evidence.source_row_count,
        "observed_scan_runtime_ms": scan_runtime,
        "observed_partition_runtime_ms": partition_runtime,
        "observed_total_runtime_ms": scan_runtime + partition_runtime,
    }
    result = MassiveFinalizedProcessingBenchmarkV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def measure_massive_finalized_source_processing_v0(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    session_authority: MassiveSessionAuthority,
    session: MassiveExchangeSession,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    hardware_contract_receipt_sha256: str,
    software_commit_sha256: str,
) -> tuple[
    tuple[MassiveExtractedTradeRow, ...],
    MassiveDailyTradeFileScanEvidenceV0,
    MassiveDailyTradePartitionManifestV0,
    MassiveFinalizedProcessingBenchmarkV0,
]:
    """Measure parsing/partitioning without contaminating deterministic receipts."""

    scan_started = time.perf_counter_ns()
    rows, scan = scan_massive_daily_trade_file_v0(
        root=root,
        loaded_source=loaded_source,
        session_authority=session_authority,
        session=session,
        correction_authority=correction_authority,
    )
    scan_runtime = max(1, (time.perf_counter_ns() - scan_started + 999_999) // 1_000_000)
    partition_started = time.perf_counter_ns()
    partition = build_massive_daily_trade_partition_manifest_v0(
        rows=rows,
        scan_evidence=scan,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        feature_domain_spec=feature_domain_spec,
    )
    partition_runtime = max(
        1, (time.perf_counter_ns() - partition_started + 999_999) // 1_000_000
    )
    benchmark = build_massive_finalized_processing_benchmark_v0(
        scan_evidence=scan,
        partition_manifest=partition,
        hardware_contract_receipt_sha256=hardware_contract_receipt_sha256,
        software_commit_sha256=software_commit_sha256,
        observed_scan_runtime_ms=scan_runtime,
        observed_partition_runtime_ms=partition_runtime,
    )
    return rows, scan, partition, benchmark


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


@dataclass(frozen=True, slots=True)
class MassiveFinalizedProcessingCapabilityV0:
    hardware_contract_receipt_sha256: str
    software_commit_sha256: str
    parser_source_sha256: str
    feature_materializer_source_sha256: str
    benchmark_receipts: tuple[str, ...]
    benchmark_source_receipts: tuple[str, ...]
    maximum_compressed_bytes: int
    maximum_decompressed_rows: int
    observed_runtime_ms: tuple[int, ...]
    p95_runtime_ms: int
    p99_runtime_ms: int
    maximum_runtime_ms: int
    allowed_processing_ms: int
    capability_scope: str
    capability_passed: bool
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_PROCESSING_CAPABILITY_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_PROCESSING_CAPABILITY_V0_SCHEMA:
            raise MassiveFinalizedProcessingCapabilityError("capability schema drifted")
        for name in (
            "hardware_contract_receipt_sha256",
            "software_commit_sha256",
            "parser_source_sha256",
            "feature_materializer_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for inventory_name in ("benchmark_receipts", "benchmark_source_receipts"):
            inventory = getattr(self, inventory_name)
            if not inventory or inventory != tuple(sorted(set(inventory))):
                raise MassiveFinalizedProcessingCapabilityError(
                    f"{inventory_name} must be sorted and unique"
                )
            for value in inventory:
                _digest(inventory_name, value)
        if len(self.benchmark_receipts) != len(self.benchmark_source_receipts):
            raise MassiveFinalizedProcessingCapabilityError("benchmark inventories differ")
        for name in (
            "maximum_compressed_bytes",
            "maximum_decompressed_rows",
            "p95_runtime_ms",
            "p99_runtime_ms",
            "maximum_runtime_ms",
            "allowed_processing_ms",
        ):
            _positive(name, getattr(self, name))
        if not self.observed_runtime_ms or self.observed_runtime_ms != tuple(sorted(self.observed_runtime_ms)):
            raise MassiveFinalizedProcessingCapabilityError("runtime inventory is not sorted")
        if any(value <= 0 for value in self.observed_runtime_ms):
            raise MassiveFinalizedProcessingCapabilityError("runtime inventory is invalid")
        if self.maximum_runtime_ms != max(self.observed_runtime_ms):
            raise MassiveFinalizedProcessingCapabilityError("maximum runtime differs")
        if self.p95_runtime_ms != _nearest_rank(self.observed_runtime_ms, 0.95):
            raise MassiveFinalizedProcessingCapabilityError("p95 runtime differs")
        if self.p99_runtime_ms != _nearest_rank(self.observed_runtime_ms, 0.99):
            raise MassiveFinalizedProcessingCapabilityError("p99 runtime differs")
        if self.allowed_processing_ms != MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS:
            raise MassiveFinalizedProcessingCapabilityError("processing allowance drifted")
        if self.capability_scope != "source-scan-and-partition-only":
            raise MassiveFinalizedProcessingCapabilityError("capability scope drifted")
        expected = self.maximum_runtime_ms <= self.allowed_processing_ms
        if self.capability_passed is not expected:
            raise MassiveFinalizedProcessingCapabilityError("capability outcome differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedProcessingCapabilityError("capability receipt differs")

    def covers(self, scan: MassiveDailyTradeFileScanEvidenceV0) -> bool:
        self.validate()
        scan.validate()
        return (
            self.capability_passed
            and scan.compressed_bytes <= self.maximum_compressed_bytes
            and scan.source_row_count <= self.maximum_decompressed_rows
            and scan.parser_source_sha256 == self.parser_source_sha256
        )


def build_massive_finalized_processing_capability_v0(
    benchmarks: Sequence[MassiveFinalizedProcessingBenchmarkV0],
) -> MassiveFinalizedProcessingCapabilityV0:
    rows = tuple(benchmarks)
    if not rows:
        raise MassiveFinalizedProcessingCapabilityError("processing benchmarks are absent")
    for row in rows:
        row.validate()
    identities = {
        (
            row.hardware_contract_receipt_sha256,
            row.software_commit_sha256,
            row.parser_source_sha256,
            row.feature_materializer_source_sha256,
        )
        for row in rows
    }
    if len(identities) != 1:
        raise MassiveFinalizedProcessingCapabilityError("benchmark implementations differ")
    hardware, commit, parser, materializer = next(iter(identities))
    receipts = tuple(sorted(row.receipt_sha256 for row in rows))
    sources = tuple(sorted(row.source_object_receipt_sha256 for row in rows))
    if len(receipts) != len(set(receipts)) or len(sources) != len(set(sources)):
        raise MassiveFinalizedProcessingCapabilityError("benchmarks must use distinct sources")
    runtimes = tuple(sorted(row.observed_total_runtime_ms for row in rows))
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_PROCESSING_CAPABILITY_V0_SCHEMA,
        "hardware_contract_receipt_sha256": hardware,
        "software_commit_sha256": commit,
        "parser_source_sha256": parser,
        "feature_materializer_source_sha256": materializer,
        "benchmark_receipts": receipts,
        "benchmark_source_receipts": sources,
        "maximum_compressed_bytes": max(row.compressed_bytes for row in rows),
        "maximum_decompressed_rows": max(row.decompressed_rows for row in rows),
        "observed_runtime_ms": runtimes,
        "p95_runtime_ms": _nearest_rank(runtimes, 0.95),
        "p99_runtime_ms": _nearest_rank(runtimes, 0.99),
        "maximum_runtime_ms": max(runtimes),
        "allowed_processing_ms": MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
        "capability_scope": "source-scan-and-partition-only",
        "capability_passed": max(runtimes) <= MASSIVE_FINALIZED_PROCESSING_ALLOWANCE_MS,
    }
    result = MassiveFinalizedProcessingCapabilityV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_FINALIZED_PROCESSING_BENCHMARK_V0_SCHEMA",
    "MASSIVE_FINALIZED_PROCESSING_CAPABILITY_V0_SCHEMA",
    "MassiveFinalizedProcessingBenchmarkV0",
    "MassiveFinalizedProcessingCapabilityError",
    "MassiveFinalizedProcessingCapabilityV0",
    "build_massive_finalized_processing_benchmark_v0",
    "build_massive_finalized_processing_capability_v0",
    "measure_massive_finalized_source_processing_v0",
]
