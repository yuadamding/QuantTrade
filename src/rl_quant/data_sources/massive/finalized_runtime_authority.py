"""Raw execution-host, clock, and runtime-environment capture authorities."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast

from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_BUCKET,
    MASSIVE_FLAT_FILE_ENDPOINT,
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

MASSIVE_HOST_EXECUTION_V2_SCHEMA = "rl-quant.massive-host-execution-authority-v2"
MASSIVE_HOST_EXECUTION_V2_DATASET = "massive-finalized-host-execution-v2"
MASSIVE_HOST_EXECUTION_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "format": "canonical-json",
        "source": "fixed-local-os-host-capture",
        "fields": (
            "capture_started_at_ms",
            "capture_finished_at_ms",
            "machine_id_sha256",
            "boot_id_sha256",
            "hostname_sha256",
            "kernel_system",
            "kernel_release",
            "kernel_version",
            "machine_architecture",
            "logical_cpu_count",
            "physical_memory_bytes",
        ),
    }
)
MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256 = semantic_sha256(
    {
        "machine_identity": "sha256(/etc/machine-id)",
        "boot_identity": "sha256(/proc/sys/kernel/random/boot_id)",
        "hostname_identity": "sha256(platform.node)",
        "hardware": "os.uname+cpu-count+physical-memory",
        "capture_clock": "non-injectable-time.time_ns",
        "generic_manifest_authorizing": False,
    }
)

MASSIVE_RAW_CHRONY_V2_SCHEMA = "rl-quant.massive-raw-chrony-capture-v2"
MASSIVE_RAW_CHRONY_V2_DATASET = "massive-finalized-raw-chrony-v2"
MASSIVE_RAW_CHRONY_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "format": "canonical-json-preserving-raw-stdout-stderr",
        "commands": (
            ("/usr/bin/chronyc", "--version"),
            ("/usr/bin/chronyc", "-c", "tracking"),
            ("/usr/bin/chronyc", "-c", "sources"),
        ),
    }
)
MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256 = semantic_sha256(
    {
        "source": "fixed-command-raw-chrony-capture",
        "maximum_qualification_window_ms": 60 * 60 * 1_000,
        "required_leap_state": "Normal",
        "allowed_stratum": "1..15",
        "selected_source_required": True,
        "signed_conversion": "decimal-nanoseconds-away-from-zero",
        "freshness": "reference-age-and-selected-last-rx",
        "freshness_floor_ms": 5 * 60 * 1_000,
        "freshness_ceiling_ms": 15 * 60 * 1_000,
        "maximum_root_delay_ns": 1_000_000_000,
        "error_bound": "max(abs(system-time),abs(last),rms,root-dispersion)+frequency-drift",
        "host_binding": MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
        "generic_manifest_authorizing": False,
    }
)

MASSIVE_RUNTIME_ENVIRONMENT_V2_SCHEMA = (
    "rl-quant.massive-runtime-execution-environment-v2"
)
MASSIVE_RUNTIME_ENVIRONMENT_V2_DATASET = (
    "massive-finalized-runtime-execution-environment-v2"
)
MASSIVE_RUNTIME_ENVIRONMENT_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "format": "canonical-json",
        "source": "fixed-process-and-os-observation",
        "components": (
            "host-hardware",
            "source-archive",
            "container-runtime",
            "python-environment",
            "network-acquisition",
            "storage-performance",
        ),
    }
)
MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256 = semantic_sha256(
    {
        "host": MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
        "source_archive": "discovered-import-root+clean-git-HEAD-tree-and-imported-blobs",
        "container": "proc-cgroup+read-only-runtime-metadata+image-digest",
        "python": "executable-bytes+version+installed-distributions",
        "network": "official-endpoint+bucket+runtime-client-type",
        "storage": "resolved-roots+device+statvfs",
        "generic_manifest_authorizing": False,
    }
)

MASSIVE_CHRONYC_BINARY = "/usr/bin/chronyc"
MASSIVE_CLOCK_QUALIFICATION_WINDOW_MS = 60 * 60 * 1_000
MASSIVE_CLOCK_FRESHNESS_FLOOR_MS = 5 * 60 * 1_000
MASSIVE_CLOCK_FRESHNESS_CEILING_MS = 15 * 60 * 1_000
MASSIVE_CLOCK_MAXIMUM_ROOT_DELAY_NS = 1_000_000_000
MASSIVE_CLOCK_ALLOWED_SOURCE_MODES = ("#", "=", "^")
MASSIVE_CONTAINER_IMAGE_DIGEST_PATH = Path(
    "/run/quanttrade/container-image-digest"
)
MASSIVE_CONTAINER_RUNTIME_METADATA_PATH = Path(
    "/run/quanttrade/container-runtime.json"
)
MASSIVE_IMPORTED_PIPELINE_SOURCE_RELATIVE_PATHS = (
    "src/rl_quant/data_sources/massive/finalized_runtime_authority.py",
    "src/rl_quant/data_sources/massive/finalized_archive_scope.py",
    "src/rl_quant/data_sources/massive/finalized_typed_decision_origin.py",
    "src/rl_quant/workflows/massive_historical_readiness_v1.py",
    "src/rl_quant/workflows/massive_production_typed_run_v2.py",
)
MASSIVE_RUNTIME_AUTHORITY_SOURCE_SHA256 = file_sha256(Path(__file__))


class MassiveRuntimeAuthorityError(ValueError):
    """Runtime evidence was not derived from the fixed local capture path."""


def _wall_ms() -> int:
    return time.time_ns() // 1_000_000


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveRuntimeAuthorityError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveRuntimeAuthorityError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveRuntimeAuthorityError(f"{name} must be a nonnegative integer")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveRuntimeAuthorityError(f"{name} must be canonical text")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_required(path: Path) -> bytes:
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise MassiveRuntimeAuthorityError(f"cannot read required host source: {path}") from exc
    if not value.strip():
        raise MassiveRuntimeAuthorityError(f"required host source is empty: {path}")
    return value


def _publish_payload(
    *,
    root: str | Path,
    relative_path: str,
    dataset_id: str,
    schema_sha256: str,
    payload: dict[str, object],
    requested_at_ms: int,
    downloaded_at_ms: int,
    entitlement_receipt_sha256: str,
    request_id: str,
) -> LoadedMassiveSourceObject:
    Path(root).mkdir(parents=True, exist_ok=True)
    committed_at_ms = _wall_ms()
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative_path,
        dataset_id=dataset_id,
        source_object_key=relative_path,
        requested_at_ms=requested_at_ms,
        downloaded_at_ms=downloaded_at_ms,
        schema_sha256=schema_sha256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=max(committed_at_ms, downloaded_at_ms),
        request_id=request_id,
    )
    return load_massive_source_bundle(
        root=root,
        relative_payload_path=relative_path,
        verified_at_ms=max(_wall_ms(), downloaded_at_ms),
    )


def _canonical_source_payload(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    dataset_id: str,
    schema_sha256: str,
) -> dict[str, object]:
    loaded_source.validate()
    if (
        loaded_source.receipt.dataset_id != dataset_id
        or loaded_source.receipt.schema_sha256 != schema_sha256
    ):
        raise MassiveRuntimeAuthorityError("runtime source dataset/schema differs")
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveRuntimeAuthorityError("runtime source is not JSON") from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveRuntimeAuthorityError("runtime source is not canonical")
    return payload


def _physical_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError) as exc:
        raise MassiveRuntimeAuthorityError("physical memory cannot be observed") from exc
    return _positive_int("physical memory", pages * page_size)


def _current_host_payload(*, started_at_ms: int, finished_at_ms: int) -> dict[str, object]:
    machine_id = _read_required(Path("/etc/machine-id"))
    boot_id = _read_required(Path("/proc/sys/kernel/random/boot_id"))
    hostname = platform.node().encode("utf-8")
    if not hostname:
        raise MassiveRuntimeAuthorityError("host name is empty")
    uname = os.uname()
    cpu_count = os.cpu_count()
    return {
        "capture_started_at_ms": started_at_ms,
        "capture_finished_at_ms": finished_at_ms,
        "machine_id_sha256": _sha256_bytes(machine_id.strip()),
        "boot_id_sha256": _sha256_bytes(boot_id.strip()),
        "hostname_sha256": _sha256_bytes(hostname),
        "kernel_system": uname.sysname,
        "kernel_release": uname.release,
        "kernel_version": uname.version,
        "machine_architecture": uname.machine,
        "logical_cpu_count": _positive_int("logical CPU count", cpu_count),
        "physical_memory_bytes": _physical_memory_bytes(),
    }


@dataclass(frozen=True, slots=True)
class MassiveHostExecutionAuthorityV2:
    host_id: str
    machine_id_sha256: str
    boot_id_sha256: str
    hostname_sha256: str
    kernel_system: str
    kernel_release: str
    kernel_version: str
    machine_architecture: str
    logical_cpu_count: int
    physical_memory_bytes: int
    capture_started_at_ms: int
    capture_finished_at_ms: int
    loaded_source: LoadedMassiveSourceObject
    host_spec_receipt_sha256: str
    parser_source_sha256: str
    captured_by_fixed_runtime: bool
    receipt_sha256: str
    schema: str = MASSIVE_HOST_EXECUTION_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_HOST_EXECUTION_V2_SCHEMA
            or not isinstance(self.captured_by_fixed_runtime, bool)
            or self.capture_started_at_ms > self.capture_finished_at_ms
            or self.loaded_source.receipt.dataset_id != MASSIVE_HOST_EXECUTION_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_HOST_EXECUTION_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.requested_at_ms > self.capture_started_at_ms
            or self.loaded_source.receipt.downloaded_at_ms < self.capture_finished_at_ms
            or self.host_spec_receipt_sha256 != MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256
            or self.parser_source_sha256 != MASSIVE_RUNTIME_AUTHORITY_SOURCE_SHA256
        ):
            raise MassiveRuntimeAuthorityError("host execution authority differs")
        self.loaded_source.validate()
        for name in (
            "host_id",
            "machine_id_sha256",
            "boot_id_sha256",
            "hostname_sha256",
            "host_spec_receipt_sha256",
            "parser_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in ("kernel_system", "kernel_release", "kernel_version", "machine_architecture"):
            _text(name, getattr(self, name))
        _positive_int("logical CPU count", self.logical_cpu_count)
        _positive_int("physical memory", self.physical_memory_bytes)
        expected_host_id = semantic_sha256(
            (self.machine_id_sha256, self.hostname_sha256, self.machine_architecture)
        )
        if self.host_id != expected_host_id or self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRuntimeAuthorityError("host identity receipt differs")


def parse_massive_host_execution_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveHostExecutionAuthorityV2:
    payload = _canonical_source_payload(
        root=root,
        loaded_source=loaded_source,
        dataset_id=MASSIVE_HOST_EXECUTION_V2_DATASET,
        schema_sha256=MASSIVE_HOST_EXECUTION_V2_SOURCE_SCHEMA_SHA256,
    )
    expected = {
        "capture_started_at_ms",
        "capture_finished_at_ms",
        "machine_id_sha256",
        "boot_id_sha256",
        "hostname_sha256",
        "kernel_system",
        "kernel_release",
        "kernel_version",
        "machine_architecture",
        "logical_cpu_count",
        "physical_memory_bytes",
    }
    if set(payload) != expected:
        raise MassiveRuntimeAuthorityError("host source field inventory differs")
    for name in ("capture_started_at_ms", "capture_finished_at_ms"):
        _nonnegative_int(name, payload[name])
    for name in ("logical_cpu_count", "physical_memory_bytes"):
        _positive_int(name, payload[name])
    for name in ("machine_id_sha256", "boot_id_sha256", "hostname_sha256"):
        _digest(name, payload[name])
    for name in ("kernel_system", "kernel_release", "kernel_version", "machine_architecture"):
        _text(name, payload[name])
    host_id = semantic_sha256(
        (
            payload["machine_id_sha256"],
            payload["hostname_sha256"],
            payload["machine_architecture"],
        )
    )
    body = {
        "schema": MASSIVE_HOST_EXECUTION_V2_SCHEMA,
        "host_id": host_id,
        **payload,
        "loaded_source": loaded_source,
        "host_spec_receipt_sha256": MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
        "parser_source_sha256": MASSIVE_RUNTIME_AUTHORITY_SOURCE_SHA256,
        "captured_by_fixed_runtime": False,
    }
    provisional = MassiveHostExecutionAuthorityV2(
        **body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,
    )
    result = replace(provisional, receipt_sha256=semantic_sha256(provisional.unsigned()))
    result.validate()
    return result


def capture_massive_host_execution_authority_v2(
    *, root: str | Path, entitlement_receipt_sha256: str
) -> MassiveHostExecutionAuthorityV2:
    """Capture the current host directly; no caller-supplied identity is accepted."""

    _digest("entitlement receipt", entitlement_receipt_sha256)
    started_at_ms = _wall_ms()
    payload = _current_host_payload(
        started_at_ms=started_at_ms,
        finished_at_ms=_wall_ms(),
    )
    finished_at_ms = cast(int, payload["capture_finished_at_ms"])
    relative_path = (
        f"execution/host-v2/{started_at_ms}-{uuid.uuid4().hex}.json"
    )
    loaded = _publish_payload(
        root=root,
        relative_path=relative_path,
        dataset_id=MASSIVE_HOST_EXECUTION_V2_DATASET,
        schema_sha256=MASSIVE_HOST_EXECUTION_V2_SOURCE_SCHEMA_SHA256,
        payload=payload,
        requested_at_ms=started_at_ms,
        downloaded_at_ms=finished_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        request_id="fixed-local-os-host-capture-v2",
    )
    parsed = parse_massive_host_execution_authority_v2(
        root=root,
        loaded_source=loaded,
    )
    captured = replace(parsed, captured_by_fixed_runtime=True, receipt_sha256="0" * 64)
    captured = replace(
        captured,
        receipt_sha256=semantic_sha256(captured.unsigned()),
    )
    captured.validate()
    return captured


def verify_current_execution_host_v2(authority: MassiveHostExecutionAuthorityV2) -> None:
    """Reject substitution of an authority captured on another host or boot."""

    authority.validate()
    payload = _current_host_payload(started_at_ms=0, finished_at_ms=0)
    if any(
        getattr(authority, name) != payload[name]
        for name in (
            "machine_id_sha256",
            "boot_id_sha256",
            "hostname_sha256",
            "kernel_system",
            "kernel_release",
            "kernel_version",
            "machine_architecture",
            "logical_cpu_count",
            "physical_memory_bytes",
        )
    ):
        raise MassiveRuntimeAuthorityError("execution host differs from captured host")


@dataclass(frozen=True, slots=True)
class MassiveRawCommandEvidenceV2:
    argv: tuple[str, ...]
    started_at_ms: int
    finished_at_ms: int
    exit_status: int
    stdout_text: str
    stderr_text: str
    stdout_sha256: str
    stderr_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        if (
            not self.argv
            or any(not isinstance(value, str) or not value for value in self.argv)
            or self.started_at_ms > self.finished_at_ms
            or isinstance(self.exit_status, bool)
            or not isinstance(self.exit_status, int)
            or self.exit_status != 0
            or not self.stdout_text.strip()
        ):
            raise MassiveRuntimeAuthorityError("raw command evidence differs")
        for name in ("stdout_sha256", "stderr_sha256", "receipt_sha256"):
            _digest(name, getattr(self, name))
        if (
            self.stdout_sha256 != _sha256_bytes(self.stdout_text.encode("utf-8"))
            or self.stderr_sha256 != _sha256_bytes(self.stderr_text.encode("utf-8"))
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveRuntimeAuthorityError("raw command byte identity differs")


def _command_from_payload(value: object) -> MassiveRawCommandEvidenceV2:
    if not isinstance(value, dict) or set(value) != {
        "argv",
        "started_at_ms",
        "finished_at_ms",
        "exit_status",
        "stdout_text",
        "stderr_text",
    }:
        raise MassiveRuntimeAuthorityError("raw command fields differ")
    argv_value = value["argv"]
    if not isinstance(argv_value, list) or not all(isinstance(item, str) for item in argv_value):
        raise MassiveRuntimeAuthorityError("raw command argv differs")
    body = {
        "argv": tuple(argv_value),
        "started_at_ms": value["started_at_ms"],
        "finished_at_ms": value["finished_at_ms"],
        "exit_status": value["exit_status"],
        "stdout_text": value["stdout_text"],
        "stderr_text": value["stderr_text"],
        "stdout_sha256": _sha256_bytes(cast(str, value["stdout_text"]).encode("utf-8")),
        "stderr_sha256": _sha256_bytes(cast(str, value["stderr_text"]).encode("utf-8")),
    }
    provisional = MassiveRawCommandEvidenceV2(**body, receipt_sha256="0" * 64)  # type: ignore[arg-type]
    result = replace(provisional, receipt_sha256=semantic_sha256(provisional.unsigned()))
    result.validate()
    return result


def conservative_chrony_seconds_to_ns_v2(value: str) -> int:
    try:
        seconds = Decimal(value)
        magnitude = (
            abs(seconds) * Decimal(1_000_000_000)
        ).to_integral_value(rounding=ROUND_CEILING)
        return int(magnitude if seconds >= 0 else -magnitude)
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise MassiveRuntimeAuthorityError("chrony seconds value is malformed") from exc


def _tracking_fields(stdout_text: str) -> dict[str, object]:
    rows = tuple(csv.reader(StringIO(stdout_text)))
    if len(rows) != 1 or len(rows[0]) != 13:
        raise MassiveRuntimeAuthorityError("chrony tracking CSV shape differs")
    row = rows[0]
    try:
        stratum = int(row[1])
        reference_time_unix_seconds = Decimal(row[2])
        frequency_skew_ppm = float(row[8])
        update_interval_seconds = Decimal(row[11])
    except (InvalidOperation, ValueError) as exc:
        raise MassiveRuntimeAuthorityError("chrony tracking values are malformed") from exc
    if (
        not row[0]
        or row[0] == "00000000"
        or not 1 <= stratum <= 15
        or not reference_time_unix_seconds.is_finite()
        or reference_time_unix_seconds <= 0
        or not math.isfinite(frequency_skew_ppm)
        or frequency_skew_ppm < 0
        or not update_interval_seconds.is_finite()
        or update_interval_seconds <= 0
        or row[12] != "Normal"
    ):
        raise MassiveRuntimeAuthorityError("chrony is not synchronized")
    return {
        "reference_id": row[0],
        "stratum": stratum,
        "reference_time_unix_ns": conservative_chrony_seconds_to_ns_v2(row[2]),
        "system_time_offset_ns": conservative_chrony_seconds_to_ns_v2(row[3]),
        "last_offset_ns": conservative_chrony_seconds_to_ns_v2(row[4]),
        "rms_offset_ns": abs(conservative_chrony_seconds_to_ns_v2(row[5])),
        "frequency_skew_ppm": frequency_skew_ppm,
        "root_delay_ns": abs(conservative_chrony_seconds_to_ns_v2(row[9])),
        "root_dispersion_ns": abs(conservative_chrony_seconds_to_ns_v2(row[10])),
        "tracking_update_interval_ms": int(
            (update_interval_seconds * Decimal(1_000)).to_integral_value(
                rounding=ROUND_CEILING
            )
        ),
        "leap_status": row[12],
    }


def _selected_source_fields(stdout_text: str) -> dict[str, object]:
    rows = tuple(row for row in csv.reader(StringIO(stdout_text)) if row)
    selected: list[dict[str, object]] = []
    for row in rows:
        if len(row) != 8:
            raise MassiveRuntimeAuthorityError("chrony sources CSV shape differs")
        try:
            reachability = int(row[5], 8)
            source_stratum = int(row[3])
            poll_exponent = int(row[4])
            last_rx_seconds = Decimal(row[6])
            last_rx_ms = int(
                (last_rx_seconds * Decimal(1_000)).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
        except (InvalidOperation, OverflowError, ValueError) as exc:
            raise MassiveRuntimeAuthorityError("chrony source reachability is malformed") from exc
        if row[1] == "*" and reachability > 0:
            if (
                row[0] not in MASSIVE_CLOCK_ALLOWED_SOURCE_MODES
                or not 0 <= source_stratum <= 15
                or not 0 <= poll_exponent <= 24
                or not last_rx_seconds.is_finite()
                or last_rx_ms < 0
            ):
                raise MassiveRuntimeAuthorityError(
                    "selected chrony source fields are unqualified"
                )
            selected.append(
                {
                    "selected_source_mode": row[0],
                    "selected_source_stratum": source_stratum,
                    "selected_source_poll_interval_ms": (2**poll_exponent) * 1_000,
                    "selected_source_last_rx_ms": last_rx_ms,
                }
            )
    if len(selected) != 1:
        raise MassiveRuntimeAuthorityError("chrony must have exactly one selected reachable source")
    return selected[0]


@dataclass(frozen=True, slots=True)
class MassiveExecutionClockAuthorityV2:
    host_authority_receipt_sha256: str
    version_command: MassiveRawCommandEvidenceV2
    tracking_command: MassiveRawCommandEvidenceV2
    sources_command: MassiveRawCommandEvidenceV2
    measurement_observed_at_ms: int
    qualification_valid_until_ms: int
    reference_id: str
    stratum: int
    reference_time_unix_ns: int
    system_time_offset_ns: int
    last_offset_ns: int
    rms_offset_ns: int
    root_delay_ns: int
    root_dispersion_ns: int
    frequency_skew_ppm: float
    tracking_update_interval_ms: int
    leap_status: str
    selected_source_count: int
    selected_source_mode: str
    selected_source_stratum: int
    selected_source_poll_interval_ms: int
    selected_source_last_rx_ms: int
    selected_source_freshness_limit_ms: int
    reference_age_ms: int
    reference_freshness_limit_ms: int
    maximum_clock_error_ns: int
    loaded_source: LoadedMassiveSourceObject
    clock_spec_receipt_sha256: str
    parser_source_sha256: str
    captured_by_fixed_runtime: bool
    receipt_sha256: str
    schema: str = MASSIVE_RAW_CHRONY_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    @property
    def maximum_clock_error_ms(self) -> int:
        return math.ceil(self.maximum_clock_error_ns / 1_000_000)

    def utc_lower_bound_ms(self, timestamp_ms: int) -> int:
        self.validate()
        return timestamp_ms - self.maximum_clock_error_ms

    def utc_upper_bound_ms(self, timestamp_ms: int) -> int:
        self.validate()
        return timestamp_ms + self.maximum_clock_error_ms

    @property
    def measurement_utc_lower_bound_ms(self) -> int:
        return self.measurement_observed_at_ms - self.maximum_clock_error_ms

    @property
    def measurement_utc_upper_bound_ms(self) -> int:
        return self.measurement_observed_at_ms + self.maximum_clock_error_ms

    @property
    def qualification_end_utc_lower_bound_ms(self) -> int:
        return self.measurement_utc_lower_bound_ms + MASSIVE_CLOCK_QUALIFICATION_WINDOW_MS

    def validate(self) -> None:
        for command in (self.version_command, self.tracking_command, self.sources_command):
            command.validate()
        expected_argv = (
            (MASSIVE_CHRONYC_BINARY, "--version"),
            (MASSIVE_CHRONYC_BINARY, "-c", "tracking"),
            (MASSIVE_CHRONYC_BINARY, "-c", "sources"),
        )
        if (
            self.schema != MASSIVE_RAW_CHRONY_V2_SCHEMA
            or tuple(command.argv for command in (self.version_command, self.tracking_command, self.sources_command))
            != expected_argv
            or not isinstance(self.captured_by_fixed_runtime, bool)
            or self.measurement_observed_at_ms != self.tracking_command.finished_at_ms
            or self.qualification_valid_until_ms - self.measurement_observed_at_ms
            != MASSIVE_CLOCK_QUALIFICATION_WINDOW_MS
            or self.leap_status != "Normal"
            or not 1 <= self.stratum <= 15
            or self.selected_source_count != 1
            or self.selected_source_mode not in MASSIVE_CLOCK_ALLOWED_SOURCE_MODES
            or not 0 <= self.selected_source_stratum <= 15
            or self.root_delay_ns > MASSIVE_CLOCK_MAXIMUM_ROOT_DELAY_NS
            or self.loaded_source.receipt.dataset_id != MASSIVE_RAW_CHRONY_V2_DATASET
            or self.loaded_source.receipt.schema_sha256 != MASSIVE_RAW_CHRONY_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.requested_at_ms
            > self.version_command.started_at_ms
            or not (
                self.version_command.finished_at_ms
                <= self.tracking_command.started_at_ms
                <= self.tracking_command.finished_at_ms
                <= self.sources_command.started_at_ms
                <= self.sources_command.finished_at_ms
                <= self.loaded_source.receipt.downloaded_at_ms
            )
            or self.loaded_source.verified_at_ms > self.qualification_valid_until_ms
            or self.clock_spec_receipt_sha256 != MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256
            or self.parser_source_sha256 != MASSIVE_RUNTIME_AUTHORITY_SOURCE_SHA256
        ):
            raise MassiveRuntimeAuthorityError("execution clock v2 authority differs")
        if "chrony" not in self.version_command.stdout_text.lower():
            raise MassiveRuntimeAuthorityError("chrony version evidence differs")
        self.loaded_source.validate()
        for name in ("host_authority_receipt_sha256", "clock_spec_receipt_sha256", "parser_source_sha256", "receipt_sha256"):
            _digest(name, getattr(self, name))
        validity_ns = MASSIVE_CLOCK_QUALIFICATION_WINDOW_MS * 1_000_000
        drift_ns = math.ceil(validity_ns * self.frequency_skew_ppm / 1_000_000)
        expected_error = max(
            abs(self.system_time_offset_ns),
            abs(self.last_offset_ns),
            self.rms_offset_ns,
            self.root_dispersion_ns,
        ) + drift_ns
        reference_age_ns = max(
            0,
            self.measurement_observed_at_ms * 1_000_000
            - self.reference_time_unix_ns,
        )
        expected_reference_age_ms = (reference_age_ns + 999_999) // 1_000_000
        expected_reference_limit_ms = max(
            3 * self.tracking_update_interval_ms,
            MASSIVE_CLOCK_FRESHNESS_FLOOR_MS,
        )
        expected_reference_limit_ms = min(
            expected_reference_limit_ms,
            MASSIVE_CLOCK_FRESHNESS_CEILING_MS,
        )
        expected_source_limit_ms = max(
            3 * self.selected_source_poll_interval_ms,
            MASSIVE_CLOCK_FRESHNESS_FLOOR_MS,
        )
        expected_source_limit_ms = min(
            expected_source_limit_ms,
            MASSIVE_CLOCK_FRESHNESS_CEILING_MS,
        )
        if (
            self.maximum_clock_error_ns != expected_error
            or self.reference_age_ms != expected_reference_age_ms
            or self.reference_freshness_limit_ms != expected_reference_limit_ms
            or self.selected_source_freshness_limit_ms != expected_source_limit_ms
            or self.reference_time_unix_ns
            > self.measurement_observed_at_ms * 1_000_000
            + self.maximum_clock_error_ns
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveRuntimeAuthorityError("execution clock v2 receipt differs")
        if self.reference_age_ms > self.reference_freshness_limit_ms:
            raise MassiveRuntimeAuthorityError("chrony tracking reference is stale")
        if (
            self.selected_source_last_rx_ms
            > self.selected_source_freshness_limit_ms
        ):
            raise MassiveRuntimeAuthorityError("selected chrony source is stale")


def parse_massive_execution_clock_authority_v2(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    host_authority: MassiveHostExecutionAuthorityV2,
    host_root: str | Path,
) -> MassiveExecutionClockAuthorityV2:
    expected_host = parse_massive_host_execution_authority_v2(
        root=host_root, loaded_source=host_authority.loaded_source
    )
    if (
        expected_host.loaded_source.receipt_sha256
        != host_authority.loaded_source.receipt_sha256
        or expected_host.host_id != host_authority.host_id
    ):
        raise MassiveRuntimeAuthorityError("clock host authority was not rederived")
    payload = _canonical_source_payload(
        root=root,
        loaded_source=loaded_source,
        dataset_id=MASSIVE_RAW_CHRONY_V2_DATASET,
        schema_sha256=MASSIVE_RAW_CHRONY_V2_SOURCE_SCHEMA_SHA256,
    )
    if set(payload) != {"host_authority_receipt_sha256", "commands"}:
        raise MassiveRuntimeAuthorityError("raw chrony source fields differ")
    commands_value = payload["commands"]
    if not isinstance(commands_value, list) or len(commands_value) != 3:
        raise MassiveRuntimeAuthorityError("raw chrony command inventory differs")
    commands = tuple(_command_from_payload(value) for value in commands_value)
    if payload["host_authority_receipt_sha256"] != host_authority.receipt_sha256:
        raise MassiveRuntimeAuthorityError("raw chrony host binding differs")
    version, tracking, sources = commands
    fields = _tracking_fields(tracking.stdout_text)
    selected_fields = _selected_source_fields(sources.stdout_text)
    measurement_at_ms = tracking.finished_at_ms
    validity_ns = MASSIVE_CLOCK_QUALIFICATION_WINDOW_MS * 1_000_000
    drift_ns = math.ceil(validity_ns * cast(float, fields["frequency_skew_ppm"]) / 1_000_000)
    maximum_error = max(
        abs(cast(int, fields["system_time_offset_ns"])),
        abs(cast(int, fields["last_offset_ns"])),
        cast(int, fields["rms_offset_ns"]),
        cast(int, fields["root_dispersion_ns"]),
    ) + drift_ns
    reference_age_ns = max(
        0,
        measurement_at_ms * 1_000_000
        - cast(int, fields["reference_time_unix_ns"]),
    )
    reference_age_ms = (reference_age_ns + 999_999) // 1_000_000
    reference_freshness_limit_ms = max(
        3 * cast(int, fields["tracking_update_interval_ms"]),
        MASSIVE_CLOCK_FRESHNESS_FLOOR_MS,
    )
    reference_freshness_limit_ms = min(
        reference_freshness_limit_ms,
        MASSIVE_CLOCK_FRESHNESS_CEILING_MS,
    )
    source_freshness_limit_ms = max(
        3 * cast(int, selected_fields["selected_source_poll_interval_ms"]),
        MASSIVE_CLOCK_FRESHNESS_FLOOR_MS,
    )
    source_freshness_limit_ms = min(
        source_freshness_limit_ms,
        MASSIVE_CLOCK_FRESHNESS_CEILING_MS,
    )
    body = {
        "schema": MASSIVE_RAW_CHRONY_V2_SCHEMA,
        "host_authority_receipt_sha256": host_authority.receipt_sha256,
        "version_command": version,
        "tracking_command": tracking,
        "sources_command": sources,
        "measurement_observed_at_ms": measurement_at_ms,
        "qualification_valid_until_ms": measurement_at_ms + MASSIVE_CLOCK_QUALIFICATION_WINDOW_MS,
        **fields,
        "selected_source_count": 1,
        **selected_fields,
        "selected_source_freshness_limit_ms": source_freshness_limit_ms,
        "reference_age_ms": reference_age_ms,
        "reference_freshness_limit_ms": reference_freshness_limit_ms,
        "maximum_clock_error_ns": maximum_error,
        "loaded_source": loaded_source,
        "clock_spec_receipt_sha256": MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256,
        "parser_source_sha256": MASSIVE_RUNTIME_AUTHORITY_SOURCE_SHA256,
        "captured_by_fixed_runtime": False,
    }
    provisional = MassiveExecutionClockAuthorityV2(**body, receipt_sha256="0" * 64)  # type: ignore[arg-type]
    result = replace(provisional, receipt_sha256=semantic_sha256(provisional.unsigned()))
    result.validate()
    return result


def _run_chrony_command(argv: tuple[str, ...]) -> dict[str, object]:
    started_at_ms = _wall_ms()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=10,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MassiveRuntimeAuthorityError(f"fixed chrony command failed: {argv}") from exc
    finished_at_ms = _wall_ms()
    try:
        stdout_text = completed.stdout.decode("utf-8")
        stderr_text = completed.stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MassiveRuntimeAuthorityError("chrony output is not UTF-8") from exc
    return {
        "argv": list(argv),
        "started_at_ms": started_at_ms,
        "finished_at_ms": finished_at_ms,
        "exit_status": completed.returncode,
        "stdout_text": stdout_text,
        "stderr_text": stderr_text,
    }


def capture_massive_execution_clock_authority_v2(
    *,
    root: str | Path,
    host_authority: MassiveHostExecutionAuthorityV2,
    host_root: str | Path,
    entitlement_receipt_sha256: str,
) -> MassiveExecutionClockAuthorityV2:
    """Run fixed chrony commands and bind their raw bytes to the current host."""

    _digest("entitlement receipt", entitlement_receipt_sha256)
    expected_host = parse_massive_host_execution_authority_v2(
        root=host_root, loaded_source=host_authority.loaded_source
    )
    if (
        expected_host.loaded_source.receipt_sha256
        != host_authority.loaded_source.receipt_sha256
        or expected_host.host_id != host_authority.host_id
        or not host_authority.captured_by_fixed_runtime
    ):
        raise MassiveRuntimeAuthorityError("clock capture host was not rederived")
    verify_current_execution_host_v2(host_authority)
    requested_at_ms = _wall_ms()
    commands = [
        _run_chrony_command((MASSIVE_CHRONYC_BINARY, "--version")),
        _run_chrony_command((MASSIVE_CHRONYC_BINARY, "-c", "tracking")),
        _run_chrony_command((MASSIVE_CHRONYC_BINARY, "-c", "sources")),
    ]
    payload: dict[str, object] = {
        "host_authority_receipt_sha256": host_authority.receipt_sha256,
        "commands": commands,
    }
    finished_at_ms = _wall_ms()
    relative_path = f"execution/clock-v2/{requested_at_ms}-{uuid.uuid4().hex}.json"
    loaded = _publish_payload(
        root=root,
        relative_path=relative_path,
        dataset_id=MASSIVE_RAW_CHRONY_V2_DATASET,
        schema_sha256=MASSIVE_RAW_CHRONY_V2_SOURCE_SCHEMA_SHA256,
        payload=payload,
        requested_at_ms=requested_at_ms,
        downloaded_at_ms=finished_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        request_id="fixed-chronyc-runtime-capture-v2",
    )
    parsed = parse_massive_execution_clock_authority_v2(
        root=root,
        loaded_source=loaded,
        host_authority=host_authority,
        host_root=host_root,
    )
    captured = replace(parsed, captured_by_fixed_runtime=True, receipt_sha256="0" * 64)
    captured = replace(
        captured,
        receipt_sha256=semantic_sha256(captured.unsigned()),
    )
    captured.validate()
    return captured


@dataclass(frozen=True, slots=True)
class MassiveSourceArchiveAuthorityV2:
    git_commit: str
    git_tree: str
    git_archive_sha256: str
    tracked_worktree_clean: bool
    imported_source_count: int
    imported_source_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        if (
            len(self.git_commit) != 40
            or len(self.git_tree) != 40
            or any(character not in "0123456789abcdef" for character in self.git_commit + self.git_tree)
            or not self.tracked_worktree_clean
            or self.imported_source_count
            != len(MASSIVE_IMPORTED_PIPELINE_SOURCE_RELATIVE_PATHS)
        ):
            raise MassiveRuntimeAuthorityError("source archive authority differs")
        _digest("git archive", self.git_archive_sha256)
        _digest("imported source inventory", self.imported_source_inventory_sha256)
        _digest("source archive receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRuntimeAuthorityError("source archive receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveContainerRuntimeAuthorityV2:
    runtime_kind: str
    cgroup_sha256: str
    mountinfo_sha256: str
    container_id_sha256: str
    container_image_digest_sha256: str
    image_digest_source_sha256: str
    runtime_metadata_sha256: str
    runtime_mount_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        if self.runtime_kind not in {"docker", "podman", "kubernetes", "container"}:
            raise MassiveRuntimeAuthorityError("container runtime kind is unqualified")
        for name in (
            "cgroup_sha256",
            "mountinfo_sha256",
            "container_id_sha256",
            "container_image_digest_sha256",
            "image_digest_source_sha256",
            "runtime_metadata_sha256",
            "runtime_mount_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRuntimeAuthorityError("container runtime receipt differs")


@dataclass(frozen=True, slots=True)
class MassivePythonEnvironmentAuthorityV2:
    implementation: str
    version: str
    executable_path: str
    executable_sha256: str
    distribution_count: int
    distribution_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        for name in ("implementation", "version", "executable_path"):
            _text(name, getattr(self, name))
        _positive_int("distribution count", self.distribution_count)
        for name in ("executable_sha256", "distribution_inventory_sha256", "receipt_sha256"):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRuntimeAuthorityError("Python environment receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveNetworkAcquisitionAuthorityV2:
    endpoint: str
    bucket: str
    client_module: str
    client_qualname: str
    runtime_endpoint: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        if self.endpoint != MASSIVE_FLAT_FILE_ENDPOINT or self.bucket != MASSIVE_FLAT_FILE_BUCKET:
            raise MassiveRuntimeAuthorityError("network acquisition endpoint differs")
        for name in ("client_module", "client_qualname", "runtime_endpoint"):
            _text(name, getattr(self, name))
        if self.runtime_endpoint.rstrip("/") != self.endpoint.rstrip("/"):
            raise MassiveRuntimeAuthorityError("runtime client endpoint differs")
        _digest("network receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRuntimeAuthorityError("network acquisition receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveStorageRootAuthorityV2:
    role: str
    resolved_path: str
    device: int
    block_size: int
    total_blocks: int
    available_blocks: int
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        _text("storage role", self.role)
        if not Path(self.resolved_path).is_absolute():
            raise MassiveRuntimeAuthorityError("storage root path is not absolute")
        for name in ("device", "block_size", "total_blocks", "available_blocks"):
            _nonnegative_int(name, getattr(self, name))
        _digest("storage root receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRuntimeAuthorityError("storage root receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveRuntimeExecutionEnvironmentAuthorityV2:
    host_authority_receipt_sha256: str
    source_archive: MassiveSourceArchiveAuthorityV2
    container_runtime: MassiveContainerRuntimeAuthorityV2
    python_environment: MassivePythonEnvironmentAuthorityV2
    network_acquisition: MassiveNetworkAcquisitionAuthorityV2
    storage_roots: tuple[MassiveStorageRootAuthorityV2, ...]
    pipeline_implementation_inventory_sha256: str
    capture_started_at_ms: int
    capture_finished_at_ms: int
    loaded_source: LoadedMassiveSourceObject
    environment_spec_receipt_sha256: str
    parser_source_sha256: str
    captured_by_fixed_runtime: bool
    receipt_sha256: str
    schema: str = MASSIVE_RUNTIME_ENVIRONMENT_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "receipt_sha256"}

    def validate(self) -> None:
        self.source_archive.validate()
        self.container_runtime.validate()
        self.python_environment.validate()
        self.network_acquisition.validate()
        roles = tuple(row.role for row in self.storage_roots)
        if (
            self.schema != MASSIVE_RUNTIME_ENVIRONMENT_V2_SCHEMA
            or not isinstance(self.captured_by_fixed_runtime, bool)
            or self.capture_started_at_ms > self.capture_finished_at_ms
            or not roles
            or roles != tuple(sorted(set(roles)))
            or self.loaded_source.receipt.dataset_id != MASSIVE_RUNTIME_ENVIRONMENT_V2_DATASET
            or self.loaded_source.receipt.schema_sha256 != MASSIVE_RUNTIME_ENVIRONMENT_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.requested_at_ms > self.capture_started_at_ms
            or self.loaded_source.receipt.downloaded_at_ms < self.capture_finished_at_ms
            or self.environment_spec_receipt_sha256 != MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256
            or self.parser_source_sha256 != MASSIVE_RUNTIME_AUTHORITY_SOURCE_SHA256
        ):
            raise MassiveRuntimeAuthorityError("runtime environment authority differs")
        for row in self.storage_roots:
            row.validate()
        self.loaded_source.validate()
        for name in ("host_authority_receipt_sha256", "pipeline_implementation_inventory_sha256", "environment_spec_receipt_sha256", "parser_source_sha256", "receipt_sha256"):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRuntimeAuthorityError("runtime environment receipt differs")


def _receipt_dataclass(cls: type[Any], body: dict[str, object]) -> Any:
    provisional = cls(**body, receipt_sha256="0" * 64)
    result = replace(provisional, receipt_sha256=semantic_sha256(provisional.unsigned()))
    result.validate()
    return result


def _discover_quanttrade_repository_root() -> Path:
    import rl_quant

    module_file = Path(cast(str, rl_quant.__file__)).resolve(strict=True)
    try:
        completed = subprocess.run(
            ("/usr/bin/git", "-C", str(module_file.parent), "rev-parse", "--show-toplevel"),
            capture_output=True,
            check=False,
            timeout=30,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MassiveRuntimeAuthorityError("executing source root discovery failed") from exc
    if completed.returncode != 0:
        raise MassiveRuntimeAuthorityError("executing source is not in a Git repository")
    root = Path(completed.stdout.decode("utf-8").strip()).resolve(strict=True)
    if not module_file.is_relative_to(root):
        raise MassiveRuntimeAuthorityError("imported rl_quant is outside the discovered source root")
    return root


def _source_archive_payload() -> dict[str, object]:
    repository_root = _discover_quanttrade_repository_root()

    def run_git(*args: str) -> bytes:
        try:
            completed = subprocess.run(
                ("/usr/bin/git", "-C", str(repository_root), *args),
                capture_output=True,
                check=False,
                timeout=120,
                env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MassiveRuntimeAuthorityError("fixed git source capture failed") from exc
        if completed.returncode != 0:
            raise MassiveRuntimeAuthorityError("fixed git source capture returned nonzero")
        return completed.stdout

    commit = run_git("rev-parse", "HEAD").decode("ascii").strip()
    tree = run_git("rev-parse", "HEAD^{tree}").decode("ascii").strip()
    status = run_git("status", "--porcelain=v1", "--untracked-files=no")
    archive = run_git("archive", "--format=tar", "HEAD")
    imported_inventory: list[tuple[str, str]] = []
    for relative_path in MASSIVE_IMPORTED_PIPELINE_SOURCE_RELATIVE_PATHS:
        source_path = (repository_root / relative_path).resolve(strict=True)
        if not source_path.is_relative_to(repository_root):
            raise MassiveRuntimeAuthorityError("imported source escaped the Git root")
        tracked_blob = run_git("rev-parse", f"HEAD:{relative_path}").decode("ascii").strip()
        current_blob = run_git("hash-object", str(source_path)).decode("ascii").strip()
        if tracked_blob != current_blob:
            raise MassiveRuntimeAuthorityError("imported pipeline source differs from Git HEAD")
        imported_inventory.append((relative_path, file_sha256(source_path)))
    return {
        "git_commit": commit,
        "git_tree": tree,
        "git_archive_sha256": _sha256_bytes(archive),
        "tracked_worktree_clean": status == b"",
        "imported_source_count": len(imported_inventory),
        "imported_source_inventory_sha256": semantic_sha256(
            tuple(imported_inventory)
        ),
    }


def _mount_provenance(path: Path, mountinfo: bytes) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    candidates: list[tuple[int, dict[str, object]]] = []
    for raw_line in mountinfo.decode("utf-8").splitlines():
        before, separator, after = raw_line.partition(" - ")
        if not separator:
            continue
        left = before.split()
        right = after.split()
        if len(left) < 6 or len(right) < 3:
            continue
        mount_point = Path(left[4].replace("\\040", " ")).resolve()
        if resolved != mount_point and not resolved.is_relative_to(mount_point):
            continue
        options = set(left[5].split(",")) | set(right[2].split(","))
        candidates.append(
            (
                len(mount_point.parts),
                {
                    "mount_point": str(mount_point),
                    "filesystem_type": right[0],
                    "mount_source_sha256": _sha256_bytes(right[1].encode("utf-8")),
                    "read_only": "ro" in options,
                },
            )
        )
    if not candidates:
        raise MassiveRuntimeAuthorityError("runtime metadata mount was not resolved")
    result = max(candidates, key=lambda item: item[0])[1]
    if not result["read_only"]:
        raise MassiveRuntimeAuthorityError("runtime metadata mount is not read-only")
    return result


def _container_runtime_payload(cgroup: bytes, mountinfo: bytes) -> dict[str, object]:
    image_bytes = _read_required(MASSIVE_CONTAINER_IMAGE_DIGEST_PATH).strip()
    metadata_bytes = _read_required(MASSIVE_CONTAINER_RUNTIME_METADATA_PATH)
    try:
        image_text = image_bytes.decode("ascii")
        metadata = json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveRuntimeAuthorityError("container runtime metadata is malformed") from exc
    if canonical_json_file_bytes(metadata) != metadata_bytes:
        raise MassiveRuntimeAuthorityError("container runtime metadata is not canonical")
    if not isinstance(metadata, dict) or set(metadata) != {
        "container_id",
        "image_digest",
        "runtime_kind",
    }:
        raise MassiveRuntimeAuthorityError("container runtime metadata fields differ")
    runtime_kind = _text("runtime kind", metadata["runtime_kind"])
    container_id = _text("container ID", metadata["container_id"])
    metadata_digest = _text("container image digest", metadata["image_digest"])
    if runtime_kind != _runtime_kind(cgroup, mountinfo):
        raise MassiveRuntimeAuthorityError("container runtime metadata kind differs")
    try:
        container_id_bytes = container_id.encode("ascii")
    except UnicodeEncodeError as exc:
        raise MassiveRuntimeAuthorityError("container ID is not ASCII") from exc
    if container_id_bytes not in cgroup:
        raise MassiveRuntimeAuthorityError("container ID is not bound to this process")
    if not image_text.startswith("sha256:") or metadata_digest != image_text:
        raise MassiveRuntimeAuthorityError("container image digest evidence differs")
    image_digest = _digest("container image digest", image_text.removeprefix("sha256:"))
    mount_inventory = (
        _mount_provenance(MASSIVE_CONTAINER_IMAGE_DIGEST_PATH, mountinfo),
        _mount_provenance(MASSIVE_CONTAINER_RUNTIME_METADATA_PATH, mountinfo),
    )
    return {
        "runtime_kind": runtime_kind,
        "cgroup_sha256": _sha256_bytes(cgroup),
        "mountinfo_sha256": _sha256_bytes(mountinfo),
        "container_id_sha256": _sha256_bytes(container_id_bytes),
        "container_image_digest_sha256": image_digest,
        "image_digest_source_sha256": _sha256_bytes(image_bytes),
        "runtime_metadata_sha256": _sha256_bytes(metadata_bytes),
        "runtime_mount_receipt_sha256": semantic_sha256(mount_inventory),
    }


def _runtime_kind(cgroup: bytes, mountinfo: bytes) -> str:
    lowered = (cgroup + b"\n" + mountinfo).lower()
    if b"kubepods" in lowered:
        return "kubernetes"
    if b"podman" in lowered:
        return "podman"
    if b"docker" in lowered:
        return "docker"
    if b"containerd" in lowered or Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return "container"
    raise MassiveRuntimeAuthorityError("execution is not inside a qualified container runtime")


def _runtime_endpoint(s3_client: Any) -> str:
    endpoint = getattr(getattr(s3_client, "meta", None), "endpoint_url", None)
    if not isinstance(endpoint, str) or not endpoint:
        endpoint = getattr(getattr(s3_client, "_endpoint", None), "host", None)
    return _text("runtime endpoint", endpoint)


def _storage_payload(storage_roots: dict[str, str | Path]) -> list[dict[str, object]]:
    if not storage_roots:
        raise MassiveRuntimeAuthorityError("storage roots are empty")
    result: list[dict[str, object]] = []
    for role, raw_path in sorted(storage_roots.items()):
        _text("storage role", role)
        path = Path(raw_path).resolve(strict=True)
        info = path.stat()
        fs = os.statvfs(path)
        result.append(
            {
                "role": role,
                "resolved_path": str(path),
                "device": info.st_dev,
                "block_size": fs.f_frsize,
                "total_blocks": fs.f_blocks,
                "available_blocks": fs.f_bavail,
            }
        )
    return result


def _environment_payload(
    *,
    host_authority: MassiveHostExecutionAuthorityV2,
    s3_client: Any,
    storage_roots: dict[str, str | Path],
    pipeline_implementation_inventory_sha256: str,
    started_at_ms: int,
) -> dict[str, object]:
    cgroup = _read_required(Path("/proc/self/cgroup"))
    mountinfo = _read_required(Path("/proc/self/mountinfo"))
    executable = Path(sys.executable).resolve(strict=True)
    distributions = tuple(
        sorted(
            (distribution.metadata["Name"].lower(), distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        )
    )
    return {
        "host_authority_receipt_sha256": host_authority.receipt_sha256,
        "capture_started_at_ms": started_at_ms,
        "capture_finished_at_ms": _wall_ms(),
        "source_archive": _source_archive_payload(),
        "container_runtime": _container_runtime_payload(cgroup, mountinfo),
        "python_environment": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_path": str(executable),
            "executable_sha256": file_sha256(executable),
            "distribution_count": len(distributions),
            "distribution_inventory_sha256": semantic_sha256(distributions),
        },
        "network_acquisition": {
            "endpoint": MASSIVE_FLAT_FILE_ENDPOINT,
            "bucket": MASSIVE_FLAT_FILE_BUCKET,
            "client_module": type(s3_client).__module__,
            "client_qualname": type(s3_client).__qualname__,
            "runtime_endpoint": _runtime_endpoint(s3_client),
        },
        "storage_roots": _storage_payload(storage_roots),
        "pipeline_implementation_inventory_sha256": _digest(
            "pipeline implementation inventory",
            pipeline_implementation_inventory_sha256,
        ),
    }


def parse_massive_runtime_execution_environment_v2(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    host_authority: MassiveHostExecutionAuthorityV2,
    host_root: str | Path,
) -> MassiveRuntimeExecutionEnvironmentAuthorityV2:
    expected_host = parse_massive_host_execution_authority_v2(
        root=host_root, loaded_source=host_authority.loaded_source
    )
    if (
        expected_host.loaded_source.receipt_sha256
        != host_authority.loaded_source.receipt_sha256
        or expected_host.host_id != host_authority.host_id
    ):
        raise MassiveRuntimeAuthorityError("environment host authority was not rederived")
    payload = _canonical_source_payload(
        root=root,
        loaded_source=loaded_source,
        dataset_id=MASSIVE_RUNTIME_ENVIRONMENT_V2_DATASET,
        schema_sha256=MASSIVE_RUNTIME_ENVIRONMENT_V2_SOURCE_SCHEMA_SHA256,
    )
    expected = {
        "host_authority_receipt_sha256",
        "capture_started_at_ms",
        "capture_finished_at_ms",
        "source_archive",
        "container_runtime",
        "python_environment",
        "network_acquisition",
        "storage_roots",
        "pipeline_implementation_inventory_sha256",
    }
    if set(payload) != expected or payload["host_authority_receipt_sha256"] != host_authority.receipt_sha256:
        raise MassiveRuntimeAuthorityError("runtime environment fields differ")
    component_specs: tuple[tuple[str, type[Any], set[str]], ...] = (
        ("source_archive", MassiveSourceArchiveAuthorityV2, {"git_commit", "git_tree", "git_archive_sha256", "tracked_worktree_clean", "imported_source_count", "imported_source_inventory_sha256"}),
        ("container_runtime", MassiveContainerRuntimeAuthorityV2, {"runtime_kind", "cgroup_sha256", "mountinfo_sha256", "container_id_sha256", "container_image_digest_sha256", "image_digest_source_sha256", "runtime_metadata_sha256", "runtime_mount_receipt_sha256"}),
        ("python_environment", MassivePythonEnvironmentAuthorityV2, {"implementation", "version", "executable_path", "executable_sha256", "distribution_count", "distribution_inventory_sha256"}),
        ("network_acquisition", MassiveNetworkAcquisitionAuthorityV2, {"endpoint", "bucket", "client_module", "client_qualname", "runtime_endpoint"}),
    )
    components: dict[str, object] = {}
    for name, cls, keys in component_specs:
        value = payload[name]
        if not isinstance(value, dict) or set(value) != keys:
            raise MassiveRuntimeAuthorityError(f"{name} fields differ")
        components[name] = _receipt_dataclass(cls, dict(value))
    roots_value = payload["storage_roots"]
    if not isinstance(roots_value, list):
        raise MassiveRuntimeAuthorityError("storage roots differ")
    roots: list[MassiveStorageRootAuthorityV2] = []
    for value in roots_value:
        if not isinstance(value, dict) or set(value) != {"role", "resolved_path", "device", "block_size", "total_blocks", "available_blocks"}:
            raise MassiveRuntimeAuthorityError("storage root fields differ")
        roots.append(_receipt_dataclass(MassiveStorageRootAuthorityV2, dict(value)))
    ordered_roots = tuple(sorted(roots, key=lambda row: row.role))
    if tuple(row.role for row in roots) != tuple(row.role for row in ordered_roots):
        raise MassiveRuntimeAuthorityError("storage root ordering differs")
    body = {
        "schema": MASSIVE_RUNTIME_ENVIRONMENT_V2_SCHEMA,
        "host_authority_receipt_sha256": host_authority.receipt_sha256,
        **components,
        "storage_roots": ordered_roots,
        "pipeline_implementation_inventory_sha256": payload["pipeline_implementation_inventory_sha256"],
        "capture_started_at_ms": payload["capture_started_at_ms"],
        "capture_finished_at_ms": payload["capture_finished_at_ms"],
        "loaded_source": loaded_source,
        "environment_spec_receipt_sha256": MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256,
        "parser_source_sha256": MASSIVE_RUNTIME_AUTHORITY_SOURCE_SHA256,
        "captured_by_fixed_runtime": False,
    }
    provisional = MassiveRuntimeExecutionEnvironmentAuthorityV2(**body, receipt_sha256="0" * 64)  # type: ignore[arg-type]
    result = replace(provisional, receipt_sha256=semantic_sha256(provisional.unsigned()))
    result.validate()
    return result


def capture_massive_runtime_execution_environment_v2(
    *,
    root: str | Path,
    host_authority: MassiveHostExecutionAuthorityV2,
    host_root: str | Path,
    s3_client: Any,
    storage_roots: dict[str, str | Path],
    pipeline_implementation_inventory_sha256: str,
    entitlement_receipt_sha256: str,
) -> MassiveRuntimeExecutionEnvironmentAuthorityV2:
    """Capture the actual process, source, container, network, and storage identities."""

    _digest("entitlement receipt", entitlement_receipt_sha256)
    expected_host = parse_massive_host_execution_authority_v2(
        root=host_root, loaded_source=host_authority.loaded_source
    )
    if (
        expected_host.loaded_source.receipt_sha256
        != host_authority.loaded_source.receipt_sha256
        or expected_host.host_id != host_authority.host_id
        or not host_authority.captured_by_fixed_runtime
    ):
        raise MassiveRuntimeAuthorityError("environment capture host was not rederived")
    verify_current_execution_host_v2(host_authority)
    started_at_ms = _wall_ms()
    payload = _environment_payload(
        host_authority=host_authority,
        s3_client=s3_client,
        storage_roots=storage_roots,
        pipeline_implementation_inventory_sha256=pipeline_implementation_inventory_sha256,
        started_at_ms=started_at_ms,
    )
    finished_at_ms = cast(int, payload["capture_finished_at_ms"])
    relative_path = f"execution/environment-v2/{started_at_ms}-{uuid.uuid4().hex}.json"
    loaded = _publish_payload(
        root=root,
        relative_path=relative_path,
        dataset_id=MASSIVE_RUNTIME_ENVIRONMENT_V2_DATASET,
        schema_sha256=MASSIVE_RUNTIME_ENVIRONMENT_V2_SOURCE_SCHEMA_SHA256,
        payload=payload,
        requested_at_ms=started_at_ms,
        downloaded_at_ms=finished_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        request_id="fixed-runtime-environment-capture-v2",
    )
    parsed = parse_massive_runtime_execution_environment_v2(
        root=root,
        loaded_source=loaded,
        host_authority=host_authority,
        host_root=host_root,
    )
    captured = replace(parsed, captured_by_fixed_runtime=True, receipt_sha256="0" * 64)
    captured = replace(
        captured,
        receipt_sha256=semantic_sha256(captured.unsigned()),
    )
    captured.validate()
    return captured


__all__ = [
    "MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256",
    "MASSIVE_HOST_EXECUTION_V2_DATASET",
    "MASSIVE_HOST_EXECUTION_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256",
    "MASSIVE_RAW_CHRONY_V2_DATASET",
    "MASSIVE_RAW_CHRONY_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_RUNTIME_ENVIRONMENT_V2_DATASET",
    "MASSIVE_RUNTIME_ENVIRONMENT_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256",
    "MassiveContainerRuntimeAuthorityV2",
    "MassiveExecutionClockAuthorityV2",
    "MassiveHostExecutionAuthorityV2",
    "MassiveNetworkAcquisitionAuthorityV2",
    "MassivePythonEnvironmentAuthorityV2",
    "MassiveRawCommandEvidenceV2",
    "MassiveRuntimeAuthorityError",
    "MassiveRuntimeExecutionEnvironmentAuthorityV2",
    "MassiveSourceArchiveAuthorityV2",
    "MassiveStorageRootAuthorityV2",
    "capture_massive_execution_clock_authority_v2",
    "capture_massive_host_execution_authority_v2",
    "capture_massive_runtime_execution_environment_v2",
    "conservative_chrony_seconds_to_ns_v2",
    "parse_massive_execution_clock_authority_v2",
    "parse_massive_host_execution_authority_v2",
    "parse_massive_runtime_execution_environment_v2",
    "verify_current_execution_host_v2",
]
