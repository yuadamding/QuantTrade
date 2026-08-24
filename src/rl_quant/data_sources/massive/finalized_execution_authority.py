"""Committed execution-clock, environment, and input-availability authorities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)

MASSIVE_EXECUTION_CLOCK_V1_SCHEMA = "rl-quant.massive-execution-clock-authority-v1"
MASSIVE_EXECUTION_CLOCK_V1_DATASET = "massive-finalized-execution-clock-v1"
MASSIVE_EXECUTION_CLOCK_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "format": "canonical-json",
        "source": "qualified-chrony-tracking-measurement",
        "fields": (
            "host_id",
            "clock_source",
            "synchronization_protocol",
            "measurement_observed_at_ms",
            "qualification_valid_until_ms",
            "last_offset_ns",
            "rms_offset_ns",
            "root_dispersion_ns",
            "frequency_skew_ppm",
        ),
    }
)
MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "committed-chrony-tracking-measurement",
        "maximum_qualification_window_ms": 60 * 60 * 1_000,
        "wall_clock": "time.time_ns",
        "monotonic_clock": "time.perf_counter_ns",
        "error_bound": "max(abs(last),rms,dispersion)+frequency-drift",
    }
)
MASSIVE_EXECUTION_ENVIRONMENT_V1_SCHEMA = (
    "rl-quant.massive-typed-execution-environment-v1"
)
MASSIVE_EXECUTION_ENVIRONMENT_V1_DATASET = (
    "massive-finalized-typed-execution-environment-v1"
)
MASSIVE_EXECUTION_ENVIRONMENT_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "format": "canonical-json",
        "fields": (
            "hardware_authority_receipt_sha256",
            "software_source_archive_sha256",
            "container_image_receipt_sha256",
            "python_environment_receipt_sha256",
            "network_contract_receipt_sha256",
            "storage_contract_receipt_sha256",
            "pipeline_implementation_inventory_sha256",
        ),
    }
)
MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "committed-execution-environment-manifest",
        "identities": (
            "hardware",
            "source-archive",
            "container",
            "python-environment",
            "network",
            "storage",
            "pipeline-implementation-inventory",
        ),
    }
)
MASSIVE_INPUT_AVAILABILITY_V1_SCHEMA = (
    "rl-quant.massive-typed-input-availability-authority-v1"
)
MASSIVE_INPUT_AVAILABILITY_V1_DATASET = "massive-finalized-typed-input-availability-v1"
MASSIVE_INPUT_AVAILABILITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "format": "canonical-json",
        "rows": (
            "input_kind",
            "artifact_receipt_sha256",
            "evidence_receipt_sha256",
            "available_at_ms",
        ),
    }
)
MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "ordering": "input-kind-then-artifact-receipt",
        "uniqueness": "input-kind",
        "production_gate": "clock-adjusted-availability-before-outer-start",
        "source_backed_inputs": "timestamps-rederived-from-loaded-source",
    }
)
MASSIVE_FINALIZED_EXECUTION_AUTHORITY_SOURCE_SHA256 = file_sha256(Path(__file__))


class MassiveFinalizedExecutionAuthorityError(ValueError):
    """Committed timing or execution evidence differs from its claimed authority."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFinalizedExecutionAuthorityError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _loaded_source(
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
        raise MassiveFinalizedExecutionAuthorityError(
            "execution authority source dataset/schema differ"
        )
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveFinalizedExecutionAuthorityError(
            "execution authority source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveFinalizedExecutionAuthorityError(
            "execution authority source is not canonical"
        )
    return payload


@dataclass(frozen=True, slots=True)
class MassiveExecutionClockAuthorityV1:
    host_id: str
    clock_source: str
    synchronization_protocol: str
    measurement_observed_at_ms: int
    qualification_valid_until_ms: int
    last_offset_ns: int
    rms_offset_ns: int
    root_dispersion_ns: int
    frequency_skew_ppm: float
    maximum_clock_error_ns: int
    loaded_source: LoadedMassiveSourceObject
    clock_spec_receipt_sha256: str
    parser_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_EXECUTION_CLOCK_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    @property
    def maximum_clock_error_ms(self) -> int:
        return math.ceil(self.maximum_clock_error_ns / 1_000_000)

    def utc_lower_bound_ms(self, local_timestamp_ms: int) -> int:
        self.validate()
        return local_timestamp_ms - self.maximum_clock_error_ms

    def utc_upper_bound_ms(self, local_timestamp_ms: int) -> int:
        self.validate()
        return local_timestamp_ms + self.maximum_clock_error_ms

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_EXECUTION_CLOCK_V1_SCHEMA
            or self.clock_source != "chrony-tracking"
            or self.synchronization_protocol not in {"NTP", "PTP"}
            or not self.host_id
            or self.measurement_observed_at_ms < 0
            or self.qualification_valid_until_ms < self.measurement_observed_at_ms
            or self.qualification_valid_until_ms - self.measurement_observed_at_ms
            > 60 * 60 * 1_000
            or isinstance(self.frequency_skew_ppm, bool)
            or not math.isfinite(self.frequency_skew_ppm)
            or self.frequency_skew_ppm < 0
            or self.rms_offset_ns < 0
            or self.root_dispersion_ns < 0
        ):
            raise MassiveFinalizedExecutionAuthorityError(
                "execution clock authority differs"
            )
        validity_ns = (
            self.qualification_valid_until_ms - self.measurement_observed_at_ms
        ) * 1_000_000
        drift_ns = math.ceil(validity_ns * self.frequency_skew_ppm / 1_000_000)
        expected_error = (
            max(abs(self.last_offset_ns), self.rms_offset_ns, self.root_dispersion_ns)
            + drift_ns
        )
        if self.maximum_clock_error_ns != expected_error:
            raise MassiveFinalizedExecutionAuthorityError(
                "execution clock error bound differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_EXECUTION_CLOCK_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_EXECUTION_CLOCK_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.commit.committed_at_ms
            > self.qualification_valid_until_ms
            or self.loaded_source.verified_at_ms > self.qualification_valid_until_ms
        ):
            raise MassiveFinalizedExecutionAuthorityError(
                "execution clock source differs"
            )
        for name in (
            "clock_spec_receipt_sha256",
            "parser_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.clock_spec_receipt_sha256 != MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256
            or self.parser_source_sha256
            != MASSIVE_FINALIZED_EXECUTION_AUTHORITY_SOURCE_SHA256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveFinalizedExecutionAuthorityError(
                "execution clock contract differs"
            )


def parse_massive_execution_clock_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveExecutionClockAuthorityV1:
    payload = _loaded_source(
        root=root,
        loaded_source=loaded_source,
        dataset_id=MASSIVE_EXECUTION_CLOCK_V1_DATASET,
        schema_sha256=MASSIVE_EXECUTION_CLOCK_V1_SOURCE_SCHEMA_SHA256,
    )
    expected_keys = {
        "host_id",
        "clock_source",
        "synchronization_protocol",
        "measurement_observed_at_ms",
        "qualification_valid_until_ms",
        "last_offset_ns",
        "rms_offset_ns",
        "root_dispersion_ns",
        "frequency_skew_ppm",
    }
    if set(payload) != expected_keys:
        raise MassiveFinalizedExecutionAuthorityError(
            "execution clock field inventory differs"
        )
    integer_fields = (
        "measurement_observed_at_ms",
        "qualification_valid_until_ms",
        "last_offset_ns",
        "rms_offset_ns",
        "root_dispersion_ns",
    )
    if (
        any(
            isinstance(payload[name], bool) or not isinstance(payload[name], int)
            for name in integer_fields
        )
        or isinstance(payload["frequency_skew_ppm"], bool)
        or not isinstance(payload["frequency_skew_ppm"], (int, float))
        or any(
            not isinstance(payload[name], str)
            for name in ("host_id", "clock_source", "synchronization_protocol")
        )
    ):
        raise MassiveFinalizedExecutionAuthorityError(
            "execution clock value types differ"
        )
    try:
        validity_ns = (
            cast(int, payload["qualification_valid_until_ms"])
            - cast(int, payload["measurement_observed_at_ms"])
        ) * 1_000_000
        drift_ns = math.ceil(
            validity_ns
            * float(cast(int | float, payload["frequency_skew_ppm"]))
            / 1_000_000
        )
        maximum_error = (
            max(
                abs(cast(int, payload["last_offset_ns"])),
                cast(int, payload["rms_offset_ns"]),
                cast(int, payload["root_dispersion_ns"]),
            )
            + drift_ns
        )
        body = {
            "schema": MASSIVE_EXECUTION_CLOCK_V1_SCHEMA,
            **payload,
            "maximum_clock_error_ns": maximum_error,
            "loaded_source": loaded_source,
            "clock_spec_receipt_sha256": MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256,
            "parser_source_sha256": MASSIVE_FINALIZED_EXECUTION_AUTHORITY_SOURCE_SHA256,
        }
    except (TypeError, ValueError) as exc:
        raise MassiveFinalizedExecutionAuthorityError(
            "execution clock values are malformed"
        ) from exc
    provisional = MassiveExecutionClockAuthorityV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveTypedExecutionEnvironmentAuthorityV1:
    hardware_authority_receipt_sha256: str
    software_source_archive_sha256: str
    container_image_receipt_sha256: str
    python_environment_receipt_sha256: str
    network_contract_receipt_sha256: str
    storage_contract_receipt_sha256: str
    pipeline_implementation_inventory_sha256: str
    loaded_source: LoadedMassiveSourceObject
    environment_spec_receipt_sha256: str
    parser_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_EXECUTION_ENVIRONMENT_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_EXECUTION_ENVIRONMENT_V1_SCHEMA:
            raise MassiveFinalizedExecutionAuthorityError(
                "execution environment schema differs"
            )
        for name in (
            "hardware_authority_receipt_sha256",
            "software_source_archive_sha256",
            "container_image_receipt_sha256",
            "python_environment_receipt_sha256",
            "network_contract_receipt_sha256",
            "storage_contract_receipt_sha256",
            "pipeline_implementation_inventory_sha256",
            "environment_spec_receipt_sha256",
            "parser_source_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_EXECUTION_ENVIRONMENT_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_EXECUTION_ENVIRONMENT_V1_SOURCE_SCHEMA_SHA256
            or self.environment_spec_receipt_sha256
            != MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
            or self.parser_source_sha256
            != MASSIVE_FINALIZED_EXECUTION_AUTHORITY_SOURCE_SHA256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveFinalizedExecutionAuthorityError(
                "execution environment contract differs"
            )


def parse_massive_typed_execution_environment_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveTypedExecutionEnvironmentAuthorityV1:
    payload = _loaded_source(
        root=root,
        loaded_source=loaded_source,
        dataset_id=MASSIVE_EXECUTION_ENVIRONMENT_V1_DATASET,
        schema_sha256=MASSIVE_EXECUTION_ENVIRONMENT_V1_SOURCE_SCHEMA_SHA256,
    )
    expected_keys = {
        "hardware_authority_receipt_sha256",
        "software_source_archive_sha256",
        "container_image_receipt_sha256",
        "python_environment_receipt_sha256",
        "network_contract_receipt_sha256",
        "storage_contract_receipt_sha256",
        "pipeline_implementation_inventory_sha256",
    }
    if set(payload) != expected_keys:
        raise MassiveFinalizedExecutionAuthorityError(
            "execution environment field inventory differs"
        )
    for name in expected_keys:
        _digest(name, payload[name])
    body = {
        "schema": MASSIVE_EXECUTION_ENVIRONMENT_V1_SCHEMA,
        **payload,
        "loaded_source": loaded_source,
        "environment_spec_receipt_sha256": MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
        "parser_source_sha256": MASSIVE_FINALIZED_EXECUTION_AUTHORITY_SOURCE_SHA256,
    }
    provisional = MassiveTypedExecutionEnvironmentAuthorityV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveInputAvailabilityRowV1:
    input_kind: str
    artifact_receipt_sha256: str
    evidence_receipt_sha256: str
    available_at_ms: int
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.input_kind
            or self.input_kind != self.input_kind.strip()
            or isinstance(self.available_at_ms, bool)
            or not isinstance(self.available_at_ms, int)
            or self.available_at_ms < 0
        ):
            raise MassiveFinalizedExecutionAuthorityError(
                "input availability row differs"
            )
        for name in (
            "artifact_receipt_sha256",
            "evidence_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedExecutionAuthorityError(
                "input availability row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveInputAvailabilityAuthorityV1:
    rows: tuple[MassiveInputAvailabilityRowV1, ...]
    loaded_source: LoadedMassiveSourceObject
    availability_spec_receipt_sha256: str
    parser_source_sha256: str
    row_inventory_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_INPUT_AVAILABILITY_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def resolve(self, input_kind: str) -> MassiveInputAvailabilityRowV1:
        self.validate()
        for row in self.rows:
            if row.input_kind == input_kind:
                return row
        raise MassiveFinalizedExecutionAuthorityError(
            f"input availability is absent for {input_kind}"
        )

    def validate(self) -> None:
        keys = tuple(row.input_kind for row in self.rows)
        if (
            self.schema != MASSIVE_INPUT_AVAILABILITY_V1_SCHEMA
            or not keys
            or keys != tuple(sorted(set(keys)))
        ):
            raise MassiveFinalizedExecutionAuthorityError(
                "input availability inventory differs"
            )
        for row in self.rows:
            row.validate()
        for name in (
            "availability_spec_receipt_sha256",
            "parser_source_sha256",
            "row_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_INPUT_AVAILABILITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_INPUT_AVAILABILITY_V1_SOURCE_SCHEMA_SHA256
            or self.availability_spec_receipt_sha256
            != MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256
            or self.parser_source_sha256
            != MASSIVE_FINALIZED_EXECUTION_AUTHORITY_SOURCE_SHA256
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveFinalizedExecutionAuthorityError(
                "input availability authority differs"
            )


def parse_massive_input_availability_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveInputAvailabilityAuthorityV1:
    payload = _loaded_source(
        root=root,
        loaded_source=loaded_source,
        dataset_id=MASSIVE_INPUT_AVAILABILITY_V1_DATASET,
        schema_sha256=MASSIVE_INPUT_AVAILABILITY_V1_SOURCE_SCHEMA_SHA256,
    )
    if set(payload) != {"rows"} or not isinstance(payload["rows"], list):
        raise MassiveFinalizedExecutionAuthorityError(
            "input availability source fields differ"
        )
    rows: list[MassiveInputAvailabilityRowV1] = []
    for value in payload["rows"]:
        if not isinstance(value, dict) or set(value) != {
            "input_kind",
            "artifact_receipt_sha256",
            "evidence_receipt_sha256",
            "available_at_ms",
        }:
            raise MassiveFinalizedExecutionAuthorityError(
                "input availability row fields differ"
            )
        body = dict(value)
        row_provisional = MassiveInputAvailabilityRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256="0" * 64,  # type: ignore[arg-type]
        )
        row = replace(
            row_provisional,
            receipt_sha256=semantic_sha256(row_provisional.unsigned()),
        )
        row.validate()
        rows.append(row)
    source_keys = tuple(row.input_kind for row in rows)
    if source_keys != tuple(sorted(set(source_keys))):
        raise MassiveFinalizedExecutionAuthorityError(
            "input availability source rows are not canonical"
        )
    ordered = tuple(sorted(rows, key=lambda row: row.input_kind))
    body = {
        "schema": MASSIVE_INPUT_AVAILABILITY_V1_SCHEMA,
        "rows": ordered,
        "loaded_source": loaded_source,
        "availability_spec_receipt_sha256": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
        "parser_source_sha256": MASSIVE_FINALIZED_EXECUTION_AUTHORITY_SOURCE_SHA256,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
    }
    authority_provisional = MassiveInputAvailabilityAuthorityV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    authority = replace(
        authority_provisional,
        receipt_sha256=semantic_sha256(authority_provisional.unsigned()),
    )
    authority.validate()
    return authority


__all__ = [
    "MASSIVE_EXECUTION_CLOCK_V1_DATASET",
    "MASSIVE_EXECUTION_CLOCK_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256",
    "MASSIVE_EXECUTION_ENVIRONMENT_V1_DATASET",
    "MASSIVE_EXECUTION_ENVIRONMENT_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256",
    "MASSIVE_INPUT_AVAILABILITY_V1_DATASET",
    "MASSIVE_INPUT_AVAILABILITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256",
    "MassiveExecutionClockAuthorityV1",
    "MassiveFinalizedExecutionAuthorityError",
    "MassiveInputAvailabilityAuthorityV1",
    "MassiveInputAvailabilityRowV1",
    "MassiveTypedExecutionEnvironmentAuthorityV1",
    "parse_massive_execution_clock_authority_v1",
    "parse_massive_input_availability_authority_v1",
    "parse_massive_typed_execution_environment_v1",
]
