"""Create-only envelopes for bounded P0 feature and target authorities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_profitability_data_gate_v1 import (
    MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_feature_accounting_v1 import (
    MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MASSIVE_PROFITABILITY_FILL_WINDOW_V1_SCHEMA,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_targets_v1 import (
    MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)

MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SCHEMA = (
    "rl-quant.massive-profitability-bounded-artifact-v1"
)
MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_DATASET = (
    "massive-profitability-bounded-artifact-v1"
)
MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_BOUNDED_COMPONENT_SCHEMAS_V1 = tuple(
    sorted(
        (
            MASSIVE_PROFITABILITY_DATA_GATE_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FILL_WINDOW_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
            MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_V1_SCHEMA,
            MASSIVE_PROFITABILITY_TARGETS_V1_SCHEMA,
        )
    )
)


class MassiveProfitabilityBoundedArtifactV1Error(ValueError):
    """A bounded component envelope or its committed bytes differ."""


class _BoundedComponent(Protocol):
    schema: str

    def validate(self) -> None: ...


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityBoundedArtifactV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveProfitabilityBoundedArtifactV1Error(
            "bounded artifact ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityBoundedArtifactV1:
    component_schema: str
    component_semantic_receipt_sha256: str
    component_payload_sha256: str
    implementation_source_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    runtime_authorizing: bool
    schema: str = MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "component_schema": self.component_schema,
            "component_semantic_receipt_sha256": (
                self.component_semantic_receipt_sha256
            ),
            "component_payload_sha256": self.component_payload_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
            "runtime_authorizing": self.runtime_authorizing,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SCHEMA
            or self.component_schema
            not in MASSIVE_PROFITABILITY_BOUNDED_COMPONENT_SCHEMAS_V1
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SOURCE_SHA256
            or self.runtime_authorizing is not False
        ):
            raise MassiveProfitabilityBoundedArtifactV1Error(
                "bounded component envelope identity differs"
            )
        for value in (
            self.component_semantic_receipt_sha256,
            self.component_payload_sha256,
            self.implementation_source_sha256,
            self.receipt_sha256,
        ):
            _digest("bounded component digest", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SOURCE_SCHEMA_SHA256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityBoundedArtifactV1Error(
                "bounded component envelope receipt differs"
            )


def _payload(
    *,
    component_schema: str,
    component_semantic_receipt_sha256: str,
    component_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SCHEMA,
        "component_schema": component_schema,
        "component_semantic_receipt_sha256": component_semantic_receipt_sha256,
        "component_payload": component_payload,
        "component_payload_sha256": semantic_sha256(component_payload),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SOURCE_SHA256
        ),
    }


def materialize_massive_profitability_bounded_artifact_v1(
    *,
    root: str | Path,
    component: _BoundedComponent,
    component_semantic_receipt_sha256: str,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityBoundedArtifactV1:
    """Publish exact validated component bytes; the generic envelope never authorizes."""

    component.validate()
    if component.schema not in MASSIVE_PROFITABILITY_BOUNDED_COMPONENT_SCHEMAS_V1:
        raise MassiveProfitabilityBoundedArtifactV1Error(
            "bounded component schema is unsupported"
        )
    semantic_receipt = _digest(
        "bounded component semantics", component_semantic_receipt_sha256
    )
    observed = getattr(component, "semantic_receipt_sha256", None)
    if observed != semantic_receipt:
        raise MassiveProfitabilityBoundedArtifactV1Error(
            "bounded component semantic receipt differs"
        )
    component_payload = asdict(component)  # type: ignore[call-overload]
    payload = _payload(
        component_schema=component.schema,
        component_semantic_receipt_sha256=semantic_receipt,
        component_payload=component_payload,
    )
    identifier = _artifact_id(artifact_id)
    relative = f"massive-profitability/bounded-v1/{component.schema}-{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "bounded artifact entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"P0-BOUNDED-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_profitability_bounded_artifact_v1(
        root=root, loaded_source=loaded
    )


def parse_massive_profitability_bounded_artifact_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityBoundedArtifactV1:
    """Reopen exact bytes; generic reloads are deliberately nonauthorizing."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityBoundedArtifactV1Error(
            "bounded component source is not JSON"
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema",
            "component_schema",
            "component_semantic_receipt_sha256",
            "component_payload",
            "component_payload_sha256",
            "implementation_source_sha256",
        }
        or not isinstance(payload["component_payload"], dict)
        or raw != canonical_json_file_bytes(payload)
        or payload["component_payload_sha256"]
        != semantic_sha256(payload["component_payload"])
        or payload["component_payload"].get("schema") != payload["component_schema"]
        or payload["component_payload"].get("semantic_receipt_sha256")
        != payload["component_semantic_receipt_sha256"]
    ):
        raise MassiveProfitabilityBoundedArtifactV1Error(
            "bounded component canonical payload differs"
        )
    body = {
        "component_schema": payload["component_schema"],
        "component_semantic_receipt_sha256": payload[
            "component_semantic_receipt_sha256"
        ],
        "component_payload_sha256": payload["component_payload_sha256"],
        "implementation_source_sha256": payload["implementation_source_sha256"],
        "loaded_source": loaded_source,
        "runtime_authorizing": False,
        "schema": payload["schema"],
    }
    provisional = MassiveProfitabilityBoundedArtifactV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,
    )
    result = MassiveProfitabilityBoundedArtifactV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_DATASET",
    "MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_BOUNDED_ARTIFACT_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_BOUNDED_COMPONENT_SCHEMAS_V1",
    "MassiveProfitabilityBoundedArtifactV1",
    "MassiveProfitabilityBoundedArtifactV1Error",
    "materialize_massive_profitability_bounded_artifact_v1",
    "parse_massive_profitability_bounded_artifact_v1",
]
