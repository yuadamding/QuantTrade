"""Create-only receipt artifacts for the typed Massive P0 V2 authorities.

The source authorities remain the executable truth.  These artifacts preserve
their complete semantic payloads and canonical inventories before outcomes are
constructed.  Generic reload is deliberately nonauthorizing; a production
workflow must also retain and revalidate the typed authority object.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_monthly_rank_bar_authority_v1 import (
    MassiveMonthlyRankBarAuthorityV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveMonthlyRankInputAuthorityV2,
    MassiveProfitabilityDecisionOriginPlanV2,
    MassiveProfitabilityProductionAcquisitionV2,
    validate_massive_profitability_production_acquisition_v2,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-profitability-frozen-authority-v1"
)
MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_DATASET = (
    "massive-profitability-frozen-authority-v1"
)
MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "components": (
            "origin-plan-v2",
            "monthly-rank-input-v2",
            "monthly-rank-bar-v1",
        ),
        "payload": "complete-authority-semantic-payload",
        "inventory": "component-specific-canonical-inventory",
        "generic-reload": "nonauthorizing",
    }
)

_COMPONENT_IDS = {
    "origin-plan-v2",
    "monthly-rank-input-v2",
    "monthly-rank-bar-v1",
}


class MassiveProfitabilityFrozenAuthorityV1Error(ValueError):
    """A frozen authority receipt is incomplete or differs from source bytes."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen authority artifact ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFrozenAuthorityArtifactV1:
    component_id: str
    authority_schema: str
    authority_semantic_receipt_sha256: str
    authority_audit_receipt_sha256: str
    authority_semantic_payload_sha256: str
    authority_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    audit_receipt_sha256: str
    runtime_qualified: bool
    schema: str = MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "component_id": self.component_id,
            "authority_schema": self.authority_schema,
            "authority_semantic_receipt_sha256": (
                self.authority_semantic_receipt_sha256
            ),
            "authority_audit_receipt_sha256": self.authority_audit_receipt_sha256,
            "authority_semantic_payload_sha256": (
                self.authority_semantic_payload_sha256
            ),
            "authority_inventory_sha256": self.authority_inventory_sha256,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SCHEMA
            or self.component_id not in _COMPONENT_IDS
            or not self.authority_schema
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SOURCE_SHA256
            or not isinstance(self.runtime_qualified, bool)
        ):
            raise MassiveProfitabilityFrozenAuthorityV1Error(
                "frozen authority identity differs"
            )
        for name in (
            "authority_semantic_receipt_sha256",
            "authority_audit_receipt_sha256",
            "authority_semantic_payload_sha256",
            "authority_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityFrozenAuthorityV1Error(
                "frozen authority semantic identity differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveProfitabilityFrozenAuthorityV1Error(
                "frozen authority source transaction differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
            }
        ):
            raise MassiveProfitabilityFrozenAuthorityV1Error(
                "frozen authority audit receipt differs"
            )


def _materialize(
    *,
    root: str | Path,
    component_id: str,
    authority_schema: str,
    semantic_payload: Mapping[str, object],
    authority_semantic_receipt_sha256: str,
    authority_audit_receipt_sha256: str,
    authority_inventory: object,
    runtime_qualified: bool,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityFrozenAuthorityArtifactV1:
    if component_id not in _COMPONENT_IDS:
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen authority component is unsupported"
        )
    semantic_payload_digest = semantic_sha256(semantic_payload)
    semantic = {
        "schema": MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SCHEMA,
        "component_id": component_id,
        "authority_schema": authority_schema,
        "authority_semantic_receipt_sha256": authority_semantic_receipt_sha256,
        "authority_audit_receipt_sha256": authority_audit_receipt_sha256,
        "authority_semantic_payload_sha256": semantic_payload_digest,
        "authority_inventory_sha256": semantic_sha256(authority_inventory),
        "protocol_receipt_sha256": (MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256),
        "specification_sha256": (MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SPEC_SHA256),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    semantic_receipt = semantic_sha256(semantic)
    payload = {
        **semantic,
        "authority_semantic_payload": semantic_payload,
        "authority_inventory": authority_inventory,
        "semantic_receipt_sha256": semantic_receipt,
    }
    identifier = _artifact_id(artifact_id)
    relative = (
        f"massive-profitability/frozen-authorities-v1/{component_id}-{identifier}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "frozen authority entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"P0-FROZEN-{component_id.upper()}-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_frozen_authority_v1(
        root=root, loaded_source=loaded
    )
    result = replace(parsed, runtime_qualified=runtime_qualified)
    result.validate()
    return result


def materialize_massive_profitability_origin_plan_v2(
    *,
    root: str | Path,
    authority: MassiveProfitabilityDecisionOriginPlanV2,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    monthly_rank_authority: MassiveMonthlyRankInputAuthorityV2,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityFrozenAuthorityArtifactV1:
    authority.validate()
    monthly_rank_authority.validate()
    validate_massive_profitability_production_acquisition_v2(
        root=root, acquisition=acquisition, require_fixed_runtime=True
    )
    if (
        authority.production_acquisition_receipt_sha256 != acquisition.receipt_sha256
        or authority.monthly_rank_authority_semantic_receipt_sha256
        != monthly_rank_authority.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen V2 origin is not bound to the supplied production authorities"
        )
    v1 = authority.origin_plan_v1
    inventory = tuple(
        sorted(
            tuple(
                (row.decision_session_date, "origin", row.receipt_sha256)
                for row in v1.origins
            )
            + tuple(
                (row.decision_session_date, "skip", row.receipt_sha256)
                for row in v1.skipped_decisions
            )
        )
    )
    return _materialize(
        root=root,
        component_id="origin-plan-v2",
        authority_schema=authority.schema,
        semantic_payload={
            "origin_plan_v2": authority.semantic_unsigned(),
            "origin_plan_v1": authority.origin_plan_v1.semantic_unsigned(),
            "origin_plan_v1_semantic_receipt_sha256": (
                authority.origin_plan_v1.semantic_receipt_sha256
            ),
        },
        authority_semantic_receipt_sha256=authority.semantic_receipt_sha256,
        authority_audit_receipt_sha256=authority.audit_receipt_sha256,
        authority_inventory=inventory,
        runtime_qualified=True,
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )


def materialize_massive_monthly_rank_input_authority_v2(
    *,
    root: str | Path,
    authority: MassiveMonthlyRankInputAuthorityV2,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityFrozenAuthorityArtifactV1:
    authority.validate()
    validate_massive_profitability_production_acquisition_v2(
        root=root, acquisition=acquisition, require_fixed_runtime=True
    )
    if authority.acquisition_receipt_sha256 != acquisition.receipt_sha256:
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen monthly rank is not bound to the production acquisition"
        )
    return _materialize(
        root=root,
        component_id="monthly-rank-input-v2",
        authority_schema=authority.schema,
        semantic_payload=authority.semantic_unsigned(),
        authority_semantic_receipt_sha256=authority.semantic_receipt_sha256,
        authority_audit_receipt_sha256=authority.audit_receipt_sha256,
        authority_inventory=tuple(
            (row.calendar_month, row.receipt_sha256) for row in authority.groups
        ),
        runtime_qualified=authority.source_data_qualified,
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )


def materialize_massive_monthly_rank_bar_authority_v1(
    *,
    root: str | Path,
    authority: MassiveMonthlyRankBarAuthorityV1,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityFrozenAuthorityArtifactV1:
    authority.validate()
    validate_massive_profitability_production_acquisition_v2(
        root=root, acquisition=acquisition, require_fixed_runtime=True
    )
    if authority.production_acquisition_receipt_sha256 != acquisition.receipt_sha256:
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen monthly rank bars are not bound to the production acquisition"
        )
    return _materialize(
        root=root,
        component_id="monthly-rank-bar-v1",
        authority_schema=authority.schema,
        semantic_payload=authority.semantic_unsigned(),
        authority_semantic_receipt_sha256=authority.semantic_receipt_sha256,
        authority_audit_receipt_sha256=authority.audit_receipt_sha256,
        authority_inventory=tuple(
            (row.source_session_date, row.receipt_sha256) for row in authority.sessions
        ),
        runtime_qualified=authority.rank_bar_data_qualified,
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )


def parse_massive_profitability_frozen_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityFrozenAuthorityArtifactV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen authority source is not JSON"
        ) from exc
    expected_fields = {
        "schema",
        "component_id",
        "authority_schema",
        "authority_semantic_receipt_sha256",
        "authority_audit_receipt_sha256",
        "authority_semantic_payload_sha256",
        "authority_inventory_sha256",
        "protocol_receipt_sha256",
        "specification_sha256",
        "implementation_source_sha256",
        "authority_semantic_payload",
        "authority_inventory",
        "semantic_receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or raw != canonical_json_file_bytes(payload)
        or semantic_sha256(payload["authority_semantic_payload"])
        != payload["authority_semantic_payload_sha256"]
        or semantic_sha256(payload["authority_inventory"])
        != payload["authority_inventory_sha256"]
    ):
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen authority canonical payload differs"
        )
    component_id = payload["component_id"]
    semantic_payload = payload["authority_semantic_payload"]
    inventory: object
    if component_id == "origin-plan-v2":
        if (
            not isinstance(semantic_payload, dict)
            or set(semantic_payload)
            != {
                "origin_plan_v2",
                "origin_plan_v1",
                "origin_plan_v1_semantic_receipt_sha256",
            }
            or semantic_sha256(semantic_payload["origin_plan_v2"])
            != payload["authority_semantic_receipt_sha256"]
            or semantic_sha256(semantic_payload["origin_plan_v1"])
            != semantic_payload["origin_plan_v1_semantic_receipt_sha256"]
        ):
            raise MassiveProfitabilityFrozenAuthorityV1Error(
                "frozen V2 origin semantic payload differs"
            )
        v1 = semantic_payload["origin_plan_v1"]
        origin_inventory = tuple(
            sorted(
                tuple(
                    (row["decision_session_date"], "origin", row["receipt_sha256"])
                    for row in v1["origins"]
                )
                + tuple(
                    (row["decision_session_date"], "skip", row["receipt_sha256"])
                    for row in v1["skipped_decisions"]
                )
            )
        )
        inventory = origin_inventory
    elif component_id == "monthly-rank-input-v2":
        if (
            not isinstance(semantic_payload, dict)
            or semantic_sha256(semantic_payload)
            != payload["authority_semantic_receipt_sha256"]
        ):
            raise MassiveProfitabilityFrozenAuthorityV1Error(
                "frozen monthly-rank semantic payload differs"
            )
        rank_inventory = tuple(
            (row["calendar_month"], row["receipt_sha256"])
            for row in semantic_payload["groups"]
        )
        inventory = rank_inventory
    elif component_id == "monthly-rank-bar-v1":
        if (
            not isinstance(semantic_payload, dict)
            or semantic_sha256(semantic_payload)
            != payload["authority_semantic_receipt_sha256"]
        ):
            raise MassiveProfitabilityFrozenAuthorityV1Error(
                "frozen monthly-rank-bar semantic payload differs"
            )
        rank_bar_inventory = tuple(
            (row["source_session_date"], row["receipt_sha256"])
            for row in semantic_payload["sessions"]
        )
        inventory = rank_bar_inventory
    else:
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen authority component is unsupported"
        )
    if semantic_sha256(inventory) != payload["authority_inventory_sha256"]:
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen authority inventory was not rederived from semantic rows"
        )
    semantic = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "authority_semantic_payload",
            "authority_inventory",
            "semantic_receipt_sha256",
        }
    }
    if semantic_sha256(semantic) != payload["semantic_receipt_sha256"]:
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen authority semantic receipt differs"
        )
    try:
        result = MassiveProfitabilityFrozenAuthorityArtifactV1(
            **semantic,
            semantic_receipt_sha256=payload["semantic_receipt_sha256"],
            loaded_source=loaded_source,
            audit_receipt_sha256=semantic_sha256(
                {
                    "semantic_receipt_sha256": payload["semantic_receipt_sha256"],
                    "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
                }
            ),
            runtime_qualified=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MassiveProfitabilityFrozenAuthorityV1Error(
            "frozen authority values are malformed"
        ) from exc
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_DATASET",
    "MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_FROZEN_AUTHORITY_V1_SPEC_SHA256",
    "MassiveProfitabilityFrozenAuthorityArtifactV1",
    "MassiveProfitabilityFrozenAuthorityV1Error",
    "materialize_massive_monthly_rank_bar_authority_v1",
    "materialize_massive_monthly_rank_input_authority_v2",
    "materialize_massive_profitability_origin_plan_v2",
    "parse_massive_profitability_frozen_authority_v1",
]
