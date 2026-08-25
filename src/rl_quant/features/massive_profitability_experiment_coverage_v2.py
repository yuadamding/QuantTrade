"""Frozen V2 security support and experiment coverage for Massive P0.

This module is the boundary between the acquired chronology authorities and
the still-unimplemented feature/fill/target data gate.  It fixes candidate,
outer-test, and lockbox dates; persists the experiment support closure; and
explicitly rejects every legacy profitability input generation.  It never
authorizes panel construction, training, reporting, or lockbox access.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from math import ceil
from pathlib import Path

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_economic_return_index_v1 import (
    MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA,
)
from rl_quant.features.massive_monthly_rank_bar_authority_v1 import (
    MassiveMonthlyRankBarAuthorityV1,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MassiveProfitabilityArchiveFreezeV1,
)
from rl_quant.features.massive_profitability_features_v1 import (
    MASSIVE_PROFITABILITY_FEATURES_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_frozen_authorities_v1 import (
    MassiveProfitabilityFrozenAuthorityArtifactV1,
)
from rl_quant.features.massive_profitability_origin_p0 import (
    MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SCHEMA,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveMonthlyRankInputAuthorityV2,
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityPhaseOriginGateV1,
    MassiveProfitabilityPhasePlanV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V2_SCHEMA = (
    "rl-quant.massive-profitability-security-support-v2"
)
MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA = (
    "rl-quant.massive-profitability-experiment-coverage-v2"
)
MASSIVE_PROFITABILITY_EXPERIMENT_V2_DATASET = "massive-profitability-experiment-v2"
MASSIVE_PROFITABILITY_EXPERIMENT_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schemas": (
            MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V2_SCHEMA,
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
        ),
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "identity_equivalence": (
            "normalized-security-master-ticker-listing-delisting-and-chain"
        ),
        "support": "decision-members-plus-predecessor-successor-chain-closure",
        "candidate_dates": "archive-freeze-v1",
        "outer_and_lockbox_dates": "phase-plan-v1-fixed-candidate-dates",
        "minimum_common_support": "max(300,ceil(0.8*decision-members))",
        "legacy_generations": "hard-rejected",
        "feature_fill_target_completeness": "deferred-to-data-gate-v1",
        "performance_authorization": False,
    }
)

MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PANEL_MATERIALIZATION_AUTHORIZED = False
MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_LOCKBOX_ACCESS_AUTHORIZED = False

MASSIVE_PROFITABILITY_LEGACY_GENERATIONS_V2 = tuple(
    sorted(
        (
            MASSIVE_ECONOMIC_RETURN_INDEX_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FEATURES_V1_SCHEMA,
            MASSIVE_PROFITABILITY_ORIGIN_PLAN_P0_SCHEMA,
        )
    )
)


class MassiveProfitabilityExperimentCoverageV2Error(ValueError):
    """Frozen experiment inputs, support, or phase geometry differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityExperimentCoverageV2Error(
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
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "experiment artifact ID is not path safe"
        )
    return value


def reject_massive_profitability_legacy_generation_v2(schema: str) -> None:
    """Fail closed when a legacy economic/origin/feature artifact is offered."""

    if schema in MASSIVE_PROFITABILITY_LEGACY_GENERATIONS_V2:
        raise MassiveProfitabilityExperimentCoverageV2Error(
            f"legacy profitability generation is prohibited: {schema}"
        )


def _identity_semantics(authority: PITSecurityUniverseAuthority) -> dict[str, object]:
    authority.validate()
    return {
        "security_master": tuple(
            (
                row.security_id,
                row.issuer_id,
                row.primary_exchange,
                row.share_class,
                row.security_type,
                row.listing_at_ms,
                row.delisting_at_ms,
                row.successor_security_id,
                row.corporate_action_chain_id,
            )
            for row in authority.security_master
        ),
        "ticker_history": tuple(
            (
                row.security_id,
                row.ticker,
                row.valid_from_ms,
                row.valid_to_ms,
                row.available_at_ms,
                row.primary_exchange,
            )
            for row in authority.ticker_history
        ),
        "listing_events": tuple(
            (
                row.event_id,
                row.security_id,
                row.effective_at_ms,
                row.available_at_ms,
                row.primary_exchange,
                row.ticker,
            )
            for row in authority.listing_events
        ),
        "delisting_events": tuple(
            (
                row.event_id,
                row.security_id,
                row.effective_at_ms,
                row.available_at_ms,
                row.reason,
                row.successor_security_id,
            )
            for row in authority.delisting_events
        ),
    }


def massive_profitability_identity_semantic_receipt_v2(
    authority: PITSecurityUniverseAuthority,
) -> str:
    """Return source-receipt-independent routing identity semantics."""

    return semantic_sha256(_identity_semantics(authority))


def _support_ids(
    *,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    identity_authority: PITSecurityUniverseAuthority,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    decision = {
        security_id
        for origin in origin_plan.origin_plan_v1.origins
        for security_id in origin.decision_member_security_ids
    }
    masters = {row.security_id: row for row in identity_authority.security_master}
    if not decision or not decision <= set(masters):
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "decision membership references absent routing identities"
        )
    adjacency: dict[str, set[str]] = {security_id: set() for security_id in masters}
    chains: defaultdict[str, set[str]] = defaultdict(set)
    for row in masters.values():
        if row.successor_security_id is not None:
            if row.successor_security_id not in masters:
                raise MassiveProfitabilityExperimentCoverageV2Error(
                    "successor identity is absent from experiment routing"
                )
            adjacency[row.security_id].add(row.successor_security_id)
            adjacency[row.successor_security_id].add(row.security_id)
        if row.corporate_action_chain_id is not None:
            chains[row.corporate_action_chain_id].add(row.security_id)
    for members in chains.values():
        for security_id in members:
            adjacency[security_id].update(members - {security_id})
    supported = set(decision)
    queue = deque(sorted(decision))
    while queue:
        current = queue.popleft()
        for linked in sorted(adjacency[current]):
            if linked not in supported:
                supported.add(linked)
                queue.append(linked)
    accounting = supported - decision
    return tuple(sorted(decision)), tuple(sorted(accounting)), tuple(sorted(supported))


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySecuritySupportV2:
    decision_member_security_ids: tuple[str, ...]
    accounting_chain_security_ids: tuple[str, ...]
    all_supported_security_ids: tuple[str, ...]
    decision_origin_member_inventory_sha256: str
    normalized_identity_semantic_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    monthly_rank_semantic_receipt_sha256: str
    monthly_rank_bar_semantic_receipt_sha256: str
    frozen_origin_artifact_semantic_receipt_sha256: str
    frozen_rank_artifact_semantic_receipt_sha256: str
    frozen_rank_bar_artifact_semantic_receipt_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    routing_identity_audit_receipt_sha256: str
    rank_identity_audit_receipt_sha256: str
    frozen_component_audit_inventory_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    components_runtime_qualified: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "decision_member_security_ids": self.decision_member_security_ids,
            "accounting_chain_security_ids": self.accounting_chain_security_ids,
            "all_supported_security_ids": self.all_supported_security_ids,
            "decision_origin_member_inventory_sha256": (
                self.decision_origin_member_inventory_sha256
            ),
            "normalized_identity_semantic_receipt_sha256": (
                self.normalized_identity_semantic_receipt_sha256
            ),
            "origin_plan_semantic_receipt_sha256": (
                self.origin_plan_semantic_receipt_sha256
            ),
            "monthly_rank_semantic_receipt_sha256": (
                self.monthly_rank_semantic_receipt_sha256
            ),
            "monthly_rank_bar_semantic_receipt_sha256": (
                self.monthly_rank_bar_semantic_receipt_sha256
            ),
            "frozen_origin_artifact_semantic_receipt_sha256": (
                self.frozen_origin_artifact_semantic_receipt_sha256
            ),
            "frozen_rank_artifact_semantic_receipt_sha256": (
                self.frozen_rank_artifact_semantic_receipt_sha256
            ),
            "frozen_rank_bar_artifact_semantic_receipt_sha256": (
                self.frozen_rank_bar_artifact_semantic_receipt_sha256
            ),
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "panel_materialization_authorized": self.panel_materialization_authorized,
            "predictive_training_authorized": self.predictive_training_authorized,
            "profitability_reporting_authorized": (
                self.profitability_reporting_authorized
            ),
            "lockbox_access_authorized": self.lockbox_access_authorized,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V2_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SOURCE_SHA256
        ):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 security-support identity differs"
            )
        decision = tuple(sorted(set(self.decision_member_security_ids)))
        accounting = tuple(sorted(set(self.accounting_chain_security_ids)))
        supported = tuple(sorted(set(self.all_supported_security_ids)))
        if (
            not decision
            or decision != self.decision_member_security_ids
            or accounting != self.accounting_chain_security_ids
            or supported != self.all_supported_security_ids
            or set(decision) & set(accounting)
            or set(supported) != set(decision) | set(accounting)
            or not isinstance(self.components_runtime_qualified, bool)
            or any(
                (
                    self.panel_materialization_authorized,
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 security-support inventory or authorization differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_EXPERIMENT_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_EXPERIMENT_V2_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 security-support source transaction differs"
            )
        for name in (
            "decision_origin_member_inventory_sha256",
            "normalized_identity_semantic_receipt_sha256",
            "origin_plan_semantic_receipt_sha256",
            "monthly_rank_semantic_receipt_sha256",
            "monthly_rank_bar_semantic_receipt_sha256",
            "frozen_origin_artifact_semantic_receipt_sha256",
            "frozen_rank_artifact_semantic_receipt_sha256",
            "frozen_rank_bar_artifact_semantic_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "routing_identity_audit_receipt_sha256",
            "rank_identity_audit_receipt_sha256",
            "frozen_component_audit_inventory_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 security-support semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "routing_identity_audit_receipt_sha256": (
                    self.routing_identity_audit_receipt_sha256
                ),
                "rank_identity_audit_receipt_sha256": (
                    self.rank_identity_audit_receipt_sha256
                ),
                "frozen_component_audit_inventory_sha256": (
                    self.frozen_component_audit_inventory_sha256
                ),
                "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
            }
        ):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 security-support audit receipt differs"
            )


def _component(
    artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    *,
    component_id: str,
    authority_receipt: str,
) -> None:
    artifact.validate()
    if (
        artifact.component_id != component_id
        or artifact.authority_semantic_receipt_sha256 != authority_receipt
    ):
        raise MassiveProfitabilityExperimentCoverageV2Error(
            f"frozen {component_id} artifact differs from its authority"
        )


def _publish(
    *,
    root: str | Path,
    semantic: Mapping[str, object],
    audit_fields: Mapping[str, object],
    artifact_kind: str,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> LoadedMassiveSourceObject:
    semantic_receipt = semantic_sha256(semantic)
    payload = {
        **semantic,
        "semantic_receipt_sha256": semantic_receipt,
        **audit_fields,
    }
    identifier = _artifact_id(artifact_id)
    relative = f"massive-profitability/experiment-v2/{artifact_kind}-{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_EXPERIMENT_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_EXPERIMENT_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "experiment entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"P0-EXPERIMENT-V2-{artifact_kind.upper()}-{identifier}",
    )
    return load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )


def materialize_massive_profitability_security_support_v2(
    *,
    root: str | Path,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    monthly_rank_authority: MassiveMonthlyRankInputAuthorityV2,
    monthly_rank_bar_authority: MassiveMonthlyRankBarAuthorityV1,
    routing_identity_authority: PITSecurityUniverseAuthority,
    rank_identity_authority: PITSecurityUniverseAuthority,
    frozen_origin_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    frozen_rank_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    frozen_rank_bar_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilitySecuritySupportV2:
    """Persist decision support after exact semantic identity reconciliation."""

    origin_plan.validate()
    monthly_rank_authority.validate()
    monthly_rank_bar_authority.validate()
    routing_identity_authority.validate()
    rank_identity_authority.validate()
    _component(
        frozen_origin_artifact,
        component_id="origin-plan-v2",
        authority_receipt=origin_plan.semantic_receipt_sha256,
    )
    _component(
        frozen_rank_artifact,
        component_id="monthly-rank-input-v2",
        authority_receipt=monthly_rank_authority.semantic_receipt_sha256,
    )
    _component(
        frozen_rank_bar_artifact,
        component_id="monthly-rank-bar-v1",
        authority_receipt=monthly_rank_bar_authority.semantic_receipt_sha256,
    )
    routing_semantics = massive_profitability_identity_semantic_receipt_v2(
        routing_identity_authority
    )
    if routing_semantics != massive_profitability_identity_semantic_receipt_v2(
        rank_identity_authority
    ):
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "routing and rank identity semantics differ"
        )
    if (
        origin_plan.origin_plan_v1.identity_authority_audit_receipt_sha256
        != rank_identity_authority.receipt_sha256
        or monthly_rank_authority.identity_authority_audit_receipt_sha256
        != rank_identity_authority.receipt_sha256
        or monthly_rank_bar_authority.identity_authority_receipt_sha256
        != routing_identity_authority.receipt_sha256
        or origin_plan.monthly_rank_authority_semantic_receipt_sha256
        != monthly_rank_authority.semantic_receipt_sha256
        or monthly_rank_bar_authority.rank_input_authority_semantic_receipt_sha256
        != monthly_rank_authority.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "V2 origin, rank, rank-bar, and identity bindings differ"
        )
    decision, accounting, supported = _support_ids(
        origin_plan=origin_plan, identity_authority=routing_identity_authority
    )
    member_inventory = semantic_sha256(
        tuple(
            (
                row.decision_session_date,
                row.decision_member_security_ids,
                row.decision_member_universe_ranks,
                row.membership_group_semantic_receipt_sha256,
            )
            for row in origin_plan.origin_plan_v1.origins
        )
    )
    frozen_audit_inventory = semantic_sha256(
        tuple(
            artifact.audit_receipt_sha256
            for artifact in (
                frozen_origin_artifact,
                frozen_rank_artifact,
                frozen_rank_bar_artifact,
            )
        )
    )
    semantic: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V2_SCHEMA,
        "decision_member_security_ids": decision,
        "accounting_chain_security_ids": accounting,
        "all_supported_security_ids": supported,
        "decision_origin_member_inventory_sha256": member_inventory,
        "normalized_identity_semantic_receipt_sha256": routing_semantics,
        "origin_plan_semantic_receipt_sha256": origin_plan.semantic_receipt_sha256,
        "monthly_rank_semantic_receipt_sha256": (
            monthly_rank_authority.semantic_receipt_sha256
        ),
        "monthly_rank_bar_semantic_receipt_sha256": (
            monthly_rank_bar_authority.semantic_receipt_sha256
        ),
        "frozen_origin_artifact_semantic_receipt_sha256": (
            frozen_origin_artifact.semantic_receipt_sha256
        ),
        "frozen_rank_artifact_semantic_receipt_sha256": (
            frozen_rank_artifact.semantic_receipt_sha256
        ),
        "frozen_rank_bar_artifact_semantic_receipt_sha256": (
            frozen_rank_bar_artifact.semantic_receipt_sha256
        ),
        "protocol_receipt_sha256": (MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256),
        "specification_sha256": (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SOURCE_SHA256
        ),
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    audit = {
        "routing_identity_audit_receipt_sha256": (
            routing_identity_authority.receipt_sha256
        ),
        "rank_identity_audit_receipt_sha256": rank_identity_authority.receipt_sha256,
        "frozen_component_audit_inventory_sha256": frozen_audit_inventory,
    }
    loaded = _publish(
        root=root,
        semantic=semantic,
        audit_fields=audit,
        artifact_kind="security-support",
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )
    parsed = parse_massive_profitability_security_support_v2(
        root=root, loaded_source=loaded
    )
    result = replace(
        parsed,
        components_runtime_qualified=all(
            artifact.runtime_qualified
            for artifact in (
                frozen_origin_artifact,
                frozen_rank_artifact,
                frozen_rank_bar_artifact,
            )
        )
        and monthly_rank_bar_authority.rank_bar_data_qualified,
    )
    result.validate()
    return result


def parse_massive_profitability_security_support_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilitySecuritySupportV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "V2 security-support source is not JSON"
        ) from exc
    semantic_fields = set(MassiveProfitabilitySecuritySupportV2.__annotations__) - {
        "semantic_receipt_sha256",
        "routing_identity_audit_receipt_sha256",
        "rank_identity_audit_receipt_sha256",
        "frozen_component_audit_inventory_sha256",
        "audit_receipt_sha256",
        "loaded_source",
        "components_runtime_qualified",
    }
    audit_fields = {
        "routing_identity_audit_receipt_sha256",
        "rank_identity_audit_receipt_sha256",
        "frozen_component_audit_inventory_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != semantic_fields | audit_fields | {"semantic_receipt_sha256"}
        or raw != canonical_json_file_bytes(payload)
    ):
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "V2 security-support canonical payload differs"
        )
    semantic = {key: payload[key] for key in semantic_fields}
    for key in (
        "decision_member_security_ids",
        "accounting_chain_security_ids",
        "all_supported_security_ids",
    ):
        semantic[key] = tuple(semantic[key])
    if semantic_sha256(semantic) != payload["semantic_receipt_sha256"]:
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "V2 security-support semantic payload differs"
        )
    result = MassiveProfitabilitySecuritySupportV2(
        **semantic,  # type: ignore[arg-type]
        semantic_receipt_sha256=payload["semantic_receipt_sha256"],
        routing_identity_audit_receipt_sha256=payload[
            "routing_identity_audit_receipt_sha256"
        ],
        rank_identity_audit_receipt_sha256=payload[
            "rank_identity_audit_receipt_sha256"
        ],
        frozen_component_audit_inventory_sha256=payload[
            "frozen_component_audit_inventory_sha256"
        ],
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": payload["semantic_receipt_sha256"],
                **{key: payload[key] for key in audit_fields},
                "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
            }
        ),
        loaded_source=loaded_source,
        components_runtime_qualified=False,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityExperimentCoverageV2:
    earliest_feature_base_session_date: str
    first_candidate_decision_session_date: str
    last_candidate_decision_session_date: str
    latest_h63_endpoint_session_date: str
    candidate_session_dates: tuple[str, ...]
    outer_test_session_inventories: tuple[tuple[str, ...], ...]
    lockbox_session_dates: tuple[str, ...]
    common_support_requirements: tuple[tuple[str, int, int], ...]
    candidate_inventory_sha256: str
    common_support_inventory_sha256: str
    archive_freeze_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    phase_origin_gate_receipt_sha256: str
    security_support_semantic_receipt_sha256: str
    frozen_origin_artifact_semantic_receipt_sha256: str
    frozen_rank_artifact_semantic_receipt_sha256: str
    frozen_rank_bar_artifact_semantic_receipt_sha256: str
    legacy_rejected_schemas: tuple[str, ...]
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    calendar_geometry_complete: bool
    origin_phase_complete: bool
    feature_input_complete: bool
    fill_source_complete: bool
    target_mark_complete: bool
    economic_action_complete: bool
    terminal_outcome_complete: bool
    data_gate_passed: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    semantic_receipt_sha256: str
    component_audit_inventory_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    source_transport_qualified: bool
    rank_bar_data_qualified: bool
    schema: str = MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        excluded = {
            "semantic_receipt_sha256",
            "component_audit_inventory_sha256",
            "audit_receipt_sha256",
            "loaded_source",
            "source_transport_qualified",
            "rank_bar_data_qualified",
        }
        return {
            key: value for key, value in asdict(self).items() if key not in excluded
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SOURCE_SHA256
        ):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 experiment-coverage identity differs"
            )
        candidates = tuple(sorted(set(self.candidate_session_dates)))
        outer = tuple(
            value for block in self.outer_test_session_inventories for value in block
        )
        if (
            not candidates
            or candidates != self.candidate_session_dates
            or self.first_candidate_decision_session_date != candidates[0]
            or self.last_candidate_decision_session_date != candidates[-1]
            or outer + self.lockbox_session_dates
            != candidates[-(len(outer) + len(self.lockbox_session_dates)) :]
            or self.candidate_inventory_sha256 != semantic_sha256(candidates)
            or self.common_support_inventory_sha256
            != semantic_sha256(self.common_support_requirements)
            or self.legacy_rejected_schemas
            != MASSIVE_PROFITABILITY_LEGACY_GENERATIONS_V2
            or self.calendar_geometry_complete is not True
            or self.data_gate_passed is not False
            or any(
                (
                    self.feature_input_complete,
                    self.fill_source_complete,
                    self.target_mark_complete,
                    self.economic_action_complete,
                    self.terminal_outcome_complete,
                    self.panel_materialization_authorized,
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
            or not isinstance(self.source_transport_qualified, bool)
            or not isinstance(self.rank_bar_data_qualified, bool)
            or self.rank_bar_data_qualified
            and not self.source_transport_qualified
        ):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 experiment geometry, completeness, or authorization differs"
            )
        expected_dates = tuple(row[0] for row in self.common_support_requirements)
        if expected_dates != tuple(sorted(set(expected_dates))):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 common-support requirements are not canonical"
            )
        for _, members, required in self.common_support_requirements:
            if members <= 0 or required != max(300, ceil(0.8 * members)):
                raise MassiveProfitabilityExperimentCoverageV2Error(
                    "V2 common-support threshold differs"
                )
        self.loaded_source.validate()
        for name in (
            "candidate_inventory_sha256",
            "common_support_inventory_sha256",
            "archive_freeze_semantic_receipt_sha256",
            "phase_plan_semantic_receipt_sha256",
            "phase_origin_gate_receipt_sha256",
            "security_support_semantic_receipt_sha256",
            "frozen_origin_artifact_semantic_receipt_sha256",
            "frozen_rank_artifact_semantic_receipt_sha256",
            "frozen_rank_bar_artifact_semantic_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "component_audit_inventory_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 experiment-coverage semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "component_audit_inventory_sha256": (
                    self.component_audit_inventory_sha256
                ),
                "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
            }
        ):
            raise MassiveProfitabilityExperimentCoverageV2Error(
                "V2 experiment-coverage audit receipt differs"
            )


def _coverage_semantic(
    *,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    phase_plan: MassiveProfitabilityPhasePlanV1,
    phase_origin_gate: MassiveProfitabilityPhaseOriginGateV1,
    support: MassiveProfitabilitySecuritySupportV2,
    frozen_origin_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    frozen_rank_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    frozen_rank_bar_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    common_support_requirements: Sequence[tuple[str, int, int]],
) -> dict[str, object]:
    candidates = archive_freeze.fixed_candidate_session_dates
    requirements = tuple(common_support_requirements)
    return {
        "schema": MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
        "earliest_feature_base_session_date": (
            archive_freeze.earliest_feature_base_session_date
        ),
        "first_candidate_decision_session_date": candidates[0],
        "last_candidate_decision_session_date": candidates[-1],
        "latest_h63_endpoint_session_date": (
            archive_freeze.latest_h63_endpoint_session_date
        ),
        "candidate_session_dates": candidates,
        "outer_test_session_inventories": (
            archive_freeze.fixed_outer_test_session_inventories
        ),
        "lockbox_session_dates": archive_freeze.fixed_lockbox_session_dates,
        "common_support_requirements": requirements,
        "candidate_inventory_sha256": archive_freeze.candidate_inventory_sha256,
        "common_support_inventory_sha256": semantic_sha256(requirements),
        "archive_freeze_semantic_receipt_sha256": (
            archive_freeze.semantic_receipt_sha256
        ),
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "phase_origin_gate_receipt_sha256": phase_origin_gate.receipt_sha256,
        "security_support_semantic_receipt_sha256": support.semantic_receipt_sha256,
        "frozen_origin_artifact_semantic_receipt_sha256": (
            frozen_origin_artifact.semantic_receipt_sha256
        ),
        "frozen_rank_artifact_semantic_receipt_sha256": (
            frozen_rank_artifact.semantic_receipt_sha256
        ),
        "frozen_rank_bar_artifact_semantic_receipt_sha256": (
            frozen_rank_bar_artifact.semantic_receipt_sha256
        ),
        "legacy_rejected_schemas": MASSIVE_PROFITABILITY_LEGACY_GENERATIONS_V2,
        "protocol_receipt_sha256": (MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256),
        "specification_sha256": (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SOURCE_SHA256
        ),
        "calendar_geometry_complete": True,
        "origin_phase_complete": (
            phase_origin_gate.outer_test_complete and phase_origin_gate.lockbox_complete
        ),
        "feature_input_complete": False,
        "fill_source_complete": False,
        "target_mark_complete": False,
        "economic_action_complete": False,
        "terminal_outcome_complete": False,
        "data_gate_passed": False,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }


def materialize_massive_profitability_experiment_coverage_v2(
    *,
    root: str | Path,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    phase_plan: MassiveProfitabilityPhasePlanV1,
    phase_origin_gate: MassiveProfitabilityPhaseOriginGateV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    support: MassiveProfitabilitySecuritySupportV2,
    frozen_origin_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    frozen_rank_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    frozen_rank_bar_artifact: MassiveProfitabilityFrozenAuthorityArtifactV1,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityExperimentCoverageV2:
    """Persist fixed coverage while leaving feature/fill/target gates false."""

    archive_freeze.validate()
    phase_plan.validate()
    phase_origin_gate.validate()
    origin_plan.validate()
    support.validate()
    if (
        phase_plan.candidate_session_dates
        != archive_freeze.fixed_candidate_session_dates
        or phase_origin_gate.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or phase_origin_gate.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or origin_plan.origin_plan_v1.candidate_decision_session_dates
        != archive_freeze.fixed_candidate_session_dates
        or support.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "V2 coverage components do not share one frozen candidate interval"
        )
    for artifact, component_id, receipt in (
        (frozen_origin_artifact, "origin-plan-v2", origin_plan.semantic_receipt_sha256),
        (
            frozen_rank_artifact,
            "monthly-rank-input-v2",
            support.monthly_rank_semantic_receipt_sha256,
        ),
        (
            frozen_rank_bar_artifact,
            "monthly-rank-bar-v1",
            support.monthly_rank_bar_semantic_receipt_sha256,
        ),
    ):
        _component(artifact, component_id=component_id, authority_receipt=receipt)
    requirements = tuple(
        (
            origin.decision_session_date,
            len(origin.decision_member_security_ids),
            max(300, ceil(0.8 * len(origin.decision_member_security_ids))),
        )
        for origin in origin_plan.origin_plan_v1.origins
    )
    semantic = _coverage_semantic(
        archive_freeze=archive_freeze,
        phase_plan=phase_plan,
        phase_origin_gate=phase_origin_gate,
        support=support,
        frozen_origin_artifact=frozen_origin_artifact,
        frozen_rank_artifact=frozen_rank_artifact,
        frozen_rank_bar_artifact=frozen_rank_bar_artifact,
        common_support_requirements=requirements,
    )
    component_audit = semantic_sha256(
        (
            archive_freeze.audit_receipt_sha256,
            phase_plan.audit_receipt_sha256,
            origin_plan.audit_receipt_sha256,
            support.audit_receipt_sha256,
            frozen_origin_artifact.audit_receipt_sha256,
            frozen_rank_artifact.audit_receipt_sha256,
            frozen_rank_bar_artifact.audit_receipt_sha256,
        )
    )
    loaded = _publish(
        root=root,
        semantic=semantic,
        audit_fields={"component_audit_inventory_sha256": component_audit},
        artifact_kind="coverage",
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )
    parsed = parse_massive_profitability_experiment_coverage_v2(
        root=root, loaded_source=loaded
    )
    result = replace(
        parsed,
        source_transport_qualified=archive_freeze.source_transport_qualified,
        rank_bar_data_qualified=archive_freeze.rank_bar_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_profitability_experiment_coverage_for_test_v2(
    *,
    root: str | Path,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    phase_plan: MassiveProfitabilityPhasePlanV1,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityExperimentCoverageV2:
    """Persist nonauthorizing coverage geometry without fabricated model data."""

    archive_freeze.validate()
    phase_plan.validate()
    requirements = tuple(
        (session_date, 500, 400)
        for session_date in archive_freeze.fixed_candidate_session_dates
    )
    dummy = semantic_sha256("test-v2-component")
    semantic = {
        "schema": MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA,
        "earliest_feature_base_session_date": (
            archive_freeze.earliest_feature_base_session_date
        ),
        "first_candidate_decision_session_date": (
            archive_freeze.fixed_candidate_session_dates[0]
        ),
        "last_candidate_decision_session_date": (
            archive_freeze.fixed_candidate_session_dates[-1]
        ),
        "latest_h63_endpoint_session_date": (
            archive_freeze.latest_h63_endpoint_session_date
        ),
        "candidate_session_dates": archive_freeze.fixed_candidate_session_dates,
        "outer_test_session_inventories": (
            archive_freeze.fixed_outer_test_session_inventories
        ),
        "lockbox_session_dates": archive_freeze.fixed_lockbox_session_dates,
        "common_support_requirements": requirements,
        "candidate_inventory_sha256": archive_freeze.candidate_inventory_sha256,
        "common_support_inventory_sha256": semantic_sha256(requirements),
        "archive_freeze_semantic_receipt_sha256": (
            archive_freeze.semantic_receipt_sha256
        ),
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "phase_origin_gate_receipt_sha256": dummy,
        "security_support_semantic_receipt_sha256": dummy,
        "frozen_origin_artifact_semantic_receipt_sha256": dummy,
        "frozen_rank_artifact_semantic_receipt_sha256": dummy,
        "frozen_rank_bar_artifact_semantic_receipt_sha256": dummy,
        "legacy_rejected_schemas": MASSIVE_PROFITABILITY_LEGACY_GENERATIONS_V2,
        "protocol_receipt_sha256": (MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256),
        "specification_sha256": (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SOURCE_SHA256
        ),
        "calendar_geometry_complete": True,
        "origin_phase_complete": False,
        "feature_input_complete": False,
        "fill_source_complete": False,
        "target_mark_complete": False,
        "economic_action_complete": False,
        "terminal_outcome_complete": False,
        "data_gate_passed": False,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    component_audit = semantic_sha256("test-v2-component-audit")
    loaded = _publish(
        root=root,
        semantic=semantic,
        audit_fields={"component_audit_inventory_sha256": component_audit},
        artifact_kind="coverage",
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )
    return parse_massive_profitability_experiment_coverage_v2(
        root=root, loaded_source=loaded
    )


def parse_massive_profitability_experiment_coverage_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityExperimentCoverageV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "V2 experiment-coverage source is not JSON"
        ) from exc
    excluded = {
        "semantic_receipt_sha256",
        "component_audit_inventory_sha256",
        "audit_receipt_sha256",
        "loaded_source",
        "source_transport_qualified",
        "rank_bar_data_qualified",
    }
    semantic_fields = (
        set(MassiveProfitabilityExperimentCoverageV2.__annotations__) - excluded
    )
    if (
        not isinstance(payload, dict)
        or set(payload)
        != semantic_fields
        | {"semantic_receipt_sha256", "component_audit_inventory_sha256"}
        or raw != canonical_json_file_bytes(payload)
    ):
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "V2 experiment-coverage canonical payload differs"
        )
    semantic = {key: payload[key] for key in semantic_fields}
    semantic["candidate_session_dates"] = tuple(semantic["candidate_session_dates"])
    semantic["outer_test_session_inventories"] = tuple(
        tuple(row) for row in semantic["outer_test_session_inventories"]
    )
    semantic["lockbox_session_dates"] = tuple(semantic["lockbox_session_dates"])
    semantic["common_support_requirements"] = tuple(
        tuple(row) for row in semantic["common_support_requirements"]
    )
    semantic["legacy_rejected_schemas"] = tuple(semantic["legacy_rejected_schemas"])
    if semantic_sha256(semantic) != payload["semantic_receipt_sha256"]:
        raise MassiveProfitabilityExperimentCoverageV2Error(
            "V2 experiment-coverage semantic payload differs"
        )
    result = MassiveProfitabilityExperimentCoverageV2(
        **semantic,  # type: ignore[arg-type]
        semantic_receipt_sha256=payload["semantic_receipt_sha256"],
        component_audit_inventory_sha256=payload["component_audit_inventory_sha256"],
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": payload["semantic_receipt_sha256"],
                "component_audit_inventory_sha256": payload[
                    "component_audit_inventory_sha256"
                ],
                "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
            }
        ),
        loaded_source=loaded_source,
        source_transport_qualified=False,
        rank_bar_data_qualified=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_LOCKBOX_ACCESS_AUTHORIZED",
    "MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PANEL_MATERIALIZATION_AUTHORIZED",
    "MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SCHEMA",
    "MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_SPEC_SHA256",
    "MASSIVE_PROFITABILITY_EXPERIMENT_V2_DATASET",
    "MASSIVE_PROFITABILITY_LEGACY_GENERATIONS_V2",
    "MASSIVE_PROFITABILITY_SECURITY_SUPPORT_V2_SCHEMA",
    "MassiveProfitabilityExperimentCoverageV2",
    "MassiveProfitabilityExperimentCoverageV2Error",
    "MassiveProfitabilitySecuritySupportV2",
    "massive_profitability_identity_semantic_receipt_v2",
    "materialize_massive_profitability_experiment_coverage_for_test_v2",
    "materialize_massive_profitability_experiment_coverage_v2",
    "materialize_massive_profitability_security_support_v2",
    "parse_massive_profitability_experiment_coverage_v2",
    "parse_massive_profitability_security_support_v2",
    "reject_massive_profitability_legacy_generation_v2",
]
