"""Create-only replay authority for checkpoint-derived RL cost ladders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_v1 import (
    MassiveAdaptiveRLCostLadderV1,
    evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)

if TYPE_CHECKING:
    from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1,
        MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    )

MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-cost-ladder-authority-v1"
)
MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-cost-ladder-authority-v1"
)
MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SCHEMA,
        "payload": "cost-ladder-identities-and-economics",
        "validation_context": "one-cost-excluded-environment-identity",
        "validation_environments": (
            "persisted-registry-plus-canonical-authorities-and-static-identities"
        ),
        "temporal_barrier": "registry-commit-strictly-precedes-outcome-commit",
        "dynamic_economic_sources": "transition-derived-separate-inventories",
        "promotion": "reload-checkpoint-rerun-primary-and-frozen-target-stresses",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLCostLadderAuthorityV1Error(ValueError):
    """The committed cost ladder did not replay from its checkpoint."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _validation_environment_types() -> tuple[type[object], type[object]]:
    from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1,
        MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    )

    return (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1,
        MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    )


def _resolve_cost_ladder_environments(
    *,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1 | None,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1 | None,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1 | None,
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ),
    fold_index: int,
    evaluation_role: str,
    outcome_committed_at_ms: int,
) -> tuple[
    tuple[
        MassiveAdaptiveProfitabilityEnvV1,
        MassiveAdaptiveProfitabilityEnvV1,
        MassiveAdaptiveProfitabilityEnvV1,
    ],
    tuple[MassiveAdaptiveRLValidationEnvironmentAuthorityV1, ...],
]:
    environment_type, registry_type = _validation_environment_types()
    supplied = (low_cost_environment, primary_environment, high_cost_environment)
    if evaluation_role == "outer_test":
        if validation_environment_registry is not None or any(
            type(row) is not MassiveAdaptiveProfitabilityEnvV1 for row in supplied
        ):
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "outer adaptive RL cost ladder cannot use a validation registry"
            )
        return supplied, ()  # type: ignore[return-value]
    if evaluation_role != "inner_validation":
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost-ladder evaluation role differs"
        )
    if any(row is not None for row in supplied) or type(
        validation_environment_registry
    ) is not registry_type:
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "inner-validation cost ladder requires its persisted validation registry"
        )
    registry = cast(
        "MassiveAdaptiveRLValidationEnvironmentRegistryV1",
        validation_environment_registry,
    )
    registry.validate()
    registry_committed_at_ms = registry.source_transaction_committed_at_ms
    if (
        not registry.source_transaction_verified
        or not registry.runtime_environments_replayed
        or not registry.development_stage_authorized
        or registry.source_receipt_sha256 is None
        or registry.source_transaction_receipt_sha256 is None
        or registry_committed_at_ms is None
        or registry_committed_at_ms >= outcome_committed_at_ms
        or registry.fold_index != fold_index
    ):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "inner-validation cost-ladder registry is not precommitted and authorized"
        )
    built = registry.build_environments()
    environments = (built[10.0], built[20.0], built[40.0])
    authorities = tuple(
        registry.environment_authority(cost) for cost in (10.0, 20.0, 40.0)
    )
    for authority, environment in zip(authorities, environments, strict=True):
        if type(authority) is not environment_type:
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "adaptive RL cost-ladder validation environment authorities differ"
            )
        authority.validate()
        authority.validate_environment(environment)
        if authority.fold_index != fold_index or not authority.source_data_qualified:
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "adaptive RL cost-ladder validation environment authorities differ"
            )
    return environments, authorities


def _payload(
    ladder: MassiveAdaptiveRLCostLadderV1,
    *,
    environments: tuple[
        MassiveAdaptiveProfitabilityEnvV1,
        MassiveAdaptiveProfitabilityEnvV1,
        MassiveAdaptiveProfitabilityEnvV1,
    ],
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ),
    validation_environment_authorities: tuple[
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1, ...
    ],
) -> dict[str, object]:
    ladder.validate()
    if tuple(row.transaction_cost_basis_points for row in environments) != (
        10.0,
        20.0,
        40.0,
    ):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost-ladder environment order differs"
        )
    registry = validation_environment_registry
    if (registry is None) != (not validation_environment_authorities):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost-ladder validation registry binding is partial"
        )
    if registry is not None and (
        registry.source_receipt_sha256 is None
        or registry.source_transaction_receipt_sha256 is None
        or registry.source_transaction_committed_at_ms is None
    ):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost-ladder registry source transaction is absent"
        )
    return {
        "fold_index": ladder.fold_index,
        "evaluation_role": ladder.evaluation_role,
        "checkpoint_authority_receipt_sha256": (
            ladder.checkpoint_authority_receipt_sha256
        ),
        "checkpoint_receipt_sha256": ladder.checkpoint_receipt_sha256,
        "validation_context_receipt_sha256": _digest(
            "validation context receipt",
            environments[1].validation_context_receipt_sha256,
        ),
        "validation_sources_authority_receipt_sha256": (
            None
            if registry is None
            else registry.validation_sources_authority_receipt_sha256
        ),
        "validation_environment_registry_receipt_sha256": (
            None if registry is None else registry.semantic_receipt_sha256
        ),
        "validation_environment_registry_source_receipt_sha256": (
            None if registry is None else registry.source_receipt_sha256
        ),
        "validation_environment_registry_commit_receipt_sha256": (
            None if registry is None else registry.source_transaction_receipt_sha256
        ),
        "validation_environment_registry_committed_at_ms": (
            None if registry is None else registry.source_transaction_committed_at_ms
        ),
        "validation_environment_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in validation_environment_authorities
        ),
        "environment_source_inventory_sha256s": tuple(
            _digest(
                "cost-ladder environment source inventory",
                row.source_inventory_sha256,
            )
            for row in environments
        ),
        "economic_compatibility_receipt_sha256s": tuple(
            _digest(
                "cost-ladder economic compatibility receipt",
                row.economic_compatibility_receipt_sha256,
            )
            for row in environments
        ),
        "primary_receipt_sha256": ladder.primary.semantic_receipt_sha256,
        "primary_trace": asdict(ladder.primary.policy_trace),
        "primary_action_evidence_inventory_sha256": (
            ladder.primary.action_evidence_inventory_sha256
        ),
        "primary_transition_inventory_sha256": (
            ladder.primary.transition_inventory_sha256
        ),
        "low_cost_trace": asdict(ladder.low_cost_trace),
        "high_cost_trace": asdict(ladder.high_cost_trace),
        "low_cost_transition_inventory_sha256": (
            ladder.low_cost_transition_inventory_sha256
        ),
        "high_cost_transition_inventory_sha256": (
            ladder.high_cost_transition_inventory_sha256
        ),
        "decision_target_inventory_sha256": (ladder.decision_target_inventory_sha256),
        "source_data_qualified": ladder.source_data_qualified,
        "cost_ladder_receipt_sha256": ladder.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost ladder payload is not canonical JSON"
        )
    return dict(value)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLCostLadderAuthorityV1:
    fold_index: int
    evaluation_role: str
    checkpoint_authority_receipt_sha256: str
    checkpoint_receipt_sha256: str
    validation_context_receipt_sha256: str
    validation_sources_authority_receipt_sha256: str | None
    validation_environment_registry_receipt_sha256: str | None
    validation_environment_registry_source_receipt_sha256: str | None
    validation_environment_registry_commit_receipt_sha256: str | None
    validation_environment_registry_committed_at_ms: int | None
    validation_environment_authority_receipts: tuple[str, ...]
    environment_source_inventory_sha256s: tuple[str, ...]
    economic_compatibility_receipt_sha256s: tuple[str, ...]
    cost_ladder_receipt_sha256: str
    primary_trace_receipt_sha256: str
    low_cost_trace_receipt_sha256: str
    high_cost_trace_receipt_sha256: str
    decision_target_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_ladder: MassiveAdaptiveRLCostLadderV1 | None
    runtime_ladder_replayed: bool
    runtime_validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    )
    runtime_validation_environment_registry_replayed: bool
    runtime_validation_environment_authorities: tuple[
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1, ...
    ]
    runtime_validation_environments_replayed: bool
    development_policy_selection_authorized: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "evaluation_role": self.evaluation_role,
            "checkpoint_authority_receipt_sha256": (
                self.checkpoint_authority_receipt_sha256
            ),
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "validation_context_receipt_sha256": (
                self.validation_context_receipt_sha256
            ),
            "validation_sources_authority_receipt_sha256": (
                self.validation_sources_authority_receipt_sha256
            ),
            "validation_environment_registry_receipt_sha256": (
                self.validation_environment_registry_receipt_sha256
            ),
            "validation_environment_registry_source_receipt_sha256": (
                self.validation_environment_registry_source_receipt_sha256
            ),
            "validation_environment_registry_commit_receipt_sha256": (
                self.validation_environment_registry_commit_receipt_sha256
            ),
            "validation_environment_registry_committed_at_ms": (
                self.validation_environment_registry_committed_at_ms
            ),
            "validation_environment_authority_receipts": (
                self.validation_environment_authority_receipts
            ),
            "environment_source_inventory_sha256s": (
                self.environment_source_inventory_sha256s
            ),
            "economic_compatibility_receipt_sha256s": (
                self.economic_compatibility_receipt_sha256s
            ),
            "cost_ladder_receipt_sha256": self.cost_ladder_receipt_sha256,
            "primary_trace_receipt_sha256": self.primary_trace_receipt_sha256,
            "low_cost_trace_receipt_sha256": self.low_cost_trace_receipt_sha256,
            "high_cost_trace_receipt_sha256": self.high_cost_trace_receipt_sha256,
            "decision_target_inventory_sha256": (self.decision_target_inventory_sha256),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_ladder is not None
        registry_runtime = self.runtime_validation_environment_registry is not None
        environment_runtime = bool(self.runtime_validation_environment_authorities)
        environment_type, registry_type = _validation_environment_types()
        if self.runtime_validation_environment_registry is not None:
            if type(self.runtime_validation_environment_registry) is not registry_type:
                raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                    "adaptive RL cost-ladder validation registry type differs"
                )
            self.runtime_validation_environment_registry.validate()
        for row in self.runtime_validation_environment_authorities:
            if type(row) is not environment_type:
                raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                    "adaptive RL cost-ladder validation environment type differs"
                )
            row.validate()
        expected = runtime and self.source_data_qualified
        registry_fields = (
            self.validation_sources_authority_receipt_sha256,
            self.validation_environment_registry_receipt_sha256,
            self.validation_environment_registry_source_receipt_sha256,
            self.validation_environment_registry_commit_receipt_sha256,
            self.validation_environment_registry_committed_at_ms,
        )
        registry_fields_present = all(value is not None for value in registry_fields)
        if any(value is not None for value in registry_fields) != registry_fields_present:
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "adaptive RL cost-ladder validation registry binding is partial"
            )
        if self.runtime_ladder is not None:
            self.runtime_ladder.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SCHEMA
            or self.evaluation_role not in {"inner_validation", "outer_test"}
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.cost_ladder_receipt_sha256
            or self.runtime_ladder_replayed != runtime
            or self.runtime_validation_environment_registry_replayed
            != registry_runtime
            or self.runtime_validation_environments_replayed != environment_runtime
            or registry_runtime != environment_runtime
            or (registry_runtime and not runtime)
            or (
                self.evaluation_role == "outer_test"
                and (
                    registry_fields_present
                    or registry_runtime
                    or self.validation_environment_authority_receipts
                    or environment_runtime
                )
            )
            or (self.evaluation_role == "inner_validation" and not registry_fields_present)
            or self.development_policy_selection_authorized
            != (
                expected
                and self.evaluation_role == "inner_validation"
                and registry_runtime
            )
            or self.outer_evaluation_authorized
            != (expected and self.evaluation_role == "outer_test")
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "adaptive RL cost ladder authority differs"
            )
        if (
            runtime
            and self.runtime_ladder is not None
            and (
                self.runtime_ladder.fold_index != self.fold_index
                or self.runtime_ladder.evaluation_role != self.evaluation_role
                or self.runtime_ladder.checkpoint_authority_receipt_sha256
                != self.checkpoint_authority_receipt_sha256
                or self.runtime_ladder.checkpoint_receipt_sha256
                != self.checkpoint_receipt_sha256
                or self.runtime_ladder.semantic_receipt_sha256
                != self.cost_ladder_receipt_sha256
                or self.runtime_ladder.primary.policy_trace.semantic_receipt_sha256
                != self.primary_trace_receipt_sha256
                or self.runtime_ladder.low_cost_trace.semantic_receipt_sha256
                != self.low_cost_trace_receipt_sha256
                or self.runtime_ladder.high_cost_trace.semantic_receipt_sha256
                != self.high_cost_trace_receipt_sha256
                or self.runtime_ladder.decision_target_inventory_sha256
                != self.decision_target_inventory_sha256
            )
        ):
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "adaptive runtime cost ladder differs from its authority"
            )
        if environment_runtime:
            assert self.runtime_validation_environment_registry is not None
            registry = self.runtime_validation_environment_registry
            authorities = self.runtime_validation_environment_authorities
            if (
                len(authorities) != 3
                or not registry.development_stage_authorized
                or registry.fold_index != self.fold_index
                or registry.validation_sources_authority_receipt_sha256
                != self.validation_sources_authority_receipt_sha256
                or registry.semantic_receipt_sha256
                != self.validation_environment_registry_receipt_sha256
                or registry.source_receipt_sha256
                != self.validation_environment_registry_source_receipt_sha256
                or registry.source_transaction_receipt_sha256
                != self.validation_environment_registry_commit_receipt_sha256
                or registry.source_transaction_committed_at_ms
                != self.validation_environment_registry_committed_at_ms
                or registry.source_transaction_committed_at_ms is None
                or registry.source_transaction_committed_at_ms
                >= self.loaded_source.commit.committed_at_ms
                or tuple(row.transaction_cost_basis_points for row in authorities)
                != (10.0, 20.0, 40.0)
                or tuple(row.fold_index for row in authorities)
                != (self.fold_index,) * 3
                or tuple(row.semantic_receipt_sha256 for row in authorities)
                != self.validation_environment_authority_receipts
                or tuple(
                    registry.environment_authority(cost).semantic_receipt_sha256
                    for cost in (10.0, 20.0, 40.0)
                )
                != self.validation_environment_authority_receipts
                or tuple(row.environment_source_inventory_sha256 for row in authorities)
                != self.environment_source_inventory_sha256s
                or tuple(
                    row.economic_compatibility_receipt_sha256 for row in authorities
                )
                != self.economic_compatibility_receipt_sha256s
                or {row.validation_context_receipt_sha256 for row in authorities}
                != {self.validation_context_receipt_sha256}
            ):
                raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                    "adaptive runtime cost-ladder environments differ"
                )
        if (
            len(self.environment_source_inventory_sha256s) != 3
            or len(self.economic_compatibility_receipt_sha256s) != 3
            or len(self.validation_environment_authority_receipts) not in {0, 3}
        ):
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "adaptive RL cost-ladder environment inventory differs"
            )
        for value in (
            *self.validation_environment_authority_receipts,
            *self.environment_source_inventory_sha256s,
            *self.economic_compatibility_receipt_sha256s,
        ):
            _digest("adaptive RL cost-ladder environment authority", value)
        for registry_value in (
            self.validation_sources_authority_receipt_sha256,
            self.validation_environment_registry_receipt_sha256,
            self.validation_environment_registry_source_receipt_sha256,
            self.validation_environment_registry_commit_receipt_sha256,
        ):
            if registry_value is not None:
                _digest("adaptive RL cost-ladder validation registry", registry_value)
        if self.validation_environment_registry_committed_at_ms is not None and (
            isinstance(self.validation_environment_registry_committed_at_ms, bool)
            or self.validation_environment_registry_committed_at_ms < 0
            or self.validation_environment_registry_committed_at_ms
            >= self.loaded_source.commit.committed_at_ms
        ):
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "adaptive RL cost-ladder registry was not committed first"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())
        _digest(
            "adaptive RL cost-ladder validation context",
            self.validation_context_receipt_sha256,
        )


def parse_massive_adaptive_rl_cost_ladder_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLCostLadderAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    primary = dict(payload["primary_trace"])  # type: ignore[call-overload]
    low = dict(payload["low_cost_trace"])  # type: ignore[call-overload]
    high = dict(payload["high_cost_trace"])  # type: ignore[call-overload]
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SCHEMA,
        "fold_index": int(str(payload["fold_index"])),
        "evaluation_role": str(payload["evaluation_role"]),
        "checkpoint_authority_receipt_sha256": str(
            payload["checkpoint_authority_receipt_sha256"]
        ),
        "checkpoint_receipt_sha256": str(payload["checkpoint_receipt_sha256"]),
        "validation_context_receipt_sha256": str(
            payload["validation_context_receipt_sha256"]
        ),
        "validation_sources_authority_receipt_sha256": (
            None
            if payload["validation_sources_authority_receipt_sha256"] is None
            else str(payload["validation_sources_authority_receipt_sha256"])
        ),
        "validation_environment_registry_receipt_sha256": (
            None
            if payload["validation_environment_registry_receipt_sha256"] is None
            else str(payload["validation_environment_registry_receipt_sha256"])
        ),
        "validation_environment_registry_source_receipt_sha256": (
            None
            if payload["validation_environment_registry_source_receipt_sha256"]
            is None
            else str(
                payload["validation_environment_registry_source_receipt_sha256"]
            )
        ),
        "validation_environment_registry_commit_receipt_sha256": (
            None
            if payload["validation_environment_registry_commit_receipt_sha256"] is None
            else str(
                payload["validation_environment_registry_commit_receipt_sha256"]
            )
        ),
        "validation_environment_registry_committed_at_ms": (
            None
            if payload["validation_environment_registry_committed_at_ms"] is None
            else int(str(payload["validation_environment_registry_committed_at_ms"]))
        ),
        "validation_environment_authority_receipts": tuple(
            str(value)
            for value in payload["validation_environment_authority_receipts"]  # type: ignore[attr-defined]
        ),
        "environment_source_inventory_sha256s": tuple(
            str(value)
            for value in payload["environment_source_inventory_sha256s"]  # type: ignore[attr-defined]
        ),
        "economic_compatibility_receipt_sha256s": tuple(
            str(value)
            for value in payload["economic_compatibility_receipt_sha256s"]  # type: ignore[attr-defined]
        ),
        "cost_ladder_receipt_sha256": str(payload["cost_ladder_receipt_sha256"]),
        "primary_trace_receipt_sha256": str(primary["semantic_receipt_sha256"]),
        "low_cost_trace_receipt_sha256": str(low["semantic_receipt_sha256"]),
        "high_cost_trace_receipt_sha256": str(high["semantic_receipt_sha256"]),
        "decision_target_inventory_sha256": str(
            payload["decision_target_inventory_sha256"]
        ),
        "source_data_qualified": bool(payload["source_data_qualified"]),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLCostLadderAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_ladder=None,
        runtime_ladder_replayed=False,
        runtime_validation_environment_registry=None,
        runtime_validation_environment_registry_replayed=False,
        runtime_validation_environment_authorities=(),
        runtime_validation_environments_replayed=False,
        development_policy_selection_authorized=False,
        outer_evaluation_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_cost_ladder_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLCostLadderAuthorityV1,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1 | None = None,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1 | None = None,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1 | None = None,
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ) = None,
) -> MassiveAdaptiveRLCostLadderAuthorityV1:
    parsed = parse_massive_adaptive_rl_cost_ladder_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed = _load_payload(root=root, loaded_source=authority.loaded_source)
    environments, environment_authorities = _resolve_cost_ladder_environments(
        primary_environment=primary_environment,
        low_cost_environment=low_cost_environment,
        high_cost_environment=high_cost_environment,
        validation_environment_registry=validation_environment_registry,
        fold_index=parsed.fold_index,
        evaluation_role=parsed.evaluation_role,
        outcome_committed_at_ms=authority.loaded_source.commit.committed_at_ms,
    )
    low_cost_environment, primary_environment, high_cost_environment = environments
    replayed = evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1(
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        primary_environment=primary_environment,
        low_cost_environment=low_cost_environment,
        high_cost_environment=high_cost_environment,
        fold_index=parsed.fold_index,
        evaluation_role=parsed.evaluation_role,
    )
    if canonical_json_file_bytes(committed) != canonical_json_file_bytes(
        _payload(
            replayed,
            environments=environments,
            validation_environment_registry=validation_environment_registry,
            validation_environment_authorities=environment_authorities,
        )
    ):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost ladder does not replay from its checkpoint"
        )
    result = replace(
        parsed,
        runtime_ladder=replayed,
        runtime_ladder_replayed=True,
        runtime_validation_environment_registry=validation_environment_registry,
        runtime_validation_environment_registry_replayed=(
            validation_environment_registry is not None
        ),
        runtime_validation_environment_authorities=environment_authorities,
        runtime_validation_environments_replayed=bool(environment_authorities),
        development_policy_selection_authorized=(
            parsed.source_data_qualified
            and parsed.evaluation_role == "inner_validation"
            and validation_environment_registry is not None
        ),
        outer_evaluation_authorized=(
            parsed.source_data_qualified and parsed.evaluation_role == "outer_test"
        ),
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_cost_ladder_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1 | None = None,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1 | None = None,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1 | None = None,
    fold_index: int,
    evaluation_role: str,
    committed_at_ms: int,
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ) = None,
) -> MassiveAdaptiveRLCostLadderAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost ladder artifact ID is not path safe"
        )
    environments, environment_authorities = _resolve_cost_ladder_environments(
        primary_environment=primary_environment,
        low_cost_environment=low_cost_environment,
        high_cost_environment=high_cost_environment,
        validation_environment_registry=validation_environment_registry,
        fold_index=fold_index,
        evaluation_role=evaluation_role,
        outcome_committed_at_ms=committed_at_ms,
    )
    low_cost_environment, primary_environment, high_cost_environment = environments
    ladder = evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1(
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        primary_environment=primary_environment,
        low_cost_environment=low_cost_environment,
        high_cost_environment=high_cost_environment,
        fold_index=fold_index,
        evaluation_role=evaluation_role,
    )
    relative = f"massive-adaptive/rl-cost-ladder-authority-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(
            canonical_json_file_bytes(
                _payload(
                    ladder,
                    environments=environments,
                    validation_environment_registry=validation_environment_registry,
                    validation_environment_authorities=environment_authorities,
                )
            )
        ),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=ladder.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-COST-LADDER-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_cost_ladder_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_cost_ladder_authority_v1(
            root=root, loaded_source=loaded
        ),
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        primary_environment=(
            primary_environment if evaluation_role == "outer_test" else None
        ),
        low_cost_environment=(
            low_cost_environment if evaluation_role == "outer_test" else None
        ),
        high_cost_environment=(
            high_cost_environment if evaluation_role == "outer_test" else None
        ),
        validation_environment_registry=validation_environment_registry,
    )


__all__ = [
    "MassiveAdaptiveRLCostLadderAuthorityV1",
    "MassiveAdaptiveRLCostLadderAuthorityV1Error",
    "authorize_massive_adaptive_rl_cost_ladder_authority_v1",
    "materialize_massive_adaptive_rl_cost_ladder_authority_v1",
    "parse_massive_adaptive_rl_cost_ladder_authority_v1",
]
