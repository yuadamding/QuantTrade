"""Create-only replay authority for checkpoint-derived RL cost ladders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
from pathlib import Path
from typing import TYPE_CHECKING

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
            "canonical-authorities-plus-static-environment-identities"
        ),
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


def _validation_environment_authority_type() -> type[
    MassiveAdaptiveRLValidationEnvironmentAuthorityV1
]:
    from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1,
    )

    return MassiveAdaptiveRLValidationEnvironmentAuthorityV1


def _validate_validation_environment_bindings(
    *,
    environments: tuple[
        MassiveAdaptiveProfitabilityEnvV1,
        MassiveAdaptiveProfitabilityEnvV1,
        MassiveAdaptiveProfitabilityEnvV1,
    ],
    environment_authorities: (
        Sequence[MassiveAdaptiveRLValidationEnvironmentAuthorityV1] | None
    ),
    fold_index: int,
    evaluation_role: str,
) -> tuple[MassiveAdaptiveRLValidationEnvironmentAuthorityV1, ...]:
    if environment_authorities is None:
        return ()
    authorities = tuple(environment_authorities)
    if not authorities:
        return ()
    if (
        evaluation_role != "inner_validation"
        or len(authorities) != 3
        or any(
            type(row) is not _validation_environment_authority_type()
            for row in authorities
        )
        or tuple(row.transaction_cost_basis_points for row in authorities)
        != (10.0, 20.0, 40.0)
    ):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost-ladder validation environment authorities differ"
        )
    for authority, environment in zip(authorities, environments, strict=True):
        authority.validate()
        authority.validate_environment(environment)
        if authority.fold_index != fold_index or not authority.source_data_qualified:
            raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                "adaptive RL cost-ladder validation environment authorities differ"
            )
    return authorities


def _payload(
    ladder: MassiveAdaptiveRLCostLadderV1,
    *,
    environments: tuple[
        MassiveAdaptiveProfitabilityEnvV1,
        MassiveAdaptiveProfitabilityEnvV1,
        MassiveAdaptiveProfitabilityEnvV1,
    ],
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
        environment_runtime = bool(self.runtime_validation_environment_authorities)
        for row in self.runtime_validation_environment_authorities:
            if type(row) is not _validation_environment_authority_type():
                raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
                    "adaptive RL cost-ladder validation environment type differs"
                )
            row.validate()
        expected = runtime and self.source_data_qualified
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
            or self.runtime_validation_environments_replayed != environment_runtime
            or (environment_runtime and not runtime)
            or (
                self.evaluation_role == "outer_test"
                and (
                    self.validation_environment_authority_receipts
                    or environment_runtime
                )
            )
            or (
                runtime
                and self.validation_environment_authority_receipts
                and not environment_runtime
            )
            or self.development_policy_selection_authorized
            != (expected and self.evaluation_role == "inner_validation")
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
            authorities = self.runtime_validation_environment_authorities
            if (
                len(authorities) != 3
                or tuple(row.transaction_cost_basis_points for row in authorities)
                != (10.0, 20.0, 40.0)
                or tuple(row.fold_index for row in authorities)
                != (self.fold_index,) * 3
                or tuple(row.semantic_receipt_sha256 for row in authorities)
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
    primary_environment: MassiveAdaptiveProfitabilityEnvV1,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    validation_environment_authorities: (
        Sequence[MassiveAdaptiveRLValidationEnvironmentAuthorityV1] | None
    ) = None,
) -> MassiveAdaptiveRLCostLadderAuthorityV1:
    parsed = parse_massive_adaptive_rl_cost_ladder_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed = _load_payload(root=root, loaded_source=authority.loaded_source)
    environments = (
        low_cost_environment,
        primary_environment,
        high_cost_environment,
    )
    environment_authorities = _validate_validation_environment_bindings(
        environments=environments,
        environment_authorities=validation_environment_authorities,
        fold_index=parsed.fold_index,
        evaluation_role=parsed.evaluation_role,
    )
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
        runtime_validation_environment_authorities=environment_authorities,
        runtime_validation_environments_replayed=bool(environment_authorities),
        development_policy_selection_authorized=(
            parsed.source_data_qualified
            and parsed.evaluation_role == "inner_validation"
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
    primary_environment: MassiveAdaptiveProfitabilityEnvV1,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    fold_index: int,
    evaluation_role: str,
    committed_at_ms: int,
    validation_environment_authorities: (
        Sequence[MassiveAdaptiveRLValidationEnvironmentAuthorityV1] | None
    ) = None,
) -> MassiveAdaptiveRLCostLadderAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLCostLadderAuthorityV1Error(
            "adaptive RL cost ladder artifact ID is not path safe"
        )
    environments = (
        low_cost_environment,
        primary_environment,
        high_cost_environment,
    )
    environment_authorities = _validate_validation_environment_bindings(
        environments=environments,
        environment_authorities=validation_environment_authorities,
        fold_index=fold_index,
        evaluation_role=evaluation_role,
    )
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
        primary_environment=primary_environment,
        low_cost_environment=low_cost_environment,
        high_cost_environment=high_cost_environment,
        validation_environment_authorities=environment_authorities,
    )


__all__ = [
    "MassiveAdaptiveRLCostLadderAuthorityV1",
    "MassiveAdaptiveRLCostLadderAuthorityV1Error",
    "authorize_massive_adaptive_rl_cost_ladder_authority_v1",
    "materialize_massive_adaptive_rl_cost_ladder_authority_v1",
    "parse_massive_adaptive_rl_cost_ladder_authority_v1",
]
