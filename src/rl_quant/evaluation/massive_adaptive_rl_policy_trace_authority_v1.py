"""Create-only replay authority for checkpoint-generated adaptive RL traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

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
from rl_quant.evaluation.massive_adaptive_rl_policy_evaluator_v1 import (
    MassiveAdaptiveRLCheckpointPolicyTraceV1,
    MassiveAdaptiveRLPolicyActionEvidenceV1,
    evaluate_massive_adaptive_rl_checkpoint_v1,
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
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
)

if TYPE_CHECKING:
    from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v1 import (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    )
    from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1,
        MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    )

MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-trace-authority-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-policy-trace-authority-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA,
        "payload": "canonical-policy-trace-and-action-evidence",
        "validation_context": "cost-excluded-environment-identity",
        "validation_environment": (
            "persisted-registry-plus-canonical-authority-and-static-identities"
        ),
        "temporal_barrier": (
            "all-four-validation-inputs-commit-strictly-precedes-outcome-commit"
        ),
        "dynamic_economic_sources": "transition-derived-separate-inventory",
        "promotion": "reload-checkpoint-rerun-actions-and-economics",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLPolicyTraceAuthorityV1Error(ValueError):
    """The committed policy trace did not replay from its attached checkpoint."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
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


def _resolve_evaluation_environment(
    *,
    environment: MassiveAdaptiveProfitabilityEnvV1 | None,
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ),
    four_fold_validation_inputs_authority: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1 | None
    ),
    fold_index: int,
    evaluation_role: str,
    outcome_committed_at_ms: int,
) -> tuple[
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None,
]:
    if evaluation_role == "outer_test":
        if (
            validation_environment_registry is not None
            or four_fold_validation_inputs_authority is not None
            or type(environment) is not MassiveAdaptiveProfitabilityEnvV1
        ):
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "outer adaptive RL policy trace cannot use a validation registry"
            )
        return environment, None
    if evaluation_role != "inner_validation":
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace evaluation role differs"
        )
    environment_type, registry_type = _validation_environment_types()
    if (
        environment is not None
        or type(validation_environment_registry) is not registry_type
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "inner-validation policy trace requires its persisted validation registry"
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
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "inner-validation policy trace registry is not precommitted and authorized"
        )
    from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v1 import (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
        validate_massive_adaptive_rl_validation_outcome_barrier_v1,
    )

    if (
        type(four_fold_validation_inputs_authority)
        is not MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "inner-validation policy trace requires the four-fold input authority"
        )
    validate_massive_adaptive_rl_validation_outcome_barrier_v1(
        authority=four_fold_validation_inputs_authority,
        validation_environment_registry=registry,
        fold_index=fold_index,
        outcome_committed_at_ms=outcome_committed_at_ms,
    )
    environments = registry.build_environments()
    resolved_environment = environments[20.0]
    environment_authority = registry.environment_authority(20.0)
    if type(environment_authority) is not environment_type:
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace validation environment authority differs"
        )
    environment_authority.validate_environment(resolved_environment)
    return resolved_environment, environment_authority


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicyTraceAuthorityV1:
    fold_index: int
    evaluation_role: str
    checkpoint_authority_receipt_sha256: str
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    validation_context_receipt_sha256: str
    validation_sources_authority_receipt_sha256: str | None
    validation_environment_registry_receipt_sha256: str | None
    validation_environment_registry_source_receipt_sha256: str | None
    validation_environment_registry_commit_receipt_sha256: str | None
    validation_environment_registry_committed_at_ms: int | None
    four_fold_validation_inputs_authority_receipt_sha256: str | None
    four_fold_validation_inputs_source_receipt_sha256: str | None
    four_fold_validation_inputs_commit_receipt_sha256: str | None
    four_fold_validation_inputs_committed_at_ms: int | None
    validation_environment_authority_receipt_sha256: str | None
    environment_source_inventory_sha256: str
    economic_compatibility_receipt_sha256: str
    policy_trace_receipt_sha256: str
    action_evidence_inventory_sha256: str
    transition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_trace: MassiveAdaptiveRLCheckpointPolicyTraceV1 | None
    runtime_trace_replayed: bool
    runtime_validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    )
    runtime_validation_environment_registry_replayed: bool
    runtime_four_fold_validation_inputs_authority: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1 | None
    )
    runtime_four_fold_validation_inputs_replayed: bool
    runtime_validation_environment_authority: (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None
    )
    runtime_validation_environment_replayed: bool
    development_policy_evaluation_authorized: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "evaluation_role": self.evaluation_role,
            "checkpoint_authority_receipt_sha256": (
                self.checkpoint_authority_receipt_sha256
            ),
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
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
            "four_fold_validation_inputs_authority_receipt_sha256": (
                self.four_fold_validation_inputs_authority_receipt_sha256
            ),
            "four_fold_validation_inputs_source_receipt_sha256": (
                self.four_fold_validation_inputs_source_receipt_sha256
            ),
            "four_fold_validation_inputs_commit_receipt_sha256": (
                self.four_fold_validation_inputs_commit_receipt_sha256
            ),
            "four_fold_validation_inputs_committed_at_ms": (
                self.four_fold_validation_inputs_committed_at_ms
            ),
            "validation_environment_authority_receipt_sha256": (
                self.validation_environment_authority_receipt_sha256
            ),
            "environment_source_inventory_sha256": (
                self.environment_source_inventory_sha256
            ),
            "economic_compatibility_receipt_sha256": (
                self.economic_compatibility_receipt_sha256
            ),
            "policy_trace_receipt_sha256": self.policy_trace_receipt_sha256,
            "action_evidence_inventory_sha256": self.action_evidence_inventory_sha256,
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_trace is not None
        registry_runtime = self.runtime_validation_environment_registry is not None
        barrier_runtime = self.runtime_four_fold_validation_inputs_authority is not None
        validation_environment_runtime = (
            self.runtime_validation_environment_authority is not None
        )
        environment_type, registry_type = _validation_environment_types()
        if self.runtime_validation_environment_registry is not None:
            if type(self.runtime_validation_environment_registry) is not registry_type:
                raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                    "adaptive RL policy trace validation registry type differs"
                )
            self.runtime_validation_environment_registry.validate()
        if self.runtime_validation_environment_authority is not None:
            if (
                type(self.runtime_validation_environment_authority)
                is not environment_type
            ):
                raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                    "adaptive RL policy trace validation environment type differs"
                )
            self.runtime_validation_environment_authority.validate()
        expected = runtime and self.source_data_qualified
        registry_fields = (
            self.validation_sources_authority_receipt_sha256,
            self.validation_environment_registry_receipt_sha256,
            self.validation_environment_registry_source_receipt_sha256,
            self.validation_environment_registry_commit_receipt_sha256,
            self.validation_environment_registry_committed_at_ms,
            self.validation_environment_authority_receipt_sha256,
        )
        registry_fields_present = all(value is not None for value in registry_fields)
        if (
            any(value is not None for value in registry_fields)
            != registry_fields_present
        ):
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "adaptive RL policy trace validation registry binding is partial"
            )
        barrier_fields = (
            self.four_fold_validation_inputs_authority_receipt_sha256,
            self.four_fold_validation_inputs_source_receipt_sha256,
            self.four_fold_validation_inputs_commit_receipt_sha256,
            self.four_fold_validation_inputs_committed_at_ms,
        )
        barrier_fields_present = all(value is not None for value in barrier_fields)
        if any(value is not None for value in barrier_fields) != barrier_fields_present:
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "adaptive RL policy trace four-fold input binding is partial"
            )
        if self.runtime_trace is not None:
            self.runtime_trace.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA
            or self.evaluation_role not in {"inner_validation", "outer_test"}
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.policy_trace_receipt_sha256
            or self.runtime_trace_replayed != runtime
            or self.runtime_validation_environment_registry_replayed != registry_runtime
            or self.runtime_four_fold_validation_inputs_replayed != barrier_runtime
            or self.runtime_validation_environment_replayed
            != validation_environment_runtime
            or registry_runtime != validation_environment_runtime
            or barrier_runtime != registry_runtime
            or (registry_runtime and not runtime)
            or (
                self.evaluation_role == "outer_test"
                and (
                    registry_fields_present
                    or barrier_fields_present
                    or registry_runtime
                    or barrier_runtime
                    or validation_environment_runtime
                )
            )
            or (
                self.evaluation_role == "inner_validation"
                and (not registry_fields_present or not barrier_fields_present)
            )
            or self.development_policy_evaluation_authorized
            != (
                expected
                and self.evaluation_role == "inner_validation"
                and registry_runtime
                and barrier_runtime
            )
            or self.outer_evaluation_authorized
            != (expected and self.evaluation_role == "outer_test")
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "adaptive RL policy trace authority differs"
            )
        if (
            runtime
            and self.runtime_trace is not None
            and (
                self.runtime_trace.fold_index != self.fold_index
                or self.runtime_trace.evaluation_role != self.evaluation_role
                or self.runtime_trace.checkpoint_authority_receipt_sha256
                != self.checkpoint_authority_receipt_sha256
                or self.runtime_trace.checkpoint_receipt_sha256
                != self.checkpoint_receipt_sha256
                or self.runtime_trace.model_state_receipt_sha256
                != self.model_state_receipt_sha256
                or self.runtime_trace.policy_trace.semantic_receipt_sha256
                != self.policy_trace_receipt_sha256
                or self.runtime_trace.action_evidence_inventory_sha256
                != self.action_evidence_inventory_sha256
                or self.runtime_trace.transition_inventory_sha256
                != self.transition_inventory_sha256
            )
        ):
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "adaptive runtime policy trace differs from its authority"
            )
        if validation_environment_runtime:
            assert self.runtime_validation_environment_registry is not None
            assert self.runtime_validation_environment_authority is not None
            registry = self.runtime_validation_environment_registry
            environment_authority = self.runtime_validation_environment_authority
            if (
                self.evaluation_role != "inner_validation"
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
                or environment_authority.fold_index != self.fold_index
                or environment_authority.semantic_receipt_sha256
                != registry.environment_authority(20.0).semantic_receipt_sha256
                or environment_authority.semantic_receipt_sha256
                != self.validation_environment_authority_receipt_sha256
                or environment_authority.validation_context_receipt_sha256
                != self.validation_context_receipt_sha256
                or environment_authority.environment_source_inventory_sha256
                != self.environment_source_inventory_sha256
                or environment_authority.economic_compatibility_receipt_sha256
                != self.economic_compatibility_receipt_sha256
            ):
                raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                    "adaptive runtime policy trace environment differs"
                )
        if barrier_runtime:
            assert self.runtime_four_fold_validation_inputs_authority is not None
            assert self.runtime_validation_environment_registry is not None
            barrier = self.runtime_four_fold_validation_inputs_authority
            from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v1 import (
                validate_massive_adaptive_rl_validation_outcome_barrier_v1,
            )

            validate_massive_adaptive_rl_validation_outcome_barrier_v1(
                authority=barrier,
                validation_environment_registry=(
                    self.runtime_validation_environment_registry
                ),
                fold_index=self.fold_index,
                outcome_committed_at_ms=self.loaded_source.commit.committed_at_ms,
            )
            if (
                barrier.semantic_receipt_sha256
                != self.four_fold_validation_inputs_authority_receipt_sha256
                or barrier.source_receipt_sha256
                != self.four_fold_validation_inputs_source_receipt_sha256
                or barrier.source_transaction_receipt_sha256
                != self.four_fold_validation_inputs_commit_receipt_sha256
                or barrier.source_transaction_committed_at_ms
                != self.four_fold_validation_inputs_committed_at_ms
            ):
                raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                    "adaptive runtime policy trace four-fold input differs"
                )
        for value in (
            self.checkpoint_authority_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.model_state_receipt_sha256,
            self.validation_context_receipt_sha256,
            self.environment_source_inventory_sha256,
            self.economic_compatibility_receipt_sha256,
            self.policy_trace_receipt_sha256,
            self.action_evidence_inventory_sha256,
            self.transition_inventory_sha256,
            self.protocol_receipt_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL policy trace authority", value)
        if self.validation_environment_authority_receipt_sha256 is not None:
            _digest(
                "adaptive RL policy trace validation environment authority",
                self.validation_environment_authority_receipt_sha256,
            )
        for registry_value in (
            self.validation_sources_authority_receipt_sha256,
            self.validation_environment_registry_receipt_sha256,
            self.validation_environment_registry_source_receipt_sha256,
            self.validation_environment_registry_commit_receipt_sha256,
            self.four_fold_validation_inputs_authority_receipt_sha256,
            self.four_fold_validation_inputs_source_receipt_sha256,
            self.four_fold_validation_inputs_commit_receipt_sha256,
        ):
            if registry_value is not None:
                _digest("adaptive RL policy trace validation registry", registry_value)
        if self.four_fold_validation_inputs_committed_at_ms is not None and (
            isinstance(self.four_fold_validation_inputs_committed_at_ms, bool)
            or self.four_fold_validation_inputs_committed_at_ms < 0
            or self.four_fold_validation_inputs_committed_at_ms
            >= self.loaded_source.commit.committed_at_ms
        ):
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "adaptive RL policy trace four-fold inputs were not committed first"
            )
        if self.validation_environment_registry_committed_at_ms is not None and (
            isinstance(self.validation_environment_registry_committed_at_ms, bool)
            or self.validation_environment_registry_committed_at_ms < 0
            or self.validation_environment_registry_committed_at_ms
            >= self.loaded_source.commit.committed_at_ms
        ):
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "adaptive RL policy trace validation registry was not committed first"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _payload(
    trace: MassiveAdaptiveRLCheckpointPolicyTraceV1,
    *,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ),
    validation_environment_authority: (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None
    ),
    four_fold_validation_inputs_authority: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1 | None
    ),
) -> dict[str, object]:
    registry = validation_environment_registry
    environment_authority = validation_environment_authority
    barrier = four_fold_validation_inputs_authority
    if (registry is None) != (environment_authority is None) or (registry is None) != (
        barrier is None
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace validation registry binding is partial"
        )
    registry_source_receipt = (
        None if registry is None else registry.source_receipt_sha256
    )
    registry_commit_receipt = (
        None if registry is None else registry.source_transaction_receipt_sha256
    )
    registry_committed_at_ms = (
        None if registry is None else registry.source_transaction_committed_at_ms
    )
    if registry is not None and (
        registry_source_receipt is None
        or registry_commit_receipt is None
        or registry_committed_at_ms is None
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace validation registry source transaction is absent"
        )
    if barrier is not None and (
        barrier.source_receipt_sha256 is None
        or barrier.source_transaction_receipt_sha256 is None
        or barrier.source_transaction_committed_at_ms is None
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace four-fold input source transaction is absent"
        )
    return {
        "fold_index": trace.fold_index,
        "evaluation_role": trace.evaluation_role,
        "checkpoint_authority_receipt_sha256": (
            trace.checkpoint_authority_receipt_sha256
        ),
        "checkpoint_receipt_sha256": trace.checkpoint_receipt_sha256,
        "model_state_receipt_sha256": trace.model_state_receipt_sha256,
        "validation_context_receipt_sha256": _digest(
            "validation context receipt",
            environment.validation_context_receipt_sha256,
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
            registry_source_receipt
        ),
        "validation_environment_registry_commit_receipt_sha256": (
            registry_commit_receipt
        ),
        "validation_environment_registry_committed_at_ms": (registry_committed_at_ms),
        "four_fold_validation_inputs_authority_receipt_sha256": (
            None if barrier is None else barrier.semantic_receipt_sha256
        ),
        "four_fold_validation_inputs_source_receipt_sha256": (
            None if barrier is None else barrier.source_receipt_sha256
        ),
        "four_fold_validation_inputs_commit_receipt_sha256": (
            None if barrier is None else barrier.source_transaction_receipt_sha256
        ),
        "four_fold_validation_inputs_committed_at_ms": (
            None if barrier is None else barrier.source_transaction_committed_at_ms
        ),
        "validation_environment_authority_receipt_sha256": (
            None
            if environment_authority is None
            else environment_authority.semantic_receipt_sha256
        ),
        "environment_source_inventory_sha256": _digest(
            "policy trace environment source inventory",
            environment.source_inventory_sha256,
        ),
        "economic_compatibility_receipt_sha256": _digest(
            "policy trace economic compatibility receipt",
            environment.economic_compatibility_receipt_sha256,
        ),
        "policy_trace": asdict(trace.policy_trace),
        "action_evidence": tuple(asdict(row) for row in trace.action_evidence),
        "action_evidence_inventory_sha256": trace.action_evidence_inventory_sha256,
        "transition_inventory_sha256": trace.transition_inventory_sha256,
        "source_data_qualified": trace.source_data_qualified,
        "checkpoint_policy_trace_receipt_sha256": trace.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace payload is not canonical JSON"
        )
    return dict(value)


def _trace_metadata(
    payload: Mapping[str, object],
) -> tuple[
    MassiveAdaptiveRLPolicyTraceV1,
    tuple[MassiveAdaptiveRLPolicyActionEvidenceV1, ...],
]:
    trace_payload = dict(payload["policy_trace"])  # type: ignore[call-overload]
    for name in (
        "decision_session_dates",
        "transition_receipts",
        "strategy_active_log_returns",
        "incremental_rl_log_returns",
    ):
        trace_payload[name] = tuple(trace_payload[name])
    trace = MassiveAdaptiveRLPolicyTraceV1(**trace_payload)  # type: ignore[arg-type]
    evidence = tuple(
        MassiveAdaptiveRLPolicyActionEvidenceV1(
            **{
                **dict(row),
                "action_values": tuple(dict(row)["action_values"]),
            }
        )
        for row in payload["action_evidence"]  # type: ignore[attr-defined]
    )
    trace.validate()
    for row in evidence:
        row.validate()
    return trace, evidence


def parse_massive_adaptive_rl_policy_trace_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLPolicyTraceAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    trace, evidence = _trace_metadata(payload)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA,
        "fold_index": int(str(payload["fold_index"])),
        "evaluation_role": str(payload["evaluation_role"]),
        "checkpoint_authority_receipt_sha256": str(
            payload["checkpoint_authority_receipt_sha256"]
        ),
        "checkpoint_receipt_sha256": str(payload["checkpoint_receipt_sha256"]),
        "model_state_receipt_sha256": str(payload["model_state_receipt_sha256"]),
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
            if payload["validation_environment_registry_source_receipt_sha256"] is None
            else str(payload["validation_environment_registry_source_receipt_sha256"])
        ),
        "validation_environment_registry_commit_receipt_sha256": (
            None
            if payload["validation_environment_registry_commit_receipt_sha256"] is None
            else str(payload["validation_environment_registry_commit_receipt_sha256"])
        ),
        "validation_environment_registry_committed_at_ms": (
            None
            if payload["validation_environment_registry_committed_at_ms"] is None
            else int(str(payload["validation_environment_registry_committed_at_ms"]))
        ),
        "four_fold_validation_inputs_authority_receipt_sha256": (
            None
            if payload["four_fold_validation_inputs_authority_receipt_sha256"] is None
            else str(payload["four_fold_validation_inputs_authority_receipt_sha256"])
        ),
        "four_fold_validation_inputs_source_receipt_sha256": (
            None
            if payload["four_fold_validation_inputs_source_receipt_sha256"] is None
            else str(payload["four_fold_validation_inputs_source_receipt_sha256"])
        ),
        "four_fold_validation_inputs_commit_receipt_sha256": (
            None
            if payload["four_fold_validation_inputs_commit_receipt_sha256"] is None
            else str(payload["four_fold_validation_inputs_commit_receipt_sha256"])
        ),
        "four_fold_validation_inputs_committed_at_ms": (
            None
            if payload["four_fold_validation_inputs_committed_at_ms"] is None
            else int(str(payload["four_fold_validation_inputs_committed_at_ms"]))
        ),
        "validation_environment_authority_receipt_sha256": (
            None
            if payload["validation_environment_authority_receipt_sha256"] is None
            else str(payload["validation_environment_authority_receipt_sha256"])
        ),
        "environment_source_inventory_sha256": str(
            payload["environment_source_inventory_sha256"]
        ),
        "economic_compatibility_receipt_sha256": str(
            payload["economic_compatibility_receipt_sha256"]
        ),
        "policy_trace_receipt_sha256": trace.semantic_receipt_sha256,
        "action_evidence_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in evidence)
        ),
        "transition_inventory_sha256": str(payload["transition_inventory_sha256"]),
        "source_data_qualified": bool(payload["source_data_qualified"]),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLPolicyTraceAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_trace=None,
        runtime_trace_replayed=False,
        runtime_validation_environment_registry=None,
        runtime_validation_environment_registry_replayed=False,
        runtime_four_fold_validation_inputs_authority=None,
        runtime_four_fold_validation_inputs_replayed=False,
        runtime_validation_environment_authority=None,
        runtime_validation_environment_replayed=False,
        development_policy_evaluation_authorized=False,
        outer_evaluation_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_policy_trace_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLPolicyTraceAuthorityV1,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1 | None = None,
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ) = None,
    four_fold_validation_inputs_authority: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1 | None
    ) = None,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLPolicyTraceAuthorityV1:
    parsed = parse_massive_adaptive_rl_policy_trace_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed = _load_payload(root=root, loaded_source=authority.loaded_source)
    resolved_environment, environment_authority = _resolve_evaluation_environment(
        environment=environment,
        validation_environment_registry=validation_environment_registry,
        four_fold_validation_inputs_authority=(four_fold_validation_inputs_authority),
        fold_index=parsed.fold_index,
        evaluation_role=parsed.evaluation_role,
        outcome_committed_at_ms=authority.loaded_source.commit.committed_at_ms,
    )
    replayed = evaluate_massive_adaptive_rl_checkpoint_v1(
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        environment=resolved_environment,
        fold_index=parsed.fold_index,
        evaluation_role=parsed.evaluation_role,
        device=device,
    )
    if canonical_json_file_bytes(committed) != canonical_json_file_bytes(
        _payload(
            replayed,
            environment=resolved_environment,
            validation_environment_registry=validation_environment_registry,
            validation_environment_authority=environment_authority,
            four_fold_validation_inputs_authority=(
                four_fold_validation_inputs_authority
            ),
        )
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace does not replay from its checkpoint"
        )
    result = replace(
        parsed,
        runtime_trace=replayed,
        runtime_trace_replayed=True,
        runtime_validation_environment_registry=validation_environment_registry,
        runtime_validation_environment_registry_replayed=(
            validation_environment_registry is not None
        ),
        runtime_four_fold_validation_inputs_authority=(
            four_fold_validation_inputs_authority
        ),
        runtime_four_fold_validation_inputs_replayed=(
            four_fold_validation_inputs_authority is not None
        ),
        runtime_validation_environment_authority=environment_authority,
        runtime_validation_environment_replayed=environment_authority is not None,
        development_policy_evaluation_authorized=(
            parsed.source_data_qualified
            and parsed.evaluation_role == "inner_validation"
            and validation_environment_registry is not None
            and four_fold_validation_inputs_authority is not None
        ),
        outer_evaluation_authorized=(
            parsed.source_data_qualified and parsed.evaluation_role == "outer_test"
        ),
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_policy_trace_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1 | None = None,
    fold_index: int,
    evaluation_role: str,
    committed_at_ms: int,
    validation_environment_registry: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None
    ) = None,
    four_fold_validation_inputs_authority: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1 | None
    ) = None,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLPolicyTraceAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace artifact ID is not path safe"
        )
    resolved_environment, environment_authority = _resolve_evaluation_environment(
        environment=environment,
        validation_environment_registry=validation_environment_registry,
        four_fold_validation_inputs_authority=(four_fold_validation_inputs_authority),
        fold_index=fold_index,
        evaluation_role=evaluation_role,
        outcome_committed_at_ms=committed_at_ms,
    )
    trace = evaluate_massive_adaptive_rl_checkpoint_v1(
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        environment=resolved_environment,
        fold_index=fold_index,
        evaluation_role=evaluation_role,
        device=device,
    )
    relative = f"massive-adaptive/rl-policy-trace-authority-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(
            canonical_json_file_bytes(
                _payload(
                    trace,
                    environment=resolved_environment,
                    validation_environment_registry=validation_environment_registry,
                    validation_environment_authority=environment_authority,
                    four_fold_validation_inputs_authority=(
                        four_fold_validation_inputs_authority
                    ),
                )
            )
        ),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=trace.policy_trace.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-POLICY-TRACE-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_policy_trace_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_policy_trace_authority_v1(
            root=root, loaded_source=loaded
        ),
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        environment=(resolved_environment if evaluation_role == "outer_test" else None),
        validation_environment_registry=validation_environment_registry,
        four_fold_validation_inputs_authority=(four_fold_validation_inputs_authority),
        device=device,
    )


__all__ = [
    "MassiveAdaptiveRLPolicyTraceAuthorityV1",
    "MassiveAdaptiveRLPolicyTraceAuthorityV1Error",
    "authorize_massive_adaptive_rl_policy_trace_authority_v1",
    "materialize_massive_adaptive_rl_policy_trace_authority_v1",
    "parse_massive_adaptive_rl_policy_trace_authority_v1",
]
