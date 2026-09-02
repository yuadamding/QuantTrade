"""Create-only replay authority for the fit-selected FC06 validation trace."""

from __future__ import annotations

from collections.abc import Mapping
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
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_evaluator_v1 import (
    MassiveAdaptiveRLFixedControlEvaluationV1,
    evaluate_massive_adaptive_rl_fixed_control_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    build_massive_adaptive_rl_fixed_control_registry_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
)

if TYPE_CHECKING:
    from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1,
    )


MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-validation-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fixed-control-validation-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (
                MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SCHEMA
            ),
            "payload": "canonical-json-replayed-fc06-validation-evaluation",
            "validation_environment": (
                "canonical-authority-plus-static-environment-identities"
            ),
            "dynamic_economic_sources": "transition-derived-separate-inventory",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "controller": "fit-selected-fc06",
        "role": "inner-validation-primary-cost-only",
        "evaluation": "package-owned-economic-replay",
        "validation_environment": "canonical-20bp-authority",
        "caller_metrics": False,
        "profitability_reporting": False,
        "outer_access": False,
    }
)


class MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(ValueError):
    """The persisted FC06 validation trace did not replay exactly."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
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


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
            "FC06 validation artifact ID is not path safe"
        )
    return value


def _validate_validation_environment_binding(
    *,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    environment_authority: MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None,
    fold_index: int,
) -> MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None:
    if environment_authority is None:
        return None
    if type(environment_authority) is not _validation_environment_authority_type():
        raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
            "FC06 validation environment authority type differs"
        )
    environment_authority.validate()
    environment_authority.validate_environment(environment)
    if (
        environment_authority.fold_index != fold_index
        or environment_authority.transaction_cost_basis_points != 20.0
        or not environment_authority.source_data_qualified
    ):
        raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
            "FC06 validation environment authority differs"
        )
    return environment_authority


def _evaluation_payload(
    evaluation: MassiveAdaptiveRLFixedControlEvaluationV1,
    *,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    validation_environment_authority: (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None
    ),
) -> dict[str, object]:
    evaluation.validate()
    environment_authority = _validate_validation_environment_binding(
        environment=environment,
        environment_authority=validation_environment_authority,
        fold_index=evaluation.fold_index,
    )
    return {
        "evaluation": asdict(evaluation),
        "validation_environment_authority_receipt_sha256": (
            None
            if environment_authority is None
            else environment_authority.semantic_receipt_sha256
        ),
        "environment_source_inventory_sha256": _digest(
            "FC06 validation environment source inventory",
            environment.source_inventory_sha256,
        ),
        "economic_compatibility_receipt_sha256": _digest(
            "FC06 validation economic compatibility receipt",
            environment.economic_compatibility_receipt_sha256,
        ),
    }


def _parse_evaluation(value: object) -> MassiveAdaptiveRLFixedControlEvaluationV1:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
            "FC06 validation evaluation payload differs"
        )
    payload = dict(value)
    trace_payload = dict(payload["policy_trace"])  # type: ignore[arg-type]
    for name in (
        "decision_session_dates",
        "transition_receipts",
        "strategy_active_log_returns",
        "incremental_rl_log_returns",
    ):
        trace_payload[name] = tuple(trace_payload[name])
    payload["policy_trace"] = MassiveAdaptiveRLPolicyTraceV1(**trace_payload)  # type: ignore[arg-type]
    payload["transition_receipts"] = tuple(payload["transition_receipts"])
    result = MassiveAdaptiveRLFixedControlEvaluationV1(**payload)  # type: ignore[arg-type]
    result.validate()
    return result


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
            "FC06 validation payload is not canonical JSON"
        )
    return dict(value)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlValidationAuthorityV1:
    fold_index: int
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_action_receipt_sha256: str
    validation_context_receipt_sha256: str
    validation_environment_authority_receipt_sha256: str | None
    environment_source_inventory_sha256: str
    economic_compatibility_receipt_sha256: str
    evaluation_receipt_sha256: str
    policy_trace_receipt_sha256: str
    transition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_evaluation: MassiveAdaptiveRLFixedControlEvaluationV1 | None
    runtime_evaluation_replayed: bool
    runtime_validation_environment_authority: (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None
    )
    runtime_validation_environment_replayed: bool
    development_validation_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "fixed_control_fit_authority_receipt_sha256": (
                self.fixed_control_fit_authority_receipt_sha256
            ),
            "fixed_control_selection_authority_receipt_sha256": (
                self.fixed_control_selection_authority_receipt_sha256
            ),
            "selected_action_receipt_sha256": self.selected_action_receipt_sha256,
            "validation_context_receipt_sha256": (
                self.validation_context_receipt_sha256
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
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "policy_trace_receipt_sha256": self.policy_trace_receipt_sha256,
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    @property
    def source_transaction_verified(self) -> bool:
        return True

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_evaluation_replayed
            and self.development_validation_authorized
            and self.source_data_qualified
        )

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_evaluation is not None
        environment_runtime = self.runtime_validation_environment_authority is not None
        if self.runtime_validation_environment_authority is not None:
            if type(self.runtime_validation_environment_authority) is not (
                _validation_environment_authority_type()
            ):
                raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
                    "FC06 validation environment authority type differs"
                )
            self.runtime_validation_environment_authority.validate()
        expected_authorized = bool(runtime and self.source_data_qualified)
        if self.runtime_evaluation is not None:
            if (
                type(self.runtime_evaluation)
                is not MassiveAdaptiveRLFixedControlEvaluationV1
            ):
                raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
                    "FC06 validation runtime type differs"
                )
            self.runtime_evaluation.validate()
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SCHEMA
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or not isinstance(self.source_data_qualified, bool)
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.evaluation_receipt_sha256
            or self.runtime_evaluation_replayed != runtime
            or self.runtime_validation_environment_replayed != environment_runtime
            or (environment_runtime and not runtime)
            or (
                runtime
                and self.validation_environment_authority_receipt_sha256 is not None
                and not environment_runtime
            )
            or self.development_validation_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
                "FC06 validation authority differs"
            )
        if runtime:
            assert self.runtime_evaluation is not None
            evaluation = self.runtime_evaluation
            if (
                evaluation.fold_index != self.fold_index
                or evaluation.fixed_control_fit_authority_receipt_sha256
                != self.fixed_control_fit_authority_receipt_sha256
                or evaluation.fixed_control_selection_authority_receipt_sha256
                != self.fixed_control_selection_authority_receipt_sha256
                or evaluation.selected_action_receipt_sha256
                != self.selected_action_receipt_sha256
                or evaluation.validation_context_receipt_sha256
                != self.validation_context_receipt_sha256
                or evaluation.semantic_receipt_sha256 != self.evaluation_receipt_sha256
                or evaluation.policy_trace.semantic_receipt_sha256
                != self.policy_trace_receipt_sha256
                or evaluation.transition_inventory_sha256
                != self.transition_inventory_sha256
                or evaluation.source_data_qualified != self.source_data_qualified
            ):
                raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
                    "FC06 validation runtime differs from its authority"
                )
        if environment_runtime:
            assert self.runtime_validation_environment_authority is not None
            environment_authority = self.runtime_validation_environment_authority
            if (
                environment_authority.fold_index != self.fold_index
                or environment_authority.transaction_cost_basis_points != 20.0
                or environment_authority.semantic_receipt_sha256
                != self.validation_environment_authority_receipt_sha256
                or environment_authority.validation_context_receipt_sha256
                != self.validation_context_receipt_sha256
                or environment_authority.environment_source_inventory_sha256
                != self.environment_source_inventory_sha256
                or environment_authority.economic_compatibility_receipt_sha256
                != self.economic_compatibility_receipt_sha256
            ):
                raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
                    "FC06 validation runtime environment differs"
                )
        for value in (
            self.fixed_control_fit_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.selected_action_receipt_sha256,
            self.validation_context_receipt_sha256,
            self.environment_source_inventory_sha256,
            self.economic_compatibility_receipt_sha256,
            self.evaluation_receipt_sha256,
            self.policy_trace_receipt_sha256,
            self.transition_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("FC06 validation authority", value)
        if self.validation_environment_authority_receipt_sha256 is not None:
            _digest(
                "FC06 validation environment authority",
                self.validation_environment_authority_receipt_sha256,
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _authority_body(
    *,
    evaluation: MassiveAdaptiveRLFixedControlEvaluationV1,
    validation_environment_authority_receipt_sha256: str | None,
    environment_source_inventory_sha256: str,
    economic_compatibility_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SCHEMA,
        "fold_index": evaluation.fold_index,
        "fixed_control_fit_authority_receipt_sha256": (
            evaluation.fixed_control_fit_authority_receipt_sha256
        ),
        "fixed_control_selection_authority_receipt_sha256": (
            evaluation.fixed_control_selection_authority_receipt_sha256
        ),
        "selected_action_receipt_sha256": (evaluation.selected_action_receipt_sha256),
        "validation_context_receipt_sha256": (
            evaluation.validation_context_receipt_sha256
        ),
        "validation_environment_authority_receipt_sha256": (
            validation_environment_authority_receipt_sha256
        ),
        "environment_source_inventory_sha256": environment_source_inventory_sha256,
        "economic_compatibility_receipt_sha256": (
            economic_compatibility_receipt_sha256
        ),
        "evaluation_receipt_sha256": evaluation.semantic_receipt_sha256,
        "policy_trace_receipt_sha256": (
            evaluation.policy_trace.semantic_receipt_sha256
        ),
        "transition_inventory_sha256": evaluation.transition_inventory_sha256,
        "source_data_qualified": evaluation.source_data_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
        ),
    }


def parse_massive_adaptive_rl_fixed_control_validation_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFixedControlValidationAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    evaluation = _parse_evaluation(payload["evaluation"])
    body = _authority_body(
        evaluation=evaluation,
        validation_environment_authority_receipt_sha256=(
            None
            if payload["validation_environment_authority_receipt_sha256"] is None
            else str(payload["validation_environment_authority_receipt_sha256"])
        ),
        environment_source_inventory_sha256=str(
            payload["environment_source_inventory_sha256"]
        ),
        economic_compatibility_receipt_sha256=str(
            payload["economic_compatibility_receipt_sha256"]
        ),
    )
    result = MassiveAdaptiveRLFixedControlValidationAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        loaded_source=loaded_source,
        runtime_evaluation=None,
        runtime_evaluation_replayed=False,
        runtime_validation_environment_authority=None,
        runtime_validation_environment_replayed=False,
        development_validation_authorized=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_fixed_control_validation_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFixedControlValidationAuthorityV1,
    fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    validation_environment_authority: (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None
    ) = None,
) -> MassiveAdaptiveRLFixedControlValidationAuthorityV1:
    parsed = parse_massive_adaptive_rl_fixed_control_validation_authority_v1(
        root=root,
        loaded_source=authority.loaded_source,
    )
    committed_payload = _load_payload(root=root, loaded_source=authority.loaded_source)
    committed = _parse_evaluation(committed_payload["evaluation"])
    environment_authority = _validate_validation_environment_binding(
        environment=environment,
        environment_authority=validation_environment_authority,
        fold_index=parsed.fold_index,
    )
    replayed = evaluate_massive_adaptive_rl_fixed_control_v1(
        registry=build_massive_adaptive_rl_fixed_control_registry_v1(),
        fit_authority=fit_authority,
        selection_authority=selection_authority,
        chronology_authority=chronology_authority,
        environment=environment,
    )
    if committed != replayed or canonical_json_file_bytes(
        committed_payload
    ) != canonical_json_file_bytes(
        _evaluation_payload(
            replayed,
            environment=environment,
            validation_environment_authority=environment_authority,
        )
    ):
        raise MassiveAdaptiveRLFixedControlValidationAuthorityV1Error(
            "FC06 validation evaluation does not replay"
        )
    result = replace(
        parsed,
        runtime_evaluation=replayed,
        runtime_evaluation_replayed=True,
        runtime_validation_environment_authority=environment_authority,
        runtime_validation_environment_replayed=environment_authority is not None,
        development_validation_authorized=replayed.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_fixed_control_validation_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    committed_at_ms: int,
    validation_environment_authority: (
        MassiveAdaptiveRLValidationEnvironmentAuthorityV1 | None
    ) = None,
) -> MassiveAdaptiveRLFixedControlValidationAuthorityV1:
    artifact = _artifact_id(artifact_id)
    environment_authority = _validate_validation_environment_binding(
        environment=environment,
        environment_authority=validation_environment_authority,
        fold_index=chronology_authority.fold_index,
    )
    evaluation = evaluate_massive_adaptive_rl_fixed_control_v1(
        registry=build_massive_adaptive_rl_fixed_control_registry_v1(),
        fit_authority=fit_authority,
        selection_authority=selection_authority,
        chronology_authority=chronology_authority,
        environment=environment,
    )
    relative = (
        f"massive-adaptive/rl-fixed-control-validation-authority-v1/{artifact}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(
            canonical_json_file_bytes(
                _evaluation_payload(
                    evaluation,
                    environment=environment,
                    validation_environment_authority=environment_authority,
                )
            )
        ),
        root=root,
        relative_payload_path=relative,
        dataset_id=(MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_DATASET),
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=evaluation.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-FC06-VALIDATION-V1-{artifact}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_fixed_control_validation_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_fixed_control_validation_authority_v1(
            root=root,
            loaded_source=loaded,
        ),
        fit_authority=fit_authority,
        selection_authority=selection_authority,
        chronology_authority=chronology_authority,
        environment=environment,
        validation_environment_authority=environment_authority,
    )


__all__ = [
    "MassiveAdaptiveRLFixedControlValidationAuthorityV1",
    "MassiveAdaptiveRLFixedControlValidationAuthorityV1Error",
    "authorize_massive_adaptive_rl_fixed_control_validation_authority_v1",
    "materialize_massive_adaptive_rl_fixed_control_validation_authority_v1",
    "parse_massive_adaptive_rl_fixed_control_validation_authority_v1",
]
