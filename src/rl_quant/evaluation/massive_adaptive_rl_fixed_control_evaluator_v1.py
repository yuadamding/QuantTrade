"""Package-owned validation evaluator for the fit-selected FC06 control."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import MassiveAdaptiveRLActionV1
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    MassiveAdaptiveRLFixedControlRegistryV1,
    registered_massive_adaptive_rl_constant_actions_v1,
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_trace_from_identities_v1,
)


MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_EVALUATION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-evaluation-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_EVALUATION_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_EVALUATION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "controller": "FC06-fit-selected-FC00-through-FC05-constant",
        "fit_role": "rl-fit-only",
        "evaluation_role": "inner-validation",
        "action_generation": "same-selected-bounded-action-every-decision",
        "caller_transitions": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLFixedControlEvaluationV1Error(ValueError):
    """FC06 was not selected on fit data and replayed on validation."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFixedControlEvaluationV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlEvaluationV1:
    fold_index: int
    fixed_control_registry_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fit_control_id: str
    selected_action_receipt_sha256: str
    validation_context_receipt_sha256: str
    policy_trace: MassiveAdaptiveRLPolicyTraceV1
    transition_receipts: tuple[str, ...]
    transition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_policy_selection_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_EVALUATION_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_EVALUATION_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_EVALUATION_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "fixed_control_registry_receipt_sha256": (
                self.fixed_control_registry_receipt_sha256
            ),
            "fixed_control_fit_authority_receipt_sha256": (
                self.fixed_control_fit_authority_receipt_sha256
            ),
            "fixed_control_selection_authority_receipt_sha256": (
                self.fixed_control_selection_authority_receipt_sha256
            ),
            "selected_fit_control_id": self.selected_fit_control_id,
            "selected_action_receipt_sha256": self.selected_action_receipt_sha256,
            "validation_context_receipt_sha256": (
                self.validation_context_receipt_sha256
            ),
            "policy_trace_receipt_sha256": self.policy_trace.semantic_receipt_sha256,
            "transition_receipts": self.transition_receipts,
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.policy_trace.validate()
        for value in (
            self.fixed_control_registry_receipt_sha256,
            self.fixed_control_fit_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.selected_action_receipt_sha256,
            self.validation_context_receipt_sha256,
            *self.transition_receipts,
            self.transition_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL FC06 validation evaluation", value)
        expected = self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_EVALUATION_V1_SCHEMA
            or self.fold_index != self.policy_trace.fold_index
            or self.policy_trace.evaluation_role != "inner_validation"
            or self.policy_trace.transaction_cost_basis_points != 20.0
            or self.policy_trace.frozen_targets_replayed
            or self.transition_receipts != self.policy_trace.transition_receipts
            or self.transition_inventory_sha256
            != semantic_sha256(self.transition_receipts)
            or self.development_policy_selection_authorized != expected
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlEvaluationV1Error(
                "adaptive RL FC06 validation evaluation differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def evaluate_massive_adaptive_rl_fixed_control_v1(
    *,
    registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveRLFixedControlEvaluationV1:
    """Run the fit-selected constant over the sealed validation chronology."""

    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
        registry=registry,
        fit_authority=fit_authority,
        selection_authority=selection_authority,
        chronology_authority=chronology_authority,
    )
    selection = selection_authority.runtime_selection
    if selection is None:
        raise MassiveAdaptiveRLFixedControlEvaluationV1Error(
            "adaptive RL FC06 selection is absent"
        )
    action_by_receipt = {
        action.semantic_receipt_sha256: action
        for _control_id, action in registered_massive_adaptive_rl_constant_actions_v1()
    }
    try:
        action: MassiveAdaptiveRLActionV1 = action_by_receipt[
            selection.selected_action_receipt_sha256
        ]
    except KeyError as error:
        raise MassiveAdaptiveRLFixedControlEvaluationV1Error(
            "adaptive RL FC06 selected an action outside the frozen grid"
        ) from error
    dates = tuple(row.decision_session_date for row in environment.inference_plan.rows)
    if (
        dates != chronology_authority.rl_validation_origin_dates
        or environment.transaction_cost_basis_points != 20.0
    ):
        raise MassiveAdaptiveRLFixedControlEvaluationV1Error(
            "adaptive RL FC06 validation chronology or cost differs"
        )
    environment.reset()
    transitions: list[MassiveAdaptiveRLTransitionV1] = []
    for index in range(len(dates)):
        _next, _reward, terminated, truncated, info = environment.step(action)
        transition = info.get("transition")
        if (
            truncated
            or terminated != (index == len(dates) - 1)
            or not isinstance(transition, MassiveAdaptiveRLTransitionV1)
        ):
            raise MassiveAdaptiveRLFixedControlEvaluationV1Error(
                "adaptive RL FC06 validation transition differs"
            )
        transitions.append(transition)
    controller_identity = semantic_sha256(
        (
            "FC06",
            registry.semantic_receipt_sha256,
            fit_authority.semantic_receipt_sha256,
            selection_authority.semantic_receipt_sha256,
            selection.selected_action_receipt_sha256,
        )
    )
    trace = build_massive_adaptive_rl_policy_trace_from_identities_v1(
        fold_index=chronology_authority.fold_index,
        checkpoint_receipt_sha256=controller_identity,
        model_state_receipt_sha256=selection.selected_action_receipt_sha256,
        update_index=0,
        training_forecast_authority_receipt_sha256=(
            chronology_authority.training_forecast_authority_receipt_sha256
        ),
        forecast_archive_receipt_sha256=(
            environment.forecast_archive.semantic_receipt_sha256
        ),
        inference_plan_receipt_sha256=(
            environment.inference_plan.semantic_receipt_sha256
        ),
        calibration_receipt_sha256=environment.calibration.semantic_receipt_sha256,
        transaction_cost_basis_points=environment.transaction_cost_basis_points,
        initial_capital=environment.initial_capital,
        transitions=tuple(transitions),
        frozen_targets_replayed=False,
        evaluation_role="inner_validation",
        checkpoint_source_data_qualified=bool(
            selection_authority.development_control_selection_authorized
            and fit_authority.development_control_fit_authorized
            and chronology_authority.development_policy_selection_authorized
        ),
    )
    transition_receipts = tuple(row.semantic_receipt_sha256 for row in transitions)
    provisional = MassiveAdaptiveRLFixedControlEvaluationV1(
        fold_index=chronology_authority.fold_index,
        fixed_control_registry_receipt_sha256=registry.semantic_receipt_sha256,
        fixed_control_fit_authority_receipt_sha256=(
            fit_authority.semantic_receipt_sha256
        ),
        fixed_control_selection_authority_receipt_sha256=(
            selection_authority.semantic_receipt_sha256
        ),
        selected_fit_control_id=selection.selected_control_id,
        selected_action_receipt_sha256=selection.selected_action_receipt_sha256,
        validation_context_receipt_sha256=(
            environment.validation_context_receipt_sha256
        ),
        policy_trace=trace,
        transition_receipts=transition_receipts,
        transition_inventory_sha256=semantic_sha256(transition_receipts),
        source_data_qualified=trace.source_data_qualified,
        semantic_receipt_sha256="0" * 64,
        development_policy_selection_authorized=trace.source_data_qualified,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLFixedControlEvaluationV1",
    "MassiveAdaptiveRLFixedControlEvaluationV1Error",
    "evaluate_massive_adaptive_rl_fixed_control_v1",
]
