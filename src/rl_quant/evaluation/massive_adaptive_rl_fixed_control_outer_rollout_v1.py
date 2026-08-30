"""Selection-authenticated outer rollout for the fit-selected FC06 control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
from pathlib import Path
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v1 import (
    MassiveAdaptiveRLOuterPlanV1,
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
    MassiveAdaptiveRLFixedControlRegistryV1,
    registered_massive_adaptive_rl_constant_actions_v1,
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_trace_from_identities_v1,
)


MASSIVE_ADAPTIVE_FIXED_CONTROL_OUTER_ACTION_EVIDENCE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-fixed-control-outer-action-evidence-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-outer-rollout-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-outer-rollout-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fixed-control-outer-rollout-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "controller": "fit-selected-FC06-constant",
        "selection": "package-replayed-FC00-through-FC05-fit-authority",
        "outer_action": "registered-selected-action-every-decision",
        "updates": False,
        "primary_cost_basis_points": 20.0,
        "caller_actions_or_transitions": False,
        "duration_semantics": False,
    }
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA,
        "payload": "fit-selection-and-outer-action-economic-trace",
        "promotion": "reopen-selection-rerun-action-and-economics",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLFixedControlOuterRolloutV1Error(ValueError):
    """The selected constant or its outer economic replay differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveFixedControlOuterActionEvidenceV1:
    decision_session_date: str
    observation_receipt_sha256: str
    fixed_control_registry_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_control_id: str
    action_receipt_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_FIXED_CONTROL_OUTER_ACTION_EVIDENCE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema
            != MASSIVE_ADAPTIVE_FIXED_CONTROL_OUTER_ACTION_EVIDENCE_V1_SCHEMA
            or not self.decision_session_date
            or self.selected_control_id
            not in tuple(
                control_id
                for control_id, _action in registered_massive_adaptive_rl_constant_actions_v1()
            )
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
                "fixed-control outer action evidence differs"
            )
        for value in (
            self.observation_receipt_sha256,
            self.fixed_control_registry_receipt_sha256,
            self.fixed_control_fit_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.action_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("fixed-control outer action evidence", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlOuterRolloutV1:
    fold_index: int
    outer_plan_receipt_sha256: str
    fixed_control_registry_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_control_id: str
    selected_action_receipt_sha256: str
    policy_trace: MassiveAdaptiveRLPolicyTraceV1
    action_evidence: tuple[MassiveAdaptiveFixedControlOuterActionEvidenceV1, ...]
    transitions: tuple[MassiveAdaptiveRLTransitionV1, ...]
    action_inventory_sha256: str
    transition_inventory_sha256: str
    environment_source_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    selected_control_replayed: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_V1_SPEC_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "outer_plan_receipt_sha256": self.outer_plan_receipt_sha256,
            "fixed_control_registry_receipt_sha256": self.fixed_control_registry_receipt_sha256,
            "fixed_control_fit_authority_receipt_sha256": self.fixed_control_fit_authority_receipt_sha256,
            "fixed_control_selection_authority_receipt_sha256": self.fixed_control_selection_authority_receipt_sha256,
            "selected_control_id": self.selected_control_id,
            "selected_action_receipt_sha256": self.selected_action_receipt_sha256,
            "policy_trace_receipt_sha256": self.policy_trace.semantic_receipt_sha256,
            "action_inventory_sha256": self.action_inventory_sha256,
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "environment_source_inventory_sha256": self.environment_source_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        self.policy_trace.validate()
        for evidence_row in self.action_evidence:
            evidence_row.validate()
        for transition_row in self.transitions:
            transition_row.validate()
        runtime = bool(self.action_evidence and self.transitions)
        expected = runtime and self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_V1_SCHEMA
            or self.fold_index != self.policy_trace.fold_index
            or self.policy_trace.evaluation_role != "outer_test"
            or self.policy_trace.transaction_cost_basis_points != 20.0
            or self.policy_trace.frozen_targets_replayed
            or self.policy_trace.model_state_receipt_sha256
            != self.selected_action_receipt_sha256
            or self.policy_trace.transition_receipts
            != tuple(row.semantic_receipt_sha256 for row in self.transitions)
            or len(self.action_evidence) != len(self.transitions)
            or tuple(row.action_receipt_sha256 for row in self.action_evidence)
            != tuple(row.action_receipt_sha256 for row in self.transitions)
            or tuple(row.observation_receipt_sha256 for row in self.action_evidence)
            != tuple(row.observation_receipt_sha256 for row in self.transitions)
            or any(
                row.selected_control_id != self.selected_control_id
                or row.action_receipt_sha256 != self.selected_action_receipt_sha256
                for row in self.action_evidence
            )
            or self.action_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.action_evidence)
            )
            or self.transition_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.transitions)
            )
            or self.selected_control_replayed != runtime
            or self.outer_evaluation_authorized != expected
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
                "fixed-control outer rollout differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def run_massive_adaptive_rl_fixed_control_outer_rollout_v1(
    *,
    outer_plan: MassiveAdaptiveRLOuterPlanV1,
    registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveRLFixedControlOuterRolloutV1:
    """Run the fit-selected registered constant on one sealed outer tape."""

    outer_plan.validate()
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
        registry=registry,
        fit_authority=fit_authority,
        selection_authority=selection_authority,
        chronology_authority=chronology_authority,
    )
    selection = selection_authority.runtime_selection
    fit_run = fit_authority.runtime_fit_run
    if selection is None or fit_run is None:
        raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
            "fixed-control fit selection is unavailable"
        )
    actions = {
        action.semantic_receipt_sha256: action
        for _control_id, action in registered_massive_adaptive_rl_constant_actions_v1()
    }
    action = actions.get(selection.selected_action_receipt_sha256)
    dates = tuple(row.decision_session_date for row in environment.inference_plan.rows)
    if (
        action is None
        or chronology_authority.fold_index != outer_plan.fold_index
        or not chronology_authority.outer_evaluation_authorized
        or chronology_authority.outer_inference_plan_receipt_sha256
        != outer_plan.outer_inference_plan_receipt_sha256
        or chronology_authority.outer_origin_dates != dates
        or environment.inference_plan.fold_index != outer_plan.fold_index
        or not bool(
            getattr(environment.inference_plan, "outer_inference_authorized", False)
        )
        or environment.inference_plan.semantic_receipt_sha256
        != outer_plan.outer_inference_plan_receipt_sha256
        or environment.forecast_archive.semantic_receipt_sha256
        != outer_plan.outer_forecast_archive_receipt_sha256
        or environment.calibration.semantic_receipt_sha256
        != outer_plan.calibration_receipt_sha256
        or environment.compiler_config.receipt_sha256
        != outer_plan.compiler_config_receipt_sha256
        or environment.initial_capital != outer_plan.primary_capital
        or environment.transaction_cost_basis_points
        != outer_plan.primary_cost_basis_points
    ):
        raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
            "fixed-control outer environment or selected action differs"
        )
    observation, _ = environment.reset()
    evidence: list[MassiveAdaptiveFixedControlOuterActionEvidenceV1] = []
    transitions: list[MassiveAdaptiveRLTransitionV1] = []
    while True:
        decision_date = environment.inference_plan.rows[
            environment.state.chronology_cursor
        ].decision_session_date
        evidence_body = {
            "schema": MASSIVE_ADAPTIVE_FIXED_CONTROL_OUTER_ACTION_EVIDENCE_V1_SCHEMA,
            "decision_session_date": decision_date,
            "observation_receipt_sha256": observation.semantic_receipt_sha256,
            "fixed_control_registry_receipt_sha256": registry.semantic_receipt_sha256,
            "fixed_control_fit_authority_receipt_sha256": fit_authority.semantic_receipt_sha256,
            "fixed_control_selection_authority_receipt_sha256": selection_authority.semantic_receipt_sha256,
            "selected_control_id": selection.selected_control_id,
            "action_receipt_sha256": action.semantic_receipt_sha256,
            "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        }
        evidence_row = MassiveAdaptiveFixedControlOuterActionEvidenceV1(
            **evidence_body,  # type: ignore[arg-type]
            semantic_receipt_sha256=semantic_sha256(evidence_body),
        )
        evidence_row.validate()
        evidence.append(evidence_row)
        next_observation, _reward, terminated, truncated, info = environment.step(
            action
        )
        transition = info.get("transition")
        if truncated or not isinstance(transition, MassiveAdaptiveRLTransitionV1):
            raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
                "fixed-control outer transition differs"
            )
        transitions.append(transition)
        if terminated:
            break
        if next_observation is None:
            raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
                "fixed-control outer next observation is absent"
            )
        observation = next_observation
    controller_identity = semantic_sha256(
        (
            "FC06-outer",
            registry.semantic_receipt_sha256,
            fit_authority.semantic_receipt_sha256,
            selection_authority.semantic_receipt_sha256,
            action.semantic_receipt_sha256,
        )
    )
    trace = build_massive_adaptive_rl_policy_trace_from_identities_v1(
        fold_index=outer_plan.fold_index,
        checkpoint_receipt_sha256=controller_identity,
        model_state_receipt_sha256=action.semantic_receipt_sha256,
        update_index=0,
        training_forecast_authority_receipt_sha256=(
            fit_run.training_forecast_authority_receipt_sha256
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
        evaluation_role="outer_test",
        checkpoint_source_data_qualified=bool(
            fit_authority.development_control_fit_authorized
            and selection_authority.development_control_selection_authorized
            and chronology_authority.outer_evaluation_authorized
            and outer_plan.outer_evaluation_authorized
        ),
    )
    source_qualified = bool(
        outer_plan.outer_evaluation_authorized
        and fit_authority.development_control_fit_authorized
        and selection_authority.development_control_selection_authorized
        and all(row.source_data_qualified for row in transitions)
    )
    action_inventory = semantic_sha256(
        tuple(row.semantic_receipt_sha256 for row in evidence)
    )
    transition_inventory = semantic_sha256(
        tuple(row.semantic_receipt_sha256 for row in transitions)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_V1_SCHEMA,
        "fold_index": outer_plan.fold_index,
        "outer_plan_receipt_sha256": outer_plan.semantic_receipt_sha256,
        "fixed_control_registry_receipt_sha256": registry.semantic_receipt_sha256,
        "fixed_control_fit_authority_receipt_sha256": fit_authority.semantic_receipt_sha256,
        "fixed_control_selection_authority_receipt_sha256": selection_authority.semantic_receipt_sha256,
        "selected_control_id": selection.selected_control_id,
        "selected_action_receipt_sha256": action.semantic_receipt_sha256,
        "policy_trace": trace,
        "action_evidence": tuple(evidence),
        "transitions": tuple(transitions),
        "action_inventory_sha256": action_inventory,
        "transition_inventory_sha256": transition_inventory,
        "environment_source_inventory_sha256": environment.source_inventory_sha256,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLFixedControlOuterRolloutV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        selected_control_replayed=True,
        outer_evaluation_authorized=source_qualified,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1:
    fold_index: int
    outer_plan_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    outer_rollout_receipt_sha256: str
    policy_trace_receipt_sha256: str
    action_inventory_sha256: str
    transition_inventory_sha256: str
    environment_source_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_rollout: MassiveAdaptiveRLFixedControlOuterRolloutV1 | None
    runtime_rollout_replayed: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "loaded_source",
                "runtime_rollout",
                "runtime_rollout_replayed",
                "outer_evaluation_authorized",
            }
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_rollout is not None
        if self.runtime_rollout is not None:
            self.runtime_rollout.validate()
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.outer_rollout_receipt_sha256
            or self.runtime_rollout_replayed != runtime
            or self.outer_evaluation_authorized
            != (runtime and self.source_data_qualified)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
                "fixed-control outer rollout authority differs"
            )
        if (
            runtime
            and self.runtime_rollout is not None
            and (
                self.runtime_rollout.fold_index != self.fold_index
                or self.runtime_rollout.outer_plan_receipt_sha256
                != self.outer_plan_receipt_sha256
                or self.runtime_rollout.fixed_control_fit_authority_receipt_sha256
                != self.fixed_control_fit_authority_receipt_sha256
                or self.runtime_rollout.fixed_control_selection_authority_receipt_sha256
                != self.fixed_control_selection_authority_receipt_sha256
                or self.runtime_rollout.semantic_receipt_sha256
                != self.outer_rollout_receipt_sha256
                or self.runtime_rollout.policy_trace.semantic_receipt_sha256
                != self.policy_trace_receipt_sha256
                or self.runtime_rollout.action_inventory_sha256
                != self.action_inventory_sha256
                or self.runtime_rollout.transition_inventory_sha256
                != self.transition_inventory_sha256
                or self.runtime_rollout.environment_source_inventory_sha256
                != self.environment_source_inventory_sha256
            )
        ):
            raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
                "runtime fixed-control outer rollout differs from its authority"
            )
        for value in (
            self.outer_plan_receipt_sha256,
            self.fixed_control_fit_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.outer_rollout_receipt_sha256,
            self.policy_trace_receipt_sha256,
            self.action_inventory_sha256,
            self.transition_inventory_sha256,
            self.environment_source_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("fixed-control outer rollout authority", value)


def _payload(
    rollout: MassiveAdaptiveRLFixedControlOuterRolloutV1,
) -> dict[str, object]:
    return {
        "fold_index": rollout.fold_index,
        "outer_plan_receipt_sha256": rollout.outer_plan_receipt_sha256,
        "fixed_control_registry_receipt_sha256": rollout.fixed_control_registry_receipt_sha256,
        "fixed_control_fit_authority_receipt_sha256": rollout.fixed_control_fit_authority_receipt_sha256,
        "fixed_control_selection_authority_receipt_sha256": rollout.fixed_control_selection_authority_receipt_sha256,
        "selected_control_id": rollout.selected_control_id,
        "selected_action_receipt_sha256": rollout.selected_action_receipt_sha256,
        "policy_trace": asdict(rollout.policy_trace),
        "action_evidence": tuple(asdict(row) for row in rollout.action_evidence),
        "action_inventory_sha256": rollout.action_inventory_sha256,
        "transition_inventory_sha256": rollout.transition_inventory_sha256,
        "environment_source_inventory_sha256": rollout.environment_source_inventory_sha256,
        "source_data_qualified": rollout.source_data_qualified,
        "outer_rollout_receipt_sha256": rollout.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
            "fixed-control outer rollout payload is not canonical JSON"
        )
    return dict(value)


def parse_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    trace = cast(Mapping[str, object], payload["policy_trace"])
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_SCHEMA,
        "fold_index": int(cast(int, payload["fold_index"])),
        "outer_plan_receipt_sha256": str(payload["outer_plan_receipt_sha256"]),
        "fixed_control_fit_authority_receipt_sha256": str(
            payload["fixed_control_fit_authority_receipt_sha256"]
        ),
        "fixed_control_selection_authority_receipt_sha256": str(
            payload["fixed_control_selection_authority_receipt_sha256"]
        ),
        "outer_rollout_receipt_sha256": str(payload["outer_rollout_receipt_sha256"]),
        "policy_trace_receipt_sha256": str(trace["semantic_receipt_sha256"]),
        "action_inventory_sha256": str(payload["action_inventory_sha256"]),
        "transition_inventory_sha256": str(payload["transition_inventory_sha256"]),
        "environment_source_inventory_sha256": str(
            payload["environment_source_inventory_sha256"]
        ),
        "source_data_qualified": bool(payload["source_data_qualified"]),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_rollout=None,
        runtime_rollout_replayed=False,
        outer_evaluation_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1,
    outer_plan: MassiveAdaptiveRLOuterPlanV1,
    registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1:
    parsed = parse_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed = _load_payload(root=root, loaded_source=authority.loaded_source)
    replayed = run_massive_adaptive_rl_fixed_control_outer_rollout_v1(
        outer_plan=outer_plan,
        registry=registry,
        fit_authority=fit_authority,
        selection_authority=selection_authority,
        chronology_authority=chronology_authority,
        environment=environment,
    )
    if canonical_json_file_bytes(committed) != canonical_json_file_bytes(
        _payload(replayed)
    ):
        raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
            "fixed-control outer trace does not replay from fit selection"
        )
    result = replace(
        parsed,
        runtime_rollout=replayed,
        runtime_rollout_replayed=True,
        outer_evaluation_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    outer_plan: MassiveAdaptiveRLOuterPlanV1,
    registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLFixedControlOuterRolloutV1Error(
            "fixed-control outer rollout artifact ID is not path safe"
        )
    rollout = run_massive_adaptive_rl_fixed_control_outer_rollout_v1(
        outer_plan=outer_plan,
        registry=registry,
        fit_authority=fit_authority,
        selection_authority=selection_authority,
        chronology_authority=chronology_authority,
        environment=environment,
    )
    relative = f"massive-adaptive/rl-fixed-control-outer-rollout-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(rollout))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_ROLLOUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=rollout.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-FIXED-CONTROL-OUTER-ROLLOUT-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1(
            root=root, loaded_source=loaded
        ),
        outer_plan=outer_plan,
        registry=registry,
        fit_authority=fit_authority,
        selection_authority=selection_authority,
        chronology_authority=chronology_authority,
        environment=environment,
    )


__all__ = [
    "MassiveAdaptiveFixedControlOuterActionEvidenceV1",
    "MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1",
    "MassiveAdaptiveRLFixedControlOuterRolloutV1",
    "MassiveAdaptiveRLFixedControlOuterRolloutV1Error",
    "authorize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1",
    "materialize_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1",
    "parse_massive_adaptive_rl_fixed_control_outer_rollout_authority_v1",
    "run_massive_adaptive_rl_fixed_control_outer_rollout_v1",
]
