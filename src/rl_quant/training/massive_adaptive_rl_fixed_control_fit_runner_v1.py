"""Package-owned prequential fitting of the protocol constant-control grid.

Every registered constant control traverses the same causal forecast blocks,
with the same source-derived execution tape and the same cross-refit economic
continuity rules as PPO.  Callers provide source-bound environment templates;
they cannot provide actions, transitions, returns, candidates, or a selected
control to the authorizing path.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
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
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import MassiveAdaptiveRLActionV1
from rl_quant.training.massive_adaptive_economic_continuity_authority_v1 import (
    MassiveAdaptiveEconomicContinuityAuthorityV1,
    build_massive_adaptive_economic_continuity_authority_v1,
)
from rl_quant.training.massive_adaptive_prequential_ppo_runner_v1 import (
    MASSIVE_ADAPTIVE_PPO_BLOCK_RUNTIME_V1_SCHEMA,
    MassiveAdaptivePPOBlockRuntimeV1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    MassiveAdaptiveRLFixedControlRegistryV1,
    build_massive_adaptive_rl_fixed_control_registry_v1,
    registered_massive_adaptive_rl_constant_actions_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlCandidateV1,
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    build_massive_adaptive_rl_fixed_control_candidate_v1,
    materialize_massive_adaptive_rl_fixed_control_selection_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_trace_from_identities_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV1,
)


MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-fit-run-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-fit-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fixed-control-fit-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "controls": "complete-protocol-owned-symmetric-constant-grid",
        "chronology": "all-RL-fit-prequential-blocks-in-authorized-order",
        "economics": "continuous-three-book-source-derived-transitions",
        "forecast_refits": "authorized-state-carry-without-liquidation",
        "selection_data": "fit-only",
        "caller_actions": False,
        "caller_transitions": False,
        "caller_returns": False,
        "duration_semantics": False,
    }
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_SCHEMA,
            "payload": "fit-run-traces-and-candidates",
            "promotion": "rerun-complete-registered-grid",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLFixedControlFitRunnerV1Error(ValueError):
    """Fixed controls were not evaluated on one complete causal fit tape."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "adaptive fixed-control fit artifact ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlFitRunV1:
    fold_index: int
    fixed_control_registry_receipt_sha256: str
    training_forecast_authority_receipt_sha256: str
    chronology_authority_receipt_sha256: str
    block_runtime_inventory_sha256: str
    continuity_authority_receipts: tuple[str, ...]
    continuity_authority_inventory_sha256: str
    control_ids: tuple[str, ...]
    action_receipts: tuple[str, ...]
    training_origin_inventory_sha256: str
    training_context_receipt_sha256: str
    traces: tuple[MassiveAdaptiveRLPolicyTraceV1, ...]
    candidates: tuple[MassiveAdaptiveRLFixedControlCandidateV1, ...]
    trace_inventory_sha256: str
    candidate_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_control_fit_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "fixed_control_registry_receipt_sha256": (
                self.fixed_control_registry_receipt_sha256
            ),
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority_receipt_sha256
            ),
            "chronology_authority_receipt_sha256": (
                self.chronology_authority_receipt_sha256
            ),
            "block_runtime_inventory_sha256": self.block_runtime_inventory_sha256,
            "continuity_authority_receipts": self.continuity_authority_receipts,
            "continuity_authority_inventory_sha256": (
                self.continuity_authority_inventory_sha256
            ),
            "control_ids": self.control_ids,
            "action_receipts": self.action_receipts,
            "training_origin_inventory_sha256": (self.training_origin_inventory_sha256),
            "training_context_receipt_sha256": self.training_context_receipt_sha256,
            "trace_inventory_sha256": self.trace_inventory_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        for trace in self.traces:
            trace.validate()
        for candidate in self.candidates:
            candidate.validate()
        expected_ids = tuple(
            control_id
            for control_id, _action in registered_massive_adaptive_rl_constant_actions_v1()
        )
        expected_actions = tuple(
            action.semantic_receipt_sha256
            for _control_id, action in registered_massive_adaptive_rl_constant_actions_v1()
        )
        expected_authorized = self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SCHEMA
            or self.fold_index < 0
            or self.control_ids != expected_ids
            or self.action_receipts != expected_actions
            or len(self.traces) != len(expected_ids)
            or len(self.candidates) != len(expected_ids)
            or tuple(row.control_id for row in self.candidates) != expected_ids
            or tuple(row.action_receipt_sha256 for row in self.candidates)
            != expected_actions
            or tuple(row.training_trace_receipt_sha256 for row in self.candidates)
            != tuple(row.semantic_receipt_sha256 for row in self.traces)
            or any(row.fold_index != self.fold_index for row in self.traces)
            or any(row.evaluation_role != "training_control" for row in self.traces)
            or any(row.transaction_cost_basis_points != 20.0 for row in self.traces)
            or any(row.frozen_targets_replayed for row in self.traces)
            or any(
                semantic_sha256(row.decision_session_dates)
                != self.training_origin_inventory_sha256
                for row in self.traces
            )
            or len({row.training_context_receipt_sha256 for row in self.candidates})
            != 1
            or self.candidates[0].training_context_receipt_sha256
            != self.training_context_receipt_sha256
            or self.trace_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.traces)
            )
            or self.candidate_inventory_sha256
            != semantic_sha256(
                tuple(sorted(row.semantic_receipt_sha256 for row in self.candidates))
            )
            or self.continuity_authority_inventory_sha256
            != semantic_sha256(self.continuity_authority_receipts)
            or self.development_control_fit_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                "adaptive fixed-control fit run differs"
            )
        for value in (
            self.fixed_control_registry_receipt_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.chronology_authority_receipt_sha256,
            self.block_runtime_inventory_sha256,
            *self.continuity_authority_receipts,
            self.continuity_authority_inventory_sha256,
            *self.action_receipts,
            self.training_origin_inventory_sha256,
            self.training_context_receipt_sha256,
            self.trace_inventory_sha256,
            self.candidate_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive fixed-control fit run", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _fresh_environment(
    template: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveProfitabilityEnvV1:
    environment = copy.copy(template)
    environment._state = None
    environment._prepared = None
    environment._observation = None
    return environment


def _bind_block_runtimes(
    *,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityV1,
    environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
) -> tuple[MassiveAdaptivePPOBlockRuntimeV1, ...]:
    grouped_dates: dict[str, list[str]] = {}
    archive_sequence: list[str] = []
    for block in training_authority.blocks:
        receipt = block.source_forecast_archive_receipt_sha256
        grouped_dates.setdefault(receipt, []).extend(block.forecast_session_dates)
        if not archive_sequence or archive_sequence[-1] != receipt:
            if receipt in archive_sequence:
                raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                    "fixed-control forecast archive recurs after a refit"
                )
            archive_sequence.append(receipt)
    if set(environments) != set(grouped_dates):
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "fixed-control environment registry has missing or extra archives"
        )
    runtimes: list[MassiveAdaptivePPOBlockRuntimeV1] = []
    cursors: dict[str, int] = {}
    for block in training_authority.blocks:
        receipt = block.source_forecast_archive_receipt_sha256
        environment = environments[receipt]
        environment.forecast_archive.validate()
        environment.calibration.validate()
        environment.inference_plan.validate()
        plan_dates = tuple(
            row.decision_session_date for row in environment.inference_plan.rows
        )
        if (
            environment.forecast_archive.semantic_receipt_sha256 != receipt
            or environment.calibration.semantic_receipt_sha256
            != block.calibration_receipt_sha256
            or tuple(grouped_dates[receipt]) != plan_dates
            or tuple(environment.forecasts) != plan_dates
            or environment.transaction_cost_basis_points != 20.0
        ):
            raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                "fixed-control block environment provenance differs"
            )
        start = cursors.get(receipt, 0)
        stop = start + len(block.forecast_session_dates)
        if plan_dates[start:stop] != block.forecast_session_dates:
            raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                "fixed-control block is not the next archive chronology"
            )
        body = {
            "schema": MASSIVE_ADAPTIVE_PPO_BLOCK_RUNTIME_V1_SCHEMA,
            "block_index": block.block_index,
            "block_receipt_sha256": block.semantic_receipt_sha256,
            "forecast_archive_receipt_sha256": receipt,
            "inference_plan_receipt_sha256": (
                environment.inference_plan.semantic_receipt_sha256
            ),
            "calibration_receipt_sha256": block.calibration_receipt_sha256,
            "environment_source_inventory_sha256": (
                environment.source_inventory_sha256
            ),
            "forecast_session_dates": block.forecast_session_dates,
            "environment_start_cursor": start,
            "environment_stop_cursor": stop,
        }
        runtime = MassiveAdaptivePPOBlockRuntimeV1(
            **body,  # type: ignore[arg-type]
            semantic_receipt_sha256=semantic_sha256(body),
        )
        runtime.validate()
        runtimes.append(runtime)
        cursors[receipt] = stop
    return tuple(runtimes)


def _bind_continuity(
    *,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityV1,
    runtimes: tuple[MassiveAdaptivePPOBlockRuntimeV1, ...],
    environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
) -> dict[int, MassiveAdaptiveEconomicContinuityAuthorityV1]:
    authorities: dict[int, MassiveAdaptiveEconomicContinuityAuthorityV1] = {}
    for next_index in range(1, len(runtimes)):
        previous = runtimes[next_index - 1]
        current = runtimes[next_index]
        if (
            previous.forecast_archive_receipt_sha256
            == current.forecast_archive_receipt_sha256
        ):
            continue
        authority = build_massive_adaptive_economic_continuity_authority_v1(
            previous_block_receipt_sha256=previous.block_receipt_sha256,
            next_block_receipt_sha256=current.block_receipt_sha256,
            previous_environment=environments[previous.forecast_archive_receipt_sha256],
            next_environment=environments[current.forecast_archive_receipt_sha256],
            source_data_qualified=training_authority.source_data_qualified,
        )
        if not authority.carry_books_authorized:
            raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                "complete fixed-control fit chronology is not economically continuous"
            )
        authorities[next_index] = authority
    return authorities


def _run_action(
    *,
    control_id: str,
    action: MassiveAdaptiveRLActionV1,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    runtimes: tuple[MassiveAdaptivePPOBlockRuntimeV1, ...],
    continuity: Mapping[int, MassiveAdaptiveEconomicContinuityAuthorityV1],
    environment_templates: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
) -> MassiveAdaptiveRLPolicyTraceV1:
    environments = {
        receipt: _fresh_environment(template)
        for receipt, template in environment_templates.items()
    }
    transitions: list[MassiveAdaptiveRLTransitionV1] = []
    current: MassiveAdaptiveProfitabilityEnvV1 | None = None
    for block_index, runtime in enumerate(runtimes):
        environment = environments[runtime.forecast_archive_receipt_sha256]
        if current is None:
            environment.reset()
        elif environment is not current:
            boundary = continuity.get(block_index)
            if boundary is None or not boundary.carry_books_authorized:
                raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                    "fixed-control refit has no continuity authority"
                )
            environment.restore_continuation(current.state)
        elif environment.state.chronology_cursor != runtime.environment_start_cursor:
            raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                "fixed-control archive cursor differs at a block boundary"
            )
        current = environment
        while current.state.chronology_cursor < runtime.environment_stop_cursor:
            last_in_runtime = (
                current.state.chronology_cursor == runtime.environment_stop_cursor - 1
            )
            next_runtime = block_index + 1
            refit_continuation = bool(
                last_in_runtime
                and next_runtime < len(runtimes)
                and next_runtime in continuity
            )
            _next, _reward, terminated, truncated, info = current.step(
                action,
                continue_economic_episode=refit_continuation,
            )
            transition = info.get("transition")
            if not isinstance(transition, MassiveAdaptiveRLTransitionV1):
                raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                    "fixed-control economic transition is absent"
                )
            final_transition = (
                block_index == len(runtimes) - 1
                and current.state.chronology_cursor == runtime.environment_stop_cursor
            )
            if (
                truncated != refit_continuation
                or terminated != final_transition
                or (terminated and truncated)
            ):
                raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                    "fixed-control fit boundary semantics differ"
                )
            transitions.append(transition)
    dates = tuple(
        row.economic_step.strategy_execution.decision_session_date
        for row in transitions
    )
    if dates != chronology_authority.rl_fit_origin_dates:
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "fixed-control fit omitted or substituted an authorized date"
        )
    if current is None:
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "fixed-control fit has no economic environment"
        )
    archives = tuple(row.forecast_archive_receipt_sha256 for row in runtimes)
    plans = tuple(row.inference_plan_receipt_sha256 for row in runtimes)
    calibrations = tuple(row.calibration_receipt_sha256 for row in runtimes)
    controller_identity = semantic_sha256(
        (
            "fixed-control-fit",
            control_id,
            action.semantic_receipt_sha256,
            training_authority.semantic_receipt_sha256,
        )
    )
    return build_massive_adaptive_rl_policy_trace_from_identities_v1(
        fold_index=chronology_authority.fold_index,
        checkpoint_receipt_sha256=controller_identity,
        model_state_receipt_sha256=action.semantic_receipt_sha256,
        update_index=0,
        training_forecast_authority_receipt_sha256=(
            training_authority.semantic_receipt_sha256
        ),
        forecast_archive_receipt_sha256=semantic_sha256(archives),
        inference_plan_receipt_sha256=semantic_sha256(plans),
        calibration_receipt_sha256=semantic_sha256(calibrations),
        transaction_cost_basis_points=20.0,
        initial_capital=current.initial_capital,
        transitions=tuple(transitions),
        frozen_targets_replayed=False,
        evaluation_role="training_control",
        checkpoint_source_data_qualified=bool(
            training_authority.reinforcement_learning_authorized
            and chronology_authority.development_rl_training_authorized
        ),
    )


def run_massive_adaptive_rl_fixed_control_fit_v1(
    *,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
    registry: MassiveAdaptiveRLFixedControlRegistryV1 | None = None,
) -> MassiveAdaptiveRLFixedControlFitRunV1:
    """Run the complete immutable constant-control grid on RL-fit data."""

    training_authority.validate()
    chronology_authority.validate()
    fixed_registry = registry or build_massive_adaptive_rl_fixed_control_registry_v1()
    fixed_registry.validate()
    if (
        not training_authority.reinforcement_learning_authorized
        or not chronology_authority.development_rl_training_authorized
        or chronology_authority.training_forecast_authority_receipt_sha256
        != training_authority.semantic_receipt_sha256
        or chronology_authority.rl_fit_origin_dates
        != training_authority.origin_session_dates
    ):
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "fixed-control fit chronology differs from PPO training"
        )
    runtimes = _bind_block_runtimes(
        training_authority=training_authority,
        environments=environments,
    )
    continuity = _bind_continuity(
        training_authority=training_authority,
        runtimes=runtimes,
        environments=environments,
    )
    registered = registered_massive_adaptive_rl_constant_actions_v1()
    training_context_receipt = semantic_sha256(
        (
            training_authority.semantic_receipt_sha256,
            chronology_authority.semantic_receipt_sha256,
            tuple(row.semantic_receipt_sha256 for row in runtimes),
            tuple(row.semantic_receipt_sha256 for row in continuity.values()),
            tuple(
                (
                    receipt,
                    environment.source_inventory_sha256,
                    environment.economic_compatibility_receipt_sha256,
                )
                for receipt, environment in sorted(environments.items())
            ),
        )
    )
    traces = tuple(
        _run_action(
            control_id=control_id,
            action=action,
            training_authority=training_authority,
            chronology_authority=chronology_authority,
            runtimes=runtimes,
            continuity=continuity,
            environment_templates=environments,
        )
        for control_id, action in registered
    )
    candidates = tuple(
        build_massive_adaptive_rl_fixed_control_candidate_v1(
            fold_index=chronology_authority.fold_index,
            control_id=control_id,
            action=action,
            training_trace=trace,
            training_context_receipt_sha256=training_context_receipt,
        )
        for (control_id, action), trace in zip(registered, traces, strict=True)
    )
    source_qualified = bool(
        training_authority.source_data_qualified
        and chronology_authority.source_data_qualified
        and all(row.source_data_qualified for row in traces)
    )
    control_ids = tuple(control_id for control_id, _action in registered)
    action_receipts = tuple(
        action.semantic_receipt_sha256 for _control_id, action in registered
    )
    continuity_receipts = tuple(
        row.semantic_receipt_sha256 for row in continuity.values()
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SCHEMA,
        "fold_index": chronology_authority.fold_index,
        "fixed_control_registry_receipt_sha256": (
            fixed_registry.semantic_receipt_sha256
        ),
        "training_forecast_authority_receipt_sha256": (
            training_authority.semantic_receipt_sha256
        ),
        "chronology_authority_receipt_sha256": (
            chronology_authority.semantic_receipt_sha256
        ),
        "block_runtime_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in runtimes)
        ),
        "continuity_authority_receipts": continuity_receipts,
        "continuity_authority_inventory_sha256": semantic_sha256(continuity_receipts),
        "control_ids": control_ids,
        "action_receipts": action_receipts,
        "training_origin_inventory_sha256": semantic_sha256(
            chronology_authority.rl_fit_origin_dates
        ),
        "training_context_receipt_sha256": training_context_receipt,
        "traces": traces,
        "candidates": candidates,
        "trace_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in traces)
        ),
        "candidate_inventory_sha256": semantic_sha256(
            tuple(sorted(row.semantic_receipt_sha256 for row in candidates))
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_RUN_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFixedControlFitRunV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        development_control_fit_authorized=source_qualified,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlFitAuthorityV1:
    fold_index: int
    fixed_control_registry_receipt_sha256: str
    training_forecast_authority_receipt_sha256: str
    chronology_authority_receipt_sha256: str
    fit_run_receipt_sha256: str
    trace_inventory_sha256: str
    candidate_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_fit_run: MassiveAdaptiveRLFixedControlFitRunV1 | None
    runtime_fit_replayed: bool
    development_control_fit_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "fixed_control_registry_receipt_sha256": (
                self.fixed_control_registry_receipt_sha256
            ),
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority_receipt_sha256
            ),
            "chronology_authority_receipt_sha256": (
                self.chronology_authority_receipt_sha256
            ),
            "fit_run_receipt_sha256": self.fit_run_receipt_sha256,
            "trace_inventory_sha256": self.trace_inventory_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_fit_run is not None
        if self.runtime_fit_run is not None:
            self.runtime_fit_run.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.fit_run_receipt_sha256
            or self.runtime_fit_replayed != runtime
            or self.development_control_fit_authorized
            != (runtime and self.source_data_qualified)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                "adaptive fixed-control fit authority differs"
            )
        if (
            runtime
            and self.runtime_fit_run is not None
            and (
                self.runtime_fit_run.fold_index != self.fold_index
                or self.runtime_fit_run.fixed_control_registry_receipt_sha256
                != self.fixed_control_registry_receipt_sha256
                or self.runtime_fit_run.training_forecast_authority_receipt_sha256
                != self.training_forecast_authority_receipt_sha256
                or self.runtime_fit_run.chronology_authority_receipt_sha256
                != self.chronology_authority_receipt_sha256
                or self.runtime_fit_run.semantic_receipt_sha256
                != self.fit_run_receipt_sha256
                or self.runtime_fit_run.trace_inventory_sha256
                != self.trace_inventory_sha256
                or self.runtime_fit_run.candidate_inventory_sha256
                != self.candidate_inventory_sha256
            )
        ):
            raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
                "runtime fixed-control fit differs from its authority"
            )
        for value in (
            self.fixed_control_registry_receipt_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.chronology_authority_receipt_sha256,
            self.fit_run_receipt_sha256,
            self.trace_inventory_sha256,
            self.candidate_inventory_sha256,
            self.protocol_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive fixed-control fit authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _payload(run: MassiveAdaptiveRLFixedControlFitRunV1) -> dict[str, object]:
    return {
        **run.semantic_unsigned(),
        "traces": tuple(asdict(row) for row in run.traces),
        "candidates": tuple(asdict(row) for row in run.candidates),
        "fit_run_receipt_sha256": run.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "adaptive fixed-control fit payload is not canonical JSON"
        )
    return dict(cast(Mapping[str, object], value))


def parse_massive_adaptive_rl_fixed_control_fit_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFixedControlFitAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_SCHEMA,
        "fold_index": int(cast(int, payload["fold_index"])),
        "fixed_control_registry_receipt_sha256": str(
            payload["fixed_control_registry_receipt_sha256"]
        ),
        "training_forecast_authority_receipt_sha256": str(
            payload["training_forecast_authority_receipt_sha256"]
        ),
        "chronology_authority_receipt_sha256": str(
            payload["chronology_authority_receipt_sha256"]
        ),
        "fit_run_receipt_sha256": str(payload["fit_run_receipt_sha256"]),
        "trace_inventory_sha256": str(payload["trace_inventory_sha256"]),
        "candidate_inventory_sha256": str(payload["candidate_inventory_sha256"]),
        "source_data_qualified": bool(payload["source_data_qualified"]),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveRLFixedControlFitAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_fit_run=None,
        runtime_fit_replayed=False,
        development_control_fit_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_fixed_control_fit_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
    registry: MassiveAdaptiveRLFixedControlRegistryV1 | None = None,
) -> MassiveAdaptiveRLFixedControlFitAuthorityV1:
    parsed = parse_massive_adaptive_rl_fixed_control_fit_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed = _load_payload(root=root, loaded_source=authority.loaded_source)
    replayed = run_massive_adaptive_rl_fixed_control_fit_v1(
        training_authority=training_authority,
        chronology_authority=chronology_authority,
        environments=environments,
        registry=registry,
    )
    if canonical_json_file_bytes(committed) != canonical_json_file_bytes(
        _payload(replayed)
    ):
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "adaptive fixed-control fit does not replay"
        )
    result = replace(
        parsed,
        runtime_fit_run=replayed,
        runtime_fit_replayed=True,
        development_control_fit_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_fixed_control_fit_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
    committed_at_ms: int,
    registry: MassiveAdaptiveRLFixedControlRegistryV1 | None = None,
) -> MassiveAdaptiveRLFixedControlFitAuthorityV1:
    identifier = _artifact_id(artifact_id)
    run = run_massive_adaptive_rl_fixed_control_fit_v1(
        training_authority=training_authority,
        chronology_authority=chronology_authority,
        environments=environments,
        registry=registry,
    )
    relative = f"massive-adaptive/rl-fixed-control-fit-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(run))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=run.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-FIXED-CONTROL-FIT-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_fixed_control_fit_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_fixed_control_fit_authority_v1(
            root=root, loaded_source=loaded
        ),
        training_authority=training_authority,
        chronology_authority=chronology_authority,
        environments=environments,
        registry=registry,
    )


def materialize_massive_adaptive_rl_fixed_control_selection_from_fit_v1(
    *,
    root: str | Path,
    artifact_id: str,
    fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFixedControlSelectionAuthorityV1:
    """Select FC06 only from candidates replayed by the fit authority."""

    fit_authority.validate()
    fit_run = fit_authority.runtime_fit_run
    if fit_run is None or not fit_authority.runtime_fit_replayed:
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "fixed-control fit must replay before FC06 selection"
        )
    selection = materialize_massive_adaptive_rl_fixed_control_selection_authority_v1(
        root=root,
        artifact_id=artifact_id,
        candidates=fit_run.candidates,
        committed_at_ms=committed_at_ms,
    )
    runtime = selection.runtime_selection
    if (
        runtime is None
        or not selection.runtime_selection_replayed
        or runtime.candidate_inventory_sha256 != fit_run.candidate_inventory_sha256
    ):
        raise MassiveAdaptiveRLFixedControlFitRunnerV1Error(
            "FC06 selection differs from the replayed fixed-control fit"
        )
    return selection


__all__ = [
    "MassiveAdaptiveRLFixedControlFitAuthorityV1",
    "MassiveAdaptiveRLFixedControlFitRunV1",
    "MassiveAdaptiveRLFixedControlFitRunnerV1Error",
    "authorize_massive_adaptive_rl_fixed_control_fit_authority_v1",
    "materialize_massive_adaptive_rl_fixed_control_fit_authority_v1",
    "materialize_massive_adaptive_rl_fixed_control_selection_from_fit_v1",
    "parse_massive_adaptive_rl_fixed_control_fit_authority_v1",
    "run_massive_adaptive_rl_fixed_control_fit_v1",
]
