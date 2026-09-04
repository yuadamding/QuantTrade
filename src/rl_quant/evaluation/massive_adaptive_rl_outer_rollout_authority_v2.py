"""Manifest-V5 outer replay from committed frozen PPO and FC06 artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import cast

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v2 import (
    MassiveAdaptiveOuterAccessCommitmentV2,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_v1 import (
    replay_massive_adaptive_rl_frozen_target_transitions_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_v1 import (
    MASSIVE_ADAPTIVE_FROZEN_RL_ACTION_EVIDENCE_V1_SCHEMA,
    MassiveAdaptiveFrozenRLActionEvidenceV1,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_evaluator_v1 import (
    _policy_from_state,
    _tensor_receipt,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptiveBoundedControlDistributionV1,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    build_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_trace_from_identities_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_DATASET = (
    "massive-adaptive-rl-outer-rollout-authority-v2"
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SCHEMA,
        "payload": "frozen-ppo-fc06-benchmark-and-fixed-target-cost-ladder",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLOuterRolloutAuthorityV2Error(ValueError):
    """Frozen inference, shared economics, or replay evidence differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            f"{name} is absent or invalid"
        )
    return value


def _terminal_adjusted_rows(
    transitions: Sequence[MassiveAdaptiveRLTransitionV1], *, book: str
) -> tuple[float, ...]:
    rows = tuple(transitions)
    if not rows or any(row.terminated for row in rows[:-1]) or not rows[-1].terminated:
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer rollout is not one complete episode"
        )
    if book == "strategy":
        values = [float(row.economic_step.strategy_net_log_return) for row in rows]
        marked = rows[-1].economic_step.strategy_posttrade_book.marked_equity
        liquidated = rows[-1].strategy_liquidation_adjusted_equity
    elif book == "neutral":
        values = [float(row.economic_step.neutral_net_log_return) for row in rows]
        marked = rows[-1].economic_step.neutral_posttrade_book.marked_equity
        liquidated = rows[-1].neutral_liquidation_adjusted_equity
    elif book == "benchmark":
        values = [float(row.economic_step.benchmark_net_log_return) for row in rows]
        marked = rows[-1].economic_step.benchmark_posttrade_book.marked_equity
        liquidated = rows[-1].benchmark_liquidation_adjusted_equity
    else:
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer economic book differs"
        )
    if marked <= 0.0 or liquidated <= 0.0:
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer terminal equity is invalid"
        )
    values[-1] += math.log(liquidated / marked)
    return tuple(values)


def _decision_target_inventory(
    transitions: Sequence[MassiveAdaptiveRLTransitionV1],
) -> str:
    return semantic_sha256(
        tuple(
            (
                row.policy_decision.security_ids,
                row.policy_decision.target_weights,
            )
            for row in transitions
        )
    )


def _trace(
    *,
    access: MassiveAdaptiveOuterAccessCommitmentV2,
    transitions: Sequence[MassiveAdaptiveRLTransitionV1],
    environment: MassiveAdaptiveProfitabilityEnvV1,
    checkpoint_receipt: str,
    model_state_receipt: str,
    update_index: int,
    training_receipt: str,
    frozen_targets_replayed: bool,
    source_data_qualified: bool,
) -> MassiveAdaptiveRLPolicyTraceV1:
    return build_massive_adaptive_rl_policy_trace_from_identities_v1(
        fold_index=access.fold_index,
        checkpoint_receipt_sha256=checkpoint_receipt,
        model_state_receipt_sha256=model_state_receipt,
        update_index=update_index,
        training_forecast_authority_receipt_sha256=training_receipt,
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
        frozen_targets_replayed=frozen_targets_replayed,
        evaluation_role="outer_test",
        checkpoint_source_data_qualified=source_data_qualified,
    )


@dataclass(frozen=True, slots=True)
class _FrozenControlReplayEvidenceV2:
    decision_session_date: str
    observation_receipt_sha256: str
    action_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterRolloutComputationV2:
    fold_index: int
    outer_access_commitment_receipt_sha256: str
    frozen_ppo_policy_receipt_sha256: str
    frozen_fc06_control_receipt_sha256: str
    decision_session_dates: tuple[str, ...]
    ppo_action_evidence_receipts: tuple[str, ...]
    ppo_action_inventory_sha256: str
    ppo_primary_trace_receipt_sha256: str
    ppo_low_cost_trace_receipt_sha256: str
    ppo_high_cost_trace_receipt_sha256: str
    fixed_control_trace_receipt_sha256: str
    fixed_control_low_cost_trace_receipt_sha256: str
    fixed_control_high_cost_trace_receipt_sha256: str
    ppo_primary_transition_inventory_sha256: str
    ppo_low_cost_transition_inventory_sha256: str
    ppo_high_cost_transition_inventory_sha256: str
    fixed_control_transition_inventory_sha256: str
    fixed_control_low_cost_transition_inventory_sha256: str
    fixed_control_high_cost_transition_inventory_sha256: str
    decision_target_inventory_sha256: str
    fixed_control_decision_target_inventory_sha256: str
    strategy_net_log_returns: tuple[float, ...]
    neutral_net_log_returns: tuple[float, ...]
    benchmark_net_log_returns: tuple[float, ...]
    fixed_control_net_log_returns: tuple[float, ...]
    active_log_returns: tuple[float, ...]
    incremental_rl_log_returns: tuple[float, ...]
    ppo_minus_fixed_control_log_returns: tuple[float, ...]
    primary_terminal_liquidation_adjusted_return: float
    low_cost_terminal_liquidation_adjusted_return: float
    high_cost_terminal_liquidation_adjusted_return: float
    fixed_control_terminal_liquidation_adjusted_return: float
    fixed_control_low_cost_terminal_liquidation_adjusted_return: float
    fixed_control_high_cost_terminal_liquidation_adjusted_return: float
    maximum_drawdown: float
    environment_source_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = "rl-quant.massive-adaptive-rl-outer-rollout-computation-v2"
    _ppo_primary_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...] = field(
        default=(), compare=False, repr=False
    )
    _ppo_low_cost_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...] = field(
        default=(), compare=False, repr=False
    )
    _ppo_high_cost_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...] = field(
        default=(), compare=False, repr=False
    )
    _fixed_control_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...] = field(
        default=(), compare=False, repr=False
    )
    _fixed_control_low_cost_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...] = (
        field(default=(), compare=False, repr=False)
    )
    _fixed_control_high_cost_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...] = (
        field(default=(), compare=False, repr=False)
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name != "semantic_receipt_sha256"
        }

    @property
    def runtime_transitions_replayed(self) -> bool:
        count = len(self.decision_session_dates)
        return bool(
            count
            and all(
                len(rows) == count
                for rows in (
                    self._ppo_primary_transitions,
                    self._ppo_low_cost_transitions,
                    self._ppo_high_cost_transitions,
                    self._fixed_control_transitions,
                    self._fixed_control_low_cost_transitions,
                    self._fixed_control_high_cost_transitions,
                )
            )
        )

    @property
    def ppo_primary_transitions(self) -> tuple[MassiveAdaptiveRLTransitionV1, ...]:
        self.validate()
        if not self.runtime_transitions_replayed:
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer PPO transitions have not been replayed"
            )
        return self._ppo_primary_transitions

    def validate(self) -> None:
        count = len(self.decision_session_dates)
        series = (
            self.strategy_net_log_returns,
            self.neutral_net_log_returns,
            self.benchmark_net_log_returns,
            self.fixed_control_net_log_returns,
            self.active_log_returns,
            self.incremental_rl_log_returns,
            self.ppo_minus_fixed_control_log_returns,
        )
        numbers = (
            *(value for row in series for value in row),
            self.primary_terminal_liquidation_adjusted_return,
            self.low_cost_terminal_liquidation_adjusted_return,
            self.high_cost_terminal_liquidation_adjusted_return,
            self.fixed_control_terminal_liquidation_adjusted_return,
            self.fixed_control_low_cost_terminal_liquidation_adjusted_return,
            self.fixed_control_high_cost_terminal_liquidation_adjusted_return,
            self.maximum_drawdown,
        )
        if (
            self.schema != "rl-quant.massive-adaptive-rl-outer-rollout-computation-v2"
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or count != 126
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or len(self.ppo_action_evidence_receipts) != count
            or any(len(row) != count for row in series)
            or any(not math.isfinite(value) for value in numbers)
            or min(
                self.primary_terminal_liquidation_adjusted_return,
                self.low_cost_terminal_liquidation_adjusted_return,
                self.high_cost_terminal_liquidation_adjusted_return,
                self.fixed_control_terminal_liquidation_adjusted_return,
                self.fixed_control_low_cost_terminal_liquidation_adjusted_return,
                self.fixed_control_high_cost_terminal_liquidation_adjusted_return,
            )
            <= -1.0
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or not (
                self.low_cost_terminal_liquidation_adjusted_return
                >= self.primary_terminal_liquidation_adjusted_return
                >= self.high_cost_terminal_liquidation_adjusted_return
            )
            or not (
                self.fixed_control_low_cost_terminal_liquidation_adjusted_return
                >= self.fixed_control_terminal_liquidation_adjusted_return
                >= self.fixed_control_high_cost_terminal_liquidation_adjusted_return
            )
            or self.active_log_returns
            != tuple(
                left - right
                for left, right in zip(
                    self.strategy_net_log_returns,
                    self.benchmark_net_log_returns,
                    strict=True,
                )
            )
            or self.incremental_rl_log_returns
            != tuple(
                left - right
                for left, right in zip(
                    self.strategy_net_log_returns,
                    self.neutral_net_log_returns,
                    strict=True,
                )
            )
            or self.ppo_minus_fixed_control_log_returns
            != tuple(
                left - right
                for left, right in zip(
                    self.strategy_net_log_returns,
                    self.fixed_control_net_log_returns,
                    strict=True,
                )
            )
            or self.ppo_action_inventory_sha256
            != semantic_sha256(self.ppo_action_evidence_receipts)
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
        ):
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer rollout computation differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        for value in self.ppo_action_evidence_receipts:
            _digest("PPO action evidence", value)
        runtime_rows = (
            self._ppo_primary_transitions,
            self._ppo_low_cost_transitions,
            self._ppo_high_cost_transitions,
            self._fixed_control_transitions,
            self._fixed_control_low_cost_transitions,
            self._fixed_control_high_cost_transitions,
        )
        if any(runtime_rows) and not self.runtime_transitions_replayed:
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer transition replay is partial"
            )
        if self.runtime_transitions_replayed:
            for rows in runtime_rows:
                for row in rows:
                    row.validate()
            transition_inventories = tuple(
                semantic_sha256(tuple(row.semantic_receipt_sha256 for row in rows))
                for rows in runtime_rows
            )
            if (
                transition_inventories
                != (
                    self.ppo_primary_transition_inventory_sha256,
                    self.ppo_low_cost_transition_inventory_sha256,
                    self.ppo_high_cost_transition_inventory_sha256,
                    self.fixed_control_transition_inventory_sha256,
                    self.fixed_control_low_cost_transition_inventory_sha256,
                    self.fixed_control_high_cost_transition_inventory_sha256,
                )
                or any(
                    tuple(
                        row.economic_step.strategy_execution.decision_session_date
                        for row in rows
                    )
                    != self.decision_session_dates
                    for rows in runtime_rows
                )
                or len({_decision_target_inventory(rows) for rows in runtime_rows[:3]})
                != 1
                or _decision_target_inventory(self._ppo_primary_transitions)
                != self.decision_target_inventory_sha256
                or len({_decision_target_inventory(rows) for rows in runtime_rows[3:]})
                != 1
                or _decision_target_inventory(self._fixed_control_transitions)
                != self.fixed_control_decision_target_inventory_sha256
            ):
                raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                    "outer transition inventory differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def execute_massive_adaptive_rl_outer_rollout_v2(
    *,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
) -> MassiveAdaptiveRLOuterRolloutComputationV2:
    """Replay only artifacts exposed by the exact outer-access commitment."""

    if type(outer_access) is not MassiveAdaptiveOuterAccessCommitmentV2:
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer rollout requires an exact V2 access commitment"
        )
    outer_access.validate()
    if not outer_access.outer_input_access_authorized:
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer access has not been exactly replayed"
        )
    frozen_policy = outer_access.frozen_policy
    frozen_control = outer_access.frozen_control
    bundle = outer_access.runtime_environment_bundle
    primary_environment = bundle.primary_environment
    device = torch.device("cpu")
    model = _policy_from_state(frozen_policy.runtime_model_state, device=device)
    observation, _ = primary_environment.reset()
    evidence: list[MassiveAdaptiveFrozenRLActionEvidenceV1] = []
    primary: list[MassiveAdaptiveRLTransitionV1] = []
    with torch.inference_mode():
        while True:
            tensor = torch.tensor(
                observation.values, dtype=torch.float32, device=device
            ).unsqueeze(0)
            output = model({"adaptive_state": tensor})
            distribution = output.distribution
            if not isinstance(
                distribution, MassiveAdaptiveBoundedControlDistributionV1
            ):
                raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                    "frozen actor emitted an unregistered distribution"
                )
            values = tuple(
                float(value)
                for value in distribution.deterministic_action()[0].cpu().tolist()
            )
            action = build_massive_adaptive_rl_action_v1(
                bucket_controls=values[:7],
                uncertainty_control=values[7],
                risk_control=values[8],
                trade_cost_control=values[9],
            )
            decision_date = primary_environment.inference_plan.rows[
                primary_environment.state.chronology_cursor
            ].decision_session_date
            evidence_body = {
                "schema": MASSIVE_ADAPTIVE_FROZEN_RL_ACTION_EVIDENCE_V1_SCHEMA,
                "decision_session_date": decision_date,
                "observation_receipt_sha256": observation.semantic_receipt_sha256,
                "frozen_policy_receipt_sha256": frozen_policy.semantic_receipt_sha256,
                "selected_checkpoint_receipt_sha256": (
                    frozen_policy.selected_checkpoint_receipt_sha256
                ),
                "frozen_model_state_receipt_sha256": (
                    frozen_policy.frozen_model_state_receipt_sha256
                ),
                "distribution_parameter_receipt_sha256": semantic_sha256(
                    (
                        _tensor_receipt(distribution.mean),
                        _tensor_receipt(distribution.log_std),
                    )
                ),
                "action_values": values,
                "action_receipt_sha256": action.semantic_receipt_sha256,
                "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
            }
            evidence_row = MassiveAdaptiveFrozenRLActionEvidenceV1(
                **evidence_body,  # type: ignore[arg-type]
                semantic_receipt_sha256=semantic_sha256(evidence_body),
            )
            evidence_row.validate()
            evidence.append(evidence_row)
            next_observation, _reward, terminated, truncated, info = (
                primary_environment.step(action)
            )
            transition = info.get("transition")
            if truncated or not isinstance(transition, MassiveAdaptiveRLTransitionV1):
                raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                    "outer PPO transition differs"
                )
            primary.append(transition)
            if terminated:
                break
            if next_observation is None:
                raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                    "outer PPO observation is absent"
                )
            observation = next_observation

    primary_rows = tuple(primary)
    low_rows = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=evidence,
        primary_transitions=primary_rows,
        environment=bundle.low_cost_environment,
    )
    high_rows = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=evidence,
        primary_transitions=primary_rows,
        environment=bundle.high_cost_environment,
    )
    fixed_environment = bundle.fixed_control_environment
    fixed_action = frozen_control.runtime_action
    fixed_environment.reset()
    fixed_rows: list[MassiveAdaptiveRLTransitionV1] = []
    while True:
        next_observation, _reward, terminated, truncated, info = fixed_environment.step(
            fixed_action
        )
        transition = info.get("transition")
        if truncated or not isinstance(transition, MassiveAdaptiveRLTransitionV1):
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer fixed-control transition differs"
            )
        fixed_rows.append(transition)
        if terminated:
            break
        if next_observation is None:
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer fixed-control observation is absent"
            )
    fixed = tuple(fixed_rows)
    fixed_action_values = (
        *fixed_action.bucket_controls,
        fixed_action.uncertainty_control,
        fixed_action.risk_control,
        fixed_action.trade_cost_control,
    )
    fixed_replay_evidence = tuple(
        _FrozenControlReplayEvidenceV2(
            decision_session_date=(
                row.economic_step.strategy_execution.decision_session_date
            ),
            observation_receipt_sha256=row.observation_receipt_sha256,
            action_values=fixed_action_values,
        )
        for row in fixed
    )
    fixed_low_rows = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=fixed_replay_evidence,
        primary_transitions=fixed,
        environment=bundle.low_cost_environment,
    )
    fixed_high_rows = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=fixed_replay_evidence,
        primary_transitions=fixed,
        environment=bundle.high_cost_environment,
    )
    source_qualified = bool(
        outer_access.source_data_qualified
        and bundle.source_data_qualified
        and frozen_policy.development_stage_authorized
        and frozen_control.development_stage_authorized
        and all(
            row.source_data_qualified
            for row in (
                *primary_rows,
                *low_rows,
                *high_rows,
                *fixed,
                *fixed_low_rows,
                *fixed_high_rows,
            )
        )
    )
    primary_trace = _trace(
        access=outer_access,
        transitions=primary_rows,
        environment=primary_environment,
        checkpoint_receipt=frozen_policy.selected_checkpoint_receipt_sha256,
        model_state_receipt=frozen_policy.frozen_model_state_receipt_sha256,
        update_index=frozen_policy.selected_update_index,
        training_receipt=frozen_policy.training_forecast_authority_receipt_sha256,
        frozen_targets_replayed=False,
        source_data_qualified=source_qualified,
    )
    low_trace = _trace(
        access=outer_access,
        transitions=low_rows,
        environment=bundle.low_cost_environment,
        checkpoint_receipt=frozen_policy.selected_checkpoint_receipt_sha256,
        model_state_receipt=frozen_policy.frozen_model_state_receipt_sha256,
        update_index=frozen_policy.selected_update_index,
        training_receipt=frozen_policy.training_forecast_authority_receipt_sha256,
        frozen_targets_replayed=True,
        source_data_qualified=source_qualified,
    )
    high_trace = _trace(
        access=outer_access,
        transitions=high_rows,
        environment=bundle.high_cost_environment,
        checkpoint_receipt=frozen_policy.selected_checkpoint_receipt_sha256,
        model_state_receipt=frozen_policy.frozen_model_state_receipt_sha256,
        update_index=frozen_policy.selected_update_index,
        training_receipt=frozen_policy.training_forecast_authority_receipt_sha256,
        frozen_targets_replayed=True,
        source_data_qualified=source_qualified,
    )
    fixed_trace = _trace(
        access=outer_access,
        transitions=fixed,
        environment=fixed_environment,
        checkpoint_receipt=frozen_control.semantic_receipt_sha256,
        model_state_receipt=frozen_control.selected_action_receipt_sha256,
        update_index=0,
        training_receipt=frozen_control.fixed_control_fit_authority_receipt_sha256,
        frozen_targets_replayed=False,
        source_data_qualified=source_qualified,
    )
    fixed_low_trace = _trace(
        access=outer_access,
        transitions=fixed_low_rows,
        environment=bundle.low_cost_environment,
        checkpoint_receipt=frozen_control.semantic_receipt_sha256,
        model_state_receipt=frozen_control.selected_action_receipt_sha256,
        update_index=0,
        training_receipt=frozen_control.fixed_control_fit_authority_receipt_sha256,
        frozen_targets_replayed=True,
        source_data_qualified=source_qualified,
    )
    fixed_high_trace = _trace(
        access=outer_access,
        transitions=fixed_high_rows,
        environment=bundle.high_cost_environment,
        checkpoint_receipt=frozen_control.semantic_receipt_sha256,
        model_state_receipt=frozen_control.selected_action_receipt_sha256,
        update_index=0,
        training_receipt=frozen_control.fixed_control_fit_authority_receipt_sha256,
        frozen_targets_replayed=True,
        source_data_qualified=source_qualified,
    )
    strategy = _terminal_adjusted_rows(primary_rows, book="strategy")
    neutral = _terminal_adjusted_rows(primary_rows, book="neutral")
    benchmark = _terminal_adjusted_rows(primary_rows, book="benchmark")
    fixed_series = _terminal_adjusted_rows(fixed, book="strategy")
    body = {
        "fold_index": outer_access.fold_index,
        "outer_access_commitment_receipt_sha256": outer_access.semantic_receipt_sha256,
        "frozen_ppo_policy_receipt_sha256": frozen_policy.semantic_receipt_sha256,
        "frozen_fc06_control_receipt_sha256": frozen_control.semantic_receipt_sha256,
        "decision_session_dates": bundle.decision_session_dates,
        "ppo_action_evidence_receipts": tuple(
            row.semantic_receipt_sha256 for row in evidence
        ),
        "ppo_action_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in evidence)
        ),
        "ppo_primary_trace_receipt_sha256": primary_trace.semantic_receipt_sha256,
        "ppo_low_cost_trace_receipt_sha256": low_trace.semantic_receipt_sha256,
        "ppo_high_cost_trace_receipt_sha256": high_trace.semantic_receipt_sha256,
        "fixed_control_trace_receipt_sha256": fixed_trace.semantic_receipt_sha256,
        "fixed_control_low_cost_trace_receipt_sha256": (
            fixed_low_trace.semantic_receipt_sha256
        ),
        "fixed_control_high_cost_trace_receipt_sha256": (
            fixed_high_trace.semantic_receipt_sha256
        ),
        "ppo_primary_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in primary_rows)
        ),
        "ppo_low_cost_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in low_rows)
        ),
        "ppo_high_cost_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in high_rows)
        ),
        "fixed_control_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in fixed)
        ),
        "fixed_control_low_cost_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in fixed_low_rows)
        ),
        "fixed_control_high_cost_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in fixed_high_rows)
        ),
        "decision_target_inventory_sha256": primary_trace.decision_target_inventory_sha256,
        "fixed_control_decision_target_inventory_sha256": (
            fixed_trace.decision_target_inventory_sha256
        ),
        "strategy_net_log_returns": strategy,
        "neutral_net_log_returns": neutral,
        "benchmark_net_log_returns": benchmark,
        "fixed_control_net_log_returns": fixed_series,
        "active_log_returns": tuple(
            left - right for left, right in zip(strategy, benchmark, strict=True)
        ),
        "incremental_rl_log_returns": tuple(
            left - right for left, right in zip(strategy, neutral, strict=True)
        ),
        "ppo_minus_fixed_control_log_returns": tuple(
            left - right for left, right in zip(strategy, fixed_series, strict=True)
        ),
        "primary_terminal_liquidation_adjusted_return": (
            primary_trace.terminal_liquidation_adjusted_return
        ),
        "low_cost_terminal_liquidation_adjusted_return": (
            low_trace.terminal_liquidation_adjusted_return
        ),
        "high_cost_terminal_liquidation_adjusted_return": (
            high_trace.terminal_liquidation_adjusted_return
        ),
        "fixed_control_terminal_liquidation_adjusted_return": (
            fixed_trace.terminal_liquidation_adjusted_return
        ),
        "fixed_control_low_cost_terminal_liquidation_adjusted_return": (
            fixed_low_trace.terminal_liquidation_adjusted_return
        ),
        "fixed_control_high_cost_terminal_liquidation_adjusted_return": (
            fixed_high_trace.terminal_liquidation_adjusted_return
        ),
        "maximum_drawdown": primary_trace.maximum_drawdown,
        "environment_source_inventory_sha256": bundle.environment_source_inventory_sha256,
        "source_data_qualified": source_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "schema": "rl-quant.massive-adaptive-rl-outer-rollout-computation-v2",
    }
    provisional = MassiveAdaptiveRLOuterRolloutComputationV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        _ppo_primary_transitions=primary_rows,
        _ppo_low_cost_transitions=low_rows,
        _ppo_high_cost_transitions=high_rows,
        _fixed_control_transitions=fixed,
        _fixed_control_low_cost_transitions=fixed_low_rows,
        _fixed_control_high_cost_transitions=fixed_high_rows,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterRolloutAuthorityV2:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    fold_index: int
    outer_access_commitment_receipt_sha256: str
    outer_access_source_receipt_sha256: str
    outer_access_commit_receipt_sha256: str
    outer_access_committed_at_ms: int
    outer_rollout_receipt_sha256: str
    decision_session_dates: tuple[str, ...]
    ppo_action_inventory_sha256: str
    ppo_primary_trace_receipt_sha256: str
    ppo_low_cost_trace_receipt_sha256: str
    ppo_high_cost_trace_receipt_sha256: str
    fixed_control_trace_receipt_sha256: str
    fixed_control_low_cost_trace_receipt_sha256: str
    fixed_control_high_cost_trace_receipt_sha256: str
    ppo_primary_transition_inventory_sha256: str
    ppo_low_cost_transition_inventory_sha256: str
    ppo_high_cost_transition_inventory_sha256: str
    fixed_control_transition_inventory_sha256: str
    fixed_control_low_cost_transition_inventory_sha256: str
    fixed_control_high_cost_transition_inventory_sha256: str
    decision_target_inventory_sha256: str
    fixed_control_decision_target_inventory_sha256: str
    primary_terminal_liquidation_adjusted_return: float
    low_cost_terminal_liquidation_adjusted_return: float
    high_cost_terminal_liquidation_adjusted_return: float
    fixed_control_terminal_liquidation_adjusted_return: float
    fixed_control_low_cost_terminal_liquidation_adjusted_return: float
    fixed_control_high_cost_terminal_liquidation_adjusted_return: float
    maximum_drawdown: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_rollout_replayed: bool = False
    outer_evaluation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    positive_profitability_authorization_eligible: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SCHEMA
    _runtime_outer_access: MassiveAdaptiveOuterAccessCommitmentV2 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_rollout: MassiveAdaptiveRLOuterRolloutComputationV2 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {
                "semantic_receipt_sha256",
                "runtime_rollout_replayed",
                "outer_evaluation_authorized",
                "positive_profitability_authorization_eligible",
            }
        }

    @property
    def source_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.receipt.receipt_sha256
        )

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.receipt_sha256
        )

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.committed_at_ms
        )

    @property
    def rollout(self) -> MassiveAdaptiveRLOuterRolloutComputationV2:
        self.validate()
        if self._runtime_rollout is None or not self.outer_evaluation_authorized:
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer rollout has not been exactly replayed"
            )
        return self._runtime_rollout

    def validate(self) -> None:
        runtime = (
            self._runtime_outer_access is not None and self._runtime_rollout is not None
        )
        any_runtime = (
            self._runtime_outer_access is not None or self._runtime_rollout is not None
        )
        _required_time("outer access time", self.outer_access_committed_at_ms)
        economic_values = (
            self.primary_terminal_liquidation_adjusted_return,
            self.low_cost_terminal_liquidation_adjusted_return,
            self.high_cost_terminal_liquidation_adjusted_return,
            self.fixed_control_terminal_liquidation_adjusted_return,
            self.fixed_control_low_cost_terminal_liquidation_adjusted_return,
            self.fixed_control_high_cost_terminal_liquidation_adjusted_return,
            self.maximum_drawdown,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or len(self.decision_session_dates) != 126
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or any(not math.isfinite(value) for value in economic_values)
            or min(economic_values[:-1]) <= -1.0
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or not (
                self.low_cost_terminal_liquidation_adjusted_return
                >= self.primary_terminal_liquidation_adjusted_return
                >= self.high_cost_terminal_liquidation_adjusted_return
            )
            or not (
                self.fixed_control_low_cost_terminal_liquidation_adjusted_return
                >= self.fixed_control_terminal_liquidation_adjusted_return
                >= self.fixed_control_high_cost_terminal_liquidation_adjusted_return
            )
            or not isinstance(self.source_data_qualified, bool)
            or any_runtime != runtime
            or self.runtime_rollout_replayed != runtime
            or self.outer_evaluation_authorized
            != bool(runtime and self.source_data_qualified)
            or self.profitability_reporting_authorized
            or self.positive_profitability_authorization_eligible
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer rollout authority differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.outer_rollout_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= self.outer_access_committed_at_ms
            ):
                raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                    "outer rollout source transaction differs"
                )
        if runtime:
            assert self._runtime_outer_access is not None
            assert self._runtime_rollout is not None
            self._runtime_outer_access.validate()
            self._runtime_rollout.validate()
            if (
                not self._runtime_outer_access.outer_input_access_authorized
                or self._runtime_outer_access.semantic_receipt_sha256
                != self.outer_access_commitment_receipt_sha256
                or self._runtime_outer_access.source_receipt_sha256
                != self.outer_access_source_receipt_sha256
                or self._runtime_outer_access.source_transaction_receipt_sha256
                != self.outer_access_commit_receipt_sha256
                or self._runtime_outer_access.source_transaction_committed_at_ms
                != self.outer_access_committed_at_ms
                or self._runtime_outer_access.experiment_id != self.experiment_id
                or self._runtime_outer_access.manifest_v5_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_outer_access.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_outer_access.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_rollout.semantic_receipt_sha256
                != self.outer_rollout_receipt_sha256
                or self._runtime_rollout.fold_index != self.fold_index
                or self._runtime_rollout.decision_session_dates
                != self.decision_session_dates
                or self._runtime_rollout.ppo_action_inventory_sha256
                != self.ppo_action_inventory_sha256
            ):
                raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                    "outer rollout runtime lineage differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def outer_rollout_authority_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer rollout fold differs"
        )
    return (
        f"adaptive-rl/{manifest.experiment_id}/outer-rollout-authority-v2/"
        f"fold-{fold_index}.json"
    )


def _authority_body(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    access: MassiveAdaptiveOuterAccessCommitmentV2,
    rollout: MassiveAdaptiveRLOuterRolloutComputationV2,
) -> dict[str, object]:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(access) is not MassiveAdaptiveOuterAccessCommitmentV2
        or type(rollout) is not MassiveAdaptiveRLOuterRolloutComputationV2
    ):
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer rollout authority requires exact V5 roots"
        )
    manifest.validate()
    access.validate()
    rollout.validate()
    if (
        manifest.experiment_id != access.experiment_id
        or manifest.semantic_receipt_sha256 != access.manifest_v5_receipt_sha256
        or manifest.scientific_protocol_projection_sha256
        != access.scientific_protocol_projection_sha256
        or rollout.fold_index != access.fold_index
        or rollout.outer_access_commitment_receipt_sha256
        != access.semantic_receipt_sha256
        or rollout.decision_session_dates != access.outer_decision_session_dates
    ):
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer rollout authority roots differ"
        )
    return {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "execution_implementation_registration_receipt_sha256": (
            access.execution_implementation_registration_receipt_sha256
        ),
        "scientific_execution_fingerprint_sha256": (
            access.scientific_execution_fingerprint_sha256
        ),
        "fold_index": access.fold_index,
        "outer_access_commitment_receipt_sha256": access.semantic_receipt_sha256,
        "outer_access_source_receipt_sha256": _digest(
            "outer access source", access.source_receipt_sha256
        ),
        "outer_access_commit_receipt_sha256": _digest(
            "outer access commit", access.source_transaction_receipt_sha256
        ),
        "outer_access_committed_at_ms": _required_time(
            "outer access time", access.source_transaction_committed_at_ms
        ),
        "outer_rollout_receipt_sha256": rollout.semantic_receipt_sha256,
        "decision_session_dates": rollout.decision_session_dates,
        "ppo_action_inventory_sha256": rollout.ppo_action_inventory_sha256,
        "ppo_primary_trace_receipt_sha256": rollout.ppo_primary_trace_receipt_sha256,
        "ppo_low_cost_trace_receipt_sha256": rollout.ppo_low_cost_trace_receipt_sha256,
        "ppo_high_cost_trace_receipt_sha256": rollout.ppo_high_cost_trace_receipt_sha256,
        "fixed_control_trace_receipt_sha256": rollout.fixed_control_trace_receipt_sha256,
        "fixed_control_low_cost_trace_receipt_sha256": (
            rollout.fixed_control_low_cost_trace_receipt_sha256
        ),
        "fixed_control_high_cost_trace_receipt_sha256": (
            rollout.fixed_control_high_cost_trace_receipt_sha256
        ),
        "ppo_primary_transition_inventory_sha256": rollout.ppo_primary_transition_inventory_sha256,
        "ppo_low_cost_transition_inventory_sha256": rollout.ppo_low_cost_transition_inventory_sha256,
        "ppo_high_cost_transition_inventory_sha256": rollout.ppo_high_cost_transition_inventory_sha256,
        "fixed_control_transition_inventory_sha256": rollout.fixed_control_transition_inventory_sha256,
        "fixed_control_low_cost_transition_inventory_sha256": (
            rollout.fixed_control_low_cost_transition_inventory_sha256
        ),
        "fixed_control_high_cost_transition_inventory_sha256": (
            rollout.fixed_control_high_cost_transition_inventory_sha256
        ),
        "decision_target_inventory_sha256": rollout.decision_target_inventory_sha256,
        "fixed_control_decision_target_inventory_sha256": (
            rollout.fixed_control_decision_target_inventory_sha256
        ),
        "primary_terminal_liquidation_adjusted_return": (
            rollout.primary_terminal_liquidation_adjusted_return
        ),
        "low_cost_terminal_liquidation_adjusted_return": (
            rollout.low_cost_terminal_liquidation_adjusted_return
        ),
        "high_cost_terminal_liquidation_adjusted_return": (
            rollout.high_cost_terminal_liquidation_adjusted_return
        ),
        "fixed_control_terminal_liquidation_adjusted_return": (
            rollout.fixed_control_terminal_liquidation_adjusted_return
        ),
        "fixed_control_low_cost_terminal_liquidation_adjusted_return": (
            rollout.fixed_control_low_cost_terminal_liquidation_adjusted_return
        ),
        "fixed_control_high_cost_terminal_liquidation_adjusted_return": (
            rollout.fixed_control_high_cost_terminal_liquidation_adjusted_return
        ),
        "maximum_drawdown": rollout.maximum_drawdown,
        "source_data_qualified": rollout.source_data_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SCHEMA,
    }


def _parse(
    *, root: str | Path, loaded: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLOuterRolloutAuthorityV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer rollout authority payload is not canonical JSON"
        )
    body = dict(value)
    body["decision_session_dates"] = tuple(
        str(item) for item in cast(Sequence[object], body["decision_session_dates"])
    )
    result = MassiveAdaptiveRLOuterRolloutAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_outer_rollout_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLOuterRolloutAuthorityV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(outer_access) is not MassiveAdaptiveOuterAccessCommitmentV2
    ):
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer rollout requires exact Manifest-V5 authorities"
        )
    manifest.validate()
    manifest_registration.validate()
    outer_access.validate()
    if (
        not manifest_registration.development_protocol_registered
        or manifest_registration.experiment_id != manifest.experiment_id
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or outer_access.experiment_id != manifest.experiment_id
        or outer_access.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
        or outer_access.manifest_v5_registration_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
            "outer rollout Manifest-V5 lineage differs"
        )
    rollout = execute_massive_adaptive_rl_outer_rollout_v2(outer_access=outer_access)
    body = _authority_body(manifest=manifest, access=outer_access, rollout=rollout)
    expected = MassiveAdaptiveRLOuterRolloutAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_rollout_replayed=True,
        outer_evaluation_authorized=rollout.source_data_qualified,
        _runtime_outer_access=outer_access,
        _runtime_rollout=rollout,
    )
    expected = replace(
        expected,
        semantic_receipt_sha256=semantic_sha256(expected.semantic_unsigned()),
    )
    expected.validate()
    relative = outer_rollout_authority_relative_path_v2(
        manifest=manifest, fold_index=outer_access.fold_index
    )
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root, experiment_id=manifest.experiment_id
    ):
        payload = Path(root) / relative
        transaction_paths = (
            payload,
            payload.with_name(payload.name + ".receipt.json"),
            payload.with_name(payload.name + ".commit.json"),
        )
        present = tuple(
            path.exists() or path.is_symlink() for path in transaction_paths
        )
        if any(present) and not all(present):
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer rollout transaction is incomplete"
            )
        if not all(present):
            if not allow_materialize:
                raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                    "outer rollout authority is absent"
                )
            access_time = outer_access.source_transaction_committed_at_ms
            if access_time is None:
                raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                    "outer access commit time is absent"
                )
            committed_at_ms = max(time.time_ns() // 1_000_000, access_time) + 1
            capability = issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1(
                root=root, authority=manifest_registration
            )
            with massive_adaptive_rl_manifest_v5_writer_scope_v1(
                root=root, capability=capability
            ):
                publish_massive_source_object(
                    stream=BytesIO(
                        canonical_json_file_bytes(expected.semantic_unsigned())
                    ),
                    root=root,
                    relative_payload_path=relative,
                    dataset_id=MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_DATASET,
                    source_object_key=relative,
                    requested_at_ms=committed_at_ms,
                    downloaded_at_ms=committed_at_ms,
                    schema_sha256=MASSIVE_ADAPTIVE_RL_OUTER_ROLLOUT_AUTHORITY_V2_SOURCE_SCHEMA_SHA256,
                    entitlement_receipt_sha256=rollout.semantic_receipt_sha256,
                    committed_at_ms=committed_at_ms,
                    request_id=(
                        f"ADAPTIVE-RL-OUTER-ROLLOUT-V2-{manifest.experiment_id}-"
                        f"FOLD{outer_access.fold_index}"
                    ),
                )
        parsed = _parse(
            root=root,
            loaded=load_massive_source_bundle(
                root=root,
                relative_payload_path=relative,
                verified_at_ms=time.time_ns() // 1_000_000,
            ),
        )
        if parsed.semantic_unsigned() != expected.semantic_unsigned():
            raise MassiveAdaptiveRLOuterRolloutAuthorityV2Error(
                "outer rollout authority does not replay"
            )
        result = replace(
            parsed,
            runtime_rollout_replayed=True,
            outer_evaluation_authorized=parsed.source_data_qualified,
            _runtime_outer_access=outer_access,
            _runtime_rollout=rollout,
        )
        result.validate()
        return result


__all__ = [
    "MassiveAdaptiveRLOuterRolloutAuthorityV2",
    "MassiveAdaptiveRLOuterRolloutAuthorityV2Error",
    "MassiveAdaptiveRLOuterRolloutComputationV2",
    "execute_massive_adaptive_rl_outer_rollout_v2",
    "outer_rollout_authority_relative_path_v2",
    "run_or_resume_massive_adaptive_rl_outer_rollout_authority_v2",
]
