"""Frozen-target 10/20/40-bp ladder for the fit-selected static control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
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
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_v1 import (
    _environment_identity,
    replay_massive_adaptive_rl_frozen_target_transitions_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_outer_rollout_v1 import (
    MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    registered_massive_adaptive_rl_constant_actions_v1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_trace_from_identities_v1,
)


MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-outer-cost-ladder-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-outer-cost-ladder-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fixed-control-outer-cost-ladder-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SPEC_SHA256 = semantic_sha256(
    {
        "primary": "fit-selected-static-control-outer-rollout",
        "stress": "same-primary-targets-at-10-and-40-basis-points",
        "controller_rerun_on_stress": False,
        "duration_semantics": False,
    }
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_SCHEMA,
        "payload": "fixed-control-primary-and-frozen-target-stress-identities",
        "promotion": "reopen-fixed-rollout-and-rerun-stress-economics",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(ValueError):
    """The selected static control or its frozen cost replay differed."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlOuterCostLadderV1:
    fold_index: int
    fixed_control_outer_rollout_authority_receipt_sha256: str
    fixed_control_outer_rollout_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_control_id: str
    selected_action_receipt_sha256: str
    primary_trace: MassiveAdaptiveRLPolicyTraceV1
    low_cost_trace: MassiveAdaptiveRLPolicyTraceV1
    high_cost_trace: MassiveAdaptiveRLPolicyTraceV1
    low_cost_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...]
    high_cost_transitions: tuple[MassiveAdaptiveRLTransitionV1, ...]
    decision_target_inventory_sha256: str
    low_cost_transition_inventory_sha256: str
    high_cost_transition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SPEC_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "fixed_control_outer_rollout_authority_receipt_sha256": (
                self.fixed_control_outer_rollout_authority_receipt_sha256
            ),
            "fixed_control_outer_rollout_receipt_sha256": (
                self.fixed_control_outer_rollout_receipt_sha256
            ),
            "fixed_control_selection_authority_receipt_sha256": (
                self.fixed_control_selection_authority_receipt_sha256
            ),
            "selected_control_id": self.selected_control_id,
            "selected_action_receipt_sha256": self.selected_action_receipt_sha256,
            "primary_trace_receipt_sha256": self.primary_trace.semantic_receipt_sha256,
            "low_cost_trace_receipt_sha256": self.low_cost_trace.semantic_receipt_sha256,
            "high_cost_trace_receipt_sha256": self.high_cost_trace.semantic_receipt_sha256,
            "decision_target_inventory_sha256": self.decision_target_inventory_sha256,
            "low_cost_transition_inventory_sha256": (
                self.low_cost_transition_inventory_sha256
            ),
            "high_cost_transition_inventory_sha256": (
                self.high_cost_transition_inventory_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        for trace in (self.low_cost_trace, self.primary_trace, self.high_cost_trace):
            trace.validate()
        for transition in (*self.low_cost_transitions, *self.high_cost_transitions):
            transition.validate()
        traces = (self.low_cost_trace, self.primary_trace, self.high_cost_trace)
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SCHEMA
            or tuple(row.transaction_cost_basis_points for row in traces)
            != (10.0, 20.0, 40.0)
            or any(row.evaluation_role != "outer_test" for row in traces)
            or any(row.fold_index != self.fold_index for row in traces)
            or self.primary_trace.frozen_targets_replayed
            or not self.low_cost_trace.frozen_targets_replayed
            or not self.high_cost_trace.frozen_targets_replayed
            or len({row.decision_target_inventory_sha256 for row in traces}) != 1
            or self.decision_target_inventory_sha256
            != self.primary_trace.decision_target_inventory_sha256
            or self.low_cost_transition_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.low_cost_transitions)
            )
            or self.high_cost_transition_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.high_cost_transitions)
            )
            or not self.low_cost_trace.terminal_liquidation_adjusted_return
            >= self.primary_trace.terminal_liquidation_adjusted_return
            >= self.high_cost_trace.terminal_liquidation_adjusted_return
            or self.outer_evaluation_authorized != self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(
                "fixed-control outer cost ladder differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def evaluate_massive_adaptive_rl_fixed_control_outer_cost_ladder_v1(
    *,
    rollout_authority: MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveRLFixedControlOuterCostLadderV1:
    rollout_authority.validate()
    primary = rollout_authority.runtime_rollout
    environments = (low_cost_environment, primary_environment, high_cost_environment)
    if (
        primary is None
        or not rollout_authority.runtime_rollout_replayed
        or tuple(row.transaction_cost_basis_points for row in environments)
        != (10.0, 20.0, 40.0)
        or len({_environment_identity(row) for row in environments}) != 1
        or primary_environment.forecast_archive.semantic_receipt_sha256
        != primary.policy_trace.forecast_archive_receipt_sha256
        or primary_environment.inference_plan.semantic_receipt_sha256
        != primary.policy_trace.inference_plan_receipt_sha256
        or primary_environment.calibration.semantic_receipt_sha256
        != primary.policy_trace.calibration_receipt_sha256
    ):
        raise MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(
            "fixed-control cost environments or primary rollout differ"
        )
    actions = dict(registered_massive_adaptive_rl_constant_actions_v1())
    action = actions.get(primary.selected_control_id)
    if (
        action is None
        or action.semantic_receipt_sha256 != primary.selected_action_receipt_sha256
    ):
        raise MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(
            "selected fixed action is absent from the registry"
        )
    values = (
        *action.bucket_controls,
        action.uncertainty_control,
        action.risk_control,
        action.trade_cost_control,
    )
    replay_evidence = tuple(
        SimpleNamespace(
            decision_session_date=row.decision_session_date,
            observation_receipt_sha256=row.observation_receipt_sha256,
            action_values=values,
        )
        for row in primary.action_evidence
    )
    low_transitions = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=replay_evidence,
        primary_transitions=primary.transitions,
        environment=low_cost_environment,
    )
    high_transitions = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=replay_evidence,
        primary_transitions=primary.transitions,
        environment=high_cost_environment,
    )

    def trace(
        environment: MassiveAdaptiveProfitabilityEnvV1,
        transitions: tuple[MassiveAdaptiveRLTransitionV1, ...],
    ) -> MassiveAdaptiveRLPolicyTraceV1:
        return build_massive_adaptive_rl_policy_trace_from_identities_v1(
            fold_index=primary.fold_index,
            checkpoint_receipt_sha256=primary.policy_trace.checkpoint_receipt_sha256,
            model_state_receipt_sha256=primary.selected_action_receipt_sha256,
            update_index=0,
            training_forecast_authority_receipt_sha256=(
                primary.policy_trace.training_forecast_authority_receipt_sha256
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
            transitions=transitions,
            frozen_targets_replayed=True,
            evaluation_role="outer_test",
            checkpoint_source_data_qualified=primary.source_data_qualified,
        )

    low_trace = trace(low_cost_environment, low_transitions)
    high_trace = trace(high_cost_environment, high_transitions)
    source_qualified = bool(
        rollout_authority.outer_evaluation_authorized
        and low_trace.source_data_qualified
        and high_trace.source_data_qualified
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SCHEMA,
        "fold_index": primary.fold_index,
        "fixed_control_outer_rollout_authority_receipt_sha256": (
            rollout_authority.semantic_receipt_sha256
        ),
        "fixed_control_outer_rollout_receipt_sha256": primary.semantic_receipt_sha256,
        "fixed_control_selection_authority_receipt_sha256": (
            primary.fixed_control_selection_authority_receipt_sha256
        ),
        "selected_control_id": primary.selected_control_id,
        "selected_action_receipt_sha256": primary.selected_action_receipt_sha256,
        "primary_trace": primary.policy_trace,
        "low_cost_trace": low_trace,
        "high_cost_trace": high_trace,
        "low_cost_transitions": low_transitions,
        "high_cost_transitions": high_transitions,
        "decision_target_inventory_sha256": primary.policy_trace.decision_target_inventory_sha256,
        "low_cost_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in low_transitions)
        ),
        "high_cost_transition_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in high_transitions)
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SPEC_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFixedControlOuterCostLadderV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        outer_evaluation_authorized=source_qualified,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _payload(
    ladder: MassiveAdaptiveRLFixedControlOuterCostLadderV1,
) -> dict[str, object]:
    ladder.validate()
    return {
        "fold_index": ladder.fold_index,
        "fixed_control_outer_rollout_authority_receipt_sha256": (
            ladder.fixed_control_outer_rollout_authority_receipt_sha256
        ),
        "fixed_control_outer_rollout_receipt_sha256": (
            ladder.fixed_control_outer_rollout_receipt_sha256
        ),
        "fixed_control_selection_authority_receipt_sha256": (
            ladder.fixed_control_selection_authority_receipt_sha256
        ),
        "selected_control_id": ladder.selected_control_id,
        "selected_action_receipt_sha256": ladder.selected_action_receipt_sha256,
        "primary_trace": asdict(ladder.primary_trace),
        "low_cost_trace": asdict(ladder.low_cost_trace),
        "high_cost_trace": asdict(ladder.high_cost_trace),
        "decision_target_inventory_sha256": ladder.decision_target_inventory_sha256,
        "low_cost_transition_inventory_sha256": (
            ladder.low_cost_transition_inventory_sha256
        ),
        "high_cost_transition_inventory_sha256": (
            ladder.high_cost_transition_inventory_sha256
        ),
        "source_data_qualified": ladder.source_data_qualified,
        "cost_ladder_receipt_sha256": ladder.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(
            "fixed-control outer cost ladder is not canonical JSON"
        )
    return dict(value)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1:
    fold_index: int
    fixed_control_outer_rollout_authority_receipt_sha256: str
    fixed_control_outer_cost_ladder_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_control_id: str
    selected_action_receipt_sha256: str
    primary_trace_receipt_sha256: str
    low_cost_trace_receipt_sha256: str
    high_cost_trace_receipt_sha256: str
    decision_target_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_ladder: MassiveAdaptiveRLFixedControlOuterCostLadderV1 | None
    runtime_ladder_replayed: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SOURCE_SHA256
    )
    schema: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_SCHEMA
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "fixed_control_outer_rollout_authority_receipt_sha256": (
                self.fixed_control_outer_rollout_authority_receipt_sha256
            ),
            "fixed_control_outer_cost_ladder_receipt_sha256": (
                self.fixed_control_outer_cost_ladder_receipt_sha256
            ),
            "fixed_control_selection_authority_receipt_sha256": (
                self.fixed_control_selection_authority_receipt_sha256
            ),
            "selected_control_id": self.selected_control_id,
            "selected_action_receipt_sha256": self.selected_action_receipt_sha256,
            "primary_trace_receipt_sha256": self.primary_trace_receipt_sha256,
            "low_cost_trace_receipt_sha256": self.low_cost_trace_receipt_sha256,
            "high_cost_trace_receipt_sha256": self.high_cost_trace_receipt_sha256,
            "decision_target_inventory_sha256": self.decision_target_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_ladder is not None
        expected = runtime and self.source_data_qualified
        if self.runtime_ladder is not None:
            self.runtime_ladder.validate()
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.fixed_control_outer_cost_ladder_receipt_sha256
            or self.runtime_ladder_replayed != runtime
            or self.outer_evaluation_authorized != expected
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(
                "fixed-control outer cost ladder authority differs"
            )
        if runtime and self.runtime_ladder is not None and (
            self.runtime_ladder.semantic_receipt_sha256
            != self.fixed_control_outer_cost_ladder_receipt_sha256
            or self.runtime_ladder.fixed_control_outer_rollout_authority_receipt_sha256
            != self.fixed_control_outer_rollout_authority_receipt_sha256
            or self.runtime_ladder.decision_target_inventory_sha256
            != self.decision_target_inventory_sha256
        ):
            raise MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(
                "fixed-control runtime ladder differs from authority"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def parse_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    primary = dict(cast(Mapping[str, object], payload["primary_trace"]))
    low = dict(cast(Mapping[str, object], payload["low_cost_trace"]))
    high = dict(cast(Mapping[str, object], payload["high_cost_trace"]))
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_SCHEMA,
        "fold_index": int(cast(int | str, payload["fold_index"])),
        "fixed_control_outer_rollout_authority_receipt_sha256": str(
            payload["fixed_control_outer_rollout_authority_receipt_sha256"]
        ),
        "fixed_control_outer_cost_ladder_receipt_sha256": str(
            payload["cost_ladder_receipt_sha256"]
        ),
        "fixed_control_selection_authority_receipt_sha256": str(
            payload["fixed_control_selection_authority_receipt_sha256"]
        ),
        "selected_control_id": str(payload["selected_control_id"]),
        "selected_action_receipt_sha256": str(payload["selected_action_receipt_sha256"]),
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
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_ladder=None,
        runtime_ladder_replayed=False,
        outer_evaluation_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1,
    rollout_authority: MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1:
    parsed = parse_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    replayed = evaluate_massive_adaptive_rl_fixed_control_outer_cost_ladder_v1(
        rollout_authority=rollout_authority,
        primary_environment=primary_environment,
        low_cost_environment=low_cost_environment,
        high_cost_environment=high_cost_environment,
    )
    if canonical_json_file_bytes(
        _load_payload(root=root, loaded_source=authority.loaded_source)
    ) != canonical_json_file_bytes(_payload(replayed)):
        raise MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(
            "fixed-control outer cost ladder did not replay"
        )
    result = replace(
        parsed,
        runtime_ladder=replayed,
        runtime_ladder_replayed=True,
        outer_evaluation_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    rollout_authority: MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLFixedControlOuterCostLadderV1Error(
            "fixed-control cost ladder artifact ID is not path safe"
        )
    ladder = evaluate_massive_adaptive_rl_fixed_control_outer_cost_ladder_v1(
        rollout_authority=rollout_authority,
        primary_environment=primary_environment,
        low_cost_environment=low_cost_environment,
        high_cost_environment=high_cost_environment,
    )
    relative = (
        "massive-adaptive/rl-fixed-control-outer-cost-ladder-authority-v1/"
        f"{artifact_id}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(ladder))),
        root=root,
        relative_payload_path=relative,
        dataset_id=(
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_DATASET
        ),
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_OUTER_COST_LADDER_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=ladder.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-FIXED-OUTER-COST-LADDER-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1(
            root=root, loaded_source=loaded
        ),
        rollout_authority=rollout_authority,
        primary_environment=primary_environment,
        low_cost_environment=low_cost_environment,
        high_cost_environment=high_cost_environment,
    )


__all__ = [
    "MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1",
    "MassiveAdaptiveRLFixedControlOuterCostLadderV1",
    "MassiveAdaptiveRLFixedControlOuterCostLadderV1Error",
    "authorize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1",
    "evaluate_massive_adaptive_rl_fixed_control_outer_cost_ladder_v1",
    "materialize_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1",
    "parse_massive_adaptive_rl_fixed_control_outer_cost_ladder_authority_v1",
]
