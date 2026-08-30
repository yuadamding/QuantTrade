"""Protocol-owned comparator registry for adaptive RL policy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    MassiveAdaptiveRLActionV1,
    build_massive_adaptive_rl_action_v1,
    neutral_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyCandidateV1,
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_candidate_v1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptiveRLCheckpointV1
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)

if TYPE_CHECKING:
    from rl_quant.evaluation.massive_adaptive_rl_fixed_control_evaluator_v1 import (
        MassiveAdaptiveRLFixedControlEvaluationV1,
    )


MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-registry-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))


class MassiveAdaptiveRLFixedControlRegistryV1Error(ValueError):
    """A required static or contextual comparator is missing or substituted."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlSpecV1:
    control_id: str
    controller_kind: str
    registered_action_receipt_sha256: str | None
    description: str
    semantic_receipt_sha256: str

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.controller_kind not in {"constant", "training-selected-constant"}
            or not self.control_id.startswith("FC")
            or not self.description
            or (self.controller_kind == "constant")
            != (self.registered_action_receipt_sha256 is not None)
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlRegistryV1Error(
                "adaptive RL fixed-control specification differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlRegistryV1:
    controls: tuple[MassiveAdaptiveRLFixedControlSpecV1, ...]
    control_ids: tuple[str, ...]
    control_inventory_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "controls": tuple(asdict(value) for value in self.controls),
            "control_ids": self.control_ids,
            "control_inventory_sha256": self.control_inventory_sha256,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        for control in self.controls:
            control.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SCHEMA
            or self.control_ids
            != ("FC00", "FC01", "FC02", "FC03", "FC04", "FC05", "FC06")
            or self.control_ids != tuple(row.control_id for row in self.controls)
            or self.control_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.controls)
            )
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlRegistryV1Error(
                "adaptive RL fixed-control registry differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _spec(
    control_id: str,
    controller_kind: str,
    action_receipt: str | None,
    description: str,
) -> MassiveAdaptiveRLFixedControlSpecV1:
    body = {
        "control_id": control_id,
        "controller_kind": controller_kind,
        "registered_action_receipt_sha256": action_receipt,
        "description": description,
    }
    result = MassiveAdaptiveRLFixedControlSpecV1(
        control_id=control_id,
        controller_kind=controller_kind,
        registered_action_receipt_sha256=action_receipt,
        description=description,
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def registered_massive_adaptive_rl_constant_actions_v1(
) -> tuple[tuple[str, MassiveAdaptiveRLActionV1], ...]:
    """Return the complete immutable FC00--FC05 fitting grid."""

    neutral = neutral_massive_adaptive_rl_action_v1()
    short = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.75, 0.50, 0.25, 0.0, -0.25, -0.50, -0.75),
        uncertainty_control=0.0,
        risk_control=0.0,
        turnover_control=0.0,
    )
    long = build_massive_adaptive_rl_action_v1(
        bucket_controls=(-0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75),
        uncertainty_control=0.0,
        risk_control=0.0,
        turnover_control=0.0,
    )
    uncertain = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.75,
        risk_control=0.0,
        turnover_control=0.0,
    )
    risky = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=0.75,
        turnover_control=0.0,
    )
    low_turnover = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=0.0,
        turnover_control=1.0,
    )
    return (
        ("FC00", neutral),
        ("FC01", short),
        ("FC02", long),
        ("FC03", uncertain),
        ("FC04", risky),
        ("FC05", low_turnover),
    )


def build_massive_adaptive_rl_fixed_control_registry_v1(
) -> MassiveAdaptiveRLFixedControlRegistryV1:
    actions = dict(registered_massive_adaptive_rl_constant_actions_v1())
    controls = (
        _spec("FC00", "constant", actions["FC00"].semantic_receipt_sha256, "neutral action"),
        _spec("FC01", "constant", actions["FC01"].semantic_receipt_sha256, "short-horizon emphasis"),
        _spec("FC02", "constant", actions["FC02"].semantic_receipt_sha256, "long-horizon emphasis"),
        _spec("FC03", "constant", actions["FC03"].semantic_receipt_sha256, "increased uncertainty aversion"),
        _spec("FC04", "constant", actions["FC04"].semantic_receipt_sha256, "increased portfolio-risk aversion"),
        _spec("FC05", "constant", actions["FC05"].semantic_receipt_sha256, "strong turnover tightening"),
        _spec("FC06", "training-selected-constant", None, "best constant action selected on RL-fit only"),
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SCHEMA,
        "controls": controls,
        "control_ids": tuple(row.control_id for row in controls),
        "control_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in controls)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLFixedControlRegistryV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(
            {**body, "controls": tuple(asdict(row) for row in controls)}
        ),
    )
    result.validate()
    return result


def validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
    *,
    registry: MassiveAdaptiveRLFixedControlRegistryV1,
    selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
) -> None:
    registry.validate()
    selection_authority.validate()
    chronology_authority.validate()
    candidates = getattr(selection_authority, "runtime_candidates", None)
    selection = getattr(selection_authority, "runtime_selection", None)
    registered_constants = tuple(
        row.control_id for row in registry.controls if row.controller_kind == "constant"
    )
    if (
        candidates is None
        or selection is None
        or not selection_authority.runtime_selection_replayed
        or tuple(sorted(row.control_id for row in candidates))
        != registered_constants
        or selection.fold_index != chronology_authority.fold_index
        or selection.training_origin_inventory_sha256
        != chronology_authority.rl_fit_origin_inventory_sha256
        or selection.selected_control_id not in registered_constants
    ):
        raise MassiveAdaptiveRLFixedControlRegistryV1Error(
            "adaptive RL comparator registry is incomplete"
        )
    constant_receipts = {
        row.control_id: row.registered_action_receipt_sha256
        for row in registry.controls
        if row.controller_kind == "constant"
    }
    supplied = {row.control_id: row.action_receipt_sha256 for row in candidates}
    if any(supplied[control_id] != receipt for control_id, receipt in constant_receipts.items()):
        raise MassiveAdaptiveRLFixedControlRegistryV1Error(
            "adaptive RL registered constant control was substituted"
        )


def build_massive_adaptive_rl_policy_candidate_with_registry_v1(
    *,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    primary_trace: MassiveAdaptiveRLPolicyTraceV1,
    low_cost_trace: MassiveAdaptiveRLPolicyTraceV1,
    high_cost_trace: MassiveAdaptiveRLPolicyTraceV1,
    fixed_control_registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fixed_control_selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    fixed_control_evaluation: MassiveAdaptiveRLFixedControlEvaluationV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
) -> MassiveAdaptiveRLPolicyCandidateV1:
    """Build a PPO candidate only from the package-generated FC06 trace."""

    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
        registry=fixed_control_registry,
        selection_authority=fixed_control_selection_authority,
        chronology_authority=chronology_authority,
    )
    fixed_control_evaluation.validate()
    if (
        fixed_control_evaluation.fixed_control_registry_receipt_sha256
        != fixed_control_registry.semantic_receipt_sha256
        or fixed_control_evaluation.fixed_control_selection_authority_receipt_sha256
        != fixed_control_selection_authority.semantic_receipt_sha256
        or fixed_control_evaluation.fold_index != chronology_authority.fold_index
    ):
        raise MassiveAdaptiveRLFixedControlRegistryV1Error(
            "adaptive RL FC06 validation evidence differs from the PPO candidate"
        )
    return build_massive_adaptive_rl_policy_candidate_v1(
        checkpoint=checkpoint,
        primary_trace=primary_trace,
        low_cost_trace=low_cost_trace,
        high_cost_trace=high_cost_trace,
        fixed_control_selection_authority=fixed_control_selection_authority,
        fixed_control_validation_trace=fixed_control_evaluation.policy_trace,
    )


__all__ = [
    "MassiveAdaptiveRLFixedControlRegistryV1",
    "MassiveAdaptiveRLFixedControlRegistryV1Error",
    "MassiveAdaptiveRLFixedControlSpecV1",
    "build_massive_adaptive_rl_fixed_control_registry_v1",
    "build_massive_adaptive_rl_policy_candidate_with_registry_v1",
    "registered_massive_adaptive_rl_constant_actions_v1",
    "validate_massive_adaptive_rl_fixed_control_registry_coverage_v1",
]
