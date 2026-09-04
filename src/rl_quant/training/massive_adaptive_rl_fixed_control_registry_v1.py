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
    from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
        MassiveAdaptiveRLFixedControlFitAuthorityV1,
    )


MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-registry-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_REGISTRY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SCIENTIFIC_INVENTORY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-scientific-inventory-v1"
)


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
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
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
            != (
                "FC00",
                "FC01",
                "FC02",
                "FC03",
                "FC04",
                "FC05",
                "FC07",
                "FC08",
                "FC09",
                "FC10",
                "FC11",
                "FC12",
                "FC06",
            )
            or self.control_ids != tuple(row.control_id for row in self.controls)
            or self.control_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.controls)
            )
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
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


def registered_massive_adaptive_rl_constant_actions_v1() -> tuple[
    tuple[str, MassiveAdaptiveRLActionV1], ...
]:
    """Return the complete immutable symmetric constant-action fitting grid."""

    neutral = neutral_massive_adaptive_rl_action_v1()
    short = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.75, 0.50, 0.25, 0.0, -0.25, -0.50, -0.75),
        uncertainty_control=0.0,
        risk_control=0.0,
        trade_cost_control=0.0,
    )
    long = build_massive_adaptive_rl_action_v1(
        bucket_controls=(-0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75),
        uncertainty_control=0.0,
        risk_control=0.0,
        trade_cost_control=0.0,
    )
    uncertain = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.75,
        risk_control=0.0,
        trade_cost_control=0.0,
    )
    risky = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=0.75,
        trade_cost_control=0.0,
    )
    high_trade_hurdle = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=0.0,
        trade_cost_control=1.0,
    )
    low_uncertainty_aversion = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=-0.75,
        risk_control=0.0,
        trade_cost_control=0.0,
    )
    low_risk_aversion = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=-0.75,
        trade_cost_control=0.0,
    )
    low_trade_hurdle = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=0.0,
        trade_cost_control=-1.0,
    )
    short_low_hurdle = build_massive_adaptive_rl_action_v1(
        bucket_controls=short.bucket_controls,
        uncertainty_control=0.0,
        risk_control=0.0,
        trade_cost_control=-1.0,
    )
    long_high_hurdle = build_massive_adaptive_rl_action_v1(
        bucket_controls=long.bucket_controls,
        uncertainty_control=0.0,
        risk_control=0.0,
        trade_cost_control=1.0,
    )
    low_risk_low_hurdle = build_massive_adaptive_rl_action_v1(
        bucket_controls=(0.0,) * 7,
        uncertainty_control=0.0,
        risk_control=-0.75,
        trade_cost_control=-1.0,
    )
    return (
        ("FC00", neutral),
        ("FC01", short),
        ("FC02", long),
        ("FC03", uncertain),
        ("FC04", risky),
        ("FC05", high_trade_hurdle),
        ("FC07", low_uncertainty_aversion),
        ("FC08", low_risk_aversion),
        ("FC09", low_trade_hurdle),
        ("FC10", short_low_hurdle),
        ("FC11", long_high_hurdle),
        ("FC12", low_risk_low_hurdle),
    )


def massive_adaptive_rl_fixed_control_scientific_inventory_v1() -> dict[str, object]:
    """Return the source-free comparator choices that affect the experiment.

    Action receipts intentionally include implementation identities.  Manifest
    V5 instead commits the numerical controls and the FC06 fit-only selection
    rule directly, so a source refactor cannot change the scientific protocol
    while a changed comparator cannot hide behind the same descriptive label.
    """

    controls = tuple(
        {
            "control_id": control_id,
            "controller_kind": "constant",
            "bucket_controls": action.bucket_controls,
            "uncertainty_control": action.uncertainty_control,
            "risk_control": action.risk_control,
            "trade_cost_control": action.trade_cost_control,
        }
        for control_id, action in registered_massive_adaptive_rl_constant_actions_v1()
    )
    return {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SCIENTIFIC_INVENTORY_V1_SCHEMA,
        "constant_controls": controls,
        "fc06": {
            "control_id": "FC06",
            "controller_kind": "training-selected-constant",
            "candidate_control_ids": tuple(row["control_id"] for row in controls),
            "selection_data": "registered-fold-rl-fit-only",
            "primary_cost_basis_points": 20.0,
            "objective": "maximum-training-incremental-log-wealth",
            "tie_breaking": "lexicographically-greatest-control-id",
            "missing_or_invalid_candidate": "selection-invalid",
        },
    }


def build_massive_adaptive_rl_fixed_control_registry_v1() -> (
    MassiveAdaptiveRLFixedControlRegistryV1
):
    actions = dict(registered_massive_adaptive_rl_constant_actions_v1())
    controls = (
        _spec(
            "FC00",
            "constant",
            actions["FC00"].semantic_receipt_sha256,
            "neutral action",
        ),
        _spec(
            "FC01",
            "constant",
            actions["FC01"].semantic_receipt_sha256,
            "short-horizon emphasis",
        ),
        _spec(
            "FC02",
            "constant",
            actions["FC02"].semantic_receipt_sha256,
            "long-horizon emphasis",
        ),
        _spec(
            "FC03",
            "constant",
            actions["FC03"].semantic_receipt_sha256,
            "increased uncertainty aversion",
        ),
        _spec(
            "FC04",
            "constant",
            actions["FC04"].semantic_receipt_sha256,
            "increased portfolio-risk aversion",
        ),
        _spec(
            "FC05",
            "constant",
            actions["FC05"].semantic_receipt_sha256,
            "high soft trade-cost hurdle",
        ),
        _spec(
            "FC07",
            "constant",
            actions["FC07"].semantic_receipt_sha256,
            "decreased uncertainty aversion",
        ),
        _spec(
            "FC08",
            "constant",
            actions["FC08"].semantic_receipt_sha256,
            "decreased portfolio-risk aversion",
        ),
        _spec(
            "FC09",
            "constant",
            actions["FC09"].semantic_receipt_sha256,
            "low soft trade-cost hurdle",
        ),
        _spec(
            "FC10",
            "constant",
            actions["FC10"].semantic_receipt_sha256,
            "short-horizon emphasis with low trade hurdle",
        ),
        _spec(
            "FC11",
            "constant",
            actions["FC11"].semantic_receipt_sha256,
            "long-horizon emphasis with high trade hurdle",
        ),
        _spec(
            "FC12",
            "constant",
            actions["FC12"].semantic_receipt_sha256,
            "lower risk aversion with low trade hurdle",
        ),
        _spec(
            "FC06",
            "training-selected-constant",
            None,
            "best symmetric-grid constant action selected on RL-fit only",
        ),
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
    fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
) -> None:
    registry.validate()
    fit_authority.validate()
    selection_authority.validate()
    chronology_authority.validate()
    fit_run = getattr(fit_authority, "runtime_fit_run", None)
    candidates = getattr(selection_authority, "runtime_candidates", None)
    selection = getattr(selection_authority, "runtime_selection", None)
    registered_constants = tuple(
        row.control_id for row in registry.controls if row.controller_kind == "constant"
    )
    if (
        fit_run is None
        or not fit_authority.runtime_fit_replayed
        or candidates is None
        or selection is None
        or not selection_authority.runtime_selection_replayed
        or tuple(sorted(row.control_id for row in candidates)) != registered_constants
        or selection.fold_index != chronology_authority.fold_index
        or selection.training_origin_inventory_sha256
        != chronology_authority.rl_fit_origin_inventory_sha256
        or selection.selected_control_id not in registered_constants
        or fit_run.fixed_control_registry_receipt_sha256
        != registry.semantic_receipt_sha256
        or fit_run.chronology_authority_receipt_sha256
        != chronology_authority.semantic_receipt_sha256
        or fit_run.training_origin_inventory_sha256
        != chronology_authority.rl_fit_origin_inventory_sha256
        or selection.candidate_inventory_sha256 != fit_run.candidate_inventory_sha256
        or tuple(sorted(row.semantic_receipt_sha256 for row in candidates))
        != tuple(sorted(row.semantic_receipt_sha256 for row in fit_run.candidates))
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
    if any(
        supplied[control_id] != receipt
        for control_id, receipt in constant_receipts.items()
    ):
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
    fixed_control_fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    fixed_control_selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    fixed_control_evaluation: MassiveAdaptiveRLFixedControlEvaluationV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    shared_validation_context_receipt_sha256: str,
) -> MassiveAdaptiveRLPolicyCandidateV1:
    """Build a PPO candidate only from the package-generated FC06 trace."""

    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
        registry=fixed_control_registry,
        fit_authority=fixed_control_fit_authority,
        selection_authority=fixed_control_selection_authority,
        chronology_authority=chronology_authority,
    )
    fixed_control_evaluation.validate()
    if (
        fixed_control_evaluation.fixed_control_registry_receipt_sha256
        != fixed_control_registry.semantic_receipt_sha256
        or fixed_control_evaluation.fixed_control_fit_authority_receipt_sha256
        != fixed_control_fit_authority.semantic_receipt_sha256
        or fixed_control_evaluation.fixed_control_selection_authority_receipt_sha256
        != fixed_control_selection_authority.semantic_receipt_sha256
        or fixed_control_evaluation.validation_context_receipt_sha256
        != shared_validation_context_receipt_sha256
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
    "MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SCIENTIFIC_INVENTORY_V1_SCHEMA",
    "MassiveAdaptiveRLFixedControlRegistryV1",
    "MassiveAdaptiveRLFixedControlRegistryV1Error",
    "MassiveAdaptiveRLFixedControlSpecV1",
    "build_massive_adaptive_rl_fixed_control_registry_v1",
    "massive_adaptive_rl_fixed_control_scientific_inventory_v1",
    "build_massive_adaptive_rl_policy_candidate_with_registry_v1",
    "registered_massive_adaptive_rl_constant_actions_v1",
    "validate_massive_adaptive_rl_fixed_control_registry_coverage_v1",
]
