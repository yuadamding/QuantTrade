"""Hard-access- and dual-cost-ladder-authenticated adaptive RL evidence.

V3 authenticates the primary-cost PPO and fit-selected static-control traces.
V4 additionally requires the prerequisite-gated outer plan and the frozen-target
10/20/40-bp ladder for both competitors.  Its additional economic gate tests
that PPO does not underperform the selected static control at 40 basis points.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_rl_fixed_control_outer_cost_ladder_v1 import (
    MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_cost_ladder_v1 import (
    MassiveAdaptiveRLOuterCostLadderAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v3 import (
    MassiveAdaptiveAuthenticatedRLOuterFoldV3,
    MassiveAdaptiveRLOuterEvidenceV3,
    build_massive_adaptive_rl_outer_evidence_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_plan_v3 import (
    MassiveAdaptiveRLOuterPlanV3,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_AUTHENTICATED_RL_OUTER_FOLD_V4_SCHEMA = (
    "rl-quant.massive-adaptive-authenticated-rl-outer-fold-v4"
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-evidence-v4"
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SPEC_SHA256 = semantic_sha256(
    {
        "primary_evidence": "ppo-and-static-authenticated-v3",
        "outer_access": "prerequisite-gated-plan-v3",
        "ppo_cost_ladder": "frozen-target-10-20-40-bp",
        "static_cost_ladder": "frozen-target-10-20-40-bp",
        "additional_gate": "mean-40bp-ppo-minus-static-nonnegative",
        "profitability_reporting": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLOuterEvidenceV4Error(ValueError):
    """The hard-access outer evidence or paired cost ladder differed."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterEvidenceV4Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAuthenticatedRLOuterFoldV4:
    fold_index: int
    authenticated_fold_v3: MassiveAdaptiveAuthenticatedRLOuterFoldV3
    outer_plan_v3_receipt_sha256: str
    outer_access_commitment_receipt_sha256: str
    ppo_outer_cost_ladder_authority_receipt_sha256: str
    ppo_outer_cost_ladder_receipt_sha256: str
    fixed_control_outer_cost_ladder_authority_receipt_sha256: str
    fixed_control_outer_cost_ladder_receipt_sha256: str
    ppo_high_cost_trace_receipt_sha256: str
    fixed_control_high_cost_trace_receipt_sha256: str
    high_cost_ppo_minus_fixed_control_log_returns: tuple[float, ...]
    mean_high_cost_ppo_minus_fixed_control_log_return: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_AUTHENTICATED_RL_OUTER_FOLD_V4_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "authenticated_fold_v3_receipt_sha256": (
                self.authenticated_fold_v3.semantic_receipt_sha256
            ),
            "outer_plan_v3_receipt_sha256": self.outer_plan_v3_receipt_sha256,
            "outer_access_commitment_receipt_sha256": (
                self.outer_access_commitment_receipt_sha256
            ),
            "ppo_outer_cost_ladder_authority_receipt_sha256": (
                self.ppo_outer_cost_ladder_authority_receipt_sha256
            ),
            "ppo_outer_cost_ladder_receipt_sha256": (
                self.ppo_outer_cost_ladder_receipt_sha256
            ),
            "fixed_control_outer_cost_ladder_authority_receipt_sha256": (
                self.fixed_control_outer_cost_ladder_authority_receipt_sha256
            ),
            "fixed_control_outer_cost_ladder_receipt_sha256": (
                self.fixed_control_outer_cost_ladder_receipt_sha256
            ),
            "ppo_high_cost_trace_receipt_sha256": (
                self.ppo_high_cost_trace_receipt_sha256
            ),
            "fixed_control_high_cost_trace_receipt_sha256": (
                self.fixed_control_high_cost_trace_receipt_sha256
            ),
            "high_cost_ppo_minus_fixed_control_log_returns": (
                self.high_cost_ppo_minus_fixed_control_log_returns
            ),
            "mean_high_cost_ppo_minus_fixed_control_log_return": (
                self.mean_high_cost_ppo_minus_fixed_control_log_return
            ),
            "source_data_qualified": self.source_data_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
        }

    def validate(self) -> None:
        self.authenticated_fold_v3.validate()
        values = self.high_cost_ppo_minus_fixed_control_log_returns
        expected_mean = sum(values) / len(values) if values else math.nan
        if (
            self.schema != MASSIVE_ADAPTIVE_AUTHENTICATED_RL_OUTER_FOLD_V4_SCHEMA
            or self.fold_index != self.authenticated_fold_v3.fold_index
            or not values
            or any(not math.isfinite(value) for value in values)
            or not math.isfinite(self.mean_high_cost_ppo_minus_fixed_control_log_return)
            or not math.isclose(
                self.mean_high_cost_ppo_minus_fixed_control_log_return,
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceV4Error(
                "dual-cost-ladder authenticated outer fold differs"
            )
        for value in (
            self.outer_plan_v3_receipt_sha256,
            self.outer_access_commitment_receipt_sha256,
            self.ppo_outer_cost_ladder_authority_receipt_sha256,
            self.ppo_outer_cost_ladder_receipt_sha256,
            self.fixed_control_outer_cost_ladder_authority_receipt_sha256,
            self.fixed_control_outer_cost_ladder_receipt_sha256,
            self.ppo_high_cost_trace_receipt_sha256,
            self.fixed_control_high_cost_trace_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("dual-cost-ladder outer fold", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_authenticated_rl_outer_fold_v4(
    *,
    authenticated_fold_v3: MassiveAdaptiveAuthenticatedRLOuterFoldV3,
    outer_plan_v3: MassiveAdaptiveRLOuterPlanV3,
    ppo_cost_ladder_authority: MassiveAdaptiveRLOuterCostLadderAuthorityV1,
    fixed_control_cost_ladder_authority: (
        MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1
    ),
) -> MassiveAdaptiveAuthenticatedRLOuterFoldV4:
    """Bind one V3 fold to its hard access gate and both cost ladders."""

    authenticated_fold_v3.validate()
    outer_plan_v3.validate()
    ppo_cost_ladder_authority.validate()
    fixed_control_cost_ladder_authority.validate()
    ppo_ladder = ppo_cost_ladder_authority.runtime_ladder
    fixed_ladder = fixed_control_cost_ladder_authority.runtime_ladder
    fold_v2 = authenticated_fold_v3.authenticated_fold_v2
    cost_fold = fold_v2.cost_fold
    if (
        ppo_ladder is None
        or not ppo_cost_ladder_authority.runtime_ladder_replayed
        or fixed_ladder is None
        or not fixed_control_cost_ladder_authority.runtime_ladder_replayed
        or outer_plan_v3.fold_index != authenticated_fold_v3.fold_index
        or outer_plan_v3.outer_plan_v2.semantic_receipt_sha256
        != authenticated_fold_v3.outer_plan_v2_receipt_sha256
        or ppo_cost_ladder_authority.semantic_receipt_sha256
        != fold_v2.outer_cost_ladder_authority_receipt_sha256
        or ppo_ladder.semantic_receipt_sha256
        != fold_v2.outer_cost_ladder_receipt_sha256
        or ppo_ladder.high_cost_trace.semantic_receipt_sha256
        != cost_fold.high_cost_trace_receipt_sha256
        or fixed_ladder.fixed_control_outer_rollout_authority_receipt_sha256
        != authenticated_fold_v3.fixed_control_outer_authority_receipt_sha256
        or fixed_ladder.fixed_control_outer_rollout_receipt_sha256
        != authenticated_fold_v3.fixed_control_outer_rollout_receipt_sha256
        or fixed_ladder.primary_trace.semantic_receipt_sha256
        != authenticated_fold_v3.fixed_control_policy_trace_receipt_sha256
        or fixed_ladder.selected_control_id
        != outer_plan_v3.outer_plan_v2.selected_fixed_control_id
        or fixed_ladder.selected_action_receipt_sha256
        != outer_plan_v3.outer_plan_v2.selected_fixed_action_receipt_sha256
        or ppo_ladder.high_cost_trace.decision_session_dates
        != fixed_ladder.high_cost_trace.decision_session_dates
        or ppo_ladder.high_cost_trace.forecast_archive_receipt_sha256
        != fixed_ladder.high_cost_trace.forecast_archive_receipt_sha256
        or ppo_ladder.high_cost_trace.inference_plan_receipt_sha256
        != fixed_ladder.high_cost_trace.inference_plan_receipt_sha256
        or ppo_ladder.high_cost_trace.calibration_receipt_sha256
        != fixed_ladder.high_cost_trace.calibration_receipt_sha256
        or ppo_ladder.high_cost_trace.initial_capital
        != fixed_ladder.high_cost_trace.initial_capital
    ):
        raise MassiveAdaptiveRLOuterEvidenceV4Error(
            "outer fold is not derived from the hard gate and paired ladders"
        )
    deltas = tuple(
        ppo - fixed
        for ppo, fixed in zip(
            ppo_ladder.high_cost_trace.incremental_rl_log_returns,
            fixed_ladder.high_cost_trace.incremental_rl_log_returns,
            strict=True,
        )
    )
    source_qualified = bool(
        authenticated_fold_v3.source_data_qualified
        and outer_plan_v3.source_data_qualified
        and ppo_cost_ladder_authority.outer_evaluation_authorized
        and fixed_control_cost_ladder_authority.outer_evaluation_authorized
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_AUTHENTICATED_RL_OUTER_FOLD_V4_SCHEMA,
        "fold_index": authenticated_fold_v3.fold_index,
        "authenticated_fold_v3": authenticated_fold_v3,
        "outer_plan_v3_receipt_sha256": outer_plan_v3.semantic_receipt_sha256,
        "outer_access_commitment_receipt_sha256": (
            outer_plan_v3.outer_access_commitment_receipt_sha256
        ),
        "ppo_outer_cost_ladder_authority_receipt_sha256": (
            ppo_cost_ladder_authority.semantic_receipt_sha256
        ),
        "ppo_outer_cost_ladder_receipt_sha256": ppo_ladder.semantic_receipt_sha256,
        "fixed_control_outer_cost_ladder_authority_receipt_sha256": (
            fixed_control_cost_ladder_authority.semantic_receipt_sha256
        ),
        "fixed_control_outer_cost_ladder_receipt_sha256": (
            fixed_ladder.semantic_receipt_sha256
        ),
        "ppo_high_cost_trace_receipt_sha256": (
            ppo_ladder.high_cost_trace.semantic_receipt_sha256
        ),
        "fixed_control_high_cost_trace_receipt_sha256": (
            fixed_ladder.high_cost_trace.semantic_receipt_sha256
        ),
        "high_cost_ppo_minus_fixed_control_log_returns": deltas,
        "mean_high_cost_ppo_minus_fixed_control_log_return": sum(deltas)
        / len(deltas),
        "source_data_qualified": source_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveAuthenticatedRLOuterFoldV4(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveAuthenticatedRLOuterFoldV4(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterEvidenceV4:
    evidence_v3: MassiveAdaptiveRLOuterEvidenceV3
    authenticated_folds: tuple[MassiveAdaptiveAuthenticatedRLOuterFoldV4, ...]
    authenticated_fold_inventory_sha256: str
    mean_high_cost_ppo_minus_fixed_control_log_return: float
    high_cost_ppo_minus_fixed_control_nonnegative: bool
    passed_gate_names: tuple[str, ...]
    failed_gate_names: tuple[str, ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "evidence_v3_receipt_sha256": self.evidence_v3.semantic_receipt_sha256,
            "authenticated_fold_inventory_sha256": (
                self.authenticated_fold_inventory_sha256
            ),
            "mean_high_cost_ppo_minus_fixed_control_log_return": (
                self.mean_high_cost_ppo_minus_fixed_control_log_return
            ),
            "high_cost_ppo_minus_fixed_control_nonnegative": (
                self.high_cost_ppo_minus_fixed_control_nonnegative
            ),
            "passed_gate_names": self.passed_gate_names,
            "failed_gate_names": self.failed_gate_names,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.evidence_v3.validate()
        for fold in self.authenticated_folds:
            fold.validate()
        values = tuple(
            value
            for fold in self.authenticated_folds
            for value in fold.high_cost_ppo_minus_fixed_control_log_returns
        )
        expected_mean = sum(values) / len(values) if values else math.nan
        gate_name = "high-cost-ppo-minus-fixed-control-nonnegative"
        expected_gate = expected_mean >= 0.0
        base = self.evidence_v3.evidence_v2.evidence_v1
        expected_passed = tuple(
            sorted((*base.passed_gate_names, *((gate_name,) if expected_gate else ())))
        )
        expected_failed = tuple(
            sorted((*base.failed_gate_names, *((gate_name,) if not expected_gate else ())))
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SCHEMA
            or tuple(fold.fold_index for fold in self.authenticated_folds)
            != tuple(range(len(self.authenticated_folds)))
            or tuple(
                fold.authenticated_fold_v3.semantic_receipt_sha256
                for fold in self.authenticated_folds
            )
            != tuple(
                fold.semantic_receipt_sha256
                for fold in self.evidence_v3.authenticated_folds
            )
            or self.authenticated_fold_inventory_sha256
            != semantic_sha256(
                tuple(fold.semantic_receipt_sha256 for fold in self.authenticated_folds)
            )
            or not math.isclose(
                self.mean_high_cost_ppo_minus_fixed_control_log_return,
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or self.high_cost_ppo_minus_fixed_control_nonnegative != expected_gate
            or self.passed_gate_names != expected_passed
            or self.failed_gate_names != expected_failed
            or self.source_data_qualified
            != all(fold.source_data_qualified for fold in self.authenticated_folds)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceV4Error(
                "dual-cost-ladder authenticated outer evidence differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_outer_evidence_v4(
    folds: Sequence[MassiveAdaptiveAuthenticatedRLOuterFoldV4],
) -> MassiveAdaptiveRLOuterEvidenceV4:
    ordered = tuple(sorted(folds, key=lambda value: value.fold_index))
    for fold in ordered:
        fold.validate()
    evidence_v3 = build_massive_adaptive_rl_outer_evidence_v3(
        tuple(fold.authenticated_fold_v3 for fold in ordered)
    )
    values = tuple(
        value
        for fold in ordered
        for value in fold.high_cost_ppo_minus_fixed_control_log_returns
    )
    mean = sum(values) / len(values)
    gate_name = "high-cost-ppo-minus-fixed-control-nonnegative"
    passed = mean >= 0.0
    base = evidence_v3.evidence_v2.evidence_v1
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SCHEMA,
        "evidence_v3": evidence_v3,
        "authenticated_folds": ordered,
        "authenticated_fold_inventory_sha256": semantic_sha256(
            tuple(fold.semantic_receipt_sha256 for fold in ordered)
        ),
        "mean_high_cost_ppo_minus_fixed_control_log_return": mean,
        "high_cost_ppo_minus_fixed_control_nonnegative": passed,
        "passed_gate_names": tuple(
            sorted((*base.passed_gate_names, *((gate_name,) if passed else ())))
        ),
        "failed_gate_names": tuple(
            sorted((*base.failed_gate_names, *((gate_name,) if not passed else ())))
        ),
        "source_data_qualified": all(fold.source_data_qualified for fold in ordered),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_V4_SOURCE_SHA256,
    }
    provisional = MassiveAdaptiveRLOuterEvidenceV4(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLOuterEvidenceV4(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveAuthenticatedRLOuterFoldV4",
    "MassiveAdaptiveRLOuterEvidenceV4",
    "MassiveAdaptiveRLOuterEvidenceV4Error",
    "build_massive_adaptive_authenticated_rl_outer_fold_v4",
    "build_massive_adaptive_rl_outer_evidence_v4",
]
