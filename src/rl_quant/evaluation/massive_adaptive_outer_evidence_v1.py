"""Four-fold statistical evidence for the frozen deterministic outer run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from statistics import mean
from typing import Sequence

from rl_quant.evaluation.massive_adaptive_outer_profitability_authority_v1 import (
    MassiveAdaptiveOuterProfitabilityAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_profit_trace_v1 import (
    MassiveAdaptiveProfitTraceV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_profit_checkpoint_selection_authority_v2 import (
    MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1,
    MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1,
)

MASSIVE_ADAPTIVE_OUTER_COST_FOLD_V1_SCHEMA = (
    "rl-quant.massive-adaptive-outer-cost-fold-v1"
)
MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-outer-evidence-v1"
)
MASSIVE_ADAPTIVE_OUTER_PRIMARY_CAPITAL_V1 = 10_000_000.0
MASSIVE_ADAPTIVE_OUTER_PRIMARY_COST_BP_V1 = 20.0
MASSIVE_ADAPTIVE_OUTER_LOW_COST_BP_V1 = 10.0
MASSIVE_ADAPTIVE_OUTER_HIGH_COST_BP_V1 = 40.0
MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_REPLICATES_V1 = 2_000
MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_BLOCK_SESSIONS_V1 = 63
MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_SEED_V1 = 0
MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "folds": MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1,
        "sessions_per_fold": MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1,
        "primary_capital": MASSIVE_ADAPTIVE_OUTER_PRIMARY_CAPITAL_V1,
        "cost_ladder_bp": (
            MASSIVE_ADAPTIVE_OUTER_LOW_COST_BP_V1,
            MASSIVE_ADAPTIVE_OUTER_PRIMARY_COST_BP_V1,
            MASSIVE_ADAPTIVE_OUTER_HIGH_COST_BP_V1,
        ),
        "actions": "primary-target-inventory-frozen-across-cost-rungs",
        "bootstrap": {
            "kind": "fold-cluster-nonwrapping-moving-block",
            "replicates": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_REPLICATES_V1,
            "block_sessions": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_BLOCK_SESSIONS_V1,
            "seed": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_SEED_V1,
            "lower_bound": "one-sided-95-percentile",
        },
        "gates": (
            "primary-mean-net-lcb-positive",
            "primary-mean-active-log-lcb-positive",
            "high-cost-mean-terminal-return-nonnegative",
            "positive-primary-folds-at-least-three",
            "all-fold-terminal-cost-ladders-monotone",
        ),
        "cost_ladder_monotonicity": (
            "derived-report-gate-not-structural-validity"
        ),
        "outer_development_conclusion": "conditional",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveOuterEvidenceV1Error(ValueError):
    """Outer folds or their paired cost ladders are not one frozen experiment."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveOuterEvidenceV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _terminal_return(trace: MassiveAdaptiveProfitTraceV1) -> float:
    return trace.final_equity / trace.initial_capital - 1.0


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterCostFoldV1:
    fold_index: int
    selected_checkpoint_receipt_sha256: str
    checkpoint_selection_authority_receipt_sha256: str
    low_cost_trace_receipt_sha256: str
    primary_trace_receipt_sha256: str
    high_cost_trace_receipt_sha256: str
    low_cost_authority_receipt_sha256: str
    primary_authority_receipt_sha256: str
    high_cost_authority_receipt_sha256: str
    decision_target_inventory_sha256: str
    primary_net_returns: tuple[float, ...]
    primary_active_log_returns: tuple[float, ...]
    low_cost_terminal_return: float
    primary_terminal_return: float
    high_cost_terminal_return: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_OUTER_COST_FOLD_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    @property
    def terminal_return_ladder_monotone(self) -> bool:
        """Whether this fold's observed cost ladder satisfies the final gate."""

        return bool(
            self.low_cost_terminal_return
            >= self.primary_terminal_return
            >= self.high_cost_terminal_return
        )

    def validate(self) -> None:
        values = (
            *self.primary_net_returns,
            *self.primary_active_log_returns,
            self.low_cost_terminal_return,
            self.primary_terminal_return,
            self.high_cost_terminal_return,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_OUTER_COST_FOLD_V1_SCHEMA
            or self.fold_index < 0
            or len(self.primary_net_returns)
            != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
            or len(self.primary_active_log_returns)
            != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
            or any(not math.isfinite(value) for value in values)
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOuterEvidenceV1Error(
                "outer cost fold differs from its frozen traces"
            )
        for value in (
            self.selected_checkpoint_receipt_sha256,
            self.checkpoint_selection_authority_receipt_sha256,
            self.low_cost_trace_receipt_sha256,
            self.primary_trace_receipt_sha256,
            self.high_cost_trace_receipt_sha256,
            self.low_cost_authority_receipt_sha256,
            self.primary_authority_receipt_sha256,
            self.high_cost_authority_receipt_sha256,
            self.decision_target_inventory_sha256,
            self.semantic_receipt_sha256,
            self.protocol_receipt_sha256,
        ):
            _digest("outer cost fold", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_outer_cost_fold_v1(
    *,
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    low_cost_trace: MassiveAdaptiveProfitTraceV1,
    primary_trace: MassiveAdaptiveProfitTraceV1,
    high_cost_trace: MassiveAdaptiveProfitTraceV1,
    low_cost_authority: MassiveAdaptiveOuterProfitabilityAuthorityV1,
    primary_authority: MassiveAdaptiveOuterProfitabilityAuthorityV1,
    high_cost_authority: MassiveAdaptiveOuterProfitabilityAuthorityV1,
) -> MassiveAdaptiveOuterCostFoldV1:
    """Derive one fold's paired 10/20/40-bp result from replayed traces."""

    checkpoint_selection.validate()
    traces = (low_cost_trace, primary_trace, high_cost_trace)
    authorities = (low_cost_authority, primary_authority, high_cost_authority)
    for value in (*traces, *authorities):
        value.validate()
    fold = primary_authority.fold_index
    target_inventory = tuple(
        row.decision_target_receipt_sha256 for row in primary_trace.rows
    )
    if (
        checkpoint_selection.selection.selected_checkpoint_receipt_sha256
        != primary_authority.selected_checkpoint_receipt_sha256
        or any(authority.fold_index != fold for authority in authorities)
        or any(
            authority.checkpoint_selection_authority_receipt_sha256
            != checkpoint_selection.semantic_receipt_sha256
            for authority in authorities
        )
        or any(
            authority.trace_receipt_sha256 != trace.semantic_receipt_sha256
            for authority, trace in zip(authorities, traces, strict=True)
        )
        or any(trace.evaluation_role != "outer_test" for trace in traces)
        or tuple(
            trace.transaction_cost_basis_points for trace in traces
        )
        != (
            MASSIVE_ADAPTIVE_OUTER_LOW_COST_BP_V1,
            MASSIVE_ADAPTIVE_OUTER_PRIMARY_COST_BP_V1,
            MASSIVE_ADAPTIVE_OUTER_HIGH_COST_BP_V1,
        )
        or any(
            trace.initial_capital != MASSIVE_ADAPTIVE_OUTER_PRIMARY_CAPITAL_V1
            for trace in traces
        )
        or primary_trace.frozen_actions_replayed
        or any(
            not trace.frozen_actions_replayed
            or trace.frozen_decision_trace_receipt_sha256
            != primary_trace.semantic_receipt_sha256
            for trace in (low_cost_trace, high_cost_trace)
        )
        or any(
            tuple(row.decision_target_receipt_sha256 for row in trace.rows)
            != target_inventory
            for trace in (low_cost_trace, high_cost_trace)
        )
    ):
        raise MassiveAdaptiveOuterEvidenceV1Error(
            "outer fold does not share one frozen selection and target inventory"
        )
    low_return, primary_return, high_return = (
        _terminal_return(trace) for trace in traces
    )
    source_qualified = bool(
        checkpoint_selection.development_checkpoint_selection_authorized
        and all(authority.outer_evaluation_authorized for authority in authorities)
        and all(trace.source_data_qualified for trace in traces)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_OUTER_COST_FOLD_V1_SCHEMA,
        "fold_index": fold,
        "selected_checkpoint_receipt_sha256": (
            primary_authority.selected_checkpoint_receipt_sha256
        ),
        "checkpoint_selection_authority_receipt_sha256": (
            checkpoint_selection.semantic_receipt_sha256
        ),
        "low_cost_trace_receipt_sha256": low_cost_trace.semantic_receipt_sha256,
        "primary_trace_receipt_sha256": primary_trace.semantic_receipt_sha256,
        "high_cost_trace_receipt_sha256": high_cost_trace.semantic_receipt_sha256,
        "low_cost_authority_receipt_sha256": (
            low_cost_authority.semantic_receipt_sha256
        ),
        "primary_authority_receipt_sha256": (
            primary_authority.semantic_receipt_sha256
        ),
        "high_cost_authority_receipt_sha256": (
            high_cost_authority.semantic_receipt_sha256
        ),
        "decision_target_inventory_sha256": semantic_sha256(target_inventory),
        "primary_net_returns": tuple(row.net_return for row in primary_trace.rows),
        "primary_active_log_returns": tuple(
            row.active_log_return for row in primary_trace.rows
        ),
        "low_cost_terminal_return": low_return,
        "primary_terminal_return": primary_return,
        "high_cost_terminal_return": high_return,
        "source_data_qualified": source_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveOuterCostFoldV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _nonwrapping_fold_cluster_lcb(
    fold_values: Sequence[Sequence[float]],
) -> float:
    """One-sided 95% mean LCB with fold and nonwrapping block resampling."""

    folds = tuple(tuple(float(value) for value in row) for row in fold_values)
    if (
        len(folds) != MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
        or any(len(row) != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1 for row in folds)
    ):
        raise MassiveAdaptiveOuterEvidenceV1Error(
            "outer bootstrap requires the exact registered fold geometry"
        )
    rng = random.Random(MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_SEED_V1)
    estimates: list[float] = []
    block = MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_BLOCK_SESSIONS_V1
    for _ in range(MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_REPLICATES_V1):
        sample: list[float] = []
        for _cluster in range(MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1):
            fold = folds[rng.randrange(len(folds))]
            within: list[float] = []
            while len(within) < len(fold):
                start = rng.randrange(0, len(fold) - block + 1)
                within.extend(fold[start : start + block])
            sample.extend(within[: len(fold)])
        estimates.append(mean(sample))
    estimates.sort()
    index = max(0, math.ceil(0.05 * len(estimates)) - 1)
    return estimates[index]


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterEvidenceV1:
    fold_indices: tuple[int, ...]
    fold_receipts: tuple[str, ...]
    fold_inventory_sha256: str
    mean_primary_net_return: float
    mean_primary_active_log_return: float
    mean_high_cost_terminal_return: float
    primary_net_return_lcb95: float
    primary_active_log_return_lcb95: float
    positive_primary_fold_count: int
    cost_ladder_monotone: bool
    passed_gate_names: tuple[str, ...]
    failed_gate_names: tuple[str, ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_development_conclusion_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "outer_development_conclusion_authorized",
            }
        }

    def validate(self) -> None:
        numbers = (
            self.mean_primary_net_return,
            self.mean_primary_active_log_return,
            self.mean_high_cost_terminal_return,
            self.primary_net_return_lcb95,
            self.primary_active_log_return_lcb95,
        )
        expected_passed: list[str] = []
        expected_failed: list[str] = []
        gates = {
            "primary-net-lcb-positive": self.primary_net_return_lcb95 > 0.0,
            "primary-active-lcb-positive": (
                self.primary_active_log_return_lcb95 > 0.0
            ),
            "high-cost-mean-terminal-return-nonnegative": (
                self.mean_high_cost_terminal_return >= 0.0
            ),
            "positive-primary-folds-at-least-three": (
                self.positive_primary_fold_count >= 3
            ),
            "cost-ladder-monotone": self.cost_ladder_monotone,
        }
        for name, passed in gates.items():
            (expected_passed if passed else expected_failed).append(name)
        if (
            self.schema != MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SCHEMA
            or self.fold_indices
            != tuple(range(MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1))
            or len(self.fold_receipts) != MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
            or self.fold_inventory_sha256 != semantic_sha256(self.fold_receipts)
            or any(not math.isfinite(value) for value in numbers)
            or not 0 <= self.positive_primary_fold_count <= len(self.fold_indices)
            or self.passed_gate_names != tuple(sorted(expected_passed))
            or self.failed_gate_names != tuple(sorted(expected_failed))
            or not isinstance(self.source_data_qualified, bool)
            or self.outer_development_conclusion_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SPEC_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOuterEvidenceV1Error(
                "outer statistical evidence differs"
            )
        for value in (
            *self.fold_receipts,
            self.fold_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("outer statistical evidence", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_outer_evidence_v1(
    folds: Sequence[MassiveAdaptiveOuterCostFoldV1],
) -> MassiveAdaptiveOuterEvidenceV1:
    """Aggregate the exact four source-derived folds and apply frozen gates."""

    ordered = tuple(sorted(folds, key=lambda row: row.fold_index))
    for fold in ordered:
        fold.validate()
    if (
        tuple(row.fold_index for row in ordered)
        != tuple(range(MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1))
    ):
        raise MassiveAdaptiveOuterEvidenceV1Error(
            "outer evidence requires exactly folds zero through three"
        )
    primary = tuple(row.primary_net_returns for row in ordered)
    active = tuple(row.primary_active_log_returns for row in ordered)
    mean_primary = mean(value for row in primary for value in row)
    mean_active = mean(value for row in active for value in row)
    # Independent fold wealth paths do not compound.  Their terminal returns are
    # averaged as the frozen high-cost diagnostic.
    mean_high = mean(row.high_cost_terminal_return for row in ordered)
    primary_lcb = _nonwrapping_fold_cluster_lcb(primary)
    active_lcb = _nonwrapping_fold_cluster_lcb(active)
    positive_folds = sum(row.primary_terminal_return > 0.0 for row in ordered)
    ladder = all(row.terminal_return_ladder_monotone for row in ordered)
    gates = {
        "primary-net-lcb-positive": primary_lcb > 0.0,
        "primary-active-lcb-positive": active_lcb > 0.0,
        "high-cost-mean-terminal-return-nonnegative": mean_high >= 0.0,
        "positive-primary-folds-at-least-three": positive_folds >= 3,
        "cost-ladder-monotone": ladder,
    }
    source_qualified = all(row.source_data_qualified for row in ordered)
    body = {
        "schema": MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SCHEMA,
        "fold_indices": tuple(row.fold_index for row in ordered),
        "fold_receipts": tuple(row.semantic_receipt_sha256 for row in ordered),
        "fold_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in ordered)
        ),
        "mean_primary_net_return": mean_primary,
        "mean_primary_active_log_return": mean_active,
        "mean_high_cost_terminal_return": mean_high,
        "primary_net_return_lcb95": primary_lcb,
        "primary_active_log_return_lcb95": active_lcb,
        "positive_primary_fold_count": positive_folds,
        "cost_ladder_monotone": ladder,
        "passed_gate_names": tuple(sorted(name for name, passed in gates.items() if passed)),
        "failed_gate_names": tuple(
            sorted(name for name, passed in gates.items() if not passed)
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SPEC_SHA256,
    }
    result = MassiveAdaptiveOuterEvidenceV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        outer_development_conclusion_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_OUTER_EVIDENCE_V1_SCHEMA",
    "MassiveAdaptiveOuterCostFoldV1",
    "MassiveAdaptiveOuterEvidenceV1",
    "MassiveAdaptiveOuterEvidenceV1Error",
    "build_massive_adaptive_outer_cost_fold_v1",
    "build_massive_adaptive_outer_evidence_v1",
]
