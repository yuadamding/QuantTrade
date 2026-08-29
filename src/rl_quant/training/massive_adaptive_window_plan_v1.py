"""Package-derived causal windows for adaptive alpha training.

Window indices are derived from the committed decision tensor and frozen split
plan.  Callers select a committed row; they cannot supply tensor indices,
origin indices, split bounds, or a free split-plan receipt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1,
)


MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SCHEMA = "rl-quant.massive-adaptive-window-plan-v1"
MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "indices": "derived-from-split-candidate-and-decision-tensor-inventories",
        "context": "up-to-504-consecutive-prior-and-current-decisions",
        "target_maturity": "origin-plus-126-strictly-before-role-boundary",
        "roles": ("training", "inner_validation"),
        "duration_prior": False,
        "downstream_authorization": False,
    }
)


class MassiveAdaptiveWindowPlanV1Error(ValueError):
    """Adaptive training windows are not causal or split safe."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveWindowRowV1:
    origin_session_date: str
    split_role: str
    candidate_origin_index: int
    tensor_origin_index: int
    target_stop_exclusive_index: int
    context_session_dates: tuple[str, ...]
    context_tensor_indices: tuple[int, ...]
    origin_output_position: int
    decision_root_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.split_role not in {"training", "inner_validation"}
            or isinstance(self.candidate_origin_index, bool)
            or self.candidate_origin_index < 0
            or isinstance(self.tensor_origin_index, bool)
            or self.tensor_origin_index < 0
            or self.candidate_origin_index
            + MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1
            >= self.target_stop_exclusive_index
            or not self.context_session_dates
            or len(self.context_session_dates)
            > MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1
            or len(self.context_session_dates) != len(self.context_tensor_indices)
            or self.context_tensor_indices
            != tuple(
                range(
                    self.context_tensor_indices[0],
                    self.context_tensor_indices[-1] + 1,
                )
            )
            or self.context_tensor_indices[-1] != self.tensor_origin_index
            or self.origin_output_position != len(self.context_tensor_indices) - 1
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveWindowPlanV1Error(
                "adaptive window row geometry or receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveWindowPlanV1:
    fold_index: int
    split_role: str
    rows: tuple[MassiveAdaptiveWindowRowV1, ...]
    decision_tensor_receipt_sha256: str
    full_decision_root_inventory_sha256: str
    origin_decision_root_inventory_sha256: str
    split_plan_receipt_sha256: str
    row_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    source_windows_replayed: bool
    development_training_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for row in self.rows:
            row.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SCHEMA
            or not self.rows
            or self.split_role not in {"training", "inner_validation"}
            or any(row.split_role != self.split_role for row in self.rows)
            or tuple(row.origin_session_date for row in self.rows)
            != tuple(sorted(set(row.origin_session_date for row in self.rows)))
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SOURCE_SHA256
            or not self.source_windows_replayed
            or self.development_training_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveWindowPlanV1Error(
                "adaptive window-plan identity or authorization differs"
            )
        assert_no_adaptive_hold_semantics(asdict(self))

    def row(self, origin_session_date: str) -> MassiveAdaptiveWindowRowV1:
        matches = tuple(
            row for row in self.rows if row.origin_session_date == origin_session_date
        )
        if len(matches) != 1:
            raise MassiveAdaptiveWindowPlanV1Error(
                "adaptive window origin is absent or duplicated"
            )
        return matches[0]


def build_massive_adaptive_window_plan_v1(
    *,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    split_plan: MassiveAdaptiveSplitPlanV1,
    fold_index: int,
    split_role: str,
) -> MassiveAdaptiveWindowPlanV1:
    """Derive every eligible origin window for one fold and training role."""

    decision_tensor.validate()
    split_plan.validate()
    if decision_tensor.runtime_tensor is None or not decision_tensor.runtime_source_replayed:
        raise MassiveAdaptiveWindowPlanV1Error(
            "adaptive decision tensor has not been source replayed"
        )
    if split_role not in {"training", "inner_validation"}:
        raise MassiveAdaptiveWindowPlanV1Error("adaptive window role is unsupported")
    try:
        fold = split_plan.outer_folds[fold_index]
    except (IndexError, TypeError) as exc:
        raise MassiveAdaptiveWindowPlanV1Error(
            "adaptive window fold is absent"
        ) from exc
    ordered_roots = tuple(
        sorted(decision_roots, key=lambda row: row.decision_session_date)
    )
    for row in ordered_roots:
        row.validate()
    runtime = decision_tensor.runtime_tensor
    if (
        tuple(row.decision_session_date for row in ordered_roots)
        != decision_tensor.decision_session_dates
        or tuple(row.feature_semantic_receipt_sha256 for row in ordered_roots)
        != decision_tensor.feature_semantic_receipts
        or tuple(row.action_origin_receipt_sha256 for row in ordered_roots)
        != decision_tensor.action_origin_receipts
        or any(
            row.session_authority_receipt_sha256
            != split_plan.session_authority_receipt_sha256
            for row in ordered_roots
        )
    ):
        raise MassiveAdaptiveWindowPlanV1Error(
            "adaptive decision roots do not bind the tensor and split plan"
        )
    candidate_index = {
        session_date: index
        for index, session_date in enumerate(split_plan.candidate_session_dates)
    }
    try:
        tensor_candidate_indices = tuple(
            candidate_index[date] for date in runtime.decision_session_dates
        )
    except KeyError as exc:
        raise MassiveAdaptiveWindowPlanV1Error(
            "adaptive tensor date is absent from the split candidate inventory"
        ) from exc
    if any(
        right != left + 1
        for left, right in zip(
            tensor_candidate_indices,
            tensor_candidate_indices[1:],
            strict=False,
        )
    ):
        raise MassiveAdaptiveWindowPlanV1Error(
            "adaptive tensor dates are not consecutive candidate sessions"
        )
    role_dates = (
        fold.fit_session_dates
        if split_role == "training"
        else fold.inner_validation_session_dates
    )
    target_stop = (
        fold.fit_target_stop_exclusive_index
        if split_role == "training"
        else fold.validation_target_stop_exclusive_index
    )
    root_by_date = {row.decision_session_date: row for row in ordered_roots}
    rows: list[MassiveAdaptiveWindowRowV1] = []
    for tensor_index, session_date in enumerate(runtime.decision_session_dates):
        origin_index = tensor_candidate_indices[tensor_index]
        if (
            session_date not in role_dates
            or origin_index + MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1
            >= target_stop
        ):
            continue
        context_start = max(
            0,
            tensor_index - MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1 + 1,
        )
        context_indices = tuple(range(context_start, tensor_index + 1))
        body = {
            "origin_session_date": session_date,
            "split_role": split_role,
            "candidate_origin_index": origin_index,
            "tensor_origin_index": tensor_index,
            "target_stop_exclusive_index": target_stop,
            "context_session_dates": tuple(
                runtime.decision_session_dates[index] for index in context_indices
            ),
            "context_tensor_indices": context_indices,
            "origin_output_position": len(context_indices) - 1,
            "decision_root_receipt_sha256": root_by_date[
                session_date
            ].semantic_receipt_sha256,
        }
        window_row = MassiveAdaptiveWindowRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        window_row.validate()
        rows.append(window_row)
    if not rows:
        raise MassiveAdaptiveWindowPlanV1Error(
            "adaptive split has no mature source-backed model windows"
        )
    root_inventory = semantic_sha256(
        tuple(row.semantic_receipt_sha256 for row in ordered_roots)
    )
    origin_root_inventory = semantic_sha256(
        tuple(row.decision_root_receipt_sha256 for row in rows)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SCHEMA,
        "fold_index": fold_index,
        "split_role": split_role,
        "rows": tuple(rows),
        "decision_tensor_receipt_sha256": decision_tensor.semantic_receipt_sha256,
        "full_decision_root_inventory_sha256": root_inventory,
        "origin_decision_root_inventory_sha256": origin_root_inventory,
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SOURCE_SHA256,
        "source_windows_replayed": True,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic_body = {**body, "rows": tuple(asdict(row) for row in rows)}
    result = MassiveAdaptiveWindowPlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(semantic_body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SCHEMA",
    "MassiveAdaptiveWindowPlanV1",
    "MassiveAdaptiveWindowPlanV1Error",
    "MassiveAdaptiveWindowRowV1",
    "build_massive_adaptive_window_plan_v1",
]
