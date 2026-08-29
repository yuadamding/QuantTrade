"""Selection-gated target-free outer inference schedule for adaptive profit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MassiveAdaptiveAlphaModelSpecV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import MassiveAdaptiveCheckpointV1
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1,
)

if TYPE_CHECKING:
    from rl_quant.training.massive_adaptive_profit_checkpoint_selection_authority_v2 import (
        MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    )

MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SCHEMA = (
    "rl-quant.massive-adaptive-outer-inference-plan-v1"
)
MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "role": "complete-outer-test-chronology",
        "checkpoint": "frozen-source-derived-inner-validation-selection-v2",
        "target_archive": "inaccessible",
        "context": "causal-consecutive-prefix-capped-at-504",
        "selection_after_outer": False,
        "lockbox": False,
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveAdaptiveOuterInferencePlanV1Error(ValueError):
    """Outer chronology was opened before source-qualified selection froze."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterInferenceRowV1:
    decision_session_date: str
    fold_index: int
    candidate_origin_index: int
    tensor_origin_index: int
    context_session_dates: tuple[str, ...]
    context_tensor_indices: tuple[int, ...]
    origin_output_position: int
    decision_root_receipt_sha256: str
    next_session_date: str
    next_session_schedule_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self, *, maximum_context_sessions: int) -> None:
        if (
            not self.decision_session_date
            or self.fold_index < 0
            or self.candidate_origin_index < 0
            or self.tensor_origin_index < 0
            or len(self.context_session_dates) != maximum_context_sessions
            or len(self.context_tensor_indices) != maximum_context_sessions
            or self.context_tensor_indices
            != tuple(
                range(
                    self.context_tensor_indices[0],
                    self.context_tensor_indices[-1] + 1,
                )
            )
            or self.context_tensor_indices[-1] != self.tensor_origin_index
            or self.origin_output_position != maximum_context_sessions - 1
            or self.next_session_date <= self.decision_session_date
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveOuterInferencePlanV1Error(
                "outer inference row geometry differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterInferencePlanV1:
    fold_index: int
    rows: tuple[MassiveAdaptiveOuterInferenceRowV1, ...]
    selected_checkpoint_receipt_sha256: str
    checkpoint_selection_receipt_sha256: str
    decision_tensor_receipt_sha256: str
    full_decision_root_inventory_sha256: str
    origin_decision_root_inventory_sha256: str
    split_plan_receipt_sha256: str
    session_authority_receipt_sha256: str
    model_spec_receipt_sha256: str
    maximum_context_sessions: int
    row_inventory_sha256: str
    next_session_schedule_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_inference_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "outer_inference_authorized"}
        }

    def validate(self) -> None:
        for row in self.rows:
            row.validate(maximum_context_sessions=self.maximum_context_sessions)
        if (
            self.schema != MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SCHEMA
            or not self.rows
            or tuple(row.decision_session_date for row in self.rows)
            != tuple(sorted(set(row.decision_session_date for row in self.rows)))
            or any(row.fold_index != self.fold_index for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.next_session_schedule_inventory_sha256
            != semantic_sha256(
                tuple(row.next_session_schedule_receipt_sha256 for row in self.rows)
            )
            or not isinstance(self.source_data_qualified, bool)
            or not self.source_data_qualified
            or not self.outer_inference_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOuterInferencePlanV1Error(
                "outer inference plan identity or authority differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_outer_inference_plan_v1(
    *,
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    split_plan: MassiveAdaptiveSplitPlanV1,
    fold_index: int,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveOuterInferencePlanV1:
    """Open the complete outer chronology only after inner selection freezes."""

    checkpoint_selection.validate()
    selection = checkpoint_selection.selection
    selected_checkpoint.validate()
    decision_tensor.validate()
    split_plan.validate()
    model_spec.validate()
    if (
        not checkpoint_selection.runtime_selection_replayed
        or not checkpoint_selection.development_checkpoint_selection_authorized
        or selection.selected_checkpoint_receipt_sha256
        != selected_checkpoint.semantic_receipt_sha256
        or not selected_checkpoint.development_training_authorized
        or selected_checkpoint.split_plan_receipt_sha256
        != split_plan.semantic_receipt_sha256
        or selected_checkpoint.model_spec_receipt_sha256 != model_spec.receipt_sha256
        or decision_tensor.runtime_tensor is None
        or not decision_tensor.model_input_authorized
    ):
        raise MassiveAdaptiveOuterInferencePlanV1Error(
            "outer inference requires the frozen source-qualified checkpoint selection"
        )
    try:
        fold = split_plan.outer_folds[fold_index]
    except (IndexError, TypeError) as exc:
        raise MassiveAdaptiveOuterInferencePlanV1Error("outer fold is absent") from exc
    ordered_roots = tuple(
        sorted(decision_roots, key=lambda row: row.decision_session_date)
    )
    for root in ordered_roots:
        root.validate()
    runtime = decision_tensor.runtime_tensor
    if (
        tuple(root.decision_session_date for root in ordered_roots)
        != decision_tensor.decision_session_dates
        or tuple(root.feature_semantic_receipt_sha256 for root in ordered_roots)
        != decision_tensor.feature_semantic_receipts
        or tuple(root.action_origin_receipt_sha256 for root in ordered_roots)
        != decision_tensor.action_origin_receipts
        or any(not root.source_data_qualified for root in ordered_roots)
    ):
        raise MassiveAdaptiveOuterInferencePlanV1Error(
            "outer roots do not qualify the decision tensor"
        )
    candidates = split_plan.candidate_session_dates
    candidate_index = {value: index for index, value in enumerate(candidates)}
    tensor_index = {
        value: index for index, value in enumerate(runtime.decision_session_dates)
    }
    role_dates = fold.outer_test_session_dates
    if any(value not in tensor_index for value in role_dates):
        raise MassiveAdaptiveOuterInferencePlanV1Error(
            "outer tensor omits a complete test chronology"
        )
    maximum_context = min(
        model_spec.maximum_context_sessions,
        MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    )
    root_by_date = {row.decision_session_date: row for row in ordered_roots}
    rows: list[MassiveAdaptiveOuterInferenceRowV1] = []
    for session_date in role_dates:
        origin = tensor_index[session_date]
        start = origin - maximum_context + 1
        candidate = candidate_index[session_date]
        if start < 0 or candidate + 1 >= len(candidates):
            raise MassiveAdaptiveOuterInferencePlanV1Error(
                "outer origin lacks context or a following session"
            )
        context_indices = tuple(range(start, origin + 1))
        context_dates = tuple(runtime.decision_session_dates[index] for index in context_indices)
        next_date = candidates[candidate + 1]
        schedule = semantic_sha256(
            {
                "session_authority": split_plan.session_authority_receipt_sha256,
                "decision_session_date": session_date,
                "next_session_date": next_date,
            }
        )
        body = {
            "decision_session_date": session_date,
            "fold_index": fold_index,
            "candidate_origin_index": candidate,
            "tensor_origin_index": origin,
            "context_session_dates": context_dates,
            "context_tensor_indices": context_indices,
            "origin_output_position": maximum_context - 1,
            "decision_root_receipt_sha256": root_by_date[
                session_date
            ].semantic_receipt_sha256,
            "next_session_date": next_date,
            "next_session_schedule_receipt_sha256": schedule,
        }
        rows.append(
            MassiveAdaptiveOuterInferenceRowV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    root_inventory = semantic_sha256(
        tuple(root_by_date[value].semantic_receipt_sha256 for value in role_dates)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SCHEMA,
        "fold_index": fold_index,
        "rows": tuple(rows),
        "selected_checkpoint_receipt_sha256": selected_checkpoint.semantic_receipt_sha256,
        "checkpoint_selection_receipt_sha256": checkpoint_selection.semantic_receipt_sha256,
        "decision_tensor_receipt_sha256": decision_tensor.semantic_receipt_sha256,
        "full_decision_root_inventory_sha256": semantic_sha256(
            tuple(root.semantic_receipt_sha256 for root in ordered_roots)
        ),
        "origin_decision_root_inventory_sha256": root_inventory,
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "session_authority_receipt_sha256": split_plan.session_authority_receipt_sha256,
        "model_spec_receipt_sha256": model_spec.receipt_sha256,
        "maximum_context_sessions": maximum_context,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "next_session_schedule_inventory_sha256": semantic_sha256(
            tuple(row.next_session_schedule_receipt_sha256 for row in rows)
        ),
        "source_data_qualified": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_OUTER_INFERENCE_PLAN_V1_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveOuterInferencePlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        outer_inference_authorized=True,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveOuterInferencePlanV1",
    "MassiveAdaptiveOuterInferencePlanV1Error",
    "MassiveAdaptiveOuterInferenceRowV1",
    "build_massive_adaptive_outer_inference_plan_v1",
]
