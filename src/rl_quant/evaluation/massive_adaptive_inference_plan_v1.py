"""Target-free causal inference schedules for adaptive alpha forecasts.

The supervised window plan deliberately requires a mature 126-session target
and therefore cannot represent a complete validation trading chronology.  An
inference plan has a narrower responsibility: select every decision in one
allowed role, bind its causal model context, and identify the following
exchange session.  It never opens a target archive and does not authorize
fills, profitability, lockbox access, or reinforcement learning.
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
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MassiveAdaptiveAlphaModelSpecV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1,
)


MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SCHEMA = (
    "rl-quant.massive-adaptive-inference-plan-v1"
)
MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "role": "complete-inner-validation-chronology-only",
        "context": "causal-consecutive-prefix-capped-by-model-spec-and-504",
        "target_archive": "inaccessible",
        "target_maturity": "not-required",
        "next_session": "schedule-identity-only-not-fill-authority",
        "outer": False,
        "lockbox": False,
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveAdaptiveInferencePlanV1Error(ValueError):
    """An adaptive inference schedule is incomplete, noncausal, or mismatched."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveInferencePlanV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveInferenceRowV1:
    decision_session_date: str
    inference_role: str
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
            self.inference_role != "inner_validation"
            or isinstance(self.fold_index, bool)
            or self.fold_index < 0
            or isinstance(self.candidate_origin_index, bool)
            or self.candidate_origin_index < 0
            or isinstance(self.tensor_origin_index, bool)
            or self.tensor_origin_index < 0
            or not self.context_session_dates
            or len(self.context_session_dates) > maximum_context_sessions
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
            or not self.next_session_date
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveInferencePlanV1Error(
                "adaptive inference row geometry or receipt differs"
            )
        _digest("adaptive inference decision root", self.decision_root_receipt_sha256)
        _digest(
            "adaptive inference next-session schedule",
            self.next_session_schedule_receipt_sha256,
        )
        _digest("adaptive inference row", self.receipt_sha256)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveInferencePlanV1:
    fold_index: int
    inference_role: str
    rows: tuple[MassiveAdaptiveInferenceRowV1, ...]
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
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    source_schedule_replayed: bool
    development_inference_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for row in self.rows:
            row.validate(maximum_context_sessions=self.maximum_context_sessions)
        if (
            self.schema != MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SCHEMA
            or self.inference_role != "inner_validation"
            or not self.rows
            or any(row.fold_index != self.fold_index for row in self.rows)
            or any(row.inference_role != self.inference_role for row in self.rows)
            or tuple(row.decision_session_date for row in self.rows)
            != tuple(sorted(set(row.decision_session_date for row in self.rows)))
            or not 0
            < self.maximum_context_sessions
            <= MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.next_session_schedule_inventory_sha256
            != semantic_sha256(
                tuple(row.next_session_schedule_receipt_sha256 for row in self.rows)
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SOURCE_SHA256
            or not self.source_schedule_replayed
            or self.development_inference_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveInferencePlanV1Error(
                "adaptive inference-plan identity or authorization differs"
            )
        for value in (
            self.decision_tensor_receipt_sha256,
            self.full_decision_root_inventory_sha256,
            self.origin_decision_root_inventory_sha256,
            self.split_plan_receipt_sha256,
            self.session_authority_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.row_inventory_sha256,
            self.next_session_schedule_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive inference plan", value)
        assert_no_adaptive_hold_semantics(asdict(self))


def build_massive_adaptive_inference_plan_v1(
    *,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    split_plan: MassiveAdaptiveSplitPlanV1,
    fold_index: int,
    inference_role: str,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveInferencePlanV1:
    """Build the complete target-free inner-validation decision schedule."""

    decision_tensor.validate()
    split_plan.validate()
    model_spec.validate()
    if (
        decision_tensor.runtime_tensor is None
        or not decision_tensor.runtime_source_replayed
    ):
        raise MassiveAdaptiveInferencePlanV1Error(
            "adaptive inference tensor has not been source replayed"
        )
    if inference_role != "inner_validation":
        raise MassiveAdaptiveInferencePlanV1Error(
            "only target-free inner-validation inference is authorized in v1"
        )
    try:
        fold = split_plan.outer_folds[fold_index]
    except (IndexError, TypeError) as exc:
        raise MassiveAdaptiveInferencePlanV1Error(
            "adaptive inference fold is absent"
        ) from exc

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
        or any(
            root.session_authority_receipt_sha256
            != split_plan.session_authority_receipt_sha256
            for root in ordered_roots
        )
    ):
        raise MassiveAdaptiveInferencePlanV1Error(
            "adaptive inference roots do not bind the tensor and split plan"
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
        raise MassiveAdaptiveInferencePlanV1Error(
            "adaptive inference tensor date is absent from the candidate schedule"
        ) from exc
    if any(
        right != left + 1
        for left, right in zip(
            tensor_candidate_indices,
            tensor_candidate_indices[1:],
            strict=False,
        )
    ):
        raise MassiveAdaptiveInferencePlanV1Error(
            "adaptive inference tensor dates are not consecutive sessions"
        )
    role_dates = fold.inner_validation_session_dates
    tensor_date_set = set(runtime.decision_session_dates)
    if any(date not in tensor_date_set for date in role_dates):
        raise MassiveAdaptiveInferencePlanV1Error(
            "adaptive inference tensor does not cover the complete validation role"
        )

    root_by_date = {root.decision_session_date: root for root in ordered_roots}
    tensor_index_by_date = {
        session_date: index
        for index, session_date in enumerate(runtime.decision_session_dates)
    }
    maximum_context = min(
        model_spec.maximum_context_sessions,
        MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    )
    first_role_tensor_index = tensor_index_by_date[role_dates[0]]
    if first_role_tensor_index < maximum_context - 1:
        raise MassiveAdaptiveInferencePlanV1Error(
            "adaptive inference tensor omits the registered causal context"
        )
    rows: list[MassiveAdaptiveInferenceRowV1] = []
    for session_date in role_dates:
        tensor_index = tensor_index_by_date[session_date]
        origin_index = candidate_index[session_date]
        if origin_index + 1 >= len(split_plan.candidate_session_dates):
            raise MassiveAdaptiveInferencePlanV1Error(
                "adaptive inference decision has no following exchange session"
            )
        context_start = max(0, tensor_index - maximum_context + 1)
        context_indices = tuple(range(context_start, tensor_index + 1))
        if len(context_indices) != maximum_context:
            raise MassiveAdaptiveInferencePlanV1Error(
                "adaptive inference row omits the registered causal context"
            )
        next_session_date = split_plan.candidate_session_dates[origin_index + 1]
        next_schedule = semantic_sha256(
            {
                "session_authority": split_plan.session_authority_receipt_sha256,
                "decision_session_date": session_date,
                "next_session_date": next_session_date,
                "fill_authority": False,
            }
        )
        body = {
            "decision_session_date": session_date,
            "inference_role": inference_role,
            "fold_index": fold_index,
            "candidate_origin_index": origin_index,
            "tensor_origin_index": tensor_index,
            "context_session_dates": tuple(
                runtime.decision_session_dates[index] for index in context_indices
            ),
            "context_tensor_indices": context_indices,
            "origin_output_position": len(context_indices) - 1,
            "decision_root_receipt_sha256": root_by_date[
                session_date
            ].semantic_receipt_sha256,
            "next_session_date": next_session_date,
            "next_session_schedule_receipt_sha256": next_schedule,
        }
        row = MassiveAdaptiveInferenceRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate(maximum_context_sessions=maximum_context)
        rows.append(row)

    full_root_receipts = tuple(root.semantic_receipt_sha256 for root in ordered_roots)
    origin_root_receipts = tuple(row.decision_root_receipt_sha256 for row in rows)
    body = {
        "schema": MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SCHEMA,
        "fold_index": fold_index,
        "inference_role": inference_role,
        "rows": tuple(rows),
        "decision_tensor_receipt_sha256": decision_tensor.semantic_receipt_sha256,
        "full_decision_root_inventory_sha256": semantic_sha256(full_root_receipts),
        "origin_decision_root_inventory_sha256": semantic_sha256(origin_root_receipts),
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "session_authority_receipt_sha256": (
            split_plan.session_authority_receipt_sha256
        ),
        "model_spec_receipt_sha256": model_spec.receipt_sha256,
        "maximum_context_sessions": maximum_context,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "next_session_schedule_inventory_sha256": semantic_sha256(
            tuple(row.next_session_schedule_receipt_sha256 for row in rows)
        ),
        "source_data_qualified": (
            decision_tensor.committed_source_data_qualified
            and split_plan.candidate_source_data_qualified
            and all(root.source_data_qualified for root in ordered_roots)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SOURCE_SHA256
        ),
        "source_schedule_replayed": True,
        "development_inference_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic_body = {**body, "rows": tuple(asdict(row) for row in rows)}
    result = MassiveAdaptiveInferencePlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(semantic_body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_INFERENCE_PLAN_V1_SCHEMA",
    "MassiveAdaptiveInferencePlanV1",
    "MassiveAdaptiveInferencePlanV1Error",
    "MassiveAdaptiveInferenceRowV1",
    "build_massive_adaptive_inference_plan_v1",
]
