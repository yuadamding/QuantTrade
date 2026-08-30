"""Target-free inference blocks for the adaptive RL-fit chronology.

Unlike the inner-validation inference plan, this plan derives its decisions
only from the fit inventory of the requested outer fold.  The full RL-fit
prefix is the expanding tail of that inventory and is partitioned into fixed
21- or 63-session forecast blocks.  Callers select a block index; they cannot
supply dates, a role, or an arbitrary prefix.

The plan is an inference schedule only.  It never opens targets and grants no
training, profitability, outer, or lockbox authority.
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
    MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1,
)


MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fit-inference-plan-v1"
)
MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_ROW_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fit-inference-row-v1"
)
MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "role": "rl_fit",
        "prefix": "fold-fit-tail-126-times-fold-index-plus-one",
        "block_sessions": (21, 63),
        "dates": "package-derived-no-caller-date-input",
        "context": "causal-consecutive-prefix-capped-by-model-spec-and-504",
        "target_archive": "inaccessible",
        "next_session": "schedule-identity-only-not-fill-authority",
        "profitability_reporting": False,
        "outer": False,
        "lockbox": False,
        "rl": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLFitInferencePlanV1Error(ValueError):
    """The derived RL-fit block is incomplete, noncausal, or mismatched."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFitInferenceRowV1:
    decision_session_date: str
    inference_role: str
    outer_fold_index: int
    block_index: int
    candidate_origin_index: int
    tensor_origin_index: int
    context_session_dates: tuple[str, ...]
    context_tensor_indices: tuple[int, ...]
    origin_output_position: int
    decision_root_receipt_sha256: str
    next_session_date: str
    next_session_schedule_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_ROW_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self, *, maximum_context_sessions: int) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_ROW_V1_SCHEMA
            or self.inference_role != "rl_fit"
            or isinstance(self.outer_fold_index, bool)
            or self.outer_fold_index < 0
            or isinstance(self.block_index, bool)
            or self.block_index < 0
            or isinstance(self.candidate_origin_index, bool)
            or self.candidate_origin_index < 0
            or isinstance(self.tensor_origin_index, bool)
            or self.tensor_origin_index < 0
            or not self.context_session_dates
            or len(self.context_session_dates) != maximum_context_sessions
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
            raise MassiveAdaptiveRLFitInferencePlanV1Error(
                "adaptive RL-fit inference row geometry or receipt differs"
            )
        _digest("adaptive RL-fit decision root", self.decision_root_receipt_sha256)
        _digest(
            "adaptive RL-fit next-session schedule",
            self.next_session_schedule_receipt_sha256,
        )
        _digest("adaptive RL-fit inference row", self.receipt_sha256)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFitInferencePlanV1:
    outer_fold_index: int
    block_index: int
    block_sessions: int
    inference_role: str
    rl_fit_prefix_session_dates: tuple[str, ...]
    rows: tuple[MassiveAdaptiveRLFitInferenceRowV1, ...]
    decision_tensor_receipt_sha256: str
    full_decision_root_inventory_sha256: str
    origin_decision_root_inventory_sha256: str
    split_plan_receipt_sha256: str
    session_authority_receipt_sha256: str
    model_spec_receipt_sha256: str
    maximum_context_sessions: int
    rl_fit_prefix_inventory_sha256: str
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
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SCHEMA

    @property
    def fold_index(self) -> int:
        """Compatibility identity used by forecast and economic replay."""

        return self.outer_fold_index

    @property
    def origin_session_dates(self) -> tuple[str, ...]:
        return tuple(row.decision_session_date for row in self.rows)

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for row in self.rows:
            row.validate(maximum_context_sessions=self.maximum_context_sessions)
        expected_prefix_sessions = MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1 * (
            self.outer_fold_index + 1
        )
        expected_block_count = expected_prefix_sessions // self.block_sessions
        block_dates = tuple(row.decision_session_date for row in self.rows)
        start = self.block_index * self.block_sessions
        stop = start + self.block_sessions
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SCHEMA
            or self.inference_role != "rl_fit"
            or self.block_sessions not in {21, 63}
            or expected_prefix_sessions % self.block_sessions != 0
            or self.block_index < 0
            or self.block_index >= expected_block_count
            or len(self.rl_fit_prefix_session_dates) != expected_prefix_sessions
            or self.rl_fit_prefix_session_dates
            != tuple(sorted(set(self.rl_fit_prefix_session_dates)))
            or block_dates != self.rl_fit_prefix_session_dates[start:stop]
            or len(block_dates) != self.block_sessions
            or any(row.outer_fold_index != self.outer_fold_index for row in self.rows)
            or any(row.block_index != self.block_index for row in self.rows)
            or any(row.inference_role != self.inference_role for row in self.rows)
            or not 0
            < self.maximum_context_sessions
            <= MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1
            or self.rl_fit_prefix_inventory_sha256
            != semantic_sha256(self.rl_fit_prefix_session_dates)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.next_session_schedule_inventory_sha256
            != semantic_sha256(
                tuple(row.next_session_schedule_receipt_sha256 for row in self.rows)
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SOURCE_SHA256
            or not self.source_schedule_replayed
            or self.development_inference_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFitInferencePlanV1Error(
                "adaptive RL-fit inference-plan identity or chronology differs"
            )
        for value in (
            self.decision_tensor_receipt_sha256,
            self.full_decision_root_inventory_sha256,
            self.origin_decision_root_inventory_sha256,
            self.split_plan_receipt_sha256,
            self.session_authority_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.rl_fit_prefix_inventory_sha256,
            self.row_inventory_sha256,
            self.next_session_schedule_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL-fit inference plan", value)
        assert_no_adaptive_hold_semantics(asdict(self))


def build_massive_adaptive_rl_fit_inference_plan_v1(
    *,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    split_plan: MassiveAdaptiveSplitPlanV1,
    outer_fold_index: int,
    block_index: int,
    block_sessions: int,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveRLFitInferencePlanV1:
    """Build one package-derived block from the expanding RL-fit prefix."""

    decision_tensor.validate()
    split_plan.validate()
    model_spec.validate()
    if (
        decision_tensor.runtime_tensor is None
        or not decision_tensor.runtime_source_replayed
    ):
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit inference tensor has not been source replayed"
        )
    if block_sessions not in {21, 63}:
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit blocks must contain 21 or 63 sessions"
        )
    try:
        fold = split_plan.outer_folds[outer_fold_index]
    except (IndexError, TypeError) as exc:
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit outer fold is absent"
        ) from exc

    prefix_count = MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1 * (outer_fold_index + 1)
    if len(fold.fit_session_dates) < prefix_count or prefix_count % block_sessions:
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit prefix cannot be partitioned by the registered block size"
        )
    prefix_dates = fold.fit_session_dates[-prefix_count:]
    block_count = prefix_count // block_sessions
    if isinstance(block_index, bool) or block_index < 0 or block_index >= block_count:
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit block index is outside the derived prefix"
        )
    block_start = block_index * block_sessions
    role_dates = prefix_dates[block_start : block_start + block_sessions]

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
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit roots do not bind the tensor and split plan"
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
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit tensor date is absent from the candidate schedule"
        ) from exc
    if any(
        right != left + 1
        for left, right in zip(
            tensor_candidate_indices, tensor_candidate_indices[1:], strict=False
        )
    ):
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit tensor dates are not consecutive sessions"
        )
    tensor_date_set = set(runtime.decision_session_dates)
    if any(date not in tensor_date_set for date in role_dates):
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit tensor does not cover the complete derived block"
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
    if tensor_index_by_date[role_dates[0]] < maximum_context - 1:
        raise MassiveAdaptiveRLFitInferencePlanV1Error(
            "adaptive RL-fit tensor omits the registered causal context"
        )

    rows: list[MassiveAdaptiveRLFitInferenceRowV1] = []
    for session_date in role_dates:
        tensor_index = tensor_index_by_date[session_date]
        origin_index = candidate_index[session_date]
        if origin_index + 1 >= len(split_plan.candidate_session_dates):
            raise MassiveAdaptiveRLFitInferencePlanV1Error(
                "adaptive RL-fit decision has no following exchange session"
            )
        context_start = tensor_index - maximum_context + 1
        context_indices = tuple(range(context_start, tensor_index + 1))
        next_session_date = split_plan.candidate_session_dates[origin_index + 1]
        next_schedule = semantic_sha256(
            {
                "session_authority": split_plan.session_authority_receipt_sha256,
                "decision_session_date": session_date,
                "next_session_date": next_session_date,
                "fill_authority": False,
            }
        )
        row_body = {
            "schema": MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_ROW_V1_SCHEMA,
            "decision_session_date": session_date,
            "inference_role": "rl_fit",
            "outer_fold_index": outer_fold_index,
            "block_index": block_index,
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
        row = MassiveAdaptiveRLFitInferenceRowV1(
            **row_body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(row_body),
        )
        row.validate(maximum_context_sessions=maximum_context)
        rows.append(row)

    full_root_receipts = tuple(root.semantic_receipt_sha256 for root in ordered_roots)
    origin_root_receipts = tuple(row.decision_root_receipt_sha256 for row in rows)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SCHEMA,
        "outer_fold_index": outer_fold_index,
        "block_index": block_index,
        "block_sessions": block_sessions,
        "inference_role": "rl_fit",
        "rl_fit_prefix_session_dates": prefix_dates,
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
        "rl_fit_prefix_inventory_sha256": semantic_sha256(prefix_dates),
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
        "specification_sha256": (MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SPEC_SHA256),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SOURCE_SHA256
        ),
        "source_schedule_replayed": True,
        "development_inference_authorized": False,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveAdaptiveRLFitInferencePlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(
            {**body, "rows": tuple(asdict(row) for row in rows)}
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FIT_INFERENCE_PLAN_V1_SCHEMA",
    "MassiveAdaptiveRLFitInferencePlanV1",
    "MassiveAdaptiveRLFitInferencePlanV1Error",
    "MassiveAdaptiveRLFitInferenceRowV1",
    "build_massive_adaptive_rl_fit_inference_plan_v1",
]
