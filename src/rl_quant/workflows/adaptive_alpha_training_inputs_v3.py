"""Build one adaptive training example from reconciled source authorities.

Unlike the engineering-only V2 adapter, this path accepts the replay wrapper
for 127-mark economic targets and package-derived window rows.  It owns the
model-output slice and never accepts caller-provided origin or tensor indices.
The package-owned trainer owns the preceding model forward call.
"""

from __future__ import annotations

import torch

from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.features.massive_adaptive_source_targets_v1 import (
    MassiveAdaptiveSourceTargetsV1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MassiveAdaptiveAlphaSequenceOutputV1,
)
from rl_quant.models.alpha_hierarchical import AlphaDistribution
from rl_quant.training.adaptive_alpha_supervised_v1 import (
    MassiveAdaptiveAlphaSupervisedV1Error,
    MassiveAdaptiveAlphaTrainingBatchV1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MassiveAdaptiveSplitPlanV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
    MassiveAdaptiveWindowRowV1,
)
from rl_quant.workflows.adaptive_alpha_training_inputs_v2 import (
    build_massive_adaptive_alpha_training_batch_v2,
)


def _select_distribution(
    value: AlphaDistribution, positions: torch.Tensor
) -> AlphaDistribution:
    return AlphaDistribution(
        mean=value.mean.index_select(1, positions),
        downside_quantile=value.downside_quantile.index_select(1, positions),
        median=value.median.index_select(1, positions),
        upside_quantile=value.upside_quantile.index_select(1, positions),
        scale=value.scale.index_select(1, positions),
    )


def _select_output(
    output: MassiveAdaptiveAlphaSequenceOutputV1,
    positions: torch.Tensor,
) -> MassiveAdaptiveAlphaSequenceOutputV1:
    return MassiveAdaptiveAlphaSequenceOutputV1(
        residual_distribution=_select_distribution(
            output.residual_distribution, positions
        ),
        raw_distribution=_select_distribution(output.raw_distribution, positions),
        factor_return_mean=output.factor_return_mean.index_select(1, positions),
        executable_score=output.executable_score.index_select(1, positions),
        bucket_router_weights=output.bucket_router_weights.index_select(1, positions),
        router_weights=output.router_weights.index_select(1, positions),
        stock_context=output.stock_context.index_select(1, positions),
        market_context=output.market_context.index_select(1, positions),
        valid=output.valid.index_select(1, positions),
    )


def build_massive_adaptive_alpha_training_batch_v3(
    *,
    full_window_output: MassiveAdaptiveAlphaSequenceOutputV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_root: MassiveAdaptiveDecisionRootV1,
    source_target: MassiveAdaptiveSourceTargetsV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
    window_plan: MassiveAdaptiveWindowPlanV1,
    window_row: MassiveAdaptiveWindowRowV1,
) -> MassiveAdaptiveAlphaTrainingBatchV1:
    """Align one trainer-owned output with one root-replayed target origin."""

    decision_tensor.validate()
    decision_root.validate()
    source_target.validate()
    split_plan.validate()
    window_plan.validate()
    window_row.validate()
    runtime = decision_tensor.runtime_tensor
    if runtime is None or not decision_tensor.runtime_source_replayed:
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive V3 decision tensor has not been package replayed"
        )
    if not isinstance(source_target, MassiveAdaptiveSourceTargetsV1):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive V3 training requires source-replayed target wrappers"
        )
    if (
        window_plan.fold_index >= len(split_plan.outer_folds)
        or window_plan.split_plan_receipt_sha256
        != split_plan.semantic_receipt_sha256
        or window_plan.decision_tensor_receipt_sha256
        != decision_tensor.semantic_receipt_sha256
        or window_row not in window_plan.rows
        or window_row.decision_root_receipt_sha256
        != decision_root.semantic_receipt_sha256
        or window_row.origin_session_date != decision_root.decision_session_date
        or source_target.decision_session_date
        != decision_root.decision_session_date
        or source_target.security_ids != decision_root.action_security_ids
        or source_target.origin_authority_receipt_sha256
        != decision_root.action_origin_receipt_sha256
        or source_target.decision_clock_receipt_sha256
        != decision_root.decision_clock_receipt_sha256
        or source_target.session_authority_receipt_sha256
        != decision_root.session_authority_receipt_sha256
        or source_target.targets.origin_receipt_sha256
        != decision_root.action_origin_receipt_sha256
        or not source_target.source_paths_replayed
    ):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive V3 decision, window, or economic target roots differ"
        )
    expected_context_dates = tuple(
        runtime.decision_session_dates[index]
        for index in window_row.context_tensor_indices
    )
    if (
        expected_context_dates != window_row.context_session_dates
        or full_window_output.valid.shape[0] != 1
        or full_window_output.valid.shape[1] != len(expected_context_dates)
        or full_window_output.valid.shape[2] != len(runtime.security_ids)
    ):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive V3 trainer output does not match its committed window"
        )
    device = full_window_output.valid.device
    output_position = torch.tensor(
        (window_row.origin_output_position,), dtype=torch.long, device=device
    )
    selected = _select_output(full_window_output, output_position)
    tensor_index = torch.tensor(
        ((window_row.tensor_origin_index,),), dtype=torch.long, device=device
    )
    origin_index = torch.tensor(
        ((window_row.candidate_origin_index,),), dtype=torch.long, device=device
    )
    role_dates = (
        split_plan.outer_folds[window_plan.fold_index].fit_session_dates
        if window_plan.split_role == "training"
        else split_plan.outer_folds[
            window_plan.fold_index
        ].inner_validation_session_dates
    )
    split_start = split_plan.candidate_session_dates.index(role_dates[0])
    return build_massive_adaptive_alpha_training_batch_v2(
        output=selected,
        decision_tensor=decision_tensor,
        target_artifacts=((source_target.targets,),),
        tensor_session_indices=tensor_index,
        origin_indices=origin_index,
        split_start_inclusive=split_start,
        split_stop_exclusive=window_row.target_stop_exclusive_index,
        split_role=window_plan.split_role,
        split_plan_receipt_sha256=split_plan.semantic_receipt_sha256,
    )


__all__ = ["build_massive_adaptive_alpha_training_batch_v3"]
