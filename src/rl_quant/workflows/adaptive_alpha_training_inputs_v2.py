"""Align adaptive source targets to a replay-promoted decision tensor axis.

V1 assumed that the target and model context axes were identical.  Adaptive
research instead uses a larger, stable context axis and a changing PIT action
set.  This package-owned adapter maps each source target row onto the committed
security axis, leaves context-only rows explicitly missing, and derives the
equal-weight benchmark and B01 benchmark return without accepting caller target
arrays.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from rl_quant.features.massive_adaptive_alpha_targets_v1 import (
    MassiveAdaptiveAlphaTargetsV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MassiveAdaptiveAlphaSequenceOutputV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
)
from rl_quant.training.adaptive_alpha_supervised_v1 import (
    MassiveAdaptiveAlphaSupervisedV1Error,
    MassiveAdaptiveAlphaTrainingBatchV1,
)


_BUCKET_COUNT = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)


def build_massive_adaptive_alpha_training_batch_v2(
    *,
    output: MassiveAdaptiveAlphaSequenceOutputV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    target_artifacts: Sequence[Sequence[MassiveAdaptiveAlphaTargetsV1]],
    tensor_session_indices: torch.Tensor,
    origin_indices: torch.Tensor,
    split_start_inclusive: int,
    split_stop_exclusive: int,
    split_role: str,
    split_plan_receipt_sha256: str,
) -> MassiveAdaptiveAlphaTrainingBatchV1:
    """Build an exact source-target batch on the committed context axis."""

    decision_tensor.validate()
    runtime = decision_tensor.runtime_tensor
    if runtime is None or not decision_tensor.model_input_authorized:
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive decision tensor has not been package replayed"
        )
    reference = output.residual_distribution.mean
    if reference.ndim != 4 or reference.shape[-1] != _BUCKET_COUNT:
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive output cannot define the V2 target geometry"
        )
    batch_count, session_count, asset_count, _ = reference.shape
    nested = tuple(tuple(rows) for rows in target_artifacts)
    if (
        asset_count != len(runtime.security_ids)
        or tensor_session_indices.shape != (batch_count, session_count)
        or tensor_session_indices.dtype != torch.long
        or tensor_session_indices.device != reference.device
        or origin_indices.shape != tensor_session_indices.shape
        or origin_indices.dtype != torch.long
        or origin_indices.device != reference.device
        or len(nested) != batch_count
        or any(len(rows) != session_count for rows in nested)
        or bool((tensor_session_indices < 0).any())
        or bool((tensor_session_indices >= len(runtime.decision_session_dates)).any())
        or bool(
            (tensor_session_indices[:, 1:] <= tensor_session_indices[:, :-1]).any()
        )
    ):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive V2 target batch axes or chronology differ"
        )
    action_mask = runtime.action_mask.to(reference.device).index_select(
        0, tensor_session_indices.reshape(-1)
    ).reshape(batch_count, session_count, asset_count)
    if not torch.equal(output.valid, action_mask):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive model output differs from the committed action mask"
        )

    dtype = reference.dtype
    device = reference.device
    raw = torch.zeros_like(reference)
    factor_component = torch.zeros_like(reference)
    residual = torch.zeros_like(reference)
    target_valid = torch.zeros_like(reference, dtype=torch.bool)
    factor_target = torch.zeros(
        (batch_count, session_count, _BUCKET_COUNT),
        dtype=dtype,
        device=device,
    )
    factor_valid = torch.zeros_like(factor_target, dtype=torch.bool)
    security_index = {
        security_id: index
        for index, security_id in enumerate(runtime.security_ids)
    }
    artifact_receipts: list[tuple[str, ...]] = []
    operator_receipts: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for batch_index, artifacts in enumerate(nested):
        batch_receipts: list[str] = []
        batch_operators: list[str] = []
        previous_decision_at_ms = -1
        for session_index, artifact in enumerate(artifacts):
            artifact.validate()
            tensor_position = int(
                tensor_session_indices[batch_index, session_index].item()
            )
            expected_date = runtime.decision_session_dates[tensor_position]
            expected_action_ids = tuple(
                runtime.security_ids[index]
                for index in torch.nonzero(
                    runtime.action_mask[tensor_position], as_tuple=False
                ).flatten().tolist()
            )
            if (
                artifact.decision_session_date != expected_date
                or artifact.security_ids != expected_action_ids
                or artifact.decision_at_ms <= previous_decision_at_ms
                or artifact.semantic_receipt_sha256 in seen
            ):
                raise MassiveAdaptiveAlphaSupervisedV1Error(
                    "adaptive V2 source target support or chronology differs"
                )
            previous_decision_at_ms = artifact.decision_at_ms
            seen.add(artifact.semantic_receipt_sha256)
            batch_receipts.append(artifact.semantic_receipt_sha256)
            batch_operators.append(artifact.residual_operator.receipt_sha256)
            factor_target[batch_index, session_index] = torch.tensor(
                artifact.factor_return_target, dtype=dtype, device=device
            )
            factor_valid[batch_index, session_index] = torch.tensor(
                artifact.factor_valid, dtype=torch.bool, device=device
            )
            for row in artifact.rows:
                asset_index = security_index[row.security_id]
                valid = torch.tensor(
                    row.training_valid_by_bucket,
                    dtype=torch.bool,
                    device=device,
                )
                target_valid[batch_index, session_index, asset_index] = valid
                raw_row = torch.tensor(
                    row.raw_bucket_returns, dtype=dtype, device=device
                )
                factor_row = torch.tensor(
                    row.factor_component_returns, dtype=dtype, device=device
                )
                residual_row = torch.tensor(
                    row.residual_bucket_returns, dtype=dtype, device=device
                )
                raw[batch_index, session_index, asset_index] = torch.where(
                    valid, raw_row, torch.zeros_like(raw_row)
                )
                factor_component[batch_index, session_index, asset_index] = (
                    torch.where(valid, factor_row, torch.zeros_like(factor_row))
                )
                residual[batch_index, session_index, asset_index] = torch.where(
                    valid, residual_row, torch.zeros_like(residual_row)
                )
        artifact_receipts.append(tuple(batch_receipts))
        operator_receipts.append(tuple(batch_operators))

    counts = action_mask.sum(dim=2, keepdim=True)
    benchmark_weights = action_mask.to(dtype) / counts.clamp_min(1).to(dtype)
    complete_b01 = (~action_mask | target_valid[..., 0]).all(dim=2)
    benchmark_returns = (benchmark_weights * raw[..., 0]).sum(dim=2)
    benchmark_returns = torch.where(
        complete_b01, benchmark_returns, torch.zeros_like(benchmark_returns)
    )
    result = MassiveAdaptiveAlphaTrainingBatchV1(
        output=output,
        raw_return_target=raw,
        factor_component_target=factor_component,
        residual_return_target=residual,
        target_valid=target_valid,
        factor_return_target=factor_target,
        factor_valid=factor_valid,
        action_mask=action_mask,
        benchmark_weights=benchmark_weights,
        benchmark_net_returns=benchmark_returns,
        initial_pretrade_weights=benchmark_weights[:, 0].clone(),
        portfolio_utility_valid=complete_b01,
        origin_indices=origin_indices,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        split_role=split_role,
        source_bundle_receipt_sha256=decision_tensor.semantic_receipt_sha256,
        target_bundle_receipt_sha256=semantic_sha256(tuple(artifact_receipts)),
        factor_operator_receipt_sha256=semantic_sha256(tuple(operator_receipts)),
        split_plan_receipt_sha256=split_plan_receipt_sha256,
    )
    result.validate()
    return result


__all__ = ["build_massive_adaptive_alpha_training_batch_v2"]
