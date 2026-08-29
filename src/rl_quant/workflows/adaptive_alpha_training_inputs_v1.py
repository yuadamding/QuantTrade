"""Workflow adapter from adaptive source targets to supervised tensors.

The legacy Phase-1 training package is intentionally forbidden from importing
the repository's feature package.  This top-layer workflow adapter owns that
integration without weakening the legacy audit or the adaptive source
contracts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from rl_quant.features.massive_adaptive_alpha_targets_v1 import (
    MassiveAdaptiveAlphaTargetsV1,
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


def build_massive_adaptive_alpha_training_batch_from_targets_v1(
    *,
    output: MassiveAdaptiveAlphaSequenceOutputV1,
    target_artifacts: Sequence[Sequence[MassiveAdaptiveAlphaTargetsV1]],
    action_security_ids: Sequence[str],
    benchmark_weights: torch.Tensor,
    benchmark_net_returns: torch.Tensor,
    initial_pretrade_weights: torch.Tensor,
    origin_indices: torch.Tensor,
    split_start_inclusive: int,
    split_stop_exclusive: int,
    split_role: str,
    source_bundle_receipt_sha256: str,
    split_plan_receipt_sha256: str,
    portfolio_utility_valid: torch.Tensor | None = None,
) -> MassiveAdaptiveAlphaTrainingBatchV1:
    """Tensorize source-shaped target artifacts without free target digests.

    The nested target inventory is ``[batch][session]``.  Missing economic or
    factor support is represented only by the committed training masks; this
    adapter never fills or substitutes a target.  It remains a development
    adapter because Target V1 is intentionally nonauthorizing.
    """

    reference = output.residual_distribution.mean
    if reference.ndim != 4 or reference.shape[-1] != _BUCKET_COUNT:
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive output cannot define a target tensor geometry"
        )
    batch_count, session_count, asset_count, _ = reference.shape
    security_ids = tuple(action_security_ids)
    nested = tuple(tuple(row) for row in target_artifacts)
    if (
        security_ids != tuple(sorted(set(security_ids)))
        or len(security_ids) != asset_count
        or len(nested) != batch_count
        or any(len(rows) != session_count for rows in nested)
    ):
        raise MassiveAdaptiveAlphaSupervisedV1Error(
            "adaptive target artifact axes differ from model output"
        )
    artifact_receipts: list[tuple[str, ...]] = []
    operator_receipts: list[tuple[str, ...]] = []
    raw_rows: list[list[tuple[tuple[float, ...], ...]]] = []
    factor_rows: list[list[tuple[tuple[float, ...], ...]]] = []
    residual_rows: list[list[tuple[tuple[float, ...], ...]]] = []
    valid_rows: list[list[tuple[tuple[bool, ...], ...]]] = []
    factor_target_rows: list[list[tuple[float, ...]]] = []
    factor_valid_rows: list[list[tuple[bool, ...]]] = []
    seen_receipts: set[str] = set()
    for batch_rows in nested:
        previous_decision_at_ms = -1
        batch_artifacts: list[str] = []
        batch_operators: list[str] = []
        batch_raw: list[tuple[tuple[float, ...], ...]] = []
        batch_factor: list[tuple[tuple[float, ...], ...]] = []
        batch_residual: list[tuple[tuple[float, ...], ...]] = []
        batch_valid: list[tuple[tuple[bool, ...], ...]] = []
        batch_factor_target: list[tuple[float, ...]] = []
        batch_factor_valid: list[tuple[bool, ...]] = []
        for artifact in batch_rows:
            artifact.validate()
            if (
                artifact.security_ids != security_ids
                or artifact.decision_at_ms <= previous_decision_at_ms
                or artifact.semantic_receipt_sha256 in seen_receipts
            ):
                raise MassiveAdaptiveAlphaSupervisedV1Error(
                    "adaptive target artifact chronology or security axis differs"
                )
            previous_decision_at_ms = artifact.decision_at_ms
            seen_receipts.add(artifact.semantic_receipt_sha256)
            batch_artifacts.append(artifact.semantic_receipt_sha256)
            batch_operators.append(artifact.residual_operator.receipt_sha256)
            batch_raw.append(
                tuple(row.raw_bucket_returns for row in artifact.rows)
            )
            batch_factor.append(
                tuple(row.factor_component_returns for row in artifact.rows)
            )
            batch_residual.append(
                tuple(row.residual_bucket_returns for row in artifact.rows)
            )
            batch_valid.append(
                tuple(row.training_valid_by_bucket for row in artifact.rows)
            )
            batch_factor_target.append(artifact.factor_return_target)
            batch_factor_valid.append(artifact.factor_valid)
        artifact_receipts.append(tuple(batch_artifacts))
        operator_receipts.append(tuple(batch_operators))
        raw_rows.append(batch_raw)
        factor_rows.append(batch_factor)
        residual_rows.append(batch_residual)
        valid_rows.append(batch_valid)
        factor_target_rows.append(batch_factor_target)
        factor_valid_rows.append(batch_factor_valid)

    device = reference.device
    dtype = reference.dtype
    target_valid = torch.tensor(valid_rows, dtype=torch.bool, device=device)

    def _masked(values: object) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=dtype, device=device)
        return torch.where(target_valid, tensor, torch.zeros_like(tensor))

    raw_target = _masked(raw_rows)
    factor_component = _masked(factor_rows)
    residual_target = _masked(residual_rows)
    factor_target = torch.tensor(factor_target_rows, dtype=dtype, device=device)
    factor_valid = torch.tensor(
        factor_valid_rows, dtype=torch.bool, device=device
    )
    action_mask = output.valid
    if portfolio_utility_valid is None:
        portfolio_utility_valid = (
            ~action_mask | target_valid[..., 0]
        ).all(dim=2)
    result = MassiveAdaptiveAlphaTrainingBatchV1(
        output=output,
        raw_return_target=raw_target,
        factor_component_target=factor_component,
        residual_return_target=residual_target,
        target_valid=target_valid,
        factor_return_target=factor_target,
        factor_valid=factor_valid,
        action_mask=action_mask,
        benchmark_weights=benchmark_weights,
        benchmark_net_returns=benchmark_net_returns,
        initial_pretrade_weights=initial_pretrade_weights,
        portfolio_utility_valid=portfolio_utility_valid,
        origin_indices=origin_indices,
        split_start_inclusive=split_start_inclusive,
        split_stop_exclusive=split_stop_exclusive,
        split_role=split_role,
        source_bundle_receipt_sha256=source_bundle_receipt_sha256,
        target_bundle_receipt_sha256=semantic_sha256(tuple(artifact_receipts)),
        factor_operator_receipt_sha256=semantic_sha256(tuple(operator_receipts)),
        split_plan_receipt_sha256=split_plan_receipt_sha256,
    )
    result.validate()
    return result


__all__ = ["build_massive_adaptive_alpha_training_batch_from_targets_v1"]
