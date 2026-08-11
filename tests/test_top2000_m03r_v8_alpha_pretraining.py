from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
)
from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03RV8AlphaFoldEvidence,
    M03RV8AlphaPretrainingBatch,
    M03RV8AlphaPretrainingError,
    build_m03r_v8_alpha_fold_evidence,
    m03r_v8_alpha_pretraining_loss,
    qualify_m03r_v8_alpha_panel,
)


def _batch(
    *,
    split: str = "training",
    fold_index: int = 0,
) -> M03RV8AlphaPretrainingBatch:
    target = torch.tensor(
        [
            [
                [-0.03, -0.04, -0.05, -0.07],
                [-0.01, -0.02, -0.03, -0.05],
                [0.01, 0.02, 0.03, 0.05],
                [0.03, 0.04, 0.05, 0.07],
            ],
            [
                [0.04, 0.05, 0.06, 0.08],
                [0.02, 0.03, 0.04, 0.06],
                [-0.02, -0.03, -0.04, -0.06],
                [-0.04, -0.05, -0.06, -0.08],
            ],
            [
                [-0.02, -0.03, -0.04, -0.06],
                [0.04, 0.05, 0.06, 0.08],
                [-0.04, -0.05, -0.06, -0.08],
                [0.02, 0.03, 0.04, 0.06],
            ],
        ],
        dtype=torch.float64,
    )
    return M03RV8AlphaPretrainingBatch(
        predicted_mean=(0.9 * target).clone().requires_grad_(True),
        predicted_log_scale=torch.full_like(target, -3.0, requires_grad=True),
        target_residual_log_return=target,
        valid=torch.ones_like(target, dtype=torch.bool),
        origin_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        split=split,  # type: ignore[arg-type]
        fold_index=fold_index,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256=f"{fold_index + 1:x}" * 64,
    )


def test_date_balanced_pretraining_loss_is_finite_and_differentiable() -> None:
    batch = _batch()
    result = m03r_v8_alpha_pretraining_loss(batch)

    assert result.valid_date_horizon_count == 12
    assert torch.isfinite(result.total)
    result.total.backward()
    assert batch.predicted_mean.grad is not None
    assert batch.predicted_log_scale.grad is not None
    assert torch.isfinite(batch.predicted_mean.grad).all()
    assert batch.predicted_mean.grad.abs().sum() > 0


def test_no_ranking_setting_removes_only_the_listwise_weight() -> None:
    batch = _batch()
    reference = m03r_v8_alpha_pretraining_loss(batch)
    no_ranking = m03r_v8_alpha_pretraining_loss(batch, ranking_loss_weight=0.0)

    torch.testing.assert_close(
        reference.robust_regression,
        no_ranking.robust_regression,
    )
    torch.testing.assert_close(reference.distributional, no_ranking.distributional)
    torch.testing.assert_close(
        reference.total - no_ranking.total,
        0.5 * reference.listwise_ranking,
    )


def test_pretraining_loss_applies_the_frozen_horizon_weights() -> None:
    batch = _batch()
    combined = m03r_v8_alpha_pretraining_loss(batch)
    isolated = []
    for horizon_index in range(batch.valid.shape[-1]):
        horizon_valid = torch.zeros_like(batch.valid)
        horizon_valid[..., horizon_index] = batch.valid[..., horizon_index]
        isolated.append(
            m03r_v8_alpha_pretraining_loss(replace(batch, valid=horizon_valid))
        )
    weights = batch.predicted_mean.new_tensor(
        M03R_V8_ALPHA_PRETRAINING.horizon_loss_weights
    )

    for field in (
        "listwise_ranking",
        "robust_regression",
        "distributional",
        "total",
    ):
        expected = torch.sum(
            weights
            * torch.stack([getattr(result, field) for result in isolated]),
        )
        torch.testing.assert_close(getattr(combined, field), expected)


def test_pretraining_rejects_targets_crossing_the_bound_split() -> None:
    batch = _batch()
    crossing = replace(batch, split_stop_exclusive=30)
    with pytest.raises(M03RV8AlphaPretrainingError, match="crosses"):
        crossing.validate()


def test_outer_score_and_lockbox_access_fail_closed() -> None:
    batch = _batch()
    with pytest.raises(M03RV8AlphaPretrainingError, match="forbids"):
        replace(batch, outer_score_accessed=True).validate()
    with pytest.raises(M03RV8AlphaPretrainingError, match="forbids"):
        replace(batch, lockbox_accessed=True).validate()


def test_fold_evidence_is_date_balanced_and_content_addressed() -> None:
    evidence = build_m03r_v8_alpha_fold_evidence(_batch(split="inner-validation"))

    assert evidence.fold_index == 0
    assert evidence.valid_date_counts == (3, 3, 3, 3)
    assert evidence.mean_spearman_rank_ic == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert all(value > 0.0 for value in evidence.mean_top_bottom_decile_spread)
    assert len(evidence.receipt_sha256) == 64


def _fold_evidence(fold: int, ic_21: float, ic_30: float) -> M03RV8AlphaFoldEvidence:
    return M03RV8AlphaFoldEvidence(
        fold_index=fold,
        mean_spearman_rank_ic=(0.0, ic_21, ic_30, 0.0),
        mean_top_bottom_decile_spread=(0.0, 0.001, 0.001, 0.0),
        valid_date_counts=(10, 10, 10, 10),
        source_array_sha256=f"{fold + 1:x}" * 64,
    )


def test_six_fold_gate_passes_only_a_qualified_21_or_30_day_horizon() -> None:
    passed = qualify_m03r_v8_alpha_panel(
        tuple(
            _fold_evidence(
                fold,
                0.031 if fold < 4 else -0.001,
                0.01 if fold < 5 else -0.01,
            )
            for fold in range(6)
        )
    )
    assert passed.passed
    assert passed.qualifying_horizons == (21,)
    assert passed.positive_rank_ic_fold_count_21d == 4

    failed = qualify_m03r_v8_alpha_panel(
        tuple(_fold_evidence(fold, 0.01, -0.01) for fold in range(6))
    )
    assert not failed.passed
    assert failed.qualifying_horizons == ()


def test_fold_inventory_and_batch_schema_drift_are_rejected() -> None:
    rows = tuple(_fold_evidence(fold, 0.03, 0.03) for fold in range(6))
    with pytest.raises(M03RV8AlphaPretrainingError, match="six folds"):
        qualify_m03r_v8_alpha_panel((*rows[:-1], rows[-2]))
    with pytest.raises(M03RV8AlphaPretrainingError, match="schema"):
        replace(_batch(), schema="drifted").validate()
