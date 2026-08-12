from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03R_V9_HORIZONS
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03R_V9_HORIZON_SCALES,
    M03RV9AlphaPretrainingBatch,
    M03RV9AlphaPretrainingError,
    build_m03r_v9_alpha_fold_evidence,
    m03r_v9_alpha_pretraining_loss,
)


def _batch(
    *, split: str = "training", target_mode: str = "factor-residual"
) -> M03RV9AlphaPretrainingBatch:
    scales = torch.tensor(M03R_V9_HORIZON_SCALES, dtype=torch.float64)
    cross_section = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float64)
    target = torch.stack(
        [
            torch.stack([cross_section * scale for scale in scales], dim=-1),
            torch.stack([-cross_section * scale for scale in scales], dim=-1),
            torch.stack([cross_section.roll(1) * scale for scale in scales], dim=-1),
        ]
    )
    return M03RV9AlphaPretrainingBatch(
        predicted_mean=(0.9 * target).clone().requires_grad_(True),
        predicted_log_scale=torch.log(scales)
        .view(1, 1, 4)
        .expand_as(target)
        .clone()
        .requires_grad_(True),
        target_log_return=target,
        valid=torch.ones_like(target, dtype=torch.bool),
        origin_indices=torch.tensor([0, 1, 2]),
        split=split,  # type: ignore[arg-type]
        target_mode=target_mode,  # type: ignore[arg-type]
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256="c" * 64 if target_mode == "factor-residual" else None,
    )


def test_loss_is_horizon_scaled_and_component_weights_remain_normalized() -> None:
    batch = _batch()
    ranked = m03r_v9_alpha_pretraining_loss(batch, ranking_enabled=True)
    no_ranking = m03r_v9_alpha_pretraining_loss(batch, ranking_enabled=False)

    assert ranked.component_weights == (0.50, 0.30, 0.20)
    assert no_ranking.component_weights == (0.0, 0.60, 0.40)
    torch.testing.assert_close(ranked.listwise_ranking, no_ranking.listwise_ranking)
    torch.testing.assert_close(ranked.robust_regression, no_ranking.robust_regression)
    torch.testing.assert_close(ranked.distributional, no_ranking.distributional)
    torch.testing.assert_close(
        no_ranking.total,
        0.60 * no_ranking.robust_regression + 0.40 * no_ranking.distributional,
    )
    ranked.total.backward()
    assert batch.predicted_mean.grad is not None
    assert torch.isfinite(batch.predicted_mean.grad).all()


def test_equal_standardized_cross_sections_have_equal_listwise_loss_by_horizon() -> (
    None
):
    batch = _batch()
    isolated = []
    for horizon in range(len(M03R_V9_HORIZONS)):
        valid = torch.zeros_like(batch.valid)
        valid[..., horizon] = True
        isolated.append(
            m03r_v9_alpha_pretraining_loss(
                replace(batch, valid=valid), ranking_enabled=True
            ).listwise_ranking
        )
    for value in isolated[1:]:
        torch.testing.assert_close(value, isolated[0])


def test_qualification_occurs_only_on_the_update_64_tail() -> None:
    evidence = build_m03r_v9_alpha_fold_evidence(_batch(split="qualification"))
    assert evidence.evaluated_update == 64
    assert evidence.mean_spearman_rank_ic == pytest.approx((1.0, 1.0, 1.0, 1.0))
    with pytest.raises(M03RV9AlphaPretrainingError, match="untouched qualification"):
        build_m03r_v9_alpha_fold_evidence(_batch(split="training"))


def test_factor_receipt_and_split_tamper_fail_closed() -> None:
    with pytest.raises(M03RV9AlphaPretrainingError, match="lack exposure"):
        replace(_batch(), exposure_receipt_sha256=None).validate()
    with pytest.raises(M03RV9AlphaPretrainingError, match="cross"):
        replace(_batch(), split_stop_exclusive=30).validate()
