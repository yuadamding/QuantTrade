from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import M03R_V10_SETTINGS
from rl_quant.training.top2000_m03r_v10_diagnostics import (
    M03RV10DiagnosticsError,
    build_m03r_v10_fold_diagnostics,
)
from rl_quant.training.top2000_m03r_v10_pretraining_step import (
    M03RV10AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)


def _batch(*, constant_prediction: bool = False) -> M03RV10AlphaPretrainingBatch:
    target_row = torch.tensor((-0.03, -0.01, 0.01, 0.03), dtype=torch.float64)
    target = target_row.view(1, 4, 1).repeat(3, 1, 4)
    prediction = torch.zeros_like(target) if constant_prediction else target.clone()
    base = M03RV9AlphaPretrainingBatch(
        predicted_mean=prediction,
        predicted_log_scale=torch.full_like(target, -3.0),
        target_log_return=target,
        valid=torch.ones_like(target, dtype=torch.bool),
        origin_indices=torch.tensor((0, 1, 2)),
        split="qualification",
        target_mode="factor-residual",
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256="c" * 64,
    )
    return M03RV10AlphaPretrainingBatch(base, M03R_V10_SETTINGS[1])


def test_v10_diagnostics_expose_ic_stability_and_prediction_dispersion() -> None:
    evidence = build_m03r_v10_fold_diagnostics(_batch())
    assert evidence.mean_spearman_rank_ic == pytest.approx((1.0,) * 4)
    assert evidence.population_std_spearman_rank_ic == pytest.approx((0.0,) * 4)
    assert evidence.positive_ic_date_fraction == pytest.approx((1.0,) * 4)
    assert all(value > 0.0 for value in evidence.mean_prediction_cross_sectional_std)
    assert all(value > 0.0 for value in evidence.mean_target_cross_sectional_std)
    assert len(evidence.receipt_sha256) == 64


def test_v10_diagnostics_make_constant_predictions_visible() -> None:
    evidence = build_m03r_v10_fold_diagnostics(_batch(constant_prediction=True))
    assert evidence.mean_spearman_rank_ic == pytest.approx((0.0,) * 4)
    assert evidence.mean_prediction_cross_sectional_std == pytest.approx((0.0,) * 4)


def test_v10_diagnostics_reject_training_split() -> None:
    batch = _batch()
    with pytest.raises(M03RV10DiagnosticsError, match="untouched"):
        build_m03r_v10_fold_diagnostics(
            replace(
                batch,
                imported_v9_batch=replace(
                    batch.imported_v9_batch,
                    split="training",
                ),
            )
        )
