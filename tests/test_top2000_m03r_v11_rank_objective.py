from __future__ import annotations

import torch

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import M03R_V11_SETTINGS
from rl_quant.training.top2000_m03r_v9_alpha_pretraining import (
    M03RV9AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v11_pretraining_runtime import (
    M03RV11AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v11_rank_objective import (
    m03r_v11_predictive_loss,
)


def _batch(setting: int) -> M03RV11AlphaPretrainingBatch:
    mean = torch.randn((3, 10, 4), requires_grad=True)
    base = M03RV9AlphaPretrainingBatch(
        predicted_mean=mean,
        predicted_log_scale=torch.zeros_like(mean),
        target_log_return=torch.randn_like(mean),
        valid=torch.ones_like(mean, dtype=torch.bool),
        origin_indices=torch.tensor([1, 2, 3]),
        split="training",
        target_mode="factor-residual",
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=100,
        source_array_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        exposure_receipt_sha256="c" * 64,
    )
    count = mean.shape[0] * mean.shape[2]
    return M03RV11AlphaPretrainingBatch(
        corrected_batch=base,
        setting=M03R_V11_SETTINGS[setting],
        residual_operator_receipt_sha256=tuple("d" * 64 for _ in range(count)),
        available_risky_asset_count=tuple(10 for _ in range(count)),
        factor_qualified_risky_asset_count=tuple(10 for _ in range(count)),
        effective_design_rank=tuple(5 for _ in range(count)),
        weighted_residual_degrees_of_freedom=tuple(5 for _ in range(count)),
    )


def test_v11_rank_losses_are_finite_and_differentiable() -> None:
    for setting in range(3):
        batch = _batch(setting)
        loss = m03r_v11_predictive_loss(batch)
        assert torch.isfinite(loss.total)
        loss.total.backward()
        assert batch.corrected_batch.predicted_mean.grad is not None
        assert torch.isfinite(batch.corrected_batch.predicted_mean.grad).all()
