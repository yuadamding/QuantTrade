from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16InnerValidationReceipt,
    M03RV16ValidationError,
    select_m03r_v16_score_checkpoint,
)


def _receipt(epoch: int, ic: float) -> M03RV16InnerValidationReceipt:
    return M03RV16InnerValidationReceipt(
        setting_index=0,
        fold_index=1,
        epoch_index=epoch,
        completed_score_updates=4 * (epoch + 1),
        origin_count=32,
        mean_selection_rank_ic=ic,
        mean_selection_top_bottom_spread=0.001 * (epoch + 1),
        selection_robust_loss=0.2 - epoch * 0.001,
        mean_timing_rank_ic=0.0,
        selection_prediction_std=0.02,
        selection_target_std=0.05,
        model_state_sha256=f"{epoch + 1:064x}",
        epoch_checkpoint_file_sha256=f"{epoch + 101:064x}",
        batch_receipt_sha256=f"{epoch + 201:064x}",
    )


def test_v16_checkpoint_selection_uses_only_selection_score_metrics() -> None:
    receipts = tuple(_receipt(epoch, 0.01 if epoch == 1 else 0.0) for epoch in range(4))
    selected = select_m03r_v16_score_checkpoint(receipts)
    assert selected.selected_epoch_index == 1
    assert not selected.stop_authorized
    assert selected.stop_reason == "continue"


def test_v16_checkpoint_selection_stops_after_frozen_patience() -> None:
    receipts = tuple(_receipt(epoch, 0.02 if epoch == 0 else 0.0) for epoch in range(5))
    selected = select_m03r_v16_score_checkpoint(receipts)
    assert selected.selected_epoch_index == 0
    assert selected.stop_authorized
    assert selected.stop_reason == "patience-exhausted"


def test_v16_checkpoint_selection_rejects_outer_tail_access() -> None:
    invalid = replace(_receipt(0, 0.0), qualification_tail_accessed=True)
    with pytest.raises(M03RV16ValidationError, match="receipt"):
        select_m03r_v16_score_checkpoint((invalid,))
