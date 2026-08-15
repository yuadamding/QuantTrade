from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.training.top2000_m03r_v15_validation_runtime import (
    M03RV15InnerValidationReceipt,
    M03RV15ValidationRuntimeError,
    select_m03r_v15_checkpoint,
)


def _receipt(epoch: int, *, ic: float, spread: float, robust: float) -> M03RV15InnerValidationReceipt:
    return M03RV15InnerValidationReceipt(
        setting_index=0,
        fold_index=2,
        epoch_index=epoch,
        completed_updates=(epoch + 1) * 6,
        origin_count=32,
        mean_action_projected_rank_ic=ic,
        mean_action_projected_top_bottom_spread=spread,
        robust_regression_loss=robust,
        action_projected_prediction_std=0.01,
        target_std=0.02,
        model_state_sha256=f"{epoch + 1:x}" * 64,
        batch_receipt_sha256="89abcdef"[epoch] * 64,
    )


def test_v15_checkpoint_selection_uses_ic_then_spread_then_robust() -> None:
    rows = tuple(
        _receipt(epoch, ic=0.01, spread=0.001, robust=0.8)
        for epoch in range(8)
    )
    rows = tuple(
        replace(row, mean_action_projected_rank_ic=0.02)
        if row.epoch_index in {2, 4, 6}
        else row
        for row in rows
    )
    rows = tuple(
        replace(row, mean_action_projected_top_bottom_spread=0.003)
        if row.epoch_index in {4, 6}
        else row
        for row in rows
    )
    rows = tuple(
        replace(row, robust_regression_loss=0.6)
        if row.epoch_index == 6
        else row
        for row in rows
    )
    selected = select_m03r_v15_checkpoint(rows)
    assert selected.selected_epoch_index == 6
    assert selected.qualification_tail_accessed is False
    assert selected.selected_validation_receipt_sha256 == rows[6].receipt_sha256


def test_v15_checkpoint_selection_rejects_missing_epoch() -> None:
    rows = tuple(
        _receipt(epoch, ic=0.01, spread=0.001, robust=0.8)
        for epoch in range(7)
    )
    with pytest.raises(M03RV15ValidationRuntimeError, match="incomplete"):
        select_m03r_v15_checkpoint(rows)


def test_v15_checkpoint_selection_keeps_first_exact_tie() -> None:
    rows = tuple(
        _receipt(epoch, ic=0.01, spread=0.001, robust=0.8)
        for epoch in range(8)
    )
    selected = select_m03r_v15_checkpoint(rows)
    assert selected.selected_epoch_index == 0
