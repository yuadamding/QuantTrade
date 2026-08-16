from __future__ import annotations

from dataclasses import replace

from rl_quant.training.top2000_m03r_v16_fit import (
    build_m03r_v16_epoch_fit_payload,
    classify_m03r_v16_training_adequacy,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16InnerValidationReceipt,
)


def _validation(epoch: int) -> M03RV16InnerValidationReceipt:
    geometry = render_m03r_v16_fold_geometries(1001)[0]
    return M03RV16InnerValidationReceipt(
        setting_index=2,
        fold_index=0,
        epoch_index=epoch,
        completed_score_updates=geometry.training_block_count * (epoch + 1),
        origin_count=63,
        mean_selection_rank_ic=0.01,
        mean_selection_top_bottom_spread=0.001,
        selection_robust_loss=0.1,
        selection_prediction_std=0.05,
        selection_target_std=0.1,
        model_state_sha256=f"{100 + epoch:064x}",
        epoch_checkpoint_file_sha256=f"{200 + epoch:064x}",
        batch_receipt_sha256=f"{300 + epoch:064x}",
    )


def _updates(epoch: int) -> tuple[tuple[dict[str, object], ...], ...]:
    row = {
        "setting_index": 2,
        "fold_index": 0,
        "total_loss": 0.1,
        "encoder_gradient_norm_before_clip": 0.2,
        "selection_head_gradient_norm_before_clip": 0.3,
        "encoder_gradient_clipped": False,
        "selection_head_gradient_clipped": False,
        "learning_rate_multiplier": 0.5 + 0.01 * epoch,
    }
    return (({**row, "distributed_rank": 0}, {**row, "distributed_rank": 1}),)


def test_v16_fit_receipts_classify_adequate_terminal_training() -> None:
    validations = tuple(_validation(epoch) for epoch in range(8))
    epochs = tuple(
        build_m03r_v16_epoch_fit_payload(
            validations[epoch],
            _updates(epoch),
            package_plan_sha256="a" * 64,
            worker_plan_sha256="b" * 64,
        )
        for epoch in range(8)
    )
    result = classify_m03r_v16_training_adequacy(validations, epochs)
    assert result.status == "adequate"


def test_v16_fit_receipts_route_collapsed_or_still_improving_fit_to_inconclusive() -> (
    None
):
    validations = tuple(_validation(epoch) for epoch in range(8))
    collapsed = (*validations[:-1], replace(validations[-1], selection_prediction_std=0.001))
    epochs = tuple(
        build_m03r_v16_epoch_fit_payload(
            collapsed[epoch],
            _updates(epoch),
            package_plan_sha256="a" * 64,
            worker_plan_sha256="b" * 64,
        )
        for epoch in range(8)
    )
    assert (
        classify_m03r_v16_training_adequacy(collapsed, epochs).status
        == "inconclusive-undertrained"
    )

    trending_ic = tuple(
        replace(
            value,
            mean_selection_rank_ic=(
                0.01 if epoch < 4 else 0.01 + 0.001 * (epoch - 3)
            ),
        )
        for epoch, value in enumerate(validations)
    )
    trending_ic_epochs = tuple(
        build_m03r_v16_epoch_fit_payload(
            trending_ic[epoch],
            _updates(epoch),
            package_plan_sha256="a" * 64,
            worker_plan_sha256="b" * 64,
        )
        for epoch in range(8)
    )
    assert (
        classify_m03r_v16_training_adequacy(
            trending_ic, trending_ic_epochs
        ).status
        == "inconclusive-undertrained"
    )

    improving_loss = tuple(
        replace(
            value,
            selection_robust_loss=(
                0.1 if epoch < 4 else 0.1 - 0.002 * (epoch - 3)
            ),
        )
        for epoch, value in enumerate(validations)
    )
    improving_loss_epochs = tuple(
        build_m03r_v16_epoch_fit_payload(
            improving_loss[epoch],
            _updates(epoch),
            package_plan_sha256="a" * 64,
            worker_plan_sha256="b" * 64,
        )
        for epoch in range(8)
    )
    assert (
        classify_m03r_v16_training_adequacy(
            improving_loss, improving_loss_epochs
        ).status
        == "inconclusive-undertrained"
    )

    overdispersed = (
        *validations[:-1],
        replace(validations[-1], selection_prediction_std=2.0),
    )
    overdispersed_epochs = tuple(
        build_m03r_v16_epoch_fit_payload(
            overdispersed[epoch],
            _updates(epoch),
            package_plan_sha256="a" * 64,
            worker_plan_sha256="b" * 64,
        )
        for epoch in range(8)
    )
    assert (
        classify_m03r_v16_training_adequacy(
            overdispersed, overdispersed_epochs
        ).status
        == "inconclusive-undertrained"
    )
