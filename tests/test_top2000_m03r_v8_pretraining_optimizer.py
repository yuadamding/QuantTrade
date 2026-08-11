from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ALPHA_PRETRAINING,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_optimizer import (
    M03RV8AlphaOptimizerError,
    build_m03r_v8_alpha_pretraining_optimizer,
    validate_m03r_v8_alpha_pretraining_optimizer,
)


def _policy(setting_index: int = 0) -> Top2000M03RV8DevelopmentPolicy:
    return Top2000M03RV8DevelopmentPolicy(
        setting_index,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def test_optimizer_uses_two_disjoint_exact_predictive_groups() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v8_alpha_pretraining_optimizer(policy)
    validate_m03r_v8_alpha_pretraining_optimizer(policy, optimizer, partition)

    assert partition.encoder_parameter_names
    assert partition.prediction_head_parameter_names
    assert set(partition.encoder_parameter_names).isdisjoint(
        partition.prediction_head_parameter_names
    )
    assert partition.encoder_learning_rate == pytest.approx(2.0e-5)
    assert partition.prediction_head_learning_rate == pytest.approx(1.0e-4)
    assert partition.receipt_sha256
    selected = set(partition.encoder_parameter_names) | set(
        partition.prediction_head_parameter_names
    )
    assert not any("hazard" in name for name in selected)
    assert not any("confidence" in name for name in selected)
    assert not any("gate_head" in name for name in selected)
    assert not any("score." in name for name in selected)


def test_no_pretraining_row_cannot_create_the_optimizer() -> None:
    with pytest.raises(M03RV8AlphaOptimizerError, match="no-alpha-pretraining"):
        build_m03r_v8_alpha_pretraining_optimizer(_policy(1))


def test_mutated_group_or_partition_fails_closed() -> None:
    policy = _policy(2)
    optimizer, partition = build_m03r_v8_alpha_pretraining_optimizer(policy)
    optimizer.param_groups[0]["lr"] = 9.0e-4
    with pytest.raises(M03RV8AlphaOptimizerError, match="group binding"):
        validate_m03r_v8_alpha_pretraining_optimizer(policy, optimizer, partition)
    with pytest.raises(M03RV8AlphaOptimizerError, match="drifted"):
        replace(partition, gradient_clip_norm=2.0).validate()


def test_protocol_freezes_bounded_early_stopping_schedule() -> None:
    spec = M03R_V8_ALPHA_PRETRAINING
    assert spec.encoder_learning_rate < spec.prediction_head_learning_rate
    assert spec.maximum_optimizer_updates == 64
    assert spec.inner_validation_interval_updates == 4
    assert spec.minimum_optimizer_updates_before_early_stop == 16
    assert spec.early_stopping_patience_evaluations == 4
    assert spec.checkpoint_selection_rule.endswith("tie-earliest-v1")
