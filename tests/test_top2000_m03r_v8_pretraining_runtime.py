from __future__ import annotations

import math

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03RV8AlphaPretrainingError,
    m03r_v8_alpha_pretraining_loss,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_runtime import (
    build_m03r_v8_alpha_pretraining_batch_from_origin_states,
    build_m03r_v8_alpha_pretraining_batch_from_states,
)


def _sequence(*, batch_size: int = 1) -> Hold30Sequence:
    positions = 70
    assets = 4
    weights = torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float32)
    weights = weights.expand(batch_size, -1).clone()
    ledger = CohortLedger.from_weights(
        weights,
        cash_index=0,
        initial_age=0,
        track_initial_units=True,
    )
    availability = torch.ones((positions, batch_size, assets), dtype=torch.bool)
    returns = torch.zeros((positions - 1, batch_size, assets))
    returns[..., 1] = 0.001
    returns[..., 2] = -0.001
    return Hold30Sequence(
        decision_state=torch.zeros((positions, batch_size, assets, 1)),
        asset_returns=returns,
        decision_available=availability,
        fill_membership=availability.clone(),
        fill_availability=availability.clone(),
        benchmark_weights=weights.unsqueeze(0).expand(positions, -1, -1).clone(),
        risk_asset_caps=torch.ones((positions, batch_size, assets)),
        risk_gross_max=torch.ones((positions, batch_size)),
        benchmark_net_returns=torch.zeros((positions - 1, batch_size)),
        initial_ledger=ledger,
        cost_rate=0.002,
        axis_id="v8-pretraining-runtime-test",
    )


def _policy() -> Top2000M03RV8DevelopmentPolicy:
    return Top2000M03RV8DevelopmentPolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def test_batch_builds_exact_benchmark_relative_targets_and_gradients() -> None:
    policy = _policy()
    sequence = _sequence()
    states = torch.randn(
        (sequence.n_positions, 1, sequence.num_assets, policy.token_dim),
        requires_grad=True,
    )
    batch = build_m03r_v8_alpha_pretraining_batch_from_states(
        policy,
        states,
        sequence,
        torch.tensor([0, 5], dtype=torch.long),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=70,
        fold_index=0,
        source_array_sha256="a" * 64,
    )

    assert batch.predicted_mean.shape == (2, 4, 4)
    assert batch.valid[:, 0].sum() == 0
    assert batch.valid[:, 1:].all()
    expected_h5 = 5.0 * math.log1p(0.001)
    assert batch.target_residual_log_return[0, 1, 0].item() == pytest.approx(
        expected_h5
    )
    assert batch.target_residual_log_return[0, 2, 0].item() == pytest.approx(
        5.0 * math.log1p(-0.001)
    )
    loss = m03r_v8_alpha_pretraining_loss(batch)
    loss.total.backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
    assert policy.alpha_log_scale_head.weight.grad is not None


def test_target_horizons_stop_at_the_declared_split() -> None:
    policy = _policy()
    sequence = _sequence()
    states = torch.randn((70, 1, 4, 16))
    batch = build_m03r_v8_alpha_pretraining_batch_from_states(
        policy,
        states,
        sequence,
        torch.tensor([30], dtype=torch.long),
        sequence_global_state_start=0,
        split="inner-validation",
        split_start_inclusive=30,
        split_stop_exclusive=60,
        fold_index=0,
        source_array_sha256="b" * 64,
    )

    assert not batch.valid[0, 0, :2].any()
    assert batch.valid[0, 1:, :2].all()
    assert not batch.valid[0, :, 2:].any()


def test_origin_aligned_states_avoid_full_placeholder_requirement() -> None:
    policy = _policy()
    sequence = _sequence()
    origins = torch.tensor([0, 5], dtype=torch.long)
    states = torch.randn((2, 1, 4, 16), requires_grad=True)
    batch = build_m03r_v8_alpha_pretraining_batch_from_origin_states(
        policy,
        states,
        sequence,
        origins,
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=70,
        fold_index=0,
        source_array_sha256="e" * 64,
    )

    assert batch.origin_indices.tolist() == [0, 5]
    m03r_v8_alpha_pretraining_loss(batch).total.backward()
    assert states.grad is not None


def test_multi_path_or_out_of_split_inputs_fail_closed() -> None:
    policy = _policy()
    with pytest.raises(M03RV8AlphaPretrainingError, match="one economic path"):
        build_m03r_v8_alpha_pretraining_batch_from_states(
            policy,
            torch.randn((70, 2, 4, 16)),
            _sequence(batch_size=2),
            torch.tensor([0], dtype=torch.long),
            sequence_global_state_start=0,
            split="training",
            split_start_inclusive=0,
            split_stop_exclusive=70,
            fold_index=0,
            source_array_sha256="c" * 64,
        )
    with pytest.raises(M03RV8AlphaPretrainingError, match="leave the declared"):
        build_m03r_v8_alpha_pretraining_batch_from_states(
            policy,
            torch.randn((70, 1, 4, 16)),
            _sequence(),
            torch.tensor([0], dtype=torch.long),
            sequence_global_state_start=100,
            split="training",
            split_start_inclusive=0,
            split_stop_exclusive=70,
            fold_index=0,
            source_array_sha256="d" * 64,
        )
