"""Causal, explicitly calibrated input normalization for the Stage-1 context encoder."""

from __future__ import annotations

import copy

import torch

from rl_quant.models.context_encoder import ContextEncoder, ContextEncoderConfig

B, A, S, F, C, BL = 2, 3, 8, 3, 2, 2
NB = S // BL


def _encoder() -> ContextEncoder:
    torch.manual_seed(7)
    return ContextEncoder(ContextEncoderConfig(
        bar_feature_dim=F,
        covariate_dim=C,
        d_model=8,
        n_heads=2,
        n_layers=4,
        feedforward_dim=16,
        dropout=0.0,
        max_seconds=S,
        block_seconds=BL,
    ))


def _inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    bars = torch.randn(B, A, S, F, generator=generator)
    mask = torch.ones(B, A, S, dtype=torch.bool)
    mask[:, 0] = False  # CASH-like row must not affect calibration.
    cov = torch.randn(B, NB, A, C, generator=generator)
    return bars, mask, cov


def test_train_mode_context_is_invariant_to_future_bar_and_covariate_values() -> None:
    encoder = _encoder()
    calibration = _inputs(1)
    encoder.calibrate_normalization(*calibration)
    encoder.train()
    bars, mask, cov = _inputs(2)

    with torch.no_grad():
        per_stock, market = encoder(bars, mask, cov)
        future_bars = bars.clone()
        future_cov = cov.clone()
        future_bars[:, :, 2 * BL:] += 1_000.0
        future_cov[:, 2:] -= 2_000.0
        changed_per_stock, changed_market = encoder(future_bars, mask, future_cov)

    torch.testing.assert_close(per_stock[:, :2], changed_per_stock[:, :2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(market[:, :2], changed_market[:, :2], atol=1e-6, rtol=1e-6)


def test_fixed_normalization_makes_dropout_free_train_and_eval_identical() -> None:
    encoder = _encoder()
    bars, mask, cov = _inputs(3)
    encoder.calibrate_normalization(bars, mask, cov)
    assert encoder.normalization_calibrated
    fixed_stats = {
        name: tensor.clone()
        for name, tensor in encoder.state_dict().items()
        if name.startswith(("bar_norm.", "cov_norm."))
    }

    encoder.train()
    with torch.no_grad():
        train_output = encoder(bars, mask, cov)
    encoder.eval()
    with torch.no_grad():
        eval_output = encoder(bars, mask, cov)

    for train_tensor, eval_tensor in zip(train_output, eval_output):
        torch.testing.assert_close(train_tensor, eval_tensor, atol=0.0, rtol=0.0)
    for name, expected in fixed_stats.items():
        torch.testing.assert_close(encoder.state_dict()[name], expected)


def test_calibration_excludes_masked_bar_and_covariate_samples() -> None:
    encoder = _encoder()
    bars = torch.tensor([[[
        [1.0, 10.0, 100.0],
        [3.0, 30.0, 300.0],
        [1_000_000.0, 1_000_000.0, 1_000_000.0],
        [1_000_000.0, 1_000_000.0, 1_000_000.0],
        [1_000_000.0, 1_000_000.0, 1_000_000.0],
        [1_000_000.0, 1_000_000.0, 1_000_000.0],
        [1_000_000.0, 1_000_000.0, 1_000_000.0],
        [1_000_000.0, 1_000_000.0, 1_000_000.0],
    ]]]).expand(1, A, S, F).clone()
    bar_mask = torch.zeros(1, A, S, dtype=torch.bool)
    bar_mask[:, 1:, :2] = True
    cov = torch.full((1, NB, A, C), 1_000_000.0)
    cov[0, 0, 1:] = torch.tensor([2.0, 20.0])
    cov_mask = torch.zeros(1, NB, A, dtype=torch.bool)
    cov_mask[0, 0, 1:] = True

    encoder.calibrate_normalization(bars, bar_mask, cov, cov_mask)

    torch.testing.assert_close(encoder.bar_norm.sample_count, torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(encoder.bar_norm.running_mean, torch.tensor([2.0, 20.0, 200.0], dtype=torch.float64))
    torch.testing.assert_close(encoder.bar_norm.running_var, torch.tensor([1.0, 100.0, 10_000.0], dtype=torch.float64))
    torch.testing.assert_close(encoder.cov_norm.sample_count, torch.tensor([2.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(encoder.cov_norm.running_mean, torch.tensor([2.0, 20.0], dtype=torch.float64))
    torch.testing.assert_close(encoder.cov_norm.running_var, torch.zeros(C, dtype=torch.float64))


def test_streaming_moment_merge_matches_single_calibration_batch() -> None:
    bars, mask, cov = _inputs(4)
    bars = bars.double() + 1_000_000_000.0
    cov = cov.double() - 1_000_000_000.0
    single = _encoder().double()
    streamed = _encoder().double()

    single.calibrate_normalization(bars, mask, cov)
    for day in range(B):
        streamed.calibrate_normalization(bars[day:day + 1], mask[day:day + 1], cov[day:day + 1])

    torch.testing.assert_close(streamed.bar_norm.sample_count, single.bar_norm.sample_count)
    torch.testing.assert_close(streamed.bar_norm.running_mean, single.bar_norm.running_mean, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(streamed.bar_norm.running_var, single.bar_norm.running_var, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(streamed.cov_norm.sample_count, single.cov_norm.sample_count)
    torch.testing.assert_close(streamed.cov_norm.running_mean, single.cov_norm.running_mean, atol=2e-7, rtol=0.0)
    torch.testing.assert_close(streamed.cov_norm.running_var, single.cov_norm.running_var, atol=1e-7, rtol=1e-7)


def test_calibrated_statistics_and_outputs_round_trip_through_state_dict() -> None:
    encoder = _encoder()
    bars, mask, cov = _inputs(5)
    encoder.calibrate_normalization(bars, mask, cov)
    encoder.eval()
    with torch.no_grad():
        expected = encoder(bars, mask, cov)

    restored = _encoder()
    restored.load_state_dict(copy.deepcopy(encoder.state_dict()))
    restored.eval()
    with torch.no_grad():
        actual = restored(bars, mask, cov)

    assert restored.normalization_calibrated
    torch.testing.assert_close(restored.bar_norm.sample_count, encoder.bar_norm.sample_count)
    torch.testing.assert_close(restored.cov_norm.sample_count, encoder.cov_norm.sample_count)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0.0, rtol=0.0)
