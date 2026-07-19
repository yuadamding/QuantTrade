from __future__ import annotations

import torch

from rl_quant.models.context_encoder import _last_valid
from rl_quant.models.daily_policy import FullDayRawEncoder
from rl_quant.models.decision_policy import RawSecondPolicyEncoder


def _raw_encoder() -> RawSecondPolicyEncoder:
    torch.manual_seed(7)
    return RawSecondPolicyEncoder(
        bar_feature_dim=5,
        d_model=8,
        block_seconds=6,
        n_heads=2,
        n_layers=2,
        feedforward_dim=16,
        dropout=0.0,
    ).eval()


@torch.no_grad()
def _dense_raw_reference(
    encoder: RawSecondPolicyEncoder, bars: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Previous dense key-padding implementation, retained only as an equivalence oracle."""
    batch, actions, seconds, features = bars.shape
    block = encoder.block_seconds
    blocks = seconds // block
    raw = bars[:, :, : blocks * block].reshape(batch * actions * blocks, block, features)
    valid = mask[:, :, : blocks * block].bool().reshape(batch * actions * blocks, block)
    hidden = encoder.input_proj(encoder._normalize_ohlcv(raw, valid))
    hidden = hidden + encoder.pos[:block].view(1, block, encoder.d_model)
    padding = ~valid
    all_padding = padding.all(dim=1)
    if all_padding.any():
        padding = padding.clone()
        padding[all_padding, 0] = False
    causal = torch.triu(torch.ones((block, block), dtype=torch.bool), diagonal=1)
    hidden = encoder.local(hidden, mask=causal, src_key_padding_mask=padding)
    hidden = encoder.out_norm(hidden)
    summary = _last_valid(hidden, valid) * valid.any(dim=1, keepdim=True).to(hidden.dtype)
    return summary.reshape(batch, actions, blocks, encoder.d_model).permute(0, 2, 1, 3)


def test_raw_second_ragged_packing_matches_dense_valid_attention() -> None:
    encoder = _raw_encoder()
    generator = torch.Generator().manual_seed(11)
    bars = torch.randn(2, 4, 12, 5, generator=generator)
    mask = torch.zeros(2, 4, 12, dtype=torch.bool)
    lengths = (0, 2, 4, 6, 1, 3, 5, 6, 4, 2, 6, 1, 5, 3, 6, 0)
    row = 0
    for batch in range(2):
        for action in range(4):
            for start in (0, 6):
                mask[batch, action, start:start + lengths[row]] = True
                row += 1
    with torch.no_grad():
        packed = encoder(bars, mask)
        dense = _dense_raw_reference(encoder, bars, mask)
    assert torch.allclose(packed, dense, atol=2e-6, rtol=1e-5)


def test_raw_second_ignores_masked_values_and_later_blocks() -> None:
    encoder = _raw_encoder()
    generator = torch.Generator().manual_seed(13)
    bars = torch.randn(1, 3, 12, 5, generator=generator)
    mask = torch.ones(1, 3, 12, dtype=torch.bool)
    mask[:, :, [1, 4, 8]] = False
    with torch.no_grad():
        baseline = encoder(bars, mask)
        changed = bars.clone()
        changed[~mask] = 1e30
        changed[:, :, 6:] += 1e4
        after = encoder(changed, mask)
    assert torch.allclose(after[:, 0], baseline[:, 0], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(after[:, 1], baseline[:, 1])


def test_full_day_ragged_paths_are_finite_differentiable_and_empty_safe() -> None:
    torch.manual_seed(17)
    encoder = FullDayRawEncoder(
        bar_feature_dim=5,
        d_model=8,
        n_heads=2,
        n_layers=4,
        feedforward_dim=16,
        dropout=0.0,
        block_seconds=6,
        max_seconds=24,
        raw_norm="instance",
        grad_checkpoint=True,
    ).train()
    bars = torch.randn(2, 4, 24, 5, requires_grad=True)
    mask = torch.rand(2, 4, 24) > 0.4
    mask[:, 0] = False
    output = encoder(bars, mask)
    output.square().mean().backward()
    assert output.shape == (2, 4, 8)
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))
    assert torch.isfinite(output).all()
    assert bars.grad is not None and torch.isfinite(bars.grad).all()
