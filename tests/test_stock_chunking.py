"""Stock-axis chunking equivalence (the TOP2000 enabler).

At A~2000 one un-chunked day is a ~0.5TB tier-1 activation -- both encoders must process the stock axis in
chunks. Both are per-stock weight-shared, and every cross-stock computation is kept OUTSIDE the chunk loop (the
Stage-1 bar BatchNorm runs over the full cross-section before chunking; the cov path + market pooling run on the
full concatenated output; the Stage-2 raw norms are per-(stock,day)) -- so chunked == un-chunked up to float
epsilon in BOTH train and eval, gradients included, with BN running stats updated exactly once."""

from __future__ import annotations

import unittest

import torch

from rl_quant.models.context_encoder import ContextEncoder, ContextEncoderConfig
from rl_quant.models.daily_policy import FullDayRawEncoder

B, A, S, Fd, C, NB = 2, 7, 60, 5, 3, 3          # 60s session, 20s blocks -> 3 blocks


def _ctx_encoder(chunk: int, gc: bool) -> ContextEncoder:
    torch.manual_seed(0)
    return ContextEncoder(ContextEncoderConfig(
        bar_feature_dim=Fd, covariate_dim=C, d_model=16, n_heads=2, n_layers=2, feedforward_dim=32,
        dropout=0.0, max_seconds=S, block_seconds=20, grad_checkpoint=gc, stock_chunk=chunk))


def _inputs():
    g = torch.Generator().manual_seed(1)
    bars = torch.randn(B, A, S, Fd, generator=g)
    mask = torch.rand(B, A, S, generator=g) > 0.3
    mask[:, 0] = False                            # CASH row: no bars
    cov = torch.randn(B, NB, A, C, generator=g)
    return bars, mask, cov


class ContextEncoderChunking(unittest.TestCase):
    def _compare(self, train: bool, gc: bool) -> None:
        bars, mask, cov = _inputs()
        e0, e1 = _ctx_encoder(0, gc), _ctx_encoder(3, gc)
        e1.load_state_dict(e0.state_dict())
        for e in (e0, e1):
            e.train(train)
        ps0, mk0 = e0(bars, mask, cov)
        ps1, mk1 = e1(bars, mask, cov)
        self.assertLess(float((ps0 - ps1).abs().max()), 1e-5)
        self.assertLess(float((mk0 - mk1).abs().max()), 1e-5)
        if train:
            l0 = ps0.square().mean() + mk0.square().mean()
            l1 = ps1.square().mean() + mk1.square().mean()
            g0 = torch.autograd.grad(l0, [p for p in e0.parameters() if p.requires_grad], allow_unused=True)
            g1 = torch.autograd.grad(l1, [p for p in e1.parameters() if p.requires_grad], allow_unused=True)
            for a_, b_ in zip(g0, g1):
                if a_ is not None and b_ is not None:
                    self.assertLess(float((a_ - b_).abs().max()), 1e-5)
            # BatchNorm running stats: full-cross-section normalization BEFORE the chunk loop -> ONE identical update
            self.assertEqual(float((e0.bar_norm.running_mean - e1.bar_norm.running_mean).abs().max()), 0.0)
            self.assertEqual(float((e0.bar_norm.running_var - e1.bar_norm.running_var).abs().max()), 0.0)

    def test_eval_equivalence(self) -> None:
        self._compare(train=False, gc=False)
        self._compare(train=False, gc=True)

    def test_train_equivalence_with_grads_and_bn(self) -> None:
        self._compare(train=True, gc=False)
        self._compare(train=True, gc=True)                       # chunk-level checkpoint path


class FullDayRawEncoderChunking(unittest.TestCase):
    def _enc(self, chunk: int, gc: bool, raw_norm: str) -> FullDayRawEncoder:
        torch.manual_seed(0)
        return FullDayRawEncoder(bar_feature_dim=Fd, d_model=8, n_heads=2, n_layers=2, feedforward_dim=16,
                                 dropout=0.0, block_seconds=20, max_seconds=S, grad_checkpoint=gc,
                                 raw_norm=raw_norm, stock_chunk=chunk)

    def _compare(self, train: bool, gc: bool, raw_norm: str) -> None:
        g = torch.Generator().manual_seed(2)
        bars = torch.empty(B, A, S, Fd)
        bars[..., :4] = 100.0 + torch.randn(B, A, S, 4, generator=g)
        bars[..., 4] = (1000.0 + 50.0 * torch.randn(B, A, S, generator=g)).clamp_min(1.0)
        mask = torch.rand(B, A, S, generator=g) > 0.3
        e0, e1 = self._enc(0, gc, raw_norm), self._enc(3, gc, raw_norm)
        e1.load_state_dict(e0.state_dict())
        for e in (e0, e1):
            e.train(train)
        o0, o1 = e0(bars, mask), e1(bars, mask)
        self.assertLess(float((o0 - o1).abs().max()), 1e-5)
        if train:
            g0 = torch.autograd.grad(o0.square().mean(), list(e0.parameters()), allow_unused=True)
            g1 = torch.autograd.grad(o1.square().mean(), list(e1.parameters()), allow_unused=True)
            for a_, b_ in zip(g0, g1):
                if a_ is not None and b_ is not None:
                    self.assertLess(float((a_ - b_).abs().max()), 1e-5)

    def test_eval_and_train_equivalence_both_norms(self) -> None:
        for raw_norm in ("level", "instance"):
            self._compare(train=False, gc=False, raw_norm=raw_norm)
            self._compare(train=True, gc=False, raw_norm=raw_norm)
            self._compare(train=True, gc=True, raw_norm=raw_norm)   # chunk-level checkpoint path

    def test_uneven_last_chunk(self) -> None:
        """A=7 with chunk 3 -> chunks of 3/3/1; the ragged tail must concatenate correctly."""
        g = torch.Generator().manual_seed(3)
        bars = 100.0 + torch.randn(B, A, S, Fd, generator=g)
        mask = torch.ones(B, A, S, dtype=torch.bool)
        e0, e1 = self._enc(0, False, "level"), self._enc(3, False, "level")
        e1.load_state_dict(e0.state_dict())
        e0.eval(), e1.eval()
        self.assertEqual(e1(bars, mask).shape, (B, A, 8))
        self.assertLess(float((e0(bars, mask) - e1(bars, mask)).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
