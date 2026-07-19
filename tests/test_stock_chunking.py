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

    def test_amp_keeps_projected_context_hot_path_bfloat16(self) -> None:
        """FP32 position/mask constants must not widen TOP2000 transformer activations under autocast."""
        bars, mask, cov = _inputs()
        encoder = _ctx_encoder(3, False).train()
        hot_path_dtypes: list[torch.dtype] = []

        def capture_input(_module, args) -> None:
            hot_path_dtypes.append(args[0].dtype)

        handles = [
            encoder.tier1[0].register_forward_pre_hook(capture_input),
            encoder.tier2[0].register_forward_pre_hook(capture_input),
            encoder.fuse.register_forward_pre_hook(capture_input),
        ]
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                per_stock, market = encoder(bars, mask, cov)
                loss = per_stock.float().square().mean() + market.float().square().mean()
            loss.backward()
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(per_stock.dtype, torch.bfloat16)
        self.assertEqual(market.dtype, torch.bfloat16)
        self.assertTrue(hot_path_dtypes)
        self.assertEqual(set(hot_path_dtypes), {torch.bfloat16})
        gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))

        # The dtype-following casts are no-ops for the explicit non-AMP path.
        encoder.eval()
        with torch.no_grad():
            per_stock_fp32, market_fp32 = encoder(bars, mask, cov)
        self.assertEqual(per_stock_fp32.dtype, torch.float32)
        self.assertEqual(market_fp32.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast dtype contract")
    def test_cuda_amp_context_output_is_bfloat16_and_differentiable(self) -> None:
        """CUDA LayerNorm returns FP32, so every final/indexed boundary must explicitly restore BF16."""
        bars, mask, cov = (value.cuda() for value in _inputs())
        encoder = _ctx_encoder(3, False).cuda().train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            per_stock, market = encoder(bars, mask, cov)
            loss = per_stock.float().square().mean() + market.float().square().mean()
        self.assertEqual(per_stock.dtype, torch.bfloat16)
        self.assertEqual(market.dtype, torch.bfloat16)
        loss.backward()
        gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))


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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast dtype contract")
    def test_cuda_amp_raw_output_is_bfloat16_and_differentiable(self) -> None:
        generator = torch.Generator().manual_seed(5)
        bars = (100.0 + torch.randn(B, A, S, Fd, generator=generator)).cuda()
        mask = torch.ones(B, A, S, dtype=torch.bool, device="cuda")
        encoder = self._enc(3, True, "level").cuda().train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = encoder(bars, mask)
            loss = output.float().square().mean()
        self.assertEqual(output.dtype, torch.bfloat16)
        loss.backward()
        gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
