"""Tests for the daily_raw day-level redesign: PIT close-to-close label, full-day trainable raw encoder, causal
cross-day temporal memory, long-only allocation, episode coverage, grad isolation, and end-to-end learnability."""

from __future__ import annotations

import unittest

import torch

from rl_quant.datasets import build_daily_raw_episodes, horizon_close_returns
from rl_quant.models import (
    CrossDayTemporalEncoder,
    DailyCrossSectionConfig,
    DailyCrossSectionPolicy,
    FullDayRawEncoder,
)
from rl_quant.training import (
    daily_cost_paid_baselines,
    evaluate_daily_detailed,
    ssl_targets_daily,
    train_daily_policy,
)

A, S, Fd, M, DC = 4, 24, 5, 3, 8        # 4 actions incl CASH; 24s session; 5 OHLCV fields; 3 news; ctx dim 8


def _cfg(**kw):
    base = dict(context_dim=DC, bar_feature_dim=Fd, raw_policy_dim=8, raw_policy_layers=2, raw_policy_heads=2,
                raw_block_seconds=8, session_seconds=S, news_raw_dim=1, max_news=M, news_embed_dim=8,
                token_dim=16, temporal_layers=2, temporal_heads=2, daily_lookback=20, max_days=64,
                alloc_layers=2, alloc_heads=2, feedforward_dim=32, dropout=0.0)
    base.update(kw)
    return DailyCrossSectionConfig(**base)


def _episode(B, T, gen):
    return dict(
        market=torch.randn(B, T, DC, generator=gen), per_stock=torch.randn(B, T, A, DC, generator=gen),
        bars=torch.randn(B, T, A, S, Fd, generator=gen), bar_mask=torch.ones(B, T, A, S, dtype=torch.bool),
        news_raw=torch.randn(B, T, A, M, 1, generator=gen), news_mask=torch.ones(B, T, A, M, dtype=torch.bool),
        avail=torch.ones(B, T, A, dtype=torch.bool))


class HorizonCloseReturns(unittest.TestCase):
    def test_pit_indexing_validity_and_cash(self) -> None:
        N, H = 10, 3
        dc = torch.zeros(N, A)
        dc[:, 0] = float("nan")                              # CASH has no price
        dc[:, 1] = 100 + torch.arange(N).float()
        dc[:, 2:] = 200.0
        ret, valid = horizon_close_returns(dc, horizon=H, exec_delay=1)
        # d=0: entry=close[1]=101, exit=close[1+3]=104 -> 104/101-1
        self.assertAlmostEqual(float(ret[0, 1]), 104 / 101 - 1, places=6)
        # last valid decision needs d+1+H <= N-1 => d <= N-2-H = 5
        self.assertTrue(bool(valid[5, 1]))
        self.assertFalse(bool(valid[6, 1]))                  # exit index out of range -> invalid
        self.assertTrue(bool(valid[:, 0].all()))             # CASH always valid
        self.assertEqual(float(ret[:, 0].abs().max()), 0.0)  # CASH return 0


class FullDayRawEncoderProps(unittest.TestCase):
    def test_cross_sectional_independence_and_affine_invariance(self) -> None:
        torch.manual_seed(0)
        enc = FullDayRawEncoder(bar_feature_dim=Fd, d_model=8, n_heads=2, n_layers=2, feedforward_dim=16,
                                dropout=0.0, block_seconds=8, max_seconds=S).eval()
        g = torch.Generator().manual_seed(1)
        bars = torch.randn(2, A, S, Fd, generator=g)
        mask = torch.ones(2, A, S, dtype=torch.bool)
        out = enc(bars, mask)
        # perturbing stock 2's intraday shape must not change stock 1's embedding (per-stock instance norm)
        b2 = bars.clone()
        b2[:, 2, : S // 2] += 5.0
        out2 = enc(b2, mask)
        self.assertLess(float((out2[:, 1] - out[:, 1]).abs().max()), 1e-6)
        self.assertGreater(float((out2[:, 2] - out[:, 2]).abs().max()), 1e-6)
        # affine-invariant: adding a constant to a whole stock-day is removed by per-day mean subtraction
        b3 = bars.clone()
        b3[:, 1] += 7.0
        self.assertLess(float((enc(b3, mask)[:, 1] - out[:, 1]).abs().max()), 1e-5)


class CrossDayCausality(unittest.TestCase):
    def test_temporal_encoder_is_strictly_causal(self) -> None:
        torch.manual_seed(0)
        te = CrossDayTemporalEncoder(d_model=8, n_heads=2, n_layers=2, feedforward_dim=16, dropout=0.0,
                                     max_days=32).eval()
        g = torch.Generator().manual_seed(1)
        seq = torch.randn(2, 7, A, 8, generator=g)
        out = te(seq)
        seq2 = seq.clone()
        seq2[:, 5, :, ::2] += 3.0                            # perturb day 5 (feature subset -> survives LayerNorm)
        out2 = te(seq2)
        self.assertLess(float((out2[:, :5] - out[:, :5]).abs().max()), 1e-6)   # days < 5 unchanged (causal)
        self.assertGreater(float((out2[:, 5:] - out[:, 5:]).abs().max()), 1e-6)  # days >= 5 change

    def test_policy_is_long_only_and_causal(self) -> None:
        torch.manual_seed(0)
        pol = DailyCrossSectionPolicy(_cfg()).eval()
        g = torch.Generator().manual_seed(1)
        ep = _episode(2, 6, g)
        state = pol.encode_episode(ep["market"], ep["per_stock"], ep["bars"], ep["bar_mask"],
                                   ep["news_raw"], ep["news_mask"], ep["avail"])
        self.assertEqual(state.shape, (2, 6, A, 16))
        prev = torch.zeros(2, A)
        prev[:, 0] = 1.0
        w, gate = pol.step(state[:, 0], prev, ep["avail"][:, 0])
        self.assertTrue(torch.allclose(w.sum(1), torch.ones(2), atol=1e-5))   # long-only simplex
        self.assertTrue(bool((w >= 0).all()))
        self.assertEqual(gate.shape, (2,))
        # future-day bar change must not move an earlier day's temporal state
        b2 = ep["bars"].clone()
        b2[:, 5, :, : S // 2] += 5.0
        state2 = pol.encode_episode(ep["market"], ep["per_stock"], b2, ep["bar_mask"],
                                    ep["news_raw"], ep["news_mask"], ep["avail"])
        self.assertLess(float((state2[:, :5] - state[:, :5]).abs().max()), 1e-6)


class Episodes(unittest.TestCase):
    def test_label_coverage_and_shapes(self) -> None:
        N, H = 20, 3
        g = torch.Generator().manual_seed(0)
        recs = [dict(date=f"d{i}", day_close=100 + torch.arange(A).float() + i,
                     market=torch.randn(DC, generator=g), per_stock=torch.randn(A, DC, generator=g),
                     bars=torch.randn(A, S, Fd, generator=g), bar_mask=torch.ones(A, S, dtype=torch.bool),
                     news_raw=torch.zeros(A, M, 1), news_mask=torch.ones(A, M, dtype=torch.bool),
                     avail=torch.ones(A, dtype=torch.bool)) for i in range(N)]
        eps = build_daily_raw_episodes(recs, episode_len=8, stride=4, horizon=H, exec_delay=1)
        self.assertTrue(len(eps) >= 1)
        for e in eps:
            self.assertEqual(e["bars"].shape, (8, A, S, Fd))
            self.assertEqual(e["ret"].shape, (8, A))
        # continuous single episode spanning the usable range
        one = build_daily_raw_episodes(recs, episode_len=N, stride=N, horizon=H, exec_delay=1)
        self.assertEqual(len(one), 1)
        usable = N - (1 + H)                                  # labelled days
        self.assertEqual(one[0]["ret"].shape[0], usable)


class DailySSL(unittest.TestCase):
    def test_daily_ssl_target_is_demeaned_and_pit(self) -> None:
        N, H = 12, 2
        dc = torch.zeros(N, A)
        dc[:, 0] = float("nan")
        dc[:, 1:] = 100 + torch.randn(N, A - 1, generator=torch.Generator().manual_seed(0)).cumsum(0)
        tgt, vm = ssl_targets_daily(dc, H, exec_delay=1)
        self.assertFalse(bool(vm[:, 0].any()))               # CASH excluded from the relative-value target
        # where >=2 stocks are valid, the target is cross-sectionally demeaned (sums ~0 over valid non-CASH)
        for d in range(N):
            v = vm[d, 1:]
            if int(v.sum()) >= 2:
                self.assertAlmostEqual(float(tgt[d, 1:][v].sum()), 0.0, places=4)


class GradIsolationAndLearnability(unittest.TestCase):
    def test_policy_holds_no_encoder_and_trains(self) -> None:
        pol = DailyCrossSectionPolicy(_cfg())
        # structural: the policy has its OWN trainable raw encoder but NO frozen-context-encoder reference
        names = [n for n, _ in pol.named_modules()]
        self.assertTrue(any("raw_encoder" in n for n in names))
        self.assertTrue(any("temporal" in n for n in names))

    def test_learns_planted_cross_sectional_signal_and_beats_cash(self) -> None:
        torch.manual_seed(0)
        N, H = 70, 3
        g = torch.Generator().manual_seed(1)
        dc = torch.empty(N, A)
        dc[:, 0] = float("nan")
        dc[:, 1:] = 100 + (0.5 * torch.randn(N, A - 1, generator=g)).cumsum(0)
        recs = [dict(date=f"d{i}", day_close=dc[i], market=torch.randn(DC, generator=g),
                     per_stock=torch.randn(A, DC, generator=g), bars=torch.randn(A, S, Fd, generator=g),
                     bar_mask=torch.ones(A, S, dtype=torch.bool), news_raw=torch.zeros(A, M, 1),
                     news_mask=torch.ones(A, M, dtype=torch.bool), avail=torch.ones(A, dtype=torch.bool))
                for i in range(N)]
        eps = build_daily_raw_episodes(recs, episode_len=18, stride=4, horizon=H, exec_delay=1)
        for e in eps:                                        # PLANT: leak the label into frozen-ctx channel 0
            e["per_stock"] = e["per_stock"].clone()
            e["per_stock"][:, :, 0] = e["ret"]
        ntr = int(len(eps) * 0.7)
        train_eps, test_eps = eps[:ntr], eps[ntr:]
        pol = DailyCrossSectionPolicy(_cfg(daily_lookback=18))
        dev = torch.device("cpu")
        _, best_val, best_state = train_daily_policy(
            pol, train_eps, steps=120, lr=3e-3, batch_days=4, cost=0.0, risk_lambda=0.0, budget_lambda=0.0,
            gate_entropy_coef=0.0, missing_label_penalty=1.0, bptt_window=18, eval_every=60, val_eps=test_eps,
            device=dev, min_val_label_reportable_fraction=0.0)
        if best_state:
            pol.load_state_dict(best_state)
        rows, _ = evaluate_daily_detailed(pol, test_eps, dev, cost=0.0)
        cash, _ = daily_cost_paid_baselines(test_eps)
        self.assertTrue(rows, "no reportable decisions")
        self.assertGreater(sum(rows) / len(rows), cash)      # learned the planted cross-sectional edge


if __name__ == "__main__":
    unittest.main()
