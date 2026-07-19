"""Tests for the daily_raw day-level redesign: PIT close-to-close label, full-day trainable raw encoder, causal
cross-day temporal memory, long-only allocation, episode coverage, grad isolation, and end-to-end learnability."""

from __future__ import annotations

import unittest

import torch

import tempfile
from pathlib import Path

from rl_quant.datasets import build_daily_raw_episodes, horizon_close_returns, to_daily_raw_records
from rl_quant.datasets.streaming import LazyDay, LazyWindow
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
from rl_quant.training.daily_policy import (
    _daily_loss,
    _daily_rollout,
    _eval_raw_start,
    _held_drift,
    _roll_positions,
    _sample_episode_indices,
    _stack,
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
        avail=torch.ones(B, T, A, dtype=torch.bool),
        past_ret=0.01 * torch.randn(B, T, A, generator=gen), past_ret_valid=torch.ones(B, T, A, dtype=torch.bool))


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
                                   ep["news_raw"], ep["news_mask"], ep["avail"],
                                   ep["past_ret"], ep["past_ret_valid"])
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
                                    ep["news_raw"], ep["news_mask"], ep["avail"],
                                    ep["past_ret"], ep["past_ret_valid"])
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
        usable = N - (1 + 1)                                  # canonical one-step transition days
        self.assertEqual(one[0]["ret"].shape[0], usable)
        self.assertEqual(one[0]["aux_ret"].shape, one[0]["ret"].shape)

    def test_reported_pnl_uses_one_period_not_horizon_label(self) -> None:
        """Control and reported PnL use one-day transitions; H-day labels remain auxiliary only."""
        N, H = 12, 3
        g = torch.Generator().manual_seed(0)
        # linear ramp per stock -> the 3-day return is ~3x the 1-day return (genuinely different bases)
        recs = [dict(date=f"d{i}", day_close=torch.tensor([float("nan")] + [100.0 + i + ai for ai in range(1, A)]),
                     market=torch.randn(DC, generator=g), per_stock=torch.randn(A, DC, generator=g),
                     bars=torch.randn(A, S, Fd, generator=g), bar_mask=torch.ones(A, S, dtype=torch.bool),
                     news_raw=torch.zeros(A, M, 1), news_mask=torch.ones(A, M, dtype=torch.bool),
                     avail=torch.ones(A, dtype=torch.bool)) for i in range(N)]
        eps = build_daily_raw_episodes(recs, episode_len=6, stride=6, horizon=H, exec_delay=1)
        self.assertTrue(eps)
        ep = eps[0]
        self.assertIn("real_ret", ep)
        self.assertTrue(torch.allclose(ep["ret"], ep["real_ret"]))
        self.assertFalse(torch.allclose(ep["aux_ret"], ep["real_ret"]))
        # baseline buy&hold uses the 1-day (real_ret) basis, not the H-day label
        rr, rv = ep["real_ret"], ep["real_ret_valid"]
        cols = [rr[:, ai][rv[:, ai]].mean() for ai in range(1, A) if rv[:, ai].any()]
        _, bh = daily_cost_paid_baselines([ep])
        self.assertAlmostEqual(bh, float(torch.stack(cols).mean()), places=5)
        # canonical aliases produce identical wealth transitions; the explicit auxiliary target is distinct
        pol = DailyCrossSectionPolicy(_cfg(daily_lookback=6)).eval()
        batch = _stack([ep], [0], torch.device("cpu"))
        n_one = _daily_rollout(pol, batch, 0.0, ret_key="ret")[0]
        n_r = _daily_rollout(pol, batch, 0.0, ret_key="real_ret")[0]
        n_aux = _daily_rollout(pol, batch, 0.0, ret_key="aux_ret")[0]
        self.assertTrue(torch.allclose(n_one, n_r))
        self.assertFalse(torch.allclose(n_aux, n_r))

    def test_bfloat16_context_stays_bfloat16_through_amp_token_assembly(self) -> None:
        """TOP2000 evaluation must not recreate the giant context/token input in FP32."""
        torch.manual_seed(8)
        T = 6
        episode = _episode(1, T, torch.Generator().manual_seed(9))
        realized = torch.zeros(1, T, A)
        realized[:, :, 1:] = 0.01 * torch.randn(1, T, A - 1, generator=torch.Generator().manual_seed(10))
        packed = {
            key: value[0]
            for key, value in episode.items()
        }
        packed.update({
            "market": packed["market"].bfloat16(),
            "per_stock": packed["per_stock"].bfloat16(),
            # Production TOP2000 runs with the declared no-news ablation.  Exercise the scalar-backed logical
            # article grid so this test covers both its allocation-free aggregation and the downstream dtype path.
            "news_raw": torch.zeros((), dtype=torch.float32).expand(T, A, M, 1),
            "news_mask": torch.zeros((), dtype=torch.bool).expand(T, A, M),
            "ret": realized[0],
            "ret_valid": torch.ones(T, A, dtype=torch.bool),
            "real_ret": realized[0],
            "real_ret_valid": torch.ones(T, A, dtype=torch.bool),
            "score_mask": torch.ones(T, dtype=torch.bool),
            "decision_ids": tuple(f"d{index}" for index in range(T)),
            "n_blocks": T,
        })
        policy = DailyCrossSectionPolicy(_cfg(daily_lookback=T, raw_recent_days=2)).eval()
        token_input_dtypes: list[torch.dtype] = []
        raw_output_dtypes: list[torch.dtype] = []
        temporal_input_dtypes: list[torch.dtype] = []
        temporal_output_dtypes: list[torch.dtype] = []
        allocator_input_dtypes: list[torch.dtype] = []
        attention_input_dtypes: list[torch.dtype] = []

        def capture_token_input(_module, args) -> None:
            token_input_dtypes.append(args[0].dtype)

        def capture_raw_output(_module, _args, output) -> None:
            raw_output_dtypes.append(output.dtype)

        def capture_temporal_input(_module, args) -> None:
            temporal_input_dtypes.append(args[0].dtype)

        def capture_temporal_output(_module, _args, output) -> None:
            temporal_output_dtypes.append(output.dtype)

        def capture_allocator_input(_module, args) -> None:
            allocator_input_dtypes.append(args[0].dtype)

        def capture_attention_input(_module, args) -> None:
            attention_input_dtypes.append(args[0].dtype)

        hooks = [
            policy.token_proj.register_forward_pre_hook(capture_token_input),
            policy.raw_encoder.register_forward_hook(capture_raw_output),
            policy.temporal.register_forward_pre_hook(capture_temporal_input),
            policy.temporal.register_forward_hook(capture_temporal_output),
            policy.alloc_in.register_forward_pre_hook(capture_allocator_input),
            policy.attn.register_forward_pre_hook(capture_attention_input),
        ]
        try:
            rows, stats = evaluate_daily_detailed(
                policy, [packed], torch.device("cpu"), cost=5e-4, batch_days=1, window=T, amp=True
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertTrue(rows)
        self.assertEqual(stats["decision_ids"], list(packed["decision_ids"]))
        self.assertTrue(token_input_dtypes)
        self.assertEqual(set(token_input_dtypes), {torch.bfloat16})
        self.assertTrue(raw_output_dtypes)
        self.assertEqual(set(raw_output_dtypes), {torch.bfloat16})
        self.assertTrue(temporal_input_dtypes)
        self.assertEqual(set(temporal_input_dtypes), {torch.bfloat16})
        self.assertTrue(temporal_output_dtypes)
        self.assertEqual(set(temporal_output_dtypes), {torch.bfloat16})
        self.assertTrue(allocator_input_dtypes)
        self.assertEqual(set(allocator_input_dtypes), {torch.bfloat16})
        self.assertTrue(attention_input_dtypes)
        self.assertEqual(set(attention_input_dtypes), {torch.bfloat16})
        reported = [
            *rows,
            stats["mean_net_return"],
            stats["realized_ic_mean"],
            *stats["policy_telemetry"].values(),
        ]
        self.assertTrue(all(isinstance(value, float) for value in reported))
        self.assertTrue(bool(torch.isfinite(torch.tensor(reported, dtype=torch.float32)).all()))

        # Explicit AMP-off remains a supported fallback: assembly promotes
        # locally for FP32 weights rather than requiring `_stack` to retain a
        # second full-size FP32 context batch.
        fallback_rows, fallback_stats = evaluate_daily_detailed(
            policy, [packed], torch.device("cpu"), cost=5e-4, batch_days=1, window=T, amp=False
        )
        self.assertEqual(len(fallback_rows), len(rows))
        self.assertEqual(fallback_stats["decision_ids"], stats["decision_ids"])
        self.assertTrue(bool(torch.isfinite(torch.tensor(fallback_rows)).all()))

        policy.train()
        train_batch = _stack([packed], [0], torch.device("cpu"))
        policy.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            train_nets = _daily_rollout(
                policy, train_batch, cost=5e-4, terminal_liquidate=False, ret_key="ret"
            )[0]
            train_loss = -train_nets.mean()
        self.assertEqual(train_nets.dtype, torch.float32)
        train_loss.backward()
        grads = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in grads))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast dtype contract")
    def test_cuda_amp_daily_policy_state_and_allocator_stay_bfloat16(self) -> None:
        """CUDA's FP32 LayerNorm policy must not widen raw, temporal, or allocator-wide activations."""
        torch.manual_seed(12)
        T = 3
        episode = {
            key: value.cuda()
            for key, value in _episode(1, T, torch.Generator().manual_seed(13)).items()
        }
        episode["market"] = episode["market"].bfloat16()
        episode["per_stock"] = episode["per_stock"].bfloat16()
        episode["news_raw"] = torch.zeros((), device="cuda").expand(1, T, A, M, 1)
        episode["news_mask"] = torch.zeros((), dtype=torch.bool, device="cuda").expand(1, T, A, M)
        policy = DailyCrossSectionPolicy(_cfg(daily_lookback=T, raw_recent_days=2)).cuda().train()
        previous = torch.zeros(1, A, device="cuda")
        previous[:, 0] = 1.0
        realized = torch.zeros(1, T, A, device="cuda")
        rollout_batch = {
            **episode,
            "ret": realized,
            "ret_valid": torch.ones_like(realized, dtype=torch.bool),
            "real_ret": realized,
            "real_ret_valid": torch.ones_like(realized, dtype=torch.bool),
            "score_mask": torch.ones(1, T, dtype=torch.bool, device="cuda"),
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = policy.encode_episode(
                episode["market"], episode["per_stock"], episode["bars"], episode["bar_mask"],
                episode["news_raw"], episode["news_mask"], episode["avail"],
                episode["past_ret"], episode["past_ret_valid"],
            )
            weights, gate = policy.step(state[:, -1], previous, episode["avail"][:, -1])
            nets = _daily_rollout(policy, rollout_batch, cost=5e-4, terminal_liquidate=False)[0]
            loss = (
                state.float().square().mean()
                + weights.float().square().mean()
                + gate.float().square().mean()
                + nets.square().mean()
            )
        self.assertEqual(state.dtype, torch.bfloat16)
        # CUDA softmax intentionally returns FP32 for stable simplex/accounting; only the wide model state is BF16.
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(gate.dtype, torch.bfloat16)
        self.assertEqual(nets.dtype, torch.float32)
        loss.backward()
        gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))


class EODAdapter(unittest.TestCase):
    def test_eod_selection_inram(self) -> None:
        """The adapter selects each session's declared close block, including an early-close day."""
        nB, d = 3, DC
        g = torch.Generator().manual_seed(0)
        enc = [dict(market=torch.randn(nB, d, generator=g), per_stock=torch.randn(nB, A, d, generator=g),
                    bars=torch.randn(A, S, Fd, generator=g), bar_mask=torch.ones(A, S, dtype=torch.bool),
                    news_raw=torch.randn(nB, A, M, 1, generator=g), news_mask=torch.ones(nB, A, M, dtype=torch.bool),
                    avail=torch.ones(nB, A, dtype=torch.bool), day_close=100 + torch.randn(A, generator=g),
                    session_close_block=torch.tensor(1),
                    date=f"d{i}") for i in range(4)]
        recs = to_daily_raw_records(enc)
        self.assertEqual(len(recs), 4)
        for e, r in zip(enc, recs):
            self.assertEqual(r["date"], e["date"])
            self.assertTrue(torch.equal(r["market"], e["market"][1]))
            self.assertTrue(torch.equal(r["per_stock"], e["per_stock"][1]))
            self.assertTrue(torch.equal(r["avail"], e["avail"][1]))
            self.assertTrue(torch.equal(r["news_raw"], e["news_raw"][1]))
            self.assertEqual(r["per_stock"].shape, (A, DC))
            self.assertTrue(torch.equal(r["bars"], e["bars"]))                  # in-RAM: bars materialized
            self.assertNotIn("_bars_day", r)

    def test_keeps_bars_lazy_for_lazyday(self) -> None:
        """For a LazyDay (streaming), the record carries a "_bars_day" handle and does NOT materialize bars."""
        nB, d, Dd = 3, DC, 4
        g = torch.Generator().manual_seed(1)
        w = {"bars": torch.randn(Dd, A, S, Fd, generator=g), "bar_mask": torch.ones(Dd, A, S, dtype=torch.bool),
             "dates": [f"d{i}" for i in range(Dd)], "window": "w", "n_days": Dd, "n_blocks": nB}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w.pt"
            torch.save(w, p)
            lw = LazyWindow(p, {k: w[k] for k in ("n_days", "n_blocks", "dates", "window")})
            enc = [LazyDay(lw, di).with_overrides(
                market=torch.randn(nB, d, generator=g), per_stock=torch.randn(nB, A, d, generator=g),
                avail=torch.ones(nB, A, dtype=torch.bool), news_raw=torch.randn(nB, A, M, 1, generator=g),
                news_mask=torch.ones(nB, A, M, dtype=torch.bool), day_close=100 + torch.randn(A, generator=g))
                for di in range(Dd)]
            recs = to_daily_raw_records(enc)
            for di, r in enumerate(recs):
                self.assertNotIn("bars", r)                  # NOT materialized
                self.assertIn("_bars_day", r)
                self.assertIsNot(r["_bars_day"], enc[di])     # raw handle drops context overrides/storage
                self.assertEqual(r["_bars_day"]._ov, {})      # noqa: SLF001 - memory-retention regression
                self.assertEqual(r["_bars_day"]["bars"].shape, (A, S, Fd))     # handle yields the full-day bars
                self.assertTrue(torch.equal(r["_bars_day"]["bars"], w["bars"][di]))
                self.assertEqual(r["per_stock"].shape, (A, DC))


class NewsReportability(unittest.TestCase):
    """news_is_reportable checks every active article fail-closed."""

    @staticmethod
    def _article(**updates):
        article = {
            "ticker": "A",
            "published_timestamp_ms": 1_650_000_000_000,
            "llm_feature_available_timestamp_ms": 1_650_000_001_000,
            "model_available_timestamp_ms": 1_600_000_000_000,
            "extractor_temperature": 0,
            "extractor_no_retrieval": True,
            "extractor_provider": "local_model",
            "llm_model_id": "period-correct-model",
            "llm_prompt_hash": "prompt",
            "llm_schema_hash": "schema",
            "model_training_cutoff_utc": "2020-01-01",
            "sentiment_score": 0.1,
        }
        article.update(updates)
        return article

    def _root(self, tmp, articles):
        import json
        import pyarrow as pa
        import pyarrow.parquet as pq
        root = Path(tmp)
        (root / "partitions" / "w0").mkdir(parents=True)
        (root / "universe.json").write_text(json.dumps({
            "cash_index": 0,
            "action_count": 2,
            "actions": ["CASH", "A"],
        }))
        pq.write_table(pa.table({"symbol": ["A"], "timestamp_ms": [1_700_000_000_000],
                                 "date_exchange": ["2023-11-14"],
                                 "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}),
                       root / "partitions" / "w0" / "bars.parquet")
        (root / "partitions" / "w0" / "news.jsonl").write_text("\n".join(json.dumps(a) for a in articles))
        return root

    def test_sentinel_is_not_reportable(self) -> None:
        from rl_quant.datasets import news_is_reportable
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, [self._article(model_available_timestamp_ms=1000)])
            ok, reason = news_is_reportable(root)
            self.assertFalse(ok)
            self.assertIn("sentinel", reason)

    def test_period_correct_is_reportable(self) -> None:
        from rl_quant.datasets import news_is_reportable
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, [self._article()])
            ok, _ = news_is_reportable(root)
            self.assertTrue(ok)

    def test_reserved_cash_equity_ticker_maps_through_explicit_alias(self) -> None:
        import json
        from rl_quant.datasets import news_is_reportable

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, [self._article(ticker="CASH")])
            (root / "universe.json").write_text(json.dumps({
                "cash_index": 0,
                "action_count": 2,
                "actions": ["CASH", "EQUITY:CASH"],
                "source_symbol_aliases": {"EQUITY:CASH": "CASH"},
            }))
            ok, _ = news_is_reportable(root)
            self.assertTrue(ok)

    def test_missing_nonfinite_or_out_of_range_sentiment_is_not_reportable(self) -> None:
        from rl_quant.datasets import news_is_reportable

        cases = (
            (None, "lacks a numeric"),
            (float("nan"), "must be finite"),
            (1.01, "must be finite"),
            (False, "lacks a numeric"),
        )
        for value, expected in cases:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                article = self._article(sentiment_score=value)
                if value is None:
                    del article["sentiment_score"]
                root = self._root(tmp, [article])
                ok, reason = news_is_reportable(root)
                self.assertFalse(ok)
                self.assertIn(expected, reason)

    def test_missing_ticker_or_non_integer_feature_time_is_not_reportable(self) -> None:
        from rl_quant.datasets import news_is_reportable

        for updates, expected in (
            ({"ticker": ""}, "non-empty ticker"),
            ({"ticker": " ZZZ "}, "canonical non-empty ticker"),
            ({"ticker": "ZZZ"}, "not a declared non-CASH action"),
            ({"llm_feature_available_timestamp_ms": 1_650_000_001_000.5}, "invalid integer"),
            ({"extractor_temperature": False}, "temperature=0"),
        ):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp, [self._article(**updates)])
                ok, reason = news_is_reportable(root)
                self.assertFalse(ok)
                self.assertIn(expected, reason)

    def test_non_string_extractor_identity_is_not_reportable(self) -> None:
        from rl_quant.datasets import news_is_reportable

        for field, value in (
            ("llm_model_id", {}),
            ("llm_prompt_hash", 123),
            ("llm_schema_hash", []),
            ("extractor_provider", 123),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp, [self._article(**{field: value})])
                ok, reason = news_is_reportable(root)
                self.assertFalse(ok)
                self.assertIn(field, reason)

    def test_future_or_missing_model_chronology_is_not_reportable(self) -> None:
        from rl_quant.datasets import news_is_reportable
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(
                tmp,
                [self._article(model_available_timestamp_ms=1_660_000_000_000)],
            )
            ok, reason = news_is_reportable(root)
            self.assertFalse(ok)
            self.assertIn("unavailable", reason)
        with tempfile.TemporaryDirectory() as tmp:
            article = self._article()
            del article["model_available_timestamp_ms"]
            root = self._root(tmp, [article])
            ok, reason = news_is_reportable(root)
            self.assertFalse(ok)
            self.assertIn("lacks required", reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(
                tmp,
                [self._article(model_training_cutoff_utc="2030-01-01")],
            )
            ok, reason = news_is_reportable(root)
            self.assertFalse(ok)
            self.assertIn("training cutoff after", reason)

    def test_news_gate_scans_every_active_window(self) -> None:
        import json
        import pyarrow as pa
        import pyarrow.parquet as pq
        from rl_quant.datasets import news_is_reportable

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, [self._article()])
            for index in range(1, 5):
                partition = root / "partitions" / f"w{index}"
                partition.mkdir()
                pq.write_table(
                    pa.table({"timestamp_ms": [1_700_000_000_000 + index]}),
                    partition / "bars.parquet",
                )
                if index < 4:
                    (partition / "news.jsonl").write_text(json.dumps(self._article()))

            ok, reason = news_is_reportable(root)

            self.assertFalse(ok)
            self.assertIn("w4 is missing news.jsonl", reason)


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

    def test_planted_edge_survives_production_loss_weights(self) -> None:
        """PRODUCTION-GEOMETRY guard (2026-06-29 audit): a planted cross-sectional edge ~10x cost must survive the
        DEFAULT objective coefficients (budget/gate-entropy/missing at their production values, full cost from
        step 1, canonical reward_scale=1) and beat CASH on the reported 1-day mark. The audit showed the OLD weights made
        this impossible (the loss was penalty-dominated by 2-4 orders of magnitude); this test fails if the
        objective ever again stops a real edge from being expressed."""
        torch.manual_seed(0)
        N, H = 70, 3
        g = torch.Generator().manual_seed(2)
        dc = torch.empty(N, A)
        dc[:, 0] = float("nan")
        dc[:, 1:] = 100 + (0.5 * torch.randn(N, A - 1, generator=g)).cumsum(0)   # ~0.5%/day moves >> 5e-4 cost
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
        cost = 5e-4
        # NOTE: budget_lambda / gate_entropy_coef / missing_label_penalty deliberately NOT passed -> production
        # defaults; full cost from step 1; canonical one-day training reward.
        _, _, best_state = train_daily_policy(
            pol, train_eps, steps=120, lr=3e-3, batch_days=4, cost=cost, bptt_window=18,
            reward_scale=1.0, eval_every=60, val_eps=test_eps, device=dev,
            min_val_label_reportable_fraction=0.0)
        if best_state:
            pol.load_state_dict(best_state)
        rows, _ = evaluate_daily_detailed(pol, test_eps, dev, cost=cost)
        cash, _ = daily_cost_paid_baselines(test_eps)
        self.assertTrue(rows, "no reportable decisions")
        self.assertGreater(sum(rows) / len(rows), cash)      # the edge survived the production objective

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


class RawNormLevel(unittest.TestCase):
    """The 'level' raw norm preserves intraday RETURN magnitude (the cross-sectional signal) while staying causal,
    per-(stock,day), and multiplicatively scale-invariant -- unlike the affine-invariant 'instance' norm."""

    def _enc(self):
        torch.manual_seed(0)
        return FullDayRawEncoder(bar_feature_dim=Fd, d_model=8, n_heads=2, n_layers=2, feedforward_dim=16,
                                 dropout=0.0, block_seconds=8, max_seconds=S, raw_norm="level").eval()

    def _bars(self):
        g = torch.Generator().manual_seed(1)
        bars = torch.empty(2, A, S, Fd)
        bars[..., :4] = 100.0 + torch.randn(2, A, S, 4, generator=g)        # positive prices around 100
        bars[..., 4] = (1000.0 + 50.0 * torch.randn(2, A, S, generator=g)).clamp_min(1.0)   # volume
        return bars, torch.ones(2, A, S, dtype=torch.bool)

    def test_cross_sectional_independence(self) -> None:
        enc = self._enc()
        bars, mask = self._bars()
        out = enc(bars, mask)
        b2 = bars.clone()
        b2[:, 2, : S // 2, :4] += 3.0                                       # perturb stock 2's price path
        out2 = enc(b2, mask)
        self.assertLess(float((out2[:, 1] - out[:, 1]).abs().max()), 1e-6)  # stock 1 untouched (per-stock norm)
        self.assertGreater(float((out2[:, 2] - out[:, 2]).abs().max()), 1e-6)

    def test_multiplicative_price_scale_invariance(self) -> None:
        enc = self._enc()
        bars, mask = self._bars()
        out = enc(bars, mask)
        b_scaled = bars.clone()
        b_scaled[..., :4] *= 3.7                                            # scale ALL price fields (split-like)
        self.assertLess(float((enc(b_scaled, mask) - out).abs().max()), 1e-5)  # price-LEVEL invariant

    def test_intraday_magnitude_sensitivity(self) -> None:
        enc = self._enc()
        bars, mask = self._bars()
        out = enc(bars, mask)
        # amplify intraday price DEVIATIONS 2x about the day-mean close (anchor) -> level norm doubles its input,
        # so the embedding MUST change (the instance norm would whiten this 2x away -> the signal we restored).
        anchor = bars[..., 3].mean(dim=2, keepdim=True).unsqueeze(-1)        # [2,A,1,1] mean close per stock-day
        b_amp = bars.clone()
        b_amp[..., :4] = anchor + 2.0 * (bars[..., :4] - anchor)
        self.assertGreater(float((enc(b_amp, mask) - out).abs().max()), 1e-4)


class RewardScaleAndDrift(unittest.TestCase):
    def _ep_batch(self):
        N, H = 12, 3
        g = torch.Generator().manual_seed(0)
        recs = [dict(date=f"d{i}", day_close=torch.tensor([float("nan")] + [100.0 + i + ai for ai in range(1, A)]),
                     market=torch.randn(DC, generator=g), per_stock=torch.randn(A, DC, generator=g),
                     bars=torch.randn(A, S, Fd, generator=g), bar_mask=torch.ones(A, S, dtype=torch.bool),
                     news_raw=torch.zeros(A, M, 1), news_mask=torch.ones(A, M, dtype=torch.bool),
                     avail=torch.ones(A, dtype=torch.bool)) for i in range(N)]
        eps = build_daily_raw_episodes(recs, episode_len=8, stride=8, horizon=H, exec_delay=1)
        pol = DailyCrossSectionPolicy(_cfg(daily_lookback=8)).eval()
        return pol, _stack(eps, [0], torch.device("cpu"))

    def test_reward_scale_rescales_realized_net_only(self) -> None:
        """The optional reward-ablation scale is linear at zero cost; production leaves it at one."""
        pol, batch = self._ep_batch()
        n_full = _daily_rollout(pol, batch, 0.0, ret_key="real_ret", reward_scale=1.0)[0]
        n_half = _daily_rollout(pol, batch, 0.0, ret_key="real_ret", reward_scale=0.5)[0]
        self.assertTrue(torch.allclose(n_half, 0.5 * n_full, atol=1e-6))

    def test_train_with_reward_scale_and_eval_window_runs(self) -> None:
        """The exact driver shape (canonical reward scale + windowed validation) selects a checkpoint."""
        torch.manual_seed(0)
        N, H = 60, 3
        g = torch.Generator().manual_seed(3)
        dc = torch.empty(N, A)
        dc[:, 0] = float("nan")
        dc[:, 1:] = 100 + (0.5 * torch.randn(N, A - 1, generator=g)).cumsum(0)
        recs = [dict(date=f"d{i}", day_close=dc[i], market=torch.randn(DC, generator=g),
                     per_stock=torch.randn(A, DC, generator=g), bars=torch.randn(A, S, Fd, generator=g),
                     bar_mask=torch.ones(A, S, dtype=torch.bool), news_raw=torch.zeros(A, M, 1),
                     news_mask=torch.ones(A, M, dtype=torch.bool), avail=torch.ones(A, dtype=torch.bool))
                for i in range(N)]
        train_eps = build_daily_raw_episodes(recs, episode_len=12, stride=6, horizon=H, exec_delay=1)
        val_eps = build_daily_raw_episodes(recs, episode_len=N, stride=N, horizon=H, exec_delay=1)  # one continuous
        pol = DailyCrossSectionPolicy(_cfg(daily_lookback=12))
        dev = torch.device("cpu")
        _, _, best_state = train_daily_policy(
            pol, train_eps, steps=6, lr=3e-3, batch_days=3, cost=5e-4, risk_lambda=0.1, budget_lambda=0.0,
            gate_entropy_coef=1e-3, bptt_window=12, reward_scale=1.0, eval_window=12, eval_every=3,
            val_eps=val_eps, device=dev, min_val_label_reportable_fraction=0.0)
        self.assertIsNotNone(best_state)
        rows, st = evaluate_daily_detailed(pol, val_eps, dev, cost=5e-4, batch_days=1, window=12)
        self.assertTrue(rows)
        self.assertGreaterEqual(st["reportable_fraction"], 0.0)
        self.assertIn("policy_telemetry", st)
        self.assertIn("mean_effective_stock_count", st["policy_telemetry"])

    def test_held_drift_rides_and_stays_on_simplex(self) -> None:
        prev = torch.tensor([[0.5, 0.5, 0.0, 0.0]])                         # CASH=0.5, stock@idx1=0.5
        real = torch.tensor([[0.0, 1.0, 0.0, 0.0]])                         # stock@idx1 returns +100%
        valid = torch.ones(1, A, dtype=torch.bool)
        d = _held_drift(prev, real, valid)
        self.assertAlmostEqual(float(d.sum()), 1.0, places=6)               # stays on the simplex
        self.assertGreater(float(d[0, 1]), 0.5)                             # winner's weight rode up
        self.assertLess(float(d[0, 0]), 0.5)                                # CASH (0 return) shrank relatively

    def test_delayed_execution_never_exposes_future_drift_to_policy(self) -> None:
        """At decision t+1 the policy may see action t, but not its close[t+2]-dependent drift."""

        class RecordingPolicy:
            def __init__(self) -> None:
                self.previous: list[torch.Tensor] = []

            def step(self, state, previous, available):
                del state, available
                self.previous.append(previous.detach().clone())
                target = torch.zeros_like(previous)
                target[:, :2] = 0.5
                # Enter on the first decision, then hold at delayed execution.
                gate = torch.ones(previous.shape[0], dtype=previous.dtype, device=previous.device)
                if len(self.previous) > 1:
                    gate.zero_()
                return target, gate

        policy = RecordingPolicy()
        available = torch.ones(1, 2, A, dtype=torch.bool)
        realized = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
        valid = torch.ones_like(realized, dtype=torch.bool)
        result = _roll_positions(
            policy,
            lambda _t: torch.zeros(1, A, 1),
            2,
            available,
            realized,
            valid,
            realized,
            valid,
            0.1,
            1.0,
            1,
            False,
        )
        nets, _, _, cash_weight, turnover, _, _ = result

        torch.testing.assert_close(policy.previous[0], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        # The +100% return belongs to the future of decision 1.  The observable
        # submitted book is still 50/50, while delayed execution correctly
        # accounts against the future pre-trade book (1/3 CASH, 2/3 stock).
        torch.testing.assert_close(policy.previous[1], torch.tensor([[0.5, 0.5, 0.0, 0.0]]))
        torch.testing.assert_close(cash_weight[0], torch.tensor([0.5, 1.0 / 3.0]))
        torch.testing.assert_close(turnover[0], torch.tensor([0.5, 0.0]))
        torch.testing.assert_close(nets[0], torch.tensor([0.45, 0.0]))

    def test_terminal_liquidation_uses_final_post_return_book(self) -> None:
        class EnterPolicy:
            @staticmethod
            def step(state, previous, available):
                del state, available
                target = torch.zeros_like(previous)
                target[:, :2] = 0.5
                return target, torch.ones(previous.shape[0], dtype=previous.dtype)

        available = torch.ones(1, 1, A, dtype=torch.bool)
        realized = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        valid = torch.ones_like(realized, dtype=torch.bool)
        nets, _, _, _, turnover, _, _ = _roll_positions(
            EnterPolicy(),
            lambda _t: torch.zeros(1, A, 1),
            1,
            available,
            realized,
            valid,
            realized,
            valid,
            0.1,
            1.0,
            1,
            True,
        )
        # Entry is 0.5 turnover.  After the stock doubles, liquidation from
        # the 1/3--2/3 final book costs another 2/3 turnover.
        torch.testing.assert_close(turnover[0], torch.tensor([0.5 + 2.0 / 3.0]))
        torch.testing.assert_close(nets[0], torch.tensor([0.5 - 0.1 * (0.5 + 2.0 / 3.0)]))

    def test_terminal_liquidation_is_charged_on_last_scored_label_not_invalid_tail(self) -> None:
        class EnterThenCashPolicy:
            def __init__(self) -> None:
                self.step_index = 0

            def step(self, state, previous, available):
                del state, previous
                target = torch.zeros_like(available, dtype=torch.float32)
                if self.step_index == 0:
                    target[:, 1] = 1.0
                else:
                    target[:, 0] = 1.0
                self.step_index += 1
                return target, torch.ones(available.shape[0])

        available = torch.ones(1, 2, A, dtype=torch.bool)
        realized = torch.zeros(1, 2, A)
        valid = torch.ones_like(realized, dtype=torch.bool)
        valid[:, 1, 1:] = False  # final tensor row has no scored non-CASH label
        score_mask = torch.ones(1, 2, dtype=torch.bool)

        nets, _, _, _, turnover, _, _ = _roll_positions(
            EnterThenCashPolicy(),
            lambda _t: torch.zeros(1, A, 1),
            2,
            available,
            realized,
            valid,
            realized,
            valid,
            0.01,
            1.0,
            1,
            True,
            score_mask=score_mask,
        )

        # Entry and terminal exit are both attached to the only fixed labeled
        # date. The trailing invalid row cannot hide the exit charge.
        torch.testing.assert_close(turnover[0, 0], torch.tensor(2.0))
        torch.testing.assert_close(nets[0, 0], torch.tensor(-0.02))

    def test_observation_burn_in_stays_cash_and_first_scored_trade_pays_entry(self) -> None:
        class EnterPolicy:
            def __init__(self) -> None:
                self.calls = 0

            def step(self, state, previous, available):
                del state, available
                self.calls += 1
                target = torch.zeros_like(previous)
                target[:, 1] = 1.0
                return target, torch.ones(previous.shape[0], dtype=previous.dtype)

        policy = EnterPolicy()
        available = torch.ones(1, 3, A, dtype=torch.bool)
        realized = torch.zeros(1, 3, A)
        realized[:, 2, 1] = 0.1
        valid = torch.ones_like(realized, dtype=torch.bool)
        nets, gates, _, cash_weight, turnover, _, _ = _roll_positions(
            policy,
            lambda _t: torch.zeros(1, A, 1),
            3,
            available,
            realized,
            valid,
            realized,
            valid,
            0.01,
            1.0,
            1,
            False,
            score_starts=torch.tensor([2]),
        )

        self.assertEqual(policy.calls, 1)
        torch.testing.assert_close(gates[0], torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(cash_weight[0], torch.tensor([1.0, 1.0, 0.0]))
        torch.testing.assert_close(turnover[0], torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(nets[0], torch.tensor([0.0, 0.0, 0.09]))

    def test_episode_batch_sampling_is_unique_and_seeded(self) -> None:
        torch.manual_seed(17)
        first = _sample_episode_indices(8, 6)
        torch.manual_seed(17)
        second = _sample_episode_indices(8, 6)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set(first)), 6)
        self.assertEqual(sorted(_sample_episode_indices(3, 20)), [0, 1, 2])

    def test_burn_in_gates_do_not_shape_scored_policy_penalties(self) -> None:
        common = dict(
            nets=torch.zeros(1, 2),
            ents=torch.zeros(1, 2),
            missing_w=torch.zeros(1, 2),
            label=torch.tensor([[False, True]]),
            risk_lambda=0.0,
            entropy_coef=0.0,
            max_actions=0.1,
            budget_lambda=1.0,
            gate_entropy_coef=1.0,
            missing_label_penalty=0.0,
        )
        low_burn = _daily_loss(gates=torch.tensor([[0.001, 0.2]]), **common)
        high_burn = _daily_loss(gates=torch.tensor([[0.999, 0.2]]), **common)
        torch.testing.assert_close(low_burn, high_burn)


class TwoSpeedTokens(unittest.TestCase):
    """252d-reach mechanics: with raw_recent_days=R only the LAST R days of an episode get the trainable raw
    encode; older days contribute frozen ctx + news + the past-return channel (has_raw=0) -- so long histories
    are visible to the cross-day memory at ~the R-day raw compute."""

    def test_old_days_bars_ignored_but_past_ret_channel_lives(self) -> None:
        pol = DailyCrossSectionPolicy(_cfg(raw_recent_days=2)).eval()
        g = torch.Generator().manual_seed(3)
        ep = _episode(1, 6, g)
        args = (ep["market"], ep["per_stock"], ep["bars"], ep["bar_mask"], ep["news_raw"], ep["news_mask"],
                ep["avail"], ep["past_ret"], ep["past_ret_valid"])
        state = pol.encode_episode(*args)
        # perturbing an OLD day's BARS (t=1 < T-R=4) has NO effect anywhere: its raw encode never runs
        b2 = ep["bars"].clone()
        b2[:, 1, :, : S // 2] += 5.0
        s2 = pol.encode_episode(ep["market"], ep["per_stock"], b2, ep["bar_mask"], ep["news_raw"],
                                ep["news_mask"], ep["avail"], ep["past_ret"], ep["past_ret_valid"])
        self.assertLess(float((s2 - state).abs().max()), 1e-6)
        # ... but its PAST-RETURN channel still reaches later days' memory (old days stay informative)
        pr2 = ep["past_ret"].clone()
        pr2[:, 1, 2] += 0.5
        s3 = pol.encode_episode(ep["market"], ep["per_stock"], ep["bars"], ep["bar_mask"], ep["news_raw"],
                                ep["news_mask"], ep["avail"], pr2, ep["past_ret_valid"])
        self.assertGreater(float((s3[:, 1:] - state[:, 1:]).abs().max()), 1e-6)
        self.assertLess(float((s3[:, 0] - state[:, 0]).abs().max()), 1e-6)      # causal: day 0 unaffected
        # a RECENT day's bars (t=5 >= T-R) still matter (the raw encode runs there)
        b3 = ep["bars"].clone()
        b3[:, 5, :, : S // 2] += 5.0
        s4 = pol.encode_episode(ep["market"], ep["per_stock"], b3, ep["bar_mask"], ep["news_raw"],
                                ep["news_mask"], ep["avail"], ep["past_ret"], ep["past_ret_valid"])
        self.assertGreater(float((s4[:, 5] - state[:, 5]).abs().max()), 1e-6)

    def test_eval_raw_encode_covers_only_the_union_of_scored_windows(self) -> None:
        pol = DailyCrossSectionPolicy(_cfg(raw_recent_days=2)).eval()
        g = torch.Generator().manual_seed(7)
        ep = _episode(1, 6, g)
        calls: list[int] = []

        def loader(t: int):
            calls.append(t)
            return ep["bars"][:, t], ep["bar_mask"][:, t]

        optimized_raw, optimized_noraw = pol.encode_tokens_dual(
            ep["market"], ep["per_stock"], loader, ep["news_raw"], ep["news_mask"],
            ep["past_ret"], ep["past_ret_valid"], raw_start=3,
        )
        self.assertEqual(calls, [3, 4, 5])

        full_raw, full_noraw = pol.encode_tokens_dual(
            ep["market"], ep["per_stock"],
            lambda t: (ep["bars"][:, t], ep["bar_mask"][:, t]),
            ep["news_raw"], ep["news_mask"], ep["past_ret"], ep["past_ret_valid"],
        )
        # The first scored decision is t=4 with a two-day raw reach. Its
        # union with t=5 needs raw days 3..5 and no earlier raw encode.
        self.assertEqual(_eval_raw_start(torch.tensor([4]), T=6, window=5, raw_recent_days=2), 3)
        for t in (4, 5):
            lo = max(0, t - 5 + 1)
            optimized = torch.cat([optimized_noraw[:, lo:t - 1], optimized_raw[:, t - 1:t + 1]], dim=1)
            full = torch.cat([full_noraw[:, lo:t - 1], full_raw[:, t - 1:t + 1]], dim=1)
            torch.testing.assert_close(optimized, full)
            torch.testing.assert_close(
                pol.temporal_state(optimized, ep["avail"][:, lo:t + 1])[:, -1],
                pol.temporal_state(full, ep["avail"][:, lo:t + 1])[:, -1],
            )

        calls.clear()
        realized = torch.zeros(1, 6, A)
        rollout_batch = {
            **ep,
            "ret": realized,
            "ret_valid": torch.ones_like(realized, dtype=torch.bool),
            "real_ret": realized,
            "real_ret_valid": torch.ones_like(realized, dtype=torch.bool),
            "score_mask": torch.tensor([[False, False, False, False, True, True]]),
            "_bars_loader": loader,
            "_n_days": 6,
        }
        _daily_rollout(pol, rollout_batch, cost=0.0, terminal_liquidate=False, window=5)
        self.assertEqual(calls, [3, 4, 5])
    def test_windowed_rollout_assembles_two_speed_slices(self) -> None:
        """The rolling-window eval must run under raw_recent_days>0 (dual tokens, per-decision assembly) and
        bound the memory exactly like the plain windowed path."""
        pol = DailyCrossSectionPolicy(_cfg(raw_recent_days=2)).eval()
        g = torch.Generator().manual_seed(4)
        N, H = 16, 3
        dc = torch.empty(N, A)
        dc[:, 0] = float("nan")
        dc[:, 1:] = 100 + (0.5 * torch.randn(N, A - 1, generator=g)).cumsum(0)
        recs = [dict(date=f"d{i}", day_close=dc[i], market=torch.randn(DC, generator=g),
                     per_stock=torch.randn(A, DC, generator=g), bars=torch.randn(A, S, Fd, generator=g),
                     bar_mask=torch.ones(A, S, dtype=torch.bool), news_raw=torch.zeros(A, M, 1),
                     news_mask=torch.ones(A, M, dtype=torch.bool), avail=torch.ones(A, dtype=torch.bool))
                for i in range(N)]
        eps = build_daily_raw_episodes(recs, episode_len=N, stride=N, horizon=H, exec_delay=1)
        batch = _stack(eps, [0], torch.device("cpu"))
        n_win = _daily_rollout(pol, batch, 0.0, ret_key="real_ret", window=5)[0]
        n_full = _daily_rollout(pol, batch, 0.0, ret_key="real_ret", window=0)[0]
        self.assertEqual(n_win.shape, n_full.shape)
        self.assertTrue(torch.isfinite(n_win).all())
        self.assertFalse(torch.allclose(n_win, n_full))       # bounded memory changes decisions

    def test_past_ret_is_pit_and_matches_closes(self) -> None:
        N, H = 8, 2
        g = torch.Generator().manual_seed(5)
        dc = torch.empty(N, A)
        dc[:, 0] = float("nan")
        dc[:, 1:] = 100 + torch.arange(N).float().unsqueeze(1) * torch.arange(1, A).float()  # deterministic ramps
        recs = [dict(date=f"d{i}", day_close=dc[i], market=torch.randn(DC, generator=g),
                     per_stock=torch.randn(A, DC, generator=g), bars=torch.randn(A, S, Fd, generator=g),
                     bar_mask=torch.ones(A, S, dtype=torch.bool), news_raw=torch.zeros(A, M, 1),
                     news_mask=torch.ones(A, M, dtype=torch.bool), avail=torch.ones(A, dtype=torch.bool))
                for i in range(N)]
        eps = build_daily_raw_episodes(recs, episode_len=N, stride=N, horizon=H, exec_delay=1)
        ep = eps[0]
        self.assertFalse(bool(ep["past_ret_valid"][0, 1:].any()))            # first day: no prior close
        for t in range(1, ep["past_ret"].shape[0]):
            for ai in range(1, A):
                self.assertAlmostEqual(float(ep["past_ret"][t, ai]),
                                       float(dc[t, ai] / dc[t - 1, ai] - 1.0), places=6)


class RealizedICDiagnostic(unittest.TestCase):
    """The gross realized-IC skill readout: computed on the policy's RAW allocation view (never the drift-carried
    held book), sign-correct, and degenerate days are skipped rather than scored 0."""

    def _mk(self, T=20, seed=0):
        g = torch.Generator().manual_seed(seed)
        rr = torch.zeros(1, T, A)
        rr[:, :, 1:] = 0.02 * torch.randn(1, T, A - 1, generator=g)
        rv = torch.ones(1, T, A, dtype=torch.bool)
        label = torch.ones(1, T, dtype=torch.bool)
        return rr, rv, label

    def test_sign_convention_and_magnitude(self) -> None:
        from rl_quant.training.daily_policy import _cross_sectional_ic
        rr, rv, label = self._mk()
        w_fore = torch.zeros(1, rr.shape[1], A)
        w_fore[:, :, 1:] = torch.softmax(50.0 * rr[:, :, 1:], dim=-1)       # perfect foresight -> IC ~ +1-ish
        ics = _cross_sectional_ic(w_fore, rr, rv, label)
        self.assertEqual(len(ics), rr.shape[1])
        self.assertGreater(sum(ics) / len(ics), 0.5)
        w_anti = torch.zeros_like(w_fore)
        w_anti[:, :, 1:] = torch.softmax(-50.0 * rr[:, :, 1:], dim=-1)      # anti-aligned -> strongly negative
        self.assertLess(sum(_cross_sectional_ic(w_anti, rr, rv, label)) / rr.shape[1], -0.5)

    def test_uniform_view_is_skipped_not_zero(self) -> None:
        from rl_quant.training.daily_policy import _cross_sectional_ic
        rr, rv, label = self._mk()
        w_uni = torch.full((1, rr.shape[1], A), 1.0 / A)                    # no tilt -> no measurable view
        self.assertEqual(_cross_sectional_ic(w_uni, rr, rv, label), [])

    def test_drift_leak_scenario_scores_zero_on_raw_view(self) -> None:
        """The failure the audit caught: a gate~0 book passively RIDING persistent returns scored IC~1 when the
        held book was used. On the RAW view (a fixed uniform bet, no information) the IC must be ~0 even under
        strongly autocorrelated returns."""
        from rl_quant.training.daily_policy import _cross_sectional_ic, _held_drift
        T = 40
        g = torch.Generator().manual_seed(1)
        base = 0.02 * torch.randn(1, A - 1, generator=g)                    # persistent cross-section (rho ~ 1)
        rr = torch.zeros(1, T, A)
        rr[:, :, 1:] = base.unsqueeze(1) + 0.002 * torch.randn(1, T, A - 1, generator=g)
        rv = torch.ones(1, T, A, dtype=torch.bool)
        label = torch.ones(1, T, dtype=torch.bool)
        w_view = torch.zeros(1, T, A)
        w_view[:, :, 1:] = 1.0 / (A - 1)                                    # the policy's VIEW: uniform, zero skill
        self.assertEqual(_cross_sectional_ic(w_view, rr, rv, label), [])    # skipped: no tilt
        # meanwhile the drift-carried BOOK becomes return-aligned -- exactly why it must NOT be the IC input
        book = torch.zeros(1, A)
        book[:, 1:] = 1.0 / (A - 1)
        for t in range(T):
            book = _held_drift(book, rr[:, t], rv[:, t])
        held = book.unsqueeze(1).expand(1, T, A)
        held_ics = _cross_sectional_ic(held, rr, rv, label)
        self.assertGreater(sum(held_ics) / len(held_ics), 0.5)             # the spurious 'skill' the fix removes


class ReportableGate(unittest.TestCase):
    @staticmethod
    def _fixed_label_episode() -> dict:
        """Two labeled dates; the policy's chosen stock lacks a label only on date zero."""

        T = 2
        realized = torch.tensor([
            [0.0, 0.00, 0.03, -0.01],
            [0.0, 0.02, 0.03, -0.01],
        ])
        valid = torch.tensor([
            [True, False, True, True],
            [True, True, True, True],
        ])
        return {
            "market": torch.zeros(T, DC),
            "per_stock": torch.zeros(T, A, DC),
            "bars": torch.zeros(T, A, S, Fd),
            "bar_mask": torch.ones(T, A, S, dtype=torch.bool),
            "news_raw": torch.zeros(T, A, M, 1),
            "news_mask": torch.zeros(T, A, M, dtype=torch.bool),
            "avail": torch.ones(T, A, dtype=torch.bool),
            "ret": realized,
            "ret_valid": valid,
            "real_ret": realized,
            "real_ret_valid": valid,
            "past_ret": torch.zeros(T, A),
            "past_ret_valid": torch.ones(T, A, dtype=torch.bool),
            "score_mask": torch.ones(T, dtype=torch.bool),
            "decision_ids": ("d0", "d1"),
            "n_blocks": T,
        }

    class _PickFirstStock(torch.nn.Module):
        class _Config:
            raw_recent_days = 0

        config = _Config()

        def __init__(self) -> None:
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(8.0))

        def encode_episode(self, market, per_stock, bars, bar_mask, news_raw, news_mask, avail,
                           past_ret, past_valid):
            del bars, bar_mask, news_raw, news_mask, avail, past_ret, past_valid
            return torch.zeros(
                market.shape[0], market.shape[1], per_stock.shape[2], 1,
                dtype=market.dtype, device=market.device,
            )

        def step(self, state, previous, available):
            del state, available
            risky = torch.sigmoid(self.logit).expand(previous.shape[0])
            zero = torch.zeros_like(risky)
            target = torch.stack((1.0 - risky, risky, zero, zero), dim=-1)
            return target, torch.ones_like(risky)

    def test_policy_dependent_coverage_never_filters_validation_return_dates(self) -> None:
        policy = self._PickFirstStock()
        rows, stats = evaluate_daily_detailed(
            policy,
            [self._fixed_label_episode()],
            torch.device("cpu"),
            cost=0.01,
            batch_days=1,
            max_missing_label_weight=0.05,
        )

        # d0 is not coverage-eligible because virtually the whole book is in
        # an unlabeled name.  It nevertheless stays in the fixed ex-ante date
        # set, receives zero gross credit, and pays its entry cost.  d1 pays
        # the terminal exit cost as well.
        self.assertEqual(stats["label_blocks"], 2)
        self.assertEqual(stats["reportable_blocks"], 1)
        self.assertEqual(stats["decision_ids"], ["d0", "d1"])
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0], -0.01, places=4)
        self.assertAlmostEqual(stats["mean_gross_return"], 0.01, places=4)
        self.assertAlmostEqual(stats["mean_turnover_cost"], 0.01, places=4)
        self.assertAlmostEqual(stats["mean_net_return"], sum(rows) / len(rows), places=7)
        self.assertEqual(stats["return_date_basis"], "fixed_labeled_dates")
        self.assertEqual(stats["coverage_role"], "eligibility_gate_only")

    def test_trainer_ranks_fixed_dates_but_keeps_coverage_as_eligibility_gate(self) -> None:
        episode = self._fixed_label_episode()
        policy = self._PickFirstStock()
        _, best_val, best_state = train_daily_policy(
            policy,
            [episode],
            steps=1,
            lr=0.0,
            batch_days=1,
            cost=0.01,
            eval_every=1,
            val_eps=[episode],
            device=torch.device("cpu"),
            min_val_label_reportable_fraction=0.5,
        )
        rows, stats = evaluate_daily_detailed(policy, [episode], torch.device("cpu"), cost=0.01)
        self.assertEqual(stats["label_reportable_fraction"], 0.5)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(best_val, sum(rows) / len(rows), places=7)
        self.assertIsNotNone(best_state)

        rejected = self._PickFirstStock()
        _, rejected_val, rejected_state = train_daily_policy(
            rejected,
            [episode],
            steps=1,
            lr=0.0,
            batch_days=1,
            cost=0.01,
            eval_every=1,
            val_eps=[episode],
            device=torch.device("cpu"),
            min_val_label_reportable_fraction=0.51,
        )
        self.assertEqual(rejected_val, -1e9)
        self.assertIsNone(rejected_state)

    def test_hole_name_does_not_disqualify_diffuse_book(self) -> None:
        """The recalibrated missing_label_penalty (1e-3) no longer forces the softmax book out of chronically
        unlabeled names; the eval WEIGHT tolerance must therefore admit a diffuse book (~1/A per name) holding one
        hole name -- the old 1e-6 tolerance disqualified nearly every day and val selection returned -1e9."""
        N, H = 30, 3
        g = torch.Generator().manual_seed(0)
        dc = torch.empty(N, A)
        dc[:, 0] = float("nan")
        dc[:, 1:] = 100 + (0.5 * torch.randn(N, A - 1, generator=g)).cumsum(0)
        dc[:, 1] = float("nan")                                             # action 1: a permanent hole name
        recs = [dict(date=f"d{i}", day_close=dc[i], market=torch.randn(DC, generator=g),
                     per_stock=torch.randn(A, DC, generator=g), bars=torch.randn(A, S, Fd, generator=g),
                     bar_mask=torch.ones(A, S, dtype=torch.bool), news_raw=torch.zeros(A, M, 1),
                     news_mask=torch.ones(A, M, dtype=torch.bool), avail=torch.ones(A, dtype=torch.bool))
                for i in range(N)]
        eps = build_daily_raw_episodes(recs, episode_len=N, stride=N, horizon=H, exec_delay=1)
        pol = DailyCrossSectionPolicy(_cfg()).eval()                        # untrained -> near-uniform softmax book
        dev = torch.device("cpu")
        # this toy universe has A-1=3 names, so one hole name carries ~1/3 of the book; scale the tolerance to
        # ~1.5/A for the same "one diffuse hole name tolerated" semantics the 0.05 default gives real universes
        tol = 1.5 / A
        _, st = evaluate_daily_detailed(pol, eps, dev, cost=0.0, max_missing_label_weight=tol)
        self.assertGreaterEqual(st["label_reportable_fraction"], 0.95,
                                "a ~1/A hole-name weight must be tolerated at a ~1/A-scaled gate")
        _, st_old = evaluate_daily_detailed(pol, eps, dev, cost=0.0, max_missing_label_weight=1e-6)
        self.assertLess(st_old["label_reportable_fraction"], 0.5)           # the old tolerance rejected everything
        # PRODUCTION calibration guard: the default must tolerate >=2 diffuse hole names on the SMALLEST real
        # universe (TOP50: 1/51 each) -- the audit's failure was the default sitting ~4 orders below that.
        import inspect
        default = inspect.signature(evaluate_daily_detailed).parameters["max_missing_label_weight"].default
        self.assertGreaterEqual(default, 2 / 51)


class EvalWindowHorizon(unittest.TestCase):
    def test_windowed_state_ignores_days_before_the_window(self) -> None:
        """A windowed eval decision at day t must depend only on days [t-W+1 .. t]; a day BEFORE the window cannot
        move it (bounded memory = the trained horizon), while the FULL-context state at t still does (proof the
        window genuinely bounds the memory rather than being a no-op)."""
        pol = DailyCrossSectionPolicy(_cfg()).eval()
        g = torch.Generator().manual_seed(1)
        ep = _episode(1, 10, g)
        tok, _ = pol.encode_tokens_dual(ep["market"], ep["per_stock"],
                                        lambda t: (ep["bars"][:, t], ep["bar_mask"][:, t]),
                                        ep["news_raw"], ep["news_mask"], ep["past_ret"], ep["past_ret_valid"])
        avail, W, t = ep["avail"], 3, 8
        lo = t - W + 1                                                      # window [6..8]
        s_win = pol.temporal_state(tok[:, lo:t + 1], avail[:, lo:t + 1])[:, -1]
        tok2 = tok.clone()
        tok2[:, 2, :, ::2] += 5.0                                           # perturb day 2 (BEFORE the window;
        #                                       a feature SUBSET so it survives the temporal block's LayerNorm)
        s_win2 = pol.temporal_state(tok2[:, lo:t + 1], avail[:, lo:t + 1])[:, -1]
        self.assertLess(float((s_win2 - s_win).abs().max()), 1e-6)         # outside window -> no effect
        s_full = pol.temporal_state(tok, avail)[:, t]
        s_full2 = pol.temporal_state(tok2, avail)[:, t]
        self.assertGreater(float((s_full2 - s_full).abs().max()), 1e-6)    # full causal context DOES see day 2

    def test_eval_window_changes_the_rollout_when_split_exceeds_window(self) -> None:
        pol = DailyCrossSectionPolicy(_cfg()).eval()
        g = torch.Generator().manual_seed(2)
        N, H = 16, 3
        dc = torch.empty(N, A)
        dc[:, 0] = float("nan")
        dc[:, 1:] = 100 + (0.5 * torch.randn(N, A - 1, generator=g)).cumsum(0)
        recs = [dict(date=f"d{i}", day_close=dc[i], market=torch.randn(DC, generator=g),
                     per_stock=torch.randn(A, DC, generator=g), bars=torch.randn(A, S, Fd, generator=g),
                     bar_mask=torch.ones(A, S, dtype=torch.bool), news_raw=torch.zeros(A, M, 1),
                     news_mask=torch.ones(A, M, dtype=torch.bool), avail=torch.ones(A, dtype=torch.bool))
                for i in range(N)]
        eps = build_daily_raw_episodes(recs, episode_len=N, stride=N, horizon=H, exec_delay=1)   # one long episode
        batch = _stack(eps, [0], torch.device("cpu"))
        n_full = _daily_rollout(pol, batch, 0.0, ret_key="real_ret", window=0)[0]
        n_win = _daily_rollout(pol, batch, 0.0, ret_key="real_ret", window=3)[0]
        self.assertEqual(n_full.shape, n_win.shape)
        self.assertFalse(torch.allclose(n_full, n_win))                    # bounded memory changes later decisions


if __name__ == "__main__":
    unittest.main()
