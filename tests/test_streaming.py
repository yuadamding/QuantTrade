"""Streaming (lazy disk) path must be BIT-EQUIVALENT to the in-RAM path -- a data-loader bug is silently-wrong
training. Covers: LazyWindow/LazyDay vs materialized slices, flatten_days lazy vs in-RAM, and the daily_raw
episode builder keeping bars lazy (handles) while the small fields match the in-RAM stack."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rl_quant.datasets import build_daily_raw_episodes, flatten_days
from rl_quant.datasets.streaming import LazyDay, LazyWindow, TENSOR_KEYS, set_cache_windows

A, S, Fd, M, C, nB = 4, 12, 5, 3, 3, 2


def _nan_safe_eq(a, b):
    if a.shape != b.shape:
        return False
    return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))


def _synthetic_window(Dd=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    ret = torch.randn(Dd, nB, A, generator=g)
    ret[:, :, 0] = 0.0
    w = {
        "bars": torch.randn(Dd, A, S, Fd, generator=g), "bar_mask": torch.ones(Dd, A, S, dtype=torch.bool),
        "cov_blocks": torch.randn(Dd, nB, A, C, generator=g), "news_raw": torch.randn(Dd, nB, A, M, 1, generator=g),
        "news_mask": torch.ones(Dd, nB, A, M, dtype=torch.bool), "avail": torch.ones(Dd, nB, A, dtype=torch.bool),
        "ret": ret, "ret_valid": torch.ones(Dd, nB, A, dtype=torch.bool),
        "day_open": 100 + torch.randn(Dd, A, generator=g), "day_close": 100 + torch.randn(Dd, A, generator=g),
        "dates": [f"2022-01-{i+3:02d}" for i in range(Dd)], "window": "w0", "n_days": Dd, "n_blocks": nB,
    }
    return w


class StreamingEquivalence(unittest.TestCase):
    def test_lazywindow_lazyday_bit_exact(self) -> None:
        w = _synthetic_window()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w.pt"
            torch.save(w, p)
            set_cache_windows(2)
            lw = LazyWindow(p, {k: w[k] for k in ("n_days", "n_blocks", "dates", "window")})
            self.assertEqual(lw["n_days"], w["n_days"])
            self.assertEqual(lw["dates"], w["dates"])
            for di in range(w["n_days"]):
                ld = LazyDay(lw, di)
                self.assertEqual(ld["date"], w["dates"][di])
                self.assertEqual(ld["n_blocks"], w["n_blocks"])
                for k in TENSOR_KEYS:
                    self.assertTrue(_nan_safe_eq(ld[k], w[k][di]), f"{k}[{di}] mismatch")

    def test_flatten_days_lazy_matches_inram(self) -> None:
        w = _synthetic_window()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w.pt"
            torch.save(w, p)
            lw = LazyWindow(p, {k: w[k] for k in ("n_days", "n_blocks", "dates", "window")})
            in_ram = flatten_days([w])
            lazy = flatten_days([lw])
            self.assertEqual(len(in_ram), len(lazy))
            self.assertTrue(all(isinstance(d, LazyDay) for d in lazy))
            for a, b in zip(in_ram, lazy):
                self.assertEqual(a["date"], b["date"])
                for k in TENSOR_KEYS:
                    self.assertTrue(_nan_safe_eq(a[k], b[k]))

    def test_daily_raw_episodes_stream_keeps_bars_lazy(self) -> None:
        """The streaming episode builder must keep per-day bar HANDLES (no [N,A,S,F] stack) while its small fields
        (market/per_stock/ret/avail/news) match the in-RAM episode exactly, and the handle yields the same bars."""
        w = _synthetic_window(Dd=8)
        d_model = 6
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w.pt"
            torch.save(w, p)
            lw = LazyWindow(p, {k: w[k] for k in ("n_days", "n_blocks", "dates", "window")})
            days = flatten_days([lw])                          # LazyDay handles (bars lazy)
            g = torch.Generator().manual_seed(1)
            in_recs, st_recs = [], []
            for di, ld in enumerate(days):
                market = torch.randn(d_model, generator=g)
                per_stock = torch.randn(A, d_model, generator=g)
                small = {"date": ld["date"], "day_close": ld["day_close"], "avail": ld["avail"][nB - 1],
                         "market": market, "per_stock": per_stock,
                         "news_raw": ld["news_raw"][nB - 1], "news_mask": ld["news_mask"][nB - 1]}
                in_recs.append({**small, "bars": ld["bars"], "bar_mask": ld["bar_mask"]})   # in-RAM: materialized bars
                st_recs.append({**small, "_bars_day": ld})                                  # streaming: lazy handle
            in_eps = build_daily_raw_episodes(in_recs, episode_len=4, stride=2, horizon=2, exec_delay=1)
            st_eps = build_daily_raw_episodes(st_recs, episode_len=4, stride=2, horizon=2, exec_delay=1)
            self.assertEqual(len(in_eps), len(st_eps))
            self.assertTrue(len(in_eps) >= 1)
            for ie, se in zip(in_eps, st_eps):
                self.assertIn("bars", ie)
                self.assertNotIn("bars", se)        # streaming episode does NOT pre-stack bars
                self.assertIn("bars_days", se)
                for k in ("market", "per_stock", "ret", "ret_valid", "avail", "news_raw", "news_mask"):
                    self.assertTrue(_nan_safe_eq(ie[k], se[k]), f"{k} differs")
                for t in range(ie["bars"].shape[0]):                       # the handle yields the same per-day bars
                    self.assertTrue(_nan_safe_eq(se["bars_days"][t]["bars"], ie["bars"][t]))


if __name__ == "__main__":
    unittest.main()
