"""Load-time bar resampling (`bar_seconds`): the same raw OHLCV fields on a coarser grid.

The lever that fits TOP2000 training in one day: at bar_seconds=60 the encoder sees 1-minute OHLCV tokens
(open=first, high=max, low=min, close=last, volume=sum per slot) resampled from the raw 1-second rows at load
time. Everything ECONOMIC stays on the raw rows' real timestamps: T+1 labels, day open/close, availability and
the PIT covariate/news joins must be IDENTICAL between grids -- only the model's bar tokens coarsen.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rl_quant.datasets.raw_window import RawWindowConfig, build_window

SESSION, BLOCK, GRID = 120, 30, 15                 # 120s session, 4 blocks, 15s slots (8 slots, 2 per block)
SYM = "AAA"
WIN, DAY = "2024-01-02_to_2024-01-03", "2024-01-02"


def _open_ms(date_iso: str) -> int:
    from rl_quant.datasets.raw_window import _open_ms as om
    return om(date_iso, RawWindowConfig(session_seconds=SESSION, block_seconds=BLOCK))


def _write_root(tmp: Path, rows: list[tuple[int, float, float, float, float, float]]) -> Path:
    """rows: (second_offset, open, high, low, close, volume) raw 1s rows for SYM on DAY."""
    o = _open_ms(DAY)
    cols = {"symbol": [], "timestamp_ms": [], "date_exchange": [],
            "open": [], "high": [], "low": [], "close": [], "volume": []}
    for s_off, op, hi, lo, cl, vol in rows:
        cols["symbol"].append(SYM)
        cols["timestamp_ms"].append(o + s_off * 1000)
        cols["date_exchange"].append(DAY)
        for k, v in (("open", op), ("high", hi), ("low", lo), ("close", cl), ("volume", vol)):
            cols[k].append(v)
    (tmp / "partitions" / WIN).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), tmp / "partitions" / WIN / "bars.parquet")
    return tmp


def _build(root: Path, gs: int):
    cfg = RawWindowConfig(session_seconds=SESSION, block_seconds=BLOCK, bar_seconds=gs)
    return build_window(root, WIN, {SYM: 1}, 2, cfg)


class ResampleAggregation(unittest.TestCase):
    ROWS = [
        # slot 0 (seconds 0..14): three raw rows -> open=first, close=last, high=max, low=min, vol=sum
        (2, 10.0, 10.5, 9.9, 10.2, 100.0),
        (7, 10.2, 11.0, 10.1, 10.9, 50.0),
        (14, 10.9, 10.9, 10.0, 10.1, 25.0),
        # slot 3 (seconds 45..59): one row
        (50, 12.0, 12.5, 11.5, 12.2, 40.0),
        # slot 7 (seconds 105..119): two rows
        (110, 13.0, 13.1, 12.9, 13.0, 10.0),
        (119, 13.0, 14.0, 12.5, 13.7, 20.0),
    ]

    def test_ohlcv_slot_aggregation_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _build(_write_root(Path(tmp), self.ROWS), gs=GRID)
            bars, mask = w["bars"][0, 1], w["bar_mask"][0, 1]        # [slots, F], [slots]
            self.assertEqual(bars.shape[0], SESSION // GRID)         # 8 slots
            self.assertEqual([bool(m) for m in mask], [True, False, False, True, False, False, False, True])
            for got, want in zip((float(bars[0, i]) for i in range(5)),
                                 (10.0, 11.0, 9.9, 10.1, 175.0)):    # first/max/min/last/sum
                self.assertAlmostEqual(got, want, places=5)
            self.assertAlmostEqual(float(bars[3, 4]), 40.0, places=5)   # single-row slot passes through
            self.assertAlmostEqual(float(bars[7, 0]), 13.0, places=5)
            self.assertAlmostEqual(float(bars[7, 3]), 13.7, places=5)
            self.assertAlmostEqual(float(bars[7, 1]), 14.0, places=5)

    def test_labels_day_prices_and_avail_are_grid_invariant(self) -> None:
        """Everything economic is computed from the raw rows' real timestamps: identical across grids."""
        rows = [(s, 100.0 + s, 100.5 + s, 99.5 + s, 100.0 + s, 10.0) for s in range(0, SESSION, 5)]
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            w1 = _build(_write_root(Path(t1), rows), gs=1)
            w2 = _build(_write_root(Path(t2), rows), gs=GRID)
            import torch
            self.assertTrue(torch.equal(w1["ret_valid"], w2["ret_valid"]))
            nan_eq = torch.isnan(w1["ret"]) & torch.isnan(w2["ret"])
            self.assertTrue(bool((nan_eq | (w1["ret"] == w2["ret"])).all()))
            for k in ("day_open", "day_close"):
                a, b = w1[k], w2[k]
                self.assertTrue(bool(((torch.isnan(a) & torch.isnan(b)) | (a == b)).all()), k)
            self.assertTrue(torch.equal(w1["avail"], w2["avail"]))
            self.assertEqual(w1["n_blocks"], w2["n_blocks"])         # decision grid unchanged
            self.assertEqual(w2["bars"].shape[2], SESSION // GRID)   # only the token grid coarsens

    def test_grid_must_divide_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_root(Path(tmp), self.ROWS)
            with self.assertRaises(ValueError):
                _build(root, gs=7)                                   # 30 % 7 != 0


class DesignValidation(unittest.TestCase):
    def test_bar_seconds_validated_and_top2000_is_one_day_sized(self) -> None:
        from rl_quant.training.designs import DESIGNS, Phase1Design
        d = DESIGNS["daily_raw_top2000"]
        self.assertEqual(d.bar_seconds, 60)
        self.assertEqual(DESIGNS["daily_raw_252"].bar_seconds, 60)
        with self.assertRaises(ValueError):
            Phase1Design("bad", "t", session_seconds=23400, block_seconds=300, bar_seconds=7, d_model=24,
                         enc_layers=1, enc_heads=2, policy_token_dim=24, policy_layers=1, policy_heads=2,
                         ssl_steps=1, policy_steps=1, ssl_batch_size=1, batch_days=1)


if __name__ == "__main__":
    unittest.main()
