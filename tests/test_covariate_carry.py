"""Point-in-time tests for the cross-window covariate as-of carry (2026-06-29 loader fix).

Fundamentals publish event-sparsely (quarterly filings, monthly snapshots) into event-time partitions, so the
last-known record for most (stock, day) pairs lives in an EARLIER partition than the one being built. The old
loader joined only records INSIDE the current ~3-day window (the model saw market_cap on ~10% of stock-days,
financials on ~2.5%), and its whole-row as-of let a later PARTIAL record (e.g. a monthly market-cap snapshot with
null financials) erase previously published fields with zeros. These tests lock the fix:
  * carry: a value published in window k is visible in window k+1 (and beyond, within cov_carry_days);
  * PIT: it is NOT visible at blocks before its available_timestamp_ms;
  * per-field forward-fill: a later partial record updates its own fields and PRESERVES the others;
  * staleness cap: records older than cov_carry_days are not carried.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rl_quant.datasets.raw_window import COV_FIELDS, RawWindowConfig, build_window

S, BL = 120, 30                                     # tiny 120s session, 4 blocks
SYM = "AAA"


def _open_ms(date_iso: str) -> int:
    from rl_quant.datasets.raw_window import _open_ms as om
    return om(date_iso, RawWindowConfig(session_seconds=S, block_seconds=BL))


def _write_bars(root: Path, window: str, dates: list[str]) -> None:
    rows = {"symbol": [], "timestamp_ms": [], "date_exchange": [],
            "open": [], "high": [], "low": [], "close": [], "volume": []}
    for d in dates:
        o = _open_ms(d)
        for s_off in range(0, S, 5):                # a bar every 5s across the session
            rows["symbol"].append(SYM)
            rows["timestamp_ms"].append(o + s_off * 1000)
            rows["date_exchange"].append(d)
            for f, v in (("open", 10.0), ("high", 10.5), ("low", 9.5), ("close", 10.0), ("volume", 100.0)):
                rows[f].append(v)
    (root / "partitions" / window).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), root / "partitions" / window / "bars.parquet")


def _write_cov(root: Path, window: str, records: list[dict]) -> None:
    cols = {"symbol": [], "available_timestamp_ms": [], **{f: [] for f in COV_FIELDS}}
    for r in records:
        cols["symbol"].append(r["symbol"])
        cols["available_timestamp_ms"].append(r["available_timestamp_ms"])
        for f in COV_FIELDS:
            cols[f].append(r.get(f))                # absent -> None (null in parquet)
    pq.write_table(pa.table(cols), root / "partitions" / window / "covariates.parquet")


class CovariateCarry(unittest.TestCase):
    W1, W2 = "2024-01-02_to_2024-01-03", "2024-01-04_to_2024-01-05"
    D1, D2 = "2024-01-02", "2024-01-04"

    def _build(self, root: Path, window: str, carry_days: int = 400):
        cfg = RawWindowConfig(session_seconds=S, block_seconds=BL, cov_carry_days=carry_days)
        return build_window(root, window, {SYM: 1}, 2, cfg)

    def _root(self, tmp, w1_recs, w2_recs):
        root = Path(tmp)
        _write_bars(root, self.W1, [self.D1])
        _write_bars(root, self.W2, [self.D2])
        _write_cov(root, self.W1, w1_recs)
        _write_cov(root, self.W2, w2_recs)
        return root

    def test_carry_and_pit(self) -> None:
        mc = COV_FIELDS.index("market_cap")
        with tempfile.TemporaryDirectory() as tmp:
            # market_cap published mid-day D1 (after block 1's end, before block 2's)
            avail = _open_ms(self.D1) + 2 * BL * 1000 + 1
            root = self._root(tmp, [dict(symbol=SYM, available_timestamp_ms=avail, market_cap=5e9)], [])
            w1 = self._build(root, self.W1)
            # PIT within the publication window: invisible before availability, visible after
            self.assertEqual(float(w1["cov_blocks"][0, 0, 1, mc]), 0.0)
            self.assertEqual(float(w1["cov_blocks"][0, 1, 1, mc]), 0.0)
            self.assertEqual(float(w1["cov_blocks"][0, 3, 1, mc]), 5e9)
            # CARRY: the NEXT window (no covariates of its own) still sees the last-known value everywhere
            w2 = self._build(root, self.W2)
            self.assertTrue(bool((w2["cov_blocks"][0, :, 1, mc] == 5e9).all()),
                            "published value must carry into later windows")
            # OLD behavior (carry disabled): the value vanishes outside its window
            w2_nc = self._build(root, self.W2, carry_days=0)
            self.assertEqual(float(w2_nc["cov_blocks"][0, :, 1, mc].abs().max()), 0.0)

    def test_per_field_forward_fill_partial_records(self) -> None:
        mc, rev = COV_FIELDS.index("market_cap"), COV_FIELDS.index("financial_revenue")
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(
                tmp,
                # W1: a quarterly filing with revenue (market_cap null)
                [dict(symbol=SYM, available_timestamp_ms=_open_ms(self.D1) - 1000, financial_revenue=7e8)],
                # W2: a monthly market-cap snapshot with NULL financials -- must NOT erase revenue
                [dict(symbol=SYM, available_timestamp_ms=_open_ms(self.D2) - 1000, market_cap=6e9)])
            w2 = self._build(root, self.W2)
            self.assertEqual(float(w2["cov_blocks"][0, 0, 1, mc]), 6e9)      # new field visible
            self.assertEqual(float(w2["cov_blocks"][0, 0, 1, rev]), 7e8)     # old field PRESERVED (was erased to 0)

    def test_staleness_cap(self) -> None:
        mc = COV_FIELDS.index("market_cap")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_w, old_d = "2022-01-03_to_2022-01-04", "2022-01-03"
            _write_bars(root, old_w, [old_d])
            _write_cov(root, old_w, [dict(symbol=SYM, available_timestamp_ms=_open_ms(old_d), market_cap=1e9)])
            _write_bars(root, self.W2, [self.D2])
            (root / "partitions" / self.W2 / "covariates.parquet").unlink(missing_ok=True)
            w2 = self._build(root, self.W2, carry_days=400)       # 2022-01 -> 2024-01 is ~730d > 400d cap
            self.assertEqual(float(w2["cov_blocks"][0, :, 1, mc].abs().max()), 0.0,
                             "records beyond cov_carry_days must not carry")
            w2_long = self._build(root, self.W2, carry_days=2000)  # within a longer horizon they do
            self.assertEqual(float(w2_long["cov_blocks"][0, 0, 1, mc]), 1e9)


if __name__ == "__main__":
    unittest.main()
