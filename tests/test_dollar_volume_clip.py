"""Regression: the market-context reducer must clip per-bar volume in dollar_volume (a negative-volume tick is
treated as 0), consistent with the already-clipped `volume` feature. Pairs with the identical QuantTradeData port
fix so the two repos stay byte-parity exact. Research/backtest only."""
from __future__ import annotations

import unittest


class DollarVolumeClipTests(unittest.TestCase):
    def _context(self, test_bar_volume: float):
        try:
            import numpy as np
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("numpy/pandas required")
        from rl_quant.features.stock_second_context import (
            StockSecondContextConfig,
            build_market_context_from_frames,
        )
        base = 1_750_000_000_000
        n = 120
        frames = {}
        for sym in ("AAA", "BBB"):
            ts = base + np.arange(n, dtype=np.int64) * 1000
            close = np.full(n, 100.0)
            vol = np.full(n, 1000.0)
            vol[60] = test_bar_volume  # the bar under test (inside the 120s context block)
            frames[sym] = pd.DataFrame({
                "timestamp_ms": ts, "close": close, "high": close + 0.1, "low": close - 0.1,
                "volume": vol, "vwap": close, "transactions": np.full(n, 5.0),
            })
        cfg = StockSecondContextConfig(decision_interval="5m", context_seconds=120, block_seconds=120,
                                       min_active_symbols=1)
        ctx, _, _ = build_market_context_from_frames(frames, decision_timestamps_ms=[base + 119_000], config=cfg)
        return np.nan_to_num(ctx.numpy())

    def test_negative_volume_is_clipped_to_zero(self) -> None:
        import numpy as np
        # A -500 volume tick must produce the SAME context as a 0 tick (clip applied); pre-fix they diverged.
        self.assertTrue(np.array_equal(self._context(-500.0), self._context(0.0)),
                        "negative volume must be clipped to 0 in dollar_volume")

    def test_not_vacuous_volume_affects_dollar_volume_features(self) -> None:
        import numpy as np
        # Sanity: the bar's (non-negative) volume DOES move the dollar-volume features, so the test exercises them.
        self.assertFalse(np.array_equal(self._context(0.0), self._context(50_000.0)),
                         "the bar volume must affect dollar-volume features")


if __name__ == "__main__":
    unittest.main()
