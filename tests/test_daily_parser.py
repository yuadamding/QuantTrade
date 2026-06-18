from __future__ import annotations

import unittest
from _support import load_script


class DailyParserTests(unittest.TestCase):
    def test_daily_parser_tolerates_ragged_yahoo_arrays(self) -> None:
        module = load_script("download_daily_ohlcv")
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [0, 86_400, 172_800],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10.0],
                                    "high": [11.0],
                                    "low": [9.0],
                                    "close": [10.5],
                                    "volume": [100],
                                }
                            ],
                            "adjclose": [{"adjclose": []}],
                        },
                    }
                ],
                "error": None,
            }
        }

        rows = module.parse_chart_payload(payload, "QQQ")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Close"], "10.500000")
        self.assertEqual(rows[0]["Adj Close"], "")
