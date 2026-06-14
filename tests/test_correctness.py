from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.hourly_transformer import (  # noqa: E402
    HourlyDataSplit,
    assert_matching_hourly_schema,
    evaluate_hourly_policy,
)
from rl_quant.strategy_data import (  # noqa: E402
    StrategyDataSplit,
    assert_matching_strategy_schema,
)


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


class BarDatasetTests(unittest.TestCase):
    def test_bar_loader_keeps_simple_and_log_returns_distinct(self) -> None:
        module = load_script("build_hourly_transformer_dataset")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "QQQ_1m.csv"
            with path.open("w", newline="") as sink:
                writer = csv.DictWriter(
                    sink,
                    fieldnames=[
                        "DatetimeUTC",
                        "DatetimeExchange",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "DatetimeUTC": "2026-01-02T14:30:00+00:00",
                        "DatetimeExchange": "2026-01-02T09:30:00-05:00",
                        "Open": "100",
                        "High": "101",
                        "Low": "99",
                        "Close": "100",
                        "Adj Close": "100",
                        "Volume": "10",
                    }
                )
                writer.writerow(
                    {
                        "DatetimeUTC": "2026-01-02T14:31:00+00:00",
                        "DatetimeExchange": "2026-01-02T09:31:00-05:00",
                        "Open": "110",
                        "High": "111",
                        "Low": "109",
                        "Close": "110",
                        "Adj Close": "110",
                        "Volume": "20",
                    }
                )

            rows = module.load_symbol_features(
                path,
                start="2026-01-02T00:00:00+00:00",
                end_exclusive="2026-01-03T00:00:00+00:00",
            )

        second = rows["2026-01-02T14:31:00+00:00"]
        self.assertAlmostEqual(second.bar_return, 0.10)
        self.assertAlmostEqual(second.bar_log_return, math.log(1.10))


class HourlySplitTests(unittest.TestCase):
    def test_split_excludes_rewards_realized_after_split_end(self) -> None:
        module = load_script("build_hourly_transformer_dataset")
        hourly = __import__("rl_quant.hourly_transformer", fromlist=["_build_split"])
        timestamps = [f"2026-01-02T14:{minute:02d}:00+00:00" for minute in range(30, 36)]
        next_timestamps = [
            "2026-01-02T14:31:00+00:00",
            "2099-01-01T00:00:00+00:00",
            "2026-01-02T14:33:00+00:00",
            "2026-01-02T14:34:00+00:00",
            "2026-01-02T14:35:00+00:00",
            "2026-01-02T14:36:00+00:00",
        ]
        payload = {
            "timestamps": timestamps,
            "next_timestamps": next_timestamps,
            "feature_names": ["x"],
            "action_names": ["CASH", "QQQ"],
            "features": torch.arange(6, dtype=torch.float32).reshape(6, 1),
            "action_returns": torch.zeros((6, 2), dtype=torch.float32),
            "bar_interval": "1m",
            "periods_per_year": module.periods_per_year_for_interval("1m"),
        }

        split = hourly._build_split(
            name="train",
            payload=payload,
            lookback=2,
            end_ts="2026-01-02T14:35:00+00:00",
            reward_end_ts="2026-01-02T14:35:00+00:00",
        )

        self.assertNotIn(1, split.valid_start_indices.tolist())
        for index in split.valid_start_indices.tolist():
            self.assertLessEqual(split.next_timestamps[index], "2026-01-02T14:35:00+00:00")


class EvaluationTests(unittest.TestCase):
    def test_evaluation_uses_valid_indices_and_resets_at_gaps(self) -> None:
        class FixedPolicy(nn.Module):
            def forward(self, state_windows: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 1.0]], device=state_windows.device).repeat(state_windows.shape[0], 1)

        valid = torch.tensor([2, 3, 5], dtype=torch.long)
        valid_mask = torch.zeros(7, dtype=torch.bool)
        valid_mask[valid] = True
        data = HourlyDataSplit(
            name="test",
            timestamps=[f"t{i}" for i in range(7)],
            next_timestamps=[f"t{i + 1}" for i in range(7)],
            feature_names=["x"],
            action_names=["CASH", "QQQ"],
            features=torch.zeros((7, 1), dtype=torch.float32),
            action_returns=torch.zeros((7, 2), dtype=torch.float32),
            session_dates=None,
            valid_start_indices=valid,
            valid_index_mask=valid_mask,
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=2,
            bar_interval="1m",
        )

        result = evaluate_hourly_policy(
            data,
            FixedPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            capture_rollout=True,
        )

        self.assertEqual([row["timestamp"] for row in result.rollout_records], ["t2", "t3", "t5"])
        self.assertEqual([row["segment_reset"] for row in result.rollout_records], [1, 0, 1])


class SchemaTests(unittest.TestCase):
    def test_hourly_schema_checks_names_and_order(self) -> None:
        valid = torch.tensor([1], dtype=torch.long)
        mask = torch.tensor([False, True, False])
        base = HourlyDataSplit(
            name="train",
            timestamps=["t0", "t1", "t2"],
            next_timestamps=["t1", "t2", "t3"],
            feature_names=["x"],
            action_names=["CASH", "QQQ"],
            features=torch.zeros((3, 1)),
            action_returns=torch.zeros((3, 2)),
            session_dates=None,
            valid_start_indices=valid,
            valid_index_mask=mask,
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=1,
        )
        changed = replace(base, name="val", action_names=["QQQ", "CASH"])

        with self.assertRaises(ValueError):
            assert_matching_hourly_schema(base, changed)

    def test_strategy_schema_checks_names_and_order(self) -> None:
        base = StrategyDataSplit(
            name="train",
            dates=["2026-01-02", "2026-01-05"],
            feature_names=["x"],
            action_names=["BH_QQQ", "CASH"],
            features=torch.zeros((2, 1)),
            action_returns=torch.zeros((2, 2)),
            valid_start_indices=torch.tensor([0]),
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=1,
        )
        changed = replace(base, name="val", feature_names=["y"])

        with self.assertRaises(ValueError):
            assert_matching_strategy_schema(base, changed)


if __name__ == "__main__":
    unittest.main()
