from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import torch
from _support import load_script


class HourlySplitTests(unittest.TestCase):
    def test_next_timestamps_are_required(self) -> None:
        from rl_quant.hourly_transformer import build_hourly_splits

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            torch.save(
                {
                    "timestamps": ["t0", "t1", "t2"],
                    "feature_names": ["x"],
                    "action_names": ["CASH"],
                    "features": torch.zeros((3, 1), dtype=torch.float32),
                    "action_returns": torch.zeros((3, 1), dtype=torch.float32),
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "next_timestamps"):
                build_hourly_splits(
                    dataset_path=path,
                    lookback=1,
                    train_end="t1",
                    val_end="t2",
                    test_start="t2",
                )

    def test_split_excludes_rewards_realized_after_split_end(self) -> None:
        module = load_script("build_hourly_transformer_dataset")
        hourly = __import__("rl_quant.datasets.hourly", fromlist=["_build_split"])
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

    def test_direct_hourly_split_orders_mixed_timezones_by_utc_instant(self) -> None:
        hourly = __import__("rl_quant.datasets.hourly", fromlist=["_build_split"])
        payload = {
            "timestamps": [
                "2026-01-02T14:30:00+00:00",
                "2026-01-02T09:45:00-05:00",
                "2026-01-02T15:00:00+00:00",
                "2026-01-02T10:15:00-05:00",
            ],
            "next_timestamps": [
                "2026-01-02T09:45:00-05:00",
                "2026-01-02T15:00:00+00:00",
                "2026-01-02T10:15:00-05:00",
                "2026-01-02T15:30:00+00:00",
            ],
            "feature_names": ["x"],
            "action_names": ["CASH", "QQQ"],
            "features": torch.zeros((4, 1), dtype=torch.float32),
            "action_returns": torch.zeros((4, 2), dtype=torch.float32),
        }

        split = hourly._build_split(name="train", payload=payload, lookback=1)

        self.assertEqual(split.timestamps[1], "2026-01-02T09:45:00-05:00")

    def test_direct_hourly_split_rejects_finite_returns_for_invalid_actions(self) -> None:
        hourly = __import__("rl_quant.datasets.hourly", fromlist=["_build_split"])
        payload = {
            "timestamps": [f"2026-01-02T14:{minute:02d}:00+00:00" for minute in range(30, 34)],
            "next_timestamps": [f"2026-01-02T14:{minute:02d}:00+00:00" for minute in range(31, 35)],
            "feature_names": ["x"],
            "action_names": ["CASH", "QQQ"],
            "features": torch.zeros((4, 1), dtype=torch.float32),
            "action_returns": torch.zeros((4, 2), dtype=torch.float32),
            "action_valid_mask": torch.tensor([[True, False], [True, True], [True, True], [True, True]]),
        }

        with self.assertRaisesRegex(ValueError, "Invalid action_returns"):
            hourly._build_split(name="train", payload=payload, lookback=1)
