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

    def test_hourly_split_carries_action_return_basis_when_present(self) -> None:
        # The direct-hourly loader now propagates the action-return basis from the payload onto HourlyDataSplit,
        # so ReturnBasis.from_mapping(split) is meaningful on the direct-hourly reportable path. Default-preserving:
        # a payload without the basis yields an all-None (not complete) basis.
        from rl_quant.protocol.action_return_basis import ReturnBasis

        hourly = __import__("rl_quant.datasets.hourly", fromlist=["_build_split"])
        base = {
            "timestamps": [f"2026-01-02T14:{m:02d}:00+00:00" for m in range(30, 34)],
            "next_timestamps": [f"2026-01-02T14:{m:02d}:00+00:00" for m in range(31, 35)],
            "feature_names": ["x"],
            "action_names": ["CASH", "QQQ"],
            "features": torch.zeros((4, 1), dtype=torch.float32),
            "action_returns": torch.zeros((4, 2), dtype=torch.float32),
        }
        # Legacy payload (no basis) -> all-None, not complete (default-preserving).
        legacy = hourly._build_split(name="train", payload=base, lookback=1)
        self.assertIsNone(legacy.action_return_weight_semantics)
        self.assertFalse(ReturnBasis.from_mapping(legacy).is_complete())
        # A basis-carrying payload (a v2 direct-hourly build) -> the split carries it and the basis is complete.
        v2 = {
            **base,
            "action_return_weight_semantics": "full_capital_single_slot_returns",
            "action_return_formula": "clipped_simple_return(decision_bar_close, next_bar_close)",
            "action_return_clip_min": -1.0,
            "action_return_clip_max": 1.0,
            "action_return_semantics_version": "v1",
            "action_return_fill_convention": "decision_bar_close_to_next_bar_close",
            "action_return_basis_version": "v2",
            "action_return_entry_fill_rule": "decision_bar_close",
            "action_return_exit_fill_rule": "next_bar_close",
            "action_return_execution_latency_ms": 0,
            "action_return_source_bar_interval": "1h",
            "action_return_price_source": "bar_close",
        }
        split = hourly._build_split(name="train", payload=v2, lookback=1)
        self.assertEqual(split.action_return_entry_fill_rule, "decision_bar_close")
        self.assertEqual(split.action_return_execution_latency_ms, 0)
        self.assertTrue(ReturnBasis.from_mapping(split).is_complete())

    def test_direct_hourly_builder_emits_complete_truthful_basis(self) -> None:
        # The direct-hourly builder emits a COMPLETE, VALID v2 basis that truthfully describes its computation
        # (clipped_simple_return(decision_bar_close, next_bar_close), zero execution latency). A run on a freshly
        # built dataset can therefore pass --strict-return-basis.
        from rl_quant.protocol.action_return_basis import ReturnBasis

        module = load_script("build_hourly_transformer_dataset")
        basis = ReturnBasis.from_mapping(module._direct_hourly_action_return_basis("1h"))
        self.assertTrue(basis.is_complete())              # complete v2 (all structured fields present)
        self.assertEqual(basis.validation_errors(), [])   # valid (latency 0 ok; clips finite, ordered)
        self.assertEqual(basis.execution_latency_ms, 0)    # truthful: zero-latency decision-bar-close fill
        self.assertEqual(basis.entry_fill_rule, "decision_bar_close")
        self.assertEqual(basis.exit_fill_rule, "next_bar_close")
        self.assertEqual(basis.source_bar_interval, "1h")  # carries the actual bar interval

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
