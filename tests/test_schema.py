from __future__ import annotations

import unittest
from dataclasses import replace
import torch
from rl_quant.hourly_transformer import (
    HourlyDataSplit,
    assert_matching_hourly_schema,
)
from rl_quant.strategy_data import (
    StrategyDataSplit,
    assert_matching_strategy_schema,
)


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
            valid_index_mask=torch.tensor([True, False]),
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=1,
        )
        changed = replace(base, name="val", feature_names=["y"])

        with self.assertRaises(ValueError):
            assert_matching_strategy_schema(base, changed)
