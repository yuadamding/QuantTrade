from __future__ import annotations

import unittest
import torch
from rl_quant.action_risk import (
    ExposureConstraintConfig,
    action_concentration,
    action_weight_tensor,
    build_action_metadata,
    reportability_flags,
    stable_action_metadata_hash,
    stable_action_risk_config_hash,
    trade_notional,
)


class ActionRiskTests(unittest.TestCase):
    def test_action_metadata_scales_leveraged_etf_weight(self) -> None:
        metadata = build_action_metadata(["CASH", "QQQ", "SOXL", "SOXS"])

        soxl = metadata[2]
        soxs = metadata[3]
        self.assertEqual(soxl.group, "semiconductor")
        self.assertEqual(soxl.leverage, 3.0)
        self.assertAlmostEqual(soxl.max_weight, 1.0 / 3.0)
        self.assertTrue(soxs.inverse)

    def test_trade_notional_scales_cost_by_position_weight(self) -> None:
        metadata = build_action_metadata(["CASH", "SOXL", "QQQ"])
        weights = action_weight_tensor(metadata, device="cpu", max_effective_leverage=1.0)

        cash_to_soxl = trade_notional(torch.tensor([0]), torch.tensor([1]), weights)
        soxl_to_qqq = trade_notional(torch.tensor([1]), torch.tensor([2]), weights)

        self.assertAlmostEqual(float(cash_to_soxl[0].item()), 1.0 / 3.0)
        self.assertAlmostEqual(float(soxl_to_qqq[0].item()), 1.0 + 1.0 / 3.0)

    def test_action_metadata_and_risk_config_hashes_are_stable(self) -> None:
        metadata = build_action_metadata(["CASH", "QQQ", "SOXL"])
        config = ExposureConstraintConfig(max_same_group_share_per_day=0.5)

        self.assertEqual(stable_action_metadata_hash(metadata), stable_action_metadata_hash(metadata))
        self.assertEqual(stable_action_risk_config_hash(config), stable_action_risk_config_hash(config))
        self.assertNotEqual(
            stable_action_risk_config_hash(config),
            stable_action_risk_config_hash(ExposureConstraintConfig(max_same_group_share_per_day=0.25)),
        )

    def test_concentration_and_reportability_flag_leveraged_group_collapse(self) -> None:
        metadata = build_action_metadata(["CASH", "SOXL", "SOXS"])
        records = [
            {"asset": "SOXL", "gross_return": 0.01, "bar_return": 0.01, "equity": 1.01, "timestamp": "t1"},
            {"asset": "SOXS", "gross_return": -0.01, "bar_return": -0.01, "equity": 1.00, "timestamp": "t2"},
        ]

        concentration = action_concentration(records, action_meta=metadata)
        flags = reportability_flags(
            test_metrics={"total_return": -0.01},
            baselines={
                "CASH": {"test": {"total_return": 0.0}},
                "BuyAndHold_QQQ": {"test": {"total_return": 0.01}},
            },
            concentration=concentration,
            max_group_share=0.75,
            max_leveraged_share=0.50,
        )

        self.assertEqual(concentration["max_group"], "semiconductor")
        self.assertEqual(concentration["max_group_share"], 1.0)
        self.assertFalse(flags["reportable"])
        self.assertIn("test_return_below_cash", flags["reasons"])
        self.assertIn("max_group_share_exceeds_limit", flags["reasons"])

    def test_reportability_ignores_cash_for_risky_group_concentration(self) -> None:
        metadata = build_action_metadata(["CASH", "QQQ"])
        concentration = action_concentration(
            [{"asset": "CASH", "bar_return": 0.0, "equity": 1.0, "timestamp": "t1"}],
            action_meta=metadata,
        )

        flags = reportability_flags(
            test_metrics={"total_return": 0.0},
            baselines={"CASH": {"test": {"total_return": 0.0}}},
            concentration=concentration,
            max_group_share=0.75,
            max_leveraged_share=0.50,
        )

        self.assertEqual(concentration["max_group"], "cash")
        self.assertEqual(concentration["max_risky_group_share"], 0.0)
        self.assertTrue(flags["reportable"])
