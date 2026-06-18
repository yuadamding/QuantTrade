from __future__ import annotations

import unittest
import torch
from rl_quant.action_risk import (
    ExposureConstraintConfig,
    action_concentration,
    action_is_inverse_tensor,
    action_is_leveraged_tensor,
    action_leverage_tensor,
    action_weight_tensor,
    apply_exposure_masks,
    build_action_metadata,
    group_ids_for_actions,
    reportability_flags,
    stable_action_metadata_hash,
    stable_action_risk_config_hash,
    trade_notional,
)
from rl_quant.trading_constraints import apply_notional_aware_hysteresis


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

    def test_notional_aware_hysteresis_uses_action_weight_cost(self) -> None:
        weights = torch.tensor([1.0, 1.0 / 3.0])
        action = apply_notional_aware_hysteresis(
            torch.tensor([[0.0, 4.0]]),
            torch.tensor([0]),
            torch.tensor([[True, True]]),
            action_weights=weights,
            one_way_cost_bps=10.0,
            extra_switch_penalty_bps=0.0,
            q_switch_margin_bps=0.0,
        )

        self.assertEqual(action.tolist(), [1])

    def test_exposure_masks_can_forbid_leveraged_and_inverse_actions(self) -> None:
        metadata = build_action_metadata(["CASH", "QQQ", "SOXL", "SOXS"])
        device = torch.device("cpu")
        group_ids, _ = group_ids_for_actions(metadata, device=device)
        mask = apply_exposure_masks(
            torch.ones((1, 4), dtype=torch.bool),
            current_action=torch.tensor([0]),
            action_leverage=action_leverage_tensor(metadata, device=device),
            action_weights=action_weight_tensor(metadata, device=device, max_effective_leverage=1.0),
            action_is_leveraged=action_is_leveraged_tensor(metadata, device=device),
            action_is_inverse=action_is_inverse_tensor(metadata, device=device),
            action_group_ids=group_ids,
            group_counts_today=torch.zeros((1, int(group_ids.max().item()) + 1), dtype=torch.long),
            steps_today=torch.tensor([0]),
            leveraged_bars_today=torch.tensor([0]),
            consecutive_leveraged_bars=torch.tensor([0]),
            constraints=ExposureConstraintConfig(allow_leveraged_actions=False, allow_inverse_actions=False),
        )

        self.assertEqual(mask.tolist(), [[True, True, False, False]])

    def test_same_group_cap_uses_prospective_share(self) -> None:
        metadata = build_action_metadata(["CASH", "SOXL", "SOXS", "QQQ"])
        device = torch.device("cpu")
        group_ids, groups = group_ids_for_actions(metadata, device=device)
        semiconductor_id = groups.index("semiconductor")
        group_counts = torch.zeros((1, len(groups)), dtype=torch.long)
        group_counts[0, semiconductor_id] = 10
        mask = apply_exposure_masks(
            torch.ones((1, 4), dtype=torch.bool),
            current_action=torch.tensor([0]),
            action_leverage=action_leverage_tensor(metadata, device=device),
            action_weights=action_weight_tensor(metadata, device=device, max_effective_leverage=1.0),
            action_is_leveraged=action_is_leveraged_tensor(metadata, device=device),
            action_is_inverse=action_is_inverse_tensor(metadata, device=device),
            action_group_ids=group_ids,
            group_counts_today=group_counts,
            steps_today=torch.tensor([19]),
            leveraged_bars_today=torch.tensor([0]),
            consecutive_leveraged_bars=torch.tensor([0]),
            constraints=ExposureConstraintConfig(max_same_group_share_per_day=0.50, min_group_share_observations=20),
        )

        self.assertTrue(bool(mask[0, 0].item()))
        self.assertFalse(bool(mask[0, 1].item()))
        self.assertFalse(bool(mask[0, 2].item()))
        self.assertTrue(bool(mask[0, 3].item()))

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
