from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from rl_quant.decision_framework import validate_reportable_summary
from rl_quant.action_risk import (
    ExposureConstraintConfig,
    action_weight_tensor,
    build_action_metadata,
)
from rl_quant.hourly_transformer import (
    HOURLY_CONSTRAINT_FEATURE_DIM,
    HourlyDataSplit,
    HourlyEnvConfig,
    VectorizedHourlyAllocationEnv,
    evaluate_hourly_policy,
)
from rl_quant.training.intraday import _apply_action_threshold
from rl_quant.strategy_data import StrategyDataSplit
from rl_quant.strategy_dqn import evaluate_strategy_policy
from rl_quant.trading_constraints import TradingConstraintConfig as BarTradingConstraintConfig
from _support import load_script


class EvaluationTests(unittest.TestCase):
    def _direct_bar_split(self) -> HourlyDataSplit:
        valid = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        valid_mask = torch.zeros(6, dtype=torch.bool)
        valid_mask[valid] = True
        return HourlyDataSplit(
            name="test",
            timestamps=[f"2026-01-02T14:3{minute}:00+00:00" for minute in range(6)],
            next_timestamps=[f"2026-01-02T14:3{minute + 1}:00+00:00" for minute in range(6)],
            feature_names=["x"],
            action_names=["CASH", "QQQ", "SPY"],
            features=torch.zeros((6, 1), dtype=torch.float32),
            action_returns=torch.zeros((6, 3), dtype=torch.float32),
            session_dates=["2026-01-02"] * 6,
            valid_start_indices=valid,
            valid_index_mask=valid_mask,
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=1,
            bar_interval="1m",
        )

    def _leveraged_direct_bar_split(self) -> HourlyDataSplit:
        valid = torch.tensor([1, 2], dtype=torch.long)
        valid_mask = torch.zeros(4, dtype=torch.bool)
        valid_mask[valid] = True
        returns = torch.zeros((4, 2), dtype=torch.float32)
        returns[1, 1] = 0.09
        returns[2, 1] = 0.03
        return HourlyDataSplit(
            name="test",
            timestamps=[f"2026-01-02T14:3{minute}:00+00:00" for minute in range(4)],
            next_timestamps=[f"2026-01-02T14:3{minute + 1}:00+00:00" for minute in range(4)],
            feature_names=["x"],
            action_names=["CASH", "SOXL"],
            features=torch.zeros((4, 1), dtype=torch.float32),
            action_returns=returns,
            session_dates=["2026-01-02"] * 4,
            valid_start_indices=valid,
            valid_index_mask=valid_mask,
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=1,
            bar_interval="1m",
        )

    def test_direct_bar_env_min_hold_allows_current_action_and_cash(self) -> None:
        data = self._direct_bar_split()
        env = VectorizedHourlyAllocationEnv(
            data,
            HourlyEnvConfig(
                lookback=1,
                num_envs=1,
                episode_length=4,
                initial_action=0,
                constraints=BarTradingConstraintConfig(min_hold_bars=15),
            ),
            torch.device("cpu"),
        )
        env.previous_actions[:] = 1
        env.bars_held[:] = 0

        mask = env.action_mask()

        # Min-hold pins the held action (1) but the zero-risk CASH (0) de-risk stays available.
        self.assertEqual(mask.tolist(), [[True, True, False]])

    def test_direct_bar_env_order_leg_cap_blocks_two_leg_rotation(self) -> None:
        data = self._direct_bar_split()
        env = VectorizedHourlyAllocationEnv(
            data,
            HourlyEnvConfig(
                lookback=1,
                num_envs=1,
                episode_length=4,
                initial_action=0,
                constraints=BarTradingConstraintConfig(max_order_legs_per_day=8.0),
            ),
            torch.device("cpu"),
        )
        env.previous_actions[:] = 1
        env.bars_held[:] = 10
        env.order_legs_today[:] = 7.0

        mask = env.action_mask()

        self.assertEqual(mask.tolist(), [[True, True, False]])
        self.assertEqual(tuple(env.constraint_features().shape), (1, HOURLY_CONSTRAINT_FEATURE_DIM))

    def test_direct_bar_env_exposure_mask_caps_leveraged_bars(self) -> None:
        data = self._leveraged_direct_bar_split()
        env = VectorizedHourlyAllocationEnv(
            data,
            HourlyEnvConfig(
                lookback=1,
                num_envs=1,
                episode_length=4,
                initial_action=0,
                exposure_constraints=ExposureConstraintConfig(
                    max_leveraged_bars_per_day=1,
                    max_consecutive_leveraged_bars=None,
                ),
            ),
            torch.device("cpu"),
        )
        env.bars_held[:] = 10
        env.leveraged_bars_today[:] = 1

        mask = env.action_mask()

        self.assertEqual(mask.tolist(), [[True, False]])

    def test_direct_hourly_eval_scales_leveraged_returns(self) -> None:
        class FixedPolicy(nn.Module):
            def forward(
                self,
                state_windows: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return torch.tensor([[0.0, 1.0]], device=state_windows.device).repeat(
                    state_windows.shape[0],
                    1,
                )

        result = evaluate_hourly_policy(
            self._leveraged_direct_bar_split(),
            FixedPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=BarTradingConstraintConfig(one_way_cost_bps=0.0),
            exposure_constraints=ExposureConstraintConfig(
                max_leveraged_bars_per_day=None,
                max_consecutive_leveraged_bars=None,
            ),
            capture_rollout=True,
        )

        self.assertAlmostEqual(result.rollout_records[0]["raw_action_return"], 0.09)
        self.assertAlmostEqual(result.rollout_records[0]["position_weight"], 1.0 / 3.0)
        self.assertAlmostEqual(result.rollout_records[0]["gross_return"], 0.03)

    def test_direct_hourly_eval_respects_inverse_action_block(self) -> None:
        class InversePolicy(nn.Module):
            def forward(
                self,
                state_windows: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor | None = None,
            ) -> torch.Tensor:
                q_values = torch.zeros((state_windows.shape[0], 3), device=state_windows.device)
                q_values[:, 2] = 100.0
                return q_values

        valid = torch.tensor([1, 2], dtype=torch.long)
        valid_mask = torch.zeros(4, dtype=torch.bool)
        valid_mask[valid] = True
        data = HourlyDataSplit(
            name="test",
            timestamps=[f"2026-01-02T14:3{minute}:00+00:00" for minute in range(4)],
            next_timestamps=[f"2026-01-02T14:3{minute + 1}:00+00:00" for minute in range(4)],
            feature_names=["x"],
            action_names=["CASH", "QQQ", "SOXS"],
            features=torch.zeros((4, 1), dtype=torch.float32),
            action_returns=torch.zeros((4, 3), dtype=torch.float32),
            session_dates=["2026-01-02"] * 4,
            valid_start_indices=valid,
            valid_index_mask=valid_mask,
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=1,
            bar_interval="1m",
        )

        result = evaluate_hourly_policy(
            data,
            InversePolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=BarTradingConstraintConfig(one_way_cost_bps=0.0),
            exposure_constraints=ExposureConstraintConfig(
                allow_inverse_actions=False,
                max_leveraged_bars_per_day=None,
                max_consecutive_leveraged_bars=None,
                max_same_group_share_per_day=None,
            ),
            capture_rollout=True,
        )

        self.assertEqual([row["asset"] for row in result.rollout_records], ["CASH", "CASH"])

    def test_direct_hourly_eval_resets_episode_switch_cap_by_episode_length(self) -> None:
        class OppositePolicy(nn.Module):
            def forward(
                self,
                state_windows: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor | None = None,
            ) -> torch.Tensor:
                q_values = torch.zeros((state_windows.shape[0], 3), device=state_windows.device)
                q_values[previous_actions == 0, 1] = 1.0
                q_values[previous_actions == 1, 0] = 1.0
                return q_values

        result = evaluate_hourly_policy(
            self._direct_bar_split(),
            OppositePolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=BarTradingConstraintConfig(max_switches_per_episode=1, one_way_cost_bps=0.0),
            episode_length=2,
        )

        # Policy A: de-risking to CASH overrides an exhausted turnover budget (shared build_action_mask), so
        # OppositePolicy now also takes the 1->CASH de-risk leg each episode that Policy B used to block. The
        # per-episode cap still resets (it is the reset that lets each episode re-take the capped 0->1 ENTER),
        # so the count is 2 enter + 2 de-risk = 4 switches / 4 order legs over the two episodes.
        self.assertEqual(result.total_switches, 4)
        self.assertEqual(result.market_order_legs, 4.0)

    def test_direct_hourly_eval_applies_daily_switch_cap(self) -> None:
        class FixedPolicy(nn.Module):
            def forward(
                self,
                state_windows: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return torch.tensor([[0.0, 1.0, 0.0]], device=state_windows.device).repeat(
                    state_windows.shape[0],
                    1,
                )

        result = evaluate_hourly_policy(
            self._direct_bar_split(),
            FixedPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=BarTradingConstraintConfig(max_switches_per_day=0, one_way_cost_bps=0.0),
            episode_length=4,
        )

        self.assertEqual(result.total_switches, 0)
        self.assertEqual(result.market_order_legs, 0.0)

    def test_direct_train_cli_wires_hard_constraints(self) -> None:
        module = load_script("train_hourly_causal_transformer_rl")

        args = module.parse_args(
            [
                "--switch-cost-bps",
                "2",
                "--min-hold-bars",
                "15",
                "--cooldown-bars",
                "5",
                "--max-switches-per-day",
                "4",
                "--max-switches-per-episode",
                "8",
                "--max-order-legs-per-day",
                "8",
                "--q-switch-margin-bps",
                "5",
                "--extra-switch-penalty-bps",
                "1",
                "--max-effective-leverage",
                "0.75",
                "--allow-leveraged-actions",
                "false",
                "--allow-inverse-actions",
                "false",
                "--max-leveraged-bars-per-day",
                "10",
                "--max-consecutive-leveraged-bars",
                "5",
                "--max-same-group-share-per-day",
                "0.5",
            ]
        )
        constraints = module.build_constraints_from_args(args)
        exposure_constraints = module.build_exposure_constraints_from_args(args)

        self.assertEqual(constraints.one_way_cost_bps, 2.0)
        self.assertEqual(constraints.min_hold_bars, 15)
        self.assertEqual(constraints.cooldown_bars, 5)
        self.assertEqual(constraints.max_switches_per_day, 4)
        self.assertEqual(constraints.max_switches_per_episode, 8)
        self.assertEqual(constraints.max_order_legs_per_day, 8.0)
        self.assertEqual(constraints.q_switch_margin_bps, 5.0)
        self.assertEqual(constraints.extra_switch_penalty_bps, 1.0)
        self.assertEqual(exposure_constraints.max_effective_leverage, 0.75)
        self.assertFalse(exposure_constraints.allow_leveraged_actions)
        self.assertFalse(exposure_constraints.allow_inverse_actions)
        self.assertEqual(exposure_constraints.max_leveraged_bars_per_day, 10)
        self.assertEqual(exposure_constraints.max_consecutive_leveraged_bars, 5)
        self.assertEqual(exposure_constraints.max_same_group_share_per_day, 0.5)

    def test_direct_train_cli_defaults_to_conservative_exposure_caps(self) -> None:
        module = load_script("train_hourly_causal_transformer_rl")
        exposure_constraints = module.build_exposure_constraints_from_args(module.parse_args([]))

        self.assertTrue(exposure_constraints.allow_leveraged_actions)
        self.assertTrue(exposure_constraints.allow_inverse_actions)
        self.assertEqual(exposure_constraints.max_leveraged_bars_per_day, 30)
        self.assertEqual(exposure_constraints.max_consecutive_leveraged_bars, 15)
        self.assertEqual(exposure_constraints.max_same_group_share_per_day, 0.5)

    def test_fixed_rollout_cost_stress_replays_same_actions(self) -> None:
        module = load_script("train_hourly_causal_transformer_rl")

        result = module.fixed_rollout_cost_stress(
            [
                {
                    "action": 1,
                    "previous_action": 0,
                    "market_order_legs": 1.0,
                    "traded_notional": 0.5,
                    "gross_return": 0.01,
                },
                {
                    "action": 1,
                    "previous_action": 1,
                    "market_order_legs": 0.0,
                    "gross_return": 0.01,
                },
            ],
            cost_bps_values=[0.0, 100.0],
            extra_switch_penalty_bps=0.0,
            periods_per_year=252.0,
        )

        self.assertEqual(result["0bps"]["total_switches"], 1)
        self.assertEqual(result["100bps"]["market_order_legs"], 1.0)
        self.assertEqual(result["100bps"]["total_traded_notional"], 0.5)
        self.assertLess(result["100bps"]["total_return"], result["0bps"]["total_return"])

    def test_fixed_rollout_cost_stress_preserves_old_rollout_cost_formula(self) -> None:
        module = load_script("train_hourly_causal_transformer_rl")

        result = module.fixed_rollout_cost_stress(
            [
                {
                    "action": 2,
                    "previous_action": 1,
                    "market_order_legs": 2.0,
                    "gross_return": 0.0,
                },
            ],
            cost_bps_values=[10.0],
            extra_switch_penalty_bps=5.0,
            periods_per_year=252.0,
        )

        self.assertEqual(result["10bps"]["total_reward_bps"], -25.0)

    def test_fixed_rollout_cost_stress_requires_gross_return(self) -> None:
        module = load_script("train_hourly_causal_transformer_rl")

        with self.assertRaisesRegex(ValueError, "requires gross_return"):
            module.fixed_rollout_cost_stress(
                [
                    {
                        "action": 1,
                        "previous_action": 0,
                        "market_order_legs": 1.0,
                        "bar_return": 0.01,
                    }
                ],
                cost_bps_values=[1.0],
                extra_switch_penalty_bps=0.0,
                periods_per_year=252.0,
            )

    def test_random_baselines_are_generated_from_rollout_shape(self) -> None:
        module = load_script("train_hourly_causal_transformer_rl")
        split = self._direct_bar_split()
        weights = action_weight_tensor(build_action_metadata(split.action_names), device="cpu", max_effective_leverage=1.0)
        rollout = [
            {"action": 1, "previous_action": 0},
            {"action": 1, "previous_action": 1},
            {"action": 2, "previous_action": 1},
            {"action": 2, "previous_action": 2},
        ]

        same_turnover = module.random_same_turnover_baseline(
            split,
            rollout,
            seed=7,
            n_paths=8,
            initial_action=0,
            cash_index=0,
            action_weights=weights,
            switch_cost_bps=1.0,
            extra_switch_penalty_bps=0.0,
        )
        same_distribution = module.random_same_action_distribution_baseline(
            split,
            rollout,
            seed=8,
            n_paths=8,
            initial_action=0,
            cash_index=0,
            action_weights=weights,
            switch_cost_bps=1.0,
            extra_switch_penalty_bps=0.0,
        )

        self.assertEqual(same_turnover["paths"], 8)
        self.assertEqual(same_turnover["target_switches"], 2)
        self.assertEqual(same_distribution["paths"], 8)
        self.assertIn("total_return", same_turnover)
        self.assertIn("total_return", same_distribution)

    def test_random_baseline_applies_full_exposure_mask(self) -> None:
        module = load_script("train_hourly_causal_transformer_rl")
        split = self._direct_bar_split()
        split = replace(
            split,
            action_names=["CASH", "QQQ", "SOXS"],
            action_returns=torch.tensor(
                [
                    [0.0, 0.0, 0.50],
                    [0.0, 0.0, 0.50],
                    [0.0, 0.0, 0.50],
                    [0.0, 0.0, 0.50],
                    [0.0, 0.0, 0.50],
                    [0.0, 0.0, 0.50],
                ],
                dtype=torch.float32,
            ),
        )
        metadata = build_action_metadata(split.action_names)
        weights = action_weight_tensor(metadata, device="cpu", max_effective_leverage=1.0)
        rollout = [{"action": 2, "previous_action": 0} for _ in split.valid_start_indices.tolist()]

        baseline = module.random_same_action_distribution_baseline(
            split,
            rollout,
            seed=9,
            n_paths=4,
            initial_action=0,
            cash_index=0,
            action_weights=weights,
            switch_cost_bps=0.0,
            extra_switch_penalty_bps=0.0,
            constraints=BarTradingConstraintConfig(one_way_cost_bps=0.0),
            exposure_constraints=ExposureConstraintConfig(
                allow_inverse_actions=False,
                max_leveraged_bars_per_day=None,
                max_consecutive_leveraged_bars=None,
                max_same_group_share_per_day=None,
            ),
            action_meta=metadata,
            episode_length=4,
        )

        self.assertEqual(baseline["total_return"], 0.0)

    def test_direct_hourly_eval_captures_valid_decision_log_payload(self) -> None:
        class FixedPolicy(nn.Module):
            def forward(
                self,
                state_windows: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return torch.tensor([[0.0, 1.0, -1.0]], device=state_windows.device).repeat(
                    state_windows.shape[0],
                    1,
                )

        module = load_script("train_hourly_causal_transformer_rl")
        result = evaluate_hourly_policy(
            self._direct_bar_split(),
            FixedPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=BarTradingConstraintConfig(one_way_cost_bps=0.0),
            capture_rollout=True,
        )
        first = result.rollout_records[0]

        self.assertIn("q_values", first)
        self.assertIn("candidates", first)
        self.assertIn("risk_checks", first)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision_logs.jsonl"
            module.write_decision_logs(path, [first])
            payload = json.loads(path.read_text().splitlines()[0])

        self.assertEqual(payload["selected_action"], first["selected_action"])
        self.assertIn(first["selected_action"], payload["q_values"])

    def test_reportability_artifacts_cover_required_summary_sections(self) -> None:
        module = load_script("train_hourly_causal_transformer_rl")
        split = self._direct_bar_split()
        metadata = [item.to_dict() for item in build_action_metadata(split.action_names)]
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.pt"
            (dataset_path.parent / "dataset_manifest.json").write_text('{"dataset_id": "demo"}\n')
            args = module.parse_args(["--dataset", str(dataset_path), "--run-name", "unit"])
            constraints = BarTradingConstraintConfig(one_way_cost_bps=0.0)
            exposure = ExposureConstraintConfig(max_same_group_share_per_day=None)
            artifacts = module.build_reportability_artifacts(
                args=args,
                run_name="unit",
                train_split=split,
                val_split=split,
                test_split=split,
                action_metadata=metadata,
                action_metadata_hash="actions",
                action_risk_config_hash="risk",
                constraints=constraints,
                exposure_constraints=exposure,
                baselines={
                    "CASH": {"test": {"total_return": 0.0}},
                    "RandomSameTurnover": {"test": {"total_return": 0.0}},
                },
                fixed_cost_stress={},
                adaptive_cost_stress={},
                test_metrics={"total_return": 0.0, "max_drawdown": 0.0, "annualized_sharpe": None},
                artifacts={},
                model_version=3,
                constraint_feature_names=[],
            )
        summary = {
            **artifacts,
            "test_metrics": {"total_return": 0.0},
            "baselines": {
                "CASH": {"test": {"total_return": 0.0}},
                "RandomSameTurnover": {"test": {"total_return": 0.0}},
            },
            "cost_stress": {"fixed_rollout": {}, "adaptive": {}},
            "action_concentration": {"max_risky_group_share": 0.0, "leveraged_action_share": 0.0},
            "return_diagnostics": {},
        }

        self.assertEqual(validate_reportable_summary(summary), [])

    def test_evaluation_uses_valid_indices_and_resets_at_gaps(self) -> None:
        class FixedPolicy(nn.Module):
            def forward(
                self,
                state_windows: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor | None = None,
            ) -> torch.Tensor:
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

    def test_direct_hourly_eval_respects_action_valid_mask(self) -> None:
        class UnavailableActionPolicy(nn.Module):
            def forward(
                self,
                state_windows: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor | None = None,
            ) -> torch.Tensor:
                q_values = torch.zeros((state_windows.shape[0], 3), device=state_windows.device)
                q_values[:, 1] = 100.0
                return q_values

        data = self._direct_bar_split()
        action_valid_mask = torch.ones_like(data.action_returns, dtype=torch.bool)
        action_valid_mask[:, 1] = False
        data = replace(data, action_valid_mask=action_valid_mask)

        result = evaluate_hourly_policy(
            data,
            UnavailableActionPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=BarTradingConstraintConfig(one_way_cost_bps=0.0),
            episode_length=4,
        )

        self.assertEqual(result.total_switches, 0)
        self.assertEqual(result.market_order_legs, 0.0)

    def test_strategy_evaluation_uses_valid_indices_and_resets_at_gaps(self) -> None:
        class FixedPolicy(nn.Module):
            def forward(self, state_windows: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 1.0]], device=state_windows.device).repeat(state_windows.shape[0], 1)

        valid = torch.tensor([1, 2, 4], dtype=torch.long)
        valid_mask = torch.zeros(6, dtype=torch.bool)
        valid_mask[valid] = True
        data = StrategyDataSplit(
            name="test",
            dates=[f"2026-01-0{i}" for i in range(1, 7)],
            feature_names=["x"],
            action_names=["CASH", "BH_QQQ"],
            features=torch.zeros((6, 1), dtype=torch.float32),
            action_returns=torch.zeros((6, 2), dtype=torch.float32),
            valid_start_indices=valid,
            valid_index_mask=valid_mask,
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=1,
        )

        result = evaluate_strategy_policy(
            data,
            FixedPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            capture_rollout=True,
        )

        self.assertEqual([row["date"] for row in result.rollout_records], ["2026-01-03", "2026-01-04", "2026-01-06"])
        self.assertEqual([row["segment_reset"] for row in result.rollout_records], [1, 0, 1])

    def test_intraday_threshold_compares_against_current_position(self) -> None:
        q_values = torch.tensor([[0.0, 0.9, 1.0]])
        current_long = torch.tensor([1])

        action = _apply_action_threshold(q_values, current_long, threshold=0.2)

        self.assertEqual(action.tolist(), [2])
