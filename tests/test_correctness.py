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

from rl_quant.decision_framework import (  # noqa: E402
    ActionEligibility,
    DataQualityReport,
    DecisionFrameworkError,
    DecisionDataset,
    FeatureManifest,
    DecisionLog,
    DecisionSnapshot,
    ReadinessConfig,
    apply_data_quality_gate,
    action_eligibilities_to_mask,
    assert_available_at,
    decision_readiness_score,
    filter_point_in_time_rows,
    readiness_band,
    validate_reportable_summary,
)
from rl_quant.action_risk import (  # noqa: E402
    ExposureConstraintConfig,
    action_is_inverse_tensor,
    action_is_leveraged_tensor,
    action_leverage_tensor,
    action_concentration,
    action_weight_tensor,
    apply_exposure_masks,
    build_action_metadata,
    group_ids_for_actions,
    reportability_flags,
    stable_action_metadata_hash,
    stable_action_risk_config_hash,
    trade_notional,
)
from rl_quant.hourly_transformer import (  # noqa: E402
    CausalTransformerQNetwork,
    HOURLY_CONSTRAINT_FEATURE_DIM,
    HourlyDataSplit,
    HourlyEnvConfig,
    VectorizedHourlyAllocationEnv,
    assert_matching_hourly_schema,
    evaluate_hourly_policy,
)
from rl_quant.intraday_dqn import _apply_action_threshold  # noqa: E402
from rl_quant.minute_to_hour_transformer import (  # noqa: E402
    HourFromMinuteDataSplit,
    MinuteToHourCausalTransformerQNetwork,
    TradingConstraintConfig,
    apply_leg_aware_hysteresis,
    build_action_mask,
    evaluate_minute_to_hour_policy,
    load_minute_to_hour_warm_start,
    make_constraint_features,
    sample_valid_actions,
    trade_legs,
)
from rl_quant.research_protocol import (  # noqa: E402
    BaselineResult,
    DatasetManifest,
    EvaluationProtocol,
    FitWindow,
    ModelManifest,
    ResearchProtocolError,
    StressTestResult,
    default_benchmark_registry,
    hash_string_sequence,
)
from rl_quant.strategy_data import (  # noqa: E402
    StrategyDataSplit,
    assert_matching_strategy_schema,
)
from rl_quant.strategy_dqn import evaluate_strategy_policy  # noqa: E402
from rl_quant.trading_constraints import (  # noqa: E402
    CONSTRAINT_FEATURE_DIM,
    CONSTRAINT_FEATURE_NAMES,
    TradingConstraintConfig as BarTradingConstraintConfig,
    apply_notional_aware_hysteresis,
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

    def test_action_return_uses_decision_close_to_next_decision_close(self) -> None:
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
                for timestamp, close in [
                    ("2026-01-02T14:30:00+00:00", "100"),
                    ("2026-01-02T14:31:00+00:00", "110"),
                    ("2026-01-02T14:32:00+00:00", "121"),
                ]:
                    writer.writerow(
                        {
                            "DatetimeUTC": timestamp,
                            "DatetimeExchange": timestamp,
                            "Open": close,
                            "High": close,
                            "Low": close,
                            "Close": close,
                            "Adj Close": close,
                            "Volume": "10",
                        }
                    )

            rows = module.load_symbol_features(
                path,
                start="2026-01-02T00:00:00+00:00",
                end_exclusive="2026-01-03T00:00:00+00:00",
            )

        current = rows["2026-01-02T14:30:00+00:00"]
        next_decision = rows["2026-01-02T14:32:00+00:00"]
        self.assertAlmostEqual(next_decision.bar_return, 0.10)
        self.assertAlmostEqual(module.clipped_simple_return(current.close, next_decision.close), 0.21)


class MinuteToHourTests(unittest.TestCase):
    @staticmethod
    def _small_minute_to_hour_split(action_names: list[str] | None = None) -> HourFromMinuteDataSplit:
        action_names = action_names or ["CASH", "QQQ"]
        return HourFromMinuteDataSplit(
            name="train",
            decision_timestamps=["2026-01-02T14:30:00+00:00", "2026-01-02T15:30:00+00:00"],
            next_timestamps=["2026-01-02T15:30:00+00:00", "2026-01-02T16:30:00+00:00"],
            minute_feature_names=["m"],
            hour_feature_names=["h"],
            action_names=action_names,
            minute_features=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((2, 1, 1), dtype=torch.float32),
            action_returns=torch.zeros((2, len(action_names)), dtype=torch.float32),
            valid_start_indices=torch.tensor([0], dtype=torch.long),
            valid_index_mask=torch.tensor([True, False]),
            minute_feature_mean=torch.zeros(1),
            minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1),
            hour_feature_std=torch.ones(1),
            hours_lookback=1,
            minutes_per_hour=1,
        )

    @staticmethod
    def _small_minute_to_hour_model() -> MinuteToHourCausalTransformerQNetwork:
        return MinuteToHourCausalTransformerQNetwork(
            minute_feature_dim=1,
            hour_feature_dim=1,
            action_count=2,
            hours_lookback=1,
            minutes_per_hour=1,
            d_model=16,
            n_heads=4,
            minute_layers=1,
            hour_layers=1,
            feedforward_dim=32,
            action_embedding_dim=4,
        )

    def test_hourly_context_uses_only_past_minutes(self) -> None:
        decision_timestamp = "2026-06-10T15:30:00+00:00"
        next_timestamp = "2026-06-10T16:30:00+00:00"
        minute_timestamps = [
            ["2026-06-10T14:31:00+00:00", "2026-06-10T14:32:00+00:00"],
            ["2026-06-10T15:29:00+00:00", "2026-06-10T15:30:00+00:00"],
        ]

        self.assertLess(decision_timestamp, next_timestamp)
        self.assertLessEqual(max(timestamp for hour in minute_timestamps for timestamp in hour), decision_timestamp)

    def test_next_hour_reward_uses_decision_close_to_next_hour_close(self) -> None:
        module = load_script("build_hourly_transformer_dataset")

        self.assertAlmostEqual(module.clipped_simple_return(100.0, 105.0), 0.05)

    def test_minute_to_hour_periods_per_year_uses_actual_schedule(self) -> None:
        module = load_script("build_hourly_from_minute_context_dataset")

        periods = module.infer_periods_per_year(
            [
                "2026-06-10T15:30:00+00:00",
                "2026-06-10T16:30:00+00:00",
                "2026-06-11T15:30:00+00:00",
                "2026-06-11T16:30:00+00:00",
                "2026-06-11T17:30:00+00:00",
            ]
        )

        self.assertEqual(periods, 630.0)

    def test_minute_to_hour_scripts_default_to_shared_data_root_when_available(self) -> None:
        builder = load_script("build_hourly_from_minute_context_dataset")
        trainer = load_script("train_hourly_from_minute_context_rl")
        expected_data_root = ROOT.parent / "data" if (ROOT.parent / "data").exists() else ROOT / "data"
        expected_derived_root = ROOT.parent / "derived" if (ROOT.parent / "derived").exists() else ROOT / "derived"

        build_args = builder.parse_args([])
        train_args = trainer.parse_args([])

        self.assertEqual(build_args.output_dir, expected_data_root / "rl_hour_from_minute" / "top_volume_1m_recent")
        self.assertEqual(build_args.decision_stride_minutes, builder.DEFAULT_DECISION_GRID_MINUTES)
        self.assertEqual(build_args.minutes_per_hour, builder.DEFAULT_CONTEXT_MINUTES_PER_GRID)
        self.assertEqual(
            build_args.stock_minute_dir,
            expected_data_root
            / "minute_ohlcv"
            / "top_us_volume_stocks_nasdaq_1000_2026-06-14_1m_2026-05-25_2026-06-15",
        )
        self.assertEqual(
            build_args.stock_universe,
            expected_derived_root / "universes" / "top_us_volume_stocks_nasdaq_1000_2026-06-14.csv",
        )
        self.assertEqual(
            train_args.dataset,
            expected_data_root / "rl_hour_from_minute" / "top_volume_1m_recent" / "hour_from_minute_dataset.pt",
        )
        self.assertEqual(train_args.output_dir, expected_data_root / "rl_hour_from_minute_runs")

    def test_minute_to_hour_builder_rejects_non_hourly_default_grid(self) -> None:
        builder = load_script("build_hourly_from_minute_context_dataset")
        args = builder.parse_args(["--decision-stride-minutes", "30"])

        with self.assertRaisesRegex(ValueError, "hourly decision grid"):
            builder.validate_hourly_grid_args(args)

    def test_min_hold_action_mask_allows_only_current_action(self) -> None:
        mask = build_action_mask(
            current_action=torch.tensor([3]),
            bars_held=torch.tensor([0]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([0]),
            max_switches_per_day=2,
            min_hold_bars=2,
            action_count=10,
        )

        self.assertEqual(int(mask.sum().item()), 1)
        self.assertTrue(bool(mask[0, 3].item()))

    def test_daily_switch_cap_masks_new_positions(self) -> None:
        mask = build_action_mask(
            current_action=torch.tensor([4]),
            bars_held=torch.tensor([5]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([2]),
            max_switches_per_day=2,
            min_hold_bars=2,
            action_count=10,
        )

        self.assertEqual(int(mask.sum().item()), 1)
        self.assertTrue(bool(mask[0, 4].item()))

    def test_episode_switch_cap_masks_new_positions(self) -> None:
        mask = build_action_mask(
            current_action=torch.tensor([2]),
            bars_held=torch.tensor([5]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([0]),
            max_switches_per_day=5,
            min_hold_bars=1,
            action_count=6,
            switches_episode=torch.tensor([3]),
            max_switches_per_episode=3,
        )

        self.assertEqual(int(mask.sum().item()), 1)
        self.assertTrue(bool(mask[0, 2].item()))

    def test_order_leg_cap_blocks_two_leg_rotation(self) -> None:
        mask = build_action_mask(
            current_action=torch.tensor([2]),
            bars_held=torch.tensor([5]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([0]),
            max_switches_per_day=5,
            min_hold_bars=1,
            action_count=6,
            order_legs_episode=torch.tensor([0.0]),
            max_order_legs_per_episode=1.0,
        )

        self.assertTrue(bool(mask[0, 0].item()))
        self.assertTrue(bool(mask[0, 2].item()))
        self.assertFalse(bool(mask[0, 5].item()))

    def test_etf_to_etf_switch_counts_two_legs(self) -> None:
        self.assertEqual(float(trade_legs(torch.tensor([2]), torch.tensor([5]))[0].item()), 2.0)

    def test_cash_to_etf_counts_one_leg(self) -> None:
        self.assertEqual(float(trade_legs(torch.tensor([0]), torch.tensor([5]))[0].item()), 1.0)

    def test_constraint_feature_scaling_uses_episode_cap(self) -> None:
        constraints = TradingConstraintConfig(max_switches_per_day=4, max_switches_per_episode=3)
        train_like = make_constraint_features(
            bars_held=torch.tensor([1]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([1]),
            switches_episode=torch.tensor([3]),
            constraints=constraints,
            episode_length=32,
        )
        eval_like = make_constraint_features(
            bars_held=torch.tensor([1]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([1]),
            switches_episode=torch.tensor([3]),
            constraints=constraints,
            episode_length=32,
        )

        self.assertTrue(bool(torch.allclose(train_like, eval_like)))
        self.assertAlmostEqual(float(train_like[0, 3].item()), 1.0)

    def test_constraint_feature_zero_cap_does_not_fallback_to_episode_length(self) -> None:
        constraints = TradingConstraintConfig(max_switches_per_day=0, max_order_legs_per_day=0.0)

        features = make_constraint_features(
            bars_held=torch.tensor([1]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([1]),
            switches_episode=torch.tensor([0]),
            order_legs_today=torch.tensor([1.0]),
            constraints=constraints,
            episode_length=32,
        )

        self.assertGreaterEqual(float(features[0, 2].item()), 1.0)
        self.assertGreaterEqual(float(features[0, 4].item()), 1.0)

    def test_constraint_feature_schema_names_match_dimension(self) -> None:
        self.assertEqual(len(CONSTRAINT_FEATURE_NAMES), CONSTRAINT_FEATURE_DIM)
        self.assertEqual(CONSTRAINT_FEATURE_NAMES[0], "bars_held_over_min_hold")

    def test_leg_aware_hysteresis_uses_two_leg_etf_rotation_cost(self) -> None:
        q_values = torch.tensor([[0.0, 10.0, 15.5]])
        action = apply_leg_aware_hysteresis(
            q_values,
            torch.tensor([1]),
            torch.tensor([[True, True, True]]),
            one_way_cost_bps=1.0,
            extra_switch_penalty_bps=1.0,
            q_switch_margin_bps=3.0,
            reward_scale=10_000.0,
        )

        self.assertEqual(action.tolist(), [1])

    def test_leg_aware_hysteresis_uses_one_leg_cash_entry_cost(self) -> None:
        q_values = torch.tensor([[10.0, 15.5]])
        action = apply_leg_aware_hysteresis(
            q_values,
            torch.tensor([0]),
            torch.tensor([[True, True]]),
            one_way_cost_bps=1.0,
            extra_switch_penalty_bps=1.0,
            q_switch_margin_bps=3.0,
            reward_scale=10_000.0,
        )

        self.assertEqual(action.tolist(), [1])

    def test_leg_aware_hysteresis_forces_exit_when_current_action_masked(self) -> None:
        action = apply_leg_aware_hysteresis(
            torch.tensor([[0.0, 100.0]]),
            torch.tensor([1]),
            torch.tensor([[True, False]]),
            one_way_cost_bps=0.0,
            extra_switch_penalty_bps=0.0,
            q_switch_margin_bps=0.0,
        )

        self.assertEqual(action.tolist(), [0])

    def test_sample_valid_actions_uses_valid_set_only(self) -> None:
        actions = sample_valid_actions(torch.tensor([[False, True, False], [True, False, False]]))

        self.assertEqual(actions.tolist(), [1, 0])

    def test_minute_to_hour_model_forward_shape(self) -> None:
        model = MinuteToHourCausalTransformerQNetwork(
            minute_feature_dim=3,
            hour_feature_dim=2,
            action_count=4,
            hours_lookback=2,
            minutes_per_hour=3,
            d_model=16,
            n_heads=4,
            minute_layers=1,
            hour_layers=1,
            feedforward_dim=32,
            action_embedding_dim=4,
        )
        q_values = model(
            torch.zeros((2, 2, 3, 3), dtype=torch.float32),
            torch.tensor([[[True, True, False], [True, True, True]], [[False, False, False], [True, False, False]]]),
            torch.zeros((2, 2, 2), dtype=torch.float32),
            torch.zeros(2, dtype=torch.long),
            torch.zeros((2, CONSTRAINT_FEATURE_DIM), dtype=torch.float32),
        )

        self.assertEqual(tuple(q_values.shape), (2, 4))
        self.assertTrue(bool(torch.isfinite(q_values).all().item()))

    def test_minute_to_hour_warm_start_loads_matching_checkpoint(self) -> None:
        train_data = self._small_minute_to_hour_split()
        source = self._small_minute_to_hour_model()
        target = self._small_minute_to_hour_model()
        for parameter in source.parameters():
            parameter.data.fill_(0.125)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "minute_feature_names": train_data.minute_feature_names,
                    "hour_feature_names": train_data.hour_feature_names,
                    "action_names": train_data.action_names,
                    "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
                    "model_version": "test",
                    "uses_constraint_features": True,
                },
                path,
            )

            info = load_minute_to_hour_warm_start(target, checkpoint_path=path, train_data=train_data)

        self.assertTrue(info["loaded"])
        for name, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, source.state_dict()[name]))

    def test_minute_to_hour_warm_start_rejects_schema_mismatch(self) -> None:
        train_data = self._small_minute_to_hour_split()
        source = self._small_minute_to_hour_model()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "minute_feature_names": train_data.minute_feature_names,
                    "hour_feature_names": train_data.hour_feature_names,
                    "action_names": ["CASH", "SPY"],
                    "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "action_names"):
                load_minute_to_hour_warm_start(self._small_minute_to_hour_model(), checkpoint_path=path, train_data=train_data)

    def test_direct_bar_model_accepts_constraint_features(self) -> None:
        model = CausalTransformerQNetwork(
            feature_dim=3,
            lookback=4,
            action_count=3,
            d_model=16,
            n_heads=4,
            n_layers=1,
            feedforward_dim=32,
            action_embedding_dim=4,
        )
        q_values = model(
            torch.zeros((2, 4, 3), dtype=torch.float32),
            torch.zeros(2, dtype=torch.long),
            torch.zeros((2, CONSTRAINT_FEATURE_DIM), dtype=torch.float32),
        )

        self.assertEqual(tuple(q_values.shape), (2, 3))
        self.assertEqual(model.constraint_feature_dim, CONSTRAINT_FEATURE_DIM)

    def test_direct_bar_model_requires_constraint_features_by_default(self) -> None:
        model = CausalTransformerQNetwork(
            feature_dim=3,
            lookback=4,
            action_count=3,
            d_model=16,
            n_heads=4,
            n_layers=1,
            feedforward_dim=32,
            action_embedding_dim=4,
        )

        with self.assertRaisesRegex(ValueError, "constraint_features are required"):
            model(torch.zeros((2, 4, 3), dtype=torch.float32), torch.zeros(2, dtype=torch.long))

    def test_direct_bar_model_legacy_constraint_feature_opt_out(self) -> None:
        model = CausalTransformerQNetwork(
            feature_dim=3,
            lookback=4,
            action_count=3,
            d_model=16,
            n_heads=4,
            n_layers=1,
            feedforward_dim=32,
            action_embedding_dim=4,
            require_constraint_features=False,
        )

        q_values = model(torch.zeros((2, 4, 3), dtype=torch.float32), torch.zeros(2, dtype=torch.long))

        self.assertEqual(tuple(q_values.shape), (2, 3))

    def test_minute_timestamp_grid_future_context_is_rejected(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_load_payload"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            torch.save(
                {
                    "decision_timestamps": ["2026-06-10T15:30:00+00:00"],
                    "next_timestamps": ["2026-06-10T16:30:00+00:00"],
                    "minute_timestamp_grid": [[["2026-06-10T15:31:00+00:00"]]],
                    "minute_feature_names": ["x"],
                    "hour_feature_names": ["h"],
                    "action_names": ["CASH", "QQQ"],
                    "minute_features": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                    "minute_mask": torch.tensor([[[True]]]),
                    "hour_features": torch.zeros((1, 1, 1), dtype=torch.float32),
                    "action_returns": torch.zeros((1, 2), dtype=torch.float32),
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "Minute context leakage"):
                module._load_payload(path)

    def test_minute_to_hour_payload_rejects_non_hourly_reward_grid(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_load_payload"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            torch.save(
                {
                    "decision_timestamps": ["2026-06-10T15:30:00+00:00"],
                    "next_timestamps": ["2026-06-10T16:00:00+00:00"],
                    "minute_timestamp_grid": [[["2026-06-10T15:30:00+00:00"]]],
                    "minute_feature_names": ["x"],
                    "hour_feature_names": ["h"],
                    "action_names": ["CASH", "QQQ"],
                    "minute_features": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                    "minute_mask": torch.tensor([[[True]]]),
                    "hour_features": torch.zeros((1, 1, 1), dtype=torch.float32),
                    "action_returns": torch.zeros((1, 2), dtype=torch.float32),
                    "decision_stride_minutes": 30,
                    "source_bar_interval": "1m",
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "hourly decision grid"):
                module._load_payload(path)

    def test_train_cli_wires_episode_and_leg_caps(self) -> None:
        module = load_script("train_hourly_from_minute_context_rl")

        args = module.parse_args(
            [
                "--max-switches-per-episode",
                "3",
                "--max-order-legs-per-day",
                "4",
                "--max-order-legs-per-episode",
                "5",
                "--warm-start-model",
                "/tmp/model.pt",
            ]
        )
        constraints = module.build_constraints_from_args(args)

        self.assertEqual(constraints.max_switches_per_episode, 3)
        self.assertEqual(constraints.max_order_legs_per_day, 4)
        self.assertEqual(constraints.max_order_legs_per_episode, 5)
        self.assertEqual(args.warm_start_model, Path("/tmp/model.pt"))

    def test_minute_to_hour_default_constraints_remain_conservative(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["default_minute_to_hour_constraints"])

        constraints = module.default_minute_to_hour_constraints()

        self.assertEqual(constraints.max_switches_per_day, 2)
        self.assertEqual(constraints.q_switch_margin_bps, 3.0)

    def test_minute_to_hour_eval_respects_action_valid_mask(self) -> None:
        class UnavailableActionPolicy(nn.Module):
            def forward(
                self,
                minute_features: torch.Tensor,
                minute_mask: torch.Tensor,
                hour_features: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor,
            ) -> torch.Tensor:
                q_values = torch.zeros((minute_features.shape[0], 2), device=minute_features.device)
                q_values[:, 1] = 100.0
                return q_values

        data = HourFromMinuteDataSplit(
            name="test",
            decision_timestamps=[
                "2026-01-02T14:30:00+00:00",
                "2026-01-02T14:31:00+00:00",
                "2026-01-02T14:32:00+00:00",
            ],
            next_timestamps=[
                "2026-01-02T14:31:00+00:00",
                "2026-01-02T14:32:00+00:00",
                "2026-01-02T14:33:00+00:00",
            ],
            minute_feature_names=["m"],
            hour_feature_names=["h"],
            action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((3, 1, 1, 1), dtype=torch.float32),
            minute_mask=torch.ones((3, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((3, 1, 1), dtype=torch.float32),
            action_returns=torch.zeros((3, 2), dtype=torch.float32),
            valid_start_indices=torch.tensor([1], dtype=torch.long),
            valid_index_mask=torch.tensor([False, True, False]),
            minute_feature_mean=torch.zeros(1),
            minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1),
            hour_feature_std=torch.ones(1),
            hours_lookback=1,
            minutes_per_hour=1,
            action_valid_mask=torch.tensor([[True, True], [True, False], [True, True]]),
        )

        result = evaluate_minute_to_hour_policy(
            data,
            UnavailableActionPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=TradingConstraintConfig(one_way_cost_bps=0.0),
            capture_rollout=True,
        )

        self.assertEqual(result.allocation_switches, 0)
        self.assertEqual([row["asset"] for row in result.rollout_records], ["CASH"])


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

    def test_direct_bar_env_min_hold_masks_to_current_action(self) -> None:
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

        self.assertEqual(mask.tolist(), [[False, True, False]])

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

        self.assertEqual(result.total_switches, 2)
        self.assertEqual(result.market_order_legs, 2.0)

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


class DecisionFrameworkTests(unittest.TestCase):
    def _eligibility(
        self,
        *,
        symbol_id: str = "CASH",
        decision_ts: str = "2026-01-02T14:30:00+00:00",
        available_ts: str = "2026-01-02T14:29:00+00:00",
        tradable: bool = True,
        reason_if_excluded: str | None = None,
        leverage_factor: float = 0.0,
        inverse: bool = False,
        risk_bucket: str = "cash",
    ) -> ActionEligibility:
        return ActionEligibility(
            symbol_id=symbol_id,
            decision_ts=decision_ts,
            available_ts=available_ts,
            source="synthetic",
            source_payload_hash="abc",
            calculation_window="20d",
            tradable=tradable,
            reason_if_excluded=reason_if_excluded,
            avg_dollar_volume_20d=0.0 if symbol_id == "CASH" else 1_000_000.0,
            median_spread_bps_20d=0.0 if symbol_id == "CASH" else 25.0,
            missing_bar_rate_5d=0.0,
            leverage_factor=leverage_factor,
            inverse=inverse,
            risk_bucket=risk_bucket,
        )

    def test_point_in_time_rows_reject_future_available_timestamp(self) -> None:
        rows = [
            {"available_timestamp": "2026-01-02T14:29:00+00:00", "value": 1},
            {"available_timestamp": "2026-01-02T14:31:00+00:00", "value": 2},
        ]

        kept = filter_point_in_time_rows(rows, decision_ts="2026-01-02T14:30:00+00:00")

        self.assertEqual([row["value"] for row in kept], [1])
        with self.assertRaises(DecisionFrameworkError):
            assert_available_at(
                decision_ts="2026-01-02T14:30:00+00:00",
                available_ts="2026-01-02T14:31:00+00:00",
                name="macro_row",
            )

    def test_action_eligibility_and_quality_gate_force_cash(self) -> None:
        eligibilities = [
            self._eligibility(
                symbol_id="CASH",
                tradable=True,
                reason_if_excluded=None,
                leverage_factor=0.0,
                inverse=False,
                risk_bucket="cash",
            ),
            self._eligibility(
                symbol_id="SOXL",
                tradable=False,
                reason_if_excluded="spread_too_wide",
                leverage_factor=3.0,
                inverse=False,
                risk_bucket="leveraged_etf",
            ),
        ]

        mask, reasons = action_eligibilities_to_mask(eligibilities)
        gated = apply_data_quality_gate(mask, data_quality_score=0.50)
        quality = DataQualityReport(
            report_id="q1",
            created_at="2026-01-02T14:30:00+00:00",
            row_count=10,
            quality_score=0.50,
            missing_bar_rate=0.10,
        )

        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(reasons, {"SOXL": "spread_too_wide"})
        self.assertEqual(gated.tolist(), [True, False])
        self.assertTrue(quality.should_force_cash())

    def test_action_eligibility_rejects_future_available_timestamp(self) -> None:
        eligibility = self._eligibility(available_ts="2026-01-02T14:31:00+00:00")

        with self.assertRaisesRegex(DecisionFrameworkError, "not point-in-time"):
            eligibility.validate()

    def test_action_eligibility_rejects_negative_leverage(self) -> None:
        eligibility = self._eligibility(symbol_id="SOXL", leverage_factor=-1.0, risk_bucket="leveraged_etf")

        with self.assertRaisesRegex(DecisionFrameworkError, "leverage_factor"):
            eligibility.validate()

    def test_action_eligibilities_to_mask_validates_items_and_cash_index(self) -> None:
        eligibilities = [
            self._eligibility(symbol_id="QQQ", leverage_factor=1.0, risk_bucket="broad_tech"),
            self._eligibility(symbol_id="CASH"),
        ]

        with self.assertRaisesRegex(DecisionFrameworkError, "cash_index must point to CASH"):
            action_eligibilities_to_mask(eligibilities, cash_index=0)

    def test_feature_manifest_rejects_fit_window_leaking_into_feature_asof(self) -> None:
        manifest = FeatureManifest(
            feature_set_id="features",
            input_dataset_ids=["dataset"],
            feature_names=["x"],
            feature_available_ts_rule="close_plus_1m",
            fit_start="2026-01-01T00:00:00+00:00",
            fit_end="2026-01-03T00:00:00+00:00",
            feature_asof="2026-01-03T00:00:00+00:00",
            normalizer_hash="norm",
            code_version="abc",
        )

        with self.assertRaisesRegex(DecisionFrameworkError, "feature_asof"):
            manifest.validate()

    def test_readiness_config_hash_validates_weights(self) -> None:
        config = ReadinessConfig(
            weights={"data_quality": 0.5, "liquidity": 0.5},
            thresholds={"normal": 0.9},
            min_data_quality=0.9,
            min_liquidity_score=0.8,
            min_constraint_budget=0.5,
            version="v1",
        )

        self.assertEqual(len(config.content_hash()), 64)

    def test_readiness_score_and_decision_log_validate(self) -> None:
        score = decision_readiness_score(
            data_quality=0.90,
            model_confidence=0.80,
            ensemble_agreement=0.75,
            regime_knownness=0.80,
            cost_score=0.90,
            liquidity_score=0.85,
            constraint_budget=1.0,
            recent_paper_performance=0.70,
        )
        snapshot = DecisionSnapshot(
            decision_ts="2026-01-02T14:30:00+00:00",
            instrument_universe_hash="abc",
            market_state=torch.zeros(4),
            portfolio_state=torch.zeros(2),
            action_valid_mask=torch.tensor([True, False]),
            action_cost_estimate_bps=torch.tensor([0.0, 3.0]),
            action_risk_features=torch.zeros((2, 3)),
            data_quality_score=0.90,
            action_names=["CASH", "SOXL"],
        )
        readiness_config = ReadinessConfig(
            weights={"data_quality": 0.5, "liquidity": 0.5},
            thresholds={"reduced_risk": 0.75},
            min_data_quality=0.9,
            min_liquidity_score=0.8,
            min_constraint_budget=0.5,
            version="v1",
        )
        log = DecisionLog(
            decision_id="d1",
            decision_ts="2026-01-02T14:30:00+00:00",
            model_id="m1",
            selected_action="CASH",
            previous_action="CASH",
            action_mask_reasons={"SOXL": "spread_too_wide"},
            q_values={"CASH": 0.0, "SOXL": -1.0},
            risk_checks={"quality_gate": True},
            expected_cost_bps=0.0,
            data_quality_score=0.90,
            readiness_score=score,
            readiness_config_hash=readiness_config.content_hash(),
            candidates={
                "CASH": {
                    "valid": True,
                    "q_value": 0.0,
                    "expected_cost_bps": 0.0,
                    "risk_bucket": "cash",
                    "reason": None,
                },
                "SOXL": {
                    "valid": False,
                    "q_value": -1.0,
                    "expected_cost_bps": 3.0,
                    "risk_bucket": "leveraged_etf",
                    "reason": "spread_too_wide",
                },
            },
        )

        snapshot.validate()
        log.validate()
        self.assertEqual(readiness_band(score), "reduced_risk")

    def test_decision_snapshot_rejects_nan_and_negative_action_cost(self) -> None:
        snapshot = DecisionSnapshot(
            decision_ts="2026-01-02T14:30:00+00:00",
            instrument_universe_hash="abc",
            market_state=torch.tensor([float("nan")]),
            portfolio_state=torch.zeros(1),
            action_valid_mask=torch.tensor([True, False]),
            action_cost_estimate_bps=torch.tensor([0.0, -1.0]),
            action_risk_features=torch.zeros((2, 1)),
            data_quality_score=0.90,
            action_names=["CASH", "SOXL"],
        )

        with self.assertRaisesRegex(DecisionFrameworkError, "market_state"):
            snapshot.validate()

    def test_decision_dataset_invalid_action_returns_must_be_nan(self) -> None:
        snapshot = DecisionSnapshot(
            decision_ts="2026-01-02T14:30:00+00:00",
            instrument_universe_hash="abc",
            market_state=torch.zeros(1),
            portfolio_state=torch.zeros(1),
            action_valid_mask=torch.tensor([True, False]),
            action_cost_estimate_bps=torch.tensor([0.0, 1.0]),
            action_risk_features=torch.zeros((2, 1)),
            data_quality_score=0.90,
            action_names=["CASH", "SOXL"],
        )
        dataset = DecisionDataset(
            snapshots=[snapshot],
            action_returns=torch.tensor([[0.0, 0.0]]),
            action_valid_mask=torch.tensor([[True, False]]),
            action_cost_bps=torch.tensor([[0.0, 1.0]]),
            next_timestamps=["2026-01-02T14:31:00+00:00"],
            manifests=["manifest"],
        )

        with self.assertRaisesRegex(DecisionFrameworkError, "Invalid action returns"):
            dataset.validate()

    def test_validate_reportable_summary_requires_decision_artifacts_and_random_baseline(self) -> None:
        errors = validate_reportable_summary(
            {
                "test_metrics": {"total_return": -0.01},
                "baselines": {
                    "CASH": {"test": {"total_return": 0.0}},
                    "BuyAndHold_QQQ": {"test": {"total_return": 0.01}},
                },
                "cost_stress": {"fixed_rollout": {}, "adaptive": {}},
                "action_concentration": {
                    "max_risky_group_share": 0.9,
                    "leveraged_action_share": 0.8,
                },
                "return_diagnostics": {},
            }
        )

        self.assertIn("missing dataset_manifest", errors)
        self.assertIn("missing baselines.RandomSameTurnover", errors)
        self.assertIn("test_return_below_cash", errors)
        self.assertIn("max_group_share_exceeds_limit", errors)

    def test_validate_reportable_summary_accepts_canonical_cost_stress_and_random_baseline(self) -> None:
        errors = validate_reportable_summary(
            {
                "dataset_manifest": {},
                "feature_manifest": {},
                "model_manifest": {},
                "data_quality_report": {},
                "action_eligibility": [],
                "test_metrics": {"total_return": 0.0},
                "baselines": {
                    "CASH": {"test": {"total_return": 0.0}},
                    "RandomSameTurnover": {"test": {"total_return": 0.0}},
                },
                "cost_stress": {"fixed_rollout": {}, "adaptive": {}},
                "action_concentration": {
                    "max_risky_group_share": 0.0,
                    "leveraged_action_share": 0.0,
                },
                "return_diagnostics": {},
            }
        )

        self.assertEqual(errors, [])


class RepositoryHygieneTests(unittest.TestCase):
    def test_no_current_tree_secret_literals(self) -> None:
        forbidden = [
            "X-RapidAPI-Key",
            "X-RapidAPI-Proxy-Secret",
            "TRADING_PWD =",
        ]
        excluded_parts = {".git", "__pycache__"}
        excluded_names = {".env.example", "test_correctness.py"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or excluded_parts.intersection(path.parts) or path.name in excluded_names:
                continue
            try:
                text = path.read_text(errors="ignore")
            except UnicodeDecodeError:
                continue
            for needle in forbidden:
                self.assertNotIn(needle, text, f"Forbidden secret-like literal {needle!r} found in {path}")


class ResearchProtocolTests(unittest.TestCase):
    def test_fit_window_must_be_prior_only(self) -> None:
        FitWindow(
            fit_start="2026-01-01T00:00:00+00:00",
            fit_end="2026-01-02T00:00:00+00:00",
            feature_asof="2026-01-03T00:00:00+00:00",
        ).validate_prior_only()

        with self.assertRaises(ResearchProtocolError):
            FitWindow(
                fit_start="2026-01-01T00:00:00+00:00",
                fit_end="2026-01-03T00:00:00+00:00",
                feature_asof="2026-01-03T00:00:00+00:00",
            ).validate_prior_only()

    def test_dataset_manifest_validates_point_in_time_universe(self) -> None:
        manifest = DatasetManifest(
            dataset_id="demo",
            created_at_utc="2026-06-14T00:00:00+00:00",
            source_vendor="synthetic",
            symbols=["QQQ"],
            universe_selection_date="2026-01-01T00:00:00+00:00",
            bar_interval="1h",
            timezone="UTC",
            adjustment="synthetic",
            feature_names=["x"],
            action_names=["CASH", "QQQ"],
            timestamps_hash=hash_string_sequence(["2026-01-02T00:00:00+00:00"]),
            next_timestamps_hash=hash_string_sequence(["2026-01-02T01:00:00+00:00"]),
            first_timestamp="2026-01-02T00:00:00+00:00",
            last_timestamp="2026-01-02T00:00:00+00:00",
        )
        manifest.validate()

        bad = DatasetManifest.from_dict(manifest.to_dict())
        bad.universe_selection_date = "2026-01-03T00:00:00+00:00"
        with self.assertRaises(ResearchProtocolError):
            bad.validate()

    def test_model_manifest_requires_baselines_and_stress_tests(self) -> None:
        protocol = EvaluationProtocol(
            name="holdout",
            train_start=None,
            train_end="2026-01-31T00:00:00+00:00",
            val_end="2026-02-28T00:00:00+00:00",
            test_start="2026-03-01T00:00:00+00:00",
            test_end="2026-03-31T00:00:00+00:00",
            benchmark_names=["CASH"],
        )
        manifest = ModelManifest(
            model_id="demo_model",
            created_at_utc="2026-06-14T00:00:00+00:00",
            algorithm="DQN",
            encoder="MinuteToHourTransformer",
            training_dataset_id="demo",
            validation_protocol=protocol,
            hyperparameter_search_space_hash="abc",
            hyperparameter_trials=1,
            selected_by="validation_net_return",
            feature_names_hash="features",
            action_names_hash="actions",
        )
        with self.assertRaises(ResearchProtocolError):
            manifest.validate_reportable()

        manifest.baseline_results.append(BaselineResult("CASH", 0.0, None, 0.0))
        manifest.cost_stress_results.append(StressTestResult("2x_cost", "cost", "multiplier", 2.0, 0.0, None, 0.0))
        manifest.frequency_stress_results.append(
            StressTestResult("min_hold_2", "frequency", "min_hold_bars", 2.0, 0.0, None, 0.0)
        )
        manifest.validate_reportable()

    def test_default_benchmark_registry_matches_action_universe(self) -> None:
        benchmarks = default_benchmark_registry(["CASH", "QQQ", "SPY"])

        self.assertIn("CASH", benchmarks)
        self.assertIn("BuyAndHold_QQQ", benchmarks)
        self.assertIn("RandomWithSameTurnover", benchmarks)


if __name__ == "__main__":
    unittest.main()
