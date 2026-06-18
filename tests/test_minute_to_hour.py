from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from datetime import (
    datetime,
    timedelta,
)
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import torch
from torch import nn
from rl_quant.confidence import (
    ACTION_CONFIDENCE_FIELD_NAMES,
    ActionConfidenceCalibrator,
    ActionConfidenceConfig,
    save_action_confidence_npz,
)
from rl_quant.data_sources.polygon_second_aggs import (
    PolygonSecondAggConfig,
    available_timestamp_ms,
    iso_to_timestamp_ms,
    load_manifest,
    timestamp_ms_to_iso,
    validate_manifest,
)
from rl_quant.data_sources.polygon_stock_covariates import (
    normalize_raw_covariate_record,
    regular_session_open_ms_after_date,
)
from rl_quant.features.stock_covariates import (
    ACTION_COVARIATE_ACTION_TYPE_FEATURE_NAMES,
    ACTION_COVARIATE_FEATURE_NAMES,
    ACTION_COVARIATE_SCHEMA_HASH,
    append_action_covariates_to_payload,
    build_action_covariate_tensor,
    build_symbol_silver_rows,
    read_covariate_coverage_manifest,
    tensor_content_hash,
    validate_action_covariate_feature_schema,
)
from rl_quant.features.news_llm import (
    NEWS_LLM_AGGREGATE_FEATURE_NAMES,
    NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH,
    NEWS_LLM_EXTRACT_SCHEMA_VERSION,
    aggregate_news_llm_features_for_symbol,
    build_action_news_llm_tensor,
    build_deterministic_news_llm_rows,
    build_news_article_rows,
    write_news_llm_feature_outputs,
)
from rl_quant.features.stock_second_context import (
    StockSecondContextConfig,
    build_second_context_payload,
    regular_session_decision_grid_ms,
    save_second_context_payload,
    validate_second_context_payload,
)
from rl_quant.research_protocol import stable_json_hash
from rl_quant.second_context_transformer import (
    SecondContextDataSplit,
    SecondContextTransformerQNetwork,
    build_second_context_splits,
    evaluate_second_context_action_scorer,
    evaluate_second_context_baselines,
    evaluate_second_context_trading_policy,
    fixed_rollout_cost_stress,
    masked_contextual_q_loss,
    second_context_missing_label_report,
)
from rl_quant.hourly_transformer import CausalTransformerQNetwork
from rl_quant.minute_to_hour_transformer import (
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
from rl_quant.trading_constraints import (
    CONSTRAINT_FEATURE_DIM,
    CONSTRAINT_FEATURE_NAMES,
)
from _support import ROOT, load_script


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

    @staticmethod
    def _write_minute_to_hour_dataset(
        path: Path,
        *,
        decisions: list[str],
        next_timestamps: list[str],
        session_ids: list[str] | None = None,
        minute_values: list[float] | None = None,
    ) -> None:
        rows = len(decisions)
        minute_values = minute_values or [0.0] * rows
        minute_grid = []
        for decision in decisions:
            context_dt = datetime.fromisoformat(decision) - timedelta(minutes=1)
            minute_grid.append([[context_dt.isoformat()]])
        payload = {
            "decision_timestamps": decisions,
            "next_timestamps": next_timestamps,
            "minute_timestamp_grid": minute_grid,
            "minute_feature_names": ["m"],
            "hour_feature_names": ["h"],
            "action_names": ["CASH", "QQQ"],
            "minute_features": torch.tensor(minute_values, dtype=torch.float32).view(rows, 1, 1, 1),
            "minute_mask": torch.ones((rows, 1, 1), dtype=torch.bool),
            "hour_features": torch.zeros((rows, 1, 1), dtype=torch.float32),
            "action_returns": torch.zeros((rows, 2), dtype=torch.float32),
            "decision_action_valid_mask": torch.ones((rows, 2), dtype=torch.bool),
            "label_valid_mask": torch.ones((rows, 2), dtype=torch.bool),
        }
        if session_ids is not None:
            payload["session_ids"] = session_ids
        torch.save(payload, path)

    def test_hourly_context_uses_only_past_minutes(self) -> None:
        # Exercise the real causal price lookups: the validity/feature price is the last close
        # AT OR BEFORE the decision (never a future bar), and the simulated fill is the first close
        # AT OR AFTER decision + execution latency (a price not observable at decision time).
        module = load_script("build_hourly_from_minute_context_dataset")
        timestamps = [
            "2026-06-10T15:28:00+00:00",
            "2026-06-10T15:29:00+00:00",
            "2026-06-10T15:31:00+00:00",
        ]
        closes = [100.0, 101.0, 102.0]
        lookup = (timestamps, closes)
        decision_ms = module.timestamp_to_epoch_ms("2026-06-10T15:30:00+00:00")
        feature_close = module.close_at_or_before(lookup, decision_ms, max_staleness_seconds=120)
        self.assertEqual(feature_close, 101.0)
        fill_close = module.close_at_or_after(lookup, decision_ms + 1_000, max_staleness_seconds=120)
        self.assertEqual(fill_close, 102.0)

    def test_next_hour_reward_uses_decision_close_to_next_hour_close(self) -> None:
        module = load_script("build_hourly_transformer_dataset")

        self.assertAlmostEqual(module.clipped_simple_return(100.0, 105.0), 0.05)

    def test_aggregate_stock_features_single_stock_fast_path_matches_contract(self) -> None:
        module = load_script("build_hourly_transformer_dataset")
        item = module.BarFeature(
            close=100.0,
            bar_return=0.02,
            bar_log_return=0.0198,
            intraday_ret=0.01,
            range_bps=25.0,
            log_volume=8.0,
            log_dollar_volume=12.0,
            dollar_volume=10_000.0,
        )

        values = module.aggregate_stock_features([item], total_symbols=10)

        self.assertEqual(values[0], 0.1)
        self.assertEqual(values[1], item.bar_return)
        self.assertEqual(values[2], item.bar_return)
        self.assertEqual(values[3], 0.0)
        self.assertEqual(values[5], item.bar_return)
        self.assertEqual(values[6], item.bar_return)
        self.assertEqual(values[8], item.range_bps)
        self.assertEqual(values[9], 0.0)
        self.assertEqual(values[10], item.log_dollar_volume)
        self.assertEqual(values[11], 0.0)
        self.assertEqual(values[12], 1.0)
        self.assertEqual(values[13], abs(item.bar_return))

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

    def test_minute_to_hour_builder_prunes_daily_parquet_shards_by_time_range(self) -> None:
        module = load_script("build_hourly_from_minute_context_dataset")

        paths = [
            Path("AAA/2026/06/2026-06-09.parquet"),
            Path("AAA/2026/06/2026-06-10.parquet"),
            Path("AAA/2026/06/2026-06-11.parquet"),
            Path("AAA/2026/06/2026-06-12.parquet"),
            Path("AAA.parquet"),
        ]
        filtered = module.filter_bar_paths_for_time_range(
            paths,
            start_dt=module.parse_utc_datetime("2026-06-10T13:30:00+00:00"),
            end_dt=module.parse_utc_datetime("2026-06-12T00:00:00+00:00"),
        )

        self.assertEqual(
            filtered,
            [
                Path("AAA/2026/06/2026-06-10.parquet"),
                Path("AAA/2026/06/2026-06-11.parquet"),
                Path("AAA.parquet"),
            ],
        )

    def test_minute_to_hour_action_lookup_preserves_staleness_semantics(self) -> None:
        module = load_script("build_hourly_from_minute_context_dataset")

        lookup = module.make_action_lookup(
            {
                "2026-06-10T14:30:00+00:00": SimpleNamespace(close=100.0),
                "2026-06-10T14:31:00+00:00": SimpleNamespace(close=101.0),
            }
        )

        self.assertEqual(
            module.close_at_or_before(
                lookup,
                "2026-06-10T14:30:30+00:00",
                max_staleness_seconds=60,
            ),
            100.0,
        )
        self.assertIsNone(
            module.close_at_or_before(
                lookup,
                "2026-06-10T14:30:30+00:00",
                max_staleness_seconds=0,
            )
        )
        self.assertEqual(
            module.close_at_or_before(
                lookup,
                module.timestamp_to_epoch_ms("2026-06-10T14:31:00+00:00"),
                max_staleness_seconds=0,
            ),
            101.0,
        )
        self.assertIsNone(
            module.close_at_or_before(
                lookup,
                "2026-06-10T14:32:01+00:00",
                max_staleness_seconds=60,
            )
        )

    def test_hourly_builder_retains_decision_valid_row_with_missing_future_label(self) -> None:
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas/pyarrow are required for Parquet builder regression")
        builder = load_script("build_hourly_from_minute_context_dataset")

        def write_bar_parquet(path: Path, symbol: str, *, rows: int) -> None:
            start_utc = builder.parse_utc_datetime("2026-01-02T14:31:00+00:00")
            records: list[dict[str, object]] = []
            for offset in range(rows):
                utc_dt = start_utc + builder.timedelta(minutes=offset)
                exchange_dt = utc_dt.astimezone(builder.timezone(builder.timedelta(hours=-5)))
                price = 100.0 + offset * 0.01 + (0.5 if symbol == "QQQ" else 0.0)
                records.append(
                    {
                        "timestamp_ms": builder.timestamp_to_epoch_ms(utc_dt),
                        "timestamp_utc": utc_dt.isoformat(),
                        "timestamp_exchange": exchange_dt.isoformat(),
                        "open": price,
                        "high": price + 0.1,
                        "low": price - 0.1,
                        "close": price,
                        "volume": 1000,
                    }
                )
            pd.DataFrame.from_records(records).to_parquet(path, index=False)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_dir = root / "stocks"
            action_dir = root / "actions"
            output_dir = root / "out"
            stock_dir.mkdir()
            action_dir.mkdir()
            write_bar_parquet(stock_dir / "AAA.parquet", "AAA", rows=120)
            write_bar_parquet(action_dir / "QQQ.parquet", "QQQ", rows=60)
            stock_universe = root / "stocks.csv"
            action_universe = root / "actions.csv"
            stock_universe.write_text("symbol\nAAA\n")
            action_universe.write_text("symbol\nQQQ\n")
            old_argv = sys.argv
            sys.argv = [
                "build_hourly_from_minute_context_dataset.py",
                "--stock-bar-dir",
                str(stock_dir),
                "--action-bar-dir",
                str(action_dir),
                "--stock-universe",
                str(stock_universe),
                "--action-universe",
                str(action_universe),
                "--output-dir",
                str(output_dir),
                "--dataset-file-name",
                "dataset.pt",
                "--start",
                "2026-01-02T00:00:00+00:00",
                "--end-exclusive",
                "2026-01-03T00:00:00+00:00",
                "--stock-limit",
                "1",
                "--action-count",
                "1",
                "--hours-lookback",
                "1",
                "--context-bars-per-hour",
                "60",
                "--min-active-stock-fraction",
                "1.0",
                "--min-context-valid-fraction",
                "1.0",
                "--min-decision-rows",
                "1",
                "--dense-hourly-grid",
                "--allow-missing-action-context",
                "--universe-selection-date",
                "2026-01-01T00:00:00+00:00",
            ]
            try:
                builder.main()
            finally:
                sys.argv = old_argv

            payload = torch.load(output_dir / "dataset.pt", map_location="cpu", weights_only=True)

        self.assertEqual(payload["decision_timestamps"], ["2026-01-02T15:30:00+00:00"])
        self.assertTrue(bool(payload["decision_action_valid_mask"][0, 1].item()))
        self.assertFalse(bool(payload["label_valid_mask"][0, 1].item()))
        self.assertTrue(torch.isnan(payload["action_returns"][0, 1]))

    def test_hourly_builder_non_dense_keeps_row_when_future_grid_point_missing(self) -> None:
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas/pyarrow are required for Parquet builder regression")
        builder = load_script("build_hourly_from_minute_context_dataset")

        def write_bar_parquet(path: Path, symbol: str, *, rows: int) -> None:
            start_utc = builder.parse_utc_datetime("2026-01-02T14:31:00+00:00")
            records: list[dict[str, object]] = []
            for offset in range(rows):
                utc_dt = start_utc + builder.timedelta(minutes=offset)
                exchange_dt = utc_dt.astimezone(builder.timezone(builder.timedelta(hours=-5)))
                price = 100.0 + offset * 0.01 + (0.5 if symbol == "QQQ" else 0.0)
                records.append(
                    {
                        "timestamp_ms": builder.timestamp_to_epoch_ms(utc_dt),
                        "timestamp_utc": utc_dt.isoformat(),
                        "timestamp_exchange": exchange_dt.isoformat(),
                        "open": price,
                        "high": price + 0.1,
                        "low": price - 0.1,
                        "close": price,
                        "volume": 1000,
                    }
                )
            pd.DataFrame.from_records(records).to_parquet(path, index=False)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_dir = root / "stocks"
            action_dir = root / "actions"
            output_dir = root / "out"
            stock_dir.mkdir()
            action_dir.mkdir()
            write_bar_parquet(stock_dir / "AAA.parquet", "AAA", rows=60)
            write_bar_parquet(action_dir / "QQQ.parquet", "QQQ", rows=60)
            stock_universe = root / "stocks.csv"
            action_universe = root / "actions.csv"
            stock_universe.write_text("symbol\nAAA\n")
            action_universe.write_text("symbol\nQQQ\n")
            old_argv = sys.argv
            sys.argv = [
                "build_hourly_from_minute_context_dataset.py",
                "--stock-bar-dir",
                str(stock_dir),
                "--action-bar-dir",
                str(action_dir),
                "--stock-universe",
                str(stock_universe),
                "--action-universe",
                str(action_universe),
                "--output-dir",
                str(output_dir),
                "--dataset-file-name",
                "dataset.pt",
                "--start",
                "2026-01-02T00:00:00+00:00",
                "--end-exclusive",
                "2026-01-03T00:00:00+00:00",
                "--stock-limit",
                "1",
                "--action-count",
                "1",
                "--hours-lookback",
                "1",
                "--context-bars-per-hour",
                "60",
                "--min-active-stock-fraction",
                "1.0",
                "--min-context-valid-fraction",
                "1.0",
                "--min-decision-rows",
                "1",
                "--allow-missing-action-context",
                "--universe-selection-date",
                "2026-01-01T00:00:00+00:00",
            ]
            try:
                builder.main()
            finally:
                sys.argv = old_argv

            payload = torch.load(output_dir / "dataset.pt", map_location="cpu", weights_only=True)

        self.assertEqual(payload["decision_timestamps"], ["2026-01-02T15:30:00+00:00"])
        self.assertEqual(payload["next_timestamps"], ["2026-01-02T16:30:00+00:00"])
        self.assertTrue(bool(payload["decision_action_valid_mask"][0, 1].item()))
        self.assertFalse(bool(payload["label_valid_mask"][0, 1].item()))
        self.assertTrue(torch.isnan(payload["action_returns"][0, 1]))

    def test_hourly_builder_dense_grid_does_not_depend_on_first_action_symbol(self) -> None:
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas/pyarrow are required for Parquet builder regression")
        builder = load_script("build_hourly_from_minute_context_dataset")

        def write_bar_parquet(path: Path, symbol: str, *, start: str, rows: int) -> None:
            start_utc = builder.parse_utc_datetime(start)
            records: list[dict[str, object]] = []
            for offset in range(rows):
                utc_dt = start_utc + builder.timedelta(minutes=offset)
                exchange_dt = utc_dt.astimezone(builder.timezone(builder.timedelta(hours=-5)))
                price = 100.0 + offset * 0.01 + (1.0 if symbol == "QQQ" else 0.0)
                records.append(
                    {
                        "timestamp_ms": builder.timestamp_to_epoch_ms(utc_dt),
                        "timestamp_utc": utc_dt.isoformat(),
                        "timestamp_exchange": exchange_dt.isoformat(),
                        "open": price,
                        "high": price + 0.1,
                        "low": price - 0.1,
                        "close": price,
                        "volume": 1000,
                    }
                )
            pd.DataFrame.from_records(records).to_parquet(path, index=False)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_dir = root / "stocks"
            action_dir = root / "actions"
            output_dir = root / "out"
            stock_dir.mkdir()
            action_dir.mkdir()
            write_bar_parquet(stock_dir / "AAA.parquet", "AAA", start="2026-01-02T14:31:00+00:00", rows=120)
            write_bar_parquet(action_dir / "OLD.parquet", "OLD", start="2026-01-05T14:31:00+00:00", rows=120)
            write_bar_parquet(action_dir / "QQQ.parquet", "QQQ", start="2026-01-02T14:31:00+00:00", rows=120)
            stock_universe = root / "stocks.csv"
            action_universe = root / "actions.csv"
            stock_universe.write_text("symbol\nAAA\n")
            action_universe.write_text("symbol\nOLD\nQQQ\n")
            old_argv = sys.argv
            sys.argv = [
                "build_hourly_from_minute_context_dataset.py",
                "--stock-bar-dir",
                str(stock_dir),
                "--action-bar-dir",
                str(action_dir),
                "--stock-universe",
                str(stock_universe),
                "--action-universe",
                str(action_universe),
                "--output-dir",
                str(output_dir),
                "--dataset-file-name",
                "dataset.pt",
                "--start",
                "2026-01-02T00:00:00+00:00",
                "--end-exclusive",
                "2026-01-03T00:00:00+00:00",
                "--stock-limit",
                "1",
                "--action-count",
                "2",
                "--hours-lookback",
                "1",
                "--context-bars-per-hour",
                "60",
                "--min-active-stock-fraction",
                "1.0",
                "--min-context-valid-fraction",
                "1.0",
                "--min-decision-rows",
                "1",
                "--dense-hourly-grid",
                "--allow-missing-action-context",
                "--universe-selection-date",
                "2026-01-01T00:00:00+00:00",
            ]
            try:
                builder.main()
            finally:
                sys.argv = old_argv

            payload = torch.load(output_dir / "dataset.pt", map_location="cpu", weights_only=True)

        self.assertIn("2026-01-02T15:30:00+00:00", payload["decision_timestamps"])
        self.assertEqual(payload["action_names"], ["CASH", "OLD", "QQQ"])
        self.assertFalse(bool(payload["decision_action_valid_mask"][0, 1].item()))
        self.assertTrue(bool(payload["decision_action_valid_mask"][0, 2].item()))

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

    def test_second_to_hour_builder_switches_to_second_defaults(self) -> None:
        builder = load_script("build_hourly_from_minute_context_dataset")

        args = builder.parse_args(["--source-bar-interval", "1s"])

        self.assertEqual(args.minutes_per_hour, 3600)
        self.assertEqual(args.max_action_staleness_seconds, 300)
        self.assertEqual(args.bar_latency_ms, 1000)
        self.assertTrue(args.dense_hourly_grid)
        self.assertTrue(args.allow_missing_action_context)
        self.assertIn("rl_hour_from_second", str(args.output_dir))

    def test_second_to_hour_payload_accepts_one_second_context(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["validate_hour_level_decision_grid"])

        module.validate_hour_level_decision_grid(
            {
                "decision_timestamps": ["2026-06-12T14:30:00+00:00"],
                "next_timestamps": ["2026-06-12T15:30:00+00:00"],
                "source_bar_interval": "1s",
                "context_bars_per_hour": 3600,
                "decision_grid_minutes": 60,
            }
        )

        with self.assertRaisesRegex(ValueError, "expects 3600 bars"):
            module.validate_hour_level_decision_grid(
                {
                    "decision_timestamps": ["2026-06-12T14:30:00+00:00"],
                    "next_timestamps": ["2026-06-12T15:30:00+00:00"],
                    "source_bar_interval": "1s",
                    "context_bars_per_hour": 60,
                    "decision_grid_minutes": 60,
                }
            )

    def test_second_to_hour_timestamp_grid_rejects_decision_second_leakage(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["validate_minute_timestamp_grid"])

        payload = {
            "decision_timestamps": ["2026-06-12T15:00:00+00:00"],
            "next_timestamps": ["2026-06-12T16:00:00+00:00"],
            "minute_timestamp_grid": [[["2026-06-12T15:00:00+00:00"]]],
            "minute_mask": torch.tensor([[[True]]], dtype=torch.bool),
            "source_bar_interval": "1s",
            "bar_latency_ms": 1000,
        }

        with self.assertRaisesRegex(ValueError, "Subhour context leakage"):
            module.validate_minute_timestamp_grid(payload)

        payload["minute_timestamp_grid"] = [[["2026-06-12T14:59:59+00:00"]]]
        module.validate_minute_timestamp_grid(payload)

        canonical_payload = {
            "decision_timestamps": ["2026-06-12T15:00:00+00:00"],
            "next_timestamps": ["2026-06-12T16:00:00+00:00"],
            "subhour_timestamp_grid": [[["2026-06-12T14:59:59+00:00"]]],
            "subhour_mask": torch.tensor([[[True]]], dtype=torch.bool),
            "source_bar_interval": "1s",
            "bar_latency_ms": 1000,
        }
        module.validate_minute_timestamp_grid(canonical_payload)

    def test_subhour_payload_aliases_replace_minute_canonical_names(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_load_payload"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            torch.save(
                {
                    "decision_timestamps": ["2026-06-10T15:30:00+00:00"],
                    "next_timestamps": ["2026-06-10T16:30:00+00:00"],
                    "subhour_timestamp_grid": [[["2026-06-10T15:29:00+00:00"]]],
                    "subhour_feature_names": ["x"],
                    "hour_feature_names": ["h"],
                    "action_names": ["CASH", "QQQ"],
                    "subhour_features": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                    "subhour_mask": torch.tensor([[[True]]], dtype=torch.bool),
                    "hour_features": torch.zeros((1, 1, 1), dtype=torch.float32),
                    "action_returns": torch.zeros((1, 2), dtype=torch.float32),
                    "source_bar_interval": "1m",
                    "decision_grid_minutes": 60,
                },
                path,
            )

            payload = module._load_payload(path)

        self.assertIn("minute_features", payload)
        self.assertIn("subhour_features", payload)
        self.assertEqual(tuple(payload["minute_features"].shape), (1, 1, 1, 1))

    def test_subhour_payload_aliases_reject_same_shape_different_values(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_canonicalize_subhour_payload"])
        payload = {
            "subhour_features": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "minute_features": torch.ones((1, 1, 1, 1), dtype=torch.float32),
        }

        with self.assertRaisesRegex(ValueError, "same values"):
            module._canonicalize_subhour_payload(payload)

    def test_second_level_model_compresses_long_intrahour_context(self) -> None:
        model = MinuteToHourCausalTransformerQNetwork(
            minute_feature_dim=2,
            hour_feature_dim=1,
            action_count=2,
            hours_lookback=1,
            minutes_per_hour=3600,
            d_model=16,
            n_heads=4,
            minute_layers=1,
            hour_layers=1,
            feedforward_dim=32,
            action_embedding_dim=4,
            max_subhour_tokens=32,
        )

        q_values = model(
            torch.zeros((2, 1, 3600, 2), dtype=torch.float32),
            torch.ones((2, 1, 3600), dtype=torch.bool),
            torch.zeros((2, 1, 1), dtype=torch.float32),
            torch.zeros(2, dtype=torch.long),
            torch.zeros((2, CONSTRAINT_FEATURE_DIM), dtype=torch.float32),
        )

        self.assertEqual(tuple(q_values.shape), (2, 2))

    def test_minute_to_hour_action_covariate_sidecar_merges_into_split(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_load_payload", "_build_split"])
        decisions = ["2026-06-10T15:30:00+00:00", "2026-06-10T16:30:00+00:00"]
        next_timestamps = ["2026-06-10T16:30:00+00:00", "2026-06-10T17:30:00+00:00"]
        action_features = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0], [1.0, 10.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.0], [3.0, 20.0, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hour_from_second_dataset.pt"
            torch.save(
                {
                    "decision_timestamps": decisions,
                    "next_timestamps": next_timestamps,
                    "minute_timestamp_grid": [
                        [["2026-06-10T15:29:59+00:00"]],
                        [["2026-06-10T16:29:59+00:00"]],
                    ],
                    "minute_feature_names": ["m"],
                    "hour_feature_names": ["h"],
                    "action_names": ["CASH", "QQQ"],
                    "minute_features": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
                    "minute_mask": torch.ones((2, 1, 1), dtype=torch.bool),
                    "hour_features": torch.zeros((2, 1, 1), dtype=torch.float32),
                    "action_returns": torch.zeros((2, 2), dtype=torch.float32),
                    "source_bar_interval": "1s",
                    "context_bars_per_hour": 3600,
                    "minutes_per_hour": 3600,
                    "decision_grid_minutes": 60,
                    "bar_latency_ms": 1000,
                },
                path,
            )
            torch.save(
                {
                    "base_dataset_file_name": path.name,
                    "base_dataset_sha256": module._file_sha256(path),
                    "decision_timestamps": decisions,
                    "action_names": ["CASH", "QQQ"],
                    "action_features": action_features,
                    "action_feature_names": [
                        "stock_covariates_v1.log_market_cap",
                        "stock_covariates_v1.days_since_listed",
                        "stock_covariates_v1_mask.log_market_cap",
                        "stock_covariates_v1_mask.days_since_listed",
                    ],
                    "action_feature_available_timestamps_ms": torch.full((2, 2, 4), -1, dtype=torch.long),
                    "action_feature_groups": {
                        "stock_covariates_v1": [0, 2],
                        "stock_covariates_v1_mask": [2, 4],
                    },
                    "action_covariate_reportability_errors": [],
                },
                path.with_name("action_covariates.pt"),
            )

            payload = module._load_payload(path)
            split = module._build_split(name="train", payload=payload)

        self.assertEqual(tuple(split.action_features.shape), (2, 2, 4))
        self.assertEqual(split.action_feature_names[-1], "stock_covariates_v1_mask.days_since_listed")
        self.assertEqual(split.action_feature_groups["stock_covariates_v1"], [0, 2])
        self.assertTrue(torch.equal(split.action_features[:, :, 2:], action_features[:, :, 2:]))

    def test_minute_to_hour_sidecar_mode_required_and_none(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_load_payload"])
        decisions = ["2026-06-10T15:30:00+00:00", "2026-06-10T16:30:00+00:00"]
        next_timestamps = ["2026-06-10T16:30:00+00:00", "2026-06-10T17:30:00+00:00"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hour_from_second_dataset.pt"
            self._write_minute_to_hour_dataset(path, decisions=decisions, next_timestamps=next_timestamps)

            payload = module._load_payload(path, action_covariate_sidecar="none")
            self.assertNotIn("action_features", payload)
            with self.assertRaisesRegex(FileNotFoundError, "Required action covariate sidecar"):
                module._load_payload(path, action_covariate_sidecar="required")

    def test_minute_to_hour_news_llm_sidecar_is_explicit_opt_in(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_load_payload"])
        decisions = ["2026-06-10T15:30:00+00:00", "2026-06-10T16:30:00+00:00"]
        next_timestamps = ["2026-06-10T16:30:00+00:00", "2026-06-10T17:30:00+00:00"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hour_from_second_dataset.pt"
            self._write_minute_to_hour_dataset(path, decisions=decisions, next_timestamps=next_timestamps)

            default_payload = module._load_payload(path, action_covariate_sidecar="none")
            self.assertNotIn("action_features", default_payload)
            with self.assertRaisesRegex(FileNotFoundError, "Required news LLM sidecar"):
                module._load_payload(path, action_covariate_sidecar="none", news_llm_sidecar="required")

            available = torch.tensor(
                [
                    [[-1, iso_to_timestamp_ms(decisions[0])], [iso_to_timestamp_ms(decisions[0]), iso_to_timestamp_ms(decisions[0])]],
                    [[-1, iso_to_timestamp_ms(decisions[1])], [iso_to_timestamp_ms(decisions[1]), iso_to_timestamp_ms(decisions[1])]],
                ],
                dtype=torch.long,
            )
            torch.save(
                {
                    "base_dataset_file_name": path.name,
                    "base_dataset_sha256": module._file_sha256(path),
                    "decision_timestamps": decisions,
                    "action_names": ["CASH", "QQQ"],
                    "action_features": torch.ones((2, 2, 2), dtype=torch.float32),
                    "action_feature_names": [
                        "stock_news_llm_v1.log1p_llm_weighted_news_count_1h",
                        "stock_news_llm_v1_mask.log1p_llm_weighted_news_count_1h",
                    ],
                    "action_feature_available_timestamps_ms": available,
                    "action_feature_groups": {
                        "stock_news_llm_v1": [0, 1],
                        "stock_news_llm_v1_mask": [1, 2],
                    },
                    "action_news_llm_reportability_errors": [],
                },
                path.with_name("action_news_llm_covariates.pt"),
            )

            payload = module._load_payload(path, action_covariate_sidecar="none", news_llm_sidecar="required")

        self.assertEqual(tuple(payload["action_features"].shape), (2, 2, 2))
        self.assertEqual(payload["action_feature_names"][0], "stock_news_llm_v1.log1p_llm_weighted_news_count_1h")
        self.assertTrue(payload["action_features_augmented_with_news_llm"])
        self.assertEqual(payload["action_feature_groups"]["stock_news_llm_v1"], [0, 1])

    def test_minute_to_hour_sidecar_requires_matching_base_hash(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_load_payload"])
        decisions = ["2026-06-10T15:30:00+00:00", "2026-06-10T16:30:00+00:00"]
        next_timestamps = ["2026-06-10T16:30:00+00:00", "2026-06-10T17:30:00+00:00"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hour_from_second_dataset.pt"
            self._write_minute_to_hour_dataset(path, decisions=decisions, next_timestamps=next_timestamps)
            torch.save(
                {
                    "base_dataset_file_name": path.name,
                    "base_dataset_sha256": "not-the-current-dataset-hash",
                    "decision_timestamps": decisions,
                    "action_names": ["CASH", "QQQ"],
                    "action_features": torch.zeros((2, 2, 1), dtype=torch.float32),
                    "action_feature_names": ["stock_covariates_v1.log_market_cap"],
                    "action_feature_available_timestamps_ms": torch.full((2, 2, 1), -1, dtype=torch.long),
                    "action_covariate_reportability_errors": [],
                },
                path.with_name("action_covariates.pt"),
            )

            with self.assertRaisesRegex(ValueError, "base_dataset_sha256"):
                module._load_payload(path, action_covariate_sidecar="required")

    def test_binary_action_features_are_not_zscored(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_action_feature_mean_std"])
        features = torch.tensor(
            [
                [[1.0, 1.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 0.0, 0.0]],
                [[3.0, 1.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        mean, std = module._action_feature_mean_std(
            features,
            [
                "stock_covariates_v1.log_market_cap",
                "stock_covariates_v1_type.is_cash_action",
                "stock_covariates_v1_type.is_non_cash_action",
                "stock_covariates_v1.is_common_stock",
                "stock_news_llm_v1_mask.log1p_llm_weighted_news_count_1h",
            ],
        )

        self.assertNotEqual(float(mean[0].item()), 0.0)
        self.assertEqual(float(mean[1].item()), 0.0)
        self.assertEqual(float(mean[2].item()), 0.0)
        self.assertEqual(float(mean[3].item()), 0.0)
        self.assertEqual(float(mean[4].item()), 0.0)
        self.assertEqual(float(std[1].item()), 1.0)
        self.assertEqual(float(std[2].item()), 1.0)
        self.assertEqual(float(std[3].item()), 1.0)
        self.assertEqual(float(std[4].item()), 1.0)

    def test_minute_to_hour_model_scores_action_features(self) -> None:
        model = MinuteToHourCausalTransformerQNetwork(
            minute_feature_dim=2,
            hour_feature_dim=1,
            action_count=3,
            hours_lookback=1,
            minutes_per_hour=2,
            d_model=16,
            n_heads=4,
            minute_layers=1,
            hour_layers=1,
            feedforward_dim=32,
            action_embedding_dim=4,
            action_feature_dim=5,
        )

        q_values = model(
            torch.zeros((2, 1, 2, 2), dtype=torch.float32),
            torch.ones((2, 1, 2), dtype=torch.bool),
            torch.zeros((2, 1, 1), dtype=torch.float32),
            torch.zeros(2, dtype=torch.long),
            torch.zeros((2, CONSTRAINT_FEATURE_DIM), dtype=torch.float32),
            action_features=torch.zeros((2, 3, 5), dtype=torch.float32),
        )

        self.assertEqual(tuple(q_values.shape), (2, 3))
        self.assertTrue(bool(torch.isfinite(q_values).all().item()))

    def test_vectorized_minute_to_hour_env_allows_last_valid_next_state(self) -> None:
        split = HourFromMinuteDataSplit(
            name="train",
            decision_timestamps=["2026-01-02T14:30:00+00:00", "2026-01-02T15:30:00+00:00"],
            next_timestamps=["2026-01-02T15:30:00+00:00", "2026-01-02T16:30:00+00:00"],
            minute_feature_names=["m"],
            hour_feature_names=["h"],
            action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((2, 1, 1), dtype=torch.float32),
            action_returns=torch.zeros((2, 2), dtype=torch.float32),
            action_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            label_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1], dtype=torch.long),
            valid_index_mask=torch.tensor([True, True], dtype=torch.bool),
            minute_feature_mean=torch.zeros(1),
            minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1),
            hour_feature_std=torch.ones(1),
            hours_lookback=1,
            minutes_per_hour=1,
        )
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["VectorizedMinuteToHourEnv"])
        env = module.VectorizedMinuteToHourEnv(
            split,
            module.MinuteToHourEnvConfig(num_envs=1, episode_length=10, initial_action=0),
            torch.device("cpu"),
        )
        env.indices[:] = 0

        result = env.step(torch.tensor([0], dtype=torch.long))

        self.assertEqual(int(result["next_indices"][0].item()), 1)
        self.assertFalse(bool(result["resets"][0].item()))

    def test_minute_to_hour_env_d0_dynamic_bookkeeping(self) -> None:
        # PR-D / D0: env tracks entry_index / unrealized_pnl / mae / mfe as PURE bookkeeping (not consumed by
        # reward/model/replay -> training is byte-identical; that part is covered by the existing trainer tests
        # staying green). Here we verify the compounding/reset semantics directly, robust to constraint
        # redirection by reading the ACTUALLY-executed action from the step result.
        split = HourFromMinuteDataSplit(
            name="train",
            decision_timestamps=[f"2026-01-02T1{h}:30:00+00:00" for h in range(4)],
            next_timestamps=[f"2026-01-02T1{h + 1}:30:00+00:00" for h in range(4)],
            minute_feature_names=["m"],
            hour_feature_names=["h"],
            action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((4, 1, 1, 1), dtype=torch.float32),
            minute_mask=torch.ones((4, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((4, 1, 1), dtype=torch.float32),
            action_returns=torch.tensor([[0.0, 0.10], [0.0, -0.20], [0.0, 0.30], [0.0, 0.0]], dtype=torch.float32),
            action_valid_mask=torch.ones((4, 2), dtype=torch.bool),
            label_valid_mask=torch.ones((4, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            valid_index_mask=torch.tensor([True, True, True, True], dtype=torch.bool),
            minute_feature_mean=torch.zeros(1),
            minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1),
            hour_feature_std=torch.ones(1),
            hours_lookback=1,
            minutes_per_hour=1,
        )
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["VectorizedMinuteToHourEnv"])
        env = module.VectorizedMinuteToHourEnv(
            split, module.MinuteToHourEnvConfig(num_envs=1, episode_length=10, initial_action=0), torch.device("cpu")
        )
        env.indices[:] = 0
        env.entry_index[:] = 0  # align bookkeeping with the manually-set start row

        for requested in (1, 1, 0):
            idx_before = int(env.indices[0].item())
            prev_action = int(env.previous_actions[0].item())
            u_before = float(env.unrealized_pnl[0].item())
            mae_before = float(env.mae[0].item())
            mfe_before = float(env.mfe[0].item())

            result = env.step(torch.tensor([requested], dtype=torch.long))

            executed = int(result["actions"][0].item())
            raw = float(split.action_returns[idx_before, executed].item())
            switched = executed != prev_action
            cum_expected = raw if switched else (1.0 + u_before) * (1.0 + raw) - 1.0
            self.assertAlmostEqual(float(env.unrealized_pnl[0].item()), cum_expected, places=6)
            if switched:
                self.assertEqual(int(env.entry_index[0].item()), idx_before)
                self.assertAlmostEqual(float(env.mae[0].item()), min(0.0, cum_expected), places=6)
                self.assertAlmostEqual(float(env.mfe[0].item()), max(0.0, cum_expected), places=6)
            else:
                self.assertAlmostEqual(float(env.mae[0].item()), min(mae_before, cum_expected), places=6)
                self.assertAlmostEqual(float(env.mfe[0].item()), max(mfe_before, cum_expected), places=6)
            # Path invariant: the latest cum sits within [MAE, MFE].
            self.assertLessEqual(float(env.mae[0].item()), float(env.unrealized_pnl[0].item()) + 1e-6)
            self.assertGreaterEqual(float(env.mfe[0].item()), float(env.unrealized_pnl[0].item()) - 1e-6)

        # The bookkeeping is NOT exported in the step dict (D2's job), so training/replay is untouched.
        self.assertNotIn("unrealized_pnl", result)
        self.assertNotIn("entry_index", result)

    def test_minute_to_hour_dynamic_env_state_checkpoint(self) -> None:
        # Reorg review P0: the dynamic env bookkeeping (entry_index/unrealized_pnl/mae/mfe) must be
        # checkpointed and restored when use_dynamic_transition_features is on -- else a resumed dynamic run
        # silently resets it mid-episode while replay still holds dynamic-aware samples. Verify the round-trip
        # restores the fields, and that a legacy checkpoint missing them fails closed for a dynamic resume
        # (and is tolerated when the flag is off).
        from rl_quant.training.minute_to_hour import _env_state_to_cpu, _load_env_state

        split = HourFromMinuteDataSplit(
            name="train",
            decision_timestamps=[f"2026-01-02T1{h}:30:00+00:00" for h in range(4)],
            next_timestamps=[f"2026-01-02T1{h + 1}:30:00+00:00" for h in range(4)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((4, 1, 1, 1), dtype=torch.float32),
            minute_mask=torch.ones((4, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((4, 1, 1), dtype=torch.float32),
            action_returns=torch.tensor([[0.0, 0.10], [0.0, -0.20], [0.0, 0.30], [0.0, 0.0]], dtype=torch.float32),
            action_valid_mask=torch.ones((4, 2), dtype=torch.bool),
            label_valid_mask=torch.ones((4, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            valid_index_mask=torch.tensor([True, True, True, True], dtype=torch.bool),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["VectorizedMinuteToHourEnv"])
        cfg = module.MinuteToHourEnvConfig(num_envs=1, episode_length=10, initial_action=0)
        env = module.VectorizedMinuteToHourEnv(split, cfg, torch.device("cpu"))
        env.indices[:] = 0
        env.entry_index[:] = 0
        for requested in (1, 1):  # hold QQQ two steps -> non-zero unrealized_pnl / MAE / MFE
            env.step(torch.tensor([requested], dtype=torch.long))
        self.assertNotEqual(float(env.unrealized_pnl[0].item()), 0.0)

        saved = _env_state_to_cpu(env)
        for key in ("entry_index", "unrealized_pnl", "mae", "mfe"):
            self.assertIn(key, saved)
        fresh = module.VectorizedMinuteToHourEnv(split, cfg, torch.device("cpu"))
        _load_env_state(fresh, saved, torch.device("cpu"), require_dynamic=True)
        for key in ("entry_index", "unrealized_pnl", "mae", "mfe"):
            self.assertTrue(torch.equal(getattr(fresh, key), getattr(env, key)), key)

        legacy = {k: v for k, v in saved.items() if k not in ("entry_index", "unrealized_pnl", "mae", "mfe")}
        _load_env_state(fresh, legacy, torch.device("cpu"), require_dynamic=False)  # tolerated when flag off
        with self.assertRaises(ValueError):  # fail closed for a dynamic resume
            _load_env_state(fresh, legacy, torch.device("cpu"), require_dynamic=True)

    def test_polygon_second_manifest_marks_incomplete_download_non_reportable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.csv"
            dataset_manifest_path = root / "dataset_manifest.json"
            manifest_path.write_text(
                "symbol,date,status,rows,output_path,output_size,sha256,elapsed_seconds,error\n"
                "AAA,2026-06-12,downloaded,10,missing.parquet,100,,1,\n"
                "BBB,2026-06-12,empty,0,,0,,1,\n"
            )
            dataset_manifest_path.write_text(
                json.dumps(
                    {
                        "source": "Polygon REST aggregate range endpoint",
                        "start": "2026-06-12",
                        "end_exclusive": "2026-06-13",
                        "symbols": 2,
                        "market_weekdays": 2,
                        "remaining_symbol_days": 2,
                        "timespan": "second",
                    }
                )
            )

            rows = load_manifest(manifest_path)
            report = validate_manifest(
                rows,
                PolygonSecondAggConfig(
                    root=root,
                    manifest_csv=manifest_path,
                    dataset_manifest_json=dataset_manifest_path,
                ),
            )

        self.assertFalse(report.reportable)
        self.assertIn("source_download_incomplete", report.reportability_errors)
        self.assertIn("manifest_references_missing_files", report.reportability_errors)
        self.assertEqual(report.source_access, "REST")

    def test_second_bar_available_timestamp_requires_latency(self) -> None:
        self.assertEqual(available_timestamp_ms(1_000, bar_latency_ms=1_000), 2_000)

        with self.assertRaisesRegex(ValueError, "at least 1000"):
            available_timestamp_ms(1_000, bar_latency_ms=500)

    def _second_context_frame(self, symbol: str, timestamps: list[str], closes: list[float]):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is required for in-memory second-context frame tests")

        rows = []
        for timestamp, close in zip(timestamps, closes):
            ms = int(pd.Timestamp(timestamp).timestamp() * 1000)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp_ms": ms,
                    "timestamp_utc": timestamp,
                    "timestamp_exchange": pd.Timestamp(timestamp).tz_convert("America/New_York").isoformat(),
                    "open": close,
                    "high": close * 1.001,
                    "low": close * 0.999,
                    "close": close,
                    "volume": 100.0,
                    "vwap": close,
                    "transactions": 3,
                    "adjusted": True,
                    "timespan": "second",
                    "multiplier": 1,
                }
            )
        return pd.DataFrame(rows)

    def _small_second_context_payload(self, *, allow_post_close_exit: bool = False) -> dict[str, object]:
        decision = "2026-06-12T14:35:00+00:00"
        stock_times = ["2026-06-12T14:34:00+00:00", "2026-06-12T14:34:59+00:00"]
        action_times = [
            "2026-06-12T14:34:59+00:00",
            "2026-06-12T14:35:01+00:00",
            "2026-06-12T14:40:01+00:00",
        ]
        config = StockSecondContextConfig(
            decision_interval="5m",
            context_seconds=60,
            block_seconds=60,
            min_active_symbols=1,
            max_action_staleness_seconds=5,
            allow_post_close_exit=allow_post_close_exit,
        )
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is required for in-memory second-context payload tests")
        return build_second_context_payload(
            stock_frames_by_symbol={
                "AAA": self._second_context_frame("AAA", stock_times, [100.0, 101.0]),
                "BBB": self._second_context_frame("BBB", stock_times, [50.0, 49.5]),
            },
            action_frames_by_symbol={
                "QQQ": self._second_context_frame("QQQ", action_times, [99.0, 100.0, 101.0]),
            },
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[int(pd.Timestamp(decision).timestamp() * 1000)],
            config=config,
            dataset_manifest={"source_download_complete": False},
            data_quality_report={"source_download_complete": False, "reportability_errors": ["source_download_incomplete"]},
        )

    def _second_context_split(
        self,
        *,
        action_names: list[str] | None = None,
        returns: list[list[float]] | torch.Tensor,
        valid_mask: list[list[bool]] | torch.Tensor | None = None,
        label_valid_mask: list[list[bool]] | torch.Tensor | None = None,
        costs: list[list[float]] | torch.Tensor | None = None,
        weights: list[list[float]] | torch.Tensor | None = None,
        decisions: list[str] | None = None,
        next_timestamps: list[str] | None = None,
        segment_ids: list[int] | torch.Tensor | None = None,
        valid_start_indices: list[int] | torch.Tensor | None = None,
        market_mask: torch.Tensor | None = None,
    ) -> SecondContextDataSplit:
        action_names = action_names or ["CASH", "QQQ"]
        action_returns = torch.as_tensor(returns, dtype=torch.float32)
        rows, action_count = action_returns.shape
        if valid_mask is None:
            action_valid_mask = torch.isfinite(action_returns)
        else:
            action_valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool)
        label_valid_tensor = None if label_valid_mask is None else torch.as_tensor(label_valid_mask, dtype=torch.bool)
        if costs is None:
            action_cost_bps = torch.zeros((rows, action_count), dtype=torch.float32)
        else:
            action_cost_bps = torch.as_tensor(costs, dtype=torch.float32)
        if weights is None:
            action_target_weights = torch.ones((rows, action_count), dtype=torch.float32)
            action_target_weights[:, 0] = 0.0
        else:
            action_target_weights = torch.as_tensor(weights, dtype=torch.float32)
        if decisions is None:
            decisions = [f"2026-06-12T14:{30 + 5 * index:02d}:00+00:00" for index in range(rows)]
        if next_timestamps is None:
            next_timestamps = [f"2026-06-12T14:{35 + 5 * index:02d}:00+00:00" for index in range(rows)]
        if segment_ids is None:
            segment_tensor = torch.zeros(rows, dtype=torch.long)
        else:
            segment_tensor = torch.as_tensor(segment_ids, dtype=torch.long)
        if valid_start_indices is None:
            valid_tensor = torch.nonzero(action_valid_mask.any(dim=1), as_tuple=False).flatten().long()
        else:
            valid_tensor = torch.as_tensor(valid_start_indices, dtype=torch.long)
        valid_index_mask = torch.zeros(rows, dtype=torch.bool)
        valid_index_mask[valid_tensor] = True
        if market_mask is None:
            market_mask = torch.ones((rows, 1), dtype=torch.bool)
        return SecondContextDataSplit(
            name="test",
            decision_timestamps=decisions,
            next_timestamps=next_timestamps,
            action_names=action_names,
            feature_names={
                "market_context": ["x"],
                "action_features": ["action_index_scaled"],
                "portfolio_state": ["p"],
                "constraint_state": ["c"],
            },
            market_context=torch.zeros((rows, market_mask.shape[1], 1), dtype=torch.float32),
            market_context_mask=market_mask.bool(),
            market_context_available_timestamps_ms=torch.arange(rows * market_mask.shape[1], dtype=torch.long).reshape(
                rows,
                market_mask.shape[1],
            ),
            action_features=torch.zeros((rows, action_count, 1), dtype=torch.float32),
            action_returns=action_returns,
            action_valid_mask=action_valid_mask,
            action_cost_bps=action_cost_bps,
            action_target_weights=action_target_weights,
            entry_execution_timestamps_ms=torch.ones((rows, action_count), dtype=torch.long),
            exit_execution_timestamps_ms=torch.ones((rows, action_count), dtype=torch.long) * 2,
            entry_price_source="test_entry",
            exit_price_source="test_exit",
            execution_model="test_execution",
            portfolio_state=torch.zeros((rows, 1), dtype=torch.float32),
            constraint_state=torch.zeros((rows, 1), dtype=torch.float32),
            segment_ids=segment_tensor,
            session_ids=["2026-06-12"] * rows,
            valid_start_indices=valid_tensor,
            valid_index_mask=valid_index_mask,
            market_mean=torch.zeros(1, dtype=torch.float32),
            market_std=torch.ones(1, dtype=torch.float32),
            action_feature_mean=torch.zeros(1, dtype=torch.float32),
            action_feature_std=torch.ones(1, dtype=torch.float32),
            periods_per_year=252.0,
            label_valid_mask=label_valid_tensor,
        )

    def test_second_context_payload_uses_latency_and_masks_sparse_blocks(self) -> None:
        payload = self._small_second_context_payload()

        self.assertEqual(tuple(payload["market_context"].shape[:2]), (1, 1))
        self.assertTrue(bool(payload["market_context_mask"][0, 0].item()))
        self.assertLessEqual(
            int(payload["market_context_available_timestamps_ms"][0, 0].item()),
            int(payload["decision_timestamps_ms"][0].item()),
        )
        self.assertAlmostEqual(float(payload["action_returns"][0, 1].item()), 0.01, places=6)
        self.assertGreaterEqual(
            int(payload["entry_execution_timestamps_ms"][0, 1].item()),
            int(payload["decision_timestamps_ms"][0].item()) + 1_000,
        )
        self.assertGreaterEqual(
            int(payload["exit_execution_timestamps_ms"][0, 1].item()),
            int(payload["next_timestamps_ms"][0].item()) + 1_000,
        )
        self.assertFalse(payload["dataset_manifest"]["reportable"])

    def test_second_context_builder_defaults_to_one_second_execution_latency(self) -> None:
        module = load_script("build_second_context_decision_dataset")
        args = module.parse_args([])

        self.assertEqual(args.execution_latency_ms, 1000)
        self.assertTrue(args.strict_action_sources)

    def test_second_context_config_requires_1s_execution_latency(self) -> None:
        # Causal invariant: a 1s-aggregate source must fill at least one bar (1000ms) after the decision,
        # mirroring bar_latency_ms. The config previously enforced only execution_latency_ms >= 0, so a
        # second source with e.g. 500ms (fill at/inside the decision bar -> look-ahead) slipped through.
        from rl_quant.features.stock_second_context import StockSecondContextConfig

        base = dict(decision_interval="5m", context_seconds=60, block_seconds=60, min_active_symbols=1,
                    max_action_staleness_seconds=5)
        StockSecondContextConfig(**base, execution_latency_ms=1000).validate()  # default-equivalent: OK
        for bad in (0, 500, 999):
            with self.assertRaises(ValueError):
                StockSecondContextConfig(**base, execution_latency_ms=bad, source_bar_interval="1s").validate()

    def test_second_context_builder_requires_point_in_time_universe_or_diagnostic_flag(self) -> None:
        module = load_script("build_second_context_decision_dataset")
        first_decision = iso_to_timestamp_ms("2026-06-12T14:35:00+00:00")
        missing_args = module.parse_args([])
        future_args = module.parse_args(["--universe-selection-timestamp", "2026-06-13T00:00:00+00:00"])
        diagnostic_args = module.parse_args(
            [
                "--universe-selection-timestamp",
                "2026-06-13T00:00:00+00:00",
                "--allow-fixed-survivor-universe-diagnostic",
            ]
        )
        lenient_action_args = module.parse_args(["--allow-missing-action-sources-for-diagnostic"])

        with self.assertRaisesRegex(ValueError, "Universe selection"):
            module.universe_reportability_errors(missing_args, first_decision_ms=first_decision)
        with self.assertRaisesRegex(ValueError, "Universe selection"):
            module.universe_reportability_errors(future_args, first_decision_ms=first_decision)
        self.assertEqual(
            module.universe_reportability_errors(diagnostic_args, first_decision_ms=first_decision),
            ["future_universe_selection_timestamp"],
        )
        self.assertFalse(lenient_action_args.strict_action_sources)

    def test_second_context_manifest_marks_future_universe_nonreportable(self) -> None:
        decision = "2026-06-12T14:35:00+00:00"
        stock_times = ["2026-06-12T14:34:00+00:00", "2026-06-12T14:34:59+00:00"]
        action_times = [
            "2026-06-12T14:34:59+00:00",
            "2026-06-12T14:35:01+00:00",
            "2026-06-12T14:40:01+00:00",
        ]
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is required for in-memory second-context payload tests")

        payload = build_second_context_payload(
            stock_frames_by_symbol={
                "AAA": self._second_context_frame("AAA", stock_times, [100.0, 101.0]),
            },
            action_frames_by_symbol={
                "QQQ": self._second_context_frame("QQQ", action_times, [99.0, 100.0, 101.0]),
            },
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[int(pd.Timestamp(decision).timestamp() * 1000)],
            config=StockSecondContextConfig(
                decision_interval="5m",
                context_seconds=60,
                block_seconds=60,
                min_active_symbols=1,
                max_action_staleness_seconds=5,
            ),
            dataset_manifest={
                "source_download_complete": True,
                "universe_selection_timestamp": "2026-06-13T00:00:00+00:00",
                "universe_method": "future_top_volume",
                "universe_source_hash": "future-hash",
            },
            data_quality_report={"source_download_complete": True, "reportability_errors": []},
        )

        self.assertFalse(payload["dataset_manifest"]["reportable"])
        self.assertIn("future_universe_selection_timestamp", payload["dataset_manifest"]["reportability_errors"])
        self.assertTrue(payload["dataset_manifest"]["retrospective_fixed_survivor_universe_diagnostic"])

    def test_second_context_manifest_marks_missing_intended_actions_nonreportable(self) -> None:
        decision = "2026-06-12T14:35:00+00:00"
        stock_times = ["2026-06-12T14:34:00+00:00", "2026-06-12T14:34:59+00:00"]
        action_times = [
            "2026-06-12T14:34:59+00:00",
            "2026-06-12T14:35:01+00:00",
            "2026-06-12T14:40:01+00:00",
        ]
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is required for in-memory second-context payload tests")

        payload = build_second_context_payload(
            stock_frames_by_symbol={
                "AAA": self._second_context_frame("AAA", stock_times, [100.0, 101.0]),
            },
            action_frames_by_symbol={
                "QQQ": self._second_context_frame("QQQ", action_times, [99.0, 100.0, 101.0]),
            },
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[int(pd.Timestamp(decision).timestamp() * 1000)],
            config=StockSecondContextConfig(
                decision_interval="5m",
                context_seconds=60,
                block_seconds=60,
                min_active_symbols=1,
                max_action_staleness_seconds=5,
            ),
            dataset_manifest={
                "source_download_complete": True,
                "universe_selection_timestamp": "2026-06-12T00:00:00+00:00",
                "intended_action_symbols": ["CASH", "QQQ", "MISSING"],
                "realized_action_symbols": ["CASH", "QQQ"],
                "missing_intended_action_source_symbols": ["MISSING"],
                "action_schema_changed_due_to_missing_sources": True,
                "reportability_errors": ["missing_intended_action_source_symbols"],
            },
            data_quality_report={"source_download_complete": True, "reportability_errors": []},
        )
        manifest = dict(payload["dataset_manifest"])

        self.assertFalse(manifest["reportable"])
        self.assertEqual(manifest["missing_intended_action_source_symbols"], ["MISSING"])
        self.assertTrue(manifest["action_schema_changed_due_to_missing_sources"])
        self.assertIn("missing_intended_action_source_symbols", manifest["reportability_errors"])

    def test_hour_from_second_builder_respects_missing_context_and_min_rows_flags(self) -> None:
        module = load_script("build_hourly_from_minute_context_dataset")

        default_args = module.parse_args(["--source-bar-interval", "1s"])
        strict_args = module.parse_args(
            [
                "--source-bar-interval",
                "1s",
                "--no-allow-missing-action-context",
                "--min-decision-rows",
                "1",
            ]
        )

        self.assertTrue(default_args.allow_missing_action_context)
        self.assertFalse(strict_args.allow_missing_action_context)
        self.assertEqual(strict_args.min_decision_rows, 1)

    def test_second_context_action_features_include_action_identity(self) -> None:
        payload = self._small_second_context_payload()
        names = payload["feature_names"]["action_features"]

        self.assertIn("action_index_scaled", names)
        self.assertIn("is_etf", names)
        self.assertIn("is_stock", names)
        self.assertIn("is_inverse", names)
        self.assertIn("is_leveraged", names)
        self.assertIn("leverage_factor", names)
        self.assertIn("target_weight", names)
        self.assertIn("feature_staleness_seconds", names)
        self.assertEqual(float(payload["action_features"][0, 0, names.index("is_cash")].item()), 1.0)
        self.assertEqual(float(payload["action_features"][0, 1, names.index("is_etf")].item()), 1.0)
        self.assertEqual(float(payload["action_features"][0, 1, names.index("is_stock")].item()), 0.0)
        self.assertEqual(float(payload["action_features"][0, 1, names.index("valid_price_flag")].item()), 1.0)
        self.assertEqual(float(payload["action_target_weights"][0, 0].item()), 0.0)
        self.assertEqual(float(payload["action_target_weights"][0, 1].item()), 1.0)
        self.assertLessEqual(
            int(payload["action_features_available_timestamps_ms"][0, 1].item()),
            int(payload["decision_timestamps_ms"][0].item()),
        )
        self.assertIn("action_returns", payload["forbidden_model_input_keys"])
        self.assertNotIn("action_returns", payload["model_input_keys"])
        self.assertEqual(payload["decision_tensor_protocol_version"], "1.0.0")
        self.assertIn("feature_schema_hash", payload)
        self.assertIn("action_metadata", payload)

    def test_second_context_payload_splits_decision_and_label_masks(self) -> None:
        decision = "2026-06-12T14:35:00+00:00"
        stock_times = ["2026-06-12T14:34:00+00:00", "2026-06-12T14:34:59+00:00"]
        action_times = [
            "2026-06-12T14:34:59+00:00",
            "2026-06-12T14:35:01+00:00",
        ]
        config = StockSecondContextConfig(
            decision_interval="5m",
            context_seconds=60,
            block_seconds=60,
            min_active_symbols=1,
            max_action_staleness_seconds=5,
        )
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is required for in-memory second-context payload tests")
        payload = build_second_context_payload(
            stock_frames_by_symbol={
                "AAA": self._second_context_frame("AAA", stock_times, [100.0, 101.0]),
            },
            action_frames_by_symbol={
                "QQQ": self._second_context_frame("QQQ", action_times, [99.0, 100.0]),
            },
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[int(pd.Timestamp(decision).timestamp() * 1000)],
            config=config,
            dataset_manifest={"source_download_complete": True},
            data_quality_report={"source_download_complete": True, "reportability_errors": []},
        )

        self.assertTrue(bool(payload["decision_action_valid_mask"][0, 1].item()))
        self.assertTrue(bool(payload["action_valid_mask"][0, 1].item()))
        self.assertFalse(bool(payload["label_valid_mask"][0, 1].item()))
        self.assertTrue(torch.isnan(payload["action_returns"][0, 1]))
        self.assertIn("decision_action_valid_mask", payload["model_input_keys"])
        self.assertNotIn("label_valid_mask", payload["model_input_keys"])
        self.assertIn("label_valid_mask", payload["forbidden_model_input_keys"])

    def test_stock_covariate_financial_date_only_is_next_session_available(self) -> None:
        record = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="financials",
            payload={
                "filing_date": "2026-01-05",
                "end_date": "2025-12-31",
                "fiscal_year": "2025",
                "fiscal_period": "Q4",
                "financials": {
                    "income_statement": {
                        "revenues": {"value": 100.0},
                        "net_income_loss": {"value": 10.0},
                    },
                    "balance_sheet": {
                        "assets": {"value": 200.0},
                        "liabilities": {"value": 50.0},
                        "cash_and_cash_equivalents": {"value": 20.0},
                    },
                    "cash_flow_statement": {
                        "net_cash_flow_from_operating_activities": {"value": 15.0},
                    },
                },
            },
        )
        rows = build_symbol_silver_rows([record])
        before = iso_to_timestamp_ms("2026-01-05T20:00:00+00:00")
        after = iso_to_timestamp_ms("2026-01-06T15:00:00+00:00")
        expected_available = regular_session_open_ms_after_date("2026-01-05")

        bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": rows},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[before, after],
            source_coverage_by_symbol={"QQQ": {"financials": True}},
            source_manifest_hash="manifest-hash",
        )
        missing_idx = ACTION_COVARIATE_FEATURE_NAMES.index("financial_missing_flag")
        margin_idx = ACTION_COVARIATE_FEATURE_NAMES.index("net_income_margin")

        self.assertEqual(record.available_timestamp_ms, expected_available)
        self.assertEqual(float(bundle["action_covariates"][0, 1, missing_idx].item()), 1.0)
        self.assertFalse(bool(bundle["action_covariate_mask"][0, 1, margin_idx].item()))
        self.assertEqual(float(bundle["action_covariates"][1, 1, missing_idx].item()), 0.0)
        self.assertTrue(bool(bundle["action_covariate_mask"][1, 1, margin_idx].item()))
        self.assertAlmostEqual(float(bundle["action_covariates"][1, 1, margin_idx].item()), 0.1, places=6)

    def test_stock_covariate_news_after_decision_is_not_counted(self) -> None:
        record = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="news",
            payload={
                "id": "n1",
                "published_utc": "2026-01-05T15:00:01Z",
                "publisher": {"name": "Wire"},
                "tickers": ["QQQ"],
            },
        )
        rows = build_symbol_silver_rows([record])
        before = iso_to_timestamp_ms("2026-01-05T15:00:00+00:00")
        after = iso_to_timestamp_ms("2026-01-05T15:01:00+00:00")
        bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": rows},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[before, after],
            source_coverage_by_symbol={"QQQ": {"news": True}},
            source_manifest_hash="manifest-hash",
        )
        news_1h_idx = ACTION_COVARIATE_FEATURE_NAMES.index("log1p_news_count_1h")

        self.assertEqual(float(bundle["action_covariates"][0, 1, news_1h_idx].item()), 0.0)
        self.assertAlmostEqual(float(bundle["action_covariates"][1, 1, news_1h_idx].item()), math.log1p(1.0), places=6)

    def test_stock_covariate_news_dedupes_and_weights_multi_ticker_articles(self) -> None:
        first = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="news",
            payload={
                "id": "same-article",
                "published_utc": "2026-01-05T14:30:00Z",
                "publisher": {"name": "Wire"},
                "tickers": ["QQQ", "AAPL", "MSFT", "NVDA"],
            },
            line_number=1,
        )
        duplicate = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="news",
            payload={
                "id": "same-article",
                "published_utc": "2026-01-05T14:31:00Z",
                "publisher": {"name": "Wire"},
                "tickers": ["QQQ", "AAPL", "MSFT", "NVDA"],
            },
            line_number=2,
        )
        single = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="news",
            payload={
                "id": "single-article",
                "published_utc": "2026-01-05T14:45:00Z",
                "publisher": {"name": "Other"},
                "tickers": ["QQQ"],
            },
            line_number=3,
        )
        rows = build_symbol_silver_rows([first, duplicate, single])
        decision = iso_to_timestamp_ms("2026-01-05T15:00:00+00:00")
        bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": rows},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[decision],
            source_coverage_by_symbol={"QQQ": {"news": True}},
            source_manifest_hash="manifest-hash",
        )
        count_idx = ACTION_COVARIATE_FEATURE_NAMES.index("log1p_news_count_1d")
        weighted_idx = ACTION_COVARIATE_FEATURE_NAMES.index("log1p_weighted_news_count_1d")
        multi_idx = ACTION_COVARIATE_FEATURE_NAMES.index("multi_ticker_news_fraction_1d")
        publisher_idx = ACTION_COVARIATE_FEATURE_NAMES.index("news_publisher_count_1d")

        self.assertAlmostEqual(float(bundle["action_covariates"][0, 1, count_idx].item()), math.log1p(2.0), places=6)
        self.assertAlmostEqual(
            float(bundle["action_covariates"][0, 1, weighted_idx].item()),
            math.log1p(1.25),
            places=6,
        )
        self.assertAlmostEqual(float(bundle["action_covariates"][0, 1, multi_idx].item()), 0.5, places=6)
        self.assertEqual(float(bundle["action_covariates"][0, 1, publisher_idx].item()), 2.0)

    def test_news_llm_article_table_dedupes_articles_across_symbol_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            (raw_root / "QQQ").mkdir()
            (raw_root / "AAPL").mkdir()
            payload = {
                "id": "article-1",
                "published_utc": "2026-01-05T15:00:00Z",
                "publisher": {"name": "Wire"},
                "title": "QQQ and AAPL rally after earnings beat",
                "description": "The companies raised guidance.",
                "tickers": ["QQQ", "AAPL"],
            }
            (raw_root / "QQQ" / "news.jsonl").write_text(json.dumps(payload) + "\n")
            duplicate = dict(payload)
            duplicate["description"] = "Duplicate row from another symbol directory."
            (raw_root / "AAPL" / "news.jsonl").write_text(json.dumps(duplicate) + "\n")

            rows, errors = build_news_article_rows(raw_root=raw_root, symbols=["QQQ", "AAPL"])

        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0]["tickers_json"]), ["AAPL", "QQQ"])
        self.assertEqual(json.loads(rows[0]["source_symbols_json"]), ["AAPL", "QQQ"])
        self.assertEqual(rows[0]["source_record_count"], 2)

    def test_news_llm_deterministic_features_are_article_ticker_and_audited(self) -> None:
        article = {
            "article_id": "article-1",
            "published_utc": "2026-01-05T15:00:00+00:00",
            "published_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "source_available_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "title": "QQQ beats earnings and raises guidance",
            "description": "Analysts upgrade the stock after strong profit growth.",
            "tickers_json": json.dumps(["QQQ", "AAPL"]),
            "primary_ticker": "QQQ",
        }

        rows = build_deterministic_news_llm_rows(
            [article],
            model_available_timestamp_ms=0,
            vendor_latency_seconds=0,
            processing_latency_seconds=0,
        )

        self.assertEqual([row["ticker"] for row in rows], ["AAPL", "QQQ"])
        qqq = next(row for row in rows if row["ticker"] == "QQQ")
        self.assertEqual(qqq["llm_schema_version"], NEWS_LLM_EXTRACT_SCHEMA_VERSION)
        self.assertEqual(qqq["llm_schema_hash"], NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH)
        self.assertTrue(qqq["extractor_no_retrieval"])
        self.assertEqual(float(qqq["extractor_temperature"]), 0.0)
        self.assertGreater(float(qqq["positive_score"]), float(qqq["negative_score"]))
        self.assertEqual(float(qqq["event_earnings"]), 1.0)
        self.assertEqual(float(qqq["event_guidance"]), 1.0)
        self.assertEqual(float(qqq["event_analyst_rating"]), 1.0)
        self.assertEqual(float(qqq["is_primary_ticker"]), 1.0)

    def test_news_llm_builder_restricts_to_allowed_universe_when_requested(self) -> None:
        article = {
            "article_id": "article-1",
            "published_utc": "2026-01-05T15:00:00+00:00",
            "published_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "source_available_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "title": "QQQ and AAPL beat earnings",
            "description": "Both tickers rallied.",
            "tickers_json": json.dumps(["QQQ", "AAPL"]),
            "primary_ticker": "QQQ",
        }

        rows = build_deterministic_news_llm_rows(
            [article],
            model_available_timestamp_ms=0,
            allowed_tickers={"QQQ"},
        )

        self.assertEqual([row["ticker"] for row in rows], ["QQQ"])

    def test_news_llm_feature_manifest_records_current_local_model_stack(self) -> None:
        if importlib.util.find_spec("pandas") is None:
            self.skipTest("pandas is required for Parquet news LLM feature output tests")
        if importlib.util.find_spec("pyarrow") is None:
            self.skipTest("pyarrow is required for Parquet news LLM feature output tests")
        article = {
            "article_id": "article-1",
            "published_utc": "2026-01-05T15:00:00+00:00",
            "published_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "source_available_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "title": "QQQ beats earnings",
            "description": "The company reports stronger profits.",
            "tickers_json": json.dumps(["QQQ"]),
            "primary_ticker": "QQQ",
        }
        rows = build_deterministic_news_llm_rows(
            [article],
            model_id="Qwen/Qwen3-1.7B",
            model_available_timestamp_ms=iso_to_timestamp_ms("2026-06-15T00:00:00+00:00"),
            model_training_cutoff_utc="unknown_for_downloaded_pretrained_model",
            vendor_latency_seconds=0,
            processing_latency_seconds=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_news_llm_feature_outputs(
                rows=rows,
                output_root=Path(directory),
                article_manifest={"article_table_hash": "article-hash"},
                model_id="Qwen/Qwen3-1.7B",
                model_available_timestamp_ms=iso_to_timestamp_ms("2026-06-15T00:00:00+00:00"),
                model_training_cutoff_utc="unknown_for_downloaded_pretrained_model",
                provider="local_transformers",
            )

        self.assertEqual(manifest["llm_feature_group"], "stock_news_llm_v1")
        self.assertEqual(manifest["primary_model_id"], "Qwen/Qwen3-1.7B")
        self.assertEqual(manifest["secondary_model_id"], "google/gemma-4-26B-A4B-it")
        self.assertEqual(manifest["fallback_model_id"], "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
        self.assertEqual(manifest["serving_engine"], "local_transformers")
        self.assertEqual(manifest["structured_output"], "prompted_json_posthoc_extract_clamp_validate")
        self.assertEqual(manifest["temperature"], 0.0)
        self.assertEqual(manifest["top_p"], 1.0)
        self.assertTrue(manifest["no_external_retrieval"])
        self.assertTrue(manifest["cached_outputs_only"])
        self.assertFalse(
            manifest["llm_analyst_model_policy"]["retrospective_historical_policy"][
                "reportable_for_2023_to_2026_backtest"
            ]
        )

    def test_news_llm_precomputed_import_fills_schema_defaults_and_filters_universe(self) -> None:
        module = load_script("build_news_llm_features")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "precomputed.jsonl"
            base = {
                "article_id": "article-1",
                "published_utc": "2026-01-05T15:00:00+00:00",
                "published_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
                "source_available_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
                "llm_feature_available_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
                "ticker_relevance": 1.0,
                "is_primary_ticker": 1.0,
                "company_specificity": 1.0,
                "is_broad_market_or_sector": 0.0,
                "sentiment_score": 0.5,
                "positive_score": 0.5,
                "negative_score": 0.0,
                "neutral_score": 0.5,
                "uncertainty_score": 0.0,
                "materiality_score": 0.5,
                "novelty_score": 1.0,
                "time_horizon": "intraday",
                "event_earnings": 1.0,
                "event_guidance": 0.0,
                "event_product": 0.0,
                "event_ai_or_technology": 0.0,
                "event_analyst_rating": 0.0,
                "event_mna": 0.0,
                "event_regulatory": 0.0,
                "event_litigation": 0.0,
                "event_macro": 0.0,
                "event_sector": 0.0,
                "event_management": 0.0,
                "event_capital_return": 0.0,
                "confidence": 0.8,
                "llm_valid": True,
                "llm_model_id": "local-test-model",
                "llm_prompt_hash": "prompt-hash",
                "extractor_provider": "local_transformers",
                "extractor_temperature": 0.0,
                "extractor_no_retrieval": True,
                "model_available_timestamp_ms": 0,
                "model_training_cutoff_utc": "unknown",
                "article_weight": 1.0,
                "ticker_count": 1.0,
            }
            external = dict(base)
            external["ticker"] = "AAPL"
            included = dict(base)
            included["ticker"] = "QQQ"
            path.write_text(json.dumps(external) + "\n" + json.dumps(included) + "\n")

            rows, errors = module.read_precomputed_rows(path, allowed_tickers={"QQQ"})

        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "QQQ")
        self.assertEqual(rows[0]["llm_schema_version"], NEWS_LLM_EXTRACT_SCHEMA_VERSION)
        self.assertEqual(rows[0]["llm_schema_hash"], NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH)

    def test_news_llm_feature_script_exposes_qwen17_default_and_local_presets(self) -> None:
        module = load_script("build_news_llm_features")

        args = module.parse_args([])
        default_manifest_path = module.local_model_manifest_path(args)
        policy = module.analyst_model_policy_from_args(args)

        self.assertEqual(args.local_model_preset, "qwen3_1_7b")
        self.assertFalse(default_manifest_path.is_absolute())
        self.assertEqual(default_manifest_path.name, "download_manifest.json")
        self.assertEqual(default_manifest_path.parent.name, "Qwen3-1.7B")
        self.assertEqual(policy["primary_model_id"], "Qwen/Qwen3-1.7B")
        self.assertEqual(policy["secondary_model_id"], "google/gemma-4-26B-A4B-it")
        self.assertEqual(policy["fallback_model_id"], "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
        self.assertEqual(policy["serving_engine"], "local_transformers")
        self.assertEqual(policy["structured_output"], "prompted_json_posthoc_extract_clamp_validate")
        self.assertEqual(policy["temperature"], 0.0)
        self.assertEqual(policy["top_p"], 1.0)
        self.assertFalse(policy["retrospective_historical_policy"]["reportable_for_2023_to_2026_backtest"])
        self.assertIn("qwen3_6_27b", module.LOCAL_MODEL_PRESETS)
        self.assertIn("qwen3_6_27b_fp8", module.LOCAL_MODEL_PRESETS)
        self.assertIn("qwen3_1_7b", module.LOCAL_MODEL_PRESETS)
        self.assertIn("gemma4_26b_a4b_it", module.LOCAL_MODEL_PRESETS)
        self.assertNotIn("qwen2_5_7b", module.LOCAL_MODEL_PRESETS)
        fp8_args = module.parse_args(["--local-model-preset", "qwen3_6_27b_fp8"])
        fp8_manifest_path = module.local_model_manifest_path(fp8_args)
        self.assertEqual(fp8_manifest_path.parent.name, "Qwen3.6-27B-FP8")
        self.assertEqual(module.LOCAL_MODEL_PRESET_REPOS["qwen3_6_27b_fp8"]["repo_id"], "Qwen/Qwen3.6-27B-FP8")
        self.assertEqual(
            module.workspace_relative_path(module.PROJECT_ROOT.parent / "LLM" / "Qwen3.6-27B" / "download_manifest.json"),
            "../LLM/Qwen3.6-27B/download_manifest.json",
        )
        sanitized = module.sanitize_local_model_manifest(
            {"local_path": str(module.PROJECT_ROOT.parent / "LLM" / "Qwen3-1.7B")}
        )
        self.assertEqual(sanitized["local_path"], "../LLM/Qwen3-1.7B")

    def test_news_llm_missing_local_preset_auto_downloads_manifest(self) -> None:
        module = load_script("build_news_llm_features")
        module.LOCAL_MODEL_PRESETS["unit_fake"] = Path("data/unit_fake_llm/download_manifest.json")
        module.LOCAL_MODEL_PRESET_REPOS["unit_fake"] = {
            "repo_id": "unit/fake-model",
            "model_type": "unit",
            "parameter_class": "tiny",
            "intended_quanttrade_role": "unit test",
        }
        args = module.parse_args(
            [
                "--precomputed-jsonl",
                "data/examples/frozen_qwen_news_outputs.jsonl",
                "--local-model-preset",
                "unit_fake",
                "--local-model-revision",
                "abc123",
            ]
        )
        manifest_path = module.local_model_manifest_path(args)
        expected = module.resolve_project_relative_path(manifest_path)

        with mock.patch.object(module, "download_local_model_preset", return_value=expected) as download:
            resolved = module.ensure_local_model_manifest(args, manifest_path)

        self.assertEqual(resolved, expected)
        download.assert_called_once_with(args, manifest_path)

    def test_news_llm_download_manifest_is_relative_and_audited(self) -> None:
        module = load_script("build_news_llm_features")
        with tempfile.TemporaryDirectory() as directory:
            local_dir = Path(directory) / "UnitModel"
            local_dir.mkdir()
            (local_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": "123"}, "weight_map": {"a": "b.safetensors"}})
            )
            manifest = module.local_model_download_manifest(
                preset="qwen3_1_7b",
                repo_id="Qwen/Qwen3-1.7B",
                revision="abc123",
                local_dir=local_dir,
            )

        self.assertEqual(manifest["repo_id"], "Qwen/Qwen3-1.7B")
        self.assertEqual(manifest["revision"], "abc123")
        self.assertFalse(Path(str(manifest["local_path"])).is_absolute())
        self.assertEqual(manifest["model_index_metadata"]["weight_map_entries"], 1)
        self.assertEqual(manifest["model_index_metadata"]["total_size_bytes"], 123)
        self.assertIn("*.safetensors", manifest["downloaded_file_policy"]["included"])
        self.assertIn("*.bin", manifest["downloaded_file_policy"]["excluded"])

    def test_news_llm_precomputed_defaults_to_qwen17_policy_model_without_manifest(self) -> None:
        module = load_script("build_news_llm_features")

        args = module.parse_args(
            [
                "--precomputed-jsonl",
                "data/examples/frozen_qwen_news_outputs.jsonl",
                "--local-model-manifest",
                "data/unit_missing_llm/download_manifest.json",
                "--no-auto-download-local-model",
                "--model-available-timestamp-utc",
                "2026-06-15T00:00:00+00:00",
            ]
        )
        model_id, _available_ms, _training_cutoff, provider, local_manifest = module.resolve_model_metadata(args)

        self.assertEqual(model_id, "Qwen/Qwen3-1.7B")
        self.assertEqual(provider, "local_transformers")
        self.assertIsNone(local_manifest)

    def test_news_llm_aggregates_are_point_in_time(self) -> None:
        article = {
            "article_id": "article-1",
            "published_utc": "2026-01-05T15:00:00+00:00",
            "published_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "source_available_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "title": "QQQ faces lawsuit and weak outlook",
            "description": "The regulatory investigation may hurt profit.",
            "tickers_json": json.dumps(["QQQ"]),
            "primary_ticker": "QQQ",
        }
        rows = build_deterministic_news_llm_rows(
            [article],
            model_available_timestamp_ms=0,
            vendor_latency_seconds=300,
            processing_latency_seconds=60,
        )
        before = iso_to_timestamp_ms("2026-01-05T15:05:59+00:00")
        after = iso_to_timestamp_ms("2026-01-05T15:06:00+00:00")
        count_idx = NEWS_LLM_AGGREGATE_FEATURE_NAMES.index("log1p_llm_weighted_news_count_1h")
        sentiment_idx = NEWS_LLM_AGGREGATE_FEATURE_NAMES.index("llm_net_sentiment_1d")
        missing_idx = NEWS_LLM_AGGREGATE_FEATURE_NAMES.index("llm_news_missing_flag")

        before_values, before_mask, _before_available, _before_age = aggregate_news_llm_features_for_symbol(
            rows=rows,
            decision_ms=before,
            source_available=True,
        )
        after_values, after_mask, after_available, _after_age = aggregate_news_llm_features_for_symbol(
            rows=rows,
            decision_ms=after,
            source_available=True,
        )

        self.assertEqual(before_values[count_idx], 0.0)
        self.assertTrue(before_mask[count_idx])
        self.assertGreater(after_values[count_idx], 0.0)
        self.assertLess(after_values[sentiment_idx], 0.0)
        self.assertEqual(after_values[missing_idx], 0.0)
        self.assertTrue(after_mask[missing_idx])
        self.assertLessEqual(after_available[count_idx], after)

    def test_action_news_llm_tensor_distinguishes_cash_missing_and_zero_news(self) -> None:
        decision = iso_to_timestamp_ms("2026-01-05T16:00:00+00:00")
        bundle = build_action_news_llm_tensor(
            news_llm_rows_by_symbol={"QQQ": []},
            action_names=["CASH", "QQQ", "MSFT"],
            decision_timestamps_ms=[decision],
            source_symbols=["QQQ"],
            source_manifest_hash="news-llm-manifest-hash",
        )
        count_idx = NEWS_LLM_AGGREGATE_FEATURE_NAMES.index("log1p_llm_weighted_news_count_1h")
        missing_idx = NEWS_LLM_AGGREGATE_FEATURE_NAMES.index("llm_news_missing_flag")

        self.assertFalse(bool(bundle["action_news_llm_mask"][0, 0].any().item()))
        self.assertTrue(bool(bundle["action_news_llm_mask"][0, 1, count_idx].item()))
        self.assertEqual(float(bundle["action_news_llm_features"][0, 1, count_idx].item()), 0.0)
        self.assertEqual(float(bundle["action_news_llm_features"][0, 1, missing_idx].item()), 0.0)
        self.assertFalse(bool(bundle["action_news_llm_mask"][0, 2, count_idx].item()))
        self.assertEqual(float(bundle["action_news_llm_features"][0, 2, missing_idx].item()), 1.0)
        self.assertNotIn("news_llm_source_manifest_hash_missing", bundle["action_news_llm_reportability_errors"])

    def test_stock_covariate_max_age_masks_stale_overview(self) -> None:
        overview = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="overview_snapshots",
            payload={
                "asof_date": "2026-01-01",
                "ticker": "QQQ",
                "type": "CS",
                "market_cap": 100.0,
                "share_class_shares_outstanding": 10.0,
                "list_date": "2020-01-01",
                "record_available": True,
            },
        )
        rows = build_symbol_silver_rows([overview])
        decision = iso_to_timestamp_ms("2026-01-20T15:00:00+00:00")
        bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": rows},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[decision],
            source_coverage_by_symbol={"QQQ": {"overview_snapshots": True}},
            source_manifest_hash="manifest-hash",
            max_age_days=3,
        )
        market_cap_idx = ACTION_COVARIATE_FEATURE_NAMES.index("log_market_cap")
        missing_idx = ACTION_COVARIATE_FEATURE_NAMES.index("overview_missing_flag")

        self.assertFalse(bool(bundle["action_covariate_mask"][0, 1, market_cap_idx].item()))
        self.assertEqual(float(bundle["action_covariates"][0, 1, missing_idx].item()), 1.0)

    def test_stock_covariate_future_dividend_and_split_are_not_trailing_inputs(self) -> None:
        dividend = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="dividends",
            payload={
                "id": "d1",
                "cash_amount": 1.0,
                "declaration_date": "2026-01-02",
                "ex_dividend_date": "2026-01-20",
            },
        )
        split = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="splits",
            payload={
                "id": "s1",
                "execution_date": "2026-01-20",
                "split_from": 1,
                "split_to": 2,
            },
        )
        rows = build_symbol_silver_rows([dividend, split])
        decision = iso_to_timestamp_ms("2026-01-06T15:00:00+00:00")
        bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": rows},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[decision],
            source_coverage_by_symbol={"QQQ": {"dividends": True, "splits": True}},
            source_manifest_hash="manifest-hash",
        )
        div_count_idx = ACTION_COVARIATE_FEATURE_NAMES.index("trailing_12m_dividend_count")
        split_count_idx = ACTION_COVARIATE_FEATURE_NAMES.index("split_events_last_365d")

        self.assertEqual(float(bundle["action_covariates"][0, 1, div_count_idx].item()), 0.0)
        self.assertEqual(float(bundle["action_covariates"][0, 1, split_count_idx].item()), 0.0)
        self.assertTrue(bool(bundle["action_covariate_mask"][0, 1, div_count_idx].item()))
        self.assertTrue(bool(bundle["action_covariate_mask"][0, 1, split_count_idx].item()))

    def test_stock_covariate_missing_news_source_differs_from_zero_news(self) -> None:
        decision = iso_to_timestamp_ms("2026-01-05T15:00:00+00:00")
        complete_bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": []},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[decision],
            source_coverage_by_symbol={"QQQ": {"news": True}},
            source_manifest_hash="manifest-hash",
        )
        missing_bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": []},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[decision],
            source_coverage_by_symbol={"QQQ": {"news": False}},
            source_manifest_hash="manifest-hash",
        )
        news_1d_idx = ACTION_COVARIATE_FEATURE_NAMES.index("log1p_news_count_1d")
        news_missing_idx = ACTION_COVARIATE_FEATURE_NAMES.index("news_missing_flag")

        self.assertTrue(bool(complete_bundle["action_covariate_mask"][0, 1, news_1d_idx].item()))
        self.assertEqual(float(complete_bundle["action_covariates"][0, 1, news_missing_idx].item()), 0.0)
        self.assertFalse(bool(missing_bundle["action_covariate_mask"][0, 1, news_1d_idx].item()))
        self.assertEqual(float(missing_bundle["action_covariates"][0, 1, news_missing_idx].item()), 1.0)

    def test_stock_covariate_source_coverage_applies_to_empty_symbol_rows(self) -> None:
        decision = iso_to_timestamp_ms("2026-01-05T15:00:00+00:00")
        bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[decision],
            source_coverage_by_symbol={"QQQ": {"news": True}},
            source_manifest_hash="manifest-hash",
        )
        news_1d_idx = ACTION_COVARIATE_FEATURE_NAMES.index("log1p_news_count_1d")
        news_missing_idx = ACTION_COVARIATE_FEATURE_NAMES.index("news_missing_flag")

        self.assertTrue(bool(bundle["action_covariate_mask"][0, 1, news_1d_idx].item()))
        self.assertEqual(float(bundle["action_covariates"][0, 1, news_missing_idx].item()), 0.0)
        self.assertEqual(bundle["action_source_coverage_names"], ["overview_snapshots", "financials", "dividends", "splits", "news"])
        self.assertTrue(bool(bundle["action_source_coverage"][0, 1, bundle["action_source_coverage_names"].index("news")].item()))
        self.assertFalse(bool(bundle["action_source_coverage"][0, 0].any().item()))
        self.assertEqual(bundle["action_covariate_action_type_feature_names"], ACTION_COVARIATE_ACTION_TYPE_FEATURE_NAMES)
        self.assertEqual(bundle["action_covariate_action_type_features"][0, 0].tolist(), [1.0, 0.0])
        self.assertEqual(bundle["action_covariate_action_type_features"][0, 1].tolist(), [0.0, 1.0])

    def test_stock_covariate_coverage_manifest_is_explicit_source_of_availability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(
                "symbol,path,rows,datasets_available,datasets_missing\n"
                "QQQ,QQQ.parquet,0,\"news,financials\",\"splits\"\n"
            )

            coverage = read_covariate_coverage_manifest(path)

        self.assertTrue(coverage["QQQ"]["news"])
        self.assertTrue(coverage["QQQ"]["financials"])
        self.assertFalse(coverage["QQQ"]["splits"])
        self.assertFalse(coverage["QQQ"]["dividends"])

    def test_stock_covariate_coverage_manifest_rejects_unknown_or_overlapping_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unknown_path = Path(directory) / "unknown.csv"
            unknown_path.write_text(
                "symbol,path,rows,datasets_available,datasets_missing\n"
                "QQQ,QQQ.parquet,0,\"news,unknown_dataset\",\"splits\"\n"
            )
            overlap_path = Path(directory) / "overlap.csv"
            overlap_path.write_text(
                "symbol,path,rows,datasets_available,datasets_missing\n"
                "QQQ,QQQ.parquet,0,\"news,financials\",\"news\"\n"
            )

            with self.assertRaisesRegex(ValueError, "Unknown covariate coverage datasets"):
                read_covariate_coverage_manifest(unknown_path)
            with self.assertRaisesRegex(ValueError, "both available and missing"):
                read_covariate_coverage_manifest(overlap_path)

    def test_tensor_content_hash_tracks_tensor_values(self) -> None:
        first = torch.tensor([[1.0, float("nan")], [2.0, 3.0]], dtype=torch.float32)
        same = torch.tensor([[1.0, float("nan")], [2.0, 3.0]], dtype=torch.float32)
        changed = first.clone()
        changed[1, 1] = 4.0

        self.assertEqual(tensor_content_hash(first), tensor_content_hash(same))
        self.assertNotEqual(tensor_content_hash(first), tensor_content_hash(changed))

    def test_stock_covariate_source_coverage_is_not_inferred_from_future_rows(self) -> None:
        record = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="news",
            payload={
                "id": "future-news",
                "published_utc": "2026-01-06T15:00:00Z",
                "publisher": {"name": "Wire"},
                "tickers": ["QQQ"],
            },
        )
        rows = build_symbol_silver_rows([record])
        decision = iso_to_timestamp_ms("2026-01-05T15:00:00+00:00")
        bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": rows},
            action_names=["CASH", "QQQ"],
            decision_timestamps_ms=[decision],
            source_manifest_hash="manifest-hash",
        )
        news_1d_idx = ACTION_COVARIATE_FEATURE_NAMES.index("log1p_news_count_1d")
        news_missing_idx = ACTION_COVARIATE_FEATURE_NAMES.index("news_missing_flag")

        self.assertFalse(bool(bundle["action_covariate_mask"][0, 1, news_1d_idx].item()))
        self.assertEqual(float(bundle["action_covariates"][0, 1, news_missing_idx].item()), 1.0)
        self.assertIn("covariate_source_coverage_manifest_missing", bundle["action_covariate_reportability_errors"])

    def test_action_covariate_feature_schema_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature_schema.json"
            path.write_text(
                json.dumps(
                    {
                        "action_covariate_feature_names": ["wrong_feature"],
                        "action_covariate_schema_hash": ACTION_COVARIATE_SCHEMA_HASH,
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "feature names"):
                validate_action_covariate_feature_schema(path)

    def test_stock_covariate_source_record_ids_are_composite(self) -> None:
        first = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="dividends",
            payload={"ticker": "QQQ", "ex_dividend_date": "2026-01-02", "cash_amount": 0.1},
            line_number=1,
        )
        second = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="dividends",
            payload={"ticker": "QQQ", "ex_dividend_date": "2026-01-02", "cash_amount": 0.2},
            line_number=2,
        )

        self.assertTrue(first.source_record_id.startswith("dividends:QQQ:QQQ:"))
        self.assertNotEqual(first.source_record_id, second.source_record_id)
        self.assertNotEqual(first.source_record_hash, second.source_record_hash)

    def test_second_context_payload_accepts_appended_action_covariates_without_mask_bias(self) -> None:
        payload = self._small_second_context_payload()
        original_width = int(payload["action_features"].shape[-1])
        original_valid = payload["decision_action_valid_mask"].clone()
        overview = normalize_raw_covariate_record(
            symbol="QQQ",
            source_dataset="overview_snapshots",
            payload={
                "asof_date": "2026-06-11",
                "ticker": "QQQ",
                "type": "CS",
                "locale": "us",
                "market_cap": 100_000_000.0,
                "share_class_shares_outstanding": 1_000_000.0,
                "list_date": "1999-03-10",
                "record_available": True,
            },
        )
        rows = build_symbol_silver_rows([overview])
        bundle = build_action_covariate_tensor(
            silver_rows_by_symbol={"QQQ": rows},
            action_names=payload["action_names"],
            decision_timestamps_ms=payload["decision_timestamps_ms"],
            source_coverage_by_symbol={"QQQ": {"overview_snapshots": True}},
            source_manifest_hash="manifest-hash",
        )
        bundle["action_covariate_feature_schema_file_hash"] = "schema-file-hash"
        augmented = append_action_covariates_to_payload(payload, bundle)

        validate_second_context_payload(augmented)
        expected_width = original_width + 2 * len(ACTION_COVARIATE_FEATURE_NAMES)
        expected_width += len(ACTION_COVARIATE_ACTION_TYPE_FEATURE_NAMES)
        self.assertEqual(int(augmented["action_features"].shape[-1]), expected_width)
        self.assertTrue(torch.equal(augmented["decision_action_valid_mask"], original_valid))
        self.assertTrue(augmented["action_features_augmented_with_covariates"])
        self.assertIn("stock_covariates_v1", augmented["action_feature_groups"])
        self.assertIn("stock_covariates_v1_mask", augmented["action_feature_groups"])
        self.assertIn("stock_covariates_v1_type", augmented["action_feature_groups"])
        self.assertTrue(augmented["action_covariate_mask_appended_to_action_features"])
        self.assertEqual(augmented["dataset_manifest"]["covariate_mode"], "flat_append_baseline")
        self.assertNotEqual(augmented["payload_hash"], payload["payload_hash"])
        self.assertIn("action_covariates_tensor_hash", augmented["tensor_content_hashes"])
        self.assertEqual(
            augmented["dataset_manifest"]["tensor_content_hashes"],
            augmented["tensor_content_hashes"],
        )
        self.assertEqual(
            augmented["dataset_manifest"]["action_covariate_feature_schema_file_hash"],
            "schema-file-hash",
        )
        self.assertEqual(tuple(augmented["action_feature_available_timestamps_ms"].shape), tuple(augmented["action_features"].shape))

    def test_second_context_payload_rejects_future_action_covariate_availability(self) -> None:
        payload = self._small_second_context_payload()
        decision_ms = int(payload["decision_timestamps_ms"][0].item())
        covariates = {
            "action_covariates": torch.zeros((1, 2, 1), dtype=torch.float32),
            "action_covariate_mask": torch.tensor([[[False], [True]]]),
            "action_covariate_available_timestamps_ms": torch.tensor([[[-1], [decision_ms + 1]]]),
            "action_covariate_feature_names": ["log_market_cap"],
            "action_covariate_schema_hash": stable_json_hash(["log_market_cap"]),
        }
        payload.update(covariates)

        with self.assertRaisesRegex(ValueError, "action covariates"):
            validate_second_context_payload(payload)

    def test_second_context_payload_rejects_forbidden_future_covariate_feature_as_input(self) -> None:
        payload = self._small_second_context_payload()
        decision_ms = int(payload["decision_timestamps_ms"][0].item())
        covariates = {
            "action_covariates": torch.zeros((1, 2, 1), dtype=torch.float32),
            "action_covariate_mask": torch.ones((1, 2, 1), dtype=torch.bool),
            "action_covariate_available_timestamps_ms": torch.full((1, 2, 1), decision_ms, dtype=torch.long),
            "action_covariate_feature_names": ["future_dividend_ex_date_unannounced"],
            "action_covariate_schema_hash": stable_json_hash(["future_dividend_ex_date_unannounced"]),
            "action_features_augmented_with_covariates": True,
        }
        payload.update(covariates)

        with self.assertRaisesRegex(ValueError, "Forbidden future-only covariate"):
            validate_second_context_payload(payload)

    def test_second_context_grid_excludes_postclose_exit_by_default(self) -> None:
        decisions = regular_session_decision_grid_ms(
            start="2026-06-12T00:00:00+00:00",
            end_exclusive="2026-06-13T00:00:00+00:00",
            decision_interval="15m",
            execution_latency_ms=1_000,
        )
        allowed = regular_session_decision_grid_ms(
            start="2026-06-12T00:00:00+00:00",
            end_exclusive="2026-06-13T00:00:00+00:00",
            decision_interval="15m",
            execution_latency_ms=1_000,
            allow_post_close_exit=True,
        )

        self.assertEqual(timestamp_ms_to_iso(decisions[-1]), "2026-06-12T19:30:00+00:00")
        self.assertEqual(timestamp_ms_to_iso(allowed[-1]), "2026-06-12T19:45:00+00:00")

    def test_second_context_postclose_exit_flag_is_reportability_metadata(self) -> None:
        payload = self._small_second_context_payload(allow_post_close_exit=True)
        manifest = payload["dataset_manifest"]

        self.assertTrue(manifest["allow_post_close_exit"])
        self.assertEqual(manifest["reportability_scope"], "extended_reward_exit_allowed")
        self.assertFalse(manifest["reportable"])
        self.assertIn("post_close_reward_exit_allowed", manifest["reportability_errors"])

    def test_second_context_payload_rejects_unavailable_context(self) -> None:
        payload = self._small_second_context_payload()
        payload["market_context_available_timestamps_ms"] = payload["market_context_available_timestamps_ms"].clone()
        payload["market_context_available_timestamps_ms"][0, 0] = payload["decision_timestamps_ms"][0] + 1

        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_second_context_payload(payload)

    def test_second_context_payload_rejects_nonzero_cash_return(self) -> None:
        payload = self._small_second_context_payload()
        payload["action_returns"] = payload["action_returns"].clone()
        payload["action_returns"][0, 0] = 0.001

        with self.assertRaisesRegex(ValueError, "CASH action return"):
            validate_second_context_payload(payload)

    def test_second_context_payload_rejects_entry_before_execution_latency(self) -> None:
        payload = self._small_second_context_payload()
        payload["entry_execution_timestamps_ms"] = payload["entry_execution_timestamps_ms"].clone()
        payload["entry_execution_timestamps_ms"][0, 1] = payload["decision_timestamps_ms"][0]

        with self.assertRaisesRegex(ValueError, "execution latency"):
            validate_second_context_payload(payload)

    def test_second_context_payload_rejects_future_action_feature_availability(self) -> None:
        payload = self._small_second_context_payload()
        payload["action_features_available_timestamps_ms"] = payload["action_features_available_timestamps_ms"].clone()
        payload["action_features_available_timestamps_ms"][0, 1] = payload["decision_timestamps_ms"][0] + 1

        with self.assertRaisesRegex(ValueError, "action features"):
            validate_second_context_payload(payload)

    def test_second_context_payload_rejects_model_input_label_overlap(self) -> None:
        payload = self._small_second_context_payload()
        payload["model_input_keys"] = [*payload["model_input_keys"], "action_returns"]

        with self.assertRaisesRegex(ValueError, "model_input_keys"):
            validate_second_context_payload(payload)

    def test_second_context_save_writes_schema_and_action_metadata_sidecars(self) -> None:
        payload = self._small_second_context_payload()
        payload["split_manifest"] = {"schema_version": "split_manifest_v1", "train": {"rows": 1}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset.pt"
            save_second_context_payload(payload, output)
            schema = json.loads((output.parent / "schema.json").read_text())
            action_metadata = json.loads((output.parent / "action_metadata.json").read_text())
            split_manifest = json.loads((output.parent / "split_manifest.json").read_text())
            saved_payload = torch.load(output, map_location="cpu", weights_only=True)

        self.assertEqual(schema["decision_tensor_protocol_version"], "1.0.0")
        self.assertEqual(schema["split_manifest_hash"], stable_json_hash(payload["split_manifest"]))
        self.assertEqual(saved_payload["dataset_manifest"]["split_manifest_hash"], stable_json_hash(payload["split_manifest"]))
        self.assertEqual(len(action_metadata["actions"]), len(payload["action_names"]))
        self.assertEqual(split_manifest["schema_version"], "split_manifest_v1")

    def test_second_context_trainer_split_manifest_uses_validation_key(self) -> None:
        module = load_script("train_second_context_action_scorer")
        split_type = type("Split", (), {})
        train = split_type()
        train.name = "train"
        train.decision_timestamps = ["2026-06-12T14:30:00+00:00"]
        train.next_timestamps = ["2026-06-12T14:45:00+00:00"]
        train.valid_start_indices = torch.tensor([0], dtype=torch.long)
        train.segment_ids = torch.tensor([0], dtype=torch.long)
        val = split_type()
        val.name = "val"
        val.decision_timestamps = ["2026-06-12T15:00:00+00:00"]
        val.next_timestamps = ["2026-06-12T15:15:00+00:00"]
        val.valid_start_indices = torch.tensor([0], dtype=torch.long)
        val.segment_ids = torch.tensor([1], dtype=torch.long)
        test = split_type()
        test.name = "test"
        test.decision_timestamps = ["2026-06-12T15:30:00+00:00"]
        test.next_timestamps = ["2026-06-12T15:45:00+00:00"]
        test.valid_start_indices = torch.tensor([0], dtype=torch.long)
        test.segment_ids = torch.tensor([2], dtype=torch.long)

        manifest = module.split_manifest_for(train, val, test)

        self.assertIn("validation", manifest)
        self.assertNotIn("val", manifest)

    def test_second_context_trainer_batch_plan_defaults_eval_to_micro_batch(self) -> None:
        module = load_script("train_second_context_action_scorer")

        plan = module.resolve_batch_plan(
            batch_size=10,
            micro_batch_size=4,
            eval_batch_size=None,
            checkpoint_every_epochs=5,
            log_every_epochs=None,
        )

        self.assertEqual(plan["batch_size"], 10)
        self.assertEqual(plan["micro_batch_size"], 4)
        self.assertEqual(plan["gradient_accumulation_steps"], 3)
        self.assertEqual(plan["eval_batch_size"], 4)
        self.assertEqual(plan["checkpoint_every_epochs"], 5)

    def test_second_context_epoch_accumulates_micro_batches_for_small_vram(self) -> None:
        module = load_script("train_second_context_action_scorer")
        split = self._second_context_split(
            returns=[
                [0.0, 0.001],
                [0.0, -0.001],
                [0.0, 0.002],
                [0.0, -0.002],
                [0.0, 0.0015],
            ]
        )
        model = SecondContextTransformerQNetwork(
            market_feature_dim=split.market_context.shape[-1],
            action_feature_dim=split.action_features.shape[-1],
            portfolio_state_dim=split.portfolio_state.shape[-1],
            constraint_state_dim=split.constraint_state.shape[-1],
            d_model=8,
            n_heads=2,
            temporal_layers=1,
            feedforward_dim=16,
            dropout=0.0,
            max_lookback_blocks=split.market_context.shape[1],
            action_count=len(split.action_names),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        stats = module.train_second_context_epoch(
            split,
            model,
            optimizer,
            scaler,
            device=torch.device("cpu"),
            batch_size=4,
            micro_batch_size=2,
            reward_scale=10_000.0,
            use_amp=False,
            grad_clip=1.0,
            pin_memory=False,
        )

        self.assertEqual(stats["optimizer_steps"], 2)
        self.assertEqual(stats["micro_batches"], 3)
        self.assertEqual(stats["rows_seen"], 5)
        self.assertEqual(stats["valid_targets_seen"], 10)
        self.assertTrue(math.isfinite(float(stats["average_loss"])))

    def test_second_context_transformer_scores_variable_action_features(self) -> None:
        payload = self._small_second_context_payload()
        model = SecondContextTransformerQNetwork(
            market_feature_dim=payload["market_context"].shape[-1],
            action_feature_dim=payload["action_features"].shape[-1],
            portfolio_state_dim=payload["portfolio_state"].shape[-1],
            constraint_state_dim=payload["constraint_state"].shape[-1],
            d_model=16,
            n_heads=4,
            temporal_layers=1,
            feedforward_dim=32,
            max_lookback_blocks=payload["market_context"].shape[1],
        )

        q_values = model(
            payload["market_context"],
            payload["market_context_mask"],
            payload["action_features"],
            payload["portfolio_state"],
            payload["constraint_state"],
        )

        self.assertEqual(tuple(q_values.shape), (1, 2))

    def test_second_context_loss_ignores_nan_invalid_returns(self) -> None:
        loss = masked_contextual_q_loss(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[0.0, float("nan")]]),
            torch.tensor([[True, False]]),
            action_cost_bps=torch.zeros((1, 2)),
        )

        self.assertTrue(bool(torch.isfinite(loss).item()))

    def test_action_confidence_outputs_all_action_tensor_and_masks_invalid_actions(self) -> None:
        config = ActionConfidenceConfig(
            hurdle_bps=1.0,
            interval_alpha=0.05,
            min_calibration_rows=10,
            q_value_scale=10_000.0,
            p_best_draws=128,
        )
        q_values = torch.tensor(
            [
                [0.0, 100.0, 50.0],
                [0.0, 200.0, -100.0],
                [0.0, 150.0, 75.0],
            ],
            dtype=torch.float32,
        )
        realized = torch.tensor(
            [
                [0.0, 0.012, float("nan")],
                [0.0, 0.018, -0.008],
                [0.0, 0.014, 0.006],
            ],
            dtype=torch.float32,
        )
        valid = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])

        calibrator = ActionConfidenceCalibrator(config).fit(q_values, realized, valid)
        output = calibrator.predict(q_values, valid)

        self.assertEqual(tuple(output.as_tensor().shape), (3, 3, len(ACTION_CONFIDENCE_FIELD_NAMES)))
        self.assertTrue(torch.isnan(output.p_positive[0, 2]))
        self.assertEqual(float(output.p_best[0, 2].item()), 0.0)
        self.assertAlmostEqual(float(output.p_best[0].sum().item()), 1.0, places=6)
        self.assertIn("calibration_rows_below_minimum", calibrator.warnings[0])

    def test_action_confidence_distinguishes_member_vote_from_draw_probability(self) -> None:
        config = ActionConfidenceConfig(
            min_calibration_rows=1,
            q_value_scale=10_000.0,
            p_best_draws=512,
            p_best_draw_seed=3,
        )
        q_values = torch.zeros((4, 2), dtype=torch.float32)
        realized = torch.tensor([[0.0, 0.001], [0.001, 0.0], [0.0, -0.001], [-0.001, 0.0]], dtype=torch.float32)
        valid = torch.ones((4, 2), dtype=torch.bool)

        calibrator = ActionConfidenceCalibrator(config).fit(q_values, realized, valid)
        output = calibrator.predict(q_values, valid)
        manifest = calibrator.manifest(split_name="test", ensemble_size=1, calibration_split="val")

        self.assertEqual(float(output.p_best_member_vote[0, 0].item()), 1.0)
        self.assertEqual(float(output.p_best_member_vote[0, 1].item()), 0.0)
        self.assertGreater(float(output.p_best_draw[0, 1].item()), 0.25)
        self.assertLess(float(output.p_best_draw[0, 1].item()), 0.75)
        self.assertTrue(torch.allclose(output.p_best, output.p_best_draw))
        self.assertIn("p_best_member_vote_is_argmax_indicator_with_single_member", manifest["warnings"])

    def test_action_confidence_manifest_marks_reused_calibration_split_diagnostic(self) -> None:
        calibrator = ActionConfidenceCalibrator(
            ActionConfidenceConfig(min_calibration_rows=1, interval_alpha=0.10, p_best_draws=64)
        )
        q_values = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        realized = torch.tensor([[0.0, 0.001], [0.0, 0.002]], dtype=torch.float32)
        valid = torch.ones((2, 2), dtype=torch.bool)
        calibrator.fit(q_values, realized, valid)
        manifest = calibrator.manifest(
            split_name="all",
            ensemble_size=1,
            calibration_split="val",
            uses_checkpoint_selection_for_calibration=True,
        )

        self.assertFalse(manifest["confidence_reportable"])
        self.assertIn(
            "calibration_split_reused_for_checkpoint_selection",
            manifest["confidence_reportability_errors"],
        )
        self.assertEqual(manifest["interval_quantiles"]["lower_quantile"], 0.10)
        self.assertEqual(manifest["interval_quantiles"]["upper_quantile"], 0.90)
        self.assertAlmostEqual(manifest["interval_quantiles"]["central_interval_coverage_target"], 0.80)

    def test_action_confidence_ood_penalty_reduces_selection_confidence(self) -> None:
        config = ActionConfidenceConfig(min_calibration_rows=1, q_value_scale=10_000.0, p_best_draws=128, ood_lambda=1.0)
        q_values = torch.zeros((2, 2), dtype=torch.float32)
        realized = torch.tensor([[0.0, 0.001], [0.0, 0.001]], dtype=torch.float32)
        valid = torch.ones((2, 2), dtype=torch.bool)
        calibrator = ActionConfidenceCalibrator(config).fit(q_values, realized, valid)
        output = calibrator.predict(q_values, valid, ood_score=torch.tensor([0.0, 2.0]))
        manifest = calibrator.manifest(split_name="test", ensemble_size=1, calibration_split="val")

        self.assertLess(float(output.selection_confidence[1, 0].item()), float(output.selection_confidence[0, 0].item()))
        self.assertEqual(manifest["ood_method"], "external_score")
        self.assertTrue(manifest["ood_penalty_active"])

    def test_action_confidence_saves_compressed_npz_and_manifest(self) -> None:
        config = ActionConfidenceConfig(hurdle_bps=1.0, min_calibration_rows=1, q_value_scale=10_000.0)
        q_values = torch.tensor([[0.0, 100.0], [0.0, 50.0]], dtype=torch.float32)
        realized = torch.tensor([[0.0, 0.011], [0.0, 0.004]], dtype=torch.float32)
        valid = torch.ones((2, 2), dtype=torch.bool)
        calibrator = ActionConfidenceCalibrator(config).fit(q_values, realized, valid)
        output = calibrator.predict(q_values, valid)
        manifest = calibrator.manifest(
            split_name="test",
            ensemble_size=1,
            calibration_split="val",
            uses_checkpoint_selection_for_calibration=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_confidence_test.npz"
            save_action_confidence_npz(
                path,
                output,
                row_indices=torch.tensor([0, 1], dtype=torch.long),
                decision_timestamps=["2026-06-12T14:30:00+00:00", "2026-06-12T14:35:00+00:00"],
                action_names=["CASH", "QQQ"],
                manifest=manifest,
            )
            import numpy as np

            loaded = np.load(path)
            saved_manifest = json.loads(path.with_suffix(".json").read_text())

        self.assertEqual(tuple(loaded["confidence_tensor"].shape), (2, 2, len(ACTION_CONFIDENCE_FIELD_NAMES)))
        self.assertEqual(saved_manifest["schema_version"], "action_confidence_v2")
        self.assertTrue(saved_manifest["uses_checkpoint_selection_for_calibration"])

    def test_action_confidence_rejects_shape_mismatch(self) -> None:
        calibrator = ActionConfidenceCalibrator(ActionConfidenceConfig(min_calibration_rows=1))

        with self.assertRaisesRegex(ValueError, "realized_returns"):
            calibrator.fit(
                torch.zeros((2, 2), dtype=torch.float32),
                torch.zeros((2, 3), dtype=torch.float32),
                torch.ones((2, 2), dtype=torch.bool),
            )

    def test_action_confidence_save_rejects_metadata_shape_mismatch(self) -> None:
        config = ActionConfidenceConfig(hurdle_bps=1.0, min_calibration_rows=1, q_value_scale=10_000.0)
        q_values = torch.tensor([[0.0, 100.0], [0.0, 50.0]], dtype=torch.float32)
        realized = torch.tensor([[0.0, 0.011], [0.0, 0.004]], dtype=torch.float32)
        valid = torch.ones((2, 2), dtype=torch.bool)
        calibrator = ActionConfidenceCalibrator(config).fit(q_values, realized, valid)
        output = calibrator.predict(q_values, valid)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "decision_timestamps"):
                save_action_confidence_npz(
                    Path(directory) / "bad.npz",
                    output,
                    row_indices=torch.tensor([0, 1], dtype=torch.long),
                    decision_timestamps=["2026-06-12T14:30:00+00:00"],
                    action_names=["CASH", "QQQ"],
                    manifest={},
                )

    def test_second_context_loss_uses_target_weights_and_weighted_costs(self) -> None:
        loss = masked_contextual_q_loss(
            torch.tensor([[0.0, 95.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.04]], dtype=torch.float32),
            torch.tensor([[True, True]]),
            action_cost_bps=torch.tensor([[0.0, 20.0]], dtype=torch.float32),
            action_target_weights=torch.tensor([[0.0, 0.25]], dtype=torch.float32),
            reward_scale=10_000.0,
        )

        self.assertLess(float(loss.item()), 1e-8)

    def test_second_context_loss_charges_signed_weight_costs_on_absolute_exposure(self) -> None:
        loss = masked_contextual_q_loss(
            torch.tensor([[0.0, -105.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.01]], dtype=torch.float32),
            torch.tensor([[True, True]]),
            action_cost_bps=torch.tensor([[0.0, 5.0]], dtype=torch.float32),
            action_target_weights=torch.tensor([[0.0, -1.0]], dtype=torch.float32),
            reward_scale=10_000.0,
        )

        self.assertLess(float(loss.item()), 1e-8)

    def test_second_context_rowwise_scorer_uses_target_weights(self) -> None:
        class WeightedModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        split = self._second_context_split(
            returns=[[0.0, 0.04]],
            costs=[[0.0, 20.0]],
            weights=[[0.0, 0.25]],
        )

        metrics = evaluate_second_context_action_scorer(split, WeightedModel(), device=torch.device("cpu"))

        self.assertAlmostEqual(float(metrics["total_return"]), 0.0095, places=6)

    def test_second_context_rowwise_scorer_uses_valid_start_indices_by_default(self) -> None:
        class WeightedModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

        split = self._second_context_split(
            returns=[[0.0, 0.50], [0.0, 0.01]],
            valid_mask=[[True, True], [True, True]],
            valid_start_indices=[1],
        )

        metrics = evaluate_second_context_action_scorer(split, WeightedModel(), device=torch.device("cpu"))

        self.assertEqual(metrics["diagnostic_rows"], "valid_start_indices")
        self.assertEqual(metrics["evaluated_rows"], 1)
        self.assertAlmostEqual(float(metrics["total_return"]), 0.01, places=6)

    def test_second_context_missing_label_report_fails_closed_for_selectable_actions(self) -> None:
        split = self._second_context_split(
            returns=[[0.0, 0.02], [0.0, 0.01]],
            valid_mask=[[True, True], [True, True]],
            label_valid_mask=[[True, False], [True, True]],
        )

        report = second_context_missing_label_report(split)

        self.assertFalse(report["evaluation_reportable"])
        self.assertEqual(report["selectable_missing_label_count"], 1)
        self.assertEqual(report["rows_with_any_selectable_missing_label"], 1)
        self.assertEqual(report["reportability_errors"], ["selectable_actions_with_missing_reward_labels"])

    def test_second_context_policy_marks_selected_missing_label_unscorable(self) -> None:
        class PickMissingLabelModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 2.0]], dtype=torch.float32)

        split = self._second_context_split(
            returns=[[0.0, 0.02]],
            valid_mask=[[True, True]],
            label_valid_mask=[[True, False]],
        )

        metrics = evaluate_second_context_trading_policy(
            split,
            PickMissingLabelModel(),
            device=torch.device("cpu"),
            return_decision_logs=True,
            return_selected_actions=True,
        )

        self.assertFalse(metrics["evaluation_reportable"])
        self.assertEqual(metrics["selected_action_missing_label_count"], 1)
        self.assertEqual(metrics["policy_unscorable_rows"], 1)
        self.assertEqual(metrics["fallback_due_to_missing_label_count"], 1)
        self.assertEqual(metrics["raw_policy_actions"], [1])
        self.assertEqual(metrics["requested_actions"], [1])
        self.assertEqual(metrics["executed_actions"], [0])
        self.assertEqual(metrics["selected_actions"], [0])
        self.assertEqual(metrics["selection_reasons"], ["fallback_due_to_missing_label"])
        self.assertEqual(metrics["decision_logs"][0]["requested_action"], "QQQ")
        self.assertEqual(metrics["decision_logs"][0]["selected_action"], "CASH")
        self.assertTrue(metrics["decision_logs"][0]["fallback_due_to_missing_label"])

    def test_second_context_rowwise_scorer_does_not_score_label_invalid_finite_values(self) -> None:
        class PickMissingLabelModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 2.0]], dtype=torch.float32)

        split = self._second_context_split(
            returns=[[0.0, 0.25]],
            valid_mask=[[True, True]],
            label_valid_mask=[[True, False]],
        )

        metrics = evaluate_second_context_action_scorer(split, PickMissingLabelModel(), device=torch.device("cpu"))

        self.assertFalse(metrics["evaluation_reportable"])
        self.assertEqual(metrics["selected_action_missing_label_count"], 1)
        self.assertEqual(metrics["rows"], 0)
        self.assertEqual(metrics["total_return"], 0.0)

    def test_second_context_sparse_market_mask_uses_last_true_context_token(self) -> None:
        class IdentityEncoder(nn.Module):
            def forward(self, x, src_key_padding_mask=None):
                del src_key_padding_mask
                return x

        class ZeroActionEncoder(nn.Module):
            def forward(self, action_features):
                return torch.zeros((*action_features.shape[:2], 1), dtype=action_features.dtype)

        class FirstStateScorer(nn.Module):
            def forward(self, pair):
                return pair[..., :1]

        model = SecondContextTransformerQNetwork(
            market_feature_dim=1,
            action_feature_dim=1,
            portfolio_state_dim=1,
            constraint_state_dim=1,
            d_model=1,
            n_heads=1,
            temporal_layers=1,
            feedforward_dim=4,
            max_lookback_blocks=3,
        )
        model.market_proj = nn.Identity()
        model.market_encoder = IdentityEncoder()
        model.portfolio_encoder = nn.Linear(1, 1, bias=False)
        model.constraint_encoder = nn.Linear(1, 1, bias=False)
        model.portfolio_encoder.weight.data.zero_()
        model.constraint_encoder.weight.data.zero_()
        model.state_norm = nn.Identity()
        model.action_encoder = ZeroActionEncoder()
        model.scorer = FirstStateScorer()

        q_values = model(
            torch.tensor([[[1.0], [999.0], [3.0]]], dtype=torch.float32),
            torch.tensor([[True, False, True]]),
            torch.zeros((1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
        )

        self.assertAlmostEqual(float(q_values[0, 0].item()), 3.0, places=6)

    def test_second_context_splits_do_not_overlap_at_boundaries(self) -> None:
        decisions = [f"2026-06-12T14:{minute:02d}:00+00:00" for minute in range(30, 60, 5)]
        next_timestamps = [*decisions[1:], "2026-06-12T15:00:00+00:00"]
        decision_ms = [iso_to_timestamp_ms(value) for value in decisions]
        next_ms = [iso_to_timestamp_ms(value) for value in next_timestamps]
        payload = {
            "schema_version": "stock_second_context_decision_v2",
            "decision_timestamps": decisions,
            "next_timestamps": next_timestamps,
            "decision_timestamps_ms": torch.tensor(decision_ms, dtype=torch.long),
            "next_timestamps_ms": torch.tensor(next_ms, dtype=torch.long),
            "market_context": torch.zeros((len(decisions), 1, 2), dtype=torch.float32),
            "market_context_mask": torch.ones((len(decisions), 1), dtype=torch.bool),
            "market_context_available_timestamps_ms": torch.tensor(
                [[value - 1] for value in decision_ms],
                dtype=torch.long,
            ),
            "action_features": torch.zeros((len(decisions), 2, 1), dtype=torch.float32),
            "action_returns": torch.zeros((len(decisions), 2), dtype=torch.float32),
            "action_valid_mask": torch.ones((len(decisions), 2), dtype=torch.bool),
            "action_cost_bps": torch.zeros((len(decisions), 2), dtype=torch.float32),
            "action_target_weights": torch.tensor([[0.0, 1.0] for _ in decisions], dtype=torch.float32),
            "entry_execution_timestamps_ms": torch.tensor(
                [[value, value] for value in decision_ms],
                dtype=torch.long,
            ),
            "exit_execution_timestamps_ms": torch.tensor(
                [[value, value] for value in next_ms],
                dtype=torch.long,
            ),
            "portfolio_state": torch.zeros((len(decisions), 1), dtype=torch.float32),
            "constraint_state": torch.zeros((len(decisions), 1), dtype=torch.float32),
            "feature_names": {
                "market_context": ["x", "y"],
                "action_features": ["a"],
                "portfolio_state": ["p"],
                "constraint_state": ["c"],
            },
            "action_names": ["CASH", "QQQ"],
            "dataset_manifest": {
                "decision_interval": "5m",
                "source_download_complete": True,
                "reportable": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            torch.save(payload, path)

            train, val, test = build_second_context_splits(
                dataset_path=path,
                train_end="2026-06-12T14:35:00+00:00",
                val_end="2026-06-12T14:45:00+00:00",
                test_start="2026-06-12T14:45:00+00:00",
            )

        self.assertEqual(train.decision_timestamps, decisions[:1])
        self.assertEqual(val.decision_timestamps, ["2026-06-12T14:40:00+00:00"])
        self.assertEqual(test.decision_timestamps, decisions[3:])
        self.assertTrue(set(train.decision_timestamps).isdisjoint(val.decision_timestamps))
        self.assertTrue(set(val.decision_timestamps).isdisjoint(test.decision_timestamps))

    def test_second_context_sequential_eval_charges_cost_only_on_switch(self) -> None:
        class ConstantActionModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

        split = SecondContextDataSplit(
            name="test",
            decision_timestamps=["2026-06-12T14:30:00+00:00", "2026-06-12T14:35:00+00:00", "2026-06-12T14:40:00+00:00"],
            next_timestamps=["2026-06-12T14:35:00+00:00", "2026-06-12T14:40:00+00:00", "2026-06-12T14:45:00+00:00"],
            action_names=["CASH", "QQQ"],
            feature_names={
                "market_context": ["x"],
                "action_features": ["action_index_scaled"],
                "portfolio_state": ["p"],
                "constraint_state": ["c"],
            },
            market_context=torch.zeros((3, 1, 1), dtype=torch.float32),
            market_context_mask=torch.ones((3, 1), dtype=torch.bool),
            market_context_available_timestamps_ms=torch.tensor([[1], [2], [3]], dtype=torch.long),
            action_features=torch.zeros((3, 2, 1), dtype=torch.float32),
            action_returns=torch.tensor([[0.0, 0.01], [0.0, 0.01], [0.0, 0.01]], dtype=torch.float32),
            action_valid_mask=torch.ones((3, 2), dtype=torch.bool),
            action_cost_bps=torch.tensor([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]], dtype=torch.float32),
            action_target_weights=torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
            entry_execution_timestamps_ms=torch.tensor([[0, 1], [0, 2], [0, 3]], dtype=torch.long),
            exit_execution_timestamps_ms=torch.tensor([[0, 2], [0, 3], [0, 4]], dtype=torch.long),
            entry_price_source="test_entry",
            exit_price_source="test_exit",
            execution_model="test_execution",
            portfolio_state=torch.zeros((3, 1), dtype=torch.float32),
            constraint_state=torch.zeros((3, 1), dtype=torch.float32),
            segment_ids=torch.zeros(3, dtype=torch.long),
            session_ids=["2026-06-12"] * 3,
            valid_start_indices=torch.tensor([0, 1, 2], dtype=torch.long),
            valid_index_mask=torch.ones(3, dtype=torch.bool),
            market_mean=torch.zeros(1, dtype=torch.float32),
            market_std=torch.ones(1, dtype=torch.float32),
            action_feature_mean=torch.zeros(1, dtype=torch.float32),
            action_feature_std=torch.ones(1, dtype=torch.float32),
            periods_per_year=252.0,
        )

        metrics = evaluate_second_context_trading_policy(
            split,
            ConstantActionModel(),
            device=torch.device("cpu"),
            return_decision_logs=True,
        )

        self.assertEqual(metrics["switches"], 1)
        self.assertAlmostEqual(float(metrics["total_return"]), (1.0 * 1.01 * 1.01) - 1.0, places=6)
        logs = metrics["decision_logs"]
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]["previous_action"], "CASH")
        self.assertEqual(logs[0]["selected_action"], "QQQ")
        self.assertEqual(logs[0]["order_legs"], 1.0)
        self.assertEqual(logs[1]["order_legs"], 0.0)
        self.assertEqual(logs[0]["entry_price_source"], "test_entry")
        self.assertIn("q_edge_vs_cash", logs[0])
        self.assertTrue(metrics["final_position_open"])

        liquidated = evaluate_second_context_trading_policy(
            split,
            ConstantActionModel(),
            device=torch.device("cpu"),
            liquidate_at_end=True,
        )

        self.assertFalse(liquidated["final_position_open"])
        self.assertAlmostEqual(float(liquidated["total_return"]), 1.0 * 1.01 * 1.01 * 0.99 - 1.0, places=6)
        self.assertAlmostEqual(float(liquidated["terminal_liquidation_cost"]), 0.01, places=6)

    def test_second_context_sequential_eval_uses_source_rows_after_skips(self) -> None:
        class ConstantActionModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

        split = self._second_context_split(
            returns=[[0.0, 0.0], [0.0, float("nan")], [0.0, 0.05]],
            valid_mask=[[True, True], [False, False], [True, True]],
            valid_start_indices=[0, 2],
        )

        metrics = evaluate_second_context_trading_policy(
            split,
            ConstantActionModel(),
            device=torch.device("cpu"),
            return_decision_logs=True,
            return_selected_actions=True,
        )

        self.assertEqual(metrics["selected_rows"], [0, 2])
        self.assertEqual([row["source_row"] for row in metrics["decision_logs"]], [0, 2])
        self.assertAlmostEqual(float(metrics["total_return"]), 0.05, places=6)

    def test_second_context_eval_resets_on_non_contiguous_valid_window(self) -> None:
        class ConstantActionModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

        split = self._second_context_split(
            returns=[[0.0, 0.01], [0.0, 0.01], [0.0, 0.01]],
            costs=[[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]],
            valid_start_indices=[0, 2],
            segment_ids=[0, 0, 0],
        )

        metrics = evaluate_second_context_trading_policy(split, ConstantActionModel(), device=torch.device("cpu"))

        self.assertEqual(metrics["gap_resets"], 1)
        self.assertEqual(metrics["switches"], 2)
        self.assertAlmostEqual(float(metrics["total_return"]), 0.0, places=6)
        # Reportability label (no P&L movement): the close-based path must NOT claim real executable trading.
        # (This synthetic split uses placeholder execution timestamps that are not point-in-time causal, so
        # it is honestly mechanically non-reportable; a real run with causal timestamps is close_based.)
        self.assertFalse(metrics["real_executable_trade_reportable"])
        self.assertIn(
            metrics["sequential_evaluation_type"],
            ("close_based_research_backtest", "non_reportable_research_diagnostic"),
        )
        self.assertIsInstance(metrics["mechanically_reportable"], bool)
        self.assertTrue(len(metrics["missing_reportability_reasons"]) > 0)

    def test_second_context_min_hold_does_not_force_initial_cash(self) -> None:
        class ConstantActionModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        split = self._second_context_split(returns=[[0.0, 0.02]])

        metrics = evaluate_second_context_trading_policy(
            split,
            ConstantActionModel(),
            device=torch.device("cpu"),
            min_hold_bars=3,
            return_selected_actions=True,
        )

        self.assertEqual(metrics["selected_actions"], [1])
        self.assertEqual(metrics["switches"], 1)

    def test_second_context_eval_resets_previous_action_on_segment_change(self) -> None:
        class ConstantActionModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

        split = SecondContextDataSplit(
            name="test",
            decision_timestamps=["2026-06-12T14:30:00+00:00", "2026-06-15T14:30:00+00:00"],
            next_timestamps=["2026-06-12T14:35:00+00:00", "2026-06-15T14:35:00+00:00"],
            action_names=["CASH", "QQQ"],
            feature_names={
                "market_context": ["x"],
                "action_features": ["action_index_scaled"],
                "portfolio_state": ["p"],
                "constraint_state": ["c"],
            },
            market_context=torch.zeros((2, 1, 1), dtype=torch.float32),
            market_context_mask=torch.ones((2, 1), dtype=torch.bool),
            market_context_available_timestamps_ms=torch.tensor([[1], [2]], dtype=torch.long),
            action_features=torch.zeros((2, 2, 1), dtype=torch.float32),
            action_returns=torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=torch.float32),
            action_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            action_cost_bps=torch.tensor([[0.0, 100.0], [0.0, 100.0]], dtype=torch.float32),
            action_target_weights=torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
            entry_execution_timestamps_ms=torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            exit_execution_timestamps_ms=torch.tensor([[0, 2], [0, 3]], dtype=torch.long),
            entry_price_source="test_entry",
            exit_price_source="test_exit",
            execution_model="test_execution",
            portfolio_state=torch.zeros((2, 1), dtype=torch.float32),
            constraint_state=torch.zeros((2, 1), dtype=torch.float32),
            segment_ids=torch.tensor([0, 1], dtype=torch.long),
            session_ids=["2026-06-12", "2026-06-15"],
            valid_start_indices=torch.tensor([0, 1], dtype=torch.long),
            valid_index_mask=torch.ones(2, dtype=torch.bool),
            market_mean=torch.zeros(1, dtype=torch.float32),
            market_std=torch.ones(1, dtype=torch.float32),
            action_feature_mean=torch.zeros(1, dtype=torch.float32),
            action_feature_std=torch.ones(1, dtype=torch.float32),
            periods_per_year=252.0,
        )

        metrics = evaluate_second_context_trading_policy(
            split,
            ConstantActionModel(),
            device=torch.device("cpu"),
        )

        self.assertEqual(metrics["segment_resets"], 1)
        self.assertEqual(metrics["switches"], 2)
        self.assertAlmostEqual(float(metrics["total_return"]), 0.0, places=6)

    def test_second_context_asset_switch_charges_both_trade_legs(self) -> None:
        class SwitchModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]], dtype=torch.float32)

        split = SecondContextDataSplit(
            name="test",
            decision_timestamps=["2026-06-12T14:30:00+00:00", "2026-06-12T14:35:00+00:00"],
            next_timestamps=["2026-06-12T14:35:00+00:00", "2026-06-12T14:40:00+00:00"],
            action_names=["CASH", "QQQ", "SPY"],
            feature_names={
                "market_context": ["x"],
                "action_features": ["action_index_scaled"],
                "portfolio_state": ["p"],
                "constraint_state": ["c"],
            },
            market_context=torch.zeros((2, 1, 1), dtype=torch.float32),
            market_context_mask=torch.ones((2, 1), dtype=torch.bool),
            market_context_available_timestamps_ms=torch.tensor([[1], [2]], dtype=torch.long),
            action_features=torch.zeros((2, 3, 1), dtype=torch.float32),
            action_returns=torch.zeros((2, 3), dtype=torch.float32),
            action_valid_mask=torch.ones((2, 3), dtype=torch.bool),
            action_cost_bps=torch.tensor([[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]], dtype=torch.float32),
            action_target_weights=torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32),
            entry_execution_timestamps_ms=torch.tensor([[0, 1, 1], [0, 2, 2]], dtype=torch.long),
            exit_execution_timestamps_ms=torch.tensor([[0, 2, 2], [0, 3, 3]], dtype=torch.long),
            entry_price_source="test_entry",
            exit_price_source="test_exit",
            execution_model="test_execution",
            portfolio_state=torch.zeros((2, 1), dtype=torch.float32),
            constraint_state=torch.zeros((2, 1), dtype=torch.float32),
            segment_ids=torch.zeros(2, dtype=torch.long),
            session_ids=["2026-06-12"] * 2,
            valid_start_indices=torch.tensor([0, 1], dtype=torch.long),
            valid_index_mask=torch.ones(2, dtype=torch.bool),
            market_mean=torch.zeros(1, dtype=torch.float32),
            market_std=torch.ones(1, dtype=torch.float32),
            action_feature_mean=torch.zeros(1, dtype=torch.float32),
            action_feature_std=torch.ones(1, dtype=torch.float32),
            periods_per_year=252.0,
        )

        metrics = evaluate_second_context_trading_policy(
            split,
            SwitchModel(),
            device=torch.device("cpu"),
            return_decision_logs=True,
        )

        self.assertEqual(metrics["switches"], 2)
        self.assertAlmostEqual(float(metrics["total_return"]), (1.0 - 0.001) * (1.0 - 0.003) - 1.0, places=6)
        logs = metrics["decision_logs"]
        self.assertEqual(logs[1]["order_legs"], 2.0)
        self.assertEqual(logs[1]["traded_notional"], 2.0)
        self.assertAlmostEqual(logs[1]["cost_bps"], 15.0)

    def test_second_context_signed_weights_pay_positive_costs(self) -> None:
        class ShortModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        split = self._second_context_split(
            returns=[[0.0, 0.0]],
            costs=[[0.0, 10.0]],
            weights=[[0.0, -1.0]],
        )

        metrics = evaluate_second_context_trading_policy(
            split,
            ShortModel(),
            device=torch.device("cpu"),
            return_decision_logs=True,
        )

        self.assertAlmostEqual(float(metrics["total_return"]), -0.001, places=6)
        log = metrics["decision_logs"][0]
        self.assertEqual(log["target_weight"], -1.0)
        self.assertEqual(log["executed_weight"], -1.0)
        self.assertAlmostEqual(log["traded_notional"], 1.0)
        self.assertAlmostEqual(log["net_return"], -0.001)

    def test_second_context_row_varying_weights_carry_executed_exposure(self) -> None:
        class HoldModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

        split = self._second_context_split(
            returns=[[0.0, 0.01], [0.0, 0.02]],
            weights=[[0.0, 1.0], [0.0, 0.5]],
        )

        metrics = evaluate_second_context_trading_policy(
            split,
            HoldModel(),
            device=torch.device("cpu"),
            return_decision_logs=True,
        )

        self.assertAlmostEqual(float(metrics["total_return"]), 1.01 * 1.02 - 1.0, places=6)
        self.assertEqual(metrics["switches"], 1)
        self.assertEqual(metrics["same_action_weight_policy"], "freeze_executed_weight_until_action_change")
        second_log = metrics["decision_logs"][1]
        self.assertEqual(second_log["same_action_weight_policy"], "freeze_executed_weight_until_action_change")
        self.assertAlmostEqual(second_log["target_weight"], 0.5)
        self.assertAlmostEqual(second_log["previous_executed_weight"], 1.0)
        self.assertAlmostEqual(second_log["executed_weight"], 1.0)
        self.assertAlmostEqual(second_log["gross_return"], 0.02)

    def test_second_context_weighted_leveraged_action_scales_return_and_cost(self) -> None:
        class LeveragedModel(nn.Module):
            def forward(self, market_context, market_context_mask, action_features, portfolio_state, constraint_state):
                del market_context, market_context_mask, action_features, portfolio_state, constraint_state
                return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        split = SecondContextDataSplit(
            name="test",
            decision_timestamps=["2026-06-12T14:30:00+00:00"],
            next_timestamps=["2026-06-12T14:45:00+00:00"],
            action_names=["CASH", "SOXL"],
            feature_names={
                "market_context": ["x"],
                "action_features": ["action_index_scaled"],
                "portfolio_state": ["p"],
                "constraint_state": ["c"],
            },
            market_context=torch.zeros((1, 1, 1), dtype=torch.float32),
            market_context_mask=torch.ones((1, 1), dtype=torch.bool),
            market_context_available_timestamps_ms=torch.tensor([[1]], dtype=torch.long),
            action_features=torch.zeros((1, 2, 1), dtype=torch.float32),
            action_returns=torch.tensor([[0.0, 0.03]], dtype=torch.float32),
            action_valid_mask=torch.ones((1, 2), dtype=torch.bool),
            action_cost_bps=torch.tensor([[0.0, 30.0]], dtype=torch.float32),
            action_target_weights=torch.tensor([[0.0, 1.0 / 3.0]], dtype=torch.float32),
            entry_execution_timestamps_ms=torch.tensor([[0, 1]], dtype=torch.long),
            exit_execution_timestamps_ms=torch.tensor([[0, 2]], dtype=torch.long),
            entry_price_source="test_entry",
            exit_price_source="test_exit",
            execution_model="test_execution",
            portfolio_state=torch.zeros((1, 1), dtype=torch.float32),
            constraint_state=torch.zeros((1, 1), dtype=torch.float32),
            segment_ids=torch.zeros(1, dtype=torch.long),
            session_ids=["2026-06-12"],
            valid_start_indices=torch.tensor([0], dtype=torch.long),
            valid_index_mask=torch.ones(1, dtype=torch.bool),
            market_mean=torch.zeros(1, dtype=torch.float32),
            market_std=torch.ones(1, dtype=torch.float32),
            action_feature_mean=torch.zeros(1, dtype=torch.float32),
            action_feature_std=torch.ones(1, dtype=torch.float32),
            periods_per_year=252.0,
        )

        metrics = evaluate_second_context_trading_policy(
            split,
            LeveragedModel(),
            device=torch.device("cpu"),
            return_decision_logs=True,
        )

        self.assertAlmostEqual(float(metrics["total_return"]), 0.009, places=6)
        log = metrics["decision_logs"][0]
        self.assertAlmostEqual(log["target_weight"], 1.0 / 3.0)
        self.assertAlmostEqual(log["traded_notional"], 1.0 / 3.0)
        self.assertAlmostEqual(log["gross_return"], 0.01, places=6)

    def test_second_context_cost_stress_replays_fixed_action_path(self) -> None:
        split = SecondContextDataSplit(
            name="test",
            decision_timestamps=["2026-06-12T14:30:00+00:00", "2026-06-12T14:35:00+00:00"],
            next_timestamps=["2026-06-12T14:35:00+00:00", "2026-06-12T14:40:00+00:00"],
            action_names=["CASH", "QQQ", "SPY"],
            feature_names={
                "market_context": ["x"],
                "action_features": ["action_index_scaled"],
                "portfolio_state": ["p"],
                "constraint_state": ["c"],
            },
            market_context=torch.zeros((2, 1, 1), dtype=torch.float32),
            market_context_mask=torch.ones((2, 1), dtype=torch.bool),
            market_context_available_timestamps_ms=torch.tensor([[1], [2]], dtype=torch.long),
            action_features=torch.zeros((2, 3, 1), dtype=torch.float32),
            action_returns=torch.tensor([[0.0, 0.01, 0.02], [0.0, 0.01, 0.02]], dtype=torch.float32),
            action_valid_mask=torch.ones((2, 3), dtype=torch.bool),
            action_cost_bps=torch.zeros((2, 3), dtype=torch.float32),
            action_target_weights=torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32),
            entry_execution_timestamps_ms=torch.tensor([[0, 1, 1], [0, 2, 2]], dtype=torch.long),
            exit_execution_timestamps_ms=torch.tensor([[0, 2, 2], [0, 3, 3]], dtype=torch.long),
            entry_price_source="test_entry",
            exit_price_source="test_exit",
            execution_model="test_execution",
            portfolio_state=torch.zeros((2, 1), dtype=torch.float32),
            constraint_state=torch.zeros((2, 1), dtype=torch.float32),
            segment_ids=torch.zeros(2, dtype=torch.long),
            session_ids=["2026-06-12"] * 2,
            valid_start_indices=torch.tensor([0, 1], dtype=torch.long),
            valid_index_mask=torch.ones(2, dtype=torch.bool),
            market_mean=torch.zeros(1, dtype=torch.float32),
            market_std=torch.ones(1, dtype=torch.float32),
            action_feature_mean=torch.zeros(1, dtype=torch.float32),
            action_feature_std=torch.ones(1, dtype=torch.float32),
            periods_per_year=252.0,
        )

        stress = fixed_rollout_cost_stress(split, torch.tensor([1, 2]), cost_bps_values=(0.0, 20.0))
        baselines = evaluate_second_context_baselines(split, reference_actions=torch.tensor([1, 2]), seed=1)

        self.assertGreater(stress["0_bps"]["total_return"], stress["20_bps"]["total_return"])
        self.assertIn("RandomSameTurnover", baselines)
        self.assertIn("RandomSameTurnoverSameTiming", baselines)
        self.assertIn("RandomSameSegments", baselines)
        self.assertIn("RandomSameActionDistribution", baselines)
        self.assertIn("CASH", baselines)

    def test_second_context_random_baselines_sample_only_valid_row_actions(self) -> None:
        split = self._second_context_split(
            action_names=["CASH", "AAA", "BBB"],
            returns=[[0.0, 0.01, float("nan")], [0.0, float("nan"), 0.02], [0.0, 0.03, float("nan")]],
            valid_mask=[[True, True, False], [True, False, True], [True, True, False]],
            valid_start_indices=[0, 1, 2],
        )

        baselines = evaluate_second_context_baselines(split, reference_actions=torch.tensor([1, 2, 1]), seed=3)

        for name, metrics in baselines.items():
            if name.startswith("Random"):
                self.assertEqual(metrics["invalid_action_attempts"], 0, name)

    def test_second_context_oracle_summary_is_diagnostic_only(self) -> None:
        module = load_script("evaluate_second_context_dataset")
        payload = self._small_second_context_payload()
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.pt"
            output = Path(directory) / "summary.json"
            torch.save(payload, dataset)
            previous_argv = sys.argv
            try:
                sys.argv = ["evaluate_second_context_dataset.py", "--dataset", str(dataset), "--output", str(output)]
                module.main()
            finally:
                sys.argv = previous_argv
            summary = json.loads(output.read_text())

        self.assertTrue(summary["diagnostic_only"])
        self.assertIn("diagnostic_oracle_best_valid_action_future_leakage", summary)
        self.assertNotIn("oracle_best_valid_action", summary)

    def test_second_context_conversion_skip_requires_current_gold_schema(self) -> None:
        module = load_script("convert_polygon_second_to_protocol")
        payload = self._small_second_context_payload()
        old_payload = dict(payload)
        for key in (
            "decision_action_valid_mask",
            "label_valid_mask",
            "entry_fill_observed_mask",
            "reward_exit_observed_mask",
        ):
            old_payload.pop(key, None)
        old_manifest = dict(old_payload.get("dataset_manifest", {}))
        old_manifest.pop("action_mask_semantics", None)
        old_payload["dataset_manifest"] = old_manifest

        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / "old" / "dataset.pt"
            current_path = Path(directory) / "current" / "dataset.pt"
            old_path.parent.mkdir(parents=True)
            current_path.parent.mkdir(parents=True)
            torch.save(old_payload, old_path)
            torch.save(payload, current_path)

            self.assertFalse(module.existing_gold_dataset_is_current(old_path))
            self.assertTrue(module.existing_gold_dataset_is_current(current_path))

    def test_second_context_conversion_skip_rejects_cache_identity_mismatch(self) -> None:
        module = load_script("convert_polygon_second_to_protocol")
        payload = self._small_second_context_payload()
        manifest = dict(payload["dataset_manifest"])
        manifest.update(
            {
                "source_manifest_hash": "source-a",
                "universe_file_hash": "universe-a",
                "conversion_config_hash": "config-a",
                "converter_identity_hash": "converter-a",
                "action_schema_hash": stable_json_hash(payload["action_names"]),
            }
        )
        payload["dataset_manifest"] = manifest
        expected = {
            "source_manifest_hash": "source-a",
            "universe_file_hash": "universe-a",
            "conversion_config_hash": "config-a",
            "converter_identity_hash": "converter-a",
            "action_schema_hash": stable_json_hash(payload["action_names"]),
        }
        mismatched = dict(expected)
        mismatched["universe_file_hash"] = "universe-b"

        with tempfile.TemporaryDirectory() as directory:
            current_path = Path(directory) / "current" / "dataset.pt"
            current_path.parent.mkdir(parents=True)
            torch.save(payload, current_path)

            self.assertTrue(module.existing_gold_dataset_is_current(current_path, expected))
            self.assertFalse(module.existing_gold_dataset_is_current(current_path, mismatched))
            self.assertEqual(
                module.validate_cache_identity(manifest, mismatched),
                ["cache_identity_mismatch:universe_file_hash"],
            )

    def test_second_context_conversion_skip_requires_current_hourly_schema(self) -> None:
        module = load_script("convert_polygon_second_to_protocol")
        current_payload = {
            "action_returns": torch.tensor([[0.0, float("nan")], [0.0, 0.01]], dtype=torch.float32),
            "decision_action_valid_mask": torch.tensor([[True, True], [True, True]]),
            "action_valid_mask": torch.tensor([[True, True], [True, True]]),
            "label_valid_mask": torch.tensor([[True, False], [True, True]]),
            "action_label_valid_mask": torch.tensor([[True, False], [True, True]]),
            "action_mask_semantics": {
                "decision_action_valid_mask": "known before decision",
                "label_valid_mask": "known after reward realization",
            },
            "model_input_keys": ["minute_features", "action_valid_mask", "decision_action_valid_mask"],
            "forbidden_model_input_keys": ["label_valid_mask", "action_label_valid_mask"],
        }
        old_payload = {
            "action_returns": torch.tensor([[0.0, float("nan")], [0.0, 0.01]], dtype=torch.float32),
            "action_valid_mask": torch.tensor([[True, False], [True, True]]),
        }
        leaky_payload = dict(current_payload)
        leaky_payload["model_input_keys"] = ["minute_features", "label_valid_mask"]

        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / "old.pt"
            current_path = Path(directory) / "current.pt"
            leaky_path = Path(directory) / "leaky.pt"
            torch.save(old_payload, old_path)
            torch.save(current_payload, current_path)
            torch.save(leaky_payload, leaky_path)

            self.assertFalse(module.existing_hourly_dataset_is_current(old_path))
            self.assertTrue(module.existing_hourly_dataset_is_current(current_path))
            self.assertFalse(module.existing_hourly_dataset_is_current(leaky_path))

    def test_second_context_conversion_stamps_hourly_cache_identity(self) -> None:
        module = load_script("convert_polygon_second_to_protocol")
        payload = {
            "action_returns": torch.tensor([[0.0, float("nan")], [0.0, 0.01]], dtype=torch.float32),
            "decision_action_valid_mask": torch.tensor([[True, True], [True, True]]),
            "action_valid_mask": torch.tensor([[True, True], [True, True]]),
            "label_valid_mask": torch.tensor([[True, False], [True, True]]),
            "action_label_valid_mask": torch.tensor([[True, False], [True, True]]),
            "action_mask_semantics": {
                "decision_action_valid_mask": "known before decision",
                "label_valid_mask": "known after reward realization",
            },
            "model_input_keys": ["minute_features", "action_valid_mask", "decision_action_valid_mask"],
            "forbidden_model_input_keys": ["label_valid_mask", "action_label_valid_mask"],
            "dataset_reportable": True,
            "dataset_reportability_errors": [],
        }
        expected = {
            "source_manifest_hash": "source-a",
            "universe_file_hash": "universe-a",
            "conversion_config_hash": "config-a",
            "converter_identity_hash": "converter-a",
            "action_schema_hash": "actions-a",
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hourly" / "dataset.pt"
            path.parent.mkdir(parents=True)
            torch.save(payload, path)

            self.assertFalse(module.existing_hourly_dataset_is_current(path, expected))
            self.assertEqual(module.stamp_cache_identity(path, expected), [])
            self.assertTrue(module.existing_hourly_dataset_is_current(path, expected))
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["dataset_manifest"]["source_manifest_hash"], "source-a")
            self.assertEqual(saved["dataset_manifest"]["reportable"], True)
            sidecar = json.loads((path.parent / "dataset_manifest.json").read_text())
            self.assertEqual(sidecar["converter_identity_hash"], "converter-a")

    def test_second_context_converter_reportability_flags_future_universe(self) -> None:
        module = load_script("convert_polygon_second_to_protocol")
        args = module.parse_args(
            [
                "--start",
                "2025-01-01",
                "--universe",
                "data/top_500_s3_volume_common_stocks_2026-06-12_tickers.txt",
            ]
        )
        errors = module.conversion_reportability_errors(
            args=args,
            source_summary={"pending_symbol_days": 0},
            status_counts=module.Counter({"ok": 1}),
            universe_asof_after_start=module.parse_date(module.infer_universe_asof(args.universe)) > module.parse_date(args.start),
            missing_action_symbols=[],
        )

        self.assertFalse(args.continue_on_error)
        self.assertFalse(args.allow_missing_action_context)
        self.assertFalse(args.allow_non_reportable)
        self.assertIn("universe_asof_after_dataset_start", errors)
        self.assertNotIn("missing_action_context_allowed", errors)

    def test_second_context_converter_keeps_intended_actions_separate_from_source_coverage(self) -> None:
        module = load_script("convert_polygon_second_to_protocol")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            universe = root / "universe_2025-01-01.txt"
            universe.write_text("AAA\nBBB\n")
            source_root = root / "source"
            (source_root / "AAA" / "2025" / "01").mkdir(parents=True)
            (source_root / "AAA" / "2025" / "01" / "2025-01-02.parquet").write_bytes(b"placeholder")
            args = module.parse_args(
                [
                    "--source-root",
                    str(source_root),
                    "--universe",
                    str(universe),
                    "--action-count",
                    "2",
                ]
            )

            actions = module.resolve_actions(args)
            missing = module.missing_action_source_symbols(args.source_root, actions)

        self.assertEqual(actions, ["CASH", "AAA", "BBB"])
        self.assertEqual(missing, ["BBB"])

    def test_second_context_converter_passes_action_covariate_flags_to_gold_builder(self) -> None:
        module = load_script("convert_polygon_second_to_protocol")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            covariates_root = root / "covariates"
            covariates_root.mkdir()
            manifest = covariates_root / "manifest.csv"
            schema = covariates_root / "feature_schema.json"
            manifest.write_text("symbol,path,rows\nAAA,AAA.parquet,1\n")
            schema.write_text('{"schema_version": "stock_covariates_silver_v2"}\n')
            expected_manifest_hash = module.file_sha256(manifest)
            expected_schema_hash = module.file_sha256(schema)
            args = module.parse_args(
                [
                    "--include-action-covariates",
                    "--covariates-root",
                    str(covariates_root),
                    "--covariate-feature-schema",
                    str(schema),
                    "--covariate-max-age-days",
                    "30",
                    "--covariate-strict-coverage",
                ]
            )

            command = module.build_gold_command(
                args=args,
                source_manifest=Path("source/manifest.csv"),
                protocol_dataset_manifest=Path("protocol.json"),
                actions=["CASH", "AAA"],
                start_day=module.parse_date("2026-01-02"),
                end_day=module.parse_date("2026-01-03"),
                output=Path("out/dataset.pt"),
            )
            config = module.conversion_config_payload(args, actions=["CASH", "AAA"], universe_asof="2026-01-01")

        self.assertIn("--include-action-covariates", command)
        self.assertIn("--covariates-root", command)
        self.assertIn("--covariate-feature-schema", command)
        self.assertIn("--covariate-strict-coverage", command)
        self.assertEqual(command[command.index("--covariate-max-age-days") + 1], "30")
        self.assertTrue(config["include_action_covariates"])
        self.assertEqual(config["covariate_max_age_days"], 30)
        self.assertEqual(config["covariate_silver_manifest_hash"], expected_manifest_hash)
        self.assertEqual(config["covariate_feature_schema_file_hash"], expected_schema_hash)

    def test_second_context_converter_passes_fixed_survivor_universe_diagnostic_flag(self) -> None:
        module = load_script("convert_polygon_second_to_protocol")
        args = module.parse_args(["--allow-fixed-survivor-universe-diagnostic"])

        command = module.build_gold_command(
            args=args,
            source_manifest=Path("source/manifest.csv"),
            protocol_dataset_manifest=Path("protocol.json"),
            actions=["CASH", "AAA"],
            start_day=module.parse_date("2026-01-02"),
            end_day=module.parse_date("2026-01-03"),
            output=Path("out/dataset.pt"),
        )
        config = module.conversion_config_payload(args, actions=["CASH", "AAA"], universe_asof="2026-01-01")

        self.assertIn("--allow-fixed-survivor-universe-diagnostic", command)
        self.assertTrue(config["allow_fixed_survivor_universe_diagnostic"])

    def test_min_hold_action_mask_allows_current_action_and_cash(self) -> None:
        mask = build_action_mask(
            current_action=torch.tensor([3]),
            bars_held=torch.tensor([0]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([0]),
            max_switches_per_day=2,
            min_hold_bars=2,
            action_count=10,
        )

        # During min-hold only the held action (3) plus the zero-risk CASH (0) fallback
        # may be selected: de-risking to cash is never blocked by hold/cooldown/caps.
        self.assertEqual(int(mask.sum().item()), 2)
        self.assertTrue(bool(mask[0, 3].item()))
        self.assertTrue(bool(mask[0, 0].item()))

    def test_cooldown_action_mask_keeps_cash_derisk_available(self) -> None:
        mask = build_action_mask(
            current_action=torch.tensor([2]),
            bars_held=torch.tensor([10]),
            cooldown_remaining=torch.tensor([3]),
            switches_today=torch.tensor([0]),
            min_hold_bars=1,
            action_count=6,
        )

        # In cooldown only the held action and CASH (de-risk) are selectable.
        self.assertEqual(int(mask.sum().item()), 2)
        self.assertTrue(bool(mask[0, 2].item()))
        self.assertTrue(bool(mask[0, 0].item()))

    def test_build_action_mask_reasons_explains_mask(self) -> None:
        # build_action_mask_reasons is a NON-INVASIVE diagnostic: its .mask field IS build_action_mask's
        # output, and reconstructing the mask from its reason tensors (plus the always-selectable current
        # action and CASH) must reproduce that mask EXACTLY. This proves the per-constraint attribution fully
        # explains the mask and cannot silently diverge from it. Randomized over rows and every cap combo.
        import random

        from rl_quant.protocol.constraints import ActionMaskResult, build_action_mask, build_action_mask_reasons

        torch.manual_seed(0)
        random.seed(0)
        action_count, cash_index = 4, 0
        for trial in range(200):
            batch = random.randint(1, 6)
            current = torch.randint(0, action_count, (batch,))
            kw = dict(
                current_action=current,
                bars_held=torch.randint(0, 5, (batch,)),
                cooldown_remaining=torch.randint(0, 3, (batch,)),
                switches_today=torch.randint(0, 4, (batch,)),
                min_hold_bars=random.choice([0, 1, 2, 3]),
                action_count=action_count, cash_index=cash_index,
                count_etf_to_etf_as_two_legs=bool(random.getrandbits(1)),
            )
            if random.getrandbits(1):
                kw["max_switches_per_day"] = random.choice([0, 1, 2])
            if random.getrandbits(1):
                kw["max_switches_per_episode"] = random.choice([0, 1, 3])
                kw["switches_episode"] = torch.randint(0, 6, (batch,))
            if random.getrandbits(1):
                kw["max_order_legs_per_day"] = random.choice([0.0, 1.0, 2.0])
                kw["order_legs_today"] = torch.randint(0, 4, (batch,)).float()
            if random.getrandbits(1):
                kw["max_order_legs_per_episode"] = random.choice([0.0, 2.0, 4.0])
                kw["order_legs_episode"] = torch.randint(0, 8, (batch,)).float()

            result = build_action_mask_reasons(**kw)
            self.assertIsInstance(result, ActionMaskResult)
            # The mask field must be byte-identical to build_action_mask (it is produced by it).
            self.assertTrue(torch.equal(result.mask, build_action_mask(**kw)))

            constrained = result.min_hold_block | result.cooldown_block | result.switch_cap_block
            recon = torch.zeros(batch, action_count, dtype=torch.bool)
            for r in range(batch):
                for a in range(action_count):
                    if a == cash_index or a == int(current[r]):
                        recon[r, a] = True            # CASH (Policy A) and the current action are always selectable
                    elif bool(constrained[r]):
                        recon[r, a] = False           # a row-level constraint pins the row to its current action
                    else:
                        recon[r, a] = not bool(result.order_leg_block[r, a])  # else only the order-leg budget blocks
            self.assertTrue(
                torch.equal(recon, result.mask),
                f"reason tensors do not reconstruct the mask (trial {trial}): {kw}",
            )

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

        # Daily switch cap exhausted: new non-cash positions are masked, but Policy A keeps the held action
        # (4) AND the CASH (0) de-risk selectable -- de-risking to cash is never blocked by a turnover budget.
        self.assertEqual(int(mask.sum().item()), 2)
        self.assertTrue(bool(mask[0, 4].item()))
        self.assertTrue(bool(mask[0, 0].item()))

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

        # Episode switch cap exhausted: the held action (2) AND the CASH (0) de-risk stay selectable under
        # Policy A (de-risking to cash overrides an exhausted turnover budget).
        self.assertEqual(int(mask.sum().item()), 2)
        self.assertTrue(bool(mask[0, 2].item()))
        self.assertTrue(bool(mask[0, 0].item()))

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

    def test_order_leg_cap_does_not_block_cash_derisk(self) -> None:
        # Policy A: even when the order-leg budget is fully exhausted (current -> CASH itself would exceed it),
        # CASH (0) de-risk stays available while every non-cash rotation is blocked, holding the current action.
        mask = build_action_mask(
            current_action=torch.tensor([2]),
            bars_held=torch.tensor([5]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([0]),
            max_switches_per_day=5,
            min_hold_bars=1,
            action_count=6,
            order_legs_today=torch.tensor([0.0]),
            max_order_legs_per_day=0.0,  # no legs affordable: current->CASH (1 leg) would exceed the budget
        )
        self.assertTrue(bool(mask[0, 0].item()))   # CASH de-risk forced valid (Policy A)
        self.assertTrue(bool(mask[0, 2].item()))   # held action always selectable
        self.assertFalse(bool(mask[0, 5].item()))  # a 2-leg ETF rotation stays blocked by the budget

    def test_etf_to_etf_switch_counts_two_legs(self) -> None:
        self.assertEqual(float(trade_legs(torch.tensor([2]), torch.tensor([5]))[0].item()), 2.0)

    def test_cash_to_etf_counts_one_leg(self) -> None:
        self.assertEqual(float(trade_legs(torch.tensor([0]), torch.tensor([5]))[0].item()), 1.0)

    def test_constraint_feature_scaling_uses_episode_cap(self) -> None:
        # With an episode switch cap, the episode-switch feature (index 3) scales by the CAP:
        # switches_episode=3 / cap=3 -> 1.0.
        capped = make_constraint_features(
            bars_held=torch.tensor([1]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([1]),
            switches_episode=torch.tensor([3]),
            constraints=TradingConstraintConfig(max_switches_per_day=4, max_switches_per_episode=3),
            episode_length=32,
        )
        self.assertAlmostEqual(float(capped[0, 3].item()), 1.0)
        # With NO episode cap it must fall back to episode_length scaling (3/32), a different and
        # much smaller value -- proving the cap (not episode_length) drives the scaling when set.
        uncapped = make_constraint_features(
            bars_held=torch.tensor([1]),
            cooldown_remaining=torch.tensor([0]),
            switches_today=torch.tensor([1]),
            switches_episode=torch.tensor([3]),
            constraints=TradingConstraintConfig(max_switches_per_day=4, max_switches_per_episode=None),
            episode_length=32,
        )
        self.assertAlmostEqual(float(uncapped[0, 3].item()), 3.0 / 32.0)
        self.assertFalse(bool(torch.allclose(capped, uncapped)))

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

            with self.assertRaisesRegex(ValueError, "context leakage"):
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
                "models/model.pt",
            ]
        )
        constraints = module.build_constraints_from_args(args)

        self.assertEqual(constraints.max_switches_per_episode, 3)
        self.assertEqual(constraints.max_order_legs_per_day, 4)
        self.assertEqual(constraints.max_order_legs_per_episode, 5)
        self.assertEqual(args.warm_start_model, Path("models/model.pt"))

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

    def test_minute_to_hour_split_rejects_finite_returns_for_invalid_actions(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_build_split"])
        payload = {
            "decision_timestamps": [
                "2026-01-02T14:30:00+00:00",
                "2026-01-02T15:30:00+00:00",
            ],
            "next_timestamps": [
                "2026-01-02T15:30:00+00:00",
                "2026-01-02T16:30:00+00:00",
            ],
            "minute_feature_names": ["m"],
            "hour_feature_names": ["h"],
            "action_names": ["CASH", "QQQ"],
            "minute_features": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            "minute_mask": torch.ones((2, 1, 1), dtype=torch.bool),
            "hour_features": torch.zeros((2, 1, 1), dtype=torch.float32),
            "action_returns": torch.zeros((2, 2), dtype=torch.float32),
            "action_valid_mask": torch.tensor([[True, False], [True, True]]),
        }

        with self.assertRaisesRegex(ValueError, "Invalid action_returns"):
            module._build_split(name="train", payload=payload)

    def test_minute_to_hour_legacy_action_valid_mask_is_diagnostic_only(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_build_split"])
        payload = {
            "decision_timestamps": [
                "2026-01-02T14:30:00+00:00",
                "2026-01-02T15:30:00+00:00",
            ],
            "next_timestamps": [
                "2026-01-02T15:30:00+00:00",
                "2026-01-02T16:30:00+00:00",
            ],
            "minute_feature_names": ["m"],
            "hour_feature_names": ["h"],
            "action_names": ["CASH", "QQQ"],
            "minute_features": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            "minute_mask": torch.ones((2, 1, 1), dtype=torch.bool),
            "hour_features": torch.zeros((2, 1, 1), dtype=torch.float32),
            "action_returns": torch.tensor([[0.0, float("nan")], [0.0, 0.01]], dtype=torch.float32),
            "action_valid_mask": torch.tensor([[True, False], [True, True]]),
        }

        split = module._build_split(name="train", payload=payload)

        self.assertFalse(split.dataset_reportable)
        self.assertEqual(split.dataset_reportability_errors, ["legacy_action_valid_mask_semantics_ambiguous"])

    def test_minute_to_hour_explicit_non_reportable_dataset_remains_diagnostic(self) -> None:
        module = __import__(
            "rl_quant.minute_to_hour_transformer",
            fromlist=["_build_split", "minute_to_hour_missing_label_report"],
        )
        payload = {
            "decision_timestamps": ["2026-01-02T14:30:00+00:00"],
            "next_timestamps": ["2026-01-02T15:30:00+00:00"],
            "minute_feature_names": ["m"],
            "hour_feature_names": ["h"],
            "action_names": ["CASH", "QQQ"],
            "minute_features": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "minute_mask": torch.ones((1, 1, 1), dtype=torch.bool),
            "hour_features": torch.zeros((1, 1, 1), dtype=torch.float32),
            "action_returns": torch.tensor([[0.0, 0.01]], dtype=torch.float32),
            "decision_action_valid_mask": torch.tensor([[True, True]]),
            "label_valid_mask": torch.tensor([[True, True]]),
            "dataset_reportable": False,
        }

        split = module._build_split(name="train", payload=payload)
        report = module.minute_to_hour_missing_label_report(split)

        self.assertFalse(split.dataset_reportable)
        self.assertFalse(report["evaluation_reportable"])
        self.assertEqual(report["reportability_errors"], ["dataset_marked_non_reportable"])

    def test_minute_to_hour_protocol_splits_decision_and_label_masks(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_build_split"])

        class PickActionPolicy(nn.Module):
            def forward(
                self,
                minute_features: torch.Tensor,
                minute_mask: torch.Tensor,
                hour_features: torch.Tensor,
                previous_actions: torch.Tensor,
                constraint_features: torch.Tensor,
            ) -> torch.Tensor:
                del minute_mask, hour_features, previous_actions, constraint_features
                return torch.tensor([[0.0, 10.0]], dtype=torch.float32).repeat(minute_features.shape[0], 1)

        payload = {
            "decision_timestamps": [
                "2026-01-02T14:30:00+00:00",
                "2026-01-02T15:30:00+00:00",
            ],
            "next_timestamps": [
                "2026-01-02T15:30:00+00:00",
                "2026-01-02T16:30:00+00:00",
            ],
            "minute_feature_names": ["m"],
            "hour_feature_names": ["h"],
            "action_names": ["CASH", "QQQ"],
            "minute_features": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            "minute_mask": torch.ones((2, 1, 1), dtype=torch.bool),
            "hour_features": torch.zeros((2, 1, 1), dtype=torch.float32),
            "action_returns": torch.tensor([[0.0, float("nan")], [0.0, 0.01]], dtype=torch.float32),
            "decision_action_valid_mask": torch.tensor([[True, True], [True, True]]),
            "action_valid_mask": torch.tensor([[True, True], [True, True]]),
            "label_valid_mask": torch.tensor([[True, False], [True, True]]),
        }

        split = module._build_split(name="train", payload=payload)
        self.assertTrue(bool(split.valid_actions(torch.tensor([0]))[0, 1].item()))
        self.assertFalse(bool(split.label_valid_actions(torch.tensor([0]))[0, 1].item()))
        self.assertEqual(split.valid_start_indices.tolist(), [1])

        result = evaluate_minute_to_hour_policy(
            split,
            PickActionPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=TradingConstraintConfig(one_way_cost_bps=0.0),
            capture_rollout=True,
        )

        self.assertEqual([row["asset"] for row in result.rollout_records], ["QQQ"])
        self.assertTrue(result.evaluation_reportable)
        self.assertEqual(result.selectable_missing_label_count, 0)
        self.assertEqual(result.requested_action_missing_label_count, 0)
        self.assertEqual(result.executed_action_missing_label_count, 0)
        self.assertEqual(result.policy_unscorable_rows, 0)
        self.assertEqual(result.reportability_errors, [])
        self.assertEqual([row["requested_asset"] for row in result.rollout_records], ["QQQ"])
        self.assertEqual([row["executed_asset"] for row in result.rollout_records], ["QQQ"])

    def test_minute_to_hour_eval_reports_mask_block_reasons(self) -> None:
        # The evaluator now tallies WHY each decision row's mask pinned the policy (a diagnostic that explains
        # turnover). The mask itself is unchanged -- build_action_mask_reasons(...).mask IS build_action_mask's
        # output -- so this is additive. A high min-hold must show up in the tally (teeth); min_hold_bars=0 must
        # not (no false positives); counts are bounded by the number of decision rows.
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit,
            TradingConstraintConfig,
            evaluate_minute_to_hour_policy,
        )
        from rl_quant.training.minute_to_hour import _ConstantActionModel

        n = 5
        split = HourFromMinuteDataSplit(
            name="t",
            decision_timestamps=[f"2026-01-02T{10 + i}:30:00+00:00" for i in range(n + 1)],
            next_timestamps=[f"2026-01-02T{11 + i}:30:00+00:00" for i in range(n + 1)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((n + 1, 1, 1, 1)), minute_mask=torch.ones((n + 1, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((n + 1, 1, 1)), action_returns=torch.zeros((n + 1, 2)),
            action_valid_mask=torch.ones((n + 1, 2), dtype=torch.bool),
            label_valid_mask=torch.ones((n + 1, 2), dtype=torch.bool),
            valid_start_indices=torch.arange(n, dtype=torch.long),
            valid_index_mask=torch.tensor([True] * n + [False]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        always_qqq = _ConstantActionModel(2, 1)

        pinned = evaluate_minute_to_hour_policy(
            split, always_qqq, device=torch.device("cpu"), initial_action=0,
            constraints=TradingConstraintConfig(one_way_cost_bps=0.0, min_hold_bars=10),
        )
        counts = pinned.mask_block_reason_row_counts
        self.assertEqual(set(counts), {"decision_rows", "min_hold", "cooldown", "switch_cap", "order_leg"})
        self.assertEqual(counts["decision_rows"], n)
        for key in ("min_hold", "cooldown", "switch_cap", "order_leg"):
            self.assertGreaterEqual(counts[key], 0)
            self.assertLessEqual(counts[key], n)
        self.assertGreaterEqual(counts["min_hold"], 1)  # teeth: a high min-hold pins rows
        self.assertEqual(pinned.to_dict()["mask_block_reason_row_counts"], counts)  # surfaced verbatim

        # No false positives: with no minimum hold, nothing is min-hold-pinned.
        free = evaluate_minute_to_hour_policy(
            split, always_qqq, device=torch.device("cpu"), initial_action=0,
            constraints=TradingConstraintConfig(one_way_cost_bps=0.0, min_hold_bars=0),
        )
        self.assertEqual(free.mask_block_reason_row_counts["min_hold"], 0)
        self.assertEqual(free.mask_block_reason_row_counts["decision_rows"], n)

    def test_latest_holdout_uses_final_complete_sessions_and_train_normalizer(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["build_hour_from_minute_splits"])
        sessions = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        decisions = [
            f"{session}T{hour}:30:00+00:00"
            for session in sessions
            for hour in ("14", "15")
        ]
        next_timestamps = [
            f"{session}T{hour}:30:00+00:00"
            for session in sessions
            for hour in ("15", "16")
        ]
        session_ids = [session for session in sessions for _ in range(2)]
        minute_values = [2.0] * 6 + [50.0] * 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            self._write_minute_to_hour_dataset(
                path,
                decisions=decisions,
                next_timestamps=next_timestamps,
                session_ids=session_ids,
                minute_values=minute_values,
            )

            train, val, test = module.build_hour_from_minute_splits(
                dataset_path=path,
                split_mode="latest_holdout",
                val_sessions=1,
                test_sessions=1,
                embargo_sessions=0,
                min_train_sessions=2,
            )

        self.assertEqual(train.split_policy["split_mode"], "latest_holdout")
        self.assertTrue(train.split_policy["test_uses_latest_complete_period"])
        self.assertTrue(train.dataset_reportable)
        self.assertEqual(train.split_policy["test_start"], decisions[-2])
        self.assertEqual([test.decision_timestamps[i] for i in test.valid_start_indices.tolist()], decisions[-2:])
        self.assertEqual([val.decision_timestamps[i] for i in val.valid_start_indices.tolist()], decisions[-4:-2])
        self.assertAlmostEqual(float(train.minute_feature_mean[0].item()), 2.0, places=6)
        self.assertTrue(torch.equal(val.minute_feature_mean, train.minute_feature_mean))
        self.assertTrue(torch.equal(test.minute_feature_mean, train.minute_feature_mean))

    def test_reward_horizon_cannot_cross_split_end(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_build_split"])
        payload = {
            "decision_timestamps": [
                "2026-01-02T14:30:00+00:00",
                "2026-01-02T15:30:00+00:00",
            ],
            "next_timestamps": [
                "2026-01-02T15:30:00+00:00",
                "2026-01-02T16:30:00+00:00",
            ],
            "minute_feature_names": ["m"],
            "hour_feature_names": ["h"],
            "action_names": ["CASH", "QQQ"],
            "minute_features": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            "minute_mask": torch.ones((2, 1, 1), dtype=torch.bool),
            "hour_features": torch.zeros((2, 1, 1), dtype=torch.float32),
            "action_returns": torch.zeros((2, 2), dtype=torch.float32),
            "decision_action_valid_mask": torch.ones((2, 2), dtype=torch.bool),
            "label_valid_mask": torch.ones((2, 2), dtype=torch.bool),
        }

        split = module._build_split(
            name="train",
            payload=payload,
            end_ts="2026-01-02T15:30:00+00:00",
            reward_end_ts="2026-01-02T15:30:00+00:00",
        )

        self.assertEqual(split.decision_timestamps, payload["decision_timestamps"])
        self.assertEqual(split.valid_start_indices.tolist(), [0])

    def test_manual_split_that_skips_latest_period_is_non_reportable(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["build_hour_from_minute_splits"])
        decisions = [f"2026-01-02T{hour}:30:00+00:00" for hour in ("14", "15", "16", "17", "18", "19")]
        next_timestamps = [f"2026-01-02T{hour}:30:00+00:00" for hour in ("15", "16", "17", "18", "19", "20")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            self._write_minute_to_hour_dataset(path, decisions=decisions, next_timestamps=next_timestamps)

            _train, _val, test = module.build_hour_from_minute_splits(
                dataset_path=path,
                split_mode="manual",
                train_end=decisions[1],
                val_end=next_timestamps[2],
                test_start=decisions[3],
                test_end=next_timestamps[3],
            )

        self.assertFalse(test.dataset_reportable)
        self.assertFalse(test.split_policy["test_uses_latest_complete_period"])
        self.assertIn("manual_split_skips_latest_complete_period", test.dataset_reportability_errors)

    def test_manual_split_requires_latest_reward_end_not_just_latest_decision(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["build_hour_from_minute_splits"])
        decisions = [f"2026-01-02T{hour}:30:00+00:00" for hour in ("14", "15", "16", "17", "18", "19")]
        next_timestamps = [f"2026-01-02T{hour}:30:00+00:00" for hour in ("15", "16", "17", "18", "19", "20")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            self._write_minute_to_hour_dataset(path, decisions=decisions, next_timestamps=next_timestamps)

            _train, _val, test = module.build_hour_from_minute_splits(
                dataset_path=path,
                split_mode="manual",
                train_end=decisions[1],
                val_end=next_timestamps[2],
                test_start=decisions[3],
                test_end=decisions[-1],
            )

        self.assertFalse(test.dataset_reportable)
        self.assertEqual(test.split_policy["max_dataset_decision_timestamp"], decisions[-1])
        self.assertEqual(test.split_policy["max_dataset_reward_end_timestamp"], next_timestamps[-1])
        self.assertFalse(test.split_policy["test_uses_latest_complete_period"])
        self.assertIn("manual_split_skips_latest_complete_period", test.dataset_reportability_errors)

    def test_latest_rows_smoke_split_is_non_reportable(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["build_hour_from_minute_splits"])
        decisions = [f"2026-01-02T{hour}:30:00+00:00" for hour in ("14", "15", "16", "17", "18")]
        next_timestamps = [f"2026-01-02T{hour}:30:00+00:00" for hour in ("15", "16", "17", "18", "19")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            self._write_minute_to_hour_dataset(path, decisions=decisions, next_timestamps=next_timestamps)

            train, val, test = module.build_hour_from_minute_splits(
                dataset_path=path,
                split_mode="latest_rows_smoke",
                val_rows=1,
                test_rows=1,
                min_train_rows=2,
            )

        self.assertEqual(train.split_policy["split_mode"], "latest_rows_smoke")
        self.assertFalse(test.dataset_reportable)
        self.assertEqual([val.decision_timestamps[i] for i in val.valid_start_indices.tolist()], [decisions[-2]])
        self.assertEqual([test.decision_timestamps[i] for i in test.valid_start_indices.tolist()], [decisions[-1]])
        self.assertIn("smoke_row_based_split", test.dataset_reportability_errors)

    def test_hour_from_second_partition_training_uses_distinct_validation_and_test_rows(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        decisions = [
            "2026-01-02T14:30:00+00:00",
            "2026-01-02T15:30:00+00:00",
            "2026-01-02T16:30:00+00:00",
            "2026-01-02T17:30:00+00:00",
            "2026-01-02T18:30:00+00:00",
        ]
        next_timestamps = [
            "2026-01-02T15:30:00+00:00",
            "2026-01-02T16:30:00+00:00",
            "2026-01-02T17:30:00+00:00",
            "2026-01-02T18:30:00+00:00",
            "2026-01-02T19:30:00+00:00",
        ]
        minute_grid = [
            [[timestamp.replace(":30:00", ":29:00")]]
            for timestamp in decisions
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            torch.save(
                {
                    "decision_timestamps": decisions,
                    "next_timestamps": next_timestamps,
                    "minute_timestamp_grid": minute_grid,
                    "minute_feature_names": ["m"],
                    "hour_feature_names": ["h"],
                    "action_names": ["CASH", "QQQ"],
                    "minute_features": torch.zeros((5, 1, 1, 1), dtype=torch.float32),
                    "minute_mask": torch.ones((5, 1, 1), dtype=torch.bool),
                    "hour_features": torch.zeros((5, 1, 1), dtype=torch.float32),
                    "action_returns": torch.zeros((5, 2), dtype=torch.float32),
                    "decision_action_valid_mask": torch.ones((5, 2), dtype=torch.bool),
                    "label_valid_mask": torch.ones((5, 2), dtype=torch.bool),
                },
                path,
            )

            train, val, test = module.build_rolling_partition_splits(path, split_mode="latest_rows_smoke")

        self.assertEqual([train.decision_timestamps[i] for i in train.valid_start_indices.tolist()], decisions[:3])
        self.assertEqual([val.decision_timestamps[i] for i in val.valid_start_indices.tolist()], [decisions[-2]])
        self.assertEqual([test.decision_timestamps[i] for i in test.valid_start_indices.tolist()], [decisions[-1]])
        self.assertEqual(train.decision_timestamps, decisions[:3])
        self.assertEqual(val.decision_timestamps, [decisions[-2]])
        self.assertEqual(test.decision_timestamps, [decisions[-1]])
        self.assertNotEqual(module.split_window(val)["first_valid_decision"], module.split_window(test)["first_valid_decision"])

    def test_hour_from_second_partition_training_defaults_to_latest_holdout_sessions(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]
        decisions = [
            f"{session}T{hour}:30:00+00:00"
            for session in sessions
            for hour in ("14", "15")
        ]
        next_timestamps = [
            f"{session}T{hour}:30:00+00:00"
            for session in sessions
            for hour in ("15", "16")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            self._write_minute_to_hour_dataset(
                path,
                decisions=decisions,
                next_timestamps=next_timestamps,
                session_ids=[session for session in sessions for _ in range(2)],
            )

            train, val, test = module.build_rolling_partition_splits(path)

        self.assertEqual(train.split_policy["split_mode"], "latest_holdout")
        self.assertTrue(test.dataset_reportable)
        self.assertEqual([train.decision_timestamps[i] for i in train.valid_start_indices.tolist()], decisions[:2])
        self.assertEqual([val.decision_timestamps[i] for i in val.valid_start_indices.tolist()], decisions[2:4])
        self.assertEqual([test.decision_timestamps[i] for i in test.valid_start_indices.tolist()], decisions[4:])

    def test_hour_from_second_partition_training_short_chunks_fallback_is_non_reportable(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        decisions = [f"2026-01-02T{hour}:30:00+00:00" for hour in ("14", "15", "16", "17", "18")]
        next_timestamps = [f"2026-01-02T{hour}:30:00+00:00" for hour in ("15", "16", "17", "18", "19")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pt"
            self._write_minute_to_hour_dataset(path, decisions=decisions, next_timestamps=next_timestamps)

            train, _val, test = module.build_rolling_partition_splits(path)

        self.assertEqual(train.split_policy["split_mode"], "latest_rows_smoke_fallback")
        self.assertFalse(test.dataset_reportable)
        self.assertIn("latest_holdout_insufficient_complete_sessions", test.dataset_reportability_errors)
        self.assertIn("fallback_to_latest_rows_smoke_split", test.dataset_reportability_errors)

    def test_hour_from_second_max_partitions_defaults_to_latest(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ["2025-01-01_to_2025-01-04", "2026-01-01_to_2026-01-04", "2026-06-01_to_2026-06-04"]:
                part = root / label
                part.mkdir()
                (part / "hour_from_second_dataset.pt").write_text("placeholder")
            args = module.parse_args(
                [
                    "--partitions-root",
                    str(root),
                    "--max-partitions",
                    "2",
                ]
            )

            selected = [path.parent.name for path in module.partition_paths(args)]
            errors = module.partition_selection_reportability_errors(args)
            allow_truncated = module.parse_args(
                [
                    "--partitions-root",
                    str(root),
                    "--max-partitions",
                    "2",
                    "--allow-truncated-training-history",
                ]
            )
            allow_truncated_errors = module.partition_selection_reportability_errors(allow_truncated)

        self.assertEqual(selected, ["2026-01-01_to_2026-01-04", "2026-06-01_to_2026-06-04"])
        self.assertTrue(any("silently excluded" in error for error in errors))
        self.assertEqual(allow_truncated_errors, [])

    def test_integrate_covariates_max_partitions_defaults_to_latest(self) -> None:
        module = load_script("integrate_stock_covariates_with_hour_partitions")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ["2025-01-01_to_2025-01-04", "2026-01-01_to_2026-01-04", "2026-06-01_to_2026-06-04"]:
                part = root / label
                part.mkdir()
                (part / "hour_from_second_dataset.pt").write_text("placeholder")
            args = module.parse_args(
                [
                    "--partitions-root",
                    str(root),
                    "--max-partitions",
                    "2",
                ]
            )

            selected = [path.parent.name for path in module.partition_paths(args)]
            errors = module.partition_selection_reportability_errors(args)
            allow_truncated = module.parse_args(
                [
                    "--partitions-root",
                    str(root),
                    "--max-partitions",
                    "2",
                    "--allow-truncated-training-history",
                ]
            )
            allow_truncated_errors = module.partition_selection_reportability_errors(allow_truncated)

        self.assertEqual(selected, ["2026-01-01_to_2026-01-04", "2026-06-01_to_2026-06-04"])
        self.assertTrue(any("silently excluded" in error for error in errors))
        self.assertEqual(allow_truncated_errors, [])

    def test_hour_from_second_earliest_partition_selection_is_diagnostic(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ["2025-01-01_to_2025-01-04", "2026-01-01_to_2026-01-04", "2026-06-01_to_2026-06-04"]:
                part = root / label
                part.mkdir()
                (part / "hour_from_second_dataset.pt").write_text("placeholder")
            args = module.parse_args(
                [
                    "--partitions-root",
                    str(root),
                    "--max-partitions",
                    "1",
                    "--partition-selection",
                    "earliest",
                ]
            )

            selected = [path.parent.name for path in module.partition_paths(args)]
            policy = module.split_policy_with_partition_selection({"reportable": True, "reportability_errors": []}, args)
            reportable, errors = module.combined_evaluation_reportability(
                evaluator_reportable=True,
                evaluator_errors=[],
                split_policy={"reportable": False, "reportability_errors": ["smoke_row_based_split"]},
                args=args,
            )
            latest_args = module.parse_args(["--partitions-root", str(root)])
            evaluator_error_reportable, evaluator_errors = module.combined_evaluation_reportability(
                evaluator_reportable=True,
                evaluator_errors=["requested_actions_with_missing_reward_labels"],
                split_policy={"reportable": True, "reportability_errors": []},
                args=latest_args,
            )

        self.assertEqual(selected, ["2025-01-01_to_2025-01-04"])
        self.assertFalse(policy["reportable"])
        self.assertTrue(
            any("not the latest available" in error for error in policy["partition_selection_reportability_errors"])
        )
        self.assertTrue(any("not the latest available" in error for error in policy["reportability_errors"]))
        self.assertFalse(reportable)
        self.assertIn("smoke_row_based_split", errors)
        self.assertTrue(any("not the latest available" in error for error in errors))
        self.assertFalse(evaluator_error_reportable)
        self.assertEqual(evaluator_errors, ["requested_actions_with_missing_reward_labels"])
